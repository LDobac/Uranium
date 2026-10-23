//! Burn inference server of the Rust RL learner (042), until the learner moves to PyTorch
//! (AMB-000-4 step 1-8; deleted with Burn in 1-9). It serves the framework-free session protocol
//! of `uranium_agent::infer` (the PyTorch server is `uranium_agent::broker` + the Python model
//! thread): a dedicated thread owns the model, each session has a fixed slot of one
//! `[capacity, …]` Core state, rows are gathered with `select_rows` and written back with
//! `scatter_rows`.

use std::path::Path;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow, bail};
use burn::tensor::{Device, Int, Tensor, TensorData};
use serde::{Deserialize, Serialize};
use uranium_agent::infer::{ActReply, HEADS, InferOptions, InferenceHandle, Req, SampleMode, checkpoint_id, host_action};
use uranium_env::actions::ACTION_SCHEMA_VERSION;
use uranium_env::features::extract::action_table;
use uranium_env::features::{FEATURE_SCHEMA_VERSION, Observation};
use uranium_model::batch::ActionColumns;
use uranium_model::checkpoint;
use uranium_model::collate::obs_batch_padded;
use uranium_model::config::CoreKind;
use uranium_model::{CoreInit, GtrxlCore, LstmCore, UraniumNet};

/// Batches slower than this are logged (normal p95 is well under a second).
const SLOW_BATCH: Duration = Duration::from_secs(5);

/// Inference device (060-R1). CUDA only exists in builds with `--features cuda`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum InferBackend {
    Cuda,
    Vulkan,
    Flex,
}

impl InferBackend {
    pub fn device(self) -> Result<Device> {
        // 041-R15: same autotune cache location as training when URANIUM_AUTOTUNE_CACHE_DIR is set.
        if self != Self::Flex {
            uranium_model::runtime::init_gpu_runtime(false)?;
        }
        Ok(match self {
            Self::Vulkan => Device::vulkan(burn::tensor::DeviceKind::DefaultDevice),
            #[cfg(feature = "cuda")]
            Self::Cuda => Device::cuda(0),
            #[cfg(not(feature = "cuda"))]
            Self::Cuda => bail!("CUDA inference needs a build with --features cuda (CUDA host only)"),
            Self::Flex => Device::flex(),
        })
    }
}

/// Loads a Burn checkpoint on its own thread; returns once the model is ready (or failed).
pub fn start(dir: &Path, sample: SampleMode, opts: InferOptions, backend: InferBackend) -> Result<InferenceHandle> {
    let id = checkpoint_id(dir)?;
    let kind = checkpoint::read_meta(dir)?.net_config()?.core.kind;
    let (tx, rx) = mpsc::channel();
    let (ready_tx, ready_rx) = mpsc::channel();
    let dir = dir.to_path_buf();
    std::thread::Builder::new().name(format!("infer-{id}")).spawn(move || {
        let res = match kind {
            CoreKind::Lstm => serve::<LstmCore>(&dir, sample, opts, backend, rx, ready_tx),
            CoreKind::Gtrxl => serve::<GtrxlCore>(&dir, sample, opts, backend, rx, ready_tx),
        };
        if let Err(e) = res {
            crate::elog!("[infer] server for {} stopped: {e:#}", dir.display());
        }
    })?;
    ready_rx.recv().map_err(|_| anyhow!("inference thread exited while loading"))??;
    crate::elog!("[infer] {id} ready ({backend:?}, {sample:?}, {kind:?}, max_batch {}, capacity {})", opts.max_batch, opts.capacity);
    Ok(InferenceHandle::new(tx, id, sample, &opts))
}

struct Pending {
    slot: usize,
    obs: Box<Observation>,
    reply: Sender<Result<ActReply>>,
    since: Instant,
}

fn serve<C: CoreInit + 'static>(dir: &Path, sample: SampleMode, opts: InferOptions, backend: InferBackend, rx: Receiver<Req>, ready: Sender<Result<()>>) -> Result<()> {
    let loaded = (|| -> Result<_> {
        let device = backend.device()?;
        let cols = ActionColumns::from_table(action_table())?;
        let (net, _) = checkpoint::load::<C>(dir, &cols, FEATURE_SCHEMA_VERSION, ACTION_SCHEMA_VERSION, &device)?;
        Ok((device, net))
    })();
    let (device, mut net) = match loaded {
        Ok(x) => {
            let _ = ready.send(Ok(()));
            x
        }
        Err(e) => {
            let msg = format!("{e:#}");
            let _ = ready.send(Err(e));
            bail!(msg);
        }
    };
    let cols = uranium_data::table::ActionColumns::current()?;
    let mut state = net.init_state(opts.capacity, &device);
    let mut used = vec![false; opts.capacity];
    let mut pending: Vec<Pending> = Vec::new();
    let mut version = 0u64;
    loop {
        // Wait for work: block when idle, otherwise until the oldest request's deadline.
        let req = if pending.is_empty() {
            match rx.recv() {
                Ok(r) => Some(r),
                Err(_) => return Ok(()),
            }
        } else {
            let waited = pending[0].since.elapsed();
            match rx.recv_timeout(opts.max_wait.saturating_sub(waited)) {
                Ok(r) => Some(r),
                Err(RecvTimeoutError::Timeout) => None,
                Err(RecvTimeoutError::Disconnected) => return Ok(()),
            }
        };
        match req {
            Some(Req::Open(reply)) => {
                let res = match used.iter().position(|u| !u) {
                    Some(slot) => {
                        used[slot] = true;
                        let rows = Tensor::<1, Int>::from_data(TensorData::new(vec![slot as i32], [1]), &device);
                        state = C::scatter_rows(state, rows, net.init_state(1, &device));
                        Ok(slot)
                    }
                    None => Err(anyhow!("all {} inference slots in use", opts.capacity)),
                };
                let _ = reply.send(res);
            }
            Some(Req::Close(slot)) => {
                used[slot] = false;
                pending.retain(|p| p.slot != slot);
            }
            Some(Req::Act { slot, obs, reply }) => pending.push(Pending { slot, obs, reply, since: Instant::now() }),
            Some(Req::Swap { net: new, version: v, reply }) => {
                let res = match new.downcast::<UraniumNet<C>>() {
                    Ok(n) => {
                        net = *n;
                        version = v;
                        Ok(())
                    }
                    Err(_) => Err(anyhow!("swap: weights are not a UraniumNet of this server's core")),
                };
                let _ = reply.send(res);
            }
            Some(Req::State { slot, reply }) => {
                let rows = Tensor::<1, Int>::from_data(TensorData::new(vec![slot as i32], [1]), &device);
                let _ = reply.send(Ok(C::state_to_host(&C::select_rows(&state, rows))));
            }
            None => {}
        }
        let active = used.iter().filter(|u| **u).count();
        let due = !pending.is_empty() && (pending.len() >= opts.max_batch || pending.len() >= active || pending[0].since.elapsed() >= opts.max_wait);
        if due {
            let n = pending.len().min(opts.max_batch);
            let batch: Vec<Pending> = pending.drain(..n).collect();
            state = run_batch(&net, &cols, &device, state, batch, sample.temp(), opts.pad_entities, version);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_batch<C: CoreInit>(net: &UraniumNet<C>, cols: &uranium_data::table::ActionColumns, device: &Device, state: C::State, batch: Vec<Pending>, temp: f32, pad: usize, version: u64) -> C::State {
    let t0 = Instant::now();
    let b = batch.len();
    let rows = Tensor::<1, Int>::from_data(TensorData::new(batch.iter().map(|p| p.slot as i32).collect::<Vec<_>>(), [b]), device);
    let refs: Vec<&Observation> = batch.iter().map(|p| &*p.obs).collect();
    let ob = obs_batch_padded(&refs, pad, device);
    let n = ob.entities();
    let out = net.act(&ob, C::select_rows(&state, rows.clone()), temp);
    let state = C::scatter_rows(state, rows, out.state);
    let ints = |t: Tensor<1, Int>| t.into_data().convert::<i64>().try_into_vec::<i64>().unwrap_or_default();
    let action = ints(out.action.action);
    let delay = ints(out.action.delay);
    let queued = ints(out.action.queued);
    let s = out.action.selected.dims()[1];
    let selected = out.action.selected.into_data().convert::<i64>().try_into_vec::<i64>().unwrap_or_default();
    let target_unit = ints(out.action.target_unit);
    let target_loc = ints(out.action.target_loc);
    let floats = |d: TensorData| d.convert::<f32>().try_into_vec::<f32>().unwrap_or_default();
    let log_prob = floats(out.log_prob.into_data());
    let heads = floats(out.head_log_prob.into_data());
    let value = floats(out.value.into_data());
    let win_open = out.win_open.map(|t| floats(t.into_data()));
    let run = t0.elapsed();
    if run >= SLOW_BATCH {
        let unix = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
        crate::elog!("[infer] slow batch: {b} requests took {:.1}s (unix {unix})", run.as_secs_f64());
    }
    for (r, p) in batch.into_iter().enumerate() {
        let reply = ActReply {
            action: host_action(cols, action[r], delay[r], queued[r], &selected[r * s..(r + 1) * s], target_unit[r], target_loc[r], n),
            log_prob: log_prob[r],
            head_log_prob: std::array::from_fn(|h| heads[r * HEADS + h]),
            value: std::array::from_fn(|v| value[r * 4 + v]),
            win_open: win_open.as_ref().map(|w| w[r]),
            obs: *p.obs,
            batch: b,
            run,
            version,
        };
        let _ = p.reply.send(Ok(reply));
    }
    state
}

/// Writes a randomly initialised Burn checkpoint for `cfg` (Burn RL tests).
pub fn write_random_checkpoint(dir: &Path, cfg: &uranium_model::UraniumNetConfig, device: &Device) -> Result<()> {
    fn save<C: CoreInit>(dir: &Path, cfg: &uranium_model::UraniumNetConfig, device: &Device) -> Result<()> {
        let cols = ActionColumns::from_table(action_table())?;
        let net = UraniumNet::<C>::new(cfg, &cols, device);
        let mut meta = checkpoint::CheckpointMeta::new(cfg, FEATURE_SCHEMA_VERSION, ACTION_SCHEMA_VERSION)?;
        meta.train = serde_json::json!({ "random_init": true });
        checkpoint::save(dir, &net, &meta)
    }
    match cfg.core.kind {
        CoreKind::Lstm => save::<LstmCore>(dir, cfg, device),
        CoreKind::Gtrxl => save::<GtrxlCore>(dir, cfg, device),
    }
}
