//! Batched inference server (032-R7, 032-R14, 001-R6). One server per (checkpoint, sampling
//! mode): a dedicated thread owns the model on the GPU, sessions send observations over a channel
//! and the thread runs one `act` over every waiting session. Each session has a fixed slot whose
//! Core state lives in one `[capacity, …]` state; the batch rows are gathered with `select_rows`
//! and written back with `scatter_rows`. Lockstep games wait for their reply, so batching changes
//! throughput only, never results.

use std::any::Any;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::time::{Duration, Instant};

/// Batches slower than this are logged (normal p95 is well under a second).
const SLOW_BATCH: Duration = Duration::from_secs(5);

use anyhow::{Context, Result, anyhow, bail};
use burn::tensor::{Device, Int, Tensor, TensorData};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uranium_env::actions::ACTION_SCHEMA_VERSION;
use uranium_env::features::executor::HostAction;
use uranium_env::features::extract::action_table;
use uranium_env::features::{FEATURE_SCHEMA_VERSION, Observation};
pub use uranium_env::surrender::{DEFAULT_CHAT, SurrenderMode};
use uranium_model::batch::{ActionColumns, TARGET_LOC, TARGET_UNIT};
use uranium_model::checkpoint::{self, WEIGHTS_FILE};
use uranium_model::collate::obs_batch_padded;
use uranium_model::config::CoreKind;
use uranium_model::heads::HEADS;
use uranium_model::{CoreInit, GtrxlCore, HostState, LstmCore, UraniumNet};

/// Inference device (060-R1). CUDA only exists in builds with `--features cuda`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum InferBackend {
    Cuda,
    Vulkan,
    Flex,
}

impl InferBackend {
    /// `URANIUM_AGENT_BACKEND` (001-R9), default `vulkan` (020-R11).
    pub fn from_env() -> Result<Self> {
        match std::env::var("URANIUM_AGENT_BACKEND").ok().filter(|v| !v.trim().is_empty()).as_deref().map(str::trim) {
            None | Some("vulkan") => Ok(Self::Vulkan),
            Some("cuda") => Ok(Self::Cuda),
            Some("flex") => Ok(Self::Flex),
            Some(other) => bail!("URANIUM_AGENT_BACKEND={other}: expected cuda, vulkan or flex"),
        }
    }

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

/// 050-R1 / 032-R15 sampling mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum SampleMode {
    /// temp 0: argmax of every head (deterministic).
    #[default]
    Argmax,
    /// temp 1: Gumbel-max sampling.
    Stochastic,
}

impl SampleMode {
    pub fn temp(self) -> f32 {
        match self {
            Self::Argmax => 0.0,
            Self::Stochastic => 1.0,
        }
    }
}

/// Batching parameters. Defaults are placeholders until measured (020-R11, TBD(M6)).
#[derive(Clone, Copy, Debug)]
pub struct InferOptions {
    pub backend: InferBackend,
    pub max_batch: usize,
    pub max_wait: Duration,
    /// Session slots (concurrent games x model players).
    pub capacity: usize,
    /// Entity count is rounded up to a multiple of this (fewer distinct shapes, 030-R7).
    pub pad_entities: usize,
    /// 030-R10 / E-5: feed `last_failed = 0` to the model in every session of this server.
    /// Why: replay playback has no command-failure events, so in BC training data the
    /// `last_failed` scalar is always 0. A BC checkpoint has never seen 1 there; in an evaluation
    /// game a 1 would be an untrained input and only add noise to the evaluation numbers. So
    /// evaluation games of BC checkpoints (core ablation 032-R17, 041-AC2 bc-vs-t0) set this
    /// (plan `zero_last_failed = true`). RL (042) leaves it off and uses the real value.
    pub zero_last_failed: bool,
    /// 012-R19 (SC:R realtime): before serving, run one batch-1 forward per padded entity count up
    /// to this many entities, so kernel autotune does not stall a live game (the first forward on
    /// CUDA took 57 s in the first SC:R game, 2026-10-22). 0 = no warm-up.
    pub warmup_entities: usize,
}

impl InferOptions {
    pub fn new(backend: InferBackend) -> Self {
        Self { backend, max_batch: 32, max_wait: Duration::from_millis(5), capacity: 128, pad_entities: 64, zero_last_failed: false, warmup_entities: 0 }
    }
}

/// One decision for one session.
#[derive(Debug)]
pub struct ActReply {
    pub action: HostAction,
    /// Joint behaviour log-prob of the used heads (032-R10, 042).
    pub log_prob: f32,
    /// Per-head log-prob, unused heads 0 (032-R15).
    pub head_log_prob: [f32; HEADS],
    /// win, build, units, value (032 §2.6, no privileged input).
    pub value: [f32; 4],
    /// 032-R18 win estimate in [-1, 1]; `None` when the checkpoint has no win_open head.
    pub win_open: Option<f32>,
    /// The observation, handed back so the caller can run the executor without a copy.
    pub obs: Observation,
    /// Rows in the batch this decision ran in.
    pub batch: usize,
    /// Model forward + host copy of that batch.
    pub run: Duration,
    /// Weights version that produced this decision (032-R16, 042-R4): 0 = the loaded checkpoint.
    pub version: u64,
}

enum Req {
    Open(Sender<Result<usize>>),
    Act { slot: usize, obs: Box<Observation>, reply: Sender<Result<ActReply>> },
    Close(usize),
    /// 032-R16: replace the weights from the next batch on (`UraniumNet<C>` of the server's core).
    Swap { net: Box<dyn Any + Send>, version: u64, reply: Sender<Result<()>> },
    /// 032-R16: the slot's Core state on the host (042 core_state0).
    State { slot: usize, reply: Sender<Result<HostState>> },
}

/// Cheap to clone; every clone talks to the same server thread.
#[derive(Clone)]
pub struct InferenceHandle {
    tx: Sender<Req>,
    /// 020-R5 policy version: checkpoint folder name + first 12 hex of `model.bpk` SHA-256.
    pub checkpoint_id: Arc<str>,
    pub sample: SampleMode,
    /// `InferOptions::zero_last_failed` of the server (030-R10, E-5).
    pub zero_last_failed: bool,
    /// 031-R12 / 020-R14: surrender mode of the sessions (default `Shadow`: check, never leave).
    pub surrender: SurrenderMode,
    /// 020-R14: chat line sent before leaving (default `gg`, empty = no chat).
    pub surrender_chat: Arc<str>,
}

impl std::fmt::Debug for InferenceHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "InferenceHandle({}, {:?})", self.checkpoint_id, self.sample)
    }
}

/// 020-R11: a relative checkpoint path is relative to `URANIUM_SHARED_DIR` (061).
pub fn resolve_checkpoint(p: &Path) -> Result<PathBuf> {
    if p.is_absolute() {
        return Ok(p.to_path_buf());
    }
    let shared = std::env::var_os("URANIUM_SHARED_DIR").filter(|v| !v.is_empty()).context("relative checkpoint path needs URANIUM_SHARED_DIR (020-R11)")?;
    Ok(PathBuf::from(shared).join(p))
}

pub fn checkpoint_id(dir: &Path) -> Result<String> {
    let bytes = std::fs::read(dir.join(WEIGHTS_FILE)).with_context(|| format!("reading {}", dir.join(WEIGHTS_FILE).display()))?;
    let sha: String = Sha256::digest(&bytes).iter().take(6).map(|b| format!("{b:02x}")).collect();
    let name = dir.file_name().map_or_else(|| "checkpoint".into(), |n| n.to_string_lossy().into_owned());
    Ok(format!("{name}@{sha}"))
}

/// The checkpoint's Core kind, lower case (`lstm` / `gtrxl`), for reports (020-R5 summary).
pub fn checkpoint_core(dir: &Path) -> Result<String> {
    Ok(format!("{:?}", checkpoint::read_meta(dir)?.net_config()?.core.kind).to_lowercase())
}

/// 020-R5: the session `policy` label of a model player (same text as `PolicySpec::label`).
pub fn model_policy_label(checkpoint_id: &str, sample: SampleMode) -> String {
    format!("model:{checkpoint_id}:{sample:?}").to_lowercase()
}

impl InferenceHandle {
    /// Loads the checkpoint on its own thread; returns once the model is ready (or failed).
    pub fn start(dir: &Path, sample: SampleMode, opts: InferOptions) -> Result<Self> {
        let id = checkpoint_id(dir)?;
        let kind = checkpoint::read_meta(dir)?.net_config()?.core.kind;
        let (tx, rx) = mpsc::channel();
        let (ready_tx, ready_rx) = mpsc::channel();
        let dir = dir.to_path_buf();
        std::thread::Builder::new().name(format!("infer-{id}")).spawn(move || {
            let res = match kind {
                CoreKind::Lstm => serve::<LstmCore>(&dir, sample, opts, rx, ready_tx),
                CoreKind::Gtrxl => serve::<GtrxlCore>(&dir, sample, opts, rx, ready_tx),
            };
            if let Err(e) = res {
                elog!("[infer] server for {} stopped: {e:#}", dir.display());
            }
        })?;
        ready_rx.recv().map_err(|_| anyhow!("inference thread exited while loading"))??;
        elog!("[infer] {id} ready ({:?}, {sample:?}, {kind:?}, max_batch {}, capacity {})", opts.backend, opts.max_batch, opts.capacity);
        Ok(Self { tx, checkpoint_id: id.into(), sample, zero_last_failed: opts.zero_last_failed, surrender: SurrenderMode::Shadow, surrender_chat: DEFAULT_CHAT.into() })
    }

    /// 031-R12 / 020-R14: surrender mode and chat line for the sessions opened from this handle.
    pub fn with_surrender(mut self, mode: SurrenderMode, chat: &str) -> Self {
        self.surrender = mode;
        self.surrender_chat = chat.into();
        self
    }

    /// 032-R16: weights for the next batches. `net` must be a `UraniumNet<C>` of the checkpoint's
    /// core kind on the inference device; `version` is reported with every later decision.
    pub fn swap(&self, net: Box<dyn Any + Send>, version: u64) -> Result<()> {
        let (tx, rx) = mpsc::channel();
        self.tx.send(Req::Swap { net, version, reply: tx }).map_err(|_| anyhow!("inference server is gone"))?;
        rx.recv().map_err(|_| anyhow!("inference server is gone"))?
    }

    /// A slot for one game player; its Core state starts at zero (032-R5).
    pub fn open(&self) -> Result<InferSession> {
        let (tx, rx) = mpsc::channel();
        self.tx.send(Req::Open(tx)).map_err(|_| anyhow!("inference server is gone"))?;
        let slot = rx.recv().map_err(|_| anyhow!("inference server is gone"))??;
        Ok(InferSession { slot, tx: self.tx.clone() })
    }
}

/// A session's slot; closing it (drop) frees the slot.
pub struct InferSession {
    slot: usize,
    tx: Sender<Req>,
}

impl InferSession {
    pub fn act(&self, obs: Observation) -> Result<ActReply> {
        let (tx, rx) = mpsc::channel();
        self.tx.send(Req::Act { slot: self.slot, obs: Box::new(obs), reply: tx }).map_err(|_| anyhow!("inference server is gone"))?;
        rx.recv().map_err(|_| anyhow!("inference server dropped the request"))?
    }

    /// 032-R16: this slot's current Core state (before the next `act`), for core_state0 (042).
    pub fn state(&self) -> Result<HostState> {
        let (tx, rx) = mpsc::channel();
        self.tx.send(Req::State { slot: self.slot, reply: tx }).map_err(|_| anyhow!("inference server is gone"))?;
        rx.recv().map_err(|_| anyhow!("inference server dropped the request"))?
    }
}

impl Drop for InferSession {
    fn drop(&mut self) {
        let _ = self.tx.send(Req::Close(self.slot));
    }
}

struct Pending {
    slot: usize,
    obs: Box<Observation>,
    reply: Sender<Result<ActReply>>,
    since: Instant,
}

fn serve<C: CoreInit + 'static>(dir: &Path, sample: SampleMode, opts: InferOptions, rx: Receiver<Req>, ready: Sender<Result<()>>) -> Result<()> {
    let loaded = (|| -> Result<_> {
        let device = opts.backend.device()?;
        let cols = ActionColumns::from_table(action_table())?;
        let (net, meta) = checkpoint::load::<C>(dir, &cols, FEATURE_SCHEMA_VERSION, ACTION_SCHEMA_VERSION, &device)?;
        if opts.warmup_entities > 0 {
            let cfg = meta.net_config()?;
            let pad = opts.pad_entities.max(1);
            for n in (pad..=opts.warmup_entities.next_multiple_of(pad)).step_by(pad) {
                let t = Instant::now();
                let (ob, _) = uranium_model::synthetic::random_obs(&cfg, &cols, 1, n, n as u64, &device);
                let out = net.act(&ob, net.init_state(1, &device), sample.temp());
                let _ = out.action.action.into_data();
                elog!("[infer] warm-up N={n}: {:.1}s", t.elapsed().as_secs_f64());
            }
        }
        Ok((device, cols, net))
    })();
    let (device, cols, mut net) = match loaded {
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
fn run_batch<C: CoreInit>(net: &UraniumNet<C>, cols: &ActionColumns, device: &Device, state: C::State, batch: Vec<Pending>, temp: f32, pad: usize, version: u64) -> C::State {
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
        // 011-AC7: OpenBW drops a LAN client after 60 s without sync, so a stall here ends the
        // game early with both sides claiming the win. Log it with wall time to line it up.
        let unix = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
        elog!("[infer] slow batch: {b} requests took {:.1}s (unix {unix})", run.as_secs_f64());
    }
    for (r, p) in batch.into_iter().enumerate() {
        let a = action[r] as usize;
        let uses = |v: &Vec<i32>| v.get(a).copied().unwrap_or(0);
        let host = HostAction {
            action: a as u16,
            delay_bucket: delay[r] as usize,
            queued: queued[r] == 1,
            selected: if uses(&cols.uses_select) == 1 { selected[r * s..(r + 1) * s].iter().take_while(|&&i| i != n as i64).map(|&i| i as usize).collect() } else { vec![] },
            target_unit: (uses(&cols.target_kind) == TARGET_UNIT).then_some(target_unit[r] as usize),
            target_loc: (uses(&cols.target_kind) == TARGET_LOC).then_some(target_loc[r] as usize),
        };
        let reply = ActReply {
            action: host,
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

/// Writes a randomly initialised checkpoint for `cfg` (smoke tests before a BC checkpoint exists).
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

/// The default (032 §4) network on the live feature / action schema with the given core.
pub fn default_config(kind: CoreKind) -> uranium_model::UraniumNetConfig {
    let base = uranium_model::UraniumNetConfig::new();
    base.clone().with_input(uranium_model::collate::input_dims()).with_core(base.core.clone().with_kind(kind))
}
