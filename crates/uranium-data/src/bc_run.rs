//! The BC run loop (041) for any trainer (AMB-000-4: the PyTorch trainer through `uranium._core`):
//! stage selection and schedule (041-R13, R14), the prefetching data thread (041-R16),
//! checkpoints by update count and wall time, Ctrl+C (042-R10), and either a local `--out`
//! folder or a 061 run on the shared drive (lock, heartbeat, push, yield, resume on any host
//! from `LATEST` with the data position, 061-R3/R5/R10/R14). The step counter, learning rate
//! (041-R3) and the training state file (`train_state.json`, 061-R10) are kept here; the trainer
//! only turns a batch into an update and writes its weights, optimizer state and model metadata.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, ensure};
use uranium_store::ckpt::{self, TrainState};
use uranium_store::storage::{self, LockStatus, PushOutcome, Run, StorageConfig, TakeOver};

use crate::bc::{Batch, BcConfig, BcStage, FeedState, Feeder, Prefetch, RunIdent, check_ident, write_ident};
use crate::data::{MapStore, ShardSource};

/// The model side of a BC run.
pub trait BcTrainer {
    /// One update on `batch` at learning rate `lr` (041-R3); `step` is the update's index (0-based,
    /// dropout seeds). Returns the loss parts for the log line.
    fn train_batch(&mut self, batch: Batch, lr: f64, step: usize) -> Result<serde_json::Value>;
    /// Weights, optimizer state and the model metadata (032-R6) into `dir`; `train` is the
    /// training summary for the metadata. The loop writes `train_state.json` itself.
    fn save(&mut self, dir: &Path, train: &serde_json::Value) -> Result<()>;
}

/// Where the trainer's starting weights come from.
pub enum Start<'a> {
    /// A new run: initial weights from the model config.
    New,
    /// Continue a checkpoint: weights and optimizer state (061-R10).
    Resume(&'a Path),
    /// Weights only, fresh optimizer (S1 -> S2, 041-R4).
    Init(&'a Path),
}

/// Where checkpoints go.
pub enum BcOut {
    /// A local folder: `step_<n>/` and `last/`.
    Local(PathBuf),
    /// 061 run `<shared>/checkpoints/<run>/` (lock, `ckpt-<step>/`, `LATEST`).
    Run { id: String, take: TakeOver },
}

pub struct BcRunArgs {
    pub dirs: Vec<PathBuf>,
    pub out: BcOut,
    pub config: PathBuf,
    pub stage: BcStage,
    /// Stop after this update (smoke runs); the schedule still follows the stage length.
    pub stop_at: Option<usize>,
    /// Local mode: continue from this checkpoint (weights, optimizer, step, data position).
    pub resume: Option<PathBuf>,
    /// Weights only (S1 -> S2, 041-R4): a checkpoint folder or a finished 061 run folder (`DONE`).
    pub init: Option<PathBuf>,
    /// For the device log (061-R15): `cuda`, `rocm` or `cpu`, and the precision (032-R8).
    pub backend: String,
    pub precision: String,
    /// Core kind for the device log (`Lstm` / `Gtrxl`).
    pub core: String,
    /// Ctrl+C state (`uranium_store::install_stop_handler`), installed before the dataset pull.
    pub stop: Arc<AtomicUsize>,
    /// Map precomputes; `None` = `URANIUM_MAPS_DIR` (B-2).
    pub maps: Option<Arc<MapStore>>,
}

/// How a `bc` run ended (the process exit code).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BcExit {
    /// The stage (or `--steps`) is complete; a run gets `DONE` (061-R13). Exit 0.
    Done,
    /// Ctrl+C clean stop after a checkpoint. Exit 0.
    Stopped,
    /// Lock taken over by another PC: nothing pushed (061-R5). Exit 0.
    LockLost,
    /// Gave the run to the host that asked (061-R14) after a checkpoint that was pushed. Exit 76.
    Yielded,
}

/// 061-R5: exit code of a run that could not take its lock (another PC's lock without
/// `--take-over`, a lost race, a lock that changed since a worker saw it expire, or a run reserved
/// by another host's yield request, R14). Watchdogs never restart it.
pub const EXIT_BUSY: i32 = 75;
/// Exit code after yielding the run (061-R14).
pub const EXIT_YIELDED: i32 = 76;

/// 061-R13 (H-1): the `--init` weights. A 061 run folder (it has `LATEST` or `DONE`) must be
/// finished: its `DONE` names the final checkpoint. A run still training (or given up) is
/// refused, so S2 never starts from S1's intermediate weights. Any other folder is a checkpoint.
pub fn resolve_init(dir: &Path) -> Result<PathBuf> {
    let done = dir.join("DONE");
    if dir.join("LATEST").is_file() || done.exists() {
        let name = std::fs::read_to_string(&done).with_context(|| format!("--init run {} is not finished (no DONE); S2 waits for it (061-R13)", dir.display()))?;
        return Ok(dir.join(name.trim()));
    }
    Ok(dir.to_path_buf())
}

/// A checkpoint folder, or the one `LATEST` names in a 061 run folder (evaluation of a run in
/// progress is fine; `--init` uses [`resolve_init`]).
pub fn resolve_checkpoint(dir: &Path) -> Result<PathBuf> {
    let latest = dir.join("LATEST");
    if latest.is_file() {
        let name = std::fs::read_to_string(&latest)?.trim().to_string();
        return Ok(dir.join(name));
    }
    Ok(dir.to_path_buf())
}

/// Step, learning-rate scale and data position of the run (061-R10).
struct Progress {
    step: usize,
    lr_scale: f64,
    data: FeedState,
}

impl Progress {
    /// `train_state.json` (061-R10): the same fields the Burn trainer wrote.
    fn state(&self) -> TrainState {
        TrainState {
            kind: "bc".into(),
            step: self.step as u64,
            update: self.step as u64,
            policy_version: self.step as u64,
            rng: vec![self.data.rng],
            extra: serde_json::json!({ "lr_scale": self.lr_scale, "games": self.data.games, "invalid_labels": self.data.invalid_labels, "data": self.data }),
        }
    }

    fn load(dir: &Path) -> Result<Self> {
        let s = ckpt::load_state(dir)?.ok_or_else(|| anyhow::anyhow!("{} has no training state ({})", dir.display(), ckpt::STATE_FILE))?;
        ensure!(s.kind == "bc", "{} is a {} checkpoint, not bc", dir.display(), s.kind);
        let data = s.extra.get("data").cloned().ok_or_else(|| anyhow::anyhow!("{}: no data position in the training state (061-R10)", dir.display()))?;
        Ok(Self { step: s.step as usize, lr_scale: s.extra["lr_scale"].as_f64().unwrap_or(1.0), data: serde_json::from_value(data)? })
    }
}

/// The model metadata's training summary (041-R10).
fn train_summary(cfg: &BcConfig, p: &Progress) -> serde_json::Value {
    serde_json::json!({ "bc": cfg, "step": p.step, "lr_scale": p.lr_scale, "games": p.data.games, "invalid_labels": p.data.invalid_labels })
}

/// Runs a BC stage. `make` builds the trainer once the start point is known (after the lock and
/// the resume lookup); `entity_chunk` is this host's `URANIUM_BC_ENTITY_CHUNK` (041-R17).
pub fn run(args: BcRunArgs, make: impl FnOnce(&BcConfig, Start<'_>, usize) -> Result<Box<dyn BcTrainer>>) -> Result<BcExit> {
    let cfg = BcConfig::load(&args.config)?.for_stage(args.stage);

    // B-2: every selected game's map must have its precompute here, checked before the lock and
    // the model. The loader also refuses such a game (`MapStore::get`).
    let maps = args.maps.clone().unwrap_or_else(|| Arc::new(MapStore::from_env()));
    let src = ShardSource::from_dirs(&args.dirs)?.select(args.stage.selection()).shuffled(cfg.data.seed);
    crate::data::check_maps(&src, Some(args.stage.selection()), &maps)?;
    // B-3: what this run trains; stored in every checkpoint and compared on resume.
    let ident = RunIdent { stage: format!("{:?}", args.stage), config: cfg.fingerprint(), dataset: crate::data::dataset_fingerprint(&src)? };
    crate::elog!("{}", serde_json::json!({ "run_ident": ident, "keep_checkpoints": cfg.keep_checkpoints() }));

    // 042-R10 / 060-R6: Ctrl+C (or the watchdog's death) before the lock: nothing to release.
    let stop = args.stop.clone();
    if uranium_store::stopping(&stop) {
        crate::elog!("[bc] Ctrl+C before start: not started");
        return Ok(BcExit::Stopped);
    }
    // 061-R13 (H-1): checked before the lock is taken, so a refusal leaves no lock behind.
    let init_dir = args.init.as_deref().map(resolve_init).transpose()?;
    // 061-R3 / R5: a run takes the lock and resumes from LATEST (or an unpushed local checkpoint).
    let mut run = match &args.out {
        BcOut::Run { id, take } => {
            let mut storage = StorageConfig::from_env()?;
            // 061-R6 (G-D6): the run's own retention (`keep_checkpoints`, 0 = all).
            storage.keep_latest = cfg.keep_checkpoints();
            let r = Run::acquire_with(storage, id, take.clone())?;
            // 061-R14: a live yield request reserves the run for the host that asked.
            match storage::read_yield(&r.cfg, id) {
                Ok(Some(req)) if req.host != r.cfg.host => {
                    let msg = format!(
                        "run {id} is reserved for {} (yield request, 061-R14). A worker removes it once it has seen it unchanged for 30 min; if that host is gone, delete locks/{id}.yield",
                        req.host
                    );
                    r.release()?;
                    return Err(anyhow::Error::new(storage::LockBusy(msg)));
                }
                Ok(Some(_)) => {
                    if let Err(e) = storage::clear_yield(&r.cfg, id) {
                        crate::elog!("[bc] could not remove this host's yield request on {id} ({e:#}); it expires after 30 min (061-R14)");
                    }
                }
                Ok(None) => {}
                Err(e) => {
                    let _ = r.release();
                    return Err(e.context("reading the yield request (061-R13: unknown, not started)"));
                }
            }
            Some(r)
        }
        BcOut::Local(_) => None,
    };
    let begun = run.as_mut().map(|r| r.begin());
    let resume_dir = match begun {
        Some(Ok(res)) => res.map(|res| res.dir),
        Some(Err(e)) if e.downcast_ref::<uranium_store::Interrupted>().is_some() => {
            crate::elog!("[bc] {e:#}; releasing the lock (042-R10)");
            run.take().expect("run mode").release()?;
            return Ok(BcExit::Stopped);
        }
        Some(Err(e)) => return Err(e),
        None => args.resume.clone(),
    };
    // 042-R10: Ctrl+C during the start: release before loading anything (nothing new to save).
    if uranium_store::stopping(&stop)
        && let Some(r) = run.take()
    {
        crate::elog!("[bc] Ctrl+C during start: releasing the lock, nothing to save (042-R10)");
        r.release()?;
        return Ok(BcExit::Stopped);
    }
    // B-3: a resume continues the same stage, config and data, or nothing.
    if let Some(dir) = &resume_dir
        && let Err(e) = check_ident(dir, &ident)
    {
        if let Some(r) = run.take() {
            r.release()?;
        }
        return Err(e);
    }
    // 061-R8 (C-3): disk usage at the start (warning only).
    check_disks(&args, &run);
    // 041-R16 / R17 / R18: per host; host and device memory only, never the batches or gradients.
    let depth = crate::bc::prefetch_depth()?;
    let (entity_chunk, pack) = (crate::bc::entity_chunk()?, crate::bc::pack_batch()?);
    crate::elog!("{}", serde_json::json!({ "prefetch": depth, "entity_chunk": entity_chunk, "pack_batch": pack }));
    let start = match (&resume_dir, &init_dir) {
        (Some(d), _) => Start::Resume(d),
        (None, Some(d)) => Start::Init(d),
        (None, None) => Start::New,
    };
    let mut trainer = match make(&cfg, start, entity_chunk) {
        Ok(t) => t,
        Err(e) => {
            if let Some(r) = run.take() {
                let _ = r.release();
            }
            return Err(e);
        }
    };

    let total = cfg.total_steps;
    let stop_at = args.stop_at.unwrap_or(total).min(total);
    let every = cfg.data.checkpoint_every.max(1);
    let every_time = Duration::from_secs(cfg.data.checkpoint_minutes * 60);
    let mut feeder = Feeder::new(&cfg).with_maps(maps.clone());
    feeder.set_pack(pack);
    let mut src = src;
    let mut p = Progress { step: 0, lr_scale: if args.stage == BcStage::S2 { cfg.s2_lr_scale } else { 1.0 }, data: feeder.state(None) };
    if let Some(dir) = &resume_dir {
        p = Progress::load(dir)?;
        // 061-R10: continue the data where the checkpoint was taken.
        feeder.restore(&p.data, &mut src)?;
        crate::elog!("{}", serde_json::json!({ "resumed": dir, "step": p.step, "games": p.data.games, "source": p.data.source }));
    }
    crate::elog!("{}", serde_json::json!({ "stage": format!("{:?}", args.stage), "total_steps": total, "stop_at": stop_at, "lr_scale": p.lr_scale }));
    let device_log = |r: &Option<Run>, line: serde_json::Value| {
        if let Some(r) = r {
            storage::log_device(&r.cfg, &r.run_id, line);
        }
    };
    device_log(&run, serde_json::json!({ "event": "start", "step": p.step, "backend": args.backend, "precision": args.precision, "core": args.core, "stage": format!("{:?}", args.stage) }));

    // Nothing to save when no update happens (resumed at or past the stop point).
    let start_step = p.step;
    let mut data = Prefetch::start(feeder, src, depth);
    let (mut last_ckpt, mut last_beat) = (Instant::now(), Instant::now());
    let (mut lock_lost, mut exhausted, mut yielded) = (false, false, false);
    // 061-R14 ②: a yield whose checkpoint push was deferred waits for the next pushed one.
    let mut yield_waiting = false;
    // Step of the last checkpoint pushed (061-R14 yield) and of the last one written at all
    // (B-8: a step is never written twice, whatever its push did).
    let mut saved_at: Option<usize> = None;
    let mut sealed_at: Option<usize> = None;
    let save = |trainer: &mut dyn BcTrainer, p: &Progress, run: &mut Option<Run>, last: bool, sealed_at: &mut Option<usize>| -> Result<Option<PushOutcome>> {
        let mut write = |dir: &Path| -> Result<()> {
            trainer.save(dir, &train_summary(&cfg, p))?;
            ckpt::save_state(dir, &p.state())?;
            write_ident(dir, &ident)?;
            ckpt::copy_configs(dir, &[&args.config])
        };
        let out = match (&args.out, run.as_mut()) {
            (BcOut::Local(out), _) => {
                write(&if last { out.join("last") } else { out.join(format!("step_{:08}", p.step)) })?;
                None
            }
            (BcOut::Run { .. }, Some(r)) => Some(run_checkpoint(r, p.step as u64, sealed_at, write)?),
            (BcOut::Run { .. }, None) => unreachable!("run mode holds a Run"),
        };
        // 061-R8 (C-3): disk usage at every checkpoint.
        check_disks(&args, run);
        Ok(out)
    };
    while p.step < stop_at && !uranium_store::stopping(&stop) {
        // 060-AC4: time blocked on the data thread (loader stall), logged per update.
        let waited = Instant::now();
        let Some(batch) = data.next()? else {
            exhausted = true;
            crate::elog!("data exhausted after {} updates (end of the epoch)", p.step);
            break;
        };
        let wait_secs = waited.elapsed().as_secs_f64();
        let started = Instant::now();
        let state = batch.state.clone();
        let lr = cfg.lr_at(p.step, p.lr_scale);
        let loss = trainer.train_batch(batch, lr, p.step)?;
        p.step += 1;
        p.data = state;
        crate::elog!(
            "{}",
            serde_json::json!({ "step": p.step, "loss": loss, "lr": lr, "games": p.data.games, "invalid_labels": p.data.invalid_labels, "no_result_games": p.data.no_result_games, "secs": (started.elapsed().as_secs_f64() * 100.0).round() / 100.0, "wait_secs": (wait_secs * 100.0).round() / 100.0 })
        );
        if let Some(r) = &mut run
            && last_beat.elapsed() >= Duration::from_secs(60)
        {
            last_beat = Instant::now();
            match r.heartbeat() {
                LockStatus::Held => {}
                LockStatus::Lost(l) => {
                    crate::elog!("[bc] lock lost to {l:?}: stopping without a push (061-R5)");
                    lock_lost = true;
                    break;
                }
                LockStatus::Unreachable(e) => crate::elog!("[bc] heartbeat: share unreachable ({e}), continuing (061-R11-3)"),
            }
        }
        if p.step % every == 0 || (!every_time.is_zero() && last_ckpt.elapsed() >= every_time) {
            match save(trainer.as_mut(), &p, &mut run, false, &mut sealed_at)? {
                Some(PushOutcome::Pushed) => {
                    device_log(&run, serde_json::json!({ "event": "checkpoint", "step": p.step }));
                    saved_at = Some(p.step);
                }
                Some(PushOutcome::LockLost(l)) => {
                    crate::elog!("[bc] lock lost to {l:?}: stopping without a push (061-R11-4)");
                    lock_lost = true;
                    break;
                }
                _ => {}
            }
            last_ckpt = Instant::now();
        }
        // 061-R14: another host asked for this run: checkpoint, push, release, leave. If the push
        // is deferred (share unreachable, R11-3), keep the lock and train on; yield after the
        // next checkpoint that is pushed (R14 ②).
        let request = run.as_ref().and_then(|r| storage::yield_by_other(&r.cfg, &r.run_id));
        if request.is_none() && yield_waiting {
            yield_waiting = false;
        }
        if let Some(req) = request {
            if saved_at == Some(p.step) {
                crate::elog!("[bc] yield requested by {} (priority {}): checkpoint pushed, releasing (061-R14)", req.host, req.priority);
                yielded = true;
                break;
            }
            if !yield_waiting {
                yield_waiting = true;
                crate::elog!("[bc] yield requested by {} (priority {}): checkpoint and release (061-R14)", req.host, req.priority);
                match save(trainer.as_mut(), &p, &mut run, false, &mut sealed_at)? {
                    Some(PushOutcome::Pushed) => {
                        device_log(&run, serde_json::json!({ "event": "checkpoint", "step": p.step }));
                        saved_at = Some(p.step);
                        yielded = true;
                        break;
                    }
                    Some(PushOutcome::LockLost(l)) => {
                        crate::elog!("[bc] lock lost to {l:?}: stopping without a push (061-R11-4)");
                        lock_lost = true;
                        break;
                    }
                    other => crate::elog!("[bc] yield: push {other:?}; keeping the lock and training on, yielding after the next pushed checkpoint (061-R14 ②)"),
                }
                last_ckpt = Instant::now();
            }
        }
    }
    if lock_lost {
        device_log(&run, serde_json::json!({ "event": "stop", "step": p.step, "reason": "lock lost" }));
        return Ok(BcExit::LockLost);
    }
    let finished = exhausted || p.step >= stop_at;
    // No update in this session (resumed at the stop point, or stopped before the first update):
    // nothing new to save; a run then just releases its lock.
    let no_update = p.step == start_step && (resume_dir.is_some() || (run.is_some() && !finished));
    let pushed = if no_update || saved_at == Some(p.step) {
        crate::elog!("[bc] step {} is already saved: nothing more to save", p.step);
        Some(PushOutcome::Pushed)
    } else {
        save(trainer.as_mut(), &p, &mut run, true, &mut sealed_at)?
    };
    let reason = if yielded {
        "yield"
    } else if finished {
        "done"
    } else {
        "ctrl-c"
    };
    device_log(&run, serde_json::json!({ "event": "stop", "step": p.step, "reason": reason }));
    if let Some(r) = run {
        // 061-R11-5/6: keep the lock when the final push did not go through.
        match pushed {
            Some(PushOutcome::Pushed) => {
                if finished && !yielded {
                    storage::mark_done(&r.cfg, &r.run_id, p.step as u64)?;
                }
                r.release()?
            }
            other => crate::elog!("[bc] final push {other:?}: keeping the lock so the next start recovers it (061-R11-5/6)"),
        }
    }
    if finished {
        crate::elog!("{}", serde_json::json!({ "done": true, "step": p.step, "games": p.data.games }));
    }
    Ok(if yielded {
        BcExit::Yielded
    } else if finished {
        BcExit::Done
    } else {
        BcExit::Stopped
    })
}

/// 061-R8 (C-3): disk usage of the shared drive and the local cache (run), or of the output
/// folder (local mode). Warnings only.
fn check_disks(args: &BcRunArgs, run: &Option<Run>) {
    match (&args.out, run) {
        (_, Some(r)) => {
            uranium_store::disk::warn_storage("bc", &r.cfg);
        }
        (BcOut::Local(out), None) => {
            let p = if out.exists() { out.as_path() } else { out.parent().filter(|p| p.exists()).unwrap_or(Path::new(".")) };
            uranium_store::disk::warn_disks("bc", &[("checkpoint folder", p)]);
        }
        (BcOut::Run { .. }, None) => {}
    }
}

/// One checkpoint of a run at `step`: `write` fills the stage folder, then seal → push (061-R4)
/// → prune. B-8: when `step` was already sealed in this session (`sealed_at`, e.g. a time
/// checkpoint whose push was deferred right before Ctrl+C), it is not written again (a second copy
/// differs in its lineage, and a share that already got the first one would refuse it forever);
/// its push is retried instead. B-7: a failed prune is only logged.
pub fn run_checkpoint(r: &mut Run, step: u64, sealed_at: &mut Option<usize>, write: impl FnOnce(&Path) -> Result<()>) -> Result<PushOutcome> {
    let out = if *sealed_at == Some(step as usize) {
        crate::elog!("[bc] {} is already written: pushing it again instead of writing it twice (B-8)", storage::ckpt_name(step));
        r.push(step)?
    } else {
        let dir = r.stage_dir(step)?;
        write(&dir)?;
        r.seal(step)?;
        *sealed_at = Some(step as usize);
        r.push(step)?
    };
    crate::elog!("[bc] checkpoint {} {:?}", storage::ckpt_name(step), out);
    if out == PushOutcome::Pushed {
        // B-7 / 061-R6: a failed delete is a warning; the run goes on.
        match r.prune() {
            Ok(0) => {}
            Ok(n) => crate::elog!("[bc] prune: {n} checkpoint folder(s) not deleted, left for the next prune (061-R6)"),
            Err(e) => crate::elog!("[bc] WARNING: prune failed ({e:#}); training goes on (061-R6)"),
        }
    }
    Ok(out)
}
