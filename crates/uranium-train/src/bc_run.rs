//! `uranium-train bc` run loop (041): stage selection and schedule length (041-R13 one epoch per
//! stage), prefetching data thread, checkpoints by update count and wall time, Ctrl+C, and either
//! a local `--out` folder or a 061 run on the shared drive (lock, heartbeat, push, resume on any
//! host from `LATEST` with the data position, 061-R3/R5/R10).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use burn::tensor::Device;
use uranium_model::collate::input_dims;
use uranium_model::config::{CoreKind, UraniumNetConfig};
use uranium_model::{CoreInit, UraniumNet};

use crate::bc::{BcConfig, Prefetch, RunIdent, Trainer};
use crate::ckpt;
use crate::data::{MapStore, ShardSource};
use crate::storage::{LockStatus, PushOutcome, Run, StorageConfig, TakeOver};

/// What a `bc` invocation trains (041-R4, R5, R13).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BcStage {
    /// S1 pre-training, all sources.
    S1,
    /// S2 fine-tuning from S1 (learning rate x `s2_lr_scale`).
    S2,
    /// Ablation (A): S2 data from scratch at the S1 learning rate.
    AblationA,
}

impl BcStage {
    pub fn selection(self) -> crate::split::Selection {
        match self {
            BcStage::S1 => crate::split::Selection::S1,
            BcStage::S2 | BcStage::AblationA => crate::split::Selection::S2,
        }
    }

    /// Schedule length of the stage from `config/bc.toml` `[steps]` (0 = keep `total_steps`).
    pub fn steps(self, cfg: &BcConfig) -> usize {
        match self {
            BcStage::S1 => cfg.steps.s1,
            BcStage::S2 => cfg.steps.s2,
            BcStage::AblationA => cfg.steps.ablation_a,
        }
    }

    /// Warmup of the stage (041-R14: S2 and (A) from `[steps]`, 0 = keep `warmup_steps`).
    pub fn warmup(self, cfg: &BcConfig) -> usize {
        match self {
            BcStage::S1 => 0,
            BcStage::S2 => cfg.steps.s2_warmup,
            BcStage::AblationA => cfg.steps.ablation_a_warmup,
        }
    }
}

/// Where checkpoints go.
pub enum BcOut {
    /// A local folder: `step_<n>/` and `last/`.
    Local(PathBuf),
    /// 061 run `<shared>/checkpoints/<run>/` (lock, `ckpt-<step>/`, `LATEST`).
    Run { id: String, take: TakeOver },
}

pub struct BcArgs {
    pub dirs: Vec<PathBuf>,
    pub out: BcOut,
    pub config: PathBuf,
    pub stage: BcStage,
    pub core: CoreKind,
    /// Stop after this update (smoke runs); the schedule still follows the stage length.
    pub stop_at: Option<usize>,
    /// Local mode: continue from this checkpoint (weights, optimizer, step, data position).
    pub resume: Option<PathBuf>,
    /// Weights only (S1 -> S2, 041-R4): a checkpoint folder or a finished 061 run folder (`DONE`).
    pub init: Option<PathBuf>,
    /// For the device log (061-R15): `cuda`, `vulkan` or `flex`.
    pub backend: String,
    /// Ctrl+C state from [`crate::install_stop_handler`], installed before the dataset pull.
    pub stop: Arc<AtomicUsize>,
}

/// How a `bc` run ended (the process exit code, see `main`).
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

/// 061-R5: exit code of a run that could not take its lock: another PC's lock (fresh or expired,
/// without `--take-over`), a lost race for a new lock, a lock that changed since a worker saw it
/// expire, or a run reserved by another host's yield request (R14). Watchdogs never restart it.
pub const EXIT_BUSY: i32 = 75;
/// Exit code after yielding the run (061-R14).
pub const EXIT_YIELDED: i32 = 76;

/// Precision actually trained in (032-R8), for the device log.
fn precision(backend: &str) -> &'static str {
    match backend {
        "cuda" => "f32 (TF32 matmul)",
        _ => "f32",
    }
}


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

pub fn run<C: CoreInit>(args: BcArgs, dev: Device) -> Result<BcExit> {
    let mut cfg = BcConfig::load(&args.config)?;
    let stage_steps = args.stage.steps(&cfg);
    if stage_steps > 0 {
        cfg.total_steps = stage_steps;
    }
    let stage_warmup = args.stage.warmup(&cfg);
    if stage_warmup > 0 {
        cfg.warmup_steps = stage_warmup;
    }
    let ad = if cfg.grad_checkpointing { dev.autodiff().gradient_checkpointing() } else { dev.autodiff() };
    let cols = uranium_model::batch::ActionColumns::from_table(uranium_env::features::extract::action_table())?;
    let load = |dir: &Path| -> Result<(UraniumNet<C>, UraniumNetConfig)> {
        let (net, meta) = uranium_model::checkpoint::load::<C>(dir, &cols, uranium_env::features::FEATURE_SCHEMA_VERSION, uranium_env::actions::ACTION_SCHEMA_VERSION, &ad)
            .with_context(|| format!("loading {}", dir.display()))?;
        Ok((net, meta.net_config()?))
    };

    // B-2: every selected game's map must have its precompute here, checked before the lock and
    // the model (a missing one would zero the base_loc / choke / wall channels and the map
    // resource entities). The loader also refuses such a game (`MapStore::get`).
    let maps = Arc::new(MapStore::from_env());
    let src = ShardSource::from_dirs(&args.dirs)?.select(args.stage.selection()).shuffled(cfg.data.seed);
    crate::data::check_maps(&src, Some(args.stage.selection()), &maps)?;
    // B-3: what this run trains; stored in every checkpoint and compared on resume.
    let ident = RunIdent { stage: format!("{:?}", args.stage), config: cfg.fingerprint(), dataset: crate::data::dataset_fingerprint(&src)? };
    crate::elog!("{}", serde_json::json!({ "run_ident": ident, "keep_checkpoints": cfg.keep_checkpoints() }));

    // 042-R10 / 060-R6: Ctrl+C (or the watchdog's death) before the lock: nothing to release.
    let stop = args.stop.clone();
    if crate::stopping(&stop) {
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
            match crate::storage::read_yield(&r.cfg, id) {
                Ok(Some(req)) if req.host != r.cfg.host => {
                    let msg = format!(
                        "run {id} is reserved for {} (yield request, 061-R14). A worker removes it once it has seen it unchanged for 30 min; if that host is gone, delete locks/{id}.yield",
                        req.host
                    );
                    r.release()?;
                    return Err(anyhow::Error::new(crate::storage::LockBusy(msg)));
                }
                Ok(Some(_)) => {
                    if let Err(e) = crate::storage::clear_yield(&r.cfg, id) {
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
        Some(Err(e)) if e.downcast_ref::<crate::Interrupted>().is_some() => {
            crate::elog!("[bc] {e:#}; releasing the lock (042-R10)");
            run.take().expect("run mode").release()?;
            return Ok(BcExit::Stopped);
        }
        Some(Err(e)) => return Err(e),
        None => args.resume.clone(),
    };
    // 042-R10: Ctrl+C during the start: release before loading anything (nothing new to save).
    if crate::stopping(&stop)
        && let Some(r) = run.take()
    {
        crate::elog!("[bc] Ctrl+C during start: releasing the lock, nothing to save (042-R10)");
        r.release()?;
        return Ok(BcExit::Stopped);
    }
    // B-3: a resume continues the same stage, config and data, or nothing.
    if let Some(dir) = &resume_dir
        && let Err(e) = crate::bc::check_ident(dir, &ident)
    {
        if let Some(r) = run.take() {
            r.release()?;
        }
        return Err(e);
    }
    // 061-R8 (C-3): disk usage at the start (warning only).
    check_disks(&args, &run);
    let (net, net_cfg) = match (&resume_dir, &args.init) {
        (Some(d), _) => load(d)?,
        (None, Some(_)) => load(init_dir.as_ref().expect("resolved above"))?,
        (None, None) => {
            let base = UraniumNetConfig::new();
            let net_cfg = base.clone().with_input(input_dims()).with_core(base.core.clone().with_kind(args.core));
            (UraniumNet::<C>::new(&net_cfg, &cols, &ad), net_cfg)
        }
    };
    let total = cfg.total_steps;
    let stop_at = args.stop_at.unwrap_or(total).min(total);
    let every = cfg.data.checkpoint_every.max(1);
    let every_time = Duration::from_secs(cfg.data.checkpoint_minutes * 60);
    let mut t = Trainer::new(net, net_cfg, cfg.clone(), ad);
    t.feeder.set_maps(maps.clone());
    if args.stage == BcStage::S2 {
        t.lr_scale = t.cfg.s2_lr_scale;
    }
    let mut src = src;
    if let Some(dir) = &resume_dir {
        if t.resume(dir)? {
            // 061-R10: continue the data where the checkpoint was taken.
            let data = t.data.clone();
            t.feeder.restore(&data, &mut src)?;
            crate::elog!("{}", serde_json::json!({ "resumed": dir, "step": t.step, "games": data.games, "source": data.source }));
        } else {
            crate::elog!("{}: no training state (older checkpoint); weights only, fresh optimizer", dir.display());
        }
    }
    crate::elog!("{}", serde_json::json!({ "stage": format!("{:?}", args.stage), "total_steps": total, "stop_at": stop_at, "lr_scale": t.lr_scale }));
    let device_log = |r: &Option<Run>, line: serde_json::Value| {
        if let Some(r) = r {
            crate::storage::log_device(&r.cfg, &r.run_id, line);
        }
    };
    device_log(
        &run,
        serde_json::json!({ "event": "start", "step": t.step, "backend": args.backend, "precision": precision(&args.backend), "core": format!("{:?}", args.core), "stage": format!("{:?}", args.stage) }),
    );

    // Nothing to save when no update happens (resumed at or past the stop point).
    let start_step = t.step;
    let feeder = std::mem::replace(&mut t.feeder, crate::bc::Feeder::new(&cfg).with_maps(maps.clone()));
    // 041-R16: per-host depth; it changes host memory only, never the batches.
    let depth = crate::bc::prefetch_depth()?;
    crate::elog!("{}", serde_json::json!({ "prefetch": depth }));
    let mut data = Prefetch::start(feeder, src, depth);
    let (mut last_ckpt, mut last_beat) = (Instant::now(), Instant::now());
    let mut lock_lost = false;
    let mut exhausted = false;
    let mut yielded = false;
    // 061-R14 ②: a yield whose checkpoint push was deferred waits for the next pushed one.
    let mut yield_waiting = false;
    // Step of the last checkpoint pushed (061-R14 yield) and of the last one written at all
    // (B-8: a step is never written twice, whatever its push did).
    let mut saved_at: Option<usize> = None;
    let mut sealed_at: Option<usize> = None;
    while t.step < stop_at && !crate::stopping(&stop) {
        // 060-AC4: time blocked on the data thread (loader stall), logged per update.
        let waited = Instant::now();
        let Some(batch) = data.next()? else {
            exhausted = true;
            crate::elog!("data exhausted after {} updates (end of the epoch)", t.step);
            break;
        };
        let wait_secs = waited.elapsed().as_secs_f64();
        let started = Instant::now();
        let l = t.train_batch(batch)?;
        crate::elog!(
            "{}",
            serde_json::json!({ "step": t.step, "loss": l, "games": t.games(), "invalid_labels": t.invalid_labels(), "no_result_games": t.data.no_result_games, "secs": (started.elapsed().as_secs_f64() * 100.0).round() / 100.0, "wait_secs": (wait_secs * 100.0).round() / 100.0 })
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
        if t.step % every == 0 || (!every_time.is_zero() && last_ckpt.elapsed() >= every_time) {
            match save(&t, &args, &mut run, false, &mut sealed_at, &ident)? {
                Some(PushOutcome::Pushed) => {
                    device_log(&run, serde_json::json!({ "event": "checkpoint", "step": t.step }));
                    saved_at = Some(t.step);
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
        let request = run.as_ref().and_then(|r| crate::storage::yield_by_other(&r.cfg, &r.run_id));
        if request.is_none() && yield_waiting {
            // The request was withdrawn or removed as expired: a new one starts over.
            yield_waiting = false;
        }
        if let Some(req) = request {
            if saved_at == Some(t.step) {
                crate::elog!("[bc] yield requested by {} (priority {}): checkpoint pushed, releasing (061-R14)", req.host, req.priority);
                yielded = true;
                break;
            }
            if !yield_waiting {
                yield_waiting = true;
                crate::elog!("[bc] yield requested by {} (priority {}): checkpoint and release (061-R14)", req.host, req.priority);
                match save(&t, &args, &mut run, false, &mut sealed_at, &ident)? {
                    Some(PushOutcome::Pushed) => {
                        device_log(&run, serde_json::json!({ "event": "checkpoint", "step": t.step }));
                        saved_at = Some(t.step);
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
        device_log(&run, serde_json::json!({ "event": "stop", "step": t.step, "reason": "lock lost" }));
        return Ok(BcExit::LockLost);
    }
    let finished = exhausted || t.step >= stop_at;
    // A yield leaves the loop only right after a pushed checkpoint (061-R14 ②), so it lands here.
    // No update in this session (resumed at the stop point, or stopped before the first update):
    // nothing new to save; a run then just releases its lock.
    let no_update = t.step == start_step && (resume_dir.is_some() || (run.is_some() && !finished));
    let pushed = if no_update || saved_at == Some(t.step) {
        crate::elog!("[bc] step {} is already saved: nothing more to save", t.step);
        Some(PushOutcome::Pushed)
    } else {
        save(&t, &args, &mut run, true, &mut sealed_at, &ident)?
    };
    let reason = if yielded {
        "yield"
    } else if finished {
        "done"
    } else {
        "ctrl-c"
    };
    device_log(&run, serde_json::json!({ "event": "stop", "step": t.step, "reason": reason }));
    if let Some(r) = run {
        // 061-R11-5/6: keep the lock when the final push did not go through.
        match pushed {
            Some(PushOutcome::Pushed) => {
                if finished && !yielded {
                    crate::storage::mark_done(&r.cfg, &r.run_id, t.step as u64)?;
                }
                r.release()?
            }
            other => crate::elog!("[bc] final push {other:?}: keeping the lock so the next start recovers it (061-R11-5/6)"),
        }
    }
    if finished {
        crate::elog!("{}", serde_json::json!({ "done": true, "step": t.step, "games": t.games() }));
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
fn check_disks(args: &BcArgs, run: &Option<Run>) {
    match (&args.out, run) {
        (_, Some(r)) => {
            crate::disk::warn_storage("bc", &r.cfg);
        }
        (BcOut::Local(out), None) => {
            let p = if out.exists() { out.as_path() } else { out.parent().filter(|p| p.exists()).unwrap_or(Path::new(".")) };
            crate::disk::warn_disks("bc", &[("checkpoint folder", p)]);
        }
        (BcOut::Run { .. }, None) => {}
    }
}

/// Local: `step_<n>/` (or `last/` when `last`). Run: stage → seal → push (061-R4), then prune.
/// B-8: a step already sealed (`sealed_at`, e.g. a time checkpoint whose push was deferred just
/// before Ctrl+C) is never written again: its contents would differ (lineage), and a share that
/// already got the first copy would then refuse every push of the second. Its push is retried.
fn save<C: CoreInit>(t: &Trainer<C>, args: &BcArgs, run: &mut Option<Run>, last: bool, sealed_at: &mut Option<usize>, ident: &RunIdent) -> Result<Option<PushOutcome>> {
    let out = match (&args.out, run.as_mut()) {
        (BcOut::Local(out), _) => {
            let dir = if last { out.join("last") } else { out.join(format!("step_{:08}", t.step)) };
            t.save(&dir)?;
            crate::bc::write_ident(&dir, ident)?;
            ckpt::copy_configs(&dir, &[&args.config])?;
            None
        }
        (BcOut::Run { .. }, Some(r)) => Some(run_checkpoint(r, t.step as u64, sealed_at, |dir| {
            t.save(dir)?;
            crate::bc::write_ident(dir, ident)?;
            ckpt::copy_configs(dir, &[&args.config])
        })?),
        (BcOut::Run { .. }, None) => unreachable!("run mode holds a Run"),
    };
    // 061-R8 (C-3): disk usage at every checkpoint.
    check_disks(args, run);
    Ok(out)
}

/// One checkpoint of a run at `step`: `write` fills the stage folder, then seal → push (061-R4)
/// → prune. B-8: when `step` was already sealed in this session (`sealed_at`, e.g. a time
/// checkpoint whose push was deferred right before Ctrl+C), it is not written again (a second copy
/// differs in its lineage, and a share that already got the first one would refuse it forever);
/// its push is retried instead. B-7: a failed prune is only logged.
pub fn run_checkpoint(r: &mut Run, step: u64, sealed_at: &mut Option<usize>, write: impl FnOnce(&Path) -> Result<()>) -> Result<PushOutcome> {
    let out = if *sealed_at == Some(step as usize) {
        crate::elog!("[bc] {} is already written: pushing it again instead of writing it twice (B-8)", crate::storage::ckpt_name(step));
        r.push(step)?
    } else {
        let dir = r.stage_dir(step)?;
        write(&dir)?;
        r.seal(step)?;
        *sealed_at = Some(step as usize);
        r.push(step)?
    };
    crate::elog!("[bc] checkpoint {} {:?}", crate::storage::ckpt_name(step), out);
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
