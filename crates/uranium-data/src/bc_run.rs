use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, ensure};
use uranium_store::ckpt::{self, TrainState};
use uranium_store::storage::{self, LockStatus, PushOutcome, Run, StorageConfig, TakeOver};

use crate::bc::{Batch, BcConfig, BcStage, FeedState, Feeder, Prefetch, RunIdent, check_ident, write_ident};
use crate::data::MapStore;

pub trait BcTrainer {
    fn train_batch(&mut self, batch: Batch, lr: f64, step: usize) -> Result<serde_json::Value>;
    fn save(&mut self, dir: &Path, train: &serde_json::Value) -> Result<()>;
}

pub enum Start<'a> {
    New,
    Resume(&'a Path),
    Init(&'a Path),
}

pub enum BcOut {
    Local(PathBuf),
    Run { id: String, take: TakeOver },
}

pub struct BcRunArgs {
    pub dirs: Vec<PathBuf>,
    pub out: BcOut,
    pub config: PathBuf,
    pub stage: BcStage,
    pub stop_at: Option<usize>,
    pub resume: Option<PathBuf>,
    pub init: Option<PathBuf>,
    pub backend: String,
    pub precision: String,
    pub core: String,
    pub stop: Arc<AtomicUsize>,
    pub maps: Option<Arc<MapStore>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BcExit {
    Done,
    Stopped,
    LockLost,
    Yielded,
}

pub fn resolve_init(dir: &Path) -> Result<PathBuf> {
    let done = dir.join("DONE");
    if dir.join("LATEST").is_file() || done.exists() {
        let name = std::fs::read_to_string(&done).with_context(|| format!("--init run {} is not finished (no DONE); S2 waits for it (061-R13)", dir.display()))?;
        return Ok(dir.join(name.trim()));
    }
    Ok(dir.to_path_buf())
}

pub fn resolve_checkpoint(dir: &Path) -> Result<PathBuf> {
    let latest = dir.join("LATEST");
    if latest.is_file() {
        let name = std::fs::read_to_string(&latest)?.trim().to_string();
        return Ok(dir.join(name));
    }
    Ok(dir.to_path_buf())
}

struct Progress {
    step: usize,
    lr_scale: f64,
    data: FeedState,
}

impl Progress {
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

fn train_summary(cfg: &BcConfig, p: &Progress) -> serde_json::Value {
    serde_json::json!({ "bc": cfg, "step": p.step, "lr_scale": p.lr_scale, "games": p.data.games, "invalid_labels": p.data.invalid_labels })
}

pub fn run(args: BcRunArgs, make: impl FnOnce(&BcConfig, Start<'_>, usize) -> Result<Box<dyn BcTrainer>>) -> Result<BcExit> {
    let cfg = BcConfig::load(&args.config)?.for_stage(args.stage);

    let maps = args.maps.clone().unwrap_or_else(|| Arc::new(MapStore::from_env()));
    let src = cfg.train_source(&args.dirs, args.stage.selection())?;
    let scan = crate::data::check_maps(&src, Some(args.stage.selection()), &maps)?;
    let src = src.skip_empty(&scan);
    if let Some(s) = &cfg.data.subsample {
        let per = crate::bc::subsample_counts(&scan, args.stage.selection(), s, cfg.data.seed);
        let absent: Vec<&String> = s.keep_sources.iter().filter(|k| !per.contains_key(*k)).collect();
        let games: serde_json::Map<String, serde_json::Value> = per.iter().map(|(k, (n, kept))| (k.clone(), serde_json::json!({ "selected": n, "kept": kept }))).collect();
        let mut line = serde_json::json!({ "subsample": s, "seed": cfg.data.seed, "games": games, "keep_sources_absent": absent, "note": "041-R21" });
        if s.fraction_by_source.is_some() {
            let missing: Vec<String> = s.share_sources().into_iter().filter(|k| !per.contains_key(k)).collect();
            line["fraction_by_source_absent"] = serde_json::json!(missing);
        }
        uranium_store::elog!("{line}");
    }
    let ident = RunIdent { stage: format!("{:?}", args.stage), config: cfg.fingerprint(), dataset: crate::data::dataset_fingerprint(&src)? };
    uranium_store::elog!("{}", serde_json::json!({ "run_ident": ident, "keep_checkpoints": cfg.keep_checkpoints() }));

    let stop = args.stop.clone();
    if uranium_store::stopping(&stop) {
        uranium_store::elog!("[bc] Ctrl+C before start: not started");
        return Ok(BcExit::Stopped);
    }
    let init_dir = args.init.as_deref().map(resolve_init).transpose()?;
    let mut run = match &args.out {
        BcOut::Run { id, take } => {
            let mut storage = StorageConfig::from_env()?;
            storage.keep_latest = cfg.keep_checkpoints();
            let r = Run::acquire_with(storage, id, take.clone())?;
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
                        uranium_store::elog!("[bc] could not remove this host's yield request on {id} ({e:#}); it expires after 30 min (061-R14)");
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
            uranium_store::elog!("[bc] {e:#}; releasing the lock (042-R10)");
            run.take().expect("run mode").release()?;
            return Ok(BcExit::Stopped);
        }
        Some(Err(e)) => return Err(e),
        None => args.resume.clone(),
    };
    if uranium_store::stopping(&stop)
        && let Some(r) = run.take()
    {
        uranium_store::elog!("[bc] Ctrl+C during start: releasing the lock, nothing to save (042-R10)");
        r.release()?;
        return Ok(BcExit::Stopped);
    }
    if let Some(dir) = &resume_dir
        && let Err(e) = check_ident(dir, &ident)
    {
        if let Some(r) = run.take() {
            r.release()?;
        }
        return Err(e);
    }
    check_disks(&args, &run);
    let depth = crate::bc::prefetch_depth()?;
    let (entity_chunk, pack) = (crate::bc::entity_chunk()?, crate::bc::pack_batch()?);
    uranium_store::elog!("{}", serde_json::json!({ "prefetch": depth, "entity_chunk": entity_chunk, "pack_batch": pack, "micro_lanes": cfg.micro_lanes }));
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
        feeder.restore(&p.data, &mut src)?;
        uranium_store::elog!("{}", serde_json::json!({ "resumed": dir, "step": p.step, "games": p.data.games, "source": p.data.source }));
    }
    uranium_store::elog!("{}", serde_json::json!({ "stage": format!("{:?}", args.stage), "total_steps": total, "stop_at": stop_at, "lr_scale": p.lr_scale }));
    let device_log = |r: &Option<Run>, line: serde_json::Value| {
        if let Some(r) = r {
            storage::log_device(&r.cfg, &r.run_id, line);
        }
    };
    device_log(&run, serde_json::json!({ "event": "start", "step": p.step, "backend": args.backend, "precision": args.precision, "core": args.core, "stage": format!("{:?}", args.stage) }));

    let start_step = p.step;
    let mut data = Prefetch::start(feeder, src, depth);
    let (mut last_ckpt, mut last_beat) = (Instant::now(), Instant::now());
    let (mut lock_lost, mut exhausted, mut yielded) = (false, false, false);
    let mut yield_waiting = false;
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
        check_disks(&args, run);
        Ok(out)
    };
    while p.step < stop_at && !uranium_store::stopping(&stop) {
        let waited = Instant::now();
        let Some(batch) = data.next()? else {
            exhausted = true;
            uranium_store::elog!("data exhausted after {} updates (end of the epoch)", p.step);
            break;
        };
        let wait_secs = waited.elapsed().as_secs_f64();
        let started = Instant::now();
        let state = batch.state.clone();
        let lr = cfg.lr_at(p.step, p.lr_scale);
        let loss = trainer.train_batch(batch, lr, p.step)?;
        p.step += 1;
        p.data = state;
        uranium_store::elog!(
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
                    uranium_store::elog!("[bc] lock lost to {l:?}: stopping without a push (061-R5)");
                    lock_lost = true;
                    break;
                }
                LockStatus::Unreachable(e) => uranium_store::elog!("[bc] heartbeat: share unreachable ({e}), continuing (061-R11-3)"),
            }
        }
        if p.step % every == 0 || (!every_time.is_zero() && last_ckpt.elapsed() >= every_time) {
            match save(trainer.as_mut(), &p, &mut run, false, &mut sealed_at)? {
                Some(PushOutcome::Pushed) => {
                    device_log(&run, serde_json::json!({ "event": "checkpoint", "step": p.step }));
                    saved_at = Some(p.step);
                }
                Some(PushOutcome::LockLost(l)) => {
                    uranium_store::elog!("[bc] lock lost to {l:?}: stopping without a push (061-R11-4)");
                    lock_lost = true;
                    break;
                }
                _ => {}
            }
            last_ckpt = Instant::now();
        }
        let request = run.as_ref().and_then(|r| storage::yield_by_other(&r.cfg, &r.run_id));
        if request.is_none() && yield_waiting {
            yield_waiting = false;
        }
        if let Some(req) = request {
            if saved_at == Some(p.step) {
                uranium_store::elog!("[bc] yield requested by {} (priority {}): checkpoint pushed, releasing (061-R14)", req.host, req.priority);
                yielded = true;
                break;
            }
            if !yield_waiting {
                yield_waiting = true;
                uranium_store::elog!("[bc] yield requested by {} (priority {}): checkpoint and release (061-R14)", req.host, req.priority);
                match save(trainer.as_mut(), &p, &mut run, false, &mut sealed_at)? {
                    Some(PushOutcome::Pushed) => {
                        device_log(&run, serde_json::json!({ "event": "checkpoint", "step": p.step }));
                        saved_at = Some(p.step);
                        yielded = true;
                        break;
                    }
                    Some(PushOutcome::LockLost(l)) => {
                        uranium_store::elog!("[bc] lock lost to {l:?}: stopping without a push (061-R11-4)");
                        lock_lost = true;
                        break;
                    }
                    other => uranium_store::elog!("[bc] yield: push {other:?}; keeping the lock and training on, yielding after the next pushed checkpoint (061-R14 ②)"),
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
    let no_update = p.step == start_step && (resume_dir.is_some() || (run.is_some() && !finished));
    let pushed = if no_update || saved_at == Some(p.step) {
        uranium_store::elog!("[bc] step {} is already saved: nothing more to save", p.step);
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
        match pushed {
            Some(PushOutcome::Pushed) => {
                if finished && !yielded {
                    storage::mark_done(&r.cfg, &r.run_id, p.step as u64)?;
                }
                r.release()?
            }
            other => uranium_store::elog!("[bc] final push {other:?}: keeping the lock so the next start recovers it (061-R11-5/6)"),
        }
    }
    if finished {
        uranium_store::elog!("{}", serde_json::json!({ "done": true, "step": p.step, "games": p.data.games }));
    }
    Ok(if yielded {
        BcExit::Yielded
    } else if finished {
        BcExit::Done
    } else {
        BcExit::Stopped
    })
}

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

pub fn run_checkpoint(r: &mut Run, step: u64, sealed_at: &mut Option<usize>, write: impl FnOnce(&Path) -> Result<()>) -> Result<PushOutcome> {
    let out = if *sealed_at == Some(step as usize) {
        uranium_store::elog!("[bc] {} is already written: pushing it again instead of writing it twice (B-8)", storage::ckpt_name(step));
        r.push(step)?
    } else {
        let dir = r.stage_dir(step)?;
        write(&dir)?;
        r.seal(step)?;
        *sealed_at = Some(step as usize);
        r.push(step)?
    };
    uranium_store::elog!("[bc] checkpoint {} {:?}", storage::ckpt_name(step), out);
    if out == PushOutcome::Pushed {
        match r.prune() {
            Ok(0) => {}
            Ok(n) => uranium_store::elog!("[bc] prune: {n} checkpoint folder(s) not deleted, left for the next prune (061-R6)"),
            Err(e) => uranium_store::elog!("[bc] WARNING: prune failed ({e:#}); training goes on (061-R6)"),
        }
    }
    Ok(out)
}
