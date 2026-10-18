//! `uranium-train bc` run loop (041): stage selection and schedule length (041-R13 one epoch per
//! stage), prefetching data thread, checkpoints by update count and wall time, Ctrl+C, and either
//! a local `--out` folder or a 061 run on the shared drive (lock, heartbeat, push, resume on any
//! host from `LATEST` with the data position, 061-R3/R5/R10).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use burn::tensor::Device;
use uranium_model::collate::input_dims;
use uranium_model::config::{CoreKind, UraniumNetConfig};
use uranium_model::{CoreInit, UraniumNet};

use crate::bc::{BcConfig, Prefetch, Trainer};
use crate::ckpt;
use crate::data::ShardSource;
use crate::storage::{LockStatus, PushOutcome, Run, StorageConfig};

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
    Run { id: String, take_over: bool },
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
    /// Weights only (S1 -> S2, 041-R4): a checkpoint folder or a 061 run folder with `LATEST`.
    pub init: Option<PathBuf>,
}

const STOP_NONE: usize = 0;
const STOP_CLEAN: usize = 1;

/// A checkpoint folder, or the one `LATEST` names in a 061 run folder.
pub fn resolve_checkpoint(dir: &Path) -> Result<PathBuf> {
    let latest = dir.join("LATEST");
    if latest.is_file() {
        let name = std::fs::read_to_string(&latest)?.trim().to_string();
        return Ok(dir.join(name));
    }
    Ok(dir.to_path_buf())
}

pub fn run<C: CoreInit>(args: BcArgs, dev: Device) -> Result<()> {
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

    // 061-R3 / R5: a run takes the lock and resumes from LATEST (or an unpushed local checkpoint).
    let mut run = match &args.out {
        BcOut::Run { id, take_over } => Some(Run::acquire(StorageConfig::from_env()?, id, *take_over)?),
        BcOut::Local(_) => None,
    };
    let resume_dir = match &mut run {
        Some(r) => r.begin()?.map(|res| res.dir),
        None => args.resume.clone(),
    };
    let (net, net_cfg) = match (&resume_dir, &args.init) {
        (Some(d), _) => load(d)?,
        (None, Some(i)) => load(&resolve_checkpoint(i)?)?,
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
    if args.stage == BcStage::S2 {
        t.lr_scale = t.cfg.s2_lr_scale;
    }
    let mut src = ShardSource::from_dirs(&args.dirs)?.select(args.stage.selection()).shuffled(cfg.data.seed);
    if let Some(dir) = &resume_dir {
        if t.resume(dir)? {
            // 061-R10: continue the data where the checkpoint was taken.
            let data = t.data.clone();
            t.feeder.restore(&data, &mut src)?;
            eprintln!("{}", serde_json::json!({ "resumed": dir, "step": t.step, "games": data.games, "source": data.source }));
        } else {
            eprintln!("{}: no training state (older checkpoint); weights only, fresh optimizer", dir.display());
        }
    }
    eprintln!("{}", serde_json::json!({ "stage": format!("{:?}", args.stage), "total_steps": total, "stop_at": stop_at, "lr_scale": t.lr_scale }));

    // 042-R10 style: first Ctrl+C = save and stop, second = exit now.
    let stop = Arc::new(AtomicUsize::new(STOP_NONE));
    {
        let stop = stop.clone();
        ctrlc::set_handler(move || {
            if stop.fetch_add(1, Ordering::SeqCst) >= STOP_CLEAN {
                eprintln!("[bc] second Ctrl+C: exiting without saving");
                std::process::exit(130);
            }
            eprintln!("[bc] Ctrl+C: saving a checkpoint after this update");
        })?;
    }

    // Nothing to save when no update happens (resumed at or past the stop point).
    let start_step = t.step;
    let feeder = std::mem::replace(&mut t.feeder, crate::bc::Feeder::new(&cfg));
    let data = Prefetch::start(feeder, src, 2);
    let (mut last_ckpt, mut last_beat) = (Instant::now(), Instant::now());
    let mut lock_lost = false;
    let mut exhausted = false;
    while t.step < stop_at && stop.load(Ordering::SeqCst) < STOP_CLEAN {
        let Some(batch) = data.next()? else {
            exhausted = true;
            eprintln!("data exhausted after {} updates (end of the epoch)", t.step);
            break;
        };
        let started = Instant::now();
        let l = t.train_batch(batch)?;
        eprintln!(
            "{}",
            serde_json::json!({ "step": t.step, "loss": l, "games": t.games(), "invalid_labels": t.invalid_labels(), "secs": (started.elapsed().as_secs_f64() * 100.0).round() / 100.0 })
        );
        if let Some(r) = &mut run
            && last_beat.elapsed() >= Duration::from_secs(60)
        {
            last_beat = Instant::now();
            match r.heartbeat() {
                LockStatus::Held => {}
                LockStatus::Lost(l) => {
                    eprintln!("[bc] lock lost to {l:?}: stopping without a push (061-R5)");
                    lock_lost = true;
                    break;
                }
                LockStatus::Unreachable(e) => eprintln!("[bc] heartbeat: share unreachable ({e}), continuing (061-R11-3)"),
            }
        }
        if t.step % every == 0 || (!every_time.is_zero() && last_ckpt.elapsed() >= every_time) {
            save(&t, &args, &mut run, false)?;
            last_ckpt = Instant::now();
        }
    }
    if lock_lost {
        return Ok(());
    }
    let pushed = if t.step == start_step && resume_dir.is_some() {
        eprintln!("[bc] no update since the checkpoint of step {}: nothing to save", t.step);
        Some(PushOutcome::Pushed)
    } else {
        save(&t, &args, &mut run, true)?
    };
    if let Some(r) = run {
        // 061-R11-5/6: keep the lock when the final push did not go through.
        match pushed {
            Some(PushOutcome::Pushed) => r.release()?,
            other => eprintln!("[bc] final push {other:?}: keeping the lock so the next start recovers it (061-R11-5/6)"),
        }
    }
    if exhausted || t.step >= total {
        eprintln!("{}", serde_json::json!({ "done": true, "step": t.step, "games": t.games() }));
    }
    Ok(())
}

/// Local: `step_<n>/` (or `last/` when `last`). Run: stage → seal → push (061-R4), then prune.
fn save<C: CoreInit>(t: &Trainer<C>, args: &BcArgs, run: &mut Option<Run>, last: bool) -> Result<Option<PushOutcome>> {
    match (&args.out, run) {
        (BcOut::Local(out), _) => {
            let dir = if last { out.join("last") } else { out.join(format!("step_{:08}", t.step)) };
            t.save(&dir)?;
            ckpt::copy_configs(&dir, &[&args.config])?;
            Ok(None)
        }
        (BcOut::Run { .. }, Some(r)) => {
            let step = t.step as u64;
            let dir = r.stage_dir(step)?;
            t.save(&dir)?;
            ckpt::copy_configs(&dir, &[&args.config])?;
            r.seal(step)?;
            let out = r.push(step)?;
            eprintln!("[bc] checkpoint {} {:?}", crate::storage::ckpt_name(step), out);
            if out == PushOutcome::Pushed {
                r.prune()?;
            }
            Ok(Some(out))
        }
        (BcOut::Run { .. }, None) => unreachable!("run mode holds a Run"),
    }
}
