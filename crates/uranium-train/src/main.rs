//! `uranium-train bc (--dataset <rel> | --shards <dir>)... (--out <ckpt dir> | --run <id>)` (041),
//! `uranium-train bc-epoch`, `uranium-train bc-eval` (041 one-epoch lengths, offline metrics),
//! `uranium-train manifest <dir> [--verify]` (061-R7),
//! `uranium-train rl --run <id> --bc <ckpt>` (042), `uranium-train z-table` (033-R8).

use std::path::PathBuf;

use anyhow::{Context, Result};
use burn::tensor::{Device, DeviceKind};
use clap::{Parser, Subcommand, ValueEnum};
use uranium_model::config::CoreKind;
use uranium_model::{GtrxlCore, LstmCore};
use uranium_train::bc::BcConfig;
use uranium_train::bc_run::{BcArgs, BcOut, BcStage};
use uranium_train::data::ShardSource;
use uranium_train::split::Split;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Backend {
    Vulkan,
    Cuda,
    Flex,
}

#[derive(Parser)]
#[command(about = "Uranium training (spec 041, 042)")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Behaviour cloning over replay shards.
    Bc {
        #[command(flatten)]
        data: DataArgs,
        /// Local checkpoint folder (`step_<n>/`, `last/`).
        #[arg(long, required_unless_present = "run", conflicts_with = "run")]
        out: Option<PathBuf>,
        /// 061 run on the shared drive (`checkpoints/<run>/`): lock, push, and resume from
        /// `LATEST` with the data position on any training PC.
        #[arg(long)]
        run: Option<String>,
        /// Take the run lock from another PC now, whatever its state, with a warning (061-R5).
        /// Use it only after checking that PC stopped: if it still trains, it stops at its next
        /// heartbeat without pushing and loses checkpoints only it had.
        #[arg(long, requires = "run")]
        take_over: bool,
        /// Run queue worker's automatic take-over (061-R5, R13): the lock as the worker saw it
        /// unchanged for `stale_after` (JSON); taken only if it is still exactly that.
        #[arg(long, requires = "run", conflicts_with = "take_over", hide = true)]
        take_over_if: Option<String>,
        #[arg(long, default_value = "config/bc.toml")]
        config: PathBuf,
        #[arg(long, value_enum, default_value = "vulkan")]
        backend: Backend,
        #[arg(long, value_enum, default_value = "lstm")]
        core: CoreArg,
        /// S2 fine-tuning: learning rate x s2_lr_scale (041-R4). Start from --init.
        #[arg(long)]
        s2: bool,
        /// Games to train on (041-R13): s1 = train split of all maps, s2 = train split of the
        /// target maps. Default: s2 with --s2, else s1 (ablation (A) = `--stage s2` without --s2).
        #[arg(long, value_enum)]
        stage: Option<StageArg>,
        /// Stop after this update (smoke runs); the schedule length stays the stage's
        /// (`[steps]` in the config).
        #[arg(long)]
        steps: Option<usize>,
        /// Local mode: continue from a checkpoint folder (weights, optimizer, step, RNG and data
        /// position, 061-R10). A run resumes from `LATEST` by itself.
        #[arg(long, conflicts_with_all = ["init", "run"])]
        resume: Option<PathBuf>,
        /// Start from a checkpoint's weights only: fresh optimizer, schedule and step counter
        /// (S1 -> S2, 041-R4). A checkpoint folder, or a 061 run folder that is finished (`DONE`,
        /// its final checkpoint); a run still training is refused (061-R13).
        #[arg(long)]
        init: Option<PathBuf>,
    },
    /// 041-R13: updates in one epoch of a stage (the trainer's lane walk without the model);
    /// the numbers for `[steps]` in `config/bc.toml`. Reads every shard once.
    BcEpoch {
        #[command(flatten)]
        data: DataArgs,
        #[arg(long, default_value = "config/bc.toml")]
        config: PathBuf,
    },
    /// 060-R6 watchdog: run `bc` / `rl` arguments as a child and restart the same run after an
    /// abnormal exit or a hang (it resumes from its last checkpoint). Exit 0 (done, step limit,
    /// Ctrl+C clean stop, lock lost) and 130 (second Ctrl+C) are not restarted.
    /// Example: `uranium-train supervise -- bc --run bc-s1 --dataset ... --backend cuda`.
    Supervise {
        /// Restarts allowed within --window-minutes; then give up.
        #[arg(long, default_value = "5")]
        max_restarts: usize,
        #[arg(long, default_value = "60")]
        window_minutes: u64,
        /// Delay before a restart (doubles per restart in the window, at most 10 min).
        #[arg(long, default_value = "60")]
        delay_seconds: u64,
        /// No stderr line (and no progress-file change) for this long = hang: kill and restart.
        #[arg(long, default_value = "30")]
        hang_minutes: u64,
        /// File whose changes count as progress (rl: <logs>/train.csv).
        #[arg(long)]
        progress_file: Option<PathBuf>,
        /// Also append the child's stderr and the restart notes to this file.
        #[arg(long)]
        log: Option<PathBuf>,
        /// The training command: `bc ...` or `rl ...`.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true, required = true)]
        args: Vec<String>,
    },
    /// 061-R13/R14 run queue worker: takes the first unfinished, untaken run of the queue and trains
    /// it (`bc --run`, 060-R6 watchdog), then the next; asks a slower PC to yield when nothing is
    /// free. Ctrl+C: checkpoint, push, release, stop.
    Worker {
        /// Queue file (TOML: [defaults] datasets/config, [[run]] id/config/core/stage/init_run/steps).
        #[arg(long)]
        queue: PathBuf,
        #[arg(long, value_enum, default_value = "vulkan")]
        backend: Backend,
        /// Faster PCs get higher numbers; only a higher priority may ask a run to yield.
        #[arg(long, default_value = "0")]
        priority: i32,
        /// Never ask other PCs to yield.
        #[arg(long)]
        no_preempt: bool,
        /// Seconds between looks at the queue when no run is free.
        #[arg(long, default_value = "60")]
        poll_seconds: u64,
        #[arg(long, default_value = "5")]
        max_restarts: usize,
        #[arg(long, default_value = "60")]
        window_minutes: u64,
        #[arg(long, default_value = "60")]
        delay_seconds: u64,
        #[arg(long, default_value = "30")]
        hang_minutes: u64,
        /// Also append children's stderr and the worker's notes here.
        #[arg(long)]
        log: Option<PathBuf>,
    },
    /// 061-R15: updates per host of a run from `runs/<id>/devices.jsonl` (spans and shares).
    Devices {
        #[arg(long)]
        run: String,
    },
    /// Where a BC update's time goes, module by module (forward marks, backward by gradient
    /// cuts), on real S1 batches with a fresh network. The first batch only warms kernels up.
    BcProfile {
        #[command(flatten)]
        data: DataArgs,
        #[arg(long, default_value = "1")]
        batches: usize,
        #[arg(long, value_enum, default_value = "vulkan")]
        backend: Backend,
        #[arg(long, default_value = "config/bc.toml")]
        config: PathBuf,
        #[arg(long, value_enum, default_value = "lstm")]
        core: CoreArg,
    },
    /// 041 offline metrics on a split (default: the target-map test set, 041-AC1): action_type
    /// top-1/top-5, delay MAE, target_loc tile error, selected F1, plus the loss terms.
    BcEval {
        #[command(flatten)]
        data: DataArgs,
        /// Checkpoint folder, or a 061 run folder (its `LATEST`).
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long, value_enum, default_value = "test")]
        split: SplitArg,
        #[arg(long, value_enum, default_value = "vulkan")]
        backend: Backend,
        #[arg(long, value_enum, default_value = "lstm")]
        core: CoreArg,
        /// Score this many games, chosen by a fixed-seed shuffle over the whole split (032-R17 ②:
        /// the core ablation uses `--split val --games 500`; every checkpoint gets the same games).
        /// Default: every game of the split.
        #[arg(long)]
        games: Option<usize>,
        /// Seed of the `--games` shuffle (032-R17 ②, G-D8).
        #[arg(long, default_value_t = 1)]
        seed: u64,
        #[arg(long, default_value = "config/bc.toml")]
        config: PathBuf,
    },
    /// 034-AC3 diagnostics: labels that fail the masks, by action and reason (JSON).
    LabelCheck {
        #[command(flatten)]
        data: DataArgs,
        /// Stop after this many games.
        #[arg(long)]
        games: Option<usize>,
    },
    /// 041-R12: game counts per source, map group and split; per source the share of games and
    /// steps without a privileged summary (E-10) and the result sources (040-R8, A-3). Reads every
    /// shard; no training.
    SplitStats {
        #[command(flatten)]
        data: DataArgs,
    },
    /// 061-R7: writes `<dir>/manifest.json` (relative path -> sha256, bytes, subfolders included) so
    /// training PCs can pull and verify the dataset. `--verify` checks an existing one instead.
    Manifest {
        dir: PathBuf,
        #[arg(long)]
        verify: bool,
    },
    /// 042 RL + league: learner, inference servers, agent server and game dispatch in one process.
    /// Checkpoints go to the shared drive run `<run>` (061); Ctrl+C stops cleanly (042-R10).
    Rl {
        /// Run id (061 checkpoints/<run>/ and the lock).
        #[arg(long)]
        run: String,
        /// BC checkpoint: starting policy and the KL reference (042).
        #[arg(long)]
        bc: PathBuf,
        #[arg(long, default_value = "config/rl.toml")]
        config: PathBuf,
        #[arg(long, value_enum, default_value = "vtrace")]
        algo: uranium_train::rl::learner::Algo,
        #[arg(long, value_enum, default_value = "vulkan")]
        backend: Backend,
        #[arg(long, value_enum, default_value = "lstm")]
        core: CoreArg,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// Take the run lock from another PC now, whatever its state, with a warning (061-R5):
        /// only after checking that PC stopped.
        #[arg(long)]
        take_over: bool,
        /// Local logs: train.csv, games.csv, games.jsonl, sessions.jsonl (042-R2).
        #[arg(long, default_value = "runs/rl")]
        logs: PathBuf,
        /// z table from `uranium-train z-table` (033-R8); without it every game has no z.
        #[arg(long)]
        z_table: Option<PathBuf>,
        /// Stop after this many updates (smoke runs).
        #[arg(long, default_value_t = 0)]
        max_updates: u64,
        /// Stop after this many minutes (smoke runs).
        #[arg(long, default_value_t = 0)]
        max_minutes: u64,
        /// 042-R12: clean stop once the run's agent steps (counted across resumes) reach this;
        /// 042-AC1 / 033-AC2 use 10,000,000. 0 = no limit.
        #[arg(long, default_value_t = 0)]
        max_agent_steps: u64,
        /// Checkpoint interval in minutes (061-R4).
        #[arg(long, default_value_t = 30)]
        checkpoint_minutes: u64,
        /// Inference batch cap and wait (TBD(M6), 020-R11).
        #[arg(long, default_value_t = 32)]
        max_batch: usize,
        #[arg(long, default_value_t = 5)]
        max_wait_ms: u64,
        /// 042-AC1 small setting: Main vs the standard T0 bot of this race on Fighting Spirit only.
        #[arg(long, value_enum)]
        ac1_race: Option<RaceArg>,
    },
    /// 033-R8: z table (one JSON line per game: build order + unit composition) from the Terran
    /// views of target-map replays in a shard folder.
    ZTable {
        #[arg(long)]
        shards: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// Include every map, not only the target maps (040-R15 families).
        #[arg(long)]
        all_maps: bool,
    },
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum RaceArg {
    Terran,
    Protoss,
    Zerg,
}

/// Shard folders to read, subfolders included, in the order given (041-R9).
#[derive(clap::Args)]
struct DataArgs {
    /// Dataset under `<URANIUM_SHARED_DIR>/datasets/` with a manifest (e.g. `replays/shards/stardata`):
    /// pulled into the cache, or verified in place with URANIUM_SHARED_LOCAL=1 (061-R11-1, R12).
    #[arg(long)]
    dataset: Vec<PathBuf>,
    /// Shard folder read as is, without a manifest check (samples, tests).
    #[arg(long)]
    shards: Vec<PathBuf>,
}

impl DataArgs {
    fn dirs(&self) -> Result<Vec<PathBuf>> {
        anyhow::ensure!(!self.dataset.is_empty() || !self.shards.is_empty(), "pass --dataset or --shards");
        let mut dirs = Vec::new();
        if !self.dataset.is_empty() {
            let cfg = uranium_train::storage::StorageConfig::from_env()?;
            // Sorted, so the game order (041-R13) and a run's data position (061-R10) do not
            // depend on the order the datasets were named in.
            let mut names = self.dataset.clone();
            names.sort();
            names.dedup();
            for d in &names {
                dirs.push(uranium_train::storage::open_dataset(&cfg, d)?);
            }
        }
        dirs.extend(self.shards.iter().cloned());
        Ok(dirs)
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum StageArg {
    S1,
    S2,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum SplitArg {
    Train,
    Val,
    Test,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CoreArg {
    Lstm,
    Gtrxl,
}

fn device(b: Backend) -> Result<Device> {
    // 041-R15: autotune + fusion are compiled in; the autotune cache must sit in the host's fixed
    // URANIUM_AUTOTUNE_CACHE_DIR before the first GPU device is made (training: required).
    if !matches!(b, Backend::Flex) {
        uranium_model::runtime::init_gpu_runtime(true)?;
    }
    Ok(match b {
        Backend::Vulkan => Device::vulkan(DeviceKind::DefaultDevice),
        #[cfg(feature = "cuda")]
        Backend::Cuda => Device::cuda(0),
        #[cfg(not(feature = "cuda"))]
        Backend::Cuda => anyhow::bail!("rebuild with --features cuda on a CUDA PC"),
        Backend::Flex => Device::flex(),
    })
}

fn rl_backend(b: Backend) -> uranium_agent::infer::InferBackend {
    match b {
        Backend::Vulkan => uranium_agent::infer::InferBackend::Vulkan,
        Backend::Cuda => uranium_agent::infer::InferBackend::Cuda,
        Backend::Flex => uranium_agent::infer::InferBackend::Flex,
    }
}

fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    match Cli::parse().cmd {
        Cmd::Rl { run, bc, config, algo, backend, core, seed, take_over, logs, z_table, max_updates, max_minutes, max_agent_steps, checkpoint_minutes, max_batch, max_wait_ms, ac1_race } => {
            let args = uranium_train::rl::run::RlArgs {
                run,
                bc,
                config,
                algo,
                backend: rl_backend(backend),
                seed,
                take_over,
                log_dir: logs,
                z_table,
                max_updates,
                max_minutes,
                max_agent_steps,
                checkpoint_minutes,
                max_batch,
                max_wait_ms,
                fixed_t0: ac1_race.map(|r| match r {
                    RaceArg::Terran => uranium_proto::Race::Terran,
                    RaceArg::Protoss => uranium_proto::Race::Protoss,
                    RaceArg::Zerg => uranium_proto::Race::Zerg,
                }),
            };
            let dev = device(backend)?;
            let res = match core {
                CoreArg::Lstm => uranium_train::rl::run::run::<LstmCore>(args, dev),
                CoreArg::Gtrxl => uranium_train::rl::run::run::<GtrxlCore>(args, dev),
            };
            // 061-R5: a lock that could not be taken ends with 75 (the watchdog does not restart it).
            match res {
                Err(e) if e.downcast_ref::<uranium_train::storage::LockBusy>().is_some() => {
                    uranium_train::elog!("{e:#}");
                    std::process::exit(uranium_train::bc_run::EXIT_BUSY)
                }
                other => other,
            }
        }
        Cmd::ZTable { shards, out, all_maps } => {
            use std::io::Write as _;
            let families = uranium_env::maps::MapFamilies::load()?;
            let mut src = ShardSource::new(&shards)?;
            let mut f = std::io::BufWriter::new(std::fs::File::create(&out)?);
            let (mut seen, mut kept) = (0usize, 0usize);
            while let Some(g) = uranium_train::data::StepSource::next_game(&mut src)? {
                seen += 1;
                let me = g.game_info.players.iter().find(|p| p.id == g.game_info.self_player);
                if me.map(|p| p.race) != Some(uranium_proto::Race::Terran) {
                    continue;
                }
                if !all_maps && families.classify(&g.game_info.map_name, "") == "other" {
                    continue;
                }
                let z = uranium_train::data::replay_z(&g);
                if z.build.is_empty() {
                    continue;
                }
                let e = uranium_train::rl::run::ZEntry { build: z.build, units: z.units };
                writeln!(f, "{}", serde_json::to_string(&e)?)?;
                kept += 1;
            }
            f.flush()?;
            println!("{}", serde_json::json!({ "games": seen, "z_entries": kept, "out": out }));
            Ok(())
        }
        Cmd::Manifest { dir, verify } => {
            let m = if verify {
                uranium_train::storage::verify_dir(&dir)?
            } else {
                let m = uranium_train::storage::Manifest::build(&dir)?;
                m.write(&dir)?;
                m
            };
            let bytes: u64 = m.files.values().map(|f| f.bytes).sum();
            println!(
                "{}",
                serde_json::json!({ "dir": dir, "files": m.files.len(), "bytes": bytes, "verified": verify })
            );
            Ok(())
        }
        Cmd::SplitStats { data } => {
            let src = ShardSource::from_dirs(&data.dirs()?)?;
            let scan = uranium_train::data::scan_cached(&src)?;
            println!("{}", serde_json::to_string_pretty(&uranium_train::split::split_stats(scan.iter().flatten()))?);
            Ok(())
        }
        Cmd::LabelCheck { data, games } => {
            let mut src = ShardSource::from_dirs(&data.dirs()?)?;
            // B-2: every map precompute is there before the first game.
            let maps = uranium_train::data::MapStore::from_env();
            uranium_train::data::check_maps(&src, None, &maps)?;
            let (mut st, mut n) = (uranium_train::data::LabelStats::default(), 0usize);
            while let Some(g) = uranium_train::data::StepSource::next_game(&mut src)? {
                uranium_train::data::label_check(&g, &mut st, &maps)?;
                n += 1;
                if games.is_some_and(|m| n >= m) {
                    break;
                }
            }
            let invalid: usize = st.fails.values().sum();
            let mut rows: Vec<_> = st
                .fails
                .iter()
                .map(|((a, w), c)| (a.clone(), *w, *c, st.totals[a]))
                .collect();
            rows.sort_by(|x, y| y.2.cmp(&x.2));
            let mut detail: Vec<_> = st.cap_detail.into_iter().collect();
            detail.sort_by(|x, y| y.1.cmp(&x.1));
            let mut type_detail: Vec<_> = st.type_detail.into_iter().collect();
            type_detail.sort_by(|x, y| y.1.cmp(&x.1));
            let labels: usize = st.totals.values().sum();
            println!(
                "{}",
                serde_json::to_string_pretty(
                    &serde_json::json!({ "games": n, "labels": labels, "invalid": invalid, "by_action_reason": rows, "select_cap_types": detail, "type_mask_prereq": type_detail })
                )?
            );
            Ok(())
        }
        Cmd::Bc { data, out, run, take_over, take_over_if, config, backend, core, s2, stage, steps, resume, init } => {
            // 042-R10 / 060-R6: before the dataset pull and the model load, so an early SIGINT is kept.
            let stop = uranium_train::install_stop_handler("bc")?;
            let stage = match (s2, stage.unwrap_or(if s2 { StageArg::S2 } else { StageArg::S1 })) {
                (true, _) => BcStage::S2,
                (false, StageArg::S2) => BcStage::AblationA,
                (false, StageArg::S1) => BcStage::S1,
            };
            let out = match (out, run) {
                (Some(o), _) => BcOut::Local(o),
                (None, Some(id)) => {
                    let take = match (take_over, take_over_if) {
                        (_, Some(seen)) => uranium_train::storage::TakeOver::IfUnchanged(serde_json::from_str(&seen).context("--take-over-if")?),
                        (true, None) => uranium_train::storage::TakeOver::Force,
                        (false, None) => uranium_train::storage::TakeOver::No,
                    };
                    BcOut::Run { id, take }
                }
                (None, None) => unreachable!("clap requires --out or --run"),
            };
            let kind = match core {
                CoreArg::Lstm => CoreKind::Lstm,
                CoreArg::Gtrxl => CoreKind::Gtrxl,
            };
            let backend_name = format!("{backend:?}").to_lowercase();
            let dirs = match data.dirs() {
                Ok(d) => d,
                // Ctrl+C during the dataset pull: no lock yet, a clean stop (042-R10).
                Err(e) if e.downcast_ref::<uranium_train::Interrupted>().is_some() => {
                    uranium_train::elog!("[bc] {e:#}");
                    return Ok(());
                }
                Err(e) => return Err(e),
            };
            let args = BcArgs { dirs, out, config, stage, core: kind, stop_at: steps, resume, init, backend: backend_name, stop };
            let dev = device(backend)?;
            let res = match core {
                CoreArg::Lstm => uranium_train::bc_run::run::<LstmCore>(args, dev),
                CoreArg::Gtrxl => uranium_train::bc_run::run::<GtrxlCore>(args, dev),
            };
            // 061-R13/R14 exit codes for the run queue worker.
            match res {
                Ok(uranium_train::bc_run::BcExit::Yielded) => std::process::exit(uranium_train::bc_run::EXIT_YIELDED),
                Ok(_) => Ok(()),
                Err(e) if e.downcast_ref::<uranium_train::storage::LockBusy>().is_some() => {
                    uranium_train::elog!("{e:#}");
                    std::process::exit(uranium_train::bc_run::EXIT_BUSY)
                }
                Err(e) => Err(e),
            }
        }
        Cmd::BcEpoch { data, config } => {
            let cfg = BcConfig::load(&config)?;
            let dirs = data.dirs()?;
            let out = uranium_train::bc_eval::epoch_table(&dirs, &cfg)?;
            println!("{}", serde_json::to_string_pretty(&out)?);
            Ok(())
        }
        Cmd::Supervise { max_restarts, window_minutes, delay_seconds, hang_minutes, progress_file, log, args } => {
            anyhow::ensure!(matches!(args.first().map(String::as_str), Some("bc" | "rl")), "supervise runs `bc ...` or `rl ...`");
            let code = uranium_train::supervise::run(uranium_train::supervise::SuperviseArgs {
                max_restarts,
                window: std::time::Duration::from_secs(window_minutes * 60),
                delay: std::time::Duration::from_secs(delay_seconds),
                hang: std::time::Duration::from_secs(hang_minutes * 60),
                progress_file,
                log,
                args,
            })?;
            std::process::exit(code);
        }
        Cmd::Worker { queue, backend, priority, no_preempt, poll_seconds, max_restarts, window_minutes, delay_seconds, hang_minutes, log } => {
            uranium_train::worker::run(uranium_train::worker::WorkerArgs {
                queue,
                backend: format!("{backend:?}").to_lowercase(),
                priority,
                preempt: !no_preempt,
                poll: std::time::Duration::from_secs(poll_seconds),
                status_every: std::time::Duration::from_secs(60),
                supervise: uranium_train::supervise::SuperviseArgs {
                    max_restarts,
                    window: std::time::Duration::from_secs(window_minutes * 60),
                    delay: std::time::Duration::from_secs(delay_seconds),
                    hang: std::time::Duration::from_secs(hang_minutes * 60),
                    progress_file: None,
                    log,
                    args: Vec::new(),
                },
            })
        }
        Cmd::Devices { run } => {
            let cfg = uranium_train::storage::StorageConfig::from_env()?;
            let spans = uranium_train::storage::device_spans(&uranium_train::storage::read_devices(&cfg, &run)?);
            let total: u64 = spans.iter().map(|s| s.to - s.from).sum();
            let mut by_host: std::collections::BTreeMap<String, u64> = std::collections::BTreeMap::new();
            for s in &spans {
                *by_host.entry(s.host.clone()).or_default() += s.to - s.from;
            }
            let hosts: serde_json::Map<String, serde_json::Value> = by_host
                .into_iter()
                .map(|(h, n)| (h, serde_json::json!({ "updates": n, "share": if total > 0 { n as f64 / total as f64 } else { 0.0 } })))
                .collect();
            println!("{}", serde_json::to_string_pretty(&serde_json::json!({ "run": run, "updates": total, "hosts": hosts, "spans": spans }))?);
            Ok(())
        }
        Cmd::BcProfile { data, batches, backend, config, core } => {
            let cfg = BcConfig::load(&config)?;
            let dev = device(backend)?;
            let rep = match core {
                CoreArg::Lstm => uranium_train::bc_prof::profile::<LstmCore>(&data.dirs()?, &cfg, batches, dev)?,
                CoreArg::Gtrxl => uranium_train::bc_prof::profile::<GtrxlCore>(&data.dirs()?, &cfg, batches, dev)?,
            };
            println!("{}", serde_json::to_string_pretty(&rep)?);
            Ok(())
        }
        Cmd::BcEval { data, checkpoint, split, backend, core, games, seed, config } => {
            let cfg = BcConfig::load(&config)?;
            let split = match split {
                SplitArg::Train => Split::Train,
                SplitArg::Val => Split::Val,
                SplitArg::Test => Split::Test,
            };
            let ckpt = uranium_train::bc_run::resolve_checkpoint(&checkpoint)?;
            let dev = device(backend)?;
            let dirs = data.dirs()?;
            let sel = uranium_train::bc_eval::EvalSelection { split, games, seed };
            let report = match core {
                CoreArg::Lstm => uranium_train::bc_eval::evaluate::<LstmCore>(&dirs, &ckpt, sel, &cfg, dev)?,
                CoreArg::Gtrxl => uranium_train::bc_eval::evaluate::<GtrxlCore>(&dirs, &ckpt, sel, &cfg, dev)?,
            };
            println!("{}", serde_json::to_string_pretty(&report)?);
            Ok(())
        }
    }
}
