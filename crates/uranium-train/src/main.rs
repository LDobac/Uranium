//! `uranium-train bc (--dataset <rel> | --shards <dir>)... --out <ckpt dir>` (041),
//! `uranium-train manifest <dir> [--verify]` (061-R7),
//! `uranium-train rl --run <id> --bc <ckpt>` (042), `uranium-train z-table` (033-R8).

use std::path::PathBuf;

use anyhow::Result;
use burn::tensor::{Device, DeviceKind};
use clap::{Parser, Subcommand, ValueEnum};
use uranium_model::collate::input_dims;
use uranium_model::config::{CoreKind, UraniumNetConfig};
use uranium_model::{GtrxlCore, LstmCore, UraniumNet};
use uranium_train::bc::{BcConfig, Trainer};
use uranium_train::data::ShardSource;
use uranium_train::split::{Selection, Split, is_target_map, split_of};

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
        #[arg(long)]
        out: PathBuf,
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
        /// Stop after this many updates (default: config total_steps).
        #[arg(long)]
        steps: Option<usize>,
        /// Continue from a checkpoint folder: weights, optimizer state, step and RNG (061-R10).
        /// The shard order starts over.
        #[arg(long, conflicts_with = "init")]
        resume: Option<PathBuf>,
        /// Start from a checkpoint's weights only: fresh optimizer, schedule and step counter
        /// (S1 -> S2, 041-R4; the S1 checkpoint folder is left as is).
        #[arg(long)]
        init: Option<PathBuf>,
    },
    /// 034-AC3 diagnostics: labels that fail the masks, by action and reason (JSON).
    LabelCheck {
        #[command(flatten)]
        data: DataArgs,
        /// Stop after this many games.
        #[arg(long)]
        games: Option<usize>,
    },
    /// 041-R12: game counts per source, map group and split (reads every shard; no training).
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
        /// Take the run lock from another PC (061-R5).
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
            for d in &self.dataset {
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
enum CoreArg {
    Lstm,
    Gtrxl,
}

fn device(b: Backend) -> Result<Device> {
    Ok(match b {
        Backend::Vulkan => Device::vulkan(DeviceKind::DefaultDevice),
        #[cfg(feature = "cuda")]
        Backend::Cuda => Device::cuda(0),
        #[cfg(not(feature = "cuda"))]
        Backend::Cuda => anyhow::bail!("rebuild with --features cuda on a CUDA PC"),
        Backend::Flex => Device::flex(),
    })
}

#[allow(clippy::too_many_arguments)]
fn run<C: uranium_model::CoreInit>(
    kind: CoreKind,
    mut src: ShardSource,
    out: PathBuf,
    cfg: BcConfig,
    dev: Device,
    s2: bool,
    steps: Option<usize>,
    resume: Option<PathBuf>,
    init: Option<PathBuf>,
) -> Result<()> {
    let ad = if cfg.grad_checkpointing {
        dev.autodiff().gradient_checkpointing()
    } else {
        dev.autodiff()
    };
    let cols = uranium_model::batch::ActionColumns::from_table(
        uranium_env::features::extract::action_table(),
    )?;
    let (net, net_cfg) = match resume.as_ref().or(init.as_ref()) {
        Some(dir) => {
            let (net, meta) = uranium_model::checkpoint::load::<C>(
                dir,
                &cols,
                uranium_env::features::FEATURE_SCHEMA_VERSION,
                uranium_env::actions::ACTION_SCHEMA_VERSION,
                &ad,
            )?;
            (net, meta.net_config()?)
        }
        None => {
            let base = UraniumNetConfig::new();
            let net_cfg = base
                .clone()
                .with_input(input_dims())
                .with_core(base.core.clone().with_kind(kind));
            (UraniumNet::<C>::new(&net_cfg, &cols, &ad), net_cfg)
        }
    };
    let total = steps.unwrap_or(cfg.total_steps);
    let every = cfg.data.checkpoint_every.max(1);
    let mut t = Trainer::new(net, net_cfg, cfg, ad);
    if s2 {
        t.lr_scale = t.cfg.s2_lr_scale;
    }
    if let Some(dir) = &resume
        && !t.resume(dir)?
    {
        eprintln!(
            "{}: no training state (older checkpoint); weights only, fresh optimizer",
            dir.display()
        );
    }
    while t.step < total {
        let Some(l) = t.train_step(&mut src)? else {
            eprintln!("data exhausted after {} updates", t.step);
            break;
        };
        eprintln!(
            "{}",
            serde_json::json!({ "step": t.step, "loss": l, "games": t.games, "invalid_labels": t.invalid_labels })
        );
        if t.step % every == 0 {
            t.save(&out.join(format!("step_{:08}", t.step)))?;
        }
    }
    t.save(&out.join("last"))
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
            match core {
                CoreArg::Lstm => uranium_train::rl::run::run::<LstmCore>(args, dev),
                CoreArg::Gtrxl => uranium_train::rl::run::run::<GtrxlCore>(args, dev),
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
            // (src, map group, split) -> games
            let mut counts = std::collections::BTreeMap::<(String, &str, Split), usize>::new();
            let mut src = ShardSource::from_dirs(&data.dirs()?)?;
            while let Some(g) = uranium_train::data::StepSource::next_game(&mut src)? {
                let group = if is_target_map(&g.meta) { "target" } else { "other" };
                *counts.entry((g.meta.src.clone(), group, split_of(&g.meta))).or_default() += 1;
            }
            let rows: Vec<_> = counts.into_iter().map(|((s, m, sp), n)| serde_json::json!({ "src": s, "maps": m, "split": sp, "games": n })).collect();
            println!("{}", serde_json::to_string_pretty(&rows)?);
            Ok(())
        }
        Cmd::LabelCheck { data, games } => {
            let mut src = ShardSource::from_dirs(&data.dirs()?)?;
            let (mut st, mut n) = (uranium_train::data::LabelStats::default(), 0usize);
            while let Some(g) = uranium_train::data::StepSource::next_game(&mut src)? {
                uranium_train::data::label_check(&g, &mut st)?;
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
        Cmd::Bc {
            data,
            out,
            config,
            backend,
            core,
            s2,
            stage,
            steps,
            resume,
            init,
        } => {
            let cfg = BcConfig::load(&config)?;
            let dev = device(backend)?;
            let sel = match stage.unwrap_or(if s2 { StageArg::S2 } else { StageArg::S1 }) {
                StageArg::S1 => Selection::S1,
                StageArg::S2 => Selection::S2,
            };
            let shards = ShardSource::from_dirs(&data.dirs()?)?.select(sel).shuffled(cfg.data.seed);
            match core {
                CoreArg::Lstm => {
                    run::<LstmCore>(CoreKind::Lstm, shards, out, cfg, dev, s2, steps, resume, init)
                }
                CoreArg::Gtrxl => {
                    run::<GtrxlCore>(CoreKind::Gtrxl, shards, out, cfg, dev, s2, steps, resume, init)
                }
            }
        }
    }
}
