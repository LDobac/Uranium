//! `uranium-train`: host data commands (`bc-epoch`, `label-check`, `split-stats`, `manifest`,
//! `devices`, `z-table`, `feed-digest`), the 060-R6 watchdog (`supervise`) and the 061-R13 run
//! queue worker, which run the PyTorch trainer (`python -m uranium bc | rl`, AMB-000-4).

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use uranium_train::bc::{BcConfig, BcStage};
use uranium_train::data::ShardSource;

/// Device labels of the PyTorch training children (001-R10).
#[derive(Clone, Copy, Debug, ValueEnum)]
enum ChildBackend {
    Cuda,
    Rocm,
    Cpu,
}

#[derive(Parser)]
#[command(about = "Uranium training (spec 041, 042)")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// 041-R13: updates in one epoch of a stage (the trainer's lane walk without the model);
    /// the numbers for `[steps]` in `config/bc.toml`. Reads every shard once.
    BcEpoch {
        #[command(flatten)]
        data: DataArgs,
        #[arg(long, default_value = "config/bc.toml")]
        config: PathBuf,
    },
    /// AMB-000-4 check: SHA-256 of every lane group of the first BC batches (`Batch::group`, the
    /// host collate). The Python side hashes the `uranium._core` arrays the same way, so the two
    /// can be compared. One JSON line per batch.
    #[command(hide = true)]
    FeedDigest {
        #[command(flatten)]
        data: DataArgs,
        #[arg(long, default_value = "config/bc.toml")]
        config: PathBuf,
        #[arg(long, value_enum, default_value = "s1")]
        stage: StageArg,
        #[arg(long, default_value = "3")]
        batches: usize,
        /// Lanes per group.
        #[arg(long, default_value = "1")]
        group: usize,
        #[arg(long, default_value = "12")]
        max_select: usize,
        /// 041-R18: packed batches.
        #[arg(long)]
        pack: bool,
        /// Test shards without map precomputes (`uranium_data::testkit`).
        #[arg(long)]
        synthetic_maps: bool,
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
    /// it (`URANIUM_PYTHON -m uranium bc --run`, 060-R6 watchdog), then the next; asks a slower PC
    /// to yield when nothing is free. Ctrl+C: checkpoint, push, release, stop.
    Worker {
        /// Queue file (TOML: [defaults] datasets/config, [[run]] id/config/core/stage/init_run/steps).
        #[arg(long)]
        queue: PathBuf,
        /// Device of the PyTorch children (001-R10).
        #[arg(long, value_enum)]
        backend: ChildBackend,
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
    /// Before hashing, every `shard_NNNNN.bin.zst` below `dir` must decode completely (040-R17).
    Manifest {
        dir: PathBuf,
        #[arg(long)]
        verify: bool,
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

/// Shard folders to read, subfolders included (041-R9); files are read in canonical key order
/// whatever the folder order or spelling (041-R13).
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
    /// 041-R9: pulled or verified datasets, then the shard folders ([`uranium_train::storage::dataset_dirs`]).
    fn dirs(&self) -> Result<Vec<PathBuf>> {
        uranium_train::storage::dataset_dirs(&self.dataset, &self.shards)
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum StageArg {
    S1,
    S2,
}




fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    match Cli::parse().cmd {
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
                let e = uranium_league::run::ZEntry { build: z.build, units: z.units };
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
                // 040-R17 (R-5): a shard that does not decode must fail here, not be hashed as valid.
                // `validate_tree` caps the workers at 4 (each holds one decoded shard).
                let jobs = std::thread::available_parallelism().map_or(4, |n| n.get());
                let (shards, games) = uranium_replay::shard::validate_tree(&dir, jobs)?;
                if shards > 0 {
                    eprintln!("[manifest] {shards} shards decode ({games} games, 040-R17)");
                }
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
            // 031-AC3 `unknown` steps (labeler version 2) are weight 0 but not invalid labels
            // (040-R19, 040-AC7): reported on their own.
            let unknown = st.totals.get("Unknown").copied().unwrap_or(0);
            let invalid: usize = st.fails.iter().filter(|((_, w), _)| *w != uranium_train::data::UNKNOWN_REASON).map(|(_, c)| c).sum();
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
            let mut cause_detail: Vec<_> = st.cause_detail.into_iter().collect();
            cause_detail.sort_by(|x, y| y.1.cmp(&x.1));
            let labels: usize = st.totals.values().sum::<usize>() - unknown;
            println!(
                "{}",
                serde_json::to_string_pretty(
                    &serde_json::json!({ "games": n, "labels": labels, "invalid": invalid, "unknown": unknown, "by_action_reason": rows, "select_cap_types": detail, "type_mask_prereq": type_detail, "target_causes": cause_detail, "per_game": st.per_game })
                )?
            );
            Ok(())
        }
        Cmd::BcEpoch { data, config } => {
            let cfg = BcConfig::load(&config)?;
            let dirs = data.dirs()?;
            let out = uranium_train::bc::epoch_table(&dirs, &cfg)?;
            println!("{}", serde_json::to_string_pretty(&out)?);
            Ok(())
        }
        Cmd::FeedDigest { data, config, stage, batches, group, max_select, pack, synthetic_maps } => {
            let stage = match stage {
                StageArg::S1 => BcStage::S1,
                StageArg::S2 => BcStage::S2,
            };
            let cfg = BcConfig::load(&config)?.for_stage(stage);
            let src = ShardSource::from_dirs(&data.dirs()?)?.select(stage.selection()).shuffled(cfg.data.seed);
            let mut feeder = uranium_train::bc::Feeder::new(&cfg);
            if synthetic_maps {
                feeder.set_maps(std::sync::Arc::new(uranium_train::data::MapStore::synthetic()));
            }
            feeder.set_pack(pack);
            let mut batches_left = batches;
            let mut p = uranium_train::bc::Prefetch::start(feeder, src, 0);
            let mut k = 0;
            while batches_left > 0 {
                let Some(b) = p.next()? else { break };
                let lanes = b.lanes();
                let groups: Vec<String> = (0..lanes).step_by(group.max(1)).map(|s| b.group(s..(s + group.max(1)).min(lanes), cfg.seq_len, cfg.pad_entities, max_select).digest()).collect();
                println!("{}", serde_json::json!({ "batch": k, "valid": b.valid(), "value_valid": b.value_valid(), "groups": groups, "state": b.state }));
                k += 1;
                batches_left -= 1;
            }
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
    }
}
