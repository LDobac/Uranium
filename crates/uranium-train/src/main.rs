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
        #[command(flatten)]
        groups: GroupArgs,
        #[arg(long, default_value = "config/bc.toml")]
        config: PathBuf,
    },
    /// 041-R26 ② (AMB-041-15): `[data.subsample] fraction_by_race` for a mix config's data seed and
    /// `keep_sources` (the teacher): the human S2 games per enemy race kept so the teacher is
    /// `--teacher-share` of the epoch's steps and every race keeps its share of the S2 human data;
    /// then the S2 epoch with those fractions (`[steps] s2` and the `s2_mix` checks of `bc-epoch`).
    /// 044-R5 ③ (041-R31): `--share teacher=0.45,distill=0.2` instead gives three shares — the
    /// teacher (kept whole), each named source (`[data.subsample] fraction_by_source`) and the human
    /// rest (per race, 041-R26) — each judged ± 3%p. 044-R8 / 022-R12 (teacher round t1, user
    /// decision 2026-11-14: `--teacher-share 0.65`, no distill): `--teacher-races Z=0.5,P=0.3,T=0.2`
    /// also judges the enemy races inside the teacher (+ dagger) steps ± 3%p; every teacher game is
    /// kept and a race below its band is reported `short` with the 022 teacher botgen games to add.
    MixFractions {
        #[command(flatten)]
        data: DataArgs,
        #[command(flatten)]
        groups: GroupArgs,
        #[arg(long)]
        config: PathBuf,
        #[arg(long, default_value = "0.5")]
        teacher_share: f64,
        /// 044-R5 ③: `teacher=<a>,<source>=<b>[,...]` (source = `GameMeta.src`, e.g. `distill`);
        /// the human share is 1 - the others. Not with --teacher-share.
        #[arg(long, value_parser = parse_shares, conflicts_with = "teacher_share")]
        share: Option<Shares>,
        /// 044-R8 / 022-R12 (AMB-022-7): enemy-race targets inside the teacher (+ dagger) share,
        /// `Z=0.5,P=0.3,T=0.2` (every race T / P / Z once, each in (0, 1), summing to 1).
        #[arg(long, value_parser = parse_teacher_races)]
        teacher_races: Option<std::collections::BTreeMap<String, f64>>,
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
    /// abnormal exit, a hang or a fatal stderr line (it resumes from its last checkpoint). Exit 0
    /// (done, step limit, Ctrl+C clean stop, lock lost) and 130 (second Ctrl+C) are not
    /// restarted; 70 (fatal device error, 042-R20) is.
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
        /// No progress for this long = hang: kill and restart. Progress = a --progress-file change
        /// when one is given, else any stderr line (060-R6).
        #[arg(long, default_value = "30")]
        hang_minutes: u64,
        /// File whose changes alone count as progress (rl: <logs>/train.csv); stderr lines then
        /// do not (060-R6, AMB-042-20).
        #[arg(long)]
        progress_file: Option<PathBuf>,
        /// Regex (repeatable): a child stderr line matching it kills the child and restarts the
        /// run like a crash (060-R6, AMB-042-20), e.g. `--fatal-pattern "illegal memory access"`.
        #[arg(long = "fatal-pattern", value_parser = parse_regex)]
        fatal_pattern: Vec<regex::Regex>,
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
    /// 052-R7: the games `bc-eval --split <s> --games <n> --seed <k>` scores (032-R17 ②), one JSON
    /// line each with keys `map_family`, `matchup`, `perspective_slot`, `replay_sha256`, `split`,
    /// `src` (serde_json writes them sorted);
    /// `--target-maps` runs the same shuffle over the split's target-map games only. Reads the
    /// scan (cached), not the steps.
    EvalGames {
        #[command(flatten)]
        data: DataArgs,
        #[arg(long)]
        split: String,
        #[arg(long)]
        games: usize,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        #[arg(long)]
        target_maps: bool,
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

/// Shard folders read as is (like `--shards`) and scanned as a group with one scan cache
/// (`bc::scan_groups`): `--group a,b,c --group d` reuses the cache of an earlier run that read
/// `a,b,c` together when `d` has changed. Read-only: no manifest is checked or written.
/// `--dataset-group a,b,c`: the same with datasets (`--dataset` names, pulled or verified through
/// their manifests as the trainer reads them, 061-R11-1 / R12).
#[derive(clap::Args)]
struct GroupArgs {
    #[arg(long = "group")]
    group: Vec<String>,
    #[arg(long = "dataset-group")]
    dataset_group: Vec<String>,
}

impl GroupArgs {
    /// The `--dataset` / `--shards` folders as one group (when given), then each
    /// `--dataset-group` and each `--group`.
    fn groups(&self, data: &DataArgs) -> Result<Vec<Vec<PathBuf>>> {
        let mut out = Vec::new();
        if !data.dataset.is_empty() || !data.shards.is_empty() {
            out.push(data.dirs()?);
        }
        for g in &self.dataset_group {
            let names: Vec<PathBuf> = g.split(',').filter(|s| !s.is_empty()).map(PathBuf::from).collect();
            out.push(uranium_train::storage::dataset_dirs(&names, &[])?);
        }
        out.extend(self.group.iter().map(|g| g.split(',').filter(|s| !s.is_empty()).map(PathBuf::from).collect::<Vec<_>>()));
        anyhow::ensure!(out.iter().any(|g| !g.is_empty()), "pass --dataset, --shards or --group");
        Ok(out)
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum StageArg {
    S1,
    S2,
}

/// `mix-fractions --share` (044-R5 ③): the teacher target and the per-source targets.
#[derive(Clone, Debug)]
struct Shares {
    teacher: f64,
    sources: std::collections::BTreeMap<String, f64>,
}

/// `teacher=0.45,distill=0.2`: `teacher` once, at least one source, every value a number in (0, 1).
fn parse_shares(s: &str) -> std::result::Result<Shares, String> {
    let (mut teacher, mut sources) = (None, std::collections::BTreeMap::new());
    for part in s.split(',').filter(|p| !p.is_empty()) {
        let (k, v) = part.split_once('=').ok_or_else(|| format!("{part:?}: expected <name>=<share>"))?;
        let v: f64 = v.trim().parse().map_err(|e| format!("{part:?}: {e}"))?;
        if !(v > 0.0 && v < 1.0) {
            return Err(format!("{part:?}: a share is in (0, 1)"));
        }
        let k = k.trim();
        if k.is_empty() {
            return Err(format!("{part:?}: empty name"));
        }
        if k == "human" {
            return Err("the human share is 1 - the others; do not name it".into());
        }
        let dup = if k == "teacher" { teacher.replace(v).is_some() } else { sources.insert(k.to_string(), v).is_some() };
        if dup {
            return Err(format!("{k:?} given twice"));
        }
    }
    let teacher = teacher.ok_or("no teacher=<share>")?;
    if sources.is_empty() {
        return Err("no source share (e.g. distill=0.2); a teacher share alone is --teacher-share".into());
    }
    Ok(Shares { teacher, sources })
}

/// `--teacher-races Z=0.5,P=0.3,T=0.2` (044-R8 / 022-R12): every race T / P / Z once, each share in
/// (0, 1), the shares summing to 1.
fn parse_teacher_races(s: &str) -> std::result::Result<std::collections::BTreeMap<String, f64>, String> {
    let mut out = std::collections::BTreeMap::new();
    for part in s.split(',').filter(|p| !p.is_empty()) {
        let (k, v) = part.split_once('=').ok_or_else(|| format!("{part:?}: expected <race>=<share>"))?;
        let k = k.trim();
        if !["T", "P", "Z"].contains(&k) {
            return Err(format!("{part:?}: the race is T, P or Z"));
        }
        let v: f64 = v.trim().parse().map_err(|e| format!("{part:?}: {e}"))?;
        if !(v > 0.0 && v < 1.0) {
            return Err(format!("{part:?}: a share is in (0, 1)"));
        }
        if out.insert(k.to_string(), v).is_some() {
            return Err(format!("{k:?} given twice"));
        }
    }
    if out.len() != 3 {
        return Err("give every race T, P and Z (e.g. Z=0.5,P=0.3,T=0.2)".into());
    }
    let sum: f64 = out.values().sum();
    if (sum - 1.0).abs() > 1e-9 {
        return Err(format!("the race shares sum to {sum}, not 1"));
    }
    Ok(out)
}

/// 044-R8 / 022-R12: the teacher races judged inside `mix` and their [`TeacherRaces`] report (a short
/// race also on stderr). `None` without `--teacher-races`.
fn teacher_race_report(
    races: Option<&std::collections::BTreeMap<String, f64>>,
    mix: uranium_train::mix::MixShares,
    epoch: &uranium_train::mix::SourceRaceCounts,
    s2: &[uranium_train::data::GameSummary],
    all: &[uranium_train::data::GameSummary],
) -> Result<(uranium_train::mix::MixShares, Option<uranium_train::mix::TeacherRaces>)> {
    let Some(r) = races else { return Ok((mix, None)) };
    let mix = mix.with_teacher_races(epoch, r);
    let tr = uranium_train::mix::TeacherRaces::new(s2, all, &mix.teacher_sources, r, &mix)?;
    for (race, x) in &tr.short {
        eprintln!(
            "[mix-fractions] teacher race {race}: below its share band, {} steps below its target ({:.0} steps per counted game, {:.1}% of its games in the selection): about {} more games of it (044-R8 top-up, user decision 2026-11-14: rush_check_plan.py --botgen-tp --extra N --part K, N = {} for every short Protoss / Terran race, plus the build's filtered share)",
            x.missing_steps,
            x.steps_per_game,
            x.selected_share * 100.0,
            x.extra_games,
            tr.extra_botgen_tp_games.map_or("-".to_string(), |n| n.to_string())
        );
    }
    Ok((mix, Some(tr)))
}

/// `--fatal-pattern` (060-R6): a regex, checked when the command line is read.
fn parse_regex(s: &str) -> std::result::Result<regex::Regex, String> {
    regex::Regex::new(s).map_err(|e| e.to_string())
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
        Cmd::EvalGames { data, split, games, seed, target_maps } => {
            use uranium_train::split::{Selection, Split};
            let sp = match split.as_str() {
                "train" => Split::Train,
                "val" => Split::Val,
                "test" => Split::Test,
                _ => anyhow::bail!("--split {split}: expected train, val or test"),
            };
            // The same source and scan as `bc-eval` (uranium_data::eval::EvalStream::new).
            let src = ShardSource::from_dirs(&data.dirs()?)?.select(Selection::Only(sp));
            let scan = uranium_train::data::scan_cached(&src)?;
            for m in uranium_train::bc::eval_game_metas(&src, &scan, games, seed, target_maps) {
                let row = serde_json::json!({ "replay_sha256": m.replay_sha256, "perspective_slot": m.perspective_slot, "src": m.src, "map_family": m.map_family, "matchup": m.matchup, "split": split });
                println!("{row}");
            }
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
        Cmd::BcEpoch { data, groups, config } => {
            let cfg = BcConfig::load(&config)?;
            let out = uranium_train::bc::epoch_table_groups(&groups.groups(&data)?, &cfg)?;
            println!("{}", serde_json::to_string_pretty(&out)?);
            Ok(())
        }
        Cmd::MixFractions { data, groups, config, teacher_share, share, teacher_races } => {
            let cfg = BcConfig::load(&config)?;
            let sub = cfg.data.subsample.clone().ok_or_else(|| anyhow::anyhow!("{}: no [data.subsample] (its keep_sources are the teacher, 041-R21 ⑥)", config.display()))?;
            let groups = groups.groups(&data)?;
            let scan = uranium_train::bc::scan_groups(&groups)?;
            let all: Vec<uranium_train::data::GameSummary> = scan.into_values().flatten().collect();
            let s2: Vec<uranium_train::data::GameSummary> = all.iter().filter(|g| uranium_train::split::selects(uranium_train::split::Selection::S2, &g.meta)).cloned().collect();
            let mut with = cfg.clone();
            let (f, source_fractions) = match &share {
                None => {
                    // Without --share a fraction_by_source source would be counted as human.
                    anyhow::ensure!(sub.fraction_by_source.is_none(), "{}: [data.subsample] fraction_by_source needs --share (044-R5 ③)", config.display());
                    // 044-R8 (user decision 2026-11-14): the teacher round has no distill share; its games would
                    // be counted as human here.
                    anyhow::ensure!(
                        teacher_races.is_none() || !s2.iter().any(|g| g.meta.src == "distill"),
                        "--teacher-races without --share: the data has distill games, but the teacher round has no distill share (044-R8); leave distill_m2 out"
                    );
                    let f = uranium_train::mix::race_fractions(&s2, &sub.keep_sources, cfg.data.seed, teacher_share)?;
                    with.data.subsample = Some(uranium_train::split::Subsample { fraction: f.fraction, keep_sources: sub.keep_sources.clone(), fraction_by_race: Some(f.fraction_by_race.clone()), fraction_by_source: None });
                    (f, None)
                }
                Some(sh) => {
                    let unnamed: Vec<String> = sub.share_sources().into_iter().filter(|s| !sh.sources.contains_key(s)).collect();
                    anyhow::ensure!(unnamed.is_empty(), "{}: fraction_by_source {unnamed:?} has no --share target (044-R5 ③)", config.display());
                    let (f, sf) = uranium_train::mix::share_fractions(&s2, &sub.keep_sources, &sh.sources, cfg.data.seed, sh.teacher)?;
                    with.data.subsample = Some(uranium_train::split::Subsample {
                        fraction: f.fraction,
                        keep_sources: sub.keep_sources.clone(),
                        fraction_by_race: Some(f.fraction_by_race.clone()),
                        fraction_by_source: Some(sf.fraction_by_source.clone()),
                    });
                    // The values the config file will get must load (fractions in (0, 1]).
                    with.data.subsample.as_ref().expect("set above").check()?;
                    (f, Some(sf))
                }
            };
            let t = uranium_train::bc::epoch_table_groups(&groups, &with)?;
            // 041-R14: warmup = 5% of the epoch, rounded half up.
            let warmup = (t.s2.updates as f64 * 0.05 + 0.5).floor() as u64;
            let mix = t.s2_mix.clone().expect("the config has [data.subsample]");
            if let Some(sh) = &share {
                // 044-R5 ③: teacher, each source and the human rest, each ± 3%p; human races ± 3%p.
                let mix = mix.with_shares(sh.teacher, &sh.sources)?;
                // 044-R8 / 022-R12: Zerg : Protoss : Terran inside the teacher + dagger steps, ± 3%p.
                let (mix, tr) = teacher_race_report(teacher_races.as_ref(), mix, &t.s2.by_source, &s2, &all)?;
                // 044-R5 ③ top-up (user decision 2026-11-14): a source whose share fails below its band
                // gets the missing steps and an estimate of the games to add (rush_check_plan.py --extra).
                let source_fractions = source_fractions.map(|sf| sf.with_shortfalls(&all, &mix));
                let mut out = serde_json::json!({ "config": config, "fractions": f, "source_fractions": source_fractions, "s2": t.s2, "s2_warmup": warmup, "s2_mix": mix });
                if let Some(tr) = &tr {
                    out["teacher_races"] = serde_json::to_value(tr)?;
                }
                println!("{}", serde_json::to_string_pretty(&out)?);
                for (s, x) in source_fractions.iter().flat_map(|sf| sf.short.iter()) {
                    eprintln!(
                        "[mix-fractions] {s}: below its share band, {} steps below its target ({:.0} steps per counted game, {:.1}% of its games in the selection): about {} more games (044-R5 ③ top-up; distill: rush_check_plan.py --distill --round R --extra N --part K, N rounded up to a multiple of 20, plus the build's filtered share)",
                        x.missing_steps,
                        x.steps_per_game,
                        x.selected_share * 100.0,
                        x.extra_games
                    );
                }
                anyhow::ensure!(
                    mix.all_ok(),
                    "the fractions miss the shares: teacher {:.4} vs {} (ok {}), sources {:?}, human {:?}, human races {:?} vs S2 {:?} (ok {}), teacher races {:?} (044-R5 ③, 041-R26, 044-R8)",
                    mix.teacher_share,
                    mix.teacher_share_target,
                    mix.teacher_ok,
                    mix.sources,
                    mix.human,
                    mix.human_race_share,
                    mix.reference_human_race_share,
                    mix.race_ok,
                    mix.teacher_races
                );
                return Ok(());
            }
            // 041-R21 ⑥ / 041-R22 ③ / 041-R29 ④: the teacher share is judged against --teacher-share
            // (0.5 = the mix epoch; a DAgger round's teacher + dagger share), same ±3% tolerance.
            let mix = mix.with_target(teacher_share);
            // 044-R8 / 022-R12 (teacher round t1: --teacher-share 0.65): the teacher races, ± 3%p.
            let (mix, tr) = teacher_race_report(teacher_races.as_ref(), mix, &t.s2.by_source, &s2, &all)?;
            let mut out = serde_json::json!({ "config": config, "fractions": f, "s2": t.s2, "s2_warmup": warmup, "s2_mix": mix });
            if let Some(tr) = &tr {
                out["teacher_races"] = serde_json::to_value(tr)?;
            }
            println!("{}", serde_json::to_string_pretty(&out)?);
            anyhow::ensure!(
                mix.teacher_races.values().all(|c| c.ok != Some(false)),
                "the teacher races miss their shares: {:?} (044-R8; short races: play more 022 teacher botgen games, see teacher_races.short)",
                mix.teacher_races
            );
            // 041-R21 ⑥ / R26 ③: fractions that miss a share are not to be used.
            anyhow::ensure!(
                mix.teacher_ok && mix.race_ok,
                "the fractions miss the mix shares: teacher {:.4} vs target {} (ok {}), human races {:?} vs S2 {:?} (ok {}) (041-R21 ⑥, R22 ③, R26)",
                mix.teacher_share,
                mix.teacher_share_target,
                mix.teacher_ok,
                mix.human_race_share,
                mix.reference_human_race_share,
                mix.race_ok
            );
            Ok(())
        }
        Cmd::FeedDigest { data, config, stage, batches, group, max_select, pack, synthetic_maps } => {
            let stage = match stage {
                StageArg::S1 => BcStage::S1,
                StageArg::S2 => BcStage::S2,
            };
            let cfg = BcConfig::load(&config)?.for_stage(stage);
            let src = cfg.train_source(&data.dirs()?, stage.selection())?;
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
        Cmd::Supervise { max_restarts, window_minutes, delay_seconds, hang_minutes, progress_file, fatal_pattern, log, args } => {
            anyhow::ensure!(matches!(args.first().map(String::as_str), Some("bc" | "rl")), "supervise runs `bc ...` or `rl ...`");
            let code = uranium_train::supervise::run(uranium_train::supervise::SuperviseArgs {
                max_restarts,
                window: std::time::Duration::from_secs(window_minutes * 60),
                delay: std::time::Duration::from_secs(delay_seconds),
                hang: std::time::Duration::from_secs(hang_minutes * 60),
                progress_file,
                fatal_patterns: fatal_pattern,
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
                    fatal_patterns: Vec::new(),
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

#[cfg(test)]
mod tests {
    use super::*;

    /// 044-R5 ③: `mix-fractions --share teacher=0.45,distill=0.2`; `--teacher-share` alone keeps
    /// its default and meaning, and the two options do not go together.
    #[test]
    fn share_option_044_r5() {
        let s = parse_shares("teacher=0.45,distill=0.2").unwrap();
        assert_eq!((s.teacher, s.sources.clone()), (0.45, [("distill".to_string(), 0.2)].into()));
        for bad in ["distill=0.2", "teacher=0.45", "teacher=0.45,distill=0", "teacher=0.45,distill=1", "teacher=0.45,distill=x", "teacher=0.45,teacher=0.4,distill=0.2", "teacher=0.45,distill=0.2,human=0.35", "teacher", "teacher=0.45,=0.2"] {
            assert!(parse_shares(bad).is_err(), "{bad}");
        }
        let cmd = |args: &[&str]| Cli::try_parse_from([&["uranium-train", "mix-fractions", "--config", "c.toml", "--shards", "d"], args].concat());
        let Cmd::MixFractions { teacher_share, share, .. } = cmd(&[]).unwrap().cmd else { panic!() };
        assert_eq!((teacher_share, share.is_none()), (0.5, true));
        let Cmd::MixFractions { teacher_share, share, .. } = cmd(&["--teacher-share", "0.5954"]).unwrap().cmd else { panic!() };
        assert_eq!((teacher_share, share.is_none()), (0.5954, true));
        let Cmd::MixFractions { share, .. } = cmd(&["--share", "teacher=0.45,distill=0.2"]).unwrap().cmd else { panic!() };
        assert_eq!(share.unwrap().teacher, 0.45);
        assert!(cmd(&["--teacher-share", "0.5", "--share", "teacher=0.45,distill=0.2"]).is_err());
    }

    /// 044-R8 / 022-R12 (AMB-022-7): `--teacher-races Z=0.5,P=0.3,T=0.2` — every race once, shares in
    /// (0, 1) summing to 1 — goes with `--teacher-share` (the teacher round t1, user decision
    /// 2026-11-14) or `--share`; without it the command line is the one before.
    #[test]
    fn teacher_races_option_044_r8() {
        let r = parse_teacher_races("Z=0.5,P=0.3,T=0.2").unwrap();
        assert_eq!(r, [("P".to_string(), 0.3), ("T".to_string(), 0.2), ("Z".to_string(), 0.5)].into());
        for bad in ["Z=0.5,P=0.3", "Z=0.5,P=0.3,T=0.3", "Z=0.5,P=0.3,T=0.2,Z=0.5", "Z=0.5,P=0.3,R=0.2", "Z=1,P=0,T=0", "Z=0.5,P=0.3,T=x", "Z=0.5,P=0.3,T"] {
            assert!(parse_teacher_races(bad).is_err(), "{bad}");
        }
        let cmd = |args: &[&str]| Cli::try_parse_from([&["uranium-train", "mix-fractions", "--config", "c.toml", "--shards", "d"], args].concat());
        let Cmd::MixFractions { teacher_share, share, teacher_races, .. } = cmd(&["--teacher-share", "0.65", "--teacher-races", "Z=0.5,P=0.3,T=0.2"]).unwrap().cmd else { panic!() };
        assert_eq!((teacher_share, share.is_none(), teacher_races.unwrap()["P"]), (0.65, true, 0.3));
        let Cmd::MixFractions { share, teacher_races, .. } = cmd(&["--share", "teacher=0.45,distill=0.2", "--teacher-races", "Z=0.5,P=0.3,T=0.2"]).unwrap().cmd else { panic!() };
        assert_eq!((share.unwrap().teacher, teacher_races.unwrap()["Z"]), (0.45, 0.5));
        let Cmd::MixFractions { teacher_races, .. } = cmd(&["--teacher-share", "0.65"]).unwrap().cmd else { panic!() };
        assert!(teacher_races.is_none());
        assert!(cmd(&["--teacher-races", "Z=0.5,P=0.3"]).is_err());
    }
}
