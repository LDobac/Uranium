use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use serde::Serialize;
use sha2::Digest as _;

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Write `<URANIUM_MAPS_DIR>/<map sha256>.resources.toml` from each map's CHK resources.
    Mapgen { maps: Vec<PathBuf> },
    /// Print header, players and per-player command counts / APM as JSON.
    Inspect { rep: PathBuf },
    /// Write the 1.16.1 (`reRS`, stored chunks) version of a replay for OpenBW.
    Convert { rep: PathBuf, out: PathBuf },
    /// Parse every `.rep` below a folder and print one JSON line per replay (no names).
    Scan { dir: PathBuf },
    /// Print each replay's command stream as one JSON line.
    Cmddump {
        #[arg(required = true)]
        reps: Vec<PathBuf>,
    },
    /// Replay games in OpenBW from our player's view and print ops metrics, one JSON line per view.
    Opsdump {
        #[arg(required = true)]
        reps: Vec<PathBuf>,
        /// BWAPI slot of our player (default: every active Terran).
        #[arg(long)]
        perspective: Option<u8>,
        #[arg(long, default_value_t = uranium_replay::opsdump::UNTIL_FRAME)]
        until_frame: u32,
        #[arg(long, default_value_t = uranium_replay::opsdump::STRIDE)]
        stride: u32,
        /// Also print the mined fields' depletion check (`field_check`).
        #[arg(long)]
        field_check: bool,
    },
    /// Play a replay in OpenBW replay mode and report the desync signals as JSON.
    Play {
        rep: PathBuf,
        /// Player id to observe (default: every active Terran).
        #[arg(long)]
        perspective: Option<u8>,
    },
    /// Check that the hostd StartReplay stream equals a local replay bit for bit.
    StreamCheck {
        rep: PathBuf,
        #[arg(long)]
        hostd: String,
        /// This machine's address as the hostd's launcher reaches it.
        #[arg(long, default_value = "127.0.0.1")]
        agent_ip: String,
        #[arg(long)]
        perspective: Option<u8>,
    },
    /// Debug: replay `runs` times and print the first observation that differs.
    Nondet {
        rep: PathBuf,
        #[arg(long)]
        perspective: u8,
        #[arg(long, default_value_t = 4)]
        runs: usize,
    },
    /// Debug: list the map's triggers (CHK `TRIG`): condition and action types with arguments.
    Triggers { rep: PathBuf },
    /// Write a dataset report (games and views by map, matchup, src, split; desync rate) from records.
    #[command(alias = "stats")]
    DatasetStats {
        /// `build` records (JSON lines); folders are searched for `*.jsonl` (not recursive).
        records: Vec<PathBuf>,
        #[arg(long)]
        out: PathBuf,
    },
    /// Label distribution and GameInfo checks over a shard folder, subfolders included (JSON).
    ShardStats { dir: PathBuf },
    /// Check the SC:R -> 1.16.1 conversion round trip over a folder (JSON summary).
    Roundtrip { dir: PathBuf },
    /// Find where a replay starts to diverge in OpenBW (JSON).
    Trace {
        rep: PathBuf,
        #[arg(long)]
        perspective: Option<u8>,
    },
    /// Opcode histogram of a corpus with the OpenBW class of each opcode (JSON).
    Ophist { dir: PathBuf },
    /// Write per-game labels and a compact early-game state as `.jsonl.gz` (analysis only).
    Granularity {
        /// Shard / replay files or folders (searched recursively).
        inputs: Vec<PathBuf>,
        /// File with one more input path per line.
        #[arg(long)]
        list: Option<PathBuf>,
        /// Inputs are replays (default: shards).
        #[arg(long)]
        replays: bool,
        /// Output `.jsonl.gz`.
        #[arg(long)]
        out: PathBuf,
        /// Source tag written into replay games.
        #[arg(long, default_value = "")]
        source: String,
        /// Shard filter: matchup, e.g. TvZ (--stage S1/S2/B and --families a,b also filter).
        #[arg(long)]
        matchup: Option<String>,
        #[arg(long)]
        stage: Option<String>,
        #[arg(long, value_delimiter = ',')]
        families: Vec<String>,
        #[arg(long, default_value_t = 1440)]
        lo: u32,
        #[arg(long, default_value_t = 6480)]
        hi: u32,
        #[arg(long, default_value_t = 24)]
        period: u32,
        #[arg(long, default_value_t = 11520)]
        at: u32,
        #[arg(long, default_value_t = 4)]
        jobs: usize,
    },
    /// Build BC shards from a replay folder (filter, desync check, labels, shards).
    Build {
        dir: PathBuf,
        /// Source tag: stardata scr_public community personal user bot botgen scr_ladder dagger distill branch
        #[arg(long)]
        src: String,
        /// Shard output folder (default `<URANIUM_SHARED_DIR>/datasets/replays/shards/<src>`).
        #[arg(long)]
        out: Option<PathBuf>,
        /// Per-replay records (JSON lines).
        #[arg(long)]
        records: PathBuf,
        #[arg(long, default_value_t = 4)]
        jobs: usize,
        /// Process at most this many replays (sorted by path).
        #[arg(long)]
        limit: Option<usize>,
        /// Storage budget in GB for this run.
        #[arg(long, default_value_t = 600)]
        budget_gb: u64,
        /// This host's part when several hosts split one source (`index % parts == part`).
        #[arg(long, default_value_t = 0)]
        part: usize,
        #[arg(long, default_value_t = 1)]
        parts: usize,
        /// File of SHA-256 hashes (one per line, first column) to skip as duplicates.
        #[arg(long)]
        skip_hashes: Option<PathBuf>,
        /// Ladder collect records with recorded winners (`--src scr_ladder` default: collect.jsonl).
        #[arg(long)]
        ladder_records: Option<PathBuf>,
        /// Store only the steps observed at frames <= this (default: the whole game).
        #[arg(long)]
        until_frame: Option<u32>,
        /// The plan's `max_frames`: a game at least this long ended on the time limit (result undecided).
        #[arg(long)]
        time_limit: Option<u32>,
        /// Teacher whose commands relabel Terran views: `terran_react` or `terran_tp` (required for dagger).
        #[arg(long)]
        relabel: Option<String>,
        /// Folder of the model decision or act logs (`<replay stem>.jsonl`) that fill each step's memory.
        #[arg(long)]
        memory_log: Option<PathBuf>,
        /// The generating job's games.jsonl (lobby races); required with `--relabel terran_tp`.
        #[arg(long)]
        games: Option<PathBuf>,
    },
    /// Compare per-player command counts and APM with the screp reference parser.
    CheckScrep {
        dir: PathBuf,
        /// screp executable (icza/screp release).
        #[arg(long)]
        screp: PathBuf,
        /// Check at most this many replays (sorted by path).
        #[arg(long)]
        limit: Option<usize>,
    },
}

#[derive(Serialize)]
struct Summary {
    format: uranium_replay::Format,
    frames: u32,
    map_name: String,
    map_width: u16,
    map_height: u16,
    game_type: u16,
    players: Vec<PlayerOut>,
    cmd_count: usize,
    unit_limit: Option<u32>,
    map_sha256: String,
}

#[derive(Serialize)]
struct PlayerOut {
    id: u8,
    kind: u8,
    race: u8,
    team: u8,
    #[serde(flatten)]
    stats: uranium_replay::PlayerStats,
}

fn summary(r: &uranium_replay::Replay) -> Summary {
    let players = r
        .header
        .players
        .iter()
        .zip(r.player_stats())
        .map(|(p, stats)| PlayerOut { id: p.id, kind: p.kind, race: p.race, team: p.team, stats })
        .collect();
    Summary {
        format: r.format,
        frames: r.header.frames,
        map_name: r.header.map_name.clone(),
        map_width: r.header.map_width,
        map_height: r.header.map_height,
        game_type: r.header.game_type,
        players,
        cmd_count: r.cmds().count(),
        unit_limit: r.unit_limit,
        map_sha256: hex(&sha2::Sha256::digest(&r.map)),
    }
}

fn write_map_resources(map: &Path) -> Result<(PathBuf, usize)> {
    use sha2::{Digest, Sha256};
    use uranium_env::maps::{MapResources, maps_dir, resources_path};
    let bytes = std::fs::read(map).with_context(|| format!("reading {}", map.display()))?;
    let hash: String = Sha256::digest(&bytes).iter().map(|b| format!("{b:02x}")).collect();
    let chk = uranium_replay::mpq::read_file(&bytes, uranium_replay::mpq::SCENARIO_CHK)?;
    let resources = uranium_env::features::mapgen::chk_resources(&chk)?;
    let dir = maps_dir()?;
    std::fs::create_dir_all(&dir)?;
    let path = resources_path(&dir, &hash);
    let source = map.file_name().map_or_else(String::new, |n| n.to_string_lossy().into_owned());
    let n = resources.len();
    let chk_sha256: String = Sha256::digest(&chk).iter().map(|b| format!("{b:02x}")).collect();
    std::fs::write(&path, toml::to_string(&MapResources::new(source, chk_sha256, resources))?)?;
    Ok((path, n))
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn load(p: &Path) -> Result<uranium_replay::Replay> {
    let bytes = std::fs::read(p).with_context(|| format!("reading {}", p.display()))?;
    uranium_replay::parse(&bytes)
}

fn reps(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    for e in std::fs::read_dir(dir)? {
        let e = e?;
        let p = e.path();
        let t = e.file_type()?;
        if t.is_dir() || (t.is_symlink() && p.is_dir()) {
            reps(&p, out)?;
        } else if p.extension().is_some_and(|x| x.eq_ignore_ascii_case("rep")) {
            out.push(p);
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    uranium_env::dotenv::load_env();
    match Cli::parse().cmd {
        Cmd::Mapgen { maps } => {
            for m in &maps {
                let (path, n) = write_map_resources(m)?;
                println!("{}: {n} resources -> {}", m.display(), path.display());
            }
        }
        Cmd::Inspect { rep } => println!("{}", serde_json::to_string_pretty(&summary(&load(&rep)?))?),
        Cmd::Convert { rep, out } => std::fs::write(&out, load(&rep)?.to_legacy_file()?)?,
        Cmd::Scan { dir } => {
            let mut files = Vec::new();
            reps(&dir, &mut files)?;
            files.sort();
            for (i, f) in files.iter().enumerate() {
                let sha = std::fs::read(f).map(|b| hex(&sha2::Sha256::digest(&b))).unwrap_or_default();
                let line = match load(f) {
                    Ok(r) => serde_json::json!({"index": i, "ok": true, "replay_sha256": sha, "summary": summary(&r)}),
                    Err(e) => serde_json::json!({"index": i, "ok": false, "error": format!("{e:#}")}),
                };
                println!("{line}");
            }
        }
        Cmd::CheckScrep { dir, screp, limit } => check_screp(&dir, &screp, limit)?,
        Cmd::Cmddump { reps } => {
            for p in &reps {
                println!("{}", load(p)?.cmddump(&p.to_string_lossy()));
            }
        }
        Cmd::Opsdump { reps, perspective, until_frame, stride, field_check } => {
            use uranium_replay::opsdump::{line, measure, perspective_slots};
            anyhow::ensure!(stride >= 1, "--stride must be at least 1");
            let cfg = uranium_replay::play::PlayConfig::from_env(uranium_replay::play::work_dir_default())?;
            for p in &reps {
                let name = p.to_string_lossy();
                let out: Vec<_> = match load(p).and_then(|rep| perspective_slots(&rep, perspective).map(|s| (rep, s))) {
                    Ok((rep, slots)) => slots.into_iter().map(|slot| line(&name, Some(slot), Some(rep.header.frames), measure(&rep, slot, &cfg, until_frame, stride), field_check)).collect(),
                    Err(e) => vec![line(&name, perspective, None, Err(e), field_check)],
                };
                for l in out {
                    println!("{}", serde_json::to_string(&l)?);
                }
            }
        }
        Cmd::Build { dir, src, out, records, jobs, limit, budget_gb, skip_hashes, part, parts, ladder_records, until_frame, time_limit, relabel, memory_log, games } => {
            anyhow::ensure!(parts >= 1 && part < parts, "--part must be below --parts");
            let bc_shards_root = std::env::var_os("URANIUM_SHARED_DIR").map(|d| PathBuf::from(d).join("datasets/replays/shards"));
            uranium_replay::pipeline::check_build_args(&src, relabel.is_some(), memory_log.as_deref(), out.as_deref(), bc_shards_root.as_deref())?;
            uranium_replay::pipeline::check_isolated_build(&src, &records, time_limit)?;
            let memory_log = memory_log.as_deref().map(uranium_replay::act_log::MemoryLogDir::open).transpose()?;
            let relabel = relabel.as_deref().map(uranium_replay::relabel::teacher_build).transpose()?;
            uranium_replay::pipeline::check_games_arg(&src, relabel, games.is_some())?;
            let games = games.as_deref().map(uranium_replay::lobby::GamesLog::load).transpose()?;
            let skip_sha256 = match skip_hashes {
                Some(p) => std::fs::read_to_string(&p)?.lines().filter_map(|l| l.split_whitespace().next().map(str::to_string)).collect(),
                None => Default::default(),
            };
            let shards_dir = match out {
                Some(o) => o,
                None => PathBuf::from(std::env::var("URANIUM_SHARED_DIR").context("URANIUM_SHARED_DIR is not set (.env)")?).join("datasets/replays/shards").join(&src),
            };
            let mut files = Vec::new();
            reps(&dir, &mut files)?;
            files.sort();
            files.truncate(limit.unwrap_or(usize::MAX));
            if memory_log.is_some() {
                uranium_replay::pipeline::unique_stems(&files)?;
            }
            if games.is_some() {
                uranium_replay::lobby::unique_game_ids(&files)?;
            }
            let mine: Vec<usize> = (0..files.len()).filter(|i| i % parts == part).collect();
            let duplicates = uranium_replay::pipeline::Duplicates::find(&files, &mine, jobs);
            eprintln!("040-R20: {} files of this part repeat an earlier file of the source", duplicates.repeats);
            let files: Vec<PathBuf> = mine.iter().map(|&i| files[i].clone()).collect();
            let cfg = uranium_replay::play::PlayConfig::from_env(uranium_replay::play::work_dir_default())?;
            uranium_replay::play::ensure_file(&cfg.launcher)?;
            std::fs::create_dir_all(&shards_dir)?;
            let first = uranium_replay::shard::next_shard_index(&shards_dir);
            let ladder_records = ladder_records.or_else(|| {
                let p = PathBuf::from(std::env::var_os("URANIUM_SHARED_DIR")?).join("datasets/replays/ladder/collect.jsonl");
                (src == "scr_ladder" && p.is_file()).then_some(p)
            });
            let recorded_winners = match &ladder_records {
                Some(p) => recorded_winners(p)?,
                None => Default::default(),
            };
            eprintln!("recorded winners: {} (040-R8)", recorded_winners.len());
            let rec_dir = records.parent().map(PathBuf::from).unwrap_or_default();
            let mut done = uranium_replay::pipeline::ResumeDone::default();
            for e in std::fs::read_dir(if rec_dir.as_os_str().is_empty() { Path::new(".") } else { &rec_dir })?.flatten() {
                let name = e.file_name().to_string_lossy().to_string();
                if !(name.starts_with(&src) && name.ends_with(".jsonl")) {
                    continue;
                }
                for l in std::fs::read_to_string(e.path()).unwrap_or_default().lines() {
                    done.add_line(l);
                }
            }
            uranium_env::maps::maps_dir()?;
            let staging_dir = uranium_replay::shard::staging_dir();
            let opt = uranium_replay::pipeline::BuildOptions { src, shards_dir, staging_dir, recorded_winners, jobs, shard_target_bytes: 512 << 20, budget_bytes: budget_gb << 30, first_shard_index: first, skip_sha256, done, duplicates, until_frame, time_limit, relabel, memory_log, games };
            let out_file = std::sync::Mutex::new(std::fs::OpenOptions::new().create(true).append(true).open(&records)?);
            let counts = std::sync::Mutex::new(std::collections::BTreeMap::<String, usize>::new());
            uranium_replay::pipeline::build(&files, &opt, &cfg, &|r| {
                use std::io::Write as _;
                let line = serde_json::to_string(r).unwrap_or_default();
                let _ = writeln!(out_file.lock().unwrap(), "{line}");
                let kind = serde_json::to_value(&r.outcome).ok().and_then(|v| v["outcome"].as_str().map(str::to_string)).unwrap_or_default();
                *counts.lock().unwrap().entry(kind).or_default() += 1;
            })?;
            println!("{}", serde_json::json!({"replays": files.len(), "resumed_skip": opt.done.len(), "by_outcome": counts.into_inner().unwrap()}));
        }
        Cmd::DatasetStats { records, out } => {
            let mut files = Vec::new();
            for p in records {
                if p.is_dir() {
                    let mut v: Vec<PathBuf> = std::fs::read_dir(&p)?.flatten().map(|e| e.path()).filter(|f| f.extension().is_some_and(|x| x == "jsonl")).collect();
                    v.sort();
                    files.extend(v);
                } else {
                    files.push(p);
                }
            }
            std::fs::write(&out, uranium_replay::stats::dataset_markdown(&files)?)?;
        }
        Cmd::ShardStats { dir } => {
            let s = uranium_replay::stats::shard_stats(&dir)?;
            println!("{}", serde_json::to_string_pretty(&s)?);
            uranium_replay::stats::check_shard_stats(&s)?;
        }
        Cmd::Nondet { rep, perspective, runs } => {
            let r = load(&rep)?;
            let cfg = uranium_replay::play::PlayConfig::from_env(uranium_replay::play::work_dir_default())?;
            let mut all: Vec<Vec<uranium_proto::RawFrame>> = Vec::new();
            for _ in 0..runs {
                let mut frames = Vec::new();
                uranium_replay::play::play_with(&r, perspective, &cfg, &[], &mut |f, _| frames.push(f.clone()))?;
                all.push(frames);
            }
            for k in 1..all.len() {
                if let Some(i) = (0..all[0].len().min(all[k].len())).find(|&i| all[0][i] != all[k][i]) {
                    let (a, b) = (&all[0][i], &all[k][i]);
                    println!("run 0 vs {k}: first difference at observation {i} (frame {})", a.frame);
                    let field = |name: &str, x: String, y: String| {
                        if x != y {
                            println!("  {name} differs:\n    {}\n    {}", &x[..x.len().min(600)], &y[..y.len().min(600)]);
                        }
                    };
                    field("self_state", format!("{:?}", a.self_state), format!("{:?}", b.self_state));
                    field("events", format!("{:?}", a.events), format!("{:?}", b.events));
                    field("effects", format!("{:?}", a.effects), format!("{:?}", b.effects));
                    field("visibility", format!("{:?}", a.visibility.count_ones()), format!("{:?}", b.visibility.count_ones()));
                    for (ua, ub) in a.units.iter().zip(&b.units) {
                        if ua != ub {
                            println!("  unit differs:\n    {ua:?}\n    {ub:?}");
                            break;
                        }
                    }
                    if a.units.len() != b.units.len() {
                        println!("  unit count {} vs {}", a.units.len(), b.units.len());
                    }
                } else {
                    println!("run 0 vs {k}: identical ({} observations)", all[0].len());
                }
            }
        }
        Cmd::StreamCheck { rep, hostd, agent_ip, perspective } => {
            let r = load(&rep)?;
            let cfg = uranium_replay::play::PlayConfig::from_env(uranium_replay::play::work_dir_default())?;
            let views = perspective.map(|p| vec![p]).unwrap_or_else(|| uranium_replay::play::terran_perspectives(&r));
            for p in views {
                let local = uranium_replay::play::play_with(&r, p, &cfg, &[], &mut |_, _| {})?;
                let remote = uranium_replay::play::play_hostd(&r, p, &hostd, &agent_ip, &[], &mut |_, _| {})?;
                println!(
                    "{}",
                    serde_json::json!({"perspective": p, "local_digest": local.digest, "hostd_digest": remote.digest, "local_obs": local.observations, "hostd_obs": remote.observations, "identical": local.digest == remote.digest && local.observations == remote.observations})
                );
            }
        }
        Cmd::Triggers { rep } => {
            let r = load(&rep)?;
            let (m, mut pos) = (&r.map, 0usize);
            let u32_at = |d: &[u8], o: usize| u32::from_le_bytes(d[o..o + 4].try_into().unwrap());
            while pos + 8 <= m.len() {
                let (name, size) = (&m[pos..pos + 4], u32_at(m, pos + 4) as usize);
                let data = &m[pos + 8..(pos + 8 + size).min(m.len())];
                if name == b"TRIG" {
                    for (i, t) in data.chunks_exact(2400).enumerate() {
                        let conds: Vec<String> = (0..16)
                            .map(|k| &t[k * 20..k * 20 + 20])
                            .filter(|c| c[15] != 0)
                            .map(|c| format!("c{}(grp {} n {} unit {} cmp {} res {})", c[15], u32_at(c, 4), u32_at(c, 8), u16::from_le_bytes([c[12], c[13]]), c[14], c[16]))
                            .collect();
                        let acts: Vec<String> = (0..64)
                            .map(|k| &t[320 + k * 32..320 + k * 32 + 32])
                            .filter(|a| a[26] != 0)
                            .map(|a| format!("a{}(grp {} num {} unit {} mod {})", a[26], u32_at(a, 16), u32_at(a, 20), u16::from_le_bytes([a[24], a[25]]), a[27]))
                            .collect();
                        let players: Vec<usize> = (0..28).filter(|&p| t[2372 + p] != 0).collect();
                        println!("trigger {i} players {players:?}: if {} then {}", conds.join(" & "), acts.join(", "));
                    }
                }
                pos += 8 + size;
            }
        }
        Cmd::Roundtrip { dir } => {
            let mut files = Vec::new();
            reps(&dir, &mut files)?;
            let (mut ok, mut failed) = (0usize, std::collections::BTreeMap::<String, usize>::new());
            for f in &files {
                match load(f).and_then(|r| r.roundtrip_check()) {
                    Ok(()) => ok += 1,
                    Err(e) => *failed.entry(format!("{e:#}").split(" at frame").next().unwrap_or("").to_string()).or_default() += 1,
                }
            }
            println!("{}", serde_json::json!({"replays": files.len(), "ok": ok, "failed": failed}));
        }
        Cmd::Trace { rep, perspective } => {
            let r = load(&rep)?;
            let cfg = uranium_replay::play::PlayConfig::from_env(uranium_replay::play::work_dir_default())?;
            let views = perspective.map(|p| vec![p]).unwrap_or_else(|| uranium_replay::play::terran_perspectives(&r));
            for p in views {
                println!("{}", serde_json::to_string_pretty(&uranium_replay::trace::trace(&r, p, &cfg)?)?);
            }
        }
        Cmd::Granularity { inputs, list, replays, out, source, matchup, stage, families, lo, hi, period, at, jobs } => {
            use uranium_replay::granularity::{ShardFilter, Window, expand, run};
            let mut inputs = inputs;
            if let Some(l) = list {
                let text = std::fs::read_to_string(&l).with_context(|| format!("reading {}", l.display()))?;
                inputs.extend(text.lines().map(str::trim).filter(|s| !s.is_empty()).map(PathBuf::from));
            }
            anyhow::ensure!(period > 0 && lo <= hi, "--period must be > 0 and --lo <= --hi");
            let files = expand(&inputs, replays)?;
            let w = Window { lo, hi, period, at };
            let (games, failed) = run(&files, replays, &out, &source, w, &ShardFilter { matchup, stage, families }, jobs)?;
            println!("{{\"files\":{},\"games\":{games},\"failed\":{failed}}}", files.len());
        }
        Cmd::Ophist { dir } => {
            let mut files = Vec::new();
            reps(&dir, &mut files)?;
            let mut hist: std::collections::BTreeMap<u8, (u64, u64)> = Default::default();
            let mut failed = 0u64;
            for f in &files {
                let Ok(r) = load(f) else {
                    failed += 1;
                    continue;
                };
                let mut seen = std::collections::BTreeSet::new();
                for c in r.cmds() {
                    let op = uranium_replay::cmd::to_legacy(c, r.unit_limit).map(|c| c.op).unwrap_or(c.op);
                    hist.entry(op).or_default().0 += 1;
                    seen.insert(op);
                }
                for op in seen {
                    hist.entry(op).or_default().1 += 1;
                }
            }
            let rows: Vec<_> = hist
                .iter()
                .map(|(op, (n, reps))| serde_json::json!({"op": format!("{op:#04x}"), "class": format!("{:?}", uranium_replay::cmd::openbw(*op)), "commands": n, "replays": reps}))
                .collect();
            println!("{}", serde_json::json!({"replays": files.len(), "failed": failed, "opcodes": rows}));
        }
        Cmd::Play { rep, perspective } => {
            let r = load(&rep)?;
            let cfg = uranium_replay::play::PlayConfig::from_env(uranium_replay::play::work_dir_default())?;
            uranium_replay::play::ensure_file(&cfg.launcher)?;
            let views = perspective.map(|p| vec![p]).unwrap_or_else(|| uranium_replay::play::terran_perspectives(&r));
            for p in views {
                println!("{}", serde_json::to_string(&uranium_replay::play::play(&r, p, &cfg)?)?);
            }
        }
    }
    Ok(())
}

fn check_screp(dir: &Path, screp: &Path, limit: Option<usize>) -> Result<()> {
    let mut files = Vec::new();
    reps(dir, &mut files)?;
    files.sort();
    files.truncate(limit.unwrap_or(usize::MAX));
    let (mut ok, mut mismatch, mut ours_failed, mut screp_failed) = (0, 0, 0, 0);
    for (i, f) in files.iter().enumerate() {
        let out = std::process::Command::new(screp).arg("-indent=false").arg(f).output()?;
        let Ok(j) = serde_json::from_slice::<serde_json::Value>(&out.stdout) else {
            screp_failed += 1;
            continue;
        };
        let r = match load(f) {
            Ok(r) => r,
            Err(e) => {
                ours_failed += 1;
                eprintln!("[{i}] ours failed: {e:#}");
                continue;
            }
        };
        let mut diffs = Vec::new();
        if j["Header"]["Frames"].as_u64() != Some(r.header.frames as u64) {
            diffs.push(format!("frames {} vs {}", j["Header"]["Frames"], r.header.frames));
        }
        let stats = r.player_stats();
        for pd in j["Computed"]["PlayerDescs"].as_array().into_iter().flatten() {
            let id = pd["PlayerID"].as_u64().unwrap_or(999) as u8;
            let Some(s) = stats.iter().find(|s| s.id == id) else {
                diffs.push(format!("player {id} missing"));
                continue;
            };
            if pd["CmdCount"].as_u64() != Some(s.cmds as u64) || pd["APM"].as_u64() != Some(s.apm as u64) {
                diffs.push(format!("player {id}: cmds {}/{} apm {}/{}", pd["CmdCount"], s.cmds, pd["APM"], s.apm));
            }
        }
        if diffs.is_empty() {
            ok += 1;
        } else {
            mismatch += 1;
            eprintln!("[{i}] screp vs ours: {}", diffs.join("; "));
        }
    }
    println!("{}", serde_json::json!({"checked": files.len(), "match": ok, "mismatch": mismatch, "ours_failed": ours_failed, "screp_failed": screp_failed}));
    Ok(())
}

fn recorded_winners(path: &Path) -> Result<std::collections::HashMap<String, String>> {
    let mut out = std::collections::HashMap::new();
    for l in std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?.lines() {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(l) else { continue };
        if let (Some(sha), Some(w)) = (v["sha256"].as_str(), v["winner"].as_str()) {
            out.insert(sha.to_string(), w.to_string());
        }
    }
    Ok(out)
}