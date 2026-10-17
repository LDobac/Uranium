//! `uranium-replay` CLI (040): `inspect` one replay, `convert` SC:R -> 1.16.1 (013-R6),
//! `scan` a folder for parse errors and counts.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use serde::Serialize;

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Print header, players and per-player command counts / APM as JSON.
    Inspect { rep: PathBuf },
    /// Write the 1.16.1 (`reRS`, stored chunks) version of a replay for OpenBW.
    Convert { rep: PathBuf, out: PathBuf },
    /// Parse every `.rep` below a folder and print one JSON line per replay (no names, 040-R14).
    Scan { dir: PathBuf },
    /// 040-AC1: compare per-player command counts and APM with the screp reference parser.
    /// Play a replay in OpenBW replay mode and report the 040-R8 desync signals as JSON.
    Play {
        rep: PathBuf,
        /// Player id to observe (default: every active Terran).
        #[arg(long)]
        perspective: Option<u8>,
    },
    /// 020-AC4 / 040-AC6: the hostd StartReplay stream equals a local replay bit for bit.
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
    /// Debug (011-AC5): replay `runs` times and print the first observation that differs.
    Nondet {
        rep: PathBuf,
        #[arg(long)]
        perspective: u8,
        #[arg(long, default_value_t = 4)]
        runs: usize,
    },
    /// Debug: list the map's triggers (CHK `TRIG`): condition and action types with arguments.
    Triggers { rep: PathBuf },
    /// 040-R6 / 040-AC4: dataset statistics markdown from `build` records.
    Stats {
        records: Vec<PathBuf>,
        #[arg(long)]
        out: PathBuf,
    },
    /// 034-AC3 / AMB-040-6: label distribution and GameInfo checks over a shard folder (JSON).
    ShardStats { dir: PathBuf },
    /// 013-AC4: SC:R -> 1.16.1 conversion round trip over a folder (JSON summary).
    Roundtrip { dir: PathBuf },
    /// 040-R8 analysis: where a replay starts to diverge in OpenBW (JSON).
    Trace {
        rep: PathBuf,
        #[arg(long)]
        perspective: Option<u8>,
    },
    /// 013: opcode histogram of a corpus with the OpenBW class of each opcode (JSON).
    Ophist { dir: PathBuf },
    /// 040: build BC shards from a replay folder (filter, desync check, labels, shards).
    Build {
        dir: PathBuf,
        /// 040-R14 source tag (stardata, scr_public, community, personal, user, bot).
        #[arg(long)]
        src: String,
        /// Shard output folder (default `<URANIUM_SHARED_DIR>/datasets/replays/shards/<src>`, 040-R13).
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
        /// Storage budget in GB for this run (060-R7 total is 600).
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
    },
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
    }
}

fn load(p: &Path) -> Result<uranium_replay::Replay> {
    let bytes = std::fs::read(p).with_context(|| format!("reading {}", p.display()))?;
    uranium_replay::parse(&bytes)
}

fn reps(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    for e in std::fs::read_dir(dir)? {
        let e = e?;
        let p = e.path();
        // The listing already carries the entry type, so no per-file stat (one SMB round trip
        // each on Windows: minutes for 65k replays on a remote share).
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
    // 001-R9: host paths come from .env; the process environment wins.
    let _ = dotenvy::dotenv();
    match Cli::parse().cmd {
        Cmd::Inspect { rep } => println!("{}", serde_json::to_string_pretty(&summary(&load(&rep)?))?),
        Cmd::Convert { rep, out } => std::fs::write(&out, load(&rep)?.to_legacy_file()?)?,
        Cmd::Scan { dir } => {
            let mut files = Vec::new();
            reps(&dir, &mut files)?;
            files.sort();
            for (i, f) in files.iter().enumerate() {
                let line = match load(f) {
                    Ok(r) => serde_json::json!({"index": i, "ok": true, "summary": summary(&r)}),
                    Err(e) => serde_json::json!({"index": i, "ok": false, "error": format!("{e:#}")}),
                };
                println!("{line}");
            }
        }
        Cmd::CheckScrep { dir, screp, limit } => check_screp(&dir, &screp, limit)?,
        Cmd::Build { dir, src, out, records, jobs, limit, budget_gb, skip_hashes, part, parts } => {
            anyhow::ensure!(parts >= 1 && part < parts, "--part must be below --parts");
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
            // Several hosts split one source by sorted-file position (each writes its own shards
            // folder and records file).
            let files: Vec<PathBuf> = files.into_iter().enumerate().filter(|(i, _)| i % parts == part).map(|(_, f)| f).collect();
            let cfg = uranium_replay::play::PlayConfig::from_env(uranium_replay::play::work_dir_default())?;
            uranium_replay::play::ensure_file(&cfg.launcher)?;
            std::fs::create_dir_all(&shards_dir)?;
            let first = std::fs::read_dir(&shards_dir).map(|d| d.filter(|e| e.as_ref().is_ok_and(|e| e.file_name().to_string_lossy().starts_with("shard_"))).count() as u32).unwrap_or(0);
            // Resume: replays in any records file of this source (same folder, name starting with
            // the source tag) were handled by an earlier run or another host.
            let rec_dir = records.parent().map(PathBuf::from).unwrap_or_default();
            let mut done_sha256 = std::collections::HashSet::new();
            for e in std::fs::read_dir(if rec_dir.as_os_str().is_empty() { Path::new(".") } else { &rec_dir })?.flatten() {
                let name = e.file_name().to_string_lossy().to_string();
                if !(name.starts_with(&src) && name.ends_with(".jsonl")) {
                    continue;
                }
                for l in std::fs::read_to_string(e.path()).unwrap_or_default().lines() {
                    if let Some(s) = serde_json::from_str::<serde_json::Value>(l).ok().and_then(|v| v["replay_sha256"].as_str().map(str::to_string)) {
                        done_sha256.insert(s);
                    }
                }
            }
            let opt = uranium_replay::pipeline::BuildOptions { src, shards_dir, jobs, shard_target_bytes: 512 << 20, budget_bytes: budget_gb << 30, first_shard_index: first, skip_sha256, done_sha256 };
            let out_file = std::sync::Mutex::new(std::fs::OpenOptions::new().create(true).append(true).open(&records)?);
            let counts = std::sync::Mutex::new(std::collections::BTreeMap::<String, usize>::new());
            uranium_replay::pipeline::build(&files, &opt, &cfg, &|r| {
                use std::io::Write as _;
                let line = serde_json::to_string(r).unwrap_or_default();
                let _ = writeln!(out_file.lock().unwrap(), "{line}");
                let kind = serde_json::to_value(&r.outcome).ok().and_then(|v| v["outcome"].as_str().map(str::to_string)).unwrap_or_default();
                *counts.lock().unwrap().entry(kind).or_default() += 1;
            })?;
            println!("{}", serde_json::json!({"replays": files.len(), "resumed_skip": opt.done_sha256.len(), "by_outcome": counts.into_inner().unwrap()}));
        }
        Cmd::Stats { records, out } => std::fs::write(&out, stats_markdown(&records)?)?,
        Cmd::ShardStats { dir } => println!("{}", serde_json::to_string_pretty(&shard_stats(&dir)?)?),
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
        Cmd::Ophist { dir } => {
            let mut files = Vec::new();
            reps(&dir, &mut files)?;
            // opcode (after 1.16.1 normalization) -> (commands, replays containing it)
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

/// Runs screp on each replay and compares frames, per-player command counts and APM.
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

/// 040-R6 / 040-AC4: counts only, no file or player names (040-R14).
/// Shard folder summary: per-action label counts (034-AC3), delay buckets and argument use, and
/// GameInfo consistency per game (AMB-040-6). Reading every shard also checks the schema version.
fn shard_stats(dir: &Path) -> Result<serde_json::Value> {
    use std::collections::BTreeMap;
    let table = uranium_env::actions::ActionTable::load()?;
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)?.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.to_string_lossy().ends_with(".bin.zst")).collect();
    files.sort();
    let (mut games, mut steps, mut bad_game_info) = (0usize, 0usize, Vec::new());
    let (mut with_select, mut with_unit, mut with_tile, mut queued) = (0usize, 0usize, 0usize, 0usize);
    let mut actions = BTreeMap::<u16, usize>::new();
    let mut delays = BTreeMap::<u32, usize>::new();
    let mut matchups = BTreeMap::<String, usize>::new();
    let mut families = BTreeMap::<String, usize>::new();
    for f in &files {
        let shard = uranium_replay::shard::read(f)?;
        for g in &shard.games {
            games += 1;
            *matchups.entry(g.meta.matchup.clone()).or_default() += 1;
            *families.entry(g.meta.map_family.clone()).or_default() += 1;
            let gi = &g.game_info;
            let sm = &gi.static_map;
            let ok = gi.map_size_tiles == (sm.width_tiles, sm.height_tiles)
                && sm.width_tiles > 0
                && sm.ground_height.len() == sm.width_tiles as usize * sm.height_tiles as usize
                && !sm.start_locations.is_empty()
                && gi.self_player == g.meta.perspective_slot
                && gi.players.iter().filter(|p| p.is_self).count() == 1
                && gi.players.iter().any(|p| p.is_self && p.id == gi.self_player);
            if !ok {
                bad_game_info.push(g.meta.replay_sha256[..12].to_string());
            }
            for s in &g.steps {
                steps += 1;
                let l = &s.action_label;
                *actions.entry(l.action).or_default() += 1;
                *delays.entry(l.delay).or_default() += 1;
                with_select += usize::from(!l.selected.is_empty());
                with_unit += usize::from(l.target_unit.is_some());
                with_tile += usize::from(l.target_tile.is_some());
                queued += usize::from(l.queued);
            }
        }
    }
    let name = |id: u16| table.actions.get(id as usize).map_or_else(|| format!("#{id}"), |a| a.name.clone());
    let mut by_count: Vec<(String, usize)> = actions.iter().map(|(&a, &n)| (name(a), n)).collect();
    by_count.sort_by(|a, b| b.1.cmp(&a.1));
    let unseen: Vec<String> = table.actions.iter().filter(|a| !actions.contains_key(&a.id)).map(|a| a.name.clone()).collect();
    let pct = |n: usize| if steps == 0 { 0.0 } else { (n as f64 * 10000.0 / steps as f64).round() / 100.0 };
    Ok(serde_json::json!({
        "shards": files.len(), "games": games, "steps": steps,
        "game_info_bad": bad_game_info,
        "matchups": matchups, "map_families": families,
        "pct_with_selected": pct(with_select), "pct_with_target_unit": pct(with_unit),
        "pct_with_target_tile": pct(with_tile), "pct_queued": pct(queued),
        "delay_buckets": delays,
        "actions": by_count.iter().map(|(n, c)| serde_json::json!([n, c, pct(*c)])).collect::<Vec<_>>(),
        "actions_unseen": unseen,
    }))
}

fn stats_markdown(records: &[PathBuf]) -> Result<String> {
    use std::collections::BTreeMap;
    use std::fmt::Write as _;
    let mut rows: Vec<serde_json::Value> = Vec::new();
    for p in records {
        for line in std::fs::read_to_string(p).with_context(|| format!("reading {}", p.display()))?.lines() {
            rows.push(serde_json::from_str(line)?);
        }
    }
    let s = |v: &serde_json::Value, k: &str| v[k].as_str().unwrap_or("").to_string();
    let mut outcomes: BTreeMap<(String, String), usize> = BTreeMap::new();
    let mut games: BTreeMap<(String, String, String), (usize, u64)> = BTreeMap::new();
    let mut desync: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let (mut cmds, mut labeled, mut spam) = (0u64, 0u64, 0u64);
    let mut replays: BTreeMap<String, std::collections::BTreeSet<(String, u64)>> = BTreeMap::new();
    for r in &rows {
        let (src, outcome) = (s(r, "src"), s(r, "outcome"));
        replays.entry(src.clone()).or_default().insert((s(r, "replay_sha256"), r["index"].as_u64().unwrap_or(0)));
        *outcomes.entry((src.clone(), outcome.clone())).or_default() += 1;
        let family = s(r, "map_family");
        if outcome == "written" || outcome == "desync" {
            let d = desync.entry(family.clone()).or_default();
            d.1 += 1;
            d.0 += (outcome == "desync") as usize;
        }
        if outcome == "written" {
            let view = &r["view"];
            let g = games.entry((family, s(view, "matchup"), s(view, "stage"))).or_default();
            g.0 += 1;
            g.1 += r["steps"].as_u64().unwrap_or(0);
            let c = &r["coverage"];
            cmds += c["commands"].as_u64().unwrap_or(0);
            labeled += c["labeled"].as_u64().unwrap_or(0);
            spam += c["merged_spam"].as_u64().unwrap_or(0);
        }
    }
    let mut md = String::new();
    writeln!(md, "# 데이터셋 통계 (040-R6, 040-AC4)\n")?;
    writeln!(md, "`uranium-replay stats`로 생성. 개수만 기록한다(040-R14). 판(game)은 테란 시점 하나다(TvT는 두 시점, 040-R2).\n")?;
    writeln!(md, "## 출처별 처리 결과\n\n| src | 리플레이 | 결과 | 레코드 |\n|---|---|---|---|")?;
    for ((src, outcome), n) in &outcomes {
        writeln!(md, "| {src} | {} | {outcome} | {n} |", replays.get(src).map_or(0, |r| r.len()))?;
    }
    writeln!(md, "\n## 저장된 판: 맵 패밀리 × 매치업 × 단계\n\n| 맵 패밀리 | 매치업 | 단계 | 판 | 판단 단계 |\n|---|---|---|---|---|")?;
    for ((f, m, st), (n, steps)) in &games {
        writeln!(md, "| {f} | {m} | {st} | {n} | {steps} |")?;
    }
    writeln!(md, "\n## 디싱크 비율 (040-AC2, 목표 < 10%)\n\n| 맵 패밀리 | 재생 시점 | 디싱크 | 비율 |\n|---|---|---|---|")?;
    let (mut dt, mut tt) = (0, 0);
    for (f, (d, t)) in &desync {
        writeln!(md, "| {f} | {t} | {d} | {:.1}% |", *d as f64 * 100.0 / (*t).max(1) as f64)?;
        dt += d;
        tt += t;
    }
    writeln!(md, "| 전체 | {tt} | {dt} | {:.1}% |", dt as f64 * 100.0 / tt.max(1) as f64)?;
    writeln!(md, "\n## 레이블 커버리지 (031-AC3, 목표 ≥ 95%)\n\n- 게임 영향 명령 {cmds}개 중 레이블 {labeled}개: {:.2}%\n- 스팸 병합(040-R4): {spam}개", labeled as f64 * 100.0 / cmds.max(1) as f64)?;
    Ok(md)
}
