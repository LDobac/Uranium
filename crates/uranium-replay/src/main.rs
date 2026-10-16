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
    /// 013: opcode histogram of a corpus with the OpenBW class of each opcode (JSON).
    Ophist { dir: PathBuf },
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
        let p = e?.path();
        if p.is_dir() {
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
