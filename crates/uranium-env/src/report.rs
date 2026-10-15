//! Per-matchup summary (021-AC1/AC2, 020-AC1) joined from games.jsonl (hostd outcomes) and
//! results.jsonl (agent sessions).

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;

use anyhow::Result;
use serde_json::Value;

use crate::plan::{Plan, Side};

fn lines(path: &Path) -> Result<Vec<Value>> {
    let text = std::fs::read_to_string(path).unwrap_or_default();
    Ok(text.lines().filter(|l| !l.trim().is_empty()).map(serde_json::from_str).collect::<Result<_, _>>()?)
}

fn side_name(s: &Side) -> String {
    match s {
        Side::T0 { race } => format!("T0 {race:?}"),
        Side::Oss { name, race } => format!("{name} ({race:?})"),
    }
}

#[derive(Default)]
struct Row {
    games: u32,
    outcomes: BTreeMap<String, u32>,
    p0_wins: u32,
    p1_wins: u32,
    draws: u32,
    /// Mirror matchups: decided games (winner side is not meaningful).
    decided_mirror: u32,
    violations: u64,
    session_errors: u32,
    minutes: Vec<f64>,
}

pub fn write(out: &Path, plan: &Plan) -> Result<()> {
    let games = lines(&out.join("games.jsonl"))?;
    let sessions = lines(&out.join("results.jsonl"))?;
    let mut by_game: BTreeMap<u64, Vec<&Value>> = BTreeMap::new();
    for s in &sessions {
        by_game.entry(s["game_id"].as_u64().unwrap_or(0)).or_default().push(s);
    }
    let mut rows: BTreeMap<(usize, String), Row> = BTreeMap::new();
    for g in &games {
        let mi = g["matchup"].as_u64().unwrap_or(0) as usize;
        let row = rows.entry((mi, g["map"].as_str().unwrap_or("?").to_string())).or_default();
        row.games += 1;
        *row.outcomes.entry(g["outcome"].as_str().unwrap_or("?").to_string()).or_default() += 1;
        let ss = by_game.get(&g["game_id"].as_u64().unwrap_or(0)).cloned().unwrap_or_default();
        for s in &ss {
            row.violations += s["visibility_violations"].as_u64().unwrap_or(0);
            if !s["error"].is_null() {
                row.session_errors += 1;
            }
        }
        let Some(first) = ss.first() else { continue };
        if let Some(f) = first["end_frame"].as_f64() {
            row.minutes.push(f * 42.0 / 60_000.0);
        }
        let m = &plan.matchups[mi];
        let p0_race = match &m.p0 {
            Side::T0 { race } | Side::Oss { race, .. } => format!("{race:?}"),
        };
        match first["winner"].as_u64() {
            None => row.draws += 1,
            Some(w) => match m.p1 {
                // T0 vs T0: both sides have sessions; attribute the win by the winner's race.
                Side::T0 { .. } => {
                    let mirror = ss.len() == 2 && ss[0]["self_race"] == ss[1]["self_race"];
                    let winner_race = ss.iter().find(|s| s["slot"].as_u64() == Some(w)).and_then(|s| s["self_race"].as_str());
                    if mirror {
                        row.decided_mirror += 1;
                    } else if winner_race.is_some_and(|r| r.eq_ignore_ascii_case(&p0_race)) {
                        row.p0_wins += 1;
                    } else {
                        row.p1_wins += 1;
                    }
                }
                // OSS bot on p1: only our (p0) session exists.
                Side::Oss { .. } => {
                    if first["slot"].as_u64() == Some(w) {
                        row.p0_wins += 1;
                    } else {
                        row.p1_wins += 1;
                    }
                }
            },
        }
    }

    let mut md = String::new();
    writeln!(md, "# {} — matchup summary\n", plan.name)?;
    writeln!(md, "| # | p0 | p1 | map | games | outcomes | p0 wins | p1 wins | draws | mirror decided | game min p50 | vis. violations | session errors |")?;
    writeln!(md, "|---|---|---|---|---|---|---|---|---|---|---|---|---|")?;
    for ((mi, map), r) in &mut rows {
        let m = &plan.matchups[*mi];
        r.minutes.sort_by(f64::total_cmp);
        let p50 = r.minutes.get(r.minutes.len() / 2).copied().unwrap_or(f64::NAN);
        let p1 = if m.p1_random_ratio > 0.0 { format!("{} (random {:.0}%)", side_name(&m.p1), m.p1_random_ratio * 100.0) } else { side_name(&m.p1) };
        writeln!(
            md,
            "| {mi} | {} | {p1} | {map} | {} | {:?} | {} | {} | {} | {} | {p50:.1} | {} | {} |",
            side_name(&m.p0),
            r.games,
            r.outcomes,
            r.p0_wins,
            r.p1_wins,
            r.draws,
            r.decided_mirror,
            r.violations,
            r.session_errors
        )?;
    }
    let aborted: Vec<&Value> = games.iter().filter(|g| g["outcome"] == "Aborted").collect();
    if !aborted.is_empty() {
        writeln!(md, "\n## Aborted games\n")?;
        for g in aborted.iter().take(20) {
            writeln!(md, "- game {}: {}", g["game_id"], g["detail"].as_str().unwrap_or(""))?;
        }
    }
    std::fs::write(out.join("summary.md"), &md)?;
    // Engine/bridge statistics in the M2 format.
    uranium_agent::summary::summarize(&out.join("results.jsonl"), &out.join("sessions.md"), &plan.name)?;
    Ok(())
}
