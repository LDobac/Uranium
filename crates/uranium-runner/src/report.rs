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
        Side::T0 { race, build } if *build != Default::default() => format!("T0 {race:?} `{}`", build.name()),
        Side::T0 { race, .. } => format!("T0 {race:?}"),
        Side::Oss { name, race } => format!("{name} ({race:?})"),
        Side::Model { checkpoint, sample } => format!("model `{checkpoint}` ({sample:?})"),
    }
}

/// 050-R2: Wilson 95% interval of `wins` out of `n`, in percent.
pub fn wilson(wins: u32, n: u32) -> (f64, f64) {
    if n == 0 {
        return (0.0, 100.0);
    }
    let (z, n, p) = (1.96f64, n as f64, wins as f64 / n as f64);
    let d = 1.0 + z * z / n;
    let c = (p + z * z / (2.0 * n)) / d;
    let h = z * (p * (1.0 - p) / n + z * z / (4.0 * n * n)).sqrt() / d;
    (100.0 * (c - h).max(0.0), 100.0 * (c + h).min(1.0))
}

fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
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
    /// Games that ended within 300 frames: an opponent that never really played
    /// (e.g. its module failed to load). Excluded from win counts.
    suspect_short: u32,
    /// Games whose two sessions report different winners (the first session's is counted).
    winner_disagree: u32,
    /// Opponent IDs seen for p1 (021-R3).
    opponents: std::collections::BTreeSet<String>,
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
        if let Some(o) = g["opponent"].as_str() {
            row.opponents.insert(o.to_string());
        }
        *row.outcomes.entry(g["outcome"].as_str().unwrap_or("?").to_string()).or_default() += 1;
        let ss = by_game.get(&g["game_id"].as_u64().unwrap_or(0)).cloned().unwrap_or_default();
        for s in &ss {
            row.violations += s["visibility_violations"].as_u64().unwrap_or(0);
            if !s["error"].is_null() {
                row.session_errors += 1;
            }
        }
        let Some(first) = ss.first() else { continue };
        if ss.len() == 2 && ss[0]["winner"] != ss[1]["winner"] {
            row.winner_disagree += 1;
        }
        if let Some(f) = first["end_frame"].as_f64() {
            row.minutes.push(f * 42.0 / 60_000.0);
            if f < 300.0 {
                row.suspect_short += 1;
                continue;
            }
        }
        let m = &plan.matchups[mi];
        let p0_race = format!("{:?}", m.p0.race());
        let has_model = m.p0.model().is_some() || m.p1.model().is_some();
        match first["winner"].as_u64() {
            None => row.draws += 1,
            // Model games: the BWAPI player id need not follow launch order (Windows TCP LAN), so
            // the side is read from the winner session's policy. Model vs model
            // with one policy has no side to tell; it counts like a mirror.
            Some(w) if has_model => {
                let winner_is_model = ss.iter().find(|s| s["slot"].as_u64() == Some(w)).and_then(|s| s["policy"].as_str()).map(|p| p.starts_with("model:"));
                let p0_model = m.p0.model().is_some();
                match winner_is_model {
                    Some(wm) if p0_model != m.p1.model().is_some() => {
                        if wm == p0_model {
                            row.p0_wins += 1;
                        } else {
                            row.p1_wins += 1;
                        }
                    }
                    _ => row.decided_mirror += 1,
                }
            }
            Some(w) => match m.p1 {
                // T0 vs T0: both sides have sessions; attribute the win by the winner's race.
                Side::T0 { .. } => {
                    let mirror = ss.len() == 2 && ss[0]["self_race"] == ss[1]["self_race"];
                    let winner_race = ss.iter().find(|s| s["slot"].as_u64() == Some(w)).and_then(|s| s["self_race"].as_str());
                    // A mirror between different builds (050 terran drop vs T0 Terran) is
                    // attributed by the winner's build.
                    let winner_build = ss.iter().find(|s| s["slot"].as_u64() == Some(w)).and_then(|s| s["t0_build"].as_str());
                    let builds_differ = ss.len() == 2 && ss[0]["t0_build"] != ss[1]["t0_build"];
                    let p1_build = match &m.p1 {
                        Side::T0 { build, .. } => build.name(),
                        Side::Oss { .. } | Side::Model { .. } => "",
                    };
                    if mirror && builds_differ {
                        if winner_build == Some(p1_build) {
                            row.p1_wins += 1;
                        } else {
                            row.p0_wins += 1;
                        }
                    } else if mirror {
                        row.decided_mirror += 1;
                    } else if winner_race.is_some_and(|r| r.eq_ignore_ascii_case(&p0_race)) {
                        row.p0_wins += 1;
                    } else {
                        row.p1_wins += 1;
                    }
                }
                // OSS bot on p1: only our (p0) session exists.
                Side::Oss { .. } | Side::Model { .. } => {
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
    writeln!(md, "| # | p0 | p1 | map | games | outcomes | p0 wins | p1 wins | draws | p0 win% [95% CI] | mirror decided | winner disagree | suspect (<300 frames) | game min p50 | vis. violations | session errors |")?;
    writeln!(md, "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")?;
    for ((mi, map), r) in &mut rows {
        let m = &plan.matchups[*mi];
        r.minutes.sort_by(f64::total_cmp);
        let p50 = r.minutes.get(r.minutes.len() / 2).copied().unwrap_or(f64::NAN);
        let mut p1 = if m.p1_random_ratio > 0.0 { format!("{} (random {:.0}%)", side_name(&m.p1), m.p1_random_ratio * 100.0) } else { side_name(&m.p1) };
        if !r.opponents.is_empty() {
            p1 = format!("{p1} `{}`", r.opponents.iter().cloned().collect::<Vec<_>>().join(", "));
        }
        // 050-R2: decided + drawn games count, draws as non-wins.
        let n = r.p0_wins + r.p1_wins + r.draws;
        let (lo, hi) = wilson(r.p0_wins, n);
        let rate = if n > 0 { format!("{:.0}% [{lo:.0}, {hi:.0}]", 100.0 * r.p0_wins as f64 / n as f64) } else { "-".into() };
        writeln!(
            md,
            "| {mi} | {} | {p1} | {map} | {} | {:?} | {} | {} | {} | {rate} | {} | {} | {} | {p50:.1} | {} | {} |",
            side_name(&m.p0),
            r.games,
            r.outcomes,
            r.p0_wins,
            r.p1_wins,
            r.draws,
            r.decided_mirror,
            r.winner_disagree,
            r.suspect_short,
            r.violations,
            r.session_errors
        )?;
    }
    // 020-R5 / 031-R4: model sessions (policy version, latency, executor rejections).
    let model: Vec<&Value> = sessions.iter().filter(|s| s["model"].is_object()).collect();
    if !model.is_empty() {
        let mut by: BTreeMap<String, Vec<&Value>> = BTreeMap::new();
        for s in &model {
            by.entry(s["policy"].as_str().unwrap_or("?").to_string()).or_default().push(s);
        }
        writeln!(md, "\n## Model sessions\n")?;
        writeln!(md, "| policy | sessions | decisions | decide ms p50 (median) | decide ms p95 (max) | run ms p50 (median) | mean batch | host ms p50 | rejected | map precompute |")?;
        writeln!(md, "|---|---|---|---|---|---|---|---|---|---|")?;
        for (policy, ss) in by {
            let f = |k: &str| ss.iter().filter_map(|s| s["model"][k].as_f64()).collect::<Vec<f64>>();
            let decisions: u64 = ss.iter().filter_map(|s| s["model"]["decisions"].as_u64()).sum();
            let mut rejected: BTreeMap<String, u64> = BTreeMap::new();
            let mut pre: BTreeMap<String, u32> = BTreeMap::new();
            for s in &ss {
                if let Some(o) = s["model"]["rejected"].as_object() {
                    for (k, v) in o {
                        *rejected.entry(k.clone()).or_default() += v.as_u64().unwrap_or(0);
                    }
                }
                *pre.entry(s["model"]["map_precompute"].as_str().unwrap_or("?").to_string()).or_default() += 1;
            }
            let p95 = f("decide_ms_p95").into_iter().fold(0.0f64, f64::max);
            writeln!(
                md,
                "| `{policy}` | {} | {decisions} | {:.1} | {p95:.1} | {:.1} | {:.2} | {:.2} | {rejected:?} | {pre:?} |",
                ss.len(),
                median(&mut f("decide_ms_p50")),
                median(&mut f("run_ms_p50")),
                median(&mut f("mean_batch")),
                median(&mut f("host_ms_p50")),
            )?;
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wilson_interval_050_r2() {
        let (lo, hi) = wilson(25, 50);
        assert!((lo - 36.6).abs() < 0.1 && (hi - 63.4).abs() < 0.1, "{lo} {hi}");
        assert_eq!(wilson(0, 0), (0.0, 100.0));
        let (lo, hi) = wilson(50, 50);
        assert!(lo > 92.0 && hi == 100.0, "{lo} {hi}");
    }
}
