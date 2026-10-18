//! Per-matchup summary (021-AC1/AC2, 020-AC1) joined from games.jsonl (hostd outcomes) and
//! results.jsonl (agent sessions).

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;

use anyhow::Result;
use serde_json::Value;

use crate::plan::{Plan, Side};

/// A model side as shown in summary.md (2026-10-18 user decision, 020 notes): Core kind, run id
/// and the session `policy` label (020-R5) the side's sessions report.
#[derive(Clone, Debug, PartialEq)]
pub struct PolicyInfo {
    /// Session `policy` label, `model:<folder>@<sha12>:<sample>`.
    pub id: String,
    /// `lstm` / `gtrxl`.
    pub core: String,
    /// `<run folder>/<checkpoint folder>`, e.g. `core-abl-lstm/ckpt-5300`.
    pub run: String,
}

/// Reads a model side's checkpoint for its display info; `None` for other sides or when the
/// checkpoint cannot be read (the report then shows the plan path only).
fn model_info(side: &Side) -> Option<PolicyInfo> {
    use uranium_agent::infer::{checkpoint_core, checkpoint_id, model_policy_label, resolve_checkpoint};
    let Side::Model { checkpoint, sample } = side else { return None };
    let dir = resolve_checkpoint(Path::new(checkpoint)).ok()?;
    let name = |p: &Path| p.file_name().map(|n| n.to_string_lossy().into_owned());
    let run = match dir.parent().and_then(name) {
        Some(parent) => format!("{parent}/{}", name(&dir)?),
        None => name(&dir)?,
    };
    Some(PolicyInfo { id: model_policy_label(&checkpoint_id(&dir).ok()?, *sample), core: checkpoint_core(&dir).ok()?, run })
}

fn lines(path: &Path) -> Result<Vec<Value>> {
    let text = std::fs::read_to_string(path).unwrap_or_default();
    Ok(text.lines().filter(|l| !l.trim().is_empty()).map(serde_json::from_str).collect::<Result<_, _>>()?)
}

/// Matchup table cell: the policy name plus, for a model side, its session policy label.
fn side_name(s: &Side, info: Option<&PolicyInfo>) -> String {
    match info {
        Some(i) => format!("{} `{}`", policy_name(s, info), i.id),
        None => policy_name(s, info),
    }
}

/// Policy name: model sides by Core kind and run id when the checkpoint was readable.
fn policy_name(s: &Side, info: Option<&PolicyInfo>) -> String {
    match (s, info) {
        (Side::Model { sample, .. }, Some(i)) => format!("model {} `{}` ({sample:?})", i.core, i.run),
        _ => plan_side_name(s),
    }
}

fn plan_side_name(s: &Side) -> String {
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
    /// Games whose two sessions report different winners; the later session decides (020-R13).
    winner_disagree: u32,
    /// Both sessions claimed the win because the other one left: a connection drop, Aborted and
    /// left out of the win counts (020-R13, 011-AC7).
    dropped: u32,
    /// Opponent IDs seen for p1 (021-R3).
    opponents: std::collections::BTreeSet<String>,
}

/// 020-R13: the session whose report decides the game: the one that ended last (largest
/// `end_frame`); on a tie the first record stands.
fn decisive<'a>(ss: &[&'a Value]) -> Option<&'a Value> {
    let frame = |s: &Value| s["end_frame"].as_u64().unwrap_or(0);
    let first = *ss.first()?;
    Some(ss.iter().skip(1).fold(first, |best, s| if frame(s) > frame(best) { *s } else { best }))
}

/// 020-R13: both sessions say the other one left and claim the win (a drop, not a GG: the
/// leaver of a real GG reports the other side as winner).
fn mutual_leave(ss: &[&Value]) -> bool {
    ss.len() == 2 && ss.iter().all(|s| s["reason"] == "Surrender" && !s["winner"].is_null() && s["winner"] == s["slot"])
}

pub fn write(out: &Path, plan: &Plan) -> Result<()> {
    let infos: Vec<[Option<PolicyInfo>; 2]> = plan.matchups.iter().map(|m| [model_info(&m.p0), model_info(&m.p1)]).collect();
    write_with(out, plan, &infos)
}

/// `write` with the model sides' display info given per matchup (`[p0, p1]`).
pub fn write_with(out: &Path, plan: &Plan, infos: &[[Option<PolicyInfo>; 2]]) -> Result<()> {
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
        let Some(first) = decisive(&ss) else { continue };
        if ss.len() == 2 && ss[0]["winner"] != ss[1]["winner"] {
            row.winner_disagree += 1;
        }
        if mutual_leave(&ss) {
            row.dropped += 1;
            continue;
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
                let winner_policy = ss.iter().find(|s| s["slot"].as_u64() == Some(w)).and_then(|s| s["policy"].as_str());
                let ids = [&infos[mi][0], &infos[mi][1]].map(|i| i.as_ref().map(|i| i.id.as_str()));
                match winner_is_model {
                    // Model vs model with two different policies (032-R17 head-to-head): the
                    // winner session's policy label tells the side.
                    Some(true) if p0_model && m.p1.model().is_some() => match (ids, winner_policy) {
                        ([Some(a), Some(b)], Some(wp)) if a != b && wp == a => row.p0_wins += 1,
                        ([Some(a), Some(b)], Some(wp)) if a != b && wp == b => row.p1_wins += 1,
                        _ => row.decided_mirror += 1,
                    },
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
    writeln!(md, "| # | p0 | p1 | map | games | outcomes | p0 wins | p1 wins | draws | p0 win% [95% CI] | mirror decided | winner disagree | dropped (Aborted) | suspect (<300 frames) | game min p50 | vis. violations | session errors |")?;
    writeln!(md, "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")?;
    // Per policy, slot swaps and maps combined: (policy, opponent) -> map -> [W, L, D, dropped].
    let mut per: BTreeMap<(String, String), BTreeMap<String, [u32; 4]>> = BTreeMap::new();
    for ((mi, map), r) in &mut rows {
        let m = &plan.matchups[*mi];
        let (n0, n1) = (policy_name(&m.p0, infos[*mi][0].as_ref()), policy_name(&m.p1, infos[*mi][1].as_ref()));
        if n0 != n1 {
            for (me, opp, w, l) in [(&n0, &n1, r.p0_wins, r.p1_wins), (&n1, &n0, r.p1_wins, r.p0_wins)] {
                let c = per.entry((me.clone(), opp.clone())).or_default().entry(map.clone()).or_default();
                c[0] += w;
                c[1] += l;
                c[2] += r.draws;
                c[3] += r.dropped;
            }
        }
        r.minutes.sort_by(f64::total_cmp);
        let p50 = r.minutes.get(r.minutes.len() / 2).copied().unwrap_or(f64::NAN);
        let p1_name = side_name(&m.p1, infos[*mi][1].as_ref());
        let mut p1 = if m.p1_random_ratio > 0.0 { format!("{p1_name} (random {:.0}%)", m.p1_random_ratio * 100.0) } else { p1_name };
        if !r.opponents.is_empty() && infos[*mi][1].is_none() {
            p1 = format!("{p1} `{}`", r.opponents.iter().cloned().collect::<Vec<_>>().join(", "));
        }
        // 050-R2: decided + drawn games count, draws as non-wins.
        let n = r.p0_wins + r.p1_wins + r.draws;
        let (lo, hi) = wilson(r.p0_wins, n);
        let rate = if n > 0 { format!("{:.0}% [{lo:.0}, {hi:.0}]", 100.0 * r.p0_wins as f64 / n as f64) } else { "-".into() };
        writeln!(
            md,
            "| {mi} | {} | {p1} | {map} | {} | {:?} | {} | {} | {} | {rate} | {} | {} | {} | {} | {p50:.1} | {} | {} |",
            side_name(&m.p0, infos[*mi][0].as_ref()),
            r.games,
            r.outcomes,
            r.p0_wins,
            r.p1_wins,
            r.draws,
            r.decided_mirror,
            r.winner_disagree,
            r.dropped,
            r.suspect_short,
            r.violations,
            r.session_errors
        )?;
    }
    // Per-policy totals over slot swaps and maps (2026-10-18 user decision, 050-R2 Wilson).
    if !per.is_empty() {
        writeln!(md, "\n## Per policy (slot swaps combined)\n")?;
        writeln!(md, "| policy | opponent | map | games | wins | losses | draws | win% [95% CI] | dropped (Aborted) |")?;
        writeln!(md, "|---|---|---|---|---|---|---|---|---|")?;
        for ((me, opp), maps) in &per {
            let mut lines: Vec<(String, [u32; 4])> = maps.iter().map(|(k, v)| (k.clone(), *v)).collect();
            if lines.len() > 1 {
                let all = lines.iter().fold([0u32; 4], |a, (_, c)| std::array::from_fn(|i| a[i] + c[i]));
                lines.push(("all".into(), all));
            }
            for (map, [w, l, d, dropped]) in lines {
                let n = w + l + d;
                let (lo, hi) = wilson(w, n);
                let rate = if n > 0 { format!("{:.0}% [{lo:.0}, {hi:.0}]", 100.0 * w as f64 / n as f64) } else { "-".into() };
                writeln!(md, "| {me} | {opp} | {map} | {n} | {w} | {l} | {d} | {rate} | {dropped} |")?;
            }
        }
    }
    // Core kind and run id of each model policy label.
    let labels: BTreeMap<&str, &PolicyInfo> = infos.iter().flatten().flatten().map(|i| (i.id.as_str(), i)).collect();
    // 020-R5 / 031-R4: model sessions (policy version, latency, executor rejections).
    let model: Vec<&Value> = sessions.iter().filter(|s| s["model"].is_object()).collect();
    if !model.is_empty() {
        let mut by: BTreeMap<String, Vec<&Value>> = BTreeMap::new();
        for s in &model {
            by.entry(s["policy"].as_str().unwrap_or("?").to_string()).or_default().push(s);
        }
        writeln!(md, "\n## Model sessions\n")?;
        writeln!(md, "| policy | core | run | sessions | decisions | decide ms p50 (median) | decide ms p95 (max) | run ms p50 (median) | mean batch | host ms p50 | rejected | map precompute |")?;
        writeln!(md, "|---|---|---|---|---|---|---|---|---|---|---|---|")?;
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
                "| `{policy}` | {} | {} | {} | {decisions} | {:.1} | {p95:.1} | {:.1} | {:.2} | {:.2} | {rejected:?} | {pre:?} |",
                labels.get(policy.as_str()).map_or("-", |i| i.core.as_str()),
                labels.get(policy.as_str()).map_or("-".to_string(), |i| format!("`{}`", i.run)),
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

    /// 020-R13: the report of the session that ended last decides a disagreement.
    #[test]
    fn decisive_session_is_the_later_one_020_r13() {
        let a: Value = serde_json::json!({"slot": 0, "winner": null, "end_frame": 8999});
        let b: Value = serde_json::json!({"slot": 1, "winner": 1, "end_frame": 9001});
        assert_eq!(decisive(&[&a, &b]).unwrap()["winner"], 1);
        assert_eq!(decisive(&[&b, &a]).unwrap()["winner"], 1);
        let c: Value = serde_json::json!({"slot": 1, "winner": 1, "end_frame": 8999});
        assert_eq!(decisive(&[&a, &c]).unwrap()["slot"], 0);
        assert!(decisive(&[]).is_none());
    }

    /// 020-R13: both sessions claiming the win because the other left is a drop.
    #[test]
    fn mutual_leave_is_a_drop_020_r13() {
        let a: Value = serde_json::json!({"slot": 0, "winner": 0, "reason": "Surrender", "end_frame": 1615});
        let b: Value = serde_json::json!({"slot": 1, "winner": 1, "reason": "Surrender", "end_frame": 1615});
        assert!(mutual_leave(&[&a, &b]));
        let gg: Value = serde_json::json!({"slot": 1, "winner": 0, "reason": "Elimination", "end_frame": 1600});
        assert!(!mutual_leave(&[&a, &gg]));
        let elim: Value = serde_json::json!({"slot": 1, "winner": 1, "reason": "Elimination", "end_frame": 8031});
        assert!(!mutual_leave(&[&a, &elim]));
        assert!(!mutual_leave(&[&a]));
    }

    /// 032-R17 head-to-head summary (2026-10-18 user decision): both sides show core, run id and
    /// policy label; wins of a model-vs-model game go to the side whose policy won; the slot-swapped
    /// matchups add up per policy with a Wilson interval (050-R2).
    #[test]
    fn model_vs_model_summary_per_policy_032_r17() {
        let dir = std::env::temp_dir().join(format!("uranium-report-h2h-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let plan: Plan = toml::from_str(
            "name='h2h'\n[[matchup]]\nmap='fighting_spirit'\ngames=2\np0={kind='model',checkpoint='checkpoints/run-a/ckpt-1',sample='stochastic'}\np1={kind='model',checkpoint='checkpoints/run-b/ckpt-1',sample='stochastic'}\n[[matchup]]\nmap='fighting_spirit'\ngames=1\np0={kind='model',checkpoint='checkpoints/run-b/ckpt-1',sample='stochastic'}\np1={kind='model',checkpoint='checkpoints/run-a/ckpt-1',sample='stochastic'}",
        )
        .unwrap();
        let a = PolicyInfo { id: "model:ckpt-1@aaaaaaaaaaaa:stochastic".into(), core: "lstm".into(), run: "run-a/ckpt-1".into() };
        let b = PolicyInfo { id: "model:ckpt-1@bbbbbbbbbbbb:stochastic".into(), core: "gtrxl".into(), run: "run-b/ckpt-1".into() };
        let infos = vec![[Some(a.clone()), Some(b.clone())], [Some(b.clone()), Some(a.clone())]];
        let game = |id: u32, mi: u32| serde_json::json!({"game_id": id, "matchup": mi, "map": "FightingSpirit", "outcome": "Finished", "opponent": "x"}).to_string();
        std::fs::write(dir.join("games.jsonl"), [game(1, 0), game(2, 0), game(3, 1)].join("\n")).unwrap();
        // The BWAPI slot of a policy need not follow the plan side (020 notes): game 1 has B on slot 0.
        let s = |g: u32, slot: u32, policy: &str, winner: Option<u32>| {
            serde_json::json!({"game_id": g, "slot": slot, "policy": policy, "winner": winner, "reason": "Elimination", "end_frame": 9000, "visibility_violations": 0, "error": null, "model": {"decisions": 10, "decide_ms_p50": 50.0, "decide_ms_p95": 90.0}}).to_string()
        };
        let results = [
            s(1, 0, &b.id, Some(0)), // B (p1) wins
            s(1, 1, &a.id, Some(0)),
            s(2, 0, &a.id, None), // draw
            s(2, 1, &b.id, None),
            s(3, 0, &b.id, Some(1)), // A (p1 of the swapped matchup) wins
            s(3, 1, &a.id, Some(1)),
        ];
        std::fs::write(dir.join("results.jsonl"), results.join("\n")).unwrap();
        write_with(&dir, &plan, &infos).unwrap();
        let md = std::fs::read_to_string(dir.join("summary.md")).unwrap();
        let (na, nb) = ("model lstm `run-a/ckpt-1` (Stochastic)", "model gtrxl `run-b/ckpt-1` (Stochastic)");
        // Matchup rows: both sides named with their policy label; p0/p1 wins, no mirror count.
        assert!(md.contains(&format!("| 0 | {na} `{}` | {nb} `{}` | FightingSpirit | 2 | {{\"Finished\": 2}} | 0 | 1 | 1 | 0% [0, 66] | 0 |", a.id, b.id)), "{md}");
        assert!(md.contains(&format!("| 1 | {nb} `{}` | {na} `{}` | FightingSpirit | 1 | {{\"Finished\": 1}} | 0 | 1 | 0 |", b.id, a.id)), "{md}");
        // Per policy: A won 1, lost 1, drew 1 over both slot orders.
        assert!(md.contains(&format!("| {na} | {nb} | FightingSpirit | 3 | 1 | 1 | 1 | 33% [6, 79] | 0 |")), "{md}");
        assert!(md.contains(&format!("| {nb} | {na} | FightingSpirit | 3 | 1 | 1 | 1 | 33% [6, 79] | 0 |")), "{md}");
        // Model sessions carry core and run.
        assert!(md.contains(&format!("| `{}` | lstm | `run-a/ckpt-1` | 3 |", a.id)), "{md}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn wilson_interval_050_r2() {
        let (lo, hi) = wilson(25, 50);
        assert!((lo - 36.6).abs() < 0.1 && (hi - 63.4).abs() < 0.1, "{lo} {hi}");
        assert_eq!(wilson(0, 0), (0.0, 100.0));
        let (lo, hi) = wilson(50, 50);
        assert!(lo > 92.0 && hi == 100.0, "{lo} {hi}");
    }
}
