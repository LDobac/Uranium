use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;

use anyhow::Result;
use serde_json::Value;

use crate::plan::{Plan, Side};

#[derive(Clone, Debug, PartialEq)]
pub struct PolicyInfo {
    pub id: String,
    pub core: String,
    pub run: String,
}

fn model_info(side: &Side) -> Option<PolicyInfo> {
    let Side::Model { checkpoint, sample, .. } = side else { return None };
    match read_model_info(Path::new(checkpoint), *sample) {
        Ok(i) => Some(i),
        Err(e) => {
            elog!("[env] WARNING: summary: model checkpoint {checkpoint} unreadable ({e:#}); its matchup is reported without per-policy rows");
            None
        }
    }
}

fn read_model_info(checkpoint: &Path, sample: uranium_agent::infer::SampleMode) -> Result<PolicyInfo> {
    use uranium_agent::infer::{checkpoint_core, checkpoint_id, model_policy_label, resolve_checkpoint};
    let dir = resolve_checkpoint(checkpoint)?;
    let name = |p: &Path| p.file_name().map(|n| n.to_string_lossy().into_owned());
    let leaf = name(&dir).ok_or_else(|| anyhow::anyhow!("no folder name in {}", dir.display()))?;
    let run = match dir.parent().and_then(name) {
        Some(parent) => format!("{parent}/{leaf}"),
        None => leaf,
    };
    Ok(PolicyInfo { id: model_policy_label(&checkpoint_id(&dir)?, sample), core: checkpoint_core(&dir)?, run })
}

fn lines(path: &Path) -> Result<Vec<Value>> {
    let text = std::fs::read_to_string(path).unwrap_or_default();
    Ok(text.lines().filter(|l| !l.trim().is_empty()).map(serde_json::from_str).collect::<Result<_, _>>()?)
}

fn side_name(s: &Side, info: Option<&PolicyInfo>) -> String {
    match info {
        Some(i) => format!("{} `{}`", policy_name(s, info), i.id),
        None => policy_name(s, info),
    }
}

fn policy_name(s: &Side, info: Option<&PolicyInfo>) -> String {
    match (s, info) {
        (Side::Model { sample, .. }, Some(i)) => format!("model {} `{}` ({sample:?})", i.core, i.run),
        _ => plan_side_name(s),
    }
}

fn plan_side_name(s: &Side) -> String {
    match s {
        Side::T0 { race, build, .. } if *build != Default::default() => format!("T0 {race:?} `{}`", build.name()),
        Side::T0 { race, .. } => format!("T0 {race:?}"),
        Side::Oss { name, race } => format!("{name} ({race:?})"),
        Side::Model { checkpoint, sample, .. } => format!("model `{checkpoint}` ({sample:?})"),
    }
}

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
    decided_mirror: u32,
    violations: u64,
    session_errors: u32,
    minutes: Vec<f64>,
    suspect_short: u32,
    winner_disagree: u32,
    dropped: u32,
    opponents: std::collections::BTreeSet<String>,
}

fn decisive<'a>(ss: &[&'a Value]) -> Option<&'a Value> {
    let frame = |s: &Value| s["end_frame"].as_u64().unwrap_or(0);
    let first = *ss.first()?;
    Some(ss.iter().skip(1).fold(first, |best, s| if frame(s) > frame(best) { *s } else { best }))
}

fn mutual_leave(ss: &[&Value]) -> bool {
    ss.len() == 2 && ss.iter().all(|s| s["reason"] == "Surrender" && !s["winner"].is_null() && s["winner"] == s["slot"])
}

pub fn write(out: &Path, plan: &Plan) -> Result<()> {
    let infos: Vec<[Option<PolicyInfo>; 2]> = plan.matchups.iter().map(|m| [model_info(&m.p0), model_info(&m.p1)]).collect();
    write_with(out, plan, &infos)
}

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
            Some(w) if has_model => {
                let winner_is_model = ss.iter().find(|s| s["slot"].as_u64() == Some(w)).and_then(|s| s["policy"].as_str()).map(|p| p.starts_with("model:"));
                let p0_model = m.p0.model().is_some();
                let winner_policy = ss.iter().find(|s| s["slot"].as_u64() == Some(w)).and_then(|s| s["policy"].as_str());
                let ids = [&infos[mi][0], &infos[mi][1]].map(|i| i.as_ref().map(|i| i.id.as_str()));
                match winner_is_model {
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
                Side::T0 { .. } => {
                    let mirror = ss.len() == 2 && ss[0]["self_race"] == ss[1]["self_race"];
                    let winner_race = ss.iter().find(|s| s["slot"].as_u64() == Some(w)).and_then(|s| s["self_race"].as_str());
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
    let labels: BTreeMap<&str, &PolicyInfo> = infos.iter().flatten().flatten().map(|i| (i.id.as_str(), i)).collect();
    let model: Vec<&Value> = sessions.iter().filter(|s| s["model"].is_object()).collect();
    if !model.is_empty() {
        let mut by: BTreeMap<String, Vec<&Value>> = BTreeMap::new();
        for s in &model {
            by.entry(s["policy"].as_str().unwrap_or("?").to_string()).or_default().push(s);
        }
        writeln!(md, "\n## Model sessions\n")?;
        writeln!(md, "| policy | core | run | sessions | decisions | decide ms p50 (median) | decide ms p95 (max) | run ms p50 (median) | mean batch | host ms p50 | rejected | map precompute | surrender | surrender held | held & won |")?;
        writeln!(md, "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")?;
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
            let (modes, held, won) = surrender_cells(&ss);
            writeln!(
                md,
                "| `{policy}` | {} | {} | {} | {decisions} | {:.1} | {p95:.1} | {:.1} | {:.2} | {:.2} | {rejected:?} | {pre:?} | {modes} | {held} | {won} |",
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
    scripted_section(&mut md, &games, &by_game)?;
    let aborted: Vec<&Value> = games.iter().filter(|g| g["outcome"] == "Aborted").collect();
    if !aborted.is_empty() {
        writeln!(md, "\n## Aborted games\n")?;
        for g in aborted.iter().take(20) {
            writeln!(md, "- game {}: {}", g["game_id"], g["detail"].as_str().unwrap_or(""))?;
        }
    }
    std::fs::write(out.join("summary.md"), &md)?;
    uranium_agent::summary::summarize(&out.join("results.jsonl"), &out.join("sessions.md"), &plan.name)?;
    Ok(())
}

fn scripted_section(md: &mut String, games: &[Value], by_game: &BTreeMap<u64, Vec<&Value>>) -> Result<()> {
    #[derive(Default)]
    struct Cell {
        sessions: u32,
        held: u32,
        won: u32,
        bunker_4m: u32,
        preemptive: u32,
        evidence: u32,
        expand_4m: u32,
        scv8: Vec<f64>,
        judged: (u32, u32, u32, u32),
        early_wins: u32,
        rules: BTreeMap<String, u32>,
        expand: Vec<f64>,
    }
    let mut cells: BTreeMap<(u64, String, String), Cell> = BTreeMap::new();
    for g in games {
        let Some(ss) = by_game.get(&g["game_id"].as_u64().unwrap_or(0)) else { continue };
        for s in ss.iter().filter(|s| s["scripted"].is_object()) {
            let key = (g["matchup"].as_u64().unwrap_or(0), s["t0_build"].as_str().unwrap_or("?").to_string(), g["map"].as_str().unwrap_or("?").to_string());
            let c = cells.entry(key).or_default();
            let m = &s["scripted"];
            c.sessions += 1;
            c.held += u32::from(m["scv_min_1m"].as_u64().is_some_and(|v| v > 4));
            c.won += u32::from(s["won"].as_bool() == Some(true));
            let bunker = m["bunker_s"].as_f64().filter(|v| *v < 240.0);
            c.bunker_4m += u32::from(bunker.is_some());
            let after = |k: &str| m["react"]["rules_s"][k].as_f64();
            c.preemptive += u32::from(bunker.is_some_and(|b| ["army", "one_base"].iter().all(|k| after(k).is_none_or(|a| a > b))));
            c.evidence += u32::from(["pool_start", "pool_done", "lings", "drones", "one_base"].iter().any(|k| m["react"]["rules_s"][k].is_number()));
            c.expand_4m += u32::from(m["react"]["expand_s"].as_f64().is_some_and(|v| v <= 240.0));
            let early = s["end_frame"].as_u64().is_some_and(|f| f < 11_520);
            match (m["scv_8m"].as_f64(), s["won"].as_bool()) {
                (Some(v), _) => {
                    c.scv8.push(v);
                    let j = c.judged;
                    c.judged = (j.0 + u32::from(v >= 16.0), j.1 + u32::from(v >= 14.0), j.2 + u32::from(v >= 8.0), j.3 + 1);
                }
                (None, Some(true)) if early => {
                    c.early_wins += 1;
                    let j = c.judged;
                    c.judged = (j.0 + 1, j.1 + 1, j.2 + 1, j.3 + 1);
                }
                (None, Some(false)) if early => c.judged.3 += 1,
                _ => {}
            }
            match m["react"]["rush_rule"].as_str() {
                Some(r) => *c.rules.entry(r.to_string()).or_default() += 1,
                None if m["react"].is_object() => *c.rules.entry("none".into()).or_default() += 1,
                None => {}
            }
            if let Some(v) = m["react"]["expand_s"].as_f64() {
                c.expand.push(v);
            }
        }
    }
    if cells.is_empty() {
        return Ok(());
    }
    writeln!(md, "\n## T0 Terran sessions (050-R8)\n")?;
    writeln!(md, "| # | build | map | sessions | held (scv_min_1m > 4) | won | bunker < 4:00 | pre-emptive bunker | ①②④′④″ held | expand ≤ 4:00 | scv_8m ≥ 16 / ≥ 14 / ≥ 8 / judged (early wins) | scv_8m p50 | first rule (021-R8) | expand s p50 |")?;
    writeln!(md, "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")?;
    for ((mi, build, map), mut c) in cells {
        let (scv16, scv14, scv8, judged) = c.judged;
        let early_wins = c.early_wins;
        let (scv_p50, exp_p50) = (median(&mut c.scv8), median(&mut c.expand));
        let fmt = |v: f64| if v.is_nan() { "-".to_string() } else { format!("{v:.0}") };
        writeln!(
            md,
            "| {mi} | {build} | {map} | {} | {} | {} | {} | {} | {} | {} | {scv16} / {scv14} / {scv8} / {judged} ({early_wins}) | {} | {:?} | {} |",
            c.sessions, c.held, c.won, c.bunker_4m, c.preemptive, c.evidence, c.expand_4m, fmt(scv_p50), c.rules, fmt(exp_p50)
        )?;
    }
    Ok(())
}

fn surrender_cells(ss: &[&Value]) -> (String, u32, String) {
    let modes: std::collections::BTreeSet<&str> = ss.iter().filter_map(|s| s["model"]["surrender"].as_str()).collect();
    let held: Vec<&&Value> = ss.iter().filter(|s| s["model"]["surrender_frame"].is_u64()).collect();
    let won = held.iter().filter(|s| s["winner"].as_u64().is_some() && s["winner"].as_u64() == s["slot"].as_u64()).count();
    let modes = if modes.is_empty() { "-".to_string() } else { modes.into_iter().collect::<Vec<_>>().join(", ") };
    let won = if held.is_empty() { "-".to_string() } else { format!("{won} ({:.0}%)", 100.0 * won as f64 / held.len() as f64) };
    (modes, held.len() as u32, won)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn surrender_cells_count_held_and_won_020_r14() {
        let s = |slot: u64, winner: Option<u64>, frame: Option<u32>| serde_json::json!({"slot": slot, "winner": winner, "model": {"surrender": "shadow", "surrender_frame": frame}});
        let v = [s(0, Some(0), Some(20_000)), s(0, Some(1), Some(15_000)), s(1, Some(1), None), s(0, None, Some(30_000))];
        let refs: Vec<&Value> = v.iter().collect();
        assert_eq!(surrender_cells(&refs), ("shadow".to_string(), 3, "1 (33%)".to_string()));
        assert_eq!(surrender_cells(&[]), ("-".to_string(), 0, "-".to_string()));
    }

    #[test]
    fn scripted_section_050_r8() {
        let games = vec![serde_json::json!({"game_id": 1, "matchup": 0, "map": "fs"}), serde_json::json!({"game_id": 2, "matchup": 0, "map": "fs"})];
        let s1 = serde_json::json!({"game_id": 1, "won": true, "t0_build": "terran_react", "scripted": {"scv_min_1m": 9, "bunker_s": 150.0, "scv_8m": 18, "react": {"rush_rule": "pool_start", "expand_s": null, "rules_s": {"pool_start": 130.0, "army": 200.0}}}});
        let s2 = serde_json::json!({"game_id": 2, "won": false, "t0_build": "terran_react", "scripted": {"scv_min_1m": 4, "bunker_s": 230.0, "scv_8m": 15, "react": {"rush_rule": "army", "expand_s": 170.0, "rules_s": {"army": 220.0}}}});
        let z = serde_json::json!({"game_id": 2, "t0_build": "zerg_rush", "scripted": null});
        let w = serde_json::json!({"game_id": 3, "won": true, "end_frame": 10_000, "t0_build": "terran_react", "scripted": {"scv_min_1m": 9, "bunker_s": null, "scv_8m": null, "react": {"rush_rule": null, "expand_s": 200.0, "rules_s": {}}}});
        let l = serde_json::json!({"game_id": 4, "won": false, "end_frame": 8_000, "t0_build": "terran_react", "scripted": {"scv_min_1m": 2, "bunker_s": null, "scv_8m": null, "react": {"rush_rule": null, "expand_s": null, "rules_s": {}}}});
        let games2 = vec![serde_json::json!({"game_id": 3, "matchup": 1, "map": "cb"}), serde_json::json!({"game_id": 4, "matchup": 1, "map": "cb"})];
        let mut by2: BTreeMap<u64, Vec<&Value>> = BTreeMap::new();
        by2.insert(3, vec![&w]);
        by2.insert(4, vec![&l]);
        let mut md2 = String::new();
        scripted_section(&mut md2, &games2, &by2).unwrap();
        assert!(md2.contains("| 1 | terran_react | cb | 2 | 1 | 1 | 0 | 0 | 0 | 1 | 1 / 1 / 1 / 2 (1) | - | {\"none\": 2} | 200 |"), "{md2}");
        let o = serde_json::json!({"game_id": 5, "won": false, "t0_build": "terran_react", "scripted": {"scv_min_1m": 8, "bunker_s": 200.0, "scv_8m": 10, "react": {"rush_rule": "one_base", "expand_s": null, "rules_s": {"one_base": 190.0}}}});
        let o2 = serde_json::json!({"game_id": 6, "won": false, "t0_build": "terran_react", "scripted": {"scv_min_1m": 8, "bunker_s": 180.0, "scv_8m": 7, "react": {"rush_rule": "one_base", "expand_s": null, "rules_s": {"one_base": 190.0}}}});
        let games3 = vec![serde_json::json!({"game_id": 5, "matchup": 2, "map": "fs"}), serde_json::json!({"game_id": 6, "matchup": 2, "map": "fs"})];
        let mut by3: BTreeMap<u64, Vec<&Value>> = BTreeMap::new();
        by3.insert(5, vec![&o]);
        by3.insert(6, vec![&o2]);
        let mut md3 = String::new();
        scripted_section(&mut md3, &games3, &by3).unwrap();
        assert!(md3.contains("| 2 | terran_react | fs | 2 | 2 | 0 | 2 | 1 | 2 | 0 | 0 / 0 / 1 / 2 (0) | 10 | {\"one_base\": 2} | - |"), "{md3}");
        let mut by_game: BTreeMap<u64, Vec<&Value>> = BTreeMap::new();
        by_game.insert(1, vec![&s1]);
        by_game.insert(2, vec![&s2, &z]);
        let mut md = String::new();
        scripted_section(&mut md, &games, &by_game).unwrap();
        assert!(md.contains("| 0 | terran_react | fs | 2 | 1 | 1 | 2 | 1 | 1 | 1 | 1 / 2 / 2 / 2 (0) | 18 | {\"army\": 1, \"pool_start\": 1} | 170 |"), "{md}");
        let mut empty = String::new();
        scripted_section(&mut empty, &games, &BTreeMap::new()).unwrap();
        assert!(empty.is_empty());
    }

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
        let s = |g: u32, slot: u32, policy: &str, winner: Option<u32>| {
            serde_json::json!({"game_id": g, "slot": slot, "policy": policy, "winner": winner, "reason": "Elimination", "end_frame": 9000, "visibility_violations": 0, "error": null, "model": {"decisions": 10, "decide_ms_p50": 50.0, "decide_ms_p95": 90.0}}).to_string()
        };
        let results = [
            s(1, 0, &b.id, Some(0)),
            s(1, 1, &a.id, Some(0)),
            s(2, 0, &a.id, None),
            s(2, 1, &b.id, None),
            s(3, 0, &b.id, Some(1)),
            s(3, 1, &a.id, Some(1)),
        ];
        std::fs::write(dir.join("results.jsonl"), results.join("\n")).unwrap();
        write_with(&dir, &plan, &infos).unwrap();
        let md = std::fs::read_to_string(dir.join("summary.md")).unwrap();
        let (na, nb) = ("model lstm `run-a/ckpt-1` (Stochastic)", "model gtrxl `run-b/ckpt-1` (Stochastic)");
        assert!(md.contains(&format!("| 0 | {na} `{}` | {nb} `{}` | FightingSpirit | 2 | {{\"Finished\": 2}} | 0 | 1 | 1 | 0% [0, 66] | 0 |", a.id, b.id)), "{md}");
        assert!(md.contains(&format!("| 1 | {nb} `{}` | {na} `{}` | FightingSpirit | 1 | {{\"Finished\": 1}} | 0 | 1 | 0 |", b.id, a.id)), "{md}");
        assert!(md.contains(&format!("| {na} | {nb} | FightingSpirit | 3 | 1 | 1 | 1 | 33% [6, 79] | 0 |")), "{md}");
        assert!(md.contains(&format!("| {nb} | {na} | FightingSpirit | 3 | 1 | 1 | 1 | 33% [6, 79] | 0 |")), "{md}");
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
