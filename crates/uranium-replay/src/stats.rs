//! Dataset statistics (040-R6, 040-AC2, 040-AC4; C-4): `reports/dataset_stats.md` from `build`
//! records, and shard GameInfo checks (B-12). Counts only, no file or player names (040-R14).

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::PathBuf;

use anyhow::{Context, Result};
use serde_json::Value;

use crate::split::{Split, split_for};

/// B-12: a shard game's GameInfo has exactly one `is_enemy` player (011-R10).
pub fn one_enemy(gi: &uranium_proto::GameInfo) -> bool {
    gi.players.iter().filter(|p| p.is_enemy).count() == 1
}

/// Games (distinct `replay_sha256`) and views (written perspectives, 040-R2) of one group.
#[derive(Default)]
struct Count {
    games: BTreeSet<String>,
    views: usize,
    steps: u64,
}

impl Count {
    fn add(&mut self, sha: &str, steps: u64) {
        self.games.insert(sha.to_string());
        self.views += 1;
        self.steps += steps;
    }
}

const MATCHUPS: [&str; 3] = ["TvT", "TvP", "TvZ"];
const OUTCOMES: [&str; 6] = ["written", "desync", "filtered", "unplayable", "error", "review"];

fn pct(n: usize, d: usize) -> String {
    format!("{:.1}%", n as f64 * 100.0 / d.max(1) as f64)
}

fn split_name(s: Split) -> &'static str {
    match s {
        Split::Train => "train",
        Split::Val => "val",
        Split::Test => "test",
    }
}

/// Markdown report from `build` records (JSON lines).
pub fn dataset_markdown(records: &[PathBuf]) -> Result<String> {
    let mut rows: Vec<Value> = Vec::new();
    for p in records {
        for line in std::fs::read_to_string(p).with_context(|| format!("reading {}", p.display()))?.lines().filter(|l| !l.trim().is_empty()) {
            rows.push(serde_json::from_str(line).with_context(|| format!("parsing {}", p.display()))?);
        }
    }
    // A re-run can record the same view twice (same replay and perspective, or the same rejected
    // replay): the last record counts. The key includes the source (R-5): a replay skipped as
    // "duplicate of another source" is a different record from that source's own. A 040-R20
    // record (a later copy of the same file) is kept apart from the first file's record too.
    let before = rows.len();
    let mut last: BTreeMap<(String, String, Option<u64>, bool), usize> = BTreeMap::new();
    for (i, r) in rows.iter().enumerate() {
        let copy = r["reason"].as_str().is_some_and(|s| s.starts_with(crate::pipeline::DUPLICATE_IN_SOURCE));
        last.insert((r["src"].as_str().unwrap_or("").to_string(), r["replay_sha256"].as_str().unwrap_or("").to_string(), r["view"]["perspective"].as_u64(), copy), i);
    }
    let keep: BTreeSet<usize> = last.into_values().collect();
    let rows: Vec<Value> = rows.into_iter().enumerate().filter(|(i, _)| keep.contains(i)).map(|(_, r)| r).collect();
    let duplicates = before - rows.len();
    let mut recorded: BTreeMap<String, usize> = BTreeMap::new();
    let s = |v: &Value, k: &str| v[k].as_str().unwrap_or("").to_string();
    // Build outcomes per src: records (views, or whole replays rejected before a view exists).
    let mut outcomes: BTreeMap<String, BTreeMap<String, usize>> = BTreeMap::new();
    // `filtered` records by reason group (040-R19 invalid labels among them) and src.
    let mut filtered: BTreeMap<(String, String), usize> = BTreeMap::new();
    let mut replays: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    // 040-AC2 desync rate = desync / (written + desync), per src and map family.
    let mut played: BTreeMap<(&'static str, String), (usize, usize)> = BTreeMap::new();
    let mut by_family_matchup: BTreeMap<(String, String), Count> = BTreeMap::new();
    let mut by_title: BTreeMap<String, (String, BTreeMap<String, Count>)> = BTreeMap::new();
    let mut by_src: BTreeMap<String, Count> = BTreeMap::new();
    let mut by_stage: BTreeMap<String, Count> = BTreeMap::new();
    let mut by_split: BTreeMap<(String, &'static str), Count> = BTreeMap::new();
    let mut by_source: BTreeMap<(String, i64), Count> = BTreeMap::new();
    let mut total = Count::default();
    let (mut cmds, mut labeled) = (0u64, 0u64);
    // 2026-10-19: labeler version (records without one are version 1) -> written views and
    // (commands, labeled, unknown) of their coverage (040 "Shard 포맷", mixed G-D1 rebuild).
    let mut by_labeler: BTreeMap<u64, (Count, [u64; 3])> = BTreeMap::new();
    for r in &rows {
        let (src, outcome, sha, family) = (s(r, "src"), s(r, "outcome"), s(r, "replay_sha256"), s(r, "map_family"));
        replays.entry(src.clone()).or_default().insert(sha.clone());
        if let Some(m) = r["recorded_winner"].as_str() {
            *recorded.entry(m.to_string()).or_default() += 1;
        }
        *outcomes.entry(src.clone()).or_default().entry(outcome.clone()).or_default() += 1;
        if outcome == "filtered" {
            *filtered.entry((filter_group(r["reason"].as_str().unwrap_or("")), src.clone())).or_default() += 1;
        }
        if outcome == "written" || outcome == "desync" {
            for key in [("전체", String::new()), ("src", src.clone()), ("map_family", family.clone())] {
                let d = played.entry(key).or_default();
                d.1 += 1;
                d.0 += usize::from(outcome == "desync");
            }
        }
        if outcome != "written" {
            continue;
        }
        let view = &r["view"];
        let (matchup, stage) = (s(view, "matchup"), s(view, "stage"));
        let steps = r["steps"].as_u64().unwrap_or(0);
        total.add(&sha, steps);
        by_family_matchup.entry((family.clone(), matchup.clone())).or_default().add(&sha, steps);
        let t = by_title.entry(s(r, "map_name")).or_insert_with(|| (family.clone(), BTreeMap::new()));
        t.1.entry(matchup.clone()).or_default().add(&sha, steps);
        by_src.entry(src.clone()).or_default().add(&sha, steps);
        by_stage.entry(stage).or_default().add(&sha, steps);
        by_split.entry((family.clone(), split_name(split_for(&sha, &family)))).or_default().add(&sha, steps);
        // Records written before schema 5 have no result: shown as "(none recorded)".
        let source = r["result_source"].as_str().unwrap_or("(none recorded)").to_string();
        by_source.entry((source, r["result"].as_i64().unwrap_or(0))).or_default().add(&sha, steps);
        let cov = |k: &str| r["coverage"][k].as_u64().unwrap_or(0);
        cmds += cov("commands");
        labeled += cov("labeled");
        let lv = by_labeler.entry(r["labeler_version"].as_u64().unwrap_or(1)).or_default();
        lv.0.add(&sha, steps);
        for (i, k) in ["commands", "labeled", "unknown"].into_iter().enumerate() {
            lv.1[i] += cov(k);
        }
    }

    let mut md = String::new();
    writeln!(md, "# 데이터셋 통계 (040-R6, 040-AC2, 040-AC4)\n")?;
    writeln!(
        md,
        "`uranium-replay dataset-stats`로 build 기록(records {}개 파일)에서 생성. 개수만 기록한다(040-R14). **게임** = 원본 리플레이(`replay_sha256`) 고유 개수, **시점** = shard에 쓴 테란 시점(TvT는 게임당 2, 040-R2), **스텝** = 판단 스텝 수.\n",
        records.len()
    )?;
    writeln!(md, "- 저장: 게임 {} / 시점 {} / 스텝 {}", total.games.len(), total.views, total.steps)?;
    writeln!(md, "- 레이블 커버리지(031-AC3, 목표 ≥ 95%): 게임 영향 명령 {cmds}개 중 {labeled}개, {:.2}%", labeled as f64 * 100.0 / cmds.max(1) as f64)?;
    writeln!(md, "- 중복 기록(같은 리플레이·시점, 마지막 것만 셈): {duplicates}개 제외")?;
    writeln!(md, "- 기록 승자(040-R8, 기록 단위): 이름 일치 {}, 불일치 {}\n", recorded.get("matched").unwrap_or(&0), recorded.get("unmatched").unwrap_or(&0))?;

    writeln!(
        md,
        "## 레이블러 버전 (040 \"Shard 포맷\")\n\n버전 1(shard schema 5)은 선택을 `max_select`로 자르고 미분류 명령 시간을 NoOp으로 채웠다. 버전 2(schema 6)는 선택 전체·`unknown` 레이블(가중치 0)·`Unload_Unit`. 기록에 버전이 없으면 1이다.\n\n| 레이블러 | 게임 | 시점 | 스텝 | 게임 영향 명령 | 레이블 | unknown |\n|---|---|---|---|---|---|---|"
    )?;
    for (v, (c, n)) in &by_labeler {
        writeln!(md, "| {v} | {} | {} | {} | {} | {} | {} |", c.games.len(), c.views, c.steps, n[0], n[1], n[2])?;
    }
    writeln!(md)?;

    writeln!(md, "## 맵 패밀리 × 매치업 (040-AC4)\n\n| 맵 패밀리 | 매치업 | 게임 | 시점 | 스텝 |\n|---|---|---|---|---|")?;
    for ((f, m), c) in &by_family_matchup {
        writeln!(md, "| {f} | {m} | {} | {} | {} |", c.games.len(), c.views, c.steps)?;
    }
    let mut fam: BTreeMap<&str, Count> = BTreeMap::new();
    for ((f, _), c) in &by_family_matchup {
        let e = fam.entry(f).or_default();
        e.games.extend(c.games.iter().cloned());
        e.views += c.views;
        e.steps += c.steps;
    }
    writeln!(md, "\n## 맵 패밀리 분포 (040-R15)\n\n| 맵 패밀리 | 게임 | 시점 | 스텝 |\n|---|---|---|---|")?;
    for (f, c) in &fam {
        writeln!(md, "| {f} | {} | {} | {} |", c.games.len(), c.views, c.steps)?;
    }

    writeln!(md, "\n## 맵 제목별 (040-R6, 게임 / 시점)\n\n| 맵 제목 | 맵 패밀리 | TvT | TvP | TvZ | 합계 |\n|---|---|---|---|---|---|")?;
    let mut titles: Vec<(&String, &(String, BTreeMap<String, Count>))> = by_title.iter().collect();
    titles.sort_by_key(|(t, (_, m))| (std::cmp::Reverse(m.values().map(|c| c.views).sum::<usize>()), (*t).clone()));
    for (t, (f, m)) in titles {
        let cell = |k: &str| m.get(k).map_or_else(|| "0 / 0".to_string(), |c| format!("{} / {}", c.games.len(), c.views));
        let games: BTreeSet<&String> = m.values().flat_map(|c| c.games.iter()).collect();
        let views: usize = m.values().map(|c| c.views).sum();
        let title = if t.is_empty() { "(빈 제목)".to_string() } else { t.replace('|', "\\|") };
        writeln!(md, "| {title} | {f} | {} | {} | {} | {} / {views} |", cell(MATCHUPS[0]), cell(MATCHUPS[1]), cell(MATCHUPS[2]), games.len())?;
    }

    let group = |md: &mut String, title: &str, key: &str, m: &BTreeMap<String, Count>| -> Result<()> {
        writeln!(md, "\n## {title}\n\n| {key} | 게임 | 시점 | 스텝 |\n|---|---|---|---|")?;
        for (k, c) in m {
            writeln!(md, "| {k} | {} | {} | {} |", c.games.len(), c.views, c.steps)?;
        }
        Ok(())
    };
    group(&mut md, "src별", "src", &by_src)?;
    group(&mut md, "stage별 (040-R9, 기록용)", "stage", &by_stage)?;
    writeln!(md, "\n## 분할별 (041-R12)\n\n| 맵 패밀리 | 분할 | 게임 | 시점 | 스텝 |\n|---|---|---|---|---|")?;
    for ((f, sp), c) in &by_split {
        writeln!(md, "| {f} | {sp} | {} | {} | {} |", c.games.len(), c.views, c.steps)?;
    }

    writeln!(md, "\n## 승패 판정 근거 (040-R8 `result_source`, 시점 단위)\n\n| result_source | result | 시점 | 비율 |\n|---|---|---|---|")?;
    for ((src, res), c) in &by_source {
        writeln!(md, "| {src} | {res} | {} | {} |", c.views, pct(c.views, total.views))?;
    }

    writeln!(md, "\n## 디싱크 폐기율 (040-AC2, 목표 < 10%)\n\n폐기율 = `desync` ÷ (`written` + `desync`), 시점 단위.\n\n| 구분 | 값 | 재생 시점 | 디싱크 | 폐기율 |\n|---|---|---|---|---|")?;
    for ((kind, v), (d, t)) in &played {
        writeln!(md, "| {kind} | {v} | {t} | {d} | {} |", pct(*d, *t))?;
    }

    writeln!(md, "\n## build 결과 (040-R6, 기록 단위)\n\n시점이 정해진 뒤의 결과는 시점 단위, 그 전에 거른 리플레이(필터·중복·파싱 오류)는 리플레이 하나가 기록 하나다.\n")?;
    writeln!(md, "| src | 리플레이 | {} | 기타 |\n|---|---|{}---|", OUTCOMES.join(" | "), "---|".repeat(OUTCOMES.len()))?;
    for (src, m) in &outcomes {
        let cells: Vec<String> = OUTCOMES.iter().map(|o| m.get(*o).copied().unwrap_or(0).to_string()).collect();
        let other: usize = m.iter().filter(|(k, _)| !OUTCOMES.contains(&k.as_str())).map(|(_, n)| n).sum();
        writeln!(md, "| {src} | {} | {} | {other} |", replays.get(src).map_or(0, |r| r.len()), cells.join(" | "))?;
    }

    writeln!(md, "\n## 필터 사유 (`filtered`, 기록 단위)\n\n`{INVALID_LABEL_GROUP}`은 시점 단위(040-R19), 나머지는 대부분 리플레이 단위다.\n\n| 사유 | src | 기록 |\n|---|---|---|")?;
    for ((reason, src), n) in &filtered {
        writeln!(md, "| {} | {src} | {n} |", reason.replace('|', "\\|"))?;
    }
    Ok(md)
}

/// Group label of the 040-R19 filter (invalid labels ≥ 20%) in the statistics.
pub const INVALID_LABEL_GROUP: &str = "invalid_labels>=20% (040-R19)";

/// A `filtered` reason without its per-record details (ratio, map name), for counting.
pub fn filter_group(reason: &str) -> String {
    if reason.starts_with(uranium_env::features::label::INVALID_LABEL_PREFIX) {
        INVALID_LABEL_GROUP.to_string()
    } else if let Some((head, _)) = reason.split_once(": ") {
        head.to_string()
    } else {
        reason.to_string()
    }
}

/// 034-AC3 / AMB-040-6 / B-12 / 040-R20: label distribution and GameInfo checks over a shard
/// folder (subfolders included), as JSON. [`check_shard_stats`] turns the checks into an error.
pub fn shard_stats(dir: &std::path::Path) -> Result<Value> {
    use std::collections::BTreeMap;
    let table = uranium_env::actions::ActionTable::load()?;
    // Subfolders too (build writes `<src>/<part>/`), finished shards only (040-R17).
    fn walk(d: &std::path::Path, out: &mut Vec<PathBuf>) -> Result<()> {
        for e in std::fs::read_dir(d)? {
            let e = e?;
            if e.file_type()?.is_dir() {
                walk(&e.path(), out)?;
            } else if crate::shard::shard_index(&e.file_name().to_string_lossy()).is_some() {
                out.push(e.path());
            }
        }
        Ok(())
    }
    let mut files = Vec::new();
    walk(dir, &mut files)?;
    files.sort();
    // 040-R20: one game per (replay, perspective) in a source.
    let mut seen = std::collections::HashSet::<(String, u8)>::new();
    let mut duplicate_games = 0usize;
    let (mut games, mut steps, mut bad_game_info) = (0usize, 0usize, Vec::new());
    let (mut with_select, mut with_unit, mut with_tile, mut queued) = (0usize, 0usize, 0usize, 0usize);
    let mut actions = BTreeMap::<u16, usize>::new();
    let mut delays = BTreeMap::<u32, usize>::new();
    let mut matchups = BTreeMap::<String, usize>::new();
    let mut families = BTreeMap::<String, usize>::new();
    let mut result_sources = BTreeMap::<String, usize>::new();
    let (mut is_enemy_violations, mut enemy_random, mut not_game_start) = (0usize, 0usize, 0usize);
    // 2026-10-19: labeler version -> shards, games, steps, unknown labels, Unload_Unit labels,
    // labels whose selection is longer than the action's max_select (F12, version 2 only).
    let mut by_labeler = BTreeMap::<u32, [usize; 6]>::new();
    let unload = table.by_name("Unload_Unit").map(|a| a.id);
    for f in &files {
        let shard = crate::shard::read(f)?;
        let lv = by_labeler.entry(shard.header.labeler_version).or_default();
        lv[0] += 1;
        for g in &shard.games {
            lv[1] += 1;
            lv[2] += g.steps.len();
            for l in g.steps.iter().map(|s| &s.action_label) {
                lv[3] += usize::from(l.is_unknown());
                lv[4] += usize::from(Some(l.action) == unload);
                lv[5] += usize::from(table.actions.get(l.action as usize).is_some_and(|a| l.selected.len() > a.max_select as usize));
            }
            games += 1;
            duplicate_games += usize::from(!seen.insert((g.meta.replay_sha256.clone(), g.meta.perspective_slot)));
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
            if !crate::stats::one_enemy(gi) {
                is_enemy_violations += 1;
            }
            *result_sources.entry(g.meta.result_source.clone()).or_default() += 1;
            // 040-R18: the first step is the game-start observation.
            let start = crate::label::GAME_START_FRAME;
            if !g.steps.first().is_some_and(|s| s.raw_frame.frame == start && s.action_label.frame == start) {
                not_game_start += 1;
            }
            enemy_random += usize::from(g.meta.enemy_random);
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
    let name = |id: u16| if id == crate::label::UNKNOWN_ACTION { "Unknown".to_string() } else { table.actions.get(id as usize).map_or_else(|| format!("#{id}"), |a| a.name.clone()) };
    let labeler: serde_json::Map<String, serde_json::Value> = by_labeler
        .iter()
        .map(|(v, n)| (v.to_string(), serde_json::json!({ "shards": n[0], "games": n[1], "steps": n[2], "unknown": n[3], "unload_unit": n[4], "selected_over_max": n[5] })))
        .collect();
    let mut by_count: Vec<(String, usize)> = actions.iter().map(|(&a, &n)| (name(a), n)).collect();
    by_count.sort_by(|a, b| b.1.cmp(&a.1));
    let unseen: Vec<String> = table.actions.iter().filter(|a| !actions.contains_key(&a.id)).map(|a| a.name.clone()).collect();
    let pct = |n: usize| if steps == 0 { 0.0 } else { (n as f64 * 10000.0 / steps as f64).round() / 100.0 };
    Ok(serde_json::json!({
        "shards": files.len(), "games": games, "steps": steps,
        "labeler_versions": labeler,
        "game_info_bad": bad_game_info,
        "is_enemy_violations": is_enemy_violations,
        "duplicate_games": duplicate_games,
        "result_sources": result_sources, "enemy_random": enemy_random,
        "first_step_not_game_start": not_game_start,
        "matchups": matchups, "map_families": families,
        "pct_with_selected": pct(with_select), "pct_with_target_unit": pct(with_unit),
        "pct_with_target_tile": pct(with_tile), "pct_queued": pct(queued),
        "delay_buckets": delays,
        "actions": by_count.iter().map(|(n, c)| serde_json::json!([n, c, pct(*c)])).collect::<Vec<_>>(),
        "actions_unseen": unseen,
    }))
}

/// `shard-stats` failure rules: a game without exactly one `is_enemy` player (B-12) or a
/// (replay, perspective) that appears twice (040-R20).
pub fn check_shard_stats(s: &Value) -> Result<()> {
    let bad = s["is_enemy_violations"].as_u64().unwrap_or(0);
    anyhow::ensure!(bad == 0, "B-12: {bad} games without exactly one is_enemy player");
    let dup = s["duplicate_games"].as_u64().unwrap_or(0);
    anyhow::ensure!(dup == 0, "040-R20: {dup} games repeat a (replay, perspective) already in this folder");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn written(sha: &str, perspective: u8, family: &str, title: &str, matchup: &str, source: &str, result: i8) -> String {
        serde_json::json!({
            "index": 0, "replay_sha256": sha, "src": "stardata", "map_family": family, "map_name": title, "map_hash": "",
            "view": {"perspective": perspective, "stage": "S1", "apm": 100, "matchup": matchup, "enemy_random": false},
            "outcome": "written", "steps": 10, "coverage": {"commands": 5, "labeled": 5, "unknown": 0, "unknown_ops": [], "unknown_orders": []},
            "result": result, "result_source": source
        })
        .to_string()
    }

    /// 040-R6 / 040-AC2 / 040-AC4: games vs views, desync rate, result_source.
    #[test]
    fn dataset_stats_count_games_and_views_040_r6() {
        let dir = std::env::temp_dir().join(format!("uranium-stats-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("stardata_p00.jsonl");
        let a = "a".repeat(64);
        let lines = [
            written(&a, 0, "fighting_spirit", "투혼 1.3", "TvT", "bridge", 1),
            written(&a, 1, "fighting_spirit", "투혼 1.3", "TvT", "bridge", -1),
            // A re-run wrote the same view again: counted once.
            written(&a, 1, "fighting_spirit", "투혼 1.3", "TvT", "bridge", -1),
            written(&"b".repeat(64), 0, "other", "Python", "TvZ", "none", 0),
            // Labeler version 2 (2026-10-19): one written view with 2 unknown commands.
            serde_json::json!({"index": 6, "replay_sha256": "f".repeat(64), "src": "stardata", "map_family": "other", "map_name": "Python", "map_hash": "", "view": {"perspective": 0, "stage": "S1", "apm": 100, "matchup": "TvZ", "enemy_random": false}, "labeler_version": 2, "outcome": "written", "steps": 7, "coverage": {"commands": 9, "labeled": 7, "unknown": 2, "unknown_ops": [[41, 2]], "unknown_orders": []}, "result": 1, "result_source": "bridge"}).to_string(),
            serde_json::json!({"index": 3, "replay_sha256": "c".repeat(64), "src": "stardata", "map_family": "other", "map_name": "Python", "map_hash": "", "view": {"perspective": 0, "stage": "S1", "apm": 90, "matchup": "TvP", "enemy_random": false}, "outcome": "desync", "reasons": ["x"]}).to_string(),
            // 040-R19: two perspectives rejected for invalid labels, counted under one reason.
            serde_json::json!({"index": 4, "replay_sha256": "d".repeat(64), "src": "stardata", "map_family": "other", "map_name": "Python", "map_hash": "", "view": {"perspective": 0, "stage": "S1", "apm": 900, "matchup": "TvT", "enemy_random": false}, "outcome": "filtered", "reason": "invalid_labels>=20% (55.0%, 11/20) (040-R19)"}).to_string(),
            serde_json::json!({"index": 4, "replay_sha256": "d".repeat(64), "src": "stardata", "map_family": "other", "map_name": "Python", "map_hash": "", "view": {"perspective": 1, "stage": "S1", "apm": 950, "matchup": "TvT", "enemy_random": false}, "outcome": "filtered", "reason": "invalid_labels>=20% (24.6%, 1119/4550) (040-R19)"}).to_string(),
            serde_json::json!({"index": 5, "replay_sha256": "e".repeat(64), "src": "stardata", "map_family": "other", "map_name": "X", "map_hash": "", "view": null, "outcome": "filtered", "reason": "excluded map (040-R16): X"}).to_string(),
            // R-5: the same replay skipped by another source is that source's own record.
            serde_json::json!({"index": 0, "replay_sha256": "e".repeat(64), "src": "community", "map_family": "", "map_name": "", "map_hash": "", "view": null, "outcome": "filtered", "reason": "duplicate of another source"}).to_string(),
            // 040-R20: a later copy of the excluded-map file must not hide the first file's reason.
            serde_json::json!({"index": 9, "replay_sha256": "e".repeat(64), "src": "stardata", "map_family": "", "map_name": "", "map_hash": "", "view": null, "outcome": "filtered", "reason": format!("{}: same file as global index 5", crate::pipeline::DUPLICATE_IN_SOURCE)}).to_string(),
        ];
        std::fs::write(&p, lines.join("\n")).unwrap();
        let md = dataset_markdown(&[p]).unwrap();
        assert!(md.contains("| fighting_spirit | TvT | 1 | 2 | 20 |"), "{md}");
        assert!(md.contains("| 투혼 1.3 | fighting_spirit | 1 / 2 | 0 / 0 | 0 / 0 | 1 / 2 |"), "{md}");
        assert!(md.contains("| none | 0 | 1 | 25.0% |"), "{md}");
        assert!(md.contains("| 전체 |  | 5 | 1 | 20.0% |"), "{md}");
        assert!(md.contains("| map_family | other | 3 | 1 | 33.3% |"), "{md}");
        // Labeler versions: records without the field are version 1.
        assert!(md.contains("| 1 | 2 | 3 | 30 | 15 | 15 | 0 |"), "{md}");
        assert!(md.contains("| 2 | 1 | 1 | 7 | 9 | 7 | 2 |"), "{md}");
        assert!(md.contains("중복 기록(같은 리플레이·시점, 마지막 것만 셈): 1개 제외"), "{md}");
        assert!(md.contains("| invalid_labels>=20% (040-R19) | stardata | 2 |"), "{md}");
        assert!(md.contains("| excluded map (040-R16) | stardata | 1 |"), "{md}");
        assert!(md.contains("| duplicate of another source | community | 1 |"), "{md}");
        assert!(md.contains("| duplicate file in this source (040-R20) | stardata | 1 |"), "{md}");
        std::fs::remove_dir_all(&dir).ok();
    }

    fn info(enemies: usize) -> uranium_proto::GameInfo {
        let p = |id: u8, is_enemy: bool| uranium_proto::PlayerInfo { id, name: String::new(), race: uranium_proto::Race::Terran, start_tile: None, is_self: id == 0, is_enemy };
        let mut players = vec![p(0, false)];
        players.extend((0..enemies).map(|i| p(i as u8 + 1, true)));
        if enemies == 0 {
            players.push(p(1, false));
        }
        uranium_proto::GameInfo {
            map_name: String::new(),
            map_hash: [0; 32],
            map_size_tiles: (4, 4),
            self_player: 0,
            players,
            static_map: uranium_proto::StaticMap {
                width_tiles: 4,
                height_tiles: 4,
                walkable: uranium_proto::BitGrid::new(16, 16),
                buildable: uranium_proto::BitGrid::new(4, 4),
                ground_height: vec![0; 16],
                start_locations: vec![],
            },
            latency_frames: 0,
        }
    }

    /// 040-R20: `shard-stats` reads subfolders and fails when a (replay, perspective) appears
    /// twice, e.g. built by two parts.
    #[test]
    fn shard_stats_fail_on_a_repeated_game_040_r20() {
        use crate::shard::{ShardWriter, tests::game};
        let dir = std::env::temp_dir().join(format!("uranium-shardstats-dup-{}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        let g = |n: u32| {
            let mut g = game(n);
            g.game_info = info(1);
            g
        };
        let mut w = ShardWriter::new(dir.join("shards").join("p0"), dir.join("local"), 0, usize::MAX).unwrap();
        w.add(g(1)).unwrap();
        w.add(g(2)).unwrap();
        w.flush().unwrap();
        let s = shard_stats(&dir.join("shards")).unwrap();
        assert_eq!((s["games"].as_u64(), s["duplicate_games"].as_u64()), (Some(2), Some(0)));
        check_shard_stats(&s).unwrap();
        let mut w = ShardWriter::new(dir.join("shards").join("p1"), dir.join("local"), 0, usize::MAX).unwrap();
        w.add(g(1)).unwrap();
        w.flush().unwrap();
        let s = shard_stats(&dir.join("shards")).unwrap();
        assert_eq!(s["duplicate_games"].as_u64(), Some(1));
        let e = format!("{:#}", check_shard_stats(&s).unwrap_err());
        assert!(e.contains("040-R20"), "{e}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// B-12: exactly one is_enemy player.
    #[test]
    fn game_info_needs_exactly_one_enemy_b12() {
        assert!(one_enemy(&info(1)));
        assert!(!one_enemy(&info(0)));
        assert!(!one_enemy(&info(2)));
    }
}
