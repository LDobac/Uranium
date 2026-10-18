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
    let s = |v: &Value, k: &str| v[k].as_str().unwrap_or("").to_string();
    // Build outcomes per src: records (views, or whole replays rejected before a view exists).
    let mut outcomes: BTreeMap<String, BTreeMap<String, usize>> = BTreeMap::new();
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
    for r in &rows {
        let (src, outcome, sha, family) = (s(r, "src"), s(r, "outcome"), s(r, "replay_sha256"), s(r, "map_family"));
        replays.entry(src.clone()).or_default().insert(sha.clone());
        *outcomes.entry(src.clone()).or_default().entry(outcome.clone()).or_default() += 1;
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
        cmds += r["coverage"]["commands"].as_u64().unwrap_or(0);
        labeled += r["coverage"]["labeled"].as_u64().unwrap_or(0);
    }

    let mut md = String::new();
    writeln!(md, "# 데이터셋 통계 (040-R6, 040-AC2, 040-AC4)\n")?;
    writeln!(
        md,
        "`uranium-replay dataset-stats`로 build 기록(records {}개 파일)에서 생성. 개수만 기록한다(040-R14). **게임** = 원본 리플레이(`replay_sha256`) 고유 개수, **시점** = shard에 쓴 테란 시점(TvT는 게임당 2, 040-R2), **스텝** = 판단 스텝 수.\n",
        records.len()
    )?;
    writeln!(md, "- 저장: 게임 {} / 시점 {} / 스텝 {}", total.games.len(), total.views, total.steps)?;
    writeln!(md, "- 레이블 커버리지(031-AC3, 목표 ≥ 95%): 게임 영향 명령 {cmds}개 중 {labeled}개, {:.2}%\n", labeled as f64 * 100.0 / cmds.max(1) as f64)?;

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
    Ok(md)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn written(sha: &str, family: &str, title: &str, matchup: &str, source: &str, result: i8) -> String {
        serde_json::json!({
            "index": 0, "replay_sha256": sha, "src": "stardata", "map_family": family, "map_name": title, "map_hash": "",
            "view": {"perspective": 0, "stage": "S1", "apm": 100, "matchup": matchup, "enemy_random": false},
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
            written(&a, "fighting_spirit", "투혼 1.3", "TvT", "bridge", 1),
            written(&a, "fighting_spirit", "투혼 1.3", "TvT", "bridge", -1),
            written(&"b".repeat(64), "other", "Python", "TvZ", "none", 0),
            serde_json::json!({"index": 3, "replay_sha256": "c".repeat(64), "src": "stardata", "map_family": "other", "map_name": "Python", "map_hash": "", "view": {"perspective": 0, "stage": "S1", "apm": 90, "matchup": "TvP", "enemy_random": false}, "outcome": "desync", "reasons": ["x"]}).to_string(),
        ];
        std::fs::write(&p, lines.join("\n")).unwrap();
        let md = dataset_markdown(&[p]).unwrap();
        assert!(md.contains("| fighting_spirit | TvT | 1 | 2 | 20 |"), "{md}");
        assert!(md.contains("| 투혼 1.3 | fighting_spirit | 1 / 2 | 0 / 0 | 0 / 0 | 1 / 2 |"), "{md}");
        assert!(md.contains("| none | 0 | 1 | 33.3% |"), "{md}");
        assert!(md.contains("| 전체 |  | 4 | 1 | 25.0% |"), "{md}");
        assert!(md.contains("| map_family | other | 2 | 1 | 50.0% |"), "{md}");
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

    /// B-12: exactly one is_enemy player.
    #[test]
    fn game_info_needs_exactly_one_enemy_b12() {
        assert!(one_enemy(&info(1)));
        assert!(!one_enemy(&info(0)));
        assert!(!one_enemy(&info(2)));
    }
}
