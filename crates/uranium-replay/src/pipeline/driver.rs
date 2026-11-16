use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use anyhow::Result;
use sha2::{Digest, Sha256};
use crate::label::Labeler;
use crate::play::{PlayConfig, slot_of};
use crate::shard::{GameMeta, ShardGame, ShardWriter};
use crate::header;
use uranium_env::maps::MapFamilies;
use super::{DUPLICATE_IN_SOURCE, View, Outcome, Record, hex};
use super::excludes::{Duplicates, MapExcludes};
use super::filter::{filter, filter_rules, invalid_label_filter};
use super::view::{recorded_slot, ViewGame, ViewOptions, build_view, write_map_precompute};
use super::args::DISTILL_SRC;

pub struct BuildOptions {
    pub src: String,
    pub shards_dir: PathBuf,
    pub staging_dir: PathBuf,
    pub recorded_winners: HashMap<String, String>,
    pub jobs: usize,
    pub shard_target_bytes: usize,
    pub budget_bytes: u64,
    pub first_shard_index: u32,
    pub skip_sha256: std::collections::HashSet<String>,
    pub done: ResumeDone,
    pub duplicates: Duplicates,
    pub until_frame: Option<u32>,
    pub time_limit: Option<u32>,
    pub relabel: Option<uranium_scripted::T0Build>,
    pub memory_log: Option<crate::act_log::MemoryLogDir>,
    pub games: Option<crate::lobby::GamesLog>,
}

#[derive(Debug, Default, Clone)]
pub struct ResumeDone {
    pub whole: std::collections::HashSet<String>,
    pub views: std::collections::HashSet<(String, u8)>,
}

impl ResumeDone {
    pub fn add_line(&mut self, line: &str) {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else { return };
        let Some(sha) = v["replay_sha256"].as_str().filter(|s| !s.is_empty()) else { return };
        if v["reason"].as_str().is_some_and(|r| r.starts_with(DUPLICATE_IN_SOURCE)) {
            return;
        }
        match v["view"]["perspective"].as_u64() {
            Some(p) => {
                self.views.insert((sha.to_string(), p as u8));
            }
            None => {
                self.whole.insert(sha.to_string());
            }
        }
    }

    pub fn replay_done(&self, sha: &str) -> bool {
        self.whole.contains(sha)
    }

    pub fn remaining(&self, sha: &str, views: Vec<View>) -> Vec<View> {
        views.into_iter().filter(|v| !self.views.contains(&(sha.to_string(), v.perspective))).collect()
    }

    pub fn len(&self) -> usize {
        self.whole.len() + self.views.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

struct Pending {
    writer: ShardWriter,
    records: Vec<Record>,
}

fn add_game(p: &mut Pending, failure: &Mutex<Option<anyhow::Error>>, game: ShardGame, r: Record, sink: &(dyn Fn(&Record) + Sync)) -> bool {
    if failure.lock().unwrap().is_some() {
        return false;
    }
    let before = p.writer.written.len();
    if let Err(e) = p.writer.add(game) {
        p.records.clear();
        *failure.lock().unwrap() = Some(e);
        return false;
    }
    p.records.push(r);
    if p.writer.written.len() > before {
        for r in p.records.drain(..) {
            sink(&r);
        }
    }
    true
}

pub fn build(files: &[PathBuf], opt: &BuildOptions, cfg: &PlayConfig, sink: &(dyn Fn(&Record) + Sync)) -> Result<()> {
    let families = MapFamilies::load()?;
    let excludes = MapExcludes::load(&families)?;
    let labeler = Labeler::new()?;
    let pending = Mutex::new(Pending { writer: ShardWriter::new(opt.shards_dir.clone(), opt.staging_dir.clone(), opt.first_shard_index, opt.shard_target_bytes)?, records: Vec::new() });
    let queue = Mutex::new(files.iter().cloned().enumerate().collect::<VecDeque<_>>());
    let written_bytes = Mutex::new(0u64);
    let failure: Mutex<Option<anyhow::Error>> = Mutex::new(None);
    std::thread::scope(|s| {
        for w in 0..opt.jobs.max(1) {
            let (families, excludes, labeler, pending, queue, written_bytes, failure) = (&families, &excludes, &labeler, &pending, &queue, &written_bytes, &failure);
            let cfg = PlayConfig { work_dir: cfg.work_dir.join(format!("w{w}")), launcher: cfg.launcher.clone(), bridge_lib: cfg.bridge_lib.clone(), data_dir: cfg.data_dir.clone() };
            s.spawn(move || {
                while let Some((index, file)) = { let mut q = queue.lock().unwrap(); q.pop_front() } {
                    for (r, game) in process(index, &file, opt, &cfg, families, excludes, labeler, written_bytes) {
                        let Some(game) = game else {
                            sink(&r);
                            continue;
                        };
                        if !add_game(&mut pending.lock().unwrap(), failure, game, r, sink) {
                            queue.lock().unwrap().clear();
                            return;
                        }
                    }
                }
            });
        }
    });
    if let Some(e) = failure.into_inner().unwrap() {
        return Err(e);
    }
    let mut p = pending.into_inner().unwrap();
    p.writer.flush()?;
    for r in p.records.drain(..) {
        sink(&r);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn process(index: usize, file: &Path, opt: &BuildOptions, cfg: &PlayConfig, families: &MapFamilies, excludes: &MapExcludes, labeler: &Labeler, written: &Mutex<u64>) -> Vec<(Record, Option<ShardGame>)> {
    let map_hash_cell = std::cell::RefCell::new(String::new());
    let recorded_cell: std::cell::RefCell<Option<&'static str>> = std::cell::RefCell::new(None);
    let rec = |sha: &str, map: &str, family: &str, view: Option<View>, outcome: Outcome| {
        let r = Record {
            index,
            replay_sha256: sha.to_string(),
            src: opt.src.clone(),
            map_family: family.to_string(),
            map_name: map.chars().filter(|c| !c.is_control()).collect(),
            map_hash: map_hash_cell.borrow().clone(),
            recorded_winner: *recorded_cell.borrow(),
            view,
            labeler_version: crate::label::LABELER_VERSION,
            until_frame: opt.until_frame,
            time_limit: opt.time_limit,
            relabel: opt.relabel.map(|b| b.name()),
            outcome,
        };
        (r, None)
    };
    let bytes = match std::fs::read(file) {
        Ok(b) => b,
        Err(e) => return vec![rec("", "", "", None, Outcome::Error { reason: e.to_string() })],
    };
    let sha = hex(&Sha256::digest(&bytes));
    if opt.done.replay_done(&sha) {
        return Vec::new();
    }
    if let Some(first) = opt.duplicates.earlier_copy(index, &sha) {
        return vec![rec(&sha, "", "", None, Outcome::Filtered { reason: format!("{DUPLICATE_IN_SOURCE}: same file as global index {first}") })];
    }
    if opt.skip_sha256.contains(&sha) {
        return vec![rec(&sha, "", "", None, Outcome::Filtered { reason: "duplicate of another source".into() })];
    }
    let rep = match crate::parse(&bytes) {
        Ok(r) => r,
        Err(e) if format!("{e:#}").contains("unknown opcode") => return vec![rec(&sha, "", "", None, Outcome::Review { reason: format!("{e:#}") })],
        Err(e) => return vec![rec(&sha, "", "", None, Outcome::Error { reason: format!("{e:#}") })],
    };
    let map_hash = hex(&Sha256::digest(&rep.map));
    *map_hash_cell.borrow_mut() = map_hash.clone();
    let family = families.classify(&rep.header.map_name, &map_hash);
    if let Some(name) = excludes.matches(&rep.header.map_name, &map_hash) {
        return vec![rec(&sha, &rep.header.map_name, &family, None, Outcome::Filtered { reason: format!("excluded map (040-R16): {name}") })];
    }
    let (bot, any_length) = filter_rules(&opt.src, opt.relabel.is_some());
    let views = match filter(&rep, &family, bot, any_length) {
        Ok(v) => v,
        Err(reason) => return vec![rec(&sha, &rep.header.map_name, &family, None, Outcome::Filtered { reason })],
    };
    let views = match &opt.games {
        Some(g) => match crate::lobby::players0_view(&rep, views, g.lobby(file), opt.relabel) {
            Ok(v) => vec![v],
            Err((view, reason)) => return vec![rec(&sha, &rep.header.map_name, &family, view, Outcome::Filtered { reason })],
        },
        None if opt.src == "botgen" && crate::lobby::is_tp_teacher_game(&rep) => {
            let reason = "022 teacher game in a botgen build without --games <games.jsonl> (022-R12: teacher view only, lobby races)".to_string();
            return vec![rec(&sha, &rep.header.map_name, &family, None, Outcome::Filtered { reason })];
        }
        None => views,
    };
    let views = opt.done.remaining(&sha, views);
    if views.is_empty() {
        return Vec::new();
    }
    if let Err(e) = rep.roundtrip_check() {
        let reason = format!("{e:#}");
        let outcome = if reason.starts_with("review:") { Outcome::Review { reason } } else { Outcome::Error { reason: format!("013-R7 round trip: {reason}") } };
        return vec![rec(&sha, &rep.header.map_name, &family, None, outcome)];
    }
    let recorded_name = opt.recorded_winners.get(&sha).map(String::as_str);
    let recorded_winner = recorded_slot(&rep, recorded_name);
    *recorded_cell.borrow_mut() = recorded_name.map(|_| if recorded_winner.is_some() { "matched" } else { "unmatched" });
    if recorded_name.is_some() && recorded_winner.is_none() {
        eprintln!("040-R8: recorded winner of {} matches no player name", &sha[..12]);
    }
    let memory = opt.memory_log.as_ref().map(|d| d.logs(file));
    let mut out = Vec::new();
    for view in views {
        if *written.lock().unwrap() >= opt.budget_bytes {
            out.push(rec(&sha, &rep.header.map_name, &family, Some(view), Outcome::Filtered { reason: "storage budget reached (040-R12)".into() }));
            continue;
        }
        let vo = ViewOptions { until_frame: opt.until_frame, time_limit: opt.time_limit, relabel: opt.relabel, map_hash: &map_hash, memory: memory.as_deref(), distill: opt.src == DISTILL_SRC };
        let outcome = match build_view(&rep, &view, cfg, labeler, recorded_winner, vo) {
            Ok(Ok(ViewGame { steps, coverage, result, result_source, game_info, repeats_dropped, memory })) => {
                let n = steps.len();
                let pre = match write_map_precompute(&map_hash, &game_info, &rep.map) {
                    Ok(p) => p,
                    Err(e) => {
                        out.push(rec(&sha, &rep.header.map_name, &family, Some(view), Outcome::Error { reason: format!("map precompute: {e:#}") }));
                        continue;
                    }
                };
                let meta = GameMeta {
                    map_hash: map_hash.clone(),
                    map_family: family.clone(),
                    matchup: view.matchup.clone(),
                    player_race: header::RACE_TERRAN,
                    perspective_slot: slot_of(&rep, view.perspective).unwrap_or(0),
                    apm: view.apm,
                    result,
                    result_source: result_source.to_string(),
                    enemy_random: view.enemy_random,
                    src: opt.src.clone(),
                    stage: view.stage.to_string(),
                    replay_sha256: sha.clone(),
                };
                let game = ShardGame { meta, game_info, steps };
                match invalid_label_filter(&game, Some(&pre)) {
                    Ok(None) => {}
                    Ok(Some(reason)) => {
                        out.push(rec(&sha, &rep.header.map_name, &family, Some(view), Outcome::Filtered { reason }));
                        continue;
                    }
                    Err(e) => {
                        out.push(rec(&sha, &rep.header.map_name, &family, Some(view), Outcome::Error { reason: format!("040-R19 label check: {e:#}") }));
                        continue;
                    }
                }
                *written.lock().unwrap() += postcard::experimental::serialized_size(&game).unwrap_or(0) as u64;
                let (r, _) = rec(&sha, &rep.header.map_name, &family, Some(view), Outcome::Written { steps: n, coverage, result, result_source: result_source.to_string(), repeats_dropped, memory });
                out.push((r, Some(game)));
                continue;
            }
            Ok(Err(o)) => o,
            Err(e) if format!("{e:#}").contains("launcher exited before the bridge connected") => Outcome::Unplayable { reason: format!("{e:#}") },
            Err(e) => Outcome::Error { reason: format!("{e:#}") },
        };
        out.push(rec(&sha, &rep.header.map_name, &family, Some(view), outcome));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::label::Coverage;
    use crate::{Replay, cmd};

    fn written_record(n: u32) -> Record {
        let coverage = Coverage::default();
        Record { index: n as usize, replay_sha256: format!("{n:064x}"), src: "t".into(), map_family: "other".into(), map_name: String::new(), map_hash: String::new(), recorded_winner: None, view: None, labeler_version: crate::label::LABELER_VERSION, until_frame: None, time_limit: None, relabel: None, outcome: Outcome::Written { steps: 3, coverage, result: 0, result_source: "none".into(), repeats_dropped: None, memory: None } }
    }

    #[test]
    fn failed_shard_records_are_never_sunk_040_r17() {
        use crate::shard::tests::{corrupting_copy, game};
        let dir = std::env::temp_dir().join(format!("uranium-pipeline-m1-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let one = postcard::experimental::serialized_size(&game(1)).unwrap();
        let mut writer = ShardWriter::new(dir.join("shared"), dir.join("local"), 0, one + 1).unwrap();
        writer.copy = corrupting_copy;
        let mut p = Pending { writer, records: Vec::new() };
        let failure = Mutex::new(None);
        let sunk = Mutex::new(Vec::<String>::new());
        let sink = |r: &Record| sunk.lock().unwrap().push(r.replay_sha256.clone());
        assert!(add_game(&mut p, &failure, game(1), written_record(1), &sink));
        assert!(!add_game(&mut p, &failure, game(2), written_record(2), &sink));
        assert!(failure.lock().unwrap().is_some() && p.records.is_empty());
        let local = |d: &str| std::fs::read_dir(dir.join(d)).map(|r| r.flatten().map(|e| e.path()).collect::<Vec<_>>()).unwrap_or_default();
        let kept = local("local");
        assert_eq!(kept.len(), 1);
        let bytes = std::fs::read(&kept[0]).unwrap();
        p.writer.copy = |from, to| std::fs::copy(from, to);
        assert!(!add_game(&mut p, &failure, game(3), written_record(3), &sink));
        assert!(!add_game(&mut p, &failure, game(4), written_record(4), &sink));
        assert!(sunk.lock().unwrap().is_empty());
        assert_eq!(local("local"), kept);
        assert_eq!(std::fs::read(&kept[0]).unwrap(), bytes);
        assert!(local("shared").is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn resume_rebuilds_unflushed_perspective_040_r17() {
        use crate::shard::tests::game;
        let dir = std::env::temp_dir().join(format!("uranium-pipeline-resume-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let sha = "ab".repeat(32);
        let view = |p: u8| View { perspective: p, stage: "S1", apm: 900, matchup: "TvT".into(), enemy_random: false };
        let record = |p: u8, outcome: Outcome| Record { view: Some(view(p)), outcome, replay_sha256: sha.clone(), ..written_record(7) };
        let lines = Mutex::new(Vec::<String>::new());
        let sink = |r: &Record| lines.lock().unwrap().push(serde_json::to_string(r).unwrap());
        let writer = ShardWriter::new(dir.join("shared"), dir.join("local"), 0, 1 << 30).unwrap();
        let mut p = Pending { writer, records: Vec::new() };
        let failure = Mutex::new(None);
        sink(&record(0, Outcome::Filtered { reason: "invalid_labels>=20% (54.6%, 20444/37441) (040-R19)".into() }));
        assert!(add_game(&mut p, &failure, game(1), record(1, written_record(7).outcome), &sink));
        drop(p);
        let mut done = ResumeDone::default();
        lines.lock().unwrap().iter().for_each(|l| done.add_line(l));
        assert!(!done.replay_done(&sha));
        let todo: Vec<u8> = done.remaining(&sha, vec![view(0), view(1)]).iter().map(|v| v.perspective).collect();
        assert_eq!(todo, vec![1], "only the lost perspective is built again");
        sink(&record(1, written_record(7).outcome));
        let mut done = ResumeDone::default();
        lines.lock().unwrap().iter().for_each(|l| done.add_line(l));
        assert!(done.remaining(&sha, vec![view(0), view(1)]).is_empty());
        assert_eq!(done.len(), 2, "one record per perspective, no duplicates");
        let mut whole = ResumeDone::default();
        whole.add_line(&serde_json::to_string(&Record { outcome: Outcome::Filtered { reason: "shorter than 5 minutes".into() }, ..written_record(9) }).unwrap());
        assert!(whole.replay_done(&format!("{:064x}", 9)));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn resume_reads_labeler_v1_records_040_r17() {
        let sha = "cd".repeat(32);
        let v1_written = format!(r#"{{"index":3,"replay_sha256":"{sha}","src":"t","map_family":"other","map_name":"m","map_hash":"h","view":{{"perspective":1,"stage":"S1","apm":200,"matchup":"TvZ","enemy_random":false}},"outcome":"written","steps":10,"coverage":{{"commands":5,"labeled":5,"unknown":0,"unknown_ops":[],"unknown_orders":[]}},"result":1,"result_source":"bridge"}}"#);
        let v1_whole = format!(r#"{{"index":4,"replay_sha256":"{}","src":"t","map_family":"other","map_name":"","map_hash":"","view":null,"outcome":"filtered","reason":"shorter than 5 minutes"}}"#, "ef".repeat(32));
        let mut done = ResumeDone::default();
        done.add_line(&v1_written);
        done.add_line(&v1_whole);
        let view = |p: u8| View { perspective: p, stage: "S1", apm: 200, matchup: "TvZ".into(), enemy_random: false };
        assert_eq!(done.remaining(&sha, vec![view(0), view(1)]).iter().map(|v| v.perspective).collect::<Vec<_>>(), vec![0]);
        assert!(done.replay_done(&"ef".repeat(32)));
        let v: serde_json::Value = serde_json::to_value(written_record(1)).unwrap();
        assert_eq!(v["labeler_version"], crate::label::LABELER_VERSION);
    }

    #[test]
    fn identical_files_build_one_game_040_r20() {
        let dir = std::env::temp_dir().join(format!("uranium-dup-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let files: Vec<PathBuf> = ["a", "b", "c", "d"].iter().map(|n| dir.join(format!("{n}.rep"))).collect();
        for (f, body) in files.iter().zip(["xx", "y", "xx", "xx"]) {
            std::fs::write(f, body).unwrap();
        }
        let sha = |b: &str| hex(&Sha256::digest(b.as_bytes()));
        let part0 = Duplicates::find(&files, &[0, 2], 2);
        let part1 = Duplicates::find(&files, &[1, 3], 2);
        assert_eq!(part0.first[&sha("xx")], 0, "the first file in sorted order keeps the replay");
        assert_eq!(part1.first[&sha("xx")], 0);
        assert!(!part1.first.contains_key(&sha("y")), "a file of unique size is not read");
        assert_eq!((part0.repeats, part1.repeats), (1, 1));
        assert_eq!(part0.earlier_copy(0, &sha("xx")), None);
        assert_eq!(part0.earlier_copy(1, &sha("xx")), Some(0));
        assert_eq!(part1.earlier_copy(0, &sha("y")), None);
        assert_eq!(part1.earlier_copy(1, &sha("xx")), Some(0));
        let missed = Duplicates { global: vec![0, 2], first: HashMap::from([(sha("xx"), 2)]), repeats: 0 };
        assert_eq!(missed.earlier_copy(0, &sha("xx")), None);
        assert_eq!(missed.earlier_copy(1, &sha("xx")), None);

        let families = MapFamilies::load().unwrap();
        let excludes = MapExcludes::load(&families).unwrap();
        let labeler = Labeler::new().unwrap();
        let cfg = PlayConfig { launcher: dir.join("none"), bridge_lib: dir.join("none"), data_dir: dir.join("none"), work_dir: dir.join("w") };
        let opt = BuildOptions { src: "t".into(), shards_dir: dir.join("s"), staging_dir: dir.join("l"), recorded_winners: HashMap::new(), jobs: 1, shard_target_bytes: 1, budget_bytes: 1 << 30, first_shard_index: 0, skip_sha256: Default::default(), done: ResumeDone::default(), duplicates: part1, until_frame: None, time_limit: None, relabel: None, memory_log: None, games: None };
        let out = process(1, &files[3], &opt, &cfg, &families, &excludes, &labeler, &Mutex::new(0));
        assert_eq!(out.len(), 1);
        let (r, game) = &out[0];
        assert!(game.is_none() && r.view.is_none() && r.replay_sha256 == sha("xx"));
        assert!(matches!(&r.outcome, Outcome::Filtered { reason } if reason == &format!("{DUPLICATE_IN_SOURCE}: same file as global index 0")), "{:?}", r.outcome);
        let kept = process(0, &files[1], &opt, &cfg, &families, &excludes, &labeler, &Mutex::new(0));
        assert!(!matches!(&kept[0].0.outcome, Outcome::Filtered { reason } if reason.starts_with(DUPLICATE_IN_SOURCE)));

        let mut done = ResumeDone::default();
        done.add_line(&serde_json::to_string(r).unwrap());
        assert!(!done.replay_done(&sha("xx")) && done.is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }

    fn tp_replay(dir: &Path, id: u64, players: [(u8, u8, &str); 2]) -> PathBuf {
        let mut h = vec![0u8; crate::container::HEADER_SIZE];
        h[1..5].copy_from_slice(&8_000u32.to_le_bytes());
        h[0x61] = b't';
        for (i, (pid, race, name)) in players.iter().enumerate() {
            let p = 0xa1 + i * 36;
            h[p] = i as u8;
            h[p + 4] = *pid;
            h[p + 8] = 2;
            h[p + 9] = *race;
            h[p + 10] = i as u8;
            h[p + 11..p + 11 + name.len()].copy_from_slice(name.as_bytes());
        }
        let train = |p: u8| cmd::Cmd { frame: 10, player: p, op: cmd::op::TRAIN, data: vec![7, 0] };
        let blocks = vec![cmd::Block { frame: 10, cmds: vec![train(players[0].0), train(players[1].0)] }];
        let rep = Replay { format: crate::Format::Legacy, header: header::parse(&h), raw_header: h, blocks, map: vec![9; 10], unit_limit: None };
        let path = dir.join(format!("g{id}.rep"));
        std::fs::write(&path, rep.to_legacy_file().unwrap()).unwrap();
        path
    }

    #[test]
    fn players0_view_and_lobby_races_from_games_jsonl_022_r12_ac3() {
        use crate::header::{RACE_PROTOSS as P, RACE_TERRAN as T, RACE_ZERG as Z};
        use uranium_scripted::T0Build;
        let dir = std::env::temp_dir().join(format!("uranium-tp-lobby-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let tp = "p0-T0-terran-terran_tp";
        let game = |id, enemy_header, enemy_name: &str| tp_replay(&dir, id, [(0, enemy_header, enemy_name), (1, T, tp)]);
        let lines = [(1, "Terran", "Terran"), (2, "Terran", "Random"), (3, "Terran", "Zerg"), (4, "Terran", "Protoss"), (6, "Terran", "Protoss"), (8, "Random", "Protoss"), (9, "Terran", "Protoss"), (10, "Terran", "Protoss"), (11, "Terran", "Protoss")];
        let text: String = lines.iter().map(|(id, a, b)| format!("{{\"game_id\":{id},\"matchup\":0,\"races\":[\"{a}\",\"{b}\"],\"outcome\":\"finished\"}}\n")).collect();
        let games = crate::lobby::GamesLog::parse(&text).unwrap();
        let families = MapFamilies::load().unwrap();
        let excludes = MapExcludes::load(&families).unwrap();
        let labeler = Labeler::new().unwrap();
        let cfg = PlayConfig { launcher: dir.join("none"), bridge_lib: dir.join("none"), data_dir: dir.join("none"), work_dir: dir.join("w") };
        let run = |path: &Path, src: &str, relabel: Option<T0Build>, games: Option<crate::lobby::GamesLog>| {
            let opt = BuildOptions { src: src.into(), shards_dir: dir.join("s"), staging_dir: dir.join("l"), recorded_winners: HashMap::new(), jobs: 1, shard_target_bytes: 1, budget_bytes: 1 << 30, first_shard_index: 0, skip_sha256: Default::default(), done: ResumeDone::default(), duplicates: Duplicates::find(&[path.to_path_buf()], &[0], 1), until_frame: Some(14_400), time_limit: Some(14_400), relabel, memory_log: None, games };
            process(0, path, &opt, &cfg, &families, &excludes, &labeler, &Mutex::new(0)).into_iter().map(|(r, g)| {
                assert!(g.is_none());
                assert_eq!(r.relabel, relabel.map(|b| b.name()));
                (r.view.map(|v| v.perspective), r.outcome)
            }).collect::<Vec<_>>()
        };
        let reason = |o: &Outcome| match o {
            Outcome::Filtered { reason } => reason.clone(),
            _ => String::new(),
        };
        let tvt = game(1, T, "p1-T0-terran");
        let got = run(&tvt, "botgen", None, Some(games.clone()));
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].0, Some(1), "{got:?}");
        assert!(reason(&got[0].1).is_empty(), "{:?}", got[0].1);
        let got = run(&tvt, "botgen", None, None);
        assert!(got.len() == 1 && got[0].0.is_none() && reason(&got[0].1).contains("without --games"), "{got:?}");
        let got = run(&tvt, "dagger", Some(T0Build::TerranTp), Some(games.clone()));
        assert_eq!(got.iter().map(|g| g.0).collect::<Vec<_>>(), vec![Some(1)]);
        let random = game(2, Z, "p1-ualbertabot");
        for (src, relabel) in [("dagger", Some(T0Build::TerranTp)), ("botgen", None)] {
            let got = run(&random, src, relabel, Some(games.clone()));
            assert!(got.len() == 1 && got[0].0 == Some(1) && reason(&got[0].1).starts_with("lobby Random") && reason(&got[0].1).contains("022-R2"), "{got:?}");
        }
        let zerg = game(3, Z, "p1-T0-zerg");
        assert!(reason(&run(&zerg, "dagger", Some(T0Build::TerranTp), Some(games.clone()))[0].1).contains("022-AC3"));
        assert!(reason(&run(&zerg, "botgen", None, Some(games.clone()))[0].1).contains("022-R12"));
        let got = run(&game(4, P, "p1-stardust"), "dagger", Some(T0Build::TerranTp), Some(games.clone()));
        assert!(got.len() == 1 && got[0].0 == Some(1) && reason(&got[0].1).is_empty(), "{got:?}");
        let got = run(&game(5, P, "p1-stardust"), "dagger", Some(T0Build::TerranTp), Some(games.clone()));
        assert!(got.len() == 1 && got[0].0.is_none() && reason(&got[0].1).contains("no games.jsonl line"), "{got:?}");
        let got = run(&game(6, T, "p1-T0-terran"), "botgen", None, Some(games.clone()));
        assert!(reason(&got[0].1).contains("header race Terran"), "{got:?}");
        let got = run(&game(8, P, "p1-stardust"), "botgen", None, Some(games.clone()));
        assert!(got.len() == 1 && got[0].0 == Some(1) && reason(&got[0].1).contains("players[0] lobby race Random"), "{got:?}");
        let got = run(&tp_replay(&dir, 9, [(0, P, "stardust"), (1, T, "teacher")]), "dagger", Some(T0Build::TerranTp), Some(games.clone()));
        assert!(got.len() == 1 && got[0].0.is_none() && reason(&got[0].1).contains("0 players[0] player names"), "{got:?}");
        let got = run(&tp_replay(&dir, 10, [(0, P, "p0-stardust"), (1, T, tp)]), "dagger", Some(T0Build::TerranTp), Some(games.clone()));
        assert!(got.len() == 1 && got[0].0.is_none() && reason(&got[0].1).contains("2 players[0] player names"), "{got:?}");
        let other = tp_replay(&dir, 11, [(0, P, "p1-stardust"), (1, T, "p0-mine")]);
        let got = run(&other, "botgen", None, Some(games.clone()));
        assert!(got.len() == 1 && got[0].0.is_none() && reason(&got[0].1).contains("not the 022 teacher"), "{got:?}");
        let got = run(&other, "dagger", Some(T0Build::TerranTp), Some(games.clone()));
        assert!(got.len() == 1 && got[0].0 == Some(1) && reason(&got[0].1).is_empty(), "{got:?}");
        assert_eq!(run(&tvt, "dagger", Some(T0Build::TerranReact), None).len(), 2);
        let react = tp_replay(&dir, 7, [(0, T, "p1-T0-terran"), (1, T, "p0-T0-terran-terran_reac")]);
        assert!(run(&react, "botgen", None, None).iter().all(|g| g.0.is_some() && reason(&g.1).is_empty()));
        assert!(run(&zerg, "dagger", Some(T0Build::TerranReact), None).iter().all(|g| reason(&g.1).is_empty()));
        std::fs::remove_dir_all(&dir).ok();
    }
}
