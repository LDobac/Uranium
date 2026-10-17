//! BC dataset build (040 pipeline): filter -> OpenBW replay A (desync check, units at command
//! frames) -> label inversion -> replay B twice (decision-time observations, 040-R8 determinism)
//! -> shards. Runs launchers locally with `jobs` workers.

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::label::{Coverage, Labeler};
use crate::play::{PlayConfig, play_with, slot_of};
use crate::shard::{GameMeta, ShardGame, ShardWriter, Step};
use crate::{Replay, cmd, header};

// Map families and the precompute folder live in uranium-env (shared with live games, AMB-031-4).
pub use uranium_env::maps::{MAP_FAMILIES_TOML, MapFamilies, maps_dir};
use uranium_env::maps::clean_title;
/// 040 filter: game length >= 5 minutes.
pub const MIN_GAME_MS: u64 = 5 * 60 * 1000;
/// 040 / AMB-040-2 quality stages.
pub const S1_MIN_APM: u32 = 80;
pub const S2_MIN_APM: u32 = 150;
/// 030-R6: map axes up to 128 tiles.
pub const MAX_MAP_TILES: u16 = 128;

pub const MAP_EXCLUDE_TOML: &str = include_str!("../../../config/map_exclude.toml");

#[derive(Deserialize)]
struct Excluded {
    name: String,
    name_patterns: Vec<String>,
    chk_sha256: Vec<String>,
}

#[derive(Deserialize)]
struct ExcludeFile {
    map: Vec<Excluded>,
}

/// 040-R16: maps whose measured desync rate is above 10%, skipped before replaying.
pub struct MapExcludes(Vec<Excluded>);

impl MapExcludes {
    /// Loads the list; a target-map family in it is a configuration error (040-R16).
    pub fn load(families: &MapFamilies) -> Result<Self> {
        let list = toml::from_str::<ExcludeFile>(MAP_EXCLUDE_TOML).context("map_exclude.toml")?.map;
        for m in &list {
            for p in &m.name_patterns {
                anyhow::ensure!(families.classify(p, "") == "other", "map_exclude.toml: {} matches a target map family (040-R16)", m.name);
            }
        }
        Ok(Self(list))
    }

    /// Name of the excluded map entry this replay map matches, if any.
    pub fn matches(&self, map_name: &str, chk_sha256: &str) -> Option<&str> {
        let title = clean_title(map_name);
        self.0.iter().find(|m| m.chk_sha256.iter().any(|h| h == chk_sha256) || m.name_patterns.iter().any(|p| title.contains(&p.to_lowercase()))).map(|m| m.name.as_str())
    }
}

/// Map title without color / control codes, lowercase (040-R15, R16 matching).
/// One player view the filter accepted.
#[derive(Debug, Clone, Serialize)]
pub struct View {
    pub perspective: u8,
    pub stage: &'static str,
    pub apm: u32,
    pub matchup: String,
}

fn race_letter(r: u8) -> char {
    match r {
        header::RACE_ZERG => 'Z',
        header::RACE_TERRAN => 'T',
        header::RACE_PROTOSS => 'P',
        _ => 'R',
    }
}

/// 040 filter and stage (040-R9): 1v1, Terran involved, >= 5 min, map <= 128 tiles; every
/// Terran player is a view (040-R2) with S2 on target maps at APM >= 150, S1 at APM >= 80.
/// The game type is not filtered: many ladder 1v1 replays are recorded as "use map settings"
/// and play fine; maps OpenBW cannot open show up as `Outcome::Unplayable` (M4 spike).
pub fn filter(rep: &Replay, family: &str, bot: bool) -> std::result::Result<Vec<View>, String> {
    let active = rep.active_players();
    if active.len() != 2 {
        return Err(format!("{} active players", active.len()));
    }
    if (rep.header.frames as u64) * crate::MS_PER_FRAME < MIN_GAME_MS {
        return Err("shorter than 5 minutes".into());
    }
    if rep.header.map_width > MAX_MAP_TILES || rep.header.map_height > MAX_MAP_TILES {
        return Err("map larger than 128 tiles".into());
    }
    let mut views = Vec::new();
    for (i, (p, s)) in active.iter().enumerate() {
        if p.race != header::RACE_TERRAN {
            continue;
        }
        let other = &active[1 - i].0;
        // 040 quality table: bot data (src bot / botgen) has no APM bar and stage B.
        let stage = if bot {
            "B"
        } else if family != "other" && s.apm >= S2_MIN_APM {
            "S2"
        } else if s.apm >= S1_MIN_APM {
            "S1"
        } else {
            continue;
        };
        views.push(View { perspective: p.id, stage, apm: s.apm, matchup: format!("Tv{}", race_letter(other.race)) });
    }
    if views.is_empty() {
        return Err("no Terran view passes the APM bar".into());
    }
    Ok(views)
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum Outcome {
    Written { steps: usize, coverage: Coverage },
    Filtered { reason: String },
    Desync { reasons: Vec<String> },
    /// 013: opcode or command OpenBW cannot run; kept for review, not discarded.
    Review { reason: String },
    /// OpenBW could not open the replay (e.g. a custom map with unsupported flags).
    Unplayable { reason: String },
    Error { reason: String },
}

#[derive(Debug, Clone, Serialize)]
pub struct Record {
    pub index: usize,
    pub replay_sha256: String,
    pub src: String,
    pub map_family: String,
    /// Map title as stored (for per-map desync statistics).
    pub map_name: String,
    pub view: Option<View>,
    #[serde(flatten)]
    pub outcome: Outcome,
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// Processes one view of one replay into a shard game.
fn build_view(rep: &Replay, view: &View, cfg: &PlayConfig, labeler: &Labeler) -> Result<std::result::Result<(Vec<Step>, Coverage, i8, uranium_proto::GameInfo), Outcome>> {
    let me = slot_of(rep, view.perspective)?;
    // Replay A: own unit types at each command frame + desync signals.
    let mut units_at: HashMap<u32, HashMap<u32, crate::label::Own>> = HashMap::new();
    let a = play_with(rep, view.perspective, cfg, &[], &mut |f, _| {
        units_at.insert(f.frame, f.units.iter().filter(|u| u.owner == me).map(|u| (u.id, crate::label::Own::from_state(u))).collect());
    })?;
    let reasons = a.desync_reasons(None);
    if !reasons.is_empty() {
        return Ok(Err(Outcome::Desync { reasons }));
    }
    let cmds: Vec<cmd::Cmd> = rep
        .cmds()
        .filter(|c| c.player == view.perspective)
        .map(|c| cmd::to_legacy(c, rep.unit_limit))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .filter(|c| cmd::openbw(c.op) == cmd::OpenBw::Supported)
        .collect();
    let empty = HashMap::new();
    let (labels, coverage) = labeler.label(&cmds, rep.header.frames, |f| units_at.get(&f).unwrap_or(&empty));
    let frames: Vec<u32> = labels.iter().map(|l| l.frame).collect();
    // Replay B: decision-time observations. B stops at every frame A stopped at, so A and B
    // form the 040-R8 double-replay check (check_digest ignores B's extra stops).
    let mut by_frame: HashMap<u32, (uranium_proto::RawFrame, Option<uranium_proto::PrivilegedSummary>)> = HashMap::new();
    let res = play_with(rep, view.perspective, cfg, &frames, &mut |f, p| {
        if frames.binary_search(&f.frame).is_ok() {
            by_frame.insert(f.frame, (f.clone(), p.cloned()));
        }
    })?;
    if a.check_digest != res.check_digest {
        return Ok(Err(Outcome::Desync { reasons: vec![format!("replays differ: {} vs {}", a.check_digest, res.check_digest)] }));
    }
    let mut steps = Vec::with_capacity(labels.len());
    for l in labels {
        let Some((raw_frame, privileged)) = by_frame.remove(&l.frame) else { break };
        steps.push(Step { raw_frame, privileged, action_label: l });
    }
    let result = match res.winner {
        Some(w) if w == me => 1,
        Some(_) => -1,
        None => 0,
    };
    let Some(game_info) = res.game_info else {
        return Ok(Err(Outcome::Error { reason: "bridge sent no GameInfo (AMB-040-6)".into() }));
    };
    Ok(Ok((steps, coverage, result, game_info)))
}

/// Writes `<maps_dir>/<map_hash>.toml` unless it exists (031, AMB-031-3).
pub fn write_map_precompute(map_hash: &str, info: &uranium_proto::GameInfo, chk: &[u8]) -> Result<()> {
    use uranium_env::features::mapgen;
    let dir = maps_dir();
    let path = dir.join(format!("{map_hash}.toml"));
    if path.exists() {
        return Ok(());
    }
    std::fs::create_dir_all(&dir)?;
    let pre = mapgen::precompute(&info.static_map, &mapgen::chk_resources(chk)?);
    // Write then rename so parallel workers never read a partial file.
    let tmp = dir.join(format!("{map_hash}.toml.{}", std::process::id()));
    std::fs::write(&tmp, toml::to_string(&pre)?)?;
    std::fs::rename(&tmp, &path).or_else(|_| std::fs::remove_file(&tmp))?;
    Ok(())
}

pub struct BuildOptions {
    pub src: String,
    pub shards_dir: PathBuf,
    pub jobs: usize,
    pub shard_target_bytes: usize,
    /// 040-R12 / 060-R7 storage budget for this run's shards.
    pub budget_bytes: u64,
    pub first_shard_index: u32,
    /// SHA-256 of replays already in another source (040 dedup by file hash); skipped.
    pub skip_sha256: std::collections::HashSet<String>,
    /// SHA-256 of replays already in this run's records (resume); not processed again.
    pub done_sha256: std::collections::HashSet<String>,
}

/// Builds shards from `files`; returns one record per replay view (or per rejected replay).
/// Shard writer plus the records of games waiting in its unflushed shard.
struct Pending {
    writer: ShardWriter,
    records: Vec<Record>,
}

/// Builds shards from `files`, handing every record to `sink` as soon as it is final: a
/// written game's record only after the shard holding it is on disk, so an interrupted run
/// can resume from its records (`BuildOptions::done_sha256`).
pub fn build(files: &[PathBuf], opt: &BuildOptions, cfg: &PlayConfig, sink: &(dyn Fn(&Record) + Sync)) -> Result<()> {
    let families = MapFamilies::load()?;
    let excludes = MapExcludes::load(&families)?;
    let labeler = Labeler::new()?;
    let pending = Mutex::new(Pending { writer: ShardWriter::new(opt.shards_dir.clone(), opt.first_shard_index, opt.shard_target_bytes)?, records: Vec::new() });
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
                        let mut p = pending.lock().unwrap();
                        let before = p.writer.written.len();
                        if let Err(e) = p.writer.add(game) {
                            *failure.lock().unwrap() = Some(e);
                            queue.lock().unwrap().clear();
                            return;
                        }
                        p.records.push(r);
                        if p.writer.written.len() > before {
                            for r in p.records.drain(..) {
                                sink(&r);
                            }
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
    let rec = |sha: &str, map: &str, family: &str, view: Option<View>, outcome: Outcome| (Record { index, replay_sha256: sha.to_string(), src: opt.src.clone(), map_family: family.to_string(), map_name: map.chars().filter(|c| !c.is_control()).collect(), view, outcome }, None);
    let bytes = match std::fs::read(file) {
        Ok(b) => b,
        Err(e) => return vec![rec("", "", "", None, Outcome::Error { reason: e.to_string() })],
    };
    let sha = hex(&Sha256::digest(&bytes));
    if opt.done_sha256.contains(&sha) {
        return Vec::new(); // already recorded by an earlier run
    }
    if opt.skip_sha256.contains(&sha) {
        return vec![rec(&sha, "", "", None, Outcome::Filtered { reason: "duplicate of another source".into() })];
    }
    let rep = match crate::parse(&bytes) {
        Ok(r) => r,
        // 013: an opcode we cannot parse is held for review, not discarded.
        Err(e) if format!("{e:#}").contains("unknown opcode") => return vec![rec(&sha, "", "", None, Outcome::Review { reason: format!("{e:#}") })],
        Err(e) => return vec![rec(&sha, "", "", None, Outcome::Error { reason: format!("{e:#}") })],
    };
    let map_hash = hex(&Sha256::digest(&rep.map));
    let family = families.classify(&rep.header.map_name, &map_hash);
    if let Some(name) = excludes.matches(&rep.header.map_name, &map_hash) {
        return vec![rec(&sha, &rep.header.map_name, &family, None, Outcome::Filtered { reason: format!("excluded map (040-R16): {name}") })];
    }
    let views = match filter(&rep, &family, matches!(opt.src.as_str(), "bot" | "botgen")) {
        Ok(v) => v,
        Err(reason) => return vec![rec(&sha, &rep.header.map_name, &family, None, Outcome::Filtered { reason })],
    };
    if let Err(e) = rep.to_legacy_file() {
        let reason = format!("{e:#}");
        let outcome = if reason.starts_with("review:") { Outcome::Review { reason } } else { Outcome::Error { reason } };
        return vec![rec(&sha, &rep.header.map_name, &family, None, outcome)];
    }
    let mut out = Vec::new();
    for view in views {
        if *written.lock().unwrap() >= opt.budget_bytes {
            out.push(rec(&sha, &rep.header.map_name, &family, Some(view), Outcome::Filtered { reason: "storage budget reached (040-R12)".into() }));
            continue;
        }
        let outcome = match build_view(&rep, &view, cfg, labeler) {
            Ok(Ok((steps, coverage, result, game_info))) => {
                let n = steps.len();
                // 031 / AMB-031-3: map precompute once per map (observation channels only).
                if let Err(e) = write_map_precompute(&map_hash, &game_info, &rep.map) {
                    eprintln!("map precompute for {map_hash}: {e:#}");
                }
                let meta = GameMeta {
                    map_hash: map_hash.clone(),
                    map_family: family.clone(),
                    matchup: view.matchup.clone(),
                    player_race: header::RACE_TERRAN,
                    perspective_slot: slot_of(&rep, view.perspective).unwrap_or(0),
                    apm: view.apm,
                    result,
                    src: opt.src.clone(),
                    stage: view.stage.to_string(),
                    replay_sha256: sha.clone(),
                };
                let game = ShardGame { meta, game_info, steps };
                *written.lock().unwrap() += postcard::experimental::serialized_size(&game).unwrap_or(0) as u64;
                let (r, _) = rec(&sha, &rep.header.map_name, &family, Some(view), Outcome::Written { steps: n, coverage });
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

    #[test]
    fn map_families_classify_040_r15() {
        let f = MapFamilies::load().unwrap();
        assert_eq!(f.classify("| iCCup | Fighting Spirit", ""), "fighting_spirit");
        assert_eq!(f.classify("투혼 1.3", ""), "fighting_spirit");
        assert_eq!(f.classify("\u{3}Circuit Breakers \u{5}1.0", ""), "circuit_breaker");
        assert_eq!(f.classify("Python", ""), "other");
    }

    #[test]
    fn high_desync_maps_are_excluded_040_r16() {
        let f = MapFamilies::load().unwrap();
        let x = MapExcludes::load(&f).unwrap();
        // Titles as stored in SC:R replays (color codes inside).
        assert_eq!(x.matches("\u{7}Attitude SE \u{5}2.1", ""), Some("Attitude SE 2.1"));
        assert_eq!(x.matches("\u{1}C\u{6}olorless \u{2}F\u{6}ate \u{5}1.1", ""), Some("Colorless Fate 1.1"));
        // Other versions and target maps are not excluded.
        assert_eq!(x.matches("Attitude SE 2.2", ""), None);
        assert_eq!(x.matches("\u{7}투혼 \u{5}1.4", ""), None);
        assert_eq!(x.matches("Octagon SE 2.0", ""), None);
    }
}
