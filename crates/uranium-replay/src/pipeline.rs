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
use crate::play::{PlayConfig, PlayResult, play_observe, play_with, slot_of};
use crate::shard::{GameMeta, ShardGame, ShardWriter, Step, result_source};
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

/// 040-R20: `filtered` reason of a replay file whose SHA-256 a file earlier in the source's full
/// sorted list already has (followed by `: same file as global index <n>`). Such a record does
/// not mark the replay done for resume (the first file's perspectives may still be waiting for
/// their shard flush).
pub const DUPLICATE_IN_SOURCE: &str = "duplicate file in this source (040-R20)";

/// 040-R20: which files of this run repeat an earlier file of the whole source. Decided on the
/// full sorted file list (before the `--part` split), so every host decides the same.
#[derive(Debug, Default, Clone)]
pub struct Duplicates {
    /// Global index (position in the source's full sorted list) of each file of this run.
    pub global: Vec<usize>,
    /// SHA-256 -> first global index, for the files that share a size with another file.
    pub first: HashMap<String, usize>,
    /// Files of this run that repeat an earlier file.
    pub repeats: usize,
}

impl Duplicates {
    /// `all`: the source's full sorted list; `mine`: global indices of this run's files (in run
    /// order). Only files whose size equals another file's size can be identical, so only size
    /// groups that contain one of `mine` are read and hashed (`jobs` threads); unreadable files
    /// are left to `process` (which records the error).
    pub fn find(all: &[PathBuf], mine: &[usize], jobs: usize) -> Self {
        let sizes: Vec<Option<u64>> = all.iter().map(|f| std::fs::metadata(f).ok().map(|m| m.len())).collect();
        let mut by_size: HashMap<u64, Vec<usize>> = HashMap::new();
        for (i, s) in sizes.iter().enumerate() {
            if let Some(s) = s {
                by_size.entry(*s).or_default().push(i);
            }
        }
        let mut todo: Vec<usize> = mine.iter().filter_map(|&i| sizes[i]).filter_map(|s| by_size.get(&s)).filter(|g| g.len() > 1).flatten().copied().collect();
        todo.sort_unstable();
        todo.dedup();
        let next = std::sync::atomic::AtomicUsize::new(0);
        let found = Mutex::new(Vec::new());
        std::thread::scope(|s| {
            for _ in 0..jobs.max(1) {
                s.spawn(|| {
                    let mut got = Vec::new();
                    while let Some(&i) = todo.get(next.fetch_add(1, std::sync::atomic::Ordering::Relaxed)) {
                        if let Ok(b) = std::fs::read(&all[i]) {
                            got.push((hex(&Sha256::digest(&b)), i));
                        }
                    }
                    found.lock().unwrap().extend(got);
                });
            }
        });
        let found = found.into_inner().unwrap();
        let mut first = HashMap::new();
        for (sha, i) in &found {
            first.entry(sha.clone()).and_modify(|j: &mut usize| *j = (*j).min(*i)).or_insert(*i);
        }
        let mine_set: std::collections::HashSet<usize> = mine.iter().copied().collect();
        let repeats = found.iter().filter(|(sha, i)| mine_set.contains(i) && first[sha] != *i).count();
        Self { global: mine.to_vec(), first, repeats }
    }

    /// The first global index of `sha` when the run's file `index` is a later copy of it.
    pub fn earlier_copy(&self, index: usize, sha: &str) -> Option<usize> {
        let me = self.global.get(index).copied().unwrap_or(index);
        // Only an EARLIER file makes this one a copy: if hashing missed this file (e.g. a
        // transient read failure), the worst case is one extra game (caught by shard-stats),
        // never both copies dropped.
        self.first.get(sha).copied().filter(|&f| f < me)
    }
}

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
    /// 030-R11: the opponent's header race is Random (lobby choice).
    pub enemy_random: bool,
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
/// `any_length` (041-R22 `dagger`): no 5-minute bar (a model game lost early to a rush is the
/// state DAgger relabels, not a player leaving).
pub fn filter(rep: &Replay, family: &str, bot: bool, any_length: bool) -> std::result::Result<Vec<View>, String> {
    let active = rep.active_players();
    if active.len() != 2 {
        return Err(format!("{} active players", active.len()));
    }
    if !any_length && (rep.header.frames as u64) * crate::MS_PER_FRAME < MIN_GAME_MS {
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
        views.push(View { perspective: p.id, stage, apm: s.apm, matchup: format!("Tv{}", race_letter(other.race)), enemy_random: other.race == header::RACE_RANDOM });
    }
    if views.is_empty() {
        return Err("no Terran view passes the APM bar".into());
    }
    Ok(views)
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum Outcome {
    /// `result` / `result_source`: 040-R8 result of the view (dataset statistics, 040-R6).
    Written { steps: usize, coverage: Coverage, result: i8, result_source: String },
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
    /// Map title (decoded, control codes removed) for per-map statistics.
    pub map_name: String,
    /// SHA-256 of the map CHK (040-R15 / R16 hash lists).
    pub map_hash: String,
    /// 040-R8 (c) / ④: `matched` / `unmatched` when the ladder records name a winner for this
    /// replay (unmatched = the name is neither player's, so it is not used).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recorded_winner: Option<&'static str>,
    pub view: Option<View>,
    /// `label::LABELER_VERSION` of this build (2026-10-19); records without it are version 1.
    pub labeler_version: u32,
    /// 040-R21: the build's `--until-frame` (absent = the whole game).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub until_frame: Option<u32>,
    /// 040-R21: the build's `--time-limit` (absent = none).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_limit: Option<u32>,
    /// 041-R22: the teacher whose commands are the labels (`--relabel`; absent = the replay's own
    /// commands).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub relabel: Option<&'static str>,
    #[serde(flatten)]
    pub outcome: Outcome,
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// 040-R8 `result` of the perspective (BWAPI id `me`) and its `result_source`: (a) bridge winner,
/// (b) the first leaver loses, (c) the recorded SC:R winner (BWAPI id), (d) undecided = 0.
pub fn judge(play: &PlayResult, recorded_winner: Option<u8>, me: u8) -> (i8, &'static str) {
    match play.decided_winner().or(recorded_winner.map(|w| (w, result_source::RECORDED))) {
        Some((w, source)) => (if w == me { 1 } else { -1 }, source),
        None => (0, result_source::NONE),
    }
}

/// BWAPI id of the recorded winner (043-R8 toon) among the two active players, matched by name.
/// Exact name first; otherwise an ASCII case-insensitive match if it names exactly one player.
fn recorded_slot(rep: &Replay, winner: Option<&str>) -> Option<u8> {
    let w = winner?.as_bytes();
    let active = rep.active_players();
    let exact = active.iter().find(|(p, _)| p.name == w);
    let p = exact.or_else(|| match active.iter().filter(|(p, _)| p.name.eq_ignore_ascii_case(w)).collect::<Vec<_>>()[..] {
        [one] => Some(one),
        _ => None,
    })?;
    slot_of(rep, p.0.id).ok()
}

/// Shard game of one view with its label coverage and 040-R8 result.
pub struct ViewGame {
    pub steps: Vec<Step>,
    pub coverage: Coverage,
    pub result: i8,
    pub result_source: &'static str,
    pub game_info: uranium_proto::GameInfo,
}

/// How one view is labelled (040-R21 cut and time limit, 041-R22 relabelling).
#[derive(Clone, Copy)]
struct ViewOptions<'a> {
    until_frame: Option<u32>,
    time_limit: Option<u32>,
    relabel: Option<uranium_scripted::T0Build>,
    /// CHK SHA-256 (map precompute file name, 031).
    map_hash: &'a str,
}

/// Processes one view of one replay into a shard game.
fn build_view(rep: &Replay, view: &View, cfg: &PlayConfig, labeler: &Labeler, recorded_winner: Option<u8>, o: ViewOptions) -> Result<std::result::Result<ViewGame, Outcome>> {
    let (until_frame, time_limit) = (o.until_frame, o.time_limit);
    let me = slot_of(rep, view.perspective)?;
    let (mut a, mut labels, coverage) = match o.relabel {
        // Replay A: own unit types at each command frame + desync signals.
        None => {
            let mut units_at: HashMap<u32, HashMap<u32, crate::label::Own>> = HashMap::new();
            let a = play_with(rep, view.perspective, cfg, &[], &mut |f, _| {
                units_at.insert(f.frame, f.units.iter().filter(|u| u.owner == me).map(|u| (u.id, crate::label::Own::from_state(u))).collect());
            })?;
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
            (a, labels, coverage)
        }
        // 041-R22 replay T: the same desync signals as replay A, and the teacher run as a shadow
        // on the perspective's observations; its commands are the labels (the player's own
        // commands only feed the 040-R8 checks).
        Some(build) => {
            // The last label before a cut takes its delay from up to 32 frames later (031-R11).
            let last = until_frame.map_or(rep.header.frames, |u| u.saturating_add(32)).min(rep.header.frames);
            let stops = crate::relabel::teacher_frames(last);
            let shadow: std::cell::RefCell<Option<crate::relabel::Shadow>> = std::cell::RefCell::new(None);
            let failed: std::cell::RefCell<Option<anyhow::Error>> = std::cell::RefCell::new(None);
            let a = play_observe(
                rep,
                view.perspective,
                cfg,
                &stops,
                &mut |gi| match write_map_precompute(o.map_hash, gi, &rep.map) {
                    // The live T0 session's hints (021-R8 맵 정보): base locations and chokes.
                    Ok(pre) => *shadow.borrow_mut() = Some(crate::relabel::Shadow::new(gi, build, uranium_scripted::MapHints { base_locs: pre.base_locs, chokes: pre.chokes }, view.perspective, last)),
                    Err(e) => *failed.borrow_mut() = Some(e),
                },
                &mut |f, _| {
                    if let Some(s) = shadow.borrow_mut().as_mut() {
                        s.observe(f);
                    }
                },
            )?;
            if let Some(e) = failed.into_inner() {
                return Err(e.context("map precompute for the relabelling teacher (041-R22)"));
            }
            let Some(shadow) = shadow.into_inner() else {
                return Ok(Err(Outcome::Error { reason: "bridge sent no GameInfo: no teacher to relabel with (041-R22)".into() }));
            };
            let (labels, coverage) = shadow.labels(labeler, rep.header.frames);
            (a, labels, coverage)
        }
    };
    a.drop_time_limit_leave(time_limit);
    let reasons = a.desync_reasons(recorded_winner);
    if !reasons.is_empty() {
        return Ok(Err(Outcome::Desync { reasons }));
    }
    // 040-R21: `--until-frame n` keeps only the steps observed at frames <= n (the label coverage
    // still counts the whole game; the result is the game's).
    cut_after(&mut labels, until_frame, |l| l.frame);
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
    // 040-R18: the sequence must start with the game-start observation.
    if let Some(&f) = frames.first().filter(|_| steps.is_empty()) {
        return Ok(Err(Outcome::Error { reason: format!("no observation at the first step frame {f} (040-R18)") }));
    }
    let mut res = res;
    res.drop_time_limit_leave(time_limit);
    let (result, result_source) = judge(&res, recorded_winner, me);
    let Some(game_info) = res.game_info else {
        return Ok(Err(Outcome::Error { reason: "bridge sent no GameInfo (AMB-040-6)".into() }));
    };
    Ok(Ok(ViewGame { steps, coverage, result, result_source, game_info }))
}

/// 040-R21: drops the items (labels / steps) observed after frame `until`; `None` keeps all.
pub fn cut_after<T>(items: &mut Vec<T>, until: Option<u32>, frame: impl Fn(&T) -> u32) {
    if let Some(n) = until {
        items.retain(|i| frame(i) <= n);
    }
}

/// Writes `<maps_dir>/<map_hash>.toml` unless a current one exists (031, AMB-031-3). A file of an
/// older `PRECOMPUTE_VERSION` (no resource list, 030 §2 E-3) is written again. Returns the
/// precompute BC loads for this map (040-R19 label check); `Err` only when it cannot be computed.
/// A failed write is reported and only loses the file (the next build writes it).
pub fn write_map_precompute(map_hash: &str, info: &uranium_proto::GameInfo, chk: &[u8]) -> Result<uranium_env::features::MapPrecompute> {
    use uranium_env::features::mapgen;
    let dir = maps_dir();
    if let Some(p) = dir.as_deref().ok().and_then(|d| uranium_env::maps::load_precompute(d, map_hash)) {
        return Ok(p);
    }
    let pre = mapgen::precompute(&info.static_map, &mapgen::chk_resources(chk)?);
    let write = || -> Result<()> {
        let dir = dir?;
        std::fs::create_dir_all(&dir)?;
        // Write then rename so parallel workers never read a partial file.
        let path = dir.join(format!("{map_hash}.toml"));
        let tmp = dir.join(format!("{map_hash}.toml.{}", std::process::id()));
        std::fs::write(&tmp, toml::to_string(&pre)?)?;
        std::fs::rename(&tmp, &path).or_else(|_| std::fs::remove_file(&tmp))?;
        Ok(())
    };
    if let Err(e) = write() {
        eprintln!("map precompute for {map_hash}: {e:#}");
    }
    Ok(pre)
}

/// 040-R19 (user decision 2026-10-19): the `filtered` reason when the perspective's share of
/// invalid labels (the BC label check, `uranium_env::features::label`) is at least
/// `INVALID_LABEL_MAX` (20%), else `None`.
pub fn invalid_label_filter(game: &ShardGame, pre: Option<&uranium_env::features::MapPrecompute>) -> Result<Option<String>> {
    let v = uranium_env::features::label::label_validity(
        &game.game_info,
        uranium_env::maps::family_index(&game.meta.map_family),
        pre,
        game.meta.enemy_random,
        game.steps.iter().map(|s| (&s.raw_frame, &s.action_label)),
    )?;
    Ok(v.rejected().then(|| v.filter_reason()))
}

pub struct BuildOptions {
    pub src: String,
    pub shards_dir: PathBuf,
    /// 040-R17: local folder where shards are completed before the verified copy.
    pub staging_dir: PathBuf,
    /// 040-R8 (c) / ④: recorded winner (player name) by replay SHA-256 (043-R8 `collect.jsonl`).
    pub recorded_winners: HashMap<String, String>,
    pub jobs: usize,
    pub shard_target_bytes: usize,
    /// 040-R12 / 060-R7 storage budget for this run's shards.
    pub budget_bytes: u64,
    pub first_shard_index: u32,
    /// SHA-256 of replays already in another source (040 dedup by file hash); skipped.
    pub skip_sha256: std::collections::HashSet<String>,
    /// What this source's records already hold (resume); not processed again.
    pub done: ResumeDone,
    /// 040-R20: later copies of identical files in the whole source.
    pub duplicates: Duplicates,
    /// 040-R21: steps after this frame are not stored (`--until-frame`; the 021-R8 teacher data
    /// cut at 8:00 = 11,520). `None` = the whole game.
    pub until_frame: Option<u32>,
    /// 040-R21: in a replay of at least this many frames the first leave is the generating plan's
    /// time limit (`--time-limit`, the teacher botgen plans' `max_frames` 14,400), not a player.
    pub time_limit: Option<u32>,
    /// 041-R22 (`--relabel`, source `dagger`): the labels are this teacher's commands on the
    /// perspective's observations instead of the replay's own commands.
    pub relabel: Option<uranium_scripted::T0Build>,
}

/// Resume state from earlier records, per perspective (040-R17 / 040-R19 review): a written
/// perspective's record waits for its shard flush while another perspective of the same replay
/// (filtered, desync, ...) is recorded at once, so "the replay has a record" does not mean every
/// perspective is done. A perspective is done when it has its own record; a replay is done as a
/// whole only through a record without a view (rejected before perspectives exist).
#[derive(Debug, Default, Clone)]
pub struct ResumeDone {
    /// Replays with a record without a view.
    pub whole: std::collections::HashSet<String>,
    /// (replay SHA-256, perspective) with their own record.
    pub views: std::collections::HashSet<(String, u8)>,
}

impl ResumeDone {
    /// Adds one records line (JSON); lines that do not parse are ignored.
    pub fn add_line(&mut self, line: &str) {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else { return };
        let Some(sha) = v["replay_sha256"].as_str().filter(|s| !s.is_empty()) else { return };
        // 040-R20: the second file of a replay says nothing about the first file's progress.
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

    /// Replay handled as a whole by an earlier run.
    pub fn replay_done(&self, sha: &str) -> bool {
        self.whole.contains(sha)
    }

    /// The perspectives of `views` still without a record.
    pub fn remaining(&self, sha: &str, views: Vec<View>) -> Vec<View> {
        views.into_iter().filter(|v| !self.views.contains(&(sha.to_string(), v.perspective))).collect()
    }

    /// Records counted (replays done as a whole + perspectives done).
    pub fn len(&self) -> usize {
        self.whole.len() + self.views.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Builds shards from `files`; returns one record per replay view (or per rejected replay).
/// Shard writer plus the records of games waiting in its unflushed shard.
struct Pending {
    writer: ShardWriter,
    records: Vec<Record>,
}

/// Adds one written game under the `pending` lock; its record (and those of the games before it
/// in the same shard) reach `sink` only once the shard is verified on the shared drive (040-R17).
/// Returns false when the build has failed: then nothing is added, and a shard whose copy failed
/// drops its records for good (they are never sunk, so a resumed build processes those replays
/// again) while the writer refuses further writes, keeping the failed local copy intact.
fn add_game(p: &mut Pending, failure: &Mutex<Option<anyhow::Error>>, game: ShardGame, r: Record, sink: &(dyn Fn(&Record) + Sync)) -> bool {
    // Checked under the `pending` lock: no worker adds after another worker's flush failed.
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

/// Builds shards from `files`, handing every record to `sink` as soon as it is final: a
/// written game's record only after the shard holding it is on disk, so an interrupted run
/// can resume from its records (`BuildOptions::done`, per perspective).
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
        return Vec::new(); // already recorded by an earlier run
    }
    // 040-R20: a later copy of a file anywhere in the source (decided before the part split).
    if let Some(first) = opt.duplicates.earlier_copy(index, &sha) {
        return vec![rec(&sha, "", "", None, Outcome::Filtered { reason: format!("{DUPLICATE_IN_SOURCE}: same file as global index {first}") })];
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
    *map_hash_cell.borrow_mut() = map_hash.clone();
    let family = families.classify(&rep.header.map_name, &map_hash);
    if let Some(name) = excludes.matches(&rep.header.map_name, &map_hash) {
        return vec![rec(&sha, &rep.header.map_name, &family, None, Outcome::Filtered { reason: format!("excluded map (040-R16): {name}") })];
    }
    let views = match filter(&rep, &family, matches!(opt.src.as_str(), "bot" | "botgen" | "dagger"), opt.relabel.is_some()) {
        Ok(v) => v,
        Err(reason) => return vec![rec(&sha, &rep.header.map_name, &family, None, Outcome::Filtered { reason })],
    };
    // Resume: only perspectives without their own record (an earlier run may have recorded one
    // perspective and lost the other with an unflushed shard).
    let views = opt.done.remaining(&sha, views);
    if views.is_empty() {
        return Vec::new();
    }
    // 013-R7 (C-7): the 1.16.1 conversion must round-trip losslessly before it is replayed.
    if let Err(e) = rep.roundtrip_check() {
        let reason = format!("{e:#}");
        let outcome = if reason.starts_with("review:") { Outcome::Review { reason } } else { Outcome::Error { reason: format!("013-R7 round trip: {reason}") } };
        return vec![rec(&sha, &rep.header.map_name, &family, None, outcome)];
    }
    let recorded_name = opt.recorded_winners.get(&sha).map(String::as_str);
    let recorded_winner = recorded_slot(&rep, recorded_name);
    // A recorded winner that names neither player is reported, not silently dropped (040-R8 (c) / ④).
    *recorded_cell.borrow_mut() = recorded_name.map(|_| if recorded_winner.is_some() { "matched" } else { "unmatched" });
    if recorded_name.is_some() && recorded_winner.is_none() {
        eprintln!("040-R8: recorded winner of {} matches no player name", &sha[..12]);
    }
    let mut out = Vec::new();
    for view in views {
        if *written.lock().unwrap() >= opt.budget_bytes {
            out.push(rec(&sha, &rep.header.map_name, &family, Some(view), Outcome::Filtered { reason: "storage budget reached (040-R12)".into() }));
            continue;
        }
        let vo = ViewOptions { until_frame: opt.until_frame, time_limit: opt.time_limit, relabel: opt.relabel, map_hash: &map_hash };
        let outcome = match build_view(&rep, &view, cfg, labeler, recorded_winner, vo) {
            Ok(Ok(ViewGame { steps, coverage, result, result_source, game_info })) => {
                let n = steps.len();
                // 031 / AMB-031-3: map precompute once per map (observation channels only).
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
                // 040-R19: a perspective with too many invalid labels is not stored.
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
                let (r, _) = rec(&sha, &rep.header.map_name, &family, Some(view), Outcome::Written { steps: n, coverage, result, result_source: result_source.to_string() });
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

    /// 040-R21: `--until-frame 11520` keeps the steps at frames <= 11,520 (8:00 included).
    #[test]
    fn until_frame_cuts_later_steps_040_r21() {
        let mut v = vec![1u32, 11_519, 11_520, 11_521, 20_000];
        cut_after(&mut v, None, |f| *f);
        assert_eq!(v.len(), 5);
        cut_after(&mut v, Some(11_520), |f| *f);
        assert_eq!(v, vec![1, 11_519, 11_520]);
    }

    #[test]
    fn map_families_classify_040_r15() {
        let f = MapFamilies::load().unwrap();
        assert_eq!(f.classify("| iCCup | Fighting Spirit", ""), "fighting_spirit");
        assert_eq!(f.classify("투혼 1.3", ""), "fighting_spirit");
        assert_eq!(f.classify("\u{3}Circuit Breakers \u{5}1.0", ""), "circuit_breaker");
        assert_eq!(f.classify("Python", ""), "other");
    }

    /// 040-R8 (A-3): bridge -> leave -> recorded -> undecided (result 0, `none`).
    #[test]
    fn result_order_bridge_leave_recorded_none_040_r8() {
        let p = PlayResult { player_slots: vec![0, 1], ..Default::default() };
        assert_eq!(judge(&PlayResult { winner: Some(0), first_leaver_slot: Some(0), ..p.clone() }, Some(1), 0), (1, result_source::BRIDGE));
        // The perspective (slot 0) left first: a loss even if a recorded winner says otherwise
        // (that disagreement is a desync, ④).
        assert_eq!(judge(&PlayResult { first_leaver_slot: Some(0), ..p.clone() }, None, 0), (-1, result_source::LEAVE));
        assert_eq!(judge(&PlayResult { first_leaver_slot: Some(1), ..p.clone() }, None, 0), (1, result_source::LEAVE));
        assert_eq!(judge(&p, Some(1), 0), (-1, result_source::RECORDED));
        assert_eq!(judge(&p, None, 0), (0, result_source::NONE));
    }

    fn written_record(n: u32) -> Record {
        let coverage = Coverage::default();
        Record { index: n as usize, replay_sha256: format!("{n:064x}"), src: "t".into(), map_family: "other".into(), map_name: String::new(), map_hash: String::new(), recorded_winner: None, view: None, labeler_version: crate::label::LABELER_VERSION, until_frame: None, time_limit: None, relabel: None, outcome: Outcome::Written { steps: 3, coverage, result: 0, result_source: "none".into() } }
    }

    /// 040-R17 (review M1): a shard whose copy fails verification stops the build. Its records
    /// (the pending ones before it too) are never sunk, later adds by other workers are refused,
    /// and the kept local copy is not overwritten.
    #[test]
    fn failed_shard_records_are_never_sunk_040_r17() {
        use crate::shard::tests::{corrupting_copy, game};
        let dir = std::env::temp_dir().join(format!("uranium-pipeline-m1-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let one = postcard::experimental::serialized_size(&game(1)).unwrap();
        // The second game fills the shard and triggers the (failing) flush.
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
        // Another worker adds afterwards: refused before touching the writer.
        p.writer.copy = |from, to| std::fs::copy(from, to);
        assert!(!add_game(&mut p, &failure, game(3), written_record(3), &sink));
        assert!(!add_game(&mut p, &failure, game(4), written_record(4), &sink));
        assert!(sunk.lock().unwrap().is_empty());
        assert_eq!(local("local"), kept);
        assert_eq!(std::fs::read(&kept[0]).unwrap(), bytes);
        assert!(local("shared").is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// 040-R17 / 040-R19 resume: perspective 0 is filtered (recorded at once) while perspective 1
    /// is written but its shard not yet flushed when the run stops. The resumed run builds only
    /// perspective 1, and afterwards the replay is done with one record per perspective.
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
        // First run: a large shard target, so nothing is flushed before the interruption.
        let writer = ShardWriter::new(dir.join("shared"), dir.join("local"), 0, 1 << 30).unwrap();
        let mut p = Pending { writer, records: Vec::new() };
        let failure = Mutex::new(None);
        sink(&record(0, Outcome::Filtered { reason: "invalid_labels>=20% (54.6%, 20444/37441) (040-R19)".into() }));
        assert!(add_game(&mut p, &failure, game(1), record(1, written_record(7).outcome), &sink));
        drop(p); // interrupted: the written perspective's record never reaches the records file
        let mut done = ResumeDone::default();
        lines.lock().unwrap().iter().for_each(|l| done.add_line(l));
        assert!(!done.replay_done(&sha));
        let todo: Vec<u8> = done.remaining(&sha, vec![view(0), view(1)]).iter().map(|v| v.perspective).collect();
        assert_eq!(todo, vec![1], "only the lost perspective is built again");
        // Resumed run writes perspective 1; a third run has nothing left.
        sink(&record(1, written_record(7).outcome));
        let mut done = ResumeDone::default();
        lines.lock().unwrap().iter().for_each(|l| done.add_line(l));
        assert!(done.remaining(&sha, vec![view(0), view(1)]).is_empty());
        assert_eq!(done.len(), 2, "one record per perspective, no duplicates");
        // A record without a view (rejected replay) marks the replay done as a whole.
        let mut whole = ResumeDone::default();
        whole.add_line(&serde_json::to_string(&Record { outcome: Outcome::Filtered { reason: "shorter than 5 minutes".into() }, ..written_record(9) }).unwrap());
        assert!(whole.replay_done(&format!("{:064x}", 9)));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// 2026-10-19 (labeler version 2 mid-rebuild): records of the labeler-1 binary (no
    /// `labeler_version` field) resume exactly as before: the same perspectives and replays are
    /// done, nothing is built twice; new records carry version 2.
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
        // Two parts (`i % 2`): global 0, 2 in part 0; 1, 3 in part 1. The copies at global 2
        // (part 0) and 3 (part 1) are decided on the full list, so both parts agree.
        let part0 = Duplicates::find(&files, &[0, 2], 2);
        let part1 = Duplicates::find(&files, &[1, 3], 2);
        assert_eq!(part0.first[&sha("xx")], 0, "the first file in sorted order keeps the replay");
        assert_eq!(part1.first[&sha("xx")], 0);
        assert!(!part1.first.contains_key(&sha("y")), "a file of unique size is not read");
        assert_eq!((part0.repeats, part1.repeats), (1, 1));
        // Run-local indices: part 0 file 0 = global 0 (kept), file 1 = global 2 (copy).
        assert_eq!(part0.earlier_copy(0, &sha("xx")), None);
        assert_eq!(part0.earlier_copy(1, &sha("xx")), Some(0));
        assert_eq!(part1.earlier_copy(0, &sha("y")), None);
        assert_eq!(part1.earlier_copy(1, &sha("xx")), Some(0));
        // The first file was not hashed in this run (transient read failure): a later first
        // index never filters an earlier file, so at most one extra game results.
        let missed = Duplicates { global: vec![0, 2], first: HashMap::from([(sha("xx"), 2)]), repeats: 0 };
        assert_eq!(missed.earlier_copy(0, &sha("xx")), None);
        assert_eq!(missed.earlier_copy(1, &sha("xx")), None);

        // process(): the copy is filtered before parsing and names the global index.
        let families = MapFamilies::load().unwrap();
        let excludes = MapExcludes::load(&families).unwrap();
        let labeler = Labeler::new().unwrap();
        let cfg = PlayConfig { launcher: dir.join("none"), bridge_lib: dir.join("none"), data_dir: dir.join("none"), work_dir: dir.join("w") };
        let opt = BuildOptions { src: "t".into(), shards_dir: dir.join("s"), staging_dir: dir.join("l"), recorded_winners: HashMap::new(), jobs: 1, shard_target_bytes: 1, budget_bytes: 1 << 30, first_shard_index: 0, skip_sha256: Default::default(), done: ResumeDone::default(), duplicates: part1, until_frame: None, time_limit: None, relabel: None };
        let out = process(1, &files[3], &opt, &cfg, &families, &excludes, &labeler, &Mutex::new(0));
        assert_eq!(out.len(), 1);
        let (r, game) = &out[0];
        assert!(game.is_none() && r.view.is_none() && r.replay_sha256 == sha("xx"));
        assert!(matches!(&r.outcome, Outcome::Filtered { reason } if reason == &format!("{DUPLICATE_IN_SOURCE}: same file as global index 0")), "{:?}", r.outcome);
        // The kept file goes on (here: it is not a replay, so it errors, but not as a duplicate).
        let kept = process(0, &files[1], &opt, &cfg, &families, &excludes, &labeler, &Mutex::new(0));
        assert!(!matches!(&kept[0].0.outcome, Outcome::Filtered { reason } if reason.starts_with(DUPLICATE_IN_SOURCE)));

        // A duplicate's record (no view) does not mark the replay done for resume.
        let mut done = ResumeDone::default();
        done.add_line(&serde_json::to_string(r).unwrap());
        assert!(!done.replay_done(&sha("xx")) && done.is_empty());
        std::fs::remove_dir_all(&dir).ok();
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

    /// 041-R22: `dagger` games have no 5-minute bar and are bot data (stage B, no APM bar).
    #[test]
    fn dagger_views_have_no_length_bar_041_r22() {
        let player = |id: u8, race: u8| crate::Player { index: id, slot: id as u16, id, kind: 2, race, team: id, name: vec![b'p', b'0' + id] };
        let train = |p: u8| cmd::Cmd { frame: 10, player: p, op: cmd::op::TRAIN, data: vec![7, 0] };
        let short = Replay {
            format: crate::Format::Legacy,
            header: crate::Header { frames: 3 * 60 * 24, map_width: 128, map_height: 112, game_type: 2, map_name: "t".into(), players: vec![player(0, header::RACE_TERRAN), player(1, header::RACE_ZERG)] },
            raw_header: vec![],
            blocks: vec![cmd::Block { frame: 10, cmds: vec![train(0), train(1)] }],
            map: vec![],
            unit_limit: None,
        };
        assert_eq!(filter(&short, "fighting_spirit", true, false).unwrap_err(), "shorter than 5 minutes");
        let views = filter(&short, "fighting_spirit", true, true).unwrap();
        assert!(!views.is_empty() && views.iter().all(|v| v.stage == "B"));
    }

    /// 041-R22 (OpenBW, ignored): relabelling a model game (`URANIUM_TEST_DAGGER_REP`, an OpenBW
    /// replay of a model game; `.env` launcher, bridge and `URANIUM_MAPS_DIR`) twice gives the same
    /// steps, and the labels are the 021-R8 teacher's (only the actions of its 034 table, the
    /// "모델 표현" row) rather than the model's own commands.
    #[test]
    #[ignore = "needs OpenBW (.env) and URANIUM_TEST_DAGGER_REP"]
    fn relabel_is_deterministic_and_from_the_teacher_041_r22() {
        let _ = dotenvy::dotenv();
        let path = std::env::var("URANIUM_TEST_DAGGER_REP").expect("URANIUM_TEST_DAGGER_REP");
        let bytes = std::fs::read(&path).unwrap();
        let rep = crate::parse(&bytes).unwrap();
        let map_hash = hex(&Sha256::digest(&rep.map));
        let families = MapFamilies::load().unwrap();
        let family = families.classify(&rep.header.map_name, &map_hash);
        let view = filter(&rep, &family, true, true).unwrap().remove(0);
        let cfg = PlayConfig::from_env(crate::play::work_dir_default().join("relabel-test")).unwrap();
        let labeler = Labeler::new().unwrap();
        let run = |relabel| {
            let vo = ViewOptions { until_frame: Some(11_520), time_limit: Some(14_400), relabel, map_hash: &map_hash };
            match build_view(&rep, &view, &cfg, &labeler, None, vo).unwrap() {
                Ok(g) => g,
                Err(o) => panic!("{o:?}"),
            }
        };
        let (a, b) = (run(Some(uranium_scripted::T0Build::TerranReact)), run(Some(uranium_scripted::T0Build::TerranReact)));
        assert_eq!(a.steps.len(), b.steps.len());
        assert!(a.steps == b.steps, "the same replay gives the same steps");
        let own = run(None);
        let table = uranium_env::actions::ActionTable::load().unwrap();
        let name = |s: &Step| table.actions.get(s.action_label.action as usize).map_or("unknown", |a| a.name.as_str()).to_string();
        let teacher = [
            "NoOp", "Build_SupplyDepot", "Build_Barracks", "Build_CommandCenter", "Build_Bunker", "Train_SCV", "Train_Marine", "Move_Pos", "Smart_Unit", "Hold", "Attack_Pos", "Attack_Unit", "Repair_Unit",
            "Gather_Unit", "ReturnCargo",
        ];
        let off: Vec<String> = a.steps.iter().map(name).filter(|n| !teacher.contains(&n.as_str())).collect();
        assert!(off.is_empty(), "labels outside the teacher's actions: {off:?}");
        assert!(a.steps.iter().any(|s| name(s) == "Train_SCV") && a.steps.iter().any(|s| name(s) == "Move_Pos"));
        let labels = |g: &ViewGame| g.steps.iter().map(|s| s.action_label.clone()).collect::<Vec<_>>();
        assert_ne!(labels(&a), labels(&own), "not the model's own commands");
        eprintln!("teacher steps {} vs own {}; coverage {:?}", a.steps.len(), own.steps.len(), a.coverage);
    }

    /// 041-R22 (OpenBW, ignored): relabelling a teacher botgen game (`URANIUM_TEST_TEACHER_REP`,
    /// the teacher as a Terran player) with the teacher gives its botgen labels: the shadow sees
    /// the same observations the live teacher saw, so its commands are the live ones and the
    /// BWAPI path (`relabel::command_cmds`) gives the same labels as the recorded commands.
    #[test]
    #[ignore = "needs OpenBW (.env) and URANIUM_TEST_TEACHER_REP"]
    fn relabel_of_a_teacher_game_gives_its_botgen_labels_041_r22() {
        let _ = dotenvy::dotenv();
        let path = std::env::var("URANIUM_TEST_TEACHER_REP").expect("URANIUM_TEST_TEACHER_REP");
        let rep = crate::parse(&std::fs::read(&path).unwrap()).unwrap();
        let map_hash = hex(&Sha256::digest(&rep.map));
        let family = MapFamilies::load().unwrap().classify(&rep.header.map_name, &map_hash);
        let view = filter(&rep, &family, true, true).unwrap().remove(0);
        let cfg = PlayConfig::from_env(crate::play::work_dir_default().join("relabel-test")).unwrap();
        let labeler = Labeler::new().unwrap();
        let run = |relabel| {
            let vo = ViewOptions { until_frame: Some(11_520), time_limit: Some(14_400), relabel, map_hash: &map_hash };
            match build_view(&rep, &view, &cfg, &labeler, None, vo).unwrap() {
                Ok(g) => g,
                Err(o) => panic!("{o:?}"),
            }
        };
        let (shadow, botgen) = (run(Some(uranium_scripted::T0Build::TerranReact)), run(None));
        // (action, sorted selection, target unit, target tile)
        type Key = (u16, Vec<u32>, Option<u32>, Option<(u16, u16)>);
        let key = |s: &Step| -> Key {
            let l = &s.action_label;
            let mut sel = l.selected.clone();
            sel.sort_unstable();
            (l.action, sel, l.target_unit, l.target_tile)
        };
        let acts = |g: &ViewGame| g.steps.iter().filter(|s| s.action_label.action != 0).map(|s| (s.action_label.frame, key(s))).collect::<Vec<_>>();
        let (a, b) = (acts(&shadow), acts(&botgen));
        // Each botgen label matched by an unused equal shadow label within 8 frames.
        let mut used = vec![false; a.len()];
        let mut offsets: std::collections::BTreeMap<i64, usize> = Default::default();
        let mut matched = 0;
        for (fb, kb) in &b {
            if let Some(i) = (0..a.len()).find(|&i| !used[i] && a[i].1 == *kb && (a[i].0 as i64 - *fb as i64).abs() <= 8) {
                used[i] = true;
                matched += 1;
                *offsets.entry(*fb as i64 - a[i].0 as i64).or_default() += 1;
            }
        }
        eprintln!("botgen actions {} shadow actions {} matched {} offsets {offsets:?}", b.len(), a.len(), matched);
        if std::env::var_os("URANIUM_TEST_VERBOSE").is_some() {
            let table = uranium_env::actions::ActionTable::load().unwrap();
            let show = |(f, k): &(u32, Key)| format!("{f} {} {:?} {:?} {:?}", table.actions[k.0 as usize].name, k.1, k.2, k.3);
            for (i, x) in a.iter().enumerate().filter(|(i, _)| !used[*i]) {
                eprintln!("shadow only #{i}: {}", show(x));
            }
            let mut used_b = vec![false; a.len()];
            for x in &b {
                match (0..a.len()).find(|&i| !used_b[i] && a[i].1 == x.1 && (a[i].0 as i64 - x.0 as i64).abs() <= 8) {
                    Some(i) => used_b[i] = true,
                    None => eprintln!("botgen only: {}", show(x)),
                }
            }
        }
        assert!(matched as f64 >= 0.95 * b.len() as f64 && matched as f64 >= 0.95 * a.len() as f64);
    }
}
