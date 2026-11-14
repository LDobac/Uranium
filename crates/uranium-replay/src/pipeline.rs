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
use crate::relabel::MemoryCheck;
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

pub(crate) fn race_letter(r: u8) -> char {
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

/// The [`filter`] flags of a source: (`bot`: stage B, no APM bar; `any_length`: no 5-minute bar).
/// Bot data (040 quality table): `bot`, `botgen`, `dagger`, 041-R28 `branch` and 044-R2 `distill`.
/// Any length: relabelled (`dagger`, 041-R22) games, `branch` trunks (a trunk lost early to the rush
/// is a point) and `distill` games (044-R2: no 5-minute bar, as `dagger`).
pub fn filter_rules(src: &str, relabelled: bool) -> (bool, bool) {
    let bot = matches!(src, "bot" | "botgen" | "dagger" | BRANCH_SRC | DISTILL_SRC);
    (bot, relabelled || src == BRANCH_SRC || src == DISTILL_SRC)
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum Outcome {
    /// `result` / `result_source`: 040-R8 result of the view (dataset statistics, 040-R6).
    /// `repeats_dropped`: 041-R22 teacher (select, order) command pairs dropped as repeats
    /// (relabelled views only).
    Written {
        steps: usize,
        coverage: Coverage,
        result: i8,
        result_source: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        repeats_dropped: Option<u32>,
        /// 041-R28 ④ / ⑤: the model decision log check of a `--memory-log` build (`ok` = every
        /// step has `memory_action`, otherwise none has).
        #[serde(skip_serializing_if = "Option::is_none")]
        memory: Option<crate::relabel::MemoryCheck>,
    },
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

/// 040-R3 / R4: the perspective's own replay commands (1.16.1 form, the opcodes OpenBW runs) as
/// labels, each against the own units observed at its command frame (`units_at`: frame -> unit id
/// -> `Own`). Shared by `build` (replay A) and `granularity` (G0 measurement).
pub(crate) fn own_labels(rep: &Replay, perspective: u8, labeler: &Labeler, units_at: &HashMap<u32, HashMap<u32, crate::label::Own>>) -> Result<(Vec<crate::label::Label>, Coverage)> {
    let cmds: Vec<cmd::Cmd> = rep
        .cmds()
        .filter(|c| c.player == perspective)
        .map(|c| cmd::to_legacy(c, rep.unit_limit))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .filter(|c| cmd::openbw(c.op) == cmd::OpenBw::Supported)
        .collect();
    let empty = HashMap::new();
    Ok(labeler.label(&cmds, rep.header.frames, |f| units_at.get(&f).unwrap_or(&empty)))
}

pub(crate) fn hex(b: &[u8]) -> String {
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
    /// 041-R22: teacher (select, order) command pairs dropped as repeats (`relabel::REPEAT_WINDOW`);
    /// `None` = not relabelled.
    pub repeats_dropped: Option<u32>,
    /// 041-R28 ④ / ⑤: the decision log check (`--memory-log` builds only).
    pub memory: Option<crate::relabel::MemoryCheck>,
}

/// How one view is labelled (040-R21 cut and time limit, 041-R22 relabelling).
#[derive(Clone, Copy)]
struct ViewOptions<'a> {
    until_frame: Option<u32>,
    time_limit: Option<u32>,
    relabel: Option<uranium_scripted::T0Build>,
    /// CHK SHA-256 (map precompute file name, 031).
    map_hash: &'a str,
    /// 041-R28: the replay's model decision logs (`--memory-log`; relabelled or not, 051-R13 ④).
    memory: Option<&'a [crate::relabel::DecisionLogRead]>,
    /// 044-R2 (`--src distill`): the steps are the game start and the logged model decisions only,
    /// with `unknown` label slots; a view whose log fails 041-R28 ④ / ⑤ is `filtered`.
    distill: bool,
}

/// 044-R2: the `filtered` reason of a distill view whose decision log fails 041-R28 ④ (no complete
/// log of the view's player: `missing`, `incomplete`, `other_player`, e.g. the T0 opponent's view of
/// a TvT game) or ⑤ (`unit_mismatch`).
pub fn distill_filter_reason(m: &MemoryCheck) -> String {
    let mut s = format!("{DISTILL_FILTER_PREFIX} {} (041-R28, 044-R2): {} decisions, {} checked, {} mismatched", m.status, m.decisions, m.checked, m.mismatched);
    if let Some(r) = &m.reason {
        s.push_str(": ");
        s.push_str(r);
    }
    s
}

/// Start of the 044-R2 `filtered` reason (statistics group by it).
pub const DISTILL_FILTER_PREFIX: &str = "decision_log";

/// Processes one view of one replay into a shard game.
fn build_view(rep: &Replay, view: &View, cfg: &PlayConfig, labeler: &Labeler, recorded_winner: Option<u8>, o: ViewOptions) -> Result<std::result::Result<ViewGame, Outcome>> {
    let (until_frame, time_limit) = (o.until_frame, o.time_limit);
    let me = slot_of(rep, view.perspective)?;
    // The last frame the relabelling teacher is fed: the last label before a cut takes its delay
    // from up to 32 frames later (031-R11).
    let last = until_frame.map_or(rep.header.frames, |u| u.saturating_add(32)).min(rep.header.frames);
    // 041-R28 ⑤: own unit ids of the perspective at the playback frames of the model's decisions.
    let mut own_at: HashMap<u32, std::collections::HashSet<u32>> = HashMap::new();
    // 041-R28 ④: the log is used only for its own player's view of the game it was written for.
    let bound = o.memory.map(|reads| crate::relabel::bind_logs(reads, me, rep.header.frames));
    // 044-R2: a distill view needs its player's complete log (④); without one it is filtered before
    // any replay (the opponent's view of a TvT game has none).
    if o.distill {
        let reason = match &bound {
            Some(Ok(_)) => None,
            Some(Err(m)) => Some(distill_filter_reason(m)),
            None => Some(distill_filter_reason(&MemoryCheck::without_log(&crate::relabel::DecisionLogRead::Missing))),
        };
        if let Some(reason) = reason {
            return Ok(Err(Outcome::Filtered { reason }));
        }
    }
    // 041-R28 ⑤: replay A (or T) also stops at the model decisions' playback frames for the unit
    // check. Extra stops only split the events between observations (the 040-R8 check ignores them).
    let check_frames = match &bound {
        Some(Ok(log)) => crate::relabel::decision_frames(&log.decisions, last),
        _ => Vec::new(),
    };
    let (mut a, mut labels, coverage, repeats_dropped) = match o.relabel {
        // Replay A: own unit types at each command frame + desync signals.
        None => {
            let mut units_at: HashMap<u32, HashMap<u32, crate::label::Own>> = HashMap::new();
            // 041-R28 (051-R13 ④): a build without relabelling takes `--memory-log` too; the
            // labels stay the replay's own commands, only the step memory is the live session's.
            let a = play_with(rep, view.perspective, cfg, &check_frames, &mut |f, _| {
                let own: HashMap<u32, crate::label::Own> = f.units.iter().filter(|u| u.owner == me).map(|u| (u.id, crate::label::Own::from_state(u))).collect();
                if check_frames.binary_search(&f.frame).is_ok() {
                    own_at.insert(f.frame, own.keys().copied().collect());
                }
                units_at.insert(f.frame, own);
            })?;
            let (labels, coverage) = own_labels(rep, view.perspective, labeler, &units_at)?;
            (a, labels, coverage, None)
        }
        // 041-R22 replay T: the same desync signals as replay A, and the teacher run as a shadow
        // on the perspective's observations; its commands are the labels (the player's own
        // commands only feed the 040-R8 checks).
        Some(build) => {
            let mut stops = crate::relabel::teacher_frames(last);
            // 041-R28 ⑤: T also stops at the check frames; the teacher reads no events, so its
            // labels are the same.
            stops.extend(&check_frames);
            stops.sort_unstable();
            stops.dedup();
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
                    if check_frames.binary_search(&f.frame).is_ok() {
                        own_at.insert(f.frame, f.units.iter().filter(|u| u.owner == me).map(|u| u.id).collect());
                    }
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
            (a, labels, coverage, Some(shadow.repeats_dropped))
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
    // 041-R28 decision-time steps (user decision 2026-11-13, 051-R13 ④): builds without relabelling
    // whose log is used also observe every logged decision's playback frame.
    let limit = until_frame.map_or(last, |u| u.min(last));
    let decision_steps = match step_labels(o.distill, o.relabel.is_some(), bound.as_ref(), &own_at, last, limit, rep.header.frames, &mut labels) {
        Ok(n) => n,
        Err(reason) => return Ok(Err(Outcome::Filtered { reason })),
    };
    let frames: Vec<u32> = labels.iter().map(|l| l.frame).collect();
    // Replay B: decision-time observations. B stops at every base stop of A (or T, 041-R22), so
    // they form the 040-R8 double-replay check (check_digest hashes the base stops only and
    // ignores the extra stops: B's label frames, T's teacher observations).
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
        steps.push(Step { raw_frame, privileged, action_label: l, memory_action: None });
    }
    // 040-R18: the sequence must start with the game-start observation.
    if let Some(&f) = frames.first().filter(|_| steps.is_empty()) {
        return Ok(Err(Outcome::Error { reason: format!("no observation at the first step frame {f} (040-R18)") }));
    }
    // 044-R2: steps past the last observation replay B gave are gone; they were all decision steps
    // (the game-start step is the first and is kept).
    let decision_steps = if o.distill { decision_steps.map(|n| n - (frames.len() - steps.len())) } else { decision_steps };
    let mut res = res;
    res.drop_time_limit_leave(time_limit);
    let (result, result_source) = judge(&res, recorded_winner, me);
    let Some(game_info) = res.game_info else {
        return Ok(Err(Outcome::Error { reason: "bridge sent no GameInfo (AMB-040-6)".into() }));
    };
    let memory = bound.map(|b| MemoryCheck { decision_steps, ..attach_memory(&mut steps, b, last, &own_at) });
    // 044-R2 (044-R3 contract): a written distill game has `memory_action` on every step.
    if o.distill && steps.iter().any(|s| s.memory_action.is_none()) {
        return Ok(Err(Outcome::Error { reason: "distill step without memory_action (044-R2)".into() }));
    }
    Ok(Ok(ViewGame { steps, coverage, result, result_source, game_info, repeats_dropped, memory }))
}

/// 041-R28 decision-time steps (2026-11-13): adds them ([`crate::relabel::add_decision_steps`]) only
/// in a build without relabelling (`relabelled` false) whose log is bound to the view and passes the
/// ⑤ unit check (the log will fill the memory); relabelled builds and unused logs keep `labels` as
/// they are. Returns the steps added (`None` = not this build's case).
pub fn decision_steps(
    relabelled: bool,
    bound: Option<&Result<&crate::relabel::ModelLog, MemoryCheck>>,
    own_at: &HashMap<u32, std::collections::HashSet<u32>>,
    last: u32,
    limit: u32,
    end_frame: u32,
    labels: &mut Vec<crate::label::Label>,
) -> Option<usize> {
    let Some(Ok(log)) = bound.filter(|_| !relabelled) else { return None };
    let (checked, mismatched) = crate::relabel::unit_check(&log.decisions, last, own_at);
    MemoryCheck::checked(log.decisions.len(), checked, mismatched).ok().then(|| crate::relabel::add_decision_steps(labels, &log.decisions, limit, end_frame))
}

/// The step labels of a view after the 040-R21 cut (`labels`: the replay's own command labels, or
/// the 041-R22 teacher's): a `distill` view (044-R2) replaces them with [`distill_steps`] (the
/// replay's own, model, commands made them for the 040-R8 checks only; `Err` = its `filtered`
/// reason); any other build keeps them and adds the 041-R28 decision-time steps where they apply
/// ([`decision_steps`]). Returns the steps at decision frames.
#[allow(clippy::too_many_arguments)]
pub fn step_labels(
    distill: bool,
    relabelled: bool,
    bound: Option<&Result<&crate::relabel::ModelLog, MemoryCheck>>,
    own_at: &HashMap<u32, std::collections::HashSet<u32>>,
    last: u32,
    limit: u32,
    end_frame: u32,
    labels: &mut Vec<crate::label::Label>,
) -> std::result::Result<Option<usize>, String> {
    if !distill {
        return Ok(decision_steps(relabelled, bound, own_at, last, limit, end_frame, labels));
    }
    let (slots, n) = distill_steps(bound, own_at, last, limit, end_frame)?;
    *labels = slots;
    Ok(Some(n))
}

/// 044-R2 (`--src distill`): the steps of a view, replacing the replay's command labels: the
/// game-start step and a step per logged decision up to `limit`, all `unknown` label slots
/// ([`crate::relabel::distill_labels`]), with the steps at decision frames; or the `filtered`
/// reason when the view's log fails 041-R28 ④ (no complete log of its player) or ⑤ (more than 1%
/// of its selections not own units at their playback frames, `own_at`, up to `last`).
pub fn distill_steps(
    bound: Option<&Result<&crate::relabel::ModelLog, MemoryCheck>>,
    own_at: &HashMap<u32, std::collections::HashSet<u32>>,
    last: u32,
    limit: u32,
    end_frame: u32,
) -> std::result::Result<(Vec<crate::label::Label>, usize), String> {
    let log = match bound {
        Some(Ok(log)) => log,
        Some(Err(m)) => return Err(distill_filter_reason(m)),
        None => return Err(distill_filter_reason(&MemoryCheck::without_log(&crate::relabel::DecisionLogRead::Missing))),
    };
    let (checked, mismatched) = crate::relabel::unit_check(&log.decisions, last, own_at);
    let m = MemoryCheck::checked(log.decisions.len(), checked, mismatched);
    if !m.ok() {
        return Err(distill_filter_reason(&m));
    }
    Ok(crate::relabel::distill_labels(&log.decisions, limit, end_frame))
}

/// 041-R28 ② / ④ / ⑤: fills every step's `memory_action` from a decision log bound to this view
/// ([`crate::relabel::bind_log`]) whose unit check passes, and leaves every step `None` otherwise
/// (all or none per game). `last`: the last frame checked (the teacher's range); `own_at`: own
/// unit ids at the decisions' playback frames.
pub fn attach_memory(steps: &mut [Step], bound: Result<&crate::relabel::ModelLog, crate::relabel::MemoryCheck>, last: u32, own_at: &HashMap<u32, std::collections::HashSet<u32>>) -> crate::relabel::MemoryCheck {
    use crate::relabel::{MemoryCheck, step_memory, unit_check};
    let log = match bound {
        Ok(l) => &l.decisions,
        Err(m) => return m,
    };
    let (checked, mismatched) = unit_check(log, last, own_at);
    let m = MemoryCheck::checked(log.len(), checked, mismatched);
    if m.ok() {
        let frames: Vec<u32> = steps.iter().map(|s| s.raw_frame.frame).collect();
        for (s, mem) in steps.iter_mut().zip(step_memory(log, &frames)) {
            s.memory_action = Some(mem);
        }
    }
    m
}

/// 041-R28 (051-R13 ④): source tag of the branch-game trunk builds (no relabelling, act-log memory,
/// decision-time steps): analysis shards, never BC data.
pub const BRANCH_SRC: &str = "branch";
/// 044-R2 / 040-R24: source tag of the self-game distillation builds (decision steps only, `unknown`
/// label slots for the 044-R3 policy labeller): BC data only after that labelling, in another folder.
pub const DISTILL_SRC: &str = "distill";

/// The sources whose build takes `--memory-log` without `--relabel` and whose shards stay outside
/// the BC shards root: 041-R28 `branch` trunks and 044-R2 `distill` games before labelling.
fn isolated(src: &str) -> bool {
    src == BRANCH_SRC || src == DISTILL_SRC
}

/// `uranium-replay build` option rules: 041-R22 relabelled games are their own source (040
/// `dagger`) and only they; 041-R28 `--memory-log` names a folder and goes with `--relabel` or,
/// without it, only with `--src branch` (051-R13 ④: branch-game trunks, whose command-label steps'
/// memory already holds the decision they are labelled with and whose decision-time steps are not
/// BC samples) or `--src distill` (044-R2, which needs it: its steps are the logged decisions). A
/// `branch` or `distill` build needs an explicit `--out` outside the BC shards root `bc_shards_root`
/// (`<shared>/datasets/replays/shards`, when known), so its shards never join a BC source (044-R2:
/// `distill` shards become BC data only through the 044-R3 labeller, which writes new shards).
pub fn check_build_args(src: &str, relabel: bool, memory_log: Option<&Path>, out: Option<&Path>, bc_shards_root: Option<&Path>) -> Result<()> {
    anyhow::ensure!((src == "dagger") == relabel, "--relabel goes with --src dagger and only with it (041-R22)");
    if let Some(d) = memory_log {
        anyhow::ensure!(d.is_dir(), "--memory-log {}: not a folder (041-R28)", d.display());
        anyhow::ensure!(relabel || isolated(src), "--memory-log without --relabel goes with --src {BRANCH_SRC} or --src {DISTILL_SRC} only (041-R28: branch trunk shards are not BC data; 044-R2)");
    }
    anyhow::ensure!(src != DISTILL_SRC || memory_log.is_some(), "--src {DISTILL_SRC} needs --memory-log <the plan's memory folder> (044-R2: a step per logged model decision)");
    if isolated(src) {
        let Some(out) = out else {
            anyhow::bail!("--src {src} needs an explicit --out outside the BC shards (e.g. <shared>/eval/branch-<date>/shards, 041-R28; <shared>/datasets/replays/distill_pre/d<R>, 044-R2)")
        };
        if let Some(root) = bc_shards_root {
            anyhow::ensure!(!path_within(out, root), "--src {src}: --out {} is inside the BC shards {} (041-R28, 044-R2: these shards are not BC data)", out.display(), root.display());
        }
    }
    Ok(())
}

/// 022-R2 / R12 / AC3 (2026-11-14): `--relabel terran_tp` needs `--games` (the generating job's
/// games.jsonl: lobby races and the players[0] view; without them a lobby-Random game or the
/// Terran opponent of a TvT game would be taken for data), and `--games` goes with that relabel or
/// with `--src botgen` (the 022 teacher's botgen games) only, so every other build is unchanged.
pub fn check_games_arg(src: &str, relabel: Option<uranium_scripted::T0Build>, games: bool) -> Result<()> {
    let tp = relabel == Some(uranium_scripted::T0Build::TerranTp);
    anyhow::ensure!(!tp || games, "--relabel terran_tp needs --games <the job's games.jsonl> (022-R2 / 022-AC3: lobby races; only players[0], the model, is a view)");
    anyhow::ensure!(!games || tp || (src == "botgen" && relabel.is_none()), "--games goes with --relabel terran_tp or --src botgen (the 022 teacher's botgen games, 022-R12) only");
    Ok(())
}

/// 051-R13 ④ branch trunk builds and 044-R2 distill builds: `--time-limit` (the generating plan's
/// `max_frames`: H + 48 for a trunk, 14,400 for a distill plan, 044-R1) is required, because those
/// games end at the time limit (without it the first leave counts as a player leaving and the 040-R8
/// desync checks can drop a view); and the records file name must start with the source tag (the
/// resume scan reads the records files of the source by that prefix; a records file it does not see
/// makes a second run build every view again, and the repeated replay hashes then make the dump's
/// view lookup ambiguous).
pub fn check_isolated_build(src: &str, records: &Path, time_limit: Option<u32>) -> Result<()> {
    if !isolated(src) {
        return Ok(());
    }
    anyhow::ensure!(time_limit.is_some(), "--src {src} needs --time-limit <the plan's max_frames> (051-R1, 051-R13 4: trunks end at the time limit; 044-R2: distill games end at the 044-R1 time limit 14400)");
    let name = records.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    anyhow::ensure!(name.starts_with(src) && name.ends_with(".jsonl"), "--records {}: the file name must start with `{src}` and end with .jsonl (the resume scan reads `<source>*.jsonl`; 051-R13 4, 044-R2)", records.display());
    // The scan reads every `<source>*.jsonl` of the records folder: not the 051-R12 output folder,
    // whose `branches.jsonl` it would take for build records.
    let dir = records.parent().filter(|d| !d.as_os_str().is_empty()).unwrap_or(Path::new("."));
    for other in ["branches.jsonl", "points.jsonl"] {
        anyhow::ensure!(!dir.join(other).exists(), "--records {}: its folder holds {other} (the 051-R12 output folder); give the build records a folder of their own (e.g. <out folder>/build/)", records.display());
    }
    Ok(())
}

/// `p` is `root` or below it (absolute paths compared component by component; ASCII case-insensitive
/// on Windows).
pub fn path_within(p: &Path, root: &Path) -> bool {
    let abs = |x: &Path| std::path::absolute(x).unwrap_or_else(|_| x.to_path_buf());
    let (p, root) = (abs(p), abs(root));
    let (mut a, b) = (p.components(), root.components());
    b.into_iter().all(|rc| {
        a.next().is_some_and(|pc| if cfg!(windows) { pc.as_os_str().eq_ignore_ascii_case(rc.as_os_str()) } else { pc == rc })
    })
}

/// 041-R28: `--memory-log` refuses a build whose replays share a file stem (the log is found by it).
pub fn unique_stems(files: &[PathBuf]) -> Result<()> {
    let mut seen: HashMap<String, &PathBuf> = HashMap::new();
    for f in files {
        let stem = f.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        if let Some(first) = seen.insert(stem.clone(), f) {
            anyhow::bail!("--memory-log: {} and {} have the same name {stem}: build one plan folder (with its memory folder) at a time (041-R28)", first.display(), f.display());
        }
    }
    Ok(())
}

/// 041-R28 (AMB-041-17): the decision log of replay `file` in the `--memory-log` folder `dir`:
/// the same game id, `g<id>.rep` -> `g<id>.jsonl` (`uranium_runner::memory_log_name`).
pub fn memory_log_path(dir: &Path, file: &Path) -> PathBuf {
    let stem = file.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    dir.join(format!("{stem}.jsonl"))
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
    let v = uranium_env::features::label::label_validity_with_memory(
        &game.game_info,
        uranium_env::maps::family_index(&game.meta.map_family),
        pre,
        game.meta.enemy_random,
        game.steps.iter().map(|s| (&s.raw_frame, &s.action_label, s.memory_action.as_deref())),
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
    /// 041-R28 (`--memory-log <dir>`, with or without `relabel`, 051-R13 ④): folder of the model
    /// decision logs, `<replay file stem>.jsonl` per replay, or the game's 051-R2 act logs
    /// `<stem>-p<k>.jsonl` (`act_log::read_memory_logs`; 040-R22 `memory_action`). `None` = every
    /// step `None`.
    pub memory_log: Option<crate::act_log::MemoryLogDir>,
    /// 022-R2 / R12 / AC3 (`--games <games.jsonl>`, [`check_games_arg`]): the runner's lobby races
    /// of the games; only players[0] is a view (`lobby::players0_view`). `None` = every Terran view.
    pub games: Option<crate::lobby::GamesLog>,
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
    let (bot, any_length) = filter_rules(&opt.src, opt.relabel.is_some());
    let views = match filter(&rep, &family, bot, any_length) {
        Ok(v) => v,
        Err(reason) => return vec![rec(&sha, &rep.header.map_name, &family, None, Outcome::Filtered { reason })],
    };
    // Resume: only perspectives without their own record (an earlier run may have recorded one
    // perspective and lost the other with an unflushed shard).
    // 022-R2 / R12 / AC3 (`--games`): only players[0] (the 022 teacher, or the model of a
    // `--relabel terran_tp` build) is a view, and the opponent's lobby race (games.jsonl) decides.
    // A 022 teacher game in a botgen build without it is refused: its TvT opponent view is no
    // teacher data and its lobby races are unknown.
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
    // 041-R28: the model's decision log of this game, named like the replay file (or the game's
    // 051-R2 act logs).
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

    /// 040-R21: `--until-frame 11520` keeps the steps at frames <= 11,520 (8:00 included).
    #[test]
    fn until_frame_cuts_later_steps_040_r21() {
        let mut v = vec![1u32, 11_519, 11_520, 11_521, 20_000];
        cut_after(&mut v, None, |f| *f);
        assert_eq!(v.len(), 5);
        cut_after(&mut v, Some(11_520), |f| *f);
        assert_eq!(v, vec![1, 11_519, 11_520]);
    }

    /// 040-AC9 / 041-R28 ④ ⑤: a complete log whose unit check passes fills every step (an empty
    /// list included); no log, an incomplete log or more than 1% unit mismatches leave every step
    /// `None` (never mixed within a game), and the record counts the check.
    #[test]
    fn memory_action_is_all_or_none_per_game_041_r28() {
        use crate::relabel::{DecisionLogRead, ModelLog, bind_log};
        use crate::shard::MemoryAction;
        let steps = || {
            let mut s = crate::shard::tests::game(1).steps;
            for (st, f) in s.iter_mut().zip([1u32, 11, 19]) {
                st.raw_frame.frame = f;
            }
            s
        };
        let m = |action, units: Vec<u32>| MemoryAction { action, delay_bucket: 0, units };
        let complete = |decisions, player| DecisionLogRead::Complete(ModelLog { decisions, end_frame: Some(500), player });
        // Player 0's view of a replay of 502 frames.
        let run = |read: &DecisionLogRead, own: &HashMap<u32, std::collections::HashSet<u32>>| {
            let mut s = steps();
            let r = attach_memory(&mut s, bind_log(read, 0, 502), 100, own);
            (r, s.into_iter().map(|x| x.memory_action).collect::<Vec<_>>())
        };
        let own: HashMap<u32, std::collections::HashSet<u32>> = HashMap::from([(7, [5].into()), (15, [5].into())]);
        // Live frames 9 and 17 (playback 7 and 15); the steps' live frames are 1, 13 and 21.
        let (r, got) = run(&complete(vec![(9, m(4, vec![5])), (17, m(0, vec![]))], 0), &own);
        assert_eq!((r.status, r.decisions, r.checked, r.mismatched), ("ok", 2, 1, 0));
        assert_eq!(got, vec![Some(vec![]), Some(vec![m(4, vec![5])]), Some(vec![m(0, vec![])])]);
        // Not all or more than 1% of the selections are own units: every step None.
        let (r, got) = run(&complete(vec![(9, m(4, vec![5])), (17, m(4, vec![6]))], 0), &own);
        assert_eq!((r.status, r.checked, r.mismatched), ("unit_mismatch", 2, 1));
        assert!(got.iter().all(Option::is_none));
        // No log, an incomplete log, the other player's log, another game's log: every step None.
        let other_game = DecisionLogRead::Complete(ModelLog { decisions: vec![(9, m(4, vec![5]))], end_frame: Some(9_000), player: 0 });
        for (read, status) in [
            (DecisionLogRead::Missing, "missing"),
            (DecisionLogRead::Incomplete("no end line".into()), "incomplete"),
            (complete(vec![(9, m(4, vec![5]))], 1), "other_player"),
            (other_game, "incomplete"),
        ] {
            let (x, got) = run(&read, &own);
            assert_eq!(x.status, status);
            assert!(got.iter().all(Option::is_none));
        }
        let mut s = steps();
        let r = attach_memory(&mut s, bind_log(&complete(vec![(9, m(4, vec![5])), (17, m(4, vec![6]))], 0), 0, 502), 100, &own);
        // Replays with the same stem cannot share one --memory-log folder.
        assert!(unique_stems(&[PathBuf::from("a/replays/g1.rep"), PathBuf::from("a/replays/g2.rep")]).is_ok());
        assert!(unique_stems(&[PathBuf::from("a/replays/g1.rep"), PathBuf::from("b/replays/g1.rep")]).is_err());
        // The record's `memory` object.
        let text = serde_json::to_string(&Outcome::Written { steps: 3, coverage: Default::default(), result: 0, result_source: "none".into(), repeats_dropped: Some(0), memory: Some(r) }).unwrap();
        assert!(text.contains(r#""memory":{"status":"unit_mismatch","decisions":2,"checked":2,"mismatched":1}"#), "{text}");
        assert_eq!(memory_log_path(Path::new("m"), Path::new("x/replays/g12.rep")), Path::new("m").join("g12.jsonl"));
    }

    /// 041-R28 (051-R13 ④): `--memory-log` without relabelling goes with `--src branch` only, and a
    /// `branch` build needs an explicit `--out` outside the BC shards root (its shards are not BC
    /// data); `--relabel` still goes with `dagger` only (with or without `--memory-log`), and the
    /// memory folder must exist.
    #[test]
    fn memory_log_without_relabel_only_for_branch_051_r13_041_r28() {
        let dir = std::env::temp_dir();
        let root = dir.join("shared").join("datasets").join("replays").join("shards");
        let eval = dir.join("shared").join("eval").join("branch-20261113").join("shards");
        let ok = |src, relabel, mem: Option<&Path>, out: Option<&Path>| check_build_args(src, relabel, mem, out, Some(&root)).is_ok();
        assert!(ok("branch", false, Some(&dir), Some(&eval)), "a branch trunk build");
        assert!(ok("branch", false, Some(&dir), Some(&eval)) && check_build_args("branch", false, Some(&dir), Some(&eval), None).is_ok());
        assert!(ok("dagger", true, Some(&dir), None) && ok("dagger", true, None, None) && ok("botgen", false, None, None));
        // No-relabel --memory-log under a BC source tag: refused (label leak, unknown chain starts).
        for src in ["botgen", "bot", "scr_ladder", "human"] {
            assert!(!ok(src, false, Some(&dir), Some(&eval)), "{src}");
        }
        // A branch build without --out, or with --out inside the BC shards root (any depth, any case on Windows).
        assert!(!ok("branch", false, Some(&dir), None));
        assert!(!ok("branch", false, Some(&dir), Some(&root.join("branch"))));
        assert!(!ok("branch", false, None, Some(&root)));
        if cfg!(windows) {
            let upper = PathBuf::from(root.to_string_lossy().to_uppercase()).join("x");
            assert!(!ok("branch", false, Some(&dir), Some(&upper)));
        }
        // A branch build needs --time-limit and a records file named `branch*.jsonl` (resume scan).
        let rec = |n: &str| dir.join(n);
        assert!(check_isolated_build("branch", &rec("branch_build.jsonl"), Some(11_568)).is_ok());
        assert!(check_isolated_build("branch", &rec("branch_build.jsonl"), None).is_err());
        assert!(check_isolated_build("branch", &rec("build_records.jsonl"), Some(11_568)).is_err());
        assert!(check_isolated_build("botgen", &rec("x.jsonl"), None).is_ok(), "other sources unchanged");
        // Not in the 051-R12 output folder (its branches.jsonl would be read as build records).
        let o = dir.join(format!("u6-branch-records-{}", std::process::id()));
        std::fs::create_dir_all(o.join("build")).unwrap();
        std::fs::write(o.join("branches.jsonl"), "").unwrap();
        assert!(check_isolated_build("branch", &o.join("branch_build.jsonl"), Some(11_568)).is_err());
        assert!(check_isolated_build("branch", &o.join("build").join("branch_build.jsonl"), Some(11_568)).is_ok());
        std::fs::remove_dir_all(&o).ok();
        assert!(!ok("dagger", false, Some(&dir), None), "dagger needs --relabel (041-R22)");
        assert!(!ok("botgen", true, None, None), "--relabel only with dagger (041-R22)");
        assert!(!ok("branch", false, Some(Path::new("no/such/memory")), Some(&eval)));
        assert!(!path_within(&eval, &root) && path_within(&root, &root));
    }

    /// 041-R28 ② ④ ⑤ and decision-time steps (user decision 2026-11-13, 051-R13 ④) on a build without
    /// relabelling: the replay's own command labels (any frames) get one observation / memory only step
    /// (`unknown`, BC weight 0) at each logged decision's playback frame L - 2 (NoOp decisions too, none
    /// where a label already is), so every decision has its exact row; the trunk's 051-R2 act log fills
    /// each step with the decisions of live frames [live_{i-1}, live_i), exactly as the equivalent
    /// 041-R28 decision log would; the scripted session's act log beside it is not used.
    #[test]
    fn act_log_memory_and_decision_steps_051_r13_041_r28() {
        use crate::act_log::tests::{log_text, obs_line};
        use crate::act_log::{parse_memory_log, read_memory_logs};
        use crate::label::{Label, UNKNOWN_ACTION};
        use crate::relabel::bind_logs;
        let dir = std::env::temp_dir().join(format!("uranium-act-mem-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // Live decisions at 1, 9, 17, 25 (forced prefix, then free; 0 = NoOp); the replay is 500 frames.
        let ours = log_text(
            vec![obs_line(0, 1, Some((0, 0, vec![])), true), obs_line(1, 9, Some((4, 2, vec![5])), true), obs_line(2, 17, Some((7, 1, vec![5, 6])), false), obs_line(3, 25, Some((0, 3, vec![])), false)],
            498,
            0,
            "model:x",
        );
        std::fs::write(dir.join("g3-p0.jsonl"), &ours).unwrap();
        std::fs::write(dir.join("g3-p1.jsonl"), log_text(vec![obs_line(0, 1, None, false)], 498, 1, "t0@3:zerg_rush")).unwrap();
        let logs = read_memory_logs(&dir, Path::new("replays/g3.rep"));
        assert_eq!(logs.len(), 1);
        let own: HashMap<u32, std::collections::HashSet<u32>> = HashMap::from([(7, [5].into()), (15, [5, 6].into())]);
        let label = |frame, action, delay| Label { frame, action, delay, queued: false, selected: vec![], target_unit: None, target_tile: None };
        // Command label steps at playback frames 1, 11, 20 (live 1, 13, 22).
        let commands = vec![label(1, 0, 10), label(11, 4, 9), label(20, 7, 480)];
        let run = |logs: &[crate::relabel::DecisionLogRead], relabelled: bool| {
            let bound = bind_logs(logs, 0, 500);
            let mut labels = commands.clone();
            let added = decision_steps(relabelled, Some(&bound), &own, 500, 500, 500, &mut labels);
            let tpl = crate::shard::tests::game(1).steps[0].clone();
            let mut s: Vec<Step> = labels.iter().map(|l| Step { raw_frame: uranium_proto::RawFrame { frame: l.frame, ..tpl.raw_frame.clone() }, action_label: l.clone(), ..tpl.clone() }).collect();
            let r = MemoryCheck { decision_steps: added, ..attach_memory(&mut s, bound, 500, &own) };
            (r, labels, s.into_iter().map(|x| x.memory_action.map(|v| v.into_iter().map(|a| a.action).collect::<Vec<_>>())).collect::<Vec<_>>())
        };
        let (r, labels, got) = run(&logs, false);
        assert_eq!((r.status, r.decisions, r.checked, r.mismatched, r.decision_steps), ("ok", 4, 2, 0, Some(3)));
        // Decision rows at playback 7, 15, 23 (live 9, 17, 25); live 1 is the existing game-start step.
        assert_eq!(labels.iter().map(|l| (l.frame, l.is_unknown(), l.delay)).collect::<Vec<_>>(), vec![(1, false, 10), (7, true, 4), (11, false, 9), (15, true, 5), (20, false, 480), (23, true, 477)]);
        assert!(labels.iter().filter(|l| l.is_unknown()).all(|l| l.action == UNKNOWN_ACTION && l.selected.is_empty()));
        // Memory before each observation: the decisions made on earlier live observations.
        assert_eq!(got, vec![Some(vec![]), Some(vec![0]), Some(vec![4]), Some(vec![]), Some(vec![7]), Some(vec![])]);
        // The same decisions as a 041-R28 decision log give the same steps and memory.
        let dl = parse_memory_log(
            &[
                r#"{"frame":1,"action":0,"delay_bucket":0,"units":[]}"#,
                r#"{"frame":9,"action":4,"delay_bucket":2,"units":[5]}"#,
                r#"{"frame":17,"action":7,"delay_bucket":1,"units":[5,6]}"#,
                r#"{"frame":25,"action":0,"delay_bucket":3,"units":[]}"#,
                r#"{"decisions":4,"end_frame":498,"player":0}"#,
            ]
            .join("\n"),
        )
        .unwrap();
        assert_eq!(run(&[dl], false), (r, labels, got));
        // No act log of the view's player: no steps added, every step None.
        let (r, labels, got) = run(&[], false);
        assert_eq!((r.status, r.decision_steps), ("missing", None));
        assert_eq!(labels, commands);
        assert!(got.iter().all(Option::is_none));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// 041-R28 decision-time steps are for builds without relabelling whose log is used: a relabelled
    /// (DAgger) build's labels, and so its shard steps, stay byte for byte what they were; so do the
    /// labels of a game whose log fails the ⑤ unit check. `--until-frame` bounds the added steps.
    #[test]
    fn decision_steps_leave_relabel_builds_unchanged_041_r28() {
        use crate::label::Label;
        use crate::relabel::{DecisionLogRead, ModelLog, bind_log};
        use crate::shard::MemoryAction;
        let label = |frame, action, delay| Label { frame, action, delay, queued: false, selected: vec![5], target_unit: None, target_tile: Some((3, 4)) };
        let teacher = vec![label(1, 0, 8), label(7, 12, 8), label(15, 4, 8), label(23, 0, 8)];
        let m = |action, units: Vec<u32>| MemoryAction { action, delay_bucket: 1, units };
        let read = DecisionLogRead::Complete(ModelLog { decisions: vec![(5, m(4, vec![5])), (13, m(0, vec![])), (30, m(7, vec![5]))], end_frame: Some(500), player: 0 });
        let bound = bind_log(&read, 0, 500);
        let own: HashMap<u32, std::collections::HashSet<u32>> = HashMap::from([(3, [5].into()), (28, [5].into())]);
        let bytes = |l: &Vec<Label>| postcard::to_allocvec(l).unwrap();
        let mut relabelled = teacher.clone();
        assert_eq!(decision_steps(true, Some(&bound), &own, 500, 500, 500, &mut relabelled), None);
        assert_eq!(bytes(&relabelled), bytes(&teacher), "a relabelled build gets no decision steps");
        // Unit check failed (selections not own units): the log is not used, nothing added.
        let mut failed = teacher.clone();
        assert_eq!(decision_steps(false, Some(&bound), &HashMap::new(), 500, 500, 500, &mut failed), None);
        assert_eq!(bytes(&failed), bytes(&teacher));
        // Without relabelling: playback 3 and 28 are added, 11 too (NoOp); --until-frame 20 keeps 3 and 11.
        let mut cut = teacher.clone();
        assert_eq!(decision_steps(false, Some(&bound), &own, 500, 20, 500, &mut cut), Some(2));
        assert_eq!(cut.iter().map(|l| l.frame).collect::<Vec<_>>(), vec![1, 3, 7, 11, 15, 23]);
        // The existing labels are unchanged (their delays included).
        assert_eq!(cut.iter().filter(|l| !l.is_unknown()).cloned().collect::<Vec<_>>(), teacher);
        assert_eq!(decision_steps(false, None, &own, 500, 500, 500, &mut teacher.clone()), None);
    }

    /// 044-AC1 (044-R2, 040-R24): a distill view's steps are the game-start step and one per logged
    /// decision (NoOp ones included) at playback frame L - 2 within `--until-frame`, never the
    /// replay's command frames; every label is an `unknown` slot (delay to the next step, the game end
    /// after the last); step i's `memory_action` holds the decisions of [obs_{i-1}, obs_i) and never
    /// its own decision, which is in step i + 1's; the shard keeps all of it (schema 7).
    #[test]
    fn distill_steps_are_the_logged_decisions_044_r2_ac1() {
        use crate::label::{Label, UNKNOWN_ACTION};
        use crate::relabel::{DecisionLogRead, ModelLog, bind_log, playback_frame};
        use crate::shard::MemoryAction;
        let m = |action, units: Vec<u32>| MemoryAction { action, delay_bucket: 2, units };
        // Live decisions 1 (NoOp), 9, 12 (NoOp), 20 and 600; the replay is 700 frames, --until-frame 300.
        let decisions = vec![(1, m(0, vec![])), (9, m(4, vec![5])), (12, m(0, vec![])), (20, m(7, vec![5, 6])), (600, m(4, vec![5]))];
        let read = DecisionLogRead::Complete(ModelLog { decisions: decisions.clone(), end_frame: Some(698), player: 0 });
        let bound = bind_log(&read, 0, 700);
        let own: HashMap<u32, std::collections::HashSet<u32>> = HashMap::from([(7, [5].into()), (18, [5, 6].into())]);
        let (last, limit) = (332, 300); // as build_view: last = until + 32, limit = until
        let (labels, n) = distill_steps(Some(&bound), &own, last, limit, 700).unwrap();
        // The game-start step is the live-1 decision's step: 4 decisions in range, 4 steps.
        assert_eq!(labels.iter().map(|l| l.frame).collect::<Vec<_>>(), vec![1, 7, 10, 18]);
        assert_eq!(n, 4);
        assert!(labels.iter().all(|l| l.action == UNKNOWN_ACTION && l.is_unknown() && l.selected.is_empty() && l.target_unit.is_none() && l.target_tile.is_none() && !l.queued));
        assert_eq!(labels.iter().map(|l| l.delay).collect::<Vec<_>>(), vec![6, 3, 8, 682]);
        // The build's label swap (`step_labels`): the replay's command labels (frames 1, 3, 11, 25)
        // are replaced by the slots, so no step sits at a command frame; a branch build of the same
        // view keeps them and adds the decision-time steps.
        let cmd = |frame, action| Label { frame, action, delay: 2, queued: false, selected: vec![5], target_unit: None, target_tile: None };
        let commands = vec![cmd(1, 0), cmd(3, 4), cmd(11, 7), cmd(25, 4)];
        let mut swapped = commands.clone();
        assert_eq!(step_labels(true, false, Some(&bound), &own, last, limit, 700, &mut swapped), Ok(Some(4)));
        assert_eq!(swapped, labels);
        assert!(swapped.iter().all(|l| ![3, 11, 25].contains(&l.frame)));
        let mut branch = commands.clone();
        assert_eq!(step_labels(false, false, Some(&bound), &own, last, limit, 700, &mut branch), Ok(Some(3)));
        assert_eq!(branch.iter().map(|l| (l.frame, l.is_unknown())).collect::<Vec<_>>(), vec![(1, false), (3, false), (7, true), (10, true), (11, false), (18, true), (25, false)]);
        let mut relabelled = commands.clone();
        assert_eq!(step_labels(false, true, Some(&bound), &own, last, limit, 700, &mut relabelled), Ok(None));
        assert_eq!(relabelled, commands);
        let mut unbound = commands.clone();
        assert!(step_labels(true, false, None, &own, last, limit, 700, &mut unbound).unwrap_err().starts_with("decision_log missing"));
        // Memory (041-R28 ②): the decisions before each step's live frame and at or after the previous one's.
        let tpl = crate::shard::tests::game(1).steps[0].clone();
        let mut steps: Vec<Step> = labels.iter().map(|l| Step { raw_frame: uranium_proto::RawFrame { frame: l.frame, ..tpl.raw_frame.clone() }, action_label: l.clone(), memory_action: None, ..tpl.clone() }).collect();
        let r = attach_memory(&mut steps, bound.clone(), last, &own);
        assert_eq!((r.status, r.decisions, r.checked, r.mismatched), ("ok", 5, 2, 0));
        let mem: Vec<Vec<MemoryAction>> = steps.iter().map(|s| s.memory_action.clone().expect("every step has memory")).collect();
        assert_eq!(mem, vec![vec![], vec![m(0, vec![])], vec![m(4, vec![5])], vec![m(0, vec![])]]);
        // A step's memory is exactly the decisions with a live frame in [live_{i-1}, live_i) ...
        let live: Vec<u32> = steps.iter().map(|s| crate::relabel::live_of_playback(s.raw_frame.frame)).collect();
        for i in 1..steps.len() {
            let want: Vec<MemoryAction> = decisions.iter().filter(|(l, _)| *l >= live[i - 1] && *l < live[i]).map(|(_, d)| d.clone()).collect();
            assert_eq!(mem[i], want, "step {i}");
        }
        // ... so the decision made on a step's own observation is outside its interval and inside
        // the next step's: never an input of the step it is made on.
        for (i, s) in steps.iter().enumerate() {
            for (l, _) in decisions.iter().filter(|(l, _)| playback_frame(*l) == s.raw_frame.frame) {
                assert!(*l >= live[i], "step {i} would hold its own decision");
                assert!(live.get(i + 1).is_none_or(|n| l < n), "step {i}'s decision belongs to step {}", i + 1);
            }
        }
        // Schema 7 keeps the slots and the memory.
        let game = ShardGame { meta: GameMeta { src: DISTILL_SRC.into(), stage: "B".into(), ..crate::shard::tests::game(1).meta }, game_info: crate::shard::tests::game(1).game_info, steps: steps.clone() };
        let shard = crate::shard::Shard { header: Default::default(), games: vec![game.clone()] };
        let back = crate::shard::decode(&postcard::to_stdvec(&shard).unwrap()).unwrap();
        assert_eq!(back.header.schema_version, crate::shard::SCHEMA_VERSION);
        assert_eq!(back.games, vec![game]);
        // Decisions on live frames 1..=3 all show playback frame 1: one step, their memory after it.
        let early = vec![(1, m(0, vec![])), (2, m(4, vec![5])), (3, m(0, vec![])), (11, m(0, vec![]))];
        let (labels, n) = crate::relabel::distill_labels(&early, 300, 700);
        assert_eq!((labels.iter().map(|l| (l.frame, l.delay)).collect::<Vec<_>>(), n), (vec![(1, 8), (9, 691)], 2));
        let mem = crate::relabel::step_memory(&early, &[1, 9]);
        assert_eq!(mem, vec![vec![], vec![m(0, vec![]), m(4, vec![5]), m(0, vec![])]]);
        // A log without a decision on the game-start observation: the game-start step is added (1 + decisions).
        let (labels, n) = crate::relabel::distill_labels(&[(9, m(0, vec![])), (17, m(0, vec![]))], 300, 700);
        assert_eq!((labels.iter().map(|l| l.frame).collect::<Vec<_>>(), n), (vec![1, 7, 15], 2));
    }

    /// 044-R2 / 041-R28 ④ ⑤: a distill view without its player's complete log (no log, an incomplete
    /// one, only the other player's: the T0 opponent's view of a TvT game) or with more than 1% unit
    /// mismatches is `filtered` with the check's status in the reason; none of them gets steps.
    #[test]
    fn distill_views_failing_the_log_checks_are_filtered_044_r2() {
        use crate::relabel::{DecisionLogRead, ModelLog, bind_log};
        use crate::shard::MemoryAction;
        let m = |units: Vec<u32>| MemoryAction { action: 4, delay_bucket: 1, units };
        let own: HashMap<u32, std::collections::HashSet<u32>> = HashMap::from([(7, [5].into()), (15, [5].into())]);
        let complete = |decisions, player| DecisionLogRead::Complete(ModelLog { decisions, end_frame: Some(500), player });
        let reason = |b: Option<&Result<&ModelLog, MemoryCheck>>| match distill_steps(b, &own, 500, 500, 500) {
            Err(reason) => reason,
            other => panic!("{other:?}"),
        };
        assert!(reason(None).starts_with("decision_log missing (041-R28, 044-R2)"));
        for (read, status) in [
            (DecisionLogRead::Missing, "missing"),
            (DecisionLogRead::Incomplete("no end line".into()), "incomplete"),
            (complete(vec![(9, m(vec![5]))], 1), "other_player"),
            (DecisionLogRead::Complete(ModelLog { decisions: vec![(9, m(vec![5]))], end_frame: Some(9_000), player: 0 }), "incomplete"),
        ] {
            let r = reason(Some(&bind_log(&read, 0, 502)));
            assert!(r.starts_with(&format!("{DISTILL_FILTER_PREFIX} {status} ")), "{r}");
        }
        let bad = complete(vec![(9, m(vec![5])), (17, m(vec![6]))], 0);
        let r = reason(Some(&bind_log(&bad, 0, 502)));
        assert_eq!(r, "decision_log unit_mismatch (041-R28, 044-R2): 2 decisions, 2 checked, 1 mismatched");
        let good = complete(vec![(9, m(vec![5])), (17, m(vec![5]))], 0);
        assert!(distill_steps(Some(&bind_log(&good, 0, 502)), &own, 500, 500, 500).is_ok());
    }

    /// 044-R2 / 040-R24: `--src distill` needs `--memory-log` and an explicit `--out` outside the BC
    /// shards root, `--time-limit` and a `distill*.jsonl` records file (as `branch`, 041-R28); it is bot
    /// data of any length (stage B, no APM or 5-minute bar, 040 quality table). The other sources'
    /// rules are unchanged.
    #[test]
    fn distill_build_args_and_filter_rules_044_r2_040_r24() {
        let dir = std::env::temp_dir();
        let root = dir.join("shared").join("datasets").join("replays").join("shards");
        let pre = dir.join("shared").join("datasets").join("replays").join("distill_pre").join("d1");
        let ok = |src, relabel, mem: Option<&Path>, out: Option<&Path>| check_build_args(src, relabel, mem, out, Some(&root)).is_ok();
        assert!(ok("distill", false, Some(&dir), Some(&pre)));
        assert!(!ok("distill", false, None, Some(&pre)), "the steps are the logged decisions");
        assert!(!ok("distill", false, Some(&dir), None), "explicit --out");
        assert!(!ok("distill", false, Some(&dir), Some(&root.join("distill_m2").join("d1"))), "not inside the BC shards");
        assert!(!ok("distill", false, Some(&dir), Some(&root)));
        assert!(!ok("distill", true, Some(&dir), Some(&pre)), "--relabel only with dagger");
        assert!(!ok("distill", false, Some(Path::new("no/such/memory")), Some(&pre)));
        let rec = |n: &str| dir.join(n);
        assert!(check_isolated_build("distill", &rec("distill_d1.jsonl"), Some(14_400)).is_ok());
        assert!(check_isolated_build("distill", &rec("distill_d1.jsonl"), None).is_err());
        assert!(check_isolated_build("distill", &rec("branch_d1.jsonl"), Some(14_400)).is_err());
        assert!(check_isolated_build("branch", &rec("distill_d1.jsonl"), Some(14_400)).is_err());
        // Filter flags: distill is bot data of any length; the earlier sources keep theirs.
        assert_eq!(filter_rules("distill", false), (true, true));
        for (src, relabel) in [("stardata", false), ("scr_public", false), ("community", false), ("personal", false), ("user", false), ("scr_ladder", false), ("bot", false), ("botgen", false), ("dagger", true), ("branch", false)] {
            let before = (matches!(src, "bot" | "botgen" | "dagger" | BRANCH_SRC), relabel || src == BRANCH_SRC);
            assert_eq!(filter_rules(src, relabel), before, "{src}");
        }
    }

    /// 044-R2 (ignored, a built shard): every game of a `--src distill` shard
    /// (`URANIUM_TEST_DISTILL_SHARD`, built with `--memory-log URANIUM_TEST_DISTILL_MEMORY` from
    /// replays named `g<id>.rep`, given as `URANIUM_TEST_DISTILL_REPLAYS`) starts at the game-start
    /// observation, has an `unknown` label slot on every step, a step at each logged decision's
    /// playback frame and nowhere else, and each step's memory is the log's [live_{i-1}, live_i).
    #[test]
    #[ignore = "needs URANIUM_TEST_DISTILL_SHARD, URANIUM_TEST_DISTILL_MEMORY and URANIUM_TEST_DISTILL_REPLAYS"]
    fn distill_shard_matches_its_logs_044_r2() {
        use crate::relabel::{bind_logs, playback_frame, step_memory};
        let var = |k: &str| PathBuf::from(std::env::var(k).unwrap_or_else(|_| panic!("{k}")));
        let shard = crate::shard::read(&var("URANIUM_TEST_DISTILL_SHARD")).unwrap();
        let mem = crate::act_log::MemoryLogDir::open(&var("URANIUM_TEST_DISTILL_MEMORY")).unwrap();
        let mut reps = Vec::new();
        for e in std::fs::read_dir(var("URANIUM_TEST_DISTILL_REPLAYS")).unwrap().flatten() {
            let b = std::fs::read(e.path()).unwrap();
            reps.push((hex(&Sha256::digest(&b)), e.path()));
        }
        assert!(!shard.games.is_empty());
        for g in &shard.games {
            assert_eq!((g.meta.src.as_str(), g.meta.stage.as_str()), (DISTILL_SRC, "B"));
            let file = &reps.iter().find(|(s, _)| *s == g.meta.replay_sha256).expect("the game's replay").1;
            let rep = crate::parse(&std::fs::read(file).unwrap()).unwrap();
            let logs = mem.logs(file);
            let log = bind_logs(&logs, g.meta.perspective_slot, rep.header.frames).expect("a bound log");
            let frames: Vec<u32> = g.steps.iter().map(|s| s.raw_frame.frame).collect();
            assert_eq!(frames[0], crate::label::GAME_START_FRAME);
            assert!(g.steps.iter().all(|s| s.action_label.is_unknown() && s.action_label.frame == s.raw_frame.frame));
            let last = *frames.last().unwrap();
            let mut want: Vec<u32> = log.decisions.iter().map(|(l, _)| playback_frame(*l)).filter(|p| *p <= last).collect();
            want.insert(0, crate::label::GAME_START_FRAME);
            want.dedup();
            assert_eq!(frames, want, "a step at each decision frame and nowhere else");
            let got: Vec<Vec<crate::shard::MemoryAction>> = g.steps.iter().map(|s| s.memory_action.clone().expect("memory on every step")).collect();
            assert_eq!(got, step_memory(&log.decisions, &frames));
            eprintln!("{}: {} decisions, {} steps", &g.meta.replay_sha256[..12], log.decisions.len(), frames.len());
        }
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
        Record { index: n as usize, replay_sha256: format!("{n:064x}"), src: "t".into(), map_family: "other".into(), map_name: String::new(), map_hash: String::new(), recorded_winner: None, view: None, labeler_version: crate::label::LABELER_VERSION, until_frame: None, time_limit: None, relabel: None, outcome: Outcome::Written { steps: 3, coverage, result: 0, result_source: "none".into(), repeats_dropped: None, memory: None } }
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
        let opt = BuildOptions { src: "t".into(), shards_dir: dir.join("s"), staging_dir: dir.join("l"), recorded_winners: HashMap::new(), jobs: 1, shard_target_bytes: 1, budget_bytes: 1 << 30, first_shard_index: 0, skip_sha256: Default::default(), done: ResumeDone::default(), duplicates: part1, until_frame: None, time_limit: None, relabel: None, memory_log: None, games: None };
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

    /// A two-player 1.16.1 replay file `<dir>/g<id>.rep` (no OpenBW needed to get to the views):
    /// `players` = (command id, header race, hostd name), each training once.
    fn tp_replay(dir: &Path, id: u64, players: [(u8, u8, &str); 2]) -> PathBuf {
        let mut h = vec![0u8; crate::container::HEADER_SIZE];
        h[1..5].copy_from_slice(&8_000u32.to_le_bytes()); // over 5 minutes (botgen keeps the 040 length bar)
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

    /// 022-R2 / R12 / AC3, 040 (2026-11-14): with `--games` (the job's games.jsonl) only players[0]
    /// (hostd name `p0-...`, whatever its command id) is a view, so a TvT 022 botgen game writes the
    /// teacher's view only; the opponent's LOBBY race comes from games.jsonl: a lobby Random that
    /// played Zerg is refused as Random (022-R2), a lobby Zerg as 022-AC3 (`--relabel terran_tp`) or
    /// 022-R12 (botgen), a game without a line or whose lobby race differs from the header race is
    /// refused, Terran and Protoss go on to the replay (no OpenBW here: an error, never a lobby
    /// refusal). A 022 teacher game in a botgen build without `--games` is refused. Builds without
    /// `--games` (the 021-R8 relabel, other botgen games) keep every Terran view.
    #[test]
    fn players0_view_and_lobby_races_from_games_jsonl_022_r12_ac3() {
        use crate::header::{RACE_PROTOSS as P, RACE_TERRAN as T, RACE_ZERG as Z};
        use uranium_scripted::T0Build;
        let dir = std::env::temp_dir().join(format!("uranium-tp-lobby-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let tp = "p0-T0-terran-terran_tp";
        // players[0] has command id 1 and the opponent id 0: the view follows the name.
        let game = |id, enemy_header, enemy_name: &str| tp_replay(&dir, id, [(0, enemy_header, enemy_name), (1, T, tp)]);
        let lines = [(1, "Terran"), (2, "Random"), (3, "Zerg"), (4, "Protoss"), (6, "Protoss")];
        let text: String = lines.iter().map(|(id, r)| format!("{{\"game_id\":{id},\"matchup\":0,\"races\":[\"Terran\",\"{r}\"],\"outcome\":\"finished\"}}\n")).collect();
        let games = crate::lobby::GamesLog::parse(&text).unwrap();
        let families = MapFamilies::load().unwrap();
        let excludes = MapExcludes::load(&families).unwrap();
        let labeler = Labeler::new().unwrap();
        let cfg = PlayConfig { launcher: dir.join("none"), bridge_lib: dir.join("none"), data_dir: dir.join("none"), work_dir: dir.join("w") };
        let run = |path: &Path, src: &str, relabel: Option<T0Build>, games: Option<crate::lobby::GamesLog>| {
            let opt = BuildOptions { src: src.into(), shards_dir: dir.join("s"), staging_dir: dir.join("l"), recorded_winners: HashMap::new(), jobs: 1, shard_target_bytes: 1, budget_bytes: 1 << 30, first_shard_index: 0, skip_sha256: Default::default(), done: ResumeDone::default(), duplicates: Duplicates::find(&[path.to_path_buf()], &[0], 1), until_frame: Some(14_400), time_limit: Some(14_400), relabel, memory_log: None, games };
            process(0, path, &opt, &cfg, &families, &excludes, &labeler, &Mutex::new(0)).into_iter().map(|(r, g)| {
                assert!(g.is_none());
                (r.view.map(|v| v.perspective), r.outcome)
            }).collect::<Vec<_>>()
        };
        let reason = |o: &Outcome| match o {
            Outcome::Filtered { reason } => reason.clone(),
            _ => String::new(),
        };
        let tvt = game(1, T, "p1-T0-terran");
        // 022 botgen, TvT: the teacher's view (command id 1) only.
        let got = run(&tvt, "botgen", None, Some(games.clone()));
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].0, Some(1), "{got:?}");
        assert!(reason(&got[0].1).is_empty(), "{:?}", got[0].1);
        // Without --games: refused as a whole (no view written).
        let got = run(&tvt, "botgen", None, None);
        assert!(got.len() == 1 && got[0].0.is_none() && reason(&got[0].1).contains("without --games"), "{got:?}");
        // `--relabel terran_tp` on a TvT self game: the model's (players[0]) view only.
        let got = run(&tvt, "dagger", Some(T0Build::TerranTp), Some(games.clone()));
        assert_eq!(got.iter().map(|g| g.0).collect::<Vec<_>>(), vec![Some(1)]);
        // A lobby Random that played Zerg: refused as Random (022-R2), not as a lobby Zerg.
        let random = game(2, Z, "p1-ualbertabot");
        for (src, relabel) in [("dagger", Some(T0Build::TerranTp)), ("botgen", None)] {
            let got = run(&random, src, relabel, Some(games.clone()));
            assert!(got.len() == 1 && got[0].0 == Some(1) && reason(&got[0].1).starts_with("lobby Random") && reason(&got[0].1).contains("022-R2"), "{got:?}");
        }
        // A lobby Zerg: 022-AC3 for the relabel, 022-R12 for botgen.
        let zerg = game(3, Z, "p1-T0-zerg");
        assert!(reason(&run(&zerg, "dagger", Some(T0Build::TerranTp), Some(games.clone()))[0].1).contains("022-AC3"));
        assert!(reason(&run(&zerg, "botgen", None, Some(games.clone()))[0].1).contains("022-R12"));
        // A lobby Protoss goes on (one view).
        let got = run(&game(4, P, "p1-stardust"), "dagger", Some(T0Build::TerranTp), Some(games.clone()));
        assert!(got.len() == 1 && got[0].0 == Some(1) && reason(&got[0].1).is_empty(), "{got:?}");
        // No games.jsonl line: refused, never guessed.
        let got = run(&game(5, P, "p1-stardust"), "dagger", Some(T0Build::TerranTp), Some(games.clone()));
        assert!(got.len() == 1 && got[0].0.is_none() && reason(&got[0].1).contains("no games.jsonl line"), "{got:?}");
        // Lobby Protoss but the header says Terran was played: refused.
        let got = run(&game(6, T, "p1-T0-terran"), "botgen", None, Some(games.clone()));
        assert!(reason(&got[0].1).contains("header race Terran"), "{got:?}");
        // Builds without --games are as before: every Terran view (the 021-R8 relabel; a botgen game
        // without the 022 teacher).
        assert_eq!(run(&tvt, "dagger", Some(T0Build::TerranReact), None).len(), 2);
        let react = tp_replay(&dir, 7, [(0, T, "p1-T0-terran"), (1, T, "p0-T0-terran-terran_reac")]);
        assert!(run(&react, "botgen", None, None).iter().all(|g| g.0.is_some() && reason(&g.1).is_empty()));
        assert!(run(&zerg, "dagger", Some(T0Build::TerranReact), None).iter().all(|g| reason(&g.1).is_empty()));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// 022-R2 / AC3: `--relabel terran_tp` needs `--games`; `--games` goes with it or with a
    /// `--src botgen` build only.
    #[test]
    fn games_arg_rules_022_r2() {
        use uranium_scripted::T0Build::{TerranReact, TerranTp};
        assert!(check_games_arg("dagger", Some(TerranTp), false).is_err());
        assert!(check_games_arg("dagger", Some(TerranTp), true).is_ok());
        assert!(check_games_arg("botgen", None, true).is_ok());
        assert!(check_games_arg("botgen", None, false).is_ok());
        assert!(check_games_arg("dagger", Some(TerranReact), true).is_err());
        assert!(check_games_arg("dagger", Some(TerranReact), false).is_ok());
        for src in ["distill", "branch", "stardata"] {
            assert!(check_games_arg(src, None, true).is_err(), "{src}");
        }
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
            let vo = ViewOptions { until_frame: Some(11_520), time_limit: Some(14_400), relabel, map_hash: &map_hash, memory: None, distill: false };
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
        teacher_game_agreement("URANIUM_TEST_TEACHER_REP", uranium_scripted::T0Build::TerranReact, 11_520);
    }

    /// 022-AC3 (OpenBW, ignored): the same for a 022 teacher botgen game against Terran or Protoss
    /// (`URANIUM_TEST_TEACHER_TP_REP`, the teacher the first Terran player, data cut at 10:00 as
    /// 022-R12): relabelled with `terran_tp` it gives its botgen labels (>= 95%).
    #[test]
    #[ignore = "needs OpenBW (.env) and URANIUM_TEST_TEACHER_TP_REP"]
    fn relabel_of_a_tp_teacher_game_gives_its_botgen_labels_022_ac3() {
        teacher_game_agreement("URANIUM_TEST_TEACHER_TP_REP", uranium_scripted::T0Build::TerranTp, 14_400);
    }

    /// The teacher-game agreement check of `relabel_of_a_teacher_game...` (041-R22, 022-AC3): the
    /// replay at env `var`, its first Terran view relabelled with `build` and labelled with its own
    /// commands, both cut at `until_frame`.
    fn teacher_game_agreement(var: &str, build: uranium_scripted::T0Build, until_frame: u32) {
        let _ = dotenvy::dotenv();
        let path = std::env::var(var).expect(var);
        let rep = crate::parse(&std::fs::read(&path).unwrap()).unwrap();
        let map_hash = hex(&Sha256::digest(&rep.map));
        let family = MapFamilies::load().unwrap().classify(&rep.header.map_name, &map_hash);
        let view = filter(&rep, &family, true, true).unwrap().remove(0);
        let cfg = PlayConfig::from_env(crate::play::work_dir_default().join("relabel-test")).unwrap();
        let labeler = Labeler::new().unwrap();
        let run = |relabel| {
            let vo = ViewOptions { until_frame: Some(until_frame), time_limit: Some(14_400), relabel, map_hash: &map_hash, memory: None, distill: false };
            match build_view(&rep, &view, &cfg, &labeler, None, vo).unwrap() {
                Ok(g) => g,
                Err(o) => panic!("{o:?}"),
            }
        };
        let (shadow, botgen) = (run(Some(build)), run(None));
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
