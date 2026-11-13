//! 041-R22 DAgger relabelling (040 출처 `dagger`): the model's own games are played back and the
//! 021-R8 teacher is run as a shadow on the model's (Terran) observations. The teacher sees every
//! observation a live T0 session would (live frame 1, then every 8 frames, `Act.advance_frames`;
//! `live_frame` maps them to playback frames) and its commands become the labels, at the frames a
//! live game records them; the model's own commands are only used for the 040-R8 checks.
//!
//! The teacher's 010 `Command`s are turned into the 1.16.1 replay commands BWAPI would send for
//! them (`GameImpl::executeCommand` with command optimizer level 0, as in the OpenBW bridge:
//! a select of the units, then the order), and those go through the same `Labeler` as every
//! replay (040-R3, R4, R11): one teacher command is one label, several commands at one
//! observation are consecutive labels with delay >= 1 (040-R4), and the time without commands is
//! NoOp labels of the largest fitting delay (031-R11), exactly as in a teacher botgen game.
//! No randomness: the same replay gives the same labels.

use std::collections::HashMap;

use anyhow::{Result, bail};
use uranium_proto::{CancelKind, CmdOrder, Command, Pos, RawFrame, Target, Tile, UnitId, UnitState};
use uranium_scripted::T0Build;
use uranium_scripted::types::{BUNKER, DROPSHIP, OVERLORD, building_tile, is_building};

use crate::cmd::{Cmd, op};
use crate::label::GAME_START_FRAME;
use crate::shard::MemoryAction;
use uranium_env::features::label::DecisionLogLine;

/// Teacher builds `uranium-replay build --relabel` accepts (041-R22): the 021-R8 teacher only.
pub fn teacher_build(name: &str) -> Result<T0Build> {
    match name {
        "terran_react" => Ok(T0Build::TerranReact),
        other => bail!("--relabel {other}: only terran_react (the 021-R8 teacher) relabels (041-R22)"),
    }
}

/// Frames between the observations of a live T0 session (`uranium_scripted` `STEP_FRAMES`, the
/// `advance_frames` of every teacher `Act`): live frames 1, 9, 17, ...
pub const TEACHER_STEP: u32 = 8;
/// A live observation of frame `L` shows the game as the replay playback shows it at frame
/// `L - LIVE_LAG`, and the commands decided on it are recorded at frame `L + RECORD_DELAY` (OpenBW
/// lockstep, measured 2026-11-07 on a teacher botgen game: shadow observations at `L - 2` give the
/// recorded commands at `L + 2`; at `L` the shadow saw each new SCV one observation early).
pub const LIVE_LAG: u32 = 2;
pub const RECORD_DELAY: u32 = 2;

/// The live frame number of the shadow observation at playback frame `f`, if `f` is one: the
/// game-start observation (011-R15, live frame 1, the earliest playback frame) and `L - LIVE_LAG`
/// for the later live frames `L` = 9, 17, ...
pub fn live_frame(f: u32) -> Option<u32> {
    if f == GAME_START_FRAME {
        return Some(GAME_START_FRAME);
    }
    let l = f + LIVE_LAG;
    (l > GAME_START_FRAME && (l - GAME_START_FRAME).is_multiple_of(TEACHER_STEP)).then_some(l)
}

/// The playback frames of the shadow observations up to playback frame `last` (inclusive).
pub fn teacher_frames(last: u32) -> Vec<u32> {
    (GAME_START_FRAME..=last.max(GAME_START_FRAME)).filter(|&f| live_frame(f).is_some()).collect()
}

/// 041-R28 ⑤: the playback frame whose observation the model saw on its live observation of
/// frame `live` (the unit check looks there): `live - LIVE_LAG` (the rule of [`live_frame`], its
/// inverse on the teacher cadence); the game-start observation (live frame 1) and the frames
/// before `1 + LIVE_LAG` at the first playback frame.
pub fn playback_frame(live: u32) -> u32 {
    live.saturating_sub(LIVE_LAG).max(GAME_START_FRAME)
}

/// 041-R28 ②: the live frame of the observation a playback frame `obs` shows: `obs + LIVE_LAG`,
/// the game-start observation (playback frame 1) at live frame 1 (as [`live_frame`]).
pub fn live_of_playback(obs: u32) -> u32 {
    if obs <= GAME_START_FRAME { GAME_START_FRAME } else { obs + LIVE_LAG }
}

/// 041-R28 ⑤ (AMB-041-17): a game whose decisions name a unit that is not an own unit at their
/// playback frame more often than this share is `None` on every step.
pub const MEMORY_MISMATCH_MAX: f64 = 0.01;
/// 041-R28 ④ (review fix, main interim decision): the log's end frame (the bridge's final frame)
/// and the replay's frame count may differ by at most this many frames, or the log is taken for
/// another game. Not measured on a real game yet: the records show both values on a mismatch.
pub const MEMORY_END_TOLERANCE: u32 = 8;

/// One model decision of a decision log (041-R28 ①): live observation frame and the decision.
pub type LoggedDecision = (u32, MemoryAction);

/// A complete decision log (041-R28 ①): the decisions in order and the end line's game end
/// frame and own player id.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelLog {
    pub decisions: Vec<LoggedDecision>,
    pub end_frame: Option<u32>,
    pub player: u8,
}

/// A game's decision log as the relabel build reads it (041-R28 ④).
#[derive(Debug, Clone, PartialEq)]
pub enum DecisionLogRead {
    /// No log file for the replay.
    Missing,
    /// Unreadable, a line that does not parse, decisions out of frame order, no end line (the
    /// session failed mid-game), lines after it, or an end line counting other decisions than the
    /// log holds.
    Incomplete(String),
    Complete(ModelLog),
}

/// Parses a decision log (`DecisionLogLine` JSON lines, 041-R28 ①).
pub fn parse_decision_log(text: &str) -> DecisionLogRead {
    let mut out: Vec<LoggedDecision> = Vec::new();
    let mut end = None;
    for (i, line) in text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
        if end.is_some() {
            return DecisionLogRead::Incomplete(format!("line {} after the end line", i + 1));
        }
        match serde_json::from_str::<DecisionLogLine>(line) {
            Ok(DecisionLogLine::Decision { frame, action, delay_bucket, units, .. }) => {
                if out.last().is_some_and(|(f, _)| *f > frame) {
                    return DecisionLogRead::Incomplete(format!("line {}: frame {frame} before the previous decision's", i + 1));
                }
                out.push((frame, MemoryAction { action, delay_bucket, units }));
            }
            Ok(DecisionLogLine::End { decisions, end_frame, player }) => {
                if decisions != out.len() as u64 {
                    return DecisionLogRead::Incomplete(format!("end line counts {decisions} decisions, the log has {}", out.len()));
                }
                end = Some((end_frame, player));
            }
            Err(e) => return DecisionLogRead::Incomplete(format!("line {}: {e}", i + 1)),
        }
    }
    let Some((end_frame, player)) = end else {
        return DecisionLogRead::Incomplete("no end line (the game did not end in the session)".into());
    };
    DecisionLogRead::Complete(ModelLog { decisions: out, end_frame, player })
}

/// Reads the decision log at `path` (041-R28 ④: absent = `Missing`).
pub fn read_decision_log(path: &std::path::Path) -> DecisionLogRead {
    match std::fs::read_to_string(path) {
        Ok(t) => parse_decision_log(&t),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => DecisionLogRead::Missing,
        Err(e) => DecisionLogRead::Incomplete(format!("reading {}: {e}", path.display())),
    }
}

/// 041-R28 ④ (review fix): the log of the view of BWAPI player `me` in a replay of
/// `replay_frames` frames, or why it is not used there: no complete log, the log of the other
/// player (`other_player`), or a game end more than [`MEMORY_END_TOLERANCE`] frames from the
/// replay's (`incomplete`: another game's log, or no final frame to tell).
pub fn bind_log(read: &DecisionLogRead, me: u8, replay_frames: u32) -> Result<&ModelLog, MemoryCheck> {
    let DecisionLogRead::Complete(log) = read else { return Err(MemoryCheck::without_log(read)) };
    let not = |status, reason: String| Err(MemoryCheck { status, reason: Some(reason), decisions: log.decisions.len(), checked: 0, mismatched: 0, decision_steps: None });
    if log.player != me {
        return not("other_player", format!("the log is player {}'s, this view is player {me}'s", log.player));
    }
    match log.end_frame {
        Some(e) if e.abs_diff(replay_frames) <= MEMORY_END_TOLERANCE => Ok(log),
        Some(e) => not("incomplete", format!("log end frame {e}, replay {replay_frames} frames (more than {MEMORY_END_TOLERANCE} apart)")),
        None => not("incomplete", format!("log without an end frame (replay {replay_frames} frames)")),
    }
}

/// 041-R28 ④ with several logs of one game (051-R2 act logs, one per model session): the log of
/// the view of BWAPI player `me` ([`bind_log`] on the complete log of that player), else why none
/// is used: no log (`missing`), a log that cannot be read whole (`incomplete`, whose player is
/// unknown), or only other players' logs (`other_player`). With one log this is [`bind_log`].
pub fn bind_logs(reads: &[DecisionLogRead], me: u8, replay_frames: u32) -> Result<&ModelLog, MemoryCheck> {
    if let Some(r) = reads.iter().find(|r| matches!(r, DecisionLogRead::Complete(l) if l.player == me)) {
        return bind_log(r, me, replay_frames);
    }
    match reads.iter().find(|r| !matches!(r, DecisionLogRead::Complete(_))).or(reads.first()) {
        Some(r) => bind_log(r, me, replay_frames),
        None => Err(MemoryCheck::without_log(&DecisionLogRead::Missing)),
    }
}

/// The playback frames at which the unit check (041-R28 ⑤) needs the perspective's own units:
/// those of the decisions up to playback frame `last`, sorted, each once.
pub fn decision_frames(log: &[LoggedDecision], last: u32) -> Vec<u32> {
    let mut v: Vec<u32> = log.iter().map(|(l, _)| playback_frame(*l)).filter(|&p| p <= last).collect();
    v.dedup(); // the log is in frame order, so the playback frames are too
    v
}

/// 041-R28 ⑤: (checked, mismatched): decisions up to playback frame `last` with a selection, and
/// those naming a unit that is not an own unit at their playback frame (`own_at`: own unit ids of
/// the perspective at a playback frame; a frame without an observation mismatches).
pub fn unit_check(log: &[LoggedDecision], last: u32, own_at: &HashMap<u32, std::collections::HashSet<u32>>) -> (usize, usize) {
    let (mut checked, mut mismatched) = (0, 0);
    for (live, d) in log {
        let p = playback_frame(*live);
        if p > last || d.units.is_empty() {
            continue;
        }
        checked += 1;
        let ok = own_at.get(&p).is_some_and(|own| d.units.iter().all(|u| own.contains(u)));
        mismatched += (!ok) as usize;
    }
    (checked, mismatched)
}

/// 041-R28 ②: each step's `memory_action` for steps observed at the playback frames `step_frames`
/// (increasing). Step i's observation is what the live game showed at live frame
/// `live_i` = [`live_of_playback`]`(obs_i)`; the model then had in memory exactly its decisions
/// made on live observations strictly before `live_i` (a decision made on live frame `live_i`
/// itself is recorded after that observation). So step i gets, in order, the decisions with live
/// frame L, `live_{i-1} <= L < live_i`; step 0 gets an empty list and decisions at or after
/// `live` of the last step belong to no step.
pub fn step_memory(log: &[LoggedDecision], step_frames: &[u32]) -> Vec<Vec<MemoryAction>> {
    let mut out = Vec::with_capacity(step_frames.len());
    let mut k = 0;
    for (i, &obs) in step_frames.iter().enumerate() {
        let live = live_of_playback(obs);
        let mut here = Vec::new();
        while k < log.len() && log[k].0 < live {
            if i > 0 {
                here.push(log[k].1.clone());
            }
            k += 1;
        }
        out.push(here);
    }
    out
}

/// 041-R28 decision-time steps (2026-11-13, user decision 2026-11-13; 051-R13 ④): in a build without
/// relabelling whose log is used, every logged decision (NoOp ones included) gets a step at its
/// playback frame L - 2 (`playback_frame`) up to `limit`, unless a label step is already at that
/// frame. There is no clean label for a step between two command labels (the earlier label's
/// delay already reaches the next one, 040-R4 / 031-R11), so the added step's label is `unknown`
/// (031-AC3: BC weight 0, not a 040-R19 invalid label) with the frames to the next step (the
/// game end after the last one) as its delay: an observation / memory only row (`memory_action`,
/// 041-R28 ③), not a BC training sample. `labels` stay sorted by frame; returns the steps added.
pub fn add_decision_steps(labels: &mut Vec<crate::label::Label>, log: &[LoggedDecision], limit: u32, end_frame: u32) -> usize {
    use crate::label::{Label, UNKNOWN_ACTION};
    let have: std::collections::HashSet<u32> = labels.iter().map(|l| l.frame).collect();
    let mut add: Vec<u32> = log.iter().map(|(l, _)| playback_frame(*l)).filter(|p| *p <= limit && !have.contains(p)).collect();
    add.sort_unstable();
    add.dedup();
    let n = add.len();
    labels.extend(add.into_iter().map(|frame| Label { frame, action: UNKNOWN_ACTION, delay: 0, queued: false, selected: Vec::new(), target_unit: None, target_tile: None }));
    labels.sort_by_key(|l| l.frame); // stable: the existing labels keep their order
    for i in 0..labels.len() {
        if labels[i].is_unknown() && labels[i].delay == 0 && !have.contains(&labels[i].frame) {
            let next = labels.get(i + 1).map_or(end_frame, |l| l.frame);
            labels[i].delay = next.saturating_sub(labels[i].frame).max(1);
        }
    }
    n
}

/// 044-R2 / 040-R24 (`--src distill`): the label slots of a decision-step game. One step at the
/// game-start observation (040-R18, playback frame 1) and one at each logged decision's playback
/// frame L - 2 ([`playback_frame`], NoOp decisions included) up to `limit`; no step for the replay's
/// own commands. The model decides on its first live observation (live frame 1, playback frame 1),
/// so the game-start step is that decision's step, and decisions whose playback frames coincide (live
/// frames 1..=3 all show playback frame 1) share one step. Every label is `unknown` (031-AC3: BC
/// weight 0) with the frames to the next step (the game end `end_frame` after the last one) as its
/// delay, as the 041-R28 decision-time steps ([`add_decision_steps`]): slots the 044-R3 policy
/// labeller fills. Returns the labels (sorted by frame) and the steps at decision frames.
pub fn distill_labels(log: &[LoggedDecision], limit: u32, end_frame: u32) -> (Vec<crate::label::Label>, usize) {
    use crate::label::{Label, UNKNOWN_ACTION};
    let mut decided: Vec<u32> = log.iter().map(|(l, _)| playback_frame(*l)).filter(|p| *p <= limit).collect();
    decided.sort_unstable();
    decided.dedup();
    let mut frames = decided.clone();
    if frames.first() != Some(&GAME_START_FRAME) {
        frames.insert(0, GAME_START_FRAME);
    }
    let labels = frames
        .iter()
        .enumerate()
        .map(|(i, &frame)| {
            let next = frames.get(i + 1).copied().unwrap_or(end_frame);
            Label { frame, action: UNKNOWN_ACTION, delay: next.saturating_sub(frame).max(1), queued: false, selected: Vec::new(), target_unit: None, target_tile: None }
        })
        .collect();
    (labels, decided.len())
}

/// 041-R28 ④ / ⑤ record (`memory` of a written relabelled perspective, build records).
#[derive(serde::Serialize, Debug, Clone, PartialEq, Eq)]
pub struct MemoryCheck {
    /// `ok` (every step has `memory_action`), `missing` (no log), `incomplete` (see
    /// [`DecisionLogRead::Incomplete`] and [`bind_log`]), `other_player` (the log is the other
    /// player's), `unit_mismatch` (more than [`MEMORY_MISMATCH_MAX`]).
    pub status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// Decision lines of the log.
    pub decisions: usize,
    /// Decisions with a selection up to the last relabelled frame, and those whose selection is
    /// not the perspective's own units at their playback frame (041-R28 ⑤).
    pub checked: usize,
    pub mismatched: usize,
    /// 041-R28 decision-time steps (2026-11-13, builds without relabelling): observation / memory
    /// only steps added at the decisions' playback frames ([`add_decision_steps`]); absent when
    /// none are added (relabelled builds, logs not used). In a `distill` build (044-R2) the steps
    /// at decision frames ([`distill_labels`]; the game-start step counts when a decision is there).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decision_steps: Option<usize>,
}

impl MemoryCheck {
    pub fn without_log(read: &DecisionLogRead) -> Self {
        let (status, reason) = match read {
            DecisionLogRead::Missing => ("missing", None),
            DecisionLogRead::Incomplete(r) => ("incomplete", Some(r.clone())),
            DecisionLogRead::Complete(_) => unreachable!("a complete log is checked"),
        };
        Self { status, reason, decisions: 0, checked: 0, mismatched: 0, decision_steps: None }
    }

    /// The unit check of a complete log: `ok` unless more than 1% mismatch.
    pub fn checked(decisions: usize, checked: usize, mismatched: usize) -> Self {
        let bad = checked > 0 && mismatched as f64 / checked as f64 > MEMORY_MISMATCH_MAX;
        Self { status: if bad { "unit_mismatch" } else { "ok" }, reason: None, decisions, checked, mismatched, decision_steps: None }
    }

    pub fn ok(&self) -> bool {
        self.status == "ok"
    }
}

/// Replay opcode for a teacher command BWAPI's path is not modelled for (none of the 021-R8
/// teacher's commands): the labeler makes it an `unknown` label (weight 0, counted in coverage).
pub const UNMODELED_OP: u8 = 0xff;
/// Unit type field of a targeted order without a type (BW `UnitTypes::None`).
const NO_TYPE: u16 = 228;
/// BWAPI `cancelTrain(-2)`: the last queue slot.
const LAST_SLOT: u16 = 254;
const SHUTTLE: u16 = 72;

fn order(name: &str) -> u8 {
    uranium_env::types::order_id(name).unwrap_or_else(|| panic!("order {name}")) as u8
}

fn le(v: u16) -> [u8; 2] {
    v.to_le_bytes()
}

/// `TargetedOrder` (0x15) payload: position, target tag, unit type, order, queued.
fn targeted(order: u8, p: Pos, tag: u32, queued: bool) -> (u8, Vec<u8>) {
    let mut d = Vec::with_capacity(10);
    d.extend(le(p.0));
    d.extend(le(p.1));
    d.extend(le(tag as u16));
    d.extend(le(NO_TYPE));
    d.extend([order, queued as u8]);
    (op::TARGETED_ORDER, d)
}

/// `RightClick` (0x14) payload: position, target tag, unit type, queued.
fn right_click(p: Pos, tag: u32, queued: bool) -> (u8, Vec<u8>) {
    let mut d = Vec::with_capacity(9);
    d.extend(le(p.0));
    d.extend(le(p.1));
    d.extend(le(tag as u16));
    d.extend(le(NO_TYPE));
    d.push(queued as u8);
    (op::RIGHT_CLICK, d)
}

/// `Build` (0x0C) payload: order, tile, unit type.
fn build(order: u8, t: Tile, ty: u16) -> (u8, Vec<u8>) {
    let mut d = vec![order];
    d.extend(le(t.0));
    d.extend(le(t.1));
    d.extend(le(ty));
    (op::BUILD, d)
}

fn select(ids: &[UnitId]) -> (u8, Vec<u8>) {
    let mut d = vec![ids.len() as u8];
    for &id in ids {
        d.extend(le(id as u16));
    }
    (op::SELECT, d)
}

/// The replay commands (select, then order) BWAPI sends for the teacher's command `c` issued at
/// observation `f` of player `me` (BWAPI id) whose replay command id is `player`. Units the bridge
/// would not find (not own, not observed) are left out like the bridge rejects them; a command
/// left without units sends nothing. Buildings in a unit command get it one by one (BWAPI
/// `canIssueCommandGrouped`), the other units as one group (the bridge's `Game::issueCommand`).
/// The result is always (select, order) pairs (`Shadow::observe` relies on it).
pub fn command_cmds(c: &Command, f: &RawFrame, me: u8, player: u8) -> Vec<Cmd> {
    let unit = |id: UnitId| f.units.iter().find(|u| u.id == id);
    let own = |id: UnitId| unit(id).filter(|u| u.owner == me);
    let at = |id: UnitId| unit(id).map_or((0, 0), |u| u.pos);
    let mut out: Vec<(u8, Vec<u8>)> = Vec::new();
    let single = |id: UnitId, out: &mut Vec<(u8, Vec<u8>)>, cmd: (u8, Vec<u8>)| {
        if own(id).is_some() {
            out.push(select(&[id]));
            out.push(cmd);
        }
    };
    match c {
        Command::Unit { units, order: o, queued } => {
            let mine: Vec<&UnitState> = units.iter().filter_map(|&id| own(id)).collect();
            let (buildings, mobile): (Vec<&UnitState>, Vec<&UnitState>) = mine.into_iter().partition(|u| is_building(u.unit_type));
            let mut groups: Vec<Vec<&UnitState>> = Vec::new();
            if !mobile.is_empty() {
                groups.push(mobile);
            }
            groups.extend(buildings.into_iter().map(|b| vec![b]));
            for g in groups {
                if let Some(cmd) = unit_order(o, *queued, g[0], &unit, &at) {
                    out.push(select(&g.iter().map(|u| u.id).collect::<Vec<_>>()));
                    out.push(cmd);
                }
            }
        }
        Command::Build { worker, unit_type, tile } => {
            let o = if is_addon(*unit_type) { order("PlaceAddon") } else { order("PlaceBuilding") };
            single(*worker, &mut out, build(o, *tile, *unit_type));
        }
        Command::BuildAddon { building, unit_type } => {
            let t = own(*building).map_or((0, 0), |b| {
                let p = building_tile(b);
                (p.0 + 4, p.1 + 1)
            });
            single(*building, &mut out, build(order("PlaceAddon"), t, *unit_type));
        }
        Command::Train { building, unit_type } => single(*building, &mut out, (op::TRAIN, le(*unit_type).to_vec())),
        Command::Morph { unit, unit_type } => {
            let o = if is_building(*unit_type) { 0x35 } else { op::UNIT_MORPH };
            single(*unit, &mut out, (o, le(*unit_type).to_vec()));
        }
        Command::Research { building, tech } => single(*building, &mut out, (0x30, vec![*tech as u8])),
        Command::Upgrade { building, upgrade } => single(*building, &mut out, (0x32, vec![*upgrade as u8])),
        Command::Cancel { unit, kind } => {
            let building = own(*unit).is_some_and(|u| is_building(u.unit_type));
            let cmd = match kind {
                CancelKind::Construction => (0x18, vec![]),
                CancelKind::Train => (op::CANCEL_TRAIN, le(LAST_SLOT).to_vec()),
                CancelKind::Research => (0x31, vec![]),
                CancelKind::Upgrade => (0x33, vec![]),
                CancelKind::Addon => (0x34, vec![]),
                CancelKind::Morph if building => (0x18, vec![]),
                CancelKind::Morph => (0x19, vec![]),
            };
            single(*unit, &mut out, cmd);
        }
        Command::SetRally { building, target } => {
            let cmd = match target {
                Target::Pos(p) => targeted(order("RallyPointTile"), *p, 0, false),
                Target::Unit(t) => targeted(order("RallyPointUnit"), at(*t), *t, false),
            };
            single(*building, &mut out, cmd);
        }
    }
    out.into_iter().map(|(op, data)| Cmd { frame: f.frame, player, op, data }).collect()
}

fn is_addon(ty: u16) -> bool {
    uranium_env::types::TypeTables::get().unit(ty).is_some_and(|t| t.is_addon())
}

/// The order command of a unit command for its first unit `u` (BWAPI `executeCommand`).
fn unit_order<'a>(o: &CmdOrder, q: bool, u: &UnitState, unit: &impl Fn(UnitId) -> Option<&'a UnitState>, at: &impl Fn(UnitId) -> Pos) -> Option<(u8, Vec<u8>)> {
    let qb = vec![q as u8];
    Some(match o {
        CmdOrder::Move(Target::Pos(p)) => targeted(order("Move"), *p, 0, q),
        CmdOrder::Move(Target::Unit(t)) => targeted(order("Follow"), at(*t), *t, q),
        CmdOrder::Attack(Target::Pos(p)) => targeted(order("AttackMove"), *p, 0, q),
        CmdOrder::Attack(Target::Unit(t)) => {
            let o = if is_building(u.unit_type) { order("TowerAttack") } else { order("Attack1") };
            targeted(o, at(*t), *t, q)
        }
        CmdOrder::Patrol(p) => targeted(order("Patrol"), *p, 0, q),
        CmdOrder::Hold => (0x2b, qb),
        CmdOrder::Stop => (0x1a, qb),
        CmdOrder::RightClick(Target::Pos(p)) => right_click(*p, 0, q),
        CmdOrder::RightClick(Target::Unit(t)) => right_click(at(*t), *t, q),
        CmdOrder::Siege => (0x26, vec![0]),
        CmdOrder::Unsiege => (0x25, vec![0]),
        CmdOrder::Cloak => (0x21, vec![0]),
        CmdOrder::Decloak => (0x22, vec![0]),
        CmdOrder::Burrow => (0x2c, vec![0]),
        CmdOrder::Unburrow => (0x2d, vec![0]),
        CmdOrder::Lift => (0x2f, vec![0; 4]),
        CmdOrder::Land(t) => build(order("BuildingLand"), *t, u.unit_type),
        // BWAPI Load: a bunker picks up (PickupBunker), a transport picks up (PickupTransport),
        // a unit loading into a bunker / transport right-clicks it.
        CmdOrder::Load(t) => {
            let transport = |ty: u16| matches!(ty, DROPSHIP | SHUTTLE | OVERLORD);
            if u.unit_type == BUNKER {
                targeted(order("PickupBunker"), at(*t), *t, q)
            } else if transport(u.unit_type) {
                targeted(order("PickupTransport"), at(*t), *t, q)
            } else if unit(*t).is_some_and(|x| x.unit_type == BUNKER || transport(x.unit_type)) {
                right_click(at(*t), *t, q)
            } else {
                return None;
            }
        }
        CmdOrder::UnloadAll(None) if u.unit_type == BUNKER => (0x28, qb),
        CmdOrder::UnloadAll(None) => targeted(order("MoveUnload"), u.pos, 0, q),
        CmdOrder::UnloadAll(Some(p)) => targeted(order("MoveUnload"), *p, 0, q),
        CmdOrder::Repair(t) => targeted(order("Repair"), at(*t), *t, q),
        CmdOrder::Gather(t) => targeted(order("Harvest1"), at(*t), *t, q),
        CmdOrder::ReturnCargo => (0x1e, qb),
        CmdOrder::Unload(t) => (op::UNLOAD, le(*t as u16).to_vec()),
        // Stim packs have their own opcode; other spells are not 021-R8 teacher commands.
        CmdOrder::UseTech { tech, target: None } if Some(*tech as u32) == stim() => (0x36, vec![]),
        CmdOrder::UseTech { .. } => (UNMODELED_OP, vec![]),
    })
}

fn stim() -> Option<u32> {
    uranium_env::types::TypeTables::get().techs.iter().find(|t| t.name == "Stim_Packs").map(|t| t.id as u32)
}

/// Own units of `me` at an observation for the labeler (040-R3 selection rules).
pub fn own_units(f: &RawFrame, me: u8) -> HashMap<u32, crate::label::Own> {
    f.units.iter().filter(|u| u.owner == me).map(|u| (u.id, crate::label::Own::from_state(u))).collect()
}

/// The teacher run as a shadow on one perspective's observations (041-R22). Built from the
/// bridge's GameInfo and the map precompute hints like a live T0 session
/// (`uranium_agent::ScriptedPolicy`); `observe` gets every observation of the playback in order
/// and acts only on the live session's frames (`teacher_frames`) up to `last`.
pub struct Shadow {
    bot: Box<dyn uranium_scripted::Bot>,
    me: u8,
    player: u8,
    last: u32,
    /// The teacher's commands as replay commands, in order.
    pub cmds: Vec<Cmd>,
    /// Own units at each frame the teacher issued commands (labeler selection rules).
    pub units_at: HashMap<u32, HashMap<u32, crate::label::Own>>,
    /// Teacher observations (diagnostics).
    pub observations: u32,
    /// Teacher replay command pairs (select, order) dropped as repeats (`REPEAT_WINDOW`, build
    /// records).
    pub repeats_dropped: u32,
    /// Frame of the last kept label of each command (order opcode and payload, select payload).
    last_kept: HashMap<RepeatKey, u32>,
}

/// A teacher command as the game receives it: order opcode, order payload, select payload.
type RepeatKey = (u8, Vec<u8>, Vec<u8>);

/// 041-R22 (11th advisory, AMB-041-12 메인 임시 결정): a teacher command equal to one kept less
/// than this many frames earlier is dropped. The model does not carry out the shadow's commands,
/// so the teacher repeats them at every observation (8 frames) while the state lasts (an idle
/// command center with minerals: Train_SCV every 8 frames); 32 is the largest delay bucket
/// (031-R11), so the kept labels of such a run are 32 frames apart with no NoOp decision between
/// them. A live teacher rarely repeats a command within 32 frames (its commands are executed and
/// it re-orders on 48-frame timers; relabelling teacher botgen games is unchanged by this rule).
pub const REPEAT_WINDOW: u32 = 32;

impl Shadow {
    /// `player`: the perspective's replay command id; `last`: the last frame the teacher is fed.
    pub fn new(info: &uranium_proto::GameInfo, build: T0Build, hints: uranium_scripted::MapHints, player: u8, last: u32) -> Self {
        Self {
            bot: uranium_scripted::for_game_hints(info, build, hints),
            me: info.self_player,
            player,
            last,
            cmds: Vec::new(),
            units_at: HashMap::new(),
            observations: 0,
            repeats_dropped: 0,
            last_kept: HashMap::new(),
        }
    }

    /// One observation of the playback; frames that are not a live teacher observation are skipped.
    /// The teacher sees the live frame number (its timers and rule boundaries count live frames)
    /// and its commands are placed where a live game records them (`RECORD_DELAY`).
    pub fn observe(&mut self, f: &RawFrame) {
        let Some(live) = live_frame(f.frame).filter(|_| f.frame <= self.last) else { return };
        self.observations += 1;
        let mut seen = f.clone();
        seen.frame = live;
        let act = self.bot.act(&seen);
        let at = live + RECORD_DELAY;
        let start = self.cmds.len();
        for c in &act.cmds {
            // `command_cmds` gives (select, order) pairs.
            let cmds = command_cmds(c, &seen, self.me, self.player);
            for pair in cmds.chunks(2) {
                let [select, order] = pair else { unreachable!("command_cmds gives (select, order) pairs") };
                // 041-R22 (11th advisory): the same command again within `REPEAT_WINDOW` of the last
                // kept one is dropped whole (no label, no NoOp: its time joins the previous delay).
                let key = (order.op, order.data.clone(), select.data.clone());
                if self.last_kept.get(&key).is_some_and(|&k| at.saturating_sub(k) < REPEAT_WINDOW) {
                    self.repeats_dropped += 1;
                    continue;
                }
                self.last_kept.insert(key, at);
                self.cmds.extend(pair.iter().map(|c| Cmd { frame: at, ..c.clone() }));
            }
        }
        if self.cmds.len() > start {
            self.units_at.insert(at, own_units(f, self.me));
        }
    }

    /// The teacher's labels (040-R4 delays, 031-R11 NoOps, 040-R18 first step) and coverage.
    pub fn labels(&self, labeler: &crate::label::Labeler, end_frame: u32) -> (Vec<crate::label::Label>, crate::label::Coverage) {
        let empty = HashMap::new();
        labeler.label(&self.cmds, end_frame, |f| self.units_at.get(&f).unwrap_or(&empty))
    }

    /// 021-R8 record of the shadow teacher (`scripted.react`, diagnostics).
    pub fn react_stats(&self) -> Option<uranium_scripted::ReactStats> {
        self.bot.react_stats()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::label::Labeler;
    use uranium_proto::flags;
    use uranium_scripted::types::{BARRACKS, CC, MARINE, MINERAL_FIELDS, SCV};

    fn unit(id: u32, owner: u8, ty: u16, pos: Pos) -> UnitState {
        UnitState { id, owner, unit_type: ty, pos, flags: flags::COMPLETED, ..Default::default() }
    }

    fn frame_at(n: u32, units: Vec<UnitState>, minerals: i32) -> RawFrame {
        let all = |w, h| uranium_proto::BitGrid::from_bytes(w, h, &vec![1; w as usize * h as usize]);
        RawFrame {
            frame: n,
            self_state: uranium_proto::PlayerState { minerals, supply_used: 8, supply_total: 20, upgrades: vec![0; 63], ..Default::default() },
            units,
            visibility: all(64, 64),
            explored: all(64, 64),
            creep: uranium_proto::BitGrid::new(64, 64),
            events: vec![],
            effects: vec![],
        }
    }

    fn frame(units: Vec<UnitState>) -> RawFrame {
        frame_at(97, units, 0)
    }

    fn info() -> uranium_proto::GameInfo {
        use uranium_proto::{PlayerInfo, Race, StaticMap};
        let all = |w, h| uranium_proto::BitGrid::from_bytes(w, h, &vec![1; w as usize * h as usize]);
        uranium_proto::GameInfo {
            map_name: "t".into(),
            map_hash: [0; 32],
            map_size_tiles: (64, 64),
            self_player: 0,
            players: vec![
                PlayerInfo { id: 0, name: "a".into(), race: Race::Terran, start_tile: Some((4, 4)), is_self: true, is_enemy: false },
                PlayerInfo { id: 1, name: "b".into(), race: Race::Zerg, start_tile: None, is_self: false, is_enemy: true },
            ],
            static_map: StaticMap { width_tiles: 64, height_tiles: 64, walkable: all(256, 256), buildable: all(64, 64), ground_height: vec![0; 64 * 64], start_locations: vec![(4, 4), (56, 56)] },
            latency_frames: 0,
        }
    }

    /// A Terran start: command center, four idle SCVs, minerals.
    fn start_units() -> Vec<UnitState> {
        let idle = |id| UnitState { flags: flags::COMPLETED | flags::IDLE, ..unit(id, 0, SCV, (200, 230)) };
        let mut v = vec![unit(10, 0, CC, (192, 176)), idle(1), idle(2), idle(3), idle(4)];
        v.extend((0..4).map(|i| unit(50 + i, 11, MINERAL_FIELDS[0], (100 + 32 * i as u16, 60))));
        v
    }

    fn names(labels: &[crate::label::Label]) -> Vec<(u32, String)> {
        let table = uranium_env::actions::ActionTable::load().unwrap();
        labels.iter().map(|l| (l.frame, table.actions[l.action as usize].name.clone())).collect()
    }

    /// 041-R22: the shadow acts only at the live session's observations (playback frames 1, 7, 15,
    /// ... up to `last`), its labels are the teacher's commands (one label each, 040-R4) where a
    /// live game records them (live frame + 2), and the same observations give the same labels.
    #[test]
    fn shadow_labels_are_the_teachers_and_deterministic_041_r22() {
        let frames: Vec<RawFrame> = [1u32, 5, 7, 9, 12, 15, 23, 31].iter().map(|&n| frame_at(n, start_units(), 50)).collect();
        let run = || {
            let mut s = Shadow::new(&info(), T0Build::TerranReact, uranium_scripted::MapHints::default(), 3, 23);
            for f in &frames {
                s.observe(f);
            }
            s
        };
        let (a, b) = (run(), run());
        assert_eq!(a.observations, 4, "playback frames 1, 7, 15, 23 only");
        assert_eq!(a.cmds, b.cmds);
        let lab = Labeler::new().unwrap();
        let (la, _) = a.labels(&lab, 60);
        assert_eq!(la, b.labels(&lab, 60).0);
        let got = names(&la);
        // Frame 1: the teacher trains an SCV and sends the four idle SCVs to mine, one label each.
        let first: Vec<&str> = got.iter().filter(|(f, n)| *f < 9 && n != "NoOp").map(|(_, n)| n.as_str()).collect();
        assert_eq!(first.iter().filter(|n| **n == "Gather_Unit").count(), 4, "{got:?}");
        assert_eq!(first.iter().filter(|n| **n == "Train_SCV").count(), 1, "{got:?}");
        // Every teacher command is recorded 2 frames after its live observation (1, 9, 17, 25).
        assert!(a.cmds.iter().all(|c| [3, 11, 19, 27].contains(&c.frame)));
        assert!(a.cmds.iter().any(|c| c.frame == 3));
        // The same observation four times: the later identical commands are repeats (dropped).
        assert!(a.repeats_dropped > 0 && a.repeats_dropped == b.repeats_dropped);
        // The steps tile the game from the game-start observation.
        assert_eq!(la[0].frame, GAME_START_FRAME);
        assert!(la.windows(2).all(|w| w[1].frame == w[0].frame + w[0].delay));
    }

    /// 041-R22 (11th advisory, AMB-041-12): a teacher command equal to one kept less than 32
    /// frames earlier (same order and payload, same selection) is dropped whole — no label and no
    /// NoOp, its time joins the previous label's delay — so a command the model never carries out
    /// is labelled every 32 frames; a different command (another unit, another order) is kept.
    #[test]
    fn repeated_teacher_commands_are_thinned_041_r22() {
        assert_eq!(REPEAT_WINDOW, *crate::label::DELAY_BUCKETS.last().unwrap());
        // An idle command center with minerals and nothing else to do: the teacher trains an SCV
        // at every observation (live frames 1, 9, ..., 65 = playback 1, 7, ..., 63).
        let cc_only = || vec![unit(10, 0, CC, (192, 176))];
        let frames: Vec<RawFrame> = teacher_frames(63).into_iter().map(|n| frame_at(n, cc_only(), 50)).collect();
        let mut s = Shadow::new(&info(), T0Build::TerranReact, uranium_scripted::MapHints::default(), 3, 63);
        frames.iter().for_each(|f| s.observe(f));
        let trains: Vec<u32> = s.cmds.iter().filter(|c| c.op == op::TRAIN).map(|c| c.frame).collect();
        // Recorded at live + 2 = 3, 11, ..., 67: kept at 3, 35 (32 later) and 67; 6 dropped.
        assert_eq!(trains, vec![3, 35, 67]);
        assert_eq!((s.observations, s.repeats_dropped), (9, 6));
        let lab = Labeler::new().unwrap();
        let (labels, _) = s.labels(&lab, 100);
        let got = names(&labels);
        let kept: Vec<(u32, u32)> = labels.iter().filter(|l| got.iter().any(|(f, n)| *f == l.frame && n == "Train_SCV")).map(|l| (l.frame, l.delay)).collect();
        // No NoOp decision between the kept labels: each Train_SCV's delay reaches the next one.
        assert_eq!(kept, vec![(3, 32), (35, 32), (67, 32)]);
        assert!(!got.iter().any(|(f, n)| (3..67).contains(f) && n == "NoOp"), "{got:?}");
        // Another command center (another selection) is another command: kept.
        let two = || vec![unit(10, 0, CC, (192, 176)), unit(11, 0, CC, (800, 800))];
        let mut s = Shadow::new(&info(), T0Build::TerranReact, uranium_scripted::MapHints::default(), 3, 7);
        s.observe(&frame_at(1, two(), 100));
        assert_eq!(s.cmds.iter().filter(|c| c.op == op::TRAIN).count(), 2);
        assert_eq!(s.repeats_dropped, 0);
    }

    /// The live T0 session's observation frames 1, 9, 17, ... (011-R15, `Act.advance_frames` 8)
    /// are the playback frames 1, 7, 15, ... (`LIVE_LAG`).
    #[test]
    fn teacher_frames_follow_the_live_cadence_041_r22() {
        assert_eq!(teacher_frames(30), vec![1, 7, 15, 23]);
        assert_eq!(teacher_frames(23), vec![1, 7, 15, 23]);
        assert_eq!(teacher_frames(0), vec![1]);
        assert_eq!((live_frame(1), live_frame(7), live_frame(15), live_frame(9), live_frame(3)), (Some(1), Some(9), Some(17), None, None));
        assert!(teacher_build("zerg_rush").is_err());
        assert_eq!(teacher_build("terran_react").unwrap(), T0Build::TerranReact);
    }

    fn dec(live: u32, action: u16, units: Vec<u32>) -> LoggedDecision {
        (live, MemoryAction { action, delay_bucket: 1, units })
    }

    /// 041-R28 ②: a model decision on live frame L is at playback frame L - 2 (the inverse of
    /// `live_frame` on the teacher cadence); the first live frames map to the game-start frame.
    #[test]
    fn decisions_move_to_playback_frames_041_r28() {
        for f in teacher_frames(400) {
            assert_eq!(playback_frame(live_frame(f).unwrap()), f);
        }
        assert_eq!([1, 2, 3, 4, 9, 100].map(playback_frame), [1, 1, 1, 2, 7, 98]);
    }

    /// 041-R28 ②: step i (playback observation obs_i, live frame live_i = obs_i + 2, the game
    /// start at 1) gets the decisions made on live frames L with live_{i-1} <= L < live_i, in
    /// order: a decision made on the live observation live_i itself is recorded after that
    /// observation, so it belongs to the next step. The first step gets an empty list; decisions
    /// at or after the last step's live frame belong to no step.
    #[test]
    fn step_memory_takes_the_half_open_interval_041_r28() {
        assert_eq!([1, 2, 11, 19].map(live_of_playback), [1, 4, 13, 21]);
        // Live frames 1, 9, 13 (= live of step 1's obs 11), 20, 21 (= live of step 2's obs 19), 40.
        let log = vec![dec(1, 1, vec![]), dec(9, 2, vec![5]), dec(13, 3, vec![]), dec(20, 4, vec![]), dec(21, 5, vec![6, 7]), dec(40, 6, vec![])];
        let got = step_memory(&log, &[1, 11, 19, 35]);
        let actions: Vec<Vec<u16>> = got.iter().map(|v| v.iter().map(|a| a.action).collect()).collect();
        assert_eq!(actions, vec![vec![], vec![1, 2], vec![3, 4], vec![5]]);
        assert_eq!(got[3][0].units, vec![6, 7]);
        // The decision made exactly on step 1's live observation (13) is not in step 1's memory
        // (the model had not decided it yet when it saw that observation), but in step 2's.
        assert!(!got[1].iter().any(|a| a.action == 3) && got[2][0].action == 3);
        // One frame earlier it is before the observation: step 1.
        let early = vec![dec(12, 3, vec![])];
        assert_eq!(step_memory(&early, &[1, 11, 19]).iter().map(Vec::len).collect::<Vec<_>>(), vec![0, 1, 0]);
        // Steps with no decision in between get an empty list, not nothing.
        assert_eq!(step_memory(&log, &[1, 2, 3]), vec![vec![], vec![log[0].1.clone()], vec![]]);
        assert!(step_memory(&[], &[1, 9]).iter().all(Vec::is_empty));
    }

    /// 041-R28 ① / ④: a complete log has its end line with the right count and frames in order;
    /// anything else is incomplete, and no file is missing.
    #[test]
    fn decision_log_completeness_041_r28() {
        let d = |f: u32| format!(r#"{{"frame":{f},"action":4,"delay_bucket":2,"units":[3,9]}}"#);
        let end = |n: u64| format!(r#"{{"decisions":{n},"end_frame":900,"player":1}}"#);
        let ok = format!("{}\n{}\n{}\n", d(1), d(9), end(2));
        let DecisionLogRead::Complete(log) = parse_decision_log(&ok) else { panic!("complete") };
        assert_eq!(log.decisions, vec![(1, MemoryAction { action: 4, delay_bucket: 2, units: vec![3, 9] }), (9, MemoryAction { action: 4, delay_bucket: 2, units: vec![3, 9] })]);
        assert_eq!((log.end_frame, log.player), (Some(900), 1));
        assert_eq!(parse_decision_log(&end(0)), DecisionLogRead::Complete(ModelLog { decisions: vec![], end_frame: Some(900), player: 1 }));
        // No end line (a session that failed mid-game), a wrong count, frames out of order, a bad
        // line, a line after the end line, an end line without the player.
        for bad in [
            format!("{}\n{}\n", d(1), d(9)),
            format!("{}\n{}\n", d(1), end(2)),
            format!("{}\n{}\n{}\n", d(9), d(1), end(2)),
            format!("{}\nnot json\n{}\n", d(1), end(1)),
            format!("{}\n{}\n{}\n", d(1), end(1), d(9)),
            format!("{}\n{}\n", d(1), r#"{"decisions":1,"end_frame":900}"#),
        ] {
            assert!(matches!(parse_decision_log(&bad), DecisionLogRead::Incomplete(_)), "{bad}");
        }
        assert_eq!(read_decision_log(std::path::Path::new("no/such/g1.jsonl")), DecisionLogRead::Missing);
    }

    /// 041-R28 ④ (review fix): a complete log is used only for its own player's view of the game
    /// that ends where the replay ends (within `MEMORY_END_TOLERANCE` frames).
    #[test]
    fn decision_log_binds_to_player_and_game_041_r28() {
        let read = |end_frame, player| DecisionLogRead::Complete(ModelLog { decisions: vec![dec(1, 0, vec![])], end_frame, player });
        assert!(bind_log(&read(Some(14_400), 0), 0, 14_398).is_ok());
        assert!(bind_log(&read(Some(14_400), 0), 0, 14_400 + MEMORY_END_TOLERANCE).is_ok());
        let other = bind_log(&read(Some(14_400), 1), 0, 14_400).unwrap_err();
        assert_eq!((other.status, other.decisions), ("other_player", 1));
        let far = bind_log(&read(Some(9_000), 0), 0, 14_400).unwrap_err();
        assert_eq!(far.status, "incomplete");
        assert!(far.reason.unwrap().contains("9000"));
        assert_eq!(bind_log(&read(None, 0), 0, 14_400).unwrap_err().status, "incomplete");
        assert_eq!(bind_log(&DecisionLogRead::Missing, 0, 14_400).unwrap_err().status, "missing");
    }

    /// 041-R28 ⑤: decisions with a selection up to `last` are checked against the own units at
    /// their playback frame (no observation = mismatch); more than 1% mismatched fails the game.
    #[test]
    fn unit_check_counts_and_one_percent_041_r28() {
        let own: HashMap<u32, std::collections::HashSet<u32>> = HashMap::from([(7, [5, 6].into()), (11, [5].into())]);
        // Playback 7: own; 11: unit 6 gone; 18: no observation; NoOp without units: not checked;
        // playback 98 after `last`: not checked.
        let log = vec![dec(9, 1, vec![5, 6]), dec(13, 2, vec![5, 6]), dec(20, 3, vec![5]), dec(21, 0, vec![]), dec(100, 4, vec![99])];
        assert_eq!(unit_check(&log, 50, &own), (3, 2));
        assert_eq!(decision_frames(&log, 50), vec![7, 11, 18, 19]);
        assert_eq!(MemoryCheck::checked(200, 100, 1).status, "ok", "1% is allowed");
        assert_eq!(MemoryCheck::checked(200, 100, 2).status, "unit_mismatch");
        assert_eq!(MemoryCheck::checked(0, 0, 0).status, "ok");
        assert_eq!(MemoryCheck::without_log(&DecisionLogRead::Missing).status, "missing");
        assert_eq!(MemoryCheck::without_log(&DecisionLogRead::Incomplete("x".into())).reason.as_deref(), Some("x"));
    }

    /// 041-R22: every 021-R8 teacher command maps to the 034 action of its "모델 표현" row
    /// through the same labeler as a botgen replay (BWAPI's replay commands).
    #[test]
    fn teacher_commands_label_as_in_botgen_041_r22() {
        let enemy = unit(90, 1, 37, (900, 900)); // a zergling
        let f = frame(vec![
            unit(1, 0, SCV, (100, 100)),
            unit(2, 0, SCV, (120, 100)),
            unit(3, 0, MARINE, (140, 100)),
            unit(4, 0, MARINE, (160, 100)),
            unit(10, 0, CC, (200, 200)),
            unit(11, 0, BARRACKS, (300, 200)),
            unit(12, 0, BUNKER, (400, 300)),
            UnitState { flags: 0, ..unit(13, 0, BARRACKS, (500, 200)) },
            unit(50, 11, MINERAL_FIELDS[0], (64, 64)),
            enemy,
            unit(77, 0, SCV, (1, 1)), // not referenced
        ]);
        let cmds = vec![
            Command::Build { worker: 1, unit_type: BUNKER, tile: (12, 9) },
            Command::Train { building: 10, unit_type: SCV },
            Command::Train { building: 11, unit_type: MARINE },
            Command::Unit { units: vec![2], order: CmdOrder::Move(Target::Pos((3000, 3000))), queued: false },
            Command::Unit { units: vec![3, 4], order: CmdOrder::Load(12), queued: false },
            Command::Unit { units: vec![3, 4], order: CmdOrder::Hold, queued: false },
            Command::Unit { units: vec![3, 4], order: CmdOrder::Attack(Target::Pos((800, 800))), queued: false },
            Command::Unit { units: vec![1, 2], order: CmdOrder::Attack(Target::Unit(90)), queued: false },
            Command::Unit { units: vec![1], order: CmdOrder::Repair(12), queued: false },
            Command::Unit { units: vec![2], order: CmdOrder::Gather(50), queued: false },
            Command::Unit { units: vec![1], order: CmdOrder::RightClick(Target::Unit(13)), queued: false },
            Command::Unit { units: vec![1], order: CmdOrder::ReturnCargo, queued: false },
            // Units the bridge would not find send nothing.
            Command::Unit { units: vec![90, 999], order: CmdOrder::Stop, queued: false },
        ];
        let replay: Vec<Cmd> = cmds.iter().flat_map(|c| command_cmds(c, &f, 0, 3)).collect();
        assert!(replay.iter().all(|c| c.frame == 97 && c.player == 3));
        let lab = Labeler::new().unwrap();
        let units = own_units(&f, 0);
        let (labels, cov) = lab.label(&replay, 200, |_| &units);
        let table = uranium_env::actions::ActionTable::load().unwrap();
        type Row<'a> = (&'a str, Vec<u32>, Option<u32>, Option<(u16, u16)>);
        let acts: Vec<Row> =
            labels.iter().filter(|l| l.frame >= 97 && table.actions[l.action as usize].name != "NoOp").map(|l| (table.actions[l.action as usize].name.as_str(), l.selected.clone(), l.target_unit, l.target_tile)).collect();
        assert_eq!(
            acts,
            vec![
                // Build is an executor action: no selection in the label (031-R1, as in botgen).
                ("Build_Bunker", vec![], None, Some((12, 9))),
                ("Train_SCV", vec![], None, None),
                ("Train_Marine", vec![], None, None),
                ("Move_Pos", vec![2], None, Some((93, 93))),
                ("Smart_Unit", vec![3, 4], Some(12), None),
                ("Hold", vec![3, 4], None, None),
                ("Attack_Pos", vec![3, 4], None, Some((25, 25))),
                ("Attack_Unit", vec![1, 2], Some(90), None),
                ("Repair_Unit", vec![1], Some(12), None),
                ("Gather_Unit", vec![2], Some(50), None),
                ("Smart_Unit", vec![1], Some(13), None),
                ("ReturnCargo", vec![1], None, None),
            ]
        );
        assert_eq!((cov.commands, cov.labeled, cov.unknown), (12, 12, 0));
        // 040-R4: the 12 commands of one observation are consecutive labels, delay 1 up to the last.
        let f97: Vec<(u32, u32)> = labels.iter().filter(|l| (97..109).contains(&l.frame)).map(|l| (l.frame, l.delay)).collect();
        assert_eq!(f97.len(), 12);
        assert!(f97.iter().enumerate().all(|(i, &(f, d))| f == 97 + i as u32 && (d == 1 || i == 11)), "{f97:?}");
    }
}
