//! Replay action labels and their validity against the observation masks (031-R4, 034-AC3,
//! 041-R11, 040-R19). One implementation for the replay build (040-R19 filter) and BC training
//! (label weight, `label-check`), so both count the same labels as invalid.

use std::collections::{BTreeMap, HashMap};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use uranium_proto::{GameInfo, RawFrame};

use super::extract::{Observation, action_table, extract};
use super::map::{MapInfo, MapPrecompute};
use super::memory::AgentMemory;
use super::schema::{CELLS, DELAY_BUCKETS, MAP, Strategy};
use crate::actions::{CAPABILITIES, LOC_CLASSES, TargetKind};

/// One replay action label (040 shard `action_label`): unit ids and tiles, before they are
/// matched to an observation.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct Label {
    /// Frame of the decision (observation frame).
    pub frame: u32,
    pub action: u16,
    /// 031 delay bucket: frames until the next observation.
    pub delay: u32,
    pub queued: bool,
    pub selected: Vec<u32>,
    pub target_unit: Option<u32>,
    /// Build tile (top-left) or position / 32.
    pub target_tile: Option<(u16, u16)>,
}

/// 040-R22 / 041-R28: one decision of the policy that actually played the game (the model of a
/// DAgger game) as its session recorded it in agent memory: the sampled 034 action, the 031 delay
/// bucket and the unit ids of the selection (whether or not the executor refused it, 031-R4). No
/// failure flag: `last_failed` comes from the observation's `CommandFailed` events (030-R10).
/// The model session's decision log line (041-R28 ①) adds the live observation frame.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct MemoryAction {
    pub action: u16,
    pub delay_bucket: u32,
    pub units: Vec<u32>,
}

/// 041-R28 ① (AMB-041-17): one line of a model session's decision log (JSON lines, one file per
/// game, named like the game's replay). A decision line per recorded decision, in order, then one
/// end line written when the game ends; a log without its end line, or whose end line counts
/// other decisions than it holds, is incomplete (041-R28 ④).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(untagged)]
pub enum DecisionLogLine {
    /// `frame`: the live observation frame the decision was made on (`RawFrame.frame`).
    /// `forced` (051-R5 ⑤): a branch player's prefix decision replayed from the trunk's act log;
    /// written only when true, so every other log is byte-identical to before. A forced decision
    /// is still a decision memory recorded (the relabel build uses it like any other).
    Decision {
        frame: u32,
        action: u16,
        delay_bucket: u32,
        units: Vec<u32>,
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        forced: bool,
    },
    /// Written only when the game ended (`GameEnd`): decision lines written, `end_frame` the
    /// final frame the bridge sent (if any) and `player` the session's own BWAPI player id, so the
    /// relabel build applies the log only to that player's view of the same game.
    End { decisions: u64, end_frame: Option<u32>, player: u8 },
}

/// `Label::action` of a game-affecting human command without a 034 action (031-AC3, 2026-10-19
/// user decision, action audit F7): the step is kept with its delay but never learned (reason
/// [`UNKNOWN_REASON`], BC weight 0). Labeler version 2 only; version 1 filled that time with NoOp.
pub const UNKNOWN_ACTION: u16 = u16::MAX;
/// [`host_label_check`] reason of an [`UNKNOWN_ACTION`] label.
pub const UNKNOWN_REASON: &str = "unknown";

impl Label {
    /// 031-AC3: a command without a 034 action (labeler version 2).
    pub fn is_unknown(&self) -> bool {
        self.action == UNKNOWN_ACTION
    }
}

/// Host form of one action label / sample (entity indices into that step's observation).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostLabel {
    pub action: u16,
    pub delay_bucket: usize,
    pub queued: bool,
    /// Entity indices in canonical ascending order (032-R12), without END.
    pub selected: Vec<usize>,
    pub target_unit: Option<usize>,
    pub target_loc: Option<usize>,
}

/// 040-R19 (user decision 2026-10-19): a Terran perspective whose share of invalid labels
/// ([`host_label_check`] fails) is at least this is not stored (`filtered`).
pub const INVALID_LABEL_MAX: f64 = 0.20;

pub fn bucket_index(delay: u32) -> usize {
    DELAY_BUCKETS.iter().rposition(|&b| b <= delay).unwrap_or(0)
}

/// Replay label (unit ids, tile) -> entity indices of this observation, checked against the masks
/// (031-R4). `false` when the label cannot be expressed or violates a mask.
pub fn host_label(l: &Label, o: &Observation) -> (HostLabel, bool) {
    let (h, why) = host_label_check(l, o);
    (h, why.is_none())
}

/// `host_label` with the first failing check (label diagnostics, 034-AC3). A label is invalid
/// (BC weight 0, 040-R19) exactly when this returns a reason.
pub fn host_label_check(l: &Label, o: &Observation) -> (HostLabel, Option<&'static str>) {
    // 031-AC3 (F7): a command without a 034 action is a step of its own that is never learned;
    // the host form is a NoOp with the label's delay (what agent memory records, as the NoOp
    // steps of labeler version 1 did) and the reason makes its BC weight 0.
    if l.is_unknown() {
        return (HostLabel { action: 0, delay_bucket: bucket_index(l.delay), queued: false, selected: vec![], target_unit: None, target_loc: None }, Some(UNKNOWN_REASON));
    }
    let index: HashMap<u32, usize> = o.entity_ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
    // Selection order of the label (the human's selection order) until the max_select cut below.
    let mut selected: Vec<usize> = Vec::with_capacity(l.selected.len());
    for i in l.selected.iter().filter_map(|id| index.get(id).copied()) {
        if !selected.contains(&i) {
            selected.push(i);
        }
    }
    let target_unit = l.target_unit.and_then(|id| index.get(&id).copied());
    let target_loc = l.target_tile.map(|(x, y)| y as usize * MAP + x as usize);
    let mut h = HostLabel { action: l.action, delay_bucket: bucket_index(l.delay), queued: l.queued, selected, target_unit, target_loc };
    let Some(a) = action_table().actions.get(l.action as usize) else {
        h.selected.sort_unstable();
        return (h, Some("unknown_action"));
    };
    let k = CAPABILITIES.len();
    let cap = |i: usize, name: &str| o.masks.unit_cap[i * k + CAPABILITIES.iter().position(|c| *c == name).unwrap()];
    let visible = h.selected.len();
    // 041-R11: the game runs a command on each selected unit that can take it and skips the rest
    // (OpenBW `action_*`), so the label keeps only those; it is void when none can.
    let mut self_only = false;
    if a.uses_select {
        h.selected.retain(|&i| cap(i, &a.select_cap));
        let capable = !h.selected.is_empty();
        // 034-R9: a unit cannot target itself; the game skips it and runs the command on the
        // other selected units (as above), so the label keeps those.
        if let (TargetKind::Unit, false, Some(t)) = (a.target, a.self_target, h.target_unit) {
            h.selected.retain(|&i| i != t);
        }
        self_only = capable && h.selected.is_empty();
        // F12 (2026-10-19 user decision): the label holds the whole human selection (labeler
        // version 2); eligibility is filtered first, then the first `max_select` eligible units
        // in selection order act (a mine order with the first vulture out of mines is laid by the
        // next vulture that has one). Version-1 labels are already cut to `max_select`, so this
        // changes nothing for them.
        h.selected.truncate(a.max_select as usize);
    }
    h.selected.sort_unstable();
    let why = if !o.masks.type_mask[a.id as usize] {
        Some("type_mask")
    } else if a.uses_select && visible == 0 {
        Some("select_empty")
    } else if a.uses_select && visible != l.selected.len() {
        Some("select_not_visible")
    } else if self_only {
        Some("target_selected")
    } else if a.uses_select && h.selected.is_empty() {
        Some("select_cap")
    } else if a.uses_select && h.selected.len() > 1 && h.selected.iter().any(|&i| o.masks.is_building[i]) {
        // 034-R7 / E-4: the game never multi-selects a building (040-R3 ①); such a label cannot
        // come from a valid replay and is void like any other mask violation.
        Some("select_building")
    } else {
        match a.target {
            TargetKind::Unit if h.target_unit.is_none() => Some("target_unit_missing"),
            TargetKind::Unit if !h.target_unit.is_some_and(|i| cap(i, &a.target_cap)) => Some("target_unit_cap"),
            TargetKind::Loc => {
                let lc = LOC_CLASSES.iter().position(|c| *c == a.loc_class).unwrap();
                (!h.target_loc.is_some_and(|t| t < CELLS && o.masks.loc_class_mask[lc * CELLS + t])).then_some("target_loc")
            }
            _ => None,
        }
    };
    (h, why)
}

/// One sampled policy decision as head indices into an observation (044-R3 ②): ActionType, delay
/// bucket, queued, the selection's entity indices in pick order (without END) and the TargetUnit /
/// Location head outputs (read only when the action uses them).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeadChoice {
    pub action: u16,
    pub delay_bucket: usize,
    pub queued: bool,
    pub selected: Vec<usize>,
    pub target_unit: usize,
    pub target_loc: usize,
}

/// 044-R3 ③: a sampled decision on observation `o` (frame `frame`) as a replay label (040
/// `action_label`), the inverse of [`host_label_check`]: the delay is the bucket's lower bound
/// (031-R11, [`bucket_index`] maps it back to the bucket), the selection the picked units' ids in
/// pick order, the target unit its id, the location cell `y * MAP + x` the tile (x, y) whose
/// centre the executor commands (unaugmented: the caller reads without symmetry). Only the parts
/// the 034 action uses are kept (`queued` without `uses_queued` is false). Indices past the
/// observation's entities (padding) are dropped; [`host_label_check`] then reports the label.
pub fn label_of_choice(frame: u32, c: &HeadChoice, o: &Observation) -> Label {
    let a = action_table().actions.get(c.action as usize);
    let id = |i: usize| o.entity_ids.get(i).copied();
    let mut selected: Vec<u32> = Vec::new();
    if a.is_some_and(|a| a.uses_select) {
        for u in c.selected.iter().filter_map(|&i| id(i)) {
            if !selected.contains(&u) {
                selected.push(u);
            }
        }
    }
    let target = a.map_or(TargetKind::None, |a| a.target);
    Label {
        frame,
        action: c.action,
        delay: DELAY_BUCKETS[c.delay_bucket.min(DELAY_BUCKETS.len() - 1)],
        queued: c.queued && a.is_some_and(|a| a.uses_queued),
        selected,
        target_unit: if target == TargetKind::Unit { id(c.target_unit) } else { None },
        target_tile: (target == TargetKind::Loc && c.target_loc < CELLS).then_some(((c.target_loc % MAP) as u16, (c.target_loc / MAP) as u16)),
    }
}

/// [`unload_target_check`] reason: the `Unload_Unit` target is not loaded in the selected transport.
pub const NOT_IN_TRANSPORT_REASON: &str = "target_not_in_transport";

/// `Unload_Unit` (F7, 2026-10-19): the target must be loaded in the selected transport (BWAPI
/// `canUnload`, the executor rejects it otherwise). The masks only know "loaded", so the check
/// reads the frame's `loaded_in`; part of the BC label check through [`LabelWalker::step`].
pub fn unload_target_check(h: &HostLabel, o: &Observation, frame: &RawFrame) -> Option<&'static str> {
    let a = action_table().actions.get(h.action as usize)?;
    if a.command != "UnitUnload" {
        return None;
    }
    let (Some(t), Some(&s)) = (h.target_unit, h.selected.first()) else { return None };
    let (target, transport) = (o.entity_ids[t], o.entity_ids[s]);
    let inside = frame.units.iter().any(|u| u.id == target && u.loaded_in == Some(transport));
    (!inside).then_some(NOT_IN_TRANSPORT_REASON)
}

/// Unit ids of the label's (capability-filtered) selection.
pub fn acting_ids(h: &HostLabel, o: &Observation) -> Vec<u32> {
    h.selected.iter().map(|&i| o.entity_ids[i]).collect()
}

/// Feature extraction over one game's labeled steps in order, with agent memory carried across
/// steps and fed the label's action (030 §4, 041-R11), as BC training sees them.
pub struct LabelWalker {
    pub map: MapInfo,
    pub mem: AgentMemory,
}

impl LabelWalker {
    /// `map_family`: 040-R15 index (`maps::family_index`); `pre`: the map precompute (031);
    /// `enemy_random`: shard `enemy_random` (030-R11).
    pub fn new(info: &GameInfo, map_family: i32, pre: Option<&MapPrecompute>, enemy_random: bool) -> Result<Self> {
        let mut mem = AgentMemory::with_lobby(info, enemy_random);
        // U-4 (user decision 2026-10-19): replay labels use the engine rule for siege transitions.
        mem.replay_labels = true;
        Ok(Self { map: MapInfo::new(info, map_family, pre)?, mem })
    }

    /// Observation of `frame`, the label as entity indices and why it is invalid (`None` = valid).
    pub fn step(&mut self, frame: &RawFrame, l: &Label, z: Option<&Strategy>) -> (Observation, HostLabel, Option<&'static str>) {
        self.step_with_memory(frame, l, z, None)
    }

    /// [`LabelWalker::step`] with the step's 040-R22 `memory_action` (041-R28 ③). `Some` (an
    /// empty list included): those decisions go into agent memory in order **before** the
    /// observation is built and the label is not recorded (it stays the training target only).
    /// `None`: the label is recorded after the observation, as before.
    pub fn step_with_memory(&mut self, frame: &RawFrame, l: &Label, z: Option<&Strategy>, memory: Option<&[MemoryAction]>) -> (Observation, HostLabel, Option<&'static str>) {
        for a in memory.unwrap_or_default() {
            self.mem.record_action(a.action, a.delay_bucket as usize, &a.units);
        }
        let o = extract(frame, &mut self.mem, &self.map, z);
        let (label, mut why) = host_label_check(l, &o);
        if why.is_none() {
            why = unload_target_check(&label, &o, frame);
        }
        if memory.is_none() {
            self.mem.record_action(label.action, label.delay_bucket, &acting_ids(&label, &o));
        }
        (o, label, why)
    }
}

/// Invalid labels of one perspective (040-R19): every labeled step counts, invalid = the
/// [`host_label_check`] reason is set.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LabelValidity {
    pub labeled: usize,
    pub invalid: usize,
    pub by_reason: BTreeMap<&'static str, usize>,
    /// 031-AC3 `unknown` steps (labeler version 2): weight 0 in BC but neither labeled nor
    /// invalid here, so the 040-R19 ratio means the same for labeler versions 1 and 2.
    pub unknown: usize,
}

impl LabelValidity {
    pub fn ratio(&self) -> f64 {
        if self.labeled == 0 { 0.0 } else { self.invalid as f64 / self.labeled as f64 }
    }

    /// 040-R19: the perspective is rejected (ratio ≥ [`INVALID_LABEL_MAX`]).
    pub fn rejected(&self) -> bool {
        self.labeled > 0 && self.ratio() >= INVALID_LABEL_MAX
    }

    /// `filtered` reason of a rejected perspective (040-R19).
    pub fn filter_reason(&self) -> String {
        format!("{INVALID_LABEL_PREFIX}>={:.0}% ({:.1}%, {}/{}) (040-R19)", INVALID_LABEL_MAX * 100.0, self.ratio() * 100.0, self.invalid, self.labeled)
    }
}

/// Start of the 040-R19 `filtered` reason (statistics group by it).
pub const INVALID_LABEL_PREFIX: &str = "invalid_labels";

/// [`LabelValidity`] of one game's steps (040-R19), with the same extraction and check as BC.
pub fn label_validity<'a>(info: &GameInfo, map_family: i32, pre: Option<&MapPrecompute>, enemy_random: bool, steps: impl IntoIterator<Item = (&'a RawFrame, &'a Label)>) -> Result<LabelValidity> {
    label_validity_with_memory(info, map_family, pre, enemy_random, steps.into_iter().map(|(f, l)| (f, l, None)))
}

/// [`label_validity`] with each step's 040-R22 `memory_action`, read the way BC reads it (041-R28 ③).
pub fn label_validity_with_memory<'a>(info: &GameInfo, map_family: i32, pre: Option<&MapPrecompute>, enemy_random: bool, steps: impl IntoIterator<Item = (&'a RawFrame, &'a Label, Option<&'a [MemoryAction]>)>) -> Result<LabelValidity> {
    let mut w = LabelWalker::new(info, map_family, pre, enemy_random)?;
    let mut v = LabelValidity::default();
    for (frame, l, m) in steps {
        let (_, _, why) = w.step_with_memory(frame, l, None, m);
        if why == Some(UNKNOWN_REASON) {
            v.unknown += 1;
            continue;
        }
        v.labeled += 1;
        if let Some(r) = why {
            v.invalid += 1;
            *v.by_reason.entry(r).or_default() += 1;
        }
    }
    Ok(v)
}
