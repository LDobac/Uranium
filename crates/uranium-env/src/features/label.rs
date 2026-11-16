use std::collections::{BTreeMap, HashMap};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use uranium_proto::{GameInfo, RawFrame};

use super::extract::{Observation, action_table, extract};
use super::map::{MapInfo, MapPrecompute};
use super::memory::AgentMemory;
use super::schema::{CELLS, DELAY_BUCKETS, MAP, Strategy};
use crate::actions::{CAPABILITIES, LOC_CLASSES, TargetKind};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct Label {
    pub frame: u32,
    pub action: u16,
    pub delay: u32,
    pub queued: bool,
    pub selected: Vec<u32>,
    pub target_unit: Option<u32>,
    pub target_tile: Option<(u16, u16)>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct MemoryAction {
    pub action: u16,
    pub delay_bucket: u32,
    pub units: Vec<u32>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(untagged)]
pub enum DecisionLogLine {
    Decision {
        frame: u32,
        action: u16,
        delay_bucket: u32,
        units: Vec<u32>,
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        forced: bool,
    },
    End { decisions: u64, end_frame: Option<u32>, player: u8 },
}

pub const UNKNOWN_ACTION: u16 = u16::MAX;
pub const UNKNOWN_REASON: &str = "unknown";

impl Label {
    pub fn is_unknown(&self) -> bool {
        self.action == UNKNOWN_ACTION
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostLabel {
    pub action: u16,
    pub delay_bucket: usize,
    pub queued: bool,
    pub selected: Vec<usize>,
    pub target_unit: Option<usize>,
    pub target_loc: Option<usize>,
}

pub const INVALID_LABEL_MAX: f64 = 0.20;

pub fn bucket_index(delay: u32) -> usize {
    DELAY_BUCKETS.iter().rposition(|&b| b <= delay).unwrap_or(0)
}

pub fn host_label(l: &Label, o: &Observation) -> (HostLabel, bool) {
    let (h, why) = host_label_check(l, o);
    (h, why.is_none())
}

pub fn host_label_check(l: &Label, o: &Observation) -> (HostLabel, Option<&'static str>) {
    if l.is_unknown() {
        return (HostLabel { action: 0, delay_bucket: bucket_index(l.delay), queued: false, selected: vec![], target_unit: None, target_loc: None }, Some(UNKNOWN_REASON));
    }
    let index: HashMap<u32, usize> = o.entity_ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
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
    let mut self_only = false;
    if a.uses_select {
        h.selected.retain(|&i| cap(i, &a.select_cap));
        let capable = !h.selected.is_empty();
        if let (TargetKind::Unit, false, Some(t)) = (a.target, a.self_target, h.target_unit) {
            h.selected.retain(|&i| i != t);
        }
        self_only = capable && h.selected.is_empty();
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeadChoice {
    pub action: u16,
    pub delay_bucket: usize,
    pub queued: bool,
    pub selected: Vec<usize>,
    pub target_unit: usize,
    pub target_loc: usize,
}

pub fn label_of_choice(frame: u32, c: &HeadChoice, o: &Observation) -> Result<Label> {
    let a = action_table().actions.get(c.action as usize).ok_or_else(|| anyhow::anyhow!("frame {frame}: action {} is not in the 034 table", c.action))?;
    anyhow::ensure!(c.delay_bucket < DELAY_BUCKETS.len(), "frame {frame}: delay bucket {} of {}", c.delay_bucket, DELAY_BUCKETS.len());
    let n = o.entity_ids.len();
    let id = |i: usize, what: &str| o.entity_ids.get(i).copied().ok_or_else(|| anyhow::anyhow!("frame {frame}: {} {what} entity {i} of {n}", a.name));
    let mut selected: Vec<u32> = Vec::new();
    if a.uses_select {
        for &i in &c.selected {
            let u = id(i, "selected")?;
            if !selected.contains(&u) {
                selected.push(u);
            }
        }
    }
    let target_unit = if a.target == TargetKind::Unit { Some(id(c.target_unit, "target")?) } else { None };
    let target_tile = if a.target == TargetKind::Loc {
        anyhow::ensure!(c.target_loc < CELLS, "frame {frame}: {} location cell {} of {CELLS}", a.name, c.target_loc);
        Some(((c.target_loc % MAP) as u16, (c.target_loc / MAP) as u16))
    } else {
        None
    };
    Ok(Label { frame, action: c.action, delay: DELAY_BUCKETS[c.delay_bucket], queued: c.queued && a.uses_queued, selected, target_unit, target_tile })
}

pub const NOT_IN_TRANSPORT_REASON: &str = "target_not_in_transport";

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

pub fn acting_ids(h: &HostLabel, o: &Observation) -> Vec<u32> {
    h.selected.iter().map(|&i| o.entity_ids[i]).collect()
}

pub struct LabelWalker {
    pub map: MapInfo,
    pub mem: AgentMemory,
}

impl LabelWalker {
    pub fn new(info: &GameInfo, map_family: i32, pre: Option<&MapPrecompute>, enemy_random: bool) -> Result<Self> {
        let mut mem = AgentMemory::with_lobby(info, enemy_random);
        mem.replay_labels = true;
        Ok(Self { map: MapInfo::new(info, map_family, pre)?, mem })
    }

    pub fn step(&mut self, frame: &RawFrame, l: &Label, z: Option<&Strategy>) -> (Observation, HostLabel, Option<&'static str>) {
        self.step_with_memory(frame, l, z, None)
    }

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

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LabelValidity {
    pub labeled: usize,
    pub invalid: usize,
    pub by_reason: BTreeMap<&'static str, usize>,
    pub unknown: usize,
}

impl LabelValidity {
    pub fn ratio(&self) -> f64 {
        if self.labeled == 0 { 0.0 } else { self.invalid as f64 / self.labeled as f64 }
    }

    pub fn rejected(&self) -> bool {
        self.labeled > 0 && self.ratio() >= INVALID_LABEL_MAX
    }

    pub fn filter_reason(&self) -> String {
        format!("{INVALID_LABEL_PREFIX}>={:.0}% ({:.1}%, {}/{}) (040-R19)", INVALID_LABEL_MAX * 100.0, self.ratio() * 100.0, self.invalid, self.labeled)
    }
}

pub const INVALID_LABEL_PREFIX: &str = "invalid_labels";

pub fn label_validity<'a>(info: &GameInfo, map_family: i32, pre: Option<&MapPrecompute>, enemy_random: bool, steps: impl IntoIterator<Item = (&'a RawFrame, &'a Label)>) -> Result<LabelValidity> {
    label_validity_with_memory(info, map_family, pre, enemy_random, steps.into_iter().map(|(f, l)| (f, l, None)))
}

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
