use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use uranium_proto::{flags, RawFrame, UnitState};

use crate::features::memory::AgentMemory;
use crate::types::TypeTables;

pub const MIN_FRAME: u32 = 14_400;
pub const HOLD_FRAMES: u32 = 720;
pub const MAX_WIN_PROB: f32 = 0.05;
pub const ARMY_RATIO: i32 = 3;
pub const MAX_WORKERS: usize = 5;
pub const MAX_CHAT_BYTES: usize = uranium_proto::MAX_SURRENDER_CHAT;
pub const DEFAULT_CHAT: &str = "gg";

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SurrenderMode {
    Off,
    Shadow,
    On,
}

pub fn win_prob(win_open: f32) -> f32 {
    (win_open + 1.0) / 2.0
}

#[derive(Debug, Clone, Default)]
pub struct SurrenderJudge {
    low_since: Option<u32>,
    first: Option<u32>,
}

impl SurrenderJudge {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn decide(&mut self, frame: u32, win_open: Option<f32>, rule: impl FnOnce() -> bool) -> bool {
        let Some(v) = win_open else {
            self.low_since = None;
            return false;
        };
        if !(win_prob(v) <= MAX_WIN_PROB) {
            self.low_since = None;
            return false;
        }
        let since = *self.low_since.get_or_insert(frame);
        let holds = frame >= MIN_FRAME && frame - since >= HOLD_FRAMES && rule();
        if holds && self.first.is_none() {
            self.first = Some(frame);
        }
        holds
    }

    pub fn first(&self) -> Option<u32> {
        self.first
    }
}

pub fn rule_holds(frame: &RawFrame, mem: &AgentMemory) -> bool {
    rule_holds_units(&frame.units, mem.self_player, &mem.enemy_max_seen)
}

pub fn rule_holds_units(units: &[UnitState], self_player: u8, enemy_max_seen: &HashMap<u16, u16>) -> bool {
    let tables = TypeTables::get();
    let army = |ty: u16| tables.unit(ty).filter(|t| !t.is_building() && !t.is_worker()).map_or(0, |t| t.supply_required);
    let id = |name: &str| tables.unit_by_name(name).map(|t| t.id as u16);
    let production = [id("Terran_Barracks"), id("Terran_Factory"), id("Terran_Starport")];
    let scv = id("Terran_SCV");

    let mut own_army = 0;
    let mut own_production = 0;
    let mut workers = 0;
    for u in units.iter().filter(|u| u.owner == self_player && u.flags & flags::COMPLETED != 0) {
        own_army += army(u.unit_type);
        own_production += production.contains(&Some(u.unit_type)) as usize;
        workers += (Some(u.unit_type) == scv) as usize;
    }
    let enemy_army: i32 = enemy_max_seen.iter().map(|(&ty, &n)| army(ty) * n as i32).sum();
    let gap = enemy_army > 0 && own_army * ARMY_RATIO <= enemy_army;
    let collapse = own_production == 0 || workers <= MAX_WORKERS;
    gap || collapse
}

#[cfg(test)]
mod tests {
    use super::*;

    const LOW: Option<f32> = Some(-0.95);
    const HIGH: Option<f32> = Some(-0.8);

    fn ty(name: &str) -> u16 {
        TypeTables::get().unit_by_name(name).unwrap().id as u16
    }

    fn unit(owner: u8, name: &str) -> UnitState {
        UnitState { owner, unit_type: ty(name), flags: flags::COMPLETED, ..Default::default() }
    }

    fn healthy() -> Vec<UnitState> {
        let mut v: Vec<UnitState> = (0..6).map(|_| unit(0, "Terran_SCV")).collect();
        v.push(unit(0, "Terran_Barracks"));
        v.extend((0..4).map(|_| unit(0, "Terran_Marine")));
        v
    }

    #[test]
    fn not_before_ten_minutes_031_ac5() {
        let mut j = SurrenderJudge::new();
        assert!(!j.decide(13_000, LOW, || true));
        assert!(!j.decide(MIN_FRAME - 1, LOW, || true));
        assert!(j.decide(MIN_FRAME, LOW, || true));
        assert_eq!(j.first(), Some(MIN_FRAME));
    }

    #[test]
    fn thirty_seconds_in_a_row_031_ac5() {
        let mut j = SurrenderJudge::new();
        let start = 20_000;
        assert!(!j.decide(start, LOW, || true));
        assert!(!j.decide(start + HOLD_FRAMES - 1, LOW, || true));
        assert!(j.decide(start + HOLD_FRAMES, LOW, || true));

        let mut j = SurrenderJudge::new();
        assert!(!j.decide(start, LOW, || true));
        assert!(!j.decide(start + 400, HIGH, || true));
        assert!(!j.decide(start + HOLD_FRAMES, LOW, || true));
        assert!(!j.decide(start + 400 + HOLD_FRAMES, LOW, || true));
        assert!(j.decide(start + 2 * HOLD_FRAMES, LOW, || true));
        assert_eq!(j.first(), Some(start + 2 * HOLD_FRAMES));
    }

    #[test]
    fn rule_and_missing_head_block_031_ac5() {
        let mut j = SurrenderJudge::new();
        assert!(!j.decide(20_000, LOW, || false));
        assert!(!j.decide(21_000, LOW, || false));
        assert!(j.first().is_none());

        let mut j = SurrenderJudge::new();
        assert!(!j.decide(20_000, None, || true));
        assert!(!j.decide(21_000, None, || true));
        assert!(!j.decide(22_000, Some(f32::NAN), || true));
        assert!(j.first().is_none());
    }

    #[test]
    fn rule_army_gap_or_collapse_031_ac5() {
        let marine = ty("Terran_Marine");
        let zergling = ty("Zerg_Zergling");
        let units = healthy();

        assert!(!rule_holds_units(&units, 0, &HashMap::new()));
        assert!(rule_holds_units(&units, 0, &HashMap::from([(marine, 12)])));
        let seen = HashMap::from([(marine, 11), (ty("Terran_Barracks"), 9), (ty("Terran_SCV"), 40)]);
        assert!(!rule_holds_units(&units, 0, &seen));
        assert!(!rule_holds_units(&units, 0, &HashMap::from([(zergling, 23)])));
        assert!(rule_holds_units(&units, 0, &HashMap::from([(zergling, 24)])));

        let no_prod: Vec<_> = units.iter().filter(|u| u.unit_type != ty("Terran_Barracks")).cloned().collect();
        assert!(rule_holds_units(&no_prod, 0, &HashMap::new()));
        let mut building = no_prod.clone();
        building.push(UnitState { flags: flags::BEING_CONSTRUCTED, ..unit(0, "Terran_Barracks") });
        assert!(rule_holds_units(&building, 0, &HashMap::new()));
        let mut lifted = no_prod.clone();
        lifted.push(UnitState { flags: flags::COMPLETED | flags::LIFTED, ..unit(0, "Terran_Barracks") });
        assert!(!rule_holds_units(&lifted, 0, &HashMap::new()));
        let mut few = units.clone();
        few.remove(0);
        assert!(rule_holds_units(&few, 0, &HashMap::new()));
        let mut theirs = few.clone();
        theirs.push(unit(1, "Terran_SCV"));
        assert!(rule_holds_units(&theirs, 0, &HashMap::new()));
    }
}
