use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;
use uranium_proto::*;

use super::{TpParams, TpRule};
use crate::react::Branch;
use crate::util::dist2;

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct TpStats {
    pub main_frame: Option<u32>,
    pub rush_frame: Option<u32>,
    pub rush_rule: Option<TpRule>,
    pub defences: u32,
    pub macro_frame: Option<u32>,
    pub refinery_frame: Option<u32>,
    pub factory_frame: Option<u32>,
    pub shop_frame: Option<u32>,
    pub siege_frame: Option<u32>,
    pub tank_frame: Option<u32>,
    pub expand_frame: Option<u32>,
    pub wave_frame: Option<u32>,
    pub branch: Branch,
    pub contact_frame: Option<u32>,
    pub bunker_frame: Option<u32>,
    pub rule_frames: BTreeMap<TpRule, u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Matchup {
    Protoss,
    Terran,
}

pub(crate) struct TpState {
    pub p: TpParams,
    pub(super) lobby: Race,
    pub(super) seen: Option<Race>,
    pub gas_crew: Vec<UnitId>,
    pub(super) idle_since: BTreeMap<UnitId, u32>,
    pub(super) failed_sites: Vec<(Tile, u16, u32)>,
    pub(super) gas_seen: BTreeMap<UnitId, u32>,
    pub(super) refinery_since: Option<(UnitId, u32)>,
    pub(super) last_gas: Option<u32>,
    pub(super) last_shop: Option<u32>,
    pub(super) last_research: Option<u32>,
    pub(super) last_tanks: Option<u32>,
    pub(super) prod_seen: BTreeSet<(Tile, u16)>,
    pub(super) quiet: BTreeMap<UnitId, u32>,
    pub(super) order_sieging: Option<u16>,
    pub(super) order_unsieging: Option<u16>,
    pub(super) gas_orders: Vec<u16>,
    pub(super) mineral_orders: Vec<u16>,
    pub stats: TpStats,
}

impl TpState {
    pub fn new(p: TpParams, info: &GameInfo) -> Self {
        let lobby = info.players.iter().find(|q| q.is_enemy).map_or(Race::Unknown, |q| q.race);
        let order = uranium_env::types::order_id;
        Self {
            p,
            lobby,
            seen: None,
            gas_crew: Vec::new(),
            idle_since: BTreeMap::new(),
            failed_sites: Vec::new(),
            gas_seen: BTreeMap::new(),
            refinery_since: None,
            last_gas: None,
            last_shop: None,
            last_research: None,
            last_tanks: None,
            prod_seen: BTreeSet::new(),
            quiet: BTreeMap::new(),
            order_sieging: order("Sieging"),
            order_unsieging: order("Unsieging"),
            gas_orders: ["MoveToGas", "WaitForGas", "HarvestGas", "ReturnGas"].into_iter().filter_map(order).collect(),
            mineral_orders: ["MoveToMinerals", "WaitForMinerals", "MiningMinerals", "ReturnMinerals"].into_iter().filter_map(order).collect(),
            stats: TpStats::default(),
        }
    }

    pub fn matchup(&self) -> Matchup {
        match (self.lobby, self.seen) {
            (Race::Protoss, _) | (Race::Random | Race::Unknown, Some(Race::Protoss)) => Matchup::Protoss,
            _ => Matchup::Terran,
        }
    }

    pub(super) fn due(last: Option<u32>, frame: u32, every: u32) -> bool {
        crate::react::ReactState::due_on(true, last, frame, every)
    }
}

pub(crate) fn pick_key(rings: bool, a: Pos, b: Pos) -> i64 {
    if rings { (dist2(a, b) as f64).sqrt() as i64 / 128 } else { dist2(a, b) }
}

pub(crate) fn pick_key_id(by_id: bool, a: Pos, b: Pos) -> i64 {
    if by_id { 0 } else { dist2(a, b) }
}

pub(crate) fn snap(p: Pos) -> Pos {
    (p.0 / 128 * 128 + 64, p.1 / 128 * 128 + 64)
}

pub(super) fn race_of(ty: u16) -> Option<Race> {
    match uranium_env::types::TypeTables::get().unit(ty)?.race.as_str() {
        "Protoss" => Some(Race::Protoss),
        "Terran" => Some(Race::Terran),
        "Zerg" => Some(Race::Zerg),
        _ => None,
    }
}
