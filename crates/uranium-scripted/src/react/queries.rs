use uranium_proto::*;

use super::Branch;
use crate::brain::Brain;
use crate::types::*;
use crate::util::done;

impl Brain {
    pub(crate) fn react_marines_attack(&self) -> bool {
        self.react.as_ref().is_some_and(|r| r.marines_attack)
    }

    pub(crate) fn react_guard_radius(&self) -> Option<i64> {
        self.react.as_ref().filter(|r| r.marines_attack).map(|r| r.p.marine_radius)
    }

    pub(crate) fn react_reserved_sites(&self) -> Vec<(Tile, u16)> {
        let mut out: Vec<(Tile, u16)> = self.react.as_ref().map(|r| r.main_site.map(|t| (t, BUNKER)).into_iter().chain(r.natural.map(|n| (n, CC))).collect()).unwrap_or_default();
        out.extend(self.tp_failed_sites());
        out
    }

    pub(crate) fn react_branch(&self) -> Option<Branch> {
        self.react.as_ref().map(|r| r.branch)
    }

    pub(crate) fn react_busy(&self) -> Vec<UnitId> {
        let mut out: Vec<UnitId> = self
            .react
            .as_ref()
            .map(|r| r.pulled.iter().chain(&r.repairers).chain(r.harass.iter().flat_map(|h| &h.guards)).copied().chain(r.resumers.values().map(|(scv, _)| *scv)).collect())
            .unwrap_or_default();
        out.extend(self.tp_gas_crew());
        out
    }

    pub(crate) fn react_supply_low(&self, f: &RawFrame, depots: usize) -> Option<bool> {
        let p = &self.react.as_ref()?.p;
        let used = f.self_state.supply_used as u32;
        match depots {
            0 => Some(used >= 2 * p.first_depot_supply),
            1 => Some(used >= 2 * p.second_depot_supply),
            _ => None,
        }
    }

    pub(crate) fn react_production(&self, f: &RawFrame, mine: &[&UnitState], prod_count: usize) -> Option<(usize, bool)> {
        let r = self.react.as_ref()?;
        let rax_done = mine.iter().any(|u| u.unit_type == BARRACKS && done(u));
        let cap = match r.branch {
            _ if self.tp.is_some() => self.tp_barracks_cap(mine),
            Branch::Defence => 2,
            Branch::Macro if self.natural_started(mine) || (r.natural.is_none() && rax_done) => 3,
            Branch::Macro => 1,
        };
        let ready = prod_count > 0 || f.self_state.supply_used as u32 >= 2 * r.p.barracks_supply;
        Some((cap, ready))
    }

    pub(crate) fn natural_started(&self, mine: &[&UnitState]) -> bool {
        let Some(n) = self.react.as_ref().and_then(|r| r.natural) else { return false };
        mine.iter().any(|u| is_resource_depot(u.unit_type) && building_tile(u) == n) || self.pending.iter().any(|p| p.ty == CC && p.tile == n)
    }
}
