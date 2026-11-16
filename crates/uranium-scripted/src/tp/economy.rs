use std::collections::BTreeSet;

use uranium_proto::*;

use super::{Matchup, TpState, TpStats, pick_key};
use crate::brain::Brain;
use crate::react::{Branch, within};
use crate::types::*;
use crate::util::done;

impl Brain {
    pub(crate) fn tp_gas_crew(&self) -> Vec<UnitId> {
        self.tp.as_ref().map(|t| t.gas_crew.clone()).unwrap_or_default()
    }

    fn natural_done(&self, mine: &[&UnitState]) -> bool {
        let Some(n) = self.react.as_ref().and_then(|r| r.natural) else { return false };
        mine.iter().any(|u| is_resource_depot(u.unit_type) && building_tile(u) == n && done(u))
    }

    pub(crate) fn tp_idle(&mut self, f: &RawFrame, mine: &[&UnitState]) {
        let Some(t) = self.tp.as_mut() else { return };
        let idle: Vec<UnitId> = mine.iter().filter(|u| u.unit_type == SCV && u.has(flags::COMPLETED) && u.has(flags::IDLE) && u.loaded_in.is_none()).map(|u| u.id).collect();
        t.idle_since.retain(|id, _| idle.contains(id));
        for id in idle {
            t.idle_since.entry(id).or_insert(f.frame);
        }
        let grace = t.p.idle_grace;
        let since = t.idle_since.clone();
        t.failed_sites.retain(|(_, _, until)| f.frame < *until);
        let mut failed = Vec::new();
        self.pending.retain(|q| {
            let idle = since.get(&q.worker).is_some_and(|s| f.frame >= s + grace);
            if idle {
                failed.push((q.tile, q.ty, f.frame + crate::brain::PENDING_TIMEOUT));
            }
            !idle
        });
        self.tp.as_mut().expect("checked above").failed_sites.extend(failed);
    }

    pub(crate) fn tp_failed_sites(&self) -> Vec<(Tile, u16)> {
        self.tp.as_ref().map(|t| t.failed_sites.iter().map(|(tile, ty, _)| (*tile, *ty)).collect()).unwrap_or_default()
    }

    pub(crate) fn tp_idle_long(&self, id: UnitId, frame: u32) -> bool {
        self.tp.as_ref().is_none_or(|t| t.idle_since.get(&id).is_some_and(|s| frame >= s + t.p.idle_grace))
    }

    pub(crate) fn tp_gas(&mut self, f: &RawFrame, mine: &[&UnitState], cmds: &mut Vec<Command>) {
        let Some(t) = self.tp.as_ref() else { return };
        let (want, every, last) = (t.p.gas_workers, t.p.retarget, t.last_gas);
        let refinery = mine.iter().filter(|u| u.unit_type == REFINERY && done(u)).min_by_key(|u| u.id).copied();
        let on_gas: Vec<UnitId> = mine
            .iter()
            .filter(|u| u.unit_type == SCV && (t.gas_orders.contains(&u.order) || refinery.is_some_and(|r| u.order_target == Some(r.id))))
            .map(|u| u.id)
            .collect();
        let t = self.tp.as_mut().expect("checked above");
        for id in &on_gas {
            t.gas_seen.insert(*id, f.frame);
        }
        let minerals: Vec<UnitId> = mine.iter().filter(|u| u.unit_type == SCV && t.mineral_orders.contains(&u.order)).map(|u| u.id).collect();
        t.gas_seen.retain(|id, at| on_gas.contains(id) || (f.frame < *at + 2 * every && !minerals.contains(id)));
        let crew: Vec<UnitId> = t.gas_seen.keys().copied().collect();
        t.gas_crew = crew.clone();
        t.refinery_since = match (refinery, t.refinery_since) {
            (Some(r), Some((id, at))) if r.id == id => Some((id, at)),
            (Some(r), _) => Some((r.id, f.frame)),
            (None, _) => None,
        };
        let Some(r) = refinery else { return };
        let settled = t.refinery_since.is_some_and(|(_, at)| f.frame >= at + every);
        if crew.len() >= want || !settled || !TpState::due(last, f.frame, every) {
            return;
        }
        let jobs: BTreeSet<UnitId> = self.pending.iter().map(|q| q.worker).chain(self.scout).chain(self.react_busy()).collect();
        let mut free: Vec<&&UnitState> = mine
            .iter()
            .filter(|u| u.unit_type == SCV && u.has(flags::COMPLETED) && !u.has(flags::CARRYING) && !u.has(flags::CONSTRUCTING) && !jobs.contains(&u.id))
            .collect();
        free.sort_by_key(|u| (pick_key(true, u.pos, r.pos), u.id));
        let ids: Vec<UnitId> = free.iter().take(want - crew.len()).map(|u| u.id).collect();
        if ids.is_empty() {
            return;
        }
        cmds.push(Command::Unit { units: ids.clone(), order: CmdOrder::Gather(r.id), queued: false });
        let t = self.tp.as_mut().expect("checked above");
        t.gas_crew.extend(ids);
        t.last_gas = Some(f.frame);
    }

    fn tp_order_build(&mut self, ty: u16, f: &RawFrame, mine: &[&UnitState], base: Pos, busy: &[UnitId], cmds: &mut Vec<Command>) -> Option<bool> {
        let reserved: Vec<(Tile, u16)> = self.pending.iter().map(|q| (q.tile, q.ty)).chain(self.react_reserved_sites()).collect();
        let tile = self.place(ty, f, mine, base, &reserved)?;
        Some(self.order_build_at(ty, tile, f, mine, busy, cmds))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn tp_build(&mut self, f: &RawFrame, mine: &[&UnitState], base: Pos, busy: &[UnitId], cmds: &mut Vec<Command>, minerals: &mut i32, gas: &mut i32) -> (i32, i32) {
        let Some(t) = self.tp.as_ref() else { return (0, 0) };
        let (p, last_shop, last_research) = (t.p.clone(), t.last_shop, t.last_research);
        let matchup = t.matchup();
        let (mut hold_all, mut hold_army) = (0, 0);
        let has = |s: &Self, ty: u16| mine.iter().any(|u| u.unit_type == ty) || s.pending.iter().any(|q| q.ty == ty);
        let rax_done = mine.iter().any(|u| u.unit_type == BARRACKS && done(u));
        let supply = u32::from(f.self_state.supply_used);
        fn stats(s: &mut Brain) -> &mut TpStats {
            &mut s.tp.as_mut().expect("the 022 teacher").stats
        }

        if !has(self, REFINERY) && has(self, BARRACKS) && supply >= 2 * p.refinery_supply {
            if *minerals >= cost(REFINERY) {
                if self.tp_order_build(REFINERY, f, mine, base, busy, cmds) == Some(true) {
                    *minerals -= cost(REFINERY);
                    stats(self).refinery_frame.get_or_insert(f.frame);
                }
            } else if self.place(REFINERY, f, mine, base, &[]).is_some() {
                hold_all += cost(REFINERY);
            }
        }
        if !has(self, FACTORY) && rax_done && *gas >= p.factory_gas {
            if *minerals >= cost(FACTORY) {
                if self.tp_order_build(FACTORY, f, mine, base, busy, cmds) == Some(true) {
                    *minerals -= cost(FACTORY);
                    *gas -= gas_cost(FACTORY);
                    stats(self).factory_frame.get_or_insert(f.frame);
                }
            } else {
                hold_all += cost(FACTORY);
            }
        }
        let factory = mine.iter().filter(|u| u.unit_type == FACTORY && done(u) && !u.has(flags::LIFTED)).min_by_key(|u| u.id).copied();
        let shop = mine.iter().any(|u| u.unit_type == MACHINE_SHOP);
        if let Some(fa) = factory.filter(|fa| !fa.has_addon && !shop && !fa.has(flags::TRAINING) && TpState::due(last_shop, f.frame, p.retarget)) {
            if *minerals >= cost(MACHINE_SHOP) && *gas >= gas_cost(MACHINE_SHOP) {
                cmds.push(Command::BuildAddon { building: fa.id, unit_type: MACHINE_SHOP });
                *minerals -= cost(MACHINE_SHOP);
                *gas -= gas_cost(MACHINE_SHOP);
                let t = self.tp.as_mut().expect("checked above");
                t.last_shop = Some(f.frame);
                t.stats.shop_frame.get_or_insert(f.frame);
            } else {
                hold_all += cost(MACHINE_SHOP);
            }
        }
        let bit = 1u64 << TECH_TANK_SIEGE;
        let research_started = (f.self_state.tech | f.self_state.researching) & bit != 0;
        let shop_done = mine.iter().filter(|u| u.unit_type == MACHINE_SHOP && done(u)).min_by_key(|u| u.id).copied();
        if let Some(sh) = shop_done.filter(|_| !research_started && TpState::due(last_research, f.frame, p.retarget)) {
            if *minerals >= 150 && *gas >= 150 {
                cmds.push(Command::Research { building: sh.id, tech: TECH_TANK_SIEGE });
                *minerals -= 150;
                *gas -= 150;
                let t = self.tp.as_mut().expect("checked above");
                t.last_research = Some(f.frame);
                t.stats.siege_frame.get_or_insert(f.frame);
            } else {
                hold_all += 150;
            }
        }
        let r = self.react.as_ref().expect("the 022 teacher reuses the 021-R8 state");
        let (branch, natural, last_contact) = (r.branch, r.natural, r.last_contact);
        if let Some(n) = natural.filter(|_| !self.natural_started(mine)) {
            let tanks = Self::tanks_done(mine);
            let ready = match (branch, matchup) {
                (Branch::Macro, Matchup::Protoss) => has(self, FACTORY) && tanks >= p.expand_tanks_p,
                (Branch::Macro, Matchup::Terran) => tanks >= p.expand_tanks,
                (Branch::Defence, _) => tanks >= p.expand_tanks && f.frame >= last_contact + p.quiet,
            };
            if ready {
                if *minerals >= p.expand_minerals && self.order_build_at(CC, n, f, mine, busy, cmds) {
                    *minerals -= cost(CC);
                    stats(self).expand_frame.get_or_insert(f.frame);
                } else if branch == Branch::Macro && matchup == Matchup::Protoss && f.frame >= p.expand_by {
                    hold_all += p.expand_minerals;
                } else {
                    hold_army += p.expand_minerals;
                }
            }
        }
        (hold_all, hold_army)
    }

    pub(crate) fn tp_train(&mut self, f: &RawFrame, mine: &[&UnitState], minerals: &mut i32, gas: &mut i32, supply_free: &mut i32, cmds: &mut Vec<Command>) -> i32 {
        if self.tp.is_none() || (f.self_state.tech | f.self_state.researching) & (1u64 << TECH_TANK_SIEGE) == 0 {
            return 0;
        }
        let shops: Vec<Pos> = mine.iter().filter(|u| u.unit_type == MACHINE_SHOP && done(u)).map(|u| u.pos).collect();
        let mut factories: Vec<&&UnitState> = mine
            .iter()
            .filter(|u| u.unit_type == FACTORY && done(u) && u.has_addon && !u.has(flags::TRAINING) && !u.has(flags::LIFTED))
            .filter(|u| shops.iter().any(|s| within(*s, u.pos, 4)))
            .collect();
        factories.sort_by_key(|u| u.id);
        for fa in factories {
            if *minerals >= cost(SIEGE_TANK) && *gas >= gas_cost(SIEGE_TANK) && *supply_free >= supply_cost(SIEGE_TANK) {
                cmds.push(Command::Train { building: fa.id, unit_type: SIEGE_TANK });
                *minerals -= cost(SIEGE_TANK);
                *gas -= gas_cost(SIEGE_TANK);
                *supply_free -= supply_cost(SIEGE_TANK);
                self.tp.as_mut().expect("checked above").stats.tank_frame.get_or_insert(f.frame);
            } else if *gas >= gas_cost(SIEGE_TANK) && *supply_free >= supply_cost(SIEGE_TANK) {
                return cost(SIEGE_TANK);
            }
        }
        0
    }

    pub(crate) fn tp_marine_cap(&self, mine: &[&UnitState]) -> Option<usize> {
        let t = self.tp.as_ref()?;
        let factory = mine.iter().any(|u| u.unit_type == FACTORY) || self.pending.iter().any(|q| q.ty == FACTORY);
        Some(if self.react_branch() == Some(Branch::Macro) && !factory { t.p.marines_before_factory } else { usize::MAX })
    }

    pub(crate) fn tp_max_workers(&self, mine: &[&UnitState]) -> Option<usize> {
        let t = self.tp.as_ref()?;
        let ccs = mine.iter().filter(|u| is_resource_depot(u.unit_type) && done(u)).count().max(1);
        Some((t.p.scv_per_cc * ccs).min(t.p.scv_max))
    }

    pub(crate) fn tp_barracks_cap(&self, mine: &[&UnitState]) -> usize {
        let Some(t) = self.tp.as_ref() else { return 1 };
        let macro_cap = match (self.natural_done(mine), t.matchup()) {
            (false, _) => 1,
            (true, Matchup::Protoss) => t.p.barracks_p,
            (true, Matchup::Terran) => t.p.barracks_t,
        };
        match self.react_branch() {
            Some(Branch::Defence) => macro_cap.max(t.p.barracks_d),
            _ => macro_cap,
        }
    }
}
