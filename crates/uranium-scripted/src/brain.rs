mod build;
mod memory;
mod tactics;

use std::collections::BTreeMap;

use uranium_proto::*;

use crate::Bot;
use crate::kit::{Kit, Supply, Tactic};
use crate::placement::Placer;
use crate::react::{self, Branch, MapHints, ReactState, ReactStats};
use crate::tp::{TpState, TpStats};
use crate::types::*;
use crate::util::{centroid, done, nearest};

pub(crate) const STEP_FRAMES: u32 = 8;
pub(crate) const WAVE_PERIOD: u32 = 48;
pub(crate) const PENDING_TIMEOUT: u32 = 24 * 25;
pub(crate) const CELL: u16 = 8;
pub(crate) const DROP_SIZE: usize = 8;
pub(crate) const LOAD_TIMEOUT: u32 = 24 * 20;
pub(crate) const DEFEND_RADIUS: i64 = 20 * 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DropPhase {
    Load { since: u32 },
    Fly,
    Fight,
}

struct Drop {
    ship: UnitId,
    squad: Vec<UnitId>,
    phase: DropPhase,
    last_order: u32,
}

pub(crate) struct Pending {
    pub(crate) ty: u16,
    pub(crate) tile: Tile,
    pub(crate) worker: UnitId,
    pub(crate) issued: u32,
}

pub struct Brain {
    kit: Kit,
    pub(crate) me: u8,
    pub(crate) starts: Vec<Tile>,
    pub(crate) walkable: BitGrid,
    pub(crate) placer: Placer,
    pub(crate) pending: Vec<Pending>,
    last_supply_order: u32,
    pub(crate) enemy_buildings: BTreeMap<UnitId, (Pos, u16)>,
    pub(crate) scouted: Vec<bool>,
    pub(crate) cell_seen: Vec<u32>,
    pub(crate) cells_w: u16,
    pub(crate) cells_h: u16,
    last_wave: u32,
    gas_crew: Vec<UnitId>,
    pub(crate) scout: Option<UnitId>,
    pub(crate) scout_done: bool,
    pub(crate) scout_order: u32,
    drops: Vec<Drop>,
    last_defend: u32,
    pub(crate) home: Option<Pos>,
    pub(crate) map_center: Pos,
    pub(crate) react: Option<ReactState>,
    pub(crate) tp: Option<TpState>,
}

impl Brain {
    pub fn new(info: &GameInfo, kit: Kit) -> Self {
        Self::with_hints(info, kit, &MapHints::default())
    }

    pub fn with_hints(info: &GameInfo, kit: Kit, hints: &MapHints) -> Self {
        let me = info.self_player;
        let my_start = info.players.iter().find(|p| p.is_self).and_then(|p| p.start_tile);
        let starts = info.static_map.start_locations.clone();
        let scouted = starts.iter().map(|s| Some(*s) == my_start).collect();
        let (w, h) = info.map_size_tiles;
        let (cells_w, cells_h) = (w.div_ceil(CELL), h.div_ceil(CELL));
        let home = my_start.map(react::depot_center);
        let react = kit.react.clone().map(|p| ReactState::new(p, hints, home, &starts));
        let tp = kit.tp.clone().map(|p| TpState::new(p, info));
        Self {
            tp,
            home,
            map_center: (w * 16, h * 16),
            react,
            kit,
            me,
            starts,
            walkable: info.static_map.walkable.clone(),
            placer: Placer::new(&info.static_map),
            pending: Vec::new(),
            last_supply_order: 0,
            enemy_buildings: BTreeMap::new(),
            scouted,
            cell_seen: vec![0; cells_w as usize * cells_h as usize],
            cells_w,
            cells_h,
            last_wave: 0,
            gas_crew: Vec::new(),
            scout: None,
            scout_done: false,
            scout_order: 0,
            drops: Vec::new(),
            last_defend: 0,
        }
    }
}

impl Bot for Brain {
    fn act(&mut self, f: &RawFrame) -> Act {
        let mut cmds = Vec::new();
        let k = self.kit.clone();
        self.update_memory(f);
        let mine: Vec<&UnitState> = f.units.iter().filter(|u| u.owner == self.me).collect();
        let count = |ty: u16| mine.iter().filter(|u| u.unit_type == ty).count();
        let minerals_vis: Vec<&UnitState> = f.units.iter().filter(|u| is_mineral(u.unit_type)).collect();
        self.pending.retain(|p| f.frame < p.issued + PENDING_TIMEOUT && !mine.iter().any(|u| u.unit_type == p.ty && building_tile(u) == p.tile));
        if let Some(rp) = k.react.as_ref() {
            self.pending.retain(|p| !react::bunker_order_stale(rp, p, f, &mine));
        }
        self.tp_idle(f, &mine);

        let mut minerals = f.self_state.minerals - self.pending.iter().map(|p| cost(p.ty)).sum::<i32>();
        let mut gas = f.self_state.gas - self.pending.iter().map(|p| gas_cost(p.ty)).sum::<i32>();
        let mut supply_free = f.self_state.supply_total as i32 - f.self_state.supply_used as i32;
        let base = if k.react.is_some() { self.home_depot(&mine) } else { mine.iter().find(|u| is_resource_depot(u.unit_type)).map(|u| u.pos) };
        let mut larvae: Vec<&UnitState> = mine.iter().copied().filter(|u| u.unit_type == LARVA).collect();
        let prod_count = count(k.production) + self.pending.iter().filter(|p| p.ty == k.production).count();
        if self.tp.is_some() {
            self.tp_observe(f, &mine);
        } else {
            self.react_observe(f, &mine);
        }
        let branch = self.react_branch();

        self.gas_crew.retain(|id| mine.iter().any(|u| u.id == *id));
        if k.react.is_none() && self.scout.is_some_and(|id| !mine.iter().any(|u| u.id == id)) {
            self.scout = None;
            self.scout_done = true;
        }
        let refinery = mine.iter().find(|u| is_refinery(u.unit_type) && done(u)).map(|u| u.id);
        if let Some(r) = refinery {
            while self.gas_crew.len() < k.gas_workers {
                let Some(w) = self.free_worker(&mine) else { break };
                cmds.push(Command::Unit { units: vec![w], order: CmdOrder::Gather(r), queued: false });
                self.gas_crew.push(w);
            }
        }
        self.tp_gas(f, &mine, &mut cmds);
        if k.react.is_some() {
            self.react_scouting(f, &mine, &minerals_vis, &mut cmds);
            self.react_scvs(f, &mine, &minerals_vis, &mut cmds);
        } else {
            if k.scout && !self.scout_done && self.scout.is_none() && prod_count > 0 {
                self.scout = self.free_worker(&mine);
            }
            if let Some(id) = self.scout {
                self.scouting(f, &mine, id, &minerals_vis, &mut cmds);
            }
        }
        let busy: Vec<UnitId> = self.pending.iter().map(|p| p.worker).chain(self.gas_crew.iter().copied()).chain(self.scout).chain(self.react_busy()).collect();

        for w in mine.iter().filter(|u| u.unit_type == k.worker && u.has(flags::COMPLETED) && u.has(flags::IDLE)) {
            let on_gas = self.gas_crew.contains(&w.id);
            if busy.contains(&w.id) && !on_gas {
                continue;
            }
            if !self.tp_idle_long(w.id, f.frame) {
                continue;
            }
            if w.has(flags::CARRYING) {
                cmds.push(Command::Unit { units: vec![w.id], order: CmdOrder::ReturnCargo, queued: false });
            } else if let (true, Some(r)) = (on_gas, refinery) {
                cmds.push(Command::Unit { units: vec![w.id], order: CmdOrder::Gather(r), queued: false });
            } else if let Some(m) = nearest(w.pos, &minerals_vis) {
                cmds.push(Command::Unit { units: vec![w.id], order: CmdOrder::Gather(m.id), queued: false });
            }
        }

        let supply_prod = prod_count + if self.tp.is_some() { count(FACTORY) } else { 0 };
        let supply_low = f.self_state.supply_total < 400 && supply_free <= 4 + 4 * supply_prod as i32;
        match k.supply {
            Supply::Building(ty) => {
                let in_progress = mine.iter().filter(|u| u.unit_type == ty && !u.has(flags::COMPLETED)).count() + self.pending.iter().filter(|p| p.ty == ty).count();
                let depots = count(ty) + self.pending.iter().filter(|p| p.ty == ty).count();
                let supply_low = self.react_supply_low(f, depots).unwrap_or(supply_low);
                if supply_low && in_progress == 0 && minerals >= cost(ty) {
                    if let Some(b) = base {
                        if self.order_build(ty, f, &mine, b, &busy, &mut cmds) {
                            minerals -= cost(ty);
                        }
                    }
                }
            }
            Supply::Larva(ty) => {
                if supply_low && f.frame >= self.last_supply_order + 24 * 16 && minerals >= cost(ty) {
                    if let Some(l) = larvae.pop() {
                        cmds.push(Command::Train { building: l.id, unit_type: ty });
                        self.last_supply_order = f.frame;
                        minerals -= cost(ty);
                    }
                }
            }
        }

        let (mut hold_all, mut hold_army) = match base {
            Some(b) => self.react_build(f, &mine, b, &busy, &mut cmds, &mut minerals),
            None => (0, 0),
        };
        if let Some(b) = base.filter(|_| self.tp.is_some()) {
            let (all, army) = self.tp_build(f, &mine, b, &busy, &mut cmds, &mut minerals, &mut gas);
            hold_all += all;
            hold_army += army;
        }

        let workers = count(k.worker);
        let (cap, ready) = self.react_production(f, &mine, prod_count).unwrap_or((k.production_cap, workers >= k.production_at_workers[prod_count.min(2)]));
        let budget = if prod_count > 0 { minerals - hold_all } else { minerals };
        let mut prod_after = prod_count;
        if prod_count < cap && ready && budget >= cost(k.production) {
            if let Some(b) = base {
                if self.order_build(k.production, f, &mine, b, &busy, &mut cmds) {
                    minerals -= cost(k.production);
                    prod_after += 1;
                }
            }
        }
        let hold_workers = if branch == Some(Branch::Defence) && prod_after < cap { cost(k.production) } else { 0 };

        let prod_ready = mine.iter().any(|u| u.unit_type == k.production && u.has(flags::COMPLETED));
        let mut tech_done = true;
        for &ty in &k.tech {
            if mine.iter().any(|u| u.unit_type == ty && done(u)) {
                continue;
            }
            tech_done = false;
            let started = mine.iter().any(|u| u.unit_type == ty) || self.pending.iter().any(|p| p.ty == ty);
            if prod_ready && !started {
                if minerals >= cost(ty) && gas >= gas_cost(ty) && base.is_some_and(|b| self.order_tech(ty, f, &mine, b, &busy, &mut cmds)) {
                    gas -= gas_cost(ty);
                }
                minerals -= cost(ty);
            }
            break;
        }

        let tech_ready = tech_done || k.tech.iter().filter(|t| !is_refinery(**t)).all(|&t| mine.iter().any(|u| u.unit_type == t && done(u)));
        let special = k.special.filter(|_| tech_ready);
        let mut army_n = count(k.army);
        let army_cap = self.tp_marine_cap(&mine).unwrap_or(k.army_cap);
        let max_workers = self.tp_max_workers(&mine).unwrap_or(k.max_workers);
        minerals -= hold_all;
        match k.supply {
            Supply::Larva(_) => {
                while let Some(l) = larvae.pop() {
                    if let Some((LARVA, sp, cap)) = special {
                        if count(sp) < cap && minerals >= cost(sp) && gas >= gas_cost(sp) && supply_free >= supply_cost(sp) {
                            cmds.push(Command::Train { building: l.id, unit_type: sp });
                            minerals -= cost(sp);
                            gas -= gas_cost(sp);
                            supply_free -= supply_cost(sp);
                            continue;
                        }
                    }
                    let ty = if prod_ready && workers >= k.max_workers.min(9) && army_n < k.army_cap {
                        k.army
                    } else if workers < k.max_workers {
                        k.worker
                    } else {
                        break;
                    };
                    if minerals < cost(ty) || supply_free < 2 {
                        break;
                    }
                    cmds.push(Command::Train { building: l.id, unit_type: ty });
                    minerals -= cost(ty);
                    supply_free -= 2;
                    if ty == k.army {
                        army_n += 1;
                    }
                }
            }
            Supply::Building(_) => {
                let mut workers = workers;
                for d in mine.iter().filter(|u| is_resource_depot(u.unit_type) && u.has(flags::COMPLETED) && !u.has(flags::TRAINING) && !u.has(flags::LIFTED)) {
                    if workers < max_workers && minerals - hold_workers >= 50 && supply_free >= 2 {
                        cmds.push(Command::Train { building: d.id, unit_type: k.worker });
                        minerals -= 50;
                        supply_free -= 2;
                        if self.tp.is_some() {
                            workers += 1;
                        }
                    }
                }
                hold_army += self.tp_train(f, &mine, &mut minerals, &mut gas, &mut supply_free, &mut cmds);
                let mut used = Vec::new();
                if let Some((producer, sp, cap)) = special.filter(|s| s.0 != LARVA) {
                    let queued: usize = mine.iter().filter(|u| u.unit_type == producer).map(|u| u.build_queue.iter().filter(|&&q| q == sp).count()).sum();
                    let mut have = count(sp) + queued;
                    for p in mine.iter().filter(|u| u.unit_type == producer && done(u) && !u.has(flags::TRAINING) && !u.has(flags::LIFTED)) {
                        if have < cap && minerals >= cost(sp) && gas >= gas_cost(sp) && supply_free >= supply_cost(sp) {
                            cmds.push(Command::Train { building: p.id, unit_type: sp });
                            minerals -= cost(sp);
                            gas -= gas_cost(sp);
                            supply_free -= supply_cost(sp);
                            have += 1;
                            used.push(p.id);
                        }
                    }
                }
                for p in mine.iter().filter(|u| u.unit_type == k.production && u.has(flags::COMPLETED) && !u.has(flags::TRAINING) && !u.has(flags::LIFTED) && !used.contains(&u.id)) {
                    if army_n < army_cap && minerals - hold_army >= cost(k.army) && supply_free >= 2 {
                        cmds.push(Command::Train { building: p.id, unit_type: k.army });
                        minerals -= cost(k.army);
                        supply_free -= 2;
                        army_n += 1;
                    }
                }
            }
        }

        let kept = self.react_reserved_marines(&mine);
        match k.tactic {
            _ if self.tp.is_some() => {
                let guard_on = k.defend && (branch != Some(Branch::Defence) || self.react_marines_attack());
                let mut owned = kept.clone();
                if let Some((_, guards)) = base.filter(|_| guard_on).and_then(|b| self.guard_order(f, &mine, b, &kept)) {
                    owned.extend(guards);
                }
                self.tp_army(f, &mine, &owned, &mut cmds)
            }
            Tactic::Waves if branch == Some(Branch::Defence) => {}
            Tactic::Waves => {
                let army: Vec<&UnitState> = mine.iter().copied().filter(|u| u.unit_type == k.attacker && u.has(flags::COMPLETED) && u.loaded_in.is_none() && !kept.contains(&u.id)).collect();
                if army.len() >= k.first_wave && f.frame >= self.last_wave + WAVE_PERIOD {
                    let from = centroid(&army);
                    if let Some(target) = self.attack_target(f, from) {
                        let ids: Vec<UnitId> = army.iter().map(|u| u.id).collect();
                        for chunk in ids.chunks(12) {
                            cmds.push(Command::Unit { units: chunk.to_vec(), order: CmdOrder::Attack(Target::Pos(target)), queued: false });
                        }
                        self.last_wave = f.frame;
                    }
                }
            }
            Tactic::Drop => {
                if let Some(b) = base {
                    self.drop_step(f, &mine, b, &mut cmds);
                }
            }
        }
        if k.defend && (branch != Some(Branch::Defence) || self.react_marines_attack()) {
            if let Some(b) = base {
                self.defend(f, &mine, b, &kept, &mut cmds);
            }
        }
        self.react_marines(f, &mine, &mut cmds);

        Act { cmds, advance_frames: STEP_FRAMES, valid_until_frame: None, surrender: None }
    }

    fn react_stats(&self) -> Option<ReactStats> {
        if self.tp.is_some() {
            return None;
        }
        self.teacher_stats()
    }

    fn tp_stats(&self) -> Option<TpStats> {
        self.tp.as_ref().map(|t| t.stats.clone())
    }
}
