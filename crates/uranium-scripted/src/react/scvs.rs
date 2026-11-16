use std::collections::BTreeSet;

use uranium_env::scout::is_ground_army_type;
use uranium_proto::*;

use super::{ATTACK_ORDERS, BUNKER_HP, Branch, Harass, ORDER_CONSTRUCTING, ReactState, within};
use crate::brain::Brain;
use crate::types::*;
use crate::util::{dist2, done, is_enemy, nearest};

impl Brain {
    pub(crate) fn react_scvs(&mut self, f: &RawFrame, mine: &[&UnitState], minerals: &[&UnitState], cmds: &mut Vec<Command>) {
        let Some(r) = self.react.as_ref() else { return };
        let (p, branch) = (r.p.clone(), r.branch);
        let rings = self.tp.is_some();
        let home = self.home_depot(mine).or(self.home);
        let gather_home = |ids: Vec<UnitId>, cmds: &mut Vec<Command>| {
            if ids.is_empty() {
                return;
            }
            if let Some(m) = home.and_then(|h| nearest(h, minerals)) {
                for chunk in ids.chunks(12) {
                    cmds.push(Command::Unit { units: chunk.to_vec(), order: CmdOrder::Gather(m.id), queued: false });
                }
            }
        };
        let alive = |id: &UnitId| mine.iter().any(|u| u.id == *id);
        let r = self.react.as_mut().expect("checked above");
        r.pulled.retain(alive);
        r.repairers.retain(alive);
        self.react_harass(f, mine, minerals, cmds);
        self.react_resume(f, mine, cmds);

        let r = self.react.as_ref().expect("checked above");
        if branch == Branch::Macro {
            let ids: Vec<UnitId> = r.pulled.iter().chain(&r.repairers).copied().collect();
            gather_home(ids, cmds);
            let r = self.react.as_mut().expect("checked above");
            r.pulled.clear();
            r.repairers.clear();
            return;
        }
        let me = self.me;
        let depots = Self::own_depots(mine);
        let bunkers: Vec<Pos> = mine.iter().filter(|u| u.unit_type == BUNKER).map(|u| u.pos).collect();
        let mut bases = depots.clone();
        for d in &depots {
            let line: Vec<&UnitState> = minerals.iter().copied().filter(|m| within(m.pos, *d, 10)).collect();
            if !line.is_empty() {
                bases.push(crate::util::centroid(&line));
            }
        }
        let threats: Vec<&UnitState> = f
            .units
            .iter()
            .filter(|u| is_enemy(u, me) && is_ground_army_type(u.unit_type) && !u.has(flags::BURROWED))
            .filter(|u| bunkers.iter().any(|b| within(u.pos, *b, p.pull_bunker)) || bases.iter().any(|b| within(u.pos, *b, p.pull_base)))
            .collect();
        let scvs: Vec<&&UnitState> = mine.iter().filter(|u| u.unit_type == SCV && u.has(flags::COMPLETED)).collect();
        let (mut pulled, mut repairers, last_pull, last_repair) = {
            let r = self.react.as_ref().expect("checked above");
            (r.pulled.clone(), r.repairers.clone(), r.last_pull, r.last_repair)
        };
        if rings {
            let repair = uranium_env::types::order_id("Repair");
            let bunker_ids: BTreeSet<UnitId> = mine.iter().filter(|u| u.unit_type == BUNKER).map(|u| u.id).collect();
            let repairing = |u: &UnitState| Some(u.order) == repair && u.order_target.is_some_and(|t| bunker_ids.contains(&t));
            let fresh = !ReactState::due_on(true, last_repair, f.frame, p.retarget);
            repairers.retain(|id| mine.iter().copied().find(|u| u.id == *id).is_some_and(|u| fresh || repairing(u)));
            for u in mine.iter().filter(|u| u.unit_type == SCV && repairing(u)) {
                if !repairers.contains(&u.id) && !pulled.contains(&u.id) {
                    repairers.push(u.id);
                }
            }
            repairers.sort_unstable();
        }
        let others: BTreeSet<UnitId> = {
            let r = self.react.as_ref().expect("checked above");
            r.repairers.iter().chain(r.harass.iter().flat_map(|h| &h.guards)).chain(r.resumers.values().map(|(scv, _)| scv)).copied().chain(self.pending.iter().map(|q| q.worker)).chain(self.scout).collect()
        };
        let fresh = !ReactState::due_on(rings, last_pull, f.frame, p.retarget);
        let threat_ids: BTreeSet<UnitId> = threats.iter().map(|u| u.id).collect();
        let attacking = |u: &UnitState| ATTACK_ORDERS.contains(&u.order) && u.order_target.is_some_and(|t| threat_ids.contains(&t));
        pulled.retain(|id| mine.iter().copied().find(|u| u.id == *id).is_some_and(|u| fresh || attacking(u) || !(u.has(flags::GATHERING) || u.has(flags::CONSTRUCTING))));
        let joined: Vec<UnitId> = scvs.iter().map(|u| **u).filter(|u| attacking(u) && !others.contains(&u.id) && !pulled.contains(&u.id)).map(|u| u.id).collect();
        pulled.extend(joined);
        let mut pull_acted = false;
        if threats.is_empty() {
            gather_home(std::mem::take(&mut pulled), cmds);
        } else if ReactState::due_on(rings, last_pull, f.frame, p.retarget) {
            let anchors = if bunkers.is_empty() { &depots } else { &bunkers };
            let target = threats.iter().copied().min_by_key(|u| (anchors.iter().map(|a| dist2(u.pos, *a)).min().unwrap_or(0), u.id)).expect("non-empty");
            let want = p.pull_max.min(scvs.len().saturating_sub(p.pull_keep));
            let jobs: Vec<UnitId> = self.pending.iter().map(|q| q.worker).chain(self.scout).chain(repairers.iter().copied()).chain(self.react_busy().into_iter().filter(|id| !pulled.contains(id))).collect();
            let mut free: Vec<&&&UnitState> = scvs.iter().filter(|u| !u.has(flags::CONSTRUCTING) && !jobs.contains(&u.id) && !pulled.contains(&u.id)).collect();
            free.sort_by_key(|u| (crate::tp::pick_key_id(rings, u.pos, target.pos), u.id));
            for u in free.into_iter().take(want.saturating_sub(pulled.len())) {
                pulled.push(u.id);
            }
            if pulled.len() > want {
                let extra = pulled.split_off(want);
                gather_home(extra, cmds);
            }
            if !pulled.is_empty() {
                cmds.push(Command::Unit { units: pulled.clone(), order: CmdOrder::Attack(Target::Unit(target.id)), queued: false });
                pull_acted = true;
            }
        }
        let mut repair_acted = false;
        if ReactState::due_on(rings, last_repair, f.frame, p.retarget) {
            let mut bunkers: Vec<&&UnitState> = mine.iter().filter(|u| u.unit_type == BUNKER && done(u) && u.hp < BUNKER_HP).collect();
            bunkers.sort_by_key(|b| (b.hp, b.id));
            let damaged = bunkers.iter().find(|b| (b.hp as u32) * 100 < BUNKER_HP as u32 * p.repair_below_pct).or(if repairers.is_empty() { None } else { bunkers.first() });
            match damaged {
                Some(b) => {
                    let jobs: Vec<UnitId> = self.pending.iter().map(|q| q.worker).chain(self.scout).chain(pulled.iter().copied()).chain(self.react_busy().into_iter().filter(|id| !repairers.contains(id))).collect();
                    let mut free: Vec<&&&UnitState> = scvs.iter().filter(|u| !u.has(flags::CONSTRUCTING) && !jobs.contains(&u.id) && !repairers.contains(&u.id)).collect();
                    free.sort_by_key(|u| (crate::tp::pick_key_id(rings, u.pos, b.pos), u.id));
                    for u in free.into_iter().take(p.repairers.saturating_sub(repairers.len())) {
                        repairers.push(u.id);
                    }
                    if !repairers.is_empty() {
                        cmds.push(Command::Unit { units: repairers.clone(), order: CmdOrder::Repair(b.id), queued: false });
                        repair_acted = true;
                    }
                }
                None => {
                    if !repairers.is_empty() {
                        gather_home(std::mem::take(&mut repairers), cmds);
                        repair_acted = true;
                    }
                }
            }
        }
        let r = self.react.as_mut().expect("checked above");
        (r.pulled, r.repairers) = (pulled, repairers);
        if pull_acted {
            r.last_pull = Some(f.frame);
        }
        if repair_acted {
            r.last_repair = Some(f.frame);
        }
    }

    fn react_harass(&mut self, f: &RawFrame, mine: &[&UnitState], minerals: &[&UnitState], cmds: &mut Vec<Command>) {
        let me = self.me;
        let builders: Vec<UnitId> = self.pending.iter().map(|q| q.worker).collect();
        let aligned = self.tp.is_some();
        let Some(r) = self.react.as_mut() else { return };
        let p = r.p.clone();
        let workers: Vec<&UnitState> = f.units.iter().filter(|u| is_enemy(u, me) && is_worker(u.unit_type)).collect();
        if let Some(h) = r.harass.as_mut() {
            h.guards.retain(|id| mine.iter().any(|u| u.id == *id));
            if let Some(b) = mine.iter().find(|u| u.id == h.builder) {
                h.at = b.pos;
            }
            let target = workers.iter().find(|w| w.id == h.target).filter(|w| within(w.pos, h.at, p.harass_release));
            match target {
                Some(t) if !h.guards.is_empty() => {
                    if ReactState::due_on(aligned, Some(h.last_order), f.frame, p.retarget) {
                        cmds.push(Command::Unit { units: h.guards.clone(), order: CmdOrder::Attack(Target::Unit(t.id)), queued: false });
                        h.last_order = f.frame;
                    }
                    return;
                }
                _ => {
                    let guards = std::mem::take(&mut h.guards);
                    r.harass = None;
                    if let Some(m) = self.home.and_then(|h| nearest(h, minerals)).filter(|_| !guards.is_empty()) {
                        cmds.push(Command::Unit { units: guards, order: CmdOrder::Gather(m.id), queued: false });
                    }
                }
            }
        }
        let hit = mine
            .iter()
            .filter(|u| u.unit_type == SCV && u.has(flags::UNDER_ATTACK) && (u.has(flags::CONSTRUCTING) || u.order == ORDER_CONSTRUCTING || builders.contains(&u.id)))
            .find_map(|b| workers.iter().filter(|w| within(w.pos, b.pos, p.harass_radius)).min_by_key(|w| (dist2(w.pos, b.pos), w.id)).map(|w| (*b, *w)));
        let Some((builder, worker)) = hit else { return };
        let jobs: BTreeSet<UnitId> = self.react_busy().into_iter().chain(builders).chain(self.scout).collect();
        let mut free: Vec<&&UnitState> = mine.iter().filter(|u| u.unit_type == SCV && u.has(flags::COMPLETED) && u.has(flags::GATHERING) && !u.has(flags::CONSTRUCTING) && !jobs.contains(&u.id)).collect();
        free.sort_by_key(|u| (dist2(u.pos, worker.pos), u.id));
        let guards: Vec<UnitId> = free.iter().take(p.harass_guards).map(|u| u.id).collect();
        if guards.is_empty() {
            return;
        }
        cmds.push(Command::Unit { units: guards.clone(), order: CmdOrder::Attack(Target::Unit(worker.id)), queued: false });
        let r = self.react.as_mut().expect("checked above");
        r.harass = Some(Harass { guards, target: worker.id, builder: builder.id, at: builder.pos, last_order: f.frame });
    }

    fn react_resume(&mut self, f: &RawFrame, mine: &[&UnitState], cmds: &mut Vec<Command>) {
        let pending: Vec<UnitId> = self.pending.iter().map(|q| q.worker).chain(self.scout).collect();
        let busy = self.react_busy();
        let rings = self.tp.is_some();
        let Some(r) = self.react.as_mut() else { return };
        let after = r.p.resume_after;
        let unfinished: Vec<&&UnitState> = mine.iter().filter(|u| is_building(u.unit_type) && !u.has(flags::COMPLETED)).collect();
        let building = |id: UnitId| mine.iter().any(|u| u.unit_type == SCV && u.order == ORDER_CONSTRUCTING && u.order_target == Some(id));
        r.orphan_since.retain(|id, _| unfinished.iter().any(|u| u.id == *id && !building(u.id)));
        r.resumers.retain(|id, (scv, _)| r.orphan_since.contains_key(id) && mine.iter().any(|u| u.id == *scv));
        let mut taken: BTreeSet<UnitId> = r.resumers.values().map(|(scv, _)| *scv).collect();
        for b in unfinished.iter().filter(|u| !building(u.id)) {
            let since = *r.orphan_since.entry(b.id).or_insert(f.frame);
            let last = r.resumers.get(&b.id).map(|(_, at)| *at);
            if f.frame < since + after || last.is_some_and(|at| f.frame < at + after) {
                continue;
            }
            let prev = r.resumers.get(&b.id).map(|(scv, _)| *scv);
            let scv = mine
                .iter()
                .filter(|u| u.unit_type == SCV && u.has(flags::COMPLETED) && !u.has(flags::CONSTRUCTING) && u.order != ORDER_CONSTRUCTING && !pending.contains(&u.id))
                .filter(|u| Some(u.id) == prev || (!busy.contains(&u.id) && !taken.contains(&u.id)))
                .min_by_key(|u| (crate::tp::pick_key(rings, u.pos, b.pos), u.id));
            if let Some(s) = scv {
                cmds.push(Command::Unit { units: vec![s.id], order: CmdOrder::RightClick(Target::Unit(b.id)), queued: false });
                taken.insert(s.id);
                r.resumers.insert(b.id, (s.id, f.frame));
            }
        }
    }
}
