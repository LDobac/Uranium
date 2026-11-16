use uranium_proto::*;

use super::{Brain, DEFEND_RADIUS, DROP_SIZE, Drop, DropPhase, LOAD_TIMEOUT, WAVE_PERIOD};
use crate::kit::Tactic;
use crate::react::ReactState;
use crate::tp;
use crate::types::*;
use crate::util::{centroid, dist2, done, is_neutral_owner, nearest};

impl Brain {
    pub(super) fn scouting(&mut self, f: &RawFrame, mine: &[&UnitState], id: UnitId, minerals: &[&UnitState], cmds: &mut Vec<Command>) {
        let Some(s) = mine.iter().find(|u| u.id == id) else { return };
        let next = self.starts.iter().zip(&self.scouted).filter(|(_, done)| !**done).map(|(t, _)| (t.0 * 32 + 64, t.1 * 32 + 48)).min_by_key(|p| dist2(*p, s.pos));
        match next {
            Some(p) if self.enemy_buildings.is_empty() => {
                if f.frame >= self.scout_order + WAVE_PERIOD || s.has(flags::IDLE) {
                    cmds.push(Command::Unit { units: vec![id], order: CmdOrder::Move(Target::Pos(p)), queued: false });
                    self.scout_order = f.frame;
                }
            }
            _ => {
                if let Some(m) = nearest(s.pos, minerals) {
                    cmds.push(Command::Unit { units: vec![id], order: CmdOrder::Gather(m.id), queued: false });
                }
                self.scout = None;
                self.scout_done = true;
            }
        }
    }

    pub(super) fn drop_step(&mut self, f: &RawFrame, mine: &[&UnitState], base: Pos, cmds: &mut Vec<Command>) {
        let unit = |id: UnitId| mine.iter().copied().find(|u| u.id == id);
        for i in 0..self.drops.len() {
            self.drops[i].squad.retain(|id| unit(*id).is_some());
            let d = &self.drops[i];
            let (ship, squad, phase, last) = (unit(d.ship), d.squad.clone(), d.phase, d.last_order);
            let loaded = squad.iter().filter(|id| unit(**id).is_some_and(|u| u.loaded_in == Some(d.ship))).count();
            match (phase, ship) {
                (DropPhase::Load { .. } | DropPhase::Fly, None) => self.drops[i].phase = DropPhase::Fight,
                (DropPhase::Load { since }, Some(s)) => {
                    let full = loaded == squad.len() && !squad.is_empty();
                    if full || (f.frame >= since + LOAD_TIMEOUT && loaded >= DROP_SIZE / 2) {
                        if let Some(t) = self.drop_target(f, s.pos) {
                            cmds.push(Command::Unit { units: vec![s.id], order: CmdOrder::UnloadAll(Some(t)), queued: false });
                            self.drops[i].phase = DropPhase::Fly;
                            self.drops[i].last_order = f.frame;
                        }
                    } else if f.frame >= last + WAVE_PERIOD {
                        let waiting: Vec<UnitId> = squad.iter().copied().filter(|id| unit(*id).is_some_and(|u| u.loaded_in.is_none())).collect();
                        if !waiting.is_empty() {
                            cmds.push(Command::Unit { units: waiting, order: CmdOrder::Load(s.id), queued: false });
                        }
                        self.drops[i].last_order = f.frame;
                    }
                }
                (DropPhase::Fly, Some(s)) => {
                    if loaded == 0 {
                        cmds.push(Command::Unit { units: vec![s.id], order: CmdOrder::Move(Target::Pos(base)), queued: false });
                        self.drops[i].phase = DropPhase::Fight;
                    } else if f.frame >= last + 24 * 10 {
                        if let Some(t) = self.drop_target(f, s.pos) {
                            cmds.push(Command::Unit { units: vec![s.id], order: CmdOrder::UnloadAll(Some(t)), queued: false });
                        }
                        self.drops[i].last_order = f.frame;
                    }
                }
                (DropPhase::Fight, _) => {
                    let out: Vec<&UnitState> = squad.iter().filter_map(|id| unit(*id)).filter(|u| u.loaded_in.is_none()).collect();
                    if !out.is_empty() && f.frame >= last + WAVE_PERIOD {
                        if let Some(t) = self.attack_target(f, centroid(&out)) {
                            let ids: Vec<UnitId> = out.iter().map(|u| u.id).collect();
                            cmds.push(Command::Unit { units: ids, order: CmdOrder::Attack(Target::Pos(t)), queued: false });
                        }
                        self.drops[i].last_order = f.frame;
                    }
                }
            }
        }
        self.drops.retain(|d| !d.squad.is_empty());

        if self.drops.iter().any(|d| d.phase != DropPhase::Fight) {
            return;
        }
        let ships: Vec<UnitId> = self.drops.iter().map(|d| d.ship).collect();
        let Some(ship) = mine.iter().find(|u| u.unit_type == DROPSHIP && done(u) && !ships.contains(&u.id) && dist2(u.pos, base) < DEFEND_RADIUS * DEFEND_RADIUS) else { return };
        let taken: Vec<UnitId> = self.drops.iter().flat_map(|d| d.squad.iter().copied()).collect();
        let mut marines: Vec<&UnitState> = mine.iter().copied().filter(|u| u.unit_type == MARINE && u.has(flags::COMPLETED) && u.loaded_in.is_none() && !taken.contains(&u.id)).collect();
        if marines.len() < DROP_SIZE {
            return;
        }
        marines.sort_by_key(|u| (dist2(u.pos, ship.pos), u.id));
        let squad: Vec<UnitId> = marines[..DROP_SIZE].iter().map(|u| u.id).collect();
        cmds.push(Command::Unit { units: squad.clone(), order: CmdOrder::Load(ship.id), queued: false });
        self.drops.push(Drop { ship: ship.id, squad, phase: DropPhase::Load { since: f.frame }, last_order: f.frame });
    }

    pub(super) fn defend(&mut self, f: &RawFrame, mine: &[&UnitState], base: Pos, kept: &[UnitId], cmds: &mut Vec<Command>) {
        let Some((at, guards)) = self.guard_order(f, mine, base, kept) else { return };
        for chunk in guards.chunks(12) {
            cmds.push(Command::Unit { units: chunk.to_vec(), order: CmdOrder::Attack(Target::Pos(at)), queued: false });
        }
        self.last_defend = f.frame;
    }

    pub(super) fn guard_order(&self, f: &RawFrame, mine: &[&UnitState], base: Pos, kept: &[UnitId]) -> Option<(Pos, Vec<UnitId>)> {
        let react = self.kit.react.is_some();
        let has_guard = self.kit.tactic == Tactic::Drop || self.kit.attacker != self.kit.army || react;
        if !has_guard || !ReactState::due_on(self.tp.is_some(), Some(self.last_defend), f.frame, WAVE_PERIOD) {
            return None;
        }
        let bases: Vec<Pos> = if react { mine.iter().filter(|u| is_resource_depot(u.unit_type)).map(|u| u.pos).collect() } else { vec![base] };
        let near_base = |p: Pos| bases.iter().map(|b| dist2(p, *b)).min().unwrap_or(i64::MAX);
        let reach = self.react_guard_radius().map_or(DEFEND_RADIUS, |t| t * 32);
        let threat = f
            .units
            .iter()
            .filter(|u| u.owner != self.me && !is_neutral_owner(u.owner) && near_base(u.pos) < reach * reach)
            .min_by_key(|u| (near_base(u.pos), u.id));
        let t = threat?;
        let on_drop: Vec<UnitId> = self.drops.iter().flat_map(|d| d.squad.iter().copied()).collect();
        let guards: Vec<UnitId> = mine
            .iter()
            .filter(|u| u.unit_type == self.kit.army && u.has(flags::COMPLETED) && u.loaded_in.is_none() && !on_drop.contains(&u.id) && !kept.contains(&u.id))
            .filter(|u| !react || near_base(u.pos) < DEFEND_RADIUS * DEFEND_RADIUS)
            .map(|u| u.id)
            .collect();
        Some((if self.tp.is_some() { tp::snap(t.pos) } else { t.pos }, guards))
    }
}
