use uranium_env::scout::is_ground_army_type;
use uranium_proto::*;

use super::{OUT_TILES, TpState, snap};
use crate::brain::Brain;
use crate::react::{Branch, bunker_center, toward, within};
use crate::types::*;
use crate::util::{centroid, dist2, is_enemy};

impl Brain {
    pub(super) fn stale_cells(&self) -> Vec<Pos> {
        let mut cells: Vec<(u32, u16, u16, Pos)> = Vec::new();
        for cy in 0..self.cells_h {
            for cx in 0..self.cells_w {
                let p = ((cx * crate::brain::CELL + crate::brain::CELL / 2) * 32, (cy * crate::brain::CELL + crate::brain::CELL / 2) * 32);
                if self.walkable.get(p.0 / 8, p.1 / 8) {
                    cells.push((self.cell_seen[(cy * self.cells_w + cx) as usize], cy, cx, p));
                }
            }
        }
        cells.sort_unstable();
        cells.into_iter().map(|(_, _, _, p)| p).collect()
    }

    pub(super) fn tanks_done(mine: &[&UnitState]) -> usize {
        mine.iter().filter(|u| is_tank(u.unit_type) && u.has(flags::COMPLETED)).count()
    }

    pub(crate) fn tp_tank_spot(&self, mine: &[&UnitState]) -> Option<Pos> {
        let t = self.tp.as_ref()?;
        let r = self.react.as_ref()?;
        let main = self.home_depot(mine).or(self.home)?;
        let natural = r.natural.and_then(|n| mine.iter().find(|u| is_resource_depot(u.unit_type) && building_tile(u) == n).map(|u| u.pos));
        let base = match r.branch {
            Branch::Defence => self.defence_base(mine, main).0,
            Branch::Macro => natural.unwrap_or(main),
        };
        let built = mine.iter().filter(|u| u.unit_type == BUNKER && within(u.pos, base, 10)).min_by_key(|u| (dist2(u.pos, base), u.id)).map(|u| u.pos);
        let ordered = self.pending.iter().find(|q| q.ty == BUNKER && within(bunker_center(q.tile), base, 10)).map(|q| bunker_center(q.tile));
        let planned = r.main_site.filter(|_| within(base, main, 1)).map(bunker_center);
        Some(match built.or(ordered).or(planned) {
            Some(b) => toward(b, base, t.p.tank_wait),
            None => {
                let outer = (!within(base, main, 1)).then_some(main);
                toward(base, self.bunker_facing(base, outer), t.p.tank_wait)
            }
        })
    }

    pub(crate) fn tp_army(&mut self, f: &RawFrame, mine: &[&UnitState], kept: &[UnitId], cmds: &mut Vec<Command>) {
        let Some(t) = self.tp.as_ref() else { return };
        let p = t.p.clone();
        let (sieging, unsieging) = (t.order_sieging, t.order_unsieging);
        let me = self.me;
        let researched = f.self_state.tech & (1u64 << TECH_TANK_SIEGE) != 0;
        let enemy_ground: Vec<Pos> = f.units.iter().filter(|u| is_enemy(u, me) && is_ground_army_type(u.unit_type)).map(|u| u.pos).collect();
        let near = |at: Pos, tiles: i64| enemy_ground.iter().any(|e| within(*e, at, tiles));
        let tanks: Vec<&UnitState> = mine.iter().copied().filter(|u| is_tank(u.unit_type) && u.has(flags::COMPLETED) && u.loaded_in.is_none()).collect();
        let t = self.tp.as_mut().expect("checked above");
        t.quiet.retain(|id, _| tanks.iter().any(|u| u.id == *id && u.unit_type == SIEGE_TANK_SIEGED));
        for u in tanks.iter().filter(|u| u.unit_type == SIEGE_TANK_SIEGED) {
            if near(u.pos, p.unsiege_range) {
                t.quiet.insert(u.id, f.frame);
            } else {
                t.quiet.entry(u.id).or_insert(f.frame);
            }
        }
        let quiet = t.quiet.clone();
        let branch = self.react_branch().unwrap_or_default();
        let marines_all = mine.iter().filter(|u| u.unit_type == MARINE && u.has(flags::COMPLETED)).count() as u32;
        let army_supply = marines_all + 2 * tanks.len() as u32;
        let wave = branch == Branch::Macro && ((tanks.len() >= p.wave_tanks && army_supply >= p.wave_supply) || (f.frame >= p.late_wave_from && army_supply >= p.late_wave_supply));
        if wave {
            self.tp.as_mut().expect("checked above").stats.wave_frame.get_or_insert(f.frame);
        }
        let depots = Self::own_depots(mine);
        let out = |u: &UnitState| !depots.iter().any(|d| within(u.pos, *d, OUT_TILES));
        let in_group = |u: &UnitState| wave || out(u);
        let spot = self.tp_tank_spot(mine);

        let (mut siege, mut unsiege) = (Vec::new(), Vec::new());
        for u in &tanks {
            if u.unit_type == SIEGE_TANK && researched && Some(u.order) != sieging && near(u.pos, p.siege_range) {
                siege.push(u.id);
            } else if u.unit_type == SIEGE_TANK_SIEGED && Some(u.order) != unsieging {
                let calm = quiet.get(&u.id).is_some_and(|q| f.frame >= q + p.unsiege_quiet);
                let to_go = in_group(u) || spot.is_some_and(|s| !within(u.pos, s, 4));
                if calm && to_go {
                    unsiege.push(u.id);
                }
            }
        }
        for chunk in siege.chunks(12) {
            cmds.push(Command::Unit { units: chunk.to_vec(), order: CmdOrder::Siege, queued: false });
        }
        for chunk in unsiege.chunks(12) {
            cmds.push(Command::Unit { units: chunk.to_vec(), order: CmdOrder::Unsiege, queued: false });
        }

        if let Some(s) = spot.filter(|_| crate::react::ReactState::due_on(true, None, f.frame, p.retarget)) {
            let home: Vec<UnitId> = tanks
                .iter()
                .filter(|u| u.unit_type == SIEGE_TANK && !siege.contains(&u.id) && Some(u.order) != sieging && !in_group(u) && !within(u.pos, s, 4))
                .map(|u| u.id)
                .collect();
            for chunk in home.chunks(12) {
                cmds.push(Command::Unit { units: chunk.to_vec(), order: CmdOrder::Move(Target::Pos(s)), queued: false });
            }
        }

        let last = self.tp.as_ref().expect("checked above").last_tanks;
        if !TpState::due(last, f.frame, p.retarget) {
            return;
        }
        let mobile = |u: &UnitState| u.unit_type == SIEGE_TANK && !siege.contains(&u.id) && Some(u.order) != sieging;
        let group_tanks: Vec<&UnitState> = tanks.iter().copied().filter(|u| in_group(u)).collect();
        let group_marines: Vec<&UnitState> =
            mine.iter().copied().filter(|u| u.unit_type == MARINE && u.has(flags::COMPLETED) && u.loaded_in.is_none() && !kept.contains(&u.id) && in_group(u)).collect();
        let mut acted = false;
        if !group_tanks.is_empty() || !group_marines.is_empty() {
            let lead = if group_tanks.is_empty() { centroid(&group_marines) } else { centroid(&group_tanks) };
            let hunting = self.enemy_buildings.is_empty() && self.scouted.iter().all(|s| *s) && !f.units.iter().any(|u| is_enemy(u, me));
            let cells = if hunting { self.stale_cells() } else { Vec::new() };
            if !cells.is_empty() {
                let tanks_go: Vec<UnitId> = group_tanks.iter().filter(|u| mobile(u)).map(|u| u.id).collect();
                for chunk in tanks_go.chunks(12) {
                    cmds.push(Command::Unit { units: chunk.to_vec(), order: CmdOrder::Attack(Target::Pos(cells[0])), queued: false });
                }
                let mut ids: Vec<UnitId> = group_marines.iter().map(|u| u.id).collect();
                ids.sort_unstable();
                for (k, chunk) in ids.chunks(12).enumerate() {
                    cmds.push(Command::Unit { units: chunk.to_vec(), order: CmdOrder::Attack(Target::Pos(cells[k % cells.len()])), queued: false });
                }
                acted |= !(tanks_go.is_empty() && ids.is_empty());
            } else if let Some(target) = self.attack_target(f, lead).map(snap) {
                let lead_d = dist2(lead, target);
                let gap = p.group_gap * 32;
                let front = snap(toward(lead, target, (p.group_gap - 2) as f64));
                let mut units: Vec<&UnitState> = group_tanks.iter().copied().filter(|u| mobile(u)).chain(group_marines.iter().copied()).collect();
                units.sort_by_key(|u| u.id);
                for chunk in units.chunks(12) {
                    let (wait, go): (Vec<&UnitState>, Vec<&UnitState>) = chunk.iter().partition(|u| dist2(u.pos, lead) > gap * gap && dist2(u.pos, target) < lead_d);
                    for (ids, at) in [(go, target), (wait, front)] {
                        if !ids.is_empty() {
                            cmds.push(Command::Unit { units: ids.iter().map(|u| u.id).collect(), order: CmdOrder::Attack(Target::Pos(at)), queued: false });
                        }
                    }
                }
                acted |= !units.is_empty();
            }
        }
        if acted {
            self.tp.as_mut().expect("checked above").last_tanks = Some(f.frame);
        }
    }
}
