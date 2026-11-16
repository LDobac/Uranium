use uranium_proto::*;

use super::{Brain, Pending};
use crate::tp;
use crate::types::*;
use crate::util::{dist2, done};

impl Brain {
    pub(crate) fn place(&self, ty: u16, f: &RawFrame, mine: &[&UnitState], base: Pos, reserved: &[(Tile, u16)]) -> Option<Tile> {
        if is_refinery(ty) {
            return f
                .units
                .iter()
                .filter(|u| u.unit_type == GEYSER && !reserved.iter().any(|(t, _)| *t == building_tile(u)))
                .filter(|u| dist2(u.pos, base) < 12 * 32 * 12 * 32)
                .min_by_key(|u| dist2(u.pos, base))
                .map(building_tile);
        }
        match self.kit.race {
            Race::Protoss if ty != PYLON => {
                let pylons: Vec<Pos> = mine.iter().filter(|u| u.unit_type == PYLON && u.has(flags::COMPLETED)).map(|u| u.pos).collect();
                let (w, h) = tile_size(ty);
                pylons.iter().find_map(|&py| {
                    self.placer.find_where(ty, py, &f.units, reserved, 8, |(tx, tyy)| {
                        let cx = (tx * 32 + w * 16) as f64 - py.0 as f64;
                        let cy = (tyy * 32 + h * 16) as f64 - py.1 as f64;
                        (cx / 224.0).powi(2) + (cy / 128.0).powi(2) <= 1.0
                    })
                })
            }
            Race::Zerg => {
                let (w, h) = tile_size(ty);
                self.placer.find_where(ty, base, &f.units, reserved, 10, |(tx, tyy)| {
                    (tyy..tyy + h).all(|y| (tx..tx + w).all(|x| f.creep.get(x, y)))
                })
            }
            _ => self.placer.find(ty, base, &f.units, reserved),
        }
    }

    pub(crate) fn order_build(&mut self, ty: u16, f: &RawFrame, mine: &[&UnitState], base: Pos, busy: &[UnitId], cmds: &mut Vec<Command>) -> bool {
        let reserved: Vec<(Tile, u16)> = self.pending.iter().map(|p| (p.tile, p.ty)).chain(self.react_reserved_sites()).collect();
        let Some(tile) = self.place(ty, f, mine, base, &reserved) else { return false };
        let center = (tile.0 * 32 + 16, tile.1 * 32 + 16);
        let skip_constructing = !self.kit.tech.is_empty() || self.kit.tp.is_some();
        let worker = mine
            .iter()
            .filter(|u| u.unit_type == self.kit.worker && u.has(flags::COMPLETED) && !u.has(flags::CARRYING) && !busy.contains(&u.id))
            .filter(|u| !(skip_constructing && u.has(flags::CONSTRUCTING)))
            .min_by_key(|u| if self.tp.is_some() { (tp::pick_key(true, u.pos, center), u.id) } else { (dist2(u.pos, center), 0) });
        let Some(w) = worker else { return false };
        cmds.push(Command::Build { worker: w.id, unit_type: ty, tile });
        self.pending.push(Pending { ty, tile, worker: w.id, issued: f.frame });
        true
    }

    pub(super) fn order_tech(&mut self, ty: u16, f: &RawFrame, mine: &[&UnitState], base: Pos, busy: &[UnitId], cmds: &mut Vec<Command>) -> bool {
        if ty == CONTROL_TOWER {
            let Some(s) = mine.iter().find(|u| u.unit_type == STARPORT && done(u) && !u.has_addon) else { return false };
            cmds.push(Command::BuildAddon { building: s.id, unit_type: ty });
            self.pending.push(Pending { ty, tile: addon_tile(building_tile(s)), worker: s.id, issued: f.frame });
            return true;
        }
        if ty != LAIR {
            return self.order_build(ty, f, mine, base, busy, cmds);
        }
        let Some(h) = mine.iter().find(|u| u.unit_type == HATCHERY && done(u)) else { return false };
        cmds.push(Command::Morph { unit: h.id, unit_type: LAIR });
        self.pending.push(Pending { ty, tile: building_tile(h), worker: h.id, issued: f.frame });
        true
    }

    pub(crate) fn free_worker(&self, mine: &[&UnitState]) -> Option<UnitId> {
        let jobs = self.react_busy();
        let busy = |id: UnitId| self.pending.iter().any(|p| p.worker == id) || self.gas_crew.contains(&id) || self.scout == Some(id) || jobs.contains(&id);
        mine.iter()
            .filter(|u| u.unit_type == self.kit.worker && u.has(flags::COMPLETED) && !u.has(flags::CARRYING) && !u.has(flags::CONSTRUCTING) && !busy(u.id))
            .min_by_key(|u| u.id)
            .map(|u| u.id)
    }
}
