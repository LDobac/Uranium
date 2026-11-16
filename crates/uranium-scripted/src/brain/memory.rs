use uranium_proto::*;

use super::{Brain, CELL};
use crate::types::*;
use crate::util::{dist2, is_neutral_owner};

impl Brain {
    pub(super) fn update_memory(&mut self, f: &RawFrame) {
        for u in f.units.iter().filter(|u| u.owner != self.me && is_building(u.unit_type) && !is_neutral_owner(u.owner)) {
            self.enemy_buildings.insert(u.id, (u.pos, u.unit_type));
        }
        let seen: Vec<UnitId> = f.units.iter().map(|u| u.id).collect();
        self.enemy_buildings.retain(|id, (pos, _)| seen.contains(id) || !f.visibility.get(pos.0 / 32, pos.1 / 32));
        for (i, s) in self.starts.iter().enumerate() {
            if f.visibility.get(s.0 + 2, s.1 + 1) {
                self.scouted[i] = true;
            }
        }
        for cy in 0..self.cells_h {
            for cx in 0..self.cells_w {
                if f.visibility.get(cx * CELL + CELL / 2, cy * CELL + CELL / 2) {
                    self.cell_seen[(cy * self.cells_w + cx) as usize] = f.frame;
                }
            }
        }
    }

    pub(crate) fn attack_target(&mut self, f: &RawFrame, from: Pos) -> Option<Pos> {
        if let Some((p, _)) = self.enemy_buildings.values().min_by_key(|(p, _)| dist2(*p, from)) {
            return Some(*p);
        }
        if let Some(i) = self.scouted.iter().position(|s| !s) {
            let s = self.starts[i];
            return Some((s.0 * 32 + 64, s.1 * 32 + 48));
        }
        let enemy = f.units.iter().filter(|u| u.owner != self.me && !is_neutral_owner(u.owner)).min_by_key(|u| dist2(u.pos, from));
        if let Some(u) = enemy {
            return Some(u.pos);
        }
        let mut best: Option<(u32, Pos)> = None;
        for cy in 0..self.cells_h {
            for cx in 0..self.cells_w {
                let p = ((cx * CELL + CELL / 2) * 32, (cy * CELL + CELL / 2) * 32);
                if !self.walkable.get(p.0 / 8, p.1 / 8) {
                    continue;
                }
                let seen = self.cell_seen[(cy * self.cells_w + cx) as usize];
                if best.is_none_or(|(b, _)| seen < b) {
                    best = Some((seen, p));
                }
            }
        }
        best.map(|(_, p)| p)
    }

    pub(super) fn drop_target(&mut self, f: &RawFrame, from: Pos) -> Option<Pos> {
        let depot = self.enemy_buildings.values().filter(|(_, ty)| is_resource_depot(*ty)).min_by_key(|(p, _)| dist2(*p, from));
        match depot {
            Some((p, _)) => Some(*p),
            None => self.attack_target(f, from),
        }
    }
}
