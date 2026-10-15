//! Simple spiral building placement for T0 bots.

use uranium_proto::{BitGrid, Pos, StaticMap, Tile, UnitState};

use crate::types::*;

pub struct Placer {
    buildable: BitGrid,
    w: u16,
    h: u16,
}

impl Placer {
    pub fn new(map: &StaticMap) -> Self {
        Self { buildable: map.buildable.clone(), w: map.width_tiles, h: map.height_tiles }
    }

    /// First free footprint for `ty` on a spiral around `near` (pixels).
    /// Keeps a 1-tile gap around buildings and 3 tiles away from resources so mining
    /// paths stay open. `reserved` are footprints already promised to pending builds.
    pub fn find(&self, ty: u16, near: Pos, units: &[UnitState], reserved: &[(Tile, u16)]) -> Option<Tile> {
        let mut blocked = vec![false; self.w as usize * self.h as usize];
        let mut mark = |x0: i32, y0: i32, w: i32, h: i32, pad: i32| {
            for y in (y0 - pad)..(y0 + h + pad) {
                for x in (x0 - pad)..(x0 + w + pad) {
                    if x >= 0 && y >= 0 && (x as u16) < self.w && (y as u16) < self.h {
                        blocked[y as usize * self.w as usize + x as usize] = true;
                    }
                }
            }
        };
        for u in units {
            if is_resource(u.unit_type) || is_building(u.unit_type) {
                let (tx, ty_) = building_tile(u);
                let (w, h) = tile_size(u.unit_type);
                let pad = if is_resource(u.unit_type) { 3 } else { 1 };
                mark(tx as i32, ty_ as i32, w as i32, h as i32, pad);
            }
        }
        for &((tx, ty_), rty) in reserved {
            let (w, h) = tile_size(rty);
            mark(tx as i32, ty_ as i32, w as i32, h as i32, 1);
        }
        let (fw, fh) = tile_size(ty);
        let (cx, cy) = ((near.0 / 32) as i32, (near.1 / 32) as i32);
        for r in 2..24i32 {
            for dy in -r..=r {
                for dx in -r..=r {
                    if dx.abs() != r && dy.abs() != r {
                        continue;
                    }
                    let (x, y) = (cx + dx, cy + dy);
                    if self.fits(x, y, fw, fh, &blocked) {
                        return Some((x as u16, y as u16));
                    }
                }
            }
        }
        None
    }

    fn fits(&self, x: i32, y: i32, w: u16, h: u16, blocked: &[bool]) -> bool {
        if x < 1 || y < 1 || x + w as i32 >= self.w as i32 || y + h as i32 >= self.h as i32 {
            return false;
        }
        (y..y + h as i32).all(|ty| {
            (x..x + w as i32).all(|tx| self.buildable.get(tx as u16, ty as u16) && !blocked[ty as usize * self.w as usize + tx as usize])
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(w: u16, h: u16) -> StaticMap {
        let mut b = BitGrid::new(w, h);
        for y in 0..h {
            for x in 0..w {
                b.set(x, y, true);
            }
        }
        StaticMap { width_tiles: w, height_tiles: h, walkable: BitGrid::new(w * 4, h * 4), buildable: b, ground_height: vec![0; (w * h) as usize], start_locations: vec![] }
    }

    fn unit(ty: u16, tile: Tile) -> UnitState {
        let (w, h) = tile_size(ty);
        UnitState { unit_type: ty, pos: (tile.0 * 32 + w * 16, tile.1 * 32 + h * 16), ..Default::default() }
    }

    #[test]
    fn placement_avoids_cc_and_minerals() {
        let p = Placer::new(&map(64, 64));
        let units = vec![unit(CC, (30, 30)), unit(176, (30, 25))];
        let t = p.find(DEPOT, (30 * 32 + 64, 30 * 32 + 48), &units, &[]).unwrap();
        // depot footprint (3x2) must not touch the CC (+1 gap) nor the mineral (+3 gap)
        let overlaps = |x0: u16, y0: u16, w: u16, h: u16, pad: u16| {
            t.0 + 3 > x0.saturating_sub(pad) && t.0 < x0 + w + pad && t.1 + 2 > y0.saturating_sub(pad) && t.1 < y0 + h + pad
        };
        assert!(!overlaps(30, 30, 4, 3, 1), "{t:?}");
        assert!(!overlaps(30, 25, 2, 1, 3), "{t:?}");
    }

    #[test]
    fn reserved_footprints_are_respected() {
        let p = Placer::new(&map(64, 64));
        let first = p.find(BARRACKS, (1000, 1000), &[], &[]).unwrap();
        let second = p.find(BARRACKS, (1000, 1000), &[], &[(first, BARRACKS)]).unwrap();
        assert_ne!(first, second);
    }

    #[test]
    fn building_tile_inverts_center() {
        assert_eq!(building_tile(&unit(BARRACKS, (10, 20))), (10, 20));
        assert_eq!(building_tile(&unit(DEPOT, (5, 7))), (5, 7));
    }
}
