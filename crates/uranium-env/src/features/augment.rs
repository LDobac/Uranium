//! Symmetry augmentation (030-R4): transforms of the valid map region applied to features,
//! masks, action coordinates and the own start index. Training uses the four flips
//! ([`Sym::augment`], 2026-10-18); all 8 dihedral transforms stay available and exactly
//! invertible (030-AC4). Applied at load time (040-R7). Location targets are top-left anchored
//! footprints, so they are transformed as rectangles.

use super::extract::{CONT_ANGLE, CONT_POS, CONT_VEL, Observation, cell_of_pixel, position_features};
use super::masks::L;
use super::schema::{CELLS, ENTITY_CONT, MAP, SPATIAL};
use crate::actions::{LOC_CLASSES, TargetKind};
use crate::features::extract::action_table;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Sym {
    pub flip_x: bool,
    pub flip_y: bool,
    /// Applied after the flips: (x, y) -> (y, x).
    pub transpose: bool,
}

impl Sym {
    pub const IDENTITY: Sym = Sym { flip_x: false, flip_y: false, transpose: false };

    /// The training augmentations (030-R4, 2026-10-18): identity, left-right flip, up-down flip
    /// and both. The transposing half of [`Sym::all`] is not used for training.
    pub fn augment() -> [Sym; 4] {
        let mut out = [Sym::IDENTITY; 4];
        for (i, o) in out.iter_mut().enumerate() {
            *o = Sym { flip_x: i & 1 != 0, flip_y: i & 2 != 0, transpose: false };
        }
        out
    }

    /// All 8 dihedral transforms (030-AC4 round trips).
    pub fn all() -> [Sym; 8] {
        let mut out = [Sym::IDENTITY; 8];
        for (i, o) in out.iter_mut().enumerate() {
            *o = Sym { flip_x: i & 1 != 0, flip_y: i & 2 != 0, transpose: i & 4 != 0 };
        }
        out
    }

    pub fn inverse(self) -> Sym {
        if self.transpose { Sym { flip_x: self.flip_y, flip_y: self.flip_x, transpose: true } } else { self }
    }

    /// Map size (w, h) after the transform.
    pub fn size(self, (w, h): (usize, usize)) -> (usize, usize) {
        if self.transpose { (h, w) } else { (w, h) }
    }

    /// Rectangle [x, x+fw) x [y, y+fh) in a w x h map -> new top-left and size.
    pub fn rect(self, (w, h): (usize, usize), x: usize, y: usize, fw: usize, fh: usize) -> (usize, usize, usize, usize) {
        let x = if self.flip_x { w - x - fw } else { x };
        let y = if self.flip_y { h - y - fh } else { y };
        if self.transpose { (y, x, fh, fw) } else { (x, y, fw, fh) }
    }

    pub fn tile(self, size: (usize, usize), x: usize, y: usize) -> (usize, usize) {
        let (x, y, _, _) = self.rect(size, x, y, 1, 1);
        (x, y)
    }

    /// Pixel coordinates (continuous).
    pub fn pixel(self, (w, h): (usize, usize), px: f32, py: f32) -> (f32, f32) {
        let px = if self.flip_x { (w * 32) as f32 - px } else { px };
        let py = if self.flip_y { (h * 32) as f32 - py } else { py };
        if self.transpose { (py, px) } else { (px, py) }
    }

    pub fn vector(self, vx: f32, vy: f32) -> (f32, f32) {
        let vx = if self.flip_x { -vx } else { vx };
        let vy = if self.flip_y { -vy } else { vy };
        if self.transpose { (vy, vx) } else { (vx, vy) }
    }

    /// BWAPI angle in turns (0 = +x, towards +y), as stored in `entity_cont`.
    pub fn angle(self, a: f32) -> f32 {
        let mut k = (a * 256.0).round() as i32;
        if self.flip_x {
            k = 128 - k;
        }
        if self.flip_y {
            k = -k;
        }
        if self.transpose {
            k = 64 - k;
        }
        k.rem_euclid(256) as f32 / 256.0
    }
}

/// Footprint of a location class's targets (034 위치 클래스); 1x1 for point targets. In a
/// transposed observation the footprint is h x w.
pub fn loc_footprint(class: &str, transposed: bool) -> (usize, usize) {
    let (w, h) = match class {
        "buildable_4x3" | "cc_slot" | "landable_4x3" => (4, 3),
        "buildable_3x2" => (3, 2),
        "buildable_2x2" => (2, 2),
        "refinery_slot" => (4, 2),
        _ => (1, 1),
    };
    if transposed { (h, w) } else { (w, h) }
}

fn permute_grid(sym: Sym, size: (usize, usize), src: &[f32], dst: &mut [f32]) {
    for y in 0..size.1 {
        for x in 0..size.0 {
            let (nx, ny) = sym.tile(size, x, y);
            dst[ny * MAP + nx] = src[y * MAP + x];
        }
    }
}

/// Transforms an observation in place of a copy.
pub fn transform(o: &Observation, sym: Sym) -> Observation {
    let size = o.map_size;
    let new_size = sym.size(size);
    let mut out = o.clone();
    out.map_size = new_size;
    out.transposed = o.transposed ^ sym.transpose;
    let ob = &mut out.obs;
    for i in 0..ob.n {
        let (px, py) = (o.obs.entity_pos[i * 2], o.obs.entity_pos[i * 2 + 1]);
        let (nx, ny) = sym.pixel(size, px, py);
        ob.entity_pos[i * 2] = nx;
        ob.entity_pos[i * 2 + 1] = ny;
        // 030-R4 / B-6: the cell comes from the transformed pixel with the extraction rule. Flipping
        // the tile index instead puts an even-width building (centre on a multiple of 32) one
        // tile off its mirrored position.
        ob.entity_cell[i] = cell_of_pixel(nx, ny, new_size);
        let f = &mut ob.entity_cont[i * ENTITY_CONT..(i + 1) * ENTITY_CONT];
        position_features(nx, ny, new_size.0, new_size.1, &mut f[CONT_POS..CONT_POS + 34]);
        let (vx, vy) = sym.vector(f[CONT_VEL], f[CONT_VEL + 1]);
        f[CONT_VEL] = vx;
        f[CONT_VEL + 1] = vy;
        f[CONT_ANGLE] = sym.angle(f[CONT_ANGLE]);
    }
    for c in 0..SPATIAL {
        let src = &o.obs.spatial[c * CELLS..(c + 1) * CELLS];
        let dst = &mut ob.spatial[c * CELLS..(c + 1) * CELLS];
        dst.fill(0.0);
        permute_grid(sym, size, src, dst);
    }
    let m = &mut out.masks.loc_class_mask;
    m.fill(false);
    for (l, class) in LOC_CLASSES.iter().enumerate().take(L) {
        let (fw, fh) = loc_footprint(class, o.transposed);
        for y in 0..size.1 {
            for x in 0..size.0 {
                if !o.masks.loc_class_mask[l * CELLS + y * MAP + x] || x + fw > size.0 || y + fh > size.1 {
                    continue;
                }
                let (nx, ny, _, _) = sym.rect(size, x, y, fw, fh);
                m[l * CELLS + ny * MAP + nx] = true;
            }
        }
    }
    out
}

/// Footprint of a start location's resource depot (its tile is the top-left one).
const START_FOOTPRINT: (usize, usize) = (4, 3);

/// [`transform`] plus `scalar_cat.start_loc` (030-R4): the own start location moves with the
/// map, so the index becomes that of the map start location nearest to the transformed one
/// (`starts` = the map's start locations in index order, as in `MapInfo`).
pub fn transform_with_starts(o: &Observation, sym: Sym, starts: &[uranium_proto::Tile]) -> Observation {
    let mut out = transform(o, sym);
    let i = o.obs.scalar_cat[2].max(0) as usize;
    let (w, h) = o.map_size;
    let (fw, fh) = START_FOOTPRINT;
    if let Some(&(x, y)) = starts.get(i)
        && (x as usize) + fw <= w
        && (y as usize) + fh <= h
    {
        let (nx, ny, _, _) = sym.rect(o.map_size, x as usize, y as usize, fw, fh);
        let d = |&(sx, sy): &uranium_proto::Tile| {
            let (dx, dy) = (sx as i64 - nx as i64, sy as i64 - ny as i64);
            dx * dx + dy * dy
        };
        if let Some((j, _)) = starts.iter().enumerate().min_by_key(|(j, s)| (d(s), *j)) {
            out.obs.scalar_cat[2] = j.min(7) as i32;
        }
    }
    out
}

/// Transforms a location label (tile index y*128+x) of `action` (030-R4). `size` and
/// `transposed` describe the orientation the label is in.
pub fn transform_target_loc(action: u16, tile: usize, size: (usize, usize), transposed: bool, sym: Sym) -> usize {
    let Some(a) = action_table().actions.get(action as usize) else { return tile };
    if a.target != TargetKind::Loc {
        return tile;
    }
    let (fw, fh) = loc_footprint(&a.loc_class, transposed);
    let (x, y) = (tile % MAP, tile / MAP);
    if x + fw > size.0 || y + fh > size.1 {
        return tile;
    }
    let (nx, ny, _, _) = sym.rect(size, x, y, fw, fh);
    ny * MAP + nx
}
