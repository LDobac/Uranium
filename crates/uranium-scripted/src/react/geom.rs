use std::collections::BTreeSet;

use uranium_proto::*;

use super::TILE;
use crate::util::dist2;

pub fn depot_center(t: Tile) -> Pos {
    (t.0 * 32 + 64, t.1 * 32 + 48)
}

pub(crate) fn bunker_center(t: Tile) -> Pos {
    (t.0 * 32 + 48, t.1 * 32 + 32)
}

pub(crate) fn within(a: Pos, b: Pos, tiles: i64) -> bool {
    dist2(a, b) <= (tiles * TILE) * (tiles * TILE)
}

pub(crate) fn toward(from: Pos, to: Pos, tiles: f64) -> Pos {
    let (dx, dy) = (to.0 as f64 - from.0 as f64, to.1 as f64 - from.1 as f64);
    let len = (dx * dx + dy * dy).sqrt();
    if len < 1.0 {
        return from;
    }
    let k = tiles * TILE as f64 / len;
    ((from.0 as f64 + dx * k).round().max(0.0) as u16, (from.1 as f64 + dy * k).round().max(0.0) as u16)
}

pub fn ground_path(walk: &BitGrid, from: Tile, to: Tile, solid: &BTreeSet<Tile>) -> Vec<Tile> {
    let (w, h) = ((walk.width / 4) as i32, (walk.height / 4) as i32);
    if w == 0 || h == 0 {
        return Vec::new();
    }
    let idx = |x: i32, y: i32| (y * w + x) as usize;
    for strict in [true, false] {
        let ok = |x: i32, y: i32| -> bool {
            if x < 0 || y < 0 || x >= w || y >= h {
                return false;
            }
            let t = (x as u16, y as u16);
            if t == from || t == to {
                return true;
            }
            if solid.contains(&t) {
                return false;
            }
            let mut cells = (0..4u16).flat_map(|dy| (0..4u16).map(move |dx| (dx, dy))).map(|(dx, dy)| walk.get(t.0 * 4 + dx, t.1 * 4 + dy));
            if strict { cells.all(|c| c) } else { cells.any(|c| c) }
        };
        let (fx, fy, tx, ty) = (from.0 as i32, from.1 as i32, to.0 as i32, to.1 as i32);
        if !ok(fx, fy) || !ok(tx, ty) {
            return Vec::new();
        }
        let mut prev = vec![u32::MAX; (w * h) as usize];
        prev[idx(fx, fy)] = idx(fx, fy) as u32;
        let mut q = std::collections::VecDeque::from([(fx, fy)]);
        while let Some((x, y)) = q.pop_front() {
            if (x, y) == (tx, ty) {
                let mut out = vec![to];
                let mut i = idx(x, y);
                while i != idx(fx, fy) {
                    i = prev[i] as usize;
                    out.push(((i as i32 % w) as u16, (i as i32 / w) as u16));
                }
                out.reverse();
                return out;
            }
            for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)] {
                let (nx, ny) = (x + dx, y + dy);
                if !ok(nx, ny) || prev[idx(nx, ny)] != u32::MAX || (dx != 0 && dy != 0 && !(ok(x + dx, y) && ok(x, y + dy))) {
                    continue;
                }
                prev[idx(nx, ny)] = idx(x, y) as u32;
                q.push_back((nx, ny));
            }
        }
    }
    Vec::new()
}
