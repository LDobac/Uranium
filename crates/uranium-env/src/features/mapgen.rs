//! Map precompute (031 맵 사전계산, AMB-031-3: every map, map-agnostic): base locations, choke
//! tiles and wall-in hint tiles, from the static map and the map's resource placement. Output
//! feeds the `base_loc` / `choke` / `wall_candidate` observation channels only; nothing is
//! enforced on the executor (031).

use std::collections::VecDeque;

use anyhow::{Result, ensure};
use uranium_proto::{StaticMap, Tile};

use super::map::{MapPrecompute, MapResource, PRECOMPUTE_VERSION};
use crate::types::TypeTables;

/// Resources closer than this (tiles) belong to the same base.
const CLUSTER_TILES: i32 = 10;
/// Resource depots may not be built within 3 tiles of a resource (BW rule).
const NO_BUILD: i32 = 3;
/// A cluster needs this many resources to count as a base.
const MIN_RESOURCES: usize = 4;
/// Corridors at most this many tiles from a wall on either side are choke candidates.
const CHOKE_CLEARANCE: i32 = 3;
/// How far along a corridor to look for wider ground on both sides (tiles).
const CHOKE_REACH: i32 = 12;
/// Wall-in hint radius around a main base's choke (tiles).
const WALL_RADIUS: i32 = 4;

/// CHK `UNIT` record: "valid unit properties" bit for the resource amount (OpenBW map loader).
const CHK_PROP_RESOURCES: u16 = 0x10;

/// Neutral resources from the map's CHK `UNIT` sections (36-byte records: centre pixel at 4..8,
/// type at 8, valid properties at 14, resource amount u32 at 20). The amount is the initial state
/// of resources not yet seen (030 §2, E-3, AMB-030-3).
pub fn chk_resources(chk: &[u8]) -> Result<Vec<MapResource>> {
    let types = TypeTables::get();
    let mut out = vec![];
    let mut i = 0;
    while i + 8 <= chk.len() {
        let name = &chk[i..i + 4];
        let size = u32::from_le_bytes(chk[i + 4..i + 8].try_into().unwrap()) as usize;
        let body = &chk[i + 8..(i + 8 + size).min(chk.len())];
        if name == b"UNIT" {
            ensure!(size % 36 == 0, "CHK UNIT section size {size} is not a multiple of 36");
            for r in body.chunks_exact(36) {
                let x = u16::from_le_bytes([r[4], r[5]]);
                let y = u16::from_le_bytes([r[6], r[7]]);
                let ty = u16::from_le_bytes([r[8], r[9]]);
                let props = u16::from_le_bytes([r[14], r[15]]);
                let amount = if props & CHK_PROP_RESOURCES != 0 { u32::from_le_bytes([r[20], r[21], r[22], r[23]]) } else { 0 };
                if types.unit(ty).is_some_and(|t| t.is_resource()) {
                    out.push(MapResource { unit_type: ty, pos: (x, y), amount });
                }
            }
        }
        i += 8 + size;
    }
    Ok(out)
}

struct Grid {
    w: i32,
    h: i32,
}

impl Grid {
    fn idx(&self, x: i32, y: i32) -> Option<usize> {
        (x >= 0 && y >= 0 && x < self.w && y < self.h).then(|| (y * self.w + x) as usize)
    }
}

/// Map-agnostic precompute. The resource list is stored as is (030 §2, E-3).
pub fn precompute(map: &StaticMap, resources: &[MapResource]) -> MapPrecompute {
    let g = Grid { w: map.width_tiles as i32, h: map.height_tiles as i32 };
    let n = (g.w * g.h) as usize;
    let walkable: Vec<bool> = (0..n)
        .map(|i| {
            let (x, y) = ((i as i32 % g.w) as u16, (i as i32 / g.w) as u16);
            (0..4).all(|dy| (0..4).all(|dx| map.walkable.get(x * 4 + dx, y * 4 + dy)))
        })
        .collect();
    let types = TypeTables::get();

    // Resource footprints (top-left tile, w, h).
    let res: Vec<(i32, i32, i32, i32)> = resources
        .iter()
        .filter_map(|r| {
            let (ty, (px, py)) = (r.unit_type, r.pos);
            let t = types.unit(ty)?;
            Some(((px as i32 - t.dimensions[0]) / 32, (py as i32 - t.dimensions[1]) / 32, t.tile_width, t.tile_height))
        })
        .collect();
    let mut no_build = vec![false; n];
    let mut occupied = vec![false; n];
    for &(x, y, w, h) in &res {
        for ty in y - NO_BUILD..y + h + NO_BUILD {
            for tx in x - NO_BUILD..x + w + NO_BUILD {
                if let Some(i) = g.idx(tx, ty) {
                    no_build[i] = true;
                }
            }
        }
        for ty in y..y + h {
            for tx in x..x + w {
                if let Some(i) = g.idx(tx, ty) {
                    occupied[i] = true;
                }
            }
        }
    }

    // ---- bases: single-linkage clusters of resources, depot at the closest legal 4x3 spot.
    let mut cluster = vec![usize::MAX; res.len()];
    let mut clusters: Vec<Vec<usize>> = vec![];
    for i in 0..res.len() {
        if cluster[i] != usize::MAX {
            continue;
        }
        let id = clusters.len();
        let mut members = vec![i];
        cluster[i] = id;
        let mut k = 0;
        while k < members.len() {
            let a = res[members[k]];
            for j in 0..res.len() {
                let b = res[j];
                if cluster[j] == usize::MAX && (a.0 - b.0).abs().max((a.1 - b.1).abs()) <= CLUSTER_TILES {
                    cluster[j] = id;
                    members.push(j);
                }
            }
            k += 1;
        }
        clusters.push(members);
    }
    let fits = |x: i32, y: i32| {
        (0..3).all(|dy| (0..4).all(|dx| g.idx(x + dx, y + dy).is_some_and(|i| map.buildable.get((x + dx) as u16, (y + dy) as u16) && !no_build[i])))
    };
    let mut base_locs: Vec<Tile> = map.start_locations.clone();
    for members in clusters.iter().filter(|m| m.len() >= MIN_RESOURCES) {
        let (cx, cy) = members.iter().fold((0, 0), |(sx, sy), &m| (sx + res[m].0, sy + res[m].1));
        let (cx, cy) = (cx / members.len() as i32, cy / members.len() as i32);
        if base_locs.iter().any(|&(bx, by)| (bx as i32 - cx).abs() <= CLUSTER_TILES && (by as i32 - cy).abs() <= CLUSTER_TILES) {
            continue;
        }
        let mut best: Option<(i64, i32, i32)> = None;
        for y in cy - 14..=cy + 14 {
            for x in cx - 14..=cx + 14 {
                if !fits(x, y) {
                    continue;
                }
                let score: i64 = members
                    .iter()
                    .map(|&m| {
                        let (rx, ry) = (res[m].0 * 2 + res[m].2, res[m].1 * 2 + res[m].3);
                        let (dx, dy) = ((x * 2 + 4 - rx) as i64, (y * 2 + 3 - ry) as i64);
                        dx * dx + dy * dy
                    })
                    .sum();
                if best.is_none_or(|(s, bx, by)| (score, y, x) < (s, by, bx)) {
                    best = Some((score, x, y));
                }
            }
        }
        if let Some((_, x, y)) = best {
            base_locs.push((x as u16, y as u16));
        }
    }

    // ---- chokes: clearance (tile distance to the nearest unwalkable tile or resource).
    let mut clear = vec![i32::MAX; n];
    let mut q = VecDeque::new();
    for i in 0..n {
        if !walkable[i] || occupied[i] {
            clear[i] = 0;
            q.push_back(i);
        }
    }
    for y in 0..g.h {
        for x in 0..g.w {
            if (x == 0 || y == 0 || x == g.w - 1 || y == g.h - 1) && clear[(y * g.w + x) as usize] != 0 {
                clear[(y * g.w + x) as usize] = 1;
                q.push_back((y * g.w + x) as usize);
            }
        }
    }
    while let Some(i) = q.pop_front() {
        let (x, y) = (i as i32 % g.w, i as i32 / g.w);
        for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)] {
            if let Some(j) = g.idx(x + dx, y + dy) {
                if clear[j] > clear[i] + 1 {
                    clear[j] = clear[i] + 1;
                    q.push_back(j);
                }
            }
        }
    }
    // A choke centre: walkable, narrow, and narrower than open ground on two opposite sides.
    let mut chokes = vec![];
    for y in 0..g.h {
        for x in 0..g.w {
            let i = (y * g.w + x) as usize;
            let c = clear[i];
            if c == 0 || c > CHOKE_CLEARANCE {
                continue;
            }
            let wider = |dx: i32, dy: i32| {
                // Walk along the corridor; a wall ends the search.
                for k in 1..=CHOKE_REACH {
                    match g.idx(x + dx * k, y + dy * k) {
                        Some(j) if clear[j] > c + 1 => return true,
                        Some(j) if clear[j] > 0 => continue,
                        _ => return false,
                    }
                }
                false
            };
            let ridge = [(1, 0), (0, 1), (1, 1), (1, -1)].iter().any(|&(dx, dy)| {
                // Perpendicular neighbours are not wider: this is the corridor's middle line.
                let (px, py) = (-dy, dx);
                let mid = [g.idx(x + px, y + py), g.idx(x - px, y - py)].iter().all(|j| j.is_none_or(|j| clear[j] <= c));
                mid && wider(dx, dy) && wider(-dx, -dy)
            });
            if ridge {
                chokes.push((x as u16, y as u16));
            }
        }
    }

    // ---- wall-in hints: buildable tiles around the choke nearest to each start location.
    let mut wall = vec![];
    for &(sx, sy) in &map.start_locations {
        let Some(&(cx, cy)) = chokes.iter().min_by_key(|&&(x, y)| {
            let (dx, dy) = (x as i32 - sx as i32, y as i32 - sy as i32);
            (dx * dx + dy * dy, y, x)
        }) else {
            continue;
        };
        for y in cy as i32 - WALL_RADIUS..=cy as i32 + WALL_RADIUS {
            for x in cx as i32 - WALL_RADIUS..=cx as i32 + WALL_RADIUS {
                if g.idx(x, y).is_some_and(|i| map.buildable.get(x as u16, y as u16) && !occupied[i]) {
                    wall.push((x as u16, y as u16));
                }
            }
        }
    }
    MapPrecompute { version: PRECOMPUTE_VERSION, base_locs, chokes, wall_candidates: wall, resources: resources.to_vec() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uranium_proto::BitGrid;

    /// Two open fields joined by a 3-tile corridor; minerals and a geyser in the right field.
    fn two_rooms() -> (StaticMap, Vec<MapResource>) {
        let (w, h) = (64u16, 32u16);
        let mut walk = vec![0u8; w as usize * 4 * h as usize * 4];
        let mut build = vec![0u8; w as usize * h as usize];
        for ty in 0..h as usize {
            for tx in 0..w as usize {
                let open = (2..30).contains(&ty) && ((2..28).contains(&tx) || (36..62).contains(&tx)) || ((28..36).contains(&tx) && (15..18).contains(&ty));
                if open {
                    build[ty * w as usize + tx] = 1;
                    for dy in 0..4 {
                        for dx in 0..4 {
                            walk[(ty * 4 + dy) * w as usize * 4 + tx * 4 + dx] = 1;
                        }
                    }
                }
            }
        }
        let map = StaticMap {
            width_tiles: w,
            height_tiles: h,
            walkable: BitGrid::from_bytes(w * 4, h * 4, &walk),
            buildable: BitGrid::from_bytes(w, h, &build),
            ground_height: vec![0; w as usize * h as usize],
            start_locations: vec![(8, 14)],
        };
        let t = TypeTables::get();
        let mineral = t.unit_by_name("Resource_Mineral_Field").unwrap();
        let geyser = t.unit_by_name("Resource_Vespene_Geyser").unwrap();
        let mut res = vec![];
        for k in 0..6 {
            let pos = ((58 * 32 + mineral.dimensions[0]) as u16, ((8 + k * 2) * 32 + mineral.dimensions[1]) as u16);
            res.push(MapResource { unit_type: mineral.id as u16, pos, amount: 1500 });
        }
        let pos = ((50 * 32 + geyser.dimensions[0]) as u16, (4 * 32 + geyser.dimensions[1]) as u16);
        res.push(MapResource { unit_type: geyser.id as u16, pos, amount: 5000 });
        (map, res)
    }

    #[test]
    fn finds_base_choke_and_wall_hint_031() {
        let (map, res) = two_rooms();
        let p = precompute(&map, &res);
        assert!(p.base_locs.contains(&(8, 14)), "start location is a base");
        let base = *p.base_locs.iter().find(|&&b| b != (8, 14)).expect("expansion found");
        assert!((44..58).contains(&base.0) && (4..26).contains(&base.1), "expansion near the minerals: {base:?}");
        // Legal: not within 3 tiles of a resource.
        assert!(base.0 + 4 + 3 <= 58, "{base:?} too close to the mineral line");
        assert!(!p.chokes.is_empty());
        assert!(p.chokes.iter().all(|&(x, y)| (26..38).contains(&x) && (14..19).contains(&y)), "chokes off the corridor: {:?}", p.chokes);
        assert!(!p.wall_candidates.is_empty());
        assert_eq!(precompute(&map, &res), p, "deterministic");
        // 030 §2 / E-3: the resource list is kept, with the current file version.
        assert_eq!(p.resources, res);
        assert_eq!(p.version, PRECOMPUTE_VERSION);
    }

    #[test]
    fn chk_unit_section_resources_030_r2() {
        let t = TypeTables::get();
        let mineral = t.unit_by_name("Resource_Mineral_Field").unwrap().id as u16;
        let geyser = t.unit_by_name("Resource_Vespene_Geyser").unwrap().id as u16;
        let marine = t.unit_by_name("Terran_Marine").unwrap().id as u16;
        let rec = |ty: u16, x: u16, y: u16, props: u16, amount: u32| {
            let mut r = vec![0u8; 36];
            r[4..6].copy_from_slice(&x.to_le_bytes());
            r[6..8].copy_from_slice(&y.to_le_bytes());
            r[8..10].copy_from_slice(&ty.to_le_bytes());
            r[14..16].copy_from_slice(&props.to_le_bytes());
            r[20..24].copy_from_slice(&amount.to_le_bytes());
            r
        };
        // The amount counts only with the "resources valid" bit (0x10), as in OpenBW.
        let body: Vec<u8> = [rec(mineral, 100, 200, 0x10 | 0x2, 1500), rec(marine, 5, 5, 0x10, 9), rec(geyser, 300, 400, 0x2, 5000)].concat();
        let mut chk = b"VER \x02\x00\x00\x00\xcd\x00".to_vec();
        chk.extend_from_slice(b"UNIT");
        chk.extend_from_slice(&(body.len() as u32).to_le_bytes());
        chk.extend_from_slice(&body);
        assert_eq!(
            chk_resources(&chk).unwrap(),
            vec![MapResource { unit_type: mineral, pos: (100, 200), amount: 1500 }, MapResource { unit_type: geyser, pos: (300, 400), amount: 0 }]
        );
    }
}
