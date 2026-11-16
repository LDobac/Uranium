use std::collections::BTreeSet;

use uranium_proto::*;

use super::{BUNKER_SEARCH_MAX, Branch, CHOKE_MIN, ORDER_PLACE_BUILDING, ReactParams, TILE, bunker_center, depot_center, ground_path, toward, within};
use crate::brain::Brain;
use crate::types::*;
use crate::util::{dist2, done};

pub(crate) fn bunker_order_stale(p: &ReactParams, q: &crate::brain::Pending, f: &RawFrame, mine: &[&UnitState]) -> bool {
    q.ty == BUNKER
        && f.frame >= q.issued + p.bunker_order_expiry
        && !mine.iter().any(|u| u.unit_type == BUNKER && within(u.pos, bunker_center(q.tile), 10))
        && !mine.iter().any(|u| u.id == q.worker && u.order == ORDER_PLACE_BUILDING && u.order_pos.is_none_or(|p| within(p, bunker_center(q.tile), 3)))
}

pub(super) fn clear_of_units(f: &RawFrame, t: Tile) -> bool {
    let (w, h) = tile_size(BUNKER);
    let (x0, y0) = (t.0 as i32 * 32 - 8, t.1 as i32 * 32 - 8);
    let (x1, y1) = ((t.0 + w) as i32 * 32 + 8, (t.1 + h) as i32 * 32 + 8);
    let table = uranium_env::types::TypeTables::get();
    !f.units.iter().any(|u| {
        let ground = u.loaded_in.is_none() && !is_building(u.unit_type) && !is_resource(u.unit_type) && table.unit(u.unit_type).is_none_or(|t| !t.is_flyer());
        let (x, y) = (u.pos.0 as i32, u.pos.1 as i32);
        ground && x >= x0 && x < x1 && y >= y0 && y < y1
    })
}

impl Brain {
    pub(crate) fn bunker_site(&self, f: &RawFrame, center: Pos, to: Pos, reserved: &[(Tile, u16)]) -> Option<Tile> {
        let p = &self.react.as_ref()?.p;
        let (lo, hi, pad) = (p.bunker_min * TILE, p.bunker_max * TILE, p.bunker_resource_pad);
        let near = toward(center, to, (p.bunker_min + p.bunker_max) as f64 / 2.0);
        self.placer
            .find_padded(BUNKER, near, &f.units, reserved, p.bunker_max as i32 + 2, pad, |t| {
                let d2 = dist2(bunker_center(t), center);
                d2 >= lo * lo && d2 <= hi * hi && clear_of_units(f, t)
            })
            .or_else(|| self.placer.find_padded(BUNKER, near, &f.units, reserved, 2 * BUNKER_SEARCH_MAX as i32, pad, |t| within(bunker_center(t), center, BUNKER_SEARCH_MAX) && clear_of_units(f, t)))
    }

    pub(crate) fn main_bunker_site(&self, f: &RawFrame, center: Pos, to: Pos, reserved: &[(Tile, u16)]) -> Option<Tile> {
        let r = self.react.as_ref()?;
        let p = &r.p;
        let path = r.main_path.as_deref().unwrap_or_default();
        let near = p.path_near * TILE;
        let tile_center = |t: Tile| (t.0 * 32 + 16, t.1 * 32 + 16);
        let reach = p.path_near as i32 + 2;
        let candidates = |lo: i64, hi: i64| {
            let mut seen = BTreeSet::new();
            let mut cands = Vec::new();
            for &pt in path {
                let pc = tile_center(pt);
                let mut ring: Vec<(i64, Tile)> = Vec::new();
                for dy in -reach..=reach {
                    for dx in -reach..=reach {
                        let (x, y) = (pt.0 as i32 + dx - 1, pt.1 as i32 + dy);
                        if x < 0 || y < 0 {
                            continue;
                        }
                        let t = (x as u16, y as u16);
                        let c = bunker_center(t);
                        let d2 = dist2(c, center);
                        if dist2(c, pc) <= near * near && d2 >= lo * lo && d2 <= hi * hi {
                            ring.push((dist2(c, pc), t));
                        }
                    }
                }
                ring.sort();
                cands.extend(ring.into_iter().map(|(_, t)| t).filter(|t| seen.insert(*t)));
            }
            cands
        };
        self.placer
            .first_fit(BUNKER, candidates(p.path_min * TILE, p.path_max * TILE), &f.units, reserved, p.bunker_resource_pad, |t| clear_of_units(f, t))
            .or_else(|| self.bunker_site(f, center, to, reserved))
    }

    fn plan_main_path(&mut self, f: &RawFrame, base: Pos) {
        let Some(r) = self.react.as_ref() else { return };
        if r.main_path.is_some() {
            return;
        }
        let path = match r.natural {
            Some(n) => {
                let solid: BTreeSet<Tile> = f
                    .units
                    .iter()
                    .filter(|u| is_resource(u.unit_type))
                    .flat_map(|u| {
                        let (t, (w, h)) = (building_tile(u), tile_size(u.unit_type));
                        (t.1..t.1 + h).flat_map(move |y| (t.0..t.0 + w).map(move |x| (x, y)))
                    })
                    .collect();
                let nat = depot_center(n);
                ground_path(&self.walkable, (base.0 / 32, base.1 / 32), (nat.0 / 32, nat.1 / 32), &solid)
            }
            None => Vec::new(),
        };
        self.react.as_mut().expect("checked above").main_path = Some(path);
    }

    pub(crate) fn bunker_facing(&self, center: Pos, outer_of: Option<Pos>) -> Pos {
        let Some(r) = self.react.as_ref() else { return self.map_center };
        let tile = |t: &Tile| (t.0 * 32 + 16, t.1 * 32 + 16);
        let far = r.chokes.iter().map(tile).filter(|c| dist2(*c, center) >= (CHOKE_MIN * TILE).pow(2));
        let len = |a: Pos, b: Pos| (dist2(a, b) as f64).sqrt().round() as i64;
        match outer_of {
            Some(m) => far.filter(|c| dist2(*c, m) > dist2(center, m)).min_by_key(|c| (dist2(*c, center), *c)).unwrap_or(self.map_center),
            None => match r.natural.map(depot_center) {
                Some(via) => far.min_by_key(|c| (len(center, *c) + len(*c, via), dist2(*c, center), *c)).unwrap_or(via),
                None => far.min_by_key(|c| (dist2(*c, center), *c)).unwrap_or(self.map_center),
            },
        }
    }

    pub(crate) fn react_build(&mut self, f: &RawFrame, mine: &[&UnitState], base: Pos, busy: &[UnitId], cmds: &mut Vec<Command>, minerals: &mut i32) -> (i32, i32) {
        self.plan_main_path(f, base);
        let Some(r) = self.react.as_ref() else { return (0, 0) };
        let (p, branch, natural, main_known) = (r.p.clone(), r.branch, r.natural, r.enemy_main.is_some());
        let rax_done = mine.iter().any(|u| u.unit_type == BARRACKS && done(u));
        let bunker_near = |c: Pos| mine.iter().any(|u| u.unit_type == BUNKER && within(u.pos, c, 10)) || self.pending.iter().any(|q| q.ty == BUNKER && within(bunker_center(q.tile), c, 10));
        let reserved = |s: &Self| -> Vec<(Tile, u16)> { s.pending.iter().map(|q| (q.tile, q.ty)).chain(natural.map(|n| (n, CC))).collect() };
        if self.react.as_ref().is_some_and(|r| r.main_site.is_none()) {
            let (to, res) = (self.bunker_facing(base, None), reserved(self));
            let site = self.main_bunker_site(f, base, to, &res);
            self.react.as_mut().expect("checked above").main_site = site;
        }
        match branch {
            Branch::Defence => {
                let (dbase, to) = self.defence_base(mine, base);
                if bunker_near(dbase) {
                    return (0, 0);
                }
                if rax_done && *minerals >= cost(BUNKER) {
                    let res = reserved(self);
                    let site = if dbase == base { self.main_bunker_site(f, dbase, to, &res) } else { self.bunker_site(f, dbase, to, &res) };
                    if let Some(t) = site {
                        if self.order_build_at(BUNKER, t, f, mine, busy, cmds) {
                            *minerals -= cost(BUNKER);
                            return (0, 0);
                        }
                    }
                }
                (cost(BUNKER), 0)
            }
            Branch::Macro => {
                let Some(n) = natural else { return (0, 0) };
                if !self.natural_started(mine) {
                    if !rax_done || self.tp.is_some() {
                        return (0, 0);
                    }
                    let expand_ok = main_known || f.frame >= p.expand_hold_until;
                    if expand_ok && *minerals >= p.expand_minerals && self.order_build_at(CC, n, f, mine, busy, cmds) {
                        *minerals -= cost(CC);
                        if let Some(r) = self.react.as_mut() {
                            r.stats.expand_frame.get_or_insert(f.frame);
                        }
                        return (0, 0);
                    }
                    return (0, p.expand_minerals);
                }
                let nc = depot_center(n);
                let nat_done = mine.iter().any(|u| is_resource_depot(u.unit_type) && building_tile(u) == n && done(u));
                if nat_done && f.frame >= p.natural_bunker_from && rax_done && !bunker_near(nc) && *minerals >= cost(BUNKER) {
                    let to = self.bunker_facing(nc, self.home);
                    let res = reserved(self);
                    if let Some(t) = self.bunker_site(f, nc, to, &res) {
                        if self.order_build_at(BUNKER, t, f, mine, busy, cmds) {
                            *minerals -= cost(BUNKER);
                        }
                    }
                }
                (0, 0)
            }
        }
    }

    pub(crate) fn defence_base(&self, mine: &[&UnitState], main: Pos) -> (Pos, Pos) {
        let depots = Self::own_depots(mine);
        let chosen = self.react.as_ref().and_then(|r| r.def_base).and_then(|b| depots.iter().copied().find(|d| within(*d, b, 1)));
        match chosen {
            Some(b) if !within(b, main, 1) => (b, self.bunker_facing(b, Some(main))),
            _ => (main, self.bunker_facing(main, None)),
        }
    }

    pub(crate) fn order_build_at(&mut self, ty: u16, tile: Tile, f: &RawFrame, mine: &[&UnitState], busy: &[UnitId], cmds: &mut Vec<Command>) -> bool {
        let center = (tile.0 * 32 + 16, tile.1 * 32 + 16);
        let jobs: Vec<UnitId> = self.react_busy().into_iter().chain(self.pending.iter().map(|q| q.worker)).collect();
        let rings = self.tp.is_some();
        let worker = mine
            .iter()
            .filter(|u| u.unit_type == SCV && u.has(flags::COMPLETED) && !u.has(flags::CARRYING) && !u.has(flags::CONSTRUCTING) && !busy.contains(&u.id) && !jobs.contains(&u.id))
            .min_by_key(|u| (crate::tp::pick_key(rings, u.pos, center), u.id));
        let Some(w) = worker else { return false };
        cmds.push(Command::Build { worker: w.id, unit_type: ty, tile });
        self.pending.push(crate::brain::Pending { ty, tile, worker: w.id, issued: f.frame });
        true
    }
}
