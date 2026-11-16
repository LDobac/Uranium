use std::collections::BTreeSet;

use serde::Serialize;
use uranium_env::scout::{is_army_type, is_ground_army_type};
use uranium_proto::*;

use crate::types::*;
use crate::{Brain, dist2, done, is_neutral_owner, nearest};

pub use uranium_env::types::FRAMES_PER_SECOND as FPS;
const TILE: i64 = 32;
pub const BUNKER_HP: u16 = 350;
const CHOKE_MIN: i64 = 8;
const BUNKER_SEARCH_MAX: i64 = 9;
const ORDER_CONSTRUCTING: u16 = 33;
const ORDER_PLACE_BUILDING: u16 = 30;
const ORDER_MOVE: u16 = 6;
const ATTACK_ORDERS: [u16; 5] = [8, 9, 10, 11, 14];
const SCOUT_AWAY_TILES: i64 = 12;

pub(crate) fn bunker_order_stale(p: &ReactParams, q: &crate::Pending, f: &RawFrame, mine: &[&UnitState]) -> bool {
    q.ty == BUNKER
        && f.frame >= q.issued + p.bunker_order_expiry
        && !mine.iter().any(|u| u.unit_type == BUNKER && within(u.pos, bunker_center(q.tile), 10))
        && !mine.iter().any(|u| u.id == q.worker && u.order == ORDER_PLACE_BUILDING && u.order_pos.is_none_or(|p| within(p, bunker_center(q.tile), 3)))
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReactParams {
    pub first_depot_supply: u32,
    pub barracks_supply: u32,
    pub second_depot_supply: u32,
    pub pool_start_max: u32,
    pub pool_done_before: u32,
    pub lings_min: usize,
    pub lings_before: u32,
    pub lings_many: usize,
    pub lings_many_before: u32,
    pub army_radius: i64,
    pub army_min: usize,
    pub main_radius: i64,
    pub main_look: i64,
    pub main_drones_radius: i64,
    pub drones_max: usize,
    pub main_before: u32,
    pub one_base_from: u32,
    pub one_base_before: u32,
    pub judge_until: u32,
    pub expand_hold_until: u32,
    pub expand_minerals: i32,
    pub bunker_min: i64,
    pub bunker_max: i64,
    pub path_min: i64,
    pub path_max: i64,
    pub path_near: i64,
    pub bunker_resource_pad: i32,
    pub load_max: usize,
    pub pull_bunker: i64,
    pub pull_base: i64,
    pub pull_max: usize,
    pub pull_keep: usize,
    pub retarget: u32,
    pub harass_radius: i64,
    pub harass_guards: usize,
    pub harass_release: i64,
    pub repair_below_pct: u32,
    pub repairers: usize,
    pub marine_ratio: usize,
    pub marine_radius: i64,
    pub macro_marines: usize,
    pub macro_quiet: u32,
    pub macro_marines_any: usize,
    pub resume_after: u32,
    pub natural_bunker_from: u32,
    pub scout_look: i64,
    pub scout_look_max: u32,
    pub bunker_order_expiry: u32,
}

impl Default for ReactParams {
    fn default() -> Self {
        Self {
            first_depot_supply: 9,
            barracks_supply: 11,
            second_depot_supply: 15,
            pool_start_max: 95 * FPS,
            pool_done_before: 95 * FPS + 1_200,
            lings_min: 2,
            lings_before: 170 * FPS,
            lings_many: 6,
            lings_many_before: 180 * FPS,
            army_radius: 20,
            army_min: 2,
            main_radius: 10,
            main_look: 3,
            main_drones_radius: 12,
            drones_max: 6,
            main_before: 210 * FPS,
            one_base_from: 150 * FPS,
            one_base_before: 240 * FPS,
            judge_until: 480 * FPS,
            expand_hold_until: 220 * FPS,
            expand_minerals: 400,
            bunker_min: 4,
            bunker_max: 6,
            path_min: 5,
            path_max: 7,
            path_near: 2,
            bunker_resource_pad: 1,
            load_max: 4,
            pull_bunker: 3,
            pull_base: 5,
            pull_max: 6,
            pull_keep: 4,
            retarget: 48,
            harass_radius: 3,
            harass_guards: 2,
            harass_release: 6,
            repair_below_pct: 90,
            repairers: 2,
            marine_ratio: 2,
            marine_radius: 12,
            macro_marines: 8,
            macro_quiet: 60 * FPS,
            macro_marines_any: 12,
            resume_after: 48,
            natural_bunker_from: 240 * FPS,
            scout_look: 3,
            scout_look_max: 15 * FPS,
            bunker_order_expiry: 144,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Branch {
    #[default]
    Macro,
    Defence,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RushRule {
    PoolStart,
    PoolDone,
    Lings,
    Army,
    Drones,
    OneBase,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct ReactStats {
    pub main_frame: Option<u32>,
    pub rush_frame: Option<u32>,
    pub rush_rule: Option<RushRule>,
    pub defences: u32,
    pub macro_frame: Option<u32>,
    pub expand_frame: Option<u32>,
    pub branch: Branch,
    pub pool_start_frame: Option<u32>,
    pub pool_seen_frame: Option<u32>,
    pub lings_max: u32,
    pub contact_frame: Option<u32>,
    pub bunker_frame: Option<u32>,
    pub rule_frames: std::collections::BTreeMap<RushRule, u32>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MapHints {
    pub base_locs: Vec<Tile>,
    pub chokes: Vec<Tile>,
}

pub(crate) struct ReactState {
    pub p: ReactParams,
    pub branch: Branch,
    pub chokes: Vec<Tile>,
    pub natural: Option<Tile>,
    pub enemy_main: Option<Tile>,
    pub main_site: Option<Tile>,
    pub def_base: Option<Pos>,
    pub main_path: Option<Vec<Tile>>,
    pub last_contact: u32,
    pub pulled: Vec<UnitId>,
    pub repairers: Vec<UnitId>,
    pub holding: BTreeSet<UnitId>,
    pub marines_attack: bool,
    pub extractor_seen: bool,
    pub harass: Option<Harass>,
    pub orphan_since: std::collections::BTreeMap<UnitId, u32>,
    pub resumers: std::collections::BTreeMap<UnitId, (UnitId, u32)>,
    pub last_marines: Option<u32>,
    pub last_pull: Option<u32>,
    pub last_repair: Option<u32>,
    pub scout_out: bool,
    pub stats: ReactStats,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Harass {
    pub guards: Vec<UnitId>,
    pub target: UnitId,
    pub builder: UnitId,
    pub at: Pos,
    pub last_order: u32,
}

impl ReactState {
    pub fn new(p: ReactParams, hints: &MapHints, home: Option<Pos>, starts: &[Tile]) -> Self {
        let natural = home.and_then(|h| {
            hints
                .base_locs
                .iter()
                .copied()
                .filter(|b| !starts.iter().any(|s| (b.0 as i64 - s.0 as i64).abs() <= 4 && (b.1 as i64 - s.1 as i64).abs() <= 4))
                .min_by_key(|b| (dist2(depot_center(*b), h), *b))
        });
        Self {
            p,
            branch: Branch::Macro,
            chokes: hints.chokes.clone(),
            natural,
            enemy_main: None,
            main_site: None,
            def_base: None,
            main_path: None,
            last_contact: 0,
            pulled: Vec::new(),
            repairers: Vec::new(),
            holding: BTreeSet::new(),
            marines_attack: false,
            extractor_seen: false,
            harass: None,
            orphan_since: Default::default(),
            resumers: Default::default(),
            last_marines: None,
            last_pull: None,
            last_repair: None,
            scout_out: false,
            stats: ReactStats::default(),
        }
    }

    fn due(last: Option<u32>, frame: u32, every: u32) -> bool {
        last.is_none_or(|l| frame >= l + every)
    }

    pub(crate) fn due_on(aligned: bool, last: Option<u32>, frame: u32, every: u32) -> bool {
        if aligned {
            frame % every < crate::STEP_FRAMES && last.is_none_or(|l| frame / every != l / every)
        } else {
            Self::due(last, frame, every)
        }
    }
}

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

fn clear_of_units(f: &RawFrame, t: Tile) -> bool {
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

pub(crate) fn is_enemy(u: &UnitState, me: u8) -> bool {
    u.owner != me && !is_neutral_owner(u.owner)
}

pub fn pool_start_frame(frame: u32, hp: u16) -> u32 {
    frame.saturating_sub(u32::from(hp.saturating_sub(75)) * 16 / 9)
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

pub struct RuleContext<'a> {
    pub depots: &'a [Pos],
    pub starts: &'a [Tile],
    pub scout: Option<Pos>,
    pub known_hatcheries: &'a [Pos],
    pub extractor_seen: bool,
}

pub fn rush_rules(p: &ReactParams, f: &RawFrame, me: u8, cx: &RuleContext<'_>) -> Vec<RushRule> {
    let mut out = Vec::new();
    if f.frame >= p.judge_until {
        return out;
    }
    let enemies: Vec<&UnitState> = f.units.iter().filter(|u| is_enemy(u, me)).collect();
    let pools: Vec<&&UnitState> = enemies.iter().filter(|u| u.unit_type == SPAWNING_POOL).collect();
    if pools.iter().any(|u| !u.has(flags::COMPLETED) && pool_start_frame(f.frame, u.hp) <= p.pool_start_max) {
        out.push(RushRule::PoolStart);
    }
    if f.frame < p.pool_done_before && pools.iter().any(|u| u.has(flags::COMPLETED)) {
        out.push(RushRule::PoolDone);
    }
    let lings = enemies.iter().filter(|u| u.unit_type == ZERGLING).count();
    if (f.frame < p.lings_before && lings >= p.lings_min) || (f.frame < p.lings_many_before && lings >= p.lings_many) {
        out.push(RushRule::Lings);
    }
    let near_army = enemies.iter().filter(|u| is_army_type(u.unit_type) && cx.depots.iter().any(|d| within(u.pos, *d, p.army_radius))).count();
    if near_army >= p.army_min {
        out.push(RushRule::Army);
    }
    let start_hatchery = enemies.iter().filter(|u| is_resource_depot(u.unit_type)).find(|h| cx.starts.iter().any(|s| within(h.pos, depot_center(*s), p.main_radius)));
    if let (Some(h), Some(scout), false) = (start_hatchery, cx.scout, pools.is_empty()) {
        let only = cx.known_hatcheries.iter().all(|k| within(*k, h.pos, 2));
        if within(scout, h.pos, p.main_look) && only {
            let drones = enemies.iter().filter(|u| u.unit_type == DRONE && within(u.pos, h.pos, p.main_drones_radius)).count();
            if f.frame < p.main_before && drones <= p.drones_max {
                out.push(RushRule::Drones);
            }
            let pool_done = pools.iter().any(|u| u.has(flags::COMPLETED));
            if (p.one_base_from..p.one_base_before).contains(&f.frame) && pool_done && !cx.extractor_seen {
                out.push(RushRule::OneBase);
            }
        }
    }
    out
}

impl Brain {
    pub(crate) fn own_depots(mine: &[&UnitState]) -> Vec<Pos> {
        mine.iter().filter(|u| is_resource_depot(u.unit_type)).map(|u| u.pos).collect()
    }

    pub(crate) fn home_depot(&self, mine: &[&UnitState]) -> Option<Pos> {
        let depots = Self::own_depots(mine);
        match self.home {
            Some(h) => depots.into_iter().min_by_key(|d| (dist2(*d, h), *d)),
            None => depots.first().copied(),
        }
    }

    pub(crate) fn other_starts(&self) -> Vec<Tile> {
        let home = self.home;
        self.starts.iter().copied().filter(|s| Some(depot_center(*s)) != home).collect()
    }

    pub(crate) fn react_observe(&mut self, f: &RawFrame, mine: &[&UnitState]) {
        let others = self.other_starts();
        let me = self.me;
        let scout = self.scout.and_then(|id| mine.iter().find(|u| u.id == id)).map(|u| u.pos);
        let known_hatcheries: Vec<Pos> = self.enemy_buildings.values().filter(|(_, ty)| is_resource_depot(*ty)).map(|(pos, _)| *pos).collect();
        let home = self.home;
        let Some(r) = self.react.as_mut() else { return };
        let p = r.p.clone();
        if r.enemy_main.is_none() {
            let found = others.iter().copied().find(|s| self.enemy_buildings.values().any(|(pos, _)| within(*pos, depot_center(*s), p.main_radius)));
            if let Some(s) = found {
                r.enemy_main = Some(s);
                r.stats.main_frame = Some(f.frame);
            }
        }
        let depots = Self::own_depots(mine);
        let army: Vec<Pos> = f.units.iter().filter(|u| is_enemy(u, me) && is_army_type(u.unit_type)).map(|u| u.pos).collect();
        if army.iter().any(|a| depots.iter().any(|d| within(*a, *d, p.army_radius))) {
            r.last_contact = f.frame;
            r.stats.contact_frame.get_or_insert(f.frame);
        }
        for u in f.units.iter().filter(|u| is_enemy(u, me) && u.unit_type == SPAWNING_POOL) {
            r.stats.pool_seen_frame.get_or_insert(f.frame);
            if !u.has(flags::COMPLETED) && r.stats.pool_start_frame.is_none() {
                r.stats.pool_start_frame = Some(pool_start_frame(f.frame, u.hp));
            }
        }
        if f.frame < p.judge_until {
            let lings = f.units.iter().filter(|u| is_enemy(u, me) && u.unit_type == ZERGLING).count() as u32;
            r.stats.lings_max = r.stats.lings_max.max(lings);
        }
        if r.stats.bunker_frame.is_none() && mine.iter().any(|u| u.unit_type == BUNKER) {
            r.stats.bunker_frame = Some(f.frame);
        }
        r.extractor_seen |= f.units.iter().any(|u| is_enemy(u, me) && u.unit_type == EXTRACTOR);
        let rules = rush_rules(&p, f, me, &RuleContext { depots: &depots, starts: &others, scout, known_hatcheries: &known_hatcheries, extractor_seen: r.extractor_seen });
        for rule in &rules {
            r.stats.rule_frames.entry(*rule).or_insert(f.frame);
        }
        let marines = mine.iter().filter(|u| u.unit_type == MARINE && u.has(flags::COMPLETED)).count();
        if r.branch == Branch::Macro {
            if let Some(&rule) = rules.first().filter(|_| marines < p.macro_marines_any) {
                r.branch = Branch::Defence;
                r.last_contact = f.frame;
                r.stats.defences += 1;
                if r.stats.rush_frame.is_none() {
                    r.stats.rush_frame = Some(f.frame);
                    r.stats.rush_rule = Some(rule);
                }
                let main = home.and_then(|h| depots.iter().copied().min_by_key(|d| (dist2(*d, h), *d)));
                let natural = r.natural.and_then(|n| depots.iter().copied().find(|d| within(*d, depot_center(n), 1)));
                let by_army = army.iter().flat_map(|a| depots.iter().map(move |d| (dist2(*a, *d), *d))).min().map(|(_, d)| d);
                r.def_base = by_army.or(natural).or(main);
            }
        } else if (marines >= p.macro_marines && f.frame >= r.last_contact + p.macro_quiet) || marines >= p.macro_marines_any {
            r.branch = Branch::Macro;
            r.stats.macro_frame.get_or_insert(f.frame);
        }
        let outside = mine.iter().filter(|u| u.unit_type == MARINE && u.has(flags::COMPLETED) && u.loaded_in.is_none()).count();
        let close = f.units.iter().filter(|u| is_enemy(u, me) && is_ground_army_type(u.unit_type) && depots.iter().any(|d| within(u.pos, *d, p.marine_radius))).count();
        r.marines_attack = r.branch == Branch::Defence && close > 0 && outside >= p.marine_ratio * close;
        if r.marines_attack {
            r.holding.clear();
        }
        r.stats.branch = r.branch;
    }

    pub(crate) fn react_marines_attack(&self) -> bool {
        self.react.as_ref().is_some_and(|r| r.marines_attack)
    }

    pub(crate) fn react_guard_radius(&self) -> Option<i64> {
        self.react.as_ref().filter(|r| r.marines_attack).map(|r| r.p.marine_radius)
    }

    pub(crate) fn react_reserved_sites(&self) -> Vec<(Tile, u16)> {
        let mut out: Vec<(Tile, u16)> = self.react.as_ref().map(|r| r.main_site.map(|t| (t, BUNKER)).into_iter().chain(r.natural.map(|n| (n, CC))).collect()).unwrap_or_default();
        out.extend(self.tp_failed_sites());
        out
    }

    pub(crate) fn react_branch(&self) -> Option<Branch> {
        self.react.as_ref().map(|r| r.branch)
    }

    pub(crate) fn react_busy(&self) -> Vec<UnitId> {
        let mut out: Vec<UnitId> = self
            .react
            .as_ref()
            .map(|r| r.pulled.iter().chain(&r.repairers).chain(r.harass.iter().flat_map(|h| &h.guards)).copied().chain(r.resumers.values().map(|(scv, _)| *scv)).collect())
            .unwrap_or_default();
        out.extend(self.tp_gas_crew());
        out
    }

    pub(crate) fn react_supply_low(&self, f: &RawFrame, depots: usize) -> Option<bool> {
        let p = &self.react.as_ref()?.p;
        let used = f.self_state.supply_used as u32;
        match depots {
            0 => Some(used >= 2 * p.first_depot_supply),
            1 => Some(used >= 2 * p.second_depot_supply),
            _ => None,
        }
    }

    pub(crate) fn react_production(&self, f: &RawFrame, mine: &[&UnitState], prod_count: usize) -> Option<(usize, bool)> {
        let r = self.react.as_ref()?;
        let rax_done = mine.iter().any(|u| u.unit_type == BARRACKS && done(u));
        let cap = match r.branch {
            _ if self.tp.is_some() => self.tp_barracks_cap(mine),
            Branch::Defence => 2,
            Branch::Macro if self.natural_started(mine) || (r.natural.is_none() && rax_done) => 3,
            Branch::Macro => 1,
        };
        let ready = prod_count > 0 || f.self_state.supply_used as u32 >= 2 * r.p.barracks_supply;
        Some((cap, ready))
    }

    pub(crate) fn natural_started(&self, mine: &[&UnitState]) -> bool {
        let Some(n) = self.react.as_ref().and_then(|r| r.natural) else { return false };
        mine.iter().any(|u| is_resource_depot(u.unit_type) && building_tile(u) == n) || self.pending.iter().any(|p| p.ty == CC && p.tile == n)
    }

    pub(crate) fn react_scouting(&mut self, f: &RawFrame, mine: &[&UnitState], minerals: &[&UnitState], cmds: &mut Vec<Command>) {
        let Some(r) = self.react.as_ref() else { return };
        let (look, look_max, main, main_frame) = (r.p.scout_look, r.p.scout_look_max, r.enemy_main, r.stats.main_frame);
        let depot_ordered = self.pending.iter().any(|q| q.ty == DEPOT) || mine.iter().any(|u| matches!(u.unit_type, DEPOT | BARRACKS));
        if !self.scout_done && depot_ordered {
            self.pick_scout(f, mine);
        }
        let Some(id) = self.scout else { return };
        let Some(s) = mine.iter().find(|u| u.id == id) else { return };
        let target = match main {
            Some(m) if self.tp_scout_target(f, s, m).is_some() => self.tp_scout_target(f, s, m).expect("checked"),
            Some(m) => {
                let c = depot_center(m);
                let looked = within(s.pos, c, look) || main_frame.is_some_and(|mf| f.frame >= mf + look_max);
                (!looked).then_some(c)
            }
            None => self.starts.iter().zip(&self.scouted).filter(|(_, done)| !**done).map(|(t, _)| depot_center(*t)).min_by_key(|p| (dist2(*p, s.pos), *p)),
        };
        match target {
            Some(p) => {
                let settled = self.tp.is_some() && s.has(flags::IDLE) && within(s.pos, p, 3);
                if (ReactState::due_on(self.tp.is_some(), Some(self.scout_order), f.frame, crate::WAVE_PERIOD) || s.has(flags::IDLE)) && !settled {
                    cmds.push(Command::Unit { units: vec![id], order: CmdOrder::Move(Target::Pos(p)), queued: false });
                    self.scout_order = f.frame;
                }
            }
            None => {
                let home = self.home.unwrap_or(s.pos);
                if let Some(m) = nearest(home, minerals) {
                    cmds.push(Command::Unit { units: vec![id], order: CmdOrder::Gather(m.id), queued: false });
                }
                self.scout = None;
                self.scout_done = true;
            }
        }
    }

    fn pick_scout(&mut self, f: &RawFrame, mine: &[&UnitState]) {
        let Some(r) = self.react.as_ref() else { return };
        let (main_radius, was_out) = (r.p.main_radius, r.scout_out);
        let others: Vec<Pos> = self.other_starts().into_iter().map(depot_center).collect();
        let home = self.home;
        let out_of_main = |u: &UnitState| home.is_some_and(|h| !within(u.pos, h, SCOUT_AWAY_TILES));
        let heading_out = |u: &UnitState| {
            (u.order == ORDER_MOVE && u.order_pos.is_some_and(|p| others.iter().any(|c| within(p, *c, main_radius)))) || others.iter().any(|c| within(u.pos, *c, SCOUT_AWAY_TILES))
        };
        let current = self.scout.and_then(|id| mine.iter().copied().find(|u| u.id == id));
        if self.scout.is_some() && current.is_none() && was_out {
            self.scout = None;
            self.scout_done = true;
            return;
        }
        let just_ordered = !ReactState::due_on(self.tp.is_some(), Some(self.scout_order), f.frame, crate::WAVE_PERIOD);
        let keep = current.filter(|u| just_ordered || heading_out(u) || out_of_main(u)).map(|u| u.id);
        let jobs: Vec<UnitId> = self.pending.iter().map(|q| q.worker).chain(self.react_busy()).collect();
        let taken_over = || mine.iter().filter(|u| u.unit_type == SCV && u.has(flags::COMPLETED) && !u.has(flags::CONSTRUCTING) && !jobs.contains(&u.id) && heading_out(u)).min_by_key(|u| u.id).map(|u| u.id);
        let pick = match keep.or_else(taken_over) {
            Some(id) => Some(id),
            None => {
                self.scout = None;
                self.free_worker(mine)
            }
        };
        if pick != keep {
            self.scout_order = 0;
        }
        self.scout = pick;
        let out = pick.and_then(|id| mine.iter().copied().find(|u| u.id == id)).is_some_and(out_of_main);
        if let Some(r) = self.react.as_mut() {
            r.scout_out = out;
        }
    }

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

    pub(crate) fn react_reserved_marines(&self, mine: &[&UnitState]) -> Vec<UnitId> {
        match self.react_branch() {
            None => Vec::new(),
            Some(Branch::Defence) if !self.react_marines_attack() => mine.iter().filter(|u| u.unit_type == MARINE).map(|u| u.id).collect(),
            Some(_) => self.garrison(mine).into_iter().flat_map(|(_, ids)| ids).collect(),
        }
    }

    fn garrison(&self, mine: &[&UnitState]) -> Vec<(UnitId, Vec<UnitId>)> {
        let Some(r) = self.react.as_ref() else { return Vec::new() };
        let rings = self.tp.is_some();
        let mut taken: BTreeSet<UnitId> = BTreeSet::new();
        let mut out = Vec::new();
        let mut bunkers: Vec<&&UnitState> = mine.iter().filter(|u| u.unit_type == BUNKER && done(u)).collect();
        bunkers.sort_by_key(|b| b.id);
        for b in bunkers {
            let loaded = mine.iter().filter(|u| u.loaded_in == Some(b.id)).count();
            let want = r.p.load_max.saturating_sub(loaded);
            if want == 0 {
                continue;
            }
            let mut free: Vec<&&UnitState> = mine.iter().filter(|u| u.unit_type == MARINE && u.has(flags::COMPLETED) && u.loaded_in.is_none() && !taken.contains(&u.id)).collect();
            free.sort_by_key(|u| (crate::tp::pick_key(rings, u.pos, b.pos), u.id));
            let ids: Vec<UnitId> = free.iter().take(want).map(|u| u.id).collect();
            taken.extend(&ids);
            if !ids.is_empty() {
                out.push((b.id, ids));
            }
        }
        out
    }

    pub(crate) fn react_marines(&mut self, f: &RawFrame, mine: &[&UnitState], cmds: &mut Vec<Command>) {
        let aligned = self.tp.is_some();
        let Some(r) = self.react.as_mut() else { return };
        r.holding.retain(|id| mine.iter().any(|u| u.id == *id && u.loaded_in.is_none()));
        let (every, branch, last) = (r.p.retarget, r.branch, r.last_marines);
        if !ReactState::due_on(aligned, last, f.frame, every) {
            return;
        }
        let mut acted = false;
        for (bunker, ids) in self.garrison(mine) {
            cmds.push(Command::Unit { units: ids, order: CmdOrder::Load(bunker), queued: false });
            acted = true;
        }
        if branch == Branch::Defence && !self.react_marines_attack() {
            acted |= self.hold_marines(mine, cmds);
        }
        if acted {
            self.react.as_mut().expect("react kit").last_marines = Some(f.frame);
        }
    }

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
                bases.push(crate::centroid(&line));
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

    fn hold_marines(&mut self, mine: &[&UnitState], cmds: &mut Vec<Command>) -> bool {
        let Some(main) = self.home_depot(mine).or(self.home) else { return false };
        let (home, facing) = self.defence_base(mine, main);
        let loading: BTreeSet<UnitId> = self.garrison(mine).into_iter().flat_map(|(_, ids)| ids).collect();
        let bunker = mine.iter().filter(|u| u.unit_type == BUNKER && within(u.pos, home, 10)).min_by_key(|u| (dist2(u.pos, home), u.id)).map(|u| u.pos);
        let planned = self.pending.iter().find(|q| q.ty == BUNKER && within(bunker_center(q.tile), home, 10)).map(|q| bunker_center(q.tile));
        let spot = match bunker.or(planned) {
            Some(b) => toward(b, home, 2.0),
            None => toward(home, facing, 3.0),
        };
        let r = self.react.as_mut().expect("react kit");
        let marines: Vec<&&UnitState> = mine.iter().filter(|u| u.unit_type == MARINE && u.has(flags::COMPLETED) && u.loaded_in.is_none() && !loading.contains(&u.id)).collect();
        let (mut walk, mut hold) = (Vec::new(), Vec::new());
        for m in marines {
            if within(m.pos, spot, 3) {
                if r.holding.insert(m.id) {
                    hold.push(m.id);
                }
            } else {
                r.holding.remove(&m.id);
                walk.push(m.id);
            }
        }
        for chunk in walk.chunks(12) {
            cmds.push(Command::Unit { units: chunk.to_vec(), order: CmdOrder::Move(Target::Pos(spot)), queued: false });
        }
        for chunk in hold.chunks(12) {
            cmds.push(Command::Unit { units: chunk.to_vec(), order: CmdOrder::Hold, queued: false });
        }
        !(walk.is_empty() && hold.is_empty())
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
        self.pending.push(crate::Pending { ty, tile, worker: w.id, issued: f.frame });
        true
    }

    pub(crate) fn teacher_stats(&self) -> Option<ReactStats> {
        self.react.as_ref().map(|r| r.stats.clone())
    }
}

#[cfg(feature = "testkit")]
pub mod testkit {
    use super::*;
    use crate::{Bot, Kit, T0Build};

    pub const OURS: Tile = (10, 10);
    pub const THEIRS: Tile = (110, 110);

    pub fn info() -> GameInfo {
        let n = 128usize * 128;
        GameInfo {
            map_name: "t".into(),
            map_hash: [0; 32],
            map_size_tiles: (128, 128),
            self_player: 0,
            players: vec![
                PlayerInfo { id: 0, name: "a".into(), race: Race::Terran, start_tile: Some(OURS), is_self: true, is_enemy: false },
                PlayerInfo { id: 1, name: "b".into(), race: Race::Zerg, start_tile: None, is_self: false, is_enemy: true },
            ],
            static_map: StaticMap {
                width_tiles: 128,
                height_tiles: 128,
                walkable: BitGrid::from_bytes(512, 512, &vec![1; n * 16]),
                buildable: BitGrid::from_bytes(128, 128, &vec![1; n]),
                ground_height: vec![0; n],
                start_locations: vec![OURS, THEIRS],
            },
            latency_frames: 0,
        }
    }

    pub fn unit(id: u32, ty: u16, pos: Pos) -> UnitState {
        UnitState { id, owner: 0, unit_type: ty, pos, hp: 40, flags: flags::COMPLETED | flags::IDLE, ..Default::default() }
    }

    pub fn frame(n: u32, units: Vec<UnitState>, minerals: i32, supply_used: u16) -> RawFrame {
        RawFrame {
            frame: n,
            self_state: PlayerState { minerals, supply_used, supply_total: 60, upgrades: vec![0; 63], ..Default::default() },
            units,
            visibility: BitGrid::from_bytes(128, 128, &[1; 128 * 128]),
            explored: BitGrid::new(128, 128),
            creep: BitGrid::new(128, 128),
            events: vec![],
            effects: vec![],
        }
    }

    pub struct BunkerProbe {
        pub path: Vec<Tile>,
        pub natural: Pos,
        pub site: Option<Tile>,
        pub bunker: Option<(f64, bool)>,
    }

    pub fn main_bunker_probe(static_map: &StaticMap, hints: &MapHints, start: Tile, resources: &[(u16, Pos)]) -> BunkerProbe {
        let mut i = info();
        i.static_map = static_map.clone();
        i.map_size_tiles = (i.static_map.width_tiles, i.static_map.height_tiles);
        i.players[0].start_tile = Some(start);
        let mut b = Brain::with_hints(&i, Kit::for_build(Race::Terran, T0Build::TerranReact), hints);
        let h = depot_center(start);
        let mut units = vec![unit(1, CC, h)];
        for (k, r) in resources.iter().enumerate().filter(|(_, r)| within(r.1, h, 14)) {
            units.push(UnitState { owner: 11, ..unit(1_000 + k as u32, r.0, r.1) });
        }
        b.act(&frame(100, units, 0, 8));
        let r = b.react.as_ref().unwrap();
        let (path, site, natural) = (r.main_path.clone().unwrap(), r.main_site, depot_center(r.natural.unwrap()));
        let bunker = site.map(|s| {
            let c = bunker_center(s);
            let d = (dist2(c, h) as f64).sqrt() / 32.0;
            let on_path = path.iter().any(|t| within(c, (t.0 * 32 + 16, t.1 * 32 + 16), 2));
            (d, on_path)
        });
        BunkerProbe { path, natural, site, bunker }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Bot, Kit, T0Build};

    use super::testkit::*;

    const NATURAL: Tile = (10, 26);

    fn hints() -> MapHints {
        MapHints { base_locs: vec![OURS, THEIRS, NATURAL, (60, 60)], chokes: vec![(24, 10), (10, 40), (100, 100)] }
    }

    fn teacher() -> Brain {
        Brain::with_hints(&info(), Kit::for_build(Race::Terran, T0Build::TerranReact), &hints())
    }

    fn enemy(id: u32, ty: u16, pos: Pos) -> UnitState {
        UnitState { owner: 1, ..unit(id, ty, pos) }
    }

    fn home() -> Pos {
        depot_center(OURS)
    }

    fn base(scvs: u32) -> Vec<UnitState> {
        let mut v = vec![
            unit(1, CC, home()),
            unit(2, BARRACKS, (home().0 + 200, home().1 + 160)),
            UnitState { owner: 11, ..unit(3, 176, (home().0 - 200, home().1)) },
            unit(6, DEPOT, (home().0 - 48, home().1 - 160)),
            unit(7, DEPOT, (home().0 + 80, home().1 - 160)),
        ];
        v.extend((0..scvs).map(|i| unit(100 + i, SCV, (home().0 - 100, home().1 + 10 * i as u16))));
        v
    }

    fn builds(act: &Act, ty: u16) -> Vec<Tile> {
        act.cmds
            .iter()
            .filter_map(|c| match c {
                Command::Build { unit_type, tile, .. } if *unit_type == ty => Some(*tile),
                _ => None,
            })
            .collect()
    }

    fn trains(act: &Act, ty: u16) -> usize {
        act.cmds.iter().filter(|c| matches!(c, Command::Train { unit_type, .. } if *unit_type == ty)).count()
    }

    fn unit_orders(act: &Act, pick: impl Fn(&CmdOrder) -> bool) -> Vec<Vec<UnitId>> {
        act.cmds
            .iter()
            .filter_map(|c| match c {
                Command::Unit { units, order, .. } if pick(order) => Some(units.clone()),
                _ => None,
            })
            .collect()
    }

    fn branch(b: &Brain) -> Branch {
        b.react_branch().unwrap()
    }

    #[test]
    fn order_ids_match_the_order_table_013_r1() {
        use uranium_env::types::order_id;
        assert_eq!(order_id("ConstructingBuilding"), Some(ORDER_CONSTRUCTING));
        assert_eq!(order_id("PlaceBuilding"), Some(ORDER_PLACE_BUILDING));
        assert_eq!(order_id("Move"), Some(ORDER_MOVE));
        assert_eq!(["Attack1", "Attack2", "AttackUnit", "AttackFixedRange", "AttackMove"].map(|n| order_id(n).unwrap()), ATTACK_ORDERS);
    }

    #[test]
    fn rush_rules_021_r8() {
        let p = ReactParams::default();
        let depots = [home()];
        let others = [THEIRS];
        let none: [Pos; 0] = [];
        let cx = RuleContext { depots: &depots, starts: &others, scout: None, known_hatcheries: &none, extractor_seen: false };
        let at = |n: u32, units: Vec<UnitState>| rush_rules(&p, &frame(n, units, 0, 0), 0, &cx);
        let far = depot_center((60, 60));
        let building = |id: u32, hp: u16| UnitState { hp, flags: flags::BEING_CONSTRUCTED, ..enemy(id, SPAWNING_POOL, far) };
        assert_eq!((pool_start_frame(2_600, 188), pool_start_frame(2_600, 187), pool_start_frame(100, 750)), (2_400, 2_401, 0));
        assert_eq!((pool_start_frame(2_480, 188), pool_start_frame(2_480, 187)), (2_280, 2_281));
        assert_eq!(at(2_480, vec![building(9, 188)]), vec![RushRule::PoolStart]);
        assert_eq!(at(2_480, vec![building(9, 187)]), vec![]);
        assert_eq!(at(2_600, vec![building(9, 188)]), vec![], "started at 1:40");
        assert_eq!(p.pool_done_before, p.pool_start_max + 1_200);
        assert_eq!(at(3_479, vec![enemy(9, SPAWNING_POOL, far)]), vec![RushRule::PoolDone]);
        assert_eq!(at(3_480, vec![enemy(9, SPAWNING_POOL, far)]), vec![]);
        assert_eq!(at(3_576, vec![enemy(9, SPAWNING_POOL, far)]), vec![], "T0 zerg_muta's pool seen at 2:29");
        let finished = UnitState { hp: 750, ..enemy(9, SPAWNING_POOL, far) };
        assert_eq!(at(3_480, vec![finished.clone()]), vec![]);
        assert_eq!(at(3_000, vec![finished]), vec![RushRule::PoolDone]);
        let lings = |n: u32| (0..n).map(|i| enemy(30 + i, ZERGLING, far)).collect::<Vec<_>>();
        assert_eq!(at(4_079, lings(2)), vec![RushRule::Lings]);
        assert_eq!(at(4_079, lings(1)), vec![]);
        assert_eq!(at(4_080, lings(2)), vec![]);
        assert_eq!(at(4_319, lings(6)), vec![RushRule::Lings]);
        assert_eq!(at(4_319, lings(5)), vec![]);
        assert_eq!(at(4_320, lings(6)), vec![]);
        let near = |id: u32, ty: u16| enemy(id, ty, (home().0 + 19 * 32, home().1 + id as u16));
        assert_eq!(at(9_000, vec![near(9, HYDRALISK), near(10, ZERGLING)]), vec![RushRule::Army]);
        assert_eq!(at(9_000, vec![near(9, ZERGLING)]), vec![]);
        assert_eq!(at(9_000, vec![near(9, ZERGLING), enemy(10, ZERGLING, (home().0 + 21 * 32, home().1))]), vec![]);
        assert_eq!(at(9_000, vec![near(9, OVERLORD), near(10, DRONE), near(11, ZERGLING)]), vec![]);
        assert_eq!(at(11_520, vec![near(9, ZERGLING), near(10, ZERGLING)]), vec![]);
        let main = depot_center(THEIRS);
        let mut v = vec![enemy(9, HATCHERY, main), UnitState { hp: 700, flags: flags::BEING_CONSTRUCTED, ..enemy(10, SPAWNING_POOL, (main.0 + 160, main.1)) }];
        v.extend((0..6).map(|i| enemy(20 + i, DRONE, (main.0 - 100, main.1 + i as u16 * 8))));
        let known = [main];
        let two = [main, depot_center(NATURAL)];
        let scout_at = |tiles: u16| Some((main.0 - tiles * 32, main.1));
        let main_at = |n: u32, units: Vec<UnitState>, scout: Option<Pos>, known: &[Pos]| {
            rush_rules(&p, &frame(n, units, 0, 0), 0, &RuleContext { depots: &depots, starts: &others, scout, known_hatcheries: known, extractor_seen: false })
        };
        assert_eq!(main_at(4_000, v.clone(), scout_at(3), &known), vec![RushRule::Drones]);
        assert_eq!(main_at(4_000, v.clone(), scout_at(4), &known), vec![], "scout 4 tiles away");
        assert_eq!(main_at(4_000, v.clone(), None, &known), vec![], "no scout");
        assert_eq!(main_at(4_000, v.clone(), scout_at(3), &two), vec![], "a second hatchery");
        assert_eq!(main_at(5_040, v.clone(), scout_at(3), &known), vec![], "after 3:30");
        let mut seven = v.clone();
        seven.push(enemy(40, DRONE, (main.0 - 100, main.1 + 90)));
        assert_eq!(main_at(4_000, seven, scout_at(3), &known), vec![], "7 drones");
        assert_eq!(main_at(4_000, v[..1].to_vec(), scout_at(3), &known), vec![], "no pool");
        let mut w = vec![enemy(9, HATCHERY, main), enemy(10, SPAWNING_POOL, (main.0 + 160, main.1))];
        w.extend((0..12).map(|i| enemy(20 + i, DRONE, (main.0 - 100, main.1 + i as u16 * 8))));
        let one = |n: u32, units: Vec<UnitState>, scout: Option<Pos>, known: &[Pos], extractor_seen: bool| {
            rush_rules(&p, &frame(n, units, 0, 0), 0, &RuleContext { depots: &depots, starts: &others, scout, known_hatcheries: known, extractor_seen })
        };
        assert_eq!(one(3_600, w.clone(), scout_at(3), &known, false), vec![RushRule::OneBase]);
        assert_eq!(one(5_759, w.clone(), scout_at(3), &known, false), vec![RushRule::OneBase]);
        assert_eq!(one(3_599, w.clone(), scout_at(3), &known, false), vec![], "before 2:30");
        assert_eq!(one(3_479, w.clone(), scout_at(3), &known, false), vec![RushRule::PoolDone], "before 2:25: ①b");
        assert_eq!(one(5_760, w.clone(), scout_at(3), &known, false), vec![], "from 4:00");
        assert_eq!(one(4_000, w.clone(), scout_at(4), &known, false), vec![], "scout 4 tiles away");
        assert_eq!(one(4_000, w.clone(), None, &known, false), vec![], "no scout");
        assert_eq!(one(4_000, w.clone(), scout_at(3), &two, false), vec![], "a second hatchery");
        assert_eq!(one(4_000, w.clone(), scout_at(3), &known, true), vec![], "an extractor seen");
        assert_eq!(one(4_000, v.clone(), scout_at(3), &known, false), vec![RushRule::Drones], "unfinished pool: ④′ only");
        assert_eq!(one(4_000, w[..1].to_vec(), scout_at(3), &known, false), vec![], "no pool");
    }

    #[test]
    fn one_base_rule_remembers_extractors_021_r8() {
        let main = depot_center(THEIRS);
        let scout_units = || -> Vec<UnitState> {
            let mut u = base(12);
            u.push(UnitState { flags: flags::COMPLETED | flags::MOVING, ..unit(150, SCV, (main.0 - 64, main.1)) });
            u.push(enemy(9, HATCHERY, main));
            u.push(enemy(10, SPAWNING_POOL, (main.0 + 160, main.1)));
            u
        };
        let mut b = teacher();
        b.scout = Some(150);
        b.scout_done = false;
        let mut first = scout_units();
        first.push(enemy(11, EXTRACTOR, (main.0 + 200, main.1 + 200)));
        b.act(&frame(3_500, first, 0, 24));
        b.act(&frame(3_700, scout_units(), 0, 24));
        assert!(!b.teacher_stats().unwrap().rule_frames.contains_key(&RushRule::OneBase));
        let mut c = teacher();
        c.scout = Some(150);
        c.act(&frame(3_700, scout_units(), 0, 24));
        assert_eq!(c.teacher_stats().unwrap().rule_frames.get(&RushRule::OneBase), Some(&3_700));
        assert_eq!(branch(&c), Branch::Defence);
    }

    #[test]
    fn defence_at_the_natural_and_record_021_r8() {
        let mut b = teacher();
        let nat = depot_center(NATURAL);
        let mut units = base(16);
        units.push(unit(4, CC, nat));
        units.push(UnitState { hp: 300, flags: flags::BEING_CONSTRUCTED, ..enemy(70, SPAWNING_POOL, depot_center((60, 60))) });
        b.act(&frame(3_000, units.clone(), 0, 40));
        assert_eq!(branch(&b), Branch::Macro, "pool started at {}", pool_start_frame(3_000, 300));
        units.push(enemy(90, ZERGLING, (nat.0, nat.1 + 8 * 32)));
        units.push(enemy(91, ZERGLING, (nat.0 + 20, nat.1 + 8 * 32)));
        let act = b.act(&frame(6_000, units, 100, 40));
        assert_eq!(branch(&b), Branch::Defence);
        let t = builds(&act, BUNKER);
        assert_eq!(t.len(), 1, "{:?}", act.cmds);
        assert!(within(bunker_center(t[0]), nat, 10) && bunker_center(t[0]).1 > nat.1, "{t:?}");
        let s = b.teacher_stats().unwrap();
        assert_eq!((s.pool_start_frame, s.pool_seen_frame, s.lings_max, s.contact_frame, s.rush_rule), (Some(3_000 - 400), Some(3_000), 2, Some(6_000), Some(RushRule::Army)));
        assert_eq!(s.rule_frames.get(&RushRule::Army), Some(&6_000));
    }

    #[test]
    fn defence_base_fallbacks_021_r8() {
        let nat = depot_center(NATURAL);
        let mut units = base(16);
        units.push(unit(4, CC, nat));
        let mut b = teacher();
        let mut u = units.clone();
        u.push(enemy(70, SPAWNING_POOL, depot_center((60, 60))));
        let act = b.act(&frame(3_000, u, 100, 40));
        assert_eq!(b.teacher_stats().unwrap().rush_rule, Some(RushRule::PoolDone));
        let t = builds(&act, BUNKER);
        assert!(t.len() == 1 && within(bunker_center(t[0]), nat, 10) && !within(bunker_center(t[0]), home(), 10), "{t:?} {:?}", act.cmds);

        let mut b = teacher();
        let mut u = units;
        u.push(enemy(90, ZERGLING, (home().0 + 5 * 32, home().1 - 3 * 32)));
        u.push(enemy(91, ZERGLING, (home().0 + 6 * 32, home().1 - 3 * 32)));
        let act = b.act(&frame(6_000, u, 100, 40));
        assert_eq!(b.teacher_stats().unwrap().rush_rule, Some(RushRule::Army));
        let t = builds(&act, BUNKER);
        assert!(t.len() == 1 && within(bunker_center(t[0]), home(), 10), "{t:?} {:?}", act.cmds);
    }

    #[test]
    fn branch_switches_021_r8() {
        let mut b = teacher();
        b.act(&frame(2_000, base(12), 0, 24));
        assert_eq!(branch(&b), Branch::Macro);
        let mut units = base(12);
        units.push(enemy(90, ZERGLING, depot_center((60, 60))));
        units.push(enemy(91, ZERGLING, depot_center((60, 60))));
        let act = b.act(&frame(2_008, units, 100, 24));
        assert_eq!(branch(&b), Branch::Defence);
        let bunker = builds(&act, BUNKER);
        assert_eq!(bunker.len(), 1, "{:?}", act.cmds);
        let c = bunker_center(bunker[0]);
        let d = ((dist2(c, home()) as f64).sqrt() / 32.0, c.1 as i64 - home().1 as i64);
        assert!(d.0 >= 5.0 && d.0 <= 7.0 && d.1 > 0, "{bunker:?} {d:?}");
        let marines = |n: u32| (0..n).map(|i| unit(200 + i, MARINE, (home().0 + 100, home().1 + 4 * i as u16))).collect::<Vec<_>>();
        let mut u = base(12);
        u.extend(marines(7));
        b.act(&frame(2_008 + 1_400, u, 0, 40));
        assert_eq!(branch(&b), Branch::Defence);
        let mut u = base(12);
        u.extend(marines(8));
        b.act(&frame(2_008 + 1_439, u.clone(), 0, 40));
        assert_eq!(branch(&b), Branch::Defence);
        b.act(&frame(2_008 + 1_440, u.clone(), 0, 40));
        assert_eq!(branch(&b), Branch::Macro);
        u.push(enemy(92, ZERGLING, (home().0 + 300, home().1)));
        u.push(enemy(93, ZERGLING, (home().0 + 330, home().1)));
        b.act(&frame(5_000, u, 0, 40));
        assert_eq!(branch(&b), Branch::Defence);
        let s = b.teacher_stats().unwrap();
        assert_eq!((s.rush_frame, s.rush_rule, s.defences, s.macro_frame, s.branch), (Some(2_008), Some(RushRule::Lings), 2, Some(3_448), Branch::Defence));
    }

    #[test]
    fn bunker_site_skips_units_in_the_way_021_r8() {
        let b = teacher();
        let mut f = frame(3_000, base(10), 0, 40);
        let to = b.bunker_facing(home(), None);
        let first = b.bunker_site(&f, home(), to, &[]).unwrap();
        f.units.push(UnitState { flags: flags::COMPLETED, ..unit(300, OVERLORD, bunker_center(first)) });
        assert_eq!(b.bunker_site(&f, home(), to, &[]), Some(first), "a flyer does not block");
        f.units.push(enemy(301, ZERGLING, bunker_center(first)));
        let second = b.bunker_site(&f, home(), to, &[]).unwrap();
        assert_ne!(second, first);
        let d = (dist2(bunker_center(second), home()) as f64).sqrt() / 32.0;
        assert!((4.0..=6.0).contains(&d), "{second:?} {d}");
        assert!(clear_of_units(&f, second) && !clear_of_units(&f, first));
    }

    #[test]
    fn macro_expands_to_the_natural_021_r8() {
        let mut b = teacher();
        let mut units = base(14);
        units.push(enemy(80, HATCHERY, depot_center(THEIRS)));
        let act = b.act(&frame(3_000, units.clone(), 350, 30));
        assert!(b.react.as_ref().unwrap().enemy_main.is_some());
        assert_eq!((builds(&act, CC).len(), trains(&act, MARINE)), (0, 0), "{:?}", act.cmds);
        let units: Vec<UnitState> = units.into_iter().filter(|u| u.owner == 0).collect();
        let act = b.act(&frame(3_008, units, 400, 30));
        assert_eq!(builds(&act, CC), vec![NATURAL], "{:?}", act.cmds);
        assert_eq!(b.teacher_stats().unwrap().expand_frame, Some(3_008));

        let mut b = teacher();
        let act = b.act(&frame(5_279, base(14), 600, 30));
        assert_eq!((builds(&act, CC).len(), builds(&act, BUNKER).len()), (0, 0), "{:?}", act.cmds);
        let act = b.act(&frame(5_280, base(14), 600, 30));
        assert_eq!(builds(&act, CC), vec![NATURAL], "{:?}", act.cmds);
    }

    #[test]
    fn natural_bunker_not_before_four_minutes_021_r8() {
        let mut units = base(16);
        units.push(unit(4, CC, depot_center(NATURAL)));
        let mut b = teacher();
        let act = b.act(&frame(5_759, units.clone(), 500, 40));
        assert!(builds(&act, BUNKER).is_empty(), "{:?}", act.cmds);
        let act = b.act(&frame(5_760, units.clone(), 500, 40));
        let t = builds(&act, BUNKER);
        assert_eq!(t.len(), 1, "{:?}", act.cmds);
        assert!(bunker_center(t[0]).1 > depot_center(NATURAL).1, "{t:?}");
        let mut b = teacher();
        let last = units.len() - 1;
        units[last].flags = flags::BEING_CONSTRUCTED;
        let act = b.act(&frame(6_000, units, 500, 40));
        assert!(builds(&act, BUNKER).is_empty(), "{:?}", act.cmds);
    }

    #[test]
    fn defence_loads_four_and_holds_the_rest_021_r8() {
        let mut b = teacher();
        let mut units = base(12);
        let bunker = (home().0 + 5 * 32, home().1);
        units.push(UnitState { hp: BUNKER_HP, ..unit(5, BUNKER, bunker) });
        units.extend((0..6).map(|i| unit(200 + i, MARINE, (home().0 + 40, home().1 + 100 + 8 * i as u16))));
        units.push(enemy(90, ZERGLING, depot_center((60, 60))));
        units.push(enemy(91, ZERGLING, depot_center((60, 60))));
        let act = b.act(&frame(3_000, units.clone(), 0, 40));
        assert_eq!(branch(&b), Branch::Defence);
        let load = unit_orders(&act, |o| *o == CmdOrder::Load(5));
        assert_eq!(load.iter().map(Vec::len).collect::<Vec<_>>(), vec![4], "{:?}", act.cmds);
        let walk: usize = unit_orders(&act, |o| matches!(o, CmdOrder::Move(_))).iter().map(Vec::len).sum();
        assert_eq!(walk, 2, "{:?}", act.cmds);
        let spot = toward(bunker, home(), 2.0);
        for u in units.iter_mut().filter(|u| u.unit_type == MARINE) {
            u.pos = spot;
        }
        units.iter_mut().filter(|u| u.unit_type == MARINE).take(2).for_each(|u| u.loaded_in = Some(5));
        let act = b.act(&frame(3_048, units, 0, 40));
        let loads: usize = unit_orders(&act, |o| *o == CmdOrder::Load(5)).iter().map(Vec::len).sum();
        let holds: usize = unit_orders(&act, |o| *o == CmdOrder::Hold).iter().map(Vec::len).sum();
        assert_eq!((loads, holds), (2, 2), "{:?}", act.cmds);
        assert!(unit_orders(&act, |o| matches!(o, CmdOrder::Attack(Target::Pos(_)))).is_empty(), "no waves in D: {:?}", act.cmds);
    }

    #[test]
    fn defence_pulls_repairs_and_releases_scvs_021_r8() {
        let mut b = teacher();
        b.scout_done = true;
        let mut units = base(20);
        let bunker = (home().0 + 5 * 32, home().1 + 64);
        units.push(UnitState { hp: 300, ..unit(5, BUNKER, bunker) });
        units.push(enemy(90, ZERGLING, (bunker.0, bunker.1 + 64)));
        units.push(enemy(91, ZERGLING, (home().0 - 150, home().1 + 40)));
        units.push(enemy(92, ZERGLING, (home().0 + 2 * 32, home().1 + 6 * 32 - 20)));
        let act = b.act(&frame(3_000, units.clone(), 0, 40));
        assert_eq!(branch(&b), Branch::Defence);
        let attack: Vec<(Vec<UnitId>, UnitId)> = act
            .cmds
            .iter()
            .filter_map(|c| match c {
                Command::Unit { units, order: CmdOrder::Attack(Target::Unit(t)), .. } => Some((units.clone(), *t)),
                _ => None,
            })
            .collect();
        assert_eq!(attack.len(), 1, "{:?}", act.cmds);
        assert_eq!((attack[0].0.len(), attack[0].1), (6, 90), "min(6, 20 - 4) onto the zergling nearest the bunker");
        let repair = unit_orders(&act, |o| *o == CmdOrder::Repair(5));
        assert_eq!(repair.iter().map(Vec::len).collect::<Vec<_>>(), vec![2], "{:?}", act.cmds);
        assert!(repair[0].iter().all(|id| !attack[0].0.contains(id)));
        let pulled = attack[0].0.clone();
        assert!(unit_orders(&act, |o| matches!(o, CmdOrder::Gather(_))).iter().all(|g| g.iter().all(|u| !pulled.contains(u))));

        let mut away: Vec<UnitState> = units.iter().filter(|u| u.owner != 1 || u.id == 92).cloned().collect();
        for u in away.iter_mut().filter(|u| u.unit_type == BUNKER) {
            u.hp = BUNKER_HP;
        }
        let act = b.act(&frame(3_008, away.clone(), 0, 40));
        let gathers = unit_orders(&act, |o| matches!(o, CmdOrder::Gather(_)));
        assert!(gathers.contains(&pulled), "{:?}", act.cmds);
        assert!(unit_orders(&act, |o| matches!(o, CmdOrder::Attack(Target::Unit(_)))).is_empty(), "no chase: {:?}", act.cmds);
        assert!(!b.react_busy().iter().any(|id| pulled.contains(id)));
        let act = b.act(&frame(3_048, away, 0, 40));
        assert!(unit_orders(&act, |o| matches!(o, CmdOrder::Gather(_))).contains(&repair[0]), "{:?}", act.cmds);
        assert!(b.react_busy().is_empty());

        let mut c = teacher();
        c.scout_done = true;
        let mut u = base(20);
        u.push(enemy(93, ZERGLING, (home().0 + 6 * 32, home().1)));
        u.push(enemy(94, ZERGLING, (home().0 + 6 * 32, home().1 + 10)));
        let act = c.act(&frame(3_000, u.clone(), 0, 40));
        assert_eq!(branch(&c), Branch::Defence);
        assert!(unit_orders(&act, |o| matches!(o, CmdOrder::Attack(Target::Unit(_)))).is_empty(), "{:?}", act.cmds);
        u.iter_mut().filter(|x| x.id == 93).for_each(|x| x.pos = (home().0 + 5 * 32 - 4, home().1));
        let act = c.act(&frame(3_048, u, 0, 40));
        assert_eq!(unit_orders(&act, |o| *o == CmdOrder::Attack(Target::Unit(93))).iter().map(Vec::len).collect::<Vec<_>>(), vec![6], "{:?}", act.cmds);
    }

    fn defence_frame(n: u32, extra: Vec<UnitState>, minerals: i32) -> RawFrame {
        let mut units = base(12);
        units.push(enemy(90, ZERGLING, depot_center((60, 60))));
        units.push(enemy(91, ZERGLING, depot_center((60, 60))));
        units.extend(extra);
        frame(n, units, minerals, 24)
    }

    #[test]
    fn bunker_order_expires_without_a_bunker_041_r22() {
        let p = ReactParams::default();
        assert_eq!(p.bunker_order_expiry, 144);
        let mut b = teacher();
        b.scout_done = true;
        let act = b.act(&defence_frame(2_008, vec![], 100));
        assert_eq!(branch(&b), Branch::Defence);
        let site = builds(&act, BUNKER);
        assert_eq!(site.len(), 1, "{:?}", act.cmds);
        let worker = b.pending.iter().find(|q| q.ty == BUNKER).unwrap().worker;
        let act = b.act(&defence_frame(2_008 + 143, vec![], 100));
        assert!(builds(&act, BUNKER).is_empty(), "143 frames: still ordered: {:?}", act.cmds);
        let mut walking = teacher();
        walking.scout_done = true;
        walking.act(&defence_frame(2_008, vec![], 100));
        let mut f = defence_frame(2_008 + 144, vec![], 100);
        f.units.iter_mut().filter(|u| u.id == worker).for_each(|u| u.order = ORDER_PLACE_BUILDING);
        assert!(builds(&walking.act(&f), BUNKER).is_empty(), "the builder walks to the site");
        let mut elsewhere = teacher();
        elsewhere.scout_done = true;
        elsewhere.act(&defence_frame(2_008, vec![], 100));
        let mut f = defence_frame(2_008 + 144, vec![], 100);
        f.units.iter_mut().filter(|u| u.id == worker).for_each(|u| (u.order, u.order_pos) = (ORDER_PLACE_BUILDING, Some((10, 10))));
        assert_eq!(builds(&elsewhere.act(&f), BUNKER).len(), 1, "placing something else: ordered again");
        let mut started = teacher();
        started.scout_done = true;
        started.act(&defence_frame(2_008, vec![], 100));
        let near = (bunker_center(site[0]).0 + 32, bunker_center(site[0]).1);
        let act = started.act(&defence_frame(2_008 + 144, vec![UnitState { flags: flags::BEING_CONSTRUCTED, ..unit(60, BUNKER, near) }], 100));
        assert!(builds(&act, BUNKER).is_empty(), "{:?}", act.cmds);
        assert!(started.pending.iter().any(|q| q.ty == BUNKER));
        let act = b.act(&defence_frame(2_008 + 144, vec![], 100));
        assert_eq!(builds(&act, BUNKER).len(), 1, "{:?}", act.cmds);
        assert!(b.pending.iter().filter(|q| q.ty == BUNKER).all(|q| q.issued == 2_152));
        assert!(!bunker_order_stale(&p, &crate::Pending { ty: DEPOT, tile: (0, 0), worker: 1, issued: 0 }, &defence_frame(10_000, vec![], 0), &[]));
    }

    #[test]
    fn lifted_buildings_train_nothing_041_r22() {
        let mut b = teacher();
        b.scout_done = true;
        let act = b.act(&frame(3_000, base(12), 1_000, 24));
        assert_eq!((trains(&act, SCV), trains(&act, MARINE)), (1, 1), "{:?}", act.cmds);
        let mut units = base(12);
        units.iter_mut().filter(|u| matches!(u.unit_type, CC | BARRACKS)).for_each(|u| u.flags |= flags::LIFTED);
        let act = teacher().act(&frame(3_000, units, 1_000, 24));
        assert_eq!((trains(&act, SCV), trains(&act, MARINE)), (0, 0), "{:?}", act.cmds);
    }

    #[test]
    fn scout_is_chosen_again_from_each_observation_041_r22() {
        let cc_only = |scvs: u32| {
            let mut v = vec![unit(1, CC, home()), UnitState { owner: 11, ..unit(3, 176, (home().0 - 200, home().1)) }];
            v.extend((0..scvs).map(|i| unit(100 + i, SCV, (home().0 - 100, home().1 + 10 * i as u16))));
            v
        };
        let dark = |f: RawFrame| RawFrame { visibility: BitGrid::new(128, 128), ..f };
        let start = |b: &mut Brain| {
            b.act(&dark(frame(508, cc_only(9), 200, 18)));
            b.act(&dark(frame(516, cc_only(9), 100, 18)));
            b.scout.expect("scout")
        };
        let with = |n: u32, f: &dyn Fn(&mut UnitState)| {
            let mut u = cc_only(9);
            u.iter_mut().for_each(f);
            dark(frame(n, u, 100, 18))
        };
        let moves = |act: &Act| unit_orders(act, |o| *o == CmdOrder::Move(Target::Pos(depot_center(THEIRS))));
        let mut b = teacher();
        let s = start(&mut b);
        let walking = |u: &mut UnitState| {
            if u.id == s {
                (u.order, u.order_pos, u.flags) = (ORDER_MOVE, Some(depot_center(THEIRS)), flags::COMPLETED);
            }
        };
        for n in [524, 564, 600, 700] {
            let act = b.act(&with(n, &walking));
            assert_eq!(b.scout, Some(s), "frame {n}");
            assert!(moves(&act).iter().all(|m| m == &vec![s]), "{:?}", act.cmds);
        }
        let mut b = teacher();
        let s = start(&mut b);
        let mining = |u: &mut UnitState| {
            if u.id == s {
                u.flags = flags::COMPLETED | flags::GATHERING | flags::CARRYING;
            }
        };
        b.act(&with(524, &mining));
        assert_eq!(b.scout, Some(s), "within 48 frames of the order");
        let act = b.act(&with(564, &mining));
        let t = b.scout.expect("scout");
        assert_ne!(t, s);
        assert_eq!(moves(&act), vec![vec![t]], "the new scout is ordered at once");
        let model_scout = |u: &mut UnitState| {
            mining(u);
            if u.id == 108 {
                (u.order, u.order_pos, u.flags) = (ORDER_MOVE, Some(depot_center(THEIRS)), flags::COMPLETED);
            }
        };
        let act = b.act(&with(620, &model_scout));
        assert_eq!(b.scout, Some(108));
        assert_eq!(moves(&act), vec![vec![108]], "{:?}", act.cmds);
        let mut b = teacher();
        let s = start(&mut b);
        let mut u = cc_only(9);
        u.retain(|x| x.id != s);
        b.act(&dark(frame(530, u, 100, 18)));
        assert!(b.scout.is_some_and(|t| t != s) && !b.scout_done);
        let t = b.scout.unwrap();
        let far = |u: &mut UnitState| {
            if u.id == t {
                (u.pos, u.flags) = ((2_000, 2_000), flags::COMPLETED);
            }
        };
        b.act(&with(540, &far));
        let mut u = cc_only(9);
        u.retain(|x| x.id != t);
        b.act(&dark(frame(548, u, 100, 18)));
        assert!(b.scout.is_none() && b.scout_done);
    }

    #[test]
    fn pulled_scvs_are_chosen_again_from_each_observation_041_r22() {
        let setup = || {
            let mut b = teacher();
            b.scout_done = true;
            let mut units = base(20);
            let bunker = (home().0 + 5 * 32, home().1 + 64);
            units.push(UnitState { hp: BUNKER_HP, ..unit(5, BUNKER, bunker) });
            units.push(enemy(90, ZERGLING, (bunker.0, bunker.1 + 64)));
            units.push(enemy(91, ZERGLING, (bunker.0 + 40, bunker.1 + 70)));
            let act = b.act(&frame(3_000, units.clone(), 0, 40));
            let first = unit_orders(&act, |o| *o == CmdOrder::Attack(Target::Unit(90)));
            assert_eq!(first.iter().map(Vec::len).collect::<Vec<_>>(), vec![6], "{:?}", act.cmds);
            (b, units, first[0].clone())
        };
        let attack_at = |act: &Act| unit_orders(act, |o| *o == CmdOrder::Attack(Target::Unit(90)));
        let (mut b, mut units, pulled) = setup();
        for u in units.iter_mut().filter(|u| pulled.contains(&u.id)) {
            (u.order, u.order_target, u.flags) = (10, Some(90), flags::COMPLETED);
        }
        let act = b.act(&frame(3_048, units, 0, 40));
        assert_eq!(attack_at(&act), vec![pulled.clone()], "{:?}", act.cmds);
        let (mut b, mut units, pulled) = setup();
        for u in units.iter_mut() {
            if pulled.contains(&u.id) {
                (u.pos, u.flags) = ((home().0 - 300, home().1), flags::COMPLETED | flags::GATHERING);
            } else if u.id == 100 || u.id == 101 {
                (u.order, u.order_target, u.flags) = (10, Some(90), flags::COMPLETED);
            }
        }
        let act = b.act(&frame(3_008, units.clone(), 0, 40));
        assert!(attack_at(&act).is_empty() && pulled.iter().all(|id| b.react_busy().contains(id)), "{:?}", act.cmds);
        let act = b.act(&frame(3_048, units, 0, 40));
        let now = attack_at(&act);
        assert_eq!(now.len(), 1, "{:?}", act.cmds);
        assert_eq!(now[0].len(), 6);
        assert!(now[0].contains(&100) && now[0].contains(&101), "taken over: {now:?}");
        assert!(now[0].iter().all(|id| !pulled.contains(id)), "the mining ones are dropped: {now:?}");
    }

    #[test]
    fn main_bunker_faces_out_of_the_main_021_r8() {
        let hints = MapHints { base_locs: vec![OURS, THEIRS, NATURAL], chokes: vec![(13, 12), (12, 13), (10, 20)] };
        let b = Brain::with_hints(&info(), Kit::for_build(Race::Terran, T0Build::TerranReact), &hints);
        assert_eq!(b.bunker_facing(home(), None), (10 * 32 + 16, 20 * 32 + 16));
        let t = b.bunker_site(&frame(3_000, base(10), 0, 40), home(), b.bunker_facing(home(), None), &[]).unwrap();
        assert!(bunker_center(t).1 > home().1 + 3 * 32, "south, toward the natural: {t:?}");
        let hints = MapHints { base_locs: vec![OURS, THEIRS, NATURAL], chokes: vec![(13, 12)] };
        let b = Brain::with_hints(&info(), Kit::for_build(Race::Terran, T0Build::TerranReact), &hints);
        assert_eq!(b.bunker_facing(home(), None), depot_center(NATURAL));
    }

    #[test]
    fn main_bunker_site_is_kept_free_021_r8() {
        let mut b = teacher();
        let units = base(12);
        b.act(&frame(100, units.clone(), 0, 24));
        let site = b.react.as_ref().unwrap().main_site.expect("planned");
        let f = frame(200, units.clone(), 2_000, 24);
        let mine: Vec<&UnitState> = f.units.iter().filter(|u| u.owner == 0).collect();
        let mut cmds = Vec::new();
        for ty in [DEPOT, BARRACKS, DEPOT, BARRACKS, DEPOT, BARRACKS, DEPOT, DEPOT] {
            assert!(b.order_build(ty, &f, &mine, home(), &[], &mut cmds));
        }
        let (bx, by) = (site.0 as i32, site.1 as i32);
        for p in &b.pending {
            let (w, h) = tile_size(p.ty);
            let (x, y) = (p.tile.0 as i32, p.tile.1 as i32);
            assert!(x + w as i32 <= bx || bx + 3 <= x || y + h as i32 <= by || by + 2 <= y, "{:?} at {:?} covers the bunker site {site:?}", p.ty, p.tile);
            assert!(p.tile != NATURAL);
        }
        let mut u = units;
        u.push(enemy(90, ZERGLING, depot_center((60, 60))));
        u.push(enemy(91, ZERGLING, depot_center((60, 60))));
        let act = b.act(&frame(3_000, u, 100, 24));
        assert_eq!(builds(&act, BUNKER), vec![site], "{:?}", act.cmds);
    }

    #[test]
    fn fallback_bunker_site_stays_within_nine_tiles_021_r8() {
        let b = teacher();
        let mut f = frame(3_000, base(10), 0, 40);
        let (hx, hy) = ((home().0 / 32) as i32, (home().1 / 32) as i32);
        let mut id = 400;
        for y in hy - 8..=hy + 8 {
            for x in hx - 8..=hx + 8 {
                let d2 = (x - hx).pow(2) + (y - hy).pow(2);
                if x >= 0 && y >= 0 && (9..=49).contains(&d2) {
                    f.units.push(enemy(id, ZERGLING, (x as u16 * 32 + 16, y as u16 * 32 + 16)));
                    id += 1;
                }
            }
        }
        if let Some(t) = b.bunker_site(&f, home(), b.bunker_facing(home(), None), &[]) {
            assert!(within(bunker_center(t), home(), BUNKER_SEARCH_MAX), "{t:?}");
        }
        f.units.retain(|u| u.owner != 1 || dist2(u.pos, home()) < 5 * 5 * 32 * 32);
        let t = b.bunker_site(&f, home(), b.bunker_facing(home(), None), &[]).expect("a site beyond 5 tiles");
        assert!(within(bunker_center(t), home(), BUNKER_SEARCH_MAX), "{t:?}");
    }

    #[test]
    fn no_bunker_before_four_minutes_without_rush_evidence_021_r8() {
        let mut b = teacher();
        let mut bunkers = 0;
        let mut expanded = false;
        for n in (8..=5_760).step_by(8) {
            let scvs = (4 + n / 300).min(22);
            let mut u = if n < 1_600 {
                let mut v = vec![unit(1, CC, home()), UnitState { owner: 11, ..unit(3, 176, (home().0 - 200, home().1)) }];
                v.extend((0..scvs).map(|i| unit(100 + i, SCV, (home().0 - 100, home().1 + 4 * i as u16))));
                v
            } else {
                base(scvs)
            };
            if (2_800..3_100).contains(&n) {
                let m = depot_center(THEIRS);
                u.push(enemy(80, HATCHERY, m));
                u.extend((0..10).map(|i| enemy(81 + i, DRONE, (m.0 - 100, m.1 + 8 * i as u16))));
            }
            if n >= 4_000 {
                u.push(enemy(95, OVERLORD, (home().0 + 64, home().1 + 64)));
            }
            let supply = (2 * scvs) as u16;
            let act = b.act(&frame(n, u, 450, supply));
            bunkers += builds(&act, BUNKER).len();
            expanded |= builds(&act, CC).contains(&NATURAL);
        }
        assert_eq!((bunkers, branch(&b)), (0, Branch::Macro));
        assert!(expanded);
    }

    #[test]
    #[ignore = "needs URANIUM_MAPS_DIR with the target-map precomputes"]
    fn target_map_bunker_direction_021_r8() {
        let dir = std::path::PathBuf::from(std::env::var("URANIUM_MAPS_DIR").expect("URANIUM_MAPS_DIR"));
        let maps = [
            ("175b118cc1ab70c95e9bfc90cc5640ed300b7370443e9394aeb0d1dce6dbb0bc", [(117, 7), (7, 6), (7, 116), (117, 117)]),
            ("8109bae22de8e79ca9ddc9ca4bad343dbf5ac676747d7805f043cd8080bfdf56", [(117, 118), (7, 118), (7, 9), (117, 9)]),
        ];
        for (hash, starts) in maps {
            let pre = uranium_env::maps::load_precompute(&dir, hash).expect("precompute");
            let hints = MapHints { base_locs: pre.base_locs.clone(), chokes: pre.chokes.clone() };
            for s in starts {
                let mut i = info();
                i.static_map.start_locations = starts.to_vec();
                i.players[0].start_tile = Some(s);
                let b = Brain::with_hints(&i, Kit::for_build(Race::Terran, T0Build::TerranReact), &hints);
                let (h, nat) = (depot_center(s), depot_center(b.react.as_ref().unwrap().natural.unwrap()));
                let to = b.bunker_facing(h, None);
                let (v, w) = ((to.0 as f64 - h.0 as f64, to.1 as f64 - h.1 as f64), (nat.0 as f64 - h.0 as f64, nat.1 as f64 - h.1 as f64));
                let cos = (v.0 * w.0 + v.1 * w.1) / ((v.0.hypot(v.1)) * (w.0.hypot(w.1)));
                assert!(dist2(to, h) >= (8 * 32) * (8 * 32) && cos > 0.5, "{hash:.8} start {s:?}: faces {to:?}, natural {nat:?}, cos {cos:.2}");
            }
        }
    }

    #[test]
    fn opening_and_scout_021_r8() {
        let cc_only = |scvs: u32| {
            let mut v = vec![unit(1, CC, home()), UnitState { owner: 11, ..unit(3, 176, (home().0 - 200, home().1)) }];
            v.extend((0..scvs).map(|i| unit(100 + i, SCV, (home().0 - 100, home().1 + 10 * i as u16))));
            v
        };
        let dark = |f: RawFrame| RawFrame { visibility: BitGrid::new(128, 128), ..f };
        let mut b = teacher();
        let act = b.act(&dark(frame(500, cc_only(8), 200, 16)));
        assert!(builds(&act, DEPOT).is_empty(), "8 supply: {:?}", act.cmds);
        assert!(b.scout.is_none(), "no scout before the first depot order");
        let act = b.act(&dark(frame(508, cc_only(9), 200, 18)));
        assert_eq!(builds(&act, DEPOT).len(), 1, "9 supply");
        let builder = b.pending[0].worker;
        let act = b.act(&dark(frame(516, cc_only(9), 100, 18)));
        let scout = b.scout.expect("scout");
        assert_ne!(scout, builder);
        assert!(act.cmds.contains(&Command::Unit { units: vec![scout], order: CmdOrder::Move(Target::Pos(depot_center(THEIRS))), queued: false }), "{:?}", act.cmds);
        let mut units = cc_only(10);
        units.push(unit(6, DEPOT, (home().0, home().1 - 160)));
        let act = teacher().act(&frame(1_500, units.clone(), 200, 20));
        assert!(builds(&act, BARRACKS).is_empty(), "10 supply: {:?}", act.cmds);
        let act = teacher().act(&dark(frame(1_500, units.clone(), 200, 22)));
        assert_eq!(builds(&act, BARRACKS).len(), 1, "11 supply");
        units.push(UnitState { flags: flags::BEING_CONSTRUCTED, ..unit(2, BARRACKS, (home().0 + 200, home().1 + 160)) });
        let act = teacher().act(&frame(1_600, units.clone(), 200, 28));
        assert!(builds(&act, DEPOT).is_empty(), "14 supply: {:?}", act.cmds);
        let act = teacher().act(&frame(1_600, units, 200, 30));
        assert_eq!(builds(&act, DEPOT).len(), 1, "15 supply");
    }

    #[test]
    fn same_observations_same_commands_021_r8() {
        let script = |b: &mut Brain| {
            let mut out = Vec::new();
            let lings = vec![enemy(90, ZERGLING, (home().0 + 100, home().1)), enemy(91, ZERGLING, (home().0 + 130, home().1))];
            for (i, extra) in [vec![], lings, vec![]].into_iter().enumerate() {
                let mut u = base(12);
                u.extend(extra);
                u.extend((0..5).map(|j| unit(200 + j, MARINE, (home().0 + 50, home().1 + 6 * j as u16))));
                out.push(b.act(&frame(3_000 + 48 * i as u32, u, 300, 40)).cmds);
            }
            out
        };
        let (a, c) = (script(&mut teacher()), script(&mut teacher()));
        assert_eq!(a, c);
        assert!(a.iter().flatten().any(|c| matches!(c, Command::Unit { order: CmdOrder::Attack(Target::Unit(_)), .. })));
    }

    fn walled_info() -> GameInfo {
        let mut i = info();
        for y in 17..=18u16 {
            for x in (0..128u16).filter(|x| !(24..=27).contains(x)) {
                i.static_map.buildable.set(x, y, false);
                for c in 0..16 {
                    i.static_map.walkable.set(x * 4 + c % 4, y * 4 + c / 4, false);
                }
            }
        }
        i
    }

    #[test]
    fn main_bunker_site_on_the_ground_path_021_r8() {
        let i = walled_info();
        let solid = BTreeSet::new();
        let path = ground_path(&i.static_map.walkable, (12, 11), (12, 27), &solid);
        assert_eq!((path.first(), path.last()), (Some(&(12, 11)), Some(&(12, 27))));
        assert!(path.iter().any(|t| (17..=18).contains(&t.1) && (24..=27).contains(&t.0)), "{path:?}");
        assert!(path.windows(2).all(|w| (w[0].0 as i32 - w[1].0 as i32).abs() <= 1 && (w[0].1 as i32 - w[1].1 as i32).abs() <= 1));
        assert!(path.iter().all(|t| !(17..=18).contains(&t.1) || (24..=27).contains(&t.0)));
        let all: BTreeSet<Tile> = (24..=27).flat_map(|x| (17..=18).map(move |y| (x, y))).collect();
        assert!(ground_path(&i.static_map.walkable, (12, 11), (12, 27), &all).is_empty());

        let mut b = Brain::with_hints(&i, Kit::for_build(Race::Terran, T0Build::TerranReact), &hints());
        let mut units = vec![unit(1, CC, home()), UnitState { owner: 11, ..unit(3, 176, (home().0 - 200, home().1)) }];
        units.extend((0..10).map(|n| unit(100 + n, SCV, (home().0 - 100, home().1 + 10 * n as u16))));
        b.act(&frame(100, units, 0, 20));
        let r = b.react.as_ref().unwrap();
        let path = r.main_path.clone().expect("planned");
        let site = r.main_site.expect("main site");
        let c = bunker_center(site);
        let d = (dist2(c, home()) as f64).sqrt() / 32.0;
        assert!((5.0..=7.0).contains(&d), "{site:?} {d}");
        assert!(path.iter().any(|t| within(c, (t.0 * 32 + 16, t.1 * 32 + 16), 2)), "{site:?} {path:?}");
        assert!(c.0 > home().0, "toward the ramp (east), not straight south: {site:?}");
    }

    #[test]
    fn defence_marines_attack_or_hold_021_r8() {
        let run = |marines: u32| {
            let mut b = teacher();
            b.scout_done = true;
            let mut u = base(12);
            u.extend((0..marines).map(|n| unit(200 + n, MARINE, (home().0 + 100, home().1 + 100 + 4 * n as u16))));
            u.push(enemy(90, ZERGLING, (home().0 + 8 * 32, home().1)));
            u.push(enemy(91, ZERGLING, (home().0 + 8 * 32, home().1 + 20)));
            let act = b.act(&frame(3_000, u, 0, 40));
            assert_eq!(branch(&b), Branch::Defence);
            let attack: usize = unit_orders(&act, |o| matches!(o, CmdOrder::Attack(Target::Pos(_)))).iter().map(Vec::len).sum();
            let walk: usize = unit_orders(&act, |o| matches!(o, CmdOrder::Move(_))).iter().flatten().filter(|id| **id >= 200).count();
            (attack, walk, b.react_marines_attack())
        };
        assert_eq!(run(4), (4, 0, true), "4 marines vs 2 zerglings: attack");
        assert_eq!(run(3), (0, 3, false), "3 marines vs 2 zerglings: hold beside the bunker");
    }

    #[test]
    fn macro_return_at_twelve_marines_021_r8() {
        let at = |b: &mut Brain, n: u32, marines: u32| {
            let mut u = base(12);
            u.extend((0..marines).map(|k| unit(200 + k, MARINE, (home().0 + 100, home().1 + 4 * k as u16))));
            u.push(enemy(90, ZERGLING, (home().0 + 9 * 32, home().1)));
            u.push(enemy(91, ZERGLING, (home().0 + 9 * 32, home().1 + 20)));
            b.act(&frame(n, u, 0, 40));
            branch(b)
        };
        let mut b = teacher();
        b.scout_done = true;
        assert_eq!(at(&mut b, 3_000, 2), Branch::Defence);
        assert_eq!(at(&mut b, 3_048, 11), Branch::Defence, "11 marines under contact");
        assert_eq!(at(&mut b, 3_096, 12), Branch::Macro, "12 marines whatever the contact");
        assert_eq!(at(&mut b, 3_144, 12), Branch::Macro, "R with 12 marines keeps M");
        assert_eq!(b.teacher_stats().unwrap().macro_frame, Some(3_096));
        assert_eq!(at(&mut b, 3_192, 11), Branch::Defence, "fewer than 12 again: R switches to D");
    }

    #[test]
    fn construction_resumes_without_a_builder_021_r8() {
        let mut b = teacher();
        b.scout_done = true;
        let rax = (home().0 + 7 * 32, home().1 - 4 * 32);
        let mut u = base(12);
        u.push(UnitState { hp: 900, flags: flags::BEING_CONSTRUCTED, ..unit(50, BARRACKS, rax) });
        let right_click = |act: &Act| unit_orders(act, |o| *o == CmdOrder::RightClick(Target::Unit(50)));
        assert!(right_click(&b.act(&frame(3_000, u.clone(), 0, 30))).is_empty());
        assert!(right_click(&b.act(&frame(3_047, u.clone(), 0, 30))).is_empty());
        let act = b.act(&frame(3_048, u.clone(), 0, 30));
        let nearest = u.iter().filter(|x| x.unit_type == SCV).min_by_key(|x| (dist2(x.pos, rax), x.id)).unwrap().id;
        assert_eq!(right_click(&act), vec![vec![nearest]], "{:?}", act.cmds);
        assert!(b.react_busy().contains(&nearest));
        assert!(right_click(&b.act(&frame(3_090, u.clone(), 0, 30))).is_empty());
        assert_eq!(right_click(&b.act(&frame(3_096, u.clone(), 0, 30))), vec![vec![nearest]]);
        for x in u.iter_mut().filter(|x| x.id == nearest) {
            (x.order, x.order_target, x.flags) = (ORDER_CONSTRUCTING, Some(50), flags::COMPLETED | flags::CONSTRUCTING);
        }
        assert!(right_click(&b.act(&frame(3_200, u.clone(), 0, 30))).is_empty());
        assert!(b.react.as_ref().unwrap().resumers.is_empty() && !b.react_busy().contains(&nearest));
    }

    #[test]
    fn builder_harassed_by_a_worker_021_r8() {
        let mut b = teacher();
        b.scout_done = true;
        let mut u = base(12);
        for x in u.iter_mut().filter(|x| x.unit_type == SCV) {
            x.flags = flags::COMPLETED | flags::GATHERING;
        }
        let at = (home().0 + 6 * 32, home().1 - 3 * 32);
        u.push(UnitState { flags: flags::COMPLETED | flags::CONSTRUCTING | flags::UNDER_ATTACK, ..unit(99, SCV, at) });
        u.push(enemy(80, DRONE, (at.0 + 40, at.1)));
        let act = b.act(&frame(1_500, u.clone(), 0, 24));
        assert_eq!(branch(&b), Branch::Macro);
        let guards = unit_orders(&act, |o| *o == CmdOrder::Attack(Target::Unit(80)));
        assert_eq!(guards.iter().map(Vec::len).collect::<Vec<_>>(), vec![2], "{:?}", act.cmds);
        assert!(!guards[0].contains(&99));
        u.iter_mut().filter(|x| x.id == 80).for_each(|x| x.pos = (at.0 + 5 * 32, at.1));
        b.act(&frame(1_548, u.clone(), 0, 24));
        assert!(b.react_busy().contains(&guards[0][0]));
        u.iter_mut().filter(|x| x.id == 80).for_each(|x| x.pos = (at.0 + 7 * 32, at.1));
        let act = b.act(&frame(1_556, u, 0, 24));
        assert!(unit_orders(&act, |o| matches!(o, CmdOrder::Gather(_))).contains(&guards[0]), "{:?}", act.cmds);
        assert!(b.react_busy().is_empty());
    }

    #[test]
    fn terran_react_golden_digest_021_r8() {
        let mut b = teacher();
        let mut out: Vec<String> = Vec::new();
        for n in (8..=7_200u32).step_by(8) {
            let scvs = (4 + n / 300).min(22);
            let mut u = if n < 1_600 {
                let mut v = vec![unit(1, CC, home()), UnitState { owner: 11, ..unit(3, 176, (home().0 - 200, home().1)) }];
                v.extend((0..scvs).map(|i| unit(100 + i, SCV, (home().0 - 100, home().1 + 4 * i as u16))));
                v
            } else {
                base(scvs)
            };
            if (2_800..3_100).contains(&n) {
                let m = depot_center(THEIRS);
                u.push(enemy(80, HATCHERY, m));
                u.push(enemy(81, SPAWNING_POOL, (m.0 + 160, m.1)));
            }
            if (4_000..5_000).contains(&n) {
                u.push(UnitState { hp: 250, ..unit(60, BUNKER, (home().0 + 5 * 32, home().1 + 64)) });
                u.extend((0..6).map(|i| unit(200 + i, MARINE, (home().0 + 40, home().1 + 100 + 8 * i as u16))));
                u.extend((0..4).map(|i| enemy(90 + i, ZERGLING, (home().0 + 5 * 32 + 10 * i as u16, home().1 + 128))));
            }
            if n % 96 == 0 {
                u.iter_mut().filter(|x| x.unit_type == SCV).take(2).for_each(|x| x.flags = flags::COMPLETED | flags::IDLE | flags::CARRYING);
            }
            let act = b.act(&frame(n, u, 120 + (n % 400) as i32, (2 * scvs) as u16));
            out.push(format!("{n} {:?}", act.cmds));
        }
        let mut h: u64 = 0xcbf29ce484222325;
        for byte in out.join("\n").bytes() {
            h ^= u64::from(byte);
            h = h.wrapping_mul(0x100000001b3);
        }
        assert_eq!(h, 0x99c9cddbf89e4ac5, "terran_react commands changed");
    }

    #[test]
    fn build_id_021_r8() {
        let t = T0Build::TerranReact;
        assert_eq!((t.name(), t.id_tag(), t.race(), t.needs_map_hints()), ("terran_react", Some("react"), Some(Race::Terran), true));
        assert!(Kit::for_build(Race::Terran, t).react.is_some());
        assert!(Kit::for_build(Race::Zerg, t).react.is_none());
        assert!(Kit::terran().react.is_none() && Kit::terran_drop().react.is_none() && !T0Build::TerranDrop.needs_map_hints());
    }
}
