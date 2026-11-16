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
pub mod testkit;

#[cfg(test)]
mod tests;
