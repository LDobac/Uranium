//! 022 TvT / TvP teacher (`T0Build::TerranTp`, opponent id `t0-terran-tp`, plan build
//! `terran_tp`, spec 022, 021-R9).
//!
//! The teacher plays the 021-R8 opening (9 depot, 11 barracks, 15 depot) with a refinery at 12
//! supply and three SCVs on gas, at most four marines before the factory, the factory at 100 gas,
//! the machine shop once the factory is complete, siege mode research once the shop is, then
//! siege tanks and marines (022-R3). Its scout is the 021-R8 scout, but it stays in the enemy main
//! until 2:40 (or until its hit points fall below half) to count gateways / barracks and army.
//!
//! Rush rule R (022-R4) reads the current observation only: two gateways / barracks before 2:40,
//! two zealots / four marines before 3:30, the scout at the enemy start depot from 2:30 to 3:30
//! without a gateway / barracks ever seen near that start (a proxy), or two enemy army units
//! within 20 tiles of an own command center. Without R the macro branch M expands (Protoss: once
//! the factory is ordered, at the latest from 4:00; Terran: after two tanks), adds barracks and
//! a natural bunker (022-R5). R switches to the defence branch D at that observation: the 021-R8
//! main bunker, garrison, SCV pull and repair, a second barracks, and the tanks gather behind
//! the bunker (022-R6). D returns to M with 8 marines, 2 tanks and 60 s without contact, or with
//! 4 tanks. Tanks siege when enemy ground army is within 11 tiles and leave siege mode after 10 s
//! without any within 13 tiles when they have somewhere to go (022-R7). With 4 tanks and 40 army
//! supply (20 from 12:00) M attacks: the tanks lead, marines follow, units more than 8 tiles
//! ahead of the tanks wait, and the shared brain's target choice hunts the last buildings (022-R8).
//!
//! The rest is the 021-R8 teacher (`react`): bunker sites, SCV pull / repair, worker answer,
//! construction resumption, home guard, map hints, and its shadow fix-ups (022-R9). The tank
//! siege timers, the tank spot, the attack group and the gas crew are computed again from the
//! own units visible at each observation (no ids carried from the teacher's own orders). Every
//! action is one 010 `Command` the model can issue (034), at most 12 units each. No randomness.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;
use uranium_env::scout::{is_army_type, is_ground_army_type};
use uranium_proto::*;

use crate::react::{Branch, FPS, bunker_center, depot_center, is_enemy, toward, within};
use crate::types::*;
use crate::{Brain, centroid, dist2, done};

/// SCV hit points (BWAPI type table): the scout leaves the enemy main below `scout_hp_pct` of it.
const SCV_HP: u32 = 60;
/// Army units farther than this (tiles) from every own command center are out on an attack
/// (the Brain's home-guard radius, `DEFEND_RADIUS`).
const OUT_TILES: i64 = 20;

/// 022 numbers (AMB-022-2..5 Resolved, 2026-11-14). Frames are game frames (24 per second),
/// distances tiles, supply in units (a marine = 1).
#[derive(Clone, Debug, PartialEq)]
pub struct TpParams {
    /// 022-R3: the refinery at this supply (12), `gas_workers` (3) SCVs on gas, at most
    /// `marines_before_factory` (4) marines until the factory is ordered, the factory at
    /// `factory_gas` (100) gas.
    pub refinery_supply: u32,
    pub gas_workers: usize,
    pub marines_before_factory: usize,
    pub factory_gas: i32,
    /// 022-R3 scout: once it finds the enemy main before `scout_stay_until` (2:40) it stays there
    /// until then or until its hit points fall below `scout_hp_pct` (50) %; it circles
    /// `scout_circle` tiles around the enemy start (메인 임시 결정: the way it "goes around the
    /// main") and from `proxy_from` walks to the start's depot (rule ④).
    pub scout_stay_until: u32,
    pub scout_hp_pct: u32,
    pub scout_circle: i64,
    /// 022-R4 ①: at least `prod_min` (2) gateways / barracks (unfinished ones too) in one
    /// observation before `prod_before` (2:40).
    pub prod_min: usize,
    pub prod_before: u32,
    /// 022-R4 ②: at least `zealots_min` (2) zealots / `marines_min` (4) marines in one
    /// observation before `units_before` (3:30), anywhere.
    pub zealots_min: usize,
    pub marines_min: usize,
    pub units_before: u32,
    /// 022-R4 ④: the scout within `proxy_look` (3) tiles of the enemy start depot (a visible
    /// nexus / command center within `main_radius` of a start location), `proxy_from` (2:30) ≤
    /// t < `proxy_before` (3:30), and no gateway / barracks ever seen within `proxy_radius` (12)
    /// tiles of that start.
    pub proxy_from: u32,
    pub proxy_before: u32,
    pub proxy_look: i64,
    pub proxy_radius: i64,
    /// 022-R4 ③ (021-R8 ③): at least `army_min` (2) enemy army units (033) within
    /// `army_radius` (20) tiles of an own command center; one there is contact.
    pub army_radius: i64,
    pub army_min: usize,
    /// The enemy main is found when an enemy building is remembered within `main_radius` (10)
    /// tiles of a start location (021-R8).
    pub main_radius: i64,
    /// R is judged only before this frame (8:00).
    pub judge_until: u32,
    /// 022-R5 M: the natural command center at `expand_minerals` (400) — against Protoss once the
    /// factory is ordered (at the latest from `expand_by`, 4:00: then its price is held from
    /// everything), against Terran once `expand_tanks` (2) tanks are complete.
    pub expand_minerals: i32,
    pub expand_by: u32,
    pub expand_tanks: usize,
    /// 022-R5: barracks once the natural command center is complete (1 + 2 against Protoss,
    /// 1 + 1 against Terran); SCVs `scv_per_cc` (22) per complete command center, at most
    /// `scv_max` (40).
    pub barracks_p: usize,
    pub barracks_t: usize,
    pub scv_per_cc: usize,
    pub scv_max: usize,
    /// 022-R6 D: `barracks_d` (2) barracks; the natural with `expand_tanks` tanks and `quiet` (60 s)
    /// after the last contact; back to M with `macro_marines` (8, loaded ones too) marines,
    /// `macro_tanks` (2) tanks and `quiet` without contact, or with `macro_tanks_any` (4) tanks
    /// (which also keep M when R holds, 메인 임시 결정 like 021-R8's 12 marines). The tanks wait
    /// `tank_wait` (3, within the 2–4 of 022-R6) tiles behind the bunker toward its command center.
    pub barracks_d: usize,
    pub quiet: u32,
    pub macro_marines: usize,
    pub macro_tanks: usize,
    pub macro_tanks_any: usize,
    pub tank_wait: f64,
    /// 022-R7: siege mode once siege mode is researched and enemy ground army (no workers,
    /// visible) is within `siege_range` (11) tiles; tank mode after `unsiege_quiet` (10 s) without
    /// any within `unsiege_range` (13) tiles while the tank has somewhere to go.
    pub siege_range: i64,
    pub unsiege_range: i64,
    pub unsiege_quiet: u32,
    /// 022-R8: M attacks with `wave_tanks` (4) tanks and `wave_supply` (40) army supply (a marine
    /// 1, a tank 2), or from `late_wave_from` (12:00) with `late_wave_supply` (20); units more than
    /// `group_gap` (8) tiles ahead of the tanks' centre wait.
    pub wave_tanks: usize,
    pub wave_supply: u32,
    pub late_wave_from: u32,
    pub late_wave_supply: u32,
    pub group_gap: i64,
    /// Repeated orders (gas crew, machine shop, research, tank and attack moves): every this many
    /// frames (022-R1, 48).
    pub retarget: u32,
}

impl Default for TpParams {
    fn default() -> Self {
        Self {
            refinery_supply: 12,
            gas_workers: 3,
            marines_before_factory: 4,
            factory_gas: 100,
            scout_stay_until: 160 * FPS,
            scout_hp_pct: 50,
            scout_circle: 6,
            prod_min: 2,
            prod_before: 160 * FPS,
            zealots_min: 2,
            marines_min: 4,
            units_before: 210 * FPS,
            proxy_from: 150 * FPS,
            proxy_before: 210 * FPS,
            proxy_look: 3,
            proxy_radius: 12,
            army_radius: 20,
            army_min: 2,
            main_radius: 10,
            judge_until: 480 * FPS,
            expand_minerals: 400,
            expand_by: 240 * FPS,
            expand_tanks: 2,
            barracks_p: 3,
            barracks_t: 2,
            scv_per_cc: 22,
            scv_max: 40,
            barracks_d: 2,
            quiet: 60 * FPS,
            macro_marines: 8,
            macro_tanks: 2,
            macro_tanks_any: 4,
            tank_wait: 3.0,
            siege_range: 11,
            unsiege_range: 13,
            unsiege_quiet: 10 * FPS,
            wave_tanks: 4,
            wave_supply: 40,
            late_wave_from: 720 * FPS,
            late_wave_supply: 20,
            group_gap: 8,
            retarget: 48,
        }
    }
}

/// Which 022-R4 rush rule held.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TpRule {
    /// ①P 2 gateways before 2:40.
    Gates,
    /// ②P 2 zealots before 3:30.
    Zealots,
    /// ④P the scout at the enemy start nexus, 2:30–3:30, no gateway ever seen near it.
    ProxyGate,
    /// ①T 2 barracks before 2:40.
    Rax,
    /// ②T 4 marines before 3:30.
    Marines,
    /// ④T the scout at the enemy start command center, 2:30–3:30, no barracks ever seen near it.
    ProxyRax,
    /// ③ 2 enemy army units within 20 tiles of an own command center.
    Army,
}

/// What the 022 teacher did in one game (session record `scripted.tp`, 022-R10). Frames.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct TpStats {
    /// Enemy main found (an enemy building remembered within `main_radius` of a start).
    pub main_frame: Option<u32>,
    /// First rush identification and its rule.
    pub rush_frame: Option<u32>,
    pub rush_rule: Option<TpRule>,
    /// M → D switches; first D → M switch.
    pub defences: u32,
    pub macro_frame: Option<u32>,
    /// First order of the refinery, factory, machine shop, siege mode research, siege tank and
    /// natural command center.
    pub refinery_frame: Option<u32>,
    pub factory_frame: Option<u32>,
    pub shop_frame: Option<u32>,
    pub siege_frame: Option<u32>,
    pub tank_frame: Option<u32>,
    pub expand_frame: Option<u32>,
    /// First observation with the attack condition (022-R8).
    pub wave_frame: Option<u32>,
    /// Branch at the last observation.
    pub branch: Branch,
    /// First enemy army within `army_radius` of an own command center; first own bunker.
    pub contact_frame: Option<u32>,
    pub bunker_frame: Option<u32>,
    /// First observation (before 8:00) at which each rule held, in any branch.
    pub rule_frames: BTreeMap<TpRule, u32>,
}

/// The matchup build the teacher follows (022-R5): the lobby race, a lobby Random by the first
/// enemy unit seen (Terran's until then, 메인 임시 결정 — Random games are not data, 022-R2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Matchup {
    Protoss,
    Terran,
}

pub(crate) struct TpState {
    pub p: TpParams,
    lobby: Race,
    seen: Option<Race>,
    /// SCVs gathering gas at the last observation (022-R9: from their orders, not from ids).
    pub gas_crew: Vec<UnitId>,
    /// Last frame each SCV was seen with a gas order: one seen on gas within 2 × `retarget`
    /// frames still counts unless it is seen mining minerals (an SCV leaving the refinery shows
    /// other orders for a moment; OpenBW smoke: counted out, a fourth SCV was sent).
    gas_seen: BTreeMap<UnitId, u32>,
    /// The complete refinery and the first frame it was seen complete: SCVs are sent `retarget`
    /// frames later, so its builder (which starts mining on its own) is counted first.
    refinery_since: Option<(UnitId, u32)>,
    last_gas: Option<u32>,
    last_shop: Option<u32>,
    last_research: Option<u32>,
    last_tanks: Option<u32>,
    /// (start, gateway / barracks type): one was seen within `proxy_radius` of that start (R ④).
    prod_seen: BTreeSet<(Tile, u16)>,
    /// Sieged tanks: the last frame with enemy ground army within `unsiege_range` (the first
    /// frame seen sieged otherwise); dropped once a tank is not sieged (022-R7, R9).
    quiet: BTreeMap<UnitId, u32>,
    order_sieging: Option<u16>,
    order_unsieging: Option<u16>,
    gas_orders: Vec<u16>,
    mineral_orders: Vec<u16>,
    pub stats: TpStats,
}

impl TpState {
    pub fn new(p: TpParams, info: &GameInfo) -> Self {
        let lobby = info.players.iter().find(|q| q.is_enemy).map_or(Race::Unknown, |q| q.race);
        let order = uranium_env::types::order_id;
        Self {
            p,
            lobby,
            seen: None,
            gas_crew: Vec::new(),
            gas_seen: BTreeMap::new(),
            refinery_since: None,
            last_gas: None,
            last_shop: None,
            last_research: None,
            last_tanks: None,
            prod_seen: BTreeSet::new(),
            quiet: BTreeMap::new(),
            order_sieging: order("Sieging"),
            order_unsieging: order("Unsieging"),
            gas_orders: ["MoveToGas", "WaitForGas", "HarvestGas", "ReturnGas"].into_iter().filter_map(order).collect(),
            mineral_orders: ["MoveToMinerals", "WaitForMinerals", "MiningMinerals", "ReturnMinerals"].into_iter().filter_map(order).collect(),
            stats: TpStats::default(),
        }
    }

    pub fn matchup(&self) -> Matchup {
        match (self.lobby, self.seen) {
            (Race::Protoss, _) | (Race::Random | Race::Unknown, Some(Race::Protoss)) => Matchup::Protoss,
            _ => Matchup::Terran,
        }
    }

    fn due(last: Option<u32>, frame: u32, every: u32) -> bool {
        last.is_none_or(|l| frame >= l + every)
    }
}

/// The race of a unit type (BWAPI type table), for a lobby-Random enemy.
fn race_of(ty: u16) -> Option<Race> {
    match uranium_env::types::TypeTables::get().unit(ty)?.race.as_str() {
        "Protoss" => Some(Race::Protoss),
        "Terran" => Some(Race::Terran),
        "Zerg" => Some(Race::Zerg),
        _ => None,
    }
}

/// What the 022 rule needs besides the observation: own command centers, the start locations
/// other than ours, the scout's position and the (start, gateway / barracks type) pairs ever seen.
pub struct TpContext<'a> {
    pub depots: &'a [Pos],
    pub starts: &'a [Tile],
    pub scout: Option<Pos>,
    pub prod_seen: &'a BTreeSet<(Tile, u16)>,
}

/// Every 022-R4 rush rule that holds in this observation (own vision only), in rule order.
pub fn tp_rules(p: &TpParams, f: &RawFrame, me: u8, cx: &TpContext<'_>) -> Vec<TpRule> {
    let mut out = Vec::new();
    if f.frame >= p.judge_until {
        return out;
    }
    let enemies: Vec<&UnitState> = f.units.iter().filter(|u| is_enemy(u, me)).collect();
    let count = |ty: u16| enemies.iter().filter(|u| u.unit_type == ty).count();
    // ④: the scout next to a visible enemy start depot of `depot` type, no `prod` ever seen there.
    let proxy = |depot: u16, prod: u16| {
        (p.proxy_from..p.proxy_before).contains(&f.frame)
            && cx.scout.is_some_and(|sc| {
                enemies.iter().filter(|u| u.unit_type == depot && within(sc, u.pos, p.proxy_look)).any(|d| {
                    cx.starts.iter().any(|s| within(d.pos, depot_center(*s), p.main_radius) && !cx.prod_seen.contains(&(*s, prod)))
                })
            })
    };
    if f.frame < p.prod_before && count(GATEWAY) >= p.prod_min {
        out.push(TpRule::Gates);
    }
    if f.frame < p.units_before && count(ZEALOT) >= p.zealots_min {
        out.push(TpRule::Zealots);
    }
    if proxy(NEXUS, GATEWAY) {
        out.push(TpRule::ProxyGate);
    }
    if f.frame < p.prod_before && count(BARRACKS) >= p.prod_min {
        out.push(TpRule::Rax);
    }
    if f.frame < p.units_before && count(MARINE) >= p.marines_min {
        out.push(TpRule::Marines);
    }
    if proxy(CC, BARRACKS) {
        out.push(TpRule::ProxyRax);
    }
    let near_army = enemies.iter().filter(|u| is_army_type(u.unit_type) && cx.depots.iter().any(|d| within(u.pos, *d, p.army_radius))).count();
    if near_army >= p.army_min {
        out.push(TpRule::Army);
    }
    out
}

impl Brain {
    /// The shared brain's hunting cells (walkable centre) stalest first, in its scan order on ties:
    /// the first is its hunting target.
    fn stale_cells(&self) -> Vec<Pos> {
        let mut cells: Vec<(u32, u16, u16, Pos)> = Vec::new();
        for cy in 0..self.cells_h {
            for cx in 0..self.cells_w {
                let p = ((cx * crate::CELL + crate::CELL / 2) * 32, (cy * crate::CELL + crate::CELL / 2) * 32);
                if self.walkable.get(p.0 / 8, p.1 / 8) {
                    cells.push((self.cell_seen[(cy * self.cells_w + cx) as usize], cy, cx, p));
                }
            }
        }
        cells.sort_unstable();
        cells.into_iter().map(|(_, _, _, p)| p).collect()
    }

    /// SCVs on gas at the last observation (busy for the other SCV jobs).
    pub(crate) fn tp_gas_crew(&self) -> Vec<UnitId> {
        self.tp.as_ref().map(|t| t.gas_crew.clone()).unwrap_or_default()
    }

    /// Completed own siege tanks (either mode).
    fn tanks_done(mine: &[&UnitState]) -> usize {
        mine.iter().filter(|u| is_tank(u.unit_type) && u.has(flags::COMPLETED)).count()
    }

    /// The natural command center is complete.
    fn natural_done(&self, mine: &[&UnitState]) -> bool {
        let Some(n) = self.react.as_ref().and_then(|r| r.natural) else { return false };
        mine.iter().any(|u| is_resource_depot(u.unit_type) && building_tile(u) == n && done(u))
    }

    /// Per observation (022-R4, R6, R10): enemy main found, the enemy race of a lobby Random,
    /// gateways / barracks seen near starts, contact, rule R (M → D) and the D → M rule. The
    /// branch, contact and defence base live in the 021-R8 state its D parts read.
    pub(crate) fn tp_observe(&mut self, f: &RawFrame, mine: &[&UnitState]) {
        let Some(p) = self.tp.as_ref().map(|t| t.p.clone()) else { return };
        let me = self.me;
        let others = self.other_starts();
        let scout = self.scout.and_then(|id| mine.iter().find(|u| u.id == id)).map(|u| u.pos);
        let depots = Self::own_depots(mine);
        let enemies: Vec<&UnitState> = f.units.iter().filter(|u| is_enemy(u, me)).collect();
        let found = others.iter().copied().find(|s| self.enemy_buildings.values().any(|(pos, _)| within(*pos, depot_center(*s), p.main_radius)));
        let home = self.home;
        let marines = mine.iter().filter(|u| u.unit_type == MARINE && u.has(flags::COMPLETED)).count();
        let tanks = Self::tanks_done(mine);
        let t = self.tp.as_mut().expect("checked above");
        if t.seen.is_none() {
            t.seen = enemies.iter().find_map(|u| race_of(u.unit_type));
        }
        for u in enemies.iter().filter(|u| matches!(u.unit_type, GATEWAY | BARRACKS)) {
            for s in others.iter().filter(|s| within(u.pos, depot_center(**s), p.proxy_radius)) {
                t.prod_seen.insert((*s, u.unit_type));
            }
        }
        let rules = tp_rules(&p, f, me, &TpContext { depots: &depots, starts: &others, scout, prod_seen: &t.prod_seen });
        let army: Vec<Pos> = enemies.iter().filter(|u| is_army_type(u.unit_type)).map(|u| u.pos).collect();
        let contact = army.iter().any(|a| depots.iter().any(|d| within(*a, *d, p.army_radius)));
        if t.stats.bunker_frame.is_none() && mine.iter().any(|u| u.unit_type == BUNKER) {
            t.stats.bunker_frame = Some(f.frame);
        }
        for rule in &rules {
            t.stats.rule_frames.entry(*rule).or_insert(f.frame);
        }
        let r = self.react.as_mut().expect("the 022 teacher reuses the 021-R8 state");
        if let Some(s) = found.filter(|_| r.enemy_main.is_none()) {
            r.enemy_main = Some(s);
            // The 021-R8 scout's look timer reads this one.
            r.stats.main_frame = Some(f.frame);
            t.stats.main_frame = Some(f.frame);
        }
        if contact {
            r.last_contact = f.frame;
            t.stats.contact_frame.get_or_insert(f.frame);
        }
        match r.branch {
            Branch::Macro => {
                // With `macro_tanks_any` tanks R does not switch to D (it would switch back at
                // the next observation; 메인 임시 결정 like 021-R8's 12 marines).
                if let Some(&rule) = rules.first().filter(|_| tanks < p.macro_tanks_any) {
                    r.branch = Branch::Defence;
                    r.last_contact = f.frame;
                    t.stats.defences += 1;
                    if t.stats.rush_frame.is_none() {
                        t.stats.rush_frame = Some(f.frame);
                        t.stats.rush_rule = Some(rule);
                    }
                    // The defence base (021-R8, rule advisory §3): the own command center nearest
                    // the enemy army seen; without army in sight the natural once it stands, else
                    // the main.
                    let main = home.and_then(|h| depots.iter().copied().min_by_key(|d| (dist2(*d, h), *d)));
                    let natural = r.natural.and_then(|n| depots.iter().copied().find(|d| within(*d, depot_center(n), 1)));
                    let by_army = army.iter().flat_map(|a| depots.iter().map(move |d| (dist2(*a, *d), *d))).min().map(|(_, d)| d);
                    r.def_base = by_army.or(natural).or(main);
                }
            }
            Branch::Defence => {
                let calm = marines >= p.macro_marines && tanks >= p.macro_tanks && f.frame >= r.last_contact + p.quiet;
                if calm || tanks >= p.macro_tanks_any {
                    r.branch = Branch::Macro;
                    t.stats.macro_frame.get_or_insert(f.frame);
                }
            }
        }
        // D marines (021-R8 7th advisory C): attack-move while they outnumber the enemy ground
        // army near the bases 2:1, else load / hold beside the bunker.
        let outside = mine.iter().filter(|u| u.unit_type == MARINE && u.has(flags::COMPLETED) && u.loaded_in.is_none()).count();
        let close = enemies.iter().filter(|u| is_ground_army_type(u.unit_type) && depots.iter().any(|d| within(u.pos, *d, r.p.marine_radius))).count();
        r.marines_attack = r.branch == Branch::Defence && close > 0 && outside >= r.p.marine_ratio * close;
        if r.marines_attack {
            r.holding.clear();
        }
        t.stats.branch = r.branch;
    }

    /// 022-R3 scout (AMB-022-4): with the enemy main found before 2:40 the scout stays in it until
    /// 2:40 or until its hit points fall below half — circling the enemy start, from 2:30 at its
    /// depot (rule ④) — then mines again (`Some(None)`). A main found from 2:40 on is looked at
    /// as by the 021-R8 scout (`None`).
    pub(crate) fn tp_scout_target(&self, f: &RawFrame, s: &UnitState, main: Tile) -> Option<Option<Pos>> {
        let t = self.tp.as_ref()?;
        let p = &t.p;
        if t.stats.main_frame? >= p.scout_stay_until {
            return None;
        }
        if f.frame >= p.scout_stay_until || u32::from(s.hp) * 100 < SCV_HP * p.scout_hp_pct {
            return Some(None);
        }
        let c = depot_center(main);
        if f.frame >= p.proxy_from {
            return Some(Some(c));
        }
        Some(Some(self.circle_point(c, s.pos, p.scout_circle)))
    }

    /// A point `radius` tiles from `c`, a quarter turn on from `pos` (the scout goes around the
    /// enemy start), on a walkable spot of the static map (else the nearest turn that is; else `c`).
    fn circle_point(&self, c: Pos, pos: Pos, radius: i64) -> Pos {
        let a = (pos.1 as f64 - c.1 as f64).atan2(pos.0 as f64 - c.0 as f64);
        let r = (radius * 32) as f64;
        for deg in [90.0f64, 60.0, 120.0, 45.0, 135.0, 30.0, 150.0, 180.0] {
            let b = a + deg.to_radians();
            let (x, y) = (c.0 as f64 + r * b.cos(), c.1 as f64 + r * b.sin());
            if x < 0.0 || y < 0.0 {
                continue;
            }
            let (x, y) = (x.round() as u16, y.round() as u16);
            if (x / 8) < self.walkable.width && (y / 8) < self.walkable.height && self.walkable.get(x / 8, y / 8) {
                return (x, y);
            }
        }
        c
    }

    /// 022-R3 gas crew (022-R9: chosen again from each observation): the own SCVs whose order is a
    /// gas order or targets the refinery (or was, within 2 × `retarget` frames, and they are not
    /// mining minerals now); while fewer than `gas_workers`, the free SCVs nearest the refinery
    /// are sent to it (every `retarget` frames, from `retarget` frames after it is complete).
    pub(crate) fn tp_gas(&mut self, f: &RawFrame, mine: &[&UnitState], cmds: &mut Vec<Command>) {
        let Some(t) = self.tp.as_ref() else { return };
        let (want, every, last) = (t.p.gas_workers, t.p.retarget, t.last_gas);
        let refinery = mine.iter().filter(|u| u.unit_type == REFINERY && done(u)).min_by_key(|u| u.id).copied();
        let on_gas: Vec<UnitId> = mine
            .iter()
            .filter(|u| u.unit_type == SCV && (t.gas_orders.contains(&u.order) || refinery.is_some_and(|r| u.order_target == Some(r.id))))
            .map(|u| u.id)
            .collect();
        let t = self.tp.as_mut().expect("checked above");
        for id in &on_gas {
            t.gas_seen.insert(*id, f.frame);
        }
        let minerals: Vec<UnitId> = mine.iter().filter(|u| u.unit_type == SCV && t.mineral_orders.contains(&u.order)).map(|u| u.id).collect();
        t.gas_seen.retain(|id, at| on_gas.contains(id) || (f.frame < *at + 2 * every && !minerals.contains(id)));
        let crew: Vec<UnitId> = t.gas_seen.keys().copied().collect();
        t.gas_crew = crew.clone();
        t.refinery_since = match (refinery, t.refinery_since) {
            (Some(r), Some((id, at))) if r.id == id => Some((id, at)),
            (Some(r), _) => Some((r.id, f.frame)),
            (None, _) => None,
        };
        let Some(r) = refinery else { return };
        let settled = t.refinery_since.is_some_and(|(_, at)| f.frame >= at + every);
        if crew.len() >= want || !settled || !TpState::due(last, f.frame, every) {
            return;
        }
        let jobs: BTreeSet<UnitId> = self.pending.iter().map(|q| q.worker).chain(self.scout).chain(self.react_busy()).collect();
        let mut free: Vec<&&UnitState> = mine
            .iter()
            .filter(|u| u.unit_type == SCV && u.has(flags::COMPLETED) && !u.has(flags::CARRYING) && !u.has(flags::CONSTRUCTING) && !jobs.contains(&u.id))
            .collect();
        free.sort_by_key(|u| (dist2(u.pos, r.pos), u.id));
        let ids: Vec<UnitId> = free.iter().take(want - crew.len()).map(|u| u.id).collect();
        if ids.is_empty() {
            return;
        }
        cmds.push(Command::Unit { units: ids.clone(), order: CmdOrder::Gather(r.id), queued: false });
        let t = self.tp.as_mut().expect("checked above");
        t.gas_crew.extend(ids);
        t.last_gas = Some(f.frame);
    }

    /// The build site of `ty` near `base` (the Brain's placement, the planned main bunker site and
    /// the natural kept free) and its order with the nearest free SCV.
    fn tp_order_build(&mut self, ty: u16, f: &RawFrame, mine: &[&UnitState], base: Pos, busy: &[UnitId], cmds: &mut Vec<Command>) -> Option<bool> {
        let reserved: Vec<(Tile, u16)> = self.pending.iter().map(|q| (q.tile, q.ty)).chain(self.react_reserved_sites()).collect();
        let tile = self.place(ty, f, mine, base, &reserved)?;
        Some(self.order_build_at(ty, tile, f, mine, busy, cmds))
    }

    /// 022-R3 / R5 / R6 orders after the 021-R8 bunker step: refinery, factory, machine shop, siege
    /// mode research and the natural command center. Returns (minerals held from everything,
    /// minerals held from marines).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn tp_build(&mut self, f: &RawFrame, mine: &[&UnitState], base: Pos, busy: &[UnitId], cmds: &mut Vec<Command>, minerals: &mut i32, gas: &mut i32) -> (i32, i32) {
        let Some(t) = self.tp.as_ref() else { return (0, 0) };
        let (p, last_shop, last_research) = (t.p.clone(), t.last_shop, t.last_research);
        let matchup = t.matchup();
        let (mut hold_all, mut hold_army) = (0, 0);
        let has = |s: &Self, ty: u16| mine.iter().any(|u| u.unit_type == ty) || s.pending.iter().any(|q| q.ty == ty);
        let rax_done = mine.iter().any(|u| u.unit_type == BARRACKS && done(u));
        let supply = u32::from(f.self_state.supply_used);
        // The price of a refinery, factory, machine shop or research that waits for minerals is
        // held from everything after it (SCVs, barracks, marines): 022-R3 orders them at 12 supply,
        // at 100 gas and once the previous step is complete, and 022-AC4 wants the factory by 3:30
        // in D too (OpenBW smoke: held from marines only, D games ordered it at 3:15–3:31; from
        // nothing, at 3:55–4:05; from everything, by 3:13 in all 64 dev games; 메인 임시 결정).
        // D's bunker and second barracks are ordered at R, before the refinery has 100 gas.
        fn stats(s: &mut Brain) -> &mut TpStats {
            &mut s.tp.as_mut().expect("the 022 teacher").stats
        }

        // Refinery at 12 supply once the barracks is ordered (and again if it is lost).
        if !has(self, REFINERY) && has(self, BARRACKS) && supply >= 2 * p.refinery_supply {
            if *minerals >= cost(REFINERY) {
                if self.tp_order_build(REFINERY, f, mine, base, busy, cmds) == Some(true) {
                    *minerals -= cost(REFINERY);
                    stats(self).refinery_frame.get_or_insert(f.frame);
                }
            } else if self.place(REFINERY, f, mine, base, &[]).is_some() {
                hold_all += cost(REFINERY);
            }
        }
        // Factory at 100 gas once the barracks is complete.
        if !has(self, FACTORY) && rax_done && *gas >= p.factory_gas {
            if *minerals >= cost(FACTORY) {
                if self.tp_order_build(FACTORY, f, mine, base, busy, cmds) == Some(true) {
                    *minerals -= cost(FACTORY);
                    *gas -= gas_cost(FACTORY);
                    stats(self).factory_frame.get_or_insert(f.frame);
                }
            } else {
                hold_all += cost(FACTORY);
            }
        }
        // Machine shop once the factory is complete.
        let factory = mine.iter().filter(|u| u.unit_type == FACTORY && done(u) && !u.has(flags::LIFTED)).min_by_key(|u| u.id).copied();
        let shop = mine.iter().any(|u| u.unit_type == MACHINE_SHOP);
        if let Some(fa) = factory.filter(|fa| !fa.has_addon && !shop && !fa.has(flags::TRAINING) && TpState::due(last_shop, f.frame, p.retarget)) {
            if *minerals >= cost(MACHINE_SHOP) && *gas >= gas_cost(MACHINE_SHOP) {
                cmds.push(Command::BuildAddon { building: fa.id, unit_type: MACHINE_SHOP });
                *minerals -= cost(MACHINE_SHOP);
                *gas -= gas_cost(MACHINE_SHOP);
                let t = self.tp.as_mut().expect("checked above");
                t.last_shop = Some(f.frame);
                t.stats.shop_frame.get_or_insert(f.frame);
            } else {
                hold_all += cost(MACHINE_SHOP);
            }
        }
        // Siege mode research once the machine shop is complete.
        let bit = 1u64 << TECH_TANK_SIEGE;
        let research_started = (f.self_state.tech | f.self_state.researching) & bit != 0;
        let shop_done = mine.iter().filter(|u| u.unit_type == MACHINE_SHOP && done(u)).min_by_key(|u| u.id).copied();
        if let Some(sh) = shop_done.filter(|_| !research_started && TpState::due(last_research, f.frame, p.retarget)) {
            if *minerals >= 150 && *gas >= 150 {
                cmds.push(Command::Research { building: sh.id, tech: TECH_TANK_SIEGE });
                *minerals -= 150;
                *gas -= 150;
                let t = self.tp.as_mut().expect("checked above");
                t.last_research = Some(f.frame);
                t.stats.siege_frame.get_or_insert(f.frame);
            } else {
                hold_all += 150;
            }
        }
        // The natural (022-R5 M, R6 D).
        let r = self.react.as_ref().expect("the 022 teacher reuses the 021-R8 state");
        let (branch, natural, last_contact) = (r.branch, r.natural, r.last_contact);
        if let Some(n) = natural.filter(|_| !self.natural_started(mine)) {
            let tanks = Self::tanks_done(mine);
            let ready = match (branch, matchup) {
                (Branch::Macro, Matchup::Protoss) => has(self, FACTORY),
                (Branch::Macro, Matchup::Terran) => tanks >= p.expand_tanks,
                (Branch::Defence, _) => tanks >= p.expand_tanks && f.frame >= last_contact + p.quiet,
            };
            if ready {
                if *minerals >= p.expand_minerals && self.order_build_at(CC, n, f, mine, busy, cmds) {
                    *minerals -= cost(CC);
                    stats(self).expand_frame.get_or_insert(f.frame);
                } else if branch == Branch::Macro && matchup == Matchup::Protoss && f.frame >= p.expand_by {
                    hold_all += p.expand_minerals;
                } else {
                    hold_army += p.expand_minerals;
                }
            }
        }
        (hold_all, hold_army)
    }

    /// 022-R3: siege tanks from each complete, idle factory with a complete machine shop once
    /// siege mode research has started. Returns minerals to hold from marines (a tank waits for
    /// minerals while gas is there).
    pub(crate) fn tp_train(&mut self, f: &RawFrame, mine: &[&UnitState], minerals: &mut i32, gas: &mut i32, supply_free: &mut i32, cmds: &mut Vec<Command>) -> i32 {
        if self.tp.is_none() || (f.self_state.tech | f.self_state.researching) & (1u64 << TECH_TANK_SIEGE) == 0 {
            return 0;
        }
        let shops: Vec<Pos> = mine.iter().filter(|u| u.unit_type == MACHINE_SHOP && done(u)).map(|u| u.pos).collect();
        let mut factories: Vec<&&UnitState> = mine
            .iter()
            .filter(|u| u.unit_type == FACTORY && done(u) && u.has_addon && !u.has(flags::TRAINING) && !u.has(flags::LIFTED))
            .filter(|u| shops.iter().any(|s| within(*s, u.pos, 4)))
            .collect();
        factories.sort_by_key(|u| u.id);
        for fa in factories {
            if *minerals >= cost(SIEGE_TANK) && *gas >= gas_cost(SIEGE_TANK) && *supply_free >= supply_cost(SIEGE_TANK) {
                cmds.push(Command::Train { building: fa.id, unit_type: SIEGE_TANK });
                *minerals -= cost(SIEGE_TANK);
                *gas -= gas_cost(SIEGE_TANK);
                *supply_free -= supply_cost(SIEGE_TANK);
                self.tp.as_mut().expect("checked above").stats.tank_frame.get_or_insert(f.frame);
            } else if *gas >= gas_cost(SIEGE_TANK) && *supply_free >= supply_cost(SIEGE_TANK) {
                return cost(SIEGE_TANK);
            }
        }
        0
    }

    /// 022-R3: at most `marines_before_factory` marines in M until the factory is ordered.
    pub(crate) fn tp_marine_cap(&self, mine: &[&UnitState]) -> Option<usize> {
        let t = self.tp.as_ref()?;
        let factory = mine.iter().any(|u| u.unit_type == FACTORY) || self.pending.iter().any(|q| q.ty == FACTORY);
        Some(if self.react_branch() == Some(Branch::Macro) && !factory { t.p.marines_before_factory } else { usize::MAX })
    }

    /// 022-R5: `scv_per_cc` SCVs per complete command center, at most `scv_max`.
    pub(crate) fn tp_max_workers(&self, mine: &[&UnitState]) -> Option<usize> {
        let t = self.tp.as_ref()?;
        let ccs = mine.iter().filter(|u| is_resource_depot(u.unit_type) && done(u)).count().max(1);
        Some((t.p.scv_per_cc * ccs).min(t.p.scv_max))
    }

    /// 022-R5 / R6 barracks cap: 1 until the natural command center is complete, then 3 against
    /// Protoss and 2 against Terran; at least 2 in D.
    pub(crate) fn tp_barracks_cap(&self, mine: &[&UnitState]) -> usize {
        let Some(t) = self.tp.as_ref() else { return 1 };
        let macro_cap = match (self.natural_done(mine), t.matchup()) {
            (false, _) => 1,
            (true, Matchup::Protoss) => t.p.barracks_p,
            (true, Matchup::Terran) => t.p.barracks_t,
        };
        match self.react_branch() {
            Some(Branch::Defence) => macro_cap.max(t.p.barracks_d),
            _ => macro_cap,
        }
    }

    /// 022-R6 tank spot: `tank_wait` tiles behind the bunker toward its command center — D's
    /// defence base, in M the natural once it stands, else the main (its planned bunker site) —
    /// or toward the way out of that base when it has no bunker.
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

    /// 022-R7 / R8 tank and attack control, after the marines' bunker orders and before the home
    /// guard: siege mode by the siege rule (every observation), and every `retarget` frames the
    /// attack group (every tank and marine not kept for a bunker when the attack condition holds,
    /// else the ones already out, more than `OUT_TILES` from every command center) attack-moves
    /// on the Brain's target with units far ahead of the tanks waiting at its front, and the other tanks go to
    /// the tank spot.
    pub(crate) fn tp_army(&mut self, f: &RawFrame, mine: &[&UnitState], kept: &[UnitId], cmds: &mut Vec<Command>) {
        let Some(t) = self.tp.as_ref() else { return };
        let p = t.p.clone();
        let (sieging, unsieging) = (t.order_sieging, t.order_unsieging);
        let me = self.me;
        let researched = f.self_state.tech & (1u64 << TECH_TANK_SIEGE) != 0;
        let enemy_ground: Vec<Pos> = f.units.iter().filter(|u| is_enemy(u, me) && is_ground_army_type(u.unit_type)).map(|u| u.pos).collect();
        let near = |at: Pos, tiles: i64| enemy_ground.iter().any(|e| within(*e, at, tiles));
        let tanks: Vec<&UnitState> = mine.iter().copied().filter(|u| is_tank(u.unit_type) && u.has(flags::COMPLETED) && u.loaded_in.is_none()).collect();
        // Siege timers from the tanks seen sieged (no ids from the teacher's own orders).
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

        // 022-R7: the siege rule, at every observation.
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

        // Moves every `retarget` frames.
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
            // 022-R8 hunting (no enemy building remembered, every start scouted, no enemy in
            // sight): the shared brain's stalest cells, one per marine squad of 12 and the first
            // for the tanks, without waiting for the tanks (nothing to fight; OpenBW smoke: the
            // whole army walking to one cell at tank speed left 60-minute draws, 메인 임시 결정).
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
            } else if let Some(target) = self.attack_target(f, lead) {
                let lead_d = dist2(lead, target);
                let gap = p.group_gap * 32;
                let (mut go, mut wait) = (Vec::new(), Vec::new());
                for u in group_tanks.iter().filter(|u| mobile(u)).chain(group_marines.iter()) {
                    let ahead = dist2(u.pos, lead) > gap * gap && dist2(u.pos, target) < lead_d;
                    if ahead { wait.push(u.id) } else { go.push(u.id) }
                }
                for chunk in go.chunks(12) {
                    cmds.push(Command::Unit { units: chunk.to_vec(), order: CmdOrder::Attack(Target::Pos(target)), queued: false });
                }
                // Waiting = attack-moving back to the front of the group (`group_gap` − 2 tiles from
                // the tanks' centre toward the target), where they stop. Not Hold: units holding on
                // a ramp cannot be pushed aside and blocked the tanks behind them for good (OpenBW
                // smoke, 메인 임시 결정).
                let front = toward(lead, target, (p.group_gap - 2) as f64);
                for chunk in wait.chunks(12) {
                    cmds.push(Command::Unit { units: chunk.to_vec(), order: CmdOrder::Attack(Target::Pos(front)), queued: false });
                }
                acted |= !(go.is_empty() && wait.is_empty());
            }
        }
        if let Some(s) = spot {
            let home: Vec<UnitId> = tanks.iter().filter(|u| mobile(u) && !in_group(u) && !within(u.pos, s, 2)).map(|u| u.id).collect();
            for chunk in home.chunks(12) {
                cmds.push(Command::Unit { units: chunk.to_vec(), order: CmdOrder::Move(Target::Pos(s)), queued: false });
            }
            acted |= !home.is_empty();
        }
        if acted {
            self.tp.as_mut().expect("checked above").last_tanks = Some(f.frame);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::react::MapHints;
    use crate::{Bot, Kit, T0Build};

    /// Our start (10, 10), the enemy start (110, 110), the natural (10, 26); chokes past the main
    /// (24, 10), past the natural (10, 40) and at the enemy (100, 100) — the 021-R8 test map.
    const OURS: Tile = (10, 10);
    const THEIRS: Tile = (110, 110);
    const NATURAL: Tile = (10, 26);
    const DRAGOON: u16 = 66;

    fn info(enemy: Race) -> GameInfo {
        let n = 128usize * 128;
        GameInfo {
            map_name: "t".into(),
            map_hash: [0; 32],
            map_size_tiles: (128, 128),
            self_player: 0,
            players: vec![
                PlayerInfo { id: 0, name: "a".into(), race: Race::Terran, start_tile: Some(OURS), is_self: true, is_enemy: false },
                PlayerInfo { id: 1, name: "b".into(), race: enemy, start_tile: None, is_self: false, is_enemy: true },
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

    fn hints() -> MapHints {
        MapHints { base_locs: vec![OURS, THEIRS, NATURAL, (60, 60)], chokes: vec![(24, 10), (10, 40), (100, 100)] }
    }

    fn teacher_vs(enemy: Race) -> Brain {
        let i = info(enemy);
        Brain::with_hints(&i, Kit::for_build(Race::Terran, T0Build::TerranTp.for_lobby(&i)), &hints())
    }

    fn teacher() -> Brain {
        teacher_vs(Race::Protoss)
    }

    fn unit(id: u32, ty: u16, pos: Pos) -> UnitState {
        UnitState { id, owner: 0, unit_type: ty, pos, hp: 40, flags: flags::COMPLETED | flags::IDLE, ..Default::default() }
    }

    fn enemy(id: u32, ty: u16, pos: Pos) -> UnitState {
        UnitState { owner: 1, ..unit(id, ty, pos) }
    }

    fn building(id: u32, ty: u16, pos: Pos) -> UnitState {
        UnitState { flags: flags::BEING_CONSTRUCTED, ..unit(id, ty, pos) }
    }

    fn home() -> Pos {
        depot_center(OURS)
    }

    /// CC, a completed barracks, a mineral field, two depots and `scvs` SCVs at home.
    fn base(scvs: u32) -> Vec<UnitState> {
        let mut v = vec![
            unit(1, CC, home()),
            unit(2, BARRACKS, (home().0 + 200, home().1 + 160)),
            UnitState { owner: 11, ..unit(3, MINERAL_FIELDS[0], (home().0 - 200, home().1)) },
            unit(6, DEPOT, (home().0 - 48, home().1 - 160)),
            unit(7, DEPOT, (home().0 + 80, home().1 - 160)),
        ];
        v.extend((0..scvs).map(|i| unit(100 + i, SCV, (home().0 - 100, home().1 + 10 * i as u16))));
        v
    }

    /// A command center, a mineral field and `scvs` SCVs.
    fn cc_only(scvs: u32) -> Vec<UnitState> {
        let mut v = vec![unit(1, CC, home()), UnitState { owner: 11, ..unit(3, MINERAL_FIELDS[0], (home().0 - 200, home().1)) }];
        v.extend((0..scvs).map(|i| unit(100 + i, SCV, (home().0 - 100, home().1 + 10 * i as u16))));
        v
    }

    const GEYSER_AT: Pos = (10 * 32 + 64 + 7 * 32, 10 * 32 + 48 - 3 * 32);
    const FACTORY_AT: Pos = (10 * 32 + 64 + 320, 10 * 32 + 48 + 96);

    fn geyser() -> UnitState {
        UnitState { owner: 11, ..unit(5, GEYSER, GEYSER_AT) }
    }

    fn factory() -> UnitState {
        unit(8, FACTORY, FACTORY_AT)
    }

    fn shop() -> UnitState {
        unit(9, MACHINE_SHOP, (FACTORY_AT.0 + 96, FACTORY_AT.1 + 16))
    }

    fn frame(n: u32, units: Vec<UnitState>, minerals: i32, supply_used: u16) -> RawFrame {
        RawFrame {
            frame: n,
            self_state: PlayerState { minerals, supply_used, supply_total: 200, upgrades: vec![0; 63], ..Default::default() },
            units,
            visibility: BitGrid::from_bytes(128, 128, &[1; 128 * 128]),
            explored: BitGrid::new(128, 128),
            creep: BitGrid::new(128, 128),
            events: vec![],
            effects: vec![],
        }
    }

    fn gas(mut f: RawFrame, gas: i32) -> RawFrame {
        f.self_state.gas = gas;
        f
    }

    fn sieging(mut f: RawFrame, researched: bool, researching: bool) -> RawFrame {
        let bit = 1u64 << TECH_TANK_SIEGE;
        f.self_state.tech = if researched { bit } else { 0 };
        f.self_state.researching = if researching { bit } else { 0 };
        f
    }

    fn dark(f: RawFrame) -> RawFrame {
        RawFrame { visibility: BitGrid::new(128, 128), ..f }
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

    fn ordered(act: &Act, pick: impl Fn(&CmdOrder) -> bool) -> BTreeSet<UnitId> {
        act.cmds
            .iter()
            .filter_map(|c| match c {
                Command::Unit { units, order, .. } if pick(order) => Some(units.clone()),
                _ => None,
            })
            .flatten()
            .collect()
    }

    fn branch(b: &Brain) -> Branch {
        b.react_branch().unwrap()
    }

    fn stats(b: &Brain) -> TpStats {
        b.tp_stats().unwrap()
    }

    /// 022-AC1 / 022-R4: rules ①–④ (Protoss and Terran) and ③ read the current observation at
    /// their boundaries.
    #[test]
    fn rush_rules_022_r4() {
        let p = TpParams::default();
        let depots = [home()];
        let others = [THEIRS];
        let none = BTreeSet::new();
        let cx = TpContext { depots: &depots, starts: &others, scout: None, prod_seen: &none };
        let at = |n: u32, units: Vec<UnitState>| tp_rules(&p, &frame(n, units, 0, 0), 0, &cx);
        let far = depot_center((60, 60));
        let many = |ty: u16, k: u32| (0..k).map(|i| enemy(30 + i, ty, (far.0, far.1 + 8 * i as u16))).collect::<Vec<_>>();
        // ①P two gateways (one unfinished) before 2:40 (3,840); one is not enough.
        let gates = vec![enemy(20, GATEWAY, far), UnitState { owner: 1, ..building(21, GATEWAY, (far.0 + 160, far.1)) }];
        assert_eq!(at(3_839, gates.clone()), vec![TpRule::Gates]);
        assert_eq!(at(3_840, gates.clone()), vec![]);
        assert_eq!(at(3_000, gates[..1].to_vec()), vec![]);
        // ②P two zealots before 3:30 (5,040), anywhere; one is not enough.
        assert_eq!(at(5_039, many(ZEALOT, 2)), vec![TpRule::Zealots]);
        assert_eq!(at(5_040, many(ZEALOT, 2)), vec![]);
        assert_eq!(at(4_000, many(ZEALOT, 1)), vec![]);
        // ①T / ②T the same with barracks and four marines.
        let rax = vec![enemy(20, BARRACKS, far), UnitState { owner: 1, ..building(21, BARRACKS, (far.0 + 160, far.1)) }];
        assert_eq!(at(3_839, rax.clone()), vec![TpRule::Rax]);
        assert_eq!(at(3_840, rax.clone()), vec![]);
        assert_eq!(at(3_000, rax[..1].to_vec()), vec![]);
        assert_eq!(at(5_039, many(MARINE, 4)), vec![TpRule::Marines]);
        assert_eq!(at(5_040, many(MARINE, 4)), vec![]);
        assert_eq!(at(4_000, many(MARINE, 3)), vec![]);
        // ③ (021-AC3): two enemy army units within 20 tiles of a command center, after 3:30 too;
        // one is contact only; 21 tiles out, workers and buildings are no army; nothing from 8:00.
        let near = |id: u32, ty: u16| enemy(id, ty, (home().0 + 19 * 32, home().1 + id as u16));
        assert_eq!(at(9_000, vec![near(9, ZEALOT), near(10, DRAGOON)]), vec![TpRule::Army]);
        assert_eq!(at(9_000, vec![near(9, MARINE), near(10, MARINE)]), vec![TpRule::Army]);
        assert_eq!(at(9_000, vec![near(9, ZEALOT)]), vec![]);
        assert_eq!(at(9_000, vec![near(9, ZEALOT), enemy(10, ZEALOT, (home().0 + 21 * 32, home().1))]), vec![]);
        assert_eq!(at(9_000, vec![near(9, PROBE), near(10, SCV), near(11, ZEALOT), near(12, PYLON)]), vec![]);
        assert_eq!(at(11_520, vec![near(9, ZEALOT), near(10, ZEALOT)]), vec![]);
        // ④P the scout within 3 tiles of the visible enemy start nexus, 2:30 ≤ t < 3:30, no
        // gateway ever seen within 12 tiles of that start.
        let main = depot_center(THEIRS);
        let scout_at = |tiles: u16| Some((main.0 - tiles * 32, main.1));
        let proxy = |n: u32, depot: u16, scout: Option<Pos>, seen: &BTreeSet<(Tile, u16)>| {
            let cx = TpContext { depots: &depots, starts: &others, scout, prod_seen: seen };
            tp_rules(&p, &frame(n, vec![enemy(40, depot, main)], 0, 0), 0, &cx)
        };
        assert_eq!(proxy(3_600, NEXUS, scout_at(3), &none), vec![TpRule::ProxyGate]);
        assert_eq!(proxy(5_039, NEXUS, scout_at(3), &none), vec![TpRule::ProxyGate]);
        assert_eq!(proxy(3_599, NEXUS, scout_at(3), &none), vec![], "before 2:30");
        assert_eq!(proxy(5_040, NEXUS, scout_at(3), &none), vec![], "from 3:30");
        assert_eq!(proxy(4_000, NEXUS, scout_at(4), &none), vec![], "scout 4 tiles away");
        assert_eq!(proxy(4_000, NEXUS, None, &none), vec![], "no scout");
        let gate_seen = BTreeSet::from([(THEIRS, GATEWAY)]);
        assert_eq!(proxy(4_000, NEXUS, scout_at(3), &gate_seen), vec![], "a gateway seen there");
        // ④T the same at a command center with barracks.
        assert_eq!(proxy(3_600, CC, scout_at(3), &none), vec![TpRule::ProxyRax]);
        assert_eq!(proxy(3_599, CC, scout_at(3), &none), vec![]);
        assert_eq!(proxy(5_040, CC, scout_at(3), &none), vec![]);
        assert_eq!(proxy(4_000, CC, scout_at(4), &none), vec![]);
        assert_eq!(proxy(4_000, CC, scout_at(3), &BTreeSet::from([(THEIRS, BARRACKS)])), vec![]);
        assert_eq!(proxy(4_000, CC, scout_at(3), &gate_seen), vec![TpRule::ProxyRax], "a gateway is no barracks");
    }

    /// 022-R4 ④: a gateway seen near the enemy start in an earlier observation (memory) keeps
    /// the proxy rule off; without it the scout at the nexus at 2:30 identifies a proxy.
    #[test]
    fn proxy_rule_remembers_gateways_022_r4() {
        let main = depot_center(THEIRS);
        let scout_units = |extra: Vec<UnitState>| -> Vec<UnitState> {
            let mut u = base(12);
            u.push(UnitState { flags: flags::COMPLETED | flags::MOVING, ..unit(150, SCV, (main.0 - 64, main.1)) });
            u.push(enemy(9, NEXUS, main));
            u.extend(extra);
            u
        };
        let mut b = teacher();
        b.scout = Some(150);
        b.act(&frame(3_000, scout_units(vec![enemy(11, GATEWAY, (main.0 + 200, main.1 + 100))]), 0, 24));
        b.act(&frame(3_700, scout_units(vec![]), 0, 24));
        assert!(!stats(&b).rule_frames.contains_key(&TpRule::ProxyGate));
        assert_eq!(branch(&b), Branch::Macro);
        let mut c = teacher();
        c.scout = Some(150);
        c.act(&frame(3_700, scout_units(vec![]), 0, 24));
        assert_eq!(stats(&c).rule_frames.get(&TpRule::ProxyGate), Some(&3_700));
        assert_eq!((branch(&c), stats(&c).rush_rule), (Branch::Defence, Some(TpRule::ProxyGate)));
    }

    /// 022-AC1 / 022-R3 opening: depots at 9 and 15 supply, barracks at 11, refinery at 12, the
    /// factory at 100 gas, the machine shop once the factory is complete, siege mode research once
    /// the shop is, tanks once the research has started; at most 4 marines before the factory.
    #[test]
    fn opening_022_r3() {
        let act = teacher().act(&dark(frame(500, cc_only(8), 200, 16)));
        assert!(builds(&act, DEPOT).is_empty(), "8 supply: {:?}", act.cmds);
        let act = teacher().act(&dark(frame(508, cc_only(9), 200, 18)));
        assert_eq!(builds(&act, DEPOT).len(), 1, "9 supply");
        let mut u = cc_only(10);
        u.push(unit(6, DEPOT, (home().0, home().1 - 160)));
        assert!(builds(&teacher().act(&dark(frame(1_500, u.clone(), 200, 20))), BARRACKS).is_empty(), "10 supply");
        assert_eq!(builds(&teacher().act(&dark(frame(1_500, u.clone(), 200, 22))), BARRACKS).len(), 1, "11 supply");
        // Refinery at 12 supply on the geyser, after the barracks order.
        u.push(building(2, BARRACKS, (home().0 + 200, home().1 + 160)));
        u.push(geyser());
        let act = teacher().act(&dark(frame(2_000, u.clone(), 200, 22)));
        assert!(builds(&act, REFINERY).is_empty(), "11 supply: {:?}", act.cmds);
        let mut b = teacher();
        let act = b.act(&dark(frame(2_000, u.clone(), 200, 24)));
        assert_eq!(builds(&act, REFINERY), vec![building_tile(&geyser())], "12 supply");
        assert_eq!(stats(&b).refinery_frame, Some(2_000));
        assert!(builds(&act, DEPOT).is_empty(), "the second depot waits for 15");
        assert_eq!(builds(&teacher().act(&dark(frame(2_000, u, 200, 30))), DEPOT).len(), 1, "15 supply");

        // Factory at 100 gas (refinery done, barracks done).
        let mut u = base(14);
        u.push(unit(5, REFINERY, GEYSER_AT));
        assert!(builds(&teacher().act(&gas(frame(2_400, u.clone(), 300, 30), 99)), FACTORY).is_empty());
        let mut b = teacher();
        let act = b.act(&gas(frame(2_400, u.clone(), 300, 30), 100));
        assert_eq!(builds(&act, FACTORY).len(), 1, "{:?}", act.cmds);
        assert_eq!(stats(&b).factory_frame, Some(2_400));
        // Machine shop once the factory is complete.
        let mut v = u.clone();
        v.push(building(8, FACTORY, FACTORY_AT));
        let act = teacher().act(&gas(frame(3_000, v, 100, 30), 100));
        assert!(!act.cmds.iter().any(|c| matches!(c, Command::BuildAddon { .. })), "{:?}", act.cmds);
        let mut v = u.clone();
        v.push(factory());
        let mut b = teacher();
        let act = b.act(&gas(frame(3_000, v, 100, 30), 100));
        assert!(act.cmds.contains(&Command::BuildAddon { building: 8, unit_type: MACHINE_SHOP }), "{:?}", act.cmds);
        assert_eq!(stats(&b).shop_frame, Some(3_000));
        // Siege mode research once the shop is complete (not while it is built, not twice).
        let mut v = u.clone();
        v.push(UnitState { has_addon: true, ..factory() });
        v.push(building(9, MACHINE_SHOP, shop().pos));
        let act = teacher().act(&gas(frame(3_500, v.clone(), 200, 30), 200));
        assert!(!act.cmds.iter().any(|c| matches!(c, Command::Research { .. })), "{:?}", act.cmds);
        let last = v.len() - 1;
        v[last] = shop();
        let mut b = teacher();
        let act = b.act(&gas(frame(3_500, v.clone(), 200, 30), 200));
        assert!(act.cmds.contains(&Command::Research { building: 9, tech: TECH_TANK_SIEGE }), "{:?}", act.cmds);
        assert_eq!(trains(&act, SIEGE_TANK), 0, "tanks after the research order");
        assert_eq!(stats(&b).siege_frame, Some(3_500));
        let act = teacher().act(&sieging(gas(frame(3_500, v.clone(), 200, 30), 200), false, true));
        assert!(!act.cmds.iter().any(|c| matches!(c, Command::Research { .. })), "researching: {:?}", act.cmds);
        // Tanks once the research has started.
        let mut b = teacher();
        let act = b.act(&sieging(gas(frame(3_600, v, 300, 30), 200), false, true));
        assert!(act.cmds.contains(&Command::Train { building: 8, unit_type: SIEGE_TANK }), "{:?}", act.cmds);
        assert_eq!(stats(&b).tank_frame, Some(3_600));

        // At most 4 marines before the factory (macro branch); continuous once it is ordered.
        let marines = |k: u32| (0..k).map(|i| unit(200 + i, MARINE, (home().0 + 100, home().1 + 4 * i as u16))).collect::<Vec<_>>();
        let mut u = base(14);
        u.extend(marines(4));
        assert_eq!(trains(&teacher().act(&frame(2_400, u, 1_000, 36)), MARINE), 0);
        let mut u = base(14);
        u.extend(marines(3));
        assert_eq!(trains(&teacher().act(&frame(2_400, u, 1_000, 34)), MARINE), 1);
        let mut u = base(14);
        u.extend(marines(4));
        u.push(factory());
        assert_eq!(trains(&teacher().act(&frame(3_000, u, 1_000, 36)), MARINE), 1);
    }

    /// 022-AC1 user constraint (as 021-R8): a macro game with no rush evidence — the enemy main
    /// seen with one gateway and probes, a probe scouting our base — issues no bunker order from
    /// 0:00 to 6:00, stays in M and expands (against Protoss once the factory is ordered).
    #[test]
    fn no_main_bunker_without_rush_evidence_022_r5() {
        let mut b = teacher();
        let (mut bunkers, mut expanded) = (0, false);
        for n in (8..=8_640).step_by(8) {
            let scvs = (4 + n / 300).min(22);
            let mut u = if n < 1_600 { cc_only(scvs) } else { base(scvs) };
            // The factory the teacher orders at 100 gas stands from 2,400 (else its pending order
            // holds its price).
            if n >= 2_400 {
                u.push(building(8, FACTORY, FACTORY_AT));
            }
            if (2_800..3_100).contains(&n) {
                let m = depot_center(THEIRS);
                u.push(enemy(80, NEXUS, m));
                u.push(enemy(81, GATEWAY, (m.0 + 200, m.1 + 100)));
                u.extend((0..8).map(|i| enemy(82 + i, PROBE, (m.0 - 100, m.1 + 8 * i as u16))));
            }
            if n >= 4_000 {
                u.push(enemy(95, PROBE, (home().0 + 64, home().1 + 64)));
            }
            let act = b.act(&gas(frame(n, u, 600, (2 * scvs) as u16), if n >= 1_600 { 150 } else { 0 }));
            bunkers += builds(&act, BUNKER).len();
            expanded |= builds(&act, CC).contains(&NATURAL);
        }
        assert_eq!((bunkers, branch(&b), stats(&b).rush_frame), (0, Branch::Macro, None));
        assert!(expanded && stats(&b).expand_frame.is_some() && stats(&b).factory_frame < stats(&b).expand_frame);
    }

    /// 022-AC1 / 022-R6: R switches to D at that observation, and the same observation orders
    /// the main bunker (on the ground path toward the natural, 5–7 tiles out) and the second
    /// barracks; the record keeps the rule and the frame.
    #[test]
    fn rush_switches_to_defence_022_r6() {
        for (enemy_race, ty, k, rule) in [(Race::Protoss, ZEALOT, 2, TpRule::Zealots), (Race::Terran, MARINE, 4, TpRule::Marines)] {
            let mut b = teacher_vs(enemy_race);
            let mut u = base(12);
            u.extend((0..k).map(|i| enemy(90 + i, ty, depot_center((60, 60 + i as u16)))));
            let act = b.act(&frame(3_000, u, 250, 30));
            assert_eq!(branch(&b), Branch::Defence);
            let bunker = builds(&act, BUNKER);
            assert_eq!(bunker.len(), 1, "{:?}", act.cmds);
            let c = bunker_center(bunker[0]);
            let d = ((dist2(c, home()) as f64).sqrt() / 32.0, c.1 as i64 - home().1 as i64);
            assert!(d.0 >= 5.0 && d.0 <= 7.0 && d.1 > 0, "{bunker:?} {d:?}");
            assert_eq!(builds(&act, BARRACKS).len(), 1, "the second barracks: {:?}", act.cmds);
            let s = stats(&b);
            assert_eq!((s.rush_frame, s.rush_rule, s.defences, s.branch), (Some(3_000), Some(rule), 1, Branch::Defence));
        }
    }

    /// 022-R6: D → M with 8 marines (loaded ones too), 2 tanks and 60 s without contact, or with
    /// 4 tanks whatever the contact; with 4 tanks R keeps M.
    #[test]
    fn defence_returns_to_macro_022_r6() {
        let army = |marines: u32, tanks: u32| {
            let mut u = base(12);
            u.extend((0..marines).map(|i| unit(200 + i, MARINE, (home().0 + 100, home().1 + 4 * i as u16))));
            u.extend((0..tanks).map(|i| unit(300 + i, SIEGE_TANK, (home().0 + 60, home().1 + 100 + 8 * i as u16))));
            u
        };
        let zealots = |mut u: Vec<UnitState>| {
            u.push(enemy(90, ZEALOT, (home().0 + 9 * 32, home().1)));
            u.push(enemy(91, ZEALOT, (home().0 + 9 * 32, home().1 + 20)));
            u
        };
        let mut b = teacher();
        b.act(&frame(6_000, zealots(army(2, 0)), 0, 40));
        assert_eq!(branch(&b), Branch::Defence, "③ army");
        b.act(&frame(6_000 + 1_439, army(8, 2), 0, 40));
        assert_eq!(branch(&b), Branch::Defence, "59.96 s");
        b.act(&frame(6_000 + 1_440, army(7, 2), 0, 40));
        assert_eq!(branch(&b), Branch::Defence, "7 marines");
        b.act(&frame(6_000 + 1_440, army(8, 1), 0, 40));
        assert_eq!(branch(&b), Branch::Defence, "1 tank");
        let mut loaded = army(8, 2);
        loaded.iter_mut().filter(|u| u.unit_type == MARINE).take(4).for_each(|u| u.loaded_in = Some(77));
        b.act(&frame(6_000 + 1_440, loaded, 0, 40));
        assert_eq!(branch(&b), Branch::Macro, "8 marines (4 loaded), 2 tanks, 60 s");
        assert_eq!(stats(&b).macro_frame, Some(7_440));
        // Under contact: 3 tanks stay in D, 4 return to M and keep it.
        let mut c = teacher();
        c.act(&frame(6_000, zealots(army(0, 3)), 0, 40));
        assert_eq!(branch(&c), Branch::Defence);
        c.act(&frame(6_048, zealots(army(0, 4)), 0, 40));
        assert_eq!(branch(&c), Branch::Macro, "4 tanks");
        c.act(&frame(6_096, zealots(army(0, 4)), 0, 40));
        assert_eq!((branch(&c), stats(&c).defences), (Branch::Macro, 1), "R with 4 tanks keeps M");
    }

    /// 022-R5: against Protoss the natural at 400 minerals once the factory is ordered, its price
    /// held from everything from 4:00; against Terran after 2 tanks; in D after 2 tanks and 60 s
    /// without contact.
    #[test]
    fn natural_expansion_022_r5() {
        let with = |extra: Vec<UnitState>| {
            let mut u = base(16);
            u.extend(extra);
            u
        };
        let tanks = |k: u32| (0..k).map(|i| unit(300 + i, SIEGE_TANK, (home().0 + 60, home().1 + 100 + 8 * i as u16))).collect::<Vec<_>>();
        assert!(builds(&teacher().act(&frame(3_000, with(vec![]), 400, 40)), CC).is_empty(), "no factory");
        let mut b = teacher();
        let act = b.act(&frame(3_000, with(vec![building(8, FACTORY, FACTORY_AT)]), 400, 40));
        assert_eq!(builds(&act, CC), vec![NATURAL], "{:?}", act.cmds);
        assert_eq!(stats(&b).expand_frame, Some(3_000));
        let act = teacher().act(&frame(3_000, with(vec![building(8, FACTORY, FACTORY_AT)]), 399, 40));
        assert!(builds(&act, CC).is_empty());
        // Before 4:00 the price is held from marines only (an SCV is trained); from 4:00 from
        // everything.
        assert_eq!(trains(&teacher().act(&frame(5_759, with(vec![factory()]), 399, 40)), SCV), 1);
        assert_eq!(trains(&teacher().act(&frame(5_760, with(vec![factory()]), 399, 40)), SCV), 0);
        // Terran: two tanks.
        let act = teacher_vs(Race::Terran).act(&frame(5_000, with([vec![factory()], tanks(1)].concat()), 400, 40));
        assert!(builds(&act, CC).is_empty(), "{:?}", act.cmds);
        let act = teacher_vs(Race::Terran).act(&frame(5_000, with([vec![factory()], tanks(2)].concat()), 400, 40));
        assert_eq!(builds(&act, CC), vec![NATURAL]);
        // D: two tanks and 60 s after the last contact.
        let mut d = teacher();
        let mut u = with(tanks(2));
        u.push(enemy(90, ZEALOT, (home().0 + 9 * 32, home().1)));
        u.push(enemy(91, ZEALOT, (home().0 + 9 * 32, home().1 + 20)));
        let act = d.act(&frame(6_000, u, 400, 40));
        assert_eq!(branch(&d), Branch::Defence);
        assert!(builds(&act, CC).is_empty());
        // (650: the bunker and second barracks orders D keeps giving hold 250.)
        assert!(builds(&d.act(&frame(7_439, with(tanks(2)), 650, 40)), CC).is_empty());
        assert_eq!(builds(&d.act(&frame(7_440, with(tanks(2)), 650, 40)), CC), vec![NATURAL]);
        assert_eq!(branch(&d), Branch::Defence, "no marines: still D");
    }

    /// 022-AC1 / 022-R7: once siege mode is researched a tank sieges with enemy ground army within
    /// 11 tiles (not 12, not workers); a sieged tank leaves siege mode after 10 s without enemy
    /// ground army within 13 tiles when it has somewhere to go, not at its spot.
    #[test]
    fn siege_rule_022_r7() {
        let tank_at = (home().0 + 15 * 32, home().1);
        let at = |n: u32, tank: UnitState, extra: Vec<UnitState>, researched: bool| {
            let mut b = teacher();
            let mut u = base(12);
            u.push(tank);
            u.extend(extra);
            b.act(&sieging(frame(n, u, 0, 40), researched, false))
        };
        let siege = |act: &Act| ordered(act, |o| *o == CmdOrder::Siege);
        let tank = unit(300, SIEGE_TANK, tank_at);
        assert_eq!(siege(&at(6_000, tank.clone(), vec![enemy(90, ZEALOT, (tank_at.0 + 11 * 32, tank_at.1))], true)), BTreeSet::from([300]));
        assert!(siege(&at(6_000, tank.clone(), vec![enemy(90, ZEALOT, (tank_at.0 + 12 * 32, tank_at.1))], true)).is_empty());
        assert!(siege(&at(6_000, tank.clone(), vec![enemy(90, PROBE, (tank_at.0 + 5 * 32, tank_at.1))], true)).is_empty(), "a worker");
        assert!(siege(&at(6_000, tank.clone(), vec![enemy(90, ZEALOT, (tank_at.0 + 5 * 32, tank_at.1))], false)).is_empty(), "not researched");

        // Unsiege: first seen sieged at 6,000 away from its spot (15 tiles east of the main).
        let sieged = unit(300, SIEGE_TANK_SIEGED, tank_at);
        let run = |b: &mut Brain, n: u32, extra: Vec<UnitState>| {
            let mut u = base(12);
            u.push(sieged.clone());
            u.extend(extra);
            b.act(&sieging(frame(n, u, 0, 40), true, false))
        };
        let unsiege = |act: &Act| ordered(act, |o| *o == CmdOrder::Unsiege);
        let mut b = teacher();
        assert!(unsiege(&run(&mut b, 6_000, vec![])).is_empty());
        assert!(unsiege(&run(&mut b, 6_239, vec![])).is_empty());
        assert_eq!(unsiege(&run(&mut b, 6_240, vec![])), BTreeSet::from([300]));
        // An enemy within 13 tiles at 6,100 restarts the 10 s.
        let mut c = teacher();
        run(&mut c, 6_000, vec![]);
        run(&mut c, 6_100, vec![enemy(90, ZEALOT, (tank_at.0 + 13 * 32, tank_at.1))]);
        assert!(unsiege(&run(&mut c, 6_240, vec![])).is_empty());
        assert_eq!(unsiege(&run(&mut c, 6_340, vec![])), BTreeSet::from([300]));
        // At its spot it stays sieged.
        let mut d = teacher();
        let plain = base(12);
        d.act(&frame(5_000, plain.clone(), 0, 40));
        let spot = d.tp_tank_spot(&plain.iter().collect::<Vec<_>>()).unwrap();
        let mut u = base(12);
        u.push(unit(301, SIEGE_TANK_SIEGED, spot));
        d.act(&sieging(frame(6_000, u.clone(), 0, 40), true, false));
        assert!(unsiege(&d.act(&sieging(frame(6_480, u, 0, 40), true, false))).is_empty());
    }

    /// 022-R6: tanks not attacking go behind the bunker (3 tiles toward the command center).
    #[test]
    fn tanks_wait_behind_the_bunker_022_r6() {
        let mut b = teacher();
        let mut u = base(12);
        let bunker = (home().0 + 5 * 32, home().1 + 64);
        u.push(UnitState { hp: 350, ..unit(60, BUNKER, bunker) });
        u.push(unit(300, SIEGE_TANK, (home().0 - 6 * 32, home().1 - 4 * 32)));
        u.push(enemy(90, ZEALOT, depot_center((60, 60))));
        u.push(enemy(91, ZEALOT, depot_center((60, 61))));
        let act = b.act(&frame(3_000, u, 0, 40));
        assert_eq!(branch(&b), Branch::Defence);
        let spot = toward(bunker, home(), 3.0);
        assert!(act.cmds.contains(&Command::Unit { units: vec![300], order: CmdOrder::Move(Target::Pos(spot)), queued: false }), "{:?}", act.cmds);
    }

    /// 022-AC1 / 022-R8: M attacks with 4 tanks and 40 army supply (a marine 1, a tank 2), or
    /// from 12:00 with 20; units more than 8 tiles ahead of the tanks wait (Hold).
    #[test]
    fn wave_condition_022_r8() {
        let army = |marines: u32, tanks: u32| {
            let mut u = base(20);
            u.extend((0..marines).map(|i| unit(200 + i, MARINE, (home().0 + 100 + 4 * (i % 8) as u16, home().1 + 100 + 4 * (i / 8) as u16))));
            u.extend((0..tanks).map(|i| unit(300 + i, SIEGE_TANK, (home().0 + 80, home().1 + 120 + 8 * i as u16))));
            u
        };
        let attackers = |n: u32, marines: u32, tanks: u32| {
            let mut b = teacher();
            let act = b.act(&sieging(frame(n, army(marines, tanks), 0, 120), true, false));
            (ordered(&act, |o| matches!(o, CmdOrder::Attack(Target::Pos(_)))).len(), stats(&b).wave_frame)
        };
        assert_eq!(attackers(9_000, 32, 4), (36, Some(9_000)));
        assert_eq!(attackers(9_000, 31, 4), (0, None), "39 supply");
        assert_eq!(attackers(9_000, 34, 3), (0, None), "3 tanks");
        assert_eq!(attackers(17_280, 20, 0), (20, Some(17_280)));
        assert_eq!(attackers(17_279, 20, 0), (0, None), "before 12:00");
        assert_eq!(attackers(17_280, 19, 0), (0, None), "19 supply");
        // A marine far ahead of the tanks toward the target waits: it attack-moves back to the
        // group's front, 6 tiles from the tanks' centre.
        let mut b = teacher();
        let mut u = army(32, 4);
        let target = depot_center(THEIRS);
        u.push(enemy(80, NEXUS, target));
        u.iter_mut().filter(|x| x.id == 200).for_each(|x| x.pos = (home().0 + 30 * 32, home().1 + 30 * 32));
        let act = b.act(&sieging(frame(9_000, u, 0, 120), true, false));
        let lead = (home().0 + 80, home().1 + 120 + 12);
        let front = toward(lead, target, 6.0);
        assert!(ordered(&act, |o| *o == CmdOrder::Attack(Target::Pos(front))).contains(&200), "{:?}", act.cmds);
        assert!(ordered(&act, |o| *o == CmdOrder::Attack(Target::Pos(target))).contains(&300));
    }

    /// 022-R8 hunting: with no enemy building remembered, every start scouted and no enemy in
    /// sight, the marine squads of 12 go to different stalest cells and the tanks to the first.
    #[test]
    fn hunting_squads_022_r8() {
        let mut u = base(20);
        u.extend((0..32).map(|i| unit(200 + i, MARINE, (home().0 + 100 + 4 * (i % 8) as u16, home().1 + 100 + 4 * (i / 8) as u16))));
        u.extend((0..4).map(|i| unit(300 + i, SIEGE_TANK, (home().0 + 80, home().1 + 120 + 8 * i as u16))));
        let mut b = teacher();
        // Everything seen once, then a dark observation: the cells' last-seen frames decide.
        b.act(&sieging(frame(9_000, u.clone(), 0, 120), true, false));
        let act = b.act(&sieging(dark(frame(9_048, u, 0, 120)), true, false));
        let cells = b.stale_cells();
        let to = |id: UnitId| {
            act.cmds.iter().find_map(|c| match c {
                Command::Unit { units, order: CmdOrder::Attack(Target::Pos(p)), .. } if units.contains(&id) => Some(*p),
                _ => None,
            })
        };
        assert_eq!((to(300), to(200), to(212), to(224)), (Some(cells[0]), Some(cells[0]), Some(cells[1]), Some(cells[2])), "{:?}", act.cmds);
    }

    /// 022-R3 scout: the enemy main found before 2:40 keeps the scout there (circling, from 2:30 at
    /// the start depot) until 2:40, then it mines; below half hit points it leaves at once.
    #[test]
    fn scout_stays_in_the_enemy_main_022_r3() {
        let main = depot_center(THEIRS);
        let units = |hp: u16| {
            let mut u = base(12);
            u.push(UnitState { hp, flags: flags::COMPLETED | flags::MOVING, ..unit(150, SCV, (main.0 - 5 * 32, main.1)) });
            u.push(enemy(9, NEXUS, main));
            u
        };
        let moves = |act: &Act| {
            act.cmds
                .iter()
                .filter_map(|c| match c {
                    Command::Unit { units, order: CmdOrder::Move(Target::Pos(p)), .. } if units == &vec![150] => Some(*p),
                    _ => None,
                })
                .collect::<Vec<_>>()
        };
        let gathers = |act: &Act| ordered(act, |o| matches!(o, CmdOrder::Gather(_))).contains(&150);
        let mut b = teacher();
        b.scout = Some(150);
        let act = b.act(&frame(2_400, units(60), 0, 30));
        let m = moves(&act);
        assert_eq!(m.len(), 1, "{:?}", act.cmds);
        let d = (dist2(m[0], main) as f64).sqrt() / 32.0;
        assert!((5.5..=6.5).contains(&d), "circling 6 tiles out: {m:?}");
        let act = b.act(&frame(3_600, units(60), 0, 30));
        assert_eq!(moves(&act), vec![main], "from 2:30 at the nexus");
        let act = b.act(&frame(3_840, units(60), 0, 30));
        assert!(gathers(&act) && b.scout_done, "2:40: {:?}", act.cmds);
        let mut c = teacher();
        c.scout = Some(150);
        c.act(&frame(2_400, units(30), 0, 30));
        assert!(!c.scout_done, "30 HP is half");
        let act = c.act(&frame(2_448, units(29), 0, 30));
        assert!(gathers(&act) && c.scout_done, "{:?}", act.cmds);
    }

    /// 022-R9 shadow fix-ups (021-AC3 form): the bunker order memory expires after 144 frames
    /// without a bunker; a lifted factory trains no tank; the gas crew is chosen again from the
    /// SCVs' orders.
    #[test]
    fn shadow_fixups_022_r9() {
        let rush = |n: u32| {
            let mut u = base(12);
            u.push(enemy(90, ZEALOT, depot_center((60, 60))));
            u.push(enemy(91, ZEALOT, depot_center((60, 61))));
            frame(n, u, 100, 30)
        };
        let mut b = teacher();
        b.scout_done = true;
        assert_eq!(builds(&b.act(&rush(3_000)), BUNKER).len(), 1);
        assert!(builds(&b.act(&rush(3_143)), BUNKER).is_empty());
        assert_eq!(builds(&b.act(&rush(3_144)), BUNKER).len(), 1);

        let mut u = base(14);
        u.push(UnitState { has_addon: true, flags: flags::COMPLETED | flags::LIFTED, ..factory() });
        u.push(shop());
        assert_eq!(trains(&teacher().act(&sieging(gas(frame(4_000, u, 500, 30), 300), true, false)), SIEGE_TANK), 0);

        // Gas: 48 frames after the refinery is seen complete (its builder mines on its own) three
        // SCVs are sent; 48 frames later they mine minerals (not carried out): sent again.
        let mut u = base(14);
        u.push(unit(5, REFINERY, GEYSER_AT));
        let mut g = teacher();
        g.scout_done = true;
        let crew = |act: &Act| ordered(act, |o| *o == CmdOrder::Gather(5));
        assert!(crew(&g.act(&frame(3_000, u.clone(), 0, 30))).is_empty(), "just complete");
        let first = crew(&g.act(&frame(3_048, u.clone(), 0, 30)));
        assert_eq!(first.len(), 3);
        assert!(crew(&g.act(&frame(3_090, u.clone(), 0, 30))).is_empty(), "within 48 frames");
        assert_eq!(crew(&g.act(&frame(3_096, u.clone(), 0, 30))), first, "not on gas: chosen again");
        let harvest = uranium_env::types::order_id("HarvestGas").unwrap();
        for x in u.iter_mut().filter(|x| first.contains(&x.id)) {
            x.order = harvest;
        }
        assert!(crew(&g.act(&frame(3_200, u.clone(), 0, 30))).is_empty(), "three on gas");
        assert_eq!(g.tp_gas_crew().len(), 3);
        // One leaving the refinery (another order for a moment) or out of sight: still three; one
        // seen mining minerals: two, and another is sent.
        let one = *first.iter().next().unwrap();
        let mut leaving = u.clone();
        leaving.iter_mut().filter(|x| x.id == one).for_each(|x| x.order = 0);
        assert!(crew(&g.act(&frame(3_248, leaving, 0, 30))).is_empty(), "leaving the refinery");
        let away: Vec<UnitState> = u.iter().filter(|x| x.id != one).cloned().collect();
        assert!(crew(&g.act(&frame(3_256, away, 0, 30))).is_empty(), "out of sight");
        assert_eq!(g.tp_gas_crew().len(), 3);
        let mining = uranium_env::types::order_id("MiningMinerals").unwrap();
        u.iter_mut().filter(|x| x.id == one).for_each(|x| x.order = mining);
        let sent = crew(&g.act(&frame(3_300, u.clone(), 0, 30)));
        assert_eq!((g.tp_gas_crew().len(), sent.len()), (3, 1), "mining minerals: replaced");
    }

    /// 022-R2 (AMB-022-6): the lobby enemy race picks the teacher — Zerg the 021-R8 teacher,
    /// Terran, Protoss and Random the 022 one.
    #[test]
    fn lobby_race_picks_the_teacher_022_r2() {
        for (race, tp) in [(Race::Zerg, false), (Race::Terran, true), (Race::Protoss, true), (Race::Random, true), (Race::Unknown, true)] {
            let b = crate::for_game_hints(&info(race), T0Build::TerranTp, hints());
            assert_eq!((b.tp_stats().is_some(), b.react_stats().is_some()), (tp, !tp), "{race:?}");
        }
        assert_eq!(T0Build::TerranTp.for_lobby(&info(Race::Zerg)), T0Build::TerranReact);
        assert_eq!(T0Build::TerranReact.for_lobby(&info(Race::Protoss)), T0Build::TerranReact);
        // A lobby Random follows the race it shows first (the Terran build until then).
        let mut b = teacher_vs(Race::Random);
        assert_eq!(b.tp.as_ref().unwrap().matchup(), Matchup::Terran);
        let mut u = base(12);
        u.push(enemy(80, PROBE, depot_center((60, 60))));
        b.act(&frame(1_000, u, 0, 24));
        assert_eq!(b.tp.as_ref().unwrap().matchup(), Matchup::Protoss);
    }

    /// 022-R1 / 021-R3: `t0-terran-tp`, plan name `terran_tp`, Terran only, reads the map hints.
    #[test]
    fn build_id_022_r1() {
        let t = T0Build::TerranTp;
        assert_eq!((t.name(), t.id_tag(), t.race(), t.needs_map_hints()), ("terran_tp", Some("tp"), Some(Race::Terran), true));
        assert!(Kit::for_build(Race::Terran, t).tp.is_some() && Kit::for_build(Race::Terran, t).react.is_some());
        assert!(Kit::for_build(Race::Zerg, t).tp.is_none());
        assert!(Kit::terran_react().tp.is_none());
    }

    /// Scenes covering every kind of command the teacher gives: opening, gas, factory, machine
    /// shop, research, tanks, defence (bunker, load, hold, pull, repair), siege, unsiege, attack.
    fn scenes() -> Vec<(RawFrame, Act)> {
        let mut out = Vec::new();
        let mut run = |b: &mut Brain, f: RawFrame| {
            let act = b.act(&f);
            out.push((f, act));
        };
        let mut b = teacher();
        run(&mut b, dark(frame(1, cc_only(4), 50, 8)));
        run(&mut b, dark(frame(508, cc_only(9), 200, 18)));
        run(&mut b, dark(frame(516, cc_only(9), 100, 18)));
        let mut u = cc_only(10);
        u.push(unit(6, DEPOT, (home().0, home().1 - 160)));
        run(&mut teacher(), dark(frame(1_500, u, 200, 22)));
        let mut u = cc_only(12);
        u.push(unit(6, DEPOT, (home().0, home().1 - 160)));
        u.push(building(2, BARRACKS, (home().0 + 200, home().1 + 160)));
        u.push(geyser());
        run(&mut teacher(), dark(frame(2_000, u, 300, 24)));
        // R: the bunker, the second barracks, a marine.
        let mut u = base(12);
        u.extend((0..2).map(|i| enemy(90 + i, ZEALOT, depot_center((60, 60 + i as u16)))));
        run(&mut teacher(), frame(3_000, u, 400, 30));
        let mut u = base(14);
        u.push(unit(5, REFINERY, GEYSER_AT));
        run(&mut teacher(), gas(frame(2_400, u.clone(), 300, 30), 100));
        u.push(factory());
        run(&mut teacher(), gas(frame(3_000, u.clone(), 100, 30), 100));
        u.pop();
        u.push(UnitState { has_addon: true, ..factory() });
        u.push(shop());
        run(&mut teacher(), gas(frame(3_500, u.clone(), 200, 30), 200));
        run(&mut teacher(), sieging(gas(frame(3_600, u.clone(), 300, 30), 200), false, true));
        // Defence: a damaged bunker, marines, zealots at the bunker, a tank by them.
        let mut d = teacher();
        let bunker = (home().0 + 5 * 32, home().1 + 64);
        let mut u = base(20);
        u.push(UnitState { hp: 300, ..unit(60, BUNKER, bunker) });
        u.extend((0..6).map(|i| unit(200 + i, MARINE, (home().0 + 40, home().1 + 100 + 8 * i as u16))));
        u.push(unit(300, SIEGE_TANK, (home().0 + 2 * 32, home().1 + 3 * 32)));
        u.push(enemy(90, ZEALOT, (bunker.0, bunker.1 + 64)));
        u.push(enemy(91, ZEALOT, (bunker.0 + 20, bunker.1 + 64)));
        run(&mut d, sieging(frame(4_000, u, 0, 60), true, false));
        // Marines that do not load hold beside the bunker (3 against 2 zealots).
        let mut h = teacher();
        let mut u = base(12);
        u.push(UnitState { hp: 350, ..unit(60, BUNKER, bunker) });
        let spot = toward(bunker, home(), 2.0);
        u.extend((0..7).map(|i| unit(200 + i, MARINE, (spot.0, spot.1 + i as u16))));
        u.push(enemy(90, ZEALOT, (home().0 + 8 * 32, home().1 - 3 * 32)));
        u.push(enemy(91, ZEALOT, (home().0 + 8 * 32, home().1 - 3 * 32 + 20)));
        u.push(enemy(92, ZEALOT, (home().0 + 8 * 32, home().1 - 3 * 32 + 40)));
        u.push(enemy(93, ZEALOT, (home().0 + 8 * 32, home().1 - 3 * 32 + 60)));
        u.iter_mut().filter(|x| x.unit_type == MARINE).take(4).for_each(|x| x.loaded_in = Some(60));
        run(&mut h, frame(4_000, u, 0, 60));
        // Unsiege after 10 s, then the attack.
        let mut w = teacher();
        let mut u = base(20);
        u.extend((0..32).map(|i| unit(200 + i, MARINE, (home().0 + 100 + 4 * (i % 8) as u16, home().1 + 100 + 4 * (i / 8) as u16))));
        u.extend((0..3).map(|i| unit(300 + i, SIEGE_TANK, (home().0 + 80, home().1 + 120 + 8 * i as u16))));
        u.push(unit(303, SIEGE_TANK_SIEGED, (home().0 + 80, home().1 + 160)));
        run(&mut w, sieging(frame(9_000, u.clone(), 0, 120), true, false));
        run(&mut w, sieging(frame(9_240, u, 0, 120), true, false));
        out
    }

    /// 022-AC1: every command maps to one 034 action (through the replay commands BWAPI would
    /// send, `uranium-replay` relabel conversion and labeler, as in 041-R22), and the scenes use
    /// the 034 actions of spec 022 "모델 표현".
    #[test]
    fn commands_are_034_actions_022_ac1() {
        let lab = uranium_replay::label::Labeler::new().unwrap();
        let table = uranium_env::actions::ActionTable::load().unwrap();
        let mut names = BTreeSet::new();
        for (f, act) in scenes() {
            let units: HashMap<u32, uranium_replay::label::Own> = uranium_replay::relabel::own_units(&f, 0);
            for c in &act.cmds {
                let cmds = uranium_replay::relabel::command_cmds(c, &f, 0, 0);
                assert!(!cmds.is_empty() && cmds.len().is_multiple_of(2), "{c:?}");
                let (labels, _) = lab.label(&cmds, f.frame + 32, |_| &units);
                let acts: Vec<_> = labels.iter().filter(|l| l.action != 0).collect();
                assert_eq!(acts.len(), cmds.len() / 2, "{c:?} {labels:?}");
                for l in acts {
                    assert_ne!(l.action, uranium_env::features::label::UNKNOWN_ACTION, "{c:?}");
                    assert!(l.selected.len() <= 12);
                    names.insert(table.actions[l.action as usize].name.clone());
                }
            }
        }
        for n in [
            "Build_SupplyDepot", "Build_Barracks", "Build_Refinery", "Build_Factory", "Build_Bunker", "BuildAddon_MachineShop", "Research_TankSiege", "Train_SCV",
            "Train_Marine", "Train_SiegeTank", "SiegeMode", "TankMode", "Move_Pos", "Smart_Unit", "Hold", "Attack_Pos", "Attack_Unit", "Repair_Unit", "Gather_Unit",
        ] {
            assert!(names.contains(n), "{n} missing: {names:?}");
        }
    }

    /// 022-R1: the same observations give the same commands.
    #[test]
    fn same_observations_same_commands_022_r1() {
        let (a, b) = (scenes(), scenes());
        assert_eq!(a.iter().map(|(_, x)| x.cmds.clone()).collect::<Vec<_>>(), b.iter().map(|(_, x)| x.cmds.clone()).collect::<Vec<_>>());
    }
}
