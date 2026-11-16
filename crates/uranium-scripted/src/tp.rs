use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;
use uranium_env::scout::{is_army_type, is_ground_army_type};
use uranium_proto::*;

use crate::react::{Branch, FPS, bunker_center, depot_center, toward, within};
use crate::types::*;
use crate::brain::Brain;
use crate::util::{centroid, dist2, done, is_enemy};

const SCV_HP: u32 = 60;
const OUT_TILES: i64 = crate::brain::DEFEND_RADIUS / 32;

#[derive(Clone, Debug, PartialEq)]
pub struct TpParams {
    pub refinery_supply: u32,
    pub gas_workers: usize,
    pub marines_before_factory: usize,
    pub factory_gas: i32,
    pub scout_stay_until: u32,
    pub scout_hp_pct: u32,
    pub scout_circle: i64,
    pub prod_min: usize,
    pub prod_before: u32,
    pub zealots_min: usize,
    pub marines_min: usize,
    pub units_before: u32,
    pub proxy_from: u32,
    pub proxy_before: u32,
    pub proxy_look: i64,
    pub proxy_radius: i64,
    pub army_radius: i64,
    pub army_min: usize,
    pub main_radius: i64,
    pub judge_until: u32,
    pub expand_minerals: i32,
    pub expand_by: u32,
    pub expand_tanks: usize,
    pub barracks_p: usize,
    pub barracks_t: usize,
    pub scv_per_cc: usize,
    pub scv_max: usize,
    pub barracks_d: usize,
    pub quiet: u32,
    pub macro_marines: usize,
    pub macro_tanks: usize,
    pub macro_tanks_any: usize,
    pub tank_wait: f64,
    pub siege_range: i64,
    pub unsiege_range: i64,
    pub unsiege_quiet: u32,
    pub wave_tanks: usize,
    pub wave_supply: u32,
    pub late_wave_from: u32,
    pub late_wave_supply: u32,
    pub group_gap: i64,
    pub retarget: u32,
    pub repairers: usize,
    pub pull_max: usize,
    pub marine_ratio: usize,
    pub expand_tanks_p: usize,
    pub idle_grace: u32,
    pub marines_hold_p: bool,
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
            repairers: 2,
            pull_max: 6,
            marine_ratio: 2,
            expand_tanks_p: 3,
            marines_hold_p: true,
            idle_grace: 16,
        }
    }
}

impl TpParams {
    pub fn react_params(&self) -> crate::ReactParams {
        crate::ReactParams { repairers: self.repairers, pull_max: self.pull_max, marine_ratio: self.marine_ratio, ..Default::default() }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TpRule {
    Gates,
    Zealots,
    ProxyGate,
    Rax,
    Marines,
    ProxyRax,
    Army,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct TpStats {
    pub main_frame: Option<u32>,
    pub rush_frame: Option<u32>,
    pub rush_rule: Option<TpRule>,
    pub defences: u32,
    pub macro_frame: Option<u32>,
    pub refinery_frame: Option<u32>,
    pub factory_frame: Option<u32>,
    pub shop_frame: Option<u32>,
    pub siege_frame: Option<u32>,
    pub tank_frame: Option<u32>,
    pub expand_frame: Option<u32>,
    pub wave_frame: Option<u32>,
    pub branch: Branch,
    pub contact_frame: Option<u32>,
    pub bunker_frame: Option<u32>,
    pub rule_frames: BTreeMap<TpRule, u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Matchup {
    Protoss,
    Terran,
}

pub(crate) struct TpState {
    pub p: TpParams,
    lobby: Race,
    seen: Option<Race>,
    pub gas_crew: Vec<UnitId>,
    idle_since: BTreeMap<UnitId, u32>,
    failed_sites: Vec<(Tile, u16, u32)>,
    gas_seen: BTreeMap<UnitId, u32>,
    refinery_since: Option<(UnitId, u32)>,
    last_gas: Option<u32>,
    last_shop: Option<u32>,
    last_research: Option<u32>,
    last_tanks: Option<u32>,
    prod_seen: BTreeSet<(Tile, u16)>,
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
            idle_since: BTreeMap::new(),
            failed_sites: Vec::new(),
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
        crate::react::ReactState::due_on(true, last, frame, every)
    }
}

pub(crate) fn pick_key(rings: bool, a: Pos, b: Pos) -> i64 {
    if rings { (dist2(a, b) as f64).sqrt() as i64 / 128 } else { dist2(a, b) }
}

pub(crate) fn pick_key_id(by_id: bool, a: Pos, b: Pos) -> i64 {
    if by_id { 0 } else { dist2(a, b) }
}

pub(crate) fn snap(p: Pos) -> Pos {
    (p.0 / 128 * 128 + 64, p.1 / 128 * 128 + 64)
}

fn race_of(ty: u16) -> Option<Race> {
    match uranium_env::types::TypeTables::get().unit(ty)?.race.as_str() {
        "Protoss" => Some(Race::Protoss),
        "Terran" => Some(Race::Terran),
        "Zerg" => Some(Race::Zerg),
        _ => None,
    }
}

pub struct TpContext<'a> {
    pub depots: &'a [Pos],
    pub starts: &'a [Tile],
    pub scout: Option<Pos>,
    pub prod_seen: &'a BTreeSet<(Tile, u16)>,
}

pub fn tp_rules(p: &TpParams, f: &RawFrame, me: u8, cx: &TpContext<'_>) -> Vec<TpRule> {
    let mut out = Vec::new();
    if f.frame >= p.judge_until {
        return out;
    }
    let enemies: Vec<&UnitState> = f.units.iter().filter(|u| is_enemy(u, me)).collect();
    let count = |ty: u16| enemies.iter().filter(|u| u.unit_type == ty).count();
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
    fn stale_cells(&self) -> Vec<Pos> {
        let mut cells: Vec<(u32, u16, u16, Pos)> = Vec::new();
        for cy in 0..self.cells_h {
            for cx in 0..self.cells_w {
                let p = ((cx * crate::brain::CELL + crate::brain::CELL / 2) * 32, (cy * crate::brain::CELL + crate::brain::CELL / 2) * 32);
                if self.walkable.get(p.0 / 8, p.1 / 8) {
                    cells.push((self.cell_seen[(cy * self.cells_w + cx) as usize], cy, cx, p));
                }
            }
        }
        cells.sort_unstable();
        cells.into_iter().map(|(_, _, _, p)| p).collect()
    }

    pub(crate) fn tp_gas_crew(&self) -> Vec<UnitId> {
        self.tp.as_ref().map(|t| t.gas_crew.clone()).unwrap_or_default()
    }

    fn tanks_done(mine: &[&UnitState]) -> usize {
        mine.iter().filter(|u| is_tank(u.unit_type) && u.has(flags::COMPLETED)).count()
    }

    fn natural_done(&self, mine: &[&UnitState]) -> bool {
        let Some(n) = self.react.as_ref().and_then(|r| r.natural) else { return false };
        mine.iter().any(|u| is_resource_depot(u.unit_type) && building_tile(u) == n && done(u))
    }

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
        let matchup = t.matchup();
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
            r.stats.main_frame = Some(f.frame);
            t.stats.main_frame = Some(f.frame);
        }
        if contact {
            r.last_contact = f.frame;
            t.stats.contact_frame.get_or_insert(f.frame);
        }
        match r.branch {
            Branch::Macro => {
                if let Some(&rule) = rules.first().filter(|_| tanks < p.macro_tanks_any) {
                    r.branch = Branch::Defence;
                    r.last_contact = f.frame;
                    t.stats.defences += 1;
                    if t.stats.rush_frame.is_none() {
                        t.stats.rush_frame = Some(f.frame);
                        t.stats.rush_rule = Some(rule);
                    }
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
        let outside = mine.iter().filter(|u| u.unit_type == MARINE && u.has(flags::COMPLETED) && u.loaded_in.is_none()).count();
        let close = enemies.iter().filter(|u| is_ground_army_type(u.unit_type) && depots.iter().any(|d| within(u.pos, *d, r.p.marine_radius))).count();
        let hold = p.marines_hold_p && matchup == Matchup::Protoss;
        r.marines_attack = r.branch == Branch::Defence && !hold && close > 0 && outside >= r.p.marine_ratio * close;
        if r.marines_attack {
            r.holding.clear();
        }
        t.stats.branch = r.branch;
    }

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

    pub(crate) fn circle_point(&self, c: Pos, pos: Pos, radius: i64) -> Pos {
        let r = radius * 32;
        let d = (r as f64 * std::f64::consts::FRAC_1_SQRT_2).round() as i64;
        let offsets = [(r, 0), (d, d), (0, r), (-d, d), (-r, 0), (-d, -d), (0, -r), (d, -d)];
        let a = (pos.1 as f64 - c.1 as f64).atan2(pos.0 as f64 - c.0 as f64);
        let k = (a / std::f64::consts::FRAC_PI_4).round() as i64;
        for step in [2i64, 1, 3, 4, 5, 6, 7] {
            let (ox, oy) = offsets[(k + step).rem_euclid(8) as usize];
            let (x, y) = (c.0 as i64 + ox, c.1 as i64 + oy);
            if x < 0 || y < 0 {
                continue;
            }
            let (x, y) = (x as u16, y as u16);
            if (x / 8) < self.walkable.width && (y / 8) < self.walkable.height && self.walkable.get(x / 8, y / 8) {
                return (x, y);
            }
        }
        c
    }

    pub(crate) fn tp_idle(&mut self, f: &RawFrame, mine: &[&UnitState]) {
        let Some(t) = self.tp.as_mut() else { return };
        let idle: Vec<UnitId> = mine.iter().filter(|u| u.unit_type == SCV && u.has(flags::COMPLETED) && u.has(flags::IDLE) && u.loaded_in.is_none()).map(|u| u.id).collect();
        t.idle_since.retain(|id, _| idle.contains(id));
        for id in idle {
            t.idle_since.entry(id).or_insert(f.frame);
        }
        let grace = t.p.idle_grace;
        let since = t.idle_since.clone();
        t.failed_sites.retain(|(_, _, until)| f.frame < *until);
        let mut failed = Vec::new();
        self.pending.retain(|q| {
            let idle = since.get(&q.worker).is_some_and(|s| f.frame >= s + grace);
            if idle {
                failed.push((q.tile, q.ty, f.frame + crate::brain::PENDING_TIMEOUT));
            }
            !idle
        });
        self.tp.as_mut().expect("checked above").failed_sites.extend(failed);
    }

    pub(crate) fn tp_failed_sites(&self) -> Vec<(Tile, u16)> {
        self.tp.as_ref().map(|t| t.failed_sites.iter().map(|(tile, ty, _)| (*tile, *ty)).collect()).unwrap_or_default()
    }

    pub(crate) fn tp_idle_long(&self, id: UnitId, frame: u32) -> bool {
        self.tp.as_ref().is_none_or(|t| t.idle_since.get(&id).is_some_and(|s| frame >= s + t.p.idle_grace))
    }

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
        free.sort_by_key(|u| (pick_key(true, u.pos, r.pos), u.id));
        let ids: Vec<UnitId> = free.iter().take(want - crew.len()).map(|u| u.id).collect();
        if ids.is_empty() {
            return;
        }
        cmds.push(Command::Unit { units: ids.clone(), order: CmdOrder::Gather(r.id), queued: false });
        let t = self.tp.as_mut().expect("checked above");
        t.gas_crew.extend(ids);
        t.last_gas = Some(f.frame);
    }

    fn tp_order_build(&mut self, ty: u16, f: &RawFrame, mine: &[&UnitState], base: Pos, busy: &[UnitId], cmds: &mut Vec<Command>) -> Option<bool> {
        let reserved: Vec<(Tile, u16)> = self.pending.iter().map(|q| (q.tile, q.ty)).chain(self.react_reserved_sites()).collect();
        let tile = self.place(ty, f, mine, base, &reserved)?;
        Some(self.order_build_at(ty, tile, f, mine, busy, cmds))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn tp_build(&mut self, f: &RawFrame, mine: &[&UnitState], base: Pos, busy: &[UnitId], cmds: &mut Vec<Command>, minerals: &mut i32, gas: &mut i32) -> (i32, i32) {
        let Some(t) = self.tp.as_ref() else { return (0, 0) };
        let (p, last_shop, last_research) = (t.p.clone(), t.last_shop, t.last_research);
        let matchup = t.matchup();
        let (mut hold_all, mut hold_army) = (0, 0);
        let has = |s: &Self, ty: u16| mine.iter().any(|u| u.unit_type == ty) || s.pending.iter().any(|q| q.ty == ty);
        let rax_done = mine.iter().any(|u| u.unit_type == BARRACKS && done(u));
        let supply = u32::from(f.self_state.supply_used);
        fn stats(s: &mut Brain) -> &mut TpStats {
            &mut s.tp.as_mut().expect("the 022 teacher").stats
        }

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
        let r = self.react.as_ref().expect("the 022 teacher reuses the 021-R8 state");
        let (branch, natural, last_contact) = (r.branch, r.natural, r.last_contact);
        if let Some(n) = natural.filter(|_| !self.natural_started(mine)) {
            let tanks = Self::tanks_done(mine);
            let ready = match (branch, matchup) {
                (Branch::Macro, Matchup::Protoss) => has(self, FACTORY) && tanks >= p.expand_tanks_p,
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

    pub(crate) fn tp_marine_cap(&self, mine: &[&UnitState]) -> Option<usize> {
        let t = self.tp.as_ref()?;
        let factory = mine.iter().any(|u| u.unit_type == FACTORY) || self.pending.iter().any(|q| q.ty == FACTORY);
        Some(if self.react_branch() == Some(Branch::Macro) && !factory { t.p.marines_before_factory } else { usize::MAX })
    }

    pub(crate) fn tp_max_workers(&self, mine: &[&UnitState]) -> Option<usize> {
        let t = self.tp.as_ref()?;
        let ccs = mine.iter().filter(|u| is_resource_depot(u.unit_type) && done(u)).count().max(1);
        Some((t.p.scv_per_cc * ccs).min(t.p.scv_max))
    }

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

    pub(crate) fn tp_army(&mut self, f: &RawFrame, mine: &[&UnitState], kept: &[UnitId], cmds: &mut Vec<Command>) {
        let Some(t) = self.tp.as_ref() else { return };
        let p = t.p.clone();
        let (sieging, unsieging) = (t.order_sieging, t.order_unsieging);
        let me = self.me;
        let researched = f.self_state.tech & (1u64 << TECH_TANK_SIEGE) != 0;
        let enemy_ground: Vec<Pos> = f.units.iter().filter(|u| is_enemy(u, me) && is_ground_army_type(u.unit_type)).map(|u| u.pos).collect();
        let near = |at: Pos, tiles: i64| enemy_ground.iter().any(|e| within(*e, at, tiles));
        let tanks: Vec<&UnitState> = mine.iter().copied().filter(|u| is_tank(u.unit_type) && u.has(flags::COMPLETED) && u.loaded_in.is_none()).collect();
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

        if let Some(s) = spot.filter(|_| crate::react::ReactState::due_on(true, None, f.frame, p.retarget)) {
            let home: Vec<UnitId> = tanks
                .iter()
                .filter(|u| u.unit_type == SIEGE_TANK && !siege.contains(&u.id) && Some(u.order) != sieging && !in_group(u) && !within(u.pos, s, 4))
                .map(|u| u.id)
                .collect();
            for chunk in home.chunks(12) {
                cmds.push(Command::Unit { units: chunk.to_vec(), order: CmdOrder::Move(Target::Pos(s)), queued: false });
            }
        }

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
            } else if let Some(target) = self.attack_target(f, lead).map(snap) {
                let lead_d = dist2(lead, target);
                let gap = p.group_gap * 32;
                let front = snap(toward(lead, target, (p.group_gap - 2) as f64));
                let mut units: Vec<&UnitState> = group_tanks.iter().copied().filter(|u| mobile(u)).chain(group_marines.iter().copied()).collect();
                units.sort_by_key(|u| u.id);
                for chunk in units.chunks(12) {
                    let (wait, go): (Vec<&UnitState>, Vec<&UnitState>) = chunk.iter().partition(|u| dist2(u.pos, lead) > gap * gap && dist2(u.pos, target) < lead_d);
                    for (ids, at) in [(go, target), (wait, front)] {
                        if !ids.is_empty() {
                            cmds.push(Command::Unit { units: ids.iter().map(|u| u.id).collect(), order: CmdOrder::Attack(Target::Pos(at)), queued: false });
                        }
                    }
                }
                acted |= !units.is_empty();
            }
        }
        if acted {
            self.tp.as_mut().expect("checked above").last_tanks = Some(f.frame);
        }
    }
}

#[cfg(feature = "testkit")]
pub mod testkit;

#[cfg(test)]
mod tests;
