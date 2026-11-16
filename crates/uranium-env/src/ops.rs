use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use uranium_proto::{GameEvent, GameInfo, Pos, RawFrame, Tile, UnitId, UnitState, flags};

use crate::features::MapPrecompute;
use crate::scout::{FPS, is_army_type, is_ground_army_type};
use crate::types::{NO_TYPE, TypeTables, order_id};

pub const WINDOW_END: u32 = 14_400;
pub const RUSH_BEFORE: u32 = 7_200;
pub const RUSH_WINDOW: u32 = 2_160;
pub const INC_AT: [u32; 4] = [5_760, 8_640, 11_520, 14_400];
const MINUTE: u32 = 1_440;
const TILE: i64 = 32;
const BASE_RADIUS: i64 = 12 * TILE;
const RUSH_RADIUS: i64 = 12 * TILE;
const MLINE_WIDTH: f64 = 32.0;
const SPOT_RADIUS: i64 = 3 * TILE;
const SAT_PER_FIELD: i64 = 3;
const PULL_MIN: u32 = 3;
const SUPPLY_CAP: u16 = 400;
const TRIP: u32 = 8;
const FIELD_MATCH: i64 = 32;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct OpsMetrics {
    pub end_s: f64,
    pub mined: u32,
    pub gas: u32,
    pub inc_4m: Option<f64>,
    pub inc_6m: Option<f64>,
    pub inc_8m: Option<f64>,
    pub inc_10m: Option<f64>,
    pub inc_scv_4m: Option<f64>,
    pub inc_scv_6m: Option<f64>,
    pub inc_scv_8m: Option<f64>,
    pub inc_scv_10m: Option<f64>,
    pub mline_bldg_n: u32,
    pub mline_bldg_s: f64,
    pub cc_lift_s: f64,
    pub stranded_scv_s: f64,
    pub idle_scv_s: f64,
    pub oversat_scv_s: f64,
    pub sblock_s: f64,
    pub sblock_n: u32,
    pub sblock_first_s: Option<f64>,
    pub float_min_rs: f64,
    pub float_gas_rs: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct FieldCheck {
    pub depleted: u32,
    pub fields: u32,
    pub unknown_fields: u32,
    pub scvs: u32,
    pub scvs_max: u32,
    pub carrying: u32,
    pub lost_cargo: u32,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct RushCombat {
    pub in_s: f64,
    pub enemy_max: u32,
    pub pull_s: Option<f64>,
    pub pull_max: u32,
    pub scv_move_max: u32,
    pub scv_lost: u32,
    pub kills: u32,
    pub bunker_marines_max: u32,
    pub bunker_load_s: Option<f64>,
    pub bunker_empty_s: f64,
    pub focus_hhi: Option<f64>,
    pub focus_obs: u32,
}

#[derive(Debug, Clone)]
struct Field {
    pos: Pos,
    amount: u32,
    id: Option<UnitId>,
    init: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Harvest {
    Mineral(Option<usize>),
    Gas,
}

#[derive(Debug, Clone, Copy)]
struct Ids {
    scv: u16,
    cc: u16,
    barracks: u16,
    factory: u16,
    starport: u16,
    marine: u16,
    bunker: u16,
    refinery: u16,
    mineral: u16,
    cheapest: [(u16, i32, i32, i32); 4],
}

#[derive(Debug, Clone)]
struct Orders {
    to_minerals: Vec<u16>,
    return_minerals: u16,
    to_gas: Vec<u16>,
    harvest: Vec<u16>,
    attack: Vec<u16>,
    attack_unit: u16,
    move_: u16,
    place_building: u16,
}

fn type_id(name: &str) -> u16 {
    TypeTables::get().unit_by_name(name).map_or(u16::MAX, |u| u.id as u16)
}

fn order(name: &str) -> u16 {
    order_id(name).unwrap_or(u16::MAX)
}

fn is_mineral_type(id: u16) -> bool {
    TypeTables::get().unit(id).is_some_and(|t| t.name.starts_with("Resource_Mineral_Field"))
}

fn is_gas_type(id: u16) -> bool {
    TypeTables::get().unit(id).is_some_and(|t| matches!(t.name.as_str(), "Terran_Refinery" | "Protoss_Assimilator" | "Zerg_Extractor" | "Resource_Vespene_Geyser"))
}

fn d2(a: Pos, b: Pos) -> i64 {
    let (dx, dy) = (a.0 as i64 - b.0 as i64, a.1 as i64 - b.1 as i64);
    dx * dx + dy * dy
}

fn within(a: Pos, b: Pos, r: i64) -> bool {
    d2(a, b) <= r * r
}

fn footprint(ty: u16, pos: Pos) -> (i32, i32, i32, i32) {
    let t = TypeTables::get().unit(ty);
    let (w, h) = t.map_or((1, 1), |t| (t.tile_width.max(1), t.tile_height.max(1)));
    let x0 = (pos.0 as i32 - w * 16).div_euclid(32);
    let y0 = (pos.1 as i32 - h * 16).div_euclid(32);
    (x0, y0, w, h)
}

fn overlap(lo: u32, hi: u32, a: u32, b: u32) -> u64 {
    (hi.min(b) as i64 - lo.max(a) as i64).max(0) as u64
}

fn secs(frames: u64) -> f64 {
    frames as f64 / FPS
}

fn seg_dist(p: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    let (vx, vy) = (b.0 - a.0, b.1 - a.1);
    let len2 = vx * vx + vy * vy;
    let t = if len2 > 0.0 { (((p.0 - a.0) * vx + (p.1 - a.1) * vy) / len2).clamp(0.0, 1.0) } else { 0.0 };
    let (qx, qy) = (a.0 + t * vx, a.1 + t * vy);
    ((p.0 - qx).powi(2) + (p.1 - qy).powi(2)).sqrt()
}

#[derive(Debug, Clone, Default)]
struct Rush {
    entry: u32,
    enemy_max: u32,
    pull_frame: Option<u32>,
    pull_max: u32,
    move_max: u32,
    scv_lost: u32,
    kills: u32,
    marines_max: u32,
    load_frame: Option<u32>,
    empty: u64,
    hhi_sum: f64,
    hhi_n: u32,
}

impl Rush {
    fn end(&self) -> u32 {
        self.entry + RUSH_WINDOW
    }
}

#[derive(Debug, Clone)]
pub struct OpsTracker {
    me: u8,
    enemies: Vec<u8>,
    start: Option<Pos>,
    spots: Vec<Pos>,
    fields: Vec<Field>,
    field_of: BTreeMap<UnitId, usize>,
    carry: BTreeMap<UnitId, bool>,
    harvest: BTreeMap<UnitId, Harvest>,
    last_seen: BTreeMap<UnitId, (u8, u16)>,
    mline_seen: BTreeSet<UnitId>,
    mined_fields: BTreeSet<usize>,
    last_scvs: (u32, u32),
    scvs_max: u32,
    lost_cargo: u32,
    prev: u32,
    last: Option<u32>,
    mined: u32,
    gas: u32,
    inc_mined: [u32; 4],
    inc_scv: [u64; 4],
    mline_bldg_n: u32,
    mline: u64,
    lift: u64,
    stranded: u64,
    idle: u64,
    oversat: u64,
    sblock: u64,
    sblock_n: u32,
    sblock_first: Option<u32>,
    blocked: bool,
    float_min: u64,
    float_gas: u64,
    rush: Option<Rush>,
    ids: Ids,
    orders: Orders,
}

impl OpsTracker {
    pub fn new(me: u8, enemies: Vec<u8>, own_start: Option<Tile>, pre: Option<&MapPrecompute>) -> Self {
        let t = TypeTables::get();
        let cc = type_id("Terran_Command_Center");
        let (cw, ch) = t.unit(cc).map_or((4, 3), |u| (u.tile_width, u.tile_height));
        let spots = pre.map_or_else(Vec::new, |p| p.base_locs.iter().map(|&(x, y)| ((x as i32 * 32 + cw * 16) as u16, (y as i32 * 32 + ch * 16) as u16)).collect());
        let fields = pre.map_or_else(Vec::new, |p| p.resources.iter().filter(|r| is_mineral_type(r.unit_type)).map(|r| Field { pos: r.pos, amount: r.amount, id: None, init: Some(r.amount) }).collect());
        let cost = |producer: &str, unit: &str| {
            let u = t.unit_by_name(unit).expect("Terran unit in the type table");
            (type_id(producer), u.minerals, u.gas, u.supply_required)
        };
        let ids = Ids {
            scv: type_id("Terran_SCV"),
            cc,
            barracks: type_id("Terran_Barracks"),
            factory: type_id("Terran_Factory"),
            starport: type_id("Terran_Starport"),
            marine: type_id("Terran_Marine"),
            bunker: type_id("Terran_Bunker"),
            refinery: type_id("Terran_Refinery"),
            mineral: type_id("Resource_Mineral_Field"),
            cheapest: [
                cost("Terran_Command_Center", "Terran_SCV"),
                cost("Terran_Barracks", "Terran_Marine"),
                cost("Terran_Factory", "Terran_Vulture"),
                cost("Terran_Starport", "Terran_Wraith"),
            ],
        };
        let orders = Orders {
            to_minerals: ["MoveToMinerals", "WaitForMinerals", "MiningMinerals"].map(order).to_vec(),
            return_minerals: order("ReturnMinerals"),
            to_gas: ["MoveToGas", "WaitForGas", "HarvestGas"].map(order).to_vec(),
            harvest: ["Harvest1", "Harvest2", "Harvest3", "Harvest4"].map(order).to_vec(),
            attack: ["Attack1", "Attack2", "AttackUnit", "AttackFixedRange", "AttackMove"].map(order).to_vec(),
            attack_unit: order("AttackUnit"),
            move_: order("Move"),
            place_building: order("PlaceBuilding"),
        };
        Self {
            me,
            enemies,
            start: own_start.map(|(x, y)| (x * 32 + 16, y * 32 + 16)),
            spots,
            fields,
            field_of: BTreeMap::new(),
            carry: BTreeMap::new(),
            harvest: BTreeMap::new(),
            last_seen: BTreeMap::new(),
            mline_seen: BTreeSet::new(),
            mined_fields: BTreeSet::new(),
            last_scvs: (0, 0),
            scvs_max: 0,
            lost_cargo: 0,
            prev: 0,
            last: None,
            mined: 0,
            gas: 0,
            inc_mined: [0; 4],
            inc_scv: [0; 4],
            mline_bldg_n: 0,
            mline: 0,
            lift: 0,
            stranded: 0,
            idle: 0,
            oversat: 0,
            sblock: 0,
            sblock_n: 0,
            sblock_first: None,
            blocked: false,
            float_min: 0,
            float_gas: 0,
            rush: None,
            ids,
            orders,
        }
    }

    pub fn for_game(info: &GameInfo, pre: Option<&MapPrecompute>) -> Self {
        let me = info.players.iter().find(|p| p.is_self);
        let self_player = me.map_or(info.self_player, |p| p.id);
        let own = me.and_then(|p| p.start_tile);
        let enemies = info.players.iter().filter(|p| p.is_enemy).map(|p| p.id).collect();
        Self::new(self_player, enemies, own, pre)
    }

    pub fn play_until(&self, until: u32) -> u32 {
        self.rush.as_ref().map_or(until, |r| until.max(r.end()))
    }

    fn field_for(&mut self, id: UnitId, pos: Pos) -> usize {
        if let Some(&i) = self.field_of.get(&id) {
            return i;
        }
        let near = (0..self.fields.len()).filter(|&i| self.fields[i].id.is_none() && within(self.fields[i].pos, pos, FIELD_MATCH)).min_by_key(|&i| d2(self.fields[i].pos, pos));
        let i = near.unwrap_or_else(|| {
            self.fields.push(Field { pos, amount: 0, id: None, init: None });
            self.fields.len() - 1
        });
        self.fields[i].id = Some(id);
        self.field_of.insert(id, i);
        i
    }

    fn is_enemy(&self, u: &UnitState) -> bool {
        self.enemies.contains(&u.owner)
    }

    pub fn step(&mut self, f: &RawFrame) {
        let fr = f.frame;
        let (lo, hi) = (self.prev, fr.max(self.prev));
        if lo >= WINDOW_END && self.rush.as_ref().is_none_or(|r| lo >= r.end()) {
            self.prev = hi;
            self.last = Some(hi);
            return;
        }
        let repeat = self.last == Some(fr);
        let dt = overlap(lo, hi, 0, WINDOW_END);
        let in_window = fr <= WINDOW_END;
        let rush_open = |r: &Rush| fr > r.entry && fr <= r.end();
        for e in &f.events {
            if let GameEvent::UnitDestroyed(id) = e {
                if let Some(&i) = self.field_of.get(id) {
                    self.fields[i].amount = 0;
                }
                let minerals = matches!(self.harvest.get(id), Some(Harvest::Mineral(_)));
                if in_window && minerals && self.carry.get(id) == Some(&true) && self.last_seen.get(id).is_some_and(|&(o, t)| o == self.me && t == self.ids.scv) {
                    self.lost_cargo += TRIP;
                }
                if let (Some(r), Some(&(owner, ty))) = (self.rush.as_mut().filter(|r| rush_open(r)), self.last_seen.get(id)) {
                    if owner == self.me && ty == self.ids.scv {
                        r.scv_lost += 1;
                    } else if self.enemies.contains(&owner) && is_army_type(ty) {
                        r.kills += 1;
                    }
                }
                self.carry.remove(id);
                self.harvest.remove(id);
            }
        }
        for u in f.units.iter().filter(|u| is_mineral_type(u.unit_type)) {
            let i = self.field_for(u.id, u.pos);
            self.fields[i].amount = u.resources as u32;
        }
        let ids = self.ids;
        let mine: Vec<&UnitState> = f.units.iter().filter(|u| u.owner == self.me).collect();
        let bases: Vec<&UnitState> = mine.iter().copied().filter(|u| u.unit_type == ids.cc && u.has(flags::COMPLETED) && !u.has(flags::LIFTED)).collect();
        let lifted = mine.iter().filter(|u| u.unit_type == ids.cc && u.has(flags::LIFTED)).count() as u64;
        let scvs: Vec<&UnitState> = mine.iter().copied().filter(|u| u.unit_type == ids.scv && u.has(flags::COMPLETED)).collect();
        let n_scv = scvs.len() as u64;
        let mut idle = 0u64;
        let mut mining: Vec<usize> = Vec::new();
        let (mut attacking, mut moving) = (0u32, 0u32);
        let mut placing_cc = false;
        for u in &scvs {
            let carrying = u.has(flags::CARRYING);
            if self.carry.insert(u.id, carrying) == Some(true) && !carrying && in_window {
                if self.harvest.get(&u.id) == Some(&Harvest::Gas) {
                    self.gas += TRIP;
                } else {
                    self.mined += TRIP;
                    if let Some(&Harvest::Mineral(Some(i))) = self.harvest.get(&u.id) {
                        self.mined_fields.insert(i);
                    }
                    for (k, &x) in INC_AT.iter().enumerate() {
                        if fr > x - MINUTE && fr <= x {
                            self.inc_mined[k] += TRIP;
                        }
                    }
                }
            }
            let target = u.order_target.and_then(|t| f.units.iter().find(|v| v.id == t));
            let gathering = self.orders.to_minerals.contains(&u.order) || self.orders.to_gas.contains(&u.order) || self.orders.harvest.contains(&u.order);
            if gathering {
                let kind = match target {
                    Some(t) if is_mineral_type(t.unit_type) => Some(Harvest::Mineral(Some(self.field_for(t.id, t.pos)))),
                    Some(t) if is_gas_type(t.unit_type) => Some(Harvest::Gas),
                    _ if self.orders.to_gas.contains(&u.order) => Some(Harvest::Gas),
                    _ if self.orders.to_minerals.contains(&u.order) => Some(Harvest::Mineral(u.order_target.and_then(|t| self.field_of.get(&t).copied()))),
                    _ => None,
                };
                if let Some(k) = kind {
                    self.harvest.insert(u.id, k);
                }
            }
            if u.has(flags::IDLE) && !u.has(flags::LOADED) && u.loaded_in.is_none() {
                idle += 1;
            }
            if self.orders.to_minerals.contains(&u.order) || u.order == self.orders.return_minerals {
                let field = match (self.orders.to_minerals.contains(&u.order), u.order_target.and_then(|t| self.field_of.get(&t).copied())) {
                    (true, Some(i)) => Some(i),
                    _ => match self.harvest.get(&u.id) {
                        Some(Harvest::Mineral(i)) => *i,
                        _ => None,
                    },
                };
                if let Some(i) = field {
                    mining.push(i);
                }
            }
            if self.orders.attack.contains(&u.order) {
                attacking += 1;
            }
            if u.order == self.orders.move_ {
                moving += 1;
            }
            if u.order == self.orders.place_building {
                if let Some(p) = u.order_pos {
                    placing_cc |= self.spots.iter().any(|&s| within(p, s, SPOT_RADIUS) && !mine.iter().any(|c| c.unit_type == ids.cc && !c.has(flags::LIFTED) && within(c.pos, s, SPOT_RADIUS)));
                }
            }
        }
        let left: Vec<usize> = (0..self.fields.len()).filter(|&i| self.fields[i].amount > 0).collect();
        let owner_base = |pos: Pos| bases.iter().enumerate().filter(|(_, b)| within(b.pos, pos, BASE_RADIUS)).min_by_key(|(_, b)| d2(b.pos, pos)).map(|(k, _)| k);
        let stranded = mining.iter().filter(|&&i| !bases.iter().any(|b| within(b.pos, self.fields[i].pos, BASE_RADIUS))).count() as u64;
        let mut per_base = vec![(0i64, 0i64); bases.len()];
        for &i in &left {
            if let Some(k) = owner_base(self.fields[i].pos) {
                per_base[k].1 += 1;
            }
        }
        for &i in &mining {
            if let Some(k) = owner_base(self.fields[i].pos) {
                per_base[k].0 += 1;
            }
        }
        if in_window && !mine.is_empty() {
            let carrying = scvs.iter().filter(|u| u.has(flags::CARRYING) && matches!(self.harvest.get(&u.id), Some(Harvest::Mineral(_)))).count();
            self.last_scvs = (n_scv as u32, carrying as u32);
            self.scvs_max = self.scvs_max.max(n_scv as u32);
        }
        let oversat: i64 = per_base.iter().map(|&(n, k)| (n - SAT_PER_FIELD * k).max(0)).sum();
        let mut excluded: BTreeSet<(i32, i32)> = BTreeSet::new();
        let tiles = |ty: u16, pos: Pos, set: &mut BTreeSet<(i32, i32)>| {
            let (x0, y0, w, h) = footprint(ty, pos);
            for y in y0..y0 + h {
                for x in x0..x0 + w {
                    set.insert((x, y));
                }
            }
        };
        for b in &bases {
            tiles(b.unit_type, b.pos, &mut excluded);
        }
        for fld in &self.fields {
            tiles(ids.mineral, fld.pos, &mut excluded);
        }
        let mut mline_tiles: BTreeSet<(i32, i32)> = BTreeSet::new();
        for b in &bases {
            for &i in left.iter().filter(|&&i| within(b.pos, self.fields[i].pos, BASE_RADIUS)) {
                let (a, c) = ((b.pos.0 as f64, b.pos.1 as f64), (self.fields[i].pos.0 as f64, self.fields[i].pos.1 as f64));
                let (x0, x1) = (((a.0.min(c.0) - MLINE_WIDTH) / 32.0).floor() as i32, ((a.0.max(c.0) + MLINE_WIDTH) / 32.0).floor() as i32);
                let (y0, y1) = (((a.1.min(c.1) - MLINE_WIDTH) / 32.0).floor() as i32, ((a.1.max(c.1) + MLINE_WIDTH) / 32.0).floor() as i32);
                for y in y0..=y1 {
                    for x in x0..=x1 {
                        if !excluded.contains(&(x, y)) && seg_dist((x as f64 * 32.0 + 16.0, y as f64 * 32.0 + 16.0), a, c) <= MLINE_WIDTH {
                            mline_tiles.insert((x, y));
                        }
                    }
                }
            }
        }
        let types = TypeTables::get();
        let mut in_line = 0u64;
        if !mline_tiles.is_empty() {
            for u in &mine {
                let Some(t) = types.unit(u.unit_type) else { continue };
                if !t.is_building() || t.is_addon() || u.unit_type == ids.refinery || u.has(flags::LIFTED) {
                    continue;
                }
                let (x0, y0, w, h) = footprint(u.unit_type, u.pos);
                if (y0..y0 + h).any(|y| (x0..x0 + w).any(|x| mline_tiles.contains(&(x, y)))) {
                    in_line += 1;
                    if in_window && self.mline_seen.insert(u.id) {
                        self.mline_bldg_n += 1;
                    }
                }
            }
        }
        let st = &f.self_state;
        let producer = |ty: u16| ty == ids.cc || ty == ids.barracks || ty == ids.factory || ty == ids.starport;
        let blocked = st.supply_total < SUPPLY_CAP
            && mine.iter().any(|u| {
                producer(u.unit_type)
                    && u.has(flags::COMPLETED)
                    && !u.has(flags::LIFTED)
                    && !u.has(flags::TRAINING)
                    && !u.has(flags::CONSTRUCTING)
                    && u.build_queue.iter().all(|&q| q >= NO_TYPE)
                    && ids.cheapest.iter().find(|c| c.0 == u.unit_type).is_some_and(|&(_, m, g, s)| st.minerals >= m && st.gas >= g && st.supply_used as i32 + s > st.supply_total as i32)
            });
        if blocked && !self.blocked && in_window {
            self.sblock_n += 1;
        }
        if blocked && in_window && self.sblock_first.is_none() {
            self.sblock_first = Some(fr);
        }
        self.blocked = blocked;
        let p = mine.iter().filter(|u| producer(u.unit_type) && u.has(flags::COMPLETED)).count() as i64;
        let fs = mine.iter().filter(|u| (u.unit_type == ids.factory || u.unit_type == ids.starport) && u.has(flags::COMPLETED)).count() as i64;
        let float_min = if placing_cc { 0 } else { (st.minerals as i64 - (400 + 50 * p)).max(0) as u64 };
        let float_gas = (st.gas as i64 - (200 + 50 * fs)).max(0) as u64;
        self.mline += in_line * dt;
        self.lift += lifted * dt;
        self.stranded += stranded * dt;
        self.idle += idle * dt;
        self.oversat += oversat as u64 * dt;
        self.sblock += blocked as u64 * dt;
        self.float_min += float_min * dt;
        self.float_gas += float_gas * dt;
        for (k, &x) in INC_AT.iter().enumerate() {
            self.inc_scv[k] += n_scv * overlap(lo, hi, x - MINUTE, x);
        }
        let start = self.start;
        let near = |u: &&UnitState| start.is_some_and(|s| within(u.pos, s, RUSH_RADIUS));
        let enemy_in = f.units.iter().filter(|u| self.is_enemy(u) && is_ground_army_type(u.unit_type)).filter(near).count() as u32;
        if self.rush.is_none() && fr < RUSH_BEFORE && enemy_in > 0 {
            self.rush = Some(Rush { entry: fr, ..Default::default() });
        }
        let bunkers: BTreeSet<UnitId> = mine.iter().filter(|u| u.unit_type == ids.bunker).map(|u| u.id).collect();
        let marines: Vec<&UnitState> = mine.iter().copied().filter(|u| u.unit_type == ids.marine && u.has(flags::COMPLETED)).collect();
        let loaded = marines.iter().filter(|u| u.loaded_in.is_some_and(|b| bunkers.contains(&b))).count() as u32;
        let done_bunker = mine.iter().any(|u| u.unit_type == ids.bunker && u.has(flags::COMPLETED));
        let empty = done_bunker && loaded == 0 && !marines.is_empty();
        let mut aimed: BTreeMap<UnitId, u32> = BTreeMap::new();
        for u in &mine {
            if u.order == self.orders.attack_unit {
                if let Some(t) = u.order_target.filter(|t| f.units.iter().any(|v| v.id == *t && self.is_enemy(v))) {
                    *aimed.entry(t).or_default() += 1;
                }
            }
        }
        if let Some(r) = self.rush.as_mut() {
            r.empty += empty as u64 * overlap(lo, hi, r.entry, r.end());
            if fr >= r.entry && fr <= r.end() {
                r.enemy_max = r.enemy_max.max(enemy_in);
                if attacking >= PULL_MIN && r.pull_frame.is_none() {
                    r.pull_frame = Some(fr);
                }
                r.pull_max = r.pull_max.max(attacking);
                r.move_max = r.move_max.max(moving);
                r.marines_max = r.marines_max.max(loaded);
                if loaded > 0 && r.load_frame.is_none() {
                    r.load_frame = Some(fr);
                }
                let n: u32 = aimed.values().sum();
                if n >= 2 && !repeat {
                    r.hhi_sum += aimed.values().map(|&k| (k as f64 / n as f64).powi(2)).sum::<f64>();
                    r.hhi_n += 1;
                }
            }
        }
        for u in &f.units {
            self.last_seen.insert(u.id, (u.owner, u.unit_type));
        }
        self.prev = hi;
        self.last = Some(hi);
    }

    pub fn ops(&self) -> OpsMetrics {
        let last = self.last.unwrap_or(0);
        let inc = |k: usize| (last >= INC_AT[k]).then_some(self.inc_mined[k] as f64);
        let inc_scv = |k: usize| {
            let avg = self.inc_scv[k] as f64 / MINUTE as f64;
            (last >= INC_AT[k] && avg > 0.0).then(|| self.inc_mined[k] as f64 / avg)
        };
        OpsMetrics {
            end_s: secs(last.min(WINDOW_END) as u64),
            mined: self.mined,
            gas: self.gas,
            inc_4m: inc(0),
            inc_6m: inc(1),
            inc_8m: inc(2),
            inc_10m: inc(3),
            inc_scv_4m: inc_scv(0),
            inc_scv_6m: inc_scv(1),
            inc_scv_8m: inc_scv(2),
            inc_scv_10m: inc_scv(3),
            mline_bldg_n: self.mline_bldg_n,
            mline_bldg_s: secs(self.mline),
            cc_lift_s: secs(self.lift),
            stranded_scv_s: secs(self.stranded),
            idle_scv_s: secs(self.idle),
            oversat_scv_s: secs(self.oversat),
            sblock_s: secs(self.sblock),
            sblock_n: self.sblock_n,
            sblock_first_s: self.sblock_first.map(|f| secs(f as u64)),
            float_min_rs: secs(self.float_min),
            float_gas_rs: secs(self.float_gas),
        }
    }

    pub fn field_check(&self) -> FieldCheck {
        let (mut depleted, mut unknown) = (0u32, 0u32);
        for &i in &self.mined_fields {
            match self.fields[i].init {
                Some(a) => depleted += a.saturating_sub(self.fields[i].amount),
                None => unknown += 1,
            }
        }
        let (scvs, carrying) = self.last_scvs;
        FieldCheck { depleted, fields: self.mined_fields.len() as u32, unknown_fields: unknown, scvs, scvs_max: self.scvs_max, carrying, lost_cargo: self.lost_cargo }
    }

    pub fn rush_combat(&self) -> Option<RushCombat> {
        let r = self.rush.as_ref()?;
        let since = |f: u32| secs((f - r.entry) as u64);
        Some(RushCombat {
            in_s: secs(r.entry as u64),
            enemy_max: r.enemy_max,
            pull_s: r.pull_frame.map(since),
            pull_max: r.pull_max,
            scv_move_max: r.move_max,
            scv_lost: r.scv_lost,
            kills: r.kills,
            bunker_marines_max: r.marines_max,
            bunker_load_s: r.load_frame.map(since),
            bunker_empty_s: secs(r.empty),
            focus_hhi: (r.hhi_n > 0).then(|| r.hhi_sum / r.hhi_n as f64),
            focus_obs: r.hhi_n,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::MapResource;
    use uranium_proto::{BitGrid, PlayerInfo, PlayerState, Race, StaticMap};

    const ME: u8 = 0;
    const ENEMY: u8 = 1;
    const OURS: Tile = (10, 10);
    const THEIRS: Tile = (100, 100);
    const START: Pos = (336, 336);
    const CC_POS: Pos = (1024, 1008);
    const FIELD_POS: Pos = (1312, 1008);
    const FIELD_ID: UnitId = 900;

    fn ty(name: &str) -> u16 {
        TypeTables::get().unit_by_name(name).unwrap_or_else(|| panic!("{name}")).id as u16
    }

    fn unit(id: UnitId, owner: u8, name: &str, pos: Pos, completed: bool) -> UnitState {
        UnitState { id, owner, unit_type: ty(name), pos, flags: if completed { flags::COMPLETED } else { 0 }, build_queue: [NO_TYPE; 5], ..Default::default() }
    }

    fn with(mut u: UnitState, flag: u32) -> UnitState {
        u.flags |= flag;
        u
    }

    fn ordered(mut u: UnitState, name: &str, target: Option<UnitId>) -> UnitState {
        u.order = order(name);
        u.order_target = target;
        u
    }

    fn frame(f: u32, units: Vec<UnitState>) -> RawFrame {
        RawFrame { frame: f, self_state: PlayerState::default(), units, visibility: BitGrid::new(1, 1), explored: BitGrid::new(1, 1), creep: BitGrid::new(1, 1), events: vec![], effects: vec![] }
    }

    fn state(minerals: i32, gas: i32, used: u16, total: u16) -> PlayerState {
        PlayerState { minerals, gas, supply_used: used, supply_total: total, ..Default::default() }
    }

    fn pre(fields: &[(Pos, u32)], base_locs: Vec<Tile>) -> MapPrecompute {
        let resources = fields.iter().map(|&(pos, amount)| MapResource { unit_type: ty("Resource_Mineral_Field"), pos, amount }).collect();
        MapPrecompute { version: crate::features::PRECOMPUTE_VERSION, base_locs, chokes: vec![], wall_candidates: vec![], resources }
    }

    fn tracker() -> OpsTracker {
        OpsTracker::new(ME, vec![ENEMY], Some(OURS), Some(&pre(&[(FIELD_POS, 1500)], vec![(30, 30), (60, 60)])))
    }

    fn cc() -> UnitState {
        unit(1, ME, "Terran_Command_Center", CC_POS, true)
    }

    fn field() -> UnitState {
        UnitState { resources: 1500, ..unit(FIELD_ID, 11, "Resource_Mineral_Field", FIELD_POS, true) }
    }

    fn scv(id: UnitId, order_name: &str, target: Option<UnitId>, carrying: bool) -> UnitState {
        let u = ordered(unit(id, ME, "Terran_SCV", (1200, 1008), true), order_name, target);
        if carrying { with(u, flags::CARRYING) } else { u }
    }

    fn ling(id: UnitId, pos: Pos) -> UnitState {
        unit(id, ENEMY, "Zerg_Zergling", pos, true)
    }

    #[test]
    fn deliveries_by_last_gather_target_052_r2() {
        let mut t = tracker();
        let refinery = unit(50, ME, "Terran_Refinery", (900, 900), true);
        let base = || vec![cc(), field(), refinery.clone()];
        t.step(&frame(100, [base(), vec![scv(10, "MiningMinerals", Some(FIELD_ID), false), scv(11, "HarvestGas", Some(50), false)]].concat()));
        t.step(&frame(200, [base(), vec![scv(10, "ReturnMinerals", Some(1), true), scv(11, "ReturnGas", Some(1), true)]].concat()));
        t.step(&frame(300, [base(), vec![scv(10, "MoveToMinerals", Some(FIELD_ID), false), scv(11, "MoveToGas", Some(50), false)]].concat()));
        let m = t.ops();
        assert_eq!((m.mined, m.gas), (8, 8));
        t.step(&frame(400, [base(), vec![scv(10, "ReturnMinerals", Some(1), true)]].concat()));
        t.step(&frame(500, [base(), vec![scv(10, "ReturnMinerals", Some(1), true)]].concat()));
        t.step(&frame(600, [base(), vec![scv(10, "MiningMinerals", Some(FIELD_ID), false)]].concat()));
        assert_eq!(t.ops().mined, 16);
        t.step(&frame(700, [base(), vec![scv(11, "ReturnGas", Some(1), true)]].concat()));
        t.step(&frame(800, [base(), vec![scv(11, "MiningMinerals", Some(FIELD_ID), false)]].concat()));
        assert_eq!((t.ops().mined, t.ops().gas), (16, 16));
        t.step(&frame(14_400, [base(), vec![scv(10, "ReturnMinerals", Some(1), true)]].concat()));
        t.step(&frame(14_408, [base(), vec![scv(10, "MiningMinerals", Some(FIELD_ID), false)]].concat()));
        assert_eq!(t.ops().mined, 16);
        assert_eq!(t.ops().end_s, 600.0, "window end 10:00");
        let mut t = tracker();
        let enemy = |c| UnitState { owner: ENEMY, ..scv(20, "ReturnMinerals", Some(2), c) };
        let unborn = |c| {
            let mut u = scv(21, "ReturnMinerals", Some(2), c);
            u.flags &= !flags::COMPLETED;
            u
        };
        t.step(&frame(100, vec![enemy(true), unborn(true)]));
        t.step(&frame(200, vec![enemy(false), unborn(false)]));
        assert_eq!(t.ops().mined, 0);
    }

    #[test]
    fn income_windows_and_short_games_052_r2() {
        let trip = |t: &mut OpsTracker, at: u32, scvs: u32| {
            let all = |carry: bool| (0..scvs).map(|i| if i == 0 { scv(10, if carry { "ReturnMinerals" } else { "MiningMinerals" }, Some(FIELD_ID), carry) } else { scv(100 + i, "MiningMinerals", Some(FIELD_ID), false) }).collect::<Vec<_>>();
            t.step(&frame(at - 1, [vec![cc(), field()], all(true)].concat()));
            t.step(&frame(at, [vec![cc(), field()], all(false)].concat()));
        };
        let mut t = tracker();
        trip(&mut t, 4_320, 10);
        trip(&mut t, 4_330, 10);
        trip(&mut t, 5_760, 10);
        let m = t.ops();
        assert_eq!((m.inc_4m, m.inc_6m), (Some(16.0), None), "4,330 and 5,760 counted, 4,320 not; game not at 6:00");
        assert_eq!(m.inc_scv_4m, Some(1.6), "10 SCVs all through (4,320, 5,760]");
        assert_eq!(m.mined, 24);
        let mut t = tracker();
        trip(&mut t, 5_000, 4);
        t.step(&frame(5_759, vec![cc()]));
        let m = t.ops();
        assert_eq!((m.inc_4m, m.inc_scv_4m, m.end_s), (None, None, 5_759.0 / 24.0));
        let mut t = tracker();
        t.step(&frame(4_320, vec![]));
        t.step(&frame(5_040, (0..10).map(|i| scv(100 + i, "MiningMinerals", None, false)).collect()));
        t.step(&frame(5_760, (0..20).map(|i| scv(100 + i, "MiningMinerals", None, false)).collect()));
        assert_eq!((t.ops().inc_4m, t.ops().inc_scv_4m), (Some(0.0), Some(0.0)));
        assert_eq!(t.inc_scv[0], 15 * 1_440);
        let mut t = tracker();
        t.step(&frame(5_760, vec![]));
        assert_eq!((t.ops().inc_4m, t.ops().inc_scv_4m), (Some(0.0), None));
    }

    #[test]
    fn mineral_line_buildings_052_r2() {
        let one = |b: UnitState, cc_pos: Pos| {
            let mut t = tracker();
            t.step(&frame(240, vec![UnitState { pos: cc_pos, ..cc() }, field(), b]));
            (t.ops().mline_bldg_n, t.ops().mline_bldg_s)
        };
        let depot = |tx: u16, ty_: u16, done| unit(5, ME, "Terran_Supply_Depot", (tx * 32 + 48, ty_ * 32 + 32), done);
        assert_eq!(one(depot(35, 29, true), CC_POS), (1, 10.0));
        assert_eq!(one(depot(35, 29, true), (1024, 1009)), (0, 0.0));
        assert_eq!(one(depot(35, 33, true), CC_POS).0, 0);
        assert_eq!(one(depot(35, 32, false), CC_POS).0, 1, "under construction counts");
        assert_eq!(one(depot(42, 31, true), CC_POS).0, 0);
        assert_eq!(one(unit(6, ME, "Terran_Bunker", (36 * 32 + 48, 31 * 32 + 32), true), CC_POS).0, 1);
        assert_eq!(one(unit(7, ME, "Terran_Refinery", (36 * 32 + 64, 31 * 32 + 32), true), CC_POS).0, 0);
        assert_eq!(one(unit(8, ME, "Terran_Comsat_Station", (36 * 32 + 32, 31 * 32 + 32), true), CC_POS).0, 0);
        assert_eq!(one(with(unit(9, ME, "Terran_Barracks", (36 * 32 + 64, 31 * 32 + 48), true), flags::LIFTED), CC_POS).0, 0);
        assert_eq!(one(unit(9, ME, "Terran_Barracks", (36 * 32 + 64, 31 * 32 + 48), true), CC_POS).0, 1);
        let mut t = tracker();
        for f in [240, 480, 720] {
            t.step(&frame(f, vec![cc(), field(), depot(35, 31, true), UnitState { id: 77, owner: ENEMY, ..depot(35, 32, true) }]));
        }
        assert_eq!((t.ops().mline_bldg_n, t.ops().mline_bldg_s), (1, 30.0));
        let mut t = tracker();
        t.step(&frame(240, vec![with(cc(), flags::LIFTED), field(), depot(35, 31, true)]));
        assert_eq!(t.ops().mline_bldg_n, 0);
    }

    #[test]
    fn lifted_cc_and_stranded_scvs_052_r2() {
        let mut t = tracker();
        let miners = || vec![scv(10, "MiningMinerals", Some(FIELD_ID), false), scv(11, "MoveToMinerals", Some(FIELD_ID), false)];
        t.step(&frame(240, [vec![cc(), field()], miners()].concat()));
        assert_eq!((t.ops().cc_lift_s, t.ops().stranded_scv_s), (0.0, 0.0));
        t.step(&frame(480, [vec![with(cc(), flags::LIFTED), field()], miners()].concat()));
        assert_eq!((t.ops().cc_lift_s, t.ops().stranded_scv_s), (10.0, 20.0));
        t.step(&frame(720, vec![with(cc(), flags::LIFTED), field(), scv(10, "ReturnMinerals", Some(1), true), scv(12, "ReturnMinerals", Some(1), true)]));
        assert_eq!(t.ops().stranded_scv_s, 30.0);
        let moved = |dx: u16| UnitState { pos: (FIELD_POS.0 - dx, FIELD_POS.1), ..cc() };
        t.step(&frame(960, [vec![moved(384), field()], miners()].concat()));
        assert_eq!(t.ops().stranded_scv_s, 30.0);
        t.step(&frame(1_200, [vec![moved(385), field()], miners()].concat()));
        assert_eq!(t.ops().stranded_scv_s, 50.0);
    }

    #[test]
    fn idle_and_oversaturation_052_r2() {
        let fields: Vec<(Pos, u32)> = (0..8).map(|i| ((1312, 880 + 32 * i as u16), 1500)).collect();
        let mut t = OpsTracker::new(ME, vec![ENEMY], Some(OURS), Some(&pre(&fields, vec![])));
        let seen = |amounts: &[u16]| amounts.iter().enumerate().map(|(i, &a)| UnitState { resources: a, ..unit(900 + i as u32, 11, "Resource_Mineral_Field", fields[i].0, true) }).collect::<Vec<_>>();
        let workers = |n: u32| (0..n).map(|i| scv(10 + i, "MiningMinerals", Some(900 + i % 8), false)).collect::<Vec<_>>();
        t.step(&frame(24, [vec![cc()], seen(&[1500; 8]), workers(24)].concat()));
        assert_eq!(t.ops().oversat_scv_s, 0.0, "24 = 3 x 8");
        t.step(&frame(48, [vec![cc()], seen(&[1500; 8]), workers(25)].concat()));
        assert_eq!(t.ops().oversat_scv_s, 1.0, "25: one over");
        t.step(&frame(72, [vec![cc()], seen(&[1500, 1500, 1500, 1500, 1500, 1500, 1500, 0]), workers(24)].concat()));
        assert_eq!(t.ops().oversat_scv_s, 4.0, "24 - 21 = 3 more");
        t.step(&frame(96, [vec![cc()], workers(24)].concat()));
        assert_eq!(t.ops().oversat_scv_s, 7.0);
        let mut f = frame(120, [vec![cc()], workers(24)].concat());
        f.events = vec![GameEvent::UnitDestroyed(906)];
        t.step(&f);
        assert_eq!(t.ops().oversat_scv_s, 13.0, "24 - 18 = 6 more");
        let mut t = tracker();
        let idle = with(unit(10, ME, "Terran_SCV", (500, 500), true), flags::IDLE);
        let loaded = UnitState { loaded_in: Some(99), ..with(unit(11, ME, "Terran_SCV", (500, 500), true), flags::IDLE | flags::LOADED) };
        t.step(&frame(48, vec![idle, loaded, unit(12, ME, "Terran_SCV", (500, 500), true), with(unit(13, ENEMY, "Terran_SCV", (500, 500), true), flags::IDLE)]));
        assert_eq!(t.ops().idle_scv_s, 2.0);
    }

    #[test]
    fn supply_block_052_r3() {
        let rax = || unit(3, ME, "Terran_Barracks", (2000, 2000), true);
        let blocked = |st: PlayerState, units: Vec<UnitState>| {
            let mut t = tracker();
            let mut f = frame(24, units);
            f.self_state = st;
            t.step(&f);
            t.ops().sblock_s == 1.0
        };
        assert!(blocked(state(50, 0, 20, 20), vec![rax()]));
        assert!(!blocked(state(49, 0, 20, 20), vec![rax()]), "cannot afford a marine");
        assert!(!blocked(state(50, 0, 18, 20), vec![rax()]), "a marine fits");
        assert!(!blocked(state(500, 0, 400, 400), vec![rax()]), "200 supply");
        assert!(blocked(state(500, 0, 398, 398), vec![rax()]));
        assert!(!blocked(state(500, 0, 20, 20), vec![with(rax(), flags::TRAINING)]), "training");
        assert!(!blocked(state(500, 0, 20, 20), vec![with(rax(), flags::LIFTED)]), "lifted");
        assert!(!blocked(state(500, 0, 20, 20), vec![with(rax(), flags::CONSTRUCTING)]), "building an addon");
        assert!(!blocked(state(500, 0, 20, 20), vec![unit(3, ME, "Terran_Barracks", (2000, 2000), false)]), "not completed");
        let mut queued = rax();
        queued.build_queue[0] = ty("Terran_Marine");
        assert!(!blocked(state(500, 0, 20, 20), vec![queued]), "queue not empty");
        assert!(blocked(state(500, 0, 20, 20), vec![rax(), unit(4, ME, "Terran_Supply_Depot", (2200, 2000), false)]), "a depot under construction does not help");
        let fac = unit(5, ME, "Terran_Factory", (2000, 2200), true);
        assert!(blocked(state(75, 0, 18, 20), vec![fac.clone()]) && !blocked(state(74, 0, 18, 20), vec![fac]));
        let port = unit(6, ME, "Terran_Starport", (2000, 2400), true);
        assert!(blocked(state(150, 100, 18, 20), vec![port.clone()]) && !blocked(state(150, 99, 18, 20), vec![port]));
        assert!(blocked(state(50, 0, 20, 20), vec![cc()]));
        let mut t = tracker();
        for (f, used) in [(24, 18), (48, 20), (72, 20), (96, 18), (120, 20)] {
            let mut x = frame(f, vec![rax()]);
            x.self_state = state(100, 0, used, 20);
            t.step(&x);
        }
        let m = t.ops();
        assert_eq!((m.sblock_n, m.sblock_first_s, m.sblock_s), (2, Some(2.0), 3.0));
    }

    #[test]
    fn float_thresholds_and_cc_exception_052_r4() {
        let run = |units: Vec<UnitState>| {
            let mut t = tracker();
            let mut f = frame(24, units);
            f.self_state = state(1000, 300, 0, 0);
            t.step(&f);
            (t.ops().float_min_rs, t.ops().float_gas_rs)
        };
        assert_eq!(run(vec![]), (600.0, 100.0));
        let fac = unit(5, ME, "Terran_Factory", (2000, 2200), true);
        let rax = with(unit(3, ME, "Terran_Barracks", (2000, 2000), true), flags::LIFTED);
        assert_eq!(run(vec![cc(), rax, fac]), (450.0, 50.0));
        assert_eq!(run(vec![unit(5, ME, "Terran_Factory", (2000, 2200), false)]), (600.0, 100.0), "not completed");
        let placer = |pos: Pos| UnitState { order_pos: Some(pos), ..ordered(unit(10, ME, "Terran_SCV", (500, 500), true), "PlaceBuilding", None) };
        assert_eq!(run(vec![placer((1984 + 96, 1968))]), (0.0, 100.0), "3 tiles: CC exception");
        assert_eq!(run(vec![placer((1984 + 97, 1968))]).0, 600.0, "97 px: counted");
        assert_eq!(run(vec![cc(), placer(CC_POS)]).0, 550.0);
        assert_eq!(run(vec![UnitState { order: order("Move"), ..placer((1984, 1968)) }]).0, 600.0, "not placing");
    }

    #[test]
    fn rush_entry_boundaries_052_r5() {
        let entered = |f: u32, u: UnitState| {
            let mut t = tracker();
            t.step(&frame(f, vec![u]));
            t.rush_combat().map(|r| r.in_s)
        };
        assert_eq!(entered(7_199, ling(50, (START.0 + 384, START.1))), Some(7_199.0 / 24.0));
        assert_eq!(entered(7_199, ling(50, (START.0 + 385, START.1))), None, "385 px");
        assert_eq!(entered(7_200, ling(50, START)), None, "5:00 is not before 5:00");
        assert_eq!(entered(100, unit(50, ENEMY, "Zerg_Mutalisk", START, true)), None, "flyer");
        assert_eq!(entered(100, unit(50, ENEMY, "Zerg_Drone", START, true)), None, "worker");
        assert_eq!(entered(100, unit(50, ENEMY, "Zerg_Sunken_Colony", START, true)), None, "building");
        assert_eq!(entered(100, unit(50, 11, "Zerg_Zergling", START, true)), None, "not an enemy player");
        let mut t = tracker();
        t.step(&frame(100, vec![]));
        assert!(t.rush_combat().is_none(), "no entry: null");
    }

    #[test]
    fn rush_pull_losses_and_kills_052_r5() {
        let attacker = |id, name: &str| ordered(unit(id, ME, "Terran_SCV", START, true), name, Some(60));
        let mut t = tracker();
        t.step(&frame(2_000, vec![attacker(10, "AttackUnit"), ling(60, START), ling(61, START), unit(62, ENEMY, "Zerg_Drone", START, true)]));
        t.step(&frame(2_100, vec![attacker(10, "AttackUnit"), attacker(11, "Attack1"), ling(60, START)]));
        assert_eq!(t.play_until(14_400), 14_400);
        assert_eq!(t.play_until(3_000), 4_160, "the window end when later");
        let mut f = frame(2_400, vec![attacker(10, "AttackUnit"), attacker(11, "AttackMove"), attacker(12, "AttackFixedRange"), ordered(unit(13, ME, "Terran_SCV", START, true), "Move", None)]);
        f.events = vec![GameEvent::UnitDestroyed(60), GameEvent::UnitDestroyed(62), GameEvent::UnitDestroyed(11)];
        t.step(&f);
        let mut f = frame(2_500, vec![attacker(10, "Attack2")]);
        f.events = vec![GameEvent::UnitDestroyed(12), GameEvent::UnitDestroyed(13), GameEvent::UnitDestroyed(999)];
        t.step(&f);
        let mut f = frame(4_161, (20..30).map(|i| attacker(i, "Attack1")).collect());
        f.events = vec![GameEvent::UnitDestroyed(10), GameEvent::UnitDestroyed(61)];
        t.step(&f);
        let r = t.rush_combat().unwrap();
        assert_eq!(r.in_s, 2_000.0 / 24.0);
        assert_eq!((r.enemy_max, r.pull_s, r.pull_max, r.scv_move_max), (2, Some(400.0 / 24.0), 3, 1), "the drone is no army");
        assert_eq!((r.scv_lost, r.kills), (3, 1), "SCVs 11, 12, 13; zergling 60 (drone 62 is no army, 999 unknown)");
        let mut t = tracker();
        t.step(&frame(2_900, vec![unit(10, ME, "Terran_SCV", START, true)]));
        let mut f = frame(3_000, vec![ling(60, START)]);
        f.events = vec![GameEvent::UnitDestroyed(10)];
        t.step(&f);
        assert_eq!(t.rush_combat().unwrap().scv_lost, 0);
    }

    #[test]
    fn rush_bunker_052_r5() {
        let bunker = |done| unit(30, ME, "Terran_Bunker", (400, 400), done);
        let marine = |id, loaded: bool| {
            let u = unit(id, ME, "Terran_Marine", (400, 400), true);
            if loaded { UnitState { loaded_in: Some(30), ..with(u, flags::LOADED) } } else { u }
        };
        let mut t = tracker();
        t.step(&frame(3_000, vec![ling(60, START), bunker(true), marine(40, false)]));
        t.step(&frame(3_240, vec![bunker(true), marine(40, false), marine(41, false)]));
        t.step(&frame(3_480, vec![bunker(true), marine(40, true), marine(41, true)]));
        t.step(&frame(3_720, vec![bunker(true), marine(40, true), marine(41, false)]));
        let r = t.rush_combat().unwrap();
        assert_eq!((r.bunker_load_s, r.bunker_marines_max, r.bunker_empty_s), (Some(20.0), 2, 10.0));
        let mut t = tracker();
        t.step(&frame(2_000, vec![bunker(true), marine(40, false)]));
        t.step(&frame(3_000, vec![ling(60, START), bunker(true), marine(40, true)]));
        assert_eq!(t.rush_combat().unwrap().bunker_load_s, Some(0.0));
        let mut t = tracker();
        t.step(&frame(2_000, vec![bunker(true), marine(40, true)]));
        t.step(&frame(3_000, vec![ling(60, START), bunker(true), marine(40, false)]));
        t.step(&frame(3_480, vec![bunker(true), marine(40, true)]));
        assert_eq!(t.rush_combat().unwrap().bunker_load_s, Some(20.0));
        let mut t = tracker();
        t.step(&frame(3_000, vec![ling(60, START), bunker(false), marine(40, false)]));
        t.step(&frame(3_240, vec![bunker(false), marine(40, false)]));
        let r = t.rush_combat().unwrap();
        assert_eq!((r.bunker_load_s, r.bunker_empty_s, r.bunker_marines_max), (None, 0.0, 0));
    }

    #[test]
    fn rush_focus_hhi_052_r5() {
        let at = |id, target| ordered(unit(id, ME, "Terran_Marine", START, true), "AttackUnit", Some(target));
        let run = |obs: Vec<Vec<UnitState>>| {
            let mut t = tracker();
            for (i, units) in obs.into_iter().enumerate() {
                t.step(&frame(3_000 + 24 * i as u32, [vec![ling(60, START), ling(61, START)], units].concat()));
            }
            let r = t.rush_combat().unwrap();
            (r.focus_hhi, r.focus_obs)
        };
        assert_eq!(run(vec![vec![at(1, 60), at(2, 60)]]), (Some(1.0), 1));
        assert_eq!(run(vec![vec![at(1, 60), at(2, 61)]]), (Some(0.5), 1));
        assert_eq!(run(vec![vec![at(1, 60), at(2, 60)], vec![at(1, 60), at(2, 61)]]), (Some(0.75), 2));
        assert_eq!(run(vec![vec![at(1, 60)], vec![at(1, 60), at(2, 999)], vec![ordered(at(1, 60), "Attack1", Some(60)), at(2, 60)]]), (None, 0));
    }

    #[test]
    fn field_check_052_ac2() {
        let mut t = tracker();
        let fld = |amount| UnitState { resources: amount, ..field() };
        let far = |amount| UnitState { resources: amount, ..unit(901, 11, "Resource_Mineral_Field", (3000, 3000), true) };
        let theirs = || UnitState { resources: 1200, ..unit(902, 11, "Resource_Mineral_Field", (3000, 3200), true) };
        let scout = || scv(12, "MoveToMinerals", Some(902), false);
        t.step(&frame(100, vec![cc(), fld(1500), scv(10, "MiningMinerals", Some(FIELD_ID), false), scout(), theirs()]));
        t.step(&frame(200, vec![cc(), fld(1492), scv(10, "ReturnMinerals", Some(1), true), scout(), theirs()]));
        t.step(&frame(300, vec![cc(), fld(1484), scv(10, "MiningMinerals", Some(FIELD_ID), false), scv(11, "MiningMinerals", Some(901), false), far(300), scout(), scv(13, "HarvestGas", None, false)]));
        let gas = || scv(13, "ReturnGas", Some(1), true);
        t.step(&frame(400, vec![cc(), fld(1476), scv(10, "ReturnMinerals", Some(1), true), scv(11, "ReturnMinerals", Some(1), true), far(292), scout(), gas()]));
        t.step(&frame(500, vec![cc(), fld(1476), scv(10, "ReturnMinerals", Some(1), true), scv(11, "MiningMinerals", Some(901), false), far(284), scout(), gas()]));
        assert_eq!(t.ops().mined, 16);
        let want = FieldCheck { depleted: 24, fields: 2, unknown_fields: 1, scvs: 4, scvs_max: 4, carrying: 1, lost_cargo: 0 };
        assert_eq!(t.field_check(), want);
        t.step(&frame(14_400, vec![fld(1476)]));
        t.step(&frame(14_408, vec![cc(), fld(1000), scv(10, "MiningMinerals", Some(FIELD_ID), false)]));
        assert_eq!(t.field_check(), want);
    }

    #[test]
    fn field_check_mined_out_and_lost_cargo_052_ac2() {
        let mut t = OpsTracker::new(ME, vec![ENEMY], Some(OURS), Some(&pre(&[(FIELD_POS, 16), ((1312, 1104), 1500)], vec![(30, 30), (60, 60)])));
        let fld = |amount| UnitState { resources: amount, ..field() };
        t.step(&frame(100, vec![cc(), fld(16), scv(10, "MiningMinerals", Some(FIELD_ID), false)]));
        t.step(&frame(200, vec![cc(), fld(8), scv(10, "ReturnMinerals", Some(1), true)]));
        t.step(&frame(300, vec![cc(), scv(10, "MiningMinerals", Some(FIELD_ID), false), scv(11, "HarvestGas", None, false)]));
        assert_eq!(t.field_check(), FieldCheck { depleted: 8, fields: 1, unknown_fields: 0, scvs: 2, scvs_max: 2, carrying: 0, lost_cargo: 0 });
        let mut f = frame(400, vec![cc(), scv(10, "ReturnMinerals", Some(1), true), scv(11, "ReturnGas", Some(1), true)]);
        f.events = vec![GameEvent::UnitDestroyed(FIELD_ID)];
        t.step(&f);
        let mut f = frame(500, vec![cc()]);
        f.events = vec![GameEvent::UnitDestroyed(10), GameEvent::UnitDestroyed(11)];
        t.step(&f);
        assert_eq!(t.ops().mined, 8);
        assert_eq!(t.field_check(), FieldCheck { depleted: 16, fields: 1, unknown_fields: 0, scvs: 0, scvs_max: 2, carrying: 0, lost_cargo: 8 });
    }

    #[test]
    fn window_end_repeat_and_deaths_052_r1() {
        let rax = || unit(3, ME, "Terran_Barracks", (2000, 2000), true);
        let depot = unit(5, ME, "Terran_Supply_Depot", (35 * 32 + 48, 31 * 32 + 32), true);
        let mut t = tracker();
        let mut f = frame(14_400, vec![cc(), field()]);
        f.self_state = state(100, 0, 18, 20);
        t.step(&f);
        let mut f = frame(14_424, vec![cc(), field(), rax(), depot]);
        f.self_state = state(100, 0, 20, 20);
        t.step(&f);
        let m = t.ops();
        assert_eq!((m.sblock_n, m.sblock_first_s, m.sblock_s, m.mline_bldg_n, m.mline_bldg_s), (0, None, 0.0, 0, 0.0));
        let at = |id, target| ordered(unit(id, ME, "Terran_Marine", START, true), "AttackUnit", Some(target));
        let mut t = tracker();
        let obs = || frame(3_000, vec![ling(60, START), ling(61, START), at(1, 60), at(2, 61)]);
        t.step(&obs());
        t.step(&obs());
        assert_eq!(t.rush_combat().unwrap().focus_obs, 1);
        let mut t = tracker();
        t.step(&frame(3_000, vec![ling(60, START), unit(61, ENEMY, "Zerg_Mutalisk", START, true), scv(10, "ReturnMinerals", Some(1), true)]));
        let mut f = frame(3_024, vec![]);
        f.events = vec![GameEvent::UnitDestroyed(61), GameEvent::UnitDestroyed(10)];
        t.step(&f);
        let r = t.rush_combat().unwrap();
        assert_eq!((r.kills, r.scv_lost, t.ops().mined), (1, 1, 0));
    }

    #[test]
    fn shared_field_belongs_to_the_nearer_base_052_r2() {
        let mut t = tracker();
        let far = UnitState { id: 2, pos: (1601, 1008), ..cc() };
        let workers = |n: u32| (0..n).map(|i| scv(10 + i, "MiningMinerals", Some(FIELD_ID), false)).collect::<Vec<_>>();
        t.step(&frame(24, [vec![cc(), far, field()], workers(4)].concat()));
        assert_eq!(t.ops().oversat_scv_s, 1.0, "4 SCVs on the one field of the nearer base: 1 over, the other base has none");
    }

    #[test]
    fn integrals_ignore_observation_spacing_052_ac1() {
        let world = |f: u32| {
            let phase = (f.saturating_sub(1) / 960) % 4;
            let mut units = vec![if phase == 1 { with(cc(), flags::LIFTED) } else { cc() }, field(), unit(3, ME, "Terran_Barracks", (2000, 2000), true), unit(6, ME, "Terran_Bunker", (36 * 32 + 48, 31 * 32 + 32), true)];
            units.extend((0..(4 + phase)).map(|i| if i == 0 { with(scv(10, "MiningMinerals", Some(FIELD_ID), false), flags::IDLE) } else { scv(10 + i, "MiningMinerals", Some(FIELD_ID), false) }));
            if (2_880..5_040).contains(&f) {
                units.push(ling(60, START));
            }
            if f > 2_880 {
                units.push(unit(40, ME, "Terran_Marine", (400, 400), true));
            }
            let mut fr = frame(f, units);
            fr.self_state = state(300 + 100 * phase as i32, 250, if phase == 2 { 20 } else { 10 }, 20);
            fr
        };
        let run = |step: u32| {
            let mut t = tracker();
            let mut f = 0;
            while f <= 15_360 {
                t.step(&world(f));
                f += step;
            }
            (t.ops(), t.rush_combat().unwrap().bunker_empty_s)
        };
        let (a, ea) = run(96);
        for step in [8, 24] {
            let (b, eb) = run(step);
            for (x, y, k) in [
                (a.mline_bldg_s, b.mline_bldg_s, "mline"),
                (a.cc_lift_s, b.cc_lift_s, "lift"),
                (a.idle_scv_s, b.idle_scv_s, "idle"),
                (a.stranded_scv_s, b.stranded_scv_s, "stranded"),
                (a.sblock_s, b.sblock_s, "sblock"),
                (a.float_min_rs, b.float_min_rs, "float_min"),
                (a.float_gas_rs, b.float_gas_rs, "float_gas"),
                (a.oversat_scv_s, b.oversat_scv_s, "oversat"),
                (ea, eb, "bunker_empty"),
            ] {
                assert_eq!(x, y, "{k} at step {step}");
            }
            assert_eq!((a.inc_scv_4m, a.inc_scv_10m, a.end_s, a.sblock_n), (b.inc_scv_4m, b.inc_scv_10m, b.end_s, b.sblock_n));
        }
        assert!(a.cc_lift_s > 0.0 && a.sblock_s > 0.0 && a.float_min_rs > 0.0 && a.mline_bldg_s > 0.0 && ea > 0.0 && a.idle_scv_s > 0.0 && a.stranded_scv_s > 0.0);
        assert_eq!(a.end_s, 600.0);
    }

    #[test]
    fn own_vision_only_033_r5_052_r1() {
        let player = |id, start, is_self, is_enemy| PlayerInfo { id, name: String::new(), race: Race::Zerg, start_tile: Some(start), is_self, is_enemy };
        let info = |enemy_start: Tile| GameInfo {
            map_name: String::new(),
            map_hash: [0; 32],
            map_size_tiles: (128, 128),
            self_player: ME,
            players: vec![player(ME, OURS, true, false), player(ENEMY, enemy_start, false, true)],
            static_map: StaticMap { width_tiles: 128, height_tiles: 128, walkable: BitGrid::new(1, 1), buildable: BitGrid::new(1, 1), ground_height: vec![], start_locations: vec![OURS, THEIRS] },
            latency_frames: 0,
        };
        let p = pre(&[(FIELD_POS, 1500)], vec![(30, 30)]);
        let obs = || vec![frame(100, vec![cc(), field(), scv(10, "MiningMinerals", Some(FIELD_ID), false)]), frame(3_000, vec![cc(), ling(60, START)]), frame(3_100, vec![cc()])];
        let run = |t: &mut OpsTracker| {
            for f in obs() {
                t.step(&f);
            }
            (t.ops(), t.rush_combat())
        };
        let a = run(&mut OpsTracker::for_game(&info(THEIRS), Some(&p)));
        let b = run(&mut OpsTracker::for_game(&info(OURS), Some(&p)));
        assert_eq!(a, b, "the enemy's start tile is not read");
        assert!(a.1.is_some());
        let mut t = OpsTracker::for_game(&info(THEIRS), Some(&p));
        for mut f in obs() {
            f.units.retain(|u| u.owner != ENEMY);
            t.step(&f);
        }
        assert!(t.rush_combat().is_none());
        assert_eq!(t.ops(), a.0);
    }

    #[test]
    fn object_keys_in_interface_order_052_r6() {
        let t = tracker();
        let text = serde_json::to_string(&t.ops()).unwrap();
        assert_eq!(text, r#"{"end_s":0.0,"mined":0,"gas":0,"inc_4m":null,"inc_6m":null,"inc_8m":null,"inc_10m":null,"inc_scv_4m":null,"inc_scv_6m":null,"inc_scv_8m":null,"inc_scv_10m":null,"mline_bldg_n":0,"mline_bldg_s":0.0,"cc_lift_s":0.0,"stranded_scv_s":0.0,"idle_scv_s":0.0,"oversat_scv_s":0.0,"sblock_s":0.0,"sblock_n":0,"sblock_first_s":null,"float_min_rs":0.0,"float_gas_rs":0.0}"#);
        let r = serde_json::to_string(&RushCombat::default()).unwrap();
        assert_eq!(r, r#"{"in_s":0.0,"enemy_max":0,"pull_s":null,"pull_max":0,"scv_move_max":0,"scv_lost":0,"kills":0,"bunker_marines_max":0,"bunker_load_s":null,"bunker_empty_s":0.0,"focus_hhi":null,"focus_obs":0}"#);
        assert_eq!(t.ids.cheapest.iter().map(|c| (c.1, c.2, c.3)).collect::<Vec<_>>(), vec![(50, 0, 2), (50, 0, 2), (75, 0, 4), (150, 100, 4)]);
    }
}
