//! Snapshot -> `RawFrame` / `GameInfo` with the OpenBW bridge's meaning (012-R3, 012-R12, 012-R13).
//! The OpenBW shim reads BWAPI 4.2 with CompleteMapInformation on and filters with
//! `visible_to_self`; this module recomputes the same BWAPI values from raw game memory. Source
//! references: UU = BWAPI/Source/BWAPI/UnitUpdate.cpp, GU = GameUnits.cpp, UL =
//! BWAPILIB/Source/Unit.cpp, shim = bridge/openbw/shim/shim.cpp, mapping = reports/m7_scr_field_mapping.md.

use std::collections::{HashMap, HashSet};

use sha2::{Digest, Sha256};
use uranium_env::types::flag;
use uranium_proto::*;

use crate::snapshot::{MOVING_OR_ACCELERATING, RawStatic, RawUnit, Snapshot, UnitRef, status, tile};
use crate::tables::{Tables, has_weapon, unit_type};

/// BWAPI values of one live unit (all units are accessible: CompleteMapInformation semantics).
#[derive(Debug, Clone, Default)]
pub struct Derived {
    pub bwapi_type: u16,
    pub completed: bool,
    pub morphing: bool,
    pub constructing: bool,
    pub idle: bool,
    pub training: bool,
    /// BWAPI `getOrder` (mapped through the order table).
    pub order: u8,
    pub transport: UnitRef,
    pub pos: (i32, i32),
    /// BWAPI `isVisible(self)` and `isDetected` (UU:94-108).
    pub visible: bool,
    pub detected: bool,
    /// Passes the shim's `visible_to_self` (own unit, or seen and detected).
    pub observed: bool,
    /// Live add-on (BWAPI `getAddon`), 0 = none.
    pub addon: UnitRef,
}

/// Per-game state the observation needs between frames.
#[derive(Default)]
pub struct ObserveState {
    /// BWAPI order of each unit last frame: `isMoving` reads the previous frame's order (UU:294).
    prev_order: HashMap<u32, u8>,
}

fn has_flag(t: u16, f: &str) -> bool {
    unit_type(t).is_some_and(|u| u.has(f))
}

fn is_building(t: u16) -> bool {
    has_flag(t, flag::BUILDING)
}

fn ceil256(v: i32) -> u16 {
    if v <= 0 { 0 } else { ((v + 255) / 256).min(u16::MAX as i32) as u16 }
}

/// BWAPI `UnitType` of a live unit (UU:136-155): mineral fields fold into one type, a morphing
/// building shows what it becomes.
fn bwapi_type(u: &RawUnit, t: &Tables) -> u16 {
    if t.is_mineral_field(u.unit_type) {
        return t.units.mineral_field;
    }
    let o = &t.orders;
    if is_building(u.unit_type) && [o.zerg_birth, o.zerg_building_morph, o.zerg_unit_morph, o.incomplete_morphing].contains(&u.order) {
        let q = u.build_queue[(u.build_slot % 5) as usize];
        if q != t.units.none {
            return q;
        }
    }
    u.unit_type
}

fn has_empty_queue(u: &RawUnit, t: &Tables) -> bool {
    u.build_slot < 5 && u.build_queue[u.build_slot as usize] == t.units.none
}

/// BWAPI `isScannerVisible`: any tile within 64 px of the sweep is visible.
fn scanner_visible(s: &Snapshot, pos: (i32, i32)) -> bool {
    let (l, tp, r, b) = ((pos.0 - 64) / 32, (pos.1 - 64) / 32, (pos.0 + 64) / 32, (pos.1 + 64) / 32);
    (l..=r).any(|x| (tp..=b).any(|y| s.visible(x, y)))
}

/// BWAPI values of every unit of the snapshot, in `s.units` order.
pub fn derive(s: &Snapshot, t: &Tables) -> Vec<Derived> {
    let o = &t.orders;
    let me = s.local;
    let mut d: Vec<Derived> = s
        .units
        .iter()
        .map(|u| {
            let ty = bwapi_type(u, t);
            let morphing = [o.zerg_birth, o.zerg_building_morph, o.zerg_unit_morph, o.incomplete_morphing].contains(&u.order);
            let completed = u.status & status::COMPLETED != 0 && !morphing;
            // Transport (UU:157-169).
            let fighter = [t.units.interceptor, t.units.scarab, t.units.spider_mine].contains(&ty);
            let in_transport = u.status & (status::IN_TRANSPORT | status::IN_BUILDING) != 0;
            let transport = if fighter {
                if u.fighter_outside || in_transport { u.fighter_parent } else { 0 }
            } else if in_transport {
                u.related
            } else {
                0
            };
            let tt = unit_type(ty);
            // isTraining (UU:346-355).
            let training = if ty == t.units.nuclear_silo && u.secondary_order == o.train {
                true
            } else if !tt.is_some_and(|x| x.has(flag::PRODUCES_UNITS)) {
                false
            } else if tt.is_some_and(|x| x.race == "Zerg" && x.has(flag::RESOURCE_DEPOT)) {
                false
            } else {
                !has_empty_queue(u, t)
            };
            // isConstructing (UU:366-381).
            let build_unit = u.current_build_unit != 0;
            let constructing = morphing
                || o.constructing.contains(&u.order)
                || u.order == o.build_addon
                || u.secondary_order == o.build_addon
                || (!completed && build_unit);
            // isIdle (UU:384-404), on raw orders.
            let idle = !(training || constructing || morphing || u.order == o.research_tech || u.order == o.upgrade) && o.idle.contains(&u.order);
            // Visibility and detection for our player (UU:79-108).
            let burrowed = u.status & status::BURROWED != 0;
            let (visible, detected) = if !u.has_sprite {
                (false, false)
            } else if u.player == me {
                (true, true)
            } else if u.unit_type == t.units.scanner_sweep {
                (scanner_visible(s, (u.pos.0 as i32, u.pos.1 as i32)), true)
            } else {
                let mask = 1u32 << me;
                let mut vis = u.visibility_mask as u32 & mask != 0;
                let needs = u.status & status::REQUIRES_DETECTION != 0;
                if needs {
                    vis &= u.detection_status == u32::MAX
                        || u.detection_status & mask != 0
                        || u.flingy_flags & MOVING_OR_ACCELERATING != 0
                        || u.order == o.move_
                        || u.ground_cd > 0
                        || u.air_cd > 0
                        || !burrowed;
                }
                let can_detect = !needs || u.detection_status == u32::MAX || u.detection_status & mask != 0;
                (vis, vis && can_detect)
            };
            Derived { bwapi_type: ty, completed, morphing, constructing, idle, training, order: t.bwapi_order(u.order), transport, pos: (0, 0), visible, detected, observed: false, addon: 0 }
        })
        .collect();
    // Positions follow the transport (UU:172); add-ons (UU:415-428).
    for i in 0..d.len() {
        let u = &s.units[i];
        let src = s.unit(d[i].transport).unwrap_or(u);
        d[i].pos = (src.pos.0 as i32, src.pos.1 as i32);
        if is_building(d[i].bwapi_type) {
            let addon_of = |r: UnitRef| s.index.get(&r).filter(|&&j| unit_type(s.units[j].unit_type).is_some_and(|x| x.is_addon())).map(|_| r);
            d[i].addon = addon_of(u.current_build_unit).or_else(|| addon_of(u.addon)).unwrap_or(0);
        }
    }
    // GU augmentUnitData:259-290: a builder and its building, a building and its unfinished add-on.
    let mut marks: Vec<usize> = Vec::new();
    for i in 0..d.len() {
        let u = &s.units[i];
        if let Some(&j) = s.index.get(&u.order_target).filter(|_| d[i].order == o.constructing_building) {
            marks.extend([i, j]);
        } else if let Some(&j) = s.index.get(&d[i].addon).filter(|&&j| !d[j].completed) {
            marks.extend([i, j]);
        }
    }
    for k in marks {
        d[k].constructing = true;
        d[k].idle = false;
    }
    // shim `visible_to_self`: own units, else seen (any footprint tile visible) and detected.
    for i in 0..d.len() {
        let u = &s.units[i];
        d[i].observed = if u.player == me {
            true
        } else if !d[i].visible {
            false
        } else {
            let tt = unit_type(d[i].bwapi_type);
            let (tw, th) = tt.map_or((1, 1), |x| (x.tile_width, x.tile_height));
            let (tx, ty) = ((d[i].pos.0 - tw * 16) / 32, (d[i].pos.1 - th * 16) / 32);
            let any = (ty..ty + th).any(|y| (tx..tx + tw).any(|x| s.visible(x, y)));
            let cloaked = u.status & status::CLOAKED != 0 && u.status & status::BURROWED == 0;
            let burrowed = u.status & status::BURROWED != 0;
            any && !((cloaked || burrowed) && !d[i].detected)
        };
    }
    d
}

/// BWAPI `isGatheringMinerals || isGatheringGas` (UnitShared.cpp:449-530), on BWAPI orders.
fn gathering(s: &Snapshot, d: &[Derived], i: usize, t: &Tables) -> bool {
    let u = &s.units[i];
    let o = &t.orders;
    if !has_flag(d[i].bwapi_type, flag::WORKER) || u.status & status::IS_GATHERING == 0 {
        return false;
    }
    let order = d[i].order;
    let target_ok = |r: UnitRef, gas: bool| {
        s.index.get(&r).is_some_and(|&j| {
            let tj = d[j].bwapi_type;
            if gas {
                d[j].completed && s.units[j].player == u.player && tj != t.units.vespene_geyser && (t.units.refineries.contains(&tj) || has_flag(tj, flag::RESOURCE_DEPOT))
            } else {
                t.is_mineral_field(tj) || (d[j].completed && s.units[j].player == u.player && has_flag(tj, flag::RESOURCE_DEPOT))
            }
        })
    };
    let minerals = o.gather_minerals.contains(&order)
        && if order == o.reset_collision {
            u.carried == 2
        } else if [o.wait_for_minerals, o.mining_minerals, o.return_minerals].contains(&order) {
            true
        } else {
            target_ok(u.move_target, false) || target_ok(u.order_target, false)
        };
    let gas = o.gather_gas.contains(&order)
        && if order == o.reset_collision {
            u.carried == 1
        } else if [o.wait_for_gas, o.harvest_gas, o.return_gas].contains(&order) {
            true
        } else {
            target_ok(u.move_target, true) || target_ok(u.order_target, true)
        };
    minerals || gas
}

/// BWAPI `getTrainingQueue` (UU:576-597) padded with `None`.
fn training_queue(u: &RawUnit, ty: u16, t: &Tables) -> [u16; 5] {
    let none = t.units.none;
    let mut q = [none; 5];
    let mut n = 0;
    if !has_empty_queue(u, t) {
        let mut i = (u.build_slot % 5) as usize;
        while u.build_queue[i] != none && n < 5 {
            q[n] = u.build_queue[i];
            n += 1;
            i = (i + 1) % 5;
        }
    }
    if ty == t.units.nuclear_silo && u.secondary_order == t.orders.train {
        q[0] = t.units.nuclear_missile;
    }
    q
}

/// The `UnitState` of observed unit `i` (shim `fill_unit`); `ids` names observed units only.
fn unit_state(s: &Snapshot, d: &[Derived], i: usize, t: &Tables, st: &ObserveState, observed_tag: &dyn Fn(UnitRef) -> Option<u32>) -> UnitState {
    let u = &s.units[i];
    let di = &d[i];
    let o = &t.orders;
    let ty = di.bwapi_type;
    let tt = unit_type(ty);
    let raw_t = unit_type(u.unit_type);
    let mut flags = 0u32;
    let mut set = |b: bool, bit: u32| {
        if b {
            flags |= bit;
        }
    };
    let burrowed = u.status & status::BURROWED != 0;
    set(di.completed, flags::COMPLETED);
    set(di.constructing, flags::CONSTRUCTING);
    set(ty == t.units.siege_tank_siege || ty == t.units.hero_duke_siege, flags::SIEGED);
    set(burrowed, flags::BURROWED);
    set(u.status & status::CLOAKED != 0 && !burrowed, flags::CLOAKED);
    set(di.detected, flags::DETECTED);
    set(u.status & status::IN_AIR != 0 && raw_t.is_some_and(|x| x.is_building()), flags::LIFTED);
    set(u.timers[7] != 0, flags::STIMMED);
    let worker = tt.is_some_and(|x| x.is_worker());
    set(worker && (u.carried == 1 || u.carried == 2), flags::CARRYING);
    set(di.transport != 0, flags::LOADED);
    set(di.idle, flags::IDLE);
    // UNDER_ATTACK stays 0: BWAPI 4.2 hard-codes recentlyAttacked = false (UU:468, mapping §4-1).
    set(gathering(s, d, i, t), flags::GATHERING);
    set(di.training, flags::TRAINING);
    // UL isBeingConstructed.
    let being_constructed = di.morphing || (!di.completed && (tt.is_some_and(|x| x.race != "Terran") || u.current_build_unit != 0));
    set(being_constructed, flags::BEING_CONSTRUCTED);
    let prev = st.prev_order.get(&u.tag).copied();
    set(u.flingy_flags & MOVING_OR_ACCELERATING != 0 || prev == Some(o.move_), flags::MOVING);

    // Remaining times (UU:586-660): inside info, sent for every unit as OpenBW does (AMB-030-5
    // hides another player's in the agent).
    let mut remaining_train = s.unit(u.current_build_unit).map_or(0, |b| b.remaining_build_time);
    if [t.units.hatchery, t.units.lair, t.units.hive].contains(&ty) {
        let larvae = s.units.iter().filter(|l| l.unit_type == t.units.larva && l.related == u.ptr).count();
        remaining_train = if larvae >= 3 { 0 } else { u.larva_timer as u16 * 9 + ((u.order_wait as u16 + 8) % 9) };
    }
    let build_type_set = u.order == o.incomplete_morphing
        || ([o.zerg_unit_morph, o.zerg_building_morph].contains(&u.order) && u.build_queue[(u.build_slot % 5) as usize] != t.units.none);
    let remaining_build = if !di.completed && (!di.morphing || build_type_set) { u.remaining_build_time } else { 0 };
    let remaining_research = if u.order == o.research_tech || u.order == o.upgrade { u.research_time } else { 0 };
    let ground_cd = if ty == t.units.reaver { u.order_timer } else { u.ground_cd };
    UnitState {
        id: u.tag,
        owner: u.player,
        unit_type: ty,
        pos: (di.pos.0.max(0) as u16, di.pos.1.max(0) as u16),
        vel: (u.speed.0 as i16, u.speed.1 as i16),
        // BWAPI angle (facing - 64) as radians, back to 256 steps in the shim.
        angle: u.facing.wrapping_sub(64),
        hp: ceil256(u.hp),
        shield: if tt.is_some_and(|x| x.max_shields > 0) { ceil256(u.shields) } else { 0 },
        energy: if tt.is_some_and(|x| x.has(flag::SPELLCASTER)) { ceil256(u.energy as i32) } else { 0 },
        order: di.order as u16,
        order_target: observed_tag(u.order_target),
        order_pos: {
            let (x, y) = (u.order_target_pos.0 as i32, u.order_target_pos.1 as i32);
            let valid = x >= 0 && y >= 0 && x < s.width as i32 * 32 && y < s.height as i32 * 32;
            valid.then_some((x as u16, y as u16))
        },
        ground_cd,
        air_cd: u.air_cd,
        spell_cd: u.spell_cd,
        flags,
        build_queue: training_queue(u, ty, t),
        remaining_train,
        remaining_build,
        remaining_research,
        resources: if tt.is_some_and(|x| x.is_resource()) { u.resource_amount } else { 0 },
        loaded_in: observed_tag(di.transport),
        status_timers: StatusTimers {
            lockdown: u.timers[0] as u16,
            irradiate: u.timers[1] as u16,
            matrix: u.timers[2] as u16,
            stasis: u.timers[3] as u16,
            plague: u.timers[4] as u16,
            ensnare: u.timers[5] as u16,
            maelstrom: u.timers[6] as u16,
            stim: u.timers[7] as u16,
            blind: u.blind,
            parasite: u.parasited,
        },
        spider_mines: if t.units.vultures.contains(&ty) { u.spider_mines } else { 0 },
        nuke_armed: ty == t.units.nuclear_silo && u.nuke_ready,
        has_addon: di.addon != 0,
    }
}

/// Ghost nuke tracking (GU:229-249): the painted target, and whether its launch was reported.
#[derive(Default)]
pub struct NukeState {
    target: HashMap<u32, (i32, i32)>,
    detected: HashSet<u32>,
}

impl NukeState {
    /// Nuke dots this frame (shim: only visible ones) and new launches. A launch is reported only
    /// where we see the target (BWAPI without CompleteMapInformation; mapping §4-3, 013 list).
    pub fn update(&mut self, s: &Snapshot, t: &Tables, ghost: u16) -> (Vec<Pos>, Vec<Pos>) {
        let (mut dots, mut launches) = (vec![], vec![]);
        for u in s.units.iter().filter(|u| u.unit_type == ghost) {
            if u.order == t.orders.nuke_paint {
                self.target.insert(u.tag, (u.order_target_pos.0 as i32, u.order_target_pos.1 as i32));
            }
            if u.order != t.orders.nuke_track {
                self.detected.remove(&u.tag);
                continue;
            }
            let p = self.target.get(&u.tag).copied().unwrap_or((0, 0));
            let seen = s.visible(p.0 / 32, p.1 / 32);
            let pos = (p.0.max(0) as u16, p.1.max(0) as u16);
            if seen {
                dots.push(pos);
            }
            if self.detected.insert(u.tag) && seen {
                launches.push(pos);
            }
        }
        (dots, launches)
    }
}

/// The local player's `PlayerState` (shim `fill_player_state`, PlayerImpl.cpp:185-238).
pub fn player_state(s: &Snapshot) -> PlayerState {
    let me = &s.me;
    let bits = |v: &[bool]| v.iter().enumerate().filter(|(_, b)| **b).fold(0u64, |a, (i, _)| a | 1 << i);
    let mut tech = bits(&me.tech);
    // Nuclear_Strike (45) is "researched" when the missile is available (PlayerImpl.cpp:210).
    if me.nuke_available {
        tech |= 1 << 45;
    }
    let mut upgrades = me.upgrades.clone();
    upgrades.resize(63, 0);
    PlayerState {
        minerals: me.minerals as i32,
        gas: me.gas as i32,
        supply_used: me.supply_used.min(u16::MAX as u32) as u16,
        supply_total: me.supply_provided.min(me.supply_max).min(u16::MAX as u32) as u16,
        upgrades,
        tech,
        researching: bits(&me.researching),
        upgrading: bits(&me.upgrading),
    }
}

/// `UnitState`s of the observed units (`d` from `derive`), sorted as the OpenBW bridge sorts them
/// (`raw_frame`). References (order target, transport) name observed units only (shim
/// `id_or_none`).
pub fn observe_units(s: &Snapshot, d: &[Derived], t: &Tables, st: &ObserveState) -> Vec<UnitState> {
    let observed: HashMap<UnitRef, u32> = s.units.iter().zip(d).filter(|(_, x)| x.observed).map(|(u, _)| (u.ptr, u.tag)).collect();
    let tag = |r: UnitRef| observed.get(&r).copied();
    let mut units: Vec<UnitState> = (0..s.units.len()).filter(|&i| d[i].observed).map(|i| unit_state(s, d, i, t, st, &tag)).collect();
    units.sort_by_key(|u| (u.id, u.owner, u.unit_type, u.pos));
    units
}

impl ObserveState {
    /// Remember this frame's BWAPI orders for the next frame's `MOVING` flag.
    pub fn end_frame(&mut self, s: &Snapshot, d: &[Derived]) {
        self.prev_order = s.units.iter().zip(d).map(|(u, x)| (u.tag, x.order)).collect();
    }
}

/// A full observation (shim `observe`): units, grids, effects, given events.
pub fn raw_frame(s: &Snapshot, t: &Tables, units: Vec<UnitState>, nuke_dots: &[Pos], events: Vec<GameEvent>) -> RawFrame {
    let (w, h) = (s.width, s.height);
    let mut vis = Vec::with_capacity(w as usize * h as usize);
    let mut exp = Vec::with_capacity(vis.capacity());
    let mut creep = Vec::with_capacity(vis.capacity());
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let v = s.visible(x, y);
            vis.push(v as u8);
            exp.push(s.explored(x, y) as u8);
            let f = s.tile(x, y).unwrap_or(0);
            creep.push((v && f & (tile::HAS_CREEP | tile::TEMPORARY_CREEP) != 0) as u8);
        }
    }
    // Bullets whose centre tile we see now (AMB-030-4, shim:521-526).
    let mut effects: Vec<Effect> = s
        .bullets
        .iter()
        .filter(|b| s.visible(b.pos.0 as i32 / 32, b.pos.1 as i32 / 32))
        .map(|b| Effect {
            kind: t.bullet_effect(b.flingy_id),
            pos: (b.pos.0.max(0) as u16, b.pos.1.max(0) as u16),
            radius: 0,
            remaining_frames: Some(b.remove_timer as u16),
            source_owner: s.unit(b.parent).map(|p| p.player),
        })
        .collect();
    effects.extend(nuke_dots.iter().map(|&pos| Effect { kind: EffectKind::NukeDot, pos, radius: 0, remaining_frames: None, source_owner: None }));
    effects.extend(units.iter().filter_map(|u| t.spell_unit_effect(u.unit_type).map(|kind| Effect { kind, pos: u.pos, radius: 0, remaining_frames: None, source_owner: Some(u.owner) })));
    effects.sort_by_key(|e| (format!("{:?}", e.kind), e.pos, e.remaining_frames, e.source_owner));
    let mut events = events;
    events.sort_by_key(event_order);
    RawFrame {
        frame: s.frame,
        self_state: player_state(s),
        units,
        visibility: BitGrid::from_bytes(w, h, &vis),
        explored: BitGrid::from_bytes(w, h, &exp),
        creep: BitGrid::from_bytes(w, h, &creep),
        events,
        effects,
    }
}

/// Deterministic event order within one observation (as the OpenBW bridge, convert.rs).
pub fn event_order(e: &GameEvent) -> (u8, u32, u32) {
    match *e {
        GameEvent::UnitCreated(id) => (0, id, 0),
        GameEvent::UnitDestroyed(id) => (1, id, 0),
        GameEvent::UnitMorph(id) => (2, id, 0),
        GameEvent::UnitDiscover(id) => (3, id, 0),
        GameEvent::UnitEvade(id) => (4, id, 0),
        GameEvent::NukeDetect((x, y)) => (5, x as u32, y as u32),
        GameEvent::CommandFailed { cmd_index, error } => (6, cmd_index as u32, error as u32),
    }
}

fn race(r: u8) -> Race {
    match r {
        0 => Race::Zerg,
        1 => Race::Terran,
        2 => Race::Protoss,
        6 => Race::Random,
        _ => Race::Unknown,
    }
}

/// BWAPI player types counted as players (GameEvents.cpp:138-142: None < type < Closed), minus
/// Neutral.
fn is_player(player_type: u8) -> bool {
    (1..7).contains(&player_type)
}

/// `GameInfo` (012-R13, shim onStart). `map_bytes` is the map file for the hash (None = zeros,
/// logged by the caller). A lobby-Random player keeps race Random (030-R11).
pub fn game_info(r: &RawStatic, map_bytes: Option<&[u8]>) -> GameInfo {
    let (w, h) = (r.width, r.height);
    let start_tile = |p: (u16, u16)| ((p.0 as i32 - 64) / 32, (p.1 as i32 - 48) / 32);
    let players = r
        .players
        .iter()
        .filter(|p| is_player(p.player_type))
        .map(|p| {
            let (x, y) = r.start_positions.get(p.id as usize).copied().filter(|s| *s != (0, 0)).map(start_tile).unwrap_or((-1, -1));
            PlayerInfo {
                id: p.id,
                name: p.name.clone(),
                race: if p.lobby_race == 6 { Race::Random } else { race(p.race) },
                start_tile: (x >= 0 && y >= 0).then_some((x as u16, y as u16)),
                is_self: p.id == r.local,
                is_enemy: p.is_enemy,
            }
        })
        .collect();
    let tiles = |f: &dyn Fn(u32) -> u8| r.tiles.iter().map(|&v| f(v)).collect::<Vec<u8>>();
    let buildable = tiles(&|v| (v & tile::UNBUILDABLE == 0) as u8);
    let height = tiles(&|v| (v & tile::VERY_HIGH != 0) as u8 + 2 * (v & tile::MIDDLE != 0) as u8 + 4 * (v & tile::HIGH != 0) as u8);
    GameInfo {
        map_name: r.map_title.clone(),
        map_hash: map_bytes.map_or([0; 32], |b| Sha256::digest(b).into()),
        map_size_tiles: (w, h),
        self_player: r.local,
        players,
        static_map: StaticMap {
            width_tiles: w,
            height_tiles: h,
            walkable: BitGrid::from_bytes(w * 4, h * 4, &r.walkable),
            buildable: BitGrid::from_bytes(w, h, &buildable),
            ground_height: height,
            start_locations: r
                .start_positions
                .iter()
                .filter(|p| **p != (0, 0))
                .map(|&p| start_tile(p))
                .map(|(x, y)| (x.max(0) as u16, y.max(0) as u16))
                .collect(),
        },
        latency_frames: if r.multiplayer { 2 } else { 1 },
    }
}

/// BWAPI `isWalkable` at 8 px (Map.cpp:101-108, OpenBW `is_walkable`): `minitile(x, y)` is the
/// vf4 walkable bit of the tile's megatile, `None` when the tileset data could not be read (then
/// a partially walkable tile counts as walkable).
pub fn walkability(w: u16, h: u16, tiles: &[u32], minitile: &dyn Fn(usize, usize, usize) -> Option<bool>) -> Vec<u8> {
    let (ww, wh) = (w as usize * 4, h as usize * 4);
    let mut out = vec![0u8; ww * wh];
    for y in 0..wh {
        for x in 0..ww {
            if y >= wh.saturating_sub(4) || (y >= wh.saturating_sub(8) && (x < 20 || x >= ww.saturating_sub(20))) {
                continue;
            }
            let ti = (y / 4) * w as usize + x / 4;
            let f = tiles[ti];
            let walk = if f & tile::HAS_CREEP != 0 {
                true
            } else if f & tile::PARTIALLY_WALKABLE != 0 {
                minitile(ti, x % 4, y % 4).unwrap_or(true)
            } else {
                f & tile::WALKABLE != 0
            };
            out[y * ww + x] = walk as u8;
        }
    }
    out
}

/// Whether unit type `ty` can attack-move (BWAPI `canAttackMove`, Templates.h:582: armed or a
/// medic, and able to move). Used to send unarmed units a move instead (shim:168-171).
pub fn can_attack_move(ty: u16, t: &Tables) -> bool {
    let Some(x) = unit_type(ty) else { return false };
    let armed = has_weapon(x.ground_weapon) || has_weapon(x.air_weapon) || ty == t.units.carrier || ty == t.units.reaver;
    (ty == t.units.medic || armed) && x.has("AutoAttackAndMove")
}

pub fn can_move(ty: u16) -> bool {
    has_flag(ty, "AutoAttackAndMove")
}
