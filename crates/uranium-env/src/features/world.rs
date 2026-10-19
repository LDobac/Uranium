//! Per-frame view shared by feature extraction (030), masks (034) and the executor (031), so all
//! three judge eligibility the same way (031-R4, 031-R7).

use std::collections::{BTreeMap, HashMap};
use std::sync::OnceLock;

use uranium_proto::{RawFrame, UnitId, UnitState, flags};

use super::map::MapInfo;
use super::memory::{AgentMemory, ResourceTile, resource_tile};
use super::schema::{CELLS, MAP, ORDER_NONE, OWNER_ENEMY, OWNER_NEUTRAL, OWNER_SELF};
use crate::types::{TypeTables, UnitType};

/// Unit / tech / upgrade ids resolved by name from the BWAPI tables (013-R1).
#[derive(Debug)]
pub struct Ids {
    pub scv: u16,
    pub marine: u16,
    pub firebat: u16,
    pub medic: u16,
    pub ghost: u16,
    pub vulture: u16,
    pub tank: u16,
    pub tank_sieged: u16,
    pub goliath: u16,
    pub wraith: u16,
    pub dropship: u16,
    pub vessel: u16,
    pub battlecruiser: u16,
    pub spider_mine: u16,
    pub nuke: u16,
    pub cc: u16,
    pub comsat: u16,
    pub silo: u16,
    pub bunker: u16,
    pub turret: u16,
    pub refinery: u16,
    pub factory: u16,
    pub starport: u16,
    pub science_facility: u16,
    pub geyser: u16,
    pub scanner_sweep: u16,
    pub tech_stim: u16,
    pub tech_lockdown: u16,
    pub tech_emp: u16,
    pub tech_mines: u16,
    pub tech_siege: u16,
    pub tech_matrix: u16,
    pub tech_irradiate: u16,
    pub tech_yamato: u16,
    pub tech_cloak_field: u16,
    pub tech_personnel_cloak: u16,
    pub tech_restoration: u16,
    pub tech_flare: u16,
    pub tech_healing: u16,
    pub tech_nuke: u16,
    pub tech_scanner: u16,
    pub order_research: u16,
    pub order_upgrade: u16,
    pub order_constructing: u16,
    pub order_sieging: u16,
    pub order_unsieging: u16,
    pub order_harvest_gas: u16,
    /// Orders that cannot be interrupted (orders.dat `can_be_interrupted` = 0, read 2026-10-19):
    /// BWAPI refuses most commands (Unit_Busy) while a unit runs one (액션 감사 F15·F16).
    pub orders_uninterruptible: Vec<u16>,
    /// `UnitTypes::None` (empty training queue slot).
    pub none: u16,
}



impl Ids {
    pub fn get() -> &'static Ids {
        static I: OnceLock<Ids> = OnceLock::new();
        I.get_or_init(|| {
            let t = TypeTables::get();
            let u = |n: &str| t.unit_by_name(n).unwrap_or_else(|| panic!("unit type {n}")).id as u16;
            let tech = |n: &str| t.techs.iter().find(|x| x.name == n).unwrap_or_else(|| panic!("tech {n}")).id as u16;
            let order = |n: &str| crate::types::order_id(n).unwrap_or_else(|| panic!("order {n}"));
            Ids {
                scv: u("Terran_SCV"),
                marine: u("Terran_Marine"),
                firebat: u("Terran_Firebat"),
                medic: u("Terran_Medic"),
                ghost: u("Terran_Ghost"),
                vulture: u("Terran_Vulture"),
                tank: u("Terran_Siege_Tank_Tank_Mode"),
                tank_sieged: u("Terran_Siege_Tank_Siege_Mode"),
                goliath: u("Terran_Goliath"),
                wraith: u("Terran_Wraith"),
                dropship: u("Terran_Dropship"),
                vessel: u("Terran_Science_Vessel"),
                battlecruiser: u("Terran_Battlecruiser"),
                spider_mine: u("Terran_Vulture_Spider_Mine"),
                nuke: u("Terran_Nuclear_Missile"),
                cc: u("Terran_Command_Center"),
                comsat: u("Terran_Comsat_Station"),
                silo: u("Terran_Nuclear_Silo"),
                bunker: u("Terran_Bunker"),
                turret: u("Terran_Missile_Turret"),
                refinery: u("Terran_Refinery"),
                factory: u("Terran_Factory"),
                starport: u("Terran_Starport"),
                science_facility: u("Terran_Science_Facility"),
                geyser: u("Resource_Vespene_Geyser"),
                scanner_sweep: u("Spell_Scanner_Sweep"),
                tech_stim: tech("Stim_Packs"),
                tech_lockdown: tech("Lockdown"),
                tech_emp: tech("EMP_Shockwave"),
                tech_mines: tech("Spider_Mines"),
                tech_siege: tech("Tank_Siege_Mode"),
                tech_matrix: tech("Defensive_Matrix"),
                tech_irradiate: tech("Irradiate"),
                tech_yamato: tech("Yamato_Gun"),
                tech_cloak_field: tech("Cloaking_Field"),
                tech_personnel_cloak: tech("Personnel_Cloaking"),
                tech_restoration: tech("Restoration"),
                tech_flare: tech("Optical_Flare"),
                tech_healing: tech("Healing"),
                tech_nuke: tech("Nuclear_Strike"),
                tech_scanner: tech("Scanner_Sweep"),
                order_research: order("ResearchTech"),
                order_upgrade: order("Upgrade"),
                order_constructing: order("ConstructingBuilding"),
                order_sieging: order("Sieging"),
                order_unsieging: order("Unsieging"),
                order_harvest_gas: order("HarvestGas"),
                orders_uninterruptible: [
                    "VultureMine", "BuildingLiftOff", "LiftingOff", "HarvestGas", "Harvest3", "Harvest4", "Interrupted", "Sieging",
                    "Unsieging", "NukeTrack", "ResetCollision", "ResetHarvestCollision", "MedicHealToIdle",
                ]
                .into_iter()
                .map(order)
                .collect(),
                none: u("None"),
            }
        })
    }
}

/// One entity row (030 §2 order: own, visible enemy, remembered enemy buildings, neutral).
#[derive(Debug, Clone)]
pub struct Entity {
    pub unit: UnitState,
    pub owner: i32,
    pub remembered: bool,
    pub seen_frame: u32,
}

pub struct World<'a> {
    pub frame: &'a RawFrame,
    pub mem: &'a AgentMemory,
    pub map: &'a MapInfo,
    pub types: &'static TypeTables,
    pub ids: &'static Ids,
    pub entities: Vec<Entity>,
    /// Own completed unit counts by type.
    pub own_completed: HashMap<u16, u32>,
    /// Space used in each own transport (dropship / bunker).
    pub cargo: HashMap<u32, i32>,
    /// Own building id -> attached addon type.
    pub addon_of: HashMap<u32, u16>,
    /// An own nuclear silo holds an armed nuke (034 can_cast_nuke).
    pub nuke_ready: bool,
    /// Tiles covered by any known building or resource (own, visible, remembered), 034 location masks.
    pub blocked: Vec<bool>,
}

/// Unit ids of resources known only from the map precompute (never seen, so their game id is
/// unknown): `UNSEEN_RESOURCE_ID + index in MapInfo::resources`. Not targetable (034 target
/// classes need a real id) and never sent in a command.
pub const UNSEEN_RESOURCE_ID: UnitId = 0xFFFF_0000;

pub fn is_unseen_resource_id(id: UnitId) -> bool {
    id >= UNSEEN_RESOURCE_ID
}

/// OpenBW `unit_is_disabled`: lockdown, stasis or maelstrom. Such a unit takes no command, so it
/// has no selection capability and no executor picks it (034-R6, B-11).
pub fn is_disabled(u: &UnitState) -> bool {
    let s = &u.status_timers;
    s.lockdown > 0 || s.stasis > 0 || s.maelstrom > 0
}

/// 030-R2 exception (010-R5, E-3, AMB-030-3): resources out of vision as remembered neutral
/// entities, in tile order. Seen ones keep their last observed state and frame; resources never
/// seen take the map's initial state with `seen_frame = 0` (known from the map at game start).
/// Tiles seen empty (`resources_gone`) are left out. The current amount of a resource out of
/// vision is never used (no map hack).
fn unseen_resources(mem: &AgentMemory, map: &MapInfo, visible: &std::collections::HashSet<UnitId>) -> Vec<Entity> {
    let types = TypeTables::get();
    let mut out: BTreeMap<ResourceTile, Entity> = BTreeMap::new();
    for (k, r) in &mem.resources {
        if !visible.contains(&r.unit.id) {
            out.insert(*k, Entity { unit: r.unit.clone(), owner: OWNER_NEUTRAL, remembered: true, seen_frame: r.seen_frame });
        }
    }
    // Display fields (owner, hp, shield, energy) as a seen resource of that type reports them, the
    // first seen in tile order (deterministic); order, targets, flags and timers are reset.
    let mut templates: HashMap<u16, &UnitState> = HashMap::new();
    for s in mem.resources.values() {
        templates.entry(s.unit.unit_type).or_insert(&s.unit);
    }
    for (i, r) in map.resources.iter().enumerate() {
        let Some(k) = resource_tile(r.unit_type, r.pos) else { continue };
        if mem.resources.contains_key(&k) || mem.resources_gone.contains(&k) || out.contains_key(&k) {
            continue;
        }
        let max_hp = types.unit(r.unit_type).map_or(0, |t| t.max_hp.clamp(0, u16::MAX as i32) as u16);
        let (owner, hp, shield, energy) = templates.get(&r.unit_type).map_or((NEUTRAL_PLAYER, max_hp, 0, 0), |s| (s.owner, s.hp, s.shield, s.energy));
        let unit = UnitState {
            id: UNSEEN_RESOURCE_ID + i as UnitId,
            owner,
            unit_type: r.unit_type,
            pos: r.pos,
            hp,
            shield,
            energy,
            order: ORDER_NONE,
            flags: flags::COMPLETED,
            build_queue: [Ids::get().none; 5],
            resources: r.amount.min(u16::MAX as u32) as u16,
            ..Default::default()
        };
        out.insert(k, Entity { unit, owner: OWNER_NEUTRAL, remembered: true, seen_frame: 0 });
    }
    out.into_values().collect()
}

/// BW neutral player (owner of map resources).
const NEUTRAL_PLAYER: u8 = 11;

pub fn tile_of(u: &UnitState) -> (i32, i32) {
    (u.pos.0 as i32 / 32, u.pos.1 as i32 / 32)
}

/// Top-left tile of a building footprint from its centre position. BWAPI places a building at
/// `tile * 32 + tile_size * 16`; the pixel `dimensions` are asymmetric for some types (Machine
/// Shop: 39 left, 31 right) and would land one tile off.
pub fn top_left(u: &UnitState, t: &UnitType) -> (i32, i32) {
    ((u.pos.0 as i32 - t.tile_width * 16).div_euclid(32), (u.pos.1 as i32 - t.tile_height * 16).div_euclid(32))
}

impl<'a> World<'a> {
    pub fn new(frame: &'a RawFrame, mem: &'a AgentMemory, map: &'a MapInfo) -> Self {
        let types = TypeTables::get();
        let ids = Ids::get();
        let ty = |u: &UnitState| types.unit(u.unit_type);
        let is_spell = |u: &UnitState| ty(u).is_none_or(|t| t.name.starts_with("Spell_"));

        let mut own = vec![];
        let mut enemy = vec![];
        let mut neutral = vec![];
        let mut visible = std::collections::HashSet::new();
        for u in &frame.units {
            visible.insert(u.id);
            if is_spell(u) {
                continue;
            }
            let e = Entity { unit: u.clone(), owner: 0, remembered: false, seen_frame: frame.frame };
            if u.owner == mem.self_player {
                own.push(Entity { owner: OWNER_SELF, ..e });
            } else if mem.is_enemy(u.owner) {
                enemy.push(Entity { owner: OWNER_ENEMY, ..e });
            } else if ty(u).is_some_and(|t| t.is_resource()) {
                neutral.push(Entity { owner: OWNER_NEUTRAL, ..e });
            }
        }
        let remembered: Vec<Entity> = mem
            .enemy_buildings
            .iter()
            .filter(|(id, _)| !visible.contains(id))
            .map(|(_, r)| Entity { unit: r.unit.clone(), owner: OWNER_ENEMY, remembered: true, seen_frame: r.seen_frame })
            .collect();
        let unseen = unseen_resources(mem, map, &visible);
        let entities: Vec<Entity> = own.into_iter().chain(enemy).chain(remembered).chain(neutral).chain(unseen).collect();

        let mut own_completed = HashMap::new();
        let mut cargo = HashMap::new();
        for u in frame.units.iter().filter(|u| u.owner == mem.self_player) {
            if u.has(flags::COMPLETED) {
                *own_completed.entry(u.unit_type).or_insert(0) += 1;
            }
            if let (Some(tr), Some(t)) = (u.loaded_in, ty(u)) {
                *cargo.entry(tr).or_insert(0) += t.space_required;
            }
        }

        // Addons sit right of the building's footprint, one tile down (BW layout).
        let mut addon_of = HashMap::new();
        let own_buildings: Vec<&UnitState> = frame.units.iter().filter(|u| u.owner == mem.self_player && ty(u).is_some_and(|t| t.is_building())).collect();
        let addons: BTreeMap<(i32, i32), u16> = own_buildings
            .iter()
            .filter(|u| ty(u).is_some_and(|t| t.is_addon()))
            .map(|u| (top_left(u, ty(u).unwrap()), u.unit_type))
            .collect();
        for b in &own_buildings {
            let t = ty(b).unwrap();
            if t.is_addon() || !b.has_addon {
                continue;
            }
            let (x, y) = top_left(b, t);
            if let Some(&a) = addons.get(&(x + t.tile_width, y + t.tile_height - 2)) {
                addon_of.insert(b.id, a);
            }
        }

        let mut blocked = vec![false; CELLS];
        let mut block = |u: &UnitState| {
            let Some(t) = ty(u) else { return };
            if !(t.is_building() || t.is_resource()) || u.has(flags::LIFTED) {
                return;
            }
            let (x, y) = top_left(u, t);
            for dy in 0..t.tile_height {
                for dx in 0..t.tile_width {
                    let (tx, ty) = (x + dx, y + dy);
                    if tx >= 0 && ty >= 0 && (tx as usize) < MAP && (ty as usize) < MAP {
                        blocked[ty as usize * MAP + tx as usize] = true;
                    }
                }
            }
        };
        for e in &entities {
            block(&e.unit);
        }
        for r in mem.resources.values() {
            block(&r.unit);
        }

        let nuke_ready = frame.units.iter().any(|s| s.owner == mem.self_player && s.unit_type == ids.silo && s.nuke_armed);
        Self { frame, mem, map, types, ids, entities, own_completed, cargo, addon_of, nuke_ready, blocked }
    }

    pub fn ty(&self, u: &UnitState) -> Option<&'static UnitType> {
        self.types.unit(u.unit_type)
    }

    pub fn researched(&self, tech: u16) -> bool {
        // Techs without a research cost (scan, matrix, healing, nuke) are available from the start.
        let free = self.types.techs.get(tech as usize).is_some_and(|t| t.minerals == 0);
        free || self.frame.self_state.tech & (1u64 << tech) != 0
    }

    pub fn tech_energy(&self, tech: u16) -> u16 {
        self.types.techs.get(tech as usize).map_or(0, |t| t.energy.max(0) as u16)
    }

    pub fn own(&self) -> impl Iterator<Item = &Entity> {
        self.entities.iter().filter(|e| e.owner == OWNER_SELF)
    }

    pub fn has_completed(&self, unit_type: u16) -> bool {
        self.own_completed.get(&unit_type).copied().unwrap_or(0) > 0
    }

    /// 034-R8: a completed own building that is constructing its addon (BWAPI `isConstructing`
    /// on a completed building, or an incomplete addon in its slot).
    pub fn building_addon(&self, u: &UnitState) -> bool {
        self.ty(u).is_some_and(|t| t.is_building())
            && u.has(flags::COMPLETED)
            && (u.has(flags::CONSTRUCTING) || (u.has_addon && super::masks::addon_incomplete(self, u)))
    }

    /// 031-R1: own completed, landed, idle-enough buildings of `producer` type that can run
    /// `what` now, ordered by (queue length, id) — or by energy for the scanner.
    pub fn producers(&self, producer: u16, need_addon: Option<u16>, research: bool) -> Vec<&UnitState> {
        let mut v: Vec<&UnitState> = self
            .frame
            .units
            .iter()
            .filter(|u| u.owner == self.mem.self_player && u.unit_type == producer)
            .filter(|u| u.has(flags::COMPLETED) && !u.has(flags::LIFTED))
            // 034-R6: a disabled (e.g. stasis-ed) building takes no command.
            .filter(|u| !is_disabled(u))
            // 034-R8 (사용자 결정 2026-10-19): the game refuses Train / Research / Upgrade
            // (Unit_Busy) while the building constructs its addon.
            .filter(|u| !self.building_addon(u))
            .filter(|u| need_addon.is_none_or(|a| self.addon_of.get(&u.id) == Some(&a)))
            .filter(|u| if research { u.remaining_research == 0 && queue_len(u) == 0 } else { queue_len(u) < 5 && u.remaining_research == 0 })
            .collect();
        v.sort_by_key(|u| (queue_len(u), u.id));
        v
    }
}

pub fn queue_len(u: &UnitState) -> usize {
    let none = Ids::get().none;
    u.build_queue.iter().filter(|&&t| t != none).count()
}
