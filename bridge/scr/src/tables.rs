//! SC:R bridge tables (013-R1, 012-R5): order normalisation, effects and command opcodes from
//! `bridge/scr/tables/*.toml`, plus the BWAPI unit / order / tech ids the bridge needs, looked up
//! by name in the uranium-env BWAPI tables. No version-specific number lives in the code.

use std::collections::HashMap;
use std::sync::OnceLock;

use serde::Deserialize;
use uranium_env::types::{TypeTables, UnitType, order_id};
use uranium_proto::EffectKind;

/// Version key every table must carry (013-R1, AMB-012-2).
pub const VERSION: &str = "scr-1.23.10.13515";

const ORDERS_TOML: &str = include_str!("../tables/orders.toml");
const EFFECTS_TOML: &str = include_str!("../tables/effects.toml");
const COMMANDS_TOML: &str = include_str!("../tables/commands.toml");

#[derive(Deserialize)]
struct OrdersFile {
    version: String,
    move_to_attack: String,
    overrides: Vec<(String, String)>,
}

#[derive(Deserialize)]
struct EffectsFile {
    version: String,
    bullets: Vec<(u16, String)>,
    spell_units: Vec<(u16, String)>,
}

#[derive(Deserialize, Clone, Copy, Debug)]
pub struct Opcodes {
    pub select: u8,
    pub right_click: u8,
    pub targeted_order: u8,
    pub unload_unit: u8,
    pub place: u8,
    pub train: u8,
    pub train_fighter: u8,
    pub unit_morph: u8,
    pub building_morph: u8,
    pub research: u8,
    pub upgrade: u8,
    pub hold_position: u8,
    pub stop: u8,
    pub reaver_stop: u8,
    pub carrier_stop: u8,
    pub return_cargo: u8,
    pub siege: u8,
    pub unsiege: u8,
    pub cloak: u8,
    pub decloak: u8,
    pub burrow: u8,
    pub unburrow: u8,
    pub lift: u8,
    pub unload_all: u8,
    pub stim: u8,
    pub merge_archon: u8,
    pub merge_dark_archon: u8,
    pub cancel_construction: u8,
    pub cancel_train: u8,
    pub cancel_research: u8,
    pub cancel_upgrade: u8,
    pub cancel_addon: u8,
    pub cancel_unit_morph: u8,
    pub chat: u8,
}

#[derive(Deserialize)]
struct CommandsFile {
    version: String,
    opcodes: Opcodes,
    tech_orders: Vec<String>,
}

/// BWAPI order ids by name (the order enum is the raw game order id).
#[derive(Debug, Clone, Copy)]
pub struct Orders {
    pub die: u8,
    pub stop: u8,
    pub guard: u8,
    pub player_guard: u8,
    pub move_: u8,
    pub follow: u8,
    pub attack1: u8,
    /// Order of a fog-ghost attack (012-R17): the engine's MoveToAttack, named in orders.toml.
    pub move_to_attack: u8,
    pub attack_move: u8,
    pub tower_attack: u8,
    pub carrier_attack: u8,
    pub reaver_attack: u8,
    pub patrol: u8,
    pub harvest1: u8,
    pub harvest2: u8,
    pub repair: u8,
    pub pickup_bunker: u8,
    pub pickup_transport: u8,
    pub move_unload: u8,
    pub rally_point_tile: u8,
    pub rally_point_unit: u8,
    pub place_building: u8,
    pub place_addon: u8,
    pub building_land: u8,
    pub heal_move: u8,
    pub research_tech: u8,
    pub upgrade: u8,
    pub train: u8,
    pub build_addon: u8,
    pub constructing_building: u8,
    pub zerg_birth: u8,
    pub zerg_building_morph: u8,
    pub zerg_unit_morph: u8,
    pub incomplete_morphing: u8,
    pub incomplete_building: u8,
    pub incomplete_warping: u8,
    pub nuke_paint: u8,
    pub nuke_track: u8,
    pub reset_collision: u8,
    pub wait_for_minerals: u8,
    pub mining_minerals: u8,
    pub return_minerals: u8,
    pub wait_for_gas: u8,
    pub harvest_gas: u8,
    pub return_gas: u8,
    /// Raw orders after which BWAPI counts a unit idle (UnitUpdate.cpp isIdle).
    pub idle: [u8; 14],
    /// BWAPI `isConstructing` orders (UnitUpdate.cpp), besides morphing and add-ons.
    pub constructing: [u8; 11],
    pub gather_minerals: [u8; 7],
    pub gather_gas: [u8; 7],
}

/// BWAPI unit type ids by name.
#[derive(Debug, Clone, Copy)]
pub struct Units {
    pub mineral_field: u16,
    pub mineral_field_2: u16,
    pub mineral_field_3: u16,
    pub vespene_geyser: u16,
    pub medic: u16,
    pub bunker: u16,
    pub dropship: u16,
    pub shuttle: u16,
    pub overlord: u16,
    pub carrier: u16,
    pub reaver: u16,
    pub interceptor: u16,
    pub scarab: u16,
    pub spider_mine: u16,
    pub scanner_sweep: u16,
    pub nuclear_silo: u16,
    pub nuclear_missile: u16,
    pub siege_tank_siege: u16,
    pub hatchery: u16,
    pub lair: u16,
    pub hive: u16,
    pub larva: u16,
    pub mutalisk: u16,
    pub hydralisk: u16,
    pub spire: u16,
    pub creep_colony: u16,
    pub refineries: [u16; 3],
    pub vultures: [u16; 2],
    pub ghost: u16,
    pub hero_duke_siege: u16,
    pub none: u16,
}

/// BWAPI tech ids by name.
#[derive(Debug, Clone, Copy)]
pub struct Techs {
    pub stim_packs: u16,
    pub tank_siege_mode: u16,
    pub personnel_cloaking: u16,
    pub cloaking_field: u16,
    pub burrowing: u16,
    pub archon_warp: u16,
    pub dark_archon_meld: u16,
    pub healing: u16,
}

pub struct Tables {
    /// Raw order -> BWAPI order (identity unless listed).
    order_map: [u8; 256],
    bullets: HashMap<u16, EffectKind>,
    spell_units: HashMap<u16, EffectKind>,
    pub op: Opcodes,
    /// Targeted order of each tech (None = not a targeted order).
    tech_order: Vec<Option<u8>>,
    pub orders: Orders,
    pub units: Units,
    pub techs: Techs,
}

fn order(name: &str) -> Result<u8, String> {
    order_id(name).and_then(|v| u8::try_from(v).ok()).ok_or_else(|| format!("unknown BWAPI order {name}"))
}

fn unit(name: &str) -> Result<u16, String> {
    TypeTables::get().unit_by_name(name).map(|u| u.id as u16).ok_or_else(|| format!("unknown BWAPI unit type {name}"))
}

fn tech(name: &str) -> Result<u16, String> {
    TypeTables::get().techs.iter().find(|t| t.name == name).map(|t| t.id as u16).ok_or_else(|| format!("unknown BWAPI tech {name}"))
}

fn effect_kind(name: &str) -> Result<EffectKind, String> {
    Ok(match name {
        "PsionicStorm" => EffectKind::PsionicStorm,
        "DarkSwarm" => EffectKind::DarkSwarm,
        "DisruptionWeb" => EffectKind::DisruptionWeb,
        "EmpShockwave" => EffectKind::EmpShockwave,
        "Plague" => EffectKind::Plague,
        "Ensnare" => EffectKind::Ensnare,
        "Irradiate" => EffectKind::Irradiate,
        _ => return Err(format!("unknown effect kind {name}")),
    })
}

fn check_version(file: &str, v: &str) -> Result<(), String> {
    if v != VERSION { Err(format!("{file}: version {v:?}, the bridge supports {VERSION:?} (013-R1)")) } else { Ok(()) }
}

impl Tables {
    pub fn parse() -> Result<Self, String> {
        let o: OrdersFile = toml::from_str(ORDERS_TOML).map_err(|e| format!("orders.toml: {e}"))?;
        check_version("orders.toml", &o.version)?;
        let mut order_map = [0u8; 256];
        for (i, v) in order_map.iter_mut().enumerate() {
            *v = i as u8;
        }
        for (from, to) in &o.overrides {
            order_map[order(from)? as usize] = order(to)?;
        }
        let e: EffectsFile = toml::from_str(EFFECTS_TOML).map_err(|e| format!("effects.toml: {e}"))?;
        check_version("effects.toml", &e.version)?;
        let kinds = |rows: &[(u16, String)]| rows.iter().map(|(id, k)| Ok((*id, effect_kind(k)?))).collect::<Result<HashMap<_, _>, String>>();
        let c: CommandsFile = toml::from_str(COMMANDS_TOML).map_err(|e| format!("commands.toml: {e}"))?;
        check_version("commands.toml", &c.version)?;
        let tech_order = c.tech_orders.iter().map(|n| if n == "None" { Ok(None) } else { order(n).map(Some) }).collect::<Result<Vec<_>, String>>()?;
        let orders = Orders {
            die: order("Die")?,
            stop: order("Stop")?,
            guard: order("Guard")?,
            player_guard: order("PlayerGuard")?,
            move_: order("Move")?,
            follow: order("Follow")?,
            attack1: order("Attack1")?,
            move_to_attack: order(&o.move_to_attack)?,
            attack_move: order("AttackMove")?,
            tower_attack: order("TowerAttack")?,
            carrier_attack: order("CarrierAttack")?,
            reaver_attack: order("ReaverAttack")?,
            patrol: order("Patrol")?,
            harvest1: order("Harvest1")?,
            harvest2: order("Harvest2")?,
            repair: order("Repair")?,
            pickup_bunker: order("PickupBunker")?,
            pickup_transport: order("PickupTransport")?,
            move_unload: order("MoveUnload")?,
            rally_point_tile: order("RallyPointTile")?,
            rally_point_unit: order("RallyPointUnit")?,
            place_building: order("PlaceBuilding")?,
            place_addon: order("PlaceAddon")?,
            building_land: order("BuildingLand")?,
            heal_move: order("HealMove")?,
            research_tech: order("ResearchTech")?,
            upgrade: order("Upgrade")?,
            train: order("Train")?,
            build_addon: order("BuildAddon")?,
            constructing_building: order("ConstructingBuilding")?,
            zerg_birth: order("ZergBirth")?,
            zerg_building_morph: order("ZergBuildingMorph")?,
            zerg_unit_morph: order("ZergUnitMorph")?,
            incomplete_morphing: order("IncompleteMorphing")?,
            incomplete_building: order("IncompleteBuilding")?,
            incomplete_warping: order("IncompleteWarping")?,
            nuke_paint: order("NukePaint")?,
            nuke_track: order("NukeTrack")?,
            reset_collision: order("ResetCollision")?,
            wait_for_minerals: order("WaitForMinerals")?,
            mining_minerals: order("MiningMinerals")?,
            return_minerals: order("ReturnMinerals")?,
            wait_for_gas: order("WaitForGas")?,
            harvest_gas: order("HarvestGas")?,
            return_gas: order("ReturnGas")?,
            idle: [
                order("PlayerGuard")?,
                order("Guard")?,
                order("Stop")?,
                order("PickupIdle")?,
                order("Nothing")?,
                order("Medic")?,
                order("Carrier")?,
                order("Reaver")?,
                order("Critter")?,
                order("Neutral")?,
                order("TowerGuard")?,
                order("Burrowed")?,
                order("NukeTrain")?,
                order("Larva")?,
            ],
            constructing: [
                order("ConstructingBuilding")?,
                order("PlaceBuilding")?,
                order("DroneBuild")?,
                order("DroneStartBuild")?,
                order("DroneLand")?,
                order("PlaceProtossBuilding")?,
                order("CreateProtossBuilding")?,
                order("IncompleteBuilding")?,
                order("IncompleteWarping")?,
                order("IncompleteMorphing")?,
                order("BuildNydusExit")?,
            ],
            gather_minerals: [
                order("Harvest1")?,
                order("Harvest2")?,
                order("MoveToMinerals")?,
                order("WaitForMinerals")?,
                order("MiningMinerals")?,
                order("ReturnMinerals")?,
                order("ResetCollision")?,
            ],
            gather_gas: [
                order("Harvest1")?,
                order("Harvest2")?,
                order("MoveToGas")?,
                order("WaitForGas")?,
                order("HarvestGas")?,
                order("ReturnGas")?,
                order("ResetCollision")?,
            ],
        };
        let units = Units {
            mineral_field: unit("Resource_Mineral_Field")?,
            mineral_field_2: unit("Resource_Mineral_Field_Type_2")?,
            mineral_field_3: unit("Resource_Mineral_Field_Type_3")?,
            vespene_geyser: unit("Resource_Vespene_Geyser")?,
            medic: unit("Terran_Medic")?,
            bunker: unit("Terran_Bunker")?,
            dropship: unit("Terran_Dropship")?,
            shuttle: unit("Protoss_Shuttle")?,
            overlord: unit("Zerg_Overlord")?,
            carrier: unit("Protoss_Carrier")?,
            reaver: unit("Protoss_Reaver")?,
            interceptor: unit("Protoss_Interceptor")?,
            scarab: unit("Protoss_Scarab")?,
            spider_mine: unit("Terran_Vulture_Spider_Mine")?,
            scanner_sweep: unit("Spell_Scanner_Sweep")?,
            nuclear_silo: unit("Terran_Nuclear_Silo")?,
            nuclear_missile: unit("Terran_Nuclear_Missile")?,
            siege_tank_siege: unit("Terran_Siege_Tank_Siege_Mode")?,
            hatchery: unit("Zerg_Hatchery")?,
            lair: unit("Zerg_Lair")?,
            hive: unit("Zerg_Hive")?,
            larva: unit("Zerg_Larva")?,
            mutalisk: unit("Zerg_Mutalisk")?,
            hydralisk: unit("Zerg_Hydralisk")?,
            spire: unit("Zerg_Spire")?,
            creep_colony: unit("Zerg_Creep_Colony")?,
            refineries: [unit("Terran_Refinery")?, unit("Protoss_Assimilator")?, unit("Zerg_Extractor")?],
            hero_duke_siege: unit("Hero_Edmund_Duke_Siege_Mode")?,
            vultures: [unit("Terran_Vulture")?, unit("Hero_Jim_Raynor_Vulture")?],
            ghost: unit("Terran_Ghost")?,
            none: unit("None")?,
        };
        let techs = Techs {
            stim_packs: tech("Stim_Packs")?,
            tank_siege_mode: tech("Tank_Siege_Mode")?,
            personnel_cloaking: tech("Personnel_Cloaking")?,
            cloaking_field: tech("Cloaking_Field")?,
            burrowing: tech("Burrowing")?,
            archon_warp: tech("Archon_Warp")?,
            dark_archon_meld: tech("Dark_Archon_Meld")?,
            healing: tech("Healing")?,
        };
        Ok(Self { order_map, bullets: kinds(&e.bullets)?, spell_units: kinds(&e.spell_units)?, op: c.opcodes, tech_order, orders, units, techs })
    }

    /// Shared parsed copy; `Err` disables the bridge (012-R6).
    pub fn get() -> Result<&'static Tables, String> {
        static T: OnceLock<Result<Tables, String>> = OnceLock::new();
        T.get_or_init(Self::parse).as_ref().map_err(Clone::clone)
    }

    /// BWAPI order of a raw game order (BWAPI `BWtoBWAPI_Order`).
    pub fn bwapi_order(&self, raw: u8) -> u8 {
        self.order_map[raw as usize]
    }

    pub fn bullet_effect(&self, flingy: u16) -> EffectKind {
        self.bullets.get(&flingy).copied().unwrap_or(EffectKind::Other(flingy))
    }

    pub fn spell_unit_effect(&self, unit_type: u16) -> Option<EffectKind> {
        self.spell_units.get(&unit_type).copied()
    }

    pub fn tech_order(&self, tech: u16) -> Option<u8> {
        self.tech_order.get(tech as usize).copied().flatten()
    }

    pub fn is_mineral_field(&self, t: u16) -> bool {
        t == self.units.mineral_field || t == self.units.mineral_field_2 || t == self.units.mineral_field_3
    }
}

/// BWAPI unit type data (`None` for ids outside the table).
pub fn unit_type(id: u16) -> Option<&'static UnitType> {
    TypeTables::get().unit(id)
}

/// BWAPI `WeaponTypes::None`.
pub fn has_weapon(id: i32) -> bool {
    TypeTables::get().weapon(id).is_some_and(|w| w.name != "None")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 013-R1 / 013-R2: every table parses, carries the version key, and names resolve.
    #[test]
    fn tables_parse_013_r1() {
        let t = Tables::get().unwrap();
        let o = &t.orders;
        assert_eq!(t.bwapi_order(order("Attack1").unwrap()), order("AttackUnit").unwrap());
        assert_eq!(t.bwapi_order(order("NukePaint").unwrap()), order("CastNuclearStrike").unwrap());
        assert_eq!(t.bwapi_order(o.move_), o.move_, "unlisted orders keep their id");
        assert_eq!((o.move_, o.follow, o.attack_move, o.harvest1), (6, 49, 14, 79));
        // 012-R17: the fog-ghost attack is the engine's MoveToAttack, raw order 9.
        assert_eq!(o.move_to_attack, 9);
        assert_eq!(t.bullet_effect(157), EffectKind::PsionicStorm);
        assert_eq!(t.bullet_effect(99), EffectKind::Other(99));
        assert_eq!(t.spell_unit_effect(202), Some(EffectKind::DarkSwarm));
        assert_eq!(t.tech_order(t.techs.healing), Some(order("MedicHeal").unwrap()));
        assert_eq!(t.tech_order(t.techs.stim_packs), None);
        assert_eq!((t.op.select, t.op.right_click, t.op.chat), (0x63, 0x60, 0x5c));
        assert_eq!(t.units.none, 228);
        assert!(!has_weapon(unit_type(t.units.medic).unwrap().ground_weapon));
        assert!(has_weapon(unit_type(0).unwrap().ground_weapon), "marine");
    }
}
