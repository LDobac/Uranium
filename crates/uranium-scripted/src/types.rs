use uranium_proto::{Tile, UnitState};

pub const MARINE: u16 = 0;
pub const SCV: u16 = 7;
pub const DROPSHIP: u16 = 11;
pub const CC: u16 = 106;
pub const DEPOT: u16 = 109;
pub const REFINERY: u16 = 110;
pub const BARRACKS: u16 = 111;
pub const FACTORY: u16 = 113;
pub const STARPORT: u16 = 114;
pub const CONTROL_TOWER: u16 = 115;
pub const BUNKER: u16 = 125;
pub const SIEGE_TANK: u16 = 5;
pub const SIEGE_TANK_SIEGED: u16 = 30;
pub const MACHINE_SHOP: u16 = 120;
pub const TECH_TANK_SIEGE: u16 = 5;
pub const ZERGLING: u16 = 37;
pub const HYDRALISK: u16 = 38;
pub const MUTALISK: u16 = 43;
pub const LARVA: u16 = 35;
pub const DRONE: u16 = 41;
pub const OVERLORD: u16 = 42;
pub const HATCHERY: u16 = 131;
pub const LAIR: u16 = 132;
pub const HIVE: u16 = 133;
pub const SPIRE: u16 = 141;
pub const SPAWNING_POOL: u16 = 142;
pub const EXTRACTOR: u16 = 149;
pub const DARK_TEMPLAR: u16 = 61;
pub const PROBE: u16 = 64;
pub const ZEALOT: u16 = 65;
pub const NEXUS: u16 = 154;
pub const PYLON: u16 = 156;
pub const ASSIMILATOR: u16 = 157;
pub const GATEWAY: u16 = 160;
pub const CITADEL: u16 = 163;
pub const CYBERNETICS_CORE: u16 = 164;
pub const TEMPLAR_ARCHIVES: u16 = 165;
pub const MINERAL_FIELDS: [u16; 3] = [176, 177, 178];
pub const GEYSER: u16 = 188;

pub fn is_mineral(ty: u16) -> bool {
    MINERAL_FIELDS.contains(&ty)
}

pub fn is_resource(ty: u16) -> bool {
    is_mineral(ty) || ty == GEYSER
}

pub fn is_building(ty: u16) -> bool {
    (106..=175).contains(&ty) && ty != 153
}

pub fn is_resource_depot(ty: u16) -> bool {
    matches!(ty, CC | NEXUS | HATCHERY | LAIR | HIVE)
}

pub fn is_tank(ty: u16) -> bool {
    matches!(ty, SIEGE_TANK | SIEGE_TANK_SIEGED)
}

pub fn is_worker(ty: u16) -> bool {
    matches!(ty, SCV | PROBE | DRONE)
}

pub fn is_refinery(ty: u16) -> bool {
    matches!(ty, REFINERY | EXTRACTOR | ASSIMILATOR)
}

pub fn cost(ty: u16) -> i32 {
    match ty {
        DEPOT | PYLON | OVERLORD => 100,
        BARRACKS | GATEWAY => 150,
        SPAWNING_POOL => 200,
        CC | NEXUS => 400,
        HATCHERY => 300,
        REFINERY | ASSIMILATOR | BUNKER => 100,
        ZEALOT => 100,
        FACTORY | SPIRE | CYBERNETICS_CORE => 200,
        STARPORT | LAIR | CITADEL | TEMPLAR_ARCHIVES | SIEGE_TANK => 150,
        DROPSHIP | MUTALISK => 100,
        CONTROL_TOWER | MACHINE_SHOP => 50,
        DARK_TEMPLAR => 125,
        _ => 50,
    }
}

pub fn gas_cost(ty: u16) -> i32 {
    match ty {
        FACTORY | STARPORT | DROPSHIP | LAIR | MUTALISK | CITADEL | DARK_TEMPLAR | SIEGE_TANK => 100,
        SPIRE => 150,
        TEMPLAR_ARCHIVES => 200,
        CONTROL_TOWER | MACHINE_SHOP => 50,
        _ => 0,
    }
}

pub fn supply_cost(ty: u16) -> i32 {
    match ty {
        ZEALOT | DARK_TEMPLAR | MUTALISK | DROPSHIP | SIEGE_TANK => 4,
        _ => 2,
    }
}

pub fn tile_size(ty: u16) -> (u16, u16) {
    match ty {
        CC | BARRACKS | NEXUS | GATEWAY | HATCHERY | LAIR | HIVE => (4, 3),
        DEPOT | BUNKER | SPAWNING_POOL | CYBERNETICS_CORE | CITADEL | TEMPLAR_ARCHIVES => (3, 2),
        PYLON | SPIRE | CONTROL_TOWER => (2, 2),
        REFINERY | EXTRACTOR | ASSIMILATOR | GEYSER => (4, 2),
        t if is_mineral(t) => (2, 1),
        t if is_building(t) => (4, 3),
        _ => (1, 1),
    }
}

pub fn footprint(ty: u16) -> (u16, u16) {
    let (w, h) = tile_size(ty);
    match ty {
        FACTORY | STARPORT => (w + 2, h),
        _ => (w, h),
    }
}

pub fn addon_tile(parent: Tile) -> Tile {
    (parent.0 + 4, parent.1 + 1)
}

pub fn building_tile(u: &UnitState) -> Tile {
    let (w, h) = tile_size(u.unit_type);
    ((u.pos.0.saturating_sub(w * 16)) / 32, (u.pos.1.saturating_sub(h * 16)) / 32)
}

#[cfg(test)]
mod tests {
    use uranium_env::types::TypeTables;

    use super::*;

    #[test]
    fn unit_and_tech_ids_match_the_type_tables_013_r1() {
        let t = TypeTables::get();
        let units = [
            (MARINE, "Terran_Marine"),
            (SCV, "Terran_SCV"),
            (DROPSHIP, "Terran_Dropship"),
            (CC, "Terran_Command_Center"),
            (DEPOT, "Terran_Supply_Depot"),
            (REFINERY, "Terran_Refinery"),
            (BARRACKS, "Terran_Barracks"),
            (FACTORY, "Terran_Factory"),
            (STARPORT, "Terran_Starport"),
            (CONTROL_TOWER, "Terran_Control_Tower"),
            (BUNKER, "Terran_Bunker"),
            (SIEGE_TANK, "Terran_Siege_Tank_Tank_Mode"),
            (SIEGE_TANK_SIEGED, "Terran_Siege_Tank_Siege_Mode"),
            (MACHINE_SHOP, "Terran_Machine_Shop"),
            (ZERGLING, "Zerg_Zergling"),
            (HYDRALISK, "Zerg_Hydralisk"),
            (MUTALISK, "Zerg_Mutalisk"),
            (LARVA, "Zerg_Larva"),
            (DRONE, "Zerg_Drone"),
            (OVERLORD, "Zerg_Overlord"),
            (HATCHERY, "Zerg_Hatchery"),
            (LAIR, "Zerg_Lair"),
            (HIVE, "Zerg_Hive"),
            (SPIRE, "Zerg_Spire"),
            (SPAWNING_POOL, "Zerg_Spawning_Pool"),
            (EXTRACTOR, "Zerg_Extractor"),
            (DARK_TEMPLAR, "Protoss_Dark_Templar"),
            (PROBE, "Protoss_Probe"),
            (ZEALOT, "Protoss_Zealot"),
            (NEXUS, "Protoss_Nexus"),
            (PYLON, "Protoss_Pylon"),
            (ASSIMILATOR, "Protoss_Assimilator"),
            (GATEWAY, "Protoss_Gateway"),
            (CITADEL, "Protoss_Citadel_of_Adun"),
            (CYBERNETICS_CORE, "Protoss_Cybernetics_Core"),
            (TEMPLAR_ARCHIVES, "Protoss_Templar_Archives"),
            (MINERAL_FIELDS[0], "Resource_Mineral_Field"),
            (MINERAL_FIELDS[1], "Resource_Mineral_Field_Type_2"),
            (MINERAL_FIELDS[2], "Resource_Mineral_Field_Type_3"),
            (GEYSER, "Resource_Vespene_Geyser"),
        ];
        for (id, name) in units {
            assert_eq!(t.unit_by_name(name).map(|u| u.id as u16), Some(id), "{name}");
        }
        assert_eq!(t.techs.iter().find(|x| x.name == "Tank_Siege_Mode").map(|x| x.id as u16), Some(TECH_TANK_SIEGE));
    }

    #[test]
    fn is_building_matches_the_type_tables_013_r1() {
        let race_units = TypeTables::get().units.iter().filter(|u| ["Terran_", "Zerg_", "Protoss_"].iter().any(|p| u.name.starts_with(p)));
        for u in race_units {
            assert_eq!(is_building(u.id as u16), u.is_building(), "{}", u.name);
        }
    }
}
