//! The canonical BWAPI unit type ids the T0 bots need.

use uranium_proto::{Tile, UnitState};

// Terran
pub const MARINE: u16 = 0;
pub const SCV: u16 = 7;
pub const CC: u16 = 106;
pub const DEPOT: u16 = 109;
pub const REFINERY: u16 = 110;
pub const BARRACKS: u16 = 111;
// Zerg
pub const ZERGLING: u16 = 37;
pub const LARVA: u16 = 35;
pub const DRONE: u16 = 41;
pub const OVERLORD: u16 = 42;
pub const HATCHERY: u16 = 131;
pub const LAIR: u16 = 132;
pub const HIVE: u16 = 133;
pub const SPAWNING_POOL: u16 = 142;
// Protoss
pub const PROBE: u16 = 64;
pub const ZEALOT: u16 = 65;
pub const NEXUS: u16 = 154;
pub const PYLON: u16 = 156;
pub const GATEWAY: u16 = 160;
// Neutral
pub const MINERAL_FIELDS: [u16; 3] = [176, 177, 178];
pub const GEYSER: u16 = 188;

pub fn is_mineral(ty: u16) -> bool {
    MINERAL_FIELDS.contains(&ty)
}

pub fn is_resource(ty: u16) -> bool {
    is_mineral(ty) || ty == GEYSER
}

/// Terran 106–130, Zerg 131–152, Protoss 154–175 (BWAPI UnitTypes).
pub fn is_building(ty: u16) -> bool {
    (106..=175).contains(&ty) && ty != 153
}

pub fn is_resource_depot(ty: u16) -> bool {
    matches!(ty, CC | NEXUS | HATCHERY | LAIR | HIVE)
}

pub fn is_worker(ty: u16) -> bool {
    matches!(ty, SCV | PROBE | DRONE)
}

pub fn cost(ty: u16) -> i32 {
    match ty {
        DEPOT | PYLON | OVERLORD => 100,
        BARRACKS | GATEWAY => 150,
        SPAWNING_POOL => 200,
        CC | NEXUS => 400,
        HATCHERY => 300,
        REFINERY => 100,
        ZEALOT => 100,
        _ => 50,
    }
}

/// Footprint in tiles.
pub fn tile_size(ty: u16) -> (u16, u16) {
    match ty {
        CC | BARRACKS | NEXUS | GATEWAY | HATCHERY | LAIR | HIVE => (4, 3),
        DEPOT | SPAWNING_POOL => (3, 2),
        PYLON => (2, 2),
        REFINERY | GEYSER => (4, 2),
        t if is_mineral(t) => (2, 1),
        t if is_building(t) => (4, 3),
        _ => (1, 1),
    }
}

/// Top-left tile of a building/resource given its center position.
pub fn building_tile(u: &UnitState) -> Tile {
    let (w, h) = tile_size(u.unit_type);
    ((u.pos.0.saturating_sub(w * 16)) / 32, (u.pos.1.saturating_sub(h * 16)) / 32)
}
