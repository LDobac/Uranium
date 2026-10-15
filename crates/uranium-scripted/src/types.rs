//! The few canonical BWAPI unit type ids the T0 bots need.

use uranium_proto::{Tile, UnitState};

pub const MARINE: u16 = 0;
pub const SCV: u16 = 7;
pub const CC: u16 = 106;
pub const DEPOT: u16 = 109;
pub const REFINERY: u16 = 110;
pub const BARRACKS: u16 = 111;
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

pub fn cost(ty: u16) -> i32 {
    match ty {
        DEPOT => 100,
        BARRACKS => 150,
        CC => 400,
        REFINERY => 100,
        _ => 50,
    }
}

/// Footprint in tiles.
pub fn tile_size(ty: u16) -> (u16, u16) {
    match ty {
        CC | BARRACKS => (4, 3),
        DEPOT => (3, 2),
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
