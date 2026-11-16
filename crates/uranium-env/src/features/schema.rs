use crate::actions::{CAPABILITIES, LOC_CLASSES};
use crate::types::TypeTables;

pub const FEATURE_SCHEMA_VERSION: u32 = 7;

pub const FEATURE_SCHEMA_LOAD_COMPAT: &[u32] = &[6];

pub const MAP: usize = 128;
pub const CELLS: usize = MAP * MAP;
pub const N_MAX: usize = 512;
pub const ENTITY_CAT: usize = 4;
pub const ENTITY_CONT: usize = 76;
pub const SPATIAL: usize = 26;
pub const Z_BUILD: usize = 20;
pub const DELAY_BUCKETS: [u32; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32];
pub const ORDERS: usize = 190;
pub const ORDER_NONE: u16 = 189;

pub const OWNER_SELF: i32 = 0;
pub const OWNER_ENEMY: i32 = 1;
pub const OWNER_NEUTRAL: i32 = 2;

pub const RACE_TERRAN: i32 = 0;
pub const RACE_PROTOSS: i32 = 1;
pub const RACE_ZERG: i32 = 2;
pub const RACE_UNKNOWN: i32 = 3;

pub mod ch {
    pub const HEIGHT_LOW: usize = 0;
    pub const HEIGHT_HIGH: usize = 1;
    pub const HEIGHT_VERY_HIGH: usize = 2;
    pub const WALKABLE: usize = 3;
    pub const BUILDABLE: usize = 4;
    pub const START_LOC: usize = 5;
    pub const BASE_LOC: usize = 6;
    pub const CHOKE: usize = 7;
    pub const WALL_CANDIDATE: usize = 8;
    pub const VALID_MAP: usize = 9;
    pub const VISIBLE_NOW: usize = 10;
    pub const EXPLORED: usize = 11;
    pub const LAST_SEEN_AGE: usize = 12;
    pub const CREEP: usize = 13;
    pub const OWN_BUILDING: usize = 14;
    pub const ENEMY_BUILDING: usize = 15;
    pub const OWN_GROUND: usize = 16;
    pub const OWN_AIR: usize = 17;
    pub const ENEMY_GROUND: usize = 18;
    pub const ENEMY_AIR: usize = 19;
    pub const HAZARD_STORM: usize = 20;
    pub const HAZARD_SWARM: usize = 21;
    pub const HAZARD_WEB: usize = 22;
    pub const HAZARD_NUKE: usize = 23;
    pub const HAZARD_OTHER: usize = 24;
    pub const SCANNER: usize = 25;
}

#[derive(Debug, Clone, PartialEq)]
pub struct FeatureDims {
    pub terran_types: Vec<u16>,
    pub observed_types: Vec<u16>,
    pub actions: usize,
}

impl FeatureDims {
    pub fn new(types: &TypeTables, actions: usize) -> Self {
        Self { terran_types: types.terran_subset(), observed_types: types.all_races_subset(), actions }
    }

    pub fn z_units(&self) -> usize {
        self.terran_types.len()
    }

    pub fn scalar(&self) -> usize {
        2 + 5 + 63 + 47 + 47 + 63 + 2 * self.terran_types.len() + self.observed_types.len() + self.actions + DELAY_BUCKETS.len() + 1
    }

    pub fn privileged(&self) -> usize {
        5 + 63 + 47 + 2 * self.observed_types.len()
    }

    pub fn caps(&self) -> usize {
        CAPABILITIES.len()
    }

    pub fn loc_classes(&self) -> usize {
        LOC_CLASSES.len()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ObsTensors {
    pub n: usize,
    pub entity_cat: Vec<i32>,
    pub entity_cont: Vec<f32>,
    pub entity_cell: Vec<i32>,
    pub entity_pos: Vec<f32>,
    pub spatial: Vec<f32>,
    pub scalar: Vec<f32>,
    pub scalar_cat: [i32; 3],
    pub z_build: [i32; Z_BUILD],
    pub z_units: Vec<f32>,
    pub z_present: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Masks {
    pub type_mask: Vec<bool>,
    pub unit_cap: Vec<bool>,
    pub loc_class_mask: Vec<bool>,
    pub is_building: Vec<bool>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PrivilegedTensors {
    pub summary: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Strategy {
    pub build: Vec<u16>,
    pub units: Vec<f32>,
}
