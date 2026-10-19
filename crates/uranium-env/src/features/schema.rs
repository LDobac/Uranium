//! 030 §1 tensor contract on the host side. `ObsTensors` holds model inputs only;
//! privileged data has its own type (`PrivilegedTensors`, 030-R9, 030-AC5).

use crate::actions::{CAPABILITIES, LOC_CLASSES};
use crate::types::TypeTables;

/// 030-R5 / 032-R6: bump when any feature meaning, order or normalisation changes.
/// 2: channel 12 is last_seen_age (AMB-030-1).
/// 3: unit_cap gains can_right_click_unit; can_rally without completion, maelstrom disables
///    (034, AMB-041-1).
/// 4 (2026-10-19, BC pre-training review): resources out of vision are remembered entities
///    (030-R2 exception, E-3) and seen ones stay targets; hazard discs use each effect's radius
///    (030 §3, E-1); flipped `entity_cell` from the flipped pixel (030-R4, B-6); disabled units
///    lose every selection class (034-R6, B-11); Random enemy race unknown until seen (030-R11,
///    E-6); `Masks::is_building` for the single-building rule (034-R7, E-4).
/// 5 (2026-10-19, action audit F7): action schema 2 (`Unload_Unit`, A = 101) and unit_cap gains
///    `loaded_unit` (034).
/// 6 (2026-10-19, 액션 감사 F1·F5·F6·F11·F15·F16, 사용자 결정): unit_cap gains can_stop and
///    can_unload_pos (K 48 -> 50); target classes drop units in production and loaded units;
///    can_attack drops the bunker; constructing SCVs and siege transitions lose every selection
///    class but can_stop; Build_* open only with a `pick_worker` candidate.
pub const FEATURE_SCHEMA_VERSION: u32 = 6;

/// 030-R6: spatial side in build tiles.
pub const MAP: usize = 128;
pub const CELLS: usize = MAP * MAP;
/// 030-R7
pub const N_MAX: usize = 512;
/// 030 §1 entity_cat columns.
pub const ENTITY_CAT: usize = 4;
/// 030 §2 width (6+34+3+3+2+12+10+2+2+2).
pub const ENTITY_CONT: usize = 76;
/// 030 §3 channel count.
pub const SPATIAL: usize = 26;
/// 030 §5
pub const Z_BUILD: usize = 20;
/// 031 delay buckets.
pub const DELAY_BUCKETS: [u32; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32];
/// BWAPI order vocabulary used by the model (030 §1: 190). `Unknown` (190) maps to `None` (189).
pub const ORDERS: usize = 190;
pub const ORDER_NONE: u16 = 189;

/// Owner column of `entity_cat` (030 §1).
pub const OWNER_SELF: i32 = 0;
pub const OWNER_ENEMY: i32 = 1;
pub const OWNER_NEUTRAL: i32 = 2;

/// enemy_race column of `scalar_cat` (030 §1: T / P / Z / unknown).
pub const RACE_TERRAN: i32 = 0;
pub const RACE_PROTOSS: i32 = 1;
pub const RACE_ZERG: i32 = 2;
pub const RACE_UNKNOWN: i32 = 3;

/// 030 §3 channel indices.
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
    /// Time since the tile was last visible (scouting record, AMB-030-1).
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

/// Dimensions that depend on the type tables (030 §4, §5).
#[derive(Debug, Clone, PartialEq)]
pub struct FeatureDims {
    /// Terran types counted for own units and used by z_units (U).
    pub terran_types: Vec<u16>,
    /// All-race types whose maximum simultaneous observation is tracked.
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

    /// 030 §4 layout: time 2, resources 5, upgrade levels 63, tech done 47, researching 47,
    /// upgrading 63, own completed / in production per Terran type, observed enemy maxima,
    /// previous action one-hot A + delay bucket 16 + failed 1.
    pub fn scalar(&self) -> usize {
        2 + 5 + 63 + 47 + 47 + 63 + 2 * self.terran_types.len() + self.observed_types.len() + self.actions + DELAY_BUCKETS.len() + 1
    }

    /// `PrivilegedTensors` width: enemy resources 5, upgrades 63, tech 47, completed and in
    /// production counts per observed type.
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

/// Model inputs for one game at one decision step (030 §1, unbatched).
#[derive(Debug, Clone, PartialEq)]
pub struct ObsTensors {
    /// Number of entities n (<= N_MAX).
    pub n: usize,
    /// [n,4] unit_type, owner, order, last_action_type.
    pub entity_cat: Vec<i32>,
    /// [n,76]
    pub entity_cont: Vec<f32>,
    /// [n] y*128+x
    pub entity_cell: Vec<i32>,
    /// [n,2] pixels
    pub entity_pos: Vec<f32>,
    /// [26,128,128]
    pub spatial: Vec<f32>,
    /// [F_s]
    pub scalar: Vec<f32>,
    /// enemy_race, map_family, start_loc
    pub scalar_cat: [i32; 3],
    pub z_build: [i32; Z_BUILD],
    /// [U]
    pub z_units: Vec<f32>,
    pub z_present: bool,
}

/// 030-R8 / 034: masks computed from the same state as the features.
#[derive(Debug, Clone, PartialEq)]
pub struct Masks {
    /// [A]
    pub type_mask: Vec<bool>,
    /// [n,K]
    pub unit_cap: Vec<bool>,
    /// [L,128,128]
    pub loc_class_mask: Vec<bool>,
    /// [n] the entity is a building (landed or lifted, bunker and turret included): the Selected
    /// head's single-building rule (034-R7, 032 §2.5 / §5, E-4).
    pub is_building: Vec<bool>,
}

/// Training-only inputs (030-R9). Never part of `ObsTensors`.
#[derive(Debug, Clone, PartialEq)]
pub struct PrivilegedTensors {
    pub summary: Vec<f32>,
}

/// Strategy condition z (030 §5).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Strategy {
    /// First 20 Build / Train / Research / Upgrade action ids.
    pub build: Vec<u16>,
    /// Target composition over `FeatureDims::terran_types` (fractions).
    pub units: Vec<f32>,
}
