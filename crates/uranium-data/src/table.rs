use anyhow::{Result, ensure};
use serde::Serialize;
use uranium_env::actions::{ActionTable, CAPABILITIES, LOC_CLASSES, TargetKind};

pub use uranium_env::actions::ACTION_SCHEMA_VERSION;
pub use uranium_env::features::FEATURE_SCHEMA_VERSION;
pub use uranium_env::features::DELAY_BUCKETS;
pub use uranium_env::features::N_MAX;

pub const TARGET_NONE: i32 = 0;
pub const TARGET_UNIT: i32 = 1;
pub const TARGET_LOC: i32 = 2;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ActionColumns {
    pub uses_select: Vec<i32>,
    pub max_select: Vec<i32>,
    pub select_cap: Vec<i32>,
    pub uses_queued: Vec<i32>,
    pub target_kind: Vec<i32>,
    pub target_cap: Vec<i32>,
    pub loc_class: Vec<i32>,
    pub self_target: Vec<i32>,
}

impl ActionColumns {
    pub fn from_table(table: &ActionTable) -> Result<Self> {
        let cap = |name: &str| CAPABILITIES.iter().position(|c| *c == name).map_or(0, |i| i as i32);
        let loc = |name: &str| LOC_CLASSES.iter().position(|c| *c == name).map_or(0, |i| i as i32);
        let mut c = Self { uses_select: vec![], max_select: vec![], select_cap: vec![], uses_queued: vec![], target_kind: vec![], target_cap: vec![], loc_class: vec![], self_target: vec![] };
        for a in &table.actions {
            ensure!(a.max_select as usize <= 12, "action {}: max_select > 12 (031-R3)", a.name);
            c.uses_select.push(a.uses_select as i32);
            c.max_select.push(if a.uses_select { a.max_select as i32 } else { 0 });
            c.select_cap.push(cap(&a.select_cap));
            c.uses_queued.push(a.uses_queued as i32);
            c.target_kind.push(match a.target {
                TargetKind::None => TARGET_NONE,
                TargetKind::Unit => TARGET_UNIT,
                TargetKind::Loc => TARGET_LOC,
            });
            c.target_cap.push(cap(&a.target_cap));
            c.loc_class.push(loc(&a.loc_class));
            c.self_target.push(a.self_target as i32);
        }
        Ok(c)
    }

    pub fn current() -> Result<Self> {
        Self::from_table(uranium_env::features::extract::action_table())
    }
}

pub fn feature_dims() -> &'static uranium_env::features::FeatureDims {
    uranium_env::features::extract::dims()
}
