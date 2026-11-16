use std::collections::BTreeSet;

use serde::Serialize;
use uranium_proto::*;

use super::{MapHints, ReactParams, RushRule, depot_center};
use crate::util::dist2;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Branch {
    #[default]
    Macro,
    Defence,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct ReactStats {
    pub main_frame: Option<u32>,
    pub rush_frame: Option<u32>,
    pub rush_rule: Option<RushRule>,
    pub defences: u32,
    pub macro_frame: Option<u32>,
    pub expand_frame: Option<u32>,
    pub branch: Branch,
    pub pool_start_frame: Option<u32>,
    pub pool_seen_frame: Option<u32>,
    pub lings_max: u32,
    pub contact_frame: Option<u32>,
    pub bunker_frame: Option<u32>,
    pub rule_frames: std::collections::BTreeMap<RushRule, u32>,
}

pub(crate) struct ReactState {
    pub p: ReactParams,
    pub branch: Branch,
    pub chokes: Vec<Tile>,
    pub natural: Option<Tile>,
    pub enemy_main: Option<Tile>,
    pub main_site: Option<Tile>,
    pub def_base: Option<Pos>,
    pub main_path: Option<Vec<Tile>>,
    pub last_contact: u32,
    pub pulled: Vec<UnitId>,
    pub repairers: Vec<UnitId>,
    pub holding: BTreeSet<UnitId>,
    pub marines_attack: bool,
    pub extractor_seen: bool,
    pub harass: Option<Harass>,
    pub orphan_since: std::collections::BTreeMap<UnitId, u32>,
    pub resumers: std::collections::BTreeMap<UnitId, (UnitId, u32)>,
    pub last_marines: Option<u32>,
    pub last_pull: Option<u32>,
    pub last_repair: Option<u32>,
    pub scout_out: bool,
    pub stats: ReactStats,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Harass {
    pub guards: Vec<UnitId>,
    pub target: UnitId,
    pub builder: UnitId,
    pub at: Pos,
    pub last_order: u32,
}

impl ReactState {
    pub fn new(p: ReactParams, hints: &MapHints, home: Option<Pos>, starts: &[Tile]) -> Self {
        let natural = home.and_then(|h| {
            hints
                .base_locs
                .iter()
                .copied()
                .filter(|b| !starts.iter().any(|s| (b.0 as i64 - s.0 as i64).abs() <= 4 && (b.1 as i64 - s.1 as i64).abs() <= 4))
                .min_by_key(|b| (dist2(depot_center(*b), h), *b))
        });
        Self {
            p,
            branch: Branch::Macro,
            chokes: hints.chokes.clone(),
            natural,
            enemy_main: None,
            main_site: None,
            def_base: None,
            main_path: None,
            last_contact: 0,
            pulled: Vec::new(),
            repairers: Vec::new(),
            holding: BTreeSet::new(),
            marines_attack: false,
            extractor_seen: false,
            harass: None,
            orphan_since: Default::default(),
            resumers: Default::default(),
            last_marines: None,
            last_pull: None,
            last_repair: None,
            scout_out: false,
            stats: ReactStats::default(),
        }
    }

    fn due(last: Option<u32>, frame: u32, every: u32) -> bool {
        last.is_none_or(|l| frame >= l + every)
    }

    pub(crate) fn due_on(aligned: bool, last: Option<u32>, frame: u32, every: u32) -> bool {
        if aligned {
            frame % every < crate::brain::STEP_FRAMES && last.is_none_or(|l| frame / every != l / every)
        } else {
            Self::due(last, frame, every)
        }
    }
}
