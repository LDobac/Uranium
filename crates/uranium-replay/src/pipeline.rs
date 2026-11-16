use serde::Serialize;
use crate::label::Coverage;
use crate::header;
pub use uranium_env::maps::{MAP_FAMILIES_TOML, MapFamilies, maps_dir};

mod args;
mod driver;
mod excludes;
mod filter;
mod view;

pub use excludes::{Duplicates, MapExcludes};
pub use filter::{filter, filter_rules, distill_filter_reason, DISTILL_FILTER_PREFIX, invalid_label_filter};
pub(crate) use view::own_labels;
pub use view::{judge, ViewGame, decision_steps, step_labels, distill_steps, attach_memory, memory_log_path, cut_after, write_map_precompute};
pub use args::{BRANCH_SRC, DISTILL_SRC, check_build_args, check_games_arg, check_isolated_build, path_within, unique_stems};
pub use driver::{BuildOptions, ResumeDone, build};

pub const MIN_GAME_MS: u64 = 5 * 60 * 1000;
pub const S1_MIN_APM: u32 = 80;
pub const S2_MIN_APM: u32 = 150;
pub const MAX_MAP_TILES: u16 = 128;

pub const MAP_EXCLUDE_TOML: &str = include_str!("../../../config/map_exclude.toml");

pub const DUPLICATE_IN_SOURCE: &str = "duplicate file in this source (040-R20)";

#[derive(Debug, Clone, Serialize)]
pub struct View {
    pub perspective: u8,
    pub stage: &'static str,
    pub apm: u32,
    pub matchup: String,
    pub enemy_random: bool,
}

pub(crate) fn race_letter(r: u8) -> char {
    match r {
        header::RACE_ZERG => 'Z',
        header::RACE_TERRAN => 'T',
        header::RACE_PROTOSS => 'P',
        _ => 'R',
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum Outcome {
    Written {
        steps: usize,
        coverage: Coverage,
        result: i8,
        result_source: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        repeats_dropped: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        memory: Option<crate::relabel::MemoryCheck>,
    },
    Filtered { reason: String },
    Desync { reasons: Vec<String> },
    Review { reason: String },
    Unplayable { reason: String },
    Error { reason: String },
}

#[derive(Debug, Clone, Serialize)]
pub struct Record {
    pub index: usize,
    pub replay_sha256: String,
    pub src: String,
    pub map_family: String,
    pub map_name: String,
    pub map_hash: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recorded_winner: Option<&'static str>,
    pub view: Option<View>,
    pub labeler_version: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub until_frame: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_limit: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub relabel: Option<&'static str>,
    #[serde(flatten)]
    pub outcome: Outcome,
}

pub fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_families_classify_040_r15() {
        let f = MapFamilies::load().unwrap();
        assert_eq!(f.classify("| iCCup | Fighting Spirit", ""), "fighting_spirit");
        assert_eq!(f.classify("투혼 1.3", ""), "fighting_spirit");
        assert_eq!(f.classify("\u{3}Circuit Breakers \u{5}1.0", ""), "circuit_breaker");
        assert_eq!(f.classify("Python", ""), "other");
    }
}
