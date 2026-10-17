//! Map-level data shared by replays, BC training and live games: map families (040-R15), the map
//! precompute folder and its lookup (031 맵 사전계산, AMB-031-4).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use uranium_proto::{GameInfo, Pos};

use crate::features::MapPrecompute;
use crate::features::mapgen;

pub const MAP_FAMILIES_TOML: &str = include_str!("../../../config/map_families.toml");

#[derive(Deserialize)]
struct Family {
    name: String,
    name_patterns: Vec<String>,
    chk_sha256: Vec<String>,
}

#[derive(Deserialize)]
struct Families {
    family: Vec<Family>,
}

pub struct MapFamilies(Vec<Family>);

impl MapFamilies {
    pub fn load() -> Result<Self> {
        Ok(Self(toml::from_str::<Families>(MAP_FAMILIES_TOML).context("map_families.toml")?.family))
    }

    /// 040-R15: by CHK hash first, then by title pattern (control codes removed).
    pub fn classify(&self, map_name: &str, chk_sha256: &str) -> String {
        let title = clean_title(map_name);
        self.0
            .iter()
            .find(|f| f.chk_sha256.iter().any(|h| h == chk_sha256) || f.name_patterns.iter().any(|p| title.contains(&p.to_lowercase())))
            .map_or_else(|| "other".to_string(), |f| f.name.clone())
    }
}

/// Map title without color / control codes, lower case.
pub fn clean_title(map_name: &str) -> String {
    map_name.chars().filter(|c| !c.is_control()).collect::<String>().to_lowercase()
}

/// 040-R15 family -> `scalar_cat` map_family index (030).
pub fn family_index(family: &str) -> i32 {
    match family {
        "fighting_spirit" => 0,
        "circuit_breaker" => 1,
        _ => 2,
    }
}

/// Directory of `<map_hash>.toml` map precomputes (031): `URANIUM_MAPS_DIR`, default `assets/maps`.
pub fn maps_dir() -> PathBuf {
    std::env::var_os("URANIUM_MAPS_DIR").map_or_else(|| PathBuf::from("assets/maps"), PathBuf::from)
}

/// `<dir>/<map_hash>.toml`, if present and readable.
pub fn load_precompute(dir: &Path, map_hash: &str) -> Option<MapPrecompute> {
    toml::from_str(&std::fs::read_to_string(dir.join(format!("{map_hash}.toml"))).ok()?).ok()
}

/// Neutral resource placement of a map file (AMB-031-4), written by `uranium-replay mapgen`.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct MapResources {
    /// Map file name the resources were read from (informational).
    pub source: String,
    /// SHA-256 of the map's CHK: the key replay builds use for the same map (031).
    pub chk_sha256: String,
    /// (unit type, centre pixel) from the CHK `UNIT` section.
    pub resources: Vec<(u16, Pos)>,
}

pub fn resources_path(dir: &Path, map_hash: &str) -> PathBuf {
    dir.join(format!("{map_hash}.resources.toml"))
}

/// Where a live game's precompute came from (logged once per session).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrecomputeSource {
    /// `<map file hash>.toml`.
    Cached,
    /// `<CHK hash>.toml` written by a replay build of the same map (identical to BC's).
    ReplayBuild,
    /// Built at game start from `<map file hash>.resources.toml` + `GameInfo.static_map`.
    Resources,
    /// Neither file: the precompute channels stay 0 (031, AMB-031-4).
    Missing,
}

/// 031 / AMB-031-4: precompute for a live game, keyed by the map file hash in `GameInfo`. Uses the
/// same algorithm and inputs (static map + CHK resources) as the replay pipeline, so a target map
/// gets the channels BC was trained with.
pub fn live_precompute(dir: &Path, info: &GameInfo) -> (Option<MapPrecompute>, PrecomputeSource) {
    let hash: String = info.map_hash.iter().map(|b| format!("{b:02x}")).collect();
    if let Some(p) = load_precompute(dir, &hash) {
        return (Some(p), PrecomputeSource::Cached);
    }
    let res = std::fs::read_to_string(resources_path(dir, &hash)).ok().and_then(|t| toml::from_str::<MapResources>(&t).ok());
    match res {
        Some(r) => match load_precompute(dir, &r.chk_sha256) {
            Some(p) => (Some(p), PrecomputeSource::ReplayBuild),
            None => (Some(mapgen::precompute(&info.static_map, &r.resources)), PrecomputeSource::Resources),
        },
        None => (None, PrecomputeSource::Missing),
    }
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
        assert_eq!(family_index(&f.classify("Python", "")), 2);
    }
}
