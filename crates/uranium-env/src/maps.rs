//! Map-level data shared by replays, BC training and live games: map families (040-R15), the map
//! precompute folder and its lookup (031 맵 사전계산, AMB-031-4).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use uranium_proto::GameInfo;

use crate::features::mapgen;
use crate::features::{MapPrecompute, MapResource, PRECOMPUTE_VERSION};

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

/// `<dir>/<map_hash>.toml`, if present, readable and of the current [`PRECOMPUTE_VERSION`]. An
/// older file (no resource list, E-3) counts as missing; a replay build writes it again (031).
pub fn load_precompute(dir: &Path, map_hash: &str) -> Option<MapPrecompute> {
    let p: MapPrecompute = toml::from_str(&std::fs::read_to_string(dir.join(format!("{map_hash}.toml"))).ok()?).ok()?;
    (p.version == PRECOMPUTE_VERSION).then_some(p)
}

/// `MapResources::version` written and accepted (2 = resources carry their initial amount, E-3).
pub const RESOURCES_VERSION: u32 = 2;

fn resources_v1() -> u32 {
    1
}

/// Neutral resource placement of a map file (AMB-031-4), written by `uranium-replay mapgen`.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct MapResources {
    /// [`RESOURCES_VERSION`] of the writer; older files (no field) are rejected and must be
    /// written again with `uranium-replay mapgen`.
    #[serde(default = "resources_v1")]
    pub version: u32,
    /// Map file name the resources were read from (informational).
    pub source: String,
    /// SHA-256 of the map's CHK: the key replay builds use for the same map (031).
    pub chk_sha256: String,
    /// Type, centre pixel and initial amount from the CHK `UNIT` section.
    pub resources: Vec<MapResource>,
}

impl MapResources {
    pub fn new(source: String, chk_sha256: String, resources: Vec<MapResource>) -> Self {
        Self { version: RESOURCES_VERSION, source, chk_sha256, resources }
    }
}

/// `<dir>/<map file hash>.resources.toml` of the current [`RESOURCES_VERSION`].
pub fn load_resources(dir: &Path, map_hash: &str) -> Option<MapResources> {
    let r: MapResources = toml::from_str(&std::fs::read_to_string(resources_path(dir, map_hash)).ok()?).ok()?;
    (r.version == RESOURCES_VERSION).then_some(r)
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
    match load_resources(dir, &hash) {
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

    /// 031 / 030 §2 (E-3): a precompute or resource file of an older version (no resource amounts)
    /// counts as missing, so a rebuild writes it again; the current version round-trips.
    #[test]
    fn stale_precompute_versions_are_rejected_030_r2() {
        let dir = std::env::temp_dir().join(format!("uranium-maps-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("old.toml"), "base_locs = [[1, 2]]\nchokes = []\nwall_candidates = []\n").unwrap();
        assert_eq!(load_precompute(&dir, "old"), None);
        let res = vec![MapResource { unit_type: 176, pos: (100, 200), amount: 1500 }];
        let cur = MapPrecompute { version: PRECOMPUTE_VERSION, base_locs: vec![(1, 2)], resources: res.clone(), ..Default::default() };
        std::fs::write(dir.join("cur.toml"), toml::to_string(&cur).unwrap()).unwrap();
        assert_eq!(load_precompute(&dir, "cur"), Some(cur));
        std::fs::write(resources_path(&dir, "old"), "source = \"m.scx\"\nchk_sha256 = \"ab\"\nresources = [[176, [100, 200]]]\n").unwrap();
        assert_eq!(load_resources(&dir, "old"), None);
        let r = MapResources::new("m.scx".into(), "ab".into(), res);
        std::fs::write(resources_path(&dir, "cur"), toml::to_string(&r).unwrap()).unwrap();
        assert_eq!(load_resources(&dir, "cur"), Some(r));
        std::fs::remove_dir_all(&dir).ok();
    }
}
