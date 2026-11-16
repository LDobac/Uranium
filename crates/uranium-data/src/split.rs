use uranium_replay::shard::GameMeta;
pub use uranium_replay::split::{Split, TARGET_FAMILIES, is_target_map, split_of};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Selection {
    S1,
    S2,
    Only(Split),
}

pub fn selects(sel: Selection, meta: &GameMeta) -> bool {
    match sel {
        Selection::S1 => split_of(meta) == Split::Train,
        Selection::S2 => split_of(meta) == Split::Train && is_target_map(meta),
        Selection::Only(s) => split_of(meta) == s,
    }
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Subsample {
    pub fraction: f64,
    #[serde(default)]
    pub keep_sources: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fraction_by_race: Option<std::collections::BTreeMap<String, f64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fraction_by_source: Option<std::collections::BTreeMap<String, f64>>,
}

pub const SUBSAMPLE_RACES: [&str; 3] = ["T", "P", "Z"];

pub fn enemy_race(matchup: &str) -> Option<&str> {
    matchup.split_once('v').map(|(_, e)| e)
}

impl Subsample {
    pub fn check(&self) -> anyhow::Result<()> {
        anyhow::ensure!(self.fraction > 0.0 && self.fraction <= 1.0, "[data.subsample] fraction must be in (0, 1], got {} (041-R21)", self.fraction);
        for (r, f) in self.fraction_by_race.iter().flatten() {
            anyhow::ensure!(SUBSAMPLE_RACES.contains(&r.as_str()), "[data.subsample] fraction_by_race key {r:?} must be one of {SUBSAMPLE_RACES:?} (041-R26)");
            anyhow::ensure!(*f > 0.0 && *f <= 1.0, "[data.subsample] fraction_by_race {r} must be in (0, 1], got {f} (041-R26)");
        }
        for (s, f) in self.fraction_by_source.iter().flatten() {
            anyhow::ensure!(!self.keep_sources.contains(s), "[data.subsample] fraction_by_source key {s:?} is also in keep_sources (044-R5)");
            anyhow::ensure!(*f > 0.0 && *f <= 1.0, "[data.subsample] fraction_by_source {s} must be in (0, 1], got {f} (044-R5)");
        }
        Ok(())
    }

    pub fn share_sources(&self) -> Vec<String> {
        self.fraction_by_source.iter().flatten().map(|(s, _)| s.clone()).collect()
    }

    pub fn fraction_of(&self, meta: &GameMeta) -> f64 {
        if let Some(f) = self.fraction_by_source.as_ref().and_then(|m| m.get(&meta.src)) {
            return *f;
        }
        self.fraction_by_race.as_ref().and_then(|m| enemy_race(&meta.matchup).and_then(|r| m.get(r))).copied().unwrap_or(self.fraction)
    }

    pub fn keeps(&self, meta: &GameMeta, seed: u64) -> bool {
        self.keep_sources.iter().any(|s| *s == meta.src) || crate::data::game_unit(&meta.replay_sha256, seed, crate::data::SUBSAMPLE_SALT) < self.fraction_of(meta)
    }
}

#[derive(serde::Serialize, Debug, Default, Clone, PartialEq)]
pub struct SourceStats {
    pub src: String,
    pub games: usize,
    pub steps: u64,
    pub games_privileged_none: usize,
    pub steps_privileged_none: u64,
    pub games_privileged_none_ratio: f64,
    pub steps_privileged_none_ratio: f64,
    pub result_source: std::collections::BTreeMap<String, usize>,
}

#[derive(serde::Serialize, Debug, Default)]
pub struct SplitStats {
    pub splits: Vec<serde_json::Value>,
    pub sources: Vec<SourceStats>,
}

pub fn split_stats<'a>(games: impl IntoIterator<Item = &'a crate::data::GameSummary>) -> SplitStats {
    use std::collections::BTreeMap;
    let mut counts = BTreeMap::<(String, &str, Split), usize>::new();
    let mut by_src = BTreeMap::<String, SourceStats>::new();
    for g in games {
        let group = if is_target_map(&g.meta) { "target" } else { "other" };
        *counts.entry((g.meta.src.clone(), group, split_of(&g.meta))).or_default() += 1;
        let s = by_src.entry(g.meta.src.clone()).or_insert_with(|| SourceStats { src: g.meta.src.clone(), ..Default::default() });
        s.games += 1;
        s.steps += g.steps as u64;
        s.games_privileged_none += (g.privileged_none > 0) as usize;
        s.steps_privileged_none += g.privileged_none as u64;
        *s.result_source.entry(g.meta.result_source.clone()).or_default() += 1;
    }
    let sources = by_src
        .into_values()
        .map(|mut s| {
            s.games_privileged_none_ratio = s.games_privileged_none as f64 / s.games.max(1) as f64;
            s.steps_privileged_none_ratio = s.steps_privileged_none as f64 / s.steps.max(1) as f64;
            s
        })
        .collect();
    let splits = counts.into_iter().map(|((s, m, sp), n)| serde_json::json!({ "src": s, "maps": m, "split": sp, "games": n })).collect();
    SplitStats { splits, sources }
}
