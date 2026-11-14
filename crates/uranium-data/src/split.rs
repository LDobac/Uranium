//! BC data split and stage selection (041-R12, R13; 040-R5).
//!
//! The split is a pure function of the original replay hash, so both perspectives of a TvT game
//! land on the same side and every stage (S1, S2, ablation A/B) sees the same split. The split
//! itself lives in `uranium_replay::split` so the dataset statistics (040-R6) share it.

use uranium_replay::shard::GameMeta;
pub use uranium_replay::split::{Split, TARGET_FAMILIES, is_target_map, split_of};

/// Games of one split and stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Selection {
    /// S1 pre-training: train split of every source and map (041-R13).
    S1,
    /// S2 fine-tuning: train split of the target maps only (041-R13).
    S2,
    /// One split, all maps (evaluation, diagnostics).
    Only(Split),
}

pub fn selects(sel: Selection, meta: &GameMeta) -> bool {
    match sel {
        Selection::S1 => split_of(meta) == Split::Train,
        Selection::S2 => split_of(meta) == Split::Train && is_target_map(meta),
        Selection::Only(s) => split_of(meta) == s,
    }
}

/// 041-R21 game subsample (BC config `[data.subsample]`): keeps a `fraction` share of the games
/// of every source except `keep_sources`, which are kept whole (the teacher games of the
/// fine-tune). Per game: [`crate::data::game_unit`] of the replay SHA-256 and the data seed, so
/// both perspectives of a TvT game are kept or dropped together and the subset is the same on
/// every PC and on resume. Applied after the stage selection, so it never moves a game between
/// the train / val / test splits (041-R12).
/// Unknown keys are an error (a typo such as `fraction_by_sources` would otherwise silently train
/// another mix, 044-R5).
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Subsample {
    /// Share of games kept, in (0, 1].
    pub fraction: f64,
    /// Sources (`GameMeta.src`, 040-R14) kept whole.
    #[serde(default)]
    pub keep_sources: Vec<String>,
    /// 041-R26: the share kept per enemy race (`"T"`, `"P"`, `"Z"`: the race after the `v` of
    /// `GameMeta.matchup`) for the games outside `keep_sources`; a race not listed keeps
    /// `fraction`. Same per-game hash rule, so both TvT perspectives stay together. Absent =
    /// `fraction` for every game and the B-3 fingerprint (`bc_mix_*`, `bc_dagger_*` included) from
    /// before the key.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fraction_by_race: Option<std::collections::BTreeMap<String, f64>>,
    /// 044-R5 ③ / 041-R31: the share kept per source (`GameMeta.src`, e.g. `distill`) of a source
    /// that is neither the teacher (`keep_sources`) nor human data. A source listed here keeps
    /// this share of its games by the same per-game hash rule, in place of `fraction` /
    /// `fraction_by_race`, and is left out of the human part of the mix ([`crate::mix`]). Absent =
    /// the 041-R26 subsample and the B-3 fingerprint from before the key (044-AC3).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fraction_by_source: Option<std::collections::BTreeMap<String, f64>>,
    /// 044-R8 / 022-R12 (AMB-022-7): the share kept per enemy race (`"T"`, `"P"`, `"Z"`) of the
    /// `keep_sources` games (teacher `botgen` + `dagger`), so the races inside the teacher + dagger
    /// share are Zerg : Protoss : Terran = 50 : 30 : 20 (`uranium-train mix-fractions
    /// --teacher-races`). Same per-game hash rule as the other keys; a race not listed keeps its
    /// teacher games whole. Absent = every teacher game kept (041-R21) and the B-3 fingerprint from
    /// before the key.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub keep_fraction_by_race: Option<std::collections::BTreeMap<String, f64>>,
}

/// 041-R26 races of `Subsample::fraction_by_race`.
pub const SUBSAMPLE_RACES: [&str; 3] = ["T", "P", "Z"];

/// The enemy race letter of a matchup ("TvZ" -> "Z"), from the perspective player's side.
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
        if let Some(m) = &self.keep_fraction_by_race {
            anyhow::ensure!(!self.keep_sources.is_empty(), "[data.subsample] keep_fraction_by_race without keep_sources (044-R8)");
            for (r, f) in m {
                anyhow::ensure!(SUBSAMPLE_RACES.contains(&r.as_str()), "[data.subsample] keep_fraction_by_race key {r:?} must be one of {SUBSAMPLE_RACES:?} (044-R8)");
                anyhow::ensure!(*f > 0.0 && *f <= 1.0, "[data.subsample] keep_fraction_by_race {r} must be in (0, 1], got {f} (044-R8)");
            }
        }
        Ok(())
    }

    /// The `fraction_by_source` sources (044-R5: neither teacher nor human), sorted.
    pub fn share_sources(&self) -> Vec<String> {
        self.fraction_by_source.iter().flatten().map(|(s, _)| s.clone()).collect()
    }

    /// The share kept of `meta`'s games outside `keep_sources` (041-R21, R26; 044-R5 per source).
    pub fn fraction_of(&self, meta: &GameMeta) -> f64 {
        if let Some(f) = self.fraction_by_source.as_ref().and_then(|m| m.get(&meta.src)) {
            return *f;
        }
        self.fraction_by_race.as_ref().and_then(|m| enemy_race(&meta.matchup).and_then(|r| m.get(r))).copied().unwrap_or(self.fraction)
    }

    /// The share kept of `meta`'s game when its source is in `keep_sources`: 1 (kept whole), or the
    /// enemy race's `keep_fraction_by_race` (044-R8).
    pub fn keep_fraction_of(&self, meta: &GameMeta) -> f64 {
        self.keep_fraction_by_race.as_ref().and_then(|m| enemy_race(&meta.matchup).and_then(|r| m.get(r))).copied().unwrap_or(1.0)
    }

    pub fn keeps(&self, meta: &GameMeta, seed: u64) -> bool {
        if self.keep_sources.contains(&meta.src) {
            // 041-R21: kept whole without `keep_fraction_by_race` (no hash drawn, as before the key).
            return self.keep_fraction_by_race.is_none() || crate::data::game_unit(&meta.replay_sha256, seed, crate::data::SUBSAMPLE_SALT) < self.keep_fraction_of(meta);
        }
        crate::data::game_unit(&meta.replay_sha256, seed, crate::data::SUBSAMPLE_SALT) < self.fraction_of(meta)
    }
}

/// Per source: games and steps without a privileged summary (E-10: the value head then sees all
/// zeros), and games per `result_source` (040-R8; `none` = value weight 0, 041-R7 / A-3).
#[derive(serde::Serialize, Debug, Default, Clone, PartialEq)]
pub struct SourceStats {
    pub src: String,
    pub games: usize,
    pub steps: u64,
    /// Games with at least one step without a privileged summary.
    pub games_privileged_none: usize,
    pub steps_privileged_none: u64,
    pub games_privileged_none_ratio: f64,
    pub steps_privileged_none_ratio: f64,
    pub result_source: std::collections::BTreeMap<String, usize>,
}

/// `uranium-train split-stats` output (041-R12, E-10).
#[derive(serde::Serialize, Debug, Default)]
pub struct SplitStats {
    /// Games per (source, map group, split).
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
