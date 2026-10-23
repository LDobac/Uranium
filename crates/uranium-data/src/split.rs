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
