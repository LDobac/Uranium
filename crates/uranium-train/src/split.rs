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
