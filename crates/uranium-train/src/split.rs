//! BC data split and stage selection (041-R12, R13; 040-R5).
//!
//! The split is a pure function of the original replay hash, so both perspectives of a TvT game
//! land on the same side and every stage (S1, S2, ablation A/B) sees the same split.

use sha2::{Digest, Sha256};
use uranium_replay::shard::GameMeta;

/// 040-R15 target map families (S2).
pub const TARGET_FAMILIES: [&str; 2] = ["fighting_spirit", "circuit_breaker"];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Split {
    Train,
    Val,
    Test,
}

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

pub fn is_target_map(meta: &GameMeta) -> bool {
    TARGET_FAMILIES.contains(&meta.map_family.as_str())
}

/// Bucket 0..100 from the replay hash (first 64 bits of the hex SHA-256; other strings are hashed).
fn bucket(replay_sha256: &str) -> u64 {
    let v = replay_sha256.get(..16).and_then(|h| u64::from_str_radix(h, 16).ok()).unwrap_or_else(|| {
        let d = Sha256::digest(replay_sha256.as_bytes());
        u64::from_be_bytes(d[..8].try_into().unwrap())
    });
    v % 100
}

/// 041-R12: target maps train 80 / val 10 / test 10, other maps train 95 / val 5 (no test).
pub fn split_of(meta: &GameMeta) -> Split {
    let b = bucket(&meta.replay_sha256);
    if is_target_map(meta) {
        match b {
            0..80 => Split::Train,
            80..90 => Split::Val,
            _ => Split::Test,
        }
    } else if b < 95 {
        Split::Train
    } else {
        Split::Val
    }
}

pub fn selects(sel: Selection, meta: &GameMeta) -> bool {
    match sel {
        Selection::S1 => split_of(meta) == Split::Train,
        Selection::S2 => split_of(meta) == Split::Train && is_target_map(meta),
        Selection::Only(s) => split_of(meta) == s,
    }
}
