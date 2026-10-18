//! BC data split (041-R12; 040-R5), a pure function of the original replay hash so both
//! perspectives of a TvT game land on the same side. Lives here so the dataset statistics
//! (040-R6) and `uranium-train` (re-export) use the same function.

use sha2::{Digest, Sha256};

use crate::shard::GameMeta;

/// 040-R15 target map families (S2).
pub const TARGET_FAMILIES: [&str; 2] = ["fighting_spirit", "circuit_breaker"];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Split {
    Train,
    Val,
    Test,
}

pub fn is_target_family(map_family: &str) -> bool {
    TARGET_FAMILIES.contains(&map_family)
}

pub fn is_target_map(meta: &GameMeta) -> bool {
    is_target_family(&meta.map_family)
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
pub fn split_for(replay_sha256: &str, map_family: &str) -> Split {
    let b = bucket(replay_sha256);
    if is_target_family(map_family) {
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

pub fn split_of(meta: &GameMeta) -> Split {
    split_for(&meta.replay_sha256, &meta.map_family)
}
