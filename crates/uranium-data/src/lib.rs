//! uranium-data: the behaviour cloning data side (041) — shard sources, game order and data RNG
//! (041-R13), lanes and batches (041-R2), the resumable data position (061-R10), prefetch
//! (041-R16), compressed planes (041-R18) and the host collate into the 030 §1 batch layout. No
//! tensors: the same code feeds the trainer in Rust and in Python (`uranium._core`, AMB-000-4), so
//! the data order and fingerprints stay the same.

pub use uranium_store::elog;

pub mod bc;
pub mod collate;
pub mod data;
pub mod pack;
pub mod split;
pub mod table;
#[cfg(feature = "testkit")]
pub mod testkit;
