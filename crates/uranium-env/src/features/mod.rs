//! Observation features (spec 030), masks (034) and the shared per-frame view used by the
//! executor (031). One code path for training, replays and SC:R (030 Purpose).

pub mod augment;
pub mod executor;
pub mod extract;
pub mod label;
pub mod map;
pub mod mapgen;
pub mod masks;
pub mod memory;
pub mod schema;
pub mod world;

pub use extract::{Observation, extract, privileged};
pub use map::{MapInfo, MapPrecompute, MapResource, PRECOMPUTE_VERSION};
pub use memory::AgentMemory;
pub use schema::*;
