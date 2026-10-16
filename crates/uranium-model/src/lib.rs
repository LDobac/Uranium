//! Uranium policy / value network (spec 032) on Burn 0.22. Backend-agnostic: the `Device` the
//! module and tensors live on picks Flex (tests), Vulkan (BC250) or CUDA (032-R1).

pub mod batch;
pub mod checkpoint;
pub mod collate;
pub mod config;
pub mod core;
pub mod entity;
pub mod heads;
pub mod layers;
pub mod net;
pub mod scalar;
pub mod spatial;
pub mod synthetic;
pub mod value;

pub use batch::{ActionBatch, ActionColumns, ObsBatch, PrivBatch};
pub use config::UraniumNetConfig;
pub use core::{Core, GtrxlCore, LstmCore};
pub use net::{ActOutput, CoreInit, SeqOutput, UraniumNet};
