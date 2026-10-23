//! uranium-league: reinforcement learning without the model (spec 042, AMB-000-4). The league
//! and curriculum, the trajectories and segment queues of learning sessions, the return targets
//! (the reference for the PyTorch learner), the host collate of segment batches and the run loop
//! (games, queues, weight pushes, snapshots, checkpoints, 061 locks) around a learner that lives
//! elsewhere ([`learner::RlLearners`]: the PyTorch learner through `uranium._core`).

pub use uranium_store::{Interrupted, ckpt, disk, elog, install_stop_handler, storage, stopping};

pub mod batch;
pub mod config;
pub mod league;
pub mod learner;
pub mod returns;
pub mod run;
#[cfg(any(test, feature = "testkit"))]
pub mod testkit;
pub mod trajectory;
