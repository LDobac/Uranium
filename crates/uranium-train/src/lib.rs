//! uranium-train: the training host commands, the 060-R6 watchdog and the 061-R13 run queue
//! worker. Training itself is the PyTorch trainer (`python -m uranium bc | rl`, AMB-000-4); the
//! data side (041) is in `uranium-data`, shared storage (061) in `uranium-store` and RL (042) in
//! `uranium-league`. Data and storage are re-exported here under their old paths.

pub use uranium_store::{Interrupted, STOP_CLEAN, elog, install_stop_handler, stop_requested, stopping};
pub use uranium_store::{disk, storage};
pub use uranium_data::{bc, bc_run, data, pack, split};

pub mod supervise;
pub mod worker;
