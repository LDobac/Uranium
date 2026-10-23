//! uranium-train: the Burn behaviour cloning trainer (spec 041, kept until the PyTorch parity
//! check, AMB-000-4), the run queue worker and watchdog (061, 060-R6) and the host data commands.
//! The data side (041) is in `uranium-data`, shared storage (061) in `uranium-store` and RL (042)
//! in `uranium-league`; data and storage are re-exported here under their old paths.

pub use uranium_store::{Interrupted, STOP_CLEAN, elog, install_stop_handler, stop_requested, stopping};
pub use uranium_store::{disk, storage};
pub use uranium_data::{data, pack, split};

pub mod bc;
pub mod bc_eval;
pub mod bc_prof;
pub mod bc_run;
pub mod ckpt;
pub mod clip;
pub mod supervise;
pub mod worker;
