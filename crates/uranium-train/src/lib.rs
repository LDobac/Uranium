//! uranium-train: behaviour cloning (spec 041), training state and shared storage (061), RL and
//! the league (spec 042). The data side (041) is in `uranium-data` and shared storage (061) in
//! `uranium-store` (AMB-000-4); both are re-exported here under their old paths.

pub use uranium_store::{Interrupted, STOP_CLEAN, elog, install_stop_handler, stop_requested, stopping};
pub use uranium_store::{disk, storage};
pub use uranium_data::{data, pack, split};

pub mod bc;
pub mod bc_eval;
pub mod bc_prof;
pub mod bc_run;
pub mod ckpt;
pub mod clip;
pub mod rl;
pub mod supervise;
pub mod worker;
