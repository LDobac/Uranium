//! Training state stored next to the weights (061-R10, 042-R1). The plain-file parts (step
//! counters, RNG, league blob, config copies) live in `uranium_store::ckpt`; this module adds the
//! Burn optimizer state per agent. Weights stay in `uranium_model::checkpoint` (032-R6).
//!
//! Everything is device independent (Burn records are host data), so a checkpoint saved on one
//! host resumes on another (042-R1, 061-AC5).

use std::path::Path;

use anyhow::{Result, anyhow};
use burn::optim::ModuleOptimizer;

pub use uranium_store::ckpt::*;
use uranium_store::storage::check_name;

pub fn optim_file(name: &str) -> String {
    format!("optim-{name}.bpk")
}

pub fn save_optimizer(dir: &Path, name: &str, optim: &ModuleOptimizer) -> Result<()> {
    check_name(name)?;
    std::fs::create_dir_all(dir)?;
    optim.save(dir.join(optim_file(name))).map_err(|e| anyhow!("saving optimizer {name}: {e:?}"))
}

/// Loads the state into `optim` (built with the same config). The state attaches to parameters
/// by id, which the weights file keeps (burnpack), so load the weights from the same checkpoint.
pub fn load_optimizer(dir: &Path, name: &str, optim: ModuleOptimizer) -> Result<ModuleOptimizer> {
    check_name(name)?;
    optim.load(dir.join(optim_file(name))).map_err(|e| anyhow!("loading optimizer {name}: {e:?}"))
}

pub fn has_optimizer(dir: &Path, name: &str) -> bool {
    dir.join(optim_file(name)).is_file()
}
