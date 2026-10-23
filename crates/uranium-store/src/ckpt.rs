//! Training state stored next to the weights (061-R10, 042-R1): step counters and RNG state, the
//! policy version, an opaque league blob, and copies of the config files in use (042-R9, 033-R4).
//! Weights and optimizer state are written by the trainer that owns them; a multi-agent RL
//! checkpoint keeps one weights folder per agent under `agents/<name>/`.
//!
//! Everything here is plain files, independent of the device and of the framework that wrote the
//! weights (042-R1, 061-AC5).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};

use crate::storage::check_name;

pub const STATE_FILE: &str = "train_state.json";
pub const LEAGUE_FILE: &str = "league.bin";
pub const CONFIG_DIR: &str = "config";
pub const AGENTS_DIR: &str = "agents";

/// Counters and RNG of a training process (061-R10).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct TrainState {
    /// "bc" or "rl".
    pub kind: String,
    /// Agent steps (RL) or updates (BC) so far; names the checkpoint (`ckpt-<step>`, 061-R4).
    pub step: u64,
    /// Optimizer updates so far.
    pub update: u64,
    /// Version of the weights the actors run (042-R4).
    pub policy_version: u64,
    /// RNG states, in an order the owner defines.
    pub rng: Vec<u64>,
    /// Owner-specific counters (games seen, invalid labels, lr scale, ...).
    pub extra: serde_json::Value,
}

/// Weights folder of one agent in a multi-agent checkpoint.
pub fn agent_dir(dir: &Path, name: &str) -> Result<PathBuf> {
    check_name(name)?;
    Ok(dir.join(AGENTS_DIR).join(name))
}

pub fn save_state(dir: &Path, s: &TrainState) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    std::fs::write(dir.join(STATE_FILE), serde_json::to_vec_pretty(s)?)?;
    Ok(())
}

/// `None` for a checkpoint written before this file existed (weights only).
pub fn load_state(dir: &Path) -> Result<Option<TrainState>> {
    let p = dir.join(STATE_FILE);
    match std::fs::read(&p) {
        Ok(b) => Ok(Some(serde_json::from_slice(&b).with_context(|| format!("reading {}", p.display()))?)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e.into()),
    }
}

/// League tables and snapshot list, serialised by their owner (042-R1).
pub fn save_league(dir: &Path, bytes: &[u8]) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    std::fs::write(dir.join(LEAGUE_FILE), bytes)?;
    Ok(())
}

pub fn load_league(dir: &Path) -> Result<Option<Vec<u8>>> {
    match std::fs::read(dir.join(LEAGUE_FILE)) {
        Ok(b) => Ok(Some(b)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e.into()),
    }
}

/// 042-R9 / 033-R4: copies of the config files a run used, under `config/<file name>`.
pub fn copy_configs(dir: &Path, files: &[&Path]) -> Result<()> {
    let to = dir.join(CONFIG_DIR);
    std::fs::create_dir_all(&to)?;
    for f in files {
        let name = f.file_name().ok_or_else(|| anyhow!("{} has no file name", f.display()))?;
        std::fs::copy(f, to.join(name)).with_context(|| format!("copying {}", f.display()))?;
    }
    Ok(())
}
