use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};

use crate::storage::check_name;

pub const STATE_FILE: &str = "train_state.json";
pub const LEAGUE_FILE: &str = "league.bin";
pub const CONFIG_DIR: &str = "config";
pub const AGENTS_DIR: &str = "agents";
pub const OPTIM_PREFIX: &str = "optim-";

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct TrainState {
    pub kind: String,
    pub step: u64,
    pub update: u64,
    pub policy_version: u64,
    pub rng: Vec<u64>,
    pub extra: serde_json::Value,
}

pub fn agent_dir(dir: &Path, name: &str) -> Result<PathBuf> {
    check_name(name)?;
    Ok(dir.join(AGENTS_DIR).join(name))
}

pub fn save_state(dir: &Path, s: &TrainState) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    std::fs::write(dir.join(STATE_FILE), serde_json::to_vec_pretty(s)?)?;
    Ok(())
}

pub fn load_state(dir: &Path) -> Result<Option<TrainState>> {
    let p = dir.join(STATE_FILE);
    match std::fs::read(&p) {
        Ok(b) => Ok(Some(serde_json::from_slice(&b).with_context(|| format!("reading {}", p.display()))?)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e.into()),
    }
}

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

pub fn copy_configs(dir: &Path, files: &[&Path]) -> Result<()> {
    let to = dir.join(CONFIG_DIR);
    std::fs::create_dir_all(&to)?;
    for f in files {
        let name = f.file_name().ok_or_else(|| anyhow!("{} has no file name", f.display()))?;
        std::fs::copy(f, to.join(name)).with_context(|| format!("copying {}", f.display()))?;
    }
    Ok(())
}
