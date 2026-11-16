use std::fs;
use std::path::PathBuf;
use std::time::Duration;
use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use super::{StorageConfig, now_secs, share_subdir};
use super::manifest::write_atomic;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct YieldRequest {
    pub host: String,
    pub priority: i32,
    pub time: u64,
}

pub const YIELD_EXPIRY: Duration = Duration::from_secs(30 * 60);

fn yield_path(cfg: &StorageConfig, run_id: &str) -> PathBuf {
    cfg.shared.join("locks").join(format!("{run_id}.yield"))
}

pub fn read_yield(cfg: &StorageConfig, run_id: &str) -> Result<Option<YieldRequest>> {
    let p = yield_path(cfg, run_id);
    let bytes = match fs::read(&p) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(anyhow!(e).context(format!("reading {}", p.display()))),
    };
    Ok(Some(serde_json::from_slice(&bytes).with_context(|| format!("parsing {}", p.display()))?))
}

pub fn remove_yield_if(cfg: &StorageConfig, run_id: &str, seen: &YieldRequest) -> Result<bool> {
    if read_yield(cfg, run_id)?.as_ref() != Some(seen) {
        return Ok(false);
    }
    let p = yield_path(cfg, run_id);
    match fs::remove_file(&p) {
        Ok(()) => Ok(true),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(anyhow!(e).context(format!("removing {}", p.display()))),
    }
}

pub fn yield_by_other(cfg: &StorageConfig, run_id: &str) -> Option<YieldRequest> {
    read_yield(cfg, run_id).ok().flatten().filter(|y| y.host != cfg.host)
}

pub fn request_yield(cfg: &StorageConfig, run_id: &str, priority: i32) -> Result<()> {
    share_subdir(cfg, "locks")?;
    let req = YieldRequest { host: cfg.host.clone(), priority, time: now_secs() };
    write_atomic(&yield_path(cfg, run_id), &serde_json::to_vec_pretty(&req)?)
}

pub fn clear_yield(cfg: &StorageConfig, run_id: &str) -> Result<()> {
    let p = yield_path(cfg, run_id);
    let bytes = match fs::read(&p) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(anyhow!(e).context(format!("reading {}", p.display()))),
    };
    if serde_json::from_slice::<YieldRequest>(&bytes).is_ok_and(|y| y.host == cfg.host) {
        match fs::remove_file(&p) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(anyhow!(e).context(format!("removing {}", p.display()))),
        }
    }
    Ok(())
}
