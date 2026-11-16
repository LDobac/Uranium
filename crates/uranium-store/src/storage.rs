use std::fs;
use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use anyhow::{Context, Result, anyhow, ensure};

mod lock;
mod manifest;
mod run;
mod status;
mod watch;
mod yields;

pub use manifest::{FileEntry, Manifest, sha256_file, verify_dir, write_atomic, pull_tree, open_dataset, dataset_dirs};
pub use lock::{LockInfo, TakeOver, LockStatus, Lease, LockBusy, peek_lock};
pub use run::{PushOutcome, Resume, PushStage, ckpt_name, Run, ckpt_id};
pub use watch::{LockView, ContentObserver, Watched, LockObserver, YieldObserver};
pub use yields::{YieldRequest, YIELD_EXPIRY, read_yield, remove_yield_if, yield_by_other, request_yield, clear_yield};
pub use status::{done_path, is_done, mark_done, log_device, DeviceSpan, device_spans, read_devices, WorkerStatus, write_worker, read_worker};

pub const MANIFEST_FILE: &str = "manifest.json";
pub const LATEST_FILE: &str = "LATEST";
const CKPT_PREFIX: &str = "ckpt-";

#[derive(Debug, Clone)]
pub struct StorageConfig {
    pub shared: PathBuf,
    pub cache: PathBuf,
    pub shared_local: bool,
    pub host: String,
    pub heartbeat_every: Duration,
    pub stale_after: Duration,
    pub keep_latest: usize,
}

impl StorageConfig {
    pub fn new(shared: impl Into<PathBuf>, cache: impl Into<PathBuf>, host: impl Into<String>) -> Self {
        Self {
            shared: shared.into(),
            cache: cache.into(),
            shared_local: false,
            host: host.into(),
            heartbeat_every: Duration::from_secs(60),
            stale_after: Duration::from_secs(600),
            keep_latest: 5,
        }
    }

    pub fn from_env() -> Result<Self> {
        let var = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
        let shared = var("URANIUM_SHARED_DIR").context("URANIUM_SHARED_DIR is not set (.env)")?;
        let cache = var("URANIUM_CACHE_DIR").context("URANIUM_CACHE_DIR is not set (.env)")?;
        let host = var("URANIUM_HOST_NAME").or_else(|| var("COMPUTERNAME")).or_else(|| var("HOSTNAME")).or_else(os_host_name).unwrap_or_else(|| "host".into());
        let mut cfg = Self::new(shared, cache, host);
        cfg.shared_local = var("URANIUM_SHARED_LOCAL").is_some_and(|v| v == "1");
        Ok(cfg)
    }
}

fn os_host_name() -> Option<String> {
    fs::read_to_string("/etc/hostname").ok().map(|s| s.trim().to_string()).filter(|s| !s.is_empty())
}

pub fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

pub fn check_name(name: &str) -> Result<()> {
    ensure!(!name.is_empty() && !name.starts_with('.'), "bad name {name:?}");
    ensure!(!name.chars().any(|c| "<>:\"/\\|?*".contains(c) || c.is_control()), "name {name:?} has a character SMB/Windows forbid (061-R9)");
    Ok(())
}

fn share_subdir(cfg: &StorageConfig, name: &str) -> Result<PathBuf> {
    let d = cfg.shared.join(name);
    match fs::create_dir(&d) {
        Ok(()) => Ok(d),
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => Ok(d),
        Err(e) => Err(anyhow!(e).context(format!("creating {}", d.display()))),
    }
}

pub fn probe_share(cfg: &StorageConfig) -> Result<()> {
    let d = cfg.shared.join("checkpoints");
    let m = fs::metadata(&d).with_context(|| format!("shared drive not reachable ({})", d.display()))?;
    ensure!(m.is_dir(), "{} is not a folder", d.display());
    fs::read_dir(&d).with_context(|| format!("listing {}", d.display()))?;
    Ok(())
}
