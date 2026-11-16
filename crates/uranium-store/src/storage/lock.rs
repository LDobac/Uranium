use std::fs;
use std::io::Write;
use std::path::PathBuf;
use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use super::{StorageConfig, now_secs, check_name};
use super::manifest::write_atomic;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct LockInfo {
    pub host: String,
    pub pid: u32,
    pub started: u64,
    pub heartbeat: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TakeOver {
    No,
    Force,
    IfUnchanged(LockInfo),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LockStatus {
    Held,
    Lost(LockInfo),
    Unreachable(String),
}

pub(super) fn take_lock(cfg: &StorageConfig, id: &str, take: &TakeOver) -> Result<(LockInfo, Option<LockInfo>)> {
    let locks = cfg.shared.join("locks");
    fs::create_dir_all(&locks).with_context(|| format!("creating {}", locks.display()))?;
    let now = now_secs();
    let me = LockInfo { host: cfg.host.clone(), pid: std::process::id(), started: now, heartbeat: now };
    let path = locks.join(format!("{id}.json"));
    match fs::OpenOptions::new().write(true).create_new(true).open(&path) {
        Ok(mut f) => {
            f.write_all(&serde_json::to_vec_pretty(&me)?)?;
            f.sync_all()?;
            return Ok((me, None));
        }
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(e) => return Err(anyhow!(e).context(format!("creating lock {}", path.display()))),
    }
    let bytes = fs::read(&path).with_context(|| format!("reading lock {}", path.display()))?;
    let Ok(held) = serde_json::from_slice::<LockInfo>(&bytes) else {
        return Err(anyhow::Error::new(LockBusy(format!("run {id}: lock {} is being written by another process (061-R5)", path.display()))));
    };
    if held.host != me.host {
        match take {
            TakeOver::No => {
                return Err(anyhow::Error::new(LockBusy(format!(
                    "run {id} is locked by {} (pid {}); a lock of another PC is taken only with --take-over (061-R5)",
                    held.host, held.pid
                ))));
            }
            TakeOver::Force => crate::elog!(
                "WARNING: --take-over: taking run {id} from {} (pid {}) now (061-R5). If that training is still alive it stops without a push at its next heartbeat, and checkpoints only it had (not pushed) may be lost (061-R11-4).",
                held.host, held.pid
            ),
            TakeOver::IfUnchanged(seen) if *seen == held => {
                crate::elog!("taking over run {id}: the lock of {} (pid {}) did not change for {} min (061-R5, R13)", held.host, held.pid, cfg.stale_after.as_secs() / 60)
            }
            TakeOver::IfUnchanged(_) => {
                return Err(anyhow::Error::new(LockBusy(format!("run {id}: the lock of {} changed since it was seen expired; not taken (061-R5)", held.host))));
            }
        }
    }
    write_atomic(&path, &serde_json::to_vec_pretty(&me)?)?;
    let now: LockInfo = fs::read(&path).map_err(anyhow::Error::from).and_then(|b| Ok(serde_json::from_slice(&b)?)).with_context(|| format!("reading back lock {}", path.display()))?;
    if !(now.host == me.host && now.pid == me.pid) {
        return Err(anyhow::Error::new(LockBusy(format!("run {id}: {} (pid {}) took the lock at the same moment (061-R5)", now.host, now.pid))));
    }
    Ok((me, Some(held)))
}

pub struct Lease {
    pub cfg: StorageConfig,
    pub id: String,
    me: LockInfo,
}

impl Lease {
    pub fn acquire(cfg: StorageConfig, id: &str, take: TakeOver) -> Result<Self> {
        check_name(id)?;
        check_name(&cfg.host)?;
        let (me, _) = take_lock(&cfg, id, &take)?;
        Ok(Self { cfg, id: id.to_string(), me })
    }

    fn path(&self) -> PathBuf {
        self.cfg.shared.join("locks").join(format!("{}.json", self.id))
    }

    fn read(&self) -> Result<LockInfo> {
        Ok(serde_json::from_slice(&fs::read(self.path())?)?)
    }

    pub fn heartbeat(&mut self) -> LockStatus {
        let held = match self.read() {
            Ok(l) => l,
            Err(e) => return LockStatus::Unreachable(format!("{e:#}")),
        };
        if !(held.host == self.me.host && held.pid == self.me.pid) {
            return LockStatus::Lost(held);
        }
        self.me.heartbeat = now_secs();
        match serde_json::to_vec_pretty(&self.me).map_err(anyhow::Error::from).and_then(|b| write_atomic(&self.path(), &b)) {
            Ok(()) => LockStatus::Held,
            Err(e) => LockStatus::Unreachable(format!("{e:#}")),
        }
    }

    pub fn release(self) -> Result<()> {
        if self.read().is_ok_and(|l| l.host == self.me.host && l.pid == self.me.pid) {
            fs::remove_file(self.path())?;
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct LockBusy(pub String);

impl std::fmt::Display for LockBusy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for LockBusy {}

pub fn peek_lock(cfg: &StorageConfig, run_id: &str) -> Result<Option<LockInfo>> {
    let path = cfg.shared.join("locks").join(format!("{run_id}.json"));
    match fs::read(&path) {
        Ok(bytes) => Ok(Some(serde_json::from_slice(&bytes).with_context(|| format!("parsing lock {}", path.display()))?)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(anyhow!(e).context(format!("reading lock {}", path.display()))),
    }
}
