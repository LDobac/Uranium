//! Shared drive storage (061): run lock with a heartbeat lease (R5), atomic checkpoint push (R4),
//! verified pull into the local cache (R3, R7, R11-1/2), retention (R6), recovery of checkpoints
//! that were not pushed (R11-3..6) and the local mode of the PC that hosts the share (R12).
//!
//! Layout (061-R2): `<shared>/checkpoints/<run>/{ckpt-<step>/, LATEST, promoted/, league/}`,
//! `<shared>/locks/<run>.json`; the cache mirrors `checkpoints/<run>/` under `<cache>`.
//!
//! A checkpoint is written into [`Run::stage_dir`], sealed with a manifest ([`Run::seal`]) and
//! then pushed ([`Run::push`]). In local mode (R12) the stage lives on the share itself and
//! sealing already publishes it (same-disk rename).

use std::collections::BTreeMap;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow, bail, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const MANIFEST_FILE: &str = "manifest.json";
pub const LATEST_FILE: &str = "LATEST";
const CKPT_PREFIX: &str = "ckpt-";

/// Where the shared drive and the cache are, and the lease timing (061-R1, R5, R6, R12).
#[derive(Debug, Clone)]
pub struct StorageConfig {
    pub shared: PathBuf,
    pub cache: PathBuf,
    /// 061-R12: the share is this PC's own disk; read originals in place.
    pub shared_local: bool,
    /// Lock owner name (061-R5). `URANIUM_HOST_NAME`, else the OS host name.
    pub host: String,
    pub heartbeat_every: Duration,
    pub stale_after: Duration,
    /// Newest checkpoints kept per run (061-R6).
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

    /// 061-R1 / R12 / 001-R9: `URANIUM_SHARED_DIR`, `URANIUM_CACHE_DIR`, `URANIUM_SHARED_LOCAL`,
    /// `URANIUM_HOST_NAME` (the process environment wins over `.env`).
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

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// 061-R9: names that are safe on SMB and Windows.
pub fn check_name(name: &str) -> Result<()> {
    ensure!(!name.is_empty() && !name.starts_with('.'), "bad name {name:?}");
    ensure!(!name.chars().any(|c| "<>:\"/\\|?*".contains(c) || c.is_control()), "name {name:?} has a character SMB/Windows forbid (061-R9)");
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Manifest (061-R4, R7)

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct FileEntry {
    pub sha256: String,
    pub bytes: u64,
}

/// SHA-256 of every file under a directory, by `/`-separated relative path.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Default)]
pub struct Manifest {
    pub files: BTreeMap<String, FileEntry>,
}

pub fn sha256_file(path: &Path) -> Result<FileEntry> {
    let mut f = fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    let mut bytes = 0u64;
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
        bytes += n as u64;
    }
    Ok(FileEntry { sha256: h.finalize().iter().map(|b| format!("{b:02x}")).collect(), bytes })
}

fn walk(root: &Path, dir: &Path, out: &mut Vec<String>) -> Result<()> {
    for e in fs::read_dir(dir)? {
        let e = e?;
        let p = e.path();
        if e.file_type()?.is_dir() {
            walk(root, &p, out)?;
        } else {
            let rel = p.strip_prefix(root)?.components().map(|c| c.as_os_str().to_string_lossy().into_owned()).collect::<Vec<_>>().join("/");
            if rel != MANIFEST_FILE && !rel.ends_with(".part") {
                out.push(rel);
            }
        }
    }
    Ok(())
}

impl Manifest {
    pub fn build(dir: &Path) -> Result<Self> {
        let mut rels = Vec::new();
        walk(dir, dir, &mut rels)?;
        let mut files = BTreeMap::new();
        for r in rels {
            files.insert(r.clone(), sha256_file(&dir.join(&r))?);
        }
        Ok(Self { files })
    }

    pub fn read(dir: &Path) -> Result<Self> {
        let p = dir.join(MANIFEST_FILE);
        Ok(serde_json::from_slice(&fs::read(&p).with_context(|| format!("reading {}", p.display()))?)?)
    }

    /// Written through a temporary name so a reader never sees half a manifest.
    pub fn write(&self, dir: &Path) -> Result<()> {
        write_atomic(&dir.join(MANIFEST_FILE), &serde_json::to_vec_pretty(self)?)
    }

    /// 061-R7: every listed file exists with the listed hash.
    pub fn verify(&self, dir: &Path) -> Result<()> {
        for (rel, want) in &self.files {
            let got = sha256_file(&dir.join(rel))?;
            ensure!(&got == want, "{} does not match its manifest (061-R7)", dir.join(rel).display());
        }
        Ok(())
    }
}

/// Reads the directory's manifest and verifies it.
pub fn verify_dir(dir: &Path) -> Result<Manifest> {
    let m = Manifest::read(dir)?;
    m.verify(dir)?;
    Ok(m)
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = path.with_file_name(format!("{}.tmp-{}", path.file_name().unwrap().to_string_lossy(), std::process::id()));
    {
        let mut f = fs::File::create(&tmp).with_context(|| format!("writing {}", tmp.display()))?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    fs::rename(&tmp, path).with_context(|| format!("renaming onto {}", path.display()))
}

/// 061-R11-1/2: brings `dst` to the manifest of `src`. Files already matching are skipped, the
/// rest are copied as `<name>.part` and renamed only after their hash checks out; the manifest is
/// written last, so an interrupted pull resumes where it stopped.
pub fn pull_tree(src: &Path, dst: &Path) -> Result<Manifest> {
    let m = Manifest::read(src)?;
    for (rel, want) in &m.files {
        let to = dst.join(rel);
        if to.is_file() && sha256_file(&to).is_ok_and(|got| &got == want) {
            continue;
        }
        fs::create_dir_all(to.parent().unwrap())?;
        let part = to.with_file_name(format!("{}.part", to.file_name().unwrap().to_string_lossy()));
        fs::copy(src.join(rel), &part).with_context(|| format!("copying {}", src.join(rel).display()))?;
        let got = sha256_file(&part)?;
        if &got != want {
            fs::remove_file(&part).ok();
            bail!("{} does not match its manifest (061-R7)", src.join(rel).display());
        }
        fs::rename(&part, &to)?;
    }
    m.write(dst)?;
    Ok(m)
}

/// 061-R3 / R11-1 / R12: a dataset folder (with a manifest) under `<shared>/datasets/`, ready to
/// read: copied into `<cache>/datasets/` (only what changed), or verified in place in local mode.
pub fn open_dataset(cfg: &StorageConfig, rel: &Path) -> Result<PathBuf> {
    let src = cfg.shared.join("datasets").join(rel);
    if cfg.shared_local {
        verify_dir(&src)?;
        return Ok(src);
    }
    let dst = cfg.cache.join("datasets").join(rel);
    pull_tree(&src, &dst)?;
    Ok(dst)
}

fn copy_tree(src: &Path, dst: &Path) -> Result<()> {
    fs::create_dir_all(dst)?;
    for e in fs::read_dir(src)? {
        let e = e?;
        let to = dst.join(e.file_name());
        if e.file_type()?.is_dir() {
            copy_tree(&e.path(), &to)?;
        } else {
            fs::copy(e.path(), &to).with_context(|| format!("copying {}", e.path().display()))?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Run lock (061-R5)

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct LockInfo {
    pub host: String,
    pub pid: u32,
    /// Unix seconds (writer's clock; staleness uses the share's file times instead).
    pub started: u64,
    pub heartbeat: u64,
}

/// Result of a heartbeat (061-R5, R11-4).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LockStatus {
    Held,
    /// Another PC took the run over (`--take-over`): stop training and do not push (R11-4).
    Lost(LockInfo),
    /// The share cannot be reached; keep training and retry (R11-3).
    Unreachable(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PushOutcome {
    Pushed,
    /// The share was unreachable; the checkpoint stays in the cache and is retried (R11-3).
    Deferred(String),
    /// The lock belongs to someone else now; nothing was pushed (R11-4).
    LockLost(LockInfo),
}

/// Where a run resumes from (061-R3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Resume {
    pub step: u64,
    /// Verified checkpoint folder to load (cache copy, or the share itself in local mode).
    pub dir: PathBuf,
    /// The checkpoint came from the cache, pushed at start (R11-5).
    pub recovered: bool,
}

/// Steps of a push, for fault injection in tests (061-AC2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PushStage {
    /// Files copied into `ckpt-<step>.tmp-<host>/`, manifest not written yet.
    Copied,
    /// Manifest written, not renamed yet.
    Manifested,
    /// Renamed to `ckpt-<step>/`, `LATEST` not updated yet.
    Renamed,
    Done,
}

pub fn ckpt_name(step: u64) -> String {
    format!("{CKPT_PREFIX}{step}")
}

fn parse_ckpt(name: &str) -> Option<u64> {
    name.strip_prefix(CKPT_PREFIX)?.parse().ok()
}

/// Sealed checkpoint steps in a folder (ignores `.tmp-*` stages).
fn ckpt_steps(dir: &Path) -> Result<Vec<u64>> {
    let mut v = Vec::new();
    if let Ok(rd) = fs::read_dir(dir) {
        for e in rd.flatten() {
            if e.file_type().is_ok_and(|t| t.is_dir())
                && let Some(s) = parse_ckpt(&e.file_name().to_string_lossy())
            {
                v.push(s);
            }
        }
    }
    v.sort_unstable();
    Ok(v)
}

/// One training run holding its lock (061-R5). Dropping it without [`Run::release`] keeps the
/// lock file, as a crash would: the lease then expires after `stale_after`.
pub struct Run {
    pub cfg: StorageConfig,
    pub run_id: String,
    me: LockInfo,
    /// Lock found at acquire time (same host after a crash, or a taken-over stale lock).
    previous: Option<LockInfo>,
}

impl Run {
    /// 061-R5 / AC3: takes the run lock. A fresh lock of another PC is refused; a stale one
    /// (no heartbeat for `stale_after`) only with `take_over`. A lock of this same host (its
    /// process is gone after a crash) is taken back: the spec refuses other PCs only.
    pub fn acquire(cfg: StorageConfig, run_id: &str, take_over: bool) -> Result<Self> {
        check_name(run_id)?;
        check_name(&cfg.host)?;
        let locks = cfg.shared.join("locks");
        fs::create_dir_all(&locks).with_context(|| format!("creating {}", locks.display()))?;
        fs::create_dir_all(cfg.shared.join("checkpoints").join(run_id))?;
        let now = now_secs();
        let me = LockInfo { host: cfg.host.clone(), pid: std::process::id(), started: now, heartbeat: now };
        let path = locks.join(format!("{run_id}.json"));
        let mut run = Self { cfg, run_id: run_id.to_string(), me, previous: None };
        match fs::OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(mut f) => {
                f.write_all(&serde_json::to_vec_pretty(&run.me)?)?;
                f.sync_all()?;
                return Ok(run);
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(e) => return Err(anyhow!(e).context(format!("creating lock {}", path.display()))),
        }
        let held: LockInfo = serde_json::from_slice(&fs::read(&path)?).with_context(|| format!("reading lock {}", path.display()))?;
        let age = run.lock_age()?;
        if held.host != run.me.host {
            if age < run.cfg.stale_after {
                return Err(anyhow::Error::new(LockBusy(format!("run {run_id} is locked by {} (pid {}, last heartbeat {}s ago) (061-R5)", held.host, held.pid, age.as_secs()))));
            }
            ensure!(take_over, "run {run_id}: lock of {} expired ({}s without heartbeat); pass --take-over to take it (061-R5)", held.host, age.as_secs());
        }
        run.previous = Some(held);
        write_atomic(&path, &serde_json::to_vec_pretty(&run.me)?)?;
        Ok(run)
    }

    pub fn lock_path(&self) -> PathBuf {
        self.cfg.shared.join("locks").join(format!("{}.json", self.run_id))
    }

    /// Checkpoints of the run on the share.
    pub fn shared_dir(&self) -> PathBuf {
        self.cfg.shared.join("checkpoints").join(&self.run_id)
    }

    /// Checkpoints of the run in the cache (unused for checkpoints in local mode).
    pub fn cache_dir(&self) -> PathBuf {
        self.cfg.cache.join("checkpoints").join(&self.run_id)
    }

    /// Where checkpoints are written before they are published.
    fn work_dir(&self) -> PathBuf {
        if self.cfg.shared_local { self.shared_dir() } else { self.cache_dir() }
    }

    pub fn me(&self) -> &LockInfo {
        &self.me
    }

    /// The lock that was there before this run took it, if any.
    pub fn previous_owner(&self) -> Option<&LockInfo> {
        self.previous.as_ref()
    }

    /// Lock age by the share's own clock: a probe file is touched and its modification time is
    /// compared with the lock's (host clocks may differ by minutes).
    fn lock_age(&self) -> Result<Duration> {
        let probe = self.cfg.shared.join("locks").join(format!(".clock-{}", self.cfg.host));
        fs::write(&probe, b"")?;
        let now = fs::metadata(&probe)?.modified()?;
        let lock = fs::metadata(self.lock_path())?.modified()?;
        Ok(now.duration_since(lock).unwrap_or(Duration::ZERO))
    }

    fn read_lock(&self) -> Result<LockInfo> {
        Ok(serde_json::from_slice(&fs::read(self.lock_path())?)?)
    }

    fn owns(&self, l: &LockInfo) -> bool {
        l.host == self.me.host && l.pid == self.me.pid
    }

    /// 061-R5 / R11-4: refreshes the lease (call every `heartbeat_every`). After an outage the
    /// lock is read again first; if it changed hands the caller must stop without pushing.
    pub fn heartbeat(&mut self) -> LockStatus {
        let held = match self.read_lock() {
            Ok(l) => l,
            Err(e) => return LockStatus::Unreachable(format!("{e:#}")),
        };
        if !self.owns(&held) {
            return LockStatus::Lost(held);
        }
        self.me.heartbeat = now_secs();
        match serde_json::to_vec_pretty(&self.me).map_err(anyhow::Error::from).and_then(|b| write_atomic(&self.lock_path(), &b)) {
            Ok(()) => LockStatus::Held,
            Err(e) => LockStatus::Unreachable(format!("{e:#}")),
        }
    }

    /// Step named by the share's `LATEST` (061-R4).
    pub fn latest(&self) -> Result<Option<u64>> {
        match fs::read_to_string(self.shared_dir().join(LATEST_FILE)) {
            Ok(s) => Ok(Some(parse_ckpt(s.trim()).ok_or_else(|| anyhow!("bad LATEST {:?}", s.trim()))?)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(anyhow!(e).context("reading LATEST")),
        }
    }

    /// 061-R3 start: recover unpushed checkpoints (R11-5), then pull `LATEST` into the cache
    /// (or verify it in place, R12). `None` = new run.
    pub fn begin(&mut self) -> Result<Option<Resume>> {
        let latest = self.latest()?;
        if !self.cfg.shared_local
            && let Some(step) = self.recoverable(latest)?
        {
            match self.push(step)? {
                PushOutcome::Pushed => return Ok(Some(Resume { step, dir: self.cache_dir().join(ckpt_name(step)), recovered: true })),
                other => bail!("could not push the unpushed checkpoint {} at start (061-R11-5): {other:?}", ckpt_name(step)),
            }
        }
        let Some(step) = latest else { return Ok(None) };
        let src = self.shared_dir().join(ckpt_name(step));
        if self.cfg.shared_local {
            verify_dir(&src)?;
            return Ok(Some(Resume { step, dir: src, recovered: false }));
        }
        let dst = self.cache_dir().join(ckpt_name(step));
        pull_tree(&src, &dst)?;
        Ok(Some(Resume { step, dir: dst, recovered: false }))
    }

    /// 061-R11-5: the newest cached checkpoint newer than `LATEST`, if this host was the last
    /// owner of the lock and the checkpoint verifies.
    fn recoverable(&self, latest: Option<u64>) -> Result<Option<u64>> {
        let last_owner_is_me = self.previous.as_ref().is_some_and(|p| p.host == self.me.host);
        if !last_owner_is_me {
            return Ok(None);
        }
        for step in ckpt_steps(&self.cache_dir())?.into_iter().rev() {
            if latest.is_some_and(|l| step <= l) {
                break;
            }
            if verify_dir(&self.cache_dir().join(ckpt_name(step))).is_ok() {
                return Ok(Some(step));
            }
        }
        Ok(None)
    }

    fn stage_name(&self, step: u64) -> String {
        format!("{}.tmp-{}", ckpt_name(step), self.cfg.host)
    }

    /// An empty folder to write the checkpoint of `step` into (061-R4 step 1).
    pub fn stage_dir(&self, step: u64) -> Result<PathBuf> {
        let d = self.work_dir().join(self.stage_name(step));
        if d.exists() {
            fs::remove_dir_all(&d)?;
        }
        fs::create_dir_all(&d)?;
        Ok(d)
    }

    /// 061-R4 steps 1–2 where the checkpoint was written: manifest, then rename to
    /// `ckpt-<step>/`. In local mode (R12) this is on the share, and `LATEST` is updated too.
    pub fn seal(&mut self, step: u64) -> Result<PathBuf> {
        let stage = self.work_dir().join(self.stage_name(step));
        Manifest::build(&stage)?.write(&stage)?;
        let done = self.work_dir().join(ckpt_name(step));
        if done.exists() {
            fs::remove_dir_all(&done)?;
        }
        fs::rename(&stage, &done)?;
        if self.cfg.shared_local {
            self.write_latest(step)?;
        }
        Ok(done)
    }

    fn write_latest(&self, step: u64) -> Result<()> {
        write_atomic(&self.shared_dir().join(LATEST_FILE), format!("{}\n", ckpt_name(step)).as_bytes())
    }

    /// 061-R4: publishes the sealed checkpoint `step` (tmp folder → manifest → rename → LATEST).
    /// R11-3: an unreachable share defers; R11-4: a lost lock pushes nothing.
    pub fn push(&mut self, step: u64) -> Result<PushOutcome> {
        self.push_until(step, PushStage::Done)
    }

    /// [`Run::push`] that stops after `stage`, to simulate a crash mid-push (061-AC2).
    #[doc(hidden)]
    pub fn push_until(&mut self, step: u64, stage: PushStage) -> Result<PushOutcome> {
        match self.heartbeat() {
            LockStatus::Held => {}
            LockStatus::Lost(l) => return Ok(PushOutcome::LockLost(l)),
            LockStatus::Unreachable(e) => return Ok(PushOutcome::Deferred(e)),
        }
        if self.cfg.shared_local {
            // Already on the share (seal); only LATEST may be behind.
            if stage >= PushStage::Done && self.latest()?.is_none_or(|l| l < step) {
                self.write_latest(step)?;
            }
            return Ok(PushOutcome::Pushed);
        }
        let local = self.cache_dir().join(ckpt_name(step));
        let manifest = Manifest::read(&local).with_context(|| format!("{} is not sealed", local.display()))?;
        match self.copy_to_share(step, &local, &manifest, stage) {
            Ok(()) => Ok(PushOutcome::Pushed),
            // The checkpoint itself is fine; any I/O failure here is the share (R11-3).
            Err(e) => Ok(PushOutcome::Deferred(format!("{e:#}"))),
        }
    }

    fn copy_to_share(&self, step: u64, local: &Path, manifest: &Manifest, stage: PushStage) -> Result<()> {
        let done = self.shared_dir().join(ckpt_name(step));
        if !done.exists() {
            let tmp = self.shared_dir().join(self.stage_name(step));
            if tmp.exists() {
                fs::remove_dir_all(&tmp)?;
            }
            for rel in manifest.files.keys() {
                let to = tmp.join(rel);
                fs::create_dir_all(to.parent().unwrap())?;
                fs::copy(local.join(rel), &to)?;
            }
            if stage == PushStage::Copied {
                return Ok(());
            }
            manifest.write(&tmp)?;
            manifest.verify(&tmp)?;
            if stage == PushStage::Manifested {
                return Ok(());
            }
            fs::rename(&tmp, &done)?;
            if stage == PushStage::Renamed {
                return Ok(());
            }
        } else {
            ensure!(&Manifest::read(&done)? == manifest, "{} already exists with other contents", done.display());
        }
        if stage >= PushStage::Done && self.latest()?.is_none_or(|l| l < step) {
            self.write_latest(step)?;
        }
        Ok(())
    }

    /// Cached checkpoints not on the share yet, oldest first (R11-3). Pushes go oldest first, so
    /// these are the cached steps after `LATEST` (older ones were pushed, maybe pruned since).
    /// With the share unreachable every cached checkpoint counts. Empty in local mode.
    pub fn unpushed(&self) -> Result<Vec<u64>> {
        if self.cfg.shared_local {
            return Ok(Vec::new());
        }
        let latest = self.latest().ok().flatten();
        let shared = ckpt_steps(&self.shared_dir())?;
        Ok(ckpt_steps(&self.cache_dir())?.into_iter().filter(|s| latest.is_none_or(|l| *s > l) && !shared.contains(s)).collect())
    }

    /// R11-3: retries every unpushed checkpoint, oldest first; stops at the first failure.
    pub fn push_pending(&mut self) -> Result<PushOutcome> {
        for step in self.unpushed()? {
            match self.push(step)? {
                PushOutcome::Pushed => {}
                other => return Ok(other),
            }
        }
        Ok(PushOutcome::Pushed)
    }

    /// 061-R6: keeps the newest `keep_latest` checkpoints on the share (always the one `LATEST`
    /// names) and in the cache only unpushed ones plus the newest. `promoted/` and `league/` are
    /// never touched. Only the lock holder prunes; an unreachable share prunes nothing.
    pub fn prune(&mut self) -> Result<()> {
        if self.heartbeat() != LockStatus::Held {
            return Ok(());
        }
        let latest = self.latest()?;
        let shared = ckpt_steps(&self.shared_dir())?;
        let keep_from = shared.len().saturating_sub(self.cfg.keep_latest);
        for (i, step) in shared.iter().enumerate() {
            if i < keep_from && Some(*step) != latest {
                fs::remove_dir_all(self.shared_dir().join(ckpt_name(*step)))?;
            }
        }
        if !self.cfg.shared_local {
            let unpushed = self.unpushed()?;
            let local = ckpt_steps(&self.cache_dir())?;
            let newest = local.last().copied();
            for step in local {
                if !unpushed.contains(&step) && Some(step) != newest {
                    fs::remove_dir_all(self.cache_dir().join(ckpt_name(step)))?;
                }
            }
        }
        Ok(())
    }

    fn publish_copy(&self, step: u64, kind: &str, name: &str) -> Result<PathBuf> {
        check_name(name)?;
        let src = self.shared_dir().join(ckpt_name(step));
        verify_dir(&src)?;
        let parent = self.shared_dir().join(kind);
        let dst = parent.join(name);
        ensure!(!dst.exists(), "{} already exists", dst.display());
        let tmp = parent.join(format!("{name}.tmp-{}", self.cfg.host));
        if tmp.exists() {
            fs::remove_dir_all(&tmp)?;
        }
        copy_tree(&src, &tmp)?;
        verify_dir(&tmp)?;
        fs::rename(&tmp, &dst)?;
        Ok(dst)
    }

    /// 061-R2 `promoted/<name>/`: a pushed checkpoint that passed the 050 gate (kept forever).
    pub fn promote(&self, step: u64, name: &str) -> Result<PathBuf> {
        self.publish_copy(step, "promoted", name)
    }

    /// 061-R2 `league/<name>/`: a 042 league snapshot (kept forever).
    pub fn league_snapshot(&self, step: u64, name: &str) -> Result<PathBuf> {
        self.publish_copy(step, "league", name)
    }

    /// 061-R5 / 042-R10: normal exit, after the final push. Removes the lock if it is still ours.
    pub fn release(self) -> Result<()> {
        if self.read_lock().is_ok_and(|l| self.owns(&l)) {
            fs::remove_file(self.lock_path())?;
        }
        Ok(())
    }
}

/// A fresh lock of another PC holds the run (061-R5); the run queue worker moves on (061-R13).
#[derive(Debug)]
pub struct LockBusy(pub String);

impl std::fmt::Display for LockBusy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for LockBusy {}

// ---------------------------------------------------------------------------------------------
// Run queue support (061-R13..R15)

/// The run's lock as another process sees it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LockView {
    Free,
    /// Held with a heartbeat younger than `stale_after`.
    Held(LockInfo),
    /// Held, but no heartbeat for `stale_after` or longer.
    Stale(LockInfo),
}

/// Age of `path` by the share's clock (see `Run::lock_age`).
fn share_age(cfg: &StorageConfig, path: &Path) -> Result<Duration> {
    fs::create_dir_all(cfg.shared.join("locks"))?;
    let probe = cfg.shared.join("locks").join(format!(".clock-{}", cfg.host));
    fs::write(&probe, b"")?;
    let now = fs::metadata(&probe)?.modified()?;
    let t = fs::metadata(path)?.modified()?;
    Ok(now.duration_since(t).unwrap_or(Duration::ZERO))
}

pub fn peek_lock(cfg: &StorageConfig, run_id: &str) -> Result<LockView> {
    let path = cfg.shared.join("locks").join(format!("{run_id}.json"));
    let Ok(bytes) = fs::read(&path) else { return Ok(LockView::Free) };
    let Ok(info) = serde_json::from_slice::<LockInfo>(&bytes) else { return Ok(LockView::Free) };
    Ok(if share_age(cfg, &path)? >= cfg.stale_after { LockView::Stale(info) } else { LockView::Held(info) })
}

/// 061-R14 yield request on a run: the requesting host takes the run next.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct YieldRequest {
    pub host: String,
    pub priority: i32,
    pub time: u64,
}

/// Requests older than this are ignored (061-R14).
pub const YIELD_EXPIRY: Duration = Duration::from_secs(30 * 60);

fn yield_path(cfg: &StorageConfig, run_id: &str) -> PathBuf {
    cfg.shared.join("locks").join(format!("{run_id}.yield"))
}

/// A live (not expired) yield request on the run.
pub fn read_yield(cfg: &StorageConfig, run_id: &str) -> Option<YieldRequest> {
    let p = yield_path(cfg, run_id);
    let req: YieldRequest = serde_json::from_slice(&fs::read(&p).ok()?).ok()?;
    (share_age(cfg, &p).ok()? < YIELD_EXPIRY).then_some(req)
}

pub fn request_yield(cfg: &StorageConfig, run_id: &str, priority: i32) -> Result<()> {
    fs::create_dir_all(cfg.shared.join("locks"))?;
    let req = YieldRequest { host: cfg.host.clone(), priority, time: now_secs() };
    write_atomic(&yield_path(cfg, run_id), &serde_json::to_vec_pretty(&req)?)
}

pub fn clear_yield(cfg: &StorageConfig, run_id: &str) {
    let _ = fs::remove_file(yield_path(cfg, run_id));
}

/// `checkpoints/<id>/DONE`: the run finished (061-R13).
pub fn done_path(cfg: &StorageConfig, run_id: &str) -> PathBuf {
    cfg.shared.join("checkpoints").join(run_id).join("DONE")
}

pub fn is_done(cfg: &StorageConfig, run_id: &str) -> bool {
    done_path(cfg, run_id).is_file()
}

pub fn mark_done(cfg: &StorageConfig, run_id: &str, step: u64) -> Result<()> {
    let p = done_path(cfg, run_id);
    fs::create_dir_all(p.parent().unwrap())?;
    write_atomic(&p, format!("{}\n", ckpt_name(step)).as_bytes())
}

/// 061-R15: appends one line (plus host and time) to `runs/<id>/devices.jsonl`.
pub fn log_device(cfg: &StorageConfig, run_id: &str, line: serde_json::Value) {
    let dir = cfg.shared.join("runs").join(run_id);
    let mut v = line;
    if let Some(o) = v.as_object_mut() {
        o.insert("host".into(), serde_json::json!(cfg.host));
        o.insert("time".into(), serde_json::json!(now_secs()));
    }
    let res = fs::create_dir_all(&dir).and_then(|_| {
        let mut f = fs::OpenOptions::new().create(true).append(true).open(dir.join("devices.jsonl"))?;
        writeln!(f, "{v}")
    });
    if let Err(e) = res {
        eprintln!("devices.jsonl not written ({e}); training continues (061-R11-3)");
    }
}

/// 061-R13 worker status in `workers/<host>.json`.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct WorkerStatus {
    pub host: String,
    pub priority: i32,
    pub backend: String,
    pub run: Option<String>,
    pub time: u64,
}

pub fn write_worker(cfg: &StorageConfig, priority: i32, backend: &str, run: Option<&str>) -> Result<()> {
    let dir = cfg.shared.join("workers");
    fs::create_dir_all(&dir)?;
    let s = WorkerStatus { host: cfg.host.clone(), priority, backend: backend.into(), run: run.map(str::to_string), time: now_secs() };
    write_atomic(&dir.join(format!("{}.json", cfg.host)), &serde_json::to_vec_pretty(&s)?)
}

pub fn read_worker(cfg: &StorageConfig, host: &str) -> Option<WorkerStatus> {
    serde_json::from_slice(&fs::read(cfg.shared.join("workers").join(format!("{host}.json"))).ok()?).ok()
}
