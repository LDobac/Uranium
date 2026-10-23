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
    /// Newest checkpoints kept per run on the share (061-R6, default 5; 0 = all, G-D6). Runs
    /// set it from their config (`keep_checkpoints`).
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

    /// 061-R4 push check: like [`Manifest::verify`] on files this client just wrote to the share,
    /// but each file is `fsync`ed and read past this client's cache, so the hash is of what the
    /// file server holds (same rule as the 040-R17 shard copy; a cached read-back passed shards
    /// that were zero-filled on the share, R-5).
    pub fn verify_on_server(&self, dir: &Path) -> Result<()> {
        for (rel, want) in &self.files {
            let p = dir.join(rel);
            uranium_replay::share_io::sync_file(&p).with_context(|| format!("syncing {}", p.display()))?;
            let mut h = Sha256::new();
            let bytes = uranium_replay::share_io::read_uncached_with(&p, &mut |b| h.update(b)).with_context(|| format!("reading back {}", p.display()))?;
            let got = FileEntry { sha256: h.finalize().iter().map(|b| format!("{b:02x}")).collect(), bytes };
            ensure!(&got == want, "{} on the share does not match its manifest (061-R4)", p.display());
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

/// Writes through a temporary file unique to this process and call (threads of one process may
/// write the same file, e.g. the worker status, 061-R13), then renames it over `path`.
fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let tmp = path.with_file_name(format!("{}.tmp-{}-{n}", path.file_name().unwrap().to_string_lossy(), std::process::id()));
    {
        let mut f = fs::File::create(&tmp).with_context(|| format!("writing {}", tmp.display()))?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    fs::rename(&tmp, path).with_context(|| format!("renaming onto {}", path.display()))
}

/// 061-R11-1/2: brings `dst` to the manifest of `src`. Files already matching are skipped, the
/// rest are copied as `<name>.part` and renamed only after their hash checks out; the manifest is
/// written last, so an interrupted pull resumes where it stopped. Every file prints one progress
/// line on stderr (a first dataset pull of tens of GB is otherwise silent for longer than the
/// 060-R6 hang limit could tolerate).
pub fn pull_tree(src: &Path, dst: &Path) -> Result<Manifest> {
    let m = Manifest::read(src)?;
    let n = m.files.len();
    for (i, (rel, want)) in m.files.iter().enumerate() {
        if crate::stop_requested() {
            return Err(anyhow::Error::new(crate::Interrupted(format!("Ctrl+C: pull of {} stopped after {i}/{n} files; it resumes there next time (061-R11-2)", src.display()))));
        }
        let to = dst.join(rel);
        if to.is_file() && sha256_file(&to).is_ok_and(|got| &got == want) {
            crate::elog!("[pull] {}/{n} {} ok ({} B, cached)", i + 1, src.join(rel).display(), want.bytes);
            continue;
        }
        crate::elog!("[pull] {}/{n} {} copying ({} B)", i + 1, src.join(rel).display(), want.bytes);
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
        // Hashing in place: one progress line per file, as in `pull_tree`.
        let m = Manifest::read(&src)?;
        let n = m.files.len();
        for (i, (rel, want)) in m.files.iter().enumerate() {
            if crate::stop_requested() {
                return Err(anyhow::Error::new(crate::Interrupted(format!("Ctrl+C: verification of {} stopped after {i}/{n} files", src.display()))));
            }
            crate::elog!("[verify] {}/{n} {} ({} B)", i + 1, src.join(rel).display(), want.bytes);
            let got = sha256_file(&src.join(rel))?;
            ensure!(&got == want, "{} does not match its manifest (061-R7)", src.join(rel).display());
        }
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
    /// Unix seconds by the writer's clock. Never compared with another PC's clock: expiry is a
    /// heartbeat that stays the same while a worker watches it (061-R5, [`LockObserver`]).
    pub started: u64,
    pub heartbeat: u64,
}

/// How [`Run::acquire_with`] treats a lock of another PC (061-R5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TakeOver {
    /// Refused (exit 75), fresh or expired alike.
    No,
    /// Manual `--take-over`: taken at once, with a warning.
    Force,
    /// Worker auto take-over (061-R13): taken only if the lock still has exactly the contents the
    /// worker saw unchanged for `stale_after` ([`LockObserver`]); otherwise refused (exit 75).
    IfUnchanged(LockInfo),
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
/// lock file, as a crash would: the same host takes it back on restart, a worker elsewhere after
/// watching it unchanged for `stale_after` (061-R5), anyone with `--take-over`.
pub struct Run {
    pub cfg: StorageConfig,
    pub run_id: String,
    me: LockInfo,
    /// Lock found at acquire time (same host after a crash, or a taken-over lock).
    previous: Option<LockInfo>,
    /// 061-R11-5 / R14 ③: ids ([`ckpt_id`]) of the checkpoints this run continues, oldest first:
    /// the resumed checkpoint's own lineage plus itself, then every checkpoint sealed here.
    lineage: Vec<String>,
    /// Folder delete of [`Run::prune`] (`fs::remove_dir_all`; replaced in tests).
    remove_dir: fn(&Path) -> std::io::Result<()>,
}

/// Lineage entries kept in a checkpoint (`lineage.json`); an unpushed checkpoint is always a
/// few checkpoints past `LATEST`, so older entries are not needed.
const LINEAGE_KEEP: usize = 256;
const LINEAGE_FILE: &str = "lineage.json";

/// Identity of a sealed checkpoint: SHA-256 of its manifest (every file's hash, the lineage
/// file included, so equal ids mean equal contents and equal history).
pub fn ckpt_id(dir: &Path) -> Result<String> {
    let m = Manifest::read(dir)?;
    Ok(Sha256::digest(serde_json::to_vec(&m)?).iter().map(|b| format!("{b:02x}")).collect())
}

/// The lineage stored in a checkpoint (empty for checkpoints written before 061-R11-5 lineage).
fn read_lineage(dir: &Path) -> Vec<String> {
    fs::read(dir.join(LINEAGE_FILE)).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default()
}

impl Run {
    /// 061-R5 / AC3: takes the run lock; `take_over` = manual `--take-over` ([`TakeOver::Force`]).
    pub fn acquire(cfg: StorageConfig, run_id: &str, take_over: bool) -> Result<Self> {
        Self::acquire_with(cfg, run_id, if take_over { TakeOver::Force } else { TakeOver::No })
    }

    /// 061-R5 / AC3: takes the run lock. A lock of another PC, fresh or expired, is refused with
    /// [`LockBusy`] (exit 75) unless taken over: [`TakeOver::Force`] at once with a warning,
    /// [`TakeOver::IfUnchanged`] only if the lock is still the one a worker watched expire. No
    /// file time or clock is compared here (061-AC3). A lock of this same host (its process is
    /// gone after a crash) is taken back: the spec refuses other PCs only.
    pub fn acquire_with(cfg: StorageConfig, run_id: &str, take: TakeOver) -> Result<Self> {
        check_name(run_id)?;
        check_name(&cfg.host)?;
        let locks = cfg.shared.join("locks");
        fs::create_dir_all(&locks).with_context(|| format!("creating {}", locks.display()))?;
        fs::create_dir_all(cfg.shared.join("checkpoints").join(run_id))?;
        let now = now_secs();
        let me = LockInfo { host: cfg.host.clone(), pid: std::process::id(), started: now, heartbeat: now };
        let path = locks.join(format!("{run_id}.json"));
        let mut run = Self { cfg, run_id: run_id.to_string(), me, previous: None, lineage: Vec::new(), remove_dir: |p| fs::remove_dir_all(p) };
        match fs::OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(mut f) => {
                f.write_all(&serde_json::to_vec_pretty(&run.me)?)?;
                f.sync_all()?;
                return Ok(run);
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(e) => return Err(anyhow!(e).context(format!("creating lock {}", path.display()))),
        }
        let bytes = fs::read(&path).with_context(|| format!("reading lock {}", path.display()))?;
        let Ok(held) = serde_json::from_slice::<LockInfo>(&bytes) else {
            // Another process created it a moment ago and has not written it yet.
            return Err(anyhow::Error::new(LockBusy(format!("run {run_id}: lock {} is being written by another process (061-R5)", path.display()))));
        };
        if held.host != run.me.host {
            match &take {
                TakeOver::No => {
                    return Err(anyhow::Error::new(LockBusy(format!(
                        "run {run_id} is locked by {} (pid {}); a lock of another PC is taken only with --take-over (061-R5)",
                        held.host, held.pid
                    ))));
                }
                TakeOver::Force => crate::elog!(
                    "WARNING: --take-over: taking run {run_id} from {} (pid {}) now (061-R5). If that training is still alive it stops without a push at its next heartbeat, and checkpoints only it had (not pushed) may be lost (061-R11-4).",
                    held.host, held.pid
                ),
                TakeOver::IfUnchanged(seen) if *seen == held => {
                    crate::elog!("taking over run {run_id}: the lock of {} (pid {}) did not change for {} min (061-R5, R13)", held.host, held.pid, run.cfg.stale_after.as_secs() / 60)
                }
                TakeOver::IfUnchanged(_) => {
                    return Err(anyhow::Error::new(LockBusy(format!("run {run_id}: the lock of {} changed since it was seen expired; not taken (061-R5)", held.host))));
                }
            }
        }
        run.previous = Some(held);
        write_atomic(&path, &serde_json::to_vec_pretty(&run.me)?)?;
        // Two takers at once both rename their lock in; the last rename wins. Reading it back
        // narrows the double-ownership window but does not close it: a loser whose read comes
        // before the winner's rename still starts, and stops (Lost, no push) at its next
        // heartbeat, within a minute (061-R5, R11-4). A loser that reads after it exits 75 now.
        let now = run.read_lock().with_context(|| format!("reading back lock {}", path.display()))?;
        if !run.owns(&now) {
            return Err(anyhow::Error::new(LockBusy(format!("run {run_id}: {} (pid {}) took the lock at the same moment (061-R5)", now.host, now.pid))));
        }
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
            let dir = self.cache_dir().join(ckpt_name(step));
            self.continue_from(&dir)?;
            match self.push(step)? {
                PushOutcome::Pushed => return Ok(Some(Resume { step, dir, recovered: true })),
                other => bail!("could not push the unpushed checkpoint {} at start (061-R11-5): {other:?}", ckpt_name(step)),
            }
        }
        let Some(step) = latest else { return Ok(None) };
        let src = self.shared_dir().join(ckpt_name(step));
        if self.cfg.shared_local {
            verify_dir(&src)?;
            self.continue_from(&src)?;
            return Ok(Some(Resume { step, dir: src, recovered: false }));
        }
        let dst = self.cache_dir().join(ckpt_name(step));
        pull_tree(&src, &dst)?;
        self.continue_from(&dst)?;
        Ok(Some(Resume { step, dir: dst, recovered: false }))
    }

    /// The run continues the checkpoint in `dir`: its lineage plus itself (061-R11-5).
    fn continue_from(&mut self, dir: &Path) -> Result<()> {
        let mut l = read_lineage(dir);
        l.push(ckpt_id(dir)?);
        self.lineage = l;
        Ok(())
    }

    /// 061-R11-5: the newest cached checkpoint newer than `LATEST`, if this host was the last
    /// owner of the lock, the checkpoint verifies and it descends from `LATEST` (lineage
    /// condition, R14 ③: a checkpoint left from before another PC pushed its own is never used).
    fn recoverable(&self, latest: Option<u64>) -> Result<Option<u64>> {
        let last_owner_is_me = self.previous.as_ref().is_some_and(|p| p.host == self.me.host);
        if !last_owner_is_me {
            return Ok(None);
        }
        let latest_id = match latest {
            Some(l) => Some(ckpt_id(&self.shared_dir().join(ckpt_name(l))).context("reading the manifest LATEST names")?),
            None => None,
        };
        for step in ckpt_steps(&self.cache_dir())?.into_iter().rev() {
            if latest.is_some_and(|l| step <= l) {
                break;
            }
            let dir = self.cache_dir().join(ckpt_name(step));
            let descends = latest_id.as_ref().is_none_or(|id| read_lineage(&dir).contains(id));
            if !descends {
                crate::elog!("{} in the cache does not descend from LATEST: not recovered (061-R11-5)", dir.display());
                continue;
            }
            if verify_dir(&dir).is_ok() {
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
        let keep_from = self.lineage.len().saturating_sub(LINEAGE_KEEP);
        fs::write(stage.join(LINEAGE_FILE), serde_json::to_vec(&self.lineage[keep_from..])?)?;
        Manifest::build(&stage)?.write(&stage)?;
        let done = self.work_dir().join(ckpt_name(step));
        if done.exists() {
            fs::remove_dir_all(&done)?;
        }
        fs::rename(&stage, &done)?;
        self.lineage.push(ckpt_id(&done)?);
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
            manifest.verify_on_server(&tmp)?;
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
    /// With the share unreachable every cached checkpoint counts. Only checkpoints of this run's
    /// lineage count (061-R14 ③: another lineage's leftovers are never pushed). Empty in local
    /// mode.
    pub fn unpushed(&self) -> Result<Vec<u64>> {
        if self.cfg.shared_local {
            return Ok(Vec::new());
        }
        let latest = self.latest().ok().flatten();
        let shared = ckpt_steps(&self.shared_dir())?;
        Ok(ckpt_steps(&self.cache_dir())?
            .into_iter()
            .filter(|s| latest.is_none_or(|l| *s > l) && !shared.contains(s))
            .filter(|s| ckpt_id(&self.cache_dir().join(ckpt_name(*s))).is_ok_and(|id| self.lineage.contains(&id)))
            .collect())
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
    /// names; `keep_latest = 0` keeps all, G-D6) and in the cache only unpushed ones plus the
    /// newest. `promoted/` and `league/` are never touched. Only the lock holder prunes; an
    /// unreachable share prunes nothing. A checkpoint that cannot be deleted (B-7: an SMB delete
    /// error) is logged and left for the next prune; the others are still deleted. Returns how
    /// many deletes failed.
    pub fn prune(&mut self) -> Result<usize> {
        if self.heartbeat() != LockStatus::Held {
            return Ok(0);
        }
        let mut failed = 0;
        let rm = self.remove_dir;
        let mut remove = |dir: PathBuf| {
            if let Err(e) = rm(&dir) {
                failed += 1;
                crate::elog!("[prune] WARNING: could not delete {} ({e}); training goes on, the next prune tries again (061-R6)", dir.display());
            }
        };
        let latest = self.latest()?;
        if self.cfg.keep_latest > 0 {
            let shared = ckpt_steps(&self.shared_dir())?;
            let keep_from = shared.len().saturating_sub(self.cfg.keep_latest);
            for (i, step) in shared.iter().enumerate() {
                if i < keep_from && Some(*step) != latest {
                    remove(self.shared_dir().join(ckpt_name(*step)));
                }
            }
        }
        if !self.cfg.shared_local {
            let unpushed = self.unpushed()?;
            let local = ckpt_steps(&self.cache_dir())?;
            let newest = local.last().copied();
            for step in local {
                if !unpushed.contains(&step) && Some(step) != newest {
                    remove(self.cache_dir().join(ckpt_name(step)));
                }
            }
        }
        Ok(failed)
    }

    /// Replaces the folder delete used by [`Run::prune`] (fault injection in tests, B-7).
    #[doc(hidden)]
    pub fn set_remove_dir(&mut self, f: fn(&Path) -> std::io::Result<()>) {
        self.remove_dir = f;
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

/// `<shared>/<name>/`, created if missing, but never the share root itself: an unmounted mount
/// point must not quietly become a local "share" (061-R13, H-2).
fn share_subdir(cfg: &StorageConfig, name: &str) -> Result<PathBuf> {
    let d = cfg.shared.join(name);
    match fs::create_dir(&d) {
        Ok(()) => Ok(d),
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => Ok(d),
        Err(e) => Err(anyhow!(e).context(format!("creating {}", d.display()))),
    }
}

/// 061-R13 (H-2): the share is there. A shared folder that lost its mount (an empty mount point)
/// or a dropped connection fails here, so "no lock / no DONE" is never read off a missing share.
/// The worker creates `checkpoints/` when it starts.
pub fn probe_share(cfg: &StorageConfig) -> Result<()> {
    let d = cfg.shared.join("checkpoints");
    let m = fs::metadata(&d).with_context(|| format!("shared drive not reachable ({})", d.display()))?;
    ensure!(m.is_dir(), "{} is not a folder", d.display());
    fs::read_dir(&d).with_context(|| format!("listing {}", d.display()))?;
    Ok(())
}

/// The run's lock file as read now: `None` = no lock. A read or parse failure is an error, never
/// "no lock" (061-R13, H-2: unknown is not Free).
pub fn peek_lock(cfg: &StorageConfig, run_id: &str) -> Result<Option<LockInfo>> {
    let path = cfg.shared.join("locks").join(format!("{run_id}.json"));
    match fs::read(&path) {
        Ok(bytes) => Ok(Some(serde_json::from_slice(&bytes).with_context(|| format!("parsing lock {}", path.display()))?)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(anyhow!(e).context(format!("reading lock {}", path.display()))),
    }
}

/// The run's lock as a worker sees it (061-R5, R13).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LockView {
    Free,
    /// Held, and not (yet) seen unchanged for `stale_after`.
    Held(LockInfo),
    /// The worker saw exactly this lock unchanged for `stale_after` on its own monotonic clock.
    Stale(LockInfo),
    /// Could not be read (061-R13 "unknown"): neither free nor expired.
    Unknown(String),
}

/// Expiry by watching (061-R5 locks, H-3; 061-R14 yield requests): a file's contents count as
/// expired only when this process has seen them stay exactly the same for the limit, by its own
/// monotonic clock. File times and the clocks written in the files are never compared, so PC
/// clock differences and the share's time stamps do not matter. A change, a read failure or a
/// gap in watching (longer than half of the limit, e.g. while the worker trained another run)
/// starts over.
#[derive(Debug)]
pub struct ContentObserver<T> {
    /// run id -> (contents, first seen, last seen).
    seen: std::collections::HashMap<String, (T, std::time::Instant, std::time::Instant)>,
}

impl<T> Default for ContentObserver<T> {
    fn default() -> Self {
        Self { seen: std::collections::HashMap::new() }
    }
}

/// One read of a watched file, as [`ContentObserver::watch`] sees it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Watched<T> {
    /// No file.
    Absent,
    /// Present, not (yet) seen unchanged for the limit.
    Fresh(T),
    /// Seen unchanged for the limit on this process's monotonic clock.
    Expired(T),
    /// Could not be read (061-R13 "unknown").
    Unknown(String),
}

impl<T: Clone + PartialEq> ContentObserver<T> {
    /// Records one read of the run's file at `now` against `limit`.
    pub fn watch(&mut self, run_id: &str, read: &Result<Option<T>>, now: std::time::Instant, limit: Duration) -> Watched<T> {
        let v = match read {
            Ok(None) => {
                self.seen.remove(run_id);
                return Watched::Absent;
            }
            Err(e) => {
                self.seen.remove(run_id);
                return Watched::Unknown(format!("{e:#}"));
            }
            Ok(Some(v)) => v,
        };
        let continued = self.seen.get(run_id).is_some_and(|(x, _, last)| x == v && now.saturating_duration_since(*last) <= limit / 2);
        let first = if continued { self.seen[run_id].1 } else { now };
        self.seen.insert(run_id.to_string(), (v.clone(), first, now));
        if now.saturating_duration_since(first) >= limit { Watched::Expired(v.clone()) } else { Watched::Fresh(v.clone()) }
    }

    /// Forgets every observation (the worker stops watching while it trains).
    pub fn reset(&mut self) {
        self.seen.clear();
    }
}

/// 061-R5 expiry for the worker's automatic take-over: lock contents (host, pid, heartbeat)
/// watched unchanged for `stale_after`.
pub type LockObserver = ContentObserver<LockInfo>;

/// 061-R14 expiry of yield requests: request contents watched unchanged for [`YIELD_EXPIRY`].
pub type YieldObserver = ContentObserver<YieldRequest>;

impl ContentObserver<LockInfo> {
    /// Records one read of the run's lock at `now` and says what it means.
    pub fn observe(&mut self, run_id: &str, read: &Result<Option<LockInfo>>, now: std::time::Instant, stale_after: Duration) -> LockView {
        match self.watch(run_id, read, now, stale_after) {
            Watched::Absent => LockView::Free,
            Watched::Fresh(l) => LockView::Held(l),
            Watched::Expired(l) => LockView::Stale(l),
            Watched::Unknown(e) => LockView::Unknown(e),
        }
    }
}

/// 061-R14 yield request on a run: the requesting host takes the run next.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct YieldRequest {
    pub host: String,
    pub priority: i32,
    pub time: u64,
}

/// A request seen unchanged this long by a worker's own monotonic clock is expired and removed
/// (061-R14, [`YieldObserver`]).
pub const YIELD_EXPIRY: Duration = Duration::from_secs(30 * 60);

fn yield_path(cfg: &StorageConfig, run_id: &str) -> PathBuf {
    cfg.shared.join("locks").join(format!("{run_id}.yield"))
}

/// The run's yield request as written; `Ok(None)` = none. A read failure is an error (061-R13
/// "unknown"). Expiry is not judged here: workers watch requests ([`YieldObserver`]) and remove
/// expired ones ([`remove_yield_if`]).
pub fn read_yield(cfg: &StorageConfig, run_id: &str) -> Result<Option<YieldRequest>> {
    let p = yield_path(cfg, run_id);
    let bytes = match fs::read(&p) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(anyhow!(e).context(format!("reading {}", p.display()))),
    };
    Ok(Some(serde_json::from_slice(&bytes).with_context(|| format!("parsing {}", p.display()))?))
}

/// Removes the run's yield request if it still has exactly the contents `seen` (an expired
/// request, 061-R14; whoever wrote it).
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

/// A live yield request by another host (training loop check: a read failure counts as none,
/// training goes on, 061-R11-3).
pub fn yield_by_other(cfg: &StorageConfig, run_id: &str) -> Option<YieldRequest> {
    read_yield(cfg, run_id).ok().flatten().filter(|y| y.host != cfg.host)
}

pub fn request_yield(cfg: &StorageConfig, run_id: &str, priority: i32) -> Result<()> {
    share_subdir(cfg, "locks")?;
    let req = YieldRequest { host: cfg.host.clone(), priority, time: now_secs() };
    write_atomic(&yield_path(cfg, run_id), &serde_json::to_vec_pretty(&req)?)
}

/// Removes the run's yield request if this host wrote it (061-R14: when the host takes the run,
/// takes another run, or its worker stops). Another host's request is left alone.
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

/// `checkpoints/<id>/DONE`: the run finished (061-R13).
pub fn done_path(cfg: &StorageConfig, run_id: &str) -> PathBuf {
    cfg.shared.join("checkpoints").join(run_id).join("DONE")
}

/// Whether the run has `DONE`. A read failure is an error, not "not done" (061-R13, H-2).
pub fn is_done(cfg: &StorageConfig, run_id: &str) -> Result<bool> {
    let p = done_path(cfg, run_id);
    match fs::metadata(&p) {
        Ok(m) => Ok(m.is_file()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(anyhow!(e).context(format!("reading {}", p.display()))),
    }
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
        crate::elog!("devices.jsonl not written ({e}); training continues (061-R11-3)");
    }
}

/// One training session's stretch of updates on one host (061-R15): updates `from..to`.
#[derive(Serialize, Debug, Clone, PartialEq, Eq)]
pub struct DeviceSpan {
    pub host: String,
    pub backend: Option<String>,
    pub precision: Option<String>,
    pub from: u64,
    pub to: u64,
    /// The `stop` reason, or `crash` when the session has no `stop` line.
    pub end: String,
}

/// 061-R15 aggregation of `devices.jsonl` lines (in file order). A session starts at a `start`
/// line; its `checkpoint` and `stop` lines are the following ones of the same host (a PC that
/// lost the lock may write its `stop` after the next owner's `start`). It ends at its `stop`
/// step; without a `stop` (crash, kill), or after a lost lock (nothing pushed after the last
/// checkpoint), at its last `checkpoint` step (its start step if none): later updates were not
/// kept and the next session does them again. A next session that starts at a smaller step cuts
/// it there.
pub fn device_spans(lines: &[serde_json::Value]) -> Vec<DeviceSpan> {
    struct S {
        span: DeviceSpan,
        last_ckpt: u64,
        stop: Option<(u64, String)>,
    }
    let s = |v: &serde_json::Value, k: &str| v.get(k).and_then(|x| x.as_str()).map(str::to_string);
    let mut sessions: Vec<S> = Vec::new();
    for v in lines {
        let (Some(event), Some(host)) = (s(v, "event"), s(v, "host")) else { continue };
        let step = v.get("step").and_then(|x| x.as_u64()).unwrap_or(0);
        if event == "start" {
            let span = DeviceSpan { host, backend: s(v, "backend"), precision: s(v, "precision"), from: step, to: step, end: "crash".into() };
            sessions.push(S { span, last_ckpt: step, stop: None });
            continue;
        }
        let Some(cur) = sessions.iter_mut().rev().find(|x| x.span.host == host) else { continue };
        if cur.stop.is_some() {
            continue;
        }
        match event.as_str() {
            "checkpoint" => cur.last_ckpt = cur.last_ckpt.max(step),
            "stop" => cur.stop = Some((step, s(v, "reason").unwrap_or_else(|| "stop".into()))),
            _ => {}
        }
    }
    let starts: Vec<u64> = sessions.iter().map(|x| x.span.from).collect();
    sessions
        .into_iter()
        .enumerate()
        .map(|(i, x)| {
            let mut span = x.span;
            let (end, why) = match x.stop {
                Some((_, r)) if r == "lock lost" => (x.last_ckpt, r),
                Some((step, r)) => (step, r),
                None => (x.last_ckpt, "crash".into()),
            };
            let mut to = end.max(span.from);
            if let Some(&next) = starts.get(i + 1)
                && next < to
            {
                to = next.max(span.from);
            }
            span.to = to;
            span.end = why;
            span
        })
        .collect()
}

/// `runs/<id>/devices.jsonl` as JSON lines; a torn last line (crash mid-write) is skipped.
pub fn read_devices(cfg: &StorageConfig, run_id: &str) -> Result<Vec<serde_json::Value>> {
    let p = cfg.shared.join("runs").join(run_id).join("devices.jsonl");
    let text = fs::read_to_string(&p).with_context(|| format!("reading {}", p.display()))?;
    Ok(text.lines().filter_map(|l| serde_json::from_str(l).ok()).collect())
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
    share_subdir(cfg, "workers")?;
    let s = WorkerStatus { host: cfg.host.clone(), priority, backend: backend.into(), run: run.map(str::to_string), time: now_secs() };
    write_atomic(&dir.join(format!("{}.json", cfg.host)), &serde_json::to_vec_pretty(&s)?)
}

pub fn read_worker(cfg: &StorageConfig, host: &str) -> Option<WorkerStatus> {
    serde_json::from_slice(&fs::read(cfg.shared.join("workers").join(format!("{host}.json"))).ok()?).ok()
}
