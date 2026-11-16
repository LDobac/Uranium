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

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct FileEntry {
    pub sha256: String,
    pub bytes: u64,
}

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

    pub fn write(&self, dir: &Path) -> Result<()> {
        write_atomic(&dir.join(MANIFEST_FILE), &serde_json::to_vec_pretty(self)?)
    }

    pub fn verify(&self, dir: &Path) -> Result<()> {
        for (rel, want) in &self.files {
            let got = sha256_file(&dir.join(rel))?;
            ensure!(&got == want, "{} does not match its manifest (061-R7)", dir.join(rel).display());
        }
        Ok(())
    }

    pub fn verify_on_server(&self, dir: &Path) -> Result<()> {
        for (rel, want) in &self.files {
            let p = dir.join(rel);
            crate::share_io::sync_file(&p).with_context(|| format!("syncing {}", p.display()))?;
            let mut h = Sha256::new();
            let bytes = crate::share_io::read_uncached_with(&p, &mut |b| h.update(b)).with_context(|| format!("reading back {}", p.display()))?;
            let got = FileEntry { sha256: h.finalize().iter().map(|b| format!("{b:02x}")).collect(), bytes };
            ensure!(&got == want, "{} on the share does not match its manifest (061-R4)", p.display());
        }
        Ok(())
    }
}

pub fn verify_dir(dir: &Path) -> Result<Manifest> {
    let m = Manifest::read(dir)?;
    m.verify(dir)?;
    Ok(m)
}

pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
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

pub fn open_dataset(cfg: &StorageConfig, rel: &Path) -> Result<PathBuf> {
    let src = cfg.shared.join("datasets").join(rel);
    if cfg.shared_local {
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PushOutcome {
    Pushed,
    Deferred(String),
    LockLost(LockInfo),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Resume {
    pub step: u64,
    pub dir: PathBuf,
    pub recovered: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PushStage {
    Copied,
    Manifested,
    Renamed,
    Done,
}

pub fn ckpt_name(step: u64) -> String {
    format!("{CKPT_PREFIX}{step}")
}

fn parse_ckpt(name: &str) -> Option<u64> {
    name.strip_prefix(CKPT_PREFIX)?.parse().ok()
}

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

fn take_lock(cfg: &StorageConfig, id: &str, take: &TakeOver) -> Result<(LockInfo, Option<LockInfo>)> {
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

pub struct Run {
    pub cfg: StorageConfig,
    pub run_id: String,
    me: LockInfo,
    previous: Option<LockInfo>,
    lineage: Vec<String>,
    remove_dir: fn(&Path) -> std::io::Result<()>,
}

const LINEAGE_KEEP: usize = 256;
const LINEAGE_FILE: &str = "lineage.json";

pub fn ckpt_id(dir: &Path) -> Result<String> {
    let m = Manifest::read(dir)?;
    Ok(Sha256::digest(serde_json::to_vec(&m)?).iter().map(|b| format!("{b:02x}")).collect())
}

fn read_lineage(dir: &Path) -> Vec<String> {
    fs::read(dir.join(LINEAGE_FILE)).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default()
}

impl Run {
    pub fn acquire(cfg: StorageConfig, run_id: &str, take_over: bool) -> Result<Self> {
        Self::acquire_with(cfg, run_id, if take_over { TakeOver::Force } else { TakeOver::No })
    }

    pub fn acquire_with(cfg: StorageConfig, run_id: &str, take: TakeOver) -> Result<Self> {
        check_name(run_id)?;
        check_name(&cfg.host)?;
        fs::create_dir_all(cfg.shared.join("checkpoints").join(run_id))?;
        let (me, previous) = take_lock(&cfg, run_id, &take)?;
        Ok(Self { cfg, run_id: run_id.to_string(), me, previous, lineage: Vec::new(), remove_dir: |p| fs::remove_dir_all(p) })
    }

    pub fn lock_path(&self) -> PathBuf {
        self.cfg.shared.join("locks").join(format!("{}.json", self.run_id))
    }

    pub fn shared_dir(&self) -> PathBuf {
        self.cfg.shared.join("checkpoints").join(&self.run_id)
    }

    pub fn cache_dir(&self) -> PathBuf {
        self.cfg.cache.join("checkpoints").join(&self.run_id)
    }

    fn work_dir(&self) -> PathBuf {
        if self.cfg.shared_local { self.shared_dir() } else { self.cache_dir() }
    }

    pub fn me(&self) -> &LockInfo {
        &self.me
    }

    pub fn previous_owner(&self) -> Option<&LockInfo> {
        self.previous.as_ref()
    }

    fn read_lock(&self) -> Result<LockInfo> {
        Ok(serde_json::from_slice(&fs::read(self.lock_path())?)?)
    }

    fn owns(&self, l: &LockInfo) -> bool {
        l.host == self.me.host && l.pid == self.me.pid
    }

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

    pub fn latest(&self) -> Result<Option<u64>> {
        match fs::read_to_string(self.shared_dir().join(LATEST_FILE)) {
            Ok(s) => Ok(Some(parse_ckpt(s.trim()).ok_or_else(|| anyhow!("bad LATEST {:?}", s.trim()))?)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(anyhow!(e).context("reading LATEST")),
        }
    }

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

    fn continue_from(&mut self, dir: &Path) -> Result<()> {
        let mut l = read_lineage(dir);
        l.push(ckpt_id(dir)?);
        self.lineage = l;
        Ok(())
    }

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

    pub fn stage_dir(&self, step: u64) -> Result<PathBuf> {
        let d = self.work_dir().join(self.stage_name(step));
        if d.exists() {
            fs::remove_dir_all(&d)?;
        }
        fs::create_dir_all(&d)?;
        Ok(d)
    }

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

    pub fn push(&mut self, step: u64) -> Result<PushOutcome> {
        self.push_until(step, PushStage::Done)
    }

    #[doc(hidden)]
    pub fn push_until(&mut self, step: u64, stage: PushStage) -> Result<PushOutcome> {
        match self.heartbeat() {
            LockStatus::Held => {}
            LockStatus::Lost(l) => return Ok(PushOutcome::LockLost(l)),
            LockStatus::Unreachable(e) => return Ok(PushOutcome::Deferred(e)),
        }
        if self.cfg.shared_local {
            if stage >= PushStage::Done && self.latest()?.is_none_or(|l| l < step) {
                self.write_latest(step)?;
            }
            return Ok(PushOutcome::Pushed);
        }
        let local = self.cache_dir().join(ckpt_name(step));
        let manifest = Manifest::read(&local).with_context(|| format!("{} is not sealed", local.display()))?;
        match self.copy_to_share(step, &local, &manifest, stage) {
            Ok(()) => Ok(PushOutcome::Pushed),
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

    pub fn push_pending(&mut self) -> Result<PushOutcome> {
        for step in self.unpushed()? {
            match self.push(step)? {
                PushOutcome::Pushed => {}
                other => return Ok(other),
            }
        }
        Ok(PushOutcome::Pushed)
    }

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

    pub fn promote(&self, step: u64, name: &str) -> Result<PathBuf> {
        self.publish_copy(step, "promoted", name)
    }

    pub fn league_snapshot(&self, step: u64, name: &str, agent: &str) -> Result<PathBuf> {
        check_name(name)?;
        check_name(agent)?;
        let src = self.shared_dir().join(ckpt_name(step));
        let prefix = format!("{}/{agent}/", crate::ckpt::AGENTS_DIR);
        let files: BTreeMap<String, FileEntry> = Manifest::read(&src)?
            .files
            .into_iter()
            .filter(|(rel, _)| rel.strip_prefix(&prefix).is_some_and(|f| !f.rsplit('/').next().unwrap_or(f).starts_with(crate::ckpt::OPTIM_PREFIX)))
            .collect();
        ensure!(!files.is_empty(), "{} has no files under {prefix}", src.display());
        let sub = Manifest { files };
        sub.verify(&src)?;
        let parent = self.shared_dir().join("league");
        let dst = parent.join(name);
        ensure!(!dst.exists(), "{} already exists", dst.display());
        let tmp = parent.join(format!("{name}.tmp-{}", self.cfg.host));
        if tmp.exists() {
            fs::remove_dir_all(&tmp)?;
        }
        let copied = (|| -> Result<()> {
            for rel in sub.files.keys() {
                let to = tmp.join(rel);
                fs::create_dir_all(to.parent().expect("a file under agents/<agent>"))?;
                fs::copy(src.join(rel), &to).with_context(|| format!("copying {}", src.join(rel).display()))?;
            }
            sub.write(&tmp)?;
            verify_dir(&tmp)?;
            fs::rename(&tmp, &dst)?;
            Ok(())
        })();
        if let Err(e) = copied {
            let _ = fs::remove_dir_all(&tmp);
            return Err(e);
        }
        Ok(dst)
    }

    pub fn release(self) -> Result<()> {
        if self.read_lock().is_ok_and(|l| self.owns(&l)) {
            fs::remove_file(self.lock_path())?;
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

pub fn peek_lock(cfg: &StorageConfig, run_id: &str) -> Result<Option<LockInfo>> {
    let path = cfg.shared.join("locks").join(format!("{run_id}.json"));
    match fs::read(&path) {
        Ok(bytes) => Ok(Some(serde_json::from_slice(&bytes).with_context(|| format!("parsing lock {}", path.display()))?)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(anyhow!(e).context(format!("reading lock {}", path.display()))),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LockView {
    Free,
    Held(LockInfo),
    Stale(LockInfo),
    Unknown(String),
}

#[derive(Debug)]
pub struct ContentObserver<T> {
    seen: std::collections::HashMap<String, (T, std::time::Instant, std::time::Instant)>,
}

impl<T> Default for ContentObserver<T> {
    fn default() -> Self {
        Self { seen: std::collections::HashMap::new() }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Watched<T> {
    Absent,
    Fresh(T),
    Expired(T),
    Unknown(String),
}

impl<T: Clone + PartialEq> ContentObserver<T> {
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

    pub fn reset(&mut self) {
        self.seen.clear();
    }
}

pub type LockObserver = ContentObserver<LockInfo>;

pub type YieldObserver = ContentObserver<YieldRequest>;

impl ContentObserver<LockInfo> {
    pub fn observe(&mut self, run_id: &str, read: &Result<Option<LockInfo>>, now: std::time::Instant, stale_after: Duration) -> LockView {
        match self.watch(run_id, read, now, stale_after) {
            Watched::Absent => LockView::Free,
            Watched::Fresh(l) => LockView::Held(l),
            Watched::Expired(l) => LockView::Stale(l),
            Watched::Unknown(e) => LockView::Unknown(e),
        }
    }
}

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

pub fn done_path(cfg: &StorageConfig, run_id: &str) -> PathBuf {
    cfg.shared.join("checkpoints").join(run_id).join("DONE")
}

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

#[derive(Serialize, Debug, Clone, PartialEq, Eq)]
pub struct DeviceSpan {
    pub host: String,
    pub backend: Option<String>,
    pub precision: Option<String>,
    pub from: u64,
    pub to: u64,
    pub end: String,
}

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

pub fn read_devices(cfg: &StorageConfig, run_id: &str) -> Result<Vec<serde_json::Value>> {
    let p = cfg.shared.join("runs").join(run_id).join("devices.jsonl");
    let text = fs::read_to_string(&p).with_context(|| format!("reading {}", p.display()))?;
    Ok(text.lines().filter_map(|l| serde_json::from_str(l).ok()).collect())
}

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

pub fn dataset_dirs(datasets: &[PathBuf], shards: &[PathBuf]) -> Result<Vec<PathBuf>> {
    ensure!(!datasets.is_empty() || !shards.is_empty(), "pass --dataset or --shards");
    let mut dirs = Vec::new();
    if !datasets.is_empty() {
        let cfg = StorageConfig::from_env()?;
        let mut names = datasets.to_vec();
        names.sort();
        names.dedup();
        for d in &names {
            dirs.push(open_dataset(&cfg, d)?);
        }
    }
    dirs.extend(shards.iter().cloned());
    Ok(dirs)
}
