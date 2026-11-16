use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use anyhow::{Context, Result, anyhow, bail, ensure};
use sha2::{Digest, Sha256};
use super::{LATEST_FILE, CKPT_PREFIX, StorageConfig, now_secs, check_name};
use super::manifest::{FileEntry, Manifest, verify_dir, write_atomic, pull_tree, copy_tree};
use super::lock::{LockInfo, TakeOver, LockStatus, take_lock};

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
