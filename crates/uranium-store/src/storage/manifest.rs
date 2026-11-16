use std::collections::BTreeMap;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use super::{MANIFEST_FILE, StorageConfig};

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

pub(super) fn copy_tree(src: &Path, dst: &Path) -> Result<()> {
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
