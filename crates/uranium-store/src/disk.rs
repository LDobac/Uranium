use std::path::Path;

use anyhow::{Context, Result};

pub const WARN_USAGE: f64 = 0.80;

pub const REPLAY_BUDGET_BYTES: u64 = 100 << 30;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiskSpace {
    pub total: u64,
    pub available: u64,
    pub used: u64,
}

impl DiskSpace {
    pub fn usage(&self) -> f64 {
        let denom = self.used + self.available;
        if denom == 0 { 0.0 } else { self.used as f64 / denom as f64 }
    }
}

#[cfg(unix)]
pub fn space(path: &Path) -> Result<DiskSpace> {
    use std::os::unix::ffi::OsStrExt;
    let c = std::ffi::CString::new(path.as_os_str().as_bytes()).context("path with a NUL byte")?;
    let mut st: libc::statvfs = unsafe { std::mem::zeroed() };
    let r = unsafe { libc::statvfs(c.as_ptr(), &mut st) };
    if r != 0 {
        return Err(std::io::Error::last_os_error()).with_context(|| format!("statvfs {}", path.display()));
    }
    let frsize = st.f_frsize as u64;
    let total = st.f_blocks as u64 * frsize;
    let free = st.f_bfree as u64 * frsize;
    Ok(DiskSpace { total, available: st.f_bavail as u64 * frsize, used: total.saturating_sub(free) })
}

#[cfg(windows)]
pub fn space(path: &Path) -> Result<DiskSpace> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::GetDiskFreeSpaceExW;
    let mut wide: Vec<u16> = path.as_os_str().encode_wide().collect();
    if path.is_dir() && wide.last().is_some_and(|&c| c != b'\\' as u16 && c != b'/' as u16) {
        wide.push(b'\\' as u16);
    }
    wide.push(0);
    let (mut avail, mut total, mut free) = (0u64, 0u64, 0u64);
    let ok = unsafe { GetDiskFreeSpaceExW(wide.as_ptr(), &mut avail, &mut total, &mut free) };
    if ok == 0 {
        return Err(std::io::Error::last_os_error()).with_context(|| format!("GetDiskFreeSpaceExW {}", path.display()));
    }
    Ok(DiskSpace { total, available: avail, used: total.saturating_sub(free) })
}

#[derive(Debug, Clone, PartialEq)]
pub enum DiskCheck {
    Ok(DiskSpace),
    Warn(DiskSpace),
    Unknown(String),
}

pub fn warn_disks(tag: &str, paths: &[(&str, &Path)]) -> Vec<DiskCheck> {
    paths
        .iter()
        .map(|(name, p)| {
            let c = match space(p) {
                Ok(s) if s.usage() >= WARN_USAGE => DiskCheck::Warn(s),
                Ok(s) => DiskCheck::Ok(s),
                Err(e) => DiskCheck::Unknown(format!("{e:#}")),
            };
            match &c {
                DiskCheck::Warn(s) => crate::elog!(
                    "[{tag}] WARNING: the disk of the {name} ({}) is {:.1}% full ({} GB free of {} GB; 061-R8 warns at {:.0}%). Training goes on.",
                    p.display(),
                    s.usage() * 100.0,
                    s.available >> 30,
                    s.total >> 30,
                    WARN_USAGE * 100.0
                ),
                DiskCheck::Unknown(e) => crate::elog!("[{tag}] WARNING: could not read the disk usage of the {name} ({}): {e} (061-R8). Training goes on.", p.display()),
                DiskCheck::Ok(_) => {}
            }
            c
        })
        .collect()
}

pub fn replay_bytes(shared: &Path) -> Result<u64> {
    fn tree(p: &Path) -> Result<u64> {
        let mut n = 0;
        let rd = match std::fs::read_dir(p) {
            Ok(rd) => rd,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(0),
            Err(e) => return Err(e).with_context(|| format!("listing {}", p.display())),
        };
        for e in rd {
            let e = e?;
            let t = e.file_type()?;
            if t.is_dir() {
                n += tree(&e.path())?;
            } else if t.is_file() {
                n += e.metadata()?.len();
            }
        }
        Ok(n)
    }
    let runs = shared.join("runs");
    let rd = match std::fs::read_dir(&runs) {
        Ok(rd) => rd,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(e) => return Err(e).with_context(|| format!("listing {}", runs.display())),
    };
    let mut total = 0;
    for e in rd {
        let e = e?;
        if e.file_type()?.is_dir() {
            total += tree(&e.path().join("replays"))?;
        }
    }
    Ok(total)
}

pub fn warn_replay_budget(shared: &Path, budget: u64) -> Option<u64> {
    match replay_bytes(shared) {
        Ok(n) => {
            if n > budget {
                crate::elog!(
                    "[rl] WARNING: RL replays under {} take {:.1} GB, over the {} GB budget (060-R7). They are kept (042-R11); move or delete them by hand.",
                    shared.join("runs").display(),
                    n as f64 / (1u64 << 30) as f64,
                    budget >> 30
                );
            }
            Some(n)
        }
        Err(e) => {
            crate::elog!("[rl] WARNING: could not measure the RL replays ({e:#}) (060-R7)");
            None
        }
    }
}

pub fn warn_storage(tag: &str, cfg: &crate::storage::StorageConfig) -> Vec<DiskCheck> {
    let mut paths: Vec<(&str, &Path)> = vec![("shared drive", &cfg.shared)];
    if !cfg.shared_local && cfg.cache.exists() {
        paths.push(("local cache", &cfg.cache));
    }
    warn_disks(tag, &paths)
}

pub fn spawn_replay_check(shared: &Path, budget: u64) {
    use std::sync::atomic::{AtomicBool, Ordering};
    static RUNNING: AtomicBool = AtomicBool::new(false);
    if RUNNING.swap(true, Ordering::SeqCst) {
        return;
    }
    let shared = shared.to_path_buf();
    std::thread::spawn(move || {
        warn_replay_budget(&shared, budget);
        RUNNING.store(false, Ordering::SeqCst);
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usage_like_df_061_r8() {
        let s = DiskSpace { total: 100, available: 15, used: 80 };
        assert!((s.usage() - 80.0 / 95.0).abs() < 1e-12);
        assert_eq!(DiskSpace { total: 0, available: 0, used: 0 }.usage(), 0.0);
    }

    #[test]
    fn reads_the_disk_of_a_folder_061_r8() {
        let s = space(&std::env::temp_dir()).unwrap();
        assert!(s.total > 0 && s.used <= s.total && s.available <= s.total, "{s:?}");
        assert!((0.0..=1.0).contains(&s.usage()));
        let missing = std::env::temp_dir().join("uranium-no-such-dir-061-r8").join("x");
        let c = warn_disks("test", &[("missing", &missing)]);
        assert!(matches!(c[0], DiskCheck::Unknown(_)), "{c:?}");
    }

    #[test]
    fn replay_total_sums_every_run_060_r7() {
        let root = std::env::temp_dir().join(format!("uranium-replay-budget-{}", std::process::id()));
        std::fs::remove_dir_all(&root).ok();
        assert_eq!(replay_bytes(&root).unwrap(), 0, "no runs folder");
        for (run, n) in [("a", 3usize), ("b", 5)] {
            let d = root.join("runs").join(run).join("replays");
            std::fs::create_dir_all(&d).unwrap();
            std::fs::write(d.join("g1-sample.rep"), vec![0u8; n]).unwrap();
        }
        std::fs::write(root.join("runs").join("a").join("devices.jsonl"), b"0123456789").unwrap();
        assert_eq!(replay_bytes(&root).unwrap(), 8);
        assert_eq!(warn_replay_budget(&root, 7), Some(8));
        std::fs::remove_dir_all(&root).ok();
    }
}
