//! hostd settings from `.env` / process environment (spec 020-R9, 001-R9).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

#[derive(Debug, Clone)]
pub struct Config {
    pub listen: String,
    /// BWAPI build dir containing `bin/BWAPILauncher[.exe]`.
    pub bwapi_dir: PathBuf,
    pub bridge_lib: PathBuf,
    pub data_dir: PathBuf,
    pub maps_dir: PathBuf,
    pub max_games: u32,
    /// Per-game working directories.
    pub work_dir: PathBuf,
    /// Third-party bots: `<dir>/<name>/module.{so,dll}` (+ optional `bwapi-data/`), 021 T1.
    pub oss_bots_dir: Option<PathBuf>,
    /// Windows: first TCP port for OpenBW LAN pairs (011-R16).
    #[cfg_attr(not(windows), allow(dead_code))]
    pub port_base: u16,
}

fn var(name: &str) -> Option<String> {
    std::env::var(name).ok().map(|v| v.trim().to_string()).filter(|v| !v.is_empty())
}

fn path_var(name: &str) -> Result<PathBuf> {
    let p = PathBuf::from(var(name).with_context(|| format!("{name} is not set (.env, spec 020-R9)"))?);
    Ok(p)
}

fn must_exist(p: &Path, what: &str) -> Result<()> {
    if !p.exists() {
        bail!("{what} not found: {}", p.display());
    }
    Ok(())
}

impl Config {
    pub fn from_env() -> Result<Self> {
        let cores = std::thread::available_parallelism().map(|n| n.get() as u32).unwrap_or(2);
        let cfg = Self {
            listen: var("URANIUM_HOSTD_LISTEN").unwrap_or_else(|| "127.0.0.1:7700".into()),
            bwapi_dir: path_var("URANIUM_BWAPI_DIR")?,
            bridge_lib: path_var("URANIUM_BRIDGE_LIB")?,
            data_dir: path_var("URANIUM_DATA_DIR")?,
            maps_dir: path_var("URANIUM_MAPS_DIR")?,
            max_games: var("URANIUM_HOSTD_MAX_GAMES").and_then(|v| v.parse().ok()).unwrap_or(cores.saturating_sub(1).max(1)),
            work_dir: var("URANIUM_HOSTD_WORK_DIR").map(PathBuf::from).unwrap_or_else(|| std::env::temp_dir().join("uranium-hostd")),
            oss_bots_dir: var("URANIUM_OSS_BOTS_DIR").map(PathBuf::from),
            port_base: var("URANIUM_HOSTD_PORT_BASE").and_then(|v| v.parse().ok()).unwrap_or(17_000),
        };
        must_exist(&cfg.launcher(), "BWAPILauncher")?;
        must_exist(&cfg.bridge_lib, "bridge library")?;
        must_exist(&cfg.data_dir, "data dir")?;
        must_exist(&cfg.maps_dir, "maps dir")?;
        std::fs::create_dir_all(&cfg.work_dir)?;
        Ok(cfg)
    }

    pub fn launcher(&self) -> PathBuf {
        let exe = if cfg!(windows) { "BWAPILauncher.exe" } else { "BWAPILauncher" };
        self.bwapi_dir.join("bin").join(exe)
    }

    /// Registered third-party bots (directories containing a module file).
    pub fn oss_bots(&self) -> Vec<String> {
        let Some(dir) = &self.oss_bots_dir else { return vec![] };
        let Ok(rd) = std::fs::read_dir(dir) else { return vec![] };
        let mut v: Vec<String> = rd
            .filter_map(|e| e.ok())
            .filter(|e| oss_module(&e.path()).is_some())
            .filter_map(|e| e.file_name().into_string().ok())
            .collect();
        v.sort();
        v
    }

    pub fn oss_bot_dir(&self, name: &str) -> Option<PathBuf> {
        let d = self.oss_bots_dir.as_ref()?.join(name);
        oss_module(&d).map(|_| d)
    }
}

/// `module.so` on Linux, `module.dll` on Windows.
pub fn oss_module(bot_dir: &Path) -> Option<PathBuf> {
    let f = bot_dir.join(if cfg!(windows) { "module.dll" } else { "module.so" });
    f.is_file().then_some(f)
}
