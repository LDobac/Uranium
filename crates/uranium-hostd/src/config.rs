//! hostd settings from `.env` / process environment (spec 020-R9, 001-R9).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

#[derive(Debug, Clone)]
pub struct Config {
    pub listen: String,
    /// BWAPI build dir containing `bin/BWAPILauncher` (Windows: `bin/Release/BWAPILauncher.exe`).
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
    /// Wall-time limit of one replay playback job (020-R3, M4).
    pub replay_timeout_s: u32,
    /// Where saved game replays go (040 botgen); default: the game work dir.
    pub replay_dir: Option<PathBuf>,
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
            max_games: max_games(var("URANIUM_HOSTD_MAX_GAMES"), var("URANIUM_CACHE_DIR").map(PathBuf::from), cores).0,
            work_dir: var("URANIUM_HOSTD_WORK_DIR").map(PathBuf::from).unwrap_or_else(|| std::env::temp_dir().join("uranium-hostd")),
            oss_bots_dir: var("URANIUM_OSS_BOTS_DIR").map(PathBuf::from),
            port_base: var("URANIUM_HOSTD_PORT_BASE").and_then(|v| v.parse().ok()).unwrap_or(17_000),
            replay_timeout_s: var("URANIUM_HOSTD_REPLAY_TIMEOUT_S").and_then(|v| v.parse().ok()).unwrap_or(1800),
            replay_dir: var("URANIUM_HOSTD_REPLAY_DIR").map(PathBuf::from),
        };
        must_exist(&cfg.launcher(), "BWAPILauncher")?;
        must_exist(&cfg.bridge_lib, "bridge library")?;
        must_exist(&cfg.data_dir, "data dir")?;
        must_exist(&cfg.maps_dir, "maps dir")?;
        std::fs::create_dir_all(&cfg.work_dir)?;
        Ok(cfg)
    }

    pub fn launcher(&self) -> PathBuf {
        // MSVC generators are multi-config: outputs go to bin/Release (tools/openbw/build.ps1).
        if cfg!(windows) {
            self.bwapi_dir.join("bin").join("Release").join("BWAPILauncher.exe")
        } else {
            self.bwapi_dir.join("bin").join("BWAPILauncher")
        }
    }

    /// Registered third-party bots (directories containing a module file) as opponent IDs
    /// `<name>@<version>` from the `VERSION` file the build script writes (021-R3).
    pub fn oss_bots(&self) -> Vec<String> {
        let Some(dir) = &self.oss_bots_dir else { return vec![] };
        let Ok(rd) = std::fs::read_dir(dir) else { return vec![] };
        let mut v: Vec<String> = rd
            .filter_map(|e| e.ok())
            .filter(|e| oss_module(&e.path()).is_some())
            .filter_map(|e| {
                let name = e.file_name().into_string().ok()?;
                let ver = std::fs::read_to_string(e.path().join("VERSION")).ok().map(|v| v.trim().to_string()).filter(|v| v.starts_with(&format!("{name}@")));
                Some(ver.unwrap_or(name))
            })
            .collect();
        v.sort();
        v
    }

    pub fn oss_bot_dir(&self, name: &str) -> Option<PathBuf> {
        let d = self.oss_bots_dir.as_ref()?.join(name);
        oss_module(&d).map(|_| d)
    }
}

/// Where the concurrency limit came from (020-R10).
#[derive(Debug, PartialEq, Eq)]
pub enum MaxGamesSource {
    Env,
    Calibration,
    Default,
}

#[derive(serde::Deserialize)]
struct CalibrationFile {
    max_games: u32,
    measured_at: Option<String>,
}

/// 020-R10: `URANIUM_HOSTD_MAX_GAMES` → `<cache>/hostd_calibration.toml` (written by
/// `uranium-env calibrate`) → conservative default `max(1, (cores - 1) / 2)`.
pub fn max_games(env: Option<String>, cache_dir: Option<PathBuf>, cores: u32) -> (u32, MaxGamesSource) {
    if let Some(n) = env.and_then(|v| v.parse::<u32>().ok()).filter(|n| *n > 0) {
        eprintln!("[hostd] max concurrent games {n} (URANIUM_HOSTD_MAX_GAMES)");
        return (n, MaxGamesSource::Env);
    }
    if let Some(path) = cache_dir.map(|d| d.join("hostd_calibration.toml")) {
        match std::fs::read_to_string(&path).map(|t| toml::from_str::<CalibrationFile>(&t)) {
            Ok(Ok(f)) if f.max_games > 0 => {
                eprintln!("[hostd] max concurrent games {} (calibrated {})", f.max_games, f.measured_at.as_deref().unwrap_or("?"));
                return (f.max_games, MaxGamesSource::Calibration);
            }
            Ok(_) => eprintln!("[hostd] warning: ignoring malformed {}", path.display()),
            Err(_) => {}
        }
    }
    let n = (cores.saturating_sub(1) / 2).max(1);
    eprintln!("[hostd] warning: max concurrent games {n} (uncalibrated default); run `uranium-env calibrate` on this host (020-R10)");
    (n, MaxGamesSource::Default)
}

/// `module.so` on Linux, `module.dll` on Windows.
pub fn oss_module(bot_dir: &Path) -> Option<PathBuf> {
    let f = bot_dir.join(if cfg!(windows) { "module.dll" } else { "module.so" });
    f.is_file().then_some(f)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_games_priority_020_r10() {
        let dir = std::env::temp_dir().join(format!("uranium-hostd-cal-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(max_games(None, Some(dir.clone()), 12), (5, MaxGamesSource::Default));
        assert_eq!(max_games(None, None, 2), (1, MaxGamesSource::Default));
        std::fs::write(dir.join("hostd_calibration.toml"), "max_games = 4\nmeasured_at = \"2026-10-16\"\ncores = 12\n").unwrap();
        assert_eq!(max_games(None, Some(dir.clone()), 12), (4, MaxGamesSource::Calibration));
        assert_eq!(max_games(Some("7".into()), Some(dir.clone()), 12), (7, MaxGamesSource::Env));
        assert_eq!(max_games(Some("x".into()), Some(dir.clone()), 12), (4, MaxGamesSource::Calibration));
        std::fs::write(dir.join("hostd_calibration.toml"), "nonsense").unwrap();
        assert_eq!(max_games(None, Some(dir.clone()), 12), (5, MaxGamesSource::Default));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
