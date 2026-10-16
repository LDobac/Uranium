//! Host settings from `.env` and the process environment (001-R9, 043 Interfaces).

use std::path::PathBuf;
use std::str::FromStr;

use anyhow::{Context, Result, bail};

/// 043-R4: combined request rate (local API + replay downloads), requests per second.
pub const DEFAULT_RPS: f64 = 0.5;
/// 043-R4: replay downloads per local calendar day.
pub const DEFAULT_DAILY_MAX: u32 = 1000;

/// Loads host settings from `.env` (or `$URANIUM_ENV_FILE`) before anything reads the
/// environment; variables already set in the process win (spec 001-R9).
pub fn load_env() {
    let res = match std::env::var_os("URANIUM_ENV_FILE") {
        Some(path) => dotenvy::from_path(path),
        None => dotenvy::dotenv().map(|_| ()),
    };
    if let Err(e) = res {
        if !e.not_found() {
            eprintln!("warning: failed to load .env: {e}");
        }
    }
}

#[derive(Clone, Debug)]
pub struct Settings {
    /// 043-R3: fixed local API port; `None` means detect it from the StarCraft process.
    pub port: Option<u16>,
    pub rps: f64,
    pub daily_max: u32,
    pub shared_dir: Option<PathBuf>,
}

impl Settings {
    pub fn from_env() -> Result<Self> {
        let rps = parse_opt("URANIUM_LADDER_RPS")?.unwrap_or(DEFAULT_RPS);
        if !(rps > 0.0 && rps.is_finite()) {
            bail!("URANIUM_LADDER_RPS must be a positive number");
        }
        Ok(Self {
            port: parse_opt("URANIUM_SCR_API_PORT")?,
            rps,
            daily_max: parse_opt("URANIUM_LADDER_DAILY_MAX")?.unwrap_or(DEFAULT_DAILY_MAX),
            shared_dir: std::env::var_os("URANIUM_SHARED_DIR").filter(|s| !s.is_empty()).map(PathBuf::from),
        })
    }

    /// 043-R8: `<URANIUM_SHARED_DIR>/datasets/replays/ladder` (collector state, kept out of git).
    pub fn ladder_dir(&self) -> Result<PathBuf> {
        let shared = self.shared_dir.as_ref().context("URANIUM_SHARED_DIR is not set")?;
        Ok(shared.join("datasets").join("replays").join("ladder"))
    }
}

fn parse_opt<T: FromStr>(key: &str) -> Result<Option<T>>
where
    T::Err: std::fmt::Display,
{
    match std::env::var(key) {
        Ok(v) if !v.trim().is_empty() => {
            v.trim().parse().map(Some).map_err(|e| anyhow::anyhow!("{key}={v:?}: {e}"))
        }
        _ => Ok(None),
    }
}
