//! 043-R4: request pacing and the daily download budget.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Spaces every request (local API and downloads together) at least `1/rps` apart.
pub struct Limiter {
    interval: Duration,
    next: Option<Instant>,
}

impl Limiter {
    pub fn new(rps: f64) -> Self {
        Self { interval: Duration::from_secs_f64(1.0 / rps), next: None }
    }

    /// Reserves the next request slot at or after `now`; returns how long to wait for it.
    pub fn reserve(&mut self, now: Instant) -> Duration {
        let at = match self.next {
            Some(n) if n > now => n,
            _ => now,
        };
        self.next = Some(at + self.interval);
        at - now
    }

    pub fn wait(&mut self) {
        let d = self.reserve(Instant::now());
        if !d.is_zero() {
            std::thread::sleep(d);
        }
    }
}

#[derive(Serialize, Deserialize, Default, PartialEq, Debug)]
struct DailyFile {
    date: String,
    downloads: u32,
}

/// Replay downloads counted per local calendar day, persisted so restarts share the budget.
pub struct Daily {
    path: PathBuf,
    max: u32,
    state: DailyFile,
}

impl Daily {
    pub fn load(path: &Path, max: u32, today: chrono::NaiveDate) -> Result<Self> {
        let date = today.format("%Y-%m-%d").to_string();
        let state = match std::fs::read_to_string(path) {
            Ok(s) => serde_json::from_str::<DailyFile>(&s).with_context(|| format!("{}", path.display()))?,
            Err(_) => DailyFile::default(),
        };
        let state = if state.date == date { state } else { DailyFile { date, downloads: 0 } };
        Ok(Self { path: path.to_path_buf(), max, state })
    }

    pub fn load_today(path: &Path, max: u32) -> Result<Self> {
        Self::load(path, max, chrono::Local::now().date_naive())
    }

    pub fn remaining(&self) -> u32 {
        self.max.saturating_sub(self.state.downloads)
    }

    /// Counts one finished download and saves the counter.
    pub fn record(&mut self) -> Result<()> {
        self.state.downloads += 1;
        if let Some(dir) = self.path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(&self.path, serde_json::to_string(&self.state)?)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn r043_r4_limiter_spaces_requests() {
        let mut l = Limiter::new(0.5);
        let t0 = Instant::now();
        assert_eq!(l.reserve(t0), Duration::ZERO);
        assert_eq!(l.reserve(t0), Duration::from_secs(2));
        assert_eq!(l.reserve(t0 + Duration::from_secs(1)), Duration::from_secs(3));
        // After an idle gap the next request goes out at once.
        assert_eq!(l.reserve(t0 + Duration::from_secs(60)), Duration::ZERO);
    }

    #[test]
    fn r043_r4_daily_budget_resets_on_new_day() {
        let dir = std::env::temp_dir().join(format!("uranium-ladder-daily-{}", std::process::id()));
        let path = dir.join("daily.json");
        let _ = std::fs::remove_dir_all(&dir);
        let d1 = chrono::NaiveDate::from_ymd_opt(2026, 10, 16).unwrap();
        let mut d = Daily::load(&path, 2, d1).unwrap();
        d.record().unwrap();
        assert_eq!(Daily::load(&path, 2, d1).unwrap().remaining(), 1);
        let mut d = Daily::load(&path, 2, d1).unwrap();
        d.record().unwrap();
        assert_eq!(d.remaining(), 0);
        assert_eq!(Daily::load(&path, 2, d1.succ_opt().unwrap()).unwrap().remaining(), 2);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
