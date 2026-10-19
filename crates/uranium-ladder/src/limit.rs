//! 043-R4: request pacing and the daily download budget.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Spaces every request (local API and downloads together) at least `1/rps` apart.
pub struct Limiter {
    interval: Duration,
    next: Option<Instant>,
    /// When each paced request was let through (043-AC4 log).
    sent: Vec<Instant>,
}

impl Limiter {
    pub fn new(rps: f64) -> Self {
        Self { interval: Duration::from_secs_f64(1.0 / rps), next: None, sent: Vec::new() }
    }

    pub fn interval(&self) -> Duration {
        self.interval
    }

    /// 043-AC4: rate actually used by the requests let through so far.
    pub fn stats(&self) -> RateStats {
        RateStats::of(&self.sent)
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
        self.sent.push(Instant::now());
    }
}

/// 043-AC4: observed request rate of one run, printed at the end of the run.
#[derive(Debug, Clone, PartialEq)]
pub struct RateStats {
    pub requests: usize,
    /// Shortest time between two consecutive requests.
    pub min_gap: Option<Duration>,
    /// Most requests inside any one-second window `[t, t + 1 s)`.
    pub max_per_second: usize,
}

impl RateStats {
    /// `sent` in send order.
    pub fn of(sent: &[Instant]) -> Self {
        let min_gap = sent.windows(2).map(|w| w[1].saturating_duration_since(w[0])).min();
        let mut max_per_second = 0;
        let mut lo = 0;
        for hi in 0..sent.len() {
            while sent[hi].saturating_duration_since(sent[lo]) >= Duration::from_secs(1) {
                lo += 1;
            }
            max_per_second = max_per_second.max(hi - lo + 1);
        }
        Self { requests: sent.len(), min_gap, max_per_second }
    }

    /// No one-second window holds more than `ceil(rps)` requests (043-R4: at most `rps` per
    /// second), and no two are closer than `1/rps` up to 5 % of sleep wake-up jitter (the slots
    /// themselves are exactly `1/rps` apart; a late wake-up shortens the next gap a little).
    pub fn within(&self, rps: f64) -> bool {
        let interval = Duration::from_secs_f64(0.95 / rps);
        self.min_gap.is_none_or(|g| g >= interval) && self.max_per_second <= rps.ceil().max(1.0) as usize
    }

    /// One log line (no URLs or identifiers, 043-R11).
    pub fn line(&self, rps: f64) -> String {
        let gap = self.min_gap.map_or("-".to_string(), |g| format!("{:.3} s", g.as_secs_f64()));
        let verdict = if self.within(rps) { "within" } else { "OVER" };
        format!("requests: {} paced, min gap {gap}, max {} in any 1 s, limit {rps} req/s: {verdict} (043-R4, 043-AC4)", self.requests, self.max_per_second)
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

    /// 043-AC4: the per-run rate summary from send times (fake clock).
    #[test]
    fn r043_ac4_rate_stats_from_send_times() {
        let t0 = Instant::now();
        let at = |ms: u64| t0 + Duration::from_millis(ms);
        // Limiter slots at 0.5 req/s: 2 s apart, never two in one second.
        let mut l = Limiter::new(0.5);
        let mut now = t0;
        let mut sent = vec![];
        for _ in 0..5 {
            now += l.reserve(now);
            sent.push(now);
        }
        let s = RateStats::of(&sent);
        assert_eq!((s.requests, s.min_gap, s.max_per_second), (5, Some(Duration::from_secs(2)), 1));
        assert!(s.within(0.5));
        assert!(s.line(0.5).contains("within"));
        // Two requests 0.4 s apart break a 0.5 req/s limit.
        let s = RateStats::of(&[at(0), at(2000), at(2400)]);
        assert_eq!(s.max_per_second, 2);
        assert!(!s.within(0.5));
        assert!(s.line(0.5).contains("OVER"));
        assert!(RateStats::of(&[]).within(0.5));
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
