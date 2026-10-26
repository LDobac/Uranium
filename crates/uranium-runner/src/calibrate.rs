//! `uranium-env calibrate` (spec 020-R10): finds the largest number of concurrent games this
//! host sustains while the bridge latency stays within the 011-AC2 target, and stores it in
//! `<URANIUM_CACHE_DIR>/hostd_calibration.toml` for uranium-hostd.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use serde::Serialize;
use serde_json::Value;

use crate::plan::{self, HostEntry, MapSel, Matchup, Plan, RaceSel, Side};

/// 011-AC2: bridge overhead p99 per frame <= 0.3 ms. Judged on the median over sessions.
pub const TARGET_US: f64 = 300.0;
pub const FILE_NAME: &str = "hostd_calibration.toml";

#[derive(Serialize, Debug, Clone, PartialEq)]
pub struct Step {
    pub games: u32,
    pub sessions: usize,
    pub aborted: usize,
    pub serialize_p99_median_us: f64,
    pub serialize_p99_max_us: f64,
    pub engine_fps_p50: f64,
    /// Game frames simulated per wall-clock second over all concurrent games.
    pub throughput_fps: f64,
}

impl Step {
    pub fn passes(&self) -> bool {
        self.aborted == 0 && self.sessions > 0 && self.serialize_p99_median_us <= TARGET_US
    }
}

#[derive(Serialize)]
struct CalibrationFile<'a> {
    max_games: u32,
    measured_at: String,
    cores: u32,
    criterion: String,
    frames_per_game: u32,
    step: &'a [Step],
}

/// Largest passing concurrency, measured in increasing order; stops at the first failure.
pub fn recommend(steps: &[Step]) -> u32 {
    steps.iter().take_while(|s| s.passes()).map(|s| s.games).last().unwrap_or(1)
}

fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// Summarizes one step from its session records and hostd outcomes.
pub fn summarize(games: u32, sessions: &[Value], outcomes: &[Value], wall_s: f64) -> Step {
    let mut p99: Vec<f64> = sessions.iter().filter_map(|s| s["stats"]["serialize_us_p99"].as_f64()).collect();
    let mut fps: Vec<f64> = sessions
        .iter()
        .filter_map(|s| {
            let (gf, gt) = (s["stats"]["game_frames"].as_f64()?, s["stats"]["game_time_ms"].as_f64()?);
            (gt > 0.0).then(|| gf / (gt / 1e3))
        })
        .collect();
    // Both sides of a T0 game report the same frames: count each game once (slot 0).
    let frames: f64 = sessions.iter().filter(|s| s["slot"].as_u64() == Some(0)).filter_map(|s| s["end_frame"].as_f64()).sum();
    let aborted = outcomes.iter().filter(|g| g["outcome"] != "Finished").count();
    let max = p99.iter().copied().fold(0.0, f64::max);
    Step {
        games,
        sessions: sessions.len(),
        aborted,
        serialize_p99_median_us: median(&mut p99),
        serialize_p99_max_us: max,
        engine_fps_p50: median(&mut fps),
        throughput_fps: if wall_s > 0.0 { frames / wall_s } else { 0.0 },
    }
}

/// 2N games of T0 cross-race play so every slot is busy; races rotate T-P, T-Z, P-Z.
fn step_plan(n: u32, frames: u32) -> Plan {
    let pairs = [(RaceSel::Terran, RaceSel::Protoss), (RaceSel::Terran, RaceSel::Zerg), (RaceSel::Protoss, RaceSel::Zerg)];
    let total = 2 * n;
    let matchups = pairs
        .iter()
        .enumerate()
        .map(|(i, (a, b))| Matchup {
            map: MapSel::Both,
            games: total / 3 + u32::from((i as u32) < total % 3),
            p0: Side::T0 { race: *a, build: Default::default(), name: None },
            p1: Side::T0 { race: *b, build: Default::default(), name: None },
            p1_random_ratio: 0.0,
        })
        .filter(|m| m.games > 0)
        .collect();
    Plan {
        name: format!("calibrate-n{n}"),
        seed_base: 50_000,
        same_seed: false,
        max_frames: frames,
        wall_timeout_s: 600,
        privileged: false,
        save_replay: false,
        zero_last_failed: false,
        surrender: uranium_agent::infer::SurrenderMode::Off,
        surrender_chat: uranium_agent::infer::DEFAULT_CHAT.into(),
        matchups,
    }
}

fn read_jsonl(path: &Path) -> Vec<Value> {
    std::fs::read_to_string(path).unwrap_or_default().lines().filter_map(|l| serde_json::from_str(l).ok()).collect()
}

fn free_port() -> Result<u16> {
    Ok(std::net::TcpListener::bind("127.0.0.1:0")?.local_addr()?.port())
}

struct Hostd(Child);

impl Drop for Hostd {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Starts `uranium-hostd serve` from the same build directory with a fixed concurrency.
fn spawn_hostd(n: u32, addr: &str, work: &Path) -> Result<Hostd> {
    let exe = std::env::current_exe()?.with_file_name(if cfg!(windows) { "uranium-hostd.exe" } else { "uranium-hostd" });
    if !exe.is_file() {
        bail!("{} not found: build it with `cargo build --release -p uranium-hostd`", exe.display());
    }
    let log = std::fs::File::create(work.join("hostd.log"))?;
    let child = Command::new(&exe)
        .arg("serve")
        .env("URANIUM_HOSTD_LISTEN", addr)
        .env("URANIUM_HOSTD_MAX_GAMES", n.to_string())
        .env("URANIUM_HOSTD_WORK_DIR", work.join("games"))
        .stdout(Stdio::null())
        .stderr(log)
        .spawn()
        .with_context(|| format!("starting {}", exe.display()))?;
    let hostd = Hostd(child);
    let t = Instant::now();
    while t.elapsed() < Duration::from_secs(20) {
        if crate::capabilities(addr).is_ok() {
            return Ok(hostd);
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    bail!("hostd did not come up; see {}", work.join("hostd.log").display())
}

/// Civil date in UTC for the file header; no date crate needed.
fn today() -> String {
    let days = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs() / 86_400).unwrap_or(0) as i64;
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02} UTC")
}

fn table(steps: &[Step]) -> String {
    let mut s = String::from("| games | sessions | aborted | bridge p99 median (µs) | max (µs) | engine fps p50 | throughput (frames/s) | pass |\n|---|---|---|---|---|---|---|---|\n");
    for st in steps {
        let _ = writeln!(
            s,
            "| {} | {} | {} | {:.0} | {:.0} | {:.0} | {:.0} | {} |",
            st.games,
            st.sessions,
            st.aborted,
            st.serialize_p99_median_us,
            st.serialize_p99_max_us,
            st.engine_fps_p50,
            st.throughput_fps,
            if st.passes() { "yes" } else { "no" }
        );
    }
    s
}

pub fn run(max: Option<u32>, frames: u32) -> Result<()> {
    let cache = std::env::var("URANIUM_CACHE_DIR")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .map(PathBuf::from)
        .context("URANIUM_CACHE_DIR is not set: add it to .env (see .env.example, 020-R10)")?;
    let cores = std::thread::available_parallelism().map(|n| n.get() as u32).unwrap_or(2);
    let max = max.unwrap_or(cores).max(1);
    let work = cache.join("calibrate");
    let _ = std::fs::remove_dir_all(&work);
    std::fs::create_dir_all(&work)?;

    let results = work.join("results.jsonl");
    let agent = crate::start_agent("127.0.0.1:0", &results, Default::default(), Default::default(), Default::default())?;
    eprintln!("[calibrate] {cores} logical cores, trying 1..={max} concurrent games, {frames} frames per game");

    let mut steps = Vec::new();
    let mut id_base = 0u64;
    for n in 1..=max {
        let dir = work.join(format!("n{n}"));
        std::fs::create_dir_all(&dir)?;
        let addr = format!("127.0.0.1:{}", free_port()?);
        let _hostd = spawn_hostd(n, &addr, &dir)?;
        let mut jobs = plan::expand(&step_plan(n, frames), &agent);
        for j in &mut jobs {
            j.spec.game_id += id_base;
        }
        let ids: std::collections::HashSet<u64> = jobs.iter().map(|j| j.spec.game_id).collect();
        id_base += jobs.len() as u64;
        let t = Instant::now();
        let left = crate::run_jobs(jobs, &[HostEntry { addr, max_games: Some(n) }], &dir.join("games.jsonl"), &uranium_agent::Routes::default(), &crate::Models::new())?;
        let wall = t.elapsed().as_secs_f64();
        let sessions: Vec<Value> = read_jsonl(&results).into_iter().filter(|s| s["game_id"].as_u64().is_some_and(|g| ids.contains(&g))).collect();
        let mut step = summarize(n, &sessions, &read_jsonl(&dir.join("games.jsonl")), wall);
        step.aborted += left;
        eprintln!(
            "[calibrate] {n} games: bridge p99 median {:.0}µs (max {:.0}), aborted {}, {:.0} frames/s -> {}",
            step.serialize_p99_median_us,
            step.serialize_p99_max_us,
            step.aborted,
            step.throughput_fps,
            if step.passes() { "pass" } else { "fail" }
        );
        let pass = step.passes();
        steps.push(step);
        if !pass {
            break;
        }
    }

    let best = recommend(&steps);
    if !steps.first().is_some_and(Step::passes) {
        eprintln!("[calibrate] warning: even 1 game misses the {TARGET_US}µs target; using 1");
    }
    let file = CalibrationFile {
        max_games: best,
        measured_at: today(),
        cores,
        criterion: format!("serialize_p99_median_us <= {TARGET_US}"),
        frames_per_game: frames,
        step: &steps,
    };
    let path = cache.join(FILE_NAME);
    std::fs::write(&path, toml::to_string(&file)?)?;
    println!("{}", table(&steps));
    println!("recommended concurrent games: {best} (written to {})", path.display());
    println!("uranium-hostd uses it on its next start unless URANIUM_HOSTD_MAX_GAMES is set.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn step(games: u32, p99: f64, aborted: usize) -> Step {
        Step { games, sessions: 4, aborted, serialize_p99_median_us: p99, serialize_p99_max_us: p99, engine_fps_p50: 0.0, throughput_fps: 0.0 }
    }

    #[test]
    fn recommend_takes_largest_passing_prefix_020_r10() {
        assert_eq!(recommend(&[step(1, 120.0, 0), step(2, 200.0, 0), step(3, 290.0, 0), step(4, 900.0, 0)]), 3);
        // A pass after a failure does not count (measurement stops at the first failure).
        assert_eq!(recommend(&[step(1, 120.0, 0), step(2, 400.0, 0), step(3, 200.0, 0)]), 1);
        assert_eq!(recommend(&[step(1, 120.0, 0), step(2, 150.0, 1)]), 1);
        assert_eq!(recommend(&[step(1, 500.0, 0)]), 1);
        assert_eq!(recommend(&[]), 1);
    }

    #[test]
    fn step_plan_fills_every_slot_twice() {
        for n in 1..=7 {
            let p = step_plan(n, 8000);
            assert_eq!(p.matchups.iter().map(|m| m.games).sum::<u32>(), 2 * n);
        }
    }

    #[test]
    fn summarize_uses_median_of_session_p99() {
        let s = |p: f64, slot: u64| serde_json::json!({"slot": slot, "end_frame": 1000, "stats": {"serialize_us_p99": p, "game_frames": 1000.0, "game_time_ms": 500.0}});
        let sessions = [s(100.0, 0), s(400.0, 1), s(200.0, 0)];
        let games = [serde_json::json!({"outcome": "Finished"}), serde_json::json!({"outcome": "Aborted"})];
        let st = summarize(2, &sessions, &games, 1.0);
        assert_eq!(st.serialize_p99_median_us, 200.0);
        assert_eq!(st.serialize_p99_max_us, 400.0);
        assert_eq!(st.aborted, 1);
        assert_eq!(st.throughput_fps, 2000.0);
        assert_eq!(st.engine_fps_p50, 2000.0);
    }

    #[test]
    fn today_is_a_date() {
        let d = today();
        assert_eq!(d.len(), 14);
        assert!(d.starts_with("20"));
    }
}
