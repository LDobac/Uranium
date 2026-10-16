//! `uranium-env`: game scheduler (spec 020). Runs the agent server in-process, spreads the
//! games of a matchup plan over one or more `uranium-hostd`s and writes results.

mod calibrate;
mod plan;
mod report;

use std::collections::{HashMap, VecDeque};
use std::io::{BufReader, BufWriter, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use serde::Serialize;
use uranium_agent::{PolicyKind, ServeOptions, load_env, open_sink, serve};
use uranium_proto::control::*;
use uranium_proto::{read_msg, write_msg};

use crate::plan::{HostEntry, Job};

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Play every game of a matchup plan.
    Run {
        #[arg(long)]
        plan: PathBuf,
        /// Output directory (default runs/<plan name>).
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Measure how many concurrent games this host sustains within the bridge latency
    /// target and store it for uranium-hostd (020-R10). Run on the game host itself.
    Calibrate {
        /// Largest concurrency to try (default: logical cores).
        #[arg(long)]
        max: Option<u32>,
        /// Frame limit per game.
        #[arg(long, default_value_t = 8000)]
        frames: u32,
    },
}

/// One line of games.jsonl (hostd's view of a game).
#[derive(Serialize)]
struct GameRecord {
    game_id: u64,
    matchup: usize,
    host: String,
    map: String,
    races: [String; 2],
    p1_kind: String,
    /// Opponent ID `<name>@<version>` (021-R3).
    opponent: String,
    outcome: String,
    detail: Option<String>,
    wall_s: u32,
}

fn env_or(name: &str, default: &str) -> String {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty()).unwrap_or_else(|| default.to_string())
}

fn main() -> Result<()> {
    load_env();
    match Cli::parse().cmd {
        Cmd::Run { plan, out } => run(&plan, out),
        Cmd::Calibrate { max, frames } => calibrate::run(max, frames),
    }
}

fn hosts() -> Result<Vec<HostEntry>> {
    match std::env::var("URANIUM_HOSTS_FILE").ok().filter(|v| !v.is_empty()) {
        Some(f) => plan::load_hosts(f.as_ref()),
        None => Ok(vec![HostEntry { addr: env_or("URANIUM_HOSTD_LISTEN", "127.0.0.1:7700"), max_games: None }]),
    }
}

fn run(plan_path: &std::path::Path, out: Option<PathBuf>) -> Result<()> {
    let plan = plan::load_plan(plan_path)?;
    let out = out.unwrap_or_else(|| PathBuf::from("runs").join(&plan.name));
    std::fs::create_dir_all(&out)?;
    std::fs::copy(plan_path, out.join("plan.toml"))?;
    let _ = std::fs::remove_file(out.join("games.jsonl"));
    let _ = std::fs::remove_file(out.join("results.jsonl"));

    let public = start_agent(&env_or("URANIUM_AGENT_LISTEN", "127.0.0.1:7600"), &out.join("results.jsonl"))?;
    let jobs = plan::expand(&plan, &public);
    eprintln!("[env] plan `{}`: {} games, agent at {public}", plan.name, jobs.len());
    let started = Instant::now();
    let left = run_jobs(jobs, &hosts()?, &out.join("games.jsonl"))?;
    eprintln!("[env] finished in {:.0}s ({left} games not run)", started.elapsed().as_secs_f64());
    report::write(&out, &plan)?;
    println!("{}", std::fs::read_to_string(out.join("summary.md"))?);
    if left > 0 {
        bail!("{left} games could not be scheduled");
    }
    Ok(())
}

/// Starts the embedded agent server (001-R6) and returns the address bridges connect to.
/// Bridges on every host must reach it (`URANIUM_AGENT_PUBLIC_ADDR` when it differs).
pub(crate) fn start_agent(listen: &str, results: &std::path::Path) -> Result<String> {
    let listener = TcpListener::bind(listen).with_context(|| format!("binding agent {listen}"))?;
    let bound = listener.local_addr()?.to_string();
    let public = std::env::var("URANIUM_AGENT_PUBLIC_ADDR").ok().filter(|v| !v.trim().is_empty()).unwrap_or(bound);
    let sink = open_sink(results)?;
    std::thread::spawn(move || {
        if let Err(e) = serve(listener, ServeOptions { policy: PolicyKind::ScriptedT0, delay_ms: 0, max_sessions: 0 }, sink) {
            eprintln!("[env] agent server stopped: {e:#}");
        }
    });
    Ok(public)
}

/// Plays `jobs` on `hosts`, appending hostd outcomes to `games_path`. Returns the number of
/// games that could not be scheduled.
pub(crate) fn run_jobs(jobs: Vec<Job>, hosts: &[HostEntry], games_path: &std::path::Path) -> Result<usize> {
    let total = jobs.len();
    let queue = Arc::new(Mutex::new(VecDeque::from(jobs)));
    let file = std::fs::OpenOptions::new().create(true).append(true).open(games_path)?;
    let games_out = Arc::new(Mutex::new(BufWriter::new(file)));
    let done = Arc::new(Mutex::new(0usize));
    let mut workers = Vec::new();
    for host in hosts {
        // 020-R4: an unreachable host is skipped, the others still play the plan.
        let caps = match capabilities(&host.addr) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[env] WARNING: hostd {} unreachable, skipping it: {e:#}", host.addr);
                continue;
            }
        };
        let slots = host.max_games.map_or(caps.max_concurrent_games, |m| m.min(caps.max_concurrent_games));
        eprintln!("[env] host {}: {:?}, {} slots, oss bots {:?}", host.addr, caps.os, slots, caps.oss_bots);
        let ids: Arc<HashMap<String, String>> = Arc::new(caps.oss_bots.iter().map(|id| (id.split('@').next().unwrap_or(id).to_string(), id.clone())).collect());
        for _ in 0..slots {
            let (addr, q, g, d, ids) = (host.addr.clone(), queue.clone(), games_out.clone(), done.clone(), ids.clone());
            workers.push(std::thread::spawn(move || worker(&addr, &q, &g, &d, &ids, total)));
        }
    }
    if workers.is_empty() {
        bail!("no reachable hostd among {} configured host(s)", hosts.len());
    }
    for w in workers {
        if let Err(e) = w.join().expect("worker panicked") {
            eprintln!("[env] worker failed: {e:#}");
        }
    }
    games_out.lock().unwrap().flush()?;
    // Session records are written when each bridge sends GameEnd, before its launcher exits.
    std::thread::sleep(Duration::from_secs(1));
    let left = queue.lock().unwrap().len();
    Ok(left)
}

fn connect(addr: &str) -> Result<(BufReader<TcpStream>, BufWriter<TcpStream>)> {
    let s = TcpStream::connect(addr)?;
    s.set_nodelay(true).ok();
    // A game host that powers off never closes the socket; keepalive turns that into a read
    // error within a few minutes instead of a worker blocked forever (020-R4).
    let ka = socket2::TcpKeepalive::new().with_time(Duration::from_secs(30)).with_interval(Duration::from_secs(10));
    socket2::SockRef::from(&s).set_tcp_keepalive(&ka).ok();
    let mut r = BufReader::new(s.try_clone()?);
    let mut w = BufWriter::new(s);
    write_msg(&mut w, &ControlRequest::Hello { control_version: CONTROL_VERSION })?;
    match read_msg(&mut r)? {
        ControlResponse::Hello { .. } => Ok((r, w)),
        other => bail!("handshake failed: {other:?}"),
    }
}

pub(crate) fn capabilities(addr: &str) -> Result<Capabilities> {
    let (mut r, mut w) = connect(addr)?;
    write_msg(&mut w, &ControlRequest::ListCapabilities)?;
    match read_msg(&mut r)? {
        ControlResponse::Capabilities(c) => Ok(c),
        other => bail!("unexpected {other:?}"),
    }
}

fn worker(addr: &str, queue: &Mutex<VecDeque<Job>>, games: &Mutex<BufWriter<std::fs::File>>, done: &Mutex<usize>, ids: &HashMap<String, String>, total: usize) -> Result<()> {
    let (mut r, mut w) = connect(addr)?;
    loop {
        let Some(job) = queue.lock().unwrap().pop_front() else { return Ok(()) };
        let reply = write_msg(&mut w, &ControlRequest::StartGame(job.spec.clone())).and_then(|_| read_msg(&mut r));
        let reply = match reply {
            Ok(m) => m,
            Err(e) => {
                // The game never started: leave it for the remaining hosts (020-R4).
                queue.lock().unwrap().push_front(job);
                return Err(anyhow::Error::from(e).context(format!("hostd {addr} lost")));
            }
        };
        match reply {
            ControlResponse::Started { .. } => {}
            ControlResponse::Rejected { reason } => {
                eprintln!("[env] {addr} rejected game {}: {reason}", job.spec.game_id);
                if reason.contains("capacity") {
                    queue.lock().unwrap().push_back(job);
                    std::thread::sleep(Duration::from_secs(2));
                    continue;
                }
                record(games, addr, &job, ids, "Rejected", Some(reason), 0)?;
                continue;
            }
            other => bail!("unexpected {other:?}"),
        }
        let fin = loop {
            match read_msg(&mut r) {
                Ok(ControlResponse::Finished(f)) if f.game_id == job.spec.game_id => break f,
                Ok(other) => eprintln!("[env] ignoring {other:?}"),
                Err(e) => {
                    // 020-R4: a game in flight on a lost host is Aborted, not silently dropped.
                    record(games, addr, &job, ids, "Aborted", Some(format!("hostd connection lost: {e}")), 0)?;
                    *done.lock().unwrap() += 1;
                    eprintln!("[env] game {} Aborted: hostd {addr} lost", job.spec.game_id);
                    return Err(anyhow::Error::from(e).context(format!("hostd {addr} lost")));
                }
            }
        };
        let (outcome, detail) = match &fin.outcome {
            GameOutcome::Finished => ("Finished", None),
            GameOutcome::Aborted { reason } => ("Aborted", Some(reason.clone())),
            GameOutcome::Timeout => ("Timeout", None),
            GameOutcome::Stopped => ("Stopped", None),
        };
        record(games, addr, &job, ids, outcome, detail, fin.wall_s)?;
        let mut d = done.lock().unwrap();
        *d += 1;
        eprintln!("[env] {}/{total} game {} {outcome} ({}s)", *d, job.spec.game_id, fin.wall_s);
    }
}

/// 021-R3: T0 bots are versioned with the workspace, OSS bots by the host's VERSION file.
fn opponent_id(p: &PlayerSpec, ids: &HashMap<String, String>) -> String {
    match &p.kind {
        PlayerKind::Bridge { .. } => format!("t0-{}@{}", format!("{:?}", p.race).to_lowercase(), env!("CARGO_PKG_VERSION")),
        PlayerKind::OssBot { name } => ids.get(name).cloned().unwrap_or_else(|| format!("{name}@unknown")),
    }
}

#[allow(clippy::too_many_arguments)]
fn record(games: &Mutex<BufWriter<std::fs::File>>, host: &str, job: &Job, ids: &HashMap<String, String>, outcome: &str, detail: Option<String>, wall_s: u32) -> Result<()> {
    let p1_kind = match &job.spec.players[1].kind {
        PlayerKind::Bridge { .. } => "t0".to_string(),
        PlayerKind::OssBot { name } => format!("oss:{name}"),
    };
    let rec = GameRecord {
        game_id: job.spec.game_id,
        matchup: job.matchup,
        host: host.to_string(),
        map: format!("{:?}", job.spec.map),
        races: [format!("{:?}", job.spec.players[0].race), format!("{:?}", job.spec.players[1].race)],
        p1_kind,
        opponent: opponent_id(&job.spec.players[1], ids),
        outcome: outcome.to_string(),
        detail,
        wall_s,
    };
    let mut g = games.lock().unwrap();
    writeln!(g, "{}", serde_json::to_string(&rec)?)?;
    g.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal hostd: answers the handshake and capabilities, plays every game for `game_ms`.
    /// A `flaky` host drops the connection right after starting its first game.
    fn fake_hostd(flaky: bool, game_ms: u64) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        std::thread::spawn(move || {
            for s in listener.incoming().flatten() {
                std::thread::spawn(move || -> Result<()> {
                    let mut r = BufReader::new(s.try_clone()?);
                    let mut w = BufWriter::new(s);
                    loop {
                        let reply = match read_msg::<ControlRequest>(&mut r)? {
                            ControlRequest::Hello { .. } => ControlResponse::Hello { control_version: CONTROL_VERSION, host_name_hint: String::new() },
                            ControlRequest::ListCapabilities => ControlResponse::Capabilities(Capabilities {
                                os: HostOs::Linux,
                                cores: 2,
                                max_concurrent_games: 1,
                                max_concurrent_replays: 0,
                                oss_bots: vec![],
                                maps: vec![],
                            }),
                            ControlRequest::StartGame(g) => {
                                write_msg(&mut w, &ControlResponse::Started { game_id: g.game_id })?;
                                if flaky {
                                    return Ok(());
                                }
                                std::thread::sleep(Duration::from_millis(game_ms));
                                ControlResponse::Finished(GameFinished { game_id: g.game_id, outcome: GameOutcome::Finished, exit_codes: [Some(0), Some(0)], wall_s: 0 })
                            }
                            _ => return Ok(()),
                        };
                        write_msg(&mut w, &reply)?;
                    }
                });
            }
        });
        addr
    }

    #[test]
    fn lost_and_unreachable_hosts_do_not_stop_the_plan_020_r4() {
        let dead = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().to_string();
        let hosts: Vec<HostEntry> = [dead, fake_hostd(false, 200), fake_hostd(true, 0)].into_iter().map(|addr| HostEntry { addr, max_games: None }).collect();
        let plan: plan::Plan = toml::from_str("name='t'\n[[matchup]]\nmap='both'\ngames=4\np0={kind='t0',race='terran'}\np1={kind='t0',race='terran'}").unwrap();
        let dir = std::env::temp_dir().join(format!("uranium-env-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let games = dir.join("games.jsonl");
        let left = run_jobs(plan::expand(&plan, "127.0.0.1:1"), &hosts, &games).unwrap();
        let lines: Vec<serde_json::Value> = std::fs::read_to_string(&games).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(left, 0);
        assert_eq!(lines.len(), 4, "{lines:?}");
        assert_eq!(lines.iter().filter(|g| g["outcome"] == "Aborted").count(), 1, "{lines:?}");
        assert_eq!(lines.iter().filter(|g| g["outcome"] == "Finished").count(), 3, "{lines:?}");
    }

    #[test]
    fn no_reachable_host_is_an_error_020_r4() {
        let dead = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().to_string();
        let dir = std::env::temp_dir().join(format!("uranium-env-test-none-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let res = run_jobs(vec![], &[HostEntry { addr: dead, max_games: None }], &dir.join("games.jsonl"));
        std::fs::remove_dir_all(&dir).ok();
        assert!(res.is_err());
    }
}
