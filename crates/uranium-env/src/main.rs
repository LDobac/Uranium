//! `uranium-env`: game scheduler (spec 020). Runs the agent server in-process, spreads the
//! games of a matchup plan over one or more `uranium-hostd`s and writes results.

mod plan;
mod report;

use std::collections::VecDeque;
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

    // Embedded agent server (001-R6): bridges on every host must reach this address.
    let listen = env_or("URANIUM_AGENT_LISTEN", "127.0.0.1:7600");
    let public = env_or("URANIUM_AGENT_PUBLIC_ADDR", &listen);
    let listener = TcpListener::bind(&listen).with_context(|| format!("binding agent {listen}"))?;
    let sink = open_sink(&out.join("results.jsonl"))?;
    std::thread::spawn(move || {
        if let Err(e) = serve(listener, ServeOptions { policy: PolicyKind::ScriptedT0, delay_ms: 0, max_sessions: 0 }, sink) {
            eprintln!("[env] agent server stopped: {e:#}");
        }
    });

    let jobs: VecDeque<Job> = plan::expand(&plan, &public).into();
    let total = jobs.len();
    let queue = Arc::new(Mutex::new(jobs));
    let games_out = Arc::new(Mutex::new(BufWriter::new(std::fs::File::create(out.join("games.jsonl"))?)));
    let done = Arc::new(Mutex::new(0usize));
    let started = Instant::now();
    eprintln!("[env] plan `{}`: {total} games, agent at {public}", plan.name);

    let mut workers = Vec::new();
    for host in hosts()? {
        let caps = capabilities(&host.addr).with_context(|| format!("hostd {}", host.addr))?;
        let slots = host.max_games.map_or(caps.max_concurrent_games, |m| m.min(caps.max_concurrent_games));
        eprintln!("[env] host {}: {:?}, {} slots, oss bots {:?}", host.addr, caps.os, slots, caps.oss_bots);
        for _ in 0..slots {
            let (addr, q, g, d) = (host.addr.clone(), queue.clone(), games_out.clone(), done.clone());
            workers.push(std::thread::spawn(move || worker(&addr, &q, &g, &d, total)));
        }
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
    eprintln!("[env] finished in {:.0}s ({left} games not run)", started.elapsed().as_secs_f64());
    report::write(&out, &plan)?;
    println!("{}", std::fs::read_to_string(out.join("summary.md"))?);
    if left > 0 {
        bail!("{left} games could not be scheduled");
    }
    Ok(())
}

fn connect(addr: &str) -> Result<(BufReader<TcpStream>, BufWriter<TcpStream>)> {
    let s = TcpStream::connect(addr)?;
    s.set_nodelay(true).ok();
    let mut r = BufReader::new(s.try_clone()?);
    let mut w = BufWriter::new(s);
    write_msg(&mut w, &ControlRequest::Hello { control_version: CONTROL_VERSION })?;
    match read_msg(&mut r)? {
        ControlResponse::Hello { .. } => Ok((r, w)),
        other => bail!("handshake failed: {other:?}"),
    }
}

fn capabilities(addr: &str) -> Result<Capabilities> {
    let (mut r, mut w) = connect(addr)?;
    write_msg(&mut w, &ControlRequest::ListCapabilities)?;
    match read_msg(&mut r)? {
        ControlResponse::Capabilities(c) => Ok(c),
        other => bail!("unexpected {other:?}"),
    }
}

fn worker(addr: &str, queue: &Mutex<VecDeque<Job>>, games: &Mutex<BufWriter<std::fs::File>>, done: &Mutex<usize>, total: usize) -> Result<()> {
    let (mut r, mut w) = connect(addr)?;
    loop {
        let Some(job) = queue.lock().unwrap().pop_front() else { return Ok(()) };
        write_msg(&mut w, &ControlRequest::StartGame(job.spec.clone()))?;
        match read_msg(&mut r)? {
            ControlResponse::Started { .. } => {}
            ControlResponse::Rejected { reason } => {
                eprintln!("[env] {addr} rejected game {}: {reason}", job.spec.game_id);
                if reason.contains("capacity") {
                    queue.lock().unwrap().push_back(job);
                    std::thread::sleep(Duration::from_secs(2));
                    continue;
                }
                record(games, addr, &job, "Rejected", Some(reason), 0)?;
                continue;
            }
            other => bail!("unexpected {other:?}"),
        }
        let fin = loop {
            match read_msg(&mut r)? {
                ControlResponse::Finished(f) if f.game_id == job.spec.game_id => break f,
                other => eprintln!("[env] ignoring {other:?}"),
            }
        };
        let (outcome, detail) = match &fin.outcome {
            GameOutcome::Finished => ("Finished", None),
            GameOutcome::Aborted { reason } => ("Aborted", Some(reason.clone())),
            GameOutcome::Timeout => ("Timeout", None),
            GameOutcome::Stopped => ("Stopped", None),
        };
        record(games, addr, &job, outcome, detail, fin.wall_s)?;
        let mut d = done.lock().unwrap();
        *d += 1;
        eprintln!("[env] {}/{total} game {} {outcome} ({}s)", *d, job.spec.game_id, fin.wall_s);
    }
}

fn record(games: &Mutex<BufWriter<std::fs::File>>, host: &str, job: &Job, outcome: &str, detail: Option<String>, wall_s: u32) -> Result<()> {
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
        outcome: outcome.to_string(),
        detail,
        wall_s,
    };
    let mut g = games.lock().unwrap();
    writeln!(g, "{}", serde_json::to_string(&rec)?)?;
    g.flush()?;
    Ok(())
}
