//! Agent server (spec 001-R6). Bridges connect to it; every connection is one player of one
//! game (`Hello.game_id` / `Hello.slot`, 010-R9) and is served by its own policy instance.
//!
//! M2 ships only the scripted T0 policy; learned policies plug in behind the same trait.

mod summary;

use std::fs::OpenOptions;
use std::io::{BufReader, BufWriter, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand, ValueEnum};
use serde::Serialize;
use uranium_proto::*;

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Accept bridge connections and play.
    Serve {
        #[arg(long, default_value = "127.0.0.1:7600")]
        listen: String,
        #[arg(long, value_enum, default_value = "scripted-terran")]
        policy: PolicyKind,
        /// JSONL file receiving one record per finished session.
        #[arg(long)]
        out: PathBuf,
        /// Artificial delay before every Act (011-AC7 lag test).
        #[arg(long, default_value_t = 0)]
        respond_delay_ms: u64,
        /// Exit after this many finished sessions (0 = run forever).
        #[arg(long, default_value_t = 0)]
        sessions: usize,
    },
    /// Summarize a JSONL results file into Markdown.
    Summarize {
        input: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value = "")]
        title: String,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum PolicyKind {
    ScriptedTerran,
}

/// A policy controls one player for one game.
trait Policy: Send {
    fn on_privileged(&mut self, _p: &PrivilegedSummary) {}
    fn on_frame(&mut self, f: &RawFrame) -> Act;
}

impl Policy for uranium_scripted::TerranRush {
    fn on_frame(&mut self, f: &RawFrame) -> Act {
        self.act(f)
    }
}

fn make_policy(kind: PolicyKind, info: &GameInfo) -> Box<dyn Policy> {
    match kind {
        PolicyKind::ScriptedTerran => Box::new(uranium_scripted::TerranRush::new(info)),
    }
}

/// One JSONL record (one player of one game).
#[derive(Serialize)]
pub struct SessionRecord {
    pub game_id: u64,
    pub slot: u8,
    pub map: String,
    pub map_hash: String,
    pub winner: Option<u8>,
    pub won: Option<bool>,
    pub reason: String,
    pub end_frame: u32,
    pub wall_s: f64,
    pub policy_ms: f64,
    pub stats: Option<BridgeStats>,
    /// M1 diagnosis: mineral fields visible in the first observation.
    pub first_frame_minerals: usize,
    pub first_frame_units: usize,
    pub commands: u64,
    pub failed_commands: u64,
    /// FNV-1a over every observation's encoded bytes: identical runs give identical digests
    /// (011-AC4); privileged mode on/off must not change it (010-AC6).
    pub frame_digest: String,
    pub privileged_msgs: u64,
    /// 010-AC6: enemy units in an observation that our player could not see (must be 0).
    pub visibility_violations: u64,
    pub error: Option<String>,
}

/// BWAPI neutral player id.
const NEUTRAL_PLAYER: u8 = 11;

/// Enemy units that are not on (or next to) a currently visible tile, or that are cloaked or
/// burrowed without being detected. The bridge must never send these (010-R5, 010-AC6).
fn visibility_violations(f: &RawFrame, me: u8) -> u64 {
    f.units
        .iter()
        .filter(|u| u.owner != me && u.owner != NEUTRAL_PLAYER)
        .filter(|u| {
            // BWAPI counts a unit visible when any tile of its footprint is visible; check the
            // footprint plus a 1-tile margin (sight is computed per 8 px walk tile).
            let (w, h) = uranium_scripted::types::tile_size(u.unit_type);
            let (x0, y0) = ((u.pos.0 as i32 - w as i32 * 16) / 32 - 1, (u.pos.1 as i32 - h as i32 * 16) / 32 - 1);
            let seen = (y0..=y0 + h as i32 + 1).any(|ty| (x0..=x0 + w as i32 + 1).any(|tx| tx >= 0 && ty >= 0 && f.visibility.get(tx as u16, ty as u16)));
            let hidden = (u.has(flags::CLOAKED) || u.has(flags::BURROWED)) && !u.has(flags::DETECTED);
            if (!seen || hidden) && std::env::var("URANIUM_AGENT_DEBUG_VIS").is_ok() {
                eprintln!("[vis] frame {} slot {me} unit {} type {} owner {} pos {:?} flags {:#x} seen {seen} hidden {hidden}", f.frame, u.id, u.unit_type, u.owner, u.pos, u.flags);
            }
            !seen || hidden
        })
        .count() as u64
}

fn fnv1a(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Loads host settings from `.env` (or `$URANIUM_ENV_FILE`) before anything reads the
/// environment; variables already set in the process win (spec 001-R9).
fn load_env() {
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

fn main() -> Result<()> {
    load_env();
    match Cli::parse().cmd {
        Cmd::Serve { listen, policy, out, respond_delay_ms, sessions } => serve(&listen, policy, out, respond_delay_ms, sessions),
        Cmd::Summarize { input, out, title } => summary::summarize(&input, &out, &title),
    }
}

fn serve(listen: &str, policy: PolicyKind, out: PathBuf, delay_ms: u64, max_sessions: usize) -> Result<()> {
    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let file = OpenOptions::new().create(true).append(true).open(&out).with_context(|| format!("opening {}", out.display()))?;
    let sink = Arc::new(Mutex::new(BufWriter::new(file)));
    let listener = TcpListener::bind(listen).with_context(|| format!("binding {listen}"))?;
    eprintln!("[agent] listening on {listen}");
    let finished = Arc::new(Mutex::new(0usize));
    let mut handles = Vec::new();
    for conn in listener.incoming() {
        let stream = conn?;
        stream.set_nodelay(true).ok();
        let (sink, done) = (sink.clone(), finished.clone());
        handles.push(std::thread::spawn(move || {
            let rec = session(stream, policy, delay_ms);
            let line = serde_json::to_string(&rec).expect("record serializes");
            let mut s = sink.lock().unwrap();
            writeln!(s, "{line}").ok();
            s.flush().ok();
            *done.lock().unwrap() += 1;
        }));
        handles.retain(|h| !h.is_finished());
        if max_sessions > 0 && *finished.lock().unwrap() >= max_sessions {
            break;
        }
        // Accept loop blocks; finished-count is re-checked on the next connection. The
        // selfmatch harness stops the server explicitly instead of relying on this.
    }
    for h in handles {
        h.join().ok();
    }
    Ok(())
}

fn session(stream: TcpStream, kind: PolicyKind, delay_ms: u64) -> SessionRecord {
    let started = Instant::now();
    let mut rec = SessionRecord {
        game_id: 0,
        slot: 0,
        map: String::new(),
        map_hash: String::new(),
        winner: None,
        won: None,
        reason: "Disconnected".into(),
        end_frame: 0,
        wall_s: 0.0,
        policy_ms: 0.0,
        stats: None,
        first_frame_minerals: 0,
        first_frame_units: 0,
        commands: 0,
        failed_commands: 0,
        frame_digest: String::new(),
        privileged_msgs: 0,
        visibility_violations: 0,
        error: None,
    };
    if let Err(e) = run_session(stream, kind, delay_ms, &mut rec) {
        eprintln!("[agent] game {} slot {}: {e:#}", rec.game_id, rec.slot);
        rec.error = Some(format!("{e:#}"));
    }
    rec.wall_s = started.elapsed().as_secs_f64();
    rec
}

fn run_session(stream: TcpStream, kind: PolicyKind, delay_ms: u64, rec: &mut SessionRecord) -> Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut writer = BufWriter::new(stream);
    let BridgeMsg::Hello(hello) = read_msg(&mut reader)? else { bail!("expected Hello") };
    rec.game_id = hello.game_id;
    rec.slot = hello.slot;
    if let Err(why) = check_hello(&hello) {
        write_msg(&mut writer, &AgentMsg::Reject(why.clone()))?;
        bail!(why);
    }
    let BridgeMsg::GameInfo(info) = read_msg(&mut reader)? else { bail!("expected GameInfo") };
    rec.map = info.map_name.clone();
    rec.map_hash = info.map_hash.iter().map(|b| format!("{b:02x}")).collect();
    eprintln!("[agent] game {} slot {} on {:?} ({} start locations)", hello.game_id, hello.slot, info.map_name, info.static_map.start_locations.len());
    let mut policy = make_policy(kind, &info);
    let mut policy_time = Duration::ZERO;
    let mut first = true;
    let mut digest: u64 = 0xcbf2_9ce4_8422_2325;
    loop {
        match read_msg::<BridgeMsg>(&mut reader)? {
            BridgeMsg::Privileged(p) => {
                rec.privileged_msgs += 1;
                policy.on_privileged(&p)
            }
            BridgeMsg::Frame(f) => {
                let mut bytes = Vec::new();
                write_msg(&mut bytes, &*f)?;
                digest = fnv1a(digest, &bytes);
                // Debug aid: URANIUM_AGENT_DUMP=<dir> writes observations before URANIUM_AGENT_DUMP_MAX_FRAME (default 240).
                if let Ok(dir) = std::env::var("URANIUM_AGENT_DUMP") {
                    let max: u32 = std::env::var("URANIUM_AGENT_DUMP_MAX_FRAME").ok().and_then(|v| v.parse().ok()).unwrap_or(240);
                    if f.frame < max {
                        let path = format!("{dir}/g{}-s{}.txt", hello.game_id, hello.slot);
                        if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
                            writeln!(file, "{f:#?}").ok();
                        }
                    }
                }
                if first {
                    rec.first_frame_units = f.units.len();
                    rec.first_frame_minerals = f.units.iter().filter(|u| uranium_scripted::types::is_mineral(u.unit_type)).count();
                    first = false;
                }
                rec.visibility_violations += visibility_violations(&f, hello.slot);
                rec.failed_commands += f.events.iter().filter(|e| matches!(e, GameEvent::CommandFailed { .. })).count() as u64;
                let t = Instant::now();
                let act = policy.on_frame(&f);
                policy_time += t.elapsed();
                rec.commands += act.cmds.len() as u64;
                if delay_ms > 0 {
                    std::thread::sleep(Duration::from_millis(delay_ms));
                }
                write_msg(&mut writer, &AgentMsg::Act(act))?;
            }
            BridgeMsg::End(end) => {
                rec.winner = end.winner;
                rec.won = end.winner.map(|w| w == hello.slot);
                rec.reason = format!("{:?}", end.reason);
                rec.end_frame = end.frame;
                rec.stats = end.stats;
                rec.policy_ms = policy_time.as_secs_f64() * 1e3;
                rec.frame_digest = format!("{digest:016x}");
                return Ok(());
            }
            other => bail!("unexpected message {other:?}"),
        }
    }
}
