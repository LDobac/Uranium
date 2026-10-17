//! Agent server library (spec 001-R6). Bridges connect to it; every connection is one player
//! of one game (`Hello.game_id` / `Hello.slot`, 010-R9) and is served by its own policy instance.
//! Used by the `uranium-agent` binary and embedded by `uranium-env` (020).

pub mod summary;

use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::{BufReader, BufWriter, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde::Serialize;
use uranium_proto::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum PolicyKind {
    /// T0 rule-based bot for our own race (021): picked from `GameInfo` at game start.
    ScriptedT0,
}

/// A policy controls one player for one game.
pub trait Policy: Send {
    fn on_privileged(&mut self, _p: &PrivilegedSummary) {}
    fn on_frame(&mut self, f: &RawFrame) -> Act;
}

pub use uranium_scripted::T0Build;

/// T0 build per (game id, `GameSpec.players` index) for scenario bots (050 E-Regression).
/// The scheduler fills it before games start; players not listed play the standard build.
pub type T0Builds = Arc<Mutex<HashMap<(u64, u8), T0Build>>>;

/// Our `GameSpec.players` index: uranium-hostd names each player `uranium-p<index>` because
/// the BWAPI player id (`Hello.slot`) need not follow launch order. Falls back to the slot.
pub fn spec_index(info: &GameInfo, slot: u8) -> u8 {
    let name = info.players.iter().find(|p| p.is_self).map(|p| p.name.as_str()).unwrap_or("");
    name.strip_prefix("uranium-p").and_then(|i| i.parse().ok()).unwrap_or(slot)
}

pub fn make_policy(kind: PolicyKind, info: &GameInfo, build: T0Build) -> Box<dyn Policy> {
    match kind {
        PolicyKind::ScriptedT0 => Box::new(uranium_scripted::for_game_build(info, build)),
    }
}

impl Policy for Box<dyn uranium_scripted::Bot> {
    fn on_frame(&mut self, f: &RawFrame) -> Act {
        self.act(f)
    }
}

/// One JSONL record (one player of one game).
#[derive(Serialize)]
pub struct SessionRecord {
    pub game_id: u64,
    pub slot: u8,
    pub map: String,
    pub map_hash: String,
    pub self_race: String,
    pub enemy_race: String,
    /// T0 build this player was asked to play (050 E-Regression scenario bots).
    pub t0_build: String,
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

pub fn fnv1a(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

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
pub struct ServeOptions {
    pub policy: PolicyKind,
    /// Artificial delay before every Act (011-AC7 lag test).
    pub delay_ms: u64,
    /// Return after this many finished sessions (0 = run forever).
    pub max_sessions: usize,
    /// Scenario builds by game and slot (empty = every T0 player plays the standard build).
    pub t0_builds: T0Builds,
}

/// JSONL sink shared by all sessions.
pub type RecordSink = Arc<Mutex<dyn Write + Send>>;

pub fn open_sink(out: &Path) -> Result<RecordSink> {
    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let file = OpenOptions::new().create(true).append(true).open(out).with_context(|| format!("opening {}", out.display()))?;
    Ok(Arc::new(Mutex::new(BufWriter::new(file))))
}

pub fn serve_addr(listen: &str, opts: ServeOptions, out: PathBuf) -> Result<()> {
    let listener = TcpListener::bind(listen).with_context(|| format!("binding {listen}"))?;
    eprintln!("[agent] listening on {listen}");
    serve(listener, opts, open_sink(&out)?)
}

/// Accepts bridge sessions until `max_sessions` are done (or forever).
pub fn serve(listener: TcpListener, opts: ServeOptions, sink: RecordSink) -> Result<()> {
    let ServeOptions { policy, delay_ms, max_sessions, t0_builds } = opts;
    let finished = Arc::new(Mutex::new(0usize));
    let mut handles = Vec::new();
    for conn in listener.incoming() {
        let stream = conn?;
        stream.set_nodelay(true).ok();
        let (sink, done, builds) = (sink.clone(), finished.clone(), t0_builds.clone());
        handles.push(std::thread::spawn(move || {
            let rec = session(stream, policy, delay_ms, &builds);
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

pub fn session(stream: TcpStream, kind: PolicyKind, delay_ms: u64, builds: &T0Builds) -> SessionRecord {
    let started = Instant::now();
    let mut rec = SessionRecord {
        game_id: 0,
        slot: 0,
        map: String::new(),
        map_hash: String::new(),
        self_race: String::new(),
        enemy_race: String::new(),
        t0_build: String::new(),
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
    if let Err(e) = run_session(stream, kind, delay_ms, builds, &mut rec) {
        eprintln!("[agent] game {} slot {}: {e:#}", rec.game_id, rec.slot);
        rec.error = Some(format!("{e:#}"));
    }
    rec.wall_s = started.elapsed().as_secs_f64();
    rec
}

fn run_session(stream: TcpStream, kind: PolicyKind, delay_ms: u64, builds: &T0Builds, rec: &mut SessionRecord) -> Result<()> {
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
    rec.self_race = info.players.iter().find(|p| p.is_self).map(|p| format!("{:?}", p.race)).unwrap_or_default();
    rec.enemy_race = info.players.iter().find(|p| p.is_enemy).map(|p| format!("{:?}", p.race)).unwrap_or_default();
    eprintln!("[agent] game {} slot {} on {:?} ({} start locations)", hello.game_id, hello.slot, info.map_name, info.static_map.start_locations.len());
    let build = builds.lock().unwrap().get(&(hello.game_id, spec_index(&info, hello.slot))).copied().unwrap_or_default();
    rec.t0_build = build.name().to_string();
    let mut policy = make_policy(kind, &info, build);
    let mut policy_time = Duration::ZERO;
    let mut first = true;
    let mut digest: u64 = 0xcbf2_9ce4_8422_2325;
    let mut last_cmds: Vec<Command> = Vec::new();
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
                // Debug aid: with URANIUM_AGENT_DUMP set, failed commands go to g<id>-s<slot>-failed.txt at any frame.
                if let Ok(dir) = std::env::var("URANIUM_AGENT_DUMP") {
                    for e in &f.events {
                        if let GameEvent::CommandFailed { cmd_index, error } = e {
                            let path = format!("{dir}/g{}-s{}-failed.txt", hello.game_id, hello.slot);
                            if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
                                writeln!(file, "{} error {error} {:?}", f.frame, last_cmds.get(*cmd_index as usize)).ok();
                            }
                        }
                    }
                }
                let t = Instant::now();
                let act = policy.on_frame(&f);
                policy_time += t.elapsed();
                rec.commands += act.cmds.len() as u64;
                last_cmds.clone_from(&act.cmds);
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
