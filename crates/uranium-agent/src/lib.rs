//! Agent server library (spec 001-R6). Bridges connect to it; every connection is one player
//! of one game (`Hello.game_id` / `Hello.slot`, 010-R9) and is served by its own policy instance,
//! chosen from the routing table (020-R11). Used by the `uranium-agent` binary and embedded by the
//! `uranium-env` scheduler (crate `uranium-runner`, 020).

/// `eprintln!` that never panics: inside `uranium-train rl` this runs in a training child whose
/// stderr is a pipe to its watchdog; when the watchdog is killed hard the pipe closes and
/// `eprintln!` would panic (EPIPE) instead of letting the run save, push and release (060-R6).
macro_rules! elog {
    ($($t:tt)*) => {{
        use std::io::Write as _;
        let _ = writeln!(std::io::stderr(), $($t)*);
    }};
}

pub mod infer;
pub mod model_policy;
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

use crate::infer::InferenceHandle;
use crate::model_policy::{LearnerHook, ModelPolicy, ModelStats};

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum PolicyKind {
    /// T0 rule-based bot for our own race (021): picked from `GameInfo` at game start.
    ScriptedT0,
    /// Checkpoint policy (020-R11), Terran only.
    Model,
}

/// What plays one session (020-R11).
#[derive(Clone)]
pub enum PolicySpec {
    ScriptedT0,
    Model(InferenceHandle),
    /// A model session that records its trajectory for the learner (042).
    Learner(Arc<dyn LearnerHook>),
}

impl std::fmt::Debug for PolicySpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ScriptedT0 => write!(f, "ScriptedT0"),
            Self::Model(h) => write!(f, "Model({h:?})"),
            Self::Learner(l) => write!(f, "Learner({:?})", l.handle()),
        }
    }
}

impl PolicySpec {
    /// Session record label (020-R5 policy version).
    pub fn label(&self) -> String {
        match self {
            Self::ScriptedT0 => format!("t0@{}", env!("CARGO_PKG_VERSION")),
            Self::Model(h) => infer::model_policy_label(&h.checkpoint_id, h.sample),
            Self::Learner(l) => format!("learner:{}:{:?}", l.handle().checkpoint_id, l.handle().sample).to_lowercase(),
        }
    }
}

/// `(game_id, GameSpec.players index) -> policy` registered by the scheduler before `StartGame`
/// (020-R11). The index comes from the player name (`spec_index`), not the BWAPI player id.
/// Sessions not in the table use the server default.
#[derive(Clone, Default)]
pub struct Routes(Arc<Mutex<HashMap<(u64, u8), PolicySpec>>>);

impl Routes {
    pub fn insert(&self, game_id: u64, slot: u8, spec: PolicySpec) {
        self.0.lock().unwrap().insert((game_id, slot), spec);
    }

    /// Removes the entry: a game id is played once.
    pub fn take(&self, game_id: u64, slot: u8) -> Option<PolicySpec> {
        self.0.lock().unwrap().remove(&(game_id, slot))
    }
}

/// A policy controls one player for one game.
pub trait Policy: Send {
    fn on_privileged(&mut self, _p: &PrivilegedSummary) {}
    fn on_frame(&mut self, f: &RawFrame) -> Result<Act>;
    /// Model policies report latency and executor rejections (020-R5, 031-R4).
    fn model_stats(&mut self) -> Option<ModelStats> {
        None
    }
    /// Game over; `None` = Aborted or disconnected (042-R3). `final_frame`: the own-vision state at
    /// the end with the events since the last frame (010-R12), when the bridge sent one.
    fn on_end(&mut self, _outcome: Option<uranium_env::reward::Outcome>, _final_frame: Option<&RawFrame>) {}
}

pub use uranium_scripted::T0Build;

/// One session's GameEnd report (020-R13).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionEnd {
    /// BWAPI player id of the reporting session (`Hello.slot`).
    pub slot: u8,
    pub winner: Option<u8>,
    pub frame: u32,
    pub reason: EndReason,
}

/// 020-R13 (2026-10-17, AMB-020-3): when the sessions of one game report different winners, the
/// session that ended last (largest end frame) decides; on a tie the first report stands. The
/// loser's client can mark the winner as defeated too on its final frame (OpenBW leave timing,
/// not a desync). Returns the chosen report and whether the reports disagreed.
pub fn game_result(ends: &[SessionEnd]) -> Option<(SessionEnd, bool)> {
    let first = *ends.first()?;
    let chosen = ends.iter().skip(1).fold(first, |best, e| if e.frame > best.frame { *e } else { best });
    let disagree = ends.iter().any(|e| e.winner != first.winner);
    if mutual_leave(ends) {
        return Some((SessionEnd { winner: None, reason: EndReason::Aborted, ..chosen }, disagree));
    }
    Some((chosen, disagree))
}

/// 020-R13: both sessions say the other one left and claim the win. A real leave (a bot's GG)
/// is reported as a loss by the leaver, so this is a connection drop (OpenBW drops a client that
/// stops syncing, 011-AC7): the game is Aborted and its segments revoked (042-R3).
pub fn mutual_leave(ends: &[SessionEnd]) -> bool {
    ends.len() == 2 && ends.iter().all(|e| e.reason == EndReason::Surrender && e.winner == Some(e.slot))
}

/// GameEnd reports by game id, filled by the agent server (020-R13). The RL league reads the
/// reconciled result from here once hostd says the game is over (042 win table, 033 r_win).
#[derive(Clone, Default)]
pub struct EndBoard(Arc<Mutex<HashMap<u64, Vec<SessionEnd>>>>);

impl EndBoard {
    pub fn push(&self, game_id: u64, e: SessionEnd) {
        self.0.lock().unwrap().entry(game_id).or_default().push(e);
    }

    /// Removes and returns the reports of a game.
    pub fn take(&self, game_id: u64) -> Vec<SessionEnd> {
        self.0.lock().unwrap().remove(&game_id).unwrap_or_default()
    }
}

/// T0 build per (game id, `GameSpec.players` index) for scenario bots (050 E-Regression).
/// The scheduler fills it before games start; players not listed play the standard build.
pub type T0Builds = Arc<Mutex<HashMap<(u64, u8), T0Build>>>;

/// Our `GameSpec.players` index: uranium-hostd names each player `uranium-p<index>` because
/// the BWAPI player id (`Hello.slot`) need not follow launch order. Falls back to the slot.
pub fn spec_index(info: &GameInfo, slot: u8) -> u8 {
    let name = info.players.iter().find(|p| p.is_self).map(|p| p.name.as_str()).unwrap_or("");
    name.strip_prefix("uranium-p").and_then(|i| i.parse().ok()).unwrap_or(slot)
}

pub fn make_policy(spec: &PolicySpec, game_id: u64, info: &GameInfo, build: T0Build) -> Result<Box<dyn Policy>> {
    let terran = || -> Result<()> {
        let me = info.players.iter().find(|p| p.is_self).map(|p| p.race);
        if me != Some(Race::Terran) {
            bail!("model policy plays Terran only, got {me:?}");
        }
        Ok(())
    };
    Ok(match spec {
        PolicySpec::ScriptedT0 => Box::new(uranium_scripted::for_game_build(info, build)),
        PolicySpec::Model(h) => {
            terran()?;
            Box::new(ModelPolicy::new(h, info)?)
        }
        PolicySpec::Learner(l) => {
            terran()?;
            let (z, observer) = l.session(game_id, info);
            Box::new(ModelPolicy::new(l.handle(), info)?.with_observer(z, observer))
        }
    })
}

/// 033 outcome for our player from the bridge's GameEnd; `None` when the game did not finish
/// (Aborted, agent timeout): such episodes are left out of training (042-R3).
pub fn end_outcome(end: &GameEnd, me: u8) -> Option<uranium_env::reward::Outcome> {
    use uranium_env::reward::Outcome;
    match end.reason {
        EndReason::Aborted | EndReason::AgentTimeout => None,
        EndReason::TimeLimitDraw => Some(Outcome::Draw),
        EndReason::Elimination | EndReason::Surrender => Some(match end.winner {
            Some(w) if w == me => Outcome::Win,
            Some(_) => Outcome::Loss,
            None => Outcome::Draw,
        }),
    }
}

impl Policy for Box<dyn uranium_scripted::Bot> {
    fn on_frame(&mut self, f: &RawFrame) -> Result<Act> {
        Ok(self.act(f))
    }
}

impl Policy for ModelPolicy {
    fn on_privileged(&mut self, p: &PrivilegedSummary) {
        ModelPolicy::on_privileged(self, p)
    }

    fn on_frame(&mut self, f: &RawFrame) -> Result<Act> {
        ModelPolicy::on_frame(self, f)
    }

    fn model_stats(&mut self) -> Option<ModelStats> {
        Some(self.stats())
    }

    fn on_end(&mut self, outcome: Option<uranium_env::reward::Outcome>, final_frame: Option<&RawFrame>) {
        ModelPolicy::on_end(self, outcome, final_frame)
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
    /// Race the enemy actually played: the lobby race, or for a lobby-Random enemy (030-R11) the
    /// race of its first unit we saw ("Unknown" if never seen). For per-race statistics.
    pub enemy_race: String,
    /// Race as `GameInfo` gave it at game start ("Unknown" for a lobby-Random enemy, 030-R11).
    pub enemy_race_lobby: String,
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
    /// 010-R12: (frame, units, events) of `GameEnd.final_frame`, `None` if the bridge sent none.
    pub final_frame: Option<(u32, usize, usize)>,
    /// 020-R5 policy version (`t0@<ver>` or `model:<checkpoint id>:<sample>`).
    pub policy: String,
    pub model: Option<ModelStats>,
    pub error: Option<String>,
}

/// BWAPI neutral player id.
const NEUTRAL_PLAYER: u8 = 11;

/// 030-R11: race of the first non-neutral, non-own unit in the frame ("Terran" / "Protoss" /
/// "Zerg"), for the session record of a lobby-Random enemy. Assumes 1v1 (any such unit is the
/// enemy).
fn revealed_enemy_race(f: &RawFrame, me: u8) -> Option<String> {
    let types = uranium_env::types::TypeTables::get();
    f.units
        .iter()
        .filter(|u| u.owner != me && u.owner != NEUTRAL_PLAYER)
        .filter_map(|u| types.unit(u.unit_type))
        .find(|t| matches!(t.race.as_str(), "Terran" | "Protoss" | "Zerg"))
        .map(|t| t.race.clone())
}

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
                elog!("[vis] frame {} slot {me} unit {} type {} owner {} pos {:?} flags {:#x} seen {seen} hidden {hidden}", f.frame, u.id, u.unit_type, u.owner, u.pos, u.flags);
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
            elog!("warning: failed to load .env: {e}");
        }
    }
}

#[derive(Clone)]
pub struct ServeOptions {
    /// Policy for sessions without a route (020-R11).
    pub default: PolicySpec,
    pub routes: Routes,
    /// Artificial delay before every Act (011-AC7 lag test).
    pub delay_ms: u64,
    /// Return after this many finished sessions (0 = run forever).
    pub max_sessions: usize,
    /// Scenario builds by game and slot (empty = every T0 player plays the standard build).
    pub t0_builds: T0Builds,
    /// Every session's GameEnd report (020-R13).
    pub ends: EndBoard,
}

impl ServeOptions {
    pub fn new(default: PolicySpec) -> Self {
        Self { default, routes: Routes::default(), delay_ms: 0, max_sessions: 0, t0_builds: T0Builds::default(), ends: EndBoard::default() }
    }
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
    elog!("[agent] listening on {listen}");
    serve(listener, opts, open_sink(&out)?)
}

/// Accepts bridge sessions until `max_sessions` are done (or forever).
pub fn serve(listener: TcpListener, opts: ServeOptions, sink: RecordSink) -> Result<()> {
    let max_sessions = opts.max_sessions;
    let finished = Arc::new(Mutex::new(0usize));
    let mut handles = Vec::new();
    for conn in listener.incoming() {
        let stream = conn?;
        stream.set_nodelay(true).ok();
        let (sink, done, opts) = (sink.clone(), finished.clone(), opts.clone());
        handles.push(std::thread::spawn(move || {
            let rec = session(stream, &opts);
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

pub fn session(stream: TcpStream, opts: &ServeOptions) -> SessionRecord {
    let started = Instant::now();
    let mut rec = SessionRecord {
        game_id: 0,
        slot: 0,
        map: String::new(),
        map_hash: String::new(),
        self_race: String::new(),
        enemy_race: String::new(),
        enemy_race_lobby: String::new(),
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
        final_frame: None,
        policy: String::new(),
        model: None,
        error: None,
    };
    if let Err(e) = run_session(stream, opts, &mut rec) {
        elog!("[agent] game {} slot {}: {e:#}", rec.game_id, rec.slot);
        rec.error = Some(format!("{e:#}"));
    }
    rec.wall_s = started.elapsed().as_secs_f64();
    rec
}

fn run_session(stream: TcpStream, opts: &ServeOptions, rec: &mut SessionRecord) -> Result<()> {
    let delay_ms = opts.delay_ms;
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
    // 030-R11: a lobby-Random enemy arrives as Unknown; `play` fills in the race once seen.
    rec.enemy_race_lobby = rec.enemy_race.clone();
    elog!("[agent] game {} slot {} on {:?} ({} start locations)", hello.game_id, hello.slot, info.map_name, info.static_map.start_locations.len());
    // 020-R11 / 050: policies and T0 builds are registered by `GameSpec.players` index; the BWAPI
    // player id (`Hello.slot`) need not follow launch order, so it comes from the player name.
    let index = spec_index(&info, hello.slot);
    let spec = opts.routes.take(hello.game_id, index).unwrap_or_else(|| opts.default.clone());
    rec.policy = spec.label();
    let build = opts.t0_builds.lock().unwrap().get(&(hello.game_id, index)).copied().unwrap_or_default();
    rec.t0_build = build.name().to_string();
    let mut policy = make_policy(&spec, hello.game_id, &info, build)?;
    let res = play(&mut reader, &mut writer, &mut *policy, &hello, rec, delay_ms, &opts.ends);
    if res.is_err() {
        // Disconnected or failed mid-game: not a finished episode (042-R3).
        policy.on_end(None, None);
    }
    res
}

#[allow(clippy::too_many_arguments)]
fn play(reader: &mut BufReader<TcpStream>, writer: &mut BufWriter<TcpStream>, policy: &mut dyn Policy, hello: &Hello, rec: &mut SessionRecord, delay_ms: u64, ends: &EndBoard) -> Result<()> {
    let (mut reader, mut writer) = (reader, writer);
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
                if matches!(rec.enemy_race.as_str(), "Unknown" | "Random") {
                    if let Some(r) = revealed_enemy_race(&f, hello.slot) {
                        rec.enemy_race = r;
                    }
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
                let act = match policy.on_frame(&f) {
                    Ok(a) => a,
                    Err(e) => {
                        rec.model = policy.model_stats();
                        return Err(e);
                    }
                };
                policy_time += t.elapsed();
                rec.commands += act.cmds.len() as u64;
                last_cmds.clone_from(&act.cmds);
                if delay_ms > 0 {
                    std::thread::sleep(Duration::from_millis(delay_ms));
                }
                write_msg(&mut writer, &AgentMsg::Act(act))?;
            }
            BridgeMsg::End(end) => {
                let session_end = SessionEnd { slot: hello.slot, winner: end.winner, frame: end.frame, reason: end.reason };
                let outcome = end_outcome(&end, hello.slot);
                rec.winner = end.winner;
                rec.won = end.winner.map(|w| w == hello.slot);
                rec.reason = format!("{:?}", end.reason);
                rec.end_frame = end.frame;
                rec.stats = end.stats;
                rec.policy_ms = policy_time.as_secs_f64() * 1e3;
                rec.frame_digest = format!("{digest:016x}");
                rec.model = policy.model_stats();
                ends.push(hello.game_id, session_end);
                if let Some(last) = &end.final_frame {
                    // 010-AC6 also holds for the final frame (010-R12 keeps 010-R5).
                    rec.visibility_violations += visibility_violations(last, hello.slot);
                    rec.final_frame = Some((last.frame, last.units.len(), last.events.len()));
                }
                policy.on_end(outcome, end.final_frame.as_deref());
                return Ok(());
            }
            other => bail!("unexpected message {other:?}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn end(slot: u8, winner: Option<u8>, frame: u32) -> SessionEnd {
        SessionEnd { slot, winner, frame, reason: EndReason::Elimination }
    }

    /// 020-R13: on disagreement the session with the larger end frame decides.
    #[test]
    fn game_result_takes_the_later_session_020_r13() {
        let (r, d) = game_result(&[end(0, Some(1), 9000), end(1, Some(1), 9000)]).unwrap();
        assert_eq!((r.winner, d), (Some(1), false));
        // Loser (slot 0) ended first and marked both as defeated: the winner's later report wins.
        let (r, d) = game_result(&[end(0, None, 8999), end(1, Some(1), 9001)]).unwrap();
        assert_eq!((r.winner, r.slot, d), (Some(1), 1, true));
        let (r, d) = game_result(&[end(1, Some(1), 9001), end(0, None, 8999)]).unwrap();
        assert_eq!((r.winner, d), (Some(1), true));
        // Tie: the first report stands.
        let (r, _) = game_result(&[end(0, Some(0), 500), end(1, Some(1), 500)]).unwrap();
        assert_eq!(r.winner, Some(0));
        assert!(game_result(&[]).is_none());
    }

    /// 020-R13: both sides claiming the win because the other left is a drop: Aborted. A real
    /// leave (the leaver reports the other side as winner) and a normal elimination stand.
    #[test]
    fn mutual_leave_is_aborted_020_r13() {
        let left = |slot, winner, frame| SessionEnd { slot, winner, frame, reason: EndReason::Surrender };
        let (r, d) = game_result(&[left(0, Some(0), 1615), left(1, Some(1), 1615)]).unwrap();
        assert_eq!((r.reason, r.winner, d), (EndReason::Aborted, None, true));
        let (r, _) = game_result(&[end(0, Some(1), 4000), left(1, Some(1), 4010)]).unwrap();
        assert_eq!((r.reason, r.winner), (EndReason::Surrender, Some(1)));
        let (r, _) = game_result(&[end(0, Some(0), 8000), left(1, Some(1), 8031)]).unwrap();
        assert_eq!((r.reason, r.winner), (EndReason::Surrender, Some(1)));
        assert!(!mutual_leave(&[left(0, Some(0), 1615)]));
    }
}
