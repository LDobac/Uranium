//! Game scheduler library (spec 020, crate `uranium-runner`, AMB-001-3). Runs the agent server
//! in-process and spreads games over one or more `uranium-hostd`s. Games come from a `JobSource`
//! (020-R12): a matchup plan is a fixed queue (`QueueSource`), the RL league (042) picks each
//! game when a host slot frees up. Used by the `uranium-env` binary and by `uranium-train rl`.

/// `eprintln!` that never panics: inside `uranium-train rl` this runs in a training child whose
/// stderr is a pipe to its watchdog; when the watchdog is killed hard the pipe closes and
/// `eprintln!` would panic (EPIPE) instead of letting the run save, push and release (060-R6).
macro_rules! elog {
    ($($t:tt)*) => {{
        use std::io::Write as _;
        let _ = writeln!(std::io::stderr(), $($t)*);
    }};
}

pub mod calibrate;
pub mod eval;
pub mod plan;
pub mod report;

use std::collections::{HashMap, HashSet, VecDeque};
use std::io::{BufReader, BufWriter, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde::Serialize;
use uranium_agent::infer::InferenceHandle;
use uranium_agent::{EndBoard, PolicySpec, Routes, ServeOptions, T0Build, T0Builds, open_sink, serve};
use uranium_proto::control::*;
use uranium_proto::{read_msg, write_msg};

pub use crate::plan::{HostEntry, Job, ModelKey};

/// Model servers by (checkpoint, sampling mode) (020-R11).
pub type Models = HashMap<ModelKey, PolicySpec>;

/// hostd's verdict on a game that was handed to a host (020-R4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum JobOutcome {
    Finished,
    Aborted,
    Timeout,
    Stopped,
    Rejected,
}

impl JobOutcome {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Finished => "Finished",
            Self::Aborted => "Aborted",
            Self::Timeout => "Timeout",
            Self::Stopped => "Stopped",
            Self::Rejected => "Rejected",
        }
    }
}

/// Where workers get their games (020-R12).
pub trait JobSource: Send + Sync {
    /// The next game for a free host slot; `None` ends that worker.
    fn next(&self) -> Option<Job>;
    /// A game that never started (its host was lost): play it elsewhere.
    fn requeue(&self, job: Job);
    /// hostd's outcome for a game that was started (or rejected), with the replay hostd sent
    /// back for a `save_replay` game (020-R1).
    fn done(&self, _job: &Job, _outcome: JobOutcome, _replay: Option<Vec<u8>>) {}
    /// [`JobSource::done`] with the recorded detail (020-R15 retries depend on it).
    fn done_with(&self, job: &Job, outcome: JobOutcome, _detail: Option<&str>, replay: Option<Vec<u8>>) {
        self.done(job, outcome, replay)
    }
    /// Games still waiting (reported as "not run" when the workers stop).
    fn left(&self) -> usize {
        0
    }
}

/// A fixed list of games (matchup plans).
#[derive(Default)]
pub struct QueueSource {
    jobs: Mutex<VecDeque<Job>>,
    /// Where replays hostd sends back are written (`g<id>.rep`); `None` drops them.
    replay_dir: Option<PathBuf>,
}

impl QueueSource {
    pub fn new(jobs: Vec<Job>) -> Self {
        Self { jobs: Mutex::new(VecDeque::from(jobs)), replay_dir: None }
    }

    /// 020-R5: replays of `save_replay` games land in `dir` on the scheduler's PC, wherever the
    /// game ran.
    pub fn with_replays(mut self, dir: PathBuf) -> Self {
        self.replay_dir = Some(dir);
        self
    }
}

impl JobSource for QueueSource {
    fn next(&self) -> Option<Job> {
        self.jobs.lock().unwrap().pop_front()
    }
    fn requeue(&self, job: Job) {
        self.jobs.lock().unwrap().push_front(job);
    }
    fn done(&self, job: &Job, _outcome: JobOutcome, replay: Option<Vec<u8>>) {
        let (Some(dir), Some(bytes)) = (&self.replay_dir, replay) else { return };
        let path = dir.join(format!("g{}.rep", job.spec.game_id));
        if let Err(e) = std::fs::create_dir_all(dir).and_then(|_| std::fs::write(&path, bytes)) {
            elog!("[env] could not write {}: {e}", path.display());
        }
    }
    fn left(&self) -> usize {
        self.jobs.lock().unwrap().len()
    }
}

/// Control connections of games in flight, for `StopGame` on a clean stop (042-R10).
#[derive(Clone, Default)]
pub struct InFlight {
    games: Arc<Mutex<HashMap<u64, Arc<Mutex<BufWriter<TcpStream>>>>>>,
    stopping: Arc<AtomicBool>,
}

impl InFlight {
    /// Sends `StopGame` for every game in flight; hostd ends them as `Stopped` (042-R10 ②).
    pub fn stop_all(&self) {
        self.stopping.store(true, Ordering::SeqCst);
        for (id, w) in self.games.lock().unwrap().iter() {
            let mut w = w.lock().unwrap();
            if let Err(e) = write_msg(&mut *w, &ControlRequest::StopGame { game_id: *id }) {
                elog!("[env] StopGame {id} failed: {e}");
            }
        }
    }

    pub fn len(&self) -> usize {
        self.games.lock().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Shared state every worker needs.
#[derive(Clone)]
pub struct Dispatch {
    pub routes: Routes,
    pub models: Models,
    pub t0_builds: T0Builds,
    pub inflight: InFlight,
    /// The board of the agent this scheduler started (020-R4 bridge check); `None` when the
    /// caller reads the board itself (042 league).
    pub ends: Option<EndBoard>,
    /// 020-R15 ⑥: hosts that get no new game (their worker is leaving); a slot worker on one of
    /// them ends after its game.
    pub leaving: Arc<Mutex<HashSet<String>>>,
    /// 020-R15 ⑧: what games.jsonl records about the games' inference (evaluation jobs).
    pub infer: Option<InferTag>,
}

/// 020-R15 ⑧ / ⑦: the inference host and backend of an evaluation job's games, and its stage
/// names (a game's stage comes from its hostd id, [`eval::split_id`]).
#[derive(Clone, Debug)]
pub struct InferTag {
    pub host: String,
    pub backend: String,
    pub stages: Arc<Vec<String>>,
}

impl Dispatch {
    pub fn new(routes: Routes, models: Models, t0_builds: T0Builds) -> Self {
        Self { routes, models, t0_builds, inflight: InFlight::default(), ends: None, leaving: Arc::default(), infer: None }
    }

    /// Checks finished games against `ends` (020-R4) and drops their reports afterwards.
    pub fn with_ends(mut self, ends: EndBoard) -> Self {
        self.ends = Some(ends);
        self
    }
}

/// 020-R4: hostd reports `Finished` for a game whose bridge gave up reaching the agent (the shim
/// leaves, the launcher exits 0). Such a game has fewer sessions past the Hello check than bridge
/// players; it is recorded as Aborted with this detail, and an evaluation job plays it again
/// (020-R15).
pub const BRIDGE_NEVER_CONNECTED: &str = "bridge never connected";

fn bridge_check(spec: &GameSpec, ends: &EndBoard) -> Option<String> {
    let bridges = spec.players.iter().filter(|p| matches!(p.kind, PlayerKind::Bridge { .. })).count();
    let connected = ends.connected(spec.game_id);
    ends.take(spec.game_id);
    (connected < bridges).then(|| format!("{BRIDGE_NEVER_CONNECTED} ({connected} of {bridges} sessions)"))
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
    /// p1's T0 build when it is an E-Regression scenario bot (050).
    #[serde(skip_serializing_if = "Option::is_none")]
    t0_build: Option<String>,
    outcome: String,
    detail: Option<String>,
    wall_s: u32,
    /// 020-R15: evaluation jobs only (stage name, inference host and backend).
    #[serde(skip_serializing_if = "Option::is_none")]
    stage: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    infer_host: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    backend: Option<String>,
}

pub fn env_or(name: &str, default: &str) -> String {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty()).unwrap_or_else(|| default.to_string())
}

/// `URANIUM_HOSTS_FILE`, else the local hostd (020).
pub fn hosts() -> Result<Vec<HostEntry>> {
    match std::env::var("URANIUM_HOSTS_FILE").ok().filter(|v| !v.is_empty()) {
        Some(f) => plan::load_hosts(f.as_ref()),
        None => Ok(vec![HostEntry { addr: env_or("URANIUM_HOSTD_LISTEN", "127.0.0.1:7700"), max_games: None }]),
    }
}

/// The (checkpoint, sampling mode) pairs the plan's players use, each once (020-R11). Their
/// inference servers run in the Python process (`python -m uranium run`, AMB-000-4).
pub fn plan_models(plan: &plan::Plan) -> Vec<ModelKey> {
    let mut keys = Vec::new();
    for m in &plan.matchups {
        for key in [m.p0.model(), m.p1.model()].into_iter().flatten() {
            if !keys.contains(&key) {
                keys.push(key);
            }
        }
    }
    keys
}

/// Model policies of the plan from one inference handle per [`plan_models`] key, with the plan's
/// session settings: 031-R12 / 020-R14 surrender mode and chat line, 030-R10 / E-5
/// `last_failed = 0` for BC evaluation plans (reason at `Plan::zero_last_failed`).
pub fn plan_policies(plan: &plan::Plan, handles: &HashMap<ModelKey, InferenceHandle>) -> Result<Models> {
    let mut models = Models::new();
    for key in plan_models(plan) {
        let h = handles.get(&key).with_context(|| format!("no inference server for {} ({:?})", key.0, key.1))?;
        let mut h = h.clone().with_surrender(plan.surrender, &plan.surrender_chat);
        h.zero_last_failed = plan.zero_last_failed;
        models.insert(key, PolicySpec::Model(h));
    }
    Ok(models)
}

/// Plays every game of a plan (020) with the agent server in-process and writes the results and
/// report under `out` (default `runs/<plan name>`). `models`: [`plan_policies`] (empty for a plan
/// of scripted players only).
pub fn run_plan(plan_path: &Path, out: Option<PathBuf>, models: Models) -> Result<()> {
    let plan = plan::load_plan(plan_path)?;
    let out = out.unwrap_or_else(|| default_out(&plan));
    std::fs::create_dir_all(&out)?;
    std::fs::copy(plan_path, out.join("plan.toml"))?;
    let _ = std::fs::remove_file(out.join("games.jsonl"));
    let _ = std::fs::remove_file(out.join("results.jsonl"));

    let builds = T0Builds::default();
    let routes = Routes::default();
    let ends = EndBoard::default();
    let public = start_agent(&env_or("URANIUM_AGENT_LISTEN", "127.0.0.1:7600"), &out.join("results.jsonl"), routes.clone(), builds.clone(), ends.clone())?;
    let jobs = plan::expand(&plan, &public);
    elog!("[env] plan `{}`: {} games, agent at {public}", plan.name, jobs.len());
    let started = std::time::Instant::now();
    // Scenario builds (050) and model routes are registered per game by the workers (020-R11).
    let dispatch = Dispatch::new(routes, models, builds).with_ends(ends);
    let source = Arc::new(QueueSource::new(jobs).with_replays(out.join("replays")));
    run_source(source.clone(), &hosts()?, &out.join("games.jsonl"), &dispatch)?;
    let left = JobSource::left(&*source);
    elog!("[env] finished in {:.0}s ({left} games not run)", started.elapsed().as_secs_f64());
    report::write(&out, &plan)?;
    println!("{}", std::fs::read_to_string(out.join("summary.md"))?);
    if left > 0 {
        bail!("{left} games could not be scheduled");
    }
    Ok(())
}

/// Starts the embedded agent server (001-R6) and returns the address bridges connect to.
/// Bridges on every host must reach it (`URANIUM_AGENT_PUBLIC_ADDR` when it differs).
pub fn start_agent(listen: &str, results: &Path, routes: Routes, t0_builds: T0Builds, ends: EndBoard) -> Result<String> {
    let listener = TcpListener::bind(listen).with_context(|| format!("binding agent {listen}"))?;
    let bound = listener.local_addr()?.to_string();
    let public = std::env::var("URANIUM_AGENT_PUBLIC_ADDR").ok().filter(|v| !v.trim().is_empty()).unwrap_or(bound);
    let sink = open_sink(results)?;
    std::thread::spawn(move || {
        if let Err(e) = serve(listener, ServeOptions { routes, t0_builds, ends, ..ServeOptions::new(PolicySpec::ScriptedT0) }, sink) {
            elog!("[env] agent server stopped: {e:#}");
        }
    });
    Ok(public)
}

/// Plays `jobs` on `hosts`, appending hostd outcomes to `games_path`. Returns the number of
/// games that could not be scheduled.
pub fn run_jobs(jobs: Vec<Job>, hosts: &[HostEntry], games_path: &Path, routes: &Routes, models: &Models) -> Result<usize> {
    let replays = games_path.parent().unwrap_or(Path::new(".")).join("replays");
    let source = Arc::new(QueueSource::new(jobs).with_replays(replays));
    run_source(source.clone(), hosts, games_path, &Dispatch::new(routes.clone(), models.clone(), T0Builds::default()))?;
    Ok(source.left())
}

/// Plays games from `source` on `hosts` until every worker got `None` (or lost its host).
pub fn run_source(source: Arc<dyn JobSource>, hosts: &[HostEntry], games_path: &Path, dispatch: &Dispatch) -> Result<()> {
    let file = std::fs::OpenOptions::new().create(true).append(true).open(games_path)?;
    let games_out = Arc::new(Mutex::new(BufWriter::new(file)));
    let done = Arc::new(Mutex::new(0usize));
    let mut workers = Vec::new();
    for host in hosts {
        // 020-R4: an unreachable host is skipped, the others still play.
        let caps = match capabilities(&host.addr) {
            Ok(c) => c,
            Err(e) => {
                elog!("[env] WARNING: hostd {} unreachable, skipping it: {e:#}", host.addr);
                continue;
            }
        };
        let slots = host.max_games.map_or(caps.max_concurrent_games, |m| m.min(caps.max_concurrent_games));
        elog!("[env] host {}: {:?}, {} slots, oss bots {:?}", host.addr, caps.os, slots, caps.oss_bots);
        let ids: Arc<HashMap<String, String>> = Arc::new(caps.oss_bots.iter().map(|id| (id.split('@').next().unwrap_or(id).to_string(), id.clone())).collect());
        for _ in 0..slots {
            let (addr, src, g, d, ids, dispatch) = (host.addr.clone(), source.clone(), games_out.clone(), done.clone(), ids.clone(), dispatch.clone());
            workers.push(std::thread::spawn(move || worker(&addr, &*src, &g, &d, &ids, &dispatch)));
        }
    }
    if workers.is_empty() {
        bail!("no reachable hostd among {} configured host(s)", hosts.len());
    }
    for w in workers {
        if let Err(e) = w.join().expect("worker panicked") {
            elog!("[env] worker failed: {e:#}");
        }
    }
    games_out.lock().unwrap().flush()?;
    // Session records are written when each bridge sends GameEnd, before its launcher exits.
    std::thread::sleep(Duration::from_secs(1));
    Ok(())
}

fn connect(addr: &str) -> Result<(BufReader<TcpStream>, BufWriter<TcpStream>)> {
    // A powered-off host does not refuse, it never answers: without a limit the OS waits about
    // two minutes per try, and an evaluation job probes absent hosts every minute (020-R15 ⑥).
    use std::net::ToSocketAddrs;
    let target = addr.to_socket_addrs()?.next().with_context(|| format!("{addr}: no address"))?;
    let s = TcpStream::connect_timeout(&target, Duration::from_secs(5))?;
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

/// How many games hostd runs now (020-R1 `Status`; 020-R15 ⑥ waits for 0 before stopping it).
pub fn host_status(addr: &str) -> Result<usize> {
    let (mut r, mut w) = connect(addr)?;
    write_msg(&mut w, &ControlRequest::Status)?;
    match read_msg(&mut r)? {
        ControlResponse::Status { running, .. } => Ok(running.len()),
        other => bail!("unexpected {other:?}"),
    }
}

pub fn capabilities(addr: &str) -> Result<Capabilities> {
    let (mut r, mut w) = connect(addr)?;
    write_msg(&mut w, &ControlRequest::ListCapabilities)?;
    match read_msg(&mut r)? {
        ControlResponse::Capabilities(c) => Ok(c),
        other => bail!("unexpected {other:?}"),
    }
}

pub(crate) fn worker(addr: &str, source: &dyn JobSource, games: &Mutex<BufWriter<std::fs::File>>, done: &Mutex<usize>, ids: &HashMap<String, String>, dispatch: &Dispatch) -> Result<()> {
    let (mut r, w) = connect(addr)?;
    let w = Arc::new(Mutex::new(w));
    loop {
        if dispatch.inflight.stopping.load(Ordering::SeqCst) || dispatch.leaving.lock().unwrap().contains(addr) {
            return Ok(());
        }
        let Some(job) = source.next() else { return Ok(()) };
        // 020-R15 ⑥: the host started leaving while this slot waited for a game.
        if dispatch.leaving.lock().unwrap().contains(addr) {
            source.requeue(job);
            return Ok(());
        }
        // A model player without its server would quietly play as the agent's default T0 bot.
        if let Some(k) = (0..2).find_map(|s| job.models[s].as_ref().filter(|k| job.policies[s].is_none() && !dispatch.models.contains_key(*k))) {
            let k = k.clone();
            source.requeue(job);
            bail!("no inference server for {k:?} (020-R11)");
        }
        // 020-R11: model players are routed by (game_id, slot) before the bridges connect.
        for slot in 0..2 {
            let spec = job.policies[slot].clone().or_else(|| job.models[slot].as_ref().and_then(|k| dispatch.models.get(k)).cloned());
            if let Some(spec) = spec {
                dispatch.routes.insert(job.spec.game_id, slot as u8, spec);
            }
            if job.t0_builds[slot] != T0Build::Standard {
                dispatch.t0_builds.lock().unwrap().insert((job.spec.game_id, slot as u8), job.t0_builds[slot]);
            }
        }
        let reply = {
            let mut wl = w.lock().unwrap();
            write_msg(&mut *wl, &ControlRequest::StartGame(job.spec.clone()))
        }
        .and_then(|_| read_msg(&mut r));
        let reply = match reply {
            Ok(m) => m,
            Err(e) => {
                // The game never started: leave it for the remaining hosts (020-R4).
                source.requeue(job);
                return Err(anyhow::Error::from(e).context(format!("hostd {addr} lost")));
            }
        };
        match reply {
            ControlResponse::Started { .. } => {}
            ControlResponse::Rejected { reason } => {
                elog!("[env] {addr} rejected game {}: {reason}", job.spec.game_id);
                if reason.contains("capacity") {
                    source.requeue(job);
                    std::thread::sleep(Duration::from_secs(2));
                    continue;
                }
                let ok = record_retry(games, addr, &job, ids, dispatch, JobOutcome::Rejected, Some(reason.clone()), 0);
                report_done(source, &job, ok, JobOutcome::Rejected, Some(&reason), None);
                continue;
            }
            other => {
                source.requeue(job);
                bail!("unexpected {other:?}");
            }
        }
        dispatch.inflight.games.lock().unwrap().insert(job.spec.game_id, w.clone());
        if dispatch.inflight.stopping.load(Ordering::SeqCst) {
            let mut wl = w.lock().unwrap();
            write_msg(&mut *wl, &ControlRequest::StopGame { game_id: job.spec.game_id }).ok();
        }
        let fin = loop {
            match read_msg(&mut r) {
                Ok(ControlResponse::Finished(f)) if f.game_id == job.spec.game_id => break f,
                Ok(other) => elog!("[env] ignoring {other:?}"),
                Err(e) => {
                    dispatch.inflight.games.lock().unwrap().remove(&job.spec.game_id);
                    // 020-R4: a game in flight on a lost host is Aborted, not silently dropped.
                    let why = format!("{}: {e}", eval::HOSTD_LOST);
                    let ok = record_retry(games, addr, &job, ids, dispatch, JobOutcome::Aborted, Some(why.clone()), 0);
                    report_done(source, &job, ok, JobOutcome::Aborted, Some(&why), None);
                    *done.lock().unwrap() += 1;
                    elog!("[env] game {} Aborted: hostd {addr} lost", job.spec.game_id);
                    return Err(anyhow::Error::from(e).context(format!("hostd {addr} lost")));
                }
            }
        };
        dispatch.inflight.games.lock().unwrap().remove(&job.spec.game_id);
        let (outcome, detail) = match &fin.outcome {
            GameOutcome::Finished => match dispatch.ends.as_ref().and_then(|e| bridge_check(&job.spec, e)) {
                Some(why) => (JobOutcome::Aborted, Some(why)),
                None => (JobOutcome::Finished, None),
            },
            GameOutcome::Aborted { reason } => (JobOutcome::Aborted, Some(reason.clone())),
            GameOutcome::Timeout => (JobOutcome::Timeout, None),
            GameOutcome::Stopped => (JobOutcome::Stopped, None),
        };
        if let Some(e) = &dispatch.ends {
            e.take(job.spec.game_id);
        }
        let ok = record_retry(games, addr, &job, ids, dispatch, outcome, detail.clone(), fin.wall_s);
        report_done(source, &job, ok, outcome, detail.as_deref(), fin.replay);
        let mut d = done.lock().unwrap();
        *d += 1;
        elog!("[env] {} game {} {} ({}s)", *d, job.spec.game_id, outcome.as_str(), fin.wall_s);
    }
}

/// 021-R3: T0 bots are versioned with the workspace (scenario builds add their tag:
/// `t0-zerg-rush@<ver>`), OSS bots by the host's VERSION file; model players by checkpoint id
/// and sampling mode (020-R5).
pub fn opponent_id(p: &PlayerSpec, build: T0Build, model: Option<&PolicySpec>, ids: &HashMap<String, String>) -> String {
    if let Some(m) = model {
        return m.label();
    }
    match &p.kind {
        PlayerKind::Bridge { .. } => {
            let race = format!("{:?}", p.race).to_lowercase();
            match build.id_tag() {
                Some(tag) => format!("t0-{race}-{tag}@{}", env!("CARGO_PKG_VERSION")),
                None => format!("t0-{race}@{}", env!("CARGO_PKG_VERSION")),
            }
        }
        PlayerKind::OssBot { name } => ids.get(name).cloned().unwrap_or_else(|| format!("{name}@unknown")),
    }
}

/// [`record`] tried a few times (games.jsonl may sit on a shared drive, 061). Returns whether it
/// was written; the caller tells the source either way, so a game handed out is never left in
/// flight by a failed write.
#[allow(clippy::too_many_arguments)]
fn record_retry(games: &Mutex<BufWriter<std::fs::File>>, host: &str, job: &Job, ids: &HashMap<String, String>, dispatch: &Dispatch, outcome: JobOutcome, detail: Option<String>, wall_s: u32) -> bool {
    for attempt in 1..=5 {
        match record(games, host, job, ids, dispatch, outcome, detail.clone(), wall_s) {
            Ok(()) => return true,
            Err(e) if attempt < 5 => {
                elog!("[env] writing the record of game {} failed ({e:#}); retrying", job.spec.game_id);
                std::thread::sleep(Duration::from_secs(2 * attempt));
            }
            Err(e) => elog!("[env] ERROR: the record of game {} ({}) was not written: {e:#}", job.spec.game_id, outcome.as_str()),
        }
    }
    false
}

/// The source hears a game's end; a game whose record could not be written is reported as
/// stopped, so an evaluation job plays it again under a new id (020-R15 ④ ⑦).
fn report_done(source: &dyn JobSource, job: &Job, written: bool, outcome: JobOutcome, detail: Option<&str>, replay: Option<Vec<u8>>) {
    if written {
        source.done_with(job, outcome, detail, replay);
    } else {
        source.done_with(job, JobOutcome::Stopped, Some("record not written"), None);
    }
}

#[allow(clippy::too_many_arguments)]
fn record(games: &Mutex<BufWriter<std::fs::File>>, host: &str, job: &Job, ids: &HashMap<String, String>, dispatch: &Dispatch, outcome: JobOutcome, detail: Option<String>, wall_s: u32) -> Result<()> {
    let models = &dispatch.models;
    let model1 = job.policies[1].as_ref().or_else(|| job.models[1].as_ref().and_then(|k| models.get(k)));
    let p1_kind = match (&job.spec.players[1].kind, model1) {
        (PlayerKind::Bridge { .. }, Some(_)) => "model".to_string(),
        (PlayerKind::Bridge { .. }, None) => "t0".to_string(),
        (PlayerKind::OssBot { name }, _) => format!("oss:{name}"),
    };
    let rec = GameRecord {
        game_id: job.spec.game_id,
        matchup: job.matchup,
        host: host.to_string(),
        map: format!("{:?}", job.spec.map),
        races: [format!("{:?}", job.spec.players[0].race), format!("{:?}", job.spec.players[1].race)],
        p1_kind,
        opponent: job.opponent.clone().unwrap_or_else(|| opponent_id(&job.spec.players[1], job.t0_builds[1], model1, ids)),
        t0_build: Some(job.t0_builds[1]).filter(|b| *b != T0Build::Standard).map(|b| b.name().to_string()),
        outcome: outcome.as_str().to_string(),
        detail,
        wall_s,
        stage: dispatch.infer.as_ref().and_then(|t| t.stages.get(eval::split_id(job.spec.game_id).1).cloned()),
        infer_host: dispatch.infer.as_ref().map(|t| t.host.clone()),
        backend: dispatch.infer.as_ref().map(|t| t.backend.clone()),
    };
    // The whole line in one write past the buffer: a failed try leaves nothing buffered that a
    // later flush would write again.
    let line = format!("{}\n", serde_json::to_string(&rec)?);
    let mut g = games.lock().unwrap();
    g.flush()?;
    g.get_mut().write_all(line.as_bytes())?;
    Ok(())
}

/// Output directory default for a plan.
pub fn default_out(plan: &plan::Plan) -> PathBuf {
    PathBuf::from("runs").join(&plan.name)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal hostd: answers the handshake and capabilities, plays every game for `game_ms`.
    /// A `flaky` host drops the connection right after starting its first game. `StopGame` ends
    /// the running game as `Stopped`. Requests are read on their own thread (a timed-out socket
    /// read leaves a Windows socket in an undefined state).
    pub(crate) fn fake_hostd(flaky: bool, game_ms: u64) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        std::thread::spawn(move || {
            for s in listener.incoming().flatten() {
                std::thread::spawn(move || -> Result<()> {
                    let (tx, rx) = std::sync::mpsc::channel();
                    let mut r = BufReader::new(s.try_clone()?);
                    std::thread::spawn(move || {
                        while let Ok(m) = read_msg::<ControlRequest>(&mut r) {
                            if tx.send(m).is_err() {
                                break;
                            }
                        }
                    });
                    let mut w = BufWriter::new(s);
                    loop {
                        let reply = match rx.recv()? {
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
                                    // The reader thread holds a clone: close the socket itself.
                                    w.flush()?;
                                    w.get_ref().shutdown(std::net::Shutdown::Both)?;
                                    return Ok(());
                                }
                                // Play until the game time passes or a StopGame arrives.
                                let outcome = match rx.recv_timeout(Duration::from_millis(game_ms)) {
                                    Ok(ControlRequest::StopGame { .. }) => GameOutcome::Stopped,
                                    _ => GameOutcome::Finished,
                                };
                                ControlResponse::Finished(GameFinished { game_id: g.game_id, outcome, exit_codes: [Some(0), Some(0)], wall_s: 0, replay: g.save_replay.then(|| b"rep".to_vec()) })
                            }
                            _ => continue,
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
        // Port 0 can never accept a connection, so this fails on every OS. (Binding port 0 and
        // dropping the listener left a free port that a parallel test's fake hostd could reuse.)
        let dead = "127.0.0.1:0".to_string();
        let hosts: Vec<HostEntry> = [dead, fake_hostd(false, 200), fake_hostd(true, 0)].into_iter().map(|addr| HostEntry { addr, max_games: None }).collect();
        let plan: plan::Plan = toml::from_str("name='t'\n[[matchup]]\nmap='both'\ngames=4\np0={kind='t0',race='terran'}\np1={kind='t0',race='terran'}").unwrap();
        let dir = std::env::temp_dir().join(format!("uranium-env-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let games = dir.join("games.jsonl");
        let left = run_jobs(plan::expand(&plan, "127.0.0.1:1"), &hosts, &games, &Routes::default(), &Models::new()).unwrap();
        let lines: Vec<serde_json::Value> = std::fs::read_to_string(&games).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(left, 0);
        assert_eq!(lines.len(), 4, "{lines:?}");
        assert_eq!(lines.iter().filter(|g| g["outcome"] == "Aborted").count(), 1, "{lines:?}");
        assert_eq!(lines.iter().filter(|g| g["outcome"] == "Finished").count(), 3, "{lines:?}");
    }

    /// 020-R1 / R5: a `save_replay` game's replay comes back from hostd and lands next to
    /// games.jsonl, whichever host played it.
    #[test]
    fn plan_replays_come_back_from_hostd_020_r1() {
        let hosts = vec![HostEntry { addr: fake_hostd(false, 50), max_games: None }];
        let plan: plan::Plan = toml::from_str("name='t'\nsave_replay=true\n[[matchup]]\nmap='both'\ngames=2\np0={kind='t0',race='terran'}\np1={kind='t0',race='terran'}").unwrap();
        let dir = std::env::temp_dir().join(format!("uranium-env-test-rep-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let jobs = plan::expand(&plan, "127.0.0.1:1");
        let ids: Vec<u64> = jobs.iter().map(|j| j.spec.game_id).collect();
        run_jobs(jobs, &hosts, &dir.join("games.jsonl"), &Routes::default(), &Models::new()).unwrap();
        for id in ids {
            assert_eq!(std::fs::read(dir.join("replays").join(format!("g{id}.rep"))).unwrap(), b"rep");
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn no_reachable_host_is_an_error_020_r4() {
        // Port 0 can never accept a connection, so this fails on every OS. (Binding port 0 and
        // dropping the listener left a free port that a parallel test's fake hostd could reuse.)
        let dead = "127.0.0.1:0".to_string();
        let dir = std::env::temp_dir().join(format!("uranium-env-test-none-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let res = run_jobs(vec![], &[HostEntry { addr: dead, max_games: None }], &dir.join("games.jsonl"), &Routes::default(), &Models::new());
        std::fs::remove_dir_all(&dir).ok();
        assert!(res.is_err());
    }

    /// 020-R12: a dynamic source decides each game when a slot frees up and hears every outcome;
    /// `stop_all` ends the game in flight as Stopped and the workers stop asking (042-R10).
    #[test]
    fn dynamic_source_and_stop_all_020_r12() {
        struct Endless {
            next_id: Mutex<u64>,
            outcomes: Mutex<Vec<(u64, JobOutcome)>>,
            template: Job,
        }
        impl JobSource for Endless {
            fn next(&self) -> Option<Job> {
                let mut id = self.next_id.lock().unwrap();
                *id += 1;
                let mut j = self.template.clone();
                j.spec.game_id = *id;
                Some(j)
            }
            fn requeue(&self, _job: Job) {}
            fn done(&self, job: &Job, outcome: JobOutcome, _replay: Option<Vec<u8>>) {
                self.outcomes.lock().unwrap().push((job.spec.game_id, outcome));
            }
        }
        let plan: plan::Plan = toml::from_str("name='t'\n[[matchup]]\nmap='both'\ngames=1\np0={kind='t0',race='terran'}\np1={kind='t0',race='terran'}").unwrap();
        let src = Arc::new(Endless { next_id: Mutex::new(0), outcomes: Mutex::new(vec![]), template: plan::expand(&plan, "x").remove(0) });
        let hosts = vec![HostEntry { addr: fake_hostd(false, 5_000), max_games: None }];
        let dir = std::env::temp_dir().join(format!("uranium-env-test-dyn-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let dispatch = Dispatch::new(Routes::default(), Models::new(), T0Builds::default());
        let d2 = dispatch.clone();
        let stopper = std::thread::spawn(move || {
            while d2.inflight.is_empty() {
                std::thread::sleep(Duration::from_millis(10));
            }
            d2.inflight.stop_all();
        });
        let t = std::time::Instant::now();
        run_source(src.clone(), &hosts, &dir.join("games.jsonl"), &dispatch).unwrap();
        stopper.join().unwrap();
        std::fs::remove_dir_all(&dir).ok();
        assert!(t.elapsed() < Duration::from_secs(4), "stop did not end the game early");
        assert_eq!(*src.outcomes.lock().unwrap(), vec![(1, JobOutcome::Stopped)]);
    }

    /// 020-R4: a Finished game with fewer agent sessions than bridge players is recorded as
    /// Aborted "bridge never connected"; a game whose bridges all connected stays Finished.
    #[test]
    fn bridge_never_connected_is_not_finished_020_r4() {
        let plan: plan::Plan = toml::from_str("name='t'\n[[matchup]]\nmap='both'\ngames=2\np0={kind='t0',race='terran'}\np1={kind='t0',race='terran'}").unwrap();
        let hosts = vec![HostEntry { addr: fake_hostd(false, 20), max_games: None }];
        let dir = std::env::temp_dir().join(format!("uranium-env-test-bridge-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let ends = EndBoard::default();
        // The fake hostd starts no bridges: game 1 as if both sessions said Hello, game 2 one.
        ends.hello(1);
        ends.hello(1);
        ends.hello(2);
        let dispatch = Dispatch::new(Routes::default(), Models::new(), T0Builds::default()).with_ends(ends.clone());
        let games = dir.join("games.jsonl");
        run_source(Arc::new(QueueSource::new(plan::expand(&plan, "x"))), &hosts, &games, &dispatch).unwrap();
        let lines: Vec<serde_json::Value> = std::fs::read_to_string(&games).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
        std::fs::remove_dir_all(&dir).ok();
        let by_id = |id: u64| lines.iter().find(|g| g["game_id"] == id).unwrap().clone();
        assert_eq!(by_id(1)["outcome"], "Finished");
        assert_eq!(by_id(2)["outcome"], "Aborted");
        assert!(by_id(2)["detail"].as_str().unwrap().starts_with(BRIDGE_NEVER_CONNECTED), "{lines:?}");
        assert_eq!((ends.connected(1), ends.connected(2)), (0, 0), "reports dropped after the game");
    }

    /// 021-R3 (2026-10-17): scenario T0 builds get their own opponent id, the standard build keeps
    /// `t0-<race>@<ver>`.
    #[test]
    fn opponent_id_021_r3_separates_t0_builds() {
        use uranium_proto::Race;
        let t0 = |race| PlayerSpec { race, kind: PlayerKind::Bridge { agent_addr: String::new() }, name: None };
        let v = env!("CARGO_PKG_VERSION");
        let ids = HashMap::new();
        assert_eq!(opponent_id(&t0(Race::Zerg), T0Build::Standard, None, &ids), format!("t0-zerg@{v}"));
        assert_eq!(opponent_id(&t0(Race::Zerg), T0Build::ZergRush, None, &ids), format!("t0-zerg-rush@{v}"));
        assert_eq!(opponent_id(&t0(Race::Terran), T0Build::TerranDrop, None, &ids), format!("t0-terran-drop@{v}"));
        assert_eq!(opponent_id(&t0(Race::Protoss), T0Build::ProtossDt, None, &ids), format!("t0-protoss-dt@{v}"));
        assert_eq!(opponent_id(&t0(Race::Zerg), T0Build::ZergMuta, None, &ids), format!("t0-zerg-muta@{v}"));
    }
}
