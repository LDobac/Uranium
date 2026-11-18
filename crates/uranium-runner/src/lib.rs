pub mod calibrate;
pub mod eval;
pub mod plan;
pub mod report;
pub mod start_slots;

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
use uranium_agent::{ConcedeBoard, EndBoard, PolicySpec, RejectionLog, Routes, ServeOptions, T0Build, T0Builds, open_sink, serve};
use uranium_match::branch::{BranchBoard, GameBranch};
use uranium_proto::control::*;
use uranium_proto::{read_msg, write_msg};
use uranium_store::elog;

pub use crate::plan::{HostEntry, Job, ModelKey};

pub type Models = HashMap<ModelKey, PolicySpec>;

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

pub trait JobSource: Send + Sync {
    fn next(&self) -> Option<Job>;
    fn requeue(&self, job: Job);
    fn done(&self, _job: &Job, _outcome: JobOutcome, _replay: Option<Vec<u8>>) {}
    fn done_with(&self, job: &Job, outcome: JobOutcome, _detail: Option<&str>, replay: Option<Vec<u8>>) {
        self.done(job, outcome, replay)
    }
    fn left(&self) -> usize {
        0
    }
    fn memory_log(&self, _job: &Job) -> Option<PathBuf> {
        None
    }
    fn game_branch(&self, _job: &Job) -> Result<Option<GameBranch>> {
        Ok(None)
    }
}

pub const AGENT_READS_BRANCH_BOARD: bool = true;

pub fn check_branch_support(plan: &plan::Plan) -> Result<()> {
    if plan.uses_branch_keys() && !AGENT_READS_BRANCH_BOARD {
        bail!("plan `{}` uses branch-game keys (act_log / snapshots / branches / verify_state / explore, 020-R22) but this agent build does not act on them yet (051)", plan.name);
    }
    Ok(())
}

pub fn memory_log_name(id: u64) -> String {
    format!("g{id}.jsonl")
}

#[derive(Default)]
pub struct QueueSource {
    jobs: Mutex<VecDeque<Job>>,
    replay_dir: Option<PathBuf>,
    memory_dir: Option<PathBuf>,
    branch: Option<(Arc<plan::Plan>, PathBuf, Option<PathBuf>)>,
}

impl QueueSource {
    pub fn new(jobs: Vec<Job>) -> Self {
        Self { jobs: Mutex::new(VecDeque::from(jobs)), replay_dir: None, memory_dir: None, branch: None }
    }

    pub fn with_branch(mut self, plan: Arc<plan::Plan>, out: PathBuf, shared: Option<PathBuf>) -> Self {
        if plan.uses_branch_keys() {
            self.branch = Some((plan, out, shared));
        }
        self
    }

    pub fn with_replays(mut self, dir: PathBuf) -> Self {
        self.replay_dir = Some(dir);
        self
    }

    pub fn with_memory_logs(mut self, dir: PathBuf) -> Self {
        self.memory_dir = Some(dir);
        self
    }
}

impl JobSource for QueueSource {
    fn memory_log(&self, job: &Job) -> Option<PathBuf> {
        self.memory_dir.as_ref().map(|d| d.join(memory_log_name(job.spec.game_id)))
    }
    fn game_branch(&self, job: &Job) -> Result<Option<GameBranch>> {
        match &self.branch {
            Some((plan, out, shared)) => plan.game_branch(job.spec.game_id, out, shared.as_deref()),
            None => Ok(None),
        }
    }
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

#[derive(Clone, Default)]
pub struct InFlight {
    games: Arc<Mutex<HashMap<u64, Arc<Mutex<BufWriter<TcpStream>>>>>>,
    stopping: Arc<AtomicBool>,
}

impl InFlight {
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

#[derive(Clone)]
pub struct Dispatch {
    pub routes: Routes,
    pub models: Models,
    pub t0_builds: T0Builds,
    pub inflight: InFlight,
    pub ends: Option<EndBoard>,
    pub leaving: Arc<Mutex<HashSet<String>>>,
    pub infer: Option<InferTag>,
    pub branches: BranchBoard,
}

#[derive(Clone, Debug)]
pub struct InferTag {
    pub host: String,
    pub backend: String,
    pub stages: Arc<Vec<String>>,
}

impl Dispatch {
    pub fn new(routes: Routes, models: Models, t0_builds: T0Builds) -> Self {
        Self { routes, models, t0_builds, inflight: InFlight::default(), ends: None, leaving: Arc::default(), infer: None, branches: BranchBoard::default() }
    }

    pub fn with_branches(mut self, branches: BranchBoard) -> Self {
        self.branches = branches;
        self
    }

    pub fn with_ends(mut self, ends: EndBoard) -> Self {
        self.ends = Some(ends);
        self
    }
}

pub const BRIDGE_NEVER_CONNECTED: &str = "bridge never connected";

fn bridge_check(spec: &GameSpec, ends: &EndBoard) -> Option<String> {
    let bridges = spec.players.iter().filter(|p| matches!(p.kind, PlayerKind::Bridge { .. })).count();
    let connected = ends.connected(spec.game_id);
    ends.take(spec.game_id);
    (connected < bridges).then(|| format!("{BRIDGE_NEVER_CONNECTED} ({connected} of {bridges} sessions)"))
}

#[derive(Serialize)]
struct GameRecord {
    game_id: u64,
    matchup: usize,
    host: String,
    map: String,
    races: [String; 2],
    p1_kind: String,
    opponent: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    t0_build: Option<String>,
    outcome: String,
    detail: Option<String>,
    wall_s: u32,
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

pub fn hosts() -> Result<Vec<HostEntry>> {
    match std::env::var("URANIUM_HOSTS_FILE").ok().filter(|v| !v.is_empty()) {
        Some(f) => plan::load_hosts(f.as_ref()),
        None => Ok(vec![HostEntry { addr: env_or("URANIUM_HOSTD_LISTEN", "127.0.0.1:7700"), max_games: None }]),
    }
}

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

pub fn run_plan(plan_path: &Path, out: Option<PathBuf>, models: Models) -> Result<()> {
    let plan = plan::load_plan(plan_path)?;
    check_branch_support(&plan)?;
    let shared = plan::shared_dir_env();
    plan.check_branch_paths(shared.as_deref())?;
    let out = out.unwrap_or_else(|| default_out(&plan));
    std::fs::create_dir_all(&out)?;
    std::fs::copy(plan_path, out.join("plan.toml"))?;
    let _ = std::fs::remove_file(out.join("games.jsonl"));
    let _ = std::fs::remove_file(out.join("results.jsonl"));
    let _ = std::fs::remove_file(out.join("unit_rejections.jsonl"));

    let builds = T0Builds::default();
    let routes = Routes::default();
    let ends = EndBoard::default();
    let branches = BranchBoard::default();
    let public = start_agent(&env_or("URANIUM_AGENT_LISTEN", "127.0.0.1:7600"), &out.join("results.jsonl"), &out.join("unit_rejections.jsonl"), routes.clone(), builds.clone(), ends.clone(), Default::default(), branches.clone())?;
    let jobs = plan::expand(&plan, &public);
    elog!("[env] plan `{}`: {} games, agent at {public}", plan.name, jobs.len());
    let started = std::time::Instant::now();
    let dispatch = Dispatch::new(routes, models, builds).with_ends(ends).with_branches(branches);
    let mut source = QueueSource::new(jobs).with_replays(out.join("replays"));
    if plan.memory_log {
        source = source.with_memory_logs(out.join("memory"));
    }
    let source = source.with_branch(Arc::new(plan.clone()), out.clone(), shared);
    let source = Arc::new(source);
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

#[allow(clippy::too_many_arguments)]
pub fn start_agent(listen: &str, results: &Path, unit_rejections: &Path, routes: Routes, t0_builds: T0Builds, ends: EndBoard, concede: ConcedeBoard, branches: BranchBoard) -> Result<String> {
    let listener = TcpListener::bind(listen).with_context(|| format!("binding agent {listen}"))?;
    let bound = listener.local_addr()?.to_string();
    let public = std::env::var("URANIUM_AGENT_PUBLIC_ADDR").ok().filter(|v| !v.trim().is_empty()).unwrap_or(bound);
    let sink = open_sink(results)?;
    let unit_rejections = RejectionLog::new(unit_rejections.to_path_buf());
    std::thread::spawn(move || {
        if let Err(e) = serve(listener, ServeOptions { routes, t0_builds, ends, unit_rejections, concede, branches, ..ServeOptions::new(PolicySpec::ScriptedT0) }, sink) {
            elog!("[env] agent server stopped: {e:#}");
        }
    });
    Ok(public)
}

pub fn run_jobs(jobs: Vec<Job>, hosts: &[HostEntry], games_path: &Path, routes: &Routes, models: &Models) -> Result<usize> {
    let replays = games_path.parent().unwrap_or(Path::new(".")).join("replays");
    let source = Arc::new(QueueSource::new(jobs).with_replays(replays));
    run_source(source.clone(), hosts, games_path, &Dispatch::new(routes.clone(), models.clone(), T0Builds::default()))?;
    Ok(source.left())
}

pub fn run_source(source: Arc<dyn JobSource>, hosts: &[HostEntry], games_path: &Path, dispatch: &Dispatch) -> Result<()> {
    let file = std::fs::OpenOptions::new().create(true).append(true).open(games_path)?;
    let games_out = Arc::new(Mutex::new(BufWriter::new(file)));
    let done = Arc::new(Mutex::new(0usize));
    let mut workers = Vec::new();
    for host in hosts {
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
    std::thread::sleep(Duration::from_secs(1));
    Ok(())
}

fn connect(addr: &str) -> Result<(BufReader<TcpStream>, BufWriter<TcpStream>)> {
    use std::net::ToSocketAddrs;
    let target = addr.to_socket_addrs()?.next().with_context(|| format!("{addr}: no address"))?;
    let s = TcpStream::connect_timeout(&target, Duration::from_secs(5))?;
    s.set_nodelay(true).ok();
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
        if dispatch.leaving.lock().unwrap().contains(addr) {
            source.requeue(job);
            return Ok(());
        }
        if let Some(k) = (0..2).find_map(|s| job.models[s].as_ref().filter(|k| job.policies[s].is_none() && !dispatch.models.contains_key(*k))) {
            let k = k.clone();
            source.requeue(job);
            bail!("no inference server for {k:?} (020-R11)");
        }
        let branch = match source.game_branch(&job).and_then(|b| prepare_branch_dirs(b.as_ref()).map(|_| b)) {
            Ok(b) => b,
            Err(e) => {
                source.requeue(job);
                return Err(e.context("branch-game settings (020-R22)"));
            }
        };
        let mut branch_entry = BoardEntry::insert(&dispatch.branches, job.spec.game_id, branch);
        let memory_log = source.memory_log(&job);
        for slot in 0..2 {
            let spec = job.policies[slot].clone().or_else(|| job.models[slot].as_ref().and_then(|k| dispatch.models.get(k)).cloned());
            let spec = match (spec, &memory_log) {
                (Some(PolicySpec::Model(h)), Some(p)) => Some(PolicySpec::Model(h.with_memory_log(p.clone()))),
                (s, _) => s,
            };
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
                branch_entry.release();
                source.requeue(job);
                return Err(anyhow::Error::from(e).context(format!("hostd {addr} lost")));
            }
        };
        match reply {
            ControlResponse::Started { .. } => {}
            ControlResponse::Rejected { reason } => {
                elog!("[env] {addr} rejected game {}: {reason}", job.spec.game_id);
                if reason.contains("capacity") {
                    branch_entry.release();
                    source.requeue(job);
                    std::thread::sleep(Duration::from_secs(2));
                    continue;
                }
                let ok = record_retry(games, addr, &job, ids, dispatch, JobOutcome::Rejected, Some(reason.clone()), 0);
                report_done(source, &job, ok, JobOutcome::Rejected, Some(&reason), None);
                continue;
            }
            other => {
                branch_entry.release();
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

struct BoardEntry<'a> {
    board: &'a BranchBoard,
    id: u64,
    mine: Option<Arc<GameBranch>>,
}

impl<'a> BoardEntry<'a> {
    fn insert(board: &'a BranchBoard, id: u64, branch: Option<GameBranch>) -> Self {
        let mine = branch.map(Arc::new);
        if let Some(b) = &mine {
            board.lock().unwrap().insert(id, b.clone());
        }
        Self { board, id, mine }
    }

    fn release(&mut self) {
        let Some(mine) = self.mine.take() else { return };
        if let Ok(mut b) = self.board.lock()
            && b.get(&self.id).is_some_and(|e| Arc::ptr_eq(e, &mine))
        {
            b.remove(&self.id);
        }
    }
}

impl Drop for BoardEntry<'_> {
    fn drop(&mut self) {
        self.release();
    }
}

fn prepare_branch_dirs(b: Option<&GameBranch>) -> Result<()> {
    for d in b.into_iter().flat_map(|b| [&b.acts_dir, &b.states_dir]).flatten() {
        std::fs::create_dir_all(d).with_context(|| format!("creating {}", d.display()))?;
    }
    Ok(())
}

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
    let line = format!("{}\n", serde_json::to_string(&rec)?);
    let mut g = games.lock().unwrap();
    g.flush()?;
    g.get_mut().write_all(line.as_bytes())?;
    g.get_ref().sync_data()?;
    Ok(())
}

pub fn default_out(plan: &plan::Plan) -> PathBuf {
    PathBuf::from("runs").join(&plan.name)
}

#[cfg(test)]
mod tests {
    use super::*;

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
                                    w.flush()?;
                                    w.get_ref().shutdown(std::net::Shutdown::Both)?;
                                    return Ok(());
                                }
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
        let dead = "127.0.0.1:0".to_string();
        let dir = std::env::temp_dir().join(format!("uranium-env-test-none-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let res = run_jobs(vec![], &[HostEntry { addr: dead, max_games: None }], &dir.join("games.jsonl"), &Routes::default(), &Models::new());
        std::fs::remove_dir_all(&dir).ok();
        assert!(res.is_err());
    }

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

    #[test]
    fn bridge_never_connected_is_not_finished_020_r4() {
        let plan: plan::Plan = toml::from_str("name='t'\n[[matchup]]\nmap='both'\ngames=2\np0={kind='t0',race='terran'}\np1={kind='t0',race='terran'}").unwrap();
        let hosts = vec![HostEntry { addr: fake_hostd(false, 20), max_games: None }];
        let dir = std::env::temp_dir().join(format!("uranium-env-test-bridge-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let ends = EndBoard::default();
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

    #[test]
    fn game_branch_on_the_board_020_r22() {
        struct Spy {
            inner: QueueSource,
            board: BranchBoard,
            seen: Mutex<Vec<(u64, Option<GameBranch>)>>,
        }
        impl JobSource for Spy {
            fn next(&self) -> Option<Job> {
                self.inner.next()
            }
            fn requeue(&self, job: Job) {
                self.inner.requeue(job)
            }
            fn game_branch(&self, job: &Job) -> Result<Option<GameBranch>> {
                self.inner.game_branch(job)
            }
            fn done(&self, job: &Job, _outcome: JobOutcome, _replay: Option<Vec<u8>>) {
                let g = self.board.lock().unwrap().get(&job.spec.game_id).map(|g| (**g).clone());
                self.seen.lock().unwrap().push((job.spec.game_id, g));
            }
        }
        let games = "[[matchup]]\nmap='both'\ngames=2\np0={kind='t0',race='terran'}\np1={kind='t0',race='zerg'}";
        for (head, logged) in [("", false), ("act_log=true\n", true)] {
            let plan: plan::Plan = toml::from_str(&format!("name='t'\n{head}{games}")).unwrap();
            let dir = std::env::temp_dir().join(format!("uranium-env-test-branch-{logged}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            let board = BranchBoard::default();
            let inner = QueueSource::new(plan::expand(&plan, "x")).with_branch(Arc::new(plan.clone()), dir.clone(), None);
            let spy = Arc::new(Spy { inner, board: board.clone(), seen: Mutex::new(vec![]) });
            let dispatch = Dispatch::new(Routes::default(), Models::new(), T0Builds::default()).with_branches(board.clone());
            run_source(spy.clone(), &[HostEntry { addr: fake_hostd(false, 20), max_games: None }], &dir.join("games.jsonl"), &dispatch).unwrap();
            let mut seen = spy.seen.lock().unwrap().clone();
            seen.sort_by_key(|(id, _)| *id);
            assert_eq!(seen.len(), 2);
            for (id, g) in seen {
                match g {
                    Some(g) => {
                        assert!(logged);
                        assert_eq!((g.plan_game as u64, g.act_log_path(0)), (id, Some(dir.join("acts").join(format!("g{id}-p0.jsonl")))));
                    }
                    None => assert!(!logged),
                }
            }
            assert!(board.lock().unwrap().is_empty(), "entries removed after the game");
            assert_eq!(dir.join("acts").is_dir(), logged);
            std::fs::remove_dir_all(&dir).ok();
        }
    }

    #[test]
    fn board_entry_requeue_race_020_r22() {
        let board = BranchBoard::default();
        let g = |n| GameBranch { plan_game: n, ..Default::default() };
        let held = |board: &BranchBoard| board.lock().unwrap().get(&7).map(|e| e.plan_game);
        let mut a = BoardEntry::insert(&board, 7, Some(g(1)));
        a.release();
        assert_eq!(held(&board), None);
        let b = BoardEntry::insert(&board, 7, Some(g(2)));
        drop(a);
        assert_eq!(held(&board), Some(2), "A's guard leaves B's entry");
        drop(b);
        assert_eq!(held(&board), None);
        let a = BoardEntry::insert(&board, 7, Some(g(1)));
        let b = BoardEntry::insert(&board, 7, Some(g(2)));
        drop(a);
        assert_eq!(held(&board), Some(2));
        drop(b);
        assert_eq!(held(&board), None);
        let b = BoardEntry::insert(&board, 7, Some(g(2)));
        drop(BoardEntry::insert(&board, 7, None));
        assert_eq!(held(&board), Some(2));
        drop(b);
    }

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
