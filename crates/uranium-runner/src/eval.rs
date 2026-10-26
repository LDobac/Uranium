//! 020-R15 evaluation jobs: several plans played in order from a ledger on the shared drive
//! (061), by whichever worker holds the job's lease (061-R16). The lease holder alone hands out
//! new games and runs their inference; a worker with a higher `--priority` asks it to yield, and
//! it then stops handing out games, lets the lease go at once and finishes its games in flight
//! (no game changes its inference host midway). Ctrl+C finishes the games in flight and stops;
//! a second Ctrl+C stops them now and they are played again later.
//!
//! Layout under the job folder `<shared>/eval/<job>/`:
//! - `job.toml`: `stages` (plan files, relative to the job folder, played in this order) and
//!   `hosts` (020-R6 hosts file, re-read every minute).
//! - `games/<host>.jsonl`, `results/<host>.jsonl`: what each worker's scheduler and agent wrote
//!   (hostd ids, [`hostd_id`]); one writer per file.
//! - `inflight/<host>.json`: the games a worker has in flight, rewritten every few seconds.
//! - `leaving/<host addr>`: a game host that gets no new game (its worker is stopping).
//! - `<stage>/replays/g<plan id>.rep`; `<stage>/games.jsonl`, `results.jsonl`, `summary.md`,
//!   `sessions.md`: the merged view `eval-status` writes (plan ids, last attempt of each game).

use std::collections::{HashMap, HashSet, VecDeque};
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use uranium_agent::infer::InferenceHandle;
use uranium_agent::{EndBoard, Routes, T0Builds};
use uranium_store::storage::{self, ContentObserver, Lease, LockObserver, LockStatus, LockView, StorageConfig, TakeOver, Watched, YieldObserver};

use crate::plan::{self, HostEntry, ModelKey, Plan};
use crate::{BRIDGE_NEVER_CONNECTED, Dispatch, InferTag, Job, JobOutcome, JobSource, Models, capabilities};

/// 020-R15 ⑦: hostd ids are `attempt × ATTEMPT_BASE + stage number × STAGE_BASE + plan id`
/// (stage numbers from 1), so games of two stages, and two attempts of one game, never share an
/// id on a game host (work folder, socket name) or in the session records.
pub const STAGE_BASE: u64 = 100_000;
pub const ATTEMPT_BASE: u64 = 10_000_000;

pub fn hostd_id(stage: usize, plan_id: u64, attempt: u32) -> u64 {
    attempt as u64 * ATTEMPT_BASE + (stage as u64 + 1) * STAGE_BASE + plan_id
}

/// (attempt, stage index, plan id) of a hostd id.
pub fn split_id(id: u64) -> (u32, usize, u64) {
    ((id / ATTEMPT_BASE) as u32, ((id % ATTEMPT_BASE) / STAGE_BASE).saturating_sub(1) as usize, id % STAGE_BASE)
}

/// A game of the job: (stage index, plan game id).
pub type Key = (usize, u64);

fn key_of(id: u64) -> Key {
    let (_, s, g) = split_id(id);
    (s, g)
}

/// 020-R15 ④: outcomes played again with the same seed (020-R4 bridge never connected, hostd
/// lost, a game stopped by a second Ctrl+C). Timeout, a game crash and other rejections stay.
pub fn retryable(outcome: &str, detail: Option<&str>) -> bool {
    outcome == JobOutcome::Stopped.as_str() || detail.is_some_and(|d| d.starts_with(BRIDGE_NEVER_CONNECTED) || d.starts_with(HOSTD_LOST))
}

/// The detail `worker` records for a game whose hostd connection broke (020-R4).
pub const HOSTD_LOST: &str = "hostd connection lost";

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
struct JobFile {
    stages: Vec<PathBuf>,
    hosts: PathBuf,
    /// 020-R15 ④ (AMB-020-6, user decision 2026-10-26): attempts per game before it is given up
    /// (it then counts with its last record, and a warning is logged).
    #[serde(default = "d_max_attempts")]
    max_attempts: u32,
}

fn d_max_attempts() -> u32 {
    3
}

pub struct Stage {
    pub name: String,
    pub plan: Plan,
}

/// A loaded evaluation job.
pub struct EvalJob {
    pub dir: PathBuf,
    /// Lease id `eval-<folder name>` (061-R16).
    pub lease_id: String,
    pub stages: Vec<Stage>,
    pub max_attempts: Option<u32>,
    hosts_file: PathBuf,
}

impl EvalJob {
    pub fn load(dir: &Path) -> Result<Self> {
        let path = dir.join("job.toml");
        let text = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        let jf: JobFile = toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        if jf.stages.is_empty() {
            bail!("{}: no stages", path.display());
        }
        let mut stages = Vec::new();
        for p in &jf.stages {
            let plan = plan::load_plan(&dir.join(p))?;
            if stages.iter().any(|s: &Stage| s.name == plan.name) {
                bail!("{}: two stages named `{}`", path.display(), plan.name);
            }
            storage::check_name(&plan.name)?;
            let games: u64 = plan.matchups.iter().map(|m| m.games as u64).sum();
            if games >= STAGE_BASE {
                bail!("stage `{}` has {games} games, at most {} (020-R15 ⑦)", plan.name, STAGE_BASE - 1);
            }
            // One inference server per model serves every stage: their session settings must agree.
            let mine = crate::plan_models(&plan);
            for s in &stages {
                let shared = crate::plan_models(&s.plan).iter().any(|k| mine.contains(k));
                if shared && (s.plan.surrender != plan.surrender || s.plan.surrender_chat != plan.surrender_chat || s.plan.zero_last_failed != plan.zero_last_failed) {
                    bail!("stages `{}` and `{}` share a model but differ in surrender / surrender_chat / zero_last_failed", s.name, plan.name);
                }
            }
            stages.push(Stage { name: plan.name.clone(), plan });
        }
        if stages.len() >= (ATTEMPT_BASE / STAGE_BASE) as usize {
            bail!("too many stages ({})", stages.len());
        }
        let name = dir.file_name().map(|n| n.to_string_lossy().to_string()).context("job folder has no name")?;
        let lease_id = format!("eval-{name}");
        storage::check_name(&lease_id)?;
        if jf.max_attempts == 0 {
            bail!("{}: max_attempts must be at least 1", path.display());
        }
        Ok(Self { dir: dir.to_path_buf(), lease_id, stages, max_attempts: Some(jf.max_attempts), hosts_file: dir.join(jf.hosts) })
    }

    /// Every game of the job in play order (stage order, then plan order).
    pub fn keys(&self) -> Vec<Key> {
        let mut v = Vec::new();
        for (si, s) in self.stages.iter().enumerate() {
            let n: u64 = s.plan.matchups.iter().map(|m| m.games as u64).sum();
            v.extend((1..=n).map(|g| (si, g)));
        }
        v
    }

    pub fn stage_names(&self) -> Arc<Vec<String>> {
        Arc::new(self.stages.iter().map(|s| s.name.clone()).collect())
    }

    fn sub(&self, name: &str) -> PathBuf {
        self.dir.join(name)
    }
}

/// One games.jsonl line as the ledger needs it.
#[derive(Debug, Clone)]
pub struct Rec {
    pub id: u64,
    pub outcome: String,
    pub detail: Option<String>,
    pub line: serde_json::Value,
}

/// Lines of every `*.jsonl` in `dir` (a missing folder has none; a line being written is
/// skipped). A folder or file that cannot be read is an error, never "no records" (061-R13:
/// unknown is not empty).
fn read_dir_lines(dir: &Path) -> Result<Vec<serde_json::Value>> {
    let mut out = Vec::new();
    let rd = match std::fs::read_dir(dir) {
        Ok(rd) => rd,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(out),
        Err(e) => return Err(anyhow::Error::from(e).context(format!("listing {}", dir.display()))),
    };
    let mut files = Vec::new();
    for e in rd {
        let p = e.with_context(|| format!("listing {}", dir.display()))?.path();
        if p.extension().is_some_and(|x| x == "jsonl") {
            files.push(p);
        }
    }
    files.sort();
    for f in files {
        let text = std::fs::read_to_string(&f).with_context(|| format!("reading {}", f.display()))?;
        out.extend(text.lines().filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok()));
    }
    Ok(out)
}

/// The job's game records. A record whose stage name differs from the stage at its index means
/// `job.toml` changed under a running job: refused (020-R15 ⑦).
pub fn read_records(job: &EvalJob) -> Result<Vec<Rec>> {
    let mut out = Vec::new();
    for v in read_dir_lines(&job.sub("games"))? {
        let (Some(id), Some(outcome)) = (v["game_id"].as_u64(), v["outcome"].as_str()) else { continue };
        if let Some(stage) = v["stage"].as_str()
            && job.stages.get(split_id(id).1).is_none_or(|s| s.name != stage)
        {
            bail!("the record of game {id} names stage `{stage}` but job.toml has another stage there: the stage list changed under the job");
        }
        out.push(Rec { id, outcome: outcome.to_string(), detail: v["detail"].as_str().map(str::to_string), line: v });
    }
    Ok(out)
}

/// 020-R15 ④: what the records say about each game.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct LedgerState {
    /// Games with a final record (or given up), and the hostd id of the record that ends them.
    pub done: HashMap<Key, u64>,
    /// Records per game (the next attempt number).
    pub attempts: HashMap<Key, u32>,
    /// Records that are played again.
    pub retried: usize,
    /// Failed attempts per game: retried records other than `Stopped` (a user's second Ctrl+C
    /// is not a fault of the game, AMB-020-6).
    pub failures: HashMap<Key, u32>,
    /// Games given up after `max_attempts` failed attempts.
    pub gave_up: Vec<Key>,
}

/// A retried record that counts toward `max_attempts` (020-R15 ④).
fn is_failure(outcome: &str, detail: Option<&str>) -> bool {
    retryable(outcome, detail) && outcome != JobOutcome::Stopped.as_str()
}

fn keep_latest(m: &mut HashMap<Key, u64>, k: Key, id: u64) {
    let e = m.entry(k).or_insert(id);
    if split_id(id).0 >= split_id(*e).0 {
        *e = id;
    }
}

pub fn ledger_state(recs: &[Rec], max_attempts: Option<u32>) -> LedgerState {
    let mut st = LedgerState::default();
    let mut last_retry: HashMap<Key, u64> = HashMap::new();
    for r in recs {
        let k = key_of(r.id);
        let a = st.attempts.entry(k).or_default();
        *a = (*a).max(split_id(r.id).0 + 1);
        if retryable(&r.outcome, r.detail.as_deref()) {
            st.retried += 1;
            keep_latest(&mut last_retry, k, r.id);
            if is_failure(&r.outcome, r.detail.as_deref()) {
                *st.failures.entry(k).or_default() += 1;
            }
        } else {
            keep_latest(&mut st.done, k, r.id);
        }
    }
    if let Some(m) = max_attempts {
        for (k, id) in last_retry {
            if !st.done.contains_key(&k) && st.failures.get(&k).copied().unwrap_or(0) >= m {
                st.done.insert(k, id);
                st.gave_up.push(k);
            }
        }
        st.gave_up.sort_unstable();
    }
    st
}

/// Games still to hand out, in play order: no final record and not in flight anywhere.
pub fn pending(keys: &[Key], st: &LedgerState, busy: &HashSet<Key>) -> Vec<Key> {
    keys.iter().copied().filter(|k| !st.done.contains_key(k) && !busy.contains(k)).collect()
}

/// `inflight/<host>.json` (020-R15 ①).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct Inflight {
    pub host: String,
    pub pid: u32,
    pub time: u64,
    pub games: Vec<u64>,
}

fn now_secs() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// Every worker's in-flight file read now (a missing folder: none). A file that cannot be read
/// or parsed is an error: its games would otherwise look free (061-R13 "unknown").
fn read_inflight(job: &EvalJob) -> Result<Vec<Inflight>> {
    let dir = job.sub("inflight");
    let rd = match std::fs::read_dir(&dir) {
        Ok(rd) => rd,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(anyhow::Error::from(e).context(format!("listing {}", dir.display()))),
    };
    let mut v = Vec::new();
    for e in rd {
        let p = e.with_context(|| format!("listing {}", dir.display()))?.path();
        if p.extension().is_none_or(|x| x != "json") {
            continue;
        }
        let bytes = std::fs::read(&p).with_context(|| format!("reading {}", p.display()))?;
        v.push(serde_json::from_slice::<Inflight>(&bytes).with_context(|| format!("parsing {}", p.display()))?);
    }
    Ok(v)
}

/// What the in-flight files say: games other live workers have in flight, and the next attempt
/// number past every id in any file. That includes this host's file left by an earlier process
/// and a dead worker's: such an attempt has no record, and hostd or the session records may still
/// hold its id (020-R15 ⑦). A worker counts as gone when its file stayed exactly the same for
/// `stale_after` on this process's clock (061-R5 expiry by watching; a live worker rewrites it
/// every few seconds), and its games are handed out again.
pub struct Flights {
    pub busy: HashSet<Key>,
    pub attempts: HashMap<Key, u32>,
}

fn flights(job: &EvalJob, me: &str, obs: &mut ContentObserver<Inflight>, stale_after: Duration) -> Result<Flights> {
    let mut f = Flights { busy: HashSet::new(), attempts: HashMap::new() };
    let now = Instant::now();
    for file in read_inflight(job)? {
        if file.host == me && file.pid == std::process::id() {
            continue;
        }
        for &id in &file.games {
            let a = f.attempts.entry(key_of(id)).or_default();
            *a = (*a).max(split_id(id).0 + 1);
        }
        if file.host != me && !matches!(obs.watch(&file.host, &Ok(Some(file.clone())), now, stale_after), Watched::Expired(_)) {
            f.busy.extend(file.games.iter().map(|&id| key_of(id)));
        }
    }
    Ok(f)
}

/// 020-R15: the games of one worker's lease, handed to its slot workers. `next` waits while
/// other workers' games are in flight (they may come back for another attempt) or the ledger is
/// unknown (paused), and ends when nothing is left or the source is closed (yield, Ctrl+C, lease
/// lost). Only games of `loaded` stages (their models are served here) are handed out.
pub struct LedgerSource {
    job: Arc<EvalJob>,
    host: String,
    /// Plan jobs per stage (plan game id `i + 1` at index `i`), bridges pointed at this agent.
    jobs: Vec<Vec<Job>>,
    loaded: HashSet<usize>,
    inner: Mutex<Inner>,
    /// Serialises in-flight file writes; each writer snapshots under it, so the last rename
    /// carries the latest state.
    write: Mutex<()>,
}

struct Inner {
    /// Checked and cleared under this lock: no game is handed out once `close` returned.
    open: bool,
    /// The ledger or the lease is unknown right now (share unreachable): no new game.
    paused: bool,
    queue: VecDeque<Key>,
    /// Games left in stages whose models are not loaded here (the caller loads them next).
    unloaded: usize,
    /// In flight here: key -> hostd id.
    mine: HashMap<Key, u64>,
    /// Games this process recorded as final (a refresh can read the ledger before the record).
    finished: HashSet<Key>,
    attempts: HashMap<Key, u32>,
    failures: HashMap<Key, u32>,
    busy_elsewhere: HashSet<Key>,
    /// A refresh read the ledger at least once: before that, "nothing queued" means nothing.
    known: bool,
}

impl LedgerSource {
    pub fn new(job: Arc<EvalJob>, host: &str, agent_addr: &str, loaded: HashSet<usize>) -> Self {
        let jobs = job.stages.iter().map(|s| plan::expand(&s.plan, agent_addr)).collect();
        let inner = Inner { open: true, paused: false, queue: VecDeque::new(), unloaded: 0, mine: HashMap::new(), finished: HashSet::new(), attempts: HashMap::new(), failures: HashMap::new(), busy_elsewhere: HashSet::new(), known: false };
        Self { job, host: host.to_string(), jobs, loaded, inner: Mutex::new(inner), write: Mutex::new(()) }
    }

    /// Re-reads the ledger: what is left, attempt numbers, other workers' games in flight.
    pub fn refresh(&self, st: &LedgerState, fl: Flights) {
        let keys = self.job.keys();
        let mut g = self.inner.lock().unwrap();
        let mut busy = fl.busy.clone();
        busy.extend(g.mine.keys().copied());
        busy.extend(g.finished.iter().copied());
        let left = pending(&keys, st, &busy);
        g.unloaded = left.iter().filter(|k| !self.loaded.contains(&k.0)).count();
        g.queue = left.into_iter().filter(|k| self.loaded.contains(&k.0)).collect();
        for (k, a) in st.attempts.iter().chain(fl.attempts.iter()) {
            let e = g.attempts.entry(*k).or_default();
            *e = (*e).max(*a);
        }
        for (k, n) in &st.failures {
            let e = g.failures.entry(*k).or_default();
            *e = (*e).max(*n);
        }
        g.busy_elsewhere = fl.busy;
        g.known = true;
    }

    pub fn close(&self) {
        self.inner.lock().unwrap().open = false;
    }

    pub fn is_open(&self) -> bool {
        self.inner.lock().unwrap().open
    }

    pub fn pause(&self, paused: bool) {
        self.inner.lock().unwrap().paused = paused;
    }

    /// Games left whose models this period did not load.
    pub fn unloaded(&self) -> usize {
        self.inner.lock().unwrap().unloaded
    }

    pub fn in_flight(&self) -> Vec<u64> {
        let mut v: Vec<u64> = self.inner.lock().unwrap().mine.values().copied().collect();
        v.sort_unstable();
        v
    }

    /// Nothing left for this period: the ledger was read and nothing is to hand out or in flight
    /// here or elsewhere. While the ledger is unknown (never read, or paused) it is not complete.
    pub fn complete(&self) -> bool {
        let g = self.inner.lock().unwrap();
        g.known && !g.paused && g.queue.is_empty() && g.mine.is_empty() && g.busy_elsewhere.is_empty()
    }

    pub fn write_inflight(&self) -> Result<()> {
        let _w = self.write.lock().unwrap();
        let dir = self.job.sub("inflight");
        std::fs::create_dir_all(&dir)?;
        let f = Inflight { host: self.host.clone(), pid: std::process::id(), time: now_secs(), games: self.in_flight() };
        storage::write_atomic(&dir.join(format!("{}.json", self.host)), &serde_json::to_vec_pretty(&f)?)
    }

    fn write_inflight_logged(&self) {
        if let Err(e) = self.write_inflight() {
            elog!("[eval] could not write the in-flight file ({e:#}); retrying");
        }
    }

    /// 020-R15 ④ with `max_attempts`: a retried game goes back in the queue unless its failed
    /// attempts reached the limit (the ledger then counts it as given up, without a result).
    fn retry_or_give_up(&self, g: &mut Inner, k: Key, failed: bool) {
        if failed {
            *g.failures.entry(k).or_default() += 1;
        }
        let n = g.failures.get(&k).copied().unwrap_or(0);
        match self.job.max_attempts {
            Some(m) if n >= m => {
                elog!("[eval] WARNING: game {} of stage `{}` given up after {n} failed attempts (max_attempts, no result)", k.1, self.job.stages[k.0].name);
                g.finished.insert(k);
            }
            _ => g.queue.push_front(k),
        }
    }
}

impl JobSource for LedgerSource {
    fn next(&self) -> Option<Job> {
        loop {
            {
                let mut g = self.inner.lock().unwrap();
                if !g.open {
                    return None;
                }
                if !g.paused
                    && let Some(k) = g.queue.pop_front()
                {
                    let attempt = g.attempts.get(&k).copied().unwrap_or(0);
                    let mut job = self.jobs[k.0][(k.1 - 1) as usize].clone();
                    job.spec.game_id = hostd_id(k.0, k.1, attempt);
                    g.attempts.insert(k, attempt + 1);
                    g.mine.insert(k, job.spec.game_id);
                    drop(g);
                    self.write_inflight_logged();
                    return Some(job);
                }
                if g.known && !g.paused && g.queue.is_empty() && g.mine.is_empty() && g.busy_elsewhere.is_empty() {
                    return None;
                }
            }
            // Other games may still come back for another attempt (refresh fills the queue).
            std::thread::sleep(Duration::from_secs(2));
        }
    }

    fn requeue(&self, job: Job) {
        let k = key_of(job.spec.game_id);
        let mut g = self.inner.lock().unwrap();
        g.mine.remove(&k);
        // Not recorded: the same attempt number is used again.
        g.attempts.insert(k, split_id(job.spec.game_id).0);
        g.queue.push_front(k);
        drop(g);
        self.write_inflight_logged();
    }

    fn done(&self, job: &Job, outcome: JobOutcome, replay: Option<Vec<u8>>) {
        self.done_with(job, outcome, None, replay)
    }

    fn done_with(&self, job: &Job, outcome: JobOutcome, detail: Option<&str>, replay: Option<Vec<u8>>) {
        let k = key_of(job.spec.game_id);
        if let Some(bytes) = replay {
            let dir = self.job.sub(&self.job.stages[k.0].name).join("replays");
            let path = dir.join(format!("g{}.rep", k.1));
            if let Err(e) = std::fs::create_dir_all(&dir).and_then(|_| std::fs::write(&path, bytes)) {
                elog!("[eval] could not write {}: {e}", path.display());
            }
        }
        // The record was written just before (worker); a retried game goes back in the queue.
        let mut g = self.inner.lock().unwrap();
        g.mine.remove(&k);
        if retryable(outcome.as_str(), detail) {
            self.retry_or_give_up(&mut g, k, is_failure(outcome.as_str(), detail));
        } else {
            g.finished.insert(k);
        }
        drop(g);
        self.write_inflight_logged();
    }

    fn left(&self) -> usize {
        self.inner.lock().unwrap().queue.len()
    }
}

/// 020-R15 ⑥: plays the source's games on the job's hosts until the source ends. The hosts file
/// is re-read and unreachable or lost hosts are probed again every `every`; hosts marked as
/// leaving get no new game.
pub fn run_hosts(source: Arc<LedgerSource>, games_path: &Path, dispatch: &Dispatch, every: Duration) -> Result<()> {
    if let Some(d) = games_path.parent() {
        std::fs::create_dir_all(d)?;
    }
    let file = std::fs::OpenOptions::new().create(true).append(true).open(games_path)?;
    let games_out = Arc::new(Mutex::new(BufWriter::new(file)));
    let done = Arc::new(Mutex::new(0usize));
    let mut live: HashMap<String, Vec<std::thread::JoinHandle<Result<()>>>> = HashMap::new();
    let mut said_down: HashSet<String> = HashSet::new();
    let active = || source.is_open() && !source.complete();
    let all_ended = |live: &HashMap<String, Vec<std::thread::JoinHandle<Result<()>>>>| live.values().flatten().all(|h| h.is_finished());
    loop {
        for (addr, hs) in live.iter_mut() {
            for h in std::mem::take(hs) {
                if h.is_finished() {
                    match h.join() {
                        Ok(Err(e)) => elog!("[eval] slot on {addr} ended: {e:#}"),
                        Err(_) => elog!("[eval] ERROR: a slot on {addr} panicked"),
                        Ok(Ok(())) => {}
                    }
                } else {
                    hs.push(h);
                }
            }
        }
        live.retain(|_, hs| !hs.is_empty());
        if let Some(l) = read_leaving(&source.job) {
            *dispatch.leaving.lock().unwrap() = l;
        }
        let leaving = dispatch.leaving.lock().unwrap().clone();
        if !active() {
            if live.is_empty() {
                break;
            }
        } else {
            let hosts: Vec<HostEntry> = plan::load_hosts(&source.job.hosts_file).unwrap_or_else(|e| {
                elog!("[eval] could not read the hosts file ({e:#}); keeping the hosts in use");
                Vec::new()
            });
            for host in hosts {
                if live.contains_key(&host.addr) || leaving.contains(&host.addr) {
                    continue;
                }
                let caps = match capabilities(&host.addr) {
                    Ok(c) => c,
                    Err(e) => {
                        if said_down.insert(host.addr.clone()) {
                            elog!("[eval] hostd {} unreachable, trying again every {}s: {e:#}", host.addr, every.as_secs());
                        }
                        continue;
                    }
                };
                said_down.remove(&host.addr);
                let slots = host.max_games.map_or(caps.max_concurrent_games, |m| m.min(caps.max_concurrent_games));
                elog!("[eval] host {}: {} slots", host.addr, slots);
                let ids: Arc<HashMap<String, String>> = Arc::new(caps.oss_bots.iter().map(|id| (id.split('@').next().unwrap_or(id).to_string(), id.clone())).collect());
                let hs = live.entry(host.addr.clone()).or_default();
                for _ in 0..slots {
                    let (addr, src, g, d, ids, dispatch) = (host.addr.clone(), source.clone(), games_out.clone(), done.clone(), ids.clone(), dispatch.clone());
                    hs.push(std::thread::spawn(move || crate::worker(&addr, &*src, &g, &d, &ids, &dispatch)));
                }
            }
        }
        // Hosts are probed every `every`, leaving markers read every 5 s (a leaving host's slots
        // end after their game); the end of the last slot is noticed within 200 ms.
        let until = Instant::now() + every;
        let mut marks = Instant::now();
        while Instant::now() < until && !(!active() && all_ended(&live)) {
            std::thread::sleep(Duration::from_millis(200));
            if marks.elapsed() >= Duration::from_secs(5) {
                marks = Instant::now();
                if let Some(l) = read_leaving(&source.job) {
                    *dispatch.leaving.lock().unwrap() = l;
                }
            }
        }
    }
    use std::io::Write;
    games_out.lock().unwrap().flush()?;
    // Session records are written when each bridge sends GameEnd, before its launcher exits.
    std::thread::sleep(Duration::from_secs(1));
    Ok(())
}

/// File name of a leaving marker for a host address (`:` is not allowed on Windows).
fn leaving_file(addr: &str) -> String {
    addr.replace([':', '/', '\\'], "_")
}

/// Hosts marked as leaving; `None` when the folder or a marker cannot be read (the caller keeps
/// the set it had: unknown is not "nobody leaves", 061-R13).
fn read_leaving(job: &EvalJob) -> Option<HashSet<String>> {
    let mut set = HashSet::new();
    let rd = match std::fs::read_dir(job.sub("leaving")) {
        Ok(rd) => rd,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Some(set),
        Err(_) => return None,
    };
    for e in rd {
        set.insert(std::fs::read_to_string(e.ok()?.path()).ok()?.trim().to_string());
    }
    Some(set)
}

pub fn mark_leaving(job: &EvalJob, addr: &str) -> Result<()> {
    let dir = job.sub("leaving");
    std::fs::create_dir_all(&dir)?;
    storage::write_atomic(&dir.join(leaving_file(addr)), addr.as_bytes())
}

pub fn clear_leaving(job: &EvalJob, addr: &str) -> Result<()> {
    match std::fs::remove_file(job.sub("leaving").join(leaving_file(addr))) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(e.into()),
        _ => Ok(()),
    }
}

/// How a worker's dispatch period ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchEnd {
    /// Every game of the job has its final record.
    Done,
    /// A higher-priority worker took over (the games in flight here were finished first).
    Yielded,
    /// Ctrl+C (020-R15 ⑤).
    Stopped,
    /// The lease went to another worker without a yield (expired while the share was away).
    Lost,
    /// Games are left whose models were not loaded: load `needed_models` again (lease kept).
    Reload,
}

impl DispatchEnd {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Done => "done",
            Self::Yielded => "yielded",
            Self::Stopped => "stopped",
            Self::Lost => "lost",
            Self::Reload => "reload",
        }
    }
}

/// How a wait for the lease ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaitEnd {
    Lease,
    Done,
    Stopped,
}

/// 061-R16: an evaluation worker's status, `workers/<host>.eval.json` (a training queue worker
/// on the same PC keeps `workers/<host>.json`).
fn status_path(cfg: &StorageConfig, host: &str) -> PathBuf {
    cfg.shared.join("workers").join(format!("{host}.eval.json"))
}

fn write_status(cfg: &StorageConfig, priority: i32, backend: &str, lease: Option<&str>) -> Result<()> {
    std::fs::create_dir_all(cfg.shared.join("workers"))?;
    let s = storage::WorkerStatus { host: cfg.host.clone(), priority, backend: backend.into(), run: lease.map(str::to_string), time: now_secs() };
    storage::write_atomic(&status_path(cfg, &cfg.host), &serde_json::to_vec_pretty(&s)?)
}

fn read_status(cfg: &StorageConfig, host: &str) -> Option<storage::WorkerStatus> {
    serde_json::from_slice(&std::fs::read(status_path(cfg, host)).ok()?).ok()
}

/// Sets a flag when dropped (a panicking slot still ends the side loop).
struct SetOnDrop<'a>(&'a AtomicBool);

impl Drop for SetOnDrop<'_> {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

/// One evaluation worker (020-R15 ②): its agent server stays up between lease periods; models
/// are loaded by the caller for each period (only while it holds the lease).
pub struct EvalWorker {
    pub job: Arc<EvalJob>,
    pub cfg: StorageConfig,
    pub priority: i32,
    pub backend: String,
    pub local_hostd: Option<String>,
    /// Ctrl+C presses ([`uranium_store::install_drain_handler`]).
    pub stop: Arc<AtomicUsize>,
    pub agent_addr: String,
    routes: Routes,
    builds: T0Builds,
    ends: EndBoard,
    lease: Arc<Mutex<Option<Lease>>>,
    /// Set by the heartbeat thread when another worker took the lease (061-R5 Lost).
    lost: Arc<AtomicBool>,
    /// The last heartbeat could not reach the share: whether the lease is still ours is unknown.
    lease_unknown: Arc<AtomicBool>,
    /// This PC's hostd is marked as leaving (once per stop).
    marked: AtomicBool,
    /// Ends the heartbeat thread.
    closed: Arc<AtomicBool>,
    /// How often hosts are probed, the ledger re-read and files rewritten (tests shorten them).
    pub every: Duration,
    pub tick: Duration,
}

impl EvalWorker {
    pub fn start(job_dir: &Path, cfg: StorageConfig, priority: i32, backend: &str, local_hostd: Option<String>, stop: Arc<AtomicUsize>, listen: &str) -> Result<Self> {
        let job = Arc::new(EvalJob::load(job_dir)?);
        storage::check_name(&cfg.host)?;
        let (routes, builds, ends) = (Routes::default(), T0Builds::default(), EndBoard::default());
        let results = job.sub("results").join(format!("{}.jsonl", cfg.host));
        let agent_addr = crate::start_agent(listen, &results, routes.clone(), builds.clone(), ends.clone())?;
        elog!("[eval] job {} ({} stages, {} games), host {} priority {} backend {}, agent at {agent_addr}", job.dir.display(), job.stages.len(), job.keys().len(), cfg.host, priority, backend);
        if let Some(addr) = &local_hostd {
            // Coming back: this host takes games again (020-R15 ⑥).
            clear_leaving(&job, addr)?;
        }
        // Every worker reads the same hosts file: a loopback entry is a different PC for each.
        for h in plan::load_hosts(&job.hosts_file).unwrap_or_default() {
            if h.addr.starts_with("127.") || h.addr.starts_with("localhost") {
                elog!("[eval] WARNING: hosts file entry {} is a loopback address; with workers on several PCs give every host its LAN address (020-R15 ①)", h.addr);
            }
        }
        let w = Self {
            job,
            cfg,
            priority,
            backend: backend.to_string(),
            local_hostd,
            stop,
            agent_addr,
            routes,
            builds,
            ends,
            lease: Arc::new(Mutex::new(None)),
            lost: Arc::new(AtomicBool::new(false)),
            lease_unknown: Arc::new(AtomicBool::new(false)),
            marked: AtomicBool::new(false),
            closed: Arc::new(AtomicBool::new(false)),
            every: Duration::from_secs(60),
            tick: Duration::from_secs(5),
        };
        w.start_heartbeat();
        Ok(w)
    }

    /// 061-R5: the lease is refreshed every `heartbeat_every` for as long as it is held, also
    /// while the caller loads and warms up models (minutes), not only while games are handed out.
    fn start_heartbeat(&self) {
        let (lease, lost, unknown, closed, cfg) = (self.lease.clone(), self.lost.clone(), self.lease_unknown.clone(), self.closed.clone(), self.cfg.clone());
        let (priority, backend, id) = (self.priority, self.backend.clone(), self.job.lease_id.clone());
        std::thread::spawn(move || {
            let mut last = Instant::now();
            while !closed.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(500));
                if last.elapsed() < cfg.heartbeat_every {
                    continue;
                }
                last = Instant::now();
                let mut g = lease.lock().unwrap();
                let Some(l) = g.as_mut() else { continue };
                match l.heartbeat() {
                    LockStatus::Lost(by) => {
                        *g = None;
                        lost.store(true, Ordering::SeqCst);
                        unknown.store(false, Ordering::SeqCst);
                        elog!("[eval] the lease went to {} (pid {}): no new game here", by.host, by.pid);
                    }
                    LockStatus::Unreachable(e) => {
                        unknown.store(true, Ordering::SeqCst);
                        elog!("[eval] lease heartbeat failed ({e}); no new game until it works again");
                    }
                    LockStatus::Held => {
                        drop(g);
                        unknown.store(false, Ordering::SeqCst);
                        if let Err(e) = write_status(&cfg, priority, &backend, Some(id.as_str())) {
                            elog!("[eval] could not write the worker status ({e:#})");
                        }
                    }
                }
            }
        });
    }

    pub fn stop_level(&self) -> usize {
        self.stop.load(Ordering::SeqCst)
    }

    fn status(&self, holding: bool) {
        if let Err(e) = write_status(&self.cfg, self.priority, &self.backend, holding.then_some(self.job.lease_id.as_str())) {
            elog!("[eval] could not write the worker status ({e:#})");
        }
    }

    /// 020-R15 ⑥: on the first Ctrl+C, whatever the worker is doing, this PC's hostd gets no new
    /// game from anyone (also when another worker holds the lease and plays there).
    fn mark_my_hostd_leaving(&self) {
        if let Some(addr) = &self.local_hostd
            && !self.marked.swap(true, Ordering::SeqCst)
        {
            match mark_leaving(&self.job, addr) {
                Ok(()) => elog!("[eval] {addr} marked as leaving: no new game goes there"),
                Err(e) => {
                    self.marked.store(false, Ordering::SeqCst);
                    elog!("[eval] could not mark {addr} as leaving ({e:#}); retrying");
                }
            }
        }
    }

    /// The ledger and the in-flight files now (in-flight files first: a game leaves a file only
    /// after its record is written). An unreadable share is an error, never "nothing there".
    fn snapshot(&self, obs: &mut ContentObserver<Inflight>) -> Result<(LedgerState, Flights)> {
        let fl = flights(&self.job, &self.cfg.host, obs, self.cfg.stale_after)?;
        Ok((ledger_state(&read_records(&self.job)?, self.job.max_attempts), fl))
    }

    /// Models of the stages that still have games to play (load these for a lease period).
    pub fn needed_models(&self) -> Result<Vec<ModelKey>> {
        let st = ledger_state(&read_records(&self.job)?, self.job.max_attempts);
        let mut keys = Vec::new();
        for (si, s) in self.job.stages.iter().enumerate() {
            let n: u64 = s.plan.matchups.iter().map(|m| m.games as u64).sum();
            if (1..=n).all(|g| st.done.contains_key(&(si, g))) {
                continue;
            }
            for k in crate::plan_models(&s.plan) {
                if !keys.contains(&k) {
                    keys.push(k);
                }
            }
        }
        Ok(keys)
    }

    /// 061-R16 / 061-R14: waits until this worker holds the lease. A free lease is taken unless
    /// another host's live yield request reserves it; a lease of a lower-priority worker gets a
    /// yield request; an expired one is taken over.
    pub fn wait_lease(&self) -> Result<WaitEnd> {
        let mut locks = LockObserver::default();
        let mut yields = YieldObserver::default();
        let mut obs = ContentObserver::<Inflight>::default();
        let mut asked = false;
        let mut said = String::new();
        let mut note = |s: String| {
            if s != said {
                elog!("[eval] {s}");
                said = s;
            }
        };
        loop {
            if self.stop_level() > 0 {
                self.mark_my_hostd_leaving();
                if asked {
                    let _ = storage::clear_yield(&self.cfg, &self.job.lease_id);
                }
                return Ok(WaitEnd::Stopped);
            }
            self.status(false);
            match self.snapshot(&mut obs) {
                Ok((st, fl)) if pending(&self.job.keys(), &st, &fl.busy).is_empty() && fl.busy.is_empty() => {
                    if asked {
                        let _ = storage::clear_yield(&self.cfg, &self.job.lease_id);
                    }
                    return Ok(WaitEnd::Done);
                }
                Ok(_) => {}
                Err(e) => note(format!("ledger unreadable ({e:#}); waiting")),
            }
            let now = Instant::now();
            let view = locks.observe(&self.job.lease_id, &storage::peek_lock(&self.cfg, &self.job.lease_id), now, self.cfg.stale_after);
            let take = match view {
                LockView::Free => {
                    let y = yields.watch(&self.job.lease_id, &storage::read_yield(&self.cfg, &self.job.lease_id), now, storage::YIELD_EXPIRY);
                    match y {
                        Watched::Fresh(y) if y.host != self.cfg.host => {
                            note(format!("lease free but reserved for {} (yield request); waiting", y.host));
                            None
                        }
                        Watched::Expired(y) => {
                            let _ = storage::remove_yield_if(&self.cfg, &self.job.lease_id, &y);
                            Some(TakeOver::No)
                        }
                        Watched::Unknown(e) => {
                            note(format!("yield request unreadable ({e}); waiting"));
                            None
                        }
                        _ => Some(TakeOver::No),
                    }
                }
                LockView::Held(l) if l.host == self.cfg.host => Some(TakeOver::No),
                LockView::Held(l) => {
                    let theirs = read_status(&self.cfg, &l.host).filter(|w| w.run.as_deref() == Some(self.job.lease_id.as_str())).map(|w| w.priority);
                    match theirs {
                        Some(p) if self.priority > p => {
                            if !asked || storage::read_yield(&self.cfg, &self.job.lease_id).ok().flatten().is_none_or(|y| y.host != self.cfg.host) {
                                match storage::request_yield(&self.cfg, &self.job.lease_id, self.priority) {
                                    Ok(()) => {
                                        asked = true;
                                        note(format!("asked {} (priority {p}) to yield the lease (061-R16)", l.host));
                                    }
                                    Err(e) => note(format!("could not write a yield request ({e:#}); retrying")),
                                }
                            }
                        }
                        _ => note(format!("lease held by {}; waiting", l.host)),
                    }
                    None
                }
                LockView::Stale(l) => Some(TakeOver::IfUnchanged(l)),
                LockView::Unknown(e) => {
                    note(format!("lease unreadable ({e}); waiting"));
                    None
                }
            };
            if let Some(take) = take {
                match Lease::acquire(self.cfg.clone(), &self.job.lease_id, take) {
                    Ok(lease) => {
                        let _ = storage::clear_yield(&self.cfg, &self.job.lease_id);
                        *self.lease.lock().unwrap() = Some(lease);
                        self.lost.store(false, Ordering::SeqCst);
                        self.lease_unknown.store(false, Ordering::SeqCst);
                        self.status(true);
                        elog!("[eval] holding the lease of {}", self.job.lease_id);
                        return Ok(WaitEnd::Lease);
                    }
                    Err(e) => note(format!("lease not taken: {e:#}")),
                }
            }
            let until = Instant::now() + self.tick * 2;
            while Instant::now() < until && self.stop_level() == 0 {
                std::thread::sleep(Duration::from_millis(200));
            }
        }
    }

    fn release(&self, why: &str) {
        if let Some(l) = self.lease.lock().unwrap().take() {
            match l.release() {
                Ok(()) => elog!("[eval] lease released ({why})"),
                Err(e) => elog!("[eval] could not remove the lease file ({e:#}); it expires (061-R5)"),
            }
        }
        self.status(false);
    }

    /// Plays games while this worker holds the lease, with `handles` (models of the stages to play
    /// in this period, [`needed_models`]). Returns once no game of this period is in flight.
    pub fn dispatch(&self, handles: &HashMap<ModelKey, InferenceHandle>) -> Result<DispatchEnd> {
        let mut models = Models::new();
        let mut loaded = HashSet::new();
        for (si, s) in self.job.stages.iter().enumerate() {
            // A stage plays only with all its models served: a model player without its server
            // would quietly play as the agent's default T0 bot (020-R11).
            if crate::plan_models(&s.plan).iter().all(|k| handles.contains_key(k)) {
                loaded.insert(si);
                for (k, p) in crate::plan_policies(&s.plan, handles)? {
                    models.entry(k).or_insert(p);
                }
            }
        }
        let mut dispatch = Dispatch::new(self.routes.clone(), models, self.builds.clone()).with_ends(self.ends.clone());
        dispatch.infer = Some(InferTag { host: self.cfg.host.clone(), backend: self.backend.clone(), stages: self.job.stage_names() });
        let source = Arc::new(LedgerSource::new(self.job.clone(), &self.cfg.host, &self.agent_addr, loaded));
        let mut obs = ContentObserver::<Inflight>::default();
        match self.snapshot(&mut obs) {
            Ok((st, fl)) => source.refresh(&st, fl),
            Err(e) => {
                elog!("[eval] ledger unreadable ({e:#}); no new game until it can be read");
                source.pause(true);
            }
        }
        source.write_inflight_logged();
        let end = Mutex::new(None::<DispatchEnd>);
        let quit = AtomicBool::new(false);
        let games = self.job.sub("games").join(format!("{}.jsonl", self.cfg.host));
        let res = std::thread::scope(|s| {
            s.spawn(|| self.watch(&source, &dispatch, &end, &quit, &mut obs));
            let _quit = SetOnDrop(&quit);
            run_hosts(source.clone(), &games, &dispatch, self.every)
        });
        source.write_inflight_logged();
        res?;
        let end = end.lock().unwrap().unwrap_or(if self.stop_level() > 0 {
            DispatchEnd::Stopped
        } else if source.unloaded() > 0 {
            DispatchEnd::Reload
        } else {
            DispatchEnd::Done
        });
        if end == DispatchEnd::Done {
            self.release("job complete");
        }
        Ok(end)
    }

    /// The lease holder's side loop: Ctrl+C, yield requests, a lost or unknown lease, ledger
    /// refresh (an unreadable ledger keeps the last queue and pauses new games), in-flight file.
    fn watch(&self, source: &LedgerSource, dispatch: &Dispatch, end: &Mutex<Option<DispatchEnd>>, quit: &AtomicBool, obs: &mut ContentObserver<Inflight>) {
        let mut forced = false;
        let mut last_refresh = Instant::now();
        let mut ledger_unknown = false;
        let set_end = |e: DispatchEnd| {
            let mut g = end.lock().unwrap();
            if g.is_none() {
                *g = Some(e);
            }
        };
        while !quit.load(Ordering::SeqCst) {
            let level = self.stop_level();
            if level >= 1 {
                self.mark_my_hostd_leaving();
            }
            if level >= 1 && source.is_open() {
                source.close();
                set_end(DispatchEnd::Stopped);
                self.release("Ctrl+C");
                elog!("[eval] finishing {} games in flight, then stopping (020-R15 ⑤)", source.in_flight().len());
            }
            if level >= 2 && !forced {
                forced = true;
                dispatch.inflight.stop_all();
            }
            if source.is_open() {
                if let Some(y) = storage::yield_by_other(&self.cfg, &self.job.lease_id) {
                    source.close();
                    set_end(DispatchEnd::Yielded);
                    self.release(&format!("yield to {} (priority {})", y.host, y.priority));
                    elog!("[eval] finishing {} games in flight here; {} plays the next games (061-R16)", source.in_flight().len(), y.host);
                } else if self.lost.load(Ordering::SeqCst) {
                    source.close();
                    set_end(DispatchEnd::Lost);
                }
            }
            if last_refresh.elapsed() >= self.tick * 6 {
                last_refresh = Instant::now();
                match self.snapshot(obs) {
                    Ok((st, fl)) => {
                        if ledger_unknown {
                            elog!("[eval] ledger readable again");
                        }
                        ledger_unknown = false;
                        if !st.gave_up.is_empty() {
                            elog!("[eval] WARNING: {} games given up after max_attempts", st.gave_up.len());
                        }
                        source.refresh(&st, fl);
                    }
                    Err(e) => {
                        if !ledger_unknown {
                            elog!("[eval] ledger unreadable ({e:#}); keeping the last queue, no new game until it can be read");
                        }
                        ledger_unknown = true;
                    }
                }
            }
            source.pause(ledger_unknown || self.lease_unknown.load(Ordering::SeqCst));
            source.write_inflight_logged();
            let until = Instant::now() + self.tick;
            while Instant::now() < until && !quit.load(Ordering::SeqCst) && self.stop_level() == level {
                std::thread::sleep(Duration::from_millis(100));
            }
        }
    }

    /// 020-R15 ⑥: before this PC's hostd (`--local-hostd`) is shut down, wait until it runs no
    /// game: another worker may hold the lease and play there. Gives up on the second Ctrl+C or
    /// when hostd cannot be reached.
    pub fn wait_local_hostd_idle(&self) {
        let Some(addr) = &self.local_hostd else { return };
        self.mark_my_hostd_leaving();
        let mut said = usize::MAX;
        loop {
            let running = match crate::host_status(addr) {
                Ok(r) => r,
                Err(e) => {
                    elog!("[eval] hostd {addr} not reachable ({e:#}); not waiting for it");
                    return;
                }
            };
            if running == 0 {
                return;
            }
            if self.stop_level() >= 2 {
                elog!("[eval] Ctrl+C again: {running} games on {addr} end with its hostd (played again later)");
                return;
            }
            if running != said {
                elog!("[eval] waiting for {running} games on {addr} before stopping its hostd (Ctrl+C again: stop now)");
                said = running;
            }
            std::thread::sleep(Duration::from_secs(5));
        }
    }

    /// Exit: lease, yield request and in-flight file of this worker go away.
    pub fn close(&self) {
        self.closed.store(true, Ordering::SeqCst);
        if self.stop_level() > 0 {
            self.mark_my_hostd_leaving();
        }
        self.release("worker stops");
        let _ = storage::clear_yield(&self.cfg, &self.job.lease_id);
        let _ = std::fs::remove_file(self.job.sub("inflight").join(format!("{}.json", self.cfg.host)));
    }
}

/// 020-R15 ⑨: progress of a job and the merged per-stage records and summaries.
pub fn status(job_dir: &Path, write_summaries: bool) -> Result<String> {
    use std::fmt::Write;
    let job = EvalJob::load(job_dir)?;
    let recs = read_records(&job)?;
    let st = ledger_state(&recs, job.max_attempts);
    let files = read_inflight(&job)?;
    let busy: HashSet<Key> = files.iter().flat_map(|f| f.games.iter().map(|&id| key_of(id))).collect();
    let mut out = String::new();
    writeln!(out, "# Evaluation job `{}`\n", job.dir.display())?;
    let cfg = StorageConfig::from_env().unwrap_or_else(|_| StorageConfig::new(job_dir, job_dir, "status"));
    match storage::peek_lock(&cfg, &job.lease_id) {
        Ok(Some(l)) => writeln!(out, "lease `{}`: {} (pid {})", job.lease_id, l.host, l.pid)?,
        Ok(None) => writeln!(out, "lease `{}`: free", job.lease_id)?,
        Err(e) => writeln!(out, "lease `{}`: unreadable ({e:#})", job.lease_id)?,
    }
    for f in &files {
        writeln!(out, "in flight on {}'s inference: {} games", f.host, f.games.len())?;
    }
    let leaving = read_leaving(&job).unwrap_or_default();
    if !leaving.is_empty() {
        let mut l: Vec<_> = leaving.into_iter().collect();
        l.sort();
        writeln!(out, "leaving hosts: {}", l.join(", "))?;
    }
    if !st.gave_up.is_empty() {
        writeln!(out, "given up after max_attempts: {} games", st.gave_up.len())?;
    }
    writeln!(out, "\n| stage | games | done | in flight | left | retried records |\n|---|---|---|---|---|---|")?;
    for (si, s) in job.stages.iter().enumerate() {
        let n: u64 = s.plan.matchups.iter().map(|m| m.games as u64).sum();
        let done = (1..=n).filter(|g| st.done.contains_key(&(si, *g))).count();
        let flying = (1..=n).filter(|g| !st.done.contains_key(&(si, *g)) && busy.contains(&(si, *g))).count();
        let retried = recs.iter().filter(|r| key_of(r.id).0 == si && retryable(&r.outcome, r.detail.as_deref())).count();
        writeln!(out, "| {} | {n} | {done} | {flying} | {} | {retried} |", s.name, n as usize - done - flying)?;
        if write_summaries && done > 0 {
            merge_stage(&job, si, &recs, &st)?;
        }
    }
    Ok(out)
}

/// Writes `<stage>/games.jsonl` / `results.jsonl` with plan ids (the last final attempt of each
/// game, `hostd_id` kept) and the 020-R13/R14 summary, plus the inference backend table.
fn merge_stage(job: &EvalJob, si: usize, recs: &[Rec], st: &LedgerState) -> Result<()> {
    use std::fmt::Write as _;
    let s = &job.stages[si];
    let dir = job.sub(&s.name);
    std::fs::create_dir_all(&dir)?;
    let finals: HashMap<u64, u64> = st.done.iter().filter(|(k, _)| k.0 == si).map(|(k, id)| (*id, k.1)).collect();
    // Given-up games (AMB-020-6) end without a result: their last attempt's sessions stay out.
    let gave_up: HashSet<u64> = st.gave_up.iter().filter(|k| k.0 == si).map(|k| st.done[k]).collect();
    let mut games: Vec<(u64, serde_json::Value)> = Vec::new();
    let mut backends: HashMap<String, usize> = HashMap::new();
    for r in recs.iter().filter(|r| finals.contains_key(&r.id)) {
        let mut v = r.line.clone();
        v["hostd_id"] = r.id.into();
        v["game_id"] = finals[&r.id].into();
        if gave_up.contains(&r.id) {
            v["gave_up"] = true.into();
        }
        games.push((finals[&r.id], v));
    }
    games.sort_by_key(|(g, _)| *g);
    games.dedup_by_key(|(g, _)| *g);
    for (_, v) in &games {
        let b = format!("{} ({})", v["backend"].as_str().unwrap_or("?"), v["infer_host"].as_str().unwrap_or("?"));
        *backends.entry(b).or_default() += 1;
    }
    let mut text = String::new();
    for (_, v) in &games {
        writeln!(text, "{v}")?;
    }
    std::fs::write(dir.join("games.jsonl"), text)?;
    let mut text = String::new();
    for mut v in read_dir_lines(&job.sub("results"))? {
        let Some(id) = v["game_id"].as_u64() else { continue };
        if let Some(g) = finals.get(&id)
            && !gave_up.contains(&id)
        {
            v["hostd_id"] = id.into();
            v["game_id"] = (*g).into();
            writeln!(text, "{v}")?;
        }
    }
    std::fs::write(dir.join("results.jsonl"), text)?;
    crate::report::write(&dir, &s.plan)?;
    let mut table = String::new();
    if !gave_up.is_empty() {
        writeln!(table, "\nGiven up after max_attempts failed attempts (no result, 020-R15 ④): {} games", gave_up.len())?;
    }
    table.push_str("\n## Inference backend (020-R15 ⑧)\n\n| backend (host) | games |\n|---|---|\n");
    let mut rows: Vec<_> = backends.into_iter().collect();
    rows.sort();
    for (b, n) in rows {
        writeln!(table, "| {b} | {n} |")?;
    }
    let summary = dir.join("summary.md");
    let mut md = std::fs::read_to_string(&summary).unwrap_or_default();
    md.push_str(&table);
    std::fs::write(summary, md)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(id: u64, outcome: &str, detail: Option<&str>) -> Rec {
        Rec { id, outcome: outcome.into(), detail: detail.map(str::to_string), line: serde_json::json!({}) }
    }

    fn no_flights() -> Flights {
        Flights { busy: HashSet::new(), attempts: HashMap::new() }
    }

    /// 020-R15 ⑦: ids keep stages and attempts apart and split back.
    #[test]
    fn hostd_ids_020_r15() {
        assert_eq!(hostd_id(0, 7, 0), 100_007);
        assert_eq!(hostd_id(3, 1999, 2), 20_401_999);
        assert_eq!(split_id(20_401_999), (2, 3, 1999));
        assert_eq!(split_id(hostd_id(0, 1, 0)), (0, 0, 1));
        assert_ne!(hostd_id(0, 5, 0), hostd_id(1, 5, 0));
        assert_ne!(hostd_id(0, 5, 0), hostd_id(0, 5, 1));
    }

    /// 020-R15 ④: final records end a game, retried ones do not; attempts count all records;
    /// `max_attempts` gives a game up with its last record.
    #[test]
    fn ledger_state_and_pending_020_r15() {
        let recs = vec![
            rec(hostd_id(0, 1, 0), "Finished", None),
            rec(hostd_id(0, 2, 0), "Aborted", Some("bridge never connected (1 of 2 sessions)")),
            rec(hostd_id(0, 3, 0), "Stopped", None),
            rec(hostd_id(0, 3, 1), "Timeout", None),
            rec(hostd_id(0, 4, 0), "Aborted", Some("hostd connection lost: io")),
            rec(hostd_id(0, 5, 0), "Aborted", Some("launcher crashed")),
            rec(hostd_id(1, 1, 0), "Rejected", Some("config quarantined")),
        ];
        let st = ledger_state(&recs, None);
        assert_eq!(st.retried, 3);
        let done: HashSet<Key> = st.done.keys().copied().collect();
        assert_eq!(done, HashSet::from([(0, 1), (0, 3), (0, 5), (1, 1)]));
        assert_eq!(st.done[&(0, 3)], hostd_id(0, 3, 1));
        assert_eq!(st.attempts[&(0, 3)], 2);
        assert_eq!(st.attempts[&(0, 2)], 1);
        let keys: Vec<Key> = (1..=6).map(|g| (0, g)).chain([(1, 1), (1, 2)]).collect();
        let busy = HashSet::from([(0, 6)]);
        assert_eq!(pending(&keys, &st, &busy), vec![(0, 2), (0, 4), (1, 2)]);
        let mut recs2 = recs.clone();
        recs2.push(rec(hostd_id(0, 2, 1), "Stopped", None));
        assert!(ledger_state(&recs2, Some(2)).gave_up.is_empty(), "a user's forced stop is not a failed attempt");
        recs2.push(rec(hostd_id(0, 2, 2), "Aborted", Some("hostd connection lost: io")));
        let st = ledger_state(&recs2, Some(2));
        assert_eq!(st.gave_up, vec![(0, 2)], "two failed attempts of game 2 reach max_attempts 2");
        assert_eq!(st.done[&(0, 2)], hostd_id(0, 2, 2));
        assert!(!st.done.contains_key(&(0, 4)), "one failure of game 4 is below the limit");
    }

    fn job_dir(tag: &str, games: u32) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("uranium-eval-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("plans")).unwrap();
        for (i, name) in ["a", "b"].iter().enumerate() {
            std::fs::write(dir.join(format!("plans/{i}.toml")), format!("name='{name}'\nsave_replay=true\n[[matchup]]\nmap='both'\ngames={games}\np0={{kind='t0',race='terran'}}\np1={{kind='t0',race='zerg'}}\n")).unwrap();
        }
        std::fs::write(dir.join("job.toml"), "stages=['plans/0.toml','plans/1.toml']\nhosts='hosts.toml'\n").unwrap();
        dir
    }

    fn all_stages() -> HashSet<usize> {
        HashSet::from([0, 1])
    }

    /// 020-R15: a source hands out every game once in play order, retries a stopped one with the
    /// next attempt, writes replays under the stage, and ends when nothing is left.
    #[test]
    fn ledger_source_hands_out_in_order_020_r15() {
        let dir = job_dir("src", 2);
        let job = Arc::new(EvalJob::load(&dir).unwrap());
        assert_eq!(job.lease_id, format!("eval-{}", dir.file_name().unwrap().to_string_lossy()));
        let src = LedgerSource::new(job.clone(), "h", "127.0.0.1:1", all_stages());
        assert!(!src.complete(), "not complete before the ledger was read");
        src.refresh(&LedgerState::default(), no_flights());
        let a = src.next().unwrap();
        assert_eq!(a.spec.game_id, hostd_id(0, 1, 0));
        assert_eq!(a.spec.seed, Some(1001), "the plan seed of game 1");
        assert_eq!(a.spec.players[1].name.as_deref(), Some("T0-zerg"));
        let b = src.next().unwrap();
        src.done(&a, JobOutcome::Finished, Some(b"rep".to_vec()));
        assert_eq!(std::fs::read(dir.join("a/replays/g1.rep")).unwrap(), b"rep");
        src.done(&b, JobOutcome::Stopped, None);
        let b2 = src.next().unwrap();
        assert_eq!(b2.spec.game_id, hostd_id(0, 2, 1), "second attempt of a stopped game");
        let c = src.next().unwrap();
        assert_eq!(key_of(c.spec.game_id), (1, 1));
        let inflight: Inflight = serde_json::from_slice(&std::fs::read(dir.join("inflight/h.json")).unwrap()).unwrap();
        assert_eq!(inflight.games, vec![hostd_id(1, 1, 0), hostd_id(0, 2, 1)]);
        src.requeue(c);
        let c2 = src.next().unwrap();
        assert_eq!(c2.spec.game_id, hostd_id(1, 1, 0), "a game that never started keeps its attempt");
        let d = src.next().unwrap();
        for j in [&b2, &c2, &d] {
            src.done(j, JobOutcome::Finished, None);
        }
        // A refresh that read the ledger before these records must not hand them out again.
        src.refresh(&LedgerState::default(), no_flights());
        assert!(src.next().is_none());
        assert!(src.complete());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// 020-R15 ④: a refresh drops games in flight on other live workers, picks up their retries
    /// and attempts; a paused source waits, a closed one hands out nothing; stages without
    /// loaded models are left for the next period.
    #[test]
    fn refresh_pause_close_and_unloaded_stages_020_r15() {
        let dir = job_dir("refresh", 3);
        let job = Arc::new(EvalJob::load(&dir).unwrap());
        let src = Arc::new(LedgerSource::new(job.clone(), "h", "x", HashSet::from([0])));
        let recs = vec![rec(hostd_id(0, 1, 0), "Finished", None), rec(hostd_id(0, 2, 0), "Stopped", None)];
        let fl = Flights { busy: HashSet::from([(0, 3)]), attempts: HashMap::from([((0, 3), 1)]) };
        src.refresh(&ledger_state(&recs, None), fl);
        assert_eq!(src.unloaded(), 3, "stage b has no models loaded here");
        src.pause(true);
        let s2 = src.clone();
        let waiter = std::thread::spawn(move || s2.next());
        std::thread::sleep(Duration::from_millis(300));
        assert!(!waiter.is_finished(), "a paused source hands out nothing");
        src.pause(false);
        let a = waiter.join().unwrap().unwrap();
        assert_eq!(a.spec.game_id, hostd_id(0, 2, 1));
        src.close();
        assert!(src.next().is_none());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// 020-R15 ④ ⑦: another worker's live in-flight games are busy; once its file stays unchanged
    /// for `stale_after` its games are free again, and every id in any file (this host's left by
    /// an earlier process too) moves the next attempt past it.
    #[test]
    fn flights_expire_dead_workers_and_bump_attempts_020_r15() {
        let dir = job_dir("flights", 2);
        let job = EvalJob::load(&dir).unwrap();
        std::fs::create_dir_all(dir.join("inflight")).unwrap();
        let write = |host: &str, ids: Vec<u64>| {
            let f = Inflight { host: host.into(), pid: 1, time: 5, games: ids };
            std::fs::write(dir.join(format!("inflight/{host}.json")), serde_json::to_vec(&f).unwrap()).unwrap();
        };
        write("dead", vec![hostd_id(0, 1, 0)]);
        write("me", vec![hostd_id(0, 2, 3)]);
        // This process's own live file does not move its attempts.
        let f = Inflight { host: "me".into(), pid: std::process::id(), time: 5, games: vec![hostd_id(1, 1, 5)] };
        std::fs::write(dir.join("inflight/me-live.json"), serde_json::to_vec(&f).unwrap()).unwrap();
        let mut obs = ContentObserver::default();
        // Watched in steps shorter than half the limit, as 061-R5 expiry requires.
        let stale = Duration::from_millis(400);
        let f = flights(&job, "me", &mut obs, stale).unwrap();
        assert_eq!(f.busy, HashSet::from([(0, 1)]), "this host's own old file is not busy");
        assert_eq!((f.attempts[&(0, 1)], f.attempts[&(0, 2)]), (1, 4));
        assert!(!f.attempts.contains_key(&(1, 1)));
        std::thread::sleep(Duration::from_millis(150));
        assert!(flights(&job, "me", &mut obs, stale).unwrap().busy.contains(&(0, 1)));
        std::thread::sleep(Duration::from_millis(150));
        flights(&job, "me", &mut obs, stale).unwrap();
        std::thread::sleep(Duration::from_millis(150));
        let f = flights(&job, "me", &mut obs, stale).unwrap();
        assert!(f.busy.is_empty(), "unchanged for stale_after: the worker is gone");
        assert_eq!(f.attempts[&(0, 1)], 1, "its attempt is not reused");
        std::fs::write(dir.join("inflight/bad.json"), b"{").unwrap();
        assert!(flights(&job, "me", &mut obs, stale).is_err(), "an unreadable file is unknown, not free");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// 020-R15 ⑦: records naming another stage than job.toml has at their index are refused; an
    /// unreadable games folder is an error, a missing one is empty.
    #[test]
    fn records_check_the_stage_list_020_r15() {
        let dir = job_dir("stages", 1);
        let job = EvalJob::load(&dir).unwrap();
        assert!(read_records(&job).unwrap().is_empty());
        std::fs::create_dir_all(dir.join("games")).unwrap();
        std::fs::write(dir.join("games/h.jsonl"), format!("{{\"game_id\":{},\"outcome\":\"Finished\",\"stage\":\"a\"}}\n", hostd_id(0, 1, 0))).unwrap();
        assert_eq!(read_records(&job).unwrap().len(), 1);
        std::fs::write(dir.join("games/h.jsonl"), format!("{{\"game_id\":{},\"outcome\":\"Finished\",\"stage\":\"a\"}}\n", hostd_id(1, 1, 0))).unwrap();
        assert!(read_records(&job).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Two stages sharing a model must agree on its session settings.
    #[test]
    fn shared_models_need_the_same_session_settings() {
        let dir = job_dir("settings", 1);
        let m = "[[matchup]]\nmap='both'\ngames=1\np0={kind='model',checkpoint='c/r/ckpt-1'}\np1={kind='t0',race='zerg'}\n";
        std::fs::write(dir.join("plans/0.toml"), format!("name='a'\nzero_last_failed=true\n{m}")).unwrap();
        std::fs::write(dir.join("plans/1.toml"), format!("name='b'\n{m}")).unwrap();
        assert!(EvalJob::load(&dir).is_err());
        std::fs::write(dir.join("plans/1.toml"), format!("name='b'\nzero_last_failed=true\n{m}")).unwrap();
        assert!(EvalJob::load(&dir).is_ok());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// 061-R16: the lease is the R5 lock without a checkpoint folder; a take-over makes the old
    /// holder's heartbeat Lost, release removes only one's own lock.
    #[test]
    fn lease_acquire_heartbeat_release_061_r16() {
        let shared = std::env::temp_dir().join(format!("uranium-lease-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&shared);
        let cfg = |host: &str| StorageConfig::new(&shared, shared.join("cache"), host);
        let mut a = Lease::acquire(cfg("a"), "eval-x", TakeOver::No).unwrap();
        assert!(!shared.join("checkpoints").exists());
        assert!(Lease::acquire(cfg("b"), "eval-x", TakeOver::No).is_err(), "a fresh lease of another host is refused");
        assert_eq!(a.heartbeat(), LockStatus::Held);
        let b = Lease::acquire(cfg("b"), "eval-x", TakeOver::Force).unwrap();
        assert!(matches!(a.heartbeat(), LockStatus::Lost(l) if l.host == "b"));
        a.release().unwrap();
        assert!(shared.join("locks/eval-x.json").exists(), "a's release leaves b's lock");
        b.release().unwrap();
        assert!(!shared.join("locks/eval-x.json").exists());
        std::fs::remove_dir_all(&shared).ok();
    }

    /// 061-R16: a waiting worker with a higher priority asks the holder to yield (its status is
    /// the eval worker file, not the training worker's), and takes the lease once it is free.
    #[test]
    fn higher_priority_worker_gets_the_lease_061_r16() {
        let dir = job_dir("handover", 2);
        let shared = dir.join("share");
        let worker = |host: &str, priority: i32| {
            let mut w = EvalWorker::start(&dir, StorageConfig::new(&shared, shared.join("cache"), host), priority, "cpu", None, Arc::new(AtomicUsize::new(0)), "127.0.0.1:0").unwrap();
            w.tick = Duration::from_millis(50);
            Arc::new(w)
        };
        let lo = worker("lo", 1);
        let hi = worker("hi", 10);
        assert_eq!(lo.wait_lease().unwrap(), WaitEnd::Lease);
        assert_eq!(read_status(&lo.cfg, "lo").and_then(|s| s.run), Some(lo.job.lease_id.clone()));
        // A training worker's file on the same host does not change the eval priority.
        storage::write_worker(&lo.cfg, 50, "cpu", None).unwrap();
        let h = hi.clone();
        let t = std::thread::spawn(move || h.wait_lease().unwrap());
        let until = Instant::now() + Duration::from_secs(10);
        while storage::yield_by_other(&lo.cfg, &lo.job.lease_id).is_none() {
            assert!(Instant::now() < until, "no yield request");
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(!t.is_finished(), "the lease is still held");
        lo.release("test yield");
        assert_eq!(t.join().unwrap(), WaitEnd::Lease);
        assert!(storage::read_yield(&hi.cfg, &hi.job.lease_id).unwrap().is_none(), "the taker removes its request");
        lo.close();
        hi.close();
        std::fs::remove_dir_all(&dir).ok();
    }
}
