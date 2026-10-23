//! 061-R13/R14 run queue worker: `uranium-train worker --queue <file> --backend <b> --priority <n>`.
//!
//! Takes the first run of the queue that is not done, whose `init_run` is done, and that can be
//! taken (no lock, this host's own lock, or a lock this worker watched expire, 061-R5), trains it
//! as a `bc --run` child under the 060-R6 watchdog, then moves on. With nothing to take, it asks a
//! slower PC's run to yield (higher `--priority` only) and waits. Shared drive trouble is
//! "unknown": nothing new is taken, the worker keeps running and looks again (061-R13, H-2).
//! Ctrl+C: the child checkpoints, pushes and releases, then the worker ends.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde::Deserialize;

use crate::storage::{self, LockObserver, LockView, StorageConfig, TakeOver, Watched, YieldObserver};
use crate::supervise::{Decision, Ending, SuperviseArgs, Watch, decide};

/// The queue file (TOML). `datasets` and `config` in `[defaults]` apply to runs that omit them.
#[derive(Deserialize, Debug, Clone, Default)]
pub struct Queue {
    #[serde(default)]
    pub defaults: Defaults,
    #[serde(default)]
    pub run: Vec<QueueRun>,
}

#[derive(Deserialize, Debug, Clone, Default)]
pub struct Defaults {
    #[serde(default)]
    pub datasets: Vec<String>,
    pub config: Option<String>,
}

/// One run of the queue: the arguments of `bc --run <id>`.
#[derive(Deserialize, Debug, Clone, PartialEq)]
pub struct QueueRun {
    pub id: String,
    pub config: Option<String>,
    /// Default: the adopted core (032-R17, 2026-10-23).
    #[serde(default = "gtrxl")]
    pub core: String,
    /// `s1`, `s2` (fine-tuning, needs `init_run`) or `ablation-a`.
    #[serde(default = "s1")]
    pub stage: String,
    /// Run whose final checkpoint gives the starting weights (S2 from S1, 041-R4). The run is
    /// taken only once this one has `DONE` (061-R13, H-1).
    pub init_run: Option<String>,
    pub datasets: Option<Vec<String>>,
    /// Stop (and mark done) after this update (test runs).
    pub steps: Option<usize>,
}

fn gtrxl() -> String {
    "gtrxl".into()
}

fn s1() -> String {
    "s1".into()
}

impl Queue {
    pub fn load(path: &Path) -> Result<Self> {
        let q: Queue = toml::from_str(&std::fs::read_to_string(path).with_context(|| path.display().to_string())?)?;
        for r in &q.run {
            storage::check_name(&r.id)?;
        }
        Ok(q)
    }
}

/// The `bc` arguments of a queue run.
pub fn bc_args(r: &QueueRun, d: &Defaults, backend: &str, shared: &Path, take: &TakeOver) -> Vec<String> {
    let mut a: Vec<String> = vec!["bc".into(), "--run".into(), r.id.clone(), "--backend".into(), backend.into(), "--core".into(), r.core.clone()];
    if let Some(c) = r.config.as_ref().or(d.config.as_ref()) {
        a.extend(["--config".into(), c.clone()]);
    }
    match r.stage.as_str() {
        "s2" => a.push("--s2".into()),
        "ablation-a" => a.extend(["--stage".into(), "s2".into()]),
        _ => {}
    }
    if let Some(i) = &r.init_run {
        a.extend(["--init".into(), shared.join("checkpoints").join(i).display().to_string()]);
    }
    for ds in r.datasets.as_ref().unwrap_or(&d.datasets) {
        a.extend(["--dataset".into(), ds.clone()]);
    }
    if let Some(s) = r.steps {
        a.extend(["--steps".into(), s.to_string()]);
    }
    match take {
        TakeOver::No => {}
        TakeOver::Force => a.push("--take-over".into()),
        TakeOver::IfUnchanged(l) => a.extend(["--take-over-if".into(), serde_json::to_string(l).expect("lock info serializes")]),
    }
    a
}

/// Trains one queue run to its end (the real launcher spawns `bc`; tests fake it).
pub trait Launcher {
    fn launch(&self, run: &QueueRun, defaults: &Defaults, take: &TakeOver) -> Result<Ending>;
}

/// A run to train and how its lock is taken.
#[derive(Debug, Clone, PartialEq)]
pub struct Pick<'a> {
    pub run: &'a QueueRun,
    /// [`TakeOver::IfUnchanged`] for a lock of another PC this worker watched expire (061-R5).
    pub take: TakeOver,
}

/// Runs skipped after too many restarts are left alone this long.
const GIVE_UP_SKIP: Duration = Duration::from_secs(3600);

/// What a worker watches between polls on its own monotonic clock: run locks (061-R5 expiry)
/// and yield requests (061-R14 expiry).
#[derive(Debug, Default)]
pub struct Observers {
    pub locks: LockObserver,
    pub yields: YieldObserver,
}

impl Observers {
    pub fn reset(&mut self) {
        self.locks.reset();
        self.yields.reset();
    }
}

/// The result of one look at the queue.
#[derive(Debug, Default)]
pub struct Picked<'a> {
    pub pick: Option<Pick<'a>>,
    /// Runs skipped this time because their state could not be read (061-R13 "unknown", per run).
    pub unknown: Vec<(String, String)>,
    /// Yield requests removed as expired (061-R14), `(run, requesting host)`.
    pub expired: Vec<(String, String)>,
}

/// The next run this host can take (061-R13). `Err` = the share itself is unreachable: take
/// nothing (H-2). A run whose `DONE`, lock or yield request cannot be read is "unknown": it is
/// skipped this time and the other runs stay pickable (2026-10-19 decision). `now` is this
/// process's monotonic clock, the only clock lock and request expiry use (061-R5, R14).
pub fn pick<'a>(cfg: &StorageConfig, q: &'a Queue, skip: &HashMap<String, Instant>, obs: &mut Observers, now: Instant) -> Result<Picked<'a>> {
    storage::probe_share(cfg)?;
    let mut out = Picked::default();
    for r in &q.run {
        match look(cfg, r, skip, obs, now, &mut out.expired) {
            Ok(Some(p)) => {
                out.pick = Some(p);
                break;
            }
            Ok(None) => {}
            Err(e) => out.unknown.push((r.id.clone(), format!("{e:#}"))),
        }
    }
    Ok(out)
}

fn look<'a>(cfg: &StorageConfig, r: &'a QueueRun, skip: &HashMap<String, Instant>, obs: &mut Observers, now: Instant, expired: &mut Vec<(String, String)>) -> Result<Option<Pick<'a>>> {
    if storage::is_done(cfg, &r.id)? || skip.get(&r.id).is_some_and(|t| t.elapsed() < GIVE_UP_SKIP) {
        return Ok(None);
    }
    // H-1: S2 starts from the finished S1 only.
    if let Some(i) = &r.init_run
        && !storage::is_done(cfg, i)?
    {
        return Ok(None);
    }
    // 061-R14: a request reserves the run for the host that asked, until a worker has seen it
    // unchanged for 30 min; then it is removed as expired.
    match obs.yields.watch(&r.id, &storage::read_yield(cfg, &r.id), now, storage::YIELD_EXPIRY) {
        Watched::Absent => {}
        Watched::Fresh(y) if y.host != cfg.host => return Ok(None),
        Watched::Fresh(_) => {}
        Watched::Expired(y) => {
            if storage::remove_yield_if(cfg, &r.id, &y)? {
                expired.push((r.id.clone(), y.host.clone()));
            }
            if y.host != cfg.host {
                return Ok(None); // looked at again next poll, without the request
            }
        }
        Watched::Unknown(e) => anyhow::bail!("yield request: {e}"),
    }
    Ok(match obs.locks.observe(&r.id, &storage::peek_lock(cfg, &r.id), now, cfg.stale_after) {
        LockView::Free => Some(Pick { run: r, take: TakeOver::No }),
        LockView::Held(l) | LockView::Stale(l) if l.host == cfg.host => Some(Pick { run: r, take: TakeOver::No }),
        LockView::Stale(l) => Some(Pick { run: r, take: TakeOver::IfUnchanged(l) }),
        LockView::Held(_) => None,
        LockView::Unknown(e) => anyhow::bail!("lock: {e}"),
    })
}

/// A run held by a PC whose worker has a lower priority than `priority`, to ask to yield
/// (061-R14). Only runs a worker trains are targets: the holder's `workers/<host>.json` must name
/// that run. A run started by hand (`bc --run`, no worker state) is never asked (user decision
/// 2026-10-19). None when a request of this host is already pending. Runs whose state cannot be
/// read are skipped (per run), and so are runs this worker gave up on for now (`skip`): it would
/// not take them either. `Err` = the share is unreachable.
pub fn preempt_target<'a>(cfg: &StorageConfig, q: &'a Queue, priority: i32, skip: &HashMap<String, Instant>) -> Result<Option<&'a QueueRun>> {
    storage::probe_share(cfg)?;
    for r in &q.run {
        if skip.get(&r.id).is_some_and(|t| t.elapsed() < GIVE_UP_SKIP) {
            continue;
        }
        let target = (|| -> Result<Option<bool>> {
            if storage::is_done(cfg, &r.id)? {
                return Ok(None);
            }
            if let Some(y) = storage::read_yield(cfg, &r.id)? {
                return Ok(if y.host == cfg.host { Some(false) } else { None });
            }
            let Some(l) = storage::peek_lock(cfg, &r.id)? else { return Ok(None) };
            if l.host == cfg.host {
                return Ok(None);
            }
            let Some(w) = storage::read_worker(cfg, &l.host) else { return Ok(None) };
            Ok((w.run.as_deref() == Some(r.id.as_str()) && priority > w.priority).then_some(true))
        })();
        match target {
            Ok(Some(true)) => return Ok(Some(r)),
            Ok(Some(false)) => return Ok(None), // this host's request is pending
            Ok(None) | Err(_) => {}
        }
    }
    Ok(None)
}

pub struct WorkerArgs {
    pub queue: PathBuf,
    pub backend: String,
    pub priority: i32,
    pub preempt: bool,
    /// How often to look for a run when there is none (and to retry after share trouble).
    pub poll: Duration,
    /// How often `workers/<host>.json` is rewritten, also while a child trains (061-R13: 1 min).
    pub status_every: Duration,
    pub supervise: SuperviseArgs,
}

/// Notes a message once until `reset` (repeated failures every poll would flood the log).
struct Once(Option<String>);

impl Once {
    fn note(&mut self, note: &dyn Fn(&str), key: &str, msg: &str) {
        if self.0.as_deref() != Some(key) {
            note(msg);
            self.0 = Some(key.to_string());
        }
    }

    fn reset(&mut self) {
        self.0 = None;
    }
}

/// The worker loop over any launcher; returns when Ctrl+C was pressed (`stop()` true). Shared
/// drive errors never end it (061-R13, H-2).
pub fn work<L: Launcher>(cfg: &StorageConfig, a: &WorkerArgs, launcher: &L, stop: &dyn Fn() -> bool, note: &dyn Fn(&str), wait: &dyn Fn(Duration) -> bool) -> Result<()> {
    // 061-R5: expiry watching restarts after a gap longer than stale_after / 2, so a slower poll
    // would never see a lock expire.
    anyhow::ensure!(
        a.poll < cfg.stale_after / 2,
        "--poll-seconds ({} s) must be shorter than half the lock expiry ({} s) (061-R5)",
        a.poll.as_secs_f64(),
        (cfg.stale_after / 2).as_secs()
    );
    // `checkpoints/` marks a reachable share (storage::probe_share); never the root itself.
    let _ = std::fs::create_dir(cfg.shared.join("checkpoints"));
    // The current run; the mutex also serialises every status write (061-R13).
    let current: Mutex<Option<String>> = Mutex::new(None);
    let finished = AtomicBool::new(false);
    let write_status = |run: Option<&str>| {
        let mut cur = current.lock().unwrap();
        *cur = run.map(str::to_string);
        let _ = storage::write_worker(cfg, a.priority, &a.backend, cur.as_deref());
    };
    std::thread::scope(|s| {
        // 061-R13: the status file every `status_every`, while a child trains as well.
        s.spawn(|| {
            let mut last: Option<Instant> = None;
            while !finished.load(Ordering::SeqCst) {
                if last.is_none_or(|t| t.elapsed() >= a.status_every) {
                    let cur = current.lock().unwrap();
                    let _ = storage::write_worker(cfg, a.priority, &a.backend, cur.as_deref());
                    drop(cur);
                    last = Some(Instant::now());
                }
                std::thread::sleep(Duration::from_millis(50).min(a.status_every));
            }
        });
        /// Ends the status thread however the loop ends (a panic included), so the scope joins.
        struct Finish<'a>(&'a AtomicBool);
        impl Drop for Finish<'_> {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }
        let _finish = Finish(&finished);
        let res = work_loop(cfg, a, launcher, stop, note, wait, &write_status);
        write_status(None);
        res
    })
}

fn work_loop<L: Launcher>(
    cfg: &StorageConfig,
    a: &WorkerArgs,
    launcher: &L,
    stop: &dyn Fn() -> bool,
    note: &dyn Fn(&str),
    wait: &dyn Fn(Duration) -> bool,
    set_current: &dyn Fn(Option<&str>),
) -> Result<()> {
    let mut skip: HashMap<String, Instant> = HashMap::new();
    let mut obs = Observers::default();
    // 061-R14 ①: the run this host asked to yield, cleared when it takes another run or stops.
    let mut my_request: Option<String> = None;
    let mut said = Once(None);
    let mut said_unknown = Once(None);
    let drop_request = |my_request: &mut Option<String>, keep: Option<&str>| {
        if let Some(id) = my_request.take() {
            if Some(id.as_str()) == keep {
                return; // the child taking it clears it (061-R14)
            }
            match storage::clear_yield(cfg, &id) {
                Ok(()) => note(&format!("withdrew the yield request on {id} (061-R14)")),
                Err(e) => {
                    note(&format!("could not withdraw the yield request on {id} ({e:#}); it expires after 30 min (061-R14)"));
                }
            }
        }
    };
    while !stop() {
        let q = match Queue::load(&a.queue) {
            Ok(q) => q,
            Err(e) => {
                said.note(note, "queue", &format!("queue unreadable ({e:#}); retrying every {} s (061-R13)", a.poll.as_secs()));
                if !wait(a.poll) {
                    break;
                }
                continue;
            }
        };
        set_current(None);
        let picked = match pick(cfg, &q, &skip, &mut obs, Instant::now()) {
            Ok(p) => {
                for (id, host) in &p.expired {
                    note(&format!("removed the yield request of {host} on {id}: unchanged for 30 min (061-R14)"));
                }
                if !p.unknown.is_empty() {
                    let ids: Vec<&str> = p.unknown.iter().map(|(id, _)| id.as_str()).collect();
                    let detail: Vec<String> = p.unknown.iter().map(|(id, e)| format!("{id}: {e}")).collect();
                    said_unknown.note(note, &format!("unknown {ids:?}"), &format!("state unknown, skipped this time (061-R13): {}", detail.join("; ")));
                } else {
                    said_unknown.reset();
                }
                p.pick
            }
            Err(e) => {
                said.note(note, "unknown", &format!("shared drive state unknown ({e:#}); taking nothing, retrying every {} s (061-R13)", a.poll.as_secs()));
                if !wait(a.poll) {
                    break;
                }
                continue;
            }
        };
        let Some(Pick { run: r, take }) = picked else {
            let target = if a.preempt { preempt_target(cfg, &q, a.priority, &skip) } else { Ok(None) };
            match target {
                Ok(Some(t)) => {
                    // An earlier request of this host (expired by now) is withdrawn first.
                    drop_request(&mut my_request, Some(&t.id));
                    match storage::request_yield(cfg, &t.id, a.priority) {
                        Ok(()) => {
                            my_request = Some(t.id.clone());
                            said.reset();
                            note(&format!("asked the holder of {} to yield (priority {}, 061-R14)", t.id, a.priority));
                        }
                        Err(e) => said.note(note, "yield", &format!("could not write a yield request on {} ({e:#}); retrying (061-R13)", t.id)),
                    }
                }
                Ok(None) => said.note(note, "idle", "no run to take; waiting"),
                Err(e) => said.note(note, "unknown", &format!("shared drive state unknown ({e:#}); retrying every {} s (061-R13)", a.poll.as_secs())),
            }
            if !wait(a.poll) {
                break;
            }
            continue;
        };
        said.reset();
        // 061-R14: the child that takes the run removes this host's request on it; if it ends
        // without taking it (e.g. 75), the request is removed below, never left as a 30-min
        // reservation.
        let asked_for_it = my_request.as_deref() == Some(r.id.as_str());
        drop_request(&mut my_request, Some(&r.id));
        set_current(Some(&r.id));
        // Not watching other locks while training: observations restart afterwards (061-R5).
        obs.reset();
        let mut restarts: Vec<Instant> = Vec::new();
        let mut take = take;
        loop {
            let ending = launcher.launch(r, &q.defaults, &take).unwrap_or_else(|e| {
                note(&format!("{}: could not run the training ({e:#})", r.id));
                Ending::Exited(Some(1))
            });
            take = TakeOver::No;
            restarts.retain(|t| t.elapsed() < a.supervise.window);
            match ending {
                Ending::Exited(Some(0)) => break,
                Ending::Exited(Some(75)) => {
                    note(&format!("{} is held by another PC; looking again", r.id));
                    break;
                }
                Ending::Exited(Some(76)) => {
                    note(&format!("{} yielded to another PC", r.id));
                    break;
                }
                _ => {}
            }
            // H-2: a child that failed while the share is gone is not a crash of the run (it
            // could not even start); wait for the share instead of burning restarts and giving up.
            if !stop()
                && let Err(e) = storage::probe_share(cfg)
            {
                note(&format!("{}: {ending:?} with the shared drive unreachable ({e:#}); not counted as a restart, retrying when it is back (061-R13)", r.id));
                let _ = wait(a.poll);
                break;
            }
            match decide(ending, stop(), restarts.len(), &a.supervise) {
                Decision::Stop => break,
                Decision::GiveUp => {
                    note(&format!("{}: {ending:?}, {} restarts within {} min; skipping it for an hour", r.id, restarts.len(), a.supervise.window.as_secs() / 60));
                    skip.insert(r.id.clone(), Instant::now());
                    break;
                }
                Decision::Restart(d) => {
                    note(&format!("{}: {ending:?}; restarting in {} s (060-R6)", r.id, d.as_secs()));
                    if !wait(d) {
                        break;
                    }
                    restarts.push(Instant::now());
                }
            }
        }
        if asked_for_it && let Err(e) = storage::clear_yield(cfg, &r.id) {
            note(&format!("could not remove this host's yield request on {} ({e:#}); workers drop it after 30 min (061-R14)", r.id));
        }
    }
    drop_request(&mut my_request, None);
    note("stopped");
    Ok(())
}

/// The launcher of the real worker: `bc` as a watched child process.
struct BcLauncher<'a> {
    watch: &'a Watch,
    backend: String,
    shared: PathBuf,
    hang: Duration,
}

impl Launcher for BcLauncher<'_> {
    fn launch(&self, r: &QueueRun, d: &Defaults, take: &TakeOver) -> Result<Ending> {
        self.watch.run_child(&bc_args(r, d, &self.backend, &self.shared, take), self.hang, None)
    }
}

pub fn run(a: WorkerArgs) -> Result<()> {
    let cfg = StorageConfig::from_env()?;
    let watch = Watch::new("worker", a.supervise.log.as_ref())?;
    watch.note(&format!("host {} backend {} priority {} queue {}", cfg.host, a.backend, a.priority, a.queue.display()));
    let launcher = BcLauncher { watch: &watch, backend: a.backend.clone(), shared: cfg.shared.clone(), hang: a.supervise.hang };
    work(&cfg, &a, &launcher, &|| watch.stopping(), &|m| watch.note(m), &|d| watch.wait(d))
}
