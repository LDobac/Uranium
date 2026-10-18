//! 061-R13/R14 run queue worker: `uranium-train worker --queue <file> --backend <b> --priority <n>`.
//!
//! Takes the first run of the queue that is not done and can be taken (no lock, this host's own
//! lock, or a stale lock), trains it as a `bc --run` child under the 060-R6 watchdog, then moves
//! on. With nothing to take, it asks a slower PC's run to yield (higher `--priority` only) and
//! waits. Ctrl+C: the child checkpoints, pushes and releases, then the worker ends.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde::Deserialize;

use crate::storage::{self, LockView, StorageConfig};
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
    #[serde(default = "lstm")]
    pub core: String,
    /// `s1`, `s2` (fine-tuning, needs `init_run`) or `ablation-a`.
    #[serde(default = "s1")]
    pub stage: String,
    /// Run whose `LATEST` gives the starting weights (S2 from S1, 041-R4).
    pub init_run: Option<String>,
    pub datasets: Option<Vec<String>>,
    /// Stop (and mark done) after this update (test runs).
    pub steps: Option<usize>,
}

fn lstm() -> String {
    "lstm".into()
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
pub fn bc_args(r: &QueueRun, d: &Defaults, backend: &str, shared: &Path, take_over: bool) -> Vec<String> {
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
    if take_over {
        a.push("--take-over".into());
    }
    a
}

/// Trains one queue run to its end (the real launcher spawns `bc`; tests fake it).
pub trait Launcher {
    fn launch(&self, run: &QueueRun, defaults: &Defaults, take_over: bool) -> Result<Ending>;
}

/// The next run this host can take, and whether it takes over a stale lock (061-R13).
pub fn pick<'a>(cfg: &StorageConfig, q: &'a Queue, skip: &HashMap<String, Instant>) -> Result<Option<(&'a QueueRun, bool)>> {
    for r in &q.run {
        if storage::is_done(cfg, &r.id) || skip.get(&r.id).is_some_and(|t| t.elapsed() < Duration::from_secs(3600)) {
            continue;
        }
        if storage::read_yield(cfg, &r.id).is_some_and(|y| y.host != cfg.host) {
            continue; // reserved for the host that asked (061-R14)
        }
        match storage::peek_lock(cfg, &r.id)? {
            LockView::Free => return Ok(Some((r, false))),
            LockView::Held(l) if l.host == cfg.host => return Ok(Some((r, false))),
            LockView::Stale(_) => return Ok(Some((r, true))),
            LockView::Held(_) => {}
        }
    }
    Ok(None)
}

/// A run held by a PC whose worker has a lower priority than `priority`, to ask to yield
/// (061-R14). None when a request of this host is already pending.
pub fn preempt_target<'a>(cfg: &StorageConfig, q: &'a Queue, priority: i32) -> Result<Option<&'a QueueRun>> {
    for r in &q.run {
        if storage::is_done(cfg, &r.id) {
            continue;
        }
        if let Some(y) = storage::read_yield(cfg, &r.id) {
            if y.host == cfg.host {
                return Ok(None);
            }
            continue;
        }
        if let LockView::Held(l) = storage::peek_lock(cfg, &r.id)?
            && l.host != cfg.host
        {
            let theirs = storage::read_worker(cfg, &l.host).map_or(0, |w| w.priority);
            if priority > theirs {
                return Ok(Some(r));
            }
        }
    }
    Ok(None)
}

pub struct WorkerArgs {
    pub queue: PathBuf,
    pub backend: String,
    pub priority: i32,
    pub preempt: bool,
    /// How often to look for a run when there is none.
    pub poll: Duration,
    pub supervise: SuperviseArgs,
}

/// The worker loop over any launcher; returns when Ctrl+C was pressed (`stop()` true).
pub fn work<L: Launcher>(cfg: &StorageConfig, a: &WorkerArgs, launcher: &L, stop: &dyn Fn() -> bool, note: &dyn Fn(&str), wait: &dyn Fn(Duration) -> bool) -> Result<()> {
    let mut skip: HashMap<String, Instant> = HashMap::new();
    let mut idle_noted = false;
    while !stop() {
        let q = Queue::load(&a.queue)?;
        let _ = storage::write_worker(cfg, a.priority, &a.backend, None);
        let Some((r, take_over)) = pick(cfg, &q, &skip)? else {
            if a.preempt
                && let Some(t) = preempt_target(cfg, &q, a.priority)?
            {
                storage::request_yield(cfg, &t.id, a.priority)?;
                note(&format!("asked the holder of {} to yield (priority {}, 061-R14)", t.id, a.priority));
            } else if !idle_noted {
                note("no run to take; waiting");
                idle_noted = true;
            }
            if !wait(a.poll) {
                break;
            }
            continue;
        };
        idle_noted = false;
        let _ = storage::write_worker(cfg, a.priority, &a.backend, Some(&r.id));
        let mut restarts: Vec<Instant> = Vec::new();
        let mut take_over = take_over;
        loop {
            let ending = launcher.launch(r, &q.defaults, take_over)?;
            take_over = false;
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
    }
    let _ = storage::write_worker(cfg, a.priority, &a.backend, None);
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
    fn launch(&self, r: &QueueRun, d: &Defaults, take_over: bool) -> Result<Ending> {
        self.watch.run_child(&bc_args(r, d, &self.backend, &self.shared, take_over), self.hang, None)
    }
}

pub fn run(a: WorkerArgs) -> Result<()> {
    let cfg = StorageConfig::from_env()?;
    let watch = Watch::new("worker", a.supervise.log.as_ref())?;
    watch.note(&format!("host {} backend {} priority {} queue {}", cfg.host, a.backend, a.priority, a.queue.display()));
    let launcher = BcLauncher { watch: &watch, backend: a.backend.clone(), shared: cfg.shared.clone(), hang: a.supervise.hang };
    work(&cfg, &a, &launcher, &|| watch.stopping(), &|m| watch.note(m), &|d| watch.wait(d))
}
