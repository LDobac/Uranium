//! 061-AC8 / R13 / R14: run-queue workers (fake training, no GPU) on one shared folder — never two
//! owners of a run, two runs on two workers at once, yield requests hand a run to the higher
//! priority and are withdrawn, a Ctrl+C stop hands it back, `init_run` must be done (H-1), shared
//! drive trouble never ends a worker (H-2), automatic take-over only after watching an unchanged
//! heartbeat on the worker's own clock (H-3).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use anyhow::Result;
use uranium_train::storage::{self, LockBusy, LockInfo, LockObserver, LockView, Run, StorageConfig, TakeOver};
use uranium_train::supervise::{Ending, SuperviseArgs};
use uranium_train::worker::{Defaults, Launcher, Observers, Pick, Queue, QueueRun, WorkerArgs, pick, preempt_target, work};

/// Simulated `bc --run`: same lock / yield / done protocol, one "update" per tick, progress kept
/// on the share (as a checkpoint would be).
struct Fake {
    cfg: StorageConfig,
    target: u64,
    tick: Duration,
    owners: Arc<Mutex<HashMap<String, String>>>,
    events: Arc<Mutex<Vec<String>>>,
    stop: Arc<AtomicBool>,
}

impl Fake {
    fn event(&self, e: String) {
        self.events.lock().unwrap().push(e);
    }
}

impl Launcher for Fake {
    fn launch(&self, r: &QueueRun, _d: &Defaults, take: &TakeOver) -> Result<Ending> {
        let me = self.cfg.host.clone();
        let run = match Run::acquire_with(self.cfg.clone(), &r.id, take.clone()) {
            Ok(run) => run,
            Err(e) if e.downcast_ref::<LockBusy>().is_some() => return Ok(Ending::Exited(Some(75))),
            Err(e) => return Err(e),
        };
        match storage::read_yield(&self.cfg, &r.id)? {
            Some(y) if y.host != me => {
                run.release()?;
                return Ok(Ending::Exited(Some(75)));
            }
            Some(_) => storage::clear_yield(&self.cfg, &r.id)?,
            None => {}
        }
        if let Some(other) = self.owners.lock().unwrap().insert(r.id.clone(), me.clone()) {
            panic!("{} took {} while {other} owned it", me, r.id);
        }
        let progress = self.cfg.shared.join("checkpoints").join(&r.id).join("fake_step");
        let mut step: u64 = std::fs::read_to_string(&progress).ok().and_then(|s| s.trim().parse().ok()).unwrap_or(0);
        self.event(format!("{me} start {} at {step}", r.id));
        let mut run = run;
        let end = |run: Run, why: &str, code: i32, step: u64| -> Result<Ending> {
            self.owners.lock().unwrap().remove(&r.id);
            self.event(format!("{me} {why} {} at {step}", r.id));
            run.release()?;
            Ok(Ending::Exited(Some(code)))
        };
        loop {
            if step >= self.target {
                storage::mark_done(&self.cfg, &r.id, step)?;
                return end(run, "done", 0, step);
            }
            std::thread::sleep(self.tick);
            step += 1;
            std::fs::write(&progress, step.to_string())?;
            let _ = run.heartbeat();
            if self.stop.load(Ordering::SeqCst) {
                return end(run, "stopped", 0, step);
            }
            if storage::yield_by_other(&self.cfg, &r.id).is_some() {
                return end(run, "yielded", 76, step);
            }
        }
    }
}

fn args(queue: PathBuf, priority: i32) -> WorkerArgs {
    WorkerArgs {
        queue,
        backend: "flex".into(),
        priority,
        preempt: true,
        poll: Duration::from_millis(100),
        status_every: Duration::from_secs(60),
        supervise: SuperviseArgs { max_restarts: 3, window: Duration::from_secs(3600), delay: Duration::from_millis(10), hang: Duration::from_secs(600), progress_file: None, log: None, args: vec![] },
    }
}

fn wait_for(what: &str, t: Duration, f: impl Fn() -> bool) {
    let start = Instant::now();
    while !f() {
        assert!(start.elapsed() < t, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(20));
    }
}

fn tmp(tag: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!("uranium-worker-{tag}-{}", std::process::id()));
    std::fs::remove_dir_all(&root).ok();
    std::fs::create_dir_all(root.join("shared")).unwrap();
    root
}

fn local_cfg(root: &Path, host: &str) -> StorageConfig {
    let mut c = StorageConfig::new(root.join("shared"), root.join(format!("cache-{host}")), host);
    c.shared_local = true;
    c
}

/// Runs a worker on a thread; `stop` ends it like Ctrl+C. Notes are collected in `notes`.
fn spawn_worker<L: Launcher + Send + 'static>(cfg: StorageConfig, a: WorkerArgs, launcher: L, stop: Arc<AtomicBool>, notes: Arc<Mutex<Vec<String>>>) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let stopped = || stop.load(Ordering::SeqCst);
        let wait = |d: Duration| {
            std::thread::sleep(d);
            !stop.load(Ordering::SeqCst)
        };
        let note = |m: &str| notes.lock().unwrap().push(m.to_string());
        work(&cfg, &a, &launcher, &stopped, &note, &wait).unwrap();
    })
}

#[test]
fn two_workers_share_a_queue_061_ac8() {
    let root = tmp("ac8");
    let queue = root.join("queue.toml");
    std::fs::write(&queue, "[[run]]\nid = \"c\"\n").unwrap();
    let owners = Arc::new(Mutex::new(HashMap::new()));
    let events = Arc::new(Mutex::new(Vec::new()));
    let notes = Arc::new(Mutex::new(Vec::new()));
    let cfg = |host: &str| local_cfg(&root, host);
    let spawn = |host: &'static str, priority: i32, stop: Arc<AtomicBool>| {
        let fake = Fake { cfg: cfg(host), target: 60, tick: Duration::from_millis(40), owners: owners.clone(), events: events.clone(), stop: stop.clone() };
        spawn_worker(cfg(host), args(queue.clone(), priority), fake, stop, notes.clone())
    };
    let has = |e: &str| events.lock().unwrap().iter().any(|x| x.starts_with(e));

    // The slow PC takes the only run.
    let low_stop = Arc::new(AtomicBool::new(false));
    let low = spawn("low", 0, low_stop.clone());
    wait_for("low to start c", Duration::from_secs(10), || has("low start c"));
    // The fast PC finds nothing free, asks for the run, gets it at the next update.
    let high_stop = Arc::new(AtomicBool::new(false));
    let high = spawn("high", 10, high_stop.clone());
    wait_for("low to yield", Duration::from_secs(10), || has("low yielded c"));
    wait_for("high to take c", Duration::from_secs(10), || has("high start c"));
    // Ctrl+C on the fast PC: it stops cleanly, the slow PC takes the run back and finishes it.
    std::thread::sleep(Duration::from_millis(300));
    high_stop.store(true, Ordering::SeqCst);
    high.join().unwrap();
    wait_for("c done", Duration::from_secs(20), || storage::is_done(&cfg("low"), "c").unwrap());
    low_stop.store(true, Ordering::SeqCst);
    low.join().unwrap();

    let ev = events.lock().unwrap().clone();
    let names: Vec<String> = ev.iter().map(|e| e.split(" at ").next().unwrap().to_string()).collect();
    assert_eq!(names, ["low start c", "low yielded c", "high start c", "high stopped c", "low start c", "low done c"], "{ev:?}");
    // Progress only moves forward across owners (each start resumes where the last one stopped).
    let steps: Vec<u64> = ev.iter().map(|e| e.rsplit(' ').next().unwrap().parse().unwrap()).collect();
    assert!(steps.windows(2).all(|w| w[0] <= w[1]), "{ev:?}");
    assert_eq!(*steps.last().unwrap(), 60);
    assert!(storage::read_yield(&cfg("low"), "c").unwrap().is_none(), "request cleared by the host that asked");
    assert_eq!(storage::peek_lock(&cfg("low"), "c").unwrap(), None);
    std::fs::remove_dir_all(&root).ok();
}

/// 061-AC8 with two runs: two workers train different runs at the same time, each run has one
/// owner at any moment, and both finish.
#[test]
fn two_workers_two_runs_at_once_061_ac8() {
    let root = tmp("ac8-two");
    let queue = root.join("queue.toml");
    std::fs::write(&queue, "[[run]]\nid = \"a\"\n[[run]]\nid = \"b\"\n").unwrap();
    let owners = Arc::new(Mutex::new(HashMap::new()));
    let events = Arc::new(Mutex::new(Vec::new()));
    let notes = Arc::new(Mutex::new(Vec::new()));
    let both_seen = Arc::new(AtomicBool::new(false));
    let stops: Vec<Arc<AtomicBool>> = (0..2).map(|_| Arc::new(AtomicBool::new(false))).collect();
    let mut handles = Vec::new();
    for (host, stop) in ["pc1", "pc2"].into_iter().zip(&stops) {
        let fake = Fake { cfg: local_cfg(&root, host), target: 25, tick: Duration::from_millis(40), owners: owners.clone(), events: events.clone(), stop: stop.clone() };
        handles.push(spawn_worker(local_cfg(&root, host), args(queue.clone(), 1), fake, stop.clone(), notes.clone()));
    }
    let cfg = local_cfg(&root, "check");
    wait_for("both runs done", Duration::from_secs(30), || {
        if owners.lock().unwrap().len() == 2 {
            both_seen.store(true, Ordering::SeqCst);
        }
        storage::is_done(&cfg, "a").unwrap() && storage::is_done(&cfg, "b").unwrap()
    });
    for s in &stops {
        s.store(true, Ordering::SeqCst);
    }
    for h in handles {
        h.join().unwrap();
    }
    assert!(both_seen.load(Ordering::SeqCst), "the two runs trained at the same time: {:?}", events.lock().unwrap());
    let ev = events.lock().unwrap().clone();
    let starters = |id: &str| ev.iter().filter(|e| e.contains(&format!(" start {id} "))).map(|e| e.split(' ').next().unwrap().to_string()).collect::<Vec<_>>();
    let (sa, sb) = (starters("a"), starters("b"));
    assert_eq!((sa.len(), sb.len()), (1, 1), "{ev:?}");
    assert_ne!(sa[0], sb[0], "different workers: {ev:?}");
    std::fs::remove_dir_all(&root).ok();
}

fn queue(text: &str) -> Queue {
    toml::from_str(text).unwrap()
}

/// 061-R13 (H-1): a run whose `init_run` has no `DONE` is not taken, even when free.
#[test]
fn init_run_not_done_is_not_taken_061_r13() {
    let root = tmp("h1");
    let cfg = local_cfg(&root, "me");
    std::fs::create_dir_all(root.join("shared").join("checkpoints")).unwrap();
    let q = queue("[[run]]\nid = \"s1\"\n[[run]]\nid = \"s2\"\nstage = \"s2\"\ninit_run = \"s1\"\n");
    // S1 trains on another PC.
    let s1 = Run::acquire(local_cfg(&root, "other"), "s1", false).unwrap();
    let mut obs = Observers::default();
    let skip = HashMap::new();
    assert_eq!(pick(&cfg, &q, &skip, &mut obs, Instant::now()).unwrap().pick, None, "S2 waits for S1");
    // S1 given up (lock gone, no DONE): still not.
    s1.release().unwrap();
    assert_eq!(pick(&cfg, &q, &skip, &mut obs, Instant::now()).unwrap().pick.map(|p| p.run.id.clone()), Some("s1".into()));
    let mut skip_s1 = HashMap::new();
    skip_s1.insert("s1".to_string(), Instant::now());
    assert_eq!(pick(&cfg, &q, &skip_s1, &mut obs, Instant::now()).unwrap().pick, None, "S1 skipped, S2 still waits");
    // S1 done: S2 is next.
    storage::mark_done(&cfg, "s1", 10).unwrap();
    assert_eq!(pick(&cfg, &q, &skip, &mut obs, Instant::now()).unwrap().pick.map(|p| p.run.id.clone()), Some("s2".into()));
    std::fs::remove_dir_all(&root).ok();
}

/// 061-R13 (H-1): `--init` refuses a run folder without `DONE` and uses its final checkpoint.
#[test]
fn init_needs_a_finished_run_061_r13() {
    let root = tmp("h1-init");
    let run = root.join("s1");
    std::fs::create_dir_all(run.join("ckpt-5")).unwrap();
    std::fs::write(run.join("LATEST"), "ckpt-5\n").unwrap();
    let err = uranium_train::bc_run::resolve_init(&run).unwrap_err();
    assert!(format!("{err:#}").contains("not finished"), "{err:#}");
    std::fs::write(run.join("DONE"), "ckpt-5\n").unwrap();
    assert_eq!(uranium_train::bc_run::resolve_init(&run).unwrap(), run.join("ckpt-5"));
    // A checkpoint folder is used as is.
    assert_eq!(uranium_train::bc_run::resolve_init(&run.join("ckpt-5")).unwrap(), run.join("ckpt-5"));
    std::fs::remove_dir_all(&root).ok();
}

/// 061-R13 (H-2): an unreachable share takes nothing; a run whose lock cannot be read is unknown
/// (not free) and skipped for that poll while the other runs stay pickable (2026-10-19).
#[test]
fn unknown_share_state_takes_nothing_061_r13() {
    let root = tmp("h2-pick");
    let cfg = local_cfg(&root, "me");
    let q = queue("[[run]]\nid = \"r\"\n[[run]]\nid = \"s\"\n");
    let mut obs = Observers::default();
    let skip = HashMap::new();
    // No `checkpoints/` (e.g. an empty mount point): unknown.
    assert!(pick(&cfg, &q, &skip, &mut obs, Instant::now()).is_err());
    std::fs::create_dir_all(root.join("shared").join("checkpoints")).unwrap();
    std::fs::create_dir_all(root.join("shared").join("locks")).unwrap();
    // A lock that cannot be parsed is not Free: `r` is skipped, `s` is taken.
    std::fs::write(root.join("shared").join("locks").join("r.json"), b"{").unwrap();
    let p = pick(&cfg, &q, &skip, &mut obs, Instant::now()).unwrap();
    assert_eq!(p.pick.map(|p| p.run.id.clone()), Some("s".into()));
    assert_eq!(p.unknown.iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>(), ["r"]);
    // An unreadable yield request is unknown too.
    std::fs::remove_file(root.join("shared").join("locks").join("r.json")).unwrap();
    std::fs::write(root.join("shared").join("locks").join("r.yield"), b"not json").unwrap();
    let p = pick(&cfg, &q, &skip, &mut obs, Instant::now()).unwrap();
    assert_eq!((p.pick.map(|p| p.run.id.clone()), p.unknown.len()), (Some("s".into()), 1));
    std::fs::remove_file(root.join("shared").join("locks").join("r.yield")).unwrap();
    assert_eq!(pick(&cfg, &q, &skip, &mut obs, Instant::now()).unwrap().pick.map(|p| p.run.id.clone()), Some("r".into()));
    std::fs::remove_dir_all(&root).ok();
}

/// 061-R14: a yield request expires only after a worker has seen it unchanged for 30 min on its
/// own monotonic clock (file times play no part); the expired request is removed.
#[test]
fn yield_request_expires_on_the_own_clock_061_r14() {
    let root = tmp("r14-expiry");
    let cfg = local_cfg(&root, "me");
    std::fs::create_dir_all(root.join("shared").join("checkpoints")).unwrap();
    let q = queue("[[run]]\nid = \"r\"\n");
    let other = local_cfg(&root, "other");
    storage::request_yield(&other, "r", 5).unwrap();
    // An old file time does not expire it.
    let path = root.join("shared").join("locks").join("r.yield");
    std::fs::File::options().write(true).open(&path).unwrap().set_modified(SystemTime::now() - Duration::from_secs(7200)).unwrap();
    let skip = HashMap::new();
    let mut obs = Observers::default();
    let t0 = Instant::now();
    for m in [0, 10, 20, 29] {
        let p = pick(&cfg, &q, &skip, &mut obs, t0 + Duration::from_secs(m * 60)).unwrap();
        assert!(p.pick.is_none() && p.expired.is_empty(), "reserved for `other` at {m} min");
    }
    // Seen unchanged for 30 min: removed; the run is free at the next look.
    let p = pick(&cfg, &q, &skip, &mut obs, t0 + Duration::from_secs(30 * 60)).unwrap();
    assert_eq!(p.expired, [("r".to_string(), "other".to_string())]);
    assert!(storage::read_yield(&cfg, "r").unwrap().is_none());
    assert_eq!(pick(&cfg, &q, &skip, &mut obs, t0 + Duration::from_secs(31 * 60)).unwrap().pick.map(|p| p.run.id.clone()), Some("r".into()));
    // A renewed request (new contents) starts the 30 minutes over.
    storage::request_yield(&other, "r", 5).unwrap();
    let t1 = t0 + Duration::from_secs(40 * 60);
    assert!(pick(&cfg, &q, &skip, &mut obs, t1).unwrap().pick.is_none());
    std::fs::remove_dir_all(&root).ok();
}

/// 061-R14 (user decision 2026-10-19): a run started by hand (`bc --run`, no worker state naming
/// it) is never asked to yield; a run a slower worker trains is.
#[test]
fn manual_runs_are_not_asked_to_yield_061_r14() {
    let root = tmp("r14-manual");
    let cfg = local_cfg(&root, "fast");
    std::fs::create_dir_all(root.join("shared").join("checkpoints")).unwrap();
    let q = queue("[[run]]\nid = \"r\"\n");
    let slow = local_cfg(&root, "slow");
    let _lock = Run::acquire(slow.clone(), "r", false).unwrap();
    // No worker state at all: manual.
    assert_eq!(preempt_target(&cfg, &q, 10, &HashMap::new()).unwrap(), None);
    // A worker on that host, but on another run (or idle): the run is still the manual one.
    storage::write_worker(&slow, 0, "vulkan", None).unwrap();
    assert_eq!(preempt_target(&cfg, &q, 10, &HashMap::new()).unwrap(), None);
    storage::write_worker(&slow, 0, "vulkan", Some("other")).unwrap();
    assert_eq!(preempt_target(&cfg, &q, 10, &HashMap::new()).unwrap(), None);
    // The slower worker trains it: asked (higher priority only).
    storage::write_worker(&slow, 0, "vulkan", Some("r")).unwrap();
    assert_eq!(preempt_target(&cfg, &q, 10, &HashMap::new()).unwrap().map(|r| r.id.clone()), Some("r".into()));
    assert_eq!(preempt_target(&cfg, &q, 0, &HashMap::new()).unwrap(), None);
    // A run this worker gave up on for now (GiveUp skip) is not asked either.
    let skip: HashMap<String, Instant> = [("r".to_string(), Instant::now())].into();
    assert_eq!(preempt_target(&cfg, &q, 10, &skip).unwrap(), None);
    std::fs::remove_dir_all(&root).ok();
}

/// 061-R5: a poll interval of half the lock expiry or more could never see a lock expire.
#[test]
fn slow_poll_is_refused_061_r5() {
    let root = tmp("poll");
    let cfg = local_cfg(&root, "me");
    let mut a = args(root.join("queue.toml"), 1);
    a.poll = cfg.stale_after / 2;
    let err = work(&cfg, &a, &Instant0 { cfg: cfg.clone(), started: Arc::new(Mutex::new(Vec::new())) }, &|| false, &|_| {}, &|_| false).unwrap_err();
    assert!(format!("{err:#}").contains("--poll-seconds"), "{err:#}");
    std::fs::remove_dir_all(&root).ok();
}

/// Ends with 75 every time (the lock went elsewhere between the look and the start).
struct Busy {
    calls: Arc<AtomicUsize>,
}

impl Launcher for Busy {
    fn launch(&self, _r: &QueueRun, _d: &Defaults, _take: &TakeOver) -> Result<Ending> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(Ending::Exited(Some(75)))
    }
}

/// 061-R14: when the requested run is picked but its child ends with 75 (never took it), this
/// host's request is removed at once, not left as a 30-minute reservation.
#[test]
fn request_removed_when_the_child_does_not_take_the_run_061_r14() {
    let root = tmp("r14-75");
    let cfg = local_cfg(&root, "high");
    std::fs::create_dir_all(root.join("shared").join("checkpoints")).unwrap();
    let queue = root.join("queue.toml");
    std::fs::write(&queue, "[[run]]\nid = \"a\"\n").unwrap();
    // The slower worker trains `a`: the fast one asks for it.
    let low = local_cfg(&root, "low");
    storage::write_worker(&low, 0, "vulkan", Some("a")).unwrap();
    let lock = Run::acquire(low.clone(), "a", false).unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let notes = Arc::new(Mutex::new(Vec::new()));
    let stop = Arc::new(AtomicBool::new(false));
    let w = spawn_worker(cfg.clone(), args(queue, 10), Busy { calls: calls.clone() }, stop.clone(), notes.clone());
    wait_for("request", Duration::from_secs(10), || storage::read_yield(&cfg, "a").unwrap().is_some());
    // The slow PC yields (lock released); the fast one picks `a`, its child ends with 75.
    lock.release().unwrap();
    wait_for("child ran", Duration::from_secs(10), || calls.load(Ordering::SeqCst) >= 1);
    stop.store(true, Ordering::SeqCst);
    w.join().unwrap();
    assert!(storage::read_yield(&cfg, "a").unwrap().is_none(), "request left behind: {:?}", notes.lock().unwrap());
    std::fs::remove_dir_all(&root).ok();
}

/// 060-R6 (must-fix): a training child whose stderr pipe closed (watchdog killed hard) keeps
/// running through `elog!` instead of panicking on EPIPE like `eprintln!`. The test runs itself
/// as a child with stderr piped and the read end closed at once.
#[test]
fn logging_survives_a_closed_stderr_060_r6() {
    const MODE: &str = "URANIUM_ELOG_CHILD";
    if let Ok(mode) = std::env::var(MODE) {
        let marker = std::path::PathBuf::from(std::env::var("URANIUM_ELOG_MARKER").unwrap());
        std::thread::sleep(Duration::from_millis(300)); // the parent closes the pipe meanwhile
        for i in 0..2000 {
            if mode == "elog" {
                uranium_train::elog!("line {i} after the watchdog died");
            } else {
                eprintln!("line {i} after the watchdog died");
            }
        }
        std::fs::write(&marker, b"saved").unwrap(); // stands for save, push, release
        return;
    }
    let root = tmp("elog");
    let run = |mode: &str| -> (bool, bool) {
        let marker = root.join(format!("marker-{mode}"));
        let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "logging_survives_a_closed_stderr_060_r6", "--nocapture", "--test-threads=1"])
            .env(MODE, mode)
            .env("URANIUM_ELOG_MARKER", &marker)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        drop(child.stderr.take()); // the watchdog is gone
        let ok = child.wait().unwrap().success();
        (ok, marker.exists())
    };
    assert_eq!(run("elog"), (true, true), "elog! must not stop the child");
    // Control: plain eprintln! panics on the closed pipe before the "save".
    assert_eq!(run("eprintln").1, false, "eprintln! was expected to fail on a closed stderr");
    std::fs::remove_dir_all(&root).ok();
}

/// Moves the share away / back (an outage); retries while files are briefly open (Windows).
fn rename_retry(from: &Path, to: &Path) {
    for _ in 0..200 {
        if to.exists() {
            std::fs::remove_dir_all(to).ok(); // what a status write recreated meanwhile
        }
        if std::fs::rename(from, to).is_ok() {
            return;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    panic!("could not rename {} to {}", from.display(), to.display());
}

/// Fails while the share is gone (it takes it away on its first calls), then trains to done.
struct Outage {
    cfg: StorageConfig,
    away: PathBuf,
    calls: Arc<AtomicUsize>,
}

impl Launcher for Outage {
    fn launch(&self, r: &QueueRun, _d: &Defaults, _take: &TakeOver) -> Result<Ending> {
        let n = self.calls.fetch_add(1, Ordering::SeqCst);
        if n < 4 {
            rename_retry(&self.cfg.shared, &self.away);
            return Ok(Ending::Exited(Some(1)));
        }
        storage::mark_done(&self.cfg, &r.id, 1)?;
        Ok(Ending::Exited(Some(0)))
    }
}

/// 061-R13 (H-2): queue unreadable at start, children failing while the share is unreachable:
/// the worker stays up, does not burn restarts or give the run up, and finishes it later.
#[test]
fn share_outage_keeps_the_worker_alive_061_r13() {
    let root = tmp("h2");
    let cfg = local_cfg(&root, "me");
    let away = root.join("shared-away");
    let queue = root.join("queue.toml");
    let calls = Arc::new(AtomicUsize::new(0));
    let notes = Arc::new(Mutex::new(Vec::new()));
    let stop = Arc::new(AtomicBool::new(false));
    let mut a = args(queue.clone(), 1);
    a.supervise.max_restarts = 1; // the old loop would have given up after two failures
    a.poll = Duration::from_millis(50);
    let w = spawn_worker(cfg.clone(), a, Outage { cfg: cfg.clone(), away: away.clone(), calls: calls.clone() }, stop.clone(), notes.clone());
    // No queue yet: the worker keeps looking.
    std::thread::sleep(Duration::from_millis(300));
    assert!(!w.is_finished(), "worker ended on an unreadable queue");
    std::fs::write(&queue, "[[run]]\nid = \"r\"\n").unwrap();
    // The share comes back a little after each outage.
    let restorer = {
        let (shared, away, stop) = (cfg.shared.clone(), away.clone(), stop.clone());
        std::thread::spawn(move || {
            while !stop.load(Ordering::SeqCst) {
                if away.exists() {
                    std::thread::sleep(Duration::from_millis(150));
                    rename_retry(&away, &shared);
                }
                std::thread::sleep(Duration::from_millis(10));
            }
        })
    };
    wait_for("r done", Duration::from_secs(30), || storage::is_done(&cfg, "r").unwrap_or(false));
    stop.store(true, Ordering::SeqCst);
    w.join().unwrap();
    restorer.join().unwrap();
    let notes = notes.lock().unwrap().clone();
    assert_eq!(calls.load(Ordering::SeqCst), 5, "{notes:?}");
    assert!(notes.iter().any(|n| n.contains("queue unreadable")), "{notes:?}");
    assert!(notes.iter().any(|n| n.contains("not counted as a restart")), "{notes:?}");
    assert!(!notes.iter().any(|n| n.contains("skipping it for an hour")), "{notes:?}");
    std::fs::remove_dir_all(&root).ok();
}

fn lock(host: &str, beat: u64) -> LockInfo {
    LockInfo { host: host.into(), pid: 7, started: 1, heartbeat: beat }
}

/// 061-R5 / AC3 (H-3): expiry = the same heartbeat watched for `stale_after` on the observer's
/// own monotonic clock; a change, a read failure or a gap in watching starts over.
#[test]
fn expiry_is_watched_on_the_own_clock_061_ac3() {
    let stale = Duration::from_secs(600);
    let t0 = Instant::now();
    let m = |s: u64| t0 + Duration::from_secs(s);
    let mut o = LockObserver::default();
    let held = |v: LockView| matches!(v, LockView::Held(_));
    assert!(held(o.observe("r", &Ok(Some(lock("pc", 100))), m(0), stale)));
    assert!(held(o.observe("r", &Ok(Some(lock("pc", 100))), m(299), stale)));
    assert!(held(o.observe("r", &Ok(Some(lock("pc", 100))), m(598), stale)));
    assert_eq!(o.observe("r", &Ok(Some(lock("pc", 100))), m(600), stale), LockView::Stale(lock("pc", 100)));
    // A new heartbeat starts over.
    assert!(held(o.observe("r", &Ok(Some(lock("pc", 160))), m(660), stale)));
    assert!(held(o.observe("r", &Ok(Some(lock("pc", 160))), m(900), stale)));
    // A read failure is unknown and starts over.
    assert!(matches!(o.observe("r", &Err(anyhow::anyhow!("share gone")), m(1000), stale), LockView::Unknown(_)));
    assert!(held(o.observe("r", &Ok(Some(lock("pc", 160))), m(1100), stale)));
    // A gap longer than half of stale_after (not watching) starts over.
    assert!(held(o.observe("r", &Ok(Some(lock("pc", 160))), m(1500), stale))); // 400 s gap: from 1500
    assert!(held(o.observe("r", &Ok(Some(lock("pc", 160))), m(1750), stale)));
    assert!(held(o.observe("r", &Ok(Some(lock("pc", 160))), m(2000), stale)));
    assert!(held(o.observe("r", &Ok(Some(lock("pc", 160))), m(2099), stale)));
    assert!(matches!(o.observe("r", &Ok(Some(lock("pc", 160))), m(2100), stale), LockView::Stale(_)));
    assert_eq!(o.observe("r", &Ok(None), m(2101), stale), LockView::Free);
}

/// 061-AC3 (H-3): a worker takes another PC's lock over only after watching it unchanged; the
/// lock file's mtime changes nothing, and a lock that changed since it was seen is not taken.
#[test]
fn auto_take_over_ignores_file_times_061_ac3() {
    let root = tmp("h3");
    let cfg = local_cfg(&root, "me");
    std::fs::create_dir_all(root.join("shared").join("checkpoints")).unwrap();
    let q = queue("[[run]]\nid = \"r\"\n");
    let mut other = Run::acquire(local_cfg(&root, "other"), "r", false).unwrap();
    let path = other.lock_path();
    let set_mtime = |age: Duration| std::fs::File::options().write(true).open(&path).unwrap().set_modified(SystemTime::now() - age).unwrap();
    let skip = HashMap::new();
    let t0 = Instant::now();
    let mut obs = Observers::default();
    // An hour-old file time on the first look: not expired.
    set_mtime(Duration::from_secs(3600));
    assert_eq!(pick(&cfg, &q, &skip, &mut obs, t0).unwrap().pick, None);
    // Watched for 10 minutes (polls every 4): expired, whatever the file time says.
    set_mtime(Duration::ZERO);
    assert_eq!(pick(&cfg, &q, &skip, &mut obs, t0 + Duration::from_secs(240)).unwrap().pick, None);
    assert_eq!(pick(&cfg, &q, &skip, &mut obs, t0 + Duration::from_secs(480)).unwrap().pick, None);
    let p = pick(&cfg, &q, &skip, &mut obs, t0 + Duration::from_secs(600)).unwrap().pick.unwrap();
    let Pick { take: TakeOver::IfUnchanged(seen), .. } = p else { panic!("{p:?}") };
    // The owner wakes up before the take-over: the changed lock is not taken (exit 75).
    std::thread::sleep(Duration::from_millis(1100));
    assert_eq!(other.heartbeat(), storage::LockStatus::Held);
    let err = Run::acquire_with(cfg.clone(), "r", TakeOver::IfUnchanged(seen)).err().unwrap();
    assert!(err.downcast_ref::<LockBusy>().is_some(), "{err:#}");
    // Unchanged: taken; the old owner sees it lost.
    let now = storage::peek_lock(&cfg, "r").unwrap().unwrap();
    let mine = Run::acquire_with(cfg.clone(), "r", TakeOver::IfUnchanged(now)).unwrap();
    assert_eq!(mine.previous_owner().unwrap().host, "other");
    assert!(matches!(other.heartbeat(), storage::LockStatus::Lost(l) if l.host == "me"));
    std::fs::remove_dir_all(&root).ok();
}

/// A launcher that finishes a run at once.
struct Instant0 {
    cfg: StorageConfig,
    started: Arc<Mutex<Vec<String>>>,
}

impl Launcher for Instant0 {
    fn launch(&self, r: &QueueRun, _d: &Defaults, take: &TakeOver) -> Result<Ending> {
        let run = Run::acquire_with(self.cfg.clone(), &r.id, take.clone())?;
        self.started.lock().unwrap().push(r.id.clone());
        storage::mark_done(&self.cfg, &r.id, 1)?;
        run.release()?;
        Ok(Ending::Exited(Some(0)))
    }
}

/// 061-R14 ①: a host withdraws its yield request when it takes another run and when its worker
/// stops; the status file keeps being rewritten (061-R13).
#[test]
fn yield_request_is_withdrawn_061_r14() {
    let root = tmp("r14");
    let shared = root.join("shared");
    std::fs::create_dir_all(shared.join("checkpoints")).unwrap();
    let queue = root.join("queue.toml");
    std::fs::write(&queue, "[[run]]\nid = \"a\"\n[[run]]\nid = \"b\"\ninit_run = \"pre\"\n").unwrap();
    // `a` trains on a slower PC (priority 0); `b` waits for `pre`.
    let low = local_cfg(&root, "low");
    storage::write_worker(&low, 0, "vulkan", Some("a")).unwrap();
    let _a = Run::acquire(low.clone(), "a", false).unwrap();
    let high = local_cfg(&root, "high");
    let started = Arc::new(Mutex::new(Vec::new()));
    let notes = Arc::new(Mutex::new(Vec::new()));
    let stop = Arc::new(AtomicBool::new(false));
    let mut a = args(queue.clone(), 10);
    a.status_every = Duration::from_millis(100);
    let w = spawn_worker(high.clone(), a, Instant0 { cfg: high.clone(), started: started.clone() }, stop.clone(), notes.clone());
    let req_on_a = || storage::read_yield(&high, "a").unwrap().is_some_and(|y| y.host == "high");
    wait_for("a yield request", Duration::from_secs(10), req_on_a);
    // The status file is refreshed while the worker waits.
    let t1 = storage::read_worker(&high, "high").unwrap().time;
    wait_for("status refresh", Duration::from_secs(5), || storage::read_worker(&high, "high").is_some_and(|w| w.time > t1));
    // `b` becomes takeable: the worker takes it and withdraws its request on `a`.
    storage::mark_done(&high, "pre", 1).unwrap();
    wait_for("b taken", Duration::from_secs(10), || started.lock().unwrap().contains(&"b".to_string()));
    wait_for("request withdrawn", Duration::from_secs(10), || notes.lock().unwrap().iter().any(|n| n.contains("withdrew the yield request on a")));
    // Nothing left to take: it asks again; stopping withdraws that request too.
    wait_for("a asked again", Duration::from_secs(10), req_on_a);
    stop.store(true, Ordering::SeqCst);
    w.join().unwrap();
    assert!(!req_on_a(), "request left behind after the worker stopped");
    assert_eq!(storage::read_worker(&high, "high").unwrap().run, None);
    // Another host's request is never removed by this host.
    storage::request_yield(&low, "a", 0).unwrap();
    storage::clear_yield(&high, "a").unwrap();
    assert!(storage::read_yield(&high, "a").unwrap().is_some());
    std::fs::remove_dir_all(&root).ok();
}
