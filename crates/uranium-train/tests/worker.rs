//! 061-AC8: two run-queue workers (fake training, no GPU) on one shared folder — never two owners
//! of a run, yield requests hand a run to the higher priority, a Ctrl+C stop hands it back.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::Result;
use uranium_train::storage::{self, LockBusy, Run, StorageConfig};
use uranium_train::supervise::{Ending, SuperviseArgs};
use uranium_train::worker::{Defaults, Launcher, QueueRun, WorkerArgs, work};

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
    fn launch(&self, r: &QueueRun, _d: &Defaults, take_over: bool) -> Result<Ending> {
        let me = self.cfg.host.clone();
        let run = match Run::acquire(self.cfg.clone(), &r.id, take_over) {
            Ok(run) => run,
            Err(e) if e.downcast_ref::<LockBusy>().is_some() => return Ok(Ending::Exited(Some(75))),
            Err(e) => return Err(e),
        };
        match storage::read_yield(&self.cfg, &r.id) {
            Some(y) if y.host != me => {
                run.release()?;
                return Ok(Ending::Exited(Some(75)));
            }
            Some(_) => storage::clear_yield(&self.cfg, &r.id),
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
            if storage::read_yield(&self.cfg, &r.id).is_some_and(|y| y.host != me) {
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

#[test]
fn two_workers_share_a_queue_061_ac8() {
    let root = std::env::temp_dir().join(format!("uranium-worker-{}", std::process::id()));
    std::fs::remove_dir_all(&root).ok();
    let shared = root.join("shared");
    std::fs::create_dir_all(&shared).unwrap();
    let queue = root.join("queue.toml");
    std::fs::write(&queue, "[[run]]\nid = \"c\"\n").unwrap();
    let owners = Arc::new(Mutex::new(HashMap::new()));
    let events = Arc::new(Mutex::new(Vec::new()));
    let cfg = |host: &str| {
        let mut c = StorageConfig::new(&shared, root.join(format!("cache-{host}")), host);
        c.shared_local = true;
        c
    };
    let spawn = |host: &'static str, priority: i32, stop: Arc<AtomicBool>| {
        let fake = Fake { cfg: cfg(host), target: 60, tick: Duration::from_millis(40), owners: owners.clone(), events: events.clone(), stop: stop.clone() };
        let (c, a) = (cfg(host), args(queue.clone(), priority));
        std::thread::spawn(move || {
            let stopped = || stop.load(Ordering::SeqCst);
            let wait = |d: Duration| {
                std::thread::sleep(d);
                !stop.load(Ordering::SeqCst)
            };
            work(&c, &a, &fake, &stopped, &|_m| {}, &wait).unwrap();
        })
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
    wait_for("c done", Duration::from_secs(20), || storage::is_done(&cfg("low"), "c"));
    low_stop.store(true, Ordering::SeqCst);
    low.join().unwrap();

    let ev = events.lock().unwrap().clone();
    let names: Vec<String> = ev.iter().map(|e| e.split(" at ").next().unwrap().to_string()).collect();
    assert_eq!(names, ["low start c", "low yielded c", "high start c", "high stopped c", "low start c", "low done c"], "{ev:?}");
    // Progress only moves forward across owners (each start resumes where the last one stopped).
    let steps: Vec<u64> = ev.iter().map(|e| e.rsplit(' ').next().unwrap().parse().unwrap()).collect();
    assert!(steps.windows(2).all(|w| w[0] <= w[1]), "{ev:?}");
    assert_eq!(*steps.last().unwrap(), 60);
    assert!(storage::read_yield(&cfg("low"), "c").is_none(), "request cleared by the host that asked");
    assert_eq!(storage::peek_lock(&cfg("low"), "c").unwrap(), storage::LockView::Free);
    std::fs::remove_dir_all(&root).ok();
}
