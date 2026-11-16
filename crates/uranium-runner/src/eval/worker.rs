use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use anyhow::Result;
use uranium_agent::infer::InferenceHandle;
use uranium_agent::{EndBoard, Routes, T0Builds};
use uranium_store::elog;
use uranium_store::storage::{self, ContentObserver, Lease, LockObserver, LockStatus, LockView, StorageConfig, TakeOver, Watched, YieldObserver, now_secs};
use crate::plan::{self, ModelKey};
use crate::{Dispatch, InferTag, Models};
use super::{Key, EvalJob};
use super::ledger::{unit_rejections_file, read_records, LedgerState, ledger_state, pending, Inflight, Flights, flights};
use super::source::LedgerSource;
use super::dispatch::{run_hosts, mark_leaving, clear_leaving, DispatchEnd};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaitEnd {
    Lease,
    Done,
    Stopped,
}

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

struct SetOnDrop<'a>(&'a AtomicBool);

impl Drop for SetOnDrop<'_> {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

pub struct EvalWorker {
    pub job: Arc<EvalJob>,
    pub cfg: StorageConfig,
    pub priority: i32,
    pub backend: String,
    pub local_hostd: Option<String>,
    pub stop: Arc<AtomicUsize>,
    pub agent_addr: String,
    routes: Routes,
    builds: T0Builds,
    ends: EndBoard,
    branches: uranium_env::branch::BranchBoard,
    lease: Arc<Mutex<Option<Lease>>>,
    recorded: Mutex<HashSet<Key>>,
    lost: Arc<AtomicBool>,
    lease_unknown: Arc<AtomicBool>,
    marked: AtomicBool,
    closed: Arc<AtomicBool>,
    pub every: Duration,
    pub tick: Duration,
}

impl EvalWorker {
    pub fn start(job_dir: &Path, cfg: StorageConfig, priority: i32, backend: &str, local_hostd: Option<String>, stop: Arc<AtomicUsize>, listen: &str) -> Result<Self> {
        let job = Arc::new(EvalJob::load(job_dir)?);
        storage::check_name(&cfg.host)?;
        for s in &job.stages {
            crate::check_branch_support(&s.plan)?;
            s.plan.check_branch_paths(plan::shared_dir_env().as_deref())?;
        }
        let (routes, builds, ends) = (Routes::default(), T0Builds::default(), EndBoard::default());
        let branches = uranium_env::branch::BranchBoard::default();
        let results = job.sub("results").join(format!("{}.jsonl", cfg.host));
        let agent_addr = crate::start_agent(listen, &results, &unit_rejections_file(&job, &cfg.host), routes.clone(), builds.clone(), ends.clone(), Default::default(), branches.clone())?;
        elog!("[eval] job {} ({} stages, {} games), host {} priority {} backend {}, agent at {agent_addr}", job.dir.display(), job.stages.len(), job.keys().len(), cfg.host, priority, backend);
        if let Some(addr) = &local_hostd {
            clear_leaving(&job, addr)?;
        }
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
            branches,
            lease: Arc::new(Mutex::new(None)),
            recorded: Mutex::new(HashSet::new()),
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

    fn snapshot(&self, obs: &mut ContentObserver<Inflight>) -> Result<(LedgerState, Flights)> {
        let fl = flights(&self.job, &self.cfg.host, obs, self.cfg.stale_after)?;
        Ok((ledger_state(&read_records(&self.job)?, self.job.max_attempts), fl))
    }

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

    pub fn dispatch(&self, handles: &HashMap<ModelKey, InferenceHandle>) -> Result<DispatchEnd> {
        let mut models = Models::new();
        let mut loaded = HashSet::new();
        for (si, s) in self.job.stages.iter().enumerate() {
            if crate::plan_models(&s.plan).iter().all(|k| handles.contains_key(k)) {
                loaded.insert(si);
                for (k, p) in crate::plan_policies(&s.plan, handles)? {
                    models.entry(k).or_insert(p);
                }
            }
        }
        let mut dispatch = Dispatch::new(self.routes.clone(), models, self.builds.clone()).with_ends(self.ends.clone()).with_branches(self.branches.clone());
        dispatch.infer = Some(InferTag { host: self.cfg.host.clone(), backend: self.backend.clone(), stages: self.job.stage_names() });
        let source = Arc::new(LedgerSource::new(self.job.clone(), &self.cfg.host, &self.agent_addr, loaded));
        source.remember_finished(&self.recorded.lock().unwrap());
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
        self.recorded.lock().unwrap().extend(source.finished());
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::testkit::job_dir;

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
