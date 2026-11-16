use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use anyhow::Result;
use uranium_store::elog;
use uranium_store::storage::{self, now_secs};
use crate::plan;
use crate::{Job, JobOutcome, JobSource};
use super::{hostd_id, split_id, Key, key_of, retryable, EvalJob};
use super::ledger::{LedgerState, is_failure, pending, Inflight, Flights};

pub struct LedgerSource {
    pub(super) job: Arc<EvalJob>,
    host: String,
    jobs: Vec<Vec<Job>>,
    loaded: HashSet<usize>,
    inner: Mutex<Inner>,
    write: Mutex<()>,
}

struct Inner {
    open: bool,
    paused: bool,
    queue: VecDeque<Key>,
    unloaded: usize,
    mine: HashMap<Key, u64>,
    finished: HashSet<Key>,
    attempts: HashMap<Key, u32>,
    failures: HashMap<Key, u32>,
    busy_elsewhere: HashSet<Key>,
    known: bool,
}

impl LedgerSource {
    pub fn new(job: Arc<EvalJob>, host: &str, agent_addr: &str, loaded: HashSet<usize>) -> Self {
        let jobs = job.stages.iter().map(|s| plan::expand(&s.plan, agent_addr)).collect();
        let inner = Inner { open: true, paused: false, queue: VecDeque::new(), unloaded: 0, mine: HashMap::new(), finished: HashSet::new(), attempts: HashMap::new(), failures: HashMap::new(), busy_elsewhere: HashSet::new(), known: false };
        Self { job, host: host.to_string(), jobs, loaded, inner: Mutex::new(inner), write: Mutex::new(()) }
    }

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

    pub fn remember_finished(&self, keys: &HashSet<Key>) {
        self.inner.lock().unwrap().finished.extend(keys.iter().copied());
    }

    pub fn finished(&self) -> HashSet<Key> {
        self.inner.lock().unwrap().finished.clone()
    }

    pub fn is_open(&self) -> bool {
        self.inner.lock().unwrap().open
    }

    pub fn pause(&self, paused: bool) {
        self.inner.lock().unwrap().paused = paused;
    }

    pub fn unloaded(&self) -> usize {
        self.inner.lock().unwrap().unloaded
    }

    pub fn in_flight(&self) -> Vec<u64> {
        let mut v: Vec<u64> = self.inner.lock().unwrap().mine.values().copied().collect();
        v.sort_unstable();
        v
    }

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

    pub(super) fn write_inflight_logged(&self) {
        if let Err(e) = self.write_inflight() {
            elog!("[eval] could not write the in-flight file ({e:#}); retrying");
        }
    }

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
            std::thread::sleep(Duration::from_secs(2));
        }
    }

    fn requeue(&self, job: Job) {
        let k = key_of(job.spec.game_id);
        let mut g = self.inner.lock().unwrap();
        g.mine.remove(&k);
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

    fn memory_log(&self, job: &Job) -> Option<PathBuf> {
        let k = key_of(job.spec.game_id);
        let stage = self.job.stages.get(k.0)?;
        stage.plan.memory_log.then(|| self.job.sub(&stage.name).join("memory").join(crate::memory_log_name(k.1)))
    }

    fn game_branch(&self, job: &Job) -> Result<Option<uranium_env::branch::GameBranch>> {
        let k = key_of(job.spec.game_id);
        let Some(stage) = self.job.stages.get(k.0) else { return Ok(None) };
        stage.plan.game_branch(k.1, &self.job.sub(&stage.name), plan::shared_dir_env().as_deref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uranium_store::storage::StorageConfig;
    use crate::eval::ledger::ledger_state;
    use crate::eval::worker::EvalWorker;
    use crate::eval::testkit::{rec, job_dir, all_stages};

    fn no_flights() -> Flights {
        Flights { busy: HashSet::new(), attempts: HashMap::new() }
    }

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
        src.refresh(&LedgerState::default(), no_flights());
        assert!(src.next().is_none());
        assert!(src.complete());
        let next_period = LedgerSource::new(job.clone(), "h", "127.0.0.1:1", all_stages());
        next_period.remember_finished(&src.finished());
        next_period.refresh(&LedgerState::default(), no_flights());
        assert!(next_period.next().is_none(), "games recorded in an earlier period are not played again");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn memory_log_beside_the_stage_replay_041_r28() {
        let dir = job_dir("memlog", 2);
        let p = dir.join("plans/1.toml");
        let text = std::fs::read_to_string(&p).unwrap().replace("save_replay=true\n", "save_replay=true\nmemory_log=true\n");
        std::fs::write(&p, text).unwrap();
        let job = Arc::new(EvalJob::load(&dir).unwrap());
        let src = LedgerSource::new(job.clone(), "h", "127.0.0.1:1", all_stages());
        let mut j = plan::expand(&job.stages[1].plan, "x").remove(1);
        j.spec.game_id = hostd_id(1, 2, 3);
        assert_eq!(src.memory_log(&j), Some(dir.join("b/memory/g2.jsonl")));
        j.spec.game_id = hostd_id(0, 2, 0);
        assert_eq!(src.memory_log(&j), None, "stage a has no memory_log");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn act_logs_under_the_stage_020_r22() {
        let dir = job_dir("actlog", 2);
        let p = dir.join("plans/1.toml");
        let text = std::fs::read_to_string(&p).unwrap().replace("save_replay=true\n", "save_replay=true\nact_log=true\n");
        std::fs::write(&p, text).unwrap();
        let job = Arc::new(EvalJob::load(&dir).unwrap());
        let src = LedgerSource::new(job.clone(), "h", "127.0.0.1:1", all_stages());
        let mut j = plan::expand(&job.stages[1].plan, "x").remove(1);
        j.spec.game_id = hostd_id(1, 2, 3);
        let g = src.game_branch(&j).unwrap().unwrap();
        assert_eq!((g.plan_game, g.act_log_path(1)), (2, Some(dir.join("b").join("acts").join("g2-p1.jsonl"))));
        assert!(g.states_dir.is_none() && g.branch.is_none());
        j.spec.game_id = hostd_id(0, 2, 0);
        assert_eq!(src.game_branch(&j).unwrap(), None, "stage a has no 051 key");
        let cfg = StorageConfig::new(&dir, &dir, "h");
        let w = EvalWorker::start(&dir, cfg, 0, "cpu", None, Arc::default(), "127.0.0.1:0").unwrap();
        w.close();
        std::fs::remove_dir_all(&dir).ok();
    }

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
}
