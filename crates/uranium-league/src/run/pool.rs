use std::collections::{BTreeSet, HashMap, VecDeque};
use std::path::PathBuf;
use std::time::{Duration, Instant};
use anyhow::{Context, Result};
use uranium_agent::infer::InferenceHandle;
use crate::learner::SnapshotServers;
use super::Shared;

const SNAPSHOT_SERVERS_MAX: usize = 8;

const SNAPSHOT_RETRY_AFTER: Duration = Duration::from_secs(300);

struct LiveServer {
    key: String,
    id: u64,
    handle: InferenceHandle,
    close: Box<dyn FnOnce() + Send>,
    alive: Box<dyn Fn() -> bool + Send>,
    games: usize,
}

pub(super) struct SnapshotPool {
    pub(super) dirs: HashMap<String, PathBuf>,
    live: VecDeque<LiveServer>,
    cap: usize,
    pinned: Option<String>,
    servers: SnapshotServers,
    failed: HashMap<String, Instant>,
    retry_after: Duration,
    next_id: u64,
}

type Closer = (String, Box<dyn FnOnce() + Send>);

impl SnapshotPool {
    pub(super) fn new(dirs: HashMap<String, PathBuf>, pinned: Option<String>, servers: SnapshotServers) -> Self {
        Self { dirs, live: VecDeque::new(), cap: SNAPSHOT_SERVERS_MAX, pinned, servers, failed: HashMap::new(), retry_after: SNAPSHOT_RETRY_AFTER, next_id: 0 }
    }

    pub(super) fn acquire(&mut self, key: &str) -> Result<(InferenceHandle, u64)> {
        if let Some(at) = self.failed.get(key) {
            if at.elapsed() < self.retry_after {
                anyhow::bail!("its server failed {:.0} s ago (retried after {} s)", at.elapsed().as_secs_f64(), self.retry_after.as_secs());
            }
            self.failed.remove(key);
        }
        if let Some(i) = self.live.iter().position(|l| l.key == key) {
            let mut l = self.live.remove(i).unwrap();
            if !(l.alive)() {
                (l.close)();
                self.failed.insert(key.to_string(), Instant::now());
                anyhow::bail!("its inference server stopped (model thread ended)");
            }
            l.games += 1;
            let out = (l.handle.clone(), l.id);
            self.live.push_back(l);
            return Ok(out);
        }
        let dir = self.dirs.get(key).with_context(|| format!("no published model folder for {key}"))?;
        let s = match (self.servers)(dir) {
            Ok(s) => s,
            Err(e) => {
                self.failed.insert(key.to_string(), Instant::now());
                return Err(e);
            }
        };
        self.next_id += 1;
        let out = (s.handle.clone(), self.next_id);
        self.live.push_back(LiveServer { key: key.to_string(), id: self.next_id, handle: s.handle, close: s.close, alive: s.alive, games: 1 });
        Ok(out)
    }

    pub(super) fn release(&mut self, id: u64) {
        if let Some(l) = self.live.iter_mut().find(|l| l.id == id) {
            l.games = l.games.saturating_sub(1);
        }
    }

    fn prune(&mut self, in_use: &BTreeSet<String>) -> Vec<Closer> {
        let pinned = |l: &LiveServer| self.pinned.as_deref() == Some(l.key.as_str());
        let (mut keep, mut out) = (VecDeque::new(), Vec::new());
        for l in self.live.drain(..) {
            if l.games == 0 && !in_use.contains(&l.key) && !pinned(&l) {
                out.push((l.key, l.close));
            } else {
                keep.push_back(l);
            }
        }
        while keep.len() > self.cap {
            let Some(i) = keep.iter().position(|l| l.games == 0 && !pinned(l)) else { break };
            let l = keep.remove(i).unwrap();
            out.push((l.key, l.close));
        }
        self.live = keep;
        out
    }

    #[cfg(test)]
    fn live_keys(&self) -> Vec<&str> {
        self.live.iter().map(|l| l.key.as_str()).collect()
    }
}

pub(super) fn prune_servers(shared: &Shared) {
    let in_use = shared.league.lock().unwrap().model_opponents();
    let closers = shared.snapshots.lock().unwrap().prune(&in_use);
    for (key, close) in closers {
        close();
        uranium_store::elog!("[rl] inference server of {key} shut down: no opponent choice uses it (042)");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use uranium_runner::JobSource;
    use crate::league::AgentKind;
    use crate::run::testkit::{fake_servers, fake_servers_with, source, opponent_handle};

    #[test]
    fn snapshot_servers_shut_down_when_unused_042() {
        let (servers, opened, closed) = fake_servers(false);
        let src = source([0.0, 1.0, 0.0], "checkpoints/x/ckpt-9", 1, servers);
        let publish = |n: u64| {
            let mut l = src.shared.league.lock().unwrap();
            for _ in 0..n {
                l.on_update(AgentKind::Main);
            }
            let id = l.add_snapshot(AgentKind::Main);
            l.mark_published(&id);
            src.shared.snapshots.lock().unwrap().dirs.insert(id.clone(), PathBuf::from(format!("league/{n}/agents/main")));
            id
        };
        let a = src.next().unwrap();
        assert_eq!(a.opponent.as_deref(), Some("anchor@ckpt-9"));
        assert_eq!(opponent_handle(&a), "anchor/agents/main");
        let first = publish(250);
        let j1 = src.next().unwrap();
        assert_eq!(j1.opponent.as_deref(), Some(first.as_str()));
        let second = publish(500);
        prune_servers(&src.shared);
        assert_eq!(closed.load(Ordering::SeqCst), 0, "the first snapshot still has a game");
        let j2 = src.next().unwrap();
        assert_eq!(j2.opponent.as_deref(), Some(second.as_str()));
        src.requeue(j1);
        assert_eq!(closed.load(Ordering::SeqCst), 1, "first snapshot's server down after its game");
        src.requeue(a);
        src.requeue(j2);
        assert_eq!(closed.load(Ordering::SeqCst), 1, "the candidate's and the pinned anchor's servers stay");
        let keys = src.shared.snapshots.lock().unwrap().live_keys().iter().map(|k| k.to_string()).collect::<Vec<_>>();
        assert_eq!(keys, ["anchor@ckpt-9".to_string(), second.clone()]);
        assert_eq!(opened.lock().unwrap().len(), 3);
        let j3 = src.next().unwrap();
        assert_eq!(j3.opponent.as_deref(), Some(second.as_str()));
        assert_eq!(opened.lock().unwrap().len(), 3);
    }

    #[test]
    fn snapshot_pool_cap_spares_busy_and_pinned_servers() {
        let (servers, _, closed) = fake_servers(false);
        let dirs = (0..4).map(|i| (format!("s{i}"), PathBuf::from(format!("d{i}")))).collect();
        let mut pool = SnapshotPool::new(dirs, Some("s0".into()), servers);
        pool.cap = 2;
        let ids: Vec<u64> = (0..4).map(|i| pool.acquire(&format!("s{i}")).unwrap().1).collect();
        pool.release(ids[0]);
        pool.release(ids[1]);
        let all: BTreeSet<String> = (0..4).map(|i| format!("s{i}")).collect();
        let out = pool.prune(&all);
        assert_eq!(out.iter().map(|(k, _)| k.as_str()).collect::<Vec<_>>(), ["s1"]);
        for (_, c) in out {
            c();
        }
        assert_eq!(closed.load(Ordering::SeqCst), 1);
        assert_eq!(pool.live_keys(), ["s0", "s2", "s3"], "busy ones stay above the cap");
        pool.release(ids[2]);
        assert_eq!(pool.prune(&BTreeSet::new()).iter().map(|(k, _)| k.as_str()).collect::<Vec<_>>(), ["s2"], "unused and idle");
        assert!(pool.acquire("nope").is_err());
    }

    #[test]
    fn dead_and_failing_servers_are_not_reused_042() {
        let fail = Arc::new(AtomicBool::new(false));
        let (servers, opened, closed, alive) = fake_servers_with(fail.clone());
        let src = source([0.0, 0.0, 1.0], "checkpoints/x/ckpt-9", 4, servers);
        let first = src.next().unwrap();
        assert_eq!(first.opponent.as_deref(), Some("anchor@ckpt-9"));
        alive.lock().unwrap()[0].store(false, Ordering::SeqCst);
        for _ in 0..3 {
            let j = src.next().unwrap();
            assert_eq!(j.opponent.as_deref(), Some("uranium-main@r1"), "dead anchor server: self-play");
        }
        assert_eq!((closed.load(Ordering::SeqCst), opened.lock().unwrap().len()), (1, 1), "closed once, not restarted at once");
        src.shared.snapshots.lock().unwrap().retry_after = Duration::ZERO;
        let again = src.next().unwrap();
        assert_eq!(again.opponent.as_deref(), Some("anchor@ckpt-9"));
        assert_eq!(opened.lock().unwrap().len(), 2);
        let new_id = src.shared.games.lock().unwrap()[&again.spec.game_id].server.unwrap();
        src.requeue(first);
        assert_eq!(src.shared.snapshots.lock().unwrap().live.iter().find(|l| l.id == new_id).unwrap().games, 1);
        let mut pool = src.shared.snapshots.lock().unwrap();
        pool.retry_after = Duration::from_secs(300);
        pool.live.clear();
        fail.store(true, Ordering::SeqCst);
        assert!(pool.acquire("anchor@ckpt-9").is_err());
        fail.store(false, Ordering::SeqCst);
        let err = pool.acquire("anchor@ckpt-9").unwrap_err();
        assert!(format!("{err:#}").contains("failed"), "{err:#}");
        assert_eq!(opened.lock().unwrap().len(), 2, "no reload within the back-off");
    }
}
