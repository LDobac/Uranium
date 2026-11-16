use std::time::Duration;
use anyhow::Result;
use super::lock::LockInfo;
use super::yields::YieldRequest;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LockView {
    Free,
    Held(LockInfo),
    Stale(LockInfo),
    Unknown(String),
}

#[derive(Debug)]
pub struct ContentObserver<T> {
    seen: std::collections::HashMap<String, (T, std::time::Instant, std::time::Instant)>,
}

impl<T> Default for ContentObserver<T> {
    fn default() -> Self {
        Self { seen: std::collections::HashMap::new() }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Watched<T> {
    Absent,
    Fresh(T),
    Expired(T),
    Unknown(String),
}

impl<T: Clone + PartialEq> ContentObserver<T> {
    pub fn watch(&mut self, run_id: &str, read: &Result<Option<T>>, now: std::time::Instant, limit: Duration) -> Watched<T> {
        let v = match read {
            Ok(None) => {
                self.seen.remove(run_id);
                return Watched::Absent;
            }
            Err(e) => {
                self.seen.remove(run_id);
                return Watched::Unknown(format!("{e:#}"));
            }
            Ok(Some(v)) => v,
        };
        let continued = self.seen.get(run_id).is_some_and(|(x, _, last)| x == v && now.saturating_duration_since(*last) <= limit / 2);
        let first = if continued { self.seen[run_id].1 } else { now };
        self.seen.insert(run_id.to_string(), (v.clone(), first, now));
        if now.saturating_duration_since(first) >= limit { Watched::Expired(v.clone()) } else { Watched::Fresh(v.clone()) }
    }

    pub fn reset(&mut self) {
        self.seen.clear();
    }
}

pub type LockObserver = ContentObserver<LockInfo>;

pub type YieldObserver = ContentObserver<YieldRequest>;

impl ContentObserver<LockInfo> {
    pub fn observe(&mut self, run_id: &str, read: &Result<Option<LockInfo>>, now: std::time::Instant, stale_after: Duration) -> LockView {
        match self.watch(run_id, read, now, stale_after) {
            Watched::Absent => LockView::Free,
            Watched::Fresh(l) => LockView::Held(l),
            Watched::Expired(l) => LockView::Stale(l),
            Watched::Unknown(e) => LockView::Unknown(e),
        }
    }
}
