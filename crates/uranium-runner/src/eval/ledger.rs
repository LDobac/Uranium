use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use uranium_store::storage::{ContentObserver, Watched};
use crate::JobOutcome;
use super::{split_id, Key, key_of, retryable, EvalJob};

#[derive(Debug, Clone)]
pub struct Rec {
    pub id: u64,
    pub outcome: String,
    pub detail: Option<String>,
    pub line: serde_json::Value,
}

pub(super) fn read_dir_lines(dir: &Path) -> Result<Vec<serde_json::Value>> {
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

pub(super) fn unit_rejections_file(job: &EvalJob, host: &str) -> PathBuf {
    job.sub("results").join("unit_rejections").join(format!("{host}.jsonl"))
}

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

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct LedgerState {
    pub done: HashMap<Key, u64>,
    pub attempts: HashMap<Key, u32>,
    pub retried: usize,
    pub failures: HashMap<Key, u32>,
    pub gave_up: Vec<Key>,
}

pub(super) fn is_failure(outcome: &str, detail: Option<&str>) -> bool {
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

pub fn pending(keys: &[Key], st: &LedgerState, busy: &HashSet<Key>) -> Vec<Key> {
    keys.iter().copied().filter(|k| !st.done.contains_key(k) && !busy.contains(k)).collect()
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct Inflight {
    pub host: String,
    pub pid: u32,
    pub time: u64,
    pub games: Vec<u64>,
}

pub(super) fn read_inflight(job: &EvalJob) -> Result<Vec<Inflight>> {
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

pub struct Flights {
    pub busy: HashSet<Key>,
    pub attempts: HashMap<Key, u32>,
}

pub(super) fn flights(job: &EvalJob, me: &str, obs: &mut ContentObserver<Inflight>, stale_after: Duration) -> Result<Flights> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::hostd_id;
    use crate::eval::testkit::{rec, job_dir};

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

    #[test]
    fn results_skip_unit_rejection_files_031_r10() {
        let dir = job_dir("rej", 1);
        let job = EvalJob::load(&dir).unwrap();
        let file = unit_rejections_file(&job, "h");
        assert_eq!(file, dir.join("results").join("unit_rejections").join("h.jsonl"));
        std::fs::create_dir_all(file.parent().unwrap()).unwrap();
        std::fs::write(job.sub("results").join("h.jsonl"), "{\"game_id\": 1}\n").unwrap();
        std::fs::write(&file, "{\"game_id\": 1, \"unit\": 5}\n").unwrap();
        let lines = read_dir_lines(&job.sub("results")).unwrap();
        assert_eq!(lines, vec![serde_json::json!({"game_id": 1})]);
        let _ = std::fs::remove_dir_all(&dir);
    }

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
        let f = Inflight { host: "me".into(), pid: std::process::id(), time: 5, games: vec![hostd_id(1, 1, 5)] };
        std::fs::write(dir.join("inflight/me-live.json"), serde_json::to_vec(&f).unwrap()).unwrap();
        let mut obs = ContentObserver::default();
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
}
