use std::collections::{HashMap, HashSet};
use std::io::BufWriter;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use anyhow::Result;
use uranium_store::elog;
use uranium_store::storage;
use crate::plan::{self, HostEntry};
use crate::{Dispatch, capabilities};
use super::EvalJob;
use super::source::LedgerSource;

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
    std::thread::sleep(Duration::from_secs(1));
    Ok(())
}

fn leaving_file(addr: &str) -> String {
    addr.replace([':', '/', '\\'], "_")
}

pub(super) fn read_leaving(job: &EvalJob) -> Option<HashSet<String>> {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchEnd {
    Done,
    Yielded,
    Stopped,
    Lost,
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
