use std::fs;
use std::io::Write;
use std::path::PathBuf;
use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use super::{StorageConfig, now_secs, share_subdir};
use super::manifest::write_atomic;
use super::run::ckpt_name;

pub fn done_path(cfg: &StorageConfig, run_id: &str) -> PathBuf {
    cfg.shared.join("checkpoints").join(run_id).join("DONE")
}

pub fn is_done(cfg: &StorageConfig, run_id: &str) -> Result<bool> {
    let p = done_path(cfg, run_id);
    match fs::metadata(&p) {
        Ok(m) => Ok(m.is_file()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(anyhow!(e).context(format!("reading {}", p.display()))),
    }
}

pub fn mark_done(cfg: &StorageConfig, run_id: &str, step: u64) -> Result<()> {
    let p = done_path(cfg, run_id);
    fs::create_dir_all(p.parent().unwrap())?;
    write_atomic(&p, format!("{}\n", ckpt_name(step)).as_bytes())
}

pub fn log_device(cfg: &StorageConfig, run_id: &str, line: serde_json::Value) {
    let dir = cfg.shared.join("runs").join(run_id);
    let mut v = line;
    if let Some(o) = v.as_object_mut() {
        o.insert("host".into(), serde_json::json!(cfg.host));
        o.insert("time".into(), serde_json::json!(now_secs()));
    }
    let res = fs::create_dir_all(&dir).and_then(|_| {
        let mut f = fs::OpenOptions::new().create(true).append(true).open(dir.join("devices.jsonl"))?;
        writeln!(f, "{v}")
    });
    if let Err(e) = res {
        crate::elog!("devices.jsonl not written ({e}); training continues (061-R11-3)");
    }
}

#[derive(Serialize, Debug, Clone, PartialEq, Eq)]
pub struct DeviceSpan {
    pub host: String,
    pub backend: Option<String>,
    pub precision: Option<String>,
    pub from: u64,
    pub to: u64,
    pub end: String,
}

pub fn device_spans(lines: &[serde_json::Value]) -> Vec<DeviceSpan> {
    struct S {
        span: DeviceSpan,
        last_ckpt: u64,
        stop: Option<(u64, String)>,
    }
    let s = |v: &serde_json::Value, k: &str| v.get(k).and_then(|x| x.as_str()).map(str::to_string);
    let mut sessions: Vec<S> = Vec::new();
    for v in lines {
        let (Some(event), Some(host)) = (s(v, "event"), s(v, "host")) else { continue };
        let step = v.get("step").and_then(|x| x.as_u64()).unwrap_or(0);
        if event == "start" {
            let span = DeviceSpan { host, backend: s(v, "backend"), precision: s(v, "precision"), from: step, to: step, end: "crash".into() };
            sessions.push(S { span, last_ckpt: step, stop: None });
            continue;
        }
        let Some(cur) = sessions.iter_mut().rev().find(|x| x.span.host == host) else { continue };
        if cur.stop.is_some() {
            continue;
        }
        match event.as_str() {
            "checkpoint" => cur.last_ckpt = cur.last_ckpt.max(step),
            "stop" => cur.stop = Some((step, s(v, "reason").unwrap_or_else(|| "stop".into()))),
            _ => {}
        }
    }
    let starts: Vec<u64> = sessions.iter().map(|x| x.span.from).collect();
    sessions
        .into_iter()
        .enumerate()
        .map(|(i, x)| {
            let mut span = x.span;
            let (end, why) = match x.stop {
                Some((_, r)) if r == "lock lost" => (x.last_ckpt, r),
                Some((step, r)) => (step, r),
                None => (x.last_ckpt, "crash".into()),
            };
            let mut to = end.max(span.from);
            if let Some(&next) = starts.get(i + 1)
                && next < to
            {
                to = next.max(span.from);
            }
            span.to = to;
            span.end = why;
            span
        })
        .collect()
}

pub fn read_devices(cfg: &StorageConfig, run_id: &str) -> Result<Vec<serde_json::Value>> {
    let p = cfg.shared.join("runs").join(run_id).join("devices.jsonl");
    let text = fs::read_to_string(&p).with_context(|| format!("reading {}", p.display()))?;
    Ok(text.lines().filter_map(|l| serde_json::from_str(l).ok()).collect())
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct WorkerStatus {
    pub host: String,
    pub priority: i32,
    pub backend: String,
    pub run: Option<String>,
    pub time: u64,
}

pub fn write_worker(cfg: &StorageConfig, priority: i32, backend: &str, run: Option<&str>) -> Result<()> {
    let dir = cfg.shared.join("workers");
    share_subdir(cfg, "workers")?;
    let s = WorkerStatus { host: cfg.host.clone(), priority, backend: backend.into(), run: run.map(str::to_string), time: now_secs() };
    write_atomic(&dir.join(format!("{}.json", cfg.host)), &serde_json::to_vec_pretty(&s)?)
}

pub fn read_worker(cfg: &StorageConfig, host: &str) -> Option<WorkerStatus> {
    serde_json::from_slice(&fs::read(cfg.shared.join("workers").join(format!("{host}.json"))).ok()?).ok()
}
