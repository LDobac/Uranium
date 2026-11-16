use std::collections::{HashMap, HashSet};
use std::path::Path;
use anyhow::Result;
use uranium_store::storage::{self, StorageConfig};
use crate::JobOutcome;
use super::{Key, key_of, retryable, EvalJob};
use super::ledger::{Rec, read_dir_lines, read_records, LedgerState, ledger_state, read_inflight};
use super::dispatch::read_leaving;

pub fn status(job_dir: &Path, write_summaries: bool) -> Result<String> {
    use std::fmt::Write;
    let job = EvalJob::load(job_dir)?;
    let recs = read_records(&job)?;
    let st = ledger_state(&recs, job.max_attempts);
    let files = read_inflight(&job)?;
    let busy: HashSet<Key> = files.iter().flat_map(|f| f.games.iter().map(|&id| key_of(id))).collect();
    let mut out = String::new();
    writeln!(out, "# Evaluation job `{}`\n", job.dir.display())?;
    let cfg = StorageConfig::from_env().unwrap_or_else(|_| StorageConfig::new(job_dir, job_dir, "status"));
    match storage::peek_lock(&cfg, &job.lease_id) {
        Ok(Some(l)) => writeln!(out, "lease `{}`: {} (pid {})", job.lease_id, l.host, l.pid)?,
        Ok(None) => writeln!(out, "lease `{}`: free", job.lease_id)?,
        Err(e) => writeln!(out, "lease `{}`: unreadable ({e:#})", job.lease_id)?,
    }
    for f in &files {
        writeln!(out, "in flight on {}'s inference: {} games", f.host, f.games.len())?;
    }
    let leaving = read_leaving(&job).unwrap_or_default();
    if !leaving.is_empty() {
        let mut l: Vec<_> = leaving.into_iter().collect();
        l.sort();
        writeln!(out, "leaving hosts: {}", l.join(", "))?;
    }
    if !st.gave_up.is_empty() {
        writeln!(out, "given up after max_attempts: {} games", st.gave_up.len())?;
    }
    writeln!(out, "\n| stage | games | done | in flight | left | retried records |\n|---|---|---|---|---|---|")?;
    for (si, s) in job.stages.iter().enumerate() {
        let n: u64 = s.plan.matchups.iter().map(|m| m.games as u64).sum();
        let done = (1..=n).filter(|g| st.done.contains_key(&(si, *g))).count();
        let flying = (1..=n).filter(|g| !st.done.contains_key(&(si, *g)) && busy.contains(&(si, *g))).count();
        let retried = recs.iter().filter(|r| key_of(r.id).0 == si && retryable(&r.outcome, r.detail.as_deref())).count();
        writeln!(out, "| {} | {n} | {done} | {flying} | {} | {retried} |", s.name, n as usize - done - flying)?;
        if write_summaries && done > 0 {
            merge_stage(&job, si, &recs, &st)?;
        }
    }
    Ok(out)
}

pub fn session_lost(record: &serde_json::Value, ended_sessions: usize) -> bool {
    let bridges = if record["p1_kind"].as_str().is_some_and(|k| k.starts_with("oss:")) { 1 } else { 2 };
    record["outcome"].as_str() == Some(JobOutcome::Finished.as_str()) && ended_sessions < bridges
}

fn merge_stage(job: &EvalJob, si: usize, recs: &[Rec], st: &LedgerState) -> Result<()> {
    use std::fmt::Write as _;
    let s = &job.stages[si];
    let dir = job.sub(&s.name);
    std::fs::create_dir_all(&dir)?;
    let finals: HashMap<u64, u64> = st.done.iter().filter(|(k, _)| k.0 == si).map(|(k, id)| (*id, k.1)).collect();
    let gave_up: HashSet<u64> = st.gave_up.iter().filter(|k| k.0 == si).map(|k| st.done[k]).collect();
    let sessions: Vec<serde_json::Value> = read_dir_lines(&job.sub("results"))?.into_iter().filter(|v| v["game_id"].as_u64().is_some_and(|id| finals.contains_key(&id))).collect();
    let mut ended: HashMap<u64, usize> = HashMap::new();
    for v in &sessions {
        if v["error"].is_null() {
            *ended.entry(v["game_id"].as_u64().unwrap()).or_default() += 1;
        }
    }
    let mut lost: HashSet<u64> = HashSet::new();
    let mut games: Vec<(u64, serde_json::Value)> = Vec::new();
    let mut backends: HashMap<String, usize> = HashMap::new();
    for r in recs.iter().filter(|r| finals.contains_key(&r.id)) {
        let mut v = r.line.clone();
        v["hostd_id"] = r.id.into();
        v["game_id"] = finals[&r.id].into();
        if gave_up.contains(&r.id) {
            v["gave_up"] = true.into();
        }
        if session_lost(&v, ended.get(&r.id).copied().unwrap_or(0)) {
            v["hostd_outcome"] = v["outcome"].clone();
            v["outcome"] = JobOutcome::Aborted.as_str().into();
            v["no_result"] = "session lost".into();
            lost.insert(r.id);
        }
        games.push((finals[&r.id], v));
    }
    games.sort_by_key(|(g, _)| *g);
    games.dedup_by_key(|(g, _)| *g);
    for (_, v) in &games {
        let b = format!("{} ({})", v["backend"].as_str().unwrap_or("?"), v["infer_host"].as_str().unwrap_or("?"));
        *backends.entry(b).or_default() += 1;
    }
    let mut text = String::new();
    for (_, v) in &games {
        writeln!(text, "{v}")?;
    }
    std::fs::write(dir.join("games.jsonl"), text)?;
    let mut text = String::new();
    for mut v in sessions {
        let id = v["game_id"].as_u64().unwrap();
        if !gave_up.contains(&id) && !lost.contains(&id) {
            v["hostd_id"] = id.into();
            v["game_id"] = finals[&id].into();
            writeln!(text, "{v}")?;
        }
    }
    std::fs::write(dir.join("results.jsonl"), text)?;
    crate::report::write(&dir, &s.plan)?;
    let mut table = String::new();
    if !gave_up.is_empty() {
        writeln!(table, "\nGiven up after max_attempts failed attempts (no result, 020-R15 ④): {} games", gave_up.len())?;
    }
    if !lost.is_empty() {
        writeln!(table, "\nA session lost mid-game (no result, 020-R15 ⑨): {} games", lost.len())?;
    }
    table.push_str("\n## Inference backend (020-R15 ⑧)\n\n| backend (host) | games |\n|---|---|\n");
    let mut rows: Vec<_> = backends.into_iter().collect();
    rows.sort();
    for (b, n) in rows {
        writeln!(table, "| {b} | {n} |")?;
    }
    let summary = dir.join("summary.md");
    let mut md = std::fs::read_to_string(&summary).unwrap_or_default();
    md.push_str(&table);
    std::fs::write(summary, md)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_lost_games_have_no_result_020_r15() {
        let g = |outcome: &str, p1: &str| serde_json::json!({"outcome": outcome, "p1_kind": p1});
        assert!(session_lost(&g("Finished", "t0"), 1));
        assert!(session_lost(&g("Finished", "model"), 0));
        assert!(!session_lost(&g("Finished", "t0"), 2));
        assert!(!session_lost(&g("Finished", "oss:iron"), 1));
        assert!(session_lost(&g("Finished", "oss:iron"), 0));
        assert!(!session_lost(&g("Timeout", "t0"), 1), "only games hostd called Finished");
    }
}
