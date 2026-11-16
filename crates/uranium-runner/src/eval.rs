use std::path::{Path, PathBuf};
use std::sync::Arc;
use anyhow::{Context, Result, bail};
use serde::Deserialize;
use uranium_store::storage;
use crate::plan::{self, Plan};
use crate::{BRIDGE_NEVER_CONNECTED, JobOutcome};

mod dispatch;
mod ledger;
mod merge;
mod source;
mod worker;
#[cfg(test)]
mod testkit;

pub use ledger::{Rec, read_records, LedgerState, ledger_state, pending, Inflight, Flights};
pub use source::LedgerSource;
pub use dispatch::{run_hosts, mark_leaving, clear_leaving, DispatchEnd};
pub use worker::{WaitEnd, EvalWorker};
pub use merge::{status, session_lost};

pub const STAGE_BASE: u64 = 100_000;
pub const ATTEMPT_BASE: u64 = 10_000_000;

pub fn hostd_id(stage: usize, plan_id: u64, attempt: u32) -> u64 {
    attempt as u64 * ATTEMPT_BASE + (stage as u64 + 1) * STAGE_BASE + plan_id
}

pub fn split_id(id: u64) -> (u32, usize, u64) {
    ((id / ATTEMPT_BASE) as u32, ((id % ATTEMPT_BASE) / STAGE_BASE).saturating_sub(1) as usize, id % STAGE_BASE)
}

pub type Key = (usize, u64);

fn key_of(id: u64) -> Key {
    let (_, s, g) = split_id(id);
    (s, g)
}

pub fn retryable(outcome: &str, detail: Option<&str>) -> bool {
    outcome == JobOutcome::Stopped.as_str() || detail.is_some_and(|d| d.starts_with(BRIDGE_NEVER_CONNECTED) || d.starts_with(HOSTD_LOST))
}

pub const HOSTD_LOST: &str = "hostd connection lost";

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
struct JobFile {
    stages: Vec<PathBuf>,
    hosts: PathBuf,
    #[serde(default = "d_max_attempts")]
    max_attempts: u32,
}

fn d_max_attempts() -> u32 {
    3
}

pub struct Stage {
    pub name: String,
    pub plan: Plan,
}

pub struct EvalJob {
    pub dir: PathBuf,
    pub lease_id: String,
    pub stages: Vec<Stage>,
    pub max_attempts: Option<u32>,
    hosts_file: PathBuf,
}

impl EvalJob {
    pub fn load(dir: &Path) -> Result<Self> {
        let path = dir.join("job.toml");
        let text = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        let jf: JobFile = toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        if jf.stages.is_empty() {
            bail!("{}: no stages", path.display());
        }
        let mut stages = Vec::new();
        for p in &jf.stages {
            let plan = plan::load_plan(&dir.join(p))?;
            if stages.iter().any(|s: &Stage| s.name == plan.name) {
                bail!("{}: two stages named `{}`", path.display(), plan.name);
            }
            storage::check_name(&plan.name)?;
            let games: u64 = plan.matchups.iter().map(|m| m.games as u64).sum();
            if games >= STAGE_BASE {
                bail!("stage `{}` has {games} games, at most {} (020-R15 ⑦)", plan.name, STAGE_BASE - 1);
            }
            let mine = crate::plan_models(&plan);
            for s in &stages {
                let shared = crate::plan_models(&s.plan).iter().any(|k| mine.contains(k));
                if shared && (s.plan.surrender != plan.surrender || s.plan.surrender_chat != plan.surrender_chat || s.plan.zero_last_failed != plan.zero_last_failed) {
                    bail!("stages `{}` and `{}` share a model but differ in surrender / surrender_chat / zero_last_failed", s.name, plan.name);
                }
            }
            stages.push(Stage { name: plan.name.clone(), plan });
        }
        if stages.len() >= (ATTEMPT_BASE / STAGE_BASE) as usize {
            bail!("too many stages ({})", stages.len());
        }
        let name = dir.file_name().map(|n| n.to_string_lossy().to_string()).context("job folder has no name")?;
        let lease_id = format!("eval-{name}");
        storage::check_name(&lease_id)?;
        if jf.max_attempts == 0 {
            bail!("{}: max_attempts must be at least 1", path.display());
        }
        Ok(Self { dir: dir.to_path_buf(), lease_id, stages, max_attempts: Some(jf.max_attempts), hosts_file: dir.join(jf.hosts) })
    }

    pub fn keys(&self) -> Vec<Key> {
        let mut v = Vec::new();
        for (si, s) in self.stages.iter().enumerate() {
            let n: u64 = s.plan.matchups.iter().map(|m| m.games as u64).sum();
            v.extend((1..=n).map(|g| (si, g)));
        }
        v
    }

    pub fn stage_names(&self) -> Arc<Vec<String>> {
        Arc::new(self.stages.iter().map(|s| s.name.clone()).collect())
    }

    fn sub(&self, name: &str) -> PathBuf {
        self.dir.join(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::testkit::job_dir;

    #[test]
    fn hostd_ids_020_r15() {
        assert_eq!(hostd_id(0, 7, 0), 100_007);
        assert_eq!(hostd_id(3, 1999, 2), 20_401_999);
        assert_eq!(split_id(20_401_999), (2, 3, 1999));
        assert_eq!(split_id(hostd_id(0, 1, 0)), (0, 0, 1));
        assert_ne!(hostd_id(0, 5, 0), hostd_id(1, 5, 0));
        assert_ne!(hostd_id(0, 5, 0), hostd_id(0, 5, 1));
    }

    #[test]
    fn shared_models_need_the_same_session_settings() {
        let dir = job_dir("settings", 1);
        let m = "[[matchup]]\nmap='both'\ngames=1\np0={kind='model',checkpoint='c/r/ckpt-1'}\np1={kind='t0',race='zerg'}\n";
        std::fs::write(dir.join("plans/0.toml"), format!("name='a'\nzero_last_failed=true\n{m}")).unwrap();
        std::fs::write(dir.join("plans/1.toml"), format!("name='b'\n{m}")).unwrap();
        assert!(EvalJob::load(&dir).is_err());
        std::fs::write(dir.join("plans/1.toml"), format!("name='b'\nzero_last_failed=true\n{m}")).unwrap();
        assert!(EvalJob::load(&dir).is_ok());
        std::fs::remove_dir_all(&dir).ok();
    }
}
