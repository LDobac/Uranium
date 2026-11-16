use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use anyhow::{Context, Result};
use uranium_data::data::ZEntry;
use uranium_env::scout::PseudoMode;
use uranium_store::ckpt::{self, TrainState};
use crate::config::RlConfig;
use crate::league::{AgentKind, League};
use crate::learner::RlLearners;
use uranium_store::storage::{PushOutcome, Run};
use super::{RlArgs, Shared};

pub(super) fn checkpoint_pseudo(dir: &Path) -> Option<PseudoMode> {
    let meta = dir.join(uranium_store::ckpt::AGENTS_DIR).join(AgentKind::Main.name()).join("meta.json");
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(meta).ok()?).ok()?;
    let rl = v.get("train")?.get("rl")?;
    match rl.get("reward").and_then(|r| r.get("pseudo")) {
        None => Some(PseudoMode::Z),
        Some(p) => serde_json::from_value(p.clone()).ok(),
    }
}

pub(super) fn check_pseudo(cfg: &RlConfig, z_table: Option<&Path>, resumed: Option<PseudoMode>) -> Result<()> {
    let mode = cfg.pseudo();
    if mode == PseudoMode::Scout
        && let Some(z) = z_table
    {
        anyhow::bail!("[reward] pseudo = \"scout\" uses no z: --z-table {} is not allowed with it (033-R11)", z.display());
    }
    if let Some(was) = resumed
        && was != mode
    {
        anyhow::bail!("[reward] pseudo is {mode:?} but the resumed checkpoint was trained with {was:?}: the build / units value heads would change meaning; start a new run instead (033-R11)");
    }
    Ok(())
}

pub(super) fn state_extra(shared: &Shared, trained_steps: u64) -> serde_json::Value {
    serde_json::json!({ "games": shared.games_done.load(Ordering::SeqCst), "trained_steps": trained_steps })
}

pub(super) fn resumed_trained_steps(extra: &serde_json::Value) -> u64 {
    extra.get("trained_steps").and_then(|v| v.as_u64()).unwrap_or(0)
}

pub(super) fn load_z(path: &Option<PathBuf>) -> Result<Vec<ZEntry>> {
    let Some(p) = path else { return Ok(vec![]) };
    let text = std::fs::read_to_string(p).with_context(|| format!("reading z table {}", p.display()))?;
    text.lines().filter(|l| !l.trim().is_empty()).map(|l| Ok(serde_json::from_str(l)?)).collect()
}

pub(super) fn save_checkpoint(run: &mut Run, step: u64, learners: &mut dyn RlLearners, cfg: &RlConfig, league: &League, args: &RlArgs, state_extra: serde_json::Value) -> Result<PushOutcome> {
    let dir = run.stage_dir(step)?;
    for a in AgentKind::ALL {
        let i = a.index();
        let train = serde_json::json!({ "rl": cfg, "algo": args.algo, "update": learners.version(i), "agent": a.name() });
        learners.save(i, &ckpt::agent_dir(&dir, a.name())?, &train)?;
    }
    ckpt::save_league(&dir, &league.to_bytes())?;
    let updates: Vec<u64> = AgentKind::ALL.iter().map(|a| learners.version(a.index())).collect();
    ckpt::save_state(&dir, &TrainState { kind: "rl".into(), step, update: updates[0], policy_version: updates[0], rng: vec![], extra: state_extra })?;
    let reward = PathBuf::from("config/reward.toml");
    let mut files: Vec<&Path> = vec![&args.config];
    if reward.is_file() {
        files.push(&reward);
    }
    ckpt::copy_configs(&dir, &files)?;
    run.seal(step)?;
    let out = run.push(step);
    check_storage(run);
    out
}

pub(super) fn check_storage(run: &Run) {
    uranium_store::disk::warn_storage("rl", &run.cfg);
    uranium_store::disk::spawn_replay_check(&run.cfg.shared, uranium_store::disk::REPLAY_BUDGET_BYTES);
}

pub(super) fn prune_logged(run: &mut Run) {
    match run.prune() {
        Ok(0) => {}
        Ok(n) => uranium_store::elog!("[rl] prune: {n} checkpoint folder(s) not deleted, left for the next prune (061-R6)"),
        Err(e) => uranium_store::elog!("[rl] WARNING: prune failed ({e:#}); training goes on (061-R6)"),
    }
}

pub(super) fn finish_run(run: Run, out: PushOutcome, steps_reached: bool, step: u64) -> Result<()> {
    match out {
        PushOutcome::Pushed => {
            if steps_reached {
                uranium_store::storage::mark_done(&run.cfg, &run.run_id, step)?;
                uranium_store::elog!("[rl] {} finished: DONE at {} (061-R13)", run.run_id, uranium_store::storage::ckpt_name(step));
            }
            run.release()
        }
        other => {
            uranium_store::elog!("[rl] push {other:?}: keeping the lock so the next start recovers it (061-R11-5/6)");
            Ok(())
        }
    }
}

pub fn snapshot_name(id: &str) -> String {
    id.trim_start_matches("uranium-").replace(['@', '/'], "-")
}

#[cfg(test)]
mod tests {
    use super::*;
    use uranium_store::storage::StorageConfig;

    #[test]
    fn done_on_max_agent_steps_042_r12() {
        let root = std::env::temp_dir().join(format!("uranium-rl-done-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let cfg = StorageConfig::new(root.join("shared"), root.join("cache"), "h");
        for (run_id, out, reached, done, released) in [
            ("rl-done", PushOutcome::Pushed, true, true, true),
            ("rl-ctrlc", PushOutcome::Pushed, false, false, true),
            ("rl-deferred", PushOutcome::Deferred("share".into()), true, false, false),
        ] {
            let run = Run::acquire(cfg.clone(), run_id, false).unwrap();
            let lock = run.lock_path();
            assert!(lock.is_file(), "{run_id}");
            finish_run(run, out, reached, 3_000_007).unwrap();
            assert_eq!(uranium_store::storage::is_done(&cfg, run_id).unwrap(), done, "{run_id}");
            assert_eq!(!lock.exists(), released, "{run_id}");
        }
        let p = uranium_store::storage::done_path(&cfg, "rl-done");
        assert_eq!(std::fs::read_to_string(p).unwrap().trim(), uranium_store::storage::ckpt_name(3_000_007));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn snapshot_names_are_smb_safe_061_r9() {
        let n = snapshot_name("uranium-main-exploiter@run1/4000");
        assert_eq!(n, "main-exploiter-run1-4000");
        uranium_store::storage::check_name(&n).unwrap();
    }

    #[test]
    fn trained_steps_resume_from_checkpoint_042_r2() {
        assert_eq!(resumed_trained_steps(&serde_json::json!({ "games": 3, "trained_steps": 12_345 })), 12_345);
        assert_eq!(resumed_trained_steps(&serde_json::json!({ "games": 3 })), 0);
        assert_eq!(resumed_trained_steps(&serde_json::Value::Null), 0);
    }

    #[test]
    fn pseudo_start_errors_033_r11() {
        let mut cfg = RlConfig::default_file();
        let z = Path::new("z.jsonl");
        check_pseudo(&cfg, Some(z), None).unwrap();
        check_pseudo(&cfg, None, Some(PseudoMode::Z)).unwrap();
        assert!(check_pseudo(&cfg, None, Some(PseudoMode::Scout)).is_err());
        cfg.reward.pseudo = PseudoMode::Scout;
        assert!(check_pseudo(&cfg, Some(z), None).is_err(), "scout + --z-table");
        check_pseudo(&cfg, None, None).unwrap();
        check_pseudo(&cfg, None, Some(PseudoMode::Scout)).unwrap();
        assert!(check_pseudo(&cfg, None, Some(PseudoMode::Z)).is_err(), "resume a z checkpoint in scout");
        let dir = std::env::temp_dir().join(format!("uranium-rl-pseudo-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let main = dir.join(uranium_store::ckpt::AGENTS_DIR).join("main");
        std::fs::create_dir_all(&main).unwrap();
        assert_eq!(checkpoint_pseudo(&dir), None, "no meta.json");
        let write = |rl: serde_json::Value| std::fs::write(main.join("meta.json"), serde_json::json!({ "train": { "rl": rl, "update": 3 } }).to_string()).unwrap();
        let mut old = serde_json::to_value(RlConfig::default_file()).unwrap();
        old.as_object_mut().unwrap().remove("reward");
        write(old);
        assert_eq!(checkpoint_pseudo(&dir), Some(PseudoMode::Z), "older checkpoint without [reward]");
        write(serde_json::to_value(&cfg).unwrap());
        assert_eq!(checkpoint_pseudo(&dir), Some(PseudoMode::Scout));
        std::fs::remove_dir_all(&dir).ok();
    }
}
