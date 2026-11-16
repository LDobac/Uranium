use std::fs;
use std::path::{Path, PathBuf};

use uranium_data::bc_run::{resolve_init, run_checkpoint};
use uranium_store::storage::{PushOutcome, PushStage, Run, StorageConfig, ckpt_name, verify_dir};

struct Tmp(PathBuf);

impl Tmp {
    fn new(tag: &str) -> Self {
        let p = std::env::temp_dir().join(format!("uranium-bc-run-{tag}-{}", std::process::id()));
        fs::remove_dir_all(&p).ok();
        fs::create_dir_all(&p).unwrap();
        Self(p)
    }
}

impl Drop for Tmp {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).ok();
    }
}

fn cfg(t: &Tmp, host: &str) -> StorageConfig {
    StorageConfig::new(t.0.join("shared"), t.0.join(format!("cache-{host}")), host)
}

fn write_ckpt(run: &mut Run, step: u64) -> PathBuf {
    let d = run.stage_dir(step).unwrap();
    fs::write(d.join("model.bpk"), format!("weights {step}")).unwrap();
    fs::create_dir_all(d.join("config")).unwrap();
    fs::write(d.join("config").join("rl.toml"), "x = 1").unwrap();
    run.seal(step).unwrap()
}

fn latest_verified(run: &Run) -> Option<u64> {
    let step = run.latest().unwrap()?;
    verify_dir(&run.shared_dir().join(ckpt_name(step))).expect("LATEST names a complete checkpoint (061-AC2)");
    Some(step)
}

#[test]
fn init_needs_a_finished_run_061_r13() {
    let t = Tmp::new("h1-init");
    let root = t.0.clone();
    let run = root.join("s1");
    std::fs::create_dir_all(run.join("ckpt-5")).unwrap();
    std::fs::write(run.join("LATEST"), "ckpt-5\n").unwrap();
    let err = resolve_init(&run).unwrap_err();
    assert!(format!("{err:#}").contains("not finished"), "{err:#}");
    std::fs::write(run.join("DONE"), "ckpt-5\n").unwrap();
    assert_eq!(resolve_init(&run).unwrap(), run.join("ckpt-5"));
    assert_eq!(resolve_init(&run.join("ckpt-5")).unwrap(), run.join("ckpt-5"));
}

#[test]
fn same_step_is_never_written_twice_b8() {
    for share_has_it in [false, true] {
        let t = Tmp::new(&format!("b8-{share_has_it}"));
        let mut run = Run::acquire(cfg(&t, "pc-a"), "run1", false).unwrap();
        let mut sealed_at = None;
        let write = |d: &Path| -> anyhow::Result<()> {
            fs::write(d.join("model.bpk"), "weights 7")?;
            Ok(())
        };
        assert_eq!(run_checkpoint(&mut run, 5, &mut sealed_at, write).unwrap(), PushOutcome::Pushed);
        let lock = run.lock_path();
        let held = fs::read(&lock).unwrap();
        if share_has_it {
            write_ckpt(&mut run, 7);
            sealed_at = Some(7);
            assert_eq!(run.push_until(7, PushStage::Renamed).unwrap(), PushOutcome::Pushed);
        } else {
            fs::remove_file(&lock).unwrap();
            assert!(matches!(run_checkpoint(&mut run, 7, &mut sealed_at, write).unwrap(), PushOutcome::Deferred(_)));
            fs::write(&lock, &held).unwrap();
        }
        assert_eq!(sealed_at, Some(7));
        let id = uranium_store::storage::ckpt_id(&run.cache_dir().join(ckpt_name(7))).unwrap();
        let rewrite = |_: &Path| -> anyhow::Result<()> { panic!("step 7 written twice (B-8)") };
        assert_eq!(run_checkpoint(&mut run, 7, &mut sealed_at, rewrite).unwrap(), PushOutcome::Pushed, "share has it: {share_has_it}");
        assert_eq!(uranium_store::storage::ckpt_id(&run.cache_dir().join(ckpt_name(7))).unwrap(), id, "the first copy is kept");
        assert_eq!(latest_verified(&run), Some(7));
    }
}
