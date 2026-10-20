//! 061 shared storage on temporary folders: lock contention and take-over (AC3), atomic push
//! with faults injected mid-push (AC2), an unreachable share (AC6), retention (R6), resumable
//! verified pulls (R11-1/2) and the local mode of the hosting PC (R12, AC7).

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use uranium_train::storage::{LockBusy, LockStatus, Manifest, PushOutcome, PushStage, Run, StorageConfig, check_name, ckpt_name, open_dataset, pull_tree, verify_dir};

struct Tmp(PathBuf);

impl Tmp {
    fn new(tag: &str) -> Self {
        let p = std::env::temp_dir().join(format!("uranium-061-{tag}-{}", std::process::id()));
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

/// Writes a small checkpoint for `step` through stage → seal.
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

/// Makes the lease look `age` old (the share compares file times).
fn age_lock(run_lock: &Path, age: Duration) {
    let f = fs::File::options().write(true).open(run_lock).unwrap();
    f.set_modified(SystemTime::now() - age).unwrap();
}

#[test]
fn lock_refuses_other_pc_and_needs_take_over_061_ac3() {
    let t = Tmp::new("lock");
    let mut a = Run::acquire(cfg(&t, "pc-a"), "run1", false).unwrap();
    let err = Run::acquire(cfg(&t, "pc-b"), "run1", false).err().expect("valid lock refuses another PC");
    assert!(format!("{err:#}").contains("locked by pc-a"), "{err:#}");
    assert!(err.downcast_ref::<LockBusy>().is_some(), "refusal is exit 75 (061-R5)");
    assert_eq!(a.heartbeat(), LockStatus::Held);

    // An old file time changes nothing: still refused with 75 without --take-over (061-AC3).
    age_lock(&a.lock_path(), Duration::from_secs(11 * 60));
    let err = Run::acquire(cfg(&t, "pc-b"), "run1", false).err().expect("expired or not, the lock needs --take-over");
    assert!(err.downcast_ref::<LockBusy>().is_some() && format!("{err:#}").contains("--take-over"), "{err:#}");
    // --take-over takes a fresh lock at once (061-R5 manual take-over, with a warning).
    age_lock(&a.lock_path(), Duration::ZERO);
    assert_eq!(a.heartbeat(), LockStatus::Held);
    let b = Run::acquire(cfg(&t, "pc-b"), "run1", true).unwrap();
    assert_eq!(b.previous_owner().unwrap().host, "pc-a");
    // The old owner notices on its next heartbeat (R11-4).
    assert!(matches!(a.heartbeat(), LockStatus::Lost(l) if l.host == "pc-b"));

    // Normal exit releases: another PC starts right away without --take-over (042-R10).
    b.release().unwrap();
    Run::acquire(cfg(&t, "pc-a"), "run1", false).unwrap();
}

#[test]
fn same_host_takes_its_lock_back_after_a_crash_061_r5() {
    let t = Tmp::new("crash");
    drop(Run::acquire(cfg(&t, "pc-a"), "run1", false).unwrap());
    let again = Run::acquire(cfg(&t, "pc-a"), "run1", false).unwrap();
    assert_eq!(again.previous_owner().unwrap().host, "pc-a");
}

#[test]
fn push_is_atomic_under_faults_and_recovers_061_ac2() {
    for stage in [PushStage::Copied, PushStage::Manifested, PushStage::Renamed] {
        let t = Tmp::new(&format!("atomic-{stage:?}"));
        let mut run = Run::acquire(cfg(&t, "pc-a"), "run1", false).unwrap();
        assert_eq!(run.begin().unwrap(), None, "new run");
        write_ckpt(&mut run, 10);
        assert_eq!(run.push(10).unwrap(), PushOutcome::Pushed);
        write_ckpt(&mut run, 20);
        assert_eq!(run.push_until(20, stage).unwrap(), PushOutcome::Pushed);
        // Crash here: LATEST still names the previous complete checkpoint.
        assert_eq!(latest_verified(&run), Some(10), "{stage:?}");
        drop(run);

        // Restart on the same PC: the unpushed checkpoint is pushed first and resumed (R11-5).
        let mut run = Run::acquire(cfg(&t, "pc-a"), "run1", false).unwrap();
        let r = run.begin().unwrap().unwrap();
        assert_eq!((r.step, r.recovered), (20, true), "{stage:?}");
        assert_eq!(latest_verified(&run), Some(20));
        assert!(run.unpushed().unwrap().is_empty());
        assert!(!run.shared_dir().join(format!("{}.tmp-pc-a", ckpt_name(20))).exists(), "stage folder cleaned");
    }
}

#[test]
fn other_pc_pulls_latest_into_its_cache_061_r3() {
    let t = Tmp::new("pull");
    let mut a = Run::acquire(cfg(&t, "pc-a"), "run1", false).unwrap();
    write_ckpt(&mut a, 5);
    a.push(5).unwrap();
    a.release().unwrap();

    let mut b = Run::acquire(cfg(&t, "pc-b"), "run1", false).unwrap();
    let r = b.begin().unwrap().unwrap();
    assert_eq!((r.step, r.recovered), (5, false));
    assert!(r.dir.starts_with(t.0.join("cache-pc-b")));
    assert_eq!(fs::read_to_string(r.dir.join("model.bpk")).unwrap(), "weights 5");
}

#[test]
fn unreachable_share_defers_and_take_over_stops_push_061_ac6() {
    let t = Tmp::new("outage");
    let mut a = Run::acquire(cfg(&t, "pc-a"), "run1", false).unwrap();
    write_ckpt(&mut a, 1);
    a.push(1).unwrap();
    write_ckpt(&mut a, 2);

    // Share gone: push defers, nothing is pruned, the checkpoint stays (R11-3).
    let shared = t.0.join("shared");
    let away = t.0.join("shared-away");
    fs::rename(&shared, &away).unwrap();
    assert!(matches!(a.push(2).unwrap(), PushOutcome::Deferred(_)));
    assert!(matches!(a.heartbeat(), LockStatus::Unreachable(_)));
    a.prune().unwrap();
    assert!(a.cache_dir().join(ckpt_name(2)).exists());

    // Back: the retry pushes it.
    fs::rename(&away, &shared).unwrap();
    assert_eq!(a.push_pending().unwrap(), PushOutcome::Pushed);
    assert_eq!(latest_verified(&a), Some(2));

    // Another outage during which pc-b takes the run over: pc-a must not push (R11-4).
    write_ckpt(&mut a, 3);
    age_lock(&a.lock_path(), Duration::from_secs(11 * 60));
    let _b = Run::acquire(cfg(&t, "pc-b"), "run1", true).unwrap();
    assert!(matches!(a.push(3).unwrap(), PushOutcome::LockLost(l) if l.host == "pc-b"));
    assert_eq!(latest_verified(&a), Some(2));
}

#[test]
fn retention_keeps_newest_promoted_and_league_061_r6() {
    let t = Tmp::new("retain");
    let mut run = Run::acquire(cfg(&t, "pc-a"), "run1", false).unwrap();
    for step in 1..=7 {
        write_ckpt(&mut run, step);
        run.push(step).unwrap();
    }
    run.promote(2, "gen1").unwrap();
    run.league_snapshot(3, "main-3").unwrap();
    run.prune().unwrap();
    let names = |d: &Path| {
        let mut v: Vec<String> = fs::read_dir(d).unwrap().flatten().map(|e| e.file_name().to_string_lossy().into_owned()).filter(|n| n.starts_with("ckpt-")).collect();
        v.sort();
        v
    };
    assert_eq!(names(&run.shared_dir()), ["ckpt-3", "ckpt-4", "ckpt-5", "ckpt-6", "ckpt-7"]);
    assert_eq!(names(&run.cache_dir()), ["ckpt-7"]);
    verify_dir(&run.shared_dir().join("promoted").join("gen1")).unwrap();
    verify_dir(&run.shared_dir().join("league").join("main-3")).unwrap();
    assert_eq!(latest_verified(&run), Some(7));
}

fn ckpt_names(d: &Path) -> Vec<String> {
    let mut v: Vec<String> = fs::read_dir(d).unwrap().flatten().map(|e| e.file_name().to_string_lossy().into_owned()).filter(|n| n.starts_with("ckpt-")).collect();
    v.sort();
    v
}

/// 061-R6 (G-D6): `keep_checkpoints` sets the share's retention; 0 keeps every checkpoint (the
/// core ablation), the cache is still cleaned.
#[test]
fn keep_checkpoints_sets_retention_and_zero_keeps_all_061_r6() {
    for (keep, want) in [(0usize, vec!["ckpt-1", "ckpt-2", "ckpt-3", "ckpt-4"]), (2, vec!["ckpt-3", "ckpt-4"])] {
        let t = Tmp::new(&format!("keep-{keep}"));
        let mut c = cfg(&t, "pc-a");
        c.keep_latest = keep;
        let mut run = Run::acquire(c, "run1", false).unwrap();
        for step in 1..=4 {
            write_ckpt(&mut run, step);
            run.push(step).unwrap();
        }
        assert_eq!(run.prune().unwrap(), 0);
        assert_eq!(ckpt_names(&run.shared_dir()), want, "keep {keep}");
        assert_eq!(ckpt_names(&run.cache_dir()), ["ckpt-4"]);
    }
}

/// B-7 / 061-R6: a delete that fails (an SMB error) is logged and counted; the other deletes
/// still happen and the next prune retries it. The run is never ended by it.
#[test]
fn failed_prune_delete_is_a_warning_061_r6_b7() {
    let t = Tmp::new("prune-fail");
    let mut c = cfg(&t, "pc-a");
    c.keep_latest = 1;
    let mut run = Run::acquire(c, "run1", false).unwrap();
    for step in 1..=3 {
        write_ckpt(&mut run, step);
        run.push(step).unwrap();
    }
    // Deleting ckpt-1 anywhere fails; everything else goes through.
    run.set_remove_dir(|p| {
        if p.file_name().is_some_and(|n| n == "ckpt-1") { Err(std::io::Error::other("injected SMB delete failure")) } else { fs::remove_dir_all(p) }
    });
    assert_eq!(run.prune().unwrap(), 2, "ckpt-1 on the share and in the cache");
    assert_eq!(ckpt_names(&run.shared_dir()), ["ckpt-1", "ckpt-3"]);
    assert_eq!(ckpt_names(&run.cache_dir()), ["ckpt-1", "ckpt-3"]);
    // The next prune (delete working again) catches up.
    run.set_remove_dir(|p| fs::remove_dir_all(p));
    assert_eq!(run.prune().unwrap(), 0);
    assert_eq!(ckpt_names(&run.shared_dir()), ["ckpt-3"]);
    assert_eq!(latest_verified(&run), Some(3));
}

/// B-8: a checkpoint whose push was deferred (share unreachable) right before Ctrl+C is not
/// written a second time at the same step: the final save pushes the first copy, also when the
/// share already holds it from a push that failed after the rename (a second, different copy
/// would be refused there forever).
#[test]
fn same_step_is_never_written_twice_b8() {
    use uranium_train::bc_run::run_checkpoint;
    for share_has_it in [false, true] {
        let t = Tmp::new(&format!("b8-{share_has_it}"));
        let mut run = Run::acquire(cfg(&t, "pc-a"), "run1", false).unwrap();
        let mut sealed_at = None;
        let write = |d: &Path| -> anyhow::Result<()> {
            fs::write(d.join("model.bpk"), "weights 7")?;
            Ok(())
        };
        assert_eq!(run_checkpoint(&mut run, 5, &mut sealed_at, write).unwrap(), PushOutcome::Pushed);
        // Time checkpoint at step 7.
        let lock = run.lock_path();
        let held = fs::read(&lock).unwrap();
        if share_has_it {
            // The push got as far as the rename (LATEST not written), then the share went away.
            write_ckpt(&mut run, 7);
            sealed_at = Some(7);
            assert_eq!(run.push_until(7, PushStage::Renamed).unwrap(), PushOutcome::Pushed);
        } else {
            fs::remove_file(&lock).unwrap();
            assert!(matches!(run_checkpoint(&mut run, 7, &mut sealed_at, write).unwrap(), PushOutcome::Deferred(_)));
            fs::write(&lock, &held).unwrap();
        }
        assert_eq!(sealed_at, Some(7));
        let id = uranium_train::storage::ckpt_id(&run.cache_dir().join(ckpt_name(7))).unwrap();
        // Ctrl+C: the final save at the same step.
        let rewrite = |_: &Path| -> anyhow::Result<()> { panic!("step 7 written twice (B-8)") };
        assert_eq!(run_checkpoint(&mut run, 7, &mut sealed_at, rewrite).unwrap(), PushOutcome::Pushed, "share has it: {share_has_it}");
        assert_eq!(uranium_train::storage::ckpt_id(&run.cache_dir().join(ckpt_name(7))).unwrap(), id, "the first copy is kept");
        assert_eq!(latest_verified(&run), Some(7));
    }
}

#[test]
fn pull_skips_done_files_and_rejects_corruption_061_r11() {
    let t = Tmp::new("pulltree");
    let src = t.0.join("src");
    fs::create_dir_all(src.join("sub")).unwrap();
    fs::write(src.join("a.bin"), vec![1u8; 4096]).unwrap();
    fs::write(src.join("sub").join("b.bin"), vec![2u8; 100]).unwrap();
    Manifest::build(&src).unwrap().write(&src).unwrap();

    // An interrupted earlier pull: one good file, one leftover .part.
    let dst = t.0.join("dst");
    fs::create_dir_all(dst.join("sub")).unwrap();
    fs::copy(src.join("a.bin"), dst.join("a.bin")).unwrap();
    fs::write(dst.join("sub").join("b.bin.part"), b"half").unwrap();
    pull_tree(&src, &dst).unwrap();
    verify_dir(&dst).unwrap();
    assert!(!dst.join("sub").join("b.bin.part").exists());

    // A source file that no longer matches its manifest is refused.
    fs::write(src.join("sub").join("b.bin"), b"tampered").unwrap();
    fs::remove_dir_all(&dst).unwrap();
    assert!(pull_tree(&src, &dst).is_err());
}

/// 061-R4 push check (R-5): the read-back that bypasses this client's cache gives the same
/// result as the manifest for good files and refuses a changed one.
#[test]
fn push_check_reads_the_share_061_r4() {
    let t = Tmp::new("onserver");
    let d = t.0.join("ckpt");
    fs::create_dir_all(d.join("sub")).unwrap();
    fs::write(d.join("model.bin"), (0..(5 << 20) + 7).map(|i| (i % 253) as u8).collect::<Vec<u8>>()).unwrap();
    fs::write(d.join("sub").join("empty.bin"), b"").unwrap();
    let m = Manifest::build(&d).unwrap();
    m.verify_on_server(&d).unwrap();
    fs::write(d.join("model.bin"), vec![0u8; 4096]).unwrap();
    assert!(m.verify_on_server(&d).is_err());
}

#[test]
fn local_mode_reads_in_place_and_verifies_061_ac7() {
    let t = Tmp::new("local");
    let mut c = cfg(&t, "pc-a");
    c.shared_local = true;
    let ds = c.shared.join("datasets").join("shards").join("s1");
    fs::create_dir_all(&ds).unwrap();
    fs::write(ds.join("shard_0.bin.zst"), b"data").unwrap();
    Manifest::build(&ds).unwrap().write(&ds).unwrap();
    assert_eq!(open_dataset(&c, Path::new("shards/s1")).unwrap(), ds);
    assert!(!c.cache.join("datasets").exists(), "no cache copy (061-R12)");
    fs::write(ds.join("shard_0.bin.zst"), b"tampered").unwrap();
    assert!(open_dataset(&c, Path::new("shards/s1")).is_err(), "tampered shard refused");

    // Checkpoints: sealing publishes on the share itself.
    let mut run = Run::acquire(c, "run1", false).unwrap();
    write_ckpt(&mut run, 4);
    assert_eq!(latest_verified(&run), Some(4));
    assert_eq!(run.push(4).unwrap(), PushOutcome::Pushed);
    let r = run.begin().unwrap().unwrap();
    assert_eq!(r.dir, run.shared_dir().join(ckpt_name(4)));
}

#[test]
fn names_avoid_smb_forbidden_characters_061_r9() {
    for bad in ["a:b", "a/b", "a\\b", "a?", "a*", "a|b", "<a>", "\"a\"", "", ".hidden"] {
        assert!(check_name(bad).is_err(), "{bad:?}");
    }
    check_name("main-exploiter_2026-10-17").unwrap();
}

/// 061-R11-5 / R14 ③: an unpushed checkpoint left from before another PC pushed its own is never
/// recovered or pushed; one that continues the current `LATEST` is.
#[test]
fn unpushed_checkpoint_of_an_old_lineage_is_not_recovered_061_r14() {
    let t = Tmp::new("lineage");
    let mut a = Run::acquire(cfg(&t, "pc-a"), "run1", false).unwrap();
    assert_eq!(a.begin().unwrap(), None);
    write_ckpt(&mut a, 1);
    assert_eq!(a.push(1).unwrap(), PushOutcome::Pushed);
    write_ckpt(&mut a, 7); // never pushed
    drop(a); // crash

    // pc-b takes the run over and pushes its own continuation of ckpt-1, then crashes.
    let mut b = Run::acquire(cfg(&t, "pc-b"), "run1", true).unwrap();
    assert_eq!(b.begin().unwrap().unwrap().step, 1);
    write_ckpt(&mut b, 5);
    assert_eq!(b.push(5).unwrap(), PushOutcome::Pushed);
    drop(b);

    // pc-a takes it back, resumes from LATEST (5) and crashes before any checkpoint.
    let mut a = Run::acquire(cfg(&t, "pc-a"), "run1", true).unwrap();
    let r = a.begin().unwrap().unwrap();
    assert_eq!((r.step, r.recovered), (5, false));
    drop(a);
    // Last owner is pc-a and its cache has ckpt-7 > LATEST, but ckpt-7 does not descend from 5.
    let mut a = Run::acquire(cfg(&t, "pc-a"), "run1", false).unwrap();
    let r = a.begin().unwrap().unwrap();
    assert_eq!((r.step, r.recovered), (5, false));
    assert!(a.unpushed().unwrap().is_empty(), "the old lineage is never pushed");
    // A checkpoint of this lineage is recovered after the next crash.
    write_ckpt(&mut a, 9);
    drop(a);
    let mut a = Run::acquire(cfg(&t, "pc-a"), "run1", false).unwrap();
    let r = a.begin().unwrap().unwrap();
    assert_eq!((r.step, r.recovered), (9, true));
    assert_eq!(latest_verified(&a), Some(9));
}

/// 061-R15: device spans from `devices.jsonl` lines, crashes and lost locks ending at the last
/// checkpoint, a smaller next start cutting the previous session.
#[test]
fn device_spans_follow_the_line_format_061_r15() {
    use serde_json::json;
    use uranium_train::storage::device_spans;
    let start = |h: &str, s: u64, b: &str| json!({ "event": "start", "host": h, "step": s, "time": 1, "backend": b, "precision": "f32", "core": "Lstm", "stage": "S1" });
    let ckpt = |h: &str, s: u64| json!({ "event": "checkpoint", "host": h, "step": s, "time": 1 });
    let stop = |h: &str, s: u64, r: &str| json!({ "event": "stop", "host": h, "step": s, "time": 1, "reason": r });
    let lines = vec![
        start("a", 0, "vulkan"),
        ckpt("a", 10),
        ckpt("a", 20), // then a crash: no stop line
        start("b", 20, "cuda"),
        ckpt("b", 30),
        stop("b", 35, "ctrl-c"),
        start("a", 35, "vulkan"),
        ckpt("a", 50),
        start("b", 50, "cuda"), // took the lock over
        stop("a", 55, "lock lost"),
        stop("b", 60, "yield"),
        start("a", 60, "vulkan"),
        ckpt("a", 70),
        stop("a", 75, "ctrl-c"), // final push deferred: the next start is earlier
        start("b", 70, "cuda"),
    ];
    let got: Vec<(String, Option<String>, u64, u64, String)> = device_spans(&lines).into_iter().map(|s| (s.host, s.backend, s.from, s.to, s.end)).collect();
    let want = [("a", "vulkan", 0, 20, "crash"), ("b", "cuda", 20, 35, "ctrl-c"), ("a", "vulkan", 35, 50, "lock lost"), ("b", "cuda", 50, 60, "yield"), ("a", "vulkan", 60, 70, "ctrl-c"), ("b", "cuda", 70, 70, "crash")];
    let want: Vec<_> = want.iter().map(|(h, b, f, t, e)| (h.to_string(), Some(b.to_string()), *f, *t, e.to_string())).collect();
    assert_eq!(got, want);
}
