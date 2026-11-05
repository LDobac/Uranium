//! 060-R6 watchdog restart policy.

use std::time::Duration;

use uranium_train::supervise::{Decision, EXIT_DEVICE_FATAL, Ending, Liveness, SuperviseArgs, decide, fatal_match};

fn args() -> SuperviseArgs {
    SuperviseArgs {
        max_restarts: 3,
        window: Duration::from_secs(3600),
        delay: Duration::from_secs(60),
        hang: Duration::from_secs(1800),
        progress_file: None,
        fatal_patterns: vec![],
        log: None,
        args: vec!["bc".into()],
    }
}

#[test]
fn restart_policy_060_r6() {
    let a = args();
    // Clean ends are final: done / step limit / Ctrl+C clean stop / lock lost (0), second Ctrl+C (130).
    assert_eq!(decide(Ending::Exited(Some(0)), false, 0, &a), Decision::Stop);
    assert_eq!(decide(Ending::Exited(Some(130)), false, 0, &a), Decision::Stop);
    // Abnormal ends restart with a doubling delay, capped at 10 minutes.
    assert_eq!(decide(Ending::Exited(Some(101)), false, 0, &a), Decision::Restart(Duration::from_secs(60)));
    assert_eq!(decide(Ending::Exited(None), false, 1, &a), Decision::Restart(Duration::from_secs(120)));
    assert_eq!(decide(Ending::Hung, false, 2, &a), Decision::Restart(Duration::from_secs(240)));
    let mut big = args();
    big.max_restarts = 20;
    assert_eq!(decide(Ending::Hung, false, 9, &big), Decision::Restart(Duration::from_secs(600)));
    // Too many restarts in the window: give up.
    assert_eq!(decide(Ending::Exited(Some(1)), false, 3, &a), Decision::GiveUp);
    // After the supervisor got Ctrl+C nothing restarts.
    assert_eq!(decide(Ending::Exited(Some(1)), true, 0, &a), Decision::Stop);
    assert_eq!(decide(Ending::Hung, true, 0, &a), Decision::Stop);
}

#[test]
fn lock_busy_and_yield_are_not_restarted_060_r6() {
    let a = args();
    // 75: the lock could not be taken (061-R5); 76: yielded (061-R14).
    assert_eq!(decide(Ending::Exited(Some(75)), false, 0, &a), Decision::Stop);
    assert_eq!(decide(Ending::Exited(Some(76)), false, 0, &a), Decision::Stop);
}

/// 060-R6 (H-4): on Unix the training child gets its own process group, so a terminal Ctrl+C
/// reaches only the watchdog, which passes it on once.
#[cfg(unix)]
#[test]
fn child_runs_in_its_own_process_group_060_r6() {
    let mut child = uranium_train::supervise::child_command(std::path::Path::new("/bin/sleep")).arg("5").spawn().unwrap();
    let pid = child.id() as libc::pid_t;
    // SAFETY: plain syscalls.
    let (theirs, mine) = unsafe { (libc::getpgid(pid), libc::getpgrp()) };
    child.kill().ok();
    child.wait().ok();
    assert_eq!(theirs, pid, "the child leads its own group");
    assert_ne!(theirs, mine);
}

/// 060-R6 (AMB-042-20): a fatal device error exit (042-R20, 70) and a fatal stderr line restart
/// like a crash, within the same restart limit.
#[test]
fn fatal_endings_restart_like_a_crash_060_r6() {
    let a = args();
    assert_eq!(EXIT_DEVICE_FATAL, 70);
    assert_eq!(decide(Ending::Exited(Some(EXIT_DEVICE_FATAL)), false, 0, &a), Decision::Restart(Duration::from_secs(60)));
    assert_eq!(decide(Ending::Fatal, false, 1, &a), Decision::Restart(Duration::from_secs(120)));
    assert_eq!(decide(Ending::Fatal, false, 3, &a), Decision::GiveUp);
    assert_eq!(decide(Ending::Exited(Some(EXIT_DEVICE_FATAL)), false, 3, &a), Decision::GiveUp);
    assert_eq!(decide(Ending::Fatal, true, 0, &a), Decision::Stop);
}

/// 060-R6 (AMB-042-20): with a progress file only its changes are progress; stderr lines are
/// progress only without one (fake clock).
#[test]
fn progress_file_alone_is_progress_060_r6() {
    let t0 = std::time::Instant::now();
    let hang = Duration::from_secs(30 * 60);
    let at = |m: u64| t0 + Duration::from_secs(m * 60);
    let mtime = |s: u64| Some(std::time::UNIX_EPOCH + Duration::from_secs(s));
    // With a progress file: 40 minutes of error lines without a train.csv row is a hang.
    let mut l = Liveness::new(t0, Some(mtime(100)));
    for m in 1..40 {
        l.line(at(m));
        l.file(mtime(100), at(m));
    }
    assert!(!l.hung(at(29), hang) && l.hung(at(30), hang));
    // A changed modification time resets the clock; the same one does not.
    let mut l = Liveness::new(t0, Some(mtime(100)));
    l.file(mtime(160), at(20));
    assert!(!l.hung(at(49), hang) && l.hung(at(50), hang));
    l.file(mtime(160), at(45));
    assert!(l.hung(at(50), hang));
    // A file that does not exist yet: its first appearance is progress, a missing read is not.
    let mut l = Liveness::new(t0, Some(None));
    l.file(None, at(10));
    assert!(l.hung(at(30), hang));
    l.file(mtime(5), at(25));
    assert!(!l.hung(at(54), hang) && l.hung(at(55), hang));
    // Without a progress file every stderr line is progress (as before).
    let mut l = Liveness::new(t0, None);
    l.line(at(20));
    assert!(!l.hung(at(49), hang) && l.hung(at(50), hang));
}

/// 060-R6 `--fatal-pattern` (AMB-042-20): any of the patterns, as a regex search in the line.
#[test]
fn fatal_patterns_match_060_r6() {
    let p = vec![regex::Regex::new("illegal memory access").unwrap(), regex::Regex::new(r"^Sticky error").unwrap()];
    let line = r#"[infer] main@db6f05ce3992: AcceleratorError("CUDA error: an illegal memory access was encountered"#;
    assert_eq!(fatal_match(&p, line).map(|r| r.as_str()), Some("illegal memory access"));
    assert_eq!(fatal_match(&p, "Sticky error detected").map(|r| r.as_str()), Some(r"^Sticky error"));
    assert!(fatal_match(&p, "[rl] note: Sticky error is mentioned mid-line").is_none());
    assert!(fatal_match(&p, "[rl] checkpoint ckpt-7307566 Pushed").is_none());
    assert!(fatal_match(&[], line).is_none());
}

/// 060-R6 (AMB-042-20) with real children: a fatal stderr line kills the child at once; with a
/// progress file a child that only prints is killed as hung; without one it is not.
#[cfg(unix)]
#[test]
fn watched_children_fatal_line_and_progress_file_060_r6() {
    use std::process::Command;
    use std::time::Instant;
    use uranium_train::supervise::Watch;

    let w = Watch::without_ctrl_c("test", None).unwrap();
    let sh = |script: &str| {
        let mut c = Command::new("/bin/sh");
        c.args(["-c", script]);
        c
    };
    let poll = Duration::from_millis(50);
    let fatal = vec![regex::Regex::new("illegal memory access").unwrap()];
    let t = Instant::now();
    let e = w.watch_child(sh("echo 'step 1' >&2; echo 'CUDA error: an illegal memory access was encountered' >&2; exec sleep 30"), Duration::from_secs(600), None, &fatal, poll).unwrap();
    assert_eq!(e, Ending::Fatal);
    assert!(t.elapsed() < Duration::from_secs(10), "killed at the line, not after the sleep");
    let printer = "while true; do echo 'inference failed' >&2; sleep 0.05; done";
    let missing = std::env::temp_dir().join(format!("uranium-progress-{}.csv", std::process::id()));
    let _ = std::fs::remove_file(&missing);
    let t = Instant::now();
    let e = w.watch_child(sh(printer), Duration::from_millis(700), Some(&missing), &[], poll).unwrap();
    assert_eq!(e, Ending::Hung, "lines are not progress with a progress file");
    assert!(t.elapsed() < Duration::from_secs(10));
    let e = w.watch_child(sh("for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do echo x >&2; sleep 0.1; done"), Duration::from_millis(700), None, &[], poll).unwrap();
    assert_eq!(e, Ending::Exited(Some(0)), "lines are progress without a progress file");
}
