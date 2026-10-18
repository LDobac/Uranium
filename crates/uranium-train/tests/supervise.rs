//! 060-R6 watchdog restart policy.

use std::time::Duration;

use uranium_train::supervise::{Decision, Ending, SuperviseArgs, decide};

fn args() -> SuperviseArgs {
    SuperviseArgs {
        max_restarts: 3,
        window: Duration::from_secs(3600),
        delay: Duration::from_secs(60),
        hang: Duration::from_secs(1800),
        progress_file: None,
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
