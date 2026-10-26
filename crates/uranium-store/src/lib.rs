//! uranium-store: shared storage (061) — run locks and heartbeat, checkpoint push/pull and
//! retention, manifests, device records, disk warnings — and the training state files written
//! next to the weights (061-R10). No model or tensor code: the Rust training binary and the
//! Python trainer (via `uranium._core`) use the same implementation (001, AMB-000-4).

/// `eprintln!` that never panics (060-R6): a training child's stderr is a pipe to its watchdog,
/// and when the watchdog is killed hard the pipe closes. `eprintln!` would then panic on the next
/// line (EPIPE) and the child would die without saving; with PR_SET_PDEATHSIG = SIGINT it must
/// instead save, push and release. Lines written after the pipe closed are lost.
#[macro_export]
macro_rules! elog {
    ($($t:tt)*) => {{
        use std::io::Write as _;
        let _ = writeln!(std::io::stderr(), $($t)*);
    }};
}

pub mod ckpt;
pub mod disk;
pub mod storage;

/// First Ctrl+C (SIGINT) requested a clean stop: checkpoint, push, release (042-R10).
pub const STOP_CLEAN: usize = 1;

/// 042-R10 / 060-R6: installs the Ctrl+C handler of a `bc` / `rl` process. Called first thing,
/// before any pull, model load or lock, so an early SIGINT (Ctrl+C, a watchdog's forward, or
/// PR_SET_PDEATHSIG) is never lost: the run then stops at its next check and releases its lock
/// if it holds one. The first Ctrl+C asks for a clean stop; a second one exits with 130 at once.
pub fn install_stop_handler(tag: &'static str) -> anyhow::Result<std::sync::Arc<std::sync::atomic::AtomicUsize>> {
    use std::sync::atomic::Ordering;
    accept_ctrl_c();
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let _ = STOP.set(stop.clone());
    let s = stop.clone();
    ctrlc::set_handler(move || {
        if s.fetch_add(1, Ordering::SeqCst) >= STOP_CLEAN {
            elog!("[{tag}] second Ctrl+C: exiting without saving (042-R10)");
            std::process::exit(130);
        }
        elog!("[{tag}] Ctrl+C: stopping cleanly (checkpoint, push, release; 042-R10)");
    })?;
    Ok(stop)
}

/// 020-R15 ⑤: Ctrl+C handler of an evaluation worker. It only counts: the first press asks the
/// worker to finish its games and stop, the second to stop the games now (both handled by the
/// worker loop); a third exits with 130 at once in case the loop is stuck.
pub fn install_drain_handler(tag: &'static str) -> anyhow::Result<std::sync::Arc<std::sync::atomic::AtomicUsize>> {
    use std::sync::atomic::Ordering;
    accept_ctrl_c();
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let _ = STOP.set(stop.clone());
    let s = stop.clone();
    ctrlc::set_handler(move || match s.fetch_add(1, Ordering::SeqCst) {
        0 => elog!("[{tag}] Ctrl+C: finishing the games in flight, then stopping (Ctrl+C again: stop them now; 020-R15)"),
        1 => elog!("[{tag}] second Ctrl+C: stopping the games in flight now, they are played again later (020-R15)"),
        _ => {
            elog!("[{tag}] third Ctrl+C: exiting at once");
            std::process::exit(130);
        }
    })?;
    Ok(stop)
}

/// 042-R10 (2026-10-24): Windows lets a process ignore Ctrl+C (`SetConsoleCtrlHandler(NULL,
/// TRUE)`) and every process it starts inherits that. A worker console opened from such a parent
/// passed it down to the training child, which then never saw Ctrl+C (the run kept training, no
/// checkpoint). Clears the inherited flag so the handlers below run; a no-op elsewhere.
pub fn accept_ctrl_c() {
    #[cfg(windows)]
    // SAFETY: plain Win32 call without pointers (NULL handler = the process's Ctrl+C flag).
    unsafe {
        windows_sys::Win32::System::Console::SetConsoleCtrlHandler(None, 0);
    }
}

/// The process's Ctrl+C state once [`install_stop_handler`] ran (long pulls check it per file).
static STOP: std::sync::OnceLock<std::sync::Arc<std::sync::atomic::AtomicUsize>> = std::sync::OnceLock::new();

/// Whether this process was asked to stop (false before a handler is installed).
pub fn stop_requested() -> bool {
    STOP.get().is_some_and(|s| stopping(s))
}

/// The installed Ctrl+C state, `None` before [`install_stop_handler`] (the Python entrypoints
/// install it first, then hand it to the run loop).
pub fn stop_state() -> Option<std::sync::Arc<std::sync::atomic::AtomicUsize>> {
    STOP.get().cloned()
}

/// A long step (dataset pull) left off at Ctrl+C before any lock was taken; nothing to release.
/// The pull resumes where it stopped next time (061-R11-2).
#[derive(Debug)]
pub struct Interrupted(pub String);

impl std::fmt::Display for Interrupted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Interrupted {}

/// Whether a clean stop was requested.
pub fn stopping(stop: &std::sync::atomic::AtomicUsize) -> bool {
    stop.load(std::sync::atomic::Ordering::SeqCst) >= STOP_CLEAN
}
