//! uranium-train: behaviour cloning (spec 041), training state and shared storage (061), RL and
//! the league (spec 042).

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

pub mod bc;
pub mod bc_eval;
pub mod bc_prof;
pub mod bc_run;
pub mod ckpt;
pub mod data;
pub mod disk;
pub mod rl;
pub mod split;
pub mod storage;
pub mod supervise;
pub mod worker;

/// First Ctrl+C (SIGINT) requested a clean stop: checkpoint, push, release (042-R10).
pub const STOP_CLEAN: usize = 1;

/// 042-R10 / 060-R6: installs the Ctrl+C handler of a `bc` / `rl` process. Called first thing,
/// before any pull, model load or lock, so an early SIGINT (Ctrl+C, a watchdog's forward, or
/// PR_SET_PDEATHSIG) is never lost: the run then stops at its next check and releases its lock
/// if it holds one. The first Ctrl+C asks for a clean stop; a second one exits with 130 at once.
pub fn install_stop_handler(tag: &'static str) -> anyhow::Result<std::sync::Arc<std::sync::atomic::AtomicUsize>> {
    use std::sync::atomic::Ordering;
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

/// The process's Ctrl+C state once [`install_stop_handler`] ran (long pulls check it per file).
static STOP: std::sync::OnceLock<std::sync::Arc<std::sync::atomic::AtomicUsize>> = std::sync::OnceLock::new();

/// Whether this process was asked to stop (false before a handler is installed).
pub fn stop_requested() -> bool {
    STOP.get().is_some_and(|s| stopping(s))
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
