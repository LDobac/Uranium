#[macro_export]
macro_rules! elog {
    ($($t:tt)*) => {{
        use std::io::Write as _;
        let _ = writeln!(std::io::stderr(), $($t)*);
    }};
}

pub mod ckpt;
pub mod disk;
pub mod dotenv;
pub mod share_io;
pub mod storage;

pub const STOP_CLEAN: usize = 1;

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

pub fn accept_ctrl_c() {
    #[cfg(windows)]
    unsafe {
        windows_sys::Win32::System::Console::SetConsoleCtrlHandler(None, 0);
    }
}

static STOP: std::sync::OnceLock<std::sync::Arc<std::sync::atomic::AtomicUsize>> = std::sync::OnceLock::new();

pub fn stop_requested() -> bool {
    STOP.get().is_some_and(|s| stopping(s))
}

pub fn stop_state() -> Option<std::sync::Arc<std::sync::atomic::AtomicUsize>> {
    STOP.get().cloned()
}

#[derive(Debug)]
pub struct Interrupted(pub String);

impl std::fmt::Display for Interrupted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Interrupted {}

pub fn stopping(stop: &std::sync::atomic::AtomicUsize) -> bool {
    stop.load(std::sync::atomic::Ordering::SeqCst) >= STOP_CLEAN
}
