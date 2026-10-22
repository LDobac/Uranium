//! Bridge log file: `URANIUM_SCR_LOG_DIR/uranium_bridge_scr.log` (default: the temp folder).

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

static FILE: Mutex<Option<File>> = Mutex::new(None);

pub fn log(msg: &str) {
    let mut guard = FILE.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        let dir = std::env::var_os("URANIUM_SCR_LOG_DIR").map(std::path::PathBuf::from).unwrap_or_else(std::env::temp_dir);
        std::fs::create_dir_all(&dir).ok();
        *guard = OpenOptions::new().create(true).append(true).open(dir.join("uranium_bridge_scr.log")).ok();
    }
    if let Some(f) = guard.as_mut() {
        let t = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0);
        let _ = writeln!(f, "[{t:.3}] {msg}");
        let _ = f.flush();
    }
}
