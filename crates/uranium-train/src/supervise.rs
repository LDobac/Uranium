//! 060-R6 watchdog: `uranium-train supervise [options] -- <bc|rl args>` runs a training command
//! as a child and restarts the same run when it dies abnormally or stops making progress (GPU
//! hang, driver reset). A restarted run resumes from its last checkpoint by itself (061-R3: the
//! lock of this same host is taken back, an unpushed local checkpoint is pushed first).
//!
//! Not restarted: exit 0 (finished, `--steps`/`--max-*` reached, Ctrl+C clean stop, lock lost to
//! another PC), 130 (second Ctrl+C), 75 (the lock could not be taken, 061-R5) and 76 (yielded,
//! 061-R14), or anything after this supervisor got Ctrl+C. Restarts are limited per time window
//! and spaced with a growing delay.
//!
//! Progress and fatal lines (060-R6, AMB-042-20, 6th advisory): with a progress file only its
//! changes count as progress (a child that keeps logging errors is still hung); a stderr line
//! matching a `--fatal-pattern` kills the child and restarts it like a crash; exit 70 (fatal
//! device error, 042-R20) restarts like any abnormal exit.

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use anyhow::{Context, Result};
use regex::Regex;

pub struct SuperviseArgs {
    /// Restarts allowed within `window`.
    pub max_restarts: usize,
    pub window: Duration,
    /// Delay before the first restart; doubles per restart in the window, at most 10 minutes.
    pub delay: Duration,
    /// No progress for this long = hang: the child is killed and restarted.
    pub hang: Duration,
    /// 060-R6 (AMB-042-20): when given, only this file's modification counts as progress (RL:
    /// `<logs>/train.csv`); stderr lines then do not. Without it every stderr line does.
    pub progress_file: Option<PathBuf>,
    /// 060-R6 (AMB-042-20): a stderr line matching one of these kills the child, which is then
    /// restarted like a crash ([`Ending::Fatal`]).
    pub fatal_patterns: Vec<Regex>,
    /// Appends the child's stderr and the supervisor's notes here as well.
    pub log: Option<PathBuf>,
    pub args: Vec<String>,
}

/// How a child run ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ending {
    /// Exit code, `None` when killed by a signal.
    Exited(Option<i32>),
    /// Killed by the supervisor after `hang` without progress.
    Hung,
    /// Killed by the supervisor after a stderr line matched a `--fatal-pattern` (060-R6).
    Fatal,
}

/// Exit code of a child that hit a fatal device error (042-R20: `os._exit(70)` without a save);
/// restarted from LATEST like any abnormal exit (060-R6).
pub const EXIT_DEVICE_FATAL: i32 = 70;

/// What to do after a child ended (pure, for tests).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decision {
    Stop,
    Restart(Duration),
    GiveUp,
}

/// 060-R6 restart policy. `recent` = restarts already made within the window.
pub fn decide(ending: Ending, stopping: bool, recent: usize, a: &SuperviseArgs) -> Decision {
    if stopping {
        return Decision::Stop;
    }
    match ending {
        // 0 done / clean stop, 130 second Ctrl+C, 75 run held by another PC, 76 yielded (061-R14).
        // Everything else restarts within the limit: crashes, `Hung`, `Fatal` and
        // `EXIT_DEVICE_FATAL` (042-R20).
        Ending::Exited(Some(0 | 130 | 75 | 76)) => Decision::Stop,
        _ if recent >= a.max_restarts => Decision::GiveUp,
        _ => Decision::Restart((a.delay * 2u32.saturating_pow(recent as u32)).min(Duration::from_secs(600))),
    }
}

fn modified(p: &PathBuf) -> Option<SystemTime> {
    std::fs::metadata(p).and_then(|m| m.modified()).ok()
}

/// 060-R6 hang clock of one child (pure, for tests). With a progress file only its modification
/// time moving counts as progress (AMB-042-20: in the 6th advisory's CUDA fault the child kept
/// printing inference errors for 40 minutes without an update); without one, any stderr line.
#[derive(Clone, Debug)]
pub struct Liveness {
    last: Instant,
    by_file: bool,
    file_seen: Option<SystemTime>,
}

impl Liveness {
    /// `file`: `Some(modification time now)` when a progress file is watched (`Some(None)` = it
    /// does not exist yet), `None` without one.
    pub fn new(now: Instant, file: Option<Option<SystemTime>>) -> Self {
        Self { last: now, by_file: file.is_some(), file_seen: file.flatten() }
    }

    /// A stderr line: progress only without a progress file.
    pub fn line(&mut self, now: Instant) {
        if !self.by_file {
            self.last = now;
        }
    }

    /// The progress file's modification time as just read: progress when it changed.
    pub fn file(&mut self, m: Option<SystemTime>, now: Instant) {
        if m.is_some() && m != self.file_seen {
            self.file_seen = m;
            self.last = now;
        }
    }

    pub fn hung(&self, now: Instant, hang: Duration) -> bool {
        now.saturating_duration_since(self.last) >= hang
    }
}

/// 060-R6 `--fatal-pattern`: the first pattern `line` matches.
pub fn fatal_match<'a>(patterns: &'a [Regex], line: &str) -> Option<&'a Regex> {
    patterns.iter().find(|p| p.is_match(line))
}

/// A process that watches training children: Ctrl+C state, notes, and one child at a time.
pub struct Watch {
    pub stopping: Arc<AtomicBool>,
    child_pid: Arc<Mutex<Option<u32>>>,
    log: Option<Arc<Mutex<std::fs::File>>>,
    tag: &'static str,
}

impl Watch {
    /// Installs the Ctrl+C handler (once per process). It is remembered so a child that then
    /// exits is never restarted. Unix: the child runs in its own process group (see
    /// [`child_command`]), so a terminal Ctrl+C reaches only this process, which passes SIGINT on
    /// exactly once per Ctrl+C (060-R6, H-4). Windows: the console delivers Ctrl+C to the child
    /// as well (same console), so nothing is passed on; both first clear an inherited
    /// ignore-Ctrl+C flag ([`uranium_store::accept_ctrl_c`], 042-R10), so a watchdog started in
    /// the background on purpose also stops on its console's Ctrl+C.
    pub fn new(tag: &'static str, log: Option<&PathBuf>) -> Result<Self> {
        let w = Self::without_ctrl_c(tag, log)?;
        uranium_store::accept_ctrl_c();
        {
            let (stopping, child_pid) = (w.stopping.clone(), w.child_pid.clone());
            ctrlc::set_handler(move || {
                stopping.store(true, Ordering::SeqCst);
                #[cfg(unix)]
                if let Some(pid) = *child_pid.lock().unwrap() {
                    // SAFETY: plain syscall on the child's pid.
                    unsafe {
                        libc::kill(pid as libc::pid_t, libc::SIGINT);
                    }
                }
                #[cfg(not(unix))]
                let _ = &child_pid;
            })?;
        }
        Ok(w)
    }

    /// A watch without the Ctrl+C handler (tests; the handler can be installed once per process).
    pub fn without_ctrl_c(tag: &'static str, log: Option<&PathBuf>) -> Result<Self> {
        let log = match log {
            Some(p) => Some(Arc::new(Mutex::new(std::fs::OpenOptions::new().create(true).append(true).open(p).with_context(|| p.display().to_string())?))),
            None => None,
        };
        Ok(Self { stopping: Arc::new(AtomicBool::new(false)), child_pid: Arc::new(Mutex::new(None)), log, tag })
    }

    pub fn stopping(&self) -> bool {
        self.stopping.load(Ordering::SeqCst)
    }

    pub fn note(&self, msg: &str) {
        let line = format!("[{}] {} {msg}", self.tag, utc_time());
        eprintln!("{line}");
        if let Some(l) = &self.log {
            let _ = writeln!(l.lock().unwrap(), "{line}");
        }
    }

    /// Sleeps `d` unless Ctrl+C comes first (then returns false).
    pub fn wait(&self, d: Duration) -> bool {
        let until = Instant::now() + d;
        while Instant::now() < until {
            if self.stopping() {
                return false;
            }
            std::thread::sleep(Duration::from_millis(500));
        }
        !self.stopping()
    }

    /// Runs `URANIUM_PYTHON -m uranium <args>` (the PyTorch trainer, 060-R6, AMB-000-4) to its
    /// end, watched by [`Watch::watch_child`].
    pub fn run_child(&self, args: &[String], hang: Duration, progress_file: Option<&PathBuf>, fatal: &[Regex]) -> Result<Ending> {
        let python = python()?;
        self.note(&format!("start: {} -m uranium {}", python.display(), args.join(" ")));
        let mut cmd = child_command(&python);
        cmd.args(["-m", "uranium"]).args(args);
        self.watch_child(cmd, hang, progress_file, fatal, Duration::from_secs(2))
    }

    /// Runs `cmd` (its stderr passed through) to its end. A hang kills it ([`Ending::Hung`]): no
    /// progress for `hang` ([`Liveness`]: `progress_file` changes when given, else stderr lines).
    /// A stderr line matching `fatal` kills it too ([`Ending::Fatal`]). `poll`: how often the
    /// child is looked at.
    pub fn watch_child(&self, mut cmd: Command, hang: Duration, progress_file: Option<&PathBuf>, fatal: &[Regex], poll: Duration) -> Result<Ending> {
        let mut child = cmd.stderr(Stdio::piped()).spawn().context("starting the training command")?;
        *self.child_pid.lock().unwrap() = Some(child.id());
        let live = Arc::new(Mutex::new(Liveness::new(Instant::now(), progress_file.map(modified))));
        let fatal_line = Arc::new(Mutex::new(None::<String>));
        let reader = {
            let (live, log, fatal_line, patterns) = (live.clone(), self.log.clone(), fatal_line.clone(), fatal.to_vec());
            let err = child.stderr.take().unwrap();
            std::thread::spawn(move || {
                for line in BufReader::new(err).lines().map_while(|l| l.ok()) {
                    live.lock().unwrap().line(Instant::now());
                    eprintln!("{line}");
                    if let Some(l) = &log {
                        let _ = writeln!(l.lock().unwrap(), "{line}");
                    }
                    if let Some(p) = fatal_match(&patterns, &line) {
                        let mut f = fatal_line.lock().unwrap();
                        if f.is_none() {
                            *f = Some(format!("{line} (pattern {})", p.as_str()));
                        }
                    }
                }
            })
        };
        let ending = loop {
            if let Some(st) = child.try_wait()? {
                break Ending::Exited(st.code());
            }
            let fatal_seen = fatal_line.lock().unwrap().clone();
            if let Some(line) = fatal_seen
                && !self.stopping()
            {
                self.note(&format!("fatal stderr line: killing pid {} (060-R6): {line}", child.id()));
                let _ = child.kill();
                let _ = child.wait();
                break Ending::Fatal;
            }
            if let Some(p) = progress_file {
                live.lock().unwrap().file(modified(p), Instant::now());
            }
            if !self.stopping() && live.lock().unwrap().hung(Instant::now(), hang) {
                let what = if progress_file.is_some() { "no progress-file change" } else { "no stderr line" };
                self.note(&format!("{what} for {} min: killing pid {} (060-R6)", hang.as_secs() / 60, child.id()));
                let _ = child.kill();
                let _ = child.wait();
                break Ending::Hung;
            }
            std::thread::sleep(poll);
        };
        let _ = reader.join();
        *self.child_pid.lock().unwrap() = None;
        Ok(ending)
    }
}

/// The training child's command (060-R6, H-4). Linux / Unix: its own process group, so a
/// terminal Ctrl+C does not reach it twice (process group + forwarding: the second one would exit
/// 130 without saving). Linux: `PR_SET_PDEATHSIG` = SIGINT, so a watchdog killed hard (SIGKILL)
/// does not leave an orphan that keeps the run lock's heartbeat alive; the child then saves,
/// pushes and releases as on Ctrl+C. Its stderr pipe to the dead watchdog is closed by then, so
/// `bc` / `rl` log through `elog!`, which ignores write errors (`eprintln!` would panic on EPIPE
/// and skip the save); their output after that point is lost. The signal follows the spawning
/// *thread*: children are
/// started from the watchdog's main loop thread. Windows: unchanged (same console, Ctrl+C reaches
/// the child directly; closing the console ends both).
/// 001-R9: the interpreter of the Python training children (`URANIUM_PYTHON`, e.g. the venv's
/// python). No default: a worker must not train with whichever python is on the PATH.
pub fn python() -> Result<PathBuf> {
    match std::env::var_os("URANIUM_PYTHON").filter(|v| !v.is_empty()) {
        Some(p) => Ok(PathBuf::from(p)),
        None => anyhow::bail!("URANIUM_PYTHON is not set: put the training venv's python in .env (001-R9), e.g. URANIUM_PYTHON=/path/to/.venv/bin/python"),
    }
}

pub fn child_command(exe: &std::path::Path) -> Command {
    #[allow(unused_mut)]
    let mut cmd = Command::new(exe);
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
        #[cfg(target_os = "linux")]
        {
            let parent = std::process::id() as libc::pid_t;
            // SAFETY: only async-signal-safe calls (prctl, getppid, raise) between fork and exec.
            unsafe {
                cmd.pre_exec(move || {
                    if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGINT) != 0 {
                        return Err(std::io::Error::last_os_error());
                    }
                    // The watchdog died before prctl took effect: stop like on Ctrl+C.
                    if libc::getppid() != parent {
                        libc::raise(libc::SIGINT);
                    }
                    Ok(())
                });
            }
        }
    }
    cmd
}

pub fn run(a: SuperviseArgs) -> Result<i32> {
    let w = Watch::new("supervise", a.log.as_ref())?;
    let mut restarts: Vec<Instant> = Vec::new();
    loop {
        let ending = w.run_child(&a.args, a.hang, a.progress_file.as_ref(), &a.fatal_patterns)?;
        restarts.retain(|t| t.elapsed() < a.window);
        match decide(ending, w.stopping(), restarts.len(), &a) {
            Decision::Stop => {
                w.note(&format!("{ending:?}: not restarted"));
                return Ok(match ending {
                    Ending::Exited(Some(c)) => c,
                    _ => 0,
                });
            }
            Decision::GiveUp => {
                w.note(&format!("{ending:?}: {} restarts within {} min, giving up", restarts.len(), a.window.as_secs() / 60));
                return Ok(1);
            }
            Decision::Restart(wait) => {
                w.note(&format!("{ending:?}: restarting the same run in {} s ({} of {} in the window)", wait.as_secs(), restarts.len() + 1, a.max_restarts));
                if !w.wait(wait) {
                    w.note("Ctrl+C while waiting: not restarted");
                    return Ok(0);
                }
                restarts.push(Instant::now());
            }
        }
    }
}
/// UTC time of day as HH:MM:SSZ (log ordering).
fn utc_time() -> String {
    let s = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    format!("{:02}:{:02}:{:02}Z", (s / 3600) % 24, (s / 60) % 60, s % 60)
}
