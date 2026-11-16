use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use anyhow::{Context, Result};
use regex::Regex;

pub struct SuperviseArgs {
    pub max_restarts: usize,
    pub window: Duration,
    pub delay: Duration,
    pub hang: Duration,
    pub progress_file: Option<PathBuf>,
    pub fatal_patterns: Vec<Regex>,
    pub log: Option<PathBuf>,
    pub args: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ending {
    Exited(Option<i32>),
    Hung,
    Fatal,
}

pub const EXIT_DEVICE_FATAL: i32 = 70;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decision {
    Stop,
    Restart(Duration),
    GiveUp,
}

pub fn decide(ending: Ending, stopping: bool, recent: usize, a: &SuperviseArgs) -> Decision {
    if stopping {
        return Decision::Stop;
    }
    match ending {
        Ending::Exited(Some(0 | 130 | 75 | 76)) => Decision::Stop,
        _ if recent >= a.max_restarts => Decision::GiveUp,
        _ => Decision::Restart((a.delay * 2u32.saturating_pow(recent as u32)).min(Duration::from_secs(600))),
    }
}

fn modified(p: &PathBuf) -> Option<SystemTime> {
    std::fs::metadata(p).and_then(|m| m.modified()).ok()
}

#[derive(Clone, Debug)]
pub struct Liveness {
    last: Instant,
    by_file: bool,
    file_seen: Option<SystemTime>,
}

impl Liveness {
    pub fn new(now: Instant, file: Option<Option<SystemTime>>) -> Self {
        Self { last: now, by_file: file.is_some(), file_seen: file.flatten() }
    }

    pub fn line(&mut self, now: Instant) {
        if !self.by_file {
            self.last = now;
        }
    }

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

pub fn fatal_match<'a>(patterns: &'a [Regex], line: &str) -> Option<&'a Regex> {
    patterns.iter().find(|p| p.is_match(line))
}

pub struct Watch {
    pub stopping: Arc<AtomicBool>,
    child_pid: Arc<Mutex<Option<u32>>>,
    log: Option<Arc<Mutex<std::fs::File>>>,
    tag: &'static str,
}

impl Watch {
    pub fn new(tag: &'static str, log: Option<&PathBuf>) -> Result<Self> {
        let w = Self::without_ctrl_c(tag, log)?;
        uranium_store::accept_ctrl_c();
        {
            let (stopping, child_pid) = (w.stopping.clone(), w.child_pid.clone());
            ctrlc::set_handler(move || {
                stopping.store(true, Ordering::SeqCst);
                #[cfg(unix)]
                if let Some(pid) = *child_pid.lock().unwrap() {
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

    pub fn run_child(&self, args: &[String], hang: Duration, progress_file: Option<&PathBuf>, fatal: &[Regex]) -> Result<Ending> {
        let python = python()?;
        self.note(&format!("start: {} -m uranium {}", python.display(), args.join(" ")));
        let mut cmd = child_command(&python);
        cmd.args(["-m", "uranium"]).args(args);
        self.watch_child(cmd, hang, progress_file, fatal, Duration::from_secs(2))
    }

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
        if let Ending::Exited(_) = ending {
            let _ = reader.join();
        }
        *self.child_pid.lock().unwrap() = None;
        Ok(ending)
    }
}

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
            unsafe {
                cmd.pre_exec(move || {
                    if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGINT) != 0 {
                        return Err(std::io::Error::last_os_error());
                    }
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
fn utc_time() -> String {
    let s = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    format!("{:02}:{:02}:{:02}Z", (s / 3600) % 24, (s / 60) % 60, s % 60)
}
