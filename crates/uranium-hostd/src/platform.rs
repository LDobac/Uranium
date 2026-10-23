//! OS-specific parts of hostd (spec 020-R8): OpenBW LAN pairing and process cleanup.
//! Linux: LOCAL sockets + process groups with `PR_SET_PDEATHSIG`.
//! Windows: TCP port pairs (011-R16) + Job Objects with kill-on-close.

use std::process::{Child, Command};

use anyhow::Result;

use crate::config::Config;

/// Environment for the two launchers of one game so they find each other (011-R4, 011-R16).
pub struct LanSetup {
    pub env: [Vec<(String, String)>; 2],
    /// Resources to release when the game ends (socket file or ports).
    pub cleanup: LanCleanup,
}

pub enum LanCleanup {
    #[cfg_attr(not(unix), allow(dead_code))]
    File(std::path::PathBuf),
    /// Windows TCP port pair starting at this port (docs/commands.md, "Windows").
    #[cfg_attr(not(windows), allow(dead_code))]
    Ports(u16),
}

pub trait Platform: Send + Sync {
    fn lan_setup(&self, game_id: u64) -> Result<LanSetup>;
    fn release(&self, lan: LanSetup);
    /// Blocks until the first launcher's LAN endpoint exists, so player slots are assigned
    /// in launch order and seeded games are reproducible (011-AC4).
    fn wait_first_ready(&self, lan: &LanSetup, timeout: std::time::Duration) -> bool;
    /// Spawn so that the launcher and everything it starts can be killed as a unit and dies
    /// with hostd (020-AC3).
    fn spawn(&self, cmd: Command) -> Result<Child>;
    /// Kill the launcher and its descendants.
    fn kill_tree(&self, child: &mut Child);
    /// Release per-process resources of a launcher that already exited.
    fn reap(&self, _child: &mut Child) {}
}

/// Port pairs `(p, p+1)` for Windows TCP LAN pairing (011-R16, 020-R8). Pairs come from
/// `[base, base + 2 * pairs)`; a pair is skipped while `busy` reports either port in use.
#[cfg_attr(not(windows), allow(dead_code))]
pub struct PortPool {
    base: u16,
    taken: std::sync::Mutex<Vec<bool>>,
}

#[cfg_attr(not(windows), allow(dead_code))]
impl PortPool {
    pub fn new(base: u16, pairs: u32) -> Result<Self> {
        let pairs = pairs.max(1) as usize;
        anyhow::ensure!(base as usize + 2 * pairs <= u16::MAX as usize + 1, "URANIUM_HOSTD_PORT_BASE {base} leaves no room for {pairs} port pairs");
        Ok(Self { base, taken: std::sync::Mutex::new(vec![false; pairs]) })
    }

    /// Lowest free pair, or an error when every pair is taken or busy.
    pub fn acquire(&self, busy: impl Fn(u16) -> bool) -> Result<u16> {
        let mut taken = self.taken.lock().unwrap();
        for (i, t) in taken.iter_mut().enumerate() {
            let p = self.base + 2 * i as u16;
            if !*t && !busy(p) && !busy(p + 1) {
                *t = true;
                return Ok(p);
            }
        }
        anyhow::bail!("no free LAN port pair in {}..{}", self.base, self.base as usize + 2 * taken.len())
    }

    pub fn release(&self, p: u16) {
        if let Some(t) = self.taken.lock().unwrap().get_mut(((p - self.base) / 2) as usize) {
            *t = false;
        }
    }
}

#[cfg(unix)]
mod unix {
    use std::os::unix::process::CommandExt;
    use std::path::PathBuf;
    use std::process::{Child, Command};

    use anyhow::{Context, Result};

    use super::*;

    pub struct UnixPlatform {
        sock_dir: PathBuf,
    }

    impl UnixPlatform {
        pub fn new() -> Result<Self> {
            // Unix socket paths are limited to 108 bytes: keep them short (011, M2 finding).
            let base = std::env::var_os("XDG_RUNTIME_DIR").map(PathBuf::from).unwrap_or_else(std::env::temp_dir);
            let sock_dir = base.join("uranium");
            std::fs::create_dir_all(&sock_dir).with_context(|| format!("creating {}", sock_dir.display()))?;
            Ok(Self { sock_dir })
        }
    }

    impl Platform for UnixPlatform {
        fn lan_setup(&self, game_id: u64) -> Result<LanSetup> {
            let sock = self.sock_dir.join(format!("h{}-{game_id}.sock", std::process::id()));
            let _ = std::fs::remove_file(&sock);
            let env = vec![("OPENBW_LAN_MODE".to_string(), "LOCAL".to_string()), ("OPENBW_LOCAL_PATH".to_string(), sock.display().to_string())];
            Ok(LanSetup { env: [env.clone(), env], cleanup: LanCleanup::File(sock) })
        }

        fn release(&self, lan: LanSetup) {
            if let LanCleanup::File(p) = lan.cleanup {
                let _ = std::fs::remove_file(p);
            }
        }

        fn wait_first_ready(&self, lan: &LanSetup, timeout: std::time::Duration) -> bool {
            let LanCleanup::File(p) = &lan.cleanup else { return true };
            let t = std::time::Instant::now();
            while t.elapsed() < timeout {
                if p.exists() {
                    return true;
                }
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
            false
        }

        fn spawn(&self, mut cmd: Command) -> Result<Child> {
            cmd.process_group(0);
            // SAFETY: only async-signal-safe calls between fork and exec.
            unsafe {
                cmd.pre_exec(|| {
                    // Die with the hostd thread that spawned us (kill -9 of hostd, 020-AC3).
                    if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0 {
                        return Err(std::io::Error::last_os_error());
                    }
                    Ok(())
                });
            }
            Ok(cmd.spawn()?)
        }

        fn kill_tree(&self, child: &mut Child) {
            let pgid = child.id() as i32;
            // SAFETY: plain syscall; the group id is the child's pid (process_group(0)).
            unsafe {
                libc::kill(-pgid, libc::SIGKILL);
            }
            let _ = child.wait();
        }
    }
}

#[cfg(windows)]
mod windows {
    use std::collections::HashMap;
    use std::os::windows::io::AsRawHandle;
    use std::os::windows::process::CommandExt;
    use std::process::{Child, Command};
    use std::sync::Mutex;

    use anyhow::{Context, Result};
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
    use windows_sys::Win32::NetworkManagement::IpHelper::{GetExtendedTcpTable, MIB_TCPROW_OWNER_PID, TCP_TABLE_OWNER_PID_LISTENER};
    use windows_sys::Win32::Networking::WinSock::AF_INET;
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JobObjectExtendedLimitInformation, SetInformationJobObject, TerminateJobObject,
    };
    use windows_sys::Win32::System::Threading::CREATE_NO_WINDOW;

    use super::*;

    pub struct WindowsPlatform {
        ports: PortPool,
        /// Job handle (as usize, for Send) per launcher pid. Closing it kills the launcher, so
        /// the launchers die with hostd even on `taskkill /F` (020-AC3).
        jobs: Mutex<HashMap<u32, usize>>,
    }

    impl WindowsPlatform {
        pub fn new(cfg: &Config) -> Result<Self> {
            Ok(Self { ports: PortPool::new(cfg.port_base, cfg.max_games)?, jobs: Mutex::new(HashMap::new()) })
        }

        fn take_job(&self, child: &Child) -> Option<HANDLE> {
            self.jobs.lock().unwrap().remove(&child.id()).map(|h| h as HANDLE)
        }
    }

    /// IPv4 TCP ports in LISTEN state, from the system TCP table.
    fn listening_ports() -> Vec<u16> {
        let mut size = 0u32;
        // SAFETY: size query with a null buffer, then a buffer of the reported size.
        unsafe {
            GetExtendedTcpTable(std::ptr::null_mut(), &mut size, 0, AF_INET as u32, TCP_TABLE_OWNER_PID_LISTENER, 0);
            let mut buf = vec![0u32; (size as usize).div_ceil(4) + 64];
            size = (buf.len() * 4) as u32;
            if GetExtendedTcpTable(buf.as_mut_ptr().cast(), &mut size, 0, AF_INET as u32, TCP_TABLE_OWNER_PID_LISTENER, 0) != 0 {
                return Vec::new();
            }
            let n = buf[0] as usize;
            let rows = std::slice::from_raw_parts(buf.as_ptr().add(1).cast::<MIB_TCPROW_OWNER_PID>(), n);
            // dwLocalPort holds the port in network byte order in its low 16 bits.
            rows.iter().map(|r| u16::from_be(r.dwLocalPort as u16)).collect()
        }
    }

    impl Platform for WindowsPlatform {
        fn lan_setup(&self, _game_id: u64) -> Result<LanSetup> {
            // Ports another program listens on are skipped. OpenBW binds with SO_REUSEADDR, so
            // TIME_WAIT leftovers of earlier games do not block a pair; the LISTEN table decides.
            let listening = listening_ports();
            let p = self.ports.acquire(|port| listening.contains(&port))?;
            let env = |listen: u16, connect: u16| {
                vec![
                    ("OPENBW_LAN_MODE".to_string(), "TCP".to_string()),
                    ("OPENBW_TCP_LISTEN_HOSTNAME".to_string(), "127.0.0.1".to_string()),
                    ("OPENBW_TCP_LISTEN_PORT".to_string(), listen.to_string()),
                    ("OPENBW_TCP_CONNECT_HOSTNAME".to_string(), "127.0.0.1".to_string()),
                    ("OPENBW_TCP_CONNECT_PORT".to_string(), connect.to_string()),
                ]
            };
            Ok(LanSetup { env: [env(p, p + 1), env(p + 1, p)], cleanup: LanCleanup::Ports(p) })
        }

        fn release(&self, lan: LanSetup) {
            if let LanCleanup::Ports(p) = lan.cleanup {
                self.ports.release(p);
            }
        }

        /// Slot 0 listens on p and connects to p+1 once (refused, nothing there yet); slot 1
        /// then connects to p. A probe connect to p would be accepted as a client, so the
        /// LISTEN table is polled instead (011-AC4).
        fn wait_first_ready(&self, lan: &LanSetup, timeout: std::time::Duration) -> bool {
            let LanCleanup::Ports(p) = lan.cleanup else { return true };
            let t = std::time::Instant::now();
            while t.elapsed() < timeout {
                if listening_ports().contains(&p) {
                    return true;
                }
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
            false
        }

        fn spawn(&self, mut cmd: Command) -> Result<Child> {
            // SAFETY: plain Win32 calls; every handle is checked and closed on failure.
            unsafe {
                let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
                anyhow::ensure!(!job.is_null(), "CreateJobObjectW: {}", std::io::Error::last_os_error());
                let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
                info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
                let ok = SetInformationJobObject(
                    job,
                    JobObjectExtendedLimitInformation,
                    (&info as *const JOBOBJECT_EXTENDED_LIMIT_INFORMATION).cast(),
                    std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                );
                if ok == 0 {
                    let e = std::io::Error::last_os_error();
                    CloseHandle(job);
                    anyhow::bail!("SetInformationJobObject: {e}");
                }
                // BWAPILauncher starts no child processes, so assigning right after spawn
                // leaves nothing outside the job.
                let mut child = match cmd.creation_flags(CREATE_NO_WINDOW).spawn().context("spawning launcher") {
                    Ok(c) => c,
                    Err(e) => {
                        CloseHandle(job);
                        return Err(e);
                    }
                };
                if AssignProcessToJobObject(job, child.as_raw_handle() as HANDLE) == 0 {
                    let e = std::io::Error::last_os_error();
                    let _ = child.kill();
                    let _ = child.wait();
                    CloseHandle(job);
                    anyhow::bail!("AssignProcessToJobObject: {e}");
                }
                self.jobs.lock().unwrap().insert(child.id(), job as usize);
                Ok(child)
            }
        }

        fn kill_tree(&self, child: &mut Child) {
            if let Some(job) = self.take_job(child) {
                // SAFETY: job is a handle this platform created and still owns.
                unsafe {
                    TerminateJobObject(job, 1);
                    CloseHandle(job);
                }
            } else {
                let _ = child.kill();
            }
            let _ = child.wait();
        }

        fn reap(&self, child: &mut Child) {
            if let Some(job) = self.take_job(child) {
                // SAFETY: as above; the launcher already exited, closing kills any leftovers.
                unsafe {
                    CloseHandle(job);
                }
            }
        }
    }
}

#[cfg_attr(unix, allow(unused_variables))]
pub fn native(cfg: &Config) -> Result<Box<dyn Platform>> {
    #[cfg(unix)]
    return Ok(Box::new(unix::UnixPlatform::new()?));
    #[cfg(windows)]
    return Ok(Box::new(windows::WindowsPlatform::new(cfg)?));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn port_pool_acquire_release_exhaust_020_r8() {
        let pool = PortPool::new(17000, 2).unwrap();
        assert_eq!(pool.acquire(|_| false).unwrap(), 17000);
        assert_eq!(pool.acquire(|_| false).unwrap(), 17002);
        assert!(pool.acquire(|_| false).is_err());
        pool.release(17000);
        assert_eq!(pool.acquire(|_| false).unwrap(), 17000);
    }

    #[test]
    fn port_pool_skips_busy_ports_020_r8() {
        let pool = PortPool::new(17000, 2).unwrap();
        // 17001 is held by another program: the whole first pair is skipped.
        assert_eq!(pool.acquire(|p| p == 17001).unwrap(), 17002);
        assert!(pool.acquire(|p| p == 17001).is_err());
        assert!(PortPool::new(65535, 1).is_err());
    }
}
