//! OS-specific parts of hostd (spec 020-R8): OpenBW LAN pairing and process cleanup.
//! Linux is implemented here; the Windows implementation (TCP port pairs + Job Object) is
//! tracked in `docs/windows.md` and written in a Windows session.

use std::process::{Child, Command};

use anyhow::Result;

/// Environment for the two launchers of one game so they find each other (011-R4, 011-R16).
pub struct LanSetup {
    pub env: [Vec<(String, String)>; 2],
    /// Resources to release when the game ends (socket file or ports).
    pub cleanup: LanCleanup,
}

pub enum LanCleanup {
    #[cfg_attr(not(unix), allow(dead_code))]
    File(std::path::PathBuf),
    /// Windows TCP port pair starting at this port (docs/windows.md).
    #[cfg_attr(not(windows), allow(dead_code))]
    Ports(u16),
}

pub trait Platform: Send + Sync {
    fn lan_setup(&self, game_id: u64) -> Result<LanSetup>;
    fn release(&self, lan: LanSetup);
    /// Spawn so that the launcher and everything it starts can be killed as a unit and dies
    /// with hostd (020-AC3).
    fn spawn(&self, cmd: Command) -> Result<Child>;
    /// Kill the launcher and its descendants.
    fn kill_tree(&self, child: &mut Child);
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
    use anyhow::{Result, bail};

    use super::*;

    /// Placeholder until the Windows session implements TCP port pairs and Job Objects
    /// (docs/windows.md, spec 020-R8 / 011-R16).
    pub struct WindowsPlatform;

    impl WindowsPlatform {
        pub fn new() -> Result<Self> {
            bail!("uranium-hostd: Windows platform not implemented yet — see docs/windows.md")
        }
    }

    impl Platform for WindowsPlatform {
        fn lan_setup(&self, _game_id: u64) -> Result<LanSetup> {
            unreachable!()
        }
        fn release(&self, _lan: LanSetup) {}
        fn spawn(&self, _cmd: Command) -> Result<Child> {
            unreachable!()
        }
        fn kill_tree(&self, _child: &mut Child) {}
    }
}

pub fn native() -> Result<Box<dyn Platform>> {
    #[cfg(unix)]
    return Ok(Box::new(unix::UnixPlatform::new()?));
    #[cfg(windows)]
    return Ok(Box::new(windows::WindowsPlatform::new()?));
}
