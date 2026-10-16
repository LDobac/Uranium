//! Replay playback through OpenBW replay mode (011-R10..R12) with this process as the agent:
//! observes the perspective player at each of its command frames and measures the desync
//! signals of 040-R8 (AMB-040-5): invalid selections, last frame reached, leaver ahead.

use std::io::{BufReader, BufWriter};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde::Serialize;
use uranium_proto::{Act, AgentMsg, BridgeMsg, GameEnd, PrivilegedSummary, RawFrame, read_msg, write_msg};

use crate::Replay;
use crate::cmd::op;

/// Paths from `.env` (001-R9), same keys as uranium-hostd (020-R9).
pub struct PlayConfig {
    pub launcher: PathBuf,
    pub bridge_lib: PathBuf,
    pub data_dir: PathBuf,
    pub work_dir: PathBuf,
}

impl PlayConfig {
    pub fn from_env(work_dir: PathBuf) -> Result<Self> {
        let var = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty()).with_context(|| format!("{k} is not set (.env)"));
        let bwapi = PathBuf::from(var("URANIUM_BWAPI_DIR")?);
        let launcher = if cfg!(windows) { bwapi.join("bin/Release/BWAPILauncher.exe") } else { bwapi.join("bin/BWAPILauncher") };
        Ok(Self { launcher, bridge_lib: var("URANIUM_BRIDGE_LIB")?.into(), data_dir: var("URANIUM_DATA_DIR")?.into(), work_dir })
    }
}

#[derive(Serialize, Debug, Clone, Default)]
pub struct PlayResult {
    pub perspective: u8,
    pub header_frames: u32,
    /// Frame of the bridge's GameEnd (None if the launcher died first).
    pub end_frame: Option<u32>,
    pub winner: Option<u8>,
    pub observations: u32,
    pub select_cmds: u32,
    /// Select commands naming at least one tag that is not among the observed units. Players
    /// also select visible enemy and neutral units (minerals), so any owner counts.
    pub invalid_select_cmds: u32,
    pub total_tags: u32,
    pub invalid_tags: u32,
    /// FNV-1a over the serialized RawFrames (040-R8 double-replay check).
    pub digest: String,
    pub final_frame: u32,
    pub final_self_supply: u16,
    pub final_enemy_supply: Option<u16>,
    /// Player id of the first leave command of the two players, if any.
    pub first_leaver: Option<u8>,
    /// Supplies on the last observation before the first leave command (040-R8 ③).
    pub pre_leave_self_supply: Option<u16>,
    pub pre_leave_enemy_supply: Option<u16>,
    pub wall_s: f64,
}

/// 040-R8 ① (user decision 2026-10-16): more than this share of invalid selections is a desync.
pub const MAX_INVALID_SELECT: f64 = 0.15;
/// 040-R8 ③: leaver supply at least this many times the opponent's (opponent > 0).
pub const LEAVER_AHEAD_RATIO: f64 = 1.5;

impl PlayResult {
    /// 040-R8 desync reasons (empty = keep). `recorded_winner` is the SC:R recorded winner
    /// (player id) when the replay has one (④).
    pub fn desync_reasons(&self, recorded_winner: Option<u8>) -> Vec<String> {
        let mut why = Vec::new();
        if self.select_cmds > 0 && self.invalid_select_cmds as f64 / self.select_cmds as f64 > MAX_INVALID_SELECT {
            why.push(format!("invalid selections {}/{}", self.invalid_select_cmds, self.select_cmds));
        }
        match self.end_frame {
            Some(f) if f >= self.header_frames => {}
            f => why.push(format!("ended at {f:?} before recorded frame {}", self.header_frames)),
        }
        if let (Some(leaver), Some(me), Some(them)) = (self.first_leaver, self.pre_leave_self_supply, self.pre_leave_enemy_supply) {
            let (l, o) = if leaver == self.perspective { (me, them) } else { (them, me) };
            if o > 0 && l as f64 >= o as f64 * LEAVER_AHEAD_RATIO {
                why.push(format!("leaver ahead before leaving ({l} vs {o})"));
            }
        }
        if let (Some(rec), Some(w)) = (recorded_winner, self.winner) {
            if rec != w {
                why.push(format!("recorded winner {rec} != replayed winner {w}"));
            }
        }
        why
    }
}

pub fn play(rep: &Replay, perspective: u8, cfg: &PlayConfig) -> Result<PlayResult> {
    play_with(rep, perspective, cfg, &[], &mut |_, _| {})
}

/// Observation callback: the RawFrame and the PrivilegedSummary sent with it (011-R12).
pub type OnFrame<'a> = dyn FnMut(&RawFrame, Option<&PrivilegedSummary>) + 'a;

/// Like `play`, also stopping at `extra_stops` and handing every observation to `on_frame`
/// (labels and shard steps, 040).
pub fn play_with(rep: &Replay, perspective: u8, cfg: &PlayConfig, extra_stops: &[u32], on_frame: &mut OnFrame) -> Result<PlayResult> {
    let t0 = Instant::now();
    std::fs::create_dir_all(&cfg.work_dir)?;
    let name = format!("r{}-{}-{perspective}.rep", std::process::id(), t0.elapsed().as_nanos() ^ rand_salt());
    let rep_path = cfg.work_dir.join(&name);
    std::fs::write(&rep_path, rep.to_legacy_file()?)?;
    let listener = TcpListener::bind("127.0.0.1:0")?;
    let addr = listener.local_addr()?.to_string();
    let log = std::fs::File::create(cfg.work_dir.join(format!("{name}.log")))?;
    let mut child = Command::new(&cfg.launcher)
        .current_dir(&cfg.work_dir)
        .env("URANIUM_DATA_DIR", &cfg.data_dir)
        .env("OPENBW_ENABLE_UI", "0")
        .env("BWAPI_CONFIG_AI__AI", &cfg.bridge_lib)
        .env("BWAPI_CONFIG_AUTO_MENU__AUTO_MENU", "SINGLE_PLAYER")
        .env("BWAPI_CONFIG_AUTO_MENU__MAP", &rep_path)
        .env("BWAPI_CONFIG_AUTO_MENU__AUTO_RESTART", "OFF")
        .env("BWAPI_CONFIG_AUTO_MENU__SAVE_REPLAY", "")
        .env("URANIUM_AGENT_ADDR", &addr)
        // Commands carry the player id, BWAPI players are slots (M4 spike).
        .env("URANIUM_REPLAY_PERSPECTIVE", slot_of(rep, perspective)?.to_string())
        .env("URANIUM_PRIVILEGED", "1")
        .env("URANIUM_MAX_FRAMES", (rep.header.frames + 240).to_string())
        .stdin(Stdio::null())
        .stdout(log.try_clone()?)
        .stderr(log)
        .spawn()
        .with_context(|| format!("starting {}", cfg.launcher.display()))?;
    let res = accept(&listener, Some(&mut child), Duration::from_secs(60)).and_then(|s| session(rep, perspective, s, extra_stops, on_frame));
    let deadline = Instant::now() + Duration::from_secs(10);
    while child.try_wait()?.is_none() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(50));
    }
    let _ = child.kill();
    let _ = child.wait();
    let mut res = res?;
    std::fs::remove_file(&rep_path).ok();
    res.wall_s = t0.elapsed().as_secs_f64();
    Ok(res)
}

/// BWAPI player id of the player with command id `id`: BWAPI numbers the occupied header slots
/// 0, 1, 2, ... in array order (M4 spike, matched by name against GameInfo).
pub fn slot_of(rep: &Replay, id: u8) -> Result<u8> {
    rep.header.players.iter().position(|p| p.id == id).map(|i| i as u8).with_context(|| format!("no player with id {id}"))
}

fn rand_salt() -> u128 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0)
}

/// 020-AC4 / 040-AC6: plays the replay through a hostd `StartReplay` job instead of a local
/// launcher. `agent_ip` is this machine's address as the hostd's launcher can reach it.
pub fn play_hostd(rep: &Replay, perspective: u8, hostd: &str, agent_ip: &str, extra_stops: &[u32], on_frame: &mut OnFrame) -> Result<PlayResult> {
    use uranium_proto::control::{CONTROL_VERSION, ControlRequest, ControlResponse, ReplaySpec};
    let t0 = Instant::now();
    let listener = TcpListener::bind((agent_ip, 0))?;
    let agent_addr = listener.local_addr()?.to_string();
    let ctl = TcpStream::connect(hostd).with_context(|| format!("connecting hostd {hostd}"))?;
    let mut cr = BufReader::new(ctl.try_clone()?);
    let mut cw = BufWriter::new(ctl);
    write_msg(&mut cw, &ControlRequest::Hello { control_version: CONTROL_VERSION })?;
    let _: ControlResponse = read_msg(&mut cr)?;
    let job_id = (rand_salt() as u64) & 0xffff_ffff;
    let spec = ReplaySpec { job_id, rep_bytes: rep.to_legacy_file()?, perspective_slot: slot_of(rep, perspective)?, agent_addr, privileged: true };
    write_msg(&mut cw, &ControlRequest::StartReplay(spec))?;
    match read_msg(&mut cr)? {
        ControlResponse::Started { .. } => {}
        other => bail!("hostd did not start the replay: {other:?}"),
    }
    let mut none: Option<std::process::Child> = None;
    let mut res = accept(&listener, none.as_mut(), Duration::from_secs(60)).and_then(|s| session(rep, perspective, s, extra_stops, on_frame))?;
    // Wait for the job to finish so the next job does not hit the capacity limit.
    loop {
        match read_msg::<ControlResponse>(&mut cr)? {
            ControlResponse::Finished(f) if f.game_id == job_id => break,
            _ => {}
        }
    }
    res.wall_s = t0.elapsed().as_secs_f64();
    Ok(res)
}

fn accept(listener: &TcpListener, mut child: Option<&mut std::process::Child>, timeout: Duration) -> Result<TcpStream> {
    listener.set_nonblocking(true)?;
    let deadline = Instant::now() + timeout;
    loop {
        match listener.accept() {
            Ok((s, _)) => {
                s.set_nonblocking(false)?;
                s.set_nodelay(true).ok();
                return Ok(s);
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                if let Some(st) = child.as_mut().map(|c| c.try_wait()).transpose()?.flatten() {
                    bail!("launcher exited before the bridge connected ({st})");
                }
                if Instant::now() >= deadline {
                    bail!("bridge did not connect within {timeout:?}");
                }
                std::thread::sleep(Duration::from_millis(20));
            }
            Err(e) => bail!("accept: {e}"),
        }
    }
}

fn fnv1a(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

fn session(rep: &Replay, perspective: u8, stream: TcpStream, extra_stops: &[u32], on_frame: &mut OnFrame) -> Result<PlayResult> {
    stream.set_read_timeout(Some(Duration::from_secs(60)))?;
    let mut r = BufReader::new(stream.try_clone()?);
    let mut w = BufWriter::new(stream);
    // Frames at which the perspective player issued commands, plus the last frame.
    let mine: Vec<&crate::cmd::Cmd> = rep.cmds().filter(|c| c.player == perspective).collect();
    let players: Vec<u8> = rep.active_players().iter().map(|(p, _)| p.id).collect();
    let leave = rep.cmds().find(|c| c.op == op::LEAVE_GAME && players.contains(&c.player)).map(|c| (c.player, c.frame));
    // 040-R8 ③ needs the supplies on the frame before the first leave command.
    let pre_leave = leave.map(|(_, f)| f.saturating_sub(1)).filter(|&f| f >= 1);
    let mut stops: Vec<u32> = mine.iter().map(|c| c.frame).chain([rep.header.frames.saturating_sub(1)]).chain(pre_leave).chain(extra_stops.iter().copied()).collect();
    stops.sort_unstable();
    stops.dedup();
    let mut out = PlayResult { perspective, header_frames: rep.header.frames, first_leaver: leave.map(|(p, _)| p), ..Default::default() };
    let mut digest = 0xcbf2_9ce4_8422_2325u64;
    let mut checked = 0usize; // commands of `mine` already evaluated
    let mut enemy_supply = None;
    let mut privileged: Option<PrivilegedSummary> = None;
    let expected_slot = slot_of(rep, perspective)?;
    loop {
        let msg: BridgeMsg = match read_msg(&mut r) {
            Ok(m) => m,
            Err(_) => break, // launcher exited without GameEnd
        };
        match msg {
            BridgeMsg::Hello(h) => {
                if !h.replay {
                    bail!("bridge is not in replay mode");
                }
                anyhow::ensure!(h.slot == expected_slot, "bridge observes player {} instead of {expected_slot} (013 player numbering)", h.slot);
            }
            BridgeMsg::GameInfo(gi) => {
                if std::env::var_os("URANIUM_REPLAY_DEBUG").is_some() {
                    // Header player (array index, slot, command id) -> BWAPI id, matched by name.
                    for p in &rep.header.players {
                        let bw = gi.players.iter().find(|q| q.name.as_bytes() == p.name.as_slice()).map(|q| (q.id, q.is_self));
                        eprintln!("[debug] header index {} slot {} id {} race {} -> bwapi {:?}", p.index, p.slot, p.id, p.race, bw);
                    }
                    let unmatched = gi.players.iter().filter(|q| !rep.header.players.iter().any(|p| p.name == q.name.as_bytes())).map(|q| q.id).collect::<Vec<_>>();
                    eprintln!("[debug] bwapi ids without a header name match: {unmatched:?}");
                }
            }
            BridgeMsg::Privileged(p) => {
                enemy_supply = Some(p.enemy.supply_used);
                privileged = Some(p);
            }
            BridgeMsg::Frame(f) => {
                let mut bytes = Vec::new();
                write_msg(&mut bytes, &*f)?;
                digest = fnv1a(digest, &bytes);
                out.observations += 1;
                on_frame(&f, privileged.as_ref().filter(|p| p.frame == f.frame));
                out.final_frame = f.frame;
                out.final_self_supply = f.self_state.supply_used;
                out.final_enemy_supply = enemy_supply;
                if pre_leave.is_some_and(|p| f.frame <= p) {
                    out.pre_leave_self_supply = Some(f.self_state.supply_used);
                    out.pre_leave_enemy_supply = enemy_supply;
                }
                // Evaluate every command issued up to this frame against the current units.
                while checked < mine.len() && mine[checked].frame <= f.frame {
                    let c = mine[checked];
                    checked += 1;
                    let legacy = crate::cmd::to_legacy(c, rep.unit_limit)?;
                    if matches!(legacy.op, op::SELECT | op::SELECT_ADD) {
                        let n = legacy.data[0] as usize;
                        let mut bad = 0;
                        for i in 0..n {
                            let tag = u16::from_le_bytes([legacy.data[1 + 2 * i], legacy.data[2 + 2 * i]]) as u32;
                            if !f.units.iter().any(|u| u.id == tag) {
                                bad += 1;
                            }
                        }
                        out.select_cmds += 1;
                        out.total_tags += n as u32;
                        out.invalid_tags += bad;
                        out.invalid_select_cmds += (bad > 0) as u32;
                    }
                }
                let next = stops.iter().copied().find(|&s| s > f.frame).unwrap_or(f.frame + 1);
                let act = Act { cmds: vec![], advance_frames: next - f.frame, valid_until_frame: None };
                write_msg(&mut w, &AgentMsg::Act(act))?;
            }
            BridgeMsg::End(GameEnd { winner, frame, .. }) => {
                out.end_frame = Some(frame);
                out.winner = winner;
                break;
            }
        }
    }
    out.digest = format!("{digest:016x}");
    Ok(out)
}

/// Default perspectives: every active Terran player (040-R2: both sides of TvT).
pub fn terran_perspectives(rep: &Replay) -> Vec<u8> {
    rep.active_players().iter().filter(|(p, _)| p.race == crate::header::RACE_TERRAN).map(|(p, _)| p.id).collect()
}

pub fn work_dir_default() -> PathBuf {
    std::env::temp_dir().join("uranium-replay")
}

pub fn ensure_file(p: &Path) -> Result<()> {
    if !p.is_file() {
        bail!("{} not found", p.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok() -> PlayResult {
        PlayResult { perspective: 0, header_frames: 1000, end_frame: Some(1002), select_cmds: 100, invalid_select_cmds: 5, ..Default::default() }
    }

    #[test]
    fn desync_reasons_follow_040_r8() {
        assert!(ok().desync_reasons(None).is_empty());
        assert_eq!(PlayResult { invalid_select_cmds: 16, ..ok() }.desync_reasons(None).len(), 1);
        assert!(PlayResult { invalid_select_cmds: 15, ..ok() }.desync_reasons(None).is_empty());
        assert_eq!(PlayResult { end_frame: Some(900), ..ok() }.desync_reasons(None).len(), 1);
        assert_eq!(PlayResult { end_frame: None, ..ok() }.desync_reasons(None).len(), 1);
        // leaver 1 (the enemy) had 90 vs our 60 just before leaving: 1.5x -> desync
        let lv = PlayResult { first_leaver: Some(1), pre_leave_self_supply: Some(60), pre_leave_enemy_supply: Some(90), ..ok() };
        assert_eq!(lv.desync_reasons(None).len(), 1);
        let fair = PlayResult { pre_leave_enemy_supply: Some(80), ..lv.clone() };
        assert!(fair.desync_reasons(None).is_empty());
        let dead = PlayResult { pre_leave_self_supply: Some(0), ..lv };
        assert!(dead.desync_reasons(None).is_empty());
        assert_eq!(PlayResult { winner: Some(1), ..ok() }.desync_reasons(Some(0)).len(), 1);
    }
}
