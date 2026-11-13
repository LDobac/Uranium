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
    /// FNV-1a over the serialized RawFrames.
    pub digest: String,
    /// 040-R8 double-replay check: FNV-1a over the observations that do not depend on
    /// `extra_stops` (events in between accumulated and sorted), so a run with extra stops
    /// compares equal to one without them.
    #[serde(skip)]
    pub check_digest: String,
    pub final_frame: u32,
    pub final_self_supply: u16,
    pub final_enemy_supply: Option<u16>,
    /// Player id of the first leave command of the two players, if any.
    pub first_leaver: Option<u8>,
    /// BWAPI id (slot) of that player (040-R8 result rule (b)).
    pub first_leaver_slot: Option<u8>,
    /// BWAPI ids (slots) of the two active players.
    pub player_slots: Vec<u8>,
    /// Supplies on the last observation before the first leave command (040-R8 ③).
    pub pre_leave_self_supply: Option<u16>,
    pub pre_leave_enemy_supply: Option<u16>,
    pub wall_s: f64,
    /// The bridge's GameInfo (static map), stored in shards (AMB-040-6).
    #[serde(skip)]
    pub game_info: Option<uranium_proto::GameInfo>,
}

/// 040-R8 ① (user decision 2026-10-16): more than this share of invalid selections is a desync.
pub const MAX_INVALID_SELECT: f64 = 0.15;
/// 040-R8 ③: leaver supply at least this many times the opponent's (opponent > 0).
pub const LEAVER_AHEAD_RATIO: f64 = 1.5;

impl PlayResult {
    /// 040-R8 result rules (a) and (b): the bridge's winner, else the opponent of the first player
    /// to leave (LEAVE 0x57). BWAPI id (slot) and `result_source`.
    pub fn decided_winner(&self) -> Option<(u8, &'static str)> {
        use crate::shard::result_source;
        if let Some(w) = self.winner {
            return Some((w, result_source::BRIDGE));
        }
        let leaver = self.first_leaver_slot?;
        match self.player_slots[..] {
            [a, b] if a == leaver => Some((b, result_source::LEAVE)),
            [a, b] if b == leaver => Some((a, result_source::LEAVE)),
            _ => None,
        }
    }

    /// 040-R21 time limit (`uranium-replay build --time-limit n`): a replay of at least n frames
    /// (the generating plan's `max_frames`; every bridge leaves there, 011 shim, and the leave
    /// commands land a frame or so before the replay's last frame) ended by the time limit, not by
    /// a player, since a leave in a 1v1 ends the game. Its first leave is then no leave (no 040-R8
    /// ③ leaver-ahead signal, no result rule (b)) and the bridge winner is dropped too (in replay
    /// mode it is the player that did not leave first): the result is undecided. `None` keeps all.
    pub fn drop_time_limit_leave(&mut self, limit: Option<u32>) {
        if let Some(n) = limit {
            if self.header_frames >= n {
                self.winner = None;
                self.first_leaver = None;
                self.first_leaver_slot = None;
                self.pre_leave_self_supply = None;
                self.pre_leave_enemy_supply = None;
            }
        }
    }

    /// 040-R8 desync reasons (empty = keep). `recorded_winner` is the recorded winner of an SC:R
    /// replay (BWAPI id, 043-R8) when there is one; ④ compares it with `decided_winner`.
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
        if let (Some(rec), Some((w, source))) = (recorded_winner, self.decided_winner()) {
            if rec != w {
                why.push(format!("recorded winner {rec} != replayed winner {w} ({source})"));
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
/// GameInfo callback: the bridge's GameInfo, before the first observation (041-R22: the DAgger
/// teacher is built from it like a live T0 session).
pub type OnInfo<'a> = dyn FnMut(&uranium_proto::GameInfo) + 'a;

/// Like `play`, also stopping at `extra_stops` and handing every observation to `on_frame`
/// (labels and shard steps, 040).
pub fn play_with(rep: &Replay, perspective: u8, cfg: &PlayConfig, extra_stops: &[u32], on_frame: &mut OnFrame) -> Result<PlayResult> {
    play_observe(rep, perspective, cfg, extra_stops, &mut |_| {}, on_frame)
}

/// Like `play_with`, also handing the bridge's GameInfo to `on_info` before the first observation.
pub fn play_observe(rep: &Replay, perspective: u8, cfg: &PlayConfig, extra_stops: &[u32], on_info: &mut OnInfo, on_frame: &mut OnFrame) -> Result<PlayResult> {
    let t0 = Instant::now();
    let (mut child, listener, rep_path) = launch(rep, perspective, cfg, true)?;
    let res = accept(&listener, Some(&mut child), Duration::from_secs(60)).and_then(|s| session(rep, perspective, s, extra_stops, on_info, on_frame));
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

/// Starts the launcher on `rep` from `perspective` (011-R10): the 1.16.1 replay file written into
/// `cfg.work_dir`, the listener the bridge connects to, the launcher process. `privileged` asks the
/// bridge for 011-R12 summaries.
fn launch(rep: &Replay, perspective: u8, cfg: &PlayConfig, privileged: bool) -> Result<(std::process::Child, TcpListener, PathBuf)> {
    std::fs::create_dir_all(&cfg.work_dir)?;
    let name = format!("r{}-{}-{perspective}.rep", std::process::id(), rand_salt());
    let rep_path = cfg.work_dir.join(&name);
    std::fs::write(&rep_path, rep.to_legacy_file()?)?;
    let listener = TcpListener::bind("127.0.0.1:0")?;
    let addr = listener.local_addr()?.to_string();
    let log = std::fs::File::create(cfg.work_dir.join(format!("{name}.log")))?;
    let mut cmd = Command::new(&cfg.launcher);
    // 011-R10: the opponent is the other active player (empty slots are left out of GameInfo).
    if let [opponent] = rep.active_players().iter().filter(|(p, _)| p.id != perspective).map(|(p, _)| p.id).collect::<Vec<_>>()[..] {
        cmd.env("URANIUM_REPLAY_ENEMY", slot_of(rep, opponent)?.to_string());
    }
    let child = cmd
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
        .env("URANIUM_PRIVILEGED", if privileged { "1" } else { "0" })
        .env("URANIUM_MAX_FRAMES", (rep.header.frames + 240).to_string())
        .stdin(Stdio::null())
        .stdout(log.try_clone()?)
        .stderr(log)
        .spawn()
        .with_context(|| format!("starting {}", cfg.launcher.display()))?;
    Ok((child, listener, rep_path))
}

/// What a stride playback saw (052-R7 `opsdump`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StrideResult {
    pub observations: u32,
    /// Frame of the last observation handed over.
    pub last_frame: Option<u32>,
    /// The bridge's GameEnd frame (`None` when the playback was cut or the launcher died first).
    pub end_frame: Option<u32>,
    /// The observer stopped the playback (`on_frame` returned false).
    pub cut: bool,
    pub wall_s: f64,
}

impl StrideResult {
    /// 052-R7 / 052 Risks: a playback that neither reached the game end nor was cut by the
    /// observer (launcher crash, 60 s read timeout, broken stream) is no measurement.
    pub fn completed(&self) -> Result<()> {
        anyhow::ensure!(self.observations > 0, "no observation");
        anyhow::ensure!(self.cut || self.end_frame.is_some(), "playback stopped at frame {:?} without the game end (launcher exit or timeout)", self.last_frame);
        Ok(())
    }
}

/// 052-R7: plays `rep` from `perspective` stopping only at multiples of `stride` frames (after the
/// bridge's first observation), without privileged summaries (033-R5), handing each observation to
/// `on_frame`. Playback ends at the game end or as soon as `on_frame` returns false: the launcher is
/// then killed (the cut, "재생을 끊는다").
pub fn play_stride(rep: &Replay, perspective: u8, cfg: &PlayConfig, stride: u32, on_info: &mut OnInfo, on_frame: &mut dyn FnMut(&RawFrame) -> bool) -> Result<StrideResult> {
    anyhow::ensure!(stride >= 1, "stride must be at least 1");
    let t0 = Instant::now();
    let (mut child, listener, rep_path) = launch(rep, perspective, cfg, false)?;
    let expected = slot_of(rep, perspective)?;
    let res = accept(&listener, Some(&mut child), Duration::from_secs(60)).and_then(|s| stride_session(s, expected, stride, on_info, on_frame));
    let cut = res.as_ref().is_ok_and(|r| r.cut);
    let deadline = Instant::now() + Duration::from_secs(if cut { 0 } else { 10 });
    while matches!(child.try_wait(), Ok(None)) && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(50));
    }
    let _ = child.kill();
    let _ = child.wait();
    let mut res = res?;
    // Thousands of replays go through here (052-R7): a played replay leaves no file behind; a
    // failed one keeps its copy and launcher log for diagnosis.
    std::fs::remove_file(&rep_path).ok();
    let mut log = rep_path.into_os_string();
    log.push(".log");
    std::fs::remove_file(&log).ok();
    res.wall_s = t0.elapsed().as_secs_f64();
    Ok(res)
}

/// The agent side of a stride playback (052-R7): every observation goes to `on_frame`, the next
/// stop is the next multiple of `stride`.
fn stride_session(stream: TcpStream, expected_slot: u8, stride: u32, on_info: &mut OnInfo, on_frame: &mut dyn FnMut(&RawFrame) -> bool) -> Result<StrideResult> {
    stream.set_read_timeout(Some(Duration::from_secs(60)))?;
    let mut r = BufReader::new(stream.try_clone()?);
    let mut w = BufWriter::new(stream);
    let mut out = StrideResult::default();
    loop {
        let msg: BridgeMsg = match read_msg(&mut r) {
            Ok(m) => m,
            Err(_) => break, // launcher exited without GameEnd
        };
        match msg {
            BridgeMsg::Hello(h) => {
                anyhow::ensure!(h.replay, "bridge is not in replay mode");
                anyhow::ensure!(h.slot == expected_slot, "bridge observes player {} instead of {expected_slot} (013 player numbering)", h.slot);
            }
            BridgeMsg::GameInfo(gi) => on_info(&gi),
            BridgeMsg::Privileged(_) => {}
            BridgeMsg::Frame(f) => {
                out.observations += 1;
                out.last_frame = Some(f.frame);
                if !on_frame(&f) {
                    out.cut = true;
                    break;
                }
                let next = (f.frame / stride + 1) * stride;
                let act = Act { cmds: vec![], advance_frames: next - f.frame, valid_until_frame: None, surrender: None };
                write_msg(&mut w, &AgentMsg::Act(act))?;
            }
            BridgeMsg::End(GameEnd { frame, .. }) => {
                out.end_frame = Some(frame);
                break;
            }
        }
    }
    Ok(out)
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
    let mut res = accept(&listener, none.as_mut(), Duration::from_secs(60)).and_then(|s| session(rep, perspective, s, extra_stops, &mut |_| {}, on_frame))?;
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

/// Folds one observation (`state` = encoded frame without events) and the events accumulated
/// since the previous hashed observation, in sorted order, into the 040-R8 check digest.
fn check_step(h: u64, state: &[u8], events: &mut Vec<Vec<u8>>) -> Result<u64> {
    events.sort_unstable();
    let mut ev = Vec::new();
    write_msg(&mut ev, events)?;
    events.clear();
    Ok(fnv1a(fnv1a(h, state), &ev))
}

fn fnv1a(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

fn session(rep: &Replay, perspective: u8, stream: TcpStream, extra_stops: &[u32], on_info: &mut OnInfo, on_frame: &mut OnFrame) -> Result<PlayResult> {
    stream.set_read_timeout(Some(Duration::from_secs(60)))?;
    let mut r = BufReader::new(stream.try_clone()?);
    let mut w = BufWriter::new(stream);
    // Frames at which the perspective player issued commands, plus the last frame.
    let mine: Vec<&crate::cmd::Cmd> = rep.cmds().filter(|c| c.player == perspective).collect();
    let players: Vec<u8> = rep.active_players().iter().map(|(p, _)| p.id).collect();
    let leave = rep.cmds().find(|c| c.op == op::LEAVE_GAME && players.contains(&c.player)).map(|c| (c.player, c.frame));
    // 040-R8 ③ needs the supplies on the frame before the first leave command.
    let pre_leave = leave.map(|(_, f)| f.saturating_sub(1)).filter(|&f| f >= 1);
    let mut base: Vec<u32> = mine.iter().map(|c| c.frame).chain([rep.header.frames.saturating_sub(1)]).chain(pre_leave).collect();
    base.sort_unstable();
    base.dedup();
    let mut extra: Vec<u32> = extra_stops.iter().copied().filter(|f| base.binary_search(f).is_err()).collect();
    extra.sort_unstable();
    extra.dedup();
    let mut stops: Vec<u32> = base.iter().chain(&extra).copied().collect();
    stops.sort_unstable();
    let mut out = PlayResult {
        perspective,
        header_frames: rep.header.frames,
        first_leaver: leave.map(|(p, _)| p),
        first_leaver_slot: leave.map(|(p, _)| slot_of(rep, p)).transpose()?,
        player_slots: players.iter().map(|&p| slot_of(rep, p)).collect::<Result<_>>()?,
        ..Default::default()
    };
    let mut digest = 0xcbf2_9ce4_8422_2325u64;
    let mut check = 0xcbf2_9ce4_8422_2325u64;
    let mut pending_events: Vec<Vec<u8>> = Vec::new();
    let last_base = base.last().copied().unwrap_or(0);
    let mut first = true;
    let mut tail: Option<Vec<u8>> = None;
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
                on_info(&gi);
                out.game_info = Some(*gi);
            }
            BridgeMsg::Privileged(p) => {
                enemy_supply = Some(p.enemy.supply_used);
                privileged = Some(p);
            }
            BridgeMsg::Frame(f) => {
                let mut bytes = Vec::new();
                write_msg(&mut bytes, &*f)?;
                digest = fnv1a(digest, &bytes);
                // 040-R8 check: hash the first observation and every base stop. Extra stops only
                // split the events, which accumulate between observations. After the last base stop
                // the runs may step differently, so only the last observation there is hashed.
                for e in &f.events {
                    let mut eb = Vec::new();
                    write_msg(&mut eb, e)?;
                    pending_events.push(eb);
                }
                let mut state = Vec::new();
                if first || base.binary_search(&f.frame).is_ok() || f.frame > last_base {
                    write_msg(&mut state, &(f.frame, &f.self_state, &f.units, &f.visibility, &f.explored, &f.creep, &f.effects))?;
                }
                if first || base.binary_search(&f.frame).is_ok() {
                    check = check_step(check, &state, &mut pending_events)?;
                } else if f.frame > last_base {
                    tail = Some(state);
                }
                first = false;
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
                let act = Act { cmds: vec![], advance_frames: next - f.frame, valid_until_frame: None, surrender: None };
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
    if let Some(state) = tail {
        check = check_step(check, &state, &mut pending_events)?;
    }
    out.check_digest = format!("{check:016x}");
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

    /// A fake replay-mode bridge: Hello (slot 1), then an observation at `first` and at every frame
    /// the agent advances to, until the agent hangs up or `end` is reached (GameEnd). Returns the
    /// frames the agent asked for.
    fn fake_bridge(addr: std::net::SocketAddr, first: u32, end: u32) -> std::thread::JoinHandle<Vec<u32>> {
        use uranium_proto::{Backend, BitGrid, EndReason, Hello, Mode, PROTOCOL_VERSION, PlayerState};
        std::thread::spawn(move || {
            let s = TcpStream::connect(addr).unwrap();
            let mut r = BufReader::new(s.try_clone().unwrap());
            let mut w = BufWriter::new(s);
            let hello = Hello { protocol_version: PROTOCOL_VERSION, backend: Backend::OpenBw, game_version: String::new(), mode: Mode::Lockstep, game_id: 0, slot: 1, replay: true };
            write_msg(&mut w, &BridgeMsg::Hello(hello)).unwrap();
            let (mut f, mut asked) = (first, vec![]);
            loop {
                if f >= end {
                    let _ = write_msg(&mut w, &BridgeMsg::End(GameEnd { winner: None, frame: end, reason: EndReason::Elimination, stats: None, final_frame: None }));
                    break;
                }
                let raw = RawFrame { frame: f, self_state: PlayerState::default(), units: vec![], visibility: BitGrid::new(1, 1), explored: BitGrid::new(1, 1), creep: BitGrid::new(1, 1), events: vec![], effects: vec![] };
                if write_msg(&mut w, &BridgeMsg::Frame(Box::new(raw))).is_err() {
                    break;
                }
                match read_msg::<AgentMsg>(&mut r) {
                    Ok(AgentMsg::Act(a)) => {
                        f += a.advance_frames;
                        asked.push(f);
                    }
                    _ => break,
                }
            }
            asked
        })
    }

    /// 052-R7: stride playback stops at every multiple of the stride after the first observation,
    /// is cut as soon as the observer says so, and otherwise runs to the game end.
    #[test]
    fn stride_session_stops_and_cut_052_r7() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let bridge = fake_bridge(listener.local_addr().unwrap(), 1, 10_000);
        let (s, _) = listener.accept().unwrap();
        let mut seen = vec![];
        let res = stride_session(s, 1, 8, &mut |_| {}, &mut |f| {
            seen.push(f.frame);
            f.frame < 40
        })
        .unwrap();
        assert_eq!(seen, vec![1, 8, 16, 24, 32, 40]);
        assert_eq!((res.observations, res.last_frame, res.end_frame, res.cut), (6, Some(40), None, true));
        res.completed().unwrap();
        assert_eq!(bridge.join().unwrap(), vec![8, 16, 24, 32, 40], "nothing asked after the cut");
        // A game ending first.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let bridge = fake_bridge(listener.local_addr().unwrap(), 0, 20);
        let (s, _) = listener.accept().unwrap();
        let res = stride_session(s, 1, 8, &mut |_| {}, &mut |_| true).unwrap();
        assert_eq!((res.observations, res.last_frame, res.end_frame, res.cut), (3, Some(16), Some(20), false));
        res.completed().unwrap();
        bridge.join().unwrap();
        // The bridge gone mid-game (no GameEnd, not cut): no measurement.
        let broken = StrideResult { observations: 3, last_frame: Some(16), ..Default::default() };
        assert!(broken.completed().is_err() && StrideResult::default().completed().is_err());
        // The wrong player's view is refused.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let bridge = fake_bridge(listener.local_addr().unwrap(), 0, 20);
        let (s, _) = listener.accept().unwrap();
        assert!(stride_session(s, 0, 8, &mut |_| {}, &mut |_| true).is_err());
        bridge.join().unwrap();
    }

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

    /// 040-R8 result rules (A-3): bridge winner first, else the first leaver loses; ④ compares a
    /// recorded winner with either.
    #[test]
    fn leaver_loses_and_recorded_winner_is_checked_040_r8() {
        use crate::shard::result_source;
        let two = PlayResult { player_slots: vec![0, 1], ..ok() };
        assert_eq!(two.decided_winner(), None);
        assert_eq!(PlayResult { winner: Some(1), first_leaver_slot: Some(1), ..two.clone() }.decided_winner(), Some((1, result_source::BRIDGE)));
        let left = PlayResult { first_leaver_slot: Some(0), ..two.clone() };
        assert_eq!(left.decided_winner(), Some((1, result_source::LEAVE)));
        assert_eq!(PlayResult { first_leaver_slot: Some(1), ..two.clone() }.decided_winner(), Some((0, result_source::LEAVE)));
        // A leaver that is not one of the two players decides nothing.
        assert_eq!(PlayResult { first_leaver_slot: Some(3), ..two.clone() }.decided_winner(), None);
        // 040-R21 time limit: a leave at or after the limit is no leave (no ③, no rule (b)).
        let timed = PlayResult { header_frames: 14_400, end_frame: Some(14_400), first_leaver: Some(0), first_leaver_slot: Some(0), pre_leave_self_supply: Some(138), pre_leave_enemy_supply: Some(20), ..two.clone() };
        assert_eq!(timed.desync_reasons(None).len(), 1);
        let mut t = timed.clone();
        t.drop_time_limit_leave(None);
        assert_eq!((t.desync_reasons(None).len(), t.decided_winner()), (1, Some((1, result_source::LEAVE))));
        t.drop_time_limit_leave(Some(14_401));
        assert_eq!(t.decided_winner(), Some((1, result_source::LEAVE)));
        t.drop_time_limit_leave(Some(14_400));
        assert!(t.desync_reasons(None).is_empty());
        assert_eq!(t.decided_winner(), None);
        // The replay-mode bridge winner of a time-limit game is the player that left second.
        let mut won = PlayResult { winner: Some(1), ..timed.clone() };
        assert_eq!(won.decided_winner(), Some((1, result_source::BRIDGE)));
        won.drop_time_limit_leave(Some(14_400));
        assert_eq!(won.decided_winner(), None);
        // A shorter game (a leave before the limit) keeps its leave and winner.
        let mut short = PlayResult { header_frames: 9_923, end_frame: Some(9_923), winner: Some(1), ..timed };
        short.drop_time_limit_leave(Some(14_400));
        assert_eq!(short.decided_winner(), Some((1, result_source::BRIDGE)));
        // ④: the recorded winner must match the leave-decided winner too.
        assert!(left.desync_reasons(Some(1)).is_empty());
        assert_eq!(left.desync_reasons(Some(0)).len(), 1);
        // Nothing decided: a recorded winner cannot disagree.
        assert!(two.desync_reasons(Some(0)).is_empty());
    }
}
