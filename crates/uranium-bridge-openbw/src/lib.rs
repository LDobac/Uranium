//! Rust half of the OpenBW bridge (spec 011). Exposes the C ABI declared in
//! `bridge/openbw/include/uranium_bridge.h`; the C++ shim calls these from BWAPI callbacks.
//!
//! Lockstep: `uranium_bridge_step` blocks until the agent's `Act` arrives (011-R3).

mod convert;
mod ffi;

use std::ffi::CStr;
use std::io::{BufReader, BufWriter};
use std::net::TcpStream;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};
use uranium_proto::*;

use crate::ffi::*;

struct Bridge {
    reader: BufReader<TcpStream>,
    writer: BufWriter<TcpStream>,
    width: u16,
    height: u16,
    next_frame: u32,
    cmds: Vec<UrCmd>,
    pool: Vec<i32>,
    started: Instant,
    in_bridge: Duration,
    agent_wait: Duration,
    serialize_us: Vec<f64>,
    observations: u32,
    /// 011-R10: replay playback, commands are rejected (011-R11).
    replay: bool,
}

static BRIDGE: Mutex<Option<Bridge>> = Mutex::new(None);

fn env_u64(name: &str) -> Option<u64> {
    std::env::var(name).ok().and_then(|v| v.trim().parse().ok())
}

fn log(msg: &str) {
    eprintln!("[uranium-bridge] {msg}");
}

#[unsafe(no_mangle)]
pub extern "C" fn uranium_bridge_max_frames() -> u32 {
    env_u64("URANIUM_MAX_FRAMES").map(|v| v as u32).unwrap_or(86_400)
}

#[unsafe(no_mangle)]
pub extern "C" fn uranium_bridge_privileged_requested() -> u8 {
    (std::env::var("URANIUM_PRIVILEGED").as_deref() == Ok("1")) as u8
}

#[unsafe(no_mangle)]
pub extern "C" fn uranium_bridge_replay_perspective() -> i32 {
    env_u64("URANIUM_REPLAY_PERSPECTIVE").map(|v| v as i32).unwrap_or(-1)
}

#[unsafe(no_mangle)]
pub extern "C" fn uranium_bridge_replay_enemy() -> i32 {
    env_u64("URANIUM_REPLAY_ENEMY").map(|v| v as i32).unwrap_or(-1)
}

#[unsafe(no_mangle)]
pub extern "C" fn uranium_bridge_next_frame() -> u32 {
    BRIDGE.lock().unwrap().as_ref().map(|b| b.next_frame).unwrap_or(u32::MAX)
}

fn connect() -> Result<TcpStream, String> {
    let addr = std::env::var("URANIUM_AGENT_ADDR").map_err(|_| "URANIUM_AGENT_ADDR not set".to_string())?;
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        match TcpStream::connect(&addr) {
            Ok(s) => {
                s.set_nodelay(true).ok();
                return Ok(s);
            }
            Err(e) if Instant::now() < deadline => {
                log(&format!("connect {addr}: {e}; retrying"));
                std::thread::sleep(Duration::from_millis(250));
            }
            Err(e) => return Err(format!("connect {addr}: {e}")),
        }
    }
}

fn map_hash(path: &str) -> [u8; 32] {
    match std::fs::read(path) {
        Ok(bytes) => Sha256::digest(&bytes).into(),
        Err(e) => {
            log(&format!("cannot read map {path} for hashing: {e}"));
            [0; 32]
        }
    }
}

unsafe fn cstr(p: *const std::ffi::c_char) -> String {
    if p.is_null() { String::new() } else { unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned() }
}

fn start(info: &UrStatic) -> Result<Bridge, String> {
    let (w, h) = (info.width_tiles, info.height_tiles);
    let tiles = w as usize * h as usize;
    // SAFETY: the shim passes arrays sized as documented in uranium_bridge.h.
    let (walk, build, height, starts, players) = unsafe {
        (
            slice(info.walkable, (tiles * 16) as u32),
            slice(info.buildable, tiles as u32),
            slice(info.ground_height, tiles as u32),
            slice(info.start_locations, info.n_start_locations * 2),
            slice(info.players, info.n_players),
        )
    };
    // BWAPI iterates players in pointer order; sort for determinism (011-R10).
    let mut players: Vec<_> = players.iter().collect();
    players.sort_by_key(|p| p.id);
    let players = players
        .into_iter()
        .map(|p| PlayerInfo {
            id: p.id,
            name: unsafe { cstr(p.name.as_ptr()) },
            race: convert::race(p.race),
            start_tile: (p.start_x >= 0 && p.start_y >= 0).then(|| (p.start_x as u16, p.start_y as u16)),
            is_self: p.is_self != 0,
            is_enemy: p.is_enemy != 0,
        })
        .collect();
    let map_path = unsafe { cstr(info.map_path) };
    let game_info = GameInfo {
        map_name: unsafe { cstr(info.map_name) },
        map_hash: map_hash(&map_path),
        map_size_tiles: (w, h),
        self_player: info.self_id,
        players,
        static_map: StaticMap {
            width_tiles: w,
            height_tiles: h,
            walkable: BitGrid::from_bytes(w * 4, h * 4, walk),
            buildable: BitGrid::from_bytes(w, h, build),
            ground_height: height.to_vec(),
            start_locations: starts.chunks(2).map(|c| (c[0], c[1])).collect(),
        },
        latency_frames: 0,
    };
    let hello = Hello {
        protocol_version: PROTOCOL_VERSION,
        backend: Backend::OpenBw,
        game_version: "openbw-bwapi-4.2".into(),
        mode: Mode::Lockstep,
        game_id: env_u64("URANIUM_GAME_ID").unwrap_or(0),
        slot: info.self_id,
        replay: info.replay != 0,
    };
    let stream = connect()?;
    let mut writer = BufWriter::new(stream.try_clone().map_err(|e| e.to_string())?);
    write_msg(&mut writer, &BridgeMsg::Hello(hello)).map_err(|e| e.to_string())?;
    write_msg(&mut writer, &BridgeMsg::GameInfo(Box::new(game_info))).map_err(|e| e.to_string())?;
    Ok(Bridge {
        reader: BufReader::new(stream),
        writer,
        width: w,
        height: h,
        // Replay ids are still 0 on frame 0 (CompleteMapInformation is enabled in onStart), so
        // the first observation is frame 1 (011-R15).
        next_frame: 1,
        cmds: Vec::new(),
        pool: Vec::new(),
        started: Instant::now(),
        in_bridge: Duration::ZERO,
        agent_wait: Duration::ZERO,
        serialize_us: Vec::new(),
        observations: 0,
        replay: info.replay != 0,
    })
}

/// # Safety
/// `info` must point to a valid `UrStatic` (see uranium_bridge.h).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn uranium_bridge_start(info: *const UrStatic) -> i32 {
    match start(unsafe { &*info }) {
        Ok(b) => {
            *BRIDGE.lock().unwrap() = Some(b);
            0
        }
        Err(e) => {
            log(&format!("start failed: {e}"));
            1
        }
    }
}

fn raw_frame(b: &Bridge, f: &UrFrame) -> RawFrame {
    let tiles = (b.width as usize * b.height as usize) as u32;
    // SAFETY: arrays provided by the shim for the duration of the call.
    let (units, vis, exp, creep, events, effects) = unsafe {
        (
            slice(f.units, f.n_units),
            slice(f.visible, tiles),
            slice(f.explored, tiles),
            slice(f.creep, tiles),
            slice(f.events, f.n_events),
            slice(f.effects, f.n_effects),
        )
    };
    // BWAPI unit/bullet sets iterate in pointer order, which differs between processes.
    // Sort so identical games produce identical observations (011-AC4, 010-AC6).
    let mut units: Vec<UnitState> = units.iter().map(convert::unit).collect();
    // Replay ids are unique from frame 1 on; the content tie-break only guards against a
    // duplicate ever reaching the sort (heap order differs per run under ASLR).
    units.sort_by_key(|u| (u.id, u.owner, u.unit_type, u.pos));
    let mut effects: Vec<Effect> = effects.iter().map(convert::effect).collect();
    effects.extend(units.iter().filter_map(convert::spell_unit_effect));
    effects.sort_by_key(|e| (format!("{:?}", e.kind), e.pos, e.remaining_frames, e.source_owner));
    RawFrame {
        frame: f.frame,
        self_state: convert::player_state(&f.self_state),
        units,
        visibility: BitGrid::from_bytes(b.width, b.height, vis),
        explored: BitGrid::from_bytes(b.width, b.height, exp),
        creep: BitGrid::from_bytes(b.width, b.height, creep),
        events: {
            // Callbacks for units created in the same frame fire in pointer order; sort.
            let mut ev: Vec<GameEvent> = events.iter().filter_map(convert::event).collect();
            ev.sort_by_key(convert::event_order);
            ev
        },
        effects,
    }
}

fn step(b: &mut Bridge, f: &UrFrame, out: &mut UrAct) -> Result<(), String> {
    let t0 = Instant::now();
    if !f.privileged.is_null() {
        let p = convert::privileged(f.frame, unsafe { &*f.privileged });
        write_msg(&mut b.writer, &BridgeMsg::Privileged(p)).map_err(|e| e.to_string())?;
    }
    let msg = BridgeMsg::Frame(Box::new(raw_frame(b, f)));
    write_msg(&mut b.writer, &msg).map_err(|e| e.to_string())?;
    let t1 = Instant::now();
    let reply: AgentMsg = read_msg(&mut b.reader).map_err(|e| e.to_string())?;
    let t2 = Instant::now();
    let act = match reply {
        AgentMsg::Act(a) => a,
        AgentMsg::Reject(r) => return Err(format!("agent rejected: {r}")),
    };
    if b.replay && !act.cmds.is_empty() {
        return Err(format!("011-R11: {} commands sent in replay mode", act.cmds.len()));
    }
    convert::commands(&act.cmds, &mut b.cmds, &mut b.pool);
    out.cmds = b.cmds.as_ptr();
    out.n_cmds = b.cmds.len() as u32;
    out.unit_pool = b.pool.as_ptr();
    out.advance_frames = act.advance_frames.max(1);
    b.next_frame = f.frame + out.advance_frames;
    b.observations += 1;
    b.serialize_us.push((t1 - t0).as_secs_f64() * 1e6);
    b.agent_wait += t2 - t1;
    b.in_bridge += t2 - t0;
    Ok(())
}

/// # Safety
/// `frame` and `out` must be valid pointers (see uranium_bridge.h).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn uranium_bridge_step(frame: *const UrFrame, out: *mut UrAct) -> i32 {
    let mut guard = BRIDGE.lock().unwrap();
    let Some(b) = guard.as_mut() else { return 1 };
    match step(b, unsafe { &*frame }, unsafe { &mut *out }) {
        Ok(()) => 0,
        Err(e) => {
            log(&format!("step failed at frame {}: {e}", unsafe { (*frame).frame }));
            1
        }
    }
}

fn p99(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f64::total_cmp);
    v[((v.len() as f64 * 0.99) as usize).min(v.len() - 1)]
}

/// # Safety
/// `last` is NULL or points to a valid `UrFrame` for the duration of the call (uranium_bridge.h).
/// 010-R12: `last` is the own-vision state at the end with the events since the last step; it is
/// sent as `GameEnd.final_frame` (not in replay mode, where the agent only records labels).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn uranium_bridge_end(winner: i32, reason: u8, frame: u32, last: *const UrFrame) {
    let Some(mut b) = BRIDGE.lock().unwrap().take() else { return };
    let final_frame = (!last.is_null() && !b.replay).then(|| Box::new(raw_frame(&b, unsafe { &*last })));
    let total = b.started.elapsed();
    let stats = BridgeStats {
        game_frames: frame,
        game_time_ms: total.saturating_sub(b.in_bridge).as_secs_f64() * 1e3,
        agent_wait_ms: b.agent_wait.as_secs_f64() * 1e3,
        serialize_us_p99: p99(&mut b.serialize_us),
        observations: b.observations,
    };
    let end = GameEnd {
        winner: (winner >= 0).then_some(winner as u8),
        frame,
        reason: match reason {
            0 => EndReason::Elimination,
            2 => EndReason::TimeLimitDraw,
            // 1: the enemy left or was dropped with buildings standing (011-R17).
            _ => EndReason::Surrender,
        },
        stats: Some(stats),
        final_frame,
    };
    if let Err(e) = write_msg(&mut b.writer, &BridgeMsg::End(end)) {
        log(&format!("sending GameEnd failed: {e}"));
    }
}
