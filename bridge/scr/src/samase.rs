//! samase plugin entry and frame hook (012). Checks the environment once (012-R6), then on every
//! game frame: snapshot (012-R2) -> `Game` -> network commands (012-R4) and messages for the I/O
//! thread (012-R1). A panic disables the bridge instead of taking the game down.

use std::collections::VecDeque;
use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use samase_plugin::{PluginApi, VarId};
use sha2::{Digest, Sha256};
use uranium_proto::*;

use crate::bw;
use crate::game::Game;
use crate::io::{Io, Status};
use crate::log::log;
use crate::raw::{ExtArray, Sources};
use crate::snapshot::Snapshot;
use crate::tables::Tables;

/// SC:R build the bridge supports (012-R6, AMB-012-2): x64 `StarCraft.exe` SHA-256.
const SUPPORTED_EXE_SHA256: &str = "ce3ab05dc9a6aa35418947e5d95e19651ad6fbcd5e2c1856cd729c5b7c31281c";
pub const GAME_VERSION: &str = "1.23.10.13515";
/// samase plugin API version the bridge was written against (012-R6).
const MIN_API_VERSION: u16 = 44;
const DEFAULT_AGENT_ADDR: &str = "127.0.0.1:7700";
/// Frame hook timing is logged every this many frames (012-AC4).
const HOOK_STATS_EVERY: usize = 24 * 60 * 5;

type Getter = unsafe extern "C" fn() -> *mut c_void;

/// The samase functions the bridge uses (all present, else the bridge stays off: 012-R6).
struct Api {
    game: Getter,
    players: Getter,
    first_active_unit: Getter,
    first_hidden_unit: Getter,
    first_active_bullet: Getter,
    map_tile_flags: unsafe extern "C" fn() -> *mut u32,
    unit_array_len: unsafe extern "C" fn(*mut *mut c_void, *mut usize),
    send_command: unsafe extern "C" fn(*const c_void, u32),
    local_player_id: unsafe extern "C" fn() -> u32,
    is_multiplayer: unsafe extern "C" fn() -> u32,
    is_replay: unsafe extern "C" fn() -> u32,
    print_text: unsafe extern "C" fn(*const u8),
    read_vars: unsafe extern "C" fn(*const u16, *mut usize, usize),
    ext: Vec<ExtArray>,
}

// SAFETY: plain function pointers and read-only game memory addresses; used on the game thread.
unsafe impl Send for Api {}

struct Session {
    game: Game,
    io: Io,
    game_ptr: usize,
    last_frame: u32,
    /// Acts read but not yet applied (the schedule takes one per frame).
    inbox: VecDeque<AgentMsg>,
    warned: bool,
    snapshot: Snapshot,
}

struct Runtime {
    api: Api,
    tables: &'static Tables,
    session: Option<Session>,
    hook_us: Vec<f64>,
}

static RUNTIME: Mutex<Option<Runtime>> = Mutex::new(None);
static DISABLED: AtomicBool = AtomicBool::new(false);

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

/// 012-R6: the running exe must be the supported SC:R build.
fn exe_supported() -> Result<(), String> {
    let exe = std::env::current_exe().map_err(|e| format!("current_exe: {e}"))?;
    let bytes = std::fs::read(&exe).map_err(|e| format!("reading {}: {e}", exe.display()))?;
    let hash = sha256_hex(&bytes);
    if hash != SUPPORTED_EXE_SHA256 {
        return Err(format!("{} has SHA-256 {hash}, the bridge supports SC:R {GAME_VERSION} only ({SUPPORTED_EXE_SHA256})", exe.display()));
    }
    Ok(())
}

unsafe fn resolve(api: *const PluginApi) -> Result<Api, String> {
    let a = unsafe { &*api };
    if a.version < MIN_API_VERSION {
        return Err(format!("samase plugin API {} < {MIN_API_VERSION}", a.version));
    }
    macro_rules! get {
        ($f:ident) => {
            unsafe { (a.$f)() }.ok_or_else(|| format!("samase does not provide {}", stringify!($f)))?
        };
    }
    // Variables must be registered with load_vars before read_vars returns them (as aise does);
    // 0 = not found, 1 = needs a newer samase, 2 = read only, 3 = read / write.
    let vars = [VarId::TilesetIndexedMapTiles as u16, VarId::MinitileData as u16];
    let mut loaded = [0u8; 2];
    unsafe { (a.load_vars)(vars.as_ptr(), loaded.as_mut_ptr(), vars.len()) };
    log(&format!("load_vars TilesetIndexedMapTiles {} MinitileData {} (>= 2 = readable)", loaded[0], loaded[1]));
    let mut ext_ptr: *mut samase_plugin::ExtendedArray = std::ptr::null_mut();
    let ext_len = unsafe { (a.extended_arrays)(&mut ext_ptr) };
    let ext = (0..ext_len)
        .map(|i| {
            let e = unsafe { &*ext_ptr.add(i) };
            ExtArray { ptr: e.pointer, len: (e.end as usize).saturating_sub(e.pointer as usize) }
        })
        .collect::<Vec<_>>();
    Ok(Api {
        game: get!(game),
        players: get!(players),
        first_active_unit: get!(first_active_unit),
        first_hidden_unit: get!(first_hidden_unit),
        first_active_bullet: get!(first_active_bullet),
        map_tile_flags: get!(map_tile_flags),
        unit_array_len: get!(unit_array_len),
        send_command: get!(send_command),
        local_player_id: get!(local_player_id),
        is_multiplayer: get!(is_multiplayer),
        is_replay: get!(is_replay),
        print_text: get!(print_text),
        read_vars: a.read_vars,
        ext,
    })
}

impl Api {
    unsafe fn sources(&self) -> Sources {
        let mut units: *mut c_void = std::ptr::null_mut();
        let mut len = 0usize;
        unsafe { (self.unit_array_len)(&mut units, &mut len) };
        unsafe {
            Sources {
                game: (self.game)() as *const bw::Game,
                players: (self.players)() as *const bw::Player,
                first_active: (self.first_active_unit)() as *const bw::Unit,
                first_hidden: (self.first_hidden_unit)() as *const bw::Unit,
                units: units as *const bw::Unit,
                units_len: len,
                tile_flags: (self.map_tile_flags)(),
                first_bullet: (self.first_active_bullet)() as *const bw::Bullet,
                local: (self.local_player_id)() as u8,
                ext: self.ext.clone(),
            }
        }
    }

    fn var(&self, id: VarId) -> usize {
        let mut v = 0usize;
        let id = id as u16;
        unsafe { (self.read_vars)(&id, &mut v, 1) };
        v
    }

    fn print(&self, msg: &str) {
        let mut buf = msg.as_bytes().to_vec();
        buf.push(0);
        unsafe { (self.print_text)(buf.as_ptr()) };
    }

    fn send(&self, packet: &[u8]) {
        unsafe { (self.send_command)(packet.as_ptr() as *const c_void, packet.len() as u32) };
    }
}

/// The map file for `GameInfo.map_hash` (012-R13): `Game.map_path` as is, else under the exe's
/// install folder.
fn map_bytes(path: &str) -> Option<Vec<u8>> {
    let p = Path::new(path);
    if let Ok(b) = std::fs::read(p) {
        return Some(b);
    }
    let root = std::env::current_exe().ok()?.parent()?.parent()?.to_path_buf();
    std::fs::read(root.join(p)).ok()
}

fn env_or(name: &str, default: &str) -> String {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty()).unwrap_or_else(|| default.to_string())
}

/// Starts the session of a new game (first frame hook of the game).
fn start_session(rt: &Runtime, src: &Sources) -> Option<Session> {
    let api = &rt.api;
    if unsafe { (api.is_replay)() } != 0 {
        log("replay: the bridge stays off (012 covers live games)");
        return None;
    }
    let tiles_ptr = api.var(VarId::TilesetIndexedMapTiles) as *const u16;
    let vf4_ptr = api.var(VarId::MinitileData) as *const u16;
    let g = unsafe { &*src.game };
    let n_tiles = g.map_width_tiles as usize * g.map_height_tiles as usize;
    let megatile = |i: usize| (!tiles_ptr.is_null() && i < n_tiles).then(|| unsafe { *tiles_ptr.add(i) });
    let vf4 = |i: usize| (!vf4_ptr.is_null()).then(|| unsafe { *vf4_ptr.add(i) });
    let multiplayer = unsafe { (api.is_multiplayer)() } != 0;
    let r = unsafe { src.static_info(multiplayer, &megatile, &vf4) };
    if tiles_ptr.is_null() || vf4_ptr.is_null() {
        log("tileset data missing: partially walkable tiles count as walkable (StaticMap.walkable)");
    }
    let bytes = map_bytes(&r.map_path);
    if bytes.is_none() {
        log(&format!("cannot read the map file {:?}: GameInfo.map_hash is zero (012-R13)", r.map_path));
    }
    let info = crate::observe::game_info(&r, bytes.as_deref());
    let enemy = info.players.iter().find(|p| p.is_enemy).map(|p| p.id);
    let game_id = std::env::var("URANIUM_GAME_ID").ok().and_then(|v| v.trim().parse().ok()).unwrap_or_else(|| SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_nanos() as u64));
    let hello = Hello { protocol_version: PROTOCOL_VERSION, backend: Backend::Remastered, game_version: GAME_VERSION.into(), mode: Mode::Realtime, game_id, slot: src.local, replay: false };
    log(&format!(
        "game {game_id}: map {:?} ({}x{}), local player {}, enemy {enemy:?}, multiplayer {multiplayer}, players {:?}",
        info.map_name,
        r.width,
        r.height,
        src.local,
        r.players.iter().filter(|p| p.player_type != 0).map(|p| (p.id, p.player_type, p.race, p.lobby_race, p.name.clone())).collect::<Vec<_>>()
    ));
    if let Some(path) = std::env::var_os("URANIUM_SCR_DUMP_GAMEINFO") {
        let mut f = std::fs::File::create(PathBuf::from(path)).ok();
        if let Some(f) = f.as_mut() {
            write_msg(f, &BridgeMsg::GameInfo(Box::new(info.clone()))).ok();
        }
    }
    let io = Io::start(env_or("URANIUM_SCR_AGENT_ADDR", DEFAULT_AGENT_ADDR), hello, info, log);
    Some(Session { game: Game::new(rt.tables, src.local, enemy), io, game_ptr: src.game as usize, last_frame: g.frame_count, inbox: VecDeque::new(), warned: false, snapshot: Snapshot::default() })
}

fn hook_stats(v: &mut Vec<f64>) -> String {
    v.sort_by(f64::total_cmp);
    let q = |p: f64| v[((v.len() - 1) as f64 * p) as usize] / 1e3;
    let s = format!("012-AC4 hook ms over {} frames: p50 {:.4} p99 {:.4} max {:.4}", v.len(), q(0.5), q(0.99), q(1.0));
    v.clear();
    s
}

fn on_frame(rt: &mut Runtime) {
    let src = unsafe { rt.api.sources() };
    if src.game.is_null() {
        return;
    }
    let frame = unsafe { (*src.game).frame_count };
    // A new game: another Game object, or the frame counter went back.
    let new_game = rt.session.as_ref().is_none_or(|s| s.game_ptr != src.game as usize || frame < s.last_frame);
    if new_game {
        if let Some(old) = rt.session.take() {
            if !old.game.done {
                old.io.send(BridgeMsg::End(GameEnd { winner: None, frame: old.last_frame, reason: EndReason::Aborted, stats: None, final_frame: None }));
            }
        }
        rt.session = start_session(rt, &src);
    }
    let Some(s) = rt.session.as_mut() else { return };
    s.last_frame = frame;
    if s.game.done {
        return;
    }
    match s.io.status() {
        Status::Failed | Status::Closed => {
            if !s.warned {
                s.warned = true;
                rt.api.print("uranium: agent not connected, the bot is inactive this game (012-R16)");
            }
            return;
        }
        Status::Connecting => return,
        Status::Connected => {}
    }
    while let Some(m) = s.io.try_recv() {
        s.inbox.push_back(m);
    }
    unsafe { src.snapshot(rt.tables, &mut s.snapshot) };
    let out = s.game.frame(&s.snapshot, s.inbox.pop_front());
    for p in &out.packets {
        rt.api.send(p);
    }
    for line in &out.log {
        log(line);
    }
    for m in out.to_agent {
        s.io.send(m);
    }
    if s.game.done {
        log(&format!("012-AC6 {}", s.game.summary()));
        if !rt.hook_us.is_empty() {
            log(&hook_stats(&mut rt.hook_us));
        }
    }
}

unsafe extern "C" fn frame_hook() {
    if DISABLED.load(Ordering::Relaxed) {
        return;
    }
    let t0 = Instant::now();
    let res = std::panic::catch_unwind(|| {
        let mut guard = RUNTIME.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(rt) = guard.as_mut() {
            on_frame(rt);
            rt.hook_us.push(t0.elapsed().as_secs_f64() * 1e6);
            if rt.hook_us.len() >= HOOK_STATS_EVERY {
                log(&hook_stats(&mut rt.hook_us));
            }
        }
    });
    if res.is_err() {
        DISABLED.store(true, Ordering::Relaxed);
        log("panic in the frame hook: the bridge is off for the rest of this run");
    }
}

/// samase entry point.
///
/// # Safety
/// Called by samase with its plugin API table.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn samase_plugin_init(api: *const PluginApi) {
    log(&format!("uranium SC:R bridge {} loading (samase plugin API {})", env!("CARGO_PKG_VERSION"), unsafe { (*api).version }));
    let setup = || -> Result<Runtime, String> {
        exe_supported()?;
        let tables = Tables::get()?;
        let api = unsafe { resolve(api)? };
        Ok(Runtime { api, tables, session: None, hook_us: Vec::new() })
    };
    match setup() {
        Ok(rt) => {
            log(&format!("012-R6 checks passed, {} extended arrays", rt.api.ext.len()));
            *RUNTIME.lock().unwrap_or_else(|e| e.into_inner()) = Some(rt);
            if unsafe { ((*api).hook_step_objects)(frame_hook, 1) } == 0 {
                log("hook_step_objects failed: the bridge is off");
            }
        }
        Err(e) => log(&format!("012-R6: the bridge stays off: {e}")),
    }
}
