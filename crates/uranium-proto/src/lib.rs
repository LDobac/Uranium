//! Bridge <-> agent protocol (spec 010).
//!
//! All IDs (unit type, order, tech, upgrade) are canonical BWAPI 4.2 enum values (010-R3).
//! Positions are pixels with the origin at the top-left of the map (010-R4).
//! Transport: `u32` little-endian length prefix + postcard payload (010-R1).

pub mod control;
mod framing;
mod grid;

pub use framing::{MAX_MESSAGE_BYTES, ProtoError, read_msg, write_msg};
pub use grid::BitGrid;
use serde::{Deserialize, Serialize};

/// Bumped on any incompatible schema change (010-R2). 2: `GameEnd.final_frame` (010-R12).
/// 3: `CmdOrder::Unload` (034 `Unload_Unit`, 010-R11, 2026-10-19). 4: `Act.surrender` (010-R13,
/// 2026-10-21). 5: `GameEvent::UnitRejected` (010-R14, 2026-10-31).
pub const PROTOCOL_VERSION: u32 = 5;

pub type UnitId = u32;
/// Pixel position.
pub type Pos = (u16, u16);
/// Build tile position (32 px).
pub type Tile = (u16, u16);

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    OpenBw,
    Remastered,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Lockstep,
    Realtime,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Hello {
    pub protocol_version: u32,
    pub backend: Backend,
    pub game_version: String,
    pub mode: Mode,
    /// 010-R9: distinguishes games when several bridges share one agent server.
    pub game_id: u64,
    /// Player slot (BWAPI player id) of this bridge.
    pub slot: u8,
    /// 011-R11: replay playback mode (no commands accepted).
    pub replay: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Race {
    Zerg,
    Terran,
    Protoss,
    Random,
    Unknown,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct PlayerInfo {
    pub id: u8,
    pub name: String,
    pub race: Race,
    pub start_tile: Option<Tile>,
    pub is_self: bool,
    pub is_enemy: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct StaticMap {
    pub width_tiles: u16,
    pub height_tiles: u16,
    /// Walkability at 8 px resolution (width*4 x height*4).
    pub walkable: BitGrid,
    /// Buildability at tile resolution.
    pub buildable: BitGrid,
    /// Ground height per tile (0..=5, BWAPI `getGroundHeight`).
    pub ground_height: Vec<u8>,
    pub start_locations: Vec<Tile>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct GameInfo {
    pub map_name: String,
    /// SHA-256 of the map file (matches `config/assets.toml`).
    pub map_hash: [u8; 32],
    pub map_size_tiles: (u16, u16),
    pub self_player: u8,
    pub players: Vec<PlayerInfo>,
    pub static_map: StaticMap,
    /// Network command latency in frames (013-R3); 0 for lockstep OpenBW.
    pub latency_frames: u8,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct PlayerState {
    pub minerals: i32,
    pub gas: i32,
    /// BWAPI units (2 per zergling-sized supply).
    pub supply_used: u16,
    pub supply_total: u16,
    /// Level per upgrade type (63 entries).
    pub upgrades: Vec<u8>,
    /// Bit i = tech type i researched (47 types).
    pub tech: u64,
    pub researching: u64,
    pub upgrading: u64,
}

/// `UnitState::flags` bits.
pub mod flags {
    pub const COMPLETED: u32 = 1 << 0;
    pub const CONSTRUCTING: u32 = 1 << 1;
    pub const SIEGED: u32 = 1 << 2;
    pub const BURROWED: u32 = 1 << 3;
    pub const CLOAKED: u32 = 1 << 4;
    pub const DETECTED: u32 = 1 << 5;
    pub const LIFTED: u32 = 1 << 6;
    pub const STIMMED: u32 = 1 << 7;
    pub const CARRYING: u32 = 1 << 8;
    pub const LOADED: u32 = 1 << 9;
    pub const IDLE: u32 = 1 << 10;
    pub const UNDER_ATTACK: u32 = 1 << 11;
    pub const GATHERING: u32 = 1 << 12;
    pub const TRAINING: u32 = 1 << 13;
    pub const BEING_CONSTRUCTED: u32 = 1 << 14;
    pub const MOVING: u32 = 1 << 15;
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct StatusTimers {
    pub lockdown: u16,
    pub irradiate: u16,
    pub matrix: u16,
    pub stasis: u16,
    pub plague: u16,
    pub ensnare: u16,
    pub maelstrom: u16,
    pub stim: u16,
    pub blind: bool,
    pub parasite: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct UnitState {
    /// Unique for the unit's lifetime (BWAPI unit id).
    pub id: UnitId,
    pub owner: u8,
    pub unit_type: u16,
    pub pos: Pos,
    /// Velocity in 1/256 px per frame.
    pub vel: (i16, i16),
    pub angle: u8,
    pub hp: u16,
    pub shield: u16,
    pub energy: u16,
    pub order: u16,
    pub order_target: Option<UnitId>,
    pub order_pos: Option<Pos>,
    pub ground_cd: u8,
    pub air_cd: u8,
    pub spell_cd: u8,
    pub flags: u32,
    pub build_queue: [u16; 5],
    pub remaining_train: u16,
    pub remaining_build: u16,
    pub remaining_research: u16,
    pub resources: u16,
    pub loaded_in: Option<UnitId>,
    pub status_timers: StatusTimers,
    pub spider_mines: u8,
    pub nuke_armed: bool,
    pub has_addon: bool,
}

impl UnitState {
    pub fn has(&self, flag: u32) -> bool {
        self.flags & flag != 0
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum GameEvent {
    UnitCreated(UnitId),
    UnitDestroyed(UnitId),
    UnitMorph(UnitId),
    UnitDiscover(UnitId),
    UnitEvade(UnitId),
    NukeDetect(Pos),
    /// 031-R10: command `cmd_index` of the previous `Act` failed with BWAPI error `error`.
    CommandFailed { cmd_index: u16, error: u16 },
    /// 010-R14: unit `unit` of `Command::Unit` `cmd_index` of the previous `Act` did not take the
    /// command. `unit_type` is its canonical unit type (`None` when the unit was not found).
    /// `command` is the BWAPI `UnitCommandType` id the bridge attempted for that unit: for an issued
    /// unit after the per-unit rewrite (011-R20 ③); for a unit not found after target resolution
    /// (a gone resource → Move included, 011-R20 ①-1) but before any per-unit rewrite; for a
    /// fog-ghost command (011-R20 ①-2) Attack_Unit / Right_Click_Unit. `error` is the BWAPI error
    /// id. Diagnostic only: `CommandFailed` keeps its meaning (031-R10). Last variant so stored
    /// frames keep their postcard indices.
    UnitRejected { cmd_index: u16, unit: UnitId, unit_type: Option<u16>, command: u8, error: u16 },
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectKind {
    PsionicStorm,
    DarkSwarm,
    DisruptionWeb,
    NukeDot,
    EmpShockwave,
    Irradiate,
    Plague,
    Ensnare,
    /// Any other visible bullet (BWAPI bullet type id).
    Other(u16),
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Effect {
    pub kind: EffectKind,
    pub pos: Pos,
    pub radius: u16,
    pub remaining_frames: Option<u16>,
    pub source_owner: Option<u8>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct RawFrame {
    pub frame: u32,
    pub self_state: PlayerState,
    /// Own units + currently visible enemy and neutral units only (010-R5).
    pub units: Vec<UnitState>,
    /// Tile resolution.
    pub visibility: BitGrid,
    pub explored: BitGrid,
    pub creep: BitGrid,
    pub events: Vec<GameEvent>,
    pub effects: Vec<Effect>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
pub enum Target {
    Unit(UnitId),
    Pos(Pos),
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum CmdOrder {
    Move(Target),
    Attack(Target),
    Patrol(Pos),
    Hold,
    Stop,
    RightClick(Target),
    Siege,
    Unsiege,
    Cloak,
    Decloak,
    Burrow,
    Unburrow,
    Lift,
    Land(Tile),
    Load(UnitId),
    UnloadAll(Option<Pos>),
    Repair(UnitId),
    Gather(UnitId),
    ReturnCargo,
    UseTech { tech: u16, target: Option<Target> },
    /// One loaded unit leaves the transport (034 `Unload_Unit`, 2026-10-19). Appended last so the
    /// other variants keep their postcard index; protocol version 3.
    Unload(UnitId),
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum CancelKind {
    Construction,
    Train,
    Research,
    Upgrade,
    Addon,
    Morph,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum Command {
    /// Up to 12 units (game selection limit, 031-R3).
    Unit { units: Vec<UnitId>, order: CmdOrder, queued: bool },
    Build { worker: UnitId, unit_type: u16, tile: Tile },
    BuildAddon { building: UnitId, unit_type: u16 },
    /// Also used for larva/zerg morphs and Protoss warps by T0 bots (021-R6).
    Train { building: UnitId, unit_type: u16 },
    Morph { unit: UnitId, unit_type: u16 },
    Research { building: UnitId, tech: u16 },
    Upgrade { building: UnitId, upgrade: u16 },
    Cancel { unit: UnitId, kind: CancelKind },
    SetRally { building: UnitId, target: Target },
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Act {
    pub cmds: Vec<Command>,
    /// Frames to advance before the next observation (>= 1).
    pub advance_frames: u32,
    /// 010-R7: realtime deadline.
    pub valid_until_frame: Option<u32>,
    /// 010-R13: `Some(text)` = surrender (031-R12): no `cmds`; the bridge sends `text` as chat when
    /// it is not empty (at most [`MAX_SURRENDER_CHAT`] bytes) and leaves the game.
    pub surrender: Option<String>,
}

/// 010-R13: longest surrender chat line in UTF-8 bytes (the game's 80-byte chat buffer with its NUL,
/// as BWAPI `sendText`).
pub const MAX_SURRENDER_CHAT: usize = 79;

/// Lockstep training mode only (010-R10). Never a policy input.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct PrivilegedSummary {
    pub frame: u32,
    pub enemy: PlayerState,
    /// (unit_type, completed count) for every enemy unit type present.
    pub enemy_unit_counts: Vec<(u16, u16)>,
    /// (unit_type, in production count).
    pub enemy_in_production: Vec<(u16, u16)>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndReason {
    Elimination,
    Surrender,
    /// 020: 60-minute limit, draw.
    TimeLimitDraw,
    AgentTimeout,
    Aborted,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct BridgeStats {
    pub game_frames: u32,
    /// Wall time spent inside the game engine (excluding agent wait and serialization).
    pub game_time_ms: f64,
    pub agent_wait_ms: f64,
    pub serialize_us_p99: f64,
    pub observations: u32,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct GameEnd {
    pub winner: Option<u8>,
    pub frame: u32,
    pub reason: EndReason,
    pub stats: Option<BridgeStats>,
    /// 010-R12: what the agent would have observed at the end (own vision only, 010-R5): units
    /// visible at the last frame and the events since the last `RawFrame`, so rewards accrued
    /// after the last decision reach it (033-R7, 042). `None` when the bridge could not build it
    /// (failed shim) or in replay mode.
    pub final_frame: Option<Box<RawFrame>>,
}

/// Everything a bridge sends.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum BridgeMsg {
    Hello(Hello),
    GameInfo(Box<GameInfo>),
    Privileged(PrivilegedSummary),
    Frame(Box<RawFrame>),
    End(GameEnd),
}

/// Everything an agent sends.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum AgentMsg {
    Act(Act),
    /// 010-R2: handshake rejected (e.g. version mismatch).
    Reject(String),
}

/// 010-R2: returns an error message when the peer is incompatible.
pub fn check_hello(h: &Hello) -> Result<(), String> {
    if h.protocol_version != PROTOCOL_VERSION {
        return Err(format!("protocol version mismatch: bridge {} != agent {}", h.protocol_version, PROTOCOL_VERSION));
    }
    Ok(())
}
