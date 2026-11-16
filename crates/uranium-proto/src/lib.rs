pub mod control;
mod framing;
mod grid;

pub use framing::{MAX_MESSAGE_BYTES, ProtoError, read_msg, write_msg};
pub use grid::BitGrid;
use serde::{Deserialize, Serialize};

pub const PROTOCOL_VERSION: u32 = 5;

pub type UnitId = u32;
pub type Pos = (u16, u16);
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
    pub game_id: u64,
    pub slot: u8,
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
    pub walkable: BitGrid,
    pub buildable: BitGrid,
    pub ground_height: Vec<u8>,
    pub start_locations: Vec<Tile>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct GameInfo {
    pub map_name: String,
    pub map_hash: [u8; 32],
    pub map_size_tiles: (u16, u16),
    pub self_player: u8,
    pub players: Vec<PlayerInfo>,
    pub static_map: StaticMap,
    pub latency_frames: u8,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct PlayerState {
    pub minerals: i32,
    pub gas: i32,
    pub supply_used: u16,
    pub supply_total: u16,
    pub upgrades: Vec<u8>,
    pub tech: u64,
    pub researching: u64,
    pub upgrading: u64,
}

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
    pub id: UnitId,
    pub owner: u8,
    pub unit_type: u16,
    pub pos: Pos,
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
    CommandFailed { cmd_index: u16, error: u16 },
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
    pub units: Vec<UnitState>,
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
    Unit { units: Vec<UnitId>, order: CmdOrder, queued: bool },
    Build { worker: UnitId, unit_type: u16, tile: Tile },
    BuildAddon { building: UnitId, unit_type: u16 },
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
    pub advance_frames: u32,
    pub valid_until_frame: Option<u32>,
    pub surrender: Option<String>,
}

pub const MAX_SURRENDER_CHAT: usize = 79;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct PrivilegedSummary {
    pub frame: u32,
    pub enemy: PlayerState,
    pub enemy_unit_counts: Vec<(u16, u16)>,
    pub enemy_in_production: Vec<(u16, u16)>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndReason {
    Elimination,
    Surrender,
    TimeLimitDraw,
    AgentTimeout,
    Aborted,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct BridgeStats {
    pub game_frames: u32,
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
    pub final_frame: Option<Box<RawFrame>>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum BridgeMsg {
    Hello(Hello),
    GameInfo(Box<GameInfo>),
    Privileged(PrivilegedSummary),
    Frame(Box<RawFrame>),
    End(GameEnd),
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum AgentMsg {
    Act(Act),
    Reject(String),
}

pub fn check_hello(h: &Hello) -> Result<(), String> {
    if h.protocol_version != PROTOCOL_VERSION {
        return Err(format!("protocol version mismatch: bridge {} != agent {}", h.protocol_version, PROTOCOL_VERSION));
    }
    Ok(())
}
