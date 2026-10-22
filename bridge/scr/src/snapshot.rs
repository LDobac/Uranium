//! Plain copies of the game state of one frame (012-R2). `raw` fills them from game memory; the
//! rest of the bridge (observation, events, end, commands) works on these values only, so it is
//! testable without the game.

/// Unit identity within one frame: the unit's address (0 = none). Pointers inside the game
/// (order target, transport, build unit) are resolved through it.
pub type UnitRef = usize;

/// One live unit (BWAPI "alive": active list, hidden list, or a scanner sweep).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RawUnit {
    pub ptr: UnitRef,
    /// SC:R unit tag: array index + 1 (13 bits) | reuse counter << 13 (012-R17).
    pub tag: u32,
    /// Found in the hidden-unit list (in a transport / building).
    pub hidden: bool,
    pub player: u8,
    /// Raw unit type (the BWAPI id).
    pub unit_type: u16,
    pub pos: (i16, i16),
    /// Fixed point (x256) game values.
    pub hp: i32,
    pub shields: i32,
    pub energy: u16,
    pub order: u8,
    pub order_state: u8,
    pub secondary_order: u8,
    pub order_target_pos: (i16, i16),
    pub order_target: UnitRef,
    pub move_target: UnitRef,
    pub facing: u8,
    pub speed: (i32, i32),
    pub flingy_flags: u8,
    /// Weapon cooldowns of the unit that deals damage (its subunit, if any).
    pub ground_cd: u8,
    pub air_cd: u8,
    pub spell_cd: u8,
    pub order_timer: u8,
    pub order_wait: u8,
    /// `BW::StatusFlags`.
    pub status: u32,
    pub has_sprite: bool,
    /// Sprite visibility: bit p = player p sees the unit.
    pub visibility_mask: u8,
    pub detection_status: u32,
    pub build_queue: [u16; 5],
    pub build_slot: u8,
    pub remaining_build_time: u16,
    /// BWAPI `currentBuildUnit` (secondary order target unit).
    pub current_build_unit: UnitRef,
    /// BWAPI `connectedUnit` (the transport of a loaded unit).
    pub related: UnitRef,
    /// Interceptor / scarab / spider mine parent and whether it is outside its hangar.
    pub fighter_parent: UnitRef,
    pub fighter_outside: bool,
    pub addon: UnitRef,
    pub research_time: u16,
    pub larva_timer: u8,
    pub resource_amount: u16,
    pub spider_mines: u8,
    pub nuke_ready: bool,
    /// lockdown, irradiate, matrix, stasis, plague, ensnare, maelstrom, stim (011 shim order).
    pub timers: [u8; 8],
    pub blind: bool,
    pub parasited: bool,
    /// Worker cargo: 1 = gas, 2 = minerals (BWAPI `resourceType`).
    pub carried: u8,
}

/// One bullet (BWAPI bullet: type = flingy id).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RawBullet {
    pub flingy_id: u16,
    pub pos: (i16, i16),
    pub remove_timer: u8,
    /// Source unit (BWAPI owner = the source's player while it lives).
    pub parent: UnitRef,
}

/// Player data of the local player that `PlayerState` needs.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RawSelf {
    pub minerals: u32,
    pub gas: u32,
    pub supply_used: u32,
    pub supply_provided: u32,
    pub supply_max: u32,
    /// Upgrade levels 0..61 (BW::UPGRADE_TYPE_COUNT).
    pub upgrades: Vec<u8>,
    pub upgrading: Vec<bool>,
    /// Tech researched / in progress 0..44 (BW::TECH_TYPE_COUNT).
    pub tech: Vec<bool>,
    pub researching: Vec<bool>,
    /// Nuclear missile available (BWAPI's Nuclear_Strike "researched").
    pub nuke_available: bool,
}

/// One frame of game state.
#[derive(Debug, Clone, Default)]
pub struct Snapshot {
    pub frame: u32,
    pub local: u8,
    pub width: u16,
    pub height: u16,
    /// `map_tile_flags`, one u32 per tile: bits 0-7 not visible, 8-15 not explored (per player),
    /// 16-31 tile flags.
    pub tiles: Vec<u32>,
    pub units: Vec<RawUnit>,
    pub bullets: Vec<RawBullet>,
    pub me: RawSelf,
    pub victory_state: [u8; 8],
    /// `Player.player_type` of the 12 players.
    pub player_types: [u8; 12],
    /// `Game.alliances[local][p] != 0` (BWAPI `isAlly`).
    pub allies: [bool; 12],
    /// `units` position by address (`index_units`).
    pub index: std::collections::HashMap<UnitRef, usize>,
}

/// Game data read once at game start (`GameInfo`).
#[derive(Debug, Clone, Default)]
pub struct RawStatic {
    pub map_title: String,
    pub map_path: String,
    pub width: u16,
    pub height: u16,
    pub local: u8,
    pub multiplayer: bool,
    /// 8x8 walk tiles (width*4 x height*4), BWAPI `isWalkable`.
    pub walkable: Vec<u8>,
    pub tiles: Vec<u32>,
    /// `Game.start_position` (pixels; (0, 0) = none).
    pub start_positions: [(u16, u16); 8],
    pub players: Vec<RawPlayer>,
}

#[derive(Debug, Clone, Default)]
pub struct RawPlayer {
    pub id: u8,
    pub player_type: u8,
    /// Race the game uses (rolled for Random) and the race picked in the lobby.
    pub race: u8,
    pub lobby_race: u8,
    pub name: String,
    pub is_enemy: bool,
}

/// `map_tile_flags` tile flag bits (OpenBW `tile_t` flags << 16).
pub mod tile {
    pub const WALKABLE: u32 = 0x1 << 16;
    pub const HAS_CREEP: u32 = 0x40 << 16;
    pub const UNBUILDABLE: u32 = 0x80 << 16;
    pub const VERY_HIGH: u32 = 0x100 << 16;
    pub const MIDDLE: u32 = 0x200 << 16;
    pub const HIGH: u32 = 0x400 << 16;
    pub const PARTIALLY_WALKABLE: u32 = 0x2000 << 16;
    pub const TEMPORARY_CREEP: u32 = 0x4000 << 16;
}

/// `BW::StatusFlags`.
pub mod status {
    pub const COMPLETED: u32 = 0x1;
    pub const IN_AIR: u32 = 0x4;
    pub const DISABLED: u32 = 0x8;
    pub const BURROWED: u32 = 0x10;
    pub const IN_BUILDING: u32 = 0x20;
    pub const IN_TRANSPORT: u32 = 0x40;
    pub const REQUIRES_DETECTION: u32 = 0x100;
    pub const CLOAKED: u32 = 0x200;
    pub const UNPOWERED: u32 = 0x400;
    pub const CAN_NOT_RECEIVE_ORDERS: u32 = 0x1000;
    pub const IS_GATHERING: u32 = 0x80_0000;
    pub const INVINCIBLE: u32 = 0x400_0000;
    pub const HALLUCINATION: u32 = 0x4000_0000;
}

/// `BW::MovementFlags` (flingy flags): moving or accelerating.
pub const MOVING_OR_ACCELERATING: u8 = 0x12;

impl Snapshot {
    pub fn tile(&self, tx: i32, ty: i32) -> Option<u32> {
        (tx >= 0 && ty >= 0 && (tx as u16) < self.width && (ty as u16) < self.height).then(|| self.tiles[ty as usize * self.width as usize + tx as usize])
    }

    /// The local player sees the tile now (BWAPI `isVisible`).
    pub fn visible(&self, tx: i32, ty: i32) -> bool {
        self.tile(tx, ty).is_some_and(|f| f & (1 << self.local) == 0)
    }

    pub fn explored(&self, tx: i32, ty: i32) -> bool {
        self.tile(tx, ty).is_some_and(|f| f & (1 << (8 + self.local)) == 0)
    }

    pub fn allied(&self, p: u8) -> bool {
        self.allies.get(p as usize).copied().unwrap_or(false)
    }

    /// Builds `index`; call after `units` is filled.
    pub fn index_units(&mut self) {
        self.index = self.units.iter().enumerate().map(|(i, u)| (u.ptr, i)).collect();
    }

    /// A live unit by address (`None` for 0 or a unit not alive this frame).
    pub fn unit(&self, r: UnitRef) -> Option<&RawUnit> {
        self.index.get(&r).map(|&i| &self.units[i])
    }
}
