pub type UnitRef = usize;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct RawUnit {
    pub ptr: UnitRef,
    pub tag: u32,
    pub hidden: bool,
    pub player: u8,
    pub unit_type: u16,
    pub pos: (i16, i16),
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
    pub ground_cd: u8,
    pub air_cd: u8,
    pub spell_cd: u8,
    pub order_timer: u8,
    pub order_wait: u8,
    pub status: u32,
    pub has_sprite: bool,
    pub visibility_mask: u8,
    pub detection_status: u32,
    pub build_queue: [u16; 5],
    pub build_slot: u8,
    pub remaining_build_time: u16,
    pub current_build_unit: UnitRef,
    pub related: UnitRef,
    pub fighter_parent: UnitRef,
    pub fighter_outside: bool,
    pub addon: UnitRef,
    pub research_time: u16,
    pub larva_timer: u8,
    pub resource_amount: u16,
    pub spider_mines: u8,
    pub nuke_ready: bool,
    pub timers: [u8; 8],
    pub blind: bool,
    pub parasited: bool,
    pub carried: u8,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct RawBullet {
    pub flingy_id: u16,
    pub pos: (i16, i16),
    pub remove_timer: u8,
    pub parent: UnitRef,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct RawSelf {
    pub minerals: u32,
    pub gas: u32,
    pub supply_used: u32,
    pub supply_provided: u32,
    pub supply_max: u32,
    pub upgrades: Vec<u8>,
    pub upgrading: Vec<bool>,
    pub tech: Vec<bool>,
    pub researching: Vec<bool>,
    pub nuke_available: bool,
}

#[derive(Debug, Clone, Default)]
pub struct Snapshot {
    pub frame: u32,
    pub local: u8,
    pub width: u16,
    pub height: u16,
    pub tiles: Vec<u32>,
    pub units: Vec<RawUnit>,
    pub bullets: Vec<RawBullet>,
    pub me: RawSelf,
    pub victory_state: [u8; 8],
    pub player_types: [u8; 12],
    pub allies: [bool; 12],
    pub index: std::collections::HashMap<UnitRef, usize>,
}

#[derive(Debug, Clone, Default)]
pub struct RawStatic {
    pub map_title: String,
    pub map_path: String,
    pub width: u16,
    pub height: u16,
    pub local: u8,
    pub multiplayer: bool,
    pub walkable: Vec<u8>,
    pub tiles: Vec<u32>,
    pub start_positions: [(u16, u16); 8],
    pub players: Vec<RawPlayer>,
}

#[derive(Debug, Clone, Default)]
pub struct RawPlayer {
    pub id: u8,
    pub player_type: u8,
    pub race: u8,
    pub lobby_race: u8,
    pub name: String,
    pub is_enemy: bool,
}

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

pub const MOVING_OR_ACCELERATING: u8 = 0x12;

impl Snapshot {
    pub fn tile(&self, tx: i32, ty: i32) -> Option<u32> {
        (tx >= 0 && ty >= 0 && (tx as u16) < self.width && (ty as u16) < self.height).then(|| self.tiles[ty as usize * self.width as usize + tx as usize])
    }

    pub fn visible(&self, tx: i32, ty: i32) -> bool {
        self.tile(tx, ty).is_some_and(|f| f & (1 << self.local) == 0)
    }

    pub fn explored(&self, tx: i32, ty: i32) -> bool {
        self.tile(tx, ty).is_some_and(|f| f & (1 << (8 + self.local)) == 0)
    }

    pub fn allied(&self, p: u8) -> bool {
        self.allies.get(p as usize).copied().unwrap_or(false)
    }

    pub fn index_units(&mut self) {
        self.index = self.units.iter().enumerate().map(|(i, u)| (u.ptr, i)).collect();
    }

    pub fn unit(&self, r: UnitRef) -> Option<&RawUnit> {
        self.index.get(&r).map(|&i| &self.units[i])
    }
}
