//! `#[repr(C)]` mirrors of `bridge/openbw/include/uranium_bridge.h`. Keep both in sync.

use std::ffi::c_char;

pub const UR_NONE: i32 = -1;

#[repr(C)]
pub struct UrPlayer {
    pub id: u8,
    pub race: u8,
    pub is_self: u8,
    pub is_enemy: u8,
    pub start_x: i32,
    pub start_y: i32,
    pub name: [c_char; 32],
}

#[repr(C)]
pub struct UrStatic {
    pub map_name: *const c_char,
    pub map_path: *const c_char,
    pub width_tiles: u16,
    pub height_tiles: u16,
    pub walkable: *const u8,
    pub buildable: *const u8,
    pub ground_height: *const u8,
    pub start_locations: *const u16,
    pub n_start_locations: u32,
    pub players: *const UrPlayer,
    pub n_players: u32,
    pub self_id: u8,
    pub privileged: u8,
    /// 011-R10: replay playback; `self_id` is the perspective player.
    pub replay: u8,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct UrPlayerState {
    pub minerals: i32,
    pub gas: i32,
    pub supply_used: u16,
    pub supply_total: u16,
    pub upgrades: [u8; 63],
    pub tech: u64,
    pub researching: u64,
    pub upgrading: u64,
}

#[repr(C)]
pub struct UrUnit {
    pub id: u32,
    pub owner: u8,
    pub angle: u8,
    pub ty: u16,
    pub x: u16,
    pub y: u16,
    pub vx: i16,
    pub vy: i16,
    pub hp: u16,
    pub shield: u16,
    pub energy: u16,
    pub order: u16,
    pub order_target: i32,
    pub order_x: i32,
    pub order_y: i32,
    pub ground_cd: u8,
    pub air_cd: u8,
    pub spell_cd: u8,
    pub spider_mines: u8,
    pub flags: u32,
    pub build_queue: [u16; 5],
    pub remaining_train: u16,
    pub remaining_build: u16,
    pub remaining_research: u16,
    pub resources: u16,
    pub loaded_in: i32,
    pub timers: [u16; 8],
    pub blind: u8,
    pub parasite: u8,
    pub nuke_armed: u8,
    pub has_addon: u8,
}

pub const UR_EV_CREATE: u8 = 0;
pub const UR_EV_DESTROY: u8 = 1;
pub const UR_EV_MORPH: u8 = 2;
pub const UR_EV_DISCOVER: u8 = 3;
pub const UR_EV_EVADE: u8 = 4;
pub const UR_EV_NUKE: u8 = 5;
pub const UR_EV_CMD_FAILED: u8 = 6;
/// 011-R20 ④, 010-R14: one unit of a `Command::Unit` did not take the command.
pub const UR_EV_UNIT_REJECTED: u8 = 7;

#[repr(C)]
pub struct UrEvent {
    pub kind: u8,
    /// Unit id. UR_EV_CMD_FAILED: command index. UR_EV_UNIT_REJECTED: rejected (requested) unit id.
    pub unit: i32,
    /// UR_EV_NUKE: position. UR_EV_CMD_FAILED: x = BWAPI error. UR_EV_UNIT_REJECTED: x = command
    /// index, y = BWAPI error.
    pub x: i32,
    pub y: i32,
    /// UR_EV_UNIT_REJECTED: BWAPI UnitType id, 0xFFFF when the unit was not found (and for other kinds).
    pub unit_type: u16,
    /// UR_EV_UNIT_REJECTED: BWAPI UnitCommandType id the bridge attempted for that unit. Issued
    /// units: after the per-unit rewrite. Units not found: after target resolution (a gone
    /// resource → Move included) but before any per-unit rewrite. Fog-ghost commands:
    /// Attack_Unit / Right_Click_Unit. 0 for other kinds.
    pub command: u8,
}

// 011 "C ABI 이벤트": the C layout is kind@0, unit@4, x@8, y@12, unit_type@16, command@18, size 20
// (the shim asserts the same on the C side).
const _: () = {
    assert!(std::mem::size_of::<UrEvent>() == 20);
    assert!(std::mem::align_of::<UrEvent>() == 4);
    assert!(std::mem::offset_of!(UrEvent, unit) == 4);
    assert!(std::mem::offset_of!(UrEvent, x) == 8);
    assert!(std::mem::offset_of!(UrEvent, y) == 12);
    assert!(std::mem::offset_of!(UrEvent, unit_type) == 16);
    assert!(std::mem::offset_of!(UrEvent, command) == 18);
};

#[repr(C)]
pub struct UrEffect {
    pub bullet_type: u16,
    pub x: u16,
    pub y: u16,
    pub remaining: i32,
    pub owner: i32,
}

#[repr(C)]
pub struct UrTypeCount {
    pub ty: u16,
    pub count: u16,
}

#[repr(C)]
pub struct UrPrivileged {
    pub enemy: UrPlayerState,
    pub completed: *const UrTypeCount,
    pub n_completed: u32,
    pub in_production: *const UrTypeCount,
    pub n_in_production: u32,
}

#[repr(C)]
pub struct UrFrame {
    pub frame: u32,
    pub self_state: UrPlayerState,
    pub units: *const UrUnit,
    pub n_units: u32,
    pub visible: *const u8,
    pub explored: *const u8,
    pub creep: *const u8,
    pub events: *const UrEvent,
    pub n_events: u32,
    pub effects: *const UrEffect,
    pub n_effects: u32,
    pub privileged: *const UrPrivileged,
}

pub const UR_CMD_UNIT: u8 = 0;
pub const UR_CMD_BUILD: u8 = 1;
pub const UR_CMD_BUILD_ADDON: u8 = 2;
pub const UR_CMD_TRAIN: u8 = 3;
pub const UR_CMD_MORPH: u8 = 4;
pub const UR_CMD_RESEARCH: u8 = 5;
pub const UR_CMD_UPGRADE: u8 = 6;
pub const UR_CMD_CANCEL: u8 = 7;
pub const UR_CMD_SET_RALLY: u8 = 8;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UrCmd {
    pub kind: u8,
    pub order: u8,
    pub queued: u8,
    pub cancel_kind: u8,
    pub unit: i32,
    pub target_unit: i32,
    pub x: i32,
    pub y: i32,
    pub ty: u16,
    pub n_units: u16,
    pub units_offset: u32,
}

impl Default for UrCmd {
    fn default() -> Self {
        Self { kind: 0, order: 0, queued: 0, cancel_kind: 0, unit: UR_NONE, target_unit: UR_NONE, x: UR_NONE, y: UR_NONE, ty: 0, n_units: 0, units_offset: 0 }
    }
}

#[repr(C)]
pub struct UrAct {
    pub cmds: *const UrCmd,
    pub n_cmds: u32,
    pub unit_pool: *const i32,
    pub advance_frames: u32,
    /// 010-R13 / 011-R21: non-zero = surrender, `surrender_chat` NUL-terminated (empty = no chat).
    pub surrender: u8,
    pub surrender_chat: *const c_char,
}

/// 011-R21 (`UR_SURRENDER_LEAVE_FRAMES`): frames between the surrender chat and leaving.
pub const SURRENDER_LEAVE_FRAMES: u32 = 24;

/// # Safety
/// `ptr` must point to `n` valid elements (or `n == 0`).
pub unsafe fn slice<'a, T>(ptr: *const T, n: u32) -> &'a [T] {
    if n == 0 || ptr.is_null() { &[] } else { unsafe { std::slice::from_raw_parts(ptr, n as usize) } }
}
