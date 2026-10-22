//! Game memory -> `Snapshot` / `RawStatic` (012-R2). The only module that dereferences game
//! pointers; it only reads (012-R4). Pointers come from the samase plugin API.

use std::mem::size_of;

use uranium_env::types::TypeTables;

use crate::bw;
use crate::snapshot::{RawBullet, RawPlayer, RawSelf, RawStatic, RawUnit, Snapshot};
use crate::tables::Tables;

/// One samase extended array (`samase_plugin::ExtendedArray`): per-player data of ids beyond the
/// legacy tables, 12 bytes per id.
#[derive(Clone, Copy)]
pub struct ExtArray {
    pub ptr: *const u8,
    pub len: usize,
}

impl ExtArray {
    unsafe fn read(&self, offset: usize) -> u8 {
        if offset < self.len { unsafe { *self.ptr.add(offset) } } else { 0 }
    }
}

/// Pointers of one frame.
pub struct Sources {
    pub game: *const bw::Game,
    /// Array of 12 players.
    pub players: *const bw::Player,
    pub first_active: *const bw::Unit,
    pub first_hidden: *const bw::Unit,
    pub units: *const bw::Unit,
    pub units_len: usize,
    pub tile_flags: *const u32,
    pub first_bullet: *const bw::Bullet,
    pub local: u8,
    /// samase extended arrays (0 upgrade level, 2 tech researched, 4 upgrading, 5 researching).
    pub ext: Vec<ExtArray>,
}

/// Unit chain safety bound (the array has 3400 units).
const MAX_CHAIN: usize = 4096;

fn cstr(bytes: &[u8]) -> String {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

impl Sources {
    /// SC:R unit tag of a unit pointer (012-R17): array index + 1, reuse counter above.
    unsafe fn tag(&self, u: *const bw::Unit) -> u32 {
        let index = (u as usize - self.units as usize) / size_of::<bw::Unit>();
        let shift = if self.units_len > 1700 { 13 } else { 11 };
        (index as u32 + 1) | (unsafe { (*u).minor_unique_index } as u32) << shift
    }

    unsafe fn read_unit(&self, p: *const bw::Unit, hidden: bool, t: &Tables) -> Option<RawUnit> {
        let u = unsafe { &*p };
        let f = &u.flingy;
        // BWAPI isUnitAlive (GameUnits.cpp:38-48): a sprite and not dying; hidden scarabs are dead.
        if f.sprite.is_null() || u.order == t.orders.die {
            return None;
        }
        if hidden && u.unit_id == t.units.scarab {
            return None;
        }
        let sprite = unsafe { &*f.sprite };
        let dealer = if u.subunit.is_null() { u } else { unsafe { &*u.subunit } };
        let s1 = &u.unit_specific;
        let s2 = &u.unit_specific2;
        Some(RawUnit {
            ptr: p as usize,
            tag: unsafe { self.tag(p) },
            hidden,
            player: u.player,
            unit_type: u.unit_id,
            pos: (f.position.x, f.position.y),
            hp: f.hitpoints,
            shields: u.shields,
            energy: u.energy,
            order: u.order,
            order_state: u.order_state,
            secondary_order: u.secondary_order,
            order_target_pos: (u.order_target.pos.x, u.order_target.pos.y),
            order_target: u.order_target.unit as usize,
            move_target: f.move_target.unit as usize,
            facing: f.facing_direction,
            speed: (f.current_speed_x, f.current_speed_y),
            flingy_flags: f.flingy_flags,
            ground_cd: dealer.ground_cooldown,
            air_cd: dealer.air_cooldown,
            spell_cd: u.spell_cooldown,
            order_timer: u.order_timer,
            order_wait: u.order_wait,
            status: u.flags,
            has_sprite: true,
            visibility_mask: sprite.visibility_mask,
            detection_status: u.detection_status,
            build_queue: u.build_queue,
            build_slot: u.current_build_slot,
            remaining_build_time: u.remaining_build_time,
            current_build_unit: u.secondary_order_target.unit as usize,
            related: u.related as usize,
            // SAFETY (unions): read as plain bytes; observe uses each only for matching types.
            fighter_parent: unsafe { s1.interceptor.parent } as usize,
            fighter_outside: unsafe { s1.interceptor.is_outside_hangar } != 0,
            addon: unsafe { s1.building.addon } as usize,
            research_time: unsafe { s1.building.research_time_remaining },
            larva_timer: unsafe { s1.building.larva_timer },
            resource_amount: unsafe { s2.resource.amount },
            spider_mines: unsafe { s1.vulture.mines },
            nuke_ready: unsafe { s2.nuke_silo.has_nuke } != 0,
            timers: [u.lockdown_timer, u.irradiate_timer, u.matrix_timer, u.stasis_timer, u.plague_timer, u.ensnare_timer, u.maelstrom_timer, u.stim_timer],
            blind: u.is_blind != 0,
            parasited: u.parasited_by_players != 0,
            carried: u.carried_powerup_flags & 3,
        })
    }

    /// Live units: the active list, the hidden list, and scanner sweeps (which BWAPI finds in a
    /// list samase does not expose; scanned from the unit array instead).
    unsafe fn read_units(&self, t: &Tables, out: &mut Vec<RawUnit>) {
        out.clear();
        for (first, hidden) in [(self.first_active, false), (self.first_hidden, true)] {
            let mut p = first;
            let mut n = 0;
            while !p.is_null() && n < MAX_CHAIN {
                if let Some(u) = unsafe { self.read_unit(p, hidden, t) } {
                    out.push(u);
                }
                p = unsafe { (*p).flingy.next } as *const bw::Unit;
                n += 1;
            }
        }
        if !self.units.is_null() {
            for i in 0..self.units_len {
                let p = unsafe { self.units.add(i) };
                if unsafe { (*p).unit_id } == t.units.scanner_sweep && !out.iter().any(|u| u.ptr == p as usize) {
                    if let Some(u) = unsafe { self.read_unit(p, false, t) } {
                        out.push(u);
                    }
                }
            }
        }
    }

    unsafe fn read_bullets(&self, out: &mut Vec<RawBullet>) {
        out.clear();
        let mut p = self.first_bullet;
        let mut n = 0;
        while !p.is_null() && n < MAX_CHAIN {
            let b = unsafe { &*p };
            if !b.flingy.sprite.is_null() {
                out.push(RawBullet { flingy_id: b.flingy.flingy_id, pos: (b.flingy.position.x, b.flingy.position.y), remove_timer: b.death_timer, parent: b.parent as usize });
            }
            p = b.flingy.next as *const bw::Bullet;
            n += 1;
        }
    }

    /// Local player's resources, supply, upgrades and techs (PlayerImpl.cpp:185-238).
    unsafe fn read_self(&self, g: &bw::Game, t: &Tables) -> RawSelf {
        let me = self.local as usize;
        let race = unsafe { (*self.players.add(me)).race } as usize;
        let ext = |i: usize| self.ext.get(i).copied();
        let upgrade_level = |id: usize| match ext(0) {
            Some(a) => unsafe { a.read(id * 12 + me) },
            None if id >= 0x2e => g.upgrade_level_bw[me][id - 0x2e],
            None => g.upgrade_level_sc[me][id],
        };
        let tech_level = |id: usize| match ext(2) {
            Some(a) => unsafe { a.read(id * 12 + me) },
            None if id >= 0x18 => g.tech_level_bw[me][id - 0x18],
            None => g.tech_level_sc[me][id],
        };
        let bit = |arr: Option<ExtArray>, legacy: &[u8], id: usize| {
            let b = match arr {
                Some(a) => unsafe { a.read((id / 8) * 12 + me) },
                None => legacy.get(id / 8).copied().unwrap_or(0),
            };
            b & (1 << (id & 7)) != 0
        };
        let types = TypeTables::get();
        let supplies = g.supplies.get(race.min(2)).map_or((0, 0, 0), |s| (s.used[me], s.provided[me], s.max[me]));
        RawSelf {
            minerals: g.minerals[me],
            gas: g.gas[me],
            supply_used: supplies.0,
            supply_provided: supplies.1,
            supply_max: supplies.2,
            upgrades: (0..61).map(upgrade_level).collect(),
            upgrading: (0..61).map(|i| bit(ext(4), &g.upgrade_in_progress[me], i)).collect(),
            // A tech no building researches counts as researched (PlayerImpl.cpp:195).
            tech: (0..44).map(|i| types.techs.get(i).is_some_and(|x| x.what_researches < 0) || tech_level(i) != 0).collect(),
            researching: (0..44).map(|i| bit(ext(5), &g.tech_in_progress[me], i)).collect(),
            nuke_available: g.unit_availability[me].get(t.units.nuclear_missile as usize).is_some_and(|&v| v != 0),
        }
    }

    /// One frame (`out` is reused between frames).
    pub unsafe fn snapshot(&self, t: &Tables, out: &mut Snapshot) {
        let g = unsafe { &*self.game };
        out.frame = g.frame_count;
        out.local = self.local;
        out.width = g.map_width_tiles;
        out.height = g.map_height_tiles;
        let n = out.width as usize * out.height as usize;
        out.tiles.clear();
        if !self.tile_flags.is_null() {
            out.tiles.extend_from_slice(unsafe { std::slice::from_raw_parts(self.tile_flags, n) });
        } else {
            // Unknown map state: nothing visible.
            out.tiles.resize(n, u32::MAX);
        }
        unsafe {
            self.read_units(t, &mut out.units);
            self.read_bullets(&mut out.bullets);
            out.me = self.read_self(g, t);
        }
        out.victory_state = g.victory_state;
        for p in 0..12 {
            out.player_types[p] = unsafe { (*self.players.add(p)).player_type };
            out.allies[p] = g.alliances[self.local as usize][p] != 0;
        }
        out.index_units();
    }

    /// Data read once at game start (`GameInfo`, shim onStart). `megatile(i)` and `vf4(m)` give
    /// the tileset data for walkability (`None` when samase did not provide it).
    pub unsafe fn static_info(&self, multiplayer: bool, megatile: &dyn Fn(usize) -> Option<u16>, vf4: &dyn Fn(usize) -> Option<u16>) -> RawStatic {
        let g = unsafe { &*self.game };
        let (w, h) = (g.map_width_tiles, g.map_height_tiles);
        let n = w as usize * h as usize;
        let tiles = if self.tile_flags.is_null() { vec![0; n] } else { unsafe { std::slice::from_raw_parts(self.tile_flags, n) }.to_vec() };
        let minitile = |ti: usize, mx: usize, my: usize| {
            let m = megatile(ti)? as usize & 0x7fff;
            vf4(m * 16 + my * 4 + mx).map(|f| f & 1 != 0)
        };
        let walkable = crate::observe::walkability(w, h, &tiles, &minitile);
        let me = self.local as usize;
        let players = (0..8u8)
            .map(|p| {
                let pl = unsafe { &*self.players.add(p as usize) };
                RawPlayer {
                    id: p,
                    player_type: pl.player_type,
                    race: pl.race,
                    lobby_race: g.starting_races[p as usize],
                    name: cstr(&pl.name),
                    is_enemy: p as usize != me && g.alliances[me][p as usize] == 0,
                }
            })
            .collect();
        let mut starts = [(0u16, 0u16); 8];
        for (i, s) in g.start_position.iter().enumerate() {
            starts[i] = (s[0], s[1]);
        }
        RawStatic {
            map_title: cstr(&g.map_title),
            map_path: cstr(&g.map_path),
            width: w,
            height: h,
            local: self.local,
            multiplayer,
            walkable,
            tiles,
            start_positions: starts,
            players,
        }
    }
}
