//! Synthetic games for RL unit tests (no game data needed): a small map with a few Terran units
//! and an enemy base, run through the real feature extraction (030).

use uranium_env::features::{AgentMemory, MapInfo, ORDER_NONE, Observation, extract};
use uranium_env::types::TypeTables;
use uranium_proto::*;

const W: u16 = 64;
const H: u16 = 64;

pub fn game_info() -> GameInfo {
    let n = W as usize * H as usize;
    GameInfo {
        map_name: "synthetic".into(),
        map_hash: [0; 32],
        map_size_tiles: (W, H),
        self_player: 0,
        players: vec![
            PlayerInfo { id: 0, name: "uranium-p0".into(), race: Race::Terran, start_tile: Some((8, 8)), is_self: true, is_enemy: false },
            PlayerInfo { id: 1, name: "uranium-p1".into(), race: Race::Zerg, start_tile: Some((50, 50)), is_self: false, is_enemy: true },
        ],
        static_map: StaticMap {
            width_tiles: W,
            height_tiles: H,
            walkable: BitGrid::from_bytes(W * 4, H * 4, &vec![1; n * 16]),
            buildable: BitGrid::from_bytes(W, H, &vec![1; n]),
            ground_height: vec![0; n],
            start_locations: vec![(8, 8), (50, 50)],
        },
        latency_frames: 0,
    }
}

fn unit(id: u32, owner: u8, name: &str, tile: (i32, i32)) -> UnitState {
    let t = TypeTables::get().unit_by_name(name).unwrap_or_else(|| panic!("{name}"));
    let none = TypeTables::get().unit_by_name("None").unwrap().id as u16;
    let pos = if t.is_building() || t.is_resource() {
        ((tile.0 * 32 + t.tile_width * 16) as u16, (tile.1 * 32 + t.tile_height * 16) as u16)
    } else {
        ((tile.0 * 32 + 16) as u16, (tile.1 * 32 + 16) as u16)
    };
    UnitState {
        id,
        owner,
        unit_type: t.id as u16,
        pos,
        hp: t.max_hp as u16,
        shield: t.max_shields as u16,
        order: ORDER_NONE,
        flags: flags::COMPLETED,
        build_queue: [none; 5],
        resources: if t.is_resource() { 1500 } else { 0 },
        ..Default::default()
    }
}

/// Frame `i` of a synthetic game: our base with `2 + i % 5` SCVs and minerals.
pub fn frame(i: u32) -> RawFrame {
    let full = |v: bool| BitGrid::from_bytes(W, H, &vec![v as u8; W as usize * H as usize]);
    let mut units = vec![unit(1, 0, "Terran_Command_Center", (8, 8)), unit(2, 11, "Resource_Mineral_Field", (4, 8)), unit(3, 1, "Zerg_Hatchery", (50, 50))];
    for k in 0..(2 + i % 5) {
        units.push(unit(10 + k, 0, "Terran_SCV", (12 + k as i32, 12)));
    }
    RawFrame {
        frame: i * 8,
        self_state: PlayerState { minerals: 50 + i as i32 * 10, gas: 0, supply_used: 8, supply_total: 20, upgrades: vec![0; 63], tech: 0, researching: 0, upgrading: 0 },
        units,
        visibility: full(true),
        explored: full(true),
        creep: full(false),
        events: vec![],
        effects: vec![],
    }
}

/// `n` consecutive observations of one synthetic game (memory carried like a real session).
pub fn observations(n: usize) -> Vec<Observation> {
    let info = game_info();
    let mut mem = AgentMemory::new(&info);
    let map = MapInfo::new(&info, 0, None).unwrap();
    (0..n).map(|i| extract(&frame(i as u32), &mut mem, &map, None)).collect()
}
