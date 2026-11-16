use uranium_proto::*;

use super::{MapHints, bunker_center, depot_center, within};
use crate::brain::Brain;
use crate::types::*;
use crate::util::dist2;
use crate::{Bot, Kit, T0Build};

pub const OURS: Tile = (10, 10);
pub const THEIRS: Tile = (110, 110);

pub fn info() -> GameInfo {
    let n = 128usize * 128;
    GameInfo {
        map_name: "t".into(),
        map_hash: [0; 32],
        map_size_tiles: (128, 128),
        self_player: 0,
        players: vec![
            PlayerInfo { id: 0, name: "a".into(), race: Race::Terran, start_tile: Some(OURS), is_self: true, is_enemy: false },
            PlayerInfo { id: 1, name: "b".into(), race: Race::Zerg, start_tile: None, is_self: false, is_enemy: true },
        ],
        static_map: StaticMap {
            width_tiles: 128,
            height_tiles: 128,
            walkable: BitGrid::from_bytes(512, 512, &vec![1; n * 16]),
            buildable: BitGrid::from_bytes(128, 128, &vec![1; n]),
            ground_height: vec![0; n],
            start_locations: vec![OURS, THEIRS],
        },
        latency_frames: 0,
    }
}

pub fn unit(id: u32, ty: u16, pos: Pos) -> UnitState {
    UnitState { id, owner: 0, unit_type: ty, pos, hp: 40, flags: flags::COMPLETED | flags::IDLE, ..Default::default() }
}

pub fn frame(n: u32, units: Vec<UnitState>, minerals: i32, supply_used: u16) -> RawFrame {
    RawFrame {
        frame: n,
        self_state: PlayerState { minerals, supply_used, supply_total: 60, upgrades: vec![0; 63], ..Default::default() },
        units,
        visibility: BitGrid::from_bytes(128, 128, &[1; 128 * 128]),
        explored: BitGrid::new(128, 128),
        creep: BitGrid::new(128, 128),
        events: vec![],
        effects: vec![],
    }
}

pub struct BunkerProbe {
    pub path: Vec<Tile>,
    pub natural: Pos,
    pub site: Option<Tile>,
    pub bunker: Option<(f64, bool)>,
}

pub fn main_bunker_probe(static_map: &StaticMap, hints: &MapHints, start: Tile, resources: &[(u16, Pos)]) -> BunkerProbe {
    let mut i = info();
    i.static_map = static_map.clone();
    i.map_size_tiles = (i.static_map.width_tiles, i.static_map.height_tiles);
    i.players[0].start_tile = Some(start);
    let mut b = Brain::with_hints(&i, Kit::for_build(Race::Terran, T0Build::TerranReact), hints);
    let h = depot_center(start);
    let mut units = vec![unit(1, CC, h)];
    for (k, r) in resources.iter().enumerate().filter(|(_, r)| within(r.1, h, 14)) {
        units.push(UnitState { owner: 11, ..unit(1_000 + k as u32, r.0, r.1) });
    }
    b.act(&frame(100, units, 0, 8));
    let r = b.react.as_ref().unwrap();
    let (path, site, natural) = (r.main_path.clone().unwrap(), r.main_site, depot_center(r.natural.unwrap()));
    let bunker = site.map(|s| {
        let c = bunker_center(s);
        let d = (dist2(c, h) as f64).sqrt() / 32.0;
        let on_path = path.iter().any(|t| within(c, (t.0 * 32 + 16, t.1 * 32 + 16), 2));
        (d, on_path)
    });
    BunkerProbe { path, natural, site, bunker }
}
