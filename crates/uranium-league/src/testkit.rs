use uranium_env::features::{AgentMemory, MapInfo, Observation, extract};
use uranium_env::testkit::{flat_game_info, unit};
use uranium_proto::*;

const W: u16 = 64;
const H: u16 = 64;

pub fn game_info() -> GameInfo {
    flat_game_info("synthetic", (W, H), ["uranium-p0", "uranium-p1"], [(8, 8), (50, 50)])
}

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

pub fn observations(n: usize) -> Vec<Observation> {
    let info = game_info();
    let mut mem = AgentMemory::new(&info);
    let map = MapInfo::new(&info, 0, None).unwrap();
    (0..n).map(|i| extract(&frame(i as u32), &mut mem, &map, None)).collect()
}
