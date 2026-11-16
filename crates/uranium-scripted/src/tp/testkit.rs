use super::*;
use crate::react::MapHints;
use crate::{Bot, Kit, T0Build};

pub const OURS: Tile = (10, 10);
pub const THEIRS: Tile = (110, 110);
pub const NATURAL: Tile = (10, 26);
pub const DRAGOON: u16 = 66;

pub fn info(enemy: Race) -> GameInfo {
    let n = 128usize * 128;
    GameInfo {
        map_name: "t".into(),
        map_hash: [0; 32],
        map_size_tiles: (128, 128),
        self_player: 0,
        players: vec![
            PlayerInfo { id: 0, name: "a".into(), race: Race::Terran, start_tile: Some(OURS), is_self: true, is_enemy: false },
            PlayerInfo { id: 1, name: "b".into(), race: enemy, start_tile: None, is_self: false, is_enemy: true },
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

pub fn hints() -> MapHints {
    MapHints { base_locs: vec![OURS, THEIRS, NATURAL, (60, 60)], chokes: vec![(24, 10), (10, 40), (100, 100)] }
}

pub fn teacher_vs(enemy: Race) -> Brain {
    let i = info(enemy);
    Brain::with_hints(&i, Kit::for_build(Race::Terran, T0Build::TerranTp.for_lobby(&i)), &hints())
}

pub fn teacher() -> Brain {
    teacher_vs(Race::Protoss)
}

pub fn unit(id: u32, ty: u16, pos: Pos) -> UnitState {
    UnitState { id, owner: 0, unit_type: ty, pos, hp: 40, flags: flags::COMPLETED | flags::IDLE, ..Default::default() }
}

pub fn enemy(id: u32, ty: u16, pos: Pos) -> UnitState {
    UnitState { owner: 1, ..unit(id, ty, pos) }
}

pub fn building(id: u32, ty: u16, pos: Pos) -> UnitState {
    UnitState { flags: flags::BEING_CONSTRUCTED, ..unit(id, ty, pos) }
}

pub fn home() -> Pos {
    depot_center(OURS)
}

pub fn base(scvs: u32) -> Vec<UnitState> {
    let mut v = vec![
        unit(1, CC, home()),
        unit(2, BARRACKS, (home().0 + 200, home().1 + 160)),
        UnitState { owner: 11, ..unit(3, MINERAL_FIELDS[0], (home().0 - 200, home().1)) },
        unit(6, DEPOT, (home().0 - 48, home().1 - 160)),
        unit(7, DEPOT, (home().0 + 80, home().1 - 160)),
    ];
    v.extend((0..scvs).map(|i| unit(100 + i, SCV, (home().0 - 100, home().1 + 10 * i as u16))));
    v
}

pub fn cc_only(scvs: u32) -> Vec<UnitState> {
    let mut v = vec![unit(1, CC, home()), UnitState { owner: 11, ..unit(3, MINERAL_FIELDS[0], (home().0 - 200, home().1)) }];
    v.extend((0..scvs).map(|i| unit(100 + i, SCV, (home().0 - 100, home().1 + 10 * i as u16))));
    v
}

pub const GEYSER_AT: Pos = (10 * 32 + 64 + 7 * 32, 10 * 32 + 48 - 3 * 32);
pub const FACTORY_AT: Pos = (10 * 32 + 64 + 320, 10 * 32 + 48 + 96);

pub fn geyser() -> UnitState {
    UnitState { owner: 11, ..unit(5, GEYSER, GEYSER_AT) }
}

pub fn factory() -> UnitState {
    unit(8, FACTORY, FACTORY_AT)
}

pub fn shop() -> UnitState {
    unit(9, MACHINE_SHOP, (FACTORY_AT.0 + 96, FACTORY_AT.1 + 16))
}

pub fn frame(n: u32, units: Vec<UnitState>, minerals: i32, supply_used: u16) -> RawFrame {
    RawFrame {
        frame: n,
        self_state: PlayerState { minerals, supply_used, supply_total: 200, upgrades: vec![0; 63], ..Default::default() },
        units,
        visibility: BitGrid::from_bytes(128, 128, &[1; 128 * 128]),
        explored: BitGrid::new(128, 128),
        creep: BitGrid::new(128, 128),
        events: vec![],
        effects: vec![],
    }
}

pub fn gas(mut f: RawFrame, gas: i32) -> RawFrame {
    f.self_state.gas = gas;
    f
}

pub fn sieging(mut f: RawFrame, researched: bool, researching: bool) -> RawFrame {
    let bit = 1u64 << TECH_TANK_SIEGE;
    f.self_state.tech = if researched { bit } else { 0 };
    f.self_state.researching = if researching { bit } else { 0 };
    f
}

pub fn dark(f: RawFrame) -> RawFrame {
    RawFrame { visibility: BitGrid::new(128, 128), ..f }
}

pub fn scenes() -> Vec<(RawFrame, Act)> {
    let mut out = Vec::new();
    let mut run = |b: &mut Brain, f: RawFrame| {
        let act = b.act(&f);
        out.push((f, act));
    };
    let mut b = teacher();
    run(&mut b, dark(frame(1, cc_only(4), 50, 8)));
    run(&mut b, dark(frame(508, cc_only(9), 200, 18)));
    run(&mut b, dark(frame(516, cc_only(9), 100, 18)));
    let mut u = cc_only(10);
    u.push(unit(6, DEPOT, (home().0, home().1 - 160)));
    run(&mut teacher(), dark(frame(1_500, u, 200, 22)));
    let mut u = cc_only(12);
    u.push(unit(6, DEPOT, (home().0, home().1 - 160)));
    u.push(building(2, BARRACKS, (home().0 + 200, home().1 + 160)));
    u.push(geyser());
    run(&mut teacher(), dark(frame(2_000, u, 300, 24)));
    let mut u = base(12);
    u.extend((0..2).map(|i| enemy(90 + i, ZEALOT, depot_center((60, 60 + i as u16)))));
    run(&mut teacher(), frame(3_000, u, 400, 30));
    let mut u = base(14);
    u.push(unit(5, REFINERY, GEYSER_AT));
    run(&mut teacher(), gas(frame(2_400, u.clone(), 300, 30), 100));
    u.push(factory());
    run(&mut teacher(), gas(frame(3_024, u.clone(), 100, 30), 100));
    u.pop();
    u.push(UnitState { has_addon: true, ..factory() });
    u.push(shop());
    run(&mut teacher(), gas(frame(3_504, u.clone(), 200, 30), 200));
    run(&mut teacher(), sieging(gas(frame(3_600, u.clone(), 300, 30), 200), false, true));
    let mut d = teacher();
    let bunker = (home().0 + 5 * 32, home().1 + 64);
    let mut u = base(20);
    u.push(UnitState { hp: 300, ..unit(60, BUNKER, bunker) });
    u.extend((0..6).map(|i| unit(200 + i, MARINE, (home().0 + 40, home().1 + 100 + 8 * i as u16))));
    u.push(unit(300, SIEGE_TANK, (home().0 + 2 * 32, home().1 + 3 * 32)));
    u.push(enemy(90, ZEALOT, (bunker.0, bunker.1 + 64)));
    u.push(enemy(91, ZEALOT, (bunker.0 + 20, bunker.1 + 64)));
    run(&mut d, sieging(frame(4_032, u, 0, 60), true, false));
    let mut h = teacher();
    let mut u = base(12);
    u.push(UnitState { hp: 350, ..unit(60, BUNKER, bunker) });
    let spot = toward(bunker, home(), 2.0);
    u.extend((0..7).map(|i| unit(200 + i, MARINE, (spot.0, spot.1 + i as u16))));
    u.push(enemy(90, ZEALOT, (home().0 + 8 * 32, home().1 - 3 * 32)));
    u.push(enemy(91, ZEALOT, (home().0 + 8 * 32, home().1 - 3 * 32 + 20)));
    u.push(enemy(92, ZEALOT, (home().0 + 8 * 32, home().1 - 3 * 32 + 40)));
    u.push(enemy(93, ZEALOT, (home().0 + 8 * 32, home().1 - 3 * 32 + 60)));
    u.iter_mut().filter(|x| x.unit_type == MARINE).take(4).for_each(|x| x.loaded_in = Some(60));
    run(&mut h, frame(4_032, u, 0, 60));
    let mut w = teacher();
    let mut u = base(20);
    u.extend((0..32).map(|i| unit(200 + i, MARINE, (home().0 + 100 + 4 * (i % 8) as u16, home().1 + 100 + 4 * (i / 8) as u16))));
    u.extend((0..3).map(|i| unit(300 + i, SIEGE_TANK, (home().0 + 80, home().1 + 120 + 8 * i as u16))));
    u.push(unit(303, SIEGE_TANK_SIEGED, (home().0 + 80, home().1 + 160)));
    run(&mut w, sieging(frame(9_024, u.clone(), 0, 120), true, false));
    run(&mut w, sieging(frame(9_264, u, 0, 120), true, false));
    out
}
