use crate::brain::{CELL, DROP_SIZE};
use crate::types::*;
use crate::*;

fn info(race: Race) -> GameInfo {
    let mut walk = BitGrid::new(256, 256);
    for y in 0..256 {
        for x in 0..256 {
            walk.set(x, y, true);
        }
    }
    GameInfo {
        map_name: "t".into(),
        map_hash: [0; 32],
        map_size_tiles: (64, 64),
        self_player: 0,
        players: vec![
            PlayerInfo { id: 0, name: "a".into(), race, start_tile: Some((4, 4)), is_self: true, is_enemy: false },
            PlayerInfo { id: 1, name: "b".into(), race: Race::Zerg, start_tile: None, is_self: false, is_enemy: true },
        ],
        static_map: StaticMap {
            width_tiles: 64,
            height_tiles: 64,
            walkable: walk,
            buildable: BitGrid::from_bytes(64, 64, &[1; 64 * 64]),
            ground_height: vec![0; 64 * 64],
            start_locations: vec![(4, 4), (56, 56)],
        },
        latency_frames: 0,
    }
}

fn unit(id: u32, ty: u16, pos: Pos) -> UnitState {
    UnitState { id, owner: 0, unit_type: ty, pos, flags: flags::COMPLETED | flags::IDLE, ..Default::default() }
}

fn frame(units: Vec<UnitState>, minerals: i32, creep_all: bool) -> RawFrame {
    let mut creep = BitGrid::new(64, 64);
    if creep_all {
        creep = BitGrid::from_bytes(64, 64, &[1; 64 * 64]);
    }
    RawFrame {
        frame: 100,
        self_state: PlayerState { minerals, supply_used: 8, supply_total: 18, upgrades: vec![0; 63], ..Default::default() },
        units,
        visibility: BitGrid::from_bytes(64, 64, &[1; 64 * 64]),
        explored: BitGrid::new(64, 64),
        creep,
        events: vec![],
        effects: vec![],
    }
}

#[test]
fn race_dispatch_and_worker_training() {
    let mut z = for_game(&info(Race::Zerg));
    let act = z.act(&frame(vec![unit(1, HATCHERY, (200, 200)), unit(2, LARVA, (220, 220))], 50, true));
    assert!(act.cmds.iter().any(|c| matches!(c, Command::Train { building: 2, unit_type: DRONE })), "{:?}", act.cmds);
    let mut p = for_game(&info(Race::Protoss));
    let act = p.act(&frame(vec![unit(1, NEXUS, (200, 200))], 50, false));
    assert!(act.cmds.iter().any(|c| matches!(c, Command::Train { building: 1, unit_type: PROBE })), "{:?}", act.cmds);
}

#[test]
fn protoss_gateway_goes_inside_pylon_power() {
    let mut units = vec![unit(1, NEXUS, (400, 400)), unit(2, PYLON, (700, 400))];
    units.extend((10..23).map(|i| unit(i, PROBE, (420, 460))));
    let mut p = Brain::new(&info(Race::Protoss), Kit::protoss());
    let act = p.act(&frame(units, 400, false));
    let gw = act.cmds.iter().find_map(|c| match c {
        Command::Build { unit_type: GATEWAY, tile, .. } => Some(*tile),
        _ => None,
    });
    let t = gw.expect("gateway ordered");
    let (cx, cy) = (t.0 as f64 * 32.0 + 64.0 - 700.0, t.1 as f64 * 32.0 + 48.0 - 400.0);
    assert!((cx / 224.0).powi(2) + (cy / 128.0).powi(2) <= 1.0, "{t:?}");
}

#[test]
fn hunting_targets_stalest_cell_when_nothing_known() {
    let mut b = Brain::new(&info(Race::Terran), Kit::terran());
    b.scouted.iter_mut().for_each(|s| *s = true);
    let mut f = frame(vec![], 0, false);
    f.visibility = BitGrid::new(64, 64);
    b.cell_seen.iter_mut().for_each(|c| *c = 100);
    let last = b.cell_seen.len() - 1;
    b.cell_seen[last] = 0;
    let t = b.attack_target(&f, (0, 0)).unwrap();
    assert_eq!((t.0 / 32 / CELL, t.1 / 32 / CELL), (7, 7));
}

fn enemy(id: u32, ty: u16, pos: Pos) -> UnitState {
    UnitState { owner: 1, ..unit(id, ty, pos) }
}

fn trains(act: &Act, ty: u16) -> usize {
    act.cmds.iter().filter(|c| matches!(c, Command::Train { unit_type, .. } if *unit_type == ty)).count()
}

#[test]
fn e_regression_050_scenario_needs_its_race() {
    assert_eq!(Kit::for_build(Race::Zerg, T0Build::ZergMuta).attacker, MUTALISK);
    assert_eq!(Kit::for_build(Race::Terran, T0Build::TerranDrop).tactic, Tactic::Drop);
    let k = Kit::for_build(Race::Protoss, T0Build::ZergRush);
    assert_eq!((k.race, k.army, k.attacker, k.tech.len()), (Race::Protoss, ZEALOT, ZEALOT, 0));
    let k = Kit::for_build(Race::Zerg, T0Build::Standard);
    assert_eq!((k.first_wave, k.army_cap, k.special), (12, usize::MAX, None));
}

#[test]
fn e_regression_050_zerg_rush_pools_at_five_drones() {
    let mut units = vec![unit(1, HATCHERY, (400, 400))];
    units.extend((10..15).map(|i| unit(i, DRONE, (420, 460))));
    let mut rush = Brain::new(&info(Race::Zerg), Kit::zerg_rush());
    let act = rush.act(&frame(units.clone(), 200, true));
    assert!(act.cmds.iter().any(|c| matches!(c, Command::Build { unit_type: SPAWNING_POOL, .. })), "{:?}", act.cmds);
    let mut std = Brain::new(&info(Race::Zerg), Kit::zerg());
    let act = std.act(&frame(units, 200, true));
    assert!(!act.cmds.iter().any(|c| matches!(c, Command::Build { unit_type: SPAWNING_POOL, .. })), "{:?}", act.cmds);
}

#[test]
fn e_regression_050_drop_builds_refinery_on_geyser_then_gas_crew() {
    let geyser = UnitState { owner: 11, ..unit(5, GEYSER, (560, 400)) };
    let mut units = vec![unit(1, CC, (400, 400)), unit(2, BARRACKS, (400, 560)), geyser.clone()];
    units.extend((10..22).map(|i| unit(i, SCV, (420, 460))));
    let mut b = Brain::new(&info(Race::Terran), Kit::terran_drop());
    let act = b.act(&frame(units.clone(), 100, false));
    let tile = act.cmds.iter().find_map(|c| match c {
        Command::Build { unit_type: REFINERY, tile, .. } => Some(*tile),
        _ => None,
    });
    assert_eq!(tile, Some(building_tile(&geyser)), "{:?}", act.cmds);

    units[2] = unit(5, REFINERY, (560, 400));
    let act = b.act(&frame(units.clone(), 0, false));
    let gas: std::collections::BTreeSet<UnitId> = act
        .cmds
        .iter()
        .filter_map(|c| match c {
            Command::Unit { units, order: CmdOrder::Gather(5), .. } => Some(units[0]),
            _ => None,
        })
        .collect();
    assert_eq!(gas.into_iter().collect::<Vec<_>>(), vec![10, 11, 12], "{:?}", act.cmds);
    let act = b.act(&frame(units, 0, false));
    let regather = act.cmds.iter().filter(|c| matches!(c, Command::Unit { order: CmdOrder::Gather(5), .. })).count();
    assert_eq!(regather, 3, "idle gas workers return to gas: {:?}", act.cmds);
}

#[test]
fn e_regression_050_muta_morphs_lair_then_spire_then_mutas() {
    let mut units = vec![unit(1, HATCHERY, (400, 400)), unit(2, SPAWNING_POOL, (300, 300)), unit(3, EXTRACTOR, (560, 400)), unit(4, LARVA, (420, 420))];
    units.extend((10..26).map(|i| unit(i, DRONE, (420, 460))));
    let mut b = Brain::new(&info(Race::Zerg), Kit::zerg_muta());
    let mut f = frame(units.clone(), 150, true);
    f.self_state.gas = 100;
    let act = b.act(&f);
    assert!(act.cmds.contains(&Command::Morph { unit: 1, unit_type: LAIR }), "{:?}", act.cmds);
    assert_eq!(trains(&act, ZERGLING) + trains(&act, DRONE), 0, "{:?}", act.cmds);

    units[0] = unit(1, LAIR, (400, 400));
    let mut f = frame(units.clone(), 200, true);
    f.self_state.gas = 150;
    f.frame = 200;
    let act = b.act(&f);
    assert!(act.cmds.iter().any(|c| matches!(c, Command::Build { unit_type: SPIRE, .. })), "{:?}", act.cmds);

    units.push(unit(6, SPIRE, (300, 400)));
    let mut b = Brain::new(&info(Race::Zerg), Kit::zerg_muta());
    let mut f = frame(units, 100, true);
    f.self_state.gas = 100;
    f.frame = 300;
    let act = b.act(&f);
    assert_eq!(trains(&act, MUTALISK), 1, "{:?}", act.cmds);

    let mut units = vec![unit(1, LAIR, (400, 400)), unit(2, SPAWNING_POOL, (300, 300)), unit(4, LARVA, (420, 420)), unit(6, SPIRE, (300, 400))];
    units.extend((10..26).map(|i| unit(i, DRONE, (420, 460))));
    units.push(UnitState { owner: 11, ..unit(5, GEYSER, (560, 400)) });
    let mut b = Brain::new(&info(Race::Zerg), Kit::zerg_muta());
    let mut f = frame(units, 150, true);
    f.self_state.gas = 100;
    let act = b.act(&f);
    assert!(act.cmds.iter().any(|c| matches!(c, Command::Build { unit_type: EXTRACTOR, .. })), "{:?}", act.cmds);
    assert_eq!(trains(&act, MUTALISK), 1, "{:?}", act.cmds);
}

#[test]
fn e_regression_050_dark_templar_before_zealots() {
    let mut units = vec![
        unit(1, NEXUS, (400, 400)),
        unit(2, PYLON, (600, 400)),
        unit(3, GATEWAY, (600, 500)),
        unit(4, ASSIMILATOR, (560, 300)),
        unit(5, CYBERNETICS_CORE, (700, 300)),
        unit(6, CITADEL, (700, 500)),
        unit(7, TEMPLAR_ARCHIVES, (800, 400)),
    ];
    units.extend((10..30).map(|i| unit(i, PROBE, (420, 460))));
    let mut b = Brain::new(&info(Race::Protoss), Kit::protoss_dt());
    let mut f = frame(units.clone(), 300, false);
    f.self_state.gas = 100;
    let act = b.act(&f);
    assert_eq!((trains(&act, DARK_TEMPLAR), trains(&act, ZEALOT)), (1, 0), "{:?}", act.cmds);

    units[6].flags = flags::BEING_CONSTRUCTED;
    units.extend((40..43).map(|i| unit(i, ZEALOT, (500, 500))));
    let mut f = frame(units, 300, false);
    f.self_state.gas = 100;
    let act = b.act(&f);
    assert_eq!((trains(&act, DARK_TEMPLAR), trains(&act, ZEALOT)), (0, 0), "{:?}", act.cmds);
}

#[test]
fn e_regression_050_drop_loads_marines_and_unloads_on_enemy_main() {
    let mut units = vec![unit(1, CC, (400, 400)), unit(2, DROPSHIP, (450, 450)), enemy(99, CC, (56 * 32 + 64, 56 * 32 + 48))];
    units.extend((10..20).map(|i| unit(i, MARINE, (460, 460))));
    let mut b = Brain::new(&info(Race::Terran), Kit::terran_drop());
    let act = b.act(&frame(units.clone(), 0, false));
    let squad = act.cmds.iter().find_map(|c| match c {
        Command::Unit { units, order: CmdOrder::Load(2), .. } => Some(units.clone()),
        _ => None,
    });
    let squad = squad.expect("marines ordered into the dropship");
    assert_eq!(squad.len(), DROP_SIZE);

    for u in units.iter_mut().filter(|u| squad.contains(&u.id)) {
        u.loaded_in = Some(2);
    }
    let mut f = frame(units.clone(), 0, false);
    f.frame = 116;
    let act = b.act(&f);
    assert!(act.cmds.contains(&Command::Unit { units: vec![2], order: CmdOrder::UnloadAll(Some((56 * 32 + 64, 56 * 32 + 48))), queued: false }), "{:?}", act.cmds);

    for u in units.iter_mut() {
        u.loaded_in = None;
    }
    let mut f = frame(units, 0, false);
    f.frame = 200;
    let act = b.act(&f);
    assert!(act.cmds.contains(&Command::Unit { units: vec![2], order: CmdOrder::Move(Target::Pos((400, 400))), queued: false }), "{:?}", act.cmds);
    let mut f2 = f.clone();
    f2.frame = 260;
    let act = b.act(&f2);
    assert!(act.cmds.iter().any(|c| matches!(c, Command::Unit { units, order: CmdOrder::Attack(_), .. } if units.len() == DROP_SIZE)), "{:?}", act.cmds);
}

#[test]
fn e_regression_050_guard_defends_base_but_standard_kit_has_none() {
    let mut units = vec![unit(1, NEXUS, (400, 400)), enemy(99, ZERGLING, (500, 500))];
    units.extend((40..43).map(|i| unit(i, ZEALOT, (420, 420))));
    let mut dt = Brain::new(&info(Race::Protoss), Kit::protoss_dt());
    let act = dt.act(&frame(units.clone(), 0, false));
    assert!(act.cmds.contains(&Command::Unit { units: vec![40, 41, 42], order: CmdOrder::Attack(Target::Pos((500, 500))), queued: false }), "{:?}", act.cmds);
    let mut std = Brain::new(&info(Race::Protoss), Kit::protoss());
    let act = std.act(&frame(units, 0, false));
    assert!(!act.cmds.iter().any(|c| matches!(c, Command::Unit { order: CmdOrder::Attack(_), .. })), "{:?}", act.cmds);
}

#[test]
fn e_regression_050_drop_adds_control_tower_before_dropships() {
    let mut units = vec![unit(1, CC, (400, 400)), unit(2, BARRACKS, (400, 560)), unit(3, REFINERY, (560, 400)), unit(4, FACTORY, (700, 560)), unit(5, STARPORT, (800, 400))];
    units.extend((10..30).map(|i| unit(i, SCV, (420, 460))));
    let mut b = Brain::new(&info(Race::Terran), Kit::terran_drop());
    let mut f = frame(units.clone(), 300, false);
    f.self_state.gas = 200;
    let act = b.act(&f);
    assert!(act.cmds.contains(&Command::BuildAddon { building: 5, unit_type: CONTROL_TOWER }), "{:?}", act.cmds);
    assert_eq!(trains(&act, DROPSHIP), 0, "{:?}", act.cmds);

    units[4].has_addon = true;
    units.push(unit(6, CONTROL_TOWER, (800 + 64 + 32, 400 + 16)));
    let mut b = Brain::new(&info(Race::Terran), Kit::terran_drop());
    let mut f = frame(units, 300, false);
    f.self_state.gas = 200;
    let act = b.act(&f);
    assert_eq!(trains(&act, DROPSHIP), 1, "{:?}", act.cmds);
}

#[test]
fn e_regression_050_constructing_scv_is_not_picked_to_build() {
    let geyser = UnitState { owner: 11, ..unit(5, GEYSER, (560, 400)) };
    let busy = UnitState { flags: flags::COMPLETED | flags::CONSTRUCTING, ..unit(10, SCV, (560, 420)) };
    let mut units = vec![unit(1, CC, (400, 400)), unit(2, BARRACKS, (400, 560)), geyser, busy];
    units.extend((11..22).map(|i| unit(i, SCV, (300, 300))));
    let mut b = Brain::new(&info(Race::Terran), Kit::terran_drop());
    let act = b.act(&frame(units, 100, false));
    let builder = act.cmds.iter().find_map(|c| match c {
        Command::Build { unit_type: REFINERY, worker, .. } => Some(*worker),
        _ => None,
    });
    assert!(builder.is_some_and(|w| w != 10), "{:?}", act.cmds);
}

#[test]
fn e_regression_050_drop_scout_walks_to_unscouted_start() {
    let mut units = vec![unit(1, CC, (400, 400)), unit(2, BARRACKS, (400, 560))];
    units.extend((10..20).map(|i| unit(i, SCV, (420, 460))));
    let mut b = Brain::new(&info(Race::Terran), Kit::terran_drop());
    let mut f = frame(units, 0, false);
    f.visibility = BitGrid::new(64, 64);
    let act = b.act(&f);
    assert!(act.cmds.contains(&Command::Unit { units: vec![10], order: CmdOrder::Move(Target::Pos((56 * 32 + 64, 56 * 32 + 48))), queued: false }), "{:?}", act.cmds);
}
