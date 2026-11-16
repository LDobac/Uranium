use std::collections::{BTreeMap, BTreeSet};

use uranium_proto::*;

use super::testkit::*;
use super::*;
use crate::brain::Brain;
use crate::react::{Branch, bunker_center, depot_center, toward};
use crate::types::*;
use crate::util::dist2;
use crate::{Bot, Kit, T0Build};

fn builds(act: &Act, ty: u16) -> Vec<Tile> {
    act.cmds
        .iter()
        .filter_map(|c| match c {
            Command::Build { unit_type, tile, .. } if *unit_type == ty => Some(*tile),
            _ => None,
        })
        .collect()
}

fn trains(act: &Act, ty: u16) -> usize {
    act.cmds.iter().filter(|c| matches!(c, Command::Train { unit_type, .. } if *unit_type == ty)).count()
}

fn ordered(act: &Act, pick: impl Fn(&CmdOrder) -> bool) -> BTreeSet<UnitId> {
    act.cmds
        .iter()
        .filter_map(|c| match c {
            Command::Unit { units, order, .. } if pick(order) => Some(units.clone()),
            _ => None,
        })
        .flatten()
        .collect()
}

fn branch(b: &Brain) -> Branch {
    b.react_branch().unwrap()
}

fn stats(b: &Brain) -> TpStats {
    b.tp_stats().unwrap()
}

#[test]
fn fixture_ids_match_the_type_tables_013_r1() {
    assert_eq!(uranium_env::types::TypeTables::get().unit_by_name("Protoss_Dragoon").map(|u| u.id as u16), Some(DRAGOON));
}

#[test]
fn rush_rules_022_r4() {
    let p = TpParams::default();
    let depots = [home()];
    let others = [THEIRS];
    let none = BTreeSet::new();
    let cx = TpContext { depots: &depots, starts: &others, scout: None, prod_seen: &none };
    let at = |n: u32, units: Vec<UnitState>| tp_rules(&p, &frame(n, units, 0, 0), 0, &cx);
    let far = depot_center((60, 60));
    let many = |ty: u16, k: u32| (0..k).map(|i| enemy(30 + i, ty, (far.0, far.1 + 8 * i as u16))).collect::<Vec<_>>();
    let gates = vec![enemy(20, GATEWAY, far), UnitState { owner: 1, ..building(21, GATEWAY, (far.0 + 160, far.1)) }];
    assert_eq!(at(3_839, gates.clone()), vec![TpRule::Gates]);
    assert_eq!(at(3_840, gates.clone()), vec![]);
    assert_eq!(at(3_000, gates[..1].to_vec()), vec![]);
    assert_eq!(at(5_039, many(ZEALOT, 2)), vec![TpRule::Zealots]);
    assert_eq!(at(5_040, many(ZEALOT, 2)), vec![]);
    assert_eq!(at(4_000, many(ZEALOT, 1)), vec![]);
    let rax = vec![enemy(20, BARRACKS, far), UnitState { owner: 1, ..building(21, BARRACKS, (far.0 + 160, far.1)) }];
    assert_eq!(at(3_839, rax.clone()), vec![TpRule::Rax]);
    assert_eq!(at(3_840, rax.clone()), vec![]);
    assert_eq!(at(3_000, rax[..1].to_vec()), vec![]);
    assert_eq!(at(5_039, many(MARINE, 4)), vec![TpRule::Marines]);
    assert_eq!(at(5_040, many(MARINE, 4)), vec![]);
    assert_eq!(at(4_000, many(MARINE, 3)), vec![]);
    let near = |id: u32, ty: u16| enemy(id, ty, (home().0 + 19 * 32, home().1 + id as u16));
    assert_eq!(at(9_000, vec![near(9, ZEALOT), near(10, DRAGOON)]), vec![TpRule::Army]);
    assert_eq!(at(9_000, vec![near(9, MARINE), near(10, MARINE)]), vec![TpRule::Army]);
    assert_eq!(at(9_000, vec![near(9, ZEALOT)]), vec![]);
    assert_eq!(at(9_000, vec![near(9, ZEALOT), enemy(10, ZEALOT, (home().0 + 21 * 32, home().1))]), vec![]);
    assert_eq!(at(9_000, vec![near(9, PROBE), near(10, SCV), near(11, ZEALOT), near(12, PYLON)]), vec![]);
    assert_eq!(at(11_520, vec![near(9, ZEALOT), near(10, ZEALOT)]), vec![]);
    let main = depot_center(THEIRS);
    let scout_at = |tiles: u16| Some((main.0 - tiles * 32, main.1));
    let proxy = |n: u32, depot: u16, scout: Option<Pos>, seen: &BTreeSet<(Tile, u16)>| {
        let cx = TpContext { depots: &depots, starts: &others, scout, prod_seen: seen };
        tp_rules(&p, &frame(n, vec![enemy(40, depot, main)], 0, 0), 0, &cx)
    };
    assert_eq!(proxy(3_600, NEXUS, scout_at(3), &none), vec![TpRule::ProxyGate]);
    assert_eq!(proxy(5_039, NEXUS, scout_at(3), &none), vec![TpRule::ProxyGate]);
    assert_eq!(proxy(3_599, NEXUS, scout_at(3), &none), vec![], "before 2:30");
    assert_eq!(proxy(5_040, NEXUS, scout_at(3), &none), vec![], "from 3:30");
    assert_eq!(proxy(4_000, NEXUS, scout_at(4), &none), vec![], "scout 4 tiles away");
    assert_eq!(proxy(4_000, NEXUS, None, &none), vec![], "no scout");
    let gate_seen = BTreeSet::from([(THEIRS, GATEWAY)]);
    assert_eq!(proxy(4_000, NEXUS, scout_at(3), &gate_seen), vec![], "a gateway seen there");
    assert_eq!(proxy(3_600, CC, scout_at(3), &none), vec![TpRule::ProxyRax]);
    assert_eq!(proxy(3_599, CC, scout_at(3), &none), vec![]);
    assert_eq!(proxy(5_040, CC, scout_at(3), &none), vec![]);
    assert_eq!(proxy(4_000, CC, scout_at(4), &none), vec![]);
    assert_eq!(proxy(4_000, CC, scout_at(3), &BTreeSet::from([(THEIRS, BARRACKS)])), vec![]);
    assert_eq!(proxy(4_000, CC, scout_at(3), &gate_seen), vec![TpRule::ProxyRax], "a gateway is no barracks");
}

#[test]
fn proxy_rule_remembers_gateways_022_r4() {
    let main = depot_center(THEIRS);
    let scout_units = |extra: Vec<UnitState>| -> Vec<UnitState> {
        let mut u = base(12);
        u.push(UnitState { flags: flags::COMPLETED | flags::MOVING, ..unit(150, SCV, (main.0 - 64, main.1)) });
        u.push(enemy(9, NEXUS, main));
        u.extend(extra);
        u
    };
    let mut b = teacher();
    b.scout = Some(150);
    b.act(&frame(3_000, scout_units(vec![enemy(11, GATEWAY, (main.0 + 200, main.1 + 100))]), 0, 24));
    b.act(&frame(3_700, scout_units(vec![]), 0, 24));
    assert!(!stats(&b).rule_frames.contains_key(&TpRule::ProxyGate));
    assert_eq!(branch(&b), Branch::Macro);
    let mut c = teacher();
    c.scout = Some(150);
    c.act(&frame(3_700, scout_units(vec![]), 0, 24));
    assert_eq!(stats(&c).rule_frames.get(&TpRule::ProxyGate), Some(&3_700));
    assert_eq!((branch(&c), stats(&c).rush_rule), (Branch::Defence, Some(TpRule::ProxyGate)));
}

#[test]
fn opening_022_r3() {
    let act = teacher().act(&dark(frame(500, cc_only(8), 200, 16)));
    assert!(builds(&act, DEPOT).is_empty(), "8 supply: {:?}", act.cmds);
    let act = teacher().act(&dark(frame(508, cc_only(9), 200, 18)));
    assert_eq!(builds(&act, DEPOT).len(), 1, "9 supply");
    let mut u = cc_only(10);
    u.push(unit(6, DEPOT, (home().0, home().1 - 160)));
    assert!(builds(&teacher().act(&dark(frame(1_500, u.clone(), 200, 20))), BARRACKS).is_empty(), "10 supply");
    assert_eq!(builds(&teacher().act(&dark(frame(1_500, u.clone(), 200, 22))), BARRACKS).len(), 1, "11 supply");
    u.push(building(2, BARRACKS, (home().0 + 200, home().1 + 160)));
    u.push(geyser());
    let act = teacher().act(&dark(frame(2_000, u.clone(), 200, 22)));
    assert!(builds(&act, REFINERY).is_empty(), "11 supply: {:?}", act.cmds);
    let mut b = teacher();
    let act = b.act(&dark(frame(2_000, u.clone(), 200, 24)));
    assert_eq!(builds(&act, REFINERY), vec![building_tile(&geyser())], "12 supply");
    assert_eq!(stats(&b).refinery_frame, Some(2_000));
    assert!(builds(&act, DEPOT).is_empty(), "the second depot waits for 15");
    assert_eq!(builds(&teacher().act(&dark(frame(2_000, u, 200, 30))), DEPOT).len(), 1, "15 supply");

    let mut u = base(14);
    u.push(unit(5, REFINERY, GEYSER_AT));
    assert!(builds(&teacher().act(&gas(frame(2_400, u.clone(), 300, 30), 99)), FACTORY).is_empty());
    let mut b = teacher();
    let act = b.act(&gas(frame(2_400, u.clone(), 300, 30), 100));
    assert_eq!(builds(&act, FACTORY).len(), 1, "{:?}", act.cmds);
    assert_eq!(stats(&b).factory_frame, Some(2_400));
    let mut v = u.clone();
    v.push(building(8, FACTORY, FACTORY_AT));
    let act = teacher().act(&gas(frame(3_024, v, 100, 30), 100));
    assert!(!act.cmds.iter().any(|c| matches!(c, Command::BuildAddon { .. })), "{:?}", act.cmds);
    let mut v = u.clone();
    v.push(factory());
    let mut b = teacher();
    let act = b.act(&gas(frame(3_024, v, 100, 30), 100));
    assert!(act.cmds.contains(&Command::BuildAddon { building: 8, unit_type: MACHINE_SHOP }), "{:?}", act.cmds);
    assert_eq!(stats(&b).shop_frame, Some(3_024));
    let mut v = u.clone();
    v.push(UnitState { has_addon: true, ..factory() });
    v.push(building(9, MACHINE_SHOP, shop().pos));
    let act = teacher().act(&gas(frame(3_504, v.clone(), 200, 30), 200));
    assert!(!act.cmds.iter().any(|c| matches!(c, Command::Research { .. })), "{:?}", act.cmds);
    let last = v.len() - 1;
    v[last] = shop();
    let mut b = teacher();
    let act = b.act(&gas(frame(3_504, v.clone(), 200, 30), 200));
    assert!(act.cmds.contains(&Command::Research { building: 9, tech: TECH_TANK_SIEGE }), "{:?}", act.cmds);
    assert_eq!(trains(&act, SIEGE_TANK), 0, "tanks after the research order");
    assert_eq!(stats(&b).siege_frame, Some(3_504));
    let act = teacher().act(&sieging(gas(frame(3_504, v.clone(), 200, 30), 200), false, true));
    assert!(!act.cmds.iter().any(|c| matches!(c, Command::Research { .. })), "researching: {:?}", act.cmds);
    let mut b = teacher();
    let act = b.act(&sieging(gas(frame(3_600, v, 300, 30), 200), false, true));
    assert!(act.cmds.contains(&Command::Train { building: 8, unit_type: SIEGE_TANK }), "{:?}", act.cmds);
    assert_eq!(stats(&b).tank_frame, Some(3_600));

    let marines = |k: u32| (0..k).map(|i| unit(200 + i, MARINE, (home().0 + 100, home().1 + 4 * i as u16))).collect::<Vec<_>>();
    let mut u = base(14);
    u.extend(marines(4));
    assert_eq!(trains(&teacher().act(&frame(2_400, u, 1_000, 36)), MARINE), 0);
    let mut u = base(14);
    u.extend(marines(3));
    assert_eq!(trains(&teacher().act(&frame(2_400, u, 1_000, 34)), MARINE), 1);
    let mut u = base(14);
    u.extend(marines(4));
    u.push(factory());
    assert_eq!(trains(&teacher().act(&frame(3_000, u, 1_000, 36)), MARINE), 1);
}

#[test]
fn no_main_bunker_without_rush_evidence_022_r5() {
    let mut b = teacher();
    let (mut bunkers, mut expanded) = (0, false);
    for n in (8..=8_640).step_by(8) {
        let scvs = (4 + n / 300).min(22);
        let mut u = if n < 1_600 { cc_only(scvs) } else { base(scvs) };
        if n >= 2_400 {
            u.push(building(8, FACTORY, FACTORY_AT));
        }
        if n >= 7_200 {
            u.extend((0..3).map(|i| unit(300 + i, SIEGE_TANK, (home().0 + 60, home().1 + 100 + 8 * i as u16))));
        }
        if (2_800..3_100).contains(&n) {
            let m = depot_center(THEIRS);
            u.push(enemy(80, NEXUS, m));
            u.push(enemy(81, GATEWAY, (m.0 + 200, m.1 + 100)));
            u.extend((0..8).map(|i| enemy(82 + i, PROBE, (m.0 - 100, m.1 + 8 * i as u16))));
        }
        if n >= 4_000 {
            u.push(enemy(95, PROBE, (home().0 + 64, home().1 + 64)));
        }
        let act = b.act(&gas(frame(n, u, 600, (2 * scvs) as u16), if n >= 1_600 { 150 } else { 0 }));
        bunkers += builds(&act, BUNKER).len();
        expanded |= builds(&act, CC).contains(&NATURAL);
    }
    assert_eq!((bunkers, branch(&b), stats(&b).rush_frame), (0, Branch::Macro, None));
    assert!(expanded && stats(&b).expand_frame.is_some() && stats(&b).factory_frame < stats(&b).expand_frame);
}

#[test]
fn rush_switches_to_defence_022_r6() {
    for (enemy_race, ty, k, rule) in [(Race::Protoss, ZEALOT, 2, TpRule::Zealots), (Race::Terran, MARINE, 4, TpRule::Marines)] {
        let mut b = teacher_vs(enemy_race);
        let mut u = base(12);
        u.extend((0..k).map(|i| enemy(90 + i, ty, depot_center((60, 60 + i as u16)))));
        let act = b.act(&frame(3_000, u, 250, 30));
        assert_eq!(branch(&b), Branch::Defence);
        let bunker = builds(&act, BUNKER);
        assert_eq!(bunker.len(), 1, "{:?}", act.cmds);
        let c = bunker_center(bunker[0]);
        let d = ((dist2(c, home()) as f64).sqrt() / 32.0, c.1 as i64 - home().1 as i64);
        assert!(d.0 >= 5.0 && d.0 <= 7.0 && d.1 > 0, "{bunker:?} {d:?}");
        assert_eq!(builds(&act, BARRACKS).len(), 1, "the second barracks: {:?}", act.cmds);
        let s = stats(&b);
        assert_eq!((s.rush_frame, s.rush_rule, s.defences, s.branch), (Some(3_000), Some(rule), 1, Branch::Defence));
    }
}

#[test]
fn defence_returns_to_macro_022_r6() {
    let army = |marines: u32, tanks: u32| {
        let mut u = base(12);
        u.extend((0..marines).map(|i| unit(200 + i, MARINE, (home().0 + 100, home().1 + 4 * i as u16))));
        u.extend((0..tanks).map(|i| unit(300 + i, SIEGE_TANK, (home().0 + 60, home().1 + 100 + 8 * i as u16))));
        u
    };
    let zealots = |mut u: Vec<UnitState>| {
        u.push(enemy(90, ZEALOT, (home().0 + 9 * 32, home().1)));
        u.push(enemy(91, ZEALOT, (home().0 + 9 * 32, home().1 + 20)));
        u
    };
    let mut b = teacher();
    b.act(&frame(6_000, zealots(army(2, 0)), 0, 40));
    assert_eq!(branch(&b), Branch::Defence, "③ army");
    b.act(&frame(6_000 + 1_439, army(8, 2), 0, 40));
    assert_eq!(branch(&b), Branch::Defence, "59.96 s");
    b.act(&frame(6_000 + 1_440, army(7, 2), 0, 40));
    assert_eq!(branch(&b), Branch::Defence, "7 marines");
    b.act(&frame(6_000 + 1_440, army(8, 1), 0, 40));
    assert_eq!(branch(&b), Branch::Defence, "1 tank");
    let mut loaded = army(8, 2);
    loaded.iter_mut().filter(|u| u.unit_type == MARINE).take(4).for_each(|u| u.loaded_in = Some(77));
    b.act(&frame(6_000 + 1_440, loaded, 0, 40));
    assert_eq!(branch(&b), Branch::Macro, "8 marines (4 loaded), 2 tanks, 60 s");
    assert_eq!(stats(&b).macro_frame, Some(7_440));
    let mut c = teacher();
    c.act(&frame(6_000, zealots(army(0, 3)), 0, 40));
    assert_eq!(branch(&c), Branch::Defence);
    c.act(&frame(6_048, zealots(army(0, 4)), 0, 40));
    assert_eq!(branch(&c), Branch::Macro, "4 tanks");
    c.act(&frame(6_096, zealots(army(0, 4)), 0, 40));
    assert_eq!((branch(&c), stats(&c).defences), (Branch::Macro, 1), "R with 4 tanks keeps M");
}

#[test]
fn natural_expansion_022_r5() {
    let with = |extra: Vec<UnitState>| {
        let mut u = base(16);
        u.extend(extra);
        u
    };
    let tanks = |k: u32| (0..k).map(|i| unit(300 + i, SIEGE_TANK, (home().0 + 60, home().1 + 100 + 8 * i as u16))).collect::<Vec<_>>();
    assert!(builds(&teacher().act(&frame(3_000, with(tanks(3)), 400, 40)), CC).is_empty(), "no factory");
    let fac_tanks = |k: u32| [vec![factory()], tanks(k)].concat();
    assert!(builds(&teacher().act(&frame(5_000, with(fac_tanks(2)), 400, 40)), CC).is_empty(), "2 tanks");
    let mut b = teacher();
    let act = b.act(&frame(5_000, with(fac_tanks(3)), 400, 40));
    assert_eq!(builds(&act, CC), vec![NATURAL], "{:?}", act.cmds);
    assert_eq!(stats(&b).expand_frame, Some(5_000));
    let act = teacher().act(&frame(5_000, with(fac_tanks(3)), 399, 40));
    assert!(builds(&act, CC).is_empty());
    let first = TpParams { expand_tanks_p: 0, ..TpParams::default() };
    let mut d0 = Brain::with_hints(&info(Race::Protoss), Kit { tp: Some(first.clone()), react: Some(first.react_params()), ..Kit::terran_tp() }, &hints());
    assert_eq!(builds(&d0.act(&frame(3_000, with(vec![building(8, FACTORY, FACTORY_AT)]), 400, 40)), CC), vec![NATURAL]);
    assert_eq!(trains(&teacher().act(&frame(5_759, with(fac_tanks(3)), 399, 40)), SCV), 1);
    assert_eq!(trains(&teacher().act(&frame(5_760, with(fac_tanks(3)), 399, 40)), SCV), 0);
    let act = teacher_vs(Race::Terran).act(&frame(5_000, with([vec![factory()], tanks(1)].concat()), 400, 40));
    assert!(builds(&act, CC).is_empty(), "{:?}", act.cmds);
    let act = teacher_vs(Race::Terran).act(&frame(5_000, with([vec![factory()], tanks(2)].concat()), 400, 40));
    assert_eq!(builds(&act, CC), vec![NATURAL]);
    let mut d = teacher();
    let mut u = with(tanks(2));
    u.push(enemy(90, ZEALOT, (home().0 + 9 * 32, home().1)));
    u.push(enemy(91, ZEALOT, (home().0 + 9 * 32, home().1 + 20)));
    let act = d.act(&frame(6_000, u, 400, 40));
    assert_eq!(branch(&d), Branch::Defence);
    assert!(builds(&act, CC).is_empty());
    assert!(builds(&d.act(&frame(7_439, with(tanks(2)), 650, 40)), CC).is_empty());
    assert_eq!(builds(&d.act(&frame(7_440, with(tanks(2)), 650, 40)), CC), vec![NATURAL]);
    assert_eq!(branch(&d), Branch::Defence, "no marines: still D");
}

#[test]
fn siege_rule_022_r7() {
    let tank_at = (home().0 + 15 * 32, home().1);
    let at = |n: u32, tank: UnitState, extra: Vec<UnitState>, researched: bool| {
        let mut b = teacher();
        let mut u = base(12);
        u.push(tank);
        u.extend(extra);
        b.act(&sieging(frame(n, u, 0, 40), researched, false))
    };
    let siege = |act: &Act| ordered(act, |o| *o == CmdOrder::Siege);
    let tank = unit(300, SIEGE_TANK, tank_at);
    assert_eq!(siege(&at(6_000, tank.clone(), vec![enemy(90, ZEALOT, (tank_at.0 + 11 * 32, tank_at.1))], true)), BTreeSet::from([300]));
    assert!(siege(&at(6_000, tank.clone(), vec![enemy(90, ZEALOT, (tank_at.0 + 12 * 32, tank_at.1))], true)).is_empty());
    assert!(siege(&at(6_000, tank.clone(), vec![enemy(90, PROBE, (tank_at.0 + 5 * 32, tank_at.1))], true)).is_empty(), "a worker");
    assert!(siege(&at(6_000, tank.clone(), vec![enemy(90, ZEALOT, (tank_at.0 + 5 * 32, tank_at.1))], false)).is_empty(), "not researched");

    let sieged = unit(300, SIEGE_TANK_SIEGED, tank_at);
    let run = |b: &mut Brain, n: u32, extra: Vec<UnitState>| {
        let mut u = base(12);
        u.push(sieged.clone());
        u.extend(extra);
        b.act(&sieging(frame(n, u, 0, 40), true, false))
    };
    let unsiege = |act: &Act| ordered(act, |o| *o == CmdOrder::Unsiege);
    let mut b = teacher();
    assert!(unsiege(&run(&mut b, 6_000, vec![])).is_empty());
    assert!(unsiege(&run(&mut b, 6_239, vec![])).is_empty());
    assert_eq!(unsiege(&run(&mut b, 6_240, vec![])), BTreeSet::from([300]));
    let mut c = teacher();
    run(&mut c, 6_000, vec![]);
    run(&mut c, 6_100, vec![enemy(90, ZEALOT, (tank_at.0 + 13 * 32, tank_at.1))]);
    assert!(unsiege(&run(&mut c, 6_240, vec![])).is_empty());
    assert_eq!(unsiege(&run(&mut c, 6_340, vec![])), BTreeSet::from([300]));
    let mut d = teacher();
    let plain = base(12);
    d.act(&frame(5_000, plain.clone(), 0, 40));
    let spot = d.tp_tank_spot(&plain.iter().collect::<Vec<_>>()).unwrap();
    let mut u = base(12);
    u.push(unit(301, SIEGE_TANK_SIEGED, spot));
    d.act(&sieging(frame(6_000, u.clone(), 0, 40), true, false));
    assert!(unsiege(&d.act(&sieging(frame(6_480, u, 0, 40), true, false))).is_empty());
}

#[test]
fn tanks_wait_behind_the_bunker_022_r6() {
    let mut b = teacher();
    let mut u = base(12);
    let bunker = (home().0 + 5 * 32, home().1 + 64);
    u.push(UnitState { hp: 350, ..unit(60, BUNKER, bunker) });
    u.push(unit(300, SIEGE_TANK, (home().0 - 6 * 32, home().1 - 4 * 32)));
    u.push(enemy(90, ZEALOT, depot_center((60, 60))));
    u.push(enemy(91, ZEALOT, depot_center((60, 61))));
    let act = b.act(&frame(3_024, u, 0, 40));
    assert_eq!(branch(&b), Branch::Defence);
    let spot = toward(bunker, home(), 3.0);
    assert!(act.cmds.contains(&Command::Unit { units: vec![300], order: CmdOrder::Move(Target::Pos(spot)), queued: false }), "{:?}", act.cmds);
}

#[test]
fn wave_condition_022_r8() {
    let army = |marines: u32, tanks: u32| {
        let mut u = base(20);
        u.extend((0..marines).map(|i| unit(200 + i, MARINE, (home().0 + 100 + 4 * (i % 8) as u16, home().1 + 100 + 4 * (i / 8) as u16))));
        u.extend((0..tanks).map(|i| unit(300 + i, SIEGE_TANK, (home().0 + 80, home().1 + 120 + 8 * i as u16))));
        u
    };
    let attackers = |n: u32, marines: u32, tanks: u32| {
        let mut b = teacher();
        let act = b.act(&sieging(frame(n, army(marines, tanks), 0, 120), true, false));
        (ordered(&act, |o| matches!(o, CmdOrder::Attack(Target::Pos(_)))).len(), stats(&b).wave_frame)
    };
    assert_eq!(attackers(9_024, 32, 4), (36, Some(9_024)));
    assert_eq!(attackers(9_024, 31, 4), (0, None), "39 supply");
    assert_eq!(attackers(9_024, 34, 3), (0, None), "3 tanks");
    assert_eq!(attackers(17_280, 20, 0), (20, Some(17_280)));
    assert_eq!(attackers(17_279, 20, 0), (0, None), "before 12:00");
    assert_eq!(attackers(17_280, 19, 0), (0, None), "19 supply");
    let mut b = teacher();
    let mut u = army(32, 4);
    let target = depot_center(THEIRS);
    u.push(enemy(80, NEXUS, target));
    u.iter_mut().filter(|x| x.id == 200).for_each(|x| x.pos = (home().0 + 30 * 32, home().1 + 30 * 32));
    let act = b.act(&sieging(frame(9_024, u, 0, 120), true, false));
    let goal = snap(target);
    let lead = (home().0 + 80, home().1 + 120 + 12);
    let front = snap(toward(lead, goal, 6.0));
    assert!(ordered(&act, |o| *o == CmdOrder::Attack(Target::Pos(front))).contains(&200), "{:?}", act.cmds);
    assert!(ordered(&act, |o| *o == CmdOrder::Attack(Target::Pos(goal))).contains(&300));
}

#[test]
fn hunting_squads_022_r8() {
    let mut u = base(20);
    u.extend((0..32).map(|i| unit(200 + i, MARINE, (home().0 + 100 + 4 * (i % 8) as u16, home().1 + 100 + 4 * (i / 8) as u16))));
    u.extend((0..4).map(|i| unit(300 + i, SIEGE_TANK, (home().0 + 80, home().1 + 120 + 8 * i as u16))));
    let mut b = teacher();
    b.act(&sieging(frame(9_024, u.clone(), 0, 120), true, false));
    let act = b.act(&sieging(dark(frame(9_072, u, 0, 120)), true, false));
    let cells = b.stale_cells();
    let to = |id: UnitId| {
        act.cmds.iter().find_map(|c| match c {
            Command::Unit { units, order: CmdOrder::Attack(Target::Pos(p)), .. } if units.contains(&id) => Some(*p),
            _ => None,
        })
    };
    assert_eq!((to(300), to(200), to(212), to(224)), (Some(cells[0]), Some(cells[0]), Some(cells[1]), Some(cells[2])), "{:?}", act.cmds);
}

#[test]
fn home_guard_owns_marines_during_an_attack_022_r8() {
    let mut u = base(20);
    u.extend((0..32).map(|i| unit(200 + i, MARINE, (home().0 + 100 + 4 * (i % 8) as u16, home().1 + 100 + 4 * (i / 8) as u16))));
    u.extend((0..4).map(|i| unit(300 + i, SIEGE_TANK, (home().0 + 80, home().1 + 120 + 8 * i as u16))));
    let zealot = (home().0 + 15 * 32, home().1);
    u.push(enemy(90, ZEALOT, zealot));
    let target = depot_center(THEIRS);
    u.push(enemy(80, NEXUS, target));
    let mut b = teacher();
    let act = b.act(&sieging(frame(9_024, u, 0, 120), true, false));
    assert_eq!(stats(&b).wave_frame, Some(9_024));
    let mut orders: BTreeMap<UnitId, Vec<CmdOrder>> = BTreeMap::new();
    for c in &act.cmds {
        if let Command::Unit { units, order, .. } = c {
            for id in units {
                orders.entry(*id).or_default().push(order.clone());
            }
        }
    }
    for id in 200..232 {
        assert_eq!(orders.get(&id), Some(&vec![CmdOrder::Attack(Target::Pos(snap(zealot)))]), "marine {id}: {:?}", act.cmds);
    }
    for id in 300..304 {
        assert_eq!(orders.get(&id), Some(&vec![CmdOrder::Attack(Target::Pos(snap(target)))]), "tank {id}: {:?}", act.cmds);
    }
}

#[test]
fn defence_marines_hold_vs_protoss_022_r6() {
    let run = |race: Race, ty: u16| {
        let mut b = teacher_vs(race);
        b.scout_done = true;
        let mut u = base(12);
        u.extend((0..6).map(|n| unit(200 + n, MARINE, (home().0 + 100, home().1 + 100 + 4 * n as u16))));
        u.push(enemy(90, ty, (home().0 + 8 * 32, home().1)));
        u.push(enemy(91, ty, (home().0 + 8 * 32, home().1 + 20)));
        b.act(&frame(6_000, u, 0, 40));
        (branch(&b), b.react_marines_attack())
    };
    assert_eq!(run(Race::Protoss, ZEALOT), (Branch::Defence, false));
    assert_eq!(run(Race::Terran, MARINE), (Branch::Defence, true));
}

#[test]
fn damaged_bunker_gets_repairers_022_r6() {
    let tp = TpParams { repairers: 5, ..TpParams::default() };
    let mut b = Brain::with_hints(&info(Race::Protoss), Kit { react: Some(tp.react_params()), tp: Some(tp), ..Kit::terran_tp() }, &hints());
    b.scout_done = true;
    let mut u = base(20);
    let bunker = (home().0 + 5 * 32, home().1 + 64);
    u.push(UnitState { hp: 120, ..unit(60, BUNKER, bunker) });
    u.extend((0..5).map(|i| enemy(90 + i, ZEALOT, (bunker.0 + 40, bunker.1 + 10 * i as u16))));
    let act = b.act(&frame(7_008, u, 100, 40));
    assert_eq!(branch(&b), Branch::Defence);
    let repair = ordered(&act, |o| *o == CmdOrder::Repair(60));
    assert_eq!(repair.len(), 5, "{:?}", act.cmds);
}

#[test]
fn scout_stays_in_the_enemy_main_022_r3() {
    let main = depot_center(THEIRS);
    let units = |hp: u16| {
        let mut u = base(12);
        u.push(UnitState { hp, flags: flags::COMPLETED | flags::MOVING, ..unit(150, SCV, (main.0 - 5 * 32, main.1)) });
        u.push(enemy(9, NEXUS, main));
        u
    };
    let moves = |act: &Act| {
        act.cmds
            .iter()
            .filter_map(|c| match c {
                Command::Unit { units, order: CmdOrder::Move(Target::Pos(p)), .. } if units == &vec![150] => Some(*p),
                _ => None,
            })
            .collect::<Vec<_>>()
    };
    let gathers = |act: &Act| ordered(act, |o| matches!(o, CmdOrder::Gather(_))).contains(&150);
    let mut b = teacher();
    b.scout = Some(150);
    let act = b.act(&frame(2_400, units(60), 0, 30));
    let m = moves(&act);
    assert_eq!(m.len(), 1, "{:?}", act.cmds);
    let d = (dist2(m[0], main) as f64).sqrt() / 32.0;
    assert!((5.5..=6.5).contains(&d), "circling 6 tiles out: {m:?}");
    let act = b.act(&frame(3_600, units(60), 0, 30));
    assert_eq!(moves(&act), vec![main], "from 2:30 at the nexus");
    let act = b.act(&frame(3_840, units(60), 0, 30));
    assert!(gathers(&act) && b.scout_done, "2:40: {:?}", act.cmds);
    let mut c = teacher();
    c.scout = Some(150);
    c.act(&frame(2_400, units(30), 0, 30));
    assert!(!c.scout_done, "30 HP is half");
    let act = c.act(&frame(2_448, units(29), 0, 30));
    assert!(gathers(&act) && c.scout_done, "{:?}", act.cmds);
}

#[test]
fn shadow_fixups_022_r9() {
    let rush = |n: u32| {
        let mut u = base(12);
        u.iter_mut().filter(|x| x.unit_type == SCV).for_each(|x| x.flags = flags::COMPLETED | flags::GATHERING);
        u.push(enemy(90, ZEALOT, depot_center((60, 60))));
        u.push(enemy(91, ZEALOT, depot_center((60, 61))));
        frame(n, u, 100, 30)
    };
    let mut b = teacher();
    b.scout_done = true;
    assert_eq!(builds(&b.act(&rush(3_000)), BUNKER).len(), 1);
    assert!(builds(&b.act(&rush(3_143)), BUNKER).is_empty());
    assert_eq!(builds(&b.act(&rush(3_144)), BUNKER).len(), 1);

    let mut u = base(14);
    u.push(UnitState { has_addon: true, flags: flags::COMPLETED | flags::LIFTED, ..factory() });
    u.push(shop());
    assert_eq!(trains(&teacher().act(&sieging(gas(frame(4_000, u, 500, 30), 300), true, false)), SIEGE_TANK), 0);

    let mut u = base(14);
    u.push(unit(5, REFINERY, GEYSER_AT));
    let mut g = teacher();
    g.scout_done = true;
    let crew = |act: &Act| ordered(act, |o| *o == CmdOrder::Gather(5));
    assert!(crew(&g.act(&frame(3_000, u.clone(), 0, 30))).is_empty(), "just complete");
    assert!(crew(&g.act(&frame(3_024, u.clone(), 0, 30))).is_empty(), "complete 24 frames");
    let first = crew(&g.act(&frame(3_072, u.clone(), 0, 30)));
    assert_eq!(first.len(), 3);
    assert!(crew(&g.act(&frame(3_080, u.clone(), 0, 30))).is_empty(), "not a window start");
    assert_eq!(crew(&g.act(&frame(3_120, u.clone(), 0, 30))), first, "not on gas: chosen again");
    let harvest = uranium_env::types::order_id("HarvestGas").unwrap();
    for x in u.iter_mut().filter(|x| first.contains(&x.id)) {
        x.order = harvest;
    }
    assert!(crew(&g.act(&frame(3_216, u.clone(), 0, 30))).is_empty(), "three on gas");
    assert_eq!(g.tp_gas_crew().len(), 3);
    let one = *first.iter().next().unwrap();
    let mut leaving = u.clone();
    leaving.iter_mut().filter(|x| x.id == one).for_each(|x| x.order = 0);
    assert!(crew(&g.act(&frame(3_264, leaving, 0, 30))).is_empty(), "leaving the refinery");
    let away: Vec<UnitState> = u.iter().filter(|x| x.id != one).cloned().collect();
    assert!(crew(&g.act(&frame(3_272, away, 0, 30))).is_empty(), "out of sight");
    assert_eq!(g.tp_gas_crew().len(), 3);
    let mining = uranium_env::types::order_id("MiningMinerals").unwrap();
    u.iter_mut().filter(|x| x.id == one).for_each(|x| x.order = mining);
    let sent = crew(&g.act(&frame(3_312, u.clone(), 0, 30)));
    assert_eq!((g.tp_gas_crew().len(), sent.len()), (3, 1), "mining minerals: replaced");
}

#[test]
fn idle_scvs_from_the_observation_022_ac3() {
    let gathers = |act: &Act| ordered(act, |o| matches!(o, CmdOrder::Gather(_)));
    let mut b = teacher();
    b.scout_done = true;
    let mut u = base(12);
    u.iter_mut().filter(|x| x.unit_type == SCV).for_each(|x| x.flags = flags::COMPLETED | flags::GATHERING);
    u.iter_mut().filter(|x| x.id == 100).for_each(|x| x.flags = flags::COMPLETED | flags::IDLE);
    assert!(!gathers(&b.act(&frame(3_000, u.clone(), 0, 40))).contains(&100), "just idle");
    assert!(!gathers(&b.act(&frame(3_008, u.clone(), 0, 40))).contains(&100));
    assert!(gathers(&b.act(&frame(3_016, u.clone(), 0, 40))).contains(&100), "idle 16 frames");
    let mut r = Brain::with_hints(&info(Race::Zerg), Kit::for_build(Race::Terran, T0Build::TerranTp.for_lobby(&info(Race::Zerg))), &hints());
    r.scout_done = true;
    assert!(gathers(&r.act(&frame(3_000, u, 0, 40))).contains(&100), "021-R8: at once");

    let depot = |b: &Brain| b.pending.iter().filter(|q| q.ty == DEPOT).map(|q| (q.worker, q.issued)).collect::<Vec<_>>();
    let mut d = teacher();
    d.scout_done = true;
    let mut u = base(12);
    u.iter_mut().filter(|x| x.unit_type == SCV).for_each(|x| x.flags = flags::COMPLETED | flags::GATHERING);
    let mut f = frame(3_000, u.clone(), 200, 40);
    f.self_state.supply_total = 40;
    d.act(&f);
    let first = depot(&d);
    assert_eq!(first.len(), 1, "a depot ordered");
    let w = first[0].0;
    let at = |n: u32, flags_w: u32| {
        let mut v = u.clone();
        v.iter_mut().filter(|x| x.id == w).for_each(|x| x.flags = flags_w);
        let mut f = frame(n, v, 200, 40);
        f.self_state.supply_total = 40;
        f
    };
    d.act(&at(3_008, flags::COMPLETED | flags::IDLE));
    d.act(&at(3_016, flags::COMPLETED | flags::IDLE));
    assert_eq!(depot(&d), first, "idle 8 frames: kept");
    let site = d.pending.iter().find(|q| q.ty == DEPOT).unwrap().tile;
    d.act(&at(3_024, flags::COMPLETED | flags::IDLE));
    assert!(depot(&d).iter().all(|(_, issued)| *issued == 3_024), "idle 16 frames: ordered again {:?}", depot(&d));
    assert!(d.pending.iter().filter(|q| q.ty == DEPOT).all(|q| q.tile != site), "{site:?}");
    assert_eq!(d.tp_failed_sites(), vec![(site, DEPOT)]);
    let mut e = teacher();
    e.scout_done = true;
    e.act(&f);
    for n in [3_008, 3_016, 3_024, 3_032] {
        e.act(&at(n, flags::COMPLETED | flags::MOVING));
    }
    assert_eq!(depot(&e), first, "walking: kept");
}

#[test]
fn scout_circle_waypoints_are_fixed_022_ac3() {
    let b = teacher();
    let c = depot_center(THEIRS);
    let around = |dx: i32, dy: i32| ((c.0 as i32 + dx) as u16, (c.1 as i32 + dy) as u16);
    let p = b.circle_point(c, around(-150, 10), 6);
    assert_eq!(p, b.circle_point(c, around(-146, 14), 6), "2 frames later");
    assert_eq!(p, around(0, -192), "west of the start: a quarter turn on is north (y down: -90° ... +90°)");
    let d = ((dist2(b.circle_point(c, around(100, 100), 6), c)) as f64).sqrt() / 32.0;
    assert!((5.9..=6.1).contains(&d), "{d}");
}

#[test]
fn picks_and_targets_robust_to_two_frames_022_ac3() {
    let at = (1000, 1000);
    assert_eq!(pick_key(true, (1000 + 60, 1000), at), pick_key(true, (1000 + 66, 1000), at));
    assert!(pick_key(true, (1000 + 60, 1000), at) < pick_key(true, (1000 + 200, 1000), at));
    assert_eq!(pick_key(false, (1000 + 60, 1000), at), 3_600);
    assert_eq!(snap((1000, 1010)), snap((1006, 1004)));
    assert_eq!(snap((1000, 1010)), (7 * 128 + 64, 7 * 128 + 64));
}

#[test]
fn settled_scout_is_not_ordered_again_022_ac3() {
    let main = depot_center(THEIRS);
    let units = |dx: u16| {
        let mut u = base(12);
        u.push(UnitState { flags: flags::COMPLETED | flags::IDLE, ..unit(150, SCV, (main.0 - dx, main.1)) });
        u.push(enemy(9, NEXUS, main));
        u
    };
    let moves = |act: &Act| ordered(act, |o| matches!(o, CmdOrder::Move(_))).contains(&150);
    let mut b = teacher();
    b.scout = Some(150);
    b.act(&frame(2_400, units(5 * 32), 0, 30));
    assert!(!moves(&b.act(&frame(3_700, units(2 * 32), 0, 30))), "idle 2 tiles from the depot");
    let mut c = teacher();
    c.scout = Some(150);
    c.act(&frame(2_400, units(5 * 32), 0, 30));
    assert!(moves(&c.act(&frame(3_700, units(5 * 32), 0, 30))), "idle 5 tiles away");
}

#[test]
fn cadences_on_game_clock_windows_022_ac3() {
    use crate::react::ReactState;
    assert!(ReactState::due_on(true, None, 97, 48), "the first observation of window 96..143");
    assert!(!ReactState::due_on(true, None, 105, 48), "not the first observation");
    assert!(!ReactState::due_on(true, Some(97), 100, 48), "same window");
    assert!(ReactState::due_on(true, Some(97), 145, 48), "next window");
    assert!(!ReactState::due_on(false, Some(100), 144, 48), "021-R8: 44 frames after");
    assert!(ReactState::due_on(false, Some(100), 148, 48));
}

#[test]
fn repairers_and_tank_moves_from_the_observation_022_ac3() {
    let repair = uranium_env::types::order_id("Repair").unwrap();
    let mut b = teacher();
    b.scout_done = true;
    let bunker = (home().0 + 5 * 32, home().1 + 64);
    let scene = |repairing: &[UnitId]| {
        let mut u = base(20);
        u.iter_mut().filter(|x| x.unit_type == SCV).for_each(|x| x.flags = flags::COMPLETED | flags::GATHERING);
        u.push(UnitState { hp: 120, ..unit(60, BUNKER, bunker) });
        u.extend((0..5).map(|i| enemy(90 + i, ZEALOT, (bunker.0 + 40, bunker.1 + 10 * i as u16))));
        for x in u.iter_mut().filter(|x| repairing.contains(&x.id)) {
            (x.order, x.order_target) = (repair, Some(60));
        }
        u
    };
    let act = b.act(&frame(7_008, scene(&[]), 100, 40));
    let first = ordered(&act, |o| *o == CmdOrder::Repair(60));
    assert_eq!(first.len(), 2, "{:?}", act.cmds);
    let act = b.act(&frame(7_056, scene(&[110, 111]), 100, 40));
    assert_eq!(ordered(&act, |o| *o == CmdOrder::Repair(60)), BTreeSet::from([110, 111]), "{:?}", act.cmds);

    let mut t = teacher();
    t.scout_done = true;
    let tank_at = (home().0 - 6 * 32, home().1 - 4 * 32);
    let tank = |order: u16, to: Option<Pos>| UnitState { order, order_pos: to, ..unit(300, SIEGE_TANK, tank_at) };
    t.act(&frame(6_000, [base(12), vec![tank(0, None)]].concat(), 0, 40));
    let spot = t.tp_tank_spot(&base(12).iter().collect::<Vec<_>>()).unwrap();
    let moved = |act: &Act| ordered(act, |o| *o == CmdOrder::Move(Target::Pos(spot))).contains(&300);
    assert!(moved(&t.act(&frame(6_048, [base(12), vec![tank(0, None)]].concat(), 0, 40))), "a window start");
    assert!(!moved(&t.act(&frame(6_056, [base(12), vec![tank(6, Some(spot))]].concat(), 0, 40))), "not a window start");
    assert!(moved(&t.act(&frame(6_096, [base(12), vec![tank(6, Some(spot))]].concat(), 0, 40))), "next window, still away");
}

#[test]
fn lobby_race_picks_the_teacher_022_r2() {
    for (race, tp) in [(Race::Zerg, false), (Race::Terran, true), (Race::Protoss, true), (Race::Random, true), (Race::Unknown, true)] {
        let b = crate::for_game_hints(&info(race), T0Build::TerranTp, hints());
        assert_eq!((b.tp_stats().is_some(), b.react_stats().is_some()), (tp, !tp), "{race:?}");
    }
    assert_eq!(T0Build::TerranTp.for_lobby(&info(Race::Zerg)), T0Build::TerranReact);
    assert_eq!(T0Build::TerranReact.for_lobby(&info(Race::Protoss)), T0Build::TerranReact);
    let mut b = teacher_vs(Race::Random);
    assert_eq!(b.tp.as_ref().unwrap().matchup(), Matchup::Terran);
    let mut u = base(12);
    u.push(enemy(80, PROBE, depot_center((60, 60))));
    b.act(&frame(1_000, u, 0, 24));
    assert_eq!(b.tp.as_ref().unwrap().matchup(), Matchup::Protoss);
}

#[test]
fn build_id_022_r1() {
    let t = T0Build::TerranTp;
    assert_eq!((t.name(), t.id_tag(), t.race(), t.needs_map_hints()), ("terran_tp", Some("tp"), Some(Race::Terran), true));
    assert!(Kit::for_build(Race::Terran, t).tp.is_some() && Kit::for_build(Race::Terran, t).react.is_some());
    assert!(Kit::for_build(Race::Zerg, t).tp.is_none());
    assert!(Kit::terran_react().tp.is_none());
}

#[test]
fn same_observations_same_commands_022_r1() {
    let (a, b) = (scenes(), scenes());
    assert_eq!(a.iter().map(|(_, x)| x.cmds.clone()).collect::<Vec<_>>(), b.iter().map(|(_, x)| x.cmds.clone()).collect::<Vec<_>>());
}
