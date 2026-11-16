use super::*;
use crate::{Bot, Kit, T0Build};

use super::testkit::*;

const NATURAL: Tile = (10, 26);

fn hints() -> MapHints {
    MapHints { base_locs: vec![OURS, THEIRS, NATURAL, (60, 60)], chokes: vec![(24, 10), (10, 40), (100, 100)] }
}

fn teacher() -> Brain {
    Brain::with_hints(&info(), Kit::for_build(Race::Terran, T0Build::TerranReact), &hints())
}

fn enemy(id: u32, ty: u16, pos: Pos) -> UnitState {
    UnitState { owner: 1, ..unit(id, ty, pos) }
}

fn home() -> Pos {
    depot_center(OURS)
}

fn base(scvs: u32) -> Vec<UnitState> {
    let mut v = vec![
        unit(1, CC, home()),
        unit(2, BARRACKS, (home().0 + 200, home().1 + 160)),
        UnitState { owner: 11, ..unit(3, 176, (home().0 - 200, home().1)) },
        unit(6, DEPOT, (home().0 - 48, home().1 - 160)),
        unit(7, DEPOT, (home().0 + 80, home().1 - 160)),
    ];
    v.extend((0..scvs).map(|i| unit(100 + i, SCV, (home().0 - 100, home().1 + 10 * i as u16))));
    v
}

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

fn unit_orders(act: &Act, pick: impl Fn(&CmdOrder) -> bool) -> Vec<Vec<UnitId>> {
    act.cmds
        .iter()
        .filter_map(|c| match c {
            Command::Unit { units, order, .. } if pick(order) => Some(units.clone()),
            _ => None,
        })
        .collect()
}

fn branch(b: &Brain) -> Branch {
    b.react_branch().unwrap()
}

#[test]
fn order_ids_match_the_order_table_013_r1() {
    use uranium_env::types::order_id;
    assert_eq!(order_id("ConstructingBuilding"), Some(ORDER_CONSTRUCTING));
    assert_eq!(order_id("PlaceBuilding"), Some(ORDER_PLACE_BUILDING));
    assert_eq!(order_id("Move"), Some(ORDER_MOVE));
    assert_eq!(["Attack1", "Attack2", "AttackUnit", "AttackFixedRange", "AttackMove"].map(|n| order_id(n).unwrap()), ATTACK_ORDERS);
}

#[test]
fn rush_rules_021_r8() {
    let p = ReactParams::default();
    let depots = [home()];
    let others = [THEIRS];
    let none: [Pos; 0] = [];
    let cx = RuleContext { depots: &depots, starts: &others, scout: None, known_hatcheries: &none, extractor_seen: false };
    let at = |n: u32, units: Vec<UnitState>| rush_rules(&p, &frame(n, units, 0, 0), 0, &cx);
    let far = depot_center((60, 60));
    let building = |id: u32, hp: u16| UnitState { hp, flags: flags::BEING_CONSTRUCTED, ..enemy(id, SPAWNING_POOL, far) };
    assert_eq!((pool_start_frame(2_600, 188), pool_start_frame(2_600, 187), pool_start_frame(100, 750)), (2_400, 2_401, 0));
    assert_eq!((pool_start_frame(2_480, 188), pool_start_frame(2_480, 187)), (2_280, 2_281));
    assert_eq!(at(2_480, vec![building(9, 188)]), vec![RushRule::PoolStart]);
    assert_eq!(at(2_480, vec![building(9, 187)]), vec![]);
    assert_eq!(at(2_600, vec![building(9, 188)]), vec![], "started at 1:40");
    assert_eq!(p.pool_done_before, p.pool_start_max + 1_200);
    assert_eq!(at(3_479, vec![enemy(9, SPAWNING_POOL, far)]), vec![RushRule::PoolDone]);
    assert_eq!(at(3_480, vec![enemy(9, SPAWNING_POOL, far)]), vec![]);
    assert_eq!(at(3_576, vec![enemy(9, SPAWNING_POOL, far)]), vec![], "T0 zerg_muta's pool seen at 2:29");
    let finished = UnitState { hp: 750, ..enemy(9, SPAWNING_POOL, far) };
    assert_eq!(at(3_480, vec![finished.clone()]), vec![]);
    assert_eq!(at(3_000, vec![finished]), vec![RushRule::PoolDone]);
    let lings = |n: u32| (0..n).map(|i| enemy(30 + i, ZERGLING, far)).collect::<Vec<_>>();
    assert_eq!(at(4_079, lings(2)), vec![RushRule::Lings]);
    assert_eq!(at(4_079, lings(1)), vec![]);
    assert_eq!(at(4_080, lings(2)), vec![]);
    assert_eq!(at(4_319, lings(6)), vec![RushRule::Lings]);
    assert_eq!(at(4_319, lings(5)), vec![]);
    assert_eq!(at(4_320, lings(6)), vec![]);
    let near = |id: u32, ty: u16| enemy(id, ty, (home().0 + 19 * 32, home().1 + id as u16));
    assert_eq!(at(9_000, vec![near(9, HYDRALISK), near(10, ZERGLING)]), vec![RushRule::Army]);
    assert_eq!(at(9_000, vec![near(9, ZERGLING)]), vec![]);
    assert_eq!(at(9_000, vec![near(9, ZERGLING), enemy(10, ZERGLING, (home().0 + 21 * 32, home().1))]), vec![]);
    assert_eq!(at(9_000, vec![near(9, OVERLORD), near(10, DRONE), near(11, ZERGLING)]), vec![]);
    assert_eq!(at(11_520, vec![near(9, ZERGLING), near(10, ZERGLING)]), vec![]);
    let main = depot_center(THEIRS);
    let mut v = vec![enemy(9, HATCHERY, main), UnitState { hp: 700, flags: flags::BEING_CONSTRUCTED, ..enemy(10, SPAWNING_POOL, (main.0 + 160, main.1)) }];
    v.extend((0..6).map(|i| enemy(20 + i, DRONE, (main.0 - 100, main.1 + i as u16 * 8))));
    let known = [main];
    let two = [main, depot_center(NATURAL)];
    let scout_at = |tiles: u16| Some((main.0 - tiles * 32, main.1));
    let main_at = |n: u32, units: Vec<UnitState>, scout: Option<Pos>, known: &[Pos]| {
        rush_rules(&p, &frame(n, units, 0, 0), 0, &RuleContext { depots: &depots, starts: &others, scout, known_hatcheries: known, extractor_seen: false })
    };
    assert_eq!(main_at(4_000, v.clone(), scout_at(3), &known), vec![RushRule::Drones]);
    assert_eq!(main_at(4_000, v.clone(), scout_at(4), &known), vec![], "scout 4 tiles away");
    assert_eq!(main_at(4_000, v.clone(), None, &known), vec![], "no scout");
    assert_eq!(main_at(4_000, v.clone(), scout_at(3), &two), vec![], "a second hatchery");
    assert_eq!(main_at(5_040, v.clone(), scout_at(3), &known), vec![], "after 3:30");
    let mut seven = v.clone();
    seven.push(enemy(40, DRONE, (main.0 - 100, main.1 + 90)));
    assert_eq!(main_at(4_000, seven, scout_at(3), &known), vec![], "7 drones");
    assert_eq!(main_at(4_000, v[..1].to_vec(), scout_at(3), &known), vec![], "no pool");
    let mut w = vec![enemy(9, HATCHERY, main), enemy(10, SPAWNING_POOL, (main.0 + 160, main.1))];
    w.extend((0..12).map(|i| enemy(20 + i, DRONE, (main.0 - 100, main.1 + i as u16 * 8))));
    let one = |n: u32, units: Vec<UnitState>, scout: Option<Pos>, known: &[Pos], extractor_seen: bool| {
        rush_rules(&p, &frame(n, units, 0, 0), 0, &RuleContext { depots: &depots, starts: &others, scout, known_hatcheries: known, extractor_seen })
    };
    assert_eq!(one(3_600, w.clone(), scout_at(3), &known, false), vec![RushRule::OneBase]);
    assert_eq!(one(5_759, w.clone(), scout_at(3), &known, false), vec![RushRule::OneBase]);
    assert_eq!(one(3_599, w.clone(), scout_at(3), &known, false), vec![], "before 2:30");
    assert_eq!(one(3_479, w.clone(), scout_at(3), &known, false), vec![RushRule::PoolDone], "before 2:25: ①b");
    assert_eq!(one(5_760, w.clone(), scout_at(3), &known, false), vec![], "from 4:00");
    assert_eq!(one(4_000, w.clone(), scout_at(4), &known, false), vec![], "scout 4 tiles away");
    assert_eq!(one(4_000, w.clone(), None, &known, false), vec![], "no scout");
    assert_eq!(one(4_000, w.clone(), scout_at(3), &two, false), vec![], "a second hatchery");
    assert_eq!(one(4_000, w.clone(), scout_at(3), &known, true), vec![], "an extractor seen");
    assert_eq!(one(4_000, v.clone(), scout_at(3), &known, false), vec![RushRule::Drones], "unfinished pool: ④′ only");
    assert_eq!(one(4_000, w[..1].to_vec(), scout_at(3), &known, false), vec![], "no pool");
}

#[test]
fn one_base_rule_remembers_extractors_021_r8() {
    let main = depot_center(THEIRS);
    let scout_units = || -> Vec<UnitState> {
        let mut u = base(12);
        u.push(UnitState { flags: flags::COMPLETED | flags::MOVING, ..unit(150, SCV, (main.0 - 64, main.1)) });
        u.push(enemy(9, HATCHERY, main));
        u.push(enemy(10, SPAWNING_POOL, (main.0 + 160, main.1)));
        u
    };
    let mut b = teacher();
    b.scout = Some(150);
    b.scout_done = false;
    let mut first = scout_units();
    first.push(enemy(11, EXTRACTOR, (main.0 + 200, main.1 + 200)));
    b.act(&frame(3_500, first, 0, 24));
    b.act(&frame(3_700, scout_units(), 0, 24));
    assert!(!b.teacher_stats().unwrap().rule_frames.contains_key(&RushRule::OneBase));
    let mut c = teacher();
    c.scout = Some(150);
    c.act(&frame(3_700, scout_units(), 0, 24));
    assert_eq!(c.teacher_stats().unwrap().rule_frames.get(&RushRule::OneBase), Some(&3_700));
    assert_eq!(branch(&c), Branch::Defence);
}

#[test]
fn defence_at_the_natural_and_record_021_r8() {
    let mut b = teacher();
    let nat = depot_center(NATURAL);
    let mut units = base(16);
    units.push(unit(4, CC, nat));
    units.push(UnitState { hp: 300, flags: flags::BEING_CONSTRUCTED, ..enemy(70, SPAWNING_POOL, depot_center((60, 60))) });
    b.act(&frame(3_000, units.clone(), 0, 40));
    assert_eq!(branch(&b), Branch::Macro, "pool started at {}", pool_start_frame(3_000, 300));
    units.push(enemy(90, ZERGLING, (nat.0, nat.1 + 8 * 32)));
    units.push(enemy(91, ZERGLING, (nat.0 + 20, nat.1 + 8 * 32)));
    let act = b.act(&frame(6_000, units, 100, 40));
    assert_eq!(branch(&b), Branch::Defence);
    let t = builds(&act, BUNKER);
    assert_eq!(t.len(), 1, "{:?}", act.cmds);
    assert!(within(bunker_center(t[0]), nat, 10) && bunker_center(t[0]).1 > nat.1, "{t:?}");
    let s = b.teacher_stats().unwrap();
    assert_eq!((s.pool_start_frame, s.pool_seen_frame, s.lings_max, s.contact_frame, s.rush_rule), (Some(3_000 - 400), Some(3_000), 2, Some(6_000), Some(RushRule::Army)));
    assert_eq!(s.rule_frames.get(&RushRule::Army), Some(&6_000));
}

#[test]
fn defence_base_fallbacks_021_r8() {
    let nat = depot_center(NATURAL);
    let mut units = base(16);
    units.push(unit(4, CC, nat));
    let mut b = teacher();
    let mut u = units.clone();
    u.push(enemy(70, SPAWNING_POOL, depot_center((60, 60))));
    let act = b.act(&frame(3_000, u, 100, 40));
    assert_eq!(b.teacher_stats().unwrap().rush_rule, Some(RushRule::PoolDone));
    let t = builds(&act, BUNKER);
    assert!(t.len() == 1 && within(bunker_center(t[0]), nat, 10) && !within(bunker_center(t[0]), home(), 10), "{t:?} {:?}", act.cmds);

    let mut b = teacher();
    let mut u = units;
    u.push(enemy(90, ZERGLING, (home().0 + 5 * 32, home().1 - 3 * 32)));
    u.push(enemy(91, ZERGLING, (home().0 + 6 * 32, home().1 - 3 * 32)));
    let act = b.act(&frame(6_000, u, 100, 40));
    assert_eq!(b.teacher_stats().unwrap().rush_rule, Some(RushRule::Army));
    let t = builds(&act, BUNKER);
    assert!(t.len() == 1 && within(bunker_center(t[0]), home(), 10), "{t:?} {:?}", act.cmds);
}

#[test]
fn branch_switches_021_r8() {
    let mut b = teacher();
    b.act(&frame(2_000, base(12), 0, 24));
    assert_eq!(branch(&b), Branch::Macro);
    let mut units = base(12);
    units.push(enemy(90, ZERGLING, depot_center((60, 60))));
    units.push(enemy(91, ZERGLING, depot_center((60, 60))));
    let act = b.act(&frame(2_008, units, 100, 24));
    assert_eq!(branch(&b), Branch::Defence);
    let bunker = builds(&act, BUNKER);
    assert_eq!(bunker.len(), 1, "{:?}", act.cmds);
    let c = bunker_center(bunker[0]);
    let d = ((dist2(c, home()) as f64).sqrt() / 32.0, c.1 as i64 - home().1 as i64);
    assert!(d.0 >= 5.0 && d.0 <= 7.0 && d.1 > 0, "{bunker:?} {d:?}");
    let marines = |n: u32| (0..n).map(|i| unit(200 + i, MARINE, (home().0 + 100, home().1 + 4 * i as u16))).collect::<Vec<_>>();
    let mut u = base(12);
    u.extend(marines(7));
    b.act(&frame(2_008 + 1_400, u, 0, 40));
    assert_eq!(branch(&b), Branch::Defence);
    let mut u = base(12);
    u.extend(marines(8));
    b.act(&frame(2_008 + 1_439, u.clone(), 0, 40));
    assert_eq!(branch(&b), Branch::Defence);
    b.act(&frame(2_008 + 1_440, u.clone(), 0, 40));
    assert_eq!(branch(&b), Branch::Macro);
    u.push(enemy(92, ZERGLING, (home().0 + 300, home().1)));
    u.push(enemy(93, ZERGLING, (home().0 + 330, home().1)));
    b.act(&frame(5_000, u, 0, 40));
    assert_eq!(branch(&b), Branch::Defence);
    let s = b.teacher_stats().unwrap();
    assert_eq!((s.rush_frame, s.rush_rule, s.defences, s.macro_frame, s.branch), (Some(2_008), Some(RushRule::Lings), 2, Some(3_448), Branch::Defence));
}

#[test]
fn bunker_site_skips_units_in_the_way_021_r8() {
    let b = teacher();
    let mut f = frame(3_000, base(10), 0, 40);
    let to = b.bunker_facing(home(), None);
    let first = b.bunker_site(&f, home(), to, &[]).unwrap();
    f.units.push(UnitState { flags: flags::COMPLETED, ..unit(300, OVERLORD, bunker_center(first)) });
    assert_eq!(b.bunker_site(&f, home(), to, &[]), Some(first), "a flyer does not block");
    f.units.push(enemy(301, ZERGLING, bunker_center(first)));
    let second = b.bunker_site(&f, home(), to, &[]).unwrap();
    assert_ne!(second, first);
    let d = (dist2(bunker_center(second), home()) as f64).sqrt() / 32.0;
    assert!((4.0..=6.0).contains(&d), "{second:?} {d}");
    assert!(clear_of_units(&f, second) && !clear_of_units(&f, first));
}

#[test]
fn macro_expands_to_the_natural_021_r8() {
    let mut b = teacher();
    let mut units = base(14);
    units.push(enemy(80, HATCHERY, depot_center(THEIRS)));
    let act = b.act(&frame(3_000, units.clone(), 350, 30));
    assert!(b.react.as_ref().unwrap().enemy_main.is_some());
    assert_eq!((builds(&act, CC).len(), trains(&act, MARINE)), (0, 0), "{:?}", act.cmds);
    let units: Vec<UnitState> = units.into_iter().filter(|u| u.owner == 0).collect();
    let act = b.act(&frame(3_008, units, 400, 30));
    assert_eq!(builds(&act, CC), vec![NATURAL], "{:?}", act.cmds);
    assert_eq!(b.teacher_stats().unwrap().expand_frame, Some(3_008));

    let mut b = teacher();
    let act = b.act(&frame(5_279, base(14), 600, 30));
    assert_eq!((builds(&act, CC).len(), builds(&act, BUNKER).len()), (0, 0), "{:?}", act.cmds);
    let act = b.act(&frame(5_280, base(14), 600, 30));
    assert_eq!(builds(&act, CC), vec![NATURAL], "{:?}", act.cmds);
}

#[test]
fn natural_bunker_not_before_four_minutes_021_r8() {
    let mut units = base(16);
    units.push(unit(4, CC, depot_center(NATURAL)));
    let mut b = teacher();
    let act = b.act(&frame(5_759, units.clone(), 500, 40));
    assert!(builds(&act, BUNKER).is_empty(), "{:?}", act.cmds);
    let act = b.act(&frame(5_760, units.clone(), 500, 40));
    let t = builds(&act, BUNKER);
    assert_eq!(t.len(), 1, "{:?}", act.cmds);
    assert!(bunker_center(t[0]).1 > depot_center(NATURAL).1, "{t:?}");
    let mut b = teacher();
    let last = units.len() - 1;
    units[last].flags = flags::BEING_CONSTRUCTED;
    let act = b.act(&frame(6_000, units, 500, 40));
    assert!(builds(&act, BUNKER).is_empty(), "{:?}", act.cmds);
}

#[test]
fn defence_loads_four_and_holds_the_rest_021_r8() {
    let mut b = teacher();
    let mut units = base(12);
    let bunker = (home().0 + 5 * 32, home().1);
    units.push(UnitState { hp: BUNKER_HP, ..unit(5, BUNKER, bunker) });
    units.extend((0..6).map(|i| unit(200 + i, MARINE, (home().0 + 40, home().1 + 100 + 8 * i as u16))));
    units.push(enemy(90, ZERGLING, depot_center((60, 60))));
    units.push(enemy(91, ZERGLING, depot_center((60, 60))));
    let act = b.act(&frame(3_000, units.clone(), 0, 40));
    assert_eq!(branch(&b), Branch::Defence);
    let load = unit_orders(&act, |o| *o == CmdOrder::Load(5));
    assert_eq!(load.iter().map(Vec::len).collect::<Vec<_>>(), vec![4], "{:?}", act.cmds);
    let walk: usize = unit_orders(&act, |o| matches!(o, CmdOrder::Move(_))).iter().map(Vec::len).sum();
    assert_eq!(walk, 2, "{:?}", act.cmds);
    let spot = toward(bunker, home(), 2.0);
    for u in units.iter_mut().filter(|u| u.unit_type == MARINE) {
        u.pos = spot;
    }
    units.iter_mut().filter(|u| u.unit_type == MARINE).take(2).for_each(|u| u.loaded_in = Some(5));
    let act = b.act(&frame(3_048, units, 0, 40));
    let loads: usize = unit_orders(&act, |o| *o == CmdOrder::Load(5)).iter().map(Vec::len).sum();
    let holds: usize = unit_orders(&act, |o| *o == CmdOrder::Hold).iter().map(Vec::len).sum();
    assert_eq!((loads, holds), (2, 2), "{:?}", act.cmds);
    assert!(unit_orders(&act, |o| matches!(o, CmdOrder::Attack(Target::Pos(_)))).is_empty(), "no waves in D: {:?}", act.cmds);
}

#[test]
fn defence_pulls_repairs_and_releases_scvs_021_r8() {
    let mut b = teacher();
    b.scout_done = true;
    let mut units = base(20);
    let bunker = (home().0 + 5 * 32, home().1 + 64);
    units.push(UnitState { hp: 300, ..unit(5, BUNKER, bunker) });
    units.push(enemy(90, ZERGLING, (bunker.0, bunker.1 + 64)));
    units.push(enemy(91, ZERGLING, (home().0 - 150, home().1 + 40)));
    units.push(enemy(92, ZERGLING, (home().0 + 2 * 32, home().1 + 6 * 32 - 20)));
    let act = b.act(&frame(3_000, units.clone(), 0, 40));
    assert_eq!(branch(&b), Branch::Defence);
    let attack: Vec<(Vec<UnitId>, UnitId)> = act
        .cmds
        .iter()
        .filter_map(|c| match c {
            Command::Unit { units, order: CmdOrder::Attack(Target::Unit(t)), .. } => Some((units.clone(), *t)),
            _ => None,
        })
        .collect();
    assert_eq!(attack.len(), 1, "{:?}", act.cmds);
    assert_eq!((attack[0].0.len(), attack[0].1), (6, 90), "min(6, 20 - 4) onto the zergling nearest the bunker");
    let repair = unit_orders(&act, |o| *o == CmdOrder::Repair(5));
    assert_eq!(repair.iter().map(Vec::len).collect::<Vec<_>>(), vec![2], "{:?}", act.cmds);
    assert!(repair[0].iter().all(|id| !attack[0].0.contains(id)));
    let pulled = attack[0].0.clone();
    assert!(unit_orders(&act, |o| matches!(o, CmdOrder::Gather(_))).iter().all(|g| g.iter().all(|u| !pulled.contains(u))));

    let mut away: Vec<UnitState> = units.iter().filter(|u| u.owner != 1 || u.id == 92).cloned().collect();
    for u in away.iter_mut().filter(|u| u.unit_type == BUNKER) {
        u.hp = BUNKER_HP;
    }
    let act = b.act(&frame(3_008, away.clone(), 0, 40));
    let gathers = unit_orders(&act, |o| matches!(o, CmdOrder::Gather(_)));
    assert!(gathers.contains(&pulled), "{:?}", act.cmds);
    assert!(unit_orders(&act, |o| matches!(o, CmdOrder::Attack(Target::Unit(_)))).is_empty(), "no chase: {:?}", act.cmds);
    assert!(!b.react_busy().iter().any(|id| pulled.contains(id)));
    let act = b.act(&frame(3_048, away, 0, 40));
    assert!(unit_orders(&act, |o| matches!(o, CmdOrder::Gather(_))).contains(&repair[0]), "{:?}", act.cmds);
    assert!(b.react_busy().is_empty());

    let mut c = teacher();
    c.scout_done = true;
    let mut u = base(20);
    u.push(enemy(93, ZERGLING, (home().0 + 6 * 32, home().1)));
    u.push(enemy(94, ZERGLING, (home().0 + 6 * 32, home().1 + 10)));
    let act = c.act(&frame(3_000, u.clone(), 0, 40));
    assert_eq!(branch(&c), Branch::Defence);
    assert!(unit_orders(&act, |o| matches!(o, CmdOrder::Attack(Target::Unit(_)))).is_empty(), "{:?}", act.cmds);
    u.iter_mut().filter(|x| x.id == 93).for_each(|x| x.pos = (home().0 + 5 * 32 - 4, home().1));
    let act = c.act(&frame(3_048, u, 0, 40));
    assert_eq!(unit_orders(&act, |o| *o == CmdOrder::Attack(Target::Unit(93))).iter().map(Vec::len).collect::<Vec<_>>(), vec![6], "{:?}", act.cmds);
}

fn defence_frame(n: u32, extra: Vec<UnitState>, minerals: i32) -> RawFrame {
    let mut units = base(12);
    units.push(enemy(90, ZERGLING, depot_center((60, 60))));
    units.push(enemy(91, ZERGLING, depot_center((60, 60))));
    units.extend(extra);
    frame(n, units, minerals, 24)
}

#[test]
fn bunker_order_expires_without_a_bunker_041_r22() {
    let p = ReactParams::default();
    assert_eq!(p.bunker_order_expiry, 144);
    let mut b = teacher();
    b.scout_done = true;
    let act = b.act(&defence_frame(2_008, vec![], 100));
    assert_eq!(branch(&b), Branch::Defence);
    let site = builds(&act, BUNKER);
    assert_eq!(site.len(), 1, "{:?}", act.cmds);
    let worker = b.pending.iter().find(|q| q.ty == BUNKER).unwrap().worker;
    let act = b.act(&defence_frame(2_008 + 143, vec![], 100));
    assert!(builds(&act, BUNKER).is_empty(), "143 frames: still ordered: {:?}", act.cmds);
    let mut walking = teacher();
    walking.scout_done = true;
    walking.act(&defence_frame(2_008, vec![], 100));
    let mut f = defence_frame(2_008 + 144, vec![], 100);
    f.units.iter_mut().filter(|u| u.id == worker).for_each(|u| u.order = ORDER_PLACE_BUILDING);
    assert!(builds(&walking.act(&f), BUNKER).is_empty(), "the builder walks to the site");
    let mut elsewhere = teacher();
    elsewhere.scout_done = true;
    elsewhere.act(&defence_frame(2_008, vec![], 100));
    let mut f = defence_frame(2_008 + 144, vec![], 100);
    f.units.iter_mut().filter(|u| u.id == worker).for_each(|u| (u.order, u.order_pos) = (ORDER_PLACE_BUILDING, Some((10, 10))));
    assert_eq!(builds(&elsewhere.act(&f), BUNKER).len(), 1, "placing something else: ordered again");
    let mut started = teacher();
    started.scout_done = true;
    started.act(&defence_frame(2_008, vec![], 100));
    let near = (bunker_center(site[0]).0 + 32, bunker_center(site[0]).1);
    let act = started.act(&defence_frame(2_008 + 144, vec![UnitState { flags: flags::BEING_CONSTRUCTED, ..unit(60, BUNKER, near) }], 100));
    assert!(builds(&act, BUNKER).is_empty(), "{:?}", act.cmds);
    assert!(started.pending.iter().any(|q| q.ty == BUNKER));
    let act = b.act(&defence_frame(2_008 + 144, vec![], 100));
    assert_eq!(builds(&act, BUNKER).len(), 1, "{:?}", act.cmds);
    assert!(b.pending.iter().filter(|q| q.ty == BUNKER).all(|q| q.issued == 2_152));
    assert!(!bunker_order_stale(&p, &crate::Pending { ty: DEPOT, tile: (0, 0), worker: 1, issued: 0 }, &defence_frame(10_000, vec![], 0), &[]));
}

#[test]
fn lifted_buildings_train_nothing_041_r22() {
    let mut b = teacher();
    b.scout_done = true;
    let act = b.act(&frame(3_000, base(12), 1_000, 24));
    assert_eq!((trains(&act, SCV), trains(&act, MARINE)), (1, 1), "{:?}", act.cmds);
    let mut units = base(12);
    units.iter_mut().filter(|u| matches!(u.unit_type, CC | BARRACKS)).for_each(|u| u.flags |= flags::LIFTED);
    let act = teacher().act(&frame(3_000, units, 1_000, 24));
    assert_eq!((trains(&act, SCV), trains(&act, MARINE)), (0, 0), "{:?}", act.cmds);
}

#[test]
fn scout_is_chosen_again_from_each_observation_041_r22() {
    let cc_only = |scvs: u32| {
        let mut v = vec![unit(1, CC, home()), UnitState { owner: 11, ..unit(3, 176, (home().0 - 200, home().1)) }];
        v.extend((0..scvs).map(|i| unit(100 + i, SCV, (home().0 - 100, home().1 + 10 * i as u16))));
        v
    };
    let dark = |f: RawFrame| RawFrame { visibility: BitGrid::new(128, 128), ..f };
    let start = |b: &mut Brain| {
        b.act(&dark(frame(508, cc_only(9), 200, 18)));
        b.act(&dark(frame(516, cc_only(9), 100, 18)));
        b.scout.expect("scout")
    };
    let with = |n: u32, f: &dyn Fn(&mut UnitState)| {
        let mut u = cc_only(9);
        u.iter_mut().for_each(f);
        dark(frame(n, u, 100, 18))
    };
    let moves = |act: &Act| unit_orders(act, |o| *o == CmdOrder::Move(Target::Pos(depot_center(THEIRS))));
    let mut b = teacher();
    let s = start(&mut b);
    let walking = |u: &mut UnitState| {
        if u.id == s {
            (u.order, u.order_pos, u.flags) = (ORDER_MOVE, Some(depot_center(THEIRS)), flags::COMPLETED);
        }
    };
    for n in [524, 564, 600, 700] {
        let act = b.act(&with(n, &walking));
        assert_eq!(b.scout, Some(s), "frame {n}");
        assert!(moves(&act).iter().all(|m| m == &vec![s]), "{:?}", act.cmds);
    }
    let mut b = teacher();
    let s = start(&mut b);
    let mining = |u: &mut UnitState| {
        if u.id == s {
            u.flags = flags::COMPLETED | flags::GATHERING | flags::CARRYING;
        }
    };
    b.act(&with(524, &mining));
    assert_eq!(b.scout, Some(s), "within 48 frames of the order");
    let act = b.act(&with(564, &mining));
    let t = b.scout.expect("scout");
    assert_ne!(t, s);
    assert_eq!(moves(&act), vec![vec![t]], "the new scout is ordered at once");
    let model_scout = |u: &mut UnitState| {
        mining(u);
        if u.id == 108 {
            (u.order, u.order_pos, u.flags) = (ORDER_MOVE, Some(depot_center(THEIRS)), flags::COMPLETED);
        }
    };
    let act = b.act(&with(620, &model_scout));
    assert_eq!(b.scout, Some(108));
    assert_eq!(moves(&act), vec![vec![108]], "{:?}", act.cmds);
    let mut b = teacher();
    let s = start(&mut b);
    let mut u = cc_only(9);
    u.retain(|x| x.id != s);
    b.act(&dark(frame(530, u, 100, 18)));
    assert!(b.scout.is_some_and(|t| t != s) && !b.scout_done);
    let t = b.scout.unwrap();
    let far = |u: &mut UnitState| {
        if u.id == t {
            (u.pos, u.flags) = ((2_000, 2_000), flags::COMPLETED);
        }
    };
    b.act(&with(540, &far));
    let mut u = cc_only(9);
    u.retain(|x| x.id != t);
    b.act(&dark(frame(548, u, 100, 18)));
    assert!(b.scout.is_none() && b.scout_done);
}

#[test]
fn pulled_scvs_are_chosen_again_from_each_observation_041_r22() {
    let setup = || {
        let mut b = teacher();
        b.scout_done = true;
        let mut units = base(20);
        let bunker = (home().0 + 5 * 32, home().1 + 64);
        units.push(UnitState { hp: BUNKER_HP, ..unit(5, BUNKER, bunker) });
        units.push(enemy(90, ZERGLING, (bunker.0, bunker.1 + 64)));
        units.push(enemy(91, ZERGLING, (bunker.0 + 40, bunker.1 + 70)));
        let act = b.act(&frame(3_000, units.clone(), 0, 40));
        let first = unit_orders(&act, |o| *o == CmdOrder::Attack(Target::Unit(90)));
        assert_eq!(first.iter().map(Vec::len).collect::<Vec<_>>(), vec![6], "{:?}", act.cmds);
        (b, units, first[0].clone())
    };
    let attack_at = |act: &Act| unit_orders(act, |o| *o == CmdOrder::Attack(Target::Unit(90)));
    let (mut b, mut units, pulled) = setup();
    for u in units.iter_mut().filter(|u| pulled.contains(&u.id)) {
        (u.order, u.order_target, u.flags) = (10, Some(90), flags::COMPLETED);
    }
    let act = b.act(&frame(3_048, units, 0, 40));
    assert_eq!(attack_at(&act), vec![pulled.clone()], "{:?}", act.cmds);
    let (mut b, mut units, pulled) = setup();
    for u in units.iter_mut() {
        if pulled.contains(&u.id) {
            (u.pos, u.flags) = ((home().0 - 300, home().1), flags::COMPLETED | flags::GATHERING);
        } else if u.id == 100 || u.id == 101 {
            (u.order, u.order_target, u.flags) = (10, Some(90), flags::COMPLETED);
        }
    }
    let act = b.act(&frame(3_008, units.clone(), 0, 40));
    assert!(attack_at(&act).is_empty() && pulled.iter().all(|id| b.react_busy().contains(id)), "{:?}", act.cmds);
    let act = b.act(&frame(3_048, units, 0, 40));
    let now = attack_at(&act);
    assert_eq!(now.len(), 1, "{:?}", act.cmds);
    assert_eq!(now[0].len(), 6);
    assert!(now[0].contains(&100) && now[0].contains(&101), "taken over: {now:?}");
    assert!(now[0].iter().all(|id| !pulled.contains(id)), "the mining ones are dropped: {now:?}");
}

#[test]
fn main_bunker_faces_out_of_the_main_021_r8() {
    let hints = MapHints { base_locs: vec![OURS, THEIRS, NATURAL], chokes: vec![(13, 12), (12, 13), (10, 20)] };
    let b = Brain::with_hints(&info(), Kit::for_build(Race::Terran, T0Build::TerranReact), &hints);
    assert_eq!(b.bunker_facing(home(), None), (10 * 32 + 16, 20 * 32 + 16));
    let t = b.bunker_site(&frame(3_000, base(10), 0, 40), home(), b.bunker_facing(home(), None), &[]).unwrap();
    assert!(bunker_center(t).1 > home().1 + 3 * 32, "south, toward the natural: {t:?}");
    let hints = MapHints { base_locs: vec![OURS, THEIRS, NATURAL], chokes: vec![(13, 12)] };
    let b = Brain::with_hints(&info(), Kit::for_build(Race::Terran, T0Build::TerranReact), &hints);
    assert_eq!(b.bunker_facing(home(), None), depot_center(NATURAL));
}

#[test]
fn main_bunker_site_is_kept_free_021_r8() {
    let mut b = teacher();
    let units = base(12);
    b.act(&frame(100, units.clone(), 0, 24));
    let site = b.react.as_ref().unwrap().main_site.expect("planned");
    let f = frame(200, units.clone(), 2_000, 24);
    let mine: Vec<&UnitState> = f.units.iter().filter(|u| u.owner == 0).collect();
    let mut cmds = Vec::new();
    for ty in [DEPOT, BARRACKS, DEPOT, BARRACKS, DEPOT, BARRACKS, DEPOT, DEPOT] {
        assert!(b.order_build(ty, &f, &mine, home(), &[], &mut cmds));
    }
    let (bx, by) = (site.0 as i32, site.1 as i32);
    for p in &b.pending {
        let (w, h) = tile_size(p.ty);
        let (x, y) = (p.tile.0 as i32, p.tile.1 as i32);
        assert!(x + w as i32 <= bx || bx + 3 <= x || y + h as i32 <= by || by + 2 <= y, "{:?} at {:?} covers the bunker site {site:?}", p.ty, p.tile);
        assert!(p.tile != NATURAL);
    }
    let mut u = units;
    u.push(enemy(90, ZERGLING, depot_center((60, 60))));
    u.push(enemy(91, ZERGLING, depot_center((60, 60))));
    let act = b.act(&frame(3_000, u, 100, 24));
    assert_eq!(builds(&act, BUNKER), vec![site], "{:?}", act.cmds);
}

#[test]
fn fallback_bunker_site_stays_within_nine_tiles_021_r8() {
    let b = teacher();
    let mut f = frame(3_000, base(10), 0, 40);
    let (hx, hy) = ((home().0 / 32) as i32, (home().1 / 32) as i32);
    let mut id = 400;
    for y in hy - 8..=hy + 8 {
        for x in hx - 8..=hx + 8 {
            let d2 = (x - hx).pow(2) + (y - hy).pow(2);
            if x >= 0 && y >= 0 && (9..=49).contains(&d2) {
                f.units.push(enemy(id, ZERGLING, (x as u16 * 32 + 16, y as u16 * 32 + 16)));
                id += 1;
            }
        }
    }
    if let Some(t) = b.bunker_site(&f, home(), b.bunker_facing(home(), None), &[]) {
        assert!(within(bunker_center(t), home(), BUNKER_SEARCH_MAX), "{t:?}");
    }
    f.units.retain(|u| u.owner != 1 || dist2(u.pos, home()) < 5 * 5 * 32 * 32);
    let t = b.bunker_site(&f, home(), b.bunker_facing(home(), None), &[]).expect("a site beyond 5 tiles");
    assert!(within(bunker_center(t), home(), BUNKER_SEARCH_MAX), "{t:?}");
}

#[test]
fn no_bunker_before_four_minutes_without_rush_evidence_021_r8() {
    let mut b = teacher();
    let mut bunkers = 0;
    let mut expanded = false;
    for n in (8..=5_760).step_by(8) {
        let scvs = (4 + n / 300).min(22);
        let mut u = if n < 1_600 {
            let mut v = vec![unit(1, CC, home()), UnitState { owner: 11, ..unit(3, 176, (home().0 - 200, home().1)) }];
            v.extend((0..scvs).map(|i| unit(100 + i, SCV, (home().0 - 100, home().1 + 4 * i as u16))));
            v
        } else {
            base(scvs)
        };
        if (2_800..3_100).contains(&n) {
            let m = depot_center(THEIRS);
            u.push(enemy(80, HATCHERY, m));
            u.extend((0..10).map(|i| enemy(81 + i, DRONE, (m.0 - 100, m.1 + 8 * i as u16))));
        }
        if n >= 4_000 {
            u.push(enemy(95, OVERLORD, (home().0 + 64, home().1 + 64)));
        }
        let supply = (2 * scvs) as u16;
        let act = b.act(&frame(n, u, 450, supply));
        bunkers += builds(&act, BUNKER).len();
        expanded |= builds(&act, CC).contains(&NATURAL);
    }
    assert_eq!((bunkers, branch(&b)), (0, Branch::Macro));
    assert!(expanded);
}

#[test]
#[ignore = "needs URANIUM_MAPS_DIR with the target-map precomputes"]
fn target_map_bunker_direction_021_r8() {
    let dir = std::path::PathBuf::from(std::env::var("URANIUM_MAPS_DIR").expect("URANIUM_MAPS_DIR"));
    let maps = [
        ("175b118cc1ab70c95e9bfc90cc5640ed300b7370443e9394aeb0d1dce6dbb0bc", [(117, 7), (7, 6), (7, 116), (117, 117)]),
        ("8109bae22de8e79ca9ddc9ca4bad343dbf5ac676747d7805f043cd8080bfdf56", [(117, 118), (7, 118), (7, 9), (117, 9)]),
    ];
    for (hash, starts) in maps {
        let pre = uranium_env::maps::load_precompute(&dir, hash).expect("precompute");
        let hints = MapHints { base_locs: pre.base_locs.clone(), chokes: pre.chokes.clone() };
        for s in starts {
            let mut i = info();
            i.static_map.start_locations = starts.to_vec();
            i.players[0].start_tile = Some(s);
            let b = Brain::with_hints(&i, Kit::for_build(Race::Terran, T0Build::TerranReact), &hints);
            let (h, nat) = (depot_center(s), depot_center(b.react.as_ref().unwrap().natural.unwrap()));
            let to = b.bunker_facing(h, None);
            let (v, w) = ((to.0 as f64 - h.0 as f64, to.1 as f64 - h.1 as f64), (nat.0 as f64 - h.0 as f64, nat.1 as f64 - h.1 as f64));
            let cos = (v.0 * w.0 + v.1 * w.1) / ((v.0.hypot(v.1)) * (w.0.hypot(w.1)));
            assert!(dist2(to, h) >= (8 * 32) * (8 * 32) && cos > 0.5, "{hash:.8} start {s:?}: faces {to:?}, natural {nat:?}, cos {cos:.2}");
        }
    }
}

#[test]
fn opening_and_scout_021_r8() {
    let cc_only = |scvs: u32| {
        let mut v = vec![unit(1, CC, home()), UnitState { owner: 11, ..unit(3, 176, (home().0 - 200, home().1)) }];
        v.extend((0..scvs).map(|i| unit(100 + i, SCV, (home().0 - 100, home().1 + 10 * i as u16))));
        v
    };
    let dark = |f: RawFrame| RawFrame { visibility: BitGrid::new(128, 128), ..f };
    let mut b = teacher();
    let act = b.act(&dark(frame(500, cc_only(8), 200, 16)));
    assert!(builds(&act, DEPOT).is_empty(), "8 supply: {:?}", act.cmds);
    assert!(b.scout.is_none(), "no scout before the first depot order");
    let act = b.act(&dark(frame(508, cc_only(9), 200, 18)));
    assert_eq!(builds(&act, DEPOT).len(), 1, "9 supply");
    let builder = b.pending[0].worker;
    let act = b.act(&dark(frame(516, cc_only(9), 100, 18)));
    let scout = b.scout.expect("scout");
    assert_ne!(scout, builder);
    assert!(act.cmds.contains(&Command::Unit { units: vec![scout], order: CmdOrder::Move(Target::Pos(depot_center(THEIRS))), queued: false }), "{:?}", act.cmds);
    let mut units = cc_only(10);
    units.push(unit(6, DEPOT, (home().0, home().1 - 160)));
    let act = teacher().act(&frame(1_500, units.clone(), 200, 20));
    assert!(builds(&act, BARRACKS).is_empty(), "10 supply: {:?}", act.cmds);
    let act = teacher().act(&dark(frame(1_500, units.clone(), 200, 22)));
    assert_eq!(builds(&act, BARRACKS).len(), 1, "11 supply");
    units.push(UnitState { flags: flags::BEING_CONSTRUCTED, ..unit(2, BARRACKS, (home().0 + 200, home().1 + 160)) });
    let act = teacher().act(&frame(1_600, units.clone(), 200, 28));
    assert!(builds(&act, DEPOT).is_empty(), "14 supply: {:?}", act.cmds);
    let act = teacher().act(&frame(1_600, units, 200, 30));
    assert_eq!(builds(&act, DEPOT).len(), 1, "15 supply");
}

#[test]
fn same_observations_same_commands_021_r8() {
    let script = |b: &mut Brain| {
        let mut out = Vec::new();
        let lings = vec![enemy(90, ZERGLING, (home().0 + 100, home().1)), enemy(91, ZERGLING, (home().0 + 130, home().1))];
        for (i, extra) in [vec![], lings, vec![]].into_iter().enumerate() {
            let mut u = base(12);
            u.extend(extra);
            u.extend((0..5).map(|j| unit(200 + j, MARINE, (home().0 + 50, home().1 + 6 * j as u16))));
            out.push(b.act(&frame(3_000 + 48 * i as u32, u, 300, 40)).cmds);
        }
        out
    };
    let (a, c) = (script(&mut teacher()), script(&mut teacher()));
    assert_eq!(a, c);
    assert!(a.iter().flatten().any(|c| matches!(c, Command::Unit { order: CmdOrder::Attack(Target::Unit(_)), .. })));
}

fn walled_info() -> GameInfo {
    let mut i = info();
    for y in 17..=18u16 {
        for x in (0..128u16).filter(|x| !(24..=27).contains(x)) {
            i.static_map.buildable.set(x, y, false);
            for c in 0..16 {
                i.static_map.walkable.set(x * 4 + c % 4, y * 4 + c / 4, false);
            }
        }
    }
    i
}

#[test]
fn main_bunker_site_on_the_ground_path_021_r8() {
    let i = walled_info();
    let solid = BTreeSet::new();
    let path = ground_path(&i.static_map.walkable, (12, 11), (12, 27), &solid);
    assert_eq!((path.first(), path.last()), (Some(&(12, 11)), Some(&(12, 27))));
    assert!(path.iter().any(|t| (17..=18).contains(&t.1) && (24..=27).contains(&t.0)), "{path:?}");
    assert!(path.windows(2).all(|w| (w[0].0 as i32 - w[1].0 as i32).abs() <= 1 && (w[0].1 as i32 - w[1].1 as i32).abs() <= 1));
    assert!(path.iter().all(|t| !(17..=18).contains(&t.1) || (24..=27).contains(&t.0)));
    let all: BTreeSet<Tile> = (24..=27).flat_map(|x| (17..=18).map(move |y| (x, y))).collect();
    assert!(ground_path(&i.static_map.walkable, (12, 11), (12, 27), &all).is_empty());

    let mut b = Brain::with_hints(&i, Kit::for_build(Race::Terran, T0Build::TerranReact), &hints());
    let mut units = vec![unit(1, CC, home()), UnitState { owner: 11, ..unit(3, 176, (home().0 - 200, home().1)) }];
    units.extend((0..10).map(|n| unit(100 + n, SCV, (home().0 - 100, home().1 + 10 * n as u16))));
    b.act(&frame(100, units, 0, 20));
    let r = b.react.as_ref().unwrap();
    let path = r.main_path.clone().expect("planned");
    let site = r.main_site.expect("main site");
    let c = bunker_center(site);
    let d = (dist2(c, home()) as f64).sqrt() / 32.0;
    assert!((5.0..=7.0).contains(&d), "{site:?} {d}");
    assert!(path.iter().any(|t| within(c, (t.0 * 32 + 16, t.1 * 32 + 16), 2)), "{site:?} {path:?}");
    assert!(c.0 > home().0, "toward the ramp (east), not straight south: {site:?}");
}

#[test]
fn defence_marines_attack_or_hold_021_r8() {
    let run = |marines: u32| {
        let mut b = teacher();
        b.scout_done = true;
        let mut u = base(12);
        u.extend((0..marines).map(|n| unit(200 + n, MARINE, (home().0 + 100, home().1 + 100 + 4 * n as u16))));
        u.push(enemy(90, ZERGLING, (home().0 + 8 * 32, home().1)));
        u.push(enemy(91, ZERGLING, (home().0 + 8 * 32, home().1 + 20)));
        let act = b.act(&frame(3_000, u, 0, 40));
        assert_eq!(branch(&b), Branch::Defence);
        let attack: usize = unit_orders(&act, |o| matches!(o, CmdOrder::Attack(Target::Pos(_)))).iter().map(Vec::len).sum();
        let walk: usize = unit_orders(&act, |o| matches!(o, CmdOrder::Move(_))).iter().flatten().filter(|id| **id >= 200).count();
        (attack, walk, b.react_marines_attack())
    };
    assert_eq!(run(4), (4, 0, true), "4 marines vs 2 zerglings: attack");
    assert_eq!(run(3), (0, 3, false), "3 marines vs 2 zerglings: hold beside the bunker");
}

#[test]
fn macro_return_at_twelve_marines_021_r8() {
    let at = |b: &mut Brain, n: u32, marines: u32| {
        let mut u = base(12);
        u.extend((0..marines).map(|k| unit(200 + k, MARINE, (home().0 + 100, home().1 + 4 * k as u16))));
        u.push(enemy(90, ZERGLING, (home().0 + 9 * 32, home().1)));
        u.push(enemy(91, ZERGLING, (home().0 + 9 * 32, home().1 + 20)));
        b.act(&frame(n, u, 0, 40));
        branch(b)
    };
    let mut b = teacher();
    b.scout_done = true;
    assert_eq!(at(&mut b, 3_000, 2), Branch::Defence);
    assert_eq!(at(&mut b, 3_048, 11), Branch::Defence, "11 marines under contact");
    assert_eq!(at(&mut b, 3_096, 12), Branch::Macro, "12 marines whatever the contact");
    assert_eq!(at(&mut b, 3_144, 12), Branch::Macro, "R with 12 marines keeps M");
    assert_eq!(b.teacher_stats().unwrap().macro_frame, Some(3_096));
    assert_eq!(at(&mut b, 3_192, 11), Branch::Defence, "fewer than 12 again: R switches to D");
}

#[test]
fn construction_resumes_without_a_builder_021_r8() {
    let mut b = teacher();
    b.scout_done = true;
    let rax = (home().0 + 7 * 32, home().1 - 4 * 32);
    let mut u = base(12);
    u.push(UnitState { hp: 900, flags: flags::BEING_CONSTRUCTED, ..unit(50, BARRACKS, rax) });
    let right_click = |act: &Act| unit_orders(act, |o| *o == CmdOrder::RightClick(Target::Unit(50)));
    assert!(right_click(&b.act(&frame(3_000, u.clone(), 0, 30))).is_empty());
    assert!(right_click(&b.act(&frame(3_047, u.clone(), 0, 30))).is_empty());
    let act = b.act(&frame(3_048, u.clone(), 0, 30));
    let nearest = u.iter().filter(|x| x.unit_type == SCV).min_by_key(|x| (dist2(x.pos, rax), x.id)).unwrap().id;
    assert_eq!(right_click(&act), vec![vec![nearest]], "{:?}", act.cmds);
    assert!(b.react_busy().contains(&nearest));
    assert!(right_click(&b.act(&frame(3_090, u.clone(), 0, 30))).is_empty());
    assert_eq!(right_click(&b.act(&frame(3_096, u.clone(), 0, 30))), vec![vec![nearest]]);
    for x in u.iter_mut().filter(|x| x.id == nearest) {
        (x.order, x.order_target, x.flags) = (ORDER_CONSTRUCTING, Some(50), flags::COMPLETED | flags::CONSTRUCTING);
    }
    assert!(right_click(&b.act(&frame(3_200, u.clone(), 0, 30))).is_empty());
    assert!(b.react.as_ref().unwrap().resumers.is_empty() && !b.react_busy().contains(&nearest));
}

#[test]
fn builder_harassed_by_a_worker_021_r8() {
    let mut b = teacher();
    b.scout_done = true;
    let mut u = base(12);
    for x in u.iter_mut().filter(|x| x.unit_type == SCV) {
        x.flags = flags::COMPLETED | flags::GATHERING;
    }
    let at = (home().0 + 6 * 32, home().1 - 3 * 32);
    u.push(UnitState { flags: flags::COMPLETED | flags::CONSTRUCTING | flags::UNDER_ATTACK, ..unit(99, SCV, at) });
    u.push(enemy(80, DRONE, (at.0 + 40, at.1)));
    let act = b.act(&frame(1_500, u.clone(), 0, 24));
    assert_eq!(branch(&b), Branch::Macro);
    let guards = unit_orders(&act, |o| *o == CmdOrder::Attack(Target::Unit(80)));
    assert_eq!(guards.iter().map(Vec::len).collect::<Vec<_>>(), vec![2], "{:?}", act.cmds);
    assert!(!guards[0].contains(&99));
    u.iter_mut().filter(|x| x.id == 80).for_each(|x| x.pos = (at.0 + 5 * 32, at.1));
    b.act(&frame(1_548, u.clone(), 0, 24));
    assert!(b.react_busy().contains(&guards[0][0]));
    u.iter_mut().filter(|x| x.id == 80).for_each(|x| x.pos = (at.0 + 7 * 32, at.1));
    let act = b.act(&frame(1_556, u, 0, 24));
    assert!(unit_orders(&act, |o| matches!(o, CmdOrder::Gather(_))).contains(&guards[0]), "{:?}", act.cmds);
    assert!(b.react_busy().is_empty());
}

#[test]
fn terran_react_golden_digest_021_r8() {
    let mut b = teacher();
    let mut out: Vec<String> = Vec::new();
    for n in (8..=7_200u32).step_by(8) {
        let scvs = (4 + n / 300).min(22);
        let mut u = if n < 1_600 {
            let mut v = vec![unit(1, CC, home()), UnitState { owner: 11, ..unit(3, 176, (home().0 - 200, home().1)) }];
            v.extend((0..scvs).map(|i| unit(100 + i, SCV, (home().0 - 100, home().1 + 4 * i as u16))));
            v
        } else {
            base(scvs)
        };
        if (2_800..3_100).contains(&n) {
            let m = depot_center(THEIRS);
            u.push(enemy(80, HATCHERY, m));
            u.push(enemy(81, SPAWNING_POOL, (m.0 + 160, m.1)));
        }
        if (4_000..5_000).contains(&n) {
            u.push(UnitState { hp: 250, ..unit(60, BUNKER, (home().0 + 5 * 32, home().1 + 64)) });
            u.extend((0..6).map(|i| unit(200 + i, MARINE, (home().0 + 40, home().1 + 100 + 8 * i as u16))));
            u.extend((0..4).map(|i| enemy(90 + i, ZERGLING, (home().0 + 5 * 32 + 10 * i as u16, home().1 + 128))));
        }
        if n % 96 == 0 {
            u.iter_mut().filter(|x| x.unit_type == SCV).take(2).for_each(|x| x.flags = flags::COMPLETED | flags::IDLE | flags::CARRYING);
        }
        let act = b.act(&frame(n, u, 120 + (n % 400) as i32, (2 * scvs) as u16));
        out.push(format!("{n} {:?}", act.cmds));
    }
    let mut h: u64 = 0xcbf29ce484222325;
    for byte in out.join("\n").bytes() {
        h ^= u64::from(byte);
        h = h.wrapping_mul(0x100000001b3);
    }
    assert_eq!(h, 0x99c9cddbf89e4ac5, "terran_react commands changed");
}

#[test]
fn build_id_021_r8() {
    let t = T0Build::TerranReact;
    assert_eq!((t.name(), t.id_tag(), t.race(), t.needs_map_hints()), ("terran_react", Some("react"), Some(Race::Terran), true));
    assert!(Kit::for_build(Race::Terran, t).react.is_some());
    assert!(Kit::for_build(Race::Zerg, t).react.is_none());
    assert!(Kit::terran().react.is_none() && Kit::terran_drop().react.is_none() && !T0Build::TerranDrop.needs_map_hints());
}
