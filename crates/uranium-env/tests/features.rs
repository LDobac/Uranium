//! 030 / 031 / 034 acceptance tests on synthetic states.

mod common;

use common::*;
use proptest::prelude::*;
use uranium_env::actions::{CAPABILITIES, LOC_CLASSES, TargetKind};
use uranium_env::features::augment::{Sym, transform, transform_target_loc};
use uranium_env::features::executor::{HostAction, execute, pick_worker};
use uranium_env::features::extract::action_table;
use uranium_env::features::world::World;
use uranium_env::features::*;
use uranium_proto::{Command, GameEvent};

fn observe(techs: bool) -> (Observation, uranium_proto::RawFrame, AgentMemory, MapInfo) {
    let map = map_info();
    let mut mem = AgentMemory::new(&game_info());
    let f = frame(100, rich_units().units, techs);
    let o = extract(&f, &mut mem, &map, None);
    (o, f, mem, map)
}

fn cap_rows<'a>(o: &'a Observation, name: &str) -> impl Iterator<Item = usize> + 'a {
    let k = CAPABILITIES.iter().position(|c| *c == name).unwrap();
    (0..o.obs.n).filter(move |&i| o.masks.unit_cap[i * CAPABILITIES.len() + k])
}

/// 034-AC2: every action type is enabled in one of the synthetic states, and whenever it is,
/// its selection / target / location masks are non-empty.
#[test]
fn every_action_has_arguments_when_allowed_034_ac2() {
    let table = action_table();
    let mut covered = vec![false; table.actions.len()];
    for techs in [true, false] {
        let (o, ..) = observe(techs);
        for a in &table.actions {
            if !o.masks.type_mask[a.id as usize] {
                continue;
            }
            covered[a.id as usize] = true;
            if a.uses_select {
                assert!(cap_rows(&o, &a.select_cap).any(|i| o.obs.entity_cat[i * 4 + 1] == OWNER_SELF), "{}: empty selection", a.name);
            }
            match a.target {
                TargetKind::Unit => assert!(cap_rows(&o, &a.target_cap).next().is_some(), "{}: empty target", a.name),
                TargetKind::Loc => {
                    let l = LOC_CLASSES.iter().position(|c| *c == a.loc_class).unwrap();
                    assert!(o.masks.loc_class_mask[l * CELLS..(l + 1) * CELLS].iter().any(|&b| b), "{}: empty location", a.name);
                }
                TargetKind::None => {}
            }
        }
    }
    let missing: Vec<&str> = table.actions.iter().filter(|a| !covered[a.id as usize]).map(|a| a.name.as_str()).collect();
    assert!(missing.is_empty(), "never enabled: {missing:?}");
}

/// 030-AC2: units that are not visible now do not leak; only remembered buildings (030-R2)
/// and the observed-maximum scalar remain.
#[test]
fn hidden_enemies_do_not_leak_030_ac2() {
    let map = map_info();
    let mut mem = AgentMemory::new(&game_info());
    let mut b = Builder::new();
    b.add(SELF, "Terran_SCV", (10, 10));
    let ling = b.add(ENEMY, "Zerg_Zergling", (40, 40));
    let hatch = b.add(ENEMY, "Zerg_Hatchery", (60, 60));
    let seen = frame(10, b.units.clone(), false);
    extract(&seen, &mut mem, &map, None);

    // Both enemies leave vision: their tiles are no longer visible.
    let own: Vec<_> = b.units.iter().filter(|u| u.owner == SELF).cloned().collect();
    let mut later = frame(20, own, false);
    for y in 30..90 {
        for x in 30..90 {
            later.visibility.set(x, y, false);
        }
    }
    let o = extract(&later, &mut mem, &map, None);
    assert!(!o.entity_ids.contains(&ling), "hidden zergling leaked as an entity");
    let i = o.entity_ids.iter().position(|&id| id == hatch).expect("hatchery remembered (030-R2)");
    let cont = &o.obs.entity_cont[i * ENTITY_CONT..(i + 1) * ENTITY_CONT];
    assert_eq!(cont[72], 1.0, "remembered flag");
    assert!((cont[73] - 10.0 / 2048.0).abs() < 1e-6, "frames since seen");
    let enemy_ground = &o.obs.spatial[ch::ENEMY_GROUND * CELLS..(ch::ENEMY_GROUND + 1) * CELLS];
    assert!(enemy_ground.iter().all(|&v| v == 0.0), "hidden unit in the density channel");
    assert_eq!(o.obs.spatial[ch::VISIBLE_NOW * CELLS + 40 * MAP + 40], 0.0);
    // The remembered building is not attackable while unseen.
    let k = CAPABILITIES.iter().position(|c| *c == "attackable_enemy").unwrap();
    assert!(!o.masks.unit_cap[i * CAPABILITIES.len() + k]);

    // Once its tile is visible again and it is gone, it is forgotten.
    let mut b2 = Builder::new();
    b2.add(SELF, "Terran_SCV", (10, 10));
    let o = extract(&frame(30, b2.units, false), &mut mem, &map, None);
    assert!(!o.entity_ids.contains(&hatch));
}

/// AMB-030-1: last_seen_age is 0 when visible, grows with time since last seen, 1 when never seen.
#[test]
fn last_seen_age_is_a_scouting_record_030() {
    let map = map_info();
    let mut mem = AgentMemory::new(&game_info());
    let mut first = frame(100, vec![], false);
    for y in 0..H {
        for x in 0..W {
            first.visibility.set(x, y, x < 40);
        }
    }
    extract(&first, &mut mem, &map, None);
    let mut later = frame(100 + 2400, vec![], false);
    for y in 0..H {
        for x in 0..W {
            later.visibility.set(x, y, x < 10);
        }
    }
    let o = extract(&later, &mut mem, &map, None);
    let age = |x: usize, y: usize| o.obs.spatial[ch::LAST_SEEN_AGE * CELLS + y * MAP + x];
    assert_eq!(age(5, 5), 0.0, "visible now");
    let expect = (1.0f32 + 2400.0).ln() / (1.0f32 + 86400.0).ln();
    assert!((age(20, 5) - expect).abs() < 1e-5, "seen 2400 frames ago: {}", age(20, 5));
    assert_eq!(age(60, 5), 1.0, "never seen");
    assert_eq!(age(120, 120), 0.0, "outside the map stays padding");
}

/// 030-AC4: every symmetry followed by its inverse restores the observation exactly, including
/// location targets.
#[test]
fn symmetry_round_trip_030_ac4() {
    let (o, ..) = observe(true);
    for sym in Sym::all() {
        let t = transform(&o, sym);
        if sym != Sym::IDENTITY {
            assert_ne!(t.obs.spatial, o.obs.spatial, "{sym:?} changed nothing");
        }
        let back = transform(&t, sym.inverse());
        assert_eq!(back, o, "{sym:?} is not inverted exactly");
        for a in action_table().actions.iter().filter(|a| a.target == TargetKind::Loc) {
            for tile in [0, 5 * MAP + 7, 40 * MAP + 33, 90 * MAP + 3] {
                let fwd = transform_target_loc(a.id, tile, o.map_size, false, sym);
                let back = transform_target_loc(a.id, fwd, sym.size(o.map_size), sym.transpose, sym.inverse());
                assert_eq!(back, tile, "{} {sym:?} tile {tile}", a.name);
            }
        }
    }
}

/// 030-AC5: `ObsTensors` has exactly these fields. Adding one (e.g. privileged data) breaks
/// this exhaustive pattern at compile time; privileged data lives in `PrivilegedTensors`.
#[test]
fn obs_tensors_have_no_privileged_field_030_ac5() {
    let (o, ..) = observe(false);
    let ObsTensors { n, entity_cat, entity_cont, entity_cell, entity_pos, spatial, scalar, scalar_cat, z_build, z_units, z_present } = o.obs;
    let d = uranium_env::features::extract::dims();
    assert_eq!(entity_cat.len(), n * ENTITY_CAT);
    assert_eq!(entity_cont.len(), n * ENTITY_CONT);
    assert_eq!((entity_cell.len(), entity_pos.len()), (n, 2 * n));
    assert_eq!(spatial.len(), SPATIAL * CELLS);
    assert_eq!(scalar.len(), d.scalar());
    assert_eq!(z_units.len(), d.z_units());
    assert_eq!((scalar_cat[0], z_build, z_present), (RACE_ZERG, [0; Z_BUILD], false));
    let p = privileged(&uranium_proto::PrivilegedSummary::default());
    assert_eq!(p.summary.len(), d.privileged());
}

/// 030-R1 / 031-R7: same inputs, same outputs.
#[test]
fn extraction_is_deterministic_030_r1() {
    let (a, ..) = observe(true);
    let (b, ..) = observe(true);
    assert_eq!(a, b);
}

/// 030-AC3: feature + mask extraction for 512 entities <= 1 ms on the CPU. Timing is only
/// meaningful in release: cargo test --release -p uranium-env --test features -- --ignored --nocapture
#[test]
#[ignore]
fn extraction_512_entities_under_1ms_030_ac3() {
    let map = map_info();
    let mut b = rich_units();
    let mut i = 0;
    while b.units.len() < 560 {
        let owner = if i % 2 == 0 { SELF } else { ENEMY };
        b.add(owner, if owner == SELF { "Terran_Marine" } else { "Zerg_Zergling" }, (5 + (i % 80) as i32, 60 + (i / 80) as i32));
        i += 1;
    }
    let f = frame(1000, b.units, true);
    let mut mem = AgentMemory::new(&game_info());
    let o = extract(&f, &mut mem, &map, None);
    assert_eq!(o.obs.n, N_MAX);
    for _ in 0..10 {
        std::hint::black_box(extract(&f, &mut mem, &map, None));
    }
    let iters = 50;
    let t = std::time::Instant::now();
    for _ in 0..iters {
        std::hint::black_box(extract(&f, &mut mem, &map, None));
    }
    let ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
    eprintln!("extract + masks, 512 entities: {ms:.3} ms");
    assert!(ms <= 1.0, "{ms:.3} ms > 1 ms (030-AC3)");
}

/// Breakdown of the 030-AC3 budget (release, --ignored --nocapture).
#[test]
#[ignore]
fn extraction_breakdown_030_ac3() {
    use uranium_env::features::masks::{loc_masks, type_mask, unit_caps};
    let map = map_info();
    let mut b = rich_units();
    let mut i = 0;
    while b.units.len() < 560 {
        let owner = if i % 2 == 0 { SELF } else { ENEMY };
        b.add(owner, if owner == SELF { "Terran_Marine" } else { "Zerg_Zergling" }, (5 + (i % 80) as i32, 60 + (i / 80) as i32));
        i += 1;
    }
    let f = frame(1000, b.units, true);
    let mut mem = AgentMemory::new(&game_info());
    extract(&f, &mut mem, &map, None);
    let time = |name: &str, g: &mut dyn FnMut()| {
        let t = std::time::Instant::now();
        for _ in 0..50 {
            g();
        }
        eprintln!("{name}: {:.3} ms", t.elapsed().as_secs_f64() * 1e3 / 50.0);
    };
    time("observe", &mut || mem.clone().observe(&f, uranium_env::types::TypeTables::get()));
    time("world", &mut || {
        std::hint::black_box(World::new(&f, &mem, &map));
    });
    let w = World::new(&f, &mem, &map);
    time("caps", &mut || {
        for e in w.entities.iter().take(512) {
            std::hint::black_box(unit_caps(&w, e));
        }
    });
    time("locs", &mut || {
        std::hint::black_box(loc_masks(&w));
    });
    let caps: Vec<_> = w.entities.iter().take(512).map(|e| unit_caps(&w, e)).collect();
    let locs = loc_masks(&w);
    time("type_mask", &mut || {
        std::hint::black_box(type_mask(&w, action_table(), &caps, &locs));
    });
    time("extract", &mut || {
        std::hint::black_box(extract(&f, &mut mem.clone(), &map, None));
    });
}

fn host(action: &str, selected: Vec<usize>, target_unit: Option<usize>, target_loc: Option<usize>) -> HostAction {
    let a = action_table().by_name(action).unwrap();
    HostAction { action: a.id, delay_bucket: 0, queued: false, selected, target_unit, target_loc }
}

/// 031-R1: Train goes to the eligible producer with the shortest queue (ties by id).
#[test]
fn pick_building_shortest_queue_031_r1() {
    let (o, f, mem, map) = observe(true);
    let w = World::new(&f, &mem, &map);
    let out = execute(&w, &o, &host("Train_Marine", vec![], None, None)).unwrap();
    let Command::Train { building, .. } = out.commands[0] else { panic!("{out:?}") };
    let b = f.units.iter().find(|u| u.id == building).unwrap();
    assert_eq!(b.build_queue[0], ty("None"), "busy barracks chosen over the idle one");
    // The machine-shop factory builds tanks; the bare factory does not.
    let out = execute(&w, &o, &host("Train_SiegeTank", vec![], None, None)).unwrap();
    let Command::Train { building, .. } = out.commands[0] else { panic!() };
    assert!(w.addon_of.get(&building).is_some());
}

/// 031-R2 / 031-AC4: the nearest non-carrying SCV builds at exactly the chosen tile.
#[test]
fn build_uses_nearest_free_scv_and_exact_tile_031_ac4() {
    let (o, f, mem, map) = observe(true);
    let w = World::new(&f, &mem, &map);
    let l = LOC_CLASSES.iter().position(|c| *c == "buildable_3x2").unwrap();
    let tile = (0..CELLS).find(|&t| o.masks.loc_class_mask[l * CELLS + t] && t % MAP > 60 && t / MAP > 20).unwrap();
    let out = execute(&w, &o, &host("Build_SupplyDepot", vec![], None, Some(tile))).unwrap();
    let Command::Build { worker, tile: built, .. } = out.commands[0] else { panic!() };
    assert_eq!(built, ((tile % MAP) as u16, (tile / MAP) as u16));
    let chosen = pick_worker(&w, ((tile % MAP) as i32, (tile / MAP) as i32)).unwrap();
    assert_eq!(chosen.id, worker);
    assert!(!chosen.has(uranium_proto::flags::CARRYING));
}

/// 031-R10: a failed command is visible in the next observation's scalars.
#[test]
fn command_failure_reaches_scalars_031_r10() {
    let map = map_info();
    let mut mem = AgentMemory::new(&game_info());
    let mut f = frame(10, rich_units().units, true);
    f.events.push(GameEvent::CommandFailed { cmd_index: 0, error: 1 });
    let o = extract(&f, &mut mem, &map, None);
    assert_eq!(*o.obs.scalar.last().unwrap(), 1.0);
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 400, .. ProptestConfig::default() })]

    /// 031-AC2: random policy outputs are executed only when they satisfy every mask.
    #[test]
    fn masked_actions_never_execute_031_ac2(action in 0u16..100, sel in proptest::collection::vec(0usize..80, 0..13),
                                            tu in proptest::option::of(0usize..80), tl in proptest::option::of(0usize..CELLS), techs: bool) {
        let (o, f, mem, map) = observe(techs);
        let w = World::new(&f, &mem, &map);
        let a = &action_table().actions[action as usize];
        let act = HostAction { action, delay_bucket: 0, queued: false, selected: sel.clone(), target_unit: tu, target_loc: tl };
        if let Ok(out) = execute(&w, &o, &act) {
            prop_assert!(o.masks.type_mask[action as usize]);
            let k = CAPABILITIES.len();
            let cap = |i: usize, name: &str| o.masks.unit_cap[i * k + CAPABILITIES.iter().position(|c| *c == name).unwrap()];
            if a.uses_select {
                prop_assert!(!sel.is_empty() && sel.len() <= a.max_select as usize);
                for &i in &sel {
                    prop_assert!(i < o.obs.n && cap(i, &a.select_cap));
                }
            }
            if a.target == TargetKind::Unit {
                prop_assert!(cap(tu.unwrap(), &a.target_cap));
            }
            if a.target == TargetKind::Loc {
                let l = LOC_CLASSES.iter().position(|c| *c == a.loc_class).unwrap();
                prop_assert!(o.masks.loc_class_mask[l * CELLS + tl.unwrap()]);
            }
            prop_assert!(out.commands.len() <= 1, "031-R3: one command per action");
        }
    }
}
