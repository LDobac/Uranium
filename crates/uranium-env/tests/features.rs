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
    time("observe", &mut || mem.clone().observe(&f, uranium_env::types::TypeTables::get(), &map));
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

/// 034-R3: footprint tiles from BWAPI centre positions (STARDATA game: Factory at (544, 3856) with
/// its Machine Shop at (640, 3872)); the shop's asymmetric pixel dimensions must not shift it.
#[test]
fn addon_footprint_matches_bwapi_positions_034_r3() {
    let tables = uranium_env::types::TypeTables::get();
    let at = |name: &str, pos: (u16, u16)| {
        let t = tables.unit_by_name(name).unwrap();
        uranium_env::features::world::top_left(&uranium_proto::UnitState { unit_type: t.id as u16, pos, ..Default::default() }, t)
    };
    let (fx, fy) = at("Terran_Factory", (544, 3856));
    assert_eq!((fx, fy), (15, 119));
    assert_eq!(at("Terran_Machine_Shop", (640, 3872)), (fx + 4, fy + 1));
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
    fn masked_actions_never_execute_031_ac2(action in 0u16..action_table().actions.len() as u16, sel in proptest::collection::vec(0usize..80, 0..13),
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

/// 030-AC6: the four training flips keep the map size and footprints (no transpose), remap the
/// own start index to the start location the own base moved to, and are deterministic.
#[test]
fn training_flips_keep_footprints_and_start_loc_030_ac6() {
    use uranium_env::features::augment::transform_with_starts;
    let (o, _, _, map) = observe(true);
    let (w, h) = o.map_size;
    assert_eq!(Sym::augment().len(), 4);
    assert!(Sym::augment().iter().all(|s| !s.transpose), "training uses no transposes");
    // A start list that the flips map exactly onto itself: corners of a w x h map (4x3 depots).
    let corners: Vec<uranium_proto::Tile> = vec![(4, 4), ((w - 8) as u16, 4), (4, (h - 7) as u16), ((w - 8) as u16, (h - 7) as u16)];
    let mut own0 = o.clone();
    own0.obs.scalar_cat[2] = 0;
    for (k, sym) in Sym::augment().into_iter().enumerate() {
        let t = transform_with_starts(&own0, sym, &corners);
        assert_eq!(t.map_size, (w, h), "{sym:?}");
        assert!(!t.transposed, "{sym:?}: footprints stay w x h");
        assert_eq!(t.obs.scalar_cat[2] as usize, k, "{sym:?}: own start corner 0 moves to corner {k}");
        assert_eq!(t, transform_with_starts(&own0, sym, &corners), "{sym:?} is deterministic");

        // The test map's own start: the new index names the map start nearest to where the own
        // depot moved, and that place is a start location in the transformed spatial channel.
        let t = transform_with_starts(&o, sym, &map.start_locations);
        let own = map.start_locations[o.obs.scalar_cat[2] as usize];
        let (nx, ny, fw, fh) = sym.rect((w, h), own.0 as usize, own.1 as usize, 4, 3);
        let dist = |s: &uranium_proto::Tile| (s.0 as i64 - nx as i64).pow(2) + (s.1 as i64 - ny as i64).pow(2);
        let best = map.start_locations.iter().map(dist).min().unwrap();
        assert_eq!(dist(&map.start_locations[t.obs.scalar_cat[2] as usize]), best, "{sym:?}");
        let ch = |x: usize, y: usize| t.obs.spatial[ch::START_LOC * CELLS + y * MAP + x];
        for dy in 0..fh {
            for dx in 0..fw {
                assert_eq!(ch(nx + dx, ny + dy), 1.0, "{sym:?}: own depot tile ({}, {}) not a start location", nx + dx, ny + dy);
            }
        }
    }
}

fn cap_of(o: &Observation, i: usize, name: &str) -> bool {
    o.masks.unit_cap[i * CAPABILITIES.len() + CAPABILITIES.iter().position(|c| *c == name).unwrap()]
}

fn index_of(o: &Observation, id: u32) -> usize {
    o.entity_ids.iter().position(|&e| e == id).unwrap_or_else(|| panic!("entity {id} missing"))
}

/// 030 §3 / E-1: each hazard paints the tiles its own radius overlaps (game data table in
/// `hazard_radius`), Plague and Ensnare included; a radius sent by the bridge wins.
#[test]
fn hazard_uses_effect_radius_030_e1() {
    use uranium_env::features::extract::hazard_radius;
    use uranium_proto::{Effect, EffectKind};
    assert_eq!(
        [EffectKind::PsionicStorm, EffectKind::DarkSwarm, EffectKind::DisruptionWeb, EffectKind::NukeDot, EffectKind::EmpShockwave, EffectKind::Irradiate, EffectKind::Plague, EffectKind::Ensnare, EffectKind::Other(1)]
            .map(hazard_radius),
        [48, 80, 60, 256, 64, 32, 64, 64, 16]
    );
    let map = map_info();
    let mut mem = AgentMemory::new(&game_info());
    let mut f = frame(10, vec![], false);
    let ef = |kind, pos, radius| Effect { kind, pos, radius, remaining_frames: None, source_owner: None };
    f.effects = vec![ef(EffectKind::PsionicStorm, (1000, 1000), 0), ef(EffectKind::Plague, (1000, 2000), 0), ef(EffectKind::Ensnare, (2000, 1000), 0), ef(EffectKind::Other(3), (2500, 2500), 100)];
    let o = extract(&f, &mut mem, &map, None);
    let at = |c: usize, x: usize, y: usize| o.obs.spatial[c * CELLS + y * MAP + x];
    // Storm (r 48) at pixel 1000 = tile 31.25: tiles 29..=32 on its row, not 28 or 33.
    let storm: Vec<usize> = (26..37).filter(|&x| at(ch::HAZARD_STORM, x, 31) == 1.0).collect();
    assert_eq!(storm, vec![29, 30, 31, 32]);
    // Plague (r 64): tiles 29..=33; diagonal (29, 60) overlaps, (28, 59) does not.
    let plague: Vec<usize> = (26..37).filter(|&x| at(ch::HAZARD_OTHER, x, 62) == 1.0).collect();
    assert_eq!(plague, vec![29, 30, 31, 32, 33]);
    assert_eq!(at(ch::HAZARD_OTHER, 29, 60), 1.0);
    assert_eq!(at(ch::HAZARD_OTHER, 28, 59), 0.0);
    // Ensnare (r 64) also goes to the "other" channel.
    assert_eq!(at(ch::HAZARD_OTHER, 62, 31), 1.0);
    assert_eq!(at(ch::HAZARD_OTHER, 64, 31), 1.0);
    assert_eq!(at(ch::HAZARD_OTHER, 65, 31), 0.0);
    // A bridge radius (100) overrides the 16 px default of other projectiles.
    assert_eq!(at(ch::HAZARD_OTHER, 78 - 3, 78), 1.0);
    assert_eq!(at(ch::HAZARD_STORM, 31, 62), 0.0, "plague is not a storm");
}

/// AMB-030-4 / 010-R5: an effect whose centre tile is not visible now is not observed, even when
/// its circle reaches visible tiles (the OpenBW shim sends every bullet).
#[test]
fn hazard_ignores_effects_with_unseen_centre_amb_030_4() {
    use uranium_proto::{Effect, EffectKind};
    let map = map_info();
    let mut mem = AgentMemory::new(&game_info());
    let mut f = frame(10, vec![], false);
    let ef = |kind, pos| Effect { kind, pos, radius: 0, remaining_frames: None, source_owner: None };
    f.effects = vec![ef(EffectKind::PsionicStorm, (1000, 1000)), ef(EffectKind::PsionicStorm, (2000, 2000))];
    // Hide only the second storm's centre tile (62, 62); its neighbours stay visible.
    let (w, h) = (f.visibility.width, f.visibility.height);
    let mut bits = vec![1u8; w as usize * h as usize];
    bits[62 * w as usize + 62] = 0;
    f.visibility = uranium_proto::BitGrid::from_bytes(w, h, &bits);
    let o = extract(&f, &mut mem, &map, None);
    let at = |c: usize, x: usize, y: usize| o.obs.spatial[c * CELLS + y * MAP + x];
    assert_eq!(at(ch::HAZARD_STORM, 31, 31), 1.0, "visible storm is observed");
    for (x, y) in [(61, 62), (62, 62), (63, 62), (62, 61)] {
        assert_eq!(at(ch::HAZARD_STORM, x, y), 0.0, "storm with an unseen centre leaks at ({x}, {y})");
    }
}

/// 030-R4 / B-6: a flipped even-width building's cell is the cell of the mirrored building
/// (W = 96, Barracks x0 = 20 -> mirrored x0 72, centre tile 74), recomputed from the pixel.
#[test]
fn flip_cell_from_pixel_even_width_030_r4() {
    let (o, f, ..) = observe(true);
    let rax_pos = ((20 * 32 + 64) as u16, (10 * 32 + 48) as u16);
    let rax = f.units.iter().find(|u| u.unit_type == ty("Terran_Barracks") && u.pos == rax_pos).unwrap().id;
    let i = index_of(&o, rax);
    assert_eq!(o.obs.entity_cell[i], (11 * MAP + 22) as i32);
    let t = transform(&o, Sym { flip_x: true, flip_y: false, transpose: false });
    assert_eq!(t.obs.entity_cell[i], (11 * MAP + 74) as i32, "x: 96*32 - 704 = 2368 -> tile 74 (B-6)");
    assert_eq!(t.obs.entity_pos[i * 2], 2368.0);
    let t = transform(&o, Sym { flip_x: false, flip_y: true, transpose: false });
    assert_eq!(t.obs.entity_cell[i], (84 * MAP + 22) as i32, "y: 96*32 - 368 = 2704 -> tile 84");
}

fn map_with_resources(resources: Vec<MapResource>) -> MapInfo {
    let pre = MapPrecompute { version: PRECOMPUTE_VERSION, resources, ..Default::default() };
    MapInfo::new(&game_info(), 0, Some(&pre)).unwrap()
}

fn resource_at(name: &str, tile: (i32, i32), amount: u32) -> MapResource {
    let t = uranium_env::types::TypeTables::get().unit_by_name(name).unwrap();
    MapResource { unit_type: t.id as u16, pos: ((tile.0 * 32 + t.tile_width * 16) as u16, (tile.1 * 32 + t.tile_height * 16) as u16), amount }
}

fn neutrals(o: &Observation) -> usize {
    (0..o.obs.n).filter(|&i| o.obs.entity_cat[i * 4 + 1] == OWNER_NEUTRAL).count()
}

/// 030-R2 exception / 010-R5 / E-3: map resources out of vision are remembered neutral entities:
/// never seen -> map initial amount, `seen_frame` 0, not targetable (no id); seen -> last seen
/// amount and frame, real id, still a gather / right-click target; seen empty -> dropped. The
/// current amount out of vision never leaks.
#[test]
fn unseen_resources_are_remembered_entities_030_r2() {
    let far = resource_at("Resource_Mineral_Field", (70, 20), 777);
    let geyser = resource_at("Resource_Vespene_Geyser", (40, 60), 5000);
    let map = map_with_resources(vec![far, geyser]);
    let mut mem = AgentMemory::new(&game_info());
    let mut b = Builder::new();
    b.add(SELF, "Terran_SCV", (10, 10));
    let own: Vec<_> = b.units.clone();
    let mineral = b.add(NEUTRAL, "Resource_Mineral_Field", (70, 20));
    let blind = |f: &mut uranium_proto::RawFrame| {
        for y in 0..H {
            for x in 30..W {
                f.visibility.set(x, y, false);
            }
        }
    };
    let cont = |o: &Observation, i: usize| o.obs.entity_cont[i * ENTITY_CONT..(i + 1) * ENTITY_CONT].to_vec();

    // Frame 100: only x < 30 visible. Both map resources are remembered entities.
    let mut f = frame(100, own.clone(), false);
    blind(&mut f);
    let o = extract(&f, &mut mem.clone(), &map, None);
    assert_eq!(o, extract(&f, &mut mem.clone(), &map, None), "deterministic (030-R1)");
    let o = extract(&f, &mut mem, &map, None);
    assert_eq!(neutrals(&o), 2);
    let i = (0..o.obs.n).find(|&i| o.obs.entity_cat[i * 4 + 1] == OWNER_NEUTRAL && o.obs.entity_cat[i * 4] == far.unit_type as i32).unwrap();
    assert!(uranium_env::features::world::is_unseen_resource_id(o.entity_ids[i]));
    let c = cont(&o, i);
    assert_eq!((c[72], c[73], c[74]), (1.0, 100.0 / 2048.0, 777.0 / 1500.0), "remembered, seen_frame 0, map amount");
    assert_eq!(o.obs.entity_pos[i * 2..i * 2 + 2], [far.pos.0 as f32, far.pos.1 as f32]);
    assert!(!cap_of(&o, i, "target_resource") && !cap_of(&o, i, "any_unit"), "never seen: no id to target");

    // Frame 200: the field is seen with 600 left.
    let mut m = b.units.iter().find(|u| u.id == mineral).unwrap().clone();
    m.resources = 600;
    let mut units = own.clone();
    units.push(m.clone());
    let mut f = frame(200, units, false);
    blind(&mut f);
    for y in 15..25 {
        for x in 65..75 {
            f.visibility.set(x, y, true);
        }
    }
    let o = extract(&f, &mut mem, &map, None);
    let i = index_of(&o, mineral);
    assert_eq!(cont(&o, i)[72], 0.0);
    assert_eq!(neutrals(&o), 2, "the visible field replaces its map entry");

    // Frame 300: out of vision again (its real amount meanwhile 500, never reported): last seen
    // state with its real id.
    let mut f = frame(300, own.clone(), false);
    blind(&mut f);
    let o = extract(&f, &mut mem, &map, None);
    let i = index_of(&o, mineral);
    let c = cont(&o, i);
    assert_eq!((c[72], c[73], c[74]), (1.0, 100.0 / 2048.0, 600.0 / 1500.0), "last seen amount and frame");
    assert!(cap_of(&o, i, "target_resource") && cap_of(&o, i, "any_unit"), "seen before: still a target");
    assert_eq!(neutrals(&o), 2);

    // Frame 400: everything visible, the field is gone (mined out) and the geyser tile is empty.
    let o = extract(&frame(400, own.clone(), false), &mut mem, &map, None);
    assert!(!o.entity_ids.contains(&mineral));
    assert_eq!(neutrals(&o), 0);
    let mut f = frame(500, own.clone(), false);
    blind(&mut f);
    assert_eq!(neutrals(&extract(&f, &mut mem, &map, None)), 0, "seen empty stays dropped");
}

/// 030-R2 / E-3: building a refinery morphs the geyser unit itself (same id, now a player's
/// Refinery). The geyser memory goes with it: out of sight there is only the remembered enemy
/// refinery (no duplicate id, no refinery_slot); once the refinery dies and the geyser is seen
/// again, the geyser memory is back.
#[test]
fn geyser_morphed_into_refinery_is_forgotten_030_r2() {
    let geyser_res = resource_at("Resource_Vespene_Geyser", (40, 60), 5000);
    let map = map_with_resources(vec![geyser_res]);
    let mut mem = AgentMemory::new(&game_info());
    let mut b = Builder::new();
    b.add(SELF, "Terran_SCV", (10, 10));
    let own = b.units.clone();
    let g = b.add(NEUTRAL, "Resource_Vespene_Geyser", (40, 60));
    let geyser = b.units.iter().find(|u| u.id == g).unwrap().clone();
    let blind = |f: &mut uranium_proto::RawFrame| {
        for y in 0..H {
            for x in 30..W {
                f.visibility.set(x, y, false);
            }
        }
    };
    let refinery_slot = |o: &Observation| {
        let l = LOC_CLASSES.iter().position(|c| *c == "refinery_slot").unwrap();
        o.masks.loc_class_mask[l * CELLS + 60 * MAP + 40]
    };
    let at_tile = |o: &Observation| (0..o.obs.n).filter(|&i| o.obs.entity_cell[i] == (61 * MAP + 42) as i32).count();

    extract(&frame(10, [own.clone(), vec![geyser.clone()]].concat(), false), &mut mem, &map, None);
    let mut refinery = geyser.clone();
    refinery.owner = ENEMY;
    refinery.unit_type = ty("Terran_Refinery");
    extract(&frame(20, [own.clone(), vec![refinery.clone()]].concat(), false), &mut mem, &map, None);
    let mut f = frame(30, own.clone(), false);
    blind(&mut f);
    let o = extract(&f, &mut mem, &map, None);
    assert_eq!(o.entity_ids.iter().filter(|&&id| id == g).count(), 1, "one entity for the id");
    assert_eq!(at_tile(&o), 1, "only the remembered enemy refinery");
    assert_eq!(o.obs.entity_cat[index_of(&o, g) * 4 + 1], OWNER_ENEMY);
    assert!(!refinery_slot(&o));

    // The refinery dies and the geyser is seen again (a new unit, or the same id): memory is back.
    for new_id in [999, g] {
        let mut mem = mem.clone();
        let mut back = geyser.clone();
        back.id = new_id;
        let mut f = frame(40, [own.clone(), vec![back]].concat(), false);
        f.events.push(GameEvent::UnitDestroyed(g));
        extract(&f, &mut mem, &map, None);
        let mut f = frame(50, own.clone(), false);
        blind(&mut f);
        let o = extract(&f, &mut mem, &map, None);
        let i = index_of(&o, new_id);
        assert!(!o.entity_ids.contains(&g) || new_id == g, "refinery forgotten");
        assert_eq!((o.obs.entity_cat[i * 4 + 1], at_tile(&o)), (OWNER_NEUTRAL, 1));
        assert_eq!(o.obs.entity_cont[i * ENTITY_CONT + 73], 10.0 / 2048.0, "seen at frame 40");
    }
}

/// 030-R7 / 030 §2 (Q3, 2026-10-19): past N_MAX the remembered (out-of-vision) resources go first;
/// own units, visible enemies and remembered enemy buildings stay.
#[test]
fn truncation_drops_remembered_resources_first_030_r7() {
    let res: Vec<MapResource> = (0..60).map(|k| resource_at("Resource_Mineral_Field", (40 + 2 * (k % 20), 50 + k / 20 * 2), 1500)).collect();
    let map = map_with_resources(res);
    let mut mem = AgentMemory::new(&game_info());
    let mut b = Builder::new();
    let hatch = b.add(ENEMY, "Zerg_Hatchery", (80, 80));
    let mut f = frame(10, b.units.clone(), false);
    for y in 45..70 {
        for x in 0..W {
            f.visibility.set(x, y, false);
        }
    }
    extract(&f, &mut mem, &map, None);
    b.units.clear();
    let visible_mineral = b.add(NEUTRAL, "Resource_Mineral_Field", (2, 2));
    for k in 0..400 {
        b.add(SELF, "Terran_Marine", (k % 25, 5 + k / 25));
    }
    for k in 0..100 {
        b.add(ENEMY, "Zerg_Zergling", (k % 25, 25 + k / 25 % 4));
    }
    let mut f = frame(20, b.units.clone(), false);
    for y in 0..H {
        for x in 30..W {
            f.visibility.set(x, y, false);
        }
    }
    let o = extract(&f, &mut mem, &map, None);
    assert_eq!(o.obs.n, N_MAX);
    for u in b.units.iter().filter(|u| u.owner != NEUTRAL) {
        assert!(o.entity_ids.contains(&u.id), "unit {} dropped", u.id);
    }
    assert!(o.entity_ids.contains(&hatch), "remembered enemy building kept");
    // 400 + 100 + 1 building + 1 visible mineral = 502: 10 of the 60 remembered fields fit.
    assert!(o.entity_ids.contains(&visible_mineral), "visible resource kept");
    let remembered = o.entity_ids.iter().filter(|&&id| uranium_env::features::world::is_unseen_resource_id(id)).count();
    assert_eq!(remembered, N_MAX - 502);
}

/// 034-R6 / B-11: a disabled unit (lockdown, stasis, maelstrom) has no selection class at all and
/// no executor picks it.
#[test]
fn disabled_units_take_no_command_034_r6() {
    let map = map_info();
    let mut mem = AgentMemory::new(&game_info());
    let mut b = Builder::new();
    b.add(SELF, "Terran_Command_Center", (10, 10));
    let locked_scv = b.add(SELF, "Terran_SCV", (30, 30));
    b.get(locked_scv).status_timers.lockdown = 50;
    let free_scv = b.add(SELF, "Terran_SCV", (60, 60));
    let tank = b.add(SELF, "Terran_Siege_Tank_Tank_Mode", (40, 40));
    b.get(tank).status_timers.stasis = 50;
    let ship = b.add(SELF, "Terran_Dropship", (42, 42));
    b.get(ship).status_timers.maelstrom = 20;
    let cargo = b.add(SELF, "Terran_Marine", (42, 42));
    b.get(cargo).loaded_in = Some(ship);
    b.get(cargo).flags |= uranium_proto::flags::LOADED;
    let rax = b.add(SELF, "Terran_Barracks", (20, 40));
    b.get(rax).status_timers.stasis = 50;
    b.get(rax).build_queue[0] = ty("Terran_Marine");
    let f = frame(10, b.units.clone(), true);
    let o = extract(&f, &mut mem, &map, None);
    for (id, caps) in [
        (locked_scv, &["is_worker", "can_move", "can_be_loaded"][..]),
        (tank, &["can_siege", "can_move", "can_attack"][..]),
        (ship, &["is_transport_loaded", "can_move"][..]),
        (rax, &["can_rally", "can_lift", "can_cancel_train"][..]),
    ] {
        let i = index_of(&o, id);
        for c in caps {
            assert!(!cap_of(&o, i, c), "disabled {id} has {c}");
        }
    }
    assert!(cap_of(&o, index_of(&o, free_scv), "is_worker"));
    // pick_worker skips the nearer locked SCV; Train_Marine has no producer.
    let w = World::new(&f, &mem, &map);
    assert_eq!(pick_worker(&w, (30, 30)).unwrap().id, free_scv);
    let train = action_table().by_name("Train_Marine").unwrap();
    assert!(!o.masks.type_mask[train.id as usize], "only a stasis-ed barracks");
}

/// 034-R7 / E-4: `is_building` covers landed, lifted and bunker; the executor accepts a building
/// only alone; SetRally selects one building.
#[test]
fn building_selected_alone_034_r7() {
    use uranium_env::features::executor::Rejected;
    let (o, f, mem, map) = observe(true);
    let id_of = |name: &str| f.units.iter().find(|u| u.unit_type == ty(name) && u.owner == SELF).unwrap().id;
    let bunker = index_of(&o, id_of("Terran_Bunker"));
    let marine = index_of(&o, id_of("Terran_Marine"));
    let firebat = index_of(&o, id_of("Terran_Firebat"));
    assert!(o.masks.is_building[bunker] && !o.masks.is_building[marine]);
    let lifted = f.units.iter().find(|u| u.unit_type == ty("Terran_Command_Center") && u.has(uranium_proto::flags::LIFTED)).unwrap().id;
    assert!(o.masks.is_building[index_of(&o, lifted)], "lifted buildings count");
    let _ = bunker;
    assert_eq!(o.masks.is_building.len(), o.obs.n);
    let w = World::new(&f, &mem, &map);
    // A lifted building moves like a unit (can_move), so it is the building of a mixed selection.
    let lifted = index_of(&o, lifted);
    assert!(cap_of(&o, lifted, "can_move"));
    let mv = |sel: Vec<usize>| execute(&w, &o, &host("Move_Pos", sel, None, Some(20 * 128 + 20)));
    assert_eq!(mv(vec![lifted, marine]), Err(Rejected::BadSelection));
    assert_eq!(mv(vec![marine, lifted]), Err(Rejected::BadSelection));
    assert!(mv(vec![lifted]).is_ok());
    assert!(mv(vec![marine, firebat]).is_ok());
    for name in ["SetRally_Pos", "SetRally_Unit"] {
        assert_eq!(action_table().by_name(name).unwrap().max_select, 1, "{name}");
    }
}

/// 030-R11 / E-6: a lobby-Random enemy is `unknown` until its first unit is observed, in live
/// games (GameInfo race Random) and replays (`enemy_random`; GameInfo holds the played race).
#[test]
fn random_enemy_race_unknown_until_seen_030_r11() {
    let map = map_info();
    let mut live = game_info();
    live.players[1].race = uranium_proto::Race::Random;
    for mut mem in [AgentMemory::new(&live), AgentMemory::with_lobby(&game_info(), true)] {
        let mut b = Builder::new();
        b.add(SELF, "Terran_SCV", (10, 10));
        let o = extract(&frame(10, b.units.clone(), false), &mut mem, &map, None);
        assert_eq!(o.obs.scalar_cat[0], RACE_UNKNOWN);
        b.add(ENEMY, "Zerg_Drone", (30, 30));
        let o = extract(&frame(20, b.units.clone(), false), &mut mem, &map, None);
        assert_eq!(o.obs.scalar_cat[0], RACE_ZERG);
    }
    // A race chosen in the lobby is known from the start.
    let o = extract(&frame(10, vec![], false), &mut AgentMemory::with_lobby(&game_info(), false), &map, None);
    assert_eq!(o.obs.scalar_cat[0], RACE_ZERG);
}

/// 030-R10 / E-5: with `zero_last_failed` (BC evaluation games) a failed command leaves the
/// `last_failed` scalar at 0, as in BC data.
#[test]
fn zero_last_failed_flag_030_r10() {
    let map = map_info();
    let mut mem = AgentMemory::new(&game_info());
    mem.zero_last_failed = true;
    let mut f = frame(10, rich_units().units, true);
    f.events.push(GameEvent::CommandFailed { cmd_index: 0, error: 1 });
    let o = extract(&f, &mut mem, &map, None);
    assert_eq!(*o.obs.scalar.last().unwrap(), 0.0);
}

/// 010-AC4: every 034 action type yields an executor `Command` (NoOp: none, by design) whose kind
/// matches the table's `command`, and that command survives the protocol round trip (postcard
/// framing, as the agent sends it in `Act`) unchanged.
#[test]
fn every_action_type_command_round_trips_010_ac4() {
    use uranium_proto::{Act, AgentMsg, read_msg, write_msg};
    let table = action_table();
    let mut done = vec![false; table.actions.len()];
    for techs in [true, false] {
        let (o, f, mem, map) = observe(techs);
        let w = World::new(&f, &mem, &map);
        for a in &table.actions {
            if done[a.id as usize] {
                continue;
            }
            let Some(act) = valid_action(&o, a, &|_, i| Some(i as i64), &|t| Some(t as i64)) else { continue };
            let out = execute(&w, &o, &act).unwrap_or_else(|e| panic!("{}: {e:?}", a.name));
            if a.id == 0 {
                assert!(out.commands.is_empty(), "NoOp sends nothing");
                done[0] = true;
                continue;
            }
            assert_eq!(out.commands.len(), 1, "{}: one command (031-R3)", a.name);
            let cmd = &out.commands[0];
            assert_eq!(command_name(cmd), a.command, "{}", a.name);
            let msg = AgentMsg::Act(Act { cmds: out.commands.clone(), advance_frames: 1, valid_until_frame: None, surrender: None });
            let mut buf = Vec::new();
            write_msg(&mut buf, &msg).unwrap();
            let back: AgentMsg = read_msg(&mut buf.as_slice()).unwrap();
            assert_eq!(back, msg, "{}", a.name);
            done[a.id as usize] = true;
        }
    }
    let missing: Vec<&str> = table.actions.iter().filter(|a| !done[a.id as usize]).map(|a| a.name.as_str()).collect();
    assert!(missing.is_empty(), "no command for {missing:?}");
}

/// 034-R8: a building constructing its addon takes no Train / Research / Upgrade / Lift (the game
/// answers Unit_Busy). Seen either as BWAPI `isConstructing` on the completed building or as an
/// incomplete addon in its slot (replay observations).
#[test]
fn building_constructing_addon_is_busy_034_r8() {
    let map = map_info();
    let vulture = action_table().by_name("Train_Vulture").unwrap().id as usize;
    for flag_only in [true, false] {
        let mut b = Builder::new();
        b.add(SELF, "Terran_Command_Center", (10, 10));
        b.add(SELF, "Terran_SCV", (16, 16));
        let fac = b.add(SELF, "Terran_Factory", (26, 10));
        if flag_only {
            b.get(fac).flags |= uranium_proto::flags::CONSTRUCTING;
        } else {
            b.get(fac).has_addon = true;
            let shop = b.add(SELF, "Terran_Machine_Shop", (30, 11));
            b.get(shop).flags &= !uranium_proto::flags::COMPLETED;
        }
        let f = frame(10, b.units.clone(), false);
        let mut mem = AgentMemory::new(&game_info());
        let o = extract(&f, &mut mem, &map, None);
        assert!(!o.masks.type_mask[vulture], "flag_only={flag_only}: busy factory trains");
        assert!(!cap_of(&o, index_of(&o, fac), "can_lift"), "flag_only={flag_only}: busy factory lifts");
        // A second, idle factory is the one picked.
        let idle = b.add(SELF, "Terran_Factory", (26, 30));
        let f = frame(20, b.units.clone(), false);
        let o = extract(&f, &mut mem, &map, None);
        assert!(o.masks.type_mask[vulture]);
        let w = World::new(&f, &mem, &map);
        let out = execute(&w, &o, &host("Train_Vulture", vec![], None, None)).unwrap();
        assert!(matches!(out.commands[0], Command::Train { building, .. } if building == idle), "{out:?}");
    }
}

/// 034-R9: a unit never targets itself. The executor rejects it; a label targeting one of its own
/// selected units keeps the others (the game skips the target) and is void when none remain.
#[test]
fn no_self_target_034_r9() {
    use uranium_env::features::executor::Rejected;
    use uranium_env::features::label::{Label, host_label_check};
    let (o, f, mem, map) = observe(true);
    let w = World::new(&f, &mem, &map);
    let marines: Vec<u32> =
        f.units.iter().filter(|u| u.unit_type == ty("Terran_Marine") && u.loaded_in.is_none() && u.status_timers.lockdown == 0).map(|u| u.id).collect();
    let (m0, m1) = (index_of(&o, marines[0]), index_of(&o, marines[1]));
    for name in ["Move_Unit", "Smart_Unit"] {
        assert_eq!(execute(&w, &o, &host(name, vec![m0], Some(m0), None)), Err(Rejected::BadTarget), "{name}");
        assert!(execute(&w, &o, &host(name, vec![m0], Some(m1), None)).is_ok(), "{name}");
    }
    let mv = action_table().by_name("Move_Unit").unwrap().id;
    let label = |selected: Vec<u32>| Label { frame: 0, action: mv, delay: 8, queued: false, selected, target_unit: Some(marines[0]), target_tile: None };
    let (h, why) = host_label_check(&label(vec![marines[0], marines[1]]), &o);
    assert_eq!((h.selected, why), (vec![m1], None));
    assert_eq!(host_label_check(&label(vec![marines[0]]), &o).1, Some("target_selected"));
    // SetRally_Unit is the exception (`self_target`): a rally on the building itself clears it.
    let rax = f.units.iter().find(|u| u.unit_type == ty("Terran_Barracks") && !u.has(uranium_proto::flags::LIFTED)).unwrap().id;
    let r = index_of(&o, rax);
    assert!(execute(&w, &o, &host("SetRally_Unit", vec![r], Some(r), None)).is_ok());
    let rally = action_table().by_name("SetRally_Unit").unwrap().id;
    let l = Label { frame: 0, action: rally, delay: 8, queued: false, selected: vec![rax], target_unit: Some(rax), target_tile: None };
    assert_eq!(host_label_check(&l, &o).1, None);
}

/// Action audit F7 (2026-10-19): `Unload_Unit` selects a transport with cargo and targets an own
/// loaded unit; the executor only sends it for a unit in the selected transport (031-R4, BWAPI
/// `canUnload`) and the protocol carries `CmdOrder::Unload`.
#[test]
fn unload_unit_masks_and_executor_f7() {
    use uranium_env::features::executor::Rejected;
    use uranium_env::features::label::{Label, LabelWalker, NOT_IN_TRANSPORT_REASON, host_label_check, unload_target_check};
    let mut b = rich_units();
    let ship2 = b.add(SELF, "Terran_Dropship", (24, 20));
    let cargo2 = b.add(SELF, "Terran_Firebat", (24, 20));
    b.get(cargo2).loaded_in = Some(ship2);
    b.get(cargo2).flags |= uranium_proto::flags::LOADED;
    let empty_ship = b.add(SELF, "Terran_Dropship", (26, 20));
    let f = frame(100, b.units.clone(), true);
    let map = map_info();
    let mut mem = AgentMemory::new(&game_info());
    let o = extract(&f, &mut mem, &map, None);
    let w = World::new(&f, &mem, &map);
    let ship = f.units.iter().find(|u| u.unit_type == ty("Terran_Dropship") && u.id != ship2 && u.id != empty_ship).unwrap().id;
    let cargo = f.units.iter().find(|u| u.loaded_in == Some(ship)).unwrap().id;
    let (s1, s2, se, c1, c2) = (index_of(&o, ship), index_of(&o, ship2), index_of(&o, empty_ship), index_of(&o, cargo), index_of(&o, cargo2));
    assert!(cap_of(&o, s1, "is_transport_loaded") && cap_of(&o, s2, "is_transport_loaded") && !cap_of(&o, se, "is_transport_loaded"));
    assert!(cap_of(&o, c1, "loaded_unit") && cap_of(&o, c2, "loaded_unit"));
    let loaded: Vec<usize> = cap_rows(&o, "loaded_unit").collect();
    assert_eq!(loaded.len(), 2, "only own loaded units are targets: {loaded:?}");
    let unload = action_table().by_name("Unload_Unit").unwrap().id;
    assert!(o.masks.type_mask[unload as usize]);
    let out = execute(&w, &o, &host("Unload_Unit", vec![s1], Some(c1), None)).unwrap();
    assert_eq!(out.commands, vec![Command::Unit { units: vec![ship], order: uranium_proto::CmdOrder::Unload(cargo), queued: false }]);
    // Cargo of another transport: the game (BWAPI) refuses, so the executor does too.
    assert_eq!(execute(&w, &o, &host("Unload_Unit", vec![s1], Some(c2), None)), Err(Rejected::BadTarget));
    assert_eq!(execute(&w, &o, &host("Unload_Unit", vec![se], Some(c1), None)), Err(Rejected::BadSelection));
    // A replay label (labeler version 2: transport selected, opcode 0x29 target) is valid.
    let l = Label { frame: 0, action: unload, delay: 8, queued: false, selected: vec![ship], target_unit: Some(cargo), target_tile: None };
    let (h, why) = host_label_check(&l, &o);
    assert_eq!((h.selected.clone(), h.target_unit, why), (vec![s1], Some(c1), None));
    assert_eq!(unload_target_check(&h, &o, &f), None);
    // The BC label check (label walker) voids a target outside the selected transport.
    let other = Label { target_unit: Some(cargo2), ..l };
    let (h2, why2) = host_label_check(&other, &o);
    assert_eq!(why2, None, "the masks only say loaded");
    assert_eq!(unload_target_check(&h2, &o, &f), Some(NOT_IN_TRANSPORT_REASON));
    let mut walker = LabelWalker::new(&game_info(), 0, None, false).unwrap();
    assert_eq!(walker.step(&f, &other, None).2, Some(NOT_IN_TRANSPORT_REASON));
}

/// Action audit F12 (2026-10-19): a max_select = 1 label keeps the whole selection (labeler
/// version 2); the check filters eligibility first and then keeps the first eligible unit, so a
/// mine order whose first vulture has no mines is laid by the next one. A version-1 label (cut
/// to the first unit by the labeler) stays as before.
#[test]
fn max_select_one_picks_first_eligible_f12() {
    use uranium_env::features::label::{Label, host_label_check};
    let mut b = rich_units();
    let empty = b.add(SELF, "Terran_Vulture", (30, 30));
    let armed = b.add(SELF, "Terran_Vulture", (30, 31));
    b.get(armed).spider_mines = 2;
    let armed2 = b.add(SELF, "Terran_Vulture", (30, 32));
    b.get(armed2).spider_mines = 1;
    let f = frame(100, b.units.clone(), true);
    let map = map_info();
    let mut mem = AgentMemory::new(&game_info());
    let o = extract(&f, &mut mem, &map, None);
    let mine = action_table().by_name("SpiderMine_Pos").unwrap().id;
    let label = |selected: Vec<u32>| Label { frame: 0, action: mine, delay: 8, queued: false, selected, target_unit: None, target_tile: Some((31, 30)) };
    // Version 2: whole selection, first eligible in selection order (not entity order).
    let (h, why) = host_label_check(&label(vec![empty, armed2, armed]), &o);
    assert_eq!((h.selected, why), (vec![index_of(&o, armed2)], None));
    // Version 1: the labeler kept only the first unit; out of mines -> void, as before.
    assert_eq!(host_label_check(&label(vec![empty]), &o).1, Some("select_cap"));
    assert_eq!(host_label_check(&label(vec![armed]), &o).0.selected, vec![index_of(&o, armed)]);
    // A max_select = 12 action is unaffected (v1 and v2 store the same selection).
    let mv = action_table().by_name("Move_Pos").unwrap().id;
    let l = Label { action: mv, ..label(vec![empty, armed, armed2]) };
    let mut all = vec![index_of(&o, empty), index_of(&o, armed), index_of(&o, armed2)];
    all.sort_unstable();
    assert_eq!(host_label_check(&l, &o), (uranium_env::features::label::HostLabel { action: mv, delay_bucket: uranium_env::features::label::bucket_index(8), queued: false, selected: all, target_unit: None, target_loc: Some(30 * MAP + 31) }, None));
}

/// 031-AC3 (F7, 2026-10-19): a command without a 034 action is a label of its own with reason
/// `unknown` (BC weight 0); its host form is a NoOp with the label's delay.
#[test]
fn unknown_label_has_weight_zero_031_ac3() {
    use uranium_env::features::label::{Label, UNKNOWN_ACTION, UNKNOWN_REASON, bucket_index, host_label_check};
    let (o, ..) = observe(true);
    let l = Label { frame: 0, action: UNKNOWN_ACTION, delay: 12, queued: true, selected: vec![1, 2], target_unit: Some(3), target_tile: Some((1, 1)) };
    let (h, why) = host_label_check(&l, &o);
    assert_eq!(why, Some(UNKNOWN_REASON));
    assert_eq!((h.action, h.delay_bucket, h.selected.len(), h.target_unit, h.target_loc), (0, bucket_index(12), 0, None, None));
}

/// 액션 감사 F1 (사용자 결정 2026-10-19): a unit still in production (incomplete non-building) or
/// loaded in a transport is no target of any class; a building under construction still is.
#[test]
fn units_in_production_or_loaded_are_no_targets_034_audit_f1() {
    let map = map_info();
    let mut mem = AgentMemory::new(&game_info());
    let mut b = Builder::new();
    b.add(SELF, "Terran_Command_Center", (10, 10));
    b.add(SELF, "Terran_Medic", (20, 20));
    b.add(SELF, "Terran_Science_Vessel", (21, 20));
    b.add(SELF, "Terran_SCV", (22, 22));
    // An SCV being trained in the CC and a marine in production (hurt: hp rises while it trains).
    let scv = b.add(SELF, "Terran_SCV", (12, 12));
    b.get(scv).flags = 0;
    let marine = b.add(SELF, "Terran_Marine", (14, 14));
    b.get(marine).flags = 0;
    b.get(marine).hp = 10;
    b.get(marine).status_timers.irradiate = 5;
    let ship = b.add(SELF, "Terran_Dropship", (30, 30));
    let cargo = b.add(SELF, "Terran_Marine", (30, 30));
    b.get(cargo).loaded_in = Some(ship);
    b.get(cargo).flags |= uranium_proto::flags::LOADED;
    b.get(cargo).hp = 10;
    let depot = b.add(SELF, "Terran_Supply_Depot", (40, 40));
    b.get(depot).flags = 0;
    b.get(depot).hp = 10;
    let f = frame(10, b.units.clone(), true);
    let o = extract(&f, &mut mem, &map, None);
    let targets = ["any_unit", "own_unit", "healable", "restorable", "irradiatable", "matrixable", "repairable", "target_resource", "loadable_transport"];
    for id in [scv, marine, cargo] {
        let i = index_of(&o, id);
        for c in targets {
            assert!(!cap_of(&o, i, c), "unit {id} is a {c} target");
        }
    }
    let d = index_of(&o, depot);
    assert!(cap_of(&o, d, "any_unit") && cap_of(&o, d, "own_unit"), "a building under construction stays a target");
    // Move_Unit has a target (the completed SCV, the medic, ...) but never the trained ones.
    let w = World::new(&f, &mem, &map);
    let medic = index_of(&o, f.units.iter().find(|u| u.unit_type == ty("Terran_Medic")).unwrap().id);
    for t in [scv, marine, cargo] {
        assert!(execute(&w, &o, &host("Move_Unit", vec![medic], Some(index_of(&o, t)), None)).is_err());
    }
}

/// 액션 감사 F5·F6·F11·F15·F16 (사용자 결정 2026-10-19): Stop takes sieged tanks, turrets, lifted
/// buildings and constructing SCVs (BWAPI `canStop`), not grounded buildings; UnloadAll_Pos takes
/// a loaded dropship, not a loaded bunker; a bunker is no attacker; a constructing SCV or a tank in
/// a siege transition takes no other command; Build_* close when no SCV is free to build.
#[test]
fn stop_unload_attack_and_busy_selection_classes_034_audit() {
    use uranium_env::types::order_id;
    let map = map_info();
    let mut mem = AgentMemory::new(&game_info());
    let mut b = Builder::new();
    b.add(SELF, "Terran_Command_Center", (10, 10));
    let sieged = b.add(SELF, "Terran_Siege_Tank_Siege_Mode", (20, 20));
    let turret = b.add(SELF, "Terran_Missile_Turret", (24, 24));
    let bunker = b.add(SELF, "Terran_Bunker", (30, 30));
    let in_bunker = b.add(SELF, "Terran_Marine", (30, 30));
    b.get(in_bunker).loaded_in = Some(bunker);
    b.get(in_bunker).flags |= uranium_proto::flags::LOADED;
    let ship = b.add(SELF, "Terran_Dropship", (40, 40));
    let in_ship = b.add(SELF, "Terran_Marine", (40, 40));
    b.get(in_ship).loaded_in = Some(ship);
    b.get(in_ship).flags |= uranium_proto::flags::LOADED;
    let builder = b.add(SELF, "Terran_SCV", (50, 50));
    b.get(builder).order = order_id("ConstructingBuilding").unwrap();
    b.get(builder).flags |= uranium_proto::flags::CONSTRUCTING;
    let sieging = b.add(SELF, "Terran_Siege_Tank_Tank_Mode", (55, 55));
    b.get(sieging).order = order_id("Sieging").unwrap();
    let f = frame(10, b.units.clone(), true);
    let o = extract(&f, &mut mem, &map, None);
    let i = |id| index_of(&o, id);
    for id in [sieged, turret, builder, sieging] {
        assert!(cap_of(&o, i(id), "can_stop"), "{id} can stop");
    }
    assert!(!cap_of(&o, i(bunker), "can_stop"), "a grounded bunker cannot stop (BWAPI)");
    assert!(!cap_of(&o, i(sieged), "can_move") && !cap_of(&o, i(turret), "can_move"));
    assert!(cap_of(&o, i(bunker), "is_transport_loaded") && !cap_of(&o, i(bunker), "can_unload_pos"));
    assert!(cap_of(&o, i(ship), "is_transport_loaded") && cap_of(&o, i(ship), "can_unload_pos"));
    assert!(!cap_of(&o, i(bunker), "can_attack"), "a bunker is no attacker");
    for c in ["is_worker", "can_move", "can_be_loaded", "can_right_click_unit"] {
        assert!(!cap_of(&o, i(builder), c), "constructing SCV has {c}");
    }
    for c in ["can_siege", "can_move", "can_attack"] {
        assert!(!cap_of(&o, i(sieging), c), "sieging tank has {c}");
    }
    let open = |name: &str| o.masks.type_mask[action_table().by_name(name).unwrap().id as usize];
    assert!(open("Stop") && open("UnloadAll_Pos"));
    // The only SCV is constructing: no Build_* (034-R6, pick_worker has no candidate).
    assert!(!open("Build_SupplyDepot") && !open("Build_Barracks"));
    let w = World::new(&f, &mem, &map);
    assert!(execute(&w, &o, &host("Stop", vec![i(sieged)], None, None)).is_ok());
    // Stop alone in BC labels: a sieged-tank Stop is a valid label now.
    use uranium_env::features::label::{Label, host_label_check};
    let l = Label { frame: 10, action: action_table().by_name("Stop").unwrap().id, delay: 8, queued: false, selected: vec![sieged], target_unit: None, target_tile: None };
    assert_eq!(host_label_check(&l, &o).1, None);
}

/// 034-R10 (review 2026-10-19): eggs, lurker eggs, cocoons and merging archons are incomplete but
/// stay targets (BWAPI `canTargetUnit` accepts morphing units and archons, so does the game).
#[test]
fn eggs_cocoons_and_merging_archons_are_targets_034_r10() {
    let map = map_info();
    let mut mem = AgentMemory::new(&game_info());
    let mut b = Builder::new();
    b.add(SELF, "Terran_Command_Center", (10, 10));
    let ids: Vec<u32> = ["Zerg_Egg", "Zerg_Lurker_Egg", "Zerg_Cocoon", "Protoss_Archon", "Protoss_Dark_Archon"]
        .iter()
        .enumerate()
        .map(|(k, n)| {
            let id = b.add(ENEMY, n, (30 + 3 * k as i32, 30));
            b.get(id).flags = uranium_proto::flags::DETECTED;
            id
        })
        .collect();
    let incomplete_marine = b.add(ENEMY, "Zerg_Zergling", (50, 50));
    b.get(incomplete_marine).flags = 0;
    let o = extract(&frame(10, b.units.clone(), true), &mut mem, &map, None);
    for id in ids {
        let i = index_of(&o, id);
        for c in ["any_unit", "attackable_enemy", "irradiatable"] {
            assert!(cap_of(&o, i, c), "{id} not a {c} target");
        }
    }
    assert!(!cap_of(&o, index_of(&o, incomplete_marine), "any_unit"), "other incomplete units stay excluded");
}

/// U-4 (user decision 2026-10-19, action audit F16): a tank in a siege transition takes no
/// command live (BWAPI Unit_Busy) but its replay labels are valid (the game queues the command).
#[test]
fn siege_transition_live_blocked_labels_valid_u4() {
    use uranium_env::features::label::{Label, host_label_check};
    use uranium_env::types::order_id;
    let map = map_info();
    let mut b = Builder::new();
    b.add(SELF, "Terran_Command_Center", (10, 10));
    let tank = b.add(SELF, "Terran_Siege_Tank_Tank_Mode", (30, 30));
    b.get(tank).order = order_id("Sieging").unwrap();
    let f = frame(10, b.units.clone(), true);
    let siege = action_table().by_name("SiegeMode").unwrap().id;
    let label = Label { frame: 10, action: siege, delay: 8, queued: false, selected: vec![tank], target_unit: None, target_tile: None };
    let mut live = AgentMemory::new(&game_info());
    let o = extract(&f, &mut live, &map, None);
    assert!(!cap_of(&o, index_of(&o, tank), "can_siege") && !o.masks.type_mask[siege as usize]);
    let mut replay = AgentMemory::new(&game_info());
    replay.replay_labels = true;
    let o = extract(&f, &mut replay, &map, None);
    assert!(cap_of(&o, index_of(&o, tank), "can_siege"));
    assert_eq!(host_label_check(&label, &o).1, None);
}
