use std::collections::{BTreeSet, HashMap};

use uranium_scripted::MapHints;
use uranium_scripted::react::testkit::main_bunker_probe;

#[test]
fn commands_are_034_actions_022_ac1() {
    let lab = uranium_replay::label::Labeler::new().unwrap();
    let table = uranium_env::actions::ActionTable::load().unwrap();
    let mut names = BTreeSet::new();
    for (f, act) in uranium_scripted::tp::testkit::scenes() {
        let units: HashMap<u32, uranium_replay::label::Own> = uranium_replay::relabel::own_units(&f, 0);
        for c in &act.cmds {
            let cmds = uranium_replay::relabel::command_cmds(c, &f, 0, 0);
            assert!(!cmds.is_empty() && cmds.len().is_multiple_of(2), "{c:?}");
            let (labels, _) = lab.label(&cmds, f.frame + 32, |_| &units);
            let acts: Vec<_> = labels.iter().filter(|l| l.action != 0).collect();
            assert_eq!(acts.len(), cmds.len() / 2, "{c:?} {labels:?}");
            for l in acts {
                assert_ne!(l.action, uranium_env::features::label::UNKNOWN_ACTION, "{c:?}");
                assert!(l.selected.len() <= 12);
                names.insert(table.actions[l.action as usize].name.clone());
            }
        }
    }
    for n in [
        "Build_SupplyDepot", "Build_Barracks", "Build_Refinery", "Build_Factory", "Build_Bunker", "BuildAddon_MachineShop", "Research_TankSiege", "Train_SCV",
        "Train_Marine", "Train_SiegeTank", "SiegeMode", "TankMode", "Move_Pos", "Smart_Unit", "Hold", "Attack_Pos", "Attack_Unit", "Repair_Unit", "Gather_Unit",
    ] {
        assert!(names.contains(n), "{n} missing: {names:?}");
    }
}

#[test]
#[ignore = "needs URANIUM_MAPS_DIR and URANIUM_TEST_SHARDS with target-map games"]
fn target_map_bunker_on_the_ground_path_021_r8() {
    let maps = std::path::PathBuf::from(std::env::var("URANIUM_MAPS_DIR").expect("URANIUM_MAPS_DIR"));
    let shards = std::path::PathBuf::from(std::env::var("URANIUM_TEST_SHARDS").expect("URANIUM_TEST_SHARDS"));
    let families = uranium_env::maps::MapFamilies::load().unwrap();
    let hashes = [
        ("fighting_spirit", "226b8267504ef85938621f604e99e74d195a8c87b5beb8ec59542d8d5d218ace"),
        ("fighting_spirit", "175b118cc1ab70c95e9bfc90cc5640ed300b7370443e9394aeb0d1dce6dbb0bc"),
        ("circuit_breaker", "8109bae22de8e79ca9ddc9ca4bad343dbf5ac676747d7805f043cd8080bfdf56"),
    ];
    let mut found = BTreeSet::new();
    let mut on_path_starts = 0u32;
    let mut files: Vec<_> = std::fs::read_dir(&shards).unwrap().filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.to_string_lossy().ends_with(".bin.zst")).collect();
    files.sort();
    for path in files {
        for g in uranium_replay::shard::read(&path).unwrap().games {
            let family = families.classify(&g.game_info.map_name, "");
            if !hashes.iter().any(|(f, _)| *f == family) || !found.insert(family.clone()) {
                continue;
            }
            let pre = uranium_env::maps::live_precompute(&maps, &g.game_info)
                .0
                .or_else(|| hashes.iter().filter(|(f, _)| *f == family).find_map(|(_, h)| uranium_env::maps::load_precompute(&maps, h)))
                .expect("precompute");
            let hints = MapHints { base_locs: pre.base_locs.clone(), chokes: pre.chokes.clone() };
            let resources: Vec<_> = pre.resources.iter().map(|r| (r.unit_type, r.pos)).collect();
            let starts = g.game_info.static_map.start_locations.clone();
            for s in starts.iter().copied() {
                let p = main_bunker_probe(&g.game_info.static_map, &hints, s, &resources);
                let (path, site, nat) = (p.path, p.site, p.natural);
                assert!(path.last().is_some_and(|t| *t == (nat.0 / 32, nat.1 / 32)), "{family} {s:?}: no path to the natural");
                let (d, on_path) = p.bunker.unwrap_or_else(|| panic!("{family} {s:?}: no main site"));
                eprintln!("{family} start {s:?}: bunker {site:?} at {d:.1} tiles, on the path {on_path}, path {} tiles", path.len());
                assert!(if on_path { (5.0..=7.0).contains(&d) } else { (4.0..=6.0).contains(&d) }, "{family} {s:?}: site {site:?} {d:.1} tiles");
                on_path_starts += u32::from(on_path);
            }
        }
    }
    assert_eq!(found.len(), 2, "both target maps in the shards: {found:?}");
    assert!(on_path_starts >= 5, "{on_path_starts} starts on the path");
}
