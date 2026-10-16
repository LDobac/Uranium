//! 033-AC1: reward signs and magnitudes on synthetic scenarios.

mod common;

use common::*;
use uranium_env::actions::ActionTable;
use uranium_env::features::Strategy;
use uranium_env::features::extract::dims;
use uranium_env::reward::*;
use uranium_proto::GameEvent;

fn id(name: &str) -> u16 {
    ActionTable::load().unwrap().by_name(name).unwrap().id
}

fn z() -> Strategy {
    let mut units = vec![0.0; dims().z_units()];
    let marine = dims().terran_types.iter().position(|&t| t == ty("Terran_Marine")).unwrap();
    units[marine] = 1.0;
    Strategy { build: vec![id("Build_SupplyDepot"), id("Build_Barracks"), id("Train_Marine")], units }
}

#[test]
fn levenshtein_basics() {
    assert_eq!(levenshtein(&[1, 2, 3], &[1, 2, 3]), 0);
    assert_eq!(levenshtein(&[1, 2, 3], &[]), 3);
    assert_eq!(levenshtein(&[1, 2, 3], &[1, 3]), 1);
    assert_eq!(levenshtein(&[1, 2], &[2, 1]), 2);
}

#[test]
fn following_z_build_is_rewarded_033_ac1() {
    let mut r = RewardTracker::new(Some(z()), SELF);
    r.step(&frame(1, vec![], false));
    r.record_build(id("Build_SupplyDepot"));
    let p = r.step(&frame(2, vec![], false));
    assert_eq!(p.build, 1.0, "matching step lowers the distance by one");
    // A deviation does not lower the distance (it can be read as a substitution): reward 0.
    r.record_build(id("Build_Factory"));
    let p = r.step(&frame(3, vec![], false));
    assert_eq!(p.build, 0.0, "{p:?}");
    // Actions beyond the plan raise the distance: negative.
    r.record_build(id("Train_Marine"));
    r.step(&frame(4, vec![], false));
    r.record_build(id("Build_Starport"));
    let p = r.step(&frame(5, vec![], false));
    assert_eq!(p.build, -1.0, "{p:?}");
}

#[test]
fn matching_unit_composition_is_rewarded_033_ac1() {
    let mut r = RewardTracker::new(Some(z()), SELF);
    let mut b = Builder::new();
    let marine = b.add(SELF, "Terran_Marine", (5, 5));
    let mut f = frame(10, b.units.clone(), false);
    f.events.push(GameEvent::UnitCreated(marine));
    let p = r.step(&f);
    assert!((p.units - 1.0).abs() < 1e-6, "composition moved from L1 1.0 to 0.0: {p:?}");
    let scv = b.add(SELF, "Terran_SCV", (6, 6));
    let mut f = frame(20, b.units.clone(), false);
    f.events.push(GameEvent::UnitCreated(scv));
    let p = r.step(&f);
    assert!(p.units < 0.0, "an off-plan unit moves away from z: {p:?}");
}

#[test]
fn pseudo_rewards_need_z_033_r2() {
    let mut r = RewardTracker::new(None, SELF);
    r.record_build(id("Build_SupplyDepot"));
    let mut b = Builder::new();
    let m = b.add(SELF, "Terran_Marine", (5, 5));
    let mut f = frame(10, b.units, false);
    f.events.push(GameEvent::UnitCreated(m));
    let p = r.step(&f);
    assert_eq!((p.build, p.units), (0.0, 0.0));
}

#[test]
fn only_observed_destruction_counts_033_r5() {
    let mut r = RewardTracker::new(None, SELF);
    let mut b = Builder::new();
    let ling = b.add(ENEMY, "Zerg_Zergling", (40, 40));
    let marine = b.add(SELF, "Terran_Marine", (5, 5));
    r.step(&frame(10, b.units.clone(), false));
    // Observed kill of a zergling (50 minerals) -> +0.05.
    let mut f = frame(20, vec![b.units[1].clone()], false);
    f.events.push(GameEvent::UnitDestroyed(ling));
    let p = r.step(&f);
    assert!((p.value - 0.05).abs() < 1e-6, "{p:?}");
    // A destruction of a unit never seen does not count.
    let mut f = frame(30, vec![b.units[1].clone()], false);
    f.events.push(GameEvent::UnitDestroyed(9999));
    assert_eq!(r.step(&f).value, 0.0);
    // Losing the marine (50) -> -0.05.
    let mut f = frame(40, vec![], false);
    f.events.push(GameEvent::UnitDestroyed(marine));
    assert!((r.step(&f).value + 0.05).abs() < 1e-6);
}

#[test]
fn terminal_and_weights_033() {
    assert_eq!(RewardTracker::end(Outcome::Win).win, 1.0);
    assert_eq!(RewardTracker::end(Outcome::Loss).win, -1.0);
    assert_eq!(RewardTracker::end(Outcome::Draw).win, 0.0);
    let w = RewardWeights::load().unwrap();
    assert_eq!(w, RewardWeights { win: 1.0, build: 0.3, units: 0.3, value: 0.05 });
    let p = RewardParts { win: 1.0, build: 1.0, units: -1.0, value: 2.0 };
    assert!((p.weighted(&w) - (1.0 + 0.3 - 0.3 + 0.1)).abs() < 1e-6);
}
