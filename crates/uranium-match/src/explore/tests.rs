use super::*;
use uranium_env::features::extract::action_table;

fn id(name: &str) -> u16 {
    action_table().by_name(name).unwrap().id
}

fn game(seed: u32) -> ExploreGame {
    ExploreGame { cfg: ExploreConfig::default(), seed }
}

fn mask_with(open: &[u16]) -> Vec<bool> {
    let n = action_table().actions.len();
    (0..n).map(|a| a == 0 || open.contains(&(a as u16))).collect()
}

#[test]
fn op_actions_are_the_dump_groups_054() {
    let t = action_table();
    let ops = op_actions(t);
    assert!(ops.windows(2).all(|w| w[0] < w[1]));
    for a in &t.actions {
        let want = OP_GROUPS.contains(&a.group.as_str()) || OP_EXTRA.contains(&a.name.as_str());
        assert_eq!(ops.contains(&a.id), want, "{}", a.name);
    }
    assert!(ops.contains(&id("Build_Bunker")) && ops.contains(&id("Lift")) && !ops.contains(&0));
}

#[test]
fn config_defaults_and_checks_020_r22() {
    let c: ExploreConfig = toml::from_str("").unwrap();
    assert_eq!(c, ExploreConfig { points: 3, window_s: [120, 540], gap_s: 60, k: 4, epsilon: 0.5, min_open: 2 });
    c.validate().unwrap();
    assert!(toml::from_str::<ExploreConfig>("bogus = 1").is_err());
    let bad = |c: ExploreConfig, want: &str| {
        let e = format!("{:#}", c.validate().unwrap_err());
        assert!(e.contains(want), "{want}: {e}");
    };
    bad(ExploreConfig { points: 0, ..c.clone() }, "points");
    bad(ExploreConfig { window_s: [540, 540], ..c.clone() }, "empty window");
    bad(ExploreConfig { points: 8, ..c.clone() }, "do not fit");
    bad(ExploreConfig { k: 0, ..c.clone() }, "k must");
    bad(ExploreConfig { epsilon: 1.5, ..c.clone() }, "epsilon");
    bad(ExploreConfig { epsilon: f64::NAN, ..c.clone() }, "epsilon");
    bad(ExploreConfig { min_open: 1, ..c }, "min_open");
}

#[test]
fn schedule_is_seeded_inside_the_window_and_apart_054_ac1() {
    let c = ExploreConfig::default();
    let mut differ = 0;
    for seed in 0..500u32 {
        let t = schedule(&c, seed).unwrap();
        assert_eq!(t, schedule(&c, seed).unwrap(), "same seed, same times");
        assert_eq!(t.len(), 3);
        assert!(t.iter().all(|&f| (2_880..12_960).contains(&f)), "seed {seed}: {t:?}");
        assert!(t.windows(2).all(|w| w[1] - w[0] >= 1_440), "seed {seed}: {t:?}");
        differ += (t != schedule(&c, seed + 1).unwrap()) as u32;
    }
    assert!(differ > 490, "seeds give different times");
    let all: Vec<u32> = (0..2_000u32).flat_map(|s| schedule(&c, s).unwrap()).collect();
    assert!(all.iter().any(|&f| f < 2_880 + 200) && all.iter().any(|&f| f >= 12_960 - 200), "both window ends are reached");
    let tight = ExploreConfig { window_s: [120, 241], gap_s: 60, ..c };
    for seed in 0..50 {
        let t = schedule(&tight, seed).unwrap();
        assert!(t[0] >= 2_880 && t[2] < 241 * 24 && t[1] - t[0] >= 1_440 && t[2] - t[1] >= 1_440, "{t:?}");
    }
}

#[test]
fn points_fire_at_the_first_decision_with_two_open_op_actions_054_ac1() {
    let g = game(7);
    let mut r = ExploreRun::new(&g, action_table()).unwrap();
    let t = r.taus().to_vec();
    let two = mask_with(&[id("Build_Bunker"), id("Train_Marine")]);
    let one = mask_with(&[id("Build_Bunker"), id("Move_Pos")]);
    assert_eq!(r.check(t[0] - 1, &two), None, "before τ1");
    assert_eq!(r.check(t[0], &one), None, "one open op action: wait");
    assert_eq!(r.check(t[0] + 5, &two), Some(0));
    assert_eq!(r.check(t[0] + 9, &two), None, "τ1 fired once");
    assert_eq!(r.check(t[2] + 3, &two), Some(1), "τ2 waits; one point per decision");
    assert_eq!(r.check(t[2] + 4, &two), Some(2), "τ3 at the next matching decision");
    r.finish(false, Some(14_400));
    assert!(r.end().missed.is_empty());
    assert_eq!(r.end().taus, t);
}

#[test]
fn points_without_a_decision_before_9m_leave_a_reason_054_ac1() {
    let g = game(11);
    let two = mask_with(&[id("Build_Bunker"), id("Train_Marine")]);
    let mut r = ExploreRun::new(&g, action_table()).unwrap();
    let t = r.taus().to_vec();
    r.check(t[0], &two);
    assert_eq!(r.check(12_960, &two), None, "the window is [2:00, 9:00)");
    r.finish(false, Some(14_400));
    let m = r.end().missed;
    assert_eq!(m, vec![ExploreMiss { point: 2, tau: t[1], reason: MissReason::WindowEnd }, ExploreMiss { point: 3, tau: t[2], reason: MissReason::WindowEnd }]);
    let mut r = ExploreRun::new(&g, action_table()).unwrap();
    r.check(t[0] - 1, &two);
    r.finish(false, Some(t[0] + 10));
    assert!(r.end().missed.iter().all(|x| x.reason == MissReason::GameEnd) && r.end().missed.len() == 3);
    let mut r = ExploreRun::new(&g, action_table()).unwrap();
    r.finish(true, None);
    assert!(r.summary().missed.iter().all(|x| x.reason == MissReason::Aborted));
}

#[test]
fn candidates_are_the_top_k_open_op_actions_without_a0_054_ac1() {
    let ops = op_actions(action_table());
    let open = [id("Build_Bunker"), id("Train_Marine"), id("Build_Barracks"), id("Build_SupplyDepot"), id("Train_SCV"), id("Move_Pos")];
    let mask = mask_with(&open);
    let mut logp = vec![f32::NEG_INFINITY; mask.len()];
    for (i, &a) in open.iter().enumerate() {
        logp[a as usize] = -(i as f32 + 1.0);
    }
    logp[id("Move_Pos") as usize] = -0.01;
    logp[0] = -0.5;
    let c = candidates(&logp, &mask, &ops, 0, 4);
    assert_eq!(c.iter().map(|x| x.0).collect::<Vec<_>>(), open[..4].to_vec(), "Move_Pos is no op action");
    assert_eq!(c[0].1, (-1f32).exp());
    let c = candidates(&logp, &mask, &ops, id("Train_Marine"), 4);
    assert_eq!(c.iter().map(|x| x.0).collect::<Vec<_>>(), vec![open[0], open[2], open[3], open[4]], "a0 left out");
    let mut tie = logp.clone();
    tie[open[1] as usize] = -1.0;
    let c = candidates(&tie, &mask, &ops, 0, 2);
    assert_eq!(c.iter().map(|x| x.0).collect::<Vec<_>>(), vec![open[0].min(open[1]), open[0].max(open[1])], "ties by id");
    let top = policy_top(&logp, &mask);
    assert_eq!(top.len(), open.len() + 1);
    assert_eq!((top[0].0, top[1].0), (id("Move_Pos"), 0));
    assert!(top.windows(2).all(|w| w[0].1 >= w[1].1));
    let f = force_mask(&mask, open[2]);
    assert_eq!(f.iter().filter(|b| **b).count(), 1);
    assert!(f[open[2] as usize] && !f[0]);
}

#[test]
fn coin_choice_and_propensity_are_seeded_054_ac1() {
    let open = [id("Build_Bunker"), id("Train_Marine"), id("Build_Barracks"), id("Build_SupplyDepot"), id("Train_SCV")];
    let mask = mask_with(&open);
    let mut logp = vec![f32::NEG_INFINITY; mask.len()];
    for (i, &a) in open.iter().enumerate() {
        logp[a as usize] = -(i as f32 + 1.0);
    }
    let (mut forced, mut picked) = (0, std::collections::BTreeMap::new());
    for seed in 0..400u32 {
        for p in 0..3 {
            let run = || {
                let mut r = ExploreRun::new(&game(seed), action_table()).unwrap();
                let t = r.taus()[p];
                r.decide(p, t + 1, open[1], &logp, &mask)
            };
            let e = run();
            assert_eq!(e, run(), "seed {seed} point {p}");
            assert_eq!((e.point, e.k, e.epsilon, e.a0), (p as u32 + 1, 4, 0.5, open[1]));
            assert_eq!(e.cands.iter().map(|x| x.0).collect::<Vec<_>>(), vec![open[0], open[2], open[3], open[4]]);
            assert_eq!(draw(&ExploreConfig::default(), seed, p as u32 + 1).0, e.coin);
            match e.coin {
                Coin::Forced => {
                    forced += 1;
                    let c = e.c.unwrap();
                    assert!(e.cands.iter().any(|x| x.0 == c) && c != e.a0);
                    assert_eq!(e.propensity, 0.5 / 4.0);
                    *picked.entry(c).or_insert(0) += 1;
                }
                Coin::Control => assert_eq!((e.c, e.propensity), (None, 0.5)),
            }
        }
    }
    assert!((540..660).contains(&forced), "ε = 0.5 of 1,200 decisions: {forced}");
    assert_eq!(picked.len(), 4);
    assert!(picked.values().all(|&n| (100..200).contains(&n)), "uniform over the candidates: {picked:?}");
    let few = mask_with(&open[..2]);
    let mut r = ExploreRun::new(&game(3), action_table()).unwrap();
    let e = r.decide(0, 3_000, 0, &logp, &few);
    assert_eq!(e.cands.len(), 2);
    if e.coin == Coin::Forced {
        assert_eq!(e.propensity, 0.25, "ε / |cands|");
    }
    let never = ExploreGame { cfg: ExploreConfig { epsilon: 0.0, ..ExploreConfig::default() }, seed: 1 };
    let always = ExploreGame { cfg: ExploreConfig { epsilon: 1.0, ..ExploreConfig::default() }, seed: 1 };
    assert_eq!(ExploreRun::new(&never, action_table()).unwrap().decide(0, 3_000, 0, &logp, &mask).coin, Coin::Control);
    assert_eq!(ExploreRun::new(&always, action_table()).unwrap().decide(0, 3_000, 0, &logp, &mask).coin, Coin::Forced);
}

#[test]
fn explore_records_round_trip_054_r4() {
    let e = ExploreEntry { point: 1, tau: 4123, frame: 4130, epsilon: 0.5, k: 4, cands: vec![(37, 0.21), (20, 0.08), (41, 0.05), (44, 0.02)], a0: 0, coin: Coin::Forced, c: Some(20), rejected: Some(false), propensity: 0.125 };
    let text = serde_json::to_string(&e).unwrap();
    assert_eq!(text, r#"{"point":1,"tau":4123,"frame":4130,"epsilon":0.5,"k":4,"cands":[[37,0.21],[20,0.08],[41,0.05],[44,0.02]],"a0":0,"coin":"forced","c":20,"rejected":false,"propensity":0.125}"#);
    assert_eq!(serde_json::from_str::<ExploreEntry>(&text).unwrap(), e);
    let m = ExploreMiss { point: 3, tau: 12_000, reason: MissReason::WindowEnd };
    assert_eq!(serde_json::to_string(&m).unwrap(), r#"{"point":3,"tau":12000,"reason":"window_end"}"#);
}
