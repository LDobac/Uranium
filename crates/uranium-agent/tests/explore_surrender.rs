use uranium_env::testkit as common;

use std::sync::Arc;
use std::time::Duration;

use common::*;
use uranium_agent::GameBranch;
use uranium_agent::broker::{BatchOut, Broker};
use uranium_agent::infer::{HEADS, HostState, InferOptions, InferenceHandle, SampleMode, SurrenderMode};
use uranium_agent::model_policy::ModelPolicy;
use uranium_match::explore::{ExploreConfig, ExploreGame, ExploreMiss, MissReason, schedule};
use uranium_proto::RawFrame;

fn mock() -> InferenceHandle {
    let opts = InferOptions { max_batch: 1, max_wait: Duration::from_millis(50), capacity: 4, ..InferOptions::default() };
    let (mut b, h) = Broker::new("mock@000000000000".into(), SampleMode::Stochastic, opts).unwrap();
    std::thread::spawn(move || {
        while !b.closed() {
            let Some(w) = b.next(Duration::from_millis(50)).unwrap() else { continue };
            b.reply_restores(w.restores.iter().map(|_| None).collect()).unwrap();
            b.reply_states(w.states.iter().map(|_| HostState { parts: vec![(vec![1, 1], vec![0.0])] }).collect()).unwrap();
            if let Some(batch) = w.batch {
                let (r, a) = (batch.slots.len(), batch.obs.a);
                let zeros = vec![0i64; r];
                let sel = vec![batch.obs.n as i64; r * 12];
                let lp = vec![-0.5f32; r];
                let heads = vec![0f32; r * HEADS];
                let value = vec![0f32; r * 4];
                let type_logp: Vec<f32> = batch
                    .logp_rows
                    .iter()
                    .flat_map(|&row| {
                        let m = &batch.obs.type_mask[row * a..(row + 1) * a];
                        let z = (m.iter().filter(|x| **x).count() as f32).ln();
                        m.iter().map(move |x| if *x { -z } else { f32::NEG_INFINITY }).collect::<Vec<_>>()
                    })
                    .collect();
                let out = BatchOut { action: &zeros, delay: &zeros, queued: &zeros, selected: &sel, target_unit: &zeros, target_loc: &zeros, log_prob: &lp, head_log_prob: &heads, value: &value, win_open: None, type_logp: Some(&type_logp) };
                b.reply(out).unwrap();
            }
        }
    });
    h
}

fn frames(steps: usize) -> Vec<RawFrame> {
    (0..steps).map(|t| frame(24 * t as u32 + 1, rich_units().units, t % 2 == 0)).collect()
}

#[test]
fn a_surrender_on_an_explore_decision_is_missed_not_logged_054_r4() {
    let cfg = ExploreConfig { window_s: [0, 30], gap_s: 5, ..ExploreConfig::default() };
    let seed = 4;
    let taus = schedule(&cfg, seed).unwrap();
    let fs = frames(45);
    let at = fs.iter().position(|f| f.frame >= taus[0]).unwrap();
    unsafe { std::env::set_var("URANIUM_TEST_SURRENDER_FRAME", fs[at].frame.to_string()) };
    let game = Arc::new(GameBranch { plan_game: 1, explore: Some(ExploreGame { cfg, seed }), ..Default::default() });
    let h = mock().with_surrender(SurrenderMode::On, "gg");
    let mut p = ModelPolicy::new(&h, &game_info()).unwrap().with_branch(game, 0).unwrap();
    for (i, f) in fs.iter().enumerate() {
        let act = p.on_frame(f).unwrap();
        let x = p.act_log_extra().unwrap();
        assert!(x.explore.is_none(), "observation {i}: no explore line before or at the surrender");
        if i == at {
            assert!(act.surrender.is_some(), "the explore decision surrenders");
            break;
        }
    }
    p.note_end(Some(false), fs[at].frame + 1);
    p.on_end(Some(uranium_env::reward::Outcome::Loss), None);
    let s = p.branch_record().unwrap().explore.unwrap();
    assert!(s.entries.is_empty(), "no entry, no propensity for a decision not sent");
    let want: Vec<ExploreMiss> = std::iter::once(ExploreMiss { point: 1, tau: taus[0], reason: MissReason::Surrendered })
        .chain((1..3).map(|i| ExploreMiss { point: i + 1, tau: taus[i as usize], reason: MissReason::GameEnd }))
        .collect();
    assert_eq!(s.missed, want);
    assert_eq!(p.explore_end().unwrap().missed, want);
}
