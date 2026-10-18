//! 032 acceptance tests on Flex (032-R1). Every test runs for both cores (032-AC6).

mod common;

use burn::module::Module;
use burn::tensor::{Device, Tensor};
use common::*;
use uranium_model::checkpoint::{self, CheckpointMeta};
use uranium_model::config::{CoreKind, UraniumNetConfig};
use uranium_model::{CoreInit, GtrxlCore, LstmCore, UraniumNet};

fn dev() -> Device {
    Device::flex()
}

fn net<C: CoreInit>(seed: u64) -> (UraniumNet<C>, UraniumNetConfig) {
    let d = dev();
    d.seed(seed);
    let cfg = tiny_config(C::KIND);
    (UraniumNet::<C>::new(&cfg, &columns(), &d), cfg)
}

/// 032-AC4 / 031-AC2: sampled actions always satisfy the masks.
fn masks_hold<C: CoreInit>(rows: usize) {
    let (net, cfg) = net::<C>(1);
    let cols = columns();
    let (b, n) = (500, 12);
    let mut checked = 0;
    let mut seed = 100;
    while checked < rows {
        let (obs, hm) = random_obs(&cfg, &cols, b, n, seed, &dev());
        let out = net.act(&obs, net.init_state(b, &dev()), 1.0);
        checked += check_actions(&cols, &hm, &host_actions(&out.action));
        seed += 1;
    }
}

/// 032-AC5: privileged inputs change the value but never the policy.
fn privileged_only_reaches_value<C: CoreInit>() {
    let (net, cfg) = net::<C>(2);
    let cols = columns();
    let b = 6;
    let (obs, _) = random_obs(&cfg, &cols, b, 9, 7, &dev());
    let labels = net.act(&obs, net.init_state(b, &dev()), 1.0).action;
    let p1 = random_priv(&cfg, b, 1, &dev());
    let p2 = random_priv(&cfg, b, 2, &dev());
    let o1 = net.forward_seq(&obs, &labels, Some(&p1), net.init_state(b, &dev()), 1);
    let o2 = net.forward_seq(&obs, &labels, Some(&p2), net.init_state(b, &dev()), 1);
    assert_eq!(floats(&o1.heads.logp.sum()), floats(&o2.heads.logp.sum()), "policy depends on privileged input");
    assert_ne!(floats(&o1.value), floats(&o2.value), "value ignores privileged input");
}

/// Teacher forcing reproduces the sampling path: log p(sampled action) from `forward_seq`
/// equals the log-prob `act` reported (032-R3, R10).
fn teacher_matches_sampling<C: CoreInit>() {
    let (net, cfg) = net::<C>(3);
    let cols = columns();
    let b = 32;
    let (obs, _) = random_obs(&cfg, &cols, b, 10, 11, &dev());
    let out = net.act(&obs, net.init_state(b, &dev()), 1.0);
    let seq = net.forward_seq(&obs, &out.action, None, net.init_state(b, &dev()), 1);
    let (a, t) = (floats(&out.log_prob), floats(&seq.heads.logp.sum()));
    for (i, (x, y)) in a.iter().zip(&t).enumerate() {
        assert!((x - y).abs() < 1e-3, "row {i}: act log-prob {x} != teacher log-prob {y}");
        assert!(*x > -1e3, "row {i}: sampled a masked choice (log-prob {x})");
    }
    assert_eq!(ints1(&out.action.action), ints1(&seq.heads.chosen.action));
    assert_eq!(ints2(&out.action.selected), ints2(&seq.heads.chosen.selected));
}

/// Stepping the core T times equals one `forward_seq` over T (032-R5 state handling).
fn step_matches_sequence<C: CoreInit>() {
    let (net, _) = net::<C>(4);
    let core = net.core();
    let (b, t, d) = (3, 7, 32 + 32 + 32);
    let x = Tensor::<3>::from_data(burn::tensor::TensorData::new((0..b * t * d).map(|i| ((i * 37 % 101) as f32 / 50.0) - 1.0).collect::<Vec<_>>(), [b, t, d]), &dev());
    let (seq, _) = core.forward_seq(x.clone(), core.init_state(b, &dev()));
    let mut s = core.init_state(b, &dev());
    for i in 0..t {
        let (h, s2) = core.step(x.clone().narrow(1, i, 1).reshape([b, d]), s);
        s = s2;
        let want = floats(&seq.clone().narrow(1, i, 1).reshape([b, core.hidden()]));
        for (g, w) in floats(&h).iter().zip(&want) {
            assert!((g - w).abs() < 1e-4, "step {i}: {g} != {w}");
        }
    }
}

/// Gradients reach every parameter group, incl. the distance-bias table (032-R13).
fn backward_reaches_parameters<C: CoreInit>() {
    let ad = dev().autodiff();
    ad.seed(5);
    let cfg = tiny_config(C::KIND);
    let cols = columns();
    let net = UraniumNet::<C>::new(&cfg, &cols, &ad);
    let b = 4;
    let (obs, _) = random_obs(&cfg, &cols, b, 8, 13, &ad);
    let labels = net.act(&obs, net.init_state(b, &ad), 1.0).action;
    let priv_ = random_priv(&cfg, b, 3, &ad);
    let out = net.forward_seq(&obs, &labels, Some(&priv_), net.init_state(b, &ad), 1);
    let loss = out.heads.logp.sum().neg().mean() + out.value.powi_scalar(2).mean();
    let grads = loss.backward();
    let mut with_grad = 0;
    let mut total = 0;
    net.visit(&mut GradCounter { grads: &grads, with_grad: &mut with_grad, total: &mut total });
    assert!(total > 0);
    assert_eq!(with_grad, total, "{} of {total} parameters got no gradient", total - with_grad);
}

struct GradCounter<'a> {
    grads: &'a burn::tensor::Gradients,
    with_grad: &'a mut usize,
    total: &'a mut usize,
}

impl burn::module::ModuleVisitor for GradCounter<'_> {
    fn visit_float<const D: usize>(&mut self, param: &burn::module::Param<Tensor<D>>) {
        *self.total += 1;
        if param.val().grad(self.grads).is_some() {
            *self.with_grad += 1;
        }
    }
}

/// 032-R6: checkpoint round trip reproduces greedy actions and values.
fn checkpoint_round_trip<C: CoreInit>() {
    let (net, cfg) = net::<C>(6);
    let cols = columns();
    let dir = std::env::temp_dir().join(format!("uranium_model_ckpt_{:?}_{}", C::KIND, std::process::id()));
    let meta = CheckpointMeta::new(&cfg, 7, uranium_env::actions::ACTION_SCHEMA_VERSION).unwrap();
    checkpoint::save(&dir, &net, &meta).unwrap();
    let (loaded, meta2) = checkpoint::load::<C>(&dir, &cols, 7, uranium_env::actions::ACTION_SCHEMA_VERSION, &dev()).unwrap();
    assert_eq!(meta, meta2);
    assert!(checkpoint::load::<C>(&dir, &cols, 8, uranium_env::actions::ACTION_SCHEMA_VERSION, &dev()).is_err(), "feature version mismatch must fail");
    let b = 5;
    let (obs, _) = random_obs(&cfg, &cols, b, 6, 17, &dev());
    let a = net.act(&obs, net.init_state(b, &dev()), 0.0);
    let l = loaded.act(&obs, loaded.init_state(b, &dev()), 0.0);
    assert_eq!(ints1(&a.action.action), ints1(&l.action.action));
    assert_eq!(floats(&a.value), floats(&l.value));
    std::fs::remove_dir_all(&dir).ok();
}

/// 032-R14: inference-server slots. Sessions step in changing subsets through
/// `select_rows` / `scatter_rows` on one slot state and must match stepping each session alone;
/// rows not in a batch stay untouched.
fn slots_match_single_sessions<C: CoreInit>() {
    use burn::tensor::{Int, TensorData};
    let (net, _) = net::<C>(8);
    let core = net.core();
    let (cap, d) = (5, 32 + 32 + 32);
    let mut slots = core.init_state(cap, &dev());
    let mut alone: Vec<C::State> = (0..cap).map(|_| core.init_state(1, &dev())).collect();
    // Which slots act at each step (distinct rows, any order).
    let plan: [&[usize]; 6] = [&[0, 2, 4], &[3], &[4, 1], &[2, 3, 0, 1], &[1], &[0, 4, 3]];
    for (t, rows) in plan.iter().enumerate() {
        let b = rows.len();
        let x = Tensor::<2>::from_data(TensorData::new((0..b * d).map(|i| (((i + 13 * t) * 29 % 97) as f32 / 48.0) - 1.0).collect::<Vec<_>>(), [b, d]), &dev());
        let idx = Tensor::<1, Int>::from_data(TensorData::new(rows.iter().map(|&r| r as i32).collect::<Vec<_>>(), [b]), &dev());
        let (h, sub) = core.step(x.clone(), C::select_rows(&slots, idx.clone()));
        slots = C::scatter_rows(slots, idx, sub);
        let got = floats(&h);
        for (k, &r) in rows.iter().enumerate() {
            let (h1, s1) = core.step(x.clone().narrow(0, k, 1), alone[r].clone());
            alone[r] = s1;
            let want = floats(&h1);
            for (g, w) in got[k * core.hidden()..(k + 1) * core.hidden()].iter().zip(&want) {
                assert!((g - w).abs() < 1e-4, "step {t} slot {r}: {g} != {w}");
            }
        }
    }
}

/// 032-R16: a state survives the host round trip bit for bit, and host states of single rows
/// stacked with `HostState::cat` equal the batch state.
fn host_state_round_trip<C: CoreInit>() {
    use burn::tensor::{Int, TensorData};
    let (net, _) = net::<C>(10);
    let core = net.core();
    let (cap, d) = (3, 32 + 32 + 32);
    let x = Tensor::<2>::from_data(TensorData::new((0..cap * d).map(|i| ((i * 31 % 89) as f32 / 44.0) - 1.0).collect::<Vec<_>>(), [cap, d]), &dev());
    let (_, s) = core.step(x, core.init_state(cap, &dev()));
    let host = C::state_to_host(&s);
    assert_eq!(host.batch(), cap);
    assert_eq!(C::state_to_host(&C::state_from_host(&host, &dev())), host);
    let rows: Vec<_> = (0..cap)
        .map(|r| C::state_to_host(&C::select_rows(&s, Tensor::<1, Int>::from_data(TensorData::new(vec![r as i32], [1]), &dev()))))
        .collect();
    assert_eq!(uranium_model::HostState::cat(&rows.iter().collect::<Vec<_>>()), host);
}

/// 032-R15: per-head log-probs add up to the joint log-prob; argmax (temp 0) is deterministic.
fn head_log_probs_and_argmax<C: CoreInit>() {
    let (net, cfg) = net::<C>(9);
    let cols = columns();
    let b = 16;
    let (obs, _) = random_obs(&cfg, &cols, b, 10, 21, &dev());
    let out = net.act(&obs, net.init_state(b, &dev()), 1.0);
    assert_eq!(out.head_log_prob.dims(), [b, uranium_model::heads::HEADS]);
    let joint = floats(&out.log_prob);
    for (r, row) in floats(&out.head_log_prob).chunks(uranium_model::heads::HEADS).enumerate() {
        let s: f32 = row.iter().sum();
        assert!((s - joint[r]).abs() < 1e-3, "row {r}: heads {s} != joint {}", joint[r]);
    }
    let a = net.act(&obs, net.init_state(b, &dev()), 0.0);
    let c = net.act(&obs, net.init_state(b, &dev()), 0.0);
    assert_eq!(host_actions(&a.action).action, host_actions(&c.action).action);
    assert_eq!(ints2(&a.action.selected), ints2(&c.action.selected));
    assert_eq!(ints1(&a.action.target_loc), ints1(&c.action.target_loc));
}

macro_rules! core_tests {
    ($m:ident, $core:ty) => {
        mod $m {
            use super::*;
            #[test]
            fn masks_hold_032_ac4() {
                masks_hold::<$core>(1_000);
            }
            /// Full 032-AC4 run: `cargo test --release -p uranium-model -- --ignored masks_hold_100k`.
            #[test]
            #[ignore]
            fn masks_hold_100k_032_ac4() {
                masks_hold::<$core>(100_000);
            }
            #[test]
            fn privileged_only_reaches_value_032_ac5() {
                privileged_only_reaches_value::<$core>();
            }
            #[test]
            fn teacher_matches_sampling_032_r3() {
                teacher_matches_sampling::<$core>();
            }
            #[test]
            fn step_matches_sequence_032_r5() {
                step_matches_sequence::<$core>();
            }
            #[test]
            fn backward_reaches_parameters_032_r13() {
                backward_reaches_parameters::<$core>();
            }
            #[test]
            fn checkpoint_round_trip_032_r6() {
                checkpoint_round_trip::<$core>();
            }
            #[test]
            fn slots_match_single_sessions_032_r14() {
                slots_match_single_sessions::<$core>();
            }
            #[test]
            fn host_state_round_trip_032_r16() {
                host_state_round_trip::<$core>();
            }
            #[test]
            fn head_log_probs_and_argmax_032_r15() {
                head_log_probs_and_argmax::<$core>();
            }
        }
    };
}

core_tests!(lstm, LstmCore);
core_tests!(gtrxl, GtrxlCore);

/// 032-R2: the default configuration is ~12M parameters (allowed 10M–40M).
#[test]
fn default_size_032_r2() {
    let cfg = UraniumNetConfig::new();
    let net = UraniumNet::<LstmCore>::new(&cfg, &columns(), &dev());
    let params = net.num_params();
    eprintln!("default UraniumNet<LstmCore>: {params} parameters");
    assert!((10_000_000..=40_000_000).contains(&params), "{params} parameters outside 10M-40M");
    let gtrxl = UraniumNet::<GtrxlCore>::new(&cfg.clone().with_core(cfg.core.clone().with_kind(CoreKind::Gtrxl)), &columns(), &dev());
    eprintln!("default UraniumNet<GtrxlCore>: {} parameters", gtrxl.num_params());
}

/// Parameter gradients in visit order.
struct Grads<'a>(&'a burn::tensor::Gradients, Vec<Vec<f32>>);

impl burn::module::ModuleVisitor for Grads<'_> {
    fn visit_float<const D: usize>(&mut self, p: &burn::module::Param<Tensor<D>>) {
        self.1.push(p.val().grad(self.0).map(|g| g.into_data().convert::<f32>().try_to_vec::<f32>().unwrap()).unwrap_or_default());
    }
}

/// 032-R5: the batched LSTM (`LstmCore::forward_seq`, `step`) gives Burn's `Lstm` outputs, states
/// and parameter gradients for the same weights (BC throughput change 2026-10-18).
#[test]
fn lstm_core_matches_burn_lstm_032_r5() {
    use burn::module::Module;
    use burn::nn::LstmState;
    use burn::tensor::Distribution;
    use uranium_model::Core;
    let ad = dev().autodiff();
    ad.seed(41);
    let core = LstmCore::new(12, &uranium_model::config::CoreConfig::new().with_hidden(16), &ad);
    let (b, t) = (3, 5);
    let x = Tensor::<3>::random([b, t, 12], Distribution::Normal(0.0, 1.0), &ad);
    let s0 = || LstmState::new(Tensor::<2>::random([b, 16], Distribution::Normal(0.0, 0.5), &ad), Tensor::<2>::random([b, 16], Distribution::Normal(0.0, 0.5), &ad));
    let c0 = s0();
    let w = Tensor::<3>::random([b, t, 16], Distribution::Normal(0.0, 1.0), &ad);
    let run = |fast: bool| {
        let s = LstmState::new(c0.cell.clone(), c0.hidden.clone());
        let (out, st) = if fast { core.forward_seq(x.clone(), s) } else { core.forward_seq_reference(x.clone(), s) };
        let loss = (out.clone() * w.clone()).sum() + st.cell.clone().sum() * 0.3 + st.hidden.clone().sum() * 0.7;
        let grads = loss.backward();
        let mut g = Grads(&grads, Vec::new());
        core.visit(&mut g);
        let flat = |t: Tensor<3>| t.into_data().convert::<f32>().try_to_vec::<f32>().unwrap();
        (flat(out), st.cell.into_data().convert::<f32>().try_to_vec::<f32>().unwrap(), g.1)
    };
    let (fo, fc, fg) = run(true);
    let (ro, rc, rg) = run(false);
    let close = |a: &[f32], b: &[f32], what: &str| {
        let d = a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()));
        assert!(d <= 1e-5, "{what}: max diff {d:.2e}");
    };
    close(&fo, &ro, "outputs");
    close(&fc, &rc, "final cell");
    assert_eq!(fg.len(), rg.len());
    for (i, (a, r)) in fg.iter().zip(&rg).enumerate() {
        assert_eq!(a.len(), r.len(), "parameter {i}");
        close(a, r, &format!("gradient of parameter {i}"));
    }
}

/// Teacher forcing computes the location head only on rows whose action uses a location: the
/// loss and every parameter gradient equal the all-rows computation (2026-10-18, BC throughput).
#[test]
fn location_on_used_rows_matches_all_rows_032_r10() {
    use std::sync::atomic::Ordering;
    let ad = dev().autodiff();
    ad.seed(53);
    let cfg = tiny_config(CoreKind::Lstm);
    let cfg = cfg.clone().with_entity(cfg.entity.clone().with_dropout(0.0));
    let cols = columns();
    let net = UraniumNet::<LstmCore>::new(&cfg, &cols, &ad);
    let (b, t) = (2, 4);
    let (obs, _) = random_obs(&cfg, &cols, b * t, 16, 59, &ad);
    let labels = net.act(&obs, net.init_state(b * t, &ad), 1.0).action;
    let run = |all: bool| {
        uranium_model::heads::LOCATION_ALL_ROWS.store(all, Ordering::SeqCst);
        let out = net.forward_seq(&obs, &labels, None, net.init_state(b, &ad), t);
        uranium_model::heads::LOCATION_ALL_ROWS.store(false, Ordering::SeqCst);
        let lp = &out.heads.logp;
        let loss = (lp.action.clone() + lp.delay.clone() + lp.queued.clone() + lp.selected.clone() + lp.target_unit.clone() + lp.target_loc.clone()).neg().sum()
            + out.value.powi_scalar(2).sum();
        let used = out.heads.dists.uses_target_loc.clone().into_data().convert::<bool>().try_to_vec::<bool>().unwrap();
        let lp_loc = floats(&out.heads.logp.target_loc);
        let grads = loss.backward();
        let mut g = Grads(&grads, Vec::new());
        net.visit(&mut g);
        (floats(&loss)[0], lp_loc, used, g.1)
    };
    let (l_sub, lp_sub, used, g_sub) = run(false);
    let (l_all, lp_all, _, g_all) = run(true);
    let n_used = used.iter().filter(|u| **u).count();
    assert!(n_used > 0 && n_used < used.len(), "the sample must mix rows with and without a location ({n_used}/{})", used.len());
    assert!((l_sub - l_all).abs() <= 1e-5 * l_all.abs().max(1.0), "loss {l_sub} vs {l_all}");
    for (a, c) in lp_sub.iter().zip(&lp_all) {
        assert!((a - c).abs() <= 1e-5, "target_loc log-prob {a} vs {c}");
    }
    for (i, (a, c)) in g_sub.iter().zip(&g_all).enumerate() {
        let d = a.iter().zip(c).fold(0f32, |m, (x, y)| m.max((x - y).abs()));
        assert!(d <= 1e-5, "gradient of parameter {i}: max diff {d:.2e}");
    }
}
