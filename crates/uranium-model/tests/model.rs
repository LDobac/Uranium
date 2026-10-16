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
