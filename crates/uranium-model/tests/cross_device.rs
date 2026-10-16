//! 032-AC1: same weights and inputs give forward outputs within 1e-4 (f32) of Flex on Vulkan and
//! CUDA. Needs a GPU, so it is ignored by default:
//!   URANIUM_TEST_DEVICE=vulkan cargo test -p uranium-model --test cross_device -- --ignored --nocapture
//!   URANIUM_TEST_DEVICE=cuda cargo test -p uranium-model --features cuda --test cross_device -- --ignored --nocapture

mod common;

use burn::module::Module;
use burn::tensor::{Device, DeviceKind};
use common::*;
use uranium_model::{CoreInit, GtrxlCore, LstmCore, UraniumNet};

const TOL: f32 = 1e-4;

fn target() -> Device {
    match std::env::var("URANIUM_TEST_DEVICE").as_deref() {
        Ok("vulkan") => Device::vulkan(DeviceKind::DefaultDevice),
        #[cfg(feature = "cuda")]
        Ok("cuda") => Device::cuda(0),
        Ok(other) => panic!("URANIUM_TEST_DEVICE={other} not supported by this build (cuda needs --features cuda)"),
        Err(_) => panic!("set URANIUM_TEST_DEVICE to vulkan or cuda"),
    }
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()))
}

fn compare<C: CoreInit>() {
    let (cpu, gpu) = (Device::flex(), target());
    cpu.seed(21);
    let cfg = tiny_config(C::KIND);
    let cols = columns();
    let net = UraniumNet::<C>::new(&cfg, &cols, &cpu);
    let b = 8;
    let (obs, _) = random_obs(&cfg, &cols, b, 16, 23, &cpu);
    let labels = net.act(&obs, net.init_state(b, &cpu), 0.0).action;
    let want = net.forward_seq(&obs, &labels, None, net.init_state(b, &cpu), 1);

    let gnet = net.clone().to_device(&gpu);
    let (gobs, _) = random_obs(&cfg, &cols, b, 16, 23, &gpu);
    let glabels = uranium_model::ActionBatch {
        action: burn::tensor::Tensor::from_data(labels.action.into_data(), &gpu),
        delay: burn::tensor::Tensor::from_data(labels.delay.into_data(), &gpu),
        queued: burn::tensor::Tensor::from_data(labels.queued.into_data(), &gpu),
        selected: burn::tensor::Tensor::from_data(labels.selected.into_data(), &gpu),
        target_unit: burn::tensor::Tensor::from_data(labels.target_unit.into_data(), &gpu),
        target_loc: burn::tensor::Tensor::from_data(labels.target_loc.into_data(), &gpu),
    };
    let got = gnet.forward_seq(&gobs, &glabels, None, gnet.init_state(b, &gpu), 1);

    let pairs = [
        ("value", floats(&want.value), floats(&got.value)),
        ("logp.action", floats(&want.heads.logp.action), floats(&got.heads.logp.action)),
        ("logp.delay", floats(&want.heads.logp.delay), floats(&got.heads.logp.delay)),
        ("logp.selected", floats(&want.heads.logp.selected), floats(&got.heads.logp.selected)),
        ("logp.target_unit", floats(&want.heads.logp.target_unit), floats(&got.heads.logp.target_unit)),
        ("logp.target_loc", floats(&want.heads.logp.target_loc), floats(&got.heads.logp.target_loc)),
        ("dist.action", floats(&want.heads.dists.action), floats(&got.heads.dists.action)),
    ];
    let mut worst = 0f32;
    for (name, w, g) in &pairs {
        let d = max_abs(w, g);
        eprintln!("{:?} {name}: max |gpu - flex| = {d:.2e}", C::KIND);
        worst = worst.max(d);
    }
    assert!(worst <= TOL, "{:?}: max abs difference {worst:.2e} > {TOL:.0e} (032-AC1)", C::KIND);
}

#[test]
#[ignore]
fn lstm_matches_flex_032_ac1() {
    compare::<LstmCore>();
}

#[test]
#[ignore]
fn gtrxl_matches_flex_032_ac1() {
    compare::<GtrxlCore>();
}
