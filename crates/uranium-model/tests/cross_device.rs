//! 032-AC1: same weights and inputs give forward outputs close to Flex (CPU) on Vulkan and CUDA.
//! 032-AC8: the backward pass (BC loss and per-parameter gradients) is close to Flex as well.
//! Vulkan: <= 1e-4 (f32). CUDA: <= `CUDA_TOL`, because Burn's CUDA matmul computes f32 inputs in
//! TF32 (AMB-032-5, user decision 2026-10-16). Needs a GPU, so ignored by default:
//!   URANIUM_TEST_DEVICE=vulkan cargo test -p uranium-model --test cross_device -- --ignored --nocapture
//!   URANIUM_TEST_DEVICE=cuda cargo test --release -p uranium-model --features cuda --test cross_device -- --ignored --nocapture

mod common;

use burn::module::Module;
use burn::tensor::{Device, DeviceKind, Tensor};
use common::*;
use uranium_model::config::{CoreKind, UraniumNetConfig};
use uranium_model::{ActionBatch, CoreInit, GtrxlCore, LstmCore, UraniumNet};

/// 032-AC1 tolerance on Vulkan (and any exact-f32 backend).
const VULKAN_TOL: f32 = 1e-4;
/// AMB-032-5: absolute tolerance on CUDA (TF32 matmul) for values and per-head log-probs.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
const CUDA_TOL: f32 = 1e-2;

fn target() -> (Device, f32) {
    match std::env::var("URANIUM_TEST_DEVICE").as_deref() {
        Ok("vulkan") => (Device::vulkan(DeviceKind::DefaultDevice), VULKAN_TOL),
        #[cfg(feature = "cuda")]
        Ok("cuda") => (Device::cuda(0), CUDA_TOL),
        Ok(other) => panic!("URANIUM_TEST_DEVICE={other} not supported by this build (cuda needs --features cuda)"),
        Err(_) => panic!("set URANIUM_TEST_DEVICE to vulkan or cuda"),
    }
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()))
}

fn to(labels: &ActionBatch, d: &Device) -> ActionBatch {
    ActionBatch {
        action: Tensor::from_data(labels.action.clone().into_data(), d),
        delay: Tensor::from_data(labels.delay.clone().into_data(), d),
        queued: Tensor::from_data(labels.queued.clone().into_data(), d),
        selected: Tensor::from_data(labels.selected.clone().into_data(), d),
        target_unit: Tensor::from_data(labels.target_unit.clone().into_data(), d),
        target_loc: Tensor::from_data(labels.target_loc.clone().into_data(), d),
    }
}

/// Mean KL(p_cpu || p_gpu) over rows of two masked log-softmax tables [B,K].
fn mean_kl(cpu: &[f32], gpu: &[f32], k: usize) -> f64 {
    let rows = cpu.len() / k;
    let mut total = 0.0;
    for r in 0..rows {
        for i in 0..k {
            let (lc, lg) = (cpu[r * k + i] as f64, gpu[r * k + i] as f64);
            if lc > -1e8 {
                total += lc.exp() * (lc - lg);
            }
        }
    }
    total / rows as f64
}

struct Stats {
    worst: f32,
}

/// Teacher-forced outputs on both devices (same weights, inputs, labels) and greedy decisions.
fn measure<C: CoreInit>(cfg: &UraniumNetConfig, b: usize, n: usize, verbose: bool) -> Stats {
    let (cpu, (gpu, _)) = (Device::flex(), target());
    cpu.seed(21);
    let cols = columns();
    let net = UraniumNet::<C>::new(cfg, &cols, &cpu);
    let (obs, _) = random_obs(cfg, &cols, b, n, 23, &cpu);
    let labels = net.act(&obs, net.init_state(b, &cpu), 0.0).action;
    let want = net.forward_seq(&obs, &labels, None, net.init_state(b, &cpu), 1);
    let gnet = net.clone().to_device(&gpu);
    let (gobs, _) = random_obs(cfg, &cols, b, n, 23, &gpu);
    let got = gnet.forward_seq(&gobs, &to(&labels, &gpu), None, gnet.init_state(b, &gpu), 1);

    let pairs = [
        ("value", floats(&want.value), floats(&got.value)),
        ("logp.action", floats(&want.heads.logp.action), floats(&got.heads.logp.action)),
        ("logp.delay", floats(&want.heads.logp.delay), floats(&got.heads.logp.delay)),
        ("logp.queued", floats(&want.heads.logp.queued), floats(&got.heads.logp.queued)),
        ("logp.selected", floats(&want.heads.logp.selected), floats(&got.heads.logp.selected)),
        ("logp.target_unit", floats(&want.heads.logp.target_unit), floats(&got.heads.logp.target_unit)),
        ("logp.target_loc", floats(&want.heads.logp.target_loc), floats(&got.heads.logp.target_loc)),
    ];
    let mut worst = 0f32;
    for (name, w, g) in &pairs {
        let d = max_abs(w, g);
        if verbose {
            eprintln!("{:?} {name}: max |gpu - cpu| = {d:.2e}", C::KIND);
        }
        worst = worst.max(d);
    }
    if verbose {
        let a = cfg.input.actions;
        let m2 = cfg.input.map_size * cfg.input.map_size;
        eprintln!("{:?} KL(cpu||gpu) action_type mean = {:.2e}", C::KIND, mean_kl(&floats(&want.heads.dists.action), &floats(&got.heads.dists.action), a));
        eprintln!("{:?} KL(cpu||gpu) location mean = {:.2e}", C::KIND, mean_kl(&floats(&want.heads.dists.target_loc), &floats(&got.heads.dists.target_loc), m2));
        // Greedy decisions made fully on each device (errors cascade through the autoregressive heads).
        let gc = net.act(&obs, net.init_state(b, &cpu), 0.0).action;
        let gg = gnet.act(&gobs, gnet.init_state(b, &gpu), 0.0).action;
        let same = |x: Vec<i64>, y: Vec<i64>| x.iter().zip(&y).filter(|(a, b)| a == b).count();
        let (ac, ag) = (ints1(&gc.action), ints1(&gg.action));
        let s = gc.selected.dims()[1];
        let (sc, sg) = (ints2(&gc.selected), ints2(&gg.selected));
        let sel_same = (0..b).filter(|&r| sc[r * s..(r + 1) * s] == sg[r * s..(r + 1) * s]).count();
        let full_same = (0..b)
            .filter(|&r| {
                ac[r] == ag[r]
                    && sc[r * s..(r + 1) * s] == sg[r * s..(r + 1) * s]
                    && ints1(&gc.target_loc)[r] == ints1(&gg.target_loc)[r]
                    && ints1(&gc.target_unit)[r] == ints1(&gg.target_unit)[r]
                    && ints1(&gc.delay)[r] == ints1(&gg.delay)[r]
            })
            .count();
        eprintln!(
            "{:?} greedy agreement over {b} rows: action_type {}/{b}, selected {sel_same}/{b}, location {}/{b}, whole action {full_same}/{b}",
            C::KIND,
            same(ac, ag),
            same(ints1(&gc.target_loc), ints1(&gg.target_loc)),
        );
    }
    Stats { worst }
}

fn compare<C: CoreInit>() {
    let tol = target().1;
    let s = measure::<C>(&tiny_config(C::KIND), 8, 16, true);
    assert!(s.worst <= tol, "{:?}: max abs difference {:.2e} > {tol:.0e} (032-AC1)", C::KIND, s.worst);
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

/// AMB-032-5 impact report: the default 11M-parameter model at B=16, N=256 (release build).
#[test]
#[ignore]
fn default_size_difference_report_032_ac1() {
    let tol = target().1;
    let base = UraniumNetConfig::new();
    for kind in [CoreKind::Lstm, CoreKind::Gtrxl] {
        let cfg = base.clone().with_core(base.core.clone().with_kind(kind)).with_entity(base.entity.clone().with_dropout(0.0));
        let s = match kind {
            CoreKind::Lstm => measure::<LstmCore>(&cfg, 16, 256, true),
            CoreKind::Gtrxl => measure::<GtrxlCore>(&cfg, 16, 256, true),
        };
        eprintln!("{kind:?} default size: worst {:.2e} (tolerance {tol:.0e})", s.worst);
        assert!(s.worst <= tol);
    }
}

/// Gradients of every parameter, in visit order.
struct GradDump<'a> {
    grads: &'a burn::tensor::Gradients,
    out: Vec<Vec<f32>>,
}

impl burn::module::ModuleVisitor for GradDump<'_> {
    fn visit_float<const D: usize>(&mut self, param: &burn::module::Param<Tensor<D>>) {
        self.out.push(param.val().grad(self.grads).map(|g| floats(&g)).unwrap_or_default());
    }
}

/// Training loss and parameter gradients of one teacher-forced sequence pass (T steps, as BC).
fn loss_and_grads<C: CoreInit>(net: &UraniumNet<C>, cfg: &UraniumNetConfig, labels: &ActionBatch, b: usize, t: usize, n: usize, d: &Device) -> (f32, Vec<Vec<f32>>) {
    let cols = columns();
    let (obs, _) = random_obs(cfg, &cols, b * t, n, 23, d);
    let out = net.forward_seq(&obs, &to(labels, d), None, net.init_state(b, d), t);
    let lp = &out.heads.logp;
    let loss = (lp.action.clone() + lp.delay.clone() + lp.queued.clone() + lp.selected.clone() + lp.target_unit.clone() + lp.target_loc.clone()).neg().mean()
        + out.value.powi_scalar(2).mean();
    let l = floats(&loss)[0];
    let grads = loss.backward();
    let mut dump = GradDump { grads: &grads, out: Vec::new() };
    net.visit(&mut dump);
    (l, dump.out)
}

/// The backward pass on the GPU matches Flex: loss within the forward tolerance and every
/// parameter gradient within it relative to the gradient's scale. Guards kernel choices that
/// must not change results (Burn autotune / fusion, 2026-10-18).
fn gradients_match<C: CoreInit>(cfg: &UraniumNetConfig, b: usize, t: usize, n: usize) {
    let (cpu, (gpu, tol)) = (Device::flex(), target());
    cpu.seed(31);
    let cols = columns();
    let (cpu_ad, gpu_ad) = (cpu.clone().autodiff(), gpu.clone().autodiff());
    let net = UraniumNet::<C>::new(cfg, &cols, &cpu_ad);
    let (obs, _) = random_obs(cfg, &cols, b * t, n, 23, &cpu_ad);
    let labels = net.act(&obs, net.init_state(b * t, &cpu_ad), 0.0).action;
    let (lc, gc) = loss_and_grads(&net.clone(), cfg, &labels, b, t, n, &cpu_ad);
    // `fork` keeps the parameters tracked on the new device (`to_device` would not).
    let (lg, gg) = loss_and_grads(&net.fork(&gpu_ad), cfg, &labels, b, t, n, &gpu_ad);
    eprintln!("{:?} loss cpu {lc:.6} gpu {lg:.6}", C::KIND);
    assert!((lc - lg).abs() <= tol * lc.abs().max(1.0), "loss differs: {lc} vs {lg}");
    // Per parameter: ||g_gpu - g_cpu|| / ||g_cpu|| (L2), and the max-abs ratio for the report.
    let l2 = |v: &[f32]| v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    // Parameters whose gradient is ~0 in exact arithmetic (e.g. a bias right before a norm) only
    // carry rounding noise: measure against 1e-3 of the largest parameter gradient at least.
    let floor = 1e-3 * gc.iter().map(|a| l2(a)).fold(0.0, f64::max);
    let mut rows = Vec::new();
    for (i, (a, g)) in gc.iter().zip(&gg).enumerate() {
        assert_eq!(a.is_empty(), g.is_empty(), "parameter {i}: gradient on one device only");
        if a.is_empty() {
            continue;
        }
        let diff: Vec<f32> = a.iter().zip(g).map(|(x, y)| x - y).collect();
        let rel_l2 = l2(&diff) / l2(a).max(floor).max(1e-12);
        let rel_max = max_abs(a, g) / a.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-12);
        rows.push((i, a.len(), rel_l2, rel_max));
    }
    rows.sort_by(|x, y| y.2.total_cmp(&x.2));
    for (i, len, rel_l2, rel_max) in rows.iter().take(5) {
        eprintln!("{:?} parameter {i} ({len} values): L2 rel {rel_l2:.2e}, max-abs rel {rel_max:.2e}", C::KIND);
    }
    let worst = rows.first().map_or(0.0, |r| r.2);
    eprintln!("{:?} {} parameters, worst L2 relative gradient difference {worst:.2e}", C::KIND, rows.len());
    // 032-AC8: gradients sum many more rounding steps than one forward value: Vulkan f32 lands
    // near 5e-4 with or without kernel changes (2026-10-18), so its bound is 1e-3. The CUDA bound
    // is TBD(M6) (032-AC8: re-measure with autotune + fusion, then confirm with the user); until
    // then the current value 1e-2 (the forward CUDA tolerance) is kept.
    let grad_tol = (tol as f64).max(1e-3);
    assert!(worst <= grad_tol, "gradient L2 relative difference {worst:.2e} > {grad_tol:.0e}");
}

#[test]
#[ignore]
fn lstm_gradients_match_flex_032_ac8() {
    let base = tiny_config(CoreKind::Lstm);
    let cfg = base.clone().with_entity(base.entity.clone().with_dropout(0.0));
    gradients_match::<LstmCore>(&cfg, 2, 4, 16);
}
