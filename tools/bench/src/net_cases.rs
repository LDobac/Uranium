//! Full `UraniumNet` benchmarks (spec 032): inference latency (032-AC2), training throughput
//! (032-AC3) and the distance-bias backward / forward ratio (032-R13). Inputs are synthetic
//! (`uranium_model::synthetic`), so no game data is needed.

use std::time::Instant;

use burn::tensor::Device;
use uranium_model::batch::ActionBatch;
use uranium_model::config::{CoreKind, UraniumNetConfig};
use uranium_model::entity::EntityEncoder;
use uranium_model::synthetic::{columns, random_obs, random_priv};
use uranium_model::{CoreInit, GtrxlCore, LstmCore, UraniumNet};

use crate::cases::{Case, Ctx};
use crate::util::{Row, sync};

pub fn all() -> Vec<Case> {
    vec![
        Case { name: "net_act_lstm_b16_n256", run: act::<LstmCore> },
        Case { name: "net_act_gtrxl_b16_n256", run: act::<GtrxlCore> },
        Case { name: "net_train_lstm_b2_t16_n256", run: |c| train::<LstmCore>(c, 2, 16) },
        Case { name: "net_train_lstm_b4_t16_n256", run: |c| train::<LstmCore>(c, 4, 16) },
        Case { name: "net_train_gtrxl_b2_t16_n256", run: |c| train::<GtrxlCore>(c, 2, 16) },
        Case { name: "net_entity_bias_ratio_b16_n256", run: entity_ratio },
        Case { name: "net_act_parts_entity_b16_n256", run: |c| parts(c, Part::Entity) },
        Case { name: "net_act_parts_spatial_b16_n256", run: |c| parts(c, Part::Spatial) },
    ]
}

enum Part {
    Entity,
    Spatial,
}

/// Where `act()` time goes: the entity encoder alone, and entity + spatial encoders.
fn parts(c: &Ctx, part: Part) -> Row {
    let (b, n) = (if c.quick { 2 } else { 16 }, 256);
    let cfg = config(CoreKind::Lstm, c.quick);
    let cols = columns();
    let enc = EntityEncoder::new(&cfg.input, &cfg.entity, &c.device);
    let sp = uranium_model::spatial::SpatialEncoder::new(&cfg.input, &cfg.spatial, cfg.entity.d_model, &c.device);
    let (obs, _) = random_obs(&cfg, &cols, b, n, 5, &c.device);
    let name = match part {
        Part::Entity => "net_act_parts_entity",
        Part::Spatial => "net_act_parts_entity_spatial",
    };
    let mut row = Row::new(name, format!("B={b} N={n}"));
    let (med, p95) = time_p95(c.warmup, c.iters.max(50), || {
        let e = enc.forward(&obs);
        match part {
            Part::Entity => {
                let _ = e.pool.into_data();
            }
            Part::Spatial => {
                let _ = sp.forward(&obs, e.e).pool.into_data();
            }
        }
        sync(&c.device);
    });
    row.fwd_ms = Some(med);
    row.note = format!("p95 {p95:.1} ms");
    row
}

fn config(kind: CoreKind, quick: bool) -> UraniumNetConfig {
    let cfg = UraniumNetConfig::new();
    let cfg = cfg.clone().with_core(cfg.core.clone().with_kind(kind));
    // BC dropout 0.1 is in the default config; latency is measured without it (inference).
    if quick { cfg.clone().with_input(cfg.input.clone().with_map_size(64)) } else { cfg }
}

/// (median, p95) in ms; `f` must block until the device has finished.
fn time_p95(warmup: usize, iters: usize, mut f: impl FnMut()) -> (f64, f64) {
    for _ in 0..warmup {
        f();
    }
    let mut s: Vec<f64> = (0..iters)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    s.sort_by(f64::total_cmp);
    let p95 = s[((s.len() as f64 * 0.95).ceil() as usize).clamp(1, s.len()) - 1];
    (s[s.len() / 2], p95)
}

/// 032-AC2: `act()` for B=16 games with N=256 entities, p95 <= 40 ms.
fn act<C: CoreInit>(c: &Ctx) -> Row {
    let (b, n) = (if c.quick { 2 } else { 16 }, 256);
    let cfg = config(C::KIND, c.quick);
    let cols = columns();
    let net = UraniumNet::<C>::new(&cfg, &cols, &c.device);
    let (obs, _) = random_obs(&cfg, &cols, b, n, 1, &c.device);
    let mut row = Row::new(&format!("net_act_{:?}", C::KIND).to_lowercase(), format!("B={b} N={n} map={}", cfg.input.map_size));
    let (med, p95) = time_p95(c.warmup, c.iters.max(50), || {
        let out = net.act(&obs, net.init_state(b, &c.device), 1.0);
        let _ = out.action.action.into_data();
        sync(&c.device);
    });
    row.fwd_ms = Some(med);
    row.note = format!("p95 {p95:.1} ms (032-AC2 <= 40 ms), {} params", net_params(&net));
    row
}

fn net_params<C: CoreInit>(net: &UraniumNet<C>) -> usize {
    use burn::module::Module;
    net.num_params()
}

/// 032-AC3: teacher-forced forward + backward over B x T steps; reports samples/s.
fn train<C: CoreInit>(c: &Ctx, b: usize, t: usize) -> Row {
    let (b, t) = if c.quick { (1, 4) } else { (b, t) };
    let n = 256;
    let ad = c.device.clone().autodiff();
    let cfg = config(C::KIND, c.quick);
    let cols = columns();
    let net = UraniumNet::<C>::new(&cfg, &cols, &ad);
    let (obs, _) = random_obs(&cfg, &cols, b * t, n, 2, &ad);
    let labels: ActionBatch = net.act(&obs, net.init_state(b * t, &ad), 1.0).action;
    let priv_ = random_priv(&cfg, b * t, 3, &ad);
    let mut row = Row::new(&format!("net_train_{:?}_b{b}_t{t}", C::KIND).to_lowercase(), format!("B={b} T={t} N={n} map={}", cfg.input.map_size));
    let (med, _) = time_p95(c.warmup, c.iters, || {
        let out = net.forward_seq(&obs, &labels, Some(&priv_), net.init_state(b, &ad), t);
        let loss = out.heads.logp.sum().neg().mean() + out.value.powi_scalar(2).mean();
        let _ = loss.backward();
        sync(&ad);
    });
    row.fwd_bwd_ms = Some(med);
    row.note = format!("{:.1} samples/s", (b * t) as f64 / (med / 1e3));
    row
}

/// 032-R13: EntityEncoder (with distance bias) forward+backward must stay within 5x forward.
fn entity_ratio(c: &Ctx) -> Row {
    let (b, n) = (if c.quick { 2 } else { 16 }, 256);
    let cfg = config(CoreKind::Lstm, c.quick);
    let cols = columns();
    let mut row = Row::new("net_entity_bias_ratio", format!("B={b} N={n} L3 h8"));
    let run = |device: &Device, backward: bool| {
        let enc = EntityEncoder::new(&cfg.input, &cfg.entity, device);
        let (obs, _) = random_obs(&cfg, &cols, b, n, 4, device);
        time_p95(c.warmup, c.iters, || {
            let out = enc.forward(&obs);
            if backward {
                let _ = out.e.sum().backward();
            } else {
                let _ = out.e.sum().into_data();
            }
            sync(device);
        })
        .0
    };
    let fwd = run(&c.device, false);
    let fwd_bwd = run(&c.device.clone().autodiff(), true);
    row.fwd_ms = Some(fwd);
    row.fwd_bwd_ms = Some(fwd_bwd);
    row.note = format!("fwd+bwd / fwd = {:.1}x (032-R13 <= 5x backward / forward)", fwd_bwd / fwd);
    row
}
