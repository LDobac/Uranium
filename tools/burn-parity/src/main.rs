//! 032-AC10 (PyTorch switch, AMB-000-4; temporary, removed with Burn): writes, for one model
//! config, everything the PyTorch port must reproduce on CPU:
//! - `weights.safetensors`: every float parameter under its Burn module path (Burn layout)
//! - `inputs.safetensors`: an observation batch (B·T rows, b-major), labels, privileged inputs,
//!   and the BC loss weights / results (041-R1, R7)
//! - `outputs.safetensors`: teacher-forced per-head log-prob and entropy, value, win_open, loss
//! - `grads.safetensors`: the loss gradient of every parameter (same paths)
//! - `clipped.safetensors`, `adamw.safetensors`: the gradients after the global clip (041-R3,
//!   041-R19) and the parameters after one AdamW step
//! - `meta.json`: the model config (Burn `UraniumNetConfig` JSON), B, T, N, loss settings
//!
//! Flex (CPU), dropout 0. `burn-parity --out <dir> --config tiny-lstm|tiny-gtrxl|full-lstm|full-gtrxl`.

use std::path::{Path, PathBuf};

use anyhow::Result;
use burn::module::{Module, ModuleVisitor, Param, ParamId};
use burn::optim::{AdamWConfig, GradientsParams};
use burn::tensor::{Device, Tensor, TensorData};
use clap::Parser;
use safetensors::tensor::{Dtype, TensorView};
use uranium_model::batch::{ActionBatch, ObsBatch};
use uranium_model::config::{CoreKind, UraniumNetConfig};
use uranium_model::synthetic::{columns, random_obs, random_priv, tiny_config};
use uranium_model::value::VALUE_WIN;
use uranium_model::{CoreInit, GtrxlCore, LstmCore, UraniumNet};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    out: PathBuf,
    /// tiny-lstm | tiny-gtrxl | full-lstm | full-gtrxl
    #[arg(long, required_unless_present = "export")]
    config: Option<String>,
    /// BC parity (AMB-000-4 step 1-6): write this Burn checkpoint's weights (`weights.safetensors`,
    /// Burn paths) and net config (`meta.json`) to `--out` instead of a model dump.
    #[arg(long)]
    export: Option<PathBuf>,
    #[arg(long, default_value = "7")]
    seed: u64,
}

/// Head weights of `config/bc.toml` (041-R1) and the value weights (041-R7, R19).
const HEADS: [f32; 6] = [1.0, 0.5, 0.2, 1.0, 1.0, 1.0];
const VALUE_WEIGHT: f32 = 0.5;
const WIN_OPEN_WEIGHT: f32 = 0.5;
const LR: f64 = 3e-4;
const WEIGHT_DECAY: f32 = 1e-4;
/// Small, so the clip is active and the win_open head is clipped on its own (041-R19).
const CLIP: f32 = 0.1;

/// Host tensors to write: name, dtype, shape, little-endian bytes.
#[derive(Default)]
struct Out(Vec<(String, Dtype, Vec<usize>, Vec<u8>)>);

impl Out {
    fn f32(&mut self, name: &str, data: TensorData) {
        let shape = data.shape().to_vec();
        let v: Vec<f32> = data.convert::<f32>().try_into_vec().unwrap();
        self.0.push((name.into(), Dtype::F32, shape, v.iter().flat_map(|x| x.to_le_bytes()).collect()));
    }
    fn i32(&mut self, name: &str, data: TensorData) {
        let shape = data.shape().to_vec();
        let v: Vec<i32> = data.convert::<i32>().try_into_vec().unwrap();
        self.0.push((name.into(), Dtype::I32, shape, v.iter().flat_map(|x| x.to_le_bytes()).collect()));
    }
    fn bool(&mut self, name: &str, t: Tensor<1, burn::tensor::Bool>) {
        self.i32(name, t.int().into_data());
    }
    fn write(&self, path: &Path) -> Result<()> {
        let views: Vec<(String, TensorView)> = self.0.iter().map(|(n, d, s, b)| (n.clone(), TensorView::new(*d, s.clone(), b).unwrap())).collect();
        std::fs::write(path, safetensors::tensor::serialize(views, None)?)?;
        Ok(())
    }
}

/// Float parameters by Burn module path ("entity.layers.0.attn.q.weight").
struct Params<'a> {
    path: Vec<String>,
    grads: Option<&'a GradientsParams>,
    out: Out,
    ids: Vec<(String, ParamId)>,
}

impl ModuleVisitor for Params<'_> {
    fn enter_module(&mut self, name: &str, _container: &str) {
        self.path.push(name.to_string());
    }
    fn exit_module(&mut self, _name: &str, _container: &str) {
        self.path.pop();
    }
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        let name = self.path.join(".");
        self.ids.push((name.clone(), param.id));
        match self.grads {
            None => self.out.f32(&name, param.val().into_data()),
            Some(g) => {
                if let Some(t) = g.get::<D>(param.id) {
                    self.out.f32(&name, t.into_data());
                }
            }
        }
    }
}

fn params<M: Module>(m: &M, grads: Option<&GradientsParams>) -> (Out, Vec<(String, ParamId)>) {
    let mut v = Params { path: Vec::new(), grads, out: Out::default(), ids: Vec::new() };
    m.visit(&mut v);
    (v.out, v.ids)
}

fn obs_out(o: &mut Out, ob: &ObsBatch) {
    o.i32("entity_cat", ob.entity_cat.clone().into_data());
    o.f32("entity_cont", ob.entity_cont.clone().into_data());
    o.i32("entity_mask", ob.entity_mask.clone().int().into_data());
    o.i32("entity_cell", ob.entity_cell.clone().into_data());
    o.f32("entity_pos", ob.entity_pos.clone().into_data());
    o.f32("spatial", ob.spatial.clone().into_data());
    o.f32("scalar", ob.scalar.clone().into_data());
    o.i32("scalar_cat", ob.scalar_cat.clone().into_data());
    o.i32("z_build", ob.z_build.clone().into_data());
    o.f32("z_units", ob.z_units.clone().into_data());
    o.bool("z_present", ob.z_present.clone());
    o.i32("type_mask", ob.type_mask.clone().int().into_data());
    o.i32("unit_cap", ob.unit_cap.clone().int().into_data());
    o.i32("loc_class_mask", ob.loc_class_mask.clone().int().into_data());
    o.i32("is_building", ob.is_building.clone().int().into_data());
}

fn labels_out(o: &mut Out, a: &ActionBatch) {
    o.i32("action", a.action.clone().into_data());
    o.i32("delay", a.delay.clone().into_data());
    o.i32("queued", a.queued.clone().into_data());
    o.i32("selected", a.selected.clone().into_data());
    o.i32("target_unit", a.target_unit.clone().into_data());
    o.i32("target_loc", a.target_loc.clone().into_data());
}

/// A Burn checkpoint's weights by module path and its net config (flex).
fn export(dir: &Path, out: &Path) -> Result<()> {
    std::fs::create_dir_all(out)?;
    let meta = uranium_model::checkpoint::read_meta(dir)?;
    let cfg = meta.net_config()?;
    match cfg.core.kind {
        CoreKind::Lstm => export_as::<LstmCore>(dir, out),
        CoreKind::Gtrxl => export_as::<GtrxlCore>(dir, out),
    }
}

fn export_as<C: CoreInit>(dir: &Path, out: &Path) -> Result<()> {
    let dev = Device::flex();
    let (net, meta) = uranium_model::checkpoint::load::<C>(dir, &columns(), uranium_env::features::FEATURE_SCHEMA_VERSION, uranium_env::actions::ACTION_SCHEMA_VERSION, &dev)?;
    let (w, ids) = params(&net, None);
    w.write(&out.join("weights.safetensors"))?;
    let m = serde_json::json!({ "config": meta.config, "params": ids.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>(), "num_params": net.num_params(), "source": dir });
    std::fs::write(out.join("meta.json"), serde_json::to_vec_pretty(&m)?)?;
    println!("{} params -> {}", net.num_params(), out.display());
    Ok(())
}

fn dump<C: CoreInit>(cfg: UraniumNetConfig, out: &Path, b: usize, t: usize, n: usize, seed: u64) -> Result<()> {
    std::fs::create_dir_all(out)?;
    let ad = Device::flex().autodiff();
    ad.seed(seed);
    let cols = columns();
    let net = UraniumNet::<C>::new(&cfg, &cols, &ad);
    let bt = b * t;
    let (obs, _) = random_obs(&cfg, &cols, bt, n, seed ^ 0x11, &ad);
    let labels = net.act(&obs, net.init_state(bt, &ad), 1.0).action;
    let privs = random_priv(&cfg, bt, seed ^ 0x22, &ad);
    // 041-R1 / R7 weights: some void rows, one undecided lane per three (A-3).
    let weights: Vec<f32> = (0..bt).map(|r| if r % 5 == 4 { 0.0 } else { 1.0 }).collect();
    let results: Vec<f32> = (0..bt).map(|r| if (r / t) % 2 == 0 { 1.0 } else { -1.0 }).collect();
    let value_w: Vec<f32> = (0..bt).map(|r| if (r / t) % 3 == 2 { 0.0 } else { weights[r] }).collect();
    let valid = weights.iter().sum::<f32>().max(1.0);
    let value_valid = value_w.iter().sum::<f32>().max(1.0);

    let (w0, _) = params(&net, None);
    w0.write(&out.join("weights.safetensors"))?;
    let mut inp = Out::default();
    obs_out(&mut inp, &obs);
    labels_out(&mut inp, &labels);
    inp.f32("privileged", privs.summary.clone().into_data());
    inp.f32("weights", TensorData::new(weights.clone(), [bt]));
    inp.f32("results", TensorData::new(results.clone(), [bt]));
    inp.f32("value_w", TensorData::new(value_w.clone(), [bt]));
    inp.write(&out.join("inputs.safetensors"))?;

    let seq = net.forward_seq(&obs, &labels, Some(&privs), net.init_state(b, &ad), t);
    let w = Tensor::<1>::from_data(TensorData::new(weights, [bt]), &ad);
    let ce = |logp: Tensor<1>| (logp.neg() * w.clone()).sum() / valid as f64;
    let lp = &seq.heads.logp;
    let ent = &seq.heads.entropy;
    let heads = [lp.action.clone(), lp.delay.clone(), lp.queued.clone(), lp.selected.clone(), lp.target_unit.clone(), lp.target_loc.clone()];
    let ents = [ent.action.clone(), ent.delay.clone(), ent.queued.clone(), ent.selected.clone(), ent.target_unit.clone(), ent.target_loc.clone()];
    let target = Tensor::<1>::from_data(TensorData::new(results, [bt]), &ad);
    let vw = Tensor::<1>::from_data(TensorData::new(value_w, [bt]), &ad);
    let win = seq.value.clone().narrow(1, VALUE_WIN, 1).reshape([bt]);
    let mut loss = ((win - target.clone()).powi_scalar(2) * vw.clone()).sum() / value_valid as f64 * VALUE_WEIGHT as f64;
    for (h, k) in heads.iter().zip(HEADS) {
        loss = loss + ce(h.clone()) * k as f64;
    }
    if let Some(open) = seq.win_open.clone() {
        loss = loss + ((open - target).powi_scalar(2) * vw).sum() / value_valid as f64 * WIN_OPEN_WEIGHT as f64;
    }
    let mut o = Out::default();
    for (i, name) in ["action", "delay", "queued", "selected", "target_unit", "target_loc"].iter().enumerate() {
        o.f32(&format!("logp.{name}"), heads[i].clone().into_data());
        o.f32(&format!("entropy.{name}"), ents[i].clone().into_data());
    }
    o.f32("value", seq.value.clone().into_data());
    if let Some(open) = &seq.win_open {
        o.f32("win_open", open.clone().into_data());
    }
    o.f32("loss", loss.clone().into_data());
    o.write(&out.join("outputs.safetensors"))?;

    let grads = GradientsParams::from_grads(loss.backward(), &net);
    let (g, ids) = params(&net, Some(&grads));
    g.write(&out.join("grads.safetensors"))?;
    let (clipped, norm) = uranium_train::clip::clip_global_norm_except(&net, grads, CLIP, &net.win_open_params());
    let (c, _) = params(&net, Some(&clipped));
    c.write(&out.join("clipped.safetensors"))?;
    let mut optim = AdamWConfig::new().with_weight_decay(WEIGHT_DECAY).init();
    let stepped = optim.step(LR, net.clone(), clipped);
    let (a, _) = params(&stepped, None);
    a.write(&out.join("adamw.safetensors"))?;

    let win_open_params: Vec<String> = {
        let open: std::collections::HashSet<ParamId> = net.win_open_params().into_iter().collect();
        ids.iter().filter(|(_, id)| open.contains(id)).map(|(n, _)| n.clone()).collect()
    };
    let norm = norm.map(|x| x.into_data().convert::<f32>().try_into_vec::<f32>().unwrap()[0]);
    let meta = serde_json::json!({
        "config": serde_json::to_value(&cfg)?,
        "b": b, "t": t, "n": n, "seed": seed,
        "head_weights": HEADS, "value_weight": VALUE_WEIGHT, "win_open_weight": WIN_OPEN_WEIGHT,
        "valid": valid, "value_valid": value_valid,
        "lr": LR, "weight_decay": WEIGHT_DECAY, "adam": {"beta1": 0.9, "beta2": 0.999, "eps": 1e-5},
        "clip": CLIP, "clip_norm": norm, "win_open_params": win_open_params,
        "params": ids.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>(),
        "num_params": net.num_params(),
    });
    std::fs::write(out.join("meta.json"), serde_json::to_vec_pretty(&meta)?)?;
    println!("{} params -> {}", net.num_params(), out.display());
    Ok(())
}

fn main() -> Result<()> {
    let a = Args::parse();
    let full = |kind: CoreKind| {
        let c = UraniumNetConfig::new();
        let c = c.clone().with_core(c.core.clone().with_kind(kind));
        c.clone().with_entity(c.entity.clone().with_dropout(0.0))
    };
    let with_open = |c: UraniumNetConfig| c.clone().with_value(c.value.clone().with_win_open(true));
    if let Some(dir) = &a.export {
        return export(dir, &a.out);
    }
    match a.config.as_deref().unwrap_or_default() {
        "tiny-lstm" => dump::<LstmCore>(tiny_config(CoreKind::Lstm), &a.out, 2, 4, 12, a.seed),
        "tiny-gtrxl" => dump::<GtrxlCore>(with_open(tiny_config(CoreKind::Gtrxl)), &a.out, 2, 4, 12, a.seed),
        "full-lstm" => dump::<LstmCore>(full(CoreKind::Lstm), &a.out, 2, 8, 64, a.seed),
        "full-gtrxl" => dump::<GtrxlCore>(with_open(full(CoreKind::Gtrxl)), &a.out, 2, 8, 64, a.seed),
        other => anyhow::bail!("unknown config {other}"),
    }
}
