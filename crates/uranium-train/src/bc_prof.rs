//! `uranium-train bc-profile`: where a BC update's time goes, module by module, on real batches.
//!
//! Forward: `uranium_model::prof` marks (device synced at each). Backward: the same loss is
//! back-propagated with the gradient stopped at successively later points
//! (`UraniumNet::forward_seq_cut`); the differences are the modules' backward costs. Per head:
//! backward of one head's loss term with encoders and core cut. Lane groups are formed as in
//! training (`micro_lanes`).

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use burn::tensor::{Device, Tensor, TensorData};
use serde::Serialize;
use uranium_model::collate::{HostLabel, action_batch, obs_batch_padded, priv_batch};
use uranium_model::net::{Cut, SeqOutput};
use uranium_model::value::VALUE_WIN;
use uranium_model::{CoreInit, UraniumNet};

use crate::bc::{BcConfig, Feeder};
use crate::data::{ShardSource, TrainStep};
use crate::split::Selection;

#[derive(Serialize, Debug, Default)]
pub struct ProfReport {
    pub groups: usize,
    pub rows_per_group: usize,
    pub entities_mean: f64,
    pub entities_max: usize,
    /// Seconds per lane group (x lanes / micro_lanes = per update).
    pub groups_per_update: usize,
    pub collate: f64,
    pub forward: BTreeMap<String, f64>,
    pub forward_total: f64,
    pub loss: f64,
    pub backward_total: f64,
    /// Backward per module (differences of cut runs).
    pub backward: BTreeMap<String, f64>,
    /// Backward of one loss term with encoders and core cut.
    pub backward_head_only: BTreeMap<String, f64>,
}

fn loss_of<C: CoreInit>(out: &SeqOutput<C::State>, weights: &[f32], results: &[f32], valid: f32, cfg: &BcConfig, only: Option<usize>, d: &Device) -> Tensor<1> {
    let n = weights.len();
    let w = Tensor::<1>::from_data(TensorData::new(weights.to_vec(), [n]), d);
    let ce = |logp: Tensor<1>| (logp.neg() * w.clone()).sum() / valid as f64;
    let h = &cfg.heads;
    let lp = &out.heads.logp;
    let terms = [
        (lp.action.clone(), h.action),
        (lp.delay.clone(), h.delay),
        (lp.queued.clone(), h.queued),
        (lp.selected.clone(), h.selected),
        (lp.target_unit.clone(), h.target_unit),
        (lp.target_loc.clone(), h.target_loc),
    ];
    let target = Tensor::<1>::from_data(TensorData::new(results.to_vec(), [n]), d);
    let win = out.value.clone().narrow(1, VALUE_WIN, 1).reshape([n]);
    let value = ((win - target).powi_scalar(2) * w.clone()).sum() / valid as f64 * cfg.value.win as f64;
    match only {
        Some(6) => value,
        Some(i) => ce(terms[i].0.clone()) * terms[i].1 as f64,
        None => terms.into_iter().fold(value, |acc, (l, k)| acc + ce(l) * k as f64),
    }
}

pub fn profile<C: CoreInit>(dirs: &[PathBuf], cfg: &BcConfig, batches: usize, dev: Device) -> Result<ProfReport> {
    let ad = dev.autodiff();
    let cols = uranium_model::batch::ActionColumns::from_table(uranium_env::features::extract::action_table())?;
    let base = uranium_model::config::UraniumNetConfig::new();
    let net_cfg = base.clone().with_input(uranium_model::collate::input_dims()).with_core(base.core.clone().with_kind(C::KIND));
    let net = UraniumNet::<C>::new(&net_cfg, &cols, &ad);
    let mut feeder = Feeder::new(cfg);
    let mut src = ShardSource::from_dirs(dirs)?.select(Selection::S1).shuffled(cfg.data.seed);
    let t = cfg.seq_len;
    let g = match cfg.micro_lanes {
        0 => cfg.lanes,
        m => m.min(cfg.lanes),
    };
    let mut rep = ProfReport { rows_per_group: g * t, groups_per_update: cfg.lanes.div_ceil(g), ..Default::default() };
    let mut ents = Vec::new();
    let mut bwd: BTreeMap<&str, f64> = BTreeMap::new();
    let sync = |d: &Device| {
        let _ = d.sync();
    };
    let mut warm = true;
    for _ in 0..batches + 1 {
        let Some(batch) = feeder.next_batch(&mut src, None)? else { break };
        let (chunks, _, results) = batch.parts();
        for lanes in (0..chunks.len()).collect::<Vec<_>>().chunks(g) {
            let mut rows: Vec<&TrainStep> = Vec::new();
            let (mut weights, mut res) = (Vec::new(), Vec::new());
            for &l in lanes {
                let c = &chunks[l];
                for k in 0..t {
                    rows.push(&c[k.min(c.len() - 1)]);
                    weights.push(if k < c.len() { c[k].weight } else { 0.0 });
                    res.push(results[l]);
                }
            }
            let valid = weights.iter().sum::<f32>().max(1.0);
            let b = lanes.len();
            sync(&ad);
            let t0 = Instant::now();
            let obs: Vec<_> = rows.iter().map(|s| &s.obs).collect();
            let labels: Vec<&HostLabel> = rows.iter().map(|s| &s.label).collect();
            let privs: Vec<_> = rows.iter().map(|s| &s.privileged).collect();
            let ob = obs_batch_padded(&obs, cfg.pad_entities, &ad);
            let ab = action_batch(&labels, ob.entities(), net_cfg.heads.max_select, &ad);
            let pb = priv_batch(&privs, &ad);
            sync(&ad);
            let collate = t0.elapsed().as_secs_f64();
            // The first batch only warms kernels up (compilation, autotune).
            let run = |cut: Cut, only: Option<usize>, marks: bool| -> (f64, f64, Vec<(&'static str, f64)>) {
                if marks {
                    uranium_model::prof::enable(&ad);
                    uranium_model::prof::reset_clock();
                }
                let out = net.forward_seq_cut(&ob, &ab, Some(&pb), net.init_state(b, &ad), t, cut);
                let fwd = uranium_model::prof::take();
                sync(&ad);
                let t1 = Instant::now();
                let loss = loss_of::<C>(&out, &weights, &res, valid, cfg, only, &ad);
                sync(&ad);
                let l = t1.elapsed().as_secs_f64();
                let t2 = Instant::now();
                let _ = loss.backward();
                sync(&ad);
                (l, t2.elapsed().as_secs_f64(), fwd)
            };
            let (l, b_none, fwd) = run(Cut::None, None, true);
            let (_, b_ent, _) = run(Cut::Entity, None, false);
            let (_, b_sp, _) = run(Cut::Spatial, None, false);
            let (_, b_enc, _) = run(Cut::Encoders, None, false);
            let (_, b_core, _) = run(Cut::Core, None, false);
            let heads: Vec<f64> = (0..7).map(|i| run(Cut::Core, Some(i), false).1).collect();
            if warm {
                continue;
            }
            rep.groups += 1;
            ents.push(ob.entities());
            rep.collate += collate;
            rep.loss += l;
            for (p, s) in fwd {
                *rep.forward.entry(p.to_string()).or_default() += s;
            }
            for (k, v) in [("none", b_none), ("entity", b_ent), ("spatial", b_sp), ("encoders", b_enc), ("core", b_core)] {
                *bwd.entry(k).or_default() += v;
            }
            for (i, name) in ["action", "delay", "queued", "selected", "target_unit", "target_loc", "value"].iter().enumerate() {
                *rep.backward_head_only.entry(name.to_string()).or_default() += heads[i];
            }
        }
        warm = false;
    }
    let n = rep.groups.max(1) as f64;
    let avg = |m: &mut BTreeMap<String, f64>| m.values_mut().for_each(|v| *v /= n);
    avg(&mut rep.forward);
    avg(&mut rep.backward_head_only);
    rep.collate /= n;
    rep.loss /= n;
    rep.forward_total = rep.forward.values().sum();
    let b = |k: &str| bwd.get(k).copied().unwrap_or(0.0) / n;
    rep.backward_total = b("none");
    rep.backward.insert("entity_encoder".into(), b("none") - b("entity"));
    rep.backward.insert("spatial_encoder".into(), b("none") - b("spatial"));
    rep.backward.insert("encoders_all".into(), b("none") - b("encoders"));
    rep.backward.insert("core".into(), b("encoders") - b("core"));
    rep.backward.insert("heads_value".into(), b("core"));
    rep.entities_mean = ents.iter().sum::<usize>() as f64 / ents.len().max(1) as f64;
    rep.entities_max = ents.iter().copied().max().unwrap_or(0);
    Ok(rep)
}
