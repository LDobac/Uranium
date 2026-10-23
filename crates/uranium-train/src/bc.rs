//! Behaviour cloning trainer (041). B lanes each walk through games; every step trains on the
//! next T observations of every lane with the core state carried over between chunks of the same
//! game and reset when a lane starts a new game (041-R2, 032-R5). Lanes run forward/backward in
//! groups of `micro_lanes` and their gradients are summed before one optimizer step, so peak GPU
//! memory follows the group, not B (entity attention is N² per row).

use std::path::Path;

use anyhow::Result;
use burn::module::Module;
use burn::optim::{AdamWConfig, GradientsAccumulator, GradientsParams, ModuleOptimizer};
use burn::tensor::{Bool, Device, Tensor, TensorData};
use serde::Serialize;
use uranium_model::checkpoint::{self, CheckpointMeta};
use uranium_model::collate::{actions_from_host, obs_from_host, priv_from_host};
use uranium_model::config::UraniumNetConfig;
use uranium_model::value::VALUE_WIN;
use uranium_model::entity::EntityOut;
use uranium_model::{CoreInit, ObsBatch, UraniumNet};

use crate::ckpt::{self, TrainState};
use crate::data::{Rng, StepSource};
// 041 data side (config, fingerprint, schedule, lanes, batches, prefetch, run identity) moved to
// uranium-data (AMB-000-4); re-exported so `crate::bc::*` paths keep working.
pub use uranium_data::bc::*;

/// Loss parts of one update, for logging.
#[derive(Serialize, Debug, Clone, Copy, Default)]
pub struct LossParts {
    pub total: f32,
    pub action: f32,
    pub delay: f32,
    pub queued: f32,
    pub selected: f32,
    pub target_unit: f32,
    pub target_loc: f32,
    pub value: f32,
    /// 041-R19 win_open MSE (0 without the head or its weight).
    pub win_open: f32,
    /// Steps with a usable label in this batch.
    pub valid: f32,
}

impl LossParts {
    /// Adds the loss terms of another lane group (each already divided by the batch-wide valid).
    fn add(&mut self, o: &LossParts) {
        self.total += o.total;
        self.action += o.action;
        self.delay += o.delay;
        self.queued += o.queued;
        self.selected += o.selected;
        self.target_unit += o.target_unit;
        self.target_loc += o.target_loc;
        self.value += o.value;
        self.win_open += o.win_open;
    }
}

pub struct Trainer<C: CoreInit> {
    pub net: UraniumNet<C>,
    pub cfg: BcConfig,
    pub net_cfg: UraniumNetConfig,
    optim: ModuleOptimizer,
    device: Device,
    /// Data for [`Trainer::train_step`] (the prefetching path keeps its own, see [`Prefetch`]).
    pub feeder: Feeder,
    /// Core state per lane group (a group of `micro_lanes` always holds the same lanes).
    states: Vec<Option<C::State>>,
    pub step: usize,
    /// 1.0 for S1, `s2_lr_scale` for S2 (041-R4).
    pub lr_scale: f64,
    /// Data state after the last trained batch (saved in checkpoints, 061-R10).
    pub data: FeedState,
    /// 041-R17: rows per entity encoder chunk, 0 = the encoder inside the forward graph.
    entity_chunk: usize,
}

/// 041-R17: one lane group's entity encoder output, kept outside the graph, and how to
/// recompute it after the group's backward.
struct EntityRecompute {
    ob: ObsBatch,
    e: Tensor<3>,
    pool: Tensor<2>,
    /// (first row, rows, dropout seed) per chunk.
    chunks: Vec<(usize, usize, u64)>,
    /// Device seed after the recompute, so later draws do not repeat a chunk's.
    rest_seed: u64,
}

impl<C: CoreInit> Trainer<C> {
    /// `device` must be an autodiff device.
    pub fn new(net: UraniumNet<C>, net_cfg: UraniumNetConfig, cfg: BcConfig, device: Device) -> Self {
        // 041-R3 (G-D4): no Burn clipping (it is per tensor); `train_batch` clips the global norm.
        let optim = AdamWConfig::new().with_weight_decay(cfg.weight_decay).init();
        // 041-R19: the head comes from the model config (a new run: `[value] win_open` present;
        // a resume: the checkpoint), the loss weight from this config.
        match (net_cfg.value.win_open, cfg.value.win_open) {
            (false, Some(_)) => crate::elog!("[bc] WARNING: [value] win_open is set but the model has no win_open head (checkpoint from before 032-R18): no win_open loss"),
            (true, None) => crate::elog!("[bc] WARNING: the model has a win_open head but [value] win_open is not set: the head is not trained (041-R19)"),
            _ => {}
        }
        let feeder = Feeder::new(&cfg);
        let data = feeder.state(None);
        Self { net, cfg, net_cfg, optim, device, feeder, states: Vec::new(), step: 0, lr_scale: 1.0, data, entity_chunk: 0 }
    }

    /// 041-R17: run the entity encoder in chunks of `rows` rows and recompute it for the
    /// backward pass (0 = off). Changes device memory, not the gradient.
    pub fn set_entity_chunk(&mut self, rows: usize) {
        self.entity_chunk = rows;
    }

    pub fn games(&self) -> usize {
        self.data.games
    }

    pub fn invalid_labels(&self) -> usize {
        self.data.invalid_labels
    }

    /// Lanes per forward/backward pass.
    fn group_size(&self) -> usize {
        match self.cfg.micro_lanes {
            0 => self.cfg.lanes,
            m => m.min(self.cfg.lanes),
        }
    }

    /// One gradient update over B lanes x T steps from `src`. `None` when the source is exhausted.
    pub fn train_step(&mut self, src: &mut dyn StepSource) -> Result<Option<LossParts>> {
        match self.feeder.next_batch(src, None)? {
            Some(b) => self.train_batch(b).map(Some),
            None => Ok(None),
        }
    }

    /// One gradient update over a prepared batch.
    pub fn train_batch(&mut self, batch: Batch) -> Result<LossParts> {
        let mut prof = Prof::new(&self.device);
        // Losses are divided by the usable steps of the whole batch, so the summed group
        // gradients equal the gradient of one pass over all lanes.
        let valid = batch.chunks.iter().flat_map(|c| c.iter().map(|s| s.weight)).sum::<f32>().max(1.0);
        // 041-R7 / A-3: the value loss is divided by the usable steps of decided games only, so
        // its scale does not follow the share of undecided games in the batch.
        let value_valid = batch.chunks.iter().zip(&batch.value_weights).map(|(c, vw)| c.iter().map(|s| s.weight * vw).sum::<f32>()).sum::<f32>().max(1.0);
        let lanes = batch.chunks.len();
        let g = self.group_size();
        let groups = lanes.div_ceil(g);
        self.states.resize_with(groups, || None);
        let mut acc = GradientsAccumulator::<UraniumNet<C>>::new();
        let mut report = LossParts { valid, ..Default::default() };
        for i in 0..groups {
            let (loss, parts, recompute) = self.group_loss(&batch, i, i * g..((i + 1) * g).min(lanes), valid, value_valid, &mut prof);
            report.add(&parts);
            let mut grads = loss.backward();
            let entity = recompute.map(|r| {
                let ge = r.e.grad_remove(&mut grads).expect("041-R17: entity output has no gradient");
                let gp = r.pool.grad_remove(&mut grads).expect("041-R17: entity pool has no gradient");
                (r, ge, gp)
            });
            acc.accumulate(&self.net, GradientsParams::from_grads(grads, &self.net));
            prof.mark("backward");
            // 041-R17: the entity encoder's share of the gradient, one row chunk at a time: the
            // same chunk with the same dropout seed gives the forward's output again, and
            // sum(out * d loss/d out) carries d loss/d out back into its parameters.
            if let Some((r, ge, gp)) = entity {
                for &(start, len, seed) in &r.chunks {
                    self.device.seed(seed);
                    let o = self.net.entity(&r.ob.entity_rows(start, len));
                    let s = (o.e * ge.clone().narrow(0, start, len)).sum() + (o.pool * gp.clone().narrow(0, start, len)).sum();
                    acc.accumulate(&self.net, GradientsParams::from_grads(s.backward(), &self.net));
                }
                self.device.seed(r.rest_seed);
                prof.mark("entity_recompute");
            }
            prof.mark("accumulate");
        }
        // 041-R3 (G-D4): one global L2-norm clip over the summed gradients of all lane groups.
        // 041-R19: the win_open head is clipped on its own, the rest updates as without it.
        let (grads, _norm) = crate::clip::clip_global_norm_except(&self.net, acc.grads(), self.cfg.grad_clip, &self.net.win_open_params());
        prof.mark("clip");
        let lr = self.cfg.lr_at(self.step, self.lr_scale);
        self.net = self.optim.step(lr, self.net.clone(), grads);
        prof.mark("optimizer");
        prof.report(self.step);
        self.step += 1;
        self.data = batch.state;
        Ok(report)
    }

    /// Forward pass and weighted loss of lane group `group` (lanes `lanes`), divided by the
    /// batch-wide `valid` (policy heads) and `value_valid` (value head, decided games only).
    /// Carries the group's core state to its next chunk. With the entity recompute on (041-R17)
    /// the entity encoder runs outside the graph and the caller finishes its backward.
    fn group_loss(&mut self, batch: &Batch, group: usize, lanes: std::ops::Range<usize>, valid: f32, value_valid: f32, prof: &mut Prof) -> (Tensor<1>, LossParts, Option<EntityRecompute>) {
        let (b, t) = (lanes.len(), self.cfg.seq_len);
        // Rows in b-major order, a game shorter than the chunk padded with its last step at
        // weight 0, a packed batch's planes restored for this group only (041-R18), value-loss
        // weight = label weight x the game's value weight (041-R7 / A-3): `Batch::group`, the
        // same host collate the Python trainer uses (AMB-000-4).
        let hg = batch.group(lanes, t, self.cfg.pad_entities, self.net_cfg.heads.max_select);
        let d = &self.device;
        let ob = obs_from_host(&hg.obs, d);
        let ab = actions_from_host(&hg.labels, d);
        let pb = priv_from_host(&hg.privileged, d);
        let HostGroup { weights, results, value_w, keep, .. } = hg;
        prof.mark("collate");

        let keep_t = Tensor::<1, Bool>::from_data(TensorData::new(keep, [b]), d);
        let s0 = match self.states[group].take() {
            Some(s) => C::mask_state(s, keep_t),
            None => self.net.init_state(b, d),
        };
        let mut recompute = None;
        let out = if self.entity_chunk == 0 {
            self.net.forward_seq(&ob, &ab, Some(&pb), s0, t)
        } else {
            // 041-R17: the entity encoder in row chunks, each with its own dropout seed and no
            // graph recorded, so no chunk's activations outlive it.
            let mut rng = Rng::new(self.cfg.data.seed ^ ((self.step as u64) << 24) ^ ((group as u64) << 4) ^ 0x0E17);
            let chunks: Vec<(usize, usize, u64)> = (0..b * t).step_by(self.entity_chunk).map(|s| (s, self.entity_chunk.min(b * t - s), rng.next())).collect();
            let (mut es, mut pools) = (Vec::with_capacity(chunks.len()), Vec::with_capacity(chunks.len()));
            for &(start, len, seed) in &chunks {
                d.seed(seed);
                let o = self.net.entity_untracked(&ob.entity_rows(start, len));
                es.push(o.e.detach());
                pools.push(o.pool.detach());
            }
            let e = Tensor::cat(es, 0).require_grad();
            let pool = Tensor::cat(pools, 0).require_grad();
            d.seed(rng.next());
            let ent = EntityOut { e: e.clone(), pool: pool.clone() };
            let out = self.net.forward_seq_with_entity(&ob, &ab, Some(&pb), s0, t, ent);
            recompute = Some(EntityRecompute { ob: ob.clone(), e, pool, chunks, rest_seed: rng.next() });
            out
        };
        // Detached when stored so a waiting group's state does not keep its forward graph alive
        // while the other groups run (the truncation point of 041-R2 is the same).
        self.states[group] = Some(C::detach_state(out.state));
        prof.mark("forward");

        let w = Tensor::<1>::from_data(TensorData::new(weights, [b * t]), d);
        let ce = |logp: Tensor<1>| (logp.neg() * w.clone()).sum() / valid as f64;
        let h = &self.cfg.heads;
        let lp = &out.heads.logp;
        let parts = [
            (ce(lp.action.clone()), h.action),
            (ce(lp.delay.clone()), h.delay),
            (ce(lp.queued.clone()), h.queued),
            (ce(lp.selected.clone()), h.selected),
            (ce(lp.target_unit.clone()), h.target_unit),
            (ce(lp.target_loc.clone()), h.target_loc),
        ];
        // 041-R7: win value head regresses the replay result (privileged inputs), MSE; games
        // with an undecided result (A-3) have value weight 0.
        let target = Tensor::<1>::from_data(TensorData::new(results, [b * t]), d);
        let vw = Tensor::<1>::from_data(TensorData::new(value_w, [b * t]), d);
        let win = out.value.narrow(1, VALUE_WIN, 1).reshape([b * t]);
        let value_loss = ((win - target.clone()).powi_scalar(2) * vw.clone()).sum() / value_valid as f64;
        let mut loss = value_loss.clone() * self.cfg.value.win as f64;
        for (l, k) in &parts {
            loss = loss + l.clone() * *k as f64;
        }
        // 041-R19: win_open regresses the same target with the same weights (detached input,
        // 032-R18), added last so the other terms sum as before.
        let open_loss = match (out.win_open, self.cfg.value.win_open) {
            (Some(open), Some(k)) => {
                let l = ((open - target).powi_scalar(2) * vw).sum() / value_valid as f64;
                loss = loss + l.clone() * k as f64;
                Some(l)
            }
            _ => None,
        };
        prof.mark("loss");
        let scalar = |x: &Tensor<1>| x.clone().into_data().convert::<f32>().try_to_vec::<f32>().unwrap()[0];
        let report = LossParts {
            total: scalar(&loss),
            action: scalar(&parts[0].0),
            delay: scalar(&parts[1].0),
            queued: scalar(&parts[2].0),
            selected: scalar(&parts[3].0),
            target_unit: scalar(&parts[4].0),
            target_loc: scalar(&parts[5].0),
            value: scalar(&value_loss),
            win_open: open_loss.as_ref().map_or(0.0, scalar),
            valid: 0.0,
        };
        prof.mark("loss_read");
        (loss, report, recompute)
    }

    /// 032-R6 / 041-R10: weights + config + schema versions + hyper-parameters; 061-R10: optimizer
    /// state, step and RNG so a run can resume on any host.
    pub fn save(&self, dir: &Path) -> Result<()> {
        let mut meta = CheckpointMeta::new(&self.net_cfg, uranium_env::features::FEATURE_SCHEMA_VERSION, uranium_env::actions::ACTION_SCHEMA_VERSION)?;
        meta.train = serde_json::json!({ "bc": self.cfg, "step": self.step, "lr_scale": self.lr_scale, "games": self.data.games, "invalid_labels": self.data.invalid_labels, "burn": uranium_model::runtime::settings_json() });
        checkpoint::save(dir, &self.net.valid(), &meta)?;
        ckpt::save_optimizer(dir, OPTIM_NAME, &self.optim)?;
        let state = TrainState {
            kind: "bc".into(),
            step: self.step as u64,
            update: self.step as u64,
            policy_version: self.step as u64,
            rng: vec![self.data.rng],
            extra: serde_json::json!({ "lr_scale": self.lr_scale, "games": self.data.games, "invalid_labels": self.data.invalid_labels, "data": self.data }),
        };
        ckpt::save_state(dir, &state)
    }

    /// 061-R10 / 042-R1: restores optimizer state, counters and RNG from `dir`. Load the weights
    /// from the same checkpoint first (`uranium_model::checkpoint::load` on the autodiff device).
    /// Returns false for a checkpoint without training state (written before 061-R10 was
    /// implemented): the weights are used as they are and the optimizer starts fresh.
    /// The data position is in [`Trainer::data`]; [`Feeder::restore`] puts a source back there
    /// (lanes here start empty: [`Trainer::train_step`] then reads the source from where it is).
    pub fn resume(&mut self, dir: &Path) -> Result<bool> {
        let Some(state) = ckpt::load_state(dir)? else { return Ok(false) };
        anyhow::ensure!(state.kind == "bc", "{} is a {} checkpoint, not bc", dir.display(), state.kind);
        let optim = std::mem::replace(&mut self.optim, AdamWConfig::new().init());
        self.optim = ckpt::load_optimizer(dir, OPTIM_NAME, optim)?;
        self.step = state.step as usize;
        self.lr_scale = state.extra["lr_scale"].as_f64().unwrap_or(self.lr_scale);
        self.data = match state.extra.get("data") {
            Some(d) => serde_json::from_value(d.clone())?,
            // Checkpoints before the data position was stored: counters and RNG only.
            None => FeedState {
                lanes: Vec::new(),
                source: None,
                rng: *state.rng.first().ok_or_else(|| anyhow::anyhow!("no rng state in {}", dir.display()))?,
                games: state.extra["games"].as_u64().unwrap_or(0) as usize,
                invalid_labels: state.extra["invalid_labels"].as_u64().unwrap_or(0) as usize,
                files: String::new(),
                no_result_games: 0,
            },
        };
        self.feeder.restore_counters(&self.data);
        self.states.clear();
        Ok(true)
    }

    /// Drops the games in progress and the core states: the next update starts every lane on a
    /// new game from the source.
    pub fn reset_lanes(&mut self) {
        self.feeder.reset_lanes();
        self.states.clear();
    }
}

/// `URANIUM_BC_PROFILE=1`: wall time per phase of an update (device synced at every mark, so
/// GPU work is charged to the phase that queued it). One JSON line per update on stderr.
struct Prof {
    device: Option<Device>,
    last: std::time::Instant,
    spent: Vec<(&'static str, f64)>,
}

impl Prof {
    fn new(device: &Device) -> Self {
        let on = std::env::var("URANIUM_BC_PROFILE").is_ok_and(|v| v == "1");
        Self { device: on.then(|| device.clone()), last: std::time::Instant::now(), spent: Vec::new() }
    }

    fn mark(&mut self, phase: &'static str) {
        let Some(d) = &self.device else { return };
        let _ = d.sync();
        let now = std::time::Instant::now();
        let s = (now - self.last).as_secs_f64();
        self.last = now;
        match self.spent.iter_mut().find(|(p, _)| *p == phase) {
            Some(e) => e.1 += s,
            None => self.spent.push((phase, s)),
        }
    }

    fn report(&self, step: usize) {
        if self.device.is_none() {
            return;
        }
        let m: serde_json::Map<String, serde_json::Value> = self.spent.iter().map(|(p, s)| (p.to_string(), serde_json::json!((s * 1000.0).round() / 1000.0))).collect();
        crate::elog!("{}", serde_json::json!({ "profile_step": step, "secs": m }));
    }
}

/// Optimizer name in a BC checkpoint (`optim-main.bpk`).
const OPTIM_NAME: &str = "main";
