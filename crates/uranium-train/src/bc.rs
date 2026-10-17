//! Behaviour cloning trainer (041). B lanes each walk through games; every step trains on the
//! next T observations of every lane with the core state carried over between chunks of the same
//! game and reset when a lane starts a new game (041-R2, 032-R5). Lanes run forward/backward in
//! groups of `micro_lanes` and their gradients are summed before one optimizer step, so peak GPU
//! memory follows the group, not B (entity attention is N² per row).

use std::path::Path;

use anyhow::Result;
use burn::module::Module;
use burn::optim::{AdamWConfig, GradientsAccumulator, GradientsParams, ModuleOptimizer};
use burn::optim::grad_clipping::GradientClippingConfig;
use burn::tensor::{Bool, Device, Tensor, TensorData};
use serde::{Deserialize, Serialize};
use uranium_env::features::augment::Sym;
use uranium_model::checkpoint::{self, CheckpointMeta};
use uranium_model::collate::{HostLabel, action_batch, obs_batch_padded, priv_batch};
use uranium_model::config::UraniumNetConfig;
use uranium_model::value::VALUE_WIN;
use uranium_model::{CoreInit, UraniumNet};

use crate::ckpt::{self, TrainState};
use crate::data::{GameStream, Rng, StepSource, TrainStep};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct HeadWeights {
    pub action: f32,
    pub delay: f32,
    pub queued: f32,
    pub selected: f32,
    pub target_unit: f32,
    pub target_loc: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ValueWeights {
    pub win: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct DataConfig {
    pub z_drop: f64,
    pub augment: bool,
    pub seed: u64,
    pub checkpoint_every: usize,
}

/// `config/bc.toml` (041-R3, R10).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct BcConfig {
    pub lanes: usize,
    pub seq_len: usize,
    /// Lanes per forward/backward pass; gradients of all groups are accumulated into one update
    /// (same result as one pass over all lanes). 0 = all lanes at once.
    #[serde(default)]
    pub micro_lanes: usize,
    /// Recompute memory-bound activations during backward (Burn balanced checkpointing).
    #[serde(default)]
    pub grad_checkpointing: bool,
    /// Round the padded entity count N up to a multiple of this (0 or 1 = batch max N only).
    /// Fewer distinct tensor shapes per run (results are unchanged: padding is masked).
    #[serde(default)]
    pub pad_entities: usize,
    pub lr: f64,
    pub min_lr: f64,
    pub warmup_steps: usize,
    pub total_steps: usize,
    pub weight_decay: f32,
    pub grad_clip: f32,
    pub s2_lr_scale: f64,
    pub heads: HeadWeights,
    pub value: ValueWeights,
    pub data: DataConfig,
}

impl BcConfig {
    pub fn load(path: &Path) -> Result<Self> {
        Ok(toml::from_str(&std::fs::read_to_string(path)?)?)
    }

    /// Linear warmup then cosine decay to `min_lr` (041-R3); `scale` = 1 (S1) or `s2_lr_scale`.
    pub fn lr_at(&self, step: usize, scale: f64) -> f64 {
        let lr = if step < self.warmup_steps {
            self.lr * (step + 1) as f64 / self.warmup_steps as f64
        } else {
            let t = ((step - self.warmup_steps) as f64 / (self.total_steps.saturating_sub(self.warmup_steps)).max(1) as f64).min(1.0);
            self.min_lr + 0.5 * (self.lr - self.min_lr) * (1.0 + (std::f64::consts::PI * t).cos())
        };
        lr * scale
    }
}

/// One lane: the current game and the position inside it.
struct Lane {
    game: Option<GameStream>,
    /// Steps of the current BPTT window (041-R2).
    chunk: Vec<TrainStep>,
    fresh: bool,
}

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
    }
}

pub struct Trainer<C: CoreInit> {
    pub net: UraniumNet<C>,
    pub cfg: BcConfig,
    pub net_cfg: UraniumNetConfig,
    optim: ModuleOptimizer,
    device: Device,
    lanes: Vec<Lane>,
    /// Core state per lane group (a group of `micro_lanes` always holds the same lanes).
    states: Vec<Option<C::State>>,
    rng: Rng,
    pub step: usize,
    /// 1.0 for S1, `s2_lr_scale` for S2 (041-R4).
    pub lr_scale: f64,
    pub invalid_labels: usize,
    pub games: usize,
}

impl<C: CoreInit> Trainer<C> {
    /// `device` must be an autodiff device.
    pub fn new(net: UraniumNet<C>, net_cfg: UraniumNetConfig, cfg: BcConfig, device: Device) -> Self {
        let optim = AdamWConfig::new()
            .with_weight_decay(cfg.weight_decay)
            .with_grad_clipping(Some(GradientClippingConfig::Norm(cfg.grad_clip)))
            .init();
        let lanes = (0..cfg.lanes).map(|_| Lane { game: None, chunk: Vec::new(), fresh: true }).collect();
        let rng = Rng::new(cfg.data.seed);
        Self { net, cfg, net_cfg, optim, device, lanes, states: Vec::new(), rng, step: 0, lr_scale: 1.0, invalid_labels: 0, games: 0 }
    }

    /// Fills lanes whose game is used up; returns false when the source ran dry for some lane.
    fn refill(&mut self, src: &mut dyn StepSource) -> Result<bool> {

        for i in 0..self.lanes.len() {
            if self.lanes[i].game.as_ref().is_some_and(|g| g.remaining() > 0) {
                continue;
            }
            let game = loop {
                let Some(game) = src.next_game()? else { return Ok(false) };
                if !game.steps.is_empty() {
                    break game;
                }
            };
            let z_present = self.rng.unit() >= self.cfg.data.z_drop;
            let sym = if self.cfg.data.augment { Sym::all()[(self.rng.next() % 8) as usize] } else { Sym::IDENTITY };
            self.games += 1;
            self.lanes[i] = Lane { game: Some(GameStream::new(game, z_present, sym)?), chunk: Vec::new(), fresh: true };
        }
        Ok(true)
    }

    /// Lanes per forward/backward pass.
    fn group_size(&self) -> usize {
        match self.cfg.micro_lanes {
            0 => self.lanes.len(),
            m => m.min(self.lanes.len()),
        }
    }

    /// One gradient update over B lanes x T steps. `None` when the source is exhausted.
    pub fn train_step(&mut self, src: &mut dyn StepSource) -> Result<Option<LossParts>> {
        if !self.refill(src)? {
            return Ok(None);
        }
        let t = self.cfg.seq_len;
        for lane in &mut self.lanes {
            let (chunk, invalid) = lane.game.as_mut().unwrap().next_chunk(t);
            self.invalid_labels += invalid;
            lane.chunk = chunk;
        }
        // Losses are divided by the usable steps of the whole batch, so the summed group
        // gradients equal the gradient of one pass over all lanes.
        let valid = self.lanes.iter().flat_map(|l| l.chunk.iter().map(|s| s.weight)).sum::<f32>().max(1.0);
        let g = self.group_size();
        let groups = self.lanes.len().div_ceil(g);
        self.states.resize_with(groups, || None);
        let mut acc = GradientsAccumulator::<UraniumNet<C>>::new();
        let mut report = LossParts { valid, ..Default::default() };
        for i in 0..groups {
            let (loss, parts) = self.group_loss(i, i * g..((i + 1) * g).min(self.lanes.len()), valid);
            report.add(&parts);
            let grads = GradientsParams::from_grads(loss.backward(), &self.net);
            acc.accumulate(&self.net, grads);
        }
        let lr = self.cfg.lr_at(self.step, self.lr_scale);
        self.net = self.optim.step(lr, self.net.clone(), acc.grads());
        self.step += 1;
        // Lanes that finished their game start fresh next time (state reset on refill).
        for lane in &mut self.lanes {
            lane.fresh = false;
            lane.chunk.clear();
            if lane.game.as_ref().is_some_and(|g| g.remaining() == 0) {
                lane.game = None;
            }
        }
        Ok(Some(report))
    }

    /// Forward pass and weighted loss of lane group `group` (lanes `lanes`), divided by the
    /// batch-wide `valid`. Carries the group's core state to its next chunk.
    fn group_loss(&mut self, group: usize, lanes: std::ops::Range<usize>, valid: f32) -> (Tensor<1>, LossParts) {
        let (b, t) = (lanes.len(), self.cfg.seq_len);
        // Gather rows in b-major order; a game shorter than the chunk is padded with its last
        // step at weight 0.
        let mut rows: Vec<&TrainStep> = Vec::with_capacity(b * t);
        let mut weights = Vec::with_capacity(b * t);
        let mut results = Vec::with_capacity(b * t);
        let mut keep = Vec::with_capacity(b);
        for lane in &self.lanes[lanes] {
            let result = lane.game.as_ref().unwrap().result;
            keep.push(!lane.fresh);
            for k in 0..t {
                let s = &lane.chunk[k.min(lane.chunk.len() - 1)];
                rows.push(s);
                weights.push(if k < lane.chunk.len() { s.weight } else { 0.0 });
                results.push(result);
            }
        }
        let obs: Vec<_> = rows.iter().map(|s| &s.obs).collect();
        let labels: Vec<&HostLabel> = rows.iter().map(|s| &s.label).collect();
        let privs: Vec<_> = rows.iter().map(|s| &s.privileged).collect();
        let d = &self.device;
        let ob = obs_batch_padded(&obs, self.cfg.pad_entities, d);
        let n = ob.entities();
        let ab = action_batch(&labels, n, self.net_cfg.heads.max_select, d);
        let pb = priv_batch(&privs, d);

        let keep_t = Tensor::<1, Bool>::from_data(TensorData::new(keep, [b]), d);
        let s0 = match self.states[group].take() {
            Some(s) => C::mask_state(s, keep_t),
            None => self.net.init_state(b, d),
        };
        let out = self.net.forward_seq(&ob, &ab, Some(&pb), s0, t);
        // Detached when stored so a waiting group's state does not keep its forward graph alive
        // while the other groups run (the truncation point of 041-R2 is the same).
        self.states[group] = Some(C::detach_state(out.state));

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
        // 041-R7: win value head regresses the replay result (privileged inputs), MSE.
        let target = Tensor::<1>::from_data(TensorData::new(results, [b * t]), d);
        let win = out.value.narrow(1, VALUE_WIN, 1).reshape([b * t]);
        let value_loss = ((win - target).powi_scalar(2) * w.clone()).sum() / valid as f64;
        let mut loss = value_loss.clone() * self.cfg.value.win as f64;
        for (l, k) in &parts {
            loss = loss + l.clone() * *k as f64;
        }
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
            valid: 0.0,
        };
        (loss, report)
    }

    /// 032-R6 / 041-R10: weights + config + schema versions + hyper-parameters; 061-R10: optimizer
    /// state, step and RNG so a run can resume on any host.
    pub fn save(&self, dir: &Path) -> Result<()> {
        let mut meta = CheckpointMeta::new(&self.net_cfg, uranium_env::features::FEATURE_SCHEMA_VERSION, uranium_env::actions::ACTION_SCHEMA_VERSION)?;
        meta.train = serde_json::json!({ "bc": self.cfg, "step": self.step, "lr_scale": self.lr_scale, "games": self.games, "invalid_labels": self.invalid_labels });
        checkpoint::save(dir, &self.net.valid(), &meta)?;
        ckpt::save_optimizer(dir, OPTIM_NAME, &self.optim)?;
        let state = TrainState {
            kind: "bc".into(),
            step: self.step as u64,
            update: self.step as u64,
            policy_version: self.step as u64,
            rng: vec![self.rng.state()],
            extra: serde_json::json!({ "lr_scale": self.lr_scale, "games": self.games, "invalid_labels": self.invalid_labels }),
        };
        ckpt::save_state(dir, &state)
    }

    /// 061-R10 / 042-R1: restores optimizer state, counters and RNG from `dir`. Load the weights
    /// from the same checkpoint first (`uranium_model::checkpoint::load` on the autodiff device).
    /// Returns false for a checkpoint without training state (written before 061-R10 was
    /// implemented): the weights are used as they are and the optimizer starts fresh.
    /// The data position is not stored: lanes start over on the source given next.
    pub fn resume(&mut self, dir: &Path) -> Result<bool> {
        let Some(state) = ckpt::load_state(dir)? else { return Ok(false) };
        anyhow::ensure!(state.kind == "bc", "{} is a {} checkpoint, not bc", dir.display(), state.kind);
        let optim = std::mem::replace(&mut self.optim, AdamWConfig::new().init());
        self.optim = ckpt::load_optimizer(dir, OPTIM_NAME, optim)?;
        self.step = state.step as usize;
        self.rng = Rng::from_state(*state.rng.first().ok_or_else(|| anyhow::anyhow!("no rng state in {}", dir.display()))?);
        self.lr_scale = state.extra["lr_scale"].as_f64().unwrap_or(self.lr_scale);
        self.games = state.extra["games"].as_u64().unwrap_or(0) as usize;
        self.invalid_labels = state.extra["invalid_labels"].as_u64().unwrap_or(0) as usize;
        self.reset_lanes();
        Ok(true)
    }

    /// Drops the games in progress and the core states: the next update starts every lane on a
    /// new game from the source (what a resumed trainer does).
    pub fn reset_lanes(&mut self) {
        for l in &mut self.lanes {
            *l = Lane { game: None, chunk: Vec::new(), fresh: true };
        }
        self.states.clear();
    }
}

/// Optimizer name in a BC checkpoint (`optim-main.bpk`).
const OPTIM_NAME: &str = "main";
