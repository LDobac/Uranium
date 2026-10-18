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
use crate::data::{GameId, GameStream, Rng, ShardSource, SourcePos, StepSource, TrainStep};

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
    /// Checkpoint at least this often in wall time (0 = updates only).
    #[serde(default)]
    pub checkpoint_minutes: u64,
}

/// Updates of one epoch per stage (041-R13, 2026-10-18: S1, S2 and ablation (A) train one epoch
/// each and the cosine schedule ends there). `uranium-train bc-epoch` counts them.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct StageSteps {
    pub s1: usize,
    pub s2: usize,
    pub ablation_a: usize,
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
    /// Schedule length (041-R3). The `bc` command sets it from `steps` for the stage it trains.
    pub total_steps: usize,
    #[serde(default)]
    pub steps: StageSteps,
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
    /// Where the game came from (resume, 061-R10).
    id: Option<GameId>,
    /// The game started at the next batch (core state reset, 041-R2).
    fresh: bool,
}

impl Lane {
    fn empty() -> Self {
        Self { game: None, id: None, fresh: true }
    }
}

/// A lane in a data snapshot: its game, how far it got and the choices made for it.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct LaneState {
    pub id: GameId,
    pub pos: usize,
    pub z_present: bool,
    /// Index into `Sym::all()`.
    pub sym: u8,
}

/// Data position after a batch (061-R10): restoring it continues with the batch that followed,
/// without training the games already seen again. Lanes resume mid-game with a fresh core state.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct FeedState {
    pub lanes: Vec<Option<LaneState>>,
    pub source: Option<SourcePos>,
    pub rng: u64,
    pub games: usize,
    pub invalid_labels: usize,
    /// [`ShardSource::fingerprint`] the position refers to (empty: not a shard source).
    #[serde(default)]
    pub files: String,
}

/// The CPU half of an update: B lanes x T steps of features and labels (041-R2), plus the data
/// state after it.
pub struct Batch {
    chunks: Vec<Vec<TrainStep>>,
    fresh: Vec<bool>,
    results: Vec<f32>,
    pub state: FeedState,
}

impl Batch {
    /// Per lane: the T steps (fewer at a game's end), whether the game started here, its result.
    pub fn parts(&self) -> (&[Vec<TrainStep>], &[bool], &[f32]) {
        (&self.chunks, &self.fresh, &self.results)
    }
}

/// Walks B lanes through the games of a source (041-R2) and turns the next T steps of every lane
/// into a [`Batch`]. Owns the data RNG (z drop 041-R8, symmetry 040-R7). Runs on the trainer's
/// thread ([`Trainer::train_step`]) or ahead of it on its own thread ([`Prefetch`]); the batches
/// are the same.
pub struct Feeder {
    lanes: Vec<Lane>,
    rng: Rng,
    seq_len: usize,
    z_drop: f64,
    augment: bool,
    games: usize,
    invalid_labels: usize,
    /// Fingerprint of the shard source (set on the first batch from it).
    files: String,
    /// Evaluation: when the source runs dry, idle lanes give empty chunks until every lane is
    /// done (training instead ends the epoch at the first lane it cannot fill).
    pub drain: bool,
}

impl Feeder {
    pub fn new(cfg: &BcConfig) -> Self {
        Self {
            lanes: (0..cfg.lanes).map(|_| Lane::empty()).collect(),
            rng: Rng::new(cfg.data.seed),
            seq_len: cfg.seq_len,
            z_drop: cfg.data.z_drop,
            augment: cfg.data.augment,
            games: 0,
            invalid_labels: 0,
            files: String::new(),
            drain: false,
        }
    }

    /// Fills lanes whose game is used up; returns false when the source ran dry for some lane.
    fn refill(&mut self, src: &mut dyn StepSource) -> Result<bool> {
        for i in 0..self.lanes.len() {
            if self.lanes[i].game.as_ref().is_some_and(|g| g.remaining() > 0) {
                continue;
            }
            let game = loop {
                let Some(game) = src.next_game()? else {
                    if self.drain {
                        return Ok(self.lanes.iter().any(|l| l.game.is_some()));
                    }
                    return Ok(false);
                };
                if !game.steps.is_empty() {
                    break game;
                }
            };
            let z_present = self.rng.unit() >= self.z_drop;
            let sym = if self.augment { Sym::all()[(self.rng.next() % 8) as usize] } else { Sym::IDENTITY };
            self.games += 1;
            self.lanes[i] = Lane { game: Some(GameStream::new(game, z_present, sym)?), id: src.last_id(), fresh: true };
        }
        Ok(true)
    }

    /// The next batch, or `None` when the source is exhausted (end of the epoch).
    pub fn next_batch(&mut self, src: &mut dyn StepSource, pos: Option<SourcePos>) -> Result<Option<Batch>> {
        if !self.refill(src)? {
            return Ok(None);
        }
        let mut chunks = Vec::with_capacity(self.lanes.len());
        let mut fresh = Vec::with_capacity(self.lanes.len());
        let mut results = Vec::with_capacity(self.lanes.len());
        for lane in &mut self.lanes {
            let Some(g) = lane.game.as_mut() else {
                // Drained lane (evaluation only).
                chunks.push(Vec::new());
                fresh.push(true);
                results.push(0.0);
                continue;
            };
            let (chunk, invalid) = g.next_chunk(self.seq_len);
            self.invalid_labels += invalid;
            chunks.push(chunk);
            fresh.push(lane.fresh);
            results.push(g.result);
            lane.fresh = false;
            if g.remaining() == 0 {
                lane.game = None;
                lane.id = None;
            }
        }
        Ok(Some(Batch { chunks, fresh, results, state: self.state(pos) }))
    }

    /// [`Feeder::next_batch`] from a shard source, with its read position in the batch's state.
    pub fn next_batch_from(&mut self, src: &mut ShardSource) -> Result<Option<Batch>> {
        if self.files.is_empty() {
            self.files = src.fingerprint();
        }
        let b = self.next_batch(src, None)?;
        Ok(b.map(|mut b| {
            b.state.source = Some(src.position());
            b.state.files = self.files.clone();
            b
        }))
    }

    /// The data state now; `pos` = the source's read position (resumable sources only).
    pub fn state(&self, pos: Option<SourcePos>) -> FeedState {
        let lanes = self
            .lanes
            .iter()
            .map(|l| {
                let (g, id) = (l.game.as_ref()?, l.id?);
                Some(LaneState { id, pos: g.pos(), z_present: g.z_present(), sym: Sym::all().iter().position(|s| *s == g.sym()).unwrap_or(0) as u8 })
            })
            .collect();
        FeedState { lanes, source: pos, rng: self.rng.state(), games: self.games, invalid_labels: self.invalid_labels, files: self.files.clone() }
    }

    /// Back to `st` (061-R10 resume): counters and RNG, lanes re-read from `src` and advanced to
    /// their recorded step, and `src` sought to the recorded position.
    pub fn restore(&mut self, st: &FeedState, src: &mut ShardSource) -> Result<()> {
        if !st.files.is_empty() {
            let here = src.fingerprint();
            anyhow::ensure!(
                st.files == here,
                "the checkpoint's data position is for another shard list ({} here {}): pass the same datasets, selection and seed (061-R10)",
                st.files,
                here
            );
            self.files = here;
        }
        self.rng = Rng::from_state(st.rng);
        self.games = st.games;
        self.invalid_labels = st.invalid_labels;
        for (lane, saved) in self.lanes.iter_mut().zip(st.lanes.iter().chain(std::iter::repeat(&None))) {
            *lane = Lane::empty();
            let Some(s) = saved else { continue };
            let mut g = GameStream::new(src.game(s.id)?, s.z_present, Sym::all()[s.sym as usize % 8])?;
            g.advance_to(s.pos);
            if g.remaining() > 0 {
                *lane = Lane { game: Some(g), id: Some(s.id), fresh: true };
            }
        }
        if let Some(p) = st.source {
            src.seek(p)?;
        }
        Ok(())
    }

    /// Counters and RNG only, lanes empty (a checkpoint without a data position).
    pub fn restore_counters(&mut self, st: &FeedState) {
        self.rng = Rng::from_state(st.rng);
        self.games = st.games;
        self.invalid_labels = st.invalid_labels;
        self.reset_lanes();
    }

    pub fn reset_lanes(&mut self) {
        for l in &mut self.lanes {
            *l = Lane::empty();
        }
    }
}

/// [`Feeder`] on its own thread, `depth` batches ahead of the trainer (041 throughput: feature
/// extraction and shard reads overlap the GPU update; the batches are the ones
/// [`Feeder::next_batch`] gives in order).
pub struct Prefetch {
    rx: std::sync::mpsc::Receiver<Result<Option<Batch>>>,
    _thread: std::thread::JoinHandle<()>,
}

impl Prefetch {
    pub fn start(mut feeder: Feeder, mut src: ShardSource, depth: usize) -> Self {
        let (tx, rx) = std::sync::mpsc::sync_channel(depth.max(1));
        let thread = std::thread::spawn(move || {
            loop {
                let b = feeder.next_batch_from(&mut src);
                let end = !matches!(b, Ok(Some(_)));
                if tx.send(b).is_err() || end {
                    return;
                }
            }
        });
        Self { rx, _thread: thread }
    }

    pub fn next(&self) -> Result<Option<Batch>> {
        self.rx.recv().map_err(|_| anyhow::anyhow!("data thread stopped"))?
    }
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
    /// Data for [`Trainer::train_step`] (the prefetching path keeps its own, see [`Prefetch`]).
    pub feeder: Feeder,
    /// Core state per lane group (a group of `micro_lanes` always holds the same lanes).
    states: Vec<Option<C::State>>,
    pub step: usize,
    /// 1.0 for S1, `s2_lr_scale` for S2 (041-R4).
    pub lr_scale: f64,
    /// Data state after the last trained batch (saved in checkpoints, 061-R10).
    pub data: FeedState,
}

impl<C: CoreInit> Trainer<C> {
    /// `device` must be an autodiff device.
    pub fn new(net: UraniumNet<C>, net_cfg: UraniumNetConfig, cfg: BcConfig, device: Device) -> Self {
        let optim = AdamWConfig::new()
            .with_weight_decay(cfg.weight_decay)
            .with_grad_clipping(Some(GradientClippingConfig::Norm(cfg.grad_clip)))
            .init();
        let feeder = Feeder::new(&cfg);
        let data = feeder.state(None);
        Self { net, cfg, net_cfg, optim, device, feeder, states: Vec::new(), step: 0, lr_scale: 1.0, data }
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
        let lanes = batch.chunks.len();
        let g = self.group_size();
        let groups = lanes.div_ceil(g);
        self.states.resize_with(groups, || None);
        let mut acc = GradientsAccumulator::<UraniumNet<C>>::new();
        let mut report = LossParts { valid, ..Default::default() };
        for i in 0..groups {
            let (loss, parts) = self.group_loss(&batch, i, i * g..((i + 1) * g).min(lanes), valid, &mut prof);
            report.add(&parts);
            let grads = GradientsParams::from_grads(loss.backward(), &self.net);
            prof.mark("backward");
            acc.accumulate(&self.net, grads);
            prof.mark("accumulate");
        }
        let lr = self.cfg.lr_at(self.step, self.lr_scale);
        self.net = self.optim.step(lr, self.net.clone(), acc.grads());
        prof.mark("optimizer");
        prof.report(self.step);
        self.step += 1;
        self.data = batch.state;
        Ok(report)
    }

    /// Forward pass and weighted loss of lane group `group` (lanes `lanes`), divided by the
    /// batch-wide `valid`. Carries the group's core state to its next chunk.
    fn group_loss(&mut self, batch: &Batch, group: usize, lanes: std::ops::Range<usize>, valid: f32, prof: &mut Prof) -> (Tensor<1>, LossParts) {
        let (b, t) = (lanes.len(), self.cfg.seq_len);
        // Gather rows in b-major order; a game shorter than the chunk is padded with its last
        // step at weight 0.
        let mut rows: Vec<&TrainStep> = Vec::with_capacity(b * t);
        let mut weights = Vec::with_capacity(b * t);
        let mut results = Vec::with_capacity(b * t);
        let mut keep = Vec::with_capacity(b);
        for l in lanes {
            let chunk = &batch.chunks[l];
            keep.push(!batch.fresh[l]);
            for k in 0..t {
                let s = &chunk[k.min(chunk.len() - 1)];
                rows.push(s);
                weights.push(if k < chunk.len() { s.weight } else { 0.0 });
                results.push(batch.results[l]);
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
        prof.mark("collate");

        let keep_t = Tensor::<1, Bool>::from_data(TensorData::new(keep, [b]), d);
        let s0 = match self.states[group].take() {
            Some(s) => C::mask_state(s, keep_t),
            None => self.net.init_state(b, d),
        };
        let out = self.net.forward_seq(&ob, &ab, Some(&pb), s0, t);
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
        // 041-R7: win value head regresses the replay result (privileged inputs), MSE.
        let target = Tensor::<1>::from_data(TensorData::new(results, [b * t]), d);
        let win = out.value.narrow(1, VALUE_WIN, 1).reshape([b * t]);
        let value_loss = ((win - target).powi_scalar(2) * w.clone()).sum() / valid as f64;
        let mut loss = value_loss.clone() * self.cfg.value.win as f64;
        for (l, k) in &parts {
            loss = loss + l.clone() * *k as f64;
        }
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
            valid: 0.0,
        };
        prof.mark("loss_read");
        (loss, report)
    }

    /// 032-R6 / 041-R10: weights + config + schema versions + hyper-parameters; 061-R10: optimizer
    /// state, step and RNG so a run can resume on any host.
    pub fn save(&self, dir: &Path) -> Result<()> {
        let mut meta = CheckpointMeta::new(&self.net_cfg, uranium_env::features::FEATURE_SCHEMA_VERSION, uranium_env::actions::ACTION_SCHEMA_VERSION)?;
        meta.train = serde_json::json!({ "bc": self.cfg, "step": self.step, "lr_scale": self.lr_scale, "games": self.data.games, "invalid_labels": self.data.invalid_labels });
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
        eprintln!("{}", serde_json::json!({ "profile_step": step, "secs": m }));
    }
}

/// Optimizer name in a BC checkpoint (`optim-main.bpk`).
const OPTIM_NAME: &str = "main";
