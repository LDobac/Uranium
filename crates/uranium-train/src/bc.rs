//! Behaviour cloning trainer (041). B lanes each walk through games; every step trains on the
//! next T observations of every lane with the core state carried over between chunks of the same
//! game and reset when a lane starts a new game (041-R2, 032-R5). Lanes run forward/backward in
//! groups of `micro_lanes` and their gradients are summed before one optimizer step, so peak GPU
//! memory follows the group, not B (entity attention is N² per row).

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use burn::module::Module;
use burn::optim::{AdamWConfig, GradientsAccumulator, GradientsParams, ModuleOptimizer};
use burn::tensor::{Bool, Device, Tensor, TensorData};
use serde::{Deserialize, Serialize};
use uranium_env::features::augment::Sym;
use uranium_model::checkpoint::{self, CheckpointMeta};
use uranium_model::collate::{HostLabel, action_batch, obs_batch_padded, priv_batch};
use uranium_model::config::UraniumNetConfig;
use uranium_model::value::VALUE_WIN;
use uranium_model::{CoreInit, UraniumNet};

use crate::ckpt::{self, TrainState};
use crate::data::{GameId, GameStream, MapStore, Rng, ShardSource, SourcePos, StepSource, TrainStep};

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
    /// 030-R11 BC exception (user decision 2026-10-19): share of games whose enemy race is
    /// `unknown` until first seen (per game, hash of replay SHA-256 + `seed`). Default 0.05.
    #[serde(default = "default_enemy_race_mask")]
    pub enemy_race_mask: f64,
}

fn default_enemy_race_mask() -> f64 {
    0.05
}

/// Updates of one epoch per stage (041-R13, 2026-10-18: S1, S2 and ablation (A) train one epoch
/// each and the cosine schedule ends there). `uranium-train bc-epoch` counts them.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct StageSteps {
    pub s1: usize,
    pub s2: usize,
    pub ablation_a: usize,
    /// Warmup of S2 and (A): 5% of their epoch, rounded half up (2026-10-18). 0 = `warmup_steps`.
    #[serde(default)]
    pub s2_warmup: usize,
    #[serde(default)]
    pub ablation_a_warmup: usize,
}

/// 041-R14 short-stage warmup: `steps` x 5% rounded half up.
pub fn short_stage_warmup(steps: usize) -> usize {
    (steps * 5 + 50) / 100
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
    /// Global L2-norm clip over all parameter gradients (041-R3, `crate::clip`).
    pub grad_clip: f32,
    pub s2_lr_scale: f64,
    pub heads: HeadWeights,
    pub value: ValueWeights,
    pub data: DataConfig,
    /// 061-R6 (G-D6, 2026-10-19): newest checkpoints a run keeps on the shared drive. Absent = 5
    /// (061-R6 default); 0 = keep every checkpoint (the core ablation, 032-R17 ⑤).
    #[serde(default)]
    pub keep_checkpoints: Option<usize>,
}

/// B-3: the `BcConfig` settings that change what a run learns; [`BcConfig::fingerprint`] hashes
/// only these. Add a new result-affecting field here.
#[derive(Serialize, Debug, Clone, PartialEq)]
pub struct ResultSettings {
    pub lanes: usize,
    pub seq_len: usize,
    pub lr: f64,
    pub min_lr: f64,
    pub warmup_steps: usize,
    pub total_steps: usize,
    pub weight_decay: f32,
    pub grad_clip: f32,
    pub s2_lr_scale: f64,
    pub heads: HeadWeights,
    pub value: ValueWeights,
    pub z_drop: f64,
    pub augment: bool,
    pub seed: u64,
    pub enemy_race_mask: f64,
    /// Not a config field: the gradient clip semantics (041-R3 global L2 norm since 2026-10-19,
    /// `crate::clip::CLIP_MODE`), so runs from before the change do not match the fingerprint.
    pub clip_mode: &'static str,
}

/// B-3: `BcConfig` fields (as `toml` keys, `data.` for the `[data]` table) left out of the
/// fingerprint because they do not change the updates (041-R3: `micro_lanes` gives the same
/// update; padding, checkpointing, saving and retention do not touch the math) or because the
/// stage's share is already in `total_steps` / `warmup_steps` (`steps`). New fields that do not
/// change results must be listed here, result-affecting ones go into [`ResultSettings`]; a field
/// in neither fails `config_fingerprint_fields_are_classified_b3`.
pub const EXECUTION_ONLY: [&str; 7] = ["micro_lanes", "grad_checkpointing", "pad_entities", "steps", "keep_checkpoints", "data.checkpoint_every", "data.checkpoint_minutes"];

/// 061-R6 default number of checkpoints kept per run.
pub const DEFAULT_KEEP_CHECKPOINTS: usize = 5;

impl BcConfig {
    pub fn load(path: &Path) -> Result<Self> {
        Ok(toml::from_str(&std::fs::read_to_string(path)?)?)
    }

    /// 061-R6: checkpoints kept on the share (0 = all).
    pub fn keep_checkpoints(&self) -> usize {
        self.keep_checkpoints.unwrap_or(DEFAULT_KEEP_CHECKPOINTS)
    }

    /// B-3: SHA-256 (16 hex digits) of [`BcConfig::results_view`], the settings that decide what
    /// a run learns. A whitelist: a new `BcConfig` field is outside the fingerprint until it is
    /// added to [`ResultSettings`], and the test `config_fingerprint_fields_are_classified_b3`
    /// fails until the field is listed there or in [`EXECUTION_ONLY`].
    pub fn fingerprint(&self) -> String {
        use sha2::{Digest, Sha256};
        let json = serde_json::to_vec(&self.results_view()).expect("ResultSettings serialises");
        Sha256::digest(&json).iter().take(8).map(|b| format!("{b:02x}")).collect()
    }

    /// The result-affecting settings (B-3): the effective config after the stage's `[steps]`
    /// (schedule length and warmup in `total_steps` / `warmup_steps`), learning rates, clip, loss
    /// weights and the data settings.
    pub fn results_view(&self) -> ResultSettings {
        ResultSettings {
            lanes: self.lanes,
            seq_len: self.seq_len,
            lr: self.lr,
            min_lr: self.min_lr,
            warmup_steps: self.warmup_steps,
            total_steps: self.total_steps,
            weight_decay: self.weight_decay,
            grad_clip: self.grad_clip,
            s2_lr_scale: self.s2_lr_scale,
            heads: self.heads.clone(),
            value: self.value.clone(),
            z_drop: self.data.z_drop,
            augment: self.data.augment,
            seed: self.data.seed,
            enemy_race_mask: self.data.enemy_race_mask,
            clip_mode: crate::clip::CLIP_MODE,
        }
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
    /// Index into `Sym::augment()` (030-R4).
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
    /// 041-R7 / A-3: games started so far whose result is undecided (`result_source = none`,
    /// value weight 0).
    #[serde(default)]
    pub no_result_games: usize,
}

/// The CPU half of an update: B lanes x T steps of features and labels (041-R2), plus the data
/// state after it.
pub struct Batch {
    chunks: Vec<Vec<TrainStep>>,
    fresh: Vec<bool>,
    results: Vec<f32>,
    /// 041-R7 / A-3: per lane, the value-loss weight of its game (0 = undecided result).
    value_weights: Vec<f32>,
    pub state: FeedState,
}

impl Batch {
    /// Per lane: the T steps (fewer at a game's end), whether the game started here, its result.
    pub fn parts(&self) -> (&[Vec<TrainStep>], &[bool], &[f32]) {
        (&self.chunks, &self.fresh, &self.results)
    }

    /// Per lane: the value-loss weight of its game (041-R7 / A-3).
    pub fn value_weights(&self) -> &[f32] {
        &self.value_weights
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
    /// 030-R11: `data.enemy_race_mask` and `data.seed`.
    race_mask: f64,
    seed: u64,
    games: usize,
    invalid_labels: usize,
    no_result_games: usize,
    /// Fingerprint of the shard source (set on the first batch from it).
    files: String,
    /// Evaluation: when the source runs dry, idle lanes give empty chunks until every lane is
    /// done (training instead ends the epoch at the first lane it cannot fill).
    pub drain: bool,
    /// Map precomputes (B-2): `URANIUM_MAPS_DIR` unless replaced with [`Feeder::with_maps`].
    maps: Arc<MapStore>,
}

impl Feeder {
    /// Map precomputes from `URANIUM_MAPS_DIR` (B-2: a game without one is an error).
    pub fn new(cfg: &BcConfig) -> Self {
        Self {
            lanes: (0..cfg.lanes).map(|_| Lane::empty()).collect(),
            rng: Rng::new(cfg.data.seed),
            seq_len: cfg.seq_len,
            z_drop: cfg.data.z_drop,
            augment: cfg.data.augment,
            race_mask: cfg.data.enemy_race_mask,
            seed: cfg.data.seed,
            games: 0,
            invalid_labels: 0,
            no_result_games: 0,
            files: String::new(),
            drain: false,
            maps: Arc::new(MapStore::from_env()),
        }
    }

    /// Another map precompute store (shared with a start check, or synthetic maps in tests).
    pub fn with_maps(mut self, maps: Arc<MapStore>) -> Self {
        self.maps = maps;
        self
    }

    pub fn set_maps(&mut self, maps: Arc<MapStore>) {
        self.maps = maps;
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
            let sym = if self.augment { Sym::augment()[(self.rng.next() % 4) as usize] } else { Sym::IDENTITY };
            self.games += 1;
            let g = GameStream::new(game, z_present, sym, &self.maps)?.with_race_mask(self.race_mask, self.seed);
            self.no_result_games += (g.value_weight == 0.0) as usize;
            self.lanes[i] = Lane { game: Some(g), id: src.last_id(), fresh: true };
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
        let mut value_weights = Vec::with_capacity(self.lanes.len());
        for lane in &mut self.lanes {
            let Some(g) = lane.game.as_mut() else {
                // Drained lane (evaluation only).
                chunks.push(Vec::new());
                fresh.push(true);
                results.push(0.0);
                value_weights.push(0.0);
                continue;
            };
            let (chunk, invalid) = g.next_chunk(self.seq_len);
            self.invalid_labels += invalid;
            chunks.push(chunk);
            fresh.push(lane.fresh);
            results.push(g.result);
            value_weights.push(g.value_weight);
            lane.fresh = false;
            if g.remaining() == 0 {
                lane.game = None;
                lane.id = None;
            }
        }
        Ok(Some(Batch { chunks, fresh, results, value_weights, state: self.state(pos) }))
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
                Some(LaneState { id, pos: g.pos(), z_present: g.z_present(), sym: Sym::augment().iter().position(|s| *s == g.sym()).unwrap_or(0) as u8 })
            })
            .collect();
        FeedState {
            lanes,
            source: pos,
            rng: self.rng.state(),
            games: self.games,
            invalid_labels: self.invalid_labels,
            files: self.files.clone(),
            no_result_games: self.no_result_games,
        }
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
        self.no_result_games = st.no_result_games;
        // B-13: every shard the lanes need is decoded once, not once per lane.
        let saved: Vec<&LaneState> = st.lanes.iter().flatten().collect();
        let ids: Vec<GameId> = saved.iter().map(|s| s.id).collect();
        let mut games = src.games(&ids)?.into_iter();
        for (lane, saved) in self.lanes.iter_mut().zip(st.lanes.iter().chain(std::iter::repeat(&None))) {
            *lane = Lane::empty();
            let Some(s) = saved else { continue };
            let game = games.next().expect("one game per saved lane");
            let mut g = GameStream::new(game, s.z_present, Sym::augment()[s.sym as usize % 4], &self.maps)?.with_race_mask(self.race_mask, self.seed);
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
        self.no_result_games = st.no_result_games;
        self.reset_lanes();
    }

    pub fn reset_lanes(&mut self) {
        for l in &mut self.lanes {
            *l = Lane::empty();
        }
    }
}

/// 041-R16: batches built ahead of the trainer when nothing in `.env` says otherwise.
pub const PREFETCH_DEPTH_DEFAULT: usize = 2;

/// 041-R16: this host's prefetch depth, `URANIUM_BC_PREFETCH` (001-R9), default
/// [`PREFETCH_DEPTH_DEFAULT`].
pub fn prefetch_depth() -> Result<usize> {
    match std::env::var("URANIUM_BC_PREFETCH") {
        Ok(v) if !v.trim().is_empty() => v.trim().parse().with_context(|| format!("URANIUM_BC_PREFETCH={v:?}: expected a batch count (0 = no data thread)")),
        _ => Ok(PREFETCH_DEPTH_DEFAULT),
    }
}

/// [`Feeder`] on its own thread, `depth` batches ahead of the trainer (041 throughput: feature
/// extraction and shard reads overlap the GPU update), or with `depth` 0 on the caller's thread
/// when it asks, so only the batch being trained is in host memory (041-R16, a ~4 GB batch on a
/// host sharing its RAM with the GPU). Either way the batches are the ones
/// [`Feeder::next_batch_from`] gives in order.
pub struct Prefetch(PrefetchMode);

enum PrefetchMode {
    Thread { rx: std::sync::mpsc::Receiver<Result<Option<Batch>>>, _thread: std::thread::JoinHandle<()> },
    Inline { feeder: Box<Feeder>, src: ShardSource },
}

impl Prefetch {
    pub fn start(mut feeder: Feeder, mut src: ShardSource, depth: usize) -> Self {
        if depth == 0 {
            return Self(PrefetchMode::Inline { feeder: Box::new(feeder), src });
        }
        let (tx, rx) = std::sync::mpsc::sync_channel(depth);
        let thread = std::thread::spawn(move || {
            loop {
                let b = feeder.next_batch_from(&mut src);
                let end = !matches!(b, Ok(Some(_)));
                if tx.send(b).is_err() || end {
                    return;
                }
            }
        });
        Self(PrefetchMode::Thread { rx, _thread: thread })
    }

    // Not an Iterator: a batch can fail (`Err`) and the training loop stops on the first `None`.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Result<Option<Batch>> {
        match &mut self.0 {
            PrefetchMode::Thread { rx, .. } => rx.recv().map_err(|_| anyhow::anyhow!("data thread stopped"))?,
            PrefetchMode::Inline { feeder, src } => feeder.next_batch_from(src),
        }
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
        // 041-R3 (G-D4): no Burn clipping (it is per tensor); `train_batch` clips the global norm.
        let optim = AdamWConfig::new().with_weight_decay(cfg.weight_decay).init();
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
            let (loss, parts) = self.group_loss(&batch, i, i * g..((i + 1) * g).min(lanes), valid, value_valid, &mut prof);
            report.add(&parts);
            let grads = GradientsParams::from_grads(loss.backward(), &self.net);
            prof.mark("backward");
            acc.accumulate(&self.net, grads);
            prof.mark("accumulate");
        }
        // 041-R3 (G-D4): one global L2-norm clip over the summed gradients of all lane groups.
        let (grads, _norm) = crate::clip::clip_global_norm(&self.net, acc.grads(), self.cfg.grad_clip);
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
    /// Carries the group's core state to its next chunk.
    fn group_loss(&mut self, batch: &Batch, group: usize, lanes: std::ops::Range<usize>, valid: f32, value_valid: f32, prof: &mut Prof) -> (Tensor<1>, LossParts) {
        let (b, t) = (lanes.len(), self.cfg.seq_len);
        // Gather rows in b-major order; a game shorter than the chunk is padded with its last
        // step at weight 0.
        let mut rows: Vec<&TrainStep> = Vec::with_capacity(b * t);
        let mut weights = Vec::with_capacity(b * t);
        let mut results = Vec::with_capacity(b * t);
        // 041-R7 / A-3: value-loss weight per row = label weight x the game's value weight.
        let mut value_w = Vec::with_capacity(b * t);
        let mut keep = Vec::with_capacity(b);
        for l in lanes {
            let chunk = &batch.chunks[l];
            keep.push(!batch.fresh[l]);
            for k in 0..t {
                let s = &chunk[k.min(chunk.len() - 1)];
                rows.push(s);
                let w = if k < chunk.len() { s.weight } else { 0.0 };
                weights.push(w);
                results.push(batch.results[l]);
                value_w.push(w * batch.value_weights[l]);
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
        // 041-R7: win value head regresses the replay result (privileged inputs), MSE; games
        // with an undecided result (A-3) have value weight 0.
        let target = Tensor::<1>::from_data(TensorData::new(results, [b * t]), d);
        let vw = Tensor::<1>::from_data(TensorData::new(value_w, [b * t]), d);
        let win = out.value.narrow(1, VALUE_WIN, 1).reshape([b * t]);
        let value_loss = ((win - target).powi_scalar(2) * vw).sum() / value_valid as f64;
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

/// B-3 (2026-10-19): what a BC run trains, stored in each checkpoint's training state
/// (`extra.run`) and compared on resume, so a run never continues with another stage, config or
/// dataset (another data order, schedule length or learning rate) without notice.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct RunIdent {
    /// `S1`, `S2` or `AblationA` (041-R4, R5, R13).
    pub stage: String,
    /// [`BcConfig::fingerprint`] of the effective config.
    pub config: String,
    /// [`crate::data::dataset_fingerprint`] of the selected shard files.
    pub dataset: String,
}

const IDENT_KEY: &str = "run";

/// Adds `ident` to the training state of the checkpoint in `dir` (after [`Trainer::save`]).
pub fn write_ident(dir: &Path, ident: &RunIdent) -> Result<()> {
    let mut state = ckpt::load_state(dir)?.ok_or_else(|| anyhow::anyhow!("{} has no training state", dir.display()))?;
    match state.extra.as_object_mut() {
        Some(o) => {
            o.insert(IDENT_KEY.into(), serde_json::to_value(ident)?);
        }
        None => anyhow::bail!("{}: training state extra is not an object", dir.display()),
    }
    ckpt::save_state(dir, &state)
}

/// The [`RunIdent`] stored in a checkpoint, if any (checkpoints before B-3 have none).
pub fn read_ident(dir: &Path) -> Result<Option<RunIdent>> {
    let Some(state) = ckpt::load_state(dir)? else { return Ok(None) };
    state.extra.get(IDENT_KEY).map(|v| serde_json::from_value(v.clone())).transpose().map_err(Into::into)
}

/// B-3: refuses to resume from `dir` when its stage, config or dataset differs from `want`.
/// A checkpoint written before B-3 (no identity) is resumed with a warning. There is no override
/// (start a new run, or `--init` from the checkpoint for weights only).
pub fn check_ident(dir: &Path, want: &RunIdent) -> Result<()> {
    let Some(have) = read_ident(dir)? else {
        crate::elog!("[bc] WARNING: {} has no stage / config / dataset fingerprint (written before B-3); resuming unchecked", dir.display());
        return Ok(());
    };
    let mut diff = Vec::new();
    if have.stage != want.stage {
        diff.push(format!("stage {} (checkpoint) != {} (now)", have.stage, want.stage));
    }
    if have.config != want.config {
        diff.push(format!("config fingerprint {} != {} (another --config, or bc.toml changed: schedule, lr, loss or data settings)", have.config, want.config));
    }
    if have.dataset != want.dataset {
        diff.push(format!("dataset fingerprint {} != {} (other --dataset / --shards, or the shards were rebuilt)", have.dataset, want.dataset));
    }
    anyhow::ensure!(
        diff.is_empty(),
        "refusing to resume from {} (B-3, 061-R10): {}. Use the stage, config and datasets the run started with, or start a new run (`--init` takes the weights only)",
        dir.display(),
        diff.join("; ")
    );
    Ok(())
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
