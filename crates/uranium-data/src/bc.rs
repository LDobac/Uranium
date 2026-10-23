//! Behaviour cloning data side (041): the BC config and its fingerprint (B-3), the learning-rate
//! schedule (041-R3), B lanes walking through games into batches of T steps per lane (041-R2) with
//! the data RNG (z drop 041-R8, symmetry 040-R7), the resumable data position (061-R10), the
//! prefetch thread (041-R16) and the run identity checked on resume (B-3). No tensors: the trainer
//! (Burn until the PyTorch switch, then Python via `uranium._core`, AMB-000-4) turns a [`Batch`]
//! into an update. Also the one-epoch lengths per stage (041-R13).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use uranium_env::features::augment::Sym;
use uranium_store::ckpt;

use crate::data::{GameId, GameStream, GameSummary, MapStore, Rng, ShardSource, SourcePos, StepSource, TrainStep, epoch_updates, sample_games};
use crate::pack::Planes;
use crate::split::Selection;

/// The clip semantics, part of the BC config fingerprint (B-3): one global L2 norm (041-R3), not
/// Burn's per-tensor clip used before 2026-10-19.
pub const CLIP_MODE: &str = "global_l2";

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
    /// 041-R19: win_open head MSE weight. Present = a new run builds the head (032-R18). Absent
    /// (configs before 2026-10-21, the core ablation) is left out of the fingerprint JSON, so
    /// their B-3 fingerprints do not change.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub win_open: Option<f32>,
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
    /// Global L2-norm clip over all parameter gradients (041-R3).
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
    /// [`CLIP_MODE`]), so runs from before the change do not match the fingerprint.
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
            clip_mode: CLIP_MODE,
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

/// What a `bc` invocation trains (041-R4, R5, R13).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BcStage {
    /// S1 pre-training, all sources.
    S1,
    /// S2 fine-tuning from S1 (learning rate x `s2_lr_scale`).
    S2,
    /// Ablation (A): S2 data from scratch at the S1 learning rate.
    AblationA,
}

impl BcStage {
    pub fn selection(self) -> Selection {
        match self {
            BcStage::S1 => Selection::S1,
            BcStage::S2 | BcStage::AblationA => Selection::S2,
        }
    }

    /// Schedule length of the stage from `config/bc.toml` `[steps]` (0 = keep `total_steps`).
    pub fn steps(self, cfg: &BcConfig) -> usize {
        match self {
            BcStage::S1 => cfg.steps.s1,
            BcStage::S2 => cfg.steps.s2,
            BcStage::AblationA => cfg.steps.ablation_a,
        }
    }

    /// Warmup of the stage (041-R14: S2 and (A) from `[steps]`, 0 = keep `warmup_steps`).
    pub fn warmup(self, cfg: &BcConfig) -> usize {
        match self {
            BcStage::S1 => 0,
            BcStage::S2 => cfg.steps.s2_warmup,
            BcStage::AblationA => cfg.steps.ablation_a_warmup,
        }
    }
}

impl BcConfig {
    /// The effective config of `stage`: the schedule length and warmup from `[steps]` (041-R14;
    /// 0 keeps `total_steps` / `warmup_steps`). The fingerprint (B-3) is taken from this config.
    pub fn for_stage(mut self, stage: BcStage) -> Self {
        let steps = stage.steps(&self);
        if steps > 0 {
            self.total_steps = steps;
        }
        let warmup = stage.warmup(&self);
        if warmup > 0 {
            self.warmup_steps = warmup;
        }
        self
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
    pub chunks: Vec<Vec<TrainStep>>,
    /// 041-R18: per lane and step, the planes taken out of `chunks` (empty = not packed).
    pub planes: Vec<Vec<Planes>>,
    pub fresh: Vec<bool>,
    pub results: Vec<f32>,
    /// 041-R7 / A-3: per lane, the value-loss weight of its game (0 = undecided result).
    pub value_weights: Vec<f32>,
    pub state: FeedState,
}

impl Batch {
    /// Per lane: the T steps (fewer at a game's end), whether the game started here, its result.
    pub fn parts(&self) -> (&[Vec<TrainStep>], &[bool], &[f32]) {
        assert!(self.planes.is_empty(), "041-R18: a packed batch is read by the trainer only");
        (&self.chunks, &self.fresh, &self.results)
    }

    /// Per lane: the value-loss weight of its game (041-R7 / A-3).
    pub fn value_weights(&self) -> &[f32] {
        &self.value_weights
    }
}

/// Walks B lanes through the games of a source (041-R2) and turns the next T steps of every lane
/// into a [`Batch`]. Owns the data RNG (z drop 041-R8, symmetry 040-R7). Runs on the trainer's
/// thread (the trainer) or ahead of it on its own thread ([`Prefetch`]); the batches
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
    /// 041-R18: batches with their planes compressed ([`Feeder::set_pack`]).
    pack: bool,
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
            pack: false,
        }
    }

    /// 041-R18: keep the batches' spatial planes and location masks compressed until their lane
    /// group trains. Does not change the batches.
    pub fn set_pack(&mut self, pack: bool) {
        self.pack = pack;
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
        let mut planes = Vec::new();
        for lane in &mut self.lanes {
            let Some(g) = lane.game.as_mut() else {
                // Drained lane (evaluation only).
                if self.pack {
                    planes.push(Vec::new());
                }
                chunks.push(Vec::new());
                fresh.push(true);
                results.push(0.0);
                value_weights.push(0.0);
                continue;
            };
            let (mut chunk, invalid) = g.next_chunk(self.seq_len);
            self.invalid_labels += invalid;
            if self.pack {
                planes.push(chunk.iter_mut().map(|s| Planes::take(&mut s.obs)).collect());
            }
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
        Ok(Some(Batch { chunks, planes, fresh, results, value_weights, state: self.state(pos) }))
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

/// 041-R17: this host's entity encoder chunk, `URANIUM_BC_ENTITY_CHUNK` (001-R9) rows, unset,
/// empty or 0 = off.
pub fn entity_chunk() -> Result<usize> {
    match std::env::var("URANIUM_BC_ENTITY_CHUNK") {
        Ok(v) if !v.trim().is_empty() => v.trim().parse().with_context(|| format!("URANIUM_BC_ENTITY_CHUNK={v:?}: expected a row count (0 = off)")),
        _ => Ok(0),
    }
}

/// 041-R18: whether this host keeps batches compressed, `URANIUM_BC_PACK_BATCH` (001-R9): `1`
/// on, unset, empty or `0` off.
pub fn pack_batch() -> Result<bool> {
    match std::env::var("URANIUM_BC_PACK_BATCH").as_deref().map(str::trim) {
        Ok("1") => Ok(true),
        Ok("" | "0") | Err(_) => Ok(false),
        Ok(v) => anyhow::bail!("URANIUM_BC_PACK_BATCH={v:?}: expected 0 or 1"),
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

/// Adds `ident` to the training state of the checkpoint in `dir` (after the trainer saved it).
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

/// One-epoch length of a stage (041-R13): updates the trainer's lane walk makes.
#[derive(Serialize, Debug, Clone, Copy, Default)]
pub struct EpochCount {
    pub updates: u64,
    pub games: usize,
    /// Every step of the games, the game-start NoOp steps (040-R18, E-2) included.
    pub steps: u64,
}

#[derive(Serialize, Debug)]
pub struct EpochTable {
    pub lanes: usize,
    pub seq_len: usize,
    pub seed: u64,
    pub s1: EpochCount,
    pub s2: EpochCount,
    /// Ablation (A) reads the S2 games in the same order.
    pub ablation_a: EpochCount,
}

/// Reads every shard once (in parallel) and counts the updates of one epoch per stage.
pub fn epoch_table(dirs: &[PathBuf], cfg: &BcConfig) -> Result<EpochTable> {
    let base = ShardSource::from_dirs(dirs)?;
    let scan = crate::data::scan_cached(&base)?;
    let by_path: std::collections::HashMap<&PathBuf, &Vec<GameSummary>> = base.files().iter().zip(scan.iter()).collect();
    let count = |sel: Selection| -> Result<EpochCount> {
        let src = ShardSource::from_dirs(dirs)?.select(sel).shuffled(cfg.data.seed);
        let mut lens = Vec::new();
        for (k, f) in src.files().iter().enumerate() {
            lens.extend(src.order_file(k, by_path[f].clone()).into_iter().map(|i| i.steps));
        }
        let games = lens.iter().filter(|&&l| l > 0).count();
        let steps = lens.iter().map(|&l| l as u64).sum();
        Ok(EpochCount { updates: epoch_updates(lens, cfg.lanes, cfg.seq_len), games, steps })
    };
    let s1 = count(Selection::S1)?;
    let s2 = count(Selection::S2)?;
    Ok(EpochTable { lanes: cfg.lanes, seq_len: cfg.seq_len, seed: cfg.data.seed, s1, s2, ablation_a: s2 })
}

/// 032-R17 ② (G-D8): the games `bc-eval --games n` scores: a fixed-seed shuffle over every game
/// of the split `src` selects (from a scan of its files; `src` has no game shuffle), first
/// `n`. The same for any checkpoint and core.
pub fn eval_games(src: &ShardSource, scan: &[Vec<GameSummary>], n: usize, seed: u64) -> Vec<crate::data::GameId> {
    let counts: Vec<usize> = scan.iter().enumerate().map(|(k, items)| src.order_file(k, items.clone()).len()).collect();
    sample_games(&counts, n, seed)
}

/// One lane group of a batch, collated on the host (041-R2, R3): what the trainer feeds one
/// forward/backward. Rows are b-major (row = lane * T + step); a game shorter than the chunk is
/// padded with its last step at weight 0.
#[derive(Debug, Clone, PartialEq)]
pub struct HostGroup {
    /// Lanes in the group.
    pub b: usize,
    /// Steps per lane (`seq_len`).
    pub t: usize,
    pub obs: crate::collate::HostObsBatch,
    pub labels: crate::collate::HostActionBatch,
    pub privileged: crate::collate::HostPrivBatch,
    /// [b * t] label weight (0 = void label or chunk padding).
    pub weights: Vec<f32>,
    /// [b * t] the game result of the row's lane (041-R7).
    pub results: Vec<f32>,
    /// [b * t] value-loss weight: label weight x the game's value weight (041-R7 / A-3).
    pub value_w: Vec<f32>,
    /// [b] true when the lane continues a game from the last batch (core state kept, 041-R2).
    pub keep: Vec<bool>,
}

/// Field order of [`HostGroup::digest`] (the Python side hashes its arrays in the same order).
pub const GROUP_FIELDS: [&str; 26] = [
    "entity_cat", "entity_cont", "entity_mask", "entity_cell", "entity_pos", "spatial", "scalar", "scalar_cat", "z_build", "z_units", "z_present", "type_mask", "unit_cap",
    "loc_class_mask", "is_building", "action", "delay", "queued", "selected", "target_unit", "target_loc", "privileged", "weights", "results", "value_w", "keep",
];

impl HostGroup {
    /// SHA-256 (hex) over the group's buffers in [`GROUP_FIELDS`] order: i32 / f32 little-endian,
    /// bool one byte 0 / 1. Lets another implementation check it reads the same batch.
    pub fn digest(&self) -> String {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        let i = |h: &mut Sha256, v: &[i32]| v.iter().for_each(|x| h.update(x.to_le_bytes()));
        let f = |h: &mut Sha256, v: &[f32]| v.iter().for_each(|x| h.update(x.to_le_bytes()));
        let b = |h: &mut Sha256, v: &[bool]| h.update(v.iter().map(|&x| x as u8).collect::<Vec<u8>>());
        let (o, l) = (&self.obs, &self.labels);
        i(&mut h, &o.entity_cat);
        f(&mut h, &o.entity_cont);
        b(&mut h, &o.entity_mask);
        i(&mut h, &o.entity_cell);
        f(&mut h, &o.entity_pos);
        f(&mut h, &o.spatial);
        f(&mut h, &o.scalar);
        i(&mut h, &o.scalar_cat);
        i(&mut h, &o.z_build);
        f(&mut h, &o.z_units);
        b(&mut h, &o.z_present);
        b(&mut h, &o.type_mask);
        b(&mut h, &o.unit_cap);
        b(&mut h, &o.loc_class_mask);
        b(&mut h, &o.is_building);
        i(&mut h, &l.action);
        i(&mut h, &l.delay);
        i(&mut h, &l.queued);
        i(&mut h, &l.selected);
        i(&mut h, &l.target_unit);
        i(&mut h, &l.target_loc);
        f(&mut h, &self.privileged.summary);
        f(&mut h, &self.weights);
        f(&mut h, &self.results);
        f(&mut h, &self.value_w);
        b(&mut h, &self.keep);
        h.finalize().iter().map(|x| format!("{x:02x}")).collect()
    }
}

impl Batch {
    /// Lanes in the batch.
    pub fn lanes(&self) -> usize {
        self.chunks.len()
    }

    /// Usable label steps of the whole batch (at least 1): every lane group's policy loss is
    /// divided by it, so the summed group gradients equal one pass over all lanes (041-R3).
    pub fn valid(&self) -> f32 {
        self.chunks.iter().flat_map(|c| c.iter().map(|s| s.weight)).sum::<f32>().max(1.0)
    }

    /// Usable label steps of decided games (at least 1): the value-loss divisor (041-R7 / A-3).
    pub fn value_valid(&self) -> f32 {
        self.chunks.iter().zip(&self.value_weights).map(|(c, vw)| c.iter().map(|s| s.weight * vw).sum::<f32>()).sum::<f32>().max(1.0)
    }

    /// Lanes `lanes` collated for one forward/backward: `t` steps per lane, entities padded to a
    /// multiple of `pad` (030-R7), selections cut at `max_select` with END = padded n (032-R12).
    /// A packed batch (041-R18) gets its planes back here, for this group only.
    pub fn group(&self, lanes: std::ops::Range<usize>, t: usize, pad: usize, max_select: usize) -> HostGroup {
        let b = lanes.len();
        let mut rows: Vec<&TrainStep> = Vec::with_capacity(b * t);
        let mut weights = Vec::with_capacity(b * t);
        let mut results = Vec::with_capacity(b * t);
        let mut value_w = Vec::with_capacity(b * t);
        let mut keep = Vec::with_capacity(b);
        let mut restored = Vec::new();
        for l in lanes {
            let chunk = &self.chunks[l];
            keep.push(!self.fresh[l]);
            for k in 0..t {
                let i = k.min(chunk.len() - 1);
                let s = &chunk[i];
                if !self.planes.is_empty() {
                    restored.push(self.planes[l][i].restore(&s.obs));
                }
                rows.push(s);
                let w = if k < chunk.len() { s.weight } else { 0.0 };
                weights.push(w);
                results.push(self.results[l]);
                value_w.push(w * self.value_weights[l]);
            }
        }
        let obs: Vec<_> = if restored.is_empty() { rows.iter().map(|s| &s.obs).collect() } else { restored.iter().collect() };
        let obs = crate::collate::obs_batch(&obs, pad);
        let labels: Vec<_> = rows.iter().map(|s| &s.label).collect();
        let labels = crate::collate::action_batch(&labels, obs.n, max_select);
        let privs: Vec<_> = rows.iter().map(|s| &s.privileged).collect();
        let privileged = crate::collate::priv_batch(&privs);
        HostGroup { b, t, obs, labels, privileged, weights, results, value_w, keep }
    }
}
