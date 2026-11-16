use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use uranium_env::features::augment::Sym;
use uranium_store::ckpt;

use crate::data::{GameId, GameStream, GameSummary, MapStore, Rng, ShardSource, SourcePos, StepSource, TrainStep, epoch_updates, sample_games};
use crate::pack::Planes;
use crate::split::{Selection, Subsample};

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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub win_open: Option<f32>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct DataConfig {
    pub z_drop: f64,
    pub augment: bool,
    pub seed: u64,
    pub checkpoint_every: usize,
    #[serde(default)]
    pub checkpoint_minutes: u64,
    #[serde(default = "default_enemy_race_mask")]
    pub enemy_race_mask: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subsample: Option<Subsample>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chain_start_sources: Option<Vec<String>>,
}

fn default_enemy_race_mask() -> f64 {
    0.05
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
#[serde(deny_unknown_fields)]
pub struct AuxWeights {
    #[serde(rename = "priv", default, skip_serializing_if = "Option::is_none")]
    pub privileged: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub jepa: Option<f32>,
}

impl AuxWeights {
    pub fn check(&self) -> Result<()> {
        anyhow::ensure!(self.privileged.is_some() || self.jepa.is_some(), "[aux] has neither priv nor jepa: remove the empty table (041-R23)");
        for (k, v) in [("priv", self.privileged), ("jepa", self.jepa)] {
            if let Some(x) = v {
                anyhow::ensure!(x.is_finite() && x >= 0.0, "[aux] {k} must be a finite weight >= 0, got {x} (041-R23)");
            }
        }
        Ok(())
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
#[serde(deny_unknown_fields)]
pub struct TrainSettings {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_seed: Option<u64>,
}

impl TrainSettings {
    pub fn check(&self) -> Result<()> {
        anyhow::ensure!(self.model_seed.is_some(), "[train] has no model_seed: remove the empty table (041-R24)");
        Ok(())
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
#[serde(deny_unknown_fields)]
pub struct MonitorConfig {
    #[serde(default)]
    pub spatial_every: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct StageSteps {
    pub s1: usize,
    pub s2: usize,
    pub ablation_a: usize,
    #[serde(default)]
    pub s2_warmup: usize,
    #[serde(default)]
    pub ablation_a_warmup: usize,
}

pub fn short_stage_warmup(steps: usize) -> usize {
    (steps * 5 + 50) / 100
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct BcConfig {
    pub lanes: usize,
    pub seq_len: usize,
    #[serde(default)]
    pub micro_lanes: usize,
    #[serde(default)]
    pub grad_checkpointing: bool,
    #[serde(default)]
    pub pad_entities: usize,
    pub lr: f64,
    pub min_lr: f64,
    pub warmup_steps: usize,
    pub total_steps: usize,
    #[serde(default)]
    pub steps: StageSteps,
    pub weight_decay: f32,
    pub grad_clip: f32,
    pub s2_lr_scale: f64,
    pub heads: HeadWeights,
    pub value: ValueWeights,
    pub data: DataConfig,
    #[serde(default)]
    pub keep_checkpoints: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aux: Option<AuxWeights>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub monitor: Option<MonitorConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub train: Option<TrainSettings>,
}

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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subsample: Option<Subsample>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aux: Option<AuxWeights>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub train: Option<TrainSettings>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chain_start_sources: Option<Vec<String>>,
    pub clip_mode: &'static str,
}

pub const EXECUTION_ONLY: [&str; 8] = ["micro_lanes", "grad_checkpointing", "pad_entities", "steps", "keep_checkpoints", "data.checkpoint_every", "data.checkpoint_minutes", "monitor"];

pub const DEFAULT_KEEP_CHECKPOINTS: usize = 5;

impl BcConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let mut cfg: Self = toml::from_str(&std::fs::read_to_string(path)?)?;
        if let Some(m) = micro_lanes_override()? {
            cfg.micro_lanes = m;
        }
        if let Some(s) = &cfg.data.subsample {
            s.check().with_context(|| path.display().to_string())?;
        }
        if let Some(a) = &cfg.aux {
            a.check().with_context(|| path.display().to_string())?;
        }
        if let Some(t) = &cfg.train {
            t.check().with_context(|| path.display().to_string())?;
        }
        if let Some(c) = &cfg.data.chain_start_sources {
            crate::data::check_chain_start_sources(c).with_context(|| path.display().to_string())?;
        }
        Ok(cfg)
    }

    pub fn train_source(&self, dirs: &[PathBuf], sel: Selection) -> Result<ShardSource> {
        Ok(ShardSource::from_dirs(dirs)?.select(sel).subsample(self.data.subsample.clone(), self.data.seed).shuffled(self.data.seed))
    }

    pub fn keep_checkpoints(&self) -> usize {
        self.keep_checkpoints.unwrap_or(DEFAULT_KEEP_CHECKPOINTS)
    }

    pub fn fingerprint(&self) -> String {
        use sha2::{Digest, Sha256};
        let json = serde_json::to_vec(&self.results_view()).expect("ResultSettings serialises");
        Sha256::digest(&json).iter().take(8).map(|b| format!("{b:02x}")).collect()
    }

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
            subsample: self.data.subsample.clone(),
            aux: self.aux.clone(),
            train: self.train.clone(),
            chain_start_sources: self.data.chain_start_sources.clone().map(|mut v| {
                v.sort();
                v.dedup();
                v
            }),
            clip_mode: CLIP_MODE,
        }
    }

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BcStage {
    S1,
    S2,
    AblationA,
}

impl BcStage {
    pub fn selection(self) -> Selection {
        match self {
            BcStage::S1 => Selection::S1,
            BcStage::S2 | BcStage::AblationA => Selection::S2,
        }
    }

    pub fn steps(self, cfg: &BcConfig) -> usize {
        match self {
            BcStage::S1 => cfg.steps.s1,
            BcStage::S2 => cfg.steps.s2,
            BcStage::AblationA => cfg.steps.ablation_a,
        }
    }

    pub fn warmup(self, cfg: &BcConfig) -> usize {
        match self {
            BcStage::S1 => 0,
            BcStage::S2 => cfg.steps.s2_warmup,
            BcStage::AblationA => cfg.steps.ablation_a_warmup,
        }
    }
}

impl BcConfig {
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

struct Lane {
    game: Option<GameStream>,
    id: Option<GameId>,
    fresh: bool,
}

impl Lane {
    fn empty() -> Self {
        Self { game: None, id: None, fresh: true }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct LaneState {
    pub id: GameId,
    pub pos: usize,
    pub z_present: bool,
    pub sym: u8,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct FeedState {
    pub lanes: Vec<Option<LaneState>>,
    pub source: Option<SourcePos>,
    pub rng: u64,
    pub games: usize,
    pub invalid_labels: usize,
    #[serde(default)]
    pub files: String,
    #[serde(default)]
    pub no_result_games: usize,
}

pub struct Batch {
    pub chunks: Vec<Vec<TrainStep>>,
    pub planes: Vec<Vec<Planes>>,
    pub fresh: Vec<bool>,
    pub results: Vec<f32>,
    pub value_weights: Vec<f32>,
    pub enemy_races: Vec<String>,
    pub game_ids: Vec<String>,
    pub game_keys: Vec<Option<GameId>>,
    pub perspective_slots: Vec<u8>,
    pub state: FeedState,
}

impl Batch {
    pub fn parts(&self) -> (&[Vec<TrainStep>], &[bool], &[f32]) {
        assert!(self.planes.is_empty(), "041-R18: a packed batch is read by the trainer only");
        (&self.chunks, &self.fresh, &self.results)
    }

    pub fn value_weights(&self) -> &[f32] {
        &self.value_weights
    }
}

pub struct Feeder {
    lanes: Vec<Lane>,
    rng: Rng,
    seq_len: usize,
    z_drop: f64,
    augment: bool,
    race_mask: f64,
    seed: u64,
    games: usize,
    invalid_labels: usize,
    no_result_games: usize,
    files: String,
    pub drain: bool,
    maps: Arc<MapStore>,
    pack: bool,
    threads: usize,
    chain_start: Vec<String>,
}

impl Feeder {
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
            threads: feed_threads(),
            chain_start: cfg.data.chain_start_sources.clone().unwrap_or_default(),
        }
    }

    pub fn set_chain_start(&mut self, sources: Vec<String>) {
        self.chain_start = sources;
    }

    pub fn set_pack(&mut self, pack: bool) {
        self.pack = pack;
    }

    pub fn set_threads(&mut self, threads: usize) {
        self.threads = threads.max(1);
    }

    pub fn with_maps(mut self, maps: Arc<MapStore>) -> Self {
        self.maps = maps;
        self
    }

    pub fn set_maps(&mut self, maps: Arc<MapStore>) {
        self.maps = maps;
    }

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
            let g = GameStream::new(game, z_present, sym, &self.maps)?.with_race_mask(self.race_mask, self.seed).with_chain_start(&self.chain_start);
            self.no_result_games += (g.value_weight == 0.0) as usize;
            self.lanes[i] = Lane { game: Some(g), id: src.last_id(), fresh: true };
        }
        Ok(true)
    }

    pub fn next_batch(&mut self, src: &mut dyn StepSource, pos: Option<SourcePos>) -> Result<Option<Batch>> {
        if !self.refill(src)? {
            return Ok(None);
        }
        let mut chunks = Vec::with_capacity(self.lanes.len());
        let mut fresh = Vec::with_capacity(self.lanes.len());
        let mut results = Vec::with_capacity(self.lanes.len());
        let mut value_weights = Vec::with_capacity(self.lanes.len());
        let mut enemy_races = Vec::with_capacity(self.lanes.len());
        let mut game_ids = Vec::with_capacity(self.lanes.len());
        let mut game_keys = Vec::with_capacity(self.lanes.len());
        let mut perspective_slots = Vec::with_capacity(self.lanes.len());
        let mut planes = Vec::new();
        let (seq_len, pack) = (self.seq_len, self.pack);
        let made = par_lanes(&mut self.lanes, self.threads, |lane| {
            lane.game.as_mut().map(|g| {
                let (mut chunk, invalid) = g.next_chunk(seq_len);
                let p: Vec<Planes> = if pack { chunk.iter_mut().map(|s| Planes::take(&mut s.obs)).collect() } else { Vec::new() };
                (chunk, invalid, p)
            })
        });
        for (lane, made) in self.lanes.iter_mut().zip(made) {
            let (Some(g), Some((chunk, invalid, p))) = (lane.game.as_mut(), made) else {
                if self.pack {
                    planes.push(Vec::new());
                }
                chunks.push(Vec::new());
                fresh.push(true);
                results.push(0.0);
                value_weights.push(0.0);
                enemy_races.push(String::new());
                game_ids.push(String::new());
                game_keys.push(None);
                perspective_slots.push(0);
                continue;
            };
            self.invalid_labels += invalid;
            if self.pack {
                planes.push(p);
            }
            chunks.push(chunk);
            fresh.push(lane.fresh);
            results.push(g.result);
            value_weights.push(g.value_weight);
            enemy_races.push(g.enemy_race().to_string());
            game_ids.push(g.replay_sha256().to_string());
            game_keys.push(lane.id);
            perspective_slots.push(g.perspective_slot());
            lane.fresh = false;
            if g.remaining() == 0 {
                lane.game = None;
                lane.id = None;
            }
        }
        Ok(Some(Batch { chunks, planes, fresh, results, value_weights, enemy_races, game_ids, game_keys, perspective_slots, state: self.state(pos) }))
    }

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
        let saved: Vec<&LaneState> = st.lanes.iter().flatten().collect();
        let ids: Vec<GameId> = saved.iter().map(|s| s.id).collect();
        let mut games = src.games(&ids)?.into_iter();
        for (lane, saved) in self.lanes.iter_mut().zip(st.lanes.iter().chain(std::iter::repeat(&None))) {
            *lane = Lane::empty();
            let Some(s) = saved else { continue };
            let game = games.next().expect("one game per saved lane");
            let mut g = GameStream::new(game, s.z_present, Sym::augment()[s.sym as usize % 4], &self.maps)?.with_race_mask(self.race_mask, self.seed).with_chain_start(&self.chain_start);
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

fn par_lanes<T: Send>(lanes: &mut [Lane], threads: usize, f: impl Fn(&mut Lane) -> T + Sync) -> Vec<T> {
    let threads = threads.clamp(1, lanes.len().max(1));
    if threads == 1 {
        return lanes.iter_mut().map(&f).collect();
    }
    let per = lanes.len().div_ceil(threads);
    std::thread::scope(|s| {
        let hs: Vec<_> = lanes.chunks_mut(per).map(|run| s.spawn(|| run.iter_mut().map(&f).collect::<Vec<T>>())).collect();
        hs.into_iter().flat_map(|h| h.join().expect("feeder lane thread panicked")).collect()
    })
}

pub fn feed_threads() -> usize {
    match std::env::var("URANIUM_BC_FEED_THREADS").ok().and_then(|v| v.trim().parse::<usize>().ok()) {
        Some(n) if n > 0 => n,
        _ => FEED_THREADS_DEFAULT,
    }
}

pub const FEED_THREADS_DEFAULT: usize = 3;

pub fn micro_lanes_override() -> Result<Option<usize>> {
    parse_micro_lanes(std::env::var("URANIUM_BC_MICRO_LANES").ok().as_deref())
}

pub fn parse_micro_lanes(v: Option<&str>) -> Result<Option<usize>> {
    match v.map(str::trim) {
        Some(t) if !t.is_empty() => Ok(Some(t.parse().with_context(|| format!("URANIUM_BC_MICRO_LANES={t:?}: expected a lane count (0 = all lanes)"))?)),
        _ => Ok(None),
    }
}

pub const PREFETCH_DEPTH_DEFAULT: usize = 2;

pub fn prefetch_depth() -> Result<usize> {
    match std::env::var("URANIUM_BC_PREFETCH") {
        Ok(v) if !v.trim().is_empty() => v.trim().parse().with_context(|| format!("URANIUM_BC_PREFETCH={v:?}: expected a batch count (0 = no data thread)")),
        _ => Ok(PREFETCH_DEPTH_DEFAULT),
    }
}

pub fn entity_chunk() -> Result<usize> {
    match std::env::var("URANIUM_BC_ENTITY_CHUNK") {
        Ok(v) if !v.trim().is_empty() => v.trim().parse().with_context(|| format!("URANIUM_BC_ENTITY_CHUNK={v:?}: expected a row count (0 = off)")),
        _ => Ok(0),
    }
}

pub fn pack_batch() -> Result<bool> {
    match std::env::var("URANIUM_BC_PACK_BATCH").as_deref().map(str::trim) {
        Ok("1") => Ok(true),
        Ok("" | "0") | Err(_) => Ok(false),
        Ok(v) => anyhow::bail!("URANIUM_BC_PACK_BATCH={v:?}: expected 0 or 1"),
    }
}

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

    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Result<Option<Batch>> {
        match &mut self.0 {
            PrefetchMode::Thread { rx, .. } => rx.recv().map_err(|_| anyhow::anyhow!("data thread stopped"))?,
            PrefetchMode::Inline { feeder, src } => feeder.next_batch_from(src),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct RunIdent {
    pub stage: String,
    pub config: String,
    pub dataset: String,
}

const IDENT_KEY: &str = "run";

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

pub fn read_ident(dir: &Path) -> Result<Option<RunIdent>> {
    let Some(state) = ckpt::load_state(dir)? else { return Ok(None) };
    state.extra.get(IDENT_KEY).map(|v| serde_json::from_value(v.clone())).transpose().map_err(Into::into)
}

pub fn check_ident(dir: &Path, want: &RunIdent) -> Result<()> {
    let Some(have) = read_ident(dir)? else {
        uranium_store::elog!("[bc] WARNING: {} has no stage / config / dataset fingerprint (written before B-3); resuming unchecked", dir.display());
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

pub fn subsample_counts(scan: &[Vec<GameSummary>], sel: Selection, sub: &Subsample, seed: u64) -> std::collections::BTreeMap<String, (usize, usize)> {
    let mut per = std::collections::BTreeMap::<String, (usize, usize)>::new();
    for g in scan.iter().flatten().filter(|g| crate::split::selects(sel, &g.meta)) {
        let e = per.entry(g.meta.src.clone()).or_default();
        e.0 += 1;
        e.1 += sub.keeps(&g.meta, seed) as usize;
    }
    per
}

#[derive(Serialize, Debug, Clone, Default)]
pub struct EpochCount {
    pub updates: u64,
    pub games: usize,
    pub steps: u64,
    pub by_source: crate::mix::SourceRaceCounts,
}

#[derive(Serialize, Debug)]
pub struct EpochTable {
    pub lanes: usize,
    pub seq_len: usize,
    pub seed: u64,
    pub s1: EpochCount,
    pub s2: EpochCount,
    pub ablation_a: EpochCount,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub s2_mix: Option<crate::mix::MixShares>,
}

pub fn epoch_table(dirs: &[PathBuf], cfg: &BcConfig) -> Result<EpochTable> {
    epoch_table_groups(&[dirs.to_vec()], cfg)
}

pub fn scan_groups(groups: &[Vec<PathBuf>]) -> Result<std::collections::HashMap<PathBuf, Vec<GameSummary>>> {
    let mut out = std::collections::HashMap::new();
    for g in groups.iter().filter(|g| !g.is_empty()) {
        let src = ShardSource::from_dirs(g)?;
        let scan = crate::data::scan_cached(&src)?;
        out.extend(src.files().iter().cloned().zip(scan));
    }
    Ok(out)
}

pub fn epoch_table_groups(groups: &[Vec<PathBuf>], cfg: &BcConfig) -> Result<EpochTable> {
    let dirs: Vec<PathBuf> = groups.iter().flatten().cloned().collect();
    let base = ShardSource::from_dirs(&dirs)?;
    let by_path = scan_groups(groups)?;
    let scan: Vec<&Vec<GameSummary>> = base.files().iter().map(|f| &by_path[f]).collect();
    let dirs = &dirs[..];
    let count = |sel: Selection| -> Result<EpochCount> {
        let src = cfg.train_source(dirs, sel)?;
        let mut kept: Vec<GameSummary> = Vec::new();
        for (k, f) in src.files().iter().enumerate() {
            kept.extend(src.order_file(k, by_path[f].clone()));
        }
        let lens: Vec<usize> = kept.iter().map(|g| g.steps).collect();
        let games = lens.iter().filter(|&&l| l > 0).count();
        let steps = lens.iter().map(|&l| l as u64).sum();
        let by_source = crate::mix::count_by_source(&kept);
        Ok(EpochCount { updates: epoch_updates(lens, cfg.lanes, cfg.seq_len), games, steps, by_source })
    };
    let s1 = count(Selection::S1)?;
    let s2 = count(Selection::S2)?;
    let s2_mix = cfg.data.subsample.as_ref().map(|sub| {
        let all: Vec<GameSummary> = scan.iter().copied().flatten().filter(|g| crate::split::selects(Selection::S2, &g.meta)).cloned().collect();
        crate::mix::MixShares::with_sources(&s2.by_source, &crate::mix::count_by_source(&all), &sub.keep_sources, &sub.share_sources())
    });
    Ok(EpochTable { lanes: cfg.lanes, seq_len: cfg.seq_len, seed: cfg.data.seed, s1, ablation_a: s2.clone(), s2, s2_mix })
}

pub fn eval_games(src: &ShardSource, scan: &[Vec<GameSummary>], n: usize, seed: u64) -> Vec<crate::data::GameId> {
    let counts: Vec<usize> = scan.iter().enumerate().map(|(k, items)| src.order_file(k, items.clone()).len()).collect();
    sample_games(&counts, n, seed)
}

pub fn eval_game_metas(src: &ShardSource, scan: &[Vec<GameSummary>], n: usize, seed: u64, target_only: bool) -> Vec<uranium_replay::shard::GameMeta> {
    let per_file: Vec<Vec<GameSummary>> = scan
        .iter()
        .enumerate()
        .map(|(k, items)| {
            let mut v = src.order_file(k, items.clone());
            v.retain(|g| !target_only || crate::split::is_target_map(&g.meta));
            v
        })
        .collect();
    let counts: Vec<usize> = per_file.iter().map(Vec::len).collect();
    sample_games(&counts, n, seed).into_iter().map(|id| per_file[id.file][id.game].meta.clone()).collect()
}

#[derive(Debug, Clone, PartialEq)]
pub struct HostGroup {
    pub b: usize,
    pub t: usize,
    pub obs: crate::collate::HostObsBatch,
    pub labels: crate::collate::HostActionBatch,
    pub privileged: crate::collate::HostPrivBatch,
    pub weights: Vec<f32>,
    pub results: Vec<f32>,
    pub value_w: Vec<f32>,
    pub keep: Vec<bool>,
    pub real: Vec<bool>,
    pub priv_present: Vec<bool>,
}

pub const GROUP_FIELDS: [&str; 28] = [
    "entity_cat", "entity_cont", "entity_mask", "entity_cell", "entity_pos", "spatial", "scalar", "scalar_cat", "z_build", "z_units", "z_present", "type_mask", "unit_cap",
    "loc_class_mask", "is_building", "action", "delay", "queued", "selected", "target_unit", "target_loc", "privileged", "weights", "results", "value_w", "keep", "real",
    "priv_present",
];

impl HostGroup {
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
        b(&mut h, &self.real);
        b(&mut h, &self.priv_present);
        h.finalize().iter().map(|x| format!("{x:02x}")).collect()
    }
}

impl Batch {
    pub fn lanes(&self) -> usize {
        self.chunks.len()
    }

    pub fn valid(&self) -> f32 {
        self.chunks.iter().flat_map(|c| c.iter().map(|s| s.weight)).sum::<f32>().max(1.0)
    }

    pub fn value_valid(&self) -> f32 {
        self.chunks.iter().zip(&self.value_weights).map(|(c, vw)| c.iter().map(|s| s.weight * vw).sum::<f32>()).sum::<f32>().max(1.0)
    }

    pub fn group(&self, lanes: std::ops::Range<usize>, t: usize, pad: usize, max_select: usize) -> HostGroup {
        let b = lanes.len();
        let mut rows: Vec<&TrainStep> = Vec::with_capacity(b * t);
        let mut weights = Vec::with_capacity(b * t);
        let mut results = Vec::with_capacity(b * t);
        let mut value_w = Vec::with_capacity(b * t);
        let mut keep = Vec::with_capacity(b);
        let mut real_rows = Vec::with_capacity(b * t);
        let mut priv_present = Vec::with_capacity(b * t);
        let mut packed = Vec::new();
        let filler = self.chunks.iter().position(|c| !c.is_empty()).expect("a batch has a lane with steps");
        for l in lanes {
            let (src, chunk) = if self.chunks[l].is_empty() { (filler, &self.chunks[filler]) } else { (l, &self.chunks[l]) };
            let real = self.chunks[l].len();
            keep.push(!self.fresh[l]);
            for k in 0..t {
                let i = k.min(chunk.len() - 1).min(real.saturating_sub(1));
                let s = &chunk[i];
                if !self.planes.is_empty() {
                    packed.push((src, i));
                }
                rows.push(s);
                let w = if k < real { s.weight } else { 0.0 };
                weights.push(w);
                results.push(self.results[l]);
                value_w.push(w * self.value_weights[l]);
                real_rows.push(k < real);
                priv_present.push(s.has_privileged);
            }
        }
        let obs: Vec<_> = rows.iter().map(|s| &s.obs).collect();
        let obs = if packed.is_empty() {
            crate::collate::obs_batch(&obs, pad)
        } else {
            let planes: Vec<_> = packed.iter().map(|&(src, i)| &self.planes[src][i]).collect();
            crate::collate::obs_batch_packed(&obs, &planes, pad, feed_threads())
        };
        let labels: Vec<_> = rows.iter().map(|s| &s.label).collect();
        let labels = crate::collate::action_batch(&labels, obs.n, max_select);
        let privs: Vec<_> = rows.iter().map(|s| &s.privileged).collect();
        let privileged = crate::collate::priv_batch(&privs);
        HostGroup { b, t, obs, labels, privileged, weights, results, value_w, keep, real: real_rows, priv_present }
    }

    pub fn lane_steps(&self) -> Vec<usize> {
        self.chunks.iter().map(|c| c.len()).collect()
    }

    pub fn lane_priv_steps(&self, t: usize) -> Vec<usize> {
        self.chunks.iter().map(|c| c.iter().take(t).filter(|s| s.has_privileged).count()).collect()
    }

    pub fn label_extras(&self, lanes: std::ops::Range<usize>, t: usize) -> (Vec<bool>, Vec<i32>) {
        let (mut has_loc, mut sel_len) = (Vec::new(), Vec::new());
        for l in lanes {
            let c = &self.chunks[l];
            for k in 0..t {
                let lab = c.get(k).map(|s| &s.label);
                has_loc.push(lab.is_some_and(|x| x.target_loc.is_some()));
                sel_len.push(lab.map_or(0, |x| x.selected.len() as i32));
            }
        }
        (has_loc, sel_len)
    }
}
