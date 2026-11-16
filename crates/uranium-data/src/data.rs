use std::collections::{HashSet, VecDeque};
use anyhow::Result;
use uranium_env::features::extract::{action_table, dims};
use uranium_env::features::*;
use uranium_env::features::label::HostLabel;
use uranium_replay::shard::{GameMeta, ShardGame, result_source};
pub use uranium_env::maps::family_index;
pub use uranium_env::features::label::{UNKNOWN_REASON, acting_ids, host_label, host_label_check};

mod chosen;
mod label_diag;
mod maps;
mod scan;
mod source;
mod stream;

pub use source::{ShardSource, shuffle, shard_keys, shard_files};
pub use maps::MapStore;
pub use scan::{GameSummary, scan_files, check_maps, scan_cache_dir, scan_cached, scan_cached_in, dataset_fingerprint};
pub use chosen::{CHOSEN_PREFETCH_FILES, CHOSEN_PREFETCH_FILES_CUT, ChosenGames, cut_game};
pub use label_diag::{LabelStats, label_check};
pub use stream::{enemy_race_masked, GameStream, chained_step, CHAIN_START_SOURCES, check_chain_start_sources, prepare};
pub(crate) use stream::{SUBSAMPLE_SALT, game_unit};

pub trait StepSource {
    fn next_game(&mut self) -> Result<Option<ShardGame>>;

    fn last_id(&self) -> Option<GameId> {
        None
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct GameId {
    pub file: usize,
    pub game: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SourcePos {
    pub next: GameId,
}

pub trait HasMeta {
    fn meta(&self) -> &uranium_replay::shard::GameMeta;
}

impl HasMeta for ShardGame {
    fn meta(&self) -> &uranium_replay::shard::GameMeta {
        &self.meta
    }
}

pub struct VecSource(pub VecDeque<ShardGame>);

impl StepSource for VecSource {
    fn next_game(&mut self) -> Result<Option<ShardGame>> {
        Ok(self.0.pop_front())
    }
}

pub fn epoch_updates(lens: impl IntoIterator<Item = usize>, lanes: usize, seq_len: usize) -> u64 {
    let mut lens = lens.into_iter().filter(|&l| l > 0);
    let mut rem = vec![0usize; lanes];
    let mut updates = 0u64;
    loop {
        for r in &mut rem {
            if *r == 0 {
                match lens.next() {
                    Some(l) => *r = l,
                    None => return updates,
                }
            }
        }
        for r in &mut rem {
            *r -= (*r).min(seq_len);
        }
        updates += 1;
    }
}

#[derive(Debug, Clone)]
pub struct TrainStep {
    pub obs: Observation,
    pub label: HostLabel,
    pub privileged: PrivilegedTensors,
    pub has_privileged: bool,
    pub weight: f32,
    pub frame: u32,
    pub surrender_rule: bool,
}

#[derive(Debug, Clone)]
pub struct GameSeq {
    pub steps: Vec<TrainStep>,
    pub result: f32,
    pub invalid_labels: usize,
}

pub fn value_weight(meta: &GameMeta) -> f32 {
    if meta.result_source == result_source::NONE { 0.0 } else { 1.0 }
}

pub fn sample_games(counts: &[usize], n: usize, seed: u64) -> Vec<GameId> {
    let mut ids: Vec<GameId> = counts.iter().enumerate().flat_map(|(file, &c)| (0..c).map(move |game| GameId { file, game })).collect();
    shuffle(&mut ids, &mut Rng::new(seed));
    ids.truncate(n);
    ids.sort_by_key(|g| (g.file, g.game));
    ids
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq)]
pub struct ZEntry {
    pub build: Vec<u16>,
    pub units: Vec<f32>,
}

pub fn replay_z(game: &ShardGame) -> Strategy {
    let table = action_table();
    let build: Vec<u16> = game
        .steps
        .iter()
        .map(|s| s.action_label.action)
        .filter(|&a| table.actions.get(a as usize).is_some_and(|x| matches!(x.command.as_str(), "Build" | "BuildAddon" | "Train" | "Research" | "Upgrade")))
        .take(20)
        .collect();
    let d = dims();
    let me = game.game_info.self_player;
    let mut seen: HashSet<u32> = HashSet::new();
    let mut counts = vec![0f32; d.z_units()];
    for s in &game.steps {
        for u in s.raw_frame.units.iter().filter(|u| u.owner == me) {
            if let Some(i) = d.terran_types.iter().position(|&t| t == u.unit_type) {
                if seen.insert(u.id) {
                    counts[i] += 1.0;
                }
            }
        }
    }
    let total: f32 = counts.iter().sum();
    if total > 0.0 {
        counts.iter_mut().for_each(|c| *c /= total);
    }
    Strategy { build, units: counts }
}

#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    pub fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    pub fn state(&self) -> u64 {
        self.0
    }
    pub fn from_state(s: u64) -> Self {
        Self(s)
    }
}
