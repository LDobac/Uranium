//! BC data (041): one interface over stored shards and replay streaming (041-R9). A game turns
//! into a sequence of (observation, label, privileged, weight) steps by running the same feature
//! extraction as RL and SC:R (030), so stored and streamed games give identical inputs (040-AC6).

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;

use anyhow::{Context, Result};
use uranium_env::actions::{CAPABILITIES, LOC_CLASSES, TargetKind};
use uranium_env::features::augment::{Sym, transform, transform_target_loc};
use uranium_env::features::extract::{action_table, dims};
use uranium_env::features::*;
use uranium_model::collate::HostLabel;
use uranium_replay::label::Label;
use uranium_replay::shard::{ShardGame, read};

/// One replay game from any source (041-R9).
pub trait StepSource {
    /// Next game, or `None` when the source is exhausted.
    fn next_game(&mut self) -> Result<Option<ShardGame>>;
}

/// Stored shards (`shard_*.bin.zst`) in a directory, in file-name order.
pub struct ShardSource {
    files: VecDeque<PathBuf>,
    games: VecDeque<ShardGame>,
}

impl ShardSource {
    pub fn new(dir: &std::path::Path) -> Result<Self> {
        let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
            .with_context(|| dir.display().to_string())?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("shard_") && n.ends_with(".bin.zst")))
            .collect();
        files.sort();
        Ok(Self { files: files.into(), games: VecDeque::new() })
    }
}

impl StepSource for ShardSource {
    fn next_game(&mut self) -> Result<Option<ShardGame>> {
        while self.games.is_empty() {
            let Some(f) = self.files.pop_front() else { return Ok(None) };
            self.games.extend(read(&f)?.games);
        }
        Ok(self.games.pop_front())
    }
}

/// In-memory games (tests, replay streaming adapters).
pub struct VecSource(pub VecDeque<ShardGame>);

impl StepSource for VecSource {
    fn next_game(&mut self) -> Result<Option<ShardGame>> {
        Ok(self.0.pop_front())
    }
}

/// A training step: inputs, label and loss weight (0 = label inconsistent with the masks).
#[derive(Debug, Clone)]
pub struct TrainStep {
    pub obs: Observation,
    pub label: HostLabel,
    pub privileged: PrivilegedTensors,
    pub weight: f32,
}

/// A prepared game.
#[derive(Debug, Clone)]
pub struct GameSeq {
    pub steps: Vec<TrainStep>,
    /// 041-R7 win target: +1 win, -1 loss, 0 unknown.
    pub result: f32,
    pub invalid_labels: usize,
}

/// 040-R15 family -> `scalar_cat` map_family index.
pub fn family_index(family: &str) -> i32 {
    match family {
        "fighting_spirit" => 0,
        "circuit_breaker" => 1,
        _ => 2,
    }
}

fn map_precompute(map_hash: &str) -> Option<MapPrecompute> {
    let path = uranium_replay::pipeline::maps_dir().join(format!("{map_hash}.toml"));
    toml::from_str(&std::fs::read_to_string(path).ok()?).ok()
}

/// 030 §5 BC z: the replay's first 20 Build / Train / Research / Upgrade actions and the
/// composition of every own Terran unit that appeared.
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

fn bucket_index(delay: u32) -> usize {
    DELAY_BUCKETS.iter().rposition(|&b| b <= delay).unwrap_or(0)
}

/// Replay label (unit ids, tile) -> entity indices of this observation, checked against the masks
/// (031-R4). `None` when the label cannot be expressed or violates a mask.
pub fn host_label(l: &Label, o: &Observation) -> (HostLabel, bool) {
    let index: HashMap<u32, usize> = o.entity_ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
    let mut selected: Vec<usize> = l.selected.iter().filter_map(|id| index.get(id).copied()).collect();
    selected.sort_unstable();
    selected.dedup();
    let target_unit = l.target_unit.and_then(|id| index.get(&id).copied());
    let target_loc = l.target_tile.map(|(x, y)| y as usize * MAP + x as usize);
    let h = HostLabel { action: l.action, delay_bucket: bucket_index(l.delay), queued: l.queued, selected, target_unit, target_loc };
    let Some(a) = action_table().actions.get(l.action as usize) else { return (h, false) };
    let k = CAPABILITIES.len();
    let cap = |i: usize, name: &str| o.masks.unit_cap[i * k + CAPABILITIES.iter().position(|c| *c == name).unwrap()];
    let mut ok = o.masks.type_mask[a.id as usize];
    if a.uses_select {
        ok &= !h.selected.is_empty()
            && h.selected.len() == l.selected.len()
            && h.selected.len() <= a.max_select as usize
            && h.selected.iter().all(|&i| cap(i, &a.select_cap));
    }
    match a.target {
        TargetKind::Unit => ok &= h.target_unit.is_some_and(|i| cap(i, &a.target_cap)),
        TargetKind::Loc => {
            let lc = LOC_CLASSES.iter().position(|c| *c == a.loc_class).unwrap();
            ok &= h.target_loc.is_some_and(|t| t < CELLS && o.masks.loc_class_mask[lc * CELLS + t]);
        }
        TargetKind::None => {}
    }
    (h, ok)
}

/// Tiny deterministic RNG for data choices (z drop, symmetry).
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
}

/// Feature extraction over one game in order (memory carries across steps), producing training
/// steps a chunk at a time so only the current BPTT window is held in memory (041-R2: a whole
/// game of spatial features is gigabytes). `z_present` and `sym` are chosen by the caller (041-R8,
/// 040-R7).
pub struct GameStream {
    game: ShardGame,
    map: MapInfo,
    mem: AgentMemory,
    z: Option<Strategy>,
    sym: Sym,
    pos: usize,
    zero_priv: PrivilegedTensors,
    /// 041-R7 win target: +1 win, -1 loss, 0 unknown.
    pub result: f32,
}

impl GameStream {
    pub fn new(game: ShardGame, z_present: bool, sym: Sym) -> Result<Self> {
        let map = MapInfo::new(&game.game_info, family_index(&game.meta.map_family), map_precompute(&game.meta.map_hash).as_ref())?;
        let mem = AgentMemory::new(&game.game_info);
        let z = z_present.then(|| replay_z(&game));
        let zero_priv = PrivilegedTensors { summary: vec![0.0; dims().privileged()] };
        let result = game.meta.result as f32;
        Ok(Self { game, map, mem, z, sym, pos: 0, zero_priv, result })
    }

    /// Steps not yet produced.
    pub fn remaining(&self) -> usize {
        self.game.steps.len() - self.pos
    }

    /// Up to `n` next steps and how many of them have labels inconsistent with the masks.
    pub fn next_chunk(&mut self, n: usize) -> (Vec<TrainStep>, usize) {
        let end = (self.pos + n).min(self.game.steps.len());
        let mut steps = Vec::with_capacity(end - self.pos);
        let mut invalid = 0;
        for s in &self.game.steps[self.pos..end] {
            let o = extract(&s.raw_frame, &mut self.mem, &self.map, self.z.as_ref());
            let (mut label, ok) = host_label(&s.action_label, &o);
            invalid += !ok as usize;
            // Memory sees what the policy did, like in RL (030 §4 previous action, §1 last action).
            self.mem.record_action(label.action, label.delay_bucket, &s.action_label.selected);
            let o = if self.sym == Sym::IDENTITY {
                o
            } else {
                if let Some(t) = label.target_loc {
                    label.target_loc = Some(transform_target_loc(label.action, t, o.map_size, o.transposed, self.sym));
                }
                transform(&o, self.sym)
            };
            let privileged = s.privileged.as_ref().map_or_else(|| self.zero_priv.clone(), privileged);
            steps.push(TrainStep { obs: o, label, privileged, weight: if ok { 1.0 } else { 0.0 } });
        }
        self.pos = end;
        (steps, invalid)
    }
}

/// Whole game at once (tests, short games): `GameStream` drained in one chunk.
pub fn prepare(game: &ShardGame, z_present: bool, sym: Sym) -> Result<GameSeq> {
    let mut stream = GameStream::new(game.clone(), z_present, sym)?;
    let (steps, invalid_labels) = stream.next_chunk(usize::MAX);
    Ok(GameSeq { steps, result: stream.result, invalid_labels })
}
