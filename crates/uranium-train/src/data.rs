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

/// Stored shards (`shard_*.bin.zst`) under one or more dataset folders, subfolders included (a
/// full build writes `<src>/<part>/shard_*.bin.zst`, 040-R13). Folders are read in the given
/// order, each in path order, so the game order is deterministic. `select` keeps one split/stage
/// (041-R12, R13); `shuffled` permutes the shard files and the games inside each file with a fixed
/// seed (041-R13).
pub struct ShardSource {
    files: VecDeque<PathBuf>,
    games: VecDeque<ShardGame>,
    select: Option<crate::split::Selection>,
    rng: Option<Rng>,
}

impl ShardSource {
    pub fn new(dir: &std::path::Path) -> Result<Self> {
        Self::from_dirs(std::slice::from_ref(&dir.to_path_buf()))
    }

    pub fn from_dirs(dirs: &[PathBuf]) -> Result<Self> {
        let mut files = Vec::new();
        for d in dirs {
            files.extend(shard_files(d)?);
        }
        Ok(Self { files: files.into(), games: VecDeque::new(), select: None, rng: None })
    }

    pub fn select(mut self, sel: crate::split::Selection) -> Self {
        self.select = Some(sel);
        self
    }

    pub fn shuffled(mut self, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        shuffle(self.files.make_contiguous(), &mut rng);
        self.rng = Some(rng);
        self
    }
}

/// Fisher-Yates.
fn shuffle<T>(v: &mut [T], rng: &mut Rng) {
    for i in (1..v.len()).rev() {
        v.swap(i, (rng.next() % (i as u64 + 1)) as usize);
    }
}

/// Shard files under `dir` (recursive), sorted by path.
pub fn shard_files(dir: &std::path::Path) -> Result<Vec<PathBuf>> {
    fn walk(dir: &std::path::Path, out: &mut Vec<PathBuf>) -> Result<()> {
        for e in std::fs::read_dir(dir).with_context(|| dir.display().to_string())? {
            let e = e?;
            let p = e.path();
            if e.file_type()?.is_dir() {
                walk(&p, out)?;
            } else if p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("shard_") && n.ends_with(".bin.zst")) {
                out.push(p);
            }
        }
        Ok(())
    }
    let mut files = Vec::new();
    walk(dir, &mut files)?;
    files.sort();
    Ok(files)
}

impl StepSource for ShardSource {
    fn next_game(&mut self) -> Result<Option<ShardGame>> {
        while self.games.is_empty() {
            let Some(f) = self.files.pop_front() else { return Ok(None) };
            let mut games = read(&f).with_context(|| f.display().to_string())?.games;
            if let Some(sel) = self.select {
                games.retain(|g| crate::split::selects(sel, &g.meta));
            }
            if let Some(rng) = &mut self.rng {
                shuffle(&mut games, rng);
            }
            self.games.extend(games);
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

pub use uranium_env::maps::family_index;

fn map_precompute(map_hash: &str) -> Option<MapPrecompute> {
    uranium_env::maps::load_precompute(&uranium_env::maps::maps_dir(), map_hash)
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
/// (031-R4). `false` when the label cannot be expressed or violates a mask.
pub fn host_label(l: &Label, o: &Observation) -> (HostLabel, bool) {
    let (h, why) = host_label_check(l, o);
    (h, why.is_none())
}

/// `host_label` with the first failing check (label diagnostics, 034-AC3).
pub fn host_label_check(l: &Label, o: &Observation) -> (HostLabel, Option<&'static str>) {
    let index: HashMap<u32, usize> = o.entity_ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
    let mut selected: Vec<usize> = l.selected.iter().filter_map(|id| index.get(id).copied()).collect();
    selected.sort_unstable();
    selected.dedup();
    let target_unit = l.target_unit.and_then(|id| index.get(&id).copied());
    let target_loc = l.target_tile.map(|(x, y)| y as usize * MAP + x as usize);
    let mut h = HostLabel { action: l.action, delay_bucket: bucket_index(l.delay), queued: l.queued, selected, target_unit, target_loc };
    let Some(a) = action_table().actions.get(l.action as usize) else { return (h, Some("unknown_action")) };
    let k = CAPABILITIES.len();
    let cap = |i: usize, name: &str| o.masks.unit_cap[i * k + CAPABILITIES.iter().position(|c| *c == name).unwrap()];
    let visible = h.selected.len();
    // 041-R11: the game runs a command on each selected unit that can take it and skips the rest
    // (OpenBW `action_*`), so the label keeps only those; it is void when none can.
    if a.uses_select {
        h.selected.retain(|&i| cap(i, &a.select_cap));
    }
    let why = if !o.masks.type_mask[a.id as usize] {
        Some("type_mask")
    } else if a.uses_select && visible == 0 {
        Some("select_empty")
    } else if a.uses_select && visible != l.selected.len() {
        Some("select_not_visible")
    } else if a.uses_select && h.selected.is_empty() {
        Some("select_cap")
    } else if a.uses_select && h.selected.len() > a.max_select as usize {
        Some("select_too_many")
    } else {
        match a.target {
            TargetKind::Unit if h.target_unit.is_none() => Some("target_unit_missing"),
            TargetKind::Unit if !h.target_unit.is_some_and(|i| cap(i, &a.target_cap)) => Some("target_unit_cap"),
            TargetKind::Loc => {
                let lc = LOC_CLASSES.iter().position(|c| *c == a.loc_class).unwrap();
                (!h.target_loc.is_some_and(|t| t < CELLS && o.masks.loc_class_mask[lc * CELLS + t])).then_some("target_loc")
            }
            _ => None,
        }
    };
    (h, why)
}

/// Unit ids of the label's (capability-filtered) selection.
fn acting_ids(h: &HostLabel, o: &Observation) -> Vec<u32> {
    h.selected.iter().map(|&i| o.entity_ids[i]).collect()
}

/// Label diagnostics (034-AC3): labels failing `host_label_check` per (action, reason), labels per
/// action, and for `select_cap` failures the unit types in the selection that lack the capability.
#[derive(Debug, Default)]
pub struct LabelStats {
    pub fails: std::collections::BTreeMap<(String, &'static str), usize>,
    pub totals: std::collections::BTreeMap<String, usize>,
    /// "<action> <- <type>+<type> (kept|void)" -> count: selected unit types without the
    /// capability, dropped from a label that stays valid or leaving none (041-R11).
    pub cap_detail: std::collections::BTreeMap<String, usize>,
    /// "<action> <- <condition> (new|old)" -> count for `type_mask` failures of Train / Research /
    /// Upgrade: which prerequisite fails, and whether it already failed at the previous step
    /// (old) or only became false since (new).
    pub type_detail: std::collections::BTreeMap<String, usize>,
}

/// First failing Train / Research / Upgrade prerequisite of `a` in `me` (034-R3 order).
fn prereq_fail(a: &uranium_env::actions::Action, me: &uranium_proto::PlayerState) -> Option<&'static str> {
    let t = uranium_env::types::TypeTables::get();
    let afford = |m: i32, g: i32| me.minerals >= m && me.gas >= g;
    match a.command.as_str() {
        "Train" => {
            let u = t.unit(a.unit_type?)?;
            let supply = u.supply_required == 0 || me.supply_total as i32 - me.supply_used as i32 >= u.supply_required;
            if !afford(u.minerals, u.gas) {
                Some("afford")
            } else if !supply {
                Some("supply")
            } else {
                None
            }
        }
        "Research" => {
            let tech = a.tech?;
            let r = &t.techs[tech as usize];
            if me.tech & (1u64 << tech) != 0 {
                Some("researched")
            } else if me.researching & (1u64 << tech) != 0 {
                Some("researching")
            } else if !afford(r.minerals, r.gas) {
                Some("afford")
            } else {
                None
            }
        }
        "Upgrade" => {
            let up = a.upgrade?;
            let u = &t.upgrades[up as usize];
            let level = me.upgrades.get(up as usize).copied().unwrap_or(0) as i32;
            if level >= u.max_repeats {
                Some("max_level")
            } else if me.upgrading & (1u64 << up) != 0 {
                Some("upgrading")
            } else if !afford(u.minerals(level + 1), u.gas(level + 1)) {
                Some("afford")
            } else {
                None
            }
        }
        _ => None,
    }
}

pub fn label_check(game: &ShardGame, st: &mut LabelStats) -> Result<()> {
    let map = MapInfo::new(&game.game_info, family_index(&game.meta.map_family), map_precompute(&game.meta.map_hash).as_ref())?;
    let mut mem = AgentMemory::new(&game.game_info);
    let k = CAPABILITIES.len();
    let mut prev: Option<&uranium_proto::PlayerState> = None;
    for s in &game.steps {
        let o = extract(&s.raw_frame, &mut mem, &map, None);
        let (label, why) = host_label_check(&s.action_label, &o);
        mem.record_action(label.action, label.delay_bucket, &acting_ids(&label, &o));
        let a = action_table().actions.get(label.action as usize);
        let name = a.map_or_else(|| format!("#{}", label.action), |a| a.name.clone());
        *st.totals.entry(name.clone()).or_default() += 1;
        if let Some(a) = a.filter(|a| a.uses_select && matches!(why, None | Some("select_cap"))) {
            let c = CAPABILITIES.iter().position(|c| *c == a.select_cap).unwrap();
            let mut types: Vec<&str> = s
                .action_label
                .selected
                .iter()
                .filter_map(|id| o.entity_ids.iter().position(|e| e == id))
                .filter(|&i| !o.masks.unit_cap[i * k + c])
                .filter_map(|i| s.raw_frame.units.iter().find(|u| u.id == o.entity_ids[i]))
                .filter_map(|u| uranium_env::types::TypeTables::get().unit(u.unit_type).map(|t| t.name.as_str()))
                .collect();
            types.sort_unstable();
            types.dedup();
            if !types.is_empty() {
                let kind = if why.is_some() { "void" } else { "kept" };
                *st.cap_detail.entry(format!("{name} <- {} ({kind})", types.join("+"))).or_default() += 1;
            }
        }
        if let Some(w @ ("target_unit_cap" | "target_unit_missing")) = why {
            // Target owner (self / enemy / neutral / unseen) and type, from the raw frame.
            let target = s.action_label.target_unit.and_then(|id| s.raw_frame.units.iter().find(|u| u.id == id));
            let me = game.game_info.self_player;
            let desc = target.map_or_else(
                || "not_in_frame".to_string(),
                |u| {
                    let owner = if u.owner == me { "self" } else if game.game_info.players.iter().any(|p| p.id == u.owner) { "enemy" } else { "neutral" };
                    let ty = uranium_env::types::TypeTables::get().unit(u.unit_type).map_or("?", |t| t.name.as_str());
                    format!("{owner} {ty}")
                },
            );
            *st.type_detail.entry(format!("{name} [{w}] -> {desc}")).or_default() += 1;
        }
        if let (Some(a), Some("type_mask")) = (a, why) {
            let me = &s.raw_frame.self_state;
            let cond = prereq_fail(a, me).unwrap_or("producer_or_required");
            let age = match prev.and_then(|p| prereq_fail(a, p)) {
                Some(c) if c == cond => "old",
                _ if cond == "producer_or_required" => "-",
                _ => "new",
            };
            *st.type_detail.entry(format!("{name} <- {cond} ({age})")).or_default() += 1;
        }
        if let Some(w) = why {
            *st.fails.entry((name, w)).or_default() += 1;
        }
        prev = Some(&s.raw_frame.self_state);
    }
    Ok(())
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
    /// Raw state for checkpoints (061-R10).
    pub fn state(&self) -> u64 {
        self.0
    }
    pub fn from_state(s: u64) -> Self {
        Self(s)
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
            // Memory sees what the policy did, like in RL (030 §4 previous action, §1 last action):
            // the units that act on the command (041-R11).
            self.mem.record_action(label.action, label.delay_bucket, &acting_ids(&label, &o));
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
