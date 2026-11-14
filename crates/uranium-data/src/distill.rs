//! The data side of the 044-R3 policy labeller (`python -m uranium distill-label`, 040-R24): the
//! `distill_pre` games (044-R2 decision-step shards, every label an `unknown` slot) read through
//! the BC loader exactly as `bc-eval` reads a split (030 features and 034 masks, 041-R28 ③ memory
//! from `memory_action` before each observation, map precomputes, `last_failed` 0, no z,
//! symmetry or enemy race mask, every lane drained to its game's last step), the labeller's head
//! choices turned into 040 `action_label`s ([`label_of_choice`]), and the labelled games written
//! as new shards with the 040-R19 invalid-label filter (applied after labelling), the 040-R17
//! verified copy and one records line per view with its `label_policy` (044-R3 ⑤, 044-R4). The
//! model and the sampling are the PyTorch side (AMB-000-4).

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use serde::Serialize;
use sha2::{Digest, Sha256};
use uranium_env::features::label::{HeadChoice, Label, label_of_choice, label_validity_with_memory};
use uranium_replay::shard::{self, Shard, ShardGame};

use crate::bc::{Batch, BcConfig, Feeder, feed_threads};
use crate::data::{GameId, MapStore, StepSource, shard_files};

/// 040-R17 staging folder of the output shards (`URANIUM_SHARD_STAGING_DIR`, as the build).
pub use uranium_replay::shard::staging_dir;

/// 040-R14 / 044-R2: the source tag of the games this labeller reads and writes.
pub const DISTILL_SRC: &str = uranium_replay::pipeline::DISTILL_SRC;
/// 044-R3 ②: `sample` (one draw from the labeller's distribution at temperature 1, the training
/// data) or `argmax` (a tool option, never training data).
pub const LABEL_MODES: [&str; 2] = ["sample", "argmax"];

/// 044-R3 ②: the RNG seed of one game's draws, a hash of (`replay_sha256`, perspective, label
/// seed): SHA-256 of `044-R3/<replay_sha256>/<perspective_slot>/<label_seed>`, its first 8 bytes
/// little-endian, shifted right by one (63 bits, a valid seed for any RNG API).
pub fn game_seed(replay_sha256: &str, perspective_slot: u8, label_seed: u64) -> u64 {
    let d = Sha256::digest(format!("044-R3/{replay_sha256}/{perspective_slot}/{label_seed}").as_bytes());
    u64::from_le_bytes(d[..8].try_into().expect("8 bytes")) >> 1
}

/// The 044-R2 contract of a `distill_pre` game: source `distill`, `memory_action` on every step
/// (memory comes from the model's decisions, never from labels) and frames strictly increasing
/// (the step identity is (`replay_sha256`, perspective, frame)).
pub fn check_pre_game(g: &ShardGame) -> Result<()> {
    let who = || format!("{} view {}", g.meta.replay_sha256, g.meta.perspective_slot);
    ensure!(g.meta.src == DISTILL_SRC, "{}: source {:?}, expected {DISTILL_SRC:?} (044-R3 labels distill_pre shards of `uranium-replay build --src distill`)", who(), g.meta.src);
    ensure!(g.steps.iter().all(|s| s.memory_action.is_some()), "{}: a step without memory_action (044-R2: every distill step has the model's decisions; schema 7)", who());
    ensure!(g.steps.windows(2).all(|w| w[0].raw_frame.frame < w[1].raw_frame.frame), "{}: step frames are not strictly increasing (044-R2: one step per playback frame)", who());
    Ok(())
}

/// The shard files under `input` in path order with their path relative to it (`/`-separated):
/// the output file of each is `<out>/<relative path>` and records name it by that path.
pub fn input_files(input: &Path) -> Result<Vec<(String, PathBuf)>> {
    Ok(shard_files(input)?
        .into_iter()
        .map(|f| {
            let rel = f.strip_prefix(input).unwrap_or(&f).to_string_lossy().replace('\\', "/");
            (rel, f)
        })
        .collect())
}

/// What an existing records file says (044-R5 ③ top-up: the round's later `distill_pre` files are
/// labelled into the same output folder with the same records file): the input files it has
/// lines for, its views and the `label_policy` of its first line (`None` for a new file).
#[derive(Serialize, Debug, Clone, Default)]
pub struct Labelled {
    pub inputs: std::collections::BTreeSet<String>,
    pub views: usize,
    pub policy: Option<serde_json::Value>,
}

/// [`Labelled`] of `records` (empty when the file does not exist). A line that is not a records
/// line (a write cut short) is an error naming it.
pub fn read_labelled(records: &Path) -> Result<Labelled> {
    let mut l = Labelled::default();
    if !records.exists() {
        return Ok(l);
    }
    let text = std::fs::read_to_string(records).with_context(|| records.display().to_string())?;
    for (i, line) in text.lines().enumerate().filter(|(_, x)| !x.trim().is_empty()) {
        let v: serde_json::Value = serde_json::from_str(line).with_context(|| format!("{} line {}: not a records line (044-R3)", records.display(), i + 1))?;
        let input = v["input"].as_str().with_context(|| format!("{} line {}: no `input` (not a distill-label records file)", records.display(), i + 1))?;
        l.inputs.insert(input.to_string());
        l.views += 1;
        if l.policy.is_none() {
            l.policy = Some(v["label_policy"].clone());
        }
    }
    Ok(l)
}

/// 044-R3 argument rules, before any game is read: `input` holds shards; `out` is not inside
/// `input` nor holds it; the records file is not named `distill*.jsonl` (the `--src distill`
/// build's resume scan reads those files of its records folder as build records); an `argmax`
/// labelling (never training data, 044-R3 ②) is not written inside the BC shards root
/// `bc_shards_root` (`<shared>/datasets/replays/shards`, when known). A new records file needs an
/// `out` without shards. An existing one continues that labelling (044-R5 ③ top-up, or a run
/// stopped between files): the input files it names are skipped, at least one is left, and none of
/// the others has its output file yet (one there was published by a run stopped before its
/// records lines: remove it and run again). Returns the input files to label.
pub fn check_args(input: &Path, out: &Path, records: &Path, mode: &str, bc_shards_root: Option<&Path>) -> Result<Vec<String>> {
    use uranium_replay::pipeline::path_within;
    ensure!(LABEL_MODES.contains(&mode), "--mode {mode:?}: expected one of {LABEL_MODES:?} (044-R3)");
    ensure!(input.is_dir(), "--in {}: not a folder", input.display());
    let files = input_files(input)?;
    ensure!(!files.is_empty(), "--in {}: no shard_*.bin.zst under it", input.display());
    ensure!(!path_within(out, input) && !path_within(input, out), "--out {} and --in {} overlap", out.display(), input.display());
    let name = records.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    ensure!(name.ends_with(".jsonl"), "--records {}: a .jsonl file", records.display());
    ensure!(!name.starts_with(DISTILL_SRC), "--records {}: a name starting with `{DISTILL_SRC}` is read as build records by the `--src distill` build's resume scan (044-R2); use e.g. label_d<R>.jsonl", records.display());
    if mode == "argmax"
        && let Some(root) = bc_shards_root
    {
        ensure!(!path_within(out, root), "--mode argmax: --out {} is inside the BC shards {} (044-R3 ②: argmax labels are never training data)", out.display(), root.display());
    }
    let done = read_labelled(records)?;
    if done.inputs.is_empty() && out.is_dir() {
        let there = shard_files(out)?;
        ensure!(there.is_empty(), "--out {} already holds {} shard(s) (e.g. {}) and --records {} has none: label into a new folder, or give that labelling's records to continue it (044-R3)", out.display(), there.len(), there[0].display(), records.display());
    }
    let todo: Vec<String> = files.into_iter().map(|(rel, _)| rel).filter(|rel| !done.inputs.contains(rel)).collect();
    ensure!(!todo.is_empty(), "--records {} already has every input file of {}: nothing to label", records.display(), input.display());
    for rel in &todo {
        let dest = out.join(rel);
        ensure!(!dest.exists(), "{} exists but --records {} has no line for its input {rel}: a labelling stopped after publishing it; remove it and run again (044-R3)", dest.display(), records.display());
    }
    Ok(todo)
}

/// The games of the shard files under a folder, in path order, each checked by [`check_pre_game`];
/// [`GameId`] = (file index, game index in the file).
struct InputGames {
    files: Vec<PathBuf>,
    next_file: usize,
    cur: VecDeque<(usize, ShardGame)>,
    cur_file: usize,
    last: Option<GameId>,
}

impl InputGames {
    fn read_file(path: &Path) -> Result<Vec<ShardGame>> {
        let games = shard::read(path).with_context(|| path.display().to_string())?.games;
        for g in &games {
            check_pre_game(g).with_context(|| path.display().to_string())?;
        }
        Ok(games)
    }
}

impl StepSource for InputGames {
    fn next_game(&mut self) -> Result<Option<ShardGame>> {
        while self.cur.is_empty() {
            if self.next_file >= self.files.len() {
                return Ok(None);
            }
            let k = self.next_file;
            self.next_file += 1;
            self.cur_file = k;
            self.cur = Self::read_file(&self.files[k])?.into_iter().enumerate().collect();
        }
        let (i, g) = self.cur.pop_front().expect("a game");
        self.last = Some(GameId { file: self.cur_file, game: i });
        Ok(Some(g))
    }

    fn last_id(&self) -> Option<GameId> {
        self.last
    }
}

/// Per view of the output: one records line (044-R3 ⑤, 044-R4).
#[derive(Serialize, Debug, Clone)]
pub struct LabelRecord {
    pub replay_sha256: String,
    pub perspective_slot: u8,
    pub src: String,
    pub matchup: String,
    pub map_family: String,
    /// The `distill_pre` shard the view came from (relative to `--in`).
    pub input: String,
    /// `written` or `filtered` (040-R19 after labelling).
    pub status: &'static str,
    /// The output shard (relative to `--out`), written views only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shard: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    pub steps: usize,
    /// 040-R19 counts of the labelled view (the build's function): labelled steps, invalid ones
    /// (BC weight 0, 031-R4 / 041-R11) by reason.
    pub labeled: usize,
    pub invalid: usize,
    pub by_reason: BTreeMap<&'static str, usize>,
    /// 044-R3 ⑤: the labeller and how it labelled (checkpoint path and sha256, mode, label seed,
    /// device and precision, git commit; the caller's fields) plus this view's `game_seed` and
    /// `invalid_labels`.
    pub label_policy: serde_json::Value,
}

/// What [`DistillLabeller::write`] did.
#[derive(Serialize, Debug, Clone, Default)]
pub struct WriteSummary {
    pub views: usize,
    pub written: usize,
    pub filtered: usize,
    pub steps_written: usize,
    pub invalid_labels: usize,
    pub files_written: usize,
}

/// The labelling pass over a `distill_pre` folder: the BC loader's batches ([`Feeder`] in the
/// `bc-eval` configuration), each lane's game identity and seed, the labels the model's choices
/// make, and the output shards.
pub struct DistillLabeller {
    feeder: Feeder,
    src: InputGames,
    /// Path of each of `src.files` relative to the input folder ([`input_files`]).
    rels: Vec<String>,
    label_seed: u64,
    labels: HashMap<GameId, Vec<Label>>,
    maps: Arc<MapStore>,
}

impl DistillLabeller {
    /// `cfg`: the BC config whose lanes / `seq_len` the pass uses; its data settings are those of
    /// `bc-eval` (no z, no symmetry, no race mask, no chain-start observation). `only`: the input
    /// files to read (relative paths, [`check_args`]); `None` = every file under `input`.
    pub fn new(input: &Path, cfg: &BcConfig, maps: Arc<MapStore>, label_seed: u64, only: Option<&[String]>) -> Result<Self> {
        let (rels, files): (Vec<String>, Vec<PathBuf>) = input_files(input)?.into_iter().filter(|(rel, _)| only.is_none_or(|o| o.contains(rel))).unzip();
        ensure!(!files.is_empty(), "{}: no shard_*.bin.zst to label under it", input.display());
        if let Some(o) = only {
            ensure!(o.len() == files.len(), "{}: {} of the {} input files to label are not under it", input.display(), o.len() - files.len(), o.len());
        }
        let mut ecfg = cfg.clone();
        // 044-R3 ①: z absent (`z_present = false`), no augmentation, the real enemy race (030-R11
        // mask is training-only), every step its own observation (041-R30 does not apply, 044-R4).
        ecfg.data.z_drop = 1.0;
        ecfg.data.augment = false;
        ecfg.data.enemy_race_mask = 0.0;
        ecfg.data.chain_start_sources = None;
        let mut feeder = Feeder::new(&ecfg).with_maps(maps.clone());
        feeder.drain = true;
        let src = InputGames { files, next_file: 0, cur: VecDeque::new(), cur_file: 0, last: None };
        Ok(Self { feeder, src, rels, label_seed, labels: HashMap::new(), maps })
    }

    /// The next batch; `None` once every lane is drained.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Result<Option<Batch>> {
        let Some(b) = self.feeder.next_batch(&mut self.src, None)? else { return Ok(None) };
        if b.chunks.iter().all(|c| c.is_empty()) {
            return Ok(None);
        }
        Ok(Some(b))
    }

    /// Per lane of `b`: its game and the game's 044-R3 seed ([`game_seed`]); `None` for a drained lane.
    pub fn lanes(&self, b: &Batch) -> Vec<Option<(GameId, u64)>> {
        (0..b.chunks.len())
            .map(|l| {
                let id = b.game_keys[l].filter(|_| !b.chunks[l].is_empty())?;
                Some((id, game_seed(&b.game_ids[l], b.perspective_slots[l], self.label_seed)))
            })
            .collect()
    }

    /// The choices for lane `lane`'s steps of `b`, in step order (one per step): their labels on
    /// the steps' observations ([`label_of_choice`], frame = the step's frame).
    pub fn put(&mut self, b: &Batch, lane: usize, choices: &[HeadChoice]) -> Result<()> {
        let chunk = b.chunks.get(lane).with_context(|| format!("lane {lane} of {}", b.chunks.len()))?;
        ensure!(choices.len() == chunk.len(), "lane {lane}: {} choices for {} steps", choices.len(), chunk.len());
        let Some(id) = b.game_keys[lane] else {
            ensure!(chunk.is_empty(), "lane {lane}: steps without a game id");
            return Ok(());
        };
        let labels = self.labels.entry(id).or_default();
        labels.extend(chunk.iter().zip(choices).map(|(s, c)| label_of_choice(s.frame, c, &s.obs)));
        Ok(())
    }

    /// Games read so far.
    pub fn games(&self) -> usize {
        self.feeder.state(None).games
    }

    /// Writes every input shard file read again under `out` (same relative path) with each view's
    /// labels put in place of its `unknown` slots (`raw_frame`, `privileged`, `memory_action` and
    /// `meta` unchanged, 044-R2 contract), drops the views whose labelled share of invalid labels
    /// is at least 20% (040-R19, the build's function and reason) and appends one
    /// [`LabelRecord`] per view to `records` after its shard file is published (040-R17: completed
    /// in `staging`, copied, read back past the cache, renamed; an existing records file is
    /// continued, [`check_args`]). `policy` is the records' `label_policy` object. A view without a
    /// label for every step is an error.
    pub fn write(&mut self, out: &Path, records: &Path, policy: &serde_json::Value, staging: &Path, threads: Option<usize>) -> Result<WriteSummary> {
        ensure!(policy.is_object(), "label_policy must be a JSON object");
        std::fs::create_dir_all(out).with_context(|| out.display().to_string())?;
        std::fs::create_dir_all(staging).with_context(|| format!("creating {} (040-R17 staging)", staging.display()))?;
        if let Some(d) = records.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(d)?;
        }
        let mut rec = std::fs::OpenOptions::new().create(true).append(true).open(records).with_context(|| records.display().to_string())?;
        let threads = threads.unwrap_or_else(feed_threads).max(1);
        let mut sum = WriteSummary::default();
        for (k, (file, rel_text)) in self.src.files.clone().iter().zip(self.rels.clone()).enumerate() {
            let rel = PathBuf::from(&rel_text);
            let mut games = InputGames::read_file(file)?;
            for (i, g) in games.iter_mut().enumerate() {
                let labels = self.labels.remove(&GameId { file: k, game: i }).unwrap_or_default();
                let who = format!("{rel_text} game {i} ({} view {})", g.meta.replay_sha256, g.meta.perspective_slot);
                ensure!(labels.len() == g.steps.len(), "{who}: {} labels for {} steps (044-R3: every decision step is labelled)", labels.len(), g.steps.len());
                for (s, l) in g.steps.iter_mut().zip(labels) {
                    ensure!(l.frame == s.raw_frame.frame, "{who}: label frame {} on step frame {}", l.frame, s.raw_frame.frame);
                    s.action_label = l;
                }
            }
            let checks = validity(&games, &self.maps, threads)?;
            let mut kept = Vec::new();
            let mut lines = Vec::new();
            for (g, v) in games.into_iter().zip(checks) {
                let mut lp = policy.clone();
                lp["game_seed"] = game_seed(&g.meta.replay_sha256, g.meta.perspective_slot, self.label_seed).into();
                lp["invalid_labels"] = v.invalid.into();
                let rejected = v.rejected() || g.steps.is_empty();
                let reason = if g.steps.is_empty() { Some("no steps".to_string()) } else { v.rejected().then(|| v.filter_reason()) };
                sum.views += 1;
                sum.invalid_labels += v.invalid;
                let line = LabelRecord {
                    replay_sha256: g.meta.replay_sha256.clone(),
                    perspective_slot: g.meta.perspective_slot,
                    src: g.meta.src.clone(),
                    matchup: g.meta.matchup.clone(),
                    map_family: g.meta.map_family.clone(),
                    input: rel_text.clone(),
                    status: if rejected { "filtered" } else { "written" },
                    shard: (!rejected).then(|| rel_text.clone()),
                    reason,
                    steps: g.steps.len(),
                    labeled: v.labeled,
                    invalid: v.invalid,
                    by_reason: v.by_reason.clone(),
                    label_policy: lp,
                };
                lines.push(serde_json::to_string(&line)?);
                if rejected {
                    sum.filtered += 1;
                } else {
                    sum.written += 1;
                    sum.steps_written += g.steps.len();
                    kept.push(g);
                }
            }
            if !kept.is_empty() {
                let dest = out.join(&rel);
                if let Some(d) = dest.parent() {
                    std::fs::create_dir_all(d)?;
                }
                let name = rel.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos());
                let local = staging.join(format!("{}-{stamp}-{k}-{name}", std::process::id()));
                shard::write(&local, &Shard { header: shard::ShardHeader::default(), games: kept })?;
                shard::publish(&local, &dest, |from, to| std::fs::copy(from, to), shard::COPY_RETRIES)?;
                sum.files_written += 1;
            }
            // Records after the shard is in place (040-R17: a records line never names a missing
            // shard), the file's lines in one write.
            let mut text = lines.join("\n");
            text.push('\n');
            rec.write_all(text.as_bytes())?;
            rec.flush()?;
        }
        ensure!(self.labels.is_empty(), "{} labelled games are not in the input files", self.labels.len());
        Ok(sum)
    }
}

/// 040-R19 counts of each game (the build's [`label_validity_with_memory`]: the BC loader's
/// extraction with `memory_action` memory), on up to `threads` threads, in game order.
fn validity(games: &[ShardGame], maps: &MapStore, threads: usize) -> Result<Vec<uranium_env::features::label::LabelValidity>> {
    let one = |g: &ShardGame| -> Result<uranium_env::features::label::LabelValidity> {
        let pre = maps.get(&g.meta.map_hash)?;
        label_validity_with_memory(
            &g.game_info,
            uranium_env::maps::family_index(&g.meta.map_family),
            pre.as_deref(),
            g.meta.enemy_random,
            g.steps.iter().map(|s| (&s.raw_frame, &s.action_label, s.memory_action.as_deref())),
        )
    };
    if games.is_empty() {
        return Ok(Vec::new());
    }
    let per = games.len().div_ceil(threads.clamp(1, games.len()));
    std::thread::scope(|s| {
        let hs: Vec<_> = games.chunks(per).map(|run| s.spawn(move || run.iter().map(one).collect::<Result<Vec<_>>>())).collect();
        let mut out = Vec::with_capacity(games.len());
        for h in hs {
            out.extend(h.join().map_err(|_| anyhow::anyhow!("040-R19 check thread panicked"))??);
        }
        Ok(out)
    })
}
