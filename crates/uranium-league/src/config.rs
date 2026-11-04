//! `config/rl.toml` (042 initial values, AMB-042-3). Every value is a tuning target recorded in
//! each checkpoint (042-R9).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use uranium_env::concede::ConcedeConfig;
use uranium_env::scout::{PseudoMode, RewardConfig};

use crate::league::{BandConfig, LadderEntry, LadderPickSetting, Stage, T1Bot};
pub use crate::sil::{SilConfig, SilMode};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct OptimizerConfig {
    pub lr: f64,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    /// Global L2-norm clip over all parameter gradients (042 table = 041-R3, `crate::clip`).
    pub grad_clip: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct LossConfig {
    pub gamma: f32,
    pub td_lambda: f32,
    pub vtrace_rho_bar: f32,
    pub vtrace_c_bar: f32,
    pub pg: f32,
    pub upgo: f32,
    pub value: f32,
    /// 042-R13: win_open MSE weight (same 0.5 x weight x squared error form as `value`), used
    /// only when the model has the head (032-R18). Absent = 0.5.
    #[serde(default = "default_win_open")]
    pub win_open: f32,
}

fn default_win_open() -> f32 {
    0.5
}

/// Per-head entropy coefficients, summed over used heads only (032-R10).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct EntropyConfig {
    pub action: f32,
    pub delay: f32,
    pub queued: f32,
    pub selected: f32,
    pub target_unit: f32,
    pub target_loc: f32,
}

/// `[kl]` (042 KL 계수 스케줄, 042-R9). `update` is the agent's learner updates in this run: a
/// new run (`--init` included) starts at 0, a resumed one continues from its checkpoint
/// (AMB-042-17).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct KlConfig {
    pub init: f32,
    pub floor: f32,
    pub half_life_updates: f32,
    /// `[kl.head_scale]` (042-R9, AMB-042-17): per-head multiplier of the KL coefficient.
    /// Absent = 1.0 each (the earlier loss).
    #[serde(default)]
    pub head_scale: KlHeadScale,
}

impl KlConfig {
    /// `max(floor, init * 0.5^(update / half_life))` (042 KL schedule).
    pub fn coef(&self, update: u64) -> f32 {
        (self.init * 0.5f32.powf(update as f32 / self.half_life_updates)).max(self.floor)
    }

    /// 042 KL 설정 검사 (AMB-042-17): `init > 0`, `0 <= floor <= init`, `half_life_updates > 0`,
    /// every head multiplier >= 0.
    pub fn validate(&self) -> Result<()> {
        let finite = [self.init, self.floor, self.half_life_updates].iter().all(|x| x.is_finite());
        if !finite || self.init <= 0.0 || self.floor < 0.0 || self.floor > self.init || self.half_life_updates <= 0.0 {
            bail!("[kl] needs init > 0, 0 <= floor <= init and half_life_updates > 0, got init {} floor {} half_life_updates {} (042-R9, AMB-042-17)", self.init, self.floor, self.half_life_updates);
        }
        if let Some((h, x)) = self.head_scale.by_head().into_iter().find(|(_, x)| !(*x >= 0.0 && x.is_finite())) {
            bail!("[kl.head_scale] {h} must be a finite value >= 0, got {x} (042-R9, AMB-042-17)");
        }
        Ok(())
    }
}

/// `[kl.head_scale]` (042-R9, AMB-042-17): head h's KL coefficient = the schedule value x its
/// multiplier; the used-heads rule (032-R10) and the exact KL (AMB-042-9) stay.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(deny_unknown_fields, default)]
pub struct KlHeadScale {
    pub action: f32,
    pub delay: f32,
    pub queued: f32,
    pub selected: f32,
    pub target_unit: f32,
    pub target_loc: f32,
}

impl Default for KlHeadScale {
    fn default() -> Self {
        Self { action: 1.0, delay: 1.0, queued: 1.0, selected: 1.0, target_unit: 1.0, target_loc: 1.0 }
    }
}

impl KlHeadScale {
    /// The multipliers in the 032-R15 head order (`learner::HEAD_NAMES`).
    pub fn by_head(&self) -> [(&'static str, f32); 6] {
        [("action", self.action), ("delay", self.delay), ("queued", self.queued), ("selected", self.selected), ("target_unit", self.target_unit), ("target_loc", self.target_loc)]
    }
}

/// 042 queue table: what a push into a full trajectory queue does.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum QueueFull {
    /// Backpressure: the pushing game waits for a place, FIFO, at most `queue_wait_max_s`.
    #[default]
    Wait,
    /// Previous behaviour: never wait, drop the oldest segment.
    DropOldest,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ActorsConfig {
    pub weight_push_every: u64,
    pub max_policy_lag: u64,
    /// Queue capacity in batches of `lanes` segments; at least 1 (042 queue table).
    pub queue_batches: usize,
    /// Absent in configs copied into checkpoints before 2026-10-31: `wait`.
    #[serde(default)]
    pub queue_full: QueueFull,
    /// Longest single wait for a queue place before the oldest segment is dropped (AMB-042-11).
    #[serde(default = "default_queue_wait_max_s")]
    pub queue_wait_max_s: u64,
    /// Wall-clock limit of one RL game, `StartGame.wall_timeout_s` (AMB-042-12, 020-R3); at least 1.
    #[serde(default = "default_game_wall_timeout_s")]
    pub game_wall_timeout_s: u32,
}

fn default_queue_wait_max_s() -> u64 {
    120
}

fn default_game_wall_timeout_s() -> u32 {
    7_200
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct LeagueConfig {
    pub share_main: f64,
    pub share_main_exploiter: f64,
    pub share_league_exploiter: f64,
    pub main_pfsp: f64,
    pub main_self: f64,
    pub main_t1: f64,
    pub pfsp_power: f64,
    pub main_exploiter_fallback_win_rate: f64,
    pub main_exploiter_fallback_share: f64,
    /// 042 승률 표: per-game EMA decay d of every win-table pick (ladder f_var, C1 snapshot
    /// f_hard, C2 PFSP, Main Exploiter f_var); `rush-s2` 0.97 (AMB-042-16).
    pub win_rate_decay: f64,
    pub win_rate_prior: f64,
    pub race_uniform_mix: f64,
    pub bot_random_race: f64,
    /// 042 anchor (AMB-042-14): a fixed opponent checkpoint, the folder `ckpt-<step>` (its
    /// `agents/main`) or that `agents/main` folder; relative = under `URANIUM_SHARED_DIR`
    /// (020-R11). Empty / absent = no anchor.
    #[serde(default)]
    pub anchor: String,
    /// 042-R15 (AMB-042-17): game seconds after which an RL training game against a model
    /// (current Main, snapshot, anchor) ends as a draw (`TimeLimitDraw`, 020-R3); 0 = no cap
    /// (the 60-minute limit). 0 or 60..=3600.
    #[serde(default)]
    pub model_game_max_s: u32,
    /// `[league.rush_judge]` (042-R17, AMB-042-18): the rush mini-game. Absent / `groups = []` =
    /// off (the earlier game lengths and results). Left out of the serialised config at its
    /// defaults, so checkpoints of runs without it record the same config as before.
    #[serde(default, skip_serializing_if = "RushJudgeConfig::is_default")]
    pub rush_judge: RushJudgeConfig,
}

/// 020-R3: the 60-minute game limit in frames (24 per game second).
pub const GAME_MAX_FRAMES: u32 = 86_400;

/// `[league.rush_judge]` (042-R17, 020-R19, AMB-042-18): RL training games of these ladder groups
/// end at `at_s` game seconds (`GameSpec.max_frames`); a game that ends there as `TimeLimitDraw`
/// is a win when the learning session's own SCVs at its last observation are at least `min_scv`,
/// else a loss. Key names and ranges: 메인 임시 결정.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(deny_unknown_fields, default)]
pub struct RushJudgeConfig {
    /// `[[ladder]] group` names; empty = off (`at_s`, `min_scv` unused).
    pub groups: Vec<String>,
    pub at_s: u32,
    pub min_scv: u32,
}

impl Default for RushJudgeConfig {
    fn default() -> Self {
        Self { groups: Vec::new(), at_s: 480, min_scv: 8 }
    }
}

impl RushJudgeConfig {
    fn is_default(&self) -> bool {
        *self == Self::default()
    }

    /// Whether a game of this ladder group (`None` = not a ladder game) is judged.
    pub fn judges(&self, ladder_group: Option<&str>) -> bool {
        ladder_group.is_some_and(|g| self.groups.iter().any(|x| x == g))
    }

    /// `min(86,400, at_s x 24)` (042-R15's 24 fps; 480 s = 11,520 frames = 8:00).
    pub fn max_frames(&self) -> u32 {
        GAME_MAX_FRAMES.min(self.at_s.saturating_mul(24))
    }

    /// 042-R17 판정: own SCVs at the last observation >= `min_scv` = win, else loss.
    pub fn judge(&self, scv: u32) -> uranium_env::reward::Outcome {
        if scv >= self.min_scv { uranium_env::reward::Outcome::Win } else { uranium_env::reward::Outcome::Loss }
    }

    /// 042-R17 검사 (only with `groups` set): every name in `[curriculum] ladder_groups`,
    /// `at_s` 60..=3600 (042-R15's range), `min_scv >= 1`.
    pub fn validate(&self, ladder_groups: &std::collections::BTreeMap<String, f64>) -> Result<()> {
        if self.groups.is_empty() {
            return Ok(());
        }
        if let Some(g) = self.groups.iter().find(|g| !ladder_groups.contains_key(*g)) {
            bail!("[league.rush_judge] groups names {g:?}, which is not in [curriculum] ladder_groups (042-R17, AMB-042-18)");
        }
        if !(60..=3_600).contains(&self.at_s) {
            bail!("[league.rush_judge] at_s must be 60..=3600 game seconds, got {} (042-R17, AMB-042-18)", self.at_s);
        }
        if self.min_scv < 1 {
            bail!("[league.rush_judge] min_scv must be at least 1 (042-R17, AMB-042-18)");
        }
        Ok(())
    }
}

impl LeagueConfig {
    /// 042-R15 / 020-R3 (AMB-042-17): `GameSpec.max_frames` of an RL training game; a game
    /// against a model gets `min(86,400, model_game_max_s x 24)` when the cap is set, every other
    /// game the 60-minute limit.
    pub fn game_max_frames(&self, vs_model: bool) -> u32 {
        if vs_model && self.model_game_max_s > 0 { GAME_MAX_FRAMES.min(self.model_game_max_s.saturating_mul(24)) } else { GAME_MAX_FRAMES }
    }

    /// 042-R17 (AMB-042-18): whether an RL training game is a rush mini-game — a ladder game of a
    /// `[league.rush_judge]` group; games against a model never are.
    pub fn rush_judged(&self, vs_model: bool, ladder_group: Option<&str>) -> bool {
        !vs_model && self.rush_judge.judges(ladder_group)
    }

    /// 042-R15 / R17 / 020-R3: `GameSpec.max_frames` of an RL training game by its opponent:
    /// against a model the 042-R15 value, a rush mini-game `min(86,400, at_s x 24)`, any other
    /// game 86,400.
    pub fn game_max_frames_of(&self, vs_model: bool, ladder_group: Option<&str>) -> u32 {
        if self.rush_judged(vs_model, ladder_group) { self.rush_judge.max_frames() } else { self.game_max_frames(vs_model) }
    }

    /// 042 anchor (AMB-042-14): its checkpoint name (`ckpt-<step>`, also for an `agents/main`
    /// path), the win-table key being `anchor@<name>`; `None` without an anchor.
    pub fn anchor_name(&self) -> Option<String> {
        let parts: Vec<&str> = self.anchor.trim().split(['/', '\\']).filter(|c| !c.is_empty() && *c != ".").collect();
        let n = parts.len();
        match n {
            0 => None,
            _ if n >= 3 && parts[n - 1] == "main" && parts[n - 2] == crate::ckpt::AGENTS_DIR => Some(parts[n - 3].to_string()),
            _ => Some(parts[n - 1].to_string()),
        }
    }
}

/// An agent folder with the files the model loader reads (weights and `meta.json`).
pub fn has_model_files(d: &Path) -> bool {
    d.join(uranium_agent::infer::WEIGHTS_FILE).is_file() && d.join("meta.json").is_file()
}

/// 042 anchor (AMB-042-14): the anchor's model folder (`<anchor>/agents/main`, or `<anchor>`
/// when it is that folder), relative paths under `shared` (`URANIUM_SHARED_DIR`, 020-R11). An
/// error when neither holds the model files (weights and `meta.json`): a start error (042).
pub fn anchor_agent_dir(shared: &Path, anchor: &str) -> Result<PathBuf> {
    let p = Path::new(anchor.trim());
    let p = if p.is_absolute() { p.to_path_buf() } else { shared.join(p) };
    let main = p.join(crate::ckpt::AGENTS_DIR).join("main");
    if has_model_files(&main) {
        return Ok(main);
    }
    let is_main = p.file_name().is_some_and(|n| n == "main") && p.parent().and_then(|q| q.file_name()).is_some_and(|n| n == crate::ckpt::AGENTS_DIR);
    if is_main && has_model_files(&p) {
        return Ok(p);
    }
    bail!("[league] anchor {anchor}: no model files ({}, meta.json) in {} (expected a checkpoint folder `ckpt-<step>` or its `agents/main`, 042)", uranium_agent::infer::WEIGHTS_FILE, main.display())
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct SnapshotConfig {
    pub main_every_updates: u64,
    pub exploiter_win_rate: f64,
    pub exploiter_win_window: usize,
    pub exploiter_max_updates: u64,
    pub league_exploiter_reset_prob: f64,
    /// 042 C1 snapshot candidates (AMB-042-14): the most recent this many published Main
    /// snapshots, 0 = all. Absent = 4.
    #[serde(default = "default_main_candidates")]
    pub main_candidates: usize,
}

fn default_main_candidates() -> usize {
    4
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct CurriculumConfig {
    pub window_games: usize,
    pub c0_t0: f64,
    pub c0_self: f64,
    pub c0_to_c1_win_rate_vs_t0: f64,
    pub c1_t0: f64,
    pub c1_t1: f64,
    pub c1_self: f64,
    pub c1_to_c2_win_rate_vs_t1: f64,
    pub pseudo_reward_decay_updates: f64,
    /// 042-R5 (AMB-042-13): stage of a new league (no LATEST in the run), `C0` or `C1`; a
    /// resumed league keeps the stage of its `league.bin`. Absent = `C0`.
    #[serde(default = "default_start_stage")]
    pub start_stage: Stage,
    /// 042-R5 (AMB-042-13): `false` = the win-rate bars never change the stage (a stage change
    /// is the user's decision); the transition windows are still recorded. Absent = `true`.
    #[serde(default = "default_auto_advance")]
    pub auto_advance: bool,
    /// 033 r_value / 042-R5 (AMB-042-13): stages that use r_value, weight 0 in the others.
    /// Absent = `["C0"]`.
    #[serde(default = "default_r_value_stages")]
    pub r_value_stages: Vec<Stage>,
    /// 042 C1 model opponents (AMB-042-14): share of past Main snapshots (PFSP f_hard over the
    /// `[snapshot] main_candidates` most recent published ones). `c1_self` stays the current
    /// Main's share. Absent = 0.
    #[serde(default)]
    pub c1_snapshot: f64,
    /// 042 (AMB-042-14): share of the anchor (`[league] anchor`). Absent = 0.
    #[serde(default)]
    pub c1_anchor: f64,
    /// 033 / 042 "r_value의 게임별 적용" (AMB-042-14): `false` = r_value has policy weight 0 in
    /// games against a model (current Main, snapshot, anchor); bot games follow
    /// `r_value_stages`. The value head learns r_value in every game. Absent = `true`.
    #[serde(default = "default_r_value_vs_model")]
    pub r_value_vs_model: bool,
    /// 042 C1 러시 사다리 (AMB-042-15): share of the `[[ladder]]` opponents. Absent = 0 (the
    /// earlier C1 picks).
    #[serde(default)]
    pub c1_ladder: f64,
    /// 042 사다리 안 상대 선택 (AMB-042-15 / 17): `"f_var"` or `"band"` for every group, or a
    /// table group name → method (groups not listed: `"f_var"`); other values are load errors.
    #[serde(default)]
    pub ladder_pick: LadderPickSetting,
    /// `[curriculum.band]`: the numbers of `"band"` (042, AMB-042-17).
    #[serde(default)]
    pub band: BandConfig,
    /// 042 사다리 그룹 (AMB-042-16): group name → share. Non-empty: a ladder game first draws a
    /// group by the shares, then an entry of that group by `ladder_pick`. Absent / empty = one
    /// pool of every entry (the earlier picks; `group` is then only recorded).
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub ladder_groups: std::collections::BTreeMap<String, f64>,
}

fn default_r_value_vs_model() -> bool {
    true
}

fn default_start_stage() -> Stage {
    Stage::C0
}

fn default_auto_advance() -> bool {
    true
}

fn default_r_value_stages() -> Vec<Stage> {
    vec![Stage::C0]
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct PpoConfig {
    pub clip: f32,
    pub gae_lambda: f32,
    pub epochs: usize,
    pub adopt_margin: f64,
}

/// 042-R11: replays kept from training games.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ReplaysConfig {
    /// Keep the replay of every game whose id is a multiple of this (0 = no regular sample);
    /// winner-disagreement games (020-R13) are always kept.
    pub sample_every: u64,
}

impl Default for ReplaysConfig {
    fn default() -> Self {
        Self { sample_every: 100 }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct RlConfig {
    pub lanes: usize,
    pub seq_len: usize,
    pub micro_lanes: usize,
    pub optimizer: OptimizerConfig,
    pub loss: LossConfig,
    pub entropy: EntropyConfig,
    pub kl: KlConfig,
    pub actors: ActorsConfig,
    pub league: LeagueConfig,
    pub snapshot: SnapshotConfig,
    pub curriculum: CurriculumConfig,
    pub ppo: PpoConfig,
    /// Absent in configs copied into checkpoints before 042-R11.
    #[serde(default)]
    pub replays: ReplaysConfig,
    /// 061-R6 (G-D6): newest checkpoints kept on the share; absent = 5, 0 = all.
    #[serde(default)]
    pub keep_checkpoints: Option<usize>,
    /// `[[t1]]`: T1 bots and the races they play (021). Like every value here it comes from the
    /// current config file on each start, resume included (042 "재개 시 설정", AMB-042-13).
    #[serde(default)]
    pub t1: Vec<T1Bot>,
    /// `[[ladder]]`: the C1 rush ladder (042, AMB-042-15), from the current config file on each
    /// start like `[[t1]]`. Only C1 uses it (not the C2 PFSP pool).
    #[serde(default)]
    pub ladder: Vec<LadderEntry>,
    /// `[concede]`: 042-R14 collapse concede of RL training games (AMB-042-15). Absent = off.
    #[serde(default)]
    pub concede: ConcedeConfig,
    /// `[reward]`: 033-R11 channel mode and the r_scout / r_survive settings (033-R9 / R10,
    /// AMB-042-16), from the current config file on each start. Absent = `"z"` with the default
    /// settings (the earlier behaviour).
    #[serde(default)]
    pub reward: RewardConfig,
    /// `[sil]`: 042-R16 self-imitation / held-game replay (AMB-042-17). Absent = off.
    #[serde(default)]
    pub sil: SilConfig,
}

impl RlConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let c: Self = toml::from_str(&std::fs::read_to_string(path)?)?;
        c.validate().with_context(|| format!("{}", path.display()))?;
        Ok(c)
    }

    /// Values the 042 queue table forbids.
    pub fn validate(&self) -> Result<()> {
        if self.actors.queue_batches < 1 {
            bail!("[actors] queue_batches must be at least 1 (042 queue table)");
        }
        // A waiting game's OpenBW client is paused too: stay below its drop time (011-R18, 600 s).
        if !(1..600).contains(&self.actors.queue_wait_max_s) {
            bail!("[actors] queue_wait_max_s must be at least 1 and below 600 (042 queue table, 011-R18)");
        }
        if self.actors.game_wall_timeout_s < 1 {
            bail!("[actors] game_wall_timeout_s must be at least 1 (042, 020-R3)");
        }
        // 042-R5 (AMB-042-13): a new league starts in C0 or C1 only.
        if !matches!(self.curriculum.start_stage, Stage::C0 | Stage::C1) {
            bail!("[curriculum] start_stage must be \"C0\" or \"C1\" (042-R5)");
        }
        // 021 / 042: a T1 share needs T1 bots, and every bot plays at least one race.
        if self.t1.is_empty() && (self.curriculum.c1_t1 > 0.0 || self.league.main_t1 > 0.0) {
            bail!("[[t1]] is empty but [curriculum] c1_t1 or [league] main_t1 is above 0 (042, 021)");
        }
        if let Some(b) = self.t1.iter().find(|b| b.races.is_empty()) {
            bail!("[[t1]] {} has no races (021)", b.name);
        }
        // 042 (AMB-042-14): an anchor share needs an anchor (its model files are checked at start).
        if self.curriculum.c1_anchor > 0.0 && self.league.anchor.trim().is_empty() {
            bail!("[curriculum] c1_anchor is above 0 but [league] anchor is empty (042, AMB-042-14)");
        }
        // 042 사다리 설정 검사 (AMB-042-15).
        if self.curriculum.c1_ladder > 0.0 && self.ladder.is_empty() {
            bail!("[curriculum] c1_ladder is above 0 but [[ladder]] is empty (042, AMB-042-15)");
        }
        let mut keys = std::collections::BTreeSet::new();
        for (i, e) in self.ladder.iter().enumerate() {
            let (_, key, _) = e.opponent().map_err(|why| anyhow::anyhow!("[[ladder]] entry {}: {why}", i + 1))?;
            if !keys.insert(key.clone()) {
                bail!("[[ladder]] entry {}: win-table key {key} appears twice (042, AMB-042-15)", i + 1);
            }
        }
        // 042 사다리 그룹 설정 검사 (AMB-042-16).
        let groups = &self.curriculum.ladder_groups;
        // Group names go into games.csv as they are: letters, digits, `_` and `-` only.
        let name_ok = |g: &str| !g.is_empty() && g.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-');
        if let Some(g) = groups.keys().find(|g| !name_ok(g)) {
            bail!("[curriculum] ladder_groups name {g:?} must be letters, digits, '_' or '-' (042, AMB-042-16)");
        }
        if let Some((i, e)) = self.ladder.iter().enumerate().find(|(_, e)| !e.group.is_empty() && !name_ok(&e.group)) {
            bail!("[[ladder]] entry {}: group {:?} must be letters, digits, '_' or '-' (042, AMB-042-16)", i + 1, e.group);
        }
        if !groups.is_empty() {
            if let Some((g, x)) = groups.iter().find(|(_, x)| !(**x >= 0.0 && x.is_finite())) {
                bail!("[curriculum] ladder_groups share of {g:?} must be a finite value >= 0, got {x} (042, AMB-042-16)");
            }
            if groups.values().sum::<f64>() <= 0.0 {
                bail!("[curriculum] ladder_groups shares must sum to more than 0 (042, AMB-042-16)");
            }
            if let Some(g) = groups.keys().find(|g| !self.ladder.iter().any(|e| &e.group == *g)) {
                bail!("[curriculum] ladder_groups lists {g:?} but no [[ladder]] entry has group = {g:?} (042, AMB-042-16)");
            }
            if let Some((i, e)) = self.ladder.iter().enumerate().find(|(_, e)| !groups.contains_key(&e.group)) {
                bail!("[[ladder]] entry {}: group {:?} is not in [curriculum] ladder_groups (042, AMB-042-16)", i + 1, e.group);
            }
        }
        // 042 사다리 선택 (AMB-042-17): group methods only for listed groups, band numbers.
        if let LadderPickSetting::PerGroup(m) = &self.curriculum.ladder_pick
            && let Some(g) = m.keys().find(|g| !groups.contains_key(*g))
        {
            bail!("[curriculum] ladder_pick names group {g:?}, which is not in ladder_groups (042, AMB-042-17)");
        }
        self.curriculum.band.validate().map_err(anyhow::Error::msg)?;
        // 042-R9 KL schedule and head multipliers (AMB-042-17).
        self.kl.validate()?;
        // 042-R15 (AMB-042-17): 0 or 60 s ..= 60 min (메인 임시 결정).
        let cap = self.league.model_game_max_s;
        if cap != 0 && !(60..=3_600).contains(&cap) {
            bail!("[league] model_game_max_s must be 0 (no cap) or 60..=3600 game seconds, got {cap} (042-R15, AMB-042-17)");
        }
        // 042-R17 (AMB-042-18, 메인 임시 결정).
        self.league.rush_judge.validate(groups)?;
        // 042-R16 (AMB-042-17).
        self.sil.validate(self.lanes, groups).map_err(anyhow::Error::msg)?;
        // 042-R14 (AMB-042-15).
        self.concede.validate().map_err(anyhow::Error::msg)?;
        // 033-R11 (AMB-042-16).
        self.reward.validate()?;
        Ok(())
    }

    /// 033-R11: the reward channel mode.
    pub fn pseudo(&self) -> PseudoMode {
        self.reward.pseudo
    }

    /// The repository's `config/rl.toml`.
    pub fn default_file() -> Self {
        toml::from_str(include_str!("../../../config/rl.toml")).expect("config/rl.toml parses")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::league::LadderPick;

    #[test]
    fn repository_rl_toml_parses_042_r9() {
        let c = RlConfig::default_file();
        assert_eq!(c.seq_len, 64, "042-R8");
        assert!(c.league.share_main >= 0.5, "042-R7");
        assert!((c.kl.coef(0) - 0.1).abs() < 1e-6);
        assert!((c.kl.coef(20_000) - 0.05).abs() < 1e-6);
        assert!((c.kl.coef(10_000_000) - 0.01).abs() < 1e-6);
        assert_eq!(c.replays.sample_every, 100, "042-R11");
        // 042 queue table, AMB-042-11 / 12.
        assert_eq!(c.actors.queue_full, QueueFull::Wait);
        assert_eq!(c.actors.queue_wait_max_s, 120);
        assert_eq!(c.actors.game_wall_timeout_s, 7_200);
        c.validate().unwrap();
    }

    fn with_actors(actors: &str) -> Result<RlConfig> {
        let text = include_str!("../../../config/rl.toml");
        let start = text.find("[actors]").unwrap();
        let end = start + text[start..].find("\n[league]").unwrap();
        let c: RlConfig = toml::from_str(&format!("{}[actors]\n{actors}\n{}", &text[..start], &text[end..]))?;
        c.validate()?;
        Ok(c)
    }

    /// 042 queue table: `queue_batches` 0, `game_wall_timeout_s` 0, `queue_wait_max_s` outside
    /// 1..600 and unknown `queue_full` values are load errors; absent new keys (configs copied into older checkpoints) take the
    /// defaults.
    #[test]
    fn actors_queue_settings_are_checked_042() {
        let base = "weight_push_every = 4\nmax_policy_lag = 32\n";
        let c = with_actors(&format!("{base}queue_batches = 4")).unwrap();
        assert_eq!((c.actors.queue_full, c.actors.queue_wait_max_s, c.actors.game_wall_timeout_s), (QueueFull::Wait, 120, 7_200));
        let c = with_actors(&format!("{base}queue_batches = 1\nqueue_full = \"drop_oldest\"")).unwrap();
        assert_eq!(c.actors.queue_full, QueueFull::DropOldest);
        assert!(with_actors(&format!("{base}queue_batches = 0")).is_err());
        assert!(with_actors(&format!("{base}queue_batches = 4\nqueue_full = \"block\"")).is_err());
        assert!(with_actors(&format!("{base}queue_batches = 4\ngame_wall_timeout_s = 0")).is_err());
        assert!(with_actors(&format!("{base}queue_batches = 4\nqueue_wait_max_s = 0")).is_err());
        assert!(with_actors(&format!("{base}queue_batches = 4\nqueue_wait_max_s = 600")).is_err());
        assert_eq!(with_actors(&format!("{base}queue_batches = 4\nqueue_wait_max_s = 599")).unwrap().actors.queue_wait_max_s, 599);
    }

    const NEW_CURRICULUM_KEYS: [&str; 3] = ["start_stage", "auto_advance", "r_value_stages"];

    /// `config/rl.toml` without the three AMB-042-13 keys (as in a config copied into an older
    /// checkpoint), `lines` added at the top of `[curriculum]`.
    fn with_curriculum(lines: &str) -> Result<RlConfig> {
        let text: String = include_str!("../../../config/rl.toml")
            .lines()
            .filter(|l| !NEW_CURRICULUM_KEYS.iter().any(|k| l.starts_with(k)))
            .map(|l| format!("{l}\n"))
            .collect();
        let at = text.find("[curriculum]\n").unwrap() + "[curriculum]\n".len();
        let c: RlConfig = toml::from_str(&format!("{}{lines}\n{}", &text[..at], &text[at..]))?;
        c.validate()?;
        Ok(c)
    }

    /// 042-R5 / 033 (AMB-042-13): defaults reproduce the earlier behaviour, absent keys take
    /// them, `start_stage` is "C0" or "C1" only.
    #[test]
    fn curriculum_stage_settings_042_r5() {
        let c = RlConfig::default_file();
        assert_eq!((c.curriculum.start_stage, c.curriculum.auto_advance), (Stage::C0, true));
        assert_eq!(c.curriculum.r_value_stages, vec![Stage::C0]);
        assert_eq!(with_curriculum("").unwrap().curriculum, c.curriculum, "absent = defaults");
        let text = include_str!("../../../config/rl.toml");
        for k in NEW_CURRICULUM_KEYS {
            assert!(text.lines().any(|l| l.starts_with(k)), "rl.toml lists {k}");
        }
        assert_eq!(with_curriculum("start_stage = \"C1\"").unwrap().curriculum.start_stage, Stage::C1);
        for bad in ["\"C2\"", "\"C3\"", "\"c1\"", "1"] {
            assert!(with_curriculum(&format!("start_stage = {bad}")).is_err(), "start_stage = {bad}");
        }
        let c = with_curriculum("start_stage = \"C1\"\nauto_advance = false\nr_value_stages = [\"C0\", \"C1\"]");
        let c = c.unwrap_or_else(|e| panic!("{e:#}"));
        assert_eq!((c.curriculum.start_stage, c.curriculum.auto_advance), (Stage::C1, false));
        assert_eq!(c.curriculum.r_value_stages, vec![Stage::C0, Stage::C1]);
    }

    /// 042 "T1·셀프 run" (AMB-042-13): `config/rl_t1self.toml` is `rl.toml` with the listed
    /// changes only.
    #[test]
    fn t1self_config_042_r5() {
        let t: RlConfig = toml::from_str(include_str!("../../../config/rl_t1self.toml")).unwrap();
        t.validate().unwrap();
        let cc = &t.curriculum;
        assert_eq!((cc.start_stage, cc.auto_advance), (Stage::C1, false));
        assert_eq!((cc.c1_t0, cc.c1_t1, cc.c1_self), (0.0, 0.5, 0.5));
        assert_eq!(cc.r_value_stages, vec![Stage::C0, Stage::C1]);
        let names: Vec<&str> = t.t1.iter().map(|b| b.name.as_str()).collect();
        assert_eq!(names, ["ualbertabot", "steamhammer", "stardust", "zzzkbot", "iron"], "McRave left out");
        // 021-R4: three T1 bots per race.
        for r in [uranium_proto::Race::Terran, uranium_proto::Race::Protoss, uranium_proto::Race::Zerg] {
            assert_eq!(t.t1.iter().filter(|b| b.races.contains(&r)).count(), 3, "{r:?}");
        }
        // Everything else (other curriculum values included) as in rl.toml.
        let mut d = RlConfig::default_file();
        let dc = &mut d.curriculum;
        (dc.start_stage, dc.auto_advance, dc.r_value_stages) = (cc.start_stage, cc.auto_advance, cc.r_value_stages.clone());
        (dc.c1_t0, dc.c1_t1, dc.c1_self) = (cc.c1_t0, cc.c1_t1, cc.c1_self);
        d.t1.retain(|b| b.name != "mcrave");
        assert_eq!(d, t);
    }

    const AMB_042_14_KEYS: [&str; 5] = ["c1_snapshot", "c1_anchor", "r_value_vs_model", "anchor", "main_candidates"];

    /// 042 / 033 (AMB-042-14): rl.toml lists the new keys at their defaults (C1 as before: no
    /// snapshot or anchor share, r_value vs models as before, no anchor, 4 candidates); a config
    /// without them (copied into an older checkpoint) gets the same values.
    #[test]
    fn model_opponent_keys_default_042_amb_042_14() {
        let c = RlConfig::default_file();
        assert_eq!((c.curriculum.c1_snapshot, c.curriculum.c1_anchor, c.curriculum.r_value_vs_model), (0.0, 0.0, true));
        assert_eq!((c.league.anchor.as_str(), c.league.anchor_name(), c.snapshot.main_candidates), ("", None, 4));
        let full = include_str!("../../../config/rl.toml");
        let key_line = |l: &str, k: &str| l.starts_with(&format!("{k} ="));
        for k in AMB_042_14_KEYS {
            assert_eq!(full.lines().filter(|l| key_line(l, k)).count(), 1, "rl.toml lists {k}");
        }
        let text: String = full.lines().filter(|l| !AMB_042_14_KEYS.iter().any(|k| key_line(l, k))).map(|l| format!("{l}\n")).collect();
        let old: RlConfig = toml::from_str(&text).unwrap();
        assert_eq!(old, c, "absent = defaults");
    }

    /// 042 anchor (AMB-042-14): `c1_anchor > 0` with an empty anchor is a load error.
    #[test]
    fn anchor_share_needs_an_anchor_042() {
        let mut c = RlConfig::default_file();
        c.curriculum.c1_anchor = 0.1;
        assert!(c.validate().is_err());
        c.league.anchor = "  ".into();
        assert!(c.validate().is_err());
        c.league.anchor = "checkpoints/r/ckpt-1".into();
        c.validate().unwrap();
        c.curriculum.c1_anchor = 0.0;
        c.league.anchor.clear();
        c.curriculum.c1_snapshot = 0.3;
        c.validate().unwrap();
    }

    /// 042 anchor (AMB-042-14): the win-table name is the checkpoint folder's, for either path form.
    #[test]
    fn anchor_name_is_the_checkpoint_folder_042() {
        let name = |a: &str| LeagueConfig { anchor: a.into(), ..RlConfig::default_file().league }.anchor_name();
        assert_eq!(name("checkpoints/ac1-vtrace-s1/ckpt-3133569").as_deref(), Some("ckpt-3133569"));
        assert_eq!(name("checkpoints/ac1-vtrace-s1/ckpt-3133569/agents/main/").as_deref(), Some("ckpt-3133569"));
        assert_eq!(name(r"D:\share\checkpoints\r\ckpt-7\agents\main").as_deref(), Some("ckpt-7"));
        assert_eq!(name("/abs/ckpt-9/").as_deref(), Some("ckpt-9"));
        assert_eq!(name(""), None);
        assert_eq!(name(" "), None);
    }

    /// 042 anchor (AMB-042-14): the model folder of a checkpoint folder or of its agents/main,
    /// relative to the share; missing model files are an error (a start error).
    #[test]
    fn anchor_agent_dir_needs_model_files_042() {
        let shared = std::env::temp_dir().join(format!("uranium-anchor-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&shared);
        let main = shared.join("checkpoints").join("r").join("ckpt-5").join("agents").join("main");
        std::fs::create_dir_all(&main).unwrap();
        assert!(anchor_agent_dir(&shared, "checkpoints/r/ckpt-5").is_err(), "no files");
        std::fs::write(main.join(uranium_agent::infer::WEIGHTS_FILE), b"w").unwrap();
        assert!(anchor_agent_dir(&shared, "checkpoints/r/ckpt-5").is_err(), "no meta.json");
        std::fs::write(main.join("meta.json"), b"{}").unwrap();
        assert_eq!(anchor_agent_dir(&shared, "checkpoints/r/ckpt-5").unwrap(), main);
        assert_eq!(anchor_agent_dir(&shared, "checkpoints/r/ckpt-5/agents/main").unwrap(), main);
        assert_eq!(anchor_agent_dir(Path::new("unused"), main.to_str().unwrap()).unwrap(), main, "absolute");
        assert!(anchor_agent_dir(&shared, "checkpoints/r/ckpt-6").is_err());
        assert!(anchor_agent_dir(&shared, "checkpoints/r/ckpt-5/agents").is_err());
        std::fs::remove_dir_all(&shared).ok();
    }

    /// 042 "T1·셀프 run" 2차 (AMB-042-14): `config/rl_t1self2.toml` is `rl_t1self.toml` with the
    /// listed changes only.
    #[test]
    fn t1self2_config_042_amb_042_14() {
        use uranium_proto::Race;
        let t: RlConfig = toml::from_str(include_str!("../../../config/rl_t1self2.toml")).unwrap();
        t.validate().unwrap();
        let cc = &t.curriculum;
        assert_eq!((cc.c1_t0, cc.c1_t1, cc.c1_self, cc.c1_snapshot, cc.c1_anchor), (0.0, 0.5, 0.15, 0.25, 0.10));
        assert!(!cc.r_value_vs_model);
        assert_eq!(cc.r_value_stages, vec![Stage::C0, Stage::C1]);
        assert_eq!(t.league.anchor, "checkpoints/ac1-vtrace-s1/ckpt-3133569");
        assert_eq!(t.league.anchor_name().as_deref(), Some("ckpt-3133569"));
        assert_eq!((t.snapshot.main_every_updates, t.snapshot.main_candidates), (250, 4));
        let sh = t.t1.iter().find(|b| b.name == "steamhammer").unwrap();
        assert_eq!(sh.races, vec![Race::Terran, Race::Zerg]);
        // 021-R4: at least two T1 bots per race (T 3, P 2, Z 3).
        for (r, n) in [(Race::Terran, 3), (Race::Protoss, 2), (Race::Zerg, 3)] {
            assert_eq!(t.t1.iter().filter(|b| b.races.contains(&r)).count(), n, "{r:?}");
        }
        let mut d: RlConfig = toml::from_str(include_str!("../../../config/rl_t1self.toml")).unwrap();
        let dc = &mut d.curriculum;
        (dc.c1_t1, dc.c1_self, dc.c1_snapshot, dc.c1_anchor, dc.r_value_vs_model) = (cc.c1_t1, cc.c1_self, cc.c1_snapshot, cc.c1_anchor, cc.r_value_vs_model);
        d.league.anchor = t.league.anchor.clone();
        (d.snapshot.main_every_updates, d.snapshot.main_candidates) = (250, 4);
        d.t1.iter_mut().find(|b| b.name == "steamhammer").unwrap().races = sh.races.clone();
        assert_eq!(d, t);
    }

    /// 042 / 021: an empty `[[t1]]` with a T1 share above 0, or a T1 bot without races, is a
    /// load error; an empty list with no T1 share is fine.
    #[test]
    fn t1_list_is_checked_021() {
        let mut c = RlConfig::default_file();
        c.validate().unwrap();
        c.t1[0].races.clear();
        assert!(c.validate().is_err(), "bot without races");
        let mut c = RlConfig::default_file();
        c.t1.clear();
        assert!(c.validate().is_err(), "c1_t1 and main_t1 above 0");
        c.curriculum.c1_t1 = 0.0;
        assert!(c.validate().is_err(), "main_t1 above 0");
        c.league.main_t1 = 0.0;
        c.validate().unwrap();
        c.curriculum.c1_t1 = 0.5;
        assert!(c.validate().is_err(), "c1_t1 above 0");
    }

    const AMB_042_15_KEYS: [&str; 6] = ["c1_ladder", "ladder_pick", "enabled", "max_supply", "hold_s", "after_s"];

    /// 042 / 042-R14 (AMB-042-15): rl.toml lists the new keys at their defaults (no ladder share,
    /// f_var, concede off with 4 / 120 / 180); a config without them (and without `[concede]`,
    /// e.g. one copied into an older checkpoint) gets the same values.
    #[test]
    fn ladder_and_concede_keys_default_042_amb_042_15() {
        let c = RlConfig::default_file();
        assert_eq!((c.curriculum.c1_ladder, c.curriculum.ladder_pick.clone()), (0.0, LadderPickSetting::All(LadderPick::FVar)));
        assert!(c.ladder.is_empty());
        assert_eq!(c.concede, ConcedeConfig::default());
        assert!(!c.concede.enabled);
        let full = include_str!("../../../config/rl.toml");
        let key_line = |l: &str, k: &str| l.starts_with(&format!("{k} ="));
        for k in AMB_042_15_KEYS {
            assert_eq!(full.lines().filter(|l| key_line(l, k)).count(), 1, "rl.toml lists {k}");
        }
        let text: String = full.lines().filter(|l| !AMB_042_15_KEYS.iter().any(|k| key_line(l, k)) && !l.starts_with("[concede]")).map(|l| format!("{l}\n")).collect();
        let old: RlConfig = toml::from_str(&text).unwrap();
        assert_eq!(old, c, "absent = defaults");
    }

    /// `config/rl.toml` with `extra` appended (a `[[ladder]]` list), `curriculum` in place of its
    /// `c1_ladder` / `ladder_pick` lines and `concede` as the body of `[concede]`, validated.
    fn with_ladder(curriculum: &str, extra: &str, concede: &str) -> Result<RlConfig> {
        let text: String = include_str!("../../../config/rl.toml").lines().filter(|l| !l.starts_with("c1_ladder =") && !l.starts_with("ladder_pick =")).map(|l| format!("{l}\n")).collect();
        let text = text.replace("[curriculum]\n", &format!("[curriculum]\n{curriculum}\n"));
        let start = text.find("\n[concede]").unwrap() + 1;
        let end = start + text[start..].find("\n# T1 bots").unwrap();
        let text = format!("{}[concede]\n{concede}{}\n{extra}", &text[..start], &text[end..]);
        let c: RlConfig = toml::from_str(&text)?;
        c.validate()?;
        Ok(c)
    }

    const RUSH_LADDER: &str = "[[ladder]]\nkind = \"t0\"\nbuild = \"zerg_rush\"\nrace = \"Zerg\"\n[[ladder]]\nkind = \"t1\"\nname = \"zzzkbot\"\nrace = \"Zerg\"\n";

    /// 042 사다리 설정 검사 / 042-R14 설정 (AMB-042-15): the load errors.
    #[test]
    fn ladder_and_concede_settings_are_checked_042() {
        let ok = with_ladder("c1_ladder = 0.7", RUSH_LADDER, "enabled = true\n").unwrap_or_else(|e| panic!("{e:#}"));
        assert_eq!(ok.ladder.len(), 2);
        assert!(ok.concede.enabled);
        assert!(with_ladder("c1_ladder = 0.7", "", "").is_err(), "share without a ladder");
        with_ladder("c1_ladder = 0.0", "", "").unwrap();
        with_ladder("c1_ladder = 0.0", RUSH_LADDER, "").unwrap();
        let entry = |body: &str| with_ladder("c1_ladder = 0.7", &format!("[[ladder]]\n{body}\n"), "");
        assert!(entry("kind = \"t2\"\nname = \"x\"\nrace = \"Zerg\"").is_err(), "kind");
        assert!(entry("kind = \"t0\"\nbuild = \"zerg_rushes\"\nrace = \"Zerg\"").is_err(), "unknown build");
        assert!(entry("kind = \"t0\"\nbuild = \"zerg_rush\"\nrace = \"Terran\"").is_err(), "build of another race");
        assert!(entry("kind = \"t0\"\nbuild = \"standard\"\nrace = \"Random\"").is_err(), "Random");
        assert!(entry("kind = \"t1\"\nname = \"zzzkbot\"\nrace = \"Random\"").is_err(), "Random");
        assert!(entry("kind = \"t1\"\nrace = \"Zerg\"").is_err(), "t1 without a name");
        assert!(entry("kind = \"t0\"\nname = \"zzzkbot\"\nrace = \"Zerg\"").is_err(), "t0 without a build");
        assert!(entry("kind = \"t1\"\nname = \"zzzkbot\"\nrace = \"Zerg\"\nraces = [\"Zerg\"]").is_err(), "unknown key");
        entry("kind = \"t0\"\nbuild = \"standard\"\nrace = \"Protoss\"").unwrap();
        assert!(with_ladder("c1_ladder = 0.7", &format!("{RUSH_LADDER}[[ladder]]\nkind = \"t1\"\nname = \"zzzkbot\"\nrace = \"Zerg\"\n"), "").is_err(), "same key twice");
        assert!(with_ladder("c1_ladder = 0.7\nladder_pick = \"f_hard\"", RUSH_LADDER, "").is_err(), "ladder_pick");
        let lp = with_ladder("c1_ladder = 0.7\nladder_pick = \"f_var\"", RUSH_LADDER, "").unwrap();
        assert_eq!(lp.curriculum.ladder_pick, LadderPickSetting::All(LadderPick::FVar));
        for bad in ["max_supply = 201\n", "max_supply = -1\n", "hold_s = 0\n", "after_s = -5\n", "enable = true\n"] {
            assert!(with_ladder("c1_ladder = 0.0", "", bad).is_err(), "{bad}");
        }
        let c = with_ladder("c1_ladder = 0.0", "", "enabled = true\nmax_supply = 0\nhold_s = 1\nafter_s = 0\n").unwrap();
        assert_eq!(c.concede, ConcedeConfig { enabled: true, max_supply: 0, hold_s: 1, after_s: 0 });
    }

    /// 042 "러시 커리큘럼 run" (AMB-042-15): `config/rl_rush1.toml` is `rl_t1self2.toml` with the
    /// listed changes only: shares, f_var, the zerg rush ladder and concede on at the defaults.
    #[test]
    fn rush1_config_042_amb_042_15() {
        use uranium_proto::Race;
        let t: RlConfig = toml::from_str(include_str!("../../../config/rl_rush1.toml")).unwrap();
        t.validate().unwrap();
        let cc = &t.curriculum;
        assert_eq!((cc.c1_ladder, cc.c1_anchor, cc.c1_snapshot, cc.c1_self, cc.c1_t1, cc.c1_t0), (0.70, 0.10, 0.15, 0.05, 0.0, 0.0));
        assert_eq!(cc.ladder_pick, LadderPickSetting::All(LadderPick::FVar));
        assert_eq!(t.concede, ConcedeConfig { enabled: true, ..ConcedeConfig::default() });
        let keys: Vec<String> = t.ladder.iter().map(|e| e.opponent().unwrap().1).collect();
        assert_eq!(keys, ["ualbertabot9pool-zerg", "t0-zerg-rush", "ualbertabot12poollings-zerg", "ualbertabot-zerg", "ualbertabot9poollings-zerg", "steamhammeroverpool-zerg", "zzzkbot-zerg"], "user decision 2026-11-03");
        assert!(t.ladder.iter().all(|e| e.race == Race::Zerg));
        let mut d: RlConfig = toml::from_str(include_str!("../../../config/rl_t1self2.toml")).unwrap();
        let dc = &mut d.curriculum;
        (dc.c1_ladder, dc.c1_anchor, dc.c1_snapshot, dc.c1_self, dc.c1_t1, dc.c1_t0) = (cc.c1_ladder, cc.c1_anchor, cc.c1_snapshot, cc.c1_self, cc.c1_t1, cc.c1_t0);
        d.ladder = t.ladder.clone();
        d.concede.enabled = true;
        assert_eq!(d, t, "everything else as in t1self-s2 ([[t1]] included)");
    }

    /// 042 "러시·운영 혼합 run" (AMB-042-16): `config/rl_rush2.toml` is `rl_rush1.toml` with the
    /// decided changes only — win_rate_decay 0.97, ladder groups rush 0.5 / macro 0.5 (the 7
    /// rush-s1 entries as "rush"), `[reward] pseudo = "scout"` at the confirmed values, the anchor
    /// unchanged; the "macro" group is the 5 operational zerg opponents of the user decision
    /// 2026-11-04 (AMB-042-16).
    #[test]
    fn rush2_config_042_amb_042_16() {
        use uranium_env::scout::{ScoutConfig, SurviveConfig};
        let t: RlConfig = toml::from_str(include_str!("../../../config/rl_rush2.toml")).unwrap();
        t.validate().unwrap();
        assert_eq!(t.league.win_rate_decay, 0.97);
        assert_eq!(t.league.anchor, "checkpoints/ac1-vtrace-s1/ckpt-3133569", "AC1 anchor kept");
        let groups: Vec<(&str, f64)> = t.curriculum.ladder_groups.iter().map(|(g, x)| (g.as_str(), *x)).collect();
        assert_eq!(groups, [("macro", 0.5), ("rush", 0.5)]);
        assert_eq!(t.reward, RewardConfig { pseudo: PseudoMode::Scout, scout: ScoutConfig::default(), survive: SurviveConfig::default() });
        let s = &t.reward.scout;
        assert_eq!((s.weight, s.cap, s.vs_model, s.full_until_s, s.zero_at_s, s.main_radius_tiles, s.workers_min), (0.15, 1.0, 0.5, 240.0, 480.0, 10.0, 4));
        assert_eq!((s.main, s.building_type, s.expansion, s.workers, s.army), (0.3, 0.1, 0.1, 0.1, 0.1));
        let v = &t.reward.survive;
        assert_eq!((v.weight, v.vs_model, v.from_s, v.until_s, v.scv_target), (0.1, 0.0, 180.0, 480.0, 20));
        let rush1: RlConfig = toml::from_str(include_str!("../../../config/rl_rush1.toml")).unwrap();
        let rush: Vec<&LadderEntry> = t.ladder.iter().filter(|e| e.group == "rush").collect();
        assert_eq!(rush.len(), 7);
        for (a, b) in rush.iter().zip(&rush1.ladder) {
            assert_eq!(LadderEntry { group: String::new(), ..(*a).clone() }, *b, "the rush-s1 entries as they were");
        }
        let macro_keys: Vec<String> = t.ladder.iter().filter(|e| e.group == "macro").map(|e| e.opponent().unwrap().1).collect();
        assert_eq!(macro_keys, ["t0-zerg", "t0-zerg-muta", "ualbertabot2hatchhydra-zerg", "ualbertabot3hatchmuta-zerg", "steamhammer2hatchmuta-zerg"], "user decision 2026-11-04 (AMB-042-16)");
        assert!(t.ladder.iter().all(|e| e.race == uranium_proto::Race::Zerg));
        assert_eq!(t.ladder.len(), 12, "7 rush + 5 macro");
        let mut d = rush1;
        d.league.win_rate_decay = 0.97;
        d.curriculum.ladder_groups = t.curriculum.ladder_groups.clone();
        d.ladder = t.ladder.clone();
        d.reward = t.reward.clone();
        assert_eq!(d, t, "everything else as in rush-s1");
    }

    /// 033-R11 / 042 (AMB-042-16): rl.toml lists `[reward]` at the defaults ("z", the confirmed
    /// r_scout / r_survive values) and no ladder groups; a config without `[reward]` (the earlier
    /// files, configs copied into older checkpoints) gets the same values, `[[ladder]]` entries
    /// without `group` get the empty name.
    #[test]
    fn reward_and_group_keys_default_amb_042_16() {
        let c = RlConfig::default_file();
        assert_eq!(c.reward, RewardConfig::default());
        assert_eq!(c.pseudo(), PseudoMode::Z);
        assert!(c.curriculum.ladder_groups.is_empty());
        let full = include_str!("../../../config/rl.toml");
        let at = full.find("\n[reward]").unwrap() + 1;
        let end = at + full[at..].find("\n[concede]").unwrap() + 1;
        let old: RlConfig = toml::from_str(&format!("{}{}", &full[..at], &full[end..])).unwrap();
        assert_eq!(old, c, "absent = defaults");
        for f in [include_str!("../../../config/rl_rush1.toml"), include_str!("../../../config/rl_t1self2.toml")] {
            let o: RlConfig = toml::from_str(f).unwrap();
            o.validate().unwrap();
            assert_eq!((o.reward.clone(), o.curriculum.ladder_groups.is_empty()), (RewardConfig::default(), true));
            assert!(o.ladder.iter().all(|e| e.group.is_empty()));
        }
        // The serialised config (checkpoint meta.json) reads back the same.
        let t: RlConfig = toml::from_str(include_str!("../../../config/rl_rush2.toml")).unwrap();
        let back: RlConfig = serde_json::from_str(&serde_json::to_string(&t).unwrap()).unwrap();
        assert_eq!(back, t);
    }

    /// 042 사다리 그룹 설정 검사 (AMB-042-16): negative or non-finite shares, shares summing to 0, a
    /// listed group without entries and an entry of a group that is not listed are load errors;
    /// without `ladder_groups` any `group` is fine.
    #[test]
    fn ladder_groups_are_checked_042_amb_042_16() {
        let ladder = |groups: &[&str]| groups.iter().map(|g| format!("[[ladder]]\nkind = \"t0\"\nbuild = \"standard\"\nrace = \"Zerg\"\ngroup = \"{g}\"\n[[ladder]]\nkind = \"t1\"\nname = \"bot-{g}\"\nrace = \"Zerg\"\ngroup = \"{g}\"\n")).collect::<String>().replacen("build = \"standard\"", "build = \"zerg_rush\"", 1);
        let two = ladder(&["rush", "macro"]);
        let ok = with_ladder("c1_ladder = 0.7\nladder_groups = { rush = 0.5, macro = 0.5 }", &two, "").unwrap_or_else(|e| panic!("{e:#}"));
        assert_eq!(ok.ladder.iter().map(|e| e.group.as_str()).collect::<Vec<_>>(), ["rush", "rush", "macro", "macro"]);
        with_ladder("c1_ladder = 0.7\nladder_groups = { rush = 1.0, macro = 0.0 }", &two, "").unwrap();
        with_ladder("c1_ladder = 0.7", &two, "").unwrap();
        with_ladder("c1_ladder = 0.7\nladder_groups = {}", &ladder(&[""]), "").unwrap();
        for (groups, why) in [
            ("{ rush = -0.5, macro = 0.5 }", "negative share"),
            ("{ rush = 0.0, macro = 0.0 }", "sum 0"),
            ("{ rush = 0.5, macro = 0.5, late = 0.2 }", "group without entries"),
            ("{ rush = 1.0 }", "macro entries not listed"),
            ("{ rush = nan, macro = 0.5 }", "not finite"),
        ] {
            assert!(with_ladder(&format!("c1_ladder = 0.7\nladder_groups = {groups}"), &two, "").is_err(), "{why}");
        }
        // With groups every entry needs one (the empty name cannot be listed).
        let mixed = format!("{two}[[ladder]]\nkind = \"t1\"\nname = \"nogroup\"\nrace = \"Zerg\"\n");
        assert!(with_ladder("c1_ladder = 0.7\nladder_groups = { rush = 0.5, macro = 0.5 }", &mixed, "").is_err());
        assert!(with_ladder("c1_ladder = 0.7\nladder_groups = { rush = 0.5, macro = 0.5, \"\" = 0.1 }", &mixed, "").is_err());
        // Names: letters, digits, '_' and '-' only (they go into games.csv), with or without groups.
        with_ladder("c1_ladder = 0.7\nladder_groups = { rush_1 = 0.5, macro-B = 0.5 }", &ladder(&["rush_1", "macro-B"]), "").unwrap();
        for bad in ["a,b", "a b", "a\\nb", "a\\\"b", "\u{e9}"] {
            assert!(with_ladder(&format!("c1_ladder = 0.7\nladder_groups = {{ \"{bad}\" = 1.0 }}"), &ladder(&[bad]), "").is_err(), "{bad:?}");
            assert!(with_ladder("c1_ladder = 0.7", &ladder(&[bad]), "").is_err(), "entry {bad:?} without groups");
        }
    }

    /// 033-R11 설정 키 검사 through the RL config: unknown modes and keys and out-of-range values
    /// are load errors.
    #[test]
    fn reward_settings_are_load_errors_033_r11() {
        let full = include_str!("../../../config/rl.toml");
        let load = |from: &str, to: &str| -> Result<RlConfig> {
            assert!(full.contains(from), "{from}");
            let c: RlConfig = toml::from_str(&full.replacen(from, to, 1))?;
            c.validate()?;
            Ok(c)
        };
        assert_eq!(load("pseudo = \"z\"", "pseudo = \"scout\"").unwrap().pseudo(), PseudoMode::Scout);
        for (from, to) in [
            ("pseudo = \"z\"", "pseudo = \"zscout\""),
            ("weight = 0.15", "weight = -0.15"),
            ("weight = 0.15", "wieght = 0.15"),
            ("vs_model = 0.5", "vs_model = 2.0"),
            ("zero_at_s = 480.0", "zero_at_s = 200.0"),
            ("until_s = 480.0", "until_s = 100.0"),
            ("scv_target = 20", "scv_target = 0"),
            ("workers_min = 4", "workers_min = 0"),
            ("main_radius_tiles = 10.0", "main_radius_tiles = 0.0"),
            ("army = 0.1", "army = -1.0"),
        ] {
            assert!(load(from, to).is_err(), "{to}");
        }
    }

    /// `config/rl.toml` with `from` replaced by `to` (once), validated.
    fn rl_with(from: &str, to: &str) -> Result<RlConfig> {
        // A Windows checkout may have CRLF line ends; `from` / `to` use LF.
        let full = include_str!("../../../config/rl.toml").replace("\r\n", "\n");
        assert!(full.contains(from), "{from}");
        let c: RlConfig = toml::from_str(&full.replacen(from, to, 1))?;
        c.validate()?;
        Ok(c)
    }

    /// 042-R9 / R15 / R16, 042 사다리 선택 (AMB-042-17): rl.toml lists the new settings at their
    /// defaults (head multipliers 1.0, no model-game cap, band 0.5 / 0.15 / 0.10, f_var, `[sil]`
    /// off); a config without them (the earlier files, configs copied into older checkpoints) gets
    /// the same values.
    #[test]
    fn amb_042_17_keys_default() {
        let c = RlConfig::default_file();
        assert_eq!(c.kl.head_scale, KlHeadScale::default());
        assert!(c.kl.head_scale.by_head().iter().all(|(_, x)| *x == 1.0));
        assert_eq!(c.league.model_game_max_s, 0);
        assert_eq!(c.curriculum.band, BandConfig { center: 0.5, width: 0.15, exclude_below: 0.10 });
        assert_eq!(c.curriculum.ladder_pick, LadderPickSetting::All(LadderPick::FVar));
        assert_eq!(c.sil, SilConfig::default());
        assert!(!c.sil.enabled);
        let s = &c.sil;
        assert_eq!((s.mode, s.select_group.as_str(), s.select_scv_4m, s.until_s, s.buffer_segments, s.lanes, s.w_pg, s.w_v, s.max_uses), (SilMode::Sil, "rush", 12, 480, 2_048, 4, 0.05, 0.01, 4), "042-R16 기본안(미확정)");
        let full = include_str!("../../../config/rl.toml");
        let cut = |text: &str, from: &str| {
            let at = text.find(from).unwrap();
            let end = at + 1 + text[at + 1..].find("\n[").unwrap();
            format!("{}{}", &text[..at], &text[end..])
        };
        let old = cut(&cut(full, "\n[kl.head_scale]"), "\n[curriculum.band]");
        let old: String = old.lines().filter(|l| !l.starts_with("model_game_max_s =")).map(|l| format!("{l}\n")).collect();
        assert!(!old.lines().any(|l| !l.starts_with('#') && (l.contains("head_scale") || l.starts_with("center =") || l.starts_with("model_game_max_s ="))));
        let o: RlConfig = toml::from_str(&old).unwrap();
        assert_eq!(o, c, "absent = defaults");
        for f in [include_str!("../../../config/rl_rush2.toml"), include_str!("../../../config/rl_rush1.toml")] {
            let o: RlConfig = toml::from_str(f).unwrap();
            o.validate().unwrap();
            assert_eq!((o.kl.head_scale.clone(), o.league.model_game_max_s, o.curriculum.band, o.sil.clone()), (KlHeadScale::default(), 0, BandConfig::default(), SilConfig::default()));
        }
    }

    /// 042-R9 KL 설정 검사 (AMB-042-17): init > 0, 0 <= floor <= init, half_life_updates > 0,
    /// head multipliers >= 0 (unknown head names are errors); the schedule from update 0.
    #[test]
    fn kl_settings_are_checked_042_r9() {
        for (to, ok) in [
            ("init = 0.03\nfloor = 0.01\nhalf_life_updates = 1500", true),
            ("init = 0.01\nfloor = 0.01\nhalf_life_updates = 1", true),
            ("init = 0.1\nfloor = 0.0\nhalf_life_updates = 20000", true),
            ("init = 0.0\nfloor = 0.0\nhalf_life_updates = 20000", false),
            ("init = -0.1\nfloor = 0.01\nhalf_life_updates = 20000", false),
            ("init = 0.1\nfloor = 0.2\nhalf_life_updates = 20000", false),
            ("init = 0.1\nfloor = -0.01\nhalf_life_updates = 20000", false),
            ("init = 0.1\nfloor = 0.01\nhalf_life_updates = 0", false),
            ("init = 0.1\nfloor = 0.01\nhalf_life_updates = nan", false),
        ] {
            assert_eq!(rl_with("init = 0.1\nfloor = 0.01\nhalf_life_updates = 20000", to).is_ok(), ok, "{to}");
        }
        let c = rl_with("init = 0.1\nfloor = 0.01\nhalf_life_updates = 20000", "init = 0.03\nfloor = 0.01\nhalf_life_updates = 1500").unwrap();
        assert!((c.kl.coef(0) - 0.03).abs() < 1e-9, "a new run's update 0 = init");
        assert!((c.kl.coef(1500) - 0.015).abs() < 1e-7);
        assert!((c.kl.coef(2_000) - 0.03 * 0.5f32.powf(2_000.0 / 1_500.0)).abs() < 1e-7 && c.kl.coef(2_000) > 0.01);
        assert!((c.kl.coef(2_400) - 0.01).abs() < 1e-9 && (c.kl.coef(1_000_000) - 0.01).abs() < 1e-9, "floor from about update 2,380");
        // `action = 1.0` is only in [kl.head_scale] ([entropy] action is 3e-3).
        let scale = "action = 1.0";
        assert_eq!(rl_with(scale, "action = 2.5").unwrap().kl.head_scale.action, 2.5);
        assert_eq!(rl_with(scale, "action = 0.0").unwrap().kl.head_scale.action, 0.0);
        assert!(rl_with(scale, "action = -0.5").is_err());
        assert!(rl_with(scale, "actions = 1.0").is_err(), "unknown head");
        let partial = rl_with("action = 1.0\ndelay = 1.0\nqueued = 1.0\nselected = 1.0", "selected = 0.5").unwrap();
        assert_eq!(partial.kl.head_scale, KlHeadScale { selected: 0.5, ..KlHeadScale::default() }, "absent heads 1.0");
    }

    /// 042-R15 설정 검사 (AMB-042-17, 메인 임시 결정): 0 or 60..=3600.
    #[test]
    fn model_game_max_s_is_checked_042_r15() {
        for (s, ok) in [(0, true), (60, true), (1_500, true), (3_600, true), (59, false), (1, false), (3_601, false)] {
            assert_eq!(rl_with("model_game_max_s = 0", &format!("model_game_max_s = {s}")).is_ok(), ok, "{s}");
        }
        assert!(rl_with("model_game_max_s = 0", "model_game_max_s = -5").is_err());
    }

    const GROUPED_LADDER: &str = "[[ladder]]\nkind = \"t0\"\nbuild = \"zerg_rush\"\nrace = \"Zerg\"\ngroup = \"rush\"\n[[ladder]]\nkind = \"t0\"\nbuild = \"standard\"\nrace = \"Zerg\"\ngroup = \"macro\"\n";

    /// 042 사다리 선택 (AMB-042-17): `ladder_pick` is a method for every group or a group table
    /// ("f_var" / "band" only, groups in `ladder_groups` only); `[curriculum.band]` ranges.
    #[test]
    fn ladder_pick_settings_are_checked_042_amb_042_17() {
        let groups = "c1_ladder = 0.7\nladder_groups = { rush = 0.5, macro = 0.5 }";
        let load = |pick: &str| with_ladder(&format!("{groups}\nladder_pick = {pick}"), GROUPED_LADDER, "");
        assert_eq!(load("\"band\"").unwrap().curriculum.ladder_pick, LadderPickSetting::All(LadderPick::Band));
        let t = load("{ rush = \"band\", macro = \"f_var\" }").unwrap().curriculum.ladder_pick;
        assert_eq!((t.for_group(Some("rush")), t.for_group(Some("macro")), t.for_group(None)), (LadderPick::Band, LadderPick::FVar, LadderPick::FVar));
        let t = load("{ rush = \"band\" }").unwrap().curriculum.ladder_pick;
        assert_eq!(t.for_group(Some("macro")), LadderPick::FVar, "not in the table: f_var");
        for bad in ["\"f_hard\"", "{ rush = \"pfsp\" }", "{ late = \"band\" }", "{ rush = 1 }", "1"] {
            assert!(load(bad).is_err(), "{bad}");
        }
        // Without ladder_groups a table may not name a group.
        assert!(with_ladder("c1_ladder = 0.7\nladder_pick = { rush = \"band\" }", RUSH_LADDER, "").is_err());
        with_ladder("c1_ladder = 0.7\nladder_pick = \"band\"", RUSH_LADDER, "").unwrap();
        let band = "center = 0.5\nwidth = 0.15\nexclude_below = 0.10";
        assert_eq!(rl_with(band, "center = 0.4\nwidth = 0.2\nexclude_below = 0.0").unwrap().curriculum.band, BandConfig { center: 0.4, width: 0.2, exclude_below: 0.0 });
        for bad in ["center = 1.5\nwidth = 0.15\nexclude_below = 0.10", "center = 0.5\nwidth = 0.0\nexclude_below = 0.10", "center = 0.5\nwidth = 0.15\nexclude_below = 1.0", "center = 0.5\nwidth = 0.15\nexclude = 0.1"] {
            assert!(rl_with(band, bad).is_err(), "{bad}");
        }
        // The serialised config (checkpoint meta.json) reads back the same, string or table.
        for pick in ["\"f_var\"", "{ rush = \"band\", macro = \"f_var\" }"] {
            let c = load(pick).unwrap();
            let back: RlConfig = serde_json::from_str(&serde_json::to_string(&c).unwrap()).unwrap();
            assert_eq!(back, c);
        }
    }

    /// 042-R16 설정 검사 (AMB-042-17): the mode and unknown keys always; the values only when
    /// enabled (select_group in ladder_groups, lanes below the batch lanes, ...).
    #[test]
    fn sil_settings_load_042_r16() {
        let groups = "c1_ladder = 0.7\nladder_groups = { rush = 0.5, macro = 0.5 }";
        let load = |sil: &str| -> Result<RlConfig> {
            let text: String = include_str!("../../../config/rl.toml").lines().filter(|l| !l.starts_with("c1_ladder =")).map(|l| format!("{l}\n")).collect();
            let text = text.replacen("[curriculum]\n", &format!("[curriculum]\n{groups}\n"), 1).replacen("\n[reward]\n", &format!("\n[sil]\n{sil}\n\n[reward]\n"), 1);
            let c: RlConfig = toml::from_str(&format!("{text}\n{GROUPED_LADDER}"))?;
            c.validate()?;
            Ok(c)
        };
        assert!(!load("enabled = false").unwrap().sil.enabled);
        let on = load("enabled = true").unwrap_or_else(|e| panic!("{e:#}"));
        assert_eq!(on.sil, SilConfig { enabled: true, ..SilConfig::default() });
        assert_eq!(load("enabled = true\nmode = \"replay\"").unwrap().sil.mode, SilMode::Replay);
        for bad in ["mode = \"imitate\"", "enable = true", "enabled = true\nselect_group = \"late\"", "enabled = true\nlanes = 32", "enabled = true\nlanes = 0", "enabled = true\nbuffer_segments = 2", "enabled = true\nw_pg = -1.0", "enabled = true\nmax_uses = 0", "enabled = true\nuntil_s = -1"] {
            assert!(load(bad).is_err(), "{bad}");
        }
        // Off: the values are not checked (they are confirmed when it is turned on).
        load("enabled = false\nlanes = 0\nselect_group = \"late\"").unwrap();
    }

    /// 042 "신호 밀도·KL run" (AMB-042-17): `config/rl_rush3.toml` is `rl_rush2.toml` with the
    /// decided changes only — shares ladder 0.90 / anchor 0.05 / snapshots 0.05 / no current Main,
    /// rush `band` (0.5, 0.15, 0.10) and macro `f_var`, model games capped at 1,500 s, KL the
    /// rl.toml schedule 0.1 / 0.01 / 20,000 (the 0.03 / 0.01 / 1,500 decision reverted after the F2
    /// diagnostics, 2026-11-04) with every head multiplier 1.0, λ 0.95, `[sil]` off; the rush-s2 ladder (7 rush
    /// and 5 macro) plus the three 50-60% rung UAlbertaBot variants in the rush group (user
    /// decision 2026-11-04 after the measurement).
    #[test]
    fn rush3_config_042_amb_042_17() {
        let t: RlConfig = toml::from_str(include_str!("../../../config/rl_rush3.toml")).unwrap();
        t.validate().unwrap();
        let cc = &t.curriculum;
        assert_eq!((cc.c1_ladder, cc.c1_anchor, cc.c1_snapshot, cc.c1_self, cc.c1_t1, cc.c1_t0), (0.90, 0.05, 0.05, 0.0, 0.0, 0.0));
        assert_eq!(cc.ladder_pick, LadderPickSetting::PerGroup([("rush".to_string(), LadderPick::Band), ("macro".to_string(), LadderPick::FVar)].into_iter().collect()));
        assert_eq!(cc.band, BandConfig { center: 0.5, width: 0.15, exclude_below: 0.10 });
        assert_eq!(t.league.model_game_max_s, 1_500);
        assert_eq!(t.league.game_max_frames(true), 36_000);
        assert_eq!((t.kl.init, t.kl.floor, t.kl.half_life_updates), (0.1, 0.01, 20_000.0), "rl.toml schedule (reverted 2026-11-04)");
        assert_eq!(t.kl, RlConfig::default_file().kl, "KL as in rl.toml");
        assert_eq!(t.kl.head_scale, KlHeadScale::default());
        assert_eq!(t.loss.td_lambda, 0.95);
        assert!(!t.sil.enabled);
        assert_eq!(t.sil, SilConfig::default());
        let rush2: RlConfig = toml::from_str(include_str!("../../../config/rl_rush2.toml")).unwrap();
        // The rush rung (user decision 2026-11-04, AMB-042-17): 9pool10lings 63%, 9pool12lings 57%,
        // 9poolhydra 43% (9pool8lings 77% not added); every rush-s2 entry kept, in order.
        let rung = ["ualbertabot9pool10lings", "ualbertabot9pool12lings", "ualbertabot9poolhydra"];
        let added: Vec<&LadderEntry> = t.ladder.iter().filter(|e| !rush2.ladder.contains(e)).collect();
        let keys: Vec<String> = added.iter().map(|e| e.opponent().unwrap().1).collect();
        assert_eq!(keys, rung.map(|n| format!("{n}-zerg")));
        assert!(added.iter().all(|e| e.group == "rush" && e.race == uranium_proto::Race::Zerg && e.kind == crate::league::LadderKind::T1));
        let kept: Vec<LadderEntry> = t.ladder.iter().filter(|e| rush2.ladder.contains(e)).cloned().collect();
        assert_eq!(kept, rush2.ladder, "the rush-s2 entries as they were");
        assert_eq!((t.ladder.iter().filter(|e| e.group == "rush").count(), t.ladder.iter().filter(|e| e.group == "macro").count()), (10, 5));
        let mut d = rush2;
        d.ladder = t.ladder.clone();
        let dc = &mut d.curriculum;
        (dc.c1_ladder, dc.c1_anchor, dc.c1_snapshot, dc.c1_self) = (0.90, 0.05, 0.05, 0.0);
        dc.ladder_pick = cc.ladder_pick.clone();
        d.league.model_game_max_s = 1_500;
        d.loss.td_lambda = 0.95;
        assert_eq!(d, t, "everything else as in rush-s2");
        let back: RlConfig = serde_json::from_str(&serde_json::to_string(&t).unwrap()).unwrap();
        assert_eq!(back, t);
    }

    /// 042-R17 / 042-AC8 (AMB-042-18): rl.toml and the earlier run files have no
    /// `[league.rush_judge]` (off: every game length as before, the serialised config without the
    /// key) and `[sil] wins_only` false; the defaults are groups [] / 480 / 8.
    #[test]
    fn rush_judge_and_wins_only_default_042_r17() {
        assert_eq!(RushJudgeConfig::default(), RushJudgeConfig { groups: vec![], at_s: 480, min_scv: 8 });
        for f in [
            include_str!("../../../config/rl.toml"),
            include_str!("../../../config/rl_rush3.toml"),
            include_str!("../../../config/rl_rush2.toml"),
            include_str!("../../../config/rl_rush1.toml"),
        ] {
            let c: RlConfig = toml::from_str(f).unwrap();
            c.validate().unwrap();
            assert_eq!(c.league.rush_judge, RushJudgeConfig::default());
            assert!(!c.sil.wins_only);
            for g in [None, Some("rush"), Some("macro"), Some("warmup")] {
                assert!(!c.league.rush_judged(false, g));
                assert_eq!(c.league.game_max_frames_of(false, g), GAME_MAX_FRAMES, "{g:?}");
                assert_eq!(c.league.game_max_frames_of(true, g), c.league.game_max_frames(true));
            }
            let json = serde_json::to_value(&c).unwrap();
            assert!(json["league"].get("rush_judge").is_none(), "checkpoint config as before");
            assert_eq!(serde_json::from_value::<RlConfig>(json).unwrap(), c);
        }
        assert!(!include_str!("../../../config/rl.toml").lines().any(|l| l.starts_with("[league.rush_judge]") || l.starts_with("wins_only")), "rl.toml: comments only");
    }

    /// 042-R17 설정 검사 / 042-AC8 (AMB-042-18, 메인 임시 결정): with groups set, every name in
    /// `ladder_groups`, `at_s` 60..=3600, `min_scv >= 1`; unknown keys are load errors; groups []
    /// checks nothing. The per-game frame cap and the SCV boundary.
    #[test]
    fn rush_judge_settings_are_checked_042_r17() {
        let base: RlConfig = toml::from_str(include_str!("../../../config/rl_rush4.toml")).unwrap();
        base.validate().unwrap();
        let with = |groups: &[&str], at_s: u32, min_scv: u32| {
            let mut c = base.clone();
            c.league.rush_judge = RushJudgeConfig { groups: groups.iter().map(|g| g.to_string()).collect(), at_s, min_scv };
            c.validate()
        };
        for (groups, at_s, min_scv, ok) in [
            (&["rush"][..], 480, 8, true),
            (&["rush", "warmup", "macro"][..], 480, 8, true),
            (&["rush"][..], 60, 1, true),
            (&["rush"][..], 3_600, 8, true),
            (&["rush"][..], 59, 8, false),
            (&["rush"][..], 3_601, 8, false),
            (&["rush"][..], 0, 8, false),
            (&["rush"][..], 480, 0, false),
            (&["late"][..], 480, 8, false),
            (&["rush", ""][..], 480, 8, false),
            (&[][..], 0, 0, true),
        ] {
            assert_eq!(with(groups, at_s, min_scv).is_ok(), ok, "{groups:?} {at_s} {min_scv}");
        }
        // Not without ladder groups either.
        let mut c = RlConfig::default_file();
        c.league.rush_judge.groups = vec!["rush".into()];
        assert!(c.validate().is_err());
        let text = include_str!("../../../config/rl_rush4.toml").replace("\r\n", "\n");
        for (from, to) in [("min_scv = 8", "min_scvs = 8"), ("at_s = 480\nmin_scv", "at_s = -1\nmin_scv"), ("groups = [\"rush\", \"warmup\"]", "groups = \"rush\"")] {
            assert!(text.contains(from), "{from}");
            assert!(toml::from_str::<RlConfig>(&text.replacen(from, to, 1)).is_err(), "{to}");
        }
        // Absent keys take the defaults.
        let only_groups: RlConfig = toml::from_str(&text.replacen("at_s = 480\nmin_scv = 8\n", "", 1)).unwrap();
        assert_eq!(only_groups.league.rush_judge, base.league.rush_judge);
        // Frame caps: judged groups min(86,400, at_s x 24), models 042-R15, the rest 86,400.
        let l = &base.league;
        for (vs_model, g, want) in [(false, Some("rush"), 11_520), (false, Some("warmup"), 11_520), (false, Some("macro"), 86_400), (false, None, 86_400), (true, None, 36_000), (true, Some("rush"), 36_000)] {
            assert_eq!(l.game_max_frames_of(vs_model, g), want, "{vs_model} {g:?}");
            assert_eq!(l.rush_judged(vs_model, g), want == 11_520);
        }
        let at = |s: u32| RushJudgeConfig { groups: vec!["rush".into()], at_s: s, min_scv: 8 }.max_frames();
        assert_eq!((at(60), at(480), at(3_600)), (1_440, 11_520, 86_400));
        // min_scv boundary.
        use uranium_env::reward::Outcome;
        let j = &l.rush_judge;
        assert_eq!((j.judge(8), j.judge(7), j.judge(0), j.judge(50)), (Outcome::Win, Outcome::Loss, Outcome::Loss, Outcome::Win));
    }

    /// 042 "러시 미니게임 run" (AMB-042-18): `config/rl_rush4.toml` is `rl_rush3.toml` with the
    /// decided changes only — groups warmup 0.05 / rush 0.65 / macro 0.20 (warmup =
    /// ualbertabot9pool; rush = T0 zerg_rush and three UAlbertaBot variants; five rush opponents
    /// left out; macro as before), win_rate_decay 0.9, `[league.rush_judge]` rush + warmup at 480 s
    /// and 8 SCVs, `[sil]` on with the confirmed numbers and `wins_only`.
    #[test]
    fn rush4_config_042_amb_042_18() {
        let t: RlConfig = toml::from_str(include_str!("../../../config/rl_rush4.toml")).unwrap();
        t.validate().unwrap();
        let groups: Vec<(&str, f64)> = t.curriculum.ladder_groups.iter().map(|(g, x)| (g.as_str(), *x)).collect();
        assert_eq!(groups, [("macro", 0.20), ("rush", 0.65), ("warmup", 0.05)]);
        let cc = &t.curriculum;
        assert_eq!((cc.c1_ladder, cc.c1_anchor, cc.c1_snapshot, cc.c1_self, cc.c1_t1, cc.c1_t0), (0.90, 0.05, 0.05, 0.0, 0.0, 0.0));
        assert_eq!(cc.ladder_pick.for_group(Some("warmup")), LadderPick::FVar);
        assert_eq!(t.league.win_rate_decay, 0.9);
        assert_eq!(t.league.rush_judge, RushJudgeConfig { groups: vec!["rush".into(), "warmup".into()], at_s: 480, min_scv: 8 });
        assert_eq!(t.league.rush_judge.max_frames(), 11_520);
        assert_eq!(t.sil, SilConfig { enabled: true, mode: SilMode::Sil, lanes: 4, w_pg: 0.05, w_v: 0.01, until_s: 480, buffer_segments: 2_048, select_group: "rush".into(), wins_only: true, ..SilConfig::default() });
        let keys = |g: &str| t.ladder.iter().filter(|e| e.group == g).map(|e| e.opponent().unwrap().1).collect::<Vec<_>>();
        assert_eq!(keys("warmup"), ["ualbertabot9pool-zerg"]);
        assert_eq!(keys("rush"), ["t0-zerg-rush", "ualbertabot12poollings-zerg", "ualbertabot9pool10lings-zerg", "ualbertabot9poolhydra-zerg"]);
        assert_eq!(keys("macro"), ["t0-zerg", "t0-zerg-muta", "ualbertabot2hatchhydra-zerg", "ualbertabot3hatchmuta-zerg", "steamhammer2hatchmuta-zerg"]);
        assert_eq!(t.ladder.len(), 10);
        // Everything else as in rush-s3.
        let mut d: RlConfig = toml::from_str(include_str!("../../../config/rl_rush3.toml")).unwrap();
        let left_out = ["ualbertabot9pool12lings", "ualbertabot9poollings", "ualbertabot", "steamhammeroverpool", "zzzkbot"];
        d.ladder.retain(|e| !e.name.as_deref().is_some_and(|n| left_out.contains(&n)));
        d.ladder.iter_mut().filter(|e| e.name.as_deref() == Some("ualbertabot9pool")).for_each(|e| e.group = "warmup".into());
        d.curriculum.ladder_groups = t.curriculum.ladder_groups.clone();
        d.league.win_rate_decay = 0.9;
        d.league.rush_judge = t.league.rush_judge.clone();
        d.sil = t.sil.clone();
        assert_eq!(d, t);
        let back: RlConfig = serde_json::from_str(&serde_json::to_string(&t).unwrap()).unwrap();
        assert_eq!(back, t, "the serialised config (checkpoint) reads back the same");
    }
}
