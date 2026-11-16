use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use uranium_env::concede::ConcedeConfig;
use uranium_env::scout::{PseudoMode, RewardConfig};

use crate::league::{BandConfig, LadderEntry, LadderPickSetting, Stage, T1Bot};
pub use crate::sil::{SilConfig, SilMode};
pub use uranium_data::bc::{AuxWeights, MonitorConfig};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct OptimizerConfig {
    pub lr: f64,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
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
    #[serde(default = "default_win_open")]
    pub win_open: f32,
}

fn default_win_open() -> f32 {
    0.5
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct EntropyConfig {
    pub action: f32,
    pub delay: f32,
    pub queued: f32,
    pub selected: f32,
    pub target_unit: f32,
    pub target_loc: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct KlConfig {
    pub init: f32,
    pub floor: f32,
    pub half_life_updates: f32,
    #[serde(default)]
    pub head_scale: KlHeadScale,
    #[serde(default, skip_serializing_if = "KlReference::is_empty")]
    pub reference: KlReference,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
#[serde(deny_unknown_fields, default)]
pub struct KlReference {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub zerg: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub terran: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub protoss: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub random: Option<String>,
}

impl KlReference {
    pub fn is_empty(&self) -> bool {
        self.by_race().iter().all(|(_, v)| v.is_none())
    }

    pub fn by_race(&self) -> [(&'static str, Option<&str>); 4] {
        [("zerg", self.zerg.as_deref()), ("terran", self.terran.as_deref()), ("protoss", self.protoss.as_deref()), ("random", self.random.as_deref())]
    }
}

impl KlConfig {
    pub fn coef(&self, update: u64) -> f32 {
        (self.init * 0.5f32.powf(update as f32 / self.half_life_updates)).max(self.floor)
    }

    pub fn validate(&self) -> Result<()> {
        let finite = [self.init, self.floor, self.half_life_updates].iter().all(|x| x.is_finite());
        if !finite || self.init <= 0.0 || self.floor < 0.0 || self.floor > self.init || self.half_life_updates <= 0.0 {
            bail!("[kl] needs init > 0, 0 <= floor <= init and half_life_updates > 0, got init {} floor {} half_life_updates {} (042-R9, AMB-042-17)", self.init, self.floor, self.half_life_updates);
        }
        if let Some((h, x)) = self.head_scale.by_head().into_iter().find(|(_, x)| !(*x >= 0.0 && x.is_finite())) {
            bail!("[kl.head_scale] {h} must be a finite value >= 0, got {x} (042-R9, AMB-042-17)");
        }
        if let Some((race, _)) = self.reference.by_race().into_iter().find(|(_, v)| v.is_some_and(|p| p.trim().is_empty())) {
            bail!("[kl.reference] {race} is empty: give a checkpoint folder or leave the key out (the --bc model, 044-R9)");
        }
        Ok(())
    }
}

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
    pub fn by_head(&self) -> [(&'static str, f32); 6] {
        [("action", self.action), ("delay", self.delay), ("queued", self.queued), ("selected", self.selected), ("target_unit", self.target_unit), ("target_loc", self.target_loc)]
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum QueueFull {
    #[default]
    Wait,
    DropOldest,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ActorsConfig {
    pub weight_push_every: u64,
    pub max_policy_lag: u64,
    pub queue_batches: usize,
    #[serde(default)]
    pub queue_full: QueueFull,
    #[serde(default = "default_queue_wait_max_s")]
    pub queue_wait_max_s: u64,
    #[serde(default = "default_game_wall_timeout_s")]
    pub game_wall_timeout_s: u32,
    #[serde(default = "default_dead_learner_minutes", skip_serializing_if = "is_default_dead_learner_minutes")]
    pub dead_learner_minutes: u64,
}

fn default_queue_wait_max_s() -> u64 {
    120
}

pub const DEFAULT_DEAD_LEARNER_MINUTES: u64 = 15;

fn default_dead_learner_minutes() -> u64 {
    DEFAULT_DEAD_LEARNER_MINUTES
}

fn is_default_dead_learner_minutes(m: &u64) -> bool {
    *m == DEFAULT_DEAD_LEARNER_MINUTES
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
    pub win_rate_decay: f64,
    pub win_rate_prior: f64,
    pub race_uniform_mix: f64,
    pub bot_random_race: f64,
    #[serde(default)]
    pub anchor: String,
    #[serde(default)]
    pub model_game_max_s: u32,
    #[serde(default, skip_serializing_if = "RushJudgeConfig::is_default")]
    pub rush_judge: RushJudgeConfig,
    #[serde(default, skip_serializing_if = "StartMixConfig::is_default")]
    pub start_mix: StartMixConfig,
}

pub const GAME_MAX_FRAMES: u32 = 86_400;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(deny_unknown_fields, default)]
pub struct StartMixConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rush_hit_share: Option<f64>,
    pub groups: Vec<String>,
}

impl Default for StartMixConfig {
    fn default() -> Self {
        Self { rush_hit_share: None, groups: vec!["rush".into()] }
    }
}

impl StartMixConfig {
    fn is_default(&self) -> bool {
        *self == Self::default()
    }

    pub fn share_for(&self, ladder_group: Option<&str>) -> Option<f64> {
        let g = ladder_group?;
        self.rush_hit_share.filter(|_| self.groups.iter().any(|x| x == g))
    }

    pub fn validate(&self, ladder_groups: &std::collections::BTreeMap<String, f64>) -> Result<()> {
        let Some(x) = self.rush_hit_share else { return Ok(()) };
        if !(0.0..=1.0).contains(&x) {
            bail!("[league.start_mix] rush_hit_share must be in 0..=1, got {x} (042-R18, AMB-042-19)");
        }
        if self.groups.is_empty() {
            bail!("[league.start_mix] groups is empty (042-R18, AMB-042-19)");
        }
        if let Some(g) = self.groups.iter().find(|g| !ladder_groups.contains_key(*g)) {
            bail!("[league.start_mix] groups names {g:?}, which is not in [curriculum] ladder_groups (042-R18, AMB-042-19)");
        }
        Ok(())
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(deny_unknown_fields, default)]
pub struct RushJudgeConfig {
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

    pub fn judges(&self, ladder_group: Option<&str>) -> bool {
        ladder_group.is_some_and(|g| self.groups.iter().any(|x| x == g))
    }

    pub fn max_frames(&self) -> u32 {
        GAME_MAX_FRAMES.min(self.at_s.saturating_mul(24))
    }

    pub fn judge(&self, scv: u32) -> uranium_env::reward::Outcome {
        if scv >= self.min_scv { uranium_env::reward::Outcome::Win } else { uranium_env::reward::Outcome::Loss }
    }

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
    pub fn game_max_frames(&self, vs_model: bool) -> u32 {
        if vs_model && self.model_game_max_s > 0 { GAME_MAX_FRAMES.min(self.model_game_max_s.saturating_mul(24)) } else { GAME_MAX_FRAMES }
    }

    pub fn rush_judged(&self, vs_model: bool, ladder_group: Option<&str>) -> bool {
        !vs_model && self.rush_judge.judges(ladder_group)
    }

    pub fn game_max_frames_of(&self, vs_model: bool, ladder_group: Option<&str>) -> u32 {
        if self.rush_judged(vs_model, ladder_group) { self.rush_judge.max_frames() } else { self.game_max_frames(vs_model) }
    }

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

pub fn has_model_files(d: &Path) -> bool {
    d.join(uranium_agent::infer::WEIGHTS_FILE).is_file() && d.join("meta.json").is_file()
}

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
    #[serde(default = "default_start_stage")]
    pub start_stage: Stage,
    #[serde(default = "default_auto_advance")]
    pub auto_advance: bool,
    #[serde(default = "default_r_value_stages")]
    pub r_value_stages: Vec<Stage>,
    #[serde(default)]
    pub c1_snapshot: f64,
    #[serde(default)]
    pub c1_anchor: f64,
    #[serde(default = "default_r_value_vs_model")]
    pub r_value_vs_model: bool,
    #[serde(default)]
    pub c1_ladder: f64,
    #[serde(default)]
    pub ladder_pick: LadderPickSetting,
    #[serde(default)]
    pub band: BandConfig,
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

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ReplaysConfig {
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
    #[serde(default)]
    pub replays: ReplaysConfig,
    #[serde(default)]
    pub keep_checkpoints: Option<usize>,
    #[serde(default)]
    pub t1: Vec<T1Bot>,
    #[serde(default)]
    pub ladder: Vec<LadderEntry>,
    #[serde(default)]
    pub concede: ConcedeConfig,
    #[serde(default)]
    pub reward: RewardConfig,
    #[serde(default)]
    pub sil: SilConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aux: Option<AuxWeights>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub monitor: Option<MonitorConfig>,
}

impl RlConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let c: Self = toml::from_str(&std::fs::read_to_string(path)?)?;
        c.validate().with_context(|| format!("{}", path.display()))?;
        Ok(c)
    }

    pub fn validate(&self) -> Result<()> {
        if self.actors.queue_batches < 1 {
            bail!("[actors] queue_batches must be at least 1 (042 queue table)");
        }
        if !(1..600).contains(&self.actors.queue_wait_max_s) {
            bail!("[actors] queue_wait_max_s must be at least 1 and below 600 (042 queue table, 011-R18)");
        }
        if self.actors.game_wall_timeout_s < 1 {
            bail!("[actors] game_wall_timeout_s must be at least 1 (042, 020-R3)");
        }
        if !matches!(self.curriculum.start_stage, Stage::C0 | Stage::C1) {
            bail!("[curriculum] start_stage must be \"C0\" or \"C1\" (042-R5)");
        }
        if self.t1.is_empty() && (self.curriculum.c1_t1 > 0.0 || self.league.main_t1 > 0.0) {
            bail!("[[t1]] is empty but [curriculum] c1_t1 or [league] main_t1 is above 0 (042, 021)");
        }
        if let Some(b) = self.t1.iter().find(|b| b.races.is_empty()) {
            bail!("[[t1]] {} has no races (021)", b.name);
        }
        if self.curriculum.c1_anchor > 0.0 && self.league.anchor.trim().is_empty() {
            bail!("[curriculum] c1_anchor is above 0 but [league] anchor is empty (042, AMB-042-14)");
        }
        if self.curriculum.c1_ladder > 0.0 && self.ladder.is_empty() {
            bail!("[curriculum] c1_ladder is above 0 but [[ladder]] is empty (042, AMB-042-15)");
        }
        let mut keys = std::collections::BTreeSet::new();
        for (i, e) in self.ladder.iter().enumerate() {
            let (_, key, _) = e.opponent().map_err(|why| anyhow::anyhow!("[[ladder]] entry {}: {why}", i + 1))?;
            if !keys.insert(key.clone()) {
                bail!("[[ladder]] entry {}: win-table key {key} appears twice (042, AMB-042-15)", i + 1);
            }
            if e.race == uranium_proto::Race::Random && e.kind == crate::league::LadderKind::T1 {
                use uranium_proto::Race;
                let name = e.name.as_deref().unwrap_or_default();
                if !self.t1.iter().any(|b| b.name == name && [Race::Terran, Race::Protoss, Race::Zerg].iter().all(|r| b.races.contains(r))) {
                    bail!("[[ladder]] entry {}: race Random needs a [[t1]] bot named {name:?} listing Terran, Protoss and Zerg (021-R7)", i + 1);
                }
            }
        }
        let groups = &self.curriculum.ladder_groups;
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
        if let LadderPickSetting::PerGroup(m) = &self.curriculum.ladder_pick
            && let Some(g) = m.keys().find(|g| !groups.contains_key(*g))
        {
            bail!("[curriculum] ladder_pick names group {g:?}, which is not in ladder_groups (042, AMB-042-17)");
        }
        self.curriculum.band.validate().map_err(anyhow::Error::msg)?;
        let group_of = |e: &LadderEntry| if groups.is_empty() { None } else { Some(e.group.clone()) };
        for (i, e) in self.ladder.iter().enumerate() {
            if let Some(w) = e.weight {
                if !(w >= 0.0 && w.is_finite()) {
                    bail!("[[ladder]] entry {}: weight must be a finite value >= 0, got {w} (042-R19, AMB-042-19)", i + 1);
                }
                if self.curriculum.ladder_pick.for_group(group_of(e).as_deref()) != crate::league::LadderPick::Fixed {
                    bail!("[[ladder]] entry {}: weight is set but its group {:?} is not picked \"fixed\" (042-R19, AMB-042-19)", i + 1, e.group);
                }
            }
        }
        let fixed_groups: std::collections::BTreeSet<Option<String>> = self.ladder.iter().map(group_of).filter(|g| self.curriculum.ladder_pick.for_group(g.as_deref()) == crate::league::LadderPick::Fixed).collect();
        for g in fixed_groups {
            let sum: f64 = self.ladder.iter().filter(|e| group_of(e) == g).map(LadderEntry::fixed_weight).sum();
            if sum <= 0.0 {
                bail!("[[ladder]] weights of the \"fixed\" group {:?} sum to 0 (042-R19, AMB-042-19)", g.unwrap_or_default());
            }
        }
        self.league.start_mix.validate(groups)?;
        if let Some(a) = &self.aux {
            a.check()?;
        }
        self.kl.validate()?;
        let cap = self.league.model_game_max_s;
        if cap != 0 && !(60..=3_600).contains(&cap) {
            bail!("[league] model_game_max_s must be 0 (no cap) or 60..=3600 game seconds, got {cap} (042-R15, AMB-042-17)");
        }
        self.league.rush_judge.validate(groups)?;
        self.sil.validate(self.lanes, groups).map_err(anyhow::Error::msg)?;
        if self.sil.enabled
            && let Some(keys) = &self.sil.select_keys
        {
            let group_keys: Vec<String> = self.ladder.iter().filter(|e| e.group == self.sil.select_group).filter_map(|e| e.opponent().ok().map(|(_, k, _)| k)).collect();
            if let Some(k) = keys.iter().find(|k| !group_keys.contains(k)) {
                bail!("[sil] select_keys names {k:?}, which is not the win-table key of a [[ladder]] entry of select_group {:?} ({}) (042-R16, AMB-042-19)", self.sil.select_group, group_keys.join(", "));
            }
        }
        self.concede.validate().map_err(anyhow::Error::msg)?;
        self.reward.validate()?;
        Ok(())
    }

    pub fn pseudo(&self) -> PseudoMode {
        self.reward.pseudo
    }

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

    #[test]
    fn dead_learner_minutes_default_042_r20() {
        let base = "weight_push_every = 4\nmax_policy_lag = 32\nqueue_batches = 4\n";
        let c = with_actors(base).unwrap();
        assert_eq!(c.actors.dead_learner_minutes, DEFAULT_DEAD_LEARNER_MINUTES);
        assert_eq!(DEFAULT_DEAD_LEARNER_MINUTES, 15);
        assert!(!serde_json::to_string(&c).unwrap().contains("dead_learner_minutes"));
        let c = with_actors(&format!("{base}dead_learner_minutes = 0")).unwrap();
        assert_eq!(c.actors.dead_learner_minutes, 0);
        assert!(serde_json::to_string(&c).unwrap().contains("\"dead_learner_minutes\":0"));
        assert_eq!(RlConfig::default_file().actors.dead_learner_minutes, 15);
    }

    const NEW_CURRICULUM_KEYS: [&str; 3] = ["start_stage", "auto_advance", "r_value_stages"];

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
        for r in [uranium_proto::Race::Terran, uranium_proto::Race::Protoss, uranium_proto::Race::Zerg] {
            assert_eq!(t.t1.iter().filter(|b| b.races.contains(&r)).count(), 3, "{r:?}");
        }
        let mut d = RlConfig::default_file();
        let dc = &mut d.curriculum;
        (dc.start_stage, dc.auto_advance, dc.r_value_stages) = (cc.start_stage, cc.auto_advance, cc.r_value_stages.clone());
        (dc.c1_t0, dc.c1_t1, dc.c1_self) = (cc.c1_t0, cc.c1_t1, cc.c1_self);
        d.t1.retain(|b| b.name != "mcrave");
        assert_eq!(d, t);
    }

    const AMB_042_14_KEYS: [&str; 5] = ["c1_snapshot", "c1_anchor", "r_value_vs_model", "anchor", "main_candidates"];

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

    fn with_ladder(curriculum: &str, extra: &str, concede: &str) -> Result<RlConfig> {
        let text: String = include_str!("../../../config/rl.toml").lines().filter(|l| !l.starts_with("c1_ladder =") && !l.starts_with("ladder_pick =")).map(|l| format!("{l}\n")).collect();
        let text = text.replace("[curriculum]\n", &format!("[curriculum]\n{curriculum}\n"));
        let start = text.find("\n[concede]").unwrap() + 1;
        let end = start + text[start..].find("\n[[t1]]").unwrap();
        let text = format!("{}[concede]\n{concede}{}\n{extra}", &text[..start], &text[end..]);
        let c: RlConfig = toml::from_str(&text)?;
        c.validate()?;
        Ok(c)
    }

    const RUSH_LADDER: &str = "[[ladder]]\nkind = \"t0\"\nbuild = \"zerg_rush\"\nrace = \"Zerg\"\n[[ladder]]\nkind = \"t1\"\nname = \"zzzkbot\"\nrace = \"Zerg\"\n";

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
        assert!(entry("kind = \"t0\"\nbuild = \"zerg_rush\"\nrace = \"Random\"").is_err(), "Random one-race build");
        assert!(entry("kind = \"t1\"\nname = \"zzzkbot\"\nrace = \"Random\"").is_err(), "Random one-race bot");
        assert!(entry("kind = \"t1\"\nname = \"ualbertabot9pool10lings\"\nrace = \"Random\"").is_err(), "Random bot not in [[t1]]");
        assert!(entry("kind = \"t1\"\nname = \"ualbertabot\"\nrace = \"Unknown\"").is_err(), "Unknown");
        let r = entry("kind = \"t0\"\nbuild = \"standard\"\nrace = \"Random\"").unwrap_or_else(|e| panic!("{e:#}"));
        assert_eq!(r.ladder[0].opponent().unwrap().1, "t0-random");
        let r = entry("kind = \"t1\"\nname = \"ualbertabot\"\nrace = \"Random\"").unwrap_or_else(|e| panic!("{e:#}"));
        assert_eq!(r.ladder[0].opponent().unwrap().1, "ualbertabot-random");
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
        let t: RlConfig = toml::from_str(include_str!("../../../config/rl_rush2.toml")).unwrap();
        let back: RlConfig = serde_json::from_str(&serde_json::to_string(&t).unwrap()).unwrap();
        assert_eq!(back, t);
    }

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
        let mixed = format!("{two}[[ladder]]\nkind = \"t1\"\nname = \"nogroup\"\nrace = \"Zerg\"\n");
        assert!(with_ladder("c1_ladder = 0.7\nladder_groups = { rush = 0.5, macro = 0.5 }", &mixed, "").is_err());
        assert!(with_ladder("c1_ladder = 0.7\nladder_groups = { rush = 0.5, macro = 0.5, \"\" = 0.1 }", &mixed, "").is_err());
        with_ladder("c1_ladder = 0.7\nladder_groups = { rush_1 = 0.5, macro-B = 0.5 }", &ladder(&["rush_1", "macro-B"]), "").unwrap();
        for bad in ["a,b", "a b", "a\\nb", "a\\\"b", "\u{e9}"] {
            assert!(with_ladder(&format!("c1_ladder = 0.7\nladder_groups = {{ \"{bad}\" = 1.0 }}"), &ladder(&[bad]), "").is_err(), "{bad:?}");
            assert!(with_ladder("c1_ladder = 0.7", &ladder(&[bad]), "").is_err(), "entry {bad:?} without groups");
        }
    }

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

    #[test]
    fn aux_and_monitor_tables_042_r21() {
        let c = RlConfig::default_file();
        assert!(c.aux.is_none() && c.monitor.is_none());
        let json = serde_json::to_value(&c).unwrap();
        assert!(json.get("aux").is_none() && json.get("monitor").is_none());
        let ck: RlConfig = toml::from_str(include_str!("../../../releases/checkpoint-1-rl/config/rl.toml")).unwrap();
        assert!(ck.aux.is_none() && ck.monitor.is_none());
        let back: RlConfig = serde_json::from_str(&serde_json::to_string(&ck).unwrap()).unwrap();
        assert_eq!(back, ck);
        let full = include_str!("../../../config/rl.toml").replace("\r\n", "\n");
        let on: RlConfig = toml::from_str(&format!("{full}\n[aux]\npriv = 0.2\njepa = 0.1\n\n[monitor]\nspatial_every = 100\n")).unwrap();
        on.validate().unwrap();
        assert_eq!(on.aux, Some(AuxWeights { privileged: Some(0.2), jepa: Some(0.1) }));
        assert_eq!(on.monitor, Some(MonitorConfig { spatial_every: 100 }));
        let json = serde_json::to_value(&on).unwrap();
        assert_eq!((json["aux"]["priv"].as_f64().map(|x| x as f32), json["monitor"]["spatial_every"].as_u64()), (Some(0.2), Some(100)));
        let bad: RlConfig = toml::from_str(&format!("{full}\n[aux]\npriv = -0.2\n")).unwrap();
        assert!(bad.validate().is_err());
        let empty: RlConfig = toml::from_str(&format!("{full}
[aux]
")).unwrap();
        assert!(empty.validate().is_err(), "an empty [aux] table is refused");
        assert!(toml::from_str::<RlConfig>(&format!("{full}\n[aux]\nprivileged = 0.2\n")).is_err());
    }

    #[test]
    fn rl_m2_config_042_r21() {
        let mut t: RlConfig = toml::from_str(include_str!("../../../config/rl_m2.toml")).unwrap();
        t.validate().unwrap();
        assert_eq!(t.aux, Some(AuxWeights { privileged: Some(0.2), jepa: Some(0.1) }), "AMB-041-13b / AMB-042-21");
        assert_eq!(t.monitor, Some(MonitorConfig { spatial_every: 100 }), "AMB-041-14");
        (t.aux, t.monitor) = (None, None);
        assert_eq!(t, RlConfig::default_file(), "everything else as config/rl.toml");
    }

    fn rl_with(from: &str, to: &str) -> Result<RlConfig> {
        let full = include_str!("../../../config/rl.toml").replace("\r\n", "\n");
        assert!(full.contains(from), "{from}");
        let c: RlConfig = toml::from_str(&full.replacen(from, to, 1))?;
        c.validate()?;
        Ok(c)
    }

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
        let scale = "action = 1.0";
        assert_eq!(rl_with(scale, "action = 2.5").unwrap().kl.head_scale.action, 2.5);
        assert_eq!(rl_with(scale, "action = 0.0").unwrap().kl.head_scale.action, 0.0);
        assert!(rl_with(scale, "action = -0.5").is_err());
        assert!(rl_with(scale, "actions = 1.0").is_err(), "unknown head");
        let partial = rl_with("action = 1.0\ndelay = 1.0\nqueued = 1.0\nselected = 1.0", "selected = 0.5").unwrap();
        assert_eq!(partial.kl.head_scale, KlHeadScale { selected: 0.5, ..KlHeadScale::default() }, "absent heads 1.0");
    }

    #[test]
    fn kl_reference_table_044_r9() {
        let c = RlConfig::default_file();
        assert!(c.kl.reference.is_empty());
        assert!(!serde_json::to_string(&c).unwrap().contains("reference"), "absent: the JSON as before");
        let tail = "target_loc = 1.0\n";
        let with = |body: &str| rl_with(tail, &format!("target_loc = 1.0\n\n[kl.reference]\n{body}\n"));
        let r = with("zerg = \"checkpoints/s/ckpt-1\"\nrandom = \"checkpoints/b/ckpt-2\"").unwrap().kl.reference;
        assert_eq!(r, KlReference { zerg: Some("checkpoints/s/ckpt-1".into()), random: Some("checkpoints/b/ckpt-2".into()), ..Default::default() });
        assert_eq!(r.by_race().map(|(k, _)| k), ["zerg", "terran", "protoss", "random"]);
        let js: serde_json::Value = serde_json::from_str(&serde_json::to_string(&with("zerg = \"a\"").unwrap()).unwrap()).unwrap();
        assert_eq!(js["kl"]["reference"], serde_json::json!({"zerg": "a"}), "only the given keys");
        assert!(with("").unwrap().kl.reference.is_empty(), "an empty table = none");
        assert!(with("zergs = \"a\"").is_err(), "unknown race key");
        assert!(with("terran = \"  \"").is_err(), "empty value");
    }

    #[test]
    fn model_game_max_s_is_checked_042_r15() {
        for (s, ok) in [(0, true), (60, true), (1_500, true), (3_600, true), (59, false), (1, false), (3_601, false)] {
            assert_eq!(rl_with("model_game_max_s = 0", &format!("model_game_max_s = {s}")).is_ok(), ok, "{s}");
        }
        assert!(rl_with("model_game_max_s = 0", "model_game_max_s = -5").is_err());
    }

    const GROUPED_LADDER: &str = "[[ladder]]\nkind = \"t0\"\nbuild = \"zerg_rush\"\nrace = \"Zerg\"\ngroup = \"rush\"\n[[ladder]]\nkind = \"t0\"\nbuild = \"standard\"\nrace = \"Zerg\"\ngroup = \"macro\"\n";

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
        assert!(with_ladder("c1_ladder = 0.7\nladder_pick = { rush = \"band\" }", RUSH_LADDER, "").is_err());
        with_ladder("c1_ladder = 0.7\nladder_pick = \"band\"", RUSH_LADDER, "").unwrap();
        let band = "center = 0.5\nwidth = 0.15\nexclude_below = 0.10";
        assert_eq!(rl_with(band, "center = 0.4\nwidth = 0.2\nexclude_below = 0.0").unwrap().curriculum.band, BandConfig { center: 0.4, width: 0.2, exclude_below: 0.0 });
        for bad in ["center = 1.5\nwidth = 0.15\nexclude_below = 0.10", "center = 0.5\nwidth = 0.0\nexclude_below = 0.10", "center = 0.5\nwidth = 0.15\nexclude_below = 1.0", "center = 0.5\nwidth = 0.15\nexclude = 0.1"] {
            assert!(rl_with(band, bad).is_err(), "{bad}");
        }
        for pick in ["\"f_var\"", "{ rush = \"band\", macro = \"f_var\" }"] {
            let c = load(pick).unwrap();
            let back: RlConfig = serde_json::from_str(&serde_json::to_string(&c).unwrap()).unwrap();
            assert_eq!(back, c);
        }
    }

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
        load("enabled = false\nlanes = 0\nselect_group = \"late\"").unwrap();
    }

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
        assert!(!include_str!("../../../config/rl.toml").lines().any(|l| l.starts_with("[league.rush_judge]") || l.starts_with("wins_only")), "rl.toml: not set");
    }

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
        let mut c = RlConfig::default_file();
        c.league.rush_judge.groups = vec!["rush".into()];
        assert!(c.validate().is_err());
        let text = include_str!("../../../config/rl_rush4.toml").replace("\r\n", "\n");
        for (from, to) in [("\nmin_scv = 8\n", "\nmin_scvs = 8\n"), ("at_s = 480\nmin_scv", "at_s = -1\nmin_scv"), ("groups = [\"rush\", \"warmup\"]", "groups = \"rush\"")] {
            assert!(text.contains(from), "{from}");
            assert!(toml::from_str::<RlConfig>(&text.replacen(from, to, 1)).is_err(), "{to}");
        }
        let only_groups: RlConfig = toml::from_str(&text.replacen("at_s = 480\nmin_scv = 8\n", "", 1)).unwrap();
        assert_eq!(only_groups.league.rush_judge, base.league.rush_judge);
        let l = &base.league;
        for (vs_model, g, want) in [(false, Some("rush"), 11_520), (false, Some("warmup"), 11_520), (false, Some("macro"), 86_400), (false, None, 86_400), (true, None, 36_000), (true, Some("rush"), 36_000)] {
            assert_eq!(l.game_max_frames_of(vs_model, g), want, "{vs_model} {g:?}");
            assert_eq!(l.rush_judged(vs_model, g), want == 11_520);
        }
        let at = |s: u32| RushJudgeConfig { groups: vec!["rush".into()], at_s: s, min_scv: 8 }.max_frames();
        assert_eq!((at(60), at(480), at(3_600)), (1_440, 11_520, 86_400));
        use uranium_env::reward::Outcome;
        let j = &l.rush_judge;
        assert_eq!((j.judge(8), j.judge(7), j.judge(0), j.judge(50)), (Outcome::Win, Outcome::Loss, Outcome::Loss, Outcome::Win));
    }

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
        assert_eq!(cc.band, BandConfig { center: 0.5, width: 0.30, exclude_below: 0.0 });
        assert_eq!(t.league.rush_judge, RushJudgeConfig { groups: vec!["rush".into(), "warmup".into()], at_s: 480, min_scv: 8 });
        assert_eq!(t.league.rush_judge.max_frames(), 11_520);
        assert_eq!(t.sil, SilConfig { enabled: true, mode: SilMode::Sil, lanes: 4, w_pg: 0.05, w_v: 0.01, until_s: 480, buffer_segments: 256, select_group: "rush".into(), wins_only: true, ..SilConfig::default() });
        let keys = |g: &str| t.ladder.iter().filter(|e| e.group == g).map(|e| e.opponent().unwrap().1).collect::<Vec<_>>();
        assert_eq!(keys("warmup"), ["ualbertabot9pool-zerg"]);
        assert_eq!(keys("rush"), ["t0-zerg-rush", "ualbertabot12poollings-zerg", "ualbertabot9pool10lings-zerg", "ualbertabot9poolhydra-zerg"]);
        assert_eq!(keys("macro"), ["t0-zerg", "t0-zerg-muta", "ualbertabot2hatchhydra-zerg", "ualbertabot3hatchmuta-zerg", "steamhammer2hatchmuta-zerg"]);
        assert_eq!(t.ladder.len(), 10);
        let mut d: RlConfig = toml::from_str(include_str!("../../../config/rl_rush3.toml")).unwrap();
        let left_out = ["ualbertabot9pool12lings", "ualbertabot9poollings", "ualbertabot", "steamhammeroverpool", "zzzkbot"];
        d.ladder.retain(|e| !e.name.as_deref().is_some_and(|n| left_out.contains(&n)));
        d.ladder.iter_mut().filter(|e| e.name.as_deref() == Some("ualbertabot9pool")).for_each(|e| e.group = "warmup".into());
        d.curriculum.ladder_groups = t.curriculum.ladder_groups.clone();
        d.league.win_rate_decay = 0.9;
        d.curriculum.band = BandConfig { center: 0.5, width: 0.30, exclude_below: 0.0 };
        d.league.rush_judge = t.league.rush_judge.clone();
        d.sil = t.sil.clone();
        assert_eq!(d, t);
        let back: RlConfig = serde_json::from_str(&serde_json::to_string(&t).unwrap()).unwrap();
        assert_eq!(back, t, "the serialised config (checkpoint) reads back the same");
    }

    #[test]
    fn amb_042_19_keys_default() {
        assert_eq!(StartMixConfig::default(), StartMixConfig { rush_hit_share: None, groups: vec!["rush".into()] });
        for f in [
            include_str!("../../../config/rl.toml"),
            include_str!("../../../config/rl_rush1.toml"),
            include_str!("../../../config/rl_rush2.toml"),
            include_str!("../../../config/rl_rush3.toml"),
            include_str!("../../../config/rl_rush4.toml"),
        ] {
            let c: RlConfig = toml::from_str(f).unwrap();
            c.validate().unwrap();
            assert_eq!(c.league.start_mix, StartMixConfig::default());
            for g in [None, Some("rush"), Some("macro"), Some("warmup")] {
                assert_eq!(c.league.start_mix.share_for(g), None);
                assert_ne!(c.curriculum.ladder_pick.for_group(g), LadderPick::Fixed);
            }
            assert!(c.ladder.iter().all(|e| e.weight.is_none()));
            assert_eq!((c.sil.select_keys.as_ref(), c.sil.select_army_before_s, c.sil.select_start_hit), (None, None, None));
            let json = serde_json::to_value(&c).unwrap();
            assert!(json["league"].get("start_mix").is_none() && json["sil"].get("select_keys").is_none() && json["sil"].get("select_army_before_s").is_none() && json["sil"].get("select_start_hit").is_none());
            assert!(json["ladder"].as_array().unwrap().iter().all(|e| e.get("weight").is_none()), "checkpoint config as before");
        }
        let rl = include_str!("../../../config/rl.toml");
        assert!(!rl.lines().any(|l| ["[league.start_mix]", "rush_hit_share", "select_keys", "select_army_before_s", "select_start_hit", "[[ladder]]"].iter().any(|k| l.starts_with(k))), "rl.toml: not set");
    }

    #[test]
    fn amb_042_19_settings_are_checked() {
        let base: RlConfig = toml::from_str(include_str!("../../../config/rl_rush5.toml")).unwrap();
        base.validate().unwrap();
        let check = |f: &dyn Fn(&mut RlConfig)| {
            let mut c = base.clone();
            f(&mut c);
            c.validate()
        };
        for (share, ok) in [(0.0, true), (1.0, true), (0.5, true), (-0.01, false), (1.01, false), (f64::NAN, false)] {
            assert_eq!(check(&|c| c.league.start_mix.rush_hit_share = Some(share)).is_ok(), ok, "{share}");
        }
        assert!(check(&|c| c.league.start_mix.groups = vec![]).is_err(), "empty groups");
        assert!(check(&|c| c.league.start_mix.groups = vec!["late".into()]).is_err(), "not in ladder_groups");
        check(&|c| c.league.start_mix.groups = vec!["rush".into(), "macro".into()]).unwrap();
        check(&|c| (c.league.start_mix.rush_hit_share, c.league.start_mix.groups) = (None, vec![])).unwrap();
        let mut no_groups = RlConfig::default_file();
        no_groups.league.start_mix.rush_hit_share = Some(0.5);
        assert!(no_groups.validate().is_err(), "no ladder_groups");
        fn rush(c: &mut RlConfig) -> Vec<&mut LadderEntry> {
            c.ladder.iter_mut().filter(|e| e.group == "rush").collect()
        }
        assert!(check(&|c| rush(c)[0].weight = Some(-0.1)).is_err(), "negative weight");
        assert!(check(&|c| rush(c)[0].weight = Some(f64::INFINITY)).is_err());
        assert!(check(&|c| rush(c).into_iter().for_each(|e| e.weight = Some(0.0))).is_err(), "fixed group sums to 0");
        check(&|c| rush(c)[0].weight = Some(0.0)).unwrap();
        check(&|c| rush(c).into_iter().for_each(|e| e.weight = None)).unwrap();
        assert!(check(&|c| c.ladder.iter_mut().find(|e| e.group == "macro").unwrap().weight = Some(1.0)).is_err(), "weight on a f_var group");
        assert!(check(&|c| c.curriculum.ladder_pick = LadderPickSetting::PerGroup([("rush".to_string(), LadderPick::Band)].into_iter().collect())).is_err(), "weights without fixed");
        check(&|c| c.curriculum.ladder_pick = LadderPickSetting::All(LadderPick::Fixed)).unwrap();
        let mut flat = RlConfig::default_file();
        flat.curriculum.c1_ladder = 0.5;
        flat.ladder = base.ladder.iter().map(|e| LadderEntry { group: String::new(), ..e.clone() }).collect();
        assert!(flat.validate().is_err(), "weights with the default f_var");
        flat.curriculum.ladder_pick = LadderPickSetting::All(LadderPick::Fixed);
        flat.validate().unwrap();
        assert!(check(&|c| c.sil.select_keys = Some(vec!["t0-zerg".into()])).is_err(), "a macro key");
        assert!(check(&|c| c.sil.select_keys = Some(vec!["nobody-zerg".into()])).is_err());
        assert!(check(&|c| c.sil.select_keys = Some(vec![])).is_err());
        assert!(check(&|c| c.sil.select_army_before_s = Some(-1.0)).is_err());
        check(&|c| c.sil.select_keys = Some(vec!["ualbertabot9poolhydra-zerg".into()])).unwrap();
        check(&|c| (c.sil.select_keys, c.sil.select_army_before_s) = (None, None)).unwrap();
        let text = include_str!("../../../config/rl_rush5.toml").replace("\r\n", "\n");
        for (from, to) in [("\nrush_hit_share = 0.5\n", "\nrush_hit_shares = 0.5\n"), ("\nrush_hit_share = 0.5\n", "\nrush_hit_share = \"half\"\n"), ("\nselect_army_before_s = 225\n", "\nselect_army_before = 225\n"), ("\nweight = 0.35\n", "\nweight = \"0.35\"\n"), ("\nladder_pick = { rush = \"fixed\"", "\nladder_pick = { rush = \"fix\"")] {
            assert!(text.contains(from), "{from}");
            assert!(toml::from_str::<RlConfig>(&text.replacen(from, to, 1)).is_err(), "{to}");
        }
    }

    #[test]
    fn rush5_config_042_amb_042_19() {
        let t: RlConfig = toml::from_str(include_str!("../../../config/rl_rush5.toml")).unwrap();
        t.validate().unwrap();
        let groups: Vec<(&str, f64)> = t.curriculum.ladder_groups.iter().map(|(g, x)| (g.as_str(), *x)).collect();
        assert_eq!(groups, [("macro", 0.20), ("rush", 0.70)]);
        let cc = &t.curriculum;
        assert_eq!((cc.c1_ladder, cc.c1_anchor, cc.c1_snapshot, cc.c1_self, cc.c1_t1, cc.c1_t0), (0.90, 0.05, 0.05, 0.0, 0.0, 0.0));
        assert_eq!((cc.ladder_pick.for_group(Some("rush")), cc.ladder_pick.for_group(Some("macro"))), (LadderPick::Fixed, LadderPick::FVar));
        let weights: Vec<(String, Option<f64>)> = t.ladder.iter().filter(|e| e.group == "rush").map(|e| (e.opponent().unwrap().1, e.weight)).collect();
        assert_eq!(weights, [("t0-zerg-rush".to_string(), Some(0.35)), ("ualbertabot12poollings-zerg".into(), Some(0.30)), ("ualbertabot9pool10lings-zerg".into(), Some(0.20)), ("ualbertabot9poolhydra-zerg".into(), Some(0.15))]);
        assert!(t.ladder.iter().filter(|e| e.group == "macro").all(|e| e.weight.is_none()));
        assert_eq!(t.league.start_mix, StartMixConfig { rush_hit_share: Some(0.5), groups: vec!["rush".into()] });
        assert_eq!((t.league.start_mix.share_for(Some("rush")), t.league.start_mix.share_for(Some("macro")), t.league.start_mix.share_for(None)), (Some(0.5), None, None));
        assert_eq!(t.league.rush_judge, RushJudgeConfig { groups: vec!["rush".into()], at_s: 480, min_scv: 8 });
        assert_eq!(t.sil.select_keys, Some(vec!["t0-zerg-rush".to_string(), "ualbertabot12poollings-zerg".into(), "ualbertabot9pool10lings-zerg".into()]));
        assert_eq!(t.sil.select_army_before_s, Some(225.0));
        assert_eq!(t.sil.select_start_hit, Some(true));
        assert_eq!(t.ladder.len(), 9);
        let mut d: RlConfig = toml::from_str(include_str!("../../../config/rl_rush4.toml")).unwrap();
        d.ladder.retain(|e| e.group != "warmup");
        for e in d.ladder.iter_mut() {
            e.weight = t.ladder.iter().find(|x| x.opponent().unwrap().1 == e.opponent().unwrap().1).unwrap().weight;
        }
        d.curriculum.ladder_groups = t.curriculum.ladder_groups.clone();
        d.curriculum.ladder_pick = t.curriculum.ladder_pick.clone();
        d.league.start_mix = t.league.start_mix.clone();
        d.league.rush_judge.groups = vec!["rush".into()];
        (d.sil.select_keys, d.sil.select_army_before_s, d.sil.select_start_hit) = (t.sil.select_keys.clone(), t.sil.select_army_before_s, t.sil.select_start_hit);
        assert_eq!(d, t);
        let back: RlConfig = serde_json::from_str(&serde_json::to_string(&t).unwrap()).unwrap();
        assert_eq!(back, t, "the serialised config (checkpoint) reads back the same");
    }

    #[test]
    fn rl_m2_race_config_044_r10() {
        use uranium_proto::Race;
        let text = include_str!("../../../config/rl_m2_race.toml").replace("\r\n", "\n");
        let s = "checkpoints/m2-dagger-r2/ckpt-1408";
        let filled = format!("\nzerg = \"{s}\"\n");
        assert!(text.contains(&filled), "S = r2 (041-R29 ④, 2026-11-14)");
        let empty: RlConfig = toml::from_str(&text.replacen(&filled, "\nzerg = \"\"\n", 1)).unwrap();
        let err = format!("{:#}", empty.validate().unwrap_err());
        assert!(err.contains("[kl.reference] zerg is empty"), "an empty S refuses to start: {err}");
        let t: RlConfig = toml::from_str(&text).unwrap();
        t.validate().unwrap_or_else(|e| panic!("{e:#}"));
        let base = "checkpoints/m2-s2e2/ckpt-11387";
        assert_eq!(t.kl.reference, KlReference { zerg: Some(s.into()), terran: Some(base.into()), protoss: Some(base.into()), random: Some(base.into()) });
        let cc = &t.curriculum;
        assert_eq!((cc.start_stage, cc.auto_advance), (Stage::C1, false));
        assert_eq!((cc.c1_ladder, cc.c1_t0, cc.c1_t1, cc.c1_self, cc.c1_snapshot, cc.c1_anchor), (1.0, 0.0, 0.0, 0.0, 0.0, 0.0));
        assert_eq!(cc.ladder_pick, LadderPickSetting::All(LadderPick::Fixed));
        let groups: Vec<(&str, f64)> = cc.ladder_groups.iter().map(|(g, x)| (g.as_str(), *x)).collect();
        assert_eq!(groups, [("protoss", 0.28), ("random", 0.20), ("terran", 0.24), ("zerg", 0.28)]);
        assert_eq!(cc.r_value_stages, vec![Stage::C0, Stage::C1]);
        let t1: Vec<(&str, &[Race])> = t.t1.iter().map(|b| (b.name.as_str(), b.races.as_slice())).collect();
        assert_eq!(t1, [("ualbertabot", &[Race::Terran, Race::Protoss, Race::Zerg][..]), ("steamhammer", &[Race::Terran, Race::Zerg][..]), ("stardust", &[Race::Protoss][..]), ("zzzkbot", &[Race::Zerg][..]), ("iron", &[Race::Terran][..])]);
        let race_group = |r: Race| match r {
            Race::Terran => "terran",
            Race::Protoss => "protoss",
            Race::Zerg => "zerg",
            _ => "random",
        };
        assert!(t.ladder.iter().all(|e| e.weight.is_some() && e.group == race_group(e.race)), "every entry weighted, in its race's group");
        assert!(t.ladder.iter().all(|e| e.build.is_none_or(|b| matches!(b, uranium_agent::T0Build::Standard | uranium_agent::T0Build::ZergRush))));
        assert!(t.ladder.iter().all(|e| !e.name.as_deref().unwrap_or_default().starts_with("mcrave")));
        assert_eq!(t.ladder.len(), 17);
        let mut d: RlConfig = toml::from_str(include_str!("../../../config/rl_m2.toml")).unwrap();
        d.kl.reference = t.kl.reference.clone();
        {
            let dc = &mut d.curriculum;
            (dc.start_stage, dc.auto_advance, dc.r_value_stages) = (cc.start_stage, cc.auto_advance, cc.r_value_stages.clone());
            (dc.c1_ladder, dc.c1_t0, dc.c1_t1, dc.c1_self) = (cc.c1_ladder, cc.c1_t0, cc.c1_t1, cc.c1_self);
            (dc.ladder_pick, dc.ladder_groups) = (cc.ladder_pick.clone(), cc.ladder_groups.clone());
        }
        (d.t1, d.ladder) = (t.t1.clone(), t.ladder.clone());
        assert_eq!(d, t);
        let back: RlConfig = serde_json::from_str(&serde_json::to_string(&t).unwrap()).unwrap();
        assert_eq!(back, t, "the serialised config (checkpoint) reads back the same");
    }
}
