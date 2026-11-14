//! League (042): who the learning agents play next, the win-rate table, PFSP, the curriculum
//! C0 → C1 → C2 (042-R5), snapshots and resets, learner time share (042-R7) and the reward
//! weight schedule (033). Pure logic with its own RNG so it is testable and resumable (042-R1).
//! Results are judged with 020-R13 before they reach `record`.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use serde::{Deserialize, Serialize};
use uranium_agent::T0Build;
use uranium_env::reward::{Outcome, RewardWeights};
use uranium_env::scout::PseudoMode;
use uranium_proto::Race;

use crate::config::RlConfig;

/// Learning agents (042 league table).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum AgentKind {
    Main,
    MainExploiter,
    LeagueExploiter,
}

impl AgentKind {
    pub const ALL: [AgentKind; 3] = [AgentKind::Main, AgentKind::MainExploiter, AgentKind::LeagueExploiter];

    /// Name in snapshot ids and checkpoint folders (`agents/<name>/`).
    pub fn name(self) -> &'static str {
        match self {
            Self::Main => "main",
            Self::MainExploiter => "main-exploiter",
            Self::LeagueExploiter => "league-exploiter",
        }
    }

    pub fn index(self) -> usize {
        self as usize
    }
}

/// Curriculum stage (042-R5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Stage {
    C0,
    C1,
    C2,
}

/// A T1 bot and the races it plays (021).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct T1Bot {
    pub name: String,
    pub races: Vec<Race>,
}

/// Kind of a rush ladder entry (042 C1 러시 사다리, AMB-042-15).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LadderKind {
    T0,
    T1,
}

/// One `[[ladder]]` entry (042 C1 러시 사다리, AMB-042-15): a T0 build or a T1 bot folder (021
/// config variants included) with the one race it plays (no race sampling). 021-R7 / 044-R10
/// (2026-11-14): `race = "Random"` makes every game of the entry a lobby Random game (the
/// `[league] bot_random_race` draw is never applied to ladder entries); only the standard T0
/// build and a `[[t1]]` bot listing all three races may be Random (the bot is checked in
/// `RlConfig::validate`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LadderEntry {
    pub kind: LadderKind,
    /// `kind = "t0"`: the T0 build (`T0Build` name, e.g. `"zerg_rush"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub build: Option<T0Build>,
    /// `kind = "t1"`: the T1 bot folder name (e.g. `"steamhammeroverpool"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub race: Race,
    /// 042 사다리 그룹 (AMB-042-16): the entry's group (`rush`, `macro`, ...); absent = the empty
    /// name. Drawn by `[curriculum] ladder_groups` when that is set, recorded in `games.csv`.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub group: String,
    /// 042-R19 (AMB-042-19): the entry's weight in a `"fixed"` group (>= 0, absent = 1.0); only
    /// for entries of a `"fixed"` group (load error otherwise). Left out of the serialised config
    /// when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight: Option<f64>,
}

impl LadderEntry {
    /// 042-R19: the `"fixed"` weight (absent = 1.0).
    pub fn fixed_weight(&self) -> f64 {
        self.weight.unwrap_or(1.0)
    }

    /// The opponent, its win-table key and class by the existing rules (042, 021-R3): T0
    /// `t0-<race>[-<build tag>]` / T0, T1 `<name>-<race>` / T1. An error for the 042 사다리 설정
    /// 검사 cases (missing or extra field, Unknown race, a build of another race — so Random on
    /// a one-race build). A Random entry's key is `t0-random` / `<name>-random` (021-R7, 044-R10).
    pub fn opponent(&self) -> Result<(Opponent, String, Class), String> {
        let race = self.race;
        if !RACES.contains(&race) && race != Race::Random {
            return Err(format!("[[ladder]] race must be Terran, Protoss, Zerg or Random, got {race:?} (042, 021-R7)"));
        }
        match (self.kind, self.build, self.name.as_deref()) {
            (LadderKind::T0, Some(build), None) => {
                if build.race().is_some_and(|r| r != race) {
                    return Err(format!("[[ladder]] T0 build {} plays {:?}, not {race:?} (042)", build.name(), build.race().unwrap()));
                }
                Ok((Opponent::T0 { race, build }, t0_key(race, build), Class::T0))
            }
            (LadderKind::T1, None, Some(name)) if !name.trim().is_empty() => Ok((Opponent::T1 { name: name.to_string(), race }, format!("{name}-{}", race_name(race)), Class::T1)),
            (LadderKind::T0, _, _) => Err("[[ladder]] kind = \"t0\" needs `build` and no `name` (042)".into()),
            (LadderKind::T1, _, _) => Err("[[ladder]] kind = \"t1\" needs a non-empty `name` and no `build` (042)".into()),
        }
    }
}

/// How the C1 rush ladder picks among its entries (042 사다리 안 상대 선택, AMB-042-15).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LadderPick {
    /// PFSP f_var(x) = x(1 - x) on Main's win-table value; all weights 0 = uniform.
    #[default]
    FVar,
    /// 042 사다리 선택 `band` (AMB-042-17): `exp(-((x - center) / width)^2)` with the
    /// `exclude_below` rule (`[curriculum.band]`).
    Band,
    /// 042-R19 사다리 선택 `fixed` (AMB-042-19): in proportion to `[[ladder]] weight` (absent
    /// 1.0); the win table is not read.
    Fixed,
}

/// `[curriculum] ladder_pick` (042, AMB-042-17): one method for every group (a string, the
/// earlier form) or a table group name → method; groups not in the table use `"f_var"`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum LadderPickSetting {
    All(LadderPick),
    PerGroup(BTreeMap<String, LadderPick>),
}

impl Default for LadderPickSetting {
    fn default() -> Self {
        Self::All(LadderPick::FVar)
    }
}

impl LadderPickSetting {
    /// The method of a ladder game drawn from `group` (`None` = no `ladder_groups`).
    pub fn for_group(&self, group: Option<&str>) -> LadderPick {
        match self {
            Self::All(p) => *p,
            Self::PerGroup(m) => group.and_then(|g| m.get(g)).copied().unwrap_or(LadderPick::FVar),
        }
    }
}

/// `[curriculum.band]`: the numbers of `ladder_pick = "band"` (042 사다리 선택 `band`,
/// AMB-042-17). Absent = 0.5 / 0.15 / 0.10.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct BandConfig {
    pub center: f64,
    pub width: f64,
    pub exclude_below: f64,
}

impl Default for BandConfig {
    fn default() -> Self {
        Self { center: 0.5, width: 0.15, exclude_below: 0.10 }
    }
}

impl BandConfig {
    /// 042 `band` weight `exp(-((x - center) / width)^2)`.
    pub fn weight(&self, x: f64) -> f64 {
        let z = (x - self.center) / self.width;
        (-z * z).exp()
    }

    /// The `band` weights of the win-table values `xs` of one group's entries (042,
    /// AMB-042-17): entries below `exclude_below` get 0 when any entry of the group is at or above
    /// it (none is excluded when every entry is below); when every remaining weight is 0
    /// (underflow) the remaining entries get 1 each (uniform).
    pub fn weights(&self, xs: &[f64]) -> Vec<f64> {
        let any_above = xs.iter().any(|x| *x >= self.exclude_below);
        let kept = |x: f64| !(any_above && x < self.exclude_below);
        let w: Vec<f64> = xs.iter().map(|&x| if kept(x) { self.weight(x) } else { 0.0 }).collect();
        if w.iter().sum::<f64>() > 0.0 {
            return w;
        }
        xs.iter().map(|&x| if kept(x) { 1.0 } else { 0.0 }).collect()
    }

    /// 042 설정 검사: `0 <= center <= 1`, `width > 0`, `0 <= exclude_below < 1`.
    pub fn validate(&self) -> Result<(), String> {
        if !(0.0..=1.0).contains(&self.center) {
            return Err(format!("[curriculum.band] center must be in 0..=1, got {} (042, AMB-042-17)", self.center));
        }
        if !(self.width > 0.0 && self.width.is_finite()) {
            return Err(format!("[curriculum.band] width must be above 0, got {} (042, AMB-042-17)", self.width));
        }
        if !(0.0..1.0).contains(&self.exclude_below) {
            return Err(format!("[curriculum.band] exclude_below must be in 0..1, got {} (042, AMB-042-17)", self.exclude_below));
        }
        Ok(())
    }
}

/// One opponent choice.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Opponent {
    /// T0 scripted bot; `race` is Random for the 021-R7 Random share.
    T0 { race: Race, build: T0Build },
    /// T1 bot by name (021); `race` may be Random for multi-race bots (021-R7).
    T1 { name: String, race: Race },
    /// The learning agent's current Main weights (TvT self-play).
    SelfPlay,
    /// A frozen league snapshot (042 snapshot id `uranium-<agent>@<run>/<update>`).
    Snapshot { id: String },
    /// The fixed anchor opponent `[league] anchor` (042, AMB-042-14); `name` = its checkpoint
    /// name, win-table key `anchor@<name>`.
    Anchor { name: String },
}

/// Opponent class for the curriculum windows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Class {
    T0,
    T1,
    Model,
}

/// The next game for a learning agent.
#[derive(Debug, Clone, PartialEq)]
pub struct Pick {
    pub agent: AgentKind,
    pub opponent: Opponent,
    /// Win-table key (021-R3 opponent id, without the bot's version for T1).
    pub key: String,
    pub class: Class,
    /// Fighting Spirit (true) or Circuit Breaker.
    pub fighting_spirit: bool,
    /// 042 사다리 그룹 (AMB-042-16): the `[[ladder]]` entry's `group` for a ladder game, `None`
    /// for any other game (`games.csv` `ladder_group`).
    pub ladder_group: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct Window {
    games: VecDeque<f64>,
}

impl Window {
    fn push(&mut self, x: f64, cap: usize) {
        self.games.push_back(x);
        while self.games.len() > cap {
            self.games.pop_front();
        }
    }
    fn mean(&self) -> Option<f64> {
        (!self.games.is_empty()).then(|| self.games.iter().sum::<f64>() / self.games.len() as f64)
    }
    fn full_mean(&self, cap: usize) -> Option<f64> {
        (self.games.len() >= cap).then(|| self.mean().unwrap())
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct AgentState {
    /// Win-rate EMA per opponent key.
    win: BTreeMap<String, f64>,
    /// Updates of this agent (its policy version).
    updates: u64,
    /// Updates at the last snapshot / reset.
    since_snapshot: u64,
    /// Exploiter snapshot window (042: last 200 games).
    recent: Window,
}

/// League state; serialised into the checkpoint (`ckpt::save_league`, 042-R1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct League {
    pub run: String,
    pub stage: Stage,
    /// Main updates when C2 started (pseudo-reward decay, 033).
    pub c2_start: Option<u64>,
    agents: Vec<AgentState>,
    /// Main's results per (class, race) for the curriculum (042-R5) and race sampling.
    windows: BTreeMap<String, Window>,
    /// Snapshot ids by the agent that produced them.
    pub snapshots: Vec<(AgentKind, String)>,
    // The T1 list is not league state any more: it is `[[t1]]` of the current config
    // (`RlConfig::t1`, 042 "재개 시 설정", AMB-042-13). The `t1` field of older league.bin
    // files is ignored on load.
    /// 042-AC1 small setting: Main only, always the standard T0 bot of this race on Fighting
    /// Spirit (no self-play, no curriculum change).
    #[serde(default)]
    pub fixed_t0: Option<Race>,
    rng: u64,
    #[serde(skip)]
    cfg: Option<RlConfig>,
    /// Snapshot ids whose model files are on the share (042 스냅샷 공개, AMB-042-14): only these
    /// are C1 snapshot candidates. Not league state: the run loop marks them after a publish and,
    /// on resume, from the share.
    #[serde(skip)]
    published: BTreeSet<String>,
}

fn win_value(o: Outcome) -> f64 {
    match o {
        Outcome::Win => 1.0,
        Outcome::Loss => 0.0,
        Outcome::Draw => 0.5,
    }
}

fn race_name(r: Race) -> &'static str {
    match r {
        Race::Terran => "terran",
        Race::Protoss => "protoss",
        Race::Zerg => "zerg",
        _ => "random",
    }
}

const RACES: [Race; 3] = [Race::Terran, Race::Protoss, Race::Zerg];

/// 021-R3 win-table key of a T0 build: `t0-<race>-<tag>`, `t0-<race>` for the standard build.
fn t0_key(race: Race, build: T0Build) -> String {
    match build.id_tag() {
        Some(tag) => format!("t0-{}-{tag}", race_name(race)),
        None => format!("t0-{}", race_name(race)),
    }
}

fn t0_builds(race: Race) -> Vec<T0Build> {
    let mut v = vec![T0Build::Standard];
    for b in [T0Build::ZergRush, T0Build::TerranDrop, T0Build::ProtossDt, T0Build::ZergMuta] {
        if b.race() == Some(race) {
            v.push(b);
        }
    }
    v
}

impl League {
    /// A new league (no LATEST in the run). It starts in `[curriculum] start_stage` (042-R5,
    /// AMB-042-13); the 042-AC1 small setting (`fixed_t0`) always starts in C0 as before.
    pub fn new(run: &str, seed: u64, fixed_t0: Option<Race>, cfg: RlConfig) -> Self {
        Self {
            run: run.to_string(),
            stage: if fixed_t0.is_some() { Stage::C0 } else { cfg.curriculum.start_stage },
            c2_start: None,
            agents: vec![AgentState::default(); 3],
            windows: BTreeMap::new(),
            snapshots: Vec::new(),
            fixed_t0,
            rng: seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1,
            cfg: Some(cfg),
            published: BTreeSet::new(),
        }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        serde_json::to_vec_pretty(self).expect("league serialises")
    }

    /// A resumed league (042-R1). Stage, win tables, curriculum windows and snapshots come from
    /// `b`; opponent ratios, `auto_advance`, `r_value_stages`, `r_value_vs_model`, the anchor,
    /// `main_candidates` and the T1 list follow `cfg`, the current config file (042 "재개 시
    /// 설정", AMB-042-13 / 14). No snapshot counts as published until the run loop marks it.
    pub fn from_bytes(b: &[u8], cfg: RlConfig) -> anyhow::Result<Self> {
        let mut l: League = serde_json::from_slice(b)?;
        l.cfg = Some(cfg);
        Ok(l)
    }

    fn cfg(&self) -> &RlConfig {
        self.cfg.as_ref().expect("league config")
    }

    fn unit(&mut self) -> f64 {
        // SplitMix64.
        self.rng = self.rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Index drawn with probability proportional to `w` (uniform if all are 0).
    fn weighted(&mut self, w: &[f64]) -> usize {
        let total: f64 = w.iter().sum();
        let u = self.unit();
        if total <= 0.0 {
            return ((u * w.len() as f64) as usize).min(w.len() - 1);
        }
        let mut acc = 0.0;
        for (i, x) in w.iter().enumerate() {
            acc += x / total;
            if u < acc {
                return i;
            }
        }
        // Rounding left `acc` just below 1: the last index with a weight, never a weight-0 entry
        // (042-R19 `fixed` weight 0 = never picked).
        w.iter().rposition(|x| *x > 0.0).unwrap_or(w.len() - 1)
    }

    pub fn updates(&self, a: AgentKind) -> u64 {
        self.agents[a.index()].updates
    }

    pub fn win_rate(&self, a: AgentKind, key: &str) -> f64 {
        self.agents[a.index()].win.get(key).copied().unwrap_or(self.cfg().league.win_rate_prior)
    }

    fn window_key(class: Class, race: Race) -> String {
        format!("{class:?}/{}", race_name(race))
    }

    /// Main's recent win rate against a race (all classes), for race sampling.
    fn race_win(&self, r: Race) -> f64 {
        let prior = self.cfg().league.win_rate_prior;
        let (mut s, mut n) = (0.0, 0usize);
        for c in [Class::T0, Class::T1] {
            if let Some(w) = self.windows.get(&Self::window_key(c, r)) {
                s += w.games.iter().sum::<f64>();
                n += w.games.len();
            }
        }
        if n == 0 { prior } else { s / n as f64 }
    }

    /// 042 race sampling: `mix/3 + (1-mix)(1-w_r)/Σ(1-w)`, each race at least 1/6.
    pub fn race_probs(&self) -> [f64; 3] {
        let mix = self.cfg().league.race_uniform_mix;
        let loss: Vec<f64> = RACES.iter().map(|&r| 1.0 - self.race_win(r)).collect();
        let total: f64 = loss.iter().sum();
        let mut p = [0.0; 3];
        for i in 0..3 {
            let weak = if total > 0.0 { loss[i] / total } else { 1.0 / 3.0 };
            p[i] = (mix / 3.0 + (1.0 - mix) * weak).max(1.0 / 6.0);
        }
        let s: f64 = p.iter().sum();
        p.map(|x| x / s)
    }

    fn pick_race(&mut self) -> Race {
        let p = self.race_probs();
        RACES[self.weighted(&p)]
    }

    fn t0_pick(&mut self) -> (Opponent, String, Class) {
        let race = self.pick_race();
        let builds = t0_builds(race);
        let build = builds[(self.unit() * builds.len() as f64) as usize % builds.len()];
        // 021-R7: Random only for the standard (multi-race) T0 build.
        let random = build == T0Build::Standard && self.unit() < self.cfg().league.bot_random_race;
        let key = t0_key(race, build);
        (Opponent::T0 { race: if random { Race::Random } else { race }, build }, key, Class::T0)
    }

    fn t1_pick(&mut self) -> Option<(Opponent, String, Class)> {
        if self.cfg().t1.is_empty() {
            return None;
        }
        let race = self.pick_race();
        let all = &self.cfg().t1;
        let bots: Vec<T1Bot> = all.iter().filter(|b| b.races.contains(&race)).cloned().collect();
        let bots = if bots.is_empty() { all.clone() } else { bots };
        let bot = bots[(self.unit() * bots.len() as f64) as usize % bots.len()].clone();
        let race = if bot.races.contains(&race) { race } else { bot.races[0] };
        // 021-R7 (AMB-042-14): Random only for a bot listing all three races; a two-race bot
        // plays the sampled race of its list (Random could give one it does not list). The
        // draw is made only for those bots, so three-race bots keep the earlier draw order.
        let random = RACES.iter().all(|r| bot.races.contains(r)) && self.unit() < self.cfg().league.bot_random_race;
        let key = format!("{}-{}", bot.name, race_name(race));
        Some((Opponent::T1 { name: bot.name, race: if random { Race::Random } else { race } }, key, Class::T1))
    }

    /// Every PFSP candidate (snapshots + T0 + T1) with its key.
    fn pfsp_pool(&self, only_main_snapshots: bool) -> Vec<(Opponent, String, Class)> {
        let mut v: Vec<(Opponent, String, Class)> = self
            .snapshots
            .iter()
            .filter(|(a, _)| !only_main_snapshots || *a == AgentKind::Main)
            .map(|(_, id)| (Opponent::Snapshot { id: id.clone() }, id.clone(), Class::Model))
            .collect();
        if !only_main_snapshots {
            for r in RACES {
                for b in t0_builds(r) {
                    v.push((Opponent::T0 { race: r, build: b }, t0_key(r, b), Class::T0));
                }
            }
            for bot in &self.cfg().t1 {
                for &r in &bot.races {
                    v.push((Opponent::T1 { name: bot.name.clone(), race: r }, format!("{}-{}", bot.name, race_name(r)), Class::T1));
                }
            }
        }
        v
    }

    /// PFSP (AlphaStar): weight f(x) with x = the agent's win rate against the candidate.
    fn pfsp(&mut self, agent: AgentKind, pool: Vec<(Opponent, String, Class)>, var: bool) -> Option<(Opponent, String, Class)> {
        if pool.is_empty() {
            return None;
        }
        let p = self.cfg().league.pfsp_power;
        let w: Vec<f64> = pool
            .iter()
            .map(|(_, k, _)| {
                let x = self.win_rate(agent, k);
                if var { x * (1.0 - x) } else { (1.0 - x).powf(p) }
            })
            .collect();
        let i = self.weighted(&w);
        Some(pool[i].clone())
    }

    /// 042 사다리 선택 `band` (AMB-042-17): one entry of `pool` (one group's entries, or every
    /// entry without groups) by `BandConfig::weights` over the agent's win-table values (unseen =
    /// the prior, 0.5).
    fn band_pick(&mut self, agent: AgentKind, pool: Vec<(Opponent, String, Class)>) -> Option<(Opponent, String, Class)> {
        if pool.is_empty() {
            return None;
        }
        let xs: Vec<f64> = pool.iter().map(|(_, k, _)| self.win_rate(agent, k)).collect();
        let w = self.cfg().curriculum.band.weights(&xs);
        let i = self.weighted(&w);
        Some(pool[i].clone())
    }

    /// 042 C1 러시 사다리 (AMB-042-15): an entry of `[[ladder]]` by `[curriculum] ladder_pick`
    /// (f_var over Main's win-table values, unseen = the prior; all weights 0 = uniform), with the
    /// entry's group. 042 사다리 그룹 (AMB-042-16): with `ladder_groups` set, a group is drawn by
    /// the shares first and the pick is among that group's entries; without it no extra draw is
    /// made (the earlier picks). `None` for an empty list (a load error when `c1_ladder > 0`).
    fn ladder_pick(&mut self, agent: AgentKind) -> Option<((Opponent, String, Class), String)> {
        let groups: Vec<(String, f64)> = self.cfg().curriculum.ladder_groups.iter().map(|(g, x)| (g.clone(), *x)).collect();
        let group = if groups.is_empty() {
            None
        } else {
            let shares: Vec<f64> = groups.iter().map(|(_, x)| *x).collect();
            Some(groups[self.weighted(&shares)].0.clone())
        };
        let entries: Vec<(Opponent, String, Class, String, f64)> = self
            .cfg()
            .ladder
            .iter()
            .filter(|e| group.as_ref().is_none_or(|g| &e.group == g))
            .filter_map(|e| e.opponent().ok().map(|(o, k, c)| (o, k, c, e.group.clone(), e.fixed_weight())))
            .collect();
        let pool: Vec<(Opponent, String, Class)> = entries.iter().map(|(o, k, c, _, _)| (o.clone(), k.clone(), *c)).collect();
        // 042 (AMB-042-17): the group's method; "f_var" is the earlier pick, draw for draw.
        let picked = match self.cfg().curriculum.ladder_pick.for_group(group.as_deref()) {
            LadderPick::FVar => self.pfsp(agent, pool, true),
            LadderPick::Band => self.band_pick(agent, pool),
            // 042-R19 (AMB-042-19): by `[[ladder]] weight`, the win table not read.
            LadderPick::Fixed if pool.is_empty() => None,
            LadderPick::Fixed => {
                let w: Vec<f64> = entries.iter().map(|e| e.4).collect();
                Some(pool[self.weighted(&w)].clone())
            }
        }?;
        let g = entries.iter().find(|(_, k, _, _, _)| *k == picked.1).map(|(_, _, _, g, _)| g.clone()).unwrap_or_default();
        Some((picked, g))
    }

    /// Win-table key of the current Main as an opponent (self-play).
    pub fn self_play_key(&self) -> String {
        format!("uranium-main@{}", self.run)
    }

    /// 042 anchor (AMB-042-14): its win-table key `anchor@<checkpoint name>`, `None` without one.
    pub fn anchor_key(&self) -> Option<String> {
        self.cfg().league.anchor_name().map(|n| format!("anchor@{n}"))
    }

    /// The snapshot's model files are on the share (042 스냅샷 공개, AMB-042-14).
    pub fn mark_published(&mut self, id: &str) {
        self.published.insert(id.to_string());
    }

    /// 042 C1 snapshot candidates (AMB-042-14): the most recent `[snapshot] main_candidates`
    /// published Main snapshots (0 = all), oldest first.
    pub fn c1_candidates(&self) -> Vec<String> {
        let all: Vec<&String> = self.snapshots.iter().filter(|(a, id)| *a == AgentKind::Main && self.published.contains(id)).map(|(_, id)| id).collect();
        let n = self.cfg().snapshot.main_candidates;
        let skip = if n == 0 { 0 } else { all.len().saturating_sub(n) };
        all[skip..].iter().map(|id| (*id).clone()).collect()
    }

    fn anchor_pick(&self) -> Option<(Opponent, String, Class)> {
        let name = self.cfg().league.anchor_name()?;
        Some((Opponent::Anchor { name: name.clone() }, format!("anchor@{name}"), Class::Model))
    }

    /// The model opponents (snapshot ids, anchor key) the next picks can choose in the current
    /// stage; the run loop shuts the inference servers of the others down (042 스냅샷 추론 서버,
    /// AMB-042-14).
    pub fn model_opponents(&self) -> BTreeSet<String> {
        let mut out = BTreeSet::new();
        if self.fixed_t0.is_some() {
            return out;
        }
        let cc = &self.cfg().curriculum;
        match self.stage {
            Stage::C0 => {}
            Stage::C1 => {
                let cands = if cc.c1_snapshot > 0.0 { self.c1_candidates() } else { vec![] };
                let to_anchor = cc.c1_anchor > 0.0 || (cc.c1_snapshot > 0.0 && cands.is_empty());
                out.extend(cands);
                if to_anchor {
                    out.extend(self.anchor_key());
                }
            }
            // C2 PFSP: every snapshot, as before.
            Stage::C2 => out.extend(self.snapshots.iter().map(|(_, id)| id.clone())),
        }
        out
    }

    /// 042-R7: which agent plays the next game; C0/C1 train Main only.
    fn pick_agent(&mut self) -> AgentKind {
        if self.stage < Stage::C2 {
            return AgentKind::Main;
        }
        let l = &self.cfg().league;
        let share = [l.share_main, l.share_main_exploiter, l.share_league_exploiter];
        AgentKind::ALL[self.weighted(&share)]
    }

    /// The next game (042 league table, curriculum, PFSP).
    pub fn next(&mut self) -> Pick {
        if let Some(race) = self.fixed_t0 {
            let opponent = Opponent::T0 { race, build: T0Build::Standard };
            return Pick { agent: AgentKind::Main, opponent, key: format!("t0-{}", race_name(race)), class: Class::T0, fighting_spirit: true, ladder_group: None };
        }
        let agent = self.pick_agent();
        let fighting_spirit = self.unit() < 0.5;
        let main_key = self.self_play_key();
        let selfplay = (Opponent::SelfPlay, main_key.clone(), Class::Model);
        let cc = self.cfg().curriculum.clone();
        let lc = self.cfg().league.clone();
        let mut ladder_group = None;
        let (opponent, key, class) = match agent {
            AgentKind::Main => match self.stage {
                Stage::C0 => {
                    if self.unit() < cc.c0_t0 / (cc.c0_t0 + cc.c0_self) { self.t0_pick() } else { selfplay }
                }
                Stage::C1 => {
                    // 042 C1 (AMB-042-14 / 15): proportional over T0, T1, the rush ladder, current
                    // Main, past Main snapshots and the anchor. With the ladder, snapshot and
                    // anchor shares at 0 (their default) the draws and picks are the earlier ones
                    // (`x + 0.0 == x`).
                    let models = cc.c1_snapshot + cc.c1_anchor;
                    let bots = cc.c1_t0 + cc.c1_t1 + cc.c1_ladder;
                    let u = self.unit() * (bots + cc.c1_self + models);
                    if u < cc.c1_t0 {
                        self.t0_pick()
                    } else if u < cc.c1_t0 + cc.c1_t1 {
                        self.t1_pick().unwrap_or_else(|| self.t0_pick())
                    } else if u < bots {
                        match self.ladder_pick(agent) {
                            Some((p, g)) => {
                                ladder_group = Some(g);
                                p
                            }
                            None => selfplay,
                        }
                    } else if models <= 0.0 || u < bots + cc.c1_self {
                        selfplay
                    } else if cc.c1_anchor <= 0.0 || u < bots + cc.c1_self + cc.c1_snapshot {
                        // Snapshot share (PFSP over the candidates); no published snapshot yet:
                        // the anchor, else the current Main.
                        let pool: Vec<(Opponent, String, Class)> = self.c1_candidates().into_iter().map(|id| (Opponent::Snapshot { id: id.clone() }, id, Class::Model)).collect();
                        match self.pfsp(agent, pool, false) {
                            Some(p) => p,
                            None => self.anchor_pick().unwrap_or(selfplay),
                        }
                    } else {
                        self.anchor_pick().unwrap_or(selfplay)
                    }
                }
                Stage::C2 => {
                    let u = self.unit() * (lc.main_pfsp + lc.main_self + lc.main_t1);
                    if u < lc.main_pfsp {
                        let pool = self.pfsp_pool(false);
                        self.pfsp(agent, pool, false).unwrap_or(selfplay)
                    } else if u < lc.main_pfsp + lc.main_self {
                        selfplay
                    } else {
                        self.t1_pick().unwrap_or(selfplay)
                    }
                }
            },
            AgentKind::MainExploiter => {
                let low = self.win_rate(agent, &main_key) < lc.main_exploiter_fallback_win_rate;
                if low && self.unit() < lc.main_exploiter_fallback_share {
                    let pool = self.pfsp_pool(true);
                    self.pfsp(agent, pool, true).unwrap_or(selfplay)
                } else {
                    selfplay
                }
            }
            AgentKind::LeagueExploiter => {
                let pool = self.pfsp_pool(false);
                self.pfsp(agent, pool, false).unwrap_or(selfplay)
            }
        };
        Pick { agent, opponent, key, class, fighting_spirit, ladder_group }
    }

    /// A finished game (outcome judged with 020-R13). `enemy_race` is the race actually played.
    /// Returns true when Main's curriculum stage advanced (042-R5).
    pub fn record(&mut self, pick: &Pick, enemy_race: Race, outcome: Outcome) -> bool {
        let x = win_value(outcome);
        let decay = self.cfg().league.win_rate_decay;
        let prior = self.cfg().league.win_rate_prior;
        let window = self.cfg().snapshot.exploiter_win_window;
        let a = &mut self.agents[pick.agent.index()];
        let w = a.win.entry(pick.key.clone()).or_insert(prior);
        *w = decay * *w + (1.0 - decay) * x;
        a.recent.push(x, window);
        if pick.agent != AgentKind::Main || pick.class == Class::Model {
            return false;
        }
        let cap = self.cfg().curriculum.window_games;
        self.windows.entry(Self::window_key(pick.class, enemy_race)).or_default().push(x, cap);
        self.advance()
    }

    /// 042-R5: every race's last `window_games` against T0 (C0→C1) or T1 (C1→C2) at the bar.
    /// Never with `[curriculum] auto_advance = false` (AMB-042-13); `record` keeps filling the
    /// windows either way.
    fn advance(&mut self) -> bool {
        if self.fixed_t0.is_some() {
            return false;
        }
        let cc = self.cfg().curriculum.clone();
        if !cc.auto_advance {
            return false;
        }
        let all = |class: Class, bar: f64| RACES.iter().all(|&r| self.windows.get(&Self::window_key(class, r)).and_then(|w| w.full_mean(cc.window_games)).is_some_and(|m| m >= bar));
        let next = match self.stage {
            Stage::C0 if all(Class::T0, cc.c0_to_c1_win_rate_vs_t0) => Some(Stage::C1),
            Stage::C1 if all(Class::T1, cc.c1_to_c2_win_rate_vs_t1) => Some(Stage::C2),
            _ => None,
        };
        if let Some(s) = next {
            self.stage = s;
            if s == Stage::C2 {
                self.c2_start = Some(self.updates(AgentKind::Main));
            }
        }
        next.is_some()
    }

    /// 033 weights for an agent's advantage: r_value only in the `[curriculum] r_value_stages`
    /// stages (default C0, AMB-042-13), r_build / r_units decay to 0 over
    /// `pseudo_reward_decay_updates` Main updates after C2 starts. 033-R11 `"scout"`
    /// (AMB-042-16): the build / units channels carry r_scout / r_survive at `[reward.scout]` /
    /// `[reward.survive]` `weight` (the per-game `vs_model` multiplier is per lane,
    /// `Segment::pseudo_scale`); the C2 decay applies to `"z"` only (`"scout"` in C2 is asked
    /// again before C2, the weights stay).
    pub fn reward_weights(&self, base: &RewardWeights) -> [f32; 4] {
        let value = if self.cfg().curriculum.r_value_stages.contains(&self.stage) { base.value } else { 0.0 };
        if self.cfg().reward.pseudo == PseudoMode::Scout {
            let r = &self.cfg().reward;
            return [base.win, r.scout.weight as f32, r.survive.weight as f32, value];
        }
        let decay = match self.c2_start {
            Some(s) => (1.0 - (self.updates(AgentKind::Main) - s) as f64 / self.cfg().curriculum.pseudo_reward_decay_updates).max(0.0) as f32,
            None => 1.0,
        };
        [base.win, base.build * decay, base.units * decay, value]
    }

    /// 033 / 042 "r_value의 게임별 적용" (AMB-042-14): whether r_value keeps its policy weight
    /// in a game against an opponent of `class` (the stage gate is `reward_weights`): always
    /// against a bot, against a model only with `r_value_vs_model`.
    pub fn r_value_in_policy(&self, class: Class) -> bool {
        class != Class::Model || self.cfg().curriculum.r_value_vs_model
    }

    /// An agent finished an update. Returns `Some(reset)` when it should be snapshotted now
    /// (042 snapshot table); `reset` = go back to the BC policy afterwards.
    pub fn on_update(&mut self, agent: AgentKind) -> Option<bool> {
        let sc = self.cfg().snapshot.clone();
        let reset_draw = self.unit();
        let a = &mut self.agents[agent.index()];
        a.updates += 1;
        let since = a.updates - a.since_snapshot;
        let due = match agent {
            AgentKind::Main => since >= sc.main_every_updates,
            _ => since >= sc.exploiter_max_updates || a.recent.full_mean(sc.exploiter_win_window).is_some_and(|m| m >= sc.exploiter_win_rate),
        };
        if !due {
            return None;
        }
        a.since_snapshot = a.updates;
        let reset = match agent {
            AgentKind::Main => false,
            AgentKind::MainExploiter => true,
            AgentKind::LeagueExploiter => reset_draw < sc.league_exploiter_reset_prob,
        };
        if reset {
            a.recent = Window::default();
            a.win.clear();
        }
        Some(reset)
    }

    /// 042 snapshot id `uranium-<agent>@<run>/<update>`; adds it to the PFSP pool.
    pub fn add_snapshot(&mut self, agent: AgentKind) -> String {
        let id = format!("uranium-{}@{}/{}", agent.name(), self.run, self.updates(agent));
        self.snapshots.push((agent, id.clone()));
        id
    }

    /// 042-R7: the agent the learner should update next among `ready` ones, the one furthest
    /// below its share of updates (Main only before C2).
    pub fn learner_turn(&self, ready: &[AgentKind]) -> Option<AgentKind> {
        if self.stage < Stage::C2 {
            return ready.contains(&AgentKind::Main).then_some(AgentKind::Main);
        }
        let l = &self.cfg().league;
        let share = [l.share_main, l.share_main_exploiter, l.share_league_exploiter];
        let total: u64 = AgentKind::ALL.iter().map(|&a| self.updates(a)).sum::<u64>().max(1);
        ready.iter().copied().max_by(|&a, &b| {
            let deficit = |k: AgentKind| share[k.index()] - self.updates(k) as f64 / total as f64;
            deficit(a).total_cmp(&deficit(b))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t1() -> Vec<T1Bot> {
        vec![
            T1Bot { name: "ualbertabot".into(), races: vec![Race::Terran, Race::Protoss, Race::Zerg] },
            T1Bot { name: "zzzkbot".into(), races: vec![Race::Zerg] },
        ]
    }

    fn cfg() -> RlConfig {
        let mut c = RlConfig::default_file();
        c.t1 = t1();
        c
    }

    fn league() -> League {
        League::new("r1", 7, None, cfg())
    }

    /// 042 C0: T0 80% / self 20%, races and builds spread, Random only on the standard build.
    #[test]
    fn c0_mix_042_r5() {
        let mut l = league();
        let n = 4000;
        let (mut t0, mut selfp, mut random, mut scenario) = (0, 0, 0, 0);
        for _ in 0..n {
            let p = l.next();
            assert_eq!(p.agent, AgentKind::Main);
            match p.opponent {
                Opponent::T0 { race, build } => {
                    t0 += 1;
                    if race == Race::Random {
                        random += 1;
                        assert_eq!(build, T0Build::Standard, "021-R7");
                    }
                    if build != T0Build::Standard {
                        scenario += 1;
                    }
                }
                Opponent::SelfPlay => selfp += 1,
                other => panic!("C0 has no {other:?}"),
            }
        }
        let f = t0 as f64 / n as f64;
        assert!((0.77..0.83).contains(&f), "T0 share {f}");
        assert_eq!(t0 + selfp, n);
        assert!(random > 0 && scenario > 0);
    }

    /// 042-R5: C0 → C1 only when every race's last 300 T0 games are at 70%+.
    #[test]
    fn curriculum_advances_on_full_windows_042_r5() {
        let mut l = league();
        let pick = |race: Race| Pick { agent: AgentKind::Main, opponent: Opponent::T0 { race, build: T0Build::Standard }, key: format!("t0-{}", race_name(race)), class: Class::T0, fighting_spirit: true, ladder_group: None };
        for _ in 0..300 {
            assert!(!l.record(&pick(Race::Terran), Race::Terran, Outcome::Win));
            assert!(!l.record(&pick(Race::Protoss), Race::Protoss, Outcome::Win));
        }
        assert_eq!(l.stage, Stage::C0, "zerg window still empty");
        let mut advanced = false;
        for i in 0..300 {
            advanced |= l.record(&pick(Race::Zerg), Race::Zerg, if i % 4 == 0 { Outcome::Loss } else { Outcome::Win });
        }
        assert!(advanced);
        assert_eq!(l.stage, Stage::C1);
    }

    /// PFSP f_hard favours opponents the agent loses to.
    #[test]
    fn pfsp_prefers_hard_opponents_042() {
        let mut l = league();
        l.stage = Stage::C2;
        let mut lose = Pick { agent: AgentKind::LeagueExploiter, opponent: Opponent::T1 { name: "zzzkbot".into(), race: Race::Zerg }, key: "zzzkbot-zerg".into(), class: Class::T1, fighting_spirit: true, ladder_group: None };
        for _ in 0..400 {
            l.record(&lose, Race::Zerg, Outcome::Loss);
        }
        lose.key = "t0-terran".into();
        for _ in 0..400 {
            l.record(&lose, Race::Terran, Outcome::Win);
        }
        let pool = l.pfsp_pool(false);
        let (mut hard, mut easy) = (0, 0);
        for _ in 0..3000 {
            let (_, k, _) = l.pfsp(AgentKind::LeagueExploiter, pool.clone(), false).unwrap();
            if k == "zzzkbot-zerg" {
                hard += 1;
            }
            if k == "t0-terran" {
                easy += 1;
            }
        }
        assert!(hard > 10 * easy.max(1), "hard {hard} easy {easy}");
    }

    /// 042-R7: before C2 only Main trains; in C2 the learner follows the shares.
    #[test]
    fn learner_time_share_042_r7() {
        let mut l = league();
        assert_eq!(l.learner_turn(&[AgentKind::MainExploiter]), None);
        l.stage = Stage::C2;
        let mut counts = [0u64; 3];
        for _ in 0..1000 {
            let a = l.learner_turn(&AgentKind::ALL).unwrap();
            counts[a.index()] += 1;
            l.agents[a.index()].updates += 1;
        }
        assert!((counts[0] as f64 / 1000.0 - 0.5).abs() < 0.01, "{counts:?}");
    }

    /// 042 snapshots: Main every N updates without reset; Main Exploiter resets.
    #[test]
    fn snapshots_and_resets_042() {
        let mut l = league();
        let every = l.cfg().snapshot.main_every_updates;
        let mut snaps = 0;
        for _ in 0..every * 2 {
            if let Some(reset) = l.on_update(AgentKind::Main) {
                assert!(!reset);
                snaps += 1;
            }
        }
        assert_eq!(snaps, 2);
        let max = l.cfg().snapshot.exploiter_max_updates;
        let mut r = None;
        for _ in 0..max {
            r = r.or(l.on_update(AgentKind::MainExploiter));
        }
        assert_eq!(r, Some(true));
        let id = l.add_snapshot(AgentKind::Main);
        assert_eq!(id, format!("uranium-main@r1/{}", every * 2));
    }

    /// 033: r_value only in C0; pseudo-rewards decay to 0 after C2.
    #[test]
    fn reward_weight_schedule_033() {
        let base = RewardWeights { win: 1.0, build: 0.3, units: 0.3, value: 0.05 };
        let mut l = league();
        assert_eq!(l.reward_weights(&base), [1.0, 0.3, 0.3, 0.05]);
        l.stage = Stage::C1;
        assert_eq!(l.reward_weights(&base), [1.0, 0.3, 0.3, 0.0]);
        l.stage = Stage::C2;
        l.c2_start = Some(0);
        l.agents[0].updates = (l.cfg().curriculum.pseudo_reward_decay_updates / 2.0) as u64;
        let w = l.reward_weights(&base);
        assert!((w[1] - 0.15).abs() < 1e-6);
        l.agents[0].updates *= 3;
        assert_eq!(l.reward_weights(&base), [1.0, 0.0, 0.0, 0.0]);
    }

    /// 042-AC1: the small setting plays one T0 race on Fighting Spirit and never advances.
    #[test]
    fn fixed_t0_setting_042_ac1() {
        let mut l = league();
        l.fixed_t0 = Some(Race::Zerg);
        for _ in 0..50 {
            let p = l.next();
            assert_eq!((p.agent, p.fighting_spirit, p.key.as_str()), (AgentKind::Main, true, "t0-zerg"));
            assert_eq!(p.opponent, Opponent::T0 { race: Race::Zerg, build: T0Build::Standard });
        }
        for _ in 0..400 {
            let p = l.next();
            assert!(!l.record(&p, Race::Zerg, Outcome::Win));
        }
        assert_eq!(l.stage, Stage::C0);
    }

    /// 042-R1: the league survives the checkpoint round trip.
    #[test]
    fn league_round_trip_042_r1() {
        let mut l = league();
        for _ in 0..10 {
            l.next();
        }
        let b = l.to_bytes();
        let mut m = League::from_bytes(&b, cfg()).unwrap();
        assert_eq!(l.next(), m.next());
    }

    /// The 042 "T1·셀프 run" league (config/rl_t1self.toml values, McRave left out).
    fn t1self_cfg() -> RlConfig {
        let mut c = cfg();
        c.curriculum.start_stage = Stage::C1;
        c.curriculum.auto_advance = false;
        (c.curriculum.c1_t0, c.curriculum.c1_t1, c.curriculum.c1_self) = (0.0, 0.5, 0.5);
        c.curriculum.r_value_stages = vec![Stage::C0, Stage::C1];
        c
    }

    fn t1_pick(name: &str, race: Race) -> Pick {
        Pick { agent: AgentKind::Main, opponent: Opponent::T1 { name: name.into(), race }, key: format!("{name}-{}", race_name(race)), class: Class::T1, fighting_spirit: true, ladder_group: None }
    }

    /// 042-R5 `start_stage = "C1"` (AMB-042-13): a new league starts in C1, trains Main only and,
    /// with `c1_t0 = 0`, never picks a T0 bot: about half T1, half self-play.
    #[test]
    fn start_stage_c1_t1_self_mix_042_r5() {
        let mut l = League::new("r1", 7, None, t1self_cfg());
        assert_eq!(l.stage, Stage::C1);
        let n = 4000;
        let (mut bots, mut selfp) = (0, 0);
        for _ in 0..n {
            let p = l.next();
            assert_eq!(p.agent, AgentKind::Main, "042-R7: C1 trains Main only");
            match p.opponent {
                Opponent::T1 { name, .. } => {
                    assert!(t1().iter().any(|b| b.name == name));
                    bots += 1;
                }
                Opponent::SelfPlay => selfp += 1,
                other => panic!("c1_t0 = 0 picked {other:?}"),
            }
        }
        assert_eq!(bots + selfp, n);
        let f = bots as f64 / n as f64;
        assert!((0.47..0.53).contains(&f), "T1 share {f}");
        // Default config: C0 as before; the AC1 small setting stays C0 whatever start_stage says.
        assert_eq!(league().stage, Stage::C0);
        assert_eq!(League::new("r1", 7, Some(Race::Zerg), t1self_cfg()).stage, Stage::C0);
    }

    /// 042-R5 `auto_advance = false` (AMB-042-13): full winning windows never change the stage,
    /// but the windows are still recorded (they drive race sampling).
    #[test]
    fn auto_advance_off_keeps_stage_042_r5() {
        let mut l = League::new("r1", 7, None, t1self_cfg());
        let before = l.race_probs();
        for _ in 0..400 {
            for (name, race) in [("ualbertabot", Race::Terran), ("ualbertabot", Race::Protoss), ("zzzkbot", Race::Zerg)] {
                assert!(!l.record(&t1_pick(name, race), race, if race == Race::Zerg { Outcome::Loss } else { Outcome::Win }));
            }
        }
        assert_eq!(l.stage, Stage::C1);
        assert_eq!(l.windows.get("T1/zerg").map(|w| w.games.len()), Some(l.cfg().curriculum.window_games));
        assert_ne!(l.race_probs(), before, "windows recorded");
        // The same windows advance a league with auto_advance on.
        let mut c = t1self_cfg();
        c.curriculum.auto_advance = true;
        let mut on = League::from_bytes(&l.to_bytes(), c).unwrap();
        assert!(!on.record(&t1_pick("zzzkbot", Race::Zerg), Race::Zerg, Outcome::Loss), "zerg below 50%");
        for _ in 0..300 {
            on.record(&t1_pick("zzzkbot", Race::Zerg), Race::Zerg, Outcome::Win);
        }
        assert_eq!(on.stage, Stage::C2);
    }

    /// 033 r_value (AMB-042-13): non-zero in the `r_value_stages` stages only.
    #[test]
    fn r_value_follows_r_value_stages_033() {
        let base = RewardWeights { win: 1.0, build: 0.3, units: 0.3, value: 0.05 };
        let mut l = League::new("r1", 7, None, t1self_cfg());
        assert_eq!(l.reward_weights(&base), [1.0, 0.3, 0.3, 0.05], "C1 listed");
        l.stage = Stage::C0;
        assert_eq!(l.reward_weights(&base)[3], 0.05);
        l.stage = Stage::C2;
        l.c2_start = Some(0);
        assert_eq!(l.reward_weights(&base)[3], 0.0, "C2 not listed");
        let mut c = cfg();
        c.curriculum.start_stage = Stage::C1;
        let l = League::new("r1", 7, None, c);
        assert_eq!(l.reward_weights(&base)[3], 0.0, "default: C0 only");
        let mut c = cfg();
        c.curriculum.r_value_stages = vec![];
        assert_eq!(League::new("r1", 7, None, c).reward_weights(&base)[3], 0.0);
    }

    /// 042 "재개 시 설정" (AMB-042-13): a resumed league keeps its stage, windows and snapshots
    /// from league.bin and takes ratios, auto_advance, r_value_stages and T1 list from the
    /// current config.
    #[test]
    fn resume_follows_current_config_042_r1() {
        let mut l = league();
        l.stage = Stage::C1;
        let id = l.add_snapshot(AgentKind::Main);
        for _ in 0..20 {
            l.record(&t1_pick("zzzkbot", Race::Zerg), Race::Zerg, Outcome::Win);
        }
        let bytes = l.to_bytes();
        // New config: no T0, only T1 (one new bot), r_value in C1, start_stage ignored.
        let mut c = cfg();
        c.curriculum.start_stage = Stage::C0;
        (c.curriculum.c1_t0, c.curriculum.c1_t1, c.curriculum.c1_self) = (0.0, 1.0, 0.0);
        c.curriculum.r_value_stages = vec![Stage::C1];
        c.curriculum.auto_advance = false;
        c.t1 = vec![T1Bot { name: "iron".into(), races: vec![Race::Terran] }];
        let mut m = League::from_bytes(&bytes, c).unwrap();
        assert_eq!(m.stage, Stage::C1);
        assert_eq!(m.snapshots, vec![(AgentKind::Main, id)]);
        assert_eq!(m.windows.get("T1/zerg").map(|w| w.games.len()), Some(20));
        for _ in 0..200 {
            match m.next().opponent {
                Opponent::T1 { name, .. } => assert_eq!(name, "iron"),
                other => panic!("only T1 iron, got {other:?}"),
            }
        }
        let base = RewardWeights { win: 1.0, build: 0.3, units: 0.3, value: 0.05 };
        assert_eq!(m.reward_weights(&base)[3], 0.05);
        assert_eq!(m.cfg().t1.len(), 1);
    }

    /// The 042 "T1·셀프 run" 2차 league (config/rl_t1self2.toml shares, AMB-042-14).
    fn t1self2_cfg() -> RlConfig {
        let mut c = t1self_cfg();
        let cc = &mut c.curriculum;
        (cc.c1_t0, cc.c1_t1, cc.c1_self, cc.c1_snapshot, cc.c1_anchor) = (0.0, 0.5, 0.15, 0.25, 0.10);
        cc.r_value_vs_model = false;
        c.league.anchor = "checkpoints/ac1-vtrace-s1/ckpt-3133569".into();
        c.snapshot.main_candidates = 4;
        c
    }

    /// Shares of `n` picks by opponent: (T1, current Main, snapshot, anchor), and the snapshot
    /// ids picked.
    fn c1_counts(l: &mut League, n: usize) -> ([f64; 4], BTreeMap<String, usize>) {
        let (mut c, mut ids) = ([0usize; 4], BTreeMap::new());
        for _ in 0..n {
            let p = l.next();
            assert_eq!((p.agent, p.class == Class::Model), (AgentKind::Main, !matches!(p.opponent, Opponent::T1 { .. })));
            match p.opponent {
                Opponent::T1 { .. } => c[0] += 1,
                Opponent::SelfPlay => {
                    assert_eq!(p.key, "uranium-main@r1");
                    c[1] += 1;
                }
                Opponent::Snapshot { id } => {
                    assert_eq!(p.key, id);
                    *ids.entry(id).or_insert(0) += 1;
                    c[2] += 1;
                }
                Opponent::Anchor { name } => {
                    assert_eq!((name.as_str(), p.key.as_str()), ("ckpt-3133569", "anchor@ckpt-3133569"));
                    c[3] += 1;
                }
                other => panic!("c1_t0 = 0 picked {other:?}"),
            }
        }
        (c.map(|x| x as f64 / n as f64), ids)
    }

    fn near(got: [f64; 4], want: [f64; 4]) {
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 0.025, "got {got:?}, want {want:?}");
        }
    }

    /// 042 C1 model opponents (AMB-042-14): T1 / current Main / snapshots / anchor in proportion
    /// to `c1_*`; before any published snapshot the snapshot share goes to the anchor; the
    /// snapshot share draws only among the latest `main_candidates` published Main snapshots.
    #[test]
    fn c1_model_opponent_shares_042_amb_042_14() {
        let mut l = League::new("r1", 7, None, t1self2_cfg());
        let (c, _) = c1_counts(&mut l, 8000);
        near(c, [0.5, 0.15, 0.0, 0.35]);
        // Six Main snapshots and one exploiter snapshot; the newest Main one is not published.
        let mut main = vec![];
        for i in 0..6 {
            l.agents[0].updates = 250 * (i + 1);
            main.push(l.add_snapshot(AgentKind::Main));
        }
        let other = l.add_snapshot(AgentKind::MainExploiter);
        for id in main.iter().take(5).chain([&other]) {
            l.mark_published(id);
        }
        assert_eq!(l.c1_candidates(), main[1..5].to_vec(), "latest 4 published Main snapshots");
        let (c, ids) = c1_counts(&mut l, 8000);
        near(c, [0.5, 0.15, 0.25, 0.10]);
        assert_eq!(ids.keys().cloned().collect::<BTreeSet<_>>(), main[1..5].iter().cloned().collect());
        // 0 = every published Main snapshot.
        let mut cfg = t1self2_cfg();
        cfg.snapshot.main_candidates = 0;
        l.cfg = Some(cfg);
        assert_eq!(l.c1_candidates(), main[..5].to_vec());
    }

    /// 042 (AMB-042-14): without an anchor, the anchor share and (before any snapshot) the
    /// snapshot share go to the current Main.
    #[test]
    fn c1_without_anchor_falls_back_to_current_main_042() {
        let mut cfg = t1self2_cfg();
        cfg.curriculum.c1_anchor = 0.0;
        cfg.league.anchor.clear();
        let mut l = League::new("r1", 7, None, cfg);
        let (c, _) = c1_counts(&mut l, 6000);
        let total = 0.5 + 0.15 + 0.25;
        near(c, [0.5 / total, 0.4 / total, 0.0, 0.0]);
        // An anchor without its own share still takes the snapshot share before any snapshot.
        let mut cfg = t1self2_cfg();
        cfg.curriculum.c1_anchor = 0.0;
        let mut l = League::new("r1", 7, None, cfg);
        let (c, _) = c1_counts(&mut l, 6000);
        near(c, [0.5 / total, 0.15 / total, 0.0, 0.25 / total]);
    }

    /// 042 C1 snapshot share (AMB-042-14): PFSP f_hard within the candidates (unseen = 0.5).
    #[test]
    fn c1_snapshot_pfsp_prefers_hard_snapshots_042() {
        let mut cfg = t1self2_cfg();
        (cfg.curriculum.c1_t1, cfg.curriculum.c1_self, cfg.curriculum.c1_anchor) = (0.0, 0.0, 0.0);
        let mut l = League::new("r1", 7, None, cfg);
        let ids: Vec<String> = (1..=3)
            .map(|i| {
                l.agents[0].updates = i * 250;
                let id = l.add_snapshot(AgentKind::Main);
                l.mark_published(&id);
                id
            })
            .collect();
        let pick = |id: &str| Pick { agent: AgentKind::Main, opponent: Opponent::Snapshot { id: id.into() }, key: id.into(), class: Class::Model, fighting_spirit: true, ladder_group: None };
        for _ in 0..300 {
            l.record(&pick(&ids[0]), Race::Terran, Outcome::Win);
            l.record(&pick(&ids[1]), Race::Terran, Outcome::Loss);
        }
        assert!(l.windows.is_empty(), "model games stay out of the curriculum windows");
        let (_, n) = c1_counts(&mut l, 3000);
        let (easy, hard, unseen) = (n.get(&ids[0]).copied().unwrap_or(0), n[&ids[1]], n[&ids[2]]);
        assert!(hard > 2 * unseen && unseen > 10 * easy.max(1), "easy {easy} hard {hard} unseen {unseen}");
    }

    /// 042 (AMB-042-14): with `c1_snapshot = c1_anchor = 0` (the defaults) the picks are those of
    /// the earlier C1 (T0 / T1 / current Main), whatever snapshots and anchor exist.
    #[test]
    fn zero_model_shares_keep_earlier_c1_picks_042() {
        let mut base = cfg();
        base.curriculum.start_stage = Stage::C1;
        let mut with = base.clone();
        with.league.anchor = "checkpoints/x/ckpt-1".into();
        let mut a = League::new("r1", 9, None, base);
        let mut b = League::new("r1", 9, None, with);
        let id = b.add_snapshot(AgentKind::Main);
        b.mark_published(&id);
        for _ in 0..2000 {
            let (p, q) = (a.next(), b.next());
            assert!(!matches!(p.opponent, Opponent::Snapshot { .. } | Opponent::Anchor { .. }));
            assert_eq!(p, q);
        }
    }

    /// 042 스냅샷 추론 서버 (AMB-042-14): the model opponents the picks can choose per stage.
    #[test]
    fn model_opponents_per_stage_042() {
        let mut l = League::new("r1", 7, None, t1self2_cfg());
        assert_eq!(l.model_opponents(), BTreeSet::from(["anchor@ckpt-3133569".to_string()]));
        let ids: Vec<String> = (1..=6)
            .map(|i| {
                l.agents[0].updates = i * 250;
                let id = l.add_snapshot(AgentKind::Main);
                l.mark_published(&id);
                id
            })
            .collect();
        let mut want: BTreeSet<String> = ids[2..].iter().cloned().collect();
        want.insert("anchor@ckpt-3133569".into());
        assert_eq!(l.model_opponents(), want);
        // No anchor share and candidates there: the anchor is not a choice.
        let mut c = t1self2_cfg();
        c.curriculum.c1_anchor = 0.0;
        l.cfg = Some(c);
        assert_eq!(l.model_opponents(), ids[2..].iter().cloned().collect());
        l.stage = Stage::C2;
        assert_eq!(l.model_opponents(), ids.iter().cloned().collect(), "C2 PFSP: every snapshot");
        l.stage = Stage::C0;
        assert!(l.model_opponents().is_empty());
    }

    /// 033 / 042 (AMB-042-14): r_value policy weight per game by opponent class.
    #[test]
    fn r_value_per_game_by_opponent_033() {
        let l = League::new("r1", 7, None, t1self2_cfg());
        assert!(l.r_value_in_policy(Class::T1) && l.r_value_in_policy(Class::T0));
        assert!(!l.r_value_in_policy(Class::Model));
        let l = league();
        assert!(l.cfg().curriculum.r_value_vs_model, "default true");
        assert!(l.r_value_in_policy(Class::Model) && l.r_value_in_policy(Class::T1));
    }

    /// 042-R1 / 스냅샷 공개 (AMB-042-14): which snapshots are published is not league state:
    /// league.bin keeps its format and a resumed league has no candidate until the run loop marks
    /// the snapshots it finds on the share.
    #[test]
    fn published_is_not_league_state_042() {
        let mut l = League::new("r1", 7, None, t1self2_cfg());
        let id = l.add_snapshot(AgentKind::Main);
        l.mark_published(&id);
        assert_eq!(l.c1_candidates(), std::slice::from_ref(&id));
        let v: serde_json::Value = serde_json::from_slice(&l.to_bytes()).unwrap();
        assert!(v.get("published").is_none());
        let mut m = League::from_bytes(&l.to_bytes(), t1self2_cfg()).unwrap();
        assert_eq!(m.snapshots, vec![(AgentKind::Main, id.clone())]);
        assert!(m.c1_candidates().is_empty());
        m.mark_published(&id);
        assert_eq!(m.c1_candidates(), [id]);
    }

    /// 021-R7 (AMB-042-14): Random only for bots listing all three races; a two-race bot plays a
    /// race of its list.
    #[test]
    fn random_only_for_three_race_bots_021_r7() {
        let mut c = t1self_cfg();
        c.curriculum.c1_self = 0.0;
        c.t1 = vec![
            T1Bot { name: "ualbertabot".into(), races: vec![Race::Terran, Race::Protoss, Race::Zerg] },
            T1Bot { name: "steamhammer".into(), races: vec![Race::Terran, Race::Zerg] },
        ];
        let mut l = League::new("r1", 7, None, c);
        let (mut ua, mut ua_random, mut sh) = (0, 0, 0);
        for _ in 0..6000 {
            match l.next().opponent {
                Opponent::T1 { name, race } if name == "steamhammer" => {
                    assert!(matches!(race, Race::Terran | Race::Zerg), "steamhammer {race:?}");
                    sh += 1;
                }
                Opponent::T1 { race, .. } => {
                    ua += 1;
                    ua_random += usize::from(race == Race::Random);
                }
                other => panic!("{other:?}"),
            }
        }
        let f = ua_random as f64 / ua as f64;
        assert!(sh > 1000 && (0.17..0.23).contains(&f), "steamhammer {sh}, ualbertabot Random {f}");
    }

    /// 042-R1: a league.bin written before AMB-042-13 (with the T1 list inside) still loads; the
    /// list in it is ignored for the config's.
    #[test]
    fn old_league_bin_loads_042_r1() {
        let mut l = league();
        l.stage = Stage::C1;
        let mut v: serde_json::Value = serde_json::from_slice(&l.to_bytes()).unwrap();
        assert!(v.get("t1").is_none(), "not written any more");
        v["t1"] = serde_json::json!([{ "name": "mcrave", "races": ["Terran", "Protoss", "Zerg"] }]);
        let old = serde_json::to_vec(&v).unwrap();
        let mut m = League::from_bytes(&old, cfg()).unwrap();
        assert_eq!(m.stage, Stage::C1);
        for _ in 0..500 {
            if let Opponent::T1 { name, .. } = m.next().opponent {
                assert_ne!(name, "mcrave");
            }
        }
        // And the 042-AC1 league.bin shape (fixed_t0 absent from the oldest ones).
        v.as_object_mut().unwrap().remove("fixed_t0");
        assert!(League::from_bytes(&serde_json::to_vec(&v).unwrap(), cfg()).unwrap().fixed_t0.is_none());
    }

    fn t0_entry(build: T0Build, race: Race) -> LadderEntry {
        LadderEntry { kind: LadderKind::T0, build: Some(build), name: None, race, group: String::new(), weight: None }
    }

    fn t1_entry(name: &str, race: Race) -> LadderEntry {
        LadderEntry { kind: LadderKind::T1, build: None, name: Some(name.into()), race, group: String::new(), weight: None }
    }

    /// The 042 "러시 커리큘럼 run" league (config/rl_rush1.toml shares, AMB-042-15).
    fn rush_cfg() -> RlConfig {
        let mut c = t1self2_cfg();
        let cc = &mut c.curriculum;
        (cc.c1_t0, cc.c1_t1, cc.c1_ladder, cc.c1_self, cc.c1_snapshot, cc.c1_anchor) = (0.0, 0.0, 0.70, 0.05, 0.15, 0.10);
        c.ladder = vec![t0_entry(T0Build::ZergRush, Race::Zerg), t1_entry("ualbertabot", Race::Zerg), t1_entry("steamhammeroverpool", Race::Zerg), t1_entry("zzzkbot", Race::Zerg)];
        c
    }

    const RUSH_KEYS: [&str; 4] = ["t0-zerg-rush", "ualbertabot-zerg", "steamhammeroverpool-zerg", "zzzkbot-zerg"];

    /// 042 C1 러시 사다리 (AMB-042-15): ladder / current Main / anchor shares as `c1_*` (before any
    /// snapshot the snapshot share goes to the anchor); ladder picks use the existing keys and
    /// classes, the entry's fixed race (never Random) and the T0 build as listed.
    #[test]
    fn c1_ladder_shares_and_keys_042_amb_042_15() {
        let mut l = League::new("r1", 7, None, rush_cfg());
        assert_eq!(l.stage, Stage::C1);
        let n = 8000;
        let (mut ladder, mut selfp, mut anchor) = (0, 0, 0);
        let mut keys = BTreeSet::new();
        for _ in 0..n {
            let p = l.next();
            assert_eq!(p.agent, AgentKind::Main);
            match &p.opponent {
                Opponent::T0 { race, build } => {
                    assert_eq!((*race, *build, p.key.as_str(), p.class), (Race::Zerg, T0Build::ZergRush, "t0-zerg-rush", Class::T0));
                    ladder += 1;
                }
                Opponent::T1 { name, race } => {
                    assert_eq!((*race, p.class), (Race::Zerg, Class::T1), "fixed race, no Random (021-R7)");
                    assert_eq!(p.key, format!("{name}-zerg"));
                    ladder += 1;
                }
                Opponent::SelfPlay => selfp += 1,
                Opponent::Anchor { .. } => anchor += 1,
                other => panic!("{other:?}"),
            }
            if matches!(p.class, Class::T0 | Class::T1) {
                keys.insert(p.key);
            }
        }
        let f = |x: usize| x as f64 / n as f64;
        assert!((f(ladder) - 0.70).abs() < 0.025 && (f(selfp) - 0.05).abs() < 0.015 && (f(anchor) - 0.25).abs() < 0.025, "ladder {ladder} self {selfp} anchor {anchor}");
        assert_eq!(keys, RUSH_KEYS.iter().map(|k| k.to_string()).collect::<BTreeSet<_>>());
        // r_value (bot games) and the curriculum windows follow the class rules.
        assert!(l.r_value_in_policy(Class::T0) && l.r_value_in_policy(Class::T1) && !l.r_value_in_policy(Class::Model));
        let p = Pick { agent: AgentKind::Main, opponent: Opponent::T0 { race: Race::Zerg, build: T0Build::ZergRush }, key: "t0-zerg-rush".into(), class: Class::T0, fighting_spirit: true, ladder_group: None };
        l.record(&p, Race::Zerg, Outcome::Loss);
        assert_eq!(l.windows.get("T0/zerg").map(|w| w.games.len()), Some(1));
        assert!(l.win_rate(AgentKind::Main, "t0-zerg-rush") < 0.5);
    }

    /// 042 사다리 안 상대 선택 (AMB-042-15): PFSP f_var x(1 - x) — an entry near 50% (here unseen,
    /// 0.5) is picked far more than ones Main nearly always beats or loses to; all weights 0 =
    /// uniform.
    #[test]
    fn ladder_f_var_prefers_even_opponents_042_amb_042_15() {
        let mut c = rush_cfg();
        (c.curriculum.c1_self, c.curriculum.c1_snapshot, c.curriculum.c1_anchor) = (0.0, 0.0, 0.0);
        let mut l = League::new("r1", 7, None, c);
        let pick = |e: &LadderEntry| {
            let (opponent, key, class) = e.opponent().unwrap();
            Pick { agent: AgentKind::Main, opponent, key, class, fighting_spirit: true, ladder_group: None }
        };
        let entries = l.cfg().ladder.clone();
        for i in 0..600 {
            l.record(&pick(&entries[0]), Race::Zerg, Outcome::Win);
            l.record(&pick(&entries[1]), Race::Zerg, Outcome::Loss);
            l.record(&pick(&entries[2]), Race::Zerg, if i % 2 == 0 { Outcome::Win } else { Outcome::Loss });
        }
        let count = |l: &mut League, n: usize| {
            let mut m: BTreeMap<String, usize> = BTreeMap::new();
            for _ in 0..n {
                *m.entry(l.next().key).or_default() += 1;
            }
            m
        };
        let m = count(&mut l, 4000);
        let get = |k: &str| m.get(k).copied().unwrap_or(0);
        assert!(get("zzzkbot-zerg") > 10 * get("t0-zerg-rush").max(1), "{m:?}");
        assert!(get("steamhammeroverpool-zerg") > 10 * get("ualbertabot-zerg").max(1), "{m:?}");
        // Every entry at x(1 - x) = 0: uniform.
        for k in RUSH_KEYS {
            l.agents[0].win.insert(k.to_string(), if k.starts_with("t0") { 0.0 } else { 1.0 });
        }
        let m = count(&mut l, 8000);
        assert_eq!(m.len(), 4, "{m:?}");
        assert!(m.values().all(|&v| (v as f64 / 8000.0 - 0.25).abs() < 0.025), "{m:?}");
    }

    /// 042 (AMB-042-15): with `c1_ladder = 0` (the default) the picks are the earlier ones,
    /// whatever `[[ladder]]` lists; the ladder never enters the C2 PFSP pool.
    #[test]
    fn zero_ladder_share_keeps_earlier_picks_042() {
        for base in [cfg(), t1self_cfg(), t1self2_cfg()] {
            let mut with = base.clone();
            with.ladder = rush_cfg().ladder;
            let (mut a, mut b) = (League::new("r1", 9, None, base), League::new("r1", 9, None, with));
            for _ in 0..3000 {
                assert_eq!(a.next(), b.next());
            }
        }
        let mut l = League::new("r1", 7, None, rush_cfg());
        l.stage = Stage::C2;
        assert!(!l.pfsp_pool(false).iter().any(|(_, k, _)| k.starts_with("steamhammeroverpool")));
    }

    /// 042 "재개 시 설정" (AMB-042-15): the ladder list and its share come from the current config
    /// on resume; the entries' win rates go on through the win table keys.
    #[test]
    fn resume_takes_the_current_ladder_042_r1() {
        let mut l = League::new("r1", 7, None, rush_cfg());
        let (opponent, key, class) = t1_entry("zzzkbot", Race::Zerg).opponent().unwrap();
        for _ in 0..50 {
            l.record(&Pick { agent: AgentKind::Main, opponent: opponent.clone(), key: key.clone(), class, fighting_spirit: true, ladder_group: None }, Race::Zerg, Outcome::Win);
        }
        let w = l.win_rate(AgentKind::Main, "zzzkbot-zerg");
        let mut c = rush_cfg();
        c.curriculum.c1_ladder = 1.0;
        (c.curriculum.c1_self, c.curriculum.c1_snapshot, c.curriculum.c1_anchor) = (0.0, 0.0, 0.0);
        c.ladder = vec![t1_entry("zzzkbot", Race::Zerg), t1_entry("ualbertabot9pool", Race::Zerg)];
        let mut m = League::from_bytes(&l.to_bytes(), c).unwrap();
        assert_eq!(m.win_rate(AgentKind::Main, "zzzkbot-zerg"), w);
        for _ in 0..300 {
            let k = m.next().key;
            assert!(k == "zzzkbot-zerg" || k == "ualbertabot9pool-zerg", "{k}");
        }
    }

    /// 044-R10 / 042-R22 ② (user decisions 2026-11-14): `config/rl_m2_race.toml` realises lobby
    /// races terran 24 : protoss 28 : zerg 28 : Random 20 over many picks, T0 / T1 half and half in
    /// each, half of zerg T0 the rush ladder (4 entries equal), T1 bots of a race equal, both maps,
    /// Main only, no self-play or model opponent and no E-Regression build (fixed seed sample).
    #[test]
    fn m2_race_shares_044_r10() {
        let text = include_str!("../../../config/rl_m2_race.toml").replace("\r\n", "\n").replacen("\nzerg = \"\"\n", "\nzerg = \"checkpoints/s/ckpt-1\"\n", 1);
        let c: RlConfig = toml::from_str(&text).unwrap();
        c.validate().unwrap_or_else(|e| panic!("{e:#}"));
        let mut l = League::new("m2-rl-fix-1", 7, None, c);
        const RUSH: [&str; 4] = ["t0-zerg-rush", "ualbertabot9pool10lings-zerg", "ualbertabot9pool12lings-zerg", "ualbertabot9poolhydra-zerg"];
        let n = 100_000;
        let mut keys: BTreeMap<String, usize> = BTreeMap::new();
        // (race, "t0" | "t1") with the zerg rush ladder in zerg T0 (044-R10).
        let mut split: BTreeMap<(&str, &str), usize> = BTreeMap::new();
        let mut fs = 0;
        for _ in 0..n {
            let p = l.next();
            assert_eq!(p.agent, AgentKind::Main);
            let race = match &p.opponent {
                Opponent::T0 { race, build } => {
                    assert!(matches!(build, T0Build::Standard | T0Build::ZergRush), "no E-Regression build: {build:?}");
                    *race
                }
                Opponent::T1 { race, .. } => *race,
                o => panic!("no self-play or model opponent: {o:?}"),
            };
            let r = race_name(race);
            assert_eq!(p.ladder_group.as_deref(), Some(r), "{}: the entry's group is its lobby race", p.key);
            let tier = if p.class == Class::T0 || RUSH.contains(&p.key.as_str()) { "t0" } else { "t1" };
            *split.entry((r, tier)).or_default() += 1;
            *keys.entry(p.key).or_default() += 1;
            fs += usize::from(p.fighting_spirit);
        }
        let f = |x: usize| x as f64 / n as f64;
        let race_share = |r: &str| f(split.iter().filter(|((x, _), _)| *x == r).map(|(_, c)| *c).sum());
        for (r, want) in [("terran", 0.24), ("protoss", 0.28), ("zerg", 0.28), ("random", 0.20)] {
            let got = race_share(r);
            assert!((got - want).abs() < 0.006, "{r}: {got} vs {want}");
            let t0 = f(split.get(&(r, "t0")).copied().unwrap_or(0)) / got;
            assert!((t0 - 0.5).abs() < 0.012, "{r}: T0 share {t0}");
        }
        let zerg_t0 = split[&("zerg", "t0")] as f64;
        let rush: usize = RUSH.iter().map(|k| keys.get(*k).copied().unwrap_or(0)).sum();
        assert!((rush as f64 / zerg_t0 - 0.5).abs() < 0.02, "rush ladder {rush} of zerg T0 {zerg_t0}");
        // Every entry at its share of all games (%).
        let want: [(&str, f64); 17] = [
            ("t0-terran", 12.0), ("ualbertabot-terran", 4.0), ("steamhammer-terran", 4.0), ("iron-terran", 4.0),
            ("t0-protoss", 14.0), ("ualbertabot-protoss", 7.0), ("stardust-protoss", 7.0),
            ("t0-zerg", 7.0), ("t0-zerg-rush", 1.75), ("ualbertabot9pool10lings-zerg", 1.75), ("ualbertabot9pool12lings-zerg", 1.75), ("ualbertabot9poolhydra-zerg", 1.75),
            ("ualbertabot-zerg", 28.0 / 6.0), ("steamhammer-zerg", 28.0 / 6.0), ("zzzkbot-zerg", 28.0 / 6.0),
            ("t0-random", 10.0), ("ualbertabot-random", 10.0),
        ];
        assert_eq!(keys.keys().cloned().collect::<BTreeSet<_>>(), want.iter().map(|(k, _)| k.to_string()).collect::<BTreeSet<_>>());
        for (k, pct) in want {
            let got = 100.0 * f(keys[k]);
            assert!((got - pct).abs() < 0.5, "{k}: {got}% vs {pct}%");
        }
        assert!((f(fs) - 0.5).abs() < 0.01, "both maps: fighting spirit {}", f(fs));
    }

    /// 042 사다리 설정 검사 (AMB-042-15): an entry's opponent, key and class, and the entries
    /// that are errors.
    #[test]
    fn ladder_entry_opponents_042() {
        let (o, k, c) = t0_entry(T0Build::ZergRush, Race::Zerg).opponent().unwrap();
        assert_eq!((o, k.as_str(), c), (Opponent::T0 { race: Race::Zerg, build: T0Build::ZergRush }, "t0-zerg-rush", Class::T0));
        let (_, k, _) = t0_entry(T0Build::Standard, Race::Protoss).opponent().unwrap();
        assert_eq!(k, "t0-protoss");
        let (o, k, c) = t1_entry("steamhammeroverpool", Race::Zerg).opponent().unwrap();
        assert_eq!((o, k.as_str(), c), (Opponent::T1 { name: "steamhammeroverpool".into(), race: Race::Zerg }, "steamhammeroverpool-zerg", Class::T1));
        assert!(t0_entry(T0Build::ZergRush, Race::Terran).opponent().is_err(), "build of another race");
        assert!(t1_entry("zzzkbot", Race::Unknown).opponent().is_err());
        assert!(t0_entry(T0Build::Standard, Race::Unknown).opponent().is_err());
        // 021-R7 / 044-R10 (2026-11-14): a Random entry plays lobby Random in every game, keys
        // `t0-random` / `<name>-random`; a one-race build cannot be Random (a T1 entry's
        // three-race check against [[t1]] is in the config check).
        let (o, k, c) = t0_entry(T0Build::Standard, Race::Random).opponent().unwrap();
        assert_eq!((o, k.as_str(), c), (Opponent::T0 { race: Race::Random, build: T0Build::Standard }, "t0-random", Class::T0));
        let (o, k, c) = t1_entry("ualbertabot", Race::Random).opponent().unwrap();
        assert_eq!((o, k.as_str(), c), (Opponent::T1 { name: "ualbertabot".into(), race: Race::Random }, "ualbertabot-random", Class::T1));
        for b in [T0Build::ZergRush, T0Build::TerranDrop, T0Build::ProtossDt, T0Build::ZergMuta, T0Build::TerranReact] {
            assert!(t0_entry(b, Race::Random).opponent().is_err(), "{b:?}");
        }
        assert!(t1_entry(" ", Race::Zerg).opponent().is_err());
        assert!(LadderEntry { kind: LadderKind::T0, build: Some(T0Build::ZergRush), name: Some("x".into()), race: Race::Zerg, group: String::new(), weight: None }.opponent().is_err());
        assert!(LadderEntry { kind: LadderKind::T0, build: None, name: None, race: Race::Zerg, group: String::new(), weight: None }.opponent().is_err());
        assert!(LadderEntry { kind: LadderKind::T1, build: Some(T0Build::ZergRush), name: Some("x".into()), race: Race::Zerg, group: String::new(), weight: None }.opponent().is_err());
    }

    fn grouped(e: LadderEntry, g: &str) -> LadderEntry {
        LadderEntry { group: g.into(), ..e }
    }

    /// The `rush-s2` league shape (AMB-042-16): the rush entries of `rush_cfg` as "rush", two
    /// macro entries, ladder only (no model shares) and groups `rush` / `macro` by `shares`.
    fn grouped_cfg(shares: &[(&str, f64)]) -> RlConfig {
        let mut c = rush_cfg();
        (c.curriculum.c1_self, c.curriculum.c1_snapshot, c.curriculum.c1_anchor) = (0.0, 0.0, 0.0);
        c.ladder = c.ladder.into_iter().map(|e| grouped(e, "rush")).collect();
        c.ladder.push(grouped(t0_entry(T0Build::Standard, Race::Zerg), "macro"));
        c.ladder.push(grouped(t1_entry("ualbertabot2hatchhydra", Race::Zerg), "macro"));
        c.curriculum.ladder_groups = shares.iter().map(|(g, x)| (g.to_string(), *x)).collect();
        c
    }

    const MACRO_KEYS: [&str; 2] = ["t0-zerg", "ualbertabot2hatchhydra-zerg"];

    /// 042 사다리 그룹 / 042-AC6 (AMB-042-16): ladder games draw the group by `ladder_groups`
    /// (fixed seed sample ratio), whatever the win rates; the pick names its group.
    #[test]
    fn ladder_groups_share_games_042_amb_042_16() {
        for (rush, mac) in [(0.5, 0.5), (0.8, 0.2), (1.0, 0.0)] {
            let mut l = League::new("r1", 11, None, grouped_cfg(&[("rush", rush), ("macro", mac)]));
            // Main beats the macro entries every time: f_var weights 0 there, uniform inside the group.
            for k in MACRO_KEYS {
                l.agents[0].win.insert(k.to_string(), 1.0);
            }
            let n = 8000;
            let mut counts: BTreeMap<String, usize> = BTreeMap::new();
            for _ in 0..n {
                let p = l.next();
                let g = p.ladder_group.clone().expect("a ladder game names its group");
                let want = if MACRO_KEYS.contains(&p.key.as_str()) { "macro" } else { "rush" };
                assert_eq!(g, want, "{}", p.key);
                *counts.entry(g).or_default() += 1;
            }
            let share = counts.get("macro").copied().unwrap_or(0) as f64 / n as f64;
            assert!((share - mac).abs() < 0.02, "{rush}/{mac}: {counts:?}");
        }
        // Not a ladder game: no group.
        let mut c = grouped_cfg(&[("rush", 0.5), ("macro", 0.5)]);
        (c.curriculum.c1_ladder, c.curriculum.c1_self) = (0.0, 1.0);
        let mut l = League::new("r1", 11, None, c);
        assert!((0..200).all(|_| l.next().ladder_group.is_none()));
    }

    /// 042 사다리 그룹 / 042-AC6 (AMB-042-16): inside a group the pick is f_var over that group's
    /// entries only.
    #[test]
    fn ladder_f_var_within_group_042_amb_042_16() {
        let mut l = League::new("r1", 5, None, grouped_cfg(&[("rush", 0.0), ("macro", 1.0)]));
        l.agents[0].win.insert("t0-zerg".into(), 0.99);
        let mut m: BTreeMap<String, usize> = BTreeMap::new();
        for _ in 0..4000 {
            *m.entry(l.next().key).or_default() += 1;
        }
        assert_eq!(m.keys().cloned().collect::<BTreeSet<_>>(), MACRO_KEYS.iter().map(|k| k.to_string()).collect::<BTreeSet<_>>(), "macro entries only: {m:?}");
        let get = |k: &str| m.get(k).copied().unwrap_or(0);
        assert!(get("ualbertabot2hatchhydra-zerg") > 10 * get("t0-zerg").max(1), "{m:?}");
    }

    /// 042-R19 / 042-AC9 (AMB-042-19): in a `"fixed"` group the entries come in proportion to
    /// `[[ladder]] weight` (absent 1.0) on a fixed-seed sample whatever the win table says; the
    /// other group keeps its method (f_var); a resumed league uses the current file's weights.
    #[test]
    fn ladder_fixed_by_weight_042_r19() {
        let fixed = |rush_weights: &[Option<f64>]| {
            let mut c = grouped_cfg(&[("rush", 1.0), ("macro", 0.0)]);
            for (e, w) in c.ladder.iter_mut().zip(rush_weights) {
                e.weight = *w;
            }
            c.curriculum.ladder_pick = LadderPickSetting::PerGroup([("rush".to_string(), LadderPick::Fixed), ("macro".to_string(), LadderPick::FVar)].into_iter().collect());
            c
        };
        let count = |l: &mut League, n: usize| {
            let mut m: BTreeMap<String, usize> = BTreeMap::new();
            for _ in 0..n {
                *m.entry(l.next().key).or_default() += 1;
            }
            m
        };
        let n = 20_000;
        let ws = [Some(0.35), Some(0.30), Some(0.20), Some(0.15)];
        for win in [0.0, 0.5, 1.0] {
            let mut l = League::new("r1", 13, None, fixed(&ws));
            // The win table is not read: an extreme table changes nothing.
            for k in RUSH_KEYS {
                l.agents[0].win.insert(k.to_string(), win);
            }
            let m = count(&mut l, n);
            for (k, w) in RUSH_KEYS.iter().zip(ws) {
                let share = m.get(*k).copied().unwrap_or(0) as f64 / n as f64;
                assert!((share - w.unwrap()).abs() < 0.015, "win {win}: {k} {share} vs {w:?} ({m:?})");
            }
        }
        // Absent weight = 1.0; weight 0 = never.
        let mut l = League::new("r1", 13, None, fixed(&[None, Some(2.0), Some(0.0), None]));
        let m = count(&mut l, n);
        let share = |k: &str| m.get(k).copied().unwrap_or(0) as f64 / n as f64;
        assert!((share("t0-zerg-rush") - 0.25).abs() < 0.015 && (share("ualbertabot-zerg") - 0.5).abs() < 0.015 && (share("zzzkbot-zerg") - 0.25).abs() < 0.015, "{m:?}");
        assert_eq!(share("steamhammeroverpool-zerg"), 0.0);
        // The macro group keeps f_var.
        let mut c = fixed(&ws);
        c.curriculum.ladder_groups = [("rush".to_string(), 0.0), ("macro".to_string(), 1.0)].into_iter().collect();
        let mut l = League::new("r1", 5, None, c.clone());
        l.agents[0].win.insert("t0-zerg".into(), 0.99);
        let m = count(&mut l, 4000);
        assert!(m.get("ualbertabot2hatchhydra-zerg").copied().unwrap_or(0) > 10 * m.get("t0-zerg").copied().unwrap_or(1).max(1), "{m:?}");
        // 042 "재개 시 설정": the resumed league picks by the current file's weights.
        let l = League::new("r1", 13, None, fixed(&ws));
        let mut r = League::from_bytes(&l.to_bytes(), fixed(&[Some(0.0), Some(0.0), Some(0.0), Some(1.0)])).unwrap();
        assert!((0..500).all(|_| r.next().key == "zzzkbot-zerg"));
    }

    /// 042-R19 / 042-AC9: `f_var` and `band` never read `weight` (the same draws with and without
    /// it), and an entry without `weight` serialises as before (no key).
    #[test]
    fn no_fixed_keeps_earlier_picks_042_r19() {
        let base = grouped_cfg(&[("rush", 0.7), ("macro", 0.3)]);
        let mut band = base.clone();
        band.curriculum.ladder_pick = LadderPickSetting::PerGroup([("rush".to_string(), LadderPick::Band)].into_iter().collect());
        for c in [base, band] {
            assert!(!serde_json::to_string(&c.ladder).unwrap().contains("weight"), "no weight in the serialised entries");
            let mut weighted = c.clone();
            for (i, e) in weighted.ladder.iter_mut().enumerate() {
                e.weight = Some(0.1 + i as f64);
            }
            let (mut a, mut b) = (League::new("r1", 9, None, c), League::new("r1", 9, None, weighted));
            assert!((0..3000).all(|_| a.next() == b.next()));
        }
    }

    /// 042 사다리 그룹 (AMB-042-16): without `ladder_groups` the picks are the earlier ones (same
    /// draws, same opponents) whatever `group` the entries carry; the group is still recorded.
    #[test]
    fn no_ladder_groups_keeps_earlier_picks_042_amb_042_16() {
        let base = rush_cfg();
        let mut tagged = base.clone();
        tagged.ladder = tagged.ladder.into_iter().enumerate().map(|(i, e)| grouped(e, if i % 2 == 0 { "rush" } else { "macro" })).collect();
        let (mut a, mut b) = (League::new("r1", 9, None, base), League::new("r1", 9, None, tagged.clone()));
        for _ in 0..3000 {
            let (x, y) = (a.next(), b.next());
            assert_eq!((&x.opponent, &x.key, x.class, x.fighting_spirit), (&y.opponent, &y.key, y.class, y.fighting_spirit));
            assert_eq!(x.ladder_group.is_some(), y.ladder_group.is_some());
            if let Some(g) = &y.ladder_group {
                assert_eq!(x.ladder_group.as_deref(), Some(""));
                let e = tagged.ladder.iter().find(|e| e.opponent().unwrap().1 == y.key).unwrap();
                assert_eq!(g, &e.group);
            }
        }
    }

    /// 042 승률 표 / 042-AC6 (AMB-042-16): the EMA decay is `[league] win_rate_decay` of the
    /// current config (resume included); every pick reads this one table.
    #[test]
    fn win_rate_decay_from_config_042_amb_042_16() {
        let pick = |key: &str| Pick { agent: AgentKind::Main, opponent: Opponent::T1 { name: "zzzkbot".into(), race: Race::Zerg }, key: key.into(), class: Class::T1, fighting_spirit: true, ladder_group: None };
        let mut c = cfg();
        assert_eq!(c.league.win_rate_decay, 0.99, "rl.toml default");
        let mut l = League::new("r1", 7, None, c.clone());
        l.record(&pick("zzzkbot-zerg"), Race::Zerg, Outcome::Loss);
        assert!((l.win_rate(AgentKind::Main, "zzzkbot-zerg") - 0.495).abs() < 1e-12);
        c.league.win_rate_decay = 0.97;
        let mut m = League::from_bytes(&l.to_bytes(), c).unwrap();
        assert!((m.win_rate(AgentKind::Main, "zzzkbot-zerg") - 0.495).abs() < 1e-12, "values go on");
        m.record(&pick("zzzkbot-zerg"), Race::Zerg, Outcome::Win);
        assert!((m.win_rate(AgentKind::Main, "zzzkbot-zerg") - (0.97 * 0.495 + 0.03)).abs() < 1e-12, "the new decay from the next game");
        m.record(&pick("uranium-main@r1"), Race::Terran, Outcome::Draw);
        assert!((m.win_rate(AgentKind::Main, "uranium-main@r1") - 0.5).abs() < 1e-12);
    }

    /// 033-R11 (AMB-042-16): in the "scout" mode the build / units channel weights are
    /// `[reward.scout]` / `[reward.survive]` `weight`, r_value follows its stages, and no C2 decay
    /// applies; "z" keeps the earlier schedule.
    #[test]
    fn scout_mode_reward_weights_033_r11() {
        let base = RewardWeights { win: 1.0, build: 0.3, units: 0.3, value: 0.05 };
        let mut c = cfg();
        c.reward.pseudo = PseudoMode::Scout;
        c.curriculum.r_value_stages = vec![Stage::C0, Stage::C1];
        let mut l = League::new("r1", 7, None, c);
        assert_eq!(l.reward_weights(&base), [1.0, 0.15, 0.1, 0.05]);
        l.stage = Stage::C1;
        assert_eq!(l.reward_weights(&base), [1.0, 0.15, 0.1, 0.05]);
        l.stage = Stage::C2;
        l.c2_start = Some(0);
        l.agents[0].updates = 1_000_000;
        assert_eq!(l.reward_weights(&base), [1.0, 0.15, 0.1, 0.0], "no C2 decay in scout");
        let mut z = league();
        z.stage = Stage::C2;
        z.c2_start = Some(0);
        z.agents[0].updates = 1_000_000;
        assert_eq!(z.reward_weights(&base), [1.0, 0.0, 0.0, 0.0], "z: decayed as before");
    }

    /// 042 사다리 선택 `band` / 042-AC7 (AMB-042-17): the weight formula (center, width), the
    /// exclusion below `exclude_below` only when some entry is at or above it, no exclusion when
    /// every entry is below, uniform over the remaining entries when every weight underflows.
    #[test]
    fn band_weights_042_amb_042_17() {
        let b = BandConfig::default();
        assert_eq!((b.center, b.width, b.exclude_below), (0.5, 0.15, 0.10));
        let close = |a: f64, x: f64| (a - x).abs() < 1e-12;
        assert!(close(b.weight(0.5), 1.0));
        assert!(close(b.weight(0.65), (-1.0f64).exp()) && close(b.weight(0.35), (-1.0f64).exp()));
        assert!(close(b.weight(0.2), (-4.0f64).exp()));
        let other = BandConfig { center: 0.3, width: 0.1, exclude_below: 0.2 };
        assert!(close(other.weight(0.5), (-4.0f64).exp()) && close(other.weight(0.3), 1.0));
        // Exclusion: 0.05 and 0.09 drop out because 0.5 and 0.1 are at or above 0.10.
        let w = b.weights(&[0.05, 0.5, 0.09, 0.10]);
        assert_eq!((w[0], w[2]), (0.0, 0.0));
        assert!(close(w[1], 1.0) && close(w[3], b.weight(0.10)));
        // Every entry below: none excluded, the formula for all.
        let w = b.weights(&[0.0, 0.05, 0.09]);
        assert!(w.iter().zip([0.0, 0.05, 0.09]).all(|(a, x)| close(*a, b.weight(x)) && *a > 0.0), "{w:?}");
        // Every remaining weight 0 (underflow): the remaining entries uniform.
        let narrow = BandConfig { center: 0.5, width: 1e-6, exclude_below: 0.10 };
        assert_eq!(narrow.weights(&[0.05, 0.9, 0.95]), [0.0, 1.0, 1.0]);
        assert_eq!(narrow.weights(&[0.01, 0.02]), [1.0, 1.0]);
        // 설정 검사.
        b.validate().unwrap();
        BandConfig { center: 0.0, width: 1.0, exclude_below: 0.0 }.validate().unwrap();
        BandConfig { center: 1.0, width: 0.01, exclude_below: 0.99 }.validate().unwrap();
        for bad in [BandConfig { center: -0.1, ..b }, BandConfig { center: 1.1, ..b }, BandConfig { width: 0.0, ..b }, BandConfig { width: -1.0, ..b }, BandConfig { exclude_below: 1.0, ..b }, BandConfig { exclude_below: -0.1, ..b }, BandConfig { width: f64::NAN, ..b }] {
            assert!(bad.validate().is_err(), "{bad:?}");
        }
    }

    /// 042 사다리 선택 `band` / 042-AC7 (AMB-042-17) in the league: inside the band group the picks
    /// follow the band weights over Main's win-table values (unseen = 0.5, excluded entries never
    /// picked); groups not in the per-group table use f_var.
    #[test]
    fn band_pick_per_group_042_amb_042_17() {
        let mut c = grouped_cfg(&[("rush", 1.0), ("macro", 0.0)]);
        c.curriculum.ladder_pick = LadderPickSetting::PerGroup(BTreeMap::from([("rush".to_string(), LadderPick::Band)]));
        c.validate().unwrap();
        let mut l = League::new("r1", 13, None, c.clone());
        // t0-zerg-rush unseen (0.5), ualbertabot 0.35, steamhammeroverpool 0.05 (excluded), zzzkbot 0.8.
        for (k, x) in [("ualbertabot-zerg", 0.35), ("steamhammeroverpool-zerg", 0.05), ("zzzkbot-zerg", 0.8)] {
            l.agents[0].win.insert(k.to_string(), x);
        }
        let n = 20_000;
        let mut m: BTreeMap<String, usize> = BTreeMap::new();
        for _ in 0..n {
            *m.entry(l.next().key).or_default() += 1;
        }
        let b = BandConfig::default();
        let w = [b.weight(0.5), b.weight(0.35), 0.0, b.weight(0.8)];
        let total: f64 = w.iter().sum();
        for (k, wk) in RUSH_KEYS.iter().zip(w) {
            let got = m.get(*k).copied().unwrap_or(0) as f64 / n as f64;
            assert!((got - wk / total).abs() < 0.015, "{k}: {got} vs {} ({m:?})", wk / total);
        }
        assert_eq!(m.get("steamhammeroverpool-zerg"), None, "excluded");
        // The macro group is not in the table: f_var, the same picks as a plain "f_var" setting.
        let mut f = grouped_cfg(&[("rush", 0.0), ("macro", 1.0)]);
        f.curriculum.ladder_pick = c.curriculum.ladder_pick.clone();
        let mut plain = f.clone();
        plain.curriculum.ladder_pick = LadderPickSetting::All(LadderPick::FVar);
        let (mut x, mut y) = (League::new("r1", 3, None, f), League::new("r1", 3, None, plain));
        for _ in 0..2000 {
            assert_eq!(x.next(), y.next());
        }
    }

    /// 042 (AMB-042-17): `ladder_pick = "f_var"` (string, table naming every group f_var, or
    /// absent) gives the earlier picks draw for draw; `"band"` for every group differs.
    #[test]
    fn f_var_settings_keep_earlier_picks_042_amb_042_17() {
        let base = grouped_cfg(&[("rush", 0.5), ("macro", 0.5)]);
        let mut table = base.clone();
        table.curriculum.ladder_pick = LadderPickSetting::PerGroup(BTreeMap::from([("rush".to_string(), LadderPick::FVar), ("macro".to_string(), LadderPick::FVar)]));
        let mut empty = base.clone();
        empty.curriculum.ladder_pick = LadderPickSetting::PerGroup(BTreeMap::new());
        let mut band = base.clone();
        band.curriculum.ladder_pick = LadderPickSetting::All(LadderPick::Band);
        let wins = [("ualbertabot-zerg", 0.9), ("zzzkbot-zerg", 0.02), ("t0-zerg", 0.3)];
        let league = |c: RlConfig| {
            let mut l = League::new("r1", 21, None, c);
            for (k, x) in wins {
                l.agents[0].win.insert(k.to_string(), x);
            }
            l
        };
        let mut a = league(base.clone());
        let (mut b, mut e, mut d) = (league(table), league(empty), league(band));
        let mut differs = false;
        for _ in 0..3000 {
            let x = a.next();
            assert_eq!(x, b.next());
            assert_eq!(x, e.next());
            differs |= x != d.next();
        }
        assert!(differs, "band picks differently");
        assert_eq!(base.curriculum.ladder_pick, LadderPickSetting::default());
        assert_eq!(LadderPickSetting::default().for_group(Some("rush")), LadderPick::FVar);
    }
}
