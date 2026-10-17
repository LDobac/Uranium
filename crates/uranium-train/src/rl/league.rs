//! League (042): who the learning agents play next, the win-rate table, PFSP, the curriculum
//! C0 → C1 → C2 (042-R5), snapshots and resets, learner time share (042-R7) and the reward
//! weight schedule (033). Pure logic with its own RNG so it is testable and resumable (042-R1).
//! Results are judged with 020-R13 before they reach `record`.

use std::collections::{BTreeMap, VecDeque};

use serde::{Deserialize, Serialize};
use uranium_agent::T0Build;
use uranium_env::reward::{Outcome, RewardWeights};
use uranium_proto::Race;

use super::config::RlConfig;

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
    pub t1: Vec<T1Bot>,
    /// 042-AC1 small setting: Main only, always the standard T0 bot of this race on Fighting
    /// Spirit (no self-play, no curriculum change).
    #[serde(default)]
    pub fixed_t0: Option<Race>,
    rng: u64,
    #[serde(skip)]
    cfg: Option<RlConfig>,
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
    pub fn new(run: &str, t1: Vec<T1Bot>, seed: u64, cfg: RlConfig) -> Self {
        Self {
            run: run.to_string(),
            stage: Stage::C0,
            c2_start: None,
            agents: vec![AgentState::default(); 3],
            windows: BTreeMap::new(),
            snapshots: Vec::new(),
            t1,
            fixed_t0: None,
            rng: seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1,
            cfg: Some(cfg),
        }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        serde_json::to_vec_pretty(self).expect("league serialises")
    }

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
        w.len() - 1
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
        let key = match build.id_tag() {
            Some(tag) => format!("t0-{}-{tag}", race_name(race)),
            None => format!("t0-{}", race_name(race)),
        };
        (Opponent::T0 { race: if random { Race::Random } else { race }, build }, key, Class::T0)
    }

    fn t1_pick(&mut self) -> Option<(Opponent, String, Class)> {
        if self.t1.is_empty() {
            return None;
        }
        let race = self.pick_race();
        let bots: Vec<T1Bot> = self.t1.iter().filter(|b| b.races.contains(&race)).cloned().collect();
        let bots = if bots.is_empty() { self.t1.clone() } else { bots };
        let bot = bots[(self.unit() * bots.len() as f64) as usize % bots.len()].clone();
        let race = if bot.races.contains(&race) { race } else { bot.races[0] };
        let random = bot.races.len() > 1 && self.unit() < self.cfg().league.bot_random_race;
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
                    let key = match b.id_tag() {
                        Some(tag) => format!("t0-{}-{tag}", race_name(r)),
                        None => format!("t0-{}", race_name(r)),
                    };
                    v.push((Opponent::T0 { race: r, build: b }, key, Class::T0));
                }
            }
            for bot in &self.t1 {
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
            return Pick { agent: AgentKind::Main, opponent, key: format!("t0-{}", race_name(race)), class: Class::T0, fighting_spirit: true };
        }
        let agent = self.pick_agent();
        let fighting_spirit = self.unit() < 0.5;
        let main_key = format!("uranium-main@{}", self.run);
        let selfplay = (Opponent::SelfPlay, main_key.clone(), Class::Model);
        let cc = self.cfg().curriculum.clone();
        let lc = self.cfg().league.clone();
        let (opponent, key, class) = match agent {
            AgentKind::Main => match self.stage {
                Stage::C0 => {
                    if self.unit() < cc.c0_t0 / (cc.c0_t0 + cc.c0_self) { self.t0_pick() } else { selfplay }
                }
                Stage::C1 => {
                    let u = self.unit() * (cc.c1_t0 + cc.c1_t1 + cc.c1_self);
                    if u < cc.c1_t0 {
                        self.t0_pick()
                    } else if u < cc.c1_t0 + cc.c1_t1 {
                        self.t1_pick().unwrap_or_else(|| self.t0_pick())
                    } else {
                        selfplay
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
        Pick { agent, opponent, key, class, fighting_spirit }
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
    fn advance(&mut self) -> bool {
        if self.fixed_t0.is_some() {
            return false;
        }
        let cc = self.cfg().curriculum.clone();
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

    /// 033 weights for an agent's advantage: r_value only in C0, r_build / r_units decay to 0
    /// over `pseudo_reward_decay_updates` Main updates after C2 starts.
    pub fn reward_weights(&self, base: &RewardWeights) -> [f32; 4] {
        let value = if self.stage == Stage::C0 { base.value } else { 0.0 };
        let decay = match self.c2_start {
            Some(s) => (1.0 - (self.updates(AgentKind::Main) - s) as f64 / self.cfg().curriculum.pseudo_reward_decay_updates).max(0.0) as f32,
            None => 1.0,
        };
        [base.win, base.build * decay, base.units * decay, value]
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

    fn league() -> League {
        League::new("r1", t1(), 7, RlConfig::default_file())
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
        let pick = |race: Race| Pick { agent: AgentKind::Main, opponent: Opponent::T0 { race, build: T0Build::Standard }, key: format!("t0-{}", race_name(race)), class: Class::T0, fighting_spirit: true };
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
        let mut lose = Pick { agent: AgentKind::LeagueExploiter, opponent: Opponent::T1 { name: "zzzkbot".into(), race: Race::Zerg }, key: "zzzkbot-zerg".into(), class: Class::T1, fighting_spirit: true };
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
        let mut m = League::from_bytes(&b, RlConfig::default_file()).unwrap();
        assert_eq!(l.next(), m.next());
    }
}
