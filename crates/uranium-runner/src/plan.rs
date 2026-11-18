use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;
use uranium_agent::T0Build;
use uranium_agent::infer::{DEFAULT_CHAT, SampleMode, SurrenderMode};
use uranium_match::branch::{BranchPlanLine, BranchPlayer, GameBranch, HORIZON_MARGIN, Intervention, SnapshotRule, Snapshots};
use uranium_match::explore::{ExploreConfig, ExploreGame};
use uranium_env::features::extract::action_table;
use uranium_proto::Race;
use uranium_store::elog;
use uranium_proto::control::{GameSpec, MapKey, PlayerKind, PlayerSpec};

#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct Plan {
    pub name: String,
    #[serde(default = "d_seed_base")]
    pub seed_base: u32,
    #[serde(default)]
    pub same_seed: bool,
    #[serde(default)]
    pub seeds: Option<Vec<u32>>,
    #[serde(default = "d_max_frames")]
    pub max_frames: u32,
    #[serde(default = "d_wall_timeout")]
    pub wall_timeout_s: u32,
    #[serde(default)]
    pub privileged: bool,
    #[serde(default)]
    pub save_replay: bool,
    #[serde(default)]
    pub zero_last_failed: bool,
    #[serde(default = "d_surrender")]
    pub surrender: SurrenderMode,
    #[serde(default = "d_surrender_chat")]
    pub surrender_chat: String,
    #[serde(default)]
    pub memory_log: bool,
    #[serde(default)]
    pub act_log: bool,
    #[serde(default)]
    pub snapshots: Vec<SnapshotRule>,
    #[serde(default)]
    pub branches: Option<String>,
    #[serde(default)]
    pub verify_state: bool,
    #[serde(default)]
    pub explore: Option<ExploreConfig>,
    #[serde(skip)]
    pub branch_lines: Vec<BranchPlanLine>,
    #[serde(rename = "matchup")]
    pub matchups: Vec<Matchup>,
}

fn d_seed_base() -> u32 {
    1000
}
fn d_max_frames() -> u32 {
    86_400
}
fn d_wall_timeout() -> u32 {
    1800
}
fn d_surrender() -> SurrenderMode {
    SurrenderMode::Shadow
}
fn d_surrender_chat() -> String {
    DEFAULT_CHAT.into()
}

pub fn check_surrender_chat(chat: &str) -> Result<()> {
    if chat.len() > uranium_proto::MAX_SURRENDER_CHAT {
        bail!("surrender_chat is {} bytes, at most {} (020-R14)", chat.len(), uranium_proto::MAX_SURRENDER_CHAT);
    }
    if chat.contains('\0') {
        bail!("surrender_chat contains NUL (020-R14)");
    }
    Ok(())
}

#[derive(Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MapSel {
    FightingSpirit,
    CircuitBreaker,
    Both,
}

#[derive(Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RaceSel {
    Terran,
    Protoss,
    Zerg,
    Random,
}

impl From<RaceSel> for Race {
    fn from(r: RaceSel) -> Self {
        match r {
            RaceSel::Terran => Race::Terran,
            RaceSel::Protoss => Race::Protoss,
            RaceSel::Zerg => Race::Zerg,
            RaceSel::Random => Race::Random,
        }
    }
}

#[derive(Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Side {
    T0 {
        race: RaceSel,
        #[serde(default)]
        build: T0Build,
        #[serde(default)]
        name: Option<String>,
    },
    Oss { name: String, race: RaceSel },
    Model {
        checkpoint: String,
        #[serde(default)]
        sample: SampleMode,
        #[serde(default)]
        name: Option<String>,
    },
}

impl Side {
    pub fn race(&self) -> RaceSel {
        match self {
            Side::T0 { race, .. } | Side::Oss { race, .. } => *race,
            Side::Model { .. } => RaceSel::Terran,
        }
    }

    pub fn model(&self) -> Option<ModelKey> {
        match self {
            Side::Model { checkpoint, sample, .. } => Some((checkpoint.clone(), *sample)),
            _ => None,
        }
    }

    pub fn display_name(&self, random: bool) -> String {
        match self {
            Side::Model { name: Some(n), .. } | Side::T0 { name: Some(n), .. } => n.clone(),
            Side::Model { checkpoint, .. } => run_name(checkpoint),
            Side::T0 { race, build, .. } => {
                let race = if random { "random".to_string() } else { format!("{race:?}").to_lowercase() };
                match *build {
                    T0Build::Standard => format!("T0-{race}"),
                    b => format!("T0-{race}-{}", b.name()),
                }
            }
            Side::Oss { name, .. } => name.clone(),
        }
    }
}

pub fn run_name(checkpoint: &str) -> String {
    let parts: Vec<&str> = checkpoint.split(['/', '\\']).filter(|s| !s.is_empty()).collect();
    match parts.as_slice() {
        [.., run, last] if last.starts_with("ckpt-") || *last == "last" => run.to_string(),
        [.., last] => last.to_string(),
        [] => checkpoint.to_string(),
    }
}

pub type ModelKey = (String, SampleMode);

#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct Matchup {
    pub map: MapSel,
    pub games: u32,
    pub p0: Side,
    pub p1: Side,
    #[serde(default)]
    pub p1_random_ratio: f64,
}

#[derive(Deserialize, Debug, Clone)]
pub struct HostEntry {
    pub addr: String,
    pub max_games: Option<u32>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct HostsFile {
    #[serde(rename = "host")]
    pub hosts: Vec<HostEntry>,
}

pub fn load_plan(path: &Path) -> Result<Plan> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let mut plan: Plan = toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
    if plan.matchups.is_empty() {
        bail!("plan has no [[matchup]]");
    }
    check_surrender_chat(&plan.surrender_chat)?;
    plan.validate_seeds()?;
    for (i, m) in plan.matchups.iter().enumerate() {
        for side in [&m.p0, &m.p1] {
            if let Side::T0 { race, build, .. } = side {
                if build.race().is_some_and(|r| r != Race::from(*race)) {
                    bail!("matchup {i}: T0 build `{}` plays {:?}, not {race:?}", build.name(), build.race().unwrap());
                }
            }
        }
        if m.p1_random_ratio > 0.0 && matches!(m.p1, Side::T0 { build, .. } if build != T0Build::Standard) {
            bail!("matchup {i}: p1_random_ratio needs a multi-race bot, not a scenario build (021-R7)");
        }
        if plan.memory_log && m.p0.model().is_some() && m.p1.model().is_some() {
            bail!("matchup {i}: memory_log needs one model player per game (one log per game, 041-R28)");
        }
    }
    if plan.memory_log && !plan.save_replay {
        bail!("memory_log goes with save_replay: the relabel build reads the log beside the replay (041-R28)");
    }
    if plan.memory_log && plan.matchups.iter().all(|m| m.p0.model().is_none() && m.p1.model().is_none()) {
        elog!("[env] WARNING: plan `{}` has memory_log = true but no model player: no decision log is written (041-R28)", plan.name);
    }
    plan.check_branch_keys(path.parent().unwrap_or(Path::new(".")))
        .with_context(|| format!("plan {} (020-R22, 051)", path.display()))?;
    Ok(plan)
}

pub fn map_name(m: MapKey) -> &'static str {
    match m {
        MapKey::FightingSpirit => "fighting_spirit",
        MapKey::CircuitBreaker => "circuit_breaker",
    }
}

pub fn branch_opponent(job: &Job) -> String {
    let id = crate::opponent_id(&job.spec.players[1], job.t0_builds[1], None, &Default::default());
    id.split('@').next().unwrap_or(&id).to_string()
}

pub fn shared_dir_env() -> Option<PathBuf> {
    std::env::var_os("URANIUM_SHARED_DIR").filter(|v| !v.is_empty()).map(PathBuf::from)
}

fn resolve_shared(p: &str, shared: Option<&Path>) -> Result<PathBuf> {
    let p = Path::new(p);
    if p.is_absolute() {
        return Ok(p.to_path_buf());
    }
    let s = shared.with_context(|| format!("branch list path {} is relative to the shared drive: URANIUM_SHARED_DIR is not set (051-R10)", p.display()))?;
    Ok(s.join(p))
}

fn is_sha256(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

impl Plan {
    pub fn games(&self) -> u64 {
        self.matchups.iter().map(|m| u64::from(m.games)).sum()
    }

    pub fn validate_seeds(&self) -> Result<()> {
        if let Some(seeds) = &self.seeds {
            if self.same_seed {
                bail!("plan has both `seeds` and `same_seed = true` (020-R20)");
            }
            if seeds.len() as u64 != self.games() {
                bail!("plan `seeds` has {} seeds for {} games (020-R20)", seeds.len(), self.games());
            }
        }
        Ok(())
    }

    pub fn seed_of(&self, id: u64) -> u32 {
        match &self.seeds {
            Some(seeds) => seeds[(id - 1) as usize],
            None if self.same_seed => self.seed_base,
            None => self.seed_base.wrapping_add(id as u32),
        }
    }

    pub fn uses_branch_keys(&self) -> bool {
        self.act_log || !self.snapshots.is_empty() || self.branches.is_some() || self.verify_state || self.explore.is_some()
    }

    pub fn horizon(&self) -> Option<u32> {
        self.max_frames.checked_sub(HORIZON_MARGIN).filter(|h| *h > 0)
    }

    pub fn branch_none_actions(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for r in self.branch_lines.iter().flat_map(|l| &l.alt.rules) {
            if !out.contains(&r.action) {
                out.push(r.action.clone());
            }
        }
        out
    }

    fn check_explore_keys(&self) -> Result<()> {
        let Some(cfg) = &self.explore else { return Ok(()) };
        cfg.validate()?;
        ensure!(self.act_log && self.memory_log, "explore needs act_log = true and memory_log = true (020-R22 ⑤)");
        ensure!(self.snapshots.is_empty() && self.branches.is_none(), "explore with snapshots or branches: an exploration game is neither a trunk nor a branch (054)");
        for (i, m) in self.matchups.iter().enumerate() {
            ensure!(matches!(m.p0, Side::Model { sample: SampleMode::Stochastic, .. }), "explore: matchup {i}: players[0] must be a model with sample = \"stochastic\" (054-R1)");
        }
        Ok(())
    }

    fn check_branch_keys(&mut self, dir: &Path) -> Result<()> {
        self.check_explore_keys()?;
        ensure!(!self.verify_state || self.branches.is_some(), "verify_state goes with branches (020-R22 ③)");
        let trunk = !self.snapshots.is_empty();
        let branch = self.branches.is_some();
        if !trunk && !branch {
            return Ok(());
        }
        ensure!(!(trunk && branch), "snapshots and branches in one plan: a branch game is not a trunk (051-R3, 051-R10)");
        let what = if trunk { "snapshots" } else { "branches" };
        ensure!(!trunk || self.act_log, "snapshots go with act_log = true (the trunk's act log is the branch input, 051-R2, 051-R10)");
        ensure!(self.surrender == SurrenderMode::Off, "{what}: surrender must be \"off\" (051-R1 ③: the branch prefix has no inference to carry the 031-R12 check)");
        ensure!(!self.privileged, "{what}: privileged must be false (051-R1 ③)");
        ensure!(self.save_replay, "{what}: save_replay must be true (051-R1 ③)");
        ensure!(self.zero_last_failed, "{what}: zero_last_failed must be true (051-R1 ③)");
        let horizon = self.horizon().with_context(|| format!("{what}: max_frames {} must be H + {HORIZON_MARGIN} with H > 0 (051-R1)", self.max_frames))?;
        for (i, m) in self.matchups.iter().enumerate() {
            ensure!(matches!(m.p0, Side::Model { sample: SampleMode::Stochastic, .. }), "{what}: matchup {i}: players[0] must be a model with sample = \"stochastic\" (051-R1 ②)");
            ensure!(matches!(m.p1, Side::T0 { .. }), "{what}: matchup {i}: players[1] must be a T0 bot (051-R1 ①)");
            ensure!(m.p1_random_ratio == 0.0, "{what}: matchup {i}: p1_random_ratio must be 0 (the opponent must be the trunk's, 051-R1 ④)");
        }
        let table = action_table();
        Snapshots::new(&self.snapshots, table)?;
        if let Some(file) = &self.branches {
            let path = dir.join(file);
            let f = std::fs::File::open(&path).with_context(|| format!("branch list {}", path.display()))?;
            let lines: Vec<BranchPlanLine> = uranium_match::branch::read_jsonl(std::io::BufReader::new(f)).with_context(|| format!("branch list {}", path.display()))?;
            ensure!(lines.len() as u64 == self.games(), "branch list {} has {} lines for {} games (020-R22 ③)", path.display(), lines.len(), self.games());
            for (job, l) in expand(self, "branch-check").iter().zip(&lines) {
                let n = job.spec.game_id;
                let ctx = || format!("branch list {} line {n}", path.display());
                ensure!(job.spec.seed == Some(l.seed), "{}: seed {} but plan game {n} has seed {:?} (020-R22 ③)", ctx(), l.seed, job.spec.seed);
                ensure!(map_name(job.spec.map) == l.map, "{}: map {} but plan game {n} is on {} (020-R22 ③)", ctx(), l.map, map_name(job.spec.map));
                let opp = branch_opponent(job);
                ensure!(opp == l.opponent, "{}: opponent {} but plan game {n} plays {opp} (020-R22 ③)", ctx(), l.opponent);
                ensure!(l.f_t < horizon, "{}: f_t {} is not before the horizon {horizon} (051-R1)", ctx(), l.f_t);
                ensure!(is_sha256(&l.state_sha256), "{}: state_sha256 is not a lowercase sha256", ctx());
                ensure!(l.repeat >= 1, "{}: repeat numbers start at 1", ctx());
                ensure!(!l.alt.name.is_empty(), "{}: alternative without a name", ctx());
                Intervention::new(&l.alt, table, l.f_t).with_context(ctx)?;
            }
            self.branch_lines = lines;
        }
        Ok(())
    }

    pub fn check_branch_paths(&self, shared: Option<&Path>) -> Result<()> {
        for l in &self.branch_lines {
            resolve_shared(&l.acts, shared)?;
            resolve_shared(&l.state, shared)?;
        }
        Ok(())
    }

    pub fn game_branch(&self, n: u64, stage_dir: &Path, shared: Option<&Path>) -> Result<Option<GameBranch>> {
        if !self.uses_branch_keys() {
            return Ok(None);
        }
        let branch = match self.branch_lines.get((n as usize).wrapping_sub(1)) {
            Some(line) => Some(BranchPlayer {
                line: line.clone(),
                acts: resolve_shared(&line.acts, shared)?,
                state: resolve_shared(&line.state, shared)?,
                horizon: self.horizon().context("branch plan without a horizon (051-R1)")?,
                none_actions: self.branch_none_actions(),
                verify_state: self.verify_state,
            }),
            None if self.branches.is_some() => bail!("plan game {n} has no branch list line (020-R22 ③)"),
            None => None,
        };
        Ok(Some(GameBranch {
            plan_game: u32::try_from(n).context("plan game number exceeds u32")?,
            acts_dir: self.act_log.then(|| stage_dir.join("acts")),
            snapshots: self.snapshots.clone(),
            states_dir: (!self.snapshots.is_empty()).then(|| stage_dir.join("states")),
            branch,
            explore: self.explore.clone().map(|cfg| ExploreGame { cfg, seed: self.seed_of(n) }),
        }))
    }
}

pub fn load_hosts(path: &Path) -> Result<Vec<HostEntry>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(toml::from_str::<HostsFile>(&text)?.hosts)
}

#[derive(Debug, Clone)]
pub struct Job {
    pub matchup: usize,
    pub spec: GameSpec,
    pub t0_builds: [T0Build; 2],
    pub models: [Option<ModelKey>; 2],
    pub policies: [Option<uranium_agent::PolicySpec>; 2],
    pub opponent: Option<String>,
}

fn unit_hash(i: u64) -> f64 {
    let mut x = i.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    (x >> 11) as f64 / (1u64 << 53) as f64
}

pub fn expand(plan: &Plan, agent_addr: &str) -> Vec<Job> {
    let mut jobs = Vec::new();
    let mut id: u64 = 0;
    for (mi, m) in plan.matchups.iter().enumerate() {
        for g in 0..m.games {
            id += 1;
            let map = match m.map {
                MapSel::FightingSpirit => MapKey::FightingSpirit,
                MapSel::CircuitBreaker => MapKey::CircuitBreaker,
                MapSel::Both if g % 2 == 0 => MapKey::FightingSpirit,
                MapSel::Both => MapKey::CircuitBreaker,
            };
            let side = |s: &Side, random: bool| {
                let (race, kind) = match s {
                    Side::T0 { race, .. } => (*race, PlayerKind::Bridge { agent_addr: agent_addr.to_string() }),
                    Side::Model { .. } => (RaceSel::Terran, PlayerKind::Bridge { agent_addr: agent_addr.to_string() }),
                    Side::Oss { name, race } => (*race, PlayerKind::OssBot { name: name.clone() }),
                };
                PlayerSpec { race: if random { Race::Random } else { race.into() }, kind, name: Some(s.display_name(random)) }
            };
            let random = unit_hash(id) < m.p1_random_ratio && m.p1.model().is_none();
            let build = |s: &Side| match s {
                Side::T0 { build, .. } => *build,
                Side::Oss { .. } | Side::Model { .. } => T0Build::Standard,
            };
            jobs.push(Job {
                matchup: mi,
                t0_builds: [build(&m.p0), build(&m.p1)],
                models: [m.p0.model(), m.p1.model()],
                policies: [None, None],
                opponent: None,
                spec: GameSpec {
                    game_id: id,
                    map,
                    players: [side(&m.p0, false), side(&m.p1, random)],
                    seed: Some(plan.seed_of(id)),
                    max_frames: plan.max_frames,
                    privileged: plan.privileged,
                    save_replay: plan.save_replay,
                    wall_timeout_s: plan.wall_timeout_s,
                },
            });
        }
    }
    jobs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn surrender_options_020_r14() {
        let base = "[[matchup]]\nmap='both'\ngames=1\np0={kind='t0',race='terran'}\np1={kind='t0',race='zerg'}";
        let p: Plan = toml::from_str(&format!("name='x'\n{base}")).unwrap();
        assert_eq!((p.surrender, p.surrender_chat.as_str()), (SurrenderMode::Shadow, "gg"));
        let p: Plan = toml::from_str(&format!("name='x'\nsurrender='on'\nsurrender_chat='good game'\n{base}")).unwrap();
        assert_eq!((p.surrender, p.surrender_chat.as_str()), (SurrenderMode::On, "good game"));
        let p: Plan = toml::from_str(&format!("name='x'\nsurrender='off'\nsurrender_chat=''\n{base}")).unwrap();
        assert_eq!((p.surrender, p.surrender_chat.as_str()), (SurrenderMode::Off, ""));
        assert!(toml::from_str::<Plan>(&format!("name='x'\nsurrender='maybe'\n{base}")).is_err());
        assert!(check_surrender_chat("").is_ok());
        assert!(check_surrender_chat(&"a".repeat(79)).is_ok());
        assert!(check_surrender_chat(&"a".repeat(80)).is_err());
    }

    const PLAN: &str = r#"
name = "t"
seed_base = 10
[[matchup]]
map = "both"
games = 4
p0 = { kind = "t0", race = "terran" }
p1 = { kind = "oss", name = "ualbertabot", race = "protoss" }
p1_random_ratio = 1.0
[[matchup]]
map = "circuit_breaker"
games = 1
p0 = { kind = "t0", race = "zerg" }
p1 = { kind = "t0", race = "protoss" }
"#;

    #[test]
    fn display_names_020_r16() {
        assert_eq!(run_name("checkpoints/core-abl-gtrxl-pt/ckpt-10000"), "core-abl-gtrxl-pt");
        assert_eq!(run_name(r"D:\shared\checkpoints\arch-abl-all-pt\last"), "arch-abl-all-pt");
        assert_eq!(run_name("checkpoints/bc-s1"), "bc-s1");
        let text = "name='x'
[[matchup]]
map='both'
games=1
p0={kind='model',checkpoint='checkpoints/arch-abl-pool-pt/ckpt-10000'}
p1={kind='t0',race='zerg',build='zerg_muta'}
[[matchup]]
map='both'
games=1
p0={kind='model',checkpoint='c/r/ckpt-1',name='mine'}
p1={kind='oss',name='iron',race='terran'}";
        let plan: Plan = toml::from_str(text).unwrap();
        let jobs = expand(&plan, "a");
        let names = |j: &Job| j.spec.players.clone().map(|p| p.name.unwrap());
        assert_eq!(names(&jobs[0]), ["arch-abl-pool-pt".to_string(), "T0-zerg-zerg_muta".to_string()]);
        assert_eq!(names(&jobs[1]), ["mine".to_string(), "iron".to_string()]);
        let plan: Plan = toml::from_str(PLAN).unwrap();
        assert_eq!(expand(&plan, "a")[4].spec.players[0].name.as_deref(), Some("T0-zerg"));
        assert_eq!(Side::T0 { race: RaceSel::Zerg, build: T0Build::Standard, name: None }.display_name(true), "T0-random");
    }

    #[test]
    fn plan_expands_to_jobs() {
        let plan: Plan = toml::from_str(PLAN).unwrap();
        let jobs = expand(&plan, "127.0.0.1:7600");
        assert_eq!(jobs.len(), 5);
        assert_eq!(jobs[0].spec.map, MapKey::FightingSpirit);
        assert_eq!(jobs[1].spec.map, MapKey::CircuitBreaker);
        assert_eq!(jobs[0].spec.seed, Some(11));
        assert_eq!(jobs[0].spec.players[1].race, Race::Random);
        assert!(matches!(&jobs[0].spec.players[1].kind, PlayerKind::OssBot { name } if name == "ualbertabot"));
        assert!(matches!(&jobs[0].spec.players[0].kind, PlayerKind::Bridge { agent_addr } if agent_addr == "127.0.0.1:7600"));
        assert_eq!(jobs[4].matchup, 1);
        assert_eq!(jobs[4].spec.players[1].race, Race::Protoss);
        let ids: Vec<u64> = jobs.iter().map(|j| j.spec.game_id).collect();
        assert_eq!(ids, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn random_ratio_is_roughly_respected() {
        let mut plan: Plan = toml::from_str(PLAN).unwrap();
        plan.matchups[0].games = 1000;
        plan.matchups[0].p1_random_ratio = 0.2;
        let n = expand(&plan, "x").iter().filter(|j| j.matchup == 0 && j.spec.players[1].race == Race::Random).count();
        assert!((150..250).contains(&n), "{n}");
    }

    #[test]
    fn e_regression_050_scenario_builds_reach_jobs() {
        let plan: Plan = toml::from_str("name='x'\n[[matchup]]\nmap='both'\ngames=2\np0={kind='t0',race='terran'}\np1={kind='t0',race='zerg',build='zerg_muta'}").unwrap();
        let jobs = expand(&plan, "x");
        assert_eq!(jobs[1].t0_builds, [T0Build::Standard, T0Build::ZergMuta]);
    }

    #[test]
    fn e_regression_050_scenario_build_race_and_random_are_checked() {
        let dir = std::env::temp_dir().join(format!("uranium-plan-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("p.toml");
        let base = "name='x'\n[[matchup]]\nmap='both'\ngames=1\np0={kind='t0',race='terran'}\n";
        std::fs::write(&path, format!("{base}p1={{kind='t0',race='protoss',build='zerg_rush'}}")).unwrap();
        assert!(load_plan(&path).is_err());
        std::fs::write(&path, format!("{base}p1={{kind='t0',race='zerg',build='zerg_rush'}}\np1_random_ratio=0.2")).unwrap();
        assert!(load_plan(&path).is_err());
        std::fs::write(&path, format!("{base}p1={{kind='t0',race='zerg',build='zerg_rush'}}")).unwrap();
        assert!(load_plan(&path).is_ok());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn memory_log_plan_key_041_r28() {
        let dir = std::env::temp_dir().join(format!("uranium-plan-memlog-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("p.toml");
        let model = "{kind='model',checkpoint='c',sample='stochastic'}";
        let plan = |head: &str, p1: &str| format!("name='x'\n{head}\n[[matchup]]\nmap='both'\ngames=1\np0={model}\np1={p1}\n");
        std::fs::write(&path, plan("save_replay=true", "{kind='t0',race='zerg'}")).unwrap();
        assert!(!load_plan(&path).unwrap().memory_log, "off by default");
        std::fs::write(&path, plan("save_replay=true\nmemory_log=true", "{kind='t0',race='zerg'}")).unwrap();
        assert!(load_plan(&path).unwrap().memory_log);
        std::fs::write(&path, plan("memory_log=true", "{kind='t0',race='zerg'}")).unwrap();
        assert!(format!("{:#}", load_plan(&path).unwrap_err()).contains("save_replay"));
        std::fs::write(&path, plan("save_replay=true\nmemory_log=true", model)).unwrap();
        assert!(format!("{:#}", load_plan(&path).unwrap_err()).contains("one model player"));
        let jobs = expand(&toml::from_str::<Plan>(&plan("save_replay=true\nmemory_log=true", "{kind='t0',race='zerg'}")).unwrap(), "a:1");
        use crate::JobSource;
        let q = crate::QueueSource::new(jobs.clone()).with_replays(dir.join("replays"));
        assert_eq!(q.memory_log(&jobs[0]), None, "no log without with_memory_logs");
        let q = q.with_memory_logs(dir.join("memory"));
        assert_eq!(q.memory_log(&jobs[0]), Some(dir.join("memory").join("g1.jsonl")));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn model_side_plays_terran_over_the_bridge_020_r11() {
        let plan: Plan = toml::from_str("name='m'\n[[matchup]]\nmap='fighting_spirit'\ngames=2\np0={kind='model',checkpoint='ckpt/bc',sample='stochastic'}\np1={kind='t0',race='zerg'}").unwrap();
        let jobs = expand(&plan, "a:1");
        assert_eq!(jobs[0].spec.players[0].race, Race::Terran);
        assert!(matches!(&jobs[0].spec.players[0].kind, PlayerKind::Bridge { .. }));
        assert_eq!(jobs[0].models, [Some(("ckpt/bc".to_string(), SampleMode::Stochastic)), None]);
        assert_eq!(jobs[0].t0_builds, [T0Build::Standard, T0Build::Standard]);
        let argmax: Plan = toml::from_str("name='m'\n[[matchup]]\nmap='both'\ngames=1\np0={kind='model',checkpoint='c'}\np1={kind='t0',race='terran'}").unwrap();
        assert_eq!(expand(&argmax, "a:1")[0].models[0], Some(("c".to_string(), SampleMode::Argmax)));
    }

    #[test]
    fn plan_seeds_020_r20() {
        let base = "[[matchup]]\nmap='fighting_spirit'\ngames=2\np0={kind='t0',race='terran'}\np1={kind='t0',race='zerg'}\n[[matchup]]\nmap='circuit_breaker'\ngames=1\np0={kind='t0',race='terran'}\np1={kind='t0',race='zerg'}";
        let seeds = |text: &str| expand(&toml::from_str::<Plan>(text).unwrap(), "a").iter().map(|j| j.spec.seed.unwrap()).collect::<Vec<_>>();
        assert_eq!(seeds(&format!("name='x'\nseeds=[45001, 7, 45003]\n{base}")), [45_001, 7, 45_003]);
        assert_eq!(seeds(&format!("name='x'\nseed_base=10\nseeds=[1, 2, 3]\n{base}")), [1, 2, 3], "seed_base unused");
        assert_eq!(seeds(&format!("name='x'\nseed_base=10\n{base}")), [11, 12, 13], "no seeds: seed_base + n");
        assert_eq!(seeds(&format!("name='x'\nseed_base=10\nsame_seed=true\n{base}")), [10, 10, 10]);
        assert_eq!(seeds(&format!("name='x'\n{base}")), [1_001, 1_002, 1_003], "default seed_base");
        let dir = std::env::temp_dir().join(format!("uranium-plan-seeds-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("p.toml");
        let load = |head: &str| {
            std::fs::write(&path, format!("name='x'\n{head}\n{base}")).unwrap();
            load_plan(&path)
        };
        assert_eq!(load("seeds=[5, 6, 7]").unwrap().seeds, Some(vec![5, 6, 7]));
        assert!(load("seeds=[5, 6]").is_err(), "too short");
        assert!(load("seeds=[5, 6, 7, 8]").is_err(), "too long");
        assert!(load("seeds=[]").is_err());
        assert!(load("seeds=[5, 6, 7]\nsame_seed=true").is_err(), "with same_seed");
        assert!(load("seeds=[5, -6, 7]").is_err(), "u32");
        load("same_seed=true").unwrap();
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rush5_check_plan_050_r6() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../config/plans/rush5_check.toml");
        let plan = load_plan(&path).unwrap();
        assert_eq!(plan.seeds.as_deref(), Some(&crate::start_slots::check_seeds(4, crate::start_slots::CHECK_FIRST_SEED)[..]));
        let blocks: Vec<(MapSel, Side)> = plan.matchups.iter().map(|m| (m.map, m.p1.clone())).collect();
        let t0 = Side::T0 { race: RaceSel::Zerg, build: T0Build::ZergRush, name: None };
        let uab = Side::Oss { name: "ualbertabot12poollings".into(), race: RaceSel::Zerg };
        assert_eq!(blocks, [(MapSel::FightingSpirit, t0.clone()), (MapSel::CircuitBreaker, t0), (MapSel::FightingSpirit, uab.clone()), (MapSel::CircuitBreaker, uab)]);
        for m in &plan.matchups {
            assert_eq!((m.games, m.p1_random_ratio), (32, 0.0));
            assert!(matches!(&m.p0, Side::Model { sample: SampleMode::Stochastic, .. }), "{:?}", m.p0);
        }
        assert!(plan.save_replay && !plan.zero_last_failed && !plan.privileged);
        assert_eq!((plan.max_frames, plan.surrender), (86_400, SurrenderMode::Shadow));
        let jobs = expand(&plan, "a");
        assert_eq!(jobs.len(), 128);
        assert!(jobs.iter().zip(plan.seeds.as_ref().unwrap()).all(|(j, s)| j.spec.seed == Some(*s) && j.spec.players[1].race == Race::Zerg));
    }

    #[test]
    fn teacher_check_plans_050_r8() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../config/plans");
        let rush = load_plan(&dir.join("rush5_check.toml")).unwrap();
        let plan = load_plan(&dir.join("teacher_check.toml")).unwrap();
        assert_eq!(plan.seeds, rush.seeds);
        let teacher = Side::T0 { race: RaceSel::Terran, build: T0Build::TerranReact, name: None };
        for (m, r) in plan.matchups.iter().zip(&rush.matchups) {
            assert_eq!((m.map, &m.p1, m.games, m.p0.clone()), (r.map, &r.p1, 32, teacher.clone()));
        }
        assert_eq!(plan.matchups.len(), 4);
        assert!(plan.save_replay && plan.matchups.iter().all(|m| m.p0.model().is_none() && m.p1.model().is_none()));
        for p in ["teacher_check.toml", "teacher_macro_check.toml", "teacher_t0zerg_check.toml"] {
            assert_eq!(load_plan(&dir.join(p)).unwrap().max_frames, 14_400, "{p}");
        }
        let jobs = expand(&plan, "a");
        assert!(jobs.iter().all(|j| j.t0_builds[0] == T0Build::TerranReact && j.spec.players[0].race == Race::Terran));

        let mac = load_plan(&dir.join("teacher_macro_check.toml")).unwrap();
        let opponents: Vec<Side> = mac.matchups.iter().map(|m| m.p1.clone()).collect();
        let oss = |n: &str| Side::Oss { name: n.into(), race: RaceSel::Zerg };
        assert_eq!(
            opponents,
            [oss("ualbertabot2hatchhydra"), oss("ualbertabot3hatchmuta"), oss("steamhammer2hatchmuta"), Side::T0 { race: RaceSel::Zerg, build: T0Build::ZergMuta, name: None }]
        );
        assert!(mac.matchups.iter().all(|m| m.p0 == teacher && m.games == 10 && m.map == MapSel::Both && m.p1_random_ratio == 0.0));
        assert_eq!(expand(&mac, "a").len(), 40);
        let t0 = load_plan(&dir.join("teacher_t0zerg_check.toml")).unwrap();
        assert_eq!(t0.matchups.len(), 1);
        let m = &t0.matchups[0];
        assert_eq!((&m.p0, &m.p1, m.games, m.map), (&teacher, &Side::T0 { race: RaceSel::Zerg, build: T0Build::Standard, name: None }, 10, MapSel::Both));
    }

    #[test]
    fn teacher_botgen_plans_040_r14() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../config/plans");
        let teacher = Side::T0 { race: RaceSel::Terran, build: T0Build::TerranReact, name: None };
        let oss = |n: &str| Side::Oss { name: n.into(), race: RaceSel::Zerg };
        let t0 = |build| Side::T0 { race: RaceSel::Zerg, build, name: None };
        let rush = load_plan(&dir.join("teacher_botgen_rush.toml")).unwrap();
        let macro_ = load_plan(&dir.join("teacher_botgen_macro.toml")).unwrap();
        let blocks = |p: &Plan| p.matchups.iter().map(|m| (m.p1.clone(), m.map, m.games)).collect::<Vec<_>>();
        let both = |opps: Vec<Side>, n: u32| opps.into_iter().flat_map(|o| [(o.clone(), MapSel::FightingSpirit, n), (o, MapSel::CircuitBreaker, n)]).collect::<Vec<_>>();
        assert_eq!(blocks(&rush), both(vec![t0(T0Build::ZergRush), oss("ualbertabot12poollings"), oss("ualbertabot9pool10lings"), oss("ualbertabot9poolhydra")], 200));
        assert_eq!(
            blocks(&macro_),
            both(vec![oss("ualbertabot2hatchhydra"), oss("ualbertabot3hatchmuta"), oss("steamhammer2hatchmuta"), t0(T0Build::ZergMuta), t0(T0Build::Standard)], 40)
        );
        let mut seen = std::collections::HashSet::new();
        for p in ["teacher_check.toml", "teacher_macro_check.toml", "teacher_t0zerg_check.toml"] {
            let c = load_plan(&dir.join(p)).unwrap();
            seen.extend(expand(&c, "c").iter().map(|j| j.spec.seed.unwrap()));
        }
        for (p, n) in [(&rush, 1_600), (&macro_, 400)] {
            assert!(p.save_replay && p.max_frames == 14_400 && p.matchups.iter().all(|m| m.p0 == teacher && m.p1_random_ratio == 0.0));
            let jobs = expand(p, "b");
            assert_eq!(jobs.len(), n);
            assert!(jobs.iter().all(|j| seen.insert(j.spec.seed.unwrap())), "a botgen seed repeats");
        }
    }

    #[test]
    fn teacher_tp_plans_022_r11_r12() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../config/plans");
        let teacher = Side::T0 { race: RaceSel::Terran, build: T0Build::TerranTp, name: None };
        let t0 = |race| Side::T0 { race, build: T0Build::Standard, name: None };
        let oss = |n: &str, race| Side::Oss { name: n.into(), race };
        let opponents =
            vec![t0(RaceSel::Protoss), t0(RaceSel::Terran), oss("ualbertabot", RaceSel::Protoss), oss("stardust", RaceSel::Protoss), oss("ualbertabot", RaceSel::Terran), oss("iron", RaceSel::Terran)];
        let mut seen = std::collections::HashSet::new();
        for p in ["teacher_check.toml", "teacher_macro_check.toml", "teacher_t0zerg_check.toml", "teacher_botgen_rush.toml", "teacher_botgen_macro.toml", "teacher_botgen_macro2.toml", "teacher_botgen_macro3.toml"] {
            seen.extend(expand(&load_plan(&dir.join(p)).unwrap(), "o").iter().map(|j| j.spec.seed.unwrap()));
        }
        for (file, t0_games, t1_games, total, max_frames) in [("teacher_tp_check.toml", 32, 10, 208, 86_400), ("teacher_botgen_tp.toml", 150, 25, 800, 14_400)] {
            let plan = load_plan(&dir.join(file)).unwrap();
            assert!(plan.save_replay && plan.max_frames == max_frames, "{file}");
            let blocks: Vec<(Side, MapSel, u32)> = plan.matchups.iter().map(|m| (m.p1.clone(), m.map, m.games)).collect();
            let want: Vec<(Side, MapSel, u32)> = opponents
                .iter()
                .enumerate()
                .flat_map(|(i, o)| {
                    let g = if i < 2 { t0_games } else { t1_games };
                    [(o.clone(), MapSel::FightingSpirit, g), (o.clone(), MapSel::CircuitBreaker, g)]
                })
                .collect();
            assert_eq!(blocks, want, "{file}");
            assert!(plan.matchups.iter().all(|m| m.p0 == teacher && m.p1_random_ratio == 0.0 && m.p0.model().is_none()), "{file}");
            let jobs = expand(&plan, "t");
            assert_eq!(jobs.len(), total, "{file}");
            assert!(jobs.iter().all(|j| j.t0_builds[0] == T0Build::TerranTp && j.spec.players[0].race == Race::Terran));
            assert!(jobs.iter().all(|j| seen.insert(j.spec.seed.unwrap())), "{file}: a seed repeats");
        }
    }

    const BRANCH_HEAD: &str = "save_replay=true\nzero_last_failed=true\nsurrender='off'\nmax_frames=11568\nact_log=true\n";
    const BRANCH_GAMES: &str = "[[matchup]]\nmap='fighting_spirit'\ngames=2\np0={kind='model',checkpoint='checkpoints/m2-s1/ckpt-39096',sample='stochastic'}\np1={kind='t0',race='zerg',build='zerg_rush'}\n[[matchup]]\nmap='circuit_breaker'\ngames=1\np0={kind='model',checkpoint='checkpoints/m2-s1/ckpt-39096',sample='stochastic'}\np1={kind='t0',race='zerg',build='zerg_rush'}\n";
    const SNAPSHOTS: &str = "[[snapshots]]\nname='b1-p1'\nfrom_frame=3240\nto_frame=5760\nany_open=['Build_Bunker']\n[[snapshots]]\nname='b1-p2'\nfrom_frame=3960\nto_frame=5760\nany_open=['Build_Bunker']\n";

    fn temp_dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("uranium-plan-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn line(seed: u32, map: &str, alt: uranium_match::branch::Alternative, repeat: u32) -> BranchPlanLine {
        BranchPlanLine {
            trunk: "trunk-job/b1-trunk/g1".into(),
            point: "trunk-job/b1-trunk/g1#b1-p1".into(),
            acts: "eval/trunk-job/b1-trunk/acts/g1-p0.jsonl".into(),
            opponent_acts: "eval/trunk-job/b1-trunk/acts/g1-p1.jsonl".into(),
            state: "eval/trunk-job/b1-trunk/states/g1-o412.state".into(),
            state_sha256: "a".repeat(64),
            i_t: 412,
            f_t: 3256,
            seed,
            map: map.into(),
            opponent: "t0-zerg-rush".into(),
            alt,
            repeat,
        }
    }

    fn alt(name: &str, rules: &[(&str, &str, u32, u32)]) -> uranium_match::branch::Alternative {
        use uranium_match::branch::{InterventionRule, Op};
        let rules = rules.iter().map(|(op, a, f, t)| InterventionRule { op: if *op == "force" { Op::Force } else { Op::Ban }, action: (*a).into(), from_s: *f, to_s: *t }).collect();
        uranium_match::branch::Alternative { name: name.into(), rules }
    }

    fn write_lines(path: &Path, lines: &[BranchPlanLine]) {
        let mut buf = Vec::new();
        uranium_match::branch::write_jsonl(&mut buf, lines).unwrap();
        std::fs::write(path, buf).unwrap();
    }

    #[test]
    fn plan_without_branch_keys_unchanged_020_r22() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../config/plans");
        for p in ["rush5_check.toml", "teacher_check.toml", "dagger_r1.toml"] {
            let plan = load_plan(&dir.join(p)).unwrap();
            assert!(!plan.uses_branch_keys() && plan.branch_lines.is_empty(), "{p}");
            assert!(plan.game_branch(1, Path::new("out"), None).unwrap().is_none(), "{p}");
        }
        let text = format!("name='x'\n{BRANCH_GAMES}");
        let plain: Plan = toml::from_str(&text).unwrap();
        assert!(!plain.act_log && plain.snapshots.is_empty() && plain.branches.is_none() && !plain.verify_state);
        assert_eq!(plain.horizon(), Some(86_400 - 48));
        let keyed: Plan = toml::from_str(&format!("name='x'\nact_log=true\n{BRANCH_GAMES}")).unwrap();
        let specs = |p: &Plan| expand(p, "a:1").into_iter().map(|j| (j.spec, j.models, j.t0_builds)).collect::<Vec<_>>();
        assert_eq!(specs(&plain), specs(&keyed));
        assert!(crate::check_branch_support(&plain).is_ok());
        assert!(crate::check_branch_support(&keyed).is_ok(), "the agent sessions read the board (051 U3)");
    }

    #[test]
    fn act_log_and_snapshots_settings_020_r22() {
        let dir = temp_dir("trunk");
        let path = dir.join("p.toml");
        std::fs::write(&path, "name='x'\nact_log=true\n[[matchup]]\nmap='both'\ngames=2\np0={kind='t0',race='terran'}\np1={kind='t0',race='zerg',build='zerg_rush'}\n").unwrap();
        let plan = load_plan(&path).unwrap();
        let g = plan.game_branch(2, Path::new("stage"), None).unwrap().unwrap();
        assert_eq!((g.plan_game, g.act_log_path(1)), (2, Some(Path::new("stage").join("acts").join("g2-p1.jsonl"))));
        assert!(g.states_dir.is_none() && g.snapshots.is_empty() && g.branch.is_none());
        std::fs::write(&path, format!("name='x'\n{BRANCH_HEAD}{SNAPSHOTS}{BRANCH_GAMES}")).unwrap();
        let plan = load_plan(&path).unwrap();
        assert_eq!(plan.horizon(), Some(11_520));
        let g = plan.game_branch(3, Path::new("stage"), None).unwrap().unwrap();
        assert_eq!(g.snapshots.iter().map(|r| r.name.as_str()).collect::<Vec<_>>(), ["b1-p1", "b1-p2"]);
        assert_eq!(g.state_path(412), Some(Path::new("stage").join("states").join("g3-o412.state")));
        assert!(g.branch.is_none());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn b2_trunk_plan_loads_051_r16() {
        let dir = temp_dir("b2");
        let path = dir.join("p.toml");
        let rule = |n: u32, after: &str| format!("[[snapshots]]\nname='b2-p{n}'\nfrom_frame=4320\nto_frame=11520\nany_open=[]\nmin_open_ops=3\n{after}");
        let snaps = format!("{}{}{}", rule(1, ""), rule(2, "after_rule='b2-p1'\nafter_s=60\nafter_gap=true\n"), rule(3, "after_rule='b2-p2'\nafter_s=60\nafter_gap=true\n"));
        let p0 = "p0={kind='model',checkpoint='checkpoints/m2-teacher-t2/ckpt-3258',sample='stochastic'}";
        let games = format!("[[matchup]]\nmap='fighting_spirit'\ngames=1\n{p0}\np1={{kind='t0',race='zerg'}}\n[[matchup]]\nmap='circuit_breaker'\ngames=1\n{p0}\np1={{kind='t0',race='protoss'}}\n");
        let head = BRANCH_HEAD.replace("max_frames=11568", "max_frames=17328");
        std::fs::write(&path, format!("name='b2'\n{head}{snaps}{games}")).unwrap();
        let plan = load_plan(&path).unwrap();
        assert_eq!(plan.horizon(), Some(17_280));
        let g = plan.game_branch(2, Path::new("stage"), None).unwrap().unwrap();
        assert_eq!(g.snapshots.iter().map(|r| (r.name.as_str(), r.after_gap, r.min_open_ops)).collect::<Vec<_>>(), [("b2-p1", false, Some(3)), ("b2-p2", true, Some(3)), ("b2-p3", true, Some(3))]);
        std::fs::write(&path, format!("name='b2'\n{head}{}{games}", rule(1, "after_gap=true\n"))).unwrap();
        assert!(load_plan(&path).is_err(), "after_gap without after_rule");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn branch_plan_load_checks_051_r1() {
        let dir = temp_dir("checks");
        let path = dir.join("p.toml");
        let load = |text: String| {
            std::fs::write(&path, text).unwrap();
            load_plan(&path).map_err(|e| format!("{e:#}"))
        };
        load(format!("name='x'\n{BRANCH_HEAD}{SNAPSHOTS}{BRANCH_GAMES}")).unwrap();
        let bad = |head: &str, games: &str, want: &str| {
            let e = load(format!("name='x'\n{head}{SNAPSHOTS}{games}")).unwrap_err();
            assert!(e.contains(want), "{want}: {e}");
        };
        bad(&BRANCH_HEAD.replace("surrender='off'", "surrender='shadow'"), BRANCH_GAMES, "surrender");
        bad(&BRANCH_HEAD.replace("surrender='off'\n", ""), BRANCH_GAMES, "surrender");
        bad(&format!("{BRANCH_HEAD}privileged=true\n"), BRANCH_GAMES, "privileged");
        bad(&BRANCH_HEAD.replace("save_replay=true", "save_replay=false"), BRANCH_GAMES, "save_replay");
        bad(&BRANCH_HEAD.replace("zero_last_failed=true", "zero_last_failed=false"), BRANCH_GAMES, "zero_last_failed");
        bad(&BRANCH_HEAD.replace("max_frames=11568", "max_frames=48"), BRANCH_GAMES, "max_frames");
        bad(BRANCH_HEAD, &BRANCH_GAMES.replacen(",sample='stochastic'", "", 1), "stochastic");
        bad(BRANCH_HEAD, &BRANCH_GAMES.replacen("{kind='t0',race='zerg',build='zerg_rush'}", "{kind='oss',name='iron',race='terran'}", 1), "T0 bot");
        bad(BRANCH_HEAD, &format!("{BRANCH_GAMES}p1_random_ratio=0.5\n").replacen("build='zerg_rush'", "build='standard'", 2), "p1_random_ratio");
        bad(&format!("{BRANCH_HEAD}verify_state=true\n"), BRANCH_GAMES, "verify_state");
        bad(&BRANCH_HEAD.replace("act_log=true\n", ""), BRANCH_GAMES, "act_log");
        let rules = |r: &str| load(format!("name='x'\n{BRANCH_HEAD}{r}{BRANCH_GAMES}")).unwrap_err();
        assert!(rules("[[snapshots]]\nname='a'\nfrom_frame=1\nto_frame=2\nany_open=['Build_Bunkr']\n").contains("Build_Bunkr"));
        assert!(rules("[[snapshots]]\nname='a'\nfrom_frame=1\nto_frame=2\n[[snapshots]]\nname='a'\nfrom_frame=1\nto_frame=2\n").contains("twice"));
        assert!(rules("[[snapshots]]\nname='a'\nfrom_frame=2\nto_frame=2\n").contains("empty window"));
        assert!(rules("[[snapshots]]\nname='a'\nfrom_frame=1\nto_frame=2\nbogus=1\n").contains("bogus"));
        write_lines(&dir.join("b.jsonl"), &[]);
        assert!(load(format!("name='x'\n{BRANCH_HEAD}branches='b.jsonl'\n{SNAPSHOTS}{BRANCH_GAMES}")).unwrap_err().contains("snapshots and branches"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn branch_list_checks_020_r22() {
        let dir = temp_dir("branches");
        let path = dir.join("p.toml");
        let seeds = "seeds=[45001, 45002, 45140]\n";
        let none = alt("none", &[]);
        let bunker = alt("force_bunker", &[("force", "Build_Bunker", 0, 0)]);
        let ban = alt("ban_bunker_90s", &[("ban", "Build_Bunker", 0, 90)]);
        let good = vec![line(45001, "fighting_spirit", none.clone(), 1), line(45002, "fighting_spirit", bunker.clone(), 1), line(45140, "circuit_breaker", ban.clone(), 2)];
        let load = |lines: &[BranchPlanLine], extra: &str| {
            write_lines(&dir.join("branches_plan.jsonl"), lines);
            std::fs::write(&path, format!("name='x'\n{BRANCH_HEAD}{seeds}{extra}branches='branches_plan.jsonl'\n{BRANCH_GAMES}")).unwrap();
            load_plan(&path).map_err(|e| format!("{e:#}"))
        };
        let plan = load(&good, "verify_state=true\n").unwrap();
        assert_eq!(plan.branch_lines, good);
        assert_eq!(plan.branch_none_actions(), ["Build_Bunker"]);
        let shared = dir.join("shared");
        let g = plan.game_branch(2, Path::new("stage"), Some(&shared)).unwrap().unwrap();
        let b = g.branch.as_ref().unwrap();
        assert_eq!(b.line, good[1]);
        assert_eq!(b.acts, shared.join("eval/trunk-job/b1-trunk/acts/g1-p0.jsonl"));
        assert_eq!(b.state, shared.join("eval/trunk-job/b1-trunk/states/g1-o412.state"));
        assert_eq!((b.horizon, b.verify_state, b.none_actions.clone()), (11_520, true, vec!["Build_Bunker".to_string()]));
        assert_eq!(g.act_log_path(0), Some(Path::new("stage").join("acts").join("g2-p0.jsonl")));
        assert!(g.states_dir.is_none(), "a branch game saves no state");
        assert!(plan.check_branch_paths(None).unwrap_err().to_string().contains("URANIUM_SHARED_DIR"));
        assert!(plan.game_branch(1, Path::new("stage"), None).is_err());
        plan.check_branch_paths(Some(&shared)).unwrap();
        let mut abs = good.clone();
        for l in &mut abs {
            l.acts = shared.join("a.jsonl").to_string_lossy().into_owned();
            l.state = shared.join("s.state").to_string_lossy().into_owned();
        }
        load(&abs, "").unwrap().check_branch_paths(None).unwrap();
        let err = |lines: Vec<BranchPlanLine>, want: &str| {
            let e = load(&lines, "").unwrap_err();
            assert!(e.contains(want), "{want}: {e}");
        };
        err(good[..2].to_vec(), "2 lines for 3 games");
        err([good.clone(), vec![good[0].clone()]].concat(), "4 lines for 3 games");
        let edit = |i: usize, f: &dyn Fn(&mut BranchPlanLine)| {
            let mut v = good.clone();
            f(&mut v[i]);
            v
        };
        err(edit(1, &|l| l.seed = 45003), "seed 45003");
        err(edit(2, &|l| l.map = "fighting_spirit".into()), "map fighting_spirit");
        err(edit(0, &|l| l.opponent = "t0-zerg".into()), "opponent t0-zerg");
        err(edit(0, &|l| l.f_t = 11_520), "horizon");
        err(edit(0, &|l| l.state_sha256 = "A".repeat(64)), "sha256");
        err(edit(0, &|l| l.repeat = 0), "repeat");
        err(edit(1, &|l| l.alt = alt("x", &[("force", "Build_Bunkr", 0, 0)])), "Build_Bunkr");
        err(edit(1, &|l| l.alt = alt("x", &[("ban", "NoOp", 0, 90)])), "NoOp");
        err(edit(1, &|l| l.alt = alt("x", &[("ban", "Build_Bunker", 90, 0)])), "empty window");
        let plans = [
            alt("cc_now", &[("force", "Build_CommandCenter", 0, 90)]),
            alt("factory_now", &[("force", "Build_Factory", 0, 60), ("ban", "Build_Barracks", 0, 120)]),
            alt("rax_more", &[("force", "Build_Barracks", 0, 30), ("force", "Build_Barracks", 30, 75), ("ban", "Build_Factory", 0, 180), ("ban", "Train_SCV", 0, 90), ("force", "Build_Bunker", 0, 30)]),
        ];
        let b3: Vec<_> = good.iter().zip(plans).map(|(l, a)| BranchPlanLine { alt: a, ..l.clone() }).collect();
        assert_eq!(load(&b3, "").unwrap().branch_none_actions(), ["Build_CommandCenter", "Build_Factory", "Build_Barracks", "Train_SCV", "Build_Bunker"], "055-R1");
        std::fs::write(dir.join("branches_plan.jsonl"), "{\"bogus\":1}\n").unwrap();
        std::fs::write(&path, format!("name='x'\n{BRANCH_HEAD}{seeds}branches='branches_plan.jsonl'\n{BRANCH_GAMES}")).unwrap();
        assert!(load_plan(&path).is_err(), "unknown fields");
        std::fs::write(&path, format!("name='x'\n{BRANCH_HEAD}{seeds}branches='missing.jsonl'\n{BRANCH_GAMES}")).unwrap();
        assert!(format!("{:#}", load_plan(&path).unwrap_err()).contains("missing.jsonl"));
        std::fs::remove_dir_all(&dir).ok();
    }

    const EXPLORE_HEAD: &str = "save_replay=true\nmemory_log=true\nact_log=true\nmax_frames=14400\nseeds=[7001, 7002, 7003]\n";
    const EXPLORE_GAMES: &str = "[[matchup]]\nmap='both'\ngames=2\np0={kind='model',checkpoint='checkpoints/f/ckpt-1',sample='stochastic'}\np1={kind='t0',race='zerg',build='zerg_rush'}\n[[matchup]]\nmap='circuit_breaker'\ngames=1\np0={kind='model',checkpoint='checkpoints/f/ckpt-1',sample='stochastic'}\np1={kind='oss',name='stardust',race='protoss'}\n";

    #[test]
    fn explore_plan_key_020_r22_054_r1() {
        let dir = temp_dir("explore");
        let path = dir.join("p.toml");
        let load = |text: String| {
            std::fs::write(&path, text).unwrap();
            load_plan(&path).map_err(|e| format!("{e:#}"))
        };
        let explore = "[explore]\npoints=3\nwindow_s=[120, 540]\ngap_s=60\nk=4\nepsilon=0.5\nmin_open=2\n";
        let plan = load(format!("name='x'\n{EXPLORE_HEAD}{explore}{EXPLORE_GAMES}")).unwrap();
        assert_eq!(plan.explore, Some(ExploreConfig::default()));
        assert!(plan.uses_branch_keys());
        let g = plan.game_branch(2, Path::new("stage"), None).unwrap().unwrap();
        assert_eq!(g.explore, Some(ExploreGame { cfg: ExploreConfig::default(), seed: 7002 }), "the plan game's seed (054-R2)");
        assert_eq!(g.act_log_path(0), Some(Path::new("stage").join("acts").join("g2-p0.jsonl")));
        assert!(g.states_dir.is_none() && g.branch.is_none() && g.snapshots.is_empty());
        assert_eq!(load(format!("name='x'\n{EXPLORE_HEAD}[explore]\n{EXPLORE_GAMES}")).unwrap().explore, Some(ExploreConfig::default()), "the 020-R22 values by default");
        let bad = |text: String, want: &str| {
            let e = load(text).unwrap_err();
            assert!(e.contains(want), "{want}: {e}");
        };
        bad(format!("name='x'\n{}{explore}{EXPLORE_GAMES}", EXPLORE_HEAD.replace("act_log=true\n", "")), "act_log = true and memory_log = true");
        bad(format!("name='x'\n{}{explore}{EXPLORE_GAMES}", EXPLORE_HEAD.replace("memory_log=true\n", "")), "act_log = true and memory_log = true");
        bad(format!("name='x'\n{EXPLORE_HEAD}{explore}{}", EXPLORE_GAMES.replacen(",sample='stochastic'", "", 1)), "stochastic");
        bad(format!("name='x'\n{EXPLORE_HEAD}{}{EXPLORE_GAMES}", explore.replace("epsilon=0.5", "epsilon=2.0")), "epsilon");
        bad(format!("name='x'\n{EXPLORE_HEAD}{}{EXPLORE_GAMES}", explore.replace("k=4", "k=4\nbogus=1")), "bogus");
        bad(format!("name='x'\n{EXPLORE_HEAD}{SNAPSHOTS}{explore}{EXPLORE_GAMES}"), "neither a trunk nor a branch");
        let plain = load(format!("name='x'\n{EXPLORE_HEAD}{EXPLORE_GAMES}")).unwrap();
        assert!(plain.explore.is_none() && plain.game_branch(1, Path::new("stage"), None).unwrap().unwrap().explore.is_none());
        let specs = |p: &Plan| expand(p, "a:1").into_iter().map(|j| (j.spec, j.models, j.t0_builds)).collect::<Vec<_>>();
        assert_eq!(specs(&plan), specs(&plain), "explore changes no game spec");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn unknown_fields_are_rejected() {
        assert!(toml::from_str::<Plan>("name='x'\nbogus=1\n[[matchup]]\nmap='both'\ngames=1\np0={kind='t0',race='terran'}\np1={kind='t0',race='zerg'}").is_err());
    }
}
