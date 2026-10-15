//! Matchup plan files and host lists (spec 020, 021-R7).

use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use uranium_proto::Race;
use uranium_proto::control::{GameSpec, MapKey, PlayerKind, PlayerSpec};

#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct Plan {
    pub name: String,
    #[serde(default = "d_seed_base")]
    pub seed_base: u32,
    /// Every game uses `seed_base` (determinism check, 011-AC4).
    #[serde(default)]
    pub same_seed: bool,
    #[serde(default = "d_max_frames")]
    pub max_frames: u32,
    #[serde(default = "d_wall_timeout")]
    pub wall_timeout_s: u32,
    #[serde(default)]
    pub privileged: bool,
    #[serde(default)]
    pub save_replay: bool,
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

#[derive(Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MapSel {
    FightingSpirit,
    CircuitBreaker,
    /// Alternate between both target maps.
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
    /// T0 scripted bot for the given race, served by the embedded agent (021 T0).
    T0 { race: RaceSel },
    /// Registered third-party bot on the host (021 T1).
    Oss { name: String, race: RaceSel },
}

#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct Matchup {
    pub map: MapSel,
    pub games: u32,
    pub p0: Side,
    pub p1: Side,
    /// Probability that p1 picks Random instead of its race (021-R7).
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
    let plan: Plan = toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
    if plan.matchups.is_empty() {
        bail!("plan has no [[matchup]]");
    }
    Ok(plan)
}

pub fn load_hosts(path: &Path) -> Result<Vec<HostEntry>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(toml::from_str::<HostsFile>(&text)?.hosts)
}

/// One scheduled game plus what the summary needs to know about it.
#[derive(Debug, Clone)]
pub struct Job {
    pub matchup: usize,
    pub spec: GameSpec,
}

/// Deterministic pseudo-random in [0,1) from the game index (no RNG state to persist).
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
                    Side::T0 { race } => (*race, PlayerKind::Bridge { agent_addr: agent_addr.to_string() }),
                    Side::Oss { name, race } => (*race, PlayerKind::OssBot { name: name.clone() }),
                };
                PlayerSpec { race: if random { Race::Random } else { race.into() }, kind }
            };
            let random = unit_hash(id) < m.p1_random_ratio;
            jobs.push(Job {
                matchup: mi,
                spec: GameSpec {
                    game_id: id,
                    map,
                    players: [side(&m.p0, false), side(&m.p1, random)],
                    seed: Some(if plan.same_seed { plan.seed_base } else { plan.seed_base.wrapping_add(id as u32) }),
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
    fn unknown_fields_are_rejected() {
        assert!(toml::from_str::<Plan>("name='x'\nbogus=1\n[[matchup]]\nmap='both'\ngames=1\np0={kind='t0',race='terran'}\np1={kind='t0',race='zerg'}").is_err());
    }
}
