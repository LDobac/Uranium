//! 041-R21 ⑥ / 041-R26 (AMB-041-15): the teacher / human mix of a BC epoch. Counts the epoch's
//! steps per source and enemy race, checks the shares (teacher = the `keep_sources` = 50 ± 3% of
//! the update steps; inside the human part the enemy-race shares of the S2 human data ± 3%p), and
//! computes the per-race human fractions (`[data.subsample] fraction_by_race`) that hit them for a
//! data seed. Every step of an epoch is trained exactly once (041-R13 lane walk), so step counts
//! are update-step counts.

use std::collections::BTreeMap;

use anyhow::{Result, ensure};
use serde::Serialize;

use crate::data::{GameSummary, SUBSAMPLE_SALT, game_unit};
use crate::split::{SUBSAMPLE_RACES, enemy_race};

/// 041-R21 ⑥: the teacher share of the mix epoch's update steps, and its tolerance.
pub const TEACHER_SHARE: f64 = 0.5;
pub const TEACHER_TOLERANCE: f64 = 0.03;
/// 041-R26 (AMB-041-15): the tolerance of each enemy race's share inside the human part.
pub const RACE_TOLERANCE: f64 = 0.03;
/// Float slack of the tolerance checks (shares are ratios of integers).
const EPS: f64 = 1e-12;

/// Games (with at least one step) and steps.
#[derive(Serialize, Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Count {
    pub games: usize,
    pub steps: u64,
}

/// Source -> enemy race (`T` / `P` / `Z`, `?` for a matchup without one) -> count.
pub type SourceRaceCounts = BTreeMap<String, BTreeMap<String, Count>>;

fn race_key(matchup: &str) -> String {
    enemy_race(matchup).unwrap_or("?").to_string()
}

/// Per source and enemy race: the games (with steps, as `EpochCount::games`) and steps of `games`.
pub fn count_by_source(games: &[GameSummary]) -> SourceRaceCounts {
    let mut out = SourceRaceCounts::new();
    for g in games.iter().filter(|g| g.steps > 0) {
        let c = out.entry(g.meta.src.clone()).or_default().entry(race_key(&g.meta.matchup)).or_default();
        c.games += 1;
        c.steps += g.steps as u64;
    }
    out
}

/// Steps per enemy race of the sources not in `teacher`.
fn human_race_steps(c: &SourceRaceCounts, teacher: &[String]) -> BTreeMap<String, u64> {
    let mut out = BTreeMap::new();
    for (_, races) in c.iter().filter(|(s, _)| !teacher.contains(s)) {
        for (r, n) in races {
            *out.entry(r.clone()).or_default() += n.steps;
        }
    }
    out
}

fn shares(steps: &BTreeMap<String, u64>) -> BTreeMap<String, f64> {
    let total: u64 = steps.values().sum();
    steps.iter().map(|(r, &n)| (r.clone(), if total > 0 { n as f64 / total as f64 } else { 0.0 })).collect()
}

/// 041-R21 ⑥ / R26 ③: an epoch's mix shares against the targets.
#[derive(Serialize, Debug, Clone, PartialEq)]
pub struct MixShares {
    /// The teacher sources (`keep_sources`: the 021-R8 teacher `botgen`, and `dagger` in a DAgger
    /// round).
    pub teacher_sources: Vec<String>,
    pub teacher_steps: u64,
    pub human_steps: u64,
    /// teacher_steps / all steps; 041-R21 ⑥ wants 50 ± 3%.
    pub teacher_share: f64,
    pub human_race_steps: BTreeMap<String, u64>,
    /// Share of each enemy race inside the human steps.
    pub human_race_share: BTreeMap<String, f64>,
    /// The same shares in the reference (the S2 human data without the subsample).
    pub reference_human_race_share: BTreeMap<String, f64>,
    /// |teacher_share - 0.5| <= 0.03.
    pub teacher_ok: bool,
    /// Every race's |share - reference share| <= 0.03 (041-R26, AMB-041-15).
    pub race_ok: bool,
}

impl MixShares {
    /// `epoch`: the epoch's counts; `reference`: the counts of the S2 selection without the
    /// subsample; `teacher`: the `keep_sources`.
    pub fn new(epoch: &SourceRaceCounts, reference: &SourceRaceCounts, teacher: &[String]) -> Self {
        let total: u64 = epoch.values().flat_map(|r| r.values()).map(|c| c.steps).sum();
        let teacher_steps: u64 = epoch.iter().filter(|(s, _)| teacher.contains(s)).flat_map(|(_, r)| r.values()).map(|c| c.steps).sum();
        let reference_human_race_share = shares(&human_race_steps(reference, teacher));
        let human_race_steps = human_race_steps(epoch, teacher);
        let human_race_share = shares(&human_race_steps);
        let teacher_share = if total > 0 { teacher_steps as f64 / total as f64 } else { 0.0 };
        let race_ok = human_race_share.keys().chain(reference_human_race_share.keys()).all(|r| {
            let (a, b) = (human_race_share.get(r).copied().unwrap_or(0.0), reference_human_race_share.get(r).copied().unwrap_or(0.0));
            (a - b).abs() <= RACE_TOLERANCE + EPS
        });
        Self {
            teacher_sources: teacher.to_vec(),
            teacher_steps,
            human_steps: total - teacher_steps,
            teacher_share,
            human_race_steps,
            human_race_share,
            reference_human_race_share,
            teacher_ok: (teacher_share - TEACHER_SHARE).abs() <= TEACHER_TOLERANCE + EPS,
            race_ok,
        }
    }
}

/// The shortest decimal `x` with `lo < x <= hi` (`hi` itself when none is shorter), so a fraction
/// written to a config keeps exactly the games below it.
fn shortest_in(lo: f64, hi: f64) -> f64 {
    for d in 1..=17 {
        let p = 10f64.powi(d);
        let x = (hi * p).floor() / p;
        if x > lo && x <= hi {
            return x;
        }
    }
    hi
}

/// 041-R26 ② (AMB-041-15): per-race human fractions for one data seed.
#[derive(Serialize, Debug, Clone, PartialEq)]
pub struct RaceFractions {
    pub seed: u64,
    pub teacher_sources: Vec<String>,
    /// The teacher share the fractions aim at (041-R21 ⑥: 0.5).
    pub teacher_share_target: f64,
    /// Steps of the teacher games (kept whole) in the selection.
    pub teacher_steps: u64,
    /// The selection's human steps per enemy race (the reference of the shares) and their shares.
    pub reference_human_race_steps: BTreeMap<String, u64>,
    pub reference_human_race_share: BTreeMap<String, f64>,
    /// Human steps per race to keep: reference share x teacher_steps x (1 - target) / target.
    pub target_human_race_steps: BTreeMap<String, u64>,
    /// The steps the fractions keep per race (the nearest the per-game hash allows).
    pub kept_human_race_steps: BTreeMap<String, u64>,
    /// One fraction for every human game that would aim at the same total (`[data.subsample]
    /// fraction`; with every race in `fraction_by_race` it only applies to a matchup without a
    /// T / P / Z enemy).
    pub fraction: f64,
    pub fraction_by_race: BTreeMap<String, f64>,
}

/// 041-R26 ② (AMB-041-15): for the games of a stage selection (`games`, e.g. S2), the
/// `fraction_by_race` whose subsample with `seed` keeps human steps closest to `teacher_share`
/// of the epoch with every race at its reference share. Per race the games are taken in the
/// order of their keep number (`game_unit`, the `Subsample::keeps` rule; both perspectives of a
/// TvT game share it), and the fraction is the shortest decimal that keeps exactly the prefix
/// whose steps are nearest the race's target.
pub fn race_fractions(games: &[GameSummary], teacher: &[String], seed: u64, teacher_share: f64) -> Result<RaceFractions> {
    ensure!(teacher_share > 0.0 && teacher_share < 1.0, "teacher share must be in (0, 1), got {teacher_share}");
    ensure!(!teacher.is_empty(), "no teacher sources (`[data.subsample] keep_sources`, 041-R21 ⑥)");
    let teacher_steps: u64 = games.iter().filter(|g| teacher.contains(&g.meta.src)).map(|g| g.steps as u64).sum();
    ensure!(teacher_steps > 0, "the selection has no steps of the teacher sources {teacher:?}");
    let mut by_race: BTreeMap<String, Vec<(f64, u64)>> = BTreeMap::new();
    for g in games.iter().filter(|g| !teacher.contains(&g.meta.src) && g.steps > 0) {
        by_race.entry(race_key(&g.meta.matchup)).or_default().push((game_unit(&g.meta.replay_sha256, seed, SUBSAMPLE_SALT), g.steps as u64));
    }
    let reference: BTreeMap<String, u64> = by_race.iter().map(|(r, v)| (r.clone(), v.iter().map(|x| x.1).sum())).collect();
    let human: u64 = reference.values().sum();
    ensure!(human > 0, "the selection has no human steps");
    let want_total = teacher_steps as f64 * (1.0 - teacher_share) / teacher_share;
    // Four decimals (at least 1e-4): only a matchup without a T / P / Z enemy uses it.
    let fraction = (((want_total / human as f64).min(1.0) * 1e4).round() / 1e4).max(1e-4);
    let (mut target, mut kept, mut fraction_by_race) = (BTreeMap::new(), BTreeMap::new(), BTreeMap::new());
    for (r, mut v) in by_race {
        if !SUBSAMPLE_RACES.contains(&r.as_str()) {
            continue; // kept by `fraction` (no `fraction_by_race` key for it)
        }
        let want = reference[&r] as f64 / human as f64 * want_total;
        v.sort_by(|a, b| a.0.total_cmp(&b.0));
        // Groups of equal keep numbers (the two perspectives of a TvT game) go together.
        let mut groups: Vec<(f64, u64)> = Vec::new();
        for (u, s) in v {
            match groups.last_mut() {
                Some(g) if g.0 == u => g.1 += s,
                _ => groups.push((u, s)),
            }
        }
        let (mut best_k, mut best_d, mut cum) = (0usize, want, 0u64);
        for (k, g) in groups.iter().enumerate() {
            cum += g.1;
            let d = (cum as f64 - want).abs();
            if d < best_d {
                (best_k, best_d) = (k + 1, d);
            }
        }
        let lo = if best_k == 0 { 0.0 } else { groups[best_k - 1].0 };
        let hi = groups.get(best_k).map_or(1.0, |g| g.0);
        fraction_by_race.insert(r.clone(), shortest_in(lo, hi));
        target.insert(r.clone(), want.round() as u64);
        kept.insert(r, groups[..best_k].iter().map(|g| g.1).sum());
    }
    let reference_human_race_share = shares(&reference);
    Ok(RaceFractions {
        seed,
        teacher_sources: teacher.to_vec(),
        teacher_share_target: teacher_share,
        teacher_steps,
        reference_human_race_steps: reference,
        reference_human_race_share,
        target_human_race_steps: target,
        kept_human_race_steps: kept,
        fraction,
        fraction_by_race,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shortest_decimal_keeps_the_interval() {
        assert_eq!(shortest_in(0.0412, 0.0437), 0.043);
        assert_eq!(shortest_in(0.04, 0.05), 0.05);
        let x = shortest_in(0.12345, 0.123456);
        assert!(x > 0.12345 && x <= 0.123456 && format!("{x}").len() <= 8, "{x}");
        let x = shortest_in(0.3, 0.30000000001);
        assert!(x > 0.3 && x <= 0.30000000001);
        assert_eq!(shortest_in(0.9, 1.0), 1.0);
    }
}
