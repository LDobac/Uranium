//! 041-R21 ⑥ / 041-R26 (AMB-041-15): the teacher / human mix of a BC epoch. Counts the epoch's
//! steps per source and enemy race, checks the shares (teacher = the `keep_sources` = 50 ± 3% of
//! the update steps; inside the human part the enemy-race shares of the S2 human data ± 3%p), and
//! computes the per-race human fractions (`[data.subsample] fraction_by_race`) that hit them for a
//! data seed. Every step of an epoch is trained exactly once (041-R13 lane walk), so step counts
//! are update-step counts.
//!
//! 044-R5 ③ / 041-R31: a distillation round has three shares — teacher (+ dagger) = the
//! `keep_sources`, the `fraction_by_source` sources (`distill`), and the human rest — each against
//! its own target ± 3%p, with the 041-R26 race rule inside the human part ([`share_fractions`],
//! [`MixShares::with_shares`]). Without `fraction_by_source` everything is the two-share mix above.

use std::collections::BTreeMap;

use anyhow::{Result, ensure};
use serde::Serialize;

use crate::data::{GameSummary, SUBSAMPLE_SALT, game_unit};
use crate::split::{SUBSAMPLE_RACES, enemy_race};

/// 041-R21 ⑥: the teacher share of the mix epoch's update steps, and its tolerance (also the
/// tolerance around a DAgger round's `--teacher-share`, 041-R22 ③).
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

/// Steps per enemy race of the sources in neither `teacher` nor `others` (the human sources).
fn human_race_steps(c: &SourceRaceCounts, teacher: &[String], others: &[String]) -> BTreeMap<String, u64> {
    let mut out = BTreeMap::new();
    for (_, races) in c.iter().filter(|(s, _)| !teacher.contains(s) && !others.contains(s)) {
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
    /// The teacher share `teacher_ok` is judged against: 0.5 (041-R21 ⑥ mix epoch), or the
    /// `mix-fractions --teacher-share` of a DAgger round (041-R22 ③, R29 ④; [`Self::with_target`]).
    pub teacher_share_target: f64,
    /// |teacher_share - teacher_share_target| <= 0.03.
    pub teacher_ok: bool,
    /// Every race's |share - reference share| <= 0.03 (041-R26, AMB-041-15).
    pub race_ok: bool,
    /// 044-R5 ③: the `fraction_by_source` sources (neither teacher nor human: `distill`) and their
    /// shares; left out of the JSON when there are none (the two-share mix output unchanged).
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub sources: BTreeMap<String, ShareCheck>,
    /// 044-R5 ③: the human share against its target (1 - the teacher and source targets), set by
    /// [`Self::with_shares`]; absent otherwise.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub human: Option<ShareCheck>,
}

/// 044-R5 ③: one share of a three-share epoch: its steps, its share of all steps and, once judged
/// ([`MixShares::with_shares`]), the target and |share - target| <= 0.03.
#[derive(Serialize, Debug, Clone, PartialEq)]
pub struct ShareCheck {
    pub steps: u64,
    pub share: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ok: Option<bool>,
}

impl ShareCheck {
    fn judged(mut self, target: f64) -> Self {
        self.target = Some(target);
        self.ok = Some(teacher_share_ok(self.share, target));
        self
    }
}

impl MixShares {
    /// `epoch`: the epoch's counts; `reference`: the counts of the S2 selection without the
    /// subsample; `teacher`: the `keep_sources`.
    pub fn new(epoch: &SourceRaceCounts, reference: &SourceRaceCounts, teacher: &[String]) -> Self {
        Self::with_sources(epoch, reference, teacher, &[])
    }

    /// [`Self::new`] with `sources` (the `fraction_by_source` keys, 044-R5) counted apart from both
    /// the teacher and the human part; with none it is [`Self::new`].
    pub fn with_sources(epoch: &SourceRaceCounts, reference: &SourceRaceCounts, teacher: &[String], sources: &[String]) -> Self {
        let total: u64 = epoch.values().flat_map(|r| r.values()).map(|c| c.steps).sum();
        let share_of = |n: u64| if total > 0 { n as f64 / total as f64 } else { 0.0 };
        let teacher_steps: u64 = epoch.iter().filter(|(s, _)| teacher.contains(s)).flat_map(|(_, r)| r.values()).map(|c| c.steps).sum();
        let source_checks: BTreeMap<String, ShareCheck> = sources
            .iter()
            .map(|s| {
                let steps: u64 = epoch.get(s).map_or(0, |r| r.values().map(|c| c.steps).sum());
                (s.clone(), ShareCheck { steps, share: share_of(steps), target: None, ok: None })
            })
            .collect();
        let source_steps: u64 = source_checks.values().map(|c| c.steps).sum();
        let reference_human_race_share = shares(&human_race_steps(reference, teacher, sources));
        let human_race_steps = human_race_steps(epoch, teacher, sources);
        let human_race_share = shares(&human_race_steps);
        let teacher_share = if total > 0 { teacher_steps as f64 / total as f64 } else { 0.0 };
        let race_ok = human_race_share.keys().chain(reference_human_race_share.keys()).all(|r| {
            let (a, b) = (human_race_share.get(r).copied().unwrap_or(0.0), reference_human_race_share.get(r).copied().unwrap_or(0.0));
            (a - b).abs() <= RACE_TOLERANCE + EPS
        });
        Self {
            teacher_sources: teacher.to_vec(),
            teacher_steps,
            human_steps: total - teacher_steps - source_steps,
            teacher_share,
            human_race_steps,
            human_race_share,
            reference_human_race_share,
            teacher_share_target: TEACHER_SHARE,
            teacher_ok: teacher_share_ok(teacher_share, TEACHER_SHARE),
            race_ok,
            sources: source_checks,
            human: None,
        }
    }

    /// The same shares with `teacher_ok` judged against `target` (same ±3% tolerance): a DAgger
    /// round's teacher + dagger share (041-R22 ③, 041-R29 ④), not the mix epoch's 50%.
    pub fn with_target(mut self, target: f64) -> Self {
        self.teacher_share_target = target;
        self.teacher_ok = teacher_share_ok(self.teacher_share, target);
        self
    }

    /// 044-R5 ③: the three shares judged (each ± 3%p): the teacher against `teacher`, each source
    /// of `sources` against its target (a source the epoch has no steps of is share 0) and the
    /// human part against 1 - `teacher` - the source targets. The sources must be the ones the
    /// shares were counted with ([`Self::with_sources`]): a target for a source counted as human
    /// would judge nothing.
    pub fn with_shares(self, teacher: f64, sources: &BTreeMap<String, f64>) -> Result<Self> {
        let counted: Vec<&String> = self.sources.keys().collect();
        ensure!(counted == sources.keys().collect::<Vec<_>>(), "share targets for {:?}, counted sources {counted:?} (044-R5)", sources.keys().collect::<Vec<_>>());
        let mut out = self.with_target(teacher);
        for (s, &t) in sources {
            let c = out.sources.remove(s).expect("checked above");
            out.sources.insert(s.clone(), c.judged(t));
        }
        let total = out.teacher_steps + out.human_steps + out.sources.values().map(|c| c.steps).sum::<u64>();
        let share = if total > 0 { out.human_steps as f64 / total as f64 } else { 0.0 };
        out.human = Some(ShareCheck { steps: out.human_steps, share, target: None, ok: None }.judged(1.0 - teacher - sources.values().sum::<f64>()));
        Ok(out)
    }

    /// Every judged share within its bound: the teacher, the human races and, when judged
    /// ([`Self::with_shares`]), every source and the human share.
    pub fn all_ok(&self) -> bool {
        self.teacher_ok && self.race_ok && self.sources.values().chain(self.human.as_ref()).all(|c| c.ok != Some(false))
    }
}

fn teacher_share_ok(share: f64, target: f64) -> bool {
    (share - target).abs() <= TEACHER_TOLERANCE + EPS
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
    /// Human steps per race to keep: reference share x teacher_steps x (1 - target) / target
    /// (044-R5 three shares: x the human target / the teacher target instead of (1 - target) / target).
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
    let teacher_steps = teacher_steps_of(games, teacher)?;
    let want_total = teacher_steps as f64 * (1.0 - teacher_share) / teacher_share;
    human_race_fractions(games, teacher, &[], seed, teacher_share, teacher_steps, want_total)
}

/// Steps of the teacher games (kept whole) in `games`; an error without any.
fn teacher_steps_of(games: &[GameSummary], teacher: &[String]) -> Result<u64> {
    ensure!(!teacher.is_empty(), "no teacher sources (`[data.subsample] keep_sources`, 041-R21 ⑥)");
    let teacher_steps: u64 = games.iter().filter(|g| teacher.contains(&g.meta.src)).map(|g| g.steps as u64).sum();
    ensure!(teacher_steps > 0, "the selection has no steps of the teacher sources {teacher:?}");
    Ok(teacher_steps)
}

/// The games of keep numbers below a fraction, as (keep number, steps) of the games of one race
/// or source: the shortest decimal fraction keeping exactly the prefix (in keep-number order)
/// whose steps are nearest `want`, and the steps it keeps. Games of equal keep numbers (the two
/// perspectives of a TvT game) go together.
fn prefix_fraction(mut v: Vec<(f64, u64)>, want: f64) -> (f64, u64) {
    v.sort_by(|a, b| a.0.total_cmp(&b.0));
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
    (shortest_in(lo, hi), groups[..best_k].iter().map(|g| g.1).sum())
}

/// The human part of [`race_fractions`] / [`share_fractions`]: human = the games in neither
/// `teacher` nor `others`, `want_total` human steps split by the races' reference shares.
fn human_race_fractions(games: &[GameSummary], teacher: &[String], others: &[String], seed: u64, teacher_share: f64, teacher_steps: u64, want_total: f64) -> Result<RaceFractions> {
    let mut by_race: BTreeMap<String, Vec<(f64, u64)>> = BTreeMap::new();
    for g in games.iter().filter(|g| !teacher.contains(&g.meta.src) && !others.contains(&g.meta.src) && g.steps > 0) {
        by_race.entry(race_key(&g.meta.matchup)).or_default().push((game_unit(&g.meta.replay_sha256, seed, SUBSAMPLE_SALT), g.steps as u64));
    }
    let reference: BTreeMap<String, u64> = by_race.iter().map(|(r, v)| (r.clone(), v.iter().map(|x| x.1).sum())).collect();
    let human: u64 = reference.values().sum();
    ensure!(human > 0, "the selection has no human steps");
    // Four decimals (at least 1e-4): only a matchup without a T / P / Z enemy uses it.
    let fraction = (((want_total / human as f64).min(1.0) * 1e4).round() / 1e4).max(1e-4);
    let (mut target, mut kept, mut fraction_by_race) = (BTreeMap::new(), BTreeMap::new(), BTreeMap::new());
    for (r, v) in by_race {
        if !SUBSAMPLE_RACES.contains(&r.as_str()) {
            continue; // kept by `fraction` (no `fraction_by_race` key for it)
        }
        let want = reference[&r] as f64 / human as f64 * want_total;
        let (f, k) = prefix_fraction(v, want);
        fraction_by_race.insert(r.clone(), f);
        target.insert(r.clone(), want.round() as u64);
        kept.insert(r, k);
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

/// 044-R5 ③: the fractions of the non-human, non-teacher sources (`[data.subsample]
/// fraction_by_source`) for one data seed, beside the human [`RaceFractions`].
#[derive(Serialize, Debug, Clone, PartialEq)]
pub struct SourceFractions {
    pub seed: u64,
    pub teacher_share_target: f64,
    /// Target share of each source (`--share <source>=<b>`).
    pub source_share_target: BTreeMap<String, f64>,
    /// 1 - the teacher and source targets.
    pub human_share_target: f64,
    /// Steps of the teacher games (all kept) and the epoch the targets are shares of
    /// (teacher_steps / teacher target).
    pub teacher_steps: u64,
    pub target_total_steps: u64,
    /// Per source: the selection's steps, the steps to keep (target share x the epoch) and the
    /// steps the fraction keeps (the nearest the per-game hash allows; all of them when the source
    /// has fewer steps than its target, which the share check then reports).
    pub reference_source_steps: BTreeMap<String, u64>,
    /// Per source: the selection's games (with steps).
    pub reference_source_games: BTreeMap<String, usize>,
    pub target_source_steps: BTreeMap<String, u64>,
    pub kept_source_steps: BTreeMap<String, u64>,
    pub fraction_by_source: BTreeMap<String, f64>,
    /// 044-R5 ③ top-up (user decision 2026-11-14): the sources whose selection has fewer steps
    /// than their target, set by [`Self::with_shortfalls`]; left out of the JSON when none is short.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub short: BTreeMap<String, Shortfall>,
}

/// 044-R5 ③ top-up (user decision 2026-11-14): how far a source (`distill`) is below its target
/// and about how many more games of it would fill it. More games of the source do not move the
/// target (the epoch is the teacher steps / the teacher target).
#[derive(Serialize, Debug, Clone, PartialEq)]
pub struct Shortfall {
    /// Target steps - the selection's steps (all of them are kept).
    pub missing_steps: u64,
    /// The selection's steps per game (the counted games: train split, target maps).
    pub steps_per_game: f64,
    /// The source's games in the selection and in the whole scan (every split): their ratio is
    /// the share of a new game that lands in the selection (041-R12 split by replay hash).
    pub selected_games: usize,
    pub scanned_games: usize,
    pub selected_share: f64,
    /// ceil(missing_steps / steps_per_game / selected_share): games to play beyond those built.
    /// Games the build filters (044-R2 `filtered`) are not in the scan, so add the build's
    /// filtered share on top.
    pub extra_games: u64,
}

impl SourceFractions {
    /// [`Self::short`] for the sources whose selection misses its target; `all`: every game of the
    /// scan (all splits), for the selected share.
    pub fn with_shortfalls(mut self, all: &[GameSummary]) -> Self {
        self.short.clear();
        for (s, &target) in &self.target_source_steps {
            let have = self.reference_source_steps[s];
            if have >= target {
                continue;
            }
            let selected_games = self.reference_source_games[s];
            let scanned_games = all.iter().filter(|g| g.meta.src == *s && g.steps > 0).count().max(selected_games);
            let steps_per_game = have as f64 / selected_games as f64;
            let selected_share = selected_games as f64 / scanned_games as f64;
            let missing_steps = target - have;
            let extra_games = (missing_steps as f64 / steps_per_game / selected_share).ceil() as u64;
            self.short.insert(s.clone(), Shortfall { missing_steps, steps_per_game, selected_games, scanned_games, selected_share, extra_games });
        }
        self
    }
}

/// 044-R5 ③ (041-R31): the subsample of a three-share epoch for one data seed. The teacher games
/// (`teacher` = `keep_sources`) are kept whole (044-R5 ③ "교사 + dagger는 전부 쓰므로"), so the
/// epoch is teacher_steps / `teacher_share`; each source of `sources` (`distill`) keeps the
/// game prefix nearest its target share of that epoch and the human games keep 1 - the targets of
/// it per race at the S2 human race shares (041-R26), by the [`race_fractions`] rule. Every
/// share is then judged by [`MixShares::with_shares`].
pub fn share_fractions(games: &[GameSummary], teacher: &[String], sources: &BTreeMap<String, f64>, seed: u64, teacher_share: f64) -> Result<(RaceFractions, SourceFractions)> {
    ensure!(!sources.is_empty(), "no source shares (044-R5: e.g. distill)");
    let human_share = 1.0 - teacher_share - sources.values().sum::<f64>();
    // 1e-9: shares summing to 1 up to float rounding leave no human rest.
    ensure!(teacher_share > 0.0 && sources.values().all(|&b| b > 0.0) && human_share > 1e-9, "shares must be > 0 with a human rest > 0: teacher {teacher_share}, sources {sources:?}, human {human_share}");
    for s in sources.keys() {
        ensure!(!teacher.contains(s), "share source {s:?} is a teacher source (`keep_sources`, 044-R5)");
    }
    let teacher_steps = teacher_steps_of(games, teacher)?;
    let total = teacher_steps as f64 / teacher_share;
    let others: Vec<String> = sources.keys().cloned().collect();
    let (mut reference, mut reference_games, mut target, mut kept, mut fraction_by_source) = (BTreeMap::new(), BTreeMap::new(), BTreeMap::new(), BTreeMap::new(), BTreeMap::new());
    for (s, &b) in sources {
        let v: Vec<(f64, u64)> = games.iter().filter(|g| g.meta.src == *s && g.steps > 0).map(|g| (game_unit(&g.meta.replay_sha256, seed, SUBSAMPLE_SALT), g.steps as u64)).collect();
        let have: u64 = v.iter().map(|x| x.1).sum();
        ensure!(have > 0, "the selection has no steps of the source {s:?} (044-R5)");
        let want = b * total;
        reference_games.insert(s.clone(), v.len());
        let (f, k) = prefix_fraction(v, want);
        reference.insert(s.clone(), have);
        target.insert(s.clone(), want.round() as u64);
        kept.insert(s.clone(), k);
        fraction_by_source.insert(s.clone(), f);
    }
    let race = human_race_fractions(games, teacher, &others, seed, teacher_share, teacher_steps, human_share * total)?;
    let src = SourceFractions {
        seed,
        teacher_share_target: teacher_share,
        source_share_target: sources.clone(),
        human_share_target: human_share,
        teacher_steps,
        target_total_steps: total.round() as u64,
        reference_source_steps: reference,
        reference_source_games: reference_games,
        target_source_steps: target,
        kept_source_steps: kept,
        fraction_by_source,
        short: BTreeMap::new(),
    };
    Ok((race, src))
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

    /// S2-like scan: teacher `botgen` + `dagger` TvZ, human games of three enemy races (TvT as two
    /// perspectives) and, with `distill` > 0, that many `distill` games (044-R1: TvP and TvT, the
    /// model's view only).
    fn scan(distill: u64) -> Vec<GameSummary> {
        let summary = |i: u64, src: &str, mu: &str, slot: u8, steps: usize| {
            let meta = uranium_replay::shard::GameMeta {
                map_hash: String::new(),
                map_family: "fighting_spirit".into(),
                matchup: mu.into(),
                player_race: 1,
                perspective_slot: slot,
                apm: 0,
                result: 0,
                result_source: "none".into(),
                enemy_random: false,
                src: src.into(),
                stage: "S2".into(),
                replay_sha256: format!("{:064x}", i.wrapping_mul(0x9E37_79B9_7F4A_7C15)),
            };
            GameSummary { meta, steps, privileged_none: 0 }
        };
        let mut v = Vec::new();
        for i in 0..300 {
            v.push(summary(i, if i % 4 == 0 { "dagger" } else { "botgen" }, "TvZ", 0, 400 + (i as usize * 37) % 300));
        }
        for i in 1_000..2_000 {
            let steps = 300 + (i as usize * 53) % 900;
            v.push(summary(i, "community", "TvT", 0, steps));
            v.push(summary(i, "community", "TvT", 1, steps + (i as usize % 7) * 10));
        }
        for i in 3_000..6_000 {
            v.push(summary(i, if i % 3 == 0 { "stardata" } else { "scr_ladder" }, "TvP", 0, 300 + (i as usize * 71) % 1_000));
        }
        for i in 7_000..8_000 {
            v.push(summary(i, "personal", "TvZ", 0, 250 + (i as usize * 29) % 800));
        }
        for i in 10_000..10_000 + distill {
            v.push(summary(i, "distill", if i % 5 < 3 { "TvP" } else { "TvT" }, 0, 200 + (i as usize * 43) % 900));
        }
        v
    }

    /// The two-share `race_fractions` before 044 (commit 72193d6), verbatim: the reference of
    /// "`--teacher-share` alone gives the same result" (044-AC3).
    fn race_fractions_before_044(games: &[GameSummary], teacher: &[String], seed: u64, teacher_share: f64) -> RaceFractions {
        let teacher_steps: u64 = games.iter().filter(|g| teacher.contains(&g.meta.src)).map(|g| g.steps as u64).sum();
        let mut by_race: BTreeMap<String, Vec<(f64, u64)>> = BTreeMap::new();
        for g in games.iter().filter(|g| !teacher.contains(&g.meta.src) && g.steps > 0) {
            by_race.entry(race_key(&g.meta.matchup)).or_default().push((game_unit(&g.meta.replay_sha256, seed, SUBSAMPLE_SALT), g.steps as u64));
        }
        let reference: BTreeMap<String, u64> = by_race.iter().map(|(r, v)| (r.clone(), v.iter().map(|x| x.1).sum())).collect();
        let human: u64 = reference.values().sum();
        let want_total = teacher_steps as f64 * (1.0 - teacher_share) / teacher_share;
        let fraction = (((want_total / human as f64).min(1.0) * 1e4).round() / 1e4).max(1e-4);
        let (mut target, mut kept, mut fraction_by_race) = (BTreeMap::new(), BTreeMap::new(), BTreeMap::new());
        for (r, mut v) in by_race {
            if !SUBSAMPLE_RACES.contains(&r.as_str()) {
                continue;
            }
            let want = reference[&r] as f64 / human as f64 * want_total;
            v.sort_by(|a, b| a.0.total_cmp(&b.0));
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
        RaceFractions {
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
        }
    }

    fn sub(race: &RaceFractions, teacher: &[String], by_source: Option<BTreeMap<String, f64>>) -> Subsample {
        Subsample { fraction: race.fraction, keep_sources: teacher.to_vec(), fraction_by_race: Some(race.fraction_by_race.clone()), fraction_by_source: by_source }
    }

    use crate::split::Subsample;

    /// 044-AC3: `--teacher-share` alone (no `--share`, no `fraction_by_source`) gives the fractions
    /// of the code before 044, and its mix judgement JSON has no three-share fields.
    #[test]
    fn teacher_share_alone_is_unchanged_044_ac3() {
        let teacher = vec!["botgen".to_string(), "dagger".to_string()];
        for distill in [0, 400] {
            let games = scan(distill);
            for seed in [3u64, 13, 15] {
                for share in [0.5, 0.5954] {
                    let f = race_fractions(&games, &teacher, seed, share).unwrap();
                    assert_eq!(f, race_fractions_before_044(&games, &teacher, seed, share), "seed {seed} share {share} distill {distill}");
                    let s = sub(&f, &teacher, None);
                    let kept: Vec<_> = games.iter().filter(|g| s.keeps(&g.meta, seed)).cloned().collect();
                    let all = count_by_source(&games);
                    let mix = MixShares::new(&count_by_source(&kept), &all, &teacher).with_target(share);
                    assert_eq!(mix, MixShares::with_sources(&count_by_source(&kept), &all, &teacher, &[]).with_target(share));
                    assert_eq!(mix.all_ok(), mix.teacher_ok && mix.race_ok);
                    assert_eq!(mix.human_steps + mix.teacher_steps, kept.iter().map(|g| g.steps as u64).sum::<u64>());
                    let json = serde_json::to_value(&mix).unwrap();
                    assert!(json.get("sources").is_none() && json.get("human").is_none(), "the two-share JSON: {json}");
                }
            }
        }
    }

    /// 044-AC3 / 044-R5 ③: three shares — teacher + dagger 45%, distill 20%, human 35% of the update
    /// steps, each within ± 3%p (here far inside), the human races at the S2 human shares ± 3%p
    /// (distill not counted as human), every teacher and dagger game kept, the result the same on
    /// every call; off mixes fail.
    #[test]
    fn three_shares_hit_the_targets_044_ac3() {
        let teacher = vec!["botgen".to_string(), "dagger".to_string()];
        let targets: BTreeMap<String, f64> = [("distill".to_string(), 0.20)].into();
        let games = scan(600);
        let all = count_by_source(&games);
        for seed in [15u64, 16, 3] {
            let (race, src) = share_fractions(&games, &teacher, &targets, seed, 0.45).unwrap();
            assert_eq!((race.clone(), src.clone()), share_fractions(&games, &teacher, &targets, seed, 0.45).unwrap(), "deterministic");
            assert!((src.human_share_target - 0.35).abs() < 1e-12);
            assert!(race.reference_human_race_steps.values().sum::<u64>() < games.iter().filter(|g| !teacher.contains(&g.meta.src)).map(|g| g.steps as u64).sum::<u64>(), "distill is not human");
            let s = sub(&race, &teacher, Some(src.fraction_by_source.clone()));
            s.check().unwrap();
            let kept: Vec<_> = games.iter().filter(|g| s.keeps(&g.meta, seed)).cloned().collect();
            let mix = MixShares::with_sources(&count_by_source(&kept), &all, &teacher, &s.share_sources()).with_shares(0.45, &targets).unwrap();
            assert!(mix.all_ok(), "seed {seed}: {mix:?}");
            assert!((mix.teacher_share - 0.45).abs() < 0.01, "seed {seed}: teacher {}", mix.teacher_share);
            let d = &mix.sources["distill"];
            assert!((d.share - 0.20).abs() < 0.01 && d.ok == Some(true) && d.target == Some(0.20), "seed {seed}: {d:?}");
            let h = mix.human.as_ref().unwrap();
            assert!((h.share - 0.35).abs() < 0.01 && h.ok == Some(true), "seed {seed}: {h:?}");
            assert_eq!(d.steps, src.kept_source_steps["distill"], "the fraction keeps what it says");
            assert_eq!(mix.human_race_steps, race.kept_human_race_steps);
            assert_eq!(mix.teacher_steps, src.teacher_steps, "every teacher + dagger game");
            for (r, x) in &mix.human_race_share {
                assert!((x - mix.reference_human_race_share[r]).abs() < 0.01, "seed {seed} {r}");
            }
            // The S2 human race reference leaves distill out (its TvP / TvT would shift it).
            assert_eq!(mix.reference_human_race_share, shares(&human_race_steps(&all, &teacher, &["distill".to_string()])));
            let json = serde_json::to_value(&mix).unwrap();
            assert!(json["sources"]["distill"]["ok"].as_bool().unwrap() && json["human"]["target"].is_f64());
        }
        // Too few distill games: all kept, the distill share is short and the judgement fails.
        let short = scan(60);
        let (race, src) = share_fractions(&short, &teacher, &targets, 15, 0.45).unwrap();
        assert_eq!(src.fraction_by_source["distill"], 1.0);
        assert_eq!(src.kept_source_steps["distill"], src.reference_source_steps["distill"]);
        // 044-R5 ③ top-up (user decision 2026-11-14): the missing steps and the extra games, with the
        // selected share of the scan (here 60 selected of 75 scanned distill games = 80%).
        assert!(src.short.is_empty() && serde_json::to_value(&src).unwrap().get("short").is_none(), "only with_shortfalls fills it");
        let mut scanned = short.clone();
        scanned.extend(scan(75).into_iter().filter(|g| g.meta.src == "distill").skip(60));
        let src = src.with_shortfalls(&scanned);
        let x = &src.short["distill"];
        let have = src.reference_source_steps["distill"];
        assert_eq!(x.missing_steps, src.target_source_steps["distill"] - have);
        assert_eq!((x.selected_games, x.scanned_games), (60, 75));
        assert!((x.steps_per_game - have as f64 / 60.0).abs() < 1e-9 && (x.selected_share - 0.8).abs() < 1e-12);
        assert_eq!(x.extra_games, (x.missing_steps as f64 / x.steps_per_game / 0.8).ceil() as u64);
        // Playing that many more games (same steps per game, 80% selected) reaches the target.
        assert!(have as f64 + x.extra_games as f64 * 0.8 * x.steps_per_game >= src.target_source_steps["distill"] as f64);
        let (_, full) = share_fractions(&games, &teacher, &targets, 15, 0.45).unwrap();
        assert!(full.with_shortfalls(&games).short.is_empty(), "no shortfall when the source covers its target");
        let s = sub(&race, &teacher, Some(src.fraction_by_source.clone()));
        let kept: Vec<_> = short.iter().filter(|g| s.keeps(&g.meta, 15)).cloned().collect();
        let mix = MixShares::with_sources(&count_by_source(&kept), &count_by_source(&short), &teacher, &s.share_sources()).with_shares(0.45, &targets).unwrap();
        assert!(!mix.all_ok() && mix.sources["distill"].ok == Some(false), "{mix:?}");
        // A distill fraction 3x too wide fails distill (and so the teacher share).
        let (race, src) = share_fractions(&games, &teacher, &targets, 15, 0.45).unwrap();
        let wide = sub(&race, &teacher, Some([("distill".to_string(), (src.fraction_by_source["distill"] * 3.0).min(1.0))].into()));
        let kept: Vec<_> = games.iter().filter(|g| wide.keeps(&g.meta, 15)).cloned().collect();
        let mix = MixShares::with_sources(&count_by_source(&kept), &all, &teacher, &wide.share_sources()).with_shares(0.45, &targets).unwrap();
        assert!(!mix.all_ok() && mix.sources["distill"].ok == Some(false));
        // Bad inputs: no distill games, shares without a human rest, a teacher source as a share
        // source, targets for sources the shares were not counted with.
        assert!(share_fractions(&scan(0), &teacher, &targets, 15, 0.45).is_err());
        assert!(share_fractions(&games, &teacher, &[("distill".to_string(), 0.6)].into(), 15, 0.45).is_err());
        assert!(share_fractions(&games, &teacher, &[("dagger".to_string(), 0.2)].into(), 15, 0.45).is_err());
        assert!(MixShares::new(&count_by_source(&kept), &all, &teacher).with_shares(0.45, &targets).is_err());
    }
}
