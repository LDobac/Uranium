use std::collections::BTreeMap;

use anyhow::{Result, ensure};
use serde::Serialize;

use crate::data::{GameSummary, SUBSAMPLE_SALT, game_unit};
use crate::split::{SUBSAMPLE_RACES, enemy_race};

pub const TEACHER_SHARE: f64 = 0.5;
pub const TEACHER_TOLERANCE: f64 = 0.03;
pub const RACE_TOLERANCE: f64 = 0.03;
const EPS: f64 = 1e-12;

#[derive(Serialize, Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Count {
    pub games: usize,
    pub steps: u64,
}

pub type SourceRaceCounts = BTreeMap<String, BTreeMap<String, Count>>;

fn race_key(matchup: &str) -> String {
    enemy_race(matchup).unwrap_or("?").to_string()
}

pub fn count_by_source(games: &[GameSummary]) -> SourceRaceCounts {
    let mut out = SourceRaceCounts::new();
    for g in games.iter().filter(|g| g.steps > 0) {
        let c = out.entry(g.meta.src.clone()).or_default().entry(race_key(&g.meta.matchup)).or_default();
        c.games += 1;
        c.steps += g.steps as u64;
    }
    out
}

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

#[derive(Serialize, Debug, Clone, PartialEq)]
pub struct MixShares {
    pub teacher_sources: Vec<String>,
    pub teacher_steps: u64,
    pub human_steps: u64,
    pub teacher_share: f64,
    pub human_race_steps: BTreeMap<String, u64>,
    pub human_race_share: BTreeMap<String, f64>,
    pub reference_human_race_share: BTreeMap<String, f64>,
    pub teacher_share_target: f64,
    pub teacher_ok: bool,
    pub race_ok: bool,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub sources: BTreeMap<String, ShareCheck>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub human: Option<ShareCheck>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub teacher_races: BTreeMap<String, ShareCheck>,
}

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
    pub fn new(epoch: &SourceRaceCounts, reference: &SourceRaceCounts, teacher: &[String]) -> Self {
        Self::with_sources(epoch, reference, teacher, &[])
    }

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
            teacher_races: BTreeMap::new(),
        }
    }

    pub fn with_target(mut self, target: f64) -> Self {
        self.teacher_share_target = target;
        self.teacher_ok = teacher_share_ok(self.teacher_share, target);
        self
    }

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

    pub fn with_teacher_races(mut self, epoch: &SourceRaceCounts, races: &BTreeMap<String, f64>) -> Self {
        let steps = teacher_race_steps(epoch, &self.teacher_sources);
        let total: u64 = steps.values().sum();
        self.teacher_races = steps
            .keys()
            .chain(races.keys())
            .map(|r| {
                let n = steps.get(r).copied().unwrap_or(0);
                let share = if total > 0 { n as f64 / total as f64 } else { 0.0 };
                (r.clone(), ShareCheck { steps: n, share, target: None, ok: None }.judged(races.get(r).copied().unwrap_or(0.0)))
            })
            .collect();
        self
    }

    pub fn all_ok(&self) -> bool {
        self.teacher_ok && self.race_ok && self.sources.values().chain(self.human.as_ref()).chain(self.teacher_races.values()).all(|c| c.ok != Some(false))
    }
}

fn teacher_race_steps(c: &SourceRaceCounts, teacher: &[String]) -> BTreeMap<String, u64> {
    let mut out = BTreeMap::new();
    for (_, races) in c.iter().filter(|(s, _)| teacher.contains(s)) {
        for (r, n) in races.iter().filter(|(_, n)| n.steps > 0) {
            *out.entry(r.clone()).or_default() += n.steps;
        }
    }
    out
}

fn teacher_share_ok(share: f64, target: f64) -> bool {
    (share - target).abs() <= TEACHER_TOLERANCE + EPS
}

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

#[derive(Serialize, Debug, Clone, PartialEq)]
pub struct RaceFractions {
    pub seed: u64,
    pub teacher_sources: Vec<String>,
    pub teacher_share_target: f64,
    pub teacher_steps: u64,
    pub reference_human_race_steps: BTreeMap<String, u64>,
    pub reference_human_race_share: BTreeMap<String, f64>,
    pub target_human_race_steps: BTreeMap<String, u64>,
    pub kept_human_race_steps: BTreeMap<String, u64>,
    pub fraction: f64,
    pub fraction_by_race: BTreeMap<String, f64>,
}

pub fn race_fractions(games: &[GameSummary], teacher: &[String], seed: u64, teacher_share: f64) -> Result<RaceFractions> {
    ensure!(teacher_share > 0.0 && teacher_share < 1.0, "teacher share must be in (0, 1), got {teacher_share}");
    let teacher_steps = teacher_steps_of(games, teacher)?;
    let want_total = teacher_steps as f64 * (1.0 - teacher_share) / teacher_share;
    human_race_fractions(games, teacher, &[], seed, teacher_share, teacher_steps, want_total)
}

fn teacher_steps_of(games: &[GameSummary], teacher: &[String]) -> Result<u64> {
    ensure!(!teacher.is_empty(), "no teacher sources (`[data.subsample] keep_sources`, 041-R21 ⑥)");
    let teacher_steps: u64 = games.iter().filter(|g| teacher.contains(&g.meta.src)).map(|g| g.steps as u64).sum();
    ensure!(teacher_steps > 0, "the selection has no steps of the teacher sources {teacher:?}");
    Ok(teacher_steps)
}

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

fn human_race_fractions(games: &[GameSummary], teacher: &[String], others: &[String], seed: u64, teacher_share: f64, teacher_steps: u64, want_total: f64) -> Result<RaceFractions> {
    let mut by_race: BTreeMap<String, Vec<(f64, u64)>> = BTreeMap::new();
    for g in games.iter().filter(|g| !teacher.contains(&g.meta.src) && !others.contains(&g.meta.src) && g.steps > 0) {
        by_race.entry(race_key(&g.meta.matchup)).or_default().push((game_unit(&g.meta.replay_sha256, seed, SUBSAMPLE_SALT), g.steps as u64));
    }
    let reference: BTreeMap<String, u64> = by_race.iter().map(|(r, v)| (r.clone(), v.iter().map(|x| x.1).sum())).collect();
    let human: u64 = reference.values().sum();
    ensure!(human > 0, "the selection has no human steps");
    let fraction = (((want_total / human as f64).min(1.0) * 1e4).round() / 1e4).max(1e-4);
    let (mut target, mut kept, mut fraction_by_race) = (BTreeMap::new(), BTreeMap::new(), BTreeMap::new());
    for (r, v) in by_race {
        if !SUBSAMPLE_RACES.contains(&r.as_str()) {
            continue;
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

#[derive(Serialize, Debug, Clone, PartialEq)]
pub struct SourceFractions {
    pub seed: u64,
    pub teacher_share_target: f64,
    pub source_share_target: BTreeMap<String, f64>,
    pub human_share_target: f64,
    pub teacher_steps: u64,
    pub target_total_steps: u64,
    pub reference_source_steps: BTreeMap<String, u64>,
    pub reference_source_games: BTreeMap<String, usize>,
    pub target_source_steps: BTreeMap<String, u64>,
    pub kept_source_steps: BTreeMap<String, u64>,
    pub fraction_by_source: BTreeMap<String, f64>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub short: BTreeMap<String, Shortfall>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub below_target: BTreeMap<String, Shortfall>,
}

#[derive(Serialize, Debug, Clone, PartialEq)]
pub struct Shortfall {
    pub missing_steps: u64,
    pub steps_per_game: f64,
    pub selected_games: usize,
    pub scanned_games: usize,
    pub selected_share: f64,
    pub extra_games: u64,
}

pub const TP_BOTGEN_RACES: [&str; 2] = ["P", "T"];
pub const TP_BOTGEN_RACE_UNIT: u64 = 16;
pub const TP_BOTGEN_EXTRA_MAX: u64 = 992;

#[derive(Serialize, Debug, Clone, PartialEq)]
pub struct TopUpCheck {
    pub games: BTreeMap<String, u64>,
    pub parts: Vec<BTreeMap<String, u64>>,
    pub steps: BTreeMap<String, u64>,
    pub share: BTreeMap<String, f64>,
    pub ok: bool,
}

#[derive(Serialize, Debug, Clone, PartialEq)]
pub struct TeacherRaces {
    pub target_share: BTreeMap<String, f64>,
    pub steps: BTreeMap<String, u64>,
    pub steps_by_source: BTreeMap<String, BTreeMap<String, u64>>,
    pub anchor: Vec<String>,
    pub target_steps: BTreeMap<String, u64>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub short: BTreeMap<String, Shortfall>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub below_target: BTreeMap<String, Shortfall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra_botgen_tp: Option<BTreeMap<String, u64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub after_top_up: Option<TopUpCheck>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

impl TeacherRaces {
    pub fn new(selection: &[GameSummary], all: &[GameSummary], teacher: &[String], races: &BTreeMap<String, f64>, mix: &MixShares) -> Result<Self> {
        ensure!(!races.is_empty(), "no teacher race targets (044-R8)");
        for (r, &x) in races {
            ensure!(SUBSAMPLE_RACES.contains(&r.as_str()), "teacher race {r:?} must be one of {SUBSAMPLE_RACES:?} (044-R8)");
            ensure!(x > 0.0 && x < 1.0, "teacher race share {r} must be in (0, 1), got {x} (044-R8)");
        }
        let sum: f64 = races.values().sum();
        ensure!((sum - 1.0).abs() <= 1e-9, "teacher race shares sum to {sum}, not 1 (044-R8)");
        let is_teacher = |g: &&GameSummary| teacher.contains(&g.meta.src) && g.steps > 0;
        let mut steps_by_source: BTreeMap<String, BTreeMap<String, u64>> = BTreeMap::new();
        for g in selection.iter().filter(is_teacher) {
            let r = race_key(&g.meta.matchup);
            ensure!(races.contains_key(&r), "teacher source {} has steps against race {r:?} without a --teacher-races target (044-R8)", g.meta.src);
            *steps_by_source.entry(g.meta.src.clone()).or_default().entry(r).or_default() += g.steps as u64;
        }
        let mut steps: BTreeMap<String, u64> = races.keys().map(|r| (r.clone(), 0)).collect();
        for by in steps_by_source.values() {
            for (r, n) in by {
                *steps.get_mut(r).expect("checked above") += n;
            }
        }
        if !mix.teacher_races.is_empty() {
            for (r, n) in &steps {
                debug_assert_eq!(mix.teacher_races.get(r).map_or(0, |c| c.steps), *n, "teacher race {r}: selection vs epoch steps");
            }
        }
        let rate = |r: &str| -> Option<(usize, usize, f64)> {
            let of_race = |g: &&GameSummary, botgen_only: bool| is_teacher(g) && race_key(&g.meta.matchup) == r && (!botgen_only || g.meta.src == "botgen");
            let botgen_only = selection.iter().any(|g| of_race(&g, true));
            let sel: Vec<&GameSummary> = selection.iter().filter(|g| of_race(g, botgen_only)).collect();
            if sel.is_empty() {
                return None;
            }
            let scanned = all.iter().filter(|g| of_race(g, botgen_only)).count().max(sel.len());
            Some((sel.len(), scanned, sel.iter().map(|g| g.steps as u64).sum::<u64>() as f64 / sel.len() as f64))
        };
        let scale = races.iter().map(|(r, &x)| steps[r] as f64 / x).fold(0.0, f64::max);
        ensure!(scale > 0.0, "the selection has no teacher steps (044-R8)");
        let anchor: Vec<String> = races.iter().filter(|(r, x)| steps[*r] as f64 / **x >= scale * (1.0 - 1e-12)).map(|(r, _)| r.clone()).collect();
        let target_steps: BTreeMap<String, u64> = races.iter().map(|(r, &x)| (r.clone(), (x * scale).round() as u64)).collect();
        let (mut short, mut below_target) = (BTreeMap::new(), BTreeMap::new());
        for (r, &target) in &target_steps {
            let have = steps[r];
            if have >= target {
                continue;
            }
            let missing_steps = target - have;
            let x = match rate(r.as_str()) {
                None => Shortfall { missing_steps, steps_per_game: 0.0, selected_games: 0, scanned_games: 0, selected_share: 0.0, extra_games: 0 },
                Some((selected_games, scanned_games, steps_per_game)) => {
                    let selected_share = selected_games as f64 / scanned_games as f64;
                    let extra_games = (missing_steps as f64 / steps_per_game / selected_share).ceil() as u64;
                    Shortfall { missing_steps, steps_per_game, selected_games, scanned_games, selected_share, extra_games }
                }
            };
            let failed_low = mix.teacher_races.get(r).is_some_and(|c| c.ok == Some(false) && c.target.is_some_and(|t| c.share < t));
            if failed_low {
                short.insert(r.clone(), x);
            } else {
                below_target.insert(r.clone(), x);
            }
        }
        let simulate = |games: &BTreeMap<String, u64>| -> (BTreeMap<String, u64>, BTreeMap<String, f64>) {
            let mut after = steps.clone();
            for (r, &n) in games {
                if let (Some(v), Some((sel, scanned, spg))) = (after.get_mut(r), rate(r)) {
                    *v += (n as f64 * sel as f64 / scanned as f64 * spg).round() as u64;
                }
            }
            let total: u64 = after.values().sum();
            let share = after.iter().map(|(r, &v)| (r.clone(), v as f64 / total as f64)).collect();
            (after, share)
        };
        let games_of = |x: &Shortfall| x.extra_games.div_ceil(TP_BOTGEN_RACE_UNIT) * TP_BOTGEN_RACE_UNIT;
        let mut extra: BTreeMap<String, u64> = TP_BOTGEN_RACES.iter().filter_map(|r| short.get(*r).map(|x| (r.to_string(), games_of(x)))).filter(|(_, n)| *n > 0).collect();
        if !extra.is_empty() {
            for _ in 0..TP_BOTGEN_RACES.len() {
                let (_, share) = simulate(&extra);
                let late: Vec<(String, u64)> = TP_BOTGEN_RACES
                    .iter()
                    .filter(|r| !extra.contains_key(**r) && share[**r] < races[**r] && !teacher_share_ok(share[**r], races[**r]))
                    .filter_map(|r| below_target.get(*r).map(|x| (r.to_string(), games_of(x))))
                    .filter(|(_, n)| *n > 0)
                    .collect();
                if late.is_empty() {
                    break;
                }
                extra.extend(late);
            }
        }
        let extra_botgen_tp = (!extra.is_empty()).then_some(extra);
        let mut warnings = Vec::new();
        let after_top_up = extra_botgen_tp.as_ref().map(|games| {
            let (after, share) = simulate(games);
            let ok = races.iter().all(|(r, &x)| teacher_share_ok(share[r], x));
            let mut parts: Vec<BTreeMap<String, u64>> = Vec::new();
            let mut room = 0;
            for (r, &n) in games {
                let mut left = n;
                while left > 0 {
                    if room == 0 {
                        parts.push(BTreeMap::new());
                        room = TP_BOTGEN_EXTRA_MAX;
                    }
                    let k = left.min(room);
                    *parts.last_mut().expect("pushed above").entry(r.clone()).or_default() += k;
                    (left, room) = (left - k, room - k);
                }
            }
            TopUpCheck { games: games.clone(), parts, steps: after, share, ok }
        });
        let run = |g: &BTreeMap<String, u64>| -> String { g.iter().map(|(r, n)| format!("--extra-{} {n}", r.to_lowercase())).collect::<Vec<_>>().join(" ") };
        if let Some(t) = &after_top_up {
            if !t.ok {
                let shares: Vec<String> = t.share.iter().map(|(r, x)| format!("{r} {:.1}% (target {:.0}%)", x * 100.0, races[r] * 100.0)).collect();
                warnings.push(format!(
                    "after the recommended top-up ({}) the races would be {}: still outside +-3%p; play it and count again, or report (044-R8)",
                    run(&t.games),
                    shares.join(", ")
                ));
            }
            if t.parts.len() > 1 {
                let runs: Vec<String> = t.parts.iter().enumerate().map(|(k, g)| format!("{} --part K{}", run(g), if k == 0 { String::new() } else { format!("+{k}") })).collect();
                warnings.push(format!(
                    "the top-up ({} games) exceeds one --part ({} at most): play it as {} runs of rush_check_plan.py --botgen-tp ({}; K = the next unused part)",
                    t.games.values().sum::<u64>(),
                    TP_BOTGEN_EXTRA_MAX,
                    t.parts.len(),
                    runs.join(", ")
                ));
            }
        }
        if short.contains_key("Z") {
            warnings.push("the Zerg share is short: Protoss / Terran top-ups (022 botgen) cannot raise it (044-R8)".to_string());
        }
        Ok(Self { target_share: races.clone(), steps, steps_by_source, anchor, target_steps, short, below_target, extra_botgen_tp, after_top_up, warnings })
    }
}

impl SourceFractions {
    pub fn with_shortfalls(mut self, all: &[GameSummary], mix: &MixShares) -> Self {
        self.short.clear();
        self.below_target.clear();
        for (s, &target) in &self.target_source_steps {
            let have = self.reference_source_steps[s];
            if have >= target {
                continue;
            }
            let failed_low = mix.sources.get(s).is_some_and(|c| c.ok == Some(false) && c.target.is_some_and(|t| c.share < t));
            let selected_games = self.reference_source_games[s];
            let scanned_games = all.iter().filter(|g| g.meta.src == *s && g.steps > 0).count().max(selected_games);
            let steps_per_game = have as f64 / selected_games as f64;
            let selected_share = selected_games as f64 / scanned_games as f64;
            let missing_steps = target - have;
            let extra_games = (missing_steps as f64 / steps_per_game / selected_share).ceil() as u64;
            let x = Shortfall { missing_steps, steps_per_game, selected_games, scanned_games, selected_share, extra_games };
            if failed_low {
                self.short.insert(s.clone(), x);
            } else {
                self.below_target.insert(s.clone(), x);
            }
        }
        self
    }
}

pub fn share_fractions(games: &[GameSummary], teacher: &[String], sources: &BTreeMap<String, f64>, seed: u64, teacher_share: f64) -> Result<(RaceFractions, SourceFractions)> {
    ensure!(!sources.is_empty(), "no source shares (044-R5: e.g. distill)");
    let human_share = 1.0 - teacher_share - sources.values().sum::<f64>();
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
        below_target: BTreeMap::new(),
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

    fn summary(i: u64, src: &str, mu: &str, slot: u8, steps: usize) -> GameSummary {
        let meta = crate::shard::GameMeta {
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
    }

    fn scan(distill: u64) -> Vec<GameSummary> {
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
            assert_eq!(mix.reference_human_race_share, shares(&human_race_steps(&all, &teacher, &["distill".to_string()])));
            let json = serde_json::to_value(&mix).unwrap();
            assert!(json["sources"]["distill"]["ok"].as_bool().unwrap() && json["human"]["target"].is_f64());
        }
        let short = scan(60);
        let (race, src) = share_fractions(&short, &teacher, &targets, 15, 0.45).unwrap();
        assert_eq!(src.fraction_by_source["distill"], 1.0);
        assert_eq!(src.kept_source_steps["distill"], src.reference_source_steps["distill"]);
        assert!(src.short.is_empty() && serde_json::to_value(&src).unwrap().get("short").is_none(), "only with_shortfalls fills it");
        let s = sub(&race, &teacher, Some(src.fraction_by_source.clone()));
        let kept: Vec<_> = short.iter().filter(|g| s.keeps(&g.meta, 15)).cloned().collect();
        let mix = MixShares::with_sources(&count_by_source(&kept), &count_by_source(&short), &teacher, &s.share_sources()).with_shares(0.45, &targets).unwrap();
        assert!(!mix.all_ok() && mix.sources["distill"].ok == Some(false), "{mix:?}");
        let mut scanned = short.clone();
        scanned.extend(scan(75).into_iter().filter(|g| g.meta.src == "distill").skip(60));
        let mut in_band = mix.clone();
        in_band.sources.get_mut("distill").unwrap().share = 0.185;
        in_band.sources.get_mut("distill").unwrap().ok = Some(true);
        let info = src.clone().with_shortfalls(&scanned, &in_band);
        assert!(info.short.is_empty() && info.below_target.contains_key("distill"));
        assert!(serde_json::to_value(&info).unwrap().get("short").is_none(), "U11 keys on `short`");
        let src = src.with_shortfalls(&scanned, &mix);
        assert!(src.below_target.is_empty());
        assert_eq!(src.short["distill"], info.below_target["distill"], "the same numbers, only the key differs");
        let x = &src.short["distill"];
        let have = src.reference_source_steps["distill"];
        assert_eq!(x.missing_steps, src.target_source_steps["distill"] - have);
        assert_eq!((x.selected_games, x.scanned_games), (60, 75));
        assert!((x.steps_per_game - have as f64 / 60.0).abs() < 1e-9 && (x.selected_share - 0.8).abs() < 1e-12);
        assert_eq!(x.extra_games, (x.missing_steps as f64 / x.steps_per_game / 0.8).ceil() as u64);
        assert!(have as f64 + x.extra_games as f64 * 0.8 * x.steps_per_game >= src.target_source_steps["distill"] as f64);
        let (_, full) = share_fractions(&games, &teacher, &targets, 15, 0.45).unwrap();
        let full = full.with_shortfalls(&games, &mix);
        assert!(full.short.is_empty() && full.below_target.is_empty(), "no shortfall when the source covers its target");
        let (race, src) = share_fractions(&games, &teacher, &targets, 15, 0.45).unwrap();
        let wide = sub(&race, &teacher, Some([("distill".to_string(), (src.fraction_by_source["distill"] * 3.0).min(1.0))].into()));
        let kept: Vec<_> = games.iter().filter(|g| wide.keeps(&g.meta, 15)).cloned().collect();
        let mix = MixShares::with_sources(&count_by_source(&kept), &all, &teacher, &wide.share_sources()).with_shares(0.45, &targets).unwrap();
        assert!(!mix.all_ok() && mix.sources["distill"].ok == Some(false));
        assert!(share_fractions(&scan(0), &teacher, &targets, 15, 0.45).is_err());
        assert!(share_fractions(&games, &teacher, &[("distill".to_string(), 0.6)].into(), 15, 0.45).is_err());
        assert!(share_fractions(&games, &teacher, &[("dagger".to_string(), 0.2)].into(), 15, 0.45).is_err());
        assert!(MixShares::new(&count_by_source(&kept), &all, &teacher).with_shares(0.45, &targets).is_err());
    }

    fn scan_tp(p: u64, t: u64) -> Vec<GameSummary> {
        let mut v = scan(0);
        for i in 0..p {
            v.push(summary(20_000 + i, if i % 3 == 0 { "dagger" } else { "botgen" }, "TvP", 0, 300 + (i as usize * 61) % 600));
        }
        for i in 0..t {
            v.push(summary(30_000 + i, if i % 3 == 0 { "dagger" } else { "botgen" }, "TvT", 0, 300 + (i as usize * 47) % 600));
        }
        v
    }

    fn teacher_round(games: &[GameSummary], races: &BTreeMap<String, f64>) -> (MixShares, TeacherRaces) {
        let teacher = vec!["botgen".to_string(), "dagger".to_string()];
        let f = race_fractions(games, &teacher, 16, 0.65).unwrap();
        let s = sub(&f, &teacher, None);
        let kept: Vec<_> = games.iter().filter(|g| s.keeps(&g.meta, 16)).cloned().collect();
        let epoch = count_by_source(&kept);
        let mix = MixShares::new(&epoch, &count_by_source(games), &teacher).with_target(0.65).with_teacher_races(&epoch, races);
        let tr = TeacherRaces::new(games, games, &teacher, races, &mix).unwrap();
        (mix, tr)
    }

    #[test]
    fn teacher_races_report_shortfalls_044_r8() {
        let races: BTreeMap<String, f64> = [("Z".to_string(), 0.5), ("P".to_string(), 0.3), ("T".to_string(), 0.2)].into();
        let fit = scan_tp(165, 110);
        let (mix, tr) = teacher_round(&fit, &races);
        assert!(mix.all_ok(), "{mix:?}");
        assert!((mix.teacher_share - 0.65).abs() <= 0.03 && mix.teacher_ok);
        assert!(tr.short.is_empty() && tr.extra_botgen_tp.is_none(), "{tr:?}");
        assert_eq!(mix.teacher_steps, tr.steps.values().sum::<u64>(), "every teacher game kept");
        let json = serde_json::to_value(&tr).unwrap();
        assert!(json.get("short").is_none() && json.get("extra_botgen_tp").is_none() && json.get("after_top_up").is_none() && json.get("warnings").is_none());
        let few = scan_tp(40, 30);
        let (mix, tr) = teacher_round(&few, &races);
        assert!(!mix.all_ok() && mix.teacher_races["P"].ok == Some(false) && mix.teacher_races["T"].ok == Some(false));
        assert_eq!(tr.anchor, ["Z"]);
        assert_eq!(mix.teacher_steps, tr.steps.values().sum::<u64>(), "no teacher game dropped");
        assert_eq!(tr.target_steps["Z"], tr.steps["Z"]);
        for r in ["P", "T"] {
            let x = &tr.short[r];
            assert_eq!(x.missing_steps, tr.target_steps[r] - tr.steps[r]);
            assert_eq!(tr.target_steps[r], (tr.steps["Z"] as f64 / 0.5 * races[r]).round() as u64);
            assert!(x.selected_games > 0 && (x.selected_share - 1.0).abs() < 1e-12);
            assert_eq!(x.extra_games, (x.missing_steps as f64 / x.steps_per_game).ceil() as u64);
        }
        let extra = tr.extra_botgen_tp.as_ref().unwrap();
        assert_eq!(extra.keys().collect::<Vec<_>>(), ["P", "T"]);
        for r in ["P", "T"] {
            let n = extra[r];
            assert!(n % TP_BOTGEN_RACE_UNIT == 0 && n >= tr.short[r].extra_games && n < tr.short[r].extra_games + TP_BOTGEN_RACE_UNIT);
            assert!(tr.steps[r] as f64 + n as f64 * tr.short[r].steps_per_game >= tr.target_steps[r] as f64);
        }
        let t = tr.after_top_up.as_ref().unwrap();
        assert_eq!((&t.games, t.steps["Z"]), (extra, tr.steps["Z"]));
        for r in ["P", "T"] {
            let want = tr.steps[r] as f64 + extra[r] as f64 * tr.short[r].steps_per_game;
            assert!((t.steps[r] as f64 - want).abs() <= 1.0, "{r}");
        }
        assert!((t.share.values().sum::<f64>() - 1.0).abs() < 1e-9);
        assert!(t.ok, "{t:?}");
        assert_eq!(t.ok, t.share.iter().all(|(r, x)| (x - races[r]).abs() <= 0.03 + 1e-12));
        assert!(tr.warnings.is_empty(), "{:?}", tr.warnings);
        let skew = scan_tp(60, 70);
        let (mix, tr) = teacher_round(&skew, &races);
        let t = tr.after_top_up.as_ref().expect("Protoss short");
        assert!(mix.teacher_races["P"].ok == Some(false) && t.games.contains_key("P"));
        if mix.teacher_races["T"].ok == Some(true) {
            let (_, alone) = {
                let mut a = tr.steps.clone();
                *a.get_mut("P").unwrap() += (t.games["P"] as f64 * tr.short["P"].steps_per_game).round() as u64;
                let total: u64 = a.values().sum();
                (a.clone(), a.iter().map(|(r, &v)| (r.clone(), v as f64 / total as f64)).collect::<BTreeMap<String, f64>>())
            };
            assert_eq!(t.games.contains_key("T"), (alone["T"] - 0.2).abs() > 0.03 && alone["T"] < 0.2, "{t:?}");
            if t.games.contains_key("T") {
                assert_eq!(t.games["T"], tr.below_target["T"].extra_games.div_ceil(TP_BOTGEN_RACE_UNIT) * TP_BOTGEN_RACE_UNIT);
            }
        }
        assert!(t.ok && !tr.warnings.iter().any(|w| w.contains("still outside")), "{tr:?}");
        let over = scan_tp(60, 300);
        let (_, tr) = teacher_round(&over, &races);
        if let Some(t) = tr.after_top_up.as_ref() {
            assert!(!t.games.contains_key("T"), "{t:?}");
        }
        let zshort = scan_tp(1_000, 200);
        let (mix, tr) = teacher_round(&zshort, &races);
        assert!(mix.teacher_races["Z"].ok == Some(false) && tr.short.contains_key("Z") && tr.anchor == ["P"]);
        assert!(tr.warnings.iter().any(|w| w.contains("Zerg share is short")));
        let mut tiny = scan(0);
        for i in 0..2 {
            tiny.push(summary(40_000 + i, "botgen", "TvP", 0, 50));
            tiny.push(summary(41_000 + i, "botgen", "TvT", 0, 50));
        }
        let (_, tr) = teacher_round(&tiny, &races);
        let t = tr.after_top_up.as_ref().unwrap();
        let total: u64 = t.games.values().sum();
        assert!(total > TP_BOTGEN_EXTRA_MAX && t.parts.len() as u64 == total.div_ceil(TP_BOTGEN_EXTRA_MAX));
        assert!(t.parts.iter().all(|g| g.values().sum::<u64>() <= TP_BOTGEN_EXTRA_MAX && g.values().all(|n| n % TP_BOTGEN_RACE_UNIT == 0)));
        assert!(t.parts[..t.parts.len() - 1].iter().all(|g| g.values().sum::<u64>() == TP_BOTGEN_EXTRA_MAX), "every part but the last full");
        for r in ["P", "T"] {
            assert_eq!(t.parts.iter().map(|g| g.get(r).copied().unwrap_or(0)).sum::<u64>(), t.games[r]);
        }
        assert!(tr.warnings.iter().any(|w| w.contains("exceeds one --part") && w.contains("--extra-p 992 --part K,")), "{:?}", tr.warnings);
        let near = scan_tp(155, 110);
        let (mix, tr) = teacher_round(&near, &races);
        let p = &mix.teacher_races["P"];
        assert!(p.ok == Some(true) && p.share < 0.3, "{p:?}");
        assert!(tr.short.is_empty() && tr.below_target.contains_key("P") && tr.extra_botgen_tp.is_none());
        assert!(tr.after_top_up.is_none() && tr.warnings.is_empty());
        let teacher = vec!["botgen".to_string(), "dagger".to_string()];
        let f = race_fractions(&fit, &teacher, 16, 0.65).unwrap();
        let s = sub(&f, &teacher, None);
        let kept: Vec<_> = fit.iter().filter(|g| s.keeps(&g.meta, 16)).cloned().collect();
        assert!(serde_json::to_value(MixShares::new(&count_by_source(&kept), &count_by_source(&fit), &teacher)).unwrap().get("teacher_races").is_none());
        let bad = |r: &[(&str, f64)]| TeacherRaces::new(&fit, &fit, &teacher, &r.iter().map(|(k, x)| (k.to_string(), *x)).collect(), &mix).is_err();
        assert!(bad(&[("Z", 0.5), ("P", 0.3), ("T", 0.3)]));
        assert!(bad(&[("Z", 0.5), ("P", 0.3), ("R", 0.2)]));
        assert!(bad(&[("Z", 0.6), ("P", 0.4)]), "TvT teacher games without a target");
    }
}
