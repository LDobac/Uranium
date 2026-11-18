use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use uranium_env::actions::ActionTable;

use crate::branch::FRAMES_PER_SECOND;

pub const OP_GROUPS: &[&str] = &["build", "addon", "train", "research", "upgrade", "cancel"];
pub const OP_EXTRA: &[&str] = &["Lift", "Land_Pos", "Land_CommandCenter_Pos"];

pub fn op_actions(table: &ActionTable) -> Vec<u16> {
    let mut ids: Vec<u16> = table.actions.iter().filter(|a| OP_GROUPS.contains(&a.group.as_str()) || OP_EXTRA.contains(&a.name.as_str())).map(|a| a.id).collect();
    ids.sort_unstable();
    ids
}

fn d_points() -> u32 {
    3
}
fn d_window() -> [u32; 2] {
    [120, 540]
}
fn d_gap() -> u32 {
    60
}
fn d_k() -> u32 {
    4
}
fn d_epsilon() -> f64 {
    0.5
}
fn d_min_open() -> u32 {
    2
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ExploreConfig {
    #[serde(default = "d_points")]
    pub points: u32,
    #[serde(default = "d_window")]
    pub window_s: [u32; 2],
    #[serde(default = "d_gap")]
    pub gap_s: u32,
    #[serde(default = "d_k")]
    pub k: u32,
    #[serde(default = "d_epsilon")]
    pub epsilon: f64,
    #[serde(default = "d_min_open")]
    pub min_open: u32,
}

impl Default for ExploreConfig {
    fn default() -> Self {
        Self { points: d_points(), window_s: d_window(), gap_s: d_gap(), k: d_k(), epsilon: d_epsilon(), min_open: d_min_open() }
    }
}

impl ExploreConfig {
    pub fn window_frames(&self) -> Result<(u32, u32)> {
        let f = |s: u32| s.checked_mul(FRAMES_PER_SECOND).with_context(|| format!("explore: {s} s overflows the frame count (054-R2)"));
        Ok((f(self.window_s[0])?, f(self.window_s[1])?))
    }

    fn span(&self) -> Result<u32> {
        let (start, end) = self.window_frames()?;
        let gaps = (self.points - 1).checked_mul(self.gap_s).and_then(|s| s.checked_mul(FRAMES_PER_SECOND)).context("explore: gap_s overflows the frame count (054-R2)")?;
        let span = (end - start).checked_sub(gaps).filter(|s| *s > 0);
        span.with_context(|| format!("explore: {} points {} s apart do not fit the window {:?} s (054-R2)", self.points, self.gap_s, self.window_s))
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(self.points >= 1, "explore: points must be at least 1 (054-R2)");
        ensure!(self.window_s[0] < self.window_s[1], "explore: empty window {:?} s (054-R2)", self.window_s);
        self.span()?;
        ensure!(self.k >= 1, "explore: k must be at least 1 (054-R3)");
        ensure!(self.epsilon.is_finite() && (0.0..=1.0).contains(&self.epsilon), "explore: epsilon {} is not in [0, 1] (054-R3)", self.epsilon);
        ensure!(self.min_open >= 2, "explore: min_open must be at least 2, a0 plus one candidate (054-R2)");
        Ok(())
    }
}

const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;
const SCHEDULE_STREAM: u64 = 0x0054_0000;

fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

struct Draws(u64);

impl Draws {
    fn new(seed: u32, stream: u64) -> Self {
        Self(mix(((u64::from(seed) << 32) ^ stream).wrapping_add(GOLDEN)))
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(GOLDEN);
        mix(self.0)
    }

    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn below(&mut self, n: u32) -> u32 {
        (((self.next() >> 32) * u64::from(n)) >> 32) as u32
    }
}

pub fn schedule(cfg: &ExploreConfig, seed: u32) -> Result<Vec<u32>> {
    cfg.validate()?;
    let (start, _) = cfg.window_frames()?;
    let span = cfg.span()?;
    let gap = cfg.gap_s * FRAMES_PER_SECOND;
    let mut d = Draws::new(seed, SCHEDULE_STREAM);
    let mut u: Vec<u32> = (0..cfg.points).map(|_| d.below(span)).collect();
    u.sort_unstable();
    Ok(u.iter().enumerate().map(|(i, x)| start + x + i as u32 * gap).collect())
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Coin {
    Forced,
    Control,
}

pub fn draw(cfg: &ExploreConfig, seed: u32, point: u32) -> (Coin, f64) {
    let mut d = Draws::new(seed, SCHEDULE_STREAM + u64::from(point));
    let coin = if d.unit() < cfg.epsilon { Coin::Forced } else { Coin::Control };
    (coin, d.unit())
}

pub fn candidates(type_logp: &[f32], type_mask: &[bool], ops: &[u16], a0: u16, k: usize) -> Vec<(u16, f32)> {
    let mut open: Vec<(u16, f32)> = ops.iter().copied().filter(|&a| a != a0 && type_mask.get(a as usize).copied().unwrap_or(false)).map(|a| (a, type_logp.get(a as usize).copied().unwrap_or(f32::NEG_INFINITY))).collect();
    open.sort_by(|x, y| y.1.total_cmp(&x.1).then(x.0.cmp(&y.0)));
    open.truncate(k);
    open.into_iter().map(|(a, lp)| (a, lp.exp())).collect()
}

pub fn policy_top(type_logp: &[f32], type_mask: &[bool]) -> Vec<(u16, f32)> {
    let mut open: Vec<(u16, f32)> = type_mask.iter().enumerate().filter(|(_, m)| **m).map(|(a, _)| (a as u16, type_logp.get(a).copied().unwrap_or(f32::NEG_INFINITY))).collect();
    open.sort_by(|x, y| y.1.total_cmp(&x.1).then(x.0.cmp(&y.0)));
    open.into_iter().map(|(a, lp)| (a, lp.exp())).collect()
}

pub fn force_mask(type_mask: &[bool], c: u16) -> Vec<bool> {
    debug_assert!(type_mask.get(c as usize).copied().unwrap_or(false), "054-R3: c is an open action");
    (0..type_mask.len()).map(|k| k == c as usize).collect()
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ExploreEntry {
    pub point: u32,
    pub tau: u32,
    pub frame: u32,
    pub epsilon: f64,
    pub k: u32,
    pub cands: Vec<(u16, f32)>,
    pub a0: u16,
    pub coin: Coin,
    pub c: Option<u16>,
    pub rejected: Option<bool>,
    pub propensity: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MissReason {
    WindowEnd,
    GameEnd,
    Aborted,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct ExploreMiss {
    pub point: u32,
    pub tau: u32,
    pub reason: MissReason,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct ExploreEnd {
    pub taus: Vec<u32>,
    pub missed: Vec<ExploreMiss>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ExploreSummary {
    pub seed: u32,
    pub epsilon: f64,
    pub k: u32,
    pub taus: Vec<u32>,
    pub entries: Vec<ExploreEntry>,
    pub missed: Vec<ExploreMiss>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ExploreGame {
    pub cfg: ExploreConfig,
    pub seed: u32,
}

#[derive(Debug, Clone)]
pub struct ExploreRun {
    cfg: ExploreConfig,
    seed: u32,
    ops: Vec<u16>,
    taus: Vec<u32>,
    fired: Vec<bool>,
    window_end: u32,
    last_frame: Option<u32>,
    entries: Vec<ExploreEntry>,
    missed: Vec<ExploreMiss>,
}

impl ExploreRun {
    pub fn new(game: &ExploreGame, table: &ActionTable) -> Result<Self> {
        let taus = schedule(&game.cfg, game.seed)?;
        let (_, window_end) = game.cfg.window_frames()?;
        Ok(Self { cfg: game.cfg.clone(), seed: game.seed, ops: op_actions(table), fired: vec![false; taus.len()], taus, window_end, last_frame: None, entries: vec![], missed: vec![] })
    }

    pub fn taus(&self) -> &[u32] {
        &self.taus
    }

    pub fn check(&mut self, frame: u32, type_mask: &[bool]) -> Option<usize> {
        self.last_frame = Some(frame);
        if frame >= self.window_end {
            return None;
        }
        let open = self.ops.iter().filter(|&&a| type_mask.get(a as usize).copied().unwrap_or(false)).count();
        if open < self.cfg.min_open as usize {
            return None;
        }
        let p = (0..self.taus.len()).find(|&i| !self.fired[i] && self.taus[i] <= frame)?;
        self.fired[p] = true;
        Some(p)
    }

    pub fn decide(&mut self, p: usize, frame: u32, a0: u16, type_logp: &[f32], type_mask: &[bool]) -> ExploreEntry {
        let point = p as u32 + 1;
        let cands = candidates(type_logp, type_mask, &self.ops, a0, self.cfg.k as usize);
        let (coin, pick) = draw(&self.cfg, self.seed, point);
        let (coin, c, propensity) = match (coin, cands.len()) {
            (Coin::Forced, n) if n > 0 => {
                let i = ((pick * n as f64) as usize).min(n - 1);
                (Coin::Forced, Some(cands[i].0), self.cfg.epsilon / n as f64)
            }
            _ => (Coin::Control, None, 1.0 - self.cfg.epsilon),
        };
        let e = ExploreEntry { point, tau: self.taus[p], frame, epsilon: self.cfg.epsilon, k: self.cfg.k, cands, a0, coin, c, rejected: None, propensity };
        self.entries.push(e.clone());
        e
    }

    pub fn note_rejected(&mut self, rejected: bool) {
        if let Some(e) = self.entries.last_mut() {
            e.rejected = Some(rejected);
        }
    }

    pub fn finish(&mut self, aborted: bool, end_frame: Option<u32>) {
        let closed = self.last_frame.max(end_frame).is_some_and(|f| f >= self.window_end);
        let reason = if aborted {
            MissReason::Aborted
        } else if closed {
            MissReason::WindowEnd
        } else {
            MissReason::GameEnd
        };
        self.missed = (0..self.taus.len()).filter(|&i| !self.fired[i]).map(|i| ExploreMiss { point: i as u32 + 1, tau: self.taus[i], reason }).collect();
    }

    pub fn end(&self) -> ExploreEnd {
        ExploreEnd { taus: self.taus.clone(), missed: self.missed.clone() }
    }

    pub fn summary(&self) -> ExploreSummary {
        ExploreSummary { seed: self.seed, epsilon: self.cfg.epsilon, k: self.cfg.k, taus: self.taus.clone(), entries: self.entries.clone(), missed: self.missed.clone() }
    }
}

#[cfg(test)]
mod tests;
