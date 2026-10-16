//! Reward (spec 033). Components are kept apart for logging and per-component value heads
//! (033-R3); weights live in a config file (033-R4). Pseudo-rewards are active only when the
//! episode has a strategy z (033-R2). Only observed destructions count (033-R5).

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use uranium_proto::{GameEvent, RawFrame, UnitId};

use crate::features::Strategy;
use crate::features::extract::dims;
use crate::types::TypeTables;

pub const REWARD_TOML: &str = include_str!("../../../config/reward.toml");

/// 033 weights (initial values; tuned, decayed by the league in M6).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct RewardWeights {
    pub win: f32,
    pub build: f32,
    pub units: f32,
    pub value: f32,
}

impl RewardWeights {
    pub fn load() -> anyhow::Result<Self> {
        Ok(toml::from_str(REWARD_TOML)?)
    }
}

/// One step's reward, unweighted per component (033-R3).
#[derive(Serialize, Debug, Clone, Copy, PartialEq, Default)]
pub struct RewardParts {
    pub win: f32,
    pub build: f32,
    pub units: f32,
    pub value: f32,
}

impl RewardParts {
    pub fn weighted(&self, w: &RewardWeights) -> f32 {
        w.win * self.win + w.build * self.build + w.units * self.units + w.value * self.value
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Win,
    Loss,
    /// Includes the 60-minute time limit (020, AMB-020-2).
    Draw,
}

/// Levenshtein distance between two action-id sequences.
pub fn levenshtein(a: &[u16], b: &[u16]) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, x) in a.iter().enumerate() {
        let mut cur = vec![i + 1; b.len() + 1];
        for (j, y) in b.iter().enumerate() {
            cur[j + 1] = (prev[j] + (x != y) as usize).min(prev[j + 1] + 1).min(cur[j] + 1);
        }
        prev = cur;
    }
    prev[b.len()]
}

/// Per-episode reward state.
#[derive(Debug, Clone)]
pub struct RewardTracker {
    z: Option<Strategy>,
    self_player: u8,
    /// Own Build / Train / Research / Upgrade actions so far (first 20 count, 030 §5).
    build: Vec<u16>,
    /// Own units created per terran type index (cumulative composition).
    created: Vec<f32>,
    /// Last known (owner, unit type) of every unit seen, for observed destructions.
    seen: HashMap<UnitId, (u8, u16)>,
    lost_value: f32,
    killed_value: f32,
    prev_build: f32,
    prev_units: f32,
    prev_value: f32,
}

impl RewardTracker {
    pub fn new(z: Option<Strategy>, self_player: u8) -> Self {
        let created = vec![0.0; dims().z_units()];
        let mut t = Self {
            z,
            self_player,
            build: vec![],
            created,
            seen: HashMap::new(),
            lost_value: 0.0,
            killed_value: 0.0,
            prev_build: 0.0,
            prev_units: 0.0,
            prev_value: 0.0,
        };
        t.prev_build = t.build_distance();
        t.prev_units = t.units_distance();
        t
    }

    fn build_distance(&self) -> f32 {
        self.z.as_ref().map_or(0.0, |z| levenshtein(&z.build, &self.build) as f32)
    }

    fn units_distance(&self) -> f32 {
        let Some(z) = &self.z else { return 0.0 };
        let total: f32 = self.created.iter().sum();
        self.created.iter().zip(&z.units).map(|(c, t)| ((if total > 0.0 { c / total } else { 0.0 }) - t).abs()).sum()
    }

    /// Records an own build-order action (Build / Train / Research / Upgrade group).
    pub fn record_build(&mut self, action: u16) {
        if self.build.len() < 20 {
            self.build.push(action);
        }
    }

    /// Folds one observation; returns the dense reward parts since the previous call.
    pub fn step(&mut self, frame: &RawFrame) -> RewardParts {
        let types = TypeTables::get();
        let d = dims();
        for e in &frame.events {
            match e {
                GameEvent::UnitCreated(id) => {
                    if let Some(u) = frame.units.iter().find(|u| u.id == *id && u.owner == self.self_player) {
                        if let Some(i) = d.terran_types.iter().position(|&t| t == u.unit_type) {
                            self.created[i] += 1.0;
                        }
                    }
                }
                GameEvent::UnitDestroyed(id) => {
                    // 033-R5: only destructions we observed, valued by the last type we saw.
                    if let Some(&(owner, ty)) = self.seen.get(id) {
                        let v = types.unit(ty).map_or(0.0, |t| (t.minerals + t.gas) as f32);
                        if owner == self.self_player {
                            self.lost_value += v;
                        } else {
                            self.killed_value += v;
                        }
                        self.seen.remove(id);
                    }
                }
                _ => {}
            }
        }
        for u in &frame.units {
            self.seen.insert(u.id, (u.owner, u.unit_type));
        }
        let (b, n) = (self.build_distance(), self.units_distance());
        let value = (self.killed_value - self.lost_value) / 1000.0;
        let parts = RewardParts {
            win: 0.0,
            build: if self.z.is_some() { self.prev_build - b } else { 0.0 },
            units: if self.z.is_some() { self.prev_units - n } else { 0.0 },
            value: value - self.prev_value,
        };
        (self.prev_build, self.prev_units, self.prev_value) = (b, n, value);
        parts
    }

    /// Terminal reward (033 r_win).
    pub fn end(outcome: Outcome) -> RewardParts {
        let win = match outcome {
            Outcome::Win => 1.0,
            Outcome::Loss => -1.0,
            Outcome::Draw => 0.0,
        };
        RewardParts { win, ..Default::default() }
    }
}
