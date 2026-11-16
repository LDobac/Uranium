use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use uranium_proto::{GameEvent, RawFrame, UnitId};

use crate::features::Strategy;
use crate::features::extract::dims;
use crate::scout::{ScoutMetrics, ScoutTracker};
use crate::types::TypeTables;

pub const REWARD_TOML: &str = include_str!("../../../config/reward.toml");

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

#[derive(Serialize, Debug, Clone, Copy, PartialEq, Default)]
pub struct RewardParts {
    pub win: f32,
    pub build: f32,
    pub units: f32,
    pub value: f32,
    pub scout: f32,
    pub survive: f32,
}

impl RewardParts {
    pub fn weighted(&self, w: &RewardWeights) -> f32 {
        w.win * self.win + w.build * self.build + w.units * self.units + w.value * self.value
    }

    pub fn add_dense(&mut self, o: &RewardParts) {
        self.build += o.build;
        self.units += o.units;
        self.value += o.value;
        self.scout += o.scout;
        self.survive += o.survive;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Win,
    Loss,
    Draw,
}

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

#[derive(Debug, Clone)]
pub struct RewardTracker {
    z: Option<Strategy>,
    self_player: u8,
    build: Vec<u16>,
    created: Vec<f32>,
    seen: HashMap<UnitId, (u8, u16)>,
    lost_value: f32,
    killed_value: f32,
    prev_build: f32,
    prev_units: f32,
    prev_value: f32,
    scout: Option<ScoutTracker>,
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
            scout: None,
        };
        t.prev_build = t.build_distance();
        t.prev_units = t.units_distance();
        t
    }

    pub fn with_scout(mut self, scout: ScoutTracker) -> Self {
        self.scout = Some(scout);
        self
    }

    pub fn scout_metrics(&self) -> Option<ScoutMetrics> {
        self.scout.as_ref().map(|s| s.metrics())
    }

    fn build_distance(&self) -> f32 {
        self.z.as_ref().map_or(0.0, |z| levenshtein(&z.build, &self.build) as f32)
    }

    fn units_distance(&self) -> f32 {
        let Some(z) = &self.z else { return 0.0 };
        let total: f32 = self.created.iter().sum();
        self.created.iter().zip(&z.units).map(|(c, t)| ((if total > 0.0 { c / total } else { 0.0 }) - t).abs()).sum()
    }

    pub fn record_build(&mut self, action: u16) {
        if self.build.len() < 20 {
            self.build.push(action);
        }
    }

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
        let s = self.scout.as_mut().map(|t| t.step(frame)).unwrap_or_default();
        let parts = RewardParts {
            win: 0.0,
            build: if self.z.is_some() { self.prev_build - b } else { 0.0 },
            units: if self.z.is_some() { self.prev_units - n } else { 0.0 },
            value: value - self.prev_value,
            scout: s.scout,
            survive: s.survive,
        };
        (self.prev_build, self.prev_units, self.prev_value) = (b, n, value);
        parts
    }

    pub fn end(outcome: Outcome) -> RewardParts {
        let win = match outcome {
            Outcome::Win => 1.0,
            Outcome::Loss => -1.0,
            Outcome::Draw => 0.0,
        };
        RewardParts { win, ..Default::default() }
    }
}
