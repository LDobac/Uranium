use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use uranium_agent::infer::{HEADS, InferenceHandle};

use crate::trajectory::Segment;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum Algo {
    Vtrace,
    Ppo,
}

pub const HEAD_NAMES: [&str; HEADS] = ["action", "delay", "queued", "selected", "target_unit", "target_loc"];

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UpdateStats {
    pub update: u64,
    pub steps: usize,
    pub loss: f32,
    pub policy: f32,
    pub value: f32,
    pub kl: f32,
    pub kl_coef: f32,
    pub kl_heads: [f32; HEADS],
    pub entropy_heads: [f32; HEADS],
    pub rho_mean: f32,
    pub adv_mean: f32,
    pub lag_mean: f32,
    pub win_open: f32,
    #[serde(default)]
    pub adv_rms: f32,
    #[serde(default)]
    pub adv_rms_channels: [f32; 4],
    #[serde(default)]
    pub value_loss_channels: [f32; 4],
    #[serde(default)]
    pub kl_p99: Option<f32>,
    #[serde(default)]
    pub kl_max: Option<f32>,
    #[serde(default)]
    pub sil_loss: Option<f32>,
    #[serde(default)]
    pub sil_pos_frac: Option<f32>,
    #[serde(default)]
    pub aux_priv: Option<f32>,
    #[serde(default)]
    pub aux_jepa: Option<f32>,
    #[serde(default)]
    pub jepa_tgt_std: Option<f32>,
    #[serde(default)]
    pub aux_jepa_slots: [Option<f32>; 3],
    #[serde(default)]
    pub spatial: Option<Vec<f32>>,
    #[serde(default)]
    pub kl_ref: [Option<f32>; 4],
}

pub const SPATIAL_LAYERS: [&str; 17] = ["stem", "c0", "c1", "c2", "c3", "r0", "r1", "r2", "r3", "r4", "r5", "d0", "d1", "d2", "d3", "b0", "b1"];

pub fn spatial_columns() -> Vec<String> {
    let mut c: Vec<String> = SPATIAL_LAYERS.iter().map(|l| format!("sp_pos_{l}")).collect();
    c.extend(SPATIAL_LAYERS.iter().map(|l| format!("sp_dead_{l}")));
    c.push("sp_const_s8".into());
    c.push("sp_const_pool".into());
    c
}

pub trait RlLearners {
    fn handle(&self, agent: usize) -> InferenceHandle;
    fn version(&self, agent: usize) -> u64;
    fn update(&mut self, agent: usize, batch: Vec<Segment>, weights: [f32; 4]) -> Result<UpdateStats>;
    fn push(&mut self, agent: usize) -> Result<()>;
    fn reset_to_bc(&mut self, agent: usize) -> Result<()>;
    fn save(&mut self, agent: usize, dir: &Path, train: &serde_json::Value) -> Result<()>;
}

pub struct SnapshotServer {
    pub handle: InferenceHandle,
    pub close: Box<dyn FnOnce() + Send>,
    pub alive: Box<dyn Fn() -> bool + Send>,
}

pub type SnapshotServers = Arc<dyn Fn(&Path) -> Result<SnapshotServer> + Send + Sync>;
