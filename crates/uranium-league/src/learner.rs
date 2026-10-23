//! The learner side of the RL run loop (042). The learner itself (forward, targets, loss, KL to
//! the frozen BC policy, optimizer, the agents' inference servers) is the PyTorch learner
//! (`uranium.rl`, AMB-000-4); the run loop talks to it through [`RlLearners`].

use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use uranium_agent::infer::{HEADS, InferenceHandle};

use crate::trajectory::Segment;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum Algo {
    /// V-trace + UPGO (042 default).
    Vtrace,
    /// PPO clip baseline (042 comparison).
    Ppo,
}

/// Head order of the per-head statistics (032-R15).
pub const HEAD_NAMES: [&str; HEADS] = ["action", "delay", "queued", "selected", "target_unit", "target_loc"];

/// One update's numbers for train.csv (042-R2).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UpdateStats {
    pub update: u64,
    pub steps: usize,
    pub loss: f32,
    pub policy: f32,
    pub value: f32,
    pub kl: f32,
    pub kl_coef: f32,
    /// Per head, mean over trained steps where the head is used.
    pub kl_heads: [f32; HEADS],
    pub entropy_heads: [f32; HEADS],
    pub rho_mean: f32,
    pub adv_mean: f32,
    /// Learner version minus the segments' oldest behaviour version, mean over the batch.
    pub lag_mean: f32,
    /// 042-R13 win_open loss (0 without the head).
    pub win_open: f32,
}

/// The learning agents (042: Main, Main Exploiter, League Exploiter; index = `AgentKind::index`).
pub trait RlLearners {
    /// The agent's inference server; its weights follow [`RlLearners::push`] (032-R16).
    fn handle(&self, agent: usize) -> InferenceHandle;
    /// Parameter updates so far (= policy version, 042-R4).
    fn version(&self, agent: usize) -> u64;
    /// One update over a batch of segments (lanes); `weights` = 033 reward weights of the
    /// advantage with the league's decay schedules applied.
    fn update(&mut self, agent: usize, batch: Vec<Segment>, weights: [f32; 4]) -> Result<UpdateStats>;
    /// The current weights to the agent's inference server (042 weight push).
    fn push(&mut self, agent: usize) -> Result<()>;
    /// 042 exploiter reset: back to the BC weights, fresh optimizer, version 0.
    fn reset_to_bc(&mut self, agent: usize) -> Result<()>;
    /// Weights, optimizer state and metadata of the agent into `dir` (061-R10); `train` goes
    /// into the metadata.
    fn save(&mut self, agent: usize, dir: &Path, train: &serde_json::Value) -> Result<()>;
}

/// Starts an inference server (stochastic sampling) for a league snapshot's agent folder (042
/// opponents). Called from the game threads.
pub type SnapshotServers = Arc<dyn Fn(&Path) -> Result<InferenceHandle> + Send + Sync>;
