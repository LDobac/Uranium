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
    /// 042-R2 진단 로그 (AMB-042-17), over the batch's trained queue-lane steps: RMS of the
    /// combined policy-gradient advantage (the value `adv_mean` averages).
    #[serde(default)]
    pub adv_rms: f32,
    /// RMS of each channel's V-trace PG advantage (win, build, units, value) before the 033
    /// weights and per-game multipliers.
    #[serde(default)]
    pub adv_rms_channels: [f32; 4],
    /// Each channel's mean 0.5 (V - target)^2, before the value loss weight.
    #[serde(default)]
    pub value_loss_channels: [f32; 4],
    /// 99th percentile and maximum of the per-step KL(π ‖ π_BC) (used heads summed, before the
    /// head multipliers); `None` without a trained step.
    #[serde(default)]
    pub kl_p99: Option<f32>,
    #[serde(default)]
    pub kl_max: Option<f32>,
    /// 042-R16: the self-imitation loss (weights applied; `"sil"` mode) and the share of the
    /// buffer lanes' steps with `max(R - V, 0) > 0`; `None` when off / not measured.
    #[serde(default)]
    pub sil_loss: Option<f32>,
    #[serde(default)]
    pub sil_pos_frac: Option<f32>,
    /// 042-R21 ③: the aux losses before λ, batch means (`None` without the module): `aux_priv`,
    /// `aux_jepa` (the mean over k), the JEPA target's mean per-dimension standard deviation and
    /// one value per configured k (`[model] aux_jepa` order, `None` past the list or without a
    /// pair in the batch).
    #[serde(default)]
    pub aux_priv: Option<f32>,
    #[serde(default)]
    pub aux_jepa: Option<f32>,
    #[serde(default)]
    pub jepa_tgt_std: Option<f32>,
    #[serde(default)]
    pub aux_jepa_slots: [Option<f32>; 3],
    /// 042-R21 ③ / 032-R25: the spatial statistics of a `[monitor] spatial_every` update in
    /// [`SPATIAL_COLUMNS`] order (`None` on the other updates and when off).
    #[serde(default)]
    pub spatial: Option<Vec<f32>>,
}

/// 041-R25 ② layer names in order (the Python `uranium.model.monitor.SPEC_LAYERS`): ReLU outputs
/// `stem`, `c0`..`c3` (stride-2 convs), `r0`..`r5` (the ResBlocks' inner ReLU, four down then two
/// bottleneck), then the block outputs `d0`..`d3`, `b0`, `b1`.
pub const SPATIAL_LAYERS: [&str; 17] = ["stem", "c0", "c1", "c2", "c3", "r0", "r1", "r2", "r3", "r4", "r5", "d0", "d1", "d2", "d3", "b0", "b1"];

/// 042-R21 ③: the spatial statistics columns, grouped: every `sp_pos_<layer>`, every
/// `sp_dead_<layer>` (same order), then `sp_const_s8`, `sp_const_pool`.
pub fn spatial_columns() -> Vec<String> {
    let mut c: Vec<String> = SPATIAL_LAYERS.iter().map(|l| format!("sp_pos_{l}")).collect();
    c.extend(SPATIAL_LAYERS.iter().map(|l| format!("sp_dead_{l}")));
    c.push("sp_const_s8".into());
    c.push("sp_const_pool".into());
    c
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

/// A running inference server of a league snapshot or the anchor (042 opponents).
pub struct SnapshotServer {
    pub handle: InferenceHandle,
    /// Stops the server and frees its device memory (042 스냅샷 추론 서버, AMB-042-14). Called
    /// once no game uses it any more; sessions still holding `handle` get errors after.
    pub close: Box<dyn FnOnce() + Send>,
    /// The server still serves: false once its model thread ended (e.g. a failed warm-up after
    /// the start returned), so a cached handle is never given to a new game.
    pub alive: Box<dyn Fn() -> bool + Send>,
}

/// Starts an inference server (stochastic sampling) for a league snapshot's or the anchor's
/// agent folder (042 opponents). Called from the game threads.
pub type SnapshotServers = Arc<dyn Fn(&Path) -> Result<SnapshotServer> + Send + Sync>;
