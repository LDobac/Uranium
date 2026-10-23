//! Batched inference sessions (032-R7, 032-R14, 001-R6). One server per (checkpoint, sampling
//! mode) owns the model; sessions send observations over a channel and the server runs one `act`
//! over every waiting session. Each session has a fixed slot whose Core state lives in one
//! `[capacity, …]` state on the server. Lockstep games wait for their reply, so batching changes
//! throughput only, never results.
//!
//! The server is the PyTorch inference thread behind [`crate::broker::Broker`] (AMB-000-4); this
//! module is the framework-free session side and the channel protocol ([`Req`]).

use std::any::Any;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{self, Sender};
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uranium_data::table::{ActionColumns, TARGET_LOC, TARGET_UNIT};
use uranium_env::features::Observation;
use uranium_env::features::executor::HostAction;
pub use uranium_data::state::HostState;
pub use uranium_env::surrender::{DEFAULT_CHAT, SurrenderMode};

/// Autoregressive heads (032 §2.5): action, delay, queued, selected, target_unit, target_loc.
pub const HEADS: usize = 6;

/// Weights file of a checkpoint (061-R10); its SHA-256 is the policy version (020-R5).
pub const WEIGHTS_FILE: &str = "model.safetensors";
/// Weights of a Burn checkpoint (kept, not loaded by the PyTorch server; Burn RL until 1-8).
pub const BURN_WEIGHTS_FILE: &str = "model.bpk";

/// 050-R1 / 032-R15 sampling mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum SampleMode {
    /// temp 0: argmax of every head (deterministic).
    #[default]
    Argmax,
    /// temp 1: Gumbel-max sampling.
    Stochastic,
}

impl SampleMode {
    pub fn temp(self) -> f32 {
        match self {
            Self::Argmax => 0.0,
            Self::Stochastic => 1.0,
        }
    }
}

/// Batching parameters. Defaults are placeholders until measured (020-R11, TBD(M6)).
#[derive(Clone, Copy, Debug)]
pub struct InferOptions {
    pub max_batch: usize,
    pub max_wait: Duration,
    /// Session slots (concurrent games x model players).
    pub capacity: usize,
    /// Entity count is rounded up to a multiple of this (fewer distinct shapes, 030-R7).
    pub pad_entities: usize,
    /// 030-R10 / E-5: feed `last_failed = 0` to the model in every session of this server.
    /// Why: replay playback has no command-failure events, so in BC training data the
    /// `last_failed` scalar is always 0. A BC checkpoint has never seen 1 there; in an evaluation
    /// game a 1 would be an untrained input and only add noise to the evaluation numbers. So
    /// evaluation games of BC checkpoints (core ablation 032-R17, 041-AC2 bc-vs-t0) set this
    /// (plan `zero_last_failed = true`). RL (042) leaves it off and uses the real value.
    pub zero_last_failed: bool,
    /// 012-R19 (SC:R realtime): before serving, run one batch-1 forward per padded entity count up
    /// to this many entities, so kernel selection does not stall a live game (the first forward on
    /// CUDA took 57 s in the first SC:R game, 2026-10-22). 0 = no warm-up.
    pub warmup_entities: usize,
}

impl Default for InferOptions {
    fn default() -> Self {
        Self { max_batch: 32, max_wait: Duration::from_millis(5), capacity: 128, pad_entities: 64, zero_last_failed: false, warmup_entities: 0 }
    }
}

/// One decision for one session.
#[derive(Debug)]
pub struct ActReply {
    pub action: HostAction,
    /// Joint behaviour log-prob of the used heads (032-R10, 042).
    pub log_prob: f32,
    /// Per-head log-prob, unused heads 0 (032-R15).
    pub head_log_prob: [f32; HEADS],
    /// win, build, units, value (032 §2.6, no privileged input).
    pub value: [f32; 4],
    /// 032-R18 win estimate in [-1, 1]; `None` when the checkpoint has no win_open head.
    pub win_open: Option<f32>,
    /// The observation, handed back so the caller can run the executor without a copy.
    pub obs: Observation,
    /// Rows in the batch this decision ran in.
    pub batch: usize,
    /// Model forward + host copy of that batch.
    pub run: Duration,
    /// Weights version that produced this decision (032-R16, 042-R4): 0 = the loaded checkpoint.
    pub version: u64,
}

/// Session → server messages (the server side is [`crate::broker::Broker`], or the Burn server of
/// the Rust RL learner until it moves to PyTorch).
pub enum Req {
    Open(Sender<Result<usize>>),
    Act { slot: usize, obs: Box<Observation>, reply: Sender<Result<ActReply>> },
    Close(usize),
    /// 032-R16: replace the weights from the next batch on (a server-specific network object).
    Swap { net: Box<dyn Any + Send>, version: u64, reply: Sender<Result<()>> },
    /// 032-R16: the slot's Core state on the host (042 core_state0).
    State { slot: usize, reply: Sender<Result<HostState>> },
}

/// Cheap to clone; every clone talks to the same server.
#[derive(Clone)]
pub struct InferenceHandle {
    tx: Sender<Req>,
    /// 020-R5 policy version: checkpoint folder name + first 12 hex of the weights' SHA-256.
    pub checkpoint_id: Arc<str>,
    pub sample: SampleMode,
    /// `InferOptions::zero_last_failed` of the server (030-R10, E-5).
    pub zero_last_failed: bool,
    /// 031-R12 / 020-R14: surrender mode of the sessions (default `Shadow`: check, never leave).
    pub surrender: SurrenderMode,
    /// 020-R14: chat line sent before leaving (default `gg`, empty = no chat).
    pub surrender_chat: Arc<str>,
}

impl std::fmt::Debug for InferenceHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "InferenceHandle({}, {:?})", self.checkpoint_id, self.sample)
    }
}

/// 020-R11: a relative checkpoint path is relative to `URANIUM_SHARED_DIR` (061).
pub fn resolve_checkpoint(p: &Path) -> Result<PathBuf> {
    if p.is_absolute() {
        return Ok(p.to_path_buf());
    }
    let shared = std::env::var_os("URANIUM_SHARED_DIR").filter(|v| !v.is_empty()).context("relative checkpoint path needs URANIUM_SHARED_DIR (020-R11)")?;
    Ok(PathBuf::from(shared).join(p))
}

/// 020-R5: `<folder>@<first 12 hex of SHA-256(model.safetensors)>` (a Burn checkpoint: of
/// `model.bpk`, for the Burn RL learner until 1-8).
pub fn checkpoint_id(dir: &Path) -> Result<String> {
    let file = if dir.join(WEIGHTS_FILE).is_file() || !dir.join(BURN_WEIGHTS_FILE).is_file() { WEIGHTS_FILE } else { BURN_WEIGHTS_FILE };
    let bytes = std::fs::read(dir.join(file)).with_context(|| format!("reading {}", dir.join(file).display()))?;
    let sha: String = Sha256::digest(&bytes).iter().take(6).map(|b| format!("{b:02x}")).collect();
    let name = dir.file_name().map_or_else(|| "checkpoint".into(), |n| n.to_string_lossy().into_owned());
    Ok(format!("{name}@{sha}"))
}

/// The checkpoint's Core kind, lower case (`lstm` / `gtrxl`), for reports (020-R5 summary).
pub fn checkpoint_core(dir: &Path) -> Result<String> {
    let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("meta.json")).with_context(|| format!("reading {}/meta.json", dir.display()))?)?;
    Ok(meta["config"]["core"]["kind"].as_str().unwrap_or("Lstm").to_lowercase())
}

/// 020-R5: the session `policy` label of a model player (same text as `PolicySpec::label`).
pub fn model_policy_label(checkpoint_id: &str, sample: SampleMode) -> String {
    format!("model:{checkpoint_id}:{sample:?}").to_lowercase()
}

impl InferenceHandle {
    /// A handle on the server that reads `tx` (its other end).
    pub fn new(tx: Sender<Req>, checkpoint_id: String, sample: SampleMode, opts: &InferOptions) -> Self {
        Self { tx, checkpoint_id: checkpoint_id.into(), sample, zero_last_failed: opts.zero_last_failed, surrender: SurrenderMode::Shadow, surrender_chat: DEFAULT_CHAT.into() }
    }

    /// 031-R12 / 020-R14: surrender mode and chat line for the sessions opened from this handle.
    pub fn with_surrender(mut self, mode: SurrenderMode, chat: &str) -> Self {
        self.surrender = mode;
        self.surrender_chat = chat.into();
        self
    }

    /// 032-R16: weights for the next batches (servers that take a network object: the Burn RL
    /// server); `version` is reported with every later decision.
    pub fn swap(&self, net: Box<dyn Any + Send>, version: u64) -> Result<()> {
        let (tx, rx) = mpsc::channel();
        self.tx.send(Req::Swap { net, version, reply: tx }).map_err(|_| anyhow!("inference server is gone"))?;
        rx.recv().map_err(|_| anyhow!("inference server is gone"))?
    }

    /// A slot for one game player; its Core state starts at zero (032-R5).
    pub fn open(&self) -> Result<InferSession> {
        let (tx, rx) = mpsc::channel();
        self.tx.send(Req::Open(tx)).map_err(|_| anyhow!("inference server is gone"))?;
        let slot = rx.recv().map_err(|_| anyhow!("inference server is gone"))??;
        Ok(InferSession { slot, tx: self.tx.clone() })
    }
}

/// A session's slot; closing it (drop) frees the slot.
pub struct InferSession {
    slot: usize,
    tx: Sender<Req>,
}

impl InferSession {
    pub fn act(&self, obs: Observation) -> Result<ActReply> {
        let (tx, rx) = mpsc::channel();
        self.tx.send(Req::Act { slot: self.slot, obs: Box::new(obs), reply: tx }).map_err(|_| anyhow!("inference server is gone"))?;
        rx.recv().map_err(|_| anyhow!("inference server dropped the request"))?
    }

    /// 032-R16: this slot's current Core state (before the next `act`), for core_state0 (042).
    pub fn state(&self) -> Result<HostState> {
        let (tx, rx) = mpsc::channel();
        self.tx.send(Req::State { slot: self.slot, reply: tx }).map_err(|_| anyhow!("inference server is gone"))?;
        rx.recv().map_err(|_| anyhow!("inference server dropped the request"))?
    }
}

impl Drop for InferSession {
    fn drop(&mut self) {
        let _ = self.tx.send(Req::Close(self.slot));
    }
}

/// The executor's view of row `r` of a batch output (034 columns gate the arguments; END = the
/// padded entity count `n` ends a selection, 032-R12).
#[allow(clippy::too_many_arguments)]
pub fn host_action(cols: &ActionColumns, action: i64, delay: i64, queued: i64, selected: &[i64], target_unit: i64, target_loc: i64, n: usize) -> HostAction {
    let a = action as usize;
    let uses = |v: &Vec<i32>| v.get(a).copied().unwrap_or(0);
    HostAction {
        action: a as u16,
        delay_bucket: delay as usize,
        queued: queued == 1,
        selected: if uses(&cols.uses_select) == 1 { selected.iter().take_while(|&&i| i != n as i64).map(|&i| i as usize).collect() } else { vec![] },
        target_unit: (uses(&cols.target_kind) == TARGET_UNIT).then_some(target_unit as usize),
        target_loc: (uses(&cols.target_kind) == TARGET_LOC).then_some(target_loc as usize),
    }
}
