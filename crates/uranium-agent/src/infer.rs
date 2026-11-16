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

pub const HEADS: usize = 6;

pub const WEIGHTS_FILE: &str = "model.safetensors";
pub const BURN_WEIGHTS_FILE: &str = "model.bpk";

/// Action sampling mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum SampleMode {
    /// Argmax of every head (deterministic).
    #[default]
    Argmax,
    /// Gumbel-max sampling at temperature 1.
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

#[derive(Clone, Copy, Debug)]
pub struct InferOptions {
    pub max_batch: usize,
    pub max_wait: Duration,
    pub capacity: usize,
    pub pad_entities: usize,
    pub zero_last_failed: bool,
    pub warmup_entities: usize,
}

impl Default for InferOptions {
    fn default() -> Self {
        Self { max_batch: 32, max_wait: Duration::from_millis(5), capacity: 128, pad_entities: 64, zero_last_failed: false, warmup_entities: 0 }
    }
}

#[derive(Debug)]
pub struct ActReply {
    pub action: HostAction,
    pub log_prob: f32,
    pub head_log_prob: [f32; HEADS],
    pub value: [f32; 4],
    pub win_open: Option<f32>,
    pub obs: Observation,
    pub batch: usize,
    pub run: Duration,
    pub version: u64,
}

pub enum Req {
    Open(Sender<Result<usize>>),
    Act { slot: usize, obs: Box<Observation>, reply: Sender<Result<ActReply>> },
    Close(usize),
    Swap { net: Box<dyn Any + Send>, version: u64, reply: Sender<Result<()>> },
    State { slot: usize, reply: Sender<Result<HostState>> },
    SetState { slot: usize, state: HostState, reply: Sender<Result<()>> },
}

#[derive(Clone)]
pub struct InferenceHandle {
    tx: Sender<Req>,
    pub checkpoint_id: Arc<str>,
    pub sample: SampleMode,
    pub zero_last_failed: bool,
    pub surrender: SurrenderMode,
    pub surrender_chat: Arc<str>,
    pub memory_log: Option<PathBuf>,
    pub infer_seed: Option<u64>,
}

impl std::fmt::Debug for InferenceHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "InferenceHandle({}, {:?})", self.checkpoint_id, self.sample)
    }
}

pub fn resolve_checkpoint(p: &Path) -> Result<PathBuf> {
    if p.is_absolute() {
        return Ok(p.to_path_buf());
    }
    let shared = std::env::var_os("URANIUM_SHARED_DIR").filter(|v| !v.is_empty()).context("relative checkpoint path needs URANIUM_SHARED_DIR (020-R11)")?;
    Ok(PathBuf::from(shared).join(p))
}

pub fn checkpoint_id(dir: &Path) -> Result<String> {
    let file = if dir.join(WEIGHTS_FILE).is_file() || !dir.join(BURN_WEIGHTS_FILE).is_file() { WEIGHTS_FILE } else { BURN_WEIGHTS_FILE };
    let bytes = std::fs::read(dir.join(file)).with_context(|| format!("reading {}", dir.join(file).display()))?;
    let sha: String = Sha256::digest(&bytes).iter().take(6).map(|b| format!("{b:02x}")).collect();
    let name = dir.file_name().map_or_else(|| "checkpoint".into(), |n| n.to_string_lossy().into_owned());
    Ok(format!("{name}@{sha}"))
}

pub fn checkpoint_core(dir: &Path) -> Result<String> {
    let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("meta.json")).with_context(|| format!("reading {}/meta.json", dir.display()))?)?;
    Ok(meta["config"]["core"]["kind"].as_str().unwrap_or("Lstm").to_lowercase())
}

pub fn model_policy_label(checkpoint_id: &str, sample: SampleMode) -> String {
    format!("model:{checkpoint_id}:{sample:?}").to_lowercase()
}

impl InferenceHandle {
    pub fn new(tx: Sender<Req>, checkpoint_id: String, sample: SampleMode, opts: &InferOptions) -> Self {
        Self { tx, checkpoint_id: checkpoint_id.into(), sample, zero_last_failed: opts.zero_last_failed, surrender: SurrenderMode::Shadow, surrender_chat: DEFAULT_CHAT.into(), memory_log: None, infer_seed: None }
    }

    pub fn with_infer_seed(mut self, seed: Option<u64>) -> Self {
        self.infer_seed = seed;
        self
    }

    pub fn with_memory_log(mut self, path: PathBuf) -> Self {
        self.memory_log = Some(path);
        self
    }

    pub fn with_surrender(mut self, mode: SurrenderMode, chat: &str) -> Self {
        self.surrender = mode;
        self.surrender_chat = chat.into();
        self
    }

    pub fn swap(&self, net: Box<dyn Any + Send>, version: u64) -> Result<()> {
        let (tx, rx) = mpsc::channel();
        self.tx.send(Req::Swap { net, version, reply: tx }).map_err(|_| anyhow!("inference server is gone"))?;
        rx.recv().map_err(|_| anyhow!("inference server is gone"))?
    }

    pub fn open(&self) -> Result<InferSession> {
        let (tx, rx) = mpsc::channel();
        self.tx.send(Req::Open(tx)).map_err(|_| anyhow!("inference server is gone"))?;
        let slot = rx.recv().map_err(|_| anyhow!("inference server is gone"))??;
        Ok(InferSession { slot, tx: self.tx.clone() })
    }
}

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

    pub fn state(&self) -> Result<HostState> {
        let (tx, rx) = mpsc::channel();
        self.tx.send(Req::State { slot: self.slot, reply: tx }).map_err(|_| anyhow!("inference server is gone"))?;
        rx.recv().map_err(|_| anyhow!("inference server dropped the request"))?
    }

    pub fn set_state(&self, state: HostState) -> Result<()> {
        let (tx, rx) = mpsc::channel();
        self.tx.send(Req::SetState { slot: self.slot, state, reply: tx }).map_err(|_| anyhow!("inference server is gone"))?;
        rx.recv().map_err(|_| anyhow!("inference server dropped the request"))?
    }
}

impl Drop for InferSession {
    fn drop(&mut self) {
        let _ = self.tx.send(Req::Close(self.slot));
    }
}

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
