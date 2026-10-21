//! `uranium-agent` binary: standalone agent server and result summarizer (spec 001-R6).

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use uranium_agent::infer::{InferBackend, InferOptions, InferenceHandle, SampleMode, default_config, resolve_checkpoint, write_random_checkpoint};
use uranium_agent::{PolicyKind, PolicySpec, ServeOptions, load_env, serve_addr, summary};

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Accept bridge connections and play.
    Serve {
        #[arg(long, default_value = "127.0.0.1:7600")]
        listen: String,
        #[arg(long, value_enum, default_value = "scripted-t0")]
        policy: PolicyKind,
        /// JSONL file receiving one record per finished session.
        #[arg(long)]
        out: PathBuf,
        /// Artificial delay before every Act (011-AC7 lag test).
        #[arg(long, default_value_t = 0)]
        respond_delay_ms: u64,
        /// Exit after this many finished sessions (0 = run forever).
        #[arg(long, default_value_t = 0)]
        sessions: usize,
        #[command(flatten)]
        model: ModelArgs,
    },
    /// Summarize a JSONL results file into Markdown.
    Summarize {
        input: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value = "")]
        title: String,
    },
    /// Write a randomly initialised checkpoint (smoke tests before a BC checkpoint exists).
    RandomCheckpoint {
        #[arg(long)]
        out: PathBuf,
        #[arg(long, value_enum, default_value = "lstm")]
        core: CoreArg,
    },
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum CoreArg {
    Lstm,
    Gtrxl,
}

/// Model policy options (020-R11). Batch defaults are placeholders until measured (TBD(M6)).
#[derive(clap::Args)]
struct ModelArgs {
    /// Checkpoint folder; relative paths are under URANIUM_SHARED_DIR (061).
    #[arg(long)]
    checkpoint: Option<PathBuf>,
    /// Inference device (default: URANIUM_AGENT_BACKEND, else vulkan).
    #[arg(long, value_enum)]
    backend: Option<InferBackend>,
    #[arg(long, value_enum, default_value = "argmax")]
    sample: SampleMode,
    #[arg(long, default_value_t = 32)]
    max_batch: usize,
    #[arg(long, default_value_t = 5)]
    max_wait_ms: u64,
    /// 030-R10 / E-5: feed last_failed = 0 (evaluation games of a BC checkpoint; BC data never
    /// has a failed command, see `InferOptions::zero_last_failed`). Off for RL.
    #[arg(long)]
    zero_last_failed: bool,
    /// 031-R12 / 020-R14: surrender mode (`off`, `shadow`: check and record only, `on`: chat and
    /// leave). SC:R games use `on` (012-R11).
    #[arg(long, value_enum, default_value = "shadow")]
    surrender: SurrenderArg,
    /// 020-R14: chat line sent before leaving (empty = no chat, at most 80 bytes).
    #[arg(long, default_value = uranium_agent::infer::DEFAULT_CHAT)]
    surrender_chat: String,
}

#[derive(clap::ValueEnum, Clone, Copy, Debug)]
enum SurrenderArg {
    Off,
    Shadow,
    On,
}

impl From<SurrenderArg> for uranium_agent::infer::SurrenderMode {
    fn from(a: SurrenderArg) -> Self {
        match a {
            SurrenderArg::Off => Self::Off,
            SurrenderArg::Shadow => Self::Shadow,
            SurrenderArg::On => Self::On,
        }
    }
}

fn main() -> Result<()> {
    load_env();
    match Cli::parse().cmd {
        Cmd::Serve { listen, policy, out, respond_delay_ms, sessions, model } => {
            let default = match policy {
                PolicyKind::ScriptedT0 => PolicySpec::ScriptedT0,
                PolicyKind::Model => {
                    let dir = resolve_checkpoint(&model.checkpoint.context("--policy model needs --checkpoint")?)?;
                    let mut opts = InferOptions::new(model.backend.map_or_else(InferBackend::from_env, Ok)?);
                    opts.max_batch = model.max_batch.max(1);
                    opts.max_wait = Duration::from_millis(model.max_wait_ms);
                    opts.zero_last_failed = model.zero_last_failed;
                    if model.surrender_chat.len() > uranium_proto::MAX_SURRENDER_CHAT {
                        anyhow::bail!("--surrender-chat is {} bytes, at most {} (020-R14)", model.surrender_chat.len(), uranium_proto::MAX_SURRENDER_CHAT);
                    }
                    PolicySpec::Model(InferenceHandle::start(&dir, model.sample, opts)?.with_surrender(model.surrender.into(), &model.surrender_chat))
                }
            };
            let opts = ServeOptions { delay_ms: respond_delay_ms, max_sessions: sessions, ..ServeOptions::new(default) };
            serve_addr(&listen, opts, out)
        }
        Cmd::Summarize { input, out, title } => summary::summarize(&input, &out, &title),
        Cmd::RandomCheckpoint { out, core } => {
            let kind = match core {
                CoreArg::Lstm => uranium_model::config::CoreKind::Lstm,
                CoreArg::Gtrxl => uranium_model::config::CoreKind::Gtrxl,
            };
            write_random_checkpoint(&out, &default_config(kind), &burn::tensor::Device::flex())?;
            println!("{}", uranium_agent::infer::checkpoint_id(&out)?);
            Ok(())
        }
    }
}
