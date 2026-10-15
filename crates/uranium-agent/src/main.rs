//! `uranium-agent` binary: standalone agent server and result summarizer (spec 001-R6).

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};
use uranium_agent::{PolicyKind, ServeOptions, load_env, serve_addr, summary};

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
    },
    /// Summarize a JSONL results file into Markdown.
    Summarize {
        input: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value = "")]
        title: String,
    },
}

fn main() -> Result<()> {
    load_env();
    match Cli::parse().cmd {
        Cmd::Serve { listen, policy, out, respond_delay_ms, sessions } => {
            serve_addr(&listen, ServeOptions { policy, delay_ms: respond_delay_ms, max_sessions: sessions }, out)
        }
        Cmd::Summarize { input, out, title } => summary::summarize(&input, &out, &title),
    }
}
