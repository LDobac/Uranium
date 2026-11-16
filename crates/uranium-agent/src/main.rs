use std::path::PathBuf;

use anyhow::{Result, bail};
use clap::{Parser, Subcommand};
use uranium_agent::{PolicyKind, PolicySpec, RejectionLog, ServeOptions, load_env, serve_addr, summary};

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    Serve {
        #[arg(long, default_value = "127.0.0.1:7600")]
        listen: String,
        #[arg(long, value_enum, default_value = "scripted-t0")]
        policy: PolicyKind,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 0)]
        respond_delay_ms: u64,
        #[arg(long, default_value_t = 0)]
        sessions: usize,
    },
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
            if policy == PolicyKind::Model {
                bail!("model players run in Python: python -m uranium serve --checkpoint <ckpt> (AMB-000-4)");
            }
            let opts = ServeOptions { delay_ms: respond_delay_ms, max_sessions: sessions, unit_rejections: RejectionLog::beside(&out), ..ServeOptions::new(PolicySpec::ScriptedT0) };
            serve_addr(&listen, opts, out)
        }
        Cmd::Summarize { input, out, title } => summary::summarize(&input, &out, &title),
    }
}
