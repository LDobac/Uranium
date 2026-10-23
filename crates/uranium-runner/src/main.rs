//! `uranium-env` binary (crate `uranium-runner`, AMB-001-3): game scheduler (spec 020). Runs the
//! agent server in-process, spreads the games of a matchup plan over one or more
//! `uranium-hostd`s and writes results. Plans with model players run in the Python process,
//! which hosts the inference servers: `python -m uranium run --plan <plan>` (AMB-000-4).

use std::path::PathBuf;

use anyhow::{Result, bail};
use clap::{Parser, Subcommand};
use uranium_agent::load_env;
use uranium_runner::{Models, calibrate, plan, plan_models, run_plan};

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Play every game of a matchup plan of scripted players.
    Run {
        #[arg(long)]
        plan: PathBuf,
        /// Output directory (default runs/<plan name>).
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Measure how many concurrent games this host sustains within the bridge latency
    /// target and store it for uranium-hostd (020-R10). Run on the game host itself.
    Calibrate {
        /// Largest concurrency to try (default: logical cores).
        #[arg(long)]
        max: Option<u32>,
        /// Frame limit per game.
        #[arg(long, default_value_t = 8000)]
        frames: u32,
    },
}

fn main() -> Result<()> {
    load_env();
    match Cli::parse().cmd {
        Cmd::Run { plan: path, out } => {
            let p = plan::load_plan(&path)?;
            if !plan_models(&p).is_empty() {
                bail!("plan `{}` has model players: run it with python -m uranium run --plan {} (AMB-000-4)", p.name, path.display());
            }
            run_plan(&path, out, Models::new())
        }
        Cmd::Calibrate { max, frames } => calibrate::run(max, frames),
    }
}
