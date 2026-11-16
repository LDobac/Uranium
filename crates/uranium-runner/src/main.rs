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
    Run {
        #[arg(long)]
        plan: PathBuf,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    Calibrate {
        #[arg(long)]
        max: Option<u32>,
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
