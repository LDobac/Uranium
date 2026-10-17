//! `uranium-env` binary (crate `uranium-runner`, AMB-001-3): game scheduler (spec 020). Runs the
//! agent server in-process, spreads the games of a matchup plan over one or more
//! `uranium-hostd`s and writes results.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use clap::{Parser, Subcommand};
use uranium_agent::infer::{InferBackend, InferOptions};
use uranium_agent::{Routes, T0Builds, load_env};
use uranium_runner::{calibrate, env_or, hosts, plan, report, start_agent, start_models};

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Play every game of a matchup plan.
    Run {
        #[arg(long)]
        plan: PathBuf,
        /// Output directory (default runs/<plan name>).
        #[arg(long)]
        out: Option<PathBuf>,
        /// Inference device for `model` players (default: URANIUM_AGENT_BACKEND, else vulkan).
        #[arg(long, value_enum)]
        backend: Option<InferBackend>,
        /// Inference batch size cap (placeholder until measured, 020-R11 TBD(M6)).
        #[arg(long, default_value_t = 32)]
        max_batch: usize,
        /// Longest wait for more sessions before running a partial batch, ms (TBD(M6)).
        #[arg(long, default_value_t = 5)]
        max_wait_ms: u64,
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
        Cmd::Run { plan, out, backend, max_batch, max_wait_ms } => {
            let mut opts = InferOptions::new(backend.map_or_else(InferBackend::from_env, Ok)?);
            opts.max_batch = max_batch.max(1);
            opts.max_wait = Duration::from_millis(max_wait_ms);
            run(&plan, out, opts)
        }
        Cmd::Calibrate { max, frames } => calibrate::run(max, frames),
    }
}

fn run(plan_path: &std::path::Path, out: Option<PathBuf>, infer: InferOptions) -> Result<()> {
    let plan = plan::load_plan(plan_path)?;
    let models = start_models(&plan, infer)?;
    let out = out.unwrap_or_else(|| uranium_runner::default_out(&plan));
    std::fs::create_dir_all(&out)?;
    std::fs::copy(plan_path, out.join("plan.toml"))?;
    let _ = std::fs::remove_file(out.join("games.jsonl"));
    let _ = std::fs::remove_file(out.join("results.jsonl"));

    let builds = T0Builds::default();
    let routes = Routes::default();
    let public = start_agent(&env_or("URANIUM_AGENT_LISTEN", "127.0.0.1:7600"), &out.join("results.jsonl"), routes.clone(), builds.clone(), Default::default())?;
    let jobs = plan::expand(&plan, &public);
    eprintln!("[env] plan `{}`: {} games, agent at {public}", plan.name, jobs.len());
    let started = Instant::now();
    // Scenario builds (050) and model routes are registered per game by the workers (020-R11).
    let dispatch = uranium_runner::Dispatch::new(routes, models, builds);
    let source = std::sync::Arc::new(uranium_runner::QueueSource::new(jobs));
    uranium_runner::run_source(source.clone(), &hosts()?, &out.join("games.jsonl"), &dispatch)?;
    let left = uranium_runner::JobSource::left(&*source);
    eprintln!("[env] finished in {:.0}s ({left} games not run)", started.elapsed().as_secs_f64());
    report::write(&out, &plan)?;
    println!("{}", std::fs::read_to_string(out.join("summary.md"))?);
    if left > 0 {
        bail!("{left} games could not be scheduled");
    }
    Ok(())
}
