//! `uranium-train bc --shards <dir> --out <ckpt dir>` (041).

use std::path::PathBuf;

use anyhow::Result;
use burn::tensor::{Device, DeviceKind};
use clap::{Parser, Subcommand, ValueEnum};
use uranium_model::collate::input_dims;
use uranium_model::config::{CoreKind, UraniumNetConfig};
use uranium_model::{GtrxlCore, LstmCore, UraniumNet};
use uranium_train::bc::{BcConfig, Trainer};
use uranium_train::data::ShardSource;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Backend {
    Vulkan,
    Cuda,
    Flex,
}

#[derive(Parser)]
#[command(about = "Uranium training (spec 041)")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Behaviour cloning over replay shards.
    Bc {
        #[arg(long)]
        shards: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value = "config/bc.toml")]
        config: PathBuf,
        #[arg(long, value_enum, default_value = "vulkan")]
        backend: Backend,
        #[arg(long, value_enum, default_value = "lstm")]
        core: CoreArg,
        /// S2 fine-tuning: learning rate x s2_lr_scale (041-R4). Start from --init.
        #[arg(long)]
        s2: bool,
        /// Stop after this many updates (default: config total_steps).
        #[arg(long)]
        steps: Option<usize>,
    },
    /// 034-AC3 diagnostics: labels that fail the masks, by action and reason (JSON).
    LabelCheck {
        #[arg(long)]
        shards: PathBuf,
        /// Stop after this many games.
        #[arg(long)]
        games: Option<usize>,
    },
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CoreArg {
    Lstm,
    Gtrxl,
}

fn device(b: Backend) -> Result<Device> {
    Ok(match b {
        Backend::Vulkan => Device::vulkan(DeviceKind::DefaultDevice),
        #[cfg(feature = "cuda")]
        Backend::Cuda => Device::cuda(0),
        #[cfg(not(feature = "cuda"))]
        Backend::Cuda => anyhow::bail!("rebuild with --features cuda on a CUDA PC"),
        Backend::Flex => Device::flex(),
    })
}

fn run<C: uranium_model::CoreInit>(kind: CoreKind, shards: PathBuf, out: PathBuf, cfg: BcConfig, dev: Device, s2: bool, steps: Option<usize>) -> Result<()> {
    let ad = dev.autodiff();
    let base = UraniumNetConfig::new();
    let net_cfg = base.clone().with_input(input_dims()).with_core(base.core.clone().with_kind(kind));
    let cols = uranium_model::batch::ActionColumns::from_table(uranium_env::features::extract::action_table())?;
    let net = UraniumNet::<C>::new(&net_cfg, &cols, &ad);
    let total = steps.unwrap_or(cfg.total_steps);
    let every = cfg.data.checkpoint_every.max(1);
    let mut t = Trainer::new(net, net_cfg, cfg, ad);
    if s2 {
        t.lr_scale = t.cfg.s2_lr_scale;
    }
    let mut src = ShardSource::new(&shards)?;
    while t.step < total {
        let Some(l) = t.train_step(&mut src)? else {
            eprintln!("data exhausted after {} updates", t.step);
            break;
        };
        eprintln!("{}", serde_json::json!({ "step": t.step, "loss": l, "games": t.games, "invalid_labels": t.invalid_labels }));
        if t.step % every == 0 {
            t.save(&out.join(format!("step_{:08}", t.step)))?;
        }
    }
    t.save(&out.join("last"))
}

fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    match Cli::parse().cmd {
        Cmd::LabelCheck { shards, games } => {
            let mut src = ShardSource::new(&shards)?;
            let (mut fails, mut totals, mut n) = (Default::default(), std::collections::BTreeMap::new(), 0usize);
            while let Some(g) = uranium_train::data::StepSource::next_game(&mut src)? {
                uranium_train::data::label_check(&g, &mut fails, &mut totals)?;
                n += 1;
                if games.is_some_and(|m| n >= m) {
                    break;
                }
            }
            let invalid: usize = fails.values().sum();
            let mut rows: Vec<_> = fails.iter().map(|((a, w), c)| (a.clone(), *w, *c, totals[a])).collect();
            rows.sort_by(|x, y| y.2.cmp(&x.2));
            let labels: usize = totals.values().sum();
            println!("{}", serde_json::to_string_pretty(&serde_json::json!({ "games": n, "labels": labels, "invalid": invalid, "by_action_reason": rows }))?);
            Ok(())
        }
        Cmd::Bc { shards, out, config, backend, core, s2, steps } => {
            let cfg = BcConfig::load(&config)?;
            let dev = device(backend)?;
            match core {
                CoreArg::Lstm => run::<LstmCore>(CoreKind::Lstm, shards, out, cfg, dev, s2, steps),
                CoreArg::Gtrxl => run::<GtrxlCore>(CoreKind::Gtrxl, shards, out, cfg, dev, s2, steps),
            }
        }
    }
}
