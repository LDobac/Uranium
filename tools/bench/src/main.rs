//! M1(a) micro-benchmarks for the Linux Vulkan and CUDA training hosts (spec 060-R2, 060-AC1/AC2, 060-AC4).
//!
//! Every case runs on the selected backend and is timed; cases with a CPU reference are also
//! run on Flex (CPU) with identical inputs and weights to report the max absolute / relative error.

mod cases;
mod net_cases;
mod report;
mod util;

use std::path::PathBuf;

use anyhow::Result;
use burn::tensor::{Device, DeviceKind, FloatDType, IntDType};
use clap::{Parser, ValueEnum};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Backend {
    Vulkan,
    /// CUDA host (060-AC4); requires `--features cuda`.
    Cuda,
    Flex,
}

#[derive(Parser, Debug)]
#[command(about = "Burn micro-benchmarks for spec 060-R2")]
struct Args {
    #[arg(long, value_enum, default_value = "vulkan")]
    backend: Backend,
    /// Smaller shapes and fewer iterations (smoke test).
    #[arg(long)]
    quick: bool,
    /// Skip the Flex (CPU) accuracy reference (faster).
    #[arg(long)]
    no_accuracy: bool,
    /// Markdown report path. Default: `reports/bench_<backend>.md` (vulkan / cuda / flex).
    #[arg(long)]
    out: Option<PathBuf>,
    /// CUDA device index for `--backend cuda`.
    #[arg(long, default_value_t = 0)]
    cuda_index: usize,
    /// Only run cases whose name contains this substring.
    #[arg(long)]
    filter: Option<String>,
    /// Make f16 the device's default float dtype (032-R8); `net_*` cases then run in f16.
    #[arg(long)]
    f16: bool,
}

#[cfg(feature = "cuda")]
fn cuda_device(index: usize) -> Result<Device> {
    Ok(Device::cuda(index))
}

#[cfg(not(feature = "cuda"))]
fn cuda_device(_index: usize) -> Result<Device> {
    anyhow::bail!("this build has no CUDA support; rebuild with `--features cuda` on a CUDA host")
}

/// 060-AC1 / 060-AC4: default report path per backend (no device label in tracked code, 001-R10).
fn default_out(backend: Backend) -> PathBuf {
    PathBuf::from(format!("reports/bench_{}.md", format!("{backend:?}").to_lowercase()))
}

fn main() -> Result<()> {
    // 001-R9: .env first, so URANIUM_AUTOTUNE_CACHE_DIR applies (060-AC4: bench under the
    // training settings, 041-R15).
    dotenvy::dotenv().ok();
    let args = Args::parse();
    // 041-R15 / 060-AC4: autotune + fusion as in training, cache in URANIUM_AUTOTUNE_CACHE_DIR.
    if !matches!(args.backend, Backend::Flex) {
        uranium_model::runtime::init_gpu_runtime(false)?;
    }
    let mut device = match args.backend {
        Backend::Vulkan => Device::vulkan(DeviceKind::DefaultDevice),
        Backend::Cuda => cuda_device(args.cuda_index)?,
        Backend::Flex => Device::flex(),
    };
    if args.f16 {
        device.configure((FloatDType::F16, IntDType::I32)).map_err(|e| anyhow::anyhow!("f16 not available: {e:?}"))?;
    }
    let reference = (!args.no_accuracy && !matches!(args.backend, Backend::Flex)).then(Device::flex);

    let ctx = cases::Ctx {
        device,
        reference,
        quick: args.quick,
        iters: if args.quick { 3 } else { 20 },
        warmup: if args.quick { 1 } else { 3 },
        cuda: matches!(args.backend, Backend::Cuda),
    };

    let mut rows = Vec::new();
    for case in cases::all() {
        if let Some(f) = &args.filter {
            if !case.name.contains(f.as_str()) {
                continue;
            }
        }
        eprintln!("[bench] {}", case.name);
        let row = util::run_guarded(case.name, || (case.run)(&ctx));
        eprintln!("        {}", row.summary());
        rows.push(row);
    }

    let backend_name = format!("{:?}{}", args.backend, if args.f16 { " f16" } else { "" }).to_lowercase();
    let out = args.out.clone().unwrap_or_else(|| default_out(args.backend));
    report::write(&out, &backend_name, args.quick, &rows)?;
    eprintln!("[bench] wrote {}", out.display());
    Ok(())
}
