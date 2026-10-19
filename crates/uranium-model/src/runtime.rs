//! 041-R15 (G-D5, AMB-032-7): Burn kernel autotune and fusion on the GPU backends (CUDA and
//! Vulkan), with the autotune cache in a fixed per-host local directory.
//!
//! Autotune and fusion are Burn cargo features (`autotune`, `fusion` on the workspace `burn`
//! dependency), so every build of the workspace — core comparison, BC, RL, inference and bench —
//! uses the same setting; there is no per-config switch. Flex (CPU, tests) has neither, so test
//! results do not change.
//!
//! The autotune cache (the kernels autotune picked, plus compiled kernels) lives in the CubeCL
//! "environment" database. [`init_gpu_runtime`] puts it under `URANIUM_AUTOTUNE_CACHE_DIR`
//! (001-R9) instead of CubeCL's default (`target/environment` of the workspace the process was
//! started in, or the user cache), so it neither moves with the working directory nor disappears
//! with `cargo clean`. On Vulkan, resuming on the same host with the same cache gives
//! bit-identical results (061-AC1); CUDA hosts (atomic-add reductions), other hosts or a rebuilt
//! cache agree only within 032-AC1 / AC8 (user decision 2026-10-19). Known limitation: on Windows
//! CubeCL 0.11 cannot open its persistent cache for writing (Turso WAL), so autotune re-runs in
//! every process.

use std::path::PathBuf;

use anyhow::{Result, bail};
use burn::cubecl::config::cache::CacheConfig;
use burn::cubecl::config::{CubeClRuntimeConfig, RuntimeConfig};

/// `.env` key (001-R9) of the per-host autotune cache directory (041-R15).
pub const AUTOTUNE_CACHE_ENV: &str = "URANIUM_AUTOTUNE_CACHE_DIR";

/// Burn kernel autotune is compiled in (workspace `burn` feature `autotune`, 041-R15).
pub const AUTOTUNE: bool = true;
/// Burn kernel fusion is compiled in (workspace `burn` feature `fusion`, 041-R15).
pub const FUSION: bool = true;

/// The Burn runtime settings recorded in checkpoints (041-R10 / R15). No host path (001-R10).
pub fn settings_json() -> serde_json::Value {
    serde_json::json!({ "autotune": AUTOTUNE, "fusion": FUSION })
}

/// `URANIUM_AUTOTUNE_CACHE_DIR`, if set and not empty.
pub fn autotune_cache_dir() -> Option<PathBuf> {
    std::env::var_os(AUTOTUNE_CACHE_ENV).filter(|v| !v.is_empty()).map(PathBuf::from)
}

/// 041-R15: points the CubeCL autotune / kernel cache of this process at
/// `URANIUM_AUTOTUNE_CACHE_DIR`. Call it once before the first CUDA or Vulkan device is used
/// (not needed for Flex). `required` (training): an unset key is an error, so a run never tunes
/// into a cache that a working-directory change or `cargo clean` would lose. Otherwise
/// (inference, bench) CubeCL's default location is used with a warning.
///
/// Returns the directory in use (`None` = CubeCL default).
pub fn init_gpu_runtime(required: bool) -> Result<Option<PathBuf>> {
    let Some(dir) = autotune_cache_dir() else {
        if required {
            bail!("{AUTOTUNE_CACHE_ENV} is not set: GPU training keeps the Burn autotune cache in a fixed local directory per host (041-R15); add it to .env (see .env.example)");
        }
        eprintln!("[burn] {AUTOTUNE_CACHE_ENV} not set: autotune cache in CubeCL's default location (041-R15)");
        return Ok(None);
    };
    std::fs::create_dir_all(&dir).map_err(|e| anyhow::anyhow!("{AUTOTUNE_CACHE_ENV} {}: {e}", dir.display()))?;
    // Absolute, but not `canonicalize`: on Windows that gives a verbatim (`\\?\`) path.
    let dir = std::path::absolute(&dir).unwrap_or(dir);
    // A cubecl.toml found from the working directory still applies; only the cache root is ours.
    let mut cfg = CubeClRuntimeConfig::from_current_dir().override_from_env();
    cfg.environment.path = CacheConfig::Directory(dir.clone());
    if !CubeClRuntimeConfig::try_set(cfg) {
        // Already initialised (a second call, or a device used before this one).
        let root = CubeClRuntimeConfig::get().environment.path.root();
        if root != dir {
            bail!("the Burn runtime was initialised before {AUTOTUNE_CACHE_ENV} was applied (cache at {}): call init_gpu_runtime before the first GPU device (041-R15)", root.display());
        }
    }
    Ok(Some(dir))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 041-R15: the checkpoint records autotune + fusion on, without a host path (001-R10).
    #[test]
    fn settings_record_autotune_and_fusion_041_r15() {
        assert_eq!(settings_json(), serde_json::json!({ "autotune": true, "fusion": true }));
    }
}
