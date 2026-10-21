//! Checkpoints (032-R6): weights (burnpack) + `meta.json` with the net config, feature and
//! action schema versions, normalisation statistics and free-form training metadata (033-R4,
//! 041-R10).

use std::path::Path;

use anyhow::{Context, Result, anyhow, ensure};
use burn::store::{BurnpackStore, ModuleSnapshot};
use burn::tensor::Device;
use serde::{Deserialize, Serialize};

use crate::batch::ActionColumns;
use crate::config::UraniumNetConfig;
use crate::net::{CoreInit, UraniumNet};

pub const WEIGHTS_FILE: &str = "model.bpk";
pub const META_FILE: &str = "meta.json";

/// Per-feature normalisation statistics (030). Empty when the fixed 030 rules are used.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct NormStats {
    pub scalar_mean: Vec<f32>,
    pub scalar_std: Vec<f32>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct CheckpointMeta {
    pub config: serde_json::Value,
    pub feature_version: u32,
    pub action_version: u32,
    pub norm: NormStats,
    /// Training hyper-parameters, reward weights, step counters (041-R10, 033-R4).
    pub train: serde_json::Value,
}

impl CheckpointMeta {
    pub fn new(cfg: &UraniumNetConfig, feature_version: u32, action_version: u32) -> Result<Self> {
        Ok(Self {
            config: serde_json::to_value(cfg)?,
            feature_version,
            action_version,
            norm: NormStats::default(),
            train: serde_json::Value::Null,
        })
    }

    /// The stored net config. Keys a config written before a field existed lacks (032-R18
    /// `value.win_open`) take that field's default first: Burn's `Config` derive needs every
    /// field on deserialisation, and older checkpoints must load unchanged.
    pub fn net_config(&self) -> Result<UraniumNetConfig> {
        let mut v = self.config.clone();
        fill_missing(&mut v, &serde_json::to_value(UraniumNetConfig::new())?);
        Ok(serde_json::from_value(v)?)
    }
}

/// Inserts every key of `defaults` that `v` lacks, recursively through nested objects. Keys
/// `v` has are kept as stored.
pub fn fill_missing(v: &mut serde_json::Value, defaults: &serde_json::Value) {
    let (Some(o), Some(d)) = (v.as_object_mut(), defaults.as_object()) else { return };
    for (k, dv) in d {
        match o.get_mut(k) {
            Some(x) => fill_missing(x, dv),
            None => {
                o.insert(k.clone(), dv.clone());
            }
        }
    }
}

pub fn save<C: CoreInit>(dir: &Path, net: &UraniumNet<C>, meta: &CheckpointMeta) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let mut store = BurnpackStore::from_file(dir.join(WEIGHTS_FILE)).overwrite(true);
    net.save_into(&mut store).map_err(|e| anyhow!("saving weights: {e:?}"))?;
    std::fs::write(dir.join(META_FILE), serde_json::to_vec_pretty(meta)?)?;
    Ok(())
}

pub fn read_meta(dir: &Path) -> Result<CheckpointMeta> {
    let text = std::fs::read(dir.join(META_FILE)).with_context(|| format!("reading {}", dir.join(META_FILE).display()))?;
    Ok(serde_json::from_slice(&text)?)
}

/// Rebuilds the net from `meta.json` and loads the weights. The caller passes the schema
/// versions it was built with; a mismatch is an error (032-R6).
pub fn load<C: CoreInit>(dir: &Path, columns: &ActionColumns, feature_version: u32, action_version: u32, device: &Device) -> Result<(UraniumNet<C>, CheckpointMeta)> {
    load_with(dir, columns, feature_version, action_version, device, |c| c)
}

/// `load` with the stored config adjusted before the net is built, for settings that have no
/// weights (e.g. 032 §2: dropout 0.1 for BC, 0 for RL).
pub fn load_with<C: CoreInit>(
    dir: &Path,
    columns: &ActionColumns,
    feature_version: u32,
    action_version: u32,
    device: &Device,
    adjust: impl FnOnce(UraniumNetConfig) -> UraniumNetConfig,
) -> Result<(UraniumNet<C>, CheckpointMeta)> {
    let mut meta = read_meta(dir)?;
    ensure!(meta.feature_version == feature_version, "feature schema {} != {}", meta.feature_version, feature_version);
    ensure!(meta.action_version == action_version, "action schema {} != {}", meta.action_version, action_version);
    let cfg = adjust(meta.net_config()?);
    meta.config = serde_json::to_value(&cfg)?;
    ensure!(cfg.core.kind == C::KIND, "checkpoint core {:?} != requested {:?}", cfg.core.kind, C::KIND);
    let mut net = UraniumNet::<C>::new(&cfg, columns, device);
    let mut store = BurnpackStore::from_file(dir.join(WEIGHTS_FILE));
    let result = net.load_from(&mut store).map_err(|e| anyhow!("loading weights: {e:?}"))?;
    ensure!(result.missing.is_empty(), "checkpoint is missing tensors: {:?}", result.missing);
    Ok((net, meta))
}
