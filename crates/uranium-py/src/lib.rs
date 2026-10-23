//! `uranium._core`: the Rust side of the PyTorch trainer (001, AMB-000-4). The BC data side
//! (041: shard order, data RNG, lanes, batches, prefetch, host collate) runs here with the GIL
//! released and hands each lane group to Python as numpy arrays (ownership moves, no copy), so
//! the Python trainer reads exactly the batches the Rust code defines.

use std::path::PathBuf;
use std::sync::Arc;

use numpy::{IntoPyArray, PyArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use uranium_data::bc::{self, BcConfig, BcStage, FeedState, Prefetch};
use uranium_data::data::{MapStore, ShardSource};

fn err(e: anyhow::Error) -> PyErr {
    PyRuntimeError::new_err(format!("{e:#}"))
}

fn stage_of(s: &str) -> PyResult<BcStage> {
    match s.to_ascii_lowercase().as_str() {
        "s1" => Ok(BcStage::S1),
        "s2" => Ok(BcStage::S2),
        "ablation_a" | "ablationa" => Ok(BcStage::AblationA),
        _ => Err(PyValueError::new_err(format!("stage {s:?}: expected s1, s2 or ablation_a"))),
    }
}

fn config(path: &PathBuf, stage: &str) -> PyResult<BcConfig> {
    Ok(BcConfig::load(path).map_err(err)?.for_stage(stage_of(stage)?))
}

fn source(dirs: &[PathBuf], stage: BcStage, cfg: &BcConfig) -> anyhow::Result<ShardSource> {
    Ok(ShardSource::from_dirs(dirs)?.select(stage.selection()).shuffled(cfg.data.seed))
}

/// BC batches from shard folders (041), built ahead on a Rust thread (041-R16).
#[pyclass(module = "uranium._core", unsendable)]
struct Feeder {
    inner: Prefetch,
    files: String,
}

#[pymethods]
impl Feeder {
    /// `config`: `config/bc.toml`; `stage`: s1 / s2 / ablation_a (its `[steps]` apply, 041-R14).
    /// `prefetch` / `pack` default to `URANIUM_BC_PREFETCH` / `URANIUM_BC_PACK_BATCH` (041-R16,
    /// R18). Maps: `maps_dir`, else `URANIUM_MAPS_DIR`; `synthetic_maps` for test games only.
    /// `restore`: a `FeedState` JSON from a checkpoint (061-R10).
    #[new]
    #[pyo3(signature = (config, dirs, stage = "s1", prefetch = None, pack = None, maps_dir = None, synthetic_maps = false, check_maps = true, restore = None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        config: PathBuf,
        dirs: Vec<PathBuf>,
        stage: &str,
        prefetch: Option<usize>,
        pack: Option<bool>,
        maps_dir: Option<PathBuf>,
        synthetic_maps: bool,
        check_maps: bool,
        restore: Option<String>,
    ) -> PyResult<Self> {
        let st = stage_of(stage)?;
        let cfg = self::config(&config, stage)?;
        py.detach(move || -> anyhow::Result<Self> {
            let maps = Arc::new(match (synthetic_maps, maps_dir) {
                (true, _) => MapStore::synthetic(),
                (false, Some(d)) => MapStore::dir(d),
                (false, None) => MapStore::from_env(),
            });
            let mut src = source(&dirs, st, &cfg)?;
            if check_maps && !synthetic_maps {
                uranium_data::data::check_maps(&src, Some(st.selection()), &maps)?;
            }
            let files = src.fingerprint();
            let mut feeder = bc::Feeder::new(&cfg).with_maps(maps);
            feeder.set_pack(match pack {
                Some(p) => p,
                None => bc::pack_batch()?,
            });
            if let Some(js) = restore {
                let state: FeedState = serde_json::from_str(&js)?;
                feeder.restore(&state, &mut src)?;
            }
            let depth = match prefetch {
                Some(d) => d,
                None => bc::prefetch_depth()?,
            };
            Ok(Self { inner: Prefetch::start(feeder, src, depth), files })
        })
        .map_err(err)
    }

    /// The next batch, `None` at the end of the epoch (041-R14).
    fn next(&mut self, py: Python<'_>) -> PyResult<Option<Batch>> {
        let inner = &mut self.inner;
        let b = py.detach(move || inner.next()).map_err(err)?;
        Ok(b.map(|inner| Batch { inner }))
    }

    /// Fingerprint of the selected shard files the data position refers to (061-R10).
    #[getter]
    fn files(&self) -> String {
        self.files.clone()
    }
}

/// One BC batch: B lanes x T steps (041-R2) and the data state after it (061-R10).
#[pyclass(module = "uranium._core", frozen)]
struct Batch {
    inner: bc::Batch,
}

#[pymethods]
impl Batch {
    #[getter]
    fn lanes(&self) -> usize {
        self.inner.lanes()
    }

    /// Policy-loss divisor of every group: usable label steps of the whole batch (041-R3).
    #[getter]
    fn valid(&self) -> f32 {
        self.inner.valid()
    }

    /// Value-loss divisor: usable label steps of decided games (041-R7 / A-3).
    #[getter]
    fn value_valid(&self) -> f32 {
        self.inner.value_valid()
    }

    /// `FeedState` JSON after this batch (saved in checkpoints, 061-R10).
    fn state_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner.state).map_err(|e| err(e.into()))
    }

    /// Lanes `start..end` collated for one forward/backward (`Batch::group`): a dict of numpy
    /// arrays in the 030 §1 shapes, rows b-major (row = lane * t + step), plus `b`, `t`, `n`.
    fn group<'py>(&self, py: Python<'py>, start: usize, end: usize, t: usize, pad: usize, max_select: usize) -> PyResult<Bound<'py, PyDict>> {
        if start >= end || end > self.inner.lanes() {
            return Err(PyValueError::new_err(format!("lanes {start}..{end} of {}", self.inner.lanes())));
        }
        let inner = &self.inner;
        let g = py.detach(move || inner.group(start..end, t, pad, max_select));
        let d = PyDict::new(py);
        let (o, l, p) = (g.obs, g.labels, g.privileged);
        let (r, n) = (o.b, o.n);
        macro_rules! put {
            ($k:expr, $v:expr, $s:expr) => {
                d.set_item($k, $v.into_pyarray(py).reshape($s)?)?
            };
        }
        put!("entity_cat", o.entity_cat, vec![r, n, uranium_data::collate::ENTITY_CAT]);
        put!("entity_cont", o.entity_cont, vec![r, n, uranium_data::collate::ENTITY_CONT]);
        put!("entity_mask", o.entity_mask, vec![r, n]);
        put!("entity_cell", o.entity_cell, vec![r, n]);
        put!("entity_pos", o.entity_pos, vec![r, n, 2]);
        put!("spatial", o.spatial, vec![r, uranium_data::collate::SPATIAL, uranium_data::collate::MAP, uranium_data::collate::MAP]);
        put!("scalar", o.scalar, vec![r, o.fs]);
        put!("scalar_cat", o.scalar_cat, vec![r, 3]);
        put!("z_build", o.z_build, vec![r, uranium_data::collate::Z_BUILD]);
        put!("z_units", o.z_units, vec![r, o.u]);
        put!("z_present", o.z_present, vec![r]);
        put!("type_mask", o.type_mask, vec![r, o.a]);
        put!("unit_cap", o.unit_cap, vec![r, n, o.k]);
        put!("loc_class_mask", o.loc_class_mask, vec![r, o.l, uranium_data::collate::MAP, uranium_data::collate::MAP]);
        put!("is_building", o.is_building, vec![r, n]);
        put!("action", l.action, vec![r]);
        put!("delay", l.delay, vec![r]);
        put!("queued", l.queued, vec![r]);
        put!("selected", l.selected, vec![r, l.s]);
        put!("target_unit", l.target_unit, vec![r]);
        put!("target_loc", l.target_loc, vec![r]);
        put!("privileged", p.summary, vec![r, p.p]);
        put!("weights", g.weights, vec![r]);
        put!("results", g.results, vec![r]);
        put!("value_w", g.value_w, vec![r]);
        put!("keep", g.keep, vec![g.b]);
        d.set_item("b", g.b)?;
        d.set_item("t", g.t)?;
        d.set_item("n", n)?;
        Ok(d)
    }
}

/// Effective config of `stage` as JSON and its fingerprint (B-3).
#[pyfunction]
#[pyo3(signature = (config, stage = "s1"))]
fn bc_config(config: PathBuf, stage: &str) -> PyResult<(String, String)> {
    let cfg = self::config(&config, stage)?;
    Ok((serde_json::to_string(&cfg).map_err(|e| err(e.into()))?, cfg.fingerprint()))
}

/// Learning rate at `step` (041-R3, the reference for the Python schedule).
#[pyfunction]
#[pyo3(signature = (config, stage, step, scale = 1.0))]
fn lr_at(config: PathBuf, stage: &str, step: usize, scale: f64) -> PyResult<f64> {
    Ok(self::config(&config, stage)?.lr_at(step, scale))
}

/// [`uranium_data::data::dataset_fingerprint`] of the files `stage` reads (B-3).
#[pyfunction]
#[pyo3(signature = (config, dirs, stage = "s1"))]
fn dataset_fingerprint(py: Python<'_>, config: PathBuf, dirs: Vec<PathBuf>, stage: &str) -> PyResult<String> {
    let st = stage_of(stage)?;
    let cfg = self::config(&config, stage)?;
    py.detach(move || uranium_data::data::dataset_fingerprint(&source(&dirs, st, &cfg)?)).map_err(err)
}

/// One-epoch updates per stage as JSON (041-R13, the same table as `uranium-train bc-epoch`).
#[pyfunction]
fn epoch_table(py: Python<'_>, config: PathBuf, dirs: Vec<PathBuf>) -> PyResult<String> {
    let cfg = BcConfig::load(&config).map_err(err)?;
    let t = py.detach(move || bc::epoch_table(&dirs, &cfg)).map_err(err)?;
    serde_json::to_string(&t).map_err(|e| err(e.into()))
}

/// Field order of the lane-group digest (`uranium-train feed-digest`).
#[pyfunction]
fn group_fields() -> Vec<&'static str> {
    bc::GROUP_FIELDS.to_vec()
}

/// 042-R10 / 060-R6: the process's Ctrl+C handler (first: clean stop, second: exit 130).
#[pyfunction]
fn install_stop_handler(tag: String) -> PyResult<()> {
    let tag: &'static str = Box::leak(tag.into_boxed_str());
    uranium_store::install_stop_handler(tag).map(|_| ()).map_err(err)
}

/// Whether a clean stop was requested (Ctrl+C or the watchdog's SIGINT).
#[pyfunction]
fn stop_requested() -> bool {
    uranium_store::stop_requested()
}

/// Tests: writes synthetic shards (`uranium_data::testkit`) under `root`: `kind` is "varied"
/// (15 games of 1..16 steps), "mixed" (games with different entity counts and void labels).
#[pyfunction]
#[pyo3(signature = (root, kind = "varied", per_file = 4))]
fn write_test_shards(root: PathBuf, kind: &str, per_file: usize) -> PyResult<()> {
    use uranium_data::testkit as tk;
    let games = match kind {
        "varied" => tk::varied_games(),
        "mixed" => vec![tk::game(), tk::game_more_units(5, 2), tk::game_with_invalid(9, 3), tk::game_more_units(16, 4), tk::game_len(7, 5)],
        _ => return Err(PyValueError::new_err(format!("kind {kind:?}: expected varied or mixed"))),
    };
    tk::write_shards(&root, &games, per_file).map_err(err)
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Feeder>()?;
    m.add_class::<Batch>()?;
    m.add_function(wrap_pyfunction!(bc_config, m)?)?;
    m.add_function(wrap_pyfunction!(lr_at, m)?)?;
    m.add_function(wrap_pyfunction!(dataset_fingerprint, m)?)?;
    m.add_function(wrap_pyfunction!(epoch_table, m)?)?;
    m.add_function(wrap_pyfunction!(group_fields, m)?)?;
    m.add_function(wrap_pyfunction!(install_stop_handler, m)?)?;
    m.add_function(wrap_pyfunction!(stop_requested, m)?)?;
    m.add_function(wrap_pyfunction!(write_test_shards, m)?)?;
    Ok(())
}
