//! `uranium._core`: the Rust side of the PyTorch trainer (001, AMB-000-4). The BC data side
//! (041: shard order, data RNG, lanes, batches, prefetch, host collate) runs here with the GIL
//! released and hands each lane group to Python as numpy arrays (ownership moves, no copy), so
//! the Python trainer reads exactly the batches the Rust code defines.

mod agent;
mod rl;

use std::path::PathBuf;
use std::sync::Arc;

/// 041-R16 (2026-10-24): the BC feed allocates and frees GBs of large buffers per update (a
/// ~1.9 MB plane per step, ~240 MB per collated lane group). The Windows heap returns blocks that
/// large to the OS on free, so every update page-faulted them in again (~0.9 M faults/s on the
/// CUDA host) and the feed thread stalled behind the training process. mimalloc keeps freed
/// memory for reuse. Allocation only: the same bytes.
#[cfg(windows)]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use numpy::{IntoPyArray, PyArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTracebackMethods};
use uranium_data::bc::{self, BcConfig, BcStage, FeedState, Prefetch};
use uranium_data::bc_run::{self, BcExit, BcOut, BcRunArgs, BcTrainer, Start};
use uranium_store::storage::{self, TakeOver};
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

/// The observation arrays of a host collate in the 030 §1 shapes (ownership moves, no copy).
pub(crate) fn put_obs(py: Python<'_>, d: &Bound<'_, PyDict>, o: uranium_data::collate::HostObsBatch) -> PyResult<()> {
    use uranium_data::collate::{ENTITY_CAT, ENTITY_CONT, MAP, SPATIAL, Z_BUILD};
    let (r, n) = (o.b, o.n);
    macro_rules! put {
        ($k:expr, $v:expr, $s:expr) => {
            d.set_item($k, $v.into_pyarray(py).reshape($s)?)?
        };
    }
    put!("entity_cat", o.entity_cat, vec![r, n, ENTITY_CAT]);
    put!("entity_cont", o.entity_cont, vec![r, n, ENTITY_CONT]);
    put!("entity_mask", o.entity_mask, vec![r, n]);
    put!("entity_cell", o.entity_cell, vec![r, n]);
    put!("entity_pos", o.entity_pos, vec![r, n, 2]);
    put!("spatial", o.spatial, vec![r, SPATIAL, MAP, MAP]);
    put!("scalar", o.scalar, vec![r, o.fs]);
    put!("scalar_cat", o.scalar_cat, vec![r, 3]);
    put!("z_build", o.z_build, vec![r, Z_BUILD]);
    put!("z_units", o.z_units, vec![r, o.u]);
    put!("z_present", o.z_present, vec![r]);
    put!("type_mask", o.type_mask, vec![r, o.a]);
    put!("unit_cap", o.unit_cap, vec![r, n, o.k]);
    put!("loc_class_mask", o.loc_class_mask, vec![r, o.l, MAP, MAP]);
    put!("is_building", o.is_building, vec![r, n]);
    Ok(())
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

    /// Steps per lane (0 = a drained lane, evaluation only).
    fn lane_steps(&self) -> Vec<usize> {
        self.inner.lane_steps()
    }

    /// `bc-eval` label facts per row of `group(start, end, t, ..)`: (has target location,
    /// full selection length) as numpy arrays.
    fn label_extras<'py>(&self, py: Python<'py>, start: usize, end: usize, t: usize) -> (Bound<'py, numpy::PyArray1<bool>>, Bound<'py, numpy::PyArray1<i32>>) {
        let (a, b) = self.inner.label_extras(start..end, t);
        (a.into_pyarray(py), b.into_pyarray(py))
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
        let (l, p) = (g.labels, g.privileged);
        let (r, n) = (g.obs.b, g.obs.n);
        put_obs(py, &d, g.obs)?;
        macro_rules! put {
            ($k:expr, $v:expr, $s:expr) => {
                d.set_item($k, $v.into_pyarray(py).reshape($s)?)?
            };
        }
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

/// `bc-eval` games and the 041-AC4 surrender replay ([`uranium_data::eval::EvalStream`]).
#[pyclass(module = "uranium._core", unsendable)]
struct EvalStream {
    inner: uranium_data::eval::EvalStream,
}

#[pymethods]
impl EvalStream {
    /// `split`: train / val / test; `games`: N games by the fixed-seed shuffle (032-R17 ②), else
    /// every game; `has_open`: replay the surrender check with the win_open head (032-R18).
    #[new]
    #[pyo3(signature = (config, dirs, split = "test", games = None, seed = 1, has_open = false, maps_dir = None, synthetic_maps = false))]
    #[allow(clippy::too_many_arguments)]
    fn new(py: Python<'_>, config: PathBuf, dirs: Vec<PathBuf>, split: &str, games: Option<usize>, seed: u64, has_open: bool, maps_dir: Option<PathBuf>, synthetic_maps: bool) -> PyResult<Self> {
        use uranium_data::split::Split;
        let split = match split {
            "train" => Split::Train,
            "val" => Split::Val,
            "test" => Split::Test,
            _ => return Err(PyValueError::new_err(format!("split {split:?}: expected train, val or test"))),
        };
        let cfg = BcConfig::load(&config).map_err(err)?;
        let maps = match (synthetic_maps, maps_dir) {
            (true, _) => Some(Arc::new(MapStore::synthetic())),
            (false, Some(d)) => Some(Arc::new(MapStore::dir(d))),
            (false, None) => None,
        };
        let sel = uranium_data::eval::EvalSelection { split, games, seed };
        let inner = py.detach(move || uranium_data::eval::EvalStream::new(&dirs, &cfg, sel, maps, has_open)).map_err(err)?;
        Ok(Self { inner })
    }

    /// The next batch, `None` once every lane is drained.
    fn next(&mut self, py: Python<'_>) -> PyResult<Option<Batch>> {
        let inner = &mut self.inner;
        let b = py.detach(move || inner.next()).map_err(err)?;
        Ok(b.map(|inner| Batch { inner }))
    }

    /// Feeds the win_open output of lanes `start..end` (rows lane * t + step) to the surrender
    /// replay and the win_open MSE.
    fn decide(&mut self, batch: &Batch, start: usize, end: usize, t: usize, open: numpy::PyReadonlyArray1<'_, f32>) -> PyResult<()> {
        let open = open.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
        if open.len() != (end - start) * t {
            return Err(PyValueError::new_err(format!("win_open has {} rows, expected {}", open.len(), (end - start) * t)));
        }
        self.inner.decide(&batch.inner, start..end, t, open);
        Ok(())
    }

    #[getter]
    fn games(&self) -> usize {
        self.inner.games()
    }

    /// JSON {win_open_mse, surrender} (null without the head); ends the open games.
    fn finish(&mut self) -> PyResult<String> {
        let (mse, sur) = self.inner.finish();
        serde_json::to_string(&serde_json::json!({ "win_open_mse": mse, "surrender": sur })).map_err(|e| err(e.into()))
    }
}

/// A Python exception as an error of the Rust loop, with its traceback.
fn py_error(py: Python<'_>, e: PyErr) -> anyhow::Error {
    let tb = e.traceback(py).and_then(|t| t.format().ok()).unwrap_or_default();
    anyhow::anyhow!("{tb}{e}")
}

/// The Python trainer object (`uranium.bc.trainer`) behind [`BcTrainer`].
struct PyTrainer(Py<PyAny>);

impl BcTrainer for PyTrainer {
    fn train_batch(&mut self, batch: bc::Batch, lr: f64, step: usize) -> anyhow::Result<serde_json::Value> {
        Python::attach(|py| {
            let b = Py::new(py, Batch { inner: batch }).map_err(|e| py_error(py, e))?;
            let out = self.0.call_method1(py, "train_batch", (b, lr, step)).map_err(|e| py_error(py, e))?;
            let js: String = out.extract(py).map_err(|e| py_error(py, e))?;
            Ok(serde_json::from_str(&js)?)
        })
    }

    fn save(&mut self, dir: &std::path::Path, train: &serde_json::Value) -> anyhow::Result<()> {
        Python::attach(|py| {
            self.0.call_method1(py, "save", (dir.to_path_buf(), train.to_string())).map_err(|e| py_error(py, e))?;
            Ok(())
        })
    }
}

/// 041 / 061: one BC stage with the Rust run loop ([`bc_run::run`]): data thread, schedule,
/// checkpoints, run lock, heartbeat, push, yield and resume. `make(config_json, start, dir,
/// entity_chunk)` builds the trainer once the start point is known (`start`: "new", "resume" or
/// "init"; `dir` the checkpoint for the last two). Returns how the run ended: "done", "stopped",
/// "lock_lost", "yielded" (exit 76) or "busy" (the lock is not ours, exit 75).
#[pyfunction]
#[pyo3(signature = (make, config, stage = "s1", out = None, run = None, take_over = false, take_over_if = None, datasets = Vec::new(), shards = Vec::new(), steps = None, resume = None, init = None, backend = "cpu", precision = "f32", core = "gtrxl", maps_dir = None, synthetic_maps = false))]
#[allow(clippy::too_many_arguments)]
fn run_bc(
    py: Python<'_>,
    make: Py<PyAny>,
    config: PathBuf,
    stage: &str,
    out: Option<PathBuf>,
    run: Option<String>,
    take_over: bool,
    take_over_if: Option<String>,
    datasets: Vec<PathBuf>,
    shards: Vec<PathBuf>,
    steps: Option<usize>,
    resume: Option<PathBuf>,
    init: Option<PathBuf>,
    backend: &str,
    precision: &str,
    core: &str,
    maps_dir: Option<PathBuf>,
    synthetic_maps: bool,
) -> PyResult<&'static str> {
    let stage = stage_of(stage)?;
    let out = match (out, run) {
        (Some(o), None) => BcOut::Local(o),
        (None, Some(id)) => {
            let take = match (take_over, take_over_if) {
                (_, Some(seen)) => TakeOver::IfUnchanged(serde_json::from_str(&seen).map_err(|e| PyValueError::new_err(format!("take_over_if: {e}")))?),
                (true, None) => TakeOver::Force,
                (false, None) => TakeOver::No,
            };
            BcOut::Run { id, take }
        }
        _ => return Err(PyValueError::new_err("pass exactly one of out or run")),
    };
    if resume.is_some() && (init.is_some() || matches!(out, BcOut::Run { .. })) {
        return Err(PyValueError::new_err("resume is for local runs without init (a run resumes from LATEST)"));
    }
    let stop = uranium_store::stop_state().unwrap_or_default();
    let maps = match (synthetic_maps, maps_dir) {
        (true, _) => Some(Arc::new(MapStore::synthetic())),
        (false, Some(d)) => Some(Arc::new(MapStore::dir(d))),
        (false, None) => None,
    };
    let (backend, precision, core) = (backend.to_string(), precision.to_string(), core.to_string());
    let res = py.detach(move || -> anyhow::Result<BcExit> {
        let dirs = storage::dataset_dirs(&datasets, &shards)?;
        let args = BcRunArgs { dirs, out, config, stage, stop_at: steps, resume, init, backend, precision, core, stop, maps };
        bc_run::run(args, |cfg, start, entity_chunk| {
            let cfg_json = serde_json::to_string(cfg)?;
            let (kind, dir) = match start {
                Start::New => ("new", None),
                Start::Resume(d) => ("resume", Some(d.to_path_buf())),
                Start::Init(d) => ("init", Some(d.to_path_buf())),
            };
            Python::attach(|py| {
                let t = make.call1(py, (cfg_json, kind, dir, entity_chunk)).map_err(|e| py_error(py, e))?;
                Ok(Box::new(PyTrainer(t)) as Box<dyn BcTrainer>)
            })
        })
    });
    match res {
        Ok(BcExit::Done) => Ok("done"),
        Ok(BcExit::Stopped) => Ok("stopped"),
        Ok(BcExit::LockLost) => Ok("lock_lost"),
        Ok(BcExit::Yielded) => Ok("yielded"),
        Err(e) if e.downcast_ref::<storage::LockBusy>().is_some() => {
            uranium_data::elog!("{e:#}");
            Ok("busy")
        }
        // Ctrl+C during the dataset pull: no lock yet, a clean stop (042-R10).
        Err(e) if e.downcast_ref::<uranium_store::Interrupted>().is_some() => {
            uranium_data::elog!("[bc] {e:#}");
            Ok("stopped")
        }
        Err(e) => Err(err(e)),
    }
}

/// 041-R9 data folders: `datasets` pulled or verified (061-R11-1, R12), then `shards` as they are.
#[pyfunction]
fn dataset_dirs(py: Python<'_>, datasets: Vec<PathBuf>, shards: Vec<PathBuf>) -> PyResult<Vec<PathBuf>> {
    py.detach(move || storage::dataset_dirs(&datasets, &shards)).map_err(err)
}

/// Checkpoint folder of `dir`: itself, or the one `LATEST` names in a 061 run folder.
#[pyfunction]
fn resolve_checkpoint(dir: PathBuf) -> PyResult<PathBuf> {
    bc_run::resolve_checkpoint(&dir).map_err(err)
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

/// 034 action table columns (032 §5) as JSON: the model's mask gather constants.
#[pyfunction]
fn action_columns() -> PyResult<String> {
    let c = uranium_data::table::ActionColumns::current().map_err(err)?;
    serde_json::to_string(&c).map_err(|e| err(e.into()))
}

/// Feature schema dimensions (030) as JSON: capability and location class counts, widths.
#[pyfunction]
fn feature_dims() -> PyResult<String> {
    use uranium_data::collate as c;
    let d = uranium_env_dims();
    serde_json::to_string(&serde_json::json!({
        "entity_cat": c::ENTITY_CAT, "entity_cont": c::ENTITY_CONT, "spatial": c::SPATIAL, "map": c::MAP, "z_build": c::Z_BUILD,
        "actions": d.0, "scalar": d.1, "z_units": d.2, "caps": d.3, "loc_classes": d.4, "privileged": d.5,
        "delay_buckets": uranium_data::table::DELAY_BUCKETS, "n_max": uranium_data::table::N_MAX, "feature_version": uranium_data::table::FEATURE_SCHEMA_VERSION, "action_version": uranium_data::table::ACTION_SCHEMA_VERSION,
    }))
    .map_err(|e| err(e.into()))
}

fn uranium_env_dims() -> (usize, usize, usize, usize, usize, usize) {
    let d = uranium_data::table::feature_dims();
    (d.actions, d.scalar(), d.z_units(), d.caps(), d.loc_classes(), d.privileged())
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
    m.add_class::<EvalStream>()?;
    m.add_class::<agent::Broker>()?;
    m.add_function(wrap_pyfunction!(agent::serve, m)?)?;
    m.add_function(wrap_pyfunction!(agent::plan_models, m)?)?;
    m.add_function(wrap_pyfunction!(agent::run_plan, m)?)?;
    m.add_function(wrap_pyfunction!(agent::play_sessions, m)?)?;
    m.add_function(wrap_pyfunction!(agent::slot_states, m)?)?;
    m.add_class::<rl::SegmentBatch>()?;
    m.add_function(wrap_pyfunction!(rl::run_rl, m)?)?;
    m.add_function(wrap_pyfunction!(rl::vtrace, m)?)?;
    m.add_function(wrap_pyfunction!(rl::upgo, m)?)?;
    m.add_function(wrap_pyfunction!(rl::gae, m)?)?;
    m.add_function(wrap_pyfunction!(rl::test_segments, m)?)?;
    m.add_function(wrap_pyfunction!(rl::rl_config, m)?)?;
    m.add_function(wrap_pyfunction!(bc_config, m)?)?;
    m.add_function(wrap_pyfunction!(lr_at, m)?)?;
    m.add_function(wrap_pyfunction!(dataset_fingerprint, m)?)?;
    m.add_function(wrap_pyfunction!(epoch_table, m)?)?;
    m.add_function(wrap_pyfunction!(group_fields, m)?)?;
    m.add_function(wrap_pyfunction!(action_columns, m)?)?;
    m.add_function(wrap_pyfunction!(feature_dims, m)?)?;
    m.add_function(wrap_pyfunction!(install_stop_handler, m)?)?;
    m.add_function(wrap_pyfunction!(stop_requested, m)?)?;
    m.add_function(wrap_pyfunction!(write_test_shards, m)?)?;
    m.add_function(wrap_pyfunction!(run_bc, m)?)?;
    m.add_function(wrap_pyfunction!(resolve_checkpoint, m)?)?;
    m.add_function(wrap_pyfunction!(dataset_dirs, m)?)?;
    Ok(())
}
