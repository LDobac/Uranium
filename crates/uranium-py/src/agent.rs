//! Inference for game sessions (001-R6, 020-R11, 032-R14, AMB-000-4): the Python model thread
//! pulls batches from a [`Broker`] while the Rust agent server, scheduler and SC:R server run
//! their sessions on Rust threads with the GIL released.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Duration;

use numpy::{IntoPyArray, PyArrayMethods, PyReadonlyArray1, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use uranium_agent::broker::BatchOut;
use uranium_agent::infer::{HostState, InferOptions, InferenceHandle, SampleMode, SurrenderMode, checkpoint_id, resolve_checkpoint};
use uranium_agent::{PolicySpec, ServeOptions};

use crate::err;

fn sample_of(s: &str) -> PyResult<SampleMode> {
    match s {
        "argmax" => Ok(SampleMode::Argmax),
        "stochastic" => Ok(SampleMode::Stochastic),
        _ => Err(PyValueError::new_err(format!("sample {s:?}: expected argmax or stochastic"))),
    }
}

fn surrender_of(s: &str) -> PyResult<SurrenderMode> {
    match s {
        "off" => Ok(SurrenderMode::Off),
        "shadow" => Ok(SurrenderMode::Shadow),
        "on" => Ok(SurrenderMode::On),
        _ => Err(PyValueError::new_err(format!("surrender {s:?}: expected off, shadow or on"))),
    }
}

/// One inference server's queue (one checkpoint and sampling mode): sessions on Rust threads
/// send observations, the Python model thread calls `next` / `reply`.
#[pyclass(module = "uranium._core")]
pub struct Broker {
    inner: Mutex<uranium_agent::broker::Broker>,
    handle: InferenceHandle,
    dir: PathBuf,
    opts: InferOptions,
}

#[pymethods]
impl Broker {
    /// `checkpoint`: a folder (relative = under URANIUM_SHARED_DIR, 020-R11); batching defaults
    /// are placeholders until measured (020-R11 TBD(M6)).
    #[new]
    #[pyo3(signature = (checkpoint, sample = "argmax", max_batch = 32, max_wait_ms = 5, capacity = 128, pad_entities = 64, zero_last_failed = false, warmup_entities = 0))]
    #[allow(clippy::too_many_arguments)]
    fn new(checkpoint: PathBuf, sample: &str, max_batch: usize, max_wait_ms: u64, capacity: usize, pad_entities: usize, zero_last_failed: bool, warmup_entities: usize) -> PyResult<Self> {
        let dir = resolve_checkpoint(&checkpoint).map_err(err)?;
        let id = checkpoint_id(&dir).map_err(err)?;
        let opts = InferOptions { max_batch: max_batch.max(1), max_wait: Duration::from_millis(max_wait_ms), capacity, pad_entities, zero_last_failed, warmup_entities };
        let (b, handle) = uranium_agent::broker::Broker::new(id, sample_of(sample)?, opts).map_err(err)?;
        Ok(Self { inner: Mutex::new(b), handle, dir, opts })
    }

    /// The resolved checkpoint folder the model thread loads.
    #[getter]
    fn dir(&self) -> PathBuf {
        self.dir.clone()
    }

    /// 020-R5 policy version.
    #[getter]
    fn checkpoint_id(&self) -> String {
        self.handle.checkpoint_id.to_string()
    }

    #[getter]
    fn temp(&self) -> f32 {
        self.handle.sample.temp()
    }

    #[getter]
    fn capacity(&self) -> usize {
        self.opts.capacity
    }

    #[getter]
    fn pad_entities(&self) -> usize {
        self.opts.pad_entities
    }

    #[getter]
    fn warmup_entities(&self) -> usize {
        self.opts.warmup_entities
    }

    /// Every session handle is gone and nothing is waiting: the model thread can stop.
    fn closed(&self) -> bool {
        self.inner.lock().unwrap().closed()
    }

    /// Waits up to `timeout_ms` for work: `None`, or a dict with `resets` (slots to zero),
    /// `states` (slots whose state to send with `reply_states`) and `batch` (`None` or a dict of
    /// `slots`, `n` and the observation arrays).
    fn next<'py>(&self, py: Python<'py>, timeout_ms: u64) -> PyResult<Option<Bound<'py, PyDict>>> {
        let inner = &self.inner;
        let w = py.detach(move || inner.lock().unwrap().next(Duration::from_millis(timeout_ms))).map_err(err)?;
        let Some(w) = w else { return Ok(None) };
        let d = PyDict::new(py);
        d.set_item("resets", w.resets)?;
        d.set_item("states", w.states)?;
        match w.batch {
            None => d.set_item("batch", py.None())?,
            Some(b) => {
                let bd = PyDict::new(py);
                bd.set_item("slots", b.slots)?;
                bd.set_item("n", b.obs.n)?;
                crate::put_obs(py, &bd, b.obs)?;
                d.set_item("batch", bd)?;
            }
        }
        Ok(Some(d))
    }

    /// Answers the last batch: int64 `action`, `delay`, `queued`, `selected` [rows*S],
    /// `target_unit`, `target_loc`; float32 `log_prob`, `head_log_prob` [rows*6], `value`
    /// [rows*4], `win_open` (or None); all flat.
    #[pyo3(signature = (action, delay, queued, selected, target_unit, target_loc, log_prob, head_log_prob, value, win_open = None))]
    #[allow(clippy::too_many_arguments)]
    fn reply(
        &self,
        py: Python<'_>,
        action: PyReadonlyArray1<'_, i64>,
        delay: PyReadonlyArray1<'_, i64>,
        queued: PyReadonlyArray1<'_, i64>,
        selected: PyReadonlyArray1<'_, i64>,
        target_unit: PyReadonlyArray1<'_, i64>,
        target_loc: PyReadonlyArray1<'_, i64>,
        log_prob: PyReadonlyArray1<'_, f32>,
        head_log_prob: PyReadonlyArray1<'_, f32>,
        value: PyReadonlyArray1<'_, f32>,
        win_open: Option<PyReadonlyArray1<'_, f32>>,
    ) -> PyResult<()> {
        let s = |a: &PyReadonlyArray1<'_, i64>| a.as_slice().map(|x| x.to_vec()).map_err(|e| PyValueError::new_err(e.to_string()));
        let f = |a: &PyReadonlyArray1<'_, f32>| a.as_slice().map(|x| x.to_vec()).map_err(|e| PyValueError::new_err(e.to_string()));
        let (a, d, q, sel, tu, tl) = (s(&action)?, s(&delay)?, s(&queued)?, s(&selected)?, s(&target_unit)?, s(&target_loc)?);
        let (lp, hl, v) = (f(&log_prob)?, f(&head_log_prob)?, f(&value)?);
        let wo = win_open.as_ref().map(f).transpose()?;
        let inner = &self.inner;
        py.detach(move || {
            let out = BatchOut { action: &a, delay: &d, queued: &q, selected: &sel, target_unit: &tu, target_loc: &tl, log_prob: &lp, head_log_prob: &hl, value: &v, win_open: wo.as_deref() };
            inner.lock().unwrap().reply(out)
        })
        .map_err(err)
    }

    /// Answers the last work's state requests: per slot, a list of float32 arrays (a slot's Core
    /// state tensors, batch dimension 1).
    fn reply_states(&self, states: Vec<Vec<PyReadonlyArray1<'_, f32>>>, shapes: Vec<Vec<Vec<usize>>>) -> PyResult<()> {
        let mut out = Vec::with_capacity(states.len());
        for (parts, shp) in states.iter().zip(shapes) {
            let mut h = HostState::default();
            for (p, s) in parts.iter().zip(shp) {
                let v = p.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?.to_vec();
                if v.len() != s.iter().product::<usize>() || p.len() != v.len() {
                    return Err(PyValueError::new_err(format!("state part of {} values for shape {s:?}", v.len())));
                }
                h.parts.push((s, v));
            }
            out.push(h);
        }
        self.inner.lock().unwrap().reply_states(out).map_err(err)
    }

    /// 032-R16: weights version reported with later decisions (after the model thread swapped).
    fn set_version(&self, v: u64) {
        self.inner.lock().unwrap().set_version(v);
    }

    /// The model failed: the waiting sessions get `msg` as their error.
    fn fail(&self, msg: &str) {
        self.inner.lock().unwrap().fail(msg);
    }

    /// The model thread is stopping: everything still queued is refused.
    fn shutdown(&self, msg: &str) -> PyResult<()> {
        self.inner.lock().unwrap().shutdown(msg).map_err(err)
    }
}

impl Broker {
    fn session_handle(&self, surrender: &str, chat: &str) -> PyResult<InferenceHandle> {
        if chat.len() > uranium_proto::MAX_SURRENDER_CHAT {
            return Err(PyValueError::new_err(format!("surrender chat is {} bytes, at most {} (020-R14)", chat.len(), uranium_proto::MAX_SURRENDER_CHAT)));
        }
        Ok(self.handle.clone().with_surrender(surrender_of(surrender)?, chat))
    }
}

/// The agent server (001-R6, 020-R11; SC:R live play 012-R19 with `surrender="on"`): bridges
/// connect to `listen`, one record per session goes to `out`. `broker`: the model of every
/// session (None = the scripted T0 bot). Runs until `sessions` sessions finished (0 = forever).
#[pyfunction]
#[pyo3(signature = (listen, out, broker = None, surrender = "shadow", surrender_chat = uranium_agent::infer::DEFAULT_CHAT, respond_delay_ms = 0, sessions = 0))]
#[allow(clippy::too_many_arguments)]
pub fn serve(py: Python<'_>, listen: String, out: PathBuf, broker: Option<PyRef<'_, Broker>>, surrender: &str, surrender_chat: &str, respond_delay_ms: u64, sessions: usize) -> PyResult<()> {
    let default = match &broker {
        Some(b) => PolicySpec::Model(b.session_handle(surrender, surrender_chat)?),
        None => PolicySpec::ScriptedT0,
    };
    drop(broker);
    let opts = ServeOptions { delay_ms: respond_delay_ms, max_sessions: sessions, ..ServeOptions::new(default) };
    py.detach(move || uranium_agent::serve_addr(&listen, opts, out)).map_err(err)
}

/// (checkpoint as written in the plan, sampling mode) of every model the plan's players use
/// (020-R11): one `Broker` + model thread each, then `run_plan`.
#[pyfunction]
pub fn plan_models(plan: PathBuf) -> PyResult<Vec<(String, String)>> {
    let p = uranium_runner::plan::load_plan(&plan).map_err(err)?;
    Ok(uranium_runner::plan_models(&p).into_iter().map(|(c, s)| (c, format!("{s:?}").to_lowercase())).collect())
}

/// Plays a matchup plan (020) with the given brokers: a list of (checkpoint, sample, broker),
/// one per `plan_models` entry. The plan's surrender mode and last_failed setting apply.
#[pyfunction]
#[pyo3(signature = (plan, models, out = None))]
pub fn run_plan(py: Python<'_>, plan: PathBuf, models: &Bound<'_, PyList>, out: Option<PathBuf>) -> PyResult<()> {
    let mut handles = HashMap::new();
    for item in models.iter() {
        let (ckpt, sample, b): (String, String, PyRef<'_, Broker>) = item.extract()?;
        handles.insert((ckpt, sample_of(&sample)?), b.handle.clone());
    }
    let p = uranium_runner::plan::load_plan(&plan).map_err(err)?;
    let policies = uranium_runner::plan_policies(&p, &handles).map_err(err)?;
    drop(handles);
    py.detach(move || uranium_runner::run_plan(&plan, out, policies)).map_err(err)
}

/// Tests (032-R14, 031-R4): `sessions` model sessions playing the synthetic frames of
/// `uranium_agent::testkit` at once; per session the (command count, advance) of each step and
/// the session stats as JSON.
#[pyfunction]
pub fn play_sessions(py: Python<'_>, broker: PyRef<'_, Broker>, sessions: usize, steps: usize) -> PyResult<Vec<(Vec<(usize, u32)>, String)>> {
    let h = broker.handle.clone();
    drop(broker);
    let res = py.detach(move || uranium_agent::testkit::play_parallel(&h, sessions, steps)).map_err(err)?;
    res.into_iter()
        .map(|(acts, stats)| Ok((acts.into_iter().map(|(c, a)| (c.len(), a)).collect(), serde_json::to_string(&stats).map_err(|e| err(e.into()))?)))
        .collect()
}

/// Tests (032-R16): a fresh slot's Core state, then after one and two decisions, as
/// [(shape, values)] per state.
#[pyfunction]
pub fn slot_states<'py>(py: Python<'py>, broker: PyRef<'py, Broker>) -> PyResult<Vec<Vec<(Vec<usize>, Bound<'py, numpy::PyArray1<f32>>)>>> {
    let h = broker.handle.clone();
    drop(broker);
    let states = py.detach(move || uranium_agent::testkit::slot_states(&h)).map_err(err)?;
    Ok(states.into_iter().map(|s| s.parts.into_iter().map(|(shape, v)| (shape, v.into_pyarray(py))).collect()).collect())
}
