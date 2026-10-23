//! Reinforcement learning (042, AMB-000-4): the Rust run loop (`uranium_league::run`: league,
//! games, segment queues, snapshots, checkpoints, 061 locks) drives the PyTorch learner object;
//! segment batches reach it as numpy lane groups.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use numpy::{IntoPyArray, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use uranium_agent::infer::InferenceHandle;
use uranium_league::config::RlConfig;
use uranium_league::learner::{Algo, RlLearners, SnapshotServers, UpdateStats};
use uranium_league::trajectory::Segment;

use crate::agent::Broker;
use crate::{err, py_error};

/// One RL batch (lanes of segments) handed to the learner's `update`.
#[pyclass(module = "uranium._core", frozen)]
pub struct SegmentBatch {
    segs: Vec<Segment>,
}

#[pymethods]
impl SegmentBatch {
    #[getter]
    fn lanes(&self) -> usize {
        self.segs.len()
    }

    /// Lanes `start..end` collated for one forward (`uranium_league::batch::group`): the
    /// observation arrays, the taken actions as labels, `privileged`, `mask`, `behaviour`,
    /// `rewards` [rows, 4], per lane `terminal`, `trained`, `versions` and `state0` (None = zero
    /// state, else a list of (shape, float32 array)), plus `b`, `t`, `n`.
    fn group<'py>(&self, py: Python<'py>, start: usize, end: usize, t: usize, max_select: usize) -> PyResult<Bound<'py, PyDict>> {
        if start >= end || end > self.segs.len() {
            return Err(PyValueError::new_err(format!("lanes {start}..{end} of {}", self.segs.len())));
        }
        let segs = &self.segs[start..end];
        let g = py.detach(move || uranium_league::batch::group(segs, t, max_select));
        let d = PyDict::new(py);
        let (r, n) = (g.obs.b, g.obs.n);
        crate::put_obs(py, &d, g.obs)?;
        let l = g.labels;
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
        put!("privileged", g.privileged.summary, vec![r, g.privileged.p]);
        put!("mask", g.mask, vec![r]);
        put!("behaviour", g.behaviour, vec![r]);
        put!("rewards", g.rewards, vec![r, 4]);
        put!("terminal", g.terminal, vec![g.b]);
        put!("trained", g.trained.iter().map(|&x| x as i64).collect::<Vec<_>>(), vec![g.b]);
        put!("versions", g.versions, vec![g.b]);
        let states = PyList::empty(py);
        for s in g.state0 {
            if s.parts.is_empty() {
                states.append(py.None())?;
            } else {
                let parts = PyList::empty(py);
                for (shape, v) in s.parts {
                    parts.append((shape, v.into_pyarray(py)))?;
                }
                states.append(parts)?;
            }
        }
        d.set_item("state0", states)?;
        d.set_item("b", g.b)?;
        d.set_item("t", g.t)?;
        d.set_item("n", n)?;
        Ok(d)
    }
}

/// The Python learner object behind [`RlLearners`]: `brokers()`, `version(i)`,
/// `update(i, batch, weights) -> json`, `push(i)`, `reset_to_bc(i)`, `save(i, dir, train_json)`,
/// `snapshot_server(dir) -> Broker`.
struct PyLearners {
    obj: Py<PyAny>,
    handles: Vec<InferenceHandle>,
}

impl RlLearners for PyLearners {
    fn handle(&self, agent: usize) -> InferenceHandle {
        self.handles[agent].clone()
    }

    fn version(&self, agent: usize) -> u64 {
        Python::attach(|py| self.obj.bind(py).call_method1("version", (agent,)).and_then(|v| v.extract::<u64>())).unwrap_or(0)
    }

    fn update(&mut self, agent: usize, batch: Vec<Segment>, weights: [f32; 4]) -> anyhow::Result<UpdateStats> {
        Python::attach(|py| {
            let b = Py::new(py, SegmentBatch { segs: batch }).map_err(|e| py_error(py, e))?;
            let out = self.obj.bind(py).call_method1("update", (agent, b, weights.to_vec())).map_err(|e| py_error(py, e))?;
            let js: String = out.extract().map_err(|e| py_error(py, e))?;
            Ok(serde_json::from_str(&js)?)
        })
    }

    fn push(&mut self, agent: usize) -> anyhow::Result<()> {
        Python::attach(|py| self.obj.bind(py).call_method1("push", (agent,)).map(|_| ()).map_err(|e| py_error(py, e)))
    }

    fn reset_to_bc(&mut self, agent: usize) -> anyhow::Result<()> {
        Python::attach(|py| self.obj.bind(py).call_method1("reset_to_bc", (agent,)).map(|_| ()).map_err(|e| py_error(py, e)))
    }

    fn save(&mut self, agent: usize, dir: &Path, train: &serde_json::Value) -> anyhow::Result<()> {
        Python::attach(|py| self.obj.bind(py).call_method1("save", (agent, dir.to_path_buf(), train.to_string())).map(|_| ()).map_err(|e| py_error(py, e)))
    }
}

fn broker_handle(b: &Bound<'_, PyAny>) -> PyResult<InferenceHandle> {
    let b: PyRef<'_, Broker> = b.extract()?;
    Ok(b.handle())
}

/// 042: RL with the Rust run loop. `make(config_json, resume_dir)` builds the learner object
/// (see [`PyLearners`]) after the run lock is taken. Returns "done", or "busy" when the lock is
/// another host's (exit 75, 061-R5).
#[pyfunction]
#[pyo3(signature = (make, run, config, algo = "vtrace", seed = 1, take_over = false, logs = PathBuf::from("runs/rl"), z_table = None, max_updates = 0, max_minutes = 0, max_agent_steps = 0, checkpoint_minutes = 30, fixed_t0 = None))]
#[allow(clippy::too_many_arguments)]
pub fn run_rl(
    py: Python<'_>,
    make: Py<PyAny>,
    run: String,
    config: PathBuf,
    algo: &str,
    seed: u64,
    take_over: bool,
    logs: PathBuf,
    z_table: Option<PathBuf>,
    max_updates: u64,
    max_minutes: u64,
    max_agent_steps: u64,
    checkpoint_minutes: u64,
    fixed_t0: Option<String>,
) -> PyResult<&'static str> {
    let algo = match algo {
        "vtrace" => Algo::Vtrace,
        "ppo" => Algo::Ppo,
        _ => return Err(PyValueError::new_err(format!("algo {algo:?}: expected vtrace or ppo"))),
    };
    let fixed_t0 = match fixed_t0.as_deref() {
        None => None,
        Some("terran") => Some(uranium_proto::Race::Terran),
        Some("protoss") => Some(uranium_proto::Race::Protoss),
        Some("zerg") => Some(uranium_proto::Race::Zerg),
        Some(r) => return Err(PyValueError::new_err(format!("fixed_t0 {r:?}: expected terran, protoss or zerg"))),
    };
    let stop = uranium_store::stop_state().unwrap_or_default();
    let args = uranium_league::run::RlArgs { run, config, algo, seed, take_over, log_dir: logs, z_table, max_updates, max_minutes, max_agent_steps, checkpoint_minutes, fixed_t0, stop };
    let res = py.detach(move || {
        uranium_league::run::run(args, |cfg: &RlConfig, resume: Option<&Path>| {
            Python::attach(|py| -> anyhow::Result<(Box<dyn RlLearners>, SnapshotServers)> {
                let cfg_json = serde_json::to_string(cfg)?;
                let obj = make.call1(py, (cfg_json, resume.map(Path::to_path_buf))).map_err(|e| py_error(py, e))?;
                let brokers = obj.bind(py).call_method0("brokers").map_err(|e| py_error(py, e))?;
                let handles = brokers.try_iter().map_err(|e| py_error(py, e))?.map(|b| b.and_then(|b| broker_handle(&b))).collect::<PyResult<Vec<_>>>().map_err(|e| py_error(py, e))?;
                anyhow::ensure!(handles.len() == 3, "brokers() gave {} servers, expected one per agent (3)", handles.len());
                let snap = obj.clone_ref(py);
                let servers: SnapshotServers = Arc::new(move |dir: &Path| {
                    Python::attach(|py| {
                        let b = snap.bind(py).call_method1("snapshot_server", (dir.to_path_buf(),)).map_err(|e| py_error(py, e))?;
                        broker_handle(&b).map_err(|e| py_error(py, e))
                    })
                });
                Ok((Box::new(PyLearners { obj, handles }), servers))
            })
        })
    });
    match res {
        Ok(()) => Ok("done"),
        Err(e) if e.downcast_ref::<uranium_store::storage::LockBusy>().is_some() => {
            uranium_data::elog!("{e:#}");
            Ok("busy")
        }
        Err(e) => Err(err(e)),
    }
}

/// 042 reference targets (`uranium_league::returns`) for the PyTorch learner's tests: V-trace
/// (vs, pg_adv, rho), one sequence; `values` has T + 1 entries.
#[pyfunction]
pub fn vtrace(rewards: Vec<f32>, values: Vec<f32>, log_rho: Vec<f32>, gamma: f32, lam: f32, rho_bar: f32, c_bar: f32) -> PyResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    check_seq(&rewards, &values, &log_rho)?;
    let v = uranium_league::returns::vtrace(&uranium_league::returns::Seq { rewards: &rewards, values: &values, log_rho: &log_rho }, gamma, lam, rho_bar, c_bar);
    Ok((v.vs, v.pg_adv, v.rho))
}

/// UPGO advantage (reference, see [`vtrace`]).
#[pyfunction]
pub fn upgo(rewards: Vec<f32>, values: Vec<f32>, log_rho: Vec<f32>, gamma: f32, rho_bar: f32) -> PyResult<Vec<f32>> {
    check_seq(&rewards, &values, &log_rho)?;
    Ok(uranium_league::returns::upgo(&uranium_league::returns::Seq { rewards: &rewards, values: &values, log_rho: &log_rho }, gamma, rho_bar))
}

/// GAE (advantages, returns) (reference, see [`vtrace`]).
#[pyfunction]
pub fn gae(rewards: Vec<f32>, values: Vec<f32>, gamma: f32, lam: f32) -> PyResult<(Vec<f32>, Vec<f32>)> {
    let zeros = vec![0.0; rewards.len()];
    check_seq(&rewards, &values, &zeros)?;
    Ok(uranium_league::returns::gae(&uranium_league::returns::Seq { rewards: &rewards, values: &values, log_rho: &zeros }, gamma, lam))
}

fn check_seq(r: &[f32], v: &[f32], l: &[f32]) -> PyResult<()> {
    if v.len() != r.len() + 1 || l.len() != r.len() {
        return Err(PyValueError::new_err(format!("{} rewards need {} values and log_rhos, got {} and {}", r.len(), r.len() + 1, v.len(), l.len())));
    }
    Ok(())
}

/// Tests (042): `lanes` segments of `t` synthetic steps (`uranium_league::testkit`), every step
/// a NoOp with behaviour log-prob 0; lane k is terminal when k is even; rewards: +1 win at the
/// end, 0.01 * step on the value component.
#[pyfunction]
pub fn test_segments(lanes: usize, t: usize) -> SegmentBatch {
    use uranium_env::features::PrivilegedTensors;
    use uranium_env::features::executor::HostAction;
    use uranium_league::trajectory::{PackedObs, Step};
    let segs = (0..lanes)
        .map(|lane| {
            let steps = uranium_league::testkit::observations(t)
                .iter()
                .enumerate()
                .map(|(k, o)| Step {
                    obs: PackedObs::pack(o),
                    action: HostAction { action: 0, delay_bucket: k % 4, queued: false, selected: vec![], target_unit: None, target_loc: None },
                    log_prob: 0.0,
                    head_log_prob: [0.0; uranium_agent::infer::HEADS],
                    reward: [if k == t - 1 { 1.0 } else { 0.0 }, 0.0, 0.0, 0.01 * k as f32],
                    privileged: PrivilegedTensors { summary: vec![0.0; uranium_env::features::extract::dims().privileged()] },
                    version: 0,
                })
                .collect();
            Segment { agent: 0, game_id: lane as u64, steps, state0: Default::default(), terminal: lane % 2 == 0 }
        })
        .collect();
    SegmentBatch { segs }
}

/// The `config/rl.toml` values as JSON (042-R9).
#[pyfunction]
pub fn rl_config(path: PathBuf) -> PyResult<String> {
    let cfg = RlConfig::load(&path).map_err(err)?;
    serde_json::to_string(&cfg).map_err(|e| err(e.into()))
}
