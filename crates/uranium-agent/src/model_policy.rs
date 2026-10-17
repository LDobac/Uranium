//! Model policy for one game player (020-R11): RawFrame -> features (030) -> batched `act` (032)
//! -> executor (031) -> `Act`. The same feature / memory path as BC (041), so a BC checkpoint
//! sees the inputs it was trained on.

use std::collections::BTreeMap;
use std::time::Instant;

use anyhow::Result;
use serde::Serialize;
use uranium_env::features::executor::{Rejected, execute};
use uranium_env::features::world::World;
use uranium_env::features::{AgentMemory, DELAY_BUCKETS, MapInfo, extract};
use uranium_env::maps::{MapFamilies, PrecomputeSource, family_index, live_precompute, maps_dir};
use uranium_proto::{Act, GameInfo, RawFrame};

use crate::infer::{InferSession, InferenceHandle, SampleMode};

/// Model-specific part of a session record (020-R5, 031-R4).
#[derive(Serialize, Debug, Clone, Default)]
pub struct ModelStats {
    /// 020-R5 policy version.
    pub checkpoint: String,
    pub sample: String,
    pub map_family: String,
    /// Where the map precompute came from (031, AMB-031-4); `replay_build` also says whether the
    /// precompute rebuilt from the map file matches it.
    pub map_precompute: String,
    pub decisions: u64,
    /// 031-R4 / AMB-031-5: executor rejections by reason.
    pub rejected: BTreeMap<String, u64>,
    /// Decision latency seen by the session (queue + batch run + host copy), ms.
    pub decide_ms_p50: f64,
    pub decide_ms_p95: f64,
    /// Batch run alone, ms.
    pub run_ms_p50: f64,
    pub run_ms_p95: f64,
    pub mean_batch: f64,
    /// Feature extraction + executor on the session thread, ms.
    pub host_ms_p50: f64,
}

/// One model decision as an RL actor needs it (042: obs, action, behaviour log-prob per head and
/// joint, value). Rewards come from `uranium_env::reward` on the same frames; the segment start
/// Core state (core_state0) is read from the inference server slot when the learner lands (M6).
pub struct Decision<'a> {
    pub frame: u32,
    pub obs: &'a uranium_env::features::Observation,
    pub action: &'a uranium_env::features::executor::HostAction,
    pub log_prob: f32,
    pub head_log_prob: [f32; uranium_model::heads::HEADS],
    pub value: [f32; 4],
    /// The executor refused the action (031-R4): no command was sent.
    pub rejected: bool,
}

/// Trajectory hook for the M6 actor (042). Sessions run without one until then.
pub trait SessionObserver: Send {
    fn on_decision(&mut self, d: &Decision<'_>);
    /// Lockstep training only (010-R10); never reaches the policy (001-R7).
    fn on_privileged(&mut self, _p: &uranium_proto::PrivilegedSummary) {}
}

pub struct ModelPolicy {
    observer: Option<Box<dyn SessionObserver>>,
    sess: InferSession,
    mem: AgentMemory,
    map: MapInfo,
    stats: ModelStats,
    decide: Vec<f32>,
    run: Vec<f32>,
    host: Vec<f32>,
    batch_sum: u64,
}

fn rejected_name(r: &Rejected) -> &'static str {
    match r {
        Rejected::UnknownAction => "unknown_action",
        Rejected::TypeMasked => "type_masked",
        Rejected::BadSelection => "bad_selection",
        Rejected::BadTarget => "bad_target",
        Rejected::BadLocation => "bad_location",
        Rejected::NoExecutorUnit => "no_executor_unit",
    }
}

fn pct(v: &mut [f32], q: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f32::total_cmp);
    v[((v.len() - 1) as f64 * q).round() as usize] as f64
}

impl ModelPolicy {
    pub fn new(handle: &InferenceHandle, info: &GameInfo) -> Result<Self> {
        let family = MapFamilies::load()?.classify(&info.map_name, "");
        let (pre, source) = live_precompute(&maps_dir(), info);
        let source = match source {
            PrecomputeSource::Cached => "cached".to_string(),
            PrecomputeSource::Resources => "resources".to_string(),
            PrecomputeSource::ReplayBuild => {
                // The same algorithm on the live static map must reproduce the replay build's channels.
                let rebuilt = live_rebuild(info);
                format!("replay_build(match={})", rebuilt.as_ref() == pre.as_ref())
            }
            PrecomputeSource::Missing => {
                eprintln!("[agent] WARNING: no map precompute for {:?}; base_loc/choke/wall channels stay 0 (031, run `uranium-replay mapgen`)", info.map_name);
                "missing".to_string()
            }
        };
        let map = MapInfo::new(info, family_index(&family), pre.as_ref())?;
        let stats = ModelStats {
            checkpoint: handle.checkpoint_id.to_string(),
            sample: match handle.sample {
                SampleMode::Argmax => "argmax".into(),
                SampleMode::Stochastic => "stochastic".into(),
            },
            map_family: family,
            map_precompute: source,
            ..Default::default()
        };
        Ok(Self { observer: None, sess: handle.open()?, mem: AgentMemory::new(info), map, stats, decide: vec![], run: vec![], host: vec![], batch_sum: 0 })
    }

    pub fn with_observer(mut self, o: Box<dyn SessionObserver>) -> Self {
        self.observer = Some(o);
        self
    }

    pub fn on_privileged(&mut self, p: &uranium_proto::PrivilegedSummary) {
        if let Some(o) = self.observer.as_mut() {
            o.on_privileged(p);
        }
    }

    pub fn on_frame(&mut self, f: &RawFrame) -> Result<Act> {
        let t0 = Instant::now();
        // z absent (041-AC2, AMB-041-2); RL z sampling comes with 033-R2.
        let obs = extract(f, &mut self.mem, &self.map, None);
        let host_extract = t0.elapsed();
        let t1 = Instant::now();
        let r = self.sess.act(obs)?;
        self.decide.push(t1.elapsed().as_secs_f32() * 1e3);
        self.run.push(r.run.as_secs_f32() * 1e3);
        self.batch_sum += r.batch as u64;
        let t2 = Instant::now();
        let a = &r.action;
        let (cmds, rejected) = {
            let w = World::new(f, &self.mem, &self.map);
            match execute(&w, &r.obs, a) {
                Ok(ex) => (ex.commands, false),
                Err(why) => {
                    // 031-R4 / AMB-031-5: no command, the sampled delay still runs.
                    *self.stats.rejected.entry(rejected_name(&why).to_string()).or_default() += 1;
                    (vec![], true)
                }
            }
        };
        if let Some(o) = self.observer.as_mut() {
            o.on_decision(&Decision { frame: f.frame, obs: &r.obs, action: a, log_prob: r.log_prob, head_log_prob: r.head_log_prob, value: r.value, rejected });
        }
        // Memory records the sampled action like BC records its label (030 §4, 041-R11).
        let ids: Vec<u32> = a.selected.iter().filter_map(|&i| r.obs.entity_ids.get(i).copied()).collect();
        self.mem.record_action(a.action, a.delay_bucket, &ids);
        self.stats.decisions += 1;
        self.host.push((host_extract + t2.elapsed()).as_secs_f32() * 1e3);
        let bucket = a.delay_bucket.min(DELAY_BUCKETS.len() - 1);
        Ok(Act { cmds, advance_frames: DELAY_BUCKETS[bucket], valid_until_frame: None })
    }

    pub fn stats(&mut self) -> ModelStats {
        let mut s = self.stats.clone();
        s.decide_ms_p50 = pct(&mut self.decide, 0.5);
        s.decide_ms_p95 = pct(&mut self.decide, 0.95);
        s.run_ms_p50 = pct(&mut self.run, 0.5);
        s.run_ms_p95 = pct(&mut self.run, 0.95);
        s.host_ms_p50 = pct(&mut self.host, 0.5);
        s.mean_batch = if s.decisions > 0 { self.batch_sum as f64 / s.decisions as f64 } else { 0.0 };
        s
    }
}

/// Precompute rebuilt from the map file's resources (ignores a cached replay-build file).
fn live_rebuild(info: &GameInfo) -> Option<uranium_env::features::MapPrecompute> {
    let hash: String = info.map_hash.iter().map(|b| format!("{b:02x}")).collect();
    let text = std::fs::read_to_string(uranium_env::maps::resources_path(&maps_dir(), &hash)).ok()?;
    let r: uranium_env::maps::MapResources = toml::from_str(&text).ok()?;
    Some(uranium_env::features::mapgen::precompute(&info.static_map, &r.resources))
}
