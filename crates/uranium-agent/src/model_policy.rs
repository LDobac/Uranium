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
use uranium_env::features::Strategy;
use uranium_env::features::extract::action_table;
use uranium_env::maps::{MapFamilies, PrecomputeSource, family_index, live_precompute_for_session, maps_dir};
use uranium_env::reward::{Outcome, RewardParts, RewardTracker};
use uranium_env::surrender::{SurrenderJudge, SurrenderMode, rule_holds};
use crate::infer::HostState;
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
    /// 030-R10 / E-5: the session fed `last_failed = 0` (BC evaluation games).
    pub zero_last_failed: bool,
    /// 031-R12 / 020-R14: surrender mode (`off` / `shadow` / `on`), and whether the checkpoint has
    /// the win_open head (without it the check never holds).
    pub surrender: String,
    pub win_open: bool,
    /// First frame the 031-R12 check held (Shadow and On); in On mode the session surrendered there.
    pub surrender_frame: Option<u32>,
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
/// joint, value, reward, core_state0).
pub struct Decision<'a> {
    pub frame: u32,
    pub obs: &'a uranium_env::features::Observation,
    pub action: &'a uranium_env::features::executor::HostAction,
    pub log_prob: f32,
    pub head_log_prob: [f32; crate::infer::HEADS],
    pub value: [f32; 4],
    /// The executor refused the action (031-R4): no command was sent.
    pub rejected: bool,
    /// Weights version of the inference server (032-R16, 042-R4).
    pub version: u64,
    /// 033-R7: reward of the frames since the previous decision (it belongs to that decision).
    pub reward_before: RewardParts,
    /// Core state before this decision, when the observer asked for it (042 core_state0).
    pub state: Option<HostState>,
}

/// Trajectory hook for the RL actor (042).
pub trait SessionObserver: Send {
    fn on_decision(&mut self, d: &Decision<'_>);
    /// Lockstep training only (010-R10); never reaches the policy (001-R7).
    fn on_privileged(&mut self, _p: &uranium_proto::PrivilegedSummary) {}
    /// Whether the next decision should carry the Core state (start of a segment, 042).
    fn wants_state(&self) -> bool {
        false
    }
    /// Game over: reward of the frames after the last decision, the terminal r_win (033-R7) and
    /// the game outcome (`None` = Aborted / disconnected, 042-R3).
    fn on_end(&mut self, _reward_after: RewardParts, _outcome: Option<Outcome>) {}
}

/// Starts RL sessions: the z of the episode and its observer (042, 033-R8).
pub trait LearnerHook: Send + Sync {
    fn handle(&self) -> &InferenceHandle;
    fn session(&self, game_id: u64, info: &GameInfo) -> (Option<Strategy>, Box<dyn SessionObserver>);
}

pub struct ModelPolicy {
    observer: Option<Box<dyn SessionObserver>>,
    /// Strategy condition of the episode (030 §5, 033-R8); `None` = z absent.
    z: Option<Strategy>,
    /// 033 rewards, kept only when an observer records the trajectory.
    reward: Option<RewardTracker>,
    self_player: u8,
    sess: InferSession,
    mem: AgentMemory,
    map: MapInfo,
    stats: ModelStats,
    decide: Vec<f32>,
    run: Vec<f32>,
    host: Vec<f32>,
    batch_sum: u64,
    /// 031-R12 check of this game, with the handle's mode and chat line (020-R14).
    judge: SurrenderJudge,
    surrender_mode: SurrenderMode,
    surrender_chat: String,
    /// The surrender Act went out (010-R13); the bridge asks for no more decisions.
    surrendered: bool,
    /// Test only (011-AC9 / 031-AC5 integration): `URANIUM_TEST_SURRENDER_FRAME`, in On mode the
    /// first decision at or after this frame surrenders whatever the check says.
    test_surrender_frame: Option<u32>,
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
        // 031: a target map (투혼, 서킷브레이커) without a current precompute refuses the session
        // (메인 결정 2026-10-19); any other map is played with the precompute channels at 0 and a
        // warning (AMB-031-4).
        let (pre, source) = live_precompute_for_session(maps_dir(), info, &family)?;
        let source = match source {
            PrecomputeSource::Cached => "cached".to_string(),
            PrecomputeSource::Resources => "resources".to_string(),
            PrecomputeSource::ReplayBuild => {
                // The same algorithm on the live static map must reproduce the replay build's channels.
                let rebuilt = live_rebuild(info);
                format!("replay_build(match={})", rebuilt.as_ref() == pre.as_ref())
            }
            PrecomputeSource::Missing => {
                elog!("[agent] WARNING: no map precompute for {:?}; base_loc/choke/wall channels stay 0 (031, run `uranium-replay mapgen`)", info.map_name);
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
            zero_last_failed: handle.zero_last_failed,
            surrender: format!("{:?}", handle.surrender).to_lowercase(),
            ..Default::default()
        };
        let test_surrender_frame = std::env::var("URANIUM_TEST_SURRENDER_FRAME").ok().and_then(|v| v.trim().parse().ok());
        let self_player = info.players.iter().find(|p| p.is_self).map_or(0, |p| p.id);
        // 030-R11: a lobby-Random enemy (GameInfo race Random / Unknown) stays `unknown` until its
        // first unit is seen (`AgentMemory::new`).
        let mut mem = AgentMemory::new(info);
        // 030-R10 / E-5: evaluation games of a BC checkpoint feed last_failed = 0 (see
        // `AgentMemory::zero_last_failed` for why); RL sessions keep the real value.
        mem.zero_last_failed = handle.zero_last_failed;
        Ok(Self {
            observer: None,
            z: None,
            reward: None,
            self_player,
            sess: handle.open()?,
            mem,
            map,
            stats,
            decide: vec![],
            run: vec![],
            host: vec![],
            batch_sum: 0,
            judge: SurrenderJudge::new(),
            surrender_mode: handle.surrender,
            surrender_chat: handle.surrender_chat.to_string(),
            surrendered: false,
            test_surrender_frame,
        })
    }

    /// 012-R14 / 012-R18: SC:R (realtime) sessions get no `CommandFailed`, so `last_failed` is 0
    /// whatever the inference server was started with.
    pub fn with_zero_last_failed(mut self) -> Self {
        self.mem.zero_last_failed = true;
        self.stats.zero_last_failed = true;
        self
    }

    /// Records the trajectory through `o` with 033 rewards for strategy `z` (042).
    pub fn with_observer(mut self, z: Option<Strategy>, o: Box<dyn SessionObserver>) -> Self {
        self.reward = Some(RewardTracker::new(z.clone(), self.self_player));
        self.z = z;
        self.observer = Some(o);
        self
    }

    /// Game over (042, 033-R7): the frames after the last decision and r_win go to the observer.
    /// E-8 / 010-R12: the dense reward of those frames comes from the bridge's final frame (own
    /// vision, events since the last decision); without one only r_win is added.
    pub fn on_end(&mut self, outcome: Option<Outcome>, final_frame: Option<&RawFrame>) {
        if let Some(o) = self.observer.as_mut() {
            let mut after = match (outcome, final_frame, self.reward.as_mut()) {
                (Some(_), Some(f), Some(r)) => r.step(f),
                _ => RewardParts::default(),
            };
            if let Some(out) = outcome {
                after.win = RewardTracker::end(out).win;
            }
            o.on_end(after, outcome);
        }
    }

    pub fn on_privileged(&mut self, p: &uranium_proto::PrivilegedSummary) {
        if let Some(o) = self.observer.as_mut() {
            o.on_privileged(p);
        }
    }

    /// 031-R12 after a decision (memory has seen `f`). `Some(act)` = surrender now (On mode).
    fn surrender_check(&mut self, f: &RawFrame, win_open: Option<f32>) -> Option<Act> {
        let mode = self.surrender_mode;
        if mode == SurrenderMode::Off || self.surrendered {
            return None;
        }
        if win_open.is_none() && self.stats.decisions == 0 {
            elog!("[agent] WARNING: checkpoint {} has no win_open head: the 031-R12 surrender check never holds", self.stats.checkpoint);
        }
        self.stats.win_open = win_open.is_some();
        let mem = &self.mem;
        let held = self.judge.decide(f.frame, win_open, || rule_holds(f, mem));
        if held && self.stats.surrender_frame.is_none() {
            self.stats.surrender_frame = Some(f.frame);
        }
        let forced = self.test_surrender_frame.is_some_and(|n| f.frame >= n);
        if mode != SurrenderMode::On || !(held || forced) {
            return None;
        }
        if forced && self.stats.surrender_frame.is_none() {
            self.stats.surrender_frame = Some(f.frame);
        }
        self.surrendered = true;
        let chat = self.surrender_chat.clone();
        elog!("[agent] 031-R12: surrender at frame {} (chat {chat:?}{})", f.frame, if forced { ", forced by URANIUM_TEST_SURRENDER_FRAME" } else { "" });
        Some(Act { cmds: vec![], advance_frames: 1, valid_until_frame: None, surrender: Some(chat) })
    }

    pub fn on_frame(&mut self, f: &RawFrame) -> Result<Act> {
        let t0 = Instant::now();
        // z absent unless an RL session set one (041-AC2 / AMB-041-2, 033-R8).
        let obs = extract(f, &mut self.mem, &self.map, self.z.as_ref());
        let reward_before = self.reward.as_mut().map(|r| r.step(f)).unwrap_or_default();
        let state = match self.observer.as_ref() {
            Some(o) if o.wants_state() => Some(self.sess.state()?),
            _ => None,
        };
        let host_extract = t0.elapsed();
        let t1 = Instant::now();
        let r = self.sess.act(obs)?;
        self.decide.push(t1.elapsed().as_secs_f32() * 1e3);
        self.run.push(r.run.as_secs_f32() * 1e3);
        self.batch_sum += r.batch as u64;
        if let Some(act) = self.surrender_check(f, r.win_open) {
            self.stats.decisions += 1;
            return Ok(act);
        }
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
            o.on_decision(&Decision {
                frame: f.frame,
                obs: &r.obs,
                action: a,
                log_prob: r.log_prob,
                head_log_prob: r.head_log_prob,
                value: r.value,
                rejected,
                version: r.version,
                reward_before,
                state,
            });
        }
        // 033 r_build: own Build / Train / Research / Upgrade actions (same group as replay z).
        if let (Some(rt), false) = (self.reward.as_mut(), rejected) {
            if action_table().actions.get(a.action as usize).is_some_and(|x| matches!(x.command.as_str(), "Build" | "BuildAddon" | "Train" | "Research" | "Upgrade")) {
                rt.record_build(a.action);
            }
        }
        // Memory records the sampled action like BC records its label (030 §4, 041-R11).
        let ids: Vec<u32> = a.selected.iter().filter_map(|&i| r.obs.entity_ids.get(i).copied()).collect();
        self.mem.record_action(a.action, a.delay_bucket, &ids);
        self.stats.decisions += 1;
        self.host.push((host_extract + t2.elapsed()).as_secs_f32() * 1e3);
        let bucket = a.delay_bucket.min(DELAY_BUCKETS.len() - 1);
        Ok(Act { cmds, advance_frames: DELAY_BUCKETS[bucket], valid_until_frame: None, surrender: None })
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
    let r = uranium_env::maps::load_resources(&maps_dir().ok()?, &hash)?;
    Some(uranium_env::features::mapgen::precompute(&info.static_map, &r.resources))
}
