use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use serde::Serialize;
use uranium_env::features::executor::{Rejected, execute};
use uranium_env::features::world::World;
use uranium_env::features::{AgentMemory, DELAY_BUCKETS, MapInfo, extract};
use uranium_env::features::Strategy;
use uranium_env::features::extract::action_table;
use uranium_env::features::label::DecisionLogLine;
use uranium_env::maps::{MapFamilies, PrecomputeSource, family_index, live_precompute_for_session, maps_dir};
use uranium_env::reward::{Outcome, RewardParts, RewardTracker};
use uranium_match::ops::{OpsMetrics, OpsTracker, RushCombat};
use uranium_env::scout::{RewardConfig, ScoutMetrics, ScoutTracker, seconds};
use uranium_match::concede::{CollapseJudge, ConcedeGame, LeaveRule};
use uranium_match::surrender::{DEFAULT_CHAT, SurrenderJudge, SurrenderMode, rule_holds};
use uranium_match::branch::{
    BranchOutcome, BranchPlayer, DilutionTracker, ForcedPrefix, GameBranch, Intervention, Invalid, InvalidReason, PrefixStep, SampleTracker, SnapshotTaken, Snapshots, StateCheck, ActLog, decode_state, encode_state, obs_digest, open_ids, quit_act, sha256_hex, state_id,
};
use uranium_match::explore::{ExploreEnd, ExploreRun, ExploreSummary, force_mask, policy_top};
use crate::infer::{ActReply, HostState};
use uranium_env::scout::FRAME_8M;
use uranium_proto::{Act, GameInfo, RawFrame};
use uranium_store::elog;

use crate::infer::{InferSession, InferenceHandle, SampleMode};

const NOOP_ACTION: u16 = 0;

#[derive(Serialize, Debug, Clone, Default)]
pub struct ModelStats {
    pub checkpoint: String,
    pub sample: String,
    pub map_family: String,
    pub map_precompute: String,
    pub zero_last_failed: bool,
    pub surrender: String,
    pub win_open: bool,
    pub surrender_frame: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub concede_frame: Option<u32>,
    pub decisions: u64,
    pub rejected: BTreeMap<String, u64>,
    pub decide_ms_p50: f64,
    pub decide_ms_p95: f64,
    pub run_ms_p50: f64,
    pub run_ms_p95: f64,
    pub mean_batch: f64,
    pub host_ms_p50: f64,
    pub scout_main_s: Option<f64>,
    pub scout_types_4m: u32,
    pub scout_types_6m: u32,
    pub barracks_s: Option<f64>,
    pub bunker_s: Option<f64>,
    pub scv_4m: Option<u32>,
    pub scv_min_1m: Option<u32>,
    pub army_s: Option<f64>,
    pub scv_8m: Option<u32>,
    pub ops: OpsMetrics,
    pub rush_combat: Option<RushCombat>,
}

pub struct Decision<'a> {
    pub frame: u32,
    pub obs: &'a uranium_env::features::Observation,
    pub action: &'a uranium_env::features::executor::HostAction,
    pub log_prob: f32,
    pub head_log_prob: [f32; crate::infer::HEADS],
    pub value: [f32; 4],
    pub rejected: bool,
    pub version: u64,
    pub reward_before: RewardParts,
    pub state: Option<HostState>,
}

pub trait SessionObserver: Send {
    fn on_decision(&mut self, d: &Decision<'_>);
    fn on_privileged(&mut self, _p: &uranium_proto::PrivilegedSummary) {}
    fn wants_state(&self) -> bool {
        false
    }
    fn on_scout(&mut self, _m: &ScoutMetrics) {}
    fn on_end(&mut self, _reward_after: RewardParts, _outcome: Option<Outcome>) {}
}

pub trait LearnerHook: Send + Sync {
    fn handle(&self) -> &InferenceHandle;
    fn session(&self, game_id: u64, info: &GameInfo) -> (Option<Strategy>, Box<dyn SessionObserver>);
    fn reward_config(&self) -> RewardConfig {
        RewardConfig::default()
    }
}

pub struct ModelPolicy {
    observer: Option<Box<dyn SessionObserver>>,
    z: Option<Strategy>,
    reward: Option<RewardTracker>,
    scout: ScoutTracker,
    ops: OpsTracker,
    self_player: u8,
    sess: InferSession,
    mem: AgentMemory,
    map: MapInfo,
    stats: ModelStats,
    decide: Vec<f32>,
    run: Vec<f32>,
    host: Vec<f32>,
    batch_sum: u64,
    judge: SurrenderJudge,
    surrender_mode: SurrenderMode,
    surrender_chat: String,
    surrendered: bool,
    test_surrender_frame: Option<u32>,
    concede: Option<(CollapseJudge, Arc<ConcedeGame>, u8)>,
    concede_reward: Option<RewardParts>,
    decision_log: Option<DecisionLog>,
    branch: Option<Box<BranchSession>>,
    won_at: Option<u32>,
    end_frame: Option<u32>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct LineExtra {
    pub decision: Option<uranium_match::branch::Decision>,
    pub state: Option<String>,
    pub state_sha256: Option<String>,
    pub forced: bool,
    pub policy_top: Option<Vec<(u16, f32)>>,
    pub explore: Option<uranium_match::explore::ExploreEntry>,
}

#[derive(Debug, Clone, Default)]
pub struct BranchSessionPart {
    pub snapshots: Vec<SnapshotTaken>,
    pub outcome: Option<BranchOutcome>,
    pub explore: Option<ExploreSummary>,
}

struct BranchSession {
    obs: u32,
    last: LineExtra,
    trunk: Option<TrunkSnapshots>,
    player: Option<Box<BranchRun>>,
    explore: Option<ExploreRun>,
}

struct TrunkSnapshots {
    rules: Snapshots,
    game: Arc<GameBranch>,
    taken: Vec<SnapshotTaken>,
}

struct BranchRun {
    cfg: BranchPlayer,
    prefix: Option<ForcedPrefix>,
    saved: Option<HostState>,
    pending: Option<(InvalidReason, String)>,
    invalid: Option<Invalid>,
    seen: u32,
    at_t: bool,
    intervention: Intervention,
    samples: SampleTracker,
    dilution: DilutionTracker,
    state_check: Option<StateCheck>,
}

impl BranchRun {
    fn load(cfg: &BranchPlayer) -> Result<Self> {
        let table = action_table();
        let line = &cfg.line;
        let intervention = Intervention::new(&line.alt, table, line.f_t)?;
        let dilution = DilutionTracker::new(&line.alt, &cfg.none_actions, table, line.f_t)?;
        let mut pending = None;
        let prefix = match ActLog::read_path(&cfg.acts).and_then(|log| ForcedPrefix::new(log, line.i_t, line.f_t)) {
            Ok(p) => Some(p),
            Err(e) => {
                pending = Some((InvalidReason::MissingInput, format!("trunk act log: {e:#}")));
                None
            }
        };
        let saved = match std::fs::read(&cfg.state) {
            Err(e) => {
                pending.get_or_insert((InvalidReason::MissingInput, format!("state file {}: {e}", cfg.state.display())));
                None
            }
            Ok(bytes) if sha256_hex(&bytes) != line.state_sha256 => {
                pending.get_or_insert((InvalidReason::State, format!("state file {}: sha256 {} is not {}", cfg.state.display(), sha256_hex(&bytes), line.state_sha256)));
                None
            }
            Ok(bytes) => match decode_state(&bytes) {
                Ok(parts) => Some(HostState { parts }),
                Err(e) => {
                    pending.get_or_insert((InvalidReason::State, format!("state file {}: {e:#}", cfg.state.display())));
                    None
                }
            },
        };
        Ok(Self {
            cfg: cfg.clone(),
            prefix,
            saved,
            pending,
            invalid: None,
            seen: 0,
            at_t: false,
            intervention,
            samples: SampleTracker::new(line.f_t, cfg.horizon),
            dilution,
            state_check: None,
        })
    }

    fn fail(&mut self, reason: InvalidReason, obs: Option<u32>, frame: Option<u32>, why: &str) {
        elog!("[agent] 051: branch {} invalid ({reason:?}) at observation {obs:?} frame {frame:?}: {why}", self.cfg.line.point);
        self.invalid = Some(Invalid { reason, obs, frame });
    }
}

fn save_state(path: &std::path::Path, state: &HostState) -> Result<String> {
    use std::io::Write as _;
    let bytes = encode_state(&state.parts)?;
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d)?;
    }
    let mut f = std::fs::File::create(path)?;
    f.write_all(&bytes)?;
    f.sync_all()?;
    Ok(sha256_hex(&bytes))
}

pub fn compare_states(a: &HostState, b: &HostState) -> StateCheck {
    let same_shape = a.parts.len() == b.parts.len() && a.parts.iter().zip(&b.parts).all(|((sa, da), (sb, db))| sa == sb && da.len() == db.len());
    if !same_shape {
        return StateCheck::new(false, f64::INFINITY);
    }
    let mut bit_equal = true;
    let mut max = 0f64;
    for ((_, da), (_, db)) in a.parts.iter().zip(&b.parts) {
        for (x, y) in da.iter().zip(db) {
            bit_equal &= x.to_bits() == y.to_bits();
            let d = (*x as f64 - *y as f64).abs();
            max = if d.is_nan() || max.is_nan() { f64::NAN } else { max.max(d) };
        }
    }
    StateCheck::new(bit_equal, max)
}

pub fn outcome_metrics(m: &ScoutMetrics, won_at: Option<u32>) -> serde_json::Map<String, serde_json::Value> {
    let secs = |f: Option<u32>| f.map(seconds);
    let v = serde_json::json!({
        "held_8m": m.scv_8m.is_some_and(|n| n >= 8) || won_at.is_some_and(|f| f < FRAME_8M),
        "held_s": m.scv_min_1m.is_some_and(|n| n > 4),
        "scv_8m": m.scv_8m,
        "scv_min_1m": m.scv_min_1m,
        "bunker_s": secs(m.bunker_frame),
        "barracks_s": secs(m.barracks_frame),
        "army_s": secs(m.army_frame),
        "scout_main_s": secs(m.main_frame),
        "scout_types_4m": m.types_4m,
        "scout_types_6m": m.types_6m,
        "scv_4m": m.scv_4m,
    });
    match v {
        serde_json::Value::Object(map) => map,
        _ => unreachable!("an object literal"),
    }
}

pub struct DecisionLog {
    path: std::path::PathBuf,
    w: Option<std::io::BufWriter<std::fs::File>>,
    decisions: u64,
    player: u8,
}

impl DecisionLog {
    pub fn create(path: &std::path::Path, player: u8) -> Self {
        let open = || -> std::io::Result<std::fs::File> {
            if let Some(d) = path.parent() {
                std::fs::create_dir_all(d)?;
            }
            std::fs::File::create(path)
        };
        let w = match open() {
            Ok(f) => Some(std::io::BufWriter::new(f)),
            Err(e) => {
                elog!("[agent] WARNING: decision log {} not written (041-R28): {e}", path.display());
                None
            }
        };
        Self { path: path.to_path_buf(), w, decisions: 0, player }
    }

    fn line(&mut self, l: &DecisionLogLine) {
        use std::io::Write as _;
        let Some(w) = self.w.as_mut() else { return };
        let text = serde_json::to_string(l).expect("decision log line serializes");
        if let Err(e) = writeln!(w, "{text}") {
            elog!("[agent] WARNING: decision log {} stopped (041-R28): {e}", self.path.display());
            self.w = None;
        }
    }

    pub fn decision(&mut self, frame: u32, action: u16, delay_bucket: usize, units: &[u32], forced: bool) {
        self.line(&DecisionLogLine::Decision { frame, action, delay_bucket: delay_bucket as u32, units: units.to_vec(), forced });
        self.decisions += 1;
    }

    pub fn end(&mut self, end_frame: Option<u32>) {
        let decisions = self.decisions;
        self.line(&DecisionLogLine::End { decisions, end_frame, player: self.player });
        self.close();
    }

    pub fn abort(&mut self) {
        self.close();
    }

    fn close(&mut self) {
        use std::io::Write as _;
        if let Some(mut w) = self.w.take() {
            if let Err(e) = w.flush().and_then(|_| w.get_ref().sync_data()) {
                elog!("[agent] WARNING: decision log {} not flushed (041-R28): {e}", self.path.display());
            }
        }
    }
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
        let (pre, source) = live_precompute_for_session(maps_dir(), info, &family)?;
        let source = match source {
            PrecomputeSource::Cached => "cached".to_string(),
            PrecomputeSource::Resources => "resources".to_string(),
            PrecomputeSource::ReplayBuild => {
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
        let mut mem = AgentMemory::new(info);
        mem.zero_last_failed = handle.zero_last_failed;
        Ok(Self {
            observer: None,
            z: None,
            reward: None,
            scout: ScoutTracker::for_game(info, RewardConfig::default()),
            ops: OpsTracker::for_game(info, pre.as_ref()),
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
            concede: None,
            concede_reward: None,
            decision_log: handle.memory_log.as_deref().map(|p| DecisionLog::create(p, self_player)),
            branch: None,
            won_at: None,
            end_frame: None,
        })
    }

    pub fn with_branch(mut self, game: Arc<GameBranch>, players_index: u8) -> Result<Self> {
        let mut b = BranchSession { obs: 0, last: LineExtra::default(), trunk: None, player: None, explore: None };
        if players_index == 0 {
            b.explore = game.explore.as_ref().map(|g| ExploreRun::new(g, action_table())).transpose()?;
            if !game.snapshots.is_empty() {
                let rules = Snapshots::new(&game.snapshots, action_table())?;
                anyhow::ensure!(game.states_dir.is_some(), "051-R3: snapshot rules without a states folder");
                b.trunk = Some(TrunkSnapshots { rules, game: game.clone(), taken: vec![] });
            }
            if let Some(cfg) = &game.branch {
                b.player = Some(Box::new(BranchRun::load(cfg)?));
                self.surrender_mode = SurrenderMode::Off;
                self.stats.surrender = "off".into();
            }
        }
        self.branch = Some(Box::new(b));
        Ok(self)
    }

    pub fn act_log_extra(&mut self) -> Option<LineExtra> {
        self.branch.as_mut().map(|b| std::mem::take(&mut b.last))
    }

    pub fn branch_record(&mut self) -> Option<BranchSessionPart> {
        let m = self.scout_metrics();
        let b = self.branch.as_ref()?;
        let metrics = outcome_metrics(&m, self.won_at);
        let outcome = b.player.as_ref().map(|r| BranchOutcome {
            valid: r.invalid.is_none(),
            invalid: r.invalid.clone(),
            prefix_obs: r.seen,
            prefix_matched: r.prefix.as_ref().map_or(0, ForcedPrefix::matched),
            state_check: r.state_check.clone(),
            intervention: r.intervention.log(),
            samples: r.samples.samples().clone(),
            metrics,
            dilution: r.dilution.entries(),
            end: None,
        });
        Some(BranchSessionPart { snapshots: b.trunk.as_ref().map(|t| t.taken.clone()).unwrap_or_default(), outcome, explore: b.explore.as_ref().map(ExploreRun::summary) })
    }

    pub fn note_end(&mut self, won: Option<bool>, end_frame: u32) {
        self.end_frame = Some(end_frame);
        self.won_at = (won == Some(true)).then_some(end_frame);
    }

    pub fn explore_end(&self) -> Option<ExploreEnd> {
        self.branch.as_ref()?.explore.as_ref().map(ExploreRun::end)
    }

    fn explore_point(&mut self, frame: u32, type_mask: &[bool]) -> Option<usize> {
        self.branch.as_deref_mut()?.explore.as_mut()?.check(frame, type_mask)
    }

    fn wants_policy_top(&self) -> bool {
        self.branch.as_ref().is_some_and(|b| b.last.state.is_some())
    }

    fn note_policy_top(&mut self, r: &ActReply) -> Result<()> {
        if !self.wants_policy_top() {
            return Ok(());
        }
        let logp = r.type_logp.as_deref().context("051-R2: the inference reply has no type_logp for the snapshot decision (054-R5)")?;
        let b = self.branch.as_deref_mut().expect("checked in wants_policy_top");
        b.last.policy_top = Some(policy_top(logp, &r.obs.masks.type_mask));
        Ok(())
    }

    fn explore_step(&mut self, p: usize, frame: u32, r0: ActReply, saved: HostState) -> Result<ActReply> {
        let b = self.branch.as_deref_mut().expect("an explore point needs the 054 session");
        let run = b.explore.as_mut().expect("an explore point needs the 054 run");
        let logp = r0.type_logp.as_deref().context("054-R5: the inference reply has no type_logp for the explore decision")?;
        let entry = run.decide(p, frame, r0.action.action, logp, &r0.obs.masks.type_mask);
        let c = entry.c;
        b.last.explore = Some(entry);
        let Some(c) = c else { return Ok(r0) };
        self.sess.set_state(saved).context("054-R3: restoring the saved state")?;
        let mut obs = r0.obs;
        obs.masks.type_mask = force_mask(&obs.masks.type_mask, c);
        self.sess.act(obs)
    }

    pub fn memory(&self) -> &AgentMemory {
        &self.mem
    }

    fn scout_metrics(&self) -> ScoutMetrics {
        self.reward.as_ref().and_then(|r| r.scout_metrics()).unwrap_or_else(|| self.scout.metrics())
    }

    fn branch_prefix(&mut self, f: &RawFrame) -> Result<Option<Act>> {
        let Some(b) = self.branch.as_deref_mut() else { return Ok(None) };
        let i = b.obs - 1;
        let Some(run) = b.player.as_deref_mut() else { return Ok(None) };
        if run.invalid.is_some() {
            return Ok(Some(quit_act()));
        }
        if let Some((reason, why)) = run.pending.take() {
            run.fail(reason, Some(i), Some(f.frame), &why);
            return Ok(Some(quit_act()));
        }
        let prefix = run.prefix.as_mut().expect("a branch without a pending load error has its prefix");
        if i > prefix.i_t() {
            return Ok(None);
        }
        run.seen += 1;
        let digest = obs_digest(f)?;
        let line = match prefix.step(i, f.frame, digest) {
            Err(inv) => {
                let why = format!("{:?} differs from the trunk", inv.reason);
                run.fail(inv.reason, inv.obs, inv.frame, &why);
                return Ok(Some(quit_act()));
            }
            Ok(PrefixStep::Forced(line)) => line.clone(),
            Ok(PrefixStep::AtT(_) | PrefixStep::Live) => return Ok(None),
        };
        let obs = extract(f, &mut self.mem, &self.map, self.z.as_ref());
        if run.cfg.verify_state {
            self.sess.act(obs)?;
        }
        let d = line.decision.clone().expect("ForcedPrefix holds a model session's log");
        self.mem.record_action(d.action, d.delay_bucket as usize, &d.units);
        if let Some(l) = self.decision_log.as_mut() {
            l.decision(f.frame, d.action, d.delay_bucket as usize, &d.units, true);
        }
        b.last = LineExtra { decision: Some(d), forced: true, ..Default::default() };
        Ok(Some(line.act))
    }

    fn branch_before_act(&mut self, f: &RawFrame, obs: &mut uranium_env::features::Observation) -> Result<Option<Act>> {
        let Some(b) = self.branch.as_deref_mut() else { return Ok(None) };
        let i = b.obs - 1;
        if let Some(t) = b.trunk.as_mut() {
            let hit = t.rules.check(i, f.frame, &obs.masks.type_mask);
            if !hit.is_empty() {
                let state = self.sess.state()?;
                let id = state_id(t.game.plan_game, i);
                let path = t.game.state_path(i).expect("checked in with_branch");
                match save_state(&path, &state) {
                    Ok(sha) => {
                        t.taken.push(SnapshotTaken {
                            rules: hit,
                            state: id.clone(),
                            state_sha256: sha.clone(),
                            i,
                            frame: f.frame,
                            digest: obs_digest(f)?,
                            type_mask: open_ids(&obs.masks.type_mask),
                            scalar: obs.obs.scalar.clone(),
                        });
                        b.last.state = Some(id);
                        b.last.state_sha256 = Some(sha);
                    }
                    Err(e) => elog!("[agent] WARNING: 051-R3 state file {} not written: {e:#}", path.display()),
                }
            }
        }
        if let Some(run) = b.player.as_deref_mut() {
            let i_t = run.prefix.as_ref().map(ForcedPrefix::i_t);
            if !run.at_t && i_t == Some(i) {
                let saved = run.saved.clone().expect("a branch without a pending load error has its state");
                if run.cfg.verify_state {
                    run.state_check = Some(compare_states(&self.sess.state()?, &saved));
                }
                if let Err(e) = self.sess.set_state(saved) {
                    run.fail(InvalidReason::State, Some(i), Some(f.frame), &format!("{e:#}"));
                    return Ok(Some(quit_act()));
                }
                run.at_t = true;
            }
            if run.at_t {
                run.intervention.apply_to(f.frame, obs);
            }
        }
        Ok(None)
    }

    fn branch_after_act(&mut self, frame: u32, action: u16, delay_bucket: usize, ids: &[u32], rejected: bool, scalar: &[f32]) {
        let Some(b) = self.branch.as_deref_mut() else { return };
        b.last.decision = Some(uranium_match::branch::Decision { action, delay_bucket: delay_bucket as u32, units: ids.to_vec(), rejected });
        if let (Some(e), Some(run)) = (b.last.explore.as_mut(), b.explore.as_mut()) {
            e.rejected = Some(rejected);
            run.note_rejected(rejected);
        }
        if let Some(run) = b.player.as_deref_mut().filter(|r| r.at_t) {
            run.intervention.note_executor(frame, rejected);
            run.samples.on_decision(frame, scalar, action != NOOP_ACTION && !rejected);
            run.dilution.on_decision(frame, action);
        }
    }

    pub fn with_zero_last_failed(mut self) -> Self {
        self.mem.zero_last_failed = true;
        self.stats.zero_last_failed = true;
        self
    }

    pub fn with_concede(mut self, game: Arc<ConcedeGame>, side: u8) -> Self {
        self.concede = Some((CollapseJudge::new(game.cfg), game, side));
        self
    }

    pub fn with_observer(mut self, z: Option<Strategy>, reward: RewardConfig, o: Box<dyn SessionObserver>) -> Self {
        let scout = self.scout.clone().with_config(reward);
        self.reward = Some(RewardTracker::new(z.clone(), self.self_player).with_scout(scout));
        self.z = z;
        self.observer = Some(o);
        self
    }

    pub fn on_end(&mut self, outcome: Option<Outcome>, final_frame: Option<&RawFrame>) {
        if let Some(l) = self.decision_log.as_mut() {
            l.end(final_frame.map(|f| f.frame));
        }
        let end_frame = self.end_frame.or(final_frame.map(|f| f.frame));
        if let Some(run) = self.branch.as_deref_mut().and_then(|b| b.explore.as_mut()) {
            run.finish(outcome.is_none(), end_frame);
        }
        if let Some(b) = self.branch.as_deref_mut() {
            let next = b.obs;
            if let Some(run) = b.player.as_deref_mut() {
                run.intervention.finish();
                if let Some((reason, why)) = run.pending.take() {
                    run.fail(reason, None, None, &why);
                } else if outcome.is_some() && !run.at_t && run.invalid.is_none() {
                    let why = format!("the game ended before t (trunk observation {:?})", run.prefix.as_ref().map(ForcedPrefix::i_t));
                    run.fail(InvalidReason::Frame, Some(next), final_frame.map(|f| f.frame), &why);
                }
            }
        }
        if let (None, Some(f)) = (self.reward.as_ref(), final_frame) {
            self.scout.step(f);
        }
        if let Some(f) = final_frame {
            self.ops.step(f);
        }
        if let Some(o) = self.observer.as_mut() {
            let mut after = match (outcome, final_frame, self.reward.as_mut()) {
                (Some(_), Some(f), Some(r)) => r.step(f),
                _ => RewardParts::default(),
            };
            if let (Some(_), Some(c)) = (outcome, self.concede_reward.take()) {
                after.add_dense(&c);
            }
            if let Some(out) = outcome {
                after.win = RewardTracker::end(out).win;
            }
            if let Some(m) = self.reward.as_ref().and_then(|r| r.scout_metrics()) {
                o.on_scout(&m);
            }
            o.on_end(after, outcome);
        }
    }

    pub fn on_abort(&mut self) {
        if let Some(l) = self.decision_log.as_mut() {
            l.abort();
        }
        self.on_end(None, None);
    }

    pub fn on_privileged(&mut self, p: &uranium_proto::PrivilegedSummary) {
        if let Some(o) = self.observer.as_mut() {
            o.on_privileged(p);
        }
    }

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
        if let Some((_, game, side)) = &self.concede {
            if !game.claim(*side, f.frame, LeaveRule::Surrender) {
                return None;
            }
        }
        if forced && self.stats.surrender_frame.is_none() {
            self.stats.surrender_frame = Some(f.frame);
        }
        self.surrendered = true;
        let chat = self.surrender_chat.clone();
        elog!("[agent] 031-R12: surrender at frame {} (chat {chat:?}{})", f.frame, if forced { ", forced by URANIUM_TEST_SURRENDER_FRAME" } else { "" });
        Some(Act { cmds: vec![], advance_frames: 1, valid_until_frame: None, surrender: Some(chat) })
    }

    fn concede_check(&mut self, f: &RawFrame) -> Option<Act> {
        let (judge, game, side) = self.concede.as_mut()?;
        if self.surrendered || game.conceded().is_some() {
            return None;
        }
        if !judge.decide(f.frame, f.self_state.supply_used) || !game.claim(*side, f.frame, LeaveRule::Concede) {
            return None;
        }
        let side = *side;
        self.surrendered = true;
        self.stats.concede_frame = Some(f.frame);
        self.concede_reward = self.reward.as_mut().map(|r| r.step(f));
        elog!("[agent] 042-R14: collapse concede at frame {} (supply_used {}, players index {side})", f.frame, f.self_state.supply_used);
        Some(Act { cmds: vec![], advance_frames: 1, valid_until_frame: None, surrender: Some(DEFAULT_CHAT.to_string()) })
    }

    pub fn on_frame(&mut self, f: &RawFrame) -> Result<Act> {
        if self.reward.is_none() {
            self.scout.step(f);
        }
        self.ops.step(f);
        if let Some(b) = self.branch.as_deref_mut() {
            b.obs += 1;
            b.last = LineExtra::default();
        }
        if let Some(act) = self.concede_check(f) {
            return Ok(act);
        }
        if let Some(act) = self.branch_prefix(f)? {
            return Ok(act);
        }
        let t0 = Instant::now();
        let mut obs = extract(f, &mut self.mem, &self.map, self.z.as_ref());
        if let Some(act) = self.branch_before_act(f, &mut obs)? {
            return Ok(act);
        }
        let explore = self.explore_point(f.frame, &obs.masks.type_mask);
        let saved = match explore {
            Some(_) => Some(self.sess.state()?),
            None => None,
        };
        let reward_before = self.reward.as_mut().map(|r| r.step(f)).unwrap_or_default();
        let state = match self.observer.as_ref() {
            Some(o) if o.wants_state() => Some(self.sess.state()?),
            _ => None,
        };
        let host_extract = t0.elapsed();
        let t1 = Instant::now();
        let r = self.sess.act_with(obs, explore.is_some() || self.wants_policy_top())?;
        self.note_policy_top(&r)?;
        let r = match (explore, saved) {
            (Some(p), Some(saved)) => self.explore_step(p, f.frame, r, saved)?,
            _ => r,
        };
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
        if let (Some(rt), false) = (self.reward.as_mut(), rejected) {
            if action_table().actions.get(a.action as usize).is_some_and(|x| matches!(x.command.as_str(), "Build" | "BuildAddon" | "Train" | "Research" | "Upgrade")) {
                rt.record_build(a.action);
            }
        }
        let ids: Vec<u32> = a.selected.iter().filter_map(|&i| r.obs.entity_ids.get(i).copied()).collect();
        self.mem.record_action(a.action, a.delay_bucket, &ids);
        if let Some(l) = self.decision_log.as_mut() {
            l.decision(f.frame, a.action, a.delay_bucket, &ids, false);
        }
        self.branch_after_act(f.frame, a.action, a.delay_bucket, &ids, rejected, &r.obs.obs.scalar);
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
        let m = self.reward.as_ref().and_then(|r| r.scout_metrics()).unwrap_or_else(|| self.scout.metrics());
        let secs = |f: Option<u32>| f.map(seconds);
        (s.scout_main_s, s.scout_types_4m, s.scout_types_6m) = (secs(m.main_frame), m.types_4m, m.types_6m);
        (s.barracks_s, s.bunker_s, s.scv_4m) = (secs(m.barracks_frame), secs(m.bunker_frame), m.scv_4m);
        (s.scv_min_1m, s.army_s, s.scv_8m) = (m.scv_min_1m, secs(m.army_frame), m.scv_8m);
        (s.ops, s.rush_combat) = (self.ops.ops(), self.ops.rush_combat());
        s
    }
}

fn live_rebuild(info: &GameInfo) -> Option<uranium_env::features::MapPrecompute> {
    let hash: String = info.map_hash.iter().map(|b| format!("{b:02x}")).collect();
    let r = uranium_env::maps::load_resources(&maps_dir().ok()?, &hash)?;
    Some(uranium_env::features::mapgen::precompute(&info.static_map, &r.resources))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_row_keys_050_r7() {
        let s = ModelStats { scv_4m: Some(20), scv_min_1m: Some(5), army_s: Some(185.0), scv_8m: None, ..Default::default() };
        let text = serde_json::to_string(&s).unwrap();
        assert!(text.contains(r#""scv_4m":20,"scv_min_1m":5,"army_s":185.0,"scv_8m":null,"ops":{"#), "{text}");
    }

    #[test]
    fn model_row_ends_with_ops_052_r6() {
        let s = ModelStats { rush_combat: Some(RushCombat { in_s: 100.0, ..Default::default() }), ..Default::default() };
        let text = serde_json::to_string(&s).unwrap();
        let keys = ["checkpoint", "sample", "map_family", "map_precompute", "zero_last_failed", "surrender", "win_open", "surrender_frame", "decisions", "rejected", "decide_ms_p50", "decide_ms_p95", "run_ms_p50", "run_ms_p95", "mean_batch", "host_ms_p50", "scout_main_s", "scout_types_4m", "scout_types_6m", "barracks_s", "bunker_s", "scv_4m", "scv_min_1m", "army_s", "scv_8m", "ops", "rush_combat"];
        let at: Vec<usize> = keys.iter().map(|k| text.find(&format!("\"{k}\":")).unwrap_or_else(|| panic!("{k} missing: {text}"))).collect();
        assert!(at.windows(2).all(|w| w[0] < w[1]), "key order: {text}");
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v.as_object().unwrap().len(), keys.len(), "no other key");
        assert_eq!(v["rush_combat"]["in_s"], 100.0);
        assert!(serde_json::to_string(&ModelStats::default()).unwrap().ends_with(r#""float_gas_rs":0.0},"rush_combat":null}"#));
    }

    #[test]
    fn outcome_metrics_hold_definitions_051_r9() {
        let m = ScoutMetrics { scv_8m: Some(8), scv_min_1m: Some(4), bunker_frame: Some(240), ..Default::default() };
        let v = outcome_metrics(&m, None);
        for k in uranium_match::branch::OUTCOME_METRICS {
            assert!(v.contains_key(*k), "{k}");
        }
        assert_eq!((v["held_8m"].as_bool(), v["held_s"].as_bool(), v["bunker_s"].as_f64()), (Some(true), Some(false), Some(10.0)));
        let v = outcome_metrics(&ScoutMetrics { scv_8m: None, scv_min_1m: Some(5), ..Default::default() }, None);
        assert_eq!((v["held_8m"].as_bool(), v["held_s"].as_bool()), (Some(false), Some(true)));
    }

    #[test]
    fn held_8m_counts_a_win_before_8m_050_r6_053_r2() {
        let held = |scv_8m: Option<u32>, won_at: Option<u32>| outcome_metrics(&ScoutMetrics { scv_8m, ..Default::default() }, won_at)["held_8m"].as_bool();
        assert_eq!(held(Some(8), None), Some(true));
        assert_eq!(held(Some(7), None), Some(false));
        assert_eq!(held(None, Some(5_000)), Some(true), "won before 8:00");
        assert_eq!(held(None, Some(11_519)), Some(true));
        assert_eq!(held(None, Some(11_520)), Some(false), "won at 8:00 has its scv_8m");
        assert_eq!(held(Some(3), Some(14_000)), Some(false), "won after 8:00 with 3 SCVs");
        assert_eq!(held(None, None), Some(false), "lost or drew before 8:00");
    }

    #[test]
    fn compare_states_051_r6() {
        let s = |v: Vec<f32>| HostState { parts: vec![(vec![1, v.len()], v)] };
        assert_eq!(compare_states(&s(vec![1.0, 2.0]), &s(vec![1.0, 2.0])), StateCheck { bit_equal: true, max_abs_diff: Some(0.0) });
        assert_eq!(compare_states(&s(vec![0.0, 2.0]), &s(vec![-0.0, 2.5])), StateCheck { bit_equal: false, max_abs_diff: Some(0.5) });
        assert_eq!(compare_states(&s(vec![1.0]), &s(vec![1.0, 2.0])), StateCheck { bit_equal: false, max_abs_diff: None });
        assert_eq!(compare_states(&s(vec![f32::NAN]), &s(vec![1.0])).max_abs_diff, None);
    }
}
