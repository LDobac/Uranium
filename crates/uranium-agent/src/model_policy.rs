//! Model policy for one game player (020-R11): RawFrame -> features (030) -> batched `act` (032)
//! -> executor (031) -> `Act`. The same feature / memory path as BC (041), so a BC checkpoint
//! sees the inputs it was trained on.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use serde::Serialize;
use uranium_env::features::executor::{Rejected, execute};
use uranium_env::features::world::World;
use uranium_env::features::{AgentMemory, DELAY_BUCKETS, MapInfo, extract};
use uranium_env::features::Strategy;
use uranium_env::features::extract::action_table;
use uranium_env::features::label::DecisionLogLine;
use uranium_env::maps::{MapFamilies, PrecomputeSource, family_index, live_precompute_for_session, maps_dir};
use uranium_env::reward::{Outcome, RewardParts, RewardTracker};
use uranium_env::scout::{RewardConfig, ScoutMetrics, ScoutTracker, seconds};
use uranium_env::concede::{CollapseJudge, ConcedeGame, LeaveRule};
use uranium_env::surrender::{DEFAULT_CHAT, SurrenderJudge, SurrenderMode, rule_holds};
use uranium_env::branch::{
    BranchOutcome, BranchPlayer, DilutionTracker, ForcedPrefix, GameBranch, Intervention, Invalid, InvalidReason, PrefixStep, SampleTracker, SnapshotTaken, Snapshots, StateCheck, ActLog, decode_state, encode_state, obs_digest, open_ids, quit_act, sha256_hex, state_id,
};
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
    /// 042-R14: frame at which this session conceded a collapsed RL game (no decision there).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub concede_frame: Option<u32>,
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
    /// 042 정찰·대응 지표 of this session's game (AMB-042-16), every model session (RL and plan /
    /// evaluation games alike, the same definition as `games.csv`): game seconds (frame / 24) of
    /// the enemy main found, the first own Barracks / Bunker (construction start); distinct enemy
    /// building types seen by 4:00 / 6:00; own SCVs at the first observation from 4:00. `None` =
    /// not reached. Metrics only: no reward in plan games.
    pub scout_main_s: Option<f64>,
    pub scout_types_4m: u32,
    pub scout_types_6m: u32,
    pub barracks_s: Option<f64>,
    pub bunker_s: Option<f64>,
    pub scv_4m: Option<u32>,
    /// 050-R7 세션 지표 (AMB-042-19), after `scv_4m` in this order: the least own SCVs from 1:00
    /// (held = `> 4`), the game seconds of the first enemy army sighting, own SCVs at the first
    /// observation from 8:00. `None` = not reached / never seen.
    pub scv_min_1m: Option<u32>,
    pub army_s: Option<f64>,
    pub scv_8m: Option<u32>,
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
    /// 042 정찰·대응 지표 of the game (AMB-042-16), right before `on_end`.
    fn on_scout(&mut self, _m: &ScoutMetrics) {}
    /// Game over: reward of the frames after the last decision, the terminal r_win (033-R7) and
    /// the game outcome (`None` = Aborted / disconnected, 042-R3).
    fn on_end(&mut self, _reward_after: RewardParts, _outcome: Option<Outcome>) {}
}

/// Starts RL sessions: the z of the episode and its observer (042, 033-R8).
pub trait LearnerHook: Send + Sync {
    fn handle(&self) -> &InferenceHandle;
    fn session(&self, game_id: u64, info: &GameInfo) -> (Option<Strategy>, Box<dyn SessionObserver>);
    /// 033-R9 / R10 settings of r_scout, r_survive and the 042 metrics (`[reward]` of the RL
    /// config, AMB-042-16).
    fn reward_config(&self) -> RewardConfig {
        RewardConfig::default()
    }
}

pub struct ModelPolicy {
    observer: Option<Box<dyn SessionObserver>>,
    /// Strategy condition of the episode (030 §5, 033-R8); `None` = z absent.
    z: Option<Strategy>,
    /// 033 rewards, kept only when an observer records the trajectory.
    reward: Option<RewardTracker>,
    /// 033-R9 / R10 tracker of this game (public game information only, 033-R5). An RL session's
    /// reward tracker gets a fresh copy with its settings (`with_observer`); every other model
    /// session (plan / evaluation games) steps this one for the 042 metrics only.
    scout: ScoutTracker,
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
    /// 042-R14 collapse concede of an RL league game: this session's judge, the game's shared
    /// state (one concede per game) and this session's `GameSpec.players` index.
    concede: Option<(CollapseJudge, Arc<ConcedeGame>, u8)>,
    /// 042-R14 / 033-R7: reward of the frames from the last real decision to the concede frame;
    /// added with the final frame's to that last decision at the end.
    concede_reward: Option<RewardParts>,
    /// 041-R28 ①: this game's decision log (`InferenceHandle::memory_log`); off by default.
    decision_log: Option<DecisionLog>,
    /// 051 (020-R22): act-log extras, snapshots and the branch player of a plan game with 051
    /// keys ([`ModelPolicy::with_branch`]); `None` otherwise, and then nothing below changes
    /// (020-R22 ④).
    branch: Option<Box<BranchSession>>,
}

/// 051-R2: the model's part of one act-log line; the session adds the observation index, frame,
/// digest and the `Act` sent.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LineExtra {
    /// The decision on this observation (`None`: no decision, e.g. a branch leaving, 051-R5 ①).
    pub decision: Option<uranium_env::branch::Decision>,
    /// 051-R3: state id saved right before this decision and its file's sha256.
    pub state: Option<String>,
    pub state_sha256: Option<String>,
    /// 051-R5 ⑤: the decision was forced from the trunk's act log.
    pub forced: bool,
}

/// 051: what a model session of a plan game with 051 keys adds to its session record (020-R22):
/// the trunk's `snapshots` (051-R3, read by the plan tool for `points.jsonl`, 051-R10) and a
/// branch player's outcome (record key `branch`, 051-R9, read for `branches.jsonl`).
#[derive(Debug, Clone, Default)]
pub struct BranchSessionPart {
    /// 051-R3: state files this session saved (trunk games).
    pub snapshots: Vec<SnapshotTaken>,
    /// 051-R9: a branch player's outcome (`end` is filled by the session from `GameEnd`).
    pub outcome: Option<BranchOutcome>,
}

/// 051 per-session state of a model session in a plan game with 051 keys.
struct BranchSession {
    /// Observations seen so far (the index of the current one is `obs - 1`).
    obs: u32,
    /// What the current observation adds to its act-log line (051-R2).
    last: LineExtra,
    /// 051-R3 snapshot rules of a trunk's `players[0]` model session.
    trunk: Option<TrunkSnapshots>,
    /// 051-R5..R9: the branch player of a branch game's `players[0]` model session.
    player: Option<Box<BranchRun>>,
}

struct TrunkSnapshots {
    rules: Snapshots,
    game: Arc<GameBranch>,
    taken: Vec<SnapshotTaken>,
}

/// 051-R5..R9 branch player state.
struct BranchRun {
    cfg: BranchPlayer,
    /// The trunk's act log of this session as a prefix checker (051-R5 ①); `None` when it could
    /// not be read (then `pending` says why).
    prefix: Option<ForcedPrefix>,
    /// The trunk's state at t (051-R3 file, sha256 checked).
    saved: Option<HostState>,
    /// A trunk input failed to load: the branch leaves at its first observation (051-R5 ①).
    pending: Option<(InvalidReason, String)>,
    invalid: Option<Invalid>,
    /// Prefix observations (i <= i_t) seen.
    seen: u32,
    /// The state was restored at t; observations from t on play live (051-R6).
    at_t: bool,
    /// Decisions from t on (0 = the t decision).
    k: u32,
    intervention: Intervention,
    samples: SampleTracker,
    dilution: DilutionTracker,
    state_check: Option<StateCheck>,
}

impl BranchRun {
    /// Loads the trunk inputs (051-R5, 051-R6 ②). A missing or incomplete act log or state file
    /// is `missing_input`, a state file whose sha256 or content is wrong is `state`; either makes
    /// the branch invalid at its first observation. A bad alternative is a configuration error
    /// (the plan load checks it, 020-R22 ③).
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
            k: 0,
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

/// 051-R3: writes one slot state as a state file (created or replaced: a game played again
/// overwrites its earlier attempt, 020-R22 ②), synced to the disk or the shared drive's server
/// (061), and returns the content's sha256.
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

/// 051-R6 (`verify_state`): bit equality and the largest absolute difference of two states
/// (different part shapes: not equal, no finite difference).
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
            // NaN anywhere makes the difference not a number (recorded as null).
            max = if d.is_nan() || max.is_nan() { f64::NAN } else { max.max(d) };
        }
    }
    StateCheck::new(bit_equal, max)
}

/// 051-R9 ② / 050-R9: the session metrics of a branch-game model session under the
/// [`uranium_env::branch::OUTCOME_METRICS`] names: the 042 / 050-R7 metrics (game seconds as in
/// `ModelStats`), `held_s` (`scv_min_1m > 4`, 050-R7) and `held_8m` (`scv_8m >= 8`, 050-R6 ③; a
/// game that ended before 8:00 fails).
pub fn outcome_metrics(m: &ScoutMetrics) -> serde_json::Map<String, serde_json::Value> {
    let secs = |f: Option<u32>| f.map(seconds);
    let v = serde_json::json!({
        "held_8m": m.scv_8m.is_some_and(|n| n >= 8),
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

/// 041-R28 ① (AMB-041-17): the JSON-lines file of one game's model decisions, written exactly as
/// the session records them in agent memory (`DecisionLogLine`), then an end line. A file that
/// cannot be opened or written is reported once and left without its end line, so the relabel
/// build sees it as incomplete (041-R28 ④); the game itself is never affected.
pub struct DecisionLog {
    path: std::path::PathBuf,
    w: Option<std::io::BufWriter<std::fs::File>>,
    decisions: u64,
    /// The session's own BWAPI player id (end line).
    player: u8,
}

impl DecisionLog {
    /// Creates (truncates: a game played again overwrites its earlier attempt) the file at `path`.
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

    /// `forced`: a branch player's prefix decision (051-R5 ⑤), written as `"forced":true`.
    pub fn decision(&mut self, frame: u32, action: u16, delay_bucket: usize, units: &[u32], forced: bool) {
        self.line(&DecisionLogLine::Decision { frame, action, delay_bucket: delay_bucket as u32, units: units.to_vec(), forced });
        self.decisions += 1;
    }

    /// The game ended (`GameEnd`): the end line, then flush and sync (the file may be on the
    /// shared drive, 020-R15). Later calls write nothing.
    pub fn end(&mut self, end_frame: Option<u32>) {
        let decisions = self.decisions;
        self.line(&DecisionLogLine::End { decisions, end_frame, player: self.player });
        self.close();
    }

    /// The session failed or the bridge disconnected mid-game: the decisions so far are flushed
    /// without the end line, so the relabel build reads the log as incomplete (041-R28 ④).
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
            scout: ScoutTracker::for_game(info, RewardConfig::default()),
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
        })
    }

    /// 020-R22 / 051: this session plays a plan game with 051 keys. Every such model session
    /// reports its act-log extras ([`Self::act_log_extra`], 051-R2); the `players[0]` session
    /// (`players_index == 0`) also checks the snapshot rules (051-R3) or, in a branch game, plays
    /// as the branch player (051-R5..R9). A branch player never surrenders (051-R1 ③: its prefix
    /// has no inference to judge on).
    pub fn with_branch(mut self, game: Arc<GameBranch>, players_index: u8) -> Result<Self> {
        let mut b = BranchSession { obs: 0, last: LineExtra::default(), trunk: None, player: None };
        if players_index == 0 {
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

    /// 051-R2: the model's part of the act-log line of the last observation (`None` outside plan
    /// games with 051 keys).
    pub fn act_log_extra(&mut self) -> Option<LineExtra> {
        self.branch.as_mut().map(|b| std::mem::take(&mut b.last))
    }

    /// 051: the session record part (snapshots, branch outcome without `end`).
    pub fn branch_record(&mut self) -> Option<BranchSessionPart> {
        let m = self.scout_metrics();
        let b = self.branch.as_ref()?;
        let metrics = outcome_metrics(&m);
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
        Some(BranchSessionPart { snapshots: b.trunk.as_ref().map(|t| t.taken.clone()).unwrap_or_default(), outcome })
    }

    /// Agent memory (tests: 051-AC1 forced prefix memory = the trunk's).
    pub fn memory(&self) -> &AgentMemory {
        &self.mem
    }

    fn scout_metrics(&self) -> ScoutMetrics {
        self.reward.as_ref().and_then(|r| r.scout_metrics()).unwrap_or_else(|| self.scout.metrics())
    }

    /// 051-R5 (observation `i`, i <= i_t of a branch player): checks the observation against the
    /// trunk and, before t, plays the recorded decision without inference. `Some(act)` = send it
    /// (a forced `Act`, or the chat-less leave of an invalid branch); `None` = play this
    /// observation through the normal path (t restores there, after t plays live).
    ///
    /// Throughput note (032-R14): prefix observations send no request, so while branch sessions
    /// are in their prefix the inference server's batches hold fewer rows and each waits up to
    /// `max_wait` for rows that do not come. That changes speed only, never a result.
    fn branch_prefix(&mut self, f: &RawFrame) -> Result<Option<Act>> {
        let Some(b) = self.branch.as_deref_mut() else { return Ok(None) };
        let i = b.obs - 1;
        let Some(run) = b.player.as_deref_mut() else { return Ok(None) };
        if run.invalid.is_some() {
            // 051-R5 ①: already leaving.
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
        // 051-R5 ②: memory follows the observation and records the trunk's decision.
        let obs = extract(f, &mut self.mem, &self.map, self.z.as_ref());
        if run.cfg.verify_state {
            // 051-R6 test option: build the state over the prefix, drop the sampled action.
            self.sess.act(obs)?;
        }
        let d = line.decision.clone().expect("ForcedPrefix holds a model session's log");
        self.mem.record_action(d.action, d.delay_bucket as usize, &d.units);
        // 051-R5 ⑤: the 041-R28 log gets the forced decision marked `forced`.
        if let Some(l) = self.decision_log.as_mut() {
            l.decision(f.frame, d.action, d.delay_bucket as usize, &d.units, true);
        }
        b.last = LineExtra { decision: Some(d), state: None, state_sha256: None, forced: true };
        // 051-R5 ③: the recorded Act as it was sent.
        Ok(Some(line.act))
    }

    /// 051-R3 / 051-R6 / 051-R7 after feature extraction, before inference: a trunk saves the
    /// state at a decision a snapshot rule picks; a branch player restores the trunk's state at t
    /// and narrows `type_mask` from t on. `Some(act)` = the branch became invalid (leave).
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
                    // The game goes on; the plan tool skips a point without its state (051-R10).
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
                // 051-R6 ②: restore; a refused state makes the branch invalid.
                if let Err(e) = self.sess.set_state(saved) {
                    run.fail(InvalidReason::State, Some(i), Some(f.frame), &format!("{e:#}"));
                    return Ok(Some(quit_act()));
                }
                run.at_t = true;
            }
            if run.at_t {
                // 051-R7 ④: the narrowed mask goes to the model and comes back for the executor.
                run.intervention.apply_to(f.frame, obs);
            }
        }
        Ok(None)
    }

    /// 051-R2 / 051-R7 ⑤ / 051-R9 after the executor: the act-log decision and, from t on, the
    /// intervention's executor verdict, the time-point samples and the dilution count.
    fn branch_after_act(&mut self, frame: u32, action: u16, delay_bucket: usize, ids: &[u32], rejected: bool, scalar: &[f32]) {
        let Some(b) = self.branch.as_deref_mut() else { return };
        b.last.decision = Some(uranium_env::branch::Decision { action, delay_bucket: delay_bucket as u32, units: ids.to_vec(), rejected });
        if let Some(run) = b.player.as_deref_mut().filter(|r| r.at_t) {
            run.intervention.note_executor(frame, rejected);
            run.samples.on_decision(run.k, frame, scalar);
            run.dilution.on_decision(frame, action);
            run.k += 1;
        }
    }

    /// 012-R14 / 012-R18: SC:R (realtime) sessions get no `CommandFailed`, so `last_failed` is 0
    /// whatever the inference server was started with.
    pub fn with_zero_last_failed(mut self) -> Self {
        self.mem.zero_last_failed = true;
        self.stats.zero_last_failed = true;
        self
    }

    /// 042-R14: judges collapse concede before every inference for this RL league game (`side` =
    /// this session's `GameSpec.players` index).
    pub fn with_concede(mut self, game: Arc<ConcedeGame>, side: u8) -> Self {
        self.concede = Some((CollapseJudge::new(game.cfg), game, side));
        self
    }

    /// Records the trajectory through `o` with 033 rewards for strategy `z` (042), r_scout /
    /// r_survive and the 042 metrics by `reward` (033-R9 / R10, AMB-042-16).
    pub fn with_observer(mut self, z: Option<Strategy>, reward: RewardConfig, o: Box<dyn SessionObserver>) -> Self {
        let scout = self.scout.clone().with_config(reward);
        self.reward = Some(RewardTracker::new(z.clone(), self.self_player).with_scout(scout));
        self.z = z;
        self.observer = Some(o);
        self
    }

    /// Game over (042, 033-R7): the frames after the last decision and r_win go to the observer.
    /// E-8 / 010-R12: the dense reward of those frames comes from the bridge's final frame (own
    /// vision, events since the last decision); without one only r_win is added.
    pub fn on_end(&mut self, outcome: Option<Outcome>, final_frame: Option<&RawFrame>) {
        if let Some(l) = self.decision_log.as_mut() {
            l.end(final_frame.map(|f| f.frame));
        }
        // 051-R7: a `force` that never fired is missed; a branch whose game finished before t never
        // reached the trunk's observation i_t at f_t (051-R5 ①: `frame`). An abort (disconnect,
        // session error, `GameEnd` Aborted / AgentTimeout: `outcome` None) is an infrastructure
        // failure, not an invalid branch: the record's `error` / `end: null` tells it apart and
        // the evaluation job plays the game again (020-R15 ④).
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
        // 042 metrics of a session without a reward tracker: the final frame too (010-R12).
        if let (None, Some(f)) = (self.reward.as_ref(), final_frame) {
            self.scout.step(f);
        }
        if let Some(o) = self.observer.as_mut() {
            let mut after = match (outcome, final_frame, self.reward.as_mut()) {
                (Some(_), Some(f), Some(r)) => r.step(f),
                _ => RewardParts::default(),
            };
            // 042-R14 / 033-R7: the frames up to the concede belong to the last real decision too.
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

    /// Mid-game failure or disconnect (no `GameEnd`): the 041-R28 decision log stays incomplete,
    /// then `on_end(None, None)` as before (042-R3).
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
        // 042-R14: in an RL league game the leave is claimed once per game, so the other model
        // session never leaves too (a concede or a surrender of the other side came first).
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

    /// 042-R14 before inference: `Some(act)` = concede now (chat + leave, 010-R13). The concede
    /// is not a decision: no inference, no trajectory step, not counted in `decisions`. Only the
    /// first session of the game that holds concedes; after that the other one stops judging.
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
        // The events of this frame happened after the last decision (033-R7).
        self.concede_reward = self.reward.as_mut().map(|r| r.step(f));
        elog!("[agent] 042-R14: collapse concede at frame {} (supply_used {}, players index {side})", f.frame, f.self_state.supply_used);
        Some(Act { cmds: vec![], advance_frames: 1, valid_until_frame: None, surrender: Some(DEFAULT_CHAT.to_string()) })
    }

    pub fn on_frame(&mut self, f: &RawFrame) -> Result<Act> {
        // 042 metrics (AMB-042-16) of a session without a reward tracker; RL sessions count them in
        // their reward tracker below. No reward effect.
        if self.reward.is_none() {
            self.scout.step(f);
        }
        if let Some(b) = self.branch.as_deref_mut() {
            b.obs += 1;
            b.last = LineExtra::default();
        }
        if let Some(act) = self.concede_check(f) {
            return Ok(act);
        }
        // 051-R5: a branch player's forced prefix (no inference) or its leave.
        if let Some(act) = self.branch_prefix(f)? {
            return Ok(act);
        }
        let t0 = Instant::now();
        // z absent unless an RL session set one (041-AC2 / AMB-041-2, 033-R8).
        let mut obs = extract(f, &mut self.mem, &self.map, self.z.as_ref());
        // 051-R3 snapshot / 051-R6 restore / 051-R7 intervention (plan games with 051 keys only).
        if let Some(act) = self.branch_before_act(f, &mut obs)? {
            return Ok(act);
        }
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
        // 041-R28 ①: the same decision, executor refusal or not (surrender decisions return
        // above and are never recorded in memory, so never logged).
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
        s
    }
}

/// Precompute rebuilt from the map file's resources (ignores a cached replay-build file).
fn live_rebuild(info: &GameInfo) -> Option<uranium_env::features::MapPrecompute> {
    let hash: String = info.map_hash.iter().map(|b| format!("{b:02x}")).collect();
    let r = uranium_env::maps::load_resources(&maps_dir().ok()?, &hash)?;
    Some(uranium_env::features::mapgen::precompute(&info.static_map, &r.resources))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 050-R7 / 050-AC3 (AMB-042-19): the `results.jsonl` model row (a session record's `model`,
    /// written in field order) has `scv_min_1m`, `army_s`, `scv_8m` right after `scv_4m`, empty
    /// values as `null`.
    #[test]
    fn model_row_keys_050_r7() {
        let s = ModelStats { scv_4m: Some(20), scv_min_1m: Some(5), army_s: Some(185.0), scv_8m: None, ..Default::default() };
        let text = serde_json::to_string(&s).unwrap();
        assert!(text.ends_with(r#""scv_4m":20,"scv_min_1m":5,"army_s":185.0,"scv_8m":null}"#), "{text}");
    }

    /// 051-R9 ② / 050-R9: every 051-R11 ④ outcome key is there; `held_8m` = `scv_8m >= 8` (a game
    /// that ended before 8:00 fails), `held_s` = `scv_min_1m > 4`, seconds as in `ModelStats`.
    #[test]
    fn outcome_metrics_hold_definitions_051_r9() {
        let m = ScoutMetrics { scv_8m: Some(8), scv_min_1m: Some(4), bunker_frame: Some(240), ..Default::default() };
        let v = outcome_metrics(&m);
        for k in uranium_env::branch::OUTCOME_METRICS {
            assert!(v.contains_key(*k), "{k}");
        }
        assert_eq!((v["held_8m"].as_bool(), v["held_s"].as_bool(), v["bunker_s"].as_f64()), (Some(true), Some(false), Some(10.0)));
        let v = outcome_metrics(&ScoutMetrics { scv_8m: None, scv_min_1m: Some(5), ..Default::default() });
        assert_eq!((v["held_8m"].as_bool(), v["held_s"].as_bool()), (Some(false), Some(true)));
    }

    /// 051-R6 `verify_state`: bit equality (so -0.0 != 0.0), the largest difference, and other
    /// shapes or NaN give no finite difference.
    #[test]
    fn compare_states_051_r6() {
        let s = |v: Vec<f32>| HostState { parts: vec![(vec![1, v.len()], v)] };
        assert_eq!(compare_states(&s(vec![1.0, 2.0]), &s(vec![1.0, 2.0])), StateCheck { bit_equal: true, max_abs_diff: Some(0.0) });
        assert_eq!(compare_states(&s(vec![0.0, 2.0]), &s(vec![-0.0, 2.5])), StateCheck { bit_equal: false, max_abs_diff: Some(0.5) });
        assert_eq!(compare_states(&s(vec![1.0]), &s(vec![1.0, 2.0])), StateCheck { bit_equal: false, max_abs_diff: None });
        assert_eq!(compare_states(&s(vec![f32::NAN]), &s(vec![1.0])).max_abs_diff, None);
    }
}
