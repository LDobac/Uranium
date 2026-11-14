//! Agent server library (spec 001-R6). Bridges connect to it; every connection is one player
//! of one game (`Hello.game_id` / `Hello.slot`, 010-R9) and is served by its own policy instance,
//! chosen from the routing table (020-R11). Used by the `uranium-agent` binary and embedded by the
//! `uranium-env` scheduler (crate `uranium-runner`, 020).

/// `eprintln!` that never panics: inside `uranium-train rl` this runs in a training child whose
/// stderr is a pipe to its watchdog; when the watchdog is killed hard the pipe closes and
/// `eprintln!` would panic (EPIPE) instead of letting the run save, push and release (060-R6).
macro_rules! elog {
    ($($t:tt)*) => {{
        use std::io::Write as _;
        let _ = writeln!(std::io::stderr(), $($t)*);
    }};
}

pub mod broker;
pub mod infer;
pub mod model_policy;
pub mod summary;
#[cfg(feature = "testkit")]
pub mod testkit;

use std::collections::{BTreeMap, HashMap};
use std::fs::OpenOptions;
use std::io::{BufReader, BufWriter, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde::Serialize;
use uranium_proto::*;

use crate::infer::InferenceHandle;
use crate::model_policy::{LearnerHook, ModelPolicy, ModelStats};

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum PolicyKind {
    /// T0 rule-based bot for our own race (021): picked from `GameInfo` at game start.
    ScriptedT0,
    /// Checkpoint policy (020-R11), Terran only.
    Model,
}

/// What plays one session (020-R11).
#[derive(Clone)]
pub enum PolicySpec {
    ScriptedT0,
    Model(InferenceHandle),
    /// A model session that records its trajectory for the learner (042).
    Learner(Arc<dyn LearnerHook>),
}

impl std::fmt::Debug for PolicySpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ScriptedT0 => write!(f, "ScriptedT0"),
            Self::Model(h) => write!(f, "Model({h:?})"),
            Self::Learner(l) => write!(f, "Learner({:?})", l.handle()),
        }
    }
}

impl PolicySpec {
    /// Session record label (020-R5 policy version).
    pub fn label(&self) -> String {
        match self {
            Self::ScriptedT0 => format!("t0@{}", env!("CARGO_PKG_VERSION")),
            Self::Model(h) => infer::model_policy_label(&h.checkpoint_id, h.sample),
            Self::Learner(l) => format!("learner:{}:{:?}", l.handle().checkpoint_id, l.handle().sample).to_lowercase(),
        }
    }
}

/// `(game_id, GameSpec.players index) -> policy` registered by the scheduler before `StartGame`
/// (020-R11). The index comes from the player name (`spec_index`), not the BWAPI player id.
/// Sessions not in the table use the server default.
#[derive(Clone, Default)]
pub struct Routes(Arc<Mutex<HashMap<(u64, u8), PolicySpec>>>);

impl Routes {
    pub fn insert(&self, game_id: u64, slot: u8, spec: PolicySpec) {
        self.0.lock().unwrap().insert((game_id, slot), spec);
    }

    /// Removes the entry: a game id is played once.
    pub fn take(&self, game_id: u64, slot: u8) -> Option<PolicySpec> {
        self.0.lock().unwrap().remove(&(game_id, slot))
    }
}

/// A policy controls one player for one game.
pub trait Policy: Send {
    fn on_privileged(&mut self, _p: &PrivilegedSummary) {}
    fn on_frame(&mut self, f: &RawFrame) -> Result<Act>;
    /// Model policies report latency and executor rejections (020-R5, 031-R4).
    fn model_stats(&mut self) -> Option<ModelStats> {
        None
    }
    /// T0 Terran sessions report the 042 / 050-R7 metrics and the teacher record (050-R8).
    fn scripted_stats(&mut self) -> Option<ScriptedStats> {
        None
    }
    /// Game over; `None` = Aborted or disconnected (042-R3). `final_frame`: the own-vision state at
    /// the end with the events since the last frame (010-R12), when the bridge sent one.
    fn on_end(&mut self, _outcome: Option<uranium_env::reward::Outcome>, _final_frame: Option<&RawFrame>) {}
    /// The session failed or the bridge disconnected mid-game (no `GameEnd`): `on_end(None, None)`
    /// for the 042-R3 bookkeeping, and a model's 041-R28 decision log is left without its end line.
    fn on_abort(&mut self) {
        self.on_end(None, None)
    }
    /// 051-R2: the policy's part of the act-log line of the observation just answered (model
    /// sessions of plan games with 051 keys); `None` = none (T0 lines carry no decision).
    fn act_log_extra(&mut self) -> Option<model_policy::LineExtra> {
        None
    }
    /// 051: the model session's part of the session record `branch` (snapshots, metrics, branch
    /// outcome); `None` for T0 sessions and outside plan games with 051 keys.
    fn branch_record(&mut self) -> Option<model_policy::BranchSessionPart> {
        None
    }
}

pub use uranium_scripted::T0Build;
pub use uranium_env::branch::{BranchBoard, GameBranch};

/// One session's GameEnd report (020-R13).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionEnd {
    /// BWAPI player id of the reporting session (`Hello.slot`).
    pub slot: u8,
    pub winner: Option<u8>,
    pub frame: u32,
    pub reason: EndReason,
}

/// 020-R13 (2026-10-17, AMB-020-3): when the sessions of one game report different winners, the
/// session that ended last (largest end frame) decides; on a tie the first report stands. The
/// loser's client can mark the winner as defeated too on its final frame (OpenBW leave timing,
/// not a desync). Returns the chosen report and whether the reports disagreed.
pub fn game_result(ends: &[SessionEnd]) -> Option<(SessionEnd, bool)> {
    let first = *ends.first()?;
    let chosen = ends.iter().skip(1).fold(first, |best, e| if e.frame > best.frame { *e } else { best });
    let disagree = ends.iter().any(|e| e.winner != first.winner);
    if mutual_leave(ends) {
        return Some((SessionEnd { winner: None, reason: EndReason::Aborted, ..chosen }, disagree));
    }
    Some((chosen, disagree))
}

/// 020-R13: both sessions say the other one left and claim the win. A real leave (a bot's GG)
/// is reported as a loss by the leaver, so this is a connection drop (OpenBW drops a client that
/// stops syncing, 011-AC7): the game is Aborted and its segments revoked (042-R3).
pub fn mutual_leave(ends: &[SessionEnd]) -> bool {
    ends.len() == 2 && ends.iter().all(|e| e.reason == EndReason::Surrender && e.winner == Some(e.slot))
}

/// GameEnd reports by game id, filled by the agent server (020-R13). The RL league reads the
/// reconciled result from here once hostd says the game is over (042 win table, 033 r_win).
/// It also counts the sessions that said a valid Hello, so the scheduler can tell a game whose
/// bridge never reached the agent (020-R4).
#[derive(Clone, Default)]
pub struct EndBoard(Arc<Mutex<Board>>);

#[derive(Default)]
struct Board {
    ends: HashMap<u64, Vec<SessionEnd>>,
    hellos: HashMap<u64, usize>,
}

impl EndBoard {
    pub fn push(&self, game_id: u64, e: SessionEnd) {
        self.0.lock().unwrap().ends.entry(game_id).or_default().push(e);
    }

    /// A session of the game passed the Hello check (020-R4).
    pub fn hello(&self, game_id: u64) {
        *self.0.lock().unwrap().hellos.entry(game_id).or_default() += 1;
    }

    /// Sessions of the game that passed the Hello check so far.
    pub fn connected(&self, game_id: u64) -> usize {
        self.0.lock().unwrap().hellos.get(&game_id).copied().unwrap_or(0)
    }

    /// Removes and returns the reports of a game (and forgets its Hello count).
    pub fn take(&self, game_id: u64) -> Vec<SessionEnd> {
        let mut b = self.0.lock().unwrap();
        b.hellos.remove(&game_id);
        b.ends.remove(&game_id).unwrap_or_default()
    }
}

/// 042-R14 collapse concede by game id: the RL run loop registers each RL league game when
/// `[concede] enabled` and takes it out when the game is over; the model sessions of a listed game
/// judge before inference. Games not listed (plan / evaluation games, 050) never concede.
#[derive(Clone, Default)]
pub struct ConcedeBoard(Arc<Mutex<HashMap<u64, Arc<uranium_env::concede::ConcedeGame>>>>);

impl ConcedeBoard {
    pub fn insert(&self, game_id: u64, game: Arc<uranium_env::concede::ConcedeGame>) {
        self.0.lock().unwrap().insert(game_id, game);
    }

    pub fn get(&self, game_id: u64) -> Option<Arc<uranium_env::concede::ConcedeGame>> {
        self.0.lock().unwrap().get(&game_id).cloned()
    }

    /// Removes the game and returns its state (who conceded, if anyone).
    pub fn take(&self, game_id: u64) -> Option<Arc<uranium_env::concede::ConcedeGame>> {
        self.0.lock().unwrap().remove(&game_id)
    }
}

/// T0 build per (game id, `GameSpec.players` index) for scenario bots (050 E-Regression).
/// The scheduler fills it before games start; players not listed play the standard build.
pub type T0Builds = Arc<Mutex<HashMap<(u64, u8), T0Build>>>;

/// Our `GameSpec.players` index: uranium-hostd names each player `p<index>-<name>` (020-R16)
/// or `uranium-p<index>` because the BWAPI player id (`Hello.slot`) need not follow launch
/// order. Falls back to the slot.
pub fn spec_index(info: &GameInfo, slot: u8) -> u8 {
    let name = info.players.iter().find(|p| p.is_self).map(|p| p.name.as_str()).unwrap_or("");
    name_index(name).unwrap_or(slot)
}

/// The `GameSpec.players` index in a hostd player name (020-R11, 020-R16).
pub fn name_index(name: &str) -> Option<u8> {
    if let Some(i) = name.strip_prefix("uranium-p") {
        return i.parse().ok();
    }
    let (i, _) = name.strip_prefix('p')?.split_once('-')?;
    i.parse().ok()
}

/// 012-R7: a realtime Act is valid until the observed frame plus this many frames (about 1 s,
/// AMB-012-5); the bridge drops it after that.
pub const REALTIME_ACT_FRAMES: u32 = 24;

/// `realtime`: the bridge said `Mode::Realtime` (SC:R, 012-R18). `concede`: the 042-R14 state of
/// an RL league game and this session's `GameSpec.players` index; model sessions judge with it.
/// `branch`: the 051 settings of a plan game with 051 keys (020-R22) and this session's
/// `GameSpec.players` index; model sessions log, snapshot or play the branch with them.
#[allow(clippy::too_many_arguments)]
pub fn make_policy(spec: &PolicySpec, game_id: u64, info: &GameInfo, build: T0Build, realtime: bool, concede: Option<(Arc<uranium_env::concede::ConcedeGame>, u8)>, branch: Option<(Arc<GameBranch>, u8)>) -> Result<Box<dyn Policy>> {
    let model = |p: ModelPolicy| -> Result<ModelPolicy> {
        let p = if realtime { p.with_zero_last_failed() } else { p };
        let p = match concede.clone() {
            Some((game, side)) => p.with_concede(game, side),
            None => p,
        };
        match branch.clone() {
            Some((game, index)) => p.with_branch(game, index),
            None => Ok(p),
        }
    };
    let terran = || -> Result<()> {
        let me = info.players.iter().find(|p| p.is_self).map(|p| p.race);
        if me != Some(Race::Terran) {
            bail!("model policy plays Terran only, got {me:?}");
        }
        Ok(())
    };
    Ok(match spec {
        PolicySpec::ScriptedT0 => Box::new(ScriptedPolicy::new(info, build)?),
        PolicySpec::Model(h) => {
            terran()?;
            Box::new(model(ModelPolicy::new(h, info)?)?)
        }
        PolicySpec::Learner(l) => {
            terran()?;
            let (z, observer) = l.session(game_id, info);
            Box::new(model(ModelPolicy::new(l.handle(), info)?.with_observer(z, l.reward_config(), observer))?)
        }
    })
}

/// 033 outcome for our player from the bridge's GameEnd; `None` when the game did not finish
/// (Aborted, agent timeout): such episodes are left out of training (042-R3).
pub fn end_outcome(end: &GameEnd, me: u8) -> Option<uranium_env::reward::Outcome> {
    use uranium_env::reward::Outcome;
    match end.reason {
        EndReason::Aborted | EndReason::AgentTimeout => None,
        EndReason::TimeLimitDraw => Some(Outcome::Draw),
        EndReason::Elimination | EndReason::Surrender => Some(match end.winner {
            Some(w) if w == me => Outcome::Win,
            Some(_) => Outcome::Loss,
            None => Outcome::Draw,
        }),
    }
}

impl Policy for Box<dyn uranium_scripted::Bot> {
    fn on_frame(&mut self, f: &RawFrame) -> Result<Act> {
        Ok(self.act(f))
    }
}

/// 050-R8 session record part of a T0 Terran session (`scripted`): the 042 정찰·대응 지표 and
/// 050-R7 세션 지표 with the same keys, definitions and code (`ScoutTracker`) as the model row,
/// folded over every observation of the session and its final frame, plus the 021-R8 teacher's
/// record (`react`, `terran_react` only), then the 052 ops / early-rush combat metrics (`ops`,
/// `rush_combat`, the model row's keys and code, 052-R6), then the 022 teacher's record (`tp`,
/// `terran_tp` only, 022-R10). Game seconds (frame / 24).
#[derive(Serialize, Debug, Clone, Default, PartialEq)]
pub struct ScriptedStats {
    pub scout_main_s: Option<f64>,
    pub scout_types_4m: u32,
    pub scout_types_6m: u32,
    pub barracks_s: Option<f64>,
    pub bunker_s: Option<f64>,
    pub scv_4m: Option<u32>,
    pub scv_min_1m: Option<u32>,
    pub army_s: Option<f64>,
    pub scv_8m: Option<u32>,
    pub react: Option<ReactRecord>,
    /// 052-R6: appended after the existing keys (their order unchanged).
    pub ops: uranium_env::ops::OpsMetrics,
    pub rush_combat: Option<uranium_env::ops::RushCombat>,
    /// 022-R10: appended after the existing keys (their order unchanged).
    pub tp: Option<TpRecord>,
}

/// 022 teacher record in game seconds (`uranium_scripted::TpStats`, 022-R10).
#[derive(Serialize, Debug, Clone, Default, PartialEq)]
pub struct TpRecord {
    /// Enemy main found (an enemy building within 10 tiles of a start location).
    pub main_s: Option<f64>,
    /// First rush identification (022-R4) and its rule (`gates`, `zealots`, `proxy_gate`, `rax`,
    /// `marines`, `proxy_rax`, `army`).
    pub rush_s: Option<f64>,
    pub rush_rule: Option<uranium_scripted::TpRule>,
    /// Macro → defence switches; first defence → macro switch.
    pub defences: u32,
    pub macro_s: Option<f64>,
    /// First order of the refinery, factory, machine shop, siege mode research, siege tank and
    /// natural command center.
    pub refinery_s: Option<f64>,
    pub factory_s: Option<f64>,
    pub shop_s: Option<f64>,
    pub siege_s: Option<f64>,
    pub tank_s: Option<f64>,
    pub expand_s: Option<f64>,
    /// First observation with the attack condition (022-R8).
    pub wave_s: Option<f64>,
    /// Branch at the end (`macro` / `defence`).
    pub branch: uranium_scripted::Branch,
    /// First enemy army within 20 tiles of an own command center; first own bunker.
    pub contact_s: Option<f64>,
    pub bunker_s: Option<f64>,
    /// First time each rule held (before 8:00, in any branch).
    pub rules_s: BTreeMap<uranium_scripted::TpRule, f64>,
}

/// 021-R8 teacher record in game seconds (`uranium_scripted::ReactStats`).
#[derive(Serialize, Debug, Clone, Default, PartialEq)]
pub struct ReactRecord {
    /// Enemy main found (the teacher's rule: an enemy building within 10 tiles of a start location).
    pub main_s: Option<f64>,
    /// First rush identification (021-R8 R) and its rule (`pool_start`, `pool_done`, `lings`,
    /// `army`, `drones`, `one_base`).
    pub rush_s: Option<f64>,
    pub rush_rule: Option<uranium_scripted::RushRule>,
    /// Macro → defence switches.
    pub defences: u32,
    /// First defence → macro switch.
    pub macro_s: Option<f64>,
    /// Natural command center ordered.
    pub expand_s: Option<f64>,
    /// Branch at the end (`macro` / `defence`).
    pub branch: uranium_scripted::Branch,
    /// 050-R8 (rule advisory §2): estimated start of the first unfinished enemy pool seen, the
    /// first pool sighting, the most zerglings in one observation before 8:00, the first enemy
    /// army within 20 tiles of an own command center, the first own bunker (construction start).
    pub pool_start_s: Option<f64>,
    pub pool_seen_s: Option<f64>,
    pub lings_max: u32,
    pub contact_s: Option<f64>,
    pub bunker_s: Option<f64>,
    /// First time each rule held (before 8:00, in any branch), e.g. `{"army": 250.3}`.
    pub rules_s: BTreeMap<uranium_scripted::RushRule, f64>,
}

/// A T0 session (021): the scripted bot, and for a Terran bot the 042 / 050-R7 and 052 metrics
/// trackers.
pub struct ScriptedPolicy {
    bot: Box<dyn uranium_scripted::Bot>,
    scout: Option<uranium_env::scout::ScoutTracker>,
    ops: Option<uranium_env::ops::OpsTracker>,
}

impl ScriptedPolicy {
    /// 021-R8 / 022-R10: the teacher builds read the 031 map precompute (base locations, chokes)
    /// with the model's session rule: a target map without one refuses the session, any other map
    /// plays on without (no expansion, bunkers toward the map centre).
    pub fn new(info: &GameInfo, build: T0Build) -> Result<Self> {
        let race = info.players.iter().find(|p| p.is_self).map(|p| p.race);
        let terran = race == Some(Race::Terran);
        let (hints, pre) = if build.needs_map_hints() && terran {
            let family = uranium_env::maps::MapFamilies::load()?.classify(&info.map_name, "");
            let (pre, _) = uranium_env::maps::live_precompute_for_session(uranium_env::maps::maps_dir(), info, &family)?;
            if pre.is_none() {
                elog!("[agent] WARNING: no map precompute for {:?}; the teacher does not expand", info.map_name);
            }
            (pre.as_ref().map(|p| uranium_scripted::MapHints { base_locs: p.base_locs.clone(), chokes: p.chokes.clone() }).unwrap_or_default(), pre)
        } else {
            // 052-R6: the ops metrics of any other T0 Terran session read the precompute when there
            // is one; a missing one never refuses the session.
            let pre = if terran { uranium_env::maps::maps_dir().ok().and_then(|d| uranium_env::maps::live_precompute(&d, info).0) } else { None };
            (uranium_scripted::MapHints::default(), pre)
        };
        let scout = terran.then(|| uranium_env::scout::ScoutTracker::for_game(info, uranium_env::scout::RewardConfig::default()));
        let ops = terran.then(|| uranium_env::ops::OpsTracker::for_game(info, pre.as_ref()));
        Ok(Self { bot: uranium_scripted::for_game_hints(info, build, hints), scout, ops })
    }
}

impl Policy for ScriptedPolicy {
    fn on_frame(&mut self, f: &RawFrame) -> Result<Act> {
        if let Some(s) = self.scout.as_mut() {
            s.step(f);
        }
        // 052-R6: every observation (8 frames apart) and the final frame below.
        if let Some(o) = self.ops.as_mut() {
            o.step(f);
        }
        Ok(self.bot.act(f))
    }

    fn on_end(&mut self, _outcome: Option<uranium_env::reward::Outcome>, final_frame: Option<&RawFrame>) {
        // 010-R12: the final frame counts like the model row's (050-R7).
        if let (Some(s), Some(f)) = (self.scout.as_mut(), final_frame) {
            s.step(f);
        }
        if let (Some(o), Some(f)) = (self.ops.as_mut(), final_frame) {
            o.step(f);
        }
    }

    fn scripted_stats(&mut self) -> Option<ScriptedStats> {
        let m = self.scout.as_ref()?.metrics();
        let secs = |f: Option<u32>| f.map(uranium_env::scout::seconds);
        let react = self.bot.react_stats().map(|r| ReactRecord {
            main_s: secs(r.main_frame),
            rush_s: secs(r.rush_frame),
            rush_rule: r.rush_rule,
            defences: r.defences,
            macro_s: secs(r.macro_frame),
            expand_s: secs(r.expand_frame),
            branch: r.branch,
            pool_start_s: secs(r.pool_start_frame),
            pool_seen_s: secs(r.pool_seen_frame),
            lings_max: r.lings_max,
            contact_s: secs(r.contact_frame),
            bunker_s: secs(r.bunker_frame),
            rules_s: r.rule_frames.iter().map(|(k, f)| (*k, uranium_env::scout::seconds(*f))).collect(),
        });
        let tp = self.bot.tp_stats().map(|r| TpRecord {
            main_s: secs(r.main_frame),
            rush_s: secs(r.rush_frame),
            rush_rule: r.rush_rule,
            defences: r.defences,
            macro_s: secs(r.macro_frame),
            refinery_s: secs(r.refinery_frame),
            factory_s: secs(r.factory_frame),
            shop_s: secs(r.shop_frame),
            siege_s: secs(r.siege_frame),
            tank_s: secs(r.tank_frame),
            expand_s: secs(r.expand_frame),
            wave_s: secs(r.wave_frame),
            branch: r.branch,
            contact_s: secs(r.contact_frame),
            bunker_s: secs(r.bunker_frame),
            rules_s: r.rule_frames.iter().map(|(k, f)| (*k, uranium_env::scout::seconds(*f))).collect(),
        });
        Some(ScriptedStats {
            scout_main_s: secs(m.main_frame),
            scout_types_4m: m.types_4m,
            scout_types_6m: m.types_6m,
            barracks_s: secs(m.barracks_frame),
            bunker_s: secs(m.bunker_frame),
            scv_4m: m.scv_4m,
            scv_min_1m: m.scv_min_1m,
            army_s: secs(m.army_frame),
            scv_8m: m.scv_8m,
            react,
            ops: self.ops.as_ref().map(|o| o.ops()).unwrap_or_default(),
            rush_combat: self.ops.as_ref().and_then(|o| o.rush_combat()),
            tp,
        })
    }
}

impl Policy for ModelPolicy {
    fn on_privileged(&mut self, p: &PrivilegedSummary) {
        ModelPolicy::on_privileged(self, p)
    }

    fn on_frame(&mut self, f: &RawFrame) -> Result<Act> {
        ModelPolicy::on_frame(self, f)
    }

    fn model_stats(&mut self) -> Option<ModelStats> {
        Some(self.stats())
    }

    fn on_end(&mut self, outcome: Option<uranium_env::reward::Outcome>, final_frame: Option<&RawFrame>) {
        ModelPolicy::on_end(self, outcome, final_frame)
    }

    fn on_abort(&mut self) {
        ModelPolicy::on_abort(self)
    }

    fn act_log_extra(&mut self) -> Option<model_policy::LineExtra> {
        ModelPolicy::act_log_extra(self)
    }

    fn branch_record(&mut self) -> Option<model_policy::BranchSessionPart> {
        ModelPolicy::branch_record(self)
    }
}

/// 051-R2 (020-R22 ①): the session record's `act_log` of a plan game with `act_log = true`.
#[derive(Serialize, Debug, Clone, Default, PartialEq)]
pub struct ActLogStatus {
    /// The file this session wrote (`<acts>/g<plan game>-p<players index>.jsonl`).
    pub path: String,
    /// The end line went out (`GameEnd`) and the file was synced: the log is complete (051-R2).
    pub complete: bool,
    /// Why the log stopped (open or write failure); the game itself went on.
    pub error: Option<String>,
}

/// 051-R2: one session's act log. A file that cannot be opened or written is reported once and
/// left without its end line (incomplete, never used for a branch); the game is not affected.
struct ActLogFile {
    path: PathBuf,
    w: Option<uranium_env::branch::ActLogWriter<BufWriter<std::fs::File>>>,
    error: Option<String>,
    complete: bool,
}

impl ActLogFile {
    /// Creates (truncates: a game played again overwrites its log, 020-R22 ①) the file.
    fn create(path: PathBuf) -> Self {
        let open = || -> std::io::Result<std::fs::File> {
            if let Some(d) = path.parent() {
                std::fs::create_dir_all(d)?;
            }
            std::fs::File::create(&path)
        };
        match open() {
            Ok(f) => Self { path, w: Some(uranium_env::branch::ActLogWriter::new(BufWriter::new(f))), error: None, complete: false },
            Err(e) => {
                let mut s = Self { path, w: None, error: None, complete: false };
                s.stop(format!("open: {e}"));
                s
            }
        }
    }

    fn stop(&mut self, why: String) {
        elog!("[agent] WARNING: act log {} stopped (051-R2): {why}", self.path.display());
        self.w = None;
        self.error.get_or_insert(why);
    }

    fn push(&mut self, line: &uranium_env::branch::ObsLine) {
        let Some(w) = self.w.as_mut() else { return };
        if let Err(e) = w.push(line) {
            self.stop(format!("{e:#}"));
        }
    }

    /// `GameEnd`: the end line, then flush and fsync (the file may be on the shared drive, 061).
    fn finish(&mut self, end: uranium_env::branch::ActLogEnd) {
        let Some(w) = self.w.take() else { return };
        let res = w.finish(end).and_then(|w| w.into_inner().map_err(|e| anyhow::anyhow!("{}", e.error()))).and_then(|f| f.sync_all().map_err(Into::into));
        match res {
            Ok(()) => self.complete = true,
            Err(e) => self.stop(format!("{e:#}")),
        }
    }

    /// Mid-game failure or disconnect: the lines so far are flushed without the end line.
    fn abort(&mut self) {
        if let Some(w) = self.w.take() {
            drop(w);
        }
    }
}

/// 051-R2 end line `git`: the commit this agent was built from, fixed at build time (`build.rs`:
/// `git rev-parse HEAD`, rebuilt when HEAD moves), `unknown` when the build had no git checkout.
pub fn git_commit() -> &'static str {
    env!("URANIUM_GIT")
}

/// One JSONL record (one player of one game).
#[derive(Serialize)]
#[cfg_attr(test, derive(Default))]
pub struct SessionRecord {
    pub game_id: u64,
    pub slot: u8,
    pub map: String,
    pub map_hash: String,
    pub self_race: String,
    /// Race the enemy actually played: the lobby race, or for a lobby-Random enemy (030-R11) the
    /// race of its first unit we saw ("Unknown" if never seen). For per-race statistics.
    pub enemy_race: String,
    /// Race as `GameInfo` gave it at game start ("Unknown" for a lobby-Random enemy, 030-R11).
    pub enemy_race_lobby: String,
    /// T0 build this player was asked to play (050 E-Regression scenario bots).
    pub t0_build: String,
    pub winner: Option<u8>,
    pub won: Option<bool>,
    pub reason: String,
    pub end_frame: u32,
    pub wall_s: f64,
    pub policy_ms: f64,
    pub stats: Option<BridgeStats>,
    /// M1 diagnosis: mineral fields visible in the first observation.
    pub first_frame_minerals: usize,
    pub first_frame_units: usize,
    pub commands: u64,
    pub failed_commands: u64,
    /// 031-R10 (031 "유닛별 거절 기록"): `UnitRejected` events of the session (010-R14). Named apart
    /// from the executor refusals of `ModelStats` (031-R4).
    pub unit_rejections: u64,
    /// `"<command>/<unit_type>/<error>"` (decimal BWAPI ids, `none` for a unit not found) -> count.
    pub unit_rejections_by: BTreeMap<String, u64>,
    /// FNV-1a over every observation's encoded bytes: identical runs give identical digests
    /// (011-AC4); privileged mode on/off must not change it (010-AC6).
    pub frame_digest: String,
    pub privileged_msgs: u64,
    /// 010-AC6: enemy units in an observation that our player could not see (must be 0).
    pub visibility_violations: u64,
    /// 010-R12: (frame, units, events) of `GameEnd.final_frame`, `None` if the bridge sent none.
    pub final_frame: Option<(u32, usize, usize)>,
    /// 020-R5 policy version (`t0@<ver>` or `model:<checkpoint id>:<sample>`).
    pub policy: String,
    pub model: Option<ModelStats>,
    pub error: Option<String>,
    /// 050-R8: T0 Terran session metrics (`model` keys) and the 021-R8 teacher record; `None` for
    /// model sessions and non-Terran T0 bots.
    pub scripted: Option<ScriptedStats>,
    /// 051 (020-R22): the three keys below appear only in plan games with 051 keys, so other
    /// records are unchanged (020-R22 ④). 051-R2: this session's act log.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub act_log: Option<ActLogStatus>,
    /// 051-R3: state files a trunk's `players[0]` model session saved (one per decision point).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub snapshots: Vec<uranium_env::branch::SnapshotTaken>,
    /// 051-R9: a branch player's outcome (`players[0]` model session of a branch game).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub branch: Option<uranium_env::branch::BranchOutcome>,
    /// 051-R8 / R12: a model session's inference server seed (set at the server's start), in plan
    /// games with 051 keys only (020-R22 ④); `None` when the server recorded none.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub infer_seed: Option<u64>,
}

/// BWAPI neutral player id.
const NEUTRAL_PLAYER: u8 = 11;

/// 030-R11: race of the first non-neutral, non-own unit in the frame ("Terran" / "Protoss" /
/// "Zerg"), for the session record of a lobby-Random enemy. Assumes 1v1 (any such unit is the
/// enemy).
fn revealed_enemy_race(f: &RawFrame, me: u8) -> Option<String> {
    let types = uranium_env::types::TypeTables::get();
    f.units
        .iter()
        .filter(|u| u.owner != me && u.owner != NEUTRAL_PLAYER)
        .filter_map(|u| types.unit(u.unit_type))
        .find(|t| matches!(t.race.as_str(), "Terran" | "Protoss" | "Zerg"))
        .map(|t| t.race.clone())
}

/// Enemy units that are not on (or next to) a currently visible tile, or that are cloaked or
/// burrowed without being detected. The bridge must never send these (010-R5, 010-AC6).
fn visibility_violations(f: &RawFrame, me: u8) -> u64 {
    f.units
        .iter()
        .filter(|u| u.owner != me && u.owner != NEUTRAL_PLAYER)
        .filter(|u| {
            // BWAPI counts a unit visible when any tile of its footprint is visible; check the
            // footprint plus a 1-tile margin (sight is computed per 8 px walk tile).
            let (w, h) = uranium_scripted::types::tile_size(u.unit_type);
            let (x0, y0) = ((u.pos.0 as i32 - w as i32 * 16) / 32 - 1, (u.pos.1 as i32 - h as i32 * 16) / 32 - 1);
            let seen = (y0..=y0 + h as i32 + 1).any(|ty| (x0..=x0 + w as i32 + 1).any(|tx| tx >= 0 && ty >= 0 && f.visibility.get(tx as u16, ty as u16)));
            let hidden = (u.has(flags::CLOAKED) || u.has(flags::BURROWED)) && !u.has(flags::DETECTED);
            if (!seen || hidden) && std::env::var("URANIUM_AGENT_DEBUG_VIS").is_ok() {
                elog!("[vis] frame {} slot {me} unit {} type {} owner {} pos {:?} flags {:#x} seen {seen} hidden {hidden}", f.frame, u.id, u.unit_type, u.owner, u.pos, u.flags);
            }
            !seen || hidden
        })
        .count() as u64
}

/// The session digest's FNV-1a, shared with the 051-R2 observation digest (one implementation).
pub use uranium_env::branch::fnv1a;

/// Loads host settings from `.env` (or `$URANIUM_ENV_FILE`) before anything reads the
/// environment; variables already set in the process win (spec 001-R9).
pub fn load_env() {
    let res = match std::env::var_os("URANIUM_ENV_FILE") {
        Some(path) => dotenvy::from_path(path),
        None => dotenvy::dotenv().map(|_| ()),
    };
    if let Err(e) = res {
        if !e.not_found() {
            elog!("warning: failed to load .env: {e}");
        }
    }
}

#[derive(Clone)]
pub struct ServeOptions {
    /// Policy for sessions without a route (020-R11).
    pub default: PolicySpec,
    pub routes: Routes,
    /// Artificial delay before every Act (011-AC7 lag test).
    pub delay_ms: u64,
    /// Return after this many finished sessions (0 = run forever).
    pub max_sessions: usize,
    /// Scenario builds by game and slot (empty = every T0 player plays the standard build).
    pub t0_builds: T0Builds,
    /// Every session's GameEnd report (020-R13).
    pub ends: EndBoard,
    /// 031-R10: per-unit rejection details (031 "유닛별 거절 기록"); disabled by default.
    pub unit_rejections: RejectionLog,
    /// 042-R14: RL league games that judge collapse concede (empty = none).
    pub concede: ConcedeBoard,
    /// 020-R22 / 051: per-game settings of plan games with 051 keys by hostd game id (act logs,
    /// snapshots, branch players); games not listed play as before (020-R22 ④).
    pub branches: BranchBoard,
}

impl ServeOptions {
    pub fn new(default: PolicySpec) -> Self {
        Self { default, routes: Routes::default(), delay_ms: 0, max_sessions: 0, t0_builds: T0Builds::default(), ends: EndBoard::default(), unit_rejections: RejectionLog::default(), concede: ConcedeBoard::default(), branches: BranchBoard::default() }
    }
}

/// 031-R10: one line of the per-unit rejection file (031 "유닛별 거절 기록").
#[derive(Serialize)]
struct UnitRejectionLine<'a> {
    game_id: u64,
    slot: u8,
    /// `RawFrame.frame` of the observation that carried the event.
    frame: u32,
    cmd_index: u16,
    /// `CmdOrder` of `Command::Unit` `cmd_index` of the previous `Act` (null if there is none).
    order: Option<&'a CmdOrder>,
    unit: UnitId,
    unit_type: Option<u16>,
    command: u8,
    error: u16,
}

/// 031-R10: the per-unit rejection file shared by every session of a server (`unit_rejections.jsonl`
/// beside the session records; an eval job's `results/unit_rejections/<host>.jsonl`). Opened for
/// append on the first rejection, so a server that never sees one (SC:R, 012-R14) writes no file.
/// Each session's lines are appended together when it ends. This type never removes lines: a plan
/// run clears its `unit_rejections.jsonl` when it starts fresh, RL and eval jobs keep appending.
#[derive(Clone, Default)]
pub struct RejectionLog(Option<Arc<(PathBuf, Mutex<Option<RecordSink>>)>>);

impl RejectionLog {
    pub fn new(path: PathBuf) -> Self {
        Self(Some(Arc::new((path, Mutex::new(None)))))
    }

    /// `unit_rejections.jsonl` in the folder of the session record file `records`.
    pub fn beside(records: &Path) -> Self {
        Self::new(records.with_file_name("unit_rejections.jsonl"))
    }

    fn append(&self, lines: &[String]) -> Result<()> {
        let Some(inner) = &self.0 else { return Ok(()) };
        if lines.is_empty() {
            return Ok(());
        }
        let mut slot = inner.1.lock().unwrap();
        if slot.is_none() {
            *slot = Some(open_sink(&inner.0)?);
        }
        let sink = slot.as_ref().expect("opened above");
        let mut w = sink.lock().unwrap();
        for l in lines {
            writeln!(w, "{l}")?;
        }
        w.flush()?;
        Ok(())
    }
}

/// 031-R10: counts the `UnitRejected` events of `f` into `rec` and adds one detail line each,
/// resolving `cmd_index` against the previous Act's commands `last_cmds`.
fn record_unit_rejections(f: &RawFrame, last_cmds: &[Command], hello: &Hello, rec: &mut SessionRecord, lines: &mut Vec<String>) {
    for e in &f.events {
        let GameEvent::UnitRejected { cmd_index, unit, unit_type, command, error } = *e else { continue };
        rec.unit_rejections += 1;
        let ty = unit_type.map_or_else(|| "none".to_string(), |t| t.to_string());
        *rec.unit_rejections_by.entry(format!("{command}/{ty}/{error}")).or_default() += 1;
        let order = match last_cmds.get(cmd_index as usize) {
            Some(Command::Unit { order, .. }) => Some(order),
            _ => None,
        };
        let line = UnitRejectionLine { game_id: hello.game_id, slot: hello.slot, frame: f.frame, cmd_index, order, unit, unit_type, command, error };
        lines.push(serde_json::to_string(&line).expect("rejection line serializes"));
    }
}

/// JSONL sink shared by all sessions.
pub type RecordSink = Arc<Mutex<dyn Write + Send>>;

pub fn open_sink(out: &Path) -> Result<RecordSink> {
    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let file = OpenOptions::new().create(true).append(true).open(out).with_context(|| format!("opening {}", out.display()))?;
    Ok(Arc::new(Mutex::new(BufWriter::new(SyncFile(file)))))
}

/// A file whose flush also reaches the disk, or the server of a shared drive (020-R15: session
/// records are read by other PCs; a Linux SMB client otherwise keeps appended lines in its cache
/// and readers see zeros).
struct SyncFile(std::fs::File);

impl Write for SyncFile {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.0.flush()?;
        self.0.sync_data()
    }
}

pub fn serve_addr(listen: &str, opts: ServeOptions, out: PathBuf) -> Result<()> {
    let listener = TcpListener::bind(listen).with_context(|| format!("binding {listen}"))?;
    elog!("[agent] listening on {listen}");
    serve(listener, opts, open_sink(&out)?)
}

/// Accepts bridge sessions until `max_sessions` are done (or forever).
pub fn serve(listener: TcpListener, opts: ServeOptions, sink: RecordSink) -> Result<()> {
    let max_sessions = opts.max_sessions;
    let finished = Arc::new(Mutex::new(0usize));
    let mut handles = Vec::new();
    for conn in listener.incoming() {
        let stream = conn?;
        stream.set_nodelay(true).ok();
        let (sink, done, opts) = (sink.clone(), finished.clone(), opts.clone());
        handles.push(std::thread::spawn(move || {
            let rec = session(stream, &opts);
            let line = serde_json::to_string(&rec).expect("record serializes");
            let mut s = sink.lock().unwrap();
            writeln!(s, "{line}").ok();
            s.flush().ok();
            *done.lock().unwrap() += 1;
        }));
        handles.retain(|h| !h.is_finished());
        if max_sessions > 0 && *finished.lock().unwrap() >= max_sessions {
            break;
        }
        // Accept loop blocks; finished-count is re-checked on the next connection. The
        // selfmatch harness stops the server explicitly instead of relying on this.
    }
    for h in handles {
        h.join().ok();
    }
    Ok(())
}

pub fn session(stream: TcpStream, opts: &ServeOptions) -> SessionRecord {
    let started = Instant::now();
    let mut rec = SessionRecord {
        game_id: 0,
        slot: 0,
        map: String::new(),
        map_hash: String::new(),
        self_race: String::new(),
        enemy_race: String::new(),
        enemy_race_lobby: String::new(),
        t0_build: String::new(),
        winner: None,
        won: None,
        reason: "Disconnected".into(),
        end_frame: 0,
        wall_s: 0.0,
        policy_ms: 0.0,
        stats: None,
        first_frame_minerals: 0,
        first_frame_units: 0,
        commands: 0,
        failed_commands: 0,
        unit_rejections: 0,
        unit_rejections_by: BTreeMap::new(),
        frame_digest: String::new(),
        privileged_msgs: 0,
        visibility_violations: 0,
        final_frame: None,
        policy: String::new(),
        model: None,
        error: None,
        scripted: None,
        act_log: None,
        snapshots: Vec::new(),
        branch: None,
        infer_seed: None,
    };
    let mut rejections = Vec::new();
    if let Err(e) = run_session(stream, opts, &mut rec, &mut rejections) {
        elog!("[agent] game {} slot {}: {e:#}", rec.game_id, rec.slot);
        rec.error = Some(format!("{e:#}"));
    }
    rec.wall_s = started.elapsed().as_secs_f64();
    // 031-R10: diagnostics only; a write failure is logged, the session record still goes out.
    if let Err(e) = opts.unit_rejections.append(&rejections) {
        elog!("[agent] game {} slot {}: writing unit rejections: {e:#}", rec.game_id, rec.slot);
    }
    rec
}

fn run_session(stream: TcpStream, opts: &ServeOptions, rec: &mut SessionRecord, rejections: &mut Vec<String>) -> Result<()> {
    let delay_ms = opts.delay_ms;
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut writer = BufWriter::new(stream);
    let BridgeMsg::Hello(hello) = read_msg(&mut reader)? else { bail!("expected Hello") };
    rec.game_id = hello.game_id;
    rec.slot = hello.slot;
    if let Err(why) = check_hello(&hello) {
        write_msg(&mut writer, &AgentMsg::Reject(why.clone()))?;
        bail!(why);
    }
    opts.ends.hello(hello.game_id);
    let BridgeMsg::GameInfo(info) = read_msg(&mut reader)? else { bail!("expected GameInfo") };
    rec.map = info.map_name.clone();
    rec.map_hash = info.map_hash.iter().map(|b| format!("{b:02x}")).collect();
    rec.self_race = info.players.iter().find(|p| p.is_self).map(|p| format!("{:?}", p.race)).unwrap_or_default();
    rec.enemy_race = info.players.iter().find(|p| p.is_enemy).map(|p| format!("{:?}", p.race)).unwrap_or_default();
    // 030-R11: a lobby-Random enemy arrives as Unknown; `play` fills in the race once seen.
    rec.enemy_race_lobby = rec.enemy_race.clone();
    elog!("[agent] game {} slot {} on {:?} ({} start locations)", hello.game_id, hello.slot, info.map_name, info.static_map.start_locations.len());
    // 020-R11 / 050: policies and T0 builds are registered by `GameSpec.players` index; the BWAPI
    // player id (`Hello.slot`) need not follow launch order, so it comes from the player name.
    let index = spec_index(&info, hello.slot);
    let spec = opts.routes.take(hello.game_id, index).unwrap_or_else(|| opts.default.clone());
    rec.policy = spec.label();
    let build = opts.t0_builds.lock().unwrap().get(&(hello.game_id, index)).copied().unwrap_or_default();
    rec.t0_build = build.name().to_string();
    // 042-R14: only RL league games are on the board (never realtime SC:R games).
    let concede = opts.concede.get(hello.game_id).filter(|_| hello.mode == Mode::Lockstep).map(|g| (g, index));
    // 020-R22 / 051: the per-game settings of a plan game with 051 keys (lockstep only, 001-R5);
    // games not on the board play exactly as before (020-R22 ④).
    let game_branch = if hello.mode == Mode::Lockstep { opts.branches.lock().unwrap().get(&hello.game_id).cloned() } else { None };
    let mut policy = make_policy(&spec, hello.game_id, &info, build, hello.mode == Mode::Realtime, concede, game_branch.clone().map(|g| (g, index)))?;
    let mut log = SessionLog { act_log: game_branch.as_ref().and_then(|g| g.act_log_path(index)).map(ActLogFile::create), index, end: None };
    let res = play(&mut reader, &mut writer, &mut *policy, &hello, rec, rejections, delay_ms, &opts.ends, &mut log);
    if res.is_err() {
        // Disconnected or failed mid-game: not a finished episode (042-R3), no complete 041-R28 log
        // and no complete 051-R2 act log.
        policy.on_abort();
        if let Some(l) = log.act_log.as_mut() {
            l.abort();
        }
    }
    if let Some(g) = &game_branch {
        log.record(g, &mut *policy, hello.slot, rec);
        // 051-R8: the model session's inference server seed (`points.jsonl` / `branches.jsonl`).
        if let PolicySpec::Model(h) = &spec {
            rec.infer_seed = h.infer_seed;
        }
    }
    res
}

/// 051 bookkeeping of one session in a plan game with 051 keys.
struct SessionLog {
    /// 051-R2: `None` = the plan has no `act_log` (or the game is not on the board).
    act_log: Option<ActLogFile>,
    /// This session's `GameSpec.players` index.
    index: u8,
    /// `GameEnd` (winner, frame, reason) once it came.
    end: Option<(Option<u8>, u32, EndReason)>,
}

impl SessionLog {
    /// 051-R2 end line on `GameEnd` (then fsync).
    fn finish(&mut self, winner: Option<u8>, frame: u32, reason: EndReason, slot: u8, policy: &str) {
        self.end = Some((winner, frame, reason));
        if let Some(l) = self.act_log.as_mut() {
            l.finish(uranium_env::branch::ActLogEnd {
                obs: 0,
                decisions: 0,
                end_frame: frame,
                reason,
                player: slot,
                players_index: self.index,
                policy: policy.to_string(),
                git: git_commit().to_string(),
                feature_schema: uranium_env::features::FEATURE_SCHEMA_VERSION,
                action_schema: uranium_env::actions::ACTION_SCHEMA_VERSION,
            });
        }
    }

    /// The session record's 051 keys (020-R22): `act_log`, `snapshots`, `branch` (with its
    /// 051-R9 ④ `end` when the game ended).
    fn record(&self, game: &GameBranch, policy: &mut dyn Policy, slot: u8, rec: &mut SessionRecord) {
        rec.act_log = self.act_log.as_ref().map(|l| ActLogStatus { path: l.path.display().to_string(), complete: l.complete, error: l.error.clone() });
        let Some(part) = policy.branch_record() else { return };
        rec.snapshots = part.snapshots;
        rec.branch = part.outcome.map(|mut o| {
            // 051-R9 ④ of a finished game only: an aborted one (no `GameEnd`, or Aborted /
            // AgentTimeout) keeps `end: null` = no outcome, and the job plays it again.
            if let (Some((winner, frame, reason)), Some(b)) = (self.end, &game.branch)
                && !matches!(reason, EndReason::Aborted | EndReason::AgentTimeout)
            {
                o.end = Some(uranium_env::branch::BranchEnd::new(winner, slot, frame, reason, b.horizon));
            }
            if let Err(e) = uranium_env::branch::FiniteRecord::check_finite(&o) {
                elog!("[agent] WARNING: 051-R9 branch outcome of game {} has a value JSON cannot hold: {e:#}", rec.game_id);
            }
            o
        });
    }
}

#[allow(clippy::too_many_arguments)]
fn play(reader: &mut BufReader<TcpStream>, writer: &mut BufWriter<TcpStream>, policy: &mut dyn Policy, hello: &Hello, rec: &mut SessionRecord, rejections: &mut Vec<String>, delay_ms: u64, ends: &EndBoard, log: &mut SessionLog) -> Result<()> {
    let (mut reader, mut writer) = (reader, writer);
    let realtime = hello.mode == Mode::Realtime;
    let mut policy_time = Duration::ZERO;
    let mut first = true;
    // 011-AC4 session digest: one FNV-1a over every observation's framed bytes, from the same
    // offset basis and bytes as the 051-R2 observation digest.
    let mut digest: u64 = uranium_env::branch::FNV_OFFSET;
    let mut obs_index: u32 = 0;
    let mut last_cmds: Vec<Command> = Vec::new();
    loop {
        match read_msg::<BridgeMsg>(&mut reader)? {
            BridgeMsg::Privileged(p) => {
                rec.privileged_msgs += 1;
                // 010-R10: realtime bridges must never send it; count it (012-AC6) but keep it away
                // from the policy.
                if realtime {
                    elog!("[agent] game {} slot {}: PrivilegedSummary in a realtime session ignored (010-R10)", hello.game_id, hello.slot);
                } else {
                    policy.on_privileged(&p)
                }
            }
            BridgeMsg::Frame(f) => {
                let bytes = uranium_env::branch::frame_bytes(&f)?;
                digest = fnv1a(digest, &bytes);
                let i = obs_index;
                obs_index += 1;
                // Debug aid: URANIUM_AGENT_DUMP=<dir> writes observations before URANIUM_AGENT_DUMP_MAX_FRAME (default 240).
                if let Ok(dir) = std::env::var("URANIUM_AGENT_DUMP") {
                    let max: u32 = std::env::var("URANIUM_AGENT_DUMP_MAX_FRAME").ok().and_then(|v| v.parse().ok()).unwrap_or(240);
                    if f.frame < max {
                        let path = format!("{dir}/g{}-s{}.txt", hello.game_id, hello.slot);
                        if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
                            writeln!(file, "{f:#?}").ok();
                        }
                    }
                }
                if first {
                    rec.first_frame_units = f.units.len();
                    rec.first_frame_minerals = f.units.iter().filter(|u| uranium_scripted::types::is_mineral(u.unit_type)).count();
                    first = false;
                }
                if matches!(rec.enemy_race.as_str(), "Unknown" | "Random") {
                    if let Some(r) = revealed_enemy_race(&f, hello.slot) {
                        rec.enemy_race = r;
                    }
                }
                rec.visibility_violations += visibility_violations(&f, hello.slot);
                rec.failed_commands += f.events.iter().filter(|e| matches!(e, GameEvent::CommandFailed { .. })).count() as u64;
                // 031-R10: per-unit rejections go to the session record only (not to features).
                record_unit_rejections(&f, &last_cmds, hello, rec, rejections);
                // Debug aid: with URANIUM_AGENT_DUMP set, failed commands go to g<id>-s<slot>-failed.txt at any frame.
                if let Ok(dir) = std::env::var("URANIUM_AGENT_DUMP") {
                    for e in &f.events {
                        if let GameEvent::CommandFailed { cmd_index, error } = e {
                            let path = format!("{dir}/g{}-s{}-failed.txt", hello.game_id, hello.slot);
                            if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
                                writeln!(file, "{} error {error} {:?}", f.frame, last_cmds.get(*cmd_index as usize)).ok();
                            }
                        }
                    }
                }
                let t = Instant::now();
                let mut act = match policy.on_frame(&f) {
                    Ok(a) => a,
                    Err(e) => {
                        rec.model = policy.model_stats();
                        return Err(e);
                    }
                };
                policy_time += t.elapsed();
                rec.commands += act.cmds.len() as u64;
                last_cmds.clone_from(&act.cmds);
                if delay_ms > 0 {
                    std::thread::sleep(Duration::from_millis(delay_ms));
                }
                if realtime {
                    // 012-R7 / 010-R7: every policy's Act expires REALTIME_ACT_FRAMES after the frame it saw.
                    act.valid_until_frame = Some(f.frame + REALTIME_ACT_FRAMES);
                }
                // 051-R2: one line per observation with the Act exactly as sent.
                if let Some(l) = log.act_log.as_mut() {
                    let x = policy.act_log_extra().unwrap_or_default();
                    l.push(&uranium_env::branch::ObsLine {
                        i,
                        frame: f.frame,
                        digest: uranium_env::branch::obs_digest_bytes(&bytes),
                        act: act.clone(),
                        decision: x.decision,
                        state: x.state,
                        state_sha256: x.state_sha256,
                        forced: x.forced,
                    });
                }
                write_msg(&mut writer, &AgentMsg::Act(act))?;
            }
            BridgeMsg::End(end) => {
                let session_end = SessionEnd { slot: hello.slot, winner: end.winner, frame: end.frame, reason: end.reason };
                let outcome = end_outcome(&end, hello.slot);
                rec.winner = end.winner;
                rec.won = end.winner.map(|w| w == hello.slot);
                rec.reason = format!("{:?}", end.reason);
                rec.end_frame = end.frame;
                rec.stats = end.stats;
                rec.policy_ms = policy_time.as_secs_f64() * 1e3;
                rec.frame_digest = format!("{digest:016x}");
                ends.push(hello.game_id, session_end);
                if let Some(last) = &end.final_frame {
                    // 010-AC6 also holds for the final frame (010-R12 keeps 010-R5).
                    rec.visibility_violations += visibility_violations(last, hello.slot);
                    rec.final_frame = Some((last.frame, last.units.len(), last.events.len()));
                    // 010-R12: the final frame carries the events of the last Act too.
                    record_unit_rejections(last, &last_cmds, hello, rec, rejections);
                }
                policy.on_end(outcome, end.final_frame.as_deref());
                // After `on_end`: the 042 metrics include the final frame (AMB-042-16).
                rec.model = policy.model_stats();
                rec.scripted = policy.scripted_stats();
                log.finish(end.winner, end.frame, end.reason, hello.slot, &rec.policy);
                return Ok(());
            }
            other => bail!("unexpected message {other:?}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn end(slot: u8, winner: Option<u8>, frame: u32) -> SessionEnd {
        SessionEnd { slot, winner, frame, reason: EndReason::Elimination }
    }

    fn t0_info(race: Race) -> GameInfo {
        let n = 64usize * 64;
        GameInfo {
            map_name: "Python".into(),
            map_hash: [0; 32],
            map_size_tiles: (64, 64),
            self_player: 0,
            players: vec![
                PlayerInfo { id: 0, name: "p0-T0".into(), race, start_tile: Some((4, 4)), is_self: true, is_enemy: false },
                PlayerInfo { id: 1, name: "p1-x".into(), race: Race::Zerg, start_tile: None, is_self: false, is_enemy: true },
            ],
            static_map: StaticMap {
                width_tiles: 64,
                height_tiles: 64,
                walkable: BitGrid::from_bytes(256, 256, &vec![1; n * 16]),
                buildable: BitGrid::from_bytes(64, 64, &vec![1; n]),
                ground_height: vec![0; n],
                start_locations: vec![(4, 4), (56, 56)],
            },
            latency_frames: 0,
        }
    }

    fn scv_frame(frame: u32, scvs: u32) -> RawFrame {
        let units = (0..scvs).map(|i| UnitState { id: 10 + i, owner: 0, unit_type: 7, pos: (200, 200), flags: flags::COMPLETED, ..Default::default() }).collect();
        RawFrame { frame, self_state: PlayerState::default(), units, visibility: BitGrid::new(64, 64), explored: BitGrid::new(64, 64), creep: BitGrid::new(64, 64), events: vec![], effects: vec![] }
    }

    /// 050-R8: a T0 Terran session folds every observation and the final frame into the 050-R7
    /// metrics (`scripted`, the model row's keys in the same order); the teacher adds `react`;
    /// a non-Terran T0 bot has no record. Off a target map the teacher plays without a precompute.
    #[test]
    fn scripted_row_050_r8() {
        let mut p = ScriptedPolicy::new(&t0_info(Race::Terran), T0Build::Standard).unwrap();
        p.on_frame(&scv_frame(1_440, 6)).unwrap();
        p.on_frame(&scv_frame(5_000, 3)).unwrap();
        p.on_end(None, Some(&scv_frame(11_520, 9)));
        let s = p.scripted_stats().unwrap();
        assert_eq!((s.scv_min_1m, s.scv_4m, s.scv_8m, s.react.clone()), (Some(3), Some(9), Some(9), None));
        let text = serde_json::to_string(&s).unwrap();
        // 052-R6: `ops` and `rush_combat` follow the existing keys, whose order is unchanged.
        assert!(text.starts_with(r#"{"scout_main_s":null,"scout_types_4m":0,"scout_types_6m":0,"barracks_s":null,"bunker_s":null,"scv_4m":9,"scv_min_1m":3,"army_s":null,"scv_8m":9,"react":null,"ops":{"end_s":480.0,"mined":0,"#), "{text}");
        // 022-R10: `tp` after them.
        assert!(text.ends_with(r#""float_gas_rs":0.0},"rush_combat":null,"tp":null}"#), "{text}");
        assert_eq!(s.ops.end_s, 480.0, "the final frame counts (052-R6, 010-R12)");

        let mut t = ScriptedPolicy::new(&t0_info(Race::Terran), T0Build::TerranReact).unwrap();
        t.on_frame(&scv_frame(100, 4)).unwrap();
        let r = t.scripted_stats().unwrap().react.expect("teacher record");
        assert_eq!((r.rush_s, r.defences, r.branch), (None, 0, uranium_scripted::Branch::Macro));
        let text = serde_json::to_string(&r).unwrap();
        assert_eq!(text, r#"{"main_s":null,"rush_s":null,"rush_rule":null,"defences":0,"macro_s":null,"expand_s":null,"branch":"macro","pool_start_s":null,"pool_seen_s":null,"lings_max":0,"contact_s":null,"bunker_s":null,"rules_s":{}}"#);

        let mut z = ScriptedPolicy::new(&t0_info(Race::Zerg), T0Build::ZergRush).unwrap();
        z.on_frame(&scv_frame(100, 0)).unwrap();
        assert!(z.scripted_stats().is_none());

        // 022-R2 / R10: `terran_tp` against a lobby Zerg is the 021-R8 teacher (`react`); against
        // Protoss it keeps the `tp` record.
        let mut tz = ScriptedPolicy::new(&t0_info(Race::Terran), T0Build::TerranTp).unwrap();
        tz.on_frame(&scv_frame(100, 4)).unwrap();
        let s = tz.scripted_stats().unwrap();
        assert!(s.react.is_some() && s.tp.is_none());
        let mut info = t0_info(Race::Terran);
        info.players[1].race = Race::Protoss;
        let mut tp = ScriptedPolicy::new(&info, T0Build::TerranTp).unwrap();
        tp.on_frame(&scv_frame(100, 4)).unwrap();
        let s = tp.scripted_stats().unwrap();
        assert!(s.react.is_none());
        let text = serde_json::to_string(&s.tp.expect("022 record")).unwrap();
        assert_eq!(
            text,
            r#"{"main_s":null,"rush_s":null,"rush_rule":null,"defences":0,"macro_s":null,"refinery_s":null,"factory_s":null,"shop_s":null,"siege_s":null,"tank_s":null,"expand_s":null,"wave_s":null,"branch":"macro","contact_s":null,"bunker_s":null,"rules_s":{}}"#
        );
    }

    /// 051-R2: the act-log `git` is the build-time constant (a commit id or `unknown`).
    #[test]
    fn git_commit_is_the_build_constant_051_r2() {
        let c = git_commit();
        assert_eq!(c, env!("URANIUM_GIT"));
        assert!(c == "unknown" || (c.len() == 40 && c.bytes().all(|b| b.is_ascii_hexdigit())), "{c}");
    }

    /// 020-R11 / 020-R16: the spec index comes from both hostd name forms, nothing from others.
    #[test]
    fn name_index_reads_both_forms_020_r16() {
        assert_eq!(name_index("uranium-p1"), Some(1));
        assert_eq!(name_index("p0-core-abl-gtrxl-pt"), Some(0));
        assert_eq!(name_index("p1-T0-zerg"), Some(1));
        assert_eq!(name_index("p1-"), Some(1));
        assert_eq!(name_index("player"), None);
        assert_eq!(name_index("pX-a"), None);
        assert_eq!(name_index("Iron"), None);
    }

    /// 020-R13: on disagreement the session with the larger end frame decides.
    #[test]
    fn game_result_takes_the_later_session_020_r13() {
        let (r, d) = game_result(&[end(0, Some(1), 9000), end(1, Some(1), 9000)]).unwrap();
        assert_eq!((r.winner, d), (Some(1), false));
        // Loser (slot 0) ended first and marked both as defeated: the winner's later report wins.
        let (r, d) = game_result(&[end(0, None, 8999), end(1, Some(1), 9001)]).unwrap();
        assert_eq!((r.winner, r.slot, d), (Some(1), 1, true));
        let (r, d) = game_result(&[end(1, Some(1), 9001), end(0, None, 8999)]).unwrap();
        assert_eq!((r.winner, d), (Some(1), true));
        // Tie: the first report stands.
        let (r, _) = game_result(&[end(0, Some(0), 500), end(1, Some(1), 500)]).unwrap();
        assert_eq!(r.winner, Some(0));
        assert!(game_result(&[]).is_none());
    }

    /// 020-R13: both sides claiming the win because the other left is a drop: Aborted. A real
    /// leave (the leaver reports the other side as winner) and a normal elimination stand.
    #[test]
    fn mutual_leave_is_aborted_020_r13() {
        let left = |slot, winner, frame| SessionEnd { slot, winner, frame, reason: EndReason::Surrender };
        let (r, d) = game_result(&[left(0, Some(0), 1615), left(1, Some(1), 1615)]).unwrap();
        assert_eq!((r.reason, r.winner, d), (EndReason::Aborted, None, true));
        let (r, _) = game_result(&[end(0, Some(1), 4000), left(1, Some(1), 4010)]).unwrap();
        assert_eq!((r.reason, r.winner), (EndReason::Surrender, Some(1)));
        let (r, _) = game_result(&[end(0, Some(0), 8000), left(1, Some(1), 8031)]).unwrap();
        assert_eq!((r.reason, r.winner), (EndReason::Surrender, Some(1)));
        assert!(!mutual_leave(&[left(0, Some(0), 1615)]));
    }

    /// 031-R10 / 010-R14: `UnitRejected` events are counted per "<command>/<unit_type>/<error>"
    /// (`none` for a unit not found), resolved against the previous Act's `Command::Unit`, and
    /// written as detail lines on session end; a server without rejections writes no file.
    #[test]
    fn unit_rejections_are_counted_and_logged_031_r10() {
        let hello = Hello { protocol_version: PROTOCOL_VERSION, backend: Backend::OpenBw, game_version: "x".into(), mode: Mode::Lockstep, game_id: 12, slot: 1, replay: false };
        let rejected = |cmd_index, unit, unit_type| GameEvent::UnitRejected { cmd_index, unit, unit_type, command: 14, error: 10 };
        let frame = RawFrame {
            frame: 96,
            self_state: PlayerState::default(),
            units: vec![],
            visibility: BitGrid::new(1, 1),
            explored: BitGrid::new(1, 1),
            creep: BitGrid::new(1, 1),
            events: vec![GameEvent::CommandFailed { cmd_index: 1, error: 10 }, rejected(0, 5, Some(0)), rejected(0, 6, Some(0)), rejected(1, 7, None)],
            effects: vec![],
        };
        let last_cmds = vec![
            Command::Unit { units: vec![5, 6], order: CmdOrder::Attack(Target::Unit(99)), queued: false },
            Command::Build { worker: 7, unit_type: 109, tile: (3, 4) },
        ];
        let (mut rec, mut lines) = (SessionRecord::default(), Vec::new());
        record_unit_rejections(&frame, &last_cmds, &hello, &mut rec, &mut lines);
        assert_eq!(rec.unit_rejections, 3);
        assert_eq!(rec.unit_rejections_by, BTreeMap::from([("14/0/10".to_string(), 2), ("14/none/10".to_string(), 1)]));
        // CommandFailed keeps its own count; rejections do not add to it.
        assert_eq!(rec.failed_commands, 0);
        let first: serde_json::Value = serde_json::from_str(&lines[0]).unwrap();
        assert_eq!(first, serde_json::json!({"game_id": 12, "slot": 1, "frame": 96, "cmd_index": 0, "order": {"Attack": {"Unit": 99}}, "unit": 5, "unit_type": 0, "command": 14, "error": 10}));
        let third: serde_json::Value = serde_json::from_str(&lines[2]).unwrap();
        assert_eq!((&third["order"], &third["unit_type"]), (&serde_json::Value::Null, &serde_json::Value::Null));

        let dir = std::env::temp_dir().join(format!("uranium-agent-rejections-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let log = RejectionLog::beside(&dir.join("results.jsonl"));
        log.append(&[]).unwrap();
        assert!(!dir.join("unit_rejections.jsonl").exists());
        log.append(&lines[..2]).unwrap();
        log.clone().append(&lines[2..]).unwrap();
        let text = std::fs::read_to_string(dir.join("unit_rejections.jsonl")).unwrap();
        assert_eq!(text.lines().collect::<Vec<_>>(), lines);
        RejectionLog::default().append(&lines).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }
}
