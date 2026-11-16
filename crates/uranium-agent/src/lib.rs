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

#[derive(Clone)]
pub enum PolicySpec {
    ScriptedT0,
    Model(InferenceHandle),
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
    pub fn label(&self) -> String {
        match self {
            Self::ScriptedT0 => format!("t0@{}", env!("CARGO_PKG_VERSION")),
            Self::Model(h) => infer::model_policy_label(&h.checkpoint_id, h.sample),
            Self::Learner(l) => format!("learner:{}:{:?}", l.handle().checkpoint_id, l.handle().sample).to_lowercase(),
        }
    }
}

#[derive(Clone, Default)]
pub struct Routes(Arc<Mutex<HashMap<(u64, u8), PolicySpec>>>);

impl Routes {
    pub fn insert(&self, game_id: u64, slot: u8, spec: PolicySpec) {
        self.0.lock().unwrap().insert((game_id, slot), spec);
    }

    pub fn take(&self, game_id: u64, slot: u8) -> Option<PolicySpec> {
        self.0.lock().unwrap().remove(&(game_id, slot))
    }
}

pub trait Policy: Send {
    fn on_privileged(&mut self, _p: &PrivilegedSummary) {}
    fn on_frame(&mut self, f: &RawFrame) -> Result<Act>;
    fn model_stats(&mut self) -> Option<ModelStats> {
        None
    }
    fn scripted_stats(&mut self) -> Option<ScriptedStats> {
        None
    }
    fn on_end(&mut self, _outcome: Option<uranium_env::reward::Outcome>, _final_frame: Option<&RawFrame>) {}
    fn on_abort(&mut self) {
        self.on_end(None, None)
    }
    fn act_log_extra(&mut self) -> Option<model_policy::LineExtra> {
        None
    }
    fn branch_record(&mut self) -> Option<model_policy::BranchSessionPart> {
        None
    }
}

pub use uranium_scripted::T0Build;
pub use uranium_env::branch::{BranchBoard, GameBranch};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionEnd {
    pub slot: u8,
    pub winner: Option<u8>,
    pub frame: u32,
    pub reason: EndReason,
}

pub fn game_result(ends: &[SessionEnd]) -> Option<(SessionEnd, bool)> {
    let first = *ends.first()?;
    let chosen = ends.iter().skip(1).fold(first, |best, e| if e.frame > best.frame { *e } else { best });
    let disagree = ends.iter().any(|e| e.winner != first.winner);
    if mutual_leave(ends) {
        return Some((SessionEnd { winner: None, reason: EndReason::Aborted, ..chosen }, disagree));
    }
    Some((chosen, disagree))
}

pub fn mutual_leave(ends: &[SessionEnd]) -> bool {
    ends.len() == 2 && ends.iter().all(|e| e.reason == EndReason::Surrender && e.winner == Some(e.slot))
}

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

    pub fn hello(&self, game_id: u64) {
        *self.0.lock().unwrap().hellos.entry(game_id).or_default() += 1;
    }

    pub fn connected(&self, game_id: u64) -> usize {
        self.0.lock().unwrap().hellos.get(&game_id).copied().unwrap_or(0)
    }

    pub fn take(&self, game_id: u64) -> Vec<SessionEnd> {
        let mut b = self.0.lock().unwrap();
        b.hellos.remove(&game_id);
        b.ends.remove(&game_id).unwrap_or_default()
    }
}

#[derive(Clone, Default)]
pub struct ConcedeBoard(Arc<Mutex<HashMap<u64, Arc<uranium_env::concede::ConcedeGame>>>>);

impl ConcedeBoard {
    pub fn insert(&self, game_id: u64, game: Arc<uranium_env::concede::ConcedeGame>) {
        self.0.lock().unwrap().insert(game_id, game);
    }

    pub fn get(&self, game_id: u64) -> Option<Arc<uranium_env::concede::ConcedeGame>> {
        self.0.lock().unwrap().get(&game_id).cloned()
    }

    pub fn take(&self, game_id: u64) -> Option<Arc<uranium_env::concede::ConcedeGame>> {
        self.0.lock().unwrap().remove(&game_id)
    }
}

pub type T0Builds = Arc<Mutex<HashMap<(u64, u8), T0Build>>>;

pub fn spec_index(info: &GameInfo, slot: u8) -> u8 {
    let name = info.players.iter().find(|p| p.is_self).map(|p| p.name.as_str()).unwrap_or("");
    name_index(name).unwrap_or(slot)
}

pub fn name_index(name: &str) -> Option<u8> {
    if let Some(i) = name.strip_prefix("uranium-p") {
        return i.parse().ok();
    }
    let (i, _) = name.strip_prefix('p')?.split_once('-')?;
    i.parse().ok()
}

pub const REALTIME_ACT_FRAMES: u32 = 24;

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
    pub ops: uranium_env::ops::OpsMetrics,
    pub rush_combat: Option<uranium_env::ops::RushCombat>,
    pub tp: Option<TpRecord>,
}

#[derive(Serialize, Debug, Clone, Default, PartialEq)]
pub struct TpRecord {
    pub main_s: Option<f64>,
    pub rush_s: Option<f64>,
    pub rush_rule: Option<uranium_scripted::TpRule>,
    pub defences: u32,
    pub macro_s: Option<f64>,
    pub refinery_s: Option<f64>,
    pub factory_s: Option<f64>,
    pub shop_s: Option<f64>,
    pub siege_s: Option<f64>,
    pub tank_s: Option<f64>,
    pub expand_s: Option<f64>,
    pub wave_s: Option<f64>,
    pub branch: uranium_scripted::Branch,
    pub contact_s: Option<f64>,
    pub bunker_s: Option<f64>,
    pub rules_s: BTreeMap<uranium_scripted::TpRule, f64>,
}

#[derive(Serialize, Debug, Clone, Default, PartialEq)]
pub struct ReactRecord {
    pub main_s: Option<f64>,
    pub rush_s: Option<f64>,
    pub rush_rule: Option<uranium_scripted::RushRule>,
    pub defences: u32,
    pub macro_s: Option<f64>,
    pub expand_s: Option<f64>,
    pub branch: uranium_scripted::Branch,
    pub pool_start_s: Option<f64>,
    pub pool_seen_s: Option<f64>,
    pub lings_max: u32,
    pub contact_s: Option<f64>,
    pub bunker_s: Option<f64>,
    pub rules_s: BTreeMap<uranium_scripted::RushRule, f64>,
}

pub struct ScriptedPolicy {
    bot: Box<dyn uranium_scripted::Bot>,
    scout: Option<uranium_env::scout::ScoutTracker>,
    ops: Option<uranium_env::ops::OpsTracker>,
}

impl ScriptedPolicy {
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
        if let Some(o) = self.ops.as_mut() {
            o.step(f);
        }
        Ok(self.bot.act(f))
    }

    fn on_end(&mut self, _outcome: Option<uranium_env::reward::Outcome>, final_frame: Option<&RawFrame>) {
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

#[derive(Serialize, Debug, Clone, Default, PartialEq)]
pub struct ActLogStatus {
    pub path: String,
    pub complete: bool,
    pub error: Option<String>,
}

struct ActLogFile {
    path: PathBuf,
    w: Option<uranium_env::branch::ActLogWriter<BufWriter<std::fs::File>>>,
    error: Option<String>,
    complete: bool,
}

impl ActLogFile {
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

    fn finish(&mut self, end: uranium_env::branch::ActLogEnd) {
        let Some(w) = self.w.take() else { return };
        let res = w.finish(end).and_then(|w| w.into_inner().map_err(|e| anyhow::anyhow!("{}", e.error()))).and_then(|f| f.sync_all().map_err(Into::into));
        match res {
            Ok(()) => self.complete = true,
            Err(e) => self.stop(format!("{e:#}")),
        }
    }

    fn abort(&mut self) {
        if let Some(w) = self.w.take() {
            drop(w);
        }
    }
}

pub fn git_commit() -> &'static str {
    env!("URANIUM_GIT")
}

#[derive(Serialize)]
#[cfg_attr(test, derive(Default))]
pub struct SessionRecord {
    pub game_id: u64,
    pub slot: u8,
    pub map: String,
    pub map_hash: String,
    pub self_race: String,
    pub enemy_race: String,
    pub enemy_race_lobby: String,
    pub t0_build: String,
    pub winner: Option<u8>,
    pub won: Option<bool>,
    pub reason: String,
    pub end_frame: u32,
    pub wall_s: f64,
    pub policy_ms: f64,
    pub stats: Option<BridgeStats>,
    pub first_frame_minerals: usize,
    pub first_frame_units: usize,
    pub commands: u64,
    pub failed_commands: u64,
    pub unit_rejections: u64,
    pub unit_rejections_by: BTreeMap<String, u64>,
    pub frame_digest: String,
    pub privileged_msgs: u64,
    pub visibility_violations: u64,
    pub final_frame: Option<(u32, usize, usize)>,
    pub policy: String,
    pub model: Option<ModelStats>,
    pub error: Option<String>,
    pub scripted: Option<ScriptedStats>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub act_log: Option<ActLogStatus>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub snapshots: Vec<uranium_env::branch::SnapshotTaken>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub branch: Option<uranium_env::branch::BranchOutcome>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub infer_seed: Option<u64>,
}

const NEUTRAL_PLAYER: u8 = 11;

fn revealed_enemy_race(f: &RawFrame, me: u8) -> Option<String> {
    let types = uranium_env::types::TypeTables::get();
    f.units
        .iter()
        .filter(|u| u.owner != me && u.owner != NEUTRAL_PLAYER)
        .filter_map(|u| types.unit(u.unit_type))
        .find(|t| matches!(t.race.as_str(), "Terran" | "Protoss" | "Zerg"))
        .map(|t| t.race.clone())
}

fn visibility_violations(f: &RawFrame, me: u8) -> u64 {
    f.units
        .iter()
        .filter(|u| u.owner != me && u.owner != NEUTRAL_PLAYER)
        .filter(|u| {
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

pub use uranium_env::branch::fnv1a;

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
    pub default: PolicySpec,
    pub routes: Routes,
    pub delay_ms: u64,
    pub max_sessions: usize,
    pub t0_builds: T0Builds,
    pub ends: EndBoard,
    pub unit_rejections: RejectionLog,
    pub concede: ConcedeBoard,
    pub branches: BranchBoard,
}

impl ServeOptions {
    pub fn new(default: PolicySpec) -> Self {
        Self { default, routes: Routes::default(), delay_ms: 0, max_sessions: 0, t0_builds: T0Builds::default(), ends: EndBoard::default(), unit_rejections: RejectionLog::default(), concede: ConcedeBoard::default(), branches: BranchBoard::default() }
    }
}

#[derive(Serialize)]
struct UnitRejectionLine<'a> {
    game_id: u64,
    slot: u8,
    frame: u32,
    cmd_index: u16,
    order: Option<&'a CmdOrder>,
    unit: UnitId,
    unit_type: Option<u16>,
    command: u8,
    error: u16,
}

#[derive(Clone, Default)]
pub struct RejectionLog(Option<Arc<(PathBuf, Mutex<Option<RecordSink>>)>>);

impl RejectionLog {
    pub fn new(path: PathBuf) -> Self {
        Self(Some(Arc::new((path, Mutex::new(None)))))
    }

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

pub type RecordSink = Arc<Mutex<dyn Write + Send>>;

pub fn open_sink(out: &Path) -> Result<RecordSink> {
    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let file = OpenOptions::new().create(true).append(true).open(out).with_context(|| format!("opening {}", out.display()))?;
    Ok(Arc::new(Mutex::new(BufWriter::new(SyncFile(file)))))
}

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
    rec.enemy_race_lobby = rec.enemy_race.clone();
    elog!("[agent] game {} slot {} on {:?} ({} start locations)", hello.game_id, hello.slot, info.map_name, info.static_map.start_locations.len());
    let index = spec_index(&info, hello.slot);
    let spec = opts.routes.take(hello.game_id, index).unwrap_or_else(|| opts.default.clone());
    rec.policy = spec.label();
    let build = opts.t0_builds.lock().unwrap().get(&(hello.game_id, index)).copied().unwrap_or_default();
    rec.t0_build = build.name().to_string();
    let concede = opts.concede.get(hello.game_id).filter(|_| hello.mode == Mode::Lockstep).map(|g| (g, index));
    let game_branch = if hello.mode == Mode::Lockstep { opts.branches.lock().unwrap().get(&hello.game_id).cloned() } else { None };
    let mut policy = make_policy(&spec, hello.game_id, &info, build, hello.mode == Mode::Realtime, concede, game_branch.clone().map(|g| (g, index)))?;
    let mut log = SessionLog { act_log: game_branch.as_ref().and_then(|g| g.act_log_path(index)).map(ActLogFile::create), index, end: None };
    let res = play(&mut reader, &mut writer, &mut *policy, &hello, rec, rejections, delay_ms, &opts.ends, &mut log);
    if res.is_err() {
        policy.on_abort();
        if let Some(l) = log.act_log.as_mut() {
            l.abort();
        }
    }
    if let Some(g) = &game_branch {
        log.record(g, &mut *policy, hello.slot, rec);
        if let PolicySpec::Model(h) = &spec {
            rec.infer_seed = h.infer_seed;
        }
    }
    res
}

struct SessionLog {
    act_log: Option<ActLogFile>,
    index: u8,
    end: Option<(Option<u8>, u32, EndReason)>,
}

impl SessionLog {
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

    fn record(&self, game: &GameBranch, policy: &mut dyn Policy, slot: u8, rec: &mut SessionRecord) {
        rec.act_log = self.act_log.as_ref().map(|l| ActLogStatus { path: l.path.display().to_string(), complete: l.complete, error: l.error.clone() });
        let Some(part) = policy.branch_record() else { return };
        rec.snapshots = part.snapshots;
        rec.branch = part.outcome.map(|mut o| {
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
    let mut digest: u64 = uranium_env::branch::FNV_OFFSET;
    let mut obs_index: u32 = 0;
    let mut last_cmds: Vec<Command> = Vec::new();
    loop {
        match read_msg::<BridgeMsg>(&mut reader)? {
            BridgeMsg::Privileged(p) => {
                rec.privileged_msgs += 1;
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
                record_unit_rejections(&f, &last_cmds, hello, rec, rejections);
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
                    act.valid_until_frame = Some(f.frame + REALTIME_ACT_FRAMES);
                }
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
                    rec.visibility_violations += visibility_violations(last, hello.slot);
                    rec.final_frame = Some((last.frame, last.units.len(), last.events.len()));
                    record_unit_rejections(last, &last_cmds, hello, rec, rejections);
                }
                policy.on_end(outcome, end.final_frame.as_deref());
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

    #[test]
    fn scripted_row_050_r8() {
        let mut p = ScriptedPolicy::new(&t0_info(Race::Terran), T0Build::Standard).unwrap();
        p.on_frame(&scv_frame(1_440, 6)).unwrap();
        p.on_frame(&scv_frame(5_000, 3)).unwrap();
        p.on_end(None, Some(&scv_frame(11_520, 9)));
        let s = p.scripted_stats().unwrap();
        assert_eq!((s.scv_min_1m, s.scv_4m, s.scv_8m, s.react.clone()), (Some(3), Some(9), Some(9), None));
        let text = serde_json::to_string(&s).unwrap();
        assert!(text.starts_with(r#"{"scout_main_s":null,"scout_types_4m":0,"scout_types_6m":0,"barracks_s":null,"bunker_s":null,"scv_4m":9,"scv_min_1m":3,"army_s":null,"scv_8m":9,"react":null,"ops":{"end_s":480.0,"mined":0,"#), "{text}");
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

    #[test]
    fn git_commit_is_the_build_constant_051_r2() {
        let c = git_commit();
        assert_eq!(c, env!("URANIUM_GIT"));
        assert!(c == "unknown" || (c.len() == 40 && c.bytes().all(|b| b.is_ascii_hexdigit())), "{c}");
    }

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

    #[test]
    fn game_result_takes_the_later_session_020_r13() {
        let (r, d) = game_result(&[end(0, Some(1), 9000), end(1, Some(1), 9000)]).unwrap();
        assert_eq!((r.winner, d), (Some(1), false));
        let (r, d) = game_result(&[end(0, None, 8999), end(1, Some(1), 9001)]).unwrap();
        assert_eq!((r.winner, r.slot, d), (Some(1), 1, true));
        let (r, d) = game_result(&[end(1, Some(1), 9001), end(0, None, 8999)]).unwrap();
        assert_eq!((r.winner, d), (Some(1), true));
        let (r, _) = game_result(&[end(0, Some(0), 500), end(1, Some(1), 500)]).unwrap();
        assert_eq!(r.winner, Some(0));
        assert!(game_result(&[]).is_none());
    }

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
