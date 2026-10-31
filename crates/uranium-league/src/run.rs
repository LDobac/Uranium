//! The RL run loop (042): one process runs the learner, the inference servers, the agent server,
//! the game dispatch (`uranium-runner`, 020-R12) and the league. The learner and the servers are
//! the PyTorch side (`python -m uranium rl`, AMB-000-4) behind [`RlLearners`].
//!
//! Game outcomes follow 020-R13: once hostd reports a game over, the GameEnd reports of its
//! sessions are reconciled (`game_result`) and that verdict sets r_win on the learner's terminal
//! segment (033) and feeds the league win table (042). Aborted games are revoked (042-R3).
//! Checkpoints go through `storage::Run` (061): stage → seal → push; a clean stop (042-R10)
//! releases the lock only when the final push succeeded (061-R11-5/6).

use std::collections::{HashMap, VecDeque};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use uranium_agent::infer::InferenceHandle;
use uranium_agent::model_policy::{LearnerHook, SessionObserver};
use uranium_agent::{EndBoard, PolicySpec, Routes, T0Build, T0Builds, game_result};
use uranium_env::features::Strategy;
use uranium_env::reward::{Outcome, RewardWeights};
use uranium_proto::control::{GameSpec, MapKey, PlayerKind, PlayerSpec};
use uranium_proto::{EndReason, GameInfo, Race};
use uranium_runner::{Dispatch, HostEntry, Job, JobOutcome, JobSource};

use crate::ckpt::{self, TrainState};
use crate::config::RlConfig;
use crate::league::{AgentKind, Class, League, Opponent, Pick, T1Bot};
use crate::learner::{Algo, HEAD_NAMES, RlLearners, SnapshotServers, UpdateStats};
use crate::storage::{LockStatus, PushOutcome, Run, StorageConfig};
use crate::trajectory::{GameResult, Segment, SegmentQueue, TrajectoryObserver, trained_steps as trained_steps_of};

/// One entry of the z table (033-R8): a replay's strategy statistic.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ZEntry {
    pub build: Vec<u16>,
    pub units: Vec<f32>,
}

pub struct RlArgs {
    pub run: String,
    pub config: PathBuf,
    pub algo: Algo,
    pub seed: u64,
    pub take_over: bool,
    /// Local log folder (train.csv, games.csv, games.jsonl, sessions).
    pub log_dir: PathBuf,
    /// z table (033-R8); `None` = every game without z.
    pub z_table: Option<PathBuf>,
    /// Stop after this many updates (smoke runs); 0 = until Ctrl+C.
    pub max_updates: u64,
    /// Stop after this many minutes (smoke runs); 0 = no limit.
    pub max_minutes: u64,
    /// 042-R12: clean stop at this many agent steps (run total, across resumes); 0 = no limit.
    pub max_agent_steps: u64,
    /// Checkpoint every this many minutes (061-R4: 30).
    pub checkpoint_minutes: u64,
    /// 042-AC1 small setting: always this T0 race on Fighting Spirit, Main only.
    pub fixed_t0: Option<Race>,
    /// Ctrl+C state (`uranium_store::install_stop_handler`, installed first by the caller).
    pub stop: Arc<AtomicUsize>,
}

/// Per-game bookkeeping between dispatch and the end of the game.
struct GameMeta {
    pick: Pick,
    /// Our learner's BWAPI player id and the enemy race it saw (set at session start).
    self_player: Option<u8>,
    enemy_race: Race,
}

struct Shared {
    league: Mutex<League>,
    games: Mutex<HashMap<u64, GameMeta>>,
    /// Terminal segments held until the 020-R13 verdict sets r_win.
    held: Mutex<HashMap<u64, Segment>>,
    results: Mutex<Receiver<GameResult>>,
    finished: Mutex<HashMap<u64, GameResult>>,
    queues: Vec<Arc<SegmentQueue>>,
    ends: EndBoard,
    next_id: AtomicU64,
    stopping: AtomicBool,
    agent_addr: String,
    handles: Vec<InferenceHandle>,
    snapshots: Mutex<SnapshotPool>,
    games_csv: Mutex<std::fs::File>,
    agent_steps: AtomicU64,
    games_done: AtomicU64,
    disagree: AtomicU64,
    aborted: AtomicU64,
    max_frames: u32,
    /// `[actors] game_wall_timeout_s` (AMB-042-12).
    game_wall_timeout_s: u32,
    /// 042-R11: `<shared>/runs/<run>/replays` and the regular sample interval.
    replay_dir: PathBuf,
    replay_every: u64,
}

/// 042-R10 ①② and the queue close (042-R3), once: run by the clean stop, and by `Drop` when the
/// run loop returns early with an error, so no game is left waiting for a queue place.
struct Halt {
    shared: Arc<Shared>,
    inflight: uranium_runner::InFlight,
    done: AtomicBool,
}

impl Halt {
    fn stop(&self) {
        if self.done.swap(true, Ordering::SeqCst) {
            return;
        }
        self.shared.stopping.store(true, Ordering::SeqCst);
        self.inflight.stop_all();
        for q in &self.shared.queues {
            q.close();
        }
    }
}

impl Drop for Halt {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Inference servers of league snapshots, least recently used first out.
struct SnapshotPool {
    dirs: HashMap<String, PathBuf>,
    live: VecDeque<(String, InferenceHandle)>,
    cap: usize,
    servers: SnapshotServers,
}

impl SnapshotPool {
    fn handle(&mut self, id: &str) -> Result<InferenceHandle> {
        if let Some(i) = self.live.iter().position(|(k, _)| k == id) {
            let e = self.live.remove(i).unwrap();
            self.live.push_back(e.clone());
            return Ok(e.1);
        }
        let dir = self.dirs.get(id).with_context(|| format!("unknown snapshot {id}"))?;
        let h = (self.servers)(dir)?;
        self.live.push_back((id.to_string(), h.clone()));
        while self.live.len() > self.cap {
            self.live.pop_front();
        }
        Ok(h)
    }
}

/// `LearnerHook` of one learning agent: z per episode and the trajectory observer (042).
struct Hook {
    agent: AgentKind,
    handle: InferenceHandle,
    seq_len: usize,
    shared: Arc<Shared>,
    tx: mpsc::Sender<GameResult>,
    z: Vec<ZEntry>,
    z_drop: f64,
    rng: Mutex<u64>,
}

impl LearnerHook for Hook {
    fn handle(&self) -> &InferenceHandle {
        &self.handle
    }

    fn session(&self, game_id: u64, info: &GameInfo) -> (Option<Strategy>, Box<dyn SessionObserver>) {
        if let Some(m) = self.shared.games.lock().unwrap().get_mut(&game_id) {
            m.self_player = Some(info.self_player);
            if let Some(e) = info.players.iter().find(|p| p.is_enemy) {
                if m.enemy_race == Race::Random || m.enemy_race == Race::Unknown {
                    m.enemy_race = e.race;
                }
            }
        }
        // 033-R2 / R8: z from the table; z_drop of the games (or all without a table) have none.
        let z = {
            let mut r = self.rng.lock().unwrap();
            let u = splitmix(&mut r);
            if self.z.is_empty() || u < self.z_drop {
                None
            } else {
                let e = &self.z[(splitmix(&mut r) * self.z.len() as f64) as usize % self.z.len()];
                Some(Strategy { build: e.build.clone(), units: e.units.clone() })
            }
        };
        let obs = TrajectoryObserver::new(self.agent.index(), game_id, self.seq_len, self.shared.queues[self.agent.index()].clone(), self.tx.clone()).hold_terminal(held_sink(self.shared.clone()));
        (z, Box::new(obs))
    }
}

fn held_sink(s: Arc<Shared>) -> Box<dyn Fn(Segment) + Send> {
    Box::new(move |seg: Segment| {
        s.held.lock().unwrap().insert(seg.game_id, seg);
    })
}

fn splitmix(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}

/// League-driven job source (020-R12).
struct LeagueSource {
    shared: Arc<Shared>,
    hooks: Vec<Arc<Hook>>,
}

impl LeagueSource {
    fn job(&self, pick: Pick) -> Result<Job> {
        let s = &self.shared;
        let id = s.next_id.fetch_add(1, Ordering::SeqCst);
        let bridge = |race| PlayerSpec { race, kind: PlayerKind::Bridge { agent_addr: s.agent_addr.clone() }, name: None };
        let mut t0_builds = [T0Build::Standard; 2];
        let mut policies: [Option<PolicySpec>; 2] = [Some(PolicySpec::Learner(self.hooks[pick.agent.index()].clone())), None];
        let (p1, enemy_race, opponent) = match &pick.opponent {
            Opponent::T0 { race, build } => {
                t0_builds[1] = *build;
                (bridge(*race), *race, format!("{}@{}", pick.key, env!("CARGO_PKG_VERSION")))
            }
            Opponent::T1 { name, race } => (PlayerSpec { race: *race, kind: PlayerKind::OssBot { name: name.clone() }, name: None }, *race, pick.key.clone()),
            Opponent::SelfPlay => {
                policies[1] = Some(PolicySpec::Model(s.handles[AgentKind::Main.index()].clone()));
                (bridge(Race::Terran), Race::Terran, pick.key.clone())
            }
            Opponent::Snapshot { id } => {
                let h = s.snapshots.lock().unwrap().handle(id)?;
                policies[1] = Some(PolicySpec::Model(h));
                (bridge(Race::Terran), Race::Terran, id.clone())
            }
        };
        let seed = (splitmix(&mut (id ^ 0xA5A5_5A5A)) * u32::MAX as f64) as u32;
        let spec = GameSpec {
            game_id: id,
            map: if pick.fighting_spirit { MapKey::FightingSpirit } else { MapKey::CircuitBreaker },
            players: [bridge(Race::Terran), p1],
            seed: Some(seed),
            max_frames: s.max_frames,
            privileged: true,
            // 042-R11: every game is recorded; `done` keeps the sample and disagreements.
            save_replay: true,
            // AMB-042-12 / 020-R3: queue waits count too; Timeout is left out like Aborted.
            wall_timeout_s: s.game_wall_timeout_s,
        };
        s.games.lock().unwrap().insert(id, GameMeta { pick, self_player: None, enemy_race });
        Ok(Job { matchup: 0, spec, t0_builds, models: [None, None], policies, opponent: Some(opponent) })
    }

    /// Waits for the session's report and the terminal segment (they race hostd's Finished).
    fn collect(&self, game_id: u64) -> (Option<GameResult>, Option<Segment>) {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            {
                let rx = self.shared.results.lock().unwrap();
                let mut f = self.shared.finished.lock().unwrap();
                while let Ok(r) = rx.try_recv() {
                    f.insert(r.game_id, r);
                }
            }
            let r = self.shared.finished.lock().unwrap().remove(&game_id);
            if r.is_some() || Instant::now() > deadline {
                let seg = self.shared.held.lock().unwrap().remove(&game_id);
                return (r, seg);
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    }
}

impl JobSource for LeagueSource {
    fn next(&self) -> Option<Job> {
        if self.shared.stopping.load(Ordering::SeqCst) {
            return None;
        }
        let pick = self.shared.league.lock().unwrap().next();
        match self.job(pick) {
            Ok(j) => Some(j),
            Err(e) => {
                crate::elog!("[rl] cannot build a game: {e:#}");
                None
            }
        }
    }

    fn requeue(&self, job: Job) {
        // A dynamic source simply forgets a game that never started.
        self.shared.games.lock().unwrap().remove(&job.spec.game_id);
    }

    fn done(&self, job: &Job, outcome: JobOutcome, replay: Option<Vec<u8>>) {
        let s = &self.shared;
        let id = job.spec.game_id;
        let Some(mut meta) = s.games.lock().unwrap().remove(&id) else { return };
        let agent = meta.pick.agent.index();
        if outcome != JobOutcome::Finished {
            // 042-R3: Aborted / Timeout / Stopped is never trained. Revoke before waiting for the
            // session's report: a session blocked on a full queue (backpressure) can report only
            // once its waiting push is cancelled.
            s.queues[agent].revoke(id);
        }
        // No lock of `shared` is held from here to the queue push below: a terminal segment that
        // waits for a place blocks only this host slot's worker (042 queue table).
        let (result, held) = self.collect(id);
        // 030-R11: a Random enemy reaches the agent as Unknown; the race its memory saw is the race
        // played, which the per-race league windows and games.csv need.
        if matches!(meta.enemy_race, Race::Random | Race::Unknown) {
            if let Some(r) = result.as_ref().and_then(|r| r.enemy_race) {
                meta.enemy_race = r;
            }
        }
        let ends = s.ends.take(id);
        // 020-R13: the session that ended last decides.
        let verdict = (outcome == JobOutcome::Finished).then(|| game_result(&ends)).flatten();
        let decided = match (&verdict, meta.self_player) {
            (Some((e, disagree)), Some(me)) => {
                if *disagree {
                    s.disagree.fetch_add(1, Ordering::SeqCst);
                }
                match e.reason {
                    EndReason::Aborted | EndReason::AgentTimeout => None,
                    EndReason::TimeLimitDraw => Some(Outcome::Draw),
                    _ => Some(match e.winner {
                        Some(w) if w == me => Outcome::Win,
                        Some(_) => Outcome::Loss,
                        None => Outcome::Draw,
                    }),
                }
            }
            _ => None,
        };
        let r_win = match decided {
            Some(Outcome::Win) => 1.0,
            Some(Outcome::Loss) => -1.0,
            _ => 0.0,
        };
        match (decided, held) {
            (Some(o), Some(mut seg)) => {
                if let Some(last) = seg.steps.last_mut() {
                    last.reward[0] += r_win;
                }
                s.queues[agent].push(seg);
                s.queues[agent].finish(id);
                let stage_up = s.league.lock().unwrap().record(&meta.pick, meta.enemy_race, o);
                if stage_up {
                    crate::elog!("[rl] curriculum advanced to {:?} (042-R5)", s.league.lock().unwrap().stage);
                }
            }
            (Some(o), None) => {
                // No terminal segment (e.g. no decision): still a game for the win table.
                s.queues[agent].finish(id);
                s.league.lock().unwrap().record(&meta.pick, meta.enemy_race, o);
            }
            (None, _) => {
                // 042-R3: Aborted / stopped / undecidable: nothing of it is trained.
                s.queues[agent].revoke(id);
                s.aborted.fetch_add(1, Ordering::SeqCst);
            }
        }
        let disagree = verdict.as_ref().is_some_and(|v| v.1);
        let (replay_path, replay_reason) = match (replay_reason(id, disagree, s.replay_every), replay) {
            (Some(why), Some(bytes)) => match keep_replay(&s.replay_dir, id, why, &bytes) {
                Ok(name) => (format!("replays/{name}"), why),
                Err(e) => {
                    crate::elog!("[rl] replay of game {id} not kept: {e:#}");
                    (String::new(), "")
                }
            },
            _ => (String::new(), ""),
        };
        let (sum, decisions) = result.as_ref().map_or(([0.0; 4], 0), |r| (r.reward_sum, r.decisions));
        s.agent_steps.fetch_add(decisions, Ordering::SeqCst);
        s.games_done.fetch_add(1, Ordering::SeqCst);
        let line = format!(
            "{},{},{},{},{:?},{},{},{:?},{},{},{},{},{},{},{},{}\n",
            id,
            AgentKind::ALL[agent].name(),
            job.opponent.clone().unwrap_or_default(),
            match meta.pick.class {
                Class::T0 => "t0",
                Class::T1 => "t1",
                Class::Model => "model",
            },
            meta.enemy_race,
            if meta.pick.fighting_spirit { "fighting_spirit" } else { "circuit_breaker" },
            outcome.as_str(),
            decided,
            disagree,
            decisions,
            r_win + sum[0],
            sum[1],
            sum[2],
            sum[3],
            replay_path,
            replay_reason
        );
        let _ = s.games_csv.lock().unwrap().write_all(line.as_bytes());
    }
}

/// 042-R11: why a game's replay is kept, if it is.
fn replay_reason(game_id: u64, disagree: bool, every: u64) -> Option<&'static str> {
    if disagree {
        Some("disagree")
    } else if every > 0 && game_id % every == 0 {
        Some("sample")
    } else {
        None
    }
}

/// 042-R11: writes `g<id>-<reason>.rep` under a temporary name, reads it back and compares the
/// bytes before the rename (a share can lose a write silently), and returns the file name.
fn keep_replay(dir: &Path, game_id: u64, reason: &str, bytes: &[u8]) -> Result<String> {
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    let name = format!("g{game_id}-{reason}.rep");
    let tmp = dir.join(format!("{name}.part"));
    std::fs::write(&tmp, bytes).with_context(|| format!("writing {}", tmp.display()))?;
    if std::fs::read(&tmp)? != bytes {
        let _ = std::fs::remove_file(&tmp);
        anyhow::bail!("{} reads back different bytes", tmp.display());
    }
    std::fs::rename(&tmp, dir.join(&name))?;
    Ok(name)
}

/// games.csv header (042-R2, R11).
const GAMES_HEADER: &str = "game_id,agent,opponent,class,enemy_race,map,hostd_outcome,outcome,winner_disagree,decisions,r_win,r_build,r_units,r_value,replay,replay_reason\n";

/// 042-R2 train.csv header / row. `trained_steps`: agent steps the learners trained on, run
/// total (continued from the checkpoint). `queue_wait_s`: seconds the games waited for a place
/// in this agent's queue (backpressure) since this agent's previous row, counted when each wait
/// ends.
fn train_header() -> String {
    let mut h = String::from("time_s,agent,update,policy_version,agent_steps,games,steps,loss,policy,value,kl,kl_coef,rho_mean,adv_mean,lag_mean");
    for n in HEAD_NAMES {
        h += &format!(",kl_{n}");
    }
    for n in HEAD_NAMES {
        h += &format!(",entropy_{n}");
    }
    h += ",dropped_lag,dropped_full,revoked,used_aborted,winner_disagree,aborted_games,steps_per_s,stage,win_open,trained_steps,queue_wait_s\n";
    h
}

/// Opens train.csv for appending. A file whose header differs (an older version with other
/// columns) is kept under a new name and a fresh file is started, so a file's rows always match
/// its header.
fn open_train_csv(path: &Path) -> Result<std::fs::File> {
    let header = train_header();
    if let Ok(f) = std::fs::File::open(path) {
        let mut first = String::new();
        std::io::BufRead::read_line(&mut std::io::BufReader::new(f), &mut first)?;
        if !first.is_empty() && first.trim_end() != header.trim_end() {
            let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
            let old = path.with_file_name(format!("train-{secs}.csv"));
            std::fs::rename(path, &old).with_context(|| format!("moving {} aside", path.display()))?;
            crate::elog!("[rl] train.csv has other columns: kept as {}, new file started (042-R2)", old.display());
        }
    }
    let mut csv = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
    if csv.metadata()?.len() == 0 {
        csv.write_all(header.as_bytes())?;
    }
    Ok(csv)
}

/// TrainState.extra of an RL checkpoint (061-R10): games and trained agent steps (042-R2).
fn state_extra(shared: &Shared, trained_steps: u64) -> serde_json::Value {
    serde_json::json!({ "games": shared.games_done.load(Ordering::SeqCst), "trained_steps": trained_steps })
}

/// 042-R2: `trained_steps` of a resumed checkpoint; 0 for one written before it was recorded.
fn resumed_trained_steps(extra: &serde_json::Value) -> u64 {
    extra.get("trained_steps").and_then(|v| v.as_u64()).unwrap_or(0)
}

fn load_z(path: &Option<PathBuf>) -> Result<Vec<ZEntry>> {
    let Some(p) = path else { return Ok(vec![]) };
    let text = std::fs::read_to_string(p).with_context(|| format!("reading z table {}", p.display()))?;
    text.lines().filter(|l| !l.trim().is_empty()).map(|l| Ok(serde_json::from_str(l)?)).collect()
}

/// Everything the checkpoint needs from the learners (061-R10).
/// B-8 does not apply here: every caller advances `step` first (snapshot and timed checkpoints
/// `step += 1`, the final one `max(step, agent steps) + 1`), so one session never writes the same
/// `ckpt-<step>` twice, whatever an earlier push did.
fn save_checkpoint(run: &mut Run, step: u64, learners: &mut dyn RlLearners, cfg: &RlConfig, league: &League, args: &RlArgs, state_extra: serde_json::Value) -> Result<PushOutcome> {
    let dir = run.stage_dir(step)?;
    for a in AgentKind::ALL {
        let i = a.index();
        let train = serde_json::json!({ "rl": cfg, "algo": args.algo, "update": learners.version(i), "agent": a.name() });
        learners.save(i, &ckpt::agent_dir(&dir, a.name())?, &train)?;
    }
    ckpt::save_league(&dir, &league.to_bytes())?;
    let updates: Vec<u64> = AgentKind::ALL.iter().map(|a| learners.version(a.index())).collect();
    ckpt::save_state(&dir, &TrainState { kind: "rl".into(), step, update: updates[0], policy_version: updates[0], rng: vec![], extra: state_extra })?;
    let reward = PathBuf::from("config/reward.toml");
    let mut files: Vec<&Path> = vec![&args.config];
    if reward.is_file() {
        files.push(&reward);
    }
    ckpt::copy_configs(&dir, &files)?;
    run.seal(step)?;
    let out = run.push(step);
    // 061-R8 / 060-R7 (C-3): at every checkpoint.
    check_storage(run);
    out
}

/// 061-R8: shared drive and cache disk usage; 060-R7: `runs/*/replays/` against the replay
/// budget (042-R11: never deleted), measured on a background thread so the learner does not wait
/// for the share walk. Warnings only.
fn check_storage(run: &Run) {
    crate::disk::warn_storage("rl", &run.cfg);
    crate::disk::spawn_replay_check(&run.cfg.shared, crate::disk::REPLAY_BUDGET_BYTES);
}

/// B-7 / 061-R6: a failed prune is logged; training goes on.
fn prune_logged(run: &mut Run) {
    match run.prune() {
        Ok(0) => {}
        Ok(n) => crate::elog!("[rl] prune: {n} checkpoint folder(s) not deleted, left for the next prune (061-R6)"),
        Err(e) => crate::elog!("[rl] WARNING: prune failed ({e:#}); training goes on (061-R6)"),
    }
}

/// Runs RL until a limit or Ctrl+C. `make(cfg, resume)` builds the learners and their inference
/// servers once the run is locked: from the BC checkpoint, or from the agent folders of the
/// resumed checkpoint (`resume`, 061-R10), and returns the league snapshot server factory.
pub fn run(args: RlArgs, make: impl FnOnce(&RlConfig, Option<&Path>) -> Result<(Box<dyn RlLearners>, SnapshotServers)>) -> Result<()> {
    // 042-R10 / 060-R6: the caller installed the handler first, so an early SIGINT is never lost.
    let stop = args.stop.clone();
    let cfg = RlConfig::load(&args.config)?;
    std::fs::create_dir_all(&args.log_dir)?;

    // 061-R3 / R5: lock the run, recover or pull LATEST.
    let storage = StorageConfig::from_env()?;
    let replay_dir = storage.shared.join("runs").join(&args.run).join("replays");
    if crate::stopping(&stop) {
        crate::elog!("[rl] Ctrl+C before start: not started");
        return Ok(());
    }
    let mut storage = storage;
    // 061-R6 (G-D6): the run's retention (`keep_checkpoints`, 0 = all).
    storage.keep_latest = cfg.keep_checkpoints.unwrap_or(uranium_data::bc::DEFAULT_KEEP_CHECKPOINTS);
    let mut run = Run::acquire(storage, &args.run, args.take_over)?;
    // 061-R8 / 060-R7 (C-3): disk usage and the replay total at the start (warnings only).
    check_storage(&run);
    let resume = match run.begin() {
        Ok(r) => r,
        Err(e) if e.downcast_ref::<crate::Interrupted>().is_some() => {
            crate::elog!("[rl] {e:#}; releasing the lock (042-R10)");
            run.release()?;
            return Ok(());
        }
        Err(e) => return Err(e),
    };
    // Ctrl+C during the pull: release before loading anything (nothing new to save).
    if crate::stopping(&stop) {
        crate::elog!("[rl] Ctrl+C during start: releasing the lock, nothing to save (042-R10)");
        run.release()?;
        return Ok(());
    }
    // The learners and their inference servers (from BC, or the resumed agents' folders).
    let (mut learners, servers) = match make(&cfg, resume.as_ref().map(|r| r.dir.as_path())) {
        Ok(x) => x,
        Err(e) => {
            let _ = run.release();
            return Err(e.context("starting the learners (042 start / KL reference)"));
        }
    };
    let t1 = cfg_t1(&args.config)?;
    let mut league = match resume.as_ref().map(|r| ckpt::load_league(&r.dir)).transpose()?.flatten() {
        Some(b) => League::from_bytes(&b, cfg.clone())?,
        None => League::new(&args.run, t1, args.seed, cfg.clone()),
    };
    if args.fixed_t0.is_some() {
        league.fixed_t0 = args.fixed_t0;
    }
    let mut step = resume.as_ref().map_or(0, |r| r.step);
    // 042-R2: trained agent steps continue from the checkpoint too (0 for one written before).
    let mut trained_steps = match &resume {
        Some(r) => ckpt::load_state(&r.dir)?.map_or(0, |s| resumed_trained_steps(&s.extra)),
        None => 0,
    };
    if let Some(r) = &resume {
        crate::elog!("[rl] resumed {} at step {} (recovered {}), stage {:?}", args.run, r.step, r.recovered, league.stage);
    }

    // The agents' inference servers already run the learners' current weights.
    let handles: Vec<InferenceHandle> = AgentKind::ALL.iter().map(|a| learners.handle(a.index())).collect();
    let snap_dirs: HashMap<String, PathBuf> = league
        .snapshots
        .iter()
        .map(|(a, id)| (id.clone(), run.shared_dir().join("league").join(snapshot_name(id)).join(ckpt::AGENTS_DIR).join(a.name())))
        .collect();

    // 042 queue table: `queue_batches` x `lanes` per agent (>= lanes, so a full queue is always
    // ready for the learner and a waiting game is never stuck behind it), backpressure when full.
    let wait_max = Duration::from_secs(cfg.actors.queue_wait_max_s);
    let queues: Vec<Arc<SegmentQueue>> = (0..3).map(|_| Arc::new(SegmentQueue::with_policy(cfg.actors.queue_batches * cfg.lanes, cfg.actors.queue_full, wait_max))).collect();
    let (tx, rx) = mpsc::channel();
    let routes = Routes::default();
    let builds = T0Builds::default();
    let ends = EndBoard::default();
    let listen = uranium_runner::env_or("URANIUM_AGENT_LISTEN", "127.0.0.1:7600");
    let agent_addr = uranium_runner::start_agent(&listen, &args.log_dir.join("sessions.jsonl"), &args.log_dir.join("unit_rejections.jsonl"), routes.clone(), builds.clone(), ends.clone())?;
    let games_csv_path = args.log_dir.join("games.csv");
    let new_games = !games_csv_path.exists();
    let mut games_csv = std::fs::OpenOptions::new().create(true).append(true).open(&games_csv_path)?;
    if new_games {
        games_csv.write_all(GAMES_HEADER.as_bytes())?;
    }
    let base_id = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0) * 1000;
    let shared = Arc::new(Shared {
        league: Mutex::new(league),
        games: Mutex::new(HashMap::new()),
        held: Mutex::new(HashMap::new()),
        results: Mutex::new(rx),
        finished: Mutex::new(HashMap::new()),
        queues: queues.clone(),
        ends,
        next_id: AtomicU64::new(base_id),
        stopping: AtomicBool::new(false),
        agent_addr,
        handles: handles.clone(),
        snapshots: Mutex::new(SnapshotPool { dirs: snap_dirs, live: VecDeque::new(), cap: 8, servers }),
        games_csv: Mutex::new(games_csv),
        // 042-R12: agent steps continue from the checkpoint (TrainState.step).
        agent_steps: AtomicU64::new(step),
        games_done: AtomicU64::new(0),
        disagree: AtomicU64::new(0),
        aborted: AtomicU64::new(0),
        max_frames: 86_400,
        game_wall_timeout_s: cfg.actors.game_wall_timeout_s,
        replay_dir,
        replay_every: cfg.replays.sample_every,
    });
    let z = load_z(&args.z_table)?;
    if z.is_empty() {
        crate::elog!("[rl] no z table: every game runs without z, pseudo-rewards off (033-R8)");
    }
    let hooks: Vec<Arc<Hook>> = AgentKind::ALL
        .iter()
        .map(|&a| Arc::new(Hook { agent: a, handle: handles[a.index()].clone(), seq_len: cfg.seq_len, shared: shared.clone(), tx: tx.clone(), z: z.clone(), z_drop: 0.25, rng: Mutex::new(args.seed ^ (a.index() as u64 + 1)) }))
        .collect();
    let source = Arc::new(LeagueSource { shared: shared.clone(), hooks });
    let dispatch = Dispatch::new(routes, Default::default(), builds);
    // 042-R3: from here on any return, an error included, stops the games and closes the queues.
    let halt = Halt { shared: shared.clone(), inflight: dispatch.inflight.clone(), done: AtomicBool::new(false) };
    let hosts: Vec<HostEntry> = uranium_runner::hosts()?;

    let games_thread = {
        let (source, dispatch, games) = (source.clone(), dispatch.clone(), args.log_dir.join("games.jsonl"));
        std::thread::spawn(move || uranium_runner::run_source(source, &hosts, &games, &dispatch))
    };

    let base = RewardWeights::load()?;
    let mut csv = open_train_csv(&args.log_dir.join("train.csv"))?;
    let started = Instant::now();
    // Each agent's queue wait total at its previous train.csv row (042-R2 `queue_wait_s`).
    let mut wait_seen = [0.0f64; 3];
    let (mut last_ckpt, mut last_beat) = (Instant::now(), Instant::now());
    let mut lock_lost = false;
    let mut updates_this_run = 0u64;
    let mut last_rate = (Instant::now(), shared.agent_steps.load(Ordering::SeqCst));
    let mut steps_per_s = 0.0f64;
    loop {
        let steps_hit = args.max_agent_steps > 0 && shared.agent_steps.load(Ordering::SeqCst) >= args.max_agent_steps;
        if steps_hit {
            crate::elog!("[rl] {} agent steps reached: clean stop (042-R12)", args.max_agent_steps);
        }
        let limit_hit = steps_hit || (args.max_updates > 0 && updates_this_run >= args.max_updates) || (args.max_minutes > 0 && started.elapsed() >= Duration::from_secs(args.max_minutes * 60));
        if crate::stopping(&stop) || limit_hit {
            break;
        }
        // 061-R5: heartbeat every minute; a lost lock stops training.
        if last_beat.elapsed() >= Duration::from_secs(60) {
            last_beat = Instant::now();
            match run.heartbeat() {
                LockStatus::Held => {}
                LockStatus::Lost(l) => {
                    crate::elog!("[rl] lock lost to {l:?}: stopping (061-R5)");
                    lock_lost = true;
                    break;
                }
                LockStatus::Unreachable(e) => crate::elog!("[rl] heartbeat: share unreachable ({e}), continuing (061-R11-3)"),
            }
        }
        let ready: Vec<AgentKind> = AgentKind::ALL.iter().copied().filter(|a| queues[a.index()].len() >= cfg.lanes).collect();
        let turn = shared.league.lock().unwrap().learner_turn(&ready);
        let Some(agent) = turn else {
            std::thread::sleep(Duration::from_millis(200));
            continue;
        };
        let i = agent.index();
        let Some(batch) = queues[i].try_take(cfg.lanes, learners.version(i), cfg.actors.max_policy_lag) else { continue };
        let weights = shared.league.lock().unwrap().reward_weights(&base);
        let trained = trained_steps_of(&batch);
        let stats = learners.update(i, batch, weights)?;
        // 042-R2 / R12: agent steps the learners trained on, run total.
        trained_steps += trained;
        updates_this_run += 1;
        let snap = shared.league.lock().unwrap().on_update(agent);
        // 042 K: weights to the actors every `weight_push_every` updates.
        if learners.version(i) % cfg.actors.weight_push_every.max(1) == 0 {
            learners.push(i)?;
        }
        step = step.max(shared.agent_steps.load(Ordering::SeqCst));
        if last_rate.0.elapsed() >= Duration::from_secs(60) {
            let now = shared.agent_steps.load(Ordering::SeqCst);
            steps_per_s = (now - last_rate.1) as f64 / last_rate.0.elapsed().as_secs_f64();
            last_rate = (Instant::now(), now);
        }
        let wait_total = queues[i].stats().wait_s;
        let queue_wait_s = wait_total - wait_seen[i];
        wait_seen[i] = wait_total;
        write_row(&mut csv, started, agent, &stats, &shared, &queues, learners.version(i), steps_per_s, trained_steps, queue_wait_s)?;
        if let Some(reset) = snap {
            // A league snapshot is a pushed checkpoint copied to league/<name> (061-R2).
            let id = shared.league.lock().unwrap().add_snapshot(agent);
            let league_now = shared.league.lock().unwrap().clone();
            step += 1;
            match save_checkpoint(&mut run, step, learners.as_mut(), &cfg, &league_now, &args, state_extra(&shared, trained_steps))? {
                PushOutcome::Pushed => {
                    let dir = run.league_snapshot(step, &snapshot_name(&id))?;
                    shared.snapshots.lock().unwrap().dirs.insert(id.clone(), dir.join(ckpt::AGENTS_DIR).join(agent.name()));
                    crate::elog!("[rl] snapshot {id}");
                }
                other => crate::elog!("[rl] snapshot {id} not published: {other:?}"),
            }
            if reset {
                learners.reset_to_bc(i)?;
                learners.push(i)?;
            }
            last_ckpt = Instant::now();
        }
        if last_ckpt.elapsed() >= Duration::from_secs(args.checkpoint_minutes.max(1) * 60) {
            step += 1;
            let league_now = shared.league.lock().unwrap().clone();
            let out = save_checkpoint(&mut run, step, learners.as_mut(), &cfg, &league_now, &args, state_extra(&shared, trained_steps))?;
            crate::elog!("[rl] checkpoint {} {:?}", crate::storage::ckpt_name(step), out);
            if out == PushOutcome::Pushed {
                prune_logged(&mut run);
            }
            last_ckpt = Instant::now();
        }
    }

    // 042-R10: ① no new games ② stop games in flight (Aborted, revoked) ③ checkpoint + push
    // ④ release the lock only when the push went through (061-R11-5/6).
    halt.stop();
    let _ = games_thread.join();
    if lock_lost {
        crate::elog!("[rl] lock lost: no final push, lock left as is (061-R11-4)");
        return Ok(());
    }
    step = step.max(shared.agent_steps.load(Ordering::SeqCst)) + 1;
    let league_now = shared.league.lock().unwrap().clone();
    let out = save_checkpoint(&mut run, step, learners.as_mut(), &cfg, &league_now, &args, state_extra(&shared, trained_steps))?;
    crate::elog!("[rl] final checkpoint {}: {out:?}", crate::storage::ckpt_name(step));
    match out {
        PushOutcome::Pushed => run.release()?,
        other => crate::elog!("[rl] push {other:?}: keeping the lock so the next start recovers it (061-R11-5/6)"),
    }
    Ok(())
}

/// Snapshot folder name (061-R9 forbids `/` and `@`): `<agent>-<run>-<update>`.
pub fn snapshot_name(id: &str) -> String {
    id.trim_start_matches("uranium-").replace(['@', '/'], "-")
}

#[allow(clippy::too_many_arguments)]
fn write_row(csv: &mut std::fs::File, started: Instant, agent: AgentKind, s: &UpdateStats, shared: &Shared, queues: &[Arc<SegmentQueue>], version: u64, steps_per_s: f64, trained_steps: u64, queue_wait_s: f64) -> Result<()> {
    let q = queues[agent.index()].stats();
    let mut row = format!(
        "{:.0},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
        started.elapsed().as_secs_f64(),
        agent.name(),
        s.update,
        version,
        shared.agent_steps.load(Ordering::SeqCst),
        shared.games_done.load(Ordering::SeqCst),
        s.steps,
        s.loss,
        s.policy,
        s.value,
        s.kl,
        s.kl_coef,
        s.rho_mean,
        s.adv_mean,
        s.lag_mean
    );
    for v in s.kl_heads {
        row += &format!(",{v}");
    }
    for v in s.entropy_heads {
        row += &format!(",{v}");
    }
    let stage = shared.league.lock().unwrap().stage;
    row += &format!(
        ",{},{},{},{},{},{},{:.2},{:?},{},{},{:.3}\n",
        q.dropped_lag,
        q.dropped_full,
        q.revoked,
        q.used_aborted,
        shared.disagree.load(Ordering::SeqCst),
        shared.aborted.load(Ordering::SeqCst),
        steps_per_s,
        stage,
        s.win_open,
        trained_steps,
        queue_wait_s
    );
    csv.write_all(row.as_bytes())?;
    Ok(())
}

/// `[[t1]]` of rl.toml: T1 bots and the races they play (021).
fn cfg_t1(path: &Path) -> Result<Vec<T1Bot>> {
    #[derive(Deserialize)]
    struct F {
        #[serde(default)]
        t1: Vec<T1Bot>,
    }
    Ok(toml::from_str::<F>(&std::fs::read_to_string(path)?)?.t1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_names_are_smb_safe_061_r9() {
        let n = snapshot_name("uranium-main-exploiter@run1/4000");
        assert_eq!(n, "main-exploiter-run1-4000");
        crate::storage::check_name(&n).unwrap();
    }

    #[test]
    fn rl_toml_lists_t1_bots_021() {
        let bots = cfg_t1(Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../config/rl.toml"))).unwrap();
        assert!(bots.iter().any(|b| b.name == "zzzkbot" && b.races == vec![Race::Zerg]));
    }

    /// 042-R2: train.csv ends with `trained_steps` and `queue_wait_s` (2026-10-31).
    #[test]
    fn train_csv_header_columns_042_r2() {
        assert_eq!(train_header().trim().split(',').count(), 15 + 12 + 11);
        assert!(train_header().trim().ends_with(",win_open,trained_steps,queue_wait_s"));
    }

    /// 042-R2: a train.csv with an older header is kept aside, the new file gets the current
    /// header; a current file is appended to as is.
    #[test]
    fn train_csv_with_other_columns_is_kept_aside_042_r2() {
        let dir = std::env::temp_dir().join(format!("uranium-rl-csv-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("train.csv");
        std::fs::write(&path, "time_s,agent\n1,main\n").unwrap();
        drop(open_train_csv(&path).unwrap());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), train_header());
        let kept: Vec<_> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().file_name().into_string().unwrap()).filter(|n| n.starts_with("train-")).collect();
        assert_eq!(kept.len(), 1);
        assert_eq!(std::fs::read_to_string(dir.join(&kept[0])).unwrap(), "time_s,agent\n1,main\n");
        let mut f = open_train_csv(&path).unwrap();
        f.write_all(b"row\n").unwrap();
        drop(f);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), format!("{}row\n", train_header()));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// 042-R2: `trained_steps` continues from the checkpoint; older checkpoints start at 0.
    #[test]
    fn trained_steps_resume_from_checkpoint_042_r2() {
        assert_eq!(resumed_trained_steps(&serde_json::json!({ "games": 3, "trained_steps": 12_345 })), 12_345);
        assert_eq!(resumed_trained_steps(&serde_json::json!({ "games": 3 })), 0);
        assert_eq!(resumed_trained_steps(&serde_json::Value::Null), 0);
    }

    /// 042-R11: one regular sample per `every` game ids, every disagreement, nothing else.
    #[test]
    fn replay_reason_samples_and_disagreements_042_r11() {
        assert_eq!(replay_reason(200, false, 100), Some("sample"));
        assert_eq!(replay_reason(201, false, 100), None);
        assert_eq!(replay_reason(201, true, 100), Some("disagree"));
        assert_eq!(replay_reason(200, true, 100), Some("disagree"));
        assert_eq!(replay_reason(200, false, 0), None, "0 = no regular sample");
        let kept = (1_000u64..2_000).filter(|id| replay_reason(*id, false, 100).is_some()).count();
        assert_eq!(kept, 10);
    }

    /// 042-R11: the kept replay is complete under its final name, no `.part` left behind.
    #[test]
    fn keep_replay_writes_verified_file_042_r11() {
        let dir = std::env::temp_dir().join(format!("uranium-rl-rep-{}", std::process::id()));
        let name = keep_replay(&dir, 7, "disagree", b"replay bytes").unwrap();
        assert_eq!(name, "g7-disagree.rep");
        assert_eq!(std::fs::read(dir.join(&name)).unwrap(), b"replay bytes");
        assert!(!dir.join("g7-disagree.rep.part").exists());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn games_csv_header_matches_rows_042_r2() {
        assert_eq!(GAMES_HEADER.trim().split(',').count(), 16);
        assert!(GAMES_HEADER.trim().ends_with(",replay,replay_reason"));
    }
}
