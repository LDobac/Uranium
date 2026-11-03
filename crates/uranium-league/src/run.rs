//! The RL run loop (042): one process runs the learner, the inference servers, the agent server,
//! the game dispatch (`uranium-runner`, 020-R12) and the league. The learner and the servers are
//! the PyTorch side (`python -m uranium rl`, AMB-000-4) behind [`RlLearners`].
//!
//! Game outcomes follow 020-R13: once hostd reports a game over, the GameEnd reports of its
//! sessions are reconciled (`game_result`) and that verdict sets r_win on the learner's terminal
//! segment (033) and feeds the league win table (042). Aborted games are revoked (042-R3).
//! Checkpoints go through `storage::Run` (061): stage → seal → push; a clean stop (042-R10)
//! releases the lock only when the final push succeeded (061-R11-5/6).

use std::collections::{BTreeSet, HashMap, VecDeque};
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
use uranium_agent::{ConcedeBoard, EndBoard, PolicySpec, Routes, T0Build, T0Builds, game_result};
use uranium_env::concede::{ConcedeConfig, ConcedeGame, LeaveRule};
use uranium_env::features::Strategy;
use uranium_env::reward::{Outcome, RewardWeights};
use uranium_env::scout::{PseudoMode, RewardConfig, ScoutMetrics, seconds};
use uranium_proto::control::{GameSpec, MapKey, PlayerKind, PlayerSpec};
use uranium_proto::{EndReason, GameInfo, Race};
use uranium_runner::{Dispatch, HostEntry, Job, JobOutcome, JobSource};

use crate::ckpt::{self, TrainState};
use crate::config::RlConfig;
use crate::league::{AgentKind, Class, League, Opponent, Pick};
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
    /// Id of the snapshot / anchor server this game holds (`SnapshotPool::acquire`).
    server: Option<u64>,
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
    /// 042-R14 (AMB-042-15): `[concede]` and the games that judge it (shared with the agent
    /// server); a game is listed from dispatch until it is done.
    concede_cfg: ConcedeConfig,
    concede: ConcedeBoard,
    /// 033-R11 channel mode: `games.csv` `r_scout` / `r_survive` are empty in `"z"` (042-R2).
    pseudo: PseudoMode,
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

/// Most snapshot / anchor inference servers kept running while idle (least recently used shut
/// down first); servers with games in flight are never shut down.
const SNAPSHOT_SERVERS_MAX: usize = 8;

/// A snapshot / anchor whose server failed (start error, or its thread died) is not tried again
/// for this long: its picks become current-Main self-play without reloading under the pool lock.
const SNAPSHOT_RETRY_AFTER: Duration = Duration::from_secs(300);

/// One running snapshot / anchor inference server.
struct LiveServer {
    key: String,
    /// Unique per started server: a game releases the server it got, not a later one of the key.
    id: u64,
    handle: InferenceHandle,
    close: Box<dyn FnOnce() + Send>,
    alive: Box<dyn Fn() -> bool + Send>,
    /// Games dispatched against it and not done yet.
    games: usize,
}

/// Inference servers of league snapshots and the anchor (042 opponents), started on first use.
/// 042 스냅샷 추론 서버 (AMB-042-14): a server no pick can choose any more
/// (`League::model_opponents`, e.g. a snapshot that left the C1 candidates) is shut down once its
/// games are done, which frees its device memory; so is the least recently used idle one above
/// `cap`. The anchor's server (`pinned`) stays. Snapshot files are never deleted.
struct SnapshotPool {
    /// Model folder per key (snapshot id, anchor key): published snapshots and the anchor.
    dirs: HashMap<String, PathBuf>,
    /// Least recently used first.
    live: VecDeque<LiveServer>,
    cap: usize,
    pinned: Option<String>,
    servers: SnapshotServers,
    /// Keys whose server failed, and when (`retry_after`).
    failed: HashMap<String, Instant>,
    retry_after: Duration,
    next_id: u64,
}

/// A shut-down function with the server's key, run without the pool's lock.
type Closer = (String, Box<dyn FnOnce() + Send>);

impl SnapshotPool {
    fn new(dirs: HashMap<String, PathBuf>, pinned: Option<String>, servers: SnapshotServers) -> Self {
        Self { dirs, live: VecDeque::new(), cap: SNAPSHOT_SERVERS_MAX, pinned, servers, failed: HashMap::new(), retry_after: SNAPSHOT_RETRY_AFTER, next_id: 0 }
    }

    /// The server of `key` for one more game (started if needed) and its id; pair with
    /// `release`. A cached server whose thread died is closed and counts as a failure: an error
    /// (the caller's self-play fallback), and no new start for `retry_after`.
    fn acquire(&mut self, key: &str) -> Result<(InferenceHandle, u64)> {
        if let Some(at) = self.failed.get(key) {
            if at.elapsed() < self.retry_after {
                anyhow::bail!("its server failed {:.0} s ago (retried after {} s)", at.elapsed().as_secs_f64(), self.retry_after.as_secs());
            }
            self.failed.remove(key);
        }
        if let Some(i) = self.live.iter().position(|l| l.key == key) {
            let mut l = self.live.remove(i).unwrap();
            if !(l.alive)() {
                (l.close)();
                self.failed.insert(key.to_string(), Instant::now());
                anyhow::bail!("its inference server stopped (model thread ended)");
            }
            l.games += 1;
            let out = (l.handle.clone(), l.id);
            self.live.push_back(l);
            return Ok(out);
        }
        let dir = self.dirs.get(key).with_context(|| format!("no published model folder for {key}"))?;
        let s = match (self.servers)(dir) {
            Ok(s) => s,
            Err(e) => {
                self.failed.insert(key.to_string(), Instant::now());
                return Err(e);
            }
        };
        self.next_id += 1;
        let out = (s.handle.clone(), self.next_id);
        self.live.push_back(LiveServer { key: key.to_string(), id: self.next_id, handle: s.handle, close: s.close, alive: s.alive, games: 1 });
        Ok(out)
    }

    /// A game dispatched with `acquire` (server `id`) is done (or never started).
    fn release(&mut self, id: u64) {
        if let Some(l) = self.live.iter_mut().find(|l| l.id == id) {
            l.games = l.games.saturating_sub(1);
        }
    }

    /// Takes out the idle servers that `in_use` does not list, then the least recently used idle
    /// ones above `cap`, never the pinned one; returns their shut-down functions.
    fn prune(&mut self, in_use: &BTreeSet<String>) -> Vec<Closer> {
        let pinned = |l: &LiveServer| self.pinned.as_deref() == Some(l.key.as_str());
        let (mut keep, mut out) = (VecDeque::new(), Vec::new());
        for l in self.live.drain(..) {
            if l.games == 0 && !in_use.contains(&l.key) && !pinned(&l) {
                out.push((l.key, l.close));
            } else {
                keep.push_back(l);
            }
        }
        while keep.len() > self.cap {
            let Some(i) = keep.iter().position(|l| l.games == 0 && !pinned(l)) else { break };
            let l = keep.remove(i).unwrap();
            out.push((l.key, l.close));
        }
        self.live = keep;
        out
    }

    #[cfg(test)]
    fn live_keys(&self) -> Vec<&str> {
        self.live.iter().map(|l| l.key.as_str()).collect()
    }
}

/// 042 스냅샷 추론 서버 (AMB-042-14): shuts down the servers no pick can choose any more once
/// their games are done (after every game and every snapshot).
fn prune_servers(shared: &Shared) {
    let in_use = shared.league.lock().unwrap().model_opponents();
    let closers = shared.snapshots.lock().unwrap().prune(&in_use);
    for (key, close) in closers {
        close();
        crate::elog!("[rl] inference server of {key} shut down: no opponent choice uses it (042)");
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
    /// `[reward]` of the config (033-R9 / R10 / R11, AMB-042-16).
    reward: RewardConfig,
}

impl LearnerHook for Hook {
    fn handle(&self) -> &InferenceHandle {
        &self.handle
    }

    fn session(&self, game_id: u64, info: &GameInfo) -> (Option<Strategy>, Box<dyn SessionObserver>) {
        let mut class = None;
        if let Some(m) = self.shared.games.lock().unwrap().get_mut(&game_id) {
            m.self_player = Some(info.self_player);
            class = Some(m.pick.class);
            if let Some(e) = info.players.iter().find(|p| p.is_enemy) {
                if m.enemy_race == Race::Random || m.enemy_race == Race::Unknown {
                    m.enemy_race = e.race;
                }
            }
        }
        // 033 / 042 "r_value의 게임별 적용" (AMB-042-14): per game, by the opponent's class.
        let r_value_policy = class.is_none_or(|c| self.shared.league.lock().unwrap().r_value_in_policy(c));
        // 033-R11 (AMB-042-16): r_scout / r_survive policy-weight multipliers, per game the same way.
        let pseudo_scale = self.reward.pseudo_scale(class == Some(Class::Model));
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
        let obs = TrajectoryObserver::new(self.agent.index(), game_id, self.seq_len, self.shared.queues[self.agent.index()].clone(), self.tx.clone()).hold_terminal(held_sink(self.shared.clone())).r_value_policy(r_value_policy).pseudo(self.reward.pseudo, pseudo_scale);
        (z, Box::new(obs))
    }

    fn reward_config(&self) -> RewardConfig {
        self.reward.clone()
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
    fn job(&self, mut pick: Pick) -> Job {
        let s = &self.shared;
        let id = s.next_id.fetch_add(1, Ordering::SeqCst);
        let bridge = |race| PlayerSpec { race, kind: PlayerKind::Bridge { agent_addr: s.agent_addr.clone() }, name: None };
        let mut t0_builds = [T0Build::Standard; 2];
        let mut policies: [Option<PolicySpec>; 2] = [Some(PolicySpec::Learner(self.hooks[pick.agent.index()].clone())), None];
        let mut server = None;
        if matches!(pick.opponent, Opponent::Snapshot { .. } | Opponent::Anchor { .. }) {
            let got = s.snapshots.lock().unwrap().acquire(&pick.key);
            match got {
                Ok((h, server_id)) => {
                    policies[1] = Some(PolicySpec::Model(h));
                    server = Some(server_id);
                }
                Err(e) => {
                    // 042 모델 상대 실패 (AMB-042-14): the game becomes current-Main self-play;
                    // the runner slot goes on.
                    let key = s.league.lock().unwrap().self_play_key();
                    crate::elog!("[rl] game {id}: opponent {} cannot be served ({e:#}); playing the current Main instead (042)", pick.key);
                    pick = Pick { opponent: Opponent::SelfPlay, key, class: Class::Model, ..pick };
                }
            }
        }
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
            // The server was taken above (`policies[1]`).
            Opponent::Snapshot { .. } | Opponent::Anchor { .. } => (bridge(Race::Terran), Race::Terran, pick.key.clone()),
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
        s.games.lock().unwrap().insert(id, GameMeta { pick, self_player: None, enemy_race, server });
        // 042-R14: every model session of an RL league game judges collapse concede.
        if s.concede_cfg.enabled {
            s.concede.insert(id, Arc::new(ConcedeGame::new(s.concede_cfg)));
        }
        Job { matchup: 0, spec, t0_builds, models: [None, None], policies, opponent: Some(opponent) }
    }

    /// The game no longer holds its snapshot / anchor server; servers nobody can pick go down.
    fn release_server(&self, meta: &GameMeta) {
        if let Some(id) = meta.server {
            self.shared.snapshots.lock().unwrap().release(id);
        }
        prune_servers(&self.shared);
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
        // Never `None` here: that would end the runner slot (042 모델 상대 실패, AMB-042-14).
        Some(self.job(pick))
    }

    fn requeue(&self, job: Job) {
        // A dynamic source simply forgets a game that never started.
        let meta = self.shared.games.lock().unwrap().remove(&job.spec.game_id);
        self.shared.concede.take(job.spec.game_id);
        if let Some(m) = meta {
            self.release_server(&m);
        }
    }

    fn done(&self, job: &Job, outcome: JobOutcome, replay: Option<Vec<u8>>) {
        let s = &self.shared;
        let id = job.spec.game_id;
        // 042-R14: off the concede board whatever happens below.
        let concede = s.concede.take(id);
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
        // 042-R14: who conceded, if a session did (players index 0 = the learning agent).
        let left = concede.and_then(|g| g.conceded());
        let conceded = left.filter(|c| c.rule == LeaveRule::Concede);
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
        // 042-R14 games.csv `end_reason`: `Concede` for a collapse concede, else the judged
        // GameEnd reason (020-R13); empty for a game without a result (Aborted, Timeout).
        let end_reason = match (&verdict, decided) {
            (_, None) => String::new(),
            (_, Some(_)) if conceded.is_some() => "Concede".to_string(),
            (Some((e, _)), Some(_)) => format!("{:?}", e.reason),
            (None, Some(_)) => String::new(),
        };
        if let Some(c) = conceded {
            let who = if c.side == 0 { "learning agent" } else { "opponent" };
            crate::elog!("[rl] game {id}: 042-R14 collapse concede by the {who} side (players index {}) at frame {}; result {:?}", c.side, c.frame, decided);
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
        let extra = scout_columns(s.pseudo, result.as_ref(), meta.pick.ladder_group.as_deref());
        let line = format!(
            "{},{},{},{},{:?},{},{},{:?},{},{},{},{},{},{},{},{},{},{}\n",
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
            replay_reason,
            end_reason,
            extra
        );
        let _ = s.games_csv.lock().unwrap().write_all(line.as_bytes());
        self.release_server(&meta);
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

/// games.csv header (042-R2, R11; `end_reason` 042-R14, 2026-11-03; then the 033 raw sums
/// `r_scout`, `r_survive` and the 042 정찰·대응 지표, AMB-042-16).
const GAMES_HEADER: &str = "game_id,agent,opponent,class,enemy_race,map,hostd_outcome,outcome,winner_disagree,decisions,r_win,r_build,r_units,r_value,replay,replay_reason,end_reason,r_scout,r_survive,ladder_group,scout_main_s,scout_types_4m,scout_types_6m,barracks_s,bunker_s,scv_4m\n";

/// 042-R2 (AMB-042-16): the games.csv columns after `end_reason`, comma-joined. `r_scout`,
/// `r_survive`: the game's raw sums, empty in the `"z"` mode (and without a session report);
/// `ladder_group`: the ladder entry's group, empty for other games; the metrics in game seconds
/// (frame / 24), empty when not reached (`scv_4m`: the game ended before 4:00) or without a
/// session report.
fn scout_columns(mode: PseudoMode, result: Option<&GameResult>, ladder_group: Option<&str>) -> String {
    let secs = |f: Option<u32>| f.map_or(String::new(), |f| format!("{:.1}", seconds(f)));
    let (r_scout, r_survive) = match (mode, result) {
        (PseudoMode::Scout, Some(r)) => (r.scout_sum.to_string(), r.survive_sum.to_string()),
        _ => (String::new(), String::new()),
    };
    let m: Option<ScoutMetrics> = result.and_then(|r| r.scout);
    let count = |x: fn(&ScoutMetrics) -> u32| m.as_ref().map_or(String::new(), |m| x(m).to_string());
    [
        r_scout,
        r_survive,
        ladder_group.unwrap_or_default().to_string(),
        secs(m.and_then(|m| m.main_frame)),
        count(|m| m.types_4m),
        count(|m| m.types_6m),
        secs(m.and_then(|m| m.barracks_frame)),
        secs(m.and_then(|m| m.bunker_frame)),
        m.and_then(|m| m.scv_4m).map_or(String::new(), |n| n.to_string()),
    ]
    .join(",")
}

/// 033-R11: the channel mode a checkpoint was trained with (`agents/main/meta.json` `train.rl`
/// `reward.pseudo`; absent = `"z"`, the mode before the key existed). `None` without a
/// readable `meta.json` (nothing to compare).
fn checkpoint_pseudo(dir: &Path) -> Option<PseudoMode> {
    let meta = dir.join(crate::ckpt::AGENTS_DIR).join(AgentKind::Main.name()).join("meta.json");
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(meta).ok()?).ok()?;
    let rl = v.get("train")?.get("rl")?;
    match rl.get("reward").and_then(|r| r.get("pseudo")) {
        None => Some(PseudoMode::Z),
        Some(p) => serde_json::from_value(p.clone()).ok(),
    }
}

/// 033-R11 start errors: `"scout"` with a z table, and a resumed checkpoint of another mode.
fn check_pseudo(cfg: &RlConfig, z_table: Option<&Path>, resumed: Option<PseudoMode>) -> Result<()> {
    let mode = cfg.pseudo();
    if mode == PseudoMode::Scout
        && let Some(z) = z_table
    {
        anyhow::bail!("[reward] pseudo = \"scout\" uses no z: --z-table {} is not allowed with it (033-R11)", z.display());
    }
    if let Some(was) = resumed
        && was != mode
    {
        anyhow::bail!("[reward] pseudo is {mode:?} but the resumed checkpoint was trained with {was:?}: the build / units value heads would change meaning; start a new run instead (033-R11)");
    }
    Ok(())
}

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

/// Opens train.csv for appending (042-R2).
fn open_train_csv(path: &Path) -> Result<std::fs::File> {
    open_csv(path, &train_header())
}

/// Opens a log CSV (`<stem>.csv`: train.csv, games.csv) for appending. A file whose header differs
/// (an older version with other columns) is kept under a new name and a fresh file is started, so
/// a file's rows always match its header. When it cannot be moved (e.g. open in a viewer on
/// Windows), it stays as is and the rows go to `<stem>-<unix>-new.csv` instead.
fn open_csv(path: &Path, header: &str) -> Result<std::fs::File> {
    let stem = path.file_stem().map_or_else(|| "log".to_string(), |s| s.to_string_lossy().into_owned());
    let name = path.file_name().map_or_else(|| stem.clone(), |s| s.to_string_lossy().into_owned());
    let mut target = path.to_path_buf();
    if let Ok(f) = std::fs::File::open(path) {
        let mut first = String::new();
        std::io::BufRead::read_line(&mut std::io::BufReader::new(f), &mut first)?;
        if !first.is_empty() && first.trim_end() != header.trim_end() {
            let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
            let old = path.with_file_name(format!("{stem}-{secs}.csv"));
            match std::fs::rename(path, &old) {
                Ok(()) => crate::elog!("[rl] {name} has other columns: kept as {}, new file started (042-R2)", old.display()),
                Err(e) => {
                    target = path.with_file_name(format!("{stem}-{secs}-new.csv"));
                    crate::elog!("[rl] WARNING: {name} has other columns and cannot be moved ({e}); writing {} instead (042-R2)", target.display());
                }
            }
        }
    }
    let mut csv = std::fs::OpenOptions::new().create(true).append(true).open(&target).with_context(|| format!("opening {}", target.display()))?;
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
    // 033-R11: "scout" with a z table is a start error (before the run lock).
    check_pseudo(&cfg, args.z_table.as_deref(), None)?;
    std::fs::create_dir_all(&args.log_dir)?;

    // 061-R3 / R5: lock the run, recover or pull LATEST.
    let storage = StorageConfig::from_env()?;
    let replay_dir = storage.shared.join("runs").join(&args.run).join("replays");
    // 042 anchor (AMB-042-14): its model files must be there at start (a start error otherwise).
    let anchor_dir = match cfg.league.anchor_name() {
        Some(_) => Some(crate::config::anchor_agent_dir(&storage.shared, &cfg.league.anchor)?),
        None => None,
    };
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
    // 033-R11: a resumed checkpoint of another channel mode is a start error.
    if let Err(e) = check_pseudo(&cfg, None, resume.as_ref().and_then(|r| checkpoint_pseudo(&r.dir))) {
        let _ = run.release();
        return Err(e);
    }
    // The learners and their inference servers (from BC, or the resumed agents' folders).
    let (mut learners, servers) = match make(&cfg, resume.as_ref().map(|r| r.dir.as_path())) {
        Ok(x) => x,
        Err(e) => {
            let _ = run.release();
            return Err(e.context("starting the learners (042 start / KL reference)"));
        }
    };
    // 042 "재개 시 설정" (AMB-042-13): stage, win tables, windows and snapshots come from the
    // checkpoint; ratios, auto_advance, r_value_stages and `[[t1]]` from the current config. A
    // new league starts in `[curriculum] start_stage` (042-R5).
    let mut league = match resume.as_ref().map(|r| ckpt::load_league(&r.dir)).transpose()?.flatten() {
        Some(b) => League::from_bytes(&b, cfg.clone())?,
        None => League::new(&args.run, args.seed, args.fixed_t0, cfg.clone()),
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
    let cc = &cfg.curriculum;
    match &resume {
        Some(r) => crate::elog!("[rl] resumed {} at step {} (recovered {}), stage {:?}", args.run, r.step, r.recovered, league.stage),
        None => crate::elog!("[rl] new league {}, stage {:?}", args.run, league.stage),
    }
    crate::elog!("[rl] curriculum: auto_advance {}, r_value in {:?}, {} T1 bot(s) (042-R5, 033)", cc.auto_advance, cc.r_value_stages, cfg.t1.len());

    // The agents' inference servers already run the learners' current weights.
    let handles: Vec<InferenceHandle> = AgentKind::ALL.iter().map(|a| learners.handle(a.index())).collect();
    // 042 스냅샷 공개 (AMB-042-14): a snapshot is an opponent once its model files are on the
    // share (a failed publish never got them there).
    let mut snap_dirs: HashMap<String, PathBuf> = HashMap::new();
    for (a, id) in league.snapshots.clone() {
        let d = run.shared_dir().join("league").join(snapshot_name(&id)).join(ckpt::AGENTS_DIR).join(a.name());
        if crate::config::has_model_files(&d) {
            league.mark_published(&id);
            snap_dirs.insert(id, d);
        } else {
            crate::elog!("[rl] snapshot {id}: no model files in {}, not an opponent (042)", d.display());
        }
    }
    let anchor_key = league.anchor_key();
    if let (Some(k), Some(d)) = (&anchor_key, &anchor_dir) {
        snap_dirs.insert(k.clone(), d.clone());
    }
    crate::elog!(
        "[rl] C1 opponents: t0 {} t1 {} current Main {} snapshots {} (latest {} published, 0 = all) anchor {} ({}); r_value vs models {} (042, AMB-042-14)",
        cc.c1_t0,
        cc.c1_t1,
        cc.c1_self,
        cc.c1_snapshot,
        cfg.snapshot.main_candidates,
        cc.c1_anchor,
        anchor_dir.as_ref().map_or_else(|| "none".to_string(), |d| d.display().to_string()),
        cc.r_value_vs_model
    );
    let ladder: Vec<String> = cfg.ladder.iter().filter_map(|e| e.opponent().ok()).map(|(_, k, _)| k).collect();
    crate::elog!("[rl] C1 rush ladder: share {} ({:?}) {:?} (042, AMB-042-15)", cc.c1_ladder, cc.ladder_pick, ladder);
    if !cc.ladder_groups.is_empty() {
        let groups: Vec<String> = cc.ladder_groups.iter().map(|(g, x)| format!("{g} {x} ({} entries)", cfg.ladder.iter().filter(|e| &e.group == g).count())).collect();
        crate::elog!("[rl] ladder groups: {} (042, AMB-042-16)", groups.join(", "));
    }
    crate::elog!("[rl] win table decay {} (042 승률 표)", cfg.league.win_rate_decay);
    let c = &cfg.concede;
    crate::elog!("[rl] collapse concede: {} (screen supply <= {} for {} s from {} s, 042-R14)", if c.enabled { "on" } else { "off" }, c.max_supply, c.hold_s, c.after_s);
    let rw = &cfg.reward;
    match rw.pseudo {
        PseudoMode::Z => crate::elog!("[rl] reward channels: z (r_build / r_units, 033-R11)"),
        PseudoMode::Scout => {
            crate::elog!(
                "[rl] reward channels: scout (033-R11) - r_scout weight {} cap {} vs models x{}, r_survive weight {} vs models x{} ({}-{} s, {} SCVs)",
                rw.scout.weight,
                rw.scout.cap,
                rw.scout.vs_model,
                rw.survive.weight,
                rw.survive.vs_model,
                rw.survive.from_s,
                rw.survive.until_s,
                rw.survive.scv_target
            );
            if league.stage == crate::league::Stage::C2 {
                crate::elog!("[rl] WARNING: stage C2 with [reward] pseudo = \"scout\": no C2 decay is applied to r_scout / r_survive (033-R11: to be decided before C2)");
            }
        }
    }

    // 042 queue table: `queue_batches` x `lanes` per agent (>= lanes, so a full queue is always
    // ready for the learner and a waiting game is never stuck behind it), backpressure when full.
    let wait_max = Duration::from_secs(cfg.actors.queue_wait_max_s);
    let queues: Vec<Arc<SegmentQueue>> = (0..3).map(|_| Arc::new(SegmentQueue::with_policy(cfg.actors.queue_batches * cfg.lanes, cfg.actors.queue_full, wait_max))).collect();
    let (tx, rx) = mpsc::channel();
    let routes = Routes::default();
    let builds = T0Builds::default();
    let ends = EndBoard::default();
    let listen = uranium_runner::env_or("URANIUM_AGENT_LISTEN", "127.0.0.1:7600");
    let concede = ConcedeBoard::default();
    let agent_addr = uranium_runner::start_agent(&listen, &args.log_dir.join("sessions.jsonl"), &args.log_dir.join("unit_rejections.jsonl"), routes.clone(), builds.clone(), ends.clone(), concede.clone())?;
    // 042-R2 / R14: a games.csv of an older version (without `end_reason`) is kept aside.
    let games_csv = open_csv(&args.log_dir.join("games.csv"), GAMES_HEADER)?;
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
        snapshots: Mutex::new(SnapshotPool::new(snap_dirs, anchor_key, servers)),
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
        concede_cfg: cfg.concede,
        concede,
        pseudo: cfg.reward.pseudo,
    });
    let z = load_z(&args.z_table)?;
    if z.is_empty() {
        crate::elog!("[rl] no z table: every game runs without z, pseudo-rewards off (033-R8)");
    }
    let hooks: Vec<Arc<Hook>> = AgentKind::ALL
        .iter()
        .map(|&a| Arc::new(Hook { agent: a, handle: handles[a.index()].clone(), seq_len: cfg.seq_len, shared: shared.clone(), tx: tx.clone(), z: z.clone(), z_drop: 0.25, rng: Mutex::new(args.seed ^ (a.index() as u64 + 1)), reward: cfg.reward.clone() }))
        .collect();
    let source = Arc::new(LeagueSource { shared: shared.clone(), hooks });
    let dispatch = Dispatch::new(routes, Default::default(), builds);
    // 042-R3: from here on any return, an error included, stops the games and closes the queues.
    let halt = Halt { shared: shared.clone(), inflight: dispatch.inflight.clone(), done: AtomicBool::new(false) };
    let hosts: Vec<HostEntry> = uranium_runner::hosts()?;

    // Before the first game: a failure here starts nothing.
    let base = RewardWeights::load()?;
    let mut csv = open_train_csv(&args.log_dir.join("train.csv"))?;
    let games_thread = {
        let (source, dispatch, games) = (source.clone(), dispatch.clone(), args.log_dir.join("games.jsonl"));
        std::thread::spawn(move || uranium_runner::run_source(source, &hosts, &games, &dispatch))
    };

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
            // A league snapshot is a pushed checkpoint whose agent model files are copied to
            // league/<name>/agents/<agent> (061-R2; 042 스냅샷 공개, AMB-042-14). Only a published
            // snapshot is an opponent; a failed one is logged and training goes on.
            let id = shared.league.lock().unwrap().add_snapshot(agent);
            let league_now = shared.league.lock().unwrap().clone();
            step += 1;
            match save_checkpoint(&mut run, step, learners.as_mut(), &cfg, &league_now, &args, state_extra(&shared, trained_steps))? {
                PushOutcome::Pushed => match run.league_snapshot(step, &snapshot_name(&id), agent.name()) {
                    Ok(dir) => {
                        shared.snapshots.lock().unwrap().dirs.insert(id.clone(), dir.join(ckpt::AGENTS_DIR).join(agent.name()));
                        shared.league.lock().unwrap().mark_published(&id);
                        crate::elog!("[rl] snapshot {id}");
                    }
                    Err(e) => crate::elog!("[rl] snapshot {id} not published ({e:#}): not an opponent (042)"),
                },
                other => crate::elog!("[rl] snapshot {id} not published: {other:?}"),
            }
            // A snapshot that left the C1 candidates: its server goes once its games are done.
            prune_servers(&shared);
            if reset {
                learners.reset_to_bc(i)?;
                // 042-R4: the reset learner never trains the previous policy's segments.
                queues[i].reset_policy();
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
        let bots = RlConfig::load(Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../config/rl.toml"))).unwrap().t1;
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
        let _ = std::fs::remove_dir_all(&dir);
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

    use crate::learner::SnapshotServer;
    use uranium_agent::infer::{InferOptions, SampleMode};

    fn fake_handle(id: &str) -> InferenceHandle {
        let (tx, _rx) = mpsc::channel();
        InferenceHandle::new(tx, id.into(), SampleMode::Stochastic, &InferOptions::default())
    }

    /// Snapshot servers for tests: `fail` = every start fails; the started folders, the number
    /// of closes and each started server's liveness flag (in start order, set false to "kill"
    /// its model thread) are kept.
    fn fake_servers(fail: bool) -> (SnapshotServers, Arc<Mutex<Vec<PathBuf>>>, Arc<AtomicUsize>) {
        let (s, o, c, _) = fake_servers_with(Arc::new(AtomicBool::new(fail)));
        (s, o, c)
    }

    type Alive = Arc<Mutex<Vec<Arc<AtomicBool>>>>;

    fn fake_servers_with(fail: Arc<AtomicBool>) -> (SnapshotServers, Arc<Mutex<Vec<PathBuf>>>, Arc<AtomicUsize>, Alive) {
        let (opened, closed, alive) = (Arc::new(Mutex::new(Vec::new())), Arc::new(AtomicUsize::new(0)), Alive::default());
        let (o, c, a) = (opened.clone(), closed.clone(), alive.clone());
        let servers: SnapshotServers = Arc::new(move |dir: &Path| {
            anyhow::ensure!(!fail.load(Ordering::SeqCst), "model files unreadable");
            o.lock().unwrap().push(dir.to_path_buf());
            let (c, up) = (c.clone(), Arc::new(AtomicBool::new(true)));
            a.lock().unwrap().push(up.clone());
            let close = Box::new(move || {
                c.fetch_add(1, Ordering::SeqCst);
            });
            Ok(SnapshotServer { handle: fake_handle(&dir.display().to_string()), close, alive: Box::new(move || up.load(Ordering::SeqCst)) })
        });
        (servers, opened, closed, alive)
    }

    /// A league source over a league in C1 with these model shares (no T0 / T1 / current Main
    /// unless given) and fake snapshot servers.
    fn source(c1: [f64; 3], anchor: &str, candidates: usize, servers: SnapshotServers) -> LeagueSource {
        let mut cfg = RlConfig::default_file();
        cfg.curriculum.start_stage = crate::league::Stage::C1;
        let cc = &mut cfg.curriculum;
        (cc.c1_t0, cc.c1_t1, cc.c1_self, cc.c1_snapshot, cc.c1_anchor) = (0.0, 0.0, c1[0], c1[1], c1[2]);
        cfg.league.anchor = anchor.into();
        cfg.snapshot.main_candidates = candidates;
        let csv = PathBuf::from(if cfg!(windows) { "NUL" } else { "/dev/null" });
        source_cfg(cfg, servers, &csv)
    }

    /// A league source over a new league of `cfg`, with fake snapshot servers, writing games.csv
    /// rows to `csv` (an existing file).
    fn source_cfg(cfg: RlConfig, servers: SnapshotServers, csv: &Path) -> LeagueSource {
        let league = League::new("r1", 3, None, cfg.clone());
        let mut dirs = HashMap::new();
        let pinned = league.anchor_key();
        if let Some(k) = &pinned {
            dirs.insert(k.clone(), PathBuf::from("anchor/agents/main"));
        }
        let (tx, rx) = mpsc::channel();
        let shared = Arc::new(Shared {
            league: Mutex::new(league),
            games: Mutex::new(HashMap::new()),
            held: Mutex::new(HashMap::new()),
            results: Mutex::new(rx),
            finished: Mutex::new(HashMap::new()),
            queues: (0..3).map(|_| Arc::new(SegmentQueue::new(4))).collect(),
            ends: EndBoard::default(),
            next_id: AtomicU64::new(1),
            stopping: AtomicBool::new(false),
            agent_addr: "127.0.0.1:1".into(),
            handles: AgentKind::ALL.iter().map(|a| fake_handle(a.name())).collect(),
            snapshots: Mutex::new(SnapshotPool::new(dirs, pinned, servers)),
            games_csv: Mutex::new(std::fs::OpenOptions::new().append(true).open(csv).unwrap()),
            agent_steps: AtomicU64::new(0),
            games_done: AtomicU64::new(0),
            disagree: AtomicU64::new(0),
            aborted: AtomicU64::new(0),
            max_frames: 1,
            game_wall_timeout_s: 1,
            replay_dir: std::env::temp_dir(),
            replay_every: 0,
            concede_cfg: cfg.concede,
            concede: ConcedeBoard::default(),
            pseudo: cfg.reward.pseudo,
        });
        let hooks = AgentKind::ALL
            .iter()
            .map(|&a| Arc::new(Hook { agent: a, handle: fake_handle(a.name()), seq_len: 4, shared: shared.clone(), tx: tx.clone(), z: vec![], z_drop: 0.25, rng: Mutex::new(1), reward: cfg.reward.clone() }))
            .collect();
        LeagueSource { shared, hooks }
    }

    fn opponent_handle(job: &Job) -> String {
        match &job.policies[1] {
            Some(PolicySpec::Model(h)) => h.checkpoint_id.to_string(),
            other => panic!("{other:?}"),
        }
    }

    /// 042 모델 상대 실패 (AMB-042-14): an anchor / snapshot whose server cannot start makes the
    /// game current-Main self-play; the source keeps giving games (a runner slot never ends).
    #[test]
    fn unservable_model_opponent_becomes_self_play_042() {
        let (servers, _, _) = fake_servers(true);
        let src = source([0.0, 0.0, 1.0], "checkpoints/x/ckpt-9", 4, servers);
        for _ in 0..3 {
            let job = src.next().expect("a game, never the end of the slot");
            assert_eq!(job.opponent.as_deref(), Some("uranium-main@r1"));
            assert_eq!(opponent_handle(&job), "main", "the current Main's server");
            let games = src.shared.games.lock().unwrap();
            let m = &games[&job.spec.game_id];
            assert_eq!((&m.pick.opponent, m.pick.class, m.server), (&Opponent::SelfPlay, Class::Model, None));
        }
        // A snapshot that was never published (no model folder): the same.
        let (servers, opened, _) = fake_servers(false);
        let src = source([0.0, 1.0, 0.0], "", 4, servers);
        {
            let mut l = src.shared.league.lock().unwrap();
            let id = l.add_snapshot(AgentKind::Main);
            l.mark_published(&id);
        }
        let job = src.next().unwrap();
        assert_eq!(job.opponent.as_deref(), Some("uranium-main@r1"));
        assert!(opened.lock().unwrap().is_empty());
    }

    /// 042 스냅샷 추론 서버 (AMB-042-14): a snapshot's server serves its games, goes down once it
    /// left the candidates and its games are done; the anchor's stays.
    #[test]
    fn snapshot_servers_shut_down_when_unused_042() {
        let (servers, opened, closed) = fake_servers(false);
        let src = source([0.0, 1.0, 0.0], "checkpoints/x/ckpt-9", 1, servers);
        let publish = |n: u64| {
            let mut l = src.shared.league.lock().unwrap();
            for _ in 0..n {
                l.on_update(AgentKind::Main);
            }
            let id = l.add_snapshot(AgentKind::Main);
            l.mark_published(&id);
            src.shared.snapshots.lock().unwrap().dirs.insert(id.clone(), PathBuf::from(format!("league/{n}/agents/main")));
            id
        };
        // No snapshot yet: the snapshot share goes to the anchor.
        let a = src.next().unwrap();
        assert_eq!(a.opponent.as_deref(), Some("anchor@ckpt-9"));
        assert_eq!(opponent_handle(&a), "anchor/agents/main");
        let first = publish(250);
        let j1 = src.next().unwrap();
        assert_eq!(j1.opponent.as_deref(), Some(first.as_str()));
        let second = publish(500);
        prune_servers(&src.shared);
        assert_eq!(closed.load(Ordering::SeqCst), 0, "the first snapshot still has a game");
        let j2 = src.next().unwrap();
        assert_eq!(j2.opponent.as_deref(), Some(second.as_str()));
        src.requeue(j1);
        assert_eq!(closed.load(Ordering::SeqCst), 1, "first snapshot's server down after its game");
        src.requeue(a);
        src.requeue(j2);
        assert_eq!(closed.load(Ordering::SeqCst), 1, "the candidate's and the pinned anchor's servers stay");
        let keys = src.shared.snapshots.lock().unwrap().live_keys().iter().map(|k| k.to_string()).collect::<Vec<_>>();
        assert_eq!(keys, ["anchor@ckpt-9".to_string(), second.clone()]);
        assert_eq!(opened.lock().unwrap().len(), 3);
        // The same snapshot again: no new server.
        let j3 = src.next().unwrap();
        assert_eq!(j3.opponent.as_deref(), Some(second.as_str()));
        assert_eq!(opened.lock().unwrap().len(), 3);
    }

    /// The idle servers above the cap go least recently used first; busy and pinned ones stay.
    #[test]
    fn snapshot_pool_cap_spares_busy_and_pinned_servers() {
        let (servers, _, closed) = fake_servers(false);
        let dirs = (0..4).map(|i| (format!("s{i}"), PathBuf::from(format!("d{i}")))).collect();
        let mut pool = SnapshotPool::new(dirs, Some("s0".into()), servers);
        pool.cap = 2;
        let ids: Vec<u64> = (0..4).map(|i| pool.acquire(&format!("s{i}")).unwrap().1).collect();
        pool.release(ids[0]);
        pool.release(ids[1]);
        let all: BTreeSet<String> = (0..4).map(|i| format!("s{i}")).collect();
        let out = pool.prune(&all);
        assert_eq!(out.iter().map(|(k, _)| k.as_str()).collect::<Vec<_>>(), ["s1"]);
        for (_, c) in out {
            c();
        }
        assert_eq!(closed.load(Ordering::SeqCst), 1);
        assert_eq!(pool.live_keys(), ["s0", "s2", "s3"], "busy ones stay above the cap");
        pool.release(ids[2]);
        assert_eq!(pool.prune(&BTreeSet::new()).iter().map(|(k, _)| k.as_str()).collect::<Vec<_>>(), ["s2"], "unused and idle");
        assert!(pool.acquire("nope").is_err());
    }

    /// A server whose model thread died after its start returned (e.g. a failed warm-up) is
    /// never handed out again: the pick that finds it closes it and becomes current-Main
    /// self-play, as do the picks of that key for `retry_after`; then the key starts afresh. A
    /// game of the dead server releasing later never touches the new one. Failed starts are not
    /// retried for `retry_after` either.
    #[test]
    fn dead_and_failing_servers_are_not_reused_042() {
        let fail = Arc::new(AtomicBool::new(false));
        let (servers, opened, closed, alive) = fake_servers_with(fail.clone());
        // The anchor only (pinned, never pruned).
        let src = source([0.0, 0.0, 1.0], "checkpoints/x/ckpt-9", 4, servers);
        let first = src.next().unwrap();
        assert_eq!(first.opponent.as_deref(), Some("anchor@ckpt-9"));
        alive.lock().unwrap()[0].store(false, Ordering::SeqCst);
        for _ in 0..3 {
            let j = src.next().unwrap();
            assert_eq!(j.opponent.as_deref(), Some("uranium-main@r1"), "dead anchor server: self-play");
        }
        assert_eq!((closed.load(Ordering::SeqCst), opened.lock().unwrap().len()), (1, 1), "closed once, not restarted at once");
        // After the back-off: a new server.
        src.shared.snapshots.lock().unwrap().retry_after = Duration::ZERO;
        let again = src.next().unwrap();
        assert_eq!(again.opponent.as_deref(), Some("anchor@ckpt-9"));
        assert_eq!(opened.lock().unwrap().len(), 2);
        let new_id = src.shared.games.lock().unwrap()[&again.spec.game_id].server.unwrap();
        // The dead server's game ends: the new server's count is untouched.
        src.requeue(first);
        assert_eq!(src.shared.snapshots.lock().unwrap().live.iter().find(|l| l.id == new_id).unwrap().games, 1);
        // A start that fails is not retried at every pick.
        let mut pool = src.shared.snapshots.lock().unwrap();
        pool.retry_after = Duration::from_secs(300);
        pool.live.clear();
        fail.store(true, Ordering::SeqCst);
        assert!(pool.acquire("anchor@ckpt-9").is_err());
        fail.store(false, Ordering::SeqCst);
        let err = pool.acquire("anchor@ckpt-9").unwrap_err();
        assert!(format!("{err:#}").contains("failed"), "{err:#}");
        assert_eq!(opened.lock().unwrap().len(), 2, "no reload within the back-off");
    }

    /// 042-R14 / 020-R18 (AMB-042-15) in the run loop: with `[concede] enabled` every RL game is
    /// on the concede board from dispatch until it is done; a conceded game is judged as a normal
    /// result (the leaver loses, not Aborted) and games.csv says `Concede`; other games give the
    /// judged GameEnd reason, games without a result an empty `end_reason`. Off = no board entry.
    #[test]
    fn concede_games_in_the_run_loop_042_r14() {
        use uranium_proto::EndReason;
        let dir = std::env::temp_dir().join(format!("uranium-rl-concede-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let csv = dir.join("games.csv");
        drop(open_csv(&csv, GAMES_HEADER).unwrap());
        let mut cfg = RlConfig::default_file();
        cfg.curriculum.start_stage = crate::league::Stage::C1;
        let cc = &mut cfg.curriculum;
        (cc.c1_t0, cc.c1_t1, cc.c1_self) = (0.0, 0.0, 1.0);
        cfg.concede.enabled = true;
        let (servers, _, _) = fake_servers(false);
        let src = source_cfg(cfg.clone(), servers, &csv);
        // Plays one game: the learner (players index / BWAPI id 0) against the current Main (1).
        let play = |ends: &[(u8, u8, EndReason)], conceded: Option<u8>, outcome: JobOutcome| {
            let job = src.next().unwrap();
            let id = job.spec.game_id;
            let game = src.shared.concede.get(id).expect("on the board from dispatch");
            if let Some(side) = conceded {
                assert!(game.claim(side, 7_200, LeaveRule::Concede));
            }
            src.shared.games.lock().unwrap().get_mut(&id).unwrap().self_player = Some(0);
            for &(slot, winner, reason) in ends {
                src.shared.ends.push(id, uranium_agent::SessionEnd { slot, winner: Some(winner), frame: 7_210 + slot as u32, reason });
            }
            let result = match ends.iter().find(|e| e.0 == 0) {
                Some(&(_, w, _)) if outcome == JobOutcome::Finished => Some(if w == 0 { Outcome::Win } else { Outcome::Loss }),
                _ => None,
            };
            src.hooks[0].tx.send(GameResult { agent: 0, game_id: id, outcome: result, reward_sum: [0.0; 4], scout_sum: 0.0, survive_sum: 0.0, scout: None, decisions: 5, enemy_race: Some(Race::Terran) }).unwrap();
            src.done(&job, outcome, None);
            assert!(src.shared.concede.get(id).is_none(), "off the board once done");
            std::fs::read_to_string(&csv).unwrap().lines().last().unwrap().to_string()
        };
        let row = |line: &str| line.split(',').map(str::to_string).collect::<Vec<_>>();
        // The learner conceded: it left (names the other side), the opponent model reports its win.
        let r = row(&play(&[(0, 1, EndReason::Surrender), (1, 1, EndReason::Surrender)], Some(0), JobOutcome::Finished));
        assert_eq!((r[7].as_str(), r[10].as_str(), r[16].as_str()), ("Some(Loss)", "-1", "Concede"), "{r:?}");
        // The opponent model conceded: a win of the learning agent.
        let r = row(&play(&[(1, 0, EndReason::Surrender), (0, 0, EndReason::Surrender)], Some(1), JobOutcome::Finished));
        assert_eq!((r[7].as_str(), r[10].as_str(), r[16].as_str()), ("Some(Win)", "1", "Concede"), "{r:?}");
        // No concede: the judged reason.
        let r = row(&play(&[(0, 0, EndReason::Elimination), (1, 0, EndReason::Elimination)], None, JobOutcome::Finished));
        assert_eq!((r[7].as_str(), r[16].as_str()), ("Some(Win)", "Elimination"));
        // Aborted: no result, empty end_reason (also after a concede).
        let r = row(&play(&[(0, 1, EndReason::Surrender)], Some(0), JobOutcome::Aborted));
        assert_eq!((r[7].as_str(), r[16].as_str()), ("None", ""));
        assert_eq!(r.len(), GAMES_HEADER.trim().split(',').count());
        // Requeued games leave the board too; concede off = no board entry.
        let job = src.next().unwrap();
        let id = job.spec.game_id;
        src.requeue(job);
        assert!(src.shared.concede.get(id).is_none());
        cfg.concede.enabled = false;
        let (servers, _, _) = fake_servers(false);
        let off = source_cfg(cfg, servers, &csv);
        let job = off.next().unwrap();
        assert!(off.shared.concede.get(job.spec.game_id).is_none());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// 042-R2 / R14: a games.csv without `end_reason` (an older version) is kept aside like an
    /// older train.csv; the new file gets the current header.
    #[test]
    fn games_csv_with_other_columns_is_kept_aside_042_r14() {
        let dir = std::env::temp_dir().join(format!("uranium-rl-games-csv-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("games.csv");
        let old = "game_id,agent,opponent,class,enemy_race,map,hostd_outcome,outcome,winner_disagree,decisions,r_win,r_build,r_units,r_value,replay,replay_reason\n1,main\n";
        std::fs::write(&path, old).unwrap();
        drop(open_csv(&path, GAMES_HEADER).unwrap());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), GAMES_HEADER);
        let kept: Vec<_> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().file_name().into_string().unwrap()).filter(|n| n.starts_with("games-")).collect();
        assert_eq!(kept.len(), 1);
        assert_eq!(std::fs::read_to_string(dir.join(&kept[0])).unwrap(), old);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn games_csv_header_matches_rows_042_r2() {
        assert_eq!(GAMES_HEADER.trim().split(',').count(), 26);
        // 042-R2 (AMB-042-16): the new columns follow end_reason in this order.
        assert!(GAMES_HEADER.trim().ends_with(",replay,replay_reason,end_reason,r_scout,r_survive,ladder_group,scout_main_s,scout_types_4m,scout_types_6m,barracks_s,bunker_s,scv_4m"));
    }

    fn result(id: u64, scout: Option<ScoutMetrics>) -> GameResult {
        GameResult { agent: 0, game_id: id, outcome: Some(Outcome::Win), reward_sum: [0.0, 0.0, 0.0, 0.25], scout_sum: 0.6, survive_sum: 0.375, scout, decisions: 9, enemy_race: Some(Race::Zerg) }
    }

    /// 042-R2 / 042-AC6 (AMB-042-16): the columns after `end_reason` — raw sums only in "scout",
    /// the ladder group of a ladder game, the metrics in game seconds, empty when not reached or
    /// without a session report.
    #[test]
    fn scout_columns_042_r2() {
        let m = ScoutMetrics { main_frame: Some(3_000), types_4m: 2, types_6m: 5, barracks_frame: Some(2_412), bunker_frame: None, scv_4m: Some(14) };
        let r = result(1, Some(m));
        assert_eq!(scout_columns(PseudoMode::Scout, Some(&r), Some("macro")), "0.6,0.375,macro,125.0,2,5,100.5,,14");
        assert_eq!(scout_columns(PseudoMode::Z, Some(&r), Some("rush")), ",,rush,125.0,2,5,100.5,,14", "z: r_scout / r_survive empty");
        assert_eq!(scout_columns(PseudoMode::Scout, Some(&result(2, Some(ScoutMetrics::default()))), None), "0.6,0.375,,,0,0,,,", "nothing reached");
        assert_eq!(scout_columns(PseudoMode::Scout, None, None), ",,,,,,,,", "no session report");
        assert_eq!(scout_columns(PseudoMode::Z, None, None).split(',').count(), 9);
    }

    /// 033-R11 start errors: "scout" with a z table; a resumed checkpoint of the other mode (a
    /// checkpoint without the key was "z"); `--init` / new runs are not checked.
    #[test]
    fn pseudo_start_errors_033_r11() {
        let mut cfg = RlConfig::default_file();
        let z = Path::new("z.jsonl");
        check_pseudo(&cfg, Some(z), None).unwrap();
        check_pseudo(&cfg, None, Some(PseudoMode::Z)).unwrap();
        assert!(check_pseudo(&cfg, None, Some(PseudoMode::Scout)).is_err());
        cfg.reward.pseudo = PseudoMode::Scout;
        assert!(check_pseudo(&cfg, Some(z), None).is_err(), "scout + --z-table");
        check_pseudo(&cfg, None, None).unwrap();
        check_pseudo(&cfg, None, Some(PseudoMode::Scout)).unwrap();
        assert!(check_pseudo(&cfg, None, Some(PseudoMode::Z)).is_err(), "resume a z checkpoint in scout");
        // The checkpoint's mode from agents/main/meta.json train.rl.
        let dir = std::env::temp_dir().join(format!("uranium-rl-pseudo-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let main = dir.join(crate::ckpt::AGENTS_DIR).join("main");
        std::fs::create_dir_all(&main).unwrap();
        assert_eq!(checkpoint_pseudo(&dir), None, "no meta.json");
        let write = |rl: serde_json::Value| std::fs::write(main.join("meta.json"), serde_json::json!({ "train": { "rl": rl, "update": 3 } }).to_string()).unwrap();
        let mut old = serde_json::to_value(RlConfig::default_file()).unwrap();
        old.as_object_mut().unwrap().remove("reward");
        write(old);
        assert_eq!(checkpoint_pseudo(&dir), Some(PseudoMode::Z), "older checkpoint without [reward]");
        write(serde_json::to_value(&cfg).unwrap());
        assert_eq!(checkpoint_pseudo(&dir), Some(PseudoMode::Scout));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// 033-R11 / 042 (AMB-042-16): a session's observer gets the mode and its game's multipliers
    /// (model opponent: `vs_model` of each reward; bot: 1 and 1) and the hook hands the reward
    /// settings to the session; games.csv rows carry the ladder group and the metrics.
    #[test]
    fn sessions_and_games_csv_in_scout_mode_042_amb_042_16() {
        use crate::league::{LadderEntry, LadderKind};
        use uranium_agent::T0Build;
        use uranium_env::features::executor::HostAction;
        use uranium_proto::EndReason;
        let dir = std::env::temp_dir().join(format!("uranium-rl-scout-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let csv = dir.join("games.csv");
        drop(open_csv(&csv, GAMES_HEADER).unwrap());
        let mut cfg = RlConfig::default_file();
        cfg.curriculum.start_stage = crate::league::Stage::C1;
        let cc = &mut cfg.curriculum;
        (cc.c1_t0, cc.c1_t1, cc.c1_self, cc.c1_ladder) = (0.0, 0.0, 0.5, 0.5);
        cc.ladder_groups = [("rush".to_string(), 0.5), ("macro".to_string(), 0.5)].into_iter().collect();
        let entry = |build, group: &str| LadderEntry { kind: LadderKind::T0, build: Some(build), name: None, race: Race::Zerg, group: group.into() };
        cfg.ladder = vec![entry(T0Build::ZergRush, "rush"), entry(T0Build::Standard, "macro")];
        cfg.reward.pseudo = PseudoMode::Scout;
        cfg.validate().unwrap();
        let (servers, _, _) = fake_servers(false);
        let src = source_cfg(cfg, servers, &csv);
        assert_eq!(src.hooks[0].reward_config().pseudo, PseudoMode::Scout);
        let info = crate::testkit::game_info();
        let obs = crate::testkit::observations(2);
        let (mut model, mut bot) = (0, 0);
        for _ in 0..40 {
            let job = src.next().unwrap();
            let id = job.spec.game_id;
            let (class, group) = {
                let g = src.shared.games.lock().unwrap();
                let p = &g.get(&id).unwrap().pick;
                (p.class, p.ladder_group.clone())
            };
            let (z, mut o) = src.hooks[0].session(id, &info);
            assert!(z.is_none());
            let a = HostAction { action: 0, delay_bucket: 0, queued: false, selected: vec![], target_unit: None, target_loc: None };
            for (i, ob) in obs.iter().enumerate() {
                let state = o.wants_state().then(uranium_agent::infer::HostState::default);
                o.on_decision(&uranium_agent::model_policy::Decision { frame: i as u32, obs: ob, action: &a, log_prob: 0.0, head_log_prob: [0.0; uranium_agent::infer::HEADS], value: [0.0; 4], rejected: false, version: 0, reward_before: Default::default(), state });
            }
            o.on_end(Default::default(), Some(Outcome::Win));
            let seg = src.shared.held.lock().unwrap().get(&id).map(|s| s.pseudo_scale).expect("held terminal segment");
            // The session's own report is replaced by one with metrics for the row below.
            src.shared.results.lock().unwrap().try_iter().for_each(drop);
            if class == Class::Model {
                assert_eq!(seg, [0.5, 0.0], "model game: vs_model");
                model += 1;
            } else {
                assert_eq!(seg, [1.0, 1.0], "bot game");
                bot += 1;
            }
            src.shared.games.lock().unwrap().get_mut(&id).unwrap().self_player = Some(0);
            for slot in [0u8, 1] {
                src.shared.ends.push(id, uranium_agent::SessionEnd { slot, winner: Some(0), frame: 9_000, reason: EndReason::Elimination });
            }
            src.hooks[0].tx.send(result(id, Some(ScoutMetrics { main_frame: Some(2_400), types_4m: 1, types_6m: 3, ..Default::default() }))).unwrap();
            src.done(&job, JobOutcome::Finished, None);
            let line = std::fs::read_to_string(&csv).unwrap().lines().last().unwrap().to_string();
            let r: Vec<&str> = line.split(',').collect();
            assert_eq!(r.len(), GAMES_HEADER.trim().split(',').count(), "{line}");
            assert_eq!(&r[17..], [&["0.6", "0.375", group.as_deref().unwrap_or("")][..], &["100.0", "1", "3", "", "", ""]].concat(), "{line}");
            if let Some(g) = &group {
                assert_eq!(r[2].starts_with("t0-zerg-rush"), g == "rush", "{line}");
            }
        }
        assert!(model > 0 && bot > 0, "{model} {bot}");
        std::fs::remove_dir_all(&dir).ok();
    }
}
