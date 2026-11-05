//! The RL run loop (042): one process runs the learner, the inference servers, the agent server,
//! the game dispatch (`uranium-runner`, 020-R12) and the league. The learner and the servers are
//! the PyTorch side (`python -m uranium rl`, AMB-000-4) behind [`RlLearners`].
//!
//! Game outcomes follow 020-R13: once hostd reports a game over, the GameEnd reports of its
//! sessions are reconciled (`game_result`) and that verdict sets r_win on the learner's terminal
//! segment (033) and feeds the league win table (042); a rush mini-game that reached its frame
//! cap is judged by the learner's SCVs first (042-R17). Aborted games are revoked (042-R3).
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
use crate::config::{LeagueConfig, RlConfig};
use crate::league::{AgentKind, Class, League, Opponent, Pick};
use crate::learner::{Algo, HEAD_NAMES, RlLearners, SnapshotServers, UpdateStats};
use crate::sil::{SilBuffer, SilConfig};
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
    /// 020-R21 start slots of `players[0]` / `players[1]` from the game's seed (`games.csv`
    /// `slot_self`, `slot_enemy`, AMB-042-19); `None` without a seed.
    slots: Option<[u8; 2]>,
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
    /// 042-R20 (AMB-042-20): finished games in a row whose learning session made no decision
    /// ([`bad_learning_session`]).
    bad_sessions: AtomicU64,
    /// 042-R20: games that ended `Finished` (the stall check counts these, not Aborted /
    /// Rejected games of a game host outage, which leave the learner without data anyway).
    finished_games: AtomicU64,
    /// `[league]` of the config: `GameSpec.max_frames` by opponent (020-R3 60 minutes, 042-R15
    /// `model_game_max_s`, AMB-042-17; 042-R17 rush mini-games, AMB-042-18) and the
    /// `[league.rush_judge]` settings.
    league_cfg: LeagueConfig,
    /// 042-R16 (AMB-042-17): `Some` when `[sil] enabled`.
    sil: Option<SilState>,
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

/// 042-R16 self-imitation state of a run (AMB-042-17): the copies of the candidate games still
/// playing and the buffer of the selected ones (not in checkpoints).
struct SilState {
    cfg: SilConfig,
    pending: Mutex<HashMap<u64, Vec<Segment>>>,
    buffer: Mutex<SilBuffer>,
    /// The "not in C1" notice was logged.
    off_stage_logged: AtomicBool,
}

impl SilState {
    fn new(cfg: SilConfig, seed: u64) -> Self {
        let buffer = Mutex::new(SilBuffer::new(cfg.buffer_segments, seed ^ 0x5115_5115));
        Self { cfg, pending: Mutex::new(HashMap::new()), buffer, off_stage_logged: AtomicBool::new(false) }
    }

    /// 042-R16 "Main만(C1)": buffer lanes only in C1; another stage is logged once.
    fn in_stage(&self, stage: crate::league::Stage) -> bool {
        if stage == crate::league::Stage::C1 {
            return true;
        }
        if !self.off_stage_logged.swap(true, Ordering::SeqCst) {
            crate::elog!("[rl] [sil] is on but the stage is {stage:?}: no self-imitation lanes outside C1 (042-R16)");
        }
        false
    }

    /// A copy of a candidate game's segment, kept when it starts before `until_s`.
    fn keep(&self, seg: &Segment) {
        if self.cfg.keeps_segment(seg.start_frame) {
            self.pending.lock().unwrap().entry(seg.game_id).or_default().push(seg.clone());
        }
    }

    /// The game is over: its copies go to the buffer when it is selected (with its r_win), else
    /// they are dropped. Returns the number of segments added.
    fn settle(&self, game_id: u64, selected: bool, r_win: f32) -> usize {
        let segs = self.pending.lock().unwrap().remove(&game_id).unwrap_or_default();
        if !selected || segs.is_empty() {
            return 0;
        }
        let n = segs.len();
        self.buffer.lock().unwrap().push_game(segs, r_win);
        n
    }
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
        let mut ladder_group = None;
        let mut key = String::new();
        let mut start_hit = None;
        if let Some(m) = self.shared.games.lock().unwrap().get_mut(&game_id) {
            m.self_player = Some(info.self_player);
            class = Some(m.pick.class);
            ladder_group = m.pick.ladder_group.clone();
            key = m.pick.key.clone();
            start_hit = slots_start_hit(m.slots);
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
        let mut obs = TrajectoryObserver::new(self.agent.index(), game_id, self.seq_len, self.shared.queues[self.agent.index()].clone(), self.tx.clone()).hold_terminal(held_sink(self.shared.clone())).r_value_policy(r_value_policy).pseudo(self.reward.pseudo, pseudo_scale);
        // 042-R16 (AMB-042-17): Main's games of the selected ladder group keep segment copies
        // until the game is judged; with `select_keys` / `select_start_hit` (AMB-042-19) only
        // games against a listed opponent with the wanted start condition (the others can never
        // be selected).
        if let Some(sil) = &self.shared.sil
            && self.agent == AgentKind::Main
            && ladder_group.as_deref() == Some(sil.cfg.select_group.as_str())
            && sil.cfg.may_select(&key, start_hit)
        {
            let s = self.shared.clone();
            obs = obs.copy_segments(Box::new(move |seg: &Segment| {
                // Not after the game was settled (an Aborted game's session may still push): the
                // `games` guard is held across the check and the keep, and `done` / `requeue`
                // remove the game from `games` before they settle, so no copy can recreate a
                // settled game's pending list. Lock order: `games`, then the SIL lists.
                let games = s.games.lock().unwrap();
                if games.contains_key(&seg.game_id)
                    && let Some(sil) = &s.sil
                {
                    sil.keep(seg);
                }
                drop(games);
            }));
        }
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

/// The seed of RL game `id` (no RNG state: the same id gives the same seed, resumes included).
fn base_seed(id: u64) -> u32 {
    (splitmix(&mut (id ^ 0xA5A5_5A5A)) * u32::MAX as f64) as u32
}

/// 042-R18 candidates beyond `s_0`: up to 64 seeds of a game.
const START_MIX_TRIES: u64 = 64;

/// 042-R18 시작 조건 섞기 (AMB-042-19): the `GameSpec.seed` of RL game `id`. Without a share (not
/// a `[league.start_mix]` group's ladder game) it is the earlier hash seed `s_0`. With one, the
/// game's target condition is a first-target hit when a uniform value of the id is below the
/// share; the first of `s_0`, `s_1`..`s_63` (hashes of the id and k) whose 020-R21 hit (ours
/// `players[0]`, the enemy `players[1]`, `random_players` Random draws first) is the target is
/// the seed, else `s_0` with a log line (about (2/3)^64, 메인 임시 결정). Hash constants:
/// implementation choice.
fn game_seed(id: u64, share: Option<f64>, random_players: u32) -> u32 {
    let Some(share) = share else { return base_seed(id) };
    mixed_seed(id, share, |s| {
        let [a, b] = uranium_runner::start_slots::start_slots(s, random_players);
        uranium_runner::start_slots::start_hit(a, b)
    })
}

/// 042-R18: the seed search of [`game_seed`] with the hit of a seed given by `hit`.
fn mixed_seed(id: u64, share: f64, hit: impl Fn(u32) -> bool) -> u32 {
    let s0 = base_seed(id);
    let target = splitmix(&mut (id ^ 0x5EED_0042_0018_0000)) < share;
    let candidate = |k: u64| if k == 0 { s0 } else { (splitmix(&mut (id.wrapping_mul(START_MIX_TRIES) ^ k ^ 0x51A7_1E5E_ED00_0000)) * u32::MAX as f64) as u32 };
    if let Some(s) = (0..START_MIX_TRIES).map(candidate).find(|&s| hit(s) == target) {
        return s;
    }
    crate::elog!("[rl] game {id}: 042-R18 none of {START_MIX_TRIES} seeds gives the target start condition (first-target hit = {target}); keeping the game's seed {s0}");
    s0
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
        let players = [bridge(Race::Terran), p1];
        // 020-R21: the Random race draws come before the slot shuffle.
        let random_players = players.iter().filter(|p| p.race == Race::Random).count() as u32;
        let share = if pick.class == Class::Model { None } else { s.league_cfg.start_mix.share_for(pick.ladder_group.as_deref()) };
        let seed = game_seed(id, share, random_players);
        let slots = uranium_runner::start_slots::start_slots(seed, random_players);
        let spec = GameSpec {
            game_id: id,
            map: if pick.fighting_spirit { MapKey::FightingSpirit } else { MapKey::CircuitBreaker },
            players,
            seed: Some(seed),
            // 042-R15 / 020-R3 (AMB-042-17): a game against a model (a failed model opponent's
            // current-Main self-play included) has the model-game cap; 042-R17 (AMB-042-18): a
            // ladder game of a `[league.rush_judge]` group ends at `at_s`; other bots the 60 minutes.
            max_frames: s.league_cfg.game_max_frames_of(pick.class == Class::Model, pick.ladder_group.as_deref()),
            privileged: true,
            // 042-R11: every game is recorded; `done` keeps the sample and disagreements.
            save_replay: true,
            // AMB-042-12 / 020-R3: queue waits count too; Timeout is left out like Aborted.
            wall_timeout_s: s.game_wall_timeout_s,
        };
        s.games.lock().unwrap().insert(id, GameMeta { pick, self_player: None, enemy_race, server, slots: Some(slots) });
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
        if let Some(sil) = &self.shared.sil {
            sil.settle(job.spec.game_id, false, 0.0);
        }
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
        // 042-R20 (AMB-042-20): the run loop's stall count and dead-inference streak.
        if outcome == JobOutcome::Finished {
            s.finished_games.fetch_add(1, Ordering::SeqCst);
            if result.is_none() {
                crate::elog!("[rl] game {id}: finished without a report of the learning session; not counted for the no-decision streak (042-R20)");
            }
        }
        match bad_learning_session(outcome, result.as_ref()) {
            Some(true) => {
                s.bad_sessions.fetch_add(1, Ordering::SeqCst);
            }
            Some(false) => s.bad_sessions.store(0, Ordering::SeqCst),
            None => {}
        }
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
        // 042-R17 / 020-R19 (AMB-042-18): a rush mini-game that ended at its frame cap
        // (`TimeLimitDraw` after 020-R13) is a win with at least `min_scv` own SCVs at the
        // learning session's last observation, else a loss; with no observation it has no result
        // (left out of training like an Aborted game, 042-R3). Earlier ends keep their outcome.
        let timed_out = verdict.as_ref().is_some_and(|(e, _)| e.reason == EndReason::TimeLimitDraw);
        let mut judge_scv = None;
        let decided = match decided {
            Some(o) if timed_out && s.league_cfg.rush_judged(meta.pick.class == Class::Model, meta.pick.ladder_group.as_deref()) => {
                let judge = &s.league_cfg.rush_judge;
                match result.as_ref().map(|r| r.scout.and_then(|m| m.scv_last)) {
                    Some(Some(n)) => {
                        let judged = judge.judge(n);
                        judge_scv = Some(n);
                        crate::elog!("[rl] game {id}: 042-R17 rush judge at {} s: {n} own SCVs (win at >= {}) -> {judged:?} (bridge {o:?})", judge.at_s, judge.min_scv);
                        Some(judged)
                    }
                    // 메인 임시 결정 (2026-11-04): a missing session report counts like no observation.
                    Some(None) => {
                        crate::elog!("[rl] game {id}: 042-R17 rush judge at {} s: no observation of the learning session to count SCVs; no result, not trained (042-R3)", judge.at_s);
                        None
                    }
                    None => {
                        crate::elog!("[rl] game {id}: 042-R17 rush judge at {} s: no session report of the learning session (timeout); no result, not trained (042-R3)", judge.at_s);
                        None
                    }
                }
            }
            other => other,
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
                // 042-R16: the terminal segment's copy (with r_win), for a candidate game.
                if let Some(sil) = &s.sil
                    && meta.pick.agent == AgentKind::Main
                    && meta.pick.ladder_group.as_deref() == Some(sil.cfg.select_group.as_str())
                    && sil.cfg.may_select(&meta.pick.key, slots_start_hit(meta.slots))
                {
                    sil.keep(&seg);
                }
                s.queues[agent].push(seg);
                s.queues[agent].finish(id);
                let stage_up = s.league.lock().unwrap().record(&meta.pick, meta.enemy_race, o);
                if stage_up {
                    stage_advanced(s);
                }
            }
            (Some(o), None) => {
                // No terminal segment (e.g. no decision): still a game for the win table.
                s.queues[agent].finish(id);
                if s.league.lock().unwrap().record(&meta.pick, meta.enemy_race, o) {
                    stage_advanced(s);
                }
            }
            (None, _) => {
                // 042-R3: Aborted / stopped / undecidable: nothing of it is trained.
                s.queues[agent].revoke(id);
                s.aborted.fetch_add(1, Ordering::SeqCst);
            }
        }
        // 042-R14 games.csv `end_reason`: `Concede` for a collapse concede, else the judged
        // GameEnd reason (020-R13); empty for a game without a result (Aborted, Timeout).
        // 042-R17 (AMB-042-18): `RushJudge` for a judged rush mini-game, also when a collapse
        // concede coincided with the frame cap (메인 임시 결정, 2026-11-04).
        let end_reason = match (&verdict, decided) {
            (_, None) => String::new(),
            (_, Some(_)) if judge_scv.is_some() => "RushJudge".to_string(),
            (_, Some(_)) if conceded.is_some() => "Concede".to_string(),
            (Some((e, _)), Some(_)) => format!("{:?}", e.reason),
            (None, Some(_)) => String::new(),
        };
        if let Some(c) = conceded {
            let who = if c.side == 0 { "learning agent" } else { "opponent" };
            crate::elog!("[rl] game {id}: 042-R14 collapse concede by the {who} side (players index {}) at frame {}; result {:?}", c.side, c.frame, decided);
        }
        // 042-R16 고를 판 (AMB-042-17): a won or held game of the group goes to the buffer.
        // 042-R16 상대·접촉 거르기 (AMB-042-19): and its opponent key / first army sighting / 020-R21
        // start hit pass `select_keys` / `select_army_before_s` / `select_start_hit`.
        if let Some(sil) = &s.sil {
            let scout = result.as_ref().and_then(|r| r.scout);
            let scv_4m = scout.and_then(|m| m.scv_4m);
            let army_s = scout.and_then(|m| m.army_frame).map(seconds);
            let selected = meta.pick.agent == AgentKind::Main && sil.cfg.selects(meta.pick.ladder_group.as_deref(), decided, scv_4m, conceded.is_some()) && sil.cfg.passes_filters(&meta.pick.key, army_s, slots_start_hit(meta.slots));
            sil.settle(id, selected, r_win);
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
        let start = start_columns(meta.slots, result.as_ref());
        let line = format!(
            "{},{},{},{},{:?},{},{},{:?},{},{},{},{},{},{},{},{},{},{},{},{}\n",
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
            extra,
            // 042-R2 / R17 (AMB-042-18): the judged SCV count, empty for games not judged.
            judge_scv.map_or(String::new(), |n| n.to_string()),
            // 042-R2 시작 슬롯·접촉 지표 (AMB-042-19).
            start
        );
        let _ = s.games_csv.lock().unwrap().write_all(line.as_bytes());
        self.release_server(&meta);
    }
}

/// 042-R20 (AMB-042-20): `Finished` games since the last learner update before a missing update
/// counts as a dead learner.
const DEAD_LEARNER_GAMES: u64 = 20;
/// 042-R20: finished games in a row whose learning session made no decision = dead inference.
const BAD_SESSION_STREAK: u64 = 10;

/// 042-R20: a finished game whose learning session reported no decision (inference failed for
/// every request) is bad (`Some(true)`), one with decisions is good; a game that did not finish
/// (Aborted, Timeout, Stopped, Rejected) or whose learning session sent no report (the caller
/// logs a warning) does not count either way.
fn bad_learning_session(outcome: JobOutcome, result: Option<&GameResult>) -> Option<bool> {
    match (outcome, result) {
        (JobOutcome::Finished, Some(r)) => Some(r.decisions == 0),
        _ => None,
    }
}

/// 042-R20 (AMB-042-20): the run loop's stall check (pure; the caller passes the clock). In the
/// 6th advisory's CUDA fault the inference failed for every game, games kept finishing with no
/// decision and the learner waited for a batch for 40 minutes: an abnormal end lets the 060-R6
/// watchdog restart the run from LATEST.
struct StallWatch {
    /// `[actors] dead_learner_minutes`; zero = off.
    after: Duration,
    last: Instant,
    games_at_last: u64,
}

impl StallWatch {
    fn new(minutes: u64, now: Instant, games: u64) -> Self {
        Self { after: Duration::from_secs(minutes * 60), last: now, games_at_last: games }
    }

    /// A learner update, or the end of the loop's own long work (checkpoint, snapshot).
    fn progress(&mut self, now: Instant, games: u64) {
        (self.last, self.games_at_last) = (now, games);
    }

    /// Why the run should end abnormally, if it should: `games` = `Finished` games in total,
    /// `bad_streak` = finished games in a row without a decision of the learning session.
    fn check(&self, now: Instant, games: u64, bad_streak: u64) -> Option<String> {
        if self.after.is_zero() {
            return None;
        }
        let since = now.saturating_duration_since(self.last);
        let finished = games.saturating_sub(self.games_at_last);
        if finished >= DEAD_LEARNER_GAMES && since >= self.after {
            return Some(format!("{finished} games finished but no learner update for {} min (dead learner)", since.as_secs() / 60));
        }
        if bad_streak >= BAD_SESSION_STREAK {
            return Some(format!("{bad_streak} finished games in a row without a decision of the learning session (inference failing)"));
        }
        None
    }
}

/// 042-R5 stage change log; 033-R11: the "scout" C2 warning at the C1 → C2 transition too.
fn stage_advanced(s: &Shared) {
    let stage = s.league.lock().unwrap().stage;
    crate::elog!("[rl] curriculum advanced to {stage:?} (042-R5)");
    if let Some(w) = scout_stage_warning(s.pseudo, stage) {
        crate::elog!("{w}");
    }
}

/// 033-R11 (AMB-042-16): C2 with `[reward] pseudo = "scout"` is to be decided before C2; until
/// then r_scout / r_survive keep their weights (no C2 decay) and this warning is logged at start,
/// at resume and at the C1 → C2 transition.
fn scout_stage_warning(mode: PseudoMode, stage: crate::league::Stage) -> Option<&'static str> {
    (mode == PseudoMode::Scout && stage == crate::league::Stage::C2).then_some("[rl] WARNING: stage C2 with [reward] pseudo = \"scout\": no C2 decay is applied to r_scout / r_survive (033-R11: to be decided before C2)")
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
/// `r_scout`, `r_survive` and the 042 정찰·대응 지표, AMB-042-16; then the 042-R17 `judge_scv`,
/// AMB-042-18; then the 시작 슬롯·접촉 지표 `slot_self`, `slot_enemy`, `start_hit`, `scv_min_1m`,
/// `army_s`, AMB-042-19).
const GAMES_HEADER: &str = "game_id,agent,opponent,class,enemy_race,map,hostd_outcome,outcome,winner_disagree,decisions,r_win,r_build,r_units,r_value,replay,replay_reason,end_reason,r_scout,r_survive,ladder_group,scout_main_s,scout_types_4m,scout_types_6m,barracks_s,bunker_s,scv_4m,judge_scv,slot_self,slot_enemy,start_hit,scv_min_1m,army_s\n";

/// 020-R21 `start_hit` of a game's start slots (`players[0]`, `players[1]`); `None` without slots.
/// The value `games.csv` `start_hit` writes and `[sil] select_start_hit` (042-R16) compares.
fn slots_start_hit(slots: Option<[u8; 2]>) -> Option<bool> {
    slots.map(|[a, b]| uranium_runner::start_slots::start_hit(a, b))
}

/// 042-R2 시작 슬롯·접촉 지표 (AMB-042-19): `slot_self`, `slot_enemy` (020-R21 slots of
/// `players[0]` / `players[1]` from the game's seed, empty without one), `start_hit` (`1` / `0`,
/// the same rule for every game, empty without slots), the learning session's `scv_min_1m` and
/// `army_s` (050-R7, game seconds; empty when not reached / never seen or without a session
/// report), comma-joined.
fn start_columns(slots: Option<[u8; 2]>, result: Option<&GameResult>) -> String {
    let m: Option<ScoutMetrics> = result.and_then(|r| r.scout);
    let (a, b, hit) = match slots {
        Some([a, b]) => (a.to_string(), b.to_string(), (uranium_runner::start_slots::start_hit(a, b) as u8).to_string()),
        None => (String::new(), String::new(), String::new()),
    };
    [
        a,
        b,
        hit,
        m.and_then(|m| m.scv_min_1m).map_or(String::new(), |n| n.to_string()),
        // 3 decimals: the value the [sil] select_army_before_s comparison sees (5,401 frames =
        // 225.042 s, not "225.0").
        m.and_then(|m| m.army_frame).map_or(String::new(), |f| format!("{:.3}", seconds(f))),
    ]
    .join(",")
}

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
    h += ",dropped_lag,dropped_full,revoked,used_aborted,winner_disagree,aborted_games,steps_per_s,stage,win_open,trained_steps,queue_wait_s";
    h += DIAG_COLUMNS;
    h += "\n";
    h
}

/// 042-R2 진단 로그 (AMB-042-17): the train.csv columns after `queue_wait_s`, in this order.
const DIAG_COLUMNS: &str = ",adv_rms,adv_rms_win,adv_rms_build,adv_rms_units,adv_rms_value,value_loss_win,value_loss_build,value_loss_units,value_loss_value,kl_p99,kl_max,sil_buffer,sil_lanes,sil_loss,sil_pos_frac";

/// The 진단 로그 values of a row (leading comma). `sil`: (buffer segments, buffer lanes of this
/// update) when `[sil]` is on; every SIL column is empty when it is off (042-R16), a value the
/// learner did not measure is empty too.
fn diag_columns(s: &UpdateStats, sil: Option<(usize, usize)>) -> String {
    let opt = |x: Option<f32>| x.map_or(String::new(), |v| v.to_string());
    let mut row = format!(",{}", s.adv_rms);
    for v in s.adv_rms_channels.iter().chain(&s.value_loss_channels) {
        row += &format!(",{v}");
    }
    let (buffer, lanes) = sil.map_or((String::new(), String::new()), |(b, l)| (b.to_string(), l.to_string()));
    let (loss, pos) = if sil.is_some() { (opt(s.sil_loss), opt(s.sil_pos_frac)) } else { (String::new(), String::new()) };
    row += &format!(",{},{},{buffer},{lanes},{loss},{pos}", opt(s.kl_p99), opt(s.kl_max));
    row
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
    let b = &cfg.curriculum.band;
    crate::elog!("[rl] ladder pick {:?}, band center {} width {} exclude below {} (042, AMB-042-17)", cc.ladder_pick, b.center, b.width, b.exclude_below);
    match cfg.league.model_game_max_s {
        0 => crate::elog!("[rl] games against a model: no time cap (60 minutes, 020-R3)"),
        s => crate::elog!("[rl] games against a model end as a draw at {s} game seconds ({} frames, 042-R15)", cfg.league.game_max_frames(true)),
    }
    let rj = &cfg.league.rush_judge;
    if rj.groups.is_empty() {
        crate::elog!("[rl] rush mini-game off (042-R17)");
    } else {
        crate::elog!("[rl] rush mini-game: ladder groups {:?} end at {} game seconds ({} frames); a TimeLimitDraw there is a win with >= {} own SCVs, else a loss (042-R17)", rj.groups, rj.at_s, rj.max_frames(), rj.min_scv);
    }
    let weights: Vec<String> = cfg.ladder.iter().filter(|e| e.weight.is_some()).filter_map(|e| e.opponent().ok().map(|(_, k, _)| format!("{k} {}", e.fixed_weight()))).collect();
    if !weights.is_empty() {
        crate::elog!("[rl] \"fixed\" ladder weights: {} (absent = 1.0, 042-R19)", weights.join(", "));
    }
    let sm = &cfg.league.start_mix;
    match sm.rush_hit_share {
        None => crate::elog!("[rl] start condition mix off: every game keeps its hash seed (042-R18)"),
        Some(x) => crate::elog!("[rl] start condition mix: ladder groups {:?} take a seed whose first-target hit (020-R21) matches a target drawn with hit share {x} (042-R18)", sm.groups),
    }
    let k = &cfg.kl;
    crate::elog!("[rl] KL coefficient max({}, {} x 0.5^(update / {})) from this run's update 0, head multipliers {:?}; td_lambda {} (042-R9)", k.floor, k.init, k.half_life_updates, k.head_scale.by_head(), cfg.loss.td_lambda);
    let sc = &cfg.sil;
    if sc.enabled {
        crate::elog!(
            "[rl] self-imitation ON (Main in C1 only): mode {:?}, {} lanes of {} from a buffer of {} segments, group {:?} {}{}{}{}, segments before {} s, w_pg {} w_v {} max_uses {} (042-R16)",
            sc.mode,
            sc.lanes,
            cfg.lanes,
            sc.buffer_segments,
            sc.select_group,
            if sc.wins_only { "won games only (wins_only)".to_string() } else { format!("won or scv_4m >= {}", sc.select_scv_4m) },
            sc.select_keys.as_ref().map_or(String::new(), |k| format!(", opponents {k:?}")),
            sc.select_army_before_s.map_or(String::new(), |t| format!(", first enemy army seen by {t} s")),
            sc.select_start_hit.map_or(String::new(), |w| format!(", first-target hit = {w}")),
            sc.until_s,
            sc.w_pg,
            sc.w_v,
            sc.max_uses
        );
    } else {
        crate::elog!("[rl] self-imitation off (042-R16)");
    }
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
            if let Some(w) = scout_stage_warning(rw.pseudo, league.stage) {
                crate::elog!("{w}");
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
        bad_sessions: AtomicU64::new(0),
        finished_games: AtomicU64::new(0),
        league_cfg: cfg.league.clone(),
        sil: cfg.sil.enabled.then(|| SilState::new(cfg.sil.clone(), args.seed)),
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
    // 042-R20 (AMB-042-20): dead learner / dead inference check.
    let mut stall = StallWatch::new(cfg.actors.dead_learner_minutes, Instant::now(), 0);
    match cfg.actors.dead_learner_minutes {
        0 => crate::elog!("[rl] dead learner check off ([actors] dead_learner_minutes = 0, 042-R20)"),
        m => crate::elog!("[rl] dead learner check: abnormal end after {DEAD_LEARNER_GAMES}+ finished games without an update for {m} min, or {BAD_SESSION_STREAK} finished games in a row without a learning-session decision (042-R20)"),
    }
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
        // 042-R20: an error return (no checkpoint: after a device fault it would fail or write
        // garbage); `halt` stops the games, the process exits non-zero and the 060-R6 watchdog
        // restarts the run from LATEST.
        if let Some(why) = stall.check(Instant::now(), shared.finished_games.load(Ordering::SeqCst), shared.bad_sessions.load(Ordering::SeqCst)) {
            crate::elog!("[rl] {why}: ending abnormally without a checkpoint, the watchdog restarts from LATEST (042-R20)");
            anyhow::bail!("042-R20 stall: {why}");
        }
        let ready: Vec<AgentKind> = AgentKind::ALL.iter().copied().filter(|a| queues[a.index()].len() >= cfg.lanes).collect();
        let turn = shared.league.lock().unwrap().learner_turn(&ready);
        let Some(agent) = turn else {
            std::thread::sleep(Duration::from_millis(200));
            continue;
        };
        let i = agent.index();
        let stage = shared.league.lock().unwrap().stage;
        let Some((batch, trained, sil_row)) = take_batch(&queues[i], shared.sil.as_ref(), agent, stage, cfg.lanes, learners.version(i), cfg.actors.max_policy_lag) else { continue };
        let weights = shared.league.lock().unwrap().reward_weights(&base);
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
        write_row(&mut csv, started, agent, &stats, &shared, &queues, learners.version(i), steps_per_s, trained_steps, queue_wait_s, sil_row)?;
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
        // 042-R20: an update (and the snapshot / checkpoint after it, which may take minutes) is
        // progress.
        stall.progress(Instant::now(), shared.finished_games.load(Ordering::SeqCst));
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

/// One update batch of `agent` (042): `lanes` segments from its queue (042-R4 lag drop), or, with
/// `[sil]` on (042-R16), for Main in C1 once the buffer holds `[sil] lanes` segments, that many
/// fewer from the queue and the buffer's segments appended last (the batch shape stays; buffer
/// lanes are not in the lag drop). Also the queue lanes' trained steps (042-R2 `trained_steps`:
/// buffer lanes are reused steps) and the train.csv SIL values (buffer segments after sampling,
/// buffer lanes), `None` with `[sil]` off. `None` when the queue has too few segments.
#[allow(clippy::type_complexity)]
fn take_batch(queue: &SegmentQueue, sil: Option<&SilState>, agent: AgentKind, stage: crate::league::Stage, lanes: usize, version: u64, max_lag: u64) -> Option<(Vec<Segment>, u64, Option<(usize, usize)>)> {
    let k = match sil {
        Some(sil) if agent == AgentKind::Main && sil.in_stage(stage) && sil.buffer.lock().unwrap().len() >= sil.cfg.lanes => sil.cfg.lanes,
        _ => 0,
    };
    let mut batch = queue.try_take(lanes - k, version, max_lag)?;
    let trained = trained_steps_of(&batch);
    let row = sil.map(|sil| {
        let mut buf = sil.buffer.lock().unwrap();
        batch.extend(buf.sample(k, sil.cfg.mode, sil.cfg.max_uses));
        (buf.len(), k)
    });
    Some((batch, trained, row))
}

/// Snapshot folder name (061-R9 forbids `/` and `@`): `<agent>-<run>-<update>`.
pub fn snapshot_name(id: &str) -> String {
    id.trim_start_matches("uranium-").replace(['@', '/'], "-")
}

#[allow(clippy::too_many_arguments)]
fn write_row(csv: &mut std::fs::File, started: Instant, agent: AgentKind, s: &UpdateStats, shared: &Shared, queues: &[Arc<SegmentQueue>], version: u64, steps_per_s: f64, trained_steps: u64, queue_wait_s: f64, sil: Option<(usize, usize)>) -> Result<()> {
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
        ",{},{},{},{},{},{},{:.2},{:?},{},{},{:.3}",
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
    row += &diag_columns(s, sil);
    row += "\n";
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

    /// 042-R2: `trained_steps` and `queue_wait_s` (2026-10-31), then the 진단 로그 columns in the
    /// spec's order (AMB-042-17).
    #[test]
    fn train_csv_header_columns_042_r2() {
        assert_eq!(train_header().trim().split(',').count(), 15 + 12 + 11 + 15);
        assert!(train_header().trim().ends_with(
            ",win_open,trained_steps,queue_wait_s,adv_rms,adv_rms_win,adv_rms_build,adv_rms_units,adv_rms_value,value_loss_win,value_loss_build,value_loss_units,value_loss_value,kl_p99,kl_max,sil_buffer,sil_lanes,sil_loss,sil_pos_frac"
        ));
    }

    /// 042-R2 진단 로그 / 042-R16 (AMB-042-17): the row values; SIL columns empty when `[sil]` is
    /// off, a value not measured is empty; the row has as many values as the header.
    #[test]
    fn train_csv_diag_columns_042_r2() {
        let s = UpdateStats { adv_rms: 0.5, adv_rms_channels: [0.25, 1.0, 2.0, 0.125], value_loss_channels: [0.5, 0.75, 0.0, 3.0], kl_p99: Some(0.375), kl_max: Some(4.0), sil_loss: Some(0.5), sil_pos_frac: Some(0.25), ..Default::default() };
        assert_eq!(diag_columns(&s, None), ",0.5,0.25,1,2,0.125,0.5,0.75,0,3,0.375,4,,,,");
        assert_eq!(diag_columns(&s, Some((100, 4))), ",0.5,0.25,1,2,0.125,0.5,0.75,0,3,0.375,4,100,4,0.5,0.25");
        let none = UpdateStats { sil_loss: Some(0.0), ..Default::default() };
        assert_eq!(diag_columns(&none, Some((2, 0))), ",0,0,0,0,0,0,0,0,0,,,2,0,0,");
        assert_eq!(diag_columns(&none, None).split(',').count() - 1, DIAG_COLUMNS.split(',').count() - 1);
        // The learner's JSON without the new keys (an older learner) still reads.
        let old: UpdateStats = serde_json::from_str(r#"{"update":1,"steps":2,"loss":0.1,"policy":0.1,"value":0.0,"kl":0.0,"kl_coef":0.1,"kl_heads":[0,0,0,0,0,0],"entropy_heads":[0,0,0,0,0,0],"rho_mean":1,"adv_mean":0,"lag_mean":0,"win_open":0}"#).unwrap();
        assert_eq!((old.kl_p99, old.sil_loss, old.adv_rms), (None, None, 0.0));
        let new: UpdateStats = serde_json::from_str(r#"{"update":1,"steps":2,"loss":0.1,"policy":0.1,"value":0.0,"kl":0.0,"kl_coef":0.1,"kl_heads":[0,0,0,0,0,0],"entropy_heads":[0,0,0,0,0,0],"rho_mean":1,"adv_mean":0,"lag_mean":0,"win_open":0,"adv_rms":0.5,"adv_rms_channels":[1,2,3,4],"value_loss_channels":[1,2,3,4],"kl_p99":null,"kl_max":2.5,"sil_loss":null,"sil_pos_frac":null,"grad_norm":1.0}"#).unwrap();
        assert_eq!((new.kl_p99, new.kl_max, new.adv_rms_channels), (None, Some(2.5), [1.0, 2.0, 3.0, 4.0]));
    }

    /// 042-R15 / 020-R3 (AMB-042-17): with the cap, only games against a model (current Main,
    /// snapshot, anchor, and a failed model opponent's self-play) get `min(86,400, s x 24)`;
    /// bot games 86,400; `model_game_max_s = 0`: every game 86,400.
    #[test]
    fn model_game_time_cap_042_r15() {
        let lc = |s: u32| crate::config::LeagueConfig { model_game_max_s: s, ..RlConfig::default_file().league };
        assert_eq!((lc(0).game_max_frames(true), lc(0).game_max_frames(false)), (86_400, 86_400));
        assert_eq!((lc(1_500).game_max_frames(true), lc(1_500).game_max_frames(false)), (36_000, 86_400));
        assert_eq!(lc(3_600).game_max_frames(true), 86_400);
        assert_eq!(lc(60).game_max_frames(true), 1_440);
        // Through the job source: current Main, anchor, snapshot, T0 bots.
        let frames = |cap: u32, c1: [f64; 3], t0: f64, fail: bool| {
            let mut cfg = RlConfig::default_file();
            cfg.curriculum.start_stage = crate::league::Stage::C1;
            let cc = &mut cfg.curriculum;
            (cc.c1_t0, cc.c1_t1, cc.c1_self, cc.c1_snapshot, cc.c1_anchor) = (t0, 0.0, c1[0], c1[1], c1[2]);
            cfg.league.anchor = "checkpoints/x/ckpt-9".into();
            cfg.league.model_game_max_s = cap;
            cfg.validate().unwrap();
            let csv = PathBuf::from(if cfg!(windows) { "NUL" } else { "/dev/null" });
            let src = source_cfg(cfg, fake_servers(fail).0, &csv);
            (0..30)
                .map(|_| {
                    let j = src.next().unwrap();
                    let class = src.shared.games.lock().unwrap()[&j.spec.game_id].pick.class;
                    (class, j.spec.max_frames)
                })
                .collect::<Vec<_>>()
        };
        for (c1, fail) in [([1.0, 0.0, 0.0], false), ([0.0, 0.0, 1.0], false), ([0.0, 0.0, 1.0], true), ([0.0, 1.0, 0.0], false)] {
            assert!(frames(1_500, c1, 0.0, fail).iter().all(|&(c, f)| c == Class::Model && f == 36_000), "{c1:?} {fail}");
            assert!(frames(0, c1, 0.0, fail).iter().all(|&(_, f)| f == 86_400));
        }
        let mixed = frames(1_500, [0.5, 0.0, 0.0], 0.5, false);
        assert!(mixed.iter().any(|&(c, _)| c == Class::T0) && mixed.iter().any(|&(c, _)| c == Class::Model));
        for (c, f) in mixed {
            assert_eq!(f, if c == Class::Model { 36_000 } else { 86_400 }, "{c:?}");
        }
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
            bad_sessions: AtomicU64::new(0),
            finished_games: AtomicU64::new(0),
            league_cfg: cfg.league.clone(),
            sil: cfg.sil.enabled.then(|| SilState::new(cfg.sil.clone(), 1)),
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

    /// 042-R20 (AMB-042-20): the run loop's stall check with a fake clock — 20 finished games and
    /// `dead_learner_minutes` without an update end the run, either alone does not; progress
    /// (an update, a checkpoint after it) restarts both counts; 10 finished games in a row without
    /// a learning-session decision end it at once; 0 turns both off.
    #[test]
    fn stall_watch_dead_learner_042_r20() {
        let t0 = Instant::now();
        let at = |m: u64| t0 + Duration::from_secs(m * 60);
        let mut w = StallWatch::new(15, t0, 100);
        assert!(w.check(at(14), 100 + DEAD_LEARNER_GAMES, 0).is_none(), "under 15 min");
        assert!(w.check(at(60), 100 + DEAD_LEARNER_GAMES - 1, 0).is_none(), "under 20 games: no games, no batch expected");
        let why = w.check(at(15), 120, 0).expect("20 games, 15 min");
        assert!(why.contains("20 games") && why.contains("dead learner"), "{why}");
        w.progress(at(15), 120);
        assert!(w.check(at(29), 400, 0).is_none());
        assert!(w.check(at(30), 139, 0).is_none());
        assert!(w.check(at(30), 140, 0).is_some());
        // Dead inference: the streak alone, whatever the clock.
        assert!(w.check(at(15), 120, BAD_SESSION_STREAK - 1).is_none());
        let why = w.check(at(15), 120, BAD_SESSION_STREAK).expect("10 in a row");
        assert!(why.contains("without a decision"), "{why}");
        // Off.
        let off = StallWatch::new(0, t0, 0);
        assert!(off.check(at(600), 10_000, 1_000).is_none());
        // What counts toward the streak: finished games only.
        let r = |d: u64| GameResult { agent: 0, game_id: 1, outcome: None, reward_sum: [0.0; 4], scout_sum: 0.0, survive_sum: 0.0, scout: None, decisions: d, enemy_race: None };
        assert_eq!(bad_learning_session(JobOutcome::Finished, Some(&r(0))), Some(true));
        assert_eq!(bad_learning_session(JobOutcome::Finished, None), None, "no report of the learning session: not counted");
        assert_eq!(bad_learning_session(JobOutcome::Finished, Some(&r(3))), Some(false));
        for o in [JobOutcome::Aborted, JobOutcome::Timeout, JobOutcome::Stopped, JobOutcome::Rejected] {
            assert_eq!(bad_learning_session(o, Some(&r(0))), None, "{o:?}");
        }
    }

    /// 042-R20 (AMB-042-20): `done` keeps the streak of finished games without a learning-session
    /// decision; a game with decisions resets it, an Aborted game leaves it.
    #[test]
    fn bad_session_streak_in_the_run_loop_042_r20() {
        use uranium_proto::EndReason;
        let mut cfg = RlConfig::default_file();
        cfg.curriculum.start_stage = crate::league::Stage::C1;
        let cc = &mut cfg.curriculum;
        (cc.c1_t0, cc.c1_t1, cc.c1_self) = (0.0, 0.0, 1.0);
        let (servers, _, _) = fake_servers(false);
        let csv = PathBuf::from(if cfg!(windows) { "NUL" } else { "/dev/null" });
        let src = source_cfg(cfg, servers, &csv);
        let play = |decisions: u64, outcome: JobOutcome| {
            let job = src.next().unwrap();
            let id = job.spec.game_id;
            src.shared.games.lock().unwrap().get_mut(&id).unwrap().self_player = Some(0);
            for slot in [0u8, 1] {
                src.shared.ends.push(id, uranium_agent::SessionEnd { slot, winner: Some(0), frame: 100 + slot as u32, reason: EndReason::Elimination });
            }
            src.hooks[0].tx.send(GameResult { agent: 0, game_id: id, outcome: None, reward_sum: [0.0; 4], scout_sum: 0.0, survive_sum: 0.0, scout: None, decisions, enemy_race: Some(Race::Terran) }).unwrap();
            src.done(&job, outcome, None);
            src.shared.bad_sessions.load(Ordering::SeqCst)
        };
        assert_eq!(play(0, JobOutcome::Finished), 1);
        assert_eq!(play(0, JobOutcome::Finished), 2);
        assert_eq!(play(0, JobOutcome::Aborted), 2, "not finished: neither");
        assert_eq!(play(7, JobOutcome::Finished), 0, "a decision resets");
        for k in 1..=BAD_SESSION_STREAK {
            assert_eq!(play(0, JobOutcome::Finished), k);
        }
        // The stall check counts `Finished` games only (not the Aborted one).
        assert_eq!(src.shared.finished_games.load(Ordering::SeqCst), 3 + BAD_SESSION_STREAK);
        assert_eq!(src.shared.games_done.load(Ordering::SeqCst), 4 + BAD_SESSION_STREAK);
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
        assert_eq!(GAMES_HEADER.trim().split(',').count(), 32);
        // 042-R2 (AMB-042-16): the new columns follow end_reason in this order; 042-R17
        // (AMB-042-18): `judge_scv` after scv_4m; 042-AC9 (AMB-042-19): the five start-slot and
        // contact columns last. The earlier 27 columns are unchanged (rl.toml / rl_rush1..4 rows
        // only gain the trailing columns).
        assert!(GAMES_HEADER.trim().ends_with(",replay,replay_reason,end_reason,r_scout,r_survive,ladder_group,scout_main_s,scout_types_4m,scout_types_6m,barracks_s,bunker_s,scv_4m,judge_scv,slot_self,slot_enemy,start_hit,scv_min_1m,army_s"));
        assert!(GAMES_HEADER.starts_with("game_id,agent,opponent,class,enemy_race,map,hostd_outcome,outcome,winner_disagree,decisions,r_win,r_build,r_units,r_value,replay,"));
    }

    /// 042-AC9 / 042-R2 (AMB-042-19): `slot_self`, `slot_enemy` = the 020-R21 generator's values,
    /// `start_hit` 1 / 0 / empty, `scv_min_1m`, `army_s` in game seconds, empty when not reached
    /// or without a session report.
    #[test]
    fn start_columns_042_ac9() {
        let m = ScoutMetrics { scv_min_1m: Some(3), army_frame: Some(4_452), ..Default::default() };
        let slots = |seed| Some(uranium_runner::start_slots::start_slots(seed, 0));
        assert_eq!(start_columns(slots(43_006), Some(&result(1, Some(m)))), "1,0,1,3,185.500", "c0 g6: (1, 0) is a hit");
        let late = ScoutMetrics { army_frame: Some(5_401), ..Default::default() };
        assert_eq!(start_columns(None, Some(&result(1, Some(late)))), ",,,,225.042", "the exact value select_army_before_s compares");
        assert_eq!(start_columns(slots(43_001), Some(&result(1, Some(ScoutMetrics::default())))), "2,3,0,,", "c0 g1: a miss, nothing reached");
        assert_eq!(start_columns(slots(43_001), None), "2,3,0,,", "no session report");
        assert_eq!(start_columns(None, None), ",,,,", "no seed");
    }

    /// 042-R18 / 042-AC9 (AMB-042-19): without a share the seed is the earlier hash seed; with one
    /// it is the same for the same game id, its 020-R21 hit is the target, about `rush_hit_share`
    /// of the games are hits, and when no candidate fits the game keeps `s_0`.
    #[test]
    fn start_mix_seeds_042_r18() {
        use uranium_runner::start_slots::{start_hit, start_slots};
        let hit = |s: u32| {
            let [a, b] = start_slots(s, 0);
            start_hit(a, b)
        };
        for id in 1..200u64 {
            assert_eq!(game_seed(id, None, 0), base_seed(id), "no share: the earlier seed");
            assert_eq!(base_seed(id), (splitmix(&mut (id ^ 0xA5A5_5A5A)) * u32::MAX as f64) as u32);
        }
        for share in [0.0, 0.5, 1.0, 0.7] {
            let n = 4_000u64;
            let hits = (1..=n).filter(|&id| hit(game_seed(id, Some(share), 0))).count() as f64 / n as f64;
            assert!((hits - share).abs() < 0.03, "share {share}: {hits}");
        }
        for id in [1u64, 77, 1_000_000] {
            assert_eq!(game_seed(id, Some(0.5), 0), game_seed(id, Some(0.5), 0), "deterministic");
            assert!(hit(game_seed(id, Some(1.0), 0)));
            assert!(!hit(game_seed(id, Some(0.0), 0)));
            // A candidate is kept as soon as it fits: s_0 when it already does.
            let s0 = base_seed(id);
            assert_eq!(game_seed(id, Some(if hit(s0) { 1.0 } else { 0.0 }), 0), s0);
            // No candidate fits: s_0.
            assert_eq!(mixed_seed(id, 1.0, |_| false), s0);
            assert_eq!(mixed_seed(id, 0.0, |_| true), s0);
        }
        // Random players draw first: the search uses the same generator input.
        let s = game_seed(5, Some(1.0), 1);
        let [a, b] = start_slots(s, 1);
        assert!(start_hit(a, b));
    }

    fn result(id: u64, scout: Option<ScoutMetrics>) -> GameResult {
        GameResult { agent: 0, game_id: id, outcome: Some(Outcome::Win), reward_sum: [0.0, 0.0, 0.0, 0.25], scout_sum: 0.6, survive_sum: 0.375, scout, decisions: 9, enemy_race: Some(Race::Zerg) }
    }

    /// 042-R2 / 042-AC6 (AMB-042-16): the columns after `end_reason` — raw sums only in "scout",
    /// the ladder group of a ladder game, the metrics in game seconds, empty when not reached or
    /// without a session report.
    #[test]
    fn scout_columns_042_r2() {
        let m = ScoutMetrics { main_frame: Some(3_000), types_4m: 2, types_6m: 5, barracks_frame: Some(2_412), bunker_frame: None, scv_4m: Some(14), scv_last: Some(9), ..Default::default() };
        let r = result(1, Some(m));
        assert_eq!(scout_columns(PseudoMode::Scout, Some(&r), Some("macro")), "0.6,0.375,macro,125.0,2,5,100.5,,14");
        assert_eq!(scout_columns(PseudoMode::Z, Some(&r), Some("rush")), ",,rush,125.0,2,5,100.5,,14", "z: r_scout / r_survive empty");
        assert_eq!(scout_columns(PseudoMode::Scout, Some(&result(2, Some(ScoutMetrics::default()))), None), "0.6,0.375,,,0,0,,,", "nothing reached");
        assert_eq!(scout_columns(PseudoMode::Scout, None, None), ",,,,,,,,", "no session report");
        assert_eq!(scout_columns(PseudoMode::Z, None, None).split(',').count(), 9);
    }

    /// 042-R16 / 042-AC7 (AMB-042-17) in the run loop: Main's games of `select_group` keep segment
    /// copies (those starting before `until_s`, the terminal one with r_win) until the game is
    /// judged; a won or held (`scv_4m >= 12`) game's copies go to the buffer with its r_win, the
    /// others' (lost and not held, another group, Aborted) are dropped; nothing stays pending.
    #[test]
    fn sil_buffer_takes_selected_games_042_r16() {
        use crate::league::{LadderEntry, LadderKind};
        use uranium_agent::T0Build;
        use uranium_env::features::executor::HostAction;
        use uranium_proto::EndReason;
        let csv = PathBuf::from(if cfg!(windows) { "NUL" } else { "/dev/null" });
        for until_s in [0u32, 2] {
            let mut cfg = RlConfig::default_file();
            cfg.curriculum.start_stage = crate::league::Stage::C1;
            let cc = &mut cfg.curriculum;
            (cc.c1_t0, cc.c1_t1, cc.c1_self, cc.c1_ladder) = (0.0, 0.0, 0.0, 1.0);
            cc.ladder_groups = [("rush".to_string(), 0.5), ("macro".to_string(), 0.5)].into_iter().collect();
            let entry = |build, group: &str| LadderEntry { kind: LadderKind::T0, build: Some(build), name: None, race: Race::Zerg, group: group.into(), weight: None };
            cfg.ladder = vec![entry(T0Build::ZergRush, "rush"), entry(T0Build::Standard, "macro")];
            cfg.sil = SilConfig { enabled: true, until_s, lanes: 2, buffer_segments: 1_000, ..SilConfig::default() };
            cfg.validate().unwrap();
            let src = source_cfg(cfg, fake_servers(false).0, &csv);
            let info = crate::testkit::game_info();
            // 10 decisions at frames 0, 10, ..., 90; segments of 4 (seq_len 4 in the test hooks)
            // start at decisions 0, 3, 6, 9 = frames 0, 30, 60, 90 (the last is terminal).
            let obs = crate::testkit::observations(10);
            let cases = [(Outcome::Win, None, false), (Outcome::Loss, Some(12), false), (Outcome::Loss, Some(5), false), (Outcome::Win, Some(20), true)];
            let (mut want, mut groups) = (0usize, BTreeSet::new());
            for g in 0..24 {
                let (outcome, scv_4m, aborted) = cases[g % cases.len()];
                let job = src.next().unwrap();
                let id = job.spec.game_id;
                let group = src.shared.games.lock().unwrap()[&id].pick.ladder_group.clone();
                groups.insert(group.clone());
                let (_, mut o) = src.hooks[0].session(id, &info);
                let a = HostAction { action: 0, delay_bucket: 0, queued: false, selected: vec![], target_unit: None, target_loc: None };
                for (i, ob) in obs.iter().enumerate() {
                    let state = o.wants_state().then(uranium_agent::infer::HostState::default);
                    o.on_decision(&uranium_agent::model_policy::Decision { frame: 10 * i as u32, obs: ob, action: &a, log_prob: 0.0, head_log_prob: [0.0; uranium_agent::infer::HEADS], value: [0.0; 4], rejected: false, version: 0, reward_before: Default::default(), state });
                }
                o.on_end(Default::default(), Some(outcome));
                src.shared.results.lock().unwrap().try_iter().for_each(drop);
                src.shared.games.lock().unwrap().get_mut(&id).unwrap().self_player = Some(0);
                let winner = if outcome == Outcome::Win { 0 } else { 1 };
                for slot in [0u8, 1] {
                    src.shared.ends.push(id, uranium_agent::SessionEnd { slot, winner: Some(winner), frame: 9_000, reason: EndReason::Elimination });
                }
                let m = ScoutMetrics { scv_4m, ..Default::default() };
                src.hooks[0].tx.send(GameResult { outcome: Some(outcome), ..result(id, Some(m)) }).unwrap();
                src.done(&job, if aborted { JobOutcome::Aborted } else { JobOutcome::Finished }, None);
                let selected = group.as_deref() == Some("rush") && !aborted && (outcome == Outcome::Win || scv_4m.is_some_and(|n| n >= 12));
                if selected {
                    want += if until_s == 0 { 4 } else { 2 };
                }
            }
            assert!(groups.contains(&Some("rush".to_string())) && groups.contains(&Some("macro".to_string())), "{groups:?}");
            let sil = src.shared.sil.as_ref().unwrap();
            assert!(sil.pending.lock().unwrap().is_empty(), "nothing left pending");
            let mut buf = sil.buffer.lock().unwrap();
            assert!(want > 0);
            assert_eq!(buf.len(), want, "until_s {until_s}");
            let segs = buf.sample(want, crate::sil::SilMode::Sil, 4);
            assert_eq!(segs.iter().filter(|s| s.terminal).count(), if until_s == 0 { want / 4 } else { 0 });
            for s in &segs {
                let r = s.sil_return.unwrap();
                assert!(r == 1.0 || r == -1.0, "{r}");
                if until_s > 0 {
                    assert!(s.start_frame < until_s * 24, "{}", s.start_frame);
                }
                if s.terminal {
                    assert_eq!(s.steps.last().unwrap().reward[0], r, "terminal copy carries r_win");
                }
            }
        }
    }

    /// 042-R16 상대·접촉 거르기 / 042-AC9 (AMB-042-19) in the run loop: with `select_keys` and
    /// `select_army_before_s` only won games against a listed opponent whose first enemy army
    /// sighting is known and at most the limit reach the buffer (an unlisted opponent's games keep
    /// no copies at all); nothing stays pending. With `select_start_hit = true` (사용자 위임
    /// 2026-11-05, 5차 자문) also only first-target hits by the game's start slots (a miss keeps no
    /// copies either); without it the same games as before.
    #[test]
    fn sil_opponent_and_contact_filters_in_the_run_loop_042_r16() {
        for select_start_hit in [None, Some(true)] {
            sil_filters_in_the_run_loop(select_start_hit);
        }
    }

    fn sil_filters_in_the_run_loop(select_start_hit: Option<bool>) {
        use crate::league::{LadderEntry, LadderKind};
        use uranium_agent::T0Build;
        use uranium_env::features::executor::HostAction;
        use uranium_proto::EndReason;
        let csv = PathBuf::from(if cfg!(windows) { "NUL" } else { "/dev/null" });
        let mut cfg = RlConfig::default_file();
        cfg.curriculum.start_stage = crate::league::Stage::C1;
        let cc = &mut cfg.curriculum;
        (cc.c1_t0, cc.c1_t1, cc.c1_self, cc.c1_ladder) = (0.0, 0.0, 0.0, 1.0);
        cc.ladder_groups = [("rush".to_string(), 0.8), ("macro".to_string(), 0.2)].into_iter().collect();
        let entry = |build, group: &str| LadderEntry { kind: LadderKind::T0, build: Some(build), name: None, race: Race::Zerg, group: group.into(), weight: None };
        cfg.ladder = vec![entry(T0Build::ZergRush, "rush"), entry(T0Build::ZergMuta, "rush"), entry(T0Build::Standard, "macro")];
        cfg.sil = SilConfig { enabled: true, until_s: 0, lanes: 2, buffer_segments: 1_000, wins_only: true, select_keys: Some(vec!["t0-zerg-rush".into()]), select_army_before_s: Some(225.0), select_start_hit, ..SilConfig::default() };
        cfg.validate().unwrap();
        let src = source_cfg(cfg, fake_servers(false).0, &csv);
        let info = crate::testkit::game_info();
        let obs = crate::testkit::observations(10);
        // army_frame: 5,400 = 225.0 s (selected), 5,401 (not), never seen (not), 2,000 (selected).
        let armies = [Some(5_400u32), Some(5_401), None, Some(2_000)];
        let (mut want, mut keys, mut misses) = (0usize, BTreeSet::new(), 0);
        for g in 0..120 {
            let army_frame = armies[g % armies.len()];
            let job = src.next().unwrap();
            let id = job.spec.game_id;
            let (key, hit) = {
                let games = src.shared.games.lock().unwrap();
                (games[&id].pick.key.clone(), slots_start_hit(games[&id].slots))
            };
            assert!(hit.is_some(), "every league game has start slots");
            let start_ok = select_start_hit.is_none_or(|w| hit == Some(w));
            keys.insert(key.clone());
            let (_, mut o) = src.hooks[0].session(id, &info);
            let a = HostAction { action: 0, delay_bucket: 0, queued: false, selected: vec![], target_unit: None, target_loc: None };
            for (i, ob) in obs.iter().enumerate() {
                let state = o.wants_state().then(uranium_agent::infer::HostState::default);
                o.on_decision(&uranium_agent::model_policy::Decision { frame: 10 * i as u32, obs: ob, action: &a, log_prob: 0.0, head_log_prob: [0.0; uranium_agent::infer::HEADS], value: [0.0; 4], rejected: false, version: 0, reward_before: Default::default(), state });
            }
            o.on_end(Default::default(), Some(Outcome::Win));
            if key == "t0-zerg-rush" && !start_ok {
                misses += 1;
                assert!(!src.shared.sil.as_ref().unwrap().pending.lock().unwrap().contains_key(&id), "a miss keeps no copies");
            }
            src.shared.results.lock().unwrap().try_iter().for_each(drop);
            src.shared.games.lock().unwrap().get_mut(&id).unwrap().self_player = Some(0);
            for slot in [0u8, 1] {
                src.shared.ends.push(id, uranium_agent::SessionEnd { slot, winner: Some(0), frame: 9_000, reason: EndReason::Elimination });
            }
            src.hooks[0].tx.send(result(id, Some(ScoutMetrics { army_frame, ..Default::default() }))).unwrap();
            src.done(&job, JobOutcome::Finished, None);
            if key == "t0-zerg-rush" && army_frame.is_some_and(|f| f <= 5_400) && start_ok {
                want += 4;
            }
        }
        assert!(keys.len() == 3, "{keys:?}");
        assert_eq!(misses > 0, select_start_hit.is_some(), "{misses} listed-opponent misses");
        let sil = src.shared.sil.as_ref().unwrap();
        assert!(sil.pending.lock().unwrap().is_empty(), "nothing left pending");
        assert!(want > 0);
        assert_eq!(sil.buffer.lock().unwrap().len(), want);
    }

    /// 042-R18 / 042-AC9 (AMB-042-19) through the job source with config/rl_rush5.toml: rush games
    /// take the `[league.start_mix]` seed (hits about 0.5), macro games and games against a model
    /// keep the earlier hash seed; config/rl_rush4.toml (no `[league.start_mix]`) keeps every seed.
    #[test]
    fn start_mix_in_the_job_source_042_r18() {
        let seeds = |file: &str| {
            let cfg: RlConfig = toml::from_str(file).unwrap();
            cfg.validate().unwrap();
            let csv = PathBuf::from(if cfg!(windows) { "NUL" } else { "/dev/null" });
            let src = source_cfg(cfg, fake_servers(false).0, &csv);
            (0..3_000)
                .map(|_| {
                    let j = src.next().unwrap();
                    let games = src.shared.games.lock().unwrap();
                    let m = &games[&j.spec.game_id];
                    assert_eq!(m.slots, Some(uranium_runner::start_slots::start_slots(j.spec.seed.unwrap(), 0)), "slots of the seed");
                    (j.spec.game_id, m.pick.class, m.pick.ladder_group.clone(), j.spec.seed.unwrap())
                })
                .collect::<Vec<_>>()
        };
        let hit = |s: u32| {
            let [a, b] = uranium_runner::start_slots::start_slots(s, 0);
            uranium_runner::start_slots::start_hit(a, b)
        };
        let five = seeds(include_str!("../../../config/rl_rush5.toml"));
        let (mut rush, mut rush_hits, mut other) = (0, 0, 0);
        for (id, class, group, seed) in &five {
            if group.as_deref() == Some("rush") {
                assert_eq!(*seed, game_seed(*id, Some(0.5), 0));
                rush += 1;
                rush_hits += hit(*seed) as usize;
            } else {
                assert_eq!(*seed, base_seed(*id), "{class:?} {group:?}: the earlier seed");
                other += 1;
            }
        }
        let share = rush_hits as f64 / rush as f64;
        assert!((share - 0.5).abs() < 0.04 && other > 0, "{rush_hits} / {rush}, others {other}");
        assert!(seeds(include_str!("../../../config/rl_rush4.toml")).iter().all(|(id, _, _, s)| *s == base_seed(*id)));
    }

    /// 042-R16 / 042-AC7 (AMB-042-17) lane swap of the run loop: buffer lanes replace the last
    /// `[sil] lanes` lanes only for Main in C1 once the buffer holds that many (the batch keeps
    /// `lanes` segments, `trained` counts the queue lanes only); buffer lanes are outside the
    /// 042-R4 lag drop ("replay": a segment leaves after `max_uses`); `[sil]` off = the plain take.
    #[test]
    fn sil_lane_swap_042_r16() {
        use crate::league::Stage;
        use crate::sil::SilMode;
        use crate::trajectory::tests::seg_at;
        let fill = |q: &SegmentQueue, ids: std::ops::Range<u64>, version: u64| ids.for_each(|i| q.push(seg_at(i, version)));
        for mode in [SilMode::Sil, SilMode::Replay] {
            let sil = SilState::new(SilConfig { enabled: true, mode, lanes: 2, buffer_segments: 100, max_uses: 2, ..SilConfig::default() }, 1);
            let q = SegmentQueue::new(64);
            // Buffer below `lanes`: the plain take.
            sil.buffer.lock().unwrap().push_game(vec![seg_at(900, 0)], 1.0);
            fill(&q, 0..4, 100);
            let (b, trained, row) = take_batch(&q, Some(&sil), AgentKind::Main, Stage::C1, 4, 100, 32).unwrap();
            assert_eq!((b.len(), trained, row), (4, 4, Some((1, 0))));
            assert!(b.iter().all(|s| s.sil_return.is_none()));
            // Enough: 2 queue lanes + 2 buffer lanes, last. The buffer's segments are 100 updates
            // old (version 0 at learner version 100, max lag 32): kept; such queue segments are dropped.
            sil.buffer.lock().unwrap().push_game(vec![seg_at(901, 0), seg_at(902, 0)], -1.0);
            fill(&q, 10..12, 0);
            fill(&q, 20..24, 100);
            let (b, trained, row) = take_batch(&q, Some(&sil), AgentKind::Main, Stage::C1, 4, 100, 32).unwrap();
            assert_eq!(b.len(), 4, "the batch shape stays");
            assert_eq!(trained, 2, "queue lanes only");
            assert_eq!(b.iter().map(|s| s.sil_return.is_some()).collect::<Vec<_>>(), [false, false, true, true]);
            assert_eq!(b.iter().take(2).map(|s| s.game_id).collect::<Vec<_>>(), [20, 21], "the lagging queue segments dropped");
            assert!(b[2..].iter().all(|s| s.game_id >= 900 && s.version() == 0), "buffer lanes outside the lag drop");
            assert_eq!(q.stats().dropped_lag, 2);
            assert_eq!(row.unwrap().1, 2);
            // Other stages and agents: no buffer lanes.
            fill(&q, 30..40, 100);
            for (agent, stage) in [(AgentKind::Main, Stage::C2), (AgentKind::Main, Stage::C0), (AgentKind::MainExploiter, Stage::C1)] {
                let (b, trained, row) = take_batch(&q, Some(&sil), agent, stage, 4, 100, 32).unwrap();
                assert!(b.iter().all(|s| s.sil_return.is_none()) && b.len() == 4 && trained == 4, "{agent:?} {stage:?}");
                assert_eq!(row.unwrap().1, 0);
            }
            // "replay": after max_uses (2) uses a segment leaves; "sil" keeps it.
            let before = sil.buffer.lock().unwrap().len();
            for _ in 0..3 {
                fill(&q, 50..52, 100);
                take_batch(&q, Some(&sil), AgentKind::Main, Stage::C1, 4, 100, 32);
            }
            let after = sil.buffer.lock().unwrap().len();
            match mode {
                SilMode::Sil => assert_eq!(after, before),
                SilMode::Replay => assert!(after < before, "{before} -> {after}"),
            }
        }
        // [sil] off: the plain take, no SIL values.
        let q = SegmentQueue::new(8);
        fill(&q, 0..4, 0);
        let (b, trained, row) = take_batch(&q, None, AgentKind::Main, Stage::C1, 4, 0, 32).unwrap();
        assert_eq!((b.len(), trained, row), (4, 4, None));
        assert!(take_batch(&q, None, AgentKind::Main, Stage::C1, 4, 0, 32).is_none(), "too few");
    }

    /// 033-R11 (AMB-042-16): the "scout" C2 warning (start, resume, C1 → C2 transition).
    #[test]
    fn scout_c2_warning_033_r11() {
        use crate::league::Stage;
        assert!(scout_stage_warning(PseudoMode::Scout, Stage::C2).is_some_and(|w| w.contains("WARNING")));
        for (m, st) in [(PseudoMode::Scout, Stage::C1), (PseudoMode::Scout, Stage::C0), (PseudoMode::Z, Stage::C2)] {
            assert!(scout_stage_warning(m, st).is_none(), "{m:?} {st:?}");
        }
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
        let entry = |build, group: &str| LadderEntry { kind: LadderKind::T0, build: Some(build), name: None, race: Race::Zerg, group: group.into(), weight: None };
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
            assert_eq!(&r[17..27], [&["0.6", "0.375", group.as_deref().unwrap_or("")][..], &["100.0", "1", "3", "", "", "", ""]].concat(), "{line}");
            // 042-AC9 (AMB-042-19): the slots of the game's seed, no scv_min_1m / army_s reached.
            let [a, b] = uranium_runner::start_slots::start_slots(job.spec.seed.unwrap(), 0);
            let hit = if uranium_runner::start_slots::start_hit(a, b) { "1" } else { "0" };
            assert_eq!(&r[27..], [a.to_string().as_str(), b.to_string().as_str(), hit, "", ""], "{line}");
            if let Some(g) = &group {
                assert_eq!(r[2].starts_with("t0-zerg-rush"), g == "rush", "{line}");
            }
        }
        assert!(model > 0 && bot > 0, "{model} {bot}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// 042-R17 / 042-AC8 판 길이 (AMB-042-18): through the job source with config/rl_rush4.toml,
    /// ladder games of the judged groups (rush, warmup) get `min(86,400, at_s x 24)` = 11,520,
    /// macro games 86,400 and games against a model the 042-R15 36,000; with rl_rush3.toml (no
    /// `[league.rush_judge]`) every ladder game keeps 86,400 as before.
    #[test]
    fn rush_judge_game_lengths_042_r17() {
        let frames = |file: &str| {
            let cfg: RlConfig = toml::from_str(file).unwrap();
            cfg.validate().unwrap();
            let csv = PathBuf::from(if cfg!(windows) { "NUL" } else { "/dev/null" });
            let src = source_cfg(cfg, fake_servers(false).0, &csv);
            (0..400)
                .map(|_| {
                    let j = src.next().unwrap();
                    let games = src.shared.games.lock().unwrap();
                    let p = &games[&j.spec.game_id].pick;
                    (p.class, p.ladder_group.clone(), j.spec.max_frames)
                })
                .collect::<Vec<_>>()
        };
        let seen = frames(include_str!("../../../config/rl_rush4.toml"));
        for g in ["warmup", "rush", "macro"] {
            assert!(seen.iter().any(|(_, x, _)| x.as_deref() == Some(g)), "{g}");
        }
        assert!(seen.iter().any(|(c, _, _)| *c == Class::Model));
        for (class, group, f) in &seen {
            let want = match (class, group.as_deref()) {
                (Class::Model, _) => 36_000,
                (_, Some("rush" | "warmup")) => 11_520,
                _ => 86_400,
            };
            assert_eq!(*f, want, "{class:?} {group:?}");
        }
        let rush3 = frames(include_str!("../../../config/rl_rush3.toml"));
        assert!(rush3.iter().any(|(_, g, _)| g.as_deref() == Some("rush")));
        for (class, group, f) in rush3 {
            assert_eq!(f, if class == Class::Model { 36_000 } else { 86_400 }, "rush-s3 as before: {class:?} {group:?}");
        }
    }

    /// 042-R17 / 020-R19 / 042-AC8 판정 (AMB-042-18) in the run loop: a judged group's game that
    /// ends as `TimeLimitDraw` is a win at `min_scv` (8) own SCVs and a loss at 7 — games.csv
    /// `outcome`, `r_win` (also on the terminal decision), `end_reason = RushJudge`, `judge_scv`,
    /// the win table and the self-imitation selection (`wins_only`: the won one only, not the
    /// lost one with `scv_4m >= 12`) use the judged result; with no observation to count the game
    /// has no result (not trained, counted with Aborted); earlier ends (Elimination, Surrender,
    /// collapse Concede) and another group's `TimeLimitDraw` keep their outcome, `judge_scv` empty.
    #[test]
    fn rush_judge_results_in_the_run_loop_042_r17() {
        use crate::config::RushJudgeConfig;
        use crate::league::{LadderEntry, LadderKind};
        use uranium_agent::T0Build;
        use uranium_env::features::executor::HostAction;
        let dir = std::env::temp_dir().join(format!("uranium-rl-rushjudge-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let csv = dir.join("games.csv");
        drop(open_csv(&csv, GAMES_HEADER).unwrap());
        let mut cfg = RlConfig::default_file();
        cfg.curriculum.start_stage = crate::league::Stage::C1;
        let cc = &mut cfg.curriculum;
        (cc.c1_t0, cc.c1_t1, cc.c1_self, cc.c1_ladder) = (0.0, 0.0, 0.0, 1.0);
        cc.ladder_groups = [("rush".to_string(), 0.5), ("macro".to_string(), 0.5)].into_iter().collect();
        let entry = |build, group: &str| LadderEntry { kind: LadderKind::T0, build: Some(build), name: None, race: Race::Zerg, group: group.into(), weight: None };
        cfg.ladder = vec![entry(T0Build::ZergRush, "rush"), entry(T0Build::Standard, "macro")];
        cfg.league.win_rate_decay = 0.9;
        cfg.league.rush_judge = RushJudgeConfig { groups: vec!["rush".into()], ..RushJudgeConfig::default() };
        cfg.sil = SilConfig { enabled: true, wins_only: true, lanes: 2, buffer_segments: 100, ..SilConfig::default() };
        cfg.concede.enabled = true;
        cfg.validate().unwrap();
        let src = source_cfg(cfg, fake_servers(false).0, &csv);
        let info = crate::testkit::game_info();
        let obs = crate::testkit::observations(2);
        struct Played {
            row: Vec<String>,
            /// Main's win-table value against the opponent after the game minus before.
            win_delta: f64,
            /// r_win on the terminal decision as queued for the learner (`None`: nothing queued).
            terminal_r_win: Option<f32>,
            buffer: usize,
            aborted: u64,
        }
        // One game of `group`: `decisions` decisions (one terminal segment), the session's report
        // with `scout` (metrics with `scv_last`), both sessions' GameEnd, optionally a collapse
        // concede of the learning agent.
        let play = |group: &str, decisions: usize, reason: EndReason, winner: Option<u8>, scout: ScoutMetrics, concede: bool| {
            let job = loop {
                let j = src.next().unwrap();
                if src.shared.games.lock().unwrap()[&j.spec.game_id].pick.ladder_group.as_deref() == Some(group) {
                    break j;
                }
                src.requeue(j);
            };
            let id = job.spec.game_id;
            let key = src.shared.games.lock().unwrap()[&id].pick.key.clone();
            let before = src.shared.league.lock().unwrap().win_rate(AgentKind::Main, &key);
            let (_, mut o) = src.hooks[0].session(id, &info);
            let a = HostAction { action: 0, delay_bucket: 0, queued: false, selected: vec![], target_unit: None, target_loc: None };
            for (i, ob) in obs.iter().take(decisions).enumerate() {
                let state = o.wants_state().then(uranium_agent::infer::HostState::default);
                o.on_decision(&uranium_agent::model_policy::Decision { frame: 24 * i as u32, obs: ob, action: &a, log_prob: 0.0, head_log_prob: [0.0; uranium_agent::infer::HEADS], value: [0.0; 4], rejected: false, version: 0, reward_before: Default::default(), state });
            }
            let session_view = match winner {
                Some(0) => Outcome::Win,
                Some(_) => Outcome::Loss,
                None => Outcome::Draw,
            };
            o.on_end(Default::default(), Some(session_view));
            src.shared.results.lock().unwrap().try_iter().for_each(drop);
            src.shared.games.lock().unwrap().get_mut(&id).unwrap().self_player = Some(0);
            if concede {
                assert!(src.shared.concede.get(id).unwrap().claim(0, 7_200, LeaveRule::Concede));
            }
            for slot in [0u8, 1] {
                src.shared.ends.push(id, uranium_agent::SessionEnd { slot, winner, frame: 11_520, reason });
            }
            src.hooks[0].tx.send(GameResult { outcome: Some(session_view), ..result(id, Some(scout)) }).unwrap();
            let aborted = src.shared.aborted.load(Ordering::SeqCst);
            src.done(&job, JobOutcome::Finished, None);
            let row: Vec<String> = std::fs::read_to_string(&csv).unwrap().lines().last().unwrap().split(',').map(str::to_string).collect();
            assert_eq!(row.len(), GAMES_HEADER.trim().split(',').count(), "{row:?}");
            let terminal_r_win = src.shared.queues[0].try_take(1, 0, 1_000).map(|b| b[0].steps.last().unwrap().reward[0]);
            let win_delta = src.shared.league.lock().unwrap().win_rate(AgentKind::Main, &key) - before;
            let buffer = src.shared.sil.as_ref().unwrap().buffer.lock().unwrap().len();
            Played { row, win_delta, terminal_r_win, buffer, aborted: src.shared.aborted.load(Ordering::SeqCst) - aborted }
        };
        let m = |scv_last: Option<u32>, scv_4m: Option<u32>| ScoutMetrics { scv_last, scv_4m, ..Default::default() };
        let cols = |p: &Played| (p.row[7].clone(), p.row[10].clone(), p.row[16].clone(), p.row[26].clone());
        let s = |x: &str| x.to_string();
        // min_scv boundary: 8 own SCVs win, 7 lose.
        let won = play("rush", 2, EndReason::TimeLimitDraw, None, m(Some(8), Some(12)), false);
        assert_eq!(cols(&won), (s("Some(Win)"), s("1"), s("RushJudge"), s("8")));
        assert_eq!(won.terminal_r_win, Some(1.0), "r_win on the last decision");
        assert!((won.win_delta - 0.05).abs() < 1e-9, "win table: a win (0.9 x 0.5 + 0.1 x 1)");
        assert_eq!(won.buffer, 1, "a won game: selected (its terminal segment)");
        let sampled = src.shared.sil.as_ref().unwrap().buffer.lock().unwrap().sample(1, crate::sil::SilMode::Sil, 4);
        assert_eq!(sampled[0].sil_return, Some(1.0), "the SIL return R is the judged r_win");
        let lost = play("rush", 2, EndReason::TimeLimitDraw, None, m(Some(7), Some(12)), false);
        assert_eq!(cols(&lost), (s("Some(Loss)"), s("-1"), s("RushJudge"), s("7")));
        assert_eq!(lost.terminal_r_win, Some(-1.0));
        assert!(lost.win_delta < 0.0, "win table: a loss");
        assert_eq!(lost.buffer, 1, "wins_only: a lost game with scv_4m >= 12 is not selected");
        // No observation to count: no result, nothing trained, the win table untouched.
        let none = play("rush", 0, EndReason::TimeLimitDraw, None, m(None, None), false);
        assert_eq!(cols(&none), (s("None"), s("0"), s(""), s("")));
        assert_eq!((none.terminal_r_win, none.win_delta, none.aborted), (None, 0.0, 1));
        // Ended before the cap: the original outcome.
        let elim = play("rush", 2, EndReason::Elimination, Some(0), m(Some(3), None), false);
        assert_eq!(cols(&elim), (s("Some(Win)"), s("1"), s("Elimination"), s("")));
        assert_eq!((elim.terminal_r_win, elim.buffer), (Some(1.0), 2), "a win before 8:00 is selected too");
        let gg = play("rush", 2, EndReason::Surrender, Some(1), m(Some(20), None), false);
        assert_eq!(cols(&gg), (s("Some(Loss)"), s("-1"), s("Surrender"), s("")));
        let conceded = play("rush", 2, EndReason::Surrender, Some(1), m(Some(20), Some(15)), true);
        assert_eq!(cols(&conceded), (s("Some(Loss)"), s("-1"), s("Concede"), s("")));
        // A collapse concede that coincides with the frame cap: RushJudge wins (메인 임시 결정).
        let both = play("rush", 2, EndReason::TimeLimitDraw, None, m(Some(3), Some(15)), true);
        assert_eq!(cols(&both), (s("Some(Loss)"), s("-1"), s("RushJudge"), s("3")));
        assert_eq!(both.buffer, 2, "a judged loss is not selected");
        // Another group's TimeLimitDraw stays a draw.
        let macro_draw = play("macro", 2, EndReason::TimeLimitDraw, None, m(Some(20), Some(15)), false);
        assert_eq!(cols(&macro_draw), (s("Some(Draw)"), s("0"), s("TimeLimitDraw"), s("")));
        assert_eq!((macro_draw.terminal_r_win, macro_draw.buffer), (Some(0.0), 2));
        assert!(macro_draw.win_delta.abs() < 1e-9, "a draw from the 0.5 prior");
        assert!(src.shared.sil.as_ref().unwrap().pending.lock().unwrap().is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }
}
