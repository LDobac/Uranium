//! `uranium-train rl` (042): one process runs the learner, the inference servers, the agent
//! server, the game dispatch (`uranium-runner`, 020-R12) and the league.
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
use burn::module::Module;
use burn::tensor::Device;
use serde::{Deserialize, Serialize};
use uranium_agent::infer::{InferBackend, InferOptions, InferenceHandle, SampleMode};
use uranium_agent::model_policy::{LearnerHook, SessionObserver};
use uranium_agent::{EndBoard, PolicySpec, Routes, T0Build, T0Builds, game_result};
use uranium_env::features::Strategy;
use uranium_env::reward::{Outcome, RewardWeights};
use uranium_model::checkpoint::{self, CheckpointMeta};
use uranium_model::CoreInit;
use uranium_proto::control::{GameSpec, MapKey, PlayerKind, PlayerSpec};
use uranium_proto::{EndReason, GameInfo, Race};
use uranium_runner::{Dispatch, HostEntry, Job, JobOutcome, JobSource};

use super::config::RlConfig;
use super::league::{AgentKind, Class, League, Opponent, Pick, T1Bot};
use super::learner::{Algo, HEAD_NAMES, Learner, UpdateStats};
use super::trajectory::{GameResult, Segment, SegmentQueue, TrajectoryObserver};
use crate::ckpt::{self, TrainState};
use crate::storage::{LockStatus, PushOutcome, Run, StorageConfig};

/// One entry of the z table (033-R8): a replay's strategy statistic.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ZEntry {
    pub build: Vec<u16>,
    pub units: Vec<f32>,
}

pub struct RlArgs {
    pub run: String,
    pub bc: PathBuf,
    pub config: PathBuf,
    pub algo: Algo,
    pub backend: InferBackend,
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
    /// Checkpoint every this many minutes (061-R4: 30).
    pub checkpoint_minutes: u64,
    pub max_batch: usize,
    pub max_wait_ms: u64,
    /// 042-AC1 small setting: always this T0 race on Fighting Spirit, Main only.
    pub fixed_t0: Option<Race>,
}

const STOP_NONE: usize = 0;
const STOP_CLEAN: usize = 1;

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
    infer: InferOptions,
    games_csv: Mutex<std::fs::File>,
    agent_steps: AtomicU64,
    games_done: AtomicU64,
    disagree: AtomicU64,
    aborted: AtomicU64,
    max_frames: u32,
}

/// Inference servers of league snapshots, least recently used first out.
struct SnapshotPool {
    dirs: HashMap<String, PathBuf>,
    live: VecDeque<(String, InferenceHandle)>,
    cap: usize,
}

impl SnapshotPool {
    fn handle(&mut self, id: &str, opts: InferOptions) -> Result<InferenceHandle> {
        if let Some(i) = self.live.iter().position(|(k, _)| k == id) {
            let e = self.live.remove(i).unwrap();
            self.live.push_back(e.clone());
            return Ok(e.1);
        }
        let dir = self.dirs.get(id).with_context(|| format!("unknown snapshot {id}"))?;
        let h = InferenceHandle::start(dir, SampleMode::Stochastic, opts)?;
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
        let bridge = |race| PlayerSpec { race, kind: PlayerKind::Bridge { agent_addr: s.agent_addr.clone() } };
        let mut t0_builds = [T0Build::Standard; 2];
        let mut policies: [Option<PolicySpec>; 2] = [Some(PolicySpec::Learner(self.hooks[pick.agent.index()].clone())), None];
        let (p1, enemy_race, opponent) = match &pick.opponent {
            Opponent::T0 { race, build } => {
                t0_builds[1] = *build;
                (bridge(*race), *race, format!("{}@{}", pick.key, env!("CARGO_PKG_VERSION")))
            }
            Opponent::T1 { name, race } => (PlayerSpec { race: *race, kind: PlayerKind::OssBot { name: name.clone() } }, *race, pick.key.clone()),
            Opponent::SelfPlay => {
                policies[1] = Some(PolicySpec::Model(s.handles[AgentKind::Main.index()].clone()));
                (bridge(Race::Terran), Race::Terran, pick.key.clone())
            }
            Opponent::Snapshot { id } => {
                let h = s.snapshots.lock().unwrap().handle(id, s.infer)?;
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
            save_replay: false,
            wall_timeout_s: 3600,
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
                eprintln!("[rl] cannot build a game: {e:#}");
                None
            }
        }
    }

    fn requeue(&self, job: Job) {
        // A dynamic source simply forgets a game that never started.
        self.shared.games.lock().unwrap().remove(&job.spec.game_id);
    }

    fn done(&self, job: &Job, outcome: JobOutcome) {
        let s = &self.shared;
        let id = job.spec.game_id;
        let Some(meta) = s.games.lock().unwrap().remove(&id) else { return };
        let agent = meta.pick.agent.index();
        let (result, held) = self.collect(id);
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
                    eprintln!("[rl] curriculum advanced to {:?} (042-R5)", s.league.lock().unwrap().stage);
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
        let (sum, decisions) = result.as_ref().map_or(([0.0; 4], 0), |r| (r.reward_sum, r.decisions));
        s.agent_steps.fetch_add(decisions, Ordering::SeqCst);
        s.games_done.fetch_add(1, Ordering::SeqCst);
        let line = format!(
            "{},{},{},{},{:?},{},{},{:?},{},{},{},{},{},{}\n",
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
            verdict.as_ref().is_some_and(|v| v.1),
            decisions,
            r_win + sum[0],
            sum[1],
            sum[2],
            sum[3]
        );
        let _ = s.games_csv.lock().unwrap().write_all(line.as_bytes());
    }
}

/// 042-R2 train.csv header / row.
fn train_header() -> String {
    let mut h = String::from("time_s,agent,update,policy_version,agent_steps,games,steps,loss,policy,value,kl,kl_coef,rho_mean,adv_mean,lag_mean");
    for n in HEAD_NAMES {
        h += &format!(",kl_{n}");
    }
    for n in HEAD_NAMES {
        h += &format!(",entropy_{n}");
    }
    h += ",dropped_lag,dropped_full,revoked,used_aborted,winner_disagree,aborted_games,steps_per_s,stage\n";
    h
}

fn load_z(path: &Option<PathBuf>) -> Result<Vec<ZEntry>> {
    let Some(p) = path else { return Ok(vec![]) };
    let text = std::fs::read_to_string(p).with_context(|| format!("reading z table {}", p.display()))?;
    text.lines().filter(|l| !l.trim().is_empty()).map(|l| Ok(serde_json::from_str(l)?)).collect()
}

/// Everything the checkpoint needs from the learners (061-R10).
fn save_checkpoint<C: CoreInit + 'static>(run: &mut Run, step: u64, learners: &[Learner<C>], league: &League, args: &RlArgs, state_extra: serde_json::Value) -> Result<PushOutcome> {
    let dir = run.stage_dir(step)?;
    for (i, l) in learners.iter().enumerate() {
        let name = AgentKind::ALL[i].name();
        let mut meta = CheckpointMeta::new(&l.net_cfg, uranium_env::features::FEATURE_SCHEMA_VERSION, uranium_env::actions::ACTION_SCHEMA_VERSION)?;
        meta.train = serde_json::json!({ "rl": l.cfg, "algo": l.algo, "update": l.update, "agent": name });
        checkpoint::save(&ckpt::agent_dir(&dir, name)?, &l.net.valid(), &meta)?;
        ckpt::save_optimizer(&dir, name, l.optimizer())?;
    }
    ckpt::save_league(&dir, &league.to_bytes())?;
    let updates: Vec<u64> = learners.iter().map(|l| l.update).collect();
    ckpt::save_state(&dir, &TrainState { kind: "rl".into(), step, update: updates[0], policy_version: updates[0], rng: vec![], extra: state_extra })?;
    let reward = PathBuf::from("config/reward.toml");
    let mut files: Vec<&Path> = vec![&args.config];
    if reward.is_file() {
        files.push(&reward);
    }
    ckpt::copy_configs(&dir, &files)?;
    run.seal(step)?;
    run.push(step)
}

pub fn run<C: CoreInit + 'static>(args: RlArgs, dev: Device) -> Result<()> {
    let cfg = RlConfig::load(&args.config)?;
    std::fs::create_dir_all(&args.log_dir)?;
    let ad = dev.autodiff();
    let cols = uranium_model::batch::ActionColumns::from_table(uranium_env::features::extract::action_table())?;
    let fv = uranium_env::features::FEATURE_SCHEMA_VERSION;
    let av = uranium_env::actions::ACTION_SCHEMA_VERSION;

    // 061-R3 / R5: lock the run, recover or pull LATEST.
    let mut run = Run::acquire(StorageConfig::from_env()?, &args.run, args.take_over)?;
    let resume = run.begin()?;
    let (bc, bc_meta) = checkpoint::load_with::<C>(&args.bc, &cols, fv, av, &ad, rl_config).context("loading the BC checkpoint (042 start / KL reference)")?;
    let net_cfg = bc_meta.net_config()?;

    let mut learners: Vec<Learner<C>> = Vec::new();
    for a in AgentKind::ALL {
        let mut l = Learner::new(bc.clone(), bc.clone(), net_cfg.clone(), cfg.clone(), args.algo, ad.clone());
        if let Some(r) = &resume {
            let dir = ckpt::agent_dir(&r.dir, a.name())?;
            let (net, meta) = checkpoint::load_with::<C>(&dir, &cols, fv, av, &ad, rl_config)?;
            l.net = net;
            l.update = meta.train["update"].as_u64().unwrap_or(0);
            if ckpt::has_optimizer(&r.dir, a.name()) {
                let o = ckpt::load_optimizer(&r.dir, a.name(), super::learner::new_optimizer(&cfg))?;
                l.set_optimizer(o);
            }
        }
        learners.push(l);
    }
    let t1 = cfg_t1(&args.config)?;
    let mut league = match resume.as_ref().map(|r| ckpt::load_league(&r.dir)).transpose()?.flatten() {
        Some(b) => League::from_bytes(&b, cfg.clone())?,
        None => League::new(&args.run, t1, args.seed, cfg.clone()),
    };
    if args.fixed_t0.is_some() {
        league.fixed_t0 = args.fixed_t0;
    }
    let mut step = resume.as_ref().map_or(0, |r| r.step);
    if let Some(r) = &resume {
        eprintln!("[rl] resumed {} at step {} (recovered {}), stage {:?}", args.run, r.step, r.recovered, league.stage);
    }

    // Inference servers start from the learners' current weights (written to the cache).
    let mut infer = InferOptions::new(args.backend);
    infer.max_batch = args.max_batch.max(1);
    infer.max_wait = Duration::from_millis(args.max_wait_ms);
    let infer_root = run.cache_dir().join("infer");
    let mut handles = Vec::new();
    for (i, l) in learners.iter().enumerate() {
        let dir = infer_root.join(AgentKind::ALL[i].name());
        let meta = CheckpointMeta::new(&net_cfg, fv, av)?;
        checkpoint::save(&dir, &l.net.valid(), &meta)?;
        let h = InferenceHandle::start(&dir, SampleMode::Stochastic, infer)?;
        h.swap(l.snapshot(), l.update)?;
        handles.push(h);
    }
    let snap_dirs: HashMap<String, PathBuf> = league
        .snapshots
        .iter()
        .map(|(a, id)| (id.clone(), run.shared_dir().join("league").join(snapshot_name(id)).join(ckpt::AGENTS_DIR).join(a.name())))
        .collect();

    let queues: Vec<Arc<SegmentQueue>> = (0..3).map(|_| Arc::new(SegmentQueue::new(cfg.actors.queue_batches * cfg.lanes))).collect();
    let (tx, rx) = mpsc::channel();
    let routes = Routes::default();
    let builds = T0Builds::default();
    let ends = EndBoard::default();
    let listen = uranium_runner::env_or("URANIUM_AGENT_LISTEN", "127.0.0.1:7600");
    let agent_addr = uranium_runner::start_agent(&listen, &args.log_dir.join("sessions.jsonl"), routes.clone(), builds.clone(), ends.clone())?;
    let games_csv_path = args.log_dir.join("games.csv");
    let new_games = !games_csv_path.exists();
    let mut games_csv = std::fs::OpenOptions::new().create(true).append(true).open(&games_csv_path)?;
    if new_games {
        games_csv.write_all(b"game_id,agent,opponent,class,enemy_race,map,hostd_outcome,outcome,winner_disagree,decisions,r_win,r_build,r_units,r_value\n")?;
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
        snapshots: Mutex::new(SnapshotPool { dirs: snap_dirs, live: VecDeque::new(), cap: 8 }),
        infer,
        games_csv: Mutex::new(games_csv),
        agent_steps: AtomicU64::new(0),
        games_done: AtomicU64::new(0),
        disagree: AtomicU64::new(0),
        aborted: AtomicU64::new(0),
        max_frames: 86_400,
    });
    let z = load_z(&args.z_table)?;
    if z.is_empty() {
        eprintln!("[rl] no z table: every game runs without z, pseudo-rewards off (033-R8)");
    }
    let hooks: Vec<Arc<Hook>> = AgentKind::ALL
        .iter()
        .map(|&a| Arc::new(Hook { agent: a, handle: handles[a.index()].clone(), seq_len: cfg.seq_len, shared: shared.clone(), tx: tx.clone(), z: z.clone(), z_drop: 0.25, rng: Mutex::new(args.seed ^ (a.index() as u64 + 1)) }))
        .collect();
    let source = Arc::new(LeagueSource { shared: shared.clone(), hooks });
    let dispatch = Dispatch::new(routes, Default::default(), builds);
    let hosts: Vec<HostEntry> = uranium_runner::hosts()?;

    // 042-R10: first Ctrl+C = clean stop, second = exit now.
    let stop = Arc::new(AtomicUsize::new(STOP_NONE));
    {
        let stop = stop.clone();
        ctrlc::set_handler(move || {
            if stop.fetch_add(1, Ordering::SeqCst) >= STOP_CLEAN {
                eprintln!("[rl] second Ctrl+C: exiting without saving (042-R10)");
                std::process::exit(130);
            }
            eprintln!("[rl] Ctrl+C: stopping games, saving and pushing the checkpoint (042-R10)");
        })?;
    }
    let games_thread = {
        let (source, dispatch, games) = (source.clone(), dispatch.clone(), args.log_dir.join("games.jsonl"));
        std::thread::spawn(move || uranium_runner::run_source(source, &hosts, &games, &dispatch))
    };

    let base = RewardWeights::load()?;
    let mut csv = std::fs::OpenOptions::new().create(true).append(true).open(args.log_dir.join("train.csv"))?;
    if csv.metadata()?.len() == 0 {
        csv.write_all(train_header().as_bytes())?;
    }
    let started = Instant::now();
    let (mut last_ckpt, mut last_beat) = (Instant::now(), Instant::now());
    let mut lock_lost = false;
    let mut updates_this_run = 0u64;
    let mut last_rate = (Instant::now(), 0u64);
    let mut steps_per_s = 0.0f64;
    loop {
        let limit_hit = (args.max_updates > 0 && updates_this_run >= args.max_updates) || (args.max_minutes > 0 && started.elapsed() >= Duration::from_secs(args.max_minutes * 60));
        if stop.load(Ordering::SeqCst) >= STOP_CLEAN || limit_hit {
            break;
        }
        // 061-R5: heartbeat every minute; a lost lock stops training.
        if last_beat.elapsed() >= Duration::from_secs(60) {
            last_beat = Instant::now();
            match run.heartbeat() {
                LockStatus::Held => {}
                LockStatus::Lost(l) => {
                    eprintln!("[rl] lock lost to {l:?}: stopping (061-R5)");
                    lock_lost = true;
                    break;
                }
                LockStatus::Unreachable(e) => eprintln!("[rl] heartbeat: share unreachable ({e}), continuing (061-R11-3)"),
            }
        }
        let ready: Vec<AgentKind> = AgentKind::ALL.iter().copied().filter(|a| queues[a.index()].len() >= cfg.lanes).collect();
        let turn = shared.league.lock().unwrap().learner_turn(&ready);
        let Some(agent) = turn else {
            std::thread::sleep(Duration::from_millis(200));
            continue;
        };
        let i = agent.index();
        let Some(batch) = queues[i].try_take(cfg.lanes, learners[i].update, cfg.actors.max_policy_lag) else { continue };
        let weights = shared.league.lock().unwrap().reward_weights(&base);
        let stats = learners[i].update(&batch, weights);
        drop(batch);
        updates_this_run += 1;
        let snap = shared.league.lock().unwrap().on_update(agent);
        // 042 K: weights to the actors every `weight_push_every` updates.
        if learners[i].update % cfg.actors.weight_push_every.max(1) == 0 {
            handles[i].swap(learners[i].snapshot(), learners[i].update)?;
        }
        step = step.max(shared.agent_steps.load(Ordering::SeqCst));
        if last_rate.0.elapsed() >= Duration::from_secs(60) {
            let now = shared.agent_steps.load(Ordering::SeqCst);
            steps_per_s = (now - last_rate.1) as f64 / last_rate.0.elapsed().as_secs_f64();
            last_rate = (Instant::now(), now);
        }
        write_row(&mut csv, started, agent, &stats, &shared, &queues, learners[i].update, steps_per_s)?;
        if let Some(reset) = snap {
            // A league snapshot is a pushed checkpoint copied to league/<name> (061-R2).
            let id = shared.league.lock().unwrap().add_snapshot(agent);
            let league_now = shared.league.lock().unwrap().clone();
            step += 1;
            match save_checkpoint(&mut run, step, &learners, &league_now, &args, serde_json::json!({ "games": shared.games_done.load(Ordering::SeqCst) }))? {
                PushOutcome::Pushed => {
                    let dir = run.league_snapshot(step, &snapshot_name(&id))?;
                    shared.snapshots.lock().unwrap().dirs.insert(id.clone(), dir.join(ckpt::AGENTS_DIR).join(agent.name()));
                    eprintln!("[rl] snapshot {id}");
                }
                other => eprintln!("[rl] snapshot {id} not published: {other:?}"),
            }
            if reset {
                learners[i].reset_to_bc();
                handles[i].swap(learners[i].snapshot(), learners[i].update)?;
            }
            last_ckpt = Instant::now();
        }
        if last_ckpt.elapsed() >= Duration::from_secs(args.checkpoint_minutes.max(1) * 60) {
            step += 1;
            let league_now = shared.league.lock().unwrap().clone();
            let out = save_checkpoint(&mut run, step, &learners, &league_now, &args, serde_json::json!({ "games": shared.games_done.load(Ordering::SeqCst) }))?;
            eprintln!("[rl] checkpoint {} {:?}", crate::storage::ckpt_name(step), out);
            run.prune().ok();
            last_ckpt = Instant::now();
        }
    }

    // 042-R10: ① no new games ② stop games in flight (Aborted, revoked) ③ checkpoint + push
    // ④ release the lock only when the push went through (061-R11-5/6).
    shared.stopping.store(true, Ordering::SeqCst);
    dispatch.inflight.stop_all();
    for q in &queues {
        q.close();
    }
    let _ = games_thread.join();
    if lock_lost {
        eprintln!("[rl] lock lost: no final push, lock left as is (061-R11-4)");
        return Ok(());
    }
    step = step.max(shared.agent_steps.load(Ordering::SeqCst)) + 1;
    let league_now = shared.league.lock().unwrap().clone();
    let out = save_checkpoint(&mut run, step, &learners, &league_now, &args, serde_json::json!({ "games": shared.games_done.load(Ordering::SeqCst) }))?;
    eprintln!("[rl] final checkpoint {}: {out:?}", crate::storage::ckpt_name(step));
    match out {
        PushOutcome::Pushed => run.release()?,
        other => eprintln!("[rl] push {other:?}: keeping the lock so the next start recovers it (061-R11-5/6)"),
    }
    Ok(())
}

/// 032 §2: dropout 0 in RL (0.1 is the BC setting). The actors run without dropout, so a
/// learner with dropout would see π ≠ μ even on-policy (V-trace ρ, KL to π_BC).
fn rl_config(mut c: uranium_model::UraniumNetConfig) -> uranium_model::UraniumNetConfig {
    c.entity.dropout = 0.0;
    c
}

/// Snapshot folder name (061-R9 forbids `/` and `@`): `<agent>-<run>-<update>`.
pub fn snapshot_name(id: &str) -> String {
    id.trim_start_matches("uranium-").replace(['@', '/'], "-")
}

#[allow(clippy::too_many_arguments)]
fn write_row(csv: &mut std::fs::File, started: Instant, agent: AgentKind, s: &UpdateStats, shared: &Shared, queues: &[Arc<SegmentQueue>], version: u64, steps_per_s: f64) -> Result<()> {
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
        ",{},{},{},{},{},{},{:.2},{:?}\n",
        q.dropped_lag,
        q.dropped_full,
        q.revoked,
        q.used_aborted,
        shared.disagree.load(Ordering::SeqCst),
        shared.aborted.load(Ordering::SeqCst),
        steps_per_s,
        stage
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

    /// 032 §2: RL nets are built without dropout.
    #[test]
    fn rl_nets_have_no_dropout_032() {
        let c = rl_config(uranium_model::UraniumNetConfig::new());
        assert_eq!(c.entity.dropout, 0.0);
    }

    #[test]
    fn train_csv_header_columns() {
        assert_eq!(train_header().trim().split(',').count(), 15 + 12 + 8);
    }
}
