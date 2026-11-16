use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use anyhow::Result;
use uranium_agent::infer::InferenceHandle;
use uranium_agent::{ConcedeBoard, EndBoard, Routes, T0Builds};
use uranium_match::concede::ConcedeConfig;
use uranium_env::reward::RewardWeights;
use uranium_env::scout::PseudoMode;
use uranium_proto::Race;
use uranium_runner::{Dispatch, HostEntry};
use uranium_store::ckpt;
use crate::config::{LeagueConfig, RlConfig};
use crate::league::{AgentKind, League, Pick};
use crate::learner::{Algo, RlLearners, SnapshotServers};
use crate::sil::{SilBuffer, SilConfig};
use uranium_store::storage::{LockStatus, PushOutcome, Run, StorageConfig};
use crate::trajectory::{GameResult, Segment, SegmentQueue, trained_steps as trained_steps_of};

mod checkpoint;
mod csv;
mod hook;
mod pool;
mod seed;
mod source;
mod watch;
#[cfg(test)]
mod testkit;

pub use checkpoint::snapshot_name;
use pool::{SnapshotPool, prune_servers};
use hook::Hook;
use source::{LeagueSource, scout_stage_warning};
use watch::{DEAD_LEARNER_GAMES, BAD_SESSION_STREAK, StallWatch};
use csv::{GAMES_HEADER, open_train_csv, open_csv, write_row};
use checkpoint::{checkpoint_pseudo, check_pseudo, state_extra, resumed_trained_steps, load_z, save_checkpoint, check_storage, prune_logged, finish_run};

pub struct RlArgs {
    pub run: String,
    pub config: PathBuf,
    pub algo: Algo,
    pub seed: u64,
    pub take_over: bool,
    pub log_dir: PathBuf,
    pub z_table: Option<PathBuf>,
    pub max_updates: u64,
    pub max_minutes: u64,
    pub max_agent_steps: u64,
    pub checkpoint_minutes: u64,
    pub fixed_t0: Option<Race>,
    pub stop: Arc<AtomicUsize>,
}

struct GameMeta {
    pick: Pick,
    self_player: Option<u8>,
    enemy_race: Race,
    server: Option<u64>,
    slots: Option<[u8; 2]>,
}

struct Shared {
    league: Mutex<League>,
    games: Mutex<HashMap<u64, GameMeta>>,
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
    bad_sessions: AtomicU64,
    finished_games: AtomicU64,
    league_cfg: LeagueConfig,
    sil: Option<SilState>,
    game_wall_timeout_s: u32,
    replay_dir: PathBuf,
    replay_every: u64,
    concede_cfg: ConcedeConfig,
    concede: ConcedeBoard,
    pseudo: PseudoMode,
}

struct SilState {
    cfg: SilConfig,
    pending: Mutex<HashMap<u64, Vec<Segment>>>,
    buffer: Mutex<SilBuffer>,
    off_stage_logged: AtomicBool,
}

impl SilState {
    fn new(cfg: SilConfig, seed: u64) -> Self {
        let buffer = Mutex::new(SilBuffer::new(cfg.buffer_segments, seed ^ 0x5115_5115));
        Self { cfg, pending: Mutex::new(HashMap::new()), buffer, off_stage_logged: AtomicBool::new(false) }
    }

    fn in_stage(&self, stage: crate::league::Stage) -> bool {
        if stage == crate::league::Stage::C1 {
            return true;
        }
        if !self.off_stage_logged.swap(true, Ordering::SeqCst) {
            uranium_store::elog!("[rl] [sil] is on but the stage is {stage:?}: no self-imitation lanes outside C1 (042-R16)");
        }
        false
    }

    fn keep(&self, seg: &Segment) {
        if self.cfg.keeps_segment(seg.start_frame) {
            self.pending.lock().unwrap().entry(seg.game_id).or_default().push(seg.clone());
        }
    }

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

pub fn run(args: RlArgs, make: impl FnOnce(&RlConfig, Option<&Path>) -> Result<(Box<dyn RlLearners>, SnapshotServers)>) -> Result<()> {
    let stop = args.stop.clone();
    let cfg = RlConfig::load(&args.config)?;
    check_pseudo(&cfg, args.z_table.as_deref(), None)?;
    std::fs::create_dir_all(&args.log_dir)?;

    let storage = StorageConfig::from_env()?;
    let replay_dir = storage.shared.join("runs").join(&args.run).join("replays");
    let anchor_dir = match cfg.league.anchor_name() {
        Some(_) => Some(crate::config::anchor_agent_dir(&storage.shared, &cfg.league.anchor)?),
        None => None,
    };
    if uranium_store::stopping(&stop) {
        uranium_store::elog!("[rl] Ctrl+C before start: not started");
        return Ok(());
    }
    let mut storage = storage;
    storage.keep_latest = cfg.keep_checkpoints.unwrap_or(uranium_data::bc::DEFAULT_KEEP_CHECKPOINTS);
    let mut run = Run::acquire(storage, &args.run, args.take_over)?;
    check_storage(&run);
    let resume = match run.begin() {
        Ok(r) => r,
        Err(e) if e.downcast_ref::<uranium_store::Interrupted>().is_some() => {
            uranium_store::elog!("[rl] {e:#}; releasing the lock (042-R10)");
            run.release()?;
            return Ok(());
        }
        Err(e) => return Err(e),
    };
    if uranium_store::stopping(&stop) {
        uranium_store::elog!("[rl] Ctrl+C during start: releasing the lock, nothing to save (042-R10)");
        run.release()?;
        return Ok(());
    }
    if let Err(e) = check_pseudo(&cfg, None, resume.as_ref().and_then(|r| checkpoint_pseudo(&r.dir))) {
        let _ = run.release();
        return Err(e);
    }
    let (mut learners, servers) = match make(&cfg, resume.as_ref().map(|r| r.dir.as_path())) {
        Ok(x) => x,
        Err(e) => {
            let _ = run.release();
            return Err(e.context("starting the learners (042 start / KL reference)"));
        }
    };
    let mut league = match resume.as_ref().map(|r| ckpt::load_league(&r.dir)).transpose()?.flatten() {
        Some(b) => League::from_bytes(&b, cfg.clone())?,
        None => League::new(&args.run, args.seed, args.fixed_t0, cfg.clone()),
    };
    if args.fixed_t0.is_some() {
        league.fixed_t0 = args.fixed_t0;
    }
    let mut step = resume.as_ref().map_or(0, |r| r.step);
    let mut trained_steps = match &resume {
        Some(r) => ckpt::load_state(&r.dir)?.map_or(0, |s| resumed_trained_steps(&s.extra)),
        None => 0,
    };
    let cc = &cfg.curriculum;
    match &resume {
        Some(r) => uranium_store::elog!("[rl] resumed {} at step {} (recovered {}), stage {:?}", args.run, r.step, r.recovered, league.stage),
        None => uranium_store::elog!("[rl] new league {}, stage {:?}", args.run, league.stage),
    }
    uranium_store::elog!("[rl] curriculum: auto_advance {}, r_value in {:?}, {} T1 bot(s) (042-R5, 033)", cc.auto_advance, cc.r_value_stages, cfg.t1.len());

    let handles: Vec<InferenceHandle> = AgentKind::ALL.iter().map(|a| learners.handle(a.index())).collect();
    let mut snap_dirs: HashMap<String, PathBuf> = HashMap::new();
    for (a, id) in league.snapshots.clone() {
        let d = run.shared_dir().join("league").join(snapshot_name(&id)).join(ckpt::AGENTS_DIR).join(a.name());
        if crate::config::has_model_files(&d) {
            league.mark_published(&id);
            snap_dirs.insert(id, d);
        } else {
            uranium_store::elog!("[rl] snapshot {id}: no model files in {}, not an opponent (042)", d.display());
        }
    }
    let anchor_key = league.anchor_key();
    if let (Some(k), Some(d)) = (&anchor_key, &anchor_dir) {
        snap_dirs.insert(k.clone(), d.clone());
    }
    uranium_store::elog!(
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
    uranium_store::elog!("[rl] C1 rush ladder: share {} ({:?}) {:?} (042, AMB-042-15)", cc.c1_ladder, cc.ladder_pick, ladder);
    if !cc.ladder_groups.is_empty() {
        let groups: Vec<String> = cc.ladder_groups.iter().map(|(g, x)| format!("{g} {x} ({} entries)", cfg.ladder.iter().filter(|e| &e.group == g).count())).collect();
        uranium_store::elog!("[rl] ladder groups: {} (042, AMB-042-16)", groups.join(", "));
    }
    uranium_store::elog!("[rl] win table decay {} (042 승률 표)", cfg.league.win_rate_decay);
    let b = &cfg.curriculum.band;
    uranium_store::elog!("[rl] ladder pick {:?}, band center {} width {} exclude below {} (042, AMB-042-17)", cc.ladder_pick, b.center, b.width, b.exclude_below);
    match cfg.league.model_game_max_s {
        0 => uranium_store::elog!("[rl] games against a model: no time cap (60 minutes, 020-R3)"),
        s => uranium_store::elog!("[rl] games against a model end as a draw at {s} game seconds ({} frames, 042-R15)", cfg.league.game_max_frames(true)),
    }
    let rj = &cfg.league.rush_judge;
    if rj.groups.is_empty() {
        uranium_store::elog!("[rl] rush mini-game off (042-R17)");
    } else {
        uranium_store::elog!("[rl] rush mini-game: ladder groups {:?} end at {} game seconds ({} frames); a TimeLimitDraw there is a win with >= {} own SCVs, else a loss (042-R17)", rj.groups, rj.at_s, rj.max_frames(), rj.min_scv);
    }
    let weights: Vec<String> = cfg.ladder.iter().filter(|e| e.weight.is_some()).filter_map(|e| e.opponent().ok().map(|(_, k, _)| format!("{k} {}", e.fixed_weight()))).collect();
    if !weights.is_empty() {
        uranium_store::elog!("[rl] \"fixed\" ladder weights: {} (absent = 1.0, 042-R19)", weights.join(", "));
    }
    let sm = &cfg.league.start_mix;
    match sm.rush_hit_share {
        None => uranium_store::elog!("[rl] start condition mix off: every game keeps its hash seed (042-R18)"),
        Some(x) => uranium_store::elog!("[rl] start condition mix: ladder groups {:?} take a seed whose first-target hit (020-R21) matches a target drawn with hit share {x} (042-R18)", sm.groups),
    }
    let k = &cfg.kl;
    uranium_store::elog!("[rl] KL coefficient max({}, {} x 0.5^(update / {})) from this run's update 0, head multipliers {:?}; td_lambda {} (042-R9)", k.floor, k.init, k.half_life_updates, k.head_scale.by_head(), cfg.loss.td_lambda);
    let sc = &cfg.sil;
    if sc.enabled {
        uranium_store::elog!(
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
        uranium_store::elog!("[rl] self-imitation off (042-R16)");
    }
    let c = &cfg.concede;
    uranium_store::elog!("[rl] collapse concede: {} (screen supply <= {} for {} s from {} s, 042-R14)", if c.enabled { "on" } else { "off" }, c.max_supply, c.hold_s, c.after_s);
    let rw = &cfg.reward;
    match rw.pseudo {
        PseudoMode::Z => uranium_store::elog!("[rl] reward channels: z (r_build / r_units, 033-R11)"),
        PseudoMode::Scout => {
            uranium_store::elog!(
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
                uranium_store::elog!("{w}");
            }
        }
    }

    let wait_max = Duration::from_secs(cfg.actors.queue_wait_max_s);
    let queues: Vec<Arc<SegmentQueue>> = (0..3).map(|_| Arc::new(SegmentQueue::with_policy(cfg.actors.queue_batches * cfg.lanes, cfg.actors.queue_full, wait_max))).collect();
    let (tx, rx) = mpsc::channel();
    let routes = Routes::default();
    let builds = T0Builds::default();
    let ends = EndBoard::default();
    let listen = uranium_runner::env_or("URANIUM_AGENT_LISTEN", "127.0.0.1:7600");
    let concede = ConcedeBoard::default();
    let agent_addr = uranium_runner::start_agent(&listen, &args.log_dir.join("sessions.jsonl"), &args.log_dir.join("unit_rejections.jsonl"), routes.clone(), builds.clone(), ends.clone(), concede.clone(), Default::default())?;
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
        uranium_store::elog!("[rl] no z table: every game runs without z, pseudo-rewards off (033-R8)");
    }
    let hooks: Vec<Arc<Hook>> = AgentKind::ALL
        .iter()
        .map(|&a| Arc::new(Hook { agent: a, handle: handles[a.index()].clone(), seq_len: cfg.seq_len, shared: shared.clone(), tx: tx.clone(), z: z.clone(), z_drop: 0.25, rng: Mutex::new(args.seed ^ (a.index() as u64 + 1)), reward: cfg.reward.clone() }))
        .collect();
    let source = Arc::new(LeagueSource { shared: shared.clone(), hooks });
    let dispatch = Dispatch::new(routes, Default::default(), builds);
    let halt = Halt { shared: shared.clone(), inflight: dispatch.inflight.clone(), done: AtomicBool::new(false) };
    let hosts: Vec<HostEntry> = uranium_runner::hosts()?;

    let base = RewardWeights::load()?;
    let mut csv = open_train_csv(&args.log_dir.join("train.csv"))?;
    let games_thread = {
        let (source, dispatch, games) = (source.clone(), dispatch.clone(), args.log_dir.join("games.jsonl"));
        std::thread::spawn(move || uranium_runner::run_source(source, &hosts, &games, &dispatch))
    };

    let started = Instant::now();
    let mut wait_seen = [0.0f64; 3];
    let (mut last_ckpt, mut last_beat) = (Instant::now(), Instant::now());
    let mut lock_lost = false;
    let mut updates_this_run = 0u64;
    let mut last_rate = (Instant::now(), shared.agent_steps.load(Ordering::SeqCst));
    let mut steps_per_s = 0.0f64;
    let mut stall = StallWatch::new(cfg.actors.dead_learner_minutes, Instant::now(), 0);
    match cfg.actors.dead_learner_minutes {
        0 => uranium_store::elog!("[rl] dead learner check off ([actors] dead_learner_minutes = 0, 042-R20)"),
        m => uranium_store::elog!("[rl] dead learner check: abnormal end after {DEAD_LEARNER_GAMES}+ finished games without an update for {m} min, or {BAD_SESSION_STREAK} finished games in a row without a learning-session decision (042-R20)"),
    }
    let mut steps_reached = false;
    loop {
        let steps_hit = args.max_agent_steps > 0 && shared.agent_steps.load(Ordering::SeqCst) >= args.max_agent_steps;
        if steps_hit {
            uranium_store::elog!("[rl] {} agent steps reached: clean stop (042-R12)", args.max_agent_steps);
            steps_reached = true;
        }
        let limit_hit = steps_hit || (args.max_updates > 0 && updates_this_run >= args.max_updates) || (args.max_minutes > 0 && started.elapsed() >= Duration::from_secs(args.max_minutes * 60));
        if uranium_store::stopping(&stop) || limit_hit {
            break;
        }
        if last_beat.elapsed() >= Duration::from_secs(60) {
            last_beat = Instant::now();
            match run.heartbeat() {
                LockStatus::Held => {}
                LockStatus::Lost(l) => {
                    uranium_store::elog!("[rl] lock lost to {l:?}: stopping (061-R5)");
                    lock_lost = true;
                    break;
                }
                LockStatus::Unreachable(e) => uranium_store::elog!("[rl] heartbeat: share unreachable ({e}), continuing (061-R11-3)"),
            }
        }
        if let Some(why) = stall.check(Instant::now(), shared.finished_games.load(Ordering::SeqCst), shared.bad_sessions.load(Ordering::SeqCst)) {
            uranium_store::elog!("[rl] {why}: ending abnormally without a checkpoint, the watchdog restarts from LATEST (042-R20)");
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
        trained_steps += trained;
        updates_this_run += 1;
        let snap = shared.league.lock().unwrap().on_update(agent);
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
            let id = shared.league.lock().unwrap().add_snapshot(agent);
            let league_now = shared.league.lock().unwrap().clone();
            step += 1;
            match save_checkpoint(&mut run, step, learners.as_mut(), &cfg, &league_now, &args, state_extra(&shared, trained_steps))? {
                PushOutcome::Pushed => match run.league_snapshot(step, &snapshot_name(&id), agent.name()) {
                    Ok(dir) => {
                        shared.snapshots.lock().unwrap().dirs.insert(id.clone(), dir.join(ckpt::AGENTS_DIR).join(agent.name()));
                        shared.league.lock().unwrap().mark_published(&id);
                        uranium_store::elog!("[rl] snapshot {id}");
                    }
                    Err(e) => uranium_store::elog!("[rl] snapshot {id} not published ({e:#}): not an opponent (042)"),
                },
                other => uranium_store::elog!("[rl] snapshot {id} not published: {other:?}"),
            }
            prune_servers(&shared);
            if reset {
                learners.reset_to_bc(i)?;
                queues[i].reset_policy();
                learners.push(i)?;
            }
            last_ckpt = Instant::now();
        }
        if last_ckpt.elapsed() >= Duration::from_secs(args.checkpoint_minutes.max(1) * 60) {
            step += 1;
            let league_now = shared.league.lock().unwrap().clone();
            let out = save_checkpoint(&mut run, step, learners.as_mut(), &cfg, &league_now, &args, state_extra(&shared, trained_steps))?;
            uranium_store::elog!("[rl] checkpoint {} {:?}", uranium_store::storage::ckpt_name(step), out);
            if out == PushOutcome::Pushed {
                prune_logged(&mut run);
            }
            last_ckpt = Instant::now();
        }
        stall.progress(Instant::now(), shared.finished_games.load(Ordering::SeqCst));
    }

    halt.stop();
    let _ = games_thread.join();
    if lock_lost {
        uranium_store::elog!("[rl] lock lost: no final push, lock left as is (061-R11-4)");
        return Ok(());
    }
    step = step.max(shared.agent_steps.load(Ordering::SeqCst)) + 1;
    let league_now = shared.league.lock().unwrap().clone();
    let out = save_checkpoint(&mut run, step, learners.as_mut(), &cfg, &league_now, &args, state_extra(&shared, trained_steps))?;
    uranium_store::elog!("[rl] final checkpoint {}: {out:?}", uranium_store::storage::ckpt_name(step));
    finish_run(run, out, steps_reached, step)
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use uranium_runner::JobSource;
    use crate::league::Class;
    use crate::run::seed::{base_seed, game_seed};
    use crate::run::testkit::{fake_servers, source_cfg};

    #[test]
    fn rl_toml_lists_t1_bots_021() {
        let bots = RlConfig::load(Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../config/rl.toml"))).unwrap().t1;
        assert!(bots.iter().any(|b| b.name == "zzzkbot" && b.races == vec![Race::Zerg]));
    }

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

    #[test]
    fn sil_lane_swap_042_r16() {
        use crate::league::Stage;
        use crate::sil::SilMode;
        use crate::trajectory::tests::seg_at;
        let fill = |q: &SegmentQueue, ids: std::ops::Range<u64>, version: u64| ids.for_each(|i| q.push(seg_at(i, version)));
        for mode in [SilMode::Sil, SilMode::Replay] {
            let sil = SilState::new(SilConfig { enabled: true, mode, lanes: 2, buffer_segments: 100, max_uses: 2, ..SilConfig::default() }, 1);
            let q = SegmentQueue::new(64);
            sil.buffer.lock().unwrap().push_game(vec![seg_at(900, 0)], 1.0);
            fill(&q, 0..4, 100);
            let (b, trained, row) = take_batch(&q, Some(&sil), AgentKind::Main, Stage::C1, 4, 100, 32).unwrap();
            assert_eq!((b.len(), trained, row), (4, 4, Some((1, 0))));
            assert!(b.iter().all(|s| s.sil_return.is_none()));
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
            fill(&q, 30..40, 100);
            for (agent, stage) in [(AgentKind::Main, Stage::C2), (AgentKind::Main, Stage::C0), (AgentKind::MainExploiter, Stage::C1)] {
                let (b, trained, row) = take_batch(&q, Some(&sil), agent, stage, 4, 100, 32).unwrap();
                assert!(b.iter().all(|s| s.sil_return.is_none()) && b.len() == 4 && trained == 4, "{agent:?} {stage:?}");
                assert_eq!(row.unwrap().1, 0);
            }
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
        let q = SegmentQueue::new(8);
        fill(&q, 0..4, 0);
        let (b, trained, row) = take_batch(&q, None, AgentKind::Main, Stage::C1, 4, 0, 32).unwrap();
        assert_eq!((b.len(), trained, row), (4, 4, None));
        assert!(take_batch(&q, None, AgentKind::Main, Stage::C1, 4, 0, 32).is_none(), "too few");
    }

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
}
