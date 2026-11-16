use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use uranium_agent::infer::{InferenceHandle, InferOptions, SampleMode};
use uranium_agent::{ConcedeBoard, EndBoard, PolicySpec};
use uranium_env::reward::Outcome;
use uranium_env::scout::ScoutMetrics;
use uranium_proto::Race;
use uranium_runner::Job;
use crate::config::RlConfig;
use crate::league::{AgentKind, League};
use crate::learner::{SnapshotServers, SnapshotServer};
use crate::trajectory::{GameResult, SegmentQueue};
use super::{Shared, SilState};
use super::pool::SnapshotPool;
use super::hook::Hook;
use super::source::LeagueSource;

fn fake_handle(id: &str) -> InferenceHandle {
    let (tx, _rx) = mpsc::channel();
    InferenceHandle::new(tx, id.into(), SampleMode::Stochastic, &InferOptions::default())
}

pub(super) fn fake_servers(fail: bool) -> (SnapshotServers, Arc<Mutex<Vec<PathBuf>>>, Arc<AtomicUsize>) {
    let (s, o, c, _) = fake_servers_with(Arc::new(AtomicBool::new(fail)));
    (s, o, c)
}

type Alive = Arc<Mutex<Vec<Arc<AtomicBool>>>>;

pub(super) fn fake_servers_with(fail: Arc<AtomicBool>) -> (SnapshotServers, Arc<Mutex<Vec<PathBuf>>>, Arc<AtomicUsize>, Alive) {
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

pub(super) fn source(c1: [f64; 3], anchor: &str, candidates: usize, servers: SnapshotServers) -> LeagueSource {
    let mut cfg = RlConfig::default_file();
    cfg.curriculum.start_stage = crate::league::Stage::C1;
    let cc = &mut cfg.curriculum;
    (cc.c1_t0, cc.c1_t1, cc.c1_self, cc.c1_snapshot, cc.c1_anchor) = (0.0, 0.0, c1[0], c1[1], c1[2]);
    cfg.league.anchor = anchor.into();
    cfg.snapshot.main_candidates = candidates;
    let csv = PathBuf::from(if cfg!(windows) { "NUL" } else { "/dev/null" });
    source_cfg(cfg, servers, &csv)
}

pub(super) fn source_cfg(cfg: RlConfig, servers: SnapshotServers, csv: &Path) -> LeagueSource {
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

pub(super) fn opponent_handle(job: &Job) -> String {
    match &job.policies[1] {
        Some(PolicySpec::Model(h)) => h.checkpoint_id.to_string(),
        other => panic!("{other:?}"),
    }
}

pub(super) fn result(id: u64, scout: Option<ScoutMetrics>) -> GameResult {
    GameResult { agent: 0, game_id: id, outcome: Some(Outcome::Win), reward_sum: [0.0, 0.0, 0.0, 0.25], scout_sum: 0.6, survive_sum: 0.375, scout, decisions: 9, enemy_race: Some(Race::Zerg) }
}
