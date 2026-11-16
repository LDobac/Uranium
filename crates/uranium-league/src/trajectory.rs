use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use uranium_env::features::executor::HostAction;
use uranium_env::features::extract::privileged;
use uranium_env::features::PrivilegedTensors;
use uranium_env::reward::{Outcome, RewardParts};
use uranium_env::scout::{PseudoMode, ScoutMetrics};
use uranium_agent::model_policy::{Decision, SessionObserver};
use uranium_data::state::HostState;
use uranium_agent::infer::HEADS;

pub use uranium_data::pack::PackedObs;

use crate::config::QueueFull;

#[derive(Clone)]
pub struct Step {
    pub obs: PackedObs,
    pub action: HostAction,
    pub log_prob: f32,
    pub head_log_prob: [f32; HEADS],
    pub reward: [f32; 4],
    pub privileged: PrivilegedTensors,
    pub has_privileged: bool,
    pub version: u64,
}

#[derive(Clone)]
pub struct Segment {
    pub agent: usize,
    pub game_id: u64,
    pub steps: Vec<Step>,
    pub state0: HostState,
    pub terminal: bool,
    pub r_value_policy: bool,
    pub pseudo_scale: [f32; 2],
    pub start_frame: u32,
    pub sil_return: Option<f32>,
    pub enemy_lobby_race: uranium_proto::Race,
}

impl Segment {
    pub fn version(&self) -> u64 {
        self.steps.iter().map(|s| s.version).min().unwrap_or(0)
    }

    pub fn newest_version(&self) -> u64 {
        self.steps.iter().map(|s| s.version).max().unwrap_or(0)
    }

    pub fn trained(&self) -> usize {
        if self.terminal { self.steps.len() } else { self.steps.len().saturating_sub(1) }
    }
}

pub fn trained_steps(batch: &[Segment]) -> u64 {
    batch.iter().map(|s| s.trained() as u64).sum()
}

#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct QueueStats {
    pub pushed: u64,
    pub dropped_lag: u64,
    pub dropped_full: u64,
    pub revoked: u64,
    pub used_aborted: u64,
    pub waits: u64,
    pub wait_s: f64,
    pub wait_max_s: f64,
}

#[derive(Default)]
struct QueueInner {
    segs: VecDeque<Segment>,
    taken: HashMap<u64, u64>,
    revoked_games: HashSet<u64>,
    line: VecDeque<u64>,
    next_ticket: u64,
    generation: u64,
    stats: QueueStats,
    closed: bool,
}

pub struct SegmentQueue {
    inner: Mutex<QueueInner>,
    ready: Condvar,
    space: Condvar,
    capacity: usize,
    full: QueueFull,
    wait_max: Duration,
}

impl SegmentQueue {
    pub fn new(capacity: usize) -> Self {
        Self::with_policy(capacity, QueueFull::DropOldest, Duration::ZERO)
    }

    pub fn with_policy(capacity: usize, full: QueueFull, wait_max: Duration) -> Self {
        Self { inner: Mutex::new(QueueInner::default()), ready: Condvar::new(), space: Condvar::new(), capacity: capacity.max(1), full, wait_max }
    }

    pub fn push(&self, s: Segment) {
        let mut q = self.inner.lock().unwrap();
        if q.closed {
            return;
        }
        if q.revoked_games.contains(&s.game_id) {
            q.stats.revoked += 1;
            return;
        }
        if self.full == QueueFull::Wait && (q.segs.len() >= self.capacity || !q.line.is_empty()) {
            let ticket = q.next_ticket;
            q.next_ticket += 1;
            q.line.push_back(ticket);
            let generation = q.generation;
            let started = Instant::now();
            let deadline = started.checked_add(self.wait_max);
            let admitted = loop {
                if q.closed {
                    break false;
                }
                if q.revoked_games.contains(&s.game_id) {
                    q.stats.revoked += 1;
                    break false;
                }
                if q.generation != generation {
                    q.stats.dropped_lag += 1;
                    break false;
                }
                let now = Instant::now();
                let expired = deadline.is_some_and(|d| now >= d);
                if q.line.front() == Some(&ticket) && (q.segs.len() < self.capacity || expired) {
                    break true;
                }
                let wait = deadline.map_or(Duration::from_secs(1), |d| d.saturating_duration_since(now)).max(Duration::from_millis(100));
                q = self.space.wait_timeout(q, wait).unwrap().0;
            };
            q.line.retain(|t| *t != ticket);
            let waited = started.elapsed().as_secs_f64();
            q.stats.waits += 1;
            q.stats.wait_s += waited;
            q.stats.wait_max_s = q.stats.wait_max_s.max(waited);
            self.space.notify_all();
            if !admitted {
                return;
            }
        }
        q.stats.pushed += 1;
        q.segs.push_back(s);
        while q.segs.len() > self.capacity {
            q.segs.pop_front();
            q.stats.dropped_full += 1;
        }
        self.ready.notify_all();
    }

    pub fn revoke(&self, game_id: u64) {
        let mut q = self.inner.lock().unwrap();
        q.revoked_games.insert(game_id);
        let before = q.segs.len();
        q.segs.retain(|s| s.game_id != game_id);
        q.stats.revoked += (before - q.segs.len()) as u64;
        if let Some(n) = q.taken.remove(&game_id) {
            q.stats.used_aborted += n;
        }
        self.space.notify_all();
    }

    pub fn finish(&self, game_id: u64) {
        self.inner.lock().unwrap().taken.remove(&game_id);
    }

    pub fn reset_policy(&self) {
        let mut q = self.inner.lock().unwrap();
        q.stats.dropped_lag += q.segs.len() as u64;
        q.segs.clear();
        q.generation += 1;
        self.space.notify_all();
    }

    fn take_locked(&self, q: &mut QueueInner, lanes: usize, learner_version: u64, max_lag: u64) -> Option<Vec<Segment>> {
        let before = q.segs.len();
        q.segs.retain(|s| learner_version.saturating_sub(s.version()) <= max_lag && s.newest_version() <= learner_version);
        q.stats.dropped_lag += (before - q.segs.len()) as u64;
        let batch = (q.segs.len() >= lanes).then(|| q.segs.drain(..lanes).collect::<Vec<Segment>>());
        if let Some(b) = &batch {
            for s in b {
                *q.taken.entry(s.game_id).or_default() += 1;
            }
        }
        if q.segs.len() < before {
            self.space.notify_all();
        }
        batch
    }

    pub fn take(&self, lanes: usize, learner_version: u64, max_lag: u64) -> Option<Vec<Segment>> {
        let mut q = self.inner.lock().unwrap();
        loop {
            if q.closed {
                return None;
            }
            if let Some(b) = self.take_locked(&mut q, lanes, learner_version, max_lag) {
                return Some(b);
            }
            q = self.ready.wait_timeout(q, Duration::from_millis(500)).unwrap().0;
        }
    }

    pub fn try_take(&self, lanes: usize, learner_version: u64, max_lag: u64) -> Option<Vec<Segment>> {
        let mut q = self.inner.lock().unwrap();
        self.take_locked(&mut q, lanes, learner_version, max_lag)
    }

    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().segs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn waiting(&self) -> usize {
        self.inner.lock().unwrap().line.len()
    }

    pub fn stats(&self) -> QueueStats {
        self.inner.lock().unwrap().stats
    }

    pub fn close(&self) {
        let mut q = self.inner.lock().unwrap();
        q.closed = true;
        q.segs.clear();
        self.ready.notify_all();
        self.space.notify_all();
    }
}

#[derive(Debug, Clone)]
pub struct GameResult {
    pub agent: usize,
    pub game_id: u64,
    pub outcome: Option<Outcome>,
    pub reward_sum: [f32; 4],
    pub scout_sum: f32,
    pub survive_sum: f32,
    pub scout: Option<ScoutMetrics>,
    pub decisions: u64,
    pub enemy_race: Option<uranium_proto::Race>,
}

pub struct TrajectoryObserver {
    agent: usize,
    game_id: u64,
    seq_len: usize,
    queue: Arc<SegmentQueue>,
    results: Sender<GameResult>,
    steps: Vec<Step>,
    state0: Option<HostState>,
    next_state0: Option<HostState>,
    privileged: PrivilegedTensors,
    has_privileged: bool,
    reward_sum: [f32; 4],
    decisions: u64,
    enemy_race: Option<uranium_proto::Race>,
    hold: Option<Box<dyn Fn(Segment) + Send>>,
    r_value_policy: bool,
    pseudo: PseudoMode,
    pseudo_scale: [f32; 2],
    scout_sum: f32,
    survive_sum: f32,
    scout: Option<ScoutMetrics>,
    seg_frame: u32,
    enemy_lobby_race: uranium_proto::Race,
    copies: Option<SegmentSink>,
}

pub type SegmentSink = Box<dyn Fn(&Segment) + Send>;

pub fn channels(r: &RewardParts, mode: PseudoMode) -> [f32; 4] {
    match mode {
        PseudoMode::Z => [r.win, r.build, r.units, r.value],
        PseudoMode::Scout => [r.win, r.scout, r.survive, r.value],
    }
}

impl TrajectoryObserver {
    pub fn new(agent: usize, game_id: u64, seq_len: usize, queue: Arc<SegmentQueue>, results: Sender<GameResult>) -> Self {
        assert!(seq_len >= 2, "segments overlap by one step");
        let privileged = PrivilegedTensors { summary: vec![0.0; uranium_env::features::extract::dims().privileged()] };
        Self { agent, game_id, seq_len, queue, results, steps: Vec::new(), state0: None, next_state0: None, privileged, has_privileged: false, reward_sum: [0.0; 4], decisions: 0, enemy_race: None, hold: None, r_value_policy: true, pseudo: PseudoMode::Z, pseudo_scale: [1.0, 1.0], scout_sum: 0.0, survive_sum: 0.0, scout: None, seg_frame: 0, copies: None, enemy_lobby_race: uranium_proto::Race::Unknown }
    }

    pub fn copy_segments(mut self, sink: SegmentSink) -> Self {
        self.copies = Some(sink);
        self
    }

    pub fn hold_terminal(mut self, sink: Box<dyn Fn(Segment) + Send>) -> Self {
        self.hold = Some(sink);
        self
    }

    pub fn r_value_policy(mut self, on: bool) -> Self {
        self.r_value_policy = on;
        self
    }

    pub fn enemy_lobby_race(mut self, race: uranium_proto::Race) -> Self {
        self.enemy_lobby_race = race;
        self
    }

    pub fn pseudo(mut self, mode: PseudoMode, scale: [f32; 2]) -> Self {
        self.pseudo = mode;
        self.pseudo_scale = scale;
        self
    }

    fn add_reward(&mut self, r: &RewardParts, with_win: bool) {
        let mut c = channels(r, self.pseudo);
        let mut named = [r.win, r.build, r.units, r.value];
        if !with_win {
            (c[0], named[0]) = (0.0, 0.0);
        }
        if let Some(last) = self.steps.last_mut() {
            for k in 0..4 {
                last.reward[k] += c[k];
                self.reward_sum[k] += named[k];
            }
            self.scout_sum += r.scout;
            self.survive_sum += r.survive;
        }
    }
}

impl SessionObserver for TrajectoryObserver {
    fn wants_state(&self) -> bool {
        self.steps.is_empty() || self.steps.len() == self.seq_len - 1
    }

    fn on_privileged(&mut self, p: &uranium_proto::PrivilegedSummary) {
        self.privileged = privileged(p);
        self.has_privileged = true;
    }

    fn on_decision(&mut self, d: &Decision<'_>) {
        self.decisions += 1;
        {
            use uranium_env::features::{RACE_PROTOSS, RACE_TERRAN, RACE_ZERG};
            use uranium_proto::Race;
            match d.obs.obs.scalar_cat[0] {
                RACE_TERRAN => self.enemy_race = Some(Race::Terran),
                RACE_PROTOSS => self.enemy_race = Some(Race::Protoss),
                RACE_ZERG => self.enemy_race = Some(Race::Zerg),
                _ => {}
            }
        }
        self.add_reward(&d.reward_before, true);
        if self.steps.is_empty() {
            self.state0 = d.state.clone();
            self.seg_frame = d.frame;
        } else if self.steps.len() == self.seq_len - 1 {
            self.next_state0 = d.state.clone();
        }
        let step = Step {
            obs: PackedObs::pack(d.obs),
            action: d.action.clone(),
            log_prob: d.log_prob,
            head_log_prob: d.head_log_prob,
            reward: [0.0; 4],
            privileged: self.privileged.clone(),
            has_privileged: self.has_privileged,
            version: d.version,
        };
        self.steps.push(step);
        if self.steps.len() == self.seq_len {
            let last = self.steps.last().unwrap().clone();
            let steps = std::mem::replace(&mut self.steps, vec![last]);
            let state0 = self.state0.take().unwrap_or_default();
            self.state0 = self.next_state0.take();
            let seg = Segment { agent: self.agent, game_id: self.game_id, steps, state0, terminal: false, r_value_policy: self.r_value_policy, pseudo_scale: self.pseudo_scale, start_frame: self.seg_frame, sil_return: None, enemy_lobby_race: self.enemy_lobby_race };
            self.seg_frame = d.frame;
            if let Some(sink) = &self.copies {
                sink(&seg);
            }
            self.queue.push(seg);
        }
    }

    fn on_scout(&mut self, m: &ScoutMetrics) {
        self.scout = Some(*m);
    }

    fn on_end(&mut self, reward_after: RewardParts, outcome: Option<Outcome>) {
        match outcome {
            None => {
                self.steps.clear();
                self.queue.revoke(self.game_id);
            }
            Some(_) => {
                self.add_reward(&reward_after, self.hold.is_none());
                if !self.steps.is_empty() {
                    let steps = std::mem::take(&mut self.steps);
                    let state0 = self.state0.take().unwrap_or_default();
                    let seg = Segment { agent: self.agent, game_id: self.game_id, steps, state0, terminal: true, r_value_policy: self.r_value_policy, pseudo_scale: self.pseudo_scale, start_frame: self.seg_frame, sil_return: None, enemy_lobby_race: self.enemy_lobby_race };
                    match &self.hold {
                        Some(sink) => sink(seg),
                        None => self.queue.push(seg),
                    }
                }
                if self.hold.is_none() {
                    self.queue.finish(self.game_id);
                }
            }
        }
        let _ = self.results.send(GameResult {
            agent: self.agent,
            game_id: self.game_id,
            outcome,
            reward_sum: self.reward_sum,
            scout_sum: self.scout_sum,
            survive_sum: self.survive_sum,
            scout: self.scout,
            decisions: self.decisions,
            enemy_race: self.enemy_race,
        });
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::sync::mpsc;
    use uranium_env::features::Observation;

    pub(crate) fn fake_obs(i: usize) -> Observation {
        crate::testkit::observations(i % 3 + 1).pop().unwrap()
    }

    fn decide(obs: &mut TrajectoryObserver, i: usize, reward: f32) {
        let o = fake_obs(i);
        let a = HostAction { action: 0, delay_bucket: 0, queued: false, selected: vec![], target_unit: None, target_loc: None };
        let state = obs.wants_state().then(|| HostState { parts: vec![(vec![1, 1], vec![i as f32])] });
        obs.on_decision(&Decision {
            frame: i as u32,
            obs: &o,
            action: &a,
            log_prob: -1.0,
            head_log_prob: [0.0; HEADS],
            value: [0.0; 4],
            rejected: false,
            version: i as u64 / 10,
            reward_before: RewardParts { value: reward, ..Default::default() },
            state,
        });
    }

    #[test]
    fn packed_obs_round_trip_is_exact() {
        let o = fake_obs(3);
        let p = PackedObs::pack(&o);
        let u = p.unpack();
        assert_eq!(u.obs.spatial, o.obs.spatial);
        assert_eq!(u.masks.loc_class_mask, o.masks.loc_class_mask);
        assert_eq!(u.obs.scalar, o.obs.scalar);
    }

    #[test]
    fn segments_overlap_and_rewards_shift_042() {
        let q = Arc::new(SegmentQueue::new(16));
        let (tx, rx) = mpsc::channel();
        let mut o = TrajectoryObserver::new(0, 7, 4, q.clone(), tx);
        for i in 0..6 {
            decide(&mut o, i, i as f32);
        }
        o.on_end(RewardParts { win: 1.0, value: 0.5, ..Default::default() }, Some(Outcome::Win));
        let a = q.try_take(1, 0, 100).unwrap().remove(0);
        let b = q.try_take(1, 0, 100).unwrap().remove(0);
        assert!(q.try_take(1, 0, 100).is_none());
        assert_eq!(a.steps.len(), 4);
        assert!(!a.terminal);
        assert_eq!(a.trained(), 3);
        assert_eq!(a.state0.parts[0].1, vec![0.0]);
        assert_eq!(a.steps.iter().take(3).map(|s| s.reward[3]).collect::<Vec<_>>(), vec![1.0, 2.0, 3.0]);
        assert_eq!(b.state0.parts[0].1, vec![3.0]);
        assert_eq!(b.steps.len(), 3);
        assert!(b.terminal);
        assert_eq!(b.steps.iter().map(|s| s.reward).collect::<Vec<_>>(), vec![[0.0, 0.0, 0.0, 4.0], [0.0, 0.0, 0.0, 5.0], [1.0, 0.0, 0.0, 0.5]]);
        let r = rx.recv().unwrap();
        assert_eq!(r.outcome, Some(Outcome::Win));
        assert_eq!(r.decisions, 6);
        let (a, b) = (std::slice::from_ref(&a), std::slice::from_ref(&b));
        assert_eq!(trained_steps(a) + trained_steps(b), r.decisions);
        assert_eq!(r.reward_sum, [1.0, 0.0, 0.0, 15.5]);
        assert_eq!(r.enemy_race, Some(uranium_proto::Race::Zerg));
    }

    #[test]
    fn privileged_presence_reaches_the_batch_042_r21() {
        let q = Arc::new(SegmentQueue::new(16));
        let (tx, _rx) = mpsc::channel();
        let mut o = TrajectoryObserver::new(0, 7, 4, q.clone(), tx);
        decide(&mut o, 0, 0.0);
        o.on_privileged(&uranium_proto::PrivilegedSummary::default());
        decide(&mut o, 1, 0.0);
        decide(&mut o, 2, 0.0);
        o.on_end(RewardParts::default(), Some(Outcome::Win));
        let seg = q.try_take(1, 0, 100).unwrap().remove(0);
        assert_eq!(seg.steps.iter().map(|s| s.has_privileged).collect::<Vec<_>>(), vec![false, true, true]);
        assert!(seg.steps[0].privileged.summary.iter().all(|&x| x == 0.0));
        let g = crate::batch::group(std::slice::from_ref(&seg), 4, 12);
        assert_eq!(g.priv_present, vec![false, true, true, true]);
        assert_eq!(g.real, vec![true, true, true, false]);
    }

    #[test]
    fn segments_carry_the_r_value_flag_033() {
        let q = Arc::new(SegmentQueue::new(16));
        for (game, on) in [(1u64, false), (2, true)] {
            let (tx, _rx) = mpsc::channel();
            let mut o = TrajectoryObserver::new(0, game, 3, q.clone(), tx);
            if !on {
                o = o.r_value_policy(false);
            }
            for i in 0..4 {
                decide(&mut o, i, 1.0);
            }
            o.on_end(RewardParts::default(), Some(Outcome::Draw));
        }
        let segs = q.try_take(4, 0, 100).unwrap();
        assert_eq!(segs.iter().map(|s| (s.game_id, s.r_value_policy)).collect::<Vec<_>>(), [(1, false), (1, false), (2, true), (2, true)]);
        assert!(segs.iter().all(|s| s.steps.iter().take(s.trained().saturating_sub(1)).all(|st| st.reward[3] == 1.0)));
        let g = crate::batch::group(&segs, 3, 12);
        assert_eq!(g.r_value_policy, [false, false, true, true]);
    }

    #[test]
    fn segments_carry_the_lobby_race_044_r9() {
        use uranium_proto::Race;
        let q = Arc::new(SegmentQueue::new(32));
        let games = [(1u64, Some(Race::Zerg)), (2, Some(Race::Terran)), (3, Some(Race::Protoss)), (4, Some(Race::Random)), (5, Some(Race::Unknown)), (6, None)];
        for (game, race) in games {
            let (tx, _rx) = mpsc::channel();
            let mut o = TrajectoryObserver::new(0, game, 3, q.clone(), tx);
            if let Some(r) = race {
                o = o.enemy_lobby_race(r);
            }
            for i in 0..2 {
                decide(&mut o, i, 0.0);
            }
            o.on_end(RewardParts::default(), Some(Outcome::Draw));
        }
        let segs = q.try_take(6, 0, 100).unwrap();
        assert_eq!(segs.iter().map(|s| s.enemy_lobby_race).collect::<Vec<_>>(), [Race::Zerg, Race::Terran, Race::Protoss, Race::Random, Race::Unknown, Race::Unknown]);
        assert_eq!(crate::batch::group(&segs, 3, 12).ref_race, [0, 1, 2, 3, 3, 3]);
        let order: Vec<&Segment> = [4, 0, 2].iter().map(|&i| &segs[i]).collect();
        assert_eq!(crate::batch::group_of(&order, 3, 12).ref_race, [3, 0, 2]);
        assert_eq!(crate::batch::REF_RACES, ["zerg", "terran", "protoss", "random"]);
    }

    fn decide_parts(obs: &mut TrajectoryObserver, i: usize, r: RewardParts) {
        let o = fake_obs(i);
        let a = HostAction { action: 0, delay_bucket: 0, queued: false, selected: vec![], target_unit: None, target_loc: None };
        let state = obs.wants_state().then(|| HostState { parts: vec![(vec![1, 1], vec![i as f32])] });
        obs.on_decision(&Decision { frame: i as u32, obs: &o, action: &a, log_prob: -1.0, head_log_prob: [0.0; HEADS], value: [0.0; 4], rejected: false, version: 0, reward_before: r, state });
    }

    #[test]
    fn scout_mode_channels_and_scale_033_r11() {
        let r = |k: f32| RewardParts { win: 0.0, build: 0.5 * k, units: 0.25 * k, value: 0.01 * k, scout: 0.1 * k, survive: 0.02 * k };
        let run = |mode: PseudoMode, scale: [f32; 2], game: u64| {
            let q = Arc::new(SegmentQueue::new(16));
            let (tx, rx) = mpsc::channel();
            let mut o = TrajectoryObserver::new(0, game, 3, q.clone(), tx).pseudo(mode, scale);
            for i in 0..4 {
                decide_parts(&mut o, i, r(i as f32));
            }
            let m = ScoutMetrics { main_frame: Some(4_000), types_4m: 3, ..Default::default() };
            o.on_scout(&m);
            o.on_end(RewardParts { win: 1.0, ..r(4.0) }, Some(Outcome::Win));
            let segs = std::iter::from_fn(|| q.try_take(1, 0, 100).map(|mut v| v.remove(0))).collect::<Vec<_>>();
            (segs, rx.recv().unwrap())
        };
        let (segs, res) = run(PseudoMode::Scout, [0.5, 0.0], 1);
        let steps: Vec<[f32; 4]> = segs.iter().flat_map(|s| s.steps[..s.trained()].iter().map(|st| st.reward)).collect();
        let want = |k: f32, win: f32| [win, 0.1 * k, 0.02 * k, 0.01 * k];
        assert_eq!(steps, [want(1.0, 0.0), want(2.0, 0.0), want(3.0, 0.0), want(4.0, 1.0)]);
        assert!(segs.iter().all(|s| s.pseudo_scale == [0.5, 0.0]));
        assert_eq!(crate::batch::group(&segs, 3, 12).pseudo_scale, [0.5, 0.0, 0.5, 0.0]);
        let close = |a: f32, b: f32| (a - b).abs() < 1e-6;
        assert!(close(res.reward_sum[1], 5.0) && close(res.reward_sum[2], 2.5), "r_build / r_units logged by name: {:?}", res.reward_sum);
        assert!(close(res.scout_sum, 1.0) && close(res.survive_sum, 0.2), "{} {}", res.scout_sum, res.survive_sum);
        assert_eq!(res.scout.map(|m| (m.main_frame, m.types_4m)), Some((Some(4_000), 3)));
        let (segs, res) = run(PseudoMode::Z, [1.0, 1.0], 2);
        let steps: Vec<[f32; 4]> = segs.iter().flat_map(|s| s.steps[..s.trained()].iter().map(|st| st.reward)).collect();
        assert_eq!(steps[0], [0.0, 0.5, 0.25, 0.01], "z: r_build / r_units");
        assert!(segs.iter().all(|s| s.pseudo_scale == [1.0, 1.0]));
        assert!(close(res.scout_sum, 1.0), "the raw sums are kept in z too (games.csv leaves them empty)");
        let q = Arc::new(SegmentQueue::new(4));
        let (tx, _rx) = mpsc::channel();
        let mut o = TrajectoryObserver::new(0, 3, 3, q.clone(), tx);
        decide_parts(&mut o, 0, r(1.0));
        o.on_end(r(1.0), Some(Outcome::Draw));
        let s = q.try_take(1, 0, 100).unwrap().remove(0);
        assert_eq!((s.pseudo_scale, s.steps[0].reward), ([1.0, 1.0], [0.0, 0.5, 0.25, 0.01]));
    }

    #[test]
    fn aborted_game_is_revoked_042_r3() {
        let q = Arc::new(SegmentQueue::new(16));
        let (tx, rx) = mpsc::channel();
        let mut o = TrajectoryObserver::new(0, 9, 3, q.clone(), tx);
        for i in 0..7 {
            decide(&mut o, i, 0.0);
        }
        assert_eq!(q.len(), 3);
        let _used = q.try_take(1, 0, 100).unwrap();
        o.on_end(RewardParts::default(), None);
        assert_eq!(q.len(), 0);
        let st = q.stats();
        assert_eq!((st.revoked, st.used_aborted), (2, 1));
        assert_eq!(rx.recv().unwrap().outcome, None);
    }

    #[test]
    fn stale_segments_are_dropped_042_r4() {
        let q = SegmentQueue::new(16);
        for v in [0, 5, 40, 41] {
            q.push(seg(v, v));
        }
        let b = q.try_take(2, 72, 32).unwrap();
        assert_eq!(b.iter().map(|s| s.game_id).collect::<Vec<_>>(), vec![40, 41]);
        assert_eq!(q.stats().dropped_lag, 2);
    }

    fn seg(game_id: u64, version: u64) -> Segment {
        seg_at(game_id, version)
    }

    pub(crate) fn seg_at(game_id: u64, version: u64) -> Segment {
        static OBS: std::sync::OnceLock<PackedObs> = std::sync::OnceLock::new();
        Segment {
            agent: 0,
            game_id,
            steps: vec![Step {
                obs: OBS.get_or_init(|| PackedObs::pack(&fake_obs(0))).clone(),
                action: HostAction { action: 0, delay_bucket: 0, queued: false, selected: vec![], target_unit: None, target_loc: None },
                log_prob: 0.0,
                head_log_prob: [0.0; HEADS],
                reward: [0.0; 4],
                privileged: PrivilegedTensors { summary: vec![] },
                has_privileged: false,
                version,
            }],
            state0: HostState::default(),
            terminal: true,
            r_value_policy: true,
            pseudo_scale: [1.0, 1.0],
            start_frame: 0,
            sil_return: None,
            enemy_lobby_race: uranium_proto::Race::Unknown,
        }
    }

    #[test]
    fn segments_carry_start_frames_and_copies_042_r16() {
        let q = Arc::new(SegmentQueue::new(16));
        let (tx, _rx) = mpsc::channel();
        let copied = Arc::new(Mutex::new(Vec::new()));
        let c = copied.clone();
        let mut o = TrajectoryObserver::new(0, 5, 3, q.clone(), tx).copy_segments(Box::new(move |s: &Segment| c.lock().unwrap().push((s.start_frame, s.steps.len(), s.terminal))));
        for i in 0..6 {
            decide(&mut o, i, 0.0);
        }
        o.on_end(RewardParts::default(), Some(Outcome::Loss));
        let segs = q.try_take(3, 0, 100).unwrap();
        assert_eq!(segs.iter().map(|s| (s.start_frame, s.terminal, s.sil_return)).collect::<Vec<_>>(), [(0, false, None), (2, false, None), (4, true, None)]);
        assert_eq!(*copied.lock().unwrap(), [(0, 3, false), (2, 3, false)], "terminal not copied here");
    }

    fn wait_queue(capacity: usize) -> Arc<SegmentQueue> {
        Arc::new(SegmentQueue::with_policy(capacity, QueueFull::Wait, Duration::from_secs(60)))
    }

    fn push_in_thread(q: &Arc<SegmentQueue>, s: Segment) -> (std::thread::JoinHandle<()>, mpsc::Receiver<u64>) {
        let (tx, rx) = mpsc::channel();
        let q = q.clone();
        let game_id = s.game_id;
        let h = std::thread::spawn(move || {
            q.push(s);
            tx.send(game_id).unwrap();
        });
        (h, rx)
    }

    fn until_waiting(q: &SegmentQueue, n: usize) {
        let end = Instant::now() + Duration::from_secs(10);
        while q.waiting() != n {
            assert!(Instant::now() < end, "expected {n} waiting pushes, have {}", q.waiting());
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    const BLOCKED: Duration = Duration::from_millis(100);
    const RELEASED: Duration = Duration::from_secs(10);

    #[test]
    fn full_queue_push_waits_until_take_042() {
        let q = wait_queue(2);
        q.push(seg(1, 0));
        q.push(seg(2, 0));
        let (h, rx) = push_in_thread(&q, seg(3, 0));
        until_waiting(&q, 1);
        assert!(rx.recv_timeout(BLOCKED).is_err(), "the push must wait while the queue is full");
        assert_eq!(q.try_take(1, 0, 100).unwrap()[0].game_id, 1);
        assert_eq!(rx.recv_timeout(RELEASED).unwrap(), 3);
        h.join().unwrap();
        let b = q.try_take(2, 0, 100).unwrap();
        assert_eq!(b.iter().map(|s| s.game_id).collect::<Vec<_>>(), vec![2, 3]);
        let st = q.stats();
        assert_eq!((st.pushed, st.dropped_full, st.waits), (3, 0, 1));
        assert!(st.wait_s > 0.0 && st.wait_max_s == st.wait_s);
    }

    #[test]
    fn waiting_pushes_go_in_fifo_order_042() {
        let q = wait_queue(1);
        q.push(seg(0, 0));
        let mut waiters = Vec::new();
        for (n, id) in [10u64, 11, 12, 13].into_iter().enumerate() {
            waiters.push(push_in_thread(&q, seg(id, 0)));
            until_waiting(&q, n + 1);
        }
        let mut order = Vec::new();
        let end = Instant::now() + RELEASED;
        while order.len() < 5 {
            assert!(Instant::now() < end, "taken so far {order:?}");
            match q.try_take(1, 0, 100) {
                Some(b) => order.push(b[0].game_id),
                None => std::thread::sleep(Duration::from_millis(2)),
            }
        }
        assert_eq!(order, vec![0, 10, 11, 12, 13]);
        for (h, _) in waiters {
            h.join().unwrap();
        }
        assert_eq!(q.stats().dropped_full, 0);
    }

    #[test]
    fn revoke_cancels_a_waiting_push_042_r3() {
        let q = wait_queue(1);
        q.push(seg(1, 0));
        let (ha, rxa) = push_in_thread(&q, seg(5, 0));
        until_waiting(&q, 1);
        let (hb, rxb) = push_in_thread(&q, seg(6, 0));
        until_waiting(&q, 2);
        q.revoke(5);
        assert_eq!(rxa.recv_timeout(RELEASED).unwrap(), 5);
        ha.join().unwrap();
        assert!(rxb.recv_timeout(BLOCKED).is_err(), "game 6 still waits for a place");
        assert_eq!(q.len(), 1);
        assert_eq!(q.stats().revoked, 1);
        assert_eq!(q.try_take(1, 0, 100).unwrap()[0].game_id, 1);
        assert_eq!(rxb.recv_timeout(RELEASED).unwrap(), 6);
        hb.join().unwrap();
        q.push(seg(5, 0));
        assert_eq!(q.try_take(1, 0, 100).unwrap()[0].game_id, 6);
        assert!(q.is_empty());
        assert_eq!(q.stats().revoked, 2);
    }

    #[test]
    fn push_of_a_revoked_game_is_ignored_042_r3() {
        for q in [wait_queue(4), Arc::new(SegmentQueue::new(4))] {
            q.revoke(9);
            q.push(seg(9, 0));
            q.push(seg(8, 0));
            assert_eq!(q.len(), 1);
            let st = q.stats();
            assert_eq!((st.pushed, st.revoked), (1, 1));
        }
    }

    #[test]
    fn close_releases_waiting_pushes_042_r10() {
        let q = wait_queue(1);
        q.push(seg(1, 0));
        let mut waiters = Vec::new();
        for id in 2..5u64 {
            waiters.push(push_in_thread(&q, seg(id, 0)));
            until_waiting(&q, id as usize - 1);
        }
        q.close();
        for (h, rx) in waiters {
            rx.recv_timeout(RELEASED).unwrap();
            h.join().unwrap();
        }
        assert!(q.is_empty());
        assert_eq!(q.waiting(), 0);
        q.push(seg(7, 0));
        assert!(q.is_empty(), "closed queue accepts nothing");
    }

    #[test]
    fn lag_drop_wakes_a_waiting_push_042_r4() {
        let q = wait_queue(1);
        q.push(seg(1, 0));
        let (h, rx) = push_in_thread(&q, seg(2, 90));
        until_waiting(&q, 1);
        assert!(q.try_take(2, 100, 32).is_none());
        assert_eq!(rx.recv_timeout(RELEASED).unwrap(), 2);
        h.join().unwrap();
        assert_eq!(q.try_take(1, 100, 32).unwrap()[0].game_id, 2);
        assert_eq!(q.stats().dropped_lag, 1);
    }

    #[test]
    fn wait_cap_drops_the_oldest_042() {
        let q = SegmentQueue::with_policy(2, QueueFull::Wait, Duration::from_millis(50));
        q.push(seg(1, 0));
        q.push(seg(2, 0));
        let t = Instant::now();
        q.push(seg(3, 0));
        assert!(t.elapsed() >= Duration::from_millis(50));
        let b = q.try_take(2, 0, 100).unwrap();
        assert_eq!(b.iter().map(|s| s.game_id).collect::<Vec<_>>(), vec![2, 3]);
        let st = q.stats();
        assert_eq!((st.pushed, st.dropped_full, st.waits), (3, 1, 1));
        assert!(st.wait_s >= 0.05);
    }

    #[test]
    fn chained_wait_caps_keep_fifo_order_042() {
        let q = Arc::new(SegmentQueue::with_policy(2, QueueFull::Wait, Duration::from_secs(1)));
        q.push(seg(1, 0));
        q.push(seg(2, 0));
        let segs: Vec<Segment> = [10u64, 11, 12].into_iter().map(|id| seg(id, 0)).collect();
        let mut waiters = Vec::new();
        for (n, s) in segs.into_iter().enumerate() {
            waiters.push(push_in_thread(&q, s));
            until_waiting(&q, n + 1);
        }
        for (h, rx) in waiters {
            rx.recv_timeout(RELEASED).unwrap();
            h.join().unwrap();
        }
        let b = q.try_take(2, 0, 100).unwrap();
        assert_eq!(b.iter().map(|s| s.game_id).collect::<Vec<_>>(), vec![11, 12]);
        let st = q.stats();
        assert_eq!((st.pushed, st.dropped_full, st.waits), (5, 3, 3));
    }

    #[test]
    fn expired_waiter_behind_the_front_does_not_jump_042() {
        let q = Arc::new(SegmentQueue::with_policy(1, QueueFull::Wait, Duration::from_secs(1)));
        q.push(seg(1, 0));
        let (s10, s11) = (seg(10, 0), seg(11, 0));
        let (ha, rxa) = push_in_thread(&q, s10);
        until_waiting(&q, 1);
        let (hb, rxb) = push_in_thread(&q, s11);
        until_waiting(&q, 2);
        {
            let _hold = q.inner.lock().unwrap();
            std::thread::sleep(Duration::from_millis(1_500));
        }
        for rx in [rxa, rxb] {
            rx.recv_timeout(RELEASED).unwrap();
        }
        ha.join().unwrap();
        hb.join().unwrap();
        assert_eq!(q.try_take(1, 0, 100).unwrap()[0].game_id, 11);
        let st = q.stats();
        assert_eq!((st.dropped_full, st.waits), (2, 2));
    }

    #[test]
    fn reset_policy_drops_the_previous_policy_042_r4() {
        let q = wait_queue(2);
        q.push(seg(1, 4_000));
        q.push(seg(2, 4_000));
        let (h, rx) = push_in_thread(&q, seg(3, 4_000));
        until_waiting(&q, 1);
        q.reset_policy();
        assert_eq!(rx.recv_timeout(RELEASED).unwrap(), 3);
        h.join().unwrap();
        assert!(q.is_empty());
        assert_eq!(q.stats().dropped_lag, 3);
        q.push(seg(4, 4_000));
        q.push(seg(5, 0));
        assert_eq!(q.try_take(1, 0, 32).unwrap()[0].game_id, 5);
        assert!(q.is_empty());
        assert_eq!(q.stats().dropped_lag, 4);
    }

    #[test]
    fn drop_oldest_mode_never_waits_042() {
        let q = SegmentQueue::with_policy(2, QueueFull::DropOldest, Duration::from_secs(60));
        for id in 1..=3 {
            q.push(seg(id, 0));
        }
        let b = q.try_take(2, 0, 100).unwrap();
        assert_eq!(b.iter().map(|s| s.game_id).collect::<Vec<_>>(), vec![2, 3]);
        let st = q.stats();
        assert_eq!((st.dropped_full, st.waits), (1, 0));
    }
}
