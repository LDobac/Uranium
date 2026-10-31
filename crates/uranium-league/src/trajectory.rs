//! RL trajectories (042): a learning session's decisions become segments of T steps that go to
//! the learner through a queue. Segments overlap by one step: the last step of a non-terminal
//! segment only bootstraps the value (it is trained as the first step of the next segment), so
//! no decision is lost and no extra forward is needed. A terminal segment trains all its steps
//! and bootstraps from 0 (γ = 1, episode over).
//!
//! 042-R3 (AMB-042-6): segments are queued as soon as they are complete; when the game turns out
//! Aborted, the ones still in the queue are revoked. 042-R4: segments whose behaviour version
//! lags the learner by more than `max_policy_lag` updates are dropped when a batch is taken.
//! A full queue applies backpressure (042 queue table, `queue_full = "wait"`): the game whose
//! segment does not fit waits for a place, in arrival order, at most `queue_wait_max_s`.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use uranium_env::features::executor::HostAction;
use uranium_env::features::extract::privileged;
use uranium_env::features::PrivilegedTensors;
use uranium_env::reward::{Outcome, RewardParts};
use uranium_agent::model_policy::{Decision, SessionObserver};
use uranium_data::state::HostState;
use uranium_agent::infer::HEADS;

/// Observations with their two large planes compressed (042), shared with BC (041-R18).
pub use uranium_data::pack::PackedObs;

use crate::config::QueueFull;

/// One decision of a learning session.
#[derive(Clone)]
pub struct Step {
    pub obs: PackedObs,
    pub action: HostAction,
    pub log_prob: f32,
    pub head_log_prob: [f32; HEADS],
    /// win, build, units, value (033-R3); known once the next decision (or the end) arrives.
    pub reward: [f32; 4],
    pub privileged: PrivilegedTensors,
    pub version: u64,
}

/// T steps (or fewer at the end of a game) for one lane of a learner batch.
pub struct Segment {
    pub agent: usize,
    pub game_id: u64,
    pub steps: Vec<Step>,
    /// Core state before `steps[0]` (042 core_state0).
    pub state0: HostState,
    /// The episode ended after the last step: train every step, bootstrap 0.
    pub terminal: bool,
}

impl Segment {
    /// Oldest behaviour version in the segment (042-R4).
    pub fn version(&self) -> u64 {
        self.steps.iter().map(|s| s.version).min().unwrap_or(0)
    }

    /// Newest behaviour version in the segment (042-R4, exploiter reset).
    pub fn newest_version(&self) -> u64 {
        self.steps.iter().map(|s| s.version).max().unwrap_or(0)
    }

    /// Steps the learner trains on.
    pub fn trained(&self) -> usize {
        if self.terminal { self.steps.len() } else { self.steps.len().saturating_sub(1) }
    }
}

/// 042-R2 `trained_steps`: agent steps a batch trains on (each decision once, the overlap
/// step of a non-terminal segment only bootstraps).
pub fn trained_steps(batch: &[Segment]) -> u64 {
    batch.iter().map(|s| s.trained() as u64).sum()
}

/// Counters of the queue (042-R2 logging).
#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct QueueStats {
    pub pushed: u64,
    /// 042-R4: behaviour version too old.
    pub dropped_lag: u64,
    /// Queue full: oldest dropped (`drop_oldest`, or a wait that reached `queue_wait_max_s`).
    pub dropped_full: u64,
    /// 042-R3: revoked because their game was Aborted (queued, waiting, or arriving late).
    pub revoked: u64,
    /// 042-R3: segments of an Aborted game that the learner had already used.
    pub used_aborted: u64,
    /// Pushes that waited for a place (042 queue table, backpressure).
    pub waits: u64,
    /// 042-R2 `queue_wait_s`: seconds those pushes waited, summed (counted when a wait ends).
    pub wait_s: f64,
    /// Longest single wait in seconds.
    pub wait_max_s: f64,
}

#[derive(Default)]
struct QueueInner {
    segs: VecDeque<Segment>,
    /// Segments already taken per game still running (for `used_aborted`).
    taken: HashMap<u64, u64>,
    /// 042-R3: games revoked so far; their pushes (waiting or late) are discarded.
    revoked_games: HashSet<u64>,
    /// Tickets of the pushes waiting for a place, in arrival order: only the front one may
    /// insert, so waiting games go FIFO (a Condvar alone wakes in no particular order).
    line: VecDeque<u64>,
    next_ticket: u64,
    /// Bumped by `reset_policy`: a push waiting from an older generation is not queued.
    generation: u64,
    stats: QueueStats,
    closed: bool,
}

/// One queue per learning agent.
pub struct SegmentQueue {
    inner: Mutex<QueueInner>,
    /// Segments arrived (learner `take`).
    ready: Condvar,
    /// A place freed up, the line moved, a game was revoked or the queue closed (waiting pushes).
    space: Condvar,
    capacity: usize,
    full: QueueFull,
    wait_max: Duration,
}

impl SegmentQueue {
    /// A queue that drops the oldest segment when full (`queue_full = "drop_oldest"`).
    pub fn new(capacity: usize) -> Self {
        Self::with_policy(capacity, QueueFull::DropOldest, Duration::ZERO)
    }

    /// 042 queue table: `full` decides what a push into a full queue does, `wait_max` caps one
    /// wait (`queue_wait_max_s`).
    pub fn with_policy(capacity: usize, full: QueueFull, wait_max: Duration) -> Self {
        Self { inner: Mutex::new(QueueInner::default()), ready: Condvar::new(), space: Condvar::new(), capacity: capacity.max(1), full, wait_max }
    }

    /// Queues a segment. A full queue in `wait` mode blocks the calling thread, which belongs to
    /// this game only (its session thread, or the runner worker of its host slot for the
    /// terminal segment), until a place frees up: waiting games go in arrival order; after
    /// `wait_max` the oldest queued segment is dropped (`dropped_full`) and this one goes in.
    /// 042-R3: nothing is queued for a revoked game (also while it waits) or a closed queue.
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
            // `None` (a cap too far to represent) = no cap; never panics under the lock.
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
                    // 042-R4: the learner was reset; this segment is the previous policy's.
                    q.stats.dropped_lag += 1;
                    break false;
                }
                let now = Instant::now();
                let expired = deadline.is_some_and(|d| now >= d);
                if q.line.front() == Some(&ticket) && (q.segs.len() < self.capacity || expired) {
                    break true;
                }
                // Past its deadline but not first: the waiter ahead has an earlier deadline and
                // wakes this one when it leaves the line; the timeout is only a safety net.
                let wait = deadline.map_or(Duration::from_secs(1), |d| d.saturating_duration_since(now)).max(Duration::from_millis(100));
                q = self.space.wait_timeout(q, wait).unwrap().0;
            };
            q.line.retain(|t| *t != ticket);
            let waited = started.elapsed().as_secs_f64();
            q.stats.waits += 1;
            q.stats.wait_s += waited;
            q.stats.wait_max_s = q.stats.wait_max_s.max(waited);
            // The next game in line may go now.
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

    /// 042-R3: the game was Aborted (or stopped); drop its queued segments, cancel its waiting
    /// push and refuse the ones that arrive later.
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

    /// The game finished normally.
    pub fn finish(&self, game_id: u64) {
        self.inner.lock().unwrap().taken.remove(&game_id);
    }

    /// 042 exploiter reset / 042-R4: the learner went back to the BC weights (version 0). Every
    /// queued segment and every waiting push is the previous policy's: dropped (`dropped_lag`),
    /// the waiting games released without queueing. Later segments of that policy carry
    /// versions above the learner's and are dropped by `take`.
    pub fn reset_policy(&self) {
        let mut q = self.inner.lock().unwrap();
        q.stats.dropped_lag += q.segs.len() as u64;
        q.segs.clear();
        q.generation += 1;
        self.space.notify_all();
    }

    /// 042-R4 lag drop, then `lanes` segments in FIFO order if there are that many. Every freed
    /// place wakes the waiting pushes. A step newer than the learner can only come from the
    /// policy before an exploiter reset (actors never run weights the learner has not reached).
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

    /// `lanes` segments not older than `max_lag` updates behind `learner_version` (042-R4), or
    /// `None` when the queue was closed. Blocks until enough segments are queued.
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

    /// Non-blocking variant of `take` (tests, time-shared learners).
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

    /// Pushes waiting for a place now.
    pub fn waiting(&self) -> usize {
        self.inner.lock().unwrap().line.len()
    }

    pub fn stats(&self) -> QueueStats {
        self.inner.lock().unwrap().stats
    }

    /// 042-R10 ① / 042-R3: stop accepting segments, release every waiting push (its segment is
    /// not queued) and wake a waiting learner.
    pub fn close(&self) {
        let mut q = self.inner.lock().unwrap();
        q.closed = true;
        q.segs.clear();
        self.ready.notify_all();
        self.space.notify_all();
    }
}

/// What a learning session reports when its game ends (league win table, 042-R2 games.csv).
#[derive(Debug, Clone)]
pub struct GameResult {
    pub agent: usize,
    pub game_id: u64,
    /// `None` = Aborted (042-R3).
    pub outcome: Option<Outcome>,
    /// Sum of each reward component over the game (033-R3 logging).
    pub reward_sum: [f32; 4],
    pub decisions: u64,
    /// Enemy race the agent's memory knew at its last decision (030-R11: a lobby-Random enemy is
    /// unknown until seen); `None` if never revealed. Per-race league statistics use it.
    pub enemy_race: Option<uranium_proto::Race>,
}

/// `SessionObserver` of a learning session (042).
pub struct TrajectoryObserver {
    agent: usize,
    game_id: u64,
    seq_len: usize,
    queue: Arc<SegmentQueue>,
    results: Sender<GameResult>,
    steps: Vec<Step>,
    state0: Option<HostState>,
    /// State before the step that will start the next segment.
    next_state0: Option<HostState>,
    privileged: PrivilegedTensors,
    reward_sum: [f32; 4],
    decisions: u64,
    enemy_race: Option<uranium_proto::Race>,
    /// 020-R13: the terminal segment waits here for the game-level verdict that sets r_win.
    hold: Option<Box<dyn Fn(Segment) + Send>>,
}

fn parts(r: &RewardParts) -> [f32; 4] {
    [r.win, r.build, r.units, r.value]
}

impl TrajectoryObserver {
    pub fn new(agent: usize, game_id: u64, seq_len: usize, queue: Arc<SegmentQueue>, results: Sender<GameResult>) -> Self {
        assert!(seq_len >= 2, "segments overlap by one step");
        let privileged = PrivilegedTensors { summary: vec![0.0; uranium_env::features::extract::dims().privileged()] };
        Self { agent, game_id, seq_len, queue, results, steps: Vec::new(), state0: None, next_state0: None, privileged, reward_sum: [0.0; 4], decisions: 0, enemy_race: None, hold: None }
    }

    /// Hands the terminal segment to `sink` without r_win instead of queueing it: the caller
    /// adds r_win from the reconciled game result (020-R13) and queues it.
    pub fn hold_terminal(mut self, sink: Box<dyn Fn(Segment) + Send>) -> Self {
        self.hold = Some(sink);
        self
    }

    fn add_reward(&mut self, r: [f32; 4]) {
        if let Some(last) = self.steps.last_mut() {
            for k in 0..4 {
                last.reward[k] += r[k];
                self.reward_sum[k] += r[k];
            }
        }
    }
}

impl SessionObserver for TrajectoryObserver {
    fn wants_state(&self) -> bool {
        // First decision of the game, or the decision that closes this segment and opens the next.
        self.steps.is_empty() || self.steps.len() == self.seq_len - 1
    }

    fn on_privileged(&mut self, p: &uranium_proto::PrivilegedSummary) {
        self.privileged = privileged(p);
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
        // 033-R7: the frames since the previous decision are that decision's reward.
        self.add_reward(parts(&d.reward_before));
        if self.steps.is_empty() {
            self.state0 = d.state.clone();
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
            version: d.version,
        };
        self.steps.push(step);
        if self.steps.len() == self.seq_len {
            // Non-terminal: the last step bootstraps here and starts the next segment.
            let last = self.steps.last().unwrap().clone();
            let steps = std::mem::replace(&mut self.steps, vec![last]);
            let state0 = self.state0.take().unwrap_or_default();
            self.state0 = self.next_state0.take();
            self.queue.push(Segment { agent: self.agent, game_id: self.game_id, steps, state0, terminal: false });
        }
    }

    fn on_end(&mut self, reward_after: RewardParts, outcome: Option<Outcome>) {
        match outcome {
            None => {
                // 042-R3: Aborted / disconnected; nothing of this game is trained.
                self.steps.clear();
                self.queue.revoke(self.game_id);
            }
            Some(_) => {
                let mut after = parts(&reward_after);
                if self.hold.is_some() {
                    after[0] = 0.0;
                }
                self.add_reward(after);
                if !self.steps.is_empty() {
                    let steps = std::mem::take(&mut self.steps);
                    let state0 = self.state0.take().unwrap_or_default();
                    let seg = Segment { agent: self.agent, game_id: self.game_id, steps, state0, terminal: true };
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
        let _ = self.results.send(GameResult { agent: self.agent, game_id: self.game_id, outcome, reward_sum: self.reward_sum, decisions: self.decisions, enemy_race: self.enemy_race });
    }
}

#[cfg(test)]
mod tests {
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

    /// Segments of T overlap by one step, carry the state before their first step, and rewards
    /// land on the decision that preceded them (033-R7); the end adds r_win to the last step.
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
        // step 0's reward = reward_before of decision 1 = 1.0, etc.
        assert_eq!(a.steps.iter().take(3).map(|s| s.reward[3]).collect::<Vec<_>>(), vec![1.0, 2.0, 3.0]);
        // Next segment starts with decision 3 and the state before it.
        assert_eq!(b.state0.parts[0].1, vec![3.0]);
        assert_eq!(b.steps.len(), 3);
        assert!(b.terminal);
        assert_eq!(b.steps.iter().map(|s| s.reward).collect::<Vec<_>>(), vec![[0.0, 0.0, 0.0, 4.0], [0.0, 0.0, 0.0, 5.0], [1.0, 0.0, 0.0, 0.5]]);
        let r = rx.recv().unwrap();
        assert_eq!(r.outcome, Some(Outcome::Win));
        assert_eq!(r.decisions, 6);
        // 042-R2 `trained_steps`: a finished game's segments train each decision once.
        let (a, b) = (std::slice::from_ref(&a), std::slice::from_ref(&b));
        assert_eq!(trained_steps(a) + trained_steps(b), r.decisions);
        assert_eq!(r.reward_sum, [1.0, 0.0, 0.0, 15.5]);
        // 030-R11: the race the agent's memory knew is reported for the per-race league tables.
        assert_eq!(r.enemy_race, Some(uranium_proto::Race::Zerg));
    }

    /// 042-R3: an Aborted game's queued segments are revoked; taken ones are counted.
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

    /// 042-R4: segments lagging more than `max_lag` updates are dropped when a batch is taken.
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

    /// A one-step terminal segment of `game_id` with behaviour version `version`. The packed
    /// observation is built once and cloned.
    fn seg(game_id: u64, version: u64) -> Segment {
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
                version,
            }],
            state0: HostState::default(),
            terminal: true,
        }
    }

    /// A `wait` queue whose waits never reach the cap within a test.
    fn wait_queue(capacity: usize) -> Arc<SegmentQueue> {
        Arc::new(SegmentQueue::with_policy(capacity, QueueFull::Wait, Duration::from_secs(60)))
    }

    /// Pushes a segment built by the caller from another thread (building one packs an
    /// observation, slow in a debug build on a slow host, so it must not eat into a wait cap);
    /// the receiver gets the game id once the push returned.
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

    /// Waits (bounded) until `n` pushes wait in line.
    fn until_waiting(q: &SegmentQueue, n: usize) {
        let end = Instant::now() + Duration::from_secs(10);
        while q.waiting() != n {
            assert!(Instant::now() < end, "expected {n} waiting pushes, have {}", q.waiting());
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    const BLOCKED: Duration = Duration::from_millis(100);
    const RELEASED: Duration = Duration::from_secs(10);

    /// 042 queue table (backpressure): a push into a full queue waits until the learner takes a
    /// batch, then goes in; nothing is dropped and the wait is counted for `queue_wait_s`.
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

    /// 042 queue table: waiting games go in the order they arrived (FIFO), one per freed place,
    /// and a later push queues behind the waiting ones.
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

    /// 042-R3: revoking a game cancels its waiting push (counted as revoked, not queued), lets
    /// the next waiter keep its place, and a later push of that game is refused.
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
        // A late segment of the revoked game: refused at once, even with a free place.
        q.push(seg(5, 0));
        assert_eq!(q.try_take(1, 0, 100).unwrap()[0].game_id, 6);
        assert!(q.is_empty());
        assert_eq!(q.stats().revoked, 2);
    }

    /// 042-R3: a game revoked before its segment arrives never gets it queued (drop_oldest too).
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

    /// 042-R10 / R3: closing the queue releases every waiting push without queueing it.
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

    /// 042-R4 + backpressure: segments dropped for lag free places and wake a waiting push.
    #[test]
    fn lag_drop_wakes_a_waiting_push_042_r4() {
        let q = wait_queue(1);
        q.push(seg(1, 0));
        let (h, rx) = push_in_thread(&q, seg(2, 90));
        until_waiting(&q, 1);
        // Not enough segments for 2 lanes, but the stale one is dropped.
        assert!(q.try_take(2, 100, 32).is_none());
        assert_eq!(rx.recv_timeout(RELEASED).unwrap(), 2);
        h.join().unwrap();
        assert_eq!(q.try_take(1, 100, 32).unwrap()[0].game_id, 2);
        assert_eq!(q.stats().dropped_lag, 1);
    }

    /// AMB-042-11: a wait that reaches `queue_wait_max_s` drops the oldest segment
    /// (`dropped_full`) and queues the waiting one.
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

    /// AMB-042-11: with several waiters the caps expire in arrival order, each dropping the then
    /// oldest segment; a waiter behind the front whose deadline already passed still waits for
    /// the front one to go first.
    #[test]
    fn chained_wait_caps_keep_fifo_order_042() {
        // The cap leaves a slow host ample time to line up all three waiters before the first
        // one's cap expires; the segments are built before the first waiter starts.
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
        // 10 dropped 1, 11 dropped 2, 12 dropped 10.
        let b = q.try_take(2, 0, 100).unwrap();
        assert_eq!(b.iter().map(|s| s.game_id).collect::<Vec<_>>(), vec![11, 12]);
        let st = q.stats();
        assert_eq!((st.pushed, st.dropped_full, st.waits), (5, 3, 3));
    }

    /// AMB-042-11: a waiter whose deadline passed while it was behind the front does not jump
    /// the line. The test holds the queue lock until both deadlines have passed, so both wake
    /// expired in any order; had the second gone first, game 10 (not 11) would survive.
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
        // 10 dropped 1, then 11 dropped 10.
        assert_eq!(q.try_take(1, 0, 100).unwrap()[0].game_id, 11);
        let st = q.stats();
        assert_eq!((st.dropped_full, st.waits), (2, 2));
    }

    /// 042-R4 exploiter reset: queued and waiting segments of the previous policy are dropped
    /// (`dropped_lag`), the waiting game is released, and a later segment of that policy
    /// (version above the reset learner's) is dropped when a batch is taken.
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
        // In-flight games keep sending the old policy's segments for a while.
        q.push(seg(4, 4_000));
        q.push(seg(5, 0));
        assert_eq!(q.try_take(1, 0, 32).unwrap()[0].game_id, 5);
        assert!(q.is_empty());
        assert_eq!(q.stats().dropped_lag, 4);
    }

    /// `queue_full = "drop_oldest"`: the previous behaviour, no waiting.
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
