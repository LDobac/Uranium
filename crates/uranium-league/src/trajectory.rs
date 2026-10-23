//! RL trajectories (042): a learning session's decisions become segments of T steps that go to
//! the learner through a queue. Segments overlap by one step: the last step of a non-terminal
//! segment only bootstraps the value (it is trained as the first step of the next segment), so
//! no decision is lost and no extra forward is needed. A terminal segment trains all its steps
//! and bootstraps from 0 (γ = 1, episode over).
//!
//! 042-R3 (AMB-042-6): segments are queued as soon as they are complete; when the game turns out
//! Aborted, the ones still in the queue are revoked. 042-R4: segments whose behaviour version
//! lags the learner by more than `max_policy_lag` updates are dropped when a batch is taken.

use std::collections::{HashMap, VecDeque};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Condvar, Mutex};

use uranium_env::features::executor::HostAction;
use uranium_env::features::extract::privileged;
use uranium_env::features::PrivilegedTensors;
use uranium_env::reward::{Outcome, RewardParts};
use uranium_agent::model_policy::{Decision, SessionObserver};
use uranium_data::state::HostState;
use uranium_agent::infer::HEADS;

/// Observations with their two large planes compressed (042), shared with BC (041-R18).
pub use uranium_data::pack::PackedObs;

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

    /// Steps the learner trains on.
    pub fn trained(&self) -> usize {
        if self.terminal { self.steps.len() } else { self.steps.len().saturating_sub(1) }
    }
}

/// Counters of the queue (042-R2 logging).
#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct QueueStats {
    pub pushed: u64,
    /// 042-R4: behaviour version too old.
    pub dropped_lag: u64,
    /// Queue full: oldest dropped.
    pub dropped_full: u64,
    /// 042-R3: revoked because their game was Aborted.
    pub revoked: u64,
    /// 042-R3: segments of an Aborted game that the learner had already used.
    pub used_aborted: u64,
}

#[derive(Default)]
struct QueueInner {
    segs: VecDeque<Segment>,
    /// Segments already taken per game still running (for `used_aborted`).
    taken: HashMap<u64, u64>,
    stats: QueueStats,
    closed: bool,
}

/// One queue per learning agent.
pub struct SegmentQueue {
    inner: Mutex<QueueInner>,
    ready: Condvar,
    capacity: usize,
}

impl SegmentQueue {
    pub fn new(capacity: usize) -> Self {
        Self { inner: Mutex::new(QueueInner::default()), ready: Condvar::new(), capacity: capacity.max(1) }
    }

    pub fn push(&self, s: Segment) {
        let mut q = self.inner.lock().unwrap();
        if q.closed {
            return;
        }
        q.stats.pushed += 1;
        q.segs.push_back(s);
        while q.segs.len() > self.capacity {
            q.segs.pop_front();
            q.stats.dropped_full += 1;
        }
        self.ready.notify_all();
    }

    /// 042-R3: the game was Aborted; drop its queued segments.
    pub fn revoke(&self, game_id: u64) {
        let mut q = self.inner.lock().unwrap();
        let before = q.segs.len();
        q.segs.retain(|s| s.game_id != game_id);
        q.stats.revoked += (before - q.segs.len()) as u64;
        if let Some(n) = q.taken.remove(&game_id) {
            q.stats.used_aborted += n;
        }
    }

    /// The game finished normally.
    pub fn finish(&self, game_id: u64) {
        self.inner.lock().unwrap().taken.remove(&game_id);
    }

    /// `lanes` segments not older than `max_lag` updates behind `learner_version` (042-R4), or
    /// `None` when the queue was closed. Blocks until enough segments are queued.
    pub fn take(&self, lanes: usize, learner_version: u64, max_lag: u64) -> Option<Vec<Segment>> {
        let mut q = self.inner.lock().unwrap();
        loop {
            let before = q.segs.len();
            q.segs.retain(|s| learner_version.saturating_sub(s.version()) <= max_lag);
            q.stats.dropped_lag += (before - q.segs.len()) as u64;
            if q.closed {
                return None;
            }
            if q.segs.len() >= lanes {
                let batch: Vec<Segment> = q.segs.drain(..lanes).collect();
                for s in &batch {
                    *q.taken.entry(s.game_id).or_default() += 1;
                }
                return Some(batch);
            }
            q = self.ready.wait_timeout(q, std::time::Duration::from_millis(500)).unwrap().0;
        }
    }

    /// Non-blocking variant of `take` (tests, time-shared learners).
    pub fn try_take(&self, lanes: usize, learner_version: u64, max_lag: u64) -> Option<Vec<Segment>> {
        let mut q = self.inner.lock().unwrap();
        let before = q.segs.len();
        q.segs.retain(|s| learner_version.saturating_sub(s.version()) <= max_lag);
        q.stats.dropped_lag += (before - q.segs.len()) as u64;
        if q.segs.len() < lanes {
            return None;
        }
        let batch: Vec<Segment> = q.segs.drain(..lanes).collect();
        for s in &batch {
            *q.taken.entry(s.game_id).or_default() += 1;
        }
        Some(batch)
    }

    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().segs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn stats(&self) -> QueueStats {
        self.inner.lock().unwrap().stats
    }

    /// 042-R10 ①: stop accepting segments and wake a waiting learner.
    pub fn close(&self) {
        let mut q = self.inner.lock().unwrap();
        q.closed = true;
        q.segs.clear();
        self.ready.notify_all();
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
        let seg = |v: u64| Segment {
            agent: 0,
            game_id: v,
            steps: vec![Step {
                obs: PackedObs::pack(&fake_obs(0)),
                action: HostAction { action: 0, delay_bucket: 0, queued: false, selected: vec![], target_unit: None, target_loc: None },
                log_prob: 0.0,
                head_log_prob: [0.0; HEADS],
                reward: [0.0; 4],
                privileged: PrivilegedTensors { summary: vec![] },
                version: v,
            }],
            state0: HostState::default(),
            terminal: true,
        };
        for v in [0, 5, 40, 41] {
            q.push(seg(v));
        }
        let b = q.try_take(2, 72, 32).unwrap();
        assert_eq!(b.iter().map(|s| s.game_id).collect::<Vec<_>>(), vec![40, 41]);
        assert_eq!(q.stats().dropped_lag, 2);
    }
}
