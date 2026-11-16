use std::time::{Duration, Instant};
use uranium_runner::JobOutcome;
use crate::trajectory::GameResult;

pub(super) const DEAD_LEARNER_GAMES: u64 = 20;
pub(super) const BAD_SESSION_STREAK: u64 = 10;

pub(super) fn bad_learning_session(outcome: JobOutcome, result: Option<&GameResult>) -> Option<bool> {
    match (outcome, result) {
        (JobOutcome::Finished, Some(r)) => Some(r.decisions == 0),
        _ => None,
    }
}

pub(super) struct StallWatch {
    after: Duration,
    last: Instant,
    games_at_last: u64,
}

impl StallWatch {
    pub(super) fn new(minutes: u64, now: Instant, games: u64) -> Self {
        Self { after: Duration::from_secs(minutes * 60), last: now, games_at_last: games }
    }

    pub(super) fn progress(&mut self, now: Instant, games: u64) {
        (self.last, self.games_at_last) = (now, games);
    }

    pub(super) fn check(&self, now: Instant, games: u64, bad_streak: u64) -> Option<String> {
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

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(w.check(at(15), 120, BAD_SESSION_STREAK - 1).is_none());
        let why = w.check(at(15), 120, BAD_SESSION_STREAK).expect("10 in a row");
        assert!(why.contains("without a decision"), "{why}");
        let off = StallWatch::new(0, t0, 0);
        assert!(off.check(at(600), 10_000, 1_000).is_none());
        let r = |d: u64| GameResult { agent: 0, game_id: 1, outcome: None, reward_sum: [0.0; 4], scout_sum: 0.0, survive_sum: 0.0, scout: None, decisions: d, enemy_race: None };
        assert_eq!(bad_learning_session(JobOutcome::Finished, Some(&r(0))), Some(true));
        assert_eq!(bad_learning_session(JobOutcome::Finished, None), None, "no report of the learning session: not counted");
        assert_eq!(bad_learning_session(JobOutcome::Finished, Some(&r(3))), Some(false));
        for o in [JobOutcome::Aborted, JobOutcome::Timeout, JobOutcome::Stopped, JobOutcome::Rejected] {
            assert_eq!(bad_learning_session(o, Some(&r(0))), None, "{o:?}");
        }
    }
}
