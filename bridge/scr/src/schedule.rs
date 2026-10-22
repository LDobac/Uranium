//! Realtime observation schedule (012-R7, 012-R8). One observation waits for its `Act` at a time;
//! the game never waits. The next observation is at max(f + d, f + 2, the frame the Act was
//! applied), and an Act read after its `valid_until_frame` is dropped and counted.

/// Minimum frames between two observations (012-R8).
pub const MIN_GAP: u32 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// Observe at or after this frame.
    Idle { next: u32 },
    /// Observation of this frame is waiting for its Act.
    Waiting { observed: u32 },
}

#[derive(Debug, Clone)]
pub struct Schedule {
    phase: Phase,
    /// Acts dropped for arriving after `valid_until_frame` (012-AC6).
    pub expired: u32,
    /// Frames from each observation to the frame its Act was applied (012-AC6).
    pub delays: Vec<u32>,
}

impl Schedule {
    pub fn new(first: u32) -> Self {
        Self { phase: Phase::Idle { next: first }, expired: 0, delays: vec![] }
    }

    pub fn due(&self, frame: u32) -> bool {
        matches!(self.phase, Phase::Idle { next } if frame >= next)
    }

    pub fn observed(&mut self, frame: u32) {
        self.phase = Phase::Waiting { observed: frame };
    }

    /// The Act for the waiting observation is read at `frame`. Returns whether its commands are
    /// sent (false: expired, or no observation was waiting).
    pub fn on_act(&mut self, frame: u32, advance: u32, valid_until: Option<u32>) -> bool {
        let Phase::Waiting { observed } = self.phase else { return false };
        let ok = valid_until.is_none_or(|v| frame <= v);
        if ok {
            self.delays.push(frame - observed);
        } else {
            self.expired += 1;
        }
        let next = (observed + advance.max(1)).max(observed + MIN_GAP).max(frame);
        self.phase = Phase::Idle { next };
        ok
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 012-R8 / 012-R7: observe at f + d, at least 2 frames apart, never before the Act came;
    /// late Acts are dropped and counted.
    #[test]
    fn schedule_012_r8() {
        let mut s = Schedule::new(1);
        assert!(!s.due(0) && s.due(1));
        s.observed(1);
        assert!(!s.due(50), "waits for the Act");
        assert!(s.on_act(2, 8, Some(25)));
        assert!(!s.due(8) && s.due(9), "f + d");
        s.observed(9);
        assert!(s.on_act(10, 1, Some(33)));
        assert!(!s.due(10) && s.due(11), "at least 2 frames apart");
        s.observed(11);
        assert!(!s.on_act(40, 4, Some(35)), "after valid_until");
        assert!(s.due(40), "a late Act moves the next observation to now");
        assert_eq!((s.expired, s.delays.clone()), (1, vec![1, 1]));
        assert!(!s.on_act(41, 4, None), "no observation waiting");
    }
}
