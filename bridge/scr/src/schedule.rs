pub const MIN_GAP: u32 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Idle { next: u32 },
    Waiting { observed: u32 },
}

#[derive(Debug, Clone)]
pub struct Schedule {
    phase: Phase,
    pub expired: u32,
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
