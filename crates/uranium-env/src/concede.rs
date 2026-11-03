//! 042-R14 collapse concede of RL training games (user decision 2026-11-03, AMB-042-15). Not a
//! model action and not the 031-R12 surrender rule: a model session of an RL league game whose own
//! supply stays at or below `max_supply` (screen units) for `hold_s` seconds after `after_s`
//! leaves with the 010-R13 surrender, so a collapsed game ends as a loss instead of dragging to the
//! 60-minute draw. Inputs are the own `PlayerState.supply_used` and the frame only (010-R5, 033-R5).
//! The agent server judges before inference; the RL run loop turns it on (`[concede]`, 042).

use std::sync::Mutex;

use serde::{Deserialize, Serialize};

/// Game frames per second for the 042-R14 seconds (020's 60 minutes = 86,400 frames, 031-R12).
pub const FPS: u32 = 24;

/// `[concede]` of the RL config (042-R14). Every key is optional; the defaults are the spec's.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ConcedeConfig {
    /// `false` (default) = no judgement at all, the earlier behaviour.
    #[serde(default)]
    pub enabled: bool,
    /// Collapse bar in screen supply (BWAPI `supplyUsed()` / 2); 0..=200.
    #[serde(default = "default_max_supply")]
    pub max_supply: u32,
    /// Seconds the collapse must last; at least 1.
    #[serde(default = "default_hold_s")]
    pub hold_s: u32,
    /// Decisions before this game second are not counted.
    #[serde(default = "default_after_s")]
    pub after_s: u32,
}

fn default_max_supply() -> u32 {
    4
}

fn default_hold_s() -> u32 {
    120
}

fn default_after_s() -> u32 {
    180
}

impl Default for ConcedeConfig {
    fn default() -> Self {
        Self { enabled: false, max_supply: default_max_supply(), hold_s: default_hold_s(), after_s: default_after_s() }
    }
}

impl ConcedeConfig {
    /// 042-R14 ranges: `max_supply` 0..=200, `hold_s` at least 1 (`after_s` is never negative).
    pub fn validate(&self) -> Result<(), String> {
        if self.max_supply > 200 {
            return Err(format!("[concede] max_supply must be 0..=200, got {} (042-R14)", self.max_supply));
        }
        if self.hold_s < 1 {
            return Err("[concede] hold_s must be at least 1 (042-R14)".into());
        }
        if self.after_s.checked_mul(FPS).is_none() || self.hold_s.checked_mul(FPS).is_none() {
            return Err("[concede] after_s / hold_s too large (042-R14)".into());
        }
        Ok(())
    }

    /// `supply_used` (BWAPI units, 2 per SCV) at or below this is a collapse: 2 x `max_supply`.
    pub fn collapse_supply(&self) -> u32 {
        2 * self.max_supply
    }

    /// First frame whose decision counts: `after_s` x 24.
    pub fn after_frame(&self) -> u32 {
        self.after_s * FPS
    }

    /// `hold_s` x 24.
    pub fn hold_frames(&self) -> u32 {
        self.hold_s * FPS
    }
}

/// One session's 042-R14 judgement: one call per decision, in frame order.
#[derive(Debug, Clone)]
pub struct CollapseJudge {
    cfg: ConcedeConfig,
    /// Frame of the first decision of the current collapse run (counted from `after_s`).
    since: Option<u32>,
}

impl CollapseJudge {
    pub fn new(cfg: ConcedeConfig) -> Self {
        Self { cfg, since: None }
    }

    /// A decision at `frame` with own `supply_used` (BWAPI units). True = concede at this decision:
    /// every decision since the run's first one (at or after `after_s`) was a collapse and this one
    /// is at least `hold_s` after it. Never with `enabled = false`.
    pub fn decide(&mut self, frame: u32, supply_used: u16) -> bool {
        if !self.cfg.enabled || frame < self.cfg.after_frame() || u32::from(supply_used) > self.cfg.collapse_supply() {
            self.since = None;
            return false;
        }
        let since = *self.since.get_or_insert(frame);
        frame.saturating_sub(since) >= self.cfg.hold_frames()
    }
}

/// Who conceded a game (042-R14 log): the `GameSpec.players` index of the session and the frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Conceded {
    pub side: u8,
    pub frame: u32,
}

/// 042-R14 per game: the config and the one concede of the game. The first session that holds
/// claims it; the other model session of the game then stops judging (both leaving would make
/// each report the other's win and 020-R13 could turn the result around).
#[derive(Debug)]
pub struct ConcedeGame {
    pub cfg: ConcedeConfig,
    done: Mutex<Option<Conceded>>,
}

impl ConcedeGame {
    pub fn new(cfg: ConcedeConfig) -> Self {
        Self { cfg, done: Mutex::new(None) }
    }

    /// A session of the game conceded (or is conceding) already.
    pub fn conceded(&self) -> Option<Conceded> {
        *self.done.lock().unwrap()
    }

    /// True for the first claim of the game only.
    pub fn claim(&self, side: u8, frame: u32) -> bool {
        let mut d = self.done.lock().unwrap();
        if d.is_some() {
            return false;
        }
        *d = Some(Conceded { side, frame });
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn on() -> ConcedeConfig {
        ConcedeConfig { enabled: true, ..Default::default() }
    }

    /// 042-AC5: defaults 4 / 120 s / 180 s, off unless enabled.
    #[test]
    fn defaults_042_r14() {
        let c = ConcedeConfig::default();
        assert_eq!((c.enabled, c.max_supply, c.hold_s, c.after_s), (false, 4, 120, 180));
        assert_eq!((c.collapse_supply(), c.after_frame(), c.hold_frames()), (8, 4_320, 2_880));
        let parsed: ConcedeConfig = toml::from_str("").unwrap();
        assert_eq!(parsed, c);
        let parsed: ConcedeConfig = toml::from_str("enabled = true").unwrap();
        assert_eq!(parsed, on());
    }

    /// 042-AC5 `after_s` boundary: a collapsed decision at 4,319 does not start the run, one at
    /// 4,320 does.
    #[test]
    fn after_s_boundary_042_ac5() {
        let mut j = CollapseJudge::new(on());
        assert!(!j.decide(4_319, 0));
        // The run starts at 4,320, not 4,319: 4,319 + 2,880 is not enough.
        assert!(!j.decide(4_320, 0));
        assert!(!j.decide(4_319 + 2_880, 0));
        assert!(j.decide(4_320 + 2_880, 0));
        // Collapsed from the start: the earliest concede is 5:00.
        let mut j = CollapseJudge::new(on());
        let mut first = None;
        for f in (0..10_000).step_by(8) {
            if j.decide(f, 4) {
                first = Some(f);
                break;
            }
        }
        assert_eq!(first, Some(7_200));
    }

    /// 042-AC5 `hold_s` boundary: 2,879 frames after the run's first decision no, 2,880 yes.
    #[test]
    fn hold_s_boundary_042_ac5() {
        let mut j = CollapseJudge::new(on());
        let s = 10_000;
        assert!(!j.decide(s, 8));
        assert!(!j.decide(s + 1_000, 8));
        assert!(!j.decide(s + 2_879, 8));
        assert!(j.decide(s + 2_880, 8));
    }

    /// 042-AC5: one decision that is not a collapse breaks the run.
    #[test]
    fn non_collapse_breaks_the_run_042_ac5() {
        let mut j = CollapseJudge::new(on());
        let s = 10_000;
        assert!(!j.decide(s, 2));
        assert!(!j.decide(s + 2_000, 9));
        assert!(!j.decide(s + 2_880, 2), "run restarted at s + 2,880");
        assert!(!j.decide(s + 5_759, 2));
        assert!(j.decide(s + 5_760, 2));
    }

    /// 042-AC5: supply in BWAPI units (2 x screen): 8 is a collapse, 9 is not; `max_supply` scales.
    #[test]
    fn supply_is_twice_the_screen_value_042_ac5() {
        let mut j = CollapseJudge::new(on());
        assert!(!j.decide(5_000, 8));
        assert!(j.decide(5_000 + 2_880, 8));
        let mut j = CollapseJudge::new(on());
        assert!(!j.decide(5_000, 9));
        assert!(!j.decide(5_000 + 2_880, 9));
        let mut j = CollapseJudge::new(ConcedeConfig { max_supply: 0, ..on() });
        assert!(!j.decide(5_000, 0));
        assert!(!j.decide(5_000 + 2_880, 1));
        assert!(!j.decide(5_000 + 2_881, 0));
        assert!(j.decide(5_000 + 2_881 + 2_880, 0));
    }

    /// 042-AC5: `enabled = false` never concedes.
    #[test]
    fn disabled_never_concedes_042_ac5() {
        let mut j = CollapseJudge::new(ConcedeConfig::default());
        for f in (0..86_400).step_by(24) {
            assert!(!j.decide(f, 0));
        }
    }

    /// 042-AC5: one concede per game; the second claim (the other model session) fails.
    #[test]
    fn once_per_game_042_ac5() {
        let g = ConcedeGame::new(on());
        assert_eq!(g.conceded(), None);
        assert!(g.claim(1, 7_300));
        assert!(!g.claim(0, 7_300));
        assert!(!g.claim(1, 7_400));
        assert_eq!(g.conceded(), Some(Conceded { side: 1, frame: 7_300 }));
    }

    /// 042-AC5 config check: `max_supply` 0..=200, `hold_s` >= 1; negative values and unknown keys
    /// do not load.
    #[test]
    fn config_check_042_ac5() {
        let ok = |s: &str| toml::from_str::<ConcedeConfig>(s).map_err(|e| e.to_string()).and_then(|c| c.validate().map(|_| c));
        assert!(ok("max_supply = 200").is_ok());
        assert!(ok("max_supply = 0").is_ok());
        assert!(ok("max_supply = 201").is_err());
        assert!(ok("max_supply = -1").is_err());
        assert!(ok("hold_s = 0").is_err());
        assert!(ok("hold_s = 1").is_ok());
        assert!(ok("after_s = 0").is_ok());
        assert!(ok("after_s = -1").is_err());
        assert!(ok("enable = true").is_err());
        assert!(ok("after_s = 4000000000").is_err());
    }
}
