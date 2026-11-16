use std::sync::Mutex;

use serde::{Deserialize, Serialize};

pub use crate::types::FRAMES_PER_SECOND as FPS;

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ConcedeConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_max_supply")]
    pub max_supply: u32,
    #[serde(default = "default_hold_s")]
    pub hold_s: u32,
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

    pub fn collapse_supply(&self) -> u32 {
        2 * self.max_supply
    }

    pub fn after_frame(&self) -> u32 {
        self.after_s * FPS
    }

    pub fn hold_frames(&self) -> u32 {
        self.hold_s * FPS
    }
}

#[derive(Debug, Clone)]
pub struct CollapseJudge {
    cfg: ConcedeConfig,
    since: Option<u32>,
}

impl CollapseJudge {
    pub fn new(cfg: ConcedeConfig) -> Self {
        Self { cfg, since: None }
    }

    pub fn decide(&mut self, frame: u32, supply_used: u16) -> bool {
        if !self.cfg.enabled || frame < self.cfg.after_frame() || u32::from(supply_used) > self.cfg.collapse_supply() {
            self.since = None;
            return false;
        }
        let since = *self.since.get_or_insert(frame);
        frame.saturating_sub(since) >= self.cfg.hold_frames()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeaveRule {
    Concede,
    Surrender,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Conceded {
    pub side: u8,
    pub frame: u32,
    pub rule: LeaveRule,
}

#[derive(Debug)]
pub struct ConcedeGame {
    pub cfg: ConcedeConfig,
    done: Mutex<Option<Conceded>>,
}

impl ConcedeGame {
    pub fn new(cfg: ConcedeConfig) -> Self {
        Self { cfg, done: Mutex::new(None) }
    }

    pub fn conceded(&self) -> Option<Conceded> {
        *self.done.lock().unwrap()
    }

    pub fn claim(&self, side: u8, frame: u32, rule: LeaveRule) -> bool {
        let mut d = self.done.lock().unwrap();
        if d.is_some() {
            return false;
        }
        *d = Some(Conceded { side, frame, rule });
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn on() -> ConcedeConfig {
        ConcedeConfig { enabled: true, ..Default::default() }
    }

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

    #[test]
    fn after_s_boundary_042_ac5() {
        let mut j = CollapseJudge::new(on());
        assert!(!j.decide(4_319, 0));
        assert!(!j.decide(4_320, 0));
        assert!(!j.decide(4_319 + 2_880, 0));
        assert!(j.decide(4_320 + 2_880, 0));
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

    #[test]
    fn hold_s_boundary_042_ac5() {
        let mut j = CollapseJudge::new(on());
        let s = 10_000;
        assert!(!j.decide(s, 8));
        assert!(!j.decide(s + 1_000, 8));
        assert!(!j.decide(s + 2_879, 8));
        assert!(j.decide(s + 2_880, 8));
    }

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

    #[test]
    fn disabled_never_concedes_042_ac5() {
        let mut j = CollapseJudge::new(ConcedeConfig::default());
        for f in (0..86_400).step_by(24) {
            assert!(!j.decide(f, 0));
        }
    }

    #[test]
    fn once_per_game_042_ac5() {
        let g = ConcedeGame::new(on());
        assert_eq!(g.conceded(), None);
        assert!(g.claim(1, 7_300, LeaveRule::Concede));
        assert!(!g.claim(0, 7_300, LeaveRule::Concede));
        assert!(!g.claim(1, 7_400, LeaveRule::Concede));
        assert!(!g.claim(0, 7_400, LeaveRule::Surrender), "no surrender after a concede");
        assert_eq!(g.conceded(), Some(Conceded { side: 1, frame: 7_300, rule: LeaveRule::Concede }));
        let g = ConcedeGame::new(on());
        assert!(g.claim(0, 20_000, LeaveRule::Surrender));
        assert!(!g.claim(1, 20_000, LeaveRule::Concede), "no concede after a surrender");
    }

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
