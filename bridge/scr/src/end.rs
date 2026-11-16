use uranium_proto::EndReason;

use crate::observe::Derived;
use crate::snapshot::Snapshot;
use crate::tables::unit_type;

const VICTORIOUS: u8 = 3;
const PLAYER_LEFT: u8 = 10;
const COMPUTER_LEFT: u8 = 11;
const NEUTRAL: u8 = 7;

pub fn defeated(v: u8) -> bool {
    matches!(v, 1 | 2 | 4 | 6)
}

pub struct EndTracker {
    me: u8,
    enemy: Option<u8>,
    enemy_buildings: Vec<u32>,
    enemy_left: bool,
    left_seen: bool,
}

impl EndTracker {
    pub fn new(me: u8, enemy: Option<u8>) -> Self {
        Self { me, enemy, enemy_buildings: vec![], enemy_left: false, left_seen: false }
    }

    pub fn enemy(&self) -> Option<u8> {
        self.enemy
    }

    pub fn update(&mut self, s: &Snapshot, d: &[Derived]) -> Option<(Option<u8>, EndReason)> {
        if let Some(e) = self.enemy {
            let ty = s.player_types.get(e as usize).copied().unwrap_or(0);
            let left = ty == PLAYER_LEFT || ty == COMPUTER_LEFT || ty == NEUTRAL;
            if left && !self.left_seen {
                self.left_seen = true;
                self.enemy_left = self.enemy_buildings.iter().any(|tag| s.units.iter().any(|u| u.tag == *tag));
            }
            if !left {
                self.enemy_buildings = s.units.iter().zip(d).filter(|(u, x)| u.player == e && unit_type(x.bwapi_type).is_some_and(|t| t.is_building())).map(|(u, _)| u.tag).collect();
            }
        }
        let mine = s.victory_state.get(self.me as usize).copied().unwrap_or(0);
        let theirs = self.enemy.and_then(|e| s.victory_state.get(e as usize).copied()).unwrap_or(0);
        let winner = if mine == VICTORIOUS {
            Some(self.me)
        } else if defeated(mine) {
            if defeated(theirs) { None } else { self.enemy }
        } else {
            return None;
        };
        Some((winner, if self.enemy_left { EndReason::Surrender } else { EndReason::Elimination }))
    }
}
