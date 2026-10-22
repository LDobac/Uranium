//! Game end (012-R15), with the OpenBW shim's winner and reason rules (shim:434-468).
//! `Game.victory_state[p]`: 3 = victorious, 1 / 2 / 4 / 6 = defeated (PlayerImpl.cpp:111-126).
//! A player whose type turns PlayerLeft (10), ComputerLeft (11) or Neutral (7) left the game
//! (PlayerImpl.cpp:266-271). An enemy that leaves with buildings standing ends the game as a
//! Surrender, not an Elimination (011-R17).

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
    /// Enemy building tags of the previous frame (shim `enemy_buildings_`).
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

    /// `Some((winner, reason))` once the game is over for our player.
    pub fn update(&mut self, s: &Snapshot, d: &[Derived]) -> Option<(Option<u8>, EndReason)> {
        if let Some(e) = self.enemy {
            let ty = s.player_types.get(e as usize).copied().unwrap_or(0);
            let left = ty == PLAYER_LEFT || ty == COMPUTER_LEFT || ty == NEUTRAL;
            if left && !self.left_seen {
                self.left_seen = true;
                // Leaving turns the buildings neutral: they still exist (shim onPlayerLeft).
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
            // Both defeated on one frame: draw (shim:452-453).
            if defeated(theirs) { None } else { self.enemy }
        } else {
            return None;
        };
        Some((winner, if self.enemy_left { EndReason::Surrender } else { EndReason::Elimination }))
    }
}
