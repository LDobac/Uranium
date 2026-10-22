//! Unit events (012-R12) as the OpenBW bridge reports them. With CompleteMapInformation every live
//! unit is accessible, so BWAPI's Create comes with Discover and Destroy with Evade
//! (GameUnits.cpp:157-216), and Morph is a BWAPI type change of a live unit (GameUnits.cpp:378).
//! The shim keeps Create / Morph / Discover of units passing `visible_to_self`, and Destroy /
//! Evade of own units or of units our player saw on their last live frame (shim:481-486). The
//! game is compared every frame, not only on observations, so short-lived units are reported.

use std::collections::HashMap;

use uranium_proto::GameEvent;

use crate::observe::Derived;
use crate::snapshot::Snapshot;

#[derive(Clone, Copy)]
struct Tracked {
    bwapi_type: u16,
    own: bool,
    /// BWAPI `isVisible(self)` on the unit's last live frame.
    visible: bool,
}

#[derive(Default)]
pub struct EventTracker {
    alive: HashMap<u32, Tracked>,
    pending: Vec<GameEvent>,
}

impl EventTracker {
    /// Compares this frame's live units (`d` in `s.units` order) with the previous frame's.
    pub fn update(&mut self, s: &Snapshot, d: &[Derived]) {
        let mut now = HashMap::with_capacity(s.units.len());
        for (u, x) in s.units.iter().zip(d) {
            match self.alive.get(&u.tag) {
                None if x.observed => self.pending.extend([GameEvent::UnitCreated(u.tag), GameEvent::UnitDiscover(u.tag)]),
                Some(prev) if prev.bwapi_type != x.bwapi_type && x.observed => self.pending.push(GameEvent::UnitMorph(u.tag)),
                _ => {}
            }
            now.insert(u.tag, Tracked { bwapi_type: x.bwapi_type, own: u.player == s.local, visible: x.visible });
        }
        for (tag, prev) in &self.alive {
            if !now.contains_key(tag) && (prev.own || prev.visible) {
                self.pending.extend([GameEvent::UnitEvade(*tag), GameEvent::UnitDestroyed(*tag)]);
            }
        }
        self.alive = now;
    }

    pub fn push(&mut self, e: GameEvent) {
        self.pending.push(e);
    }

    /// Events since the last call (sent with the next observation).
    pub fn take(&mut self) -> Vec<GameEvent> {
        std::mem::take(&mut self.pending)
    }
}
