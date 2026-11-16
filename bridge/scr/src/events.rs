use std::collections::HashMap;

use uranium_proto::GameEvent;

use crate::observe::Derived;
use crate::snapshot::Snapshot;

#[derive(Clone, Copy)]
struct Tracked {
    bwapi_type: u16,
    own: bool,
    visible: bool,
}

#[derive(Default)]
pub struct EventTracker {
    alive: HashMap<u32, Tracked>,
    pending: Vec<GameEvent>,
}

impl EventTracker {
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

    pub fn take(&mut self) -> Vec<GameEvent> {
        std::mem::take(&mut self.pending)
    }
}
