//! `AgentMemory` (030-R1, R2): what the agent remembers across observations, built only from
//! its own vision. Enemy buildings are remembered as "last seen state + frames since seen".

use std::collections::{BTreeMap, HashMap};

use uranium_proto::{GameEvent, GameInfo, Race, RawFrame, UnitId, UnitState};

use crate::types::TypeTables;

#[derive(Debug, Clone)]
pub struct Remembered {
    pub unit: UnitState,
    pub seen_frame: u32,
}

#[derive(Debug, Clone)]
pub struct AgentMemory {
    pub self_player: u8,
    pub enemies: Vec<u8>,
    /// From the lobby; Random stays unknown until an enemy unit is seen (030 §1).
    pub enemy_race: Race,
    /// Enemy buildings by id, last seen state (030 §2 entity group 3).
    pub enemy_buildings: BTreeMap<UnitId, Remembered>,
    /// Neutral resources by id (static; used by the cc_slot / refinery masks, 034).
    pub resources: BTreeMap<UnitId, UnitState>,
    /// 030 §4: maximum simultaneously observed enemy count per unit type.
    pub enemy_max_seen: HashMap<u16, u16>,
    /// 030 §1 entity_cat last_action_type for own units.
    pub unit_last_action: HashMap<UnitId, u16>,
    /// 030 §4: previous action type and delay bucket index.
    pub last_action: Option<(u16, usize)>,
    /// 031-R10: the previous act had a failed command.
    pub last_failed: bool,
    pub frame: u32,
}

impl AgentMemory {
    pub fn new(info: &GameInfo) -> Self {
        let enemies: Vec<u8> = info.players.iter().filter(|p| p.is_enemy).map(|p| p.id).collect();
        let enemy_race = info.players.iter().find(|p| p.is_enemy).map_or(Race::Unknown, |p| match p.race {
            Race::Random => Race::Unknown,
            r => r,
        });
        Self {
            self_player: info.self_player,
            enemies,
            enemy_race,
            enemy_buildings: BTreeMap::new(),
            resources: BTreeMap::new(),
            enemy_max_seen: HashMap::new(),
            unit_last_action: HashMap::new(),
            last_action: None,
            last_failed: false,
            frame: 0,
        }
    }

    pub fn is_enemy(&self, owner: u8) -> bool {
        self.enemies.contains(&owner)
    }

    /// Folds one observation into the memory. Only visible units are used (030-R2).
    pub fn observe(&mut self, frame: &RawFrame, types: &TypeTables) {
        self.frame = frame.frame;
        self.last_failed = frame.events.iter().any(|e| matches!(e, GameEvent::CommandFailed { .. }));
        let mut counts: HashMap<u16, u16> = HashMap::new();
        for u in &frame.units {
            let Some(t) = types.unit(u.unit_type) else { continue };
            if self.is_enemy(u.owner) {
                *counts.entry(u.unit_type).or_default() += 1;
                if self.enemy_race == Race::Unknown {
                    self.enemy_race = match t.race.as_str() {
                        "Terran" => Race::Terran,
                        "Zerg" => Race::Zerg,
                        "Protoss" => Race::Protoss,
                        _ => Race::Unknown,
                    };
                }
                if t.is_building() {
                    self.enemy_buildings.insert(u.id, Remembered { unit: u.clone(), seen_frame: frame.frame });
                }
            } else if u.owner != self.self_player && t.is_resource() {
                self.resources.insert(u.id, u.clone());
            }
        }
        for (ty, n) in counts {
            let m = self.enemy_max_seen.entry(ty).or_default();
            *m = (*m).max(n);
        }
        // Forget remembered buildings that are gone: destroyed events, or the tile is visible now
        // and the building is not there (destroyed or lifted off).
        let visible_ids: std::collections::HashSet<UnitId> = frame.units.iter().map(|u| u.id).collect();
        for e in &frame.events {
            if let GameEvent::UnitDestroyed(id) = e {
                self.enemy_buildings.remove(id);
                self.resources.remove(id);
                self.unit_last_action.remove(id);
            }
        }
        self.enemy_buildings.retain(|id, r| {
            visible_ids.contains(id) || !frame.visibility.get(r.unit.pos.0 / 32, r.unit.pos.1 / 32)
        });
        // Mineral fields that are visible-tile but missing were mined out.
        self.resources.retain(|id, r| visible_ids.contains(id) || !frame.visibility.get(r.pos.0 / 32, r.pos.1 / 32));
    }

    /// Records the policy's action (030 §4 previous action, §1 per-unit last action).
    pub fn record_action(&mut self, action: u16, delay_bucket: usize, units: &[UnitId]) {
        self.last_action = Some((action, delay_bucket));
        for &u in units {
            self.unit_last_action.insert(u, action);
        }
    }
}
