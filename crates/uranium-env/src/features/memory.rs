use std::collections::{BTreeMap, BTreeSet, HashMap};

use uranium_proto::{GameEvent, GameInfo, Race, RawFrame, UnitId, UnitState};

use super::map::MapInfo;
use super::world::top_left;
use crate::types::TypeTables;

#[derive(Debug, Clone)]
pub struct Remembered {
    pub unit: UnitState,
    pub seen_frame: u32,
}

pub type ResourceTile = (i32, i32);

#[derive(Debug, Clone)]
pub struct AgentMemory {
    pub self_player: u8,
    pub enemies: Vec<u8>,
    pub enemy_race: Race,
    pub enemy_buildings: BTreeMap<UnitId, Remembered>,
    pub resources: BTreeMap<ResourceTile, Remembered>,
    pub resources_gone: BTreeSet<ResourceTile>,
    pub enemy_max_seen: HashMap<u16, u16>,
    pub unit_last_action: HashMap<UnitId, u16>,
    pub last_action: Option<(u16, usize)>,
    pub last_failed: bool,
    pub zero_last_failed: bool,
    pub replay_labels: bool,
    pub frame: u32,
    pub last_visible: Vec<u32>,
}

pub const NEVER: u32 = u32::MAX;

pub fn resource_tile(ty: u16, pos: (u16, u16)) -> Option<ResourceTile> {
    let t = TypeTables::get().unit(ty)?;
    Some(top_left(&UnitState { unit_type: ty, pos, ..Default::default() }, t))
}

impl AgentMemory {
    pub fn new(info: &GameInfo) -> Self {
        Self::with_lobby(info, false)
    }

    pub fn with_lobby(info: &GameInfo, enemy_random: bool) -> Self {
        let enemies: Vec<u8> = info.players.iter().filter(|p| p.is_enemy).map(|p| p.id).collect();
        let enemy_race = if enemy_random {
            Race::Unknown
        } else {
            info.players.iter().find(|p| p.is_enemy).map_or(Race::Unknown, |p| match p.race {
                Race::Random => Race::Unknown,
                r => r,
            })
        };
        Self {
            self_player: info.self_player,
            enemies,
            enemy_race,
            enemy_buildings: BTreeMap::new(),
            resources: BTreeMap::new(),
            resources_gone: BTreeSet::new(),
            enemy_max_seen: HashMap::new(),
            unit_last_action: HashMap::new(),
            last_action: None,
            last_failed: false,
            zero_last_failed: false,
            replay_labels: false,
            frame: 0,
            last_visible: vec![NEVER; super::schema::CELLS],
        }
    }

    pub fn is_enemy(&self, owner: u8) -> bool {
        self.enemies.contains(&owner)
    }

    pub fn observe(&mut self, frame: &RawFrame, types: &TypeTables, map: &MapInfo) {
        self.frame = frame.frame;
        let (w, h) = (frame.visibility.width.min(128) as usize, frame.visibility.height.min(128) as usize);
        for y in 0..h {
            for x in 0..w {
                if frame.visibility.get(x as u16, y as u16) {
                    self.last_visible[y * super::schema::MAP + x] = frame.frame;
                }
            }
        }
        self.last_failed = !self.zero_last_failed && frame.events.iter().any(|e| matches!(e, GameEvent::CommandFailed { .. }));
        let mut counts: HashMap<u16, u16> = HashMap::new();
        let mut neutral_resources: std::collections::HashSet<UnitId> = std::collections::HashSet::new();
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
                let key = top_left(u, t);
                neutral_resources.insert(u.id);
                self.resources_gone.remove(&key);
                self.resources.insert(key, Remembered { unit: u.clone(), seen_frame: frame.frame });
            }
        }
        for (ty, n) in counts {
            let m = self.enemy_max_seen.entry(ty).or_default();
            *m = (*m).max(n);
        }
        let visible_ids: std::collections::HashSet<UnitId> = frame.units.iter().map(|u| u.id).collect();
        for e in &frame.events {
            if let GameEvent::UnitDestroyed(id) = e {
                self.enemy_buildings.remove(id);
                self.unit_last_action.remove(id);
                let dead: Vec<ResourceTile> =
                    self.resources.iter().filter(|(_, r)| r.unit.id == *id && !neutral_resources.contains(id)).map(|(k, _)| *k).collect();
                for k in dead {
                    self.resources.remove(&k);
                    self.resources_gone.insert(k);
                }
            }
        }
        self.enemy_buildings.retain(|id, r| {
            visible_ids.contains(id) || !frame.visibility.get(r.unit.pos.0 / 32, r.unit.pos.1 / 32)
        });
        let missing: Vec<ResourceTile> = self
            .resources
            .iter()
            .filter(|(_, r)| {
                let id = r.unit.id;
                if visible_ids.contains(&id) {
                    !neutral_resources.contains(&id)
                } else {
                    frame.visibility.get(r.unit.pos.0 / 32, r.unit.pos.1 / 32)
                }
            })
            .map(|(k, _)| *k)
            .collect();
        for k in missing {
            self.resources.remove(&k);
            self.resources_gone.insert(k);
        }
        for r in &map.resources {
            let Some(k) = resource_tile(r.unit_type, r.pos) else { continue };
            if !self.resources.contains_key(&k) && frame.visibility.get(r.pos.0 / 32, r.pos.1 / 32) {
                self.resources_gone.insert(k);
            }
        }
    }

    pub fn record_action(&mut self, action: u16, delay_bucket: usize, units: &[UnitId]) {
        self.last_action = Some((action, delay_bucket));
        for &u in units {
            self.unit_last_action.insert(u, action);
        }
    }
}
