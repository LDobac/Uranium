//! `AgentMemory` (030-R1, R2): what the agent remembers across observations, built only from
//! its own vision. Enemy buildings are remembered as "last seen state + frames since seen".
//! Neutral resources are remembered the same way (030-R2 exception, 010-R5, E-3).

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

/// Top-left tile of a resource footprint: the key that matches observed resources, the map
/// precompute's resources and label targets (resources never move, 030 §2, AMB-030-3).
pub type ResourceTile = (i32, i32);

#[derive(Debug, Clone)]
pub struct AgentMemory {
    pub self_player: u8,
    pub enemies: Vec<u8>,
    /// From the lobby; Random stays unknown until an enemy unit is seen (030-R11).
    pub enemy_race: Race,
    /// Enemy buildings by id, last seen state (030 §2 entity group 3).
    pub enemy_buildings: BTreeMap<UnitId, Remembered>,
    /// Neutral resources by footprint top-left tile: last observed state and frame (030-R2
    /// exception, E-3). Also used by the cc_slot / refinery masks (034).
    pub resources: BTreeMap<ResourceTile, Remembered>,
    /// Resource tiles seen empty (mined out, or a geyser under a refinery): map resources there are
    /// not shown until a resource is seen there again.
    pub resources_gone: BTreeSet<ResourceTile>,
    /// 030 §4: maximum simultaneously observed enemy count per unit type.
    pub enemy_max_seen: HashMap<u16, u16>,
    /// 030 §1 entity_cat last_action_type for own units.
    pub unit_last_action: HashMap<UnitId, u16>,
    /// 030 §4: previous action type and delay bucket index.
    pub last_action: Option<(u16, usize)>,
    /// 031-R10: the previous act had a failed command (always false with `zero_last_failed`).
    pub last_failed: bool,
    /// 030-R10 / E-5: keep `last_failed` at 0. Why: replay playback has no command-failure
    /// events, so in BC training data `last_failed` is always 0. In evaluation games of a BC
    /// checkpoint (core ablation 032-R17, 041-AC2) a 1 would be an input the model never trained
    /// on and only adds noise to the evaluation, so those games set this flag. RL (042) leaves it
    /// off and uses the real value. Set by the agent from `InferOptions::zero_last_failed`.
    pub zero_last_failed: bool,
    pub frame: u32,
    /// Frame each 128x128 tile was last visible (`NEVER` = not yet), 030 `last_seen_age`.
    pub last_visible: Vec<u32>,
}

/// `AgentMemory::last_visible` for tiles never seen.
pub const NEVER: u32 = u32::MAX;

/// Footprint top-left tile of a resource of type `ty` centred at `pos` (same rule as `top_left`).
pub fn resource_tile(ty: u16, pos: (u16, u16)) -> Option<ResourceTile> {
    let t = TypeTables::get().unit(ty)?;
    Some(top_left(&UnitState { unit_type: ty, pos, ..Default::default() }, t))
}

impl AgentMemory {
    /// Enemy race from `GameInfo`: a lobby Random (or a race the bridge does not report) is
    /// unknown until the first enemy unit is seen (030-R11).
    pub fn new(info: &GameInfo) -> Self {
        Self::with_lobby(info, false)
    }

    /// 030-R11: `enemy_random` = the enemy picked Random in the lobby (replays: the shard's
    /// `enemy_random`, where `GameInfo` already holds the race actually played). Then the enemy
    /// race is unknown until the first enemy unit is observed, as in a live game.
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
            frame: 0,
            last_visible: vec![NEVER; super::schema::CELLS],
        }
    }

    pub fn is_enemy(&self, owner: u8) -> bool {
        self.enemies.contains(&owner)
    }

    /// Folds one observation into the memory. Only visible units are used (030-R2); `map`
    /// supplies the map-public resource positions (E-3).
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
        // 030-R10: see `zero_last_failed`.
        self.last_failed = !self.zero_last_failed && frame.events.iter().any(|e| matches!(e, GameEvent::CommandFailed { .. }));
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
                let key = top_left(u, t);
                self.resources_gone.remove(&key);
                self.resources.insert(key, Remembered { unit: u.clone(), seen_frame: frame.frame });
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
                self.unit_last_action.remove(id);
                let dead: Vec<ResourceTile> = self.resources.iter().filter(|(_, r)| r.unit.id == *id).map(|(k, _)| *k).collect();
                for k in dead {
                    self.resources.remove(&k);
                    self.resources_gone.insert(k);
                }
            }
        }
        self.enemy_buildings.retain(|id, r| {
            visible_ids.contains(id) || !frame.visibility.get(r.unit.pos.0 / 32, r.unit.pos.1 / 32)
        });
        // Resources whose tile is visible but that are not there were mined out (or a geyser now
        // carries a refinery).
        let missing: Vec<ResourceTile> = self
            .resources
            .iter()
            .filter(|(_, r)| !visible_ids.contains(&r.unit.id) && frame.visibility.get(r.unit.pos.0 / 32, r.unit.pos.1 / 32))
            .map(|(k, _)| *k)
            .collect();
        for k in missing {
            self.resources.remove(&k);
            self.resources_gone.insert(k);
        }
        // Map resources never seen whose tile is visible now without them: gone before we looked.
        for r in &map.resources {
            let Some(k) = resource_tile(r.unit_type, r.pos) else { continue };
            if !self.resources.contains_key(&k) && frame.visibility.get(r.pos.0 / 32, r.pos.1 / 32) {
                self.resources_gone.insert(k);
            }
        }
    }

    /// Records the policy's action (030 §4 previous action, §1 per-unit last action).
    pub fn record_action(&mut self, action: u16, delay_bucket: usize, units: &[UnitId]) {
        self.last_action = Some((action, delay_bucket));
        for &u in units {
            self.unit_last_action.insert(u, action);
        }
    }
}
