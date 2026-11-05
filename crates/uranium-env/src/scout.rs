//! Scouting and early-survival rewards (033-R9 r_scout, 033-R10 r_survive, AMB-042-16), the
//! reward channel mode (033-R11) and the 042 scouting / reaction metrics of `games.csv`.
//!
//! 033-R5: only the learning session's own observation is read — its own units and the enemy
//! units visible in that `RawFrame` (010-R5) — plus public map information (the map's start
//! locations and our own start tile). The enemy player's `start_tile`, `PrivilegedSummary` and
//! remembered (no longer visible) enemy buildings are never used: [`ScoutTracker::for_game`]
//! keeps only the enemy players' ids from `GameInfo`.
//!
//! Game time is 24 frames per second (042-R14): 3:00 = 4,320, 4:00 = 5,760, 6:00 = 8,640,
//! 8:00 = 11,520 frames.

use std::collections::BTreeSet;

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use uranium_proto::{GameInfo, RawFrame, Tile, UnitId, flags};

use crate::types::{TypeTables, UnitType};

/// Frames per game second (042-R14).
pub const FPS: f64 = 24.0;
/// 4:00 and 6:00 in frames (042 정찰·대응 지표).
pub const FRAME_4M: u32 = 5_760;
pub const FRAME_6M: u32 = 8_640;
/// 1:00 and 8:00 in frames (050-R7 세션 지표, AMB-042-19).
pub const FRAME_1M: u32 = 1_440;
pub const FRAME_8M: u32 = 11_520;

/// 033-R11 reward channel mode `[reward] pseudo` (AMB-042-16).
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum PseudoMode {
    /// The earlier meaning: build / units channels = r_build / r_units (z games only).
    #[default]
    Z,
    /// build channel = r_scout, units channel = r_survive; no z (033-R11).
    Scout,
}

/// `[reward.scout]` (033-R9). Defaults = the values the user confirmed on 2026-11-03.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(deny_unknown_fields, default)]
pub struct ScoutConfig {
    /// Policy weight (bot games; model games × `vs_model`).
    pub weight: f64,
    /// Per-game cap of the decayed raw values.
    pub cap: f64,
    /// Multiplier of the policy weight in games against a model (0..=1).
    pub vs_model: f64,
    /// Decay: 1 up to this game second, linear to 0 at `zero_at_s`, 0 after.
    pub full_until_s: f64,
    pub zero_at_s: f64,
    /// Enemy main found: an enemy building within this many tiles of a start location that is
    /// not ours (distance between tile centres).
    pub main_radius_tiles: f64,
    /// Enemy workers: at least this many visible in one observation.
    pub workers_min: u32,
    /// Raw values of the items.
    pub main: f64,
    pub building_type: f64,
    pub expansion: f64,
    pub workers: f64,
    pub army: f64,
}

impl Default for ScoutConfig {
    fn default() -> Self {
        Self { weight: 0.15, cap: 1.0, vs_model: 0.5, full_until_s: 240.0, zero_at_s: 480.0, main_radius_tiles: 10.0, workers_min: 4, main: 0.3, building_type: 0.1, expansion: 0.1, workers: 0.1, army: 0.1 }
    }
}

/// `[reward.survive]` (033-R10). Defaults = the values the user confirmed on 2026-11-03.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(deny_unknown_fields, default)]
pub struct SurviveConfig {
    /// Policy weight (bot games; model games × `vs_model`).
    pub weight: f64,
    /// Multiplier of the policy weight in games against a model (0 = bot games only).
    pub vs_model: f64,
    /// Window in game seconds (3:00 to 8:00).
    pub from_s: f64,
    pub until_s: f64,
    /// SCV count at which the reward rate is full (`min(n / scv_target, 1)`).
    pub scv_target: u32,
}

impl Default for SurviveConfig {
    fn default() -> Self {
        Self { weight: 0.1, vs_model: 0.0, from_s: 180.0, until_s: 480.0, scv_target: 20 }
    }
}

/// `[reward]` of the RL config (033-R11): the channel mode and the two rewards' settings.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
#[serde(deny_unknown_fields)]
pub struct RewardConfig {
    #[serde(default)]
    pub pseudo: PseudoMode,
    #[serde(default)]
    pub scout: ScoutConfig,
    #[serde(default)]
    pub survive: SurviveConfig,
}

impl RewardConfig {
    /// 033-R11 설정 키 검사: weights, raw values and `cap` ≥ 0, `vs_model` in 0..=1,
    /// `full_until_s` ≤ `zero_at_s`, `from_s` < `until_s`, `scv_target`, `workers_min` and
    /// `main_radius_tiles` ≥ 1. Checked whatever the mode (the metrics use the scout settings).
    pub fn validate(&self) -> Result<()> {
        let s = &self.scout;
        let v = &self.survive;
        let nonneg = [("[reward.scout] weight", s.weight), ("[reward.scout] cap", s.cap), ("[reward.scout] main", s.main), ("[reward.scout] building_type", s.building_type), ("[reward.scout] expansion", s.expansion), ("[reward.scout] workers", s.workers), ("[reward.scout] army", s.army), ("[reward.survive] weight", v.weight)];
        for (k, x) in nonneg {
            if !(x >= 0.0 && x.is_finite()) {
                bail!("{k} must be a finite value >= 0, got {x} (033-R11)");
            }
        }
        for (k, x) in [("[reward.scout] vs_model", s.vs_model), ("[reward.survive] vs_model", v.vs_model)] {
            if !(0.0..=1.0).contains(&x) {
                bail!("{k} must be between 0 and 1, got {x} (033-R11)");
            }
        }
        if !(s.full_until_s.is_finite() && s.zero_at_s.is_finite() && s.full_until_s >= 0.0 && s.full_until_s <= s.zero_at_s) {
            bail!("[reward.scout] needs 0 <= full_until_s <= zero_at_s, got {} and {} (033-R11)", s.full_until_s, s.zero_at_s);
        }
        if !(v.from_s.is_finite() && v.until_s.is_finite() && v.from_s >= 0.0 && v.from_s < v.until_s) {
            bail!("[reward.survive] needs 0 <= from_s < until_s, got {} and {} (033-R11)", v.from_s, v.until_s);
        }
        if v.scv_target < 1 {
            bail!("[reward.survive] scv_target must be at least 1 (033-R11)");
        }
        if s.workers_min < 1 {
            bail!("[reward.scout] workers_min must be at least 1 (033-R11)");
        }
        if !(s.main_radius_tiles >= 1.0 && s.main_radius_tiles.is_finite()) {
            bail!("[reward.scout] main_radius_tiles must be at least 1, got {} (033-R11)", s.main_radius_tiles);
        }
        Ok(())
    }

    /// 033-R11 / 042: policy-weight multipliers of the build and units channels for one game.
    /// `"z"`: 1 and 1 (r_build / r_units have no per-game rule). `"scout"`: 1 in bot games,
    /// `vs_model` of each reward in games against a model.
    pub fn pseudo_scale(&self, vs_model: bool) -> [f32; 2] {
        match (self.pseudo, vs_model) {
            (PseudoMode::Scout, true) => [self.scout.vs_model as f32, self.survive.vs_model as f32],
            _ => [1.0, 1.0],
        }
    }
}

/// 042 정찰·대응 지표 of one game (frames; `games.csv` shows game seconds).
#[derive(Serialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ScoutMetrics {
    /// Observation at which 033 "적 본진 발견" held.
    pub main_frame: Option<u32>,
    /// Distinct enemy building types seen in observations up to 4:00 / 6:00 (all of the game
    /// when it ended earlier).
    pub types_4m: u32,
    pub types_6m: u32,
    /// First observation with an own Barracks / Bunker (construction start, not completion).
    pub barracks_frame: Option<u32>,
    pub bunker_frame: Option<u32>,
    /// Own SCVs (033 r_survive count) at the first observation at or after 4:00.
    pub scv_4m: Option<u32>,
    /// 042-R17 / 020-R19 판정 SCV 수 (AMB-042-18): own SCVs (the same 033-R10 count: completed,
    /// loaded ones included, in production excluded; own units only, 033-R5) at the latest
    /// observation folded — the final frame (`GameEnd.final_frame`, 010-R12) when the session got
    /// one, else its last `RawFrame`; `None` before the first observation.
    pub scv_last: Option<u32>,
    /// 050-R7 `scv_min_1m` (AMB-042-19): the least own SCVs (the same 033-R10 count) over the
    /// observations at or after 1:00 (frame 1,440); `None` when the game ended before 1:00. Held
    /// (050-R7 막음) = `> 4`.
    pub scv_min_1m: Option<u32>,
    /// 050-R7 `army_s` (AMB-042-19): the observation at which 033-R9 "병력 첫 목격" held (the
    /// `army_found` item: an enemy unit that is not a worker, building, larva, egg, cocoon,
    /// overlord or spell); `None` = never seen.
    pub army_frame: Option<u32>,
    /// 050-R7 `scv_8m` (AMB-042-19, 메인 임시 결정): own SCVs (the same count) at the first
    /// observation at or after 8:00 (frame 11,520); `None` when the game ended before.
    pub scv_8m: Option<u32>,
}

/// Game seconds of a frame (042-R14 24 fps).
pub fn seconds(frame: u32) -> f64 {
    frame as f64 / FPS
}

/// What one observation gave: the raw (unweighted) r_scout after decay and cap, and r_survive.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ScoutStep {
    pub scout: f32,
    pub survive: f32,
}

/// Per-game state of r_scout, r_survive and the metrics (033-R9 / R10, 042).
#[derive(Debug, Clone)]
pub struct ScoutTracker {
    cfg: RewardConfig,
    self_player: u8,
    enemies: Vec<u8>,
    /// Map start locations that are not ours (public map information).
    starts: Vec<Tile>,
    main_found: bool,
    building_types: BTreeSet<u16>,
    depots: BTreeSet<UnitId>,
    expansion_found: bool,
    workers_found: bool,
    army_found: bool,
    /// Decayed raw r_scout paid so far (≤ cap).
    paid: f64,
    /// Frame of the previous observation (r_survive interval); the game starts at 0.
    prev_frame: u32,
    metrics: ScoutMetrics,
    scv: u16,
    barracks: u16,
    bunker: u16,
}

fn type_id(name: &str) -> u16 {
    TypeTables::get().unit_by_name(name).map_or(u16::MAX, |u| u.id as u16)
}

/// 033 "적 병력": not a worker, building, larva, egg (Egg, Lurker Egg, Cocoon) or overlord.
/// Spell effects (`Spell_*`, e.g. a Scanner Sweep) are not units of an army either.
fn is_army(t: &UnitType) -> bool {
    !(t.is_worker() || t.is_building() || t.name.starts_with("Spell_") || matches!(t.name.as_str(), "Zerg_Larva" | "Zerg_Egg" | "Zerg_Lurker_Egg" | "Zerg_Cocoon" | "Zerg_Overlord"))
}

/// 033 "적 병력" by unit type id (an id missing from the type table is not army). The 021-R8
/// teacher (`uranium-scripted`) reads its rush rule ③ and contact with this same definition.
pub fn is_army_type(id: u16) -> bool {
    TypeTables::get().unit(id).is_some_and(is_army)
}

/// [`is_army_type`] and not a flyer: what SCVs can hit (021-R8 SCV pull targets).
pub fn is_ground_army_type(id: u16) -> bool {
    TypeTables::get().unit(id).is_some_and(|t| is_army(t) && !t.is_flyer())
}

impl ScoutTracker {
    /// `self_player`, the enemy players' ids and the map start locations other than ours (`own`).
    pub fn new(self_player: u8, enemies: Vec<u8>, start_locations: &[Tile], own: Option<Tile>, cfg: RewardConfig) -> Self {
        let starts = start_locations.iter().copied().filter(|t| Some(*t) != own).collect();
        Self {
            cfg,
            self_player,
            enemies,
            starts,
            main_found: false,
            building_types: BTreeSet::new(),
            depots: BTreeSet::new(),
            expansion_found: false,
            workers_found: false,
            army_found: false,
            paid: 0.0,
            prev_frame: 0,
            metrics: ScoutMetrics::default(),
            scv: type_id("Terran_SCV"),
            barracks: type_id("Terran_Barracks"),
            bunker: type_id("Terran_Bunker"),
        }
    }

    /// 033-R5: from `GameInfo` only the public parts — the enemy players' ids, the map's start
    /// locations and our own start tile. The enemy's `start_tile` is never read.
    pub fn for_game(info: &GameInfo, cfg: RewardConfig) -> Self {
        let me = info.players.iter().find(|p| p.is_self);
        let self_player = me.map_or(info.self_player, |p| p.id);
        let own = me.and_then(|p| p.start_tile);
        let enemies = info.players.iter().filter(|p| p.is_enemy).map(|p| p.id).collect();
        Self::new(self_player, enemies, &info.static_map.start_locations, own, cfg)
    }

    /// The same game with other settings (a tracker before its first observation).
    pub fn with_config(mut self, cfg: RewardConfig) -> Self {
        self.cfg = cfg;
        self
    }

    pub fn metrics(&self) -> ScoutMetrics {
        self.metrics
    }

    /// 033-R9 time decay d(f).
    fn decay(&self, frame: u32) -> f64 {
        let s = &self.cfg.scout;
        let (full, zero) = (s.full_until_s * FPS, s.zero_at_s * FPS);
        let f = frame as f64;
        if f <= full {
            1.0
        } else if f >= zero {
            0.0
        } else {
            (zero - f) / (zero - full)
        }
    }

    /// Tile-centre distance from a unit's tile to the nearest start location that is not ours.
    fn near_enemy_start(&self, pos: (u16, u16)) -> bool {
        let r = self.cfg.scout.main_radius_tiles;
        let (tx, ty) = ((pos.0 / 32) as f64, (pos.1 / 32) as f64);
        self.starts.iter().any(|&(sx, sy)| {
            let (dx, dy) = (tx - sx as f64, ty - sy as f64);
            (dx * dx + dy * dy).sqrt() <= r
        })
    }

    /// Folds one observation (decision frame or the final frame, frames increasing).
    pub fn step(&mut self, frame: &RawFrame) -> ScoutStep {
        let types = TypeTables::get();
        let s = self.cfg.scout.clone();
        let f = frame.frame;
        let mut raw = 0.0f64;
        let (mut scvs, mut enemy_workers) = (0u32, 0u32);
        let (mut main_now, mut army_now) = (false, false);
        for u in &frame.units {
            if u.owner == self.self_player {
                if u.unit_type == self.scv && u.has(flags::COMPLETED) {
                    // 033-R10: completed SCVs, loaded ones included, in production excluded.
                    scvs += 1;
                } else if u.unit_type == self.barracks && self.metrics.barracks_frame.is_none() {
                    self.metrics.barracks_frame = Some(f);
                } else if u.unit_type == self.bunker && self.metrics.bunker_frame.is_none() {
                    self.metrics.bunker_frame = Some(f);
                }
                continue;
            }
            if !self.enemies.contains(&u.owner) {
                continue;
            }
            let Some(t) = types.unit(u.unit_type) else { continue };
            if t.is_building() {
                if self.building_types.insert(u.unit_type) {
                    raw += s.building_type;
                }
                if !self.main_found && self.near_enemy_start(u.pos) {
                    main_now = true;
                }
                if t.has(crate::types::flag::RESOURCE_DEPOT) {
                    self.depots.insert(u.id);
                }
            } else if t.is_worker() {
                enemy_workers += 1;
            } else if is_army(t) {
                army_now = true;
            }
        }
        if main_now && !self.main_found {
            self.main_found = true;
            self.metrics.main_frame = Some(f);
            raw += s.main;
        }
        if !self.expansion_found && self.depots.len() >= 2 {
            self.expansion_found = true;
            raw += s.expansion;
        }
        if !self.workers_found && enemy_workers >= s.workers_min {
            self.workers_found = true;
            raw += s.workers;
        }
        if army_now && !self.army_found {
            self.army_found = true;
            self.metrics.army_frame = Some(f);
            raw += s.army;
        }
        // 033-R9: decay at the observation's frame, then the per-game cap.
        let scout = (raw * self.decay(f)).min((s.cap - self.paid).max(0.0));
        self.paid += scout;
        // 042 metrics.
        let n = self.building_types.len() as u32;
        if f <= FRAME_4M {
            self.metrics.types_4m = n;
        }
        if f <= FRAME_6M {
            self.metrics.types_6m = n;
        }
        if self.metrics.scv_4m.is_none() && f >= FRAME_4M {
            self.metrics.scv_4m = Some(scvs);
        }
        // 042-R17: frames arrive in order, so this ends as the last observation's count.
        self.metrics.scv_last = Some(scvs);
        // 050-R7 (AMB-042-19): the least count from 1:00, the first count from 8:00.
        if f >= FRAME_1M {
            self.metrics.scv_min_1m = Some(self.metrics.scv_min_1m.map_or(scvs, |m| m.min(scvs)));
        }
        if self.metrics.scv_8m.is_none() && f >= FRAME_8M {
            self.metrics.scv_8m = Some(scvs);
        }
        // 033-R10: the seconds of (prev, f] inside the window, at this observation's SCV count.
        let v = &self.cfg.survive;
        let (lo, hi) = (v.from_s * FPS, v.until_s * FPS);
        let overlap = ((f as f64).min(hi) - (self.prev_frame as f64).max(lo)).max(0.0);
        let rate = (scvs as f64 / v.scv_target as f64).min(1.0);
        let survive = overlap / FPS / (v.until_s - v.from_s) * rate;
        self.prev_frame = self.prev_frame.max(f);
        ScoutStep { scout: scout as f32, survive: survive as f32 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uranium_proto::{BitGrid, PlayerState, UnitState};

    const ME: u8 = 0;
    const ENEMY: u8 = 1;
    /// Our start and the two other start locations (public map information).
    const OURS: Tile = (10, 10);
    const NORTH: Tile = (100, 10);
    const SOUTH: Tile = (100, 100);

    fn tracker(cfg: RewardConfig) -> ScoutTracker {
        ScoutTracker::new(ME, vec![ENEMY], &[OURS, NORTH, SOUTH], Some(OURS), cfg)
    }

    fn unit(id: UnitId, owner: u8, name: &str, tile: (u16, u16), completed: bool) -> UnitState {
        let t = TypeTables::get().unit_by_name(name).unwrap_or_else(|| panic!("{name}"));
        UnitState {
            id,
            owner,
            unit_type: t.id as u16,
            pos: (tile.0 * 32 + 16, tile.1 * 32 + 16),
            vel: (0, 0),
            angle: 0,
            hp: 1,
            shield: 0,
            energy: 0,
            order: 0,
            order_target: None,
            order_pos: None,
            ground_cd: 0,
            air_cd: 0,
            spell_cd: 0,
            flags: if completed { flags::COMPLETED } else { 0 },
            build_queue: [228; 5],
            remaining_train: 0,
            remaining_build: 0,
            remaining_research: 0,
            resources: 0,
            loaded_in: None,
            status_timers: Default::default(),
            spider_mines: 0,
            nuke_armed: false,
            has_addon: false,
        }
    }

    fn frame(f: u32, units: Vec<UnitState>) -> RawFrame {
        RawFrame { frame: f, self_state: PlayerState::default(), units, visibility: BitGrid::new(1, 1), explored: BitGrid::new(1, 1), creep: BitGrid::new(1, 1), events: vec![], effects: vec![] }
    }

    fn scvs(n: u32) -> Vec<UnitState> {
        (0..n).map(|i| unit(1000 + i, ME, "Terran_SCV", OURS, true)).collect()
    }

    fn close(a: f32, b: f64) -> bool {
        (a as f64 - b).abs() < 1e-5
    }

    /// 033-R9 / 033-AC3: every item once; the same type or the same building seen again gives 0;
    /// a morphed type (Hatchery → Lair) is a new type.
    #[test]
    fn scout_items_pay_once_033_r9() {
        let mut t = tracker(RewardConfig::default());
        // Enemy hatchery far from any start location: building type only.
        let hatch = unit(1, ENEMY, "Zerg_Hatchery", (60, 60), true);
        assert!(close(t.step(&frame(100, vec![hatch.clone()])).scout, 0.1));
        assert!(close(t.step(&frame(200, vec![hatch.clone()])).scout, 0.0), "same building again");
        let pool = unit(2, ENEMY, "Zerg_Spawning_Pool", (61, 60), true);
        let pool2 = unit(3, ENEMY, "Zerg_Spawning_Pool", (62, 60), true);
        assert!(close(t.step(&frame(300, vec![pool.clone(), pool2])).scout, 0.1), "one type, two buildings");
        let lair = unit(1, ENEMY, "Zerg_Lair", (60, 60), true);
        assert!(close(t.step(&frame(400, vec![lair])).scout, 0.1), "morphed type is new");
        // A second resource depot (other unit id): expansion + nothing for the known Hatchery type.
        let hatch2 = unit(4, ENEMY, "Zerg_Hatchery", (20, 80), true);
        assert!(close(t.step(&frame(500, vec![hatch2.clone()])).scout, 0.1), "expansion");
        assert!(close(t.step(&frame(600, vec![hatch2])).scout, 0.0));
        // Workers: 3 below workers_min, then 4.
        let drones = |n: u32| (0..n).map(|i| unit(50 + i, ENEMY, "Zerg_Drone", (60, 62), true)).collect::<Vec<_>>();
        assert!(close(t.step(&frame(700, drones(3))).scout, 0.0), "3 < workers_min");
        assert!(close(t.step(&frame(800, drones(4))).scout, 0.1), "workers_min");
        assert!(close(t.step(&frame(900, drones(6))).scout, 0.0), "once");
        // Larva, egg, overlord, cocoon are not army; a zergling is.
        let not_army = vec![unit(60, ENEMY, "Zerg_Larva", (60, 61), true), unit(61, ENEMY, "Zerg_Egg", (60, 61), true), unit(62, ENEMY, "Zerg_Overlord", (60, 61), true), unit(63, ENEMY, "Zerg_Cocoon", (60, 61), true), unit(64, ENEMY, "Zerg_Lurker_Egg", (60, 61), true)];
        assert!(close(t.step(&frame(1000, not_army)).scout, 0.0));
        assert!(close(t.step(&frame(1100, vec![unit(70, ENEMY, "Zerg_Zergling", (30, 30), true)])).scout, 0.1), "army");
        assert!(close(t.step(&frame(1200, vec![unit(71, ENEMY, "Zerg_Zergling", (30, 30), true)])).scout, 0.0));
        // Main: an enemy building near a start location that is not ours.
        assert!(close(t.step(&frame(1300, vec![unit(5, ENEMY, "Zerg_Extractor", (103, 12), true)])).scout, 0.4), "main 0.3 + new type 0.1");
        assert_eq!(t.metrics().main_frame, Some(1300));
        assert!(close(t.step(&frame(1400, vec![unit(6, ENEMY, "Zerg_Hatchery", (100, 100), true)])).scout, 0.0), "main once");
    }

    /// 033-R5 / 033-AC3: only what is in the observation counts. An enemy known only from the
    /// enemy's start tile or privileged information (not in `units`) gives nothing; a building that
    /// left vision and comes back is not new; neutral and own units never count; an enemy building
    /// near our own start location is not the enemy main.
    #[test]
    fn scout_uses_own_vision_only_033_r5() {
        use uranium_proto::{PlayerInfo, Race, StaticMap};
        let player = |id, start, is_self, is_enemy| PlayerInfo { id, name: String::new(), race: Race::Zerg, start_tile: Some(start), is_self, is_enemy };
        let info = GameInfo {
            map_name: String::new(),
            map_hash: [0; 32],
            map_size_tiles: (128, 128),
            self_player: ME,
            players: vec![player(ME, OURS, true, false), player(ENEMY, SOUTH, false, true)],
            static_map: StaticMap { width_tiles: 128, height_tiles: 128, walkable: BitGrid::new(1, 1), buildable: BitGrid::new(1, 1), ground_height: vec![], start_locations: vec![OURS, NORTH, SOUTH] },
            latency_frames: 0,
        };
        let mut t = ScoutTracker::for_game(&info, RewardConfig::default());
        assert_eq!(t.starts, vec![NORTH, SOUTH], "both other start locations, the enemy's start tile not singled out");
        for f in 0..20 {
            assert_eq!(t.step(&frame(f * 100, scvs(4))).scout, 0.0, "nothing visible, nothing paid");
        }
        // Neutral / own buildings near the other start locations: not the enemy's.
        let neutral = 11u8;
        assert_eq!(t.step(&frame(2100, vec![unit(9, neutral, "Zerg_Hatchery", SOUTH, true), unit(10, ME, "Terran_Bunker", NORTH, true)])).scout, 0.0);
        // An enemy bunker / creep colony near our own start: building type only, not the main.
        assert!(close(t.step(&frame(2200, vec![unit(11, ENEMY, "Zerg_Creep_Colony", (12, 11), true)])).scout, 0.1));
        assert_eq!(t.metrics().main_frame, None);
        let hatch = unit(1, ENEMY, "Zerg_Hatchery", (60, 60), true);
        t.step(&frame(2300, vec![hatch.clone()]));
        t.step(&frame(2400, vec![]));
        assert_eq!(t.step(&frame(2500, vec![hatch])).scout, 0.0, "seen again after leaving vision");
    }

    /// 033-AC3: the main radius boundary (tile-centre distance ≤ main_radius_tiles).
    #[test]
    fn scout_main_radius_boundary_033_r9() {
        let at = |tile: (u16, u16)| {
            let mut t = tracker(RewardConfig::default());
            t.step(&frame(10, vec![unit(1, ENEMY, "Zerg_Spawning_Pool", tile, true)]));
            t.metrics().main_frame.is_some()
        };
        assert!(at((110, 10)), "10 tiles");
        assert!(!at((111, 10)), "11 tiles");
        assert!(at((106, 18)), "6, 8: distance 10");
        assert!(!at((107, 18)), "7, 8: distance 10.6");
    }

    /// 033-AC3: decay boundaries (5,760 frames full, 11,520 and after 0, linear between) and the
    /// per-game cap.
    #[test]
    fn scout_decay_and_cap_033_r9() {
        let one = |f: u32| tracker(RewardConfig::default()).step(&frame(f, vec![unit(1, ENEMY, "Zerg_Spawning_Pool", (60, 60), true)])).scout;
        assert!(close(one(5_760), 0.1));
        assert!(close(one(8_640), 0.05), "half way");
        assert!(close(one(5_760 + 1_440), 0.075));
        assert_eq!(one(11_520), 0.0);
        assert_eq!(one(20_000), 0.0);
        // Cap 0.25: the third type is cut to 0.05, then nothing; items still reach the metrics.
        let mut cfg = RewardConfig::default();
        cfg.scout.cap = 0.25;
        let mut t = tracker(cfg);
        let names = ["Zerg_Spawning_Pool", "Zerg_Extractor", "Zerg_Hydralisk_Den", "Zerg_Spire"];
        let got: Vec<f32> = names.iter().enumerate().map(|(i, n)| t.step(&frame(100 * i as u32, vec![unit(i as u32, ENEMY, n, (60, 60), true)])).scout).collect();
        assert!(close(got[0], 0.1) && close(got[1], 0.1) && close(got[2], 0.05) && got[3] == 0.0, "{got:?}");
        assert_eq!(t.metrics().types_4m, 4, "metrics past the cap");
        t.step(&frame(500, vec![unit(9, ENEMY, "Zerg_Hatchery", NORTH, true)]));
        assert_eq!(t.metrics().main_frame, Some(500), "main recorded past the cap");
    }

    /// 033-R10 / 033-AC3: window boundaries, overlap seconds only, production excluded,
    /// buildings only = 0, `scv_target` cap.
    #[test]
    fn survive_window_and_count_033_r10() {
        let cfg = RewardConfig::default();
        let full = 1.0 / 300.0; // one second of the 300 s window at scv_target SCVs
        let mut t = tracker(cfg.clone());
        assert_eq!(t.step(&frame(4_320, scvs(20))).survive, 0.0, "up to 3:00: 0");
        assert!(close(t.step(&frame(4_344, scvs(20))).survive, full), "one second in the window");
        assert!(close(t.step(&frame(4_368, scvs(10))).survive, full * 0.5), "10 / 20");
        assert!(close(t.step(&frame(4_392, scvs(40))).survive, full), "capped at scv_target");
        // Production and buildings: 0.
        let building = vec![unit(1, ME, "Terran_Command_Center", OURS, true), unit(2, ME, "Terran_SCV", OURS, false)];
        assert_eq!(t.step(&frame(4_416, building)).survive, 0.0, "SCV in production, buildings only");
        // An interval across the window end counts its seconds up to 8:00.
        let mut t = tracker(cfg.clone());
        t.step(&frame(11_496, scvs(20)));
        assert!(close(t.step(&frame(11_544, scvs(20))).survive, full), "11,496..11,520 only");
        assert_eq!(t.step(&frame(11_600, scvs(20))).survive, 0.0, "after 8:00");
        // An interval across the window start: only the part after 4,320.
        let mut t = tracker(cfg);
        t.step(&frame(4_000, scvs(20)));
        assert!(close(t.step(&frame(4_368, scvs(20))).survive, 2.0 * full));
    }

    /// 033-AC3: the same game observed with other decision intervals gives the same total, and
    /// the whole window at `scv_target` SCVs is 1.
    #[test]
    fn survive_total_ignores_decision_intervals_033_r10() {
        let total = |step: u32| {
            let mut t = tracker(RewardConfig::default());
            let mut sum = 0.0f64;
            let mut f = 0;
            while f <= 14_000 {
                sum += t.step(&frame(f, scvs(20))).survive as f64;
                f += step;
            }
            sum
        };
        for step in [8, 24, 96, 240, 1_000, 3_333] {
            assert!((total(step) - 1.0).abs() < 1e-4, "step {step}: {}", total(step));
        }
    }

    /// 042 정찰·대응 지표 / 042-AC6: 4:00 / 6:00 boundaries (5,760 / 8,640 included), types until the
    /// end of a shorter game, barracks / bunker at construction start, `scv_4m` empty when the game
    /// ended before 4:00.
    #[test]
    fn metrics_boundaries_042() {
        let mut t = tracker(RewardConfig::default());
        let rax = unit(20, ME, "Terran_Barracks", OURS, false);
        t.step(&frame(2_400, vec![rax.clone()]));
        t.step(&frame(5_760, [vec![unit(1, ENEMY, "Zerg_Spawning_Pool", (60, 60), true), rax.clone()], scvs(11)].concat()));
        t.step(&frame(5_784, [vec![unit(2, ENEMY, "Zerg_Extractor", (60, 60), true), unit(21, ME, "Terran_Bunker", OURS, false)], scvs(12)].concat()));
        t.step(&frame(8_640, vec![unit(3, ENEMY, "Zerg_Lair", (60, 60), true)]));
        t.step(&frame(8_664, vec![unit(4, ENEMY, "Zerg_Spire", (60, 60), true)]));
        let m = t.metrics();
        assert_eq!((m.types_4m, m.types_6m), (1, 3));
        assert_eq!((m.barracks_frame, m.bunker_frame), (Some(2_400), Some(5_784)));
        assert_eq!(m.scv_4m, Some(11), "first observation at or after 5,760");
        assert_eq!(m.main_frame, None);
        // A game over at 3:00.
        let mut t = tracker(RewardConfig::default());
        t.step(&frame(3_000, vec![unit(1, ENEMY, "Zerg_Spawning_Pool", (60, 60), true)]));
        t.step(&frame(4_320, vec![unit(2, ENEMY, "Zerg_Hatchery", (60, 60), true)]));
        let m = t.metrics();
        assert_eq!((m.types_4m, m.types_6m, m.scv_4m, m.barracks_frame, m.bunker_frame), (2, 2, None, None, None));
        assert_eq!(seconds(5_760), 240.0);
    }

    /// 042-R17 / 042-AC8 판정 SCV 수 (AMB-042-18): the latest observation's own SCVs with the
    /// 033-R10 count — completed ones, loaded ones (in a bunker) included, in production excluded,
    /// enemy SCVs never; `None` before any observation.
    #[test]
    fn judge_scv_is_the_last_observation_042_r17() {
        let mut t = tracker(RewardConfig::default());
        assert_eq!(t.metrics().scv_last, None, "no observation");
        t.step(&frame(6_000, scvs(12)));
        assert_eq!(t.metrics().scv_last, Some(12));
        let mut loaded = unit(2_000, ME, "Terran_SCV", OURS, true);
        loaded.flags |= flags::LOADED;
        loaded.loaded_in = Some(21);
        let extra = vec![loaded, unit(2_001, ME, "Terran_SCV", OURS, false), unit(2_002, ENEMY, "Terran_SCV", (60, 60), true), unit(21, ME, "Terran_Bunker", OURS, true)];
        t.step(&frame(11_520, [scvs(7), extra].concat()));
        let m = t.metrics();
        assert_eq!(m.scv_last, Some(8), "7 + 1 loaded; the one in production and the enemy's not counted");
        assert_eq!(m.scv_4m, Some(12), "scv_4m stays the 4:00 value");
    }

    /// 050-R7 / 050-AC3 세션 지표 (AMB-042-19): `scv_min_1m` counts from 1:00 (an observation at
    /// 1,439 is not counted, 1,440 is), is the least count, counts loaded SCVs and not ones in
    /// production, and is empty for a game over before 1:00; `army_frame` is the first observation
    /// with an enemy army unit (workers, buildings, larvae, eggs, cocoons, overlords not), empty
    /// when never seen; `scv_8m` is the first count at or after 11,520 and empty for a game over
    /// before 8:00.
    #[test]
    fn session_metrics_050_r7() {
        let mut t = tracker(RewardConfig::default());
        t.step(&frame(1_000, scvs(2)));
        t.step(&frame(1_439, scvs(3)));
        assert_eq!(t.metrics().scv_min_1m, None, "before 1:00 nothing counts");
        t.step(&frame(1_440, scvs(9)));
        assert_eq!(t.metrics().scv_min_1m, Some(9), "1,440 counts");
        let mut loaded = unit(2_000, ME, "Terran_SCV", OURS, true);
        loaded.flags |= flags::LOADED;
        loaded.loaded_in = Some(21);
        let mixed = vec![loaded, unit(2_001, ME, "Terran_SCV", OURS, false), unit(2_002, ENEMY, "Terran_SCV", (60, 60), true), unit(21, ME, "Terran_Bunker", OURS, true)];
        t.step(&frame(3_000, [scvs(4), mixed].concat()));
        assert_eq!(t.metrics().scv_min_1m, Some(5), "4 + 1 loaded; production and the enemy's SCV not counted");
        t.step(&frame(4_000, scvs(12)));
        assert_eq!(t.metrics().scv_min_1m, Some(5), "the least, not the last");
        assert_eq!(t.metrics().scv_8m, None);
        t.step(&frame(11_519, scvs(13)));
        assert_eq!(t.metrics().scv_8m, None, "11,519 is before 8:00");
        t.step(&frame(11_520, scvs(14)));
        t.step(&frame(11_544, scvs(3)));
        let m = t.metrics();
        assert_eq!((m.scv_8m, m.scv_min_1m), (Some(14), Some(3)), "scv_8m at 11,520; the minimum goes on after 8:00");
        // A game over before 1:00: both empty.
        let mut t = tracker(RewardConfig::default());
        t.step(&frame(1_200, scvs(4)));
        assert_eq!((t.metrics().scv_min_1m, t.metrics().scv_8m), (None, None));
        // army: not for workers, buildings, larvae, eggs, cocoons, overlords; the first army unit.
        let mut t = tracker(RewardConfig::default());
        let not_army = vec![unit(1, ENEMY, "Zerg_Drone", (60, 60), true), unit(2, ENEMY, "Zerg_Spawning_Pool", (60, 60), true), unit(3, ENEMY, "Zerg_Larva", (60, 60), true), unit(4, ENEMY, "Zerg_Egg", (60, 60), true), unit(5, ENEMY, "Zerg_Cocoon", (60, 60), true), unit(6, ENEMY, "Zerg_Overlord", (60, 60), true), unit(7, ME, "Terran_Marine", OURS, true)];
        t.step(&frame(2_000, not_army));
        assert_eq!(t.metrics().army_frame, None, "never seen");
        t.step(&frame(4_440, vec![unit(8, ENEMY, "Zerg_Zergling", (30, 30), true)]));
        t.step(&frame(4_500, vec![unit(9, ENEMY, "Zerg_Zergling", (30, 30), true)]));
        assert_eq!(t.metrics().army_frame, Some(4_440), "the first sighting");
        assert_eq!(seconds(4_440), 185.0);
    }

    /// 033-R11 설정 키 검사 and the per-game multipliers.
    #[test]
    fn reward_config_checks_033_r11() {
        let c = RewardConfig::default();
        c.validate().unwrap();
        assert_eq!(c.pseudo, PseudoMode::Z);
        assert_eq!(c.pseudo_scale(true), [1.0, 1.0], "z: no per-game multiplier");
        let s = RewardConfig { pseudo: PseudoMode::Scout, ..Default::default() };
        assert_eq!((s.pseudo_scale(false), s.pseudo_scale(true)), ([1.0, 1.0], [0.5, 0.0]));
        let bad: [fn(&mut RewardConfig); 12] = [
            |c| c.scout.weight = -0.1,
            |c| c.scout.cap = -1.0,
            |c| c.scout.main = -0.3,
            |c| c.scout.army = f64::NAN,
            |c| c.survive.weight = -0.1,
            |c| c.scout.vs_model = 1.5,
            |c| c.survive.vs_model = -0.5,
            |c| c.scout.full_until_s = 500.0,
            |c| c.survive.from_s = 480.0,
            |c| c.survive.scv_target = 0,
            |c| c.scout.workers_min = 0,
            |c| c.scout.main_radius_tiles = 0.5,
        ];
        for (i, f) in bad.iter().enumerate() {
            let mut c = RewardConfig::default();
            f(&mut c);
            assert!(c.validate().is_err(), "case {i}");
        }
        let mut edge = RewardConfig::default();
        (edge.scout.full_until_s, edge.scout.zero_at_s, edge.scout.vs_model, edge.survive.vs_model) = (300.0, 300.0, 0.0, 1.0);
        edge.validate().unwrap();
        // TOML: modes, unknown keys and values.
        let parse = |s: &str| toml::from_str::<RewardConfig>(s);
        assert_eq!(parse("pseudo = \"scout\"").unwrap().pseudo, PseudoMode::Scout);
        assert_eq!(parse("").unwrap(), RewardConfig::default());
        assert!(parse("pseudo = \"zz\"").is_err());
        assert!(parse("pseudo = \"Scout\"").is_err());
        assert!(parse("[scout]\nweigth = 0.1").is_err(), "unknown key");
        assert_eq!(parse("[scout]\nweight = 0.2").unwrap().scout, ScoutConfig { weight: 0.2, ..Default::default() });
    }
}
