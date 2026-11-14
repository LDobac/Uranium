//! T0 rule-based opponents (spec 021). They emit spec-010 `Command`s directly (021-R6).
//!
//! One shared "brain" (workers, supply, production, memory, attack waves, hunting) is
//! parameterized by a per-race kit. M3 builds are gas-free so every race reliably plays a
//! full melee game to the end (021-R1): Terran marines, Protoss zealots, Zerg zerglings.
//!
//! E-Regression scenario bots (050) reuse the brain with a tech path, gas, a special unit and
//! (Terran) a drop tactic: each one plays only its strategy (`T0Build`).
//!
//! The 021-R8 scouting-reactive Terran teacher (`T0Build::TerranReact`) is the Terran brain plus
//! the `react` module: it scouts, identifies a rush from its own vision and defends only then.
//!
//! The 022 TvT / TvP teacher (`T0Build::TerranTp`) reuses that teacher's parts (scouting, bunker,
//! SCV pull and repair, worker answer, construction resumption) with its own opening (gas,
//! factory, siege tanks), rush rules, branches, siege rule and attack (`tp` module). Against a
//! lobby Zerg its session is the 021-R8 teacher (022-R2).

pub mod placement;
pub mod react;
pub mod tp;
pub mod types;

use std::collections::BTreeMap;

use uranium_proto::*;

use crate::placement::Placer;
pub use crate::react::{Branch, MapHints, ReactParams, ReactStats, RushRule};
use crate::react::ReactState;
pub use crate::tp::{TpParams, TpRule, TpStats};
use crate::tp::TpState;
use crate::types::*;

const STEP_FRAMES: u32 = 8;
const WAVE_PERIOD: u32 = 48;
const PENDING_TIMEOUT: u32 = 24 * 25;
/// Hunting grid cell size in tiles.
const CELL: u16 = 8;
/// Marines per drop (one dropship).
const DROP_SIZE: usize = 8;
/// A drop leaves half-full after this long.
const LOAD_TIMEOUT: u32 = 24 * 20;
/// Enemies this close to the base (pixels) pull the home guard.
const DEFEND_RADIUS: i64 = 20 * 32;

/// A scripted bot controlling one player.
pub trait Bot: Send {
    fn act(&mut self, f: &RawFrame) -> Act;
    /// 021-R8 teacher record of the game so far (050-R8 session record); other builds have none.
    fn react_stats(&self) -> Option<ReactStats> {
        None
    }
    /// 022 teacher record of the game so far (`scripted.tp`, 022-R10); other builds have none.
    fn tp_stats(&self) -> Option<TpStats> {
        None
    }
}

/// Which T0 bot plays: the standard M3 build of the race, or an E-Regression scenario bot
/// that uses only one strategy (050 E-Regression).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum T0Build {
    #[default]
    Standard,
    /// 5-pool zergling rush.
    ZergRush,
    /// Marine drops with dropships.
    TerranDrop,
    /// Dark templar.
    ProtossDt,
    /// Mutalisks.
    ZergMuta,
    /// 021-R8 scouting-reactive Terran teacher (AMB-021-4): defends only after it sees a rush.
    TerranReact,
    /// 022 TvT / TvP teacher (021-R9): gas, factory and siege tanks; against a lobby Zerg the
    /// session is the 021-R8 teacher (022-R2).
    TerranTp,
}

impl T0Build {
    pub const SCENARIOS: [T0Build; 4] = [T0Build::ZergRush, T0Build::TerranDrop, T0Build::ProtossDt, T0Build::ZergMuta];

    /// 021-R3: the build part of the opponent id (`t0-<race>-<tag>@<version>`); the standard
    /// build has none, so its id stays `t0-<race>@<version>`.
    pub fn id_tag(self) -> Option<&'static str> {
        match self {
            T0Build::Standard => None,
            T0Build::ZergRush => Some("rush"),
            T0Build::TerranDrop => Some("drop"),
            T0Build::ProtossDt => Some("dt"),
            T0Build::ZergMuta => Some("muta"),
            T0Build::TerranReact => Some("react"),
            T0Build::TerranTp => Some("tp"),
        }
    }

    /// The race a scenario build plays (`None` = any race).
    pub fn race(self) -> Option<Race> {
        match self {
            T0Build::Standard => None,
            T0Build::ZergRush | T0Build::ZergMuta => Some(Race::Zerg),
            T0Build::TerranDrop | T0Build::TerranReact | T0Build::TerranTp => Some(Race::Terran),
            T0Build::ProtossDt => Some(Race::Protoss),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            T0Build::Standard => "standard",
            T0Build::ZergRush => "zerg_rush",
            T0Build::TerranDrop => "terran_drop",
            T0Build::ProtossDt => "protoss_dt",
            T0Build::ZergMuta => "zerg_muta",
            T0Build::TerranReact => "terran_react",
            T0Build::TerranTp => "terran_tp",
        }
    }

    /// 021-R8 / 022-R10: the build reads the map precompute (base locations, chokes).
    pub fn needs_map_hints(self) -> bool {
        matches!(self, T0Build::TerranReact | T0Build::TerranTp)
    }

    /// 022-R2: the teacher session of a game. `terran_tp` against a lobby Zerg is the 021-R8
    /// teacher; against Terran, Protoss or a lobby Random (resolved later, never switched) it
    /// stays `terran_tp`. Every other build is unchanged.
    pub fn for_lobby(self, info: &GameInfo) -> T0Build {
        let enemy = info.players.iter().find(|p| p.is_enemy).map(|p| p.race);
        match self {
            T0Build::TerranTp if enemy == Some(Race::Zerg) => T0Build::TerranReact,
            b => b,
        }
    }
}

/// Picks the T0 bot for our race (a Random lobby pick is already resolved at game start).
pub fn for_game(info: &GameInfo) -> Box<dyn Bot> {
    for_game_build(info, T0Build::Standard)
}

/// Like `for_game` with a build. A scenario build whose race we did not get (Random lobby
/// pick) falls back to the standard build of the race we got.
pub fn for_game_build(info: &GameInfo, build: T0Build) -> Box<dyn Bot> {
    for_game_hints(info, build, MapHints::default())
}

/// Like `for_game_build` with the map precompute hints the 021-R8 teacher uses (other builds
/// ignore them).
pub fn for_game_hints(info: &GameInfo, build: T0Build, hints: MapHints) -> Box<dyn Bot> {
    let race = info.players.iter().find(|p| p.is_self).map(|p| p.race).unwrap_or(Race::Terran);
    // 022-R2: the lobby enemy race picks the teacher.
    Box::new(Brain::with_hints(info, Kit::for_build(race, build.for_lobby(info)), &hints))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tactic {
    /// Attack-move waves of `attacker` units.
    Waves,
    /// Marines ride dropships into the enemy base (050 terran drop).
    Drop,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Supply {
    /// Built by a worker (depot, pylon).
    Building(u16),
    /// Trained from larva (overlord).
    Larva(u16),
}

/// Per-race build parameters.
#[derive(Clone, Debug)]
pub struct Kit {
    pub race: Race,
    pub worker: u16,
    pub max_workers: usize,
    pub supply: Supply,
    pub production: u16,
    pub production_cap: usize,
    /// Workers required before each production building (index = how many exist).
    pub production_at_workers: [usize; 3],
    pub army: u16,
    pub first_wave: usize,
    /// Units sent in attack waves once `first_wave` of them exist.
    pub attacker: u16,
    /// Most `army` units trained (scenario bots keep a small home guard).
    pub army_cap: usize,
    /// Tech buildings, each started once the previous one is complete, after the first
    /// production building is complete. Gas buildings go on a geyser; LAIR morphs the hatchery.
    pub tech: Vec<u16>,
    /// Workers kept on gas once the gas building is complete.
    pub gas_workers: usize,
    /// (producer, unit, cap): trained before `army` once every `tech` building is complete.
    pub special: Option<(u16, u16, usize)>,
    pub tactic: Tactic,
    /// One worker looks for the enemy start once a production building exists.
    pub scout: bool,
    /// The home guard (`army` units outside waves and drops) attack-moves on enemies near the base.
    pub defend: bool,
    /// 021-R8 teacher: opening, scouting, rush rule and branches (`react` module). `None` for
    /// every other build.
    pub react: Option<ReactParams>,
    /// 022 teacher (`tp` module, with `react` for the parts it reuses). `None` for every other
    /// build.
    pub tp: Option<TpParams>,
}

impl Kit {
    /// A scenario build whose race we did not get falls back to the standard build.
    pub fn for_build(race: Race, build: T0Build) -> Self {
        let build = if build.race() == Some(race) { build } else { T0Build::Standard };
        match (build, race) {
            (T0Build::ZergRush, _) => Kit::zerg_rush(),
            (T0Build::ZergMuta, _) => Kit::zerg_muta(),
            (T0Build::TerranDrop, _) => Kit::terran_drop(),
            (T0Build::ProtossDt, _) => Kit::protoss_dt(),
            (T0Build::TerranReact, _) => Kit::terran_react(),
            (T0Build::TerranTp, _) => Kit::terran_tp(),
            (T0Build::Standard, Race::Protoss) => Kit::protoss(),
            (T0Build::Standard, Race::Zerg) => Kit::zerg(),
            (T0Build::Standard, _) => Kit::terran(),
        }
    }

    pub fn terran() -> Self {
        Self {
            race: Race::Terran,
            worker: SCV,
            max_workers: 22,
            supply: Supply::Building(DEPOT),
            production: BARRACKS,
            production_cap: 3,
            production_at_workers: [10, 14, 14],
            army: MARINE,
            first_wave: 10,
            attacker: MARINE,
            army_cap: usize::MAX,
            tech: Vec::new(),
            gas_workers: 0,
            special: None,
            tactic: Tactic::Waves,
            scout: false,
            defend: false,
            react: None,
            tp: None,
        }
    }
    pub fn protoss() -> Self {
        Self {
            race: Race::Protoss,
            worker: PROBE,
            max_workers: 20,
            supply: Supply::Building(PYLON),
            production: GATEWAY,
            production_cap: 3,
            production_at_workers: [10, 13, 15],
            army: ZEALOT,
            first_wave: 6,
            attacker: ZEALOT,
            army_cap: usize::MAX,
            tech: Vec::new(),
            gas_workers: 0,
            special: None,
            tactic: Tactic::Waves,
            scout: false,
            defend: false,
            react: None,
            tp: None,
        }
    }
    pub fn zerg() -> Self {
        Self {
            race: Race::Zerg,
            worker: DRONE,
            max_workers: 14,
            supply: Supply::Larva(OVERLORD),
            production: SPAWNING_POOL,
            production_cap: 1,
            production_at_workers: [9, usize::MAX, usize::MAX],
            army: ZERGLING,
            first_wave: 12,
            attacker: ZERGLING,
            army_cap: usize::MAX,
            tech: Vec::new(),
            gas_workers: 0,
            special: None,
            tactic: Tactic::Waves,
            scout: false,
            defend: false,
            react: None,
            tp: None,
        }
    }

    /// 050 E-Regression "zerg early rush": spawning pool at 5 drones, zerglings only.
    pub fn zerg_rush() -> Self {
        Self { max_workers: 6, production_at_workers: [5, usize::MAX, usize::MAX], first_wave: 6, ..Self::zerg() }
    }

    /// 050 E-Regression "zerg mutalisk": extractor, lair, spire, then mutalisk waves.
    /// A few zerglings guard the base.
    pub fn zerg_muta() -> Self {
        Self {
            max_workers: 16,
            first_wave: 5,
            attacker: MUTALISK,
            army_cap: 6,
            tech: vec![EXTRACTOR, LAIR, SPIRE],
            gas_workers: 3,
            special: Some((LARVA, MUTALISK, usize::MAX)),
            defend: true,
            ..Self::zerg()
        }
    }

    /// 050 E-Regression "protoss dark templar": core, citadel, archives, then dark templar
    /// waves from two gateways. A few zealots guard the base.
    pub fn protoss_dt() -> Self {
        Self {
            production_cap: 2,
            first_wave: 2,
            attacker: DARK_TEMPLAR,
            army_cap: 3,
            tech: vec![ASSIMILATOR, CYBERNETICS_CORE, CITADEL, TEMPLAR_ARCHIVES],
            gas_workers: 3,
            special: Some((GATEWAY, DARK_TEMPLAR, usize::MAX)),
            defend: true,
            ..Self::protoss()
        }
    }

    /// 050 E-Regression "terran drop": marines plus two dropships that carry them into the
    /// enemy base. Marines not on a drop guard the base.
    pub fn terran_drop() -> Self {
        Self {
            production_cap: 2,
            production_at_workers: [10, 16, usize::MAX],
            tech: vec![REFINERY, FACTORY, STARPORT, CONTROL_TOWER],
            gas_workers: 3,
            special: Some((STARPORT, DROPSHIP, 2)),
            tactic: Tactic::Drop,
            scout: true,
            defend: true,
            ..Self::terran()
        }
    }

    /// 021-R8 scouting-reactive teacher (AMB-021-4): the `terran()` macro (marines, 3 barracks,
    /// 10-marine waves) with a home guard; the opening, barracks cap per branch, scouting, rush
    /// rule, bunkers and SCV jobs come from `react` (`ReactParams`, 6th advisory values).
    pub fn terran_react() -> Self {
        Self { defend: true, react: Some(ReactParams::default()), ..Self::terran() }
    }

    /// 022 TvT / TvP teacher: the 021-R8 teacher's parts (`ReactParams`: opening depots and
    /// barracks, scouting, bunkers, SCV pull / repair, worker answer, resumption, home guard) with
    /// the `tp` module's opening, rush rules, branches, siege tanks and attack (`TpParams`).
    pub fn terran_tp() -> Self {
        let tp = TpParams::default();
        Self { defend: true, react: Some(tp.react_params()), tp: Some(tp), ..Self::terran() }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DropPhase {
    Load { since: u32 },
    Fly,
    Fight,
}

struct Drop {
    ship: UnitId,
    squad: Vec<UnitId>,
    phase: DropPhase,
    last_order: u32,
}

struct Pending {
    ty: u16,
    tile: Tile,
    worker: UnitId,
    issued: u32,
}

pub struct Brain {
    kit: Kit,
    me: u8,
    starts: Vec<Tile>,
    walkable: BitGrid,
    placer: Placer,
    pending: Vec<Pending>,
    /// Supply units ordered from larva (zerg): frame of the last order.
    last_supply_order: u32,
    /// Last seen enemy building positions and types (own vision only, 030-R2 style memory).
    enemy_buildings: BTreeMap<UnitId, (Pos, u16)>,
    scouted: Vec<bool>,
    /// Frame each hunting cell was last visible.
    cell_seen: Vec<u32>,
    cells_w: u16,
    cells_h: u16,
    last_wave: u32,
    gas_crew: Vec<UnitId>,
    scout: Option<UnitId>,
    scout_done: bool,
    scout_order: u32,
    drops: Vec<Drop>,
    last_defend: u32,
    /// Centre of the own start command center (public map information).
    home: Option<Pos>,
    map_center: Pos,
    /// 021-R8 teacher state (`Kit::react`).
    react: Option<ReactState>,
    /// 022 teacher state (`Kit::tp`).
    tp: Option<TpState>,
}

impl Brain {
    pub fn new(info: &GameInfo, kit: Kit) -> Self {
        Self::with_hints(info, kit, &MapHints::default())
    }

    /// `hints`: the 031 map precompute's base locations and chokes (the 021-R8 teacher's natural
    /// and bunker direction; other kits ignore them).
    pub fn with_hints(info: &GameInfo, kit: Kit, hints: &MapHints) -> Self {
        let me = info.self_player;
        let my_start = info.players.iter().find(|p| p.is_self).and_then(|p| p.start_tile);
        let starts = info.static_map.start_locations.clone();
        let scouted = starts.iter().map(|s| Some(*s) == my_start).collect();
        let (w, h) = info.map_size_tiles;
        let (cells_w, cells_h) = (w.div_ceil(CELL), h.div_ceil(CELL));
        let home = my_start.map(react::depot_center);
        let react = kit.react.clone().map(|p| ReactState::new(p, hints, home, &starts));
        let tp = kit.tp.clone().map(|p| TpState::new(p, info));
        Self {
            tp,
            home,
            map_center: (w * 16, h * 16),
            react,
            kit,
            me,
            starts,
            walkable: info.static_map.walkable.clone(),
            placer: Placer::new(&info.static_map),
            pending: Vec::new(),
            last_supply_order: 0,
            enemy_buildings: BTreeMap::new(),
            scouted,
            cell_seen: vec![0; cells_w as usize * cells_h as usize],
            cells_w,
            cells_h,
            last_wave: 0,
            gas_crew: Vec::new(),
            scout: None,
            scout_done: false,
            scout_order: 0,
            drops: Vec::new(),
            last_defend: 0,
        }
    }

    fn update_memory(&mut self, f: &RawFrame) {
        for u in f.units.iter().filter(|u| u.owner != self.me && is_building(u.unit_type) && !is_neutral_owner(u.owner)) {
            self.enemy_buildings.insert(u.id, (u.pos, u.unit_type));
        }
        // Forget buildings whose tile is visible but which are no longer there.
        let seen: Vec<UnitId> = f.units.iter().map(|u| u.id).collect();
        self.enemy_buildings.retain(|id, (pos, _)| seen.contains(id) || !f.visibility.get(pos.0 / 32, pos.1 / 32));
        for (i, s) in self.starts.iter().enumerate() {
            if f.visibility.get(s.0 + 2, s.1 + 1) {
                self.scouted[i] = true;
            }
        }
        for cy in 0..self.cells_h {
            for cx in 0..self.cells_w {
                if f.visibility.get(cx * CELL + CELL / 2, cy * CELL + CELL / 2) {
                    self.cell_seen[(cy * self.cells_w + cx) as usize] = f.frame;
                }
            }
        }
    }

    /// Known enemy building → unscouted start → visible enemy unit → stalest walkable cell.
    fn attack_target(&mut self, f: &RawFrame, from: Pos) -> Option<Pos> {
        if let Some((p, _)) = self.enemy_buildings.values().min_by_key(|(p, _)| dist2(*p, from)) {
            return Some(*p);
        }
        if let Some(i) = self.scouted.iter().position(|s| !s) {
            let s = self.starts[i];
            return Some((s.0 * 32 + 64, s.1 * 32 + 48));
        }
        let enemy = f.units.iter().filter(|u| u.owner != self.me && !is_neutral_owner(u.owner)).min_by_key(|u| dist2(u.pos, from));
        if let Some(u) = enemy {
            return Some(u.pos);
        }
        let mut best: Option<(u32, Pos)> = None;
        for cy in 0..self.cells_h {
            for cx in 0..self.cells_w {
                let p = ((cx * CELL + CELL / 2) * 32, (cy * CELL + CELL / 2) * 32);
                if !self.walkable.get(p.0 / 8, p.1 / 8) {
                    continue;
                }
                let seen = self.cell_seen[(cy * self.cells_w + cx) as usize];
                if best.is_none_or(|(b, _)| seen < b) {
                    best = Some((seen, p));
                }
            }
        }
        best.map(|(_, p)| p)
    }

    /// Drop target: a known enemy main building first, then whatever waves would attack.
    fn drop_target(&mut self, f: &RawFrame, from: Pos) -> Option<Pos> {
        let depot = self.enemy_buildings.values().filter(|(_, ty)| is_resource_depot(*ty)).min_by_key(|(p, _)| dist2(*p, from));
        match depot {
            Some((p, _)) => Some(*p),
            None => self.attack_target(f, from),
        }
    }

    fn place(&self, ty: u16, f: &RawFrame, mine: &[&UnitState], base: Pos, reserved: &[(Tile, u16)]) -> Option<Tile> {
        if is_refinery(ty) {
            // Nearest free geyser next to the base.
            return f
                .units
                .iter()
                .filter(|u| u.unit_type == GEYSER && !reserved.iter().any(|(t, _)| *t == building_tile(u)))
                .filter(|u| dist2(u.pos, base) < 12 * 32 * 12 * 32)
                .min_by_key(|u| dist2(u.pos, base))
                .map(building_tile);
        }
        match self.kit.race {
            Race::Protoss if ty != PYLON => {
                // Pylon power field ≈ ellipse with radii 8 x 5 tiles around a completed pylon.
                let pylons: Vec<Pos> = mine.iter().filter(|u| u.unit_type == PYLON && u.has(flags::COMPLETED)).map(|u| u.pos).collect();
                let (w, h) = tile_size(ty);
                pylons.iter().find_map(|&py| {
                    self.placer.find_where(ty, py, &f.units, reserved, 8, |(tx, tyy)| {
                        let cx = (tx * 32 + w * 16) as f64 - py.0 as f64;
                        let cy = (tyy * 32 + h * 16) as f64 - py.1 as f64;
                        (cx / 224.0).powi(2) + (cy / 128.0).powi(2) <= 1.0
                    })
                })
            }
            Race::Zerg => {
                let (w, h) = tile_size(ty);
                self.placer.find_where(ty, base, &f.units, reserved, 10, |(tx, tyy)| {
                    (tyy..tyy + h).all(|y| (tx..tx + w).all(|x| f.creep.get(x, y)))
                })
            }
            _ => self.placer.find(ty, base, &f.units, reserved),
        }
    }
}

impl Bot for Brain {
    fn act(&mut self, f: &RawFrame) -> Act {
        let mut cmds = Vec::new();
        let k = self.kit.clone();
        self.update_memory(f);
        let mine: Vec<&UnitState> = f.units.iter().filter(|u| u.owner == self.me).collect();
        let count = |ty: u16| mine.iter().filter(|u| u.unit_type == ty).count();
        let minerals_vis: Vec<&UnitState> = f.units.iter().filter(|u| is_mineral(u.unit_type)).collect();
        self.pending.retain(|p| f.frame < p.issued + PENDING_TIMEOUT && !mine.iter().any(|u| u.unit_type == p.ty && building_tile(u) == p.tile));
        // 021-R8 (10th advisory, 041-R22): a bunker order without a bunker to show for it expires.
        if let Some(rp) = k.react.as_ref() {
            self.pending.retain(|p| !react::bunker_order_stale(rp, p, f, &mine));
        }

        let mut minerals = f.self_state.minerals - self.pending.iter().map(|p| cost(p.ty)).sum::<i32>();
        let mut gas = f.self_state.gas - self.pending.iter().map(|p| gas_cost(p.ty)).sum::<i32>();
        let mut supply_free = f.self_state.supply_total as i32 - f.self_state.supply_used as i32;
        // 021-R8: the teacher's base is the command center at its start location (it expands).
        let base = if k.react.is_some() { self.home_depot(&mine) } else { mine.iter().find(|u| is_resource_depot(u.unit_type)).map(|u| u.pos) };
        let mut larvae: Vec<&UnitState> = mine.iter().copied().filter(|u| u.unit_type == LARVA).collect();
        let prod_count = count(k.production) + self.pending.iter().filter(|p| p.ty == k.production).count();
        // 021-R8: rush rule R and the branch switches, at this observation (022: its own rules).
        if self.tp.is_some() {
            self.tp_observe(f, &mine);
        } else {
            self.react_observe(f, &mine);
        }
        let branch = self.react_branch();

        // Gas crew and scout (scenario bots only): workers with a job other than minerals.
        self.gas_crew.retain(|id| mine.iter().any(|u| u.id == *id));
        // The 021-R8 teacher re-chooses its scout from each observation (`react_scouting`).
        if k.react.is_none() && self.scout.is_some_and(|id| !mine.iter().any(|u| u.id == id)) {
            self.scout = None;
            self.scout_done = true;
        }
        let refinery = mine.iter().find(|u| is_refinery(u.unit_type) && done(u)).map(|u| u.id);
        if let Some(r) = refinery {
            while self.gas_crew.len() < k.gas_workers {
                let Some(w) = self.free_worker(&mine) else { break };
                cmds.push(Command::Unit { units: vec![w], order: CmdOrder::Gather(r), queued: false });
                self.gas_crew.push(w);
            }
        }
        // 022-R3: the gas crew, chosen again from each observation (022-R9).
        self.tp_gas(f, &mine, &mut cmds);
        if k.react.is_some() {
            self.react_scouting(f, &mine, &minerals_vis, &mut cmds);
            // 021-R8 SCV pull / repair before the idle-worker rule below.
            self.react_scvs(f, &mine, &minerals_vis, &mut cmds);
        } else {
            if k.scout && !self.scout_done && self.scout.is_none() && prod_count > 0 {
                self.scout = self.free_worker(&mine);
            }
            if let Some(id) = self.scout {
                self.scouting(f, &mine, id, &minerals_vis, &mut cmds);
            }
        }
        let busy: Vec<UnitId> = self.pending.iter().map(|p| p.worker).chain(self.gas_crew.iter().copied()).chain(self.scout).chain(self.react_busy()).collect();

        // Workers: idle ones mine (or return cargo); the gas crew goes back to gas.
        for w in mine.iter().filter(|u| u.unit_type == k.worker && u.has(flags::COMPLETED) && u.has(flags::IDLE)) {
            let on_gas = self.gas_crew.contains(&w.id);
            if busy.contains(&w.id) && !on_gas {
                continue;
            }
            if w.has(flags::CARRYING) {
                cmds.push(Command::Unit { units: vec![w.id], order: CmdOrder::ReturnCargo, queued: false });
            } else if let (true, Some(r)) = (on_gas, refinery) {
                cmds.push(Command::Unit { units: vec![w.id], order: CmdOrder::Gather(r), queued: false });
            } else if let Some(m) = nearest(w.pos, &minerals_vis) {
                cmds.push(Command::Unit { units: vec![w.id], order: CmdOrder::Gather(m.id), queued: false });
            }
        }

        // Supply (022: the factory counts as a production building).
        let supply_prod = prod_count + if self.tp.is_some() { count(FACTORY) } else { 0 };
        let supply_low = f.self_state.supply_total < 400 && supply_free <= 4 + 4 * supply_prod as i32;
        match k.supply {
            Supply::Building(ty) => {
                let in_progress = mine.iter().filter(|u| u.unit_type == ty && !u.has(flags::COMPLETED)).count() + self.pending.iter().filter(|p| p.ty == ty).count();
                // 021-R8 opening: 9 and 15 supply, then the rule above.
                let depots = count(ty) + self.pending.iter().filter(|p| p.ty == ty).count();
                let supply_low = self.react_supply_low(f, depots).unwrap_or(supply_low);
                if supply_low && in_progress == 0 && minerals >= cost(ty) {
                    if let Some(b) = base {
                        if self.order_build(ty, f, &mine, b, &busy, &mut cmds) {
                            minerals -= cost(ty);
                        }
                    }
                }
            }
            Supply::Larva(ty) => {
                if supply_low && f.frame >= self.last_supply_order + 24 * 16 && minerals >= cost(ty) {
                    if let Some(l) = larvae.pop() {
                        cmds.push(Command::Train { building: l.id, unit_type: ty });
                        self.last_supply_order = f.frame;
                        minerals -= cost(ty);
                    }
                }
            }
        }

        // 021-R8: D bunker, M natural command center and natural bunker. `hold_all` is held from
        // everything after the first barracks, `hold_army` (the expansion's price) from marines.
        let (mut hold_all, mut hold_army) = match base {
            Some(b) => self.react_build(f, &mine, b, &busy, &mut cmds, &mut minerals),
            None => (0, 0),
        };
        // 022-R3 / R5 / R6: refinery, factory, machine shop, siege research and the natural.
        if let Some(b) = base.filter(|_| self.tp.is_some()) {
            let (all, army) = self.tp_build(f, &mine, b, &busy, &mut cmds, &mut minerals, &mut gas);
            hold_all += all;
            hold_army += army;
        }

        // Production buildings.
        let workers = count(k.worker);
        let (cap, ready) = self.react_production(f, &mine, prod_count).unwrap_or((k.production_cap, workers >= k.production_at_workers[prod_count.min(2)]));
        let budget = if prod_count > 0 { minerals - hold_all } else { minerals };
        let mut prod_after = prod_count;
        if prod_count < cap && ready && budget >= cost(k.production) {
            if let Some(b) = base {
                if self.order_build(k.production, f, &mine, b, &busy, &mut cmds) {
                    minerals -= cost(k.production);
                    prod_after += 1;
                }
            }
        }
        // 021-R8 D: the second barracks (visible defence) comes before more SCVs.
        let hold_workers = if branch == Some(Branch::Defence) && prod_after < cap { cost(k.production) } else { 0 };

        // Tech path (050 scenario builds): the first unfinished step is started, or its price
        // is held back from unit production until it can be.
        let prod_ready = mine.iter().any(|u| u.unit_type == k.production && u.has(flags::COMPLETED));
        let mut tech_done = true;
        for &ty in &k.tech {
            if mine.iter().any(|u| u.unit_type == ty && done(u)) {
                continue;
            }
            tech_done = false;
            let started = mine.iter().any(|u| u.unit_type == ty) || self.pending.iter().any(|p| p.ty == ty);
            if prod_ready && !started {
                if minerals >= cost(ty) && gas >= gas_cost(ty) && base.is_some_and(|b| self.order_tech(ty, f, &mine, b, &busy, &mut cmds)) {
                    gas -= gas_cost(ty);
                }
                minerals -= cost(ty);
            }
            break;
        }

        // Units: workers first (until production exists), then army; a scenario's special
        // unit comes before army once the tech path is done. A lost gas building (being
        // rebuilt above) does not stop it while gas is banked.
        let tech_ready = tech_done || k.tech.iter().filter(|t| !is_refinery(**t)).all(|&t| mine.iter().any(|u| u.unit_type == t && done(u)));
        let special = k.special.filter(|_| tech_ready);
        let mut army_n = count(k.army);
        // 022-R3 / R5: marines before the factory, SCVs per command center.
        let army_cap = self.tp_marine_cap(&mine).unwrap_or(k.army_cap);
        let max_workers = self.tp_max_workers(&mine).unwrap_or(k.max_workers);
        minerals -= hold_all;
        match k.supply {
            Supply::Larva(_) => {
                while let Some(l) = larvae.pop() {
                    if let Some((LARVA, sp, cap)) = special {
                        if count(sp) < cap && minerals >= cost(sp) && gas >= gas_cost(sp) && supply_free >= supply_cost(sp) {
                            cmds.push(Command::Train { building: l.id, unit_type: sp });
                            minerals -= cost(sp);
                            gas -= gas_cost(sp);
                            supply_free -= supply_cost(sp);
                            continue;
                        }
                    }
                    let ty = if prod_ready && workers >= k.max_workers.min(9) && army_n < k.army_cap {
                        k.army
                    } else if workers < k.max_workers {
                        k.worker
                    } else {
                        break;
                    };
                    if minerals < cost(ty) || supply_free < 2 {
                        break;
                    }
                    cmds.push(Command::Train { building: l.id, unit_type: ty });
                    minerals -= cost(ty);
                    supply_free -= 2;
                    if ty == k.army {
                        army_n += 1;
                    }
                }
            }
            Supply::Building(_) => {
                let mut workers = workers;
                for d in mine.iter().filter(|u| is_resource_depot(u.unit_type) && u.has(flags::COMPLETED) && !u.has(flags::TRAINING) && !u.has(flags::LIFTED)) {
                    if workers < max_workers && minerals - hold_workers >= 50 && supply_free >= 2 {
                        cmds.push(Command::Train { building: d.id, unit_type: k.worker });
                        minerals -= 50;
                        supply_free -= 2;
                        // 022-R5: the SCV cap counts SCVs ordered at this observation (two command
                        // centers); the other builds keep their per-depot check.
                        if self.tp.is_some() {
                            workers += 1;
                        }
                    }
                }
                // 022-R3: siege tanks before marines.
                hold_army += self.tp_train(f, &mine, &mut minerals, &mut gas, &mut supply_free, &mut cmds);
                let mut used = Vec::new();
                if let Some((producer, sp, cap)) = special.filter(|s| s.0 != LARVA) {
                    let queued: usize = mine.iter().filter(|u| u.unit_type == producer).map(|u| u.build_queue.iter().filter(|&&q| q == sp).count()).sum();
                    let mut have = count(sp) + queued;
                    for p in mine.iter().filter(|u| u.unit_type == producer && done(u) && !u.has(flags::TRAINING) && !u.has(flags::LIFTED)) {
                        if have < cap && minerals >= cost(sp) && gas >= gas_cost(sp) && supply_free >= supply_cost(sp) {
                            cmds.push(Command::Train { building: p.id, unit_type: sp });
                            minerals -= cost(sp);
                            gas -= gas_cost(sp);
                            supply_free -= supply_cost(sp);
                            have += 1;
                            used.push(p.id);
                        }
                    }
                }
                for p in mine.iter().filter(|u| u.unit_type == k.production && u.has(flags::COMPLETED) && !u.has(flags::TRAINING) && !u.has(flags::LIFTED) && !used.contains(&u.id)) {
                    if army_n < army_cap && minerals - hold_army >= cost(k.army) && supply_free >= 2 {
                        cmds.push(Command::Train { building: p.id, unit_type: k.army });
                        minerals -= cost(k.army);
                        supply_free -= 2;
                        army_n += 1;
                    }
                }
            }
        }

        // 021-R8: marines that load or hold (D) or fill a bunker (M) stay out of waves and the guard.
        let kept = self.react_reserved_marines(&mine);
        match k.tactic {
            // 022-R7 / R8: siege tanks, the tank spot and the attack. Marines the home guard below
            // orders at this observation are its own (022-R8: the guard defends during an attack;
            // one order per unit per observation).
            _ if self.tp.is_some() => {
                let guard_on = k.defend && (branch != Some(Branch::Defence) || self.react_marines_attack());
                let mut owned = kept.clone();
                if let Some((_, guards)) = base.filter(|_| guard_on).and_then(|b| self.guard_order(f, &mine, b, &kept)) {
                    owned.extend(guards);
                }
                self.tp_army(f, &mine, &owned, &mut cmds)
            }
            // Attack-move waves (none in the teacher's defence branch).
            Tactic::Waves if branch == Some(Branch::Defence) => {}
            Tactic::Waves => {
                let army: Vec<&UnitState> = mine.iter().copied().filter(|u| u.unit_type == k.attacker && u.has(flags::COMPLETED) && u.loaded_in.is_none() && !kept.contains(&u.id)).collect();
                if army.len() >= k.first_wave && f.frame >= self.last_wave + WAVE_PERIOD {
                    let from = centroid(&army);
                    if let Some(target) = self.attack_target(f, from) {
                        let ids: Vec<UnitId> = army.iter().map(|u| u.id).collect();
                        for chunk in ids.chunks(12) {
                            cmds.push(Command::Unit { units: chunk.to_vec(), order: CmdOrder::Attack(Target::Pos(target)), queued: false });
                        }
                        self.last_wave = f.frame;
                    }
                }
            }
            Tactic::Drop => {
                if let Some(b) = base {
                    self.drop_step(f, &mine, b, &mut cmds);
                }
            }
        }
        // 021-R8 D: the home guard only while the marines attack (7th advisory C). The 022 teacher's
        // `tp_army` above read the same condition.
        if k.defend && (branch != Some(Branch::Defence) || self.react_marines_attack()) {
            if let Some(b) = base {
                self.defend(f, &mine, b, &kept, &mut cmds);
            }
        }
        self.react_marines(f, &mine, &mut cmds);

        Act { cmds, advance_frames: STEP_FRAMES, valid_until_frame: None, surrender: None }
    }

    fn react_stats(&self) -> Option<ReactStats> {
        // The 022 teacher reuses the 021-R8 state but keeps its own record.
        if self.tp.is_some() {
            return None;
        }
        self.teacher_stats()
    }

    fn tp_stats(&self) -> Option<TpStats> {
        self.tp.as_ref().map(|t| t.stats.clone())
    }
}

impl Brain {
    /// Picks a site and the nearest free worker; records the pending build.
    fn order_build(&mut self, ty: u16, f: &RawFrame, mine: &[&UnitState], base: Pos, busy: &[UnitId], cmds: &mut Vec<Command>) -> bool {
        // 021-R8: the teacher keeps its planned main bunker site and the natural free.
        let reserved: Vec<(Tile, u16)> = self.pending.iter().map(|p| (p.tile, p.ty)).chain(self.react_reserved_sites()).collect();
        let Some(tile) = self.place(ty, f, mine, base, &reserved) else { return false };
        let center = (tile.0 * 32 + 16, tile.1 * 32 + 16);
        // Scenario bots skip SCVs still constructing (BWAPI Unit_Busy); the standard M3
        // builds keep their original worker choice.
        // 022: the teacher's SCVs often construct several buildings at once (OpenBW smoke: a depot
        // given to a constructing SCV was refused and the teacher stayed supply blocked).
        let skip_constructing = !self.kit.tech.is_empty() || self.kit.tp.is_some();
        let worker = mine
            .iter()
            .filter(|u| u.unit_type == self.kit.worker && u.has(flags::COMPLETED) && !u.has(flags::CARRYING) && !busy.contains(&u.id))
            .filter(|u| !(skip_constructing && u.has(flags::CONSTRUCTING)))
            .min_by_key(|u| dist2(u.pos, center));
        let Some(w) = worker else { return false };
        cmds.push(Command::Build { worker: w.id, unit_type: ty, tile });
        self.pending.push(Pending { ty, tile, worker: w.id, issued: f.frame });
        true
    }

    /// One tech step: LAIR morphs a finished hatchery, CONTROL_TOWER is the starport's add-on
    /// (dropships need it), everything else is built by a worker.
    fn order_tech(&mut self, ty: u16, f: &RawFrame, mine: &[&UnitState], base: Pos, busy: &[UnitId], cmds: &mut Vec<Command>) -> bool {
        if ty == CONTROL_TOWER {
            let Some(s) = mine.iter().find(|u| u.unit_type == STARPORT && done(u) && !u.has_addon) else { return false };
            cmds.push(Command::BuildAddon { building: s.id, unit_type: ty });
            self.pending.push(Pending { ty, tile: addon_tile(building_tile(s)), worker: s.id, issued: f.frame });
            return true;
        }
        if ty != LAIR {
            return self.order_build(ty, f, mine, base, busy, cmds);
        }
        let Some(h) = mine.iter().find(|u| u.unit_type == HATCHERY && done(u)) else { return false };
        cmds.push(Command::Morph { unit: h.id, unit_type: LAIR });
        self.pending.push(Pending { ty, tile: building_tile(h), worker: h.id, issued: f.frame });
        true
    }

    /// The lowest-id completed worker without a job (not building, on gas or scouting).
    fn free_worker(&self, mine: &[&UnitState]) -> Option<UnitId> {
        let jobs = self.react_busy();
        let busy = |id: UnitId| self.pending.iter().any(|p| p.worker == id) || self.gas_crew.contains(&id) || self.scout == Some(id) || jobs.contains(&id);
        mine.iter()
            .filter(|u| u.unit_type == self.kit.worker && u.has(flags::COMPLETED) && !u.has(flags::CARRYING) && !u.has(flags::CONSTRUCTING) && !busy(u.id))
            .min_by_key(|u| u.id)
            .map(|u| u.id)
    }

    /// The scout walks to unscouted starts until an enemy building is known, then mines again.
    fn scouting(&mut self, f: &RawFrame, mine: &[&UnitState], id: UnitId, minerals: &[&UnitState], cmds: &mut Vec<Command>) {
        let Some(s) = mine.iter().find(|u| u.id == id) else { return };
        let next = self.starts.iter().zip(&self.scouted).filter(|(_, done)| !**done).map(|(t, _)| (t.0 * 32 + 64, t.1 * 32 + 48)).min_by_key(|p| dist2(*p, s.pos));
        match next {
            Some(p) if self.enemy_buildings.is_empty() => {
                if f.frame >= self.scout_order + WAVE_PERIOD || s.has(flags::IDLE) {
                    cmds.push(Command::Unit { units: vec![id], order: CmdOrder::Move(Target::Pos(p)), queued: false });
                    self.scout_order = f.frame;
                }
            }
            _ => {
                if let Some(m) = nearest(s.pos, minerals) {
                    cmds.push(Command::Unit { units: vec![id], order: CmdOrder::Gather(m.id), queued: false });
                }
                self.scout = None;
                self.scout_done = true;
            }
        }
    }

    /// 050 terran drop: load DROP_SIZE marines into an idle dropship at home, unload them on the
    /// enemy main, send the empty ship home and let the squad attack from there.
    fn drop_step(&mut self, f: &RawFrame, mine: &[&UnitState], base: Pos, cmds: &mut Vec<Command>) {
        let unit = |id: UnitId| mine.iter().copied().find(|u| u.id == id);
        for i in 0..self.drops.len() {
            self.drops[i].squad.retain(|id| unit(*id).is_some());
            let d = &self.drops[i];
            let (ship, squad, phase, last) = (unit(d.ship), d.squad.clone(), d.phase, d.last_order);
            let loaded = squad.iter().filter(|id| unit(**id).is_some_and(|u| u.loaded_in == Some(d.ship))).count();
            match (phase, ship) {
                (DropPhase::Load { .. } | DropPhase::Fly, None) => self.drops[i].phase = DropPhase::Fight,
                (DropPhase::Load { since }, Some(s)) => {
                    let full = loaded == squad.len() && !squad.is_empty();
                    if full || (f.frame >= since + LOAD_TIMEOUT && loaded >= DROP_SIZE / 2) {
                        if let Some(t) = self.drop_target(f, s.pos) {
                            cmds.push(Command::Unit { units: vec![s.id], order: CmdOrder::UnloadAll(Some(t)), queued: false });
                            self.drops[i].phase = DropPhase::Fly;
                            self.drops[i].last_order = f.frame;
                        }
                    } else if f.frame >= last + WAVE_PERIOD {
                        let waiting: Vec<UnitId> = squad.iter().copied().filter(|id| unit(*id).is_some_and(|u| u.loaded_in.is_none())).collect();
                        if !waiting.is_empty() {
                            cmds.push(Command::Unit { units: waiting, order: CmdOrder::Load(s.id), queued: false });
                        }
                        self.drops[i].last_order = f.frame;
                    }
                }
                (DropPhase::Fly, Some(s)) => {
                    if loaded == 0 {
                        cmds.push(Command::Unit { units: vec![s.id], order: CmdOrder::Move(Target::Pos(base)), queued: false });
                        self.drops[i].phase = DropPhase::Fight;
                    } else if f.frame >= last + 24 * 10 {
                        // The target may have changed (scouting); re-issue the unload.
                        if let Some(t) = self.drop_target(f, s.pos) {
                            cmds.push(Command::Unit { units: vec![s.id], order: CmdOrder::UnloadAll(Some(t)), queued: false });
                        }
                        self.drops[i].last_order = f.frame;
                    }
                }
                (DropPhase::Fight, _) => {
                    let out: Vec<&UnitState> = squad.iter().filter_map(|id| unit(*id)).filter(|u| u.loaded_in.is_none()).collect();
                    if !out.is_empty() && f.frame >= last + WAVE_PERIOD {
                        if let Some(t) = self.attack_target(f, centroid(&out)) {
                            let ids: Vec<UnitId> = out.iter().map(|u| u.id).collect();
                            cmds.push(Command::Unit { units: ids, order: CmdOrder::Attack(Target::Pos(t)), queued: false });
                        }
                        self.drops[i].last_order = f.frame;
                    }
                }
            }
        }
        self.drops.retain(|d| !d.squad.is_empty());

        // A new drop once no other one is loading or flying.
        if self.drops.iter().any(|d| d.phase != DropPhase::Fight) {
            return;
        }
        let ships: Vec<UnitId> = self.drops.iter().map(|d| d.ship).collect();
        let Some(ship) = mine.iter().find(|u| u.unit_type == DROPSHIP && done(u) && !ships.contains(&u.id) && dist2(u.pos, base) < DEFEND_RADIUS * DEFEND_RADIUS) else { return };
        let taken: Vec<UnitId> = self.drops.iter().flat_map(|d| d.squad.iter().copied()).collect();
        let mut marines: Vec<&UnitState> = mine.iter().copied().filter(|u| u.unit_type == MARINE && u.has(flags::COMPLETED) && u.loaded_in.is_none() && !taken.contains(&u.id)).collect();
        if marines.len() < DROP_SIZE {
            return;
        }
        marines.sort_by_key(|u| (dist2(u.pos, ship.pos), u.id));
        let squad: Vec<UnitId> = marines[..DROP_SIZE].iter().map(|u| u.id).collect();
        cmds.push(Command::Unit { units: squad.clone(), order: CmdOrder::Load(ship.id), queued: false });
        self.drops.push(Drop { ship: ship.id, squad, phase: DropPhase::Load { since: f.frame }, last_order: f.frame });
    }

    /// Home guard: `army` units that are not attackers or on a drop attack-move on the
    /// nearest enemy close to the base. 021-R8 teacher (waves of every marine): the marines
    /// within `DEFEND_RADIUS` of an own command center guard every command center, except those
    /// `kept` for a bunker.
    fn defend(&mut self, f: &RawFrame, mine: &[&UnitState], base: Pos, kept: &[UnitId], cmds: &mut Vec<Command>) {
        let Some((at, guards)) = self.guard_order(f, mine, base, kept) else { return };
        for chunk in guards.chunks(12) {
            cmds.push(Command::Unit { units: chunk.to_vec(), order: CmdOrder::Attack(Target::Pos(at)), queued: false });
        }
        self.last_defend = f.frame;
    }

    /// The home guard's order at this observation (`defend`): where it attack-moves and which
    /// units; `None` when it gives none. Read without acting by the 022 teacher, whose attack
    /// leaves these units to the guard (one order per unit per observation, 022-R8).
    fn guard_order(&self, f: &RawFrame, mine: &[&UnitState], base: Pos, kept: &[UnitId]) -> Option<(Pos, Vec<UnitId>)> {
        // With plain waves every army unit attacks; there is no separate guard.
        let react = self.kit.react.is_some();
        let has_guard = self.kit.tactic == Tactic::Drop || self.kit.attacker != self.kit.army || react;
        if !has_guard || f.frame < self.last_defend + WAVE_PERIOD {
            return None;
        }
        let bases: Vec<Pos> = if react { mine.iter().filter(|u| is_resource_depot(u.unit_type)).map(|u| u.pos).collect() } else { vec![base] };
        let near_base = |p: Pos| bases.iter().map(|b| dist2(p, *b)).min().unwrap_or(i64::MAX);
        // 021-R8 D marines attack enemies within `marine_radius` only (7th advisory C).
        let reach = self.react_guard_radius().map_or(DEFEND_RADIUS, |t| t * 32);
        let threat = f
            .units
            .iter()
            .filter(|u| u.owner != self.me && !is_neutral_owner(u.owner) && near_base(u.pos) < reach * reach)
            .min_by_key(|u| (near_base(u.pos), u.id));
        let t = threat?;
        let on_drop: Vec<UnitId> = self.drops.iter().flat_map(|d| d.squad.iter().copied()).collect();
        let guards: Vec<UnitId> = mine
            .iter()
            .filter(|u| u.unit_type == self.kit.army && u.has(flags::COMPLETED) && u.loaded_in.is_none() && !on_drop.contains(&u.id) && !kept.contains(&u.id))
            .filter(|u| !react || near_base(u.pos) < DEFEND_RADIUS * DEFEND_RADIUS)
            .map(|u| u.id)
            .collect();
        Some((t.pos, guards))
    }
}

/// Finished: complete and not morphing or under construction.
fn done(u: &UnitState) -> bool {
    u.has(flags::COMPLETED) && !u.has(flags::BEING_CONSTRUCTED) && u.remaining_build == 0
}

fn dist2(a: Pos, b: Pos) -> i64 {
    let dx = a.0 as i64 - b.0 as i64;
    let dy = a.1 as i64 - b.1 as i64;
    dx * dx + dy * dy
}

fn centroid(units: &[&UnitState]) -> Pos {
    let n = units.len().max(1) as u64;
    let (sx, sy) = units.iter().fold((0u64, 0u64), |(x, y), u| (x + u.pos.0 as u64, y + u.pos.1 as u64));
    ((sx / n) as u16, (sy / n) as u16)
}

fn nearest<'a>(p: Pos, units: &[&'a UnitState]) -> Option<&'a UnitState> {
    units.iter().copied().min_by_key(|u| dist2(p, u.pos))
}

/// Neutral / ownerless units (BWAPI neutral player 11; dying units report -1 = 255).
fn is_neutral_owner(owner: u8) -> bool {
    owner >= 8
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(race: Race) -> GameInfo {
        let mut walk = BitGrid::new(256, 256);
        for y in 0..256 {
            for x in 0..256 {
                walk.set(x, y, true);
            }
        }
        GameInfo {
            map_name: "t".into(),
            map_hash: [0; 32],
            map_size_tiles: (64, 64),
            self_player: 0,
            players: vec![
                PlayerInfo { id: 0, name: "a".into(), race, start_tile: Some((4, 4)), is_self: true, is_enemy: false },
                PlayerInfo { id: 1, name: "b".into(), race: Race::Zerg, start_tile: None, is_self: false, is_enemy: true },
            ],
            static_map: StaticMap {
                width_tiles: 64,
                height_tiles: 64,
                walkable: walk,
                buildable: BitGrid::from_bytes(64, 64, &[1; 64 * 64]),
                ground_height: vec![0; 64 * 64],
                start_locations: vec![(4, 4), (56, 56)],
            },
            latency_frames: 0,
        }
    }

    fn unit(id: u32, ty: u16, pos: Pos) -> UnitState {
        UnitState { id, owner: 0, unit_type: ty, pos, flags: flags::COMPLETED | flags::IDLE, ..Default::default() }
    }

    fn frame(units: Vec<UnitState>, minerals: i32, creep_all: bool) -> RawFrame {
        let mut creep = BitGrid::new(64, 64);
        if creep_all {
            creep = BitGrid::from_bytes(64, 64, &[1; 64 * 64]);
        }
        RawFrame {
            frame: 100,
            self_state: PlayerState { minerals, supply_used: 8, supply_total: 18, upgrades: vec![0; 63], ..Default::default() },
            units,
            visibility: BitGrid::from_bytes(64, 64, &[1; 64 * 64]),
            explored: BitGrid::new(64, 64),
            creep,
            events: vec![],
            effects: vec![],
        }
    }

    #[test]
    fn race_dispatch_and_worker_training() {
        let mut z = for_game(&info(Race::Zerg));
        let act = z.act(&frame(vec![unit(1, HATCHERY, (200, 200)), unit(2, LARVA, (220, 220))], 50, true));
        assert!(act.cmds.iter().any(|c| matches!(c, Command::Train { building: 2, unit_type: DRONE })), "{:?}", act.cmds);
        let mut p = for_game(&info(Race::Protoss));
        let act = p.act(&frame(vec![unit(1, NEXUS, (200, 200))], 50, false));
        assert!(act.cmds.iter().any(|c| matches!(c, Command::Train { building: 1, unit_type: PROBE })), "{:?}", act.cmds);
    }

    #[test]
    fn protoss_gateway_goes_inside_pylon_power() {
        let mut units = vec![unit(1, NEXUS, (400, 400)), unit(2, PYLON, (700, 400))];
        units.extend((10..23).map(|i| unit(i, PROBE, (420, 460))));
        let mut p = Brain::new(&info(Race::Protoss), Kit::protoss());
        let act = p.act(&frame(units, 400, false));
        let gw = act.cmds.iter().find_map(|c| match c {
            Command::Build { unit_type: GATEWAY, tile, .. } => Some(*tile),
            _ => None,
        });
        let t = gw.expect("gateway ordered");
        let (cx, cy) = (t.0 as f64 * 32.0 + 64.0 - 700.0, t.1 as f64 * 32.0 + 48.0 - 400.0);
        assert!((cx / 224.0).powi(2) + (cy / 128.0).powi(2) <= 1.0, "{t:?}");
    }

    #[test]
    fn hunting_targets_stalest_cell_when_nothing_known() {
        let mut b = Brain::new(&info(Race::Terran), Kit::terran());
        b.scouted.iter_mut().for_each(|s| *s = true);
        let mut f = frame(vec![], 0, false);
        f.visibility = BitGrid::new(64, 64);
        // everything seen at frame 100 except the cell around tile (60,60)
        b.cell_seen.iter_mut().for_each(|c| *c = 100);
        let last = b.cell_seen.len() - 1;
        b.cell_seen[last] = 0;
        let t = b.attack_target(&f, (0, 0)).unwrap();
        assert_eq!((t.0 / 32 / CELL, t.1 / 32 / CELL), (7, 7));
    }

    fn enemy(id: u32, ty: u16, pos: Pos) -> UnitState {
        UnitState { owner: 1, ..unit(id, ty, pos) }
    }

    fn trains(act: &Act, ty: u16) -> usize {
        act.cmds.iter().filter(|c| matches!(c, Command::Train { unit_type, .. } if *unit_type == ty)).count()
    }

    #[test]
    fn e_regression_050_scenario_needs_its_race() {
        assert_eq!(Kit::for_build(Race::Zerg, T0Build::ZergMuta).attacker, MUTALISK);
        assert_eq!(Kit::for_build(Race::Terran, T0Build::TerranDrop).tactic, Tactic::Drop);
        // A Random pick that resolved to another race plays that race's standard build.
        let k = Kit::for_build(Race::Protoss, T0Build::ZergRush);
        assert_eq!((k.race, k.army, k.attacker, k.tech.len()), (Race::Protoss, ZEALOT, ZEALOT, 0));
        let k = Kit::for_build(Race::Zerg, T0Build::Standard);
        assert_eq!((k.first_wave, k.army_cap, k.special), (12, usize::MAX, None));
    }

    #[test]
    fn e_regression_050_zerg_rush_pools_at_five_drones() {
        let mut units = vec![unit(1, HATCHERY, (400, 400))];
        units.extend((10..15).map(|i| unit(i, DRONE, (420, 460))));
        let mut rush = Brain::new(&info(Race::Zerg), Kit::zerg_rush());
        let act = rush.act(&frame(units.clone(), 200, true));
        assert!(act.cmds.iter().any(|c| matches!(c, Command::Build { unit_type: SPAWNING_POOL, .. })), "{:?}", act.cmds);
        // The standard zerg build waits for 9 drones.
        let mut std = Brain::new(&info(Race::Zerg), Kit::zerg());
        let act = std.act(&frame(units, 200, true));
        assert!(!act.cmds.iter().any(|c| matches!(c, Command::Build { unit_type: SPAWNING_POOL, .. })), "{:?}", act.cmds);
    }

    #[test]
    fn e_regression_050_drop_builds_refinery_on_geyser_then_gas_crew() {
        let geyser = UnitState { owner: 11, ..unit(5, GEYSER, (560, 400)) };
        let mut units = vec![unit(1, CC, (400, 400)), unit(2, BARRACKS, (400, 560)), geyser.clone()];
        units.extend((10..22).map(|i| unit(i, SCV, (420, 460))));
        let mut b = Brain::new(&info(Race::Terran), Kit::terran_drop());
        let act = b.act(&frame(units.clone(), 100, false));
        let tile = act.cmds.iter().find_map(|c| match c {
            Command::Build { unit_type: REFINERY, tile, .. } => Some(*tile),
            _ => None,
        });
        assert_eq!(tile, Some(building_tile(&geyser)), "{:?}", act.cmds);

        // Refinery done: three workers go to gas and stay there when idle.
        units[2] = unit(5, REFINERY, (560, 400));
        let act = b.act(&frame(units.clone(), 0, false));
        let gas: std::collections::BTreeSet<UnitId> = act
            .cmds
            .iter()
            .filter_map(|c| match c {
                Command::Unit { units, order: CmdOrder::Gather(5), .. } => Some(units[0]),
                _ => None,
            })
            .collect();
        assert_eq!(gas.into_iter().collect::<Vec<_>>(), vec![10, 11, 12], "{:?}", act.cmds);
        let act = b.act(&frame(units, 0, false));
        let regather = act.cmds.iter().filter(|c| matches!(c, Command::Unit { order: CmdOrder::Gather(5), .. })).count();
        assert_eq!(regather, 3, "idle gas workers return to gas: {:?}", act.cmds);
    }

    #[test]
    fn e_regression_050_muta_morphs_lair_then_spire_then_mutas() {
        let mut units = vec![unit(1, HATCHERY, (400, 400)), unit(2, SPAWNING_POOL, (300, 300)), unit(3, EXTRACTOR, (560, 400)), unit(4, LARVA, (420, 420))];
        units.extend((10..26).map(|i| unit(i, DRONE, (420, 460))));
        let mut b = Brain::new(&info(Race::Zerg), Kit::zerg_muta());
        let mut f = frame(units.clone(), 150, true);
        f.self_state.gas = 100;
        let act = b.act(&f);
        assert!(act.cmds.contains(&Command::Morph { unit: 1, unit_type: LAIR }), "{:?}", act.cmds);
        // While the lair is pending its price is held back: no zerglings or drones.
        assert_eq!(trains(&act, ZERGLING) + trains(&act, DRONE), 0, "{:?}", act.cmds);

        units[0] = unit(1, LAIR, (400, 400));
        let mut f = frame(units.clone(), 200, true);
        f.self_state.gas = 150;
        f.frame = 200;
        let act = b.act(&f);
        assert!(act.cmds.iter().any(|c| matches!(c, Command::Build { unit_type: SPIRE, .. })), "{:?}", act.cmds);

        // A fresh brain (no pending spire) with the spire done: larvae become mutalisks.
        units.push(unit(6, SPIRE, (300, 400)));
        let mut b = Brain::new(&info(Race::Zerg), Kit::zerg_muta());
        let mut f = frame(units, 100, true);
        f.self_state.gas = 100;
        f.frame = 300;
        let act = b.act(&f);
        assert_eq!(trains(&act, MUTALISK), 1, "{:?}", act.cmds);

        // The extractor was destroyed: it is rebuilt, and banked gas still makes mutalisks.
        let mut units = vec![unit(1, LAIR, (400, 400)), unit(2, SPAWNING_POOL, (300, 300)), unit(4, LARVA, (420, 420)), unit(6, SPIRE, (300, 400))];
        units.extend((10..26).map(|i| unit(i, DRONE, (420, 460))));
        units.push(UnitState { owner: 11, ..unit(5, GEYSER, (560, 400)) });
        let mut b = Brain::new(&info(Race::Zerg), Kit::zerg_muta());
        let mut f = frame(units, 150, true);
        f.self_state.gas = 100;
        let act = b.act(&f);
        assert!(act.cmds.iter().any(|c| matches!(c, Command::Build { unit_type: EXTRACTOR, .. })), "{:?}", act.cmds);
        assert_eq!(trains(&act, MUTALISK), 1, "{:?}", act.cmds);
    }

    #[test]
    fn e_regression_050_dark_templar_before_zealots() {
        let mut units = vec![
            unit(1, NEXUS, (400, 400)),
            unit(2, PYLON, (600, 400)),
            unit(3, GATEWAY, (600, 500)),
            unit(4, ASSIMILATOR, (560, 300)),
            unit(5, CYBERNETICS_CORE, (700, 300)),
            unit(6, CITADEL, (700, 500)),
            unit(7, TEMPLAR_ARCHIVES, (800, 400)),
        ];
        units.extend((10..30).map(|i| unit(i, PROBE, (420, 460))));
        let mut b = Brain::new(&info(Race::Protoss), Kit::protoss_dt());
        let mut f = frame(units.clone(), 300, false);
        f.self_state.gas = 100;
        let act = b.act(&f);
        assert_eq!((trains(&act, DARK_TEMPLAR), trains(&act, ZEALOT)), (1, 0), "{:?}", act.cmds);

        // Unfinished archives: no dark templar yet, zealots up to the guard cap.
        units[6].flags = flags::BEING_CONSTRUCTED;
        units.extend((40..43).map(|i| unit(i, ZEALOT, (500, 500))));
        let mut f = frame(units, 300, false);
        f.self_state.gas = 100;
        let act = b.act(&f);
        assert_eq!((trains(&act, DARK_TEMPLAR), trains(&act, ZEALOT)), (0, 0), "{:?}", act.cmds);
    }

    #[test]
    fn e_regression_050_drop_loads_marines_and_unloads_on_enemy_main() {
        let mut units = vec![unit(1, CC, (400, 400)), unit(2, DROPSHIP, (450, 450)), enemy(99, CC, (56 * 32 + 64, 56 * 32 + 48))];
        // The drop starts on its own, whatever the tech path says.
        units.extend((10..20).map(|i| unit(i, MARINE, (460, 460))));
        let mut b = Brain::new(&info(Race::Terran), Kit::terran_drop());
        let act = b.act(&frame(units.clone(), 0, false));
        let squad = act.cmds.iter().find_map(|c| match c {
            Command::Unit { units, order: CmdOrder::Load(2), .. } => Some(units.clone()),
            _ => None,
        });
        let squad = squad.expect("marines ordered into the dropship");
        assert_eq!(squad.len(), DROP_SIZE);

        // Everyone aboard: the ship unloads at the known enemy command center.
        for u in units.iter_mut().filter(|u| squad.contains(&u.id)) {
            u.loaded_in = Some(2);
        }
        let mut f = frame(units.clone(), 0, false);
        f.frame = 116;
        let act = b.act(&f);
        assert!(act.cmds.contains(&Command::Unit { units: vec![2], order: CmdOrder::UnloadAll(Some((56 * 32 + 64, 56 * 32 + 48))), queued: false }), "{:?}", act.cmds);

        // Unloaded: the ship flies home and the squad attacks.
        for u in units.iter_mut() {
            u.loaded_in = None;
        }
        let mut f = frame(units, 0, false);
        f.frame = 200;
        let act = b.act(&f);
        assert!(act.cmds.contains(&Command::Unit { units: vec![2], order: CmdOrder::Move(Target::Pos((400, 400))), queued: false }), "{:?}", act.cmds);
        let mut f2 = f.clone();
        f2.frame = 260;
        let act = b.act(&f2);
        assert!(act.cmds.iter().any(|c| matches!(c, Command::Unit { units, order: CmdOrder::Attack(_), .. } if units.len() == DROP_SIZE)), "{:?}", act.cmds);
    }

    #[test]
    fn e_regression_050_guard_defends_base_but_standard_kit_has_none() {
        let mut units = vec![unit(1, NEXUS, (400, 400)), enemy(99, ZERGLING, (500, 500))];
        units.extend((40..43).map(|i| unit(i, ZEALOT, (420, 420))));
        let mut dt = Brain::new(&info(Race::Protoss), Kit::protoss_dt());
        let act = dt.act(&frame(units.clone(), 0, false));
        assert!(act.cmds.contains(&Command::Unit { units: vec![40, 41, 42], order: CmdOrder::Attack(Target::Pos((500, 500))), queued: false }), "{:?}", act.cmds);
        let mut std = Brain::new(&info(Race::Protoss), Kit::protoss());
        let act = std.act(&frame(units, 0, false));
        assert!(!act.cmds.iter().any(|c| matches!(c, Command::Unit { order: CmdOrder::Attack(_), .. })), "{:?}", act.cmds);
    }

    #[test]
    fn e_regression_050_drop_adds_control_tower_before_dropships() {
        let mut units = vec![unit(1, CC, (400, 400)), unit(2, BARRACKS, (400, 560)), unit(3, REFINERY, (560, 400)), unit(4, FACTORY, (700, 560)), unit(5, STARPORT, (800, 400))];
        units.extend((10..30).map(|i| unit(i, SCV, (420, 460))));
        let mut b = Brain::new(&info(Race::Terran), Kit::terran_drop());
        let mut f = frame(units.clone(), 300, false);
        f.self_state.gas = 200;
        let act = b.act(&f);
        assert!(act.cmds.contains(&Command::BuildAddon { building: 5, unit_type: CONTROL_TOWER }), "{:?}", act.cmds);
        assert_eq!(trains(&act, DROPSHIP), 0, "{:?}", act.cmds);

        units[4].has_addon = true;
        units.push(unit(6, CONTROL_TOWER, (800 + 64 + 32, 400 + 16)));
        let mut b = Brain::new(&info(Race::Terran), Kit::terran_drop());
        let mut f = frame(units, 300, false);
        f.self_state.gas = 200;
        let act = b.act(&f);
        assert_eq!(trains(&act, DROPSHIP), 1, "{:?}", act.cmds);
    }

    #[test]
    fn e_regression_050_constructing_scv_is_not_picked_to_build() {
        let geyser = UnitState { owner: 11, ..unit(5, GEYSER, (560, 400)) };
        let busy = UnitState { flags: flags::COMPLETED | flags::CONSTRUCTING, ..unit(10, SCV, (560, 420)) };
        let mut units = vec![unit(1, CC, (400, 400)), unit(2, BARRACKS, (400, 560)), geyser, busy];
        units.extend((11..22).map(|i| unit(i, SCV, (300, 300))));
        let mut b = Brain::new(&info(Race::Terran), Kit::terran_drop());
        let act = b.act(&frame(units, 100, false));
        let builder = act.cmds.iter().find_map(|c| match c {
            Command::Build { unit_type: REFINERY, worker, .. } => Some(*worker),
            _ => None,
        });
        assert!(builder.is_some_and(|w| w != 10), "{:?}", act.cmds);
    }

    #[test]
    fn e_regression_050_drop_scout_walks_to_unscouted_start() {
        let mut units = vec![unit(1, CC, (400, 400)), unit(2, BARRACKS, (400, 560))];
        units.extend((10..20).map(|i| unit(i, SCV, (420, 460))));
        let mut b = Brain::new(&info(Race::Terran), Kit::terran_drop());
        let mut f = frame(units, 0, false);
        f.visibility = BitGrid::new(64, 64);
        let act = b.act(&f);
        assert!(act.cmds.contains(&Command::Unit { units: vec![10], order: CmdOrder::Move(Target::Pos((56 * 32 + 64, 56 * 32 + 48))), queued: false }), "{:?}", act.cmds);
    }
}
