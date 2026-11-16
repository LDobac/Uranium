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
const CELL: u16 = 8;
const DROP_SIZE: usize = 8;
const LOAD_TIMEOUT: u32 = 24 * 20;
const DEFEND_RADIUS: i64 = 20 * 32;

pub trait Bot: Send {
    fn act(&mut self, f: &RawFrame) -> Act;
    fn react_stats(&self) -> Option<ReactStats> {
        None
    }
    fn tp_stats(&self) -> Option<TpStats> {
        None
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum T0Build {
    #[default]
    Standard,
    ZergRush,
    TerranDrop,
    ProtossDt,
    ZergMuta,
    TerranReact,
    TerranTp,
}

impl T0Build {
    pub const SCENARIOS: [T0Build; 4] = [T0Build::ZergRush, T0Build::TerranDrop, T0Build::ProtossDt, T0Build::ZergMuta];

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

    pub fn needs_map_hints(self) -> bool {
        matches!(self, T0Build::TerranReact | T0Build::TerranTp)
    }

    pub fn for_lobby(self, info: &GameInfo) -> T0Build {
        let enemy = info.players.iter().find(|p| p.is_enemy).map(|p| p.race);
        match self {
            T0Build::TerranTp if enemy == Some(Race::Zerg) => T0Build::TerranReact,
            b => b,
        }
    }
}

pub fn for_game(info: &GameInfo) -> Box<dyn Bot> {
    for_game_build(info, T0Build::Standard)
}

pub fn for_game_build(info: &GameInfo, build: T0Build) -> Box<dyn Bot> {
    for_game_hints(info, build, MapHints::default())
}

pub fn for_game_hints(info: &GameInfo, build: T0Build, hints: MapHints) -> Box<dyn Bot> {
    let race = info.players.iter().find(|p| p.is_self).map(|p| p.race).unwrap_or(Race::Terran);
    Box::new(Brain::with_hints(info, Kit::for_build(race, build.for_lobby(info)), &hints))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tactic {
    Waves,
    Drop,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Supply {
    Building(u16),
    Larva(u16),
}

#[derive(Clone, Debug)]
pub struct Kit {
    pub race: Race,
    pub worker: u16,
    pub max_workers: usize,
    pub supply: Supply,
    pub production: u16,
    pub production_cap: usize,
    pub production_at_workers: [usize; 3],
    pub army: u16,
    pub first_wave: usize,
    pub attacker: u16,
    pub army_cap: usize,
    pub tech: Vec<u16>,
    pub gas_workers: usize,
    pub special: Option<(u16, u16, usize)>,
    pub tactic: Tactic,
    pub scout: bool,
    pub defend: bool,
    pub react: Option<ReactParams>,
    pub tp: Option<TpParams>,
}

impl Kit {
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

    pub fn zerg_rush() -> Self {
        Self { max_workers: 6, production_at_workers: [5, usize::MAX, usize::MAX], first_wave: 6, ..Self::zerg() }
    }

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

    pub fn terran_react() -> Self {
        Self { defend: true, react: Some(ReactParams::default()), ..Self::terran() }
    }

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
    last_supply_order: u32,
    enemy_buildings: BTreeMap<UnitId, (Pos, u16)>,
    scouted: Vec<bool>,
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
    home: Option<Pos>,
    map_center: Pos,
    react: Option<ReactState>,
    tp: Option<TpState>,
}

impl Brain {
    pub fn new(info: &GameInfo, kit: Kit) -> Self {
        Self::with_hints(info, kit, &MapHints::default())
    }

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

    fn drop_target(&mut self, f: &RawFrame, from: Pos) -> Option<Pos> {
        let depot = self.enemy_buildings.values().filter(|(_, ty)| is_resource_depot(*ty)).min_by_key(|(p, _)| dist2(*p, from));
        match depot {
            Some((p, _)) => Some(*p),
            None => self.attack_target(f, from),
        }
    }

    fn place(&self, ty: u16, f: &RawFrame, mine: &[&UnitState], base: Pos, reserved: &[(Tile, u16)]) -> Option<Tile> {
        if is_refinery(ty) {
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
        if let Some(rp) = k.react.as_ref() {
            self.pending.retain(|p| !react::bunker_order_stale(rp, p, f, &mine));
        }
        self.tp_idle(f, &mine);

        let mut minerals = f.self_state.minerals - self.pending.iter().map(|p| cost(p.ty)).sum::<i32>();
        let mut gas = f.self_state.gas - self.pending.iter().map(|p| gas_cost(p.ty)).sum::<i32>();
        let mut supply_free = f.self_state.supply_total as i32 - f.self_state.supply_used as i32;
        let base = if k.react.is_some() { self.home_depot(&mine) } else { mine.iter().find(|u| is_resource_depot(u.unit_type)).map(|u| u.pos) };
        let mut larvae: Vec<&UnitState> = mine.iter().copied().filter(|u| u.unit_type == LARVA).collect();
        let prod_count = count(k.production) + self.pending.iter().filter(|p| p.ty == k.production).count();
        if self.tp.is_some() {
            self.tp_observe(f, &mine);
        } else {
            self.react_observe(f, &mine);
        }
        let branch = self.react_branch();

        self.gas_crew.retain(|id| mine.iter().any(|u| u.id == *id));
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
        self.tp_gas(f, &mine, &mut cmds);
        if k.react.is_some() {
            self.react_scouting(f, &mine, &minerals_vis, &mut cmds);
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

        for w in mine.iter().filter(|u| u.unit_type == k.worker && u.has(flags::COMPLETED) && u.has(flags::IDLE)) {
            let on_gas = self.gas_crew.contains(&w.id);
            if busy.contains(&w.id) && !on_gas {
                continue;
            }
            if !self.tp_idle_long(w.id, f.frame) {
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

        let supply_prod = prod_count + if self.tp.is_some() { count(FACTORY) } else { 0 };
        let supply_low = f.self_state.supply_total < 400 && supply_free <= 4 + 4 * supply_prod as i32;
        match k.supply {
            Supply::Building(ty) => {
                let in_progress = mine.iter().filter(|u| u.unit_type == ty && !u.has(flags::COMPLETED)).count() + self.pending.iter().filter(|p| p.ty == ty).count();
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

        let (mut hold_all, mut hold_army) = match base {
            Some(b) => self.react_build(f, &mine, b, &busy, &mut cmds, &mut minerals),
            None => (0, 0),
        };
        if let Some(b) = base.filter(|_| self.tp.is_some()) {
            let (all, army) = self.tp_build(f, &mine, b, &busy, &mut cmds, &mut minerals, &mut gas);
            hold_all += all;
            hold_army += army;
        }

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
        let hold_workers = if branch == Some(Branch::Defence) && prod_after < cap { cost(k.production) } else { 0 };

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

        let tech_ready = tech_done || k.tech.iter().filter(|t| !is_refinery(**t)).all(|&t| mine.iter().any(|u| u.unit_type == t && done(u)));
        let special = k.special.filter(|_| tech_ready);
        let mut army_n = count(k.army);
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
                        if self.tp.is_some() {
                            workers += 1;
                        }
                    }
                }
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

        let kept = self.react_reserved_marines(&mine);
        match k.tactic {
            _ if self.tp.is_some() => {
                let guard_on = k.defend && (branch != Some(Branch::Defence) || self.react_marines_attack());
                let mut owned = kept.clone();
                if let Some((_, guards)) = base.filter(|_| guard_on).and_then(|b| self.guard_order(f, &mine, b, &kept)) {
                    owned.extend(guards);
                }
                self.tp_army(f, &mine, &owned, &mut cmds)
            }
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
        if k.defend && (branch != Some(Branch::Defence) || self.react_marines_attack()) {
            if let Some(b) = base {
                self.defend(f, &mine, b, &kept, &mut cmds);
            }
        }
        self.react_marines(f, &mine, &mut cmds);

        Act { cmds, advance_frames: STEP_FRAMES, valid_until_frame: None, surrender: None }
    }

    fn react_stats(&self) -> Option<ReactStats> {
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
    fn order_build(&mut self, ty: u16, f: &RawFrame, mine: &[&UnitState], base: Pos, busy: &[UnitId], cmds: &mut Vec<Command>) -> bool {
        let reserved: Vec<(Tile, u16)> = self.pending.iter().map(|p| (p.tile, p.ty)).chain(self.react_reserved_sites()).collect();
        let Some(tile) = self.place(ty, f, mine, base, &reserved) else { return false };
        let center = (tile.0 * 32 + 16, tile.1 * 32 + 16);
        let skip_constructing = !self.kit.tech.is_empty() || self.kit.tp.is_some();
        let worker = mine
            .iter()
            .filter(|u| u.unit_type == self.kit.worker && u.has(flags::COMPLETED) && !u.has(flags::CARRYING) && !busy.contains(&u.id))
            .filter(|u| !(skip_constructing && u.has(flags::CONSTRUCTING)))
            .min_by_key(|u| if self.tp.is_some() { (tp::pick_key(true, u.pos, center), u.id) } else { (dist2(u.pos, center), 0) });
        let Some(w) = worker else { return false };
        cmds.push(Command::Build { worker: w.id, unit_type: ty, tile });
        self.pending.push(Pending { ty, tile, worker: w.id, issued: f.frame });
        true
    }

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

    fn free_worker(&self, mine: &[&UnitState]) -> Option<UnitId> {
        let jobs = self.react_busy();
        let busy = |id: UnitId| self.pending.iter().any(|p| p.worker == id) || self.gas_crew.contains(&id) || self.scout == Some(id) || jobs.contains(&id);
        mine.iter()
            .filter(|u| u.unit_type == self.kit.worker && u.has(flags::COMPLETED) && !u.has(flags::CARRYING) && !u.has(flags::CONSTRUCTING) && !busy(u.id))
            .min_by_key(|u| u.id)
            .map(|u| u.id)
    }

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

    fn defend(&mut self, f: &RawFrame, mine: &[&UnitState], base: Pos, kept: &[UnitId], cmds: &mut Vec<Command>) {
        let Some((at, guards)) = self.guard_order(f, mine, base, kept) else { return };
        for chunk in guards.chunks(12) {
            cmds.push(Command::Unit { units: chunk.to_vec(), order: CmdOrder::Attack(Target::Pos(at)), queued: false });
        }
        self.last_defend = f.frame;
    }

    fn guard_order(&self, f: &RawFrame, mine: &[&UnitState], base: Pos, kept: &[UnitId]) -> Option<(Pos, Vec<UnitId>)> {
        let react = self.kit.react.is_some();
        let has_guard = self.kit.tactic == Tactic::Drop || self.kit.attacker != self.kit.army || react;
        if !has_guard || !ReactState::due_on(self.tp.is_some(), Some(self.last_defend), f.frame, WAVE_PERIOD) {
            return None;
        }
        let bases: Vec<Pos> = if react { mine.iter().filter(|u| is_resource_depot(u.unit_type)).map(|u| u.pos).collect() } else { vec![base] };
        let near_base = |p: Pos| bases.iter().map(|b| dist2(p, *b)).min().unwrap_or(i64::MAX);
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
        Some((if self.tp.is_some() { tp::snap(t.pos) } else { t.pos }, guards))
    }
}

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

fn is_neutral_owner(owner: u8) -> bool {
    owner >= 8
}

#[cfg(test)]
mod tests;
