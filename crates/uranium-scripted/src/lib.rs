//! T0 rule-based opponents (spec 021). They emit spec-010 `Command`s directly (021-R6).
//!
//! One shared "brain" (workers, supply, production, memory, attack waves, hunting) is
//! parameterized by a per-race kit. M3 builds are gas-free so every race reliably plays a
//! full melee game to the end (021-R1): Terran marines, Protoss zealots, Zerg zerglings.

pub mod placement;
pub mod types;

use std::collections::BTreeMap;

use uranium_proto::*;

use crate::placement::Placer;
use crate::types::*;

const STEP_FRAMES: u32 = 8;
const WAVE_PERIOD: u32 = 48;
const PENDING_TIMEOUT: u32 = 24 * 25;
/// Hunting grid cell size in tiles.
const CELL: u16 = 8;

/// A scripted bot controlling one player.
pub trait Bot: Send {
    fn act(&mut self, f: &RawFrame) -> Act;
}

/// Picks the T0 bot for our race (a Random lobby pick is already resolved at game start).
pub fn for_game(info: &GameInfo) -> Box<dyn Bot> {
    let race = info.players.iter().find(|p| p.is_self).map(|p| p.race).unwrap_or(Race::Terran);
    let kit = match race {
        Race::Protoss => Kit::protoss(),
        Race::Zerg => Kit::zerg(),
        _ => Kit::terran(),
    };
    Box::new(Brain::new(info, kit))
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
}

impl Kit {
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
        }
    }
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
    /// Last seen enemy building positions (own vision only, 030-R2 style memory).
    enemy_buildings: BTreeMap<UnitId, Pos>,
    scouted: Vec<bool>,
    /// Frame each hunting cell was last visible.
    cell_seen: Vec<u32>,
    cells_w: u16,
    cells_h: u16,
    last_wave: u32,
}

impl Brain {
    pub fn new(info: &GameInfo, kit: Kit) -> Self {
        let me = info.self_player;
        let my_start = info.players.iter().find(|p| p.is_self).and_then(|p| p.start_tile);
        let starts = info.static_map.start_locations.clone();
        let scouted = starts.iter().map(|s| Some(*s) == my_start).collect();
        let (w, h) = info.map_size_tiles;
        let (cells_w, cells_h) = (w.div_ceil(CELL), h.div_ceil(CELL));
        Self {
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
        }
    }

    fn update_memory(&mut self, f: &RawFrame) {
        for u in f.units.iter().filter(|u| u.owner != self.me && is_building(u.unit_type) && !is_neutral_owner(u.owner)) {
            self.enemy_buildings.insert(u.id, u.pos);
        }
        // Forget buildings whose tile is visible but which are no longer there.
        let seen: Vec<UnitId> = f.units.iter().map(|u| u.id).collect();
        self.enemy_buildings.retain(|id, pos| seen.contains(id) || !f.visibility.get(pos.0 / 32, pos.1 / 32));
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
        if let Some(p) = self.enemy_buildings.values().min_by_key(|p| dist2(**p, from)) {
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

    fn place(&self, ty: u16, f: &RawFrame, mine: &[&UnitState], base: Pos, reserved: &[(Tile, u16)]) -> Option<Tile> {
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

        let mut minerals = f.self_state.minerals - self.pending.iter().map(|p| cost(p.ty)).sum::<i32>();
        let mut supply_free = f.self_state.supply_total as i32 - f.self_state.supply_used as i32;
        let base = mine.iter().find(|u| is_resource_depot(u.unit_type)).map(|u| u.pos);
        let busy: Vec<UnitId> = self.pending.iter().map(|p| p.worker).collect();
        let mut larvae: Vec<&UnitState> = mine.iter().copied().filter(|u| u.unit_type == LARVA).collect();

        // Workers: idle ones mine (or return cargo).
        for w in mine.iter().filter(|u| u.unit_type == k.worker && u.has(flags::COMPLETED) && u.has(flags::IDLE)) {
            if busy.contains(&w.id) {
                continue;
            }
            if w.has(flags::CARRYING) {
                cmds.push(Command::Unit { units: vec![w.id], order: CmdOrder::ReturnCargo, queued: false });
            } else if let Some(m) = nearest(w.pos, &minerals_vis) {
                cmds.push(Command::Unit { units: vec![w.id], order: CmdOrder::Gather(m.id), queued: false });
            }
        }

        // Supply.
        let prod_count = count(k.production) + self.pending.iter().filter(|p| p.ty == k.production).count();
        let supply_low = f.self_state.supply_total < 400 && supply_free <= 4 + 4 * prod_count as i32;
        match k.supply {
            Supply::Building(ty) => {
                let in_progress = mine.iter().filter(|u| u.unit_type == ty && !u.has(flags::COMPLETED)).count() + self.pending.iter().filter(|p| p.ty == ty).count();
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

        // Production buildings.
        let workers = count(k.worker);
        if prod_count < k.production_cap && workers >= k.production_at_workers[prod_count.min(2)] && minerals >= cost(k.production) {
            if let Some(b) = base {
                if self.order_build(k.production, f, &mine, b, &busy, &mut cmds) {
                    minerals -= cost(k.production);
                }
            }
        }

        // Units: workers first (until production exists), then army.
        let prod_ready = mine.iter().any(|u| u.unit_type == k.production && u.has(flags::COMPLETED));
        match k.supply {
            Supply::Larva(_) => {
                while let Some(l) = larvae.pop() {
                    let ty = if prod_ready && workers >= k.max_workers.min(9) { k.army } else if workers < k.max_workers { k.worker } else { break };
                    if minerals < cost(ty) || supply_free < 2 {
                        break;
                    }
                    cmds.push(Command::Train { building: l.id, unit_type: ty });
                    minerals -= cost(ty);
                    supply_free -= 2;
                }
            }
            Supply::Building(_) => {
                for d in mine.iter().filter(|u| is_resource_depot(u.unit_type) && u.has(flags::COMPLETED) && !u.has(flags::TRAINING)) {
                    if workers < k.max_workers && minerals >= 50 && supply_free >= 2 {
                        cmds.push(Command::Train { building: d.id, unit_type: k.worker });
                        minerals -= 50;
                        supply_free -= 2;
                    }
                }
                for p in mine.iter().filter(|u| u.unit_type == k.production && u.has(flags::COMPLETED) && !u.has(flags::TRAINING)) {
                    if minerals >= cost(k.army) && supply_free >= 2 {
                        cmds.push(Command::Train { building: p.id, unit_type: k.army });
                        minerals -= cost(k.army);
                        supply_free -= 2;
                    }
                }
            }
        }

        // Army: attack-move waves.
        let army: Vec<&UnitState> = mine.iter().copied().filter(|u| u.unit_type == k.army && u.has(flags::COMPLETED)).collect();
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

        Act { cmds, advance_frames: STEP_FRAMES, valid_until_frame: None }
    }
}

impl Brain {
    /// Picks a site and the nearest free worker; records the pending build.
    fn order_build(&mut self, ty: u16, f: &RawFrame, mine: &[&UnitState], base: Pos, busy: &[UnitId], cmds: &mut Vec<Command>) -> bool {
        let reserved: Vec<(Tile, u16)> = self.pending.iter().map(|p| (p.tile, p.ty)).collect();
        let Some(tile) = self.place(ty, f, mine, base, &reserved) else { return false };
        let center = (tile.0 * 32 + 16, tile.1 * 32 + 16);
        let worker = mine
            .iter()
            .filter(|u| u.unit_type == self.kit.worker && u.has(flags::COMPLETED) && !u.has(flags::CARRYING) && !busy.contains(&u.id))
            .min_by_key(|u| dist2(u.pos, center));
        let Some(w) = worker else { return false };
        cmds.push(Command::Build { worker: w.id, unit_type: ty, tile });
        self.pending.push(Pending { ty, tile, worker: w.id, issued: f.frame });
        true
    }
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
}
