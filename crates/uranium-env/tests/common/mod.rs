//! Synthetic game states for 030 / 031 / 034 tests (no game data needed).
#![allow(dead_code)]

use uranium_env::features::{MapInfo, ORDER_NONE};
use uranium_env::types::{TypeTables, order_id};
use uranium_proto::*;

pub const W: u16 = 96;
pub const H: u16 = 96;
pub const SELF: u8 = 0;
pub const ENEMY: u8 = 1;
pub const NEUTRAL: u8 = 11;

pub fn ty(name: &str) -> u16 {
    TypeTables::get().unit_by_name(name).unwrap_or_else(|| panic!("{name}")).id as u16
}

pub fn game_info() -> GameInfo {
    let n = W as usize * H as usize;
    let static_map = StaticMap {
        width_tiles: W,
        height_tiles: H,
        walkable: BitGrid::from_bytes(W * 4, H * 4, &vec![1; n * 16]),
        buildable: BitGrid::from_bytes(W, H, &vec![1; n]),
        ground_height: (0..n).map(|i| (i % 6) as u8).collect(),
        start_locations: vec![(10, 10), (80, 80)],
    };
    GameInfo {
        map_name: "synthetic".into(),
        map_hash: [0; 32],
        map_size_tiles: (W, H),
        self_player: SELF,
        players: vec![
            PlayerInfo { id: SELF, name: "a".into(), race: Race::Terran, start_tile: Some((10, 10)), is_self: true, is_enemy: false },
            PlayerInfo { id: ENEMY, name: "b".into(), race: Race::Zerg, start_tile: Some((80, 80)), is_self: false, is_enemy: true },
        ],
        static_map,
        latency_frames: 0,
    }
}

pub fn map_info() -> MapInfo {
    MapInfo::new(&game_info(), 0, None).unwrap()
}

pub struct Builder {
    pub units: Vec<UnitState>,
    next: u32,
}

impl Builder {
    pub fn new() -> Self {
        Self { units: vec![], next: 1 }
    }

    /// Adds a completed unit; buildings and resources are placed by top-left tile, centred on
    /// their footprint like BWAPI (`tile * 32 + tile_size * 16`).
    pub fn add(&mut self, owner: u8, name: &str, tile: (i32, i32)) -> u32 {
        let t = TypeTables::get().unit_by_name(name).unwrap_or_else(|| panic!("{name}"));
        let pos = if t.is_building() || t.is_resource() {
            ((tile.0 * 32 + t.tile_width * 16) as u16, (tile.1 * 32 + t.tile_height * 16) as u16)
        } else {
            ((tile.0 * 32 + 16) as u16, (tile.1 * 32 + 16) as u16)
        };
        let id = self.next;
        self.next += 1;
        let none = ty("None");
        self.units.push(UnitState {
            id,
            owner,
            unit_type: t.id as u16,
            pos,
            hp: t.max_hp as u16,
            shield: t.max_shields as u16,
            energy: if t.max_energy > 0 { 200 } else { 0 },
            order: ORDER_NONE,
            flags: flags::COMPLETED,
            build_queue: [none; 5],
            resources: if t.is_resource() { 1500 } else { 0 },
            ..Default::default()
        });
        id
    }

    pub fn get(&mut self, id: u32) -> &mut UnitState {
        self.units.iter_mut().find(|u| u.id == id).unwrap()
    }
}

fn full(w: u16, h: u16, v: bool) -> BitGrid {
    BitGrid::from_bytes(w, h, &vec![v as u8; w as usize * h as usize])
}

pub fn frame(frame: u32, units: Vec<UnitState>, techs: bool) -> RawFrame {
    RawFrame {
        frame,
        self_state: PlayerState {
            minerals: 10_000,
            gas: 10_000,
            supply_used: 40,
            supply_total: 200,
            upgrades: vec![0; 63],
            tech: if techs { u64::MAX >> 17 } else { 0 },
            researching: 0,
            upgrading: 0,
        },
        units,
        visibility: full(W, H, true),
        explored: full(W, H, true),
        creep: full(W, H, false),
        events: vec![],
        effects: vec![],
    }
}

/// A base with every Terran building, addon, unit and state the 034 actions need.
pub fn rich_units() -> Builder {
    let mut b = Builder::new();
    let cc = b.add(SELF, "Terran_Command_Center", (10, 10));
    b.get(cc).has_addon = true;
    b.add(SELF, "Terran_Comsat_Station", (14, 11));
    b.add(SELF, "Terran_Command_Center", (10, 20));
    let lifted_cc = b.add(SELF, "Terran_Command_Center", (60, 60));
    b.get(lifted_cc).flags |= flags::LIFTED;
    let rax = b.add(SELF, "Terran_Barracks", (20, 10));
    b.get(rax).build_queue[0] = ty("Terran_Marine");
    b.get(rax).remaining_train = 100;
    b.add(SELF, "Terran_Barracks", (20, 40));
    let lifted_rax = b.add(SELF, "Terran_Barracks", (60, 70));
    b.get(lifted_rax).flags |= flags::LIFTED;
    b.add(SELF, "Terran_Academy", (20, 15));
    let busy_academy = b.add(SELF, "Terran_Academy", (20, 50));
    b.get(busy_academy).remaining_research = 100;
    b.get(busy_academy).order = order_id("ResearchTech").unwrap();
    b.add(SELF, "Terran_Engineering_Bay", (20, 20));
    let busy_bay = b.add(SELF, "Terran_Engineering_Bay", (20, 55));
    b.get(busy_bay).remaining_research = 100;
    b.get(busy_bay).order = order_id("Upgrade").unwrap();
    b.add(SELF, "Terran_Armory", (26, 20));
    let fac = b.add(SELF, "Terran_Factory", (26, 10));
    b.get(fac).has_addon = true;
    b.add(SELF, "Terran_Machine_Shop", (30, 11));
    b.add(SELF, "Terran_Factory", (26, 30));
    let sp = b.add(SELF, "Terran_Starport", (36, 10));
    b.get(sp).has_addon = true;
    b.add(SELF, "Terran_Control_Tower", (40, 11));
    b.add(SELF, "Terran_Starport", (36, 20));
    let sp3 = b.add(SELF, "Terran_Starport", (46, 10));
    b.get(sp3).has_addon = true;
    let tower = b.add(SELF, "Terran_Control_Tower", (50, 11));
    b.get(tower).flags &= !flags::COMPLETED;
    let sf = b.add(SELF, "Terran_Science_Facility", (36, 30));
    b.get(sf).has_addon = true;
    b.add(SELF, "Terran_Physics_Lab", (40, 31));
    b.add(SELF, "Terran_Science_Facility", (46, 30));
    let sf3 = b.add(SELF, "Terran_Science_Facility", (36, 40));
    b.get(sf3).has_addon = true;
    b.add(SELF, "Terran_Covert_Ops", (40, 41));
    let cc4 = b.add(SELF, "Terran_Command_Center", (46, 40));
    b.get(cc4).has_addon = true;
    let silo = b.add(SELF, "Terran_Nuclear_Silo", (50, 41));
    b.get(silo).nuke_armed = true;
    let cc5 = b.add(SELF, "Terran_Command_Center", (46, 50));
    b.get(cc5).has_addon = true;
    b.add(SELF, "Terran_Nuclear_Silo", (50, 51));
    let bunker = b.add(SELF, "Terran_Bunker", (20, 25));
    let _ = bunker;
    b.add(SELF, "Terran_Refinery", (2, 30));
    let depot = b.add(SELF, "Terran_Supply_Depot", (26, 40));
    b.get(depot).flags &= !flags::COMPLETED;
    b.get(depot).remaining_build = 200;

    b.add(SELF, "Terran_SCV", (16, 16));
    let carrying = b.add(SELF, "Terran_SCV", (17, 16));
    b.get(carrying).flags |= flags::CARRYING;
    b.add(SELF, "Terran_Marine", (18, 18));
    let hurt = b.add(SELF, "Terran_Marine", (18, 19));
    b.get(hurt).hp = 20;
    let locked = b.add(SELF, "Terran_Marine", (18, 20));
    b.get(locked).status_timers.lockdown = 50;
    b.add(SELF, "Terran_Firebat", (19, 18));
    b.add(SELF, "Terran_Medic", (19, 19));
    b.add(SELF, "Terran_Ghost", (19, 20));
    let vult = b.add(SELF, "Terran_Vulture", (19, 21));
    b.get(vult).spider_mines = 3;
    b.add(SELF, "Terran_Siege_Tank_Tank_Mode", (19, 22));
    b.add(SELF, "Terran_Siege_Tank_Siege_Mode", (19, 23));
    let gol = b.add(SELF, "Terran_Goliath", (19, 24));
    b.get(gol).hp = 50;
    b.add(SELF, "Terran_Wraith", (22, 18));
    let cloaked = b.add(SELF, "Terran_Wraith", (22, 19));
    b.get(cloaked).flags |= flags::CLOAKED;
    let ship = b.add(SELF, "Terran_Dropship", (22, 20));
    let cargo = b.add(SELF, "Terran_Marine", (22, 20));
    b.get(cargo).loaded_in = Some(ship);
    b.get(cargo).flags |= flags::LOADED;
    b.add(SELF, "Terran_Science_Vessel", (22, 21));
    b.add(SELF, "Terran_Battlecruiser", (22, 22));
    b.add(SELF, "Terran_Valkyrie", (22, 23));

    b.add(NEUTRAL, "Resource_Mineral_Field", (4, 4));
    b.add(NEUTRAL, "Resource_Vespene_Geyser", (2, 40));
    b.add(ENEMY, "Zerg_Zergling", (70, 70));
    b.add(ENEMY, "Protoss_Dragoon", (71, 70));
    b.add(ENEMY, "Zerg_Hatchery", (80, 80));
    b
}

/// Arguments for action `a` that satisfy the masks of `o` (034-R4): the selection is the
/// eligible own entity ranked first by `rank` (lowest key), the unit target likewise, the
/// location the first allowed tile ranked by `loc_rank`; the target is never a selected unit. `None` when the type mask is closed or
/// an argument has no candidate. One unit is selected: enough for every action (034-R7 allows a
/// building only alone) and for 010-AC4 / 031-AC1.
pub fn valid_action(
    o: &uranium_env::features::Observation,
    a: &uranium_env::actions::Action,
    rank: &dyn Fn(Role, usize) -> Option<i64>,
    loc_rank: &dyn Fn(usize) -> Option<i64>,
) -> Option<uranium_env::features::executor::HostAction> {
    use uranium_env::actions::{CAPABILITIES, LOC_CLASSES, TargetKind};
    use uranium_env::features::{CELLS, OWNER_SELF};
    if !o.masks.type_mask[a.id as usize] {
        return None;
    }
    let k = CAPABILITIES.len();
    let cap = |name: &str| CAPABILITIES.iter().position(|c| *c == name).unwrap();
    let best = |role: Role, name: &str, own: bool, skip: &[usize]| {
        let c = cap(name);
        (0..o.obs.n)
            .filter(|&i| o.masks.unit_cap[i * k + c] && (!own || o.obs.entity_cat[i * 4 + 1] == OWNER_SELF) && !skip.contains(&i))
            .filter_map(|i| rank(role, i).map(|r| (r, i)))
            .min()
            .map(|(_, i)| i)
    };
    let selected = if a.uses_select { vec![best(Role::Select, &a.select_cap, true, &[])?] } else { vec![] };
    // A unit never targets itself (the game rejects following or right-clicking oneself).
    let target_unit = match a.target {
        TargetKind::Unit => Some(best(Role::Target, &a.target_cap, false, &selected)?),
        _ => None,
    };
    let target_loc = match a.target {
        TargetKind::Loc => {
            let l = LOC_CLASSES.iter().position(|c| *c == a.loc_class).unwrap();
            let tiles = &o.masks.loc_class_mask[l * CELLS..(l + 1) * CELLS];
            Some((0..CELLS).filter(|&t| tiles[t]).filter_map(|t| loc_rank(t).map(|r| (r, t))).min()?.1)
        }
        _ => None,
    };
    Some(uranium_env::features::executor::HostAction { action: a.id, delay_bucket: 0, queued: false, selected, target_unit, target_loc })
}

/// Which argument `valid_action` is ranking candidates for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Select,
    Target,
}

/// The 034 `command` name a protocol command came from (010-AC4).
pub fn command_name(c: &Command) -> &'static str {
    match c {
        Command::Unit { order, .. } => match order {
            CmdOrder::Move(_) => "UnitMove",
            CmdOrder::Attack(_) => "UnitAttack",
            CmdOrder::Patrol(_) => "UnitPatrol",
            CmdOrder::Hold => "UnitHold",
            CmdOrder::Stop => "UnitStop",
            CmdOrder::RightClick(_) => "UnitRightClick",
            CmdOrder::Siege => "UnitSiege",
            CmdOrder::Unsiege => "UnitUnsiege",
            CmdOrder::Cloak => "UnitCloak",
            CmdOrder::Decloak => "UnitDecloak",
            CmdOrder::Lift => "UnitLift",
            CmdOrder::Land(_) => "UnitLand",
            CmdOrder::Load(_) => "UnitLoad",
            CmdOrder::UnloadAll(_) => "UnitUnloadAll",
            CmdOrder::Unload(_) => "UnitUnload",
            CmdOrder::Repair(_) => "UnitRepair",
            CmdOrder::Gather(_) => "UnitGather",
            CmdOrder::ReturnCargo => "UnitReturnCargo",
            CmdOrder::UseTech { .. } => "UnitUseTech",
            CmdOrder::Burrow | CmdOrder::Unburrow => "UnitBurrow",
        },
        Command::Build { .. } => "Build",
        Command::BuildAddon { .. } => "BuildAddon",
        Command::Train { .. } => "Train",
        Command::Morph { .. } => "Morph",
        Command::Research { .. } => "Research",
        Command::Upgrade { .. } => "Upgrade",
        Command::Cancel { .. } => "Cancel",
        Command::SetRally { .. } => "SetRally",
    }
}
