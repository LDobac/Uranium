//! 021-R8 scouting-reactive Terran teacher (`T0Build::TerranReact`, opponent id `t0-terran-react`,
//! AMB-021-4, 6th advisory `reports/m6_rl_stall_advice6.md` §2).
//!
//! The teacher opens 9 depot / 11 barracks / 15 depot, sends one SCV to the start locations once
//! the barracks is under construction, and identifies a rush only from the units it sees in the
//! current observation (rush rule R, 021-R8 ①–④). Without R it stays in the macro branch M: it
//! expands to the natural and plays the `terran()` macro (3 barracks, 10-marine waves, home guard,
//! one natural bunker after 4:00). R switches it at that very observation to the defence branch D:
//! a bunker toward the main choke, marines load / hold beside it, SCVs pull and repair; D returns
//! to M once it has 8 marines and 60 s without contact.
//!
//! Every action is one 010 `Command` the model can also issue (034: Build_Bunker,
//! Build_CommandCenter, Build_*, Train_*, Load_Unit, Move_Pos, Hold, Attack_Pos, Attack_Unit,
//! Repair_Unit, Gather_Unit), at most 12 units each. What the teacher carries between
//! observations is what the model also has (030 §2): remembered enemy buildings (enemy main
//! found), its own units (the bunker shows the defence) and elapsed time; the rush rules read
//! only the current observation. No randomness: the same observations give the same commands.

use std::collections::BTreeSet;

use serde::Serialize;
use uranium_env::scout::{is_army_type, is_ground_army_type};
use uranium_proto::*;

use crate::types::*;
use crate::{Brain, dist2, done, is_neutral_owner, nearest};

/// Game frames per second (042-R14).
pub const FPS: u32 = 24;
const TILE: i64 = 32;
/// Bunker hit points (BWAPI type table, 350).
pub const BUNKER_HP: u16 = 350;

/// 021-R8 numbers (6th advisory §2 values, AMB-021-4). Frames are game frames (24 per second),
/// distances tiles.
#[derive(Clone, Debug, PartialEq)]
pub struct ReactParams {
    /// Opening: first depot / barracks / second depot at this supply (units, a marine = 1).
    pub first_depot_supply: u32,
    pub barracks_supply: u32,
    pub second_depot_supply: u32,
    /// R ① spawning pool seen before this frame (2:30).
    pub pool_before: u32,
    /// R ② at least `lings_min` zerglings in one observation before `lings_before` (3:30).
    pub lings_min: usize,
    pub lings_before: u32,
    /// R ③ enemy army (033) within this many tiles of an own command center.
    pub army_radius: i64,
    /// R ④ the enemy main (a depot within `main_radius` tiles of a start location) seen with a
    /// spawning pool and at most `main_drones_max` drones there, before `main_before` (3:30,
    /// AMB-021-4 메인 임시 결정: the scouting window).
    pub main_radius: i64,
    pub main_drones_max: usize,
    pub main_before: u32,
    /// R is judged only before this frame (8:00 = the teacher data cut, 040-R21).
    pub judge_until: u32,
    /// M: without the enemy main found the expansion waits until this frame (3:40).
    pub expand_hold_until: u32,
    pub expand_minerals: i32,
    /// D bunker: centre 4–6 tiles from the command center centre, toward the nearest choke.
    pub bunker_min: i64,
    pub bunker_max: i64,
    /// Marines loaded into each bunker.
    pub load_max: usize,
    /// D SCV pull: enemy army within `pull_radius` tiles of a command center pulls
    /// min(`pull_max`, SCVs − `pull_keep`) SCVs; targets are re-picked every `retarget` frames.
    pub pull_radius: i64,
    pub pull_max: usize,
    pub pull_keep: usize,
    pub retarget: u32,
    /// D repair: a bunker below this percentage of its hit points gets `repairers` SCVs until full.
    pub repair_below_pct: u32,
    pub repairers: usize,
    /// D: pulled SCVs gather again after no enemy army within `release_radius` tiles of a
    /// command center for `release_after` frames (10 s).
    pub release_radius: i64,
    pub release_after: u32,
    /// D → M: at least `macro_marines` marines and `macro_quiet` frames (60 s) without contact
    /// (enemy army within `army_radius` tiles of a command center).
    pub macro_marines: usize,
    pub macro_quiet: u32,
    /// M: the natural bunker once the natural command center is complete and not before this
    /// frame (4:00, the user constraint: no pre-emptive bunker).
    pub natural_bunker_from: u32,
    /// The scout looks at the enemy main from this close (tiles), or for at most this long after
    /// finding it (frames), then mines again.
    pub scout_look: i64,
    pub scout_look_max: u32,
}

impl Default for ReactParams {
    fn default() -> Self {
        Self {
            first_depot_supply: 9,
            barracks_supply: 11,
            second_depot_supply: 15,
            pool_before: 150 * FPS,
            lings_min: 2,
            lings_before: 210 * FPS,
            army_radius: 20,
            main_radius: 10,
            main_drones_max: 8,
            main_before: 210 * FPS,
            judge_until: 480 * FPS,
            expand_hold_until: 220 * FPS,
            expand_minerals: 400,
            bunker_min: 4,
            bunker_max: 6,
            load_max: 4,
            pull_radius: 12,
            pull_max: 8,
            pull_keep: 4,
            retarget: 48,
            repair_below_pct: 90,
            repairers: 2,
            release_radius: 16,
            release_after: 10 * FPS,
            macro_marines: 8,
            macro_quiet: 60 * FPS,
            natural_bunker_from: 240 * FPS,
            scout_look: 5,
            scout_look_max: 10 * FPS,
        }
    }
}

/// 021-R8 branch.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Branch {
    /// M: expand and macro (also the opening, before any rush evidence).
    #[default]
    Macro,
    /// D: rush defence.
    Defence,
}

/// Which 021-R8 rush rule held.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RushRule {
    /// ① spawning pool before 2:30.
    Pool,
    /// ② two or more zerglings in one observation before 3:30.
    Lings,
    /// ③ enemy army within 20 tiles of a command center.
    Army,
    /// ④ enemy main with a pool and at most 8 drones.
    Main,
}

/// What the teacher did in one game (session record `scripted.react`, 050-R8). Frames.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct ReactStats {
    /// Enemy main found (an enemy building within `main_radius` of a start location).
    pub main_frame: Option<u32>,
    /// First rush identification and its rule.
    pub rush_frame: Option<u32>,
    pub rush_rule: Option<RushRule>,
    /// M → D switches.
    pub defences: u32,
    /// First D → M switch.
    pub macro_frame: Option<u32>,
    /// Natural command center ordered.
    pub expand_frame: Option<u32>,
    /// Branch at the last observation.
    pub branch: Branch,
}

/// Map precompute hints (031 `MapPrecompute`): resource-depot top-left tiles of every base and
/// choke tiles. Empty = no precompute: no expansion, bunkers face the map centre.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MapHints {
    pub base_locs: Vec<Tile>,
    pub chokes: Vec<Tile>,
}

pub(crate) struct ReactState {
    pub p: ReactParams,
    pub branch: Branch,
    pub chokes: Vec<Tile>,
    /// Natural: the base location nearest our start that is no start location.
    pub natural: Option<Tile>,
    /// Start tile of the enemy main once found.
    pub enemy_main: Option<Tile>,
    /// Last frame with enemy army within `army_radius` / `release_radius` of a command center.
    pub last_contact: u32,
    pub last_near: u32,
    pub pulled: Vec<UnitId>,
    pub repairers: Vec<UnitId>,
    pub holding: BTreeSet<UnitId>,
    /// Frame of the last marine (load / hold) and SCV (pull / repair) orders.
    pub last_marines: Option<u32>,
    pub last_scvs: Option<u32>,
    pub stats: ReactStats,
}

impl ReactState {
    pub fn new(p: ReactParams, hints: &MapHints, home: Option<Pos>, starts: &[Tile]) -> Self {
        // The natural: nearest base location to our command center that is not a start location
        // (ours included) — 6th advisory: "base_locs 중 CC에 가장 가까운 곳".
        let natural = home.and_then(|h| {
            hints
                .base_locs
                .iter()
                .copied()
                .filter(|b| !starts.iter().any(|s| (b.0 as i64 - s.0 as i64).abs() <= 4 && (b.1 as i64 - s.1 as i64).abs() <= 4))
                .min_by_key(|b| (dist2(depot_center(*b), h), *b))
        });
        Self {
            p,
            branch: Branch::Macro,
            chokes: hints.chokes.clone(),
            natural,
            enemy_main: None,
            last_contact: 0,
            last_near: 0,
            pulled: Vec::new(),
            repairers: Vec::new(),
            holding: BTreeSet::new(),
            last_marines: None,
            last_scvs: None,
            stats: ReactStats::default(),
        }
    }

    fn due(last: Option<u32>, frame: u32, every: u32) -> bool {
        last.is_none_or(|l| frame >= l + every)
    }
}

/// Command-center centre of a depot top-left tile.
pub fn depot_center(t: Tile) -> Pos {
    (t.0 * 32 + 64, t.1 * 32 + 48)
}

/// Bunker centre of a top-left tile.
fn bunker_center(t: Tile) -> Pos {
    (t.0 * 32 + 48, t.1 * 32 + 32)
}

fn within(a: Pos, b: Pos, tiles: i64) -> bool {
    dist2(a, b) <= (tiles * TILE) * (tiles * TILE)
}

/// `from` moved `tiles` toward `to` (pixels; `to == from` keeps `from`).
fn toward(from: Pos, to: Pos, tiles: f64) -> Pos {
    let (dx, dy) = (to.0 as f64 - from.0 as f64, to.1 as f64 - from.1 as f64);
    let len = (dx * dx + dy * dy).sqrt();
    if len < 1.0 {
        return from;
    }
    let k = tiles * TILE as f64 / len;
    ((from.0 as f64 + dx * k).round().max(0.0) as u16, (from.1 as f64 + dy * k).round().max(0.0) as u16)
}

fn is_enemy(u: &UnitState, me: u8) -> bool {
    u.owner != me && !is_neutral_owner(u.owner)
}

/// The 021-R8 rush rule that holds in this observation (own vision only), if any.
/// `depots`: own command centers; `starts`: start locations other than ours.
pub fn rush_rule(p: &ReactParams, f: &RawFrame, me: u8, depots: &[Pos], starts: &[Tile]) -> Option<RushRule> {
    if f.frame >= p.judge_until {
        return None;
    }
    let enemies: Vec<&UnitState> = f.units.iter().filter(|u| is_enemy(u, me)).collect();
    if f.frame < p.pool_before && enemies.iter().any(|u| u.unit_type == SPAWNING_POOL) {
        return Some(RushRule::Pool);
    }
    if f.frame < p.lings_before && enemies.iter().filter(|u| u.unit_type == ZERGLING).count() >= p.lings_min {
        return Some(RushRule::Lings);
    }
    if enemies.iter().any(|u| is_army_type(u.unit_type) && depots.iter().any(|d| within(u.pos, *d, p.army_radius))) {
        return Some(RushRule::Army);
    }
    if f.frame < p.main_before {
        for s in starts {
            let c = depot_center(*s);
            let near = |u: &&&UnitState| within(u.pos, c, p.main_radius);
            let depot = enemies.iter().filter(near).any(|u| is_resource_depot(u.unit_type));
            let pool = enemies.iter().filter(near).any(|u| u.unit_type == SPAWNING_POOL);
            let drones = enemies.iter().filter(near).filter(|u| u.unit_type == DRONE).count();
            if depot && pool && drones <= p.main_drones_max {
                return Some(RushRule::Main);
            }
        }
    }
    None
}

impl Brain {
    /// Own command centers (completed or not).
    fn own_depots(mine: &[&UnitState]) -> Vec<Pos> {
        mine.iter().filter(|u| is_resource_depot(u.unit_type)).map(|u| u.pos).collect()
    }

    /// The command center nearest our start location (the main), or the first one.
    pub(crate) fn home_depot(&self, mine: &[&UnitState]) -> Option<Pos> {
        let depots = Self::own_depots(mine);
        match self.home {
            Some(h) => depots.into_iter().min_by_key(|d| (dist2(*d, h), *d)),
            None => depots.first().copied(),
        }
    }

    /// Start locations other than ours.
    fn other_starts(&self) -> Vec<Tile> {
        let home = self.home;
        self.starts.iter().copied().filter(|s| Some(depot_center(*s)) != home).collect()
    }

    /// Per observation: enemy main found, contact times, rule R (M → D) and the D → M rule.
    pub(crate) fn react_observe(&mut self, f: &RawFrame, mine: &[&UnitState]) {
        let others = self.other_starts();
        let me = self.me;
        let Some(r) = self.react.as_mut() else { return };
        let p = r.p.clone();
        if r.enemy_main.is_none() {
            // Remembered enemy buildings (030 §2: the model has them as entities too).
            let found = others.iter().copied().find(|s| self.enemy_buildings.values().any(|(pos, _)| within(*pos, depot_center(*s), p.main_radius)));
            if let Some(s) = found {
                r.enemy_main = Some(s);
                r.stats.main_frame = Some(f.frame);
            }
        }
        let depots = Self::own_depots(mine);
        let army: Vec<Pos> = f.units.iter().filter(|u| is_enemy(u, me) && is_army_type(u.unit_type)).map(|u| u.pos).collect();
        if army.iter().any(|a| depots.iter().any(|d| within(*a, *d, p.army_radius))) {
            r.last_contact = f.frame;
        }
        if army.iter().any(|a| depots.iter().any(|d| within(*a, *d, p.release_radius))) {
            r.last_near = f.frame;
        }
        if r.branch == Branch::Macro {
            if let Some(rule) = rush_rule(&p, f, me, &depots, &others) {
                r.branch = Branch::Defence;
                r.last_contact = f.frame;
                r.stats.defences += 1;
                if r.stats.rush_frame.is_none() {
                    r.stats.rush_frame = Some(f.frame);
                    r.stats.rush_rule = Some(rule);
                }
            }
        } else {
            let marines = mine.iter().filter(|u| u.unit_type == MARINE && u.has(flags::COMPLETED)).count();
            if marines >= p.macro_marines && f.frame >= r.last_contact + p.macro_quiet {
                r.branch = Branch::Macro;
                r.stats.macro_frame.get_or_insert(f.frame);
            }
        }
        r.stats.branch = r.branch;
    }

    pub(crate) fn react_branch(&self) -> Option<Branch> {
        self.react.as_ref().map(|r| r.branch)
    }

    /// SCVs with a teacher job (pulled, repairing): not mining, not picked to build.
    pub(crate) fn react_busy(&self) -> Vec<UnitId> {
        self.react.as_ref().map(|r| r.pulled.iter().chain(&r.repairers).copied().collect()).unwrap_or_default()
    }

    /// Opening supply rule: the first depot at 9 supply, the second at 15, then the standard rule.
    pub(crate) fn react_supply_low(&self, f: &RawFrame, depots: usize) -> Option<bool> {
        let p = &self.react.as_ref()?.p;
        let used = f.self_state.supply_used as u32;
        match depots {
            0 => Some(used >= 2 * p.first_depot_supply),
            1 => Some(used >= 2 * p.second_depot_supply),
            _ => None,
        }
    }

    /// Barracks cap of the branch: 1 in the opening and before the expansion, 2 in D, 3 in M once
    /// the natural is started. The first barracks waits for 11 supply.
    pub(crate) fn react_production(&self, f: &RawFrame, mine: &[&UnitState], prod_count: usize) -> Option<(usize, bool)> {
        let r = self.react.as_ref()?;
        let cap = match r.branch {
            Branch::Defence => 2,
            Branch::Macro if self.natural_started(mine) => 3,
            Branch::Macro => 1,
        };
        let ready = prod_count > 0 || f.self_state.supply_used as u32 >= 2 * r.p.barracks_supply;
        Some((cap, ready))
    }

    fn natural_started(&self, mine: &[&UnitState]) -> bool {
        let Some(n) = self.react.as_ref().and_then(|r| r.natural) else { return false };
        mine.iter().any(|u| is_resource_depot(u.unit_type) && building_tile(u) == n) || self.pending.iter().any(|p| p.ty == CC && p.tile == n)
    }

    /// The scout leaves when the first barracks is under construction (the barracks SCV itself
    /// keeps building, a Terran rule), walks to the nearest unscouted start locations, looks at
    /// the enemy main from `scout_look` tiles and goes back to mining at home.
    pub(crate) fn react_scouting(&mut self, f: &RawFrame, mine: &[&UnitState], minerals: &[&UnitState], cmds: &mut Vec<Command>) {
        let Some(r) = self.react.as_ref() else { return };
        let (look, look_max, main, main_frame) = (r.p.scout_look, r.p.scout_look_max, r.enemy_main, r.stats.main_frame);
        if !self.scout_done && self.scout.is_none() && mine.iter().any(|u| u.unit_type == BARRACKS) {
            self.scout = self.free_worker(mine);
        }
        let Some(id) = self.scout else { return };
        let Some(s) = mine.iter().find(|u| u.id == id) else { return };
        let target = match main {
            Some(m) => {
                let c = depot_center(m);
                let looked = within(s.pos, c, look) || main_frame.is_some_and(|mf| f.frame >= mf + look_max);
                (!looked).then_some(c)
            }
            None => self.starts.iter().zip(&self.scouted).filter(|(_, done)| !**done).map(|(t, _)| depot_center(*t)).min_by_key(|p| (dist2(*p, s.pos), *p)),
        };
        match target {
            Some(p) => {
                if f.frame >= self.scout_order + crate::WAVE_PERIOD || s.has(flags::IDLE) {
                    cmds.push(Command::Unit { units: vec![id], order: CmdOrder::Move(Target::Pos(p)), queued: false });
                    self.scout_order = f.frame;
                }
            }
            None => {
                let home = self.home.unwrap_or(s.pos);
                if let Some(m) = nearest(home, minerals) {
                    cmds.push(Command::Unit { units: vec![id], order: CmdOrder::Gather(m.id), queued: false });
                }
                self.scout = None;
                self.scout_done = true;
            }
        }
    }

    /// Bunker site: the Placer spiral around the point 5 tiles from `center` toward `to`, first
    /// footprint whose centre is `bunker_min`..=`bunker_max` tiles from `center`; any free spot
    /// near that point when none is.
    pub(crate) fn bunker_site(&self, f: &RawFrame, center: Pos, to: Pos, reserved: &[(Tile, u16)]) -> Option<Tile> {
        let p = &self.react.as_ref()?.p;
        let (lo, hi) = (p.bunker_min * TILE, p.bunker_max * TILE);
        let near = toward(center, to, (p.bunker_min + p.bunker_max) as f64 / 2.0);
        self.placer
            .find_where(BUNKER, near, &f.units, reserved, p.bunker_max as i32 + 2, |t| {
                let d2 = dist2(bunker_center(t), center);
                d2 >= lo * lo && d2 <= hi * hi
            })
            .or_else(|| self.placer.find(BUNKER, near, &f.units, reserved))
    }

    /// Where a bunker at `center` faces: the nearest choke tile (for the natural, the nearest one
    /// farther from the main than the natural, so not the main ramp), else the map centre.
    fn bunker_facing(&self, center: Pos, outer_of: Option<Pos>) -> Pos {
        let chokes = self.react.as_ref().map(|r| r.chokes.as_slice()).unwrap_or_default();
        let tile = |t: &Tile| (t.0 * 32 + 16, t.1 * 32 + 16);
        let pick = chokes.iter().map(tile).filter(|c| outer_of.is_none_or(|m| dist2(*c, m) > dist2(center, m))).min_by_key(|c| (dist2(*c, center), *c));
        pick.unwrap_or(self.map_center)
    }

    /// D bunker and M expansion / natural bunker orders, before production buildings. Returns
    /// (minerals held from everything but the first barracks, minerals held from marines).
    pub(crate) fn react_build(&mut self, f: &RawFrame, mine: &[&UnitState], base: Pos, busy: &[UnitId], cmds: &mut Vec<Command>, minerals: &mut i32) -> (i32, i32) {
        let Some(r) = self.react.as_ref() else { return (0, 0) };
        let (p, branch, natural, main_known) = (r.p.clone(), r.branch, r.natural, r.enemy_main.is_some());
        let rax_done = mine.iter().any(|u| u.unit_type == BARRACKS && done(u));
        let bunker_near = |c: Pos| mine.iter().any(|u| u.unit_type == BUNKER && within(u.pos, c, 10)) || self.pending.iter().any(|q| q.ty == BUNKER && within(bunker_center(q.tile), c, 10));
        let reserved = |s: &Self| -> Vec<(Tile, u16)> { s.pending.iter().map(|q| (q.tile, q.ty)).chain(natural.map(|n| (n, CC))).collect() };
        match branch {
            Branch::Defence => {
                if bunker_near(base) {
                    return (0, 0);
                }
                if rax_done && *minerals >= cost(BUNKER) {
                    let to = self.bunker_facing(base, None);
                    let res = reserved(self);
                    if let Some(t) = self.bunker_site(f, base, to, &res) {
                        if self.order_build_at(BUNKER, t, f, mine, busy, cmds) {
                            *minerals -= cost(BUNKER);
                            return (0, 0);
                        }
                    }
                }
                (cost(BUNKER), 0)
            }
            Branch::Macro => {
                let Some(n) = natural else { return (0, 0) };
                if !self.natural_started(mine) {
                    if !rax_done {
                        return (0, 0);
                    }
                    let expand_ok = main_known || f.frame >= p.expand_hold_until;
                    if expand_ok && *minerals >= p.expand_minerals && self.order_build_at(CC, n, f, mine, busy, cmds) {
                        *minerals -= cost(CC);
                        if let Some(r) = self.react.as_mut() {
                            r.stats.expand_frame.get_or_insert(f.frame);
                        }
                        return (0, 0);
                    }
                    return (0, p.expand_minerals);
                }
                let nc = depot_center(n);
                let nat_done = mine.iter().any(|u| is_resource_depot(u.unit_type) && building_tile(u) == n && done(u));
                if nat_done && f.frame >= p.natural_bunker_from && rax_done && !bunker_near(nc) && *minerals >= cost(BUNKER) {
                    let to = self.bunker_facing(nc, self.home);
                    let res = reserved(self);
                    if let Some(t) = self.bunker_site(f, nc, to, &res) {
                        if self.order_build_at(BUNKER, t, f, mine, busy, cmds) {
                            *minerals -= cost(BUNKER);
                        }
                    }
                }
                (0, 0)
            }
        }
    }

    /// Marines kept out of waves and the home guard: in D all of them (they load and hold), in
    /// M the ones picked to fill a bunker.
    pub(crate) fn react_reserved_marines(&self, mine: &[&UnitState]) -> Vec<UnitId> {
        match self.react_branch() {
            None => Vec::new(),
            Some(Branch::Defence) => mine.iter().filter(|u| u.unit_type == MARINE).map(|u| u.id).collect(),
            Some(Branch::Macro) => self.garrison(mine).into_iter().flat_map(|(_, ids)| ids).collect(),
        }
    }

    /// For each completed own bunker with room: the nearest unloaded completed marines to load.
    fn garrison(&self, mine: &[&UnitState]) -> Vec<(UnitId, Vec<UnitId>)> {
        let Some(r) = self.react.as_ref() else { return Vec::new() };
        let mut taken: BTreeSet<UnitId> = BTreeSet::new();
        let mut out = Vec::new();
        let mut bunkers: Vec<&&UnitState> = mine.iter().filter(|u| u.unit_type == BUNKER && done(u)).collect();
        bunkers.sort_by_key(|b| b.id);
        for b in bunkers {
            let loaded = mine.iter().filter(|u| u.loaded_in == Some(b.id)).count();
            let want = r.p.load_max.saturating_sub(loaded);
            if want == 0 {
                continue;
            }
            let mut free: Vec<&&UnitState> = mine.iter().filter(|u| u.unit_type == MARINE && u.has(flags::COMPLETED) && u.loaded_in.is_none() && !taken.contains(&u.id)).collect();
            free.sort_by_key(|u| (dist2(u.pos, b.pos), u.id));
            let ids: Vec<UnitId> = free.iter().take(want).map(|u| u.id).collect();
            taken.extend(&ids);
            if !ids.is_empty() {
                out.push((b.id, ids));
            }
        }
        out
    }

    /// Marine control after the Brain's waves and guard: bunker garrisons (both branches), and in
    /// D the marines that do not load hold beside the bunker.
    pub(crate) fn react_marines(&mut self, f: &RawFrame, mine: &[&UnitState], cmds: &mut Vec<Command>) {
        let Some(r) = self.react.as_mut() else { return };
        r.holding.retain(|id| mine.iter().any(|u| u.id == *id && u.loaded_in.is_none()));
        let (every, branch, last) = (r.p.retarget, r.branch, r.last_marines);
        if !ReactState::due(last, f.frame, every) {
            return;
        }
        let mut acted = false;
        for (bunker, ids) in self.garrison(mine) {
            cmds.push(Command::Unit { units: ids, order: CmdOrder::Load(bunker), queued: false });
            acted = true;
        }
        if branch == Branch::Defence {
            acted |= self.hold_marines(mine, cmds);
        }
        if acted {
            self.react.as_mut().expect("react kit").last_marines = Some(f.frame);
        }
    }

    /// SCV jobs, before the Brain's idle-worker rule so a pulled or repairing SCV is not sent to
    /// mine in the same step: in D the pull (③), re-targeting, release (④) and repair; in M the
    /// release of every D job.
    pub(crate) fn react_scvs(&mut self, f: &RawFrame, mine: &[&UnitState], minerals: &[&UnitState], cmds: &mut Vec<Command>) {
        let Some(r) = self.react.as_ref() else { return };
        let (p, branch, last_s) = (r.p.clone(), r.branch, r.last_scvs);
        let home = self.home_depot(mine).or(self.home);
        let gather_home = |ids: Vec<UnitId>, cmds: &mut Vec<Command>| {
            if ids.is_empty() {
                return;
            }
            if let Some(m) = home.and_then(|h| nearest(h, minerals)) {
                for chunk in ids.chunks(12) {
                    cmds.push(Command::Unit { units: chunk.to_vec(), order: CmdOrder::Gather(m.id), queued: false });
                }
            }
        };
        let alive = |id: &UnitId| mine.iter().any(|u| u.id == *id);
        let r = self.react.as_mut().expect("checked above");
        r.pulled.retain(alive);
        r.repairers.retain(alive);

        let r = self.react.as_ref().expect("checked above");
        if branch == Branch::Macro {
            let ids: Vec<UnitId> = r.pulled.iter().chain(&r.repairers).copied().collect();
            gather_home(ids, cmds);
            let r = self.react.as_mut().expect("checked above");
            r.pulled.clear();
            r.repairers.clear();
            return;
        }
        if !ReactState::due(last_s, f.frame, p.retarget) {
            return;
        }
        let mut acted = false;
        // SCV pull (D ③): enemy ground army within `pull_radius` of a command center.
        let me = self.me;
        let depots = Self::own_depots(mine);
        let ground: Vec<&UnitState> = f.units.iter().filter(|u| is_enemy(u, me) && is_ground_army_type(u.unit_type) && !u.has(flags::BURROWED)).collect();
        let near = |u: &&UnitState, tiles: i64| depots.iter().any(|d| within(u.pos, *d, tiles));
        let threats: Vec<&UnitState> = ground.iter().copied().filter(|u| near(u, p.pull_radius)).collect();
        let targets: Vec<&UnitState> = ground.iter().copied().filter(|u| near(u, p.release_radius)).collect();
        let scvs: Vec<&&UnitState> = mine.iter().filter(|u| u.unit_type == SCV && u.has(flags::COMPLETED)).collect();
        let (last_near, mut pulled, repairers) = {
            let r = self.react.as_ref().expect("checked above");
            (r.last_near, r.pulled.clone(), r.repairers.clone())
        };
        if !threats.is_empty() {
            let want = p.pull_max.min(scvs.len().saturating_sub(p.pull_keep));
            let at = threats.iter().copied().min_by_key(|u| (depots.iter().map(|d| dist2(u.pos, *d)).min().unwrap_or(0), u.id)).map(|u| u.pos).unwrap_or_default();
            let jobs: Vec<UnitId> = self.pending.iter().map(|q| q.worker).chain(self.scout).chain(repairers.iter().copied()).collect();
            let mut free: Vec<&&&UnitState> = scvs.iter().filter(|u| !u.has(flags::CONSTRUCTING) && !jobs.contains(&u.id) && !pulled.contains(&u.id)).collect();
            free.sort_by_key(|u| (dist2(u.pos, at), u.id));
            for u in free.into_iter().take(want.saturating_sub(pulled.len())) {
                pulled.push(u.id);
            }
            if pulled.len() > want {
                let extra = pulled.split_off(want);
                gather_home(extra, cmds);
            }
        }
        if !pulled.is_empty() {
            if f.frame >= last_near + p.release_after {
                // ④ no enemy army within `release_radius` for `release_after`: back to mining.
                gather_home(std::mem::take(&mut pulled), cmds);
                acted = true;
            } else {
                let group: Vec<&UnitState> = pulled.iter().filter_map(|id| mine.iter().copied().find(|u| u.id == *id)).collect();
                let c = crate::centroid(&group);
                if let Some(t) = targets.iter().min_by_key(|u| (dist2(u.pos, c), u.id)) {
                    cmds.push(Command::Unit { units: pulled.clone(), order: CmdOrder::Attack(Target::Unit(t.id)), queued: false });
                    acted = true;
                }
            }
        }
        // Repair (D): a completed bunker below `repair_below_pct` gets `repairers` SCVs until full.
        let mut repairers = repairers;
        let mut bunkers: Vec<&&UnitState> = mine.iter().filter(|u| u.unit_type == BUNKER && done(u) && u.hp < BUNKER_HP).collect();
        bunkers.sort_by_key(|b| (b.hp, b.id));
        let damaged = bunkers.iter().find(|b| (b.hp as u32) * 100 < BUNKER_HP as u32 * p.repair_below_pct).or(if repairers.is_empty() { None } else { bunkers.first() });
        match damaged {
            Some(b) => {
                let jobs: Vec<UnitId> = self.pending.iter().map(|q| q.worker).chain(self.scout).chain(pulled.iter().copied()).collect();
                let mut free: Vec<&&&UnitState> = scvs.iter().filter(|u| !u.has(flags::CONSTRUCTING) && !jobs.contains(&u.id) && !repairers.contains(&u.id)).collect();
                free.sort_by_key(|u| (dist2(u.pos, b.pos), u.id));
                for u in free.into_iter().take(p.repairers.saturating_sub(repairers.len())) {
                    repairers.push(u.id);
                }
                if !repairers.is_empty() {
                    cmds.push(Command::Unit { units: repairers.clone(), order: CmdOrder::Repair(b.id), queued: false });
                    acted = true;
                }
            }
            None => {
                if !repairers.is_empty() {
                    gather_home(std::mem::take(&mut repairers), cmds);
                    acted = true;
                }
            }
        }
        let r = self.react.as_mut().expect("checked above");
        r.pulled = pulled;
        r.repairers = repairers;
        if acted {
            r.last_scvs = Some(f.frame);
        }
    }

    /// D ②: marines not picked to load walk beside the bunker (2 tiles behind it, toward the main)
    /// and hold position there. Returns whether a command was issued.
    fn hold_marines(&mut self, mine: &[&UnitState], cmds: &mut Vec<Command>) -> bool {
        let Some(home) = self.home_depot(mine).or(self.home) else { return false };
        let loading: BTreeSet<UnitId> = self.garrison(mine).into_iter().flat_map(|(_, ids)| ids).collect();
        let bunker = mine.iter().filter(|u| u.unit_type == BUNKER && within(u.pos, home, 10)).min_by_key(|u| (dist2(u.pos, home), u.id)).map(|u| u.pos);
        let planned = self.pending.iter().find(|q| q.ty == BUNKER && within(bunker_center(q.tile), home, 10)).map(|q| bunker_center(q.tile));
        let spot = match bunker.or(planned) {
            Some(b) => toward(b, home, 2.0),
            None => toward(home, self.bunker_facing(home, None), 3.0),
        };
        let r = self.react.as_mut().expect("react kit");
        let marines: Vec<&&UnitState> = mine.iter().filter(|u| u.unit_type == MARINE && u.has(flags::COMPLETED) && u.loaded_in.is_none() && !loading.contains(&u.id)).collect();
        let (mut walk, mut hold) = (Vec::new(), Vec::new());
        for m in marines {
            if within(m.pos, spot, 3) {
                if r.holding.insert(m.id) {
                    hold.push(m.id);
                }
            } else {
                r.holding.remove(&m.id);
                walk.push(m.id);
            }
        }
        for chunk in walk.chunks(12) {
            cmds.push(Command::Unit { units: chunk.to_vec(), order: CmdOrder::Move(Target::Pos(spot)), queued: false });
        }
        for chunk in hold.chunks(12) {
            cmds.push(Command::Unit { units: chunk.to_vec(), order: CmdOrder::Hold, queued: false });
        }
        !(walk.is_empty() && hold.is_empty())
    }

    /// Orders `ty` at `tile` with the nearest free SCV (not carrying, not constructing, no job).
    pub(crate) fn order_build_at(&mut self, ty: u16, tile: Tile, f: &RawFrame, mine: &[&UnitState], busy: &[UnitId], cmds: &mut Vec<Command>) -> bool {
        let center = (tile.0 * 32 + 16, tile.1 * 32 + 16);
        let jobs: Vec<UnitId> = self.react_busy().into_iter().chain(self.pending.iter().map(|q| q.worker)).collect();
        let worker = mine
            .iter()
            .filter(|u| u.unit_type == SCV && u.has(flags::COMPLETED) && !u.has(flags::CARRYING) && !u.has(flags::CONSTRUCTING) && !busy.contains(&u.id) && !jobs.contains(&u.id))
            .min_by_key(|u| (dist2(u.pos, center), u.id));
        let Some(w) = worker else { return false };
        cmds.push(Command::Build { worker: w.id, unit_type: ty, tile });
        self.pending.push(crate::Pending { ty, tile, worker: w.id, issued: f.frame });
        true
    }

    /// 050-R8 session record part of the teacher.
    pub(crate) fn teacher_stats(&self) -> Option<ReactStats> {
        self.react.as_ref().map(|r| r.stats.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Bot, Kit, T0Build};

    /// Our start (10, 10), the enemy start (110, 110); the natural base at (10, 26), the main
    /// choke at (24, 10) and a choke beyond the natural at (10, 40).
    const OURS: Tile = (10, 10);
    const THEIRS: Tile = (110, 110);
    const NATURAL: Tile = (10, 26);

    fn info() -> GameInfo {
        let n = 128usize * 128;
        GameInfo {
            map_name: "t".into(),
            map_hash: [0; 32],
            map_size_tiles: (128, 128),
            self_player: 0,
            players: vec![
                PlayerInfo { id: 0, name: "a".into(), race: Race::Terran, start_tile: Some(OURS), is_self: true, is_enemy: false },
                PlayerInfo { id: 1, name: "b".into(), race: Race::Zerg, start_tile: None, is_self: false, is_enemy: true },
            ],
            static_map: StaticMap {
                width_tiles: 128,
                height_tiles: 128,
                walkable: BitGrid::from_bytes(512, 512, &vec![1; n * 16]),
                buildable: BitGrid::from_bytes(128, 128, &vec![1; n]),
                ground_height: vec![0; n],
                start_locations: vec![OURS, THEIRS],
            },
            latency_frames: 0,
        }
    }

    fn hints() -> MapHints {
        MapHints { base_locs: vec![OURS, THEIRS, NATURAL, (60, 60)], chokes: vec![(24, 10), (10, 40), (100, 100)] }
    }

    fn teacher() -> Brain {
        Brain::with_hints(&info(), Kit::for_build(Race::Terran, T0Build::TerranReact), &hints())
    }

    fn unit(id: u32, ty: u16, pos: Pos) -> UnitState {
        UnitState { id, owner: 0, unit_type: ty, pos, hp: 40, flags: flags::COMPLETED | flags::IDLE, ..Default::default() }
    }

    fn enemy(id: u32, ty: u16, pos: Pos) -> UnitState {
        UnitState { owner: 1, ..unit(id, ty, pos) }
    }

    fn home() -> Pos {
        depot_center(OURS)
    }

    /// CC, two depots, a completed barracks, a mineral field and `scvs` SCVs at home.
    fn base(scvs: u32) -> Vec<UnitState> {
        let mut v = vec![
            unit(1, CC, home()),
            unit(2, BARRACKS, (home().0 + 200, home().1 + 160)),
            UnitState { owner: 11, ..unit(3, 176, (home().0 - 200, home().1)) },
            unit(6, DEPOT, (home().0 - 48, home().1 - 160)),
            unit(7, DEPOT, (home().0 + 80, home().1 - 160)),
        ];
        v.extend((0..scvs).map(|i| unit(100 + i, SCV, (home().0 - 100, home().1 + 10 * i as u16))));
        v
    }

    fn frame(n: u32, units: Vec<UnitState>, minerals: i32, supply_used: u16) -> RawFrame {
        RawFrame {
            frame: n,
            self_state: PlayerState { minerals, supply_used, supply_total: 60, upgrades: vec![0; 63], ..Default::default() },
            units,
            visibility: BitGrid::from_bytes(128, 128, &[1; 128 * 128]),
            explored: BitGrid::new(128, 128),
            creep: BitGrid::new(128, 128),
            events: vec![],
            effects: vec![],
        }
    }

    fn builds(act: &Act, ty: u16) -> Vec<Tile> {
        act.cmds
            .iter()
            .filter_map(|c| match c {
                Command::Build { unit_type, tile, .. } if *unit_type == ty => Some(*tile),
                _ => None,
            })
            .collect()
    }

    fn trains(act: &Act, ty: u16) -> usize {
        act.cmds.iter().filter(|c| matches!(c, Command::Train { unit_type, .. } if *unit_type == ty)).count()
    }

    fn unit_orders(act: &Act, pick: impl Fn(&CmdOrder) -> bool) -> Vec<Vec<UnitId>> {
        act.cmds
            .iter()
            .filter_map(|c| match c {
                Command::Unit { units, order, .. } if pick(order) => Some(units.clone()),
                _ => None,
            })
            .collect()
    }

    fn branch(b: &Brain) -> Branch {
        b.react_branch().unwrap()
    }

    /// 021-R8 R ①–④ read the current observation only, with their time limits.
    #[test]
    fn rush_rules_021_r8() {
        let p = ReactParams::default();
        let depots = [home()];
        let others = [THEIRS];
        let at = |n: u32, units: Vec<UnitState>| rush_rule(&p, &frame(n, units, 0, 0), 0, &depots, &others);
        let far = depot_center((60, 60));
        // ① pool before 2:30 only.
        assert_eq!(at(3_599, vec![enemy(9, SPAWNING_POOL, far)]), Some(RushRule::Pool));
        assert_eq!(at(3_600, vec![enemy(9, SPAWNING_POOL, far)]), None);
        // ② two zerglings in one observation before 3:30; one is not enough.
        assert_eq!(at(5_039, vec![enemy(9, ZERGLING, far), enemy(10, ZERGLING, far)]), Some(RushRule::Lings));
        assert_eq!(at(5_039, vec![enemy(9, ZERGLING, far)]), None);
        assert_eq!(at(5_040, vec![enemy(9, ZERGLING, far), enemy(10, ZERGLING, far)]), None);
        // ③ enemy army (033) within 20 tiles of a command center, until 8:00; an overlord or a
        // drone is no army.
        let near = (home().0 + 19 * 32, home().1);
        assert_eq!(at(9_000, vec![enemy(9, ZERGLING, near)]), Some(RushRule::Army));
        assert_eq!(at(9_000, vec![enemy(9, ZERGLING, (home().0 + 21 * 32, home().1))]), None);
        assert_eq!(at(9_000, vec![enemy(9, OVERLORD, near), enemy(10, DRONE, near)]), None);
        assert_eq!(at(11_520, vec![enemy(9, ZERGLING, near)]), None);
        // ④ enemy main seen with its depot, a pool and at most 8 drones, before 3:30.
        let main = depot_center(THEIRS);
        let mut v = vec![enemy(9, HATCHERY, main), enemy(10, SPAWNING_POOL, (main.0 + 160, main.1))];
        v.extend((0..8).map(|i| enemy(20 + i, DRONE, (main.0 - 100, main.1 + i as u16 * 8))));
        assert_eq!(at(4_000, v.clone()), Some(RushRule::Main));
        v.push(enemy(40, DRONE, (main.0 - 100, main.1 + 90)));
        assert_eq!(at(4_000, v.clone()), None, "9 drones");
        v.truncate(10);
        assert_eq!(at(5_040, v.clone()), None, "after 3:30");
        assert_eq!(at(4_000, vec![enemy(9, HATCHERY, main)]), None, "no pool");
    }

    /// 021-R8: M is the start; R switches to D at that observation (bunker toward the main
    /// choke); 8 marines and 60 s without contact switch back to M; a new contact switches to D
    /// again. The record keeps the first identification.
    #[test]
    fn branch_switches_021_r8() {
        let mut b = teacher();
        b.act(&frame(2_000, base(12), 0, 24));
        assert_eq!(branch(&b), Branch::Macro);
        let mut units = base(12);
        units.push(enemy(90, ZERGLING, depot_center((60, 60))));
        units.push(enemy(91, ZERGLING, depot_center((60, 60))));
        let act = b.act(&frame(2_008, units, 100, 24));
        assert_eq!(branch(&b), Branch::Defence);
        let bunker = builds(&act, BUNKER);
        assert_eq!(bunker.len(), 1, "{:?}", act.cmds);
        // 4–6 tiles from the command center, on the choke side (east).
        let c = bunker_center(bunker[0]);
        let d = ((dist2(c, home()) as f64).sqrt() / 32.0, c.0 as i64 - home().0 as i64);
        assert!(d.0 >= 4.0 && d.0 <= 6.0 && d.1 > 0, "{bunker:?} {d:?}");
        // Seven marines: still D. Eight marines, 60 s after the last contact: M.
        let marines = |n: u32| (0..n).map(|i| unit(200 + i, MARINE, (home().0 + 100, home().1 + 4 * i as u16))).collect::<Vec<_>>();
        let mut u = base(12);
        u.extend(marines(7));
        b.act(&frame(2_008 + 1_400, u, 0, 40));
        assert_eq!(branch(&b), Branch::Defence);
        let mut u = base(12);
        u.extend(marines(8));
        b.act(&frame(2_008 + 1_439, u.clone(), 0, 40));
        assert_eq!(branch(&b), Branch::Defence);
        b.act(&frame(2_008 + 1_440, u.clone(), 0, 40));
        assert_eq!(branch(&b), Branch::Macro);
        // Contact (③) again: D.
        u.push(enemy(92, ZERGLING, (home().0 + 300, home().1)));
        b.act(&frame(5_000, u, 0, 40));
        assert_eq!(branch(&b), Branch::Defence);
        let s = b.teacher_stats().unwrap();
        assert_eq!((s.rush_frame, s.rush_rule, s.defences, s.macro_frame, s.branch), (Some(2_008), Some(RushRule::Lings), 2, Some(3_448), Branch::Defence));
    }

    /// 021-R8 M: with the enemy main found and no R, the natural command center at 400 minerals;
    /// marines leave the expansion's price alone until then. Without the main found it waits
    /// until 3:40 and builds no bunker meanwhile.
    #[test]
    fn macro_expands_to_the_natural_021_r8() {
        // Main found (an enemy building next to the enemy start), 350 minerals: no CC, and no
        // marine either (the 400 are held from marines).
        let mut b = teacher();
        let mut units = base(14);
        units.push(enemy(80, HATCHERY, depot_center(THEIRS)));
        let act = b.act(&frame(3_000, units.clone(), 350, 30));
        assert!(b.react.as_ref().unwrap().enemy_main.is_some());
        assert_eq!((builds(&act, CC).len(), trains(&act, MARINE)), (0, 0), "{:?}", act.cmds);
        let units: Vec<UnitState> = units.into_iter().filter(|u| u.owner == 0).collect();
        let act = b.act(&frame(3_008, units, 400, 30));
        assert_eq!(builds(&act, CC), vec![NATURAL], "{:?}", act.cmds);
        assert_eq!(b.teacher_stats().unwrap().expand_frame, Some(3_008));

        // Main not found: no expansion before 3:40, and no bunker meanwhile; at 3:40 it expands.
        let mut b = teacher();
        let act = b.act(&frame(5_279, base(14), 600, 30));
        assert_eq!((builds(&act, CC).len(), builds(&act, BUNKER).len()), (0, 0), "{:?}", act.cmds);
        let act = b.act(&frame(5_280, base(14), 600, 30));
        assert_eq!(builds(&act, CC), vec![NATURAL], "{:?}", act.cmds);
    }

    /// 021-R8 M: the natural bunker only once the natural command center is complete and from
    /// 4:00 (the user constraint: no pre-emptive bunker), facing away from the main.
    #[test]
    fn natural_bunker_not_before_four_minutes_021_r8() {
        let mut units = base(16);
        units.push(unit(4, CC, depot_center(NATURAL)));
        let mut b = teacher();
        let act = b.act(&frame(5_759, units.clone(), 500, 40));
        assert!(builds(&act, BUNKER).is_empty(), "{:?}", act.cmds);
        let act = b.act(&frame(5_760, units.clone(), 500, 40));
        let t = builds(&act, BUNKER);
        assert_eq!(t.len(), 1, "{:?}", act.cmds);
        // Toward the outer choke (10, 40): south of the natural.
        assert!(bunker_center(t[0]).1 > depot_center(NATURAL).1, "{t:?}");
        // The natural command center under construction: no bunker.
        let mut b = teacher();
        let last = units.len() - 1;
        units[last].flags = flags::BEING_CONSTRUCTED;
        let act = b.act(&frame(6_000, units, 500, 40));
        assert!(builds(&act, BUNKER).is_empty(), "{:?}", act.cmds);
    }

    /// 021-R8 D ②: up to 4 marines load into the completed bunker, the rest walk beside it and
    /// hold there; no attack waves in D.
    #[test]
    fn defence_loads_four_and_holds_the_rest_021_r8() {
        let mut b = teacher();
        let mut units = base(12);
        let bunker = (home().0 + 5 * 32, home().1);
        units.push(UnitState { hp: BUNKER_HP, ..unit(5, BUNKER, bunker) });
        units.extend((0..6).map(|i| unit(200 + i, MARINE, (home().0 + 40, home().1 + 100 + 8 * i as u16))));
        units.push(enemy(90, ZERGLING, depot_center((60, 60))));
        units.push(enemy(91, ZERGLING, depot_center((60, 60))));
        let act = b.act(&frame(3_000, units.clone(), 0, 40));
        assert_eq!(branch(&b), Branch::Defence);
        let load = unit_orders(&act, |o| *o == CmdOrder::Load(5));
        assert_eq!(load.iter().map(Vec::len).collect::<Vec<_>>(), vec![4], "{:?}", act.cmds);
        let walk: usize = unit_orders(&act, |o| matches!(o, CmdOrder::Move(_))).iter().map(Vec::len).sum();
        assert_eq!(walk, 2, "{:?}", act.cmds);
        // Two loaded already, the others beside the bunker: two more load, the rest hold.
        let spot = toward(bunker, home(), 2.0);
        for u in units.iter_mut().filter(|u| u.unit_type == MARINE) {
            u.pos = spot;
        }
        units.iter_mut().filter(|u| u.unit_type == MARINE).take(2).for_each(|u| u.loaded_in = Some(5));
        let act = b.act(&frame(3_048, units, 0, 40));
        let loads: usize = unit_orders(&act, |o| *o == CmdOrder::Load(5)).iter().map(Vec::len).sum();
        let holds: usize = unit_orders(&act, |o| *o == CmdOrder::Hold).iter().map(Vec::len).sum();
        assert_eq!((loads, holds), (2, 2), "{:?}", act.cmds);
        assert!(unit_orders(&act, |o| matches!(o, CmdOrder::Attack(Target::Pos(_)))).is_empty(), "no waves in D: {:?}", act.cmds);
    }

    /// 021-R8 D ③ / ④: enemy army within 12 tiles pulls min(8, n − 4) SCVs onto the nearest
    /// zergling; a bunker below 90% gets 2 repairers until full; 10 s without enemy army within
    /// 16 tiles sends the pulled SCVs back to mining.
    #[test]
    fn defence_pulls_repairs_and_releases_scvs_021_r8() {
        let mut b = teacher();
        // No scout trip in this test (it would mine again at once: every start is visible).
        b.scout_done = true;
        let mut units = base(10);
        units.push(enemy(90, ZERGLING, (home().0 + 8 * 32, home().1)));
        units.push(enemy(91, ZERGLING, (home().0 + 11 * 32, home().1 + 64)));
        units.push(UnitState { hp: 300, ..unit(5, BUNKER, (home().0 + 5 * 32, home().1 + 64)) });
        let act = b.act(&frame(3_000, units.clone(), 0, 40));
        let attack: Vec<(Vec<UnitId>, UnitId)> = act
            .cmds
            .iter()
            .filter_map(|c| match c {
                Command::Unit { units, order: CmdOrder::Attack(Target::Unit(t)), .. } => Some((units.clone(), *t)),
                _ => None,
            })
            .collect();
        assert_eq!(attack.len(), 1, "{:?}", act.cmds);
        assert_eq!(attack[0].0.len(), 6, "min(8, 10 - 4)");
        assert!(attack[0].1 == 90 || attack[0].1 == 91);
        let repair = unit_orders(&act, |o| *o == CmdOrder::Repair(5));
        assert_eq!(repair.iter().map(Vec::len).collect::<Vec<_>>(), vec![2], "{:?}", act.cmds);
        assert!(repair[0].iter().all(|id| !attack[0].0.contains(id)));
        // Pulled SCVs are not sent back to mining by the idle rule.
        let pulled = attack[0].0.clone();
        assert!(unit_orders(&act, |o| matches!(o, CmdOrder::Gather(_))).iter().all(|g| g.iter().all(|u| !pulled.contains(u))));
        // Enemies gone (still D: no 60 s quiet yet), bunker full: the repairers mine again; 10 s
        // after the last enemy army within 16 tiles the pulled SCVs do too.
        let quiet: Vec<UnitState> = units.iter().filter(|u| u.owner != 1).cloned().map(|u| if u.unit_type == BUNKER { UnitState { hp: BUNKER_HP, ..u } } else { u }).collect();
        let act = b.act(&frame(3_048, quiet.clone(), 0, 40));
        assert!(unit_orders(&act, |o| matches!(o, CmdOrder::Gather(_))).iter().any(|g| *g == repair[0]), "{:?}", act.cmds);
        let act = b.act(&frame(3_000 + 239, quiet.clone(), 0, 40));
        assert!(!unit_orders(&act, |o| matches!(o, CmdOrder::Gather(_))).contains(&pulled), "{:?}", act.cmds);
        let act = b.act(&frame(3_000 + 240, quiet, 0, 40));
        assert!(unit_orders(&act, |o| matches!(o, CmdOrder::Gather(_))).contains(&pulled), "{:?}", act.cmds);
        assert_eq!(branch(&b), Branch::Defence);
        assert!(b.react_busy().is_empty());
    }

    /// 021-R8 opening: first depot at 9 supply, barracks at 11, second depot at 15; the scout
    /// leaves once the barracks is under construction.
    #[test]
    fn opening_and_scout_021_r8() {
        let cc_only = |scvs: u32| {
            let mut v = vec![unit(1, CC, home()), UnitState { owner: 11, ..unit(3, 176, (home().0 - 200, home().1)) }];
            v.extend((0..scvs).map(|i| unit(100 + i, SCV, (home().0 - 100, home().1 + 10 * i as u16))));
            v
        };
        let act = teacher().act(&frame(500, cc_only(8), 200, 16));
        assert!(builds(&act, DEPOT).is_empty(), "8 supply: {:?}", act.cmds);
        let act = teacher().act(&frame(500, cc_only(9), 200, 18));
        assert_eq!(builds(&act, DEPOT).len(), 1, "9 supply");
        let mut units = cc_only(10);
        units.push(unit(6, DEPOT, (home().0, home().1 - 160)));
        let act = teacher().act(&frame(1_500, units.clone(), 200, 20));
        assert!(builds(&act, BARRACKS).is_empty(), "10 supply: {:?}", act.cmds);
        let mut b = teacher();
        // Fog everywhere so the enemy start stays unscouted.
        let dark = |f: RawFrame| RawFrame { visibility: BitGrid::new(128, 128), ..f };
        let act = b.act(&dark(frame(1_500, units.clone(), 200, 22)));
        assert_eq!(builds(&act, BARRACKS).len(), 1, "11 supply");
        assert!(b.scout.is_none(), "the barracks is only ordered");
        // Barracks under construction: one SCV scouts toward the enemy start.
        units.push(UnitState { flags: flags::BEING_CONSTRUCTED, ..unit(2, BARRACKS, (home().0 + 200, home().1 + 160)) });
        let act = b.act(&dark(frame(1_600, units.clone(), 0, 24)));
        let scout = b.scout.expect("scout");
        assert!(act.cmds.contains(&Command::Unit { units: vec![scout], order: CmdOrder::Move(Target::Pos(depot_center(THEIRS))), queued: false }), "{:?}", act.cmds);
        let act = teacher().act(&frame(1_600, units.clone(), 200, 28));
        assert!(builds(&act, DEPOT).is_empty(), "14 supply: {:?}", act.cmds);
        let act = teacher().act(&frame(1_600, units, 200, 30));
        assert_eq!(builds(&act, DEPOT).len(), 1, "15 supply");
    }

    /// 021-R8 determinism: the same observations give the same commands.
    #[test]
    fn same_observations_same_commands_021_r8() {
        let script = |b: &mut Brain| {
            let mut out = Vec::new();
            let lings = vec![enemy(90, ZERGLING, (home().0 + 300, home().1)), enemy(91, ZERGLING, (home().0 + 330, home().1))];
            for (i, extra) in [vec![], lings, vec![]].into_iter().enumerate() {
                let mut u = base(12);
                u.extend(extra);
                u.extend((0..5).map(|j| unit(200 + j, MARINE, (home().0 + 50, home().1 + 6 * j as u16))));
                out.push(b.act(&frame(3_000 + 48 * i as u32, u, 300, 40)).cmds);
            }
            out
        };
        let (a, c) = (script(&mut teacher()), script(&mut teacher()));
        assert_eq!(a, c);
        assert!(a.iter().flatten().any(|c| matches!(c, Command::Unit { order: CmdOrder::Attack(Target::Unit(_)), .. })));
    }

    /// 021-R3 / 021-R8: `t0-terran-react`, plan name `terran_react`, Terran only; another race
    /// falls back to its standard build.
    #[test]
    fn build_id_021_r8() {
        let t = T0Build::TerranReact;
        assert_eq!((t.name(), t.id_tag(), t.race(), t.needs_map_hints()), ("terran_react", Some("react"), Some(Race::Terran), true));
        assert!(Kit::for_build(Race::Terran, t).react.is_some());
        assert!(Kit::for_build(Race::Zerg, t).react.is_none());
        assert!(Kit::terran().react.is_none() && Kit::terran_drop().react.is_none() && !T0Build::TerranDrop.needs_map_hints());
    }
}
