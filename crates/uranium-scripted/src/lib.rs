//! T0 rule-based opponents (spec 021). They emit spec-010 `Command`s directly (021-R6).
//!
//! M2 ships one Terran bot whose only job is to play a full melee game to the end
//! (011-AC1): workers, depots, three barracks, marines, then attack-move waves that
//! hunt down every known enemy building.

pub mod placement;
pub mod types;

use std::collections::BTreeMap;

use uranium_proto::*;

use crate::placement::Placer;
use crate::types::*;

const STEP_FRAMES: u32 = 8;
const MAX_SCVS: usize = 22;
const FIRST_WAVE: usize = 10;
const WAVE_PERIOD: u32 = 48;
const PENDING_TIMEOUT: u32 = 24 * 25;

struct Pending {
    ty: u16,
    tile: Tile,
    worker: UnitId,
    issued: u32,
}

pub struct TerranRush {
    me: u8,
    my_start: Option<Tile>,
    starts: Vec<Tile>,
    placer: Placer,
    pending: Vec<Pending>,
    /// Last seen enemy building positions (memory: 030-R2 style, own vision only).
    enemy_buildings: BTreeMap<UnitId, Pos>,
    scouted: Vec<bool>,
    sweep: Vec<Pos>,
    sweep_i: usize,
    last_wave: u32,
}

impl TerranRush {
    pub fn new(info: &GameInfo) -> Self {
        let me = info.self_player;
        let my_start = info.players.iter().find(|p| p.is_self).and_then(|p| p.start_tile);
        let starts = info.static_map.start_locations.clone();
        let scouted = starts.iter().map(|s| Some(*s) == my_start).collect();
        let (w, h) = info.map_size_tiles;
        let sweep = (0..h / 16)
            .flat_map(|y| (0..w / 16).map(move |x| ((x * 16 + 8) * 32, (y * 16 + 8) * 32)))
            .filter(|&(x, y)| info.static_map.walkable.get(x / 8, y / 8))
            .collect();
        Self {
            me,
            my_start,
            starts,
            placer: Placer::new(&info.static_map),
            pending: Vec::new(),
            enemy_buildings: BTreeMap::new(),
            scouted,
            sweep,
            sweep_i: 0,
            last_wave: 0,
        }
    }

    pub fn act(&mut self, f: &RawFrame) -> Act {
        let mut cmds = Vec::new();
        let mine: Vec<&UnitState> = f.units.iter().filter(|u| u.owner == self.me).collect();
        let count = |ty: u16| mine.iter().filter(|u| u.unit_type == ty).count();
        let minerals_vis: Vec<&UnitState> = f.units.iter().filter(|u| is_mineral(u.unit_type)).collect();
        self.update_memory(f);
        self.pending.retain(|p| {
            f.frame < p.issued + PENDING_TIMEOUT
                && !mine.iter().any(|u| u.unit_type == p.ty && building_tile(u) == p.tile)
        });

        let mut minerals = f.self_state.minerals - self.pending.iter().map(|p| cost(p.ty)).sum::<i32>();
        let supply_free = f.self_state.supply_total as i32 - f.self_state.supply_used as i32;
        let base = mine.iter().find(|u| u.unit_type == CC).map(|u| u.pos);

        // Workers: idle SCVs mine (or return cargo).
        let busy: Vec<UnitId> = self.pending.iter().map(|p| p.worker).collect();
        for scv in mine.iter().filter(|u| u.unit_type == SCV && u.has(flags::COMPLETED) && u.has(flags::IDLE)) {
            if busy.contains(&scv.id) {
                continue;
            }
            if scv.has(flags::CARRYING) {
                cmds.push(Command::Unit { units: vec![scv.id], order: CmdOrder::ReturnCargo, queued: false });
            } else if let Some(m) = nearest(scv.pos, &minerals_vis) {
                cmds.push(Command::Unit { units: vec![scv.id], order: CmdOrder::Gather(m.id), queued: false });
            }
        }

        // Production: SCVs, then marines.
        let scvs = count(SCV);
        for cc in mine.iter().filter(|u| u.unit_type == CC && u.has(flags::COMPLETED) && !u.has(flags::TRAINING)) {
            if scvs < MAX_SCVS && minerals >= 50 && supply_free >= 2 {
                cmds.push(Command::Train { building: cc.id, unit_type: SCV });
                minerals -= 50;
            }
        }

        // Buildings: supply depot when close to cap, barracks up to 3.
        let pending_of = |ty: u16, p: &[Pending]| p.iter().filter(|x| x.ty == ty).count();
        let depots_building = mine.iter().filter(|u| u.unit_type == DEPOT && !u.has(flags::COMPLETED)).count();
        let barracks = count(BARRACKS) + pending_of(BARRACKS, &self.pending);
        let want_depot = f.self_state.supply_total < 400
            && supply_free <= 4 + 4 * barracks as i32
            && pending_of(DEPOT, &self.pending) + depots_building == 0;
        let want_barracks = (scvs >= 10 && barracks < 1) || (scvs >= 14 && barracks < 3);
        for (ty, want) in [(DEPOT, want_depot), (BARRACKS, want_barracks)] {
            if !want || minerals < cost(ty) {
                continue;
            }
            let Some(base) = base else { continue };
            let reserved: Vec<(Tile, u16)> = self.pending.iter().map(|p| (p.tile, p.ty)).collect();
            let Some(tile) = self.placer.find(ty, base, &f.units, &reserved) else { continue };
            let center = ((tile.0 * 32 + 16) as u16, (tile.1 * 32 + 16) as u16);
            let worker = mine
                .iter()
                .filter(|u| u.unit_type == SCV && u.has(flags::COMPLETED) && !u.has(flags::CARRYING) && !busy.contains(&u.id))
                .min_by_key(|u| dist2(u.pos, center));
            if let Some(w) = worker {
                cmds.push(Command::Build { worker: w.id, unit_type: ty, tile });
                self.pending.push(Pending { ty, tile, worker: w.id, issued: f.frame });
                minerals -= cost(ty);
            }
        }
        for rax in mine.iter().filter(|u| u.unit_type == BARRACKS && u.has(flags::COMPLETED) && !u.has(flags::TRAINING)) {
            if minerals >= 50 && supply_free >= 2 {
                cmds.push(Command::Train { building: rax.id, unit_type: MARINE });
                minerals -= 50;
            }
        }

        // Army: attack-move waves.
        let army: Vec<UnitId> = mine.iter().filter(|u| u.unit_type == MARINE && u.has(flags::COMPLETED)).map(|u| u.id).collect();
        if army.len() >= FIRST_WAVE && f.frame >= self.last_wave + WAVE_PERIOD {
            if let Some(target) = self.attack_target(base.unwrap_or((0, 0))) {
                for chunk in army.chunks(12) {
                    cmds.push(Command::Unit { units: chunk.to_vec(), order: CmdOrder::Attack(Target::Pos(target)), queued: false });
                }
                self.last_wave = f.frame;
            }
        }

        Act { cmds, advance_frames: STEP_FRAMES, valid_until_frame: None }
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
    }

    fn attack_target(&mut self, base: Pos) -> Option<Pos> {
        if let Some(p) = self.enemy_buildings.values().min_by_key(|p| dist2(**p, base)) {
            return Some(*p);
        }
        if let Some(i) = self.scouted.iter().position(|s| !s) {
            let s = self.starts[i];
            return Some(((s.0 * 32 + 64), (s.1 * 32 + 48)));
        }
        let _ = self.my_start;
        if self.sweep.is_empty() {
            return None;
        }
        self.sweep_i = (self.sweep_i + 1) % self.sweep.len();
        Some(self.sweep[self.sweep_i])
    }
}

fn dist2(a: Pos, b: Pos) -> i64 {
    let dx = a.0 as i64 - b.0 as i64;
    let dy = a.1 as i64 - b.1 as i64;
    dx * dx + dy * dy
}

fn nearest<'a>(p: Pos, units: &[&'a UnitState]) -> Option<&'a UnitState> {
    units.iter().copied().min_by_key(|u| dist2(p, u.pos))
}

fn is_neutral_owner(owner: u8) -> bool {
    owner >= 8
}
