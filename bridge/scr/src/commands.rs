//! `Command` -> SC:R network commands (012-R4): what the OpenBW shim makes BWAPI send
//! (shim `exec_units` / `exec` / `unit_command`, BWAPI GameCommands.cpp `executeCommand` and
//! BW/OrderTypes.h layouts), with the SC:R 1.21 opcodes for selections and unit targets (013).
//! Only the branches that change the bytes are ported; BWAPI's other pre-checks would only have
//! dropped the command with a `CommandFailed`, which SC:R sessions do not report (012-R14), and the
//! game itself ignores an impossible order.

use std::collections::HashMap;

use uranium_proto::*;

use crate::observe::{Derived, can_attack_move, can_move};
use crate::snapshot::{RawUnit, Snapshot, status};
use crate::tables::{Tables, has_weapon, unit_type};

/// BWAPI 031-R3 / shim: at most 12 units per command.
pub const MAX_UNITS: usize = 12;
/// `UnitTypes::None` in targeted packets (BW::Orders "always0xe4"); only fog-ghost commands carry
/// a real unit type (012-R17).
const NO_UNIT_TYPE: u16 = 0xe4;
/// SC:R chat text field (80 bytes, NUL padded) after the sender byte.
pub const CHAT_TEXT: usize = 80;
/// BWAPI neutral player.
const NEUTRAL: u8 = 11;

/// A player hostile to us: not ourselves, not neutral, not allied.
fn is_enemy(s: &Snapshot, player: u8) -> bool {
    player != s.local && player != NEUTRAL && !s.allied(player)
}

/// Enemy buildings observed this game: tag -> (last position, BWAPI unit type) (012-R17,
/// 011-R20 ①-2).
pub type BuildingMemory = HashMap<u32, (Pos, u16)>;

/// 012-R17: refresh the enemy-building record from the units of an observation sent to the agent
/// (the caller calls it once per observation, not per game frame). An observed enemy building
/// records its position and type; a tag observed as anything else (a dead refinery's neutral
/// geyser, a unit reusing the tag) is dropped. Units not observed keep their entry.
pub fn remember_buildings(s: &Snapshot, d: &[Derived], record: &mut BuildingMemory) {
    for (u, du) in s.units.iter().zip(d).filter(|(_, du)| du.observed) {
        if is_enemy(s, u.player) && unit_type(du.bwapi_type).is_some_and(|x| x.is_building()) {
            record.insert(u.tag, (pos_of(u, du), du.bwapi_type));
        } else {
            record.remove(&u.tag);
        }
    }
}

/// What the commands may refer to this frame.
pub struct Ctx<'a> {
    pub s: &'a Snapshot,
    pub d: &'a [Derived],
    pub t: &'a Tables,
    /// Observed units by tag -> index in `s.units` (011-R20: units are resolved among them).
    pub observed: &'a HashMap<u32, usize>,
    /// Live neutral map resources by tag -> index, in vision or not (011-R20).
    pub resources: &'a HashMap<u32, usize>,
    /// Position of every map resource seen alive (never cleared, 011-R20).
    pub remembered: &'a HashMap<u32, Pos>,
    /// Enemy buildings observed this game (012-R17, [`remember_buildings`]).
    pub buildings: &'a BuildingMemory,
}

impl Ctx<'_> {
    fn unit(&self, tag: u32) -> Option<(&RawUnit, &Derived)> {
        self.observed.get(&tag).map(|&i| (&self.s.units[i], &self.d[i]))
    }

    fn target(&self, tag: u32) -> Option<(&RawUnit, &Derived)> {
        self.unit(tag).or_else(|| self.resources.get(&tag).map(|&i| (&self.s.units[i], &self.d[i])))
    }

    /// 012-R17: a target that is not observed now but is a remembered enemy building -> its last
    /// position and type. Checked before the map resources (a dead refinery's geyser keeps its
    /// tag) and never looked up among the live units.
    fn ghost(&self, tag: u32) -> Option<(Pos, u16)> {
        if self.observed.contains_key(&tag) { None } else { self.buildings.get(&tag).copied() }
    }
}

fn push_u16(v: &mut Vec<u8>, x: u16) {
    v.extend_from_slice(&x.to_le_bytes());
}

/// `0x61` TargetedOrder (BW::Orders::Attack): position, unit tag (0 = none), unit type, order,
/// queued.
fn targeted(t: &Tables, order: u8, pos: Pos, tag: u32, unit_type: u16, queued: bool) -> Vec<u8> {
    let mut v = vec![t.op.targeted_order];
    push_u16(&mut v, pos.0);
    push_u16(&mut v, pos.1);
    v.extend_from_slice(&tag.to_le_bytes());
    push_u16(&mut v, unit_type);
    v.extend([order, queued as u8]);
    v
}

/// `0x60` RightClick: position, unit tag (0 = none), unit type, queued.
fn right_click(t: &Tables, pos: Pos, tag: u32, unit_type: u16, queued: bool) -> Vec<u8> {
    let mut v = vec![t.op.right_click];
    push_u16(&mut v, pos.0);
    push_u16(&mut v, pos.1);
    v.extend_from_slice(&tag.to_le_bytes());
    push_u16(&mut v, unit_type);
    v.push(queued as u8);
    v
}

/// `0x0c` building placement (MakeBuilding / MakeAddon / Land): order byte, tile, unit type.
fn place(t: &Tables, order: u8, tile: Tile, unit_type: u16) -> Vec<u8> {
    let mut v = vec![t.op.place, order];
    push_u16(&mut v, tile.0);
    push_u16(&mut v, tile.1);
    push_u16(&mut v, unit_type);
    v
}

/// `0x63` Select of up to 12 units.
pub fn select(t: &Tables, tags: &[u32]) -> Vec<u8> {
    let tags = &tags[..tags.len().min(MAX_UNITS)];
    let mut v = vec![t.op.select, tags.len() as u8];
    for tag in tags {
        v.extend_from_slice(&tag.to_le_bytes());
    }
    v
}

/// `0x5C` chat as player `me` (012-R11, spike-checked): text cut at 79 bytes, NUL padded.
pub fn chat(t: &Tables, me: u8, text: &str) -> Vec<u8> {
    let mut v = vec![t.op.chat, me];
    let mut body = [0u8; CHAT_TEXT];
    let n = text.len().min(CHAT_TEXT - 1);
    body[..n].copy_from_slice(&text.as_bytes()[..n]);
    v.extend_from_slice(&body);
    v
}

/// BWAPI position of a unit (its transport's when loaded).
fn pos_of(_u: &RawUnit, d: &Derived) -> Pos {
    (d.pos.0.max(0) as u16, d.pos.1.max(0) as u16)
}

fn is_flying(u: &RawUnit, d: &Derived) -> bool {
    unit_type(d.bwapi_type).is_some_and(|x| x.is_flyer()) || (u.status & status::IN_AIR != 0 && unit_type(u.unit_type).is_some_and(|x| x.is_building()))
}

/// BWAPI `canAttackUnit(u, target)` for the shim's right-click branch (Templates.h:613-700):
/// a weapon for the target's layer, not busy, completed. A unit that cannot move would need a
/// range check; for such units both outcomes send the right click, so it is not computed.
fn can_attack_unit(c: &Ctx, (u, du): (&RawUnit, &Derived), (tu, td): (&RawUnit, &Derived)) -> bool {
    let t = c.t;
    let Some(x) = unit_type(du.bwapi_type) else { return false };
    let ty = du.bwapi_type;
    let armed = has_weapon(x.ground_weapon) || has_weapon(x.air_weapon) || ty == t.units.carrier || ty == t.units.reaver;
    if !armed || !du.completed || u.status & status::BURROWED != 0 || du.order == t.orders.constructing_building {
        return false;
    }
    if !x.is_building() && u.status & status::CAN_NOT_RECEIVE_ORDERS != 0 {
        return false;
    }
    if tu.status & status::INVINCIBLE != 0 || tu.ptr == u.ptr {
        return false;
    }
    let weapon = if is_flying(tu, td) { x.air_weapon } else { x.ground_weapon };
    has_weapon(weapon) || ty == t.units.carrier || (ty == t.units.reaver && !is_flying(tu, td))
}

/// The packet one unit of a `Command::Unit` sends (shim `unit_command` + BWAPI `executeCommand`).
/// `target` is the resolved unit target, `pos` the position argument.
fn unit_packet(c: &Ctx, me: (&RawUnit, &Derived), order: &CmdOrder, target: Option<(&RawUnit, &Derived)>, pos: Option<Pos>, q: bool) -> Option<Vec<u8>> {
    let t = c.t;
    let o = &t.orders;
    let (u, du) = me;
    let ty = du.bwapi_type;
    let tgt = |order: u8| target.map(|(tu, td)| targeted(t, order, pos_of(tu, td), tu.tag, NO_UNIT_TYPE, q));
    let at = |order: u8| pos.map(|p| targeted(t, order, p, 0, NO_UNIT_TYPE, q));
    Some(match order {
        CmdOrder::Move(_) => match target {
            Some(_) => tgt(o.follow)?,
            None => at(o.move_)?,
        },
        CmdOrder::Attack(_) => match target {
            Some(_) => tgt(if ty == t.units.carrier {
                o.carrier_attack
            } else if ty == t.units.reaver {
                o.reaver_attack
            } else if unit_type(ty).is_some_and(|x| x.is_building()) {
                o.tower_attack
            } else {
                o.attack1
            })?,
            // 034 Attack_Pos: unarmed units that can move get a move (shim:168-171).
            None if !can_attack_move(ty, t) && can_move(ty) => at(o.move_)?,
            None => at(o.attack_move)?,
        },
        CmdOrder::Patrol(_) => at(o.patrol)?,
        CmdOrder::Hold => vec![t.op.hold_position, q as u8],
        CmdOrder::Stop => {
            if ty == t.units.reaver {
                vec![t.op.reaver_stop]
            } else if ty == t.units.carrier {
                vec![t.op.carrier_stop]
            } else {
                vec![t.op.stop, q as u8]
            }
        }
        CmdOrder::RightClick(_) => match target {
            None => right_click(t, pos?, 0, NO_UNIT_TYPE, q),
            Some((tu, td)) => {
                // Audit F17 (shim:177-183): an enemy the unit cannot attack -> follow; a medic
                // attack-moves to the target's position.
                // `u` is ours (only local units are commanded), so "enemy of u" is `is_enemy`.
                if is_enemy(c.s, tu.player) && can_move(ty) && !can_attack_unit(c, me, (tu, td)) {
                    if ty == t.units.medic && can_attack_move(ty, t) {
                        targeted(t, o.attack_move, pos_of(tu, td), 0, NO_UNIT_TYPE, q)
                    } else {
                        targeted(t, o.follow, pos_of(tu, td), tu.tag, NO_UNIT_TYPE, q)
                    }
                } else {
                    right_click(t, pos_of(tu, td), tu.tag, NO_UNIT_TYPE, q)
                }
            }
        },
        CmdOrder::Siege => vec![t.op.siege, 0],
        CmdOrder::Unsiege => vec![t.op.unsiege, 0],
        CmdOrder::Cloak => vec![t.op.cloak, 0],
        CmdOrder::Decloak => vec![t.op.decloak, 0],
        CmdOrder::Burrow => vec![t.op.burrow, 0],
        CmdOrder::Unburrow => vec![t.op.unburrow, 0],
        CmdOrder::Lift => vec![t.op.lift, 0, 0, 0, 0],
        CmdOrder::Land(tile) => place(t, o.building_land, *tile, ty),
        CmdOrder::Load(_) => {
            let (tu, td) = target?;
            let transports = [t.units.dropship, t.units.shuttle, t.units.overlord];
            if ty == t.units.bunker {
                tgt(o.pickup_bunker)?
            } else if transports.contains(&ty) {
                tgt(o.pickup_transport)?
            } else if td.bwapi_type == t.units.bunker || transports.contains(&td.bwapi_type) {
                right_click(t, pos_of(tu, td), tu.tag, NO_UNIT_TYPE, q)
            } else {
                return None;
            }
        }
        CmdOrder::UnloadAll(p) => {
            if ty == t.units.bunker {
                vec![t.op.unload_all, 0]
            } else {
                targeted(t, o.move_unload, p.unwrap_or(pos_of(u, du)), 0, NO_UNIT_TYPE, q)
            }
        }
        CmdOrder::Unload(_) => {
            let (tu, _) = target?;
            let mut v = vec![t.op.unload_unit];
            v.extend_from_slice(&tu.tag.to_le_bytes());
            v
        }
        CmdOrder::Repair(_) => tgt(o.repair)?,
        CmdOrder::Gather(_) => tgt(o.harvest1)?,
        CmdOrder::ReturnCargo => vec![t.op.return_cargo, q as u8],
        CmdOrder::UseTech { tech, .. } => {
            let tech = *tech;
            let k = &t.techs;
            if let Some((tu, td)) = target {
                if tech == k.archon_warp {
                    vec![t.op.merge_archon]
                } else if tech == k.dark_archon_meld {
                    vec![t.op.merge_dark_archon]
                } else {
                    targeted(t, t.tech_order(tech)?, pos_of(tu, td), tu.tag, NO_UNIT_TYPE, false)
                }
            } else if let Some(p) = pos {
                let order = if tech == k.healing { o.heal_move } else { t.tech_order(tech)? };
                targeted(t, order, p, 0, NO_UNIT_TYPE, false)
            } else if tech == k.stim_packs {
                vec![t.op.stim]
            } else if tech == k.tank_siege_mode {
                vec![if ty == t.units.siege_tank_siege { t.op.unsiege } else { t.op.siege }, 0]
            } else if tech == k.personnel_cloaking || tech == k.cloaking_field {
                let cloaked = u.status & status::CLOAKED != 0 && u.status & status::BURROWED == 0;
                vec![if cloaked { t.op.decloak } else { t.op.cloak }, 0]
            } else if tech == k.burrowing {
                vec![if u.status & status::BURROWED != 0 { t.op.unburrow } else { t.op.burrow }, 0]
            } else {
                return None;
            }
        }
    })
}

/// Unit target and position arguments of an order (shim `needs_*`).
fn args(order: &CmdOrder) -> (Option<u32>, Option<Pos>) {
    let tp = |t: &Target| match *t {
        Target::Unit(id) => (Some(id), None),
        Target::Pos(p) => (None, Some(p)),
    };
    match order {
        CmdOrder::Move(t) | CmdOrder::Attack(t) | CmdOrder::RightClick(t) => tp(t),
        CmdOrder::Patrol(p) => (None, Some(*p)),
        CmdOrder::Land(_) | CmdOrder::Hold | CmdOrder::Stop | CmdOrder::ReturnCargo => (None, None),
        CmdOrder::Siege | CmdOrder::Unsiege | CmdOrder::Cloak | CmdOrder::Decloak | CmdOrder::Burrow | CmdOrder::Unburrow | CmdOrder::Lift => (None, None),
        CmdOrder::Load(id) | CmdOrder::Repair(id) | CmdOrder::Gather(id) | CmdOrder::Unload(id) => (Some(*id), None),
        CmdOrder::UnloadAll(p) => (None, *p),
        CmdOrder::UseTech { target, .. } => target.as_ref().map_or((None, None), tp),
    }
}

/// 012-R17 / 011-R20 ①-2: a command on a remembered enemy building out of sight, sent like a click
/// on its fog ghost: last position and unit type, no tag. Attack -> MoveToAttack for the units
/// that can attack-move (the unit-side check of 011-R23 ②), RightClick -> a typed right click,
/// Move -> a move to the position. Other orders on such a target send nothing (011-R20 ②). The
/// building's real current state is never consulted, and the per-unit corrections of 011-R20 ③
/// do not apply (the game picks each unit's order).
fn exec_ghost(c: &Ctx, units: &[u32], order: &CmdOrder, (pos, ty): (Pos, u16), q: bool, out: &mut Vec<Vec<u8>>) {
    let t = c.t;
    match order {
        CmdOrder::Attack(_) => send_grouped(c, units, out, |(_, du)| can_attack_move(du.bwapi_type, t).then(|| targeted(t, t.orders.move_to_attack, pos, 0, ty, q))),
        CmdOrder::RightClick(_) => send_grouped(c, units, out, |_| Some(right_click(t, pos, 0, ty, q))),
        CmdOrder::Move(_) => exec_units(c, units, &CmdOrder::Move(Target::Pos(pos)), q, out),
        _ => {}
    }
}

/// shim `exec_units` (011-R20): resolve the target (a remembered enemy building first, 012-R17),
/// fall back to a move for a remembered resource, then one packet group per distinct packet.
fn exec_units(c: &Ctx, units: &[u32], order: &CmdOrder, queued: bool, out: &mut Vec<Vec<u8>>) {
    let (target_id, mut pos) = args(order);
    if let Some(ghost) = target_id.and_then(|id| c.ghost(id)) {
        return exec_ghost(c, units, order, ghost, queued, out);
    }
    let mut order = order.clone();
    let target = target_id.and_then(|id| c.target(id));
    let mut target_id = target_id;
    if let (Some(id), None) = (target_id, target)
        && let Some(&p) = c.remembered.get(&id).filter(|_| matches!(order, CmdOrder::Move(_) | CmdOrder::RightClick(_) | CmdOrder::Gather(_)))
    {
        order = CmdOrder::Move(Target::Pos(p));
        target_id = None;
        pos = Some(p);
    }
    let needs_unit = matches!(order, CmdOrder::Load(_) | CmdOrder::Repair(_) | CmdOrder::Gather(_) | CmdOrder::Unload(_));
    let target = target_id.and_then(|id| c.target(id));
    if (target_id.is_some() || needs_unit) && target.is_none() {
        return;
    }
    let needs_unit_or_pos = matches!(order, CmdOrder::Move(_) | CmdOrder::Attack(_) | CmdOrder::RightClick(_));
    // Land carries its tile in the order itself.
    let needs_pos = matches!(order, CmdOrder::Patrol(_));
    if (needs_unit_or_pos && target.is_none() && pos.is_none()) || (needs_pos && pos.is_none()) {
        return;
    }
    send_grouped(c, units, out, |me| unit_packet(c, me, &order, target, pos, queued));
}

/// Our units' packets: one select + packet per distinct packet (units of one group share a
/// select); buildings and critters are sent alone (011-R20 ③).
fn send_grouped<'a>(c: &'a Ctx, units: &[u32], out: &mut Vec<Vec<u8>>, packet_of: impl Fn((&'a RawUnit, &'a Derived)) -> Option<Vec<u8>>) {
    let mut groups: Vec<(Vec<u8>, Vec<u32>)> = Vec::new();
    for &id in units.iter().take(MAX_UNITS) {
        let Some(me) = c.unit(id).filter(|(u, _)| u.player == c.s.local) else { continue };
        let Some(packet) = packet_of(me) else { continue };
        let grouped = !unit_type(me.1.bwapi_type).is_some_and(|x| x.is_building() || x.has("Critter"));
        if !grouped {
            out.push(select(c.t, &[id]));
            out.push(packet);
            continue;
        }
        match groups.iter_mut().find(|(p, _)| *p == packet) {
            Some((_, ids)) => ids.push(id),
            None => groups.push((packet, vec![id])),
        }
    }
    for (packet, ids) in groups {
        out.push(select(c.t, &ids));
        out.push(packet);
    }
}

/// The building tile of a unit (BWAPI `getTilePosition`).
fn tile_of(d: &Derived) -> (i32, i32) {
    let (tw, th) = unit_type(d.bwapi_type).map_or((1, 1), |x| (x.tile_width, x.tile_height));
    ((d.pos.0 - tw * 16) / 32, (d.pos.1 - th * 16) / 32)
}

/// Single-unit commands (shim `exec`).
fn exec_single(c: &Ctx, cmd: &Command, out: &mut Vec<Vec<u8>>) {
    let t = c.t;
    let o = &t.orders;
    let unit = match cmd {
        Command::Build { worker, .. } => *worker,
        Command::BuildAddon { building, .. } | Command::Train { building, .. } | Command::Research { building, .. } | Command::Upgrade { building, .. } | Command::SetRally { building, .. } => *building,
        Command::Morph { unit, .. } | Command::Cancel { unit, .. } => *unit,
        Command::Unit { .. } => return,
    };
    let Some((_, du)) = c.unit(unit).filter(|(u, _)| u.player == c.s.local) else { return };
    let ty = du.bwapi_type;
    let packet = match cmd {
        Command::Build { unit_type: b, tile, .. } => {
            let bt = unit_type(*b);
            if bt.is_some_and(|x| x.is_addon()) {
                place(t, o.place_addon, *tile, *b)
            } else {
                let race_order = match bt.map(|x| x.race.as_str()) {
                    Some("Zerg") => uranium_env::types::order_id("DroneStartBuild").map_or(0, |v| v as u8),
                    Some("Protoss") => uranium_env::types::order_id("PlaceProtossBuilding").map_or(0, |v| v as u8),
                    Some("Terran") => o.place_building,
                    _ => 0,
                };
                place(t, race_order, *tile, *b)
            }
        }
        Command::BuildAddon { unit_type: b, .. } => {
            let (x, y) = tile_of(du);
            let (x, y) = ((x + 4).clamp(0, c.s.width as i32 - 1), (y + 1).clamp(0, c.s.height as i32 - 1));
            place(t, o.place_addon, (x as u16, y as u16), *b)
        }
        Command::Train { unit_type: b, .. } => {
            let un = &t.units;
            let mut v;
            if [un.larva, un.mutalisk, un.hydralisk].contains(&ty) {
                v = vec![t.op.unit_morph];
            } else if [un.hatchery, un.lair, un.spire, un.creep_colony].contains(&ty) {
                v = vec![t.op.building_morph];
            } else if [un.carrier, un.reaver].contains(&ty) {
                out.push(select(t, &[unit]));
                out.push(vec![t.op.train_fighter]);
                return;
            } else {
                v = vec![t.op.train];
            }
            push_u16(&mut v, *b);
            v
        }
        Command::Morph { unit_type: b, .. } => {
            let mut v = vec![if unit_type(*b).is_some_and(|x| x.is_building()) { t.op.building_morph } else { t.op.unit_morph }];
            push_u16(&mut v, *b);
            v
        }
        Command::Research { tech, .. } => vec![t.op.research, *tech as u8],
        Command::Upgrade { upgrade, .. } => vec![t.op.upgrade, *upgrade as u8],
        Command::Cancel { kind, .. } => match kind {
            CancelKind::Construction => vec![t.op.cancel_construction],
            // BWAPI cancelTrain() = slot -2 (the last queued unit).
            CancelKind::Train => vec![t.op.cancel_train, 0xfe, 0],
            CancelKind::Research => vec![t.op.cancel_research],
            CancelKind::Upgrade => vec![t.op.cancel_upgrade],
            CancelKind::Addon => vec![t.op.cancel_addon],
            CancelKind::Morph => vec![if unit_type(ty).is_some_and(|x| x.is_building()) { t.op.cancel_construction } else { t.op.cancel_unit_morph }],
        },
        Command::SetRally { target, .. } => match *target {
            Target::Pos(p) => targeted(t, o.rally_point_tile, p, 0, NO_UNIT_TYPE, false),
            // 012-R17: a remembered enemy building out of sight -> rally to its last position.
            Target::Unit(id) => match c.ghost(id) {
                Some((p, _)) => targeted(t, o.rally_point_tile, p, 0, NO_UNIT_TYPE, false),
                None => match c.target(id) {
                    Some((tu, td)) => targeted(t, o.rally_point_unit, pos_of(tu, td), tu.tag, NO_UNIT_TYPE, false),
                    // 011-R20: a remembered resource that is gone -> rally to where it was.
                    None => match c.remembered.get(&id) {
                        Some(&p) => targeted(t, o.rally_point_tile, p, 0, NO_UNIT_TYPE, false),
                        None => return,
                    },
                },
            },
        },
        Command::Unit { .. } => return,
    };
    out.push(select(t, &[unit]));
    out.push(packet);
}

/// Network commands for one `Act` (in order).
pub fn encode(c: &Ctx, cmds: &[Command]) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    for cmd in cmds {
        match cmd {
            Command::Unit { units, order, queued } => exec_units(c, units, order, *queued, &mut out),
            other => exec_single(c, other, &mut out),
        }
    }
    out
}
