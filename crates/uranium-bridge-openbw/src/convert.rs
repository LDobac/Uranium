//! C structs <-> proto types. OpenBW uses canonical BWAPI ids, so ids map 1:1 (011-R6).

use uranium_proto::*;

use crate::ffi::*;

/// BWAPI unit types that are really spell areas.
const SPELL_DISRUPTION_WEB: u16 = 105;
const SPELL_DARK_SWARM: u16 = 202;
/// BWAPI bullet types (BulletType.h).
const BULLET_EMP_MISSILE: u16 = 151;
const BULLET_PSIONIC_STORM: u16 = 157;
const BULLET_PLAGUE_CLOUD: u16 = 168;
const BULLET_ENSNARE: u16 = 170;
const NUKE_DOT: u16 = 0xFFFF;

pub fn race(r: u8) -> Race {
    match r {
        0 => Race::Zerg,
        1 => Race::Terran,
        2 => Race::Protoss,
        6 => Race::Random,
        _ => Race::Unknown,
    }
}

pub fn player_state(s: &UrPlayerState) -> PlayerState {
    PlayerState {
        minerals: s.minerals,
        gas: s.gas,
        supply_used: s.supply_used,
        supply_total: s.supply_total,
        upgrades: s.upgrades.to_vec(),
        tech: s.tech,
        researching: s.researching,
        upgrading: s.upgrading,
    }
}

fn opt_id(v: i32) -> Option<UnitId> {
    (v >= 0).then_some(v as u32)
}

pub fn unit(u: &UrUnit) -> UnitState {
    let t = &u.timers;
    UnitState {
        id: u.id,
        owner: u.owner,
        unit_type: u.ty,
        pos: (u.x, u.y),
        vel: (u.vx, u.vy),
        angle: u.angle,
        hp: u.hp,
        shield: u.shield,
        energy: u.energy,
        order: u.order,
        order_target: opt_id(u.order_target),
        order_pos: (u.order_x >= 0 && u.order_y >= 0).then(|| (u.order_x as u16, u.order_y as u16)),
        ground_cd: u.ground_cd,
        air_cd: u.air_cd,
        spell_cd: u.spell_cd,
        flags: u.flags,
        build_queue: u.build_queue,
        remaining_train: u.remaining_train,
        remaining_build: u.remaining_build,
        remaining_research: u.remaining_research,
        resources: u.resources,
        loaded_in: opt_id(u.loaded_in),
        status_timers: StatusTimers {
            lockdown: t[0],
            irradiate: t[1],
            matrix: t[2],
            stasis: t[3],
            plague: t[4],
            ensnare: t[5],
            maelstrom: t[6],
            stim: t[7],
            blind: u.blind != 0,
            parasite: u.parasite != 0,
        },
        spider_mines: u.spider_mines,
        nuke_armed: u.nuke_armed != 0,
        has_addon: u.has_addon != 0,
    }
}

pub fn event(e: &UrEvent) -> Option<GameEvent> {
    let id = || opt_id(e.unit);
    Some(match e.kind {
        UR_EV_CREATE => GameEvent::UnitCreated(id()?),
        UR_EV_DESTROY => GameEvent::UnitDestroyed(id()?),
        UR_EV_MORPH => GameEvent::UnitMorph(id()?),
        UR_EV_DISCOVER => GameEvent::UnitDiscover(id()?),
        UR_EV_EVADE => GameEvent::UnitEvade(id()?),
        UR_EV_NUKE => GameEvent::NukeDetect((e.x.max(0) as u16, e.y.max(0) as u16)),
        UR_EV_CMD_FAILED => GameEvent::CommandFailed { cmd_index: e.unit.max(0) as u16, error: e.x.max(0) as u16 },
        _ => return None,
    })
}

/// Deterministic ordering key for events within one observation.
pub fn event_order(e: &GameEvent) -> (u8, u32, u32) {
    match *e {
        GameEvent::UnitCreated(id) => (0, id, 0),
        GameEvent::UnitDestroyed(id) => (1, id, 0),
        GameEvent::UnitMorph(id) => (2, id, 0),
        GameEvent::UnitDiscover(id) => (3, id, 0),
        GameEvent::UnitEvade(id) => (4, id, 0),
        GameEvent::NukeDetect((x, y)) => (5, x as u32, y as u32),
        GameEvent::CommandFailed { cmd_index, error } => (6, cmd_index as u32, error as u32),
    }
}

pub fn effect(e: &UrEffect) -> Effect {
    let kind = match e.bullet_type {
        NUKE_DOT => EffectKind::NukeDot,
        BULLET_PSIONIC_STORM => EffectKind::PsionicStorm,
        BULLET_EMP_MISSILE => EffectKind::EmpShockwave,
        BULLET_PLAGUE_CLOUD => EffectKind::Plague,
        BULLET_ENSNARE => EffectKind::Ensnare,
        other => EffectKind::Other(other),
    };
    Effect {
        kind,
        pos: (e.x, e.y),
        radius: 0,
        remaining_frames: (e.remaining >= 0).then(|| e.remaining.min(u16::MAX as i32) as u16),
        source_owner: (e.owner >= 0).then(|| e.owner as u8),
    }
}

/// Dark Swarm / Disruption Web are units in BWAPI; expose them as effects too.
pub fn spell_unit_effect(u: &UnitState) -> Option<Effect> {
    let kind = match u.unit_type {
        SPELL_DARK_SWARM => EffectKind::DarkSwarm,
        SPELL_DISRUPTION_WEB => EffectKind::DisruptionWeb,
        _ => return None,
    };
    Some(Effect { kind, pos: u.pos, radius: 0, remaining_frames: None, source_owner: Some(u.owner) })
}

pub fn privileged(frame: u32, p: &UrPrivileged) -> PrivilegedSummary {
    // SAFETY: arrays provided by the shim for the duration of the call.
    let (c, ip) = unsafe { (slice(p.completed, p.n_completed), slice(p.in_production, p.n_in_production)) };
    PrivilegedSummary {
        frame,
        enemy: player_state(&p.enemy),
        enemy_unit_counts: c.iter().map(|t| (t.ty, t.count)).collect(),
        enemy_in_production: ip.iter().map(|t| (t.ty, t.count)).collect(),
    }
}

fn target(t: &Target, cmd: &mut UrCmd) {
    match *t {
        Target::Unit(id) => cmd.target_unit = id as i32,
        Target::Pos((x, y)) => {
            cmd.x = x as i32;
            cmd.y = y as i32;
        }
    }
}

fn cancel_kind(k: CancelKind) -> u8 {
    match k {
        CancelKind::Construction => 0,
        CancelKind::Train => 1,
        CancelKind::Research => 2,
        CancelKind::Upgrade => 3,
        CancelKind::Addon => 4,
        CancelKind::Morph => 5,
    }
}

/// Converts proto commands into the shim's flat command array + unit pool.
pub fn commands(cmds: &[Command], out: &mut Vec<UrCmd>, pool: &mut Vec<i32>) {
    out.clear();
    pool.clear();
    for c in cmds {
        let mut u = UrCmd::default();
        match c {
            Command::Unit { units, order, queued } => {
                u.kind = UR_CMD_UNIT;
                u.queued = *queued as u8;
                u.units_offset = pool.len() as u32;
                u.n_units = units.len().min(12) as u16; // 031-R3
                pool.extend(units.iter().take(12).map(|&id| id as i32));
                u.order = match order {
                    CmdOrder::Move(t) => {
                        target(t, &mut u);
                        0
                    }
                    CmdOrder::Attack(t) => {
                        target(t, &mut u);
                        1
                    }
                    CmdOrder::Patrol(p) => {
                        target(&Target::Pos(*p), &mut u);
                        2
                    }
                    CmdOrder::Hold => 3,
                    CmdOrder::Stop => 4,
                    CmdOrder::RightClick(t) => {
                        target(t, &mut u);
                        5
                    }
                    CmdOrder::Siege => 6,
                    CmdOrder::Unsiege => 7,
                    CmdOrder::Cloak => 8,
                    CmdOrder::Decloak => 9,
                    CmdOrder::Burrow => 10,
                    CmdOrder::Unburrow => 11,
                    CmdOrder::Lift => 12,
                    CmdOrder::Land((x, y)) => {
                        u.x = *x as i32;
                        u.y = *y as i32;
                        13
                    }
                    CmdOrder::Load(id) => {
                        u.target_unit = *id as i32;
                        14
                    }
                    CmdOrder::UnloadAll(p) => {
                        if let Some(p) = p {
                            target(&Target::Pos(*p), &mut u);
                        }
                        15
                    }
                    CmdOrder::Repair(id) => {
                        u.target_unit = *id as i32;
                        16
                    }
                    CmdOrder::Gather(id) => {
                        u.target_unit = *id as i32;
                        17
                    }
                    CmdOrder::ReturnCargo => 18,
                    CmdOrder::UseTech { tech, target: t } => {
                        u.ty = *tech;
                        if let Some(t) = t {
                            target(t, &mut u);
                        }
                        19
                    }
                    // UR_ORD_UNLOAD (034 Unload_Unit, 2026-10-19): the loaded unit to drop.
                    CmdOrder::Unload(id) => {
                        u.target_unit = *id as i32;
                        20
                    }
                };
            }
            Command::Build { worker, unit_type, tile } => {
                u.kind = UR_CMD_BUILD;
                u.unit = *worker as i32;
                u.ty = *unit_type;
                u.x = tile.0 as i32;
                u.y = tile.1 as i32;
            }
            Command::BuildAddon { building, unit_type } => {
                u.kind = UR_CMD_BUILD_ADDON;
                u.unit = *building as i32;
                u.ty = *unit_type;
            }
            Command::Train { building, unit_type } => {
                u.kind = UR_CMD_TRAIN;
                u.unit = *building as i32;
                u.ty = *unit_type;
            }
            Command::Morph { unit, unit_type } => {
                u.kind = UR_CMD_MORPH;
                u.unit = *unit as i32;
                u.ty = *unit_type;
            }
            Command::Research { building, tech } => {
                u.kind = UR_CMD_RESEARCH;
                u.unit = *building as i32;
                u.ty = *tech;
            }
            Command::Upgrade { building, upgrade } => {
                u.kind = UR_CMD_UPGRADE;
                u.unit = *building as i32;
                u.ty = *upgrade;
            }
            Command::Cancel { unit, kind } => {
                u.kind = UR_CMD_CANCEL;
                u.unit = *unit as i32;
                u.cancel_kind = cancel_kind(*kind);
            }
            Command::SetRally { building, target: t } => {
                u.kind = UR_CMD_SET_RALLY;
                u.unit = *building as i32;
                target(t, &mut u);
            }
        }
        out.push(u);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_command_uses_pool_and_caps_at_12() {
        let (mut out, mut pool) = (Vec::new(), Vec::new());
        let units: Vec<u32> = (0..15).collect();
        commands(
            &[
                Command::Unit { units, order: CmdOrder::Attack(Target::Pos((10, 20))), queued: true },
                Command::Build { worker: 7, unit_type: 109, tile: (3, 4) },
                Command::Unit { units: vec![99], order: CmdOrder::Gather(5), queued: false },
            ],
            &mut out,
            &mut pool,
        );
        assert_eq!(out.len(), 3);
        assert_eq!((out[0].n_units, out[0].units_offset, out[0].order, out[0].x, out[0].y, out[0].queued), (12, 0, 1, 10, 20, 1));
        assert_eq!((out[1].kind, out[1].unit, out[1].ty, out[1].x, out[1].y), (UR_CMD_BUILD, 7, 109, 3, 4));
        assert_eq!((out[2].units_offset, out[2].target_unit, out[2].order), (12, 5, 17));
        assert_eq!(pool.len(), 13);
        assert_eq!(pool[12], 99);
    }

    /// 011-R20 (액션 감사 F2): a unit-target order carries no position, so the shim fails it at
    /// once when it cannot resolve the target instead of moving to (-1, -1).
    #[test]
    fn unit_target_orders_carry_no_position_011_r20() {
        let (mut out, mut pool) = (Vec::new(), Vec::new());
        let unit = |order| Command::Unit { units: vec![1], order, queued: false };
        commands(
            &[unit(CmdOrder::Move(Target::Unit(9))), unit(CmdOrder::RightClick(Target::Unit(9))), unit(CmdOrder::Gather(9)), unit(CmdOrder::Move(Target::Pos((5, 6))))],
            &mut out,
            &mut pool,
        );
        for c in &out[..3] {
            assert_eq!((c.target_unit, c.x, c.y), (9, UR_NONE, UR_NONE));
        }
        assert_eq!((out[3].target_unit, out[3].x, out[3].y), (UR_NONE, 5, 6));
    }

    #[test]
    fn effects_map_spells() {
        let e = effect(&UrEffect { bullet_type: 157, x: 1, y: 2, remaining: 40, owner: 1 });
        assert_eq!(e.kind, EffectKind::PsionicStorm);
        assert_eq!(e.remaining_frames, Some(40));
        assert_eq!(effect(&UrEffect { bullet_type: 0xFFFF, x: 0, y: 0, remaining: -1, owner: -1 }).kind, EffectKind::NukeDot);
    }
}
