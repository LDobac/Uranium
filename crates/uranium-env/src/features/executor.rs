//! Executor (031): semantic action -> one protocol `Command`. Minimal rules only (AMB-031-1):
//! `pick_building` picks the producer with the shortest queue (031-R1), `pick_worker` the nearest
//! free SCV (031-R2); build tiles are used as given. Deterministic (031-R7). Actions that violate
//! the masks are not executed (031-R4, 031-AC2).

use uranium_proto::{CancelKind, CmdOrder, Command, Target, UnitId, UnitState, flags};

use super::extract::{Observation, action_table};
use super::masks::{dist2_to_tile, pick_building_candidates};
use super::schema::{CELLS, MAP};
use super::world::{World, is_disabled};
use crate::actions::{Action, CAPABILITIES, Executor, LOC_CLASSES, TargetKind};

/// One policy decision in host form (entity indices refer to `Observation::entity_ids`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostAction {
    pub action: u16,
    pub delay_bucket: usize,
    pub queued: bool,
    /// Selected entity indices (END already removed).
    pub selected: Vec<usize>,
    pub target_unit: Option<usize>,
    /// Tile index y*128+x.
    pub target_loc: Option<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Executed {
    /// At most one command (031-R3); empty for NoOp.
    pub commands: Vec<Command>,
    /// Units the action was issued to (for `AgentMemory::record_action`).
    pub units: Vec<UnitId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Rejected {
    UnknownAction,
    TypeMasked,
    BadSelection,
    BadTarget,
    BadLocation,
    NoExecutorUnit,
}

fn cap_index(name: &str) -> usize {
    CAPABILITIES.iter().position(|c| *c == name).expect("capability")
}

fn tile_pos(tile: usize) -> (u16, u16) {
    (((tile % MAP) * 32 + 16) as u16, ((tile / MAP) * 32 + 16) as u16)
}

/// Checks the action against the masks it was sampled under (031-R4) and builds the command.
pub fn execute(w: &World, obs: &Observation, act: &HostAction) -> Result<Executed, Rejected> {
    let a: &Action = action_table().actions.get(act.action as usize).ok_or(Rejected::UnknownAction)?;
    if !obs.masks.type_mask[a.id as usize] {
        return Err(Rejected::TypeMasked);
    }
    if a.id == 0 {
        return Ok(Executed { commands: vec![], units: vec![] });
    }
    let n = obs.obs.n;
    let k = CAPABILITIES.len();
    let cap = |i: usize, name: &str| i < n && obs.masks.unit_cap[i * k + cap_index(name)];
    let unit = |i: usize| obs.entity_ids[i];

    // Selection (034-R4): own, eligible, distinct, 1..=max_select.
    let mut units = vec![];
    if a.uses_select {
        let mut seen = std::collections::HashSet::new();
        let ok = !act.selected.is_empty()
            && act.selected.len() <= a.max_select as usize
            && act.selected.iter().all(|&i| cap(i, &a.select_cap) && obs.obs.entity_cat[i * 4 + 1] == 0 && seen.insert(i));
        // 034-R7 / E-4: a selection with a building holds that building only.
        let building = act.selected.iter().any(|&i| obs.masks.is_building.get(i).copied().unwrap_or(false));
        if !ok || (building && act.selected.len() > 1) {
            return Err(Rejected::BadSelection);
        }
        units = act.selected.iter().map(|&i| unit(i)).collect();
    }
    let target = match a.target {
        TargetKind::Unit => {
            // 034-R9: never a unit of the selection (the game rejects self-targeting).
            let t = act.target_unit.filter(|&i| cap(i, &a.target_cap) && (a.self_target || !act.selected.contains(&i))).ok_or(Rejected::BadTarget)?;
            Some(Target::Unit(unit(t)))
        }
        TargetKind::Loc => {
            let l = LOC_CLASSES.iter().position(|c| *c == a.loc_class).expect("loc class");
            let tile = act.target_loc.filter(|&t| t < CELLS && obs.masks.loc_class_mask[l * CELLS + t]).ok_or(Rejected::BadLocation)?;
            Some(Target::Pos(tile_pos(tile)))
        }
        TargetKind::None => None,
    };
    let tile = act.target_loc.map(|t| ((t % MAP) as u16, (t / MAP) as u16));
    let target_id = || match target {
        Some(Target::Unit(id)) => id,
        _ => unreachable!("unit target checked above"),
    };
    let pos = || match target {
        Some(Target::Pos(p)) => p,
        _ => unreachable!("location target checked above"),
    };
    let unit_cmd = |order: CmdOrder| Command::Unit { units: units.clone(), order, queued: act.queued && a.uses_queued };

    let cmd = match a.command.as_str() {
        "UnitMove" => unit_cmd(CmdOrder::Move(target.unwrap())),
        "UnitAttack" => unit_cmd(CmdOrder::Attack(target.unwrap())),
        "UnitPatrol" => unit_cmd(CmdOrder::Patrol(pos())),
        "UnitHold" => unit_cmd(CmdOrder::Hold),
        "UnitStop" => unit_cmd(CmdOrder::Stop),
        "UnitRightClick" => unit_cmd(CmdOrder::RightClick(target.unwrap())),
        "UnitGather" => unit_cmd(CmdOrder::Gather(target_id())),
        "UnitReturnCargo" => unit_cmd(CmdOrder::ReturnCargo),
        "UnitRepair" => unit_cmd(CmdOrder::Repair(target_id())),
        "UnitSiege" => unit_cmd(CmdOrder::Siege),
        "UnitUnsiege" => unit_cmd(CmdOrder::Unsiege),
        "UnitCloak" => unit_cmd(CmdOrder::Cloak),
        "UnitDecloak" => unit_cmd(CmdOrder::Decloak),
        "UnitLoad" => unit_cmd(CmdOrder::Load(target_id())),
        "UnitUnloadAll" => unit_cmd(CmdOrder::UnloadAll(tile.filter(|_| a.target == TargetKind::Loc).map(|_| pos()))),
        // Unload_Unit (2026-10-19, F7): the target must be loaded in the selected transport
        // (BWAPI `canUnload` Invalid_Parameter otherwise); the masks only know "loaded" (031-R4).
        "UnitUnload" => {
            let cargo = target_id();
            if !w.frame.units.iter().any(|u| u.id == cargo && u.loaded_in == Some(units[0])) {
                return Err(Rejected::BadTarget);
            }
            Command::Unit { units: units.clone(), order: CmdOrder::Unload(cargo), queued: false }
        }
        "UnitLift" => unit_cmd(CmdOrder::Lift),
        "UnitLand" => unit_cmd(CmdOrder::Land(tile.unwrap())),
        "UnitUseTech" if a.executor == Executor::PickBuilding => {
            // Scanner sweep: the comsat with the most energy (031-R1).
            let c = *pick_building_candidates(w, a).first().ok_or(Rejected::NoExecutorUnit)?;
            units = vec![c.id];
            Command::Unit { units: units.clone(), order: CmdOrder::UseTech { tech: a.tech.unwrap(), target }, queued: false }
        }
        "UnitUseTech" => unit_cmd(CmdOrder::UseTech { tech: a.tech.unwrap(), target }),
        "Build" => {
            let (tx, ty) = tile.unwrap();
            let worker = pick_worker(w, (tx as i32, ty as i32)).ok_or(Rejected::NoExecutorUnit)?;
            units = vec![worker.id];
            Command::Build { worker: worker.id, unit_type: a.unit_type.unwrap(), tile: (tx, ty) }
        }
        "BuildAddon" => Command::BuildAddon { building: units[0], unit_type: a.unit_type.unwrap() },
        "Train" | "Research" | "Upgrade" => {
            let b = *pick_building_candidates(w, a).first().ok_or(Rejected::NoExecutorUnit)?;
            units = vec![b.id];
            match a.command.as_str() {
                "Train" => Command::Train { building: b.id, unit_type: a.unit_type.unwrap() },
                "Research" => Command::Research { building: b.id, tech: a.tech.unwrap() },
                _ => Command::Upgrade { building: b.id, upgrade: a.upgrade.unwrap() },
            }
        }
        // One building (034-R7: SetRally max_select 1), one command (031-R3).
        "SetRally" => Command::SetRally { building: units[0], target: target.unwrap() },
        "Cancel" => {
            let kind = match a.cancel.as_deref() {
                Some("Construction") => CancelKind::Construction,
                Some("Train") => CancelKind::Train,
                Some("Research") => CancelKind::Research,
                Some("Upgrade") => CancelKind::Upgrade,
                _ => CancelKind::Addon,
            };
            Command::Cancel { unit: units[0], kind }
        }
        other => unreachable!("command {other} validated by the 034 table loader"),
    };
    Ok(Executed { commands: vec![cmd], units })
}

/// 031-R2: nearest SCV that is neither carrying nor constructing; otherwise the nearest carrying
/// one. Ties by unit id.
pub fn pick_worker<'w>(w: &'w World, tile: (i32, i32)) -> Option<&'w UnitState> {
    let scvs: Vec<&UnitState> = w
        .frame
        .units
        .iter()
        // 034-R6 / B-11: a disabled (locked down, stasis, maelstrom) SCV takes no command.
        .filter(|u| u.owner == w.mem.self_player && u.unit_type == w.ids.scv && u.has(flags::COMPLETED) && u.loaded_in.is_none() && !is_disabled(u))
        .filter(|u| !u.has(flags::CONSTRUCTING))
        .collect();
    let best = |carrying: bool| scvs.iter().filter(|u| u.has(flags::CARRYING) == carrying).min_by_key(|u| (dist2_to_tile(u, tile), u.id)).copied();
    best(false).or_else(|| best(true))
}
