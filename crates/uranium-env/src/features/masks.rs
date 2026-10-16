//! 034 masks: `unit_cap` capability classes per entity, `loc_class_mask` location classes and
//! `type_mask` (034-R3..R5). Computed from the same `World` as the features (030-R8).

use uranium_proto::{UnitState, flags};

use super::schema::{CELLS, MAP, OWNER_ENEMY, OWNER_SELF};
use super::world::{Entity, World, queue_len, tile_of, top_left};
use crate::actions::{Action, ActionTable, CAPABILITIES, Executor, LOC_CLASSES, TargetKind};

pub const K: usize = CAPABILITIES.len();
pub const L: usize = LOC_CLASSES.len();

fn cap(name: &str) -> usize {
    CAPABILITIES.iter().position(|c| *c == name).unwrap_or_else(|| panic!("capability {name}"))
}

fn loc(name: &str) -> usize {
    LOC_CLASSES.iter().position(|c| *c == name).unwrap_or_else(|| panic!("location class {name}"))
}

/// "CommandCenter" -> "Terran_Command_Center" (034 table `producer` field).
pub fn producer_type(world: &World, producer: &str) -> Option<u16> {
    let mut name = String::from("Terran");
    for c in producer.chars() {
        if c.is_ascii_uppercase() {
            name.push('_');
        }
        name.push(c);
    }
    world.types.unit_by_name(&name).map(|t| t.id as u16)
}

/// Capability row for one entity (034 capability classes).
pub fn unit_caps(w: &World, e: &Entity) -> [bool; K] {
    let mut c = [false; K];
    let u = &e.unit;
    let Some(t) = w.ty(u) else { return c };
    let ids = w.ids;
    let own = e.owner == OWNER_SELF;
    let enemy = e.owner == OWNER_ENEMY && !e.remembered;
    let visible = !e.remembered;
    let done = u.has(flags::COMPLETED);
    let loaded = u.has(flags::LOADED) || u.loaded_in.is_some();
    let lifted = u.has(flags::LIFTED);
    let commandable = own && done && !loaded && u.unit_type != ids.spider_mine && u.unit_type != ids.nuke;
    let disabled = u.status_timers.lockdown > 0 || u.status_timers.stasis > 0;
    let energy = u.energy;
    let ty = u.unit_type;
    let mut set = |name: &str, v: bool| c[cap(name)] = v;

    // selection classes
    let mobile = commandable && !disabled && ((!t.is_building() && t.top_speed > 0.0) || lifted) && ty != ids.tank_sieged;
    set("can_move", mobile);
    set("can_attack", commandable && !disabled && (t.ground_weapon >= 0 || t.air_weapon >= 0 || ty == ids.bunker));
    set("is_worker", commandable && t.is_worker());
    set("can_be_loaded", commandable && !t.is_building() && !t.is_flyer() && t.space_required > 0 && t.space_required <= 8);
    set("is_transport_loaded", commandable && t.space_provided > 0 && w.cargo.get(&u.id).copied().unwrap_or(0) > 0);
    let caster = |tech: u16| commandable && !disabled && w.researched(tech) && energy >= w.tech_energy(tech);
    set("can_stim", commandable && (ty == ids.marine || ty == ids.firebat) && w.researched(ids.tech_stim) && u.hp > 10);
    set("can_siege", commandable && ty == ids.tank && w.researched(ids.tech_siege));
    set("can_unsiege", commandable && ty == ids.tank_sieged);
    let cloaker = (ty == ids.ghost && w.researched(ids.tech_personnel_cloak)) || (ty == ids.wraith && w.researched(ids.tech_cloak_field));
    let cloak_tech = if ty == ids.ghost { ids.tech_personnel_cloak } else { ids.tech_cloak_field };
    set("can_cloak", cloaker && !u.has(flags::CLOAKED) && caster(cloak_tech));
    set("can_decloak", commandable && cloaker && u.has(flags::CLOAKED));
    set("can_lay_mine", commandable && ty == ids.vulture && w.researched(ids.tech_mines) && u.spider_mines > 0);
    set("can_cast_lockdown", ty == ids.ghost && caster(ids.tech_lockdown));
    let nuke_ready = w.frame.units.iter().any(|s| s.owner == w.mem.self_player && s.unit_type == ids.silo && s.nuke_armed);
    set("can_cast_nuke", ty == ids.ghost && caster(ids.tech_nuke) && nuke_ready);
    set("can_heal", ty == ids.medic && caster(ids.tech_healing));
    set("can_cast_restoration", ty == ids.medic && caster(ids.tech_restoration));
    set("can_cast_flare", ty == ids.medic && caster(ids.tech_flare));
    set("can_cast_matrix", ty == ids.vessel && caster(ids.tech_matrix));
    set("can_cast_emp", ty == ids.vessel && caster(ids.tech_emp));
    set("can_cast_irradiate", ty == ids.vessel && caster(ids.tech_irradiate));
    set("can_cast_yamato", ty == ids.battlecruiser && caster(ids.tech_yamato));
    let busy = queue_len(u) > 0 || u.remaining_research > 0;
    let liftable = t.has(crate::types::flag::FLYING_BUILDING);
    set("can_lift", commandable && liftable && !lifted && !busy);
    set("can_land_noncc", commandable && lifted && ty != ids.cc);
    set("can_land_cc", commandable && lifted && ty == ids.cc);
    set("can_rally", commandable && t.has(crate::types::flag::PRODUCES_UNITS) && t.is_building() && !lifted);
    let addon_ok = |parent: u16| commandable && ty == parent && !lifted && !u.has_addon && !busy && addon_space_free(w, u, t);
    set("addon_capable_cc", addon_ok(ids.cc));
    set("addon_capable_factory", addon_ok(ids.factory));
    set("addon_capable_starport", addon_ok(ids.starport));
    set("addon_capable_scifac", addon_ok(ids.science_facility));
    let own_any = own && !loaded;
    set("can_cancel_construction", own_any && t.is_building() && !done);
    set("can_cancel_train", own_any && done && queue_len(u) > 0);
    set("can_cancel_research", own_any && done && u.remaining_research > 0 && u.order == ids.order_research);
    set("can_cancel_upgrade", own_any && done && u.remaining_research > 0 && u.order == ids.order_upgrade);
    set("can_cancel_addon", own_any && done && u.has_addon && w.addon_of.get(&u.id).is_none_or(|_| addon_incomplete(w, u)));

    // target classes
    let targetable = visible && !(u.has(flags::CLOAKED) && !u.has(flags::DETECTED) && e.owner != OWNER_SELF);
    let hurt = (u.hp as i32) < t.max_hp;
    let building = t.is_building();
    set("attackable_enemy", enemy && targetable && !t.has(crate::types::flag::INVINCIBLE));
    set("any_unit", targetable);
    set("own_unit", own && visible);
    set("target_resource", visible && ((t.is_resource() && ty != ids.geyser) || (own && ty == ids.refinery && done)));
    set("repairable", own && visible && (t.is_mechanical() || building) && hurt && done);
    set("lockdownable", enemy && targetable && t.is_mechanical() && !building && u.status_timers.lockdown == 0);
    set("healable", own && visible && t.is_organic() && !t.is_flyer() && !building && hurt);
    let st = &u.status_timers;
    let afflicted = st.lockdown > 0 || st.irradiate > 0 || st.plague > 0 || st.ensnare > 0 || st.maelstrom > 0 || st.blind || st.parasite;
    set("restorable", own && visible && !building && afflicted);
    set("flarable", enemy && targetable && !building && !st.blind);
    set("irradiatable", targetable && !building);
    set("matrixable", own && visible && !building);
    set("yamato_target", enemy && targetable);
    let free_space = t.space_provided - w.cargo.get(&u.id).copied().unwrap_or(0);
    set("loadable_transport", own && visible && done && (ty == ids.dropship || ty == ids.bunker) && free_space > 0);
    c
}

/// The 2x2 addon slot to the right of the footprint is buildable and free (034 addon_capable).
fn addon_space_free(w: &World, u: &UnitState, t: &crate::types::UnitType) -> bool {
    let (x, y) = top_left(u, t);
    let (ax, ay) = (x + t.tile_width, y + t.tile_height - 2);
    (0..2).all(|dy| (0..2).all(|dx| {
        let (tx, ty) = (ax + dx, ay + dy);
        w.map.in_map(tx, ty) && w.map.buildable[ty as usize * MAP + tx as usize] && !w.blocked[ty as usize * MAP + tx as usize]
    }))
}

fn addon_incomplete(w: &World, b: &UnitState) -> bool {
    let Some(t) = w.ty(b) else { return false };
    let (x, y) = top_left(b, t);
    let slot = (x + t.tile_width, y + t.tile_height - 2);
    w.frame.units.iter().any(|a| {
        a.owner == w.mem.self_player && w.ty(a).is_some_and(|at| at.is_addon() && top_left(a, at) == slot) && !a.has(flags::COMPLETED)
    })
}

/// 2D prefix sum of "cannot build here" tiles: O(1) footprint checks.
struct Bad {
    sum: Vec<u32>,
}

impl Bad {
    fn new(bad: impl Fn(usize, usize) -> bool) -> Self {
        let s = MAP + 1;
        let mut sum = vec![0u32; s * s];
        for y in 0..MAP {
            for x in 0..MAP {
                sum[(y + 1) * s + x + 1] = bad(x, y) as u32 + sum[y * s + x + 1] + sum[(y + 1) * s + x] - sum[y * s + x];
            }
        }
        Self { sum }
    }
    /// No bad tile in [x, x+w) x [y, y+h); out-of-map footprints are rejected.
    fn clear(&self, x: usize, y: usize, w: usize, h: usize) -> bool {
        if x + w > MAP || y + h > MAP {
            return false;
        }
        let s = MAP + 1;
        let v = self.sum[(y + h) * s + x + w] + self.sum[y * s + x] - self.sum[y * s + x + w] - self.sum[(y + h) * s + x];
        v == 0
    }
}

/// [L,128,128] location classes (034). Judged from the current observation (034 note).
pub fn loc_masks(w: &World) -> Vec<bool> {
    let mut m = vec![false; L * CELLS];
    let map = w.map;
    let creep = &w.frame.creep;
    let explored = &w.frame.explored;
    let bad = Bad::new(|x, y| {
        let i = y * MAP + x;
        !map.in_map(x as i32, y as i32) || !map.buildable[i] || w.blocked[i] || !explored.get(x as u16, y as u16) || creep.get(x as u16, y as u16)
    });
    // Resource no-build zone for resource depots: 3 tiles around minerals and geysers.
    let mut near_res = vec![false; CELLS];
    let resources = w.entities.iter().map(|e| &e.unit).chain(w.mem.resources.values());
    for r in resources {
        let Some(t) = w.ty(r).filter(|t| t.is_resource()) else { continue };
        let (x, y) = top_left(r, t);
        for ty in (y - 3)..(y + t.tile_height + 3) {
            for tx in (x - 3)..(x + t.tile_width + 3) {
                if w.map.in_map(tx, ty) {
                    near_res[ty as usize * MAP + tx as usize] = true;
                }
            }
        }
    }
    let near = Bad::new(|x, y| near_res[y * MAP + x]);
    let (any, walk, b43, b32, b22, cc, refinery, land) = (
        loc("any"),
        loc("walkable"),
        loc("buildable_4x3"),
        loc("buildable_3x2"),
        loc("buildable_2x2"),
        loc("cc_slot"),
        loc("refinery_slot"),
        loc("landable_4x3"),
    );
    for y in 0..MAP {
        for x in 0..MAP {
            let i = y * MAP + x;
            if !map.in_map(x as i32, y as i32) {
                continue;
            }
            m[any * CELLS + i] = true;
            m[walk * CELLS + i] = map.walkable[i];
            let f43 = bad.clear(x, y, 4, 3);
            m[b43 * CELLS + i] = f43;
            m[land * CELLS + i] = f43;
            m[b32 * CELLS + i] = bad.clear(x, y, 3, 2);
            m[b22 * CELLS + i] = bad.clear(x, y, 2, 2);
            m[cc * CELLS + i] = f43 && near.clear(x, y, 4, 3);
        }
    }
    for e in w.entities.iter().filter(|e| e.unit.unit_type == w.ids.geyser) {
        let t = w.ty(&e.unit).unwrap();
        let (x, y) = top_left(&e.unit, t);
        if w.map.in_map(x, y) {
            m[refinery * CELLS + y as usize * MAP + x as usize] = true;
        }
    }
    m
}

/// Producer type and the addon it must carry for this action (031-R1, 034-R3).
pub fn producer_for(w: &World, a: &Action) -> Option<(u16, Option<u16>)> {
    let p = producer_type(w, a.producer.as_deref()?)?;
    let need_addon = a.unit_type.and_then(|ut| {
        w.types.unit(ut)?.required_units.iter().map(|r| r[0] as u16).find(|&r| w.types.unit(r).is_some_and(|rt| rt.is_addon() && rt.what_builds == p as i32))
    });
    Some((p, need_addon))
}

/// Candidate buildings for a `pick_building` action, best first (031-R1).
pub fn pick_building_candidates<'w>(w: &'w World, a: &Action) -> Vec<&'w UnitState> {
    let Some((p, addon)) = producer_for(w, a) else { return vec![] };
    let research = matches!(a.command.as_str(), "Research" | "Upgrade");
    if a.command == "UnitUseTech" {
        // Scanner sweep: the comsat with the most energy (031-R1).
        let need = w.tech_energy(a.tech.unwrap_or(0));
        let mut v: Vec<&UnitState> =
            w.frame.units.iter().filter(|u| u.owner == w.mem.self_player && u.unit_type == p && u.has(flags::COMPLETED) && u.energy >= need).collect();
        v.sort_by_key(|u| (std::cmp::Reverse(u.energy), u.id));
        return v;
    }
    let mut v = w.producers(p, addon, research);
    if a.unit_type == Some(w.ids.nuke) {
        v.retain(|s| !s.nuke_armed && queue_len(s) == 0);
    }
    v
}

/// 034-R3 prerequisites: resources, supply, tech and required buildings.
fn prereq_ok(w: &World, a: &Action) -> bool {
    let me = &w.frame.self_state;
    let afford = |m: i32, g: i32| me.minerals >= m && me.gas >= g;
    let required_ok = |ut: u16, skip: Option<u16>| {
        w.types.unit(ut).is_some_and(|t| t.required_units.iter().all(|r| Some(r[0] as u16) == skip || w.has_completed(r[0] as u16)))
    };
    match a.command.as_str() {
        "Build" | "BuildAddon" => {
            let ut = a.unit_type.unwrap();
            let t = w.types.unit(ut).unwrap();
            // The selected parent building (addons) or the worker is checked by the selection.
            afford(t.minerals, t.gas) && required_ok(ut, (a.command == "BuildAddon").then_some(t.what_builds as u16))
        }
        "Train" => {
            let ut = a.unit_type.unwrap();
            let t = w.types.unit(ut).unwrap();
            let supply_ok = t.supply_required == 0 || (me.supply_total as i32 - me.supply_used as i32 >= t.supply_required && me.supply_used as i32 + t.supply_required <= 400);
            afford(t.minerals, t.gas) && supply_ok && required_ok(ut, None)
        }
        "Research" => {
            let tech = a.tech.unwrap();
            let t = &w.types.techs[tech as usize];
            !w.researched(tech) && me.researching & (1u64 << tech) == 0 && afford(t.minerals, t.gas)
        }
        "Upgrade" => {
            let up = a.upgrade.unwrap();
            let t = &w.types.upgrades[up as usize];
            let level = me.upgrades.get(up as usize).copied().unwrap_or(0) as i32;
            let req = t.required_units.get(level as usize).copied().unwrap_or(-1);
            level < t.max_repeats && me.upgrading & (1u64 << up) == 0 && afford(t.minerals(level + 1), t.gas(level + 1)) && (req < 0 || w.has_completed(req as u16))
        }
        _ => true,
    }
}

/// [A] action-type mask (034-R3..R5).
pub fn type_mask(w: &World, table: &ActionTable, caps: &[[bool; K]], locs: &[bool]) -> Vec<bool> {
    let cap_any = |name: &str, own_only: bool| {
        let k = cap(name);
        caps.iter().zip(&w.entities).any(|(c, e)| c[k] && (!own_only || e.owner == OWNER_SELF))
    };
    let loc_any = |class: &str| {
        let l = loc(class);
        locs[l * CELLS..(l + 1) * CELLS].iter().any(|&b| b)
    };
    table
        .actions
        .iter()
        .map(|a| {
            if a.id == 0 {
                return true;
            }
            let mut ok = prereq_ok(w, a);
            if a.uses_select {
                ok &= cap_any(&a.select_cap, true);
            }
            match a.executor {
                Executor::PickBuilding => ok &= !pick_building_candidates(w, a).is_empty(),
                Executor::PickWorker => ok &= cap_any("is_worker", true),
                Executor::None => {}
            }
            match a.target {
                TargetKind::Unit => ok &= cap_any(&a.target_cap, false),
                TargetKind::Loc => ok &= loc_any(&a.loc_class),
                TargetKind::None => {}
            }
            ok
        })
        .collect()
}

/// Own unit nearest to a tile, used by tests and the executor.
pub fn dist2_to_tile(u: &UnitState, tile: (i32, i32)) -> i64 {
    let (x, y) = (tile.0 as i64 * 32 + 16, tile.1 as i64 * 32 + 16);
    let (ux, uy) = (u.pos.0 as i64, u.pos.1 as i64);
    let _ = tile_of(u);
    (x - ux).pow(2) + (y - uy).pow(2)
}
