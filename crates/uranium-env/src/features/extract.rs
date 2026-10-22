//! Feature extraction (030-R1): `extract(&RawFrame, &mut AgentMemory, &MapInfo)` -> model
//! inputs + masks. Pure apart from the memory update; uses only own vision (030-R2).

use std::sync::OnceLock;

use uranium_proto::{EffectKind, PrivilegedSummary, Race, RawFrame, UnitId, UnitState, flags};

use super::map::{MapInfo, STATIC_CHANNELS};
use super::masks::{K, loc_masks, type_mask, unit_caps};
use super::memory::AgentMemory;
use super::schema::*;
use super::world::{Entity, World, queue_len, top_left};
use crate::actions::ActionTable;
use crate::types::TypeTables;

/// Normalisation constants not fixed by 030 (documented in reports/m5_model.md).
pub mod norm {
    /// Velocity is in 1/256 px per frame; fastest units move ~6.7 px / frame.
    pub const VELOCITY: f32 = 256.0 * 8.0;
    pub const SPELL_COOLDOWN: f32 = 64.0;
    pub const STATUS_TIMER: f32 = 256.0;
    pub const COUNT_LOG: f32 = 4.0;
    /// ln(1 + 86400): 60 minutes (020 time limit), the longest possible age.
    pub const LAST_SEEN_LOG: f32 = 11.366_755;
}

/// Everything one decision step needs: model inputs, masks and the entity -> unit id map that
/// turns entity indices back into commands (031).
#[derive(Debug, Clone, PartialEq)]
pub struct Observation {
    pub obs: ObsTensors,
    pub masks: Masks,
    pub entity_ids: Vec<UnitId>,
    /// Valid map size in tiles (width, height); needed by symmetry augmentation (030-R4).
    pub map_size: (usize, usize),
    /// The observation is in transposed orientation (030-R4): building footprints are h x w.
    pub transposed: bool,
}

pub fn action_table() -> &'static ActionTable {
    static T: OnceLock<ActionTable> = OnceLock::new();
    T.get_or_init(|| ActionTable::load().expect("compiled-in action table"))
}

pub fn dims() -> &'static FeatureDims {
    static D: OnceLock<FeatureDims> = OnceLock::new();
    D.get_or_init(|| FeatureDims::new(TypeTables::get(), action_table().actions.len()))
}

/// 030-R1. `z` is the strategy condition for this episode (030 §5); `None` = z_present false.
pub fn extract(frame: &RawFrame, mem: &mut AgentMemory, map: &MapInfo, z: Option<&Strategy>) -> Observation {
    let types = TypeTables::get();
    mem.observe(frame, types, map);
    let w = World::new(frame, mem, map);
    let mut entities: Vec<&Entity> = w.entities.iter().collect();
    // 030 §2: entities beyond N_MAX are dropped from the end (neutral first).
    entities.truncate(N_MAX);
    let n = entities.len();

    let mut entity_cat = Vec::with_capacity(n * ENTITY_CAT);
    let mut entity_cont = Vec::with_capacity(n * ENTITY_CONT);
    let mut entity_cell = Vec::with_capacity(n);
    let mut entity_pos = Vec::with_capacity(n * 2);
    let mut unit_cap = Vec::with_capacity(n * K);
    let mut caps = Vec::with_capacity(n);
    let mut is_building = Vec::with_capacity(n);
    for e in &entities {
        // 034-R7 / E-4: lifted buildings, bunkers and turrets count as buildings.
        is_building.push(w.ty(&e.unit).is_some_and(|t| t.is_building()));
        let u = &e.unit;
        let last = if e.owner == OWNER_SELF { mem.unit_last_action.get(&u.id).copied().unwrap_or(0) } else { 0 };
        let order = if (u.order as usize) < ORDERS { u.order } else { ORDER_NONE };
        entity_cat.extend_from_slice(&[u.unit_type as i32, e.owner, order as i32, last as i32]);
        entity_cell.push(cell_of(u.pos, map));
        entity_pos.extend_from_slice(&[u.pos.0 as f32, u.pos.1 as f32]);
        entity_cont.resize(entity_cont.len() + ENTITY_CONT, 0.0);
        let row = entity_cont.len() - ENTITY_CONT;
        entity_features(&w, e, map.width, map.height, &mut entity_cont[row..]);
        let c = unit_caps(&w, e);
        unit_cap.extend_from_slice(&c);
        caps.push(c);
    }
    let locs = loc_masks(&w);
    let tmask = type_mask(&w, action_table(), &caps, &locs);
    let obs = ObsTensors {
        n,
        entity_cat,
        entity_cont,
        entity_cell,
        entity_pos,
        spatial: spatial(&w, &entities),
        scalar: scalar(&w),
        scalar_cat: [race_index(mem.enemy_race), map.map_family, map.self_start.map_or(0, |i| i.min(7) as i32)],
        z_build: z_build(z),
        z_units: z.map_or_else(|| vec![0.0; dims().z_units()], |s| fit(&s.units, dims().z_units())),
        z_present: z.is_some(),
    };
    Observation {
        obs,
        masks: Masks { type_mask: tmask, unit_cap, loc_class_mask: locs, is_building },
        entity_ids: entities.iter().map(|e| e.unit.id).collect(),
        map_size: (map.width, map.height),
        transposed: false,
    }
}

fn fit(v: &[f32], len: usize) -> Vec<f32> {
    let mut out = v.to_vec();
    out.resize(len, 0.0);
    out
}

fn z_build(z: Option<&Strategy>) -> [i32; Z_BUILD] {
    let mut out = [0; Z_BUILD];
    if let Some(s) = z {
        for (o, &a) in out.iter_mut().zip(&s.build) {
            *o = a as i32;
        }
    }
    out
}

pub fn race_index(r: Race) -> i32 {
    match r {
        Race::Terran => RACE_TERRAN,
        Race::Protoss => RACE_PROTOSS,
        Race::Zerg => RACE_ZERG,
        _ => RACE_UNKNOWN,
    }
}

pub fn cell_of(pos: (u16, u16), map: &MapInfo) -> i32 {
    cell_of_pixel(pos.0 as f32, pos.1 as f32, (map.width, map.height))
}

/// 030 §1 `entity_cell` from a pixel position: tile = pixel / 32 rounded down, clamped to the
/// `size` (w, h) map and the 128 grid. Also used on transformed pixels (030-R4, B-6).
pub fn cell_of_pixel(px: f32, py: f32, (w, h): (usize, usize)) -> i32 {
    let tile = |p: f32, n: usize| ((p.max(0.0) / 32.0).floor() as usize).min(n.max(1) - 1).min(MAP - 1);
    (tile(py, h) * MAP + tile(px, w)) as i32
}

/// Offset of the position block inside `entity_cont` (2 normalised + 32 Fourier).
pub const CONT_POS: usize = 6;
pub const CONT_VEL: usize = 40;
pub const CONT_ANGLE: usize = 42;

/// x/(W*32), y/(H*32) and 8 frequencies x 2 axes x sin/cos (030 §2), from pixel coordinates.
pub fn position_features(px: f32, py: f32, width: usize, height: usize, out: &mut [f32]) {
    let xn = px / (width.max(1) as f32 * 32.0);
    let yn = py / (height.max(1) as f32 * 32.0);
    out[0] = xn;
    out[1] = yn;
    for k in 0..8 {
        let f = std::f32::consts::PI * (1u32 << k) as f32;
        out[2 + k * 4] = (f * xn).sin();
        out[3 + k * 4] = (f * xn).cos();
        out[4 + k * 4] = (f * yn).sin();
        out[5 + k * 4] = (f * yn).cos();
    }
}

fn ratio(v: i32, max: i32) -> f32 {
    if max > 0 { (v as f32 / max as f32).min(1.0) } else { 0.0 }
}

/// 030 §2, 76 values.
fn entity_features(w: &World, e: &Entity, width: usize, height: usize, f: &mut [f32]) {
    let u = &e.unit;
    let t = w.ty(u);
    let (max_hp, max_sp, max_en) = t.map_or((0, 0, 0), |t| (t.max_hp, t.max_shields, t.max_energy));

    // hp, shield, energy: ratio + sqrt(abs)/16
    for (i, (v, m)) in [(u.hp, max_hp), (u.shield, max_sp), (u.energy, max_en)].into_iter().enumerate() {
        f[i * 2] = ratio(v as i32, m);
        f[i * 2 + 1] = (v as f32).sqrt() / 16.0;
    }
    position_features(u.pos.0 as f32, u.pos.1 as f32, width, height, &mut f[CONT_POS..CONT_POS + 34]);
    f[CONT_VEL] = u.vel.0 as f32 / norm::VELOCITY;
    f[CONT_VEL + 1] = u.vel.1 as f32 / norm::VELOCITY;
    f[CONT_ANGLE] = u.angle as f32 / 256.0;
    // cooldowns
    let wcd = |id: i32| w.types.weapon(id).map_or(0, |x| x.cooldown);
    f[43] = ratio(u.ground_cd as i32, t.map_or(0, |t| wcd(t.ground_weapon)));
    f[44] = ratio(u.air_cd as i32, t.map_or(0, |t| wcd(t.air_weapon)));
    f[45] = (u.spell_cd as f32 / norm::SPELL_COOLDOWN).min(1.0);
    // progress: construction, else training; queue length / 5
    let build_time = |ty: u16| w.types.unit(ty).map_or(0, |t| t.build_time);
    f[46] = if !u.has(flags::COMPLETED) {
        1.0 - ratio(u.remaining_build as i32, build_time(u.unit_type))
    } else if queue_len(u) > 0 {
        1.0 - ratio(u.remaining_train as i32, build_time(u.build_queue[0]))
    } else {
        0.0
    };
    f[47] = queue_len(u) as f32 / 5.0;
    // 12 flags
    let flag_bits = [
        u.has(flags::COMPLETED),
        u.has(flags::SIEGED),
        u.has(flags::BURROWED),
        u.has(flags::CLOAKED),
        u.has(flags::DETECTED),
        u.has(flags::LIFTED),
        u.has(flags::CARRYING),
        u.has(flags::LOADED) || u.loaded_in.is_some(),
        u.has(flags::IDLE),
        u.has(flags::UNDER_ATTACK),
        u.has_addon,
        u.has(flags::CONSTRUCTING),
    ];
    for (i, b) in flag_bits.into_iter().enumerate() {
        f[48 + i] = b as u8 as f32;
    }
    // 10 status effects (remaining time)
    let s = &u.status_timers;
    let timers = [s.lockdown, s.irradiate, s.matrix, s.stasis, s.plague, s.ensnare, s.maelstrom];
    for (i, v) in timers.into_iter().enumerate() {
        f[60 + i] = (v as f32 / norm::STATUS_TIMER).min(1.0);
    }
    f[67] = s.blind as u8 as f32;
    f[68] = s.parasite as u8 as f32;
    f[69] = (s.stim as f32 / norm::STATUS_TIMER).min(1.0);
    // skill resources
    f[70] = u.spider_mines as f32 / 3.0;
    f[71] = u.nuke_armed as u8 as f32;
    // memory
    f[72] = e.remembered as u8 as f32;
    f[73] = w.frame.frame.saturating_sub(e.seen_frame) as f32 / 2048.0;
    // resources left
    if t.is_some_and(|t| t.is_resource() && u.unit_type != w.ids.geyser) {
        f[74] = u.resources as f32 / 1500.0;
    } else if t.is_some_and(|t| t.is_resource()) || u.unit_type == w.ids.refinery {
        f[75] = u.resources as f32 / 5000.0;
    }
}

/// 030 §3, [26,128,128].
fn spatial(w: &World, entities: &[&Entity]) -> Vec<f32> {
    let map = w.map;
    let mut s = vec![0f32; SPATIAL * CELLS];
    s[..STATIC_CHANNELS * CELLS].copy_from_slice(&map.static_channels);
    let at = |c: usize, x: i32, y: i32| -> Option<usize> { map.in_map(x, y).then(|| c * CELLS + y as usize * MAP + x as usize) };
    for y in 0..map.height as i32 {
        for x in 0..map.width as i32 {
            let i = y as usize * MAP + x as usize;
            let (gx, gy) = (x as u16, y as u16);
            s[ch::VISIBLE_NOW * CELLS + i] = w.frame.visibility.get(gx, gy) as u8 as f32;
            s[ch::EXPLORED * CELLS + i] = w.frame.explored.get(gx, gy) as u8 as f32;
            s[ch::CREEP * CELLS + i] = w.frame.creep.get(gx, gy) as u8 as f32;
            // AMB-030-1: 0 = visible now, 1 = never seen, else log(1+age)/log(1+86400).
            let seen = w.mem.last_visible[i];
            s[ch::LAST_SEEN_AGE * CELLS + i] = if seen == super::memory::NEVER {
                1.0
            } else {
                ((1.0 + w.frame.frame.saturating_sub(seen) as f32).ln() / norm::LAST_SEEN_LOG).min(1.0)
            };
        }
    }
    // Building footprints and unit densities.
    let mut counts = vec![0u16; 4 * CELLS];
    for e in entities {
        let u = &e.unit;
        let Some(t) = w.ty(u) else { continue };
        if t.is_resource() {
            continue;
        }
        let own = e.owner == OWNER_SELF;
        if t.is_building() && !u.has(flags::LIFTED) {
            let (x0, y0) = top_left(u, t);
            let c = if own { ch::OWN_BUILDING } else { ch::ENEMY_BUILDING };
            for dy in 0..t.tile_height {
                for dx in 0..t.tile_width {
                    if let Some(i) = at(c, x0 + dx, y0 + dy) {
                        s[i] = 1.0;
                    }
                }
            }
        } else if !e.remembered && u.loaded_in.is_none() {
            let air = t.is_flyer() || u.has(flags::LIFTED);
            let k = match (own, air) {
                (true, false) => 0,
                (true, true) => 1,
                (false, false) => 2,
                (false, true) => 3,
            };
            let cell = cell_of(u.pos, map) as usize;
            counts[k * CELLS + cell] += 1;
        }
    }
    // log(1+count) through a small table: most cells are 0.
    let ln1p: [f32; 64] = std::array::from_fn(|n| (1.0 + n as f32).ln());
    for (k, c) in [ch::OWN_GROUND, ch::OWN_AIR, ch::ENEMY_GROUND, ch::ENEMY_AIR].into_iter().enumerate() {
        for i in 0..CELLS {
            let n = counts[k * CELLS + i] as usize;
            if n > 0 {
                s[c * CELLS + i] = if n < ln1p.len() { ln1p[n] } else { (1.0 + n as f32).ln() };
            }
        }
    }
    // 030 §3 / E-1: hazards (observed effects only), every tile the effect's circle overlaps.
    // AMB-030-4: an effect is observed only when its centre tile is visible now, as on a
    // player's screen. The OpenBW shim sends every bullet (CompleteMapInformation makes BWAPI
    // report all bullets visible), so the check is made here and also covers existing shards.
    for ef in w.frame.effects.iter().filter(|ef| w.frame.visibility.get(ef.pos.0 / 32, ef.pos.1 / 32)) {
        let c = match ef.kind {
            EffectKind::PsionicStorm => ch::HAZARD_STORM,
            EffectKind::DarkSwarm => ch::HAZARD_SWARM,
            EffectKind::DisruptionWeb => ch::HAZARD_WEB,
            EffectKind::NukeDot => ch::HAZARD_NUKE,
            // EMP, Irradiate, Plague, Ensnare and any other projectile.
            _ => ch::HAZARD_OTHER,
        };
        let r = if ef.radius > 0 { ef.radius as i32 } else { hazard_radius(ef.kind) };
        let (cx, cy) = (ef.pos.0 as i32, ef.pos.1 as i32);
        for ty in (cy - r).div_euclid(32)..=(cy + r).div_euclid(32) {
            for tx in (cx - r).div_euclid(32)..=(cx + r).div_euclid(32) {
                // Nearest point of the tile [32tx, 32tx+32) x [32ty, 32ty+32) to the centre.
                let dx = cx.clamp(tx * 32, tx * 32 + 31) - cx;
                let dy = cy.clamp(ty * 32, ty * 32 + 31) - cy;
                if dx * dx + dy * dy <= r * r {
                    if let Some(i) = at(c, tx, ty) {
                        s[i] = 1.0;
                    }
                }
            }
        }
    }
    // Own scanner sweeps (sight range disc, unchanged by E-1).
    let mut disc = |c: usize, pos: (u16, u16), radius: i32| {
        let (cx, cy, r) = (pos.0 as i32, pos.1 as i32, radius.max(16));
        for ty in (cy - r) / 32..=(cy + r) / 32 {
            for tx in (cx - r) / 32..=(cx + r) / 32 {
                let (dx, dy) = (tx * 32 + 16 - cx, ty * 32 + 16 - cy);
                if dx * dx + dy * dy <= r * r + 32 * 32 {
                    if let Some(i) = at(c, tx, ty) {
                        s[i] = 1.0;
                    }
                }
            }
        }
    };
    for u in w.frame.units.iter().filter(|u| u.owner == w.mem.self_player && u.unit_type == w.ids.scanner_sweep) {
        disc(ch::SCANNER, u.pos, w.types.unit(u.unit_type).map_or(320, |t| t.sight_range));
    }
    s
}

/// 030 §3 / E-1: hazard radius in pixels when the bridge sends none (`Effect::radius` 0, the
/// OpenBW bridge today). Values from the 1.16.1 game data as OpenBW (`bwgame.h`) and BWAPI 4.2
/// (`WeaponType.cpp` splash radii, `unit_types.toml` dimensions) apply them; table in 030 §3.
pub fn hazard_radius(kind: EffectKind) -> i32 {
    match kind {
        // Psionic Storm weapon: outer splash 48 (bullet damage within 48 px of the storm).
        EffectKind::PsionicStorm => 48,
        // Spell_Dark_Swarm unit: dimensions 80 / 80 / 79 / 79 (a 160 x 160 box), half-width 80.
        EffectKind::DarkSwarm => 80,
        // Spell_Disruption_Web unit: dimensions 60 / 40 / 59 / 39 (120 x 80 box), half-width 60.
        EffectKind::DisruptionWeb => 60,
        // Nuclear Strike weapon: splash 128 / 192 / 256; the outer radius 256.
        EffectKind::NukeDot => 256,
        // EMP Shockwave: OpenBW `emp_shockwave` square of the weapon's inner splash 64.
        EffectKind::EmpShockwave => 64,
        // Irradiate: units within 32 px of the irradiated unit take damage.
        EffectKind::Irradiate => 32,
        // Plague / Ensnare: OpenBW `plague` / `ensnare` affect the square at +-64 px.
        EffectKind::Plague | EffectKind::Ensnare => 64,
        // Other projectiles: no area in the data we use; the tile under it (as before E-1).
        EffectKind::Other(_) => 16,
    }
}

/// 030 §4 scalar vector; layout in `FeatureDims::scalar`.
fn scalar(w: &World) -> Vec<f32> {
    let d = dims();
    let me = &w.frame.self_state;
    let mut v = Vec::with_capacity(d.scalar());
    let f = w.frame.frame as f32;
    v.push(f / 86400.0);
    v.push((1.0 + f).ln() / 12.0);
    v.push((1.0 + me.minerals.max(0) as f32).ln() / 10.0);
    v.push((1.0 + me.gas.max(0) as f32).ln() / 10.0);
    v.push(me.supply_used as f32 / 400.0);
    v.push(me.supply_total as f32 / 400.0);
    v.push((me.supply_total as f32).min(400.0) / 400.0);
    for i in 0..63 {
        v.push(me.upgrades.get(i).copied().unwrap_or(0) as f32 / 3.0);
    }
    for i in 0..47 {
        v.push((me.tech >> i & 1) as f32);
    }
    for i in 0..47 {
        v.push((me.researching >> i & 1) as f32);
    }
    for i in 0..63 {
        v.push((me.upgrading >> i & 1) as f32);
    }
    let own: Vec<&UnitState> = w.frame.units.iter().filter(|u| u.owner == w.mem.self_player).collect();
    for &ty in &d.terran_types {
        let done = own.iter().filter(|u| u.unit_type == ty && u.has(flags::COMPLETED)).count();
        let building = own.iter().filter(|u| u.unit_type == ty && !u.has(flags::COMPLETED)).count()
            + own.iter().map(|u| u.build_queue.iter().filter(|&&q| q == ty).count()).sum::<usize>();
        v.push((1.0 + done as f32).ln() / norm::COUNT_LOG);
        v.push((1.0 + building as f32).ln() / norm::COUNT_LOG);
    }
    for ty in &d.observed_types {
        v.push((1.0 + w.mem.enemy_max_seen.get(ty).copied().unwrap_or(0) as f32).ln() / norm::COUNT_LOG);
    }
    let mut last = vec![0f32; d.actions + DELAY_BUCKETS.len() + 1];
    if let Some((a, bucket)) = w.mem.last_action {
        last[a as usize] = 1.0;
        last[d.actions + bucket.min(DELAY_BUCKETS.len() - 1)] = 1.0;
    }
    last[d.actions + DELAY_BUCKETS.len()] = w.mem.last_failed as u8 as f32;
    v.extend(last);
    debug_assert_eq!(v.len(), d.scalar());
    v
}

/// 030-R9: privileged summary for the value heads only.
pub fn privileged(p: &PrivilegedSummary) -> PrivilegedTensors {
    let d = dims();
    let e = &p.enemy;
    let mut v = Vec::with_capacity(d.privileged());
    v.push((1.0 + e.minerals.max(0) as f32).ln() / 10.0);
    v.push((1.0 + e.gas.max(0) as f32).ln() / 10.0);
    v.push(e.supply_used as f32 / 400.0);
    v.push(e.supply_total as f32 / 400.0);
    v.push((e.supply_total as f32).min(400.0) / 400.0);
    for i in 0..63 {
        v.push(e.upgrades.get(i).copied().unwrap_or(0) as f32 / 3.0);
    }
    for i in 0..47 {
        v.push((e.tech >> i & 1) as f32);
    }
    let count = |list: &[(u16, u16)], ty: u16| list.iter().find(|(t, _)| *t == ty).map_or(0, |(_, n)| *n);
    for &ty in &d.observed_types {
        v.push((1.0 + count(&p.enemy_unit_counts, ty) as f32).ln() / norm::COUNT_LOG);
        v.push((1.0 + count(&p.enemy_in_production, ty) as f32).ln() / norm::COUNT_LOG);
    }
    debug_assert_eq!(v.len(), d.privileged());
    PrivilegedTensors { summary: v }
}
