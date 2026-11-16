use serde::Serialize;
use uranium_env::scout::is_army_type;
use uranium_proto::*;

use super::{ReactParams, depot_center, within};
use crate::types::*;
use crate::util::is_enemy;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RushRule {
    PoolStart,
    PoolDone,
    Lings,
    Army,
    Drones,
    OneBase,
}

pub fn pool_start_frame(frame: u32, hp: u16) -> u32 {
    frame.saturating_sub(u32::from(hp.saturating_sub(75)) * 16 / 9)
}

pub struct RuleContext<'a> {
    pub depots: &'a [Pos],
    pub starts: &'a [Tile],
    pub scout: Option<Pos>,
    pub known_hatcheries: &'a [Pos],
    pub extractor_seen: bool,
}

pub fn rush_rules(p: &ReactParams, f: &RawFrame, me: u8, cx: &RuleContext<'_>) -> Vec<RushRule> {
    let mut out = Vec::new();
    if f.frame >= p.judge_until {
        return out;
    }
    let enemies: Vec<&UnitState> = f.units.iter().filter(|u| is_enemy(u, me)).collect();
    let pools: Vec<&&UnitState> = enemies.iter().filter(|u| u.unit_type == SPAWNING_POOL).collect();
    if pools.iter().any(|u| !u.has(flags::COMPLETED) && pool_start_frame(f.frame, u.hp) <= p.pool_start_max) {
        out.push(RushRule::PoolStart);
    }
    if f.frame < p.pool_done_before && pools.iter().any(|u| u.has(flags::COMPLETED)) {
        out.push(RushRule::PoolDone);
    }
    let lings = enemies.iter().filter(|u| u.unit_type == ZERGLING).count();
    if (f.frame < p.lings_before && lings >= p.lings_min) || (f.frame < p.lings_many_before && lings >= p.lings_many) {
        out.push(RushRule::Lings);
    }
    let near_army = enemies.iter().filter(|u| is_army_type(u.unit_type) && cx.depots.iter().any(|d| within(u.pos, *d, p.army_radius))).count();
    if near_army >= p.army_min {
        out.push(RushRule::Army);
    }
    let start_hatchery = enemies.iter().filter(|u| is_resource_depot(u.unit_type)).find(|h| cx.starts.iter().any(|s| within(h.pos, depot_center(*s), p.main_radius)));
    if let (Some(h), Some(scout), false) = (start_hatchery, cx.scout, pools.is_empty()) {
        let only = cx.known_hatcheries.iter().all(|k| within(*k, h.pos, 2));
        if within(scout, h.pos, p.main_look) && only {
            let drones = enemies.iter().filter(|u| u.unit_type == DRONE && within(u.pos, h.pos, p.main_drones_radius)).count();
            if f.frame < p.main_before && drones <= p.drones_max {
                out.push(RushRule::Drones);
            }
            let pool_done = pools.iter().any(|u| u.has(flags::COMPLETED));
            if (p.one_base_from..p.one_base_before).contains(&f.frame) && pool_done && !cx.extractor_seen {
                out.push(RushRule::OneBase);
            }
        }
    }
    out
}
