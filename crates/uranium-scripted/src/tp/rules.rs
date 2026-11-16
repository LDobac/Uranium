use std::collections::BTreeSet;

use serde::Serialize;
use uranium_env::scout::is_army_type;
use uranium_proto::*;

use super::TpParams;
use crate::react::{depot_center, within};
use crate::types::*;
use crate::util::is_enemy;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TpRule {
    Gates,
    Zealots,
    ProxyGate,
    Rax,
    Marines,
    ProxyRax,
    Army,
}

pub struct TpContext<'a> {
    pub depots: &'a [Pos],
    pub starts: &'a [Tile],
    pub scout: Option<Pos>,
    pub prod_seen: &'a BTreeSet<(Tile, u16)>,
}

pub fn tp_rules(p: &TpParams, f: &RawFrame, me: u8, cx: &TpContext<'_>) -> Vec<TpRule> {
    let mut out = Vec::new();
    if f.frame >= p.judge_until {
        return out;
    }
    let enemies: Vec<&UnitState> = f.units.iter().filter(|u| is_enemy(u, me)).collect();
    let count = |ty: u16| enemies.iter().filter(|u| u.unit_type == ty).count();
    let proxy = |depot: u16, prod: u16| {
        (p.proxy_from..p.proxy_before).contains(&f.frame)
            && cx.scout.is_some_and(|sc| {
                enemies.iter().filter(|u| u.unit_type == depot && within(sc, u.pos, p.proxy_look)).any(|d| {
                    cx.starts.iter().any(|s| within(d.pos, depot_center(*s), p.main_radius) && !cx.prod_seen.contains(&(*s, prod)))
                })
            })
    };
    if f.frame < p.prod_before && count(GATEWAY) >= p.prod_min {
        out.push(TpRule::Gates);
    }
    if f.frame < p.units_before && count(ZEALOT) >= p.zealots_min {
        out.push(TpRule::Zealots);
    }
    if proxy(NEXUS, GATEWAY) {
        out.push(TpRule::ProxyGate);
    }
    if f.frame < p.prod_before && count(BARRACKS) >= p.prod_min {
        out.push(TpRule::Rax);
    }
    if f.frame < p.units_before && count(MARINE) >= p.marines_min {
        out.push(TpRule::Marines);
    }
    if proxy(CC, BARRACKS) {
        out.push(TpRule::ProxyRax);
    }
    let near_army = enemies.iter().filter(|u| is_army_type(u.unit_type) && cx.depots.iter().any(|d| within(u.pos, *d, p.army_radius))).count();
    if near_army >= p.army_min {
        out.push(TpRule::Army);
    }
    out
}
