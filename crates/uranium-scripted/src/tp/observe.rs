use uranium_env::scout::{is_army_type, is_ground_army_type};
use uranium_proto::*;

use super::state::race_of;
use super::{Matchup, SCV_HP, TpContext, tp_rules};
use crate::brain::Brain;
use crate::react::{Branch, depot_center, within};
use crate::types::*;
use crate::util::{dist2, is_enemy};

impl Brain {
    pub(crate) fn tp_observe(&mut self, f: &RawFrame, mine: &[&UnitState]) {
        let Some(p) = self.tp.as_ref().map(|t| t.p.clone()) else { return };
        let me = self.me;
        let others = self.other_starts();
        let scout = self.scout.and_then(|id| mine.iter().find(|u| u.id == id)).map(|u| u.pos);
        let depots = Self::own_depots(mine);
        let enemies: Vec<&UnitState> = f.units.iter().filter(|u| is_enemy(u, me)).collect();
        let found = others.iter().copied().find(|s| self.enemy_buildings.values().any(|(pos, _)| within(*pos, depot_center(*s), p.main_radius)));
        let home = self.home;
        let marines = mine.iter().filter(|u| u.unit_type == MARINE && u.has(flags::COMPLETED)).count();
        let tanks = Self::tanks_done(mine);
        let t = self.tp.as_mut().expect("checked above");
        if t.seen.is_none() {
            t.seen = enemies.iter().find_map(|u| race_of(u.unit_type));
        }
        let matchup = t.matchup();
        for u in enemies.iter().filter(|u| matches!(u.unit_type, GATEWAY | BARRACKS)) {
            for s in others.iter().filter(|s| within(u.pos, depot_center(**s), p.proxy_radius)) {
                t.prod_seen.insert((*s, u.unit_type));
            }
        }
        let rules = tp_rules(&p, f, me, &TpContext { depots: &depots, starts: &others, scout, prod_seen: &t.prod_seen });
        let army: Vec<Pos> = enemies.iter().filter(|u| is_army_type(u.unit_type)).map(|u| u.pos).collect();
        let contact = army.iter().any(|a| depots.iter().any(|d| within(*a, *d, p.army_radius)));
        if t.stats.bunker_frame.is_none() && mine.iter().any(|u| u.unit_type == BUNKER) {
            t.stats.bunker_frame = Some(f.frame);
        }
        for rule in &rules {
            t.stats.rule_frames.entry(*rule).or_insert(f.frame);
        }
        let r = self.react.as_mut().expect("the 022 teacher reuses the 021-R8 state");
        if let Some(s) = found.filter(|_| r.enemy_main.is_none()) {
            r.enemy_main = Some(s);
            r.stats.main_frame = Some(f.frame);
            t.stats.main_frame = Some(f.frame);
        }
        if contact {
            r.last_contact = f.frame;
            t.stats.contact_frame.get_or_insert(f.frame);
        }
        match r.branch {
            Branch::Macro => {
                if let Some(&rule) = rules.first().filter(|_| tanks < p.macro_tanks_any) {
                    r.branch = Branch::Defence;
                    r.last_contact = f.frame;
                    t.stats.defences += 1;
                    if t.stats.rush_frame.is_none() {
                        t.stats.rush_frame = Some(f.frame);
                        t.stats.rush_rule = Some(rule);
                    }
                    let main = home.and_then(|h| depots.iter().copied().min_by_key(|d| (dist2(*d, h), *d)));
                    let natural = r.natural.and_then(|n| depots.iter().copied().find(|d| within(*d, depot_center(n), 1)));
                    let by_army = army.iter().flat_map(|a| depots.iter().map(move |d| (dist2(*a, *d), *d))).min().map(|(_, d)| d);
                    r.def_base = by_army.or(natural).or(main);
                }
            }
            Branch::Defence => {
                let calm = marines >= p.macro_marines && tanks >= p.macro_tanks && f.frame >= r.last_contact + p.quiet;
                if calm || tanks >= p.macro_tanks_any {
                    r.branch = Branch::Macro;
                    t.stats.macro_frame.get_or_insert(f.frame);
                }
            }
        }
        let outside = mine.iter().filter(|u| u.unit_type == MARINE && u.has(flags::COMPLETED) && u.loaded_in.is_none()).count();
        let close = enemies.iter().filter(|u| is_ground_army_type(u.unit_type) && depots.iter().any(|d| within(u.pos, *d, r.p.marine_radius))).count();
        let hold = p.marines_hold_p && matchup == Matchup::Protoss;
        r.marines_attack = r.branch == Branch::Defence && !hold && close > 0 && outside >= r.p.marine_ratio * close;
        if r.marines_attack {
            r.holding.clear();
        }
        t.stats.branch = r.branch;
    }

    pub(crate) fn tp_scout_target(&self, f: &RawFrame, s: &UnitState, main: Tile) -> Option<Option<Pos>> {
        let t = self.tp.as_ref()?;
        let p = &t.p;
        if t.stats.main_frame? >= p.scout_stay_until {
            return None;
        }
        if f.frame >= p.scout_stay_until || u32::from(s.hp) * 100 < SCV_HP * p.scout_hp_pct {
            return Some(None);
        }
        let c = depot_center(main);
        if f.frame >= p.proxy_from {
            return Some(Some(c));
        }
        Some(Some(self.circle_point(c, s.pos, p.scout_circle)))
    }

    pub(crate) fn circle_point(&self, c: Pos, pos: Pos, radius: i64) -> Pos {
        let r = radius * 32;
        let d = (r as f64 * std::f64::consts::FRAC_1_SQRT_2).round() as i64;
        let offsets = [(r, 0), (d, d), (0, r), (-d, d), (-r, 0), (-d, -d), (0, -r), (d, -d)];
        let a = (pos.1 as f64 - c.1 as f64).atan2(pos.0 as f64 - c.0 as f64);
        let k = (a / std::f64::consts::FRAC_PI_4).round() as i64;
        for step in [2i64, 1, 3, 4, 5, 6, 7] {
            let (ox, oy) = offsets[(k + step).rem_euclid(8) as usize];
            let (x, y) = (c.0 as i64 + ox, c.1 as i64 + oy);
            if x < 0 || y < 0 {
                continue;
            }
            let (x, y) = (x as u16, y as u16);
            if (x / 8) < self.walkable.width && (y / 8) < self.walkable.height && self.walkable.get(x / 8, y / 8) {
                return (x, y);
            }
        }
        c
    }
}
