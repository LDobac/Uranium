use uranium_env::scout::{is_army_type, is_ground_army_type};
use uranium_proto::*;

use super::{Branch, ReactStats, RuleContext, depot_center, pool_start_frame, rush_rules, within};
use crate::brain::Brain;
use crate::types::*;
use crate::util::{dist2, is_enemy};

impl Brain {
    pub(crate) fn own_depots(mine: &[&UnitState]) -> Vec<Pos> {
        mine.iter().filter(|u| is_resource_depot(u.unit_type)).map(|u| u.pos).collect()
    }

    pub(crate) fn home_depot(&self, mine: &[&UnitState]) -> Option<Pos> {
        let depots = Self::own_depots(mine);
        match self.home {
            Some(h) => depots.into_iter().min_by_key(|d| (dist2(*d, h), *d)),
            None => depots.first().copied(),
        }
    }

    pub(crate) fn other_starts(&self) -> Vec<Tile> {
        let home = self.home;
        self.starts.iter().copied().filter(|s| Some(depot_center(*s)) != home).collect()
    }

    pub(crate) fn react_observe(&mut self, f: &RawFrame, mine: &[&UnitState]) {
        let others = self.other_starts();
        let me = self.me;
        let scout = self.scout.and_then(|id| mine.iter().find(|u| u.id == id)).map(|u| u.pos);
        let known_hatcheries: Vec<Pos> = self.enemy_buildings.values().filter(|(_, ty)| is_resource_depot(*ty)).map(|(pos, _)| *pos).collect();
        let home = self.home;
        let Some(r) = self.react.as_mut() else { return };
        let p = r.p.clone();
        if r.enemy_main.is_none() {
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
            r.stats.contact_frame.get_or_insert(f.frame);
        }
        for u in f.units.iter().filter(|u| is_enemy(u, me) && u.unit_type == SPAWNING_POOL) {
            r.stats.pool_seen_frame.get_or_insert(f.frame);
            if !u.has(flags::COMPLETED) && r.stats.pool_start_frame.is_none() {
                r.stats.pool_start_frame = Some(pool_start_frame(f.frame, u.hp));
            }
        }
        if f.frame < p.judge_until {
            let lings = f.units.iter().filter(|u| is_enemy(u, me) && u.unit_type == ZERGLING).count() as u32;
            r.stats.lings_max = r.stats.lings_max.max(lings);
        }
        if r.stats.bunker_frame.is_none() && mine.iter().any(|u| u.unit_type == BUNKER) {
            r.stats.bunker_frame = Some(f.frame);
        }
        r.extractor_seen |= f.units.iter().any(|u| is_enemy(u, me) && u.unit_type == EXTRACTOR);
        let rules = rush_rules(&p, f, me, &RuleContext { depots: &depots, starts: &others, scout, known_hatcheries: &known_hatcheries, extractor_seen: r.extractor_seen });
        for rule in &rules {
            r.stats.rule_frames.entry(*rule).or_insert(f.frame);
        }
        let marines = mine.iter().filter(|u| u.unit_type == MARINE && u.has(flags::COMPLETED)).count();
        if r.branch == Branch::Macro {
            if let Some(&rule) = rules.first().filter(|_| marines < p.macro_marines_any) {
                r.branch = Branch::Defence;
                r.last_contact = f.frame;
                r.stats.defences += 1;
                if r.stats.rush_frame.is_none() {
                    r.stats.rush_frame = Some(f.frame);
                    r.stats.rush_rule = Some(rule);
                }
                let main = home.and_then(|h| depots.iter().copied().min_by_key(|d| (dist2(*d, h), *d)));
                let natural = r.natural.and_then(|n| depots.iter().copied().find(|d| within(*d, depot_center(n), 1)));
                let by_army = army.iter().flat_map(|a| depots.iter().map(move |d| (dist2(*a, *d), *d))).min().map(|(_, d)| d);
                r.def_base = by_army.or(natural).or(main);
            }
        } else if (marines >= p.macro_marines && f.frame >= r.last_contact + p.macro_quiet) || marines >= p.macro_marines_any {
            r.branch = Branch::Macro;
            r.stats.macro_frame.get_or_insert(f.frame);
        }
        let outside = mine.iter().filter(|u| u.unit_type == MARINE && u.has(flags::COMPLETED) && u.loaded_in.is_none()).count();
        let close = f.units.iter().filter(|u| is_enemy(u, me) && is_ground_army_type(u.unit_type) && depots.iter().any(|d| within(u.pos, *d, p.marine_radius))).count();
        r.marines_attack = r.branch == Branch::Defence && close > 0 && outside >= p.marine_ratio * close;
        if r.marines_attack {
            r.holding.clear();
        }
        r.stats.branch = r.branch;
    }

    pub(crate) fn teacher_stats(&self) -> Option<ReactStats> {
        self.react.as_ref().map(|r| r.stats.clone())
    }
}
