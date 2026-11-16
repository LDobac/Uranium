use uranium_proto::*;

use super::{ORDER_MOVE, ReactState, SCOUT_AWAY_TILES, depot_center, within};
use crate::brain::Brain;
use crate::types::*;
use crate::util::{dist2, nearest};

impl Brain {
    pub(crate) fn react_scouting(&mut self, f: &RawFrame, mine: &[&UnitState], minerals: &[&UnitState], cmds: &mut Vec<Command>) {
        let Some(r) = self.react.as_ref() else { return };
        let (look, look_max, main, main_frame) = (r.p.scout_look, r.p.scout_look_max, r.enemy_main, r.stats.main_frame);
        let depot_ordered = self.pending.iter().any(|q| q.ty == DEPOT) || mine.iter().any(|u| matches!(u.unit_type, DEPOT | BARRACKS));
        if !self.scout_done && depot_ordered {
            self.pick_scout(f, mine);
        }
        let Some(id) = self.scout else { return };
        let Some(s) = mine.iter().find(|u| u.id == id) else { return };
        let target = match main {
            Some(m) if self.tp_scout_target(f, s, m).is_some() => self.tp_scout_target(f, s, m).expect("checked"),
            Some(m) => {
                let c = depot_center(m);
                let looked = within(s.pos, c, look) || main_frame.is_some_and(|mf| f.frame >= mf + look_max);
                (!looked).then_some(c)
            }
            None => self.starts.iter().zip(&self.scouted).filter(|(_, done)| !**done).map(|(t, _)| depot_center(*t)).min_by_key(|p| (dist2(*p, s.pos), *p)),
        };
        match target {
            Some(p) => {
                let settled = self.tp.is_some() && s.has(flags::IDLE) && within(s.pos, p, 3);
                if (ReactState::due_on(self.tp.is_some(), Some(self.scout_order), f.frame, crate::brain::WAVE_PERIOD) || s.has(flags::IDLE)) && !settled {
                    cmds.push(Command::Unit { units: vec![id], order: CmdOrder::Move(Target::Pos(p)), queued: false });
                    self.scout_order = f.frame;
                }
            }
            None => {
                let home = self.home.unwrap_or(s.pos);
                if let Some(m) = nearest(home, minerals) {
                    cmds.push(Command::Unit { units: vec![id], order: CmdOrder::Gather(m.id), queued: false });
                }
                self.scout = None;
                self.scout_done = true;
            }
        }
    }

    fn pick_scout(&mut self, f: &RawFrame, mine: &[&UnitState]) {
        let Some(r) = self.react.as_ref() else { return };
        let (main_radius, was_out) = (r.p.main_radius, r.scout_out);
        let others: Vec<Pos> = self.other_starts().into_iter().map(depot_center).collect();
        let home = self.home;
        let out_of_main = |u: &UnitState| home.is_some_and(|h| !within(u.pos, h, SCOUT_AWAY_TILES));
        let heading_out = |u: &UnitState| {
            (u.order == ORDER_MOVE && u.order_pos.is_some_and(|p| others.iter().any(|c| within(p, *c, main_radius)))) || others.iter().any(|c| within(u.pos, *c, SCOUT_AWAY_TILES))
        };
        let current = self.scout.and_then(|id| mine.iter().copied().find(|u| u.id == id));
        if self.scout.is_some() && current.is_none() && was_out {
            self.scout = None;
            self.scout_done = true;
            return;
        }
        let just_ordered = !ReactState::due_on(self.tp.is_some(), Some(self.scout_order), f.frame, crate::brain::WAVE_PERIOD);
        let keep = current.filter(|u| just_ordered || heading_out(u) || out_of_main(u)).map(|u| u.id);
        let jobs: Vec<UnitId> = self.pending.iter().map(|q| q.worker).chain(self.react_busy()).collect();
        let taken_over = || mine.iter().filter(|u| u.unit_type == SCV && u.has(flags::COMPLETED) && !u.has(flags::CONSTRUCTING) && !jobs.contains(&u.id) && heading_out(u)).min_by_key(|u| u.id).map(|u| u.id);
        let pick = match keep.or_else(taken_over) {
            Some(id) => Some(id),
            None => {
                self.scout = None;
                self.free_worker(mine)
            }
        };
        if pick != keep {
            self.scout_order = 0;
        }
        self.scout = pick;
        let out = pick.and_then(|id| mine.iter().copied().find(|u| u.id == id)).is_some_and(out_of_main);
        if let Some(r) = self.react.as_mut() {
            r.scout_out = out;
        }
    }
}
