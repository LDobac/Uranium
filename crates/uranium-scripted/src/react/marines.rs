use std::collections::BTreeSet;

use uranium_proto::*;

use super::{Branch, ReactState, bunker_center, toward, within};
use crate::brain::Brain;
use crate::types::*;
use crate::util::{dist2, done};

impl Brain {
    pub(crate) fn react_reserved_marines(&self, mine: &[&UnitState]) -> Vec<UnitId> {
        match self.react_branch() {
            None => Vec::new(),
            Some(Branch::Defence) if !self.react_marines_attack() => mine.iter().filter(|u| u.unit_type == MARINE).map(|u| u.id).collect(),
            Some(_) => self.garrison(mine).into_iter().flat_map(|(_, ids)| ids).collect(),
        }
    }

    fn garrison(&self, mine: &[&UnitState]) -> Vec<(UnitId, Vec<UnitId>)> {
        let Some(r) = self.react.as_ref() else { return Vec::new() };
        let rings = self.tp.is_some();
        let mut taken: BTreeSet<UnitId> = BTreeSet::new();
        let mut out = Vec::new();
        let mut bunkers: Vec<&&UnitState> = mine.iter().filter(|u| u.unit_type == BUNKER && done(u)).collect();
        bunkers.sort_by_key(|b| b.id);
        for b in bunkers {
            let loaded = mine.iter().filter(|u| u.loaded_in == Some(b.id)).count();
            let want = r.p.load_max.saturating_sub(loaded);
            if want == 0 {
                continue;
            }
            let mut free: Vec<&&UnitState> = mine.iter().filter(|u| u.unit_type == MARINE && u.has(flags::COMPLETED) && u.loaded_in.is_none() && !taken.contains(&u.id)).collect();
            free.sort_by_key(|u| (crate::tp::pick_key(rings, u.pos, b.pos), u.id));
            let ids: Vec<UnitId> = free.iter().take(want).map(|u| u.id).collect();
            taken.extend(&ids);
            if !ids.is_empty() {
                out.push((b.id, ids));
            }
        }
        out
    }

    pub(crate) fn react_marines(&mut self, f: &RawFrame, mine: &[&UnitState], cmds: &mut Vec<Command>) {
        let aligned = self.tp.is_some();
        let Some(r) = self.react.as_mut() else { return };
        r.holding.retain(|id| mine.iter().any(|u| u.id == *id && u.loaded_in.is_none()));
        let (every, branch, last) = (r.p.retarget, r.branch, r.last_marines);
        if !ReactState::due_on(aligned, last, f.frame, every) {
            return;
        }
        let mut acted = false;
        for (bunker, ids) in self.garrison(mine) {
            cmds.push(Command::Unit { units: ids, order: CmdOrder::Load(bunker), queued: false });
            acted = true;
        }
        if branch == Branch::Defence && !self.react_marines_attack() {
            acted |= self.hold_marines(mine, cmds);
        }
        if acted {
            self.react.as_mut().expect("react kit").last_marines = Some(f.frame);
        }
    }

    fn hold_marines(&mut self, mine: &[&UnitState], cmds: &mut Vec<Command>) -> bool {
        let Some(main) = self.home_depot(mine).or(self.home) else { return false };
        let (home, facing) = self.defence_base(mine, main);
        let loading: BTreeSet<UnitId> = self.garrison(mine).into_iter().flat_map(|(_, ids)| ids).collect();
        let bunker = mine.iter().filter(|u| u.unit_type == BUNKER && within(u.pos, home, 10)).min_by_key(|u| (dist2(u.pos, home), u.id)).map(|u| u.pos);
        let planned = self.pending.iter().find(|q| q.ty == BUNKER && within(bunker_center(q.tile), home, 10)).map(|q| bunker_center(q.tile));
        let spot = match bunker.or(planned) {
            Some(b) => toward(b, home, 2.0),
            None => toward(home, facing, 3.0),
        };
        let r = self.react.as_mut().expect("react kit");
        let marines: Vec<&&UnitState> = mine.iter().filter(|u| u.unit_type == MARINE && u.has(flags::COMPLETED) && u.loaded_in.is_none() && !loading.contains(&u.id)).collect();
        let (mut walk, mut hold) = (Vec::new(), Vec::new());
        for m in marines {
            if within(m.pos, spot, 3) {
                if r.holding.insert(m.id) {
                    hold.push(m.id);
                }
            } else {
                r.holding.remove(&m.id);
                walk.push(m.id);
            }
        }
        for chunk in walk.chunks(12) {
            cmds.push(Command::Unit { units: chunk.to_vec(), order: CmdOrder::Move(Target::Pos(spot)), queued: false });
        }
        for chunk in hold.chunks(12) {
            cmds.push(Command::Unit { units: chunk.to_vec(), order: CmdOrder::Hold, queued: false });
        }
        !(walk.is_empty() && hold.is_empty())
    }
}
