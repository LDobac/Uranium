use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use uranium_proto::*;
use uranium_scripted::T0Build;
use super::PolicySpec;

#[derive(Clone, Default)]
pub struct Routes(Arc<Mutex<HashMap<(u64, u8), PolicySpec>>>);

impl Routes {
    pub fn insert(&self, game_id: u64, slot: u8, spec: PolicySpec) {
        self.0.lock().unwrap().insert((game_id, slot), spec);
    }

    pub fn take(&self, game_id: u64, slot: u8) -> Option<PolicySpec> {
        self.0.lock().unwrap().remove(&(game_id, slot))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionEnd {
    pub slot: u8,
    pub winner: Option<u8>,
    pub frame: u32,
    pub reason: EndReason,
}

pub fn game_result(ends: &[SessionEnd]) -> Option<(SessionEnd, bool)> {
    let first = *ends.first()?;
    let chosen = ends.iter().skip(1).fold(first, |best, e| if e.frame > best.frame { *e } else { best });
    let disagree = ends.iter().any(|e| e.winner != first.winner);
    if mutual_leave(ends) {
        return Some((SessionEnd { winner: None, reason: EndReason::Aborted, ..chosen }, disagree));
    }
    Some((chosen, disagree))
}

pub fn mutual_leave(ends: &[SessionEnd]) -> bool {
    ends.len() == 2 && ends.iter().all(|e| e.reason == EndReason::Surrender && e.winner == Some(e.slot))
}

#[derive(Clone, Default)]
pub struct EndBoard(Arc<Mutex<Board>>);

#[derive(Default)]
struct Board {
    ends: HashMap<u64, Vec<SessionEnd>>,
    hellos: HashMap<u64, usize>,
}

impl EndBoard {
    pub fn push(&self, game_id: u64, e: SessionEnd) {
        self.0.lock().unwrap().ends.entry(game_id).or_default().push(e);
    }

    pub fn hello(&self, game_id: u64) {
        *self.0.lock().unwrap().hellos.entry(game_id).or_default() += 1;
    }

    pub fn connected(&self, game_id: u64) -> usize {
        self.0.lock().unwrap().hellos.get(&game_id).copied().unwrap_or(0)
    }

    pub fn take(&self, game_id: u64) -> Vec<SessionEnd> {
        let mut b = self.0.lock().unwrap();
        b.hellos.remove(&game_id);
        b.ends.remove(&game_id).unwrap_or_default()
    }
}

#[derive(Clone, Default)]
pub struct ConcedeBoard(Arc<Mutex<HashMap<u64, Arc<uranium_match::concede::ConcedeGame>>>>);

impl ConcedeBoard {
    pub fn insert(&self, game_id: u64, game: Arc<uranium_match::concede::ConcedeGame>) {
        self.0.lock().unwrap().insert(game_id, game);
    }

    pub fn get(&self, game_id: u64) -> Option<Arc<uranium_match::concede::ConcedeGame>> {
        self.0.lock().unwrap().get(&game_id).cloned()
    }

    pub fn take(&self, game_id: u64) -> Option<Arc<uranium_match::concede::ConcedeGame>> {
        self.0.lock().unwrap().remove(&game_id)
    }
}

pub type T0Builds = Arc<Mutex<HashMap<(u64, u8), T0Build>>>;

pub fn spec_index(info: &GameInfo, slot: u8) -> u8 {
    let name = info.players.iter().find(|p| p.is_self).map(|p| p.name.as_str()).unwrap_or("");
    name_index(name).unwrap_or(slot)
}

pub fn name_index(name: &str) -> Option<u8> {
    if let Some(i) = name.strip_prefix("uranium-p") {
        return i.parse().ok();
    }
    let (i, _) = name.strip_prefix('p')?.split_once('-')?;
    i.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn end(slot: u8, winner: Option<u8>, frame: u32) -> SessionEnd {
        SessionEnd { slot, winner, frame, reason: EndReason::Elimination }
    }

    #[test]
    fn name_index_reads_both_forms_020_r16() {
        assert_eq!(name_index("uranium-p1"), Some(1));
        assert_eq!(name_index("p0-core-abl-gtrxl-pt"), Some(0));
        assert_eq!(name_index("p1-T0-zerg"), Some(1));
        assert_eq!(name_index("p1-"), Some(1));
        assert_eq!(name_index("player"), None);
        assert_eq!(name_index("pX-a"), None);
        assert_eq!(name_index("Iron"), None);
    }

    #[test]
    fn game_result_takes_the_later_session_020_r13() {
        let (r, d) = game_result(&[end(0, Some(1), 9000), end(1, Some(1), 9000)]).unwrap();
        assert_eq!((r.winner, d), (Some(1), false));
        let (r, d) = game_result(&[end(0, None, 8999), end(1, Some(1), 9001)]).unwrap();
        assert_eq!((r.winner, r.slot, d), (Some(1), 1, true));
        let (r, d) = game_result(&[end(1, Some(1), 9001), end(0, None, 8999)]).unwrap();
        assert_eq!((r.winner, d), (Some(1), true));
        let (r, _) = game_result(&[end(0, Some(0), 500), end(1, Some(1), 500)]).unwrap();
        assert_eq!(r.winner, Some(0));
        assert!(game_result(&[]).is_none());
    }

    #[test]
    fn mutual_leave_is_aborted_020_r13() {
        let left = |slot, winner, frame| SessionEnd { slot, winner, frame, reason: EndReason::Surrender };
        let (r, d) = game_result(&[left(0, Some(0), 1615), left(1, Some(1), 1615)]).unwrap();
        assert_eq!((r.reason, r.winner, d), (EndReason::Aborted, None, true));
        let (r, _) = game_result(&[end(0, Some(1), 4000), left(1, Some(1), 4010)]).unwrap();
        assert_eq!((r.reason, r.winner), (EndReason::Surrender, Some(1)));
        let (r, _) = game_result(&[end(0, Some(0), 8000), left(1, Some(1), 8031)]).unwrap();
        assert_eq!((r.reason, r.winner), (EndReason::Surrender, Some(1)));
        assert!(!mutual_leave(&[left(0, Some(0), 1615)]));
    }
}
