use std::collections::HashMap;

use anyhow::{Result, bail};
use uranium_proto::{CancelKind, CmdOrder, Command, Pos, RawFrame, Target, Tile, UnitId, UnitState};
use uranium_scripted::T0Build;
use uranium_scripted::types::{BUNKER, DROPSHIP, OVERLORD, building_tile, is_building};

use crate::cmd::{Cmd, op};
use crate::label::GAME_START_FRAME;
use crate::shard::MemoryAction;
use uranium_env::features::label::DecisionLogLine;
use uranium_env::types::NO_TYPE;

pub fn teacher_build(name: &str) -> Result<T0Build> {
    match name {
        "terran_react" => Ok(T0Build::TerranReact),
        "terran_tp" => Ok(T0Build::TerranTp),
        other => bail!("--relabel {other}: only terran_react (the 021-R8 teacher) and terran_tp (the 022 teacher) relabel (041-R22, 022-R9)"),
    }
}

pub const TEACHER_STEP: u32 = 8;
pub const LIVE_LAG: u32 = 2;
pub const RECORD_DELAY: u32 = 2;

pub fn live_frame(f: u32) -> Option<u32> {
    if f == GAME_START_FRAME {
        return Some(GAME_START_FRAME);
    }
    let l = f + LIVE_LAG;
    (l > GAME_START_FRAME && (l - GAME_START_FRAME).is_multiple_of(TEACHER_STEP)).then_some(l)
}

pub fn teacher_frames(last: u32) -> Vec<u32> {
    (GAME_START_FRAME..=last.max(GAME_START_FRAME)).filter(|&f| live_frame(f).is_some()).collect()
}

pub fn playback_frame(live: u32) -> u32 {
    live.saturating_sub(LIVE_LAG).max(GAME_START_FRAME)
}

pub fn live_of_playback(obs: u32) -> u32 {
    if obs <= GAME_START_FRAME { GAME_START_FRAME } else { obs + LIVE_LAG }
}

pub const MEMORY_MISMATCH_MAX: f64 = 0.01;
pub const MEMORY_END_TOLERANCE: u32 = 8;

pub type LoggedDecision = (u32, MemoryAction);

#[derive(Debug, Clone, PartialEq)]
pub struct ModelLog {
    pub decisions: Vec<LoggedDecision>,
    pub end_frame: Option<u32>,
    pub player: u8,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DecisionLogRead {
    Missing,
    Incomplete(String),
    Complete(ModelLog),
}

pub fn parse_decision_log(text: &str) -> DecisionLogRead {
    let mut out: Vec<LoggedDecision> = Vec::new();
    let mut end = None;
    for (i, line) in text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
        if end.is_some() {
            return DecisionLogRead::Incomplete(format!("line {} after the end line", i + 1));
        }
        match serde_json::from_str::<DecisionLogLine>(line) {
            Ok(DecisionLogLine::Decision { frame, action, delay_bucket, units, .. }) => {
                if out.last().is_some_and(|(f, _)| *f > frame) {
                    return DecisionLogRead::Incomplete(format!("line {}: frame {frame} before the previous decision's", i + 1));
                }
                out.push((frame, MemoryAction { action, delay_bucket, units }));
            }
            Ok(DecisionLogLine::End { decisions, end_frame, player }) => {
                if decisions != out.len() as u64 {
                    return DecisionLogRead::Incomplete(format!("end line counts {decisions} decisions, the log has {}", out.len()));
                }
                end = Some((end_frame, player));
            }
            Err(e) => return DecisionLogRead::Incomplete(format!("line {}: {e}", i + 1)),
        }
    }
    let Some((end_frame, player)) = end else {
        return DecisionLogRead::Incomplete("no end line (the game did not end in the session)".into());
    };
    DecisionLogRead::Complete(ModelLog { decisions: out, end_frame, player })
}

pub fn read_decision_log(path: &std::path::Path) -> DecisionLogRead {
    match std::fs::read_to_string(path) {
        Ok(t) => parse_decision_log(&t),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => DecisionLogRead::Missing,
        Err(e) => DecisionLogRead::Incomplete(format!("reading {}: {e}", path.display())),
    }
}

pub fn bind_log(read: &DecisionLogRead, me: u8, replay_frames: u32) -> Result<&ModelLog, MemoryCheck> {
    let DecisionLogRead::Complete(log) = read else { return Err(MemoryCheck::without_log(read)) };
    let not = |status, reason: String| Err(MemoryCheck { status, reason: Some(reason), decisions: log.decisions.len(), checked: 0, mismatched: 0, decision_steps: None });
    if log.player != me {
        return not("other_player", format!("the log is player {}'s, this view is player {me}'s", log.player));
    }
    match log.end_frame {
        Some(e) if e.abs_diff(replay_frames) <= MEMORY_END_TOLERANCE => Ok(log),
        Some(e) => not("incomplete", format!("log end frame {e}, replay {replay_frames} frames (more than {MEMORY_END_TOLERANCE} apart)")),
        None => not("incomplete", format!("log without an end frame (replay {replay_frames} frames)")),
    }
}

pub fn bind_logs(reads: &[DecisionLogRead], me: u8, replay_frames: u32) -> Result<&ModelLog, MemoryCheck> {
    if let Some(r) = reads.iter().find(|r| matches!(r, DecisionLogRead::Complete(l) if l.player == me)) {
        return bind_log(r, me, replay_frames);
    }
    match reads.iter().find(|r| !matches!(r, DecisionLogRead::Complete(_))).or(reads.first()) {
        Some(r) => bind_log(r, me, replay_frames),
        None => Err(MemoryCheck::without_log(&DecisionLogRead::Missing)),
    }
}

pub fn decision_frames(log: &[LoggedDecision], last: u32) -> Vec<u32> {
    let mut v: Vec<u32> = log.iter().map(|(l, _)| playback_frame(*l)).filter(|&p| p <= last).collect();
    v.dedup();
    v
}

pub fn unit_check(log: &[LoggedDecision], last: u32, own_at: &HashMap<u32, std::collections::HashSet<u32>>) -> (usize, usize) {
    let (mut checked, mut mismatched) = (0, 0);
    for (live, d) in log {
        let p = playback_frame(*live);
        if p > last || d.units.is_empty() {
            continue;
        }
        checked += 1;
        let ok = own_at.get(&p).is_some_and(|own| d.units.iter().all(|u| own.contains(u)));
        mismatched += (!ok) as usize;
    }
    (checked, mismatched)
}

pub fn step_memory(log: &[LoggedDecision], step_frames: &[u32]) -> Vec<Vec<MemoryAction>> {
    let mut out = Vec::with_capacity(step_frames.len());
    let mut k = 0;
    for (i, &obs) in step_frames.iter().enumerate() {
        let live = live_of_playback(obs);
        let mut here = Vec::new();
        while k < log.len() && log[k].0 < live {
            if i > 0 {
                here.push(log[k].1.clone());
            }
            k += 1;
        }
        out.push(here);
    }
    out
}

pub fn add_decision_steps(labels: &mut Vec<crate::label::Label>, log: &[LoggedDecision], limit: u32, end_frame: u32) -> usize {
    use crate::label::{Label, UNKNOWN_ACTION};
    let have: std::collections::HashSet<u32> = labels.iter().map(|l| l.frame).collect();
    let mut add: Vec<u32> = log.iter().map(|(l, _)| playback_frame(*l)).filter(|p| *p <= limit && !have.contains(p)).collect();
    add.sort_unstable();
    add.dedup();
    let n = add.len();
    labels.extend(add.into_iter().map(|frame| Label { frame, action: UNKNOWN_ACTION, delay: 0, queued: false, selected: Vec::new(), target_unit: None, target_tile: None }));
    labels.sort_by_key(|l| l.frame);
    for i in 0..labels.len() {
        if labels[i].is_unknown() && labels[i].delay == 0 && !have.contains(&labels[i].frame) {
            let next = labels.get(i + 1).map_or(end_frame, |l| l.frame);
            labels[i].delay = next.saturating_sub(labels[i].frame).max(1);
        }
    }
    n
}

pub fn distill_labels(log: &[LoggedDecision], limit: u32, end_frame: u32) -> (Vec<crate::label::Label>, usize) {
    use crate::label::{Label, UNKNOWN_ACTION};
    let mut decided: Vec<u32> = log.iter().map(|(l, _)| playback_frame(*l)).filter(|p| *p <= limit).collect();
    decided.sort_unstable();
    decided.dedup();
    let mut frames = decided.clone();
    if frames.first() != Some(&GAME_START_FRAME) {
        frames.insert(0, GAME_START_FRAME);
    }
    let labels = frames
        .iter()
        .enumerate()
        .map(|(i, &frame)| {
            let next = frames.get(i + 1).copied().unwrap_or(end_frame);
            Label { frame, action: UNKNOWN_ACTION, delay: next.saturating_sub(frame).max(1), queued: false, selected: Vec::new(), target_unit: None, target_tile: None }
        })
        .collect();
    (labels, decided.len())
}

#[derive(serde::Serialize, Debug, Clone, PartialEq, Eq)]
pub struct MemoryCheck {
    pub status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    pub decisions: usize,
    pub checked: usize,
    pub mismatched: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decision_steps: Option<usize>,
}

impl MemoryCheck {
    pub fn without_log(read: &DecisionLogRead) -> Self {
        let (status, reason) = match read {
            DecisionLogRead::Missing => ("missing", None),
            DecisionLogRead::Incomplete(r) => ("incomplete", Some(r.clone())),
            DecisionLogRead::Complete(_) => unreachable!("a complete log is checked"),
        };
        Self { status, reason, decisions: 0, checked: 0, mismatched: 0, decision_steps: None }
    }

    pub fn checked(decisions: usize, checked: usize, mismatched: usize) -> Self {
        let bad = checked > 0 && mismatched as f64 / checked as f64 > MEMORY_MISMATCH_MAX;
        Self { status: if bad { "unit_mismatch" } else { "ok" }, reason: None, decisions, checked, mismatched, decision_steps: None }
    }

    pub fn ok(&self) -> bool {
        self.status == "ok"
    }
}

pub const UNMODELED_OP: u8 = 0xff;
const LAST_SLOT: u16 = 254;
const SHUTTLE: u16 = 72;

fn order(name: &str) -> u8 {
    uranium_env::types::order_id(name).unwrap_or_else(|| panic!("order {name}")) as u8
}

fn le(v: u16) -> [u8; 2] {
    v.to_le_bytes()
}

fn targeted(order: u8, p: Pos, tag: u32, queued: bool) -> (u8, Vec<u8>) {
    let mut d = Vec::with_capacity(10);
    d.extend(le(p.0));
    d.extend(le(p.1));
    d.extend(le(tag as u16));
    d.extend(le(NO_TYPE));
    d.extend([order, queued as u8]);
    (op::TARGETED_ORDER, d)
}

fn right_click(p: Pos, tag: u32, queued: bool) -> (u8, Vec<u8>) {
    let mut d = Vec::with_capacity(9);
    d.extend(le(p.0));
    d.extend(le(p.1));
    d.extend(le(tag as u16));
    d.extend(le(NO_TYPE));
    d.push(queued as u8);
    (op::RIGHT_CLICK, d)
}

fn build(order: u8, t: Tile, ty: u16) -> (u8, Vec<u8>) {
    let mut d = vec![order];
    d.extend(le(t.0));
    d.extend(le(t.1));
    d.extend(le(ty));
    (op::BUILD, d)
}

fn select(ids: &[UnitId]) -> (u8, Vec<u8>) {
    let mut d = vec![ids.len() as u8];
    for &id in ids {
        d.extend(le(id as u16));
    }
    (op::SELECT, d)
}

pub fn command_cmds(c: &Command, f: &RawFrame, me: u8, player: u8) -> Vec<Cmd> {
    let unit = |id: UnitId| f.units.iter().find(|u| u.id == id);
    let own = |id: UnitId| unit(id).filter(|u| u.owner == me);
    let at = |id: UnitId| unit(id).map_or((0, 0), |u| u.pos);
    let mut out: Vec<(u8, Vec<u8>)> = Vec::new();
    let single = |id: UnitId, out: &mut Vec<(u8, Vec<u8>)>, cmd: (u8, Vec<u8>)| {
        if own(id).is_some() {
            out.push(select(&[id]));
            out.push(cmd);
        }
    };
    match c {
        Command::Unit { units, order: o, queued } => {
            let mine: Vec<&UnitState> = units.iter().filter_map(|&id| own(id)).collect();
            let (buildings, mobile): (Vec<&UnitState>, Vec<&UnitState>) = mine.into_iter().partition(|u| is_building(u.unit_type));
            let mut groups: Vec<Vec<&UnitState>> = Vec::new();
            if !mobile.is_empty() {
                groups.push(mobile);
            }
            groups.extend(buildings.into_iter().map(|b| vec![b]));
            for g in groups {
                if let Some(cmd) = unit_order(o, *queued, g[0], &unit, &at) {
                    out.push(select(&g.iter().map(|u| u.id).collect::<Vec<_>>()));
                    out.push(cmd);
                }
            }
        }
        Command::Build { worker, unit_type, tile } => {
            let o = if is_addon(*unit_type) { order("PlaceAddon") } else { order("PlaceBuilding") };
            single(*worker, &mut out, build(o, *tile, *unit_type));
        }
        Command::BuildAddon { building, unit_type } => {
            let t = own(*building).map_or((0, 0), |b| {
                let p = building_tile(b);
                (p.0 + 4, p.1 + 1)
            });
            single(*building, &mut out, build(order("PlaceAddon"), t, *unit_type));
        }
        Command::Train { building, unit_type } => single(*building, &mut out, (op::TRAIN, le(*unit_type).to_vec())),
        Command::Morph { unit, unit_type } => {
            let o = if is_building(*unit_type) { 0x35 } else { op::UNIT_MORPH };
            single(*unit, &mut out, (o, le(*unit_type).to_vec()));
        }
        Command::Research { building, tech } => single(*building, &mut out, (0x30, vec![*tech as u8])),
        Command::Upgrade { building, upgrade } => single(*building, &mut out, (0x32, vec![*upgrade as u8])),
        Command::Cancel { unit, kind } => {
            let building = own(*unit).is_some_and(|u| is_building(u.unit_type));
            let cmd = match kind {
                CancelKind::Construction => (0x18, vec![]),
                CancelKind::Train => (op::CANCEL_TRAIN, le(LAST_SLOT).to_vec()),
                CancelKind::Research => (0x31, vec![]),
                CancelKind::Upgrade => (0x33, vec![]),
                CancelKind::Addon => (0x34, vec![]),
                CancelKind::Morph if building => (0x18, vec![]),
                CancelKind::Morph => (0x19, vec![]),
            };
            single(*unit, &mut out, cmd);
        }
        Command::SetRally { building, target } => {
            let cmd = match target {
                Target::Pos(p) => targeted(order("RallyPointTile"), *p, 0, false),
                Target::Unit(t) => targeted(order("RallyPointUnit"), at(*t), *t, false),
            };
            single(*building, &mut out, cmd);
        }
    }
    out.into_iter().map(|(op, data)| Cmd { frame: f.frame, player, op, data }).collect()
}

fn is_addon(ty: u16) -> bool {
    uranium_env::types::TypeTables::get().unit(ty).is_some_and(|t| t.is_addon())
}

fn unit_order<'a>(o: &CmdOrder, q: bool, u: &UnitState, unit: &impl Fn(UnitId) -> Option<&'a UnitState>, at: &impl Fn(UnitId) -> Pos) -> Option<(u8, Vec<u8>)> {
    let qb = vec![q as u8];
    Some(match o {
        CmdOrder::Move(Target::Pos(p)) => targeted(order("Move"), *p, 0, q),
        CmdOrder::Move(Target::Unit(t)) => targeted(order("Follow"), at(*t), *t, q),
        CmdOrder::Attack(Target::Pos(p)) => targeted(order("AttackMove"), *p, 0, q),
        CmdOrder::Attack(Target::Unit(t)) => {
            let o = if is_building(u.unit_type) { order("TowerAttack") } else { order("Attack1") };
            targeted(o, at(*t), *t, q)
        }
        CmdOrder::Patrol(p) => targeted(order("Patrol"), *p, 0, q),
        CmdOrder::Hold => (0x2b, qb),
        CmdOrder::Stop => (0x1a, qb),
        CmdOrder::RightClick(Target::Pos(p)) => right_click(*p, 0, q),
        CmdOrder::RightClick(Target::Unit(t)) => right_click(at(*t), *t, q),
        CmdOrder::Siege => (0x26, vec![0]),
        CmdOrder::Unsiege => (0x25, vec![0]),
        CmdOrder::Cloak => (0x21, vec![0]),
        CmdOrder::Decloak => (0x22, vec![0]),
        CmdOrder::Burrow => (0x2c, vec![0]),
        CmdOrder::Unburrow => (0x2d, vec![0]),
        CmdOrder::Lift => (0x2f, vec![0; 4]),
        CmdOrder::Land(t) => build(order("BuildingLand"), *t, u.unit_type),
        CmdOrder::Load(t) => {
            let transport = |ty: u16| matches!(ty, DROPSHIP | SHUTTLE | OVERLORD);
            if u.unit_type == BUNKER {
                targeted(order("PickupBunker"), at(*t), *t, q)
            } else if transport(u.unit_type) {
                targeted(order("PickupTransport"), at(*t), *t, q)
            } else if unit(*t).is_some_and(|x| x.unit_type == BUNKER || transport(x.unit_type)) {
                right_click(at(*t), *t, q)
            } else {
                return None;
            }
        }
        CmdOrder::UnloadAll(None) if u.unit_type == BUNKER => (0x28, qb),
        CmdOrder::UnloadAll(None) => targeted(order("MoveUnload"), u.pos, 0, q),
        CmdOrder::UnloadAll(Some(p)) => targeted(order("MoveUnload"), *p, 0, q),
        CmdOrder::Repair(t) => targeted(order("Repair"), at(*t), *t, q),
        CmdOrder::Gather(t) => targeted(order("Harvest1"), at(*t), *t, q),
        CmdOrder::ReturnCargo => (0x1e, qb),
        CmdOrder::Unload(t) => (op::UNLOAD, le(*t as u16).to_vec()),
        CmdOrder::UseTech { tech, target: None } if Some(*tech as u32) == stim() => (0x36, vec![]),
        CmdOrder::UseTech { .. } => (UNMODELED_OP, vec![]),
    })
}

fn stim() -> Option<u32> {
    uranium_env::types::TypeTables::get().techs.iter().find(|t| t.name == "Stim_Packs").map(|t| t.id as u32)
}

pub fn own_units(f: &RawFrame, me: u8) -> HashMap<u32, crate::label::Own> {
    f.units.iter().filter(|u| u.owner == me).map(|u| (u.id, crate::label::Own::from_state(u))).collect()
}

pub struct Shadow {
    bot: Box<dyn uranium_scripted::Bot>,
    me: u8,
    player: u8,
    last: u32,
    pub cmds: Vec<Cmd>,
    pub units_at: HashMap<u32, HashMap<u32, crate::label::Own>>,
    pub observations: u32,
    pub repeats_dropped: u32,
    last_kept: HashMap<RepeatKey, u32>,
}

type RepeatKey = (u8, Vec<u8>, Vec<u8>);

pub const REPEAT_WINDOW: u32 = 32;

impl Shadow {
    pub fn new(info: &uranium_proto::GameInfo, build: T0Build, hints: uranium_scripted::MapHints, player: u8, last: u32) -> Self {
        Self {
            bot: uranium_scripted::for_game_hints(info, build, hints),
            me: info.self_player,
            player,
            last,
            cmds: Vec::new(),
            units_at: HashMap::new(),
            observations: 0,
            repeats_dropped: 0,
            last_kept: HashMap::new(),
        }
    }

    pub fn observe(&mut self, f: &RawFrame) {
        let Some(live) = live_frame(f.frame).filter(|_| f.frame <= self.last) else { return };
        self.observations += 1;
        let mut seen = f.clone();
        seen.frame = live;
        let act = self.bot.act(&seen);
        let at = live + RECORD_DELAY;
        let start = self.cmds.len();
        for c in &act.cmds {
            let cmds = command_cmds(c, &seen, self.me, self.player);
            for pair in cmds.chunks(2) {
                let [select, order] = pair else { unreachable!("command_cmds gives (select, order) pairs") };
                let key = (order.op, order.data.clone(), select.data.clone());
                if self.last_kept.get(&key).is_some_and(|&k| at.saturating_sub(k) < REPEAT_WINDOW) {
                    self.repeats_dropped += 1;
                    continue;
                }
                self.last_kept.insert(key, at);
                self.cmds.extend(pair.iter().map(|c| Cmd { frame: at, ..c.clone() }));
            }
        }
        if self.cmds.len() > start {
            self.units_at.insert(at, own_units(f, self.me));
        }
    }

    pub fn labels(&self, labeler: &crate::label::Labeler, end_frame: u32) -> (Vec<crate::label::Label>, crate::label::Coverage) {
        let empty = HashMap::new();
        labeler.label(&self.cmds, end_frame, |f| self.units_at.get(&f).unwrap_or(&empty))
    }

    pub fn react_stats(&self) -> Option<uranium_scripted::ReactStats> {
        self.bot.react_stats()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::label::Labeler;
    use uranium_proto::flags;
    use uranium_scripted::types::{BARRACKS, CC, MARINE, MINERAL_FIELDS, SCV};

    fn unit(id: u32, owner: u8, ty: u16, pos: Pos) -> UnitState {
        UnitState { id, owner, unit_type: ty, pos, flags: flags::COMPLETED, ..Default::default() }
    }

    fn frame_at(n: u32, units: Vec<UnitState>, minerals: i32) -> RawFrame {
        let all = |w, h| uranium_proto::BitGrid::from_bytes(w, h, &vec![1; w as usize * h as usize]);
        RawFrame {
            frame: n,
            self_state: uranium_proto::PlayerState { minerals, supply_used: 8, supply_total: 20, upgrades: vec![0; 63], ..Default::default() },
            units,
            visibility: all(64, 64),
            explored: all(64, 64),
            creep: uranium_proto::BitGrid::new(64, 64),
            events: vec![],
            effects: vec![],
        }
    }

    fn frame(units: Vec<UnitState>) -> RawFrame {
        frame_at(97, units, 0)
    }

    fn info() -> uranium_proto::GameInfo {
        use uranium_proto::{PlayerInfo, Race, StaticMap};
        let all = |w, h| uranium_proto::BitGrid::from_bytes(w, h, &vec![1; w as usize * h as usize]);
        uranium_proto::GameInfo {
            map_name: "t".into(),
            map_hash: [0; 32],
            map_size_tiles: (64, 64),
            self_player: 0,
            players: vec![
                PlayerInfo { id: 0, name: "a".into(), race: Race::Terran, start_tile: Some((4, 4)), is_self: true, is_enemy: false },
                PlayerInfo { id: 1, name: "b".into(), race: Race::Zerg, start_tile: None, is_self: false, is_enemy: true },
            ],
            static_map: StaticMap { width_tiles: 64, height_tiles: 64, walkable: all(256, 256), buildable: all(64, 64), ground_height: vec![0; 64 * 64], start_locations: vec![(4, 4), (56, 56)] },
            latency_frames: 0,
        }
    }

    fn start_units() -> Vec<UnitState> {
        let idle = |id| UnitState { flags: flags::COMPLETED | flags::IDLE, ..unit(id, 0, SCV, (200, 230)) };
        let mut v = vec![unit(10, 0, CC, (192, 176)), idle(1), idle(2), idle(3), idle(4)];
        v.extend((0..4).map(|i| unit(50 + i, 11, MINERAL_FIELDS[0], (100 + 32 * i as u16, 60))));
        v
    }

    fn names(labels: &[crate::label::Label]) -> Vec<(u32, String)> {
        let table = uranium_env::actions::ActionTable::load().unwrap();
        labels.iter().map(|l| (l.frame, table.actions[l.action as usize].name.clone())).collect()
    }

    #[test]
    fn shadow_labels_are_the_teachers_and_deterministic_041_r22() {
        let frames: Vec<RawFrame> = [1u32, 5, 7, 9, 12, 15, 23, 31].iter().map(|&n| frame_at(n, start_units(), 50)).collect();
        let run = || {
            let mut s = Shadow::new(&info(), T0Build::TerranReact, uranium_scripted::MapHints::default(), 3, 23);
            for f in &frames {
                s.observe(f);
            }
            s
        };
        let (a, b) = (run(), run());
        assert_eq!(a.observations, 4, "playback frames 1, 7, 15, 23 only");
        assert_eq!(a.cmds, b.cmds);
        let lab = Labeler::new().unwrap();
        let (la, _) = a.labels(&lab, 60);
        assert_eq!(la, b.labels(&lab, 60).0);
        let got = names(&la);
        let first: Vec<&str> = got.iter().filter(|(f, n)| *f < 9 && n != "NoOp").map(|(_, n)| n.as_str()).collect();
        assert_eq!(first.iter().filter(|n| **n == "Gather_Unit").count(), 4, "{got:?}");
        assert_eq!(first.iter().filter(|n| **n == "Train_SCV").count(), 1, "{got:?}");
        assert!(a.cmds.iter().all(|c| [3, 11, 19, 27].contains(&c.frame)));
        assert!(a.cmds.iter().any(|c| c.frame == 3));
        assert!(a.repeats_dropped > 0 && a.repeats_dropped == b.repeats_dropped);
        assert_eq!(la[0].frame, GAME_START_FRAME);
        assert!(la.windows(2).all(|w| w[1].frame == w[0].frame + w[0].delay));
    }

    #[test]
    fn repeated_teacher_commands_are_thinned_041_r22() {
        assert_eq!(REPEAT_WINDOW, *crate::label::DELAY_BUCKETS.last().unwrap());
        let cc_only = || vec![unit(10, 0, CC, (192, 176))];
        let frames: Vec<RawFrame> = teacher_frames(63).into_iter().map(|n| frame_at(n, cc_only(), 50)).collect();
        let mut s = Shadow::new(&info(), T0Build::TerranReact, uranium_scripted::MapHints::default(), 3, 63);
        frames.iter().for_each(|f| s.observe(f));
        let trains: Vec<u32> = s.cmds.iter().filter(|c| c.op == op::TRAIN).map(|c| c.frame).collect();
        assert_eq!(trains, vec![3, 35, 67]);
        assert_eq!((s.observations, s.repeats_dropped), (9, 6));
        let lab = Labeler::new().unwrap();
        let (labels, _) = s.labels(&lab, 100);
        let got = names(&labels);
        let kept: Vec<(u32, u32)> = labels.iter().filter(|l| got.iter().any(|(f, n)| *f == l.frame && n == "Train_SCV")).map(|l| (l.frame, l.delay)).collect();
        assert_eq!(kept, vec![(3, 32), (35, 32), (67, 32)]);
        assert!(!got.iter().any(|(f, n)| (3..67).contains(f) && n == "NoOp"), "{got:?}");
        let two = || vec![unit(10, 0, CC, (192, 176)), unit(11, 0, CC, (800, 800))];
        let mut s = Shadow::new(&info(), T0Build::TerranReact, uranium_scripted::MapHints::default(), 3, 7);
        s.observe(&frame_at(1, two(), 100));
        assert_eq!(s.cmds.iter().filter(|c| c.op == op::TRAIN).count(), 2);
        assert_eq!(s.repeats_dropped, 0);
    }

    const SCENARIO_LAST: u32 = 9_000;

    fn scenario(enemy: u16) -> Vec<RawFrame> {
        use uranium_scripted::types::DEPOT;
        teacher_frames(SCENARIO_LAST)
            .into_iter()
            .map(|n| {
                let scvs = (4 + n / 300).min(20);
                let mut v = vec![unit(10, 0, CC, (192, 176))];
                for i in 0..scvs {
                    let fl = if i + 1 == scvs { flags::COMPLETED | flags::IDLE } else { flags::COMPLETED | flags::GATHERING };
                    v.push(UnitState { flags: fl, ..unit(1 + i, 0, SCV, (200 + 8 * (i % 6) as u16, 230 + 8 * (i / 6) as u16)) });
                }
                v.extend((0..4).map(|i| unit(50 + i, 11, MINERAL_FIELDS[0], (100 + 32 * i as u16, 60))));
                if n >= 1_500 {
                    v.push(unit(20, 0, DEPOT, (320, 96)));
                }
                if n >= 2_500 {
                    let fl = if n >= 3_500 { flags::COMPLETED } else { 0 };
                    v.push(UnitState { flags: fl, ..unit(21, 0, BARRACKS, (352, 256)) });
                }
                let marines = if n > 3_500 { ((n - 3_500) / 400).min(10) } else { 0 };
                v.extend((0..marines).map(|i| unit(100 + i, 0, MARINE, (300, 260 + 8 * i as u16))));
                if (4_000..4_600).contains(&n) {
                    v.extend((0..4).map(|i| unit(200 + i, 1, enemy, (260 + 32 * i as u16, 360))));
                }
                let mut f = frame_at(n, v, 50 + (n % 400) as i32);
                f.self_state.supply_used = (2 * (scvs + marines)) as u16;
                f.self_state.supply_total = 20 + 16 * (n / 2_000) as u16;
                f
            })
            .collect()
    }

    fn shadow_output(build: T0Build, info: &uranium_proto::GameInfo, enemy: u16) -> (String, Vec<(u32, String)>) {
        let mut s = Shadow::new(info, build, uranium_scripted::MapHints::default(), 3, SCENARIO_LAST);
        scenario(enemy).iter().for_each(|f| s.observe(f));
        let (labels, cov) = s.labels(&Labeler::new().unwrap(), SCENARIO_LAST + 64);
        let text = format!("{:?}|{:?}|{}|{}|{}|{}|{}|{:?}|{:?}", s.cmds, labels, s.repeats_dropped, s.observations, cov.commands, cov.labeled, cov.unknown, cov.unknown_ops, cov.unknown_orders);
        (text, names(&labels))
    }

    fn fnv1a(b: &[u8]) -> u64 {
        b.iter().fold(0xcbf2_9ce4_8422_2325, |h, &x| (h ^ x as u64).wrapping_mul(0x0100_0000_01b3))
    }

    const REACT_GOLDEN: u64 = 0xcd6e_6f30_2e17_9b1f;

    #[test]
    fn terran_react_relabel_output_unchanged_041_r22_022_r9() {
        let (text, got) = shadow_output(T0Build::TerranReact, &info(), uranium_scripted::types::ZERGLING);
        let kinds: std::collections::BTreeSet<&str> = got.iter().map(|(_, n)| n.as_str()).collect();
        eprintln!("react golden {:#x} labels {} kinds {kinds:?}", fnv1a(text.as_bytes()), got.len());
        for k in ["Build_SupplyDepot", "Build_Barracks", "Build_Bunker", "Train_SCV", "Train_Marine", "Gather_Unit", "Hold"] {
            assert!(kinds.contains(k), "{k} missing: {kinds:?}");
        }
        assert_eq!(fnv1a(text.as_bytes()), REACT_GOLDEN);
    }

    fn info_vs(race: uranium_proto::Race) -> uranium_proto::GameInfo {
        let mut i = info();
        i.players[1].race = race;
        i
    }

    #[test]
    fn terran_tp_shadow_labels_a_tvp_game_022_r9() {
        use uranium_proto::Race;
        use uranium_scripted::types::{GEYSER, ZEALOT};
        let with_geyser = |build| {
            let mut s = Shadow::new(&info_vs(Race::Protoss), build, uranium_scripted::MapHints::default(), 3, SCENARIO_LAST);
            for mut f in scenario(ZEALOT) {
                f.units.push(unit(60, 11, GEYSER, (96, 320)));
                s.observe(&f);
            }
            let (labels, _) = s.labels(&Labeler::new().unwrap(), SCENARIO_LAST + 64);
            (format!("{:?}|{:?}|{}", s.cmds, labels, s.repeats_dropped), names(&labels))
        };
        let (a, got) = with_geyser(T0Build::TerranTp);
        assert_eq!(a, with_geyser(T0Build::TerranTp).0, "same observations, same labels (022-R1)");
        let allowed = [
            "NoOp", "Build_SupplyDepot", "Build_Barracks", "Build_Refinery", "Build_Factory", "Build_CommandCenter", "Build_Bunker", "BuildAddon_MachineShop", "Research_TankSiege",
            "Train_SCV", "Train_Marine", "Train_SiegeTank", "SiegeMode", "TankMode", "Move_Pos", "Load_Unit", "Hold", "Attack_Pos", "Attack_Unit", "Repair_Unit", "Gather_Unit", "Smart_Unit",
            "ReturnCargo",
        ];
        let off: Vec<&(u32, String)> = got.iter().filter(|(_, n)| !allowed.contains(&n.as_str())).collect();
        assert!(off.is_empty(), "labels outside 022 모델 표현: {off:?}");
        assert!(got.iter().any(|(_, n)| n == "Build_Refinery"), "the 022-R3 refinery: {got:?}");
        let (react, react_names) = with_geyser(T0Build::TerranReact);
        assert_ne!(a, react, "not the 021-R8 teacher's labels");
        assert!(!react_names.iter().any(|(_, n)| n == "Build_Refinery"));
    }

    #[test]
    fn terran_tp_shadow_against_zerg_is_the_react_teacher_022_ac3() {
        assert_eq!(teacher_build("terran_tp").unwrap(), T0Build::TerranTp);
        assert!(teacher_build("standard").is_err() && teacher_build("terran_drop").is_err());
        let zerg = info_vs(uranium_proto::Race::Zerg);
        assert_eq!(
            shadow_output(T0Build::TerranTp, &zerg, uranium_scripted::types::ZERGLING),
            shadow_output(T0Build::TerranReact, &zerg, uranium_scripted::types::ZERGLING),
            "a lobby Zerg turns the 022 build into the 021-R8 teacher"
        );
    }

    #[test]
    fn teacher_frames_follow_the_live_cadence_041_r22() {
        assert_eq!(teacher_frames(30), vec![1, 7, 15, 23]);
        assert_eq!(teacher_frames(23), vec![1, 7, 15, 23]);
        assert_eq!(teacher_frames(0), vec![1]);
        assert_eq!((live_frame(1), live_frame(7), live_frame(15), live_frame(9), live_frame(3)), (Some(1), Some(9), Some(17), None, None));
        assert!(teacher_build("zerg_rush").is_err());
        assert_eq!(teacher_build("terran_react").unwrap(), T0Build::TerranReact);
    }

    fn dec(live: u32, action: u16, units: Vec<u32>) -> LoggedDecision {
        (live, MemoryAction { action, delay_bucket: 1, units })
    }

    #[test]
    fn decisions_move_to_playback_frames_041_r28() {
        for f in teacher_frames(400) {
            assert_eq!(playback_frame(live_frame(f).unwrap()), f);
        }
        assert_eq!([1, 2, 3, 4, 9, 100].map(playback_frame), [1, 1, 1, 2, 7, 98]);
    }

    #[test]
    fn step_memory_takes_the_half_open_interval_041_r28() {
        assert_eq!([1, 2, 11, 19].map(live_of_playback), [1, 4, 13, 21]);
        let log = vec![dec(1, 1, vec![]), dec(9, 2, vec![5]), dec(13, 3, vec![]), dec(20, 4, vec![]), dec(21, 5, vec![6, 7]), dec(40, 6, vec![])];
        let got = step_memory(&log, &[1, 11, 19, 35]);
        let actions: Vec<Vec<u16>> = got.iter().map(|v| v.iter().map(|a| a.action).collect()).collect();
        assert_eq!(actions, vec![vec![], vec![1, 2], vec![3, 4], vec![5]]);
        assert_eq!(got[3][0].units, vec![6, 7]);
        assert!(!got[1].iter().any(|a| a.action == 3) && got[2][0].action == 3);
        let early = vec![dec(12, 3, vec![])];
        assert_eq!(step_memory(&early, &[1, 11, 19]).iter().map(Vec::len).collect::<Vec<_>>(), vec![0, 1, 0]);
        assert_eq!(step_memory(&log, &[1, 2, 3]), vec![vec![], vec![log[0].1.clone()], vec![]]);
        assert!(step_memory(&[], &[1, 9]).iter().all(Vec::is_empty));
    }

    #[test]
    fn decision_log_completeness_041_r28() {
        let d = |f: u32| format!(r#"{{"frame":{f},"action":4,"delay_bucket":2,"units":[3,9]}}"#);
        let end = |n: u64| format!(r#"{{"decisions":{n},"end_frame":900,"player":1}}"#);
        let ok = format!("{}\n{}\n{}\n", d(1), d(9), end(2));
        let DecisionLogRead::Complete(log) = parse_decision_log(&ok) else { panic!("complete") };
        assert_eq!(log.decisions, vec![(1, MemoryAction { action: 4, delay_bucket: 2, units: vec![3, 9] }), (9, MemoryAction { action: 4, delay_bucket: 2, units: vec![3, 9] })]);
        assert_eq!((log.end_frame, log.player), (Some(900), 1));
        assert_eq!(parse_decision_log(&end(0)), DecisionLogRead::Complete(ModelLog { decisions: vec![], end_frame: Some(900), player: 1 }));
        for bad in [
            format!("{}\n{}\n", d(1), d(9)),
            format!("{}\n{}\n", d(1), end(2)),
            format!("{}\n{}\n{}\n", d(9), d(1), end(2)),
            format!("{}\nnot json\n{}\n", d(1), end(1)),
            format!("{}\n{}\n{}\n", d(1), end(1), d(9)),
            format!("{}\n{}\n", d(1), r#"{"decisions":1,"end_frame":900}"#),
        ] {
            assert!(matches!(parse_decision_log(&bad), DecisionLogRead::Incomplete(_)), "{bad}");
        }
        assert_eq!(read_decision_log(std::path::Path::new("no/such/g1.jsonl")), DecisionLogRead::Missing);
    }

    #[test]
    fn decision_log_binds_to_player_and_game_041_r28() {
        let read = |end_frame, player| DecisionLogRead::Complete(ModelLog { decisions: vec![dec(1, 0, vec![])], end_frame, player });
        assert!(bind_log(&read(Some(14_400), 0), 0, 14_398).is_ok());
        assert!(bind_log(&read(Some(14_400), 0), 0, 14_400 + MEMORY_END_TOLERANCE).is_ok());
        let other = bind_log(&read(Some(14_400), 1), 0, 14_400).unwrap_err();
        assert_eq!((other.status, other.decisions), ("other_player", 1));
        let far = bind_log(&read(Some(9_000), 0), 0, 14_400).unwrap_err();
        assert_eq!(far.status, "incomplete");
        assert!(far.reason.unwrap().contains("9000"));
        assert_eq!(bind_log(&read(None, 0), 0, 14_400).unwrap_err().status, "incomplete");
        assert_eq!(bind_log(&DecisionLogRead::Missing, 0, 14_400).unwrap_err().status, "missing");
    }

    #[test]
    fn unit_check_counts_and_one_percent_041_r28() {
        let own: HashMap<u32, std::collections::HashSet<u32>> = HashMap::from([(7, [5, 6].into()), (11, [5].into())]);
        let log = vec![dec(9, 1, vec![5, 6]), dec(13, 2, vec![5, 6]), dec(20, 3, vec![5]), dec(21, 0, vec![]), dec(100, 4, vec![99])];
        assert_eq!(unit_check(&log, 50, &own), (3, 2));
        assert_eq!(decision_frames(&log, 50), vec![7, 11, 18, 19]);
        assert_eq!(MemoryCheck::checked(200, 100, 1).status, "ok", "1% is allowed");
        assert_eq!(MemoryCheck::checked(200, 100, 2).status, "unit_mismatch");
        assert_eq!(MemoryCheck::checked(0, 0, 0).status, "ok");
        assert_eq!(MemoryCheck::without_log(&DecisionLogRead::Missing).status, "missing");
        assert_eq!(MemoryCheck::without_log(&DecisionLogRead::Incomplete("x".into())).reason.as_deref(), Some("x"));
    }

    #[test]
    fn teacher_commands_label_as_in_botgen_041_r22() {
        let enemy = unit(90, 1, 37, (900, 900));
        let f = frame(vec![
            unit(1, 0, SCV, (100, 100)),
            unit(2, 0, SCV, (120, 100)),
            unit(3, 0, MARINE, (140, 100)),
            unit(4, 0, MARINE, (160, 100)),
            unit(10, 0, CC, (200, 200)),
            unit(11, 0, BARRACKS, (300, 200)),
            unit(12, 0, BUNKER, (400, 300)),
            UnitState { flags: 0, ..unit(13, 0, BARRACKS, (500, 200)) },
            unit(50, 11, MINERAL_FIELDS[0], (64, 64)),
            enemy,
            unit(77, 0, SCV, (1, 1)),
        ]);
        let cmds = vec![
            Command::Build { worker: 1, unit_type: BUNKER, tile: (12, 9) },
            Command::Train { building: 10, unit_type: SCV },
            Command::Train { building: 11, unit_type: MARINE },
            Command::Unit { units: vec![2], order: CmdOrder::Move(Target::Pos((3000, 3000))), queued: false },
            Command::Unit { units: vec![3, 4], order: CmdOrder::Load(12), queued: false },
            Command::Unit { units: vec![3, 4], order: CmdOrder::Hold, queued: false },
            Command::Unit { units: vec![3, 4], order: CmdOrder::Attack(Target::Pos((800, 800))), queued: false },
            Command::Unit { units: vec![1, 2], order: CmdOrder::Attack(Target::Unit(90)), queued: false },
            Command::Unit { units: vec![1], order: CmdOrder::Repair(12), queued: false },
            Command::Unit { units: vec![2], order: CmdOrder::Gather(50), queued: false },
            Command::Unit { units: vec![1], order: CmdOrder::RightClick(Target::Unit(13)), queued: false },
            Command::Unit { units: vec![1], order: CmdOrder::ReturnCargo, queued: false },
            Command::Unit { units: vec![90, 999], order: CmdOrder::Stop, queued: false },
        ];
        let replay: Vec<Cmd> = cmds.iter().flat_map(|c| command_cmds(c, &f, 0, 3)).collect();
        assert!(replay.iter().all(|c| c.frame == 97 && c.player == 3));
        let lab = Labeler::new().unwrap();
        let units = own_units(&f, 0);
        let (labels, cov) = lab.label(&replay, 200, |_| &units);
        let table = uranium_env::actions::ActionTable::load().unwrap();
        type Row<'a> = (&'a str, Vec<u32>, Option<u32>, Option<(u16, u16)>);
        let acts: Vec<Row> =
            labels.iter().filter(|l| l.frame >= 97 && table.actions[l.action as usize].name != "NoOp").map(|l| (table.actions[l.action as usize].name.as_str(), l.selected.clone(), l.target_unit, l.target_tile)).collect();
        assert_eq!(
            acts,
            vec![
                ("Build_Bunker", vec![], None, Some((12, 9))),
                ("Train_SCV", vec![], None, None),
                ("Train_Marine", vec![], None, None),
                ("Move_Pos", vec![2], None, Some((93, 93))),
                ("Smart_Unit", vec![3, 4], Some(12), None),
                ("Hold", vec![3, 4], None, None),
                ("Attack_Pos", vec![3, 4], None, Some((25, 25))),
                ("Attack_Unit", vec![1, 2], Some(90), None),
                ("Repair_Unit", vec![1], Some(12), None),
                ("Gather_Unit", vec![2], Some(50), None),
                ("Smart_Unit", vec![1], Some(13), None),
                ("ReturnCargo", vec![1], None, None),
            ]
        );
        assert_eq!((cov.commands, cov.labeled, cov.unknown), (12, 12, 0));
        let f97: Vec<(u32, u32)> = labels.iter().filter(|l| (97..109).contains(&l.frame)).map(|l| (l.frame, l.delay)).collect();
        assert_eq!(f97.len(), 12);
        assert!(f97.iter().enumerate().all(|(i, &(f, d))| f == 97 + i as u32 && (d == 1 || i == 11)), "{f97:?}");
    }
}
