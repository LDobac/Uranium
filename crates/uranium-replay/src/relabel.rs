//! 041-R22 DAgger relabelling (040 출처 `dagger`): the model's own games are played back and the
//! 021-R8 teacher is run as a shadow on the model's (Terran) observations. The teacher sees every
//! observation a live T0 session would (live frame 1, then every 8 frames, `Act.advance_frames`;
//! `live_frame` maps them to playback frames) and its commands become the labels, at the frames a
//! live game records them; the model's own commands are only used for the 040-R8 checks.
//!
//! The teacher's 010 `Command`s are turned into the 1.16.1 replay commands BWAPI would send for
//! them (`GameImpl::executeCommand` with command optimizer level 0, as in the OpenBW bridge:
//! a select of the units, then the order), and those go through the same `Labeler` as every
//! replay (040-R3, R4, R11): one teacher command is one label, several commands at one
//! observation are consecutive labels with delay >= 1 (040-R4), and the time without commands is
//! NoOp labels of the largest fitting delay (031-R11), exactly as in a teacher botgen game.
//! No randomness: the same replay gives the same labels.

use std::collections::HashMap;

use anyhow::{Result, bail};
use uranium_proto::{CancelKind, CmdOrder, Command, Pos, RawFrame, Target, Tile, UnitId, UnitState};
use uranium_scripted::T0Build;
use uranium_scripted::types::{BUNKER, DROPSHIP, OVERLORD, building_tile, is_building};

use crate::cmd::{Cmd, op};
use crate::label::GAME_START_FRAME;

/// Teacher builds `uranium-replay build --relabel` accepts (041-R22): the 021-R8 teacher only.
pub fn teacher_build(name: &str) -> Result<T0Build> {
    match name {
        "terran_react" => Ok(T0Build::TerranReact),
        other => bail!("--relabel {other}: only terran_react (the 021-R8 teacher) relabels (041-R22)"),
    }
}

/// Frames between the observations of a live T0 session (`uranium_scripted` `STEP_FRAMES`, the
/// `advance_frames` of every teacher `Act`): live frames 1, 9, 17, ...
pub const TEACHER_STEP: u32 = 8;
/// A live observation of frame `L` shows the game as the replay playback shows it at frame
/// `L - LIVE_LAG`, and the commands decided on it are recorded at frame `L + RECORD_DELAY` (OpenBW
/// lockstep, measured 2026-11-07 on a teacher botgen game: shadow observations at `L - 2` give the
/// recorded commands at `L + 2`; at `L` the shadow saw each new SCV one observation early).
pub const LIVE_LAG: u32 = 2;
pub const RECORD_DELAY: u32 = 2;

/// The live frame number of the shadow observation at playback frame `f`, if `f` is one: the
/// game-start observation (011-R15, live frame 1, the earliest playback frame) and `L - LIVE_LAG`
/// for the later live frames `L` = 9, 17, ...
pub fn live_frame(f: u32) -> Option<u32> {
    if f == GAME_START_FRAME {
        return Some(GAME_START_FRAME);
    }
    let l = f + LIVE_LAG;
    (l > GAME_START_FRAME && (l - GAME_START_FRAME).is_multiple_of(TEACHER_STEP)).then_some(l)
}

/// The playback frames of the shadow observations up to playback frame `last` (inclusive).
pub fn teacher_frames(last: u32) -> Vec<u32> {
    (GAME_START_FRAME..=last.max(GAME_START_FRAME)).filter(|&f| live_frame(f).is_some()).collect()
}

/// Replay opcode for a teacher command BWAPI's path is not modelled for (none of the 021-R8
/// teacher's commands): the labeler makes it an `unknown` label (weight 0, counted in coverage).
pub const UNMODELED_OP: u8 = 0xff;
/// Unit type field of a targeted order without a type (BW `UnitTypes::None`).
const NO_TYPE: u16 = 228;
/// BWAPI `cancelTrain(-2)`: the last queue slot.
const LAST_SLOT: u16 = 254;
const SHUTTLE: u16 = 72;

fn order(name: &str) -> u8 {
    uranium_env::types::order_id(name).unwrap_or_else(|| panic!("order {name}")) as u8
}

fn le(v: u16) -> [u8; 2] {
    v.to_le_bytes()
}

/// `TargetedOrder` (0x15) payload: position, target tag, unit type, order, queued.
fn targeted(order: u8, p: Pos, tag: u32, queued: bool) -> (u8, Vec<u8>) {
    let mut d = Vec::with_capacity(10);
    d.extend(le(p.0));
    d.extend(le(p.1));
    d.extend(le(tag as u16));
    d.extend(le(NO_TYPE));
    d.extend([order, queued as u8]);
    (op::TARGETED_ORDER, d)
}

/// `RightClick` (0x14) payload: position, target tag, unit type, queued.
fn right_click(p: Pos, tag: u32, queued: bool) -> (u8, Vec<u8>) {
    let mut d = Vec::with_capacity(9);
    d.extend(le(p.0));
    d.extend(le(p.1));
    d.extend(le(tag as u16));
    d.extend(le(NO_TYPE));
    d.push(queued as u8);
    (op::RIGHT_CLICK, d)
}

/// `Build` (0x0C) payload: order, tile, unit type.
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

/// The replay commands (select, then order) BWAPI sends for the teacher's command `c` issued at
/// observation `f` of player `me` (BWAPI id) whose replay command id is `player`. Units the bridge
/// would not find (not own, not observed) are left out like the bridge rejects them; a command
/// left without units sends nothing. Buildings in a unit command get it one by one (BWAPI
/// `canIssueCommandGrouped`), the other units as one group (the bridge's `Game::issueCommand`).
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

/// The order command of a unit command for its first unit `u` (BWAPI `executeCommand`).
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
        // BWAPI Load: a bunker picks up (PickupBunker), a transport picks up (PickupTransport),
        // a unit loading into a bunker / transport right-clicks it.
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
        // Stim packs have their own opcode; other spells are not 021-R8 teacher commands.
        CmdOrder::UseTech { tech, target: None } if Some(*tech as u32) == stim() => (0x36, vec![]),
        CmdOrder::UseTech { .. } => (UNMODELED_OP, vec![]),
    })
}

fn stim() -> Option<u32> {
    uranium_env::types::TypeTables::get().techs.iter().find(|t| t.name == "Stim_Packs").map(|t| t.id as u32)
}

/// Own units of `me` at an observation for the labeler (040-R3 selection rules).
pub fn own_units(f: &RawFrame, me: u8) -> HashMap<u32, crate::label::Own> {
    f.units.iter().filter(|u| u.owner == me).map(|u| (u.id, crate::label::Own::from_state(u))).collect()
}

/// The teacher run as a shadow on one perspective's observations (041-R22). Built from the
/// bridge's GameInfo and the map precompute hints like a live T0 session
/// (`uranium_agent::ScriptedPolicy`); `observe` gets every observation of the playback in order
/// and acts only on the live session's frames (`teacher_frames`) up to `last`.
pub struct Shadow {
    bot: Box<dyn uranium_scripted::Bot>,
    me: u8,
    player: u8,
    last: u32,
    /// The teacher's commands as replay commands, in order.
    pub cmds: Vec<Cmd>,
    /// Own units at each frame the teacher issued commands (labeler selection rules).
    pub units_at: HashMap<u32, HashMap<u32, crate::label::Own>>,
    /// Teacher observations (diagnostics).
    pub observations: u32,
}

impl Shadow {
    /// `player`: the perspective's replay command id; `last`: the last frame the teacher is fed.
    pub fn new(info: &uranium_proto::GameInfo, build: T0Build, hints: uranium_scripted::MapHints, player: u8, last: u32) -> Self {
        Self { bot: uranium_scripted::for_game_hints(info, build, hints), me: info.self_player, player, last, cmds: Vec::new(), units_at: HashMap::new(), observations: 0 }
    }

    /// One observation of the playback; frames that are not a live teacher observation are skipped.
    /// The teacher sees the live frame number (its timers and rule boundaries count live frames)
    /// and its commands are placed where a live game records them (`RECORD_DELAY`).
    pub fn observe(&mut self, f: &RawFrame) {
        let Some(live) = live_frame(f.frame).filter(|_| f.frame <= self.last) else { return };
        self.observations += 1;
        let mut seen = f.clone();
        seen.frame = live;
        let act = self.bot.act(&seen);
        let at = live + RECORD_DELAY;
        let start = self.cmds.len();
        for c in &act.cmds {
            self.cmds.extend(command_cmds(c, &seen, self.me, self.player).into_iter().map(|c| Cmd { frame: at, ..c }));
        }
        if self.cmds.len() > start {
            self.units_at.insert(at, own_units(f, self.me));
        }
    }

    /// The teacher's labels (040-R4 delays, 031-R11 NoOps, 040-R18 first step) and coverage.
    pub fn labels(&self, labeler: &crate::label::Labeler, end_frame: u32) -> (Vec<crate::label::Label>, crate::label::Coverage) {
        let empty = HashMap::new();
        labeler.label(&self.cmds, end_frame, |f| self.units_at.get(&f).unwrap_or(&empty))
    }

    /// 021-R8 record of the shadow teacher (`scripted.react`, diagnostics).
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

    /// A Terran start: command center, four idle SCVs, minerals.
    fn start_units() -> Vec<UnitState> {
        let idle = |id| UnitState { flags: flags::COMPLETED | flags::IDLE, ..unit(id, 0, SCV, (200, 230)) };
        let mut v = vec![unit(10, 0, CC, (192, 176)), idle(1), idle(2), idle(3), idle(4)];
        v.extend((0..4).map(|i| unit(50 + i, 11, MINERAL_FIELDS[0], (100 + 32 * i as u16, 60))));
        v
    }

    fn names(lab: &Labeler, labels: &[crate::label::Label]) -> Vec<(u32, String)> {
        let table = uranium_env::actions::ActionTable::load().unwrap();
        let _ = lab;
        labels.iter().map(|l| (l.frame, table.actions[l.action as usize].name.clone())).collect()
    }

    /// 041-R22: the shadow acts only at the live session's observations (playback frames 1, 7, 15,
    /// ... up to `last`), its labels are the teacher's commands (one label each, 040-R4) where a
    /// live game records them (live frame + 2), and the same observations give the same labels.
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
        let got = names(&lab, &la);
        // Frame 1: the teacher trains an SCV and sends the four idle SCVs to mine, one label each.
        let first: Vec<&str> = got.iter().filter(|(f, n)| *f < 9 && n != "NoOp").map(|(_, n)| n.as_str()).collect();
        assert_eq!(first.iter().filter(|n| **n == "Gather_Unit").count(), 4, "{got:?}");
        assert_eq!(first.iter().filter(|n| **n == "Train_SCV").count(), 1, "{got:?}");
        // Every teacher command is recorded 2 frames after its live observation (1, 9, 17, 25).
        assert!(a.cmds.iter().all(|c| [3, 11, 19, 27].contains(&c.frame)));
        assert!(a.cmds.iter().any(|c| c.frame == 3) && a.cmds.iter().any(|c| c.frame == 27));
        // The steps tile the game from the game-start observation.
        assert_eq!(la[0].frame, GAME_START_FRAME);
        assert!(la.windows(2).all(|w| w[1].frame == w[0].frame + w[0].delay));
    }

    /// The live T0 session's observation frames 1, 9, 17, ... (011-R15, `Act.advance_frames` 8)
    /// are the playback frames 1, 7, 15, ... (`LIVE_LAG`).
    #[test]
    fn teacher_frames_follow_the_live_cadence_041_r22() {
        assert_eq!(teacher_frames(30), vec![1, 7, 15, 23]);
        assert_eq!(teacher_frames(23), vec![1, 7, 15, 23]);
        assert_eq!(teacher_frames(0), vec![1]);
        assert_eq!((live_frame(1), live_frame(7), live_frame(15), live_frame(9), live_frame(3)), (Some(1), Some(9), Some(17), None, None));
        assert!(teacher_build("zerg_rush").is_err());
        assert_eq!(teacher_build("terran_react").unwrap(), T0Build::TerranReact);
    }

    /// 041-R22: every 021-R8 teacher command maps to the 034 action of its "모델 표현" row
    /// through the same labeler as a botgen replay (BWAPI's replay commands).
    #[test]
    fn teacher_commands_label_as_in_botgen_041_r22() {
        let enemy = unit(90, 1, 37, (900, 900)); // a zergling
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
            unit(77, 0, SCV, (1, 1)), // not referenced
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
            // Units the bridge would not find send nothing.
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
                // Build is an executor action: no selection in the label (031-R1, as in botgen).
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
        // 040-R4: the 12 commands of one observation are consecutive labels, delay 1 up to the last.
        let f97: Vec<(u32, u32)> = labels.iter().filter(|l| (97..109).contains(&l.frame)).map(|l| (l.frame, l.delay)).collect();
        assert_eq!(f97.len(), 12);
        assert!(f97.iter().enumerate().all(|(i, &(f, d))| f == 97 + i as u32 && (d == 1 || i == 11)), "{f97:?}");
    }
}
