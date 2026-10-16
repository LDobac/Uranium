//! Label inversion (040-R2..R4, R11, 031-R11, 031-AC3): turns one player's replay commands into
//! 034 semantic actions with delay buckets. Selection is tracked from select / hotkey commands;
//! unit types come from the replay observation at the command frame (`Units`).

use std::collections::HashMap;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use uranium_env::actions::{ActionTable, Executor, TargetKind};

use crate::cmd::{Cmd, op};

/// 031: delay buckets in frames.
pub const DELAY_BUCKETS: [u32; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32];
/// 031-R3: game selection limit.
pub const MAX_SELECT: usize = 12;
/// 040-R4: commands at most this many frames apart count as "near"; identical near commands
/// are merged as spam.
pub const NEAR_FRAMES: u32 = 2;

pub const ORDERS_TOML: &str = include_str!("../tables/replay_orders.toml");

#[derive(Deserialize)]
struct OrderRow {
    #[allow(dead_code)]
    names: Vec<String>,
    ids: Vec<u8>,
    unit: String,
    pos: String,
}

#[derive(Deserialize)]
struct BuildOrders {
    place_building: u8,
    place_addon: Vec<u8>,
    building_land: u8,
}

#[derive(Deserialize)]
struct OrdersFile {
    order: Vec<OrderRow>,
    build: BuildOrders,
}

/// What the labeler needs to know about units at a command frame.
pub trait Units {
    /// Unit type of `tag` if it exists and belongs to the labeled player.
    fn own_type(&self, tag: u32) -> Option<u16>;
}

impl Units for HashMap<u32, u16> {
    fn own_type(&self, tag: u32) -> Option<u16> {
        self.get(&tag).copied()
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct Label {
    /// Frame of the decision (observation frame).
    pub frame: u32,
    pub action: u16,
    /// 031 delay bucket: frames until the next observation.
    pub delay: u32,
    pub queued: bool,
    pub selected: Vec<u32>,
    pub target_unit: Option<u32>,
    /// Build tile (top-left) or position / 32.
    pub target_tile: Option<(u16, u16)>,
}

/// Coverage counters for 031-AC3.
#[derive(Serialize, Debug, Clone, Default, PartialEq, Eq)]
pub struct Coverage {
    /// Game-affecting commands seen.
    pub commands: u32,
    pub labeled: u32,
    pub unknown: u32,
    pub merged_spam: u32,
    /// Unknown commands by opcode, for the coverage report.
    pub unknown_ops: Vec<(u8, u32)>,
}

pub struct Labeler {
    table: ActionTable,
    orders: HashMap<u8, (String, String)>,
    build: BuildOrders,
}

/// One labeled action before delays are assigned.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Raw {
    frame: u32,
    action: u16,
    queued: bool,
    selected: Vec<u32>,
    target_unit: Option<u32>,
    target_tile: Option<(u16, u16)>,
}

fn u16_at(d: &[u8], a: usize) -> u16 {
    u16::from_le_bytes([d[a], d[a + 1]])
}

fn valid_tag(t: u16) -> Option<u32> {
    (t != 0 && t != 0xffff).then_some(t as u32)
}

impl Labeler {
    pub fn new() -> Result<Self> {
        let f: OrdersFile = toml::from_str(ORDERS_TOML).context("replay_orders.toml")?;
        let table = ActionTable::load()?;
        let mut orders = HashMap::new();
        for row in f.order {
            for name in [&row.unit, &row.pos].into_iter().filter(|n| !n.is_empty()) {
                table.by_name(name).with_context(|| format!("replay_orders.toml: unknown action {name}"))?;
            }
            for id in row.ids {
                orders.insert(id, (row.unit.clone(), row.pos.clone()));
            }
        }
        Ok(Self { table, orders, build: f.build })
    }

    fn id(&self, name: &str) -> Option<u16> {
        self.table.by_name(name).map(|a| a.id)
    }

    fn by_arg(&self, command: &str, pick: impl Fn(&uranium_env::actions::Action) -> Option<u16>, value: u16) -> Option<u16> {
        self.table.actions.iter().find(|a| a.command == command && pick(a) == Some(value)).map(|a| a.id)
    }

    /// Labels the commands of one player (already in 1.16.1 form, 013). `units(frame)` returns the
    /// player's units at that frame. Returns the labels with delays (031-R11) and coverage.
    pub fn label<'a, U: Units>(&self, cmds: &[Cmd], end_frame: u32, mut units: impl FnMut(u32) -> &'a U) -> (Vec<Label>, Coverage)
    where
        U: 'a,
    {
        let mut cov = Coverage::default();
        let mut unknown_ops: HashMap<u8, u32> = HashMap::new();
        let mut selection: Vec<u32> = Vec::new();
        let mut groups: HashMap<u8, Vec<u32>> = HashMap::new();
        let mut raws: Vec<Raw> = Vec::new();
        for c in cmds {
            let d = &c.data;
            let mut push = |action: Option<u16>, queued: bool, sel: bool, target_unit: Option<u32>, target_tile: Option<(u16, u16)>, cov: &mut Coverage| {
                cov.commands += 1;
                let Some(action) = action else {
                    cov.unknown += 1;
                    *unknown_ops.entry(c.op).or_insert(0) += 1;
                    return;
                };
                let a = &self.table.actions[action as usize];
                let selected = if sel && a.uses_select { selection.iter().copied().take(a.max_select as usize).collect() } else { Vec::new() };
                let raw = Raw { frame: c.frame, action, queued: queued && a.uses_queued, selected, target_unit: target_unit.filter(|_| a.target == TargetKind::Unit), target_tile: target_tile.filter(|_| a.target == TargetKind::Loc) };
                // 040-R4: identical near commands are spam.
                if raws.last().is_some_and(|p| c.frame - p.frame <= NEAR_FRAMES && Raw { frame: p.frame, ..raw.clone() } == *p) {
                    cov.merged_spam += 1;
                } else {
                    raws.push(raw);
                }
                cov.labeled += 1;
            };
            let pos_tile = |x: u16, y: u16| Some((x / 32, y / 32));
            match c.op {
                op::SELECT => selection = tags(d),
                op::SELECT_ADD => {
                    for t in tags(d) {
                        if !selection.contains(&t) && selection.len() < MAX_SELECT {
                            selection.push(t);
                        }
                    }
                }
                op::SELECT_REMOVE => {
                    let rm = tags(d);
                    selection.retain(|t| !rm.contains(t));
                }
                // 040-R11: hotkey assign / add update groups, recall replaces the selection.
                op::HOTKEY => match d[0] {
                    0 => {
                        groups.insert(d[1], selection.clone());
                    }
                    2 => {
                        let g = groups.entry(d[1]).or_default();
                        for t in &selection {
                            if !g.contains(t) && g.len() < MAX_SELECT {
                                g.push(*t);
                            }
                        }
                    }
                    _ => selection = groups.get(&d[1]).cloned().unwrap_or_default(),
                },
                op::RIGHT_CLICK => {
                    let (x, y, tag, q) = (u16_at(d, 0), u16_at(d, 2), valid_tag(u16_at(d, 4)), d[8] != 0);
                    let u = units(c.frame);
                    // 040-R11: right click with buildings selected sets the rally point.
                    let buildings = !selection.is_empty() && selection.iter().all(|t| u.own_type(*t).is_some_and(uranium_scripted::types::is_building));
                    let name = match (buildings, tag) {
                        (true, Some(_)) => "SetRally_Unit",
                        (true, None) => "SetRally_Pos",
                        (false, Some(_)) => "Smart_Unit",
                        (false, None) => "Smart_Pos",
                    };
                    push(self.id(name), q, true, tag, pos_tile(x, y), &mut cov);
                }
                op::TARGETED_ORDER => {
                    let (x, y, tag, order, q) = (u16_at(d, 0), u16_at(d, 2), valid_tag(u16_at(d, 4)), d[8], d[9] != 0);
                    let name = self.orders.get(&order).map(|(unit, pos)| if tag.is_some() && !unit.is_empty() { unit } else { pos }).filter(|n| !n.is_empty());
                    push(name.and_then(|n| self.id(n)), q, true, tag, pos_tile(x, y), &mut cov);
                }
                op::BUILD => {
                    let (order, x, y, ty) = (d[0], u16_at(d, 1), u16_at(d, 3), u16_at(d, 5));
                    let tile = Some((x, y));
                    let action = if order == self.build.place_building {
                        self.by_arg("Build", |a| a.unit_type, ty)
                    } else if self.build.place_addon.contains(&order) {
                        self.by_arg("BuildAddon", |a| a.unit_type, ty)
                    } else if order == self.build.building_land {
                        self.id(if ty == uranium_scripted::types::CC { "Land_CommandCenter_Pos" } else { "Land_Pos" })
                    } else {
                        None
                    };
                    push(action, false, true, None, tile, &mut cov);
                }
                op::TRAIN => push(self.by_arg("Train", |a| a.unit_type, u16_at(d, 0)), false, false, None, None, &mut cov),
                0x30 => push(self.by_arg("Research", |a| a.tech, d[0] as u16), false, false, None, None, &mut cov),
                0x32 => push(self.by_arg("Upgrade", |a| a.upgrade, d[0] as u16), false, false, None, None, &mut cov),
                0x1a => push(self.id("Stop"), d[0] != 0, true, None, None, &mut cov),
                0x2b => push(self.id("Hold"), d[0] != 0, true, None, None, &mut cov),
                0x1e => push(self.id("ReturnCargo"), d[0] != 0, true, None, None, &mut cov),
                0x28 => push(self.id("UnloadAll"), d[0] != 0, true, None, None, &mut cov),
                0x26 => push(self.id("SiegeMode"), false, true, None, None, &mut cov),
                0x25 => push(self.id("TankMode"), false, true, None, None, &mut cov),
                0x21 => push(self.id("Cloak"), false, true, None, None, &mut cov),
                0x22 => push(self.id("Decloak"), false, true, None, None, &mut cov),
                0x36 => push(self.id("Stim"), false, true, None, None, &mut cov),
                0x2f => push(self.id("Lift"), false, true, None, None, &mut cov),
                op::CANCEL_TRAIN => push(self.id("Cancel_Train"), false, true, None, None, &mut cov),
                0x18 => push(self.id("Cancel_Construction"), false, true, None, None, &mut cov),
                0x31 => push(self.id("Cancel_Research"), false, true, None, None, &mut cov),
                0x33 => push(self.id("Cancel_Upgrade"), false, true, None, None, &mut cov),
                0x34 => push(self.id("Cancel_Addon"), false, true, None, None, &mut cov),
                // 040-R11: no effect on the game state (chat, ping, camera-like, network, speed).
                0x05 | 0x0d | 0x0e | 0x0f | 0x10 | 0x11 | 0x37..=0x45 | 0x48 | 0x54..=0x58 | 0x5c => {}
                // Game-affecting commands without a 034 action (e.g. single-unit unload).
                _ => push(None, false, false, None, None, &mut cov),
            }
        }
        cov.unknown_ops = {
            let mut v: Vec<(u8, u32)> = unknown_ops.into_iter().collect();
            v.sort();
            v
        };
        (with_delays(raws, end_frame, self.id("NoOp").unwrap_or(0)), cov)
    }

    /// Executor actions carry no selection (031-R1/R2): the executor picks the unit.
    pub fn uses_executor(&self, action: u16) -> bool {
        self.table.actions[action as usize].executor != Executor::None
    }
}

fn tags(d: &[u8]) -> Vec<u32> {
    (0..d[0] as usize).filter_map(|i| valid_tag(u16_at(d, 1 + 2 * i))).collect()
}

fn floor_bucket(gap: u32) -> u32 {
    DELAY_BUCKETS.iter().rev().copied().find(|&b| b <= gap).unwrap_or(1)
}

/// 031-R11 / 040-R4: every action gets the largest bucket <= the gap to the next action (at least
/// 1); the rest of the gap is filled with NoOps of the largest fitting bucket.
fn with_delays(raws: Vec<Raw>, end_frame: u32, noop: u16) -> Vec<Label> {
    let mut out = Vec::with_capacity(raws.len());
    let mut frame = raws.first().map_or(0, |r| r.frame);
    for (i, r) in raws.iter().enumerate() {
        // Near commands are shifted to keep delay >= 1 (040-R4).
        frame = frame.max(r.frame);
        let next = raws.get(i + 1).map_or(end_frame, |n| n.frame).max(frame + 1);
        let delay = floor_bucket(next - frame);
        out.push(Label { frame, action: r.action, delay, queued: r.queued, selected: r.selected.clone(), target_unit: r.target_unit, target_tile: r.target_tile });
        frame += delay;
        while frame < next {
            let d = floor_bucket(next - frame);
            out.push(Label { frame, action: noop, delay: d, queued: false, selected: vec![], target_unit: None, target_tile: None });
            frame += d;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cmd(frame: u32, op: u8, data: Vec<u8>) -> Cmd {
        Cmd { frame, player: 0, op, data }
    }

    #[test]
    fn delays_and_noops_follow_031_r11() {
        let r = |frame| Raw { frame, action: 5, queued: false, selected: vec![], target_unit: None, target_tile: None };
        let l = with_delays(vec![r(100), r(109), r(109), r(180)], 200, 0);
        let got: Vec<(u32, u16, u32)> = l.iter().map(|l| (l.frame, l.action, l.delay)).collect();
        assert_eq!(
            got,
            vec![(100, 5, 8), (108, 0, 1), (109, 5, 1), (110, 5, 32), (142, 0, 32), (174, 0, 6), (180, 5, 20)]
        );
    }

    #[test]
    fn selection_hotkeys_and_rally_040_r3_r11() {
        let lab = Labeler::new().unwrap();
        let units: HashMap<u32, u16> = [(10, 0u16), (11, 0), (20, 111)].into_iter().collect();
        let cmds = vec![
            cmd(10, op::SELECT, vec![2, 10, 0, 11, 0]),
            cmd(10, op::HOTKEY, vec![0, 1]),                                          // assign group 1
            cmd(12, op::TARGETED_ORDER, vec![64, 0, 96, 0, 0, 0, 0xe4, 0, 14, 0]),     // attack-move
            cmd(40, op::SELECT, vec![1, 20, 0]),                                     // barracks
            cmd(41, op::RIGHT_CLICK, vec![64, 0, 64, 0, 0, 0, 0xe4, 0, 0]),           // rally
            cmd(60, op::HOTKEY, vec![1, 1]),                                          // recall group 1
            cmd(61, op::RIGHT_CLICK, vec![64, 0, 64, 0, 99, 0, 0xb0, 0, 0]),          // smart on unit 99
            cmd(62, op::RIGHT_CLICK, vec![64, 0, 64, 0, 99, 0, 0xb0, 0, 0]),          // spam
            cmd(70, op::TRAIN, vec![0, 0]),                                           // marine
            cmd(71, 0x5c, vec![0; 81]),                                               // chat: ignored
            cmd(72, op::UNLOAD, vec![5, 0]),                                          // no 034 action
        ];
        let (labels, cov) = lab.label(&cmds, 100, |_| &units);
        let names: Vec<&str> = labels.iter().filter(|l| l.action != 0).map(|l| lab.table.actions[l.action as usize].name.as_str()).collect();
        assert_eq!(names, vec!["Attack_Pos", "SetRally_Pos", "Smart_Unit", "Train_Marine"]);
        let smart = labels.iter().find(|l| lab.table.actions[l.action as usize].name == "Smart_Unit").unwrap();
        assert_eq!((smart.selected.clone(), smart.target_unit), (vec![10, 11], Some(99)));
        assert_eq!(labels[0].target_tile, Some((2, 3)));
        assert!(labels.iter().find(|l| lab.table.actions[l.action as usize].name == "Train_Marine").unwrap().selected.is_empty());
        assert_eq!((cov.commands, cov.labeled, cov.unknown, cov.merged_spam), (6, 5, 1, 1));
        assert_eq!(cov.unknown_ops, vec![(op::UNLOAD, 1)]);
    }
}
