use std::collections::HashMap;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use uranium_env::actions::{ActionTable, Executor, TargetKind};

use crate::cmd::{Cmd, op};

pub use uranium_env::features::DELAY_BUCKETS;
pub const MAX_SELECT: usize = 12;
const GROUPS: usize = 10;
pub const GAME_START_FRAME: u32 = 1;

pub const ORDERS_TOML: &str = include_str!("../tables/replay_orders.toml");

pub const LABELER_VERSION: u32 = 2;

pub use uranium_env::features::label::UNKNOWN_ACTION;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Own {
    pub unit_type: u16,
    pub multi_select: bool,
    pub hidden: bool,
    pub loaded: bool,
    pub lifted: bool,
}

struct SelectIds {
    spider_mine: u16,
    nuke: u16,
    harvest_gas: u16,
}

fn select_ids() -> &'static SelectIds {
    static S: std::sync::OnceLock<SelectIds> = std::sync::OnceLock::new();
    S.get_or_init(|| {
        let t = uranium_env::types::TypeTables::get();
        let unit = |n: &str| t.unit_by_name(n).unwrap_or_else(|| panic!("unit type {n}")).id as u16;
        SelectIds {
            spider_mine: unit("Terran_Vulture_Spider_Mine"),
            nuke: unit("Terran_Nuclear_Missile"),
            harvest_gas: uranium_env::types::order_id("HarvestGas").expect("order HarvestGas"),
        }
    })
}

impl Own {
    pub fn of_type(unit_type: u16) -> Self {
        Self::new(unit_type, false, false, false, false)
    }

    pub fn from_state(u: &uranium_proto::UnitState) -> Self {
        use uranium_proto::flags;
        let s = &u.status_timers;
        let loaded = u.flags & flags::LOADED != 0 || u.loaded_in.is_some();
        let disabled = s.lockdown > 0 || s.stasis > 0 || s.maelstrom > 0;
        Self::new(u.unit_type, disabled, loaded || u.order == select_ids().harvest_gas, loaded, u.flags & flags::LIFTED != 0)
    }

    fn new(unit_type: u16, disabled: bool, hidden: bool, loaded: bool, lifted: bool) -> Self {
        let ids = select_ids();
        let building = uranium_env::types::TypeTables::get().unit(unit_type).is_some_and(|t| t.is_building());
        let multi_select = !building && !disabled && unit_type != ids.spider_mine && unit_type != ids.nuke;
        Self { unit_type, multi_select, hidden, loaded, lifted }
    }
}

pub trait Units {
    fn own(&self, tag: u32) -> Option<Own>;

    fn own_type(&self, tag: u32) -> Option<u16> {
        self.own(tag).map(|o| o.unit_type)
    }
}

impl Units for HashMap<u32, Own> {
    fn own(&self, tag: u32) -> Option<Own> {
        self.get(&tag).copied()
    }
}

impl Units for HashMap<u32, u16> {
    fn own(&self, tag: u32) -> Option<Own> {
        self.get(&tag).map(|&t| Own::of_type(t))
    }
}

fn admit(selection: &mut Vec<u32>, tag: u32, own: Option<Own>) {
    let Some(o) = own else { return };
    if o.unit_type == select_ids().nuke || o.hidden || selection.contains(&tag) || selection.len() >= MAX_SELECT {
        return;
    }
    if selection.is_empty() || o.multi_select {
        selection.push(tag);
    }
}

pub use uranium_env::features::label::Label;

#[derive(Serialize, Debug, Clone, Default, PartialEq, Eq)]
pub struct Coverage {
    pub commands: u32,
    pub labeled: u32,
    pub unknown: u32,
    pub unknown_ops: Vec<(u8, u32)>,
    pub unknown_orders: Vec<(u8, u32)>,
}

pub struct Labeler {
    table: ActionTable,
    orders: HashMap<u8, (String, String)>,
    build: BuildOrders,
}

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

    pub fn label<'a, U: Units>(&self, cmds: &[Cmd], end_frame: u32, mut units: impl FnMut(u32) -> &'a U) -> (Vec<Label>, Coverage)
    where
        U: 'a,
    {
        let mut cov = Coverage::default();
        let mut unknown_ops: HashMap<u8, u32> = HashMap::new();
        let mut unknown_orders: HashMap<u8, u32> = HashMap::new();
        let mut selection: Vec<u32> = Vec::new();
        let mut groups: HashMap<u8, Vec<u32>> = HashMap::new();
        let mut raws: Vec<Raw> = Vec::new();
        for c in cmds {
            let d = &c.data;
            let u = units(c.frame);
            selection.retain(|t| u.own(*t).is_some_and(|o| !o.loaded));
            let mut push = |action: Option<u16>, queued: bool, sel: bool, target_unit: Option<u32>, target_tile: Option<(u16, u16)>, cov: &mut Coverage| {
                cov.commands += 1;
                let Some(action) = action else {
                    cov.unknown += 1;
                    *unknown_ops.entry(c.op).or_insert(0) += 1;
                    raws.push(Raw { frame: c.frame, action: UNKNOWN_ACTION, queued: false, selected: Vec::new(), target_unit: None, target_tile: None });
                    return;
                };
                let a = &self.table.actions[action as usize];
                let selected = if sel && a.uses_select { selection.clone() } else { Vec::new() };
                raws.push(Raw { frame: c.frame, action, queued: queued && a.uses_queued, selected, target_unit: target_unit.filter(|_| a.target == TargetKind::Unit), target_tile: target_tile.filter(|_| a.target == TargetKind::Loc) });
                cov.labeled += 1;
            };
            let pos_tile = |x: u16, y: u16| Some((x / 32, y / 32));
            match c.op {
                op::SELECT => {
                    selection.clear();
                    for t in tags(d) {
                        admit(&mut selection, t, u.own(t));
                    }
                }
                op::SELECT_ADD if selection.len() + d[0] as usize <= MAX_SELECT => {
                    for t in tags(d) {
                        admit(&mut selection, t, u.own(t));
                    }
                }
                op::SELECT_ADD => {}
                op::SELECT_REMOVE => {
                    let rm = tags(d);
                    selection.retain(|t| !rm.contains(t));
                }
                op::HOTKEY if d[1] as usize >= GROUPS => {}
                op::HOTKEY => match d[0] {
                    0 => {
                        groups.insert(d[1], selection.clone());
                    }
                    2 => hotkey_add(&mut groups, d[1], &selection, u),
                    _ => {
                        if let Some(g) = groups.get_mut(&d[1]).filter(|g| !g.is_empty()) {
                            selection.clear();
                            let mut i = 0;
                            while i < g.len() {
                                let ok = u.own(g[i]).is_some_and(|o| !o.hidden && (g.len() == 1 || o.multi_select));
                                if ok {
                                    selection.push(g[i]);
                                    i += 1;
                                } else {
                                    g.swap_remove(i);
                                }
                            }
                        }
                    }
                },
                op::RIGHT_CLICK => {
                    let (x, y, tag, q) = (u16_at(d, 0), u16_at(d, 2), valid_tag(u16_at(d, 4)), d[8] != 0);
                    let buildings = !selection.is_empty() && selection.iter().all(|t| u.own(*t).is_some_and(|o| is_building(o.unit_type) && !o.lifted));
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
                    if name.is_none() {
                        *unknown_orders.entry(order).or_insert(0) += 1;
                    }
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
                        self.id(if ty == command_center() { "Land_CommandCenter_Pos" } else { "Land_Pos" })
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
                op::UNLOAD => push(self.id("Unload_Unit"), false, true, valid_tag(u16_at(d, 0)), None, &mut cov),
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
                0x05 | 0x0d | 0x0e | 0x0f | 0x10 | 0x11 | 0x37..=0x45 | 0x48 | 0x54..=0x58 | 0x5c => {}
                _ => push(None, false, false, None, None, &mut cov),
            }
        }
        let sorted = |m: HashMap<u8, u32>| {
            let mut v: Vec<(u8, u32)> = m.into_iter().collect();
            v.sort();
            v
        };
        cov.unknown_ops = sorted(unknown_ops);
        cov.unknown_orders = sorted(unknown_orders);
        (with_delays(raws, end_frame, self.id("NoOp").unwrap_or(0)), cov)
    }

    pub fn uses_executor(&self, action: u16) -> bool {
        self.table.actions[action as usize].executor != Executor::None
    }
}

fn hotkey_add<U: Units>(groups: &mut HashMap<u8, Vec<u32>>, n: u8, selection: &[u32], u: &U) {
    let g = groups.entry(n).or_default();
    if g.first().and_then(|&t| u.own(t)).is_some_and(|o| !o.multi_select) {
        return;
    }
    let mut overflow = None;
    for &t in selection {
        let Some(o) = u.own(t) else { continue };
        if !(g.is_empty() || o.multi_select) || g.contains(&t) {
            continue;
        }
        if g.len() == MAX_SELECT {
            overflow = Some(t);
            break;
        }
        g.push(t);
        if g.len() == MAX_SELECT {
            break;
        }
    }
    if let Some(t) = overflow.filter(|_| (n as usize) + 1 < GROUPS) {
        let next = groups.entry(n + 1).or_default();
        match next.first_mut() {
            Some(first) => *first = t,
            None => next.push(t),
        }
    }
}

fn tags(d: &[u8]) -> Vec<u32> {
    (0..d[0] as usize).filter_map(|i| valid_tag(u16_at(d, 1 + 2 * i))).collect()
}

fn is_building(unit_type: u16) -> bool {
    uranium_env::types::TypeTables::get().unit(unit_type).is_some_and(|t| t.is_building())
}

fn unit_id(name: &str) -> u16 {
    uranium_env::types::TypeTables::get().unit_by_name(name).unwrap_or_else(|| panic!("unit type {name}")).id as u16
}

fn command_center() -> u16 {
    static CC: std::sync::OnceLock<u16> = std::sync::OnceLock::new();
    *CC.get_or_init(|| unit_id("Terran_Command_Center"))
}

fn floor_bucket(gap: u32) -> u32 {
    DELAY_BUCKETS.iter().rev().copied().find(|&b| b <= gap).unwrap_or(1)
}

fn with_delays(raws: Vec<Raw>, end_frame: u32, noop: u16) -> Vec<Label> {
    let mut out = Vec::with_capacity(raws.len() + 1);
    let Some(first) = raws.first() else { return out };
    let mut frame = GAME_START_FRAME;
    noops(&mut out, &mut frame, first.frame, noop);
    for (i, r) in raws.iter().enumerate() {
        frame = frame.max(r.frame);
        let next = raws.get(i + 1).map_or(end_frame, |n| n.frame).max(frame + 1);
        let delay = floor_bucket(next - frame);
        out.push(Label { frame, action: r.action, delay, queued: r.queued, selected: r.selected.clone(), target_unit: r.target_unit, target_tile: r.target_tile });
        frame += delay;
        noops(&mut out, &mut frame, next, noop);
    }
    out
}

fn noops(out: &mut Vec<Label>, frame: &mut u32, next: u32, noop: u16) {
    while *frame < next {
        let d = floor_bucket(next - *frame);
        out.push(Label { frame: *frame, action: noop, delay: d, queued: false, selected: vec![], target_unit: None, target_tile: None });
        *frame += d;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cmd(frame: u32, op: u8, data: Vec<u8>) -> Cmd {
        Cmd { frame, player: 0, op, data }
    }

    fn raw(frame: u32) -> Raw {
        Raw { frame, action: 5, queued: false, selected: vec![], target_unit: None, target_tile: None }
    }

    #[test]
    fn delays_and_noops_follow_031_r11_040_r4() {
        let l = with_delays(vec![raw(100), raw(109), raw(109), raw(180)], 200, 0);
        let got: Vec<(u32, u16, u32)> = l.iter().map(|l| (l.frame, l.action, l.delay)).collect();
        assert_eq!(
            got,
            vec![
                (1, 0, 32),
                (33, 0, 32),
                (65, 0, 32),
                (97, 0, 3),
                (100, 5, 8),
                (108, 0, 1),
                (109, 5, 1),
                (110, 5, 32),
                (142, 0, 32),
                (174, 0, 6),
                (180, 5, 20)
            ]
        );
    }

    #[test]
    fn first_step_is_game_start_040_r18() {
        let l = with_delays(vec![raw(2), raw(40)], 60, 0);
        assert_eq!(l.iter().map(|l| (l.frame, l.action, l.delay)).take(2).collect::<Vec<_>>(), vec![(1, 0, 1), (2, 5, 32)]);
        let l = with_delays(vec![raw(1), raw(5)], 10, 0);
        assert_eq!((l[0].frame, l[0].action, l[0].delay), (1, 5, 4));
        let l = with_delays(vec![raw(0), raw(5)], 10, 0);
        assert_eq!((l[0].frame, l[0].action), (GAME_START_FRAME, 5));
        assert!(with_delays(vec![], 100, 0).is_empty());
    }

    #[test]
    fn repeated_commands_are_not_merged_040_r4() {
        let lab = Labeler::new().unwrap();
        let units: HashMap<u32, u16> = [(20, unit_id("Terran_Barracks")), (21, unit_id("Terran_Barracks")), (10, 0)].into_iter().collect();
        let cmds = vec![
            cmd(10, op::SELECT, sel(&[20])),
            cmd(11, op::TRAIN, vec![0, 0]),
            cmd(12, op::TRAIN, vec![0, 0]),
            cmd(12, op::TRAIN, vec![0, 0]),
            cmd(13, op::SELECT, sel(&[21])),
            cmd(13, op::TRAIN, vec![0, 0]),
            cmd(20, op::SELECT, sel(&[10])),
            cmd(21, op::RIGHT_CLICK, vec![64, 0, 64, 0, 0, 0, 0xe4, 0, 0]),
            cmd(22, op::RIGHT_CLICK, vec![64, 0, 64, 0, 0, 0, 0xe4, 0, 0]),
        ];
        let (labels, cov) = lab.label(&cmds, 100, |_| &units);
        let names: Vec<&str> = labels.iter().filter(|l| l.action != 0).map(|l| lab.table.actions[l.action as usize].name.as_str()).collect();
        assert_eq!(names, vec!["Train_Marine", "Train_Marine", "Train_Marine", "Train_Marine", "Smart_Pos", "Smart_Pos"]);
        assert_eq!((cov.commands, cov.labeled), (6, 6));
        assert!(labels.windows(2).all(|w| w[1].frame == w[0].frame + w[0].delay && w[0].delay >= 1));
    }

    #[test]
    fn selection_hotkeys_and_rally_040_r3_r11() {
        let lab = Labeler::new().unwrap();
        let units: HashMap<u32, u16> = [(10, 0u16), (11, 0), (20, 111)].into_iter().collect();
        let cmds = vec![
            cmd(10, op::SELECT, vec![2, 10, 0, 11, 0]),
            cmd(10, op::HOTKEY, vec![0, 1]),
            cmd(12, op::TARGETED_ORDER, vec![64, 0, 96, 0, 0, 0, 0xe4, 0, 14, 0]),
            cmd(40, op::SELECT, vec![1, 20, 0]),
            cmd(41, op::RIGHT_CLICK, vec![64, 0, 64, 0, 0, 0, 0xe4, 0, 0]),
            cmd(60, op::HOTKEY, vec![1, 1]),
            cmd(61, op::RIGHT_CLICK, vec![64, 0, 64, 0, 99, 0, 0xb0, 0, 0]),
            cmd(62, op::RIGHT_CLICK, vec![64, 0, 64, 0, 99, 0, 0xb0, 0, 0]),
            cmd(70, op::TRAIN, vec![0, 0]),
            cmd(71, 0x5c, vec![0; 81]),
            cmd(72, 0x2d, vec![0]),
        ];
        let (labels, cov) = lab.label(&cmds, 100, |_| &units);
        let names: Vec<&str> = labels.iter().filter(|l| l.action != 0).map(|l| name_of(&lab, l)).collect();
        assert_eq!(names, vec!["Attack_Pos", "SetRally_Pos", "Smart_Unit", "Smart_Unit", "Train_Marine", "Unknown"]);
        let smart = labels.iter().find(|l| lab.table.actions[l.action as usize].name == "Smart_Unit").unwrap();
        assert_eq!((smart.selected.clone(), smart.target_unit), (vec![10, 11], Some(99)));
        let attack = labels.iter().find(|l| lab.table.actions[l.action as usize].name == "Attack_Pos").unwrap();
        assert_eq!(attack.target_tile, Some((2, 3)));
        assert!(labels.iter().find(|l| lab.table.actions[l.action as usize].name == "Train_Marine").unwrap().selected.is_empty());
        assert_eq!((cov.commands, cov.labeled, cov.unknown), (6, 5, 1));
        assert_eq!(cov.unknown_ops, vec![(0x2d, 1)]);
    }

    fn name_of<'a>(lab: &'a Labeler, l: &Label) -> &'a str {
        if l.is_unknown() { "Unknown" } else { lab.table.actions[l.action as usize].name.as_str() }
    }

    #[test]
    fn unknown_commands_are_labels_031_ac3() {
        let lab = Labeler::new().unwrap();
        let units: HashMap<u32, u16> = [(10, 0u16)].into_iter().collect();
        let guard = vec![64, 0, 96, 0, 0, 0, 0xe4, 0, 2, 0];
        let cmds = vec![cmd(10, op::SELECT, sel(&[10])), cmd(20, op::TARGETED_ORDER, guard), cmd(40, op::TARGETED_ORDER, ATTACK.to_vec())];
        let (labels, cov) = lab.label(&cmds, 100, |_| &units);
        let got: Vec<(u32, &str, u32)> = labels.iter().map(|l| (l.frame, name_of(&lab, l), l.delay)).collect();
        assert_eq!(got, vec![(1, "NoOp", 16), (17, "NoOp", 3), (20, "Unknown", 20), (40, "Attack_Pos", 32), (72, "NoOp", 28)]);
        let unknown = &labels[2];
        assert_eq!((unknown.action, unknown.selected.is_empty(), unknown.target_unit, unknown.target_tile, unknown.queued), (UNKNOWN_ACTION, true, None, None, false));
        assert_eq!((cov.commands, cov.labeled, cov.unknown, cov.unknown_orders.clone()), (2, 1, 1, vec![(2, 1)]));
        assert!(labels.windows(2).all(|w| w[1].frame == w[0].frame + w[0].delay));
    }

    #[test]
    fn single_unload_and_order_mappings_f7() {
        let lab = Labeler::new().unwrap();
        let ship = unit_id("Terran_Dropship");
        let units: HashMap<u32, Own> = [
            (30, Own::of_type(ship)),
            (5, Own::from_state(&uranium_proto::UnitState { unit_type: 0, flags: uranium_proto::flags::LOADED, loaded_in: Some(30), ..Default::default() })),
            (40, Own::of_type(unit_id("Terran_Battlecruiser"))),
            (50, Own::of_type(unit_id("Terran_Missile_Turret"))),
            (60, Own::of_type(unit_id("Terran_Bunker"))),
        ]
        .into_iter()
        .collect();
        let order = |o: u8, tag: u8| vec![64, 0, 96, 0, tag, 0, 0xe4, 0, o, 0];
        let cmds = vec![
            cmd(10, op::SELECT, sel(&[30])),
            cmd(11, op::UNLOAD, vec![5, 0]),
            cmd(20, op::SELECT, sel(&[40])),
            cmd(21, op::TARGETED_ORDER, order(114, 99)),
            cmd(30, op::SELECT, sel(&[50])),
            cmd(31, op::TARGETED_ORDER, order(19, 99)),
            cmd(32, op::TARGETED_ORDER, order(19, 0)),
            cmd(33, op::TARGETED_ORDER, order(18, 0)),
            cmd(40, op::SELECT, sel(&[60])),
            cmd(41, op::TARGETED_ORDER, order(95, 7)),
        ];
        let (labels, cov) = lab.label(&cmds, 100, |_| &units);
        let acts: Vec<(&str, Vec<u32>, Option<u32>)> = labels.iter().filter(|l| l.action != 0).map(|l| (name_of(&lab, l), l.selected.clone(), l.target_unit)).collect();
        assert_eq!(
            acts,
            vec![
                ("Unload_Unit", vec![30], Some(5)),
                ("Yamato_Unit", vec![40], Some(99)),
                ("Attack_Unit", vec![50], Some(99)),
                ("Unknown", vec![], None),
                ("Unknown", vec![], None),
                ("Unknown", vec![], None),
            ]
        );
        assert_eq!((cov.commands, cov.labeled, cov.unknown), (6, 3, 3));
        assert_eq!(cov.unknown_orders, vec![(18, 1), (19, 1), (95, 1)]);
    }

    #[test]
    fn max_select_one_keeps_whole_selection_f12() {
        let lab = Labeler::new().unwrap();
        let vulture = unit_id("Terran_Vulture");
        let units: HashMap<u32, u16> = (1..=3).map(|t| (t, vulture)).collect();
        let mine = vec![64, 0, 96, 0, 0, 0, 0xe4, 0, 20, 0];
        let cmds = vec![cmd(10, op::SELECT, sel(&[1, 2, 3])), cmd(11, op::TARGETED_ORDER, mine)];
        let (labels, _) = lab.label(&cmds, 100, |_| &units);
        assert_eq!(selected_of(&lab, &labels, "SpiderMine_Pos"), vec![vec![1, 2, 3]]);
    }

    #[test]
    fn medic_heal_move_on_ground_is_attack_move_040_r11() {
        let lab = Labeler::new().unwrap();
        let units: HashMap<u32, u16> = [(10, 0u16)].into_iter().collect();
        let cmds = vec![
            cmd(10, op::SELECT, vec![1, 10, 0]),
            cmd(11, op::TARGETED_ORDER, vec![64, 0, 96, 0, 0, 0, 0xe4, 0, 177, 0]),
            cmd(40, op::TARGETED_ORDER, vec![64, 0, 96, 0, 10, 0, 0xe4, 0, 177, 0]),
        ];
        let (labels, cov) = lab.label(&cmds, 100, |_| &units);
        let names: Vec<&str> = labels.iter().filter(|l| l.action != 0).map(|l| lab.table.actions[l.action as usize].name.as_str()).collect();
        assert_eq!(names, vec!["Attack_Pos", "Heal_Unit"]);
        assert!(cov.unknown_orders.is_empty());
    }

    #[test]
    fn dead_units_leave_selection_and_groups_040_r3() {
        let lab = Labeler::new().unwrap();
        let alive: HashMap<u32, u16> = (1..=12).map(|t| (t, 0u16)).collect();
        let mut later = alive.clone();
        later.remove(&2);
        later.insert(13, 0);
        let mut sel12 = vec![12u8];
        for t in 1..=12u8 {
            sel12.extend([t, 0]);
        }
        let attack = vec![64, 0, 96, 0, 0, 0, 0xe4, 0, 14, 0];
        let cmds = vec![
            cmd(10, op::SELECT, sel12),
            cmd(10, op::HOTKEY, vec![0, 1]),
            cmd(50, op::HOTKEY, vec![1, 1]),
            cmd(51, op::TARGETED_ORDER, attack.clone()),
            cmd(60, op::SELECT_ADD, vec![1, 13, 0]),
            cmd(61, op::TARGETED_ORDER, attack),
        ];
        let (labels, _) = lab.label(&cmds, 100, |f| if f < 50 { &alive } else { &later });
        let attacks: Vec<&Label> = labels.iter().filter(|l| lab.table.actions[l.action as usize].name == "Attack_Pos").collect();
        let expect: Vec<u32> = [1, 12].into_iter().chain(3..=11).collect();
        assert_eq!(attacks[0].selected, expect);
        assert_eq!(attacks[1].selected, expect.iter().copied().chain([13]).collect::<Vec<_>>());
    }

    fn selected_of(lab: &Labeler, labels: &[Label], name: &str) -> Vec<Vec<u32>> {
        labels.iter().filter(|l| lab.table.actions[l.action as usize].name == name).map(|l| l.selected.clone()).collect()
    }

    fn sel(tags: &[u8]) -> Vec<u8> {
        let mut d = vec![tags.len() as u8];
        for &t in tags {
            d.extend([t, 0]);
        }
        d
    }

    const ATTACK: [u8; 10] = [64, 0, 96, 0, 0, 0, 0xe4, 0, 14, 0];

    #[test]
    fn selection_follows_game_admission_040_r3() {
        let lab = Labeler::new().unwrap();
        let units: HashMap<u32, u16> = [(1, 0u16), (2, 0), (3, unit_id("Terran_Barracks")), (4, unit_id("Terran_Supply_Depot"))].into_iter().collect();
        let mut twelve: Vec<u8> = (20..31).collect();
        twelve.push(1);
        let mut many: HashMap<u32, u16> = units.clone();
        many.extend((20..31).map(|t| (t, 0u16)));
        let cmds = vec![
            cmd(10, op::SELECT, sel(&[1, 3, 2])),
            cmd(11, op::TARGETED_ORDER, ATTACK.to_vec()),
            cmd(20, op::SELECT, sel(&[3, 4, 1])),
            cmd(21, op::TARGETED_ORDER, ATTACK.to_vec()),
            cmd(30, op::SELECT, sel(&twelve)),
            cmd(31, op::SELECT_REMOVE, sel(&[20])),
            cmd(32, op::SELECT_ADD, sel(&[2, 21])),
            cmd(33, op::TARGETED_ORDER, ATTACK.to_vec()),
        ];
        let (labels, _) = lab.label(&cmds, 100, |_| &many);
        let mut expect_last: Vec<u32> = (21..31).collect();
        expect_last.push(1);
        assert_eq!(selected_of(&lab, &labels, "Attack_Pos"), vec![vec![1, 2], vec![3, 1], expect_last]);
    }

    #[test]
    fn hotkey_groups_follow_game_rules_040_r3() {
        let lab = Labeler::new().unwrap();
        let scv = |order: u16, loaded: bool| {
            let mut u = uranium_proto::UnitState { unit_type: unit_id("Terran_SCV"), order, ..Default::default() };
            if loaded {
                u.flags |= uranium_proto::flags::LOADED;
            }
            Own::from_state(&u)
        };
        let gas = uranium_env::types::order_id("HarvestGas").unwrap();
        let before: HashMap<u32, Own> = [(1, scv(0, false)), (2, scv(0, false)), (3, Own::of_type(unit_id("Terran_Barracks")))].into_iter().collect();
        let mut later = before.clone();
        later.insert(2, scv(gas, false));
        let mut loaded = later.clone();
        loaded.insert(1, scv(0, true));
        let cmds = vec![
            cmd(10, op::SELECT, sel(&[3])),
            cmd(10, op::HOTKEY, vec![0, 4]),
            cmd(11, op::SELECT, sel(&[1, 2])),
            cmd(11, op::HOTKEY, vec![2, 4]),
            cmd(12, op::HOTKEY, vec![0, 1]),
            cmd(13, op::HOTKEY, vec![1, 4]),
            cmd(14, op::RIGHT_CLICK, vec![64, 0, 64, 0, 0, 0, 0xe4, 0, 0]),
            cmd(50, op::SELECT, sel(&[1, 2])),
            cmd(51, op::TARGETED_ORDER, ATTACK.to_vec()),
            cmd(60, op::HOTKEY, vec![1, 1]),
            cmd(61, op::TARGETED_ORDER, ATTACK.to_vec()),
            cmd(80, op::HOTKEY, vec![1, 1]),
            cmd(81, op::TARGETED_ORDER, ATTACK.to_vec()),
        ];
        let (labels, _) = lab.label(&cmds, 100, |f| if f < 50 { &before } else if f < 80 { &later } else { &loaded });
        assert_eq!(selected_of(&lab, &labels, "SetRally_Pos"), vec![vec![3]]);
        assert_eq!(selected_of(&lab, &labels, "Attack_Pos"), vec![vec![1], vec![1], vec![]]);
    }

    #[test]
    fn lifted_building_right_click_moves_040_r11() {
        let lab = Labeler::new().unwrap();
        let barracks = |lifted: bool| {
            let flags = if lifted { uranium_proto::flags::LIFTED } else { 0 };
            Own::from_state(&uranium_proto::UnitState { unit_type: unit_id("Terran_Barracks"), flags, ..Default::default() })
        };
        let landed: HashMap<u32, Own> = [(3, barracks(false))].into_iter().collect();
        let lifted: HashMap<u32, Own> = [(3, barracks(true))].into_iter().collect();
        let cmds = vec![
            cmd(10, op::SELECT, sel(&[3])),
            cmd(11, op::RIGHT_CLICK, vec![64, 0, 64, 0, 0, 0, 0xe4, 0, 0]),
            cmd(50, op::RIGHT_CLICK, vec![96, 0, 64, 0, 0, 0, 0xe4, 0, 0]),
        ];
        let (labels, _) = lab.label(&cmds, 100, |f| if f < 50 { &landed } else { &lifted });
        let names: Vec<&str> = labels.iter().filter(|l| l.action != 0).map(|l| lab.table.actions[l.action as usize].name.as_str()).collect();
        assert_eq!(names, vec!["SetRally_Pos", "Smart_Pos"]);
    }

    #[test]
    fn hotkey_add_overflow_spills_into_next_group_040_r3() {
        let mut groups: HashMap<u8, Vec<u32>> = [(3, (1..=12).collect()), (4, vec![50, 51])].into_iter().collect();
        let units: HashMap<u32, u16> = (1..=13).chain([50, 51]).map(|t| (t, 0u16)).collect();
        hotkey_add(&mut groups, 3, &[13], &units);
        assert_eq!((groups[&3].len(), groups[&4].clone()), (12, vec![13, 51]));
    }
}
