//! Label inversion (040-R2..R4, R11, 031-R11, 031-AC3): turns one player's replay commands into
//! 034 semantic actions with delay buckets. Selection is tracked from select / hotkey commands;
//! unit types come from the replay observation at the command frame (`Units`).

use std::collections::HashMap;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use uranium_env::actions::{ActionTable, Executor, TargetKind};

use crate::cmd::{Cmd, op};

/// 031 delay buckets in frames: one definition in `uranium-env` (E-7), re-exported here.
pub use uranium_env::features::DELAY_BUCKETS;
/// 031-R3: game selection limit.
pub const MAX_SELECT: usize = 12;
/// Hotkey groups 0..=9.
const GROUPS: usize = 10;
/// 040-R18: frame of the game-start observation, the first step of every sequence (the bridge's
/// first observation is frame 1, 011-R15).
pub const GAME_START_FRAME: u32 = 1;

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

/// One own unit as the game's selection rules see it (040-R3: OpenBW `actions.h`
/// `action_select` / `action_shift_select` / `action_control_group`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Own {
    pub unit_type: u16,
    /// `unit_can_be_multi_selected`: not a building, spider mine or nuke, not disabled
    /// (lockdown, stasis, maelstrom).
    pub multi_select: bool,
    /// Sprite hidden: loaded, or inside a refinery (`HarvestGas`).
    pub hidden: bool,
    /// Loading hides with deselect: the unit leaves every selection at once.
    pub loaded: bool,
    /// Lifted building: a right click moves it instead of setting the rally point.
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
    /// A unit of `unit_type` with no status effects.
    pub fn of_type(unit_type: u16) -> Self {
        Self::new(unit_type, false, false, false, false)
    }

    /// The labeled player's unit at the command frame.
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

/// What the labeler needs to know about units at a command frame.
pub trait Units {
    /// `tag` if it exists and belongs to the labeled player.
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

/// Unit types only (no status effects), for tests and simple callers.
impl Units for HashMap<u32, u16> {
    fn own(&self, tag: u32) -> Option<Own> {
        self.get(&tag).map(|&t| Own::of_type(t))
    }
}

/// `action_select` / `action_shift_select` admission: hidden units and nukes are never selected,
/// and after the first unit only multi-selectable units join.
fn admit(selection: &mut Vec<u32>, tag: u32, own: Option<Own>) {
    let Some(o) = own else { return };
    if o.unit_type == select_ids().nuke || o.hidden || selection.contains(&tag) || selection.len() >= MAX_SELECT {
        return;
    }
    if selection.is_empty() || o.multi_select {
        selection.push(tag);
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
    /// Unknown commands by opcode, for the coverage report.
    pub unknown_ops: Vec<(u8, u32)>,
    /// Targeted orders (0x15) without a 034 action, by order id (034-AC3 review).
    pub unknown_orders: Vec<(u8, u32)>,
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
        let mut unknown_orders: HashMap<u8, u32> = HashMap::new();
        let mut selection: Vec<u32> = Vec::new();
        let mut groups: HashMap<u8, Vec<u32>> = HashMap::new();
        let mut raws: Vec<Raw> = Vec::new();
        for c in cmds {
            let d = &c.data;
            // 040-R3: like the game (`on_unit_deselect`), the selection drops units that died or
            // were loaded; otherwise dead units fill the 12 slots and later additions are lost.
            // Hotkey groups keep the ids and drop missing / hidden units when recalled.
            let u = units(c.frame);
            selection.retain(|t| u.own(*t).is_some_and(|o| !o.loaded));
            let mut push = |action: Option<u16>, queued: bool, sel: bool, target_unit: Option<u32>, target_tile: Option<(u16, u16)>, cov: &mut Coverage| {
                cov.commands += 1;
                let Some(action) = action else {
                    cov.unknown += 1;
                    *unknown_ops.entry(c.op).or_insert(0) += 1;
                    return;
                };
                let a = &self.table.actions[action as usize];
                let selected = if sel && a.uses_select { selection.iter().copied().take(a.max_select as usize).collect() } else { Vec::new() };
                // 040-R4 (A-2): repeated commands are never merged; each one is a label (Train
                // presses and building cycling all reach the game).
                raws.push(Raw { frame: c.frame, action, queued: queued && a.uses_queued, selected, target_unit: target_unit.filter(|_| a.target == TargetKind::Unit), target_tile: target_tile.filter(|_| a.target == TargetKind::Loc) });
                cov.labeled += 1;
            };
            let pos_tile = |x: u16, y: u16| Some((x / 32, y / 32));
            match c.op {
                // 040-R3: selection commands follow the game's admission rules (`admit`).
                op::SELECT => {
                    selection.clear();
                    for t in tags(d) {
                        admit(&mut selection, t, u.own(t));
                    }
                }
                // A shift-add that would exceed 12 (counting every listed unit) is ignored whole.
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
                // 040-R11: hotkey assign / add update groups, recall replaces the selection.
                op::HOTKEY if d[1] as usize >= GROUPS => {}
                op::HOTKEY => match d[0] {
                    0 => {
                        groups.insert(d[1], selection.clone());
                    }
                    2 => hotkey_add(&mut groups, d[1], &selection, u),
                    _ => {
                        // Recall (`action_control_group` subaction 1): an empty group leaves the
                        // selection; missing, hidden and (in groups of several) non-multi-selectable
                        // units leave the group for good (swap with the last, then pop).
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
                    // 040-R11: right click with landed buildings selected sets the rally point; a
                    // lifted building moves (`get_default_order`).
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
        let sorted = |m: HashMap<u8, u32>| {
            let mut v: Vec<(u8, u32)> = m.into_iter().collect();
            v.sort();
            v
        };
        cov.unknown_ops = sorted(unknown_ops);
        cov.unknown_orders = sorted(unknown_orders);
        (with_delays(raws, end_frame, self.id("NoOp").unwrap_or(0)), cov)
    }

    /// Executor actions carry no selection (031-R1/R2): the executor picks the unit.
    pub fn uses_executor(&self, action: u16) -> bool {
        self.table.actions[action as usize].executor != Executor::None
    }
}

/// Hotkey group add (`action_control_group` subaction 2): refused when the group starts with a
/// live non-multi-selectable unit; after the first unit only multi-selectable units join. A unit
/// that would be the 13th lands in slot 0 of the next group (the original buffer overflow).
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

/// Building per the unit type table (E-13: `uranium-env` types, no `uranium-scripted`).
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

/// 031-R11 / 040-R4: every action gets the largest bucket <= the gap to the next action (at least
/// 1); the rest of the gap is filled with NoOps of the largest fitting bucket.
/// 040-R18: the sequence starts at the game-start observation; the wait until the first command
/// is NoOps the same way.
fn with_delays(raws: Vec<Raw>, end_frame: u32, noop: u16) -> Vec<Label> {
    let mut out = Vec::with_capacity(raws.len() + 1);
    let Some(first) = raws.first() else { return out };
    let mut frame = GAME_START_FRAME;
    noops(&mut out, &mut frame, first.frame, noop);
    for (i, r) in raws.iter().enumerate() {
        // Near commands are shifted to keep delay >= 1 (040-R4).
        frame = frame.max(r.frame);
        let next = raws.get(i + 1).map_or(end_frame, |n| n.frame).max(frame + 1);
        let delay = floor_bucket(next - frame);
        out.push(Label { frame, action: r.action, delay, queued: r.queued, selected: r.selected.clone(), target_unit: r.target_unit, target_tile: r.target_tile });
        frame += delay;
        noops(&mut out, &mut frame, next, noop);
    }
    out
}

/// NoOps of the largest fitting bucket from `frame` up to `next` (031-R11).
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
                // 040-R18: game start (frame 1) to the first command: 99 frames of NoOps.
                (1, 0, 32),
                (33, 0, 32),
                (65, 0, 32),
                (97, 0, 3),
                (100, 5, 8),
                (108, 0, 1),
                // 040-R4: the same-frame pair stays two labels, the second shifted to delay >= 1.
                (109, 5, 1),
                (110, 5, 32),
                (142, 0, 32),
                (174, 0, 6),
                (180, 5, 20)
            ]
        );
    }

    /// 040-R18 (E-2): the first step is the game-start observation (frame 1) even when the first
    /// command comes right away, and a command at frame 1 (or 0) is that first step itself.
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

    /// 040-R4 (A-2): repeated identical commands (Train presses, the same right click) are kept,
    /// one label each.
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
        // Every delay is at least 1 and the steps tile the game without gaps.
        assert!(labels.windows(2).all(|w| w[1].frame == w[0].frame + w[0].delay && w[0].delay >= 1));
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
            cmd(62, op::RIGHT_CLICK, vec![64, 0, 64, 0, 99, 0, 0xb0, 0, 0]),          // repeat: kept (040-R4)
            cmd(70, op::TRAIN, vec![0, 0]),                                           // marine
            cmd(71, 0x5c, vec![0; 81]),                                               // chat: ignored
            cmd(72, op::UNLOAD, vec![5, 0]),                                          // no 034 action
        ];
        let (labels, cov) = lab.label(&cmds, 100, |_| &units);
        let names: Vec<&str> = labels.iter().filter(|l| l.action != 0).map(|l| lab.table.actions[l.action as usize].name.as_str()).collect();
        // 040-R4: the repeated right click is a label of its own.
        assert_eq!(names, vec!["Attack_Pos", "SetRally_Pos", "Smart_Unit", "Smart_Unit", "Train_Marine"]);
        let smart = labels.iter().find(|l| lab.table.actions[l.action as usize].name == "Smart_Unit").unwrap();
        assert_eq!((smart.selected.clone(), smart.target_unit), (vec![10, 11], Some(99)));
        let attack = labels.iter().find(|l| lab.table.actions[l.action as usize].name == "Attack_Pos").unwrap();
        assert_eq!(attack.target_tile, Some((2, 3)));
        assert!(labels.iter().find(|l| lab.table.actions[l.action as usize].name == "Train_Marine").unwrap().selected.is_empty());
        assert_eq!((cov.commands, cov.labeled, cov.unknown), (6, 5, 1));
        assert_eq!(cov.unknown_ops, vec![(op::UNLOAD, 1)]);
    }

    #[test]
    fn medic_heal_move_on_ground_is_attack_move_040_r11() {
        let lab = Labeler::new().unwrap();
        let units: HashMap<u32, u16> = [(10, 0u16)].into_iter().collect();
        let cmds = vec![
            cmd(10, op::SELECT, vec![1, 10, 0]),
            cmd(11, op::TARGETED_ORDER, vec![64, 0, 96, 0, 0, 0, 0xe4, 0, 177, 0]), // HealMove, ground
            cmd(40, op::TARGETED_ORDER, vec![64, 0, 96, 0, 10, 0, 0xe4, 0, 177, 0]), // HealMove, unit
        ];
        let (labels, cov) = lab.label(&cmds, 100, |_| &units);
        let names: Vec<&str> = labels.iter().filter(|l| l.action != 0).map(|l| lab.table.actions[l.action as usize].name.as_str()).collect();
        assert_eq!(names, vec!["Attack_Pos", "Heal_Unit"]);
        assert!(cov.unknown_orders.is_empty());
    }

    #[test]
    fn dead_units_leave_selection_and_groups_040_r3() {
        let lab = Labeler::new().unwrap();
        // Twelve marines (tags 1..=12); tag 2 dies before frame 50, tag 13 is trained later.
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
            cmd(10, op::HOTKEY, vec![0, 1]),          // assign group 1 (full)
            cmd(50, op::HOTKEY, vec![1, 1]),          // recall: tag 2 is gone
            cmd(51, op::TARGETED_ORDER, attack.clone()),
            cmd(60, op::SELECT_ADD, vec![1, 13, 0]),  // the freed slot takes the new marine
            cmd(61, op::TARGETED_ORDER, attack),
        ];
        let (labels, _) = lab.label(&cmds, 100, |f| if f < 50 { &alive } else { &later });
        let attacks: Vec<&Label> = labels.iter().filter(|l| lab.table.actions[l.action as usize].name == "Attack_Pos").collect();
        // Recall pops the dead id by swapping in the last one (game order).
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
            cmd(10, op::SELECT, sel(&[1, 3, 2])),     // buildings do not join a multi-selection
            cmd(11, op::TARGETED_ORDER, ATTACK.to_vec()),
            cmd(20, op::SELECT, sel(&[3, 4, 1])),     // the first unit always enters, then multi-selectable only
            cmd(21, op::TARGETED_ORDER, ATTACK.to_vec()),
            cmd(30, op::SELECT, sel(&twelve)),        // 12 units
            cmd(31, op::SELECT_REMOVE, sel(&[20])),   // 11
            cmd(32, op::SELECT_ADD, sel(&[2, 21])),   // 11 + 2 > 12: ignored whole
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
        later.insert(2, scv(gas, false)); // inside a refinery: hidden, still selected
        let mut loaded = later.clone();
        loaded.insert(1, scv(0, true)); // loaded: leaves the selection at once
        let cmds = vec![
            cmd(10, op::SELECT, sel(&[3])),
            cmd(10, op::HOTKEY, vec![0, 4]),          // group 4 = barracks
            cmd(11, op::SELECT, sel(&[1, 2])),
            cmd(11, op::HOTKEY, vec![2, 4]),          // add to a building group: refused
            cmd(12, op::HOTKEY, vec![0, 1]),          // group 1 = both SCVs
            cmd(13, op::HOTKEY, vec![1, 4]),          // recall barracks only
            cmd(14, op::RIGHT_CLICK, vec![64, 0, 64, 0, 0, 0, 0xe4, 0, 0]),
            cmd(50, op::SELECT, sel(&[1, 2])),        // tag 2 is hidden: not selectable
            cmd(51, op::TARGETED_ORDER, ATTACK.to_vec()),
            cmd(60, op::HOTKEY, vec![1, 1]),          // recall drops hidden tag 2 from group 1
            cmd(61, op::TARGETED_ORDER, ATTACK.to_vec()),
            cmd(80, op::HOTKEY, vec![1, 1]),
            cmd(81, op::TARGETED_ORDER, ATTACK.to_vec()), // tag 1 loaded: nothing selected
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
