//! 031-AC1 (C-5): every 034 action type is executed in a real OpenBW game and the game accepts it
//! (no `CommandFailed` for at least one execution, and every rejection is a justified one, 액션
//! 감사), Smart_Unit / Gather_Unit on a resource out
//! of vision (011-R20, 액션 감사 F2), and the Attack_Pos -> move substitution
//! for unarmed units (034 이동·전투 비고, AMB-041-1: the bridge sends a move, the unit's order
//! becomes Move).
//!
//! Needs a running `uranium-hostd` with OpenBW and Fighting Spirit, so it is ignored by default:
//!
//! ```text
//! URANIUM_TEST_HOSTD=<host:port> [URANIUM_TEST_AGENT_LISTEN=0.0.0.0:0] [URANIUM_AGENT_PUBLIC_ADDR=<addr>] \
//!   cargo test -p uranium-env --test openbw_actions -- --ignored --nocapture
//! ```
//!
//! Player 0 is a scripted Terran that walks the whole tech tree with one base and issues each
//! action type through the real masks and executor (031-R4) as soon as the type mask allows it.
//! Player 1 is an idle Terran that only provides enemy targets (lockdown, flare, yamato, attack).
//! Gas is saved by cancelling upgrades and unneeded units right after they start (which also
//! covers the Cancel_* types); damage for Repair_Unit comes from our own nuke on our Engineering
//! Bay, and Heal / Restoration targets from Stim and an Irradiate on our own vehicle. When every
//! type is covered the test stops the game through hostd (`StopGame`). The coverage table and the
//! replay (when hostd returns it) are written to `CARGO_TARGET_TMPDIR`.

mod common;

use std::collections::{HashMap, HashSet, VecDeque};
use std::io::BufReader;
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::time::Instant;

use common::{Role, valid_action};
use uranium_env::actions::{Action, CAPABILITIES};
use uranium_env::features::executor::{HostAction, execute};
use uranium_env::features::extract::action_table;
use uranium_env::features::world::{World, top_left};
use uranium_env::features::*;
use uranium_env::types::{TypeTables, UnitType, order_id};
use uranium_proto::control::{ControlRequest, ControlResponse, GameOutcome, GameSpec, MapKey, PlayerKind, PlayerSpec, CONTROL_VERSION};
use uranium_proto::*;

/// Frames between observations of the scripted player.
const STEP: u32 = 8;
/// 60 game minutes (about 4 minutes of wall time); one refinery makes the tech tree slow.
const MAX_FRAMES: u32 = 24 * 60 * 60;
/// Executions tried per action type before giving up on it.
const MAX_TRIES: u32 = 8;
/// Frames between two tries of the same action type.
const RETRY_FRAMES: u32 = 96;

#[derive(Default, Clone, Debug)]
struct Stat {
    /// Steps in which the type mask allowed the action.
    opened: u32,
    /// Commands sent.
    sent: u32,
    /// `CommandFailed` received for them (031-R10).
    failed: u32,
    /// Sent and not failed: the game accepted the command.
    ok: u32,
    /// The executor refused the chosen arguments (031-R4).
    rejected: u32,
    first_ok_frame: Option<u32>,
    last_try_frame: u32,
    errors: Vec<u16>,
}

/// One command of the last `Act`, in `cmd_index` order.
#[derive(Clone, Debug)]
struct Sent {
    action: u16,
    units: Vec<UnitId>,
    /// Attack_Pos with an unarmed unit: check its order next frame.
    unarmed: bool,
    /// Smart_Unit / Gather_Unit on a resource out of vision (011-R20, 액션 감사 F2).
    fog: bool,
}

#[derive(Clone, Debug)]
enum Follow {
    Stop(Vec<UnitId>),
    /// Queued move home after a cast far away.
    Home(UnitId),
    CancelUpgrade(UnitId),
    CancelTrain(UnitId),
    CancelResearch(UnitId),
    CancelAddon(UnitId),
}

struct Coverage {
    mem: AgentMemory,
    map: MapInfo,
    st: State,
}

struct State {
    info: GameInfo,
    stats: Vec<Stat>,
    last: Vec<Sent>,
    follow: VecDeque<Follow>,
    start: (i32, i32),
    home: Option<(i32, i32)>,
    enemy_starts: Vec<(i32, i32)>,
    main_cc: Option<UnitId>,
    scout: Option<UnitId>,
    scout_target: usize,
    scout_holding: bool,
    /// Scout route: a ring around our main (explored tiles for building), then enemy starts.
    waypoints: Vec<(i32, i32)>,
    gas: Vec<UnitId>,
    /// Tile each building was lifted from (land it back there).
    origin: HashMap<UnitId, (i32, i32)>,
    /// Frame of the last order per action (economy throttle).
    pending: HashMap<String, u32>,
    unarmed_pending: Option<UnitId>,
    /// (unit type, order after Attack_Pos) for the unarmed-unit check.
    unarmed: Vec<(String, u16)>,
    /// (action, SCV) accepted on a resource out of vision: its order is checked next frame.
    fog_pending: Option<(u16, UnitId)>,
    /// (action, order after it; None = CommandFailed) for resources out of vision (액션 감사 F2).
    fog: Vec<(String, Option<u16>)>,
    fog_next: u32,
    rotate: usize,
    frame: u32,
    log: Vec<String>,
    /// (frame, checksum of every observed unit) per observation: 011-AC4 determinism.
    sums: Vec<(u32, u64)>,
    end: Option<String>,
}

fn table() -> &'static uranium_env::actions::ActionTable {
    action_table()
}

fn id_of(name: &str) -> u16 {
    table().by_name(name).unwrap_or_else(|| panic!("{name}")).id
}

fn tile_index(t: (i32, i32)) -> usize {
    t.1 as usize * MAP + t.0 as usize
}

fn d2(a: (i32, i32), b: (i32, i32)) -> i64 {
    let (dx, dy) = ((a.0 - b.0) as i64, (a.1 - b.1) as i64);
    dx * dx + dy * dy
}

fn unit_tile(u: &UnitState) -> (i32, i32) {
    (u.pos.0 as i32 / 32, u.pos.1 as i32 / 32)
}

impl Coverage {
    fn new(info: &GameInfo) -> Self {
        // As live agents (031, AMB-031-4): the map precompute knows every map resource, so the
        // cc_slot distance rule matches BWAPI `canBuildHere` (static minerals and geysers).
        let pre = uranium_env::maps::maps_dir().ok().and_then(|d| uranium_env::maps::live_precompute(&d, info).0);
        eprintln!("[031-AC1] map precompute: {}", if pre.is_some() { "loaded" } else { "none" });
        let map = MapInfo::new(info, 0, pre.as_ref()).expect("map");
        let me = info.players.iter().find(|p| p.is_self).expect("self player");
        let start = me.start_tile.map(|t| (t.0 as i32, t.1 as i32)).unwrap_or((0, 0));
        let enemy_starts = {
            let known: Vec<_> = info.players.iter().filter(|p| p.is_enemy).filter_map(|p| p.start_tile).collect();
            let all = if known.is_empty() { info.static_map.start_locations.clone() } else { known };
            all.into_iter().map(|t| (t.0 as i32, t.1 as i32)).filter(|&t| d2(t, start) > 100).collect()
        };
        Self {
            mem: AgentMemory::new(info),
            map,
            st: State {
                info: info.clone(),
                stats: vec![Stat::default(); table().actions.len()],
                last: vec![],
                follow: VecDeque::new(),
                start,
                home: None,
                enemy_starts,
                main_cc: None,
                scout: None,
                scout_target: 0,
                scout_holding: false,
                waypoints: vec![],
                gas: vec![],
                origin: HashMap::new(),
                pending: HashMap::new(),
                unarmed_pending: None,
                unarmed: vec![],
                fog_pending: None,
                fog: vec![],
                fog_next: 0,
                rotate: 0,
                frame: 0,
                log: vec![],
                sums: vec![],
                end: None,
            },
        }
    }

    fn on_frame(&mut self, f: &RawFrame) -> Act {
        self.st.frame = f.frame;
        self.st.sums.push((f.frame, frame_checksum(f)));
        self.st.settle(&f.events);
        let o = extract(f, &mut self.mem, &self.map, None);
        let w = World::new(f, &self.mem, &self.map);
        let cmds = self.st.decide(f, &o, &w);
        Act { cmds, advance_frames: STEP, valid_until_frame: None }
    }
}

/// Per-frame lookups for ranking candidates.
struct Ctx<'a> {
    f: &'a RawFrame,
    o: &'a Observation,
    w: &'a World<'a>,
    tt: &'static TypeTables,
    units: HashMap<UnitId, &'a UnitState>,
    idx: HashMap<UnitId, usize>,
    /// Tiles within 5 of a resource (keep the mineral line free).
    near_res: Vec<bool>,
    /// Tiles with an own ground unit on them.
    occupied: Vec<bool>,
    /// Addon slots of own addon-capable buildings without an addon.
    reserved: Vec<bool>,
}

impl<'a> Ctx<'a> {
    fn new(f: &'a RawFrame, o: &'a Observation, w: &'a World<'a>, me: u8) -> Self {
        let tt = TypeTables::get();
        let units: HashMap<UnitId, &UnitState> = f.units.iter().map(|u| (u.id, u)).collect();
        let idx = o.entity_ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
        let mut near_res = vec![false; CELLS];
        let mut occupied = vec![false; CELLS];
        let mut reserved = vec![false; CELLS];
        let mark = |v: &mut Vec<bool>, x0: i32, y0: i32, x1: i32, y1: i32| {
            for y in y0.max(0)..y1.min(MAP as i32) {
                for x in x0.max(0)..x1.min(MAP as i32) {
                    v[tile_index((x, y))] = true;
                }
            }
        };
        for u in &f.units {
            let Some(t) = tt.unit(u.unit_type) else { continue };
            let (x, y) = top_left(u, t);
            if t.is_resource() {
                mark(&mut near_res, x - 5, y - 5, x + t.tile_width + 5, y + t.tile_height + 5);
            } else if u.owner == me && !t.is_building() && !t.is_flyer() && u.loaded_in.is_none() {
                let (tx, ty) = unit_tile(u);
                mark(&mut occupied, tx, ty, tx + 1, ty + 1);
            } else if u.owner == me && t.is_building() && !u.has_addon && ADDON_PARENTS.contains(&t.name.as_str()) && !u.has(flags::LIFTED) {
                mark(&mut reserved, x + t.tile_width, y + t.tile_height - 2, x + t.tile_width + 3, y + t.tile_height + 1);
            }
        }
        Self { f, o, w, tt, units, idx, near_res, occupied, reserved }
    }

    fn unit(&self, i: usize) -> Option<&'a UnitState> {
        self.o.entity_ids.get(i).and_then(|id| self.units.get(id).copied())
    }

    fn name(&self, u: &UnitState) -> &'static str {
        self.tt.unit(u.unit_type).map_or("", |t| t.name.as_str())
    }

    fn is_own(&self, u: &UnitState) -> bool {
        u.owner == self.w.mem.self_player
    }

    fn own(&self, name: &str) -> Vec<&'a UnitState> {
        let mut v: Vec<_> = self.f.units.iter().filter(|u| self.is_own(u) && self.name(u) == name).collect();
        v.sort_by_key(|u| u.id);
        v
    }

    fn cap(&self, i: usize, name: &str) -> bool {
        let k = CAPABILITIES.iter().position(|c| *c == name).unwrap();
        i < self.o.obs.n && self.o.masks.unit_cap[i * CAPABILITIES.len() + k]
    }

    /// Free build spot for `name`: one tile of margin, addon space for addon parents, away from
    /// resources, units and other buildings' addon slots. Ranked by distance to `start`.
    fn build_rank(&self, name: &str, start: (i32, i32), tile: usize) -> Option<i64> {
        let t = self.tt.unit_by_name(name)?;
        let (x, y) = ((tile % MAP) as i32, (tile / MAP) as i32);
        let addon = if ADDON_PARENTS.contains(&name) { 3 } else { 0 };
        self.area_free(x - 1, y - 1, x + t.tile_width + 1 + addon, y + t.tile_height + 1, false)?;
        if addon > 0 {
            // The 2x2 addon slot itself must be buildable.
            let (ax, ay) = (x + t.tile_width, y + t.tile_height - 2);
            for yy in ay..ay + 2 {
                for xx in ax..ax + 2 {
                    if !self.w.map.in_map(xx, yy) || !self.w.map.buildable[tile_index((xx, yy))] {
                        return None;
                    }
                }
            }
        }
        Some(d2((x, y), start))
    }

    /// No blocked, resource-near, occupied or reserved tile in [x0, x1) x [y0, y1).
    fn area_free(&self, x0: i32, y0: i32, x1: i32, y1: i32, allow_reserved: bool) -> Option<()> {
        for y in y0..y1 {
            for x in x0..x1 {
                if !self.w.map.in_map(x, y) {
                    return None;
                }
                let i = tile_index((x, y));
                if self.w.blocked[i] || self.near_res[i] || self.occupied[i] || (!allow_reserved && self.reserved[i]) {
                    return None;
                }
            }
        }
        Some(())
    }
}

const ADDON_PARENTS: [&str; 4] = ["Terran_Command_Center", "Terran_Factory", "Terran_Starport", "Terran_Science_Facility"];

const ADDONS: [(&str, &str); 6] = [
    ("BuildAddon_ComsatStation", "Terran_Comsat_Station"),
    ("BuildAddon_NuclearSilo", "Terran_Nuclear_Silo"),
    ("BuildAddon_MachineShop", "Terran_Machine_Shop"),
    ("BuildAddon_ControlTower", "Terran_Control_Tower"),
    ("BuildAddon_PhysicsLab", "Terran_Physics_Lab"),
    ("BuildAddon_CovertOps", "Terran_Covert_Ops"),
];

/// (Build action, type, count) the tech tree needs.
const REQUIRED_BUILDINGS: [(&str, &str, usize); 10] = [
    ("Build_Refinery", "Terran_Refinery", 1),
    ("Build_Barracks", "Terran_Barracks", 1),
    ("Build_EngineeringBay", "Terran_Engineering_Bay", 1),
    ("Build_Academy", "Terran_Academy", 1),
    ("Build_Bunker", "Terran_Bunker", 1),
    ("Build_Factory", "Terran_Factory", 1),
    ("Build_Starport", "Terran_Starport", 1),
    ("Build_Armory", "Terran_Armory", 1),
    ("Build_ScienceFacility", "Terran_Science_Facility", 2),
    ("Build_CommandCenter", "Terran_Command_Center", 2),
];

/// (Train action, type, count) kept alive for the skills.
/// A ghost too: the one that calls our own nuke may die in it before Lockdown is covered.
const KEEP_UNITS: [(&str, &str, usize); 4] = [
    ("Train_Marine", "Terran_Marine", 4),
    ("Train_Medic", "Terran_Medic", 1),
    ("Train_ScienceVessel", "Terran_Science_Vessel", 1),
    ("Train_Ghost", "Terran_Ghost", 1),
];

/// Commands checked on a resource out of vision (011-R20, 액션 감사 F2).
const FOG_ACTIONS: [&str; 2] = ["Smart_Unit", "Gather_Unit"];

/// Orders of an SCV on its way to or at a mineral field.
fn harvest_orders() -> Vec<u16> {
    ["Harvest1", "Harvest2", "MoveToMinerals", "WaitForMinerals", "MiningMinerals"].iter().filter_map(|n| order_id(n)).collect()
}

/// Units only built for coverage: cancelled right after they start (refund, Cancel_Train).
const CANCEL_TRAIN: [&str; 4] = ["Train_Firebat", "Train_Goliath", "Train_Wraith", "Train_Valkyrie"];

impl State {
    fn ok(&self, name: &str) -> bool {
        self.stats[id_of(name) as usize].ok > 0
    }

    /// No order of this action for `frames` frames.
    fn quiet(&self, name: &str, frames: u32) -> bool {
        self.pending.get(name).is_none_or(|&f| self.frame >= f + frames)
    }

    fn done(&self) -> bool {
        self.stats.iter().all(|s| s.ok > 0) && !self.unarmed.is_empty() && FOG_ACTIONS.iter().all(|a| self.fog_ok(a))
    }

    /// A command on a resource out of vision was accepted and the SCV went for the minerals.
    fn fog_ok(&self, name: &str) -> bool {
        self.fog.iter().any(|(a, o)| a == name && o.is_some_and(|o| harvest_orders().contains(&o)))
    }

    /// 031-R10: `CommandFailed` for the previous `Act` arrives with this frame.
    fn settle(&mut self, events: &[GameEvent]) {
        let failed: HashMap<usize, u16> = events
            .iter()
            .filter_map(|e| match e {
                GameEvent::CommandFailed { cmd_index, error } => Some((*cmd_index as usize, *error)),
                _ => None,
            })
            .collect();
        for (i, s) in std::mem::take(&mut self.last).into_iter().enumerate() {
            let st = &mut self.stats[s.action as usize];
            if let Some(&e) = failed.get(&i) {
                st.failed += 1;
                st.errors.push(e);
                self.log.push(format!("frame {}: {} failed, BWAPI error {e}", self.frame, table().actions[s.action as usize].name));
                if s.fog {
                    self.fog.push((table().actions[s.action as usize].name.clone(), None));
                }
                continue;
            }
            if s.fog {
                self.fog_pending = s.units.first().map(|&u| (s.action, u));
            }
            st.ok += 1;
            st.first_ok_frame.get_or_insert(self.frame);
            if s.unarmed {
                self.unarmed_pending = s.units.first().copied();
            }
        }
    }

    fn decide(&mut self, f: &RawFrame, o: &Observation, w: &World) -> Vec<Command> {
        let me = w.mem.self_player;
        let cx = Ctx::new(f, o, w, me);
        for (a, s) in self.stats.iter_mut().enumerate() {
            s.opened += o.masks.type_mask[a] as u32;
        }
        if self.stats[0].ok == 0 {
            // NoOp: an Act without commands (delay only).
            self.stats[0].sent = 1;
            self.stats[0].ok = 1;
            self.stats[0].first_ok_frame = Some(f.frame);
        }
        if self.main_cc.is_none() {
            self.main_cc = cx.own("Terran_Command_Center").first().map(|u| u.id);
            if let Some(cc) = self.main_cc.and_then(|id| cx.units.get(&id)) {
                self.start = top_left(cc, cx.tt.unit(cc.unit_type).unwrap());
            }
        }
        if self.home.is_none() {
            self.home = Some(self.pick_home(&cx));
            let walk = |p: (i32, i32)| {
                (0..CELLS).filter(|&t| cx.w.map.walkable[t]).map(|t| ((t % MAP) as i32, (t / MAP) as i32)).min_by_key(|&t| d2(t, p)).unwrap_or(p)
            };
            for k in 0..8 {
                let ang = k as f64 * std::f64::consts::FRAC_PI_4;
                let p = (self.start.0 + (12.0 * ang.cos()) as i32, self.start.1 + (12.0 * ang.sin()) as i32);
                self.waypoints.push(walk(p));
            }
            self.waypoints.extend(self.enemy_starts.iter().copied());
        }
        // AMB-041-1: the unarmed unit's order after an accepted Attack_Pos.
        if let Some(id) = self.unarmed_pending.take() {
            if let Some(u) = cx.units.get(&id) {
                self.unarmed.push((cx.name(u).to_string(), u.order));
            }
        }
        // 액션 감사 F2: the SCV's order after a command on a resource out of vision; then it
        // stops (and mines at home again as an idle worker).
        if let Some((a, id)) = self.fog_pending.take() {
            if let Some(u) = cx.units.get(&id) {
                self.fog.push((table().actions[a as usize].name.clone(), Some(u.order)));
                self.follow.push_back(Follow::Stop(vec![id]));
            }
        }

        let mut plan: Vec<(HostAction, bool)> = vec![];
        self.follow_ups(&cx, &mut plan);
        self.economy(&cx, &mut plan);
        self.unarmed_check(&cx, &mut plan);
        let coverage = self.coverage(&cx);
        let fog = self.fog_check(&cx);

        let mut busy: HashSet<UnitId> = HashSet::new();
        let mut cmds = vec![];
        for (act, unarmed) in plan.into_iter().chain(fog.clone().map(|a| (a, false))).chain(coverage.map(|a| (a, false))) {
            let is_fog = fog.as_ref() == Some(&act);
            let st_i = act.action as usize;
            match execute(w, o, &act) {
                Ok(e) if e.commands.is_empty() => {}
                Ok(e) => {
                    if e.units.iter().any(|u| busy.contains(u)) {
                        continue;
                    }
                    busy.extend(e.units.iter().copied());
                    self.after_send(&cx, &table().actions[st_i], &e.units);
                    if table().actions[st_i].name == "Load_Unit" {
                        // Which transport the units were sent into, and the dropships' cargo.
                        let target = act.target_unit.and_then(|i| cx.unit(i)).map_or("?", |u| cx.name(u));
                        let cargo: Vec<String> = cx
                            .own("Terran_Dropship")
                            .iter()
                            .map(|d| {
                                let i = cx.idx.get(&d.id).copied();
                                format!("{} loadable={:?} flags={:#x} order={} hp={}", cx.w.cargo.get(&d.id).copied().unwrap_or(0), i.map(|i| cx.cap(i, "loadable_transport")), d.flags, d.order, d.hp)
                            })
                            .collect();
                        self.log.push(format!("frame {}: Load_Unit into {target}, dropship cargo {cargo:?}", f.frame));
                    }
                    self.stats[st_i].sent += 1;
                    self.stats[st_i].last_try_frame = f.frame;
                    self.last.push(Sent { action: act.action, units: e.units.clone(), unarmed, fog: is_fog });
                    cmds.extend(e.commands);
                }
                Err(_) => {
                    self.stats[st_i].rejected += 1;
                    self.stats[st_i].last_try_frame = f.frame;
                }
            }
        }
        cmds
    }

    fn pick_home(&self, cx: &Ctx) -> (i32, i32) {
        let (w, h) = (self.info.map_size_tiles.0 as i32, self.info.map_size_tiles.1 as i32);
        let c = (w / 2, h / 2);
        let len = (d2(c, self.start) as f64).sqrt().max(1.0);
        let want = (self.start.0 + ((c.0 - self.start.0) as f64 * 9.0 / len) as i32, self.start.1 + ((c.1 - self.start.1) as f64 * 9.0 / len) as i32);
        (0..CELLS)
            .filter(|&t| cx.w.map.walkable[t] && !cx.w.blocked[t])
            .map(|t| ((t % MAP) as i32, (t / MAP) as i32))
            .min_by_key(|&t| d2(t, want))
            .unwrap_or(self.start)
    }

    /// A host action for `name` with explicit units (masks still checked by the executor).
    fn by_ids(cx: &Ctx, name: &str, sel: &[UnitId], target: Option<UnitId>, loc: Option<(i32, i32)>, queued: bool) -> Option<HostAction> {
        let a = table().by_name(name)?;
        let selected: Vec<usize> = sel.iter().filter_map(|id| cx.idx.get(id).copied()).collect();
        if a.uses_select && selected.is_empty() {
            return None;
        }
        let target_unit = match target {
            Some(t) => Some(*cx.idx.get(&t)?),
            None => None,
        };
        Some(HostAction { action: a.id, delay_bucket: 0, queued, selected, target_unit, target_loc: loc.map(tile_index) })
    }

    fn follow_ups(&mut self, cx: &Ctx, plan: &mut Vec<(HostAction, bool)>) {
        let home = self.home.unwrap();
        for fu in std::mem::take(&mut self.follow) {
            let act = match &fu {
                Follow::Stop(ids) => Self::by_ids(cx, "Stop", ids, None, None, false),
                Follow::Home(id) => Self::by_ids(cx, "Move_Pos", &[*id], None, Some(home), true),
                Follow::CancelUpgrade(id) => Self::by_ids(cx, "Cancel_Upgrade", &[*id], None, None, false),
                Follow::CancelTrain(id) => Self::by_ids(cx, "Cancel_Train", &[*id], None, None, false),
                Follow::CancelResearch(id) => Self::by_ids(cx, "Cancel_Research", &[*id], None, None, false),
                Follow::CancelAddon(id) => Self::by_ids(cx, "Cancel_Addon", &[*id], None, None, false),
            };
            match act {
                Some(a) if cx.o.masks.type_mask[a.action as usize] => plan.push((a, false)),
                // Not possible yet (the upgrade has not started, the caster is gone): retry for a
                // while, then drop it.
                _ => {
                    if self.frame + 24 * 60 < MAX_FRAMES {
                        self.follow.push_back(fu);
                    }
                }
            }
        }
        // Keep the follow-up list from growing without bound when a unit is gone.
        while self.follow.len() > 32 {
            self.follow.pop_front();
        }
    }

    fn after_send(&mut self, cx: &Ctx, a: &Action, units: &[UnitId]) {
        let name = a.name.as_str();
        self.pending.insert(a.name.clone(), self.frame);
        match name {
            "Attack_Unit" => self.follow.push_back(Follow::Stop(units.to_vec())),
            "Lockdown_Unit" | "OpticalFlare_Unit" | "Yamato_Unit" | "EMP_Pos" | "Nuke_Pos" => {
                units.iter().for_each(|&u| self.follow.push_back(Follow::Home(u)));
            }
            "Lift" => {
                if let Some(u) = units.first().and_then(|id| cx.units.get(id)) {
                    self.origin.insert(u.id, top_left(u, cx.tt.unit(u.unit_type).unwrap()));
                }
            }
            "Research_CloakingField" => units.iter().for_each(|&u| self.follow.push_back(Follow::CancelResearch(u))),
            "BuildAddon_MachineShop" if !self.ok("Cancel_Addon") => units.iter().for_each(|&u| self.follow.push_back(Follow::CancelAddon(u))),
            _ if name.starts_with("Upgrade_") => units.iter().for_each(|&u| self.follow.push_back(Follow::CancelUpgrade(u))),
            _ if CANCEL_TRAIN.contains(&name) => units.iter().for_each(|&u| self.follow.push_back(Follow::CancelTrain(u))),
            _ => {}
        }
    }

    /// Workers, supply, the second science facility and CC, addons and research that a cancel
    /// took back, and the scout that gives vision of the idle enemy.
    fn economy(&mut self, cx: &Ctx, plan: &mut Vec<(HostAction, bool)>) {
        let me = &cx.f.self_state;
        let scvs = cx.own("Terran_SCV");
        self.gas.retain(|id| scvs.iter().any(|u| u.id == *id));
        let main = self.main_cc.and_then(|id| cx.units.get(&id).copied());
        let (scout, gas_ids) = (self.scout, self.gas.clone());
        let not_special = |u: &&&UnitState| Some(u.id) != scout && !gas_ids.contains(&u.id);

        // Idle workers mine (gas workers go back to gas).
        let refinery = cx.own("Terran_Refinery").into_iter().find(|u| u.has(flags::COMPLETED));
        let idle: Vec<UnitId> = scvs
            .iter()
            .filter(|u| u.has(flags::COMPLETED) && u.has(flags::IDLE) && Some(u.id) != scout && u.loaded_in.is_none())
            .map(|u| u.id)
            .collect();
        let (gas_idle, min_idle): (Vec<UnitId>, Vec<UnitId>) = idle.into_iter().partition(|id| self.gas.contains(id));
        if let (Some(r), false) = (refinery, gas_idle.is_empty()) {
            plan.extend(Self::by_ids(cx, "Gather_Unit", &gas_idle, Some(r.id), None, false).map(|a| (a, false)));
        }
        let mineral = main.and_then(|cc| {
            cx.f.units.iter().filter(|u| cx.name(u).starts_with("Resource_Mineral_Field")).min_by_key(|u| d2(unit_tile(u), unit_tile(cc)))
        });
        if let (Some(m), false) = (mineral, min_idle.is_empty()) {
            let ids: Vec<UnitId> = min_idle.into_iter().take(12).collect();
            plan.extend(Self::by_ids(cx, "Gather_Unit", &ids, Some(m.id), None, false).map(|a| (a, false)));
        }
        // Three on gas.
        if let Some(r) = refinery {
            if self.gas.len() < 3 {
                let mut cand: Vec<&&UnitState> =
                    scvs.iter().filter(not_special).filter(|u| u.has(flags::COMPLETED) && !u.has(flags::CONSTRUCTING) && !u.has(flags::CARRYING)).collect();
                cand.sort_by_key(|u| (d2(unit_tile(u), unit_tile(r)), u.id));
                let ids: Vec<UnitId> = cand.iter().take(3 - self.gas.len()).map(|u| u.id).collect();
                if let Some(a) = Self::by_ids(cx, "Gather_Unit", &ids, Some(r.id), None, false) {
                    self.gas.extend(ids);
                    plan.push((a, false));
                }
            }
        }
        // Workers.
        let queued_scv = main.is_some_and(|cc| cc.build_queue[0] != cx.w.ids.none);
        if scvs.len() < 22 && !queued_scv && me.minerals >= 50 {
            plan.extend(valid_action(cx.o, table().by_name("Train_SCV").unwrap(), &|_, _| Some(0), &|_| Some(0)).map(|a| (a, false)));
        }
        // Supply: keep 4+ free (8 in BWAPI half-supply units), two depots at a time late.
        // A worker walking to its site has placed nothing yet: one order per type per 30 s.
        let free = me.supply_total as i32 - me.supply_used as i32;
        let depots_building = cx.own("Terran_Supply_Depot").iter().filter(|u| !u.has(flags::COMPLETED)).count();
        let want_building = if me.supply_total >= 60 { 2 } else { 1 };
        // Late, keep room for a battlecruiser (6 supply = 12 half units).
        let reserve = if me.supply_total >= 60 { 16 } else { 8 };
        if me.supply_total < 400 && free < reserve + 16 * depots_building as i32 && depots_building < want_building && me.minerals >= 100 && self.quiet("Build_SupplyDepot", 720) {
            let a = table().by_name("Build_SupplyDepot").unwrap();
            let start = self.start;
            plan.extend(valid_action(cx.o, a, &|_, _| Some(0), &|t| cx.build_rank("Terran_Supply_Depot", start, t)).map(|a| (a, false)));
        }
        // Units the remaining skills need (marines: stim, heal, load, moves; medic: restoration;
        // vessel: irradiate), replaced when lost (e.g. to our own nuke).
        for (train, ty, need) in KEEP_UNITS {
            if self.ok(train) && cx.own(ty).len() < need && self.quiet(train, 480) {
                plan.extend(valid_action(cx.o, table().by_name(train).unwrap(), &|_, _| Some(0), &|_| Some(0)).map(|a| (a, false)));
            }
        }
        // Buildings the tech tree needs: placed again when the first order did not lead to a
        // building; a second science facility (physics lab and covert ops) and CC (silo, CC lift).
        for (build, ty, need) in REQUIRED_BUILDINGS {
            if self.ok(build) && cx.own(ty).len() < need && self.quiet(build, 1440) {
                let a = table().by_name(build).unwrap();
                let start = self.start;
                let rank = |t: usize| if ty == "Terran_Refinery" { Some(d2(((t % MAP) as i32, (t / MAP) as i32), start)) } else { cx.build_rank(ty, start, t) };
                plan.extend(valid_action(cx.o, a, &|_, _| Some(0), &rank).map(|a| (a, false)));
            }
        }
        // Rebuild an addon a cancel took back; covered addons only.
        for (action, addon) in ADDONS {
            if self.ok(action) && cx.own(addon).is_empty() && (action != "BuildAddon_NuclearSilo" || self.ok("Land_CommandCenter_Pos")) && self.quiet(action, 240) {
                let a = table().by_name(action).unwrap();
                let main = self.main_cc;
                plan.extend(
                    valid_action(cx.o, a, &|_, i| match (action, cx.unit(i)) {
                        ("BuildAddon_NuclearSilo", Some(u)) if Some(u.id) == main => None,
                        (_, Some(_)) => Some(0),
                        _ => None,
                    }, &|_| Some(0))
                    .map(|a| (a, false)),
                );
            }
        }
        // A lifted building lands again (a land order given while it is still lifting off can be
        // lost), back on its own tiles when they are free.
        if let Some(b) = cx.f.units.iter().find(|u| cx.is_own(u) && u.has(flags::LIFTED)) {
            let lifted_at = self.pending.get("Lift").copied().unwrap_or(0);
            let land = if cx.name(b) == "Terran_Command_Center" { "Land_CommandCenter_Pos" } else { "Land_Pos" };
            if self.ok(land) && self.frame >= lifted_at + 96 && self.quiet(land, 480) {
                let a = table().by_name(land).unwrap();
                let (start, origin, ty) = (self.start, self.origin.get(&b.id).copied(), cx.name(b));
                let rank = |t: usize| {
                    let xy = ((t % MAP) as i32, (t / MAP) as i32);
                    if Some(xy) == origin { Some(-1) } else { cx.build_rank(ty, start, t) }
                };
                plan.extend(valid_action(cx.o, a, &|_, i| cx.unit(i).filter(|u| u.id == b.id).map(|_| 0), &rank).map(|a| (a, false)));
            }
        }
        // Research a cancel took back (cloaking field stays cancelled: ghosts cloak instead).
        for a in table().actions.iter().filter(|a| a.name.starts_with("Research_") && a.name != "Research_CloakingField") {
            let tech = a.tech.unwrap();
            if self.ok(&a.name) && !cx.w.researched(tech) && me.researching & (1u64 << tech) == 0 && self.quiet(&a.name, 240) {
                plan.extend(valid_action(cx.o, a, &|_, _| Some(0), &|_| Some(0)).map(|a| (a, false)));
            }
        }
        // Scout: an SCV parks at the enemy base (workers do not attack on their own).
        if self.scout.is_none() && f_frame(cx) > 1500 {
            self.scout = scvs.iter().filter(not_special).filter(|u| u.has(flags::COMPLETED) && !u.has(flags::CONSTRUCTING)).map(|u| u.id).next_back();
        }
        if self.scout.and_then(|id| cx.units.get(&id)).is_some_and(|u| u.has(flags::CONSTRUCTING)) {
            // The executor picked the scout as a builder (031-R2): it stays a worker.
            self.scout = None;
            self.scout_holding = false;
        }
        if let Some(s) = self.scout.and_then(|id| cx.units.get(&id).copied()) {
            let enemy_seen = cx.f.units.iter().any(|u| !cx.is_own(u) && u.owner < 8 && cx.tt.unit(u.unit_type).is_some_and(|t| t.is_building() && !t.is_resource()));
            if enemy_seen {
                if !self.scout_holding {
                    plan.extend(Self::by_ids(cx, "Hold", &[s.id], None, None, false).map(|a| (a, false)));
                    self.scout_holding = true;
                }
            } else if let Some(&t) = self.waypoints.get(self.scout_target % self.waypoints.len().max(1)) {
                if d2(unit_tile(s), t) < 25 {
                    self.scout_target += 1;
                } else if !s.has(flags::MOVING) {
                    plan.extend(Self::by_ids(cx, "Move_Pos", &[s.id], None, Some(t), false).map(|a| (a, false)));
                }
            }
        } else if self.scout.is_some() {
            self.scout = None;
        }
    }

    /// AMB-041-1: Attack_Pos with an unarmed unit (dropship, science vessel) must reach the game
    /// as a move. Done once per unarmed type.
    fn unarmed_check(&mut self, cx: &Ctx, plan: &mut Vec<(HostAction, bool)>) {
        if self.unarmed_pending.is_some() || plan.iter().any(|(_, u)| *u) {
            return;
        }
        for ty in ["Terran_Dropship", "Terran_Science_Vessel"] {
            if self.unarmed.iter().any(|(t, _)| t == ty) {
                continue;
            }
            // Not while units are boarding it: their load turns the dropship's order into
            // PickupTransport on the next frame.
            let boarding = [order_id("PickupTransport"), order_id("PickupIdle")];
            if !self.quiet("Load_Unit", 192) {
                continue;
            }
            let Some(u) = cx.own(ty).into_iter().find(|u| u.has(flags::COMPLETED) && u.loaded_in.is_none() && !boarding.contains(&Some(u.order))) else { continue };
            let home = self.home.unwrap();
            let far = if d2(unit_tile(u), home) > d2(unit_tile(u), self.start) { home } else { self.start };
            if let Some(a) = Self::by_ids(cx, "Attack_Pos", &[u.id], None, Some(far), false) {
                plan.push((a, true));
                return;
            }
        }
    }

    /// 011-R20 / 액션 감사 F2: Smart_Unit and Gather_Unit with one SCV on a mineral field seen
    /// before and now out of vision (a remembered entity with its real id).
    fn fog_check(&mut self, cx: &Ctx) -> Option<HostAction> {
        if self.fog_pending.is_some() || self.frame < self.fog_next {
            return None;
        }
        let name = FOG_ACTIONS.iter().find(|a| !self.fog_ok(a) && self.fog.iter().filter(|(x, _)| x == *a).count() < 4)?;
        let mineral = cx.w.entities.iter().find(|e| {
            e.remembered
                && !uranium_env::features::world::is_unseen_resource_id(e.unit.id)
                && cx.tt.unit(e.unit.unit_type).is_some_and(|t| t.name.starts_with("Resource_Mineral_Field"))
        })?;
        let (scout, gas) = (self.scout, &self.gas);
        let scv = cx
            .own("Terran_SCV")
            .into_iter()
            .filter(|u| u.has(flags::COMPLETED) && !u.has(flags::CONSTRUCTING) && !u.has(flags::CARRYING) && Some(u.id) != scout && !gas.contains(&u.id))
            .min_by_key(|u| (d2(unit_tile(u), self.start), u.id))?;
        self.fog_next = self.frame + RETRY_FRAMES;
        Self::by_ids(cx, name, &[scv.id], Some(mineral.unit.id), None, false)
    }

    /// One uncovered action type whose mask is open, with ranked arguments.
    fn coverage(&mut self, cx: &Ctx) -> Option<HostAction> {
        let n = table().actions.len();
        for k in 0..n {
            let a = &table().actions[(self.rotate + k) % n];
            let s = &self.stats[a.id as usize];
            if !cx.o.masks.type_mask[a.id as usize] || s.sent >= MAX_TRIES + self.extra_tries(&a.name) || (s.last_try_frame != 0 && self.frame < s.last_try_frame + RETRY_FRAMES) {
                continue;
            }
            if s.ok > 0 && !self.repeat(cx, &a.name) {
                continue;
            }
            if !self.gate(&a.name) {
                continue;
            }
            if let Some(act) = self.pick(cx, a) {
                self.rotate = (self.rotate + k + 1) % n;
                return Some(act);
            }
        }
        None
    }

    fn extra_tries(&self, name: &str) -> u32 {
        // Repeated setup actions (see `repeat`).
        if matches!(name, "Lift" | "Load_Unit" | "Irradiate_Unit" | "Stim" | "Build_MissileTurret") { 8 } else { 0 }
    }

    /// Setup actions issued again until the action they enable is covered.
    fn repeat(&self, cx: &Ctx, name: &str) -> bool {
        let lifted = cx.f.units.iter().any(|u| cx.is_own(u) && u.has(flags::LIFTED));
        let has_cap = |cap: &str| (0..cx.o.obs.n).any(|i| cx.cap(i, cap));
        match name {
            "Lift" => self.ok("Land_Pos") && !self.ok("Land_CommandCenter_Pos") && !lifted,
            // UnloadAll_Pos needs a loaded dropship (can_unload_pos), not a loaded bunker; Unload_Unit
            // (F7) a loaded transport. A dropship still in production is no target (034-R10).
            "Load_Unit" => !(self.ok("UnloadAll_Pos") && self.ok("Unload_Unit")) && !has_cap("can_unload_pos") && cx.own("Terran_Dropship").iter().any(|u| u.has(flags::COMPLETED)),
            "Irradiate_Unit" => !self.ok("Restoration_Unit") && has_cap("can_cast_restoration") && !has_cap("restorable"),
            "Stim" => !self.ok("Heal_Unit") && has_cap("can_heal") && !has_cap("healable"),
            "Build_MissileTurret" => !self.ok("Cancel_Construction") && cx.own("Terran_Missile_Turret").iter().all(|u| u.has(flags::COMPLETED)),
            _ => false,
        }
    }

    /// Order constraints between coverage actions.
    fn gate(&self, name: &str) -> bool {
        match name {
            // The second CC lifts and lands first; its addon slot then takes the silo.
            "BuildAddon_NuclearSilo" => self.ok("Land_CommandCenter_Pos"),
            // Our own nuke damages our Engineering Bay for Repair_Unit; upgrades there first.
            "Nuke_Pos" => self.ok("Upgrade_InfantryWeapons") && self.ok("Upgrade_InfantryArmor") && self.ok("Cancel_Construction"),
            _ => true,
        }
    }

    fn pick(&self, cx: &Ctx, a: &Action) -> Option<HostAction> {
        let name = a.name.as_str();
        let home = self.home.unwrap();
        let start = self.start;
        let main = self.main_cc;
        let scout = self.scout;
        let gas = &self.gas;
        let (mw, mh) = (self.info.map_size_tiles.0 as i32, self.info.map_size_tiles.1 as i32);
        let center = (mw / 2, mh / 2);
        let enemy = self.enemy_starts.first().copied().unwrap_or(center);
        let ebay = cx.own("Terran_Engineering_Bay").first().map(|u| unit_tile(u));
        let is = |u: &UnitState, n: &str| cx.name(u) == n;
        let own = |u: &UnitState| cx.is_own(u);
        let bld = |u: &UnitState| cx.tt.unit(u.unit_type).is_some_and(|t| t.is_building());
        let mech_only = |u: &UnitState| cx.tt.unit(u.unit_type).is_some_and(|t: &UnitType| t.is_mechanical() && !t.is_organic() && !t.is_building());
        // Ordinary movers: marines first, never the scout or gas workers, never buildings.
        let mover = |u: &UnitState| -> Option<i64> {
            if bld(u) || Some(u.id) == scout || gas.contains(&u.id) {
                return None;
            }
            Some(match cx.name(u) {
                "Terran_Marine" => 0,
                "Terran_Firebat" | "Terran_Vulture" | "Terran_Goliath" => 1,
                "Terran_SCV" => 5,
                _ => 3,
            })
        };
        let near = |p: (i32, i32)| move |t: usize| Some(d2(((t % MAP) as i32, (t / MAP) as i32), p));
        let rank = |role: Role, i: usize| -> Option<i64> {
            let u = cx.unit(i)?;
            match (name, role) {
                ("Move_Pos" | "Move_Unit" | "Attack_Pos" | "Patrol_Pos" | "Hold" | "Stop" | "Smart_Pos" | "Smart_Unit", Role::Select) => mover(u),
                ("Attack_Unit", Role::Select) => mover(u).filter(|_| !is(u, "Terran_SCV")),
                ("Move_Unit" | "Smart_Unit" | "SetRally_Unit" | "DefensiveMatrix_Unit", Role::Target) => (own(u) && !bld(u)).then(|| if is(u, "Terran_Marine") { 0 } else { 1 }),
                ("Gather_Unit", Role::Select) => (Some(u.id) != scout).then(|| if u.has(flags::IDLE) { 0 } else { 1 }),
                ("Gather_Unit", Role::Target) => Some(main.and_then(|m| cx.units.get(&m)).map_or(0, |cc| d2(unit_tile(u), unit_tile(cc)))),
                ("ReturnCargo", Role::Select) => (u.has(flags::CARRYING) && Some(u.id) != scout).then_some(0),
                ("Repair_Unit", Role::Select) => (Some(u.id) != scout).then(|| d2(unit_tile(u), start)),
                ("Attack_Unit" | "Yamato_Unit", Role::Target) => Some(if bld(u) { 0 } else { 1 }),
                ("Irradiate_Unit", Role::Target) => {
                    if own(u) {
                        mech_only(u).then_some(0)
                    } else {
                        Some(1)
                    }
                }
                ("Lift", Role::Select) => {
                    if !self.ok("Land_Pos") {
                        (!is(u, "Terran_Command_Center") && !u.has_addon).then(|| if is(u, "Terran_Engineering_Bay") { 0 } else { 1 })
                    } else {
                        (is(u, "Terran_Command_Center") && Some(u.id) != main && !u.has_addon).then_some(0)
                    }
                }
                ("BuildAddon_ComsatStation", Role::Select) => Some(if Some(u.id) == main { 0 } else { 1 }),
                ("BuildAddon_NuclearSilo", Role::Select) => (Some(u.id) != main).then_some(0),
                ("Cancel_Construction", Role::Select) => is(u, "Terran_Missile_Turret").then_some(0),
                ("Cancel_Research", Role::Select) => is(u, "Terran_Control_Tower").then_some(0),
                ("Cancel_Addon", Role::Select) => is(u, "Terran_Factory").then_some(0),
                ("SetRally_Pos" | "SetRally_Unit", Role::Select) => (!is(u, "Terran_Command_Center")).then_some(0),
                ("UnloadAll_Pos" | "Unload_Unit", Role::Select) => is(u, "Terran_Dropship").then_some(0),
                ("Load_Unit", Role::Target) => Some(if is(u, "Terran_Dropship") { 0 } else { 1 }),
                ("Load_Unit", Role::Select) => is(u, "Terran_Marine").then_some(0),
                ("Stim", Role::Select) => is(u, "Terran_Marine").then_some(0),
                _ => Some(0),
            }
        };
        let loc = |t: usize| -> Option<i64> {
            match name {
                "Move_Pos" | "Attack_Pos" | "Smart_Pos" | "SetRally_Pos" | "UnloadAll_Pos" | "SpiderMine_Pos" => near(home)(t),
                "Patrol_Pos" => near((home.0 + 3, home.1))(t),
                "EMP_Pos" => near(center)(t),
                "Nuke_Pos" => near(ebay.unwrap_or(center))(t),
                "ScannerSweep_Pos" => near(enemy)(t),
                "Build_Refinery" => near(start)(t),
                "Land_Pos" | "Land_CommandCenter_Pos" => {
                    let lifted = cx.f.units.iter().find(|u| cx.is_own(u) && u.has(flags::LIFTED))?;
                    let ty = cx.name(lifted);
                    let xy = ((t % MAP) as i32, (t / MAP) as i32);
                    match self.origin.get(&lifted.id) {
                        Some(&o) if o == xy => Some(-1),
                        _ => cx.build_rank(ty, start, t),
                    }
                }
                _ if name.starts_with("Build_") => {
                    let ty = a.unit_type.and_then(|u| cx.tt.unit(u)).map(|t| t.name.clone())?;
                    cx.build_rank(&ty, start, t)
                }
                _ => Some(0),
            }
        };
        valid_action(cx.o, a, &rank, &loc)
    }
}

fn f_frame(cx: &Ctx) -> u32 {
    cx.f.frame
}

enum Report {
    Coverage(Box<State>),
    Idle,
}

/// One bridge connection: player 0 runs the coverage script, player 1 idles.
/// FNV-1a over the observed units (id order): id, type, position, hp, shield, energy, order,
/// order target, flags. Equal sequences = the two games played the same (011-AC4).
fn frame_checksum(f: &RawFrame) -> u64 {
    let mut units: Vec<&UnitState> = f.units.iter().collect();
    units.sort_by_key(|u| u.id);
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |v: u64| {
        for b in v.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    };
    eat(f.frame as u64);
    for u in units {
        for v in [u.id as u64, u.unit_type as u64, u.pos.0 as u64, u.pos.1 as u64, u.hp as u64, u.shield as u64, u.energy as u64, u.order as u64, u.order_target.map_or(u64::MAX, |t| t as u64), u.flags as u64] {
            eat(v);
        }
    }
    eat(f.self_state.minerals as u64);
    eat(f.self_state.gas as u64);
    h
}

fn session(stream: TcpStream, hostd: String, game_id: u64, stop_when_done: bool) -> Result<Report, String> {
    let e = |x: ProtoError| x.to_string();
    let mut r = BufReader::new(stream.try_clone().map_err(|x| x.to_string())?);
    let mut w = stream;
    let BridgeMsg::Hello(_) = read_msg(&mut r).map_err(e)? else { return Err("expected Hello".into()) };
    let BridgeMsg::GameInfo(info) = read_msg(&mut r).map_err(e)? else { return Err("expected GameInfo".into()) };
    let me = info.players.iter().find(|p| p.is_self).map(|p| p.name.clone()).unwrap_or_default();
    let mut cov = (me == "uranium-p0").then(|| Coverage::new(&info));
    let mut stopped = !stop_when_done;
    loop {
        let msg = match read_msg::<BridgeMsg>(&mut r) {
            Ok(m) => m,
            Err(x) => {
                if let Some(c) = &mut cov {
                    c.st.end.get_or_insert(format!("connection closed: {x}"));
                }
                break;
            }
        };
        match msg {
            BridgeMsg::Frame(f) => {
                let act = match &mut cov {
                    Some(c) => c.on_frame(&f),
                    None => Act { cmds: vec![], advance_frames: 96, valid_until_frame: None },
                };
                if write_msg(&mut w, &AgentMsg::Act(act)).is_err() {
                    break;
                }
                if let Some(c) = &cov {
                    if c.st.done() && !stopped {
                        // Every type is covered: stop the game instead of playing to the limit.
                        stopped = true;
                        let _ = control(&hostd, ControlRequest::StopGame { game_id }, false);
                    }
                }
            }
            BridgeMsg::End(end) => {
                if let Some(c) = &mut cov {
                    if let Some(ff) = &end.final_frame {
                        c.st.settle(&ff.events);
                    }
                    let me = c.st.info.self_player;
                    c.st.end = Some(format!("{:?} at frame {}, winner {:?} (we are {me})", end.reason, end.frame, end.winner));
                }
                break;
            }
            _ => {}
        }
    }
    Ok(match cov {
        Some(c) => Report::Coverage(Box::new(c.st)),
        None => Report::Idle,
    })
}

/// One request on a fresh hostd control connection; returns the first reply when `reply` (hostd
/// does not answer `StopGame`).
fn control(addr: &str, req: ControlRequest, reply: bool) -> Result<(BufReader<TcpStream>, TcpStream, Option<ControlResponse>), String> {
    let s = TcpStream::connect(addr).map_err(|e| format!("hostd {addr}: {e}"))?;
    let mut r = BufReader::new(s.try_clone().map_err(|e| e.to_string())?);
    let mut w = s;
    write_msg(&mut w, &ControlRequest::Hello { control_version: CONTROL_VERSION }).map_err(|e| e.to_string())?;
    match read_msg(&mut r).map_err(|e| e.to_string())? {
        ControlResponse::Hello { .. } => {}
        other => return Err(format!("hostd handshake: {other:?}")),
    }
    write_msg(&mut w, &req).map_err(|e| e.to_string())?;
    let resp = if reply { Some(read_msg(&mut r).map_err(|e| e.to_string())?) } else { None };
    Ok((r, w, resp))
}

/// BWAPI errors (`Errors::Enum`) the game may return although the masks allowed the action, and
/// why. Anything else is a mask / executor / bridge mismatch (031-AC1).
fn justified(action: &str, error: u16) -> Option<&'static str> {
    const UNBUILDABLE_LOCATION: u16 = 19;
    const INSUFFICIENT_MINERALS: u16 = 11;
    const INSUFFICIENT_GAS: u16 = 12;
    let spends = ["Build_", "BuildAddon_", "Train_", "Research_", "Upgrade_"].iter().any(|p| action.starts_with(p));
    match (action, error) {
        // Test artifact: the scenario sends economy and coverage commands in one Act, all judged
        // against the same minerals; the agent sends one command per decision (031-R3).
        (_, INSUFFICIENT_MINERALS | INSUFFICIENT_GAS) if spends => Some("several spending commands in one test Act"),
        // 034 위치 마스크 비고: location masks see buildings and resources of the current
        // observation, not units standing on the tiles; the game rejects a footprint with a unit.
        (a, UNBUILDABLE_LOCATION) if a.starts_with("Build_") || a.starts_with("BuildAddon_") || a.starts_with("Land_") => Some("unit on the footprint"),
        _ => None,
    }
}

fn report(st: &State) -> String {
    let mut md = String::from("| id | action | opened | sent | ok | failed | rejected | first ok frame | errors |\n|---|---|---|---|---|---|---|---|---|\n");
    for (a, s) in table().actions.iter().zip(&st.stats) {
        md += &format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {:?} |\n",
            a.id,
            a.name,
            s.opened,
            s.sent,
            s.ok,
            s.failed,
            s.rejected,
            s.first_ok_frame.map_or("-".into(), |f| f.to_string()),
            s.errors
        );
    }
    md += &format!("\nunarmed Attack_Pos (type, order after): {:?} (Move = {:?})\n", st.unarmed, order_id("Move"));
    md += &format!("resource out of vision (action, order after; None = failed): {:?} (harvest orders {:?})\n", st.fog, harvest_orders());
    md += &format!("end: {:?}\n\nfailures:\n", st.end);
    for l in &st.log {
        md += &format!("- {l}\n");
    }
    md
}

/// One scripted game on hostd: player 0 runs the coverage script (stopping the game once every
/// type is covered when `stop_when_done`), player 1 idles. Returns the script state.
fn play(seed: u32, max_frames: u32, stop_when_done: bool, tag: &str) -> Box<State> {
    let hostd = std::env::var("URANIUM_TEST_HOSTD").expect("URANIUM_TEST_HOSTD=<hostd control address>");
    let listen = std::env::var("URANIUM_TEST_AGENT_LISTEN").unwrap_or_else(|_| "127.0.0.1:0".into());
    let listener = TcpListener::bind(&listen).expect("agent listener");
    let bound = listener.local_addr().unwrap();
    // Bridges on a remote host need an address they can reach (as for `uranium-env run`).
    let agent_addr = std::env::var("URANIUM_AGENT_PUBLIC_ADDR").ok().filter(|v| !v.trim().is_empty()).map_or(bound.to_string(), |a| {
        // Keep our port: the public address names the host.
        let host = a.rsplit_once(':').map_or(a.as_str(), |(h, _)| h).to_string();
        format!("{host}:{}", bound.port())
    });
    let game_id = 0x031A_C100_0000 + (std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() & 0xFFFF_FFFF);
    let _ = tag;
    let player = PlayerSpec { race: Race::Terran, kind: PlayerKind::Bridge { agent_addr: agent_addr.clone() } };
    let spec = GameSpec {
        game_id,
        map: MapKey::FightingSpirit,
        players: [player.clone(), player],
        seed: Some(seed),
        max_frames,
        privileged: false,
        save_replay: true,
        wall_timeout_s: 3600,
    };

    let sessions = {
        let hostd = hostd.clone();
        std::thread::spawn(move || {
            let mut handles = vec![];
            for _ in 0..2 {
                let (s, _) = listener.accept().expect("bridge connection");
                s.set_nodelay(true).ok();
                let hostd = hostd.clone();
                handles.push(std::thread::spawn(move || session(s, hostd, game_id, stop_when_done)));
            }
            handles.into_iter().map(|h| h.join().expect("session panicked")).collect::<Vec<_>>()
        })
    };

    let t0 = Instant::now();
    let (mut r, _w, resp) = control(&hostd, ControlRequest::StartGame(spec), true).expect("StartGame");
    assert!(matches!(resp, Some(ControlResponse::Started { .. })), "hostd: {resp:?}");
    eprintln!("[{tag}] game {game_id} started, agent at {agent_addr}");
    let finished = loop {
        match read_msg::<ControlResponse>(&mut r).expect("hostd reply") {
            ControlResponse::Finished(f) if f.game_id == game_id => break f,
            _ => {}
        }
    };
    eprintln!("[{tag}] hostd: {:?} after {:?} (wall {} s)", finished.outcome, t0.elapsed(), finished.wall_s);
    let out = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    if let Some(rep) = &finished.replay {
        let _ = std::fs::write(out.join(format!("openbw_actions_{tag}.rep")), rep);
    }
    assert!(matches!(finished.outcome, GameOutcome::Finished | GameOutcome::Stopped), "game did not run: {:?}", finished.outcome);

    let reports = sessions.join().expect("agent thread");
    let st = reports
        .into_iter()
        .find_map(|r| match r {
            Ok(Report::Coverage(s)) => Some(s),
            _ => None,
        })
        .expect("no coverage session (player uranium-p0)");
    st
}

#[test]
#[ignore = "needs a running uranium-hostd with OpenBW (URANIUM_TEST_HOSTD)"]
fn every_action_type_runs_in_openbw_031_ac1() {
    let st = play(31, MAX_FRAMES, true, "031_ac1");
    let out = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    let md = report(&st);
    let _ = std::fs::write(out.join("openbw_actions_031_ac1.md"), &md);
    eprintln!("{md}");

    let missing: Vec<&str> = table().actions.iter().zip(&st.stats).filter(|(_, s)| s.ok == 0).map(|(a, _)| a.name.as_str()).collect();
    assert!(missing.is_empty(), "031-AC1: never accepted by the game: {missing:?}");
    // 031-AC1 (액션 감사 테스트 공백): beyond "accepted once", every rejection is a known,
    // justified one (`justified`); any other BWAPI error is a mask / executor / bridge mismatch.
    let mut bad = vec![];
    for (a, s) in table().actions.iter().zip(&st.stats) {
        let unexplained: Vec<u16> = s.errors.iter().copied().filter(|&e| justified(&a.name, e).is_none()).collect();
        if !unexplained.is_empty() {
            bad.push(format!("{} ok {} failed {} unexplained {unexplained:?}", a.name, s.ok, s.failed));
        }
    }
    assert!(bad.is_empty(), "031-AC1: rejections beyond the justified ones: {bad:?}");
    // 011-R20 / 액션 감사 F2: a resource seen before and now out of vision takes Smart_Unit and
    // Gather_Unit (the SCV heads for the minerals).
    for a in FOG_ACTIONS {
        assert!(st.fog_ok(a), "{a} on a resource out of vision: {:?}", st.fog);
    }
    // AMB-041-1: unarmed units given Attack_Pos are moving (BWAPI order Move).
    let mv = order_id("Move").unwrap();
    assert!(!st.unarmed.is_empty(), "no unarmed unit got Attack_Pos");
    for (ty, order) in &st.unarmed {
        assert_eq!(*order, mv, "{ty}: Attack_Pos did not become a move");
    }
}

/// 011-AC4 with group commands (011-R20, review 2026-10-19): the same seed and the same scripted
/// Act sequence give the same game twice — every observation's unit checksum and the GameEnd frame
/// and winner. The script plays 10 game minutes (no StopGame: its timing is wall clock).
#[test]
#[ignore = "needs a running uranium-hostd with OpenBW (URANIUM_TEST_HOSTD)"]
fn same_seed_same_acts_same_game_011_ac4() {
    let frames = 24 * 60 * 10;
    let a = play(41, frames, false, "011_ac4_a");
    let b = play(41, frames, false, "011_ac4_b");
    eprintln!("[011-AC4] a: {} observations, end {:?}; b: {} observations, end {:?}", a.sums.len(), a.end, b.sums.len(), b.end);
    let sent = |s: &State| s.stats.iter().map(|x| x.sent).sum::<u32>();
    eprintln!("[011-AC4] commands sent a {} b {}", sent(&a), sent(&b));
    let first_diff = a.sums.iter().zip(&b.sums).position(|(x, y)| x != y);
    assert_eq!(first_diff, None, "first differing observation: {:?} vs {:?}", first_diff.map(|i| a.sums[i]), first_diff.map(|i| b.sums[i]));
    assert_eq!(a.sums.len(), b.sums.len());
    assert_eq!(a.end, b.end, "GameEnd differs");
    assert!(sent(&a) > 100, "the script sent group commands");
}
