//! SC:R bridge logic on synthetic snapshots (012-R3, R4, R7, R8, R10, R11, R12, R15, R17, 011-R20).

use std::collections::HashMap;

use uranium_bridge_scr::commands::{self, BuildingMemory, Ctx};
use uranium_bridge_scr::end::EndTracker;
use uranium_bridge_scr::events::EventTracker;
use uranium_bridge_scr::game::Game;
use uranium_bridge_scr::observe::{ObserveState, derive, observe_units, raw_frame};
use uranium_bridge_scr::snapshot::{RawBullet, RawUnit, Snapshot, status};
use uranium_bridge_scr::tables::Tables;
use uranium_proto::*;

const ME: u8 = 3;
const ENEMY: u8 = 0;
const NEUTRAL: u8 = 11;
const W: u16 = 64;

fn t() -> &'static Tables {
    Tables::get().unwrap()
}

fn ty(name: &str) -> u16 {
    uranium_env::types::TypeTables::get().unit_by_name(name).unwrap().id as u16
}

fn order(name: &str) -> u8 {
    uranium_env::types::order_id(name).unwrap() as u8
}

fn unit(ptr: usize, tag: u32, player: u8, name: &str, pos: (i16, i16)) -> RawUnit {
    RawUnit {
        ptr,
        tag,
        player,
        unit_type: ty(name),
        pos,
        hp: 40 * 256,
        order: order("PlayerGuard"),
        status: status::COMPLETED,
        has_sprite: true,
        visibility_mask: 0xff,
        build_queue: [228; 5],
        ..Default::default()
    }
}

/// Every tile visible and explored for everyone.
fn snapshot(frame: u32, units: Vec<RawUnit>) -> Snapshot {
    let mut s = Snapshot { frame, local: ME, width: W, height: W, tiles: vec![0; W as usize * W as usize], units, ..Default::default() };
    s.index_units();
    s
}

fn hide_tile(s: &mut Snapshot, tx: usize, ty: usize) {
    s.tiles[ty * W as usize + tx] |= 1 << ME;
}

fn observe(s: &Snapshot) -> RawFrame {
    let d = derive(s, t());
    let units = observe_units(s, &d, t(), &ObserveState::default());
    raw_frame(s, t(), units, &[], vec![])
}

fn hex(v: &[u8]) -> String {
    v.iter().map(|b| format!("{b:02x}")).collect::<Vec<_>>().join(" ")
}

/// 012-R4 / 013-AC3: the bytes the spike sent and the game executed (reports/m7_scr_spike.md).
#[test]
fn command_bytes_match_the_spike_013_ac3() {
    let t = t();
    assert_eq!(hex(&commands::select(t, &[0x2ccb])), "63 01 cb 2c 00 00");
    let chat = commands::chat(t, 3, "uranium gg");
    assert_eq!(chat.len(), 82);
    assert_eq!(hex(&chat[..12]), "5c 03 75 72 61 6e 69 75 6d 20 67 67");
    assert!(chat[12..].iter().all(|&b| b == 0));
    // RightClick (x 3992, y 3848, no target) through the encoder.
    let s = snapshot(10, vec![unit(0x1000, 0x2ccb, ME, "Terran_SCV", (3832, 3848))]);
    let d = derive(&s, t);
    let observed = HashMap::from([(0x2ccb, 0)]);
    let empty = HashMap::new();
    let ctx = Ctx { s: &s, d: &d, t, observed: &observed, resources: &empty, remembered: &HashMap::new(), buildings: &BuildingMemory::new() };
    let out = commands::encode(&ctx, &[Command::Unit { units: vec![0x2ccb], order: CmdOrder::RightClick(Target::Pos((3992, 3848))), queued: false }]);
    assert_eq!(out.iter().map(|p| hex(p)).collect::<Vec<_>>(), ["63 01 cb 2c 00 00", "60 98 0f 08 0f 00 00 00 00 e4 00 00"]);
}

fn ctx_encode(s: &Snapshot, cmds: &[Command], remembered: HashMap<u32, Pos>) -> Vec<Vec<u8>> {
    ctx_encode_with(s, cmds, remembered, &BuildingMemory::new())
}

fn ctx_encode_with(s: &Snapshot, cmds: &[Command], remembered: HashMap<u32, Pos>, buildings: &BuildingMemory) -> Vec<Vec<u8>> {
    let t = t();
    let d = derive(s, t);
    let observed: HashMap<u32, usize> = s.units.iter().enumerate().filter(|(i, _)| d[*i].observed).map(|(i, u)| (u.tag, i)).collect();
    let resources: HashMap<u32, usize> = s.units.iter().enumerate().filter(|(_, u)| u.player == NEUTRAL).map(|(i, u)| (u.tag, i)).collect();
    let ctx = Ctx { s, d: &d, t, observed: &observed, resources: &resources, remembered: &remembered, buildings };
    commands::encode(&ctx, cmds)
}

/// A unit whose sprite we do not see (in the fog for the observation filter, 012-R3).
fn fogged(mut u: RawUnit) -> RawUnit {
    u.visibility_mask = 0;
    u
}

fn le16(x: u16) -> String {
    hex(&x.to_le_bytes())
}

/// 012-R4: one select (at most 12 units) then one order per packet group; buildings alone;
/// unarmed units attack-move as a move; other players' units are never commanded.
#[test]
fn unit_commands_group_and_convert_012_r4() {
    let mut units: Vec<RawUnit> = (0..13).map(|i| unit(0x100 + i, 10 + i as u32, ME, "Terran_Marine", (100, 100))).collect();
    units.push(unit(0x200, 50, ME, "Terran_Dropship", (200, 200)));
    units.push(unit(0x300, 60, ENEMY, "Terran_Marine", (300, 300)));
    let s = snapshot(10, units);
    let all: Vec<u32> = (10..23).collect();
    let out = ctx_encode(&s, &[Command::Unit { units: all, order: CmdOrder::Move(Target::Pos((640, 480))), queued: true }], HashMap::new());
    assert_eq!(out.len(), 2);
    assert_eq!((out[0][0], out[0][1]), (0x63, 12), "12 units per selection");
    // 0x61 x 640 y 480 tag 0, type 0xe4, order Move (6), queued 1.
    assert_eq!(hex(&out[1]), "61 80 02 e0 01 00 00 00 00 e4 00 06 01");
    let out = ctx_encode(&s, &[Command::Unit { units: vec![10, 50, 60], order: CmdOrder::Attack(Target::Pos((640, 480))), queued: false }], HashMap::new());
    // Marine attack-moves (14), the dropship moves (6), the enemy marine gets nothing.
    let orders: Vec<(u8, u8)> = out.chunks(2).map(|c| (c[0][1], c[1][11])).collect();
    assert_eq!(orders, vec![(1, 14), (1, 6)]);
    // Attacking a unit sends its tag and position; a building attacks with TowerAttack.
    let mut s2 = s.clone();
    s2.units.push(unit(0x400, 70, ME, "Terran_Bunker", (500, 500)));
    s2.index_units();
    let out = ctx_encode(&s2, &[Command::Unit { units: vec![10, 70], order: CmdOrder::Attack(Target::Unit(60)), queued: false }], HashMap::new());
    let attack: Vec<u8> = out.iter().filter(|p| p[0] == 0x61).map(|p| p[11]).collect();
    assert_eq!(attack, vec![order("TowerAttack"), order("Attack1")]);
    let marine_attack = out.iter().find(|p| p[0] == 0x61 && p[11] == order("Attack1")).unwrap();
    assert_eq!(hex(&marine_attack[1..9]), "2c 01 2c 01 3c 00 00 00", "target position and tag");
}

/// 011-R20: a remembered resource that is gone turns a gather into a move to where it was; an
/// unknown target sends nothing.
#[test]
fn remembered_resource_011_r20() {
    let s = snapshot(10, vec![unit(0x100, 10, ME, "Terran_SCV", (100, 100))]);
    let gather = |id| [Command::Unit { units: vec![10], order: CmdOrder::Gather(id), queued: false }];
    let out = ctx_encode(&s, &gather(99), HashMap::from([(99, (320, 400))]));
    assert_eq!(out.len(), 2);
    assert_eq!(out[1][11], order("Move"));
    assert_eq!(hex(&out[1][1..5]), "40 01 90 01");
    assert!(ctx_encode(&s, &gather(98), HashMap::new()).is_empty());
    // A fogged mineral field still alive is a valid target (map-public, 010-R5 exception).
    let mut s2 = snapshot(10, vec![unit(0x100, 10, ME, "Terran_SCV", (100, 100)), unit(0x500, 97, NEUTRAL, "Resource_Mineral_Field", (900, 900))]);
    hide_tile(&mut s2, 28, 28);
    let out = ctx_encode(&s2, &gather(97), HashMap::new());
    assert_eq!((out[1][11], out[1][5]), (order("Harvest1"), 97));
}

fn hexes(out: &[Vec<u8>]) -> Vec<String> {
    out.iter().map(|p| hex(p)).collect()
}

/// 012-R17 / 011-R20 ①-2: a remembered enemy building out of sight is commanded through its fog
/// ghost (last position, unit type, tag 0). The bytes do not depend on the real building: gone,
/// still there in the fog, or moved in the fog.
#[test]
fn fog_ghost_commands_012_r17() {
    let rax = ty("Terran_Barracks");
    let record = BuildingMemory::from([(77, ((640, 480), rax))]);
    let mine = || {
        vec![
            unit(0x100, 10, ME, "Terran_Marine", (100, 100)),
            unit(0x110, 11, ME, "Terran_Marine", (110, 100)),
            unit(0x200, 50, ME, "Terran_Dropship", (200, 200)),
            unit(0x300, 60, ME, "Terran_Barracks", (300, 300)),
            unit(0x400, 70, ME, "Terran_SCV", (120, 100)),
        ]
    };
    let cmd = |order, queued| [Command::Unit { units: vec![10, 11, 50], order, queued }];
    // x 640, y 480, tag 0, then the unit type.
    let ghost = format!("80 02 e0 01 00 00 00 00 {}", le16(rax));
    let three = "63 03 0a 00 00 00 0b 00 00 00 32 00 00 00";
    let worlds = [
        mine(),
        [mine(), vec![fogged(unit(0x500, 77, ENEMY, "Terran_Barracks", (640, 480)))]].concat(),
        [mine(), vec![fogged(unit(0x500, 77, ENEMY, "Terran_Barracks", (1200, 900)))]].concat(),
    ];
    for units in worlds {
        let s = snapshot(10, units);
        let enc = |cmds: &[Command]| hexes(&ctx_encode_with(&s, cmds, HashMap::new(), &record));
        // Attack -> MoveToAttack (9) for the units that can attack-move; the dropship gets nothing.
        assert_eq!(enc(&cmd(CmdOrder::Attack(Target::Unit(77)), false)), ["63 02 0a 00 00 00 0b 00 00 00".to_string(), format!("61 {ghost} 09 00")]);
        assert_eq!(enc(&cmd(CmdOrder::Attack(Target::Unit(77)), true))[1], format!("61 {ghost} 09 01"), "queued");
        // RightClick -> a typed right click for every unit (the game picks each unit's order).
        assert_eq!(enc(&cmd(CmdOrder::RightClick(Target::Unit(77)), false)), [three.to_string(), format!("60 {ghost} 00")]);
        // Move -> a move to the last position.
        assert_eq!(enc(&cmd(CmdOrder::Move(Target::Unit(77)), false)), [three, "61 80 02 e0 01 00 00 00 00 e4 00 06 00"]);
        // SetRally -> a rally to the last position.
        let rally = enc(&[Command::SetRally { building: 60, target: Target::Unit(77) }]);
        assert_eq!(rally, ["63 01 3c 00 00 00".to_string(), format!("61 80 02 e0 01 00 00 00 00 e4 00 {:02x} 00", order("RallyPointTile"))]);
        // Other orders on a record-only target send nothing (011-R20 ②).
        assert!(enc(&[Command::Unit { units: vec![70], order: CmdOrder::Repair(77), queued: false }]).is_empty());
        assert!(enc(&[Command::Unit { units: vec![70], order: CmdOrder::Gather(77), queued: false }]).is_empty());
    }
    // In sight, the building is an observed unit: the ordinary attack with its tag.
    let s = snapshot(10, [mine(), vec![unit(0x500, 77, ENEMY, "Terran_Barracks", (1200, 900))]].concat());
    let out = ctx_encode_with(&s, &cmd(CmdOrder::Attack(Target::Unit(77)), false), HashMap::new(), &record);
    assert_eq!((out[1][11], hex(&out[1][1..9])), (order("Attack1"), "b0 04 84 03 4d 00 00 00".to_string()));
    // Not observed and not remembered: nothing (no map hack).
    let s = snapshot(10, [mine(), vec![fogged(unit(0x500, 77, ENEMY, "Terran_Barracks", (640, 480)))]].concat());
    assert!(ctx_encode(&s, &cmd(CmdOrder::Attack(Target::Unit(77)), false), HashMap::new()).is_empty());
}

/// 012-R17: a dead enemy refinery leaves a neutral geyser with the same tag. Remembered as a
/// refinery, the target stays a fog ghost: resolving it as the live geyser first would tell the
/// refinery is gone.
#[test]
fn refinery_turned_geyser_stays_a_ghost_012_r17() {
    let refinery = ty("Terran_Refinery");
    let record = BuildingMemory::from([(99, ((900, 900), refinery))]);
    let s = snapshot(10, vec![unit(0x100, 10, ME, "Terran_SCV", (100, 100)), fogged(unit(0x500, 99, NEUTRAL, "Resource_Vespene_Geyser", (900, 900)))]);
    let right = [Command::Unit { units: vec![10], order: CmdOrder::RightClick(Target::Unit(99)), queued: false }];
    let out = ctx_encode_with(&s, &right, HashMap::new(), &record);
    assert_eq!(hex(&out[1]), format!("60 84 03 84 03 00 00 00 00 {} 00", le16(refinery)));
    let gather = [Command::Unit { units: vec![10], order: CmdOrder::Gather(99), queued: false }];
    assert!(ctx_encode_with(&s, &gather, HashMap::new(), &record).is_empty());
    // Without the record the same tag is the map's geyser (011-R20 ①).
    let out = ctx_encode(&s, &gather, HashMap::new());
    assert_eq!((out[1][11], out[1][5]), (order("Harvest1"), 99));
}

/// 012-R17: the record holds observed enemy buildings (last position, type); a building in the
/// fog keeps its entry; a tag observed as anything else is dropped; own, allied, neutral and
/// unobserved units are never added.
#[test]
fn building_record_rules_012_r17() {
    let t = t();
    let rax = ty("Terran_Barracks");
    let step = |rec: &mut BuildingMemory, s: Snapshot| commands::remember_buildings(&s, &derive(&s, t), rec);
    let mut rec = BuildingMemory::new();
    let mut s = snapshot(
        1,
        vec![
            unit(0x100, 1, ME, "Terran_Supply_Depot", (100, 100)),
            unit(0x200, 77, ENEMY, "Terran_Barracks", (640, 480)),
            unit(0x300, 78, ENEMY, "Terran_Marine", (300, 300)),
            unit(0x400, 99, NEUTRAL, "Resource_Vespene_Geyser", (900, 900)),
            fogged(unit(0x500, 79, ENEMY, "Terran_Factory", (1000, 1000))),
            unit(0x700, 80, 1, "Terran_Barracks", (800, 800)),
        ],
    );
    s.allies[1] = true;
    step(&mut rec, s);
    assert_eq!(rec, BuildingMemory::from([(77, ((640, 480), rax))]));
    // Lifted and moved in the fog: still the last position seen.
    step(&mut rec, snapshot(2, vec![fogged(unit(0x200, 77, ENEMY, "Terran_Barracks", (1200, 900)))]));
    assert_eq!(rec, BuildingMemory::from([(77, ((640, 480), rax))]));
    step(&mut rec, snapshot(3, vec![]));
    assert_eq!(rec.len(), 1, "a building that disappears in the fog stays");
    // Seen again: updated.
    step(&mut rec, snapshot(4, vec![unit(0x200, 77, ENEMY, "Terran_Barracks", (1200, 900))]));
    assert_eq!(rec, BuildingMemory::from([(77, ((1200, 900), rax))]));
    // The same tag seen as a neutral geyser: dropped.
    step(&mut rec, snapshot(5, vec![unit(0x200, 77, NEUTRAL, "Resource_Vespene_Geyser", (1200, 900))]));
    assert!(rec.is_empty());
}

/// 012-R17 through `Game`: the record fills from observed enemy buildings and drives a ghost
/// attack once the building is in the fog; a new game starts with an empty record.
#[test]
fn game_keeps_a_building_record_012_r17() {
    let marine = || unit(0x100, 10, ME, "Terran_Marine", (100, 100));
    let rax = || unit(0x500, 77, ENEMY, "Terran_Barracks", (640, 480));
    let attack = || vec![Command::Unit { units: vec![10], order: CmdOrder::Attack(Target::Unit(77)), queued: false }];
    let mut g = Game::new(t(), ME, Some(ENEMY));
    g.frame(&snapshot(1, vec![marine(), rax()]), None);
    let o = g.frame(&snapshot(3, vec![marine(), fogged(rax())]), Some(act(attack(), 25)));
    assert_eq!(hexes(&o.packets), ["63 01 0a 00 00 00".to_string(), format!("61 80 02 e0 01 00 00 00 00 {} 09 00", le16(ty("Terran_Barracks")))]);
    let mut g = Game::new(t(), ME, Some(ENEMY));
    g.frame(&snapshot(1, vec![marine(), fogged(rax())]), None);
    let o = g.frame(&snapshot(3, vec![marine(), fogged(rax())]), Some(act(attack(), 25)));
    assert!(o.packets.is_empty(), "never observed in this game");
}

/// 012-R17: the record follows the observations sent to the agent (012-R3 관측마다, as the OpenBW
/// shim's observe()). A building seen only on a frame without an observation is not recorded; the
/// next observation that contains it is.
#[test]
fn building_record_follows_observations_012_r17() {
    let marine = || unit(0x100, 10, ME, "Terran_Marine", (100, 100));
    let rax = || unit(0x500, 77, ENEMY, "Terran_Barracks", (640, 480));
    let attack = || vec![Command::Unit { units: vec![10], order: CmdOrder::Attack(Target::Unit(77)), queued: false }];
    let mut g = Game::new(t(), ME, Some(ENEMY));
    assert_eq!(g.frame(&snapshot(1, vec![marine()]), None).to_agent.len(), 1, "observation at frame 1");
    assert!(g.frame(&snapshot(2, vec![marine(), rax()]), None).to_agent.is_empty(), "frame 2 is not an observation");
    let o = g.frame(&snapshot(3, vec![marine(), fogged(rax())]), Some(act(attack(), 25)));
    assert!(o.packets.is_empty(), "seen between observations only: not remembered");
    // The Act moved the next observation to 1 + 8 = 9; there the barracks is in sight.
    assert_eq!(g.frame(&snapshot(9, vec![marine(), rax()]), None).to_agent.len(), 1);
    let o = g.frame(&snapshot(10, vec![marine(), fogged(rax())]), Some(act(attack(), 30)));
    assert_eq!(hexes(&o.packets), ["63 01 0a 00 00 00".to_string(), format!("61 80 02 e0 01 00 00 00 00 {} 09 00", le16(ty("Terran_Barracks")))]);
}

/// 012-R3 / 010-R5: own units always; others only when their sprite is visible to us, a
/// footprint tile is visible, and (cloaked or burrowed) they are detected.
#[test]
fn observation_filter_012_r3() {
    let mut cloaked = unit(0x300, 30, ENEMY, "Terran_Wraith", (600, 600));
    cloaked.status |= status::CLOAKED | status::REQUIRES_DETECTION;
    cloaked.detection_status = 0;
    let mut hidden_mask = unit(0x400, 40, ENEMY, "Terran_Marine", (700, 700));
    hidden_mask.visibility_mask = 0;
    let mut s = snapshot(
        10,
        vec![unit(0x100, 10, ME, "Terran_Marine", (100, 100)), unit(0x200, 20, ENEMY, "Terran_Marine", (300, 300)), unit(0x210, 21, ENEMY, "Terran_Marine", (1000, 1000)), cloaked, hidden_mask],
    );
    // BWAPI footprint tile = (pos - tile size * 16) / 32: the 1x1 marine at 1000 is on tile 30.
    hide_tile(&mut s, 30, 30);
    hide_tile(&mut s, 3, 3);
    let f = observe(&s);
    let ids: Vec<u32> = f.units.iter().map(|u| u.id).collect();
    assert_eq!(ids, vec![10, 20], "own unit even on a hidden tile; seen enemy; not the fogged, undetected or unseen ones");
    // Detected, the cloaked wraith is observed with CLOAKED | DETECTED.
    s.units[3].detection_status = 1 << ME;
    s.index_units();
    let f = observe(&s);
    let w = f.units.iter().find(|u| u.id == 30).expect("detected wraith");
    assert!(w.has(flags::CLOAKED) && w.has(flags::DETECTED));
}

/// AMB-030-4: a bullet is an effect only when its centre tile is visible; storms keep their kind.
#[test]
fn effects_need_a_visible_centre_amb_030_4() {
    let mut s = snapshot(10, vec![]);
    s.bullets = vec![
        RawBullet { flingy_id: 157, pos: (320, 320), remove_timer: 40, parent: 0 },
        RawBullet { flingy_id: 157, pos: (960, 960), remove_timer: 40, parent: 0 },
        RawBullet { flingy_id: 99, pos: (100, 100), remove_timer: 3, parent: 0 },
    ];
    hide_tile(&mut s, 30, 30);
    let f = observe(&s);
    let kinds: Vec<(EffectKind, Pos)> = f.effects.iter().map(|e| (e.kind, e.pos)).collect();
    assert_eq!(kinds, vec![(EffectKind::Other(99), (100, 100)), (EffectKind::PsionicStorm, (320, 320))]);
}

/// 012-R12: BWAPI unit values: hp rounded up, angle = facing - 64, Attack1 -> AttackUnit,
/// UNDER_ATTACK never set, a loaded unit at its transport (named only if observed).
#[test]
fn unit_values_follow_bwapi_012_r12() {
    let mut m = unit(0x100, 10, ME, "Terran_Marine", (100, 100));
    m.hp = 40 * 256 - 10;
    m.facing = 10;
    m.order = order("Attack1");
    let mut loaded = unit(0x110, 11, ME, "Terran_Marine", (0, 0));
    loaded.status |= status::IN_TRANSPORT;
    loaded.related = 0x120;
    let ship = unit(0x120, 12, ME, "Terran_Dropship", (400, 300));
    let f = observe(&snapshot(10, vec![m, loaded, ship]));
    let u = &f.units[0];
    assert_eq!((u.hp, u.angle, u.order), (40, 10u8.wrapping_sub(64), order("AttackUnit") as u16));
    assert!(!u.has(flags::UNDER_ATTACK));
    let l = &f.units[1];
    assert_eq!((l.pos, l.loaded_in), ((400, 300), Some(12)));
    assert!(l.has(flags::LOADED));
}

/// GameUnits.cpp augment: an SCV constructing and its building are both CONSTRUCTING, not idle.
#[test]
fn builder_and_building_construct_012_r12() {
    let mut scv = unit(0x100, 10, ME, "Terran_SCV", (100, 100));
    scv.order = order("ConstructingBuilding");
    scv.order_target = 0x200;
    let mut depot = unit(0x200, 20, ME, "Terran_Supply_Depot", (160, 160));
    depot.status = 0;
    depot.order = order("IncompleteBuilding");
    depot.remaining_build_time = 300;
    let f = observe(&snapshot(10, vec![scv, depot]));
    for u in &f.units {
        assert!(u.has(flags::CONSTRUCTING) && !u.has(flags::IDLE), "{}", u.id);
    }
    assert_eq!(f.units[1].remaining_build, 300);
    assert!(!f.units[1].has(flags::COMPLETED));
}

/// Events: Create + Discover for a new observed unit; Destroy + Evade when a unit we saw dies; a
/// unit never seen leaves no trace.
#[test]
fn events_follow_the_openbw_bridge_012_r12() {
    let t = t();
    let mut ev = EventTracker::default();
    let mut hidden = unit(0x300, 30, ENEMY, "Terran_Marine", (1000, 1000));
    hidden.visibility_mask = 0;
    let s1 = snapshot(1, vec![unit(0x100, 10, ME, "Terran_SCV", (100, 100)), unit(0x200, 20, ENEMY, "Terran_Marine", (300, 300)), hidden]);
    ev.update(&s1, &derive(&s1, t));
    let mut e = ev.take();
    e.sort_by_key(uranium_bridge_scr::observe::event_order);
    assert_eq!(e, vec![GameEvent::UnitCreated(10), GameEvent::UnitCreated(20), GameEvent::UnitDiscover(10), GameEvent::UnitDiscover(20)]);
    let s2 = snapshot(2, vec![unit(0x100, 10, ME, "Terran_SCV", (100, 100))]);
    ev.update(&s2, &derive(&s2, t));
    let mut e = ev.take();
    e.sort_by_key(uranium_bridge_scr::observe::event_order);
    assert_eq!(e, vec![GameEvent::UnitDestroyed(20), GameEvent::UnitEvade(20)]);
}

/// 012-R15: victory / defeat from `victory_state`; an enemy leaving with buildings is a Surrender.
#[test]
fn game_end_012_r15() {
    let t = t();
    let s = |v: [u8; 8], types: [u8; 12], units: Vec<RawUnit>| {
        let mut s = snapshot(100, units);
        s.victory_state = v;
        s.player_types = types;
        s
    };
    let playing = [2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 7];
    let rax = || vec![unit(0x500, 50, ENEMY, "Terran_Barracks", (500, 500))];
    let mut end = EndTracker::new(ME, Some(ENEMY));
    let a = s([0; 8], playing, rax());
    assert_eq!(end.update(&a, &derive(&a, t)), None);
    let mut left = playing;
    left[ENEMY as usize] = 10;
    let mut v = [0; 8];
    v[ME as usize] = 3;
    let b = s(v, left, rax());
    assert_eq!(end.update(&b, &derive(&b, t)), Some((Some(ME), EndReason::Surrender)));
    let mut end = EndTracker::new(ME, Some(ENEMY));
    let mut v = [0; 8];
    v[ME as usize] = 2;
    let c = s(v, playing, vec![]);
    assert_eq!(end.update(&c, &derive(&c, t)), Some((Some(ENEMY), EndReason::Elimination)));
    v[ENEMY as usize] = 2;
    let d = s(v, playing, vec![]);
    assert_eq!(EndTracker::new(ME, Some(ENEMY)).update(&d, &derive(&d, t)), Some((None, EndReason::Elimination)), "both defeated: draw");
}

fn act(cmds: Vec<Command>, valid_until: u32) -> AgentMsg {
    AgentMsg::Act(Act { cmds, advance_frames: 8, valid_until_frame: Some(valid_until), surrender: None })
}

/// 012-R7 / R8 / R11 / R10 through `Game`: first observation at frame 1, commands within the
/// deadline, a late Act dropped, surrender = chat + GameEnd(Surrender) with a final frame, then
/// nothing more.
#[test]
fn game_flow_012_r11() {
    let mut g = Game::new(t(), ME, Some(ENEMY));
    let units = || vec![unit(0x100, 10, ME, "Terran_SCV", (100, 100))];
    assert!(g.frame(&snapshot(0, units()), None).to_agent.is_empty());
    let o = g.frame(&snapshot(1, units()), None);
    assert!(matches!(o.to_agent.as_slice(), [BridgeMsg::Frame(f)] if f.frame == 1 && f.units.len() == 1));
    let mv = vec![Command::Unit { units: vec![10], order: CmdOrder::Move(Target::Pos((50, 50))), queued: false }];
    let o = g.frame(&snapshot(3, units()), Some(act(mv.clone(), 25)));
    assert_eq!(o.packets.len(), 2, "select + move");
    assert!(g.frame(&snapshot(8, units()), None).to_agent.is_empty(), "next observation at f + 8 = 9");
    let o = g.frame(&snapshot(9, units()), None);
    assert_eq!(o.to_agent.len(), 1);
    let o = g.frame(&snapshot(40, units()), Some(act(mv, 33)));
    assert!(o.packets.is_empty(), "expired Act dropped");
    assert_eq!(g.schedule.expired, 1);
    // The late Act moved the next observation to frame 40: observed at once.
    assert_eq!(o.to_agent.len(), 1);
    let gg = AgentMsg::Act(Act { cmds: vec![], advance_frames: 1, valid_until_frame: Some(64), surrender: Some("gg".into()) });
    let o = g.frame(&snapshot(41, units()), Some(gg));
    assert_eq!(o.packets.len(), 1);
    assert_eq!(&o.packets[0][..4], &[0x5c, ME, b'g', b'g']);
    match o.to_agent.as_slice() {
        [BridgeMsg::End(e)] => {
            assert_eq!((e.winner, e.reason, e.frame), (Some(ENEMY), EndReason::Surrender, 41));
            assert!(e.final_frame.is_some(), "010-R12 final frame");
        }
        other => panic!("expected GameEnd, got {other:?}"),
    }
    assert!(g.done);
    let o = g.frame(&snapshot(42, units()), Some(act(vec![], 70)));
    assert!(o.packets.is_empty() && o.to_agent.is_empty(), "nothing after surrender");
}
