//! Spec 010 acceptance tests: AC1 (round-trip), AC2 (handshake rejection), AC3 (size/time).

use std::time::Instant;

use proptest::prelude::*;
use uranium_proto::*;

fn arb_pos() -> impl Strategy<Value = Pos> {
    (any::<u16>(), any::<u16>())
}

fn arb_target() -> impl Strategy<Value = Target> {
    prop_oneof![any::<u32>().prop_map(Target::Unit), arb_pos().prop_map(Target::Pos)]
}

fn arb_grid() -> impl Strategy<Value = BitGrid> {
    (1u16..64, 1u16..64).prop_flat_map(|(w, h)| {
        prop::collection::vec(any::<u8>(), (w as usize * h as usize).div_ceil(8)).prop_map(move |bits| BitGrid { width: w, height: h, bits })
    })
}

fn arb_player_state() -> impl Strategy<Value = PlayerState> {
    (any::<i32>(), any::<i32>(), any::<u16>(), any::<u16>(), prop::collection::vec(0u8..4, 63), any::<u64>(), any::<u64>(), any::<u64>()).prop_map(
        |(minerals, gas, supply_used, supply_total, upgrades, tech, researching, upgrading)| PlayerState {
            minerals,
            gas,
            supply_used,
            supply_total,
            upgrades,
            tech,
            researching,
            upgrading,
        },
    )
}

fn arb_unit() -> impl Strategy<Value = UnitState> {
    (
        (any::<u32>(), any::<u8>(), any::<u16>(), arb_pos(), (any::<i16>(), any::<i16>()), any::<u8>()),
        (any::<u16>(), any::<u16>(), any::<u16>(), any::<u16>(), any::<Option<u32>>(), prop::option::of(arb_pos())),
        (any::<u8>(), any::<u8>(), any::<u8>(), any::<u32>(), any::<[u16; 5]>()),
        (any::<u16>(), any::<u16>(), any::<u16>(), any::<u16>(), any::<Option<u32>>(), any::<u8>(), any::<bool>(), any::<bool>()),
        (any::<u16>(), any::<u16>(), any::<bool>()),
    )
        .prop_map(|(a, b, c, d, e)| UnitState {
            id: a.0,
            owner: a.1,
            unit_type: a.2,
            pos: a.3,
            vel: a.4,
            angle: a.5,
            hp: b.0,
            shield: b.1,
            energy: b.2,
            order: b.3,
            order_target: b.4,
            order_pos: b.5,
            ground_cd: c.0,
            air_cd: c.1,
            spell_cd: c.2,
            flags: c.3,
            build_queue: c.4,
            remaining_train: d.0,
            remaining_build: d.1,
            remaining_research: d.2,
            resources: d.3,
            loaded_in: d.4,
            spider_mines: d.5,
            nuke_armed: d.6,
            has_addon: d.7,
            status_timers: StatusTimers { lockdown: e.0, stim: e.1, blind: e.2, ..Default::default() },
        })
}

fn arb_event() -> impl Strategy<Value = GameEvent> {
    prop_oneof![
        any::<u32>().prop_map(GameEvent::UnitCreated),
        any::<u32>().prop_map(GameEvent::UnitDestroyed),
        arb_pos().prop_map(GameEvent::NukeDetect),
        (any::<u16>(), any::<u16>()).prop_map(|(cmd_index, error)| GameEvent::CommandFailed { cmd_index, error }),
        // 010-AC9: unit_type present and absent.
        (any::<u16>(), any::<u32>(), prop::option::of(any::<u16>()), any::<u8>(), any::<u16>())
            .prop_map(|(cmd_index, unit, unit_type, command, error)| GameEvent::UnitRejected { cmd_index, unit, unit_type, command, error }),
    ]
}

fn arb_effect() -> impl Strategy<Value = Effect> {
    (prop_oneof![Just(EffectKind::PsionicStorm), Just(EffectKind::DarkSwarm), any::<u16>().prop_map(EffectKind::Other)], arb_pos(), any::<u16>(), any::<Option<u16>>(), any::<Option<u8>>())
        .prop_map(|(kind, pos, radius, remaining_frames, source_owner)| Effect { kind, pos, radius, remaining_frames, source_owner })
}

fn arb_frame() -> impl Strategy<Value = RawFrame> {
    (any::<u32>(), arb_player_state(), prop::collection::vec(arb_unit(), 0..40), arb_grid(), arb_grid(), arb_grid(), prop::collection::vec(arb_event(), 0..8), prop::collection::vec(arb_effect(), 0..4))
        .prop_map(|(frame, self_state, units, visibility, explored, creep, events, effects)| RawFrame { frame, self_state, units, visibility, explored, creep, events, effects })
}

fn arb_order() -> impl Strategy<Value = CmdOrder> {
    prop_oneof![
        arb_target().prop_map(CmdOrder::Move),
        arb_target().prop_map(CmdOrder::Attack),
        arb_pos().prop_map(CmdOrder::Patrol),
        Just(CmdOrder::Hold),
        Just(CmdOrder::Siege),
        (any::<u16>(), any::<u16>()).prop_map(CmdOrder::Land),
        prop::option::of(arb_pos()).prop_map(CmdOrder::UnloadAll),
        any::<u32>().prop_map(CmdOrder::Unload),
        any::<u32>().prop_map(CmdOrder::Gather),
        (any::<u16>(), prop::option::of(arb_target())).prop_map(|(tech, target)| CmdOrder::UseTech { tech, target }),
    ]
}

fn arb_command() -> impl Strategy<Value = Command> {
    prop_oneof![
        (prop::collection::vec(any::<u32>(), 1..=12), arb_order(), any::<bool>()).prop_map(|(units, order, queued)| Command::Unit { units, order, queued }),
        (any::<u32>(), any::<u16>(), (any::<u16>(), any::<u16>())).prop_map(|(worker, unit_type, tile)| Command::Build { worker, unit_type, tile }),
        (any::<u32>(), any::<u16>()).prop_map(|(building, unit_type)| Command::Train { building, unit_type }),
        (any::<u32>(), any::<u16>()).prop_map(|(building, tech)| Command::Research { building, tech }),
        (any::<u32>(), prop_oneof![Just(CancelKind::Train), Just(CancelKind::Construction)]).prop_map(|(unit, kind)| Command::Cancel { unit, kind }),
        (any::<u32>(), arb_target()).prop_map(|(building, target)| Command::SetRally { building, target }),
    ]
}

fn arb_bridge_msg() -> impl Strategy<Value = BridgeMsg> {
    prop_oneof![
        (any::<u32>(), any::<u64>(), any::<u8>(), any::<bool>(), "[a-z0-9.]{0,12}").prop_map(|(v, game_id, slot, replay, game_version)| BridgeMsg::Hello(Hello {
            protocol_version: v,
            backend: Backend::OpenBw,
            game_version,
            mode: Mode::Lockstep,
            game_id,
            slot,
            replay
        })),
        (arb_grid(), arb_grid(), any::<[u8; 32]>(), "\\PC{0,16}").prop_map(|(walkable, buildable, map_hash, map_name)| BridgeMsg::GameInfo(Box::new(GameInfo {
            map_name,
            map_hash,
            map_size_tiles: (buildable.width, buildable.height),
            self_player: 0,
            players: vec![PlayerInfo { id: 0, name: "p".into(), race: Race::Terran, start_tile: Some((1, 2)), is_self: true, is_enemy: false }],
            static_map: StaticMap {
                width_tiles: buildable.width,
                height_tiles: buildable.height,
                ground_height: vec![0; 4],
                start_locations: vec![(3, 4)],
                walkable,
                buildable,
            },
            latency_frames: 0,
        }))),
        (any::<u32>(), arb_player_state(), prop::collection::vec((any::<u16>(), any::<u16>()), 0..10)).prop_map(|(frame, enemy, c)| BridgeMsg::Privileged(PrivilegedSummary {
            frame,
            enemy,
            enemy_unit_counts: c.clone(),
            enemy_in_production: c
        })),
        arb_frame().prop_map(|f| BridgeMsg::Frame(Box::new(f))),
        (any::<Option<u8>>(), any::<u32>(), any::<f64>(), prop::option::of(arb_frame())).prop_map(|(winner, frame, t, last)| BridgeMsg::End(GameEnd {
            winner,
            frame,
            reason: EndReason::TimeLimitDraw,
            stats: Some(BridgeStats { game_frames: frame, game_time_ms: t, ..Default::default() }),
            // 010-R12: the final frame round-trips with and without content.
            final_frame: last.map(Box::new),
        })),
    ]
}

fn arb_agent_msg() -> impl Strategy<Value = AgentMsg> {
    prop_oneof![
        (prop::collection::vec(arb_command(), 0..20), 1u32..64, any::<Option<u32>>()).prop_map(|(cmds, advance_frames, valid_until_frame)| AgentMsg::Act(Act {
            cmds,
            advance_frames,
            valid_until_frame,
            surrender: None
        })),
        // 010-AC8: a surrender Act (no commands) with a chat line, possibly empty.
        (1u32..64, prop::option::of("\\PC{0,20}")).prop_map(|(advance_frames, surrender)| AgentMsg::Act(Act { cmds: vec![], advance_frames, valid_until_frame: None, surrender })),
        "\\PC{0,32}".prop_map(AgentMsg::Reject),
    ]
}

fn roundtrip<T: serde::Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug>(m: &T) {
    let mut buf = Vec::new();
    write_msg(&mut buf, m).unwrap();
    let back: T = read_msg(&mut buf.as_slice()).unwrap();
    assert_eq!(&back, m);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10_000))]

    /// 010-AC1
    #[test]
    fn ac1_bridge_messages_roundtrip(m in arb_bridge_msg()) {
        roundtrip(&m);
    }

    /// 010-AC1
    #[test]
    fn ac1_agent_messages_roundtrip(m in arb_agent_msg()) {
        roundtrip(&m);
    }
}

/// 010-AC2
#[test]
fn ac2_version_mismatch_is_rejected() {
    let mut h = Hello { protocol_version: PROTOCOL_VERSION, backend: Backend::OpenBw, game_version: "x".into(), mode: Mode::Lockstep, game_id: 0, slot: 0, replay: false };
    assert!(check_hello(&h).is_ok());
    h.protocol_version += 1;
    let err = check_hello(&h).unwrap_err();
    assert!(err.contains("protocol version mismatch"), "{err}");
    // 010-R12: a peer still on version 1 (GameEnd without final_frame) is rejected too.
    h.protocol_version = 1;
    assert!(PROTOCOL_VERSION >= 2 && check_hello(&h).unwrap_err().contains("protocol version mismatch"));
}

/// 010-AC9 (010-R14): `UnitRejected` is protocol 5, a version-4 peer is rejected, and the new
/// variant is appended last so frames stored before it decode unchanged.
#[test]
fn ac9_unit_rejected_version_and_variant_order() {
    assert_eq!(PROTOCOL_VERSION, 5);
    let h = Hello { protocol_version: 4, backend: Backend::OpenBw, game_version: "x".into(), mode: Mode::Lockstep, game_id: 0, slot: 0, replay: false };
    assert!(check_hello(&h).unwrap_err().contains("protocol version mismatch"));
    // Existing variants keep their postcard index (CommandFailed is 6), UnitRejected is 7.
    let failed = postcard::to_allocvec(&GameEvent::CommandFailed { cmd_index: 1, error: 2 }).unwrap();
    assert_eq!(failed[0], 6);
    for unit_type in [Some(0u16), None] {
        let ev = GameEvent::UnitRejected { cmd_index: 3, unit: 77, unit_type, command: 14, error: 10 };
        let bytes = postcard::to_allocvec(&ev).unwrap();
        assert_eq!(bytes[0], 7);
        assert_eq!(postcard::from_bytes::<GameEvent>(&bytes).unwrap(), ev);
    }
}

/// 010-AC3: 200 units ≤ 64 KB, encode+decode ≤ 1 ms.
#[test]
fn ac3_200_unit_frame_size_and_time() {
    let unit = |i: u32| UnitState {
        id: 1000 + i,
        owner: (i % 3) as u8,
        unit_type: (i % 200) as u16,
        pos: ((i * 37 % 4096) as u16, (i * 91 % 4096) as u16),
        vel: (120, -40),
        hp: 40,
        shield: 20,
        energy: 50,
        order: 10,
        order_target: Some(1),
        order_pos: Some((100, 200)),
        flags: flags::COMPLETED | flags::MOVING,
        build_queue: [228; 5],
        status_timers: StatusTimers::default(),
        ..Default::default()
    };
    let frame = RawFrame {
        frame: 12345,
        self_state: PlayerState { minerals: 500, gas: 200, supply_used: 100, supply_total: 140, upgrades: vec![0; 63], ..Default::default() },
        units: (0..200).map(unit).collect(),
        visibility: BitGrid::new(128, 128),
        explored: BitGrid::new(128, 128),
        creep: BitGrid::new(128, 128),
        events: vec![GameEvent::UnitCreated(5); 10],
        effects: vec![],
    };
    let msg = BridgeMsg::Frame(Box::new(frame));
    let mut buf = Vec::new();
    let size = write_msg(&mut buf, &msg).unwrap();
    assert!(size <= 64 * 1024, "frame is {size} bytes");

    let iters = 200;
    let t = Instant::now();
    for _ in 0..iters {
        let mut b = Vec::with_capacity(size + 4);
        write_msg(&mut b, &msg).unwrap();
        let back: BridgeMsg = read_msg(&mut b.as_slice()).unwrap();
        std::hint::black_box(back);
    }
    let per_ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
    eprintln!("200-unit frame: {size} bytes, encode+decode {per_ms:.3} ms");
    // Debug builds are much slower; the 1 ms budget applies to release (`cargo test --release`).
    if !cfg!(debug_assertions) {
        assert!(per_ms <= 1.0, "encode+decode took {per_ms:.3} ms");
    }
}

#[test]
fn oversized_length_prefix_is_rejected() {
    let bytes = (MAX_MESSAGE_BYTES as u32 + 1).to_le_bytes();
    let r: Result<AgentMsg, _> = read_msg(&mut bytes.as_slice());
    assert!(matches!(r, Err(ProtoError::TooLarge(_))));
}
