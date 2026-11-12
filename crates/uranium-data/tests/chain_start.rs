//! 041-R30 / 041-AC7: chained steps of listed sources (teacher `botgen`) are observed through
//! their chain-start step's RawFrame (events emptied, the chained frames' events carried to the
//! next step that is not chained); labels, delays, memory order and step counts stay the same.

use uranium_data::bc::{BcConfig, Feeder};
use uranium_data::data::{GameStream, MapStore, TrainStep, VecSource, chained_step, check_chain_start_sources, prepare};
use uranium_env::features::augment::Sym;
use uranium_env::features::extract::action_table;
use uranium_env::types::TypeTables;
use uranium_proto::*;
use uranium_replay::label::Label;
use uranium_replay::shard::{GameMeta, ShardGame, Step};

const W: u16 = 32;

fn unit(id: u32, owner: u8, name: &str, tile: (i32, i32)) -> UnitState {
    let t = TypeTables::get().unit_by_name(name).unwrap();
    let pos = if t.is_building() || t.is_resource() {
        ((tile.0 * 32 + t.tile_width * 16) as u16, (tile.1 * 32 + t.tile_height * 16) as u16)
    } else {
        ((tile.0 * 32 + 16) as u16, (tile.1 * 32 + 16) as u16)
    };
    let none = TypeTables::get().unit_by_name("None").unwrap().id as u16;
    UnitState {
        id,
        owner,
        unit_type: t.id as u16,
        pos,
        hp: t.max_hp as u16,
        flags: flags::COMPLETED,
        build_queue: [none; 5],
        resources: if t.is_resource() { 1500 } else { 0 },
        order: 189,
        ..Default::default()
    }
}

/// Step frames: chains at 1→2→3 (gap 1), 5→6 (5 is not chained: gap 2 after 4), 8→9; step 12 follows
/// a NoOp label at the next frame (not chained).
const FRAMES: [u32; 16] = [0, 8, 9, 10, 18, 20, 21, 29, 37, 38, 46, 54, 55, 63, 71, 79];
const CHAINED: [usize; 4] = [2, 3, 6, 9];
/// The enemy hatchery seen at step 0 and destroyed (event) in chained step 3's own frame.
const HATCH: u32 = 50;

/// A game of `src` with [`FRAMES`]; labels cycle Train_SCV, Move_Pos, Gather_Unit, NoOp (step 3 is
/// a Train_SCV with no minerals in its own frame); minerals and the privileged enemy supply differ per
/// step; the enemy hatchery is visible at step 0 only, then the map is dark (memory keeps it until
/// its `UnitDestroyed`); step 1 (a chain start) has a `CommandFailed` event.
fn chain_game(src: &str) -> ShardGame {
    let n = W as usize * W as usize;
    let info = GameInfo {
        map_name: "tiny".into(),
        map_hash: [0; 32],
        map_size_tiles: (W, W),
        self_player: 0,
        players: vec![
            PlayerInfo { id: 0, name: "t".into(), race: Race::Terran, start_tile: Some((4, 4)), is_self: true, is_enemy: false },
            PlayerInfo { id: 1, name: "z".into(), race: Race::Zerg, start_tile: Some((26, 26)), is_self: false, is_enemy: true },
        ],
        static_map: StaticMap {
            width_tiles: W,
            height_tiles: W,
            walkable: BitGrid::from_bytes(W * 4, W * 4, &vec![1; n * 16]),
            buildable: BitGrid::from_bytes(W, W, &vec![1; n]),
            ground_height: vec![0; n],
            start_locations: vec![(4, 4), (26, 26)],
        },
        latency_frames: 0,
    };
    let own = vec![unit(1, 0, "Terran_Command_Center", (4, 4)), unit(2, 0, "Terran_SCV", (9, 5)), unit(3, 11, "Resource_Mineral_Field", (14, 4))];
    let full = |v: bool| BitGrid::from_bytes(W, W, &vec![v as u8; n]);
    let id = |name: &str| action_table().by_name(name).unwrap().id;
    let steps = FRAMES
        .iter()
        .enumerate()
        .map(|(k, &frame)| {
            let (action, selected, target_unit, target_tile) = match k % 4 {
                _ if k == 3 => (id("Train_SCV"), vec![1], None, None),
                0 => (id("Train_SCV"), vec![1], None, None),
                1 => (id("Move_Pos"), vec![2], None, Some((12, 12))),
                2 => (id("Gather_Unit"), vec![2], Some(3), None),
                _ => (id("NoOp"), vec![], None, None),
            };
            let mut units = own.clone();
            if k == 0 {
                units.push(unit(HATCH, 1, "Zerg_Hatchery", (26, 26)));
            }
            let mut events = vec![];
            if k == 1 {
                events.push(GameEvent::CommandFailed { cmd_index: 0, error: 1 });
            }
            if k == 3 {
                events.push(GameEvent::UnitDestroyed(HATCH));
            }
            let minerals = if k == 3 { 0 } else { 400 + k as i32 * 10 };
            Step {
                raw_frame: RawFrame {
                    frame,
                    self_state: PlayerState { minerals, gas: 0, supply_used: 2, supply_total: 20, upgrades: vec![0; 63], ..Default::default() },
                    units,
                    visibility: full(k == 0),
                    explored: full(true),
                    creep: full(false),
                    events,
                    effects: vec![],
                },
                privileged: Some(PrivilegedSummary { frame, enemy: PlayerState { supply_used: 4 + k as u16, ..Default::default() }, ..Default::default() }),
                action_label: Label { frame, action, delay: 1, queued: false, selected, target_unit, target_tile },
                memory_action: None,
            }
        })
        .collect();
    let meta = GameMeta {
        map_hash: "none".into(),
        map_family: "other".into(),
        matchup: "TvZ".into(),
        player_race: 1,
        perspective_slot: 0,
        apm: 100,
        result: 1,
        result_source: "bridge".into(),
        enemy_random: false,
        src: src.into(),
        stage: "S2".into(),
        replay_sha256: "ab".into(),
    };
    ShardGame { meta, game_info: info, steps }
}

/// 041-R30 ③ written out on the shard game itself: each chained step's RawFrame (events emptied)
/// and privileged replaced by its chain-start step's, its own events carried to the next step that
/// is not chained. The plain loader on this game must give what the rule gives on the original.
fn rewritten(g: &ShardGame) -> ShardGame {
    let mut out = g.clone();
    let mut start = 0;
    let mut pending: Vec<GameEvent> = vec![];
    for i in 0..g.steps.len() {
        if chained_step(&g.steps, i) {
            pending.extend(g.steps[i].raw_frame.events.iter().cloned());
            out.steps[i].raw_frame = RawFrame { events: vec![], ..g.steps[start].raw_frame.clone() };
            out.steps[i].privileged = g.steps[start].privileged.clone();
        } else {
            start = i;
            let mut ev = std::mem::take(&mut pending);
            ev.extend(g.steps[i].raw_frame.events.iter().cloned());
            out.steps[i].raw_frame.events = ev;
        }
    }
    out
}

type Key = (Observation, HostLabelKey, Vec<f32>, bool, f32, u32, bool);
type HostLabelKey = uranium_env::features::label::HostLabel;
use uranium_env::features::extract::Observation;

fn key(s: &[TrainStep]) -> Vec<Key> {
    s.iter().map(|t| (t.obs.clone(), t.label.clone(), t.privileged.summary.clone(), t.has_privileged, t.weight, t.frame, t.surrender_rule)).collect()
}

fn stream(g: &ShardGame, z: bool, sym: Sym, sources: &[String]) -> Vec<TrainStep> {
    let mut s = GameStream::new(g.clone(), z, sym, &MapStore::synthetic()).unwrap().with_chain_start(sources);
    s.next_chunk(usize::MAX).0
}

fn botgen() -> Vec<String> {
    vec!["botgen".to_string()]
}

#[test]
fn chained_steps_follow_the_integer_frame_rule_041_r30() {
    let g = chain_game("botgen");
    let chained: Vec<usize> = (0..g.steps.len()).filter(|&i| chained_step(&g.steps, i)).collect();
    // Gap 2 (step 5) and a NoOp before (step 12) are not chained; step 0 never is.
    assert_eq!(chained, CHAINED);
}

/// 041-AC7: with `botgen` listed, a botgen game equals the plain loader on the rewritten game (obs,
/// label, privileged, weight, frame, surrender rule), for symmetry and z too; labels, delays and the
/// step count are the game's own. Other sources and an unlisted source read as without the key.
#[test]
fn chain_start_observation_041_r30() {
    let g = chain_game("botgen");
    let maps = MapStore::synthetic();
    for sym in [Sym::IDENTITY, Sym::all()[5]] {
        for z in [false, true] {
            let on = stream(&g, z, sym, &botgen());
            let off = stream(&g, z, sym, &[]);
            let want = prepare(&rewritten(&g), z, sym, &maps).unwrap().steps;
            assert_eq!(key(&on), key(&want), "{sym:?} z={z}");
            assert_eq!(on.len(), g.steps.len());
            assert_eq!(off.len(), g.steps.len());
            let plain = prepare(&g, z, sym, &maps).unwrap().steps;
            assert_eq!(key(&off), key(&plain), "no sources = the plain loader");
            for i in 0..g.steps.len() {
                assert_eq!((on[i].label.action, on[i].label.delay_bucket), (off[i].label.action, off[i].label.delay_bucket), "step {i}: label and delay unchanged");
            }
            for &i in &CHAINED {
                assert_ne!(on[i].obs, off[i].obs, "step {i} sees its chain start");
            }
        }
    }
    // Chain-start frame, privileged and frame of the chained steps.
    let on = stream(&g, false, Sym::IDENTITY, &botgen());
    for (i, c) in [(2, 1), (3, 1), (6, 5), (9, 8)] {
        assert_eq!(on[i].frame, FRAMES[c], "step {i}");
        assert_eq!(on[i].privileged.summary, on[c].privileged.summary, "step {i}: privileged of the chain start");
    }
    // Unlisted sources: human data, and dagger when only botgen is listed.
    for src in ["stardata", "dagger"] {
        let h = chain_game(src);
        assert_eq!(key(&stream(&h, false, Sym::IDENTITY, &botgen())), key(&stream(&h, false, Sym::IDENTITY, &[])), "{src}");
    }
    let d = chain_game("dagger");
    assert_eq!(key(&stream(&d, false, Sym::IDENTITY, &["dagger".to_string()])), key(&prepare(&rewritten(&d), false, Sym::IDENTITY, &maps).unwrap().steps), "dagger when listed");
}

/// 041-R30 ③: the chain-start frame's events are not applied again (step 2 does not see step 1's
/// `CommandFailed`); a chained step's own events wait for the next step that is not chained
/// (the hatchery destroyed in step 3's frame is still remembered at step 3 and gone at step 4).
/// The label check follows the observation: step 3's Train_SCV is void at its own frame (no
/// minerals) and valid at the chain start.
#[test]
fn chain_start_events_and_label_check_041_r30() {
    let g = chain_game("botgen");
    let on = stream(&g, false, Sym::IDENTITY, &botgen());
    let off = stream(&g, false, Sym::IDENTITY, &[]);
    let has_hatch = |s: &TrainStep| s.obs.entity_ids.contains(&HATCH);
    assert!(has_hatch(&off[2]) && !has_hatch(&off[3]), "without the rule the event lands at step 3");
    assert!(has_hatch(&on[3]) && !has_hatch(&on[4]), "with it at step 4, the next step that is not chained");
    // Step 2 observed with step 1's events kept would differ (last_failed): the events are emptied.
    let maps = MapStore::synthetic();
    let mut kept = rewritten(&g);
    kept.steps[2].raw_frame.events = g.steps[1].raw_frame.events.clone();
    let kept = prepare(&kept, false, Sym::IDENTITY, &maps).unwrap().steps;
    assert_ne!(kept[2].obs, on[2].obs, "a re-applied CommandFailed would show");
    assert_eq!(off[3].weight, 0.0, "Train_SCV without minerals is void at its own frame");
    assert_eq!(on[3].weight, 1.0, "and valid at the chain start");
}

/// Chunks and resume (061-R10): the rule's state crosses chunk boundaries, and a game advanced to
/// the middle of a chain continues as an uninterrupted one.
#[test]
fn chain_start_across_chunks_and_resume_041_r30() {
    let g = chain_game("botgen");
    let whole = stream(&g, true, Sym::all()[3], &botgen());
    for n in [1, 2, 3, 5] {
        let mut s = GameStream::new(g.clone(), true, Sym::all()[3], &MapStore::synthetic()).unwrap().with_chain_start(&botgen());
        let mut got = vec![];
        while s.remaining() > 0 {
            got.extend(s.next_chunk(n).0);
        }
        assert_eq!(key(&got), key(&whole), "chunks of {n}");
    }
    for at in [2, 3, 4, 7] {
        let mut s = GameStream::new(g.clone(), true, Sym::all()[3], &MapStore::synthetic()).unwrap().with_chain_start(&botgen());
        s.advance_to(at);
        assert_eq!(key(&s.next_chunk(usize::MAX).0), key(&whole[at..]), "resume at {at}");
    }
}

/// The feeder takes the sources from `[data] chain_start_sources`.
#[test]
fn feeder_uses_the_config_key_041_r30() {
    let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../config/bc.toml")).unwrap();
    let load = |extra: &str| -> anyhow::Result<BcConfig> {
        let path = std::env::temp_dir().join(format!("uranium-041-r30-{}-{}.toml", std::process::id(), extra.len()));
        let body = text.replacen("[data]\n", &format!("[data]\n{extra}\n"), 1).replacen("[data]\r\n", &format!("[data]\r\n{extra}\r\n"), 1);
        std::fs::write(&path, body).unwrap();
        let c = BcConfig::load(&path);
        std::fs::remove_file(&path).ok();
        c
    };
    let base = load("").unwrap();
    assert!(base.data.chain_start_sources.is_none());
    let on = load("chain_start_sources = [\"botgen\"]").unwrap();
    assert_eq!(on.data.chain_start_sources, Some(botgen()));
    assert_ne!(on.fingerprint(), base.fingerprint(), "041-R30 ②: in the fingerprint when present");
    assert!(load("chain_start_sources = []").is_err());
    assert!(load("chain_start_sources = [\"stardata\"]").is_err());
    // 041-R30 ⑥: the lineage-2 mix config lists botgen only.
    let mix = BcConfig::load(std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../config/bc_m2_mix_e1.toml"))).unwrap();
    assert_eq!(mix.data.chain_start_sources, Some(botgen()));
    assert!(check_chain_start_sources(&["botgen".into(), "dagger".into()]).is_ok());

    let mut cfg = on.clone();
    cfg.lanes = 1;
    cfg.seq_len = 64;
    cfg.data.z_drop = 1.0;
    cfg.data.augment = false;
    cfg.data.enemy_race_mask = 0.0;
    let mut f = Feeder::new(&cfg).with_maps(std::sync::Arc::new(MapStore::synthetic()));
    let b = f.next_batch(&mut VecSource(vec![chain_game("botgen")].into()), None).unwrap().unwrap();
    assert_eq!(key(&b.parts().0[0]), key(&stream(&chain_game("botgen"), false, Sym::IDENTITY, &botgen())));
}
