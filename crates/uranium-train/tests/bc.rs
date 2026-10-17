//! 041 skeleton: shard -> features -> labels -> BC updates on a synthetic game (Flex).

use std::collections::VecDeque;

use burn::tensor::Device;
use uranium_env::features::augment::Sym;
use uranium_env::features::extract::action_table;
use uranium_env::actions::LOC_CLASSES;
use uranium_env::features::CELLS;
use uranium_env::types::TypeTables;
use uranium_model::batch::ActionColumns;
use uranium_model::collate::input_dims;
use uranium_model::config::*;
use uranium_model::{LstmCore, UraniumNet};
use uranium_proto::*;
use uranium_replay::label::Label;
use uranium_replay::shard::{GameMeta, ShardGame, Step};
use uranium_train::bc::{BcConfig, DataConfig, HeadWeights, Trainer, ValueWeights};
use uranium_train::data::{VecSource, prepare};

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

fn game() -> ShardGame {
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
    let units = vec![unit(1, 0, "Terran_Command_Center", (4, 4)), unit(2, 0, "Terran_SCV", (9, 5)), unit(3, 11, "Resource_Mineral_Field", (14, 4))];
    let full = |v: bool| BitGrid::from_bytes(W, W, &vec![v as u8; n]);
    let table = action_table();
    let id = |name: &str| table.by_name(name).unwrap().id;
    let label = |frame: u32, k: usize| -> Label {
        let (action, selected, target_unit, target_tile) = match k % 4 {
            0 => (id("Train_SCV"), vec![], None, None),
            1 => (id("Move_Pos"), vec![2], None, Some((12, 12))),
            2 => (id("Gather_Unit"), vec![2], Some(3), None),
            _ => (id("NoOp"), vec![], None, None),
        };
        Label { frame, action, delay: 8, queued: false, selected, target_unit, target_tile }
    };
    let steps = (0..16)
        .map(|k| Step {
            raw_frame: RawFrame {
                frame: k as u32 * 8,
                self_state: PlayerState { minerals: 500, gas: 0, supply_used: 2, supply_total: 20, upgrades: vec![0; 63], ..Default::default() },
                units: units.clone(),
                visibility: full(true),
                explored: full(true),
                creep: full(false),
                events: vec![],
                effects: vec![],
            },
            privileged: Some(PrivilegedSummary::default()),
            action_label: label(k as u32 * 8, k),
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
        src: "test".into(),
        stage: "S1".into(),
        replay_sha256: "00".into(),
    };
    ShardGame { meta, game_info: info, steps }
}

#[test]
fn labels_convert_and_survive_symmetry_041() {
    let g = game();
    for sym in Sym::all() {
        let seq = prepare(&g, true, sym).unwrap();
        assert_eq!(seq.invalid_labels, 0, "{sym:?}");
        for s in &seq.steps {
            let a = &action_table().actions[s.label.action as usize];
            if let Some(t) = s.label.target_loc {
                let l = LOC_CLASSES.iter().position(|c| *c == a.loc_class).unwrap();
                assert!(s.obs.masks.loc_class_mask[l * CELLS + t], "{sym:?}: transformed target outside its mask");
            }
        }
    }
    // z (041-R8): the replay's own build actions.
    let seq = prepare(&g, true, Sym::IDENTITY).unwrap();
    assert!(seq.steps[0].obs.obs.z_present && seq.steps[0].obs.obs.z_build[0] != 0);
    assert!(!prepare(&g, false, Sym::IDENTITY).unwrap().steps[0].obs.obs.z_present);
}

fn tiny_cfg() -> UraniumNetConfig {
    UraniumNetConfig::new()
        .with_input(input_dims())
        .with_entity(EntityConfig::new().with_d_model(32).with_layers(1).with_heads(4).with_ffn(64).with_dropout(0.0).with_unit_type_emb(8).with_order_emb(8).with_last_action_emb(4))
        .with_spatial(SpatialConfig::new().with_scatter_channels(4).with_stem_channels(8).with_stages([8, 8, 16, 16]).with_pool(32))
        .with_scalar(ScalarConfig::new().with_hidden(32).with_race_emb(4).with_map_family_emb(4).with_start_loc_emb(4).with_z_build_emb(8).with_z_units_emb(8).with_z_out(8))
        .with_core(CoreConfig::new().with_hidden(32))
        .with_heads(HeadsConfig::new().with_ar(32).with_key(16).with_loc_ar_channels(4))
        .with_value(ValueConfig::new().with_hidden(32).with_privileged(8))
}

/// Default run: a few updates (finite loss, usable labels, checkpoint). The full overfit check is
/// slow in debug: cargo test --release -p uranium-train -- --ignored --nocapture
#[test]
fn bc_smoke_041() {
    train_tiny(3, false);
}

#[test]
#[ignore]
fn bc_overfits_a_tiny_game_041() {
    train_tiny(40, true);
}

/// Lanes trained one at a time with summed gradients give the same updates as one pass over all
/// lanes (041-R3: the batch is the hyper-parameter, `micro_lanes` only bounds GPU memory).
#[test]
fn micro_lanes_match_full_batch_041_r3() {
    // One initial net for both runs: parallel tests share the device RNG, so seeding is not enough.
    let net = tiny_net();
    let run = |micro_lanes, pad_entities| {
        let mut t = tiny_trainer_with(micro_lanes, pad_entities, net.clone());
        let mut src = VecSource((0..4).map(|_| game()).collect::<VecDeque<_>>());
        (0..4).map(|_| t.train_step(&mut src).unwrap().unwrap().total).collect::<Vec<_>>()
    };
    let full = run(0, 0);
    // Extra masked entity padding (pad_entities) must not change the result either.
    for (micro_lanes, pad) in [(1, 0), (1, 64)] {
        let other = run(micro_lanes, pad);
        for (a, b) in full.iter().zip(&other) {
            assert!((a - b).abs() <= 1e-4 * a.abs().max(1.0), "full batch {full:?} != micro_lanes {micro_lanes} pad {pad}: {other:?}");
        }
    }
}

fn tiny_net() -> UraniumNet<LstmCore> {
    let ad = Device::flex().autodiff();
    ad.seed(3);
    let cols = ActionColumns::from_table(action_table()).unwrap();
    UraniumNet::<LstmCore>::new(&tiny_cfg(), &cols, &ad)
}

fn tiny_trainer(micro_lanes: usize) -> Trainer<LstmCore> {
    tiny_trainer_with(micro_lanes, 0, tiny_net())
}

fn tiny_trainer_with(micro_lanes: usize, pad_entities: usize, net: UraniumNet<LstmCore>) -> Trainer<LstmCore> {
    let cfg = BcConfig {
        lanes: 2,
        seq_len: 4,
        micro_lanes,
        grad_checkpointing: false,
        pad_entities,
        lr: 3e-3,
        min_lr: 3e-3,
        warmup_steps: 1,
        total_steps: 100,
        weight_decay: 0.0,
        grad_clip: 1.0,
        s2_lr_scale: 0.1,
        heads: HeadWeights { action: 1.0, delay: 0.5, queued: 0.2, selected: 1.0, target_unit: 1.0, target_loc: 1.0 },
        value: ValueWeights { win: 0.5 },
        data: DataConfig { z_drop: 0.0, augment: false, seed: 1, checkpoint_every: 1000 },
    };
    Trainer::new(net, tiny_cfg(), cfg, Device::flex().autodiff())
}

fn train_tiny(updates: usize, expect_drop: bool) {
    let mut t = tiny_trainer(1);
    let cols = ActionColumns::from_table(action_table()).unwrap();
    let mut src = VecSource((0..updates).map(|_| game()).collect::<VecDeque<_>>());
    let mut losses = vec![];
    for _ in 0..updates {
        let Some(l) = t.train_step(&mut src).unwrap() else { break };
        assert!(l.total.is_finite());
        assert_eq!(l.valid, 8.0, "every label usable");
        losses.push(l.total);
    }
    assert_eq!(losses.len(), updates);
    if expect_drop {
        let first: f32 = losses[..4].iter().sum::<f32>() / 4.0;
        let last: f32 = losses[losses.len() - 4..].iter().sum::<f32>() / 4.0;
        eprintln!("loss {first:.3} -> {last:.3} over {} updates", losses.len());
        assert!(last < first * 0.5, "loss did not drop: {first} -> {last}");
    }

    let dir = std::env::temp_dir().join(format!("uranium-bc-{}", std::process::id()));
    t.save(&dir).unwrap();
    let meta = uranium_model::checkpoint::read_meta(&dir).unwrap();
    assert_eq!(meta.train["bc"]["seq_len"], 4, "hyper-parameters recorded (041-R10)");
    let (_net, _) = uranium_model::checkpoint::load::<LstmCore>(&dir, &cols, uranium_env::features::FEATURE_SCHEMA_VERSION, uranium_env::actions::ACTION_SCHEMA_VERSION, &Device::flex()).unwrap();
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn warmup_cosine_schedule_041_r3() {
    let cfg = BcConfig {
        lanes: 1,
        seq_len: 1,
        micro_lanes: 0,
        grad_checkpointing: false,
        pad_entities: 0,
        lr: 1e-3,
        min_lr: 1e-5,
        warmup_steps: 10,
        total_steps: 110,
        weight_decay: 0.0,
        grad_clip: 1.0,
        s2_lr_scale: 0.1,
        heads: HeadWeights { action: 1.0, delay: 0.5, queued: 0.2, selected: 1.0, target_unit: 1.0, target_loc: 1.0 },
        value: ValueWeights { win: 0.5 },
        data: DataConfig { z_drop: 0.25, augment: true, seed: 1, checkpoint_every: 1 },
    };
    assert!((cfg.lr_at(0, 1.0) - 1e-4).abs() < 1e-12);
    assert!((cfg.lr_at(9, 1.0) - 1e-3).abs() < 1e-12);
    assert!((cfg.lr_at(60, 1.0) - (1e-5 + 0.5 * (1e-3 - 1e-5))).abs() < 1e-9);
    assert!((cfg.lr_at(110, 1.0) - 1e-5).abs() < 1e-12);
    assert!((cfg.lr_at(9, 0.1) - 1e-4).abs() < 1e-12, "S2 = 1/10 (041-R4)");
    let file = BcConfig::load(std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../config/bc.toml"))).unwrap();
    assert_eq!((file.seq_len, file.grad_clip, file.value.win, file.data.z_drop), (64, 1.0, 0.5, 0.25));
}

/// 061-R10 / 061-AC1 / 042-R1 (flex): save -> load -> the next updates are bit-identical to the
/// run that kept going (weights, optimizer moments, step-driven lr schedule and RNG all restored).
/// Both runs start the next updates with fresh lanes, as a resumed trainer does.
#[test]
fn resume_continues_bit_identically_061_r10() {
    let cols = ActionColumns::from_table(action_table()).unwrap();
    let dir = std::env::temp_dir().join(format!("uranium-bc-resume-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    let games = |n: usize| VecSource((0..n).map(|_| game()).collect::<VecDeque<_>>());
    let next = |t: &mut Trainer<LstmCore>| {
        let mut src = games(4);
        (0..2).map(|_| t.train_step(&mut src).unwrap().unwrap().total.to_bits()).collect::<Vec<_>>()
    };

    let mut a = tiny_trainer(1);
    let mut src = games(4);
    for _ in 0..3 {
        a.train_step(&mut src).unwrap().unwrap();
    }
    a.save(&dir).unwrap();
    a.reset_lanes();
    let kept = next(&mut a);

    let ad = Device::flex().autodiff();
    let load = || uranium_model::checkpoint::load::<LstmCore>(&dir, &cols, uranium_env::features::FEATURE_SCHEMA_VERSION, uranium_env::actions::ACTION_SCHEMA_VERSION, &ad).unwrap().0;
    let mut b = tiny_trainer_with(1, 0, load());
    assert!(b.resume(&dir).unwrap(), "training state present");
    assert_eq!(b.step, 3);
    assert_eq!(next(&mut b), kept, "resumed updates differ from the run that kept going");

    // Control: the weights alone (fresh optimizer, step 0) do not reproduce the run.
    let mut c = tiny_trainer_with(1, 0, load());
    assert_ne!(next(&mut c), kept);
    std::fs::remove_dir_all(&dir).ok();
}
