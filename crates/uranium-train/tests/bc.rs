//! 041 skeleton: shard -> features -> labels -> BC updates on a synthetic game (Flex).

use std::collections::VecDeque;

use burn::module::Module;
use burn::tensor::Device;
use uranium_env::features::augment::Sym;
use uranium_env::features::extract::action_table;
use uranium_env::actions::LOC_CLASSES;
use uranium_env::features::CELLS;
use uranium_env::types::TypeTables;
use uranium_model::batch::ActionColumns;
use uranium_model::collate::{input_dims, obs_batch_padded};
use uranium_model::config::*;
use uranium_model::{LstmCore, UraniumNet};
use uranium_proto::*;
use uranium_replay::label::Label;
use uranium_replay::shard::{GameMeta, ShardGame, Step};
use uranium_train::bc::{Batch, BcConfig, DataConfig, Feeder, HeadWeights, Prefetch, Trainer, ValueWeights};
use uranium_train::data::{MapStore, ShardSource, VecSource, epoch_updates, prepare};

/// A feeder for the synthetic test maps (no map precompute files, B-2).
fn feeder(cfg: &BcConfig) -> Feeder {
    Feeder::new(cfg).with_maps(std::sync::Arc::new(MapStore::synthetic()))
}

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
        result_source: "bridge".into(),
        enemy_random: false,
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
        let seq = prepare(&g, true, sym, &MapStore::synthetic()).unwrap();
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
    let seq = prepare(&g, true, Sym::IDENTITY, &MapStore::synthetic()).unwrap();
    assert!(seq.steps[0].obs.obs.z_present && seq.steps[0].obs.obs.z_build[0] != 0);
    assert!(!prepare(&g, false, Sym::IDENTITY, &MapStore::synthetic()).unwrap().steps[0].obs.obs.z_present);
}

/// 030-R11 (user decision 2026-10-19): about 5% of games, chosen per game from the replay hash and
/// the data seed, see the enemy race as unknown until an enemy unit is seen; shard
/// `enemy_random` does the same; other games keep the lobby race.
#[test]
fn enemy_race_mask_per_game_030_r11() {
    use uranium_env::features::RACE_UNKNOWN;
    use uranium_train::data::{GameStream, enemy_race_masked};
    let shas: Vec<String> = (0..20_000u32).map(|i| format!("{:064x}", u64::from(i).wrapping_mul(0x9E37_79B9_7F4A_7C15))).collect();
    let share = shas.iter().filter(|s| enemy_race_masked(s, 1, 0.05)).count() as f64 / shas.len() as f64;
    assert!((0.04..0.06).contains(&share), "masked share {share}");
    assert!(shas.iter().all(|s| enemy_race_masked(s, 1, 0.05) == enemy_race_masked(s, 1, 0.05)), "deterministic");
    assert!(shas.iter().any(|s| enemy_race_masked(s, 1, 0.05) != enemy_race_masked(s, 2, 0.05)), "seed matters");
    assert!(!enemy_race_masked(&shas[0], 1, 0.0) && enemy_race_masked(&shas[0], 1, 1.0));

    let race = |g: &ShardGame, ratio: f64| {
        let mut s = GameStream::new(g.clone(), false, Sym::IDENTITY, &MapStore::synthetic()).unwrap().with_race_mask(ratio, 1);
        s.next_chunk(1).0[0].obs.obs.scalar_cat[0]
    };
    let g = game();
    assert_eq!(race(&g, 0.0), 2, "Zerg from the lobby");
    assert_eq!(race(&g, 1.0), RACE_UNKNOWN, "masked game");
    let mut random = game();
    random.meta.enemy_random = true;
    assert_eq!(race(&random, 0.0), RACE_UNKNOWN, "lobby Random");
}

/// 034-R7 / E-4: a label selecting a building together with other units breaks the game's
/// selection rule and is void (`select_building`); the building alone is fine.
#[test]
fn building_with_units_label_is_void_034_r7() {
    use uranium_env::features::{AgentMemory, MapInfo, extract};
    use uranium_train::data::host_label_check;
    let g = game();
    let map = MapInfo::new(&g.game_info, 2, None).unwrap();
    let mut f = g.steps[0].raw_frame.clone();
    let mut rax = unit(10, 0, "Terran_Barracks", (20, 20));
    rax.flags |= flags::LIFTED;
    f.units.push(rax);
    let o = extract(&f, &mut AgentMemory::new(&g.game_info), &map, None);
    let mv = action_table().by_name("Move_Pos").unwrap().id;
    let label = |selected: Vec<u32>| Label { frame: 0, action: mv, delay: 8, queued: false, selected, target_unit: None, target_tile: Some((12, 12)) };
    assert_eq!(host_label_check(&label(vec![2, 10]), &o).1, Some("select_building"));
    assert_eq!(host_label_check(&label(vec![10]), &o).1, None);
    assert_eq!(host_label_check(&label(vec![2]), &o).1, None);
}

/// The synthetic game stretched to `steps` steps with the first `invalid` labels void
/// (`select_empty`: the selected unit 99 does not exist).
fn game_with_invalid(steps: usize, invalid: usize) -> ShardGame {
    let mut g = game();
    let base = g.steps.clone();
    g.steps = (0..steps).map(|k| base[k % base.len()].clone()).collect();
    let mv = action_table().by_name("Move_Pos").unwrap().id;
    for s in &mut g.steps[..invalid] {
        s.action_label = Label { frame: s.action_label.frame, action: mv, delay: 8, queued: false, selected: vec![99], target_unit: None, target_tile: Some((12, 12)) };
    }
    g
}

/// 040-R19 / 040-AC7 (user decision 2026-10-19): the build rejects a perspective whose invalid labels are
/// at least 20% of its labeled steps, and keeps one below that.
#[test]
fn invalid_label_share_filters_perspective_040_r19() {
    use uranium_env::features::label::INVALID_LABEL_MAX;
    use uranium_replay::pipeline::invalid_label_filter;
    assert_eq!(INVALID_LABEL_MAX, 0.20);
    assert_eq!(invalid_label_filter(&game_with_invalid(20, 0), None).unwrap(), None);
    assert_eq!(invalid_label_filter(&game_with_invalid(20, 3), None).unwrap(), None, "15% is kept");
    // Exactly 20% is rejected (>=).
    let r = invalid_label_filter(&game_with_invalid(20, 4), None).unwrap().expect("20% is rejected");
    assert_eq!(r, "invalid_labels>=20% (20.0%, 4/20) (040-R19)");
    assert_eq!(uranium_replay::stats::filter_group(&r), uranium_replay::stats::INVALID_LABEL_GROUP);
    let r = invalid_label_filter(&game_with_invalid(20, 11), None).unwrap().unwrap();
    assert!(r.starts_with("invalid_labels>=20% (55.0%, 11/20)"), "{r}");
    // Deterministic: the same game gives the same result.
    assert_eq!(invalid_label_filter(&game_with_invalid(20, 11), None).unwrap().unwrap(), r);
}

/// 040-R19 / 040-AC7: the build's count is the BC loader's (`prepare` weight-0 labels, every symmetry and
/// z setting) and `label-check`'s, from one shared check (`uranium_env::features::label`).
#[test]
fn build_label_check_matches_training_040_r19() {
    use uranium_env::features::label::label_validity;
    use uranium_train::data::{LabelStats, label_check};
    for (steps, invalid) in [(16, 0), (20, 4), (20, 11), (33, 30)] {
        let g = game_with_invalid(steps, invalid);
        let v = label_validity(&g.game_info, 2, None, false, g.steps.iter().map(|s| (&s.raw_frame, &s.action_label))).unwrap();
        assert_eq!((v.labeled, v.invalid), (steps, invalid));
        for sym in Sym::all() {
            for z in [false, true] {
                let seq = prepare(&g, z, sym, &MapStore::synthetic()).unwrap();
                assert_eq!(seq.invalid_labels, v.invalid, "{sym:?} z={z}");
                assert_eq!(seq.steps.iter().filter(|s| s.weight == 0.0).count(), v.invalid);
            }
        }
        let mut st = LabelStats::default();
        label_check(&g, &mut st, &MapStore::synthetic()).unwrap();
        let mut by_reason = std::collections::BTreeMap::new();
        for ((_, why), n) in &st.fails {
            *by_reason.entry(*why).or_insert(0) += n;
        }
        assert_eq!(by_reason, v.by_reason);
        assert_eq!(st.totals.values().sum::<usize>(), v.labeled);
    }
}

/// Labeler versions 1 and 2 in one dataset (2026-10-19, 040 "Shard 포맷"): a version-2 game whose
/// NoOp step is an unknown command instead (031-AC3, F7) trains the same as the version-1 game
/// everywhere else: identical observations (agent memory records the unknown step as the NoOp
/// version 1 had), labels and weights; the unknown step is weight 0 but not an invalid label
/// (040-R19 / 040-AC7 count the same), and the build filter does not count it.
#[test]
fn labeler_v1_v2_train_the_same_except_unknown_steps_040() {
    use uranium_env::features::label::{UNKNOWN_ACTION, label_validity};
    let v1 = game();
    let mut v2 = game();
    let k = v2.steps.iter().position(|s| s.action_label.action == 0).unwrap();
    v2.steps[k].action_label.action = UNKNOWN_ACTION;
    for sym in [Sym::IDENTITY, Sym::all()[1]] {
        let a = prepare(&v1, true, sym, &MapStore::synthetic()).unwrap();
        let b = prepare(&v2, true, sym, &MapStore::synthetic()).unwrap();
        assert_eq!(a.invalid_labels, b.invalid_labels, "{sym:?}");
        for (i, (x, y)) in a.steps.iter().zip(&b.steps).enumerate() {
            assert_eq!(x.obs, y.obs, "{sym:?} step {i}: observation");
            assert_eq!(x.label, y.label, "{sym:?} step {i}: host label");
            assert_eq!((x.weight, y.weight), if i == k { (1.0, 0.0) } else { (x.weight, x.weight) }, "{sym:?} step {i}: weight");
        }
    }
    let v = label_validity(&v2.game_info, 2, None, false, v2.steps.iter().map(|s| (&s.raw_frame, &s.action_label))).unwrap();
    assert_eq!((v.labeled, v.invalid, v.unknown), (v1.steps.len() - 1, 0, 1));
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
    let mut t = Trainer::new(net, tiny_cfg(), tiny_bc(micro_lanes, pad_entities), Device::flex().autodiff());
    t.feeder.set_maps(std::sync::Arc::new(MapStore::synthetic()));
    t
}

fn tiny_bc(micro_lanes: usize, pad_entities: usize) -> BcConfig {
    BcConfig {
        lanes: 2,
        seq_len: 4,
        micro_lanes,
        grad_checkpointing: false,
        pad_entities,
        lr: 3e-3,
        min_lr: 3e-3,
        warmup_steps: 1,
        total_steps: 100,
        steps: Default::default(),
        weight_decay: 0.0,
        grad_clip: 1.0,
        s2_lr_scale: 0.1,
        heads: HeadWeights { action: 1.0, delay: 0.5, queued: 0.2, selected: 1.0, target_unit: 1.0, target_loc: 1.0 },
        value: ValueWeights { win: 0.5, win_open: None },
        data: DataConfig { z_drop: 0.0, augment: false, seed: 1, checkpoint_every: 1000, checkpoint_minutes: 0, enemy_race_mask: 0.0 },
        keep_checkpoints: None,
    }
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
        steps: Default::default(),
        weight_decay: 0.0,
        grad_clip: 1.0,
        s2_lr_scale: 0.1,
        heads: HeadWeights { action: 1.0, delay: 0.5, queued: 0.2, selected: 1.0, target_unit: 1.0, target_loc: 1.0 },
        value: ValueWeights { win: 0.5, win_open: None },
        data: DataConfig { z_drop: 0.25, augment: true, seed: 1, checkpoint_every: 1, checkpoint_minutes: 0, enemy_race_mask: 0.05 },
        keep_checkpoints: None,
    };
    assert!((cfg.lr_at(0, 1.0) - 1e-4).abs() < 1e-12);
    assert!((cfg.lr_at(9, 1.0) - 1e-3).abs() < 1e-12);
    assert!((cfg.lr_at(60, 1.0) - (1e-5 + 0.5 * (1e-3 - 1e-5))).abs() < 1e-9);
    assert!((cfg.lr_at(110, 1.0) - 1e-5).abs() < 1e-12);
    assert!((cfg.lr_at(9, 0.1) - 1e-4).abs() < 1e-12, "S2 = 1/10 (041-R4)");
    let file = BcConfig::load(std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../config/bc.toml"))).unwrap();
    assert_eq!((file.seq_len, file.grad_clip, file.value.win, file.data.z_drop), (64, 1.0, 0.5, 0.25));
    // 030-R11 (user decision 2026-10-19): 5% of games mask the enemy race.
    assert_eq!(file.data.enemy_race_mask, 0.05);
    // 041-R14: one epoch per stage; (A) reads the S2 games.
    assert!(file.steps.s1 > file.steps.s2 && file.steps.s2 > 0 && file.steps.ablation_a == file.steps.s2);
    // 041-R14 (2026-10-18): S2 and (A) warm up over 5% of their epoch, rounded half up.
    assert_eq!(uranium_train::bc::short_stage_warmup(7915), 396, "395.75 -> 396");
    assert_eq!(uranium_train::bc::short_stage_warmup(10), 1, "0.5 -> 1");
    assert_eq!(uranium_train::bc::short_stage_warmup(9), 0, "0.45 -> 0");
    assert_eq!(file.steps.s2_warmup, uranium_train::bc::short_stage_warmup(file.steps.s2));
    assert_eq!(file.steps.ablation_a_warmup, uranium_train::bc::short_stage_warmup(file.steps.ablation_a));
    assert_eq!(file.warmup_steps, 2000, "S1 keeps its warmup");
    // 032-R17 (AMB-032-6): the core ablation is bc.toml with a short S1 schedule only.
    let abl = BcConfig::load(std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../config/bc_core_ablation.toml"))).unwrap();
    assert_eq!((abl.steps.s1, abl.warmup_steps), (5300, uranium_train::bc::short_stage_warmup(5300)));
    let mut same = abl.clone();
    // 2026-10-23 (I-2): bc.toml's stage lengths were recounted on the frozen main-run data after
    // the ablation; the finished ablation keeps the numbers it ran with.
    same.steps = file.steps.clone();
    same.warmup_steps = file.warmup_steps;
    // 061-R6 / 032-R17 ⑤ (G-D6): the ablation keeps every checkpoint; bc.toml the default 5.
    assert_eq!((abl.keep_checkpoints(), file.keep_checkpoints(), file.keep_checkpoints), (0, 5, None));
    same.keep_checkpoints = file.keep_checkpoints;
    // 041-R19 (2026-10-21): the main BC trains the win_open head; the running ablation does not.
    assert_eq!((abl.value.win_open, file.value.win_open), (None, Some(0.5)));
    same.value.win_open = file.value.win_open;
    assert_eq!(same, file, "032-R17: only the stage lengths, warmup, retention and win_open differ from bc.toml");
    // 041-R19: an absent win_open leaves the fingerprint JSON as before the key existed.
    let view = serde_json::to_value(abl.results_view()).unwrap();
    assert_eq!(view["value"], serde_json::json!({ "win": 0.5 }));
    // B-3: the ablation's fingerprint differs (schedule), the retention alone does not count.
    assert_ne!(abl.fingerprint(), file.fingerprint());
    let mut kept = file.clone();
    kept.keep_checkpoints = Some(0);
    kept.micro_lanes = 4;
    kept.data.checkpoint_minutes = 5;
    assert_eq!(kept.fingerprint(), file.fingerprint(), "B-3: execution-only settings are not part of the fingerprint");
    kept.data.enemy_race_mask = 0.1;
    assert_ne!(kept.fingerprint(), file.fingerprint(), "B-3: data settings are");
    kept = file.clone();
    kept.total_steps += 1;
    assert_ne!(kept.fingerprint(), file.fingerprint(), "B-3: the schedule length is");
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

/// Every float parameter of a module as raw f32 bits, in visit order.
struct ParamBits(Vec<Vec<u32>>);

impl burn::module::ModuleVisitor for ParamBits {
    fn visit_float<const D: usize>(&mut self, p: &burn::module::Param<burn::tensor::Tensor<D>>) {
        self.0.push(host_f32(p.val()).iter().map(|x| x.to_bits()).collect());
    }
}

fn host_f32<const D: usize>(t: burn::tensor::Tensor<D>) -> Vec<f32> {
    t.into_data().convert::<f32>().try_to_vec::<f32>().unwrap()
}

fn param_bits<M: burn::module::Module>(m: &M) -> Vec<Vec<u32>> {
    let mut v = ParamBits(Vec::new());
    m.visit(&mut v);
    v.0
}

/// A burnpack file as sorted (tensor name, raw bytes) plus its scalars: independent of the order
/// the writer iterated its state in (Burn's optimizer keeps it in a hash map).
fn burnpack_digest(path: &std::path::Path) -> (Vec<(String, Vec<u8>)>, String) {
    let r = burn::store::burn_pack::Reader::from_file(path).unwrap();
    let mut t: Vec<(String, Vec<u8>)> = r.tensor_names().into_iter().map(|n| (n.to_string(), r.tensor_data(n).unwrap())).collect();
    t.sort();
    (t, format!("{:?}", r.scalars()))
}

/// 061-AC5 / AC1 (G-D7, flex): a saved checkpoint loads back with weights and optimizer state
/// bit-identical to what was saved (both checked on host data, so independent of the device the
/// checkpoint is loaded on), and the loaded net's forward outputs match the saved net's within
/// the 032-AC1 tolerance (Flex/Vulkan 1e-4; Flex is exact here). On a GPU the stored values are
/// the same; the forward then follows 032-AC1 (CUDA 1e-2), and a resumed run continues
/// bit-identically only on the same host with the same autotune cache (041-R15).
#[test]
fn checkpoint_round_trip_is_bit_identical_061_ac5() {
    let cols = ActionColumns::from_table(action_table()).unwrap();
    let root = std::env::temp_dir().join(format!("uranium-bc-ac5-{}", std::process::id()));
    std::fs::remove_dir_all(&root).ok();
    let (a_dir, b_dir) = (root.join("a"), root.join("b"));
    let mut a = tiny_trainer(1);
    let mut src = VecSource((0..4).map(|_| game()).collect::<VecDeque<_>>());
    for _ in 0..3 {
        a.train_step(&mut src).unwrap().unwrap();
    }
    a.save(&a_dir).unwrap();

    // Weights: loaded on a plain (non-autodiff) Flex device, bit-identical to the trained net.
    let flex = Device::flex();
    let load = |d: &Device| uranium_model::checkpoint::load::<LstmCore>(&a_dir, &cols, uranium_env::features::FEATURE_SCHEMA_VERSION, uranium_env::actions::ACTION_SCHEMA_VERSION, d).unwrap().0;
    let loaded = load(&flex);
    assert_eq!(param_bits(&loaded), param_bits(&a.net.valid()), "061-AC5: loaded weights differ from the saved ones");
    // 041-R15: the Burn runtime setting travels with the weights.
    let meta = uranium_model::checkpoint::read_meta(&a_dir).unwrap();
    assert_eq!(meta.train["burn"], uranium_model::runtime::settings_json());

    // Optimizer state: resume into a fresh trainer and save again: same moments and scalars.
    let mut b = tiny_trainer_with(1, 0, load(&Device::flex().autodiff()));
    assert!(b.resume(&a_dir).unwrap());
    b.save(&b_dir).unwrap();
    let optim = uranium_train::ckpt::optim_file("main");
    let (oa, ob) = (burnpack_digest(&a_dir.join(&optim)), burnpack_digest(&b_dir.join(&optim)));
    assert!(!oa.0.is_empty(), "the optimizer state holds the AdamW moments after 3 updates");
    assert_eq!(oa, ob, "061-AC5: optimizer state after load differs from the saved one");
    let w = uranium_model::checkpoint::WEIGHTS_FILE;
    assert_eq!(burnpack_digest(&a_dir.join(w)), burnpack_digest(&b_dir.join(w)));

    // Forward outputs: same inputs, saved net vs loaded net, within 032-AC1 (Flex 1e-4).
    let (obs, _) = uranium_model::synthetic::random_obs(&tiny_cfg(), &cols, 4, 16, 41, &flex);
    let saved = a.net.valid();
    let labels = saved.act(&obs, saved.init_state(4, &flex), 0.0).action;
    let want = saved.forward_seq(&obs, &labels, None, saved.init_state(4, &flex), 1);
    let got = loaded.forward_seq(&obs, &labels, None, loaded.init_state(4, &flex), 1);
    let max_abs = |x: Vec<f32>, y: Vec<f32>| x.iter().zip(&y).fold(0f32, |m, (p, q)| m.max((p - q).abs()));
    for (name, d) in [
        ("value", max_abs(host_f32(want.value.clone()), host_f32(got.value.clone()))),
        ("logp.action", max_abs(host_f32(want.heads.logp.action.clone()), host_f32(got.heads.logp.action.clone()))),
        ("dists.action", max_abs(host_f32(want.heads.dists.action.clone()), host_f32(got.heads.dists.action.clone()))),
    ] {
        assert!(d <= 1e-4, "061-AC5 / 032-AC1: {name} differs by {d:.2e} after load");
    }
    std::fs::remove_dir_all(&root).ok();
}

/// A game of `len` steps (1..=16) with its own replay hash.
fn game_len(len: usize, tag: u32) -> ShardGame {
    let mut g = game();
    g.steps.truncate(len);
    g.meta.replay_sha256 = format!("{tag:064x}");
    g
}

/// Shard files `<root>/p<k>/shard_00000.bin.zst` holding `games` in chunks of `per_file`.
fn shard_dir(name: &str, games: &[ShardGame], per_file: usize) -> std::path::PathBuf {
    use uranium_replay::shard::{Shard, ShardHeader, write};
    let root = std::env::temp_dir().join(format!("uranium-bc-{name}-{}", std::process::id()));
    std::fs::remove_dir_all(&root).ok();
    for (k, chunk) in games.chunks(per_file).enumerate() {
        let d = root.join(format!("p{k}"));
        std::fs::create_dir_all(&d).unwrap();
        write(&d.join("shard_00000.bin.zst"), &Shard { header: ShardHeader::default(), games: chunk.to_vec() }).unwrap();
    }
    root
}

fn varied_games() -> Vec<ShardGame> {
    [16, 3, 9, 16, 5, 1, 12, 7, 16, 2, 11, 4, 16, 8, 6].iter().enumerate().map(|(i, &l)| game_len(l, i as u32 + 1)).collect()
}

/// What a batch trains on: per lane, the label actions and delays of its steps.
fn batch_key(b: &Batch) -> Vec<Vec<(u16, usize)>> {
    b.parts().0.iter().map(|c| c.iter().map(|s| (s.label.action, s.label.delay_bucket)).collect()).collect()
}

/// 041-R13 one epoch: `epoch_updates` counts exactly the batches the trainer's lane walk makes.
#[test]
fn epoch_count_matches_the_lane_walk_041_r13() {
    for lanes in [1, 2, 3] {
        let mut cfg = tiny_bc(1, 0);
        cfg.lanes = lanes;
        let games = varied_games();
        let lens: Vec<usize> = games.iter().map(|g| g.steps.len()).collect();
        let mut f = feeder(&cfg);
        let mut src = VecSource(games.into());
        let mut n = 0u64;
        while f.next_batch(&mut src, None).unwrap().is_some() {
            n += 1;
        }
        assert_eq!(n, epoch_updates(lens, lanes, cfg.seq_len), "lanes {lanes}");
    }
}

/// 061-R10: a feeder restored from a batch's data state gives the batches that followed, with
/// no game trained twice (lanes resume mid-game; only the core state starts fresh).
#[test]
fn feeder_resume_continues_the_data_061_r10() {
    let root = shard_dir("feed-resume", &varied_games(), 4);
    let mut cfg = tiny_bc(1, 0);
    cfg.data.z_drop = 0.5;
    cfg.data.augment = true;
    let src = || ShardSource::new(&root).unwrap().shuffled(5);
    let mut a = feeder(&cfg);
    let mut sa = src();
    let mut saved = None;
    let mut kept = Vec::new();
    for i in 0..6 {
        let b = a.next_batch_from(&mut sa).unwrap().unwrap();
        if i == 2 {
            saved = Some(b.state.clone());
        } else if i > 2 {
            kept.push((batch_key(&b), b.state.clone()));
        }
    }
    let mut b = feeder(&cfg);
    let mut sb = src();
    b.restore(saved.as_ref().unwrap(), &mut sb).unwrap();
    for (key, state) in &kept {
        let x = b.next_batch_from(&mut sb).unwrap().unwrap();
        assert_eq!(&batch_key(&x), key);
        assert_eq!(&x.state, state);
    }
    std::fs::remove_dir_all(&root).ok();
}

/// The prefetch thread gives the same batches, in order, as the feeder on the trainer's thread,
/// at every depth, also 0 (no thread, 041-R16).
#[test]
fn prefetch_gives_the_feeder_batches_041_r16() {
    let root = shard_dir("prefetch", &varied_games(), 4);
    let cfg = tiny_bc(1, 0);
    let mut f = feeder(&cfg);
    let mut s = ShardSource::new(&root).unwrap().shuffled(9);
    let mut want = Vec::new();
    while let Some(b) = f.next_batch_from(&mut s).unwrap() {
        want.push((batch_key(&b), b.state));
    }
    assert!(want.len() > 3, "too few batches to tell the depths apart");
    for depth in [0, 1, 2] {
        let mut p = Prefetch::start(feeder(&cfg), ShardSource::new(&root).unwrap().shuffled(9), depth);
        let mut got = Vec::new();
        while let Some(b) = p.next().unwrap() {
            got.push((batch_key(&b), b.state));
        }
        assert_eq!(got, want, "depth {depth}");
    }
    std::fs::remove_dir_all(&root).ok();
}

/// 041-R16: `URANIUM_BC_PREFETCH` sets the depth; unset or empty keeps the default, junk fails.
#[test]
fn prefetch_depth_from_env_041_r16() {
    // SAFETY: the only test in this binary that touches this variable.
    unsafe { std::env::remove_var("URANIUM_BC_PREFETCH") };
    assert_eq!(uranium_train::bc::prefetch_depth().unwrap(), uranium_train::bc::PREFETCH_DEPTH_DEFAULT);
    for (v, want) in [("0", Some(0)), (" 1 ", Some(1)), ("", Some(uranium_train::bc::PREFETCH_DEPTH_DEFAULT)), ("two", None), ("-1", None)] {
        unsafe { std::env::set_var("URANIUM_BC_PREFETCH", v) };
        assert_eq!(uranium_train::bc::prefetch_depth().ok(), want, "{v:?}");
    }
    unsafe { std::env::remove_var("URANIUM_BC_PREFETCH") };
}

/// 041-R17 / R18: `URANIUM_BC_ENTITY_CHUNK` is a row count (unset or empty 0 = off) and
/// `URANIUM_BC_PACK_BATCH` is `1` on, unset, empty or `0` off; anything else fails.
#[test]
fn entity_chunk_and_pack_from_env_041_r17_r18() {
    // SAFETY: the only test in this binary that touches these variables.
    unsafe { std::env::remove_var("URANIUM_BC_ENTITY_CHUNK") };
    unsafe { std::env::remove_var("URANIUM_BC_PACK_BATCH") };
    assert_eq!(uranium_train::bc::entity_chunk().ok(), Some(0));
    assert_eq!(uranium_train::bc::pack_batch().ok(), Some(false));
    for (v, want) in [("16", Some(16)), (" 8 ", Some(8)), ("0", Some(0)), ("", Some(0)), ("x", None), ("-1", None)] {
        unsafe { std::env::set_var("URANIUM_BC_ENTITY_CHUNK", v) };
        assert_eq!(uranium_train::bc::entity_chunk().ok(), want, "{v:?}");
    }
    for (v, want) in [("1", Some(true)), (" 1 ", Some(true)), ("0", Some(false)), ("", Some(false)), ("yes", None), ("2", None)] {
        unsafe { std::env::set_var("URANIUM_BC_PACK_BATCH", v) };
        assert_eq!(uranium_train::bc::pack_batch().ok(), want, "{v:?}");
    }
    unsafe { std::env::remove_var("URANIUM_BC_ENTITY_CHUNK") };
    unsafe { std::env::remove_var("URANIUM_BC_PACK_BATCH") };
}

/// 041-R17 / R18: the entity encoder recomputed in row chunks (even and uneven) and packed
/// batches give the same updates as the plain path (tiny config: no dropout).
#[test]
fn entity_recompute_and_pack_keep_the_updates_041_r17_r18() {
    let net = tiny_net();
    let run = |micro_lanes, chunk, pack| {
        let mut t = tiny_trainer_with(micro_lanes, 0, net.clone());
        t.set_entity_chunk(chunk);
        t.feeder.set_pack(pack);
        let mut src = VecSource((0..4).map(|_| game()).collect::<VecDeque<_>>());
        (0..4).map(|_| t.train_step(&mut src).unwrap().unwrap().total).collect::<Vec<_>>()
    };
    for micro_lanes in [0, 1] {
        let plain = run(micro_lanes, 0, false);
        assert_eq!(run(micro_lanes, 0, true), plain, "micro_lanes {micro_lanes}: packed batches");
        for (chunk, pack) in [(1, false), (3, false), (2, true), (64, false)] {
            let other = run(micro_lanes, chunk, pack);
            for (a, b) in plain.iter().zip(&other) {
                assert!((a - b).abs() <= 1e-4 * a.abs().max(1.0), "micro_lanes {micro_lanes} chunk {chunk} pack {pack}: {plain:?} != {other:?}");
            }
        }
    }
}

/// 041-R17: with dropout on, the untracked first pass and the tracked recompute give the same
/// output under the same device seed (so the recompute's gradient is the first pass's).
#[test]
fn untracked_entity_pass_matches_the_recompute_041_r17() {
    let ad = Device::flex().autodiff();
    let mut cfg = tiny_cfg();
    cfg.entity.dropout = 0.5;
    let cols = ActionColumns::from_table(action_table()).unwrap();
    let net = UraniumNet::<LstmCore>::new(&cfg, &cols, &ad);
    let seq = prepare(&game(), true, Sym::IDENTITY, &MapStore::synthetic()).unwrap();
    let obs: Vec<_> = seq.steps.iter().map(|s| &s.obs).collect();
    let ob = obs_batch_padded(&obs, 0, &ad);
    let max_abs = |x: burn::tensor::Tensor<3>| x.abs().max().into_data().convert::<f32>().try_to_vec::<f32>().unwrap()[0];
    // Parallel tests share the device RNG: a draw of theirs between a seed and its forward
    // changes the masks, so a few attempts are allowed.
    let same = (0..5).any(|_| {
        ad.seed(11);
        let first = net.entity_untracked(&ob).e;
        ad.seed(11);
        let again = net.entity(&ob).e;
        assert!(!first.is_tracked() && again.is_tracked(), "only the recompute records a graph");
        max_abs(first - again.detach()) <= 1e-6
    });
    assert!(same, "same seed, same dropout masks");
    ad.seed(11);
    let first = net.entity_untracked(&ob).e;
    ad.seed(12);
    let other_seed = net.entity(&ob).e;
    assert!(max_abs(first - other_seed.detach()) > 1e-3, "dropout is on in both passes");
}

/// Evaluation drains the lanes: every step of every game is produced exactly once, also with
/// fewer games than lanes.
#[test]
fn drained_feeder_covers_every_step_041() {
    for (lanes, n) in [(4, 2), (2, 7), (3, 15)] {
        let mut cfg = tiny_bc(1, 0);
        cfg.lanes = lanes;
        let games: Vec<ShardGame> = varied_games().into_iter().take(n).collect();
        let want: usize = games.iter().map(|g| g.steps.len()).sum();
        let mut f = feeder(&cfg);
        f.drain = true;
        let mut src = VecSource(games.into());
        let mut got = 0;
        while let Some(b) = f.next_batch(&mut src, None).unwrap() {
            got += b.parts().0.iter().map(|c| c.len()).sum::<usize>();
        }
        assert_eq!(got, want, "lanes {lanes}, games {n}");
    }
}

/// Folder with a current-version precompute for the synthetic map hash `none` (B-2).
fn maps_dir_with_none(tag: &str) -> std::path::PathBuf {
    use uranium_env::features::{MapPrecompute, PRECOMPUTE_VERSION};
    let d = std::env::temp_dir().join(format!("uranium-bc-maps-{tag}-{}", std::process::id()));
    std::fs::remove_dir_all(&d).ok();
    std::fs::create_dir_all(&d).unwrap();
    let p = MapPrecompute { version: PRECOMPUTE_VERSION, ..Default::default() };
    std::fs::write(d.join("none.toml"), toml::to_string(&p).unwrap()).unwrap();
    d
}

/// B-2: a game whose map has no precompute is an error (never channels at 0), the start check
/// lists every missing map hash with its game count, and a folder with the file works.
#[test]
fn missing_map_precompute_fails_b2() {
    use uranium_train::data::{GameStream, check_maps};
    let empty = std::env::temp_dir().join(format!("uranium-bc-maps-empty-{}", std::process::id()));
    std::fs::create_dir_all(&empty).unwrap();
    let store = MapStore::dir(&empty);
    let err = GameStream::new(game(), false, Sym::IDENTITY, &store).err().expect("no precompute");
    assert!(format!("{err:#}").contains("no map precompute") && format!("{err:#}").contains("none"), "{err:#}");
    // The feeder (training and evaluation) fails the same way.
    let mut f = Feeder::new(&tiny_bc(1, 0)).with_maps(std::sync::Arc::new(MapStore::dir(&empty)));
    assert!(f.next_batch(&mut VecSource([game()].into()), None).is_err());

    let mut other = game_len(4, 99);
    other.meta.map_hash = "abc".into();
    let mut games = varied_games();
    games.push(other);
    let root = shard_dir("maps-check", &games, 4);
    let src = ShardSource::new(&root).unwrap();
    let err = check_maps(&src, None, &store).err().expect("start check fails");
    let msg = format!("{err:#}");
    assert!(msg.contains("2 map hash(es)") && msg.contains("abc (1 games)") && msg.contains("none (15 games)"), "{msg}");

    let dir = maps_dir_with_none("b2");
    let store = MapStore::dir(&dir);
    let err = check_maps(&src, None, &store).err().expect("abc still missing");
    assert!(format!("{err:#}").contains("1 map hash(es)"), "{err:#}");
    assert!(GameStream::new(game(), false, Sym::IDENTITY, &store).is_ok());
    std::fs::remove_dir_all(&root).ok();
    std::fs::remove_dir_all(&dir).ok();
    std::fs::remove_dir_all(&empty).ok();
}

/// B-3: the checkpoint carries the stage, config and dataset fingerprints; a resume with another
/// one is refused with the difference, the same one passes, and a checkpoint from before B-3
/// (none stored) resumes with a warning.
#[test]
fn resume_refuses_other_stage_config_or_data_b3() {
    use uranium_train::bc::{RunIdent, check_ident, read_ident, write_ident};
    let dir = std::env::temp_dir().join(format!("uranium-bc-ident-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    let t = tiny_trainer(1);
    t.save(&dir).unwrap();
    let want = RunIdent { stage: "S1".into(), config: t.cfg.fingerprint(), dataset: "3:abcd".into() };
    check_ident(&dir, &want).expect("no identity stored: resumed with a warning");
    write_ident(&dir, &want).unwrap();
    assert_eq!(read_ident(&dir).unwrap(), Some(want.clone()));
    check_ident(&dir, &want).unwrap();
    let mut b = tiny_trainer(1);
    assert!(b.resume(&dir).unwrap(), "the identity does not disturb the training state");
    for (field, other) in [
        ("stage", RunIdent { stage: "S2".into(), ..want.clone() }),
        ("config fingerprint", RunIdent { config: "0000".into(), ..want.clone() }),
        ("dataset fingerprint", RunIdent { dataset: "3:ffff".into(), ..want.clone() }),
    ] {
        let err = check_ident(&dir, &other).err().unwrap_or_else(|| panic!("{field} mismatch accepted"));
        assert!(format!("{err:#}").contains(field) && format!("{err:#}").contains("refusing to resume"), "{err:#}");
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// B-3: the dataset fingerprint follows the files' contents (manifest SHA-256, or size without a
/// manifest), not only their names.
#[test]
fn dataset_fingerprint_follows_contents_b3() {
    use uranium_train::data::dataset_fingerprint;
    let root = shard_dir("ds-fp", &varied_games(), 4);
    let src = ShardSource::new(&root).unwrap();
    let plain = dataset_fingerprint(&src).unwrap();
    assert_eq!(plain, dataset_fingerprint(&src).unwrap());
    uranium_train::storage::Manifest::build(&root).unwrap().write(&root).unwrap();
    let with_manifest = dataset_fingerprint(&src).unwrap();
    assert_ne!(plain, with_manifest, "manifest hashes are used when present");
    // Same file names, other contents (a rebuild).
    let games: Vec<ShardGame> = varied_games().into_iter().rev().collect();
    let again = shard_dir("ds-fp", &games, 4);
    uranium_train::storage::Manifest::build(&again).unwrap().write(&again).unwrap();
    let src2 = ShardSource::new(&again).unwrap();
    assert_eq!(src2.fingerprint(), src.fingerprint(), "the file list alone is the same");
    assert_ne!(dataset_fingerprint(&src2).unwrap(), with_manifest);
    std::fs::remove_dir_all(&again).ok();
}

/// 041-R7 / A-3: a game whose result is undecided (`result_source = none`) trains the policy
/// heads but adds nothing to the value loss; its count is kept in the data state.
#[test]
fn undecided_result_has_value_weight_zero_041_r7() {
    let net = tiny_net();
    let run = |source: &str| {
        let mut t = tiny_trainer_with(1, 0, net.clone());
        let mut g = game();
        g.meta.result_source = source.into();
        g.meta.result = if source == "none" { 0 } else { 1 };
        let mut src = VecSource((0..2).map(|_| g.clone()).collect::<VecDeque<_>>());
        let l = t.train_step(&mut src).unwrap().unwrap();
        (l, t.data.no_result_games)
    };
    let (none, n_none) = run("none");
    assert_eq!(none.value, 0.0, "value weight 0");
    assert!(none.action > 0.0 && none.total.is_finite(), "policy heads still train");
    assert_eq!(n_none, 2);
    let (bridge, n_bridge) = run("bridge");
    assert!(bridge.value > 0.0, "decided games train the value head");
    assert_eq!(n_bridge, 0);
    assert_eq!(uranium_train::data::value_weight(&game().meta), 1.0);
}

/// B-13: a resume decodes each shard once for all lanes; the games are the ones `game(id)` reads.
#[test]
fn batched_game_lookup_matches_single_reads_b13() {
    use uranium_train::data::GameId;
    let root = shard_dir("b13", &varied_games(), 4);
    let src = ShardSource::new(&root).unwrap().shuffled(5);
    let ids = [GameId { file: 2, game: 1 }, GameId { file: 0, game: 3 }, GameId { file: 2, game: 0 }, GameId { file: 1, game: 2 }];
    let got = src.games(&ids).unwrap();
    for (id, g) in ids.iter().zip(&got) {
        assert_eq!(&src.game(*id).unwrap(), g, "{id:?}");
    }
    assert!(src.games(&[GameId { file: 0, game: 0 }, GameId { file: 0, game: 0 }]).is_err(), "a game is asked for once");
    std::fs::remove_dir_all(&root).ok();
}

/// 032-R17 ② (G-D8): `bc-eval --games n` takes n games of a fixed-seed shuffle over the whole
/// split: the same games every time (both cores), spread over the files, and only those are read.
#[test]
fn eval_games_are_a_fixed_seed_sample_032_r17() {
    use uranium_train::data::{ChosenGames, StepSource, sample_games, scan_files};
    let a = sample_games(&[4, 0, 7, 3], 5, 1);
    assert_eq!(a, sample_games(&[4, 0, 7, 3], 5, 1), "deterministic");
    assert_eq!(a.len(), 5);
    assert!(a.windows(2).all(|w| (w[0].file, w[0].game) < (w[1].file, w[1].game)), "sorted, no repeats");
    assert!(a.iter().all(|g| g.file != 1 && g.game < [4, 0, 7, 3][g.file]));
    assert_ne!(a, sample_games(&[4, 0, 7, 3], 5, 2), "the seed matters");
    assert_eq!(sample_games(&[4, 0, 7, 3], 100, 1).len(), 14, "at most the whole split");

    let root = shard_dir("eval-sample", &varied_games(), 4);
    let pick = || {
        let src = ShardSource::new(&root).unwrap().select(uranium_train::split::Selection::Only(uranium_train::split::split_of(&game().meta)));
        let scan = scan_files(src.files()).unwrap();
        let ids = uranium_train::bc_eval::eval_games(&src, &scan, 6, 1);
        let mut chosen = ChosenGames::new(src, &ids);
        let mut got = Vec::new();
        while let Some(g) = chosen.next_game().unwrap() {
            got.push(g.meta.replay_sha256.clone());
        }
        (ids, got)
    };
    let (ids, first) = pick();
    assert_eq!(first.len(), ids.len());
    assert!(!first.is_empty());
    assert_eq!(pick().1, first, "same games on every call");
    std::fs::remove_dir_all(&root).ok();
}

/// 040-R18 / E-2: the game-start observation and the NoOp steps until the first command (delay 32
/// each, 031-R11) are ordinary usable steps: labels valid, trained, counted in the epoch.
#[test]
fn game_start_noop_steps_are_trained_and_counted_040_r18() {
    let noop = action_table().by_name("NoOp").unwrap().id;
    let mut g = game();
    let first = g.steps[0].clone();
    let lead: Vec<Step> = (0..3u32)
        .map(|k| {
            let mut s = first.clone();
            s.raw_frame.frame = 1 + 32 * k;
            s.action_label = Label { frame: 1 + 32 * k, action: noop, delay: 32, queued: false, selected: vec![], target_unit: None, target_tile: None };
            s
        })
        .collect();
    let n = g.steps.len();
    g.steps.splice(0..0, lead);
    let seq = prepare(&g, false, Sym::IDENTITY, &MapStore::synthetic()).unwrap();
    assert_eq!(seq.steps.len(), n + 3);
    assert_eq!(seq.invalid_labels, 0);
    assert!(seq.steps[..3].iter().all(|s| s.weight == 1.0 && s.label.action == noop && s.label.delay_bucket == 15), "NoOp(delay=32)");
    let mut t = tiny_trainer(1);
    let mut src = VecSource([g.clone(), g.clone()].into());
    let l = t.train_step(&mut src).unwrap().unwrap();
    assert!(l.total.is_finite() && l.valid == 8.0);
    assert_eq!(epoch_updates([g.steps.len()], 1, 4), (n as u64 + 3).div_ceil(4), "bc-epoch counts the leading steps");
}

/// E-10: split-stats gives per source the share of games and steps without a privileged summary
/// and the result sources (040-R8, A-3).
#[test]
fn split_stats_privileged_none_and_result_sources_e10() {
    use uranium_train::data::GameSummary;
    let meta = |src: &str, rs: &str| GameMeta { src: src.into(), result_source: rs.into(), ..game().meta };
    let games = [
        GameSummary { meta: meta("a", "bridge"), steps: 10, privileged_none: 0 },
        GameSummary { meta: meta("a", "none"), steps: 10, privileged_none: 5 },
        GameSummary { meta: meta("b", "leave"), steps: 4, privileged_none: 4 },
    ];
    let st = uranium_train::split::split_stats(games.iter());
    assert_eq!(st.sources.len(), 2);
    let a = &st.sources[0];
    assert_eq!((a.src.as_str(), a.games, a.games_privileged_none, a.steps_privileged_none), ("a", 2, 1, 5));
    assert_eq!((a.games_privileged_none_ratio, a.steps_privileged_none_ratio), (0.5, 0.25));
    assert_eq!(a.result_source.get("none"), Some(&1));
    assert_eq!(st.sources[1].steps_privileged_none_ratio, 1.0);
    let total: u64 = st.splits.iter().map(|r| r["games"].as_u64().unwrap()).sum();
    assert_eq!(total, 3);
}

/// Review T4a: the start scan is cached per dataset fingerprint, given back in the source's own
/// file order, and a changed dataset (another fingerprint) is scanned again, never served stale.
#[test]
fn scan_cache_follows_the_dataset_fingerprint() {
    use uranium_train::data::{scan_cached_in, scan_files};
    let root = shard_dir("scan-cache", &varied_games(), 4);
    let cache = std::env::temp_dir().join(format!("uranium-scan-cache-dir-{}", std::process::id()));
    std::fs::remove_dir_all(&cache).ok();
    let src = ShardSource::new(&root).unwrap().shuffled(3);
    let fresh = scan_files(src.files()).unwrap();
    assert_eq!(scan_cached_in(&src, Some(&cache)).unwrap(), fresh, "first call scans");
    let files: Vec<_> = std::fs::read_dir(&cache).unwrap().flatten().map(|e| e.path()).collect();
    assert_eq!(files.len(), 1, "one cache file: {files:?}");
    assert!(files[0].file_name().unwrap().to_string_lossy().starts_with("scan-"));
    // A cached answer is used (prove it by editing the cache) and reordered for another order.
    let text = std::fs::read_to_string(&files[0]).unwrap().replace("\"privileged_none\":0", "\"privileged_none\":7");
    std::fs::write(&files[0], text).unwrap();
    let other_order = ShardSource::new(&root).unwrap();
    let got = scan_cached_in(&other_order, Some(&cache)).unwrap();
    assert!(got.iter().flatten().all(|g| g.privileged_none == 7), "served from the cache");
    let want: Vec<usize> = scan_files(other_order.files()).unwrap().iter().map(|f| f.len()).collect();
    assert_eq!(got.iter().map(|f| f.len()).collect::<Vec<_>>(), want, "in the source's file order");
    // The dataset changes (a manifest appears, contents differ): new key, fresh scan.
    uranium_train::storage::Manifest::build(&root).unwrap().write(&root).unwrap();
    let again = scan_cached_in(&ShardSource::new(&root).unwrap(), Some(&cache)).unwrap();
    assert!(again.iter().flatten().all(|g| g.privileged_none == 0), "not the stale cache");
    assert_eq!(std::fs::read_dir(&cache).unwrap().count(), 2);
    std::fs::remove_dir_all(&root).ok();
    std::fs::remove_dir_all(&cache).ok();
}

/// B-3: every `BcConfig` field (and `[data]` key) is either in the fingerprinted
/// `ResultSettings` or listed in `EXECUTION_ONLY`; a new field fails here until it is classified.
/// Execution-only fields never move the fingerprint, result fields always do.
#[test]
fn config_fingerprint_fields_are_classified_b3() {
    use uranium_train::bc::EXECUTION_ONLY;
    const RESULT: [&str; 15] = [
        "lanes", "seq_len", "lr", "min_lr", "warmup_steps", "total_steps", "weight_decay", "grad_clip", "s2_lr_scale", "heads", "value", "data.z_drop", "data.augment", "data.seed", "data.enemy_race_mask",
    ];
    let mut cfg = tiny_bc(1, 0);
    cfg.keep_checkpoints = Some(3);
    let v = serde_json::to_value(&cfg).unwrap();
    let mut keys: Vec<String> = Vec::new();
    for (k, x) in v.as_object().unwrap() {
        if k == "data" {
            keys.extend(x.as_object().unwrap().keys().map(|d| format!("data.{d}")));
        } else {
            keys.push(k.clone());
        }
    }
    keys.sort();
    let mut classified: Vec<String> = RESULT.iter().chain(EXECUTION_ONLY.iter()).map(|s| s.to_string()).collect();
    classified.sort();
    assert_eq!(keys, classified, "classify new BcConfig fields: ResultSettings (result) or EXECUTION_ONLY");
    let view = serde_json::to_value(cfg.results_view()).unwrap();
    // Plus the code constants in the fingerprint (not config fields): the 041-R3 clip semantics.
    assert_eq!(view.as_object().unwrap().len(), RESULT.len() + 1, "ResultSettings holds exactly the result fields and clip_mode");
    assert_eq!(view["clip_mode"], uranium_train::clip::CLIP_MODE);

    let base = cfg.fingerprint();
    let mut c = cfg.clone();
    c.micro_lanes = 9;
    c.grad_checkpointing = true;
    c.pad_entities = 128;
    c.steps.s1 = 12345;
    c.keep_checkpoints = None;
    c.data.checkpoint_every = 7;
    c.data.checkpoint_minutes = 3;
    assert_eq!(c.fingerprint(), base, "execution-only settings");
    let changes: [fn(&mut BcConfig); 15] = [
        |c| c.lanes += 1,
        |c| c.seq_len += 1,
        |c| c.lr *= 2.0,
        |c| c.min_lr *= 2.0,
        |c| c.warmup_steps += 1,
        |c| c.total_steps += 1,
        |c| c.weight_decay += 0.1,
        |c| c.grad_clip += 0.5,
        |c| c.s2_lr_scale *= 2.0,
        |c| c.heads.delay += 0.1,
        |c| c.value.win += 0.1,
        |c| c.data.z_drop += 0.1,
        |c| c.data.augment = !c.data.augment,
        |c| c.data.seed += 1,
        |c| c.data.enemy_race_mask += 0.01,
    ];
    for (i, f) in changes.iter().enumerate() {
        let mut c = cfg.clone();
        f(&mut c);
        assert_ne!(c.fingerprint(), base, "{} changes the fingerprint", RESULT[i]);
    }
}

/// Copies the folder `src` to `dst` (files and subfolders), with or without the manifests.
fn copy_dir(src: &std::path::Path, dst: &std::path::Path, with_manifests: bool) {
    std::fs::create_dir_all(dst).unwrap();
    for e in std::fs::read_dir(src).unwrap().flatten() {
        let to = dst.join(e.file_name());
        if e.file_type().unwrap().is_dir() {
            copy_dir(&e.path(), &to, with_manifests);
        } else if with_manifests || e.file_name() != uranium_train::storage::MANIFEST_FILE {
            std::fs::copy(e.path(), &to).unwrap();
        }
    }
}

/// `<root>/datasets/replays/shards/{alpha,beta}/p<k>/shard_00000.bin.zst` with a manifest per
/// source (as the replay build leaves them on the share); `games` are split between the sources.
fn two_source_share(root: &std::path::Path, games: &[ShardGame]) -> std::path::PathBuf {
    let shards = root.join("datasets").join("replays").join("shards");
    let (a, b) = games.split_at(games.len() / 2);
    for (name, part) in [("alpha", a), ("beta", b)] {
        let tmp = shard_dir(&format!("canon-{name}-{}", root.file_name().unwrap().to_string_lossy()), part, 3);
        let dst = shards.join(name);
        copy_dir(&tmp, &dst, true);
        std::fs::remove_dir_all(&tmp).ok();
        uranium_train::storage::Manifest::build(&dst).unwrap().write(&dst).unwrap();
    }
    shards
}

/// The first `n` batches (what they train on, and their data state) of `src` shuffled with `seed`.
fn first_batches(src: ShardSource, seed: u64, n: usize) -> Vec<(Vec<Vec<(u16, usize)>>, uranium_train::bc::FeedState)> {
    let mut cfg = tiny_bc(1, 0);
    cfg.data.z_drop = 0.5;
    cfg.data.augment = true;
    let mut f = feeder(&cfg);
    let mut s = src.shuffled(seed);
    (0..n).map_while(|_| f.next_batch_from(&mut s).unwrap()).map(|b| (batch_key(&b), b.state)).collect()
}

/// 041-R13, 061-R10 / 061-AC1 (resume on another host): the same shard files give the same file
/// order, the same first batches and the same data-position and B-3 dataset fingerprints whether
/// they are opened with `--dataset` (pulled into a cache, or read in place) or `--shards`, in any
/// folder order, one folder per source or one folder holding all sources, wherever they are.
#[test]
fn same_shards_give_the_same_batches_for_any_spelling_or_root_041_r13_061_ac1() {
    use uranium_train::data::dataset_fingerprint;
    use uranium_train::storage::{StorageConfig, open_dataset};
    let base = std::env::temp_dir().join(format!("uranium-canon-{}", std::process::id()));
    std::fs::remove_dir_all(&base).ok();
    let share = base.join("share");
    let shards = two_source_share(&share, &varied_games());
    // Another host: the same files under another mount point, one level deeper.
    let other = base.join("mnt").join("deeper").join("shards");
    copy_dir(&shards, &other, true);

    // `--dataset`: pulled into a cache, or read in place (061-R12).
    let pulled = StorageConfig::new(&share, base.join("cache"), "a");
    let mut local = StorageConfig::new(&share, base.join("unused-cache"), "b");
    local.shared_local = true;
    let open = |cfg: &StorageConfig| -> Vec<std::path::PathBuf> {
        ["replays/shards/alpha", "replays/shards/beta"].iter().map(|r| open_dataset(cfg, std::path::Path::new(r)).unwrap()).collect()
    };
    let spellings: Vec<(&str, Vec<std::path::PathBuf>)> = vec![
        ("--dataset (cache)", open(&pulled)),
        ("--dataset (in place)", open(&local)),
        ("--shards in reverse order", vec![shards.join("beta"), shards.join("alpha")]),
        ("--shards at another root", vec![other.join("alpha"), other.join("beta")]),
        ("--shards of the folder of all sources", vec![other.clone()]),
    ];
    let seed = 7;
    let fp = |dirs: &[std::path::PathBuf]| dataset_fingerprint(&ShardSource::from_dirs(dirs).unwrap().shuffled(seed)).unwrap();
    let want_src = ShardSource::from_dirs(&spellings[0].1).unwrap();
    let want_keys = want_src.keys().to_vec();
    assert_eq!(want_keys.len(), 6);
    assert!(want_keys.iter().all(|k| k.starts_with("alpha/p") || k.starts_with("beta/p")), "{want_keys:?}");
    let want_list = want_src.fingerprint();
    let want_data = fp(&spellings[0].1);
    let want_batches = first_batches(want_src, seed, 6);
    assert_eq!(want_batches.len(), 6);
    for (name, dirs) in &spellings {
        let src = ShardSource::from_dirs(dirs).unwrap();
        assert_eq!(src.keys(), &want_keys[..], "{name}: file order");
        assert_eq!(src.fingerprint(), want_list, "{name}: data position fingerprint");
        assert_eq!(fp(dirs), want_data, "{name}: B-3 dataset fingerprint");
        assert_eq!(first_batches(src, seed, 6), want_batches, "{name}: first batches");
    }

    // A copy without manifests reads the same batches; its B-3 fingerprint falls back to file
    // sizes, so a resume from it is refused (with a message), never run silently on other data.
    let bare = base.join("bare");
    copy_dir(&shards, &bare, false);
    let dirs = vec![bare.join("alpha"), bare.join("beta")];
    assert_eq!(first_batches(ShardSource::from_dirs(&dirs).unwrap(), seed, 6), want_batches);
    assert_ne!(fp(&dirs), want_data);

    // Other data under the same file names (rebuilt sources): the same file list, another B-3
    // dataset fingerprint.
    let rebuilt = two_source_share(&base.join("rebuilt"), &varied_games().into_iter().rev().collect::<Vec<_>>());
    let dirs = vec![rebuilt.join("alpha"), rebuilt.join("beta")];
    assert_eq!(ShardSource::from_dirs(&dirs).unwrap().fingerprint(), want_list);
    assert_ne!(fp(&dirs), want_data);
    std::fs::remove_dir_all(&base).ok();
}

/// 041-R13: a source given twice (by both options, or from two places) is refused, not read twice.
#[test]
fn a_source_given_twice_is_refused_041_r13() {
    let base = std::env::temp_dir().join(format!("uranium-canon-dup-{}", std::process::id()));
    std::fs::remove_dir_all(&base).ok();
    let shards = two_source_share(&base, &varied_games());
    let copy = base.join("copy").join("alpha");
    copy_dir(&shards.join("alpha"), &copy, true);
    let err = ShardSource::from_dirs(&[shards.join("alpha"), copy]).err().expect("same keys twice");
    assert!(format!("{err:#}").contains("same key alpha/p0/shard_00000.bin.zst"), "{err:#}");
    // The same folder under another spelling (case on Windows): the same files, refused.
    let upper = shards.join("ALPHA");
    if upper.is_dir() {
        let err = ShardSource::from_dirs(&[shards.join("alpha"), upper]).err().expect("same files twice");
        assert!(format!("{err:#}").contains("given twice"), "{err:#}");
    }
    let err = ShardSource::from_dirs(&[shards.join("alpha"), shards.clone()]).err().expect("source and its parent");
    assert!(format!("{err:#}").contains("given twice"), "{err:#}");
    std::fs::remove_dir_all(&base).ok();
}

/// 041-R13 regression: for the build layout the canonical order and both fingerprints are exactly
/// what `--dataset` computed before the canonical keys (folders sorted by name, each in path
/// order; hashed text = the last three path parts from the file name up), so runs started with
/// `--dataset` keep their order and resume checks.
#[test]
fn build_layout_keeps_the_old_dataset_order_and_fingerprints_041_r13() {
    use sha2::{Digest, Sha256};
    use uranium_train::data::{Rng, dataset_fingerprint, shard_files, shuffle};
    let base = std::env::temp_dir().join(format!("uranium-canon-old-{}", std::process::id()));
    std::fs::remove_dir_all(&base).ok();
    let shards = two_source_share(&base, &varied_games());
    let seed = 11;
    let dirs = vec![shards.join("alpha"), shards.join("beta")];
    // Before: concatenation in (sorted) folder order, then the seeded shuffle.
    let mut old: Vec<std::path::PathBuf> = dirs.iter().flat_map(|d| shard_files(d).unwrap()).collect();
    shuffle(&mut old, &mut Rng::new(seed));
    let old_text = |f: &std::path::Path| f.components().rev().take(3).map(|c| c.as_os_str().to_string_lossy().into_owned()).collect::<Vec<_>>().join("/");
    let hex = |h: Sha256| h.finalize().iter().take(8).map(|b| format!("{b:02x}")).collect::<String>();
    let mut list = Sha256::new();
    let mut data = Sha256::new();
    for f in &old {
        list.update(old_text(f).as_bytes());
        list.update(b"\n");
        let d = dirs.iter().find(|d| f.starts_with(d)).unwrap();
        let m = uranium_train::storage::Manifest::read(d).unwrap();
        let rel = f.strip_prefix(d).unwrap().components().map(|c| c.as_os_str().to_string_lossy().into_owned()).collect::<Vec<_>>().join("/");
        data.update(old_text(f).as_bytes());
        data.update(b"\t");
        data.update(m.files[&rel].sha256.as_bytes());
        data.update(b"\n");
    }
    let (old_list, old_data) = (format!("{}:{}", old.len(), hex(list)), format!("{}:{}", old.len(), hex(data)));
    // Now, from either spelling and folder order.
    for d in [vec![shards.join("beta"), shards.join("alpha")], vec![shards.clone()]] {
        let src = ShardSource::from_dirs(&d).unwrap().shuffled(seed);
        let names: Vec<String> = src.files().iter().map(|f| old_text(f)).collect();
        assert_eq!(names, old.iter().map(|f| old_text(f)).collect::<Vec<_>>(), "order");
        assert_eq!(src.fingerprint(), old_list);
        assert_eq!(dataset_fingerprint(&src).unwrap(), old_data);
    }
    std::fs::remove_dir_all(&base).ok();
}

/// Every float parameter as (id, raw f32 bits), in visit order.
struct IdBits(Vec<(burn::module::ParamId, Vec<u32>)>);

impl burn::module::ModuleVisitor for IdBits {
    fn visit_float<const D: usize>(&mut self, p: &burn::module::Param<burn::tensor::Tensor<D>>) {
        self.0.push((p.id, host_f32(p.val()).iter().map(|x| x.to_bits()).collect()));
    }
}

/// (shared parameters, win_open head parameters) as bits.
fn split_bits(net: &UraniumNet<LstmCore>) -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
    let head = net.win_open_params();
    let mut v = IdBits(Vec::new());
    net.visit(&mut v);
    let (h, s): (Vec<_>, Vec<_>) = v.0.into_iter().partition(|(id, _)| head.contains(id));
    (s.into_iter().map(|x| x.1).collect(), h.into_iter().map(|x| x.1).collect())
}

/// Copies the float parameters of another net (as f32 data, in visit order) into every
/// parameter not in `skip`.
struct CopyFrom {
    src: Vec<Vec<f32>>,
    skip: Vec<burn::module::ParamId>,
    next: usize,
}

impl burn::module::ModuleMapper for CopyFrom {
    fn map_float<const D: usize>(&mut self, param: burn::module::Param<burn::tensor::Tensor<D>>) -> burn::module::Param<burn::tensor::Tensor<D>> {
        if self.skip.contains(&param.id) {
            return param;
        }
        let v = param.val();
        let data = burn::tensor::TensorData::new(self.src[self.next].clone(), v.shape());
        self.next += 1;
        let t = burn::tensor::Tensor::<D>::from_data(data, &v.device()).require_grad();
        burn::module::Param::initialized(param.id, t)
    }
}

/// 032-AC9 / 041-R19 (flex): the win_open head changes nothing else.
/// - Its initial weights do not come from the device RNG (032-R18 ③): two device seeds give the
///   same head. (Comparing whole nets built from one seed is not reliable here: the tests of this
///   binary run in parallel on the one Flex device RNG.)
/// - With the same shared weights (copied) and the same batches, the shared parameters stay
///   bit-identical with and without the head (head loss added, head clipped apart) and the
///   policy / value losses match; the head itself trains.
#[test]
fn win_open_head_leaves_the_rest_unchanged_032_ac9() {
    let cols = ActionColumns::from_table(action_table()).unwrap();
    let build = |open: bool, seed: u64| {
        let ad = Device::flex().autodiff();
        ad.seed(seed);
        let cfg = tiny_cfg().with_value(ValueConfig::new().with_hidden(32).with_privileged(8).with_win_open(open));
        let net = UraniumNet::<LstmCore>::new(&cfg, &cols, &ad);
        let mut bc = tiny_bc(1, 0);
        bc.value.win_open = open.then_some(0.5);
        let mut t = Trainer::new(net, cfg, bc, ad);
        t.feeder.set_maps(std::sync::Arc::new(MapStore::synthetic()));
        t
    };
    let mut off = build(false, 3);
    let mut on = build(true, 3);
    assert!(on.net.has_win_open() && !off.net.has_win_open());
    assert_eq!(on.net.win_open_params().len(), 6, "three Linear layers, weight and bias each");
    assert!(off.net.win_open_params().is_empty());
    let (_, head0) = split_bits(&on.net);
    let (_, head_other_seed) = split_bits(&build(true, 4).net);
    assert_eq!(head0, head_other_seed, "032-R18 ③: the head's initial weights do not use the device RNG");

    let (shared_off, none) = split_bits(&off.net);
    assert!(none.is_empty());
    let src = shared_off.iter().map(|b| b.iter().map(|&x| f32::from_bits(x)).collect()).collect();
    let skip = on.net.win_open_params();
    on.net = on.net.clone().map(&mut CopyFrom { src, skip, next: 0 });
    let (shared_on, head) = split_bits(&on.net);
    assert_eq!((shared_on, head), (shared_off, head0.clone()), "copied shared weights, head untouched");

    let games = || VecSource((0..4).map(|_| game()).collect::<VecDeque<_>>());
    let (mut src_off, mut src_on) = (games(), games());
    for _ in 0..2 {
        let a = off.train_step(&mut src_off).unwrap().unwrap();
        let b = on.train_step(&mut src_on).unwrap().unwrap();
        assert_eq!(a.win_open, 0.0);
        assert!(b.win_open > 0.0 && b.win_open.is_finite(), "{b:?}");
        assert_eq!((a.action.to_bits(), a.value.to_bits()), (b.action.to_bits(), b.value.to_bits()));
    }
    let (shared_off, _) = split_bits(&off.net);
    let (shared_on, head1) = split_bits(&on.net);
    assert_eq!(shared_on, shared_off, "041-R19: the other parameters update as without the head");
    assert_ne!(head1, head0, "the head trains");
}

/// 032-R18 ②: a checkpoint config written before `value.win_open` existed loads with the head off
/// (Burn's Config derive needs every field; `CheckpointMeta::net_config` fills the defaults).
#[test]
fn config_without_win_open_loads_032_r18() {
    let mut meta = uranium_model::checkpoint::CheckpointMeta::new(&tiny_cfg(), 1, 1).unwrap();
    meta.config["value"].as_object_mut().unwrap().remove("win_open").expect("serialised");
    assert!(serde_json::from_value::<UraniumNetConfig>(meta.config.clone()).is_err(), "the derive alone refuses a missing field");
    let cfg = meta.net_config().unwrap();
    assert!(!cfg.value.win_open);
    assert_eq!(cfg, tiny_cfg());
    // A stored value is kept.
    meta.config["value"]["win_open"] = serde_json::json!(true);
    assert!(meta.net_config().unwrap().value.win_open);

    // A whole checkpoint whose meta.json predates the key (as the core ablation's) loads.
    let dir = std::env::temp_dir().join(format!("uranium-bc-old-cfg-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    let t = tiny_trainer(1);
    t.save(&dir).unwrap();
    let path = dir.join(uranium_model::checkpoint::META_FILE);
    let mut json: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    json["config"]["value"].as_object_mut().unwrap().remove("win_open").expect("saved with the key");
    std::fs::write(&path, serde_json::to_string(&json).unwrap()).unwrap();
    let cols = ActionColumns::from_table(action_table()).unwrap();
    let (net, meta) = uranium_model::checkpoint::load::<LstmCore>(&dir, &cols, uranium_env::features::FEATURE_SCHEMA_VERSION, uranium_env::actions::ACTION_SCHEMA_VERSION, &Device::flex()).unwrap();
    assert!(!net.has_win_open() && !meta.net_config().unwrap().value.win_open);
    assert_eq!(param_bits(&net), param_bits(&t.net.valid()));
    std::fs::remove_dir_all(&dir).ok();
}
