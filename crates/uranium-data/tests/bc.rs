use uranium_data::bc::{AuxWeights, Batch, BcConfig, DataConfig, Feeder, HeadWeights, MonitorConfig, Prefetch, TrainSettings, ValueWeights};
use uranium_data::data::{MapStore, ShardSource, VecSource, epoch_updates, prepare};
use uranium_env::actions::LOC_CLASSES;
use uranium_env::features::CELLS;
use uranium_env::features::augment::Sym;
use uranium_env::features::extract::action_table;
use uranium_env::testkit::{flat_game_info, unit};
use uranium_env::types::TypeTables;
use uranium_proto::*;
use uranium_env::features::label::Label;
use uranium_data::shard::{GameMeta, ShardGame, Step};

fn short_stage_warmup(steps: usize) -> usize {
    (steps * 5 + 50) / 100
}

fn feeder(cfg: &BcConfig) -> Feeder {
    Feeder::new(cfg).with_maps(std::sync::Arc::new(MapStore::synthetic()))
}

const W: u16 = 32;

fn game() -> ShardGame {
    let n = W as usize * W as usize;
    let info = flat_game_info("tiny", (W, W), ["t", "z"], [(4, 4), (26, 26)]);
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
            memory_action: None,
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
    let seq = prepare(&g, true, Sym::IDENTITY, &MapStore::synthetic()).unwrap();
    assert!(seq.steps[0].obs.obs.z_present && seq.steps[0].obs.obs.z_build[0] != 0);
    assert!(!prepare(&g, false, Sym::IDENTITY, &MapStore::synthetic()).unwrap().steps[0].obs.obs.z_present);
}

#[test]
fn enemy_race_mask_per_game_030_r11() {
    use uranium_env::features::RACE_UNKNOWN;
    use uranium_data::data::{GameStream, enemy_race_masked};
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

#[test]
fn building_with_units_label_is_void_034_r7() {
    use uranium_env::features::{AgentMemory, MapInfo, extract};
    use uranium_data::data::host_label_check;
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

#[test]
fn label_check_train_supply_matches_mask_034_ac4() {
    use uranium_data::data::{LabelStats, label_check};
    let scv = TypeTables::get().unit_by_name("Terran_SCV").unwrap().id as u16;
    for queued in [false, true] {
        let mut g = game();
        for s in &mut g.steps {
            s.raw_frame.self_state.supply_used = 20;
            if queued {
                s.raw_frame.units[0].build_queue[0] = scv;
            }
        }
        let mut st = LabelStats::default();
        label_check(&g, &mut st, &MapStore::synthetic()).unwrap();
        let fails = st.fails.get(&("Train_SCV".to_string(), "type_mask")).copied().unwrap_or(0);
        let supply: usize = st.type_detail.iter().filter(|(k, _)| k.starts_with("Train_SCV <- supply")).map(|(_, n)| n).sum();
        if queued {
            assert_eq!((fails, supply), (0, 0), "queued producer: valid");
        } else {
            assert_eq!((fails, supply), (4, 4), "every queue empty: supply");
        }
    }
}

#[test]
fn build_label_check_matches_training_040_r19() {
    use uranium_env::features::label::label_validity;
    use uranium_data::data::{LabelStats, label_check};
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

#[test]
fn group_real_and_privileged_masks_041_r23() {
    use uranium_env::features::label::UNKNOWN_ACTION;
    let mut short = game_len(3, 1);
    short.steps[1].privileged = None;
    short.steps[2].action_label.action = UNKNOWN_ACTION;
    let long = game_len(16, 2);
    let cfg = tiny_bc(1, 0);
    let (lanes, t) = (cfg.lanes, cfg.seq_len);
    let mut f = feeder(&cfg);
    let mut src = VecSource(vec![short, long].into());
    let b = f.next_batch(&mut src, None).unwrap().unwrap();
    let g = b.group(0..lanes, t, 0, 12);
    let steps = b.lane_steps();
    let s = steps.iter().position(|&n| n == 3).expect("the short game's lane");
    let rows = |v: &[bool], l: usize| v[l * t..(l + 1) * t].to_vec();
    assert_eq!(rows(&g.real, s), [true, true, true, false]);
    assert_eq!(rows(&g.priv_present, s), [true, false, true, true], "the padding row repeats step 2 (has a summary)");
    assert_eq!(&g.weights[s * t..(s + 1) * t], &[1.0, 1.0, 0.0, 0.0], "unknown label: weight 0 but real");
    let p = g.privileged.p;
    assert!(g.privileged.summary[(s * t + 1) * p..(s * t + 2) * p].iter().all(|&x| x == 0.0), "no summary: zero fill");
    assert_eq!(rows(&g.real, 1 - s), [true; 4]);
    assert_eq!(rows(&g.priv_present, 1 - s), [true; 4]);
    let mut f = feeder(&cfg);
    f.drain = true;
    let mut src = VecSource(vec![game_len(2, 3)].into());
    let b = f.next_batch(&mut src, None).unwrap().unwrap();
    let g = b.group(0..lanes, t, 0, 12);
    let e = b.lane_steps().iter().position(|&n| n == 0).expect("a drained lane");
    assert_eq!(rows(&g.real, e), [false; 4]);
    assert_eq!(rows(&g.real, 1 - e), [true, true, false, false]);
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
        data: DataConfig { z_drop: 0.0, augment: false, seed: 1, checkpoint_every: 1000, checkpoint_minutes: 0, enemy_race_mask: 0.0, subsample: None, chain_start_sources: None },
        keep_checkpoints: None,
        aux: None,
        monitor: None,
        train: None,
    }
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
        data: DataConfig { z_drop: 0.25, augment: true, seed: 1, checkpoint_every: 1, checkpoint_minutes: 0, enemy_race_mask: 0.05, subsample: None, chain_start_sources: None },
        keep_checkpoints: None,
        aux: None,
        monitor: None,
        train: None,
    };
    assert!((cfg.lr_at(0, 1.0) - 1e-4).abs() < 1e-12);
    assert!((cfg.lr_at(9, 1.0) - 1e-3).abs() < 1e-12);
    assert!((cfg.lr_at(60, 1.0) - (1e-5 + 0.5 * (1e-3 - 1e-5))).abs() < 1e-9);
    assert!((cfg.lr_at(110, 1.0) - 1e-5).abs() < 1e-12);
    assert!((cfg.lr_at(9, 0.1) - 1e-4).abs() < 1e-12, "S2 = 1/10 (041-R4)");
    let file = BcConfig::load(std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../config/bc.toml"))).unwrap();
    assert_eq!((file.seq_len, file.grad_clip, file.value.win, file.data.z_drop), (64, 1.0, 0.5, 0.25));
    assert_eq!(file.data.enemy_race_mask, 0.05);
    assert!(file.steps.s1 > file.steps.s2 && file.steps.s2 > 0 && file.steps.ablation_a == file.steps.s2);
    assert_eq!(short_stage_warmup(7915), 396, "395.75 -> 396");
    assert_eq!(short_stage_warmup(10), 1, "0.5 -> 1");
    assert_eq!(short_stage_warmup(9), 0, "0.45 -> 0");
    assert_eq!(file.steps.s2_warmup, short_stage_warmup(file.steps.s2));
    assert_eq!(file.steps.ablation_a_warmup, short_stage_warmup(file.steps.ablation_a));
    assert_eq!(file.warmup_steps, 2000, "S1 keeps its warmup");
    let abl = BcConfig::load(std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../config/bc_core_ablation.toml"))).unwrap();
    assert_eq!((abl.steps.s1, abl.warmup_steps), (5300, short_stage_warmup(5300)));
    let mut same = abl.clone();
    same.steps = file.steps.clone();
    same.warmup_steps = file.warmup_steps;
    assert_eq!((abl.keep_checkpoints(), file.keep_checkpoints(), file.keep_checkpoints), (0, 5, None));
    same.keep_checkpoints = file.keep_checkpoints;
    assert_eq!((abl.value.win_open, file.value.win_open), (None, Some(0.5)));
    same.value.win_open = file.value.win_open;
    assert_eq!(same, file, "032-R17: only the stage lengths, warmup, retention and win_open differ from bc.toml");
    let abl2 = BcConfig::load(std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../config/bc_core_ablation_pt.toml"))).unwrap();
    assert_eq!((abl2.steps.s1, abl2.warmup_steps, abl2.keep_checkpoints()), (10000, 500, 0));
    assert_eq!(abl2.warmup_steps, short_stage_warmup(10000));
    let mut same2 = abl2.clone();
    same2.steps.s1 = file.steps.s1;
    same2.warmup_steps = file.warmup_steps;
    same2.keep_checkpoints = file.keep_checkpoints;
    assert_eq!(same2, file, "032-R17 (6): only the S1 length, warmup and retention differ from bc.toml");
    let view = serde_json::to_value(abl.results_view()).unwrap();
    assert_eq!(view["value"], serde_json::json!({ "win": 0.5 }));
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

fn game_len(len: usize, tag: u32) -> ShardGame {
    let mut g = game();
    g.steps.truncate(len);
    g.meta.replay_sha256 = format!("{tag:064x}");
    g
}

fn shard_dir(name: &str, games: &[ShardGame], per_file: usize) -> std::path::PathBuf {
    use uranium_data::shard::{Shard, ShardHeader, write};
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

fn batch_key(b: &Batch) -> Vec<Vec<(u16, usize)>> {
    b.parts().0.iter().map(|c| c.iter().map(|s| (s.label.action, s.label.delay_bucket)).collect()).collect()
}

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

fn epoch_digest(root: &std::path::Path) -> String {
    use sha2::{Digest, Sha256};
    let mut cfg = tiny_bc(1, 0);
    cfg.data.z_drop = 0.5;
    cfg.data.augment = true;
    cfg.data.enemy_race_mask = 0.5;
    let mut f = feeder(&cfg);
    let mut s = ShardSource::new(root).unwrap().shuffled(3);
    let mut h = Sha256::new();
    while let Some(b) = f.next_batch_from(&mut s).unwrap() {
        for l in 0..b.lanes() {
            h.update(b.group(l..l + 1, cfg.seq_len, 0, 12).digest().as_bytes());
        }
    }
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

#[test]
fn memory_action_none_keeps_batches_bit_identical_041_r28() {
    let root = shard_dir("memory-none", &varied_games(), 4);
    assert_eq!(epoch_digest(&root), MEMORY_NONE_DIGEST);
    std::fs::remove_dir_all(&root).ok();
}

const MEMORY_NONE_DIGEST: &str = "6f5f4b068ce79a7f107ad978578b23eed0fbbda5527517048c5b9d2f562a617f";

fn label_records(g: &ShardGame) -> Vec<uranium_data::shard::MemoryAction> {
    use uranium_env::features::label::{LabelWalker, acting_ids};
    let maps = MapStore::synthetic();
    let pre = maps.get(&g.meta.map_hash).unwrap();
    let mut w = LabelWalker::new(&g.game_info, uranium_env::maps::family_index(&g.meta.map_family), pre.as_deref(), g.meta.enemy_random).unwrap();
    g.steps
        .iter()
        .map(|s| {
            let (o, h, _) = w.step(&s.raw_frame, &s.action_label, None);
            uranium_data::shard::MemoryAction { action: h.action, delay_bucket: h.delay_bucket as u32, units: acting_ids(&h, &o) }
        })
        .collect()
}

fn with_label_memory(mut g: ShardGame) -> ShardGame {
    let rec = label_records(&g);
    for (i, s) in g.steps.iter_mut().enumerate() {
        s.memory_action = Some(if i == 0 { vec![] } else { vec![rec[i - 1].clone()] });
    }
    g
}

#[test]
fn loader_reads_memory_action_041_r28() {
    use uranium_data::data::TrainStep;
    let maps = MapStore::synthetic();
    let key = |s: &[TrainStep]| s.iter().map(|t| (t.label.clone(), t.weight, t.frame)).collect::<Vec<_>>();
    let g = game();
    let same = with_label_memory(g.clone());
    let mut empty = g.clone();
    empty.steps.iter_mut().for_each(|s| s.memory_action = Some(vec![]));
    for sym in [Sym::IDENTITY, Sym::all()[5]] {
        for z in [false, true] {
            let a = prepare(&g, z, sym, &maps).unwrap();
            let b = prepare(&same, z, sym, &maps).unwrap();
            let c = prepare(&empty, z, sym, &maps).unwrap();
            assert!(a.steps.iter().zip(&b.steps).all(|(x, y)| x.obs == y.obs), "{sym:?} z={z}");
            assert_eq!(key(&a.steps), key(&b.steps));
            assert_eq!(key(&a.steps), key(&c.steps), "labels and weights do not depend on memory");
            assert_eq!(a.steps[0].obs, c.steps[0].obs, "nothing recorded before the first step either way");
            assert!(a.steps.iter().zip(&c.steps).any(|(x, y)| x.obs != y.obs), "memory inputs follow memory_action");
        }
    }
}

#[test]
fn memory_action_shards_end_to_end_041_r28() {
    let same: Vec<ShardGame> = varied_games().into_iter().map(with_label_memory).collect();
    let root = shard_dir("memory-same", &same, 4);
    assert_eq!(uranium_data::shard::read(&root.join("p0/shard_00000.bin.zst")).unwrap().header.schema_version, 7);
    assert_eq!(epoch_digest(&root), MEMORY_NONE_DIGEST);
    std::fs::remove_dir_all(&root).ok();
    let mixed: Vec<ShardGame> = varied_games().into_iter().enumerate().map(|(i, g)| if i % 2 == 0 { with_label_memory(g) } else { g }).collect();
    let root = shard_dir("memory-mixed", &mixed, 4);
    assert_eq!(epoch_digest(&root), MEMORY_NONE_DIGEST);
    std::fs::remove_dir_all(&root).ok();
    let empty: Vec<ShardGame> = varied_games()
        .into_iter()
        .map(|mut g| {
            g.steps.iter_mut().for_each(|s| s.memory_action = Some(vec![]));
            g
        })
        .collect();
    let root = shard_dir("memory-empty", &empty, 4);
    assert_ne!(epoch_digest(&root), MEMORY_NONE_DIGEST);
    std::fs::remove_dir_all(&root).ok();
}

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

#[test]
fn feeder_threads_give_the_same_batches() {
    let root = shard_dir("feed-threads", &varied_games(), 4);
    let mut cfg = tiny_bc(1, 0);
    cfg.lanes = 5;
    cfg.data.z_drop = 0.5;
    cfg.data.augment = true;
    let run = |threads: usize, pack: bool| {
        let mut f = feeder(&cfg);
        f.set_threads(threads);
        f.set_pack(pack);
        let mut s = ShardSource::new(&root).unwrap().shuffled(3);
        let mut out = Vec::new();
        while let Some(b) = f.next_batch_from(&mut s).unwrap() {
            let steps: Vec<String> = b
                .chunks
                .iter()
                .enumerate()
                .flat_map(|(l, c)| c.iter().enumerate().map(move |(t, st)| (l, t, st)))
                .map(|(l, t, st)| {
                    let mut st = st.clone();
                    if pack {
                        let p = &b.planes[l][t];
                        assert_eq!(p.allocated(), p.bytes(), "packed planes keep only the compressed bytes");
                        st.obs = p.restore(&st.obs);
                    }
                    format!("{st:?}")
                })
                .collect();
            out.push((steps, b.state));
        }
        out
    };
    let want = run(1, false);
    assert!(want.len() > 2, "too few batches");
    for (threads, pack) in [(2, false), (3, true), (8, false), (1, true)] {
        assert!(run(threads, pack) == want, "{threads} threads, pack {pack}");
    }
    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn micro_lanes_from_env_041_r3() {
    use uranium_data::bc::parse_micro_lanes;
    assert_eq!(parse_micro_lanes(None).unwrap(), None);
    assert_eq!(parse_micro_lanes(Some(" ")).unwrap(), None);
    assert_eq!(parse_micro_lanes(Some("2")).unwrap(), Some(2));
    assert_eq!(parse_micro_lanes(Some("0")).unwrap(), Some(0));
    assert!(parse_micro_lanes(Some("two")).is_err());
}

#[test]
fn prefetch_depth_from_env_041_r16() {
    unsafe { std::env::remove_var("URANIUM_BC_PREFETCH") };
    assert_eq!(uranium_data::bc::prefetch_depth().unwrap(), uranium_data::bc::PREFETCH_DEPTH_DEFAULT);
    for (v, want) in [("0", Some(0)), (" 1 ", Some(1)), ("", Some(uranium_data::bc::PREFETCH_DEPTH_DEFAULT)), ("two", None), ("-1", None)] {
        unsafe { std::env::set_var("URANIUM_BC_PREFETCH", v) };
        assert_eq!(uranium_data::bc::prefetch_depth().ok(), want, "{v:?}");
    }
    unsafe { std::env::remove_var("URANIUM_BC_PREFETCH") };
}

#[test]
fn entity_chunk_and_pack_from_env_041_r17_r18() {
    unsafe { std::env::remove_var("URANIUM_BC_ENTITY_CHUNK") };
    unsafe { std::env::remove_var("URANIUM_BC_PACK_BATCH") };
    assert_eq!(uranium_data::bc::entity_chunk().ok(), Some(0));
    assert_eq!(uranium_data::bc::pack_batch().ok(), Some(false));
    for (v, want) in [("16", Some(16)), (" 8 ", Some(8)), ("0", Some(0)), ("", Some(0)), ("x", None), ("-1", None)] {
        unsafe { std::env::set_var("URANIUM_BC_ENTITY_CHUNK", v) };
        assert_eq!(uranium_data::bc::entity_chunk().ok(), want, "{v:?}");
    }
    for (v, want) in [("1", Some(true)), (" 1 ", Some(true)), ("0", Some(false)), ("", Some(false)), ("yes", None), ("2", None)] {
        unsafe { std::env::set_var("URANIUM_BC_PACK_BATCH", v) };
        assert_eq!(uranium_data::bc::pack_batch().ok(), want, "{v:?}");
    }
    unsafe { std::env::remove_var("URANIUM_BC_ENTITY_CHUNK") };
    unsafe { std::env::remove_var("URANIUM_BC_PACK_BATCH") };
}

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

#[test]
fn batch_lane_game_ids_041_r27() {
    let mut cfg = tiny_bc(1, 0);
    cfg.lanes = 3;
    let games = varied_games();
    let want: std::collections::BTreeMap<String, usize> = games.iter().map(|g| (g.meta.replay_sha256.clone(), g.steps.len())).collect();
    let mut f = feeder(&cfg);
    f.drain = true;
    let mut src = VecSource(games.into());
    let mut got = std::collections::BTreeMap::<String, usize>::new();
    let mut drained = 0;
    while let Some(b) = f.next_batch(&mut src, None).unwrap() {
        assert_eq!(b.game_ids.len(), b.lanes());
        for (l, c) in b.parts().0.iter().enumerate() {
            if c.is_empty() {
                assert!(b.game_ids[l].is_empty(), "a drained lane has no game id");
                drained += 1;
            } else {
                *got.entry(b.game_ids[l].clone()).or_default() += c.len();
            }
        }
    }
    assert_eq!(got, want);
    assert!(drained > 0, "the last batches drain lanes");
}

fn maps_dir_with_none(tag: &str) -> std::path::PathBuf {
    use uranium_env::features::{MapPrecompute, PRECOMPUTE_VERSION};
    let d = std::env::temp_dir().join(format!("uranium-bc-maps-{tag}-{}", std::process::id()));
    std::fs::remove_dir_all(&d).ok();
    std::fs::create_dir_all(&d).unwrap();
    let p = MapPrecompute { version: PRECOMPUTE_VERSION, ..Default::default() };
    std::fs::write(d.join("none.toml"), toml::to_string(&p).unwrap()).unwrap();
    d
}

#[test]
fn missing_map_precompute_fails_b2() {
    use uranium_data::data::{GameStream, check_maps};
    let empty = std::env::temp_dir().join(format!("uranium-bc-maps-empty-{}", std::process::id()));
    std::fs::create_dir_all(&empty).unwrap();
    let store = MapStore::dir(&empty);
    let err = GameStream::new(game(), false, Sym::IDENTITY, &store).err().expect("no precompute");
    assert!(format!("{err:#}").contains("no map precompute") && format!("{err:#}").contains("none"), "{err:#}");
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

#[test]
fn dataset_fingerprint_follows_contents_b3() {
    use uranium_data::data::dataset_fingerprint;
    let root = shard_dir("ds-fp", &varied_games(), 4);
    let src = ShardSource::new(&root).unwrap();
    let plain = dataset_fingerprint(&src).unwrap();
    assert_eq!(plain, dataset_fingerprint(&src).unwrap());
    uranium_store::storage::Manifest::build(&root).unwrap().write(&root).unwrap();
    let with_manifest = dataset_fingerprint(&src).unwrap();
    assert_ne!(plain, with_manifest, "manifest hashes are used when present");
    let games: Vec<ShardGame> = varied_games().into_iter().rev().collect();
    let again = shard_dir("ds-fp", &games, 4);
    uranium_store::storage::Manifest::build(&again).unwrap().write(&again).unwrap();
    let src2 = ShardSource::new(&again).unwrap();
    assert_eq!(src2.fingerprint(), src.fingerprint(), "the file list alone is the same");
    assert_ne!(dataset_fingerprint(&src2).unwrap(), with_manifest);
    std::fs::remove_dir_all(&again).ok();
}

#[test]
fn batched_game_lookup_matches_single_reads_b13() {
    use uranium_data::data::GameId;
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

#[test]
fn eval_games_are_a_fixed_seed_sample_032_r17() {
    use uranium_data::data::{ChosenGames, StepSource, sample_games, scan_files};
    let a = sample_games(&[4, 0, 7, 3], 5, 1);
    assert_eq!(a, sample_games(&[4, 0, 7, 3], 5, 1), "deterministic");
    assert_eq!(a.len(), 5);
    assert!(a.windows(2).all(|w| (w[0].file, w[0].game) < (w[1].file, w[1].game)), "sorted, no repeats");
    assert!(a.iter().all(|g| g.file != 1 && g.game < [4, 0, 7, 3][g.file]));
    assert_ne!(a, sample_games(&[4, 0, 7, 3], 5, 2), "the seed matters");
    assert_eq!(sample_games(&[4, 0, 7, 3], 100, 1).len(), 14, "at most the whole split");

    let root = shard_dir("eval-sample", &varied_games(), 4);
    let pick = || {
        let src = ShardSource::new(&root).unwrap().select(uranium_data::split::Selection::Only(uranium_data::split::split_of(&game().meta)));
        let scan = scan_files(src.files()).unwrap();
        let ids = uranium_data::bc::eval_games(&src, &scan, 6, 1);
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
    let src = || ShardSource::new(&root).unwrap().select(uranium_data::split::Selection::Only(uranium_data::split::split_of(&game().meta)));
    let mut whole = src();
    let mut want = Vec::new();
    while let Some(g) = whole.next_game().unwrap() {
        let id = whole.last_id().unwrap();
        if ids.contains(&id) {
            want.push((id, g.meta.replay_sha256.clone(), g.steps.len()));
        }
    }
    let mut chosen = ChosenGames::new(src(), &ids);
    let mut got = Vec::new();
    while let Some(g) = chosen.next_game().unwrap() {
        got.push((chosen.last_id().unwrap(), g.meta.replay_sha256.clone(), g.steps.len()));
    }
    assert_eq!(got, want);
    let f = 50;
    let mut cut = ChosenGames::with_until(src(), &ids, Some(f));
    let mut whole = ChosenGames::new(src(), &ids);
    let mut n = 0;
    while let Some(mut g) = whole.next_game().unwrap() {
        let c = cut.next_game().unwrap().expect("the same games");
        assert_eq!(cut.last_id(), whole.last_id());
        assert!(c.steps.len() >= 1 && c.steps.iter().skip(1).all(|s| s.raw_frame.frame < f));
        uranium_data::data::cut_game(&mut g, f);
        assert_eq!(c, g);
        n += 1;
    }
    assert!(cut.next_game().unwrap().is_none() && n == ids.len());
    drop(ChosenGames::new(src(), &ids));
    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn eval_game_metas_are_the_eval_games_052_r7() {
    use uranium_data::bc::{eval_game_metas, eval_games};
    use uranium_data::data::{ChosenGames, StepSource, scan_files};
    let mut games = varied_games();
    for (i, g) in games.iter_mut().enumerate() {
        g.meta.map_family = if i % 3 == 0 { "other" } else { "fighting_spirit" }.into();
    }
    let root = shard_dir("eval-metas", &games, 4);
    let src = || ShardSource::new(&root).unwrap();
    let scan = scan_files(src().files()).unwrap();
    let ids = eval_games(&src(), &scan, 6, 1);
    let mut chosen = ChosenGames::new(src(), &ids);
    let mut want = Vec::new();
    while let Some(g) = chosen.next_game().unwrap() {
        want.push(g.meta.replay_sha256.clone());
    }
    let sha = |v: Vec<uranium_data::shard::GameMeta>| v.into_iter().map(|m| m.replay_sha256).collect::<Vec<_>>();
    assert_eq!(sha(eval_game_metas(&src(), &scan, 6, 1, false)), want, "the bc-eval games");
    let t = eval_game_metas(&src(), &scan, 6, 1, true);
    assert_eq!(t.len(), 6);
    assert!(t.iter().all(|m| m.map_family == "fighting_spirit"), "target maps only");
    assert_eq!(t, eval_game_metas(&src(), &scan, 6, 1, true), "deterministic");
    assert_ne!(sha(t), sha(eval_game_metas(&src(), &scan, 6, 2, true)), "the seed matters");
    assert_eq!(eval_game_metas(&src(), &scan, 100, 1, true).len(), 10, "at most every target-map game");
    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn split_stats_privileged_none_and_result_sources_e10() {
    use uranium_data::data::GameSummary;
    let meta = |src: &str, rs: &str| GameMeta { src: src.into(), result_source: rs.into(), ..game().meta };
    let games = [
        GameSummary { meta: meta("a", "bridge"), steps: 10, privileged_none: 0 },
        GameSummary { meta: meta("a", "none"), steps: 10, privileged_none: 5 },
        GameSummary { meta: meta("b", "leave"), steps: 4, privileged_none: 4 },
    ];
    let st = uranium_data::split::split_stats(games.iter());
    assert_eq!(st.sources.len(), 2);
    let a = &st.sources[0];
    assert_eq!((a.src.as_str(), a.games, a.games_privileged_none, a.steps_privileged_none), ("a", 2, 1, 5));
    assert_eq!((a.games_privileged_none_ratio, a.steps_privileged_none_ratio), (0.5, 0.25));
    assert_eq!(a.result_source.get("none"), Some(&1));
    assert_eq!(st.sources[1].steps_privileged_none_ratio, 1.0);
    let total: u64 = st.splits.iter().map(|r| r["games"].as_u64().unwrap()).sum();
    assert_eq!(total, 3);
}

#[test]
fn scan_cache_is_per_file_041_r13() {
    use uranium_data::data::{scan_cached_in, scan_files};
    use uranium_data::shard::{Shard, ShardHeader, write};
    let root = shard_dir("scan-cache", &varied_games(), 4);
    let cache = std::env::temp_dir().join(format!("uranium-scan-cache-dir-{}", std::process::id()));
    std::fs::remove_dir_all(&cache).ok();
    let src = ShardSource::new(&root).unwrap().shuffled(3);
    let fresh = scan_files(src.files()).unwrap();
    assert_eq!(scan_cached_in(&src, Some(&cache)).unwrap(), fresh, "first call scans");
    assert_eq!(scan_cached_in(&src, Some(&cache)).unwrap(), fresh, "the cache gives back what was scanned");
    let entries = || -> Vec<std::path::PathBuf> { std::fs::read_dir(&cache).unwrap().flatten().map(|e| e.path()).collect() };
    let edit = |from: &str, to: &str| {
        for f in entries() {
            std::fs::write(&f, std::fs::read_to_string(&f).unwrap().replace(from, to)).unwrap();
        }
    };
    assert_eq!(entries().len(), src.files().len(), "one cache entry per shard file: {:?}", entries());
    assert!(entries().iter().all(|f| f.file_name().unwrap().to_string_lossy().starts_with("file-")));
    edit("\"privileged_none\":0", "\"privileged_none\":7");
    let other_order = ShardSource::new(&root).unwrap();
    let got = scan_cached_in(&other_order, Some(&cache)).unwrap();
    assert!(got.iter().flatten().all(|g| g.privileged_none == 7), "served from the cache");
    let want: Vec<usize> = scan_files(other_order.files()).unwrap().iter().map(|f| f.len()).collect();
    assert_eq!(got.iter().map(|f| f.len()).collect::<Vec<_>>(), want, "in the source's file order");
    let added = root.join("p9");
    std::fs::create_dir_all(&added).unwrap();
    write(&added.join("shard_00000.bin.zst"), &Shard { header: ShardHeader::default(), games: vec![game_len(5, 99)] }).unwrap();
    let grown = ShardSource::new(&root).unwrap().shuffled(3);
    let got = scan_cached_in(&grown, Some(&cache)).unwrap();
    assert_eq!(got.len(), src.files().len() + 1);
    for (f, games) in grown.files().iter().zip(&got) {
        let want = if f.starts_with(&added) { 0 } else { 7 };
        assert!(!games.is_empty() && games.iter().all(|g| g.privileged_none == want), "only the added file is read: {}", f.display());
    }
    assert_eq!(entries().len(), grown.files().len());
    edit("\"version\":2", "\"version\":1");
    let again = scan_cached_in(&grown, Some(&cache)).unwrap();
    assert!(again.iter().flatten().all(|g| g.privileged_none == 0), "another cache version is scanned again");
    assert_eq!(entries().len(), grown.files().len());
    edit("\"privileged_none\":0", "\"privileged_none\":7");
    uranium_store::storage::Manifest::build(&root).unwrap().write(&root).unwrap();
    let again = scan_cached_in(&ShardSource::new(&root).unwrap(), Some(&cache)).unwrap();
    assert!(again.iter().flatten().all(|g| g.privileged_none == 0), "not the stale cache");
    assert_eq!(entries().len(), 2 * grown.files().len());
    std::fs::remove_dir_all(&root).ok();
    std::fs::remove_dir_all(&cache).ok();
}

#[test]
fn config_fingerprint_fields_are_classified_b3() {
    use uranium_data::bc::EXECUTION_ONLY;
    const RESULT: [&str; 19] = [
        "lanes",
        "seq_len",
        "lr",
        "min_lr",
        "warmup_steps",
        "total_steps",
        "weight_decay",
        "grad_clip",
        "s2_lr_scale",
        "heads",
        "value",
        "data.z_drop",
        "data.augment",
        "data.seed",
        "data.enemy_race_mask",
        "data.subsample",
        "aux",
        "train",
        "data.chain_start_sources",
    ];
    let mut cfg = tiny_bc(1, 0);
    assert!(serde_json::to_value(cfg.results_view()).unwrap().get("subsample").is_none());
    assert!(serde_json::to_value(cfg.results_view()).unwrap().get("aux").is_none());
    assert!(serde_json::to_value(cfg.results_view()).unwrap().get("train").is_none());
    assert!(serde_json::to_value(cfg.results_view()).unwrap().get("chain_start_sources").is_none());
    assert!(serde_json::to_value(&cfg).unwrap()["data"].get("chain_start_sources").is_none());
    let json = serde_json::to_value(&cfg).unwrap();
    assert!(json.get("aux").is_none() && json.get("monitor").is_none() && json.get("train").is_none(), "041-R23 / R24 / R25: absent tables stay out of the config JSON");
    cfg.keep_checkpoints = Some(3);
    cfg.data.subsample = Some(uranium_data::split::Subsample { fraction: 0.25, keep_sources: vec!["botgen".into()], fraction_by_race: None, fraction_by_source: None });
    cfg.aux = Some(AuxWeights { privileged: Some(0.2), jepa: Some(0.1) });
    cfg.monitor = Some(MonitorConfig { spatial_every: 100 });
    cfg.train = Some(TrainSettings { model_seed: Some(1) });
    cfg.data.chain_start_sources = Some(vec!["botgen".into()]);
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
    assert_eq!(view.as_object().unwrap().len(), RESULT.len() + 1, "ResultSettings holds exactly the result fields and clip_mode");
    assert_eq!(view["clip_mode"], uranium_data::bc::CLIP_MODE);

    let base = cfg.fingerprint();
    let mut c = cfg.clone();
    c.micro_lanes = 9;
    c.grad_checkpointing = true;
    c.pad_entities = 128;
    c.steps.s1 = 12345;
    c.keep_checkpoints = None;
    c.data.checkpoint_every = 7;
    c.data.checkpoint_minutes = 3;
    c.monitor = None;
    assert_eq!(c.fingerprint(), base, "execution-only settings");
    c.monitor = Some(MonitorConfig { spatial_every: 7 });
    assert_eq!(c.fingerprint(), base, "041-R25: [monitor] is execution only");
    let changes: [fn(&mut BcConfig); 19] = [
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
        |c| c.data.subsample.as_mut().unwrap().fraction = 0.5,
        |c| c.aux.as_mut().unwrap().jepa = Some(0.3),
        |c| c.train.as_mut().unwrap().model_seed = Some(2),
        |c| c.data.chain_start_sources = Some(vec!["botgen".into(), "dagger".into()]),
    ];
    for (i, f) in changes.iter().enumerate() {
        let mut c = cfg.clone();
        f(&mut c);
        assert_ne!(c.fingerprint(), base, "{} changes the fingerprint", RESULT[i]);
    }
    for f in [|c: &mut BcConfig| c.data.subsample.as_mut().unwrap().keep_sources.clear(), |c: &mut BcConfig| c.data.subsample = None] {
        let mut c = cfg.clone();
        f(&mut c);
        assert_ne!(c.fingerprint(), base, "041-R21: the subsample's sources and presence change the fingerprint");
    }
    let mut c = cfg.clone();
    c.data.subsample.as_mut().unwrap().fraction_by_race = Some([("P".to_string(), 0.5)].into());
    assert_ne!(c.fingerprint(), base, "041-R26: fraction_by_race changes the fingerprint");
    for f in [|c: &mut BcConfig| c.aux = None, |c: &mut BcConfig| c.aux.as_mut().unwrap().privileged = None, |c: &mut BcConfig| c.aux.as_mut().unwrap().jepa = Some(0.0)] {
        let mut c = cfg.clone();
        f(&mut c);
        assert_ne!(c.fingerprint(), base, "041-R23: [aux] keys and presence change the fingerprint");
    }
    let mut c = cfg.clone();
    c.train = None;
    assert_ne!(c.fingerprint(), base, "041-R24: model_seed changes the fingerprint");
}

#[test]
fn checkpoint1_fingerprint_unchanged_041_ac6() {
    use uranium_data::bc::BcStage;
    let root = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../releases/checkpoint-1"));
    let cfg = BcConfig::load(&root.join("config/bc_dagger_r1.toml")).unwrap().for_stage(BcStage::S2);
    let state: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(root.join("train_state.json")).unwrap()).unwrap();
    let want = state["extra"]["run"]["config"].as_str().expect("train_state.json extra.run.config");
    assert_eq!(cfg.fingerprint(), want);
    assert!(cfg.aux.is_none() && cfg.monitor.is_none());
    let text = std::fs::read_to_string(root.join("config/bc_dagger_r1.toml")).unwrap();
    let with = |extra: &str| -> BcConfig {
        let path = std::env::temp_dir().join(format!("uranium-041-ac6-{}-{}.toml", std::process::id(), extra.len()));
        std::fs::write(&path, format!("{text}\n{extra}\n")).unwrap();
        let c = BcConfig::load(&path).unwrap().for_stage(BcStage::S2);
        std::fs::remove_file(&path).ok();
        c
    };
    let mon = with("[monitor]\nspatial_every = 100");
    assert_eq!(mon.monitor, Some(MonitorConfig { spatial_every: 100 }));
    assert_eq!(mon.fingerprint(), want, "041-R25: [monitor] is not in the fingerprint");
    let aux = with("[aux]\npriv = 0.2\njepa = 0.1");
    assert_eq!(aux.aux, Some(AuxWeights { privileged: Some(0.2), jepa: Some(0.1) }));
    assert_ne!(aux.fingerprint(), want, "041-R23: [aux] is in the fingerprint");
    let json = serde_json::to_value(&aux).unwrap();
    assert_eq!(json["aux"], serde_json::json!({ "priv": 0.2f32, "jepa": 0.1f32 }), "the Python trainer reads [aux] priv / jepa");
}

#[test]
fn aux_and_monitor_tables_are_checked_041_r23() {
    assert!(AuxWeights { privileged: Some(-0.1), jepa: None }.check().is_err());
    assert!(AuxWeights { privileged: None, jepa: Some(f32::NAN) }.check().is_err());
    assert!(AuxWeights { privileged: Some(0.0), jepa: Some(0.1) }.check().is_ok());
    assert!(AuxWeights::default().check().is_err(), "an empty [aux] table would change the fingerprint only");
    assert!(toml::from_str::<AuxWeights>("prv = 0.2").is_err());
    assert!(toml::from_str::<MonitorConfig>("spatial = 100").is_err());
    assert!(toml::from_str::<TrainSettings>("seed = 1").is_err());
    assert!(TrainSettings::default().check().is_err());
    assert_eq!(toml::from_str::<TrainSettings>("model_seed = 2").unwrap(), TrainSettings { model_seed: Some(2) });
    assert_eq!(toml::from_str::<AuxWeights>("priv = 0.2").unwrap(), AuxWeights { privileged: Some(0.2), jepa: None });
    assert_eq!(toml::from_str::<MonitorConfig>("").unwrap(), MonitorConfig { spatial_every: 0 });
}

#[test]
fn fine_tune_config_041_r21() {
    use uranium_data::bc::BcStage;
    use uranium_data::split::Subsample;
    let dir = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../config"));
    let file = BcConfig::load(&dir.join("bc.toml")).unwrap();
    let ft = BcConfig::load(&dir.join("bc_ft_empty.toml")).unwrap();
    assert!((ft.lr * ft.s2_lr_scale - 1.5e-5).abs() < 1e-12, "peak {}", ft.lr * ft.s2_lr_scale);
    assert!((ft.min_lr * ft.s2_lr_scale - 1.5e-7).abs() < 1e-14, "floor {}", ft.min_lr * ft.s2_lr_scale);
    assert_eq!(ft.data.seed, 3);
    assert_eq!(ft.data.subsample, Some(Subsample { fraction: 0.25, keep_sources: vec!["botgen".into()], fraction_by_race: None, fraction_by_source: None }));
    assert!(ft.steps.s2 > 0);
    assert_eq!(ft.steps.s2_warmup, short_stage_warmup(ft.steps.s2));
    let s2 = ft.clone().for_stage(BcStage::S2);
    assert_eq!((s2.total_steps, s2.warmup_steps), (ft.steps.s2, ft.steps.s2_warmup));
    assert!((s2.lr_at(s2.total_steps, s2.s2_lr_scale) - 1.5e-7).abs() < 1e-14, "the cosine ends at the floor");
    let mut same = ft.clone();
    same.lr = file.lr;
    same.min_lr = file.min_lr;
    same.data.seed = file.data.seed;
    same.data.subsample = None;
    same.steps = file.steps.clone();
    assert_eq!(same, file, "041-R21: only the learning rates, data seed, subsample and length differ from bc.toml");
}

#[test]
fn subsample_keeps_a_share_per_game_041_r21() {
    use uranium_data::data::enemy_race_masked;
    use uranium_data::split::Subsample;
    let sub = Subsample { fraction: 0.25, keep_sources: vec!["botgen".into()], fraction_by_race: None, fraction_by_source: None };
    let meta = |i: u32, src: &str| {
        let mut m = game().meta;
        m.replay_sha256 = format!("{:064x}", u64::from(i).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        m.src = src.into();
        m
    };
    let metas: Vec<GameMeta> = (0..20_000).map(|i| meta(i, "community")).collect();
    let kept: Vec<&GameMeta> = metas.iter().filter(|m| sub.keeps(m, 3)).collect();
    let share = kept.len() as f64 / metas.len() as f64;
    assert!((0.24..0.26).contains(&share), "kept share {share}");
    assert!(metas.iter().all(|m| sub.keeps(m, 3) == sub.keeps(m, 3)), "deterministic");
    assert!(metas.iter().any(|m| sub.keeps(m, 3) != sub.keeps(m, 4)), "the seed matters");
    let mut other = metas[0].clone();
    other.perspective_slot = 1;
    assert_eq!(sub.keeps(&other, 3), sub.keeps(&metas[0], 3), "both perspectives of a game");
    assert!((0..2_000).all(|i| sub.keeps(&meta(i, "botgen"), 3)), "teacher games are kept whole");
    let all = Subsample { fraction: 1.0, keep_sources: vec![], fraction_by_race: None, fraction_by_source: None };
    assert!(metas.iter().all(|m| all.keeps(m, 3)));
    let masked = kept.iter().filter(|m| enemy_race_masked(&m.replay_sha256, 3, 0.05)).count() as f64 / kept.len() as f64;
    assert!((0.035..0.065).contains(&masked), "race mask share among kept games {masked}");
    assert!(Subsample { fraction: 0.0, keep_sources: vec![], fraction_by_race: None, fraction_by_source: None }.check().is_err());
    assert!(Subsample { fraction: 1.5, keep_sources: vec![], fraction_by_race: None, fraction_by_source: None }.check().is_err());
    assert!(sub.check().is_ok());
}

#[test]
fn subsample_by_enemy_race_041_r26() {
    use uranium_data::split::{Subsample, enemy_race};
    assert_eq!((enemy_race("TvZ"), enemy_race("TvT"), enemy_race("T")), (Some("Z"), Some("T"), None));
    let by = Subsample { fraction: 0.25, keep_sources: vec!["botgen".into()], fraction_by_race: Some([("T".to_string(), 0.5), ("Z".to_string(), 0.1)].into()), fraction_by_source: None };
    let plain = Subsample { fraction_by_race: None, ..by.clone() };
    let meta = |i: u32, src: &str, mu: &str| {
        let mut m = game().meta;
        m.replay_sha256 = format!("{:064x}", u64::from(i).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        m.src = src.into();
        m.matchup = mu.into();
        m
    };
    for (mu, want) in [("TvT", 0.5), ("TvP", 0.25), ("TvZ", 0.1)] {
        let metas: Vec<GameMeta> = (0..20_000).map(|i| meta(i, "community", mu)).collect();
        let share = metas.iter().filter(|m| by.keeps(m, 3)).count() as f64 / metas.len() as f64;
        assert!((share - want).abs() < 0.01, "{mu}: kept share {share}, want {want}");
        assert!(metas.iter().all(|m| by.keeps(m, 3) == by.keeps(m, 3)), "deterministic");
        assert!(metas.iter().all(|m| by.fraction_of(m) == want));
        assert!((0..500).all(|i| by.keeps(&meta(i, "botgen", mu), 3)), "teacher games are kept whole");
    }
    let tvt = meta(7, "community", "TvT");
    let mut other = tvt.clone();
    other.perspective_slot = 1;
    assert_eq!(by.keeps(&other, 3), by.keeps(&tvt, 3), "both perspectives of a TvT game");
    let metas: Vec<GameMeta> = (0..2_000).map(|i| meta(i, "community", "TvP")).collect();
    assert!(metas.iter().all(|m| plain.keeps(m, 3) == Subsample { fraction: 0.25, keep_sources: vec!["botgen".into()], fraction_by_race: Some([("P".to_string(), 0.25)].into()), fraction_by_source: None }.keeps(m, 3)));
    assert!(by.check().is_ok());
    assert!(Subsample { fraction_by_race: Some([("X".to_string(), 0.5)].into()), ..plain.clone() }.check().is_err());
    assert!(Subsample { fraction_by_race: Some([("P".to_string(), 0.0)].into()), ..plain.clone() }.check().is_err());
    let s: Subsample = toml::from_str("fraction = 0.25\nkeep_sources = [\"botgen\"]\n[fraction_by_race]\nT = 0.5\nZ = 0.1\n").unwrap();
    assert_eq!(s, by);
    assert!(serde_json::to_value(&plain).unwrap().get("fraction_by_race").is_none(), "absent: the 041-R21 fingerprint JSON");
}

#[test]
fn subsample_by_source_044_r5() {
    use uranium_data::split::Subsample;
    let by = Subsample { fraction: 0.25, keep_sources: vec!["botgen".into(), "dagger".into()], fraction_by_race: Some([("T".to_string(), 0.5), ("Z".to_string(), 0.1)].into()), fraction_by_source: None };
    let with = Subsample { fraction_by_source: Some([("distill".to_string(), 0.7)].into()), ..by.clone() };
    let meta = |i: u32, src: &str, mu: &str| {
        let mut m = game().meta;
        m.replay_sha256 = format!("{:064x}", u64::from(i).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        m.src = src.into();
        m.matchup = mu.into();
        m
    };
    let d: Vec<GameMeta> = (0..20_000).map(|i| meta(i, "distill", "TvP")).collect();
    let share = d.iter().filter(|m| with.keeps(m, 15)).count() as f64 / d.len() as f64;
    assert!((share - 0.7).abs() < 0.01, "kept share {share}");
    for mu in ["TvT", "TvP", "TvZ"] {
        assert!((0..500).all(|i| with.fraction_of(&meta(i, "distill", mu)) == 0.7), "{mu}: the source share, not the race share");
        for src in ["community", "botgen", "dagger"] {
            let o: Vec<GameMeta> = (0..2_000).map(|i| meta(i, src, mu)).collect();
            assert!(o.iter().all(|m| with.keeps(m, 15) == by.keeps(m, 15) && with.fraction_of(m) == by.fraction_of(m)), "{src} {mu}: as without the key");
        }
    }
    assert_eq!(with.share_sources(), ["distill"]);
    assert!(by.share_sources().is_empty());
    assert!(with.check().is_ok());
    assert!(Subsample { fraction_by_source: Some([("distill".to_string(), 0.0)].into()), ..by.clone() }.check().is_err());
    assert!(Subsample { fraction_by_source: Some([("distill".to_string(), 1.5)].into()), ..by.clone() }.check().is_err());
    assert!(Subsample { fraction_by_source: Some([("dagger".to_string(), 0.5)].into()), ..by.clone() }.check().is_err(), "a kept source");
    let s: Subsample = toml::from_str("fraction = 0.25\nkeep_sources = [\"botgen\", \"dagger\"]\n[fraction_by_race]\nT = 0.5\nZ = 0.1\n[fraction_by_source]\ndistill = 0.7\n").unwrap();
    assert_eq!(s, with);
    assert!(serde_json::to_value(&by).unwrap().get("fraction_by_source").is_none(), "absent: the 041-R26 fingerprint JSON");
    assert!(toml::from_str::<Subsample>("fraction = 0.25\n[fraction_by_sources]\ndistill = 0.7\n").is_err());
    assert!(toml::from_str::<Subsample>("fraction = 0.25\nkeep_source = [\"botgen\"]\n").is_err());
    for dir in ["config", "releases/checkpoint-1/config"] {
        let dir = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../..")).join(dir);
        for e in std::fs::read_dir(&dir).unwrap().flatten() {
            let p = e.path();
            let text = std::fs::read_to_string(&p).unwrap_or_default();
            if p.extension().is_none_or(|x| x != "toml") || !text.lines().any(|l| l.trim() == "[data.subsample]") {
                continue;
            }
            match BcConfig::load(&p) {
                Ok(c) => assert!(c.data.subsample.is_some(), "{}", p.display()),
                Err(err) => assert!(text.contains("PLACEHOLDER") && format!("{err:#}").contains("must be in (0, 1]"), "{}: {err:#}", p.display()),
            }
        }
    }
    let dir = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../config"));
    let cfg = BcConfig::load(&dir.join("bc_m2_dagger_r2.toml")).unwrap();
    let mut c = cfg.clone();
    c.data.subsample.as_mut().unwrap().fraction_by_source = None;
    assert_eq!(c.fingerprint(), cfg.fingerprint());
    c.data.subsample.as_mut().unwrap().fraction_by_source = Some([("distill".to_string(), 0.7)].into());
    assert_ne!(c.fingerprint(), cfg.fingerprint(), "044-R5: fraction_by_source is in the fingerprint");
}

#[test]
fn bc_epoch_counts_distill_apart_044_r5() {
    use uranium_data::split::Subsample;
    let mut games: Vec<ShardGame> = (0..100).map(|i| game_len(1 + (i % 16) as usize, i + 1)).collect();
    for (i, g) in games.iter_mut().enumerate() {
        g.meta.map_family = "fighting_spirit".into();
        (g.meta.src, g.meta.matchup) = match i % 5 {
            0 => ("botgen".into(), "TvZ".into()),
            1 => ("community".into(), "TvT".into()),
            2 => ("community".into(), "TvZ".into()),
            _ => ("distill".into(), "TvP".into()),
        };
    }
    let root = shard_dir("mix-distill", &games, 9);
    let mut cfg = tiny_bc(1, 0);
    cfg.data.subsample = Some(Subsample { fraction: 1.0, keep_sources: vec!["botgen".into()], fraction_by_race: None, fraction_by_source: Some([("distill".to_string(), 1.0)].into()) });
    let t = uranium_data::bc::epoch_table(std::slice::from_ref(&root), &cfg).unwrap();
    let mix = t.s2_mix.as_ref().unwrap();
    let steps = |s: &str| -> u64 { t.s2.by_source.get(s).map_or(0, |r| r.values().map(|x| x.steps).sum()) };
    let (botgen, distill) = (steps("botgen"), steps("distill"));
    assert!(distill > 0 && botgen > 0);
    assert_eq!(mix.sources["distill"].steps, distill);
    assert_eq!((mix.teacher_steps, mix.human_steps), (botgen, t.s2.steps - botgen - distill));
    assert!(!mix.human_race_share.contains_key("P"), "the distill TvP games are not human: {:?}", mix.human_race_share);
    assert!(!mix.reference_human_race_share.contains_key("P"));
    assert!(mix.sources["distill"].ok.is_none() && mix.human.is_none(), "bc-epoch does not judge the three shares");
    cfg.data.subsample.as_mut().unwrap().fraction_by_source = None;
    let t2 = uranium_data::bc::epoch_table(std::slice::from_ref(&root), &cfg).unwrap();
    assert_eq!(t2.s2.steps, t.s2.steps);
    let m2 = t2.s2_mix.as_ref().unwrap();
    assert!(m2.sources.is_empty() && m2.human_steps == t.s2.steps - botgen && m2.human_race_share.contains_key("P"));
    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn subsampled_source_and_epoch_count_agree_041_r21() {
    use uranium_data::split::{Selection, Subsample};
    let mut games: Vec<ShardGame> = (0..60).map(|i| game_len(1 + (i % 16) as usize, i + 1)).collect();
    for g in games.iter_mut().skip(40) {
        g.meta.src = "botgen".into();
    }
    let root = shard_dir("subsample", &games, 7);
    let mut cfg = tiny_bc(1, 0);
    cfg.lanes = 2;
    cfg.data.subsample = Some(Subsample { fraction: 0.5, keep_sources: vec!["botgen".into()], fraction_by_race: None, fraction_by_source: None });
    let src = cfg.train_source(std::slice::from_ref(&root), Selection::S1).unwrap();
    let mut read: Vec<GameMeta> = Vec::new();
    for k in 0..src.files().len() {
        read.extend(src.file_games(k).unwrap().into_iter().map(|g| g.meta));
    }
    let train: Vec<&ShardGame> = games.iter().filter(|g| uranium_data::split::selects(Selection::S1, &g.meta)).collect();
    let teacher = train.iter().filter(|g| g.meta.src == "botgen").count();
    assert_eq!(read.iter().filter(|m| m.src == "botgen").count(), teacher, "every teacher train game");
    let replays = read.iter().filter(|m| m.src == "test").count();
    let all_replays = train.len() - teacher;
    assert!(replays > 0 && replays < all_replays, "a part of the other games ({replays} of {all_replays})");
    let scan = uranium_data::data::scan_files(src.files()).unwrap();
    let per = uranium_data::bc::subsample_counts(&scan, Selection::S1, cfg.data.subsample.as_ref().unwrap(), cfg.data.seed);
    assert_eq!(per["botgen"], (teacher, teacher));
    assert_eq!(per["test"], (all_replays, replays));
    let mut f = feeder(&cfg);
    let mut s = cfg.train_source(std::slice::from_ref(&root), Selection::S1).unwrap();
    let mut n = 0u64;
    while f.next_batch_from(&mut s).unwrap().is_some() {
        n += 1;
    }
    let table = uranium_data::bc::epoch_table(std::slice::from_ref(&root), &cfg).unwrap();
    assert_eq!(table.s1.updates, n, "bc-epoch counts the subsampled walk");
    assert_eq!(table.s1.games, read.len());
    let mut whole = cfg.clone();
    whole.data.subsample = None;
    assert!(uranium_data::bc::epoch_table(std::slice::from_ref(&root), &whole).unwrap().s1.games > read.len());
    std::fs::remove_dir_all(&root).ok();
}

fn mix_scan() -> Vec<uranium_data::data::GameSummary> {
    use uranium_data::data::GameSummary;
    let summary = |i: u64, src: &str, mu: &str, slot: u8, steps: usize| {
        let mut m = game().meta;
        m.replay_sha256 = format!("{:064x}", i.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        (m.src, m.matchup, m.perspective_slot, m.map_family) = (src.into(), mu.into(), slot, "fighting_spirit".into());
        GameSummary { meta: m, steps, privileged_none: 0 }
    };
    let mut v = Vec::new();
    for i in 0..300 {
        v.push(summary(i, "botgen", "TvZ", 0, 400 + (i as usize * 37) % 300));
    }
    for i in 1_000..2_000 {
        let steps = 300 + (i as usize * 53) % 900;
        v.push(summary(i, "community", "TvT", 0, steps));
        v.push(summary(i, "community", "TvT", 1, steps + (i as usize % 7) * 10));
    }
    for i in 3_000..6_000 {
        v.push(summary(i, if i % 3 == 0 { "stardata" } else { "scr_ladder" }, "TvP", 0, 300 + (i as usize * 71) % 1_000));
    }
    for i in 7_000..8_000 {
        v.push(summary(i, "personal", "TvZ", 0, 250 + (i as usize * 29) % 800));
    }
    v
}

#[test]
fn race_fractions_hit_the_mix_shares_041_r26() {
    use uranium_data::mix::{MixShares, count_by_source, race_fractions};
    use uranium_data::split::Subsample;
    let scan = mix_scan();
    let teacher = vec!["botgen".to_string()];
    let all = count_by_source(&scan);
    assert_eq!(all["community"]["T"].games, 2_000);
    assert_eq!(all["botgen"]["Z"].steps, scan.iter().filter(|g| g.meta.src == "botgen").map(|g| g.steps as u64).sum::<u64>());
    for seed in [3u64, 4, 10] {
        let f = race_fractions(&scan, &teacher, seed, 0.5).unwrap();
        assert_eq!(f, race_fractions(&scan, &teacher, seed, 0.5).unwrap(), "deterministic");
        assert_eq!(f.fraction_by_race.keys().collect::<Vec<_>>(), ["P", "T", "Z"]);
        let sub = Subsample { fraction: f.fraction, keep_sources: teacher.clone(), fraction_by_race: Some(f.fraction_by_race.clone()), fraction_by_source: None };
        sub.check().unwrap();
        let kept: Vec<_> = scan.iter().filter(|g| sub.keeps(&g.meta, seed)).cloned().collect();
        let mix = MixShares::new(&count_by_source(&kept), &all, &teacher);
        assert!(mix.teacher_ok && mix.race_ok, "seed {seed}: {mix:?}");
        assert!((mix.teacher_share - 0.5).abs() < 0.01, "seed {seed}: teacher {}", mix.teacher_share);
        for (r, s) in &mix.human_race_share {
            assert!((s - mix.reference_human_race_share[r]).abs() < 0.01, "seed {seed} {r}: {s} vs {}", mix.reference_human_race_share[r]);
        }
        assert_eq!(mix.human_race_steps, f.kept_human_race_steps, "the fractions keep what they say");
        assert_eq!(mix.teacher_steps, f.teacher_steps, "every teacher game");
        let tvt: Vec<_> = kept.iter().filter(|g| g.meta.matchup == "TvT").collect();
        assert_eq!(tvt.iter().filter(|g| g.meta.perspective_slot == 0).count(), tvt.iter().filter(|g| g.meta.perspective_slot == 1).count());
    }
    let f = race_fractions(&scan, &teacher, 3, 0.5).unwrap();
    let wide = Subsample { fraction: f.fraction * 3.0, keep_sources: teacher.clone(), fraction_by_race: None, fraction_by_source: None };
    let mix = MixShares::new(&count_by_source(&scan.iter().filter(|g| wide.keeps(&g.meta, 3)).cloned().collect::<Vec<_>>()), &all, &teacher);
    assert!(!mix.teacher_ok, "{}", mix.teacher_share);
    let mut by = f.fraction_by_race.clone();
    *by.get_mut("T").unwrap() *= 3.0;
    let skew = Subsample { fraction: f.fraction, keep_sources: teacher.clone(), fraction_by_race: Some(by), fraction_by_source: None };
    let mix = MixShares::new(&count_by_source(&scan.iter().filter(|g| skew.keeps(&g.meta, 3)).cloned().collect::<Vec<_>>()), &all, &teacher);
    assert!(!mix.race_ok, "{:?}", mix.human_race_share);
    assert!(race_fractions(&scan, &["nothing".to_string()], 3, 0.5).is_err(), "no teacher steps");
    let f = race_fractions(&scan, &teacher, 13, 0.5954).unwrap();
    let sub = Subsample { fraction: f.fraction, keep_sources: teacher.clone(), fraction_by_race: Some(f.fraction_by_race.clone()), fraction_by_source: None };
    let mix = MixShares::new(&count_by_source(&scan.iter().filter(|g| sub.keeps(&g.meta, 13)).cloned().collect::<Vec<_>>()), &all, &teacher);
    assert_eq!(mix.teacher_share_target, 0.5);
    assert!(!mix.teacher_ok && mix.race_ok, "0.5954 is outside 0.5 ± 0.03: {}", mix.teacher_share);
    let at = mix.clone().with_target(0.5954);
    assert!((at.teacher_share - 0.5954).abs() < 0.01 && at.teacher_ok && at.race_ok, "{at:?}");
    assert_eq!(at.teacher_share_target, 0.5954);
    assert_eq!(MixShares { teacher_share_target: 0.5954, teacher_ok: true, ..mix.clone() }, at, "only the judgement changes");
    assert!(!mix.clone().with_target(0.66).teacher_ok, "outside 0.66 ± 0.03");
    assert_eq!(mix.clone().with_target(0.5), mix, "--teacher-share 0.5 (the default) is the mix epoch's judgement");
}

#[test]
fn bc_epoch_counts_by_source_and_race_041_r26() {
    use uranium_data::split::Subsample;
    let mut games: Vec<ShardGame> = (0..80).map(|i| game_len(1 + (i % 16) as usize, i + 1)).collect();
    for (i, g) in games.iter_mut().enumerate() {
        g.meta.map_family = if i % 5 == 0 { "other".into() } else { "fighting_spirit".into() };
        (g.meta.src, g.meta.matchup) = match i % 4 {
            0 => ("botgen".into(), "TvZ".into()),
            1 => ("community".into(), "TvT".into()),
            2 => ("community".into(), "TvP".into()),
            _ => ("stardata".into(), "TvZ".into()),
        };
    }
    let root = shard_dir("mix-count", &games, 9);
    let mut cfg = tiny_bc(1, 0);
    let t = uranium_data::bc::epoch_table(std::slice::from_ref(&root), &cfg).unwrap();
    assert!(t.s2_mix.is_none(), "no subsample, no mix");
    for c in [&t.s1, &t.s2] {
        let steps: u64 = c.by_source.values().flat_map(|r| r.values()).map(|x| x.steps).sum();
        let n: usize = c.by_source.values().flat_map(|r| r.values()).map(|x| x.games).sum();
        assert_eq!((steps, n), (c.steps, c.games));
    }
    assert!(t.s2.by_source["community"].contains_key("T") && t.s2.by_source["community"].contains_key("P"));
    cfg.data.subsample = Some(Subsample { fraction: 0.5, keep_sources: vec!["botgen".into()], fraction_by_race: None, fraction_by_source: None });
    let t = uranium_data::bc::epoch_table(std::slice::from_ref(&root), &cfg).unwrap();
    let mix = t.s2_mix.as_ref().unwrap();
    let botgen: u64 = t.s2.by_source["botgen"].values().map(|x| x.steps).sum();
    assert_eq!((mix.teacher_steps, mix.human_steps), (botgen, t.s2.steps - botgen));
    assert!((mix.teacher_share - botgen as f64 / t.s2.steps as f64).abs() < 1e-12);
    let json = serde_json::to_value(&t).unwrap();
    assert!(json["s2_mix"]["reference_human_race_share"].is_object() && json["s2"]["by_source"]["botgen"]["Z"]["steps"].is_u64());
    let (ra, rb) = (shard_dir("mix-count-a", &games[..40], 9), shard_dir("mix-count-b", &games[40..], 9));
    let g = uranium_data::bc::epoch_table_groups(&[vec![ra.clone()], vec![rb.clone()]], &cfg).unwrap();
    let one = uranium_data::bc::epoch_table(&[ra.clone(), rb.clone()], &cfg).unwrap();
    assert_eq!(serde_json::to_value(&g).unwrap(), serde_json::to_value(&one).unwrap());
    assert_eq!(g.s2.steps, t.s2.steps);
    std::fs::remove_dir_all(&ra).ok();
    std::fs::remove_dir_all(&rb).ok();
    std::fs::remove_dir_all(&root).ok();
}

fn copy_dir(src: &std::path::Path, dst: &std::path::Path, with_manifests: bool) {
    std::fs::create_dir_all(dst).unwrap();
    for e in std::fs::read_dir(src).unwrap().flatten() {
        let to = dst.join(e.file_name());
        if e.file_type().unwrap().is_dir() {
            copy_dir(&e.path(), &to, with_manifests);
        } else if with_manifests || e.file_name() != uranium_store::storage::MANIFEST_FILE {
            std::fs::copy(e.path(), &to).unwrap();
        }
    }
}

fn two_source_share(root: &std::path::Path, games: &[ShardGame]) -> std::path::PathBuf {
    let shards = root.join("datasets").join("replays").join("shards");
    let (a, b) = games.split_at(games.len() / 2);
    for (name, part) in [("alpha", a), ("beta", b)] {
        let tmp = shard_dir(&format!("canon-{name}-{}", root.file_name().unwrap().to_string_lossy()), part, 3);
        let dst = shards.join(name);
        copy_dir(&tmp, &dst, true);
        std::fs::remove_dir_all(&tmp).ok();
        uranium_store::storage::Manifest::build(&dst).unwrap().write(&dst).unwrap();
    }
    shards
}

fn first_batches(src: ShardSource, seed: u64, n: usize) -> Vec<(Vec<Vec<(u16, usize)>>, uranium_data::bc::FeedState)> {
    let mut cfg = tiny_bc(1, 0);
    cfg.data.z_drop = 0.5;
    cfg.data.augment = true;
    let mut f = feeder(&cfg);
    let mut s = src.shuffled(seed);
    (0..n).map_while(|_| f.next_batch_from(&mut s).unwrap()).map(|b| (batch_key(&b), b.state)).collect()
}

#[test]
fn same_shards_give_the_same_batches_for_any_spelling_or_root_041_r13_061_ac1() {
    use uranium_data::data::dataset_fingerprint;
    use uranium_store::storage::{StorageConfig, open_dataset};
    let base = std::env::temp_dir().join(format!("uranium-canon-{}", std::process::id()));
    std::fs::remove_dir_all(&base).ok();
    let share = base.join("share");
    let shards = two_source_share(&share, &varied_games());
    let other = base.join("mnt").join("deeper").join("shards");
    copy_dir(&shards, &other, true);

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

    let bare = base.join("bare");
    copy_dir(&shards, &bare, false);
    let dirs = vec![bare.join("alpha"), bare.join("beta")];
    assert_eq!(first_batches(ShardSource::from_dirs(&dirs).unwrap(), seed, 6), want_batches);
    assert_ne!(fp(&dirs), want_data);

    let rebuilt = two_source_share(&base.join("rebuilt"), &varied_games().into_iter().rev().collect::<Vec<_>>());
    let dirs = vec![rebuilt.join("alpha"), rebuilt.join("beta")];
    assert_eq!(ShardSource::from_dirs(&dirs).unwrap().fingerprint(), want_list);
    assert_ne!(fp(&dirs), want_data);
    std::fs::remove_dir_all(&base).ok();
}

#[test]
fn a_source_given_twice_is_refused_041_r13() {
    let base = std::env::temp_dir().join(format!("uranium-canon-dup-{}", std::process::id()));
    std::fs::remove_dir_all(&base).ok();
    let shards = two_source_share(&base, &varied_games());
    let copy = base.join("copy").join("alpha");
    copy_dir(&shards.join("alpha"), &copy, true);
    let err = ShardSource::from_dirs(&[shards.join("alpha"), copy]).err().expect("same keys twice");
    assert!(format!("{err:#}").contains("same key alpha/p0/shard_00000.bin.zst"), "{err:#}");
    let upper = shards.join("ALPHA");
    if upper.is_dir() {
        let err = ShardSource::from_dirs(&[shards.join("alpha"), upper]).err().expect("same files twice");
        assert!(format!("{err:#}").contains("given twice"), "{err:#}");
    }
    let err = ShardSource::from_dirs(&[shards.join("alpha"), shards.clone()]).err().expect("source and its parent");
    assert!(format!("{err:#}").contains("given twice"), "{err:#}");
    std::fs::remove_dir_all(&base).ok();
}

#[test]
fn build_layout_keeps_the_old_dataset_order_and_fingerprints_041_r13() {
    use sha2::{Digest, Sha256};
    use uranium_data::data::{Rng, dataset_fingerprint, shard_files, shuffle};
    let base = std::env::temp_dir().join(format!("uranium-canon-old-{}", std::process::id()));
    std::fs::remove_dir_all(&base).ok();
    let shards = two_source_share(&base, &varied_games());
    let seed = 11;
    let dirs = vec![shards.join("alpha"), shards.join("beta")];
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
        let m = uranium_store::storage::Manifest::read(d).unwrap();
        let rel = f.strip_prefix(d).unwrap().components().map(|c| c.as_os_str().to_string_lossy().into_owned()).collect::<Vec<_>>().join("/");
        data.update(old_text(f).as_bytes());
        data.update(b"\t");
        data.update(m.files[&rel].sha256.as_bytes());
        data.update(b"\n");
    }
    let (old_list, old_data) = (format!("{}:{}", old.len(), hex(list)), format!("{}:{}", old.len(), hex(data)));
    for d in [vec![shards.join("beta"), shards.join("alpha")], vec![shards.clone()]] {
        let src = ShardSource::from_dirs(&d).unwrap().shuffled(seed);
        let names: Vec<String> = src.files().iter().map(|f| old_text(f)).collect();
        assert_eq!(names, old.iter().map(|f| old_text(f)).collect::<Vec<_>>(), "order");
        assert_eq!(src.fingerprint(), old_list);
        assert_eq!(dataset_fingerprint(&src).unwrap(), old_data);
    }
    std::fs::remove_dir_all(&base).ok();
}

#[test]
fn empty_files_are_skipped_without_changing_the_walk_041_r13() {
    use uranium_data::data::{GameId, SourcePos, StepSource, scan_files};
    use uranium_data::split::{Selection, Subsample};
    let games: Vec<ShardGame> = (0..40).map(|i| game_len(1 + (i % 16) as usize, i + 1)).collect();
    let root = shard_dir("skip-empty", &games, 2);
    let mut cfg = tiny_bc(1, 0);
    cfg.data.subsample = Some(Subsample { fraction: 0.2, keep_sources: vec![], fraction_by_race: None, fraction_by_source: None });
    let plain = || cfg.train_source(std::slice::from_ref(&root), Selection::S1).unwrap();
    let scan = scan_files(plain().files()).unwrap();
    let skipping = || plain().skip_empty(&scan);
    let unshuffled = || ShardSource::new(&root).unwrap().select(Selection::S1).subsample(cfg.data.subsample.clone(), cfg.data.seed);
    let unshuffled_scan = scan_files(unshuffled().files()).unwrap();
    let walk = |mut s: ShardSource, from: Option<SourcePos>| -> Vec<(String, Option<GameId>, SourcePos)> {
        if let Some(pos) = from {
            s.seek(pos).unwrap();
        }
        let mut out = Vec::new();
        while let Some(g) = s.next_game().unwrap() {
            out.push((g.meta.replay_sha256, s.last_id(), s.position()));
        }
        out.push((String::new(), s.last_id(), s.position()));
        out
    };
    let want = walk(plain(), None);
    let empty: Vec<usize> = (0..scan.len()).filter(|&k| plain().file_games(k).unwrap().is_empty()).collect();
    assert!(want.len() > 3 && !empty.is_empty() && empty.len() < scan.len(), "{} games, {} of {} files empty", want.len() - 1, empty.len(), scan.len());
    assert_eq!(walk(skipping(), None), want, "same games, ids and positions");
    let mut seeks: Vec<SourcePos> = want.iter().map(|w| w.2).collect();
    seeks.extend(empty.iter().map(|&k| SourcePos { next: GameId { file: k, game: 0 } }));
    for pos in seeks {
        assert_eq!(walk(skipping(), Some(pos)), walk(plain(), Some(pos)), "after a seek to {pos:?}");
    }
    let files = plain().files().to_vec();
    for &k in &empty {
        std::fs::write(&files[k], b"not a shard").unwrap();
    }
    assert!(plain().file_games(empty[0]).is_err());
    assert_eq!(walk(skipping(), None), want, "the empty files are not opened");
    assert_eq!(walk(unshuffled().skip_empty(&unshuffled_scan).shuffled(cfg.data.seed), None), want, "the skipped set follows a later shuffle");
    let mut reselected = skipping().select(Selection::S1);
    let read_all = loop {
        match reselected.next_game() {
            Ok(Some(_)) => {}
            Ok(None) => break true,
            Err(_) => break false,
        }
    };
    assert!(!read_all, "a new selection drops the skipped set");
    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn file_ahead_from_env_041_r16() {
    unsafe { std::env::remove_var("URANIUM_BC_FILE_AHEAD") };
    assert_eq!((uranium_data::bc::file_ahead().unwrap(), uranium_data::bc::FILE_AHEAD_DEFAULT), (0, 0));
    for (v, want) in [("0", Some(0)), (" 2 ", Some(2)), ("", Some(uranium_data::bc::FILE_AHEAD_DEFAULT)), ("two", None), ("-1", None)] {
        unsafe { std::env::set_var("URANIUM_BC_FILE_AHEAD", v) };
        assert_eq!(uranium_data::bc::file_ahead().ok(), want, "{v:?}");
    }
    unsafe { std::env::remove_var("URANIUM_BC_FILE_AHEAD") };
}

#[test]
fn file_read_ahead_keeps_the_walk_041_r16() {
    use uranium_data::data::{GameId, SourcePos, StepSource, scan_files};
    use uranium_data::split::{Selection, Subsample};
    let games: Vec<ShardGame> = (0..40).map(|i| game_len(1 + (i % 16) as usize, i + 1)).collect();
    let root = shard_dir("file-ahead", &games, 2);
    let mut cfg = tiny_bc(1, 0);
    cfg.data.subsample = Some(Subsample { fraction: 0.5, keep_sources: vec![], fraction_by_race: None, fraction_by_source: None });
    let plain = || cfg.train_source(std::slice::from_ref(&root), Selection::S1).unwrap();
    let scan = scan_files(plain().files()).unwrap();
    let make = |skip: bool, depth: usize| if skip { plain().skip_empty(&scan).read_ahead(depth) } else { plain().read_ahead(depth) };
    let walk = |mut s: ShardSource, from: Option<SourcePos>| -> Vec<(String, Option<GameId>, SourcePos)> {
        if let Some(pos) = from {
            s.seek(pos).unwrap();
        }
        let mut out = Vec::new();
        loop {
            let (text, end) = match s.next_game() {
                Ok(Some(g)) => (g.meta.replay_sha256, false),
                Ok(None) => (String::new(), true),
                Err(e) => (format!("error: {e:#}"), false),
            };
            out.push((text, s.last_id(), s.position()));
            if end {
                return out;
            }
        }
    };
    let variants = [(false, 1), (false, 3), (true, 0), (true, 1), (true, 3)];
    let want = walk(make(false, 0), None);
    let used: std::collections::BTreeSet<usize> = want.iter().filter_map(|w| w.1).map(|id| id.file).collect();
    assert!(used.len() > 4 && used.len() < scan.len(), "{} of {} files hold games", used.len(), scan.len());
    for (skip, depth) in variants {
        assert_eq!(walk(make(skip, depth), None), want, "skip {skip}, depth {depth}");
        for w in &want {
            assert_eq!(walk(make(skip, depth), Some(w.2)), walk(make(false, 0), Some(w.2)), "skip {skip}, depth {depth}, after a seek to {:?}", w.2);
        }
    }
    let warm = |skip: bool, depth: usize, n: usize| {
        let mut s = make(skip, depth);
        for w in &want[..n] {
            assert_eq!(s.next_game().unwrap().map(|g| g.meta.replay_sha256).as_ref(), Some(&w.0));
        }
        s
    };
    for (skip, depth) in variants {
        for pos in [want[0].2, want[3].2, want[want.len() - 2].2] {
            assert_eq!(walk(warm(skip, depth, 3), Some(pos)), walk(warm(false, 0, 3), Some(pos)), "skip {skip}, depth {depth}, a live reader then a seek to {pos:?}");
        }
        assert_eq!(walk(warm(skip, depth, 3).skip_empty(&scan), None), want[3..], "skip {skip}, depth {depth}, a live reader then skip_empty");
        assert_eq!(walk(warm(skip, depth, 3).select(Selection::S1), None), want[3..], "skip {skip}, depth {depth}, a live reader then select");
        drop(warm(skip, depth, 3));
        let waiting = warm(skip, depth, 1);
        std::thread::sleep(std::time::Duration::from_millis(50));
        drop(waiting);
    }
    let middle = *used.iter().nth(used.len() / 2).unwrap();
    std::fs::write(&plain().files()[middle], b"not a shard").unwrap();
    let broken = walk(make(false, 0), None);
    assert_eq!(broken.iter().filter(|w| w.0.starts_with("error: ")).count(), 1, "one failed call");
    assert!(!broken.last().unwrap().0.starts_with("error: "), "the walk goes on after the failed file");
    for (skip, depth) in variants {
        assert_eq!(walk(make(skip, depth), None), broken, "the same error at the same call: skip {skip}, depth {depth}");
    }
    std::fs::remove_dir_all(&root).ok();
}
