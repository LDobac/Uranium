//! 044-R3 policy labelling, data side: head choices become 040 labels the BC label check reads
//! back as the same head indices (044-AC2), the 040-R19 filter after labelling, the records'
//! `label_policy`, the per-view seed and the argument rules. The model side and the round trip
//! through the BC loader are tests/python/test_distill_label.py.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use uranium_data::bc::BcConfig;
use uranium_data::data::{GameId, MapStore, prepare};
use uranium_data::distill::{DistillLabeller, check_args, game_seed};
use uranium_data::testkit;
use uranium_env::features::augment::Sym;
use uranium_env::features::extract::action_table;
use uranium_env::features::label::{HeadChoice, bucket_index, host_label_check, label_of_choice};
use uranium_env::features::{DELAY_BUCKETS, MAP};
use uranium_replay::shard::{self, Shard, ShardGame, ShardHeader};

const TINY: &str = "lanes = 2\nseq_len = 4\nmicro_lanes = 1\npad_entities = 0\nlr = 3e-3\nmin_lr = 3e-3\nwarmup_steps = 1\ntotal_steps = 100\nweight_decay = 0.0\ngrad_clip = 1.0\ns2_lr_scale = 0.1\n\n[heads]\naction = 1.0\ndelay = 0.5\nqueued = 0.2\nselected = 1.0\ntarget_unit = 1.0\ntarget_loc = 1.0\n\n[value]\nwin = 0.5\n\n[data]\nz_drop = 0.0\naugment = true\nseed = 1\ncheckpoint_every = 1000\ncheckpoint_minutes = 0\nenemy_race_mask = 0.5\n";

fn tmp(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("uranium-distill-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn tiny_cfg(dir: &Path) -> BcConfig {
    let p = dir.join("bc_tiny.toml");
    std::fs::write(&p, TINY).unwrap();
    BcConfig::load(&p).unwrap()
}

fn id(name: &str) -> u16 {
    action_table().by_name(name).unwrap().id
}

/// A `distill_pre` game of `len` steps with its own replay hash.
fn pre_game(len: usize, tag: u32) -> ShardGame {
    let mut g = testkit::distill_games().into_iter().find(|g| g.steps.len() == 16).unwrap();
    g.steps.truncate(len);
    g.meta.replay_sha256 = format!("{tag:064x}");
    g
}

fn write_input(dir: &Path, games: Vec<ShardGame>) {
    std::fs::create_dir_all(dir).unwrap();
    shard::write(&dir.join("shard_00000.bin.zst"), &Shard { header: ShardHeader::default(), games }).unwrap();
}

/// 044-R3 ③ / 044-AC2: a choice's label reads back through the BC label check (the loader's
/// `host_label_check`) as the same head indices: action, delay bucket (its lower bound maps back),
/// queued only where the action uses it, the selection (canonical order), the target unit and the
/// location cell only where the action uses them.
#[test]
fn label_of_choice_reads_back_as_the_same_heads_044_r3() {
    let g = pre_game(16, 1);
    let seq = prepare(&g, false, Sym::IDENTITY, &MapStore::synthetic()).unwrap();
    let mut checked = 0;
    for (k, s) in seq.steps.iter().enumerate() {
        let o = &s.obs;
        let at = |unit: u32| o.entity_ids.iter().position(|&u| u == unit);
        let (scv, mineral) = (at(2).unwrap(), at(3).unwrap());
        let cell = 12 * MAP + 12;
        let cases = [
            HeadChoice { action: id("Move_Pos"), delay_bucket: k % DELAY_BUCKETS.len(), queued: true, selected: vec![scv], target_unit: mineral, target_loc: cell },
            HeadChoice { action: id("Gather_Unit"), delay_bucket: 15 - k % 16, queued: false, selected: vec![scv], target_unit: mineral, target_loc: cell },
            HeadChoice { action: id("NoOp"), delay_bucket: 3, queued: true, selected: vec![scv], target_unit: mineral, target_loc: cell },
        ];
        for c in cases {
            let l = label_of_choice(s.frame, &c, o);
            assert_eq!(l.frame, g.steps[k].raw_frame.frame);
            assert_eq!(bucket_index(l.delay), c.delay_bucket);
            let a = &action_table().actions[c.action as usize];
            let (h, why) = host_label_check(&l, o);
            assert_eq!(why, None, "{} at step {k}", a.name);
            assert_eq!((h.action, h.delay_bucket), (c.action, c.delay_bucket));
            assert_eq!(h.queued, c.queued && a.uses_queued);
            assert_eq!(h.selected, if a.uses_select { c.selected.clone() } else { vec![] });
            assert_eq!(h.target_unit, (a.target == uranium_env::actions::TargetKind::Unit).then_some(c.target_unit));
            assert_eq!(h.target_loc, (a.target == uranium_env::actions::TargetKind::Loc).then_some(c.target_loc));
            assert_eq!(l.target_tile.is_some(), a.target == uranium_env::actions::TargetKind::Loc);
            checked += 1;
        }
    }
    assert_eq!(checked, 48);
    // Every delay bucket's lower bound is that bucket (031-R11).
    for (b, &d) in DELAY_BUCKETS.iter().enumerate() {
        assert_eq!(bucket_index(d), b);
    }
}

/// The labelling pass over two views (10 steps each): view A gets one type-masked action (10%,
/// kept, weight 0 there), view B two (20%, the boundary: filtered, 040-R19 after labelling). The
/// output holds A only, with every label replaced and `raw_frame` / `privileged` /
/// `memory_action` / meta unchanged (044-R2 contract); the records have a line per view with the
/// 040-R19 counts and `label_policy` (the caller's fields plus the view's seed and invalid count).
/// The BC loader reads A back with the labels' weights.
#[test]
fn labelling_pass_filters_after_labelling_and_records_the_policy_040_r19_044_r3() {
    let d = tmp("pass");
    let cfg = tiny_cfg(&d);
    let (a, b) = (pre_game(10, 101), pre_game(10, 102));
    let input = d.join("pre");
    write_input(&input, vec![a.clone(), b.clone()]);
    let mut lab = DistillLabeller::new(&input, &cfg, Arc::new(MapStore::synthetic()), 7, None).unwrap();
    let mut done: HashMap<GameId, usize> = HashMap::new();
    let mut seeds = HashMap::new();
    while let Some(batch) = lab.next().unwrap() {
        for (lane, info) in lab.lanes(&batch).into_iter().enumerate() {
            let Some((gid, seed)) = info else {
                assert!(batch.chunks[lane].is_empty());
                continue;
            };
            seeds.insert(batch.game_ids[lane].clone(), seed);
            let pos = done.entry(gid).or_default();
            let bad = if batch.game_ids[lane] == a.meta.replay_sha256 { 1 } else { 2 };
            let choices: Vec<HeadChoice> = (0..batch.chunks[lane].len())
                .map(|k| {
                    let action = if *pos + k < bad { id("Train_Marine") } else { id("NoOp") };
                    HeadChoice { action, delay_bucket: 7, queued: false, selected: vec![], target_unit: 0, target_loc: 0 }
                })
                .collect();
            *pos += choices.len();
            lab.put(&batch, lane, &choices).unwrap();
        }
    }
    assert_eq!(lab.games(), 2);
    assert_eq!(seeds[&a.meta.replay_sha256], game_seed(&a.meta.replay_sha256, 0, 7));
    let (out, records) = (d.join("out"), d.join("rec").join("label_d1.jsonl"));
    let policy = serde_json::json!({ "checkpoint": "ck", "mode": "sample", "label_seed": 7 });
    let sum = lab.write(&out, &records, &policy, &d.join("staging"), Some(2)).unwrap();
    assert_eq!((sum.views, sum.written, sum.filtered, sum.steps_written, sum.invalid_labels, sum.files_written), (2, 1, 1, 10, 3, 1));

    let lines: Vec<serde_json::Value> = std::fs::read_to_string(&records).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    assert_eq!(lines.len(), 2);
    let (ra, rb) = (&lines[0], &lines[1]);
    assert_eq!((ra["status"].as_str(), ra["invalid"].as_u64(), ra["labeled"].as_u64()), (Some("written"), Some(1), Some(10)));
    assert_eq!(ra["by_reason"]["type_mask"], 1);
    assert_eq!(ra["shard"], "shard_00000.bin.zst");
    assert_eq!(ra["label_policy"]["game_seed"], game_seed(&a.meta.replay_sha256, 0, 7));
    assert_eq!((ra["label_policy"]["checkpoint"].as_str(), ra["label_policy"]["mode"].as_str(), ra["label_policy"]["invalid_labels"].as_u64()), (Some("ck"), Some("sample"), Some(1)));
    assert_eq!(rb["status"], "filtered");
    let reason = rb["reason"].as_str().unwrap();
    assert!(reason.starts_with("invalid_labels>=20%") && reason.contains("2/10") && reason.ends_with("(040-R19)"), "{reason}");
    assert!(rb.get("shard").is_none());

    let sh = shard::read(&out.join("shard_00000.bin.zst")).unwrap();
    assert_eq!(sh.header.schema_version, shard::SCHEMA_VERSION);
    assert_eq!(sh.games.len(), 1);
    let got = &sh.games[0];
    assert_eq!(got.meta, a.meta);
    assert_eq!(got.game_info, a.game_info);
    for (k, (s, s0)) in got.steps.iter().zip(&a.steps).enumerate() {
        assert_eq!((&s.raw_frame, &s.privileged, &s.memory_action), (&s0.raw_frame, &s0.privileged, &s0.memory_action));
        assert_eq!(s.action_label.frame, s0.raw_frame.frame);
        assert_eq!(s.action_label.action, if k < 1 { id("Train_Marine") } else { id("NoOp") });
        assert_eq!(s.action_label.delay, DELAY_BUCKETS[7]);
    }
    // Read back by the BC loader: the labels, weight 0 on the invalid one.
    let seq = prepare(got, false, Sym::IDENTITY, &MapStore::synthetic()).unwrap();
    assert_eq!(seq.invalid_labels, 1);
    assert_eq!(seq.steps.iter().map(|s| s.weight).sum::<f32>(), 9.0);
    assert!(seq.steps.iter().skip(1).all(|s| s.label.action == id("NoOp") && s.label.delay_bucket == 7));
    let _ = std::fs::remove_dir_all(&d);
}

/// A view without a label for every step is an error, never a shard with `unknown` slots left.
#[test]
fn a_view_without_all_labels_is_an_error_044_r3() {
    let d = tmp("missing");
    let cfg = tiny_cfg(&d);
    let input = d.join("pre");
    write_input(&input, vec![pre_game(6, 201)]);
    let mut lab = DistillLabeller::new(&input, &cfg, Arc::new(MapStore::synthetic()), 1, None).unwrap();
    let first = lab.next().unwrap().unwrap();
    let lane = first.chunks.iter().position(|c| !c.is_empty()).unwrap();
    let noop = HeadChoice { action: id("NoOp"), delay_bucket: 0, queued: false, selected: vec![], target_unit: 0, target_loc: 0 };
    assert!(lab.put(&first, lane, std::slice::from_ref(&noop)).is_err(), "one choice per step");
    lab.put(&first, lane, &vec![noop; first.chunks[lane].len()]).unwrap();
    let e = lab.write(&d.join("out"), &d.join("label.jsonl"), &serde_json::json!({}), &d.join("staging"), None).unwrap_err();
    assert!(format!("{e:#}").contains("4 labels for 6 steps"), "{e:#}");
    let _ = std::fs::remove_dir_all(&d);
}

/// Inputs that are not `distill_pre` games are refused (044-R2 contract: source `distill`,
/// `memory_action` on every step).
#[test]
fn only_distill_pre_games_are_labelled_044_r2() {
    let d = tmp("pre");
    let cfg = tiny_cfg(&d);
    for (tag, edit) in [("src", 0), ("memory", 1)] {
        let input = d.join(tag);
        let mut g = pre_game(5, 301);
        match edit {
            0 => g.meta.src = "dagger".into(),
            _ => g.steps[2].memory_action = None,
        }
        write_input(&input, vec![g]);
        let mut lab = DistillLabeller::new(&input, &cfg, Arc::new(MapStore::synthetic()), 1, None).unwrap();
        let e = lab.next().err().expect("refused");
        let text = format!("{e:#}");
        assert!(text.contains(if edit == 0 { "source \"dagger\"" } else { "without memory_action" }), "{text}");
    }
    let _ = std::fs::remove_dir_all(&d);
}

/// 044-R3 argument rules, and the continuation of a records file (044-R5 ③ top-up: later input
/// files of the round labelled into the same output folder).
#[test]
fn argument_rules_044_r3() {
    let d = tmp("args");
    let input = d.join("pre");
    write_input(&input, vec![pre_game(3, 401)]);
    let root = d.join("shared").join("datasets").join("replays").join("shards");
    let out = root.join("distill_m2").join("d1");
    let rec = d.join("label_d1.jsonl");
    assert_eq!(check_args(&input, &out, &rec, "sample", Some(&root)).unwrap(), ["shard_00000.bin.zst"]);
    let err = |r: anyhow::Result<Vec<String>>| format!("{:#}", r.unwrap_err());
    assert!(err(check_args(&input, &out, &rec, "softmax", Some(&root))).contains("--mode"));
    assert!(err(check_args(&input, &out, &rec, "argmax", Some(&root))).contains("never training data"));
    check_args(&input, &d.join("argmax_out"), &rec, "argmax", Some(&root)).unwrap();
    assert!(err(check_args(&input, &out, &d.join("distill_d1.jsonl"), "sample", Some(&root))).contains("resume scan"));
    assert!(err(check_args(&input, &input.join("x"), &rec, "sample", None)).contains("overlap"));
    assert!(err(check_args(&d.join("empty"), &out, &rec, "sample", None)).contains("--in"));
    write_input(&out, vec![pre_game(3, 402)]);
    assert!(err(check_args(&input, &out, &rec, "sample", None)).contains("already holds"));
    // Continuation: the records name shard_00000 (its output is there); a later input file is the
    // one left to label, unless its output already exists (published before its records lines).
    std::fs::write(&rec, "{\"input\":\"shard_00000.bin.zst\",\"label_policy\":{\"label_seed\":3}}\n").unwrap();
    assert!(err(check_args(&input, &out, &rec, "sample", None)).contains("nothing to label"));
    shard::write(&input.join("shard_00001.bin.zst"), &Shard { header: ShardHeader::default(), games: vec![pre_game(4, 403)] }).unwrap();
    assert_eq!(check_args(&input, &out, &rec, "sample", None).unwrap(), ["shard_00001.bin.zst"]);
    let l = uranium_data::distill::read_labelled(&rec).unwrap();
    assert_eq!((l.views, l.policy.unwrap()["label_seed"].as_u64()), (1, Some(3)));
    std::fs::copy(out.join("shard_00000.bin.zst"), out.join("shard_00001.bin.zst")).unwrap();
    assert!(err(check_args(&input, &out, &rec, "sample", None)).contains("stopped after publishing"));
    std::fs::write(&rec, "{\"input\":\"shard_00000.bin.zst\"").unwrap();
    assert!(err(check_args(&input, &out, &rec, "sample", None)).contains("not a records line"));
    let _ = std::fs::remove_dir_all(&d);
}

/// 044-R5 ③ top-up: a second pass over the round folder labels only the new input file and
/// appends its records; the first output file is left as it was.
#[test]
fn a_second_pass_labels_only_the_new_input_files_044_r3() {
    let d = tmp("topup");
    let cfg = tiny_cfg(&d);
    let input = d.join("pre");
    write_input(&input, vec![pre_game(5, 501)]);
    let (out, rec) = (d.join("out"), d.join("label_d1.jsonl"));
    let noop = HeadChoice { action: id("NoOp"), delay_bucket: 2, queued: false, selected: vec![], target_unit: 0, target_loc: 0 };
    let pass = |only: Vec<String>| {
        let mut lab = DistillLabeller::new(&input, &cfg, Arc::new(MapStore::synthetic()), 9, Some(&only)).unwrap();
        while let Some(b) = lab.next().unwrap() {
            for lane in 0..b.chunks.len() {
                lab.put(&b, lane, &vec![noop.clone(); b.chunks[lane].len()]).unwrap();
            }
        }
        lab.write(&out, &rec, &serde_json::json!({ "label_seed": 9 }), &d.join("staging"), None).unwrap()
    };
    let first = pass(check_args(&input, &out, &rec, "sample", None).unwrap());
    assert_eq!((first.views, first.files_written), (1, 1));
    let before = std::fs::read(out.join("shard_00000.bin.zst")).unwrap();
    shard::write(&input.join("shard_00001.bin.zst"), &Shard { header: ShardHeader::default(), games: vec![pre_game(6, 502), pre_game(2, 503)] }).unwrap();
    let todo = check_args(&input, &out, &rec, "sample", None).unwrap();
    assert_eq!(todo, ["shard_00001.bin.zst"]);
    let second = pass(todo);
    assert_eq!((second.views, second.written, second.steps_written), (2, 2, 8));
    assert_eq!(std::fs::read(out.join("shard_00000.bin.zst")).unwrap(), before);
    assert_eq!(shard::read(&out.join("shard_00001.bin.zst")).unwrap().games.len(), 2);
    let inputs: Vec<String> = std::fs::read_to_string(&rec).unwrap().lines().map(|l| serde_json::from_str::<serde_json::Value>(l).unwrap()["input"].as_str().unwrap().to_string()).collect();
    assert_eq!(inputs, ["shard_00000.bin.zst", "shard_00001.bin.zst", "shard_00001.bin.zst"]);
    let _ = std::fs::remove_dir_all(&d);
}

/// 044-R3 ②: the per-view seed is a hash of (replay hash, perspective, label seed).
#[test]
fn game_seed_is_per_view_and_label_seed_044_r3() {
    let s = game_seed("ab", 0, 5);
    assert_eq!(s, game_seed("ab", 0, 5));
    assert!(s < 1 << 63);
    assert_ne!(s, game_seed("ab", 1, 5));
    assert_ne!(s, game_seed("ac", 0, 5));
    assert_ne!(s, game_seed("ab", 0, 6));
}
