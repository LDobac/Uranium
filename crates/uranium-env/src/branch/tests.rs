//! 051-AC1 (CPU unit tests): act-log round trips, observation digests, snapshot rules, state
//! files, interventions, the forced-prefix check and the record formats.

use uranium_proto::{BitGrid, CancelKind, CmdOrder, Command, PlayerState, Target};

use super::*;
use crate::features::extract::action_table;

fn id(name: &str) -> u16 {
    action_table().by_name(name).unwrap().id
}

/// Every `Command` and `CmdOrder` variant with every argument set (051-AC1: "010 `Command` 모든
/// 종류의 인자"). The exhaustive matches fail to compile when a variant is added.
fn all_commands() -> Vec<Command> {
    let orders = vec![
        CmdOrder::Move(Target::Unit(7)),
        CmdOrder::Move(Target::Pos((1, 65_535))),
        CmdOrder::Attack(Target::Unit(u32::MAX)),
        CmdOrder::Attack(Target::Pos((300, 400))),
        CmdOrder::Patrol((5, 6)),
        CmdOrder::Hold,
        CmdOrder::Stop,
        CmdOrder::RightClick(Target::Unit(9)),
        CmdOrder::RightClick(Target::Pos((10, 11))),
        CmdOrder::Siege,
        CmdOrder::Unsiege,
        CmdOrder::Cloak,
        CmdOrder::Decloak,
        CmdOrder::Burrow,
        CmdOrder::Unburrow,
        CmdOrder::Lift,
        CmdOrder::Land((12, 13)),
        CmdOrder::Load(14),
        CmdOrder::UnloadAll(None),
        CmdOrder::UnloadAll(Some((15, 16))),
        CmdOrder::Repair(17),
        CmdOrder::Gather(18),
        CmdOrder::ReturnCargo,
        CmdOrder::UseTech { tech: 19, target: None },
        CmdOrder::UseTech { tech: 20, target: Some(Target::Unit(21)) },
        CmdOrder::UseTech { tech: 22, target: Some(Target::Pos((23, 24))) },
        CmdOrder::Unload(25),
    ];
    for o in &orders {
        match o {
            CmdOrder::Move(_)
            | CmdOrder::Attack(_)
            | CmdOrder::Patrol(_)
            | CmdOrder::Hold
            | CmdOrder::Stop
            | CmdOrder::RightClick(_)
            | CmdOrder::Siege
            | CmdOrder::Unsiege
            | CmdOrder::Cloak
            | CmdOrder::Decloak
            | CmdOrder::Burrow
            | CmdOrder::Unburrow
            | CmdOrder::Lift
            | CmdOrder::Land(_)
            | CmdOrder::Load(_)
            | CmdOrder::UnloadAll(_)
            | CmdOrder::Repair(_)
            | CmdOrder::Gather(_)
            | CmdOrder::ReturnCargo
            | CmdOrder::UseTech { .. }
            | CmdOrder::Unload(_) => {}
        }
    }
    let mut cmds: Vec<Command> = orders.into_iter().enumerate().map(|(k, order)| Command::Unit { units: vec![100 + k as u32, 4_000_000_000], order, queued: !k.is_multiple_of(2) }).collect();
    let kinds = [CancelKind::Construction, CancelKind::Train, CancelKind::Research, CancelKind::Upgrade, CancelKind::Addon, CancelKind::Morph];
    cmds.extend([
        Command::Build { worker: 30, unit_type: 125, tile: (127, 0) },
        Command::BuildAddon { building: 31, unit_type: 120 },
        Command::Train { building: 32, unit_type: 0 },
        Command::Morph { unit: 33, unit_type: 37 },
        Command::Research { building: 34, tech: 5 },
        Command::Upgrade { building: 35, upgrade: 62 },
        Command::SetRally { building: 36, target: Target::Unit(37) },
        Command::SetRally { building: 38, target: Target::Pos((39, 40)) },
    ]);
    cmds.extend(kinds.into_iter().map(|kind| Command::Cancel { unit: 41, kind }));
    for c in &cmds {
        match c {
            Command::Unit { .. }
            | Command::Build { .. }
            | Command::BuildAddon { .. }
            | Command::Train { .. }
            | Command::Morph { .. }
            | Command::Research { .. }
            | Command::Upgrade { .. }
            | Command::Cancel { .. }
            | Command::SetRally { .. } => {}
        }
    }
    cmds
}

fn act(cmds: Vec<Command>, advance: u32) -> Act {
    Act { cmds, advance_frames: advance, valid_until_frame: None, surrender: None }
}

/// The bytes the agent sends for an `Act` (010-R1 framed postcard).
fn enc(a: &Act) -> Vec<u8> {
    let mut b = Vec::new();
    write_msg(&mut b, a).unwrap();
    b
}

fn model_line(i: u32, frame: u32, digest: u64, a: Act) -> ObsLine {
    ObsLine {
        i,
        frame,
        digest,
        act: a,
        decision: Some(Decision { action: 37, delay_bucket: 5, units: vec![1203, 7], rejected: i.is_multiple_of(3) }),
        state: None,
        state_sha256: None,
        forced: false,
    }
}

fn end(obs: u64, decisions: u64) -> ActLogEnd {
    ActLogEnd {
        obs,
        decisions,
        end_frame: 11_568,
        reason: EndReason::TimeLimitDraw,
        player: 0,
        players_index: 0,
        policy: "model:ckpt-39096@abcdef:stochastic".into(),
        git: "d45cb8c".into(),
        feature_schema: 7,
        action_schema: 2,
    }
}

/// A model log with every command kind, a surrender, a realtime deadline and a saved state.
fn model_log() -> ActLog {
    let mut lines = vec![];
    for (k, c) in all_commands().into_iter().enumerate() {
        lines.push(model_line(k as u32, 8 * k as u32, 0x9f1c_0000_0000_0000 | k as u64, act(vec![c], 8)));
    }
    let n = lines.len() as u32;
    let mut l = model_line(n, 8 * n, u64::MAX, act(vec![], 32));
    l.act.valid_until_frame = Some(12_345);
    l.state = Some(state_id(7, n));
    l.state_sha256 = Some(sha256_hex(b"state"));
    l.forced = true;
    lines.push(l);
    lines.push(model_line(n + 1, 8 * n + 40, 0, Act { cmds: vec![], advance_frames: 1, valid_until_frame: None, surrender: Some("gg".into()) }));
    let e = end(lines.len() as u64, lines.len() as u64);
    ActLog { lines, end: e }
}

fn to_text(log: &ActLog) -> String {
    let mut buf = Vec::new();
    log.write(&mut buf).unwrap();
    String::from_utf8(buf).unwrap()
}

/// 051-AC1 / 051-R2: write → read gives the same log, and every `Act` encodes to the same bytes
/// (a forced prefix sends them byte for byte, 051-R5 ③).
#[test]
fn branch_act_log_round_trip_every_command_051_ac1() {
    let log = model_log();
    let text = to_text(&log);
    let back = ActLog::read(text.as_bytes()).unwrap();
    assert_eq!(back, log);
    for (a, b) in log.lines.iter().zip(&back.lines) {
        assert_eq!(enc(&a.act), enc(&b.act));
    }
    // Integers stay integers, the digest is hex text, a model line without a state says null.
    let first = text.lines().next().unwrap();
    assert!(first.starts_with(r#"{"i":0,"frame":0,"digest":"9f1c000000000000","act":{"cmds":[{"Unit":{"units":[100,4000000000]"#), "{first}");
    assert!(first.contains(r#""decision":{"action":37,"delay_bucket":5,"units":[1203,7],"rejected":true},"state":null,"forced":false}"#), "{first}");
    assert!(text.contains(r#""state":"g7-o"#) && text.contains(r#""forced":true"#));
    let last = text.lines().last().unwrap();
    assert!(last.starts_with(r#"{"end":{"obs":"#) && last.contains(r#""reason":"TimeLimitDraw""#), "{last}");
}

/// 051-R2: T0 lines carry neither `decision` nor `state`; the end line counts 0 decisions.
#[test]
fn branch_act_log_t0_lines_have_no_decision_051_r2() {
    let mut buf = Vec::new();
    let mut w = ActLogWriter::new(&mut buf);
    for i in 0..3 {
        w.push(&ObsLine { i, frame: 10 * i, digest: i as u64, act: act(vec![], 10), decision: None, state: None, state_sha256: None, forced: false }).unwrap();
    }
    // The writer counts the lines itself.
    w.finish(end(99, 99)).unwrap();
    let text = String::from_utf8(buf).unwrap();
    assert!(text.lines().take(3).all(|l| !l.contains("decision") && !l.contains("\"state")), "{text}");
    let log = ActLog::read(text.as_bytes()).unwrap();
    assert_eq!((log.end.obs, log.end.decisions), (3, 0));
}

/// 051-R2 / 051-AC1: incomplete logs are refused.
#[test]
fn branch_act_log_without_end_is_refused_051_ac1() {
    let text = to_text(&model_log());
    let lines: Vec<&str> = text.lines().collect();
    let join = |v: &[&str]| v.join("\n");
    let n = lines.len();
    // No end line.
    let err = ActLog::read(join(&lines[..n - 1]).as_bytes()).unwrap_err();
    assert!(err.to_string().contains("without its end line"), "{err}");
    // A line after the end line.
    let mut after = lines.clone();
    after.push(lines[0]);
    assert!(ActLog::read(join(&after).as_bytes()).is_err());
    // End line counts other observations (a line dropped from the middle also breaks the index).
    let mut dropped = lines.clone();
    dropped.remove(n - 2);
    assert!(ActLog::read(join(&dropped).as_bytes()).is_err());
    let mut short = lines[..n - 2].to_vec();
    short.push(lines[n - 1]);
    assert!(ActLog::read(join(&short).as_bytes()).unwrap_err().to_string().contains("observations"));
    // Decision count differs.
    let mut log = model_log();
    log.end.decisions -= 1;
    let mut buf = Vec::new();
    for l in &log.lines {
        serde_json::to_writer(&mut buf, l).unwrap();
        buf.push(b'\n');
    }
    serde_json::to_writer(&mut buf, &EndWrap { end: &log.end }).unwrap();
    assert!(ActLog::read(&buf[..]).unwrap_err().to_string().contains("decisions"));
    // Indices must count from 0; the writer refuses a gap too.
    let mut w = ActLogWriter::new(Vec::new());
    assert!(w.push(&model_line(1, 0, 0, act(vec![], 1))).is_err());
    // Bad digest text.
    let bad = text.replacen("9f1c000000000000", "9f1c", 1);
    assert!(ActLog::read(bad.as_bytes()).is_err());
    // 051-R3: a state without its sha256 (or the reverse) is refused on write and on read.
    let mut l = model_line(0, 0, 0, act(vec![], 1));
    l.state = Some("g0-o0".into());
    assert!(ActLogWriter::new(Vec::new()).push(&l).is_err());
    l.state = None;
    l.state_sha256 = Some(sha256_hex(b"s"));
    assert!(ActLogWriter::new(Vec::new()).push(&l).is_err());
    let no_sha = text.replacen(r#","state_sha256":""#, r#","x":""#, 1);
    assert_ne!(no_sha, text);
    assert!(format!("{:#}", ActLog::read(no_sha.as_bytes()).unwrap_err()).contains("state_sha256"));
    // The end frame cannot come before the last observation.
    let mut log = model_log();
    log.end.end_frame = log.lines.last().unwrap().frame - 1;
    let mut buf = Vec::new();
    log.write(&mut buf).unwrap();
    assert!(ActLog::read(&buf[..]).unwrap_err().to_string().contains("end frame"));
}

fn raw_frame(frame: u32, minerals: i32) -> RawFrame {
    RawFrame {
        frame,
        self_state: PlayerState { minerals, ..Default::default() },
        units: vec![],
        visibility: BitGrid::new(4, 4),
        explored: BitGrid::new(4, 4),
        creep: BitGrid::new(4, 4),
        events: vec![],
        effects: vec![],
    }
}

/// 051-AC1 / 051-R2: the observation digest eats the session `frame_digest` bytes (framed
/// postcard, 010-R1) and restarts from the FNV offset basis for every observation.
#[test]
fn branch_obs_digest_restarts_and_eats_session_bytes_051_ac1() {
    // FNV-1a 64 reference vector.
    assert_eq!(fnv1a(FNV_OFFSET, b"a"), 0xaf63_dc4c_8601_ec8c);
    let (f1, f2) = (raw_frame(0, 50), raw_frame(8, 58));
    let bytes = |f: &RawFrame| {
        let mut b = Vec::new();
        write_msg(&mut b, f).unwrap();
        b
    };
    assert_eq!(frame_bytes(&f1).unwrap(), bytes(&f1));
    // The session digest after one observation is that observation's digest ...
    let session1 = fnv1a(FNV_OFFSET, &bytes(&f1));
    assert_eq!(obs_digest(&f1).unwrap(), session1);
    // ... but the second observation's digest does not continue the session fold.
    let session2 = fnv1a(session1, &bytes(&f2));
    assert_eq!(obs_digest(&f2).unwrap(), fnv1a(FNV_OFFSET, &bytes(&f2)));
    assert_ne!(obs_digest(&f2).unwrap(), session2);
    assert_eq!(obs_digest_bytes(&bytes(&f2)), obs_digest(&f2).unwrap());
    // Same observation, same digest; any change, another digest.
    assert_eq!(obs_digest(&raw_frame(8, 58)).unwrap(), obs_digest(&f2).unwrap());
    assert_ne!(obs_digest(&raw_frame(8, 59)).unwrap(), obs_digest(&f2).unwrap());
    let d = obs_digest(&f2).unwrap();
    assert_eq!(parse_digest(&digest_hex(d)).unwrap(), d);
    assert_eq!(digest_hex(1), "0000000000000001");
}

fn mask_with(open: &[&str]) -> Vec<bool> {
    let mut m = vec![false; action_table().actions.len()];
    m[0] = true;
    for a in open {
        m[id(a) as usize] = true;
    }
    m
}

fn rule(name: &str, from: u32, to: u32, any_open: &[&str]) -> SnapshotRule {
    SnapshotRule { name: name.into(), from_frame: from, to_frame: to, any_open: any_open.iter().map(|s| s.to_string()).collect(), after_rule: None, after_s: None }
}

fn after(mut r: SnapshotRule, earlier: &str, secs: u32) -> SnapshotRule {
    r.after_rule = Some(earlier.into());
    r.after_s = Some(secs);
    r
}

/// 051-R3 / 051-R11 ① (user decision 2026-11-13): an `after_rule` rule that would fire on its
/// earlier rule's decision moves to the first matching decision at or after that frame + after_s
/// (window end kept); not coincident = as without the key; never moved past the window = no point.
#[test]
fn branch_snapshot_after_rule_051_r3() {
    let t = action_table();
    let bunker = mask_with(&["Build_Bunker"]);
    let none = mask_with(&[]);
    let b1 = || [rule("b1-p1", 3240, 5760, &["Build_Bunker"]), after(rule("b1-p2", 3960, 5760, &["Build_Bunker"]), "b1-p1", 30)];
    // Coincident (first open decision after 2:45): p2 moves to >= 4001 + 720 = 4721.
    let mut s = Snapshots::new(&b1(), t).unwrap();
    assert!(s.check(0, 3300, &none).is_empty());
    assert_eq!(s.check(1, 4001, &bunker), vec!["b1-p1"]);
    assert!(s.check(2, 4500, &bunker).is_empty());
    assert!(s.check(3, 4720, &bunker).is_empty(), "before p1 + 30 s");
    assert!(s.check(4, 4721, &none).is_empty(), "its condition still holds");
    assert_eq!(s.check(5, 4730, &bunker), vec!["b1-p2"]);
    assert_eq!(s.fired().collect::<Vec<_>>(), vec![("b1-p1", Some(1)), ("b1-p2", Some(5))]);
    // Not coincident: p1 before 2:45, p2 at the first open decision from 2:45 on (even < p1 + 30 s).
    let mut s = Snapshots::new(&b1(), t).unwrap();
    assert_eq!(s.check(0, 3900, &bunker), vec!["b1-p1"]);
    assert_eq!(s.check(1, 3970, &bunker), vec!["b1-p2"]);
    // Moved past the window end: p2 never fires (the trunk is replaced, 051-R11 ①).
    let mut s = Snapshots::new(&b1(), t).unwrap();
    assert_eq!(s.check(0, 5100, &bunker), vec!["b1-p1"]);
    assert!(s.check(1, 5759, &bunker).is_empty());
    assert_eq!(s.fired().collect::<Vec<_>>(), vec![("b1-p1", Some(0)), ("b1-p2", None)]);
    // Key errors: a later or unknown rule, half the pair.
    assert!(Snapshots::new(&[after(rule("a", 0, 9, &[]), "b", 1), rule("b", 0, 9, &[])], t).is_err());
    let mut half = rule("b", 0, 9, &[]);
    half.after_s = Some(3);
    assert!(Snapshots::new(&[rule("a", 0, 9, &[]), half], t).is_err());
    // The plan form.
    let r: SnapshotRule = toml::from_str("name = \"b1-p2\"\nfrom_frame = 3960\nto_frame = 5760\nany_open = [\"Build_Bunker\"]\nafter_rule = \"b1-p1\"\nafter_s = 30").unwrap();
    assert_eq!(r, b1()[1]);
}

/// 051-AC1 / 051-R3: window [from, to), `any_open`, first matching decision only, two rules on
/// one decision share one state file.
#[test]
fn branch_snapshot_rules_051_ac1() {
    let t = action_table();
    let bunker = mask_with(&["Build_Bunker"]);
    let none = mask_with(&[]);
    // Start included, end excluded.
    let mut s = Snapshots::new(&[rule("p", 100, 200, &["Build_Bunker"])], t).unwrap();
    assert!(s.check(0, 99, &bunker).is_empty());
    assert_eq!(s.check(1, 100, &bunker), vec!["p"]);
    let mut s = Snapshots::new(&[rule("p", 100, 200, &["Build_Bunker"])], t).unwrap();
    assert!(s.check(0, 200, &bunker).is_empty(), "end frame excluded");
    // any_open: closed decisions are skipped; only the first open one fires.
    let mut s = Snapshots::new(&[rule("p", 100, 200, &["Build_Bunker", "Train_Marine"])], t).unwrap();
    assert!(s.check(0, 120, &none).is_empty());
    assert_eq!(s.check(1, 130, &mask_with(&["Train_Marine"])), vec!["p"]);
    assert!(s.check(2, 140, &bunker).is_empty(), "fires once");
    assert_eq!(s.fired().collect::<Vec<_>>(), vec![("p", Some(1))]);
    // No condition: the first decision in the window.
    let mut s = Snapshots::new(&[rule("any", 100, 200, &[])], t).unwrap();
    assert_eq!(s.check(5, 150, &none), vec!["any"]);
    // Two rules on the same decision: one check, both names (one state file).
    let rules = [rule("b1-p1", 3240, 5760, &["Build_Bunker"]), rule("b1-p2", 3960, 5760, &["Build_Bunker"])];
    let mut s = Snapshots::new(&rules, t).unwrap();
    assert!(s.check(0, 3300, &none).is_empty());
    assert_eq!(s.check(1, 4000, &bunker), vec!["b1-p1", "b1-p2"]);
    assert!(s.check(2, 4100, &bunker).is_empty());
    assert_eq!(state_id(7, 1), "g7-o1");
    // Rule errors.
    assert!(Snapshots::new(&[rule("p", 100, 100, &[])], t).is_err());
    assert!(Snapshots::new(&[rule("p", 0, 1, &["Build_Nothing"])], t).is_err());
    assert!(Snapshots::new(&[rule("p", 0, 1, &[]), rule("p", 0, 2, &[])], t).is_err());
    // The plan form (020-R22 ②).
    let r: SnapshotRule = toml::from_str("name = \"b1-p1\"\nfrom_frame = 3240\nto_frame = 5760\nany_open = [\"Build_Bunker\"]").unwrap();
    assert_eq!(r, rules[0]);
    assert!(toml::from_str::<SnapshotRule>("name = \"x\"\nfrom_frame = 1\nto_frame = 2\nextra = 1").is_err());
}

/// 051-AC1 / 051-R3: state files round-trip bit for bit; the sha256 is of the content.
#[test]
fn branch_state_file_round_trip_bits_051_ac1() {
    let odd = [f32::from_bits(0x7fc0_1234), -0.0, f32::from_bits(1), f32::INFINITY, f32::NEG_INFINITY, f32::MIN_POSITIVE, 1.5, -3.25];
    let parts = vec![(vec![2, 4], odd.to_vec()), (vec![1, 0, 3], vec![]), (vec![1], vec![42.0]), (vec![3], vec![0.1, 0.2, 0.3])];
    let bytes = encode_state(&parts).unwrap();
    let back = decode_state(&bytes).unwrap();
    assert_eq!(back.len(), parts.len());
    for ((s0, d0), (s1, d1)) in parts.iter().zip(&back) {
        assert_eq!(s0, s1);
        assert_eq!(d0.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), d1.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
    }
    assert_eq!(encode_state(&back).unwrap(), bytes);
    assert_eq!(sha256_hex(b"abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    assert_eq!(sha256_hex(&bytes), sha256_hex(&encode_state(&back).unwrap()));
    // Damaged files are refused.
    assert!(decode_state(&bytes[..bytes.len() - 1]).is_err());
    let mut trailing = bytes.clone();
    trailing.push(0);
    assert!(decode_state(&trailing).is_err());
    let mut magic = bytes.clone();
    magic[0] = b'X';
    assert!(decode_state(&magic).is_err());
    assert!(encode_state(&[(vec![2, 2], vec![1.0])]).is_err(), "shape and values disagree");
    // Rank-0 parts are refused both ways.
    assert!(encode_state(&[(vec![], vec![1.0])]).is_err());
    let mut rank0 = b"URST".to_vec();
    for w in [1u32, 1, 0, 1.0f32.to_bits()] {
        rank0.extend_from_slice(&w.to_le_bytes());
    }
    assert!(decode_state(&rank0).unwrap_err().to_string().contains("rank 0"));
}

fn alt(rules: &[(Op, &str, u32, u32)]) -> Alternative {
    Alternative { name: "alt".into(), rules: rules.iter().map(|&(op, a, from_s, to_s)| InterventionRule { op, action: a.into(), from_s, to_s }).collect() }
}

fn open(m: &[bool]) -> Vec<String> {
    open_ids(m).into_iter().map(|i| action_table().actions[i as usize].name.clone()).collect()
}

/// 051-AC1 / 051-R7 ①: a force narrows the first decision in its window where the action is
/// open to {action} (NoOp closed too), and only that decision.
#[test]
fn branch_force_first_open_decision_only_051_ac1() {
    let t = action_table();
    let f_t = 1000;
    let mut iv = Intervention::new(&alt(&[(Op::Force, "Build_Bunker", 0, 10)]), t, f_t).unwrap();
    let both = mask_with(&["Build_Bunker", "Train_Marine"]);
    // Closed at t: nothing changes, the force waits.
    assert_eq!(iv.apply(f_t, &mask_with(&["Train_Marine"])), mask_with(&["Train_Marine"]));
    assert_eq!(open(&iv.apply(f_t + 8, &both)), vec!["Build_Bunker"]);
    iv.note_executor(f_t + 8, true);
    // Afterwards the mask is left alone.
    assert_eq!(iv.apply(f_t + 16, &both), both);
    let log = &iv.log()[0];
    assert_eq!((log.in_window, log.applied, log.force_frame, log.missed, log.rejected), (2, 1, Some(f_t + 8), false, Some(true)));
    // The t decision only (0/0): forced at t when open.
    let mut iv = Intervention::new(&alt(&[(Op::Force, "Build_Bunker", 0, 0)]), t, f_t).unwrap();
    assert_eq!(open(&iv.apply(f_t, &both)), vec!["Build_Bunker"]);
    assert_eq!(iv.apply(f_t + 1, &both), both);
    assert_eq!(iv.log()[0].force_frame, Some(f_t));
}

/// 051-AC1 / 051-R7 ①⑤: `missed` when the action never opens in the window.
#[test]
fn branch_force_missed_051_ac1() {
    let t = action_table();
    let f_t = 1000;
    let marine = mask_with(&["Train_Marine"]);
    // 0/0 and closed at t: missed at t itself, never forced later.
    let mut iv = Intervention::new(&alt(&[(Op::Force, "Build_Bunker", 0, 0)]), t, f_t).unwrap();
    assert_eq!(iv.apply(f_t, &marine), marine);
    assert!(iv.log()[0].missed, "missed at t");
    assert_eq!(iv.apply(f_t + 8, &mask_with(&["Build_Bunker"])), mask_with(&["Build_Bunker"]));
    let log = &iv.log()[0];
    assert!(log.missed && log.force_frame.is_none() && log.rejected.is_none());
    // A window ends without the action: missed at the first decision past it (end excluded).
    let mut iv = Intervention::new(&alt(&[(Op::Force, "Build_Bunker", 0, 1)]), t, f_t).unwrap();
    iv.apply(f_t, &marine);
    iv.apply(f_t + 23, &marine);
    assert!(!iv.log()[0].missed);
    assert_eq!(iv.apply(f_t + 24, &mask_with(&["Build_Bunker"])), mask_with(&["Build_Bunker"]), "window end excluded");
    assert!(iv.log()[0].missed);
    // The game ends inside the window.
    let mut iv = Intervention::new(&alt(&[(Op::Force, "Build_Bunker", 0, 60)]), t, f_t).unwrap();
    iv.apply(f_t, &marine);
    iv.finish();
    assert!(iv.log()[0].missed);
}

/// 051-AC1 / 051-R7 ②③: ban window boundaries, only open bits close, NoOp never closes under a
/// ban, a closed bit never opens.
#[test]
fn branch_ban_window_and_close_only_051_ac1() {
    let t = action_table();
    let f_t = 2000;
    let mut iv = Intervention::new(&alt(&[(Op::Ban, "Build_Bunker", 10, 90)]), t, f_t).unwrap();
    let m = mask_with(&["Build_Bunker", "Train_Marine"]);
    assert_eq!(iv.apply(f_t + 10 * 24 - 1, &m), m, "before the window");
    assert_eq!(open(&iv.apply(f_t + 10 * 24, &m)), vec!["NoOp", "Train_Marine"], "start included");
    assert_eq!(open(&iv.apply(f_t + 90 * 24 - 1, &m)), vec!["NoOp", "Train_Marine"]);
    assert_eq!(iv.apply(f_t + 90 * 24, &m), m, "end excluded");
    // Already closed: unchanged and not counted as applied.
    let closed = mask_with(&["Train_Marine"]);
    let mut iv = Intervention::new(&alt(&[(Op::Ban, "Build_Bunker", 0, 90)]), t, f_t).unwrap();
    assert_eq!(iv.apply(f_t, &closed), closed);
    assert_eq!(iv.apply(f_t + 8, &m), closed);
    assert_eq!((iv.log()[0].in_window, iv.log()[0].applied), (2, 1));
    // 0/0 ban: the t decision only.
    let mut iv = Intervention::new(&alt(&[(Op::Ban, "Build_Bunker", 0, 0)]), t, f_t).unwrap();
    assert_eq!(iv.apply(f_t, &m), closed);
    assert_eq!(iv.apply(f_t + 1, &m), m);
    // NoOp stays open under bans; a NoOp ban is refused as a configuration error (051-R7 ③).
    let mut iv = Intervention::new(&alt(&[(Op::Ban, "Build_Bunker", 0, 90), (Op::Ban, "Train_Marine", 0, 90)]), t, f_t).unwrap();
    assert_eq!(open(&iv.apply(f_t, &m)), vec!["NoOp"]);
    let err = Intervention::new(&alt(&[(Op::Ban, "NoOp", 0, 90)]), t, f_t).unwrap_err();
    assert!(format!("{err:#}").contains("NoOp cannot be banned"), "{err:#}");
    // Window frames that overflow u32 are refused.
    assert!(Intervention::new(&alt(&[(Op::Ban, "Build_Bunker", 0, u32::MAX / 24 + 1)]), t, f_t).is_err());
    assert!(Intervention::new(&alt(&[(Op::Ban, "Build_Bunker", 0, 90)]), t, u32::MAX - 100).is_err());
    assert!(Intervention::new(&alt(&[(Op::Force, "Build_Bunker", 0, 0)]), t, u32::MAX).is_err());
    // A force never opens a closed bit, under any rule mix; the mask never empties.
    let mut iv = Intervention::new(&alt(&[(Op::Force, "Build_Bunker", 0, 90), (Op::Ban, "Build_Bunker", 0, 30), (Op::Ban, "Train_Marine", 0, 90)]), t, f_t).unwrap();
    for (k, frame) in [f_t, f_t + 8, f_t + 30 * 24].into_iter().enumerate() {
        let out = iv.apply(frame, &m);
        assert!(out.iter().zip(&m).all(|(o, i)| !*o || *i), "opened a closed bit");
        assert!(out.iter().any(|&b| b));
        if k < 2 {
            assert_eq!(open(&out), vec!["NoOp"], "ban before force");
        } else {
            assert_eq!(open(&out), vec!["Build_Bunker"], "forced once the ban ends");
        }
    }
    // Rule errors: unknown action, empty window other than 0/0.
    assert!(Intervention::new(&alt(&[(Op::Ban, "Build_Nothing", 0, 9)]), t, f_t).is_err());
    assert!(Intervention::new(&alt(&[(Op::Ban, "Build_Bunker", 9, 9)]), t, f_t).is_err());
    assert!(Intervention::new(&alt(&[(Op::Force, "Build_Bunker", 9, 3)]), t, f_t).is_err());
    // No rules: `none` leaves every mask alone.
    let mut iv = Intervention::new(&alt(&[]), t, f_t).unwrap();
    assert_eq!(iv.apply(f_t, &m), m);
    assert!(alt(&[]).is_none());
}

/// 051-R10: t-only forces closed at t make the alternative not apply at that point.
#[test]
fn branch_closed_t_forces_051_r10() {
    let t = action_table();
    let a = alt(&[(Op::Force, "Build_Bunker", 0, 0), (Op::Force, "Train_Marine", 0, 30), (Op::Ban, "Train_Marine", 0, 0)]);
    assert_eq!(a.closed_t_forces(t, &mask_with(&[])).unwrap(), vec!["Build_Bunker"]);
    assert!(a.closed_t_forces(t, &mask_with(&["Build_Bunker"])).unwrap().is_empty());
}

/// 051-AC1 / 051-R5 ①, 051-R6 ①: frame or digest mismatches make the branch invalid; matches
/// give the recorded line (its `Act` bytes unchanged); after t the session plays live.
#[test]
fn branch_forced_prefix_051_ac1() {
    let log = model_log();
    let (i_t, f_t) = (3, log.lines[3].frame);
    assert!(ForcedPrefix::new(log.clone(), i_t, f_t + 1).is_err(), "f_t must be the trunk's frame at i_t");
    assert!(ForcedPrefix::new(log.clone(), 10_000, f_t).is_err());
    let mut p = ForcedPrefix::new(log.clone(), i_t, f_t).unwrap();
    for i in 0..i_t {
        let l = &log.lines[i as usize];
        match p.step(i, l.frame, l.digest).unwrap() {
            PrefixStep::Forced(r) => {
                assert_eq!(enc(&r.act), enc(&l.act));
                assert_eq!(r.decision.as_ref().unwrap().memory_action(), MemoryAction { action: 37, delay_bucket: 5, units: vec![1203, 7] });
            }
            other => panic!("{other:?}"),
        }
    }
    let l = &log.lines[i_t as usize];
    assert!(matches!(p.step(i_t, l.frame, l.digest).unwrap(), PrefixStep::AtT(_)));
    assert_eq!(p.step(i_t + 1, 0, 0).unwrap(), PrefixStep::Live);
    assert_eq!(p.matched(), i_t + 1);
    // Mismatches.
    let mut p = ForcedPrefix::new(log.clone(), i_t, f_t).unwrap();
    let l = &log.lines[1];
    assert_eq!(p.step(1, l.frame + 1, l.digest).unwrap_err(), Invalid { reason: InvalidReason::Frame, obs: Some(1), frame: Some(l.frame + 1) });
    assert_eq!(p.step(1, l.frame, l.digest ^ 1).unwrap_err().reason, InvalidReason::Digest);
    assert_eq!(p.step(i_t, f_t, 0).unwrap_err().reason, InvalidReason::Digest, "t itself is checked (051-R6 ①)");
    // A T0 log cannot be forced as a model session.
    let mut t0 = log.clone();
    t0.lines[0].decision = None;
    assert!(ForcedPrefix::new(t0, i_t, f_t).is_err());
    // The silent leave (010-R13 surrender with empty chat).
    let q = quit_act();
    assert!(q.cmds.is_empty() && q.surrender.as_deref() == Some("") && q.advance_frames >= 1);
    assert_eq!(serde_json::to_string(&InvalidReason::MissingInput).unwrap(), "\"missing_input\"");
}

/// 051-R12 `prefix` / 051-AC5: the compared prefix and the first divergence.
#[test]
fn branch_compare_prefix_051_r12() {
    let trunk = model_log().lines;
    assert_eq!(compare_prefix(&trunk, &trunk, 24), PrefixCompare { obs: 3, matched: 3, first_mismatch_frame: None });
    let mut b = trunk.clone();
    b[5].digest ^= 1;
    assert_eq!(compare_prefix(&trunk, &b, 24), PrefixCompare { obs: 3, matched: 3, first_mismatch_frame: Some(b[5].frame) });
    b[1].digest ^= 1;
    assert_eq!(compare_prefix(&trunk, &b, 24).matched, 2);
    // A shorter branch diverges where it ends.
    assert_eq!(compare_prefix(&trunk, &trunk[..4], 24).first_mismatch_frame, Some(trunk[4].frame));
}

/// 051-R9 ① time-point samples.
#[test]
fn branch_samples_051_r9() {
    let f_t = 3300;
    let mut s = SampleTracker::new(f_t, 11_520);
    for k in 0..=40u32 {
        let frame = f_t + k * 200;
        s.on_decision(frame, &[k as f32], true);
    }
    let x = s.samples();
    let at = |o: &Option<Sample>| o.as_ref().map(|s| s.frame);
    assert_eq!(at(&x.t60), Some(f_t + 8 * 200), "first decision with frame >= f_t + 1440");
    assert_eq!(at(&x.t120), Some(f_t + 15 * 200));
    assert_eq!(at(&x.t240), Some(f_t + 29 * 200));
    assert_eq!(at(&x.horizon), None, "not reached");
    assert_eq!(x.d8.as_ref().unwrap().scalar, vec![8.0]);
    assert_eq!(at(&x.d32), Some(f_t + 32 * 200));
    let mut s = SampleTracker::new(f_t, 4000);
    s.on_decision(3999, &[], true);
    s.on_decision(4000, &[], true);
    assert_eq!(s.samples().horizon.as_ref().unwrap().frame, 4000);
}

/// 051-R9 ① (user decision 2026-11-13): d8 / d32 count only decisions after t that sent a
/// command (not NoOp, not refused); the t decision never counts; time samples take every decision.
#[test]
fn branch_samples_count_command_decisions_051_r9() {
    let f_t = 3300;
    let mut s = SampleTracker::new(f_t, 11_520);
    // k = 0 is t (a command, not counted); after t every third decision is a NoOp or refused.
    let mut sent = 0;
    for k in 0..=60u32 {
        let cmd = k == 0 || k % 3 != 0;
        s.on_decision(f_t + k * 10, &[k as f32], cmd);
        if k > 0 && cmd {
            sent += 1;
            if sent == 8 {
                assert_eq!(s.samples().d8.as_ref().unwrap().scalar, vec![k as f32]);
            }
        }
    }
    let x = s.samples();
    // Command decisions after t: k = 1, 2, 4, 5, 7, 8, 10, 11 -> the 8th is k = 11.
    assert_eq!(x.d8.as_ref().unwrap().scalar, vec![11.0]);
    // 32nd: 16 pairs (k = 3j+1, 3j+2), the 32nd is k = 3 * 15 + 2 = 47.
    assert_eq!(x.d32.as_ref().unwrap().scalar, vec![47.0]);
    let mut s = SampleTracker::new(f_t, 11_520);
    for k in 0..=40u32 {
        s.on_decision(f_t + k * 200, &[k as f32], false);
    }
    assert!(s.samples().d8.is_none() && s.samples().d32.is_none(), "no command, no decision sample");
    assert_eq!(s.samples().t60.as_ref().map(|x| x.frame), Some(f_t + 8 * 200), "time samples take every decision");
}

/// 051-R9 ③ dilution: the alternative's actions, or the configured list for `none`; t excluded.
#[test]
fn branch_dilution_051_r9() {
    let t = action_table();
    let f_t = 1000;
    let mut d = DilutionTracker::new(&alt(&[(Op::Force, "Build_Bunker", 0, 0), (Op::Ban, "Build_Bunker", 0, 90)]), &[], t, f_t).unwrap();
    d.on_decision(f_t, id("Build_Bunker"));
    d.on_decision(f_t + 100, id("Train_Marine"));
    d.on_decision(f_t + 200, id("Build_Bunker"));
    d.on_decision(f_t + 720, id("Build_Bunker"));
    assert_eq!(d.entries(), vec![DilutionEntry { action: "Build_Bunker".into(), first_frame: Some(f_t + 200), picks_30s: 1 }]);
    let none = ["Build_Bunker".to_string(), "Train_Marine".to_string()];
    let mut d = DilutionTracker::new(&alt(&[]), &none, t, f_t).unwrap();
    d.on_decision(f_t + 1, id("Train_Marine"));
    let e = d.entries();
    assert_eq!((e[0].first_frame, e[1].first_frame, e[1].picks_30s), (None, Some(f_t + 1), 1));
}

/// 051-R10 / 051-R12: plan lines and records round-trip as JSON lines.
#[test]
fn branch_records_round_trip_051_r12() {
    let a = alt(&[(Op::Force, "Build_Bunker", 0, 0), (Op::Ban, "Train_Marine", 0, 90)]);
    let line = BranchPlanLine {
        trunk: "branch-b1-trunk/s1/g3".into(),
        point: "branch-b1-trunk/s1/g3#b1-p1".into(),
        acts: "eval/branch-b1-trunk/s1/acts/g3-p0.jsonl".into(),
        opponent_acts: "eval/branch-b1-trunk/s1/acts/g3-p1.jsonl".into(),
        state: "eval/branch-b1-trunk/s1/states/g3-o412.state".into(),
        state_sha256: sha256_hex(b"x"),
        i_t: 412,
        f_t: 3256,
        seed: 70_002,
        map: "fighting_spirit".into(),
        opponent: "zerg_rush".into(),
        alt: a.clone(),
        repeat: 2,
    };
    let mut buf = Vec::new();
    write_jsonl(&mut buf, std::slice::from_ref(&line)).unwrap();
    let text = String::from_utf8(buf.clone()).unwrap();
    assert!(text.contains(r#"{"op":"force","action":"Build_Bunker","from_s":0,"to_s":0}"#), "{text}");
    assert_eq!(read_jsonl::<BranchPlanLine>(&buf[..]).unwrap(), vec![line]);
    // An outcome with every part filled.
    let mut iv = Intervention::new(&a, action_table(), 3256).unwrap();
    iv.apply(3256, &mask_with(&["Build_Bunker"]));
    iv.note_executor(3256, false);
    let mut metrics = serde_json::Map::new();
    metrics.insert("held_8m".into(), serde_json::json!(true));
    metrics.insert("bunker_s".into(), serde_json::json!(135.5));
    let rec = BranchRecord {
        point: "p".into(),
        alt: a.name.clone(),
        rules: a.rules.clone(),
        repeat: 1,
        game: GameRef {
            job: "branch-b1".into(),
            stage: "s1".into(),
            plan_game: 9,
            hostd_game: 1 << 40,
            attempt: 0,
            replay: Some("replays/g9.rep".into()),
            host: "<host>".into(),
            backend: "rocm".into(),
            precision: "f32".into(),
            infer_seed: Some(5),
        },
        checkpoint_sha256: sha256_hex(b"ckpt"),
        git: "d45cb8c".into(),
        valid: true,
        invalid: None,
        prefix: PrefixRecord { obs: 412, matched: 412, opponent: Some(PrefixCompare { obs: 300, matched: 300, first_mismatch_frame: Some(3264) }) },
        state_check: Some(StateCheck::new(true, 0.0)),
        intervention: iv.log(),
        samples: Samples { t60: Some(Sample { frame: 4700, scalar: vec![0.1, -2.5, f32::MIN_POSITIVE] }), ..Default::default() },
        metrics,
        dilution: vec![DilutionEntry { action: "Build_Bunker".into(), first_frame: None, picks_30s: 0 }],
        end: Some(BranchEnd::new(Some(1), 0, 11_568, EndReason::TimeLimitDraw, 11_520)),
    };
    let text = serde_json::to_string(&rec).unwrap();
    assert!(text.contains(r#"{"op":"force","action":"Build_Bunker","from_s":0,"to_s":0,"in_window":1,"applied":1,"force_frame":3256,"missed":false,"rejected":false}"#), "{text}");
    assert_eq!(serde_json::from_str::<BranchRecord>(&text).unwrap(), rec);
    assert_eq!(rec.end.as_ref().unwrap().result, GameResult::Loss);
    assert!(!rec.end.as_ref().unwrap().before_horizon);
    assert_eq!(BranchEnd::new(None, 0, 5000, EndReason::Elimination, 11_520).result, GameResult::Draw);
    assert!(BranchEnd::new(Some(0), 0, 5000, EndReason::Elimination, 11_520).before_horizon);
    // A point record with its hex digest.
    let p = PointRecord {
        point: "p".into(),
        trunk: TrunkRef { job: "j".into(), stage: "s".into(), plan_game: 3, hostd_game: 7, replay: "r".into(), acts: "a".into(), opponent_acts: "o".into() },
        seed: 1,
        map: "circuit_breaker".into(),
        opponent: "zerg_rush".into(),
        condition: Some("hit".into()),
        checkpoint_sha256: "c".into(),
        policy: "model:x:stochastic".into(),
        t0_policy: "t0@1".into(),
        git: "g".into(),
        schemas: Schemas { feature: 7, action: 2 },
        backend: "rocm".into(),
        precision: "f32".into(),
        infer_seed: Some(11),
        i_t: 412,
        f_t: 3256,
        digest_t: u64::MAX - 1,
        type_mask: open_ids(&mask_with(&["Build_Bunker"])),
        scalar_t: vec![1.0],
        trunk_action: Decision { action: 0, delay_bucket: 3, units: vec![], rejected: false },
        state: "g3-o412".into(),
        state_sha256: "s".into(),
        trunk_outcome: serde_json::Map::new(),
        valid: true,
        skip_reason: None,
    };
    let text = serde_json::to_string(&p).unwrap();
    assert!(text.contains(r#""digest_t":"fffffffffffffffe""#));
    assert_eq!(serde_json::from_str::<PointRecord>(&text).unwrap(), p);
    assert_eq!(act_log_name(3, 1), "g3-p1.jsonl");
    let o = BranchOutcome { valid: true, intervention: iv.log(), ..Default::default() };
    assert_eq!(serde_json::from_str::<BranchOutcome>(&serde_json::to_string(&o).unwrap()).unwrap(), o);
    // write_jsonl round trips both record kinds.
    let mut buf = Vec::new();
    write_jsonl(&mut buf, std::slice::from_ref(&rec)).unwrap();
    assert_eq!(read_jsonl::<BranchRecord>(&buf[..]).unwrap(), vec![rec.clone()]);
    let mut buf = Vec::new();
    write_jsonl(&mut buf, std::slice::from_ref(&p)).unwrap();
    assert_eq!(read_jsonl::<PointRecord>(&buf[..]).unwrap(), vec![p.clone()]);

    // Non-finite floats would come back as unreadable nulls: refused before anything is written,
    // and a non-finite state difference is recorded as None.
    assert_eq!(StateCheck::new(false, f64::NAN).max_abs_diff, None);
    assert_eq!(StateCheck::new(false, f64::INFINITY).max_abs_diff, None);
    assert_eq!(StateCheck::new(false, 0.5).max_abs_diff, Some(0.5));
    let mut bad = rec.clone();
    bad.samples.d8 = Some(Sample { frame: 1, scalar: vec![0.0, f32::NAN] });
    let mut buf = Vec::new();
    let err = write_jsonl(&mut buf, &[rec.clone(), bad]).unwrap_err();
    assert!(format!("{err:#}").contains("samples.d8.scalar[1]"), "{err:#}");
    assert!(buf.is_empty(), "nothing written");
    let mut bad = rec.clone();
    bad.state_check = Some(StateCheck { bit_equal: false, max_abs_diff: Some(f64::NAN) });
    assert!(write_jsonl(Vec::new(), &[bad]).is_err());
    let mut bad = p.clone();
    bad.scalar_t = vec![f32::NEG_INFINITY];
    assert!(write_jsonl(Vec::new(), &[bad]).is_err());
    // The B1 metric keys (051-R11 ④), primary first.
    assert_eq!(&OUTCOME_METRICS[..2], &["held_8m", "held_s"]);
}

/// 020-R22 ① ②: a game's act-log and state file paths; `SnapshotTaken` (the trunk session
/// record's `snapshots` entries) round trips with its hex digest and refuses non-finite scalars.
#[test]
fn branch_game_settings_020_r22() {
    let g = GameBranch { plan_game: 7, acts_dir: Some("a".into()), states_dir: Some("s".into()), ..Default::default() };
    assert_eq!(g.act_log_path(1), Some(std::path::Path::new("a").join("g7-p1.jsonl")));
    assert_eq!(g.state_path(412), Some(std::path::Path::new("s").join("g7-o412.state")));
    assert_eq!(GameBranch::default().act_log_path(0), None);
    assert_eq!(GameBranch::default().state_path(0), None);
    let s = SnapshotTaken { rules: vec!["b1-p1".into(), "b1-p2".into()], state: state_id(7, 412), state_sha256: "ab".into(), i: 412, frame: 3256, digest: 0x0f, type_mask: vec![0, 37], scalar: vec![0.5] };
    let text = serde_json::to_string(&s).unwrap();
    assert!(text.contains(r#""digest":"000000000000000f""#), "{text}");
    assert_eq!(serde_json::from_str::<SnapshotTaken>(&text).unwrap(), s);
    let bad = SnapshotTaken { scalar: vec![f32::NAN], ..s };
    assert!(write_jsonl(Vec::new(), &[bad]).is_err());
}
