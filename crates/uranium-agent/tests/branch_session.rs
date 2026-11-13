//! 051 branch-game sessions (051-AC1 session part, 051-R2..R9, 020-R22 ④) with a mock model behind
//! the inference broker (AMB-000-4). The mock's Core state is its slot's decision count (shape
//! [1, 1]); it samples the only open action when `type_mask` has exactly one (a `force`), else
//! NoOp, with delay bucket = decisions so far; it logs the open action ids of every row it sees.
//! The game-level checks (051-AC3..AC6) need hostd: `tests/branch_replay.rs` and the e2e run.

#[path = "../../uranium-env/tests/common/mod.rs"]
mod common;

use std::io::{BufReader, BufWriter, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use common::*;
use uranium_agent::broker::{BatchOut, Broker};
use uranium_agent::infer::{HEADS, HostState, InferOptions, InferenceHandle, SampleMode};
use uranium_agent::model_policy::ModelPolicy;
use uranium_agent::{BranchBoard, GameBranch, PolicySpec, Routes, ServeOptions, serve};
use uranium_env::branch::*;
use uranium_env::features::extract::action_table;
use uranium_env::features::{AgentMemory, DELAY_BUCKETS, MapInfo, extract};
use uranium_proto::*;

/// (slot, open action ids) of every row the mock ran.
type Seen = Arc<Mutex<Vec<(usize, Vec<u16>)>>>;

fn mock() -> (InferenceHandle, Seen) {
    let opts = InferOptions { max_batch: 1, max_wait: Duration::from_millis(50), capacity: 8, ..InferOptions::default() };
    let (mut b, h) = Broker::new("mock@000000000000".into(), SampleMode::Stochastic, opts).unwrap();
    let seen: Seen = Arc::default();
    let log = seen.clone();
    std::thread::spawn(move || {
        let mut decisions = vec![0usize; opts.capacity];
        while !b.closed() {
            let Some(w) = b.next(Duration::from_millis(50)).unwrap() else { continue };
            for s in w.resets {
                decisions[s] = 0;
            }
            let errors = w
                .restores
                .iter()
                .map(|(s, st)| match st.parts.as_slice() {
                    [(shape, v)] if shape == &[1, 1] => {
                        decisions[*s] = v[0] as usize;
                        None
                    }
                    _ => Some("shape differs from the slot's [1, 1]".to_string()),
                })
                .collect();
            b.reply_restores(errors).unwrap();
            let states = w.states.iter().map(|&s| HostState { parts: vec![(vec![1, 1], vec![decisions[s] as f32])] }).collect();
            b.reply_states(states).unwrap();
            if let Some(batch) = w.batch {
                let (r, a) = (batch.slots.len(), batch.obs.a);
                let mut action = vec![0i64; r];
                for (row, &s) in batch.slots.iter().enumerate() {
                    let open: Vec<u16> = (0..a).filter(|&k| batch.obs.type_mask[row * a + k]).map(|k| k as u16).collect();
                    if open.len() == 1 {
                        action[row] = open[0] as i64;
                    }
                    log.lock().unwrap().push((s, open));
                }
                let delay: Vec<i64> = batch.slots.iter().map(|&s| (decisions[s] % DELAY_BUCKETS.len()) as i64).collect();
                for &s in &batch.slots {
                    decisions[s] += 1;
                }
                let zeros = vec![0i64; r];
                let sel = vec![batch.obs.n as i64; r * 12];
                let lp = vec![-0.5f32; r];
                let heads: Vec<f32> = (0..r).flat_map(|_| [-0.5f32, 0.0, 0.0, 0.0, 0.0, 0.0]).collect();
                assert_eq!(heads.len(), r * HEADS);
                let value = vec![0.25f32; r * 4];
                let out = BatchOut { action: &action, delay: &delay, queued: &zeros, selected: &sel, target_unit: &zeros, target_loc: &zeros, log_prob: &lp, head_log_prob: &heads, value: &value, win_open: None };
                b.reply(out).unwrap();
            }
        }
    });
    (h, seen)
}

/// Frames that differ per step (units move, some appear later), frame numbers 24 t + 1.
fn frames(steps: usize) -> Vec<RawFrame> {
    (0..steps)
        .map(|t| {
            let mut b = rich_units();
            let keep = b.units.len() - t % 7;
            b.units.truncate(keep);
            for u in b.units.iter_mut().filter(|u| u.owner == SELF) {
                u.pos.0 += (t * 8) as u16;
            }
            frame(24 * t as u32 + 1, b.units, t % 2 == 0)
        })
        .collect()
}

fn temp(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("uranium-agent-branch-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn postcard(a: &Act) -> Vec<u8> {
    let mut v = Vec::new();
    write_msg(&mut v, a).unwrap();
    v
}

const I_T: u32 = 5;

/// A trunk game: plan game 1, snapshot rule `p` from the frame of observation `I_T` (no condition,
/// 051-R3), state files under `dir/states`.
fn trunk_game(dir: &Path) -> Arc<GameBranch> {
    let rule = SnapshotRule { name: "p".into(), from_frame: 24 * I_T + 1, to_frame: 100_000, any_open: vec![] };
    Arc::new(GameBranch { plan_game: 1, acts_dir: Some(dir.join("acts")), snapshots: vec![rule], states_dir: Some(dir.join("states")), branch: None })
}

/// Plays `fs` with `p` the way the session does (051-R2 lines from `act_log_extra`) and returns
/// the complete log (end line included).
fn play_logged(p: &mut ModelPolicy, fs: &[RawFrame]) -> (Vec<Act>, ActLog) {
    let mut acts = vec![];
    let mut lines = vec![];
    for (i, f) in fs.iter().enumerate() {
        let act = p.on_frame(f).unwrap();
        let x = p.act_log_extra().expect("051 session");
        lines.push(ObsLine { i: i as u32, frame: f.frame, digest: obs_digest(f).unwrap(), act: act.clone(), decision: x.decision, state: x.state, state_sha256: x.state_sha256, forced: x.forced });
        acts.push(act);
        if act_is_leave(acts.last().unwrap()) {
            break;
        }
    }
    let last = fs[lines.len() - 1].frame;
    p.on_end(Some(uranium_env::reward::Outcome::Draw), None);
    let end = ActLogEnd { obs: lines.len() as u64, decisions: lines.iter().filter(|l| l.decision.is_some()).count() as u64, end_frame: last, reason: EndReason::TimeLimitDraw, player: SELF, players_index: 0, policy: "model:mock".into(), git: "x".into(), feature_schema: 7, action_schema: 2 };
    (acts, ActLog { lines, end })
}

fn act_is_leave(a: &Act) -> bool {
    a.surrender.is_some()
}

/// A trunk act log and state file on disk: (log path, trunk acts, log, state file, its sha256).
struct Trunk {
    acts_path: PathBuf,
    acts: Vec<Act>,
    log: ActLog,
    state: PathBuf,
    sha: String,
    memory: AgentMemory,
}

fn trunk(h: &InferenceHandle, dir: &Path, fs: &[RawFrame]) -> Trunk {
    let game = trunk_game(dir);
    let mut p = ModelPolicy::new(h, &game_info()).unwrap().with_branch(game.clone(), 0).unwrap();
    let (acts, log) = play_logged(&mut p, fs);
    let acts_path = dir.join("acts").join(act_log_name(1, 0));
    std::fs::create_dir_all(acts_path.parent().unwrap()).unwrap();
    log.write(std::fs::File::create(&acts_path).unwrap()).unwrap();
    let rec = p.branch_record().unwrap();
    assert_eq!(rec.snapshots.len(), 1);
    let s = &rec.snapshots[0];
    let state = game.state_path(s.i).unwrap();
    Trunk { acts_path, acts, log, state, sha: s.state_sha256.clone(), memory: p.memory().clone() }
}

fn branch_game(t: &Trunk, alt: Alternative, verify_state: bool) -> Arc<GameBranch> {
    let line = BranchPlanLine {
        trunk: "job/s/g1".into(),
        point: "job/s/g1#p".into(),
        acts: "acts/g1-p0.jsonl".into(),
        opponent_acts: "acts/g1-p1.jsonl".into(),
        state: "states/g1-o5.state".into(),
        state_sha256: t.sha.clone(),
        i_t: I_T,
        f_t: 24 * I_T + 1,
        seed: 1,
        map: "synthetic".into(),
        opponent: "t0".into(),
        alt,
        repeat: 1,
    };
    let player = BranchPlayer { line, acts: t.acts_path.clone(), state: t.state.clone(), horizon: 24 * 9 + 1, none_actions: vec!["Build_Bunker".into(), "Train_Marine".into()], verify_state };
    Arc::new(GameBranch { plan_game: 2, acts_dir: None, snapshots: vec![], states_dir: None, branch: Some(player) })
}

fn none() -> Alternative {
    Alternative { name: "none".into(), rules: vec![] }
}

fn rows_of(seen: &Seen) -> usize {
    seen.lock().unwrap().len()
}

/// 051-R3: the trunk's `players[0]` session saves its Core state right before the first decision
/// in the rule's window (one file, bit-exact, sha256 in the act-log line and the record).
#[test]
fn trunk_saves_the_state_at_the_snapshot_decision_051_r3() {
    let dir = temp("trunk");
    let (h, _) = mock();
    let fs = frames(8);
    let t = trunk(&h, &dir, &fs);
    let bytes = std::fs::read(&t.state).unwrap();
    assert_eq!(sha256_hex(&bytes), t.sha);
    // The mock's state before decision i is i.
    assert_eq!(decode_state(&bytes).unwrap(), vec![(vec![1, 1], vec![I_T as f32])]);
    let at = &t.log.lines[I_T as usize];
    assert_eq!((at.state.as_deref(), at.state_sha256.as_deref()), (Some(state_id(1, I_T).as_str()), Some(t.sha.as_str())));
    assert!(t.log.lines.iter().filter(|l| l.state.is_some()).count() == 1, "the rule fires once");
    assert!(t.log.lines.iter().all(|l| l.decision.is_some() && !l.forced));
    // The other players index logs but never snapshots (020-R22 ②).
    let mut q = ModelPolicy::new(&h, &game_info()).unwrap().with_branch(trunk_game(&dir.join("other")), 1).unwrap();
    let (_, log) = play_logged(&mut q, &fs);
    assert!(log.lines.iter().all(|l| l.state.is_none()) && q.branch_record().unwrap().snapshots.is_empty());
    std::fs::remove_dir_all(&dir).ok();
}

/// 051-R5 / 051-R6 / 051-AC1 (강제 앞부분): observations matching the trunk get the recorded Acts
/// byte for byte without inference, memory ends up as the trunk's, the restored state at t gives
/// the trunk's decision, and the act-log lines say `forced` before t.
#[test]
fn forced_prefix_replays_the_trunk_and_restores_at_t_051_r5_r6() {
    let dir = temp("prefix");
    let (h, seen) = mock();
    let fs = frames(9);
    let t = trunk(&h, &dir, &fs);
    let before = rows_of(&seen);
    let mut p = ModelPolicy::new(&h, &game_info()).unwrap().with_branch(branch_game(&t, none(), false), 0).unwrap();
    let mut acts = vec![];
    let mut extras = vec![];
    for (i, f) in fs.iter().enumerate() {
        acts.push(p.on_frame(f).unwrap());
        extras.push(p.act_log_extra().unwrap());
        if i < I_T as usize {
            assert_eq!(rows_of(&seen), before, "no inference before t (051-R5 ②)");
        }
    }
    for (i, (a, b)) in acts.iter().zip(&t.acts).enumerate() {
        assert_eq!(postcard(a), postcard(b), "observation {i}: Act bytes");
    }
    // Delay bucket at t = the restored decision count (the trunk's), not 0 of a fresh slot.
    assert_eq!(acts[I_T as usize].advance_frames, DELAY_BUCKETS[I_T as usize]);
    for (i, x) in extras.iter().enumerate() {
        assert_eq!(x.forced, i < I_T as usize, "observation {i}");
        assert_eq!(x.decision, t.log.lines[i].decision, "observation {i}: decision");
    }
    assert_eq!(rows_of(&seen), before + fs.len() - I_T as usize);
    // Memory: the same as the trunk session's after the same observations (051-AC1).
    let map = MapInfo::new(&game_info(), 0, None).unwrap();
    let next = frames(10).pop().unwrap();
    let (mut a, mut b) = (p.memory().clone(), t.memory.clone());
    assert_eq!((a.last_action, &a.unit_last_action), (b.last_action, &b.unit_last_action));
    assert_eq!(extract(&next, &mut a, &map, None), extract(&next, &mut b, &map, None));
    p.on_end(Some(uranium_env::reward::Outcome::Draw), fs.last());
    let o = p.branch_record().unwrap().outcome.unwrap();
    assert!(o.valid && o.invalid.is_none());
    assert_eq!((o.prefix_obs, o.prefix_matched, o.state_check.clone()), (I_T + 1, I_T + 1, None));
    assert!(o.intervention.is_empty());
    // 051-R9 ①: t decision is k = 0; the 8th decision after t and the horizon were not reached.
    assert_eq!((o.samples.d8.is_none(), o.samples.horizon.is_none(), o.samples.t60.is_none()), (true, true, true));
    for k in OUTCOME_METRICS {
        assert!(o.metrics.contains_key(*k), "{k}");
    }
    assert_eq!(o.dilution.iter().map(|d| d.action.as_str()).collect::<Vec<_>>(), ["Build_Bunker", "Train_Marine"]);
    std::fs::remove_dir_all(&dir).ok();
}

/// 051-R5 ①: a digest or frame different from the trunk before t makes the branch invalid and the
/// session leaves without chat (and keeps leaving); inputs that cannot be read or do not match
/// their sha256 are `missing_input` / `state` at the first observation (051-R6 ②).
#[test]
fn mismatch_or_bad_inputs_leave_without_chat_051_r5() {
    let dir = temp("mismatch");
    let (h, _) = mock();
    let fs = frames(8);
    let t = trunk(&h, &dir, &fs);
    let run = |game: Arc<GameBranch>, fs: &[RawFrame]| {
        let mut p = ModelPolicy::new(&h, &game_info()).unwrap().with_branch(game, 0).unwrap();
        let acts: Vec<Act> = fs.iter().map(|f| p.on_frame(f).unwrap()).collect();
        p.on_end(Some(uranium_env::reward::Outcome::Loss), None);
        (acts, p.branch_record().unwrap().outcome.unwrap())
    };
    let mut changed = fs.clone();
    changed[3].self_state.minerals += 1;
    let (acts, o) = run(branch_game(&t, none(), false), &changed);
    assert_eq!(o.invalid, Some(Invalid { reason: InvalidReason::Digest, obs: Some(3), frame: Some(changed[3].frame) }));
    assert!(!o.valid);
    assert_eq!((o.prefix_obs, o.prefix_matched), (4, 3));
    assert_eq!(postcard(&acts[2]), postcard(&t.acts[2]));
    for a in &acts[3..] {
        assert_eq!(a, &quit_act(), "leave without chat (010-R13 surrender, empty text)");
    }
    let mut late = fs.clone();
    late[2].frame += 1;
    let (_, o) = run(branch_game(&t, none(), false), &late);
    assert_eq!(o.invalid.unwrap().reason, InvalidReason::Frame);

    let mut bad_sha = (*branch_game(&t, none(), false)).clone();
    bad_sha.branch.as_mut().unwrap().line.state_sha256 = "0".repeat(64);
    let (acts, o) = run(Arc::new(bad_sha), &fs);
    assert_eq!(o.invalid, Some(Invalid { reason: InvalidReason::State, obs: Some(0), frame: Some(fs[0].frame) }));
    assert_eq!(acts[0], quit_act());
    let mut no_log = (*branch_game(&t, none(), false)).clone();
    no_log.branch.as_mut().unwrap().acts = dir.join("missing.jsonl");
    let (_, o) = run(Arc::new(no_log), &fs);
    assert_eq!(o.invalid.unwrap().reason, InvalidReason::MissingInput);
    // The game ended before t: the trunk's observation i_t never came.
    let (_, o) = run(branch_game(&t, none(), false), &fs[..3]);
    assert_eq!(o.invalid.unwrap().reason, InvalidReason::Frame);
    std::fs::remove_dir_all(&dir).ok();
}

/// 051-R7 (031-R4 exception): from t on the narrowed `type_mask` reaches the model and the
/// executor; a t-only `force` narrows the t decision to {action} only, a `ban` closes the action
/// over its window and nothing else; the record logs both.
#[test]
fn interventions_narrow_the_mask_from_t_051_r7() {
    let dir = temp("intervene");
    let (h, seen) = mock();
    let fs = frames(9);
    let t = trunk(&h, &dir, &fs);
    // The t observation's open actions (the trunk's view): force / ban the first non-NoOp one.
    let map = MapInfo::new(&game_info(), 0, None).unwrap();
    let mut mem = AgentMemory::new(&game_info());
    let obs: Vec<_> = fs.iter().map(|f| extract(f, &mut mem, &map, None)).collect();
    let open_t = open_ids(&obs[I_T as usize].masks.type_mask);
    let target = *open_t.iter().find(|&&a| a != 0).expect("an open action at t");
    let name = action_table().actions[target as usize].name.clone();
    let rule = |op, to_s| InterventionRule { op, action: name.clone(), from_s: 0, to_s };

    let alt = Alternative { name: "force".into(), rules: vec![rule(Op::Force, 0)] };
    let mut p = ModelPolicy::new(&h, &game_info()).unwrap().with_branch(branch_game(&t, alt, false), 0).unwrap();
    let start = rows_of(&seen);
    for f in &fs {
        p.on_frame(f).unwrap();
    }
    let rows: Vec<Vec<u16>> = seen.lock().unwrap()[start..].iter().map(|(_, o)| o.clone()).collect();
    assert_eq!(rows[0], vec![target], "the t decision sees {{action}} only");
    assert_eq!(rows[1], open_ids(&obs[I_T as usize + 1].masks.type_mask), "a t-only force leaves later decisions alone");
    p.on_end(Some(uranium_env::reward::Outcome::Draw), None);
    let o = p.branch_record().unwrap().outcome.unwrap();
    let log = &o.intervention[0];
    assert_eq!((log.applied, log.force_frame, log.missed), (1, Some(fs[I_T as usize].frame), false));
    assert!(log.rejected.is_some(), "the executor's verdict on the forced decision (051-R7 ⑤)");
    assert_eq!(o.dilution[0].action, name);

    let alt = Alternative { name: "ban".into(), rules: vec![rule(Op::Ban, 2)] };
    let mut p = ModelPolicy::new(&h, &game_info()).unwrap().with_branch(branch_game(&t, alt, false), 0).unwrap();
    let start = rows_of(&seen);
    for f in &fs {
        p.on_frame(f).unwrap();
    }
    let rows: Vec<Vec<u16>> = seen.lock().unwrap()[start..].iter().map(|(_, o)| o.clone()).collect();
    for (k, row) in rows.iter().enumerate() {
        let full = open_ids(&obs[I_T as usize + k].masks.type_mask);
        // Window [f_t, f_t + 48): observations t and t + 1 (frames 24 apart).
        let want: Vec<u16> = if k < 2 { full.into_iter().filter(|&a| a != target).collect() } else { full };
        assert_eq!(row, &want, "decision t + {k}");
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// 051-R6 test option `verify_state`: the prefix runs inference only to build the state, and at t
/// that state is compared with the saved one (here bit-equal); the Acts stay the trunk's.
#[test]
fn verify_state_compares_the_rebuilt_state_051_r6() {
    let dir = temp("verify");
    let (h, seen) = mock();
    let fs = frames(7);
    let t = trunk(&h, &dir, &fs);
    let start = rows_of(&seen);
    let mut p = ModelPolicy::new(&h, &game_info()).unwrap().with_branch(branch_game(&t, none(), true), 0).unwrap();
    let acts: Vec<Act> = fs.iter().map(|f| p.on_frame(f).unwrap()).collect();
    assert_eq!(rows_of(&seen) - start, fs.len(), "inference on every observation");
    assert!(acts.iter().zip(&t.acts).all(|(a, b)| postcard(a) == postcard(b)));
    p.on_end(None, None);
    let o = p.branch_record().unwrap().outcome.unwrap();
    assert_eq!(o.state_check, Some(StateCheck { bit_equal: true, max_abs_diff: Some(0.0) }));
    std::fs::remove_dir_all(&dir).ok();
}

struct SinkWriter(Arc<Mutex<Vec<u8>>>);

impl Write for SinkWriter {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(b);
        Ok(b.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// One session per (game id, slot) in turn on one server with `board`; slot 0 plays the model,
/// slot 1 the T0 default. Each bridge sends its frames (until the agent leaves), then GameEnd.
fn serve_games(h: &InferenceHandle, board: BranchBoard, games: &[(u64, u8, Vec<RawFrame>)]) -> Vec<serde_json::Value> {
    serve_games_ending(h, board, games, EndReason::TimeLimitDraw)
}

/// [`serve_games`] whose games that run to their last frame end with `reason`.
fn serve_games_ending(h: &InferenceHandle, board: BranchBoard, games: &[(u64, u8, Vec<RawFrame>)], reason: EndReason) -> Vec<serde_json::Value> {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let routes = Routes::default();
    for (g, slot, _) in games {
        if *slot == 0 {
            routes.insert(*g, 0, PolicySpec::Model(h.clone()));
        }
    }
    let sink: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(Vec::new()));
    let opts = ServeOptions { routes, branches: board, max_sessions: games.len(), ..ServeOptions::new(PolicySpec::ScriptedT0) };
    let server_sink = sink.clone();
    let server = std::thread::spawn(move || serve(listener, opts, Arc::new(Mutex::new(SinkWriter(server_sink)))).unwrap());
    for (game_id, slot, fs) in games.iter().cloned() {
        let s = TcpStream::connect(addr).unwrap();
        let mut r = BufReader::new(s.try_clone().unwrap());
        let mut w = BufWriter::new(s);
        let hello = Hello { protocol_version: PROTOCOL_VERSION, backend: Backend::OpenBw, game_version: "test".into(), mode: Mode::Lockstep, game_id, slot, replay: false };
        write_msg(&mut w, &BridgeMsg::Hello(hello)).unwrap();
        let mut info = game_info();
        info.self_player = slot;
        for p in &mut info.players {
            p.is_self = p.id == slot;
            p.is_enemy = p.id != slot;
        }
        info.players[1].race = Race::Terran;
        write_msg(&mut w, &BridgeMsg::GameInfo(Box::new(info))).unwrap();
        let mut end = GameEnd { winner: None, frame: fs.last().unwrap().frame + 8, reason, stats: None, final_frame: None };
        for f in fs {
            let frame = f.frame;
            write_msg(&mut w, &BridgeMsg::Frame(Box::new(f))).unwrap();
            w.flush().unwrap();
            let AgentMsg::Act(a) = read_msg::<AgentMsg>(&mut r).unwrap() else { panic!("expected Act") };
            if a.surrender.is_some() {
                end = GameEnd { winner: Some(1 - slot), frame: frame + 1, reason: EndReason::Surrender, stats: None, final_frame: None };
                break;
            }
        }
        write_msg(&mut w, &BridgeMsg::End(end)).unwrap();
        w.flush().unwrap();
        // The server writes the record when the session ends.
        let n = games.iter().position(|g| g.0 == game_id && g.1 == slot).unwrap() + 1;
        while String::from_utf8(sink.lock().unwrap().clone()).unwrap().lines().count() < n {
            std::thread::sleep(Duration::from_millis(10));
        }
    }
    std::thread::sleep(Duration::from_millis(100));
    let _ = TcpStream::connect(addr);
    server.join().unwrap();
    let text = String::from_utf8(sink.lock().unwrap().clone()).unwrap();
    text.lines().map(|l| serde_json::from_str(l).unwrap()).collect()
}

/// 051-R2 / 020-R22 ① ② ③ through the agent server: every session of a game on the board writes
/// its act log (model lines with decisions, T0 lines without; observation digests restarted per
/// observation over the bytes of the session digest; the end line), the trunk record lists its
/// snapshot, and a branch game's record carries its outcome with `end`.
#[test]
fn sessions_write_act_logs_and_branch_records_051_r2() {
    let dir = temp("serve");
    let (h, _) = mock();
    let fs = frames(9);
    let board = BranchBoard::default();
    board.lock().unwrap().insert(11, trunk_game(&dir));
    let recs = serve_games(&h, board.clone(), &[(11, 0, fs.clone()), (11, 1, fs.clone())]);
    let ours = ActLog::read_path(&dir.join("acts").join(act_log_name(1, 0))).unwrap();
    let theirs = ActLog::read_path(&dir.join("acts").join(act_log_name(1, 1))).unwrap();
    assert_eq!(ours.lines.len(), fs.len());
    let mut session = FNV_OFFSET;
    for (l, f) in ours.lines.iter().zip(&fs) {
        let bytes = frame_bytes(f).unwrap();
        session = fnv1a(session, &bytes);
        assert_eq!((l.frame, l.digest), (f.frame, obs_digest_bytes(&bytes)));
        assert!(l.decision.is_some());
    }
    assert!(theirs.lines.iter().all(|l| l.decision.is_none() && l.state.is_none()));
    assert_eq!((ours.end.player, ours.end.players_index, ours.end.reason), (0, 0, EndReason::TimeLimitDraw));
    assert_eq!((theirs.end.player, theirs.end.players_index), (1, 1));
    assert!(ours.end.policy.starts_with("model:") && theirs.end.policy.starts_with("t0@"));
    assert_eq!((ours.end.feature_schema, ours.end.action_schema), (uranium_env::features::FEATURE_SCHEMA_VERSION, uranium_env::actions::ACTION_SCHEMA_VERSION));
    let model = recs.iter().find(|r| r["slot"] == 0).unwrap();
    assert_eq!(model["frame_digest"], digest_hex(session), "011-AC4 session digest from FNV_OFFSET over the same bytes");
    assert_eq!(model["act_log"]["complete"], true);
    let snap: SnapshotTaken = serde_json::from_value(model["snapshots"][0].clone()).unwrap();
    assert_eq!((snap.i, snap.rules.clone(), snap.digest), (I_T, vec!["p".to_string()], ours.lines[I_T as usize].digest));
    assert!(model.get("branch").is_none());
    let t0 = recs.iter().find(|r| r["slot"] == 1).unwrap();
    assert!(t0.get("snapshots").is_none() && t0.get("branch").is_none());

    // The branch game, from the trunk files the server wrote.
    let sha = snap.state_sha256.clone();
    let t = Trunk { acts_path: dir.join("acts").join(act_log_name(1, 0)), acts: vec![], log: ours.clone(), state: trunk_game(&dir).state_path(I_T).unwrap(), sha, memory: AgentMemory::new(&game_info()) };
    let mut g = (*branch_game(&t, none(), false)).clone();
    g.acts_dir = Some(dir.join("acts"));
    board.lock().unwrap().insert(12, Arc::new(g));
    let recs = serve_games(&h, board, &[(12, 0, fs.clone())]);
    let b = ActLog::read_path(&dir.join("acts").join(act_log_name(2, 0))).unwrap();
    for (i, (x, y)) in b.lines.iter().zip(&ours.lines).enumerate() {
        assert_eq!((x.frame, x.digest, x.forced), (y.frame, y.digest, i < I_T as usize));
        if i <= I_T as usize {
            assert_eq!(postcard(&x.act), postcard(&y.act), "observation {i}");
        }
    }
    let o: BranchOutcome = serde_json::from_value(recs[0]["branch"].clone()).unwrap();
    assert!(o.valid);
    assert_eq!(o.end, Some(BranchEnd { result: GameResult::Draw, end_frame: fs.last().unwrap().frame + 8, reason: EndReason::TimeLimitDraw, before_horizon: true }), "H = frame 217");
    assert_eq!(recs[0]["model"]["surrender"], "off", "051-R1 ③: a branch player never surrenders");
    std::fs::remove_dir_all(&dir).ok();
}

/// 020-R22 ④: a game that is not on the board (or a board entry of another game) plays exactly as
/// before: the same Acts and digest, and the record has none of the 051 keys.
#[test]
fn games_off_the_board_are_unchanged_020_r22() {
    let dir = temp("plain");
    let (h, _) = mock();
    let fs = frames(6);
    let board = BranchBoard::default();
    board.lock().unwrap().insert(99, trunk_game(&dir));
    let plain = serve_games(&h, BranchBoard::default(), &[(21, 0, fs.clone())]);
    let other = serve_games(&h, board.clone(), &[(22, 0, fs.clone())]);
    let logged = serve_games(&h, board, &[(99, 0, fs.clone())]);
    for r in plain.iter().chain(&other) {
        for k in ["act_log", "snapshots", "branch"] {
            assert!(r.get(k).is_none(), "{k}: {r}");
        }
    }
    let strip = |r: &serde_json::Value| {
        let mut r = r.clone();
        let o = r.as_object_mut().unwrap();
        for k in ["game_id", "wall_s", "policy_ms", "act_log", "snapshots"] {
            o.remove(k);
        }
        let m = o["model"].as_object_mut().unwrap();
        for k in ["decide_ms_p50", "decide_ms_p95", "run_ms_p50", "run_ms_p95", "host_ms_p50"] {
            m.remove(k);
        }
        r
    };
    assert_eq!(strip(&plain[0]), strip(&other[0]));
    assert_eq!(strip(&plain[0]), strip(&logged[0]), "act logs and snapshots change no Act, digest or metric");
    std::fs::remove_dir_all(&dir).ok();
}

/// 051-R5 ① / 051-R9 ④: an aborted game is an infrastructure failure, not an invalid branch. A
/// disconnect before t leaves `invalid` unset (a finished game that ended before t is `frame`, see
/// `mismatch_or_bad_inputs_leave_without_chat_051_r5`), and a `GameEnd` Aborted after t gives
/// `end: null` (no outcome; the job plays the game again, 020-R15 ④).
#[test]
fn aborted_branches_are_not_invalid_051_r9() {
    let dir = temp("abort");
    let (h, _) = mock();
    let fs = frames(9);
    let t = trunk(&h, &dir, &fs);
    let mut p = ModelPolicy::new(&h, &game_info()).unwrap().with_branch(branch_game(&t, none(), false), 0).unwrap();
    for f in &fs[..3] {
        p.on_frame(f).unwrap();
    }
    p.on_abort();
    let o = p.branch_record().unwrap().outcome.unwrap();
    assert_eq!((o.valid, o.invalid, o.end), (true, None, None));

    let board = BranchBoard::default();
    board.lock().unwrap().insert(31, branch_game(&t, none(), false));
    let recs = serve_games_ending(&h, board, &[(31, 0, fs.clone())], EndReason::Aborted);
    let o: BranchOutcome = serde_json::from_value(recs[0]["branch"].clone()).unwrap();
    assert_eq!((o.valid, o.invalid, o.end), (true, None, None));
    std::fs::remove_dir_all(&dir).ok();
}

/// 051-R5 ⑤ / 041-R28: a branch player's 041-R28 decision log marks the forced prefix decisions
/// `"forced":true`; every other line, and a plain (DAgger) session's whole log, is byte-identical
/// to the format before the field (no `forced` key), and both forms read back.
#[test]
fn decision_log_marks_forced_decisions_051_r5() {
    use uranium_env::features::label::DecisionLogLine;
    let dir = temp("memlog");
    let (h, _) = mock();
    let fs = frames(8);
    let t = trunk(&h, &dir, &fs);
    let log = dir.join("memory").join("g2.jsonl");
    let mut p = ModelPolicy::new(&h.clone().with_memory_log(log.clone()), &game_info()).unwrap().with_branch(branch_game(&t, none(), false), 0).unwrap();
    for f in &fs {
        p.on_frame(f).unwrap();
    }
    p.on_end(Some(uranium_env::reward::Outcome::Draw), None);
    let text = std::fs::read_to_string(&log).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), fs.len() + 1);
    for (i, l) in lines[..fs.len()].iter().enumerate() {
        assert_eq!(l.contains(r#""forced":true"#), i < I_T as usize, "line {i}: {l}");
        let parsed: DecisionLogLine = serde_json::from_str(l).unwrap();
        assert!(matches!(parsed, DecisionLogLine::Decision { forced, .. } if forced == (i < I_T as usize)), "line {i}");
    }
    assert_eq!(lines[0], r#"{"frame":1,"action":0,"delay_bucket":0,"units":[],"forced":true}"#);
    // A plain (DAgger) session: the bytes of the format before the field.
    let plain = dir.join("memory").join("g3.jsonl");
    let mut q = ModelPolicy::new(&h.clone().with_memory_log(plain.clone()), &game_info()).unwrap();
    for f in &fs[..2] {
        q.on_frame(f).unwrap();
    }
    q.on_end(Some(uranium_env::reward::Outcome::Draw), None);
    let want = concat!(r#"{"frame":1,"action":0,"delay_bucket":0,"units":[]}"#, "\n", r#"{"frame":25,"action":0,"delay_bucket":1,"units":[]}"#, "\n", r#"{"decisions":2,"end_frame":null,"player":0}"#, "\n");
    assert_eq!(std::fs::read_to_string(&plain).unwrap(), want);
    std::fs::remove_dir_all(&dir).ok();
}
