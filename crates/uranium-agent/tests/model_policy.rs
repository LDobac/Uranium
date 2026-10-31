//! Model policy session tests (020-R11, 031-R4, 032-R14, 032-R16, 031-R12) with a mock model
//! behind the inference broker (AMB-000-4): every decision is a NoOp whose delay bucket is the
//! number of decisions its slot has made (so each reply must reach the right session and slot
//! states must not mix), with fixed log-probs, values and win_open. Checks that need the real
//! network (masked sampling, argmax determinism, batched = single play) are in
//! tests/python/test_infer.py.

#[path = "../../uranium-env/tests/common/mod.rs"]
mod common;

use std::io::{BufReader, BufWriter, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use common::*;
use uranium_agent::broker::{BatchOut, Broker};
use uranium_agent::infer::{HEADS, HostState, InferOptions, InferenceHandle, SampleMode};
use uranium_agent::model_policy::ModelPolicy;
use uranium_agent::{PolicySpec, RejectionLog, Routes, ServeOptions, serve};
use uranium_env::features::DELAY_BUCKETS;
use uranium_proto::*;

const HEAD_LOGP: [f32; HEADS] = [-0.5, -0.25, 0.0, 0.0, 0.0, 0.0];

/// A mock model thread on a broker; `win_open` is its 032-R18 output (None = no head).
fn mock(max_batch: usize, win_open: Option<f32>) -> InferenceHandle {
    let opts = InferOptions { max_batch, max_wait: Duration::from_millis(200), capacity: 8, ..InferOptions::default() };
    let (mut b, h) = Broker::new("mock@000000000000".into(), SampleMode::Argmax, opts).unwrap();
    std::thread::spawn(move || {
        let mut decisions = vec![0usize; opts.capacity];
        while !b.closed() {
            let Some(w) = b.next(Duration::from_millis(50)).unwrap() else { continue };
            for s in w.resets {
                decisions[s] = 0;
            }
            let states = w.states.iter().map(|&s| HostState { parts: vec![(vec![1, 1], vec![decisions[s] as f32])] }).collect();
            b.reply_states(states).unwrap();
            if let Some(batch) = w.batch {
                let r = batch.slots.len();
                let delay: Vec<i64> = batch.slots.iter().map(|&s| (decisions[s] % DELAY_BUCKETS.len()) as i64).collect();
                for &s in &batch.slots {
                    decisions[s] += 1;
                }
                let zeros = vec![0i64; r];
                let sel = vec![batch.obs.n as i64; r * 12];
                let lp: Vec<f32> = vec![HEAD_LOGP.iter().sum(); r];
                let heads: Vec<f32> = (0..r).flat_map(|_| HEAD_LOGP).collect();
                let value = vec![0.25f32; r * 4];
                let open = win_open.map(|v| vec![v; r]);
                let out = BatchOut { action: &zeros, delay: &delay, queued: &zeros, selected: &sel, target_unit: &zeros, target_loc: &zeros, log_prob: &lp, head_log_prob: &heads, value: &value, win_open: open.as_deref() };
                b.reply(out).unwrap();
            }
        }
    });
    h
}

/// Frames that differ per session `k` and step `t` (units move, some appear later).
fn frames(k: usize, steps: usize) -> Vec<RawFrame> {
    (0..steps)
        .map(|t| {
            let mut b = rich_units();
            let keep = b.units.len() - (k * 3 + t) % 7;
            b.units.truncate(keep);
            for u in b.units.iter_mut().filter(|u| u.owner == SELF) {
                u.pos.0 += (t * 8 + k * 4) as u16;
            }
            frame(24 * t as u32 + 1, b.units, t % 2 == 0)
        })
        .collect()
}

/// Plays `frames` and returns (commands, advance) per step.
fn play(h: &InferenceHandle, frames: &[RawFrame]) -> (Vec<(Vec<Command>, u32)>, uranium_agent::model_policy::ModelStats) {
    let mut p = ModelPolicy::new(h, &game_info()).unwrap();
    let acts = frames.iter().map(|f| p.on_frame(f).unwrap()).map(|a| (a.cmds, a.advance_frames)).collect();
    (acts, p.stats())
}

/// 032-R14 / 032-R5: sessions batched together each get their own rows back, and a slot's state
/// (here its decision count) neither mixes with others nor survives a close and reopen.
#[test]
fn batched_rows_reach_their_sessions_032_r14() {
    let h = mock(3, None);
    let run = || {
        let threads: Vec<_> = (0..3)
            .map(|k| {
                let h = h.clone();
                std::thread::spawn(move || play(&h, &frames(k, 6)))
            })
            .collect();
        threads.into_iter().map(|t| t.join().unwrap()).collect::<Vec<_>>()
    };
    for _ in 0..2 {
        let res = run();
        assert!(res.iter().any(|(_, s)| s.mean_batch > 1.0), "sessions were never batched");
        for (acts, s) in res {
            let adv: Vec<u32> = acts.iter().map(|a| a.1).collect();
            assert_eq!(adv, DELAY_BUCKETS[..6].to_vec(), "rows or slot states mixed");
            assert!(acts.iter().all(|a| a.0.is_empty()), "NoOp sends no command");
            assert_eq!(s.decisions, 6);
            assert!(s.rejected.is_empty());
        }
    }
}

/// 032-R16: a slot's Core state is read before a decision (fresh slot: zero); the broker refuses
/// a handle-side weight swap (the model thread swaps).
#[test]
fn slot_state_and_swap_032_r16() {
    use uranium_env::features::{AgentMemory, MapInfo, extract};
    let h = mock(1, None);
    let sess = h.open().unwrap();
    assert_eq!(sess.state().unwrap().parts[0].1, vec![0.0]);
    let info = game_info();
    let mut mem = AgentMemory::new(&info);
    let map = MapInfo::new(&info, 0, None).unwrap();
    let fr = frames(0, 2);
    let r = sess.act(extract(&fr[0], &mut mem, &map, None)).unwrap();
    assert_eq!((r.version, r.batch), (0, 1));
    assert_eq!(sess.state().unwrap().parts[0].1, vec![1.0]);
    assert!(h.swap(Box::new(5u32), 8).is_err());
    drop(sess);
    let again = h.open().unwrap();
    assert_eq!(again.state().unwrap().parts[0].1, vec![0.0], "a reopened slot starts from zero (032-R5)");
}

/// 020-R11: two bridges of one game on one server get the policy routed to their slot; slots
/// without a route use the default (T0).
#[test]
fn routes_pick_policy_per_slot_020_r11() {
    let h = mock(2, None);
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let routes = Routes::default();
    routes.insert(7, 0, PolicySpec::Model(h.clone()));
    let sink: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(Vec::new()));
    let opts = ServeOptions { routes, max_sessions: 2, ..ServeOptions::new(PolicySpec::ScriptedT0) };
    let server_sink = sink.clone();
    let server = std::thread::spawn(move || serve(listener, opts, Arc::new(Mutex::new(SinkWriter(server_sink)))).unwrap());
    let bridge = |slot: u8| {
        std::thread::spawn(move || {
            let s = TcpStream::connect(addr).unwrap();
            let mut r = BufReader::new(s.try_clone().unwrap());
            let mut w = BufWriter::new(s);
            let hello = Hello { protocol_version: PROTOCOL_VERSION, backend: Backend::OpenBw, game_version: "test".into(), mode: Mode::Lockstep, game_id: 7, slot, replay: false };
            write_msg(&mut w, &BridgeMsg::Hello(hello)).unwrap();
            let mut info = game_info();
            info.self_player = slot;
            for p in &mut info.players {
                p.is_self = p.id == slot;
                p.is_enemy = p.id != slot;
            }
            info.players[1].race = Race::Terran;
            write_msg(&mut w, &BridgeMsg::GameInfo(Box::new(info))).unwrap();
            for f in frames(slot as usize, 3) {
                write_msg(&mut w, &BridgeMsg::Frame(Box::new(f))).unwrap();
                w.flush().unwrap();
                let AgentMsg::Act(_) = read_msg::<AgentMsg>(&mut r).unwrap() else { panic!("expected Act") };
            }
            write_msg(&mut w, &BridgeMsg::End(GameEnd { winner: Some(0), frame: 100, reason: EndReason::Elimination, stats: None, final_frame: None })).unwrap();
            w.flush().unwrap();
        })
    };
    let (b0, b1) = (bridge(0), bridge(1));
    b0.join().unwrap();
    b1.join().unwrap();
    // The accept loop returns on the next connection after max_sessions; poke it.
    std::thread::sleep(Duration::from_millis(300));
    let _ = TcpStream::connect(addr);
    server.join().unwrap();
    let text = String::from_utf8(sink.lock().unwrap().clone()).unwrap();
    let recs: Vec<serde_json::Value> = text.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    let by_slot = |s: u64| recs.iter().find(|r| r["slot"] == s).unwrap_or_else(|| panic!("no record for slot {s}: {text}"));
    assert!(by_slot(0)["policy"].as_str().unwrap().starts_with("model:"), "{text}");
    assert_eq!(by_slot(0)["model"]["decisions"], 3);
    assert!(by_slot(1)["policy"].as_str().unwrap().starts_with("t0@"), "{text}");
    assert!(by_slot(1)["model"].is_null());
}

/// 012-R7 / 012-R18 / 010-R10: a realtime (SC:R) session's Acts expire 24 frames after the frame
/// they answer, its model plays with last_failed = 0 although the server was started without
/// it, and a PrivilegedSummary is counted but kept from the policy. A lockstep session on the same
/// server is unchanged.
#[test]
fn realtime_session_012_r18() {
    let h = mock(2, None);
    assert!(!h.zero_last_failed);
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let sink: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(Vec::new()));
    let opts = ServeOptions { max_sessions: 2, ..ServeOptions::new(PolicySpec::Model(h.clone())) };
    let server_sink = sink.clone();
    let server = std::thread::spawn(move || serve(listener, opts, Arc::new(Mutex::new(SinkWriter(server_sink)))).unwrap());
    let bridge = |game_id: u64, mode: Mode| {
        std::thread::spawn(move || {
            let s = TcpStream::connect(addr).unwrap();
            let mut r = BufReader::new(s.try_clone().unwrap());
            let mut w = BufWriter::new(s);
            let backend = if mode == Mode::Realtime { Backend::Remastered } else { Backend::OpenBw };
            let hello = Hello { protocol_version: PROTOCOL_VERSION, backend, game_version: "test".into(), mode, game_id, slot: 0, replay: false };
            write_msg(&mut w, &BridgeMsg::Hello(hello)).unwrap();
            write_msg(&mut w, &BridgeMsg::GameInfo(Box::new(game_info()))).unwrap();
            let mut deadlines = vec![];
            for f in frames(0, 3) {
                write_msg(&mut w, &BridgeMsg::Privileged(PrivilegedSummary { frame: f.frame, ..Default::default() })).unwrap();
                let seen = f.frame;
                write_msg(&mut w, &BridgeMsg::Frame(Box::new(f))).unwrap();
                w.flush().unwrap();
                let AgentMsg::Act(a) = read_msg::<AgentMsg>(&mut r).unwrap() else { panic!("expected Act") };
                deadlines.push((seen, a.valid_until_frame));
            }
            write_msg(&mut w, &BridgeMsg::End(GameEnd { winner: Some(0), frame: 100, reason: EndReason::Elimination, stats: None, final_frame: None })).unwrap();
            w.flush().unwrap();
            deadlines
        })
    };
    let realtime = bridge(1, Mode::Realtime).join().unwrap();
    let lockstep = bridge(2, Mode::Lockstep).join().unwrap();
    for (seen, until) in realtime {
        assert_eq!(until, Some(seen + uranium_agent::REALTIME_ACT_FRAMES), "012-R7 deadline");
    }
    assert!(lockstep.iter().all(|(_, until)| until.is_none()), "lockstep Acts carry no deadline");
    std::thread::sleep(Duration::from_millis(300));
    let _ = TcpStream::connect(addr);
    server.join().unwrap();
    let text = String::from_utf8(sink.lock().unwrap().clone()).unwrap();
    let recs: Vec<serde_json::Value> = text.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    let by_game = |g: u64| recs.iter().find(|r| r["game_id"] == g).unwrap_or_else(|| panic!("no record for game {g}: {text}"));
    assert_eq!(by_game(1)["model"]["zero_last_failed"], true, "012-R14: realtime sessions feed last_failed = 0");
    assert_eq!(by_game(2)["model"]["zero_last_failed"], false);
    assert_eq!(by_game(1)["privileged_msgs"], 3, "counted for 012-AC6");
    assert_eq!(by_game(1)["model"]["decisions"], 3);
}

/// M6 hook (042): every decision reaches the observer with the per-head log-probs and the joint
/// behaviour log-prob of the server (032-R15, AMB-042-4).
#[test]
fn observer_sees_behaviour_log_probs_042() {
    struct Obs(Arc<Mutex<Vec<(f32, f32)>>>);
    impl uranium_agent::model_policy::SessionObserver for Obs {
        fn on_decision(&mut self, d: &uranium_agent::model_policy::Decision<'_>) {
            self.0.lock().unwrap().push((d.log_prob, d.head_log_prob.iter().sum()));
        }
    }
    let h = mock(1, None);
    let seen = Arc::new(Mutex::new(vec![]));
    let mut p = ModelPolicy::new(&h, &game_info()).unwrap().with_observer(None, Box::new(Obs(seen.clone())));
    for f in frames(2, 5) {
        p.on_frame(&f).unwrap();
    }
    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 5);
    for (joint, heads) in seen.iter() {
        assert_eq!((*joint, *heads), (-0.75, -0.75));
    }
}

/// E-8 / 010-R12 / 033-R7: the dense reward of frames after the last decision comes from the
/// bridge's final frame and reaches the observer's end call (it lands on the last decision, 042);
/// without a final frame only r_win is added.
#[test]
fn final_frame_reward_reaches_the_last_decision_010_r12() {
    struct Obs(Arc<Mutex<Vec<uranium_env::reward::RewardParts>>>);
    impl uranium_agent::model_policy::SessionObserver for Obs {
        fn on_decision(&mut self, _d: &uranium_agent::model_policy::Decision<'_>) {}
        fn on_end(&mut self, after: uranium_env::reward::RewardParts, _o: Option<uranium_env::reward::Outcome>) {
            self.0.lock().unwrap().push(after);
        }
    }
    let h = mock(1, None);
    let run = |with_final: bool| {
        let seen = Arc::new(Mutex::new(vec![]));
        let mut p = ModelPolicy::new(&h, &game_info()).unwrap().with_observer(None, Box::new(Obs(seen.clone())));
        let fs = frames(0, 3);
        for f in &fs {
            p.on_frame(f).unwrap();
        }
        // After the last decision an enemy unit we saw dies in sight.
        let mut last = fs.last().unwrap().clone();
        let victim = last.units.iter().position(|u| u.owner == ENEMY).expect("an enemy unit in sight");
        let gone = last.units.remove(victim);
        last.frame += 12;
        last.events = vec![GameEvent::UnitDestroyed(gone.id)];
        p.on_end(Some(uranium_env::reward::Outcome::Win), with_final.then_some(&last));
        let v = seen.lock().unwrap().clone();
        assert_eq!(v.len(), 1);
        (v[0], gone.unit_type)
    };
    let (with, ty) = run(true);
    let cost = uranium_env::types::TypeTables::get().unit(ty).map_or(0.0, |t| (t.minerals + t.gas) as f32) / 1000.0;
    assert_eq!(with.win, 1.0);
    assert!(cost > 0.0 && (with.value - cost).abs() < 1e-6, "value {} vs kill {cost}", with.value);
    let (without, _) = run(false);
    assert_eq!((without.win, without.value), (1.0, 0.0));
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

/// 031-R12 / 020-R14 / 010-R13: Shadow checks without leaving; On (here forced at a frame, as the
/// integration check does) sends one surrender Act with the handle's chat line and no commands.
#[test]
fn surrender_modes_031_r12() {
    let fs = frames(0, 6);

    // Shadow (the handle default): plays as before; before 10 minutes the check never holds.
    let h = mock(1, Some(-0.95));
    assert_eq!(h.surrender, uranium_agent::infer::SurrenderMode::Shadow);
    let mut p = ModelPolicy::new(&h, &game_info()).unwrap();
    for f in &fs {
        assert!(p.on_frame(f).unwrap().surrender.is_none());
    }
    let s = p.stats();
    assert!(s.win_open, "the server answers with win_open");
    assert_eq!((s.surrender.as_str(), s.surrender_frame, s.decisions), ("shadow", None, 6));

    // On, forced from frame 49 (the third frame): one surrender Act, then the bridge leaves.
    // SAFETY: only this test sets the variable, and only On-mode sessions read it.
    unsafe { std::env::set_var("URANIUM_TEST_SURRENDER_FRAME", "49") };
    let h = h.with_surrender(uranium_agent::infer::SurrenderMode::On, "good game");
    let mut p = ModelPolicy::new(&h, &game_info()).unwrap();
    unsafe { std::env::remove_var("URANIUM_TEST_SURRENDER_FRAME") };
    let acts: Vec<Act> = fs[..3].iter().map(|f| p.on_frame(f).unwrap()).collect();
    assert!(acts[..2].iter().all(|a| a.surrender.is_none()));
    assert_eq!(acts[2].surrender.as_deref(), Some("good game"));
    assert!(acts[2].cmds.is_empty(), "010-R13: no commands with a surrender");
    let s = p.stats();
    assert_eq!((s.surrender.as_str(), s.surrender_frame, s.decisions), ("on", Some(49), 3));
}

/// 031-R10 / 010-R14 / 010-R12: `play` counts `UnitRejected` from observations and from
/// `GameEnd.final_frame`; a session that ends in an error (bridge gone mid-game) still keeps its
/// counts in the record and its detail lines in the file.
#[test]
fn unit_rejections_in_sessions_031_r10() {
    let h = mock(1, None);
    let dir = std::env::temp_dir().join(format!("uranium-agent-rej-session-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let path = dir.join("unit_rejections.jsonl");
    let opts = ServeOptions { unit_rejections: RejectionLog::new(path.clone()), ..ServeOptions::new(PolicySpec::Model(h.clone())) };
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let rej = |cmd_index: u16, unit: u32, unit_type: Option<u16>| GameEvent::UnitRejected { cmd_index, unit, unit_type, command: 14, error: 10 };
    // Two observations (the second with one rejection), then a GameEnd whose final frame carries
    // two more (`finish`), or the bridge drops the connection after the second Act.
    let run = |game_id: u64, finish: bool| {
        let bridge = std::thread::spawn(move || {
            let s = TcpStream::connect(addr).unwrap();
            let mut r = BufReader::new(s.try_clone().unwrap());
            let mut w = BufWriter::new(s);
            let hello = Hello { protocol_version: PROTOCOL_VERSION, backend: Backend::OpenBw, game_version: "test".into(), mode: Mode::Lockstep, game_id, slot: 0, replay: false };
            write_msg(&mut w, &BridgeMsg::Hello(hello)).unwrap();
            write_msg(&mut w, &BridgeMsg::GameInfo(Box::new(game_info()))).unwrap();
            let mut fr = frames(0, 3);
            fr[1].events.push(rej(0, 5, Some(0)));
            let mut last = fr.pop().unwrap();
            for f in fr {
                write_msg(&mut w, &BridgeMsg::Frame(Box::new(f))).unwrap();
                w.flush().unwrap();
                let AgentMsg::Act(_) = read_msg::<AgentMsg>(&mut r).unwrap() else { panic!("expected Act") };
            }
            if finish {
                last.events.extend([rej(0, 6, Some(0)), rej(1, 7, None)]);
                write_msg(&mut w, &BridgeMsg::End(GameEnd { winner: Some(0), frame: 100, reason: EndReason::Elimination, stats: None, final_frame: Some(Box::new(last)) })).unwrap();
                w.flush().unwrap();
            }
        });
        let (stream, _) = listener.accept().unwrap();
        let rec = uranium_agent::session(stream, &opts);
        bridge.join().unwrap();
        rec
    };
    let ended = run(1, true);
    assert!(ended.error.is_none(), "{:?}", ended.error);
    assert_eq!(ended.unit_rejections, 3, "the final frame's rejections count too");
    assert_eq!(ended.unit_rejections_by.get("14/0/10"), Some(&2));
    assert_eq!(ended.unit_rejections_by.get("14/none/10"), Some(&1));
    let dropped = run(2, false);
    assert!(dropped.error.is_some(), "the bridge left without GameEnd");
    assert_eq!(dropped.unit_rejections, 1);
    let lines: Vec<serde_json::Value> = std::fs::read_to_string(&path).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    let games: Vec<u64> = lines.iter().map(|l| l["game_id"].as_u64().unwrap()).collect();
    assert_eq!(games, vec![1, 1, 1, 2]);
    assert_eq!(lines[1]["frame"], lines[2]["frame"], "both final-frame lines carry its frame");
    assert_eq!(lines[3]["unit"], 5);
    let _ = std::fs::remove_dir_all(&dir);
}
