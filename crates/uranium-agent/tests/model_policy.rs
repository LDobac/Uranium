//! Model policy session tests on Flex (020-R11, 031-R4, 032-R14, 031-R7). A tiny network on the
//! live feature / action schema keeps them fast; no game data is needed.

#[path = "../../uranium-env/tests/common/mod.rs"]
mod common;

use std::io::{BufReader, BufWriter, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use burn::tensor::Device;
use common::*;
use uranium_agent::infer::{InferBackend, InferOptions, InferenceHandle, SampleMode, write_random_checkpoint};
use uranium_agent::model_policy::ModelPolicy;
use uranium_agent::{PolicySpec, Routes, ServeOptions, serve};
use uranium_env::features::DELAY_BUCKETS;
use uranium_model::config::*;
use uranium_proto::*;

fn tiny_cfg(kind: CoreKind) -> UraniumNetConfig {
    UraniumNetConfig::new()
        .with_input(uranium_model::collate::input_dims())
        .with_entity(EntityConfig::new().with_d_model(32).with_layers(1).with_heads(4).with_ffn(64).with_dropout(0.0).with_unit_type_emb(8).with_order_emb(8).with_last_action_emb(4))
        .with_spatial(SpatialConfig::new().with_scatter_channels(4).with_stem_channels(8).with_stages([8, 8, 16, 16]).with_pool(32))
        .with_scalar(ScalarConfig::new().with_hidden(32).with_race_emb(4).with_map_family_emb(4).with_start_loc_emb(4).with_z_build_emb(8).with_z_units_emb(8).with_z_out(8))
        .with_core(CoreConfig::new().with_kind(kind).with_hidden(32).with_gtrxl_heads(4).with_gtrxl_memory(4))
        .with_heads(HeadsConfig::new().with_ar(32).with_key(16).with_loc_ar_channels(4))
        .with_value(ValueConfig::new().with_hidden(32).with_privileged(8))
}

fn checkpoint(kind: CoreKind, tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("uranium_agent_ckpt_{tag}_{kind:?}_{}", std::process::id()));
    let d = Device::flex();
    d.seed(11);
    write_random_checkpoint(&dir, &tiny_cfg(kind), &d).unwrap();
    dir
}

fn handle(dir: &PathBuf, sample: SampleMode, max_batch: usize) -> InferenceHandle {
    let mut o = InferOptions::new(InferBackend::Flex);
    o.max_batch = max_batch;
    o.max_wait = Duration::from_millis(200);
    o.capacity = 8;
    InferenceHandle::start(dir, sample, o).unwrap()
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

fn plays_valid_acts(kind: CoreKind) {
    let dir = checkpoint(kind, "valid");
    let h = handle(&dir, SampleMode::Stochastic, 4);
    let (acts, stats) = play(&h, &frames(0, 12));
    for (cmds, adv) in &acts {
        assert!(cmds.len() <= 1, "031-R3: one command per action");
        assert!(DELAY_BUCKETS.contains(adv), "031-R9: advance {adv} is not a delay bucket");
    }
    assert_eq!(stats.decisions, 12);
    // 031-AC2: the model samples within the masks, so only executor-side causes can reject.
    for k in stats.rejected.keys() {
        assert_eq!(k, "no_executor_unit", "masked action reached the executor: {:?}", stats.rejected);
    }
    assert_eq!(stats.map_precompute, "missing");
    std::fs::remove_dir_all(&dir).ok();
}

/// 031-R7 / 020-AC2: argmax play is deterministic.
fn argmax_is_deterministic(kind: CoreKind) {
    let dir = checkpoint(kind, "det");
    let h = handle(&dir, SampleMode::Argmax, 4);
    let f = frames(1, 8);
    assert_eq!(play(&h, &f).0, play(&h, &f).0);
    std::fs::remove_dir_all(&dir).ok();
}

/// 032-R14: three sessions batched together act like each session served alone, and their Core
/// states do not mix.
fn batched_sessions_match_single(kind: CoreKind) {
    let dir = checkpoint(kind, "batch");
    let run = |max_batch: usize| -> Vec<(Vec<(Vec<Command>, u32)>, f64)> {
        let h = handle(&dir, SampleMode::Argmax, max_batch);
        let threads: Vec<_> = (0..3)
            .map(|k| {
                let h = h.clone();
                std::thread::spawn(move || {
                    let (acts, s) = play(&h, &frames(k, 10));
                    (acts, s.mean_batch)
                })
            })
            .collect();
        threads.into_iter().map(|t| t.join().unwrap()).collect()
    };
    let batched = run(3);
    let single = run(1);
    assert!(batched.iter().any(|(_, b)| *b > 1.0), "sessions were never batched: {:?}", batched.iter().map(|x| x.1).collect::<Vec<_>>());
    for (k, ((a, _), (b, _))) in batched.iter().zip(&single).enumerate() {
        assert_eq!(a, b, "session {k}: batched play differs from single play");
    }
    std::fs::remove_dir_all(&dir).ok();
}

macro_rules! core_tests {
    ($m:ident, $kind:expr) => {
        mod $m {
            use super::*;
            #[test]
            fn plays_valid_acts_031_r4() {
                plays_valid_acts($kind);
            }
            #[test]
            fn argmax_is_deterministic_031_r7() {
                argmax_is_deterministic($kind);
            }
            #[test]
            fn batched_sessions_match_single_032_r14() {
                batched_sessions_match_single($kind);
            }
        }
    };
}

core_tests!(lstm, CoreKind::Lstm);
core_tests!(gtrxl, CoreKind::Gtrxl);

/// 020-R11: two bridges of one game on one server get the policy routed to their slot; slots
/// without a route use the default (T0).
#[test]
fn routes_pick_policy_per_slot_020_r11() {
    let dir = checkpoint(CoreKind::Lstm, "route");
    let h = handle(&dir, SampleMode::Argmax, 2);
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
    std::fs::remove_dir_all(&dir).ok();
}

/// M6 hook (042): every decision reaches the observer with per-head log-probs that add up to the
/// joint behaviour log-prob (032-R15, AMB-042-4).
#[test]
fn observer_sees_behaviour_log_probs_042() {
    struct Obs(Arc<Mutex<Vec<(f32, f32)>>>);
    impl uranium_agent::model_policy::SessionObserver for Obs {
        fn on_decision(&mut self, d: &uranium_agent::model_policy::Decision<'_>) {
            self.0.lock().unwrap().push((d.log_prob, d.head_log_prob.iter().sum()));
        }
    }
    let dir = checkpoint(CoreKind::Lstm, "observer");
    let h = handle(&dir, SampleMode::Stochastic, 1);
    let seen = Arc::new(Mutex::new(vec![]));
    let mut p = ModelPolicy::new(&h, &game_info()).unwrap().with_observer(None, Box::new(Obs(seen.clone())));
    for f in frames(2, 5) {
        p.on_frame(&f).unwrap();
    }
    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 5);
    for (joint, heads) in seen.iter() {
        assert!((joint - heads).abs() < 1e-3 && *joint <= 0.0, "{joint} vs {heads}");
    }
    std::fs::remove_dir_all(&dir).ok();
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
    let dir = checkpoint(CoreKind::Lstm, "final");
    let h = handle(&dir, SampleMode::Argmax, 1);
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
    std::fs::remove_dir_all(&dir).ok();
}

/// 032-R16: a slot's Core state can be read before a decision, and swapped weights answer with
/// their version from the next batch on.
#[test]
fn weight_swap_and_slot_state_032_r16() {
    use uranium_env::features::{AgentMemory, MapInfo, extract};
    let dir = checkpoint(CoreKind::Lstm, "swap");
    let h = handle(&dir, SampleMode::Argmax, 1);
    let sess = h.open().unwrap();
    let s0 = sess.state().unwrap();
    assert!(s0.parts.iter().all(|(_, v)| v.iter().all(|x| *x == 0.0)), "fresh slot state is zero (032-R5)");
    let info = game_info();
    let mut mem = AgentMemory::new(&info);
    let map = MapInfo::new(&info, 0, None).unwrap();
    let fr = frames(0, 2);
    let r = sess.act(extract(&fr[0], &mut mem, &map, None)).unwrap();
    assert_eq!(r.version, 0);
    assert_ne!(sess.state().unwrap(), s0, "state advanced after a decision");
    let cols = uranium_model::batch::ActionColumns::from_table(uranium_env::features::extract::action_table()).unwrap();
    let (net, _) = uranium_model::checkpoint::load::<uranium_model::LstmCore>(&dir, &cols, uranium_env::features::FEATURE_SCHEMA_VERSION, uranium_env::actions::ACTION_SCHEMA_VERSION, &Device::flex()).unwrap();
    h.swap(Box::new(net), 7).unwrap();
    assert!(h.swap(Box::new(5u32), 8).is_err(), "only a network of the server's core is accepted");
    let r = sess.act(extract(&fr[1], &mut mem, &map, None)).unwrap();
    assert_eq!(r.version, 7);
    drop(sess);
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

/// 031-R12 / 020-R14 / 010-R13: Shadow checks without leaving; On (here forced at a frame, as the
/// integration check does) sends one surrender Act with the handle's chat line and no commands.
#[test]
fn surrender_modes_031_r12() {
    let dir = std::env::temp_dir().join(format!("uranium_agent_ckpt_open_{}", std::process::id()));
    let d = Device::flex();
    d.seed(11);
    let cfg = tiny_cfg(CoreKind::Lstm);
    write_random_checkpoint(&dir, &cfg.clone().with_value(cfg.value.clone().with_win_open(true)), &d).unwrap();
    let fs = frames(0, 6);

    // Shadow (the handle default): plays as before; before 10 minutes the check never holds.
    let h = handle(&dir, SampleMode::Argmax, 1);
    assert_eq!(h.surrender, uranium_agent::infer::SurrenderMode::Shadow);
    let mut p = ModelPolicy::new(&h, &game_info()).unwrap();
    for f in &fs {
        assert!(p.on_frame(f).unwrap().surrender.is_none());
    }
    let s = p.stats();
    assert!(s.win_open, "the checkpoint has the head");
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
    let _ = std::fs::remove_dir_all(&dir);
}
