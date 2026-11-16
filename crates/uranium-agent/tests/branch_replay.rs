use std::io::BufReader;
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use uranium_agent::{BranchBoard, GameBranch, ServeOptions, T0Builds, open_sink, serve};
use uranium_env::branch::{ActLog, ActLogEnd, ActLogWriter, ObsLine, PrefixCompare, act_log_name, compare_prefix, frame_bytes, obs_digest_bytes};
use uranium_proto::control::{CONTROL_VERSION, ControlRequest, ControlResponse, GameOutcome, GameSpec, MapKey, PlayerKind, PlayerSpec};
use uranium_proto::*;
use uranium_scripted::{T0Build, for_game_build};

const MAX_FRAMES: u32 = 24 * 60 * 5;
const BRANCH_FRAME: u32 = 24 * 90;
const SEED: u32 = 70_002;

#[derive(Clone)]
struct Forced {
    trunk: Arc<ActLog>,
    branch: u32,
    log: PathBuf,
}

#[derive(Default, Debug)]
struct Trace {
    prefix_mismatch: u32,
    shadow_mismatch: u32,
    branched_at: Option<u32>,
    end: Option<String>,
}

fn forced_session(stream: TcpStream, m: Forced) -> Result<Trace, String> {
    let e = |x: ProtoError| x.to_string();
    let mut r = BufReader::new(stream.try_clone().map_err(|x| x.to_string())?);
    let mut w = stream;
    let BridgeMsg::Hello(hello) = read_msg(&mut r).map_err(e)? else { return Err("expected Hello".into()) };
    let BridgeMsg::GameInfo(info) = read_msg(&mut r).map_err(e)? else { return Err("expected GameInfo".into()) };
    let mut bot = for_game_build(&info, T0Build::Standard);
    let mut log = ActLogWriter::new(std::io::BufWriter::new(std::fs::File::create(&m.log).map_err(|x| x.to_string())?));
    let mut t = Trace::default();
    let mut i = 0u32;
    let mut branched = false;
    loop {
        let msg = match read_msg::<BridgeMsg>(&mut r) {
            Ok(x) => x,
            Err(x) => {
                t.end.get_or_insert(format!("connection closed: {x}"));
                return Ok(t);
            }
        };
        match msg {
            BridgeMsg::Frame(f) => {
                let bytes = frame_bytes(&f).map_err(e)?;
                let digest = obs_digest_bytes(&bytes);
                let own = bot.act(&f);
                let act = match m.trunk.line(i) {
                    Some(rec) if !branched => {
                        if rec.frame != f.frame || rec.digest != digest {
                            t.prefix_mismatch += 1;
                        }
                        if f.frame >= m.branch && !rec.act.cmds.is_empty() {
                            branched = true;
                            t.branched_at = Some(f.frame);
                            Act { cmds: vec![], advance_frames: rec.act.advance_frames, valid_until_frame: None, surrender: None }
                        } else {
                            if own != rec.act {
                                t.shadow_mismatch += 1;
                            }
                            rec.act.clone()
                        }
                    }
                    _ => own,
                };
                log.push(&ObsLine { i, frame: f.frame, digest, act: act.clone(), decision: None, state: None, state_sha256: None, forced: !branched }).map_err(|x| x.to_string())?;
                i += 1;
                if write_msg(&mut w, &AgentMsg::Act(act)).is_err() {
                    return Ok(t);
                }
            }
            BridgeMsg::End(end) => {
                t.end = Some(format!("{:?} at frame {}, winner {:?}", end.reason, end.frame, end.winner));
                let end = ActLogEnd { obs: 0, decisions: 0, end_frame: end.frame, reason: end.reason, player: hello.slot, players_index: 0, policy: "test-forced".into(), git: uranium_agent::git_commit().into(), feature_schema: 0, action_schema: 0 };
                log.finish(end).map_err(|x| x.to_string())?;
                return Ok(t);
            }
            _ => {}
        }
    }
}

fn control(addr: &str, req: ControlRequest) -> Result<(BufReader<TcpStream>, TcpStream), String> {
    let s = TcpStream::connect(addr).map_err(|e| format!("hostd {addr}: {e}"))?;
    let mut r = BufReader::new(s.try_clone().map_err(|e| e.to_string())?);
    let mut w = s;
    write_msg(&mut w, &ControlRequest::Hello { control_version: CONTROL_VERSION }).map_err(|e| e.to_string())?;
    match read_msg(&mut r).map_err(|e| e.to_string())? {
        ControlResponse::Hello { .. } => {}
        other => return Err(format!("hostd handshake: {other:?}")),
    }
    write_msg(&mut w, &req).map_err(|e| e.to_string())?;
    Ok((r, w))
}

struct Agent {
    addr: String,
    board: BranchBoard,
    builds: T0Builds,
}

fn agent(listen: &str, out: &Path) -> Agent {
    let listener = TcpListener::bind(listen).expect("agent listener");
    let addr = listener.local_addr().unwrap().to_string();
    let (board, builds) = (BranchBoard::default(), T0Builds::default());
    let opts = ServeOptions { branches: board.clone(), t0_builds: builds.clone(), ..ServeOptions::new(uranium_agent::PolicySpec::ScriptedT0) };
    let sink = open_sink(&out.join("results.jsonl")).unwrap();
    std::thread::spawn(move || serve(listener, opts, sink));
    Agent { addr, board, builds }
}

fn play(a: &Agent, out: &Path, plan_game: u32, forced: Option<Forced>, tag: &str) -> (Option<Trace>, f64) {
    let hostd = std::env::var("URANIUM_TEST_HOSTD").expect("URANIUM_TEST_HOSTD=<hostd control address>");
    let game_id = 0x7B_0000_0000 + (std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as u64 & 0xFFFF_FFFF);
    a.board.lock().unwrap().insert(game_id, Arc::new(GameBranch { plan_game, acts_dir: Some(out.join("acts")), ..Default::default() }));
    a.builds.lock().unwrap().insert((game_id, 1), T0Build::ZergRush);
    let mut p0_addr = a.addr.clone();
    let mut forced_thread = None;
    if let Some(m) = forced {
        let ip = a.addr.rsplit_once(':').unwrap().0;
        let l = TcpListener::bind(format!("{ip}:0")).expect("forced listener");
        p0_addr = l.local_addr().unwrap().to_string();
        forced_thread = Some(std::thread::spawn(move || {
            let (s, _) = l.accept().expect("bridge connection");
            s.set_nodelay(true).ok();
            forced_session(s, m)
        }));
    }
    let spec = |race, agent_addr: &str| PlayerSpec { race, kind: PlayerKind::Bridge { agent_addr: agent_addr.to_string() }, name: None };
    let spec = GameSpec {
        game_id,
        map: MapKey::FightingSpirit,
        players: [spec(Race::Terran, &p0_addr), spec(Race::Zerg, &a.addr)],
        seed: Some(SEED),
        max_frames: MAX_FRAMES,
        privileged: false,
        save_replay: true,
        wall_timeout_s: 600,
    };
    let t0 = Instant::now();
    let (mut r, _w) = control(&hostd, ControlRequest::StartGame(spec)).expect("StartGame");
    match read_msg::<ControlResponse>(&mut r).expect("hostd reply") {
        ControlResponse::Started { .. } => {}
        other => panic!("hostd: {other:?}"),
    }
    let finished = loop {
        match read_msg::<ControlResponse>(&mut r).expect("hostd reply") {
            ControlResponse::Finished(f) if f.game_id == game_id => break f,
            _ => {}
        }
    };
    let wall = t0.elapsed().as_secs_f64();
    if let Some(rep) = &finished.replay {
        let _ = std::fs::write(out.join(format!("branch_{tag}.rep")), rep);
    }
    assert!(matches!(finished.outcome, GameOutcome::Finished), "game {tag} did not finish: {:?}", finished.outcome);
    let trace = forced_thread.map(|h| h.join().expect("forced session panicked").expect("forced session"));
    for p in 0..2u8 {
        let path = out.join("acts").join(act_log_name(plan_game, p));
        let deadline = Instant::now() + std::time::Duration::from_secs(30);
        while ActLog::read_path(&path).is_err() {
            assert!(Instant::now() < deadline, "act log {} incomplete", path.display());
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    }
    eprintln!("[branch {tag}] wall {wall:.1} s, forced {trace:?}");
    (trace, wall)
}

fn logs(out: &Path, plan_game: u32) -> (ActLog, ActLog) {
    let read = |p| ActLog::read_path(&out.join("acts").join(act_log_name(plan_game, p))).unwrap();
    (read(0), read(1))
}

#[test]
#[ignore = "needs a running uranium-hostd with OpenBW (URANIUM_TEST_HOSTD)"]
fn forced_prefix_reproduces_the_trunk_and_a_branch_diverges_after_it_051_ac3() {
    let out = std::env::var("URANIUM_TEST_BRANCH_OUT").map(PathBuf::from).unwrap_or_else(|_| std::env::temp_dir().join(format!("uranium-branch-replay-{}", std::process::id())));
    std::fs::create_dir_all(out.join("acts")).unwrap();
    let a = agent(&std::env::var("URANIUM_TEST_AGENT_LISTEN").unwrap_or_else(|_| "127.0.0.1:0".into()), &out);

    let (_, wa) = play(&a, &out, 1, None, "a_trunk");
    let (a0, a1) = logs(&out, 1);
    let forced = |branch, n| Forced { trunk: Arc::new(a0.clone()), branch, log: out.join("acts").join(act_log_name(n, 0)) };
    let (b, wb) = play(&a, &out, 2, Some(forced(u32::MAX, 2)), "b_forced");
    let (c, wc) = play(&a, &out, 3, Some(forced(BRANCH_FRAME, 3)), "c_branch");
    let (b, c) = (b.unwrap(), c.unwrap());
    let (b0, b1) = logs(&out, 2);
    let (c0, c1) = logs(&out, 3);

    let at = c.branched_at.expect("no non-empty Act after BRANCH_FRAME");
    let cmp = |x: &ActLog, y: &ActLog, before| compare_prefix(&x.lines, &y.lines, before);
    let rows: [(&str, f64, PrefixCompare, PrefixCompare, &Trace); 2] = [("B forced", wb, cmp(&a0, &b0, u32::MAX), cmp(&a1, &b1, u32::MAX), &b), ("C branch", wc, cmp(&a0, &c0, at + 1), cmp(&a1, &c1, at + 1), &c)];
    let mut md = format!("| game | wall s | p0 prefix (obs, matched, first mismatch) | p1 prefix | prefix mismatch | shadow mismatch | branched at |\n|---|---|---|---|---|---|---|\n| A trunk | {wa:.1} | {} obs | {} obs | | | |\n", a0.lines.len(), a1.lines.len());
    for (tag, w, p0, p1, t) in &rows {
        md += &format!("| {tag} | {w:.1} | {}, {}, {:?} | {}, {}, {:?} | {} | {} | {:?} |\n", p0.obs, p0.matched, p0.first_mismatch_frame, p1.obs, p1.matched, p1.first_mismatch_frame, t.prefix_mismatch, t.shadow_mismatch, t.branched_at);
    }
    eprintln!("{md}");
    let _ = std::fs::write(out.join("branch_replay.md"), &md);

    let (_, _, bp0, bp1, _) = &rows[0];
    assert_eq!((bp0.matched, bp0.first_mismatch_frame), (bp0.obs, None), "forced game differs for players[0]");
    assert_eq!((bp1.matched, bp1.first_mismatch_frame), (bp1.obs, None), "forced game differs for players[1] (opponent)");
    assert_eq!((b.prefix_mismatch, b.shadow_mismatch), (0, 0));
    assert_eq!(a0.end.end_frame, b0.end.end_frame);
    let (_, _, cp0, cp1, _) = &rows[1];
    assert_eq!((cp0.matched, cp1.matched), (cp0.obs, cp1.obs), "a difference before the branch");
    let (d0, d1) = (cp0.first_mismatch_frame.expect("branch game identical to the trunk"), cp1.first_mismatch_frame.expect("opponent unaffected by the branch"));
    assert!(d0 > at && d1 > at, "a difference before the branch: p0 {d0} p1 {d1} at {at}");
    assert_eq!((c.prefix_mismatch, c.shadow_mismatch), (0, 0));
}
