//! Branch-game feasibility (`reports/jepa_planning.md` 7, design only — no spec yet): a game can be
//! re-played up to a decision point by forcing one player's recorded `Act`s, and continued from
//! there with a different action, because OpenBW is lockstep-deterministic for the same seed and
//! the same `Act` sequences (011-AC4) and the T0 bots are deterministic in their observations
//! (`react.rs`: "No randomness").
//!
//! Three games on one hostd, same seed, T0 Terran standard (player 0, the "trunk policy") against
//! T0 Zerg 5-pool rush (player 1, the opponent):
//! - A (trunk): both bots live; player 0's `Act` per observation is recorded.
//! - B (forced): player 0 sends A's recorded `Act`s for the whole game; its bot still runs on every
//!   observation (shadow) so its state is rebuilt the way a model's recurrent state would be.
//!   Expect every observation of both players identical to A.
//! - C (branch): forced until the first non-empty recorded `Act` at or after `BRANCH_FRAME`; that
//!   one is replaced by an empty `Act` with the same advance (the alternative "do nothing"); then
//!   the shadowed bot plays live. Expect both players identical to A up to the branch observation
//!   and different after it.
//!
//! Needs a running `uranium-hostd` (as `uranium-env`'s `openbw_actions`), so it is ignored:
//!
//! ```text
//! URANIUM_TEST_HOSTD=<host:port> [URANIUM_TEST_AGENT_LISTEN=127.0.0.1:0] [URANIUM_TEST_BRANCH_OUT=<dir>] \
//!   cargo test -p uranium-scripted --test branch_replay -- --ignored --nocapture
//! ```

use std::io::BufReader;
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::time::Instant;

use uranium_proto::control::{CONTROL_VERSION, ControlRequest, ControlResponse, GameOutcome, GameSpec, MapKey, PlayerKind, PlayerSpec};
use uranium_proto::*;
use uranium_scripted::{T0Build, for_game_build};

/// 5 game minutes: the 5-pool zerglings arrive around 2:30.
const MAX_FRAMES: u32 = 24 * 60 * 5;
/// 1:30 (before the rush arrives; the Terran is still building its economy).
const BRANCH_FRAME: u32 = 24 * 90;
const SEED: u32 = 70_002;

/// What player 0 does.
#[derive(Clone)]
enum Mode {
    Live,
    /// Send `acts` (one per observation, with the frame it was sent at) up to the branch; at the
    /// first observation with frame >= `branch` whose recorded `Act` has commands, send an empty
    /// `Act` instead; afterwards play the bot live. `branch = u32::MAX` forces the whole game.
    Forced { acts: Vec<(u32, Act)>, branch: u32 },
}

#[derive(Default, Debug)]
struct Trace {
    /// (frame, checksum) per observation.
    sums: Vec<(u32, u64)>,
    /// Player 0: (frame, Act sent) per observation.
    acts: Vec<(u32, Act)>,
    /// Player 0 in forced mode: observations where the recorded frame differed from the live one.
    frame_mismatch: u32,
    /// Player 0 in forced mode: forced observations where the shadow bot's own `Act` differed from
    /// the recorded one (its state was not rebuilt).
    shadow_mismatch: u32,
    /// Frame of the observation whose `Act` was replaced (branch game).
    branched_at: Option<u32>,
    end: Option<String>,
    last: Option<(u32, PlayerState, usize)>,
}

/// FNV-1a over the observed units (id order) and own resources: equal sequences = the same game
/// from this player's view (the 011-AC4 checksum of `uranium-env/tests/openbw_actions.rs`).
fn checksum(f: &RawFrame) -> u64 {
    let mut units: Vec<&UnitState> = f.units.iter().collect();
    units.sort_by_key(|u| u.id);
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |v: u64| {
        for b in v.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    };
    eat(f.frame as u64);
    for u in units {
        for v in [u.id as u64, u.unit_type as u64, u.pos.0 as u64, u.pos.1 as u64, u.hp as u64, u.shield as u64, u.energy as u64, u.order as u64, u.order_target.map_or(u64::MAX, |t| t as u64), u.flags as u64] {
            eat(v);
        }
    }
    for v in [f.self_state.minerals as u64, f.self_state.gas as u64, f.self_state.supply_used as u64, f.self_state.supply_total as u64] {
        eat(v);
    }
    h
}

fn session(stream: TcpStream, mode: Mode) -> Result<(u8, Trace), String> {
    let e = |x: ProtoError| x.to_string();
    let mut r = BufReader::new(stream.try_clone().map_err(|x| x.to_string())?);
    let mut w = stream;
    let BridgeMsg::Hello(hello) = read_msg(&mut r).map_err(e)? else { return Err("expected Hello".into()) };
    let BridgeMsg::GameInfo(info) = read_msg(&mut r).map_err(e)? else { return Err("expected GameInfo".into()) };
    let slot = hello.slot;
    let race = info.players.iter().find(|p| p.is_self).map(|p| p.race).unwrap_or(Race::Terran);
    let p0 = race == Race::Terran;
    let mut bot = for_game_build(&info, if p0 { T0Build::Standard } else { T0Build::ZergRush });
    let mut t = Trace::default();
    let mut i = 0usize;
    let mut branched = false;
    loop {
        let msg = match read_msg::<BridgeMsg>(&mut r) {
            Ok(m) => m,
            Err(x) => {
                t.end.get_or_insert(format!("connection closed: {x}"));
                break;
            }
        };
        match msg {
            BridgeMsg::Frame(f) => {
                t.sums.push((f.frame, checksum(&f)));
                t.last = Some((f.frame, f.self_state.clone(), f.units.iter().filter(|u| u.owner == info.self_player).count()));
                // The bot sees every observation in every mode (shadow while forced).
                let own = bot.act(&f);
                let act = match (&mode, p0) {
                    (Mode::Forced { acts, branch }, true) if !branched => match acts.get(i) {
                        Some((frame, rec)) => {
                            if *frame != f.frame {
                                t.frame_mismatch += 1;
                            }
                            if f.frame >= *branch && !rec.cmds.is_empty() {
                                branched = true;
                                t.branched_at = Some(f.frame);
                                Act { cmds: vec![], advance_frames: rec.advance_frames, valid_until_frame: None, surrender: None }
                            } else {
                                if own != *rec {
                                    t.shadow_mismatch += 1;
                                }
                                rec.clone()
                            }
                        }
                        // The trunk ended earlier (should not happen with the same max_frames).
                        None => own,
                    },
                    _ => own,
                };
                if p0 {
                    t.acts.push((f.frame, act.clone()));
                }
                i += 1;
                if write_msg(&mut w, &AgentMsg::Act(act)).is_err() {
                    break;
                }
            }
            BridgeMsg::End(end) => {
                t.end = Some(format!("{:?} at frame {}, winner {:?}", end.reason, end.frame, end.winner));
                break;
            }
            _ => {}
        }
    }
    Ok((slot, t))
}

/// One request on a fresh hostd control connection; hostd answers on it until the game ends, so
/// the caller keeps both halves.
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

/// One game: returns (player 0 trace, player 1 trace, wall seconds).
fn play(mode: Mode, tag: &str) -> (Trace, Trace, f64) {
    let hostd = std::env::var("URANIUM_TEST_HOSTD").expect("URANIUM_TEST_HOSTD=<hostd control address>");
    let listen = std::env::var("URANIUM_TEST_AGENT_LISTEN").unwrap_or_else(|_| "127.0.0.1:0".into());
    let listener = TcpListener::bind(&listen).expect("agent listener");
    let agent_addr = listener.local_addr().unwrap().to_string();
    let game_id = 0x7B_0000_0000 + (std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as u64 & 0xFFFF_FFFF);
    let spec = |race| PlayerSpec { race, kind: PlayerKind::Bridge { agent_addr: agent_addr.clone() }, name: None };
    let spec = GameSpec {
        game_id,
        map: MapKey::FightingSpirit,
        players: [spec(Race::Terran), spec(Race::Zerg)],
        seed: Some(SEED),
        max_frames: MAX_FRAMES,
        privileged: false,
        save_replay: true,
        wall_timeout_s: 600,
    };
    let sessions = std::thread::spawn(move || {
        let mut handles = vec![];
        for _ in 0..2 {
            let (s, _) = listener.accept().expect("bridge connection");
            s.set_nodelay(true).ok();
            let m = mode.clone();
            handles.push(std::thread::spawn(move || session(s, m)));
        }
        handles.into_iter().map(|h| h.join().expect("session panicked")).collect::<Vec<_>>()
    });
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
    if let (Some(rep), Ok(dir)) = (&finished.replay, std::env::var("URANIUM_TEST_BRANCH_OUT")) {
        let _ = std::fs::write(PathBuf::from(dir).join(format!("branch_{tag}.rep")), rep);
    }
    assert!(matches!(finished.outcome, GameOutcome::Finished), "game {tag} did not finish: {:?}", finished.outcome);
    let mut traces = sessions.join().expect("agent thread").into_iter().map(|x| x.expect("session")).collect::<Vec<_>>();
    traces.sort_by_key(|(slot, _)| *slot);
    let mut it = traces.into_iter().map(|(_, t)| t);
    let (a, b) = (it.next().unwrap(), it.next().unwrap());
    // Player 0 is the Terran (slot order follows the GameSpec order, 011-AC4).
    let (p0, p1) = if a.acts.is_empty() { (b, a) } else { (a, b) };
    eprintln!("[branch {tag}] wall {wall:.1} s, p0 {} obs, p1 {} obs, end {:?}", p0.sums.len(), p1.sums.len(), p0.end);
    (p0, p1, wall)
}

/// Index of the first observation that differs (or the shorter length when one is a prefix).
fn first_diff(a: &[(u32, u64)], b: &[(u32, u64)]) -> Option<usize> {
    a.iter().zip(b).position(|(x, y)| x != y).or((a.len() != b.len()).then(|| a.len().min(b.len())))
}

#[test]
#[ignore = "needs a running uranium-hostd with OpenBW (URANIUM_TEST_HOSTD)"]
fn forced_prefix_reproduces_the_trunk_and_a_branch_diverges_after_it() {
    let (a0, a1, wa) = play(Mode::Live, "a_trunk");
    let (b0, b1, wb) = play(Mode::Forced { acts: a0.acts.clone(), branch: u32::MAX }, "b_forced");
    let (c0, c1, wc) = play(Mode::Forced { acts: a0.acts.clone(), branch: BRANCH_FRAME }, "c_branch");

    let sent = |t: &Trace| t.acts.iter().map(|(_, a)| a.cmds.len()).sum::<usize>();
    let mut md = String::from("| game | wall s | p0 obs | p1 obs | p0 cmds | first diff vs A (p0 / p1, frame) | frame mismatch | shadow mismatch | branched at | p0 last (frame, minerals, supply, units) | end |\n|---|---|---|---|---|---|---|---|---|---|---|\n");
    for (tag, p0, p1, w) in [("A trunk", &a0, &a1, wa), ("B forced", &b0, &b1, wb), ("C branch", &c0, &c1, wc)] {
        let d0 = first_diff(&a0.sums, &p0.sums).map(|i| p0.sums.get(i).or(a0.sums.get(i)).map_or(0, |s| s.0));
        let d1 = first_diff(&a1.sums, &p1.sums).map(|i| p1.sums.get(i).or(a1.sums.get(i)).map_or(0, |s| s.0));
        let last = p0.last.as_ref().map(|(f, s, n)| format!("{f}, {}, {}/{}, {n}", s.minerals, s.supply_used, s.supply_total)).unwrap_or_default();
        md += &format!(
            "| {tag} | {w:.1} | {} | {} | {} | {d0:?} / {d1:?} | {} | {} | {:?} | {last} | {} |\n",
            p0.sums.len(),
            p1.sums.len(),
            sent(p0),
            p0.frame_mismatch,
            p0.shadow_mismatch,
            p0.branched_at,
            p0.end.clone().unwrap_or_default()
        );
    }
    eprintln!("{md}");
    if let Ok(dir) = std::env::var("URANIUM_TEST_BRANCH_OUT") {
        let _ = std::fs::write(PathBuf::from(dir).join("branch_replay.md"), &md);
    }

    // B: forcing A's Acts reproduces A for both players, and the shadow bot agreed every time.
    assert_eq!(first_diff(&a0.sums, &b0.sums), None, "forced game differs for player 0");
    assert_eq!(first_diff(&a1.sums, &b1.sums), None, "forced game differs for player 1 (opponent)");
    assert_eq!((b0.frame_mismatch, b0.shadow_mismatch), (0, 0));
    assert_eq!(a0.end, b0.end);
    // C: identical through the branch observation, different after it.
    let at = c0.branched_at.expect("no non-empty Act after BRANCH_FRAME");
    let i0 = first_diff(&a0.sums, &c0.sums).expect("branch game identical to the trunk");
    let i1 = first_diff(&a1.sums, &c1.sums).expect("opponent unaffected by the branch");
    assert!(c0.sums[i0].0 > at && c1.sums[i1].0 > at, "a difference before the branch: p0 {:?} p1 {:?} at {at}", c0.sums[i0], c1.sums[i1]);
    assert_eq!((c0.frame_mismatch, c0.shadow_mismatch), (0, 0));
}
