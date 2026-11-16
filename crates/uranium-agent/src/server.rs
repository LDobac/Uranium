use std::collections::BTreeMap;
use std::fs::OpenOptions;
use std::io::{BufReader, BufWriter, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use anyhow::{Context, Result, bail};
use uranium_proto::*;
use uranium_store::elog;
use uranium_env::branch::{BranchBoard, GameBranch, fnv1a};
use super::{PolicySpec, Policy, REALTIME_ACT_FRAMES, make_policy};
use super::routing::{Routes, SessionEnd, EndBoard, ConcedeBoard, T0Builds, spec_index};
use super::act_log::{ActLogStatus, ActLogFile};
use super::record::{git_commit, SessionRecord, revealed_enemy_race, visibility_violations, RecordSink, open_sink};
use super::rejection::{RejectionLog, record_unit_rejections};

pub fn end_outcome(end: &GameEnd, me: u8) -> Option<uranium_env::reward::Outcome> {
    use uranium_env::reward::Outcome;
    match end.reason {
        EndReason::Aborted | EndReason::AgentTimeout => None,
        EndReason::TimeLimitDraw => Some(Outcome::Draw),
        EndReason::Elimination | EndReason::Surrender => Some(match end.winner {
            Some(w) if w == me => Outcome::Win,
            Some(_) => Outcome::Loss,
            None => Outcome::Draw,
        }),
    }
}

#[derive(Clone)]
pub struct ServeOptions {
    pub default: PolicySpec,
    pub routes: Routes,
    pub delay_ms: u64,
    pub max_sessions: usize,
    pub t0_builds: T0Builds,
    pub ends: EndBoard,
    pub unit_rejections: RejectionLog,
    pub concede: ConcedeBoard,
    pub branches: BranchBoard,
}

impl ServeOptions {
    pub fn new(default: PolicySpec) -> Self {
        Self { default, routes: Routes::default(), delay_ms: 0, max_sessions: 0, t0_builds: T0Builds::default(), ends: EndBoard::default(), unit_rejections: RejectionLog::default(), concede: ConcedeBoard::default(), branches: BranchBoard::default() }
    }
}

pub fn serve_addr(listen: &str, opts: ServeOptions, out: PathBuf) -> Result<()> {
    let listener = TcpListener::bind(listen).with_context(|| format!("binding {listen}"))?;
    elog!("[agent] listening on {listen}");
    serve(listener, opts, open_sink(&out)?)
}

pub fn serve(listener: TcpListener, opts: ServeOptions, sink: RecordSink) -> Result<()> {
    let max_sessions = opts.max_sessions;
    let finished = Arc::new(Mutex::new(0usize));
    let mut handles = Vec::new();
    for conn in listener.incoming() {
        let stream = conn?;
        stream.set_nodelay(true).ok();
        let (sink, done, opts) = (sink.clone(), finished.clone(), opts.clone());
        handles.push(std::thread::spawn(move || {
            let rec = session(stream, &opts);
            let line = serde_json::to_string(&rec).expect("record serializes");
            let mut s = sink.lock().unwrap();
            writeln!(s, "{line}").ok();
            s.flush().ok();
            *done.lock().unwrap() += 1;
        }));
        handles.retain(|h| !h.is_finished());
        if max_sessions > 0 && *finished.lock().unwrap() >= max_sessions {
            break;
        }
    }
    for h in handles {
        h.join().ok();
    }
    Ok(())
}

pub fn session(stream: TcpStream, opts: &ServeOptions) -> SessionRecord {
    let started = Instant::now();
    let mut rec = SessionRecord {
        game_id: 0,
        slot: 0,
        map: String::new(),
        map_hash: String::new(),
        self_race: String::new(),
        enemy_race: String::new(),
        enemy_race_lobby: String::new(),
        t0_build: String::new(),
        winner: None,
        won: None,
        reason: "Disconnected".into(),
        end_frame: 0,
        wall_s: 0.0,
        policy_ms: 0.0,
        stats: None,
        first_frame_minerals: 0,
        first_frame_units: 0,
        commands: 0,
        failed_commands: 0,
        unit_rejections: 0,
        unit_rejections_by: BTreeMap::new(),
        frame_digest: String::new(),
        privileged_msgs: 0,
        visibility_violations: 0,
        final_frame: None,
        policy: String::new(),
        model: None,
        error: None,
        scripted: None,
        act_log: None,
        snapshots: Vec::new(),
        branch: None,
        infer_seed: None,
    };
    let mut rejections = Vec::new();
    if let Err(e) = run_session(stream, opts, &mut rec, &mut rejections) {
        elog!("[agent] game {} slot {}: {e:#}", rec.game_id, rec.slot);
        rec.error = Some(format!("{e:#}"));
    }
    rec.wall_s = started.elapsed().as_secs_f64();
    if let Err(e) = opts.unit_rejections.append(&rejections) {
        elog!("[agent] game {} slot {}: writing unit rejections: {e:#}", rec.game_id, rec.slot);
    }
    rec
}

fn run_session(stream: TcpStream, opts: &ServeOptions, rec: &mut SessionRecord, rejections: &mut Vec<String>) -> Result<()> {
    let delay_ms = opts.delay_ms;
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut writer = BufWriter::new(stream);
    let BridgeMsg::Hello(hello) = read_msg(&mut reader)? else { bail!("expected Hello") };
    rec.game_id = hello.game_id;
    rec.slot = hello.slot;
    if let Err(why) = check_hello(&hello) {
        write_msg(&mut writer, &AgentMsg::Reject(why.clone()))?;
        bail!(why);
    }
    opts.ends.hello(hello.game_id);
    let BridgeMsg::GameInfo(info) = read_msg(&mut reader)? else { bail!("expected GameInfo") };
    rec.map = info.map_name.clone();
    rec.map_hash = info.map_hash.iter().map(|b| format!("{b:02x}")).collect();
    rec.self_race = info.players.iter().find(|p| p.is_self).map(|p| format!("{:?}", p.race)).unwrap_or_default();
    rec.enemy_race = info.players.iter().find(|p| p.is_enemy).map(|p| format!("{:?}", p.race)).unwrap_or_default();
    rec.enemy_race_lobby = rec.enemy_race.clone();
    elog!("[agent] game {} slot {} on {:?} ({} start locations)", hello.game_id, hello.slot, info.map_name, info.static_map.start_locations.len());
    let index = spec_index(&info, hello.slot);
    let spec = opts.routes.take(hello.game_id, index).unwrap_or_else(|| opts.default.clone());
    rec.policy = spec.label();
    let build = opts.t0_builds.lock().unwrap().get(&(hello.game_id, index)).copied().unwrap_or_default();
    rec.t0_build = build.name().to_string();
    let concede = opts.concede.get(hello.game_id).filter(|_| hello.mode == Mode::Lockstep).map(|g| (g, index));
    let game_branch = if hello.mode == Mode::Lockstep { opts.branches.lock().unwrap().get(&hello.game_id).cloned() } else { None };
    let mut policy = make_policy(&spec, hello.game_id, &info, build, hello.mode == Mode::Realtime, concede, game_branch.clone().map(|g| (g, index)))?;
    let mut log = SessionLog { act_log: game_branch.as_ref().and_then(|g| g.act_log_path(index)).map(ActLogFile::create), index, end: None };
    let res = play(&mut reader, &mut writer, &mut *policy, &hello, rec, rejections, delay_ms, &opts.ends, &mut log);
    if res.is_err() {
        policy.on_abort();
        if let Some(l) = log.act_log.as_mut() {
            l.abort();
        }
    }
    if let Some(g) = &game_branch {
        log.record(g, &mut *policy, hello.slot, rec);
        if let PolicySpec::Model(h) = &spec {
            rec.infer_seed = h.infer_seed;
        }
    }
    res
}

struct SessionLog {
    act_log: Option<ActLogFile>,
    index: u8,
    end: Option<(Option<u8>, u32, EndReason)>,
}

impl SessionLog {
    fn finish(&mut self, winner: Option<u8>, frame: u32, reason: EndReason, slot: u8, policy: &str) {
        self.end = Some((winner, frame, reason));
        if let Some(l) = self.act_log.as_mut() {
            l.finish(uranium_env::branch::ActLogEnd {
                obs: 0,
                decisions: 0,
                end_frame: frame,
                reason,
                player: slot,
                players_index: self.index,
                policy: policy.to_string(),
                git: git_commit().to_string(),
                feature_schema: uranium_env::features::FEATURE_SCHEMA_VERSION,
                action_schema: uranium_env::actions::ACTION_SCHEMA_VERSION,
            });
        }
    }

    fn record(&self, game: &GameBranch, policy: &mut dyn Policy, slot: u8, rec: &mut SessionRecord) {
        rec.act_log = self.act_log.as_ref().map(|l| ActLogStatus { path: l.path.display().to_string(), complete: l.complete, error: l.error.clone() });
        let Some(part) = policy.branch_record() else { return };
        rec.snapshots = part.snapshots;
        rec.branch = part.outcome.map(|mut o| {
            if let (Some((winner, frame, reason)), Some(b)) = (self.end, &game.branch)
                && !matches!(reason, EndReason::Aborted | EndReason::AgentTimeout)
            {
                o.end = Some(uranium_env::branch::BranchEnd::new(winner, slot, frame, reason, b.horizon));
            }
            if let Err(e) = uranium_env::branch::FiniteRecord::check_finite(&o) {
                elog!("[agent] WARNING: 051-R9 branch outcome of game {} has a value JSON cannot hold: {e:#}", rec.game_id);
            }
            o
        });
    }
}

#[allow(clippy::too_many_arguments)]
fn play(reader: &mut BufReader<TcpStream>, writer: &mut BufWriter<TcpStream>, policy: &mut dyn Policy, hello: &Hello, rec: &mut SessionRecord, rejections: &mut Vec<String>, delay_ms: u64, ends: &EndBoard, log: &mut SessionLog) -> Result<()> {
    let (mut reader, mut writer) = (reader, writer);
    let realtime = hello.mode == Mode::Realtime;
    let mut policy_time = Duration::ZERO;
    let mut first = true;
    let mut digest: u64 = uranium_env::branch::FNV_OFFSET;
    let mut obs_index: u32 = 0;
    let mut last_cmds: Vec<Command> = Vec::new();
    loop {
        match read_msg::<BridgeMsg>(&mut reader)? {
            BridgeMsg::Privileged(p) => {
                rec.privileged_msgs += 1;
                if realtime {
                    elog!("[agent] game {} slot {}: PrivilegedSummary in a realtime session ignored (010-R10)", hello.game_id, hello.slot);
                } else {
                    policy.on_privileged(&p)
                }
            }
            BridgeMsg::Frame(f) => {
                let bytes = uranium_env::branch::frame_bytes(&f)?;
                digest = fnv1a(digest, &bytes);
                let i = obs_index;
                obs_index += 1;
                if let Ok(dir) = std::env::var("URANIUM_AGENT_DUMP") {
                    let max: u32 = std::env::var("URANIUM_AGENT_DUMP_MAX_FRAME").ok().and_then(|v| v.parse().ok()).unwrap_or(240);
                    if f.frame < max {
                        let path = format!("{dir}/g{}-s{}.txt", hello.game_id, hello.slot);
                        if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
                            writeln!(file, "{f:#?}").ok();
                        }
                    }
                }
                if first {
                    rec.first_frame_units = f.units.len();
                    rec.first_frame_minerals = f.units.iter().filter(|u| uranium_scripted::types::is_mineral(u.unit_type)).count();
                    first = false;
                }
                if matches!(rec.enemy_race.as_str(), "Unknown" | "Random") {
                    if let Some(r) = revealed_enemy_race(&f, hello.slot) {
                        rec.enemy_race = r;
                    }
                }
                rec.visibility_violations += visibility_violations(&f, hello.slot);
                rec.failed_commands += f.events.iter().filter(|e| matches!(e, GameEvent::CommandFailed { .. })).count() as u64;
                record_unit_rejections(&f, &last_cmds, hello, rec, rejections);
                if let Ok(dir) = std::env::var("URANIUM_AGENT_DUMP") {
                    for e in &f.events {
                        if let GameEvent::CommandFailed { cmd_index, error } = e {
                            let path = format!("{dir}/g{}-s{}-failed.txt", hello.game_id, hello.slot);
                            if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
                                writeln!(file, "{} error {error} {:?}", f.frame, last_cmds.get(*cmd_index as usize)).ok();
                            }
                        }
                    }
                }
                let t = Instant::now();
                let mut act = match policy.on_frame(&f) {
                    Ok(a) => a,
                    Err(e) => {
                        rec.model = policy.model_stats();
                        return Err(e);
                    }
                };
                policy_time += t.elapsed();
                rec.commands += act.cmds.len() as u64;
                last_cmds.clone_from(&act.cmds);
                if delay_ms > 0 {
                    std::thread::sleep(Duration::from_millis(delay_ms));
                }
                if realtime {
                    act.valid_until_frame = Some(f.frame + REALTIME_ACT_FRAMES);
                }
                if let Some(l) = log.act_log.as_mut() {
                    let x = policy.act_log_extra().unwrap_or_default();
                    l.push(&uranium_env::branch::ObsLine {
                        i,
                        frame: f.frame,
                        digest: uranium_env::branch::obs_digest_bytes(&bytes),
                        act: act.clone(),
                        decision: x.decision,
                        state: x.state,
                        state_sha256: x.state_sha256,
                        forced: x.forced,
                    });
                }
                write_msg(&mut writer, &AgentMsg::Act(act))?;
            }
            BridgeMsg::End(end) => {
                let session_end = SessionEnd { slot: hello.slot, winner: end.winner, frame: end.frame, reason: end.reason };
                let outcome = end_outcome(&end, hello.slot);
                rec.winner = end.winner;
                rec.won = end.winner.map(|w| w == hello.slot);
                rec.reason = format!("{:?}", end.reason);
                rec.end_frame = end.frame;
                rec.stats = end.stats;
                rec.policy_ms = policy_time.as_secs_f64() * 1e3;
                rec.frame_digest = format!("{digest:016x}");
                ends.push(hello.game_id, session_end);
                if let Some(last) = &end.final_frame {
                    rec.visibility_violations += visibility_violations(last, hello.slot);
                    rec.final_frame = Some((last.frame, last.units.len(), last.events.len()));
                    record_unit_rejections(last, &last_cmds, hello, rec, rejections);
                }
                policy.on_end(outcome, end.final_frame.as_deref());
                rec.model = policy.model_stats();
                rec.scripted = policy.scripted_stats();
                log.finish(end.winner, end.frame, end.reason, hello.slot, &rec.policy);
                return Ok(());
            }
            other => bail!("unexpected message {other:?}"),
        }
    }
}
