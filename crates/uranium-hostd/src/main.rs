//! Game host daemon (spec 020). `uranium-env` connects over TCP and asks it to start games;
//! each accepted game is reported back on the same connection when it ends.

mod config;
mod game;
mod platform;

use std::collections::{HashMap, HashSet};
use std::io::{BufReader, BufWriter};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use uranium_proto::control::*;
use uranium_proto::{read_msg, write_msg};

use crate::config::Config;
use crate::platform::Platform;

/// 020-R4: quarantine a game configuration after this many consecutive aborts.
const QUARANTINE_AFTER: u32 = 3;

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Serve control connections (settings from .env, 020-R9).
    Serve,
}

struct State {
    cfg: Config,
    platform: Box<dyn Platform>,
    running: AtomicU32,
    games: Mutex<HashSet<u64>>,
    stop_requests: Mutex<HashSet<u64>>,
    consecutive_aborts: Mutex<HashMap<String, u32>>,
}

fn load_env() {
    let res = match std::env::var_os("URANIUM_ENV_FILE") {
        Some(path) => dotenvy::from_path(path),
        None => dotenvy::dotenv().map(|_| ()),
    };
    if let Err(e) = res {
        if !e.not_found() {
            eprintln!("warning: failed to load .env: {e}");
        }
    }
}

fn main() -> Result<()> {
    load_env();
    match Cli::parse().cmd {
        Cmd::Serve => serve(),
    }
}

fn serve() -> Result<()> {
    let cfg = Config::from_env()?;
    let platform = platform::native(&cfg)?;
    let listener = TcpListener::bind(&cfg.listen).with_context(|| format!("binding {}", cfg.listen))?;
    eprintln!("[hostd] listening on {} (max {} games, oss bots {:?})", cfg.listen, cfg.max_games, cfg.oss_bots());
    let state = Arc::new(State {
        cfg,
        platform,
        running: AtomicU32::new(0),
        games: Mutex::new(HashSet::new()),
        stop_requests: Mutex::new(HashSet::new()),
        consecutive_aborts: Mutex::new(HashMap::new()),
    });
    for conn in listener.incoming() {
        let stream = conn?;
        stream.set_nodelay(true).ok();
        let st = state.clone();
        std::thread::spawn(move || {
            if let Err(e) = connection(stream, st) {
                eprintln!("[hostd] connection closed: {e:#}");
            }
        });
    }
    Ok(())
}

fn capabilities(st: &State) -> Capabilities {
    let maps = [MapKey::FightingSpirit, MapKey::CircuitBreaker].into_iter().filter(|m| st.cfg.maps_dir.join(m.file_name()).is_file()).collect();
    Capabilities {
        os: if cfg!(windows) { HostOs::Windows } else { HostOs::Linux },
        cores: std::thread::available_parallelism().map(|n| n.get() as u32).unwrap_or(1),
        max_concurrent_games: st.cfg.max_games,
        max_concurrent_replays: st.cfg.max_games,
        oss_bots: st.cfg.oss_bots(),
        maps,
    }
}

fn connection(stream: TcpStream, st: Arc<State>) -> Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let writer = Arc::new(Mutex::new(BufWriter::new(stream)));
    let send = |w: &Arc<Mutex<BufWriter<TcpStream>>>, m: &ControlResponse| -> Result<()> {
        write_msg(&mut *w.lock().unwrap(), m)?;
        Ok(())
    };
    loop {
        let req: ControlRequest = read_msg(&mut reader)?;
        match req {
            ControlRequest::Hello { control_version } => {
                if control_version != CONTROL_VERSION {
                    send(&writer, &ControlResponse::Rejected { reason: format!("control version {control_version} != {CONTROL_VERSION}") })?;
                    return Ok(());
                }
                send(&writer, &ControlResponse::Hello { control_version: CONTROL_VERSION, host_name_hint: String::new() })?;
            }
            ControlRequest::ListCapabilities => send(&writer, &ControlResponse::Capabilities(capabilities(&st)))?,
            ControlRequest::Status => {
                let running = st.games.lock().unwrap().iter().copied().collect();
                let quarantined = st.consecutive_aborts.lock().unwrap().iter().filter(|(_, n)| **n >= QUARANTINE_AFTER).map(|(k, _)| k.clone()).collect();
                send(&writer, &ControlResponse::Status { running, quarantined })?;
            }
            ControlRequest::StopGame { game_id } => {
                st.stop_requests.lock().unwrap().insert(game_id);
            }
            ControlRequest::StartReplay(job) => {
                // Replay jobs share the concurrency budget with games (one launcher each).
                if st.running.fetch_add(1, Ordering::SeqCst) >= st.cfg.max_games {
                    st.running.fetch_sub(1, Ordering::SeqCst);
                    send(&writer, &ControlResponse::Rejected { reason: "host at capacity".into() })?;
                    continue;
                }
                st.games.lock().unwrap().insert(job.job_id);
                send(&writer, &ControlResponse::Started { game_id: job.job_id })?;
                let (st2, w2) = (st.clone(), writer.clone());
                std::thread::spawn(move || {
                    let id = job.job_id;
                    let stop = || st2.stop_requests.lock().unwrap().contains(&id);
                    let fin = game::run_replay(&st2.cfg, st2.platform.as_ref(), &job, &stop).unwrap_or_else(|e| GameFinished {
                        game_id: id,
                        outcome: GameOutcome::Aborted { reason: format!("{e:#}") },
                        exit_codes: [None, None],
                        wall_s: 0,
                    });
                    st2.games.lock().unwrap().remove(&id);
                    st2.stop_requests.lock().unwrap().remove(&id);
                    st2.running.fetch_sub(1, Ordering::SeqCst);
                    eprintln!("[hostd] replay job {id}: {:?} in {}s", fin.outcome, fin.wall_s);
                    if let Err(e) = write_msg(&mut *w2.lock().unwrap(), &ControlResponse::Finished(fin)) {
                        eprintln!("[hostd] could not report replay job {id}: {e}");
                    }
                });
            }
            ControlRequest::StartGame(spec) => {
                let key = game::config_key(&spec);
                if st.consecutive_aborts.lock().unwrap().get(&key).copied().unwrap_or(0) >= QUARANTINE_AFTER {
                    send(&writer, &ControlResponse::Rejected { reason: format!("configuration quarantined after {QUARANTINE_AFTER} aborts: {key}") })?;
                    continue;
                }
                if st.running.fetch_add(1, Ordering::SeqCst) >= st.cfg.max_games {
                    st.running.fetch_sub(1, Ordering::SeqCst);
                    send(&writer, &ControlResponse::Rejected { reason: "host at capacity".into() })?;
                    continue;
                }
                st.games.lock().unwrap().insert(spec.game_id);
                send(&writer, &ControlResponse::Started { game_id: spec.game_id })?;
                let (st2, w2) = (st.clone(), writer.clone());
                std::thread::spawn(move || {
                    let id = spec.game_id;
                    let stop = || st2.stop_requests.lock().unwrap().contains(&id);
                    let fin = game::run(&st2.cfg, st2.platform.as_ref(), &spec, &stop).unwrap_or_else(|e| GameFinished {
                        game_id: id,
                        outcome: GameOutcome::Aborted { reason: format!("{e:#}") },
                        exit_codes: [None, None],
                        wall_s: 0,
                    });
                    {
                        let mut aborts = st2.consecutive_aborts.lock().unwrap();
                        let n = aborts.entry(key.clone()).or_insert(0);
                        if matches!(fin.outcome, GameOutcome::Aborted { .. }) {
                            *n += 1;
                            if *n == QUARANTINE_AFTER {
                                eprintln!("[hostd] ALERT: quarantining configuration {key} after {QUARANTINE_AFTER} consecutive aborts");
                            }
                        } else {
                            *n = 0;
                        }
                    }
                    st2.games.lock().unwrap().remove(&id);
                    st2.stop_requests.lock().unwrap().remove(&id);
                    st2.running.fetch_sub(1, Ordering::SeqCst);
                    eprintln!("[hostd] game {id}: {:?} in {}s", fin.outcome, fin.wall_s);
                    if let Err(e) = write_msg(&mut *w2.lock().unwrap(), &ControlResponse::Finished(fin)) {
                        eprintln!("[hostd] could not report game {id}: {e}");
                    }
                });
            }
        }
    }
}
