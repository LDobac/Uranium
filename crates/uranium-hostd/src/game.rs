//! Runs one 1v1 OpenBW game: two BWAPILauncher processes (spec 020-R2/R3/R4, 011).

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use uranium_proto::Race;
use uranium_proto::control::{GameFinished, GameOutcome, GameSpec, PlayerKind};

use crate::config::{Config, oss_module};
use crate::platform::Platform;

pub fn race_name(r: Race) -> &'static str {
    match r {
        Race::Terran => "Terran",
        Race::Protoss => "Protoss",
        Race::Zerg => "Zerg",
        Race::Random | Race::Unknown => "Random",
    }
}

/// Environment for one launcher (mirrors tools/openbw/selfmatch.sh).
pub fn launcher_env(cfg: &Config, spec: &GameSpec, slot: usize, module: &Path, replay_dir: &Path) -> Vec<(String, String)> {
    let p = &spec.players[slot];
    let map = cfg.maps_dir.join(spec.map.file_name());
    let mut env = vec![
        ("URANIUM_DATA_DIR", cfg.data_dir.display().to_string()),
        ("OPENBW_ENABLE_UI", "0".to_string()),
        ("BWAPI_CONFIG_AI__AI", module.display().to_string()),
        ("BWAPI_CONFIG_AUTO_MENU__AUTO_MENU", "LAN".to_string()),
        ("BWAPI_CONFIG_AUTO_MENU__MAP", map.display().to_string()),
        ("BWAPI_CONFIG_AUTO_MENU__RACE", race_name(p.race).to_string()),
        ("BWAPI_CONFIG_AUTO_MENU__GAME_TYPE", "MELEE".to_string()),
        ("BWAPI_CONFIG_AUTO_MENU__WAIT_FOR_MIN_PLAYERS", "2".to_string()),
        ("BWAPI_CONFIG_AUTO_MENU__AUTO_RESTART", "OFF".to_string()),
        (
            "BWAPI_CONFIG_AUTO_MENU__SAVE_REPLAY",
            if spec.save_replay { replay_dir.join(format!("g{}-p{slot}.rep", spec.game_id)).display().to_string() } else { String::new() },
        ),
        ("URANIUM_GAME_ID", spec.game_id.to_string()),
        // 0004 patch: deterministic client ids => launcher slot = player slot (011-AC4).
        ("URANIUM_LAN_SLOT", slot.to_string()),
        ("URANIUM_MAX_FRAMES", spec.max_frames.to_string()),
        ("URANIUM_PRIVILEGED", if spec.privileged { "1" } else { "0" }.to_string()),
    ];
    if let PlayerKind::Bridge { agent_addr } = &p.kind {
        env.push(("URANIUM_AGENT_ADDR", agent_addr.clone()));
    }
    let mut env: Vec<(String, String)> = env.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
    // 0002 patch: every client must see the same seed for a deterministic setup (011-AC4).
    if let Some(seed) = spec.seed {
        env.push(("URANIUM_SEED".into(), seed.to_string()));
    }
    env
}

fn module_for(cfg: &Config, kind: &PlayerKind) -> Result<(PathBuf, Option<PathBuf>)> {
    match kind {
        PlayerKind::Bridge { .. } => Ok((cfg.bridge_lib.clone(), None)),
        PlayerKind::OssBot { name } => {
            let dir = cfg.oss_bot_dir(name).with_context(|| format!("unknown OSS bot `{name}`"))?;
            Ok((oss_module(&dir).expect("checked by oss_bot_dir"), Some(dir.join("bwapi-data"))))
        }
    }
}

fn copy_dir(src: &Path, dst: &Path) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    for e in std::fs::read_dir(src)? {
        let e = e?;
        let to = dst.join(e.file_name());
        if e.file_type()?.is_dir() {
            copy_dir(&e.path(), &to)?;
        } else {
            std::fs::copy(e.path(), to)?;
        }
    }
    Ok(())
}

fn tail(path: &Path, lines: usize) -> String {
    let text = std::fs::read_to_string(path).unwrap_or_default();
    let v: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    v[v.len().saturating_sub(lines)..].join(" | ")
}

/// Plays one game to the end (or timeout). `stop` is polled to support `StopGame`.
pub fn run(cfg: &Config, platform: &dyn Platform, spec: &GameSpec, stop: &dyn Fn() -> bool) -> Result<GameFinished> {
    let started = Instant::now();
    let dir = cfg.work_dir.join(format!("g{}", spec.game_id));
    let _ = std::fs::remove_dir_all(&dir);
    let lan = platform.lan_setup(spec.game_id)?;
    let mut children: Vec<Child> = Vec::new();
    let mut logs = Vec::new();
    let launch = (|| -> Result<()> {
        for slot in 0..2 {
            let pdir = dir.join(format!("p{slot}"));
            std::fs::create_dir_all(pdir.join("bwapi-data"))?;
            let (module, data) = module_for(cfg, &spec.players[slot].kind)?;
            if let Some(data) = data.filter(|d| d.is_dir()) {
                copy_dir(&data, &pdir.join("bwapi-data"))?;
            }
            let log = pdir.join("launcher.log");
            let out = std::fs::File::create(&log)?;
            let mut cmd = Command::new(cfg.launcher());
            cmd.current_dir(&pdir).stdin(Stdio::null()).stdout(out.try_clone()?).stderr(out);
            cmd.envs(launcher_env(cfg, spec, slot, &module, &dir));
            cmd.envs(lan.env[slot].iter().map(|(k, v)| (k, v)));
            children.push(platform.spawn(cmd).with_context(|| format!("spawning launcher {slot}"))?);
            logs.push(log);
            // The first client binds the LAN endpoint and becomes player slot 0; wait for it so
            // slot assignment never depends on scheduling (011-AC4).
            if slot == 0 && !platform.wait_first_ready(&lan, Duration::from_secs(20)) {
                anyhow::bail!("first launcher did not open its LAN endpoint within 20 s");
            }
        }
        Ok(())
    })();

    let timeout = Duration::from_secs(spec.wall_timeout_s as u64);
    let mut codes: [Option<i32>; 2] = [None, None];
    let mut done = [false, false];
    let mut outcome = None;
    if let Err(e) = launch {
        outcome = Some(GameOutcome::Aborted { reason: format!("{e:#}") });
    }
    while outcome.is_none() {
        for (i, c) in children.iter_mut().enumerate() {
            if !done[i] {
                if let Some(st) = c.try_wait()? {
                    done[i] = true;
                    codes[i] = st.code();
                }
            }
        }
        if done.iter().take(children.len()).all(|d| *d) && children.len() == 2 {
            let ok = codes.iter().all(|c| *c == Some(0));
            outcome = Some(if ok {
                GameOutcome::Finished
            } else {
                GameOutcome::Aborted { reason: format!("launcher exit codes {codes:?}; p0: {} ; p1: {}", tail(&logs[0], 3), tail(&logs[1], 3)) }
            });
        } else if started.elapsed() > timeout {
            outcome = Some(GameOutcome::Timeout);
        } else if stop() {
            outcome = Some(GameOutcome::Stopped);
        } else {
            std::thread::sleep(Duration::from_millis(100));
        }
    }
    for (i, c) in children.iter_mut().enumerate() {
        if !done[i] {
            platform.kill_tree(c);
        }
    }
    platform.release(lan);
    Ok(GameFinished { game_id: spec.game_id, outcome: outcome.unwrap(), exit_codes: codes, wall_s: started.elapsed().as_secs() as u32 })
}

/// Quarantine key: same map and player setup (020-R4).
pub fn config_key(spec: &GameSpec) -> String {
    let p = |i: usize| {
        let k = match &spec.players[i].kind {
            PlayerKind::Bridge { .. } => "bridge".to_string(),
            PlayerKind::OssBot { name } => format!("oss:{name}"),
        };
        format!("{}/{k}", race_name(spec.players[i].race))
    };
    format!("{:?}|{}|{}", spec.map, p(0), p(1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use uranium_proto::control::{MapKey, PlayerSpec};

    fn cfg() -> Config {
        Config {
            listen: String::new(),
            bwapi_dir: "/b".into(),
            bridge_lib: "/b/lib.so".into(),
            data_dir: "/d".into(),
            maps_dir: "/m".into(),
            max_games: 1,
            work_dir: "/w".into(),
            oss_bots_dir: None,
            port_base: 17000,
        }
    }

    fn spec(seed: Option<u32>) -> GameSpec {
        GameSpec {
            game_id: 3,
            map: MapKey::FightingSpirit,
            players: [
                PlayerSpec { race: Race::Terran, kind: PlayerKind::Bridge { agent_addr: "127.0.0.1:7600".into() } },
                PlayerSpec { race: Race::Random, kind: PlayerKind::OssBot { name: "ub".into() } },
            ],
            seed,
            max_frames: 1000,
            privileged: false,
            save_replay: false,
            wall_timeout_s: 10,
        }
    }

    #[test]
    fn launcher_env_matches_m2_harness() {
        let env = launcher_env(&cfg(), &spec(Some(9)), 0, Path::new("/b/lib.so"), Path::new("/w/g3"));
        let get = |k: &str| env.iter().find(|(a, _)| a == k).map(|(_, v)| v.clone());
        assert_eq!(get("BWAPI_CONFIG_AUTO_MENU__MAP").unwrap(), "/m/fighting_spirit_1.3.scx");
        assert_eq!(get("BWAPI_CONFIG_AUTO_MENU__RACE").unwrap(), "Terran");
        assert_eq!(get("URANIUM_AGENT_ADDR").unwrap(), "127.0.0.1:7600");
        assert_eq!(get("URANIUM_SEED").unwrap(), "9");
        assert_eq!(get("BWAPI_CONFIG_AUTO_MENU__SAVE_REPLAY").unwrap(), "");
        assert_eq!(get("URANIUM_LAN_SLOT").unwrap(), "0");
        let env1 = launcher_env(&cfg(), &spec(None), 1, Path::new("/o/module.so"), Path::new("/w/g3"));
        assert!(env1.iter().all(|(k, _)| k != "URANIUM_AGENT_ADDR" && k != "URANIUM_SEED"));
        assert!(env1.iter().any(|(k, v)| k == "BWAPI_CONFIG_AUTO_MENU__RACE" && v == "Random"));
    }

    #[test]
    fn config_key_ignores_seed_and_id() {
        let mut a = spec(Some(1));
        let b = spec(Some(2));
        a.game_id = 99;
        assert_eq!(config_key(&a), config_key(&b));
        assert!(config_key(&a).contains("oss:ub"));
    }
}
