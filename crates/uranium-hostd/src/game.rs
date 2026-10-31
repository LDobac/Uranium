//! Runs one 1v1 OpenBW game: two BWAPILauncher processes (spec 020-R2/R3/R4, 011).

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use uranium_proto::Race;
use uranium_proto::control::{GameFinished, GameOutcome, GameSpec, PlayerKind, ReplaySpec};

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

/// BWAPI drops ':' from the save_replay path ("illegal characters"), so on Windows a drive
/// letter turns the path relative to the launcher. Pass it drive-relative ("\dir\x.rep"); the
/// replay dir must be on the same drive as the hostd work dir.
fn bwapi_save_path(p: &Path) -> String {
    let s = p.display().to_string();
    if cfg!(windows) && s.as_bytes().get(1) == Some(&b':') { s[2..].to_string() } else { s }
}

/// Where launcher slot 0 saves the replay of a `save_replay` game (011-R7).
pub fn replay_file(spec: &GameSpec, replay_dir: &Path) -> PathBuf {
    replay_dir.join(format!("g{}-s{}-p0.rep", spec.game_id, spec.seed.map_or("x".to_string(), |s| s.to_string())))
}

/// 020-R1 / 042-R11: the saved replay goes back in `Finished.replay`. A file in the configured
/// replay folder stays; one in the game's work folder is only a temporary copy and is removed.
fn take_replay(cfg: &Config, spec: &GameSpec, dir: &Path) -> Option<Vec<u8>> {
    if !spec.save_replay {
        return None;
    }
    let path = replay_file(spec, cfg.replay_dir.as_deref().unwrap_or(dir));
    let bytes = std::fs::read(&path).ok()?;
    if cfg.replay_dir.is_none() {
        let _ = std::fs::remove_file(&path);
    }
    Some(bytes)
}

/// Player name prefix when the spec has no display name; the suffix is the `GameSpec.players`
/// index.
pub const PLAYER_NAME_PREFIX: &str = "uranium-p";

/// BWAPI cuts `auto_menu.character_name` to 24 bytes (AutoMenuManager.cpp).
pub const MAX_PLAYER_NAME: usize = 24;

/// 020-R16: `p<i>-<name>` cut to [`MAX_PLAYER_NAME`] bytes (the index is never cut), or
/// `uranium-p<i>` without a display name. The agent reads the index back (020-R11).
pub fn player_name(spec: &GameSpec, slot: usize) -> String {
    let Some(name) = spec.players[slot].name.as_deref() else { return format!("{PLAYER_NAME_PREFIX}{slot}") };
    let mut out = format!("p{slot}-");
    for c in name.chars() {
        if out.len() + c.len_utf8() > MAX_PLAYER_NAME {
            break;
        }
        out.push(c);
    }
    out
}

/// 011-R23 / 020-R17: turns on BWAPI patch 0016 (Train supply rule, ghost commands).
pub const BRIDGE_EXT_VAR: &str = "URANIUM_BRIDGE_EXT";

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
        ("URANIUM_OPENBW_DROP_TIMEOUT_MS", cfg.drop_timeout_ms.to_string()),
        (
            "BWAPI_CONFIG_AUTO_MENU__SAVE_REPLAY",
            // Both clients record the same game; keep slot 0 only (040 botgen dedup).
            if spec.save_replay && slot == 0 { bwapi_save_path(&replay_file(spec, replay_dir)) } else { String::new() },
        ),
        ("URANIUM_GAME_ID", spec.game_id.to_string()),
        // 0004 patch: deterministic client ids => launcher slot = player slot (011-AC4).
        ("URANIUM_LAN_SLOT", slot.to_string()),
        // The BWAPI player id need not follow launch order; the name tells the agent which
        // GameSpec player it is (050 scenario builds are looked up by it).
        ("BWAPI_CONFIG_AUTO_MENU__CHARACTER_NAME", player_name(spec, slot)),
        ("URANIUM_MAX_FRAMES", spec.max_frames.to_string()),
        ("URANIUM_PRIVILEGED", if spec.privileged { "1" } else { "0" }.to_string()),
    ];
    if let PlayerKind::Bridge { agent_addr } = &p.kind {
        env.push(("URANIUM_AGENT_ADDR", agent_addr.clone()));
        // 020-R17: only the launcher that runs our bridge gets the BWAPI extension.
        env.push((BRIDGE_EXT_VAR, "1".to_string()));
    }
    let mut env: Vec<(String, String)> = env.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
    // 0002 patch: every client must see the same seed for a deterministic setup (011-AC4).
    if let Some(seed) = spec.seed {
        env.push(("URANIUM_SEED".into(), seed.to_string()));
    }
    env
}

/// [`launcher_env`] on `cmd`. 020-R17: an OSS bot launcher never inherits [`BRIDGE_EXT_VAR`] from
/// hostd's own environment, so BWAPI patch 0016 stays off for it.
fn set_launcher_env(cmd: &mut Command, cfg: &Config, spec: &GameSpec, slot: usize, module: &Path, replay_dir: &Path) {
    cmd.envs(launcher_env(cfg, spec, slot, module, replay_dir));
    if !matches!(spec.players[slot].kind, PlayerKind::Bridge { .. }) {
        cmd.env_remove(BRIDGE_EXT_VAR);
    }
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
            set_launcher_env(&mut cmd, cfg, spec, slot, &module, cfg.replay_dir.as_deref().unwrap_or(&dir));
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
        if done[i] {
            platform.reap(c);
        } else {
            platform.kill_tree(c);
        }
    }
    platform.release(lan);
    let replay = take_replay(cfg, spec, &dir);
    Ok(GameFinished { game_id: spec.game_id, outcome: outcome.unwrap(), exit_codes: codes, wall_s: started.elapsed().as_secs() as u32, replay })
}

/// Environment for one replay playback launcher (011-R10, 020-R1 StartReplay).
pub fn replay_env(cfg: &Config, job: &ReplaySpec, rep: &Path) -> Vec<(String, String)> {
    [
        ("URANIUM_DATA_DIR", cfg.data_dir.display().to_string()),
        ("OPENBW_ENABLE_UI", "0".to_string()),
        ("BWAPI_CONFIG_AI__AI", cfg.bridge_lib.display().to_string()),
        ("BWAPI_CONFIG_AUTO_MENU__AUTO_MENU", "SINGLE_PLAYER".to_string()),
        ("BWAPI_CONFIG_AUTO_MENU__MAP", rep.display().to_string()),
        ("BWAPI_CONFIG_AUTO_MENU__AUTO_RESTART", "OFF".to_string()),
        ("BWAPI_CONFIG_AUTO_MENU__SAVE_REPLAY", String::new()),
        ("URANIUM_GAME_ID", job.job_id.to_string()),
        ("URANIUM_AGENT_ADDR", job.agent_addr.clone()),
        // 013: BWAPI id of the observed player (rank of its occupied header slot).
        ("URANIUM_REPLAY_PERSPECTIVE", job.perspective_slot.to_string()),
        ("URANIUM_PRIVILEGED", if job.privileged { "1" } else { "0" }.to_string()),
        // Replays end on their own; never cut them at the 60-minute game limit.
        ("URANIUM_MAX_FRAMES", u32::MAX.to_string()),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect()
}

/// Plays one replay (`rep_bytes`, already in 1.16.1 form, 013-R6) with a single launcher in
/// replay mode. The bridge streams RawFrames to `agent_addr` (040-R12 streaming, 020-AC4).
pub fn run_replay(cfg: &Config, platform: &dyn Platform, job: &ReplaySpec, stop: &dyn Fn() -> bool) -> Result<GameFinished> {
    let started = Instant::now();
    let dir = cfg.work_dir.join(format!("r{}", job.job_id));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("bwapi-data"))?;
    let rep = dir.join("replay.rep");
    std::fs::write(&rep, &job.rep_bytes)?;
    let log = dir.join("launcher.log");
    let out = std::fs::File::create(&log)?;
    let mut cmd = Command::new(cfg.launcher());
    cmd.current_dir(&dir).stdin(Stdio::null()).stdout(out.try_clone()?).stderr(out);
    cmd.envs(replay_env(cfg, job, &rep));
    let mut child = platform.spawn(cmd).context("spawning replay launcher")?;
    let timeout = Duration::from_secs(cfg.replay_timeout_s as u64);
    let (outcome, code, exited) = loop {
        if let Some(st) = child.try_wait()? {
            let o = if st.code() == Some(0) {
                GameOutcome::Finished
            } else {
                GameOutcome::Aborted { reason: format!("launcher exit code {:?}: {}", st.code(), tail(&log, 3)) }
            };
            break (o, st.code(), true);
        }
        if started.elapsed() > timeout {
            break (GameOutcome::Timeout, None, false);
        }
        if stop() {
            break (GameOutcome::Stopped, None, false);
        }
        std::thread::sleep(Duration::from_millis(100));
    };
    if exited { platform.reap(&mut child) } else { platform.kill_tree(&mut child) }
    let _ = std::fs::remove_file(&rep);
    Ok(GameFinished { game_id: job.job_id, outcome, exit_codes: [code, None], wall_s: started.elapsed().as_secs() as u32, replay: None })
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
            replay_timeout_s: 1800,
            replay_dir: None,
            drop_timeout_ms: crate::config::DEFAULT_DROP_TIMEOUT_MS,
        }
    }

    fn spec(seed: Option<u32>) -> GameSpec {
        GameSpec {
            game_id: 3,
            map: MapKey::FightingSpirit,
            players: [
                PlayerSpec { race: Race::Terran, kind: PlayerKind::Bridge { agent_addr: "127.0.0.1:7600".into() }, name: None },
                PlayerSpec { race: Race::Random, kind: PlayerKind::OssBot { name: "ub".into() }, name: Some("ub".into()) },
            ],
            seed,
            max_frames: 1000,
            privileged: false,
            save_replay: false,
            wall_timeout_s: 10,
        }
    }

    /// 020-R1 / 042-R11: the replay travels back in `Finished`; a work-folder copy is temporary,
    /// a file in the configured replay folder stays.
    #[test]
    fn take_replay_returns_bytes_and_cleans_work_copy_020_r1() {
        let root = std::env::temp_dir().join(format!("uranium-hostd-rep-{}", std::process::id()));
        let (work, keep) = (root.join("g3"), root.join("keep"));
        std::fs::create_dir_all(&work).unwrap();
        std::fs::create_dir_all(&keep).unwrap();
        let mut s = spec(Some(9));
        assert_eq!(take_replay(&cfg(), &s, &work), None, "not asked for");
        s.save_replay = true;
        assert_eq!(take_replay(&cfg(), &s, &work), None, "nothing written");
        std::fs::write(replay_file(&s, &work), b"rep").unwrap();
        assert_eq!(take_replay(&cfg(), &s, &work).as_deref(), Some(&b"rep"[..]));
        assert!(!replay_file(&s, &work).exists(), "temporary copy removed");
        let c = Config { replay_dir: Some(keep.clone()), ..cfg() };
        std::fs::write(replay_file(&s, &keep), b"rep2").unwrap();
        assert_eq!(take_replay(&c, &s, &work).as_deref(), Some(&b"rep2"[..]));
        assert!(replay_file(&s, &keep).exists(), "configured folder keeps the file");
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn replay_env_uses_perspective_and_no_frame_limit_011_r10() {
        let job = ReplaySpec { job_id: 9, rep_bytes: vec![], perspective_slot: 3, agent_addr: "127.0.0.1:7600".into(), privileged: true };
        let env: std::collections::HashMap<String, String> = replay_env(&cfg(), &job, Path::new("/w/r9/replay.rep")).into_iter().collect();
        assert_eq!(env["URANIUM_REPLAY_PERSPECTIVE"], "3");
        assert_eq!(env["BWAPI_CONFIG_AUTO_MENU__AUTO_MENU"], "SINGLE_PLAYER");
        assert_eq!(env["BWAPI_CONFIG_AUTO_MENU__MAP"], Path::new("/w/r9/replay.rep").display().to_string());
        assert_eq!(env["URANIUM_PRIVILEGED"], "1");
        assert_eq!(env["URANIUM_MAX_FRAMES"], u32::MAX.to_string());
    }

    #[test]
    fn launcher_env_matches_m2_harness() {
        let env = launcher_env(&cfg(), &spec(Some(9)), 0, Path::new("/b/lib.so"), Path::new("/w/g3"));
        let get = |k: &str| env.iter().find(|(a, _)| a == k).map(|(_, v)| v.clone());
        // Path separator is platform-specific (001-R8).
        let map = Path::new("/m").join("fighting_spirit_1.3.scx");
        assert_eq!(get("BWAPI_CONFIG_AUTO_MENU__MAP").unwrap(), map.to_string_lossy());
        assert_eq!(get("BWAPI_CONFIG_AUTO_MENU__RACE").unwrap(), "Terran");
        assert_eq!(get("URANIUM_AGENT_ADDR").unwrap(), "127.0.0.1:7600");
        assert_eq!(get("URANIUM_SEED").unwrap(), "9");
        assert_eq!(get("BWAPI_CONFIG_AUTO_MENU__SAVE_REPLAY").unwrap(), "");
        assert_eq!(get("URANIUM_LAN_SLOT").unwrap(), "0");
        assert_eq!(get("BWAPI_CONFIG_AUTO_MENU__CHARACTER_NAME").unwrap(), "uranium-p0");
        // 011-R18: both clients get the raised drop timeout (patch 0013).
        assert_eq!(get("URANIUM_OPENBW_DROP_TIMEOUT_MS").unwrap(), "600000");
        let env1 = launcher_env(&cfg(), &spec(None), 1, Path::new("/o/module.so"), Path::new("/w/g3"));
        assert!(env1.iter().any(|(k, v)| k == "URANIUM_OPENBW_DROP_TIMEOUT_MS" && v == "600000"));
        assert!(env1.iter().all(|(k, _)| k != "URANIUM_AGENT_ADDR" && k != "URANIUM_SEED"));
        assert!(env1.iter().any(|(k, v)| k == "BWAPI_CONFIG_AUTO_MENU__RACE" && v == "Random"));
    }

    /// 011-AC11 / 020-R17: `URANIUM_BRIDGE_EXT=1` reaches the bridge launcher only; the OSS bot
    /// launcher gets it removed even when hostd itself has it.
    #[test]
    fn bridge_ext_only_for_bridge_launcher_020_r17() {
        let s = spec(None);
        let ext = |slot: usize| {
            let mut cmd = Command::new("launcher");
            set_launcher_env(&mut cmd, &cfg(), &s, slot, Path::new("/m.so"), Path::new("/w/g3"));
            cmd.get_envs()
                .filter(|(k, _)| *k == BRIDGE_EXT_VAR)
                .map(|(_, v)| v.map(|v| v.to_string_lossy().into_owned()))
                .collect::<Vec<_>>()
        };
        assert_eq!(ext(0), vec![Some("1".to_string())], "bridge launcher (slot 0)");
        assert_eq!(ext(1), vec![None], "OSS bot launcher: removed, not inherited");
    }

    /// 020-R16: `p<i>-<name>` cut to 24 bytes without cutting the index, legacy name without one.
    #[test]
    fn player_names_020_r16() {
        let mut s = spec(None);
        assert_eq!(player_name(&s, 0), "uranium-p0");
        assert_eq!(player_name(&s, 1), "p1-ub");
        s.players[0].name = Some("core-abl-gtrxl-pt".into());
        assert_eq!(player_name(&s, 0), "p0-core-abl-gtrxl-pt");
        s.players[0].name = Some("a-very-long-checkpoint-run-name".into());
        let n = player_name(&s, 0);
        assert_eq!((n.len(), n.as_str()), (MAX_PLAYER_NAME, "p0-a-very-long-checkpoin"));
        s.players[0].name = Some("é".repeat(15));
        assert!(player_name(&s, 0).len() <= MAX_PLAYER_NAME);
        let env = launcher_env(&cfg(), &s, 1, Path::new("/o/module.so"), Path::new("/w/g3"));
        assert!(env.iter().any(|(k, v)| k == "BWAPI_CONFIG_AUTO_MENU__CHARACTER_NAME" && v == "p1-ub"));
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
