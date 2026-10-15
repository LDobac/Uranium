//! Scheduler (`uranium-env`) <-> game host daemon (`uranium-hostd`) control protocol
//! (spec 020-R1). Same framing as the game protocol (`write_msg`/`read_msg`).

use serde::{Deserialize, Serialize};

use crate::Race;

/// Bumped on incompatible control-schema changes.
pub const CONTROL_VERSION: u32 = 1;

/// Target map families (`config/assets.toml`).
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum MapKey {
    FightingSpirit,
    CircuitBreaker,
}

impl MapKey {
    /// File name inside `URANIUM_MAPS_DIR`.
    pub fn file_name(self) -> &'static str {
        match self {
            MapKey::FightingSpirit => "fighting_spirit_1.3.scx",
            MapKey::CircuitBreaker => "circuit_breaker_1.0.scx",
        }
    }
}

/// What controls one player slot.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub enum PlayerKind {
    /// Our OpenBW bridge connecting to an agent server (policy or T0 bot runs there).
    Bridge { agent_addr: String },
    /// A third-party BWAPI bot module registered on the host (021 T1).
    OssBot { name: String },
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct PlayerSpec {
    /// Race picked in the lobby (`Race::Random` allowed, 021-R7).
    pub race: Race,
    pub kind: PlayerKind,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct GameSpec {
    pub game_id: u64,
    pub map: MapKey,
    /// Exactly two players (1v1).
    pub players: [PlayerSpec; 2],
    /// `URANIUM_SEED` (011-AC4); `None` = engine-random setup.
    pub seed: Option<u32>,
    /// `URANIUM_MAX_FRAMES` (020-R3).
    pub max_frames: u32,
    /// `URANIUM_PRIVILEGED` (010-R10).
    pub privileged: bool,
    pub save_replay: bool,
    /// Kill the game after this much wall time (020-R3).
    pub wall_timeout_s: u32,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct ReplaySpec {
    pub job_id: u64,
    pub rep_bytes: Vec<u8>,
    pub perspective_slot: u8,
    pub agent_addr: String,
    pub privileged: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub enum ControlRequest {
    Hello { control_version: u32 },
    ListCapabilities,
    StartGame(GameSpec),
    /// 040-R12 streaming; hostd answers `Unsupported` until M4.
    StartReplay(ReplaySpec),
    StopGame { game_id: u64 },
    Status,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostOs {
    Linux,
    Windows,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct Capabilities {
    pub os: HostOs,
    pub cores: u32,
    pub max_concurrent_games: u32,
    pub max_concurrent_replays: u32,
    /// Opponent IDs `<name>@<version>` (021-R3); the part before `@` is the name usable in
    /// `PlayerKind::OssBot`. Bots built without a VERSION file are listed by name only.
    pub oss_bots: Vec<String>,
    pub maps: Vec<MapKey>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub enum GameOutcome {
    /// Both launchers exited cleanly (the result itself comes from the agent's GameEnd).
    Finished,
    /// A launcher crashed or failed to start (020-R4); excluded from training.
    Aborted { reason: String },
    /// Wall-clock timeout; the game was killed (020-R3).
    Timeout,
    /// Stopped by `StopGame`.
    Stopped,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct GameFinished {
    pub game_id: u64,
    pub outcome: GameOutcome,
    /// Exit codes of the two launchers (`None` = killed / no code).
    pub exit_codes: [Option<i32>; 2],
    pub wall_s: u32,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub enum ControlResponse {
    Hello { control_version: u32, host_name_hint: String },
    Capabilities(Capabilities),
    /// The game was accepted and is starting.
    Started { game_id: u64 },
    /// Sent on the same connection when an accepted game ends.
    Finished(GameFinished),
    Status { running: Vec<u64>, quarantined: Vec<String> },
    Rejected { reason: String },
    Unsupported { what: String },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{read_msg, write_msg};

    fn rt<T: Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug>(m: T) {
        let mut buf = Vec::new();
        write_msg(&mut buf, &m).unwrap();
        assert_eq!(read_msg::<T>(&mut buf.as_slice()).unwrap(), m);
    }

    #[test]
    fn control_messages_roundtrip() {
        let spec = GameSpec {
            game_id: 7,
            map: MapKey::CircuitBreaker,
            players: [
                PlayerSpec { race: Race::Terran, kind: PlayerKind::Bridge { agent_addr: "127.0.0.1:7600".into() } },
                PlayerSpec { race: Race::Random, kind: PlayerKind::OssBot { name: "ualbertabot".into() } },
            ],
            seed: Some(42),
            max_frames: 86_400,
            privileged: true,
            save_replay: false,
            wall_timeout_s: 900,
        };
        rt(ControlRequest::StartGame(spec));
        rt(ControlRequest::StopGame { game_id: 7 });
        rt(ControlResponse::Finished(GameFinished { game_id: 7, outcome: GameOutcome::Aborted { reason: "x".into() }, exit_codes: [Some(0), None], wall_s: 3 }));
        rt(ControlResponse::Capabilities(Capabilities {
            os: HostOs::Linux,
            cores: 12,
            max_concurrent_games: 5,
            max_concurrent_replays: 0,
            oss_bots: vec![],
            maps: vec![MapKey::FightingSpirit],
        }));
    }

    #[test]
    fn map_file_names_match_assets() {
        assert_eq!(MapKey::FightingSpirit.file_name(), "fighting_spirit_1.3.scx");
        assert_eq!(MapKey::CircuitBreaker.file_name(), "circuit_breaker_1.0.scx");
    }
}
