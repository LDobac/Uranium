use serde::{Deserialize, Serialize};

use crate::Race;

pub const CONTROL_VERSION: u32 = 3;

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum MapKey {
    FightingSpirit,
    CircuitBreaker,
}

impl MapKey {
    pub fn file_name(self) -> &'static str {
        match self {
            MapKey::FightingSpirit => "fighting_spirit_1.3.scx",
            MapKey::CircuitBreaker => "circuit_breaker_1.0.scx",
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub enum PlayerKind {
    Bridge { agent_addr: String },
    OssBot { name: String },
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct PlayerSpec {
    pub race: Race,
    pub kind: PlayerKind,
    pub name: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct GameSpec {
    pub game_id: u64,
    pub map: MapKey,
    pub players: [PlayerSpec; 2],
    pub seed: Option<u32>,
    pub max_frames: u32,
    pub privileged: bool,
    pub save_replay: bool,
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
    pub oss_bots: Vec<String>,
    pub maps: Vec<MapKey>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub enum GameOutcome {
    Finished,
    Aborted { reason: String },
    Timeout,
    Stopped,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct GameFinished {
    pub game_id: u64,
    pub outcome: GameOutcome,
    pub exit_codes: [Option<i32>; 2],
    pub wall_s: u32,
    pub replay: Option<Vec<u8>>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub enum ControlResponse {
    Hello { control_version: u32, host_name_hint: String },
    Capabilities(Capabilities),
    Started { game_id: u64 },
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
                PlayerSpec { race: Race::Terran, kind: PlayerKind::Bridge { agent_addr: "127.0.0.1:7600".into() }, name: Some("core-abl-gtrxl-pt".into()) },
                PlayerSpec { race: Race::Random, kind: PlayerKind::OssBot { name: "ualbertabot".into() }, name: None },
            ],
            seed: Some(42),
            max_frames: 86_400,
            privileged: true,
            save_replay: false,
            wall_timeout_s: 900,
        };
        rt(ControlRequest::StartGame(spec));
        rt(ControlRequest::StopGame { game_id: 7 });
        rt(ControlResponse::Finished(GameFinished { game_id: 7, outcome: GameOutcome::Aborted { reason: "x".into() }, exit_codes: [Some(0), None], wall_s: 3, replay: Some(vec![1, 2, 3]) }));
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
