mod brain;
mod kit;
pub mod placement;
pub mod react;
pub mod tp;
pub mod types;
mod util;

use uranium_proto::*;

pub use crate::brain::Brain;
pub use crate::kit::{Kit, Supply, Tactic};
pub use crate::react::{Branch, MapHints, ReactParams, ReactStats, RushRule};
pub use crate::tp::{TpParams, TpRule, TpStats};

pub trait Bot: Send {
    fn act(&mut self, f: &RawFrame) -> Act;
    fn react_stats(&self) -> Option<ReactStats> {
        None
    }
    fn tp_stats(&self) -> Option<TpStats> {
        None
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum T0Build {
    #[default]
    Standard,
    ZergRush,
    TerranDrop,
    ProtossDt,
    ZergMuta,
    TerranReact,
    TerranTp,
}

impl T0Build {
    pub const SCENARIOS: [T0Build; 4] = [T0Build::ZergRush, T0Build::TerranDrop, T0Build::ProtossDt, T0Build::ZergMuta];

    pub fn id_tag(self) -> Option<&'static str> {
        match self {
            T0Build::Standard => None,
            T0Build::ZergRush => Some("rush"),
            T0Build::TerranDrop => Some("drop"),
            T0Build::ProtossDt => Some("dt"),
            T0Build::ZergMuta => Some("muta"),
            T0Build::TerranReact => Some("react"),
            T0Build::TerranTp => Some("tp"),
        }
    }

    pub fn race(self) -> Option<Race> {
        match self {
            T0Build::Standard => None,
            T0Build::ZergRush | T0Build::ZergMuta => Some(Race::Zerg),
            T0Build::TerranDrop | T0Build::TerranReact | T0Build::TerranTp => Some(Race::Terran),
            T0Build::ProtossDt => Some(Race::Protoss),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            T0Build::Standard => "standard",
            T0Build::ZergRush => "zerg_rush",
            T0Build::TerranDrop => "terran_drop",
            T0Build::ProtossDt => "protoss_dt",
            T0Build::ZergMuta => "zerg_muta",
            T0Build::TerranReact => "terran_react",
            T0Build::TerranTp => "terran_tp",
        }
    }

    pub fn needs_map_hints(self) -> bool {
        matches!(self, T0Build::TerranReact | T0Build::TerranTp)
    }

    pub fn for_lobby(self, info: &GameInfo) -> T0Build {
        let enemy = info.players.iter().find(|p| p.is_enemy).map(|p| p.race);
        match self {
            T0Build::TerranTp if enemy == Some(Race::Zerg) => T0Build::TerranReact,
            b => b,
        }
    }
}

pub fn for_game(info: &GameInfo) -> Box<dyn Bot> {
    for_game_build(info, T0Build::Standard)
}

pub fn for_game_build(info: &GameInfo, build: T0Build) -> Box<dyn Bot> {
    for_game_hints(info, build, MapHints::default())
}

pub fn for_game_hints(info: &GameInfo, build: T0Build, hints: MapHints) -> Box<dyn Bot> {
    let race = info.players.iter().find(|p| p.is_self).map(|p| p.race).unwrap_or(Race::Terran);
    Box::new(Brain::with_hints(info, Kit::for_build(race, build.for_lobby(info)), &hints))
}

#[cfg(test)]
mod tests;
