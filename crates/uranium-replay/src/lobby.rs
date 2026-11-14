//! 022 teacher data (022-R2, 022-R12, 022-AC3; 041-R22 and 040 2026-11-14): a build with
//! `--games <games.jsonl>` (`--src botgen` for the 022 teacher's botgen games, required with
//! `--relabel terran_tp`) keeps only the plan's players[0] as a view (the teacher in its botgen
//! games, the model in its self games; spec 040: only the teacher's view is data, so the Terran
//! opponent of a TvT game is not), and takes the opponent's LOBBY race from the runner's
//! games.jsonl line of the game instead of the replay header, which stores the race played
//! (`header::RACE_RANDOM`). A replay without such a line is refused, never guessed.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use uranium_scripted::T0Build;

use crate::Replay;
use crate::header::{RACE_PROTOSS, RACE_RANDOM, RACE_TERRAN, RACE_ZERG};
use crate::pipeline::View;

/// Display name of the 022 teacher in a plan (`uranium-runner` `Side::display_name`); hostd names
/// the player `p<index>-<name>` (020-R16).
pub const TP_TEACHER_NAME: &str = "T0-terran-terran_tp";

/// The lobby races of the games of one runner output (games.jsonl: `game_id`, `races` =
/// `GameSpec.players` lobby races, `Random` kept, 021-R7).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GamesLog {
    races: HashMap<u64, [u8; 2]>,
}

#[derive(serde::Deserialize)]
struct GameLine {
    game_id: u64,
    races: [String; 2],
}

/// Header race code of a games.jsonl race (`uranium_proto::Race` debug name).
pub fn race_code(s: &str) -> Option<u8> {
    match s {
        "Zerg" => Some(RACE_ZERG),
        "Terran" => Some(RACE_TERRAN),
        "Protoss" => Some(RACE_PROTOSS),
        "Random" => Some(RACE_RANDOM),
        _ => None,
    }
}

fn race_name(r: u8) -> &'static str {
    match r {
        RACE_ZERG => "Zerg",
        RACE_TERRAN => "Terran",
        RACE_PROTOSS => "Protoss",
        RACE_RANDOM => "Random",
        _ => "unknown",
    }
}

impl GamesLog {
    /// games.jsonl text: one line per game attempt; every line of a game id must name the same races.
    pub fn parse(text: &str) -> Result<Self> {
        let mut races: HashMap<u64, [u8; 2]> = HashMap::new();
        for (i, l) in text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
            let g: GameLine = serde_json::from_str(l).with_context(|| format!("games.jsonl line {}", i + 1))?;
            let r = [0, 1].map(|k| race_code(&g.races[k]));
            let [Some(a), Some(b)] = r else { bail!("games.jsonl line {}: races {:?}", i + 1, g.races) };
            if let Some(old) = races.insert(g.game_id, [a, b]).filter(|o| *o != [a, b]) {
                bail!("games.jsonl: game {} has races {old:?} and {:?}", g.game_id, [a, b]);
            }
        }
        Ok(Self { races })
    }

    pub fn load(path: &Path) -> Result<Self> {
        Self::parse(&std::fs::read_to_string(path).with_context(|| format!("--games {}", path.display()))?)
    }

    /// The lobby races [players[0], players[1]] of the replay file (`g<game id>...rep`).
    pub fn lobby(&self, file: &Path) -> Option<[u8; 2]> {
        self.races.get(&game_id(file)?).copied()
    }
}

/// The game id of a runner replay file name: `g<id>.rep` (uranium-env run, evaluation jobs) or
/// hostd's `g<id>-s<seed>-p0.rep`.
pub fn game_id(file: &Path) -> Option<u64> {
    let stem = file.file_stem()?.to_str()?.strip_prefix('g')?;
    let digits = stem.split('-').next()?;
    digits.parse().ok()
}

/// The `GameSpec.players` index in a hostd player name: `p<i>-<name>` or `uranium-p<i>` (020-R16,
/// as `uranium_agent::name_index`).
pub fn name_index(name: &[u8]) -> Option<u8> {
    let name = std::str::from_utf8(name).ok()?;
    if let Some(i) = name.strip_prefix("uranium-p") {
        return i.parse().ok();
    }
    let (i, _) = name.strip_prefix('p')?.split_once('-')?;
    i.parse().ok()
}

/// A replay in which a player is the 022 teacher (hostd name `p<i>-T0-terran-terran_tp`).
pub fn is_tp_teacher_game(rep: &Replay) -> bool {
    rep.header.players.iter().any(|p| name_index(&p.name).is_some() && p.name.splitn(2, |&b| b == b'-').nth(1) == Some(TP_TEACHER_NAME.as_bytes()))
}

/// The players[0] view of a replay (one of the [`crate::pipeline::filter`] `views`), or why the
/// replay is refused (with the view when there is one): no games.jsonl line (`lobby` None), no
/// players[0] player name, players[0] not a Terran view or not a lobby Terran, a lobby Random
/// opponent (022-R2), a lobby Zerg opponent (022-AC3 with `--relabel terran_tp`, 022-R12 for the
/// 022 botgen data), or a lobby race other than the race the replay header says was played.
pub fn players0_view(rep: &Replay, views: Vec<View>, lobby: Option<[u8; 2]>, relabel: Option<T0Build>) -> std::result::Result<View, (Option<View>, String)> {
    let Some([own, enemy]) = lobby else {
        return Err((None, "no games.jsonl line for this replay: lobby races unknown (022-R2, 022-R12)".into()));
    };
    let Some(p0) = rep.header.players.iter().find(|p| name_index(&p.name) == Some(0)) else {
        return Err((None, "no players[0] player name (hostd p0-<name> / uranium-p0, 020-R16)".into()));
    };
    let Some(view) = views.into_iter().find(|v| v.perspective == p0.id) else {
        return Err((None, "players[0] is not a Terran view".into()));
    };
    if own != RACE_TERRAN {
        return Err((Some(view), format!("players[0] lobby race {} (games.jsonl), not Terran", race_name(own))));
    }
    // The opponent: the other active player (`filter` keeps 1v1 replays only).
    let played = rep.active_players().into_iter().find(|(p, _)| p.id != p0.id).map(|(p, _)| p.race);
    let reason = match enemy {
        RACE_RANDOM => Some("lobby Random opponent: no 022 teacher data from Random games (022-R2)".to_string()),
        RACE_ZERG if relabel.is_some() => Some("lobby Zerg opponent: --relabel terran_tp relabels games against Terran / Protoss only (022-AC3, 041-R22)".to_string()),
        RACE_ZERG => Some("lobby Zerg opponent: 022 teacher botgen data is against Terran / Protoss only (022-R12)".to_string()),
        r if played != Some(r) => Some(format!("lobby race {} (games.jsonl) but the replay header race {}", race_name(r), played.map_or("none", race_name))),
        _ => None,
    };
    match reason {
        Some(r) => Err((Some(view), r)),
        None => Ok(view),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn games_log_and_names_022_r2() {
        let log = GamesLog::parse(
            "{\"game_id\":3,\"matchup\":0,\"races\":[\"Terran\",\"Random\"],\"outcome\":\"aborted\"}\n\n{\"game_id\":3,\"races\":[\"Terran\",\"Random\"]}\n{\"game_id\":4,\"races\":[\"Terran\",\"Protoss\"]}\n",
        )
        .unwrap();
        assert_eq!(log.lobby(Path::new("x/replays/g3.rep")), Some([RACE_TERRAN, RACE_RANDOM]));
        assert_eq!(log.lobby(Path::new("g4-s9-p0.rep")), Some([RACE_TERRAN, RACE_PROTOSS]));
        assert_eq!(log.lobby(Path::new("g5.rep")), None);
        assert_eq!(log.lobby(Path::new("game4.rep")), None);
        assert!(GamesLog::parse("{\"game_id\":3,\"races\":[\"Terran\",\"Zerg\"]}\n{\"game_id\":3,\"races\":[\"Terran\",\"Protoss\"]}").is_err(), "conflicting lines");
        assert!(GamesLog::parse("{\"game_id\":3,\"races\":[\"Terran\",\"Elf\"]}").is_err());
        assert_eq!([name_index(b"p0-T0-terran-terran_tp"), name_index(b"uranium-p1"), name_index(b"p12-x"), name_index(b"player"), name_index(b"px-y")], [Some(0), Some(1), Some(12), None, None]);
    }
}
