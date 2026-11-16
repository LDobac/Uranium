use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use uranium_scripted::T0Build;

use crate::Replay;
use crate::header::{RACE_PROTOSS, RACE_RANDOM, RACE_TERRAN, RACE_ZERG};
use crate::pipeline::View;

pub const TP_TEACHER_NAME: &str = "T0-terran-terran_tp";

#[derive(Debug, Clone, Default, PartialEq)]
pub struct GamesLog {
    races: HashMap<u64, [u8; 2]>,
}

#[derive(serde::Deserialize)]
struct GameLine {
    game_id: u64,
    races: [String; 2],
}

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

    pub fn lobby(&self, file: &Path) -> Option<[u8; 2]> {
        self.races.get(&game_id(file)?).copied()
    }
}

pub fn game_id(file: &Path) -> Option<u64> {
    let stem = file.file_stem()?.to_str()?.strip_prefix('g')?;
    let digits = stem.split('-').next()?;
    digits.parse().ok()
}

pub fn name_index(name: &[u8]) -> Option<u8> {
    let name = std::str::from_utf8(name).ok()?;
    if let Some(i) = name.strip_prefix("uranium-p") {
        return i.parse().ok();
    }
    let (i, _) = name.strip_prefix('p')?.split_once('-')?;
    i.parse().ok()
}

pub fn is_tp_teacher_game(rep: &Replay) -> bool {
    rep.header.players.iter().any(|p| name_index(&p.name).is_some() && p.name.splitn(2, |&b| b == b'-').nth(1) == Some(TP_TEACHER_NAME.as_bytes()))
}

fn tp_teacher_p0() -> String {
    format!("p0-{TP_TEACHER_NAME}")
}

pub fn unique_game_ids(files: &[std::path::PathBuf]) -> Result<()> {
    let mut seen: HashMap<u64, &std::path::PathBuf> = HashMap::new();
    for f in files {
        if let Some(first) = game_id(f).and_then(|id| seen.insert(id, f)) {
            bail!("--games: {} and {} have the same game id: build one plan (stage) folder at a time (022-R12)", first.display(), f.display());
        }
    }
    Ok(())
}

pub fn players0_view(rep: &Replay, views: Vec<View>, lobby: Option<[u8; 2]>, relabel: Option<T0Build>) -> std::result::Result<View, (Option<View>, String)> {
    let Some([own, enemy]) = lobby else {
        return Err((None, "no games.jsonl line for this replay: lobby races unknown (022-R2, 022-R12)".into()));
    };
    let p0s: Vec<&crate::header::Player> = rep.header.players.iter().filter(|p| name_index(&p.name) == Some(0)).collect();
    let [p0] = p0s.as_slice() else {
        return Err((None, format!("{} players[0] player names, not one (hostd p0-<name> / uranium-p0, 020-R16)", p0s.len())));
    };
    if relabel.is_none() && p0.name != tp_teacher_p0().as_bytes() {
        return Err((None, format!("players[0] is not the 022 teacher by its hostd name {} (022-R12: teacher view only; plan name overrides are not supported)", tp_teacher_p0())));
    }
    let Some(view) = views.into_iter().find(|v| v.perspective == p0.id) else {
        return Err((None, "players[0] is not a Terran view".into()));
    };
    if own != RACE_TERRAN {
        return Err((Some(view), format!("players[0] lobby race {} (games.jsonl), not Terran", race_name(own))));
    }
    let played = rep.active_players().into_iter().find(|(p, _)| p.id != p0.id).map(|(p, _)| p.race);
    let reason = match enemy {
        RACE_RANDOM => Some("lobby Random opponent: no 022 teacher data from Random games (022-R2)".to_string()),
        RACE_ZERG if relabel.is_some() => Some("lobby Zerg opponent: --relabel terran_tp relabels games against Terran / Protoss only (022-R2, 022-AC3, 022-R12 ②, 044-R8 ②, 041-R22)".to_string()),
        RACE_ZERG => Some("lobby Zerg opponent: 022 teacher botgen data is against Terran / Protoss only (022-R2, 022-R12)".to_string()),
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
        let p = |s: &str| std::path::PathBuf::from(s);
        assert!(unique_game_ids(&[p("a/g3.rep"), p("a/g4.rep"), p("a/x.rep"), p("a/y.rep")]).is_ok());
        assert!(unique_game_ids(&[p("a/g3.rep"), p("a/g3-s5-p0.rep")]).is_err(), "one id, two files");
        assert!(unique_game_ids(&[p("a/replays/g3.rep"), p("b/replays/g3.rep")]).is_err(), "two plan folders");
        assert_eq!([name_index(b"p0-T0-terran-terran_tp"), name_index(b"uranium-p1"), name_index(b"p12-x"), name_index(b"player"), name_index(b"px-y")], [Some(0), Some(1), Some(12), None, None]);
    }
}
