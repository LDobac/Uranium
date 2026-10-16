//! 043-R10: every path and response field of the SC:R local web API used by the collector.
//! Measured on 2026-10-16 (`reports/m4_ladder_probe.md`); when a client update changes the
//! API, only this module changes.

use std::collections::BTreeMap;

use serde::Deserialize;
use serde_json::Value;

/// Prefixes tried in front of the `/v1/...` routes; the measured client uses `/web-api`.
pub const PATH_PREFIXES: [&str; 2] = ["", "/web-api"];
pub const LEADERBOARD_LIST: &str = "/v1/leaderboard";
pub const GATEWAYS: &str = "/v1/gateway";

pub fn leaderboard_page(id: u64, offset: u32, length: u32) -> String {
    format!("/v1/leaderboard/{id}?offset={offset}&length={length}")
}

pub fn profile(toon: &str, gateway: u32) -> String {
    format!("/v2/aurora-profile-by-toon/{}/{gateway}?request_flags=scr_profile", enc(toon))
}

pub fn match_info(link: &str) -> String {
    format!("/v1/matchmaker-gameinfo-playerinfo/{}", enc(link))
}

pub fn match_history(toon: &str, gateway: &str, gamemode: &str, season: &str) -> String {
    format!("/v1/matchmaker-gameinfo-by-toon/{}/{}/{}/{}", enc(toon), enc(gateway), enc(gamemode), enc(season))
}

/// Ladder (matchmaking) games have `MM-<UUID>` links; private lobbies have numeric ones.
pub const LADDER_LINK_PREFIX: &str = "MM-";
/// `create_time` values at or above this are placeholders (u32::MAX was seen), not times.
pub const MAX_REAL_TIME: u64 = u32::MAX as u64;

#[derive(Deserialize, Debug)]
pub struct LeaderboardList {
    pub leaderboards: BTreeMap<String, LeaderboardMeta>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct LeaderboardMeta {
    pub id: u64,
    pub gamemode_id: u32,
    pub gateway_id: u32,
    pub season_id: u32,
}

impl LeaderboardList {
    /// Current-season leaderboard of one gateway and game mode.
    pub fn current(&self, gateway: u32, gamemode: u32) -> Option<&LeaderboardMeta> {
        self.leaderboards
            .values()
            .filter(|l| l.gateway_id == gateway && l.gamemode_id == gamemode)
            .max_by_key(|l| l.season_id)
    }
}

/// A leaderboard page is a table: `columns` names, `rows` values.
#[derive(Deserialize, Debug)]
pub struct LeaderboardPage {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<Value>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LeaderboardEntry {
    pub toon: String,
    pub gateway_id: u32,
    pub rating: i64,
    pub bucket: i64,
}

impl LeaderboardPage {
    pub fn entries(&self) -> Result<Vec<LeaderboardEntry>, String> {
        let col = |name: &str| self.columns.iter().position(|c| c == name).ok_or(format!("leaderboard column {name} missing"));
        let (toon, gw, rating, bucket) = (col("toon")?, col("gateway_id")?, col("rating")?, col("bucket")?);
        self.rows
            .iter()
            .map(|r| {
                let int = |i: usize| r.get(i).and_then(Value::as_i64).ok_or(format!("leaderboard cell {} not an integer", self.columns[i]));
                Ok(LeaderboardEntry {
                    toon: r.get(toon).and_then(Value::as_str).ok_or("leaderboard toon not a string")?.to_string(),
                    gateway_id: int(gw)? as u32,
                    rating: int(rating)?,
                    bucket: int(bucket)?,
                })
            })
            .collect()
    }
}

#[derive(Deserialize, Debug)]
pub struct Profile {
    #[serde(default)]
    pub replays: Vec<ProfileReplay>,
}

#[derive(Deserialize, Debug)]
pub struct ProfileReplay {
    #[serde(default)]
    pub attributes: BTreeMap<String, Value>,
    pub create_time: u64,
    pub link: String,
}

impl ProfileReplay {
    /// 043-R5: human 1v1 ladder game.
    pub fn is_ladder_1v1(&self) -> bool {
        self.link.starts_with(LADDER_LINK_PREFIX) && self.attributes.get("replay_humans").and_then(Value::as_str) == Some("2")
    }
}

#[derive(Deserialize, Debug)]
pub struct MatchInfo {
    pub players: BTreeMap<String, MatchPlayer>,
    pub replays: Vec<MatchReplay>,
}

#[derive(Deserialize, Debug)]
pub struct MatchPlayer {
    /// The player's toon.
    pub name: String,
    pub gateway_id: u32,
    pub game_info: GameInfo,
    /// Results as this player's client reported them, keyed by toon ("" for empty slots).
    pub game_result: BTreeMap<String, GameResult>,
    pub info_attributes: BTreeMap<String, Value>,
    pub score: Score,
}

#[derive(Deserialize, Debug)]
pub struct GameInfo {
    pub attributes: BTreeMap<String, Value>,
}

#[derive(Deserialize, Debug)]
pub struct GameResult {
    pub result: String,
    #[serde(default)]
    pub attributes: BTreeMap<String, Value>,
}

#[derive(Deserialize, Debug)]
pub struct Score {
    pub base: i64,
    pub bucket_new: i64,
    #[serde(default)]
    pub season_id: Option<u32>,
}

#[derive(Deserialize, Debug)]
pub struct MatchReplay {
    pub md5: String,
    pub url: String,
}

impl MatchPlayer {
    /// Race actually played: the player's own game result has it (`terran`); the matchmaking
    /// attribute is the race picked, which can be `Random`.
    pub fn race(&self) -> Option<&str> {
        self.game_result
            .get(&self.name)
            .and_then(|g| g.attributes.get("race"))
            .and_then(Value::as_str)
            .or_else(|| self.info_attributes.get("race").and_then(Value::as_str))
    }

    pub fn game_attr(&self, key: &str) -> Option<&str> {
        self.game_info.attributes.get(key).and_then(Value::as_str)
    }
}

pub const RESULT_WIN: &str = "Win";
pub const RESULT_LOSS: &str = "Loss";

impl MatchInfo {
    /// 043 Risks: the winner only when every player's report says Win for one toon and Loss
    /// for the other.
    pub fn agreed_winner(&self) -> Option<&str> {
        let toons: Vec<&str> = self.players.values().map(|p| p.name.as_str()).collect();
        if toons.len() != 2 {
            return None;
        }
        let mut winner = None;
        for t in &toons {
            let reports: Vec<&str> = self.players.values().map(|p| p.game_result.get(*t).map_or("", |r| r.result.as_str())).collect();
            if reports.iter().all(|r| *r == RESULT_WIN) {
                winner = Some(*t);
            } else if !reports.iter().all(|r| *r == RESULT_LOSS) {
                return None;
            }
        }
        winner
    }
}

/// Percent-encodes one path segment.
pub fn enc(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => (b as char).to_string(),
            _ => format!("%{b:02X}"),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn player(name: &str, results: &[(&str, &str)]) -> Value {
        let gr: serde_json::Map<String, Value> = results.iter().map(|(t, r)| (t.to_string(), json!({"result": r, "attributes": {}}))).collect();
        json!({"name": name, "gateway_id": 30, "game_info": {"attributes": {"map_name": "Map"}},
               "game_result": gr, "info_attributes": {"race": "Terran"}, "score": {"base": 2000, "bucket_new": 6, "season_id": 20}})
    }

    #[test]
    fn r043_winner_needs_agreeing_reports() {
        let agree = json!({"players": {"1": player("a", &[("a", "Win"), ("b", "Loss"), ("", "Undecided")]),
                                       "2": player("b", &[("a", "Win"), ("b", "Loss")])}, "replays": []});
        let m: MatchInfo = serde_json::from_value(agree).unwrap();
        assert_eq!(m.agreed_winner(), Some("a"));
        let split = json!({"players": {"1": player("a", &[("a", "Undecided"), ("b", "Disconnect")]),
                                       "2": player("b", &[("a", "Loss"), ("b", "Win")])}, "replays": []});
        let m: MatchInfo = serde_json::from_value(split).unwrap();
        assert_eq!(m.agreed_winner(), None);
    }

    #[test]
    fn race_played_beats_race_picked() {
        let mut p = player("a", &[]);
        p["info_attributes"]["race"] = json!("Random");
        p["game_result"] = json!({"a": {"result": "Win", "attributes": {"race": "zerg"}}});
        let m: MatchPlayer = serde_json::from_value(p).unwrap();
        assert_eq!(m.race(), Some("zerg"));
        let m: MatchPlayer = serde_json::from_value(player("b", &[("b", "Win")])).unwrap();
        assert_eq!(m.race(), Some("Terran"));
    }

    #[test]
    fn r043_r10_leaderboard_table_by_column_name() {
        let p: LeaderboardPage = serde_json::from_value(json!({
            "columns": ["rank", "toon", "gateway_id", "rating", "bucket"],
            "rows": [[1, "x", 30, 2500, 7], [2, "y", 45, 2100, 6]]
        }))
        .unwrap();
        let e = p.entries().unwrap();
        assert_eq!(e[1], LeaderboardEntry { toon: "y".into(), gateway_id: 45, rating: 2100, bucket: 6 });
        let bad: LeaderboardPage = serde_json::from_value(json!({"columns": ["rank"], "rows": []})).unwrap();
        assert!(bad.entries().is_err());
    }

    #[test]
    fn path_segment_encoding() {
        assert_eq!(enc("a b/#c"), "a%20b%2F%23c");
        assert_eq!(enc("Toon_1.x"), "Toon_1.x");
    }
}
