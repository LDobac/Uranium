//! Game header section (0x279 bytes), offsets as in screp `parseHeader`.

use serde::Serialize;

#[derive(Serialize, Debug, Clone, PartialEq, Eq)]
pub struct Player {
    /// Position in the header's 12-slot array. The BWAPI player id is the rank among occupied
    /// slots, i.e. the position in `Header::players` (013).
    pub index: u8,
    pub slot: u16,
    /// Player id used in the command stream.
    pub id: u8,
    /// 1 = computer, 2 = human (screp `PlayerType`).
    pub kind: u8,
    /// 0 Zerg, 1 Terran, 2 Protoss, 6 Random.
    pub race: u8,
    pub team: u8,
    /// Raw name bytes (encoding differs between versions). Never written to reports (040-R14).
    #[serde(skip)]
    pub name: Vec<u8>,
}

#[derive(Serialize, Debug, Clone)]
pub struct Header {
    pub frames: u32,
    pub map_width: u16,
    pub map_height: u16,
    /// screp `GameType` id (2 Melee, 3 FFA, 4 1on1, 10 UMS, 15 TvB, ...).
    pub game_type: u16,
    /// Map title as stored (may be EUC-KR/CP949 on old replays, UTF-8 on SC:R).
    pub map_name: String,
    /// Slots with a non-empty name, in slot order (screp `OrigPlayers`).
    pub players: Vec<Player>,
}

pub const RACE_ZERG: u8 = 0;
pub const RACE_TERRAN: u8 = 1;
pub const RACE_PROTOSS: u8 = 2;

pub fn parse(h: &[u8]) -> Header {
    let u16_at = |o: usize| u16::from_le_bytes([h[o], h[o + 1]]);
    let mut players = Vec::new();
    for i in 0..12 {
        let p = &h[0xa1 + i * 36..0xa1 + (i + 1) * 36];
        let name = cstr(&p[11..36]);
        if name.is_empty() {
            continue;
        }
        players.push(Player { index: i as u8, slot: u16::from_le_bytes([p[0], p[1]]), id: p[4], kind: p[8], race: p[9], team: p[10], name: name.to_vec() });
    }
    Header {
        frames: u32::from_le_bytes(h[1..5].try_into().unwrap()),
        map_width: u16_at(0x34),
        map_height: u16_at(0x36),
        game_type: u16_at(0x3c),
        map_name: String::from_utf8_lossy(cstr(&h[0x61..0x61 + 26])).into_owned(),
        players,
    }
}

fn cstr(b: &[u8]) -> &[u8] {
    &b[..b.iter().position(|&c| c == 0).unwrap_or(b.len())]
}

impl Header {
    /// Human players that are not observers. Observers get team 0 in melee-style games only
    /// on some versions, so the stricter filter is done by the dataset filter (040).
    pub fn humans(&self) -> impl Iterator<Item = &Player> {
        self.players.iter().filter(|p| p.kind == 2)
    }
}
