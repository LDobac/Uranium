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
    /// Map title decoded by `decode_title` (UTF-8 on SC:R, CP949 on Korean 1.16.1 replays).
    pub map_name: String,
    /// Slots with a non-empty name, in slot order (screp `OrigPlayers`).
    pub players: Vec<Player>,
}

pub const RACE_ZERG: u8 = 0;
pub const RACE_TERRAN: u8 = 1;
pub const RACE_PROTOSS: u8 = 2;
/// Lobby race "Random" (screp). The replays measured so far (1.16.1 and SC:R, 2026-10-19) store
/// the race actually played, so this only appears if a header keeps the lobby choice (030-R11).
pub const RACE_RANDOM: u8 = 6;

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
        map_name: decode_title(cstr(&h[0x61..0x61 + 26])),
        players,
    }
}

/// 040-R15 / A-1: map title bytes to text. SC:R stores UTF-8; Korean 1.16.1 replays store
/// CP949 ("투혼" used to become U+FFFD and fell into `other`). Valid UTF-8 is kept as is (CP949
/// Hangul is never valid UTF-8, while UTF-8 Hangul can also parse as CP949 pairs), then CP949
/// without replacement characters, and UTF-8 with replacement characters as the last resort.
pub fn decode_title(b: &[u8]) -> String {
    match std::str::from_utf8(b) {
        Ok(s) => return s.to_string(),
        // UTF-8 cut inside its last character by the 26-byte field: keep the valid prefix.
        Err(e) if e.error_len().is_none() && e.valid_up_to() > 0 => return String::from_utf8_lossy(&b[..e.valid_up_to()]).into_owned(),
        Err(_) => {}
    }
    let cp949 = |b: &[u8]| encoding_rs::EUC_KR.decode_without_bom_handling_and_without_replacement(b).map(|s| s.into_owned());
    // The 26-byte field can cut a two-byte character after its lead byte.
    cp949(b).or_else(|| b.split_last().and_then(|(_, head)| cp949(head))).unwrap_or_else(|| String::from_utf8_lossy(b).into_owned())
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A-1 / 040-R15: CP949 titles of 1.16.1 replays decode to Hangul, UTF-8 titles stay.
    #[test]
    fn map_title_cp949_and_utf8_040_r15() {
        // "투혼 1.3" in CP949.
        assert_eq!(decode_title(&[0xc5, 0xf5, 0xc8, 0xa5, b' ', b'1', b'.', b'3']), "투혼 1.3");
        // Truncated after a lead byte (26-byte field).
        assert_eq!(decode_title(&[0xc5, 0xf5, 0xc8]), "투");
        assert_eq!(decode_title("\u{7}투혼 \u{5}1.4".as_bytes()), "\u{7}투혼 \u{5}1.4");
        // UTF-8 cut inside the last character (26-byte field): the valid prefix, not CP949.
        let utf8 = "| 서킷브레이커".as_bytes();
        assert_eq!(decode_title(&utf8[..utf8.len() - 1]), "| 서킷브레이");
        assert_eq!(decode_title(b"| iCCup | Fighting Spirit "), "| iCCup | Fighting Spirit ");
        // Neither UTF-8 nor CP949: replacement characters, never a panic.
        assert!(decode_title(&[0xff, 0xff, 0xff]).contains('\u{fffd}'));
        let mut h = vec![0u8; crate::container::HEADER_SIZE];
        h[0x61..0x65].copy_from_slice(&[0xc5, 0xf5, 0xc8, 0xa5]);
        assert_eq!(parse(&h).map_name, "투혼");
    }
}
