use serde::Serialize;

#[derive(Serialize, Debug, Clone, PartialEq, Eq)]
pub struct Player {
    pub index: u8,
    pub slot: u16,
    pub id: u8,
    pub kind: u8,
    pub race: u8,
    pub team: u8,
    #[serde(skip)]
    pub name: Vec<u8>,
}

#[derive(Serialize, Debug, Clone)]
pub struct Header {
    pub frames: u32,
    pub map_width: u16,
    pub map_height: u16,
    pub game_type: u16,
    pub map_name: String,
    pub players: Vec<Player>,
}

pub const RACE_ZERG: u8 = 0;
pub const RACE_TERRAN: u8 = 1;
pub const RACE_PROTOSS: u8 = 2;
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

pub fn decode_title(b: &[u8]) -> String {
    match std::str::from_utf8(b) {
        Ok(s) => return s.to_string(),
        Err(e) if e.error_len().is_none() && e.valid_up_to() > 0 => return String::from_utf8_lossy(&b[..e.valid_up_to()]).into_owned(),
        Err(_) => {}
    }
    let cp949 = |b: &[u8]| encoding_rs::EUC_KR.decode_without_bom_handling_and_without_replacement(b).map(|s| s.into_owned());
    cp949(b).or_else(|| b.split_last().and_then(|(_, head)| cp949(head))).unwrap_or_else(|| String::from_utf8_lossy(b).into_owned())
}

fn cstr(b: &[u8]) -> &[u8] {
    &b[..b.iter().position(|&c| c == 0).unwrap_or(b.len())]
}

impl Header {
    pub fn humans(&self) -> impl Iterator<Item = &Player> {
        self.players.iter().filter(|p| p.kind == 2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_title_cp949_and_utf8_040_r15() {
        assert_eq!(decode_title(&[0xc5, 0xf5, 0xc8, 0xa5, b' ', b'1', b'.', b'3']), "투혼 1.3");
        assert_eq!(decode_title(&[0xc5, 0xf5, 0xc8]), "투");
        assert_eq!(decode_title("\u{7}투혼 \u{5}1.4".as_bytes()), "\u{7}투혼 \u{5}1.4");
        let utf8 = "| 서킷브레이커".as_bytes();
        assert_eq!(decode_title(&utf8[..utf8.len() - 1]), "| 서킷브레이");
        assert_eq!(decode_title(b"| iCCup | Fighting Spirit "), "| iCCup | Fighting Spirit ");
        assert!(decode_title(&[0xff, 0xff, 0xff]).contains('\u{fffd}'));
        let mut h = vec![0u8; crate::container::HEADER_SIZE];
        h[0x61..0x65].copy_from_slice(&[0xc5, 0xf5, 0xc8, 0xa5]);
        assert_eq!(parse(&h).map_name, "투혼");
    }
}
