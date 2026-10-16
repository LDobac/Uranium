//! Command stream (040-R1): frame blocks of `(player, opcode, payload)`, every 1.16.1 opcode
//! plus the SC:R 1.21 extensions, and their conversion to 1.16.1 form (013 table).

use anyhow::{Result, bail, ensure};
use serde::Serialize;

pub mod op {
    pub const SELECT: u8 = 0x09;
    pub const SELECT_ADD: u8 = 0x0a;
    pub const SELECT_REMOVE: u8 = 0x0b;
    pub const BUILD: u8 = 0x0c;
    pub const HOTKEY: u8 = 0x13;
    pub const RIGHT_CLICK: u8 = 0x14;
    pub const TARGETED_ORDER: u8 = 0x15;
    pub const TRAIN: u8 = 0x1f;
    pub const UNIT_MORPH: u8 = 0x23;
    pub const UNLOAD: u8 = 0x29;
    pub const CANCEL_TRAIN: u8 = 0x20;
    pub const LEAVE_GAME: u8 = 0x57;
    pub const RIGHT_CLICK_121: u8 = 0x60;
    pub const TARGETED_ORDER_121: u8 = 0x61;
    pub const UNLOAD_121: u8 = 0x62;
    pub const SELECT_121: u8 = 0x63;
    pub const SELECT_ADD_121: u8 = 0x64;
    pub const SELECT_REMOVE_121: u8 = 0x65;
}

#[derive(Serialize, Debug, Clone, PartialEq, Eq)]
pub struct Cmd {
    pub frame: u32,
    pub player: u8,
    pub op: u8,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Block {
    pub frame: u32,
    pub cmds: Vec<Cmd>,
}

/// Payload size of a fixed-size opcode, `None` for variable-size or unknown ones
/// (layouts from screp v1.13.8 `parseCommands`).
fn fixed_size(op: u8) -> Option<usize> {
    Some(match op {
        0x05 | 0x08 | 0x10 | 0x11 | 0x18 | 0x19 | 0x1b | 0x1c | 0x1d | 0x27 | 0x2a | 0x2e | 0x31 | 0x33 | 0x34 | 0x36
        | 0x38 | 0x39 | 0x3c | 0x54 | 0x5a | 0x5b => 0,
        0x0f | 0x1a | 0x1e | 0x21 | 0x22 | 0x25 | 0x26 | 0x28 | 0x2b | 0x2c | 0x2d | 0x30 | 0x32 | 0x3a | 0x3b | 0x3d
        | 0x42 | 0x43 | 0x55 | 0x57 => 1,
        0x0d | 0x13 | 0x1f | 0x20 | 0x23 | 0x29 | 0x35 | 0x41 | 0x44 | 0x45 => 2,
        // 0x5d: SC:R, undefined in screp; 4 bytes (always 0) from 2,000 ladder replays (013).
        0x0e | 0x12 | 0x2f | 0x58 | 0x5d | 0x62 => 4,
        0x3e => 5,
        0x37 => 6,
        0x0c | 0x3f => 7,
        0x14 | 0x56 => 9,
        0x15 => 10,
        0x60 => 11,
        0x48 | 0x61 => 12,
        0x40 => 17,
        0x5c => 81,
        _ => return None,
    })
}

/// Parses the commands section into frame blocks. An unknown opcode makes the whole replay
/// unsupported (013-R2) instead of guessing the rest of the block.
pub fn parse(b: &[u8]) -> Result<Vec<Block>> {
    let mut blocks = Vec::new();
    let mut pos = 0;
    while pos < b.len() {
        ensure!(pos + 5 <= b.len(), "truncated block header at {pos}");
        let frame = u32::from_le_bytes(b[pos..pos + 4].try_into().unwrap());
        let end = pos + 5 + b[pos + 4] as usize;
        ensure!(end <= b.len(), "block at {pos} runs past the end");
        pos += 5;
        let mut cmds = Vec::new();
        while pos < end {
            ensure!(pos + 2 <= end, "truncated command at frame {frame}");
            let (player, op) = (b[pos], b[pos + 1]);
            pos += 2;
            let size = match op {
                0x09..=0x0b => 1 + 2 * *b.get(pos).unwrap_or(&0) as usize,
                0x63..=0x65 => 1 + 4 * *b.get(pos).unwrap_or(&0) as usize,
                0x06 | 0x07 => {
                    ensure!(pos + 4 <= end, "truncated save/load at frame {frame}");
                    4 + u32::from_le_bytes(b[pos..pos + 4].try_into().unwrap()) as usize
                }
                _ => match fixed_size(op) {
                    Some(n) => n,
                    None => bail!("unknown opcode {op:#04x} at frame {frame}"),
                },
            };
            ensure!(pos + size <= end, "opcode {op:#04x} at frame {frame} runs past its block");
            cmds.push(Cmd { frame, player, op, data: b[pos..pos + size].to_vec() });
            pos += size;
        }
        blocks.push(Block { frame, cmds });
    }
    Ok(blocks)
}

/// What OpenBW's replay loader does with a 1.16.1 opcode (`third_party/openbw/actions.h`,
/// `read_action`): anything it does not handle stops the replay with "unknown action".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenBw {
    /// Executed by OpenBW.
    Supported,
    /// Not handled by OpenBW and without effect on the game (speed, pause, latency, network,
    /// lobby, save/load): dropped by the converter (013).
    DropNoEffect,
    /// Parsed, affects the game, not handled by OpenBW: the replay is set aside for review.
    Unhandled,
}

pub fn openbw(op: u8) -> OpenBw {
    match op {
        0x09..=0x0e | 0x12..=0x15 | 0x18..=0x1c | 0x1e..=0x23 | 0x25..=0x36 | 0x57 | 0x58 | 0x5a | 0x5c => OpenBw::Supported,
        0x05..=0x08 | 0x0f..=0x11 | 0x37..=0x45 | 0x48 | 0x54..=0x56 | 0x5b | 0x5d => OpenBw::DropNoEffect,
        _ => OpenBw::Unhandled,
    }
}

/// Serializes blocks back into a commands section.
pub fn encode(blocks: &[Block]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    for blk in blocks {
        let body: Vec<u8> = blk.cmds.iter().flat_map(|c| [c.player, c.op].into_iter().chain(c.data.iter().copied())).collect();
        ensure!(body.len() <= 255, "block at frame {} is {} bytes", blk.frame, body.len());
        out.extend_from_slice(&blk.frame.to_le_bytes());
        out.push(body.len() as u8);
        out.extend_from_slice(&body);
    }
    Ok(out)
}

/// 1.16.1 unit limit; tags are `index (11 bits) | recycle (5 bits) << 11`.
pub const LEGACY_UNIT_LIMIT: u32 = 1700;

/// Maps an SC:R unit tag to the 1.16.1 tag OpenBW assigns (013, measured in the M4 spike).
/// `full` is the 1.21 tag with its "pad" word as the high 16 bits: the pad carries the high bits
/// of the recycle counter. With an extended unit limit (`LMTS`, e.g. 3400) the index has 13 bits
/// and player units, allocated from the top of the array, sit `limit - 1700` higher than in
/// 1.16.1; preplaced map units keep their index. The 1.16.1 recycle counter has 5 bits.
/// 0 and 0xFFFF mean "no unit".
pub fn map_tag(full: u32, unit_limit: Option<u32>) -> Result<u16> {
    if full == 0 || full == 0xffff {
        return Ok(full as u16);
    }
    let limit = unit_limit.unwrap_or(LEGACY_UNIT_LIMIT).max(LEGACY_UNIT_LIMIT);
    let bits = if limit > LEGACY_UNIT_LIMIT { 13 } else { 11 };
    let (index, recycle) = (full & ((1 << bits) - 1), full >> bits);
    let shift = limit - LEGACY_UNIT_LIMIT;
    let legacy = if index >= shift { index - shift } else { index };
    // Indices are 1-based (index 3400 occurs with limit 3400).
    ensure!(legacy <= LEGACY_UNIT_LIMIT, "unit index {index} outside the 1.16.1 range (limit {limit})");
    Ok((legacy | (recycle & 0x1f) << 11) as u16)
}

/// Converts one command to its 1.16.1 form (013 table): SC:R 1.21 opcodes lose the pad word
/// after each unit tag, and every unit tag goes through `map_tag`.
pub fn to_legacy(c: &Cmd, unit_limit: Option<u32>) -> Result<Cmd> {
    let u16_at = |d: &[u8], a: usize| u16::from_le_bytes([d[a], d[a + 1]]) as u32;
    let tag = |d: &[u8], a: usize, padded: bool| -> Result<[u8; 2]> {
        let full = u16_at(d, a) | if padded { u16_at(d, a + 2) << 16 } else { 0 };
        Ok(map_tag(full, unit_limit)?.to_le_bytes())
    };
    let d = &c.data;
    let (op, data) = match c.op {
        op::SELECT_121..=op::SELECT_REMOVE_121 | op::SELECT..=op::SELECT_REMOVE => {
            let padded = c.op >= op::SELECT_121;
            let stride = if padded { 4 } else { 2 };
            let mut out = vec![d[0]];
            for i in 0..d[0] as usize {
                out.extend(tag(d, 1 + stride * i, padded)?);
            }
            (if padded { c.op - op::SELECT_121 + op::SELECT } else { c.op }, out)
        }
        op::RIGHT_CLICK_121 | op::TARGETED_ORDER_121 => {
            let out = [&d[..4], &tag(d, 4, true)?[..], &d[8..]].concat();
            (if c.op == op::RIGHT_CLICK_121 { op::RIGHT_CLICK } else { op::TARGETED_ORDER }, out)
        }
        op::RIGHT_CLICK | op::TARGETED_ORDER => (c.op, [&d[..4], &tag(d, 4, false)?[..], &d[6..]].concat()),
        op::UNLOAD_121 => (op::UNLOAD, tag(d, 0, true)?.to_vec()),
        op::UNLOAD | op::CANCEL_TRAIN => (c.op, tag(d, 0, false)?.to_vec()),
        _ => return Ok(c.clone()),
    };
    Ok(Cmd { frame: c.frame, player: c.player, op, data })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(frame: u32, body: &[u8]) -> Vec<u8> {
        let mut v = frame.to_le_bytes().to_vec();
        v.push(body.len() as u8);
        v.extend_from_slice(body);
        v
    }

    #[test]
    fn parses_and_reencodes_blocks_040_r1() {
        // select 2 units, right click, hotkey, leave
        let mut raw = block(10, &[0, 0x09, 2, 0x34, 0x12, 0x35, 0x12, 0, 0x14, 1, 0, 2, 0, 0xff, 0xff, 0xe4, 0, 0]);
        raw.extend(block(11, &[1, 0x13, 1, 3, 1, 0x57, 1]));
        let blocks = parse(&raw).unwrap();
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].cmds[0].op, op::SELECT);
        assert_eq!(blocks[0].cmds[0].data, vec![2, 0x34, 0x12, 0x35, 0x12]);
        assert_eq!(blocks[1].cmds.iter().map(|c| c.op).collect::<Vec<_>>(), vec![op::HOTKEY, op::LEAVE_GAME]);
        assert_eq!(encode(&blocks).unwrap(), raw);
    }

    #[test]
    fn unknown_opcode_is_unsupported_013_r2() {
        assert!(parse(&block(1, &[0, 0x77, 0])).is_err());
    }

    #[test]
    fn scr_commands_convert_to_legacy_013_r6() {
        let sel = Cmd { frame: 5, player: 0, op: op::SELECT_121, data: vec![2, 0x34, 0x12, 0, 0, 0x35, 0x12, 0, 0] };
        assert_eq!(to_legacy(&sel, None).unwrap(), Cmd { frame: 5, player: 0, op: op::SELECT, data: vec![2, 0x34, 0x12, 0x35, 0x12] });
        let rc = Cmd { frame: 5, player: 0, op: op::RIGHT_CLICK_121, data: vec![1, 0, 2, 0, 0xaa, 0xbb, 0, 0, 0xe4, 0, 1] };
        assert_eq!(to_legacy(&rc, None).unwrap().data, vec![1, 0, 2, 0, 0xaa, 0xbb, 0xe4, 0, 1]);
        assert_eq!(to_legacy(&rc, None).unwrap().op, op::RIGHT_CLICK);
        let to = Cmd { frame: 5, player: 0, op: op::TARGETED_ORDER_121, data: vec![1, 0, 2, 0, 0xaa, 0xbb, 0, 0, 0xe4, 0, 0x0e, 0] };
        assert_eq!(to_legacy(&to, None).unwrap().data.len(), fixed_size(op::TARGETED_ORDER).unwrap());
        let un = Cmd { frame: 5, player: 0, op: op::UNLOAD_121, data: vec![0xaa, 0xbb, 0, 0] };
        assert_eq!(to_legacy(&un, None).unwrap(), Cmd { frame: 5, player: 0, op: op::UNLOAD, data: vec![0xaa, 0xbb] });
        // A non-zero pad holds high recycle bits; 1.16.1 keeps 5 of them.
        let padded = Cmd { frame: 5, player: 0, op: op::UNLOAD_121, data: vec![0xaa, 0xbb, 1, 0] };
        assert_eq!(to_legacy(&padded, None).unwrap().data, vec![0xaa, 0xbb]);
    }
}

#[cfg(test)]
mod tag_tests {
    use super::*;

    /// Measured in the M4 spike: SC:R tag 0x2CC3 (3400-unit replay) is OpenBW unit 0xE1F.
    #[test]
    fn extended_limit_tags_map_to_legacy_013() {
        assert_eq!(map_tag(0x2cc3, Some(3400)).unwrap(), 0x0e1f);
        assert_eq!(map_tag(0x2cc3, None).unwrap(), 0x2cc3);
        assert_eq!(map_tag(0x2cc3, Some(1700)).unwrap(), 0x2cc3);
        assert_eq!(map_tag(0xffff, Some(3400)).unwrap(), 0xffff);
        assert_eq!(map_tag(0x0005, Some(3400)).unwrap(), 0x0005); // preplaced unit
        assert_eq!(map_tag(0x1_0c64, Some(3400)).unwrap(), (3172 - 1700) | 8 << 11); // pad = recycle bit 3
        assert_eq!(map_tag(0x0d48, Some(3400)).unwrap(), 1700);
        assert!(map_tag(0x1fff, Some(4000)).is_err());
        let sel = Cmd { frame: 1, player: 0, op: op::SELECT_121, data: vec![1, 0xc3, 0x2c, 0, 0] };
        assert_eq!(to_legacy(&sel, Some(3400)).unwrap().data, vec![1, 0x1f, 0x0e]);
    }
}

#[cfg(test)]
mod openbw_tests {
    use super::*;

    /// 013: every opcode we can parse is either executed by OpenBW, dropped as no-effect, or
    /// held for review; nothing is silently lost.
    #[test]
    fn opcode_classes_match_openbw_actions_h_013() {
        assert_eq!(openbw(op::SELECT), OpenBw::Supported);
        assert_eq!(openbw(op::TARGETED_ORDER), OpenBw::Supported);
        assert_eq!(openbw(op::LEAVE_GAME), OpenBw::Supported);
        assert_eq!(openbw(0x10), OpenBw::DropNoEffect); // pause
        assert_eq!(openbw(0x55), OpenBw::DropNoEffect); // latency
        assert_eq!(openbw(0x1d), OpenBw::Unhandled); // order nothing
        assert_eq!((fixed_size(0x5d), openbw(0x5d)), (Some(4), OpenBw::DropNoEffect)); // SC:R, 013
        for op in 0u8..=0x5c {
            if fixed_size(op).is_some() && !(0x60..=0x65).contains(&op) {
                let _ = openbw(op); // every parsable legacy opcode has a class
            }
        }
    }
}
