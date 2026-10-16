//! Replay file containers (spec 013 "리플레이 명령 인코딩"): reading legacy (1.16.1, PKWare
//! implode), modern (1.18–1.20, zlib) and modern 1.21+ (`seRS`) files, and writing the legacy
//! container with stored (uncompressed) chunks that OpenBW's `replay.h` loader accepts (013-R6).

use anyhow::{Context, Result, bail, ensure};
use serde::Serialize;

/// Size of the game header section (`replay.h`: `game_info_buffer`).
pub const HEADER_SIZE: usize = 0x279;
/// Legacy replay id `reRS`.
const ID_LEGACY: u32 = 0x5352_6572;
/// 1.21+ replay id `seRS`.
const ID_121: u32 = 0x5352_6573;
/// Every chunk inflates to at most this many bytes.
const CHUNK: usize = 8192;

#[derive(Serialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// 1.16.1 and older: PKWare implode.
    Legacy,
    /// 1.18–1.20: `reRS` with zlib chunks.
    Modern,
    /// 1.21+: `seRS`, zlib, extra length field after the id section.
    Modern121,
}

/// The four sections OpenBW needs; modern-only sections are dropped (013).
#[derive(Debug, Clone)]
pub struct Sections {
    pub format: Format,
    pub header: Vec<u8>,
    pub commands: Vec<u8>,
    pub map: Vec<u8>,
    /// Unit limit from the SC:R `LMTS` section (3400 when the extended limit is on); `None`
    /// for legacy replays and modern ones without the section (1700).
    pub unit_limit: Option<u32>,
}

/// Detects the container format like screp's `detectRepFormat`.
pub fn detect(file: &[u8]) -> Result<Format> {
    ensure!(file.len() >= 30, "file too short ({} bytes)", file.len());
    if file[12] == b's' {
        return Ok(Format::Modern121);
    }
    Ok(if file[28] == 0x78 { Format::Modern } else { Format::Legacy })
}

pub fn read(file: &[u8]) -> Result<Sections> {
    let format = detect(file)?;
    let mut r = Reader { b: file, pos: 0, format };
    let id = u32::from_le_bytes(r.section(4)?.try_into().unwrap());
    ensure!(id == ID_LEGACY || id == ID_121, "not a replay (id {id:#x})");
    if format == Format::Modern121 {
        r.u32()?; // encoded length of the remaining file
    }
    let header = r.section(HEADER_SIZE).context("header section")?;
    let cmd_len = u32::from_le_bytes(r.section(4)?.try_into().unwrap()) as usize;
    let commands = r.section(cmd_len).context("commands section")?;
    let map_len = u32::from_le_bytes(r.section(4)?.try_into().unwrap()) as usize;
    let map = r.section(map_len).context("map section")?;
    let unit_limit = if format == Format::Legacy { None } else { r.modern_unit_limit() };
    Ok(Sections { format, header, commands, map, unit_limit })
}

/// Writes a legacy (`reRS`) container with stored chunks (013-R6).
pub fn write_legacy(header: &[u8], commands: &[u8], map: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(header.len() + commands.len() + map.len() + 256);
    for data in [
        &ID_LEGACY.to_le_bytes()[..],
        header,
        &(commands.len() as u32).to_le_bytes()[..],
        commands,
        &(map.len() as u32).to_le_bytes()[..],
        map,
    ] {
        put_section(&mut out, data);
    }
    out
}

fn put_section(out: &mut Vec<u8>, data: &[u8]) {
    // Like Blizzard, an empty section is not written at all (only possible for a replay whose
    // commands were all observer / no-effect commands; nothing to replay).
    if data.is_empty() {
        return;
    }
    out.extend_from_slice(&crc32(data).to_le_bytes());
    out.extend_from_slice(&(data.len().div_ceil(CHUNK) as u32).to_le_bytes());
    for chunk in data.chunks(CHUNK) {
        out.extend_from_slice(&(chunk.len() as u32).to_le_bytes());
        out.extend_from_slice(chunk);
    }
}

/// Replay section checksum: CRC-32 (0xEDB88320) without the final inversion, as in
/// OpenBW `replay.h` `crc32_t`.
pub fn crc32(data: &[u8]) -> u32 {
    let mut r = 0xffff_ffffu32;
    for &b in data {
        r ^= b as u32;
        for _ in 0..8 {
            r = (r >> 1) ^ if r & 1 != 0 { 0xedb8_8320 } else { 0 };
        }
    }
    r
}

struct Reader<'a> {
    b: &'a [u8],
    pos: usize,
    format: Format,
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8]> {
        ensure!(self.pos + n <= self.b.len(), "truncated at {} (+{n})", self.pos);
        let s = &self.b[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    /// Modern files continue with PlayerNames (0x300) and tagged sections `(id, raw size, data)`.
    /// Only `LMTS` (limits, units at 0x0c) matters for command conversion (013).
    fn modern_unit_limit(&mut self) -> Option<u32> {
        const LMTS: u32 = u32::from_le_bytes(*b"LMTS");
        // PlayerNames: 0x300 bytes before 1.21, larger since; inflate whatever is there.
        self.modern_section_any().ok()?;
        while self.pos + 8 <= self.b.len() {
            let id = self.u32().ok()?;
            let raw = self.u32().ok()? as usize;
            let start = self.pos;
            if id == LMTS {
                let d = self.modern_section_any().ok()?;
                if d.len() < 0x10 {
                    return None;
                }
                return Some(u32::from_le_bytes(d[0x0c..0x10].try_into().unwrap()));
            }
            self.pos = start + raw;
        }
        None
    }

    /// A modern section of unknown size: every chunk is zlib or stored (screp `modernDecoder`).
    fn modern_section_any(&mut self) -> Result<Vec<u8>> {
        let _crc = self.u32()?;
        let count = self.u32()? as usize;
        let mut out = Vec::new();
        for _ in 0..count {
            let len = self.u32()? as usize;
            let data = self.take(len)?;
            if len > 4 && data[0] == 0x78 && (u16::from(data[0]) << 8 | u16::from(data[1])) % 31 == 0 {
                use std::io::Read;
                flate2::read::ZlibDecoder::new(data).read_to_end(&mut out).context("zlib chunk")?;
            } else {
                out.extend_from_slice(data);
            }
        }
        Ok(out)
    }

    /// One section: crc32, chunk count, then (length, data) chunks inflating to `size` bytes.
    fn section(&mut self, size: usize) -> Result<Vec<u8>> {
        // Blizzard writes nothing for an empty section (screp `sectionHeader`).
        if size == 0 {
            return Ok(Vec::new());
        }
        let _crc = self.u32()?;
        let count = self.u32()? as usize;
        let mut out = Vec::with_capacity(size);
        for _ in 0..count {
            let len = self.u32()? as usize;
            let expect = (size - out.len()).min(CHUNK);
            let format = self.format;
            let data = self.take(len)?;
            match format {
                _ if len == expect => out.extend_from_slice(data),
                Format::Legacy => {
                    let mut x = explode::explode(data).map_err(|e| anyhow::anyhow!("implode chunk: {e:?}"))?;
                    x.truncate(expect);
                    out.extend_from_slice(&x);
                }
                Format::Modern | Format::Modern121 => {
                    if len > 4 && data[0] == 0x78 && (u16::from(data[0]) << 8 | u16::from(data[1])) % 31 == 0 {
                        use std::io::Read;
                        let mut x = Vec::with_capacity(expect);
                        flate2::read::ZlibDecoder::new(data).read_to_end(&mut x).context("zlib chunk")?;
                        out.extend_from_slice(&x);
                    } else {
                        out.extend_from_slice(data);
                    }
                }
            }
        }
        if out.len() != size {
            bail!("section inflated to {} bytes, expected {size}", out.len());
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_matches_openbw_variant() {
        // Standard CRC-32 of "123456789" is 0xCBF43926; replay.h skips the final inversion.
        assert_eq!(crc32(b"123456789"), !0xCBF4_3926u32);
    }

    #[test]
    fn legacy_writer_round_trips_013_r6() {
        let header: Vec<u8> = (0..HEADER_SIZE).map(|i| i as u8).collect();
        let commands: Vec<u8> = (0..20_000u32).map(|i| (i * 7) as u8).collect();
        let map = vec![1u8, 2, 3];
        let file = write_legacy(&header, &commands, &map);
        assert_eq!(detect(&file).unwrap(), Format::Legacy);
        let s = read(&file).unwrap();
        assert_eq!((s.header, s.commands, s.map), (header, commands, map));
    }
}
