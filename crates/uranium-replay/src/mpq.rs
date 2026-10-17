//! Minimal MPQ (format 0) reader for map files (.scm / .scx): enough to pull
//! `staredit\scenario.chk` out of the target maps for the live map precompute (031, AMB-031-4).
//! Supports encrypted hash / block tables, encrypted files (incl. fix-key), sectored or single-unit
//! files, zlib and PKWARE implode compression.

use std::io::Read;
use std::sync::OnceLock;

use anyhow::{Context, Result, bail, ensure};

pub const SCENARIO_CHK: &str = "staredit\\scenario.chk";

const FILE_IMPLODE: u32 = 0x0000_0100;
const FILE_COMPRESS: u32 = 0x0000_0200;
const FILE_ENCRYPTED: u32 = 0x0001_0000;
const FILE_FIX_KEY: u32 = 0x0002_0000;
const FILE_SINGLE_UNIT: u32 = 0x0100_0000;
const FILE_EXISTS: u32 = 0x8000_0000;
const HASH_EMPTY: u32 = 0xFFFF_FFFF;

fn crypt_table() -> &'static [u32; 0x500] {
    static T: OnceLock<[u32; 0x500]> = OnceLock::new();
    T.get_or_init(|| {
        let mut t = [0u32; 0x500];
        let mut seed: u32 = 0x0010_0001;
        for i in 0..0x100 {
            let mut idx = i;
            for _ in 0..5 {
                seed = (seed * 125 + 3) % 0x2A_AAAB;
                let hi = (seed & 0xFFFF) << 16;
                seed = (seed * 125 + 3) % 0x2A_AAAB;
                t[idx] = hi | (seed & 0xFFFF);
                idx += 0x100;
            }
        }
        t
    })
}

/// MPQ string hash; `kind` 0 = table offset, 1 / 2 = name checks, 3 = file key.
fn hash(name: &str, kind: u32) -> u32 {
    let t = crypt_table();
    let (mut s1, mut s2) = (0x7FED_7FEDu32, 0xEEEE_EEEEu32);
    for c in name.bytes().map(|b| b.to_ascii_uppercase()) {
        s1 = t[(kind * 0x100 + c as u32) as usize] ^ s1.wrapping_add(s2);
        s2 = (c as u32).wrapping_add(s1).wrapping_add(s2).wrapping_add(s2 << 5).wrapping_add(3);
    }
    s1
}

fn decrypt(data: &mut [u8], mut key: u32) {
    let t = crypt_table();
    let mut seed = 0xEEEE_EEEEu32;
    for w in data.chunks_exact_mut(4) {
        seed = seed.wrapping_add(t[0x400 + (key & 0xFF) as usize]);
        let v = u32::from_le_bytes(w.try_into().unwrap()) ^ key.wrapping_add(seed);
        key = ((!key << 0x15).wrapping_add(0x1111_1111)) | (key >> 0x0B);
        seed = v.wrapping_add(seed).wrapping_add(seed << 5).wrapping_add(3);
        w.copy_from_slice(&v.to_le_bytes());
    }
}

fn u32_at(b: &[u8], at: usize) -> Result<u32> {
    Ok(u32::from_le_bytes(b.get(at..at + 4).context("MPQ truncated")?.try_into().unwrap()))
}

fn table(archive: &[u8], base: usize, pos: u32, entries: u32, key_name: &str) -> Result<Vec<[u32; 4]>> {
    let start = base + pos as usize;
    let len = entries as usize * 16;
    let mut raw = archive.get(start..start + len).with_context(|| format!("{key_name} out of range"))?.to_vec();
    decrypt(&mut raw, hash(key_name, 3));
    Ok(raw.chunks_exact(16).map(|c| std::array::from_fn(|i| u32::from_le_bytes(c[i * 4..i * 4 + 4].try_into().unwrap()))).collect())
}

fn decompress(data: &[u8], flags: u32, expect: usize) -> Result<Vec<u8>> {
    if data.len() >= expect {
        return Ok(data[..expect].to_vec());
    }
    let mut out = if flags & FILE_IMPLODE != 0 {
        explode::explode(data).map_err(|e| anyhow::anyhow!("implode: {e:?}"))?
    } else {
        let (mask, body) = data.split_first().context("empty compressed sector")?;
        match mask {
            0x02 => {
                let mut x = Vec::with_capacity(expect);
                flate2::read::ZlibDecoder::new(body).read_to_end(&mut x).context("zlib sector")?;
                x
            }
            0x08 => explode::explode(body).map_err(|e| anyhow::anyhow!("implode sector: {e:?}"))?,
            m => bail!("unsupported MPQ compression mask {m:#04x}"),
        }
    };
    out.truncate(expect);
    ensure!(out.len() == expect, "sector decompressed to {} bytes, expected {expect}", out.len());
    Ok(out)
}

/// Reads one file from an MPQ archive held in memory.
pub fn read_file(archive: &[u8], name: &str) -> Result<Vec<u8>> {
    let base = archive.windows(4).step_by(512).position(|w| w == b"MPQ\x1a").context("no MPQ header")? * 512;
    let h = &archive[base..];
    let sector = 512usize << u16::from_le_bytes([h[14], h[15]]);
    let hashes = table(archive, base, u32_at(h, 16)?, u32_at(h, 24)?, "(hash table)")?;
    let blocks = table(archive, base, u32_at(h, 20)?, u32_at(h, 28)?, "(block table)")?;
    ensure!(!hashes.is_empty(), "empty hash table");
    let (n1, n2) = (hash(name, 1), hash(name, 2));
    let start = hash(name, 0) as usize % hashes.len();
    let block = (0..hashes.len())
        .map(|i| hashes[(start + i) % hashes.len()])
        .take_while(|e| e[3] != HASH_EMPTY)
        .find(|e| e[0] == n1 && e[1] == n2 && (e[3] as usize) < blocks.len())
        .map(|e| blocks[e[3] as usize])
        .with_context(|| format!("{name} not in archive"))?;
    let [offset, packed, size, flags] = block;
    ensure!(flags & FILE_EXISTS != 0, "{name}: deleted entry");
    let data = archive.get(base + offset as usize..base + offset as usize + packed as usize).context("file data out of range")?;
    let mut key = hash(name.rsplit('\\').next().unwrap_or(name), 3);
    if flags & FILE_FIX_KEY != 0 {
        key = key.wrapping_add(offset) ^ size;
    }
    let encrypted = flags & FILE_ENCRYPTED != 0;
    let compressed = flags & (FILE_IMPLODE | FILE_COMPRESS) != 0;
    let size = size as usize;
    if flags & FILE_SINGLE_UNIT != 0 {
        let mut d = data.to_vec();
        if encrypted {
            decrypt(&mut d, key);
        }
        return if compressed { decompress(&d, flags, size) } else { Ok(d[..size.min(d.len())].to_vec()) };
    }
    let count = size.div_ceil(sector);
    let offsets: Vec<usize> = if compressed {
        let mut t = data.get(..(count + 1) * 4).context("sector table out of range")?.to_vec();
        if encrypted {
            decrypt(&mut t, key.wrapping_sub(1));
        }
        t.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap()) as usize).collect()
    } else {
        (0..=count).map(|i| (i * sector).min(size)).collect()
    };
    let mut out = Vec::with_capacity(size);
    for i in 0..count {
        let mut s = data.get(offsets[i]..offsets[i + 1]).context("sector out of range")?.to_vec();
        if encrypted {
            decrypt(&mut s, key.wrapping_add(i as u32));
        }
        let expect = sector.min(size - i * sector);
        out.extend(if compressed { decompress(&s, flags, expect)? } else { s });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_matches_known_values() {
        // Well-known MPQ constants: the table keys used by every archive.
        assert_eq!(hash("(hash table)", 3), 0xC3AF_3770);
        assert_eq!(hash("(block table)", 3), 0xEC83_B3A3);
    }

    #[test]
    fn decrypt_inverts_encrypt() {
        // Encrypt with the reference algorithm, then decrypt.
        let plain: Vec<u8> = (0u8..64).collect();
        let key = 0x1234_5678u32;
        let t = crypt_table();
        let (mut k, mut seed) = (key, 0xEEEE_EEEEu32);
        let mut enc = plain.clone();
        for w in enc.chunks_exact_mut(4) {
            seed = seed.wrapping_add(t[0x400 + (k & 0xFF) as usize]);
            let v = u32::from_le_bytes(w.try_into().unwrap());
            let c = v ^ k.wrapping_add(seed);
            k = ((!k << 0x15).wrapping_add(0x1111_1111)) | (k >> 0x0B);
            seed = v.wrapping_add(seed).wrapping_add(seed << 5).wrapping_add(3);
            w.copy_from_slice(&c.to_le_bytes());
        }
        decrypt(&mut enc, key);
        assert_eq!(enc, plain);
    }
}
