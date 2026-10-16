//! Map CHK normalisation for OpenBW (013 "맵 CHK 정규화"): OpenBW's map loader takes CHK `VER`
//! 59 / 63 / 205 and requires `STR `; SC:R maps saved as VER 64 / 206 carry the extended string
//! table `STRx` (u32 count and offsets) instead. The converted replay gets VER 63 / 205 and an
//! equivalent `STR ` (u16); everything else is copied byte for byte.

use anyhow::{Result, bail};

/// SC:R map versions with `STRx` and their 1.16.1 counterparts.
const VERSIONS: [(u16, u16); 2] = [(64, 63), (206, 205)];

/// One CHK section in file order; `None` name marks trailing bytes that are not a valid
/// section (map protection), kept verbatim.
struct Section<'a> {
    name: Option<[u8; 4]>,
    body: &'a [u8],
}

fn sections(m: &[u8]) -> Vec<Section<'_>> {
    let mut out = Vec::new();
    let mut p = 0;
    while p < m.len() {
        if p + 8 > m.len() {
            out.push(Section { name: None, body: &m[p..] });
            break;
        }
        let name: [u8; 4] = m[p..p + 4].try_into().unwrap();
        let n = i32::from_le_bytes(m[p + 4..p + 8].try_into().unwrap());
        if n < 0 || p + 8 + n as usize > m.len() {
            out.push(Section { name: None, body: &m[p..] });
            break;
        }
        out.push(Section { name: Some(name), body: &m[p + 8..p + 8 + n as usize] });
        p += 8 + n as usize;
    }
    out
}

/// `STRx` -> `STR `: same strings in the same order; offsets outside the data become "".
fn strx_to_str(b: &[u8]) -> Result<Vec<u8>> {
    if b.len() < 4 {
        bail!("STRx section of {} bytes", b.len());
    }
    let count = u32::from_le_bytes(b[0..4].try_into().unwrap()) as usize;
    let table_end = 4 + 4 * count;
    if count > u16::MAX as usize || table_end > b.len() {
        bail!("STRx with {count} strings does not fit");
    }
    let string_at = |i: usize| -> &[u8] {
        let off = u32::from_le_bytes(b[4 + 4 * i..8 + 4 * i].try_into().unwrap()) as usize;
        if off < table_end || off >= b.len() {
            return &[];
        }
        let s = &b[off..];
        &s[..s.iter().position(|&c| c == 0).unwrap_or(s.len())]
    };
    let header = 2 + 2 * count;
    let mut data: Vec<u8> = vec![0]; // shared empty string
    let mut offsets = Vec::with_capacity(count);
    let mut seen: std::collections::HashMap<&[u8], usize> = std::collections::HashMap::new();
    for i in 0..count {
        let s = string_at(i);
        let rel = if s.is_empty() {
            0
        } else {
            *seen.entry(s).or_insert_with(|| {
                let at = data.len();
                data.extend_from_slice(s);
                data.push(0);
                at
            })
        };
        offsets.push(header + rel);
    }
    if header + data.len() > u16::MAX as usize + 1 || offsets.iter().any(|&o| o > u16::MAX as usize) {
        bail!("STRx strings ({} bytes) do not fit a u16 STR", header + data.len());
    }
    let mut out = Vec::with_capacity(header + data.len());
    out.extend_from_slice(&(count as u16).to_le_bytes());
    for o in offsets {
        out.extend_from_slice(&(o as u16).to_le_bytes());
    }
    out.extend_from_slice(&data);
    Ok(out)
}

/// The CHK OpenBW can load. Returns the input unchanged when it has no SC:R-only sections.
pub fn to_legacy(m: &[u8]) -> Result<Vec<u8>> {
    let secs = sections(m);
    let has_strx = secs.iter().any(|s| s.name == Some(*b"STRx"));
    let has_new_ver = secs.iter().any(|s| s.name == Some(*b"VER ") && s.body.len() >= 2 && VERSIONS.iter().any(|v| v.0 == u16::from_le_bytes([s.body[0], s.body[1]])));
    if !has_strx && !has_new_ver {
        return Ok(m.to_vec());
    }
    let mut out = Vec::with_capacity(m.len());
    for s in &secs {
        match s.name {
            // Trailing bytes that are not a section go out as they came.
            None => out.extend_from_slice(s.body),
            Some(n) if &n == b"VER " && s.body.len() >= 2 => {
                let v = u16::from_le_bytes([s.body[0], s.body[1]]);
                let v = VERSIONS.iter().find(|x| x.0 == v).map_or(v, |x| x.1);
                let mut body = s.body.to_vec();
                body[..2].copy_from_slice(&v.to_le_bytes());
                put(&mut out, b"VER ", &body);
            }
            Some(n) if &n == b"STRx" => put(&mut out, b"STR ", &strx_to_str(s.body)?),
            Some(n) if &n == b"STR " && has_strx => {}
            Some(n) => put(&mut out, &n, s.body),
        }
    }
    Ok(out)
}

fn put(out: &mut Vec<u8>, name: &[u8; 4], body: &[u8]) {
    out.extend_from_slice(name);
    out.extend_from_slice(&(body.len() as i32).to_le_bytes());
    out.extend_from_slice(body);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sec(name: &[u8; 4], body: &[u8]) -> Vec<u8> {
        let mut v = name.to_vec();
        v.extend_from_slice(&(body.len() as i32).to_le_bytes());
        v.extend_from_slice(body);
        v
    }

    fn strx(strings: &[&[u8]]) -> Vec<u8> {
        let n = strings.len();
        let mut offs = Vec::new();
        let mut data = Vec::new();
        for s in strings {
            offs.push((4 + 4 * n + data.len()) as u32);
            data.extend_from_slice(s);
            data.push(0);
        }
        let mut b = (n as u32).to_le_bytes().to_vec();
        for o in offs {
            b.extend_from_slice(&o.to_le_bytes());
        }
        b.extend_from_slice(&data);
        b
    }

    /// Reads a `STR ` body the way OpenBW's `tag_funcs["STR "]` does.
    fn openbw_strings(b: &[u8]) -> Vec<Vec<u8>> {
        let n = u16::from_le_bytes([b[0], b[1]]) as usize;
        (0..n)
            .map(|i| {
                let o = u16::from_le_bytes([b[2 + 2 * i], b[3 + 2 * i]]) as usize;
                let s = &b[o..];
                s[..s.iter().position(|&c| c == 0).unwrap()].to_vec()
            })
            .collect()
    }

    #[test]
    fn scr_ver_and_strx_become_openbw_chk_013() {
        let strings: [&[u8]; 4] = [b"KnockOut 1.4", b"", b"Force 1", b"KnockOut 1.4"];
        let m = [sec(b"VER ", &206u16.to_le_bytes()), sec(b"DIM ", &[128, 0, 128, 0]), sec(b"STRx", &strx(&strings)), sec(b"SPRP", &[1, 0, 3, 0])].concat();
        let out = to_legacy(&m).unwrap();
        let secs = sections(&out);
        let names: Vec<[u8; 4]> = secs.iter().map(|s| s.name.unwrap()).collect();
        assert_eq!(names, vec![*b"VER ", *b"DIM ", *b"STR ", *b"SPRP"]);
        assert_eq!(secs[0].body, &205u16.to_le_bytes());
        assert_eq!(secs[1].body, &[128, 0, 128, 0]);
        let got = openbw_strings(secs[2].body);
        assert_eq!(got, strings.iter().map(|s| s.to_vec()).collect::<Vec<_>>());
        assert_eq!(secs[3].body, &[1, 0, 3, 0]);
    }

    #[test]
    fn legacy_chk_is_unchanged_013() {
        let m = [sec(b"VER ", &205u16.to_le_bytes()), sec(b"STR ", &[1, 0, 4, 0, b'a', 0])].concat();
        assert_eq!(to_legacy(&m).unwrap(), m);
        let v64 = [sec(b"VER ", &64u16.to_le_bytes()), sec(b"STRx", &strx(&[b"x"]))].concat();
        assert_eq!(sections(&to_legacy(&v64).unwrap())[0].body, &63u16.to_le_bytes());
    }

    #[test]
    fn strx_replaces_a_stale_str_and_bad_offsets_become_empty_013() {
        // Two strings: "ok" right after the table (offset 12) and an offset past the end.
        let mut b = 2u32.to_le_bytes().to_vec();
        b.extend_from_slice(&12u32.to_le_bytes());
        b.extend_from_slice(&9999u32.to_le_bytes());
        b.extend_from_slice(b"ok\0");
        let m = [sec(b"VER ", &206u16.to_le_bytes()), sec(b"STR ", &[0, 0]), sec(b"STRx", &b)].concat();
        let out = to_legacy(&m).unwrap();
        let secs = sections(&out);
        assert_eq!(secs.iter().filter(|s| s.name == Some(*b"STR ")).count(), 1);
        assert_eq!(openbw_strings(secs.iter().find(|s| s.name == Some(*b"STR ")).unwrap().body), vec![b"ok".to_vec(), vec![]]);
    }

    #[test]
    fn trailing_garbage_is_kept_013() {
        let mut m = [sec(b"VER ", &206u16.to_le_bytes()), sec(b"STRx", &strx(&[b"x"]))].concat();
        m.extend_from_slice(b"JUNKJUNK\xff\xff");
        let out = to_legacy(&m).unwrap();
        assert!(out.ends_with(b"JUNKJUNK\xff\xff"));
    }
}
