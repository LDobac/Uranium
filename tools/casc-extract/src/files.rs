//! Which files OpenBW needs (from `third_party/openbw/bwgame.h` `global_init`) and how to
//! resolve them against the file list of a StarCraft: Remastered CASC storage.

use std::collections::HashMap;

use anyhow::{Result, bail};

/// Files read unconditionally by OpenBW's `global_init`.
pub fn fixed_files() -> Vec<String> {
    let mut v: Vec<String> = [
        "arr/units.dat",
        "arr/weapons.dat",
        "arr/upgrades.dat",
        "arr/techdata.dat",
        "arr/flingy.dat",
        "arr/sprites.dat",
        "arr/images.dat",
        "arr/orders.dat",
        "arr/images.tbl",
        "scripts/iscript.bin",
        "triggers/Melee.trg",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    // Same order and spelling as `tileset_names` in bwgame.h.
    for ts in ["badlands", "platform", "install", "AshWorld", "Jungle", "Desert", "Ice", "Twilight"] {
        v.push(format!("Tileset/{ts}.vf4"));
        v.push(format!("Tileset/{ts}.cv5"));
    }
    v
}

/// Parses a `.tbl` string table: `u16 count`, `u16 offsets[count]`, NUL-terminated strings.
pub fn parse_tbl(data: &[u8]) -> Result<Vec<String>> {
    let u16_at = |i: usize| -> Result<usize> {
        match data.get(i..i + 2) {
            Some(b) => Ok(u16::from_le_bytes([b[0], b[1]]) as usize),
            None => bail!("tbl truncated at {i}"),
        }
    };
    let count = u16_at(0)?;
    (0..count)
        .map(|i| {
            let off = u16_at(2 + i * 2)?;
            let end = data.get(off..).and_then(|s| s.iter().position(|&c| c == 0)).map(|p| off + p);
            match end {
                Some(end) => Ok(String::from_utf8_lossy(&data[off..end]).into_owned()),
                None => bail!("tbl entry {i} not terminated"),
            }
        })
        .collect()
}

/// `unit/<name>` paths for every images.tbl entry (GRPs and LO* overlays loaded by OpenBW).
pub fn unit_files(images_tbl: &[u8]) -> Result<Vec<String>> {
    let mut v: Vec<String> = parse_tbl(images_tbl)?
        .into_iter()
        .filter(|s| !s.is_empty())
        .map(|s| format!("unit/{}", s.replace('\\', "/")))
        .collect();
    v.sort();
    v.dedup();
    Ok(v)
}

/// Lower-case, forward slashes, no `namespace:` prefix.
fn normalize(name: &str) -> String {
    let n = name.trim().replace('\\', "/").to_ascii_lowercase();
    match n.split_once(':') {
        Some((ns, rest)) if !ns.contains('/') => rest.trim_start_matches('/').to_string(),
        _ => n,
    }
}

/// Index of a CASC listing, resolving OpenBW paths to stored names.
///
/// Preference: the classic path itself, then an `sd/` copy. HD asset trees are never used.
pub struct Index {
    by_norm: HashMap<String, String>,
}

impl Index {
    pub fn new<'a>(listing: impl IntoIterator<Item = &'a str>) -> Self {
        let mut by_norm = HashMap::new();
        for name in listing {
            if !name.trim().is_empty() {
                by_norm.entry(normalize(name)).or_insert_with(|| name.trim().to_string());
            }
        }
        Self { by_norm }
    }

    pub fn len(&self) -> usize {
        self.by_norm.len()
    }

    pub fn resolve(&self, path: &str) -> Option<&str> {
        let n = normalize(path);
        [n.clone(), format!("sd/{n}")].iter().find_map(|k| self.by_norm.get(k)).map(String::as_str)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_list_matches_openbw() {
        let f = fixed_files();
        assert_eq!(f.len(), 11 + 16);
        assert!(f.contains(&"scripts/iscript.bin".to_string()));
        assert!(f.contains(&"Tileset/AshWorld.cv5".to_string()));
    }

    #[test]
    fn tbl_roundtrip() {
        // 2 entries: "terran\\marine.grp" at offset 6, "" at offset 24
        let mut d = vec![2, 0, 6, 0, 24, 0];
        d.extend_from_slice(b"terran\\marine.grp\0\0");
        assert_eq!(parse_tbl(&d).unwrap(), vec!["terran\\marine.grp".to_string(), String::new()]);
        assert_eq!(unit_files(&d).unwrap(), vec!["unit/terran/marine.grp".to_string()]);
    }

    #[test]
    fn tbl_truncated_is_error() {
        assert!(parse_tbl(&[5, 0, 1]).is_err());
    }

    #[test]
    fn resolve_handles_case_backslash_namespace_and_sd() {
        let idx = Index::new(["data:Arr\\Units.dat", "SD\\unit\\terran\\marine.grp", "HD2\\unit\\terran\\marine.grp", "tileset/Jungle.cv5"]);
        assert_eq!(idx.resolve("arr/units.dat"), Some("data:Arr\\Units.dat"));
        assert_eq!(idx.resolve("unit/terran/marine.grp"), Some("SD\\unit\\terran\\marine.grp"));
        assert_eq!(idx.resolve("Tileset/Jungle.cv5"), Some("tileset/Jungle.cv5"));
        assert_eq!(idx.resolve("arr/orders.dat"), None);
    }

    #[test]
    fn classic_path_preferred_over_sd() {
        let idx = Index::new(["SD/arr/images.tbl", "arr/images.tbl"]);
        assert_eq!(idx.resolve("arr/images.tbl"), Some("arr/images.tbl"));
    }
}
