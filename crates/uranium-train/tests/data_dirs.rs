//! 041-R9 / 040-R13: shard folders are read with their subfolders (`<src>/<part>/shard_*`), in a
//! deterministic path order.

use std::fs;
use std::path::{Path, PathBuf};

use uranium_train::data::{shard_files, shard_keys};

fn touch(root: &Path, rel: &str) {
    let p = root.join(rel);
    fs::create_dir_all(p.parent().unwrap()).unwrap();
    fs::write(&p, b"").unwrap();
}

fn keys(dir: &Path) -> Vec<String> {
    shard_keys(dir).unwrap().into_iter().map(|(k, _)| k).collect()
}

/// 041-R13: in the build layout a shard's key is `<src>/<part>/<file>` whether the source folder
/// or its parent is given, and wherever (and however) that folder is spelled; the source name is
/// the folder's real name on disk.
#[test]
fn shard_keys_are_location_independent_041_r13() {
    let root = std::env::temp_dir().join(format!("uranium-041-keys-{}", std::process::id()));
    fs::remove_dir_all(&root).ok();
    let a = root.join("mnt").join("x").join("shards");
    touch(&a, "stardata/p00/shard_00001.bin.zst");
    touch(&a, "stardata/p01/shard_00000.bin.zst");
    let b = root.join("other").join("shards");
    touch(&b, "stardata/p00/shard_00001.bin.zst");
    touch(&b, "stardata/p01/shard_00000.bin.zst");
    let want = ["stardata/p00/shard_00001.bin.zst", "stardata/p01/shard_00000.bin.zst"];
    for dir in [a.join("stardata"), a.clone(), b.join("stardata"), b.clone(), b.join("stardata").join("..").join("stardata")] {
        assert_eq!(keys(&dir), want, "{}", dir.display());
    }
    // Another case of the same folder (case-insensitive file systems, e.g. Windows): same key.
    let upper = a.join("STARDATA");
    if upper.is_dir() {
        assert_eq!(keys(&upper), want, "the real folder name, not the spelling");
    }
    fs::remove_dir_all(&root).ok();
}

/// 041-R13: a layout other than `<src>/<part>/shard_*` (shards directly in the given folder,
/// deeper nesting, or both depths in one folder) is refused with the file and the layout named.
#[test]
fn other_shard_layouts_are_refused_041_r13() {
    let root = std::env::temp_dir().join(format!("uranium-041-layout-{}", std::process::id()));
    fs::remove_dir_all(&root).ok();
    let cases: [(&str, &[&str]); 3] = [
        ("flat", &["shard_00000.bin.zst"]),
        ("deep", &["s/p00/deep/shard_00000.bin.zst"]),
        ("mixed", &["p00/shard_00000.bin.zst", "s/p00/shard_00000.bin.zst"]),
    ];
    for (name, files) in cases {
        let d: PathBuf = root.join(name);
        for f in files {
            touch(&d, f);
        }
        let err = format!("{:#}", shard_keys(&d).err().unwrap_or_else(|| panic!("{name} accepted")));
        assert!(err.contains("shard_00000.bin.zst") && err.contains("<source>/<part>/shard_*.bin.zst"), "{name}: {err}");
    }
    fs::remove_dir_all(&root).ok();
}

#[test]
fn shard_files_recurse_in_path_order_041_r9() {
    let root = std::env::temp_dir().join(format!("uranium-041-dirs-{}", std::process::id()));
    fs::remove_dir_all(&root).ok();
    for f in ["p01/shard_00001.bin.zst", "p01/shard_00000.bin.zst", "p00/shard_00000.bin.zst", "p00/deep/shard_00000.bin.zst", "shard_00000.bin.zst"] {
        let p = root.join(f);
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(&p, b"").unwrap();
    }
    // Not shards: manifest, partial copies, other files.
    for f in ["manifest.json", "p00/shard_00001.bin.zst.part", "p01/notes.txt"] {
        fs::write(root.join(f), b"").unwrap();
    }
    let rel: Vec<String> = shard_files(&root)
        .unwrap()
        .iter()
        .map(|p| p.strip_prefix(&root).unwrap().components().map(|c| c.as_os_str().to_string_lossy().into_owned()).collect::<Vec<_>>().join("/"))
        .collect();
    assert_eq!(rel, ["p00/deep/shard_00000.bin.zst", "p00/shard_00000.bin.zst", "p01/shard_00000.bin.zst", "p01/shard_00001.bin.zst", "shard_00000.bin.zst"]);
    fs::remove_dir_all(&root).ok();
}
