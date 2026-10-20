//! 041-R9 / 040-R13: shard folders are read with their subfolders (`<src>/<part>/shard_*`), in a
//! deterministic path order.

use std::fs;

use std::path::Path;

use uranium_train::data::{shard_files, shard_key};

/// 041-R13: a shard's canonical key is `<src>/<part>/<file>` whatever folder above it was given
/// and wherever that folder is; a flat folder keeps its own name and nothing above it.
#[test]
fn shard_key_is_location_independent_041_r13() {
    let f = Path::new("mnt").join("x").join("shards").join("stardata").join("p00").join("shard_00001.bin.zst");
    for dir in [Path::new("mnt/x/shards/stardata"), Path::new("mnt/x/shards"), Path::new("mnt/x"), Path::new("mnt")] {
        assert_eq!(shard_key(&dir.components().collect::<std::path::PathBuf>(), &f).unwrap(), "stardata/p00/shard_00001.bin.zst", "{}", dir.display());
    }
    let flat = Path::new("a").join("b").join("sample").join("shard_00000.bin.zst");
    assert_eq!(shard_key(&Path::new("a").join("b").join("sample"), &flat).unwrap(), "sample/shard_00000.bin.zst");
    assert!(shard_key(Path::new("elsewhere"), &flat).is_err());
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
