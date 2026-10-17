//! 041-R9 / 040-R13: shard folders are read with their subfolders (`<src>/<part>/shard_*`), in a
//! deterministic path order.

use std::fs;

use uranium_train::data::shard_files;

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
