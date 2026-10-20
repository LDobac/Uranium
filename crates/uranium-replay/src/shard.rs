//! BC shard format (040 "Shard 포맷", 040-R9, R12, R14, R15): per game the decision-time RawFrame,
//! PrivilegedSummary and action label, postcard-encoded and zstd-compressed. Features and masks
//! are built at training time (040-R12), so `feature_version` is 0 here.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uranium_proto::{GameInfo, PrivilegedSummary, RawFrame};

use crate::label::Label;

/// 2: games carry the bridge GameInfo (AMB-040-6); version-1 shards must be rebuilt.
/// 3: label selections follow the game's selection rules (040-R3, 2026-10-17); rebuild 2.
/// 4: GameInfo players are the perspective and its opponent only, in id order (011-R10,
/// 2026-10-17); rebuild 3.
/// 5: `result_source` and `enemy_random` in the meta, no spam merging (040-R4), game-start first
/// step (040-R18) (2026-10-19, full rebuild G-D1); older shards are refused.
/// 6: the header carries `labeler_version` (2026-10-19 user decision, action audit F7 / F12).
/// Schema 5 differs only in the header and is still read, as labeler version 1, because the
/// full rebuild G-D1 is part schema 5 and part schema 6 (040 "Shard 포맷").
pub const SCHEMA_VERSION: u32 = 6;
/// Oldest schema [`read`] accepts.
pub const MIN_SCHEMA_VERSION: u32 = 5;
/// Labeler version of schema-5 shards (040 "Shard 포맷").
pub const SCHEMA5_LABELER_VERSION: u32 = 1;
/// Features are not stored (040-R12).
pub const FEATURE_VERSION: u32 = 0;
const ZSTD_LEVEL: i32 = 3;

/// 040-R17: local folder where `build` completes shards before copying them to the shared drive
/// (`.env`, 001-R9). Default: `<temp>/uranium-replay/shard-staging`.
pub const STAGING_DIR_ENV: &str = "URANIUM_SHARD_STAGING_DIR";
/// 040-R17: copies retried after a hash mismatch (default 3).
pub const COPY_RETRIES: u32 = 3;
/// 040-R17: suffix of a shard being copied to the shared drive.
pub const TEMP_SUFFIX: &str = ".tmp";

/// 040-R8 `result_source` values (shard meta, records).
pub mod result_source {
    /// The OpenBW replay reported a winner.
    pub const BRIDGE: &str = "bridge";
    /// No bridge winner: the first player to leave (LEAVE 0x57) lost.
    pub const LEAVE: &str = "leave";
    /// Neither: the winner recorded for an SC:R replay (043-R8).
    pub const RECORDED: &str = "recorded";
    /// Undecided: `result = 0`, value weight 0 (041-R7).
    pub const NONE: &str = "none";
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ShardHeader {
    pub schema_version: u32,
    pub feature_version: u32,
    pub action_version: u32,
    /// `label::LABELER_VERSION` the labels were made with (schema 6); 1 for schema-5 shards.
    pub labeler_version: u32,
}

impl Default for ShardHeader {
    fn default() -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            feature_version: FEATURE_VERSION,
            action_version: uranium_env::actions::ACTION_SCHEMA_VERSION,
            labeler_version: crate::label::LABELER_VERSION,
        }
    }
}

/// Schema-5 header (no labeler version); games are laid out as in schema 6.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
struct ShardHeaderV5 {
    schema_version: u32,
    feature_version: u32,
    action_version: u32,
}

#[derive(Serialize, Deserialize)]
struct ShardV5 {
    header: ShardHeaderV5,
    games: Vec<ShardGame>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct GameMeta {
    /// SHA-256 of the replay's map section (CHK).
    pub map_hash: String,
    /// 040-R15: `fighting_spirit`, `circuit_breaker` or `other`.
    pub map_family: String,
    /// e.g. "TvZ" from the perspective player's side.
    pub matchup: String,
    pub player_race: u8,
    /// BWAPI id of the perspective player (013).
    pub perspective_slot: u8,
    pub apm: u32,
    /// 040-R8: 1 win, -1 loss, 0 undecided (`result_source = none`).
    pub result: i8,
    /// 040-R8: how `result` was decided (`result_source::*`).
    pub result_source: String,
    /// 030-R11: the opponent picked Random in the lobby (header race 6), so its race is unknown
    /// until its first unit is seen.
    pub enemy_random: bool,
    /// 040-R14 source tag.
    pub src: String,
    /// 040-R9: "S1" or "S2".
    pub stage: String,
    /// SHA-256 of the original replay file: game identity for game-level splits (040-R5).
    pub replay_sha256: String,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Step {
    pub raw_frame: RawFrame,
    pub privileged: Option<PrivilegedSummary>,
    pub action_label: Label,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ShardGame {
    pub meta: GameMeta,
    /// Static map and players of the perspective (AMB-040-6): feature extraction needs it (030).
    pub game_info: GameInfo,
    pub steps: Vec<Step>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Shard {
    pub header: ShardHeader,
    pub games: Vec<ShardGame>,
}

pub fn encode(shard: &Shard) -> Result<Vec<u8>> {
    let bytes = postcard::to_stdvec(shard).context("encoding shard")?;
    Ok(zstd::encode_all(bytes.as_slice(), ZSTD_LEVEL)?)
}

/// Writes one shard to a local path (write then rename, so readers never see a partial file).
/// `build` goes through `ShardWriter`, which adds the 040-R17 copy verification.
pub fn write(path: &Path, shard: &Shard) -> Result<u64> {
    let packed = encode(shard)?;
    let tmp = temp_path(path);
    std::fs::write(&tmp, &packed)?;
    std::fs::rename(&tmp, path)?;
    Ok(packed.len() as u64)
}

/// Reads a shard of schema 6, or of schema 5 as labeler version 1 (040 "Shard 포맷": the G-D1
/// rebuild mixes both). Older schemas are refused.
pub fn read(path: &Path) -> Result<Shard> {
    let packed = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let bytes = zstd::decode_all(packed.as_slice())?;
    decode(&bytes)
}

fn decode(bytes: &[u8]) -> Result<Shard> {
    // The schema version is the first field of every header version.
    let (schema, _) = postcard::take_from_bytes::<u32>(bytes).context("decoding shard header")?;
    ensure!((MIN_SCHEMA_VERSION..=SCHEMA_VERSION).contains(&schema), "shard schema {schema} != {SCHEMA_VERSION} (schema {MIN_SCHEMA_VERSION} is also read)");
    if schema == 5 {
        let s: ShardV5 = postcard::from_bytes(bytes).context("decoding schema-5 shard")?;
        let h = s.header;
        let header = ShardHeader { schema_version: h.schema_version, feature_version: h.feature_version, action_version: h.action_version, labeler_version: SCHEMA5_LABELER_VERSION };
        return Ok(Shard { header, games: s.games });
    }
    postcard::from_bytes(bytes).context("decoding shard")
}

/// 040-R17 (R-5): every finished shard (`shard_NNNNN.bin.zst`) below `dir` decompresses and
/// decodes completely. Checked on `jobs` threads; the error lists every shard that fails, so a
/// manifest is never written over a corrupt shard. Returns (shards, games).
pub fn validate_tree(dir: &Path, jobs: usize) -> Result<(usize, usize)> {
    fn walk(d: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
        for e in std::fs::read_dir(d).with_context(|| format!("listing {}", d.display()))? {
            let e = e?;
            let p = e.path();
            if e.file_type()?.is_dir() {
                walk(&p, out)?;
            } else if shard_index(&e.file_name().to_string_lossy()).is_some() {
                out.push(p);
            }
        }
        Ok(())
    }
    let mut files = Vec::new();
    walk(dir, &mut files)?;
    files.sort();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let games = std::sync::atomic::AtomicUsize::new(0);
    let bad = std::sync::Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for _ in 0..jobs.max(1) {
            s.spawn(|| {
                while let Some(f) = files.get(next.fetch_add(1, std::sync::atomic::Ordering::Relaxed)) {
                    match read(f) {
                        Ok(sh) => {
                            games.fetch_add(sh.games.len(), std::sync::atomic::Ordering::Relaxed);
                        }
                        Err(e) => bad.lock().unwrap().push(format!("{}: {e:#}", f.display())),
                    }
                }
            });
        }
    });
    let mut bad = bad.into_inner().unwrap();
    bad.sort();
    ensure!(bad.is_empty(), "040-R17: {} of {} shards do not decode:\n{}", bad.len(), files.len(), bad.join("\n"));
    Ok((files.len(), games.into_inner()))
}

/// `<path>.tmp`: the name a shard has while it is copied (040-R17).
pub fn temp_path(path: &Path) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push(TEMP_SUFFIX);
    PathBuf::from(s)
}

/// Index of a finished shard file name `shard_NNNNN.bin.zst`; temporary copies (`*.tmp`, and
/// `*.part` / `*.bin.tmp` of older builds) are not shards (040-R17).
pub fn shard_index(file_name: &str) -> Option<u32> {
    file_name.strip_prefix("shard_")?.strip_suffix(".bin.zst")?.parse().ok()
}

/// First free shard index of a folder: one past the highest finished shard. Counting files would
/// reuse an index after a gap or count left-over temporary copies.
pub fn next_shard_index(dir: &Path) -> u32 {
    std::fs::read_dir(dir).map_or(0, |d| d.flatten().filter_map(|e| shard_index(&e.file_name().to_string_lossy())).map(|i| i + 1).max().unwrap_or(0))
}

/// 040-R17 staging folder (`URANIUM_SHARD_STAGING_DIR`).
pub fn staging_dir() -> PathBuf {
    std::env::var_os(STAGING_DIR_ENV).filter(|v| !v.is_empty()).map_or_else(|| std::env::temp_dir().join("uranium-replay").join("shard-staging"), PathBuf::from)
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

/// Copies a file to the shared drive (replaceable in tests to simulate a corrupting copy).
pub type CopyFn = fn(&Path, &Path) -> std::io::Result<u64>;

/// 040-R17: `local` is complete; copy it to `<dest>.tmp`, read the copy back and compare
/// SHA-256, rename to `dest` only on a match and then delete `local`. A mismatch removes the
/// temporary copy and copies again, `retries` times; after that the error names the local copy,
/// which is kept. The read-back is what the file server holds: the copy is `fsync`ed and read
/// past this client's cache (`share_io`, R-5: a cached read-back passed 7 shards that were
/// zero-filled on the share).
pub fn publish(local: &Path, dest: &Path, copy: CopyFn, retries: u32) -> Result<()> {
    let want = sha256_hex(&std::fs::read(local).with_context(|| format!("reading {}", local.display()))?);
    let tmp = temp_path(dest);
    for attempt in 0..=retries {
        let got = copy(local, &tmp).and_then(|_| crate::share_io::sync_file(&tmp)).and_then(|_| crate::share_io::read_uncached(&tmp)).map(|b| sha256_hex(&b));
        match got {
            Ok(h) if h == want => {
                std::fs::rename(&tmp, dest).with_context(|| format!("renaming {} (040-R17)", tmp.display()))?;
                std::fs::remove_file(local).ok();
                return Ok(());
            }
            Ok(h) => eprintln!("040-R17: copy {} of {} read back as {} instead of {}", attempt + 1, dest.display(), &h[..12], &want[..12]),
            Err(e) => eprintln!("040-R17: copy {} of {} failed: {e}", attempt + 1, dest.display()),
        }
        std::fs::remove_file(&tmp).ok();
    }
    bail!("040-R17: {} failed verification after {} copies; local copy kept at {}", dest.display(), retries + 1, local.display())
}

/// Collects games and writes `shard_XXXXX.bin.zst` files of roughly `target_bytes` (encoded size
/// before compression) each: completed in the local staging folder, then copied and verified
/// (040-R17).
pub struct ShardWriter {
    dir: PathBuf,
    staging: PathBuf,
    next_index: u32,
    target_bytes: usize,
    pending: Vec<ShardGame>,
    pending_bytes: usize,
    pub written: Vec<(PathBuf, u64)>,
    pub copy: CopyFn,
    pub retries: u32,
    /// Local copy of the shard that failed verification (040-R17); set once, then every write fails.
    failed: Option<PathBuf>,
}

impl ShardWriter {
    /// `staging` is a local folder (`staging_dir()`); several builds may share it, so shard files
    /// there carry the process id.
    pub fn new(dir: PathBuf, staging: PathBuf, first_index: u32, target_bytes: usize) -> Result<Self> {
        std::fs::create_dir_all(&dir)?;
        std::fs::create_dir_all(&staging).with_context(|| format!("creating {} (040-R17 staging)", staging.display()))?;
        Ok(Self { dir, staging, next_index: first_index, target_bytes, pending: Vec::new(), pending_bytes: 0, written: Vec::new(), copy: |from, to| std::fs::copy(from, to), retries: COPY_RETRIES, failed: None })
    }

    pub fn add(&mut self, game: ShardGame) -> Result<()> {
        self.check()?;
        self.pending_bytes += postcard::experimental::serialized_size(&game).unwrap_or(0);
        self.pending.push(game);
        if self.pending_bytes >= self.target_bytes {
            self.flush()?;
        }
        Ok(())
    }

    /// 040-R17: after a failed shard the writer refuses everything, so the kept local copy is
    /// never overwritten and no later shard is published past the failure.
    fn check(&self) -> Result<()> {
        match &self.failed {
            Some(local) => bail!("040-R17: an earlier shard failed verification (local copy {}); no more shards are written", local.display()),
            None => Ok(()),
        }
    }

    pub fn flush(&mut self) -> Result<()> {
        self.check()?;
        if self.pending.is_empty() {
            return Ok(());
        }
        // The index and the pending games are consumed whatever happens; the local name is
        // unique per process and attempt, so no later flush or run can overwrite a kept copy.
        let name = format!("shard_{:05}.bin.zst", self.next_index);
        self.next_index += 1;
        self.pending_bytes = 0;
        let path = self.dir.join(&name);
        let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos());
        let local = self.staging.join(format!("{}-{stamp}-{name}", std::process::id()));
        let shard = Shard { header: ShardHeader::default(), games: std::mem::take(&mut self.pending) };
        let res = write(&local, &shard).and_then(|n| publish(&local, &path, self.copy, self.retries).map(|_| n));
        match res {
            Ok(n) => {
                self.written.push((path, n));
                Ok(())
            }
            Err(e) => {
                self.failed = Some(local);
                Err(e)
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn game(n: u32) -> ShardGame {
        let meta = GameMeta {
            map_hash: "ab".into(),
            map_family: "fighting_spirit".into(),
            matchup: "TvZ".into(),
            player_race: 1,
            perspective_slot: 0,
            apm: 180,
            result: 0,
            result_source: result_source::NONE.into(),
            enemy_random: false,
            src: "personal".into(),
            stage: "S2".into(),
            replay_sha256: format!("{n:064x}"),
        };
        let step = Step {
            raw_frame: RawFrame {
                frame: n,
                self_state: Default::default(),
                units: vec![],
                visibility: uranium_proto::BitGrid::new(4, 4),
                explored: uranium_proto::BitGrid::new(4, 4),
                creep: uranium_proto::BitGrid::new(4, 4),
                events: vec![],
                effects: vec![],
            },
            privileged: None,
            action_label: Label { frame: n, action: 4, delay: 8, queued: false, selected: vec![1, 2], target_unit: Some(9), target_tile: None },
        };
        let game_info = GameInfo {
            map_name: "m".into(),
            map_hash: [0; 32],
            map_size_tiles: (4, 4),
            self_player: 0,
            players: vec![],
            static_map: uranium_proto::StaticMap {
                width_tiles: 4,
                height_tiles: 4,
                walkable: uranium_proto::BitGrid::new(16, 16),
                buildable: uranium_proto::BitGrid::new(4, 4),
                ground_height: vec![0; 16],
                start_locations: vec![],
            },
            latency_frames: 0,
        };
        ShardGame { meta, game_info, steps: vec![step; 3] }
    }

    fn tmp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("uranium-shard-test-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    fn files(dir: &Path) -> Vec<String> {
        let mut v: Vec<String> = std::fs::read_dir(dir).map(|d| d.flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect()).unwrap_or_default();
        v.sort();
        v
    }

    #[test]
    fn validate_tree_names_every_corrupt_shard_040_r17() {
        let dir = tmp_dir("validate");
        let mut w = ShardWriter::new(dir.join("shared").join("p0"), dir.join("local"), 0, 1).unwrap();
        for n in 1..=3 {
            w.add(game(n)).unwrap();
        }
        assert_eq!(validate_tree(&dir.join("shared"), 2).unwrap(), (3, 3));
        // R-5: an 8 MiB zero-filled head as found on the share; a truncated file.
        let p0 = &w.written[0].0;
        let mut b = std::fs::read(p0).unwrap();
        let n = 8.min(b.len());
        b[..n].fill(0);
        std::fs::write(p0, &b).unwrap();
        let p2 = &w.written[2].0;
        let b = std::fs::read(p2).unwrap();
        std::fs::write(p2, &b[..b.len() / 2]).unwrap();
        let e = format!("{:#}", validate_tree(&dir.join("shared"), 2).unwrap_err());
        assert!(e.contains("2 of 3 shards") && e.contains("shard_00000") && e.contains("shard_00002") && !e.contains("shard_00001"), "{e}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn shards_round_trip_and_roll_over_040() {
        let dir = tmp_dir("roll");
        let mut w = ShardWriter::new(dir.join("shared"), dir.join("local"), 0, 1).unwrap(); // every game flushes
        w.add(game(1)).unwrap();
        w.add(game(2)).unwrap();
        w.flush().unwrap();
        assert_eq!(w.written.len(), 2);
        let s = read(&w.written[1].0).unwrap();
        assert_eq!(s.header, ShardHeader::default());
        assert_eq!(s.games, vec![game(2)]);
        // 040-R17: verified copies leave no temporary or local files.
        assert_eq!(files(&dir.join("shared")), vec!["shard_00000.bin.zst", "shard_00001.bin.zst"]);
        assert!(files(&dir.join("local")).is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Schema 6 (2026-10-19): schema-5 shards of the same rebuild are still read, as labeler
    /// version 1, with their games unchanged (040 "Shard 포맷").
    #[test]
    fn schema5_shards_read_as_labeler_v1_040() {
        let dir = tmp_dir("schema5");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("shard_00000.bin.zst");
        let v5 = ShardV5 { header: ShardHeaderV5 { schema_version: 5, feature_version: 0, action_version: 1 }, games: vec![game(1), game(2)] };
        std::fs::write(&p, zstd::encode_all(postcard::to_stdvec(&v5).unwrap().as_slice(), 3).unwrap()).unwrap();
        let s = read(&p).unwrap();
        assert_eq!(s.header, ShardHeader { schema_version: 5, feature_version: 0, action_version: 1, labeler_version: SCHEMA5_LABELER_VERSION });
        assert_eq!(s.games, vec![game(1), game(2)]);
        // New shards carry the current labeler version.
        write(&p, &Shard { header: ShardHeader::default(), games: vec![game(3)] }).unwrap();
        assert_eq!(read(&p).unwrap().header.labeler_version, crate::label::LABELER_VERSION);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Schemas older than 5 are refused (040 "Shard 포맷").
    #[test]
    fn old_schema_is_refused_040() {
        let dir = tmp_dir("schema");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("shard_00000.bin.zst");
        let old = Shard { header: ShardHeader { schema_version: 4, ..Default::default() }, games: vec![] };
        write(&p, &old).unwrap();
        assert!(format!("{:#}", read(&p).unwrap_err()).contains("schema 4"));
        std::fs::remove_dir_all(&dir).ok();
    }

    pub(crate) fn corrupting_copy(from: &Path, to: &Path) -> std::io::Result<u64> {
        // Zeroes the first bytes, like the SMB failure of reports/m4_desync_nondet.md.
        let mut b = std::fs::read(from)?;
        b[..4].fill(0);
        std::fs::write(to, &b)?;
        Ok(b.len() as u64)
    }

    static FAILS_LEFT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

    fn flaky_copy(from: &Path, to: &Path) -> std::io::Result<u64> {
        if FAILS_LEFT.fetch_update(std::sync::atomic::Ordering::SeqCst, std::sync::atomic::Ordering::SeqCst, |n| n.checked_sub(1)).is_ok() { corrupting_copy(from, to) } else { std::fs::copy(from, to) }
    }

    /// 040-R17: a copy that reads back with another hash is retried; after the retries the shard
    /// is an error, nothing is left under the final or temporary name and the local copy stays.
    #[test]
    fn shard_copy_is_verified_and_retried_040_r17() {
        let dir = tmp_dir("verify");
        let mut w = ShardWriter::new(dir.join("shared"), dir.join("local"), 7, usize::MAX).unwrap();
        w.copy = corrupting_copy;
        w.add(game(1)).unwrap();
        let err = format!("{:#}", w.flush().unwrap_err());
        assert!(err.contains("failed verification after 4 copies"), "{err}");
        assert!(files(&dir.join("shared")).is_empty());
        let kept = files(&dir.join("local"));
        assert!(kept.len() == 1 && kept[0].starts_with(&format!("{}-", std::process::id())) && kept[0].ends_with("-shard_00007.bin.zst"), "{kept:?}");
        let kept_bytes = std::fs::read(dir.join("local").join(&kept[0])).unwrap();
        assert_eq!(read(&dir.join("local").join(&kept[0])).unwrap().games, vec![game(1)]);
        // A failed writer refuses further games and flushes, even with a working copy: the kept
        // local copy is never overwritten and nothing is published after the failure.
        w.copy = |from, to| std::fs::copy(from, to);
        assert!(w.add(game(3)).is_err());
        assert!(w.flush().is_err());
        assert_eq!(files(&dir.join("local")), kept);
        assert_eq!(std::fs::read(dir.join("local").join(&kept[0])).unwrap(), kept_bytes);
        assert!(files(&dir.join("shared")).is_empty());
        // Two bad copies, then a good one: published under the final name.
        FAILS_LEFT.store(2, std::sync::atomic::Ordering::SeqCst);
        let mut w = ShardWriter::new(dir.join("shared"), dir.join("local2"), 8, usize::MAX).unwrap();
        w.copy = flaky_copy;
        w.add(game(2)).unwrap();
        w.flush().unwrap();
        assert_eq!(read(&dir.join("shared/shard_00008.bin.zst")).unwrap().games, vec![game(2)]);
        assert!(files(&dir.join("local2")).is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// 040-R17: temporary copies do not count as shards when a build picks its first index.
    #[test]
    fn next_index_skips_temporary_files_040_r17() {
        let dir = tmp_dir("index");
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(next_shard_index(&dir), 0);
        for f in ["shard_00000.bin.zst", "shard_00002.bin.zst", "shard_00003.bin.zst.tmp", "shard_00004.bin.tmp", "shard_00005.bin.zst.part", "manifest.json"] {
            std::fs::write(dir.join(f), b"x").unwrap();
        }
        assert_eq!(next_shard_index(&dir), 3);
        assert_eq!(shard_index("shard_00012.bin.zst"), Some(12));
        assert_eq!(shard_index("shard_00012.bin.zst.tmp"), None);
        std::fs::remove_dir_all(&dir).ok();
    }
}
