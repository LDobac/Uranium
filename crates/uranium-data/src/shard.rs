use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use uranium_env::branch::sha256_hex;
use uranium_proto::{GameInfo, PrivilegedSummary, RawFrame};

use uranium_env::features::label::Label;
pub use uranium_env::features::label::MemoryAction;

pub const SCHEMA_VERSION: u32 = 7;
pub const MIN_SCHEMA_VERSION: u32 = 5;
pub const SCHEMA5_LABELER_VERSION: u32 = 1;
pub const FEATURE_VERSION: u32 = 0;
const ZSTD_LEVEL: i32 = 3;

pub const STAGING_DIR_ENV: &str = "URANIUM_SHARD_STAGING_DIR";
pub const COPY_RETRIES: u32 = 3;
pub const TEMP_SUFFIX: &str = ".tmp";

pub mod result_source {
    pub const BRIDGE: &str = "bridge";
    pub const LEAVE: &str = "leave";
    pub const RECORDED: &str = "recorded";
    pub const NONE: &str = "none";
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ShardHeader {
    pub schema_version: u32,
    pub feature_version: u32,
    pub action_version: u32,
    pub labeler_version: u32,
}

impl Default for ShardHeader {
    fn default() -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            feature_version: FEATURE_VERSION,
            action_version: uranium_env::actions::ACTION_SCHEMA_VERSION,
            labeler_version: uranium_env::features::label::LABELER_VERSION,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
struct ShardHeaderV5 {
    schema_version: u32,
    feature_version: u32,
    action_version: u32,
}

#[derive(Serialize, Deserialize)]
struct ShardV5 {
    header: ShardHeaderV5,
    games: Vec<ShardGameV6>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
struct StepV6 {
    raw_frame: RawFrame,
    privileged: Option<PrivilegedSummary>,
    action_label: Label,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
struct ShardGameV6 {
    meta: GameMeta,
    game_info: GameInfo,
    steps: Vec<StepV6>,
}

#[derive(Serialize, Deserialize)]
struct ShardV6 {
    header: ShardHeader,
    games: Vec<ShardGameV6>,
}

impl From<ShardGameV6> for ShardGame {
    fn from(g: ShardGameV6) -> Self {
        let steps = g.steps.into_iter().map(|s| Step { raw_frame: s.raw_frame, privileged: s.privileged, action_label: s.action_label, memory_action: None }).collect();
        ShardGame { meta: g.meta, game_info: g.game_info, steps }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct GameMeta {
    pub map_hash: String,
    pub map_family: String,
    pub matchup: String,
    pub player_race: u8,
    pub perspective_slot: u8,
    pub apm: u32,
    pub result: i8,
    pub result_source: String,
    pub enemy_random: bool,
    pub src: String,
    pub stage: String,
    pub replay_sha256: String,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Step {
    pub raw_frame: RawFrame,
    pub privileged: Option<PrivilegedSummary>,
    pub action_label: Label,
    pub memory_action: Option<Vec<MemoryAction>>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ShardGame {
    pub meta: GameMeta,
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
    let mut enc = zstd::stream::Encoder::new(Vec::new(), ZSTD_LEVEL)?;
    enc.include_checksum(true)?;
    std::io::Write::write_all(&mut enc, &bytes)?;
    Ok(enc.finish()?)
}

pub fn write(path: &Path, shard: &Shard) -> Result<u64> {
    let packed = encode(shard)?;
    let tmp = temp_path(path);
    std::fs::write(&tmp, &packed)?;
    std::fs::rename(&tmp, path)?;
    Ok(packed.len() as u64)
}

pub fn read(path: &Path) -> Result<Shard> {
    let packed = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let bytes = zstd::decode_all(packed.as_slice())?;
    decode(&bytes)
}

pub fn decode(bytes: &[u8]) -> Result<Shard> {
    let (schema, _) = postcard::take_from_bytes::<u32>(bytes).context("decoding shard header")?;
    ensure!((MIN_SCHEMA_VERSION..=SCHEMA_VERSION).contains(&schema), "shard schema {schema} != {SCHEMA_VERSION} (schemas {MIN_SCHEMA_VERSION}..{SCHEMA_VERSION} are read)");
    match schema {
        5 => {
            let s: ShardV5 = postcard::from_bytes(bytes).context("decoding schema-5 shard")?;
            let h = s.header;
            let header = ShardHeader { schema_version: h.schema_version, feature_version: h.feature_version, action_version: h.action_version, labeler_version: SCHEMA5_LABELER_VERSION };
            Ok(Shard { header, games: s.games.into_iter().map(ShardGame::from).collect() })
        }
        6 => {
            let s: ShardV6 = postcard::from_bytes(bytes).context("decoding schema-6 shard")?;
            Ok(Shard { header: s.header, games: s.games.into_iter().map(ShardGame::from).collect() })
        }
        _ => postcard::from_bytes(bytes).context("decoding shard"),
    }
}

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
        for _ in 0..jobs.clamp(1, 4) {
            s.spawn(|| {
                while let Some(f) = files.get(next.fetch_add(1, std::sync::atomic::Ordering::Relaxed)) {
                    match uranium_store::share_io::read_uncached(f).map_err(anyhow::Error::from).and_then(|p| decode(&zstd::decode_all(p.as_slice())?)) {
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

pub fn temp_path(path: &Path) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push(TEMP_SUFFIX);
    PathBuf::from(s)
}

pub fn shard_index(file_name: &str) -> Option<u32> {
    file_name.strip_prefix("shard_")?.strip_suffix(".bin.zst")?.parse().ok()
}

pub fn next_shard_index(dir: &Path) -> u32 {
    std::fs::read_dir(dir).map_or(0, |d| d.flatten().filter_map(|e| shard_index(&e.file_name().to_string_lossy())).map(|i| i + 1).max().unwrap_or(0))
}

pub fn staging_dir() -> PathBuf {
    std::env::var_os(STAGING_DIR_ENV).filter(|v| !v.is_empty()).map_or_else(|| std::env::temp_dir().join("uranium-replay").join("shard-staging"), PathBuf::from)
}

pub type CopyFn = fn(&Path, &Path) -> std::io::Result<u64>;

pub fn publish(local: &Path, dest: &Path, copy: CopyFn, retries: u32) -> Result<()> {
    let want = sha256_hex(&std::fs::read(local).with_context(|| format!("reading {}", local.display()))?);
    let tmp = temp_path(dest);
    for attempt in 0..=retries {
        let got = copy(local, &tmp).and_then(|_| uranium_store::share_io::sync_file(&tmp)).and_then(|_| uranium_store::share_io::read_uncached(&tmp)).map(|b| sha256_hex(&b));
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
    failed: Option<PathBuf>,
}

impl ShardWriter {
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

#[cfg(feature = "testkit")]
pub mod testkit {
    use super::*;

    pub fn game(n: u32) -> ShardGame {
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
            memory_action: None,
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

    pub fn corrupting_copy(from: &Path, to: &Path) -> std::io::Result<u64> {
        let mut b = std::fs::read(from)?;
        b[..4].fill(0);
        std::fs::write(to, &b)?;
        Ok(b.len() as u64)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::testkit::{corrupting_copy, game};
    use super::*;

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
        for n in 1..=4 {
            w.add(game(n)).unwrap();
        }
        assert_eq!(validate_tree(&dir.join("shared"), 2).unwrap(), (4, 4));
        let p0 = &w.written[0].0;
        let mut b = std::fs::read(p0).unwrap();
        let n = 8.min(b.len());
        b[..n].fill(0);
        std::fs::write(p0, &b).unwrap();
        let p2 = &w.written[2].0;
        let b = std::fs::read(p2).unwrap();
        std::fs::write(p2, &b[..b.len() / 2]).unwrap();
        let p3 = &w.written[3].0;
        let mut b = std::fs::read(p3).unwrap();
        let (lo, hi) = (b.len() / 3, 2 * b.len() / 3);
        b[lo..hi].fill(0);
        std::fs::write(p3, &b).unwrap();
        let e = format!("{:#}", validate_tree(&dir.join("shared"), 2).unwrap_err());
        assert!(e.contains("3 of 4 shards") && e.contains("shard_00000") && e.contains("shard_00002") && e.contains("shard_00003") && !e.contains("shard_00001"), "{e}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn shard_checksum_is_new_and_old_shards_still_read_040_r17() {
        let shard = Shard { header: ShardHeader::default(), games: vec![game(1), game(2)] };
        let packed = encode(&shard).unwrap();
        assert_eq!(packed[4] & 0b100, 0b100, "checksum flag set");
        assert_eq!(decode(&zstd::decode_all(packed.as_slice()).unwrap()).unwrap(), shard);
        let old = zstd::encode_all(postcard::to_stdvec(&shard).unwrap().as_slice(), ZSTD_LEVEL).unwrap();
        assert_eq!(old[4] & 0b100, 0, "older shards have no checksum");
        let dir = tmp_dir("oldsum");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("shard_00000.bin.zst"), &old).unwrap();
        assert_eq!(read(&dir.join("shard_00000.bin.zst")).unwrap(), shard);
        let mut bad = packed.clone();
        let mid = bad.len() / 2;
        bad[mid] ^= 0xff;
        assert!(zstd::decode_all(bad.as_slice()).map_err(anyhow::Error::from).and_then(|b| decode(&b)).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn shards_round_trip_and_roll_over_040() {
        let dir = tmp_dir("roll");
        let mut w = ShardWriter::new(dir.join("shared"), dir.join("local"), 0, 1).unwrap();
        w.add(game(1)).unwrap();
        w.add(game(2)).unwrap();
        w.flush().unwrap();
        assert_eq!(w.written.len(), 2);
        let s = read(&w.written[1].0).unwrap();
        assert_eq!(s.header, ShardHeader::default());
        assert_eq!(s.games, vec![game(2)]);
        assert_eq!(files(&dir.join("shared")), vec!["shard_00000.bin.zst", "shard_00001.bin.zst"]);
        assert!(files(&dir.join("local")).is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn schema5_shards_read_as_labeler_v1_040() {
        let dir = tmp_dir("schema5");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("shard_00000.bin.zst");
        let v5 = ShardV5 { header: ShardHeaderV5 { schema_version: 5, feature_version: 0, action_version: 1 }, games: vec![v6(&game(1)), v6(&game(2))] };
        std::fs::write(&p, zstd::encode_all(postcard::to_stdvec(&v5).unwrap().as_slice(), 3).unwrap()).unwrap();
        let s = read(&p).unwrap();
        assert_eq!(s.header, ShardHeader { schema_version: 5, feature_version: 0, action_version: 1, labeler_version: SCHEMA5_LABELER_VERSION });
        assert_eq!(s.games, vec![game(1), game(2)]);
        write(&p, &Shard { header: ShardHeader::default(), games: vec![game(3)] }).unwrap();
        assert_eq!(read(&p).unwrap().header.labeler_version, uranium_env::features::label::LABELER_VERSION);
        std::fs::remove_dir_all(&dir).ok();
    }

    fn v6(g: &ShardGame) -> ShardGameV6 {
        let steps = g.steps.iter().map(|s| StepV6 { raw_frame: s.raw_frame.clone(), privileged: s.privileged.clone(), action_label: s.action_label.clone() }).collect();
        ShardGameV6 { meta: g.meta.clone(), game_info: g.game_info.clone(), steps }
    }

    #[test]
    fn schema7_memory_action_round_trip_and_schema6_reads_none_040_ac9() {
        let dir = tmp_dir("schema7");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("shard_00000.bin.zst");
        let mut with = game(1);
        with.steps[0].memory_action = Some(vec![]);
        with.steps[1].memory_action = Some(vec![MemoryAction { action: 4, delay_bucket: 2, units: vec![7, 9] }, MemoryAction { action: 0, delay_bucket: 5, units: vec![] }]);
        with.steps[2].memory_action = Some(vec![MemoryAction { action: 12, delay_bucket: 0, units: vec![3] }]);
        let shard = Shard { header: ShardHeader::default(), games: vec![with.clone(), game(2)] };
        assert_eq!(shard.header.schema_version, 7);
        write(&p, &shard).unwrap();
        let back = read(&p).unwrap();
        assert_eq!(back, shard);
        assert_eq!(back.games[0].steps[0].memory_action, Some(vec![]), "an empty list stays an empty list, not None");
        assert!(back.games[1].steps.iter().all(|s| s.memory_action.is_none()));
        let old = ShardV6 { header: ShardHeader { schema_version: 6, ..ShardHeader::default() }, games: vec![v6(&with), v6(&game(2))] };
        std::fs::write(&p, zstd::encode_all(postcard::to_stdvec(&old).unwrap().as_slice(), 3).unwrap()).unwrap();
        let s = read(&p).unwrap();
        assert_eq!(s.header.schema_version, 6);
        let mut want = with.clone();
        want.steps.iter_mut().for_each(|st| st.memory_action = None);
        assert_eq!(s.games, vec![want, game(2)]);
        std::fs::remove_dir_all(&dir).ok();
    }

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

    static FAILS_LEFT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

    fn flaky_copy(from: &Path, to: &Path) -> std::io::Result<u64> {
        if FAILS_LEFT.fetch_update(std::sync::atomic::Ordering::SeqCst, std::sync::atomic::Ordering::SeqCst, |n| n.checked_sub(1)).is_ok() { corrupting_copy(from, to) } else { std::fs::copy(from, to) }
    }

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
        w.copy = |from, to| std::fs::copy(from, to);
        assert!(w.add(game(3)).is_err());
        assert!(w.flush().is_err());
        assert_eq!(files(&dir.join("local")), kept);
        assert_eq!(std::fs::read(dir.join("local").join(&kept[0])).unwrap(), kept_bytes);
        assert!(files(&dir.join("shared")).is_empty());
        FAILS_LEFT.store(2, std::sync::atomic::Ordering::SeqCst);
        let mut w = ShardWriter::new(dir.join("shared"), dir.join("local2"), 8, usize::MAX).unwrap();
        w.copy = flaky_copy;
        w.add(game(2)).unwrap();
        w.flush().unwrap();
        assert_eq!(read(&dir.join("shared/shard_00008.bin.zst")).unwrap().games, vec![game(2)]);
        assert!(files(&dir.join("local2")).is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }

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
