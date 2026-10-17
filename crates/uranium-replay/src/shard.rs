//! BC shard format (040 "Shard 포맷", 040-R9, R12, R14, R15): per game the decision-time RawFrame,
//! PrivilegedSummary and action label, postcard-encoded and zstd-compressed. Features and masks
//! are built at training time (040-R12), so `feature_version` is 0 here.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use uranium_proto::{GameInfo, PrivilegedSummary, RawFrame};

use crate::label::Label;

/// 2: games carry the bridge GameInfo (AMB-040-6); version-1 shards must be rebuilt.
/// 3: label selections follow the game's selection rules (040-R3, 2026-10-17); rebuild 2.
/// 4: GameInfo players are the perspective and its opponent only, in id order (011-R10,
/// 2026-10-17); rebuild 3.
pub const SCHEMA_VERSION: u32 = 4;
/// Features are not stored (040-R12).
pub const FEATURE_VERSION: u32 = 0;
const ZSTD_LEVEL: i32 = 3;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ShardHeader {
    pub schema_version: u32,
    pub feature_version: u32,
    pub action_version: u32,
}

impl Default for ShardHeader {
    fn default() -> Self {
        Self { schema_version: SCHEMA_VERSION, feature_version: FEATURE_VERSION, action_version: uranium_env::actions::ACTION_SCHEMA_VERSION }
    }
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
    /// 1 win, -1 loss, 0 unknown / draw.
    pub result: i8,
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

pub fn write(path: &Path, shard: &Shard) -> Result<u64> {
    let bytes = postcard::to_stdvec(shard).context("encoding shard")?;
    let packed = zstd::encode_all(bytes.as_slice(), ZSTD_LEVEL)?;
    // Write then rename so readers never see a partial shard (061-R4 style).
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, &packed)?;
    std::fs::rename(&tmp, path)?;
    Ok(packed.len() as u64)
}

pub fn read(path: &Path) -> Result<Shard> {
    let packed = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let bytes = zstd::decode_all(packed.as_slice())?;
    let shard: Shard = postcard::from_bytes(&bytes).context("decoding shard")?;
    ensure!(shard.header.schema_version == SCHEMA_VERSION, "shard schema {} != {SCHEMA_VERSION}", shard.header.schema_version);
    Ok(shard)
}

/// Collects games and writes `shard_XXXXX.bin.zst` files of roughly `target_bytes` (encoded size
/// before compression) each.
pub struct ShardWriter {
    dir: PathBuf,
    next_index: u32,
    target_bytes: usize,
    pending: Vec<ShardGame>,
    pending_bytes: usize,
    pub written: Vec<(PathBuf, u64)>,
}

impl ShardWriter {
    pub fn new(dir: PathBuf, first_index: u32, target_bytes: usize) -> Result<Self> {
        std::fs::create_dir_all(&dir)?;
        Ok(Self { dir, next_index: first_index, target_bytes, pending: Vec::new(), pending_bytes: 0, written: Vec::new() })
    }

    pub fn add(&mut self, game: ShardGame) -> Result<()> {
        self.pending_bytes += postcard::experimental::serialized_size(&game).unwrap_or(0);
        self.pending.push(game);
        if self.pending_bytes >= self.target_bytes {
            self.flush()?;
        }
        Ok(())
    }

    pub fn flush(&mut self) -> Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let path = self.dir.join(format!("shard_{:05}.bin.zst", self.next_index));
        let shard = Shard { header: ShardHeader::default(), games: std::mem::take(&mut self.pending) };
        let n = write(&path, &shard)?;
        self.written.push((path, n));
        self.next_index += 1;
        self.pending_bytes = 0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn game(n: u32) -> ShardGame {
        let meta = GameMeta {
            map_hash: "ab".into(),
            map_family: "fighting_spirit".into(),
            matchup: "TvZ".into(),
            player_race: 1,
            perspective_slot: 0,
            apm: 180,
            result: 0,
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

    #[test]
    fn shards_round_trip_and_roll_over_040() {
        let dir = std::env::temp_dir().join(format!("uranium-shard-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut w = ShardWriter::new(dir.clone(), 0, 1).unwrap(); // every game flushes
        w.add(game(1)).unwrap();
        w.add(game(2)).unwrap();
        w.flush().unwrap();
        assert_eq!(w.written.len(), 2);
        let s = read(&w.written[1].0).unwrap();
        assert_eq!(s.header, ShardHeader::default());
        assert_eq!(s.games, vec![game(2)]);
        std::fs::remove_dir_all(&dir).ok();
    }
}
