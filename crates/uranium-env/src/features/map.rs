//! `MapInfo`: per-game static map data (030 §3 static channels, 034 location masks). Built from
//! the 010 `GameInfo` plus the map precompute (`tools/mapgen`, 031 맵 사전계산, AMB-031-3).

use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use uranium_proto::{GameInfo, Tile};

use super::schema::{CELLS, MAP, ch};

/// `assets/maps/<map_hash>.toml` written by `tools/mapgen` (031).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct MapPrecompute {
    /// Resource-depot top-left tiles of every base.
    pub base_locs: Vec<Tile>,
    /// Tiles on choke points.
    pub chokes: Vec<Tile>,
    /// Wall-in hint tiles (observation only, never enforced, 031).
    pub wall_candidates: Vec<Tile>,
}

#[derive(Debug, Clone)]
pub struct MapInfo {
    pub width: usize,
    pub height: usize,
    /// Channels 0..=9 of 030 §3, [10,128,128].
    pub static_channels: Vec<f32>,
    /// Tile walkable (any of its 4x4 walk cells).
    pub walkable: Vec<bool>,
    pub buildable: Vec<bool>,
    pub start_locations: Vec<Tile>,
    /// Index of the own start location in `start_locations` (030 `start_loc`).
    pub self_start: Option<usize>,
    /// 040-R15 family index: fighting_spirit 0, circuit_breaker 1, other 2.
    pub map_family: i32,
}

pub const STATIC_CHANNELS: usize = 10;

impl MapInfo {
    /// 030-R6: maps wider or taller than 128 tiles are rejected.
    pub fn new(info: &GameInfo, map_family: i32, pre: Option<&MapPrecompute>) -> Result<Self> {
        let s = &info.static_map;
        let (w, h) = (s.width_tiles as usize, s.height_tiles as usize);
        ensure!(w <= MAP && h <= MAP, "map {w}x{h} exceeds 128x128 tiles (030-R6)");
        let mut walkable = vec![false; CELLS];
        let mut buildable = vec![false; CELLS];
        let mut chans = vec![0f32; STATIC_CHANNELS * CELLS];
        let mut set = |c: usize, x: usize, y: usize| {
            if x < MAP && y < MAP {
                chans[c * CELLS + y * MAP + x] = 1.0;
            }
        };
        for y in 0..h {
            for x in 0..w {
                let i = y * MAP + x;
                let height = s.ground_height.get(y * w + x).copied().unwrap_or(0);
                set(
                    match height {
                        0 | 1 => ch::HEIGHT_LOW,
                        2 | 3 => ch::HEIGHT_HIGH,
                        _ => ch::HEIGHT_VERY_HIGH,
                    },
                    x,
                    y,
                );
                walkable[i] = (0..4).any(|dy| (0..4).any(|dx| s.walkable.get((x * 4 + dx) as u16, (y * 4 + dy) as u16)));
                buildable[i] = s.buildable.get(x as u16, y as u16);
                if walkable[i] {
                    set(ch::WALKABLE, x, y);
                }
                if buildable[i] {
                    set(ch::BUILDABLE, x, y);
                }
                set(ch::VALID_MAP, x, y);
            }
        }
        // Start locations mark the resource-depot footprint (4x3 from the top-left tile).
        for &(x, y) in &s.start_locations {
            for dy in 0..3 {
                for dx in 0..4 {
                    set(ch::START_LOC, x as usize + dx, y as usize + dy);
                }
            }
        }
        if let Some(p) = pre {
            for &(x, y) in &p.base_locs {
                for dy in 0..3 {
                    for dx in 0..4 {
                        set(ch::BASE_LOC, x as usize + dx, y as usize + dy);
                    }
                }
            }
            for &(x, y) in &p.chokes {
                set(ch::CHOKE, x as usize, y as usize);
            }
            for &(x, y) in &p.wall_candidates {
                set(ch::WALL_CANDIDATE, x as usize, y as usize);
            }
        }
        let self_start = info
            .players
            .iter()
            .find(|p| p.is_self)
            .and_then(|p| p.start_tile)
            .and_then(|t| s.start_locations.iter().position(|&l| l == t));
        Ok(Self { width: w, height: h, static_channels: chans, walkable, buildable, start_locations: s.start_locations.clone(), self_start, map_family })
    }

    pub fn in_map(&self, x: i32, y: i32) -> bool {
        x >= 0 && y >= 0 && (x as usize) < self.width && (y as usize) < self.height
    }
}
