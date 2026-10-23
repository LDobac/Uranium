//! Synthetic BC games for tests (no game data, no map precomputes: use `MapStore::synthetic`):
//! a 32x32 map with a Command Center, an SCV and a mineral field, 16 steps cycling Train_SCV /
//! Move_Pos / Gather_Unit / NoOp labels. The same games as the Rust trainer tests, so the Python
//! trainer tests (via `uranium._core`, AMB-000-4) read the same data. Feature `testkit`.

use std::path::Path;

use anyhow::Result;
use uranium_env::features::extract::action_table;
use uranium_env::types::TypeTables;
use uranium_proto::*;
use uranium_replay::label::Label;
use uranium_replay::shard::{GameMeta, Shard, ShardGame, ShardHeader, Step, write};

pub const W: u16 = 32;

pub fn unit(id: u32, owner: u8, name: &str, tile: (i32, i32)) -> UnitState {
    let t = TypeTables::get().unit_by_name(name).unwrap();
    let pos = if t.is_building() || t.is_resource() {
        ((tile.0 * 32 + t.tile_width * 16) as u16, (tile.1 * 32 + t.tile_height * 16) as u16)
    } else {
        ((tile.0 * 32 + 16) as u16, (tile.1 * 32 + 16) as u16)
    };
    let none = TypeTables::get().unit_by_name("None").unwrap().id as u16;
    UnitState {
        id,
        owner,
        unit_type: t.id as u16,
        pos,
        hp: t.max_hp as u16,
        flags: flags::COMPLETED,
        build_queue: [none; 5],
        resources: if t.is_resource() { 1500 } else { 0 },
        order: 189,
        ..Default::default()
    }
}

pub fn game() -> ShardGame {
    let n = W as usize * W as usize;
    let info = GameInfo {
        map_name: "tiny".into(),
        map_hash: [0; 32],
        map_size_tiles: (W, W),
        self_player: 0,
        players: vec![
            PlayerInfo { id: 0, name: "t".into(), race: Race::Terran, start_tile: Some((4, 4)), is_self: true, is_enemy: false },
            PlayerInfo { id: 1, name: "z".into(), race: Race::Zerg, start_tile: Some((26, 26)), is_self: false, is_enemy: true },
        ],
        static_map: StaticMap {
            width_tiles: W,
            height_tiles: W,
            walkable: BitGrid::from_bytes(W * 4, W * 4, &vec![1; n * 16]),
            buildable: BitGrid::from_bytes(W, W, &vec![1; n]),
            ground_height: vec![0; n],
            start_locations: vec![(4, 4), (26, 26)],
        },
        latency_frames: 0,
    };
    let units = vec![unit(1, 0, "Terran_Command_Center", (4, 4)), unit(2, 0, "Terran_SCV", (9, 5)), unit(3, 11, "Resource_Mineral_Field", (14, 4))];
    let full = |v: bool| BitGrid::from_bytes(W, W, &vec![v as u8; n]);
    let table = action_table();
    let id = |name: &str| table.by_name(name).unwrap().id;
    let label = |frame: u32, k: usize| -> Label {
        let (action, selected, target_unit, target_tile) = match k % 4 {
            0 => (id("Train_SCV"), vec![], None, None),
            1 => (id("Move_Pos"), vec![2], None, Some((12, 12))),
            2 => (id("Gather_Unit"), vec![2], Some(3), None),
            _ => (id("NoOp"), vec![], None, None),
        };
        Label { frame, action, delay: 8, queued: false, selected, target_unit, target_tile }
    };
    let steps = (0..16)
        .map(|k| Step {
            raw_frame: RawFrame {
                frame: k as u32 * 8,
                self_state: PlayerState { minerals: 500, gas: 0, supply_used: 2, supply_total: 20, upgrades: vec![0; 63], ..Default::default() },
                units: units.clone(),
                visibility: full(true),
                explored: full(true),
                creep: full(false),
                events: vec![],
                effects: vec![],
            },
            privileged: Some(PrivilegedSummary::default()),
            action_label: label(k as u32 * 8, k),
        })
        .collect();
    let meta = GameMeta {
        map_hash: "none".into(),
        map_family: "other".into(),
        matchup: "TvZ".into(),
        player_race: 1,
        perspective_slot: 0,
        apm: 100,
        result: 1,
        result_source: "bridge".into(),
        enemy_random: false,
        src: "test".into(),
        stage: "S1".into(),
        replay_sha256: "00".into(),
    };
    ShardGame { meta, game_info: info, steps }
}

/// The synthetic game stretched to `steps` steps with the first `invalid` labels void
/// (`select_empty`: the selected unit 99 does not exist).
pub fn game_with_invalid(steps: usize, invalid: usize) -> ShardGame {
    let mut g = game();
    let base = g.steps.clone();
    g.steps = (0..steps).map(|k| base[k % base.len()].clone()).collect();
    let mv = action_table().by_name("Move_Pos").unwrap().id;
    for s in &mut g.steps[..invalid] {
        s.action_label = Label { frame: s.action_label.frame, action: mv, delay: 8, queued: false, selected: vec![99], target_unit: None, target_tile: Some((12, 12)) };
    }
    g
}

/// A game of `len` steps (1..=16) with its own replay hash.
pub fn game_len(len: usize, tag: u32) -> ShardGame {
    let mut g = game();
    g.steps.truncate(len);
    g.meta.replay_sha256 = format!("{tag:064x}");
    g
}

/// Fifteen games of different lengths (1..=16 steps), each with its own replay hash.
pub fn varied_games() -> Vec<ShardGame> {
    [16, 3, 9, 16, 5, 1, 12, 7, 16, 2, 11, 4, 16, 8, 6].iter().enumerate().map(|(i, &l)| game_len(l, i as u32 + 1)).collect()
}

/// A game with two more SCVs on every step (rows of one batch then have different entity counts).
pub fn game_more_units(len: usize, tag: u32) -> ShardGame {
    let mut g = game_len(len, tag);
    for s in &mut g.steps {
        s.raw_frame.units.extend([unit(4, 0, "Terran_SCV", (10, 6)), unit(5, 0, "Terran_SCV", (11, 6))]);
    }
    g
}

/// Shard files `<root>/p<k>/shard_00000.bin.zst` holding `games` in chunks of `per_file`.
pub fn write_shards(root: &Path, games: &[ShardGame], per_file: usize) -> Result<()> {
    for (k, chunk) in games.chunks(per_file.max(1)).enumerate() {
        let d = root.join(format!("p{k}"));
        std::fs::create_dir_all(&d)?;
        write(&d.join("shard_00000.bin.zst"), &Shard { header: ShardHeader::default(), games: chunk.to_vec() })?;
    }
    Ok(())
}
