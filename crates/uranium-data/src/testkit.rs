use std::path::Path;

use anyhow::Result;
use uranium_env::features::extract::action_table;
pub use uranium_env::testkit::unit;
use uranium_env::testkit::flat_game_info;
use uranium_proto::*;
use uranium_replay::label::Label;
use uranium_env::features::label::{MemoryAction, UNKNOWN_ACTION};
use uranium_replay::shard::{GameMeta, Shard, ShardGame, ShardHeader, Step, write};

pub const W: u16 = 32;

pub fn game() -> ShardGame {
    let n = W as usize * W as usize;
    let info = flat_game_info("tiny", (W, W), ["t", "z"], [(4, 4), (26, 26)]);
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
            memory_action: None,
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

pub fn game_len(len: usize, tag: u32) -> ShardGame {
    let mut g = game();
    g.steps.truncate(len);
    g.meta.replay_sha256 = format!("{tag:064x}");
    g
}

pub fn varied_games() -> Vec<ShardGame> {
    [16, 3, 9, 16, 5, 1, 12, 7, 16, 2, 11, 4, 16, 8, 6].iter().enumerate().map(|(i, &l)| game_len(l, i as u32 + 1)).collect()
}

pub fn race_games() -> Vec<ShardGame> {
    varied_games()
        .into_iter()
        .enumerate()
        .map(|(i, mut g)| {
            let (race, matchup) = [(Race::Terran, "TvT"), (Race::Protoss, "TvP"), (Race::Zerg, "TvZ"), (Race::Protoss, "TvR")][i % 4];
            g.game_info.players[1].race = race;
            g.meta.matchup = matchup.into();
            g.meta.enemy_random = matchup == "TvR";
            g
        })
        .collect()
}

pub fn chain_games() -> Vec<ShardGame> {
    varied_games()
        .into_iter()
        .map(|mut g| {
            g.meta.src = "botgen".into();
            for k in 0..g.steps.len() {
                let f = if k % 4 == 2 { g.steps[k - 1].raw_frame.frame + 1 } else { k as u32 * 8 };
                let s = &mut g.steps[k];
                s.raw_frame.frame = f;
                s.action_label.frame = f;
                s.raw_frame.self_state.minerals = 400 + 10 * k as i32;
            }
            g
        })
        .collect()
}

pub fn distill_games() -> Vec<ShardGame> {
    let table = action_table();
    let id = |name: &str| table.by_name(name).unwrap().id;
    let decision = |k: usize| -> MemoryAction {
        let (action, units) = match k % 4 {
            0 => (id("Train_SCV"), vec![1]),
            1 => (id("Move_Pos"), vec![2]),
            2 => (id("Gather_Unit"), vec![2]),
            _ => (id("NoOp"), vec![]),
        };
        MemoryAction { action, delay_bucket: 7, units }
    };
    varied_games()
        .into_iter()
        .map(|mut g| {
            g.meta.src = "distill".into();
            g.meta.stage = "B".into();
            g.meta.matchup = "TvP".into();
            g.game_info.players[1].race = Race::Protoss;
            for k in 0..g.steps.len() {
                let s = &mut g.steps[k];
                let f = k as u32 * 8 + 1;
                s.raw_frame.frame = f;
                if k >= 3 {
                    s.raw_frame.units.push(unit(4, 0, "Terran_SCV", (10, 6)));
                }
                s.action_label = Label { frame: f, action: UNKNOWN_ACTION, delay: 8, queued: false, selected: vec![], target_unit: None, target_tile: None };
                s.memory_action = Some(if k == 0 { vec![] } else { vec![decision(k - 1)] });
            }
            g
        })
        .collect()
}

pub fn game_more_units(len: usize, tag: u32) -> ShardGame {
    let mut g = game_len(len, tag);
    for s in &mut g.steps {
        s.raw_frame.units.extend([unit(4, 0, "Terran_SCV", (10, 6)), unit(5, 0, "Terran_SCV", (11, 6))]);
    }
    g
}

pub fn write_shards(root: &Path, games: &[ShardGame], per_file: usize) -> Result<()> {
    for (k, chunk) in games.chunks(per_file.max(1)).enumerate() {
        let d = root.join(format!("p{k}"));
        std::fs::create_dir_all(&d)?;
        write(&d.join("shard_00000.bin.zst"), &Shard { header: ShardHeader::default(), games: chunk.to_vec() })?;
    }
    Ok(())
}
