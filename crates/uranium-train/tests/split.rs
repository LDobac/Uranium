//! 041-R12 / R13 (040-R5): game-level split from the replay hash, stage selection and the seeded
//! shuffle of `ShardSource`.

use std::collections::BTreeMap;

use sha2::{Digest, Sha256};
use uranium_proto::*;
use uranium_replay::shard::{GameMeta, Shard, ShardGame, ShardHeader, write};
use uranium_train::data::{ShardSource, StepSource};
use uranium_train::split::{Selection, Split, selects, split_of};

fn hash(i: usize) -> String {
    Sha256::digest(format!("replay-{i}").as_bytes()).iter().map(|b| format!("{b:02x}")).collect()
}

fn meta(i: usize, family: &str, slot: u8) -> GameMeta {
    GameMeta {
        map_hash: "none".into(),
        map_family: family.into(),
        matchup: "TvT".into(),
        player_race: 1,
        perspective_slot: slot,
        apm: 100,
        result: 1,
        src: "test".into(),
        stage: "S1".into(),
        replay_sha256: hash(i),
    }
}

fn game(m: GameMeta) -> ShardGame {
    let info = GameInfo {
        map_name: "tiny".into(),
        map_hash: [0; 32],
        map_size_tiles: (4, 4),
        self_player: 0,
        players: vec![],
        static_map: StaticMap {
            width_tiles: 4,
            height_tiles: 4,
            walkable: BitGrid::from_bytes(16, 16, &[1; 256]),
            buildable: BitGrid::from_bytes(4, 4, &[1; 16]),
            ground_height: vec![0; 16],
            start_locations: vec![],
        },
        latency_frames: 0,
    };
    ShardGame { meta: m, game_info: info, steps: vec![] }
}

#[test]
fn split_ratios_and_determinism_041_r12() {
    let mut target = BTreeMap::<Split, usize>::new();
    let mut other = BTreeMap::<Split, usize>::new();
    for i in 0..20_000 {
        let t = meta(i, "fighting_spirit", 0);
        assert_eq!(split_of(&t), split_of(&t.clone()), "pure function of the meta");
        *target.entry(split_of(&t)).or_default() += 1;
        *other.entry(split_of(&meta(i, "other", 0))).or_default() += 1;
    }
    let pct = |m: &BTreeMap<Split, usize>, s| *m.get(&s).unwrap_or(&0) as f64 / 200.0;
    assert!((pct(&target, Split::Train) - 80.0).abs() < 1.5, "{target:?}");
    assert!((pct(&target, Split::Val) - 10.0).abs() < 1.0, "{target:?}");
    assert!((pct(&target, Split::Test) - 10.0).abs() < 1.0, "{target:?}");
    assert!((pct(&other, Split::Train) - 95.0).abs() < 1.0, "{other:?}");
    assert_eq!(other.get(&Split::Test), None, "no test split outside the target maps");
}

#[test]
fn tvt_perspectives_share_a_split_040_r5() {
    for i in 0..2_000 {
        for fam in ["circuit_breaker", "other"] {
            assert_eq!(split_of(&meta(i, fam, 0)), split_of(&meta(i, fam, 1)));
        }
    }
}

#[test]
fn stages_never_train_on_test_and_s2_keeps_target_maps_041_r13() {
    let mut games = Vec::new();
    for i in 0..400 {
        games.push(game(meta(i, if i % 2 == 0 { "fighting_spirit" } else { "other" }, 0)));
    }
    let root = std::env::temp_dir().join(format!("uranium-041-split-{}", std::process::id()));
    std::fs::remove_dir_all(&root).ok();
    for (k, chunk) in games.chunks(100).enumerate() {
        let d = root.join(format!("p{k}"));
        std::fs::create_dir_all(&d).unwrap();
        write(&d.join("shard_00000.bin.zst"), &Shard { header: ShardHeader::default(), games: chunk.to_vec() }).unwrap();
    }
    let read = |sel: Option<Selection>, seed: Option<u64>| {
        let mut s = ShardSource::new(&root).unwrap();
        if let Some(sel) = sel {
            s = s.select(sel);
        }
        if let Some(seed) = seed {
            s = s.shuffled(seed);
        }
        let mut out = Vec::new();
        while let Some(g) = s.next_game().unwrap() {
            out.push(g.meta);
        }
        out
    };
    let all = read(None, None);
    assert_eq!(all.len(), 400);
    let s1 = read(Some(Selection::S1), Some(7));
    let s2 = read(Some(Selection::S2), Some(7));
    assert!(s1.iter().all(|m| split_of(m) == Split::Train));
    assert!(s2.iter().all(|m| split_of(m) == Split::Train && m.map_family == "fighting_spirit"));
    assert_eq!(s1.len(), all.iter().filter(|m| selects(Selection::S1, m)).count());
    assert_eq!(s2.len(), all.iter().filter(|m| selects(Selection::S2, m)).count());
    let test = read(Some(Selection::Only(Split::Test)), None);
    assert!(!test.is_empty());
    assert!(test.iter().all(|t| !s1.contains(t) && !s2.contains(t)), "test games never reach training");
    // Seeded shuffle: same seed, same order; another seed or no shuffle, another order, same set.
    assert_eq!(s1, read(Some(Selection::S1), Some(7)));
    let other = read(Some(Selection::S1), Some(8));
    assert_ne!(s1, other);
    assert_ne!(s1, read(Some(Selection::S1), None));
    let key = |v: &[GameMeta]| {
        let mut k: Vec<_> = v.iter().map(|m| m.replay_sha256.clone()).collect();
        k.sort();
        k
    };
    assert_eq!(key(&s1), key(&other));
    std::fs::remove_dir_all(&root).ok();
}
