//! 020-R21 시드 → 시작 슬롯 (AMB-042-19): the start slot of each player of an OpenBW melee game
//! with a fixed `GameSpec.seed`, and the 050-R6 seed list of the condition-controlled rush check.
//!
//! The model is OpenBW `third_party/openbw/sync.h` `start_game` and `bwgame.h` `lcg_rand` as they
//! are: with patch 0002 (`URANIUM_SEED` is the random state, no client uid crc) and patch 0004
//! (launcher slot = the player's slot before the shuffle = its `GameSpec.players` index), the
//! state is the seed; one draw is `x ← x · 22695477 + 1`, `(x >> 16) & 0x7fff`;
//! `lcg_rand(src, from, to) = from + ((r · (to − from + 1)) >> 15)`. `start_game` first draws
//! `lcg_rand(144, 0, 2)` once per active Random player (slots 0..11 in order), then shuffles the
//! open slots of a melee map (`[0, 1, 2, 3]` on both 4-player target maps, empty open slots
//! included) by Fisher–Yates: `i = 3, 2, 1`, `j = lcg_rand(33, 0, i)`, swap (the draw is made
//! even when `j = i`). A player's start slot is the place it ends at. If patch 0002, 0004 or
//! `start_game` changes, 020-AC8 has to pass again.

/// Open slots of the two 4-player target maps (Fighting Spirit, Circuit Breaker).
pub const OPEN_SLOTS: usize = 4;

/// The four (slot_self, slot_enemy) pairs where the enemy's first target is our start (020-R21).
pub const HIT_PAIRS: [(u8, u8); 4] = [(1, 0), (0, 1), (0, 2), (0, 3)];

/// OpenBW `lcg_rand` state (`bwgame.h`).
struct Lcg(u32);

impl Lcg {
    fn draw(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(22_695_477).wrapping_add(1);
        (self.0 >> 16) & 0x7fff
    }

    /// `lcg_rand(source, from, to)`.
    fn range(&mut self, from: u32, to: u32) -> u32 {
        from + ((self.draw() * (to - from + 1)) >> 15)
    }
}

/// 020-R21: start slots (0..3) of `GameSpec.players[0]` and `players[1]` for `seed`, after
/// `random_players` Random race draws (players whose lobby race is Random; 0 for our Terran
/// against a fixed-race opponent).
pub fn start_slots(seed: u32, random_players: u32) -> [u8; 2] {
    let mut x = Lcg(seed);
    for _ in 0..random_players {
        x.range(0, 2);
    }
    // place[k] = the pre-shuffle slot of the player at place k (2 and 3 are empty open slots).
    let mut place: [u8; OPEN_SLOTS] = [0, 1, 2, 3];
    for i in (1..OPEN_SLOTS).rev() {
        let j = x.range(0, i as u32) as usize;
        place.swap(i, j);
    }
    let at = |p: u8| place.iter().position(|&q| q == p).unwrap() as u8;
    [at(0), at(1)]
}

/// 020-R21 첫 목표 적중: the enemy's first target is the lowest start slot other than its own
/// (T0 `attack_target` order, `crates/uranium-scripted/src/lib.rs`), so a hit is
/// `slot_self == min{i in 0..3, i != slot_enemy}`.
pub fn start_hit(slot_self: u8, slot_enemy: u8) -> bool {
    let first = (0..OPEN_SLOTS as u8).find(|&i| i != slot_enemy).unwrap();
    slot_self == first
}

/// Slots and the first-target hit of a game: `(slot_self, slot_enemy, start_hit)`.
pub fn start_condition(seed: u32, random_players: u32) -> (u8, u8, bool) {
    let [a, b] = start_slots(seed, random_players);
    (a, b, start_hit(a, b))
}

/// The twelve (slot_self, slot_enemy) pairs in order (self major).
pub fn all_pairs() -> Vec<(u8, u8)> {
    (0..OPEN_SLOTS as u8).flat_map(|a| (0..OPEN_SLOTS as u8).filter(move |&b| b != a).map(move |b| (a, b))).collect()
}

/// 050-R6 games per pair in one (opponent, map) block: 4 per hit pair, 2 per miss pair (16 + 16).
pub const HIT_GAMES_PER_PAIR: u32 = 4;
pub const MISS_GAMES_PER_PAIR: u32 = 2;
/// 050-R6 first candidate seed (메인 임시 결정).
pub const CHECK_FIRST_SEED: u32 = 45_001;

/// 050-R6 시드 목록 (메인 임시 결정): `blocks` blocks of 32 games in block order. Candidate seeds
/// go up by 1 from `first`; a seed whose (`players[0]`, `players[1]`, no Random) slot pair still
/// has quota in the current block (hit pair 4 games, miss pair 2) is that block's next game, and a
/// full block hands over to the next one, which goes on after the last seed used (no seed twice).
pub fn check_seeds(blocks: usize, first: u32) -> Vec<u32> {
    let pairs = all_pairs();
    let quota = |p: (u8, u8)| if HIT_PAIRS.contains(&p) { HIT_GAMES_PER_PAIR } else { MISS_GAMES_PER_PAIR };
    let per_block: u32 = pairs.iter().map(|&p| quota(p)).sum();
    let mut out = Vec::with_capacity(blocks * per_block as usize);
    let mut seed = first;
    for _ in 0..blocks {
        let mut left: Vec<u32> = pairs.iter().map(|&p| quota(p)).collect();
        let mut n = 0;
        while n < per_block {
            let [a, b] = start_slots(seed, 0);
            let k = pairs.iter().position(|&p| p == (a, b)).unwrap();
            if left[k] > 0 {
                left[k] -= 1;
                n += 1;
                out.push(seed);
            }
            seed = seed.checked_add(1).expect("050-R6 candidate seeds below 2^32");
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 020-AC8: `rush-s4` c0 check (seed 43001..43008, `<shared>/eval/rush4-20261104/c0`,
    /// stage `rush-s4-c0-t0`) — the replay headers' slots of `p0-main` (ours) and
    /// `p1-T0-zerg-zerg_rush` (read 2026-11-05 from `g1.rep`..`g8.rep`; the replays are game data
    /// and not committed).
    const C0_HEADERS: [(u32, u8, u8); 8] = [(43_001, 2, 3), (43_002, 1, 3), (43_003, 1, 3), (43_004, 0, 3), (43_005, 1, 3), (43_006, 1, 0), (43_007, 0, 1), (43_008, 1, 0)];

    #[test]
    fn c0_replay_headers_020_ac8() {
        for (seed, a, b) in C0_HEADERS {
            assert_eq!(start_slots(seed, 0), [a, b], "seed {seed}");
        }
    }

    /// 020-AC8: one draw per Random player comes before the shuffle — the slots of a game with k
    /// Random players are those of the state after k draws.
    #[test]
    fn random_players_draw_first_020_ac8() {
        for seed in [1u32, 43_001, 45_001, 4_000_000_000] {
            let after = |k: u32| {
                let mut x = Lcg(seed);
                for _ in 0..k {
                    x.draw();
                }
                start_slots(x.0, 0)
            };
            assert_eq!(start_slots(seed, 1), after(1));
            assert_eq!(start_slots(seed, 2), after(2));
        }
        // Not the same as no draw for every seed.
        assert!((0..100u32).any(|s| start_slots(s, 1) != start_slots(s, 0)));
    }

    /// 020-AC8 / 020-R21 첫 목표 적중: exactly the four hit pairs of the twelve.
    #[test]
    fn hit_pairs_020_r21() {
        let pairs = all_pairs();
        assert_eq!(pairs.len(), 12);
        let hits: Vec<(u8, u8)> = pairs.iter().copied().filter(|&(a, b)| start_hit(a, b)).collect();
        assert_eq!(hits, vec![(0, 1), (0, 2), (0, 3), (1, 0)]);
        assert!(HIT_PAIRS.iter().all(|p| hits.contains(p)));
        assert_eq!(start_condition(43_006, 0), (1, 0, true));
        assert_eq!(start_condition(43_001, 0), (2, 3, false));
    }

    /// 020-AC8 / 020-R21 분포: on a fixed sample of seeds the twelve pairs come out about evenly
    /// and the hit share is about 1/3.
    #[test]
    fn pairs_are_even_and_hits_one_third_020_r21() {
        let n = 120_000u32;
        let pairs = all_pairs();
        let mut count = vec![0u32; pairs.len()];
        let mut hits = 0;
        // A spread of seeds (consecutive seeds are strongly correlated, 020-R20).
        for i in 0..n {
            let seed = i.wrapping_mul(2_654_435_761).wrapping_add(12_345);
            let [a, b] = start_slots(seed, 0);
            count[pairs.iter().position(|&p| p == (a, b)).unwrap()] += 1;
            hits += start_hit(a, b) as u32;
        }
        let even = n as f64 / 12.0;
        for (p, c) in pairs.iter().zip(&count) {
            assert!((*c as f64 - even).abs() < 0.05 * even, "{p:?}: {c} of {n}");
        }
        let share = hits as f64 / n as f64;
        assert!((share - 1.0 / 3.0).abs() < 0.01, "{share}");
    }

    /// 050-AC3 점검 plan 생성: 128 games in 4 blocks of 16 hits + 16 misses, 4 games per hit pair
    /// and 2 per miss pair (all twelve pairs), the same list for the same input, no seed twice,
    /// every seed's condition from the generator, blocks in candidate order from 45,001.
    #[test]
    fn check_seed_list_050_ac3() {
        let seeds = check_seeds(4, CHECK_FIRST_SEED);
        assert_eq!(seeds.len(), 128);
        assert_eq!(seeds, check_seeds(4, CHECK_FIRST_SEED), "deterministic");
        assert_eq!(seeds[0], CHECK_FIRST_SEED, "45,001 is the first candidate and every pair has quota then");
        assert!(seeds.windows(2).all(|w| w[0] < w[1]), "increasing: no seed twice, blocks continue after the last used seed");
        let pairs = all_pairs();
        for block in seeds.chunks(32) {
            let conds: Vec<(u8, u8, bool)> = block.iter().map(|&s| start_condition(s, 0)).collect();
            assert_eq!(conds.iter().filter(|c| c.2).count(), 16);
            for &p in &pairs {
                let n = conds.iter().filter(|c| (c.0, c.1) == p).count();
                assert_eq!(n, if HIT_PAIRS.contains(&p) { 4 } else { 2 }, "{p:?}");
            }
        }
        // A prefix of the selection is the selection of fewer blocks.
        assert_eq!(check_seeds(2, CHECK_FIRST_SEED), seeds[..64]);
    }
}
