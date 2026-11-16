pub const OPEN_SLOTS: usize = 4;

pub const HIT_PAIRS: [(u8, u8); 4] = [(1, 0), (0, 1), (0, 2), (0, 3)];

struct Lcg(u32);

impl Lcg {
    fn draw(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(22_695_477).wrapping_add(1);
        (self.0 >> 16) & 0x7fff
    }

    fn range(&mut self, from: u32, to: u32) -> u32 {
        from + ((self.draw() * (to - from + 1)) >> 15)
    }
}

pub fn start_slots(seed: u32, random_players: u32) -> [u8; 2] {
    let mut x = Lcg(seed);
    for _ in 0..random_players {
        x.range(0, 2);
    }
    let mut place: [u8; OPEN_SLOTS] = [0, 1, 2, 3];
    for i in (1..OPEN_SLOTS).rev() {
        let j = x.range(0, i as u32) as usize;
        place.swap(i, j);
    }
    let at = |p: u8| place.iter().position(|&q| q == p).unwrap() as u8;
    [at(0), at(1)]
}

pub fn start_hit(slot_self: u8, slot_enemy: u8) -> bool {
    let first = (0..OPEN_SLOTS as u8).find(|&i| i != slot_enemy).unwrap();
    slot_self == first
}

pub fn start_condition(seed: u32, random_players: u32) -> (u8, u8, bool) {
    let [a, b] = start_slots(seed, random_players);
    (a, b, start_hit(a, b))
}

pub fn all_pairs() -> Vec<(u8, u8)> {
    (0..OPEN_SLOTS as u8).flat_map(|a| (0..OPEN_SLOTS as u8).filter(move |&b| b != a).map(move |b| (a, b))).collect()
}

pub const HIT_GAMES_PER_PAIR: u32 = 4;
pub const MISS_GAMES_PER_PAIR: u32 = 2;
pub const CHECK_FIRST_SEED: u32 = 45_001;

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

    const C0_HEADERS: [(u32, u8, u8); 8] = [(43_001, 2, 3), (43_002, 1, 3), (43_003, 1, 3), (43_004, 0, 3), (43_005, 1, 3), (43_006, 1, 0), (43_007, 0, 1), (43_008, 1, 0)];

    #[test]
    fn c0_replay_headers_020_ac8() {
        for (seed, a, b) in C0_HEADERS {
            assert_eq!(start_slots(seed, 0), [a, b], "seed {seed}");
        }
    }

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
        assert!((0..100u32).any(|s| start_slots(s, 1) != start_slots(s, 0)));
    }

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

    #[test]
    fn pairs_are_even_and_hits_one_third_020_r21() {
        let n = 120_000u32;
        let pairs = all_pairs();
        let mut count = vec![0u32; pairs.len()];
        let mut hits = 0;
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
        assert_eq!(check_seeds(2, CHECK_FIRST_SEED), seeds[..64]);
    }
}
