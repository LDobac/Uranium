pub(super) fn splitmix(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}

pub(super) fn base_seed(id: u64) -> u32 {
    (splitmix(&mut (id ^ 0xA5A5_5A5A)) * u32::MAX as f64) as u32
}

const START_MIX_TRIES: u64 = 64;

pub(super) fn game_seed(id: u64, share: Option<f64>, random_players: u32) -> u32 {
    let Some(share) = share else { return base_seed(id) };
    mixed_seed(id, share, |s| {
        let [a, b] = uranium_runner::start_slots::start_slots(s, random_players);
        uranium_runner::start_slots::start_hit(a, b)
    })
}

fn mixed_seed(id: u64, share: f64, hit: impl Fn(u32) -> bool) -> u32 {
    let s0 = base_seed(id);
    let target = splitmix(&mut (id ^ 0x5EED_0042_0018_0000)) < share;
    let candidate = |k: u64| if k == 0 { s0 } else { (splitmix(&mut (id.wrapping_mul(START_MIX_TRIES) ^ k ^ 0x51A7_1E5E_ED00_0000)) * u32::MAX as f64) as u32 };
    if let Some(s) = (0..START_MIX_TRIES).map(candidate).find(|&s| hit(s) == target) {
        return s;
    }
    uranium_store::elog!("[rl] game {id}: 042-R18 none of {START_MIX_TRIES} seeds gives the target start condition (first-target hit = {target}); keeping the game's seed {s0}");
    s0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn start_mix_seeds_042_r18() {
        use uranium_runner::start_slots::{start_hit, start_slots};
        let hit = |s: u32| {
            let [a, b] = start_slots(s, 0);
            start_hit(a, b)
        };
        for id in 1..200u64 {
            assert_eq!(game_seed(id, None, 0), base_seed(id), "no share: the earlier seed");
            assert_eq!(base_seed(id), (splitmix(&mut (id ^ 0xA5A5_5A5A)) * u32::MAX as f64) as u32);
        }
        for share in [0.0, 0.5, 1.0, 0.7] {
            let n = 4_000u64;
            let hits = (1..=n).filter(|&id| hit(game_seed(id, Some(share), 0))).count() as f64 / n as f64;
            assert!((hits - share).abs() < 0.03, "share {share}: {hits}");
        }
        for id in [1u64, 77, 1_000_000] {
            assert_eq!(game_seed(id, Some(0.5), 0), game_seed(id, Some(0.5), 0), "deterministic");
            assert!(hit(game_seed(id, Some(1.0), 0)));
            assert!(!hit(game_seed(id, Some(0.0), 0)));
            let s0 = base_seed(id);
            assert_eq!(game_seed(id, Some(if hit(s0) { 1.0 } else { 0.0 }), 0), s0);
            assert_eq!(mixed_seed(id, 1.0, |_| false), s0);
            assert_eq!(mixed_seed(id, 0.0, |_| true), s0);
        }
        let s = game_seed(5, Some(1.0), 1);
        let [a, b] = start_slots(s, 1);
        assert!(start_hit(a, b));
    }
}
