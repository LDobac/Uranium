//! 042-R16 자기 모방·막은 판 재학습 (AMB-042-17, off by default): segments of selected RL
//! training games (a ladder group's held or won games) wait in a FIFO buffer and replace some
//! lanes of Main's update batches. The learner trains those lanes with the self-imitation loss
//! (`"sil"`, Oh et al. 2018) or as ordinary lanes (`"replay"`). The defaults are the spec's
//! numbers; `rush-s4` turns it on with them and `wins_only` (AMB-042-18).

use std::collections::{BTreeMap, VecDeque};

use serde::{Deserialize, Serialize};
use uranium_env::reward::Outcome;

use crate::trajectory::Segment;

/// How the learner trains the buffer lanes (042-R16).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SilMode {
    /// Only `w_pg · mean[-log π · A⁺] + w_v · mean[0.5 · (A⁺)²]`, `A⁺ = max(R − V_win, 0)`.
    #[default]
    Sil,
    /// The ordinary loss (V-trace PG, UPGO, value, KL, entropy); a segment leaves the buffer after
    /// `max_uses` batches.
    Replay,
}

/// `[sil]` (042-R16, AMB-042-17). `enabled = false` (the default) = the learner and the league as
/// before this feature.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct SilConfig {
    pub enabled: bool,
    pub mode: SilMode,
    /// The ladder group whose games are candidates (`[[ladder]] group`).
    pub select_group: String,
    /// "Held" stand-in: `scv_4m` at least this (and no collapse concede).
    pub select_scv_4m: u32,
    /// 042-R16 고를 판 (AMB-042-18): `true` = won games only (the held stand-in and
    /// `select_scv_4m` are not used); `false` = won or held, as before. Left out of the serialised
    /// config when false, so checkpoints of earlier configs record the same config as before.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub wins_only: bool,
    /// 042-R16 상대·접촉 거르기 (AMB-042-19): only games against these win-table keys (each the
    /// key of a `select_group` `[[ladder]]` entry, not empty); absent = every opponent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub select_keys: Option<Vec<String>>,
    /// 042-R16 상대·접촉 거르기 (AMB-042-19): only games whose `army_s` (050-R7, first enemy army
    /// sighting, game seconds) is known and at most this (>= 0); absent = no filter.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub select_army_before_s: Option<f64>,
    /// 042-R16 상대·접촉 거르기 (AMB-042-19, 사용자 위임 2026-11-05, 5차 자문): only games whose
    /// 020-R21 `start_hit` equals this; a game without start slots is not selected; absent = no
    /// filter (and left out of the serialised config, as before the key).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub select_start_hit: Option<bool>,
    /// Segments whose first decision frame is below `until_s × 24`; 0 = the whole game.
    pub until_s: u32,
    /// Buffer capacity in segments (FIFO).
    pub buffer_segments: usize,
    /// Batch lanes replaced by buffer segments once the buffer holds this many.
    pub lanes: usize,
    pub w_pg: f32,
    pub w_v: f32,
    /// `"replay"`: uses of a segment before it leaves the buffer.
    pub max_uses: u32,
}

impl Default for SilConfig {
    fn default() -> Self {
        Self { enabled: false, mode: SilMode::Sil, select_group: "rush".into(), select_scv_4m: 12, wins_only: false, select_keys: None, select_army_before_s: None, select_start_hit: None, until_s: 480, buffer_segments: 2_048, lanes: 4, w_pg: 0.05, w_v: 0.01, max_uses: 4 }
    }
}

impl SilConfig {
    /// 042-R16 검사, when enabled: `w_pg`, `w_v` >= 0, `1 <= lanes < batch lanes`,
    /// `buffer_segments >= lanes`, `max_uses >= 1`, `select_group` in `ladder_groups` (the mode
    /// and unknown keys are checked by the parser).
    pub fn validate(&self, batch_lanes: usize, ladder_groups: &BTreeMap<String, f64>) -> Result<(), String> {
        if !self.enabled {
            return Ok(());
        }
        if !(self.w_pg >= 0.0 && self.w_pg.is_finite() && self.w_v >= 0.0 && self.w_v.is_finite()) {
            return Err(format!("[sil] w_pg and w_v must be finite values >= 0, got {} and {} (042-R16)", self.w_pg, self.w_v));
        }
        if self.lanes < 1 || self.lanes >= batch_lanes {
            return Err(format!("[sil] lanes must be at least 1 and below the batch lanes ({batch_lanes}), got {} (042-R16)", self.lanes));
        }
        if self.buffer_segments < self.lanes {
            return Err(format!("[sil] buffer_segments ({}) must be at least lanes ({}) (042-R16)", self.buffer_segments, self.lanes));
        }
        if self.max_uses < 1 {
            return Err("[sil] max_uses must be at least 1 (042-R16)".into());
        }
        if !ladder_groups.contains_key(&self.select_group) {
            return Err(format!("[sil] select_group {:?} is not in [curriculum] ladder_groups (042-R16)", self.select_group));
        }
        // AMB-042-19 (메인 임시 결정); the keys themselves are checked against `[[ladder]]` by
        // `RlConfig::validate`.
        if self.select_keys.as_ref().is_some_and(|k| k.is_empty()) {
            return Err("[sil] select_keys is empty (042-R16, AMB-042-19)".into());
        }
        if let Some(s) = self.select_army_before_s
            && !(s >= 0.0 && s.is_finite())
        {
            return Err(format!("[sil] select_army_before_s must be a finite value >= 0, got {s} (042-R16, AMB-042-19)"));
        }
        Ok(())
    }

    /// 042-R16 상대·접촉 거르기 (AMB-042-19), on top of `selects`: the game's opponent win-table
    /// key is in `select_keys` (when set), its `army_s` is known and at most
    /// `select_army_before_s` (when set; never seen = not selected), and its 020-R21 `start_hit`
    /// passes [`Self::passes_start_hit`].
    pub fn passes_filters(&self, key: &str, army_s: Option<f64>, start_hit: Option<bool>) -> bool {
        self.may_select(key, start_hit) && self.select_army_before_s.is_none_or(|t| army_s.is_some_and(|a| a <= t))
    }

    /// 042-R16 상대·접촉 거르기: the part of [`Self::passes_filters`] known when the game starts
    /// (opponent key and start slots), so games that can never be selected keep no segment copies.
    pub fn may_select(&self, key: &str, start_hit: Option<bool>) -> bool {
        self.select_keys.as_ref().is_none_or(|ks| ks.iter().any(|k| k == key)) && self.passes_start_hit(start_hit)
    }

    /// 042-R16 `select_start_hit` (사용자 위임 2026-11-05, 5차 자문): absent = every game; set =
    /// the game's `start_hit` is known (start slots from its seed) and equal to it.
    pub fn passes_start_hit(&self, start_hit: Option<bool>) -> bool {
        self.select_start_hit.is_none_or(|w| start_hit == Some(w))
    }

    /// 042-R16 담을 세그먼트: its first decision is before `until_s` (0 = every segment).
    pub fn keeps_segment(&self, start_frame: u32) -> bool {
        self.until_s == 0 || u64::from(start_frame) < u64::from(self.until_s) * 24
    }

    /// 042-R16 고를 판: a game of `select_group` with a result (Aborted / Timeout have none) that
    /// was won, or held (`scv_4m >= select_scv_4m`, unknown = no) without a collapse concede;
    /// with `wins_only` (AMB-042-18) won only. The outcome is the one after the 042-R17 rush
    /// judgement.
    pub fn selects(&self, ladder_group: Option<&str>, outcome: Option<Outcome>, scv_4m: Option<u32>, conceded: bool) -> bool {
        if ladder_group != Some(self.select_group.as_str()) {
            return false;
        }
        match outcome {
            None => false,
            Some(Outcome::Win) => true,
            Some(_) => !self.wins_only && !conceded && scv_4m.is_some_and(|n| n >= self.select_scv_4m),
        }
    }
}

struct Entry {
    seg: Segment,
    /// The game's r_win (γ = 1: the win channel's Monte-Carlo return).
    ret: f32,
    uses: u32,
}

/// The 042-R16 segment buffer (FIFO, not in checkpoints: a resumed run starts empty).
pub struct SilBuffer {
    entries: VecDeque<Entry>,
    capacity: usize,
    rng: u64,
}

impl SilBuffer {
    pub fn new(capacity: usize, seed: u64) -> Self {
        Self { entries: VecDeque::new(), capacity: capacity.max(1), rng: seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1 }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// A selected game's segments with its r_win; the oldest leave above the capacity.
    pub fn push_game(&mut self, segs: Vec<Segment>, ret: f32) {
        for seg in segs {
            self.entries.push_back(Entry { seg, ret, uses: 0 });
        }
        while self.entries.len() > self.capacity {
            self.entries.pop_front();
        }
    }

    fn unit(&mut self) -> f64 {
        // SplitMix64, as the league's.
        self.rng = self.rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }

    /// `k` distinct segments drawn uniformly (copies with `sil_return` set); empty when the buffer
    /// holds fewer. `"replay"`: a segment used `max_uses` times leaves the buffer.
    pub fn sample(&mut self, k: usize, mode: SilMode, max_uses: u32) -> Vec<Segment> {
        let n = self.entries.len();
        if k == 0 || n < k {
            return Vec::new();
        }
        // Partial Fisher-Yates over the indices.
        let mut idx: Vec<usize> = (0..n).collect();
        for i in 0..k {
            let j = i + ((self.unit() * (n - i) as f64) as usize).min(n - i - 1);
            idx.swap(i, j);
        }
        let mut chosen = idx[..k].to_vec();
        let out = chosen
            .iter()
            .map(|&i| {
                let e = &mut self.entries[i];
                e.uses += 1;
                let mut s = e.seg.clone();
                s.sil_return = Some(e.ret);
                s
            })
            .collect();
        if mode == SilMode::Replay {
            chosen.sort_unstable_by(|a, b| b.cmp(a));
            for i in chosen {
                if self.entries[i].uses >= max_uses {
                    self.entries.remove(i);
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trajectory::tests::seg_at;

    fn groups() -> BTreeMap<String, f64> {
        BTreeMap::from([("rush".to_string(), 0.5), ("macro".to_string(), 0.5)])
    }

    /// 042-R16 검사: off = never checked; on = the listed load errors.
    #[test]
    fn sil_settings_are_checked_042_r16() {
        let off = SilConfig { lanes: 0, w_pg: -1.0, ..SilConfig::default() };
        off.validate(32, &BTreeMap::new()).unwrap();
        let on = SilConfig { enabled: true, ..SilConfig::default() };
        on.validate(32, &groups()).unwrap();
        assert!(on.validate(32, &BTreeMap::new()).is_err(), "select_group not in ladder_groups");
        for (bad, why) in [
            (SilConfig { lanes: 0, ..on.clone() }, "lanes 0"),
            (SilConfig { lanes: 32, ..on.clone() }, "lanes = batch lanes"),
            (SilConfig { buffer_segments: 3, ..on.clone() }, "buffer below lanes"),
            (SilConfig { w_pg: -0.1, ..on.clone() }, "w_pg"),
            (SilConfig { w_v: f32::NAN, ..on.clone() }, "w_v"),
            (SilConfig { max_uses: 0, ..on.clone() }, "max_uses"),
        ] {
            assert!(bad.validate(32, &groups()).is_err(), "{why}");
        }
        SilConfig { lanes: 31, buffer_segments: 31, until_s: 0, ..on }.validate(32, &groups()).unwrap();
    }

    /// 042-R16 고를 판: the group, a result, then a win or held (scv_4m) without a concede.
    #[test]
    fn game_selection_042_r16() {
        let c = SilConfig::default();
        let rush = Some("rush");
        assert!(c.selects(rush, Some(Outcome::Win), None, false));
        assert!(c.selects(rush, Some(Outcome::Win), Some(3), true), "a won game whatever the rest");
        assert!(c.selects(rush, Some(Outcome::Loss), Some(12), false), "held: scv_4m >= 12");
        assert!(c.selects(rush, Some(Outcome::Draw), Some(20), false));
        assert!(!c.selects(rush, Some(Outcome::Loss), Some(11), false));
        assert!(!c.selects(rush, Some(Outcome::Loss), None, false), "unknown scv_4m");
        assert!(!c.selects(rush, Some(Outcome::Loss), Some(15), true), "collapse concede");
        assert!(!c.selects(rush, None, Some(15), false), "Aborted / Timeout");
        assert!(!c.selects(Some("macro"), Some(Outcome::Win), Some(15), false), "other group");
        assert!(!c.selects(None, Some(Outcome::Win), Some(15), false), "not a ladder game");
    }

    /// 042-R16 `wins_only` / 042-AC8 (AMB-042-18): `true` selects won games of the group only — a
    /// lost or drawn game with `scv_4m >= select_scv_4m` is not selected; `false` (the default)
    /// is the selection above.
    #[test]
    fn wins_only_selection_042_r16() {
        assert!(!SilConfig::default().wins_only, "default false");
        let c = SilConfig { wins_only: true, ..SilConfig::default() };
        let rush = Some("rush");
        assert!(c.selects(rush, Some(Outcome::Win), None, false));
        assert!(c.selects(rush, Some(Outcome::Win), Some(3), true));
        for o in [Outcome::Loss, Outcome::Draw] {
            assert!(!c.selects(rush, Some(o), Some(12), false), "{o:?} held: not with wins_only");
            assert!(!c.selects(rush, Some(o), Some(40), false));
            assert!(SilConfig::default().selects(rush, Some(o), Some(12), false), "{o:?} held: default");
        }
        assert!(!c.selects(rush, None, Some(20), false), "no result");
        assert!(!c.selects(Some("warmup"), Some(Outcome::Win), None, false), "other group");
        // Absent key = false; any other value is a load error.
        let parse = |t: &str| toml::from_str::<SilConfig>(t);
        assert!(!parse("enabled = true").unwrap().wins_only);
        assert!(parse("wins_only = true").unwrap().wins_only);
        assert!(parse("wins_only = 1").is_err());
        assert!(parse("win_only = true").is_err(), "unknown key");
        // Serialised (checkpoint config): absent when false, as before the key; read back the same.
        assert!(serde_json::to_value(SilConfig::default()).unwrap().get("wins_only").is_none());
        let on = serde_json::to_value(&c).unwrap();
        assert_eq!(on["wins_only"], true);
        assert_eq!(serde_json::from_value::<SilConfig>(on).unwrap(), c);
    }

    /// 042-R16 상대·접촉 거르기 / 042-AC9 (AMB-042-19): with `select_keys` an opponent outside
    /// the list is not selected; with `select_army_before_s` a game whose `army_s` is above it or
    /// empty is not; without the two keys every game passes (the earlier selection); the checks
    /// (empty list, negative or NaN seconds) and the serialised config without them as before.
    #[test]
    fn opponent_and_contact_filters_042_r16() {
        let off = SilConfig::default();
        assert!(off.passes_filters("ualbertabot9poolhydra-zerg", None, None), "no keys: every game");
        assert!(off.passes_filters("t0-zerg-rush", Some(400.0), None));
        let c = SilConfig { select_keys: Some(vec!["t0-zerg-rush".into(), "ualbertabot12poollings-zerg".into()]), select_army_before_s: Some(225.0), ..SilConfig::default() };
        assert!(c.passes_filters("t0-zerg-rush", Some(190.0), None));
        assert!(c.passes_filters("ualbertabot12poollings-zerg", Some(225.0), None), "225 s itself");
        assert!(!c.passes_filters("ualbertabot12poollings-zerg", Some(225.04), None));
        assert!(!c.passes_filters("t0-zerg-rush", None, None), "army never seen");
        assert!(!c.passes_filters("ualbertabot9poolhydra-zerg", Some(150.0), None), "opponent not listed");
        let keys_only = SilConfig { select_army_before_s: None, ..c.clone() };
        assert!(keys_only.passes_filters("t0-zerg-rush", None, None));
        let time_only = SilConfig { select_keys: None, ..c.clone() };
        assert!(time_only.passes_filters("anything-zerg", Some(10.0), None) && !time_only.passes_filters("anything-zerg", Some(300.0), None));
        let on = SilConfig { enabled: true, ..c.clone() };
        on.validate(32, &groups()).unwrap();
        assert!(SilConfig { select_keys: Some(vec![]), ..on.clone() }.validate(32, &groups()).is_err(), "empty list");
        assert!(SilConfig { select_army_before_s: Some(-1.0), ..on.clone() }.validate(32, &groups()).is_err());
        assert!(SilConfig { select_army_before_s: Some(f64::NAN), ..on.clone() }.validate(32, &groups()).is_err());
        SilConfig { select_army_before_s: Some(0.0), ..on.clone() }.validate(32, &groups()).unwrap();
        let parse = |t: &str| toml::from_str::<SilConfig>(t);
        let p = parse("select_keys = [\"t0-zerg-rush\"]\nselect_army_before_s = 225").unwrap();
        assert_eq!((p.select_keys, p.select_army_before_s), (Some(vec!["t0-zerg-rush".to_string()]), Some(225.0)));
        let v = serde_json::to_value(SilConfig::default()).unwrap();
        assert!(v.get("select_keys").is_none() && v.get("select_army_before_s").is_none(), "absent when unset");
        assert_eq!(serde_json::from_value::<SilConfig>(serde_json::to_value(&c).unwrap()).unwrap(), c);
    }

    /// 042-R16 `select_start_hit` / 042-AC9 (AMB-042-19, 사용자 위임 2026-11-05, 5차 자문):
    /// `true` selects only first-target hits, `false` only misses, a game without start slots
    /// neither; absent = every game (the earlier selection). With the other filters it is one more
    /// condition (`rush-s5`: three keys, 225 s, hits). Parse, and absent from the serialised config
    /// when unset (earlier checkpoints' config unchanged).
    #[test]
    fn start_hit_filter_042_r16() {
        let off = SilConfig::default();
        for h in [Some(true), Some(false), None] {
            assert!(off.passes_start_hit(h) && off.may_select("t0-zerg-rush", h) && off.passes_filters("t0-zerg-rush", None, h), "absent: every game ({h:?})");
        }
        let hit = SilConfig { select_start_hit: Some(true), ..SilConfig::default() };
        assert!(hit.passes_start_hit(Some(true)));
        assert!(!hit.passes_start_hit(Some(false)), "a miss");
        assert!(!hit.passes_start_hit(None), "no start slots");
        let miss = SilConfig { select_start_hit: Some(false), ..SilConfig::default() };
        assert!(miss.passes_start_hit(Some(false)) && !miss.passes_start_hit(Some(true)) && !miss.passes_start_hit(None));
        let rush5 = SilConfig { select_keys: Some(vec!["t0-zerg-rush".into()]), select_army_before_s: Some(225.0), select_start_hit: Some(true), ..SilConfig::default() };
        assert!(rush5.passes_filters("t0-zerg-rush", Some(174.0), Some(true)));
        assert!(!rush5.passes_filters("t0-zerg-rush", Some(174.0), Some(false)), "early contact but a miss");
        assert!(!rush5.passes_filters("t0-zerg-rush", Some(260.0), Some(true)), "a hit whose army came late (the 225 s guard)");
        assert!(!rush5.passes_filters("ualbertabot9poolhydra-zerg", Some(174.0), Some(true)), "opponent not listed");
        assert!(rush5.may_select("t0-zerg-rush", Some(true)) && !rush5.may_select("t0-zerg-rush", Some(false)) && !rush5.may_select("ualbertabot9poolhydra-zerg", Some(true)));
        SilConfig { enabled: true, ..rush5.clone() }.validate(32, &groups()).unwrap();
        let parse = |t: &str| toml::from_str::<SilConfig>(t);
        assert_eq!(parse("select_start_hit = true").unwrap().select_start_hit, Some(true));
        assert_eq!(parse("select_start_hit = false").unwrap().select_start_hit, Some(false));
        assert_eq!(parse("").unwrap().select_start_hit, None);
        assert!(parse("select_start_hit = 1").is_err() && parse("select_start_hits = true").is_err());
        assert!(serde_json::to_value(SilConfig::default()).unwrap().get("select_start_hit").is_none(), "absent when unset");
        let v = serde_json::to_value(&rush5).unwrap();
        assert_eq!(v["select_start_hit"], true);
        assert_eq!(serde_json::from_value::<SilConfig>(v).unwrap(), rush5);
    }

    /// 042-R16 담을 세그먼트: first decision before `until_s`, 0 = all.
    #[test]
    fn segment_window_042_r16() {
        let c = SilConfig::default();
        assert!(c.keeps_segment(0) && c.keeps_segment(480 * 24 - 1));
        assert!(!c.keeps_segment(480 * 24) && !c.keeps_segment(u32::MAX));
        let all = SilConfig { until_s: 0, ..c };
        assert!(all.keeps_segment(u32::MAX));
    }

    /// 042-R16 버퍼: FIFO capacity, k distinct samples carrying the game's R, nothing below k;
    /// `"sil"` keeps sampled segments, `"replay"` drops one after `max_uses` uses.
    #[test]
    fn buffer_fifo_and_sampling_042_r16() {
        let mut b = SilBuffer::new(5, 1);
        assert!(b.sample(1, SilMode::Sil, 4).is_empty());
        b.push_game((0..3).map(|i| seg_at(i, 0)).collect(), 1.0);
        b.push_game((3..7).map(|i| seg_at(i, 0)).collect(), -1.0);
        assert_eq!(b.len(), 5, "capacity");
        assert!(b.sample(6, SilMode::Sil, 4).is_empty(), "fewer than k");
        let s = b.sample(5, SilMode::Sil, 4);
        let mut ids: Vec<u64> = s.iter().map(|x| x.game_id).collect();
        ids.sort();
        assert_eq!(ids, [2, 3, 4, 5, 6], "the two oldest left, k distinct");
        for x in &s {
            assert_eq!(x.sil_return, Some(if x.game_id < 3 { 1.0 } else { -1.0 }));
        }
        assert_eq!(b.len(), 5, "sil mode keeps them");
        // Uniform: every segment is drawn about as often.
        let mut count = BTreeMap::new();
        for _ in 0..4_000 {
            for x in b.sample(2, SilMode::Sil, 4) {
                *count.entry(x.game_id).or_insert(0usize) += 1;
            }
        }
        assert!(count.values().all(|&n| (1_400..1_800).contains(&n)), "{count:?}");
        // Replay: after max_uses uses a segment leaves.
        let mut r = SilBuffer::new(10, 2);
        r.push_game((0..4).map(|i| seg_at(i, 0)).collect(), 1.0);
        for _ in 0..2 {
            assert_eq!(r.sample(4, SilMode::Replay, 2).len(), 4);
        }
        assert!(r.is_empty(), "every segment used twice");
        let mut r = SilBuffer::new(10, 3);
        r.push_game((0..4).map(|i| seg_at(i, 0)).collect(), 1.0);
        assert_eq!(r.sample(2, SilMode::Replay, 1).len(), 2);
        assert_eq!(r.len(), 2);
    }
}
