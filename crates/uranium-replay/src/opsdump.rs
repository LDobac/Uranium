//! `uranium-replay opsdump` (052-R7): a stored replay played again in OpenBW replay mode (011-R10)
//! from our player's view, stopping every `stride` frames (8), folded by the same
//! [`uranium_env::ops::OpsTracker`] as live sessions. Playback is cut at the later of `until`
//! (10:00) and the end of the rush combat window. `cmddump` is not used: the command stream does
//! not know carrying, idle, supply or resources.
//!
//! The replay has no final observation (010-R12: replay mode sends no `final_frame`), so the last
//! stop is the last observation. The map precompute is the replay build's (`<CHK sha256>.toml`,
//! written when missing, 031 / AMB-031-3).

use anyhow::{Context, Result};
use serde::Serialize;
use sha2::Digest as _;
use uranium_env::ops::{OpsMetrics, OpsTracker, RushCombat};

use crate::Replay;
use crate::play::{PlayConfig, play_stride, slot_of, terran_perspectives};

/// 052-R7 defaults: 10:00 and 8 frames.
pub const UNTIL_FRAME: u32 = uranium_env::ops::WINDOW_END;
pub const STRIDE: u32 = 8;

/// BWAPI slots (the session record's `slot`) to observe: the given one, else every active Terran
/// (botgen and human replays, 052-R7). A slot without an active player is refused.
pub fn perspective_slots(rep: &Replay, slot: Option<u8>) -> Result<Vec<u8>> {
    match slot {
        Some(s) => {
            let p = rep.header.players.get(s as usize).with_context(|| format!("no player in slot {s}"))?;
            anyhow::ensure!(rep.active_players().iter().any(|(a, _)| a.id == p.id), "slot {s} is not an active player");
            Ok(vec![s])
        }
        None => terran_perspectives(rep).into_iter().map(|id| slot_of(rep, id)).collect(),
    }
}

/// One output line (052 Interfaces), keys in this order: `{"rep", "perspective", "frames", "ops",
/// "rush_combat"}`; a replay that could not be read or played gets `ops` / `rush_combat` null and
/// an `error`.
#[derive(Serialize, Debug, Clone, PartialEq)]
pub struct OpsLine {
    pub rep: String,
    /// Our BWAPI slot (`None` only when the replay could not be read and none was given).
    pub perspective: Option<u8>,
    /// The replay header's frame count.
    pub frames: Option<u32>,
    pub ops: Option<OpsMetrics>,
    pub rush_combat: Option<RushCombat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

pub fn line(rep: &str, slot: Option<u8>, frames: Option<u32>, res: Result<(OpsMetrics, Option<RushCombat>)>) -> OpsLine {
    let base = OpsLine { rep: rep.to_string(), perspective: slot, frames, ops: None, rush_combat: None, error: None };
    match res {
        Ok((ops, rush)) => OpsLine { ops: Some(ops), rush_combat: rush, ..base },
        Err(e) => OpsLine { error: Some(format!("{e:#}")), ..base },
    }
}

/// Plays `rep` from BWAPI slot `slot` and returns the tracker's two objects.
pub fn measure(rep: &Replay, slot: u8, cfg: &PlayConfig, until: u32, stride: u32) -> Result<(OpsMetrics, Option<RushCombat>)> {
    let id = rep.header.players.get(slot as usize).with_context(|| format!("no player in slot {slot}"))?.id;
    let chk: String = sha2::Sha256::digest(&rep.map).iter().map(|b| format!("{b:02x}")).collect();
    let tracker: std::cell::RefCell<Option<Result<OpsTracker>>> = std::cell::RefCell::new(None);
    let res = play_stride(
        rep,
        id,
        cfg,
        stride,
        &mut |gi| {
            let t = crate::pipeline::write_map_precompute(&chk, gi, &rep.map).map(|pre| OpsTracker::for_game(gi, Some(&pre)));
            *tracker.borrow_mut() = Some(t.context("map precompute (031)"));
        },
        &mut |f| match tracker.borrow_mut().as_mut() {
            Some(Ok(t)) => {
                t.step(f);
                f.frame < t.play_until(until)
            }
            _ => false,
        },
    )?;
    let t = tracker.into_inner().context("bridge sent no GameInfo")??;
    anyhow::ensure!(res.observations > 0, "no observation");
    Ok((t.ops(), t.rush_combat()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Format, container, header};

    fn two_player(races: [u8; 2]) -> Replay {
        let mut h = vec![0u8; container::HEADER_SIZE];
        h[1..5].copy_from_slice(&100u32.to_le_bytes());
        for (i, race) in races.iter().enumerate() {
            let o = 0xa1 + 36 * i;
            h[o + 4] = i as u8; // player id
            h[o + 8] = 2; // human
            h[o + 9] = *race;
            h[o + 11] = b'p';
        }
        let cmds = (0..2u8).map(|p| crate::cmd::Cmd { frame: 3, player: p, op: crate::cmd::op::TRAIN, data: vec![7, 0] }).collect();
        let blocks = vec![crate::cmd::Block { frame: 3, cmds }];
        Replay { format: Format::Legacy, header: header::parse(&h), raw_header: h, blocks, map: vec![], unit_limit: None }
    }

    /// 052-R7: the given slot, else every active Terran; an empty slot is refused.
    #[test]
    fn perspective_slots_052_r7() {
        let tz = two_player([header::RACE_TERRAN, 0]);
        assert_eq!(perspective_slots(&tz, None).unwrap(), vec![0]);
        assert_eq!(perspective_slots(&tz, Some(1)).unwrap(), vec![1]);
        assert!(perspective_slots(&tz, Some(5)).is_err());
        let tt = two_player([header::RACE_TERRAN, header::RACE_TERRAN]);
        assert_eq!(perspective_slots(&tt, None).unwrap(), vec![0, 1], "TvT: both Terran views");
    }

    /// 052 Interfaces: the output line's keys in order, and a failed replay with null objects and
    /// an error.
    #[test]
    fn line_format_052_r7() {
        let ok = serde_json::to_string(&line("g1.rep", Some(0), Some(9_000), Ok((Default::default(), None)))).unwrap();
        assert!(ok.starts_with(r#"{"rep":"g1.rep","perspective":0,"frames":9000,"ops":{"end_s":0.0,"#), "{ok}");
        assert!(ok.ends_with(r#""float_gas_rs":0.0},"rush_combat":null}"#), "{ok}");
        let bad = serde_json::to_string(&line("g2.rep", Some(1), None, Err(anyhow::anyhow!("launcher exited")))).unwrap();
        assert_eq!(bad, r#"{"rep":"g2.rep","perspective":1,"frames":null,"ops":null,"rush_combat":null,"error":"launcher exited"}"#);
    }
}
