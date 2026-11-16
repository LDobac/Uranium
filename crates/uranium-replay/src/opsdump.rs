use anyhow::{Context, Result};
use serde::Serialize;
use sha2::Digest as _;
use uranium_match::ops::{FieldCheck, OpsMetrics, OpsTracker, RushCombat};

use crate::Replay;
use crate::play::{PlayConfig, play_stride, slot_of, terran_perspectives};

pub const UNTIL_FRAME: u32 = uranium_match::ops::WINDOW_END;
pub const STRIDE: u32 = 8;

pub fn perspective_slots(rep: &Replay, slot: Option<u8>) -> Result<Vec<u8>> {
    match slot {
        Some(s) => {
            let p = rep.header.players.get(s as usize).with_context(|| format!("no player in slot {s}"))?;
            anyhow::ensure!(rep.active_players().iter().any(|(a, _)| a.id == p.id), "slot {s} is not an active player");
            anyhow::ensure!(p.race == crate::header::RACE_TERRAN, "slot {s} is not Terran (052: Terran sessions only)");
            Ok(vec![s])
        }
        None => {
            let ids = terran_perspectives(rep);
            anyhow::ensure!(!ids.is_empty(), "no active Terran player");
            ids.into_iter().map(|id| slot_of(rep, id)).collect()
        }
    }
}

#[derive(Serialize, Debug, Clone, PartialEq)]
pub struct OpsLine {
    pub rep: String,
    pub perspective: Option<u8>,
    pub frames: Option<u32>,
    pub ops: Option<OpsMetrics>,
    pub rush_combat: Option<RushCombat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub field_check: Option<FieldCheck>,
}

pub type Measured = (OpsMetrics, Option<RushCombat>, FieldCheck);

pub fn line(rep: &str, slot: Option<u8>, frames: Option<u32>, res: Result<Measured>, field_check: bool) -> OpsLine {
    let base = OpsLine { rep: rep.to_string(), perspective: slot, frames, ops: None, rush_combat: None, error: None, field_check: None };
    match res {
        Ok((ops, rush, fc)) => OpsLine { ops: Some(ops), rush_combat: rush, field_check: field_check.then_some(fc), ..base },
        Err(e) => OpsLine { error: Some(format!("{e:#}")), ..base },
    }
}

pub fn measure(rep: &Replay, slot: u8, cfg: &PlayConfig, until: u32, stride: u32) -> Result<Measured> {
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
    res.completed()?;
    Ok((t.ops(), t.rush_combat(), t.field_check()))
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
            h[o + 4] = i as u8;
            h[o + 8] = 2;
            h[o + 9] = *race;
            h[o + 11] = b'p';
        }
        let cmds = (0..2u8).map(|p| crate::cmd::Cmd { frame: 3, player: p, op: crate::cmd::op::TRAIN, data: vec![7, 0] }).collect();
        let blocks = vec![crate::cmd::Block { frame: 3, cmds }];
        Replay { format: Format::Legacy, header: header::parse(&h), raw_header: h, blocks, map: vec![], unit_limit: None }
    }

    #[test]
    fn perspective_slots_052_r7() {
        let tz = two_player([header::RACE_TERRAN, 0]);
        assert_eq!(perspective_slots(&tz, None).unwrap(), vec![0]);
        assert_eq!(perspective_slots(&tz, Some(0)).unwrap(), vec![0]);
        assert!(perspective_slots(&tz, Some(1)).is_err(), "the Zerg slot");
        assert!(perspective_slots(&tz, Some(5)).is_err());
        assert!(perspective_slots(&two_player([0, 2]), None).is_err(), "no Terran");
        let tt = two_player([header::RACE_TERRAN, header::RACE_TERRAN]);
        assert_eq!(perspective_slots(&tt, None).unwrap(), vec![0, 1], "TvT: both Terran views");
    }

    #[test]
    fn line_format_052_r7() {
        let ok = serde_json::to_string(&line("g1.rep", Some(0), Some(9_000), Ok((Default::default(), None, Default::default())), false)).unwrap();
        assert!(ok.starts_with(r#"{"rep":"g1.rep","perspective":0,"frames":9000,"ops":{"end_s":0.0,"#), "{ok}");
        assert!(ok.ends_with(r#""float_gas_rs":0.0},"rush_combat":null}"#), "{ok}");
        let bad = serde_json::to_string(&line("g2.rep", Some(1), None, Err(anyhow::anyhow!("launcher exited")), true)).unwrap();
        assert_eq!(bad, r#"{"rep":"g2.rep","perspective":1,"frames":null,"ops":null,"rush_combat":null,"error":"launcher exited"}"#);
        let fc = FieldCheck { depleted: 24, fields: 2, unknown_fields: 0, scvs: 3, scvs_max: 5, carrying: 1, lost_cargo: 8 };
        let ok = serde_json::to_string(&line("g1.rep", Some(0), Some(9_000), Ok((Default::default(), None, fc)), true)).unwrap();
        assert!(ok.ends_with(r#""rush_combat":null,"field_check":{"depleted":24,"fields":2,"unknown_fields":0,"scvs":3,"scvs_max":5,"carrying":1,"lost_cargo":8}}"#), "{ok}");
    }
}
