//! Desync tracing (040-R8 analysis): where does an OpenBW replay start to diverge from the
//! recorded game? Follows the perspective player's selections and reports the first unit tag
//! that the player keeps selecting although it does not exist in the simulation.

use std::collections::{BTreeMap, HashMap};

use anyhow::Result;
use serde::Serialize;

use crate::Replay;
use crate::cmd::{self, op};
use crate::play::{PlayConfig, play_with, slot_of};

/// A tag selected again at least this many times after its first invalid selection is a
/// persistent miss (the player clearly still had that unit), not a stale click.
const PERSISTENT_REPEATS: usize = 3;

#[derive(Serialize, Debug, Clone, Default)]
pub struct Sighting {
    pub unit_type: u16,
    pub first_seen: u32,
    pub last_seen: u32,
    pub last_hp: u16,
    pub last_pos: (u16, u16),
}

#[derive(Serialize, Debug, Clone)]
pub struct Divergence {
    /// Frame of the first selection of the missing unit that OpenBW could not resolve.
    pub frame: u32,
    pub tag: u32,
    /// Later selections of the same tag.
    pub repeats: usize,
    /// How OpenBW saw the unit: never (`None`, e.g. its production failed) or until `last_seen`.
    pub seen: Option<Sighting>,
    /// Unit type the player's last command before divergence trained/built, if any.
    pub nearby_commands: Vec<String>,
}

#[derive(Serialize, Debug, Clone)]
pub struct Trace {
    pub perspective: u8,
    pub header_frames: u32,
    pub end_frame: Option<u32>,
    pub select_cmds: u32,
    pub invalid_select_cmds: u32,
    /// Invalid selection share per 1000-frame window: (window start, invalid, total).
    pub by_window: Vec<(u32, u32, u32)>,
    pub first_persistent: Option<Divergence>,
    pub supply_at_divergence: Option<(u16, u16)>,
}

pub fn trace(rep: &Replay, perspective: u8, cfg: &PlayConfig) -> Result<Trace> {
    let me = slot_of(rep, perspective)?;
    let cmds: Vec<cmd::Cmd> = rep.cmds().filter(|c| c.player == perspective).map(|c| cmd::to_legacy(c, rep.unit_limit)).collect::<Result<_>>()?;
    let mut seen: HashMap<u32, Sighting> = HashMap::new();
    // (frame, tag) of every unresolved selection, in order.
    let mut misses: Vec<(u32, u32)> = Vec::new();
    let mut windows: BTreeMap<u32, (u32, u32)> = BTreeMap::new();
    let mut supply_log: Vec<(u32, u16, u16)> = Vec::new();
    let (mut next, mut selects, mut invalid) = (0usize, 0u32, 0u32);
    let mut last_enemy = 0u16;
    let res = play_with(rep, perspective, cfg, &[], &mut |f, p| {
        if let Some(p) = p {
            last_enemy = p.enemy.supply_used;
        }
        supply_log.push((f.frame, f.self_state.supply_used, last_enemy));
        if std::env::var_os("URANIUM_REPLAY_DEBUG").is_some() && f.frame <= 400 {
            let trains: Vec<u32> = cmds.iter().filter(|c| c.op == op::TRAIN && c.frame <= f.frame).map(|c| c.frame).collect();
            eprintln!("[debug] frame {} minerals {} supply {} trains so far {:?}", f.frame, f.self_state.minerals, f.self_state.supply_used, trains);
        }
        if std::env::var_os("URANIUM_REPLAY_DEBUG").is_some() && supply_log.len() <= 2 {
            let own: Vec<(u32, u16)> = f.units.iter().filter(|u| u.owner == me).map(|u| (u.id, u.unit_type)).collect();
            let all: Vec<(u32, u8, u16)> = f.units.iter().filter(|u| u.owner != me).take(6).map(|u| (u.id, u.owner, u.unit_type)).collect();
            let sel: Vec<String> = cmds.iter().filter(|c| matches!(c.op, op::SELECT | op::SELECT_ADD)).take(3).map(|c| format!("f{} {:?}", c.frame, (0..c.data[0] as usize).map(|i| u16::from_le_bytes([c.data[1 + 2 * i], c.data[2 + 2 * i]])).collect::<Vec<_>>())).collect();
            eprintln!("[debug] frame {} own {own:?} others {all:?} first selects {sel:?}", f.frame);
        }
        for u in f.units.iter().filter(|u| u.owner == me) {
            let s = seen.entry(u.id).or_insert_with(|| Sighting { unit_type: u.unit_type, first_seen: f.frame, ..Default::default() });
            s.unit_type = u.unit_type;
            s.last_seen = f.frame;
            s.last_hp = u.hp;
            s.last_pos = u.pos;
        }
        while next < cmds.len() && cmds[next].frame <= f.frame {
            let c = &cmds[next];
            next += 1;
            if !matches!(c.op, op::SELECT | op::SELECT_ADD) {
                continue;
            }
            let mut bad = false;
            for i in 0..c.data[0] as usize {
                let tag = u16::from_le_bytes([c.data[1 + 2 * i], c.data[2 + 2 * i]]) as u32;
                if !f.units.iter().any(|u| u.id == tag) {
                    misses.push((c.frame, tag));
                    bad = true;
                }
            }
            selects += 1;
            invalid += bad as u32;
            let w = windows.entry(c.frame / 1000 * 1000).or_default();
            w.0 += bad as u32;
            w.1 += 1;
        }
    })?;
    let first_persistent = misses.iter().enumerate().find_map(|(i, &(frame, tag))| {
        let repeats = misses[i + 1..].iter().filter(|m| m.1 == tag).count();
        (repeats >= PERSISTENT_REPEATS).then(|| {
            let nearby_commands = cmds
                .iter()
                .filter(|c| c.frame + 240 >= frame && c.frame <= frame && !matches!(c.op, op::SELECT | op::SELECT_ADD | op::SELECT_REMOVE | op::HOTKEY))
                .map(|c| format!("f{} op{:#04x} {:02x?}", c.frame, c.op, c.data))
                .collect();
            Divergence { frame, tag, repeats, seen: seen.get(&tag).cloned(), nearby_commands }
        })
    });
    let supply_at_divergence = first_persistent.as_ref().and_then(|d| supply_log.iter().rev().find(|s| s.0 <= d.frame).map(|s| (s.1, s.2)));
    Ok(Trace {
        perspective,
        header_frames: rep.header.frames,
        end_frame: res.end_frame,
        select_cmds: selects,
        invalid_select_cmds: invalid,
        by_window: windows.into_iter().map(|(k, (b, t))| (k, b, t)).collect(),
        first_persistent,
        supply_at_divergence,
    })
}
