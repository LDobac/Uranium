//! Synthetic model sessions for tests outside this crate (uranium-py: the PyTorch inference tests,
//! AMB-000-4). The frames are the ones the session tests use (`tests/model_policy.rs`).

#[path = "../../uranium-env/tests/common/mod.rs"]
mod common;

use anyhow::Result;
use uranium_proto::{Command, RawFrame};

use crate::infer::{HostState, InferenceHandle};
use crate::model_policy::{ModelPolicy, ModelStats};

pub use common::{SELF, game_info};

/// Frames that differ per session `k` and step `t` (units move, some appear later).
pub fn frames(k: usize, steps: usize) -> Vec<RawFrame> {
    (0..steps)
        .map(|t| {
            let mut b = common::rich_units();
            let keep = b.units.len() - (k * 3 + t) % 7;
            b.units.truncate(keep);
            for u in b.units.iter_mut().filter(|u| u.owner == SELF) {
                u.pos.0 += (t * 8 + k * 4) as u16;
            }
            common::frame(24 * t as u32 + 1, b.units, t % 2 == 0)
        })
        .collect()
}

/// One session playing `frames(k, steps)`: (commands, advance) per step and the session stats.
pub fn play(h: &InferenceHandle, k: usize, steps: usize) -> Result<(Vec<(Vec<Command>, u32)>, ModelStats)> {
    let mut p = ModelPolicy::new(h, &game_info())?;
    let mut acts = Vec::with_capacity(steps);
    for f in frames(k, steps) {
        let a = p.on_frame(&f)?;
        acts.push((a.cmds, a.advance_frames));
    }
    Ok((acts, p.stats()))
}

/// `sessions` sessions at once (thread each), session `k` playing `frames(k, steps)`.
pub fn play_parallel(h: &InferenceHandle, sessions: usize, steps: usize) -> Result<Vec<(Vec<(Vec<Command>, u32)>, ModelStats)>> {
    let threads: Vec<_> = (0..sessions)
        .map(|k| {
            let h = h.clone();
            std::thread::spawn(move || play(&h, k, steps))
        })
        .collect();
    threads.into_iter().map(|t| t.join().map_err(|_| anyhow::anyhow!("session thread panicked"))?).collect()
}

/// 032-R16: a fresh slot's state, the state after one decision, after a second one.
pub fn slot_states(h: &InferenceHandle) -> Result<Vec<HostState>> {
    use uranium_env::features::{AgentMemory, MapInfo, extract};
    let sess = h.open()?;
    let info = game_info();
    let mut mem = AgentMemory::new(&info);
    let map = MapInfo::new(&info, 0, None)?;
    let fr = frames(0, 2);
    let mut out = vec![sess.state()?];
    for f in &fr {
        sess.act(extract(f, &mut mem, &map, None))?;
        out.push(sess.state()?);
    }
    Ok(out)
}
