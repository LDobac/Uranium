#[path = "../../uranium-env/tests/common/mod.rs"]
mod common;

use anyhow::Result;
use uranium_proto::{Command, RawFrame};

use crate::infer::{HostState, InferenceHandle};
use crate::model_policy::{ModelPolicy, ModelStats};

pub use common::{SELF, game_info};

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

pub fn play(h: &InferenceHandle, k: usize, steps: usize) -> Result<(Vec<(Vec<Command>, u32)>, ModelStats)> {
    let mut p = ModelPolicy::new(h, &game_info())?;
    let mut acts = Vec::with_capacity(steps);
    for f in frames(k, steps) {
        let a = p.on_frame(&f)?;
        acts.push((a.cmds, a.advance_frames));
    }
    Ok((acts, p.stats()))
}

pub fn play_parallel(h: &InferenceHandle, sessions: usize, steps: usize) -> Result<Vec<(Vec<(Vec<Command>, u32)>, ModelStats)>> {
    let threads: Vec<_> = (0..sessions)
        .map(|k| {
            let h = h.clone();
            std::thread::spawn(move || play(&h, k, steps))
        })
        .collect();
    threads.into_iter().map(|t| t.join().map_err(|_| anyhow::anyhow!("session thread panicked"))?).collect()
}

pub type ActBits = (String, Vec<u32>);

fn act_bits(r: &crate::infer::ActReply) -> ActBits {
    let mut bits = vec![r.log_prob.to_bits()];
    bits.extend(r.head_log_prob.iter().map(|v| v.to_bits()));
    bits.extend(r.value.iter().map(|v| v.to_bits()));
    bits.push(r.win_open.map_or(0, f32::to_bits));
    (format!("{:?}", r.action), bits)
}

pub struct RestorePlay {
    pub saved: HostState,
    pub restored: HostState,
    pub other_before: HostState,
    pub other_after: HostState,
    pub straight: Vec<ActBits>,
    pub resumed: Vec<ActBits>,
}

pub fn restore_play(h: &InferenceHandle, k: usize, steps: usize) -> Result<RestorePlay> {
    use uranium_env::features::{AgentMemory, MapInfo, extract};
    let info = game_info();
    let mut mem = AgentMemory::new(&info);
    let map = MapInfo::new(&info, 0, None)?;
    let obs: Vec<_> = frames(0, steps).iter().map(|f| extract(f, &mut mem, &map, None)).collect();
    let a = h.open()?;
    let mut saved = None;
    let mut straight = Vec::new();
    for (i, o) in obs.iter().enumerate() {
        if i == k {
            saved = Some(a.state()?);
        }
        let r = a.act(o.clone())?;
        if i >= k {
            straight.push(act_bits(&r));
        }
    }
    let saved = saved.ok_or_else(|| anyhow::anyhow!("k = {k} is not below steps = {steps}"))?;
    let b = h.open()?;
    let other_before = a.state()?;
    b.set_state(saved.clone())?;
    let restored = b.state()?;
    let other_after = a.state()?;
    let resumed = obs[k..].iter().map(|o| b.act(o.clone()).map(|r| act_bits(&r))).collect::<Result<_>>()?;
    Ok(RestorePlay { saved, restored, other_before, other_after, straight, resumed })
}

pub fn set_and_read(h: &InferenceHandle, state: HostState) -> Result<(HostState, std::result::Result<(), String>, HostState)> {
    use uranium_env::features::{AgentMemory, MapInfo, extract};
    let info = game_info();
    let mut mem = AgentMemory::new(&info);
    let map = MapInfo::new(&info, 0, None)?;
    let s = h.open()?;
    s.act(extract(&frames(0, 1)[0], &mut mem, &map, None))?;
    let before = s.state()?;
    let res = s.set_state(state).map_err(|e| format!("{e:#}"));
    Ok((before, res, s.state()?))
}

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
