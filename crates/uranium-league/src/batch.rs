//! Host collate of one lane group of an RL batch (042): what the learner feeds one forward.
//! Rows are b-major (row = lane * T + step); a segment shorter than T repeats its last step with
//! mask 0 (the row after the trained steps of a non-terminal segment is its bootstrap step).
//! Entities are padded to a multiple of 64 like the inference server (030-R7).

use uranium_data::collate::{HostActionBatch, HostObsBatch, HostPrivBatch, action_batch, obs_batch, priv_batch};
use uranium_data::state::HostState;
use uranium_env::features::label::HostLabel;

use crate::trajectory::Segment;

/// Entity padding of learner batches (the inference server's default, 030-R7).
pub const PAD_ENTITIES: usize = 64;

pub struct SegmentGroup {
    pub b: usize,
    pub t: usize,
    pub obs: HostObsBatch,
    /// The taken actions as labels (teacher forcing along the behaviour's choices, 042).
    pub labels: HostActionBatch,
    pub privileged: HostPrivBatch,
    /// [b*t] 1 on trained steps.
    pub mask: Vec<f32>,
    /// [b*t] behaviour log-prob of the taken action (032-R10 joint).
    pub behaviour: Vec<f32>,
    /// [b*t*4] win, build, units, value rewards (033-R3), 0 past the segment.
    pub rewards: Vec<f32>,
    /// Per lane: the episode ended after its last step (bootstrap 0).
    pub terminal: Vec<bool>,
    /// Per lane: trained steps.
    pub trained: Vec<usize>,
    /// Per lane: oldest behaviour version (042-R4).
    pub versions: Vec<u64>,
    /// Per lane: Core state before step 0 (042 core_state0); empty parts = zero state.
    pub state0: Vec<HostState>,
}

/// Lanes `segs` as one group of `t` steps; selections cut at `max_select` (032-R12).
pub fn group(segs: &[Segment], t: usize, max_select: usize) -> SegmentGroup {
    let b = segs.len();
    let mut obs = Vec::with_capacity(b * t);
    let mut labels = Vec::with_capacity(b * t);
    let mut privs = Vec::with_capacity(b * t);
    let mut mask = Vec::with_capacity(b * t);
    let mut behaviour = Vec::with_capacity(b * t);
    let mut rewards = Vec::with_capacity(b * t * 4);
    for seg in segs {
        let n = seg.steps.len();
        let trained = seg.trained();
        for k in 0..t {
            let s = &seg.steps[k.min(n - 1)];
            obs.push(s.obs.unpack());
            let a = &s.action;
            labels.push(HostLabel { action: a.action, delay_bucket: a.delay_bucket, queued: a.queued, selected: a.selected.clone(), target_unit: a.target_unit, target_loc: a.target_loc });
            privs.push(&s.privileged);
            behaviour.push(s.log_prob);
            mask.push(if k < trained { 1.0 } else { 0.0 });
            rewards.extend_from_slice(if k < n { &s.reward } else { &[0.0; 4] });
        }
    }
    let refs: Vec<_> = obs.iter().collect();
    let obs = obs_batch(&refs, PAD_ENTITIES);
    let label_refs: Vec<&HostLabel> = labels.iter().collect();
    let labels = action_batch(&label_refs, obs.n, max_select);
    let privileged = priv_batch(&privs);
    SegmentGroup {
        b,
        t,
        obs,
        labels,
        privileged,
        mask,
        behaviour,
        rewards,
        terminal: segs.iter().map(|s| s.terminal).collect(),
        trained: segs.iter().map(|s| s.trained()).collect(),
        versions: segs.iter().map(|s| s.version()).collect(),
        state0: segs.iter().map(|s| s.state0.clone()).collect(),
    }
}
