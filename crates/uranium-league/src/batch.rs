use uranium_data::collate::{HostActionBatch, HostObsBatch, HostPrivBatch, action_batch, obs_batch, priv_batch};
use uranium_data::state::HostState;
use uranium_env::features::label::HostLabel;

use crate::trajectory::Segment;

pub const PAD_ENTITIES: usize = 64;

pub const REF_RACES: [&str; 4] = ["zerg", "terran", "protoss", "random"];

pub fn ref_race(r: uranium_proto::Race) -> u8 {
    use uranium_proto::Race;
    match r {
        Race::Zerg => 0,
        Race::Terran => 1,
        Race::Protoss => 2,
        Race::Random | Race::Unknown => 3,
    }
}

pub struct SegmentGroup {
    pub b: usize,
    pub t: usize,
    pub obs: HostObsBatch,
    pub labels: HostActionBatch,
    pub privileged: HostPrivBatch,
    pub mask: Vec<f32>,
    pub real: Vec<bool>,
    pub priv_present: Vec<bool>,
    pub behaviour: Vec<f32>,
    pub rewards: Vec<f32>,
    pub terminal: Vec<bool>,
    pub trained: Vec<usize>,
    pub versions: Vec<u64>,
    pub r_value_policy: Vec<bool>,
    pub pseudo_scale: Vec<f32>,
    pub sil_return: Vec<f32>,
    pub ref_race: Vec<u8>,
    pub state0: Vec<HostState>,
}

pub fn group(segs: &[Segment], t: usize, max_select: usize) -> SegmentGroup {
    group_of(&segs.iter().collect::<Vec<_>>(), t, max_select)
}

pub fn group_of(segs: &[&Segment], t: usize, max_select: usize) -> SegmentGroup {
    let b = segs.len();
    let mut obs = Vec::with_capacity(b * t);
    let mut labels = Vec::with_capacity(b * t);
    let mut privs = Vec::with_capacity(b * t);
    let mut mask = Vec::with_capacity(b * t);
    let mut real = Vec::with_capacity(b * t);
    let mut priv_present = Vec::with_capacity(b * t);
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
            real.push(k < n);
            priv_present.push(s.has_privileged);
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
        real,
        priv_present,
        behaviour,
        rewards,
        terminal: segs.iter().map(|s| s.terminal).collect(),
        trained: segs.iter().map(|s| s.trained()).collect(),
        versions: segs.iter().map(|s| s.version()).collect(),
        r_value_policy: segs.iter().map(|s| s.r_value_policy).collect(),
        pseudo_scale: segs.iter().flat_map(|s| s.pseudo_scale).collect(),
        sil_return: segs.iter().map(|s| s.sil_return.unwrap_or(f32::NAN)).collect(),
        ref_race: segs.iter().map(|s| ref_race(s.enemy_lobby_race)).collect(),
        state0: segs.iter().map(|s| s.state0.clone()).collect(),
    }
}
