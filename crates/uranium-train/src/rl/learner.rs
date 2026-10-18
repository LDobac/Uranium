//! RL learner (042): one gradient update over a batch of segments. Targets (V-trace, UPGO, GAE)
//! are computed on the host from the detached forward (`returns`); the loss is
//!
//!   policy (V-trace PG + UPGO, or PPO clip) + value (per reward component, 033-R3)
//!   - Σ_h entropy_h (used heads, 032-R10) + kl_coef · KL(π ‖ π_BC) (exact, per head, AMB-042-9)
//!
//! KL re-runs the frozen BC network on the same inputs and labels, so both distributions are the
//! conditionals along the taken action (042, AMB-042-4). Lanes run in groups of `micro_lanes` with
//! gradients summed before one optimizer step, like BC (041).

use std::any::Any;

use burn::module::Module;
use burn::optim::{AdamConfig, GradientsAccumulator, GradientsParams, ModuleOptimizer};
use burn::tensor::{Device, Tensor, TensorData};
use serde::Serialize;
use uranium_model::collate::{HostLabel, action_batch, obs_batch_padded, priv_batch};
use uranium_model::config::UraniumNetConfig;
use uranium_model::heads::{Dists, HEADS, PerHead};
use uranium_model::{CoreInit, HostState, UraniumNet};

use super::config::RlConfig;
use super::returns::{Seq, gae, upgo, vtrace};
use super::trajectory::Segment;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, serde::Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum Algo {
    /// V-trace + UPGO (042 default).
    Vtrace,
    /// PPO clip baseline (042 comparison).
    Ppo,
}

/// Head order of `PerHead::stack` (032-R15).
pub const HEAD_NAMES: [&str; HEADS] = ["action", "delay", "queued", "selected", "target_unit", "target_loc"];

/// One update's numbers for train.csv (042-R2).
#[derive(Debug, Clone, Default, Serialize)]
pub struct UpdateStats {
    pub update: u64,
    pub steps: usize,
    pub loss: f32,
    pub policy: f32,
    pub value: f32,
    pub kl: f32,
    pub kl_coef: f32,
    /// Per head, mean over trained steps where the head is used.
    pub kl_heads: [f32; HEADS],
    pub entropy_heads: [f32; HEADS],
    pub rho_mean: f32,
    pub adv_mean: f32,
    /// Learner version minus the segments' oldest behaviour version, mean over the batch.
    pub lag_mean: f32,
}

pub struct Learner<C: CoreInit> {
    pub net: UraniumNet<C>,
    /// Frozen BC reference for KL (no gradients).
    bc: UraniumNet<C>,
    /// The BC weights as a trainable network (Exploiter reset, 042).
    bc_init: UraniumNet<C>,
    optim: ModuleOptimizer,
    pub cfg: RlConfig,
    pub net_cfg: UraniumNetConfig,
    pub algo: Algo,
    device: Device,
    /// Parameter updates so far (= policy version, 042-R4).
    pub update: u64,
}

/// Adam with the 042 settings (`[optimizer]` of rl.toml).
pub fn new_optimizer(cfg: &RlConfig) -> ModuleOptimizer {
    let o = &cfg.optimizer;
    AdamConfig::new()
        .with_beta_1(o.beta1)
        .with_beta_2(o.beta2)
        .with_epsilon(o.eps)
        // 042 optimizer table / 041-R3 (G-D4): the global-norm clip runs in `update`, not in Burn.
        .init()
}

/// Lane rows of a segment for a batch of T: (step index per row, trained mask per row).
fn lane_rows(seg: &Segment, t: usize) -> (Vec<usize>, Vec<bool>) {
    let n = seg.steps.len();
    let trained = seg.trained();
    ((0..t).map(|k| k.min(n - 1)).collect(), (0..t).map(|k| k < trained).collect())
}

fn host_f32<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    t.into_data().convert::<f32>().try_into_vec::<f32>().expect("f32 tensor")
}

/// Per-row KL(p ‖ q) per head, each [rows], already multiplied by the head's usage (032-R10).
pub fn kl_per_head(p: &Dists, q: &Dists) -> PerHead {
    let rows = p.action.dims()[0];
    let full = |a: &Tensor<2>, b: &Tensor<2>| (a.clone().exp() * (a.clone() - b.clone())).sum_dim(1).reshape([rows]);
    let used = |m: &burn::tensor::Tensor<1, burn::tensor::Bool>| m.clone().float();
    let [b, s, n1] = p.selected.dims();
    let sel = (p.selected.clone().exp() * (p.selected.clone() - q.selected.clone())).sum_dim(2).reshape([b, s]) * p.selected_active.clone().float();
    let _ = n1;
    PerHead {
        action: full(&p.action, &q.action),
        delay: full(&p.delay, &q.delay),
        queued: full(&p.queued, &q.queued) * used(&p.uses_queued),
        selected: sel.sum_dim(1).reshape([rows]) * used(&p.uses_select),
        target_unit: full(&p.target_unit, &q.target_unit) * used(&p.uses_target_unit),
        target_loc: full(&p.target_loc, &q.target_loc) * used(&p.uses_target_loc),
    }
}

/// Per-row loss coefficients of one lane group, from the detached forward.
struct Targets {
    /// V-trace: weight on log π(a) (PG + UPGO). PPO: advantage.
    coef: Vec<f32>,
    /// [rows, 4] value targets.
    vs: Vec<f32>,
    rho_sum: f32,
    adv_sum: f32,
}

impl<C: CoreInit + 'static> Learner<C> {
    /// `net` and `bc` on the autodiff `device`; `bc` is frozen (042 KL reference).
    pub fn new(net: UraniumNet<C>, bc: UraniumNet<C>, net_cfg: UraniumNetConfig, cfg: RlConfig, algo: Algo, device: Device) -> Self {
        Self { net, bc_init: bc.clone(), bc: bc.no_grad(), optim: new_optimizer(&cfg), cfg, net_cfg, algo, device, update: 0 }
    }

    /// Weights for the inference server (032-R16): the current network without autodiff.
    pub fn snapshot(&self) -> Box<dyn Any + Send> {
        Box::new(self.net.valid())
    }

    /// 042 Main Exploiter / League Exploiter reset to the BC policy (KL schedule restarts).
    pub fn reset_to_bc(&mut self) {
        self.net = self.bc_init.clone();
        self.optim = new_optimizer(&self.cfg);
        self.update = 0;
    }

    pub fn optimizer(&self) -> &ModuleOptimizer {
        &self.optim
    }

    pub fn set_optimizer(&mut self, o: ModuleOptimizer) {
        self.optim = o;
    }

    fn targets(&self, segs: &[&Segment], t: usize, logp: &[f32], values: &[f32], weights: [f32; 4]) -> Targets {
        let l = &self.cfg.loss;
        let mut coef = vec![0.0; segs.len() * t];
        let mut vs = vec![0.0; segs.len() * t * 4];
        let (mut rho_sum, mut adv_sum) = (0.0, 0.0);
        for (lane, seg) in segs.iter().enumerate() {
            let tr = seg.trained();
            if tr == 0 {
                continue;
            }
            let row = |k: usize| lane * t + k;
            let log_rho: Vec<f32> = (0..tr).map(|k| logp[row(k)] - seg.steps[k].log_prob).collect();
            let mut pg = vec![0.0f32; tr];
            let mut up = vec![0.0f32; tr];
            for c in 0..4 {
                let rewards: Vec<f32> = (0..tr).map(|k| seg.steps[k].reward[c]).collect();
                let mut v: Vec<f32> = (0..tr).map(|k| values[row(k) * 4 + c]).collect();
                v.push(if seg.terminal { 0.0 } else { values[row(tr) * 4 + c] });
                let s = Seq { rewards: &rewards, values: &v, log_rho: &log_rho };
                match self.algo {
                    Algo::Vtrace => {
                        let vt = vtrace(&s, l.gamma, l.td_lambda, l.vtrace_rho_bar, l.vtrace_c_bar);
                        let u = upgo(&s, l.gamma, l.vtrace_rho_bar);
                        for k in 0..tr {
                            pg[k] += weights[c] * vt.pg_adv[k];
                            up[k] += weights[c] * u[k];
                            vs[row(k) * 4 + c] = vt.vs[k];
                        }
                        if c == 0 {
                            rho_sum += vt.rho.iter().sum::<f32>();
                        }
                    }
                    Algo::Ppo => {
                        let (adv, ret) = gae(&s, l.gamma, self.cfg.ppo.gae_lambda);
                        for k in 0..tr {
                            pg[k] += weights[c] * adv[k];
                            vs[row(k) * 4 + c] = ret[k];
                        }
                        if c == 0 {
                            rho_sum += log_rho.iter().map(|x| x.exp()).sum::<f32>();
                        }
                    }
                }
            }
            for k in 0..tr {
                coef[row(k)] = match self.algo {
                    Algo::Vtrace => l.pg * pg[k] + l.upgo * up[k],
                    Algo::Ppo => pg[k],
                };
                adv_sum += pg[k];
            }
        }
        Targets { coef, vs, rho_sum, adv_sum }
    }

    /// One update over `batch` (lanes x T). `weights` = 033 reward weights for the advantage
    /// (with the league's decay schedules applied). Returns the update's statistics.
    pub fn update(&mut self, batch: &[Segment], weights: [f32; 4]) -> UpdateStats {
        let t = self.cfg.seq_len;
        let steps: usize = batch.iter().map(|s| s.trained()).sum();
        let n = steps.max(1) as f32;
        let g = match self.cfg.micro_lanes {
            0 => batch.len(),
            m => m.min(batch.len()),
        }
        .max(1);
        let kl_coef = self.cfg.kl.coef(self.update);
        let epochs = match self.algo {
            Algo::Vtrace => 1,
            Algo::Ppo => self.cfg.ppo.epochs.max(1),
        };
        let mut stats = UpdateStats { update: self.update + 1, steps, kl_coef, ..Default::default() };
        stats.lag_mean = batch.iter().map(|s| self.update.saturating_sub(s.version()) as f32).sum::<f32>() / batch.len().max(1) as f32;
        // PPO keeps the first epoch's advantages and returns (computed before any change).
        let mut ppo_targets: Vec<Option<Targets>> = (0..batch.len().div_ceil(g)).map(|_| None).collect();
        for epoch in 0..epochs {
            let mut acc = GradientsAccumulator::<UraniumNet<C>>::new();
            for (gi, lanes) in batch.chunks(g).enumerate() {
                let segs: Vec<&Segment> = lanes.iter().collect();
                let (loss, part) = self.group_loss(&segs, t, n, weights, kl_coef, &mut ppo_targets[gi], epoch == 0);
                if epoch == 0 {
                    stats.policy += part.policy;
                    stats.value += part.value;
                    stats.kl += part.kl;
                    stats.rho_mean += part.rho_mean;
                    stats.adv_mean += part.adv_mean;
                    for h in 0..HEADS {
                        stats.kl_heads[h] += part.kl_heads[h];
                        stats.entropy_heads[h] += part.entropy_heads[h];
                    }
                }
                let grads = GradientsParams::from_grads(loss.backward(), &self.net);
                acc.accumulate(&self.net, grads);
            }
            // 042 optimizer table / 041-R3 (G-D4): global L2-norm clip after the lane-group sum.
            let (grads, _norm) = crate::clip::clip_global_norm(&self.net, acc.grads(), self.cfg.optimizer.grad_clip);
            self.net = self.optim.step(self.cfg.optimizer.lr, self.net.clone(), grads);
        }
        stats.loss = stats.policy + stats.value + stats.kl * kl_coef - self.entropy_bonus(&stats.entropy_heads);
        stats.rho_mean /= n;
        stats.adv_mean /= n;
        self.update += 1;
        stats
    }

    fn entropy_bonus(&self, ent: &[f32; HEADS]) -> f32 {
        let e = &self.cfg.entropy;
        let c = [e.action, e.delay, e.queued, e.selected, e.target_unit, e.target_loc];
        (0..HEADS).map(|h| c[h] * ent[h]).sum()
    }

    /// Loss of one lane group (already divided by the batch-wide trained steps `n`).
    #[allow(clippy::too_many_arguments)]
    fn group_loss(&self, segs: &[&Segment], t: usize, n: f32, weights: [f32; 4], kl_coef: f32, ppo: &mut Option<Targets>, first_epoch: bool) -> (Tensor<1>, UpdateStats) {
        let d = &self.device;
        let b = segs.len();
        let mut obs = Vec::with_capacity(b * t);
        let mut labels = Vec::with_capacity(b * t);
        let mut privs = Vec::with_capacity(b * t);
        let mut mask = Vec::with_capacity(b * t);
        let mut behaviour = Vec::with_capacity(b * t);
        for seg in segs {
            let (rows, m) = lane_rows(seg, t);
            for (k, &i) in rows.iter().enumerate() {
                let s = &seg.steps[i];
                obs.push(s.obs.unpack());
                let a = &s.action;
                labels.push(HostLabel { action: a.action, delay_bucket: a.delay_bucket, queued: a.queued, selected: a.selected.clone(), target_unit: a.target_unit, target_loc: a.target_loc });
                privs.push(s.privileged.clone());
                behaviour.push(s.log_prob);
                mask.push(if m[k] { 1.0f32 } else { 0.0 });
            }
        }
        let obs_refs: Vec<_> = obs.iter().collect();
        let ob = obs_batch_padded(&obs_refs, 64, d);
        let ents = ob.entities();
        let label_refs: Vec<&HostLabel> = labels.iter().collect();
        let ab = action_batch(&label_refs, ents, self.net_cfg.heads.max_select, d);
        let priv_refs: Vec<_> = privs.iter().collect();
        let pb = priv_batch(&priv_refs, d);
        // core_state0 per lane (042); a lane without one starts from zero (032-R5).
        let zero = C::state_to_host(&self.net.init_state(1, d));
        let states: Vec<&HostState> = segs.iter().map(|s| if s.state0.parts.is_empty() { &zero } else { &s.state0 }).collect();
        let s0 = C::state_from_host(&HostState::cat(&states), d);

        let out = self.net.forward_seq(&ob, &ab, Some(&pb), s0.clone(), t);
        let logp = out.heads.logp.sum();
        let rows = b * t;
        let maskt = Tensor::<1>::from_data(TensorData::new(mask.clone(), [rows]), d);

        let tg_now;
        let tg: &Targets = match (self.algo, ppo.as_ref()) {
            (Algo::Ppo, Some(x)) => x,
            _ => {
                let lp = host_f32(logp.clone().detach());
                let vals = host_f32(out.value.clone().detach());
                let x = self.targets(segs, t, &lp, &vals, weights);
                if self.algo == Algo::Ppo && first_epoch {
                    *ppo = Some(x);
                    ppo.as_ref().unwrap()
                } else {
                    tg_now = x;
                    &tg_now
                }
            }
        };
        let coef = Tensor::<1>::from_data(TensorData::new(tg.coef.clone(), [rows]), d);
        let policy = match self.algo {
            Algo::Vtrace => (logp.clone() * coef).neg(),
            Algo::Ppo => {
                let clip = self.cfg.ppo.clip as f64;
                let old = Tensor::<1>::from_data(TensorData::new(behaviour, [rows]), d);
                let ratio = (logp.clone() - old).exp();
                let a = ratio.clone() * coef.clone();
                let b2 = ratio.clamp(1.0 - clip, 1.0 + clip) * coef;
                a.min_pair(b2).neg()
            }
        };
        let policy = (policy * maskt.clone()).sum() / n as f64;
        let vs = Tensor::<2>::from_data(TensorData::new(tg.vs.clone(), [rows, 4]), d);
        let value = ((out.value.clone() - vs).powi_scalar(2).sum_dim(1).reshape([rows]) * maskt.clone()).sum() * (0.5 * self.cfg.loss.value as f64 / n as f64);

        let e = &self.cfg.entropy;
        let ent_coef = [e.action, e.delay, e.queued, e.selected, e.target_unit, e.target_loc];
        let ent = &out.heads.entropy;
        let ent_rows = [&ent.action, &ent.delay, &ent.queued, &ent.selected, &ent.target_unit, &ent.target_loc];
        let mut entropy = Tensor::<1>::zeros([1], d);
        let mut ent_stats = [0f32; HEADS];
        for h in 0..HEADS {
            let term = (ent_rows[h].clone() * maskt.clone()).sum() / n as f64;
            ent_stats[h] = host_f32(term.clone().detach())[0];
            entropy = entropy + term * ent_coef[h] as f64;
        }

        // KL(π ‖ π_BC) with the frozen BC net on the same inputs (AMB-042-9).
        let bc = self.bc.forward_seq(&ob, &ab, Some(&pb), s0, t);
        let kl = kl_per_head(&out.heads.dists, &bc.heads.dists);
        let kl_rows = [&kl.action, &kl.delay, &kl.queued, &kl.selected, &kl.target_unit, &kl.target_loc];
        let mut kl_total = Tensor::<1>::zeros([1], d);
        let mut kl_stats = [0f32; HEADS];
        for h in 0..HEADS {
            let term = (kl_rows[h].clone() * maskt.clone()).sum() / n as f64;
            kl_stats[h] = host_f32(term.clone().detach())[0];
            kl_total = kl_total + term;
        }

        let loss = policy.clone() + value.clone() - entropy + kl_total.clone() * kl_coef as f64;
        let part = UpdateStats {
            policy: host_f32(policy.detach())[0],
            value: host_f32(value.detach())[0],
            kl: host_f32(kl_total.detach())[0],
            kl_heads: kl_stats,
            entropy_heads: ent_stats,
            rho_mean: tg.rho_sum,
            adv_mean: tg.adv_sum,
            steps: mask.iter().filter(|&&m| m > 0.0).count(),
            ..Default::default()
        };
        (loss.reshape([1]).sum(), part)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rl::testkit;
    use crate::rl::trajectory::{PackedObs, Step};
    use burn::tensor::{Int, Tensor};
    use uranium_env::features::PrivilegedTensors;
    use uranium_env::features::executor::HostAction;
    use uranium_model::LstmCore;
    use uranium_model::batch::{ActionColumns, TARGET_LOC, TARGET_UNIT};
    use uranium_model::collate::{input_dims, obs_batch};
    use uranium_model::config::CoreKind;
    use uranium_model::core::Core;

    fn tiny() -> (UraniumNetConfig, ActionColumns) {
        let cfg = uranium_model::synthetic::tiny_config(CoreKind::Lstm).with_input(input_dims());
        (cfg, uranium_model::synthetic::columns())
    }

    fn ints(t: Tensor<1, Int>) -> Vec<i64> {
        t.into_data().convert::<i64>().try_into_vec::<i64>().unwrap()
    }

    /// On-policy segments: the network itself picks every action (behaviour = target at start).
    fn segments(net: &UraniumNet<LstmCore>, cols: &ActionColumns, lanes: usize, t: usize, d: &Device) -> Vec<Segment> {
        (0..lanes)
            .map(|lane| {
                let obs = testkit::observations(t);
                let mut state = net.init_state(1, d);
                let s0 = LstmCore::state_to_host(&state);
                let mut steps = Vec::new();
                for (k, o) in obs.iter().enumerate() {
                    let ob = obs_batch(&[o], d);
                    let n = ob.entities();
                    let out = net.act(&ob, state, 1.0);
                    state = out.state;
                    let a = ints(out.action.action)[0] as usize;
                    let sel: Vec<i64> = out.action.selected.into_data().convert::<i64>().try_into_vec::<i64>().unwrap();
                    let kind = cols.target_kind[a];
                    let action = HostAction {
                        action: a as u16,
                        delay_bucket: ints(out.action.delay)[0] as usize,
                        queued: ints(out.action.queued)[0] == 1,
                        selected: if cols.uses_select[a] == 1 { sel.iter().take_while(|&&i| i != n as i64).map(|&i| i as usize).collect() } else { vec![] },
                        target_unit: (kind == TARGET_UNIT).then(|| ints(out.action.target_unit.clone())[0] as usize),
                        target_loc: (kind == TARGET_LOC).then(|| ints(out.action.target_loc.clone())[0] as usize),
                    };
                    let lp = out.log_prob.into_data().convert::<f32>().try_into_vec::<f32>().unwrap()[0];
                    steps.push(Step {
                        obs: PackedObs::pack(o),
                        action,
                        log_prob: lp,
                        head_log_prob: [0.0; HEADS],
                        reward: [if k == t - 1 { 1.0 } else { 0.0 }, 0.0, 0.0, 0.01 * k as f32],
                        privileged: PrivilegedTensors { summary: vec![0.0; uranium_env::features::extract::dims().privileged()] },
                        version: 0,
                    });
                }
                Segment { agent: 0, game_id: lane as u64, steps, state0: s0, terminal: lane % 2 == 0 }
            })
            .collect()
    }

    fn learner(algo: Algo) -> (Learner<LstmCore>, Vec<Segment>) {
        let d = Device::flex();
        d.seed(5);
        let ad = d.clone().autodiff();
        let (cfg, cols) = tiny();
        let net = UraniumNet::<LstmCore>::new(&cfg, &cols, &ad);
        let mut rl = RlConfig::default_file();
        rl.lanes = 2;
        rl.seq_len = 4;
        rl.micro_lanes = 1;
        rl.optimizer.lr = 1e-3;
        let segs = segments(&net.valid(), &cols, 2, 4, &d);
        (Learner::new(net.clone(), net, cfg, rl, algo, ad), segs)
    }

    fn value_of(l: &Learner<LstmCore>, seg: &Segment) -> Vec<f32> {
        let d = Device::flex();
        let o = seg.steps[0].obs.unpack();
        let v = l.net.valid();
        let out = v.act(&obs_batch(&[&o], &d), v.init_state(1, &d), 0.0);
        out.value.into_data().convert::<f32>().try_into_vec::<f32>().unwrap()
    }

    /// 042: a V-trace update trains the right steps (terminal: all, else all but the bootstrap
    /// step), starts with KL(π‖π_BC) = 0 when π = π_BC, changes the weights, and KL grows after.
    #[test]
    fn vtrace_update_042() {
        let (mut l, segs) = learner(Algo::Vtrace);
        let before = value_of(&l, &segs[0]);
        let s = l.update(&segs, [1.0, 0.3, 0.3, 0.05]);
        assert_eq!(s.steps, 4 + 3);
        assert!(s.loss.is_finite() && s.policy.is_finite() && s.value.is_finite(), "{s:?}");
        assert!(s.kl.abs() < 1e-4, "KL to itself {}", s.kl);
        assert!((s.rho_mean - 1.0).abs() < 1e-3, "on-policy rho {}", s.rho_mean);
        assert_eq!(l.update, 1);
        assert_ne!(value_of(&l, &segs[0]), before, "weights changed");
        let s2 = l.update(&segs, [1.0, 0.3, 0.3, 0.05]);
        assert!(s2.kl > 0.0, "KL after a step {}", s2.kl);
        assert!(s2.entropy_heads[0] > 0.0);
    }

    /// 042 PPO baseline: two epochs over the same batch with clipped ratios.
    #[test]
    fn ppo_update_042() {
        let (mut l, segs) = learner(Algo::Ppo);
        let s = l.update(&segs, [1.0, 0.0, 0.0, 0.0]);
        assert!(s.loss.is_finite() && s.kl.abs() < 1e-4, "{s:?}");
        assert_eq!(l.update, 1);
    }

    /// 042 reset: an exploiter goes back to the BC weights.
    #[test]
    fn reset_to_bc_042() {
        let (mut l, segs) = learner(Algo::Vtrace);
        let bc = value_of(&l, &segs[0]);
        l.update(&segs, [1.0, 0.0, 0.0, 0.0]);
        l.reset_to_bc();
        assert_eq!(value_of(&l, &segs[0]), bc);
        assert_eq!(l.update, 0);
    }
}
