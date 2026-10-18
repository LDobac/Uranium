//! Autoregressive action heads (032 §2.5): ActionType -> Delay -> Queued -> Selected ->
//! TargetUnit | Location. One code path serves teacher forcing (training, labels injected) and
//! sampling (inference, Gumbel-max inside the forward, 032-R3, 032-R9).

use burn::module::{Module, Param};
use burn::nn::conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig};
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::tensor::activation::{log_softmax, relu, sigmoid};
use burn::tensor::{Bool, Device, Distribution, Int, Tensor};

use crate::batch::{ActionBatch, ActionTableTensors, ObsBatch, gather_cap, gather_loc};
use crate::config::{HeadsConfig, InputDims, SpatialConfig};
use crate::layers::{GruCell, Mlp, ResMlp, entropy, flag, gumbel_argmax, log_softmax_f32, masked_log_softmax, pick, select_bool, zeros_f32};
use crate::spatial::SpatialOut;

/// Tests only: teacher forcing computes the location head on every row (the reference for the
/// used-rows path).
#[doc(hidden)]
pub static LOCATION_ALL_ROWS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub enum Mode<'a> {
    /// Inject labels (032-R3).
    Teacher(&'a ActionBatch),
    /// Sample with Gumbel-max at a temperature; 0 = greedy.
    Sample { temp: f32 },
}

/// Per-head quantities, [B] each, already multiplied by the head's usage (032-R10).
#[derive(Debug, Clone)]
pub struct PerHead {
    pub action: Tensor<1>,
    pub delay: Tensor<1>,
    pub queued: Tensor<1>,
    pub selected: Tensor<1>,
    pub target_unit: Tensor<1>,
    pub target_loc: Tensor<1>,
}

impl PerHead {
    pub fn sum(&self) -> Tensor<1> {
        self.action.clone()
            + self.delay.clone()
            + self.queued.clone()
            + self.selected.clone()
            + self.target_unit.clone()
            + self.target_loc.clone()
    }

    /// [B,6] in head order action, delay, queued, selected, target_unit, target_loc (032-R15).
    pub fn stack(&self) -> Tensor<2> {
        let b = self.action.dims()[0];
        let col = |t: &Tensor<1>| t.clone().reshape([b, 1]);
        Tensor::cat(vec![col(&self.action), col(&self.delay), col(&self.queued), col(&self.selected), col(&self.target_unit), col(&self.target_loc)], 1)
    }
}

/// Number of autoregressive heads (032 §2.5).
pub const HEADS: usize = 6;

/// Masked log-softmax distributions (for per-head KL in RL, 042) and head usage.
#[derive(Debug, Clone)]
pub struct Dists {
    /// [B,A]
    pub action: Tensor<2>,
    /// [B,16]
    pub delay: Tensor<2>,
    /// [B,2]
    pub queued: Tensor<2>,
    /// [B,S,N+1]; step k is meaningful where `selected_active[:,k]`.
    pub selected: Tensor<3>,
    /// [B,S]
    pub selected_active: Tensor<2, Bool>,
    /// [B,N]
    pub target_unit: Tensor<2>,
    /// [B,M*M]
    pub target_loc: Tensor<2>,
    pub uses_queued: Tensor<1, Bool>,
    pub uses_select: Tensor<1, Bool>,
    pub uses_target_unit: Tensor<1, Bool>,
    pub uses_target_loc: Tensor<1, Bool>,
}

#[derive(Debug, Clone)]
pub struct HeadsOut {
    /// The labels (teacher) or the sampled action (sample).
    pub chosen: ActionBatch,
    /// log p of `chosen` per head.
    pub logp: PerHead,
    pub entropy: PerHead,
    pub dists: Dists,
}

#[derive(Module, Debug)]
struct LocationHead {
    ar: Linear,
    ups: Vec<ConvTranspose2d>,
    out: Conv2d,
    #[module(skip)]
    ar_channels: usize,
}

impl LocationHead {
    fn new(input: &InputDims, h: &HeadsConfig, s: &SpatialConfig, device: &Device) -> Self {
        let side = input.map_size / 16;
        let [c1, c2, c3, c4] = s.stages;
        let up = |i, o| ConvTranspose2dConfig::new([i, o], [4, 4]).with_stride([2, 2]).with_padding([1, 1]).init(device);
        Self {
            ar: LinearConfig::new(h.ar, h.loc_ar_channels * side * side).init(device),
            ups: vec![up(h.loc_ar_channels + c4, c3), up(2 * c3, c2), up(2 * c2, c1), up(2 * c1, 32)],
            out: Conv2dConfig::new([32, 1], [1, 1]).init(device),
            ar_channels: h.loc_ar_channels,
        }
    }

    /// -> logits [B, M*M]
    fn forward(&self, ar: Tensor<2>, sp: &SpatialOut) -> Tensor<2> {
        let [b, _, s, _] = sp.s8.dims();
        let a = relu(self.ar.forward(ar)).reshape([b, self.ar_channels, s, s]);
        let mut x = Tensor::cat(vec![a, sp.s8.clone()], 1);
        let skips = [sp.skips[2].clone(), sp.skips[1].clone(), sp.skips[0].clone()];
        for (i, up) in self.ups.iter().enumerate() {
            x = relu(up.forward(x));
            if i < 3 {
                x = Tensor::cat(vec![x, skips[i].clone()], 1);
            }
        }
        let [_, _, m, _] = x.dims();
        self.out.forward(x).reshape([b, m * m])
    }
}

#[derive(Module, Debug)]
pub struct Heads {
    // 1 ActionType
    type_in: Linear,
    type_res: Vec<ResMlp>,
    type_gate: Linear,
    type_out: Linear,
    ar_h: Linear,
    ar_action: Embedding,
    // 2 Delay
    delay: Mlp,
    ar_delay: Embedding,
    // 3 Queued
    queued: Mlp,
    ar_queued: Embedding,
    // 4 Selected
    sel_key: Linear,
    sel_end: Param<Tensor<1>>,
    sel_query: Linear,
    sel_gru: GruCell,
    ar_selected: Linear,
    // 5 TargetUnit
    tgt_key: Linear,
    tgt_query: Linear,
    // 6 Location
    location: LocationHead,
    #[module(skip)]
    max_select: usize,
}

/// Re-normalise log-probs at a sampling temperature (behaviour policy log-probs, 042).
fn tempered(logp: Tensor<2>, temp: f32) -> Tensor<2> {
    if temp <= 0.0 || (temp - 1.0).abs() < 1e-6 { logp } else { log_softmax(logp / temp as f64, 1) }
}

impl Heads {
    pub fn new(input: &InputDims, h: &HeadsConfig, s: &SpatialConfig, d_core: usize, d_ctx: usize, d_entity: usize, device: &Device) -> Self {
        let lin = |i, o| LinearConfig::new(i, o).init(device);
        let emb = |n, d| EmbeddingConfig::new(n, d).init(device);
        Self {
            type_in: lin(d_core, h.ar),
            type_res: (0..2).map(|_| ResMlp::new(h.ar, device)).collect(),
            type_gate: lin(d_ctx, h.ar),
            type_out: lin(h.ar, input.actions),
            ar_h: lin(d_core, h.ar),
            ar_action: emb(input.actions, h.ar),
            delay: Mlp::new(h.ar, 128, h.delay_buckets, device),
            ar_delay: emb(h.delay_buckets, h.ar),
            queued: Mlp::new(h.ar, 64, 2, device),
            ar_queued: emb(2, h.ar),
            sel_key: lin(d_entity, h.key),
            sel_end: Param::from_tensor(Tensor::random([h.key], Distribution::Normal(0.0, 0.02), device)),
            sel_query: lin(h.ar, h.key),
            sel_gru: GruCell::new(h.key, h.key, device),
            ar_selected: lin(h.key, h.ar),
            tgt_key: lin(d_entity, h.key),
            tgt_query: lin(h.ar, h.key),
            location: LocationHead::new(input, h, s, device),
            max_select: h.max_select,
        }
    }

    /// `h` [B,core], `c` [B,ctx] (GLU gate), `e` [B,N,d] entity embeddings.
    pub fn forward(&self, mode: Mode, h: Tensor<2>, c: Tensor<2>, e: Tensor<3>, sp: &SpatialOut, obs: &ObsBatch, table: &ActionTableTensors) -> HeadsOut {
        let [b, n, _] = e.dims();
        let temp = match mode {
            Mode::Teacher(_) => 1.0,
            Mode::Sample { temp } => temp,
        };
        let choose = |logp: &Tensor<2>, label: Option<Tensor<1, Int>>| -> Tensor<1, Int> {
            match label {
                Some(l) => l.clamp(0, logp.dims()[1] as i64 - 1),
                None => gumbel_argmax(logp.clone(), temp),
            }
        };
        let labels = match mode {
            Mode::Teacher(l) => Some(l),
            Mode::Sample { .. } => None,
        };

        // 1 ActionType
        let mut x = relu(self.type_in.forward(h.clone()));
        for r in &self.type_res {
            x = r.forward(x);
        }
        let x = relu(x) * sigmoid(self.type_gate.forward(c));
        let lp_action = tempered(masked_log_softmax(self.type_out.forward(x), obs.type_mask.clone()), temp);
        let action = choose(&lp_action, labels.map(|l| l.action.clone()));
        let info = table.info(action.clone());
        let mut ar = self.ar_h.forward(h) + flat(self.ar_action.forward(action.clone().reshape([b, 1])));
        crate::prof::mark("fwd.head.action");

        // 2 Delay (always used)
        let lp_delay = tempered(log_softmax_f32(self.delay.forward(ar.clone())), temp);
        let delay = choose(&lp_delay, labels.map(|l| l.delay.clone()));
        ar = ar + flat(self.ar_delay.forward(delay.clone().reshape([b, 1])));

        // 3 Queued (fixed to 0 when unused)
        let lp_queued = tempered(log_softmax_f32(self.queued.forward(ar.clone())), temp);
        let queued = choose(&lp_queued, labels.map(|l| l.queued.clone())) * info.uses_queued.clone().int();
        ar = ar + flat(self.ar_queued.forward(queued.clone().reshape([b, 1])));
        crate::prof::mark("fwd.head.delay_queued");

        // 4 Selected
        let sel = self.selected(&mut ar, e.clone(), obs, &info, labels.map(|l| l.selected.clone()), temp);
        crate::prof::mark("fwd.head.selected");

        // 5 TargetUnit
        let keys = self.tgt_key.forward(e);
        let q = self.tgt_query.forward(ar.clone());
        let kd = q.dims()[1];
        let logits = keys.matmul(q.reshape([b, kd, 1])).reshape([b, n]);
        let tgt_ok = gather_cap(obs.unit_cap.clone(), info.target_cap.clone()).bool_and(obs.entity_mask.clone());
        let lp_tu = tempered(masked_log_softmax(logits, tgt_ok), temp);
        let target_unit = choose(&lp_tu, labels.map(|l| l.target_unit.clone()));
        crate::prof::mark("fwd.head.target_unit");

        // 6 Location
        let loc_ok = gather_loc(obs.loc_class_mask.clone(), info.loc_class.clone());
        let lp_loc = match labels {
            // Teacher forcing: only rows whose action uses a location (the others' terms are
            // multiplied by 0, 032-R10); their rows of `dists.target_loc` are 0. Same values and
            // gradients for the used rows, about half the upsampling work (2026-10-18).
            Some(_) => self.location_used_rows(ar, sp, loc_ok, info.target_loc.clone()),
            None => tempered(masked_log_softmax(self.location.forward(ar, sp), loc_ok), temp),
        };
        let target_loc = choose(&lp_loc, labels.map(|l| l.target_loc.clone()));
        crate::prof::mark("fwd.head.location");

        let f = |m: &Tensor<1, Bool>| flag(m.clone());
        let (u_q, u_s, u_tu, u_tl) = (f(&info.uses_queued), f(&info.uses_select), f(&info.target_unit), f(&info.target_loc));
        let logp = PerHead {
            action: pick(lp_action.clone(), action.clone()),
            delay: pick(lp_delay.clone(), delay.clone()),
            queued: pick(lp_queued.clone(), queued.clone()) * u_q.clone(),
            selected: sel.logp * u_s.clone(),
            target_unit: pick(lp_tu.clone(), target_unit.clone()) * u_tu.clone(),
            target_loc: pick(lp_loc.clone(), target_loc.clone()) * u_tl.clone(),
        };
        crate::prof::mark("fwd.head.pick");
        let entropy = PerHead {
            action: entropy(lp_action.clone()),
            delay: entropy(lp_delay.clone()),
            queued: entropy(lp_queued.clone()) * u_q,
            selected: sel.entropy * u_s,
            target_unit: entropy(lp_tu.clone()) * u_tu,
            target_loc: entropy(lp_loc.clone()) * u_tl,
        };
        crate::prof::mark("fwd.head.entropy");
        HeadsOut {
            chosen: ActionBatch { action, delay, queued, selected: sel.chosen, target_unit, target_loc },
            logp,
            entropy,
            dists: Dists {
                action: lp_action,
                delay: lp_delay,
                queued: lp_queued,
                selected: sel.dists,
                selected_active: sel.active,
                target_unit: lp_tu,
                target_loc: lp_loc,
                uses_queued: info.uses_queued,
                uses_select: info.uses_select,
                uses_target_unit: info.target_unit,
                uses_target_loc: info.target_loc,
            },
        }
    }

    /// Masked location log-probs [B, M*M] computed for the rows where `used` is set; other rows 0.
    fn location_used_rows(&self, ar: Tensor<2>, sp: &SpatialOut, loc_ok: Tensor<2, Bool>, used: Tensor<1, Bool>) -> Tensor<2> {
        if LOCATION_ALL_ROWS.load(std::sync::atomic::Ordering::Relaxed) {
            return masked_log_softmax(self.location.forward(ar, sp), loc_ok);
        }
        let [b, mm] = loc_ok.dims();
        let device = ar.device();
        let flags = used.into_data().convert::<bool>().try_into_vec::<bool>().expect("bool mask");
        let rows: Vec<i64> = flags.iter().enumerate().filter(|(_, u)| **u).map(|(i, _)| i as i64).collect();
        // No used row: compute every row anyway, so the location parameters still get their (zero)
        // gradient and the optimizer treats them as before (AdamW skips parameters without one).
        if rows.len() == b || rows.is_empty() {
            return masked_log_softmax(self.location.forward(ar, sp), loc_ok);
        }
        let k = rows.len();
        let idx = Tensor::<1, Int>::from_data(burn::tensor::TensorData::new(rows, [k]), &device);
        let sub = SpatialOut {
            s8: sp.s8.clone().select(0, idx.clone()),
            skips: sp.skips.clone().map(|s| s.select(0, idx.clone())),
            pool: sp.pool.clone().select(0, idx.clone()),
        };
        let lp = masked_log_softmax(self.location.forward(ar.select(0, idx.clone()), &sub), select_bool(loc_ok, 0, idx.clone()));
        Tensor::<2>::zeros([b, mm], &device).select_assign(0, idx, lp, burn::tensor::IndexingUpdateOp::Add)
    }

    /// Pointer network over own eligible units with an END key (032 §2.5 #4, 031-R3, 032-R12).
    /// At most `max_select[a]` picks; END is allowed after the first pick (or when nothing is
    /// eligible); reaching `max_select` ends the selection without an END step.
    fn selected(
        &self,
        ar: &mut Tensor<2>,
        e: Tensor<3>,
        obs: &ObsBatch,
        info: &crate::batch::ActionInfo,
        labels: Option<Tensor<2, Int>>,
        temp: f32,
    ) -> Selected {
        let [b, n, _] = e.dims();
        let device = e.device();
        let keys = self.sel_key.forward(e);
        let kd = keys.dims()[2];
        let end = self.sel_end.val().reshape([1, 1, kd]).expand([b, 1, kd]);
        let keys_all = Tensor::cat(vec![keys, end], 1);

        let base = gather_cap(obs.unit_cap.clone(), info.select_cap.clone()).bool_and(obs.is_self()).bool_and(obs.entity_mask.clone());
        let mut chosen_units = Tensor::<2, Int>::zeros([b, n], &device).equal_elem(1);
        let mut ended = info.uses_select.clone().bool_not();
        let mut q = self.sel_query.forward(ar.clone());
        let mut sum_keys = Tensor::<2>::zeros([b, kd], &device);
        let mut count = Tensor::<1>::zeros([b], &device);
        let (mut logp, mut ent) = (zeros_f32(b, &device), zeros_f32(b, &device));
        let (mut choices, mut dists, mut actives) = (vec![], vec![], vec![]);
        let true_col = Tensor::<2, Int>::ones([b, 1], &device).equal_elem(1);

        for k in 0..self.max_select {
            let active = ended.clone().bool_not().bool_and(info.max_select.clone().greater_elem(k as i64));
            let units_ok = base.clone().bool_and(chosen_units.clone().bool_not());
            let end_ok = if k == 0 { units_ok.clone().any_dim(1).bool_not() } else { true_col.clone() };
            let allowed = Tensor::cat(vec![units_ok, end_ok], 1);
            let logits = keys_all.clone().matmul(q.clone().reshape([b, kd, 1])).reshape([b, n + 1]);
            let lp = tempered(masked_log_softmax(logits, allowed), temp);
            let choice = match &labels {
                Some(l) => l.clone().narrow(1, k, 1).reshape([b]).clamp(0, n as i64),
                None => gumbel_argmax(lp.clone(), temp),
            };
            let af = flag(active.clone());
            logp = logp + pick(lp.clone(), choice.clone()) * af.clone();
            ent = ent + entropy(lp.clone()) * af;

            let is_end = choice.clone().equal_elem(n as i64);
            let take = active.clone().bool_and(is_end.clone().bool_not());
            let one_hot: Tensor<2> = choice.clone().one_hot::<2>(n + 1).float();
            let key = one_hot.clone().reshape([b, 1, n + 1]).matmul(keys_all.clone()).reshape([b, kd]);
            let tf = take.clone().float();
            let picked = one_hot.narrow(1, 0, n).equal_elem(1.0).bool_and(take.clone().reshape([b, 1]).expand([b, n]));
            chosen_units = chosen_units.bool_or(picked);
            sum_keys = sum_keys + key.clone() * tf.clone().reshape([b, 1]);
            count = count + tf;
            q = self.sel_gru.forward(key, q);
            ended = ended.bool_or(active.clone().bool_and(is_end)).bool_or(info.max_select.clone().lower_equal_elem(k as i64 + 1));

            // Output END for inactive steps so `chosen.selected` is canonical (032-R12).
            let end_fill = Tensor::<1, Int>::full([b], n as i64, &device);
            choices.push(choice.mask_where(active.clone().bool_not(), end_fill).reshape([b, 1]));
            dists.push(lp.reshape([b, 1, n + 1]));
            actives.push(active.reshape([b, 1]));
        }
        let mean_key = sum_keys / count.clamp_min(1.0).reshape([b, 1]);
        *ar = ar.clone() + self.ar_selected.forward(mean_key) * info.uses_select.clone().float().reshape([b, 1]);
        Selected {
            chosen: Tensor::cat(choices, 1),
            logp,
            entropy: ent,
            dists: Tensor::cat(dists, 1),
            active: Tensor::cat(actives, 1),
        }
    }
}

/// [B,1,D] -> [B,D]
fn flat(t: Tensor<3>) -> Tensor<2> {
    let [b, _, d] = t.dims();
    t.reshape([b, d])
}

struct Selected {
    chosen: Tensor<2, Int>,
    logp: Tensor<1>,
    entropy: Tensor<1>,
    dists: Tensor<3>,
    active: Tensor<2, Bool>,
}
