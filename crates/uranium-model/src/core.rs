//! Temporal core (032 §2.4): `LstmCore` (default) and `GtrxlCore` (M5 ablation) behind one
//! `Core` trait (032-AC6). State is reset per episode by starting from `init_state` (032-R5).

use burn::module::{Module, ModuleDisplay, Param};
use burn::nn::{LayerNorm, Linear, LinearConfig, Lstm, LstmConfig, LstmState};
use burn::tensor::activation::{relu, sigmoid, tanh};
use burn::tensor::{Bool, Device, Distribution, IndexingUpdateOp, Int, Tensor, TensorData};

use crate::config::CoreConfig;
use crate::layers::{Attention, FeedForward, layer_norm};

pub trait Core: Module + ModuleDisplay {
    type State: Clone + core::fmt::Debug + Send;

    fn init_state(&self, batch: usize, device: &Device) -> Self::State;
    /// Inference step: x [B,d_in] -> h [B,hidden].
    fn step(&self, x: Tensor<2>, s: Self::State) -> (Tensor<2>, Self::State);
    /// Training over [B,T,d_in] (032-R5 truncated BPTT; `s0` comes from the actor).
    fn forward_seq(&self, x: Tensor<3>, s0: Self::State) -> (Tensor<3>, Self::State);
    /// Cut the gradient path at a truncation boundary.
    fn detach_state(s: Self::State) -> Self::State;
    /// Zero the state of rows where `keep` is false (episode start in that lane, 032-R5).
    fn mask_state(s: Self::State, keep: Tensor<1, Bool>) -> Self::State;
    /// Rows `rows` of a slot state as a batch state (inference server slots, 032-R14).
    fn select_rows(s: &Self::State, rows: Tensor<1, Int>) -> Self::State;
    /// Writes a batch state back into rows `rows` (distinct) of a slot state; other rows are kept
    /// bit for bit (032-R14).
    fn scatter_rows(s: Self::State, rows: Tensor<1, Int>, sub: Self::State) -> Self::State;
    fn hidden(&self) -> usize;
    /// Host copy of a state (032-R16, 042 core_state0); `state_from_host` rebuilds it bit for bit.
    fn state_to_host(s: &Self::State) -> HostState;
    fn state_from_host(h: &HostState, device: &Device) -> Self::State;
}

/// A Core state on the host: every tensor as f32 values with its shape (bools as 0/1). Row 0 of
/// every shape is the batch.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct HostState {
    pub parts: Vec<(Vec<usize>, Vec<f32>)>,
}

impl HostState {
    pub fn batch(&self) -> usize {
        self.parts.first().map_or(0, |(s, _)| s[0])
    }

    /// Stacks single-batch (or any batch) states along the batch dimension.
    pub fn cat(rows: &[&HostState]) -> HostState {
        let Some(first) = rows.first() else { return HostState::default() };
        let parts = (0..first.parts.len())
            .map(|i| {
                let mut shape = first.parts[i].0.clone();
                shape[0] = rows.iter().map(|r| r.parts[i].0[0]).sum();
                let data = rows.iter().flat_map(|r| r.parts[i].1.iter().copied()).collect();
                (shape, data)
            })
            .collect();
        HostState { parts }
    }
}

fn to_host<const D: usize>(t: &Tensor<D>) -> (Vec<usize>, Vec<f32>) {
    (t.dims().to_vec(), t.clone().into_data().convert::<f32>().try_into_vec::<f32>().expect("f32 state"))
}

fn from_host<const D: usize>(p: &(Vec<usize>, Vec<f32>), device: &Device) -> Tensor<D> {
    let shape: [usize; D] = p.0.clone().try_into().expect("state rank");
    Tensor::from_data(TensorData::new(p.1.clone(), shape), device)
}

/// `t[rows] = sub` exactly: the new rows are added onto zeros (0 + v = v) and selected with a row
/// mask, so neither the written nor the untouched rows go through float arithmetic.
fn put_rows<const D: usize>(t: Tensor<D>, rows: Tensor<1, Int>, sub: Tensor<D>) -> Tensor<D> {
    let dims = t.dims();
    let device = t.device();
    let n = rows.dims()[0];
    let placed = t.zeros_like().select_assign(0, rows.clone(), sub, IndexingUpdateOp::Add);
    let hit = Tensor::<1, Int>::zeros([dims[0]], &device).select_assign(0, rows, Tensor::ones([n], &device), IndexingUpdateOp::Add);
    let mut shape = [1usize; D];
    shape[0] = dims[0];
    let mask = hit.equal_elem(1).reshape(shape).expand(dims);
    t.mask_where(mask, placed)
}

/// Bool rows go through Int like `init_state` builds them (same bool storage on every backend).
fn put_rows_bool<const D: usize>(t: Tensor<D, Bool>, rows: Tensor<1, Int>, sub: Tensor<D, Bool>) -> Tensor<D, Bool> {
    let t = t.int();
    let dims = t.dims();
    let device = t.device();
    let n = rows.dims()[0];
    let placed = t.zeros_like().select_assign(0, rows.clone(), sub.int(), IndexingUpdateOp::Add);
    let hit = Tensor::<1, Int>::zeros([dims[0]], &device).select_assign(0, rows, Tensor::ones([n], &device), IndexingUpdateOp::Add);
    let mut shape = [1usize; D];
    shape[0] = dims[0];
    t.mask_where(hit.equal_elem(1).reshape(shape).expand(dims), placed).equal_elem(1)
}

// ---------------------------------------------------------------- LSTM

#[derive(Module, Debug)]
pub struct LstmCore {
    lstm: Lstm,
    #[module(skip)]
    hidden: usize,
}

impl LstmCore {
    pub fn new(d_in: usize, c: &CoreConfig, device: &Device) -> Self {
        Self { lstm: LstmConfig::new(d_in, c.hidden, true).init(device), hidden: c.hidden }
    }

    /// The LSTM over [B,T,d] with the same parameters and math as Burn's `Lstm::forward`
    /// (`LstmConfig` defaults: sigmoid gates, tanh cell and output, separate forget gate, no clip),
    /// computed with fewer kernels: the input projections of all T steps and four gates in one
    /// matmul, then one [B,H]x[H,4H] matmul per step (2026-10-18, BC throughput).
    fn run(&self, x: Tensor<3>, s: LstmState<2>) -> (Tensor<3>, LstmState<2>) {
        let l = &self.lstm;
        let forget = l.forget_gate.as_ref().expect("LstmCore is built with a forget gate");
        let gates = [&l.input_gate, forget, &l.output_gate, &l.cell_gate];
        let [b, t, d] = x.dims();
        let h = self.hidden;
        let w_in = Tensor::cat(gates.iter().map(|g| g.input_transform.weight.val()).collect(), 1);
        let w_h = Tensor::cat(gates.iter().map(|g| g.hidden_transform.weight.val()).collect(), 1);
        let bias = |lin: &Linear| lin.bias.as_ref().expect("LSTM gates have biases").val();
        let b_all = Tensor::cat(gates.iter().map(|g| bias(&g.input_transform) + bias(&g.hidden_transform)).collect(), 0);
        let xp = (x.reshape([b * t, d]).matmul(w_in) + b_all.reshape([1, 4 * h])).reshape([b, t, 4 * h]);
        let (mut c, mut hs) = (s.cell, s.hidden);
        let mut outs = Vec::with_capacity(t);
        for k in 0..t {
            let z = xp.clone().narrow(1, k, 1).reshape([b, 4 * h]) + hs.clone().matmul(w_h.clone());
            let i = sigmoid(z.clone().narrow(1, 0, h));
            let f = sigmoid(z.clone().narrow(1, h, h));
            let o = sigmoid(z.clone().narrow(1, 2 * h, h));
            let g = tanh(z.narrow(1, 3 * h, h));
            c = f * c + i * g;
            hs = o * tanh(c.clone());
            outs.push(hs.clone().reshape([b, 1, h]));
        }
        (Tensor::cat(outs, 1), LstmState::new(c, hs))
    }

    /// Burn's own `Lstm::forward` (reference for [`LstmCore::run`] in tests).
    #[doc(hidden)]
    pub fn forward_seq_reference(&self, x: Tensor<3>, s0: LstmState<2>) -> (Tensor<3>, LstmState<2>) {
        self.lstm.forward(x, Some(s0))
    }
}

impl Core for LstmCore {
    type State = LstmState<2>;

    fn init_state(&self, batch: usize, device: &Device) -> Self::State {
        LstmState::new(Tensor::zeros([batch, self.hidden], device), Tensor::zeros([batch, self.hidden], device))
    }

    fn step(&self, x: Tensor<2>, s: Self::State) -> (Tensor<2>, Self::State) {
        let [b, d] = x.dims();
        let (out, s) = self.run(x.reshape([b, 1, d]), s);
        (out.reshape([b, self.hidden]), s)
    }

    fn forward_seq(&self, x: Tensor<3>, s0: Self::State) -> (Tensor<3>, Self::State) {
        self.run(x, s0)
    }

    fn detach_state(s: Self::State) -> Self::State {
        LstmState::new(s.cell.detach(), s.hidden.detach())
    }

    fn mask_state(s: Self::State, keep: Tensor<1, Bool>) -> Self::State {
        let b = keep.dims()[0];
        let k = keep.float().reshape([b, 1]);
        LstmState::new(s.cell * k.clone(), s.hidden * k)
    }

    fn select_rows(s: &Self::State, rows: Tensor<1, Int>) -> Self::State {
        LstmState::new(s.cell.clone().select(0, rows.clone()), s.hidden.clone().select(0, rows))
    }

    fn scatter_rows(s: Self::State, rows: Tensor<1, Int>, sub: Self::State) -> Self::State {
        LstmState::new(put_rows(s.cell, rows.clone(), sub.cell), put_rows(s.hidden, rows, sub.hidden))
    }

    fn hidden(&self) -> usize {
        self.hidden
    }

    fn state_to_host(s: &Self::State) -> HostState {
        HostState { parts: vec![to_host(&s.cell), to_host(&s.hidden)] }
    }

    fn state_from_host(h: &HostState, device: &Device) -> Self::State {
        LstmState::new(from_host(&h.parts[0], device), from_host(&h.parts[1], device))
    }
}

// ---------------------------------------------------------------- GTrXL

/// GRU-type gating residual of GTrXL: g = (1-z)*x + z*h.
#[derive(Module, Debug)]
struct GruGate {
    wr: Linear,
    ur: Linear,
    wz: Linear,
    uz: Linear,
    wg: Linear,
    ug: Linear,
}

/// Gate bias b_g: starts close to the identity map.
const GATE_BIAS: f64 = 2.0;

impl GruGate {
    fn new(d: usize, device: &Device) -> Self {
        let l = || LinearConfig::new(d, d).init(device);
        let nb = || LinearConfig::new(d, d).with_bias(false).init(device);
        Self { wr: l(), ur: nb(), wz: l(), uz: nb(), wg: l(), ug: nb() }
    }

    fn forward(&self, x: Tensor<3>, y: Tensor<3>) -> Tensor<3> {
        let r = sigmoid(self.wr.forward(y.clone()) + self.ur.forward(x.clone()));
        let z = sigmoid(self.wz.forward(y.clone()) + self.uz.forward(x.clone()) - GATE_BIAS);
        let h = tanh(self.wg.forward(y) + self.ug.forward(r * x.clone()));
        x * (z.clone().neg() + 1.0) + z * h
    }
}

#[derive(Module, Debug)]
struct GtrxlLayer {
    ln1: LayerNorm,
    attn: Attention,
    gate1: GruGate,
    ln2: LayerNorm,
    ffn: FeedForward,
    gate2: GruGate,
}

#[derive(Module, Debug)]
pub struct GtrxlCore {
    input: Linear,
    layers: Vec<GtrxlLayer>,
    /// [M+1, layers*heads] relative position bias (distance 0..=M), one-hot @ table like 032-R13.
    rel_table: Param<Tensor<2>>,
    #[module(skip)]
    hidden: usize,
    #[module(skip)]
    heads: usize,
    #[module(skip)]
    memory: usize,
}

/// Per layer the last M layer inputs (recomputing K/V from them keeps training and inference on
/// the same parameters), plus which memory slots hold real steps.
#[derive(Debug, Clone)]
pub struct GtrxlState {
    pub memory: Vec<Tensor<3>>,
    /// [B,M]
    pub valid: Tensor<2, Bool>,
}

impl GtrxlCore {
    pub fn new(d_in: usize, c: &CoreConfig, device: &Device) -> Self {
        let d = c.hidden;
        let layers = (0..c.gtrxl_layers)
            .map(|_| GtrxlLayer {
                ln1: layer_norm(d, device),
                attn: Attention::new(d, c.gtrxl_heads, device),
                gate1: GruGate::new(d, device),
                ln2: layer_norm(d, device),
                ffn: FeedForward::new(d, 4 * d, 0.0, device),
                gate2: GruGate::new(d, device),
            })
            .collect();
        let rel = Tensor::random([c.gtrxl_memory + 1, c.gtrxl_layers * c.gtrxl_heads], Distribution::Normal(0.0, 0.02), device);
        Self {
            input: LinearConfig::new(d_in, d).init(device),
            layers,
            rel_table: Param::from_tensor(rel),
            hidden: d,
            heads: c.gtrxl_heads,
            memory: c.gtrxl_memory,
        }
    }

    /// Relative distance one-hot [T, M+T, M+1] and the causal window mask [T, M+T].
    fn relative(&self, t: usize, device: &Device) -> (Tensor<3>, Tensor<2, Bool>) {
        let m = self.memory;
        let tk = m + t;
        let mut idx = Vec::with_capacity(t * tk);
        let mut ok = Vec::with_capacity(t * tk);
        for q in 0..t {
            for k in 0..tk {
                let rel = (m + q) as i64 - k as i64;
                let inside = (0..=m as i64).contains(&rel);
                idx.push(if inside { rel as i32 } else { 0 });
                ok.push(inside);
            }
        }
        let idx = Tensor::<2, Int>::from_data(TensorData::new(idx, [t, tk]), device);
        let ok = Tensor::<2, Bool>::from_data(TensorData::new(ok, [t, tk]), device);
        (idx.one_hot::<3>(m + 1).float(), ok)
    }
}

impl Core for GtrxlCore {
    type State = GtrxlState;

    fn init_state(&self, batch: usize, device: &Device) -> Self::State {
        GtrxlState {
            memory: (0..self.layers.len()).map(|_| Tensor::zeros([batch, self.memory, self.hidden], device)).collect(),
            valid: Tensor::<2, Int>::zeros([batch, self.memory], device).equal_elem(1),
        }
    }

    fn step(&self, x: Tensor<2>, s: Self::State) -> (Tensor<2>, Self::State) {
        let [b, d] = x.dims();
        let (out, s) = self.forward_seq(x.reshape([b, 1, d]), s);
        (out.reshape([b, self.hidden]), s)
    }

    fn forward_seq(&self, x: Tensor<3>, s0: Self::State) -> (Tensor<3>, Self::State) {
        let [b, t, _] = x.dims();
        let (m, h) = (self.memory, self.heads);
        let device = x.device();
        let (one_hot, window) = self.relative(t, &device);
        let lh = self.rel_table.dims()[1];
        let bias_all = one_hot.reshape([t * (m + t), m + 1]).matmul(self.rel_table.val()).reshape([t, m + t, lh]).permute([2, 0, 1]);
        // Through Int: a state gathered / scattered by the inference server may carry another bool
        // storage than a fresh one on some backends (032-R14).
        let valid_ext = Tensor::cat(vec![s0.valid.int(), Tensor::<2, Int>::ones([b, t], &device)], 1).equal_elem(1);
        let key_ok = window.reshape([1, t, m + t]).expand([b, t, m + t]).bool_and(valid_ext.clone().reshape([b, 1, m + t]).expand([b, t, m + t]));

        let mut x = self.input.forward(x);
        let mut memory = Vec::with_capacity(self.layers.len());
        for (l, (layer, mem)) in self.layers.iter().zip(s0.memory).enumerate() {
            let keys_in = Tensor::cat(vec![mem, x.clone()], 1);
            memory.push(keys_in.clone().narrow(1, t, m));
            let y = layer.ln1.forward(keys_in);
            let q = y.clone().narrow(1, m, t);
            let bias = bias_all.clone().narrow(0, l * h, h).reshape([1, h, t, m + t]).expand([b, h, t, m + t]);
            let a = layer.attn.forward(q, y, Some(bias), key_ok.clone());
            let x1 = layer.gate1.forward(x, relu(a));
            let f = layer.ffn.forward(layer.ln2.forward(x1.clone()));
            x = layer.gate2.forward(x1, relu(f));
        }
        let valid = valid_ext.narrow(1, t, m);
        (x, GtrxlState { memory, valid })
    }

    fn detach_state(s: Self::State) -> Self::State {
        GtrxlState { memory: s.memory.into_iter().map(|t| t.detach()).collect(), valid: s.valid }
    }

    fn mask_state(s: Self::State, keep: Tensor<1, Bool>) -> Self::State {
        let b = keep.dims()[0];
        let m = s.valid.dims()[1];
        let k = keep.clone().float().reshape([b, 1, 1]);
        let valid = s.valid.bool_and(keep.reshape([b, 1]).expand([b, m]));
        GtrxlState { memory: s.memory.into_iter().map(|t| t * k.clone()).collect(), valid }
    }

    fn select_rows(s: &Self::State, rows: Tensor<1, Int>) -> Self::State {
        GtrxlState { memory: s.memory.iter().map(|t| t.clone().select(0, rows.clone())).collect(), valid: crate::layers::select_bool(s.valid.clone(), 0, rows) }
    }

    fn scatter_rows(s: Self::State, rows: Tensor<1, Int>, sub: Self::State) -> Self::State {
        GtrxlState {
            memory: s.memory.into_iter().zip(sub.memory).map(|(t, u)| put_rows(t, rows.clone(), u)).collect(),
            valid: put_rows_bool(s.valid, rows, sub.valid),
        }
    }

    fn hidden(&self) -> usize {
        self.hidden
    }

    fn state_to_host(s: &Self::State) -> HostState {
        let mut parts: Vec<_> = s.memory.iter().map(to_host).collect();
        parts.push(to_host(&s.valid.clone().int().float()));
        HostState { parts }
    }

    fn state_from_host(h: &HostState, device: &Device) -> Self::State {
        let (mem, valid) = h.parts.split_at(h.parts.len() - 1);
        GtrxlState { memory: mem.iter().map(|p| from_host(p, device)).collect(), valid: from_host::<2>(&valid[0], device).equal_elem(1.0) }
    }
}
