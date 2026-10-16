//! Temporal core (032 §2.4): `LstmCore` (default) and `GtrxlCore` (M5 ablation) behind one
//! `Core` trait (032-AC6). State is reset per episode by starting from `init_state` (032-R5).

use burn::module::{Module, ModuleDisplay, Param};
use burn::nn::{LayerNorm, Linear, LinearConfig, Lstm, LstmConfig, LstmState};
use burn::tensor::activation::{relu, sigmoid, tanh};
use burn::tensor::{Bool, Device, Distribution, Int, Tensor, TensorData};

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
    fn hidden(&self) -> usize;
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
}

impl Core for LstmCore {
    type State = LstmState<2>;

    fn init_state(&self, batch: usize, device: &Device) -> Self::State {
        LstmState::new(Tensor::zeros([batch, self.hidden], device), Tensor::zeros([batch, self.hidden], device))
    }

    fn step(&self, x: Tensor<2>, s: Self::State) -> (Tensor<2>, Self::State) {
        let [b, d] = x.dims();
        let (out, s) = self.lstm.forward(x.reshape([b, 1, d]), Some(s));
        (out.reshape([b, self.hidden]), s)
    }

    fn forward_seq(&self, x: Tensor<3>, s0: Self::State) -> (Tensor<3>, Self::State) {
        self.lstm.forward(x, Some(s0))
    }

    fn detach_state(s: Self::State) -> Self::State {
        LstmState::new(s.cell.detach(), s.hidden.detach())
    }

    fn mask_state(s: Self::State, keep: Tensor<1, Bool>) -> Self::State {
        let b = keep.dims()[0];
        let k = keep.float().reshape([b, 1]);
        LstmState::new(s.cell * k.clone(), s.hidden * k)
    }

    fn hidden(&self) -> usize {
        self.hidden
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
        let valid_ext = Tensor::cat(vec![s0.valid, Tensor::<2, Int>::ones([b, t], &device).equal_elem(1)], 1);
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

    fn hidden(&self) -> usize {
        self.hidden
    }
}
