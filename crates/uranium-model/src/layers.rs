//! Building blocks shared by the encoders, core and heads (032 §2).

use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{Dropout, DropoutConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig, PaddingConfig2d};
use burn::tensor::activation::{gelu, log_softmax, relu, sigmoid, softmax, tanh};
use burn::tensor::{Bool, DType, Distribution, Int, Tensor};

/// 032-R4: masked logits.
pub const MASKED: f32 = -1e9;

/// Linear -> ReLU -> Linear.
#[derive(Module, Debug)]
pub struct Mlp {
    l1: Linear,
    l2: Linear,
}

impl Mlp {
    pub fn new(d_in: usize, d_hidden: usize, d_out: usize, device: &burn::tensor::Device) -> Self {
        Self { l1: LinearConfig::new(d_in, d_hidden).init(device), l2: LinearConfig::new(d_hidden, d_out).init(device) }
    }

    pub fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        self.l2.forward(relu(self.l1.forward(x)))
    }
}

/// Pre-activation residual MLP block: x + L2(relu(L1(relu(x)))).
#[derive(Module, Debug)]
pub struct ResMlp {
    l1: Linear,
    l2: Linear,
}

impl ResMlp {
    pub fn new(d: usize, device: &burn::tensor::Device) -> Self {
        Self { l1: LinearConfig::new(d, d).init(device), l2: LinearConfig::new(d, d).init(device) }
    }

    pub fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        x.clone() + self.l2.forward(relu(self.l1.forward(relu(x))))
    }
}

pub fn conv3x3(c_in: usize, c_out: usize, stride: usize, device: &burn::tensor::Device) -> Conv2d {
    Conv2dConfig::new([c_in, c_out], [3, 3])
        .with_stride([stride, stride])
        .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
        .init(device)
}

/// 032 §2.2 ResBlock(2 x Conv3x3).
#[derive(Module, Debug)]
pub struct ResBlock {
    c1: Conv2d,
    c2: Conv2d,
}

impl ResBlock {
    pub fn new(c: usize, device: &burn::tensor::Device) -> Self {
        Self { c1: conv3x3(c, c, 1, device), c2: conv3x3(c, c, 1, device) }
    }

    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        relu(x.clone() + self.c2.forward(relu(self.c1.forward(x))))
    }
}

/// GRU cell (Selected head query update, 032 §2.5 #4).
#[derive(Module, Debug)]
pub struct GruCell {
    x: Linear,
    h: Linear,
}

impl GruCell {
    pub fn new(d_in: usize, d: usize, device: &burn::tensor::Device) -> Self {
        Self { x: LinearConfig::new(d_in, 3 * d).init(device), h: LinearConfig::new(d, 3 * d).init(device) }
    }

    pub fn forward(&self, x: Tensor<2>, h: Tensor<2>) -> Tensor<2> {
        let gx = self.x.forward(x).chunk(3, 1);
        let gh = self.h.forward(h.clone()).chunk(3, 1);
        let r = sigmoid(gx[0].clone() + gh[0].clone());
        let z = sigmoid(gx[1].clone() + gh[1].clone());
        let cand = tanh(gx[2].clone() + r * gh[2].clone());
        z.clone() * h + (z.neg() + 1.0) * cand
    }
}

/// Multi-head attention with an additive logit bias and a key mask. Burn's built-in attention
/// takes no bias, which the distance bias (032 §2.1) and GTrXL relative positions need.
#[derive(Module, Debug)]
pub struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    #[module(skip)]
    heads: usize,
}

impl Attention {
    pub fn new(d: usize, heads: usize, device: &burn::tensor::Device) -> Self {
        assert!(d % heads == 0, "d_model must divide by heads");
        let l = || LinearConfig::new(d, d).init(device);
        Self { q: l(), k: l(), v: l(), o: l(), heads }
    }

    /// `x_q` [B,Tq,d], `x_kv` [B,Tk,d], `bias` [B or 1,H,Tq,Tk], `key_ok` [B,Tq or 1,Tk] (true = attend).
    pub fn forward(&self, x_q: Tensor<3>, x_kv: Tensor<3>, bias: Option<Tensor<4>>, key_ok: Tensor<3, Bool>) -> Tensor<3> {
        let [b, tq, d] = x_q.dims();
        let tk = x_kv.dims()[1];
        let (h, dh) = (self.heads, d / self.heads);
        let split = |t: Tensor<3>, len: usize| t.reshape([b, len, h, dh]).swap_dims(1, 2);
        let q = split(self.q.forward(x_q), tq);
        let k = split(self.k.forward(x_kv.clone()), tk);
        let v = split(self.v.forward(x_kv), tk);
        let mut scores = q.matmul(k.swap_dims(2, 3)) / (dh as f64).sqrt();
        if let Some(bias) = bias {
            scores = scores + bias;
        }
        let kq = key_ok.dims()[1];
        let mask = key_ok.bool_not().reshape([b, 1, kq, tk]).expand([b, h, tq, tk]);
        let attn = softmax(scores.mask_fill(mask, MASKED), 3);
        let out = attn.matmul(v).swap_dims(1, 2).reshape([b, tq, d]);
        self.o.forward(out)
    }
}

/// Pre-LN transformer feed-forward with GELU.
#[derive(Module, Debug)]
pub struct FeedForward {
    l1: Linear,
    l2: Linear,
    dropout: Dropout,
}

impl FeedForward {
    pub fn new(d: usize, ffn: usize, dropout: f64, device: &burn::tensor::Device) -> Self {
        Self {
            l1: LinearConfig::new(d, ffn).init(device),
            l2: LinearConfig::new(ffn, d).init(device),
            dropout: DropoutConfig::new(dropout).init(),
        }
    }

    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        self.l2.forward(self.dropout.forward(gelu(self.l1.forward(x))))
    }
}

pub fn layer_norm(d: usize, device: &burn::tensor::Device) -> LayerNorm {
    LayerNormConfig::new(d).init(device)
}

/// Mean of `x` [B,N,D] over rows where `mask` [B,N] is true (zero when none).
pub fn masked_mean(x: Tensor<3>, mask: Tensor<2, Bool>) -> Tensor<2> {
    let [b, n, _] = x.dims();
    let m = mask.float().reshape([b, n, 1]);
    let count = m.clone().sum_dim(1).clamp_min(1.0);
    let [_, _, d] = x.dims();
    (x * m).sum_dim(1).div(count).reshape([b, d])
}

/// Log-softmax over the last dim of [B,K], accumulated in f32 whatever the compute dtype (032-R8).
pub fn log_softmax_f32(logits: Tensor<2>) -> Tensor<2> {
    log_softmax(logits.cast(DType::F32), 1)
}

/// Masked log-softmax over the last dim of [B,K] in f32 (032-R4, 032-R8).
pub fn masked_log_softmax(logits: Tensor<2>, allowed: Tensor<2, Bool>) -> Tensor<2> {
    log_softmax(logits.cast(DType::F32).mask_fill(allowed.bool_not(), MASKED), 1)
}

/// f32 zeros [B] for log-prob / entropy accumulators.
pub fn zeros_f32(b: usize, device: &burn::tensor::Device) -> Tensor<1> {
    Tensor::<1>::zeros([b], device).cast(DType::F32)
}

/// Gumbel-max sample from log-probs [B,K] at temperature `temp` (032-R9). `temp == 0` is greedy.
/// Masked entries (-1e9) are never chosen because the Gumbel noise is bounded.
pub fn gumbel_argmax(logp: Tensor<2>, temp: f32) -> Tensor<1, Int> {
    let [b, k] = logp.dims();
    if temp <= 0.0 {
        return logp.argmax(1).reshape([b]);
    }
    let u = Tensor::<2>::random([b, k], Distribution::Uniform(1e-6, 1.0 - 1e-6), &logp.device()).cast(DType::F32);
    let g = u.log().neg().log().neg();
    (logp / temp as f64 + g).argmax(1).reshape([b])
}

/// log p(choice) from log-probs [B,K] and choice [B] -> [B].
pub fn pick(logp: Tensor<2>, choice: Tensor<1, Int>) -> Tensor<1> {
    let b = choice.dims()[0];
    logp.gather(1, choice.reshape([b, 1])).reshape([b])
}

/// Entropy of the distribution given by log-probs [B,K] -> [B]. Masked entries contribute 0.
pub fn entropy(logp: Tensor<2>) -> Tensor<1> {
    let b = logp.dims()[0];
    let p = logp.clone().exp();
    (p * logp).sum_dim(1).neg().reshape([b])
}

/// Bool [B] -> f32 0 / 1 (multiplies f32 log-probs).
pub fn flag(b: Tensor<1, Bool>) -> Tensor<1> {
    b.float().cast(DType::F32)
}
