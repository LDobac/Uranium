//! Benchmark cases. Shapes follow spec 032 (B=16 inference batch, N=256 entities,
//! d=256, 128x128 spatial, LSTM 768->512, 12-step pointer, 16 delay buckets).

use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig};
use burn::nn::transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput};
use burn::nn::{Linear, LinearConfig, Lstm, LstmConfig, PaddingConfig2d};
use burn::tensor::activation::{relu, sigmoid, softmax, tanh};
use burn::tensor::{Bool, DType, Device, Distribution, IndexingUpdateOp, Int, Tensor, TensorData};

use crate::util::{Row, float, ints, sync, time_ms, to_vec};

pub struct Ctx {
    pub device: Device,
    /// Flex (CPU) device for accuracy checks (None = skip).
    pub reference: Option<Device>,
    pub quick: bool,
    pub iters: usize,
    pub warmup: usize,
    /// CUDA backend: accuracy rows use the CUDA bound of 032-AC1 (060-AC4).
    pub cuda: bool,
}

impl Ctx {
    fn ad(&self) -> Device {
        self.device.clone().autodiff()
    }
    /// 060-AC4 / 032-AC1: relative tolerance of a row. `strict` is the Vulkan/Flex f32 bound;
    /// on CUDA (TF32 matmul, 032-R8) the bound is at least 1e-2.
    fn tol(&self, strict: f64) -> f64 {
        if self.cuda { strict.max(1e-2) } else { strict }
    }
    fn pick(&self, full: usize, quick: usize) -> usize {
        if self.quick { quick } else { full }
    }
}

pub struct Case {
    pub name: &'static str,
    pub run: fn(&Ctx) -> Row,
}

pub fn all() -> Vec<Case> {
    vec![
        Case { name: "matmul_f32_1024", run: |c| matmul(c, 1024, false) },
        Case { name: "matmul_f32_2048", run: |c| matmul(c, 2048, false) },
        Case { name: "matmul_f32_4096", run: |c| matmul(c, 4096, false) },
        Case { name: "matmul_f16_1024", run: |c| matmul(c, 1024, true) },
        Case { name: "matmul_f16_2048", run: |c| matmul(c, 2048, true) },
        Case { name: "matmul_f16_4096", run: |c| matmul(c, 4096, true) },
        Case { name: "conv2d_3x3_128sq", run: conv2d_128 },
        Case { name: "conv_transpose_8_to_128", run: conv_transpose },
        Case { name: "lstm_seq_b32_t64", run: lstm_seq },
        Case { name: "lstm_step_b16", run: lstm_step },
        Case { name: "transformer_n256", run: |c| transformer(c, 256) },
        Case { name: "transformer_n512", run: |c| transformer(c, 512) },
        Case { name: "attention_distance_bias_n256", run: attention_bias },
        Case { name: "scatter_add_entities_to_128sq", run: scatter_add },
        Case { name: "gather_unit_cap_mask", run: gather_mask },
        Case { name: "pointer_gru_12step", run: pointer_gru },
        Case { name: "gumbel_max_128sq", run: gumbel_max },
        Case { name: "act_approx_b16_n256", run: act_approx },
    ]
    .into_iter()
    .chain(crate::net_cases::all())
    .collect()
}

// ---------------------------------------------------------------- matmul

fn matmul(c: &Ctx, n: usize, f16: bool) -> Row {
    let n = if c.quick { n.min(1024) } else { n };
    let mut row = Row::new(if f16 { "matmul_f16" } else { "matmul_f32" }, format!("{n}x{n} @ {n}x{n}"));
    let make = |dev: &Device, seed| {
        let t = float([n, n], seed, dev);
        if f16 { t.cast(DType::F16) } else { t }
    };
    let (a, b) = (make(&c.device, 1), make(&c.device, 2));
    row.fwd_ms = Some(time_ms(c.warmup, c.iters, || {
        let _ = a.clone().matmul(b.clone());
        sync(&c.device);
    }));
    row.gflops = Some(2.0 * (n as f64).powi(3) / (row.fwd_ms.unwrap() * 1e6));
    if let (Some(r), true) = (&c.reference, n <= 1024) {
        let got = to_vec(a.matmul(b).cast(DType::F32));
        let want = to_vec(float([n, n], 1, r).matmul(float([n, n], 2, r)));
        row.set_error(&got, &want, c.tol(if f16 { 1e-2 } else { 1e-4 }));
    }
    row
}

// ---------------------------------------------------------------- convolutions

fn conv2d_128(c: &Ctx) -> Row {
    let b = c.pick(16, 2);
    let mut row = Row::new("conv2d_3x3_128sq", format!("[{b},58,128,128] -> 32ch"));
    let cfg = Conv2dConfig::new([58, 32], [3, 3]).with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1));
    let conv = cfg.init(&c.device);
    let x = float([b, 58, 128, 128], 3, &c.device);
    row.fwd_ms = Some(time_ms(c.warmup, c.iters, || {
        let _ = conv.forward(x.clone());
        sync(&c.device);
    }));
    let ad = c.ad();
    let conv_ad = cfg.init(&ad);
    let x_ad = float([b, 58, 128, 128], 3, &ad);
    row.fwd_bwd_ms = Some(time_ms(c.warmup, c.iters, || {
        let _ = conv_ad.forward(x_ad.clone()).sum().backward();
        sync(&ad);
    }));
    let flops = 2.0 * (b * 32 * 128 * 128 * 58 * 9) as f64;
    row.gflops = Some(flops / (row.fwd_ms.unwrap() * 1e6));
    if let Some(r) = &c.reference {
        let m: Conv2d = cfg.init(r);
        let want = to_vec(m.forward(float([2, 58, 32, 32], 4, r)));
        let got = to_vec(m.to_device(&c.device).forward(float([2, 58, 32, 32], 4, &c.device)));
        row.set_error(&got, &want, c.tol(1e-4));
    }
    row
}

/// LocationHead decoder: 8x8 -> 128x128 via four stride-2 transposed convs (spec 032 §2.5).
struct Decoder {
    ups: Vec<ConvTranspose2d>,
    out: Conv2d,
}

impl Decoder {
    fn new(device: &Device) -> Self {
        let chans = [(144, 128), (128, 64), (64, 32), (32, 32)];
        let ups = chans
            .iter()
            .map(|&(i, o)| ConvTranspose2dConfig::new([i, o], [4, 4]).with_stride([2, 2]).with_padding([1, 1]).init(device))
            .collect();
        let out = Conv2dConfig::new([32, 1], [1, 1]).init(device);
        Self { ups, out }
    }
    fn forward(&self, mut x: Tensor<4>) -> Tensor<4> {
        for up in &self.ups {
            x = relu(up.forward(x));
        }
        self.out.forward(x)
    }
    fn to_device(self, device: &Device) -> Self {
        Self { ups: self.ups.into_iter().map(|u| u.to_device(device)).collect(), out: self.out.to_device(device) }
    }
}

fn conv_transpose(c: &Ctx) -> Row {
    let b = c.pick(16, 2);
    let mut row = Row::new("conv_transpose_8_to_128", format!("[{b},144,8,8] -> [{b},1,128,128]"));
    let dec = Decoder::new(&c.device);
    let x = float([b, 144, 8, 8], 5, &c.device);
    row.fwd_ms = Some(time_ms(c.warmup, c.iters, || {
        let _ = dec.forward(x.clone());
        sync(&c.device);
    }));
    let ad = c.ad();
    let dec_ad = Decoder::new(&ad);
    let x_ad = float([b, 144, 8, 8], 5, &ad);
    row.fwd_bwd_ms = Some(time_ms(c.warmup, c.iters, || {
        let _ = dec_ad.forward(x_ad.clone()).sum().backward();
        sync(&ad);
    }));
    if let Some(r) = &c.reference {
        let m = Decoder::new(r);
        let want = to_vec(m.forward(float([2, 144, 8, 8], 6, r)));
        let got = to_vec(m.to_device(&c.device).forward(float([2, 144, 8, 8], 6, &c.device)));
        row.set_error(&got, &want, c.tol(1e-4));
    }
    row
}

// ---------------------------------------------------------------- recurrent core

fn lstm_seq(c: &Ctx) -> Row {
    let (b, t) = (c.pick(32, 4), c.pick(64, 8));
    let mut row = Row::new("lstm_seq_b32_t64", format!("[{b},{t},768] -> 512"));
    let cfg = LstmConfig::new(768, 512, true);
    let lstm = cfg.init(&c.device);
    let x = float([b, t, 768], 7, &c.device);
    row.fwd_ms = Some(time_ms(c.warmup, c.iters, || {
        let _ = lstm.forward(x.clone(), None);
        sync(&c.device);
    }));
    let ad = c.ad();
    let lstm_ad = cfg.init(&ad);
    let x_ad = float([b, t, 768], 7, &ad);
    row.fwd_bwd_ms = Some(time_ms(c.warmup, c.iters, || {
        let (out, _) = lstm_ad.forward(x_ad.clone(), None);
        let _ = out.sum().backward();
        sync(&ad);
    }));
    if let Some(r) = &c.reference {
        let m: Lstm = cfg.init(r);
        let want = to_vec(m.forward(float([2, 8, 768], 8, r), None).0);
        let got = to_vec(m.to_device(&c.device).forward(float([2, 8, 768], 8, &c.device), None).0);
        row.set_error(&got, &want, c.tol(1e-4));
    }
    row
}

fn lstm_step(c: &Ctx) -> Row {
    let b = 16;
    let mut row = Row::new("lstm_step_b16", format!("[{b},1,768] -> 512 (inference step)"));
    let lstm = LstmConfig::new(768, 512, true).init(&c.device);
    let x = float([b, 1, 768], 9, &c.device);
    row.fwd_ms = Some(time_ms(c.warmup, c.iters, || {
        let _ = lstm.forward(x.clone(), None);
        sync(&c.device);
    }));
    row
}

// ---------------------------------------------------------------- entity encoder

fn transformer_cfg() -> TransformerEncoderConfig {
    TransformerEncoderConfig::new(256, 1024, 8, 3).with_norm_first(true).with_dropout(0.0)
}

fn pad_mask(b: usize, n: usize, device: &Device) -> Tensor<2, Bool> {
    // last quarter of each row is padding
    let v: Vec<bool> = (0..b * n).map(|i| i % n >= n * 3 / 4).collect();
    Tensor::from_data(TensorData::new(v, [b, n]), device)
}

fn transformer(c: &Ctx, n: usize) -> Row {
    let (b, n) = (c.pick(16, 2), if c.quick { n.min(128) } else { n });
    let mut row = Row::new(if n >= 512 { "transformer_n512" } else { "transformer_n256" }, format!("[{b},{n},256] L3 h8 ff1024"));
    let enc = transformer_cfg().init(&c.device);
    let x = float([b, n, 256], 10, &c.device);
    let mask = pad_mask(b, n, &c.device);
    row.fwd_ms = Some(time_ms(c.warmup, c.iters, || {
        let _ = enc.forward(TransformerEncoderInput::new(x.clone()).mask_pad(mask.clone()));
        sync(&c.device);
    }));
    let ad = c.ad();
    let enc_ad = transformer_cfg().init(&ad);
    let x_ad = float([b, n, 256], 10, &ad);
    let mask_ad = pad_mask(b, n, &ad);
    row.fwd_bwd_ms = Some(time_ms(c.warmup, c.iters, || {
        let _ = enc_ad.forward(TransformerEncoderInput::new(x_ad.clone()).mask_pad(mask_ad.clone())).sum().backward();
        sync(&ad);
    }));
    if let Some(r) = &c.reference {
        let m: TransformerEncoder = transformer_cfg().init(r);
        let want = to_vec(m.forward(TransformerEncoderInput::new(float([2, 32, 256], 11, r)).mask_pad(pad_mask(2, 32, r))));
        let got = to_vec(
            m.to_device(&c.device)
                .forward(TransformerEncoderInput::new(float([2, 32, 256], 11, &c.device)).mask_pad(pad_mask(2, 32, &c.device))),
        );
        row.set_error(&got, &want, c.tol(1e-3));
    }
    row
}

/// Attention with a learned per-head bias indexed by pairwise distance bucket (spec 032 §2.1).
fn attn_with_bias(q: Tensor<4>, k: Tensor<4>, v: Tensor<4>, table: Tensor<2>, buckets: Tensor<1, Int>) -> Tensor<4> {
    let [b, h, n, d] = q.dims();
    let bias = table.select(1, buckets).reshape([h, b, n, n]).swap_dims(0, 1);
    let scores = q.matmul(k.swap_dims(2, 3)) / (d as f64).sqrt() + bias;
    softmax(scores, 3).matmul(v)
}

fn attention_bias(c: &Ctx) -> Row {
    let (b, n) = (c.pick(16, 2), c.pick(256, 64));
    let mut row = Row::new("attention_distance_bias_n256", format!("[{b},8,{n},32] + bias[8,32][bucket]"));
    let mk = |dev: &Device, b: usize, n: usize| {
        (
            float([b, 8, n, 32], 12, dev),
            float([b, 8, n, 32], 13, dev),
            float([b, 8, n, 32], 14, dev),
            float([8, 32], 15, dev),
            ints([b * n * n], 32, 16, dev),
        )
    };
    let (q, k, v, t, idx) = mk(&c.device, b, n);
    row.fwd_ms = Some(time_ms(c.warmup, c.iters, || {
        let _ = attn_with_bias(q.clone(), k.clone(), v.clone(), t.clone(), idx.clone());
        sync(&c.device);
    }));
    let ad = c.ad();
    let (q2, k2, v2, t2, idx2) = mk(&ad, b, n);
    let (q2, t2) = (q2.require_grad(), t2.require_grad());
    row.fwd_bwd_ms = Some(time_ms(c.warmup, c.iters, || {
        let _ = attn_with_bias(q2.clone(), k2.clone(), v2.clone(), t2.clone(), idx2.clone()).sum().backward();
        sync(&ad);
    }));
    if let Some(r) = &c.reference {
        let (q, k, v, t, i) = mk(r, 2, 32);
        let want = to_vec(attn_with_bias(q, k, v, t, i));
        let (q, k, v, t, i) = mk(&c.device, 2, 32);
        row.set_error(&to_vec(attn_with_bias(q, k, v, t, i)), &want, c.tol(1e-4));
    }
    row
}

// ---------------------------------------------------------------- scatter / gather / masks

fn scatter_entities(values: Tensor<3>, cells: Tensor<2, Int>) -> Tensor<3> {
    let [b, n, ch] = values.dims();
    let idx = cells.reshape([b, n, 1]).expand([b, n, ch]);
    Tensor::zeros([b, 128 * 128, ch], &values.device()).scatter(1, idx, values, IndexingUpdateOp::Add)
}

fn scatter_add(c: &Ctx) -> Row {
    let (b, n) = (c.pick(16, 2), 256);
    let mut row = Row::new("scatter_add_entities_to_128sq", format!("[{b},{n},32] -> [{b},16384,32]"));
    let v = float([b, n, 32], 17, &c.device);
    let cells = ints([b, n], 128 * 128, 18, &c.device);
    row.fwd_ms = Some(time_ms(c.warmup, c.iters, || {
        let _ = scatter_entities(v.clone(), cells.clone());
        sync(&c.device);
    }));
    let ad = c.ad();
    let v2 = float([b, n, 32], 17, &ad).require_grad();
    let cells2 = ints([b, n], 128 * 128, 18, &ad);
    row.fwd_bwd_ms = Some(time_ms(c.warmup, c.iters, || {
        let _ = scatter_entities(v2.clone(), cells2.clone()).sum().backward();
        sync(&ad);
    }));
    if let Some(r) = &c.reference {
        // few cells so that additions collide
        let want = to_vec(scatter_entities(float([2, 64, 32], 19, r), ints([2, 64], 16, 20, r)));
        let got = to_vec(scatter_entities(float([2, 64, 32], 19, &c.device), ints([2, 64], 16, 20, &c.device)));
        row.set_error(&got, &want, c.tol(1e-5));
    }
    row
}

fn gather_mask(c: &Ctx) -> Row {
    let (b, n, k) = (c.pick(16, 2), 256, 48);
    let mut row = Row::new("gather_unit_cap_mask", format!("unit_cap[{b},{n},{k}] gather by action"));
    let run = |dev: &Device| {
        let cap = float([b, n, k], 21, dev);
        let sel = ints([b, 1, 1], k, 22, dev).expand([b, n, 1]);
        cap.gather(2, sel)
    };
    let cap = float([b, n, k], 21, &c.device);
    let sel = ints([b, 1, 1], k, 22, &c.device).expand([b, n, 1]);
    row.fwd_ms = Some(time_ms(c.warmup, c.iters, || {
        let _ = cap.clone().gather(2, sel.clone());
        sync(&c.device);
    }));
    if let Some(r) = &c.reference {
        row.set_error(&to_vec(run(&c.device)), &to_vec(run(r)), 0.0);
    }
    row
}

// ---------------------------------------------------------------- heads

struct GruCell {
    x: Linear,
    h: Linear,
}

impl GruCell {
    fn new(d: usize, device: &Device) -> Self {
        Self { x: LinearConfig::new(d, 3 * d).init(device), h: LinearConfig::new(d, 3 * d).init(device) }
    }
    fn forward(&self, x: Tensor<2>, h: Tensor<2>) -> Tensor<2> {
        let d = h.dims()[1];
        let gx = self.x.forward(x).chunk(3, 1);
        let gh = self.h.forward(h.clone()).chunk(3, 1);
        let r = sigmoid(gx[0].clone() + gh[0].clone());
        let z = sigmoid(gx[1].clone() + gh[1].clone());
        let cand = tanh(gx[2].clone() + r * gh[2].clone());
        let _ = d;
        z.clone() * h + (z.neg() + 1.0) * cand
    }
}

/// SelectedUnits head approximation: 12 autoregressive pointer steps over N keys (spec 032 §2.5 #4).
fn pointer_steps(cell: &GruCell, keys: Tensor<3>, mut q: Tensor<2>, steps: usize) -> Tensor<2> {
    let [b, n, d] = keys.dims();
    let mut acc = Tensor::zeros([b, n], &keys.device());
    for _ in 0..steps {
        let logits = keys.clone().matmul(q.clone().reshape([b, d, 1])).reshape([b, n]);
        let p = softmax(logits, 1);
        let chosen = p.clone().reshape([b, 1, n]).matmul(keys.clone()).reshape([b, d]);
        q = cell.forward(chosen, q);
        acc = acc + p;
    }
    acc
}

fn pointer_gru(c: &Ctx) -> Row {
    let (b, n) = (c.pick(16, 2), 256);
    let mut row = Row::new("pointer_gru_12step", format!("keys[{b},{n},64], 12 steps"));
    let cell = GruCell::new(64, &c.device);
    let keys = float([b, n, 64], 23, &c.device);
    let q = float([b, 64], 24, &c.device);
    row.fwd_ms = Some(time_ms(c.warmup, c.iters, || {
        let _ = pointer_steps(&cell, keys.clone(), q.clone(), 12);
        sync(&c.device);
    }));
    let ad = c.ad();
    let cell2 = GruCell::new(64, &ad);
    let keys2 = float([b, n, 64], 23, &ad).require_grad();
    let q2 = float([b, 64], 24, &ad);
    row.fwd_bwd_ms = Some(time_ms(c.warmup, c.iters, || {
        let _ = pointer_steps(&cell2, keys2.clone(), q2.clone(), 12).sum().backward();
        sync(&ad);
    }));
    if let Some(r) = &c.reference {
        let cr = GruCell::new(64, r);
        let want = to_vec(pointer_steps(&cr, float([2, 32, 64], 25, r), float([2, 64], 26, r), 12));
        let cg = GruCell { x: cr.x.to_device(&c.device), h: cr.h.to_device(&c.device) };
        let got = to_vec(pointer_steps(&cg, float([2, 32, 64], 25, &c.device), float([2, 64], 26, &c.device), 12));
        row.set_error(&got, &want, c.tol(1e-4));
    }
    row
}

fn gumbel_max(c: &Ctx) -> Row {
    let b = c.pick(16, 2);
    let mut row = Row::new("gumbel_max_128sq", format!("logits[{b},16384] -> argmax"));
    let logits = float([b, 128 * 128], 27, &c.device);
    row.fwd_ms = Some(time_ms(c.warmup, c.iters, || {
        let u = Tensor::<2>::random([b, 128 * 128], Distribution::Uniform(1e-6, 1.0), &c.device);
        let g = u.log().neg().log().neg();
        let _ = (logits.clone() + g).argmax(1).into_data();
    }));
    row
}

// ---------------------------------------------------------------- composite

/// Rough `act()` cost (spec 032-AC2): entity encoder + spatial encoder + LSTM step + location decoder.
fn act_approx(c: &Ctx) -> Row {
    let (b, n) = (c.pick(16, 2), 256);
    let mut row = Row::new("act_approx_b16_n256", format!("B={b} N={n}, encoders + core step + loc head"));
    let d = &c.device;
    let enc = transformer_cfg().init(d);
    let stem = Conv2dConfig::new([58, 32], [3, 3]).with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1)).init(d);
    let downs: Vec<Conv2d> = [(32, 64), (64, 64), (64, 128), (128, 128)]
        .iter()
        .map(|&(i, o)| Conv2dConfig::new([i, o], [3, 3]).with_stride([2, 2]).with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1)).init(d))
        .collect();
    let core = LstmConfig::new(768, 512, true).init(d);
    let dec = Decoder::new(d);
    let ents = float([b, n, 256], 30, d);
    let mask = pad_mask(b, n, d);
    let spatial = float([b, 58, 128, 128], 31, d);
    let core_in = float([b, 1, 768], 32, d);
    let dec_in = float([b, 144, 8, 8], 33, d);
    row.fwd_ms = Some(time_ms(c.warmup, c.iters, || {
        let _ = enc.forward(TransformerEncoderInput::new(ents.clone()).mask_pad(mask.clone()));
        let mut x = relu(stem.forward(spatial.clone()));
        for down in &downs {
            x = relu(down.forward(x));
        }
        let _ = core.forward(core_in.clone(), None);
        let _ = dec.forward(dec_in.clone());
        sync(d);
    }));
    row.note = "no heads 1-5, no masking".into();
    row
}
