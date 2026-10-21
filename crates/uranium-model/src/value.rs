//! ValueHeads (032 §2.6): win (tanh), build, units, value (033-R3). Privileged inputs enter here
//! only (032-R11); without them (inference, `act`) the privileged branch is a zero vector.

use burn::module::{Module, Param};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::{relu, tanh};
use burn::tensor::{Device, Tensor, TensorData};

use crate::batch::PrivBatch;
use crate::config::{InputDims, ValueConfig};
use crate::layers::Mlp;

pub const VALUE_WIN: usize = 0;
pub const VALUE_BUILD: usize = 1;
pub const VALUE_UNITS: usize = 2;
pub const VALUE_VALUE: usize = 3;

#[derive(Module, Debug)]
pub struct ValueHeads {
    privileged: Linear,
    shared: Mlp,
    out: Linear,
    /// 032-R18: win estimate without privileged input; `None` unless `value.win_open`.
    pub(crate) open: Option<WinOpenHead>,
    #[module(skip)]
    priv_dim: usize,
}

/// 032-R18: own MLP (two hidden layers, as `shared`) over detached `h`, z and race embeddings,
/// so no gradient reaches the policy, the encoders, the core or the other value heads.
#[derive(Module, Debug)]
pub struct WinOpenHead {
    l1: Linear,
    l2: Linear,
    out: Linear,
}

/// Seeds of the win_open layers' fixed generator (032-R18 ③).
const OPEN_SEEDS: [u64; 3] = [0x3218_0001, 0x3218_0002, 0x3218_0003];

/// 032-R18 ③: a Linear with Burn's default init distribution (U(-k, k), k = 1 / sqrt(d_in), for
/// weight and bias) drawn from its own fixed-seed generator instead of the device RNG. Burn
/// parameters initialise lazily from the device RNG on first use, so a draw here would move the
/// other modules' initial weights and every later dropout mask.
fn fixed_linear(d_in: usize, d_out: usize, seed: u64, device: &Device) -> Linear {
    let k = 1.0 / (d_in as f64).sqrt();
    let mut s = seed;
    // splitmix64 -> uniform [-k, k)
    let mut next = || {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (((z >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0) * k
    };
    let w: Vec<f32> = (0..d_in * d_out).map(|_| next() as f32).collect();
    let b: Vec<f32> = (0..d_out).map(|_| next() as f32).collect();
    Linear {
        weight: Param::from_tensor(Tensor::<2>::from_data(TensorData::new(w, [d_in, d_out]), device)),
        bias: Some(Param::from_tensor(Tensor::<1>::from_data(TensorData::new(b, [d_out]), device))),
    }
}

impl ValueHeads {
    pub fn new(input: &InputDims, c: &ValueConfig, d_core: usize, d_z: usize, d_race: usize, device: &Device) -> Self {
        let privileged = LinearConfig::new(input.privileged, c.privileged).init(device);
        let shared = Mlp::new(d_core + d_z + d_race + c.privileged, c.hidden, c.hidden, device);
        let out = LinearConfig::new(c.hidden, 4).init(device);
        // 032-R18 ③: no device RNG, the other weights and dropout draws stay as without it.
        let open = c.win_open.then(|| WinOpenHead {
            l1: fixed_linear(d_core + d_z + d_race, c.hidden, OPEN_SEEDS[0], device),
            l2: fixed_linear(c.hidden, c.hidden, OPEN_SEEDS[1], device),
            out: fixed_linear(c.hidden, 1, OPEN_SEEDS[2], device),
        });
        Self { privileged, shared, out, open, priv_dim: c.privileged }
    }

    /// 032-R18: [B] in [-1, 1], `None` without the head. Takes no privileged input.
    pub fn win_open(&self, h: Tensor<2>, z: Tensor<2>, race: Tensor<2>) -> Option<Tensor<1>> {
        let head = self.open.as_ref()?;
        let x = Tensor::cat(vec![h.detach(), z.detach(), race.detach()], 1);
        let b = x.dims()[0];
        let x = relu(head.l2.forward(relu(head.l1.forward(x))));
        Some(tanh(head.out.forward(x)).reshape([b]))
    }

    /// -> [B,4] in the order win, build, units, value.
    pub fn forward(&self, h: Tensor<2>, z: Tensor<2>, race: Tensor<2>, privileged: Option<&PrivBatch>) -> Tensor<2> {
        let b = h.dims()[0];
        let p = match privileged {
            Some(p) => relu(self.privileged.forward(p.summary.clone())),
            None => Tensor::zeros([b, self.priv_dim], &h.device()),
        };
        let x = self.out.forward(relu(self.shared.forward(Tensor::cat(vec![h, z, race, p], 1))));
        let win = tanh(x.clone().narrow(1, VALUE_WIN, 1));
        Tensor::cat(vec![win, x.narrow(1, VALUE_BUILD, 3)], 1)
    }
}
