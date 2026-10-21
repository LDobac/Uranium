//! ValueHeads (032 §2.6): win (tanh), build, units, value (033-R3). Privileged inputs enter here
//! only (032-R11); without them (inference, `act`) the privileged branch is a zero vector.

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::{relu, tanh};
use burn::tensor::{Device, Tensor};

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

/// 032-R18: own MLP over detached `h`, z and race embeddings, so no gradient reaches the policy,
/// the encoders, the core or the other value heads.
#[derive(Module, Debug)]
pub struct WinOpenHead {
    mlp: Mlp,
    out: Linear,
}

impl ValueHeads {
    pub fn new(input: &InputDims, c: &ValueConfig, d_core: usize, d_z: usize, d_race: usize, device: &Device) -> Self {
        let privileged = LinearConfig::new(input.privileged, c.privileged).init(device);
        let shared = Mlp::new(d_core + d_z + d_race + c.privileged, c.hidden, c.hidden, device);
        let out = LinearConfig::new(c.hidden, 4).init(device);
        // 032-R18 ③: initialised last of the whole model, the other weights stay as without it.
        let open = c.win_open.then(|| WinOpenHead {
            mlp: Mlp::new(d_core + d_z + d_race, c.hidden, c.hidden, device),
            out: LinearConfig::new(c.hidden, 1).init(device),
        });
        Self { privileged, shared, out, open, priv_dim: c.privileged }
    }

    /// 032-R18: [B] in [-1, 1], `None` without the head. Takes no privileged input.
    pub fn win_open(&self, h: Tensor<2>, z: Tensor<2>, race: Tensor<2>) -> Option<Tensor<1>> {
        let head = self.open.as_ref()?;
        let x = Tensor::cat(vec![h.detach(), z.detach(), race.detach()], 1);
        let b = x.dims()[0];
        Some(tanh(head.out.forward(relu(head.mlp.forward(x)))).reshape([b]))
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
