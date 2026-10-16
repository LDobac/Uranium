//! ScalarEncoder (032 §2.3): continuous scalars, categorical embeddings and the strategy
//! condition z (030 §5). `z_present == false` zeroes the z vector.

use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::tensor::activation::relu;
use burn::tensor::{Device, Int, Tensor};

use crate::batch::ObsBatch;
use crate::config::{InputDims, ScalarConfig};
use crate::layers::{Mlp, masked_mean};

#[derive(Module, Debug)]
pub struct ScalarEncoder {
    cont: Mlp,
    race: Embedding,
    map_family: Embedding,
    start_loc: Embedding,
    z_build: Embedding,
    z_build_pos: Embedding,
    z_units: Linear,
    z_out: Linear,
    out: Linear,
}

pub struct ScalarOut {
    /// [B,hidden] context `c` (also the ActionType GLU gate).
    pub c: Tensor<2>,
    /// [B,z_out] (value heads input, 032 §2.6).
    pub z: Tensor<2>,
    /// [B,race_emb] (value heads input).
    pub race: Tensor<2>,
}

impl ScalarEncoder {
    pub fn new(input: &InputDims, c: &ScalarConfig, device: &Device) -> Self {
        let emb = |n, d| EmbeddingConfig::new(n, d).init(device);
        let concat = c.hidden + c.race_emb + c.map_family_emb + c.start_loc_emb + c.z_out;
        Self {
            cont: Mlp::new(input.scalar, c.hidden, c.hidden, device),
            race: emb(input.races, c.race_emb),
            map_family: emb(input.map_families, c.map_family_emb),
            start_loc: emb(input.start_locs, c.start_loc_emb),
            z_build: emb(input.actions, c.z_build_emb),
            z_build_pos: emb(input.z_build_len, c.z_build_emb),
            z_units: LinearConfig::new(input.z_units, c.z_units_emb).init(device),
            z_out: LinearConfig::new(c.z_build_emb + c.z_units_emb, c.z_out).init(device),
            out: LinearConfig::new(concat, c.hidden).init(device),
        }
    }

    pub fn forward(&self, obs: &ObsBatch) -> ScalarOut {
        let [b, _] = obs.scalar.dims();
        let cat = |i: usize| obs.scalar_cat.clone().narrow(1, i, 1);
        let flat = |t: Tensor<3>| {
            let d = t.dims()[2];
            t.reshape([b, d])
        };
        let race = flat(self.race.forward(cat(0)));
        let map_family = flat(self.map_family.forward(cat(1)));
        let start_loc = flat(self.start_loc.forward(cat(2)));

        // z_build: embedding + position embedding, masked mean over non-padding (id 0 = padding).
        let len = obs.z_build.dims()[1];
        let pos = Tensor::<1, Int>::arange(0..len as i64, &obs.z_build.device()).reshape([1, len]).expand([b, len]);
        let zb = self.z_build.forward(obs.z_build.clone()) + self.z_build_pos.forward(pos);
        let zb = masked_mean(zb, obs.z_build.clone().not_equal_elem(0));
        let zu = self.z_units.forward(obs.z_units.clone());
        let present = obs.z_present.clone().float().reshape([b, 1]);
        let z = relu(self.z_out.forward(Tensor::cat(vec![zb, zu], 1))) * present;

        let cont = relu(self.cont.forward(obs.scalar.clone()));
        let c = relu(self.out.forward(Tensor::cat(vec![cont, race.clone(), map_family, start_loc, z.clone()], 1)));
        ScalarOut { c, z, race }
    }
}
