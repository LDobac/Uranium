//! EntityEncoder (032 §2.1): embeddings + pre-LN transformer with a distance attention bias.

use burn::module::{Module, Param};
use burn::nn::{Embedding, EmbeddingConfig, LayerNorm, Linear, LinearConfig};
use burn::tensor::{Bool, Device, Distribution, Int, Tensor};

use crate::batch::ObsBatch;
use crate::config::{EntityConfig, InputDims};
use crate::layers::{Attention, FeedForward, layer_norm, masked_mean};

#[derive(Module, Debug)]
struct Layer {
    ln1: LayerNorm,
    attn: Attention,
    ln2: LayerNorm,
    ffn: FeedForward,
}

#[derive(Module, Debug)]
pub struct EntityEncoder {
    unit_type: Embedding,
    owner: Embedding,
    order: Embedding,
    last_action: Embedding,
    input: Linear,
    layers: Vec<Layer>,
    ln_out: LayerNorm,
    /// [buckets, layers*heads]: `bias[l, h, bucket]` (032 §2.1).
    bias_table: Param<Tensor<2>>,
    pool: Linear,
    #[module(skip)]
    heads: usize,
    #[module(skip)]
    dist_buckets: usize,
}

pub struct EntityOut {
    /// [B,N,d]
    pub e: Tensor<3>,
    /// [B,d]
    pub pool: Tensor<2>,
}

impl EntityEncoder {
    pub fn new(input: &InputDims, c: &EntityConfig, device: &Device) -> Self {
        let d = c.d_model;
        let emb = |n, d| EmbeddingConfig::new(n, d).init(device);
        let d_in = c.unit_type_emb + c.owner_emb + c.order_emb + c.last_action_emb + input.entity_cont;
        let layers = (0..c.layers)
            .map(|_| Layer {
                ln1: layer_norm(d, device),
                attn: Attention::new(d, c.heads, device),
                ln2: layer_norm(d, device),
                ffn: FeedForward::new(d, c.ffn, c.dropout, device),
            })
            .collect();
        let buckets = 2 * c.distance_buckets;
        let bias_table = Tensor::random([buckets, c.layers * c.heads], Distribution::Normal(0.0, 0.02), device);
        Self {
            unit_type: emb(input.unit_types, c.unit_type_emb),
            owner: emb(input.owners, c.owner_emb),
            order: emb(input.orders, c.order_emb),
            last_action: emb(input.actions, c.last_action_emb),
            input: LinearConfig::new(d_in, d).init(device),
            layers,
            ln_out: layer_norm(d, device),
            bias_table: Param::from_tensor(bias_table),
            pool: LinearConfig::new(2 * d, d).init(device),
            heads: c.heads,
            dist_buckets: c.distance_buckets,
        }
    }

    pub fn forward(&self, obs: &ObsBatch) -> EntityOut {
        let [b, n, _] = obs.entity_cat.dims();
        let cat = |i: usize| obs.entity_cat.clone().narrow(2, i, 1).reshape([b, n]);
        let x = Tensor::cat(
            vec![
                self.unit_type.forward(cat(0)),
                self.owner.forward(cat(1)),
                self.order.forward(cat(2)),
                self.last_action.forward(cat(3)),
                obs.entity_cont.clone(),
            ],
            2,
        );
        let mut x = self.input.forward(x);

        let bias = self.distance_bias(obs);
        let key_ok = obs.entity_mask.clone().reshape([b, 1, n]);
        for (l, layer) in self.layers.iter().enumerate() {
            let bias_l = bias.clone().narrow(1, l * self.heads, self.heads);
            let y = layer.ln1.forward(x.clone());
            x = x + layer.attn.forward(y.clone(), y, Some(bias_l), key_ok.clone());
            x = x.clone() + layer.ffn.forward(layer.ln2.forward(x));
        }
        let e = self.ln_out.forward(x);

        let own = obs.is_self().bool_and(obs.entity_mask.clone());
        let enemy = cat(1).equal_elem(1).bool_and(obs.entity_mask.clone());
        let pool = self.pool.forward(Tensor::cat(vec![masked_mean(e.clone(), own), masked_mean(e.clone(), enemy)], 1));
        EntityOut { e, pool }
    }

    /// Bias for every layer and head, [B, layers*heads, N, N]. 032-R13: computed as
    /// one-hot(bucket) @ table so the table gradient is a matmul, not a scatter-add.
    fn distance_bias(&self, obs: &ObsBatch) -> Tensor<4> {
        let [b, n, _] = obs.entity_pos.dims();
        let buckets = self.buckets(obs);
        let k = 2 * self.dist_buckets;
        let one_hot: Tensor<4> = buckets.one_hot::<4>(k).float();
        let lh = self.bias_table.dims()[1];
        one_hot.reshape([b * n * n, k]).matmul(self.bias_table.val()).reshape([b, n, n, lh]).permute([0, 3, 1, 2])
    }

    /// bucket(i,j) = same_owner * D + clamp(floor(log2(1 + pixel distance)), 0, D-1) (032 §2.1).
    fn buckets(&self, obs: &ObsBatch) -> Tensor<3, Int> {
        let [b, n, _] = obs.entity_pos.dims();
        let pos = obs.entity_pos.clone();
        let pi = pos.clone().reshape([b, n, 1, 2]);
        let pj = pos.reshape([b, 1, n, 2]);
        let dist = (pi - pj).powi_scalar(2).sum_dim(3).sqrt().reshape([b, n, n]);
        let log_d = (dist + 1.0).log() / std::f64::consts::LN_2;
        let dist_bucket = log_d.floor().clamp(0.0, (self.dist_buckets - 1) as f64).int();
        let owner = obs.entity_cat.clone().narrow(2, 1, 1).reshape([b, n]);
        let same: Tensor<3, Bool> =
            owner.clone().reshape([b, n, 1]).expand([b, n, n]).equal(owner.reshape([b, 1, n]).expand([b, n, n]));
        same.int() * self.dist_buckets as i64 + dist_bucket
    }
}
