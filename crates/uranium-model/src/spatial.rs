//! SpatialEncoder (032 §2.2): entity scatter-mean + stem + U-Net encoder with skips.

use burn::module::Module;
use burn::nn::conv::Conv2d;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::relu;
use burn::tensor::{Device, IndexingUpdateOp, Tensor};

use crate::batch::ObsBatch;
use crate::config::{InputDims, SpatialConfig};
use crate::layers::{ResBlock, conv3x3};

#[derive(Module, Debug)]
struct Down {
    conv: Conv2d,
    res: ResBlock,
}

#[derive(Module, Debug)]
pub struct SpatialEncoder {
    scatter: Linear,
    stem: Conv2d,
    downs: Vec<Down>,
    bottleneck: Vec<ResBlock>,
    pool: Linear,
}

pub struct SpatialOut {
    /// [B,C4,M/16,M/16]
    pub s8: Tensor<4>,
    /// Skips at M/2, M/4, M/8 (64², 32², 16² for M = 128).
    pub skips: [Tensor<4>; 3],
    /// [B,pool]
    pub pool: Tensor<2>,
}

impl SpatialEncoder {
    pub fn new(input: &InputDims, c: &SpatialConfig, d_entity: usize, device: &Device) -> Self {
        let mut c_in = c.stem_channels;
        let downs = c
            .stages
            .iter()
            .map(|&c_out| {
                let d = Down { conv: conv3x3(c_in, c_out, 2, device), res: ResBlock::new(c_out, device) };
                c_in = c_out;
                d
            })
            .collect();
        let side = input.map_size / 16;
        Self {
            scatter: LinearConfig::new(d_entity, c.scatter_channels).init(device),
            stem: conv3x3(input.spatial_channels + c.scatter_channels, c.stem_channels, 1, device),
            downs,
            bottleneck: (0..2).map(|_| ResBlock::new(c_in, device)).collect(),
            pool: LinearConfig::new(c_in * side * side, c.pool).init(device),
        }
    }

    pub fn forward(&self, obs: &ObsBatch, e: Tensor<3>) -> SpatialOut {
        let [_, _, h, w] = obs.spatial.dims();
        let scattered = self.scatter_mean(obs, e, h, w);
        let mut x = relu(self.stem.forward(Tensor::cat(vec![obs.spatial.clone(), scattered], 1)));
        let mut outs = Vec::with_capacity(4);
        for d in &self.downs {
            x = d.res.forward(relu(d.conv.forward(x)));
            outs.push(x.clone());
        }
        for r in &self.bottleneck {
            x = r.forward(x);
        }
        let pool = self.pool.forward(x.clone().flatten::<2>(1, 3));
        SpatialOut { s8: x, skips: [outs[0].clone(), outs[1].clone(), outs[2].clone()], pool }
    }

    /// Linear(E) scatter-added to `entity_cell` and divided by the per-cell count -> [B,c,H,W].
    fn scatter_mean(&self, obs: &ObsBatch, e: Tensor<3>, h: usize, w: usize) -> Tensor<4> {
        let [b, n, _] = e.dims();
        let m = obs.entity_mask.clone().float().reshape([b, n, 1]);
        let v = self.scatter.forward(e) * m.clone();
        let c = v.dims()[2];
        let device = v.device();
        let cells = obs.entity_cell.clone().reshape([b, n, 1]);
        let sum = Tensor::<3>::zeros([b, h * w, c], &device).scatter(1, cells.clone().expand([b, n, c]), v, IndexingUpdateOp::Add);
        let count = Tensor::<3>::zeros([b, h * w, 1], &device).scatter(1, cells, m, IndexingUpdateOp::Add);
        let mean = sum / count.clamp_min(1.0);
        mean.swap_dims(1, 2).reshape([b, c, h, w])
    }
}
