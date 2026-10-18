//! 041-R3 / 042 optimizer table (G-D4, AMB-041-5): global L2-norm gradient clipping.
//!
//! All parameter gradients are treated as one vector; when its L2 norm exceeds `max_norm`, every
//! gradient is scaled by the same factor `max_norm / (norm + eps)`. Burn's own
//! `GradientClippingConfig::Norm` clips each parameter tensor separately, so the BC trainer and the
//! RL learner call this on the accumulated gradients (after the micro-lane sum, before the
//! optimizer step) and build their optimizers without Burn clipping.
//!
//! The norm and the factor stay on the device (no host read), so clipping adds no sync point.

use burn::module::{Module, ModuleVisitor, Param};
use burn::optim::GradientsParams;
use burn::tensor::{DType, FloatDType, Tensor};

/// Added to the norm before dividing (as torch `clip_grad_norm_`), so a zero gradient stays zero.
pub const CLIP_EPS: f64 = 1e-6;

/// Sum of squared gradient entries over all parameters of a module, in f32.
struct SquaredSum<'a> {
    grads: &'a GradientsParams,
    sum: Option<Tensor<1>>,
}

impl ModuleVisitor for SquaredSum<'_> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        let Some(g) = self.grads.get::<D>(param.id) else { return };
        let g = match g.dtype() {
            DType::F16 | DType::BF16 => g.cast(FloatDType::F32),
            _ => g,
        };
        let s = g.square().sum();
        self.sum = Some(match self.sum.take() {
            Some(acc) => acc + s,
            None => s,
        });
    }
}

/// Multiplies every gradient by `coef` ([1]).
struct Scale<'a> {
    grads: &'a mut GradientsParams,
    coef: Tensor<1>,
}

impl ModuleVisitor for Scale<'_> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        let Some(g) = self.grads.remove::<D>(param.id) else { return };
        let dtype = g.dtype();
        let c = self.coef.clone().unsqueeze::<D>();
        let scaled = match dtype {
            DType::F16 | DType::BF16 => (g.cast(FloatDType::F32) * c).cast(dtype),
            _ => g * c,
        };
        self.grads.register::<D>(param.id, scaled);
    }
}

/// Global L2 norm of `grads` over the parameters of `module` ([1], on the gradients' device).
/// `None` when no parameter has a gradient.
pub fn global_norm<M: Module>(module: &M, grads: &GradientsParams) -> Option<Tensor<1>> {
    let mut v = SquaredSum { grads, sum: None };
    module.visit(&mut v);
    v.sum.map(|s| s.sqrt())
}

/// 041-R3: scales all gradients by `min(1, max_norm / (norm + eps))` where `norm` is the global
/// L2 norm. Returns the clipped gradients and the norm before clipping ([1], not read back).
pub fn clip_global_norm<M: Module>(module: &M, mut grads: GradientsParams, max_norm: f32) -> (GradientsParams, Option<Tensor<1>>) {
    let Some(norm) = global_norm(module, &grads) else { return (grads, None) };
    let coef = (norm.clone().add_scalar(CLIP_EPS).recip() * max_norm as f64).clamp_max(1.0);
    let mut v = Scale { grads: &mut grads, coef };
    module.visit(&mut v);
    (grads, Some(norm))
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::module::Param;
    use burn::tensor::{Device, TensorData};

    /// Two parameters (a [2] vector and a [2, 2] matrix) and their gradients set by hand.
    #[derive(Module, Debug)]
    struct Two {
        a: Param<Tensor<1>>,
        b: Param<Tensor<2>>,
    }

    fn setup(ga: [f32; 2], gb: [f32; 4]) -> (Two, GradientsParams) {
        let d = Device::flex();
        let m = Two {
            a: Param::from_tensor(Tensor::<1>::zeros([2], &d)),
            b: Param::from_tensor(Tensor::<2>::zeros([2, 2], &d)),
        };
        let mut g = GradientsParams::new();
        g.register::<1>(m.a.id, Tensor::<1>::from_data(TensorData::new(ga.to_vec(), [2]), &d));
        g.register::<2>(m.b.id, Tensor::<2>::from_data(TensorData::new(gb.to_vec(), [2, 2]), &d));
        (m, g)
    }

    fn host<const D: usize>(t: Tensor<D>) -> Vec<f32> {
        t.into_data().convert::<f32>().try_to_vec::<f32>().unwrap()
    }

    fn flat(m: &Two, g: &GradientsParams) -> Vec<f32> {
        let mut v = host(g.get::<1>(m.a.id).unwrap());
        v.extend(host(g.get::<2>(m.b.id).unwrap()));
        v
    }

    fn l2(v: &[f32]) -> f64 {
        v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt()
    }

    /// (a) 041-R3: after clipping, the global norm is at most `max_norm` (one vector, not per tensor).
    #[test]
    fn clipped_norm_is_at_most_max_041_r3() {
        let (m, g) = setup([3.0, -4.0], [10.0, 0.5, -7.0, 2.0]);
        let (g, norm) = clip_global_norm(&m, g, 1.0);
        let before = host(norm.unwrap())[0] as f64;
        assert!((before - l2(&[3.0, -4.0, 10.0, 0.5, -7.0, 2.0])).abs() < 1e-4);
        let after = l2(&flat(&m, &g));
        assert!(after <= 1.0 + 1e-6, "norm after clip {after}");
        assert!(after > 1.0 - 1e-4, "clipped to the threshold, not below it: {after}");
    }

    /// (b) 041-R3: one common factor for all parameters, so the gradient direction is unchanged.
    #[test]
    fn clipping_keeps_the_direction_041_r3() {
        let raw = [3.0, -4.0, 10.0, 0.5, -7.0, 2.0];
        let (m, g) = setup([raw[0], raw[1]], [raw[2], raw[3], raw[4], raw[5]]);
        let (g, _) = clip_global_norm(&m, g, 0.5);
        let c = flat(&m, &g);
        let k = c[0] as f64 / raw[0] as f64;
        assert!(k > 0.0 && k < 1.0);
        for (x, r) in c.iter().zip(raw) {
            assert!((*x as f64 - k * r as f64).abs() < 1e-6, "{x} vs {k} x {r}");
        }
        // Per-tensor clipping would scale `a` (norm 5) and `b` (norm ~12.4) by different factors.
        let ka = l2(&c[..2]) / 5.0;
        let kb = l2(&c[2..]) / l2(&raw[2..]);
        assert!((ka - kb).abs() < 1e-6, "same factor for both tensors: {ka} vs {kb}");
    }

    /// (c) 041-R3: below the threshold the gradients are returned unchanged (bit for bit).
    #[test]
    fn below_threshold_is_a_no_op_041_r3() {
        let (m, g) = setup([0.1, -0.2], [0.3, 0.0, -0.4, 0.05]);
        let want = flat(&m, &g);
        let (g, norm) = clip_global_norm(&m, g, 1.0);
        assert!(host(norm.unwrap())[0] < 1.0);
        assert_eq!(flat(&m, &g).iter().map(|x| x.to_bits()).collect::<Vec<_>>(), want.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
    }

    /// (d) 041-R3 hand-computed example: grads (3, 4) and (0, 0, 0, 12) have global norm
    /// sqrt(9 + 16 + 144) = 13; clip 1.0 -> factor 1/13 -> (3/13, 4/13) and (0, 0, 0, 12/13).
    #[test]
    fn hand_computed_example_041_r3() {
        let (m, g) = setup([3.0, 4.0], [0.0, 0.0, 0.0, 12.0]);
        let (g, norm) = clip_global_norm(&m, g, 1.0);
        assert!((host(norm.unwrap())[0] - 13.0).abs() < 1e-5);
        let want = [3.0 / 13.0, 4.0 / 13.0, 0.0, 0.0, 0.0, 12.0 / 13.0];
        for (x, w) in flat(&m, &g).iter().zip(want) {
            assert!((*x as f64 - w).abs() < 1e-6, "{x} vs {w}");
        }
    }

    /// No gradients at all: nothing to clip.
    #[test]
    fn empty_gradients_pass_through() {
        let (m, _) = setup([0.0; 2], [0.0; 4]);
        let (g, norm) = clip_global_norm(&m, GradientsParams::new(), 1.0);
        assert!(norm.is_none() && g.is_empty());
    }
}
