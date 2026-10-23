//! 041-R3 / 042 optimizer table (G-D4, AMB-041-5): global L2-norm gradient clipping.
//!
//! All parameter gradients are treated as one vector; when its L2 norm exceeds `max_norm`, every
//! gradient is scaled by the same factor `max_norm / (norm + eps)`. Burn's own
//! `GradientClippingConfig::Norm` clips each parameter tensor separately, so the BC trainer and the
//! RL learner call this on the accumulated gradients (after the micro-lane sum, before the
//! optimizer step) and build their optimizers without Burn clipping.
//!
//! The norm and the factor stay on the device (no host read), so clipping adds no sync point.
//!
//! A non-finite global norm (NaN / inf anywhere in the gradients) sets every gradient to 0 for
//! that update, also on the device: the step then adds no gradient (AdamW moments still decay
//! and weight decay still applies), instead of writing NaN into the weights.

use burn::module::{Module, ModuleVisitor, Param, ParamId};
use burn::optim::GradientsParams;
use burn::tensor::{Bool, DType, FloatDType, Tensor};

/// The clip semantics in the BC config fingerprint (B-3), defined with the fingerprint.
pub use uranium_data::bc::CLIP_MODE;

/// Added to the norm before dividing (as torch `clip_grad_norm_`), so a zero gradient stays zero.
pub const CLIP_EPS: f64 = 1e-6;

/// Which parameters a visitor touches (041-R19: the win_open head is clipped apart).
#[derive(Clone, Copy)]
enum Pick<'a> {
    All,
    In(&'a [ParamId]),
    NotIn(&'a [ParamId]),
}

impl Pick<'_> {
    fn takes(self, id: ParamId) -> bool {
        match self {
            Pick::All => true,
            Pick::In(ids) => ids.contains(&id),
            Pick::NotIn(ids) => !ids.contains(&id),
        }
    }
}

/// Sum of squared gradient entries over the picked parameters of a module, in f32.
struct SquaredSum<'a> {
    grads: &'a GradientsParams,
    pick: Pick<'a>,
    sum: Option<Tensor<1>>,
}

impl ModuleVisitor for SquaredSum<'_> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        if !self.pick.takes(param.id) {
            return;
        }
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

/// Multiplies every picked gradient by `coef` ([1]); where `bad` ([1]) is true the gradient
/// becomes 0 (a NaN entry times 0 would stay NaN).
struct Scale<'a> {
    grads: &'a mut GradientsParams,
    pick: Pick<'a>,
    coef: Tensor<1>,
    bad: Tensor<1, Bool>,
}

impl ModuleVisitor for Scale<'_> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        if !self.pick.takes(param.id) {
            return;
        }
        let Some(g) = self.grads.remove::<D>(param.id) else { return };
        let dtype = g.dtype();
        let c = self.coef.clone().unsqueeze::<D>();
        let shape = g.shape();
        let scaled = match dtype {
            DType::F16 | DType::BF16 => (g.cast(FloatDType::F32) * c).cast(dtype),
            _ => g * c,
        };
        let scaled = scaled.mask_fill(self.bad.clone().unsqueeze::<D>().expand(shape), 0.0);
        self.grads.register::<D>(param.id, scaled);
    }
}

fn picked_norm<M: Module>(module: &M, grads: &GradientsParams, pick: Pick<'_>) -> Option<Tensor<1>> {
    let mut v = SquaredSum { grads, pick, sum: None };
    module.visit(&mut v);
    v.sum.map(|s| s.sqrt())
}

/// Global L2 norm of `grads` over the parameters of `module` ([1], on the gradients' device).
/// `None` when no parameter has a gradient.
pub fn global_norm<M: Module>(module: &M, grads: &GradientsParams) -> Option<Tensor<1>> {
    picked_norm(module, grads, Pick::All)
}

fn clip_picked<M: Module>(module: &M, mut grads: GradientsParams, max_norm: f32, pick: Pick<'_>) -> (GradientsParams, Option<Tensor<1>>) {
    let Some(norm) = picked_norm(module, &grads, pick) else { return (grads, None) };
    // Non-finite norm: factor 0 and the gradients masked to 0 (no host read, see module docs).
    let bad = norm.clone().is_finite().bool_not();
    let coef = (norm.clone().add_scalar(CLIP_EPS).recip() * max_norm as f64).clamp_max(1.0).mask_fill(bad.clone(), 0.0);
    let mut v = Scale { grads: &mut grads, pick, coef, bad };
    module.visit(&mut v);
    (grads, Some(norm))
}

/// 041-R3: scales all gradients by `min(1, max_norm / (norm + eps))` where `norm` is the global
/// L2 norm. Returns the clipped gradients and the norm before clipping ([1], not read back).
pub fn clip_global_norm<M: Module>(module: &M, grads: GradientsParams, max_norm: f32) -> (GradientsParams, Option<Tensor<1>>) {
    clip_picked(module, grads, max_norm, Pick::All)
}

/// 041-R19 / 042-R13: [`clip_global_norm`] over the parameters not in `apart`, then the same clip
/// over `apart` alone (the win_open head, 032-R18), so the head does not change the other
/// parameters' clip factor. Returns the norm of the rest. Empty `apart` = [`clip_global_norm`].
pub fn clip_global_norm_except<M: Module>(module: &M, grads: GradientsParams, max_norm: f32, apart: &[ParamId]) -> (GradientsParams, Option<Tensor<1>>) {
    if apart.is_empty() {
        return clip_global_norm(module, grads, max_norm);
    }
    let (grads, norm) = clip_picked(module, grads, max_norm, Pick::NotIn(apart));
    let (grads, _) = clip_picked(module, grads, max_norm, Pick::In(apart));
    (grads, norm)
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

    /// 041-R3: a NaN or inf anywhere makes the global norm non-finite; every gradient becomes 0
    /// (finite) so the step writes no NaN into the weights.
    #[test]
    fn non_finite_norm_zeroes_the_gradients_041_r3() {
        for bad in [f32::NAN, f32::INFINITY] {
            let (m, g) = setup([3.0, bad], [1.0, 2.0, 3.0, 4.0]);
            let (g, norm) = clip_global_norm(&m, g, 1.0);
            assert!(!host(norm.unwrap())[0].is_finite());
            assert_eq!(flat(&m, &g), vec![0.0; 6], "{bad}");
        }
    }

    /// No gradients at all: nothing to clip.
    #[test]
    fn empty_gradients_pass_through() {
        let (m, _) = setup([0.0; 2], [0.0; 4]);
        let (g, norm) = clip_global_norm(&m, GradientsParams::new(), 1.0);
        assert!(norm.is_none() && g.is_empty());
    }

    /// 041-R19: parameters clipped apart do not change the others' factor; each part is clipped
    /// to `max_norm` on its own; no `apart` is the plain global clip.
    #[test]
    fn apart_params_clip_on_their_own_041_r19() {
        let (m, g) = setup([3.0, 4.0], [0.0, 0.0, 0.0, 100.0]);
        let apart = [m.b.id];
        let (g, norm) = clip_global_norm_except(&m, g, 1.0, &apart);
        assert!((host(norm.unwrap())[0] - 5.0).abs() < 1e-5, "returned norm is the rest's");
        let a = host(g.get::<1>(m.a.id).unwrap());
        let b = host(g.get::<2>(m.b.id).unwrap());
        assert!((a[0] - 0.6).abs() < 1e-5 && (a[1] - 0.8).abs() < 1e-5, "{a:?}");
        assert!((l2(&b) - 1.0).abs() < 1e-5, "{b:?}");

        let (m, g) = setup([3.0, 4.0], [0.0, 0.0, 0.0, 100.0]);
        let plain = flat(&m, &clip_global_norm(&m, g, 1.0).0);
        let (m2, g2) = setup([3.0, 4.0], [0.0, 0.0, 0.0, 100.0]);
        let none = flat(&m2, &clip_global_norm_except(&m2, g2, 1.0, &[]).0);
        assert_eq!(plain, none);
    }
}
