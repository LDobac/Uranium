use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

use burn::tensor::{Device, Int, Tensor, TensorData};

/// One benchmark result row.
#[derive(Debug, Default, Clone)]
pub struct Row {
    pub name: String,
    pub shape: String,
    pub fwd_ms: Option<f64>,
    pub fwd_bwd_ms: Option<f64>,
    pub gflops: Option<f64>,
    pub max_abs_err: Option<f64>,
    pub max_rel_err: Option<f64>,
    pub status: Status,
    pub note: String,
}

#[derive(Debug, Default, Clone, PartialEq)]
pub enum Status {
    #[default]
    Ok,
    /// Ran, but the accuracy check exceeded the tolerance.
    Inaccurate,
    /// Panicked / unsupported on this backend.
    Failed(String),
}

impl Row {
    pub fn new(name: &str, shape: impl Into<String>) -> Self {
        Self { name: name.to_string(), shape: shape.into(), ..Default::default() }
    }

    pub fn summary(&self) -> String {
        let f = |v: Option<f64>, unit: &str| v.map(|x| format!("{x:.3}{unit}")).unwrap_or_else(|| "-".into());
        format!(
            "status={:?} fwd={} fwd+bwd={} gflops={} abs_err={} rel_err={}",
            self.status,
            f(self.fwd_ms, "ms"),
            f(self.fwd_bwd_ms, "ms"),
            f(self.gflops, ""),
            self.max_abs_err.map(|x| format!("{x:.2e}")).unwrap_or_else(|| "-".into()),
            self.max_rel_err.map(|x| format!("{x:.2e}")).unwrap_or_else(|| "-".into()),
        )
    }

    /// Record accuracy against a reference and flag the row when above `tol` (relative).
    pub fn set_error(&mut self, got: &[f32], want: &[f32], tol: f64) {
        let (abs, rel) = max_errors(got, want);
        self.max_abs_err = Some(abs);
        self.max_rel_err = Some(rel);
        if rel > tol && self.status == Status::Ok {
            self.status = Status::Inaccurate;
            self.note = format!("rel err {rel:.2e} > tol {tol:.0e}");
        }
    }
}

pub fn run_guarded(name: &str, f: impl FnOnce() -> Row) -> Row {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(row) => row,
        Err(e) => {
            let msg = e
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "panic".into());
            let mut row = Row::new(name, "");
            row.status = Status::Failed(first_line(&msg));
            row
        }
    }
}

fn first_line(s: &str) -> String {
    let line = s.lines().next().unwrap_or("").trim();
    line.chars().take(160).collect()
}

/// Median wall time in milliseconds; `f` must block until the device has finished.
pub fn time_ms(warmup: usize, iters: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..warmup {
        f();
    }
    let mut samples: Vec<f64> = (0..iters)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

pub fn sync(device: &Device) {
    device.sync().expect("device sync failed");
}

/// Deterministic pseudo-random values in [-1, 1) (xorshift), identical on every backend.
pub fn values(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

pub fn float<const D: usize>(shape: [usize; D], seed: u64, device: &Device) -> Tensor<D> {
    let n = shape.iter().product();
    Tensor::from_data(TensorData::new(values(n, seed), shape), device)
}

pub fn ints<const D: usize>(shape: [usize; D], max: usize, seed: u64, device: &Device) -> Tensor<D, Int> {
    let n = shape.iter().product();
    let v: Vec<i32> = values(n, seed).into_iter().map(|x| (((x + 1.0) * 0.5 * max as f32) as i32).clamp(0, max as i32 - 1)).collect();
    Tensor::from_data(TensorData::new(v, shape), device)
}

pub fn to_vec<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    t.into_data().convert::<f32>().try_into_vec::<f32>().expect("tensor to vec")
}

pub fn max_errors(got: &[f32], want: &[f32]) -> (f64, f64) {
    assert_eq!(got.len(), want.len(), "accuracy check: length mismatch");
    let scale = want.iter().fold(0f64, |m, &w| m.max(w.abs() as f64)).max(1e-6);
    let abs = got.iter().zip(want).fold(0f64, |m, (&g, &w)| m.max((g as f64 - w as f64).abs()));
    (abs, abs / scale)
}
