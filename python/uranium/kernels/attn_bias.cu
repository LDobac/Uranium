// Entity encoder attention with the distance bias, fused (032 §2.1, 032-R13; 060-R1: one source,
// built as CUDA on CUDA hosts and hipified on ROCm hosts; AMB-000-4 step 2).
//
//   s[i,j] = key_ok[j] ? q_i . k_j * scale + table[bucket[i,j], h] : -1e9   (masked_fill, 032-R4)
//   p = softmax_j(s),  out_i = sum_j p[i,j] v_j
//
// The same math as `uranium.model.layers.Attention` with the gathered bias, without the
// [B, H, N, N] bias / score / probability tensors. Forward keeps each row's log-sum-exp; the
// backward recomputes p. dQ and the table gradient come from a per-query pass, dK / dV from a
// per-key pass. The table gradient is summed per block in shared memory and written as block
// partials (summed by the caller): no global atomics, deterministic.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace {

constexpr int DH = 32;        // head dim (d_model 256 / 8 heads)
constexpr int BQ = 64;        // rows per block, one thread each
constexpr int BK = 64;        // tile of the other side kept in shared memory
constexpr int NB_MAX = 64;    // bias buckets (2 x distance_buckets = 32)
constexpr int BS = BK + 4;    // bucket tile row stride in bytes (odd word stride: no bank conflicts)
constexpr float MASKED = -1e9f;
// Every kernel runs BQ threads per block and says so (__launch_bounds__): without it HIP plans for
// 1024 threads and caps the registers, and the per-thread row arrays spill to scratch (Linux ROCm
// host, backward 64 rows x N256: 48 -> 10 ms).

template <typename T>
__device__ __forceinline__ float f(T x) { return static_cast<float>(x); }

// ---------------------------------------------------------------------------------------------
// Forward: one thread per query row, online softmax over key tiles.
template <typename scalar_t>
__global__ void __launch_bounds__(BQ) fwd_kernel(const scalar_t* __restrict__ q, const scalar_t* __restrict__ k, const scalar_t* __restrict__ v,
                           const uint8_t* __restrict__ bucket, const float* __restrict__ table, const bool* __restrict__ key_ok,
                           scalar_t* __restrict__ out, float* __restrict__ stats, int H, int N, int NB, float scale) {
  const int bh = blockIdx.y, b = bh / H, h = bh % H;
  const int i0 = blockIdx.x * BQ, i = i0 + threadIdx.x;
  const bool row = i < N;
  __shared__ float ks[BK][DH];
  __shared__ float vs[BK][DH];
  __shared__ float tab[NB_MAX];
  __shared__ bool ok[BK];
  __shared__ uint8_t bk[BQ * BS];
  for (int t = threadIdx.x; t < NB; t += BQ) tab[t] = table[t * H + h];
  const size_t base = (size_t)bh * N * DH;
  float qr[DH], o[DH];
  float m = -INFINITY, l = 0.f;
#pragma unroll
  for (int d = 0; d < DH; ++d) {
    qr[d] = row ? f(q[base + (size_t)i * DH + d]) : 0.f;
    o[d] = 0.f;
  }
  for (int j0 = 0; j0 < N; j0 += BK) {
    __syncthreads();
    for (int t = threadIdx.x; t < BK * DH; t += BQ) {
      const int jj = t / DH, d = t % DH, j = j0 + jj;
      ks[jj][d] = j < N ? f(k[base + (size_t)j * DH + d]) : 0.f;
      vs[jj][d] = j < N ? f(v[base + (size_t)j * DH + d]) : 0.f;
    }
    for (int t = threadIdx.x; t < BK; t += BQ) ok[t] = j0 + t < N && key_ok[(size_t)b * N + j0 + t];
    for (int t = threadIdx.x; t < BQ * BK; t += BQ) {
      const int ii = t / BK, jj = t % BK;
      bk[ii * BS + jj] = (i0 + ii < N && j0 + jj < N) ? bucket[((size_t)b * N + i0 + ii) * N + j0 + jj] : 0;
    }
    __syncthreads();
    if (!row) continue;
    const int jn = min(BK, N - j0);
    for (int jj = 0; jj < jn; ++jj) {
      float s = MASKED;
      if (ok[jj]) {
        float dot = 0.f;
#pragma unroll
        for (int d = 0; d < DH; ++d) dot += qr[d] * ks[jj][d];
        s = dot * scale + tab[bk[threadIdx.x * BS + jj]];
      }
      const float mn = fmaxf(m, s), corr = expf(m - mn), p = expf(s - mn);
      l = l * corr + p;
#pragma unroll
      for (int d = 0; d < DH; ++d) o[d] = o[d] * corr + p * vs[jj][d];
      m = mn;
    }
  }
  if (row) {
    const float inv = 1.f / l;
#pragma unroll
    for (int d = 0; d < DH; ++d) out[base + (size_t)i * DH + d] = static_cast<scalar_t>(o[d] * inv);
    stats[((size_t)bh * N + i) * 2] = m;  // row max and 1 / sum: p = exp(s - max) / sum, as softmax
    stats[((size_t)bh * N + i) * 2 + 1] = inv;
  }
}

// ---------------------------------------------------------------------------------------------
// Backward, per query row: dQ and the table gradient (block partials [grid.x, B*H, NB]).
// delta_i = sum_d dout_i * out_i.
template <typename scalar_t>
__global__ void __launch_bounds__(BQ) bwd_q_kernel(const scalar_t* __restrict__ q, const scalar_t* __restrict__ k, const scalar_t* __restrict__ v,
                             const uint8_t* __restrict__ bucket, const float* __restrict__ table, const bool* __restrict__ key_ok,
                             const scalar_t* __restrict__ dout, const float* __restrict__ stats, const float* __restrict__ delta,
                             scalar_t* __restrict__ dq, float* __restrict__ dtab_part, int H, int N, int NB, float scale) {
  const int bh = blockIdx.y, b = bh / H, h = bh % H;
  const int i0 = blockIdx.x * BQ, i = i0 + threadIdx.x;
  const bool row = i < N;
  __shared__ float ks[BK][DH];
  __shared__ float vs[BK][DH];
  __shared__ float tab[NB_MAX];
  __shared__ float hist[NB_MAX];
  __shared__ bool ok[BK];
  __shared__ uint8_t bk[BQ * BS];
  for (int t = threadIdx.x; t < NB; t += BQ) {
    tab[t] = table[t * H + h];
    hist[t] = 0.f;
  }
  const size_t base = (size_t)bh * N * DH;
  float qr[DH], dor[DH], dqa[DH];
#pragma unroll
  for (int d = 0; d < DH; ++d) {
    qr[d] = row ? f(q[base + (size_t)i * DH + d]) : 0.f;
    dor[d] = row ? f(dout[base + (size_t)i * DH + d]) : 0.f;
    dqa[d] = 0.f;
  }
  const float M = row ? stats[((size_t)bh * N + i) * 2] : 0.f, LI = row ? stats[((size_t)bh * N + i) * 2 + 1] : 0.f;
  const float D = row ? delta[(size_t)bh * N + i] : 0.f;
  for (int j0 = 0; j0 < N; j0 += BK) {
    __syncthreads();
    for (int t = threadIdx.x; t < BK * DH; t += BQ) {
      const int jj = t / DH, d = t % DH, j = j0 + jj;
      ks[jj][d] = j < N ? f(k[base + (size_t)j * DH + d]) : 0.f;
      vs[jj][d] = j < N ? f(v[base + (size_t)j * DH + d]) : 0.f;
    }
    for (int t = threadIdx.x; t < BK; t += BQ) ok[t] = j0 + t < N && key_ok[(size_t)b * N + j0 + t];
    for (int t = threadIdx.x; t < BQ * BK; t += BQ) {
      const int ii = t / BK, jj = t % BK;
      bk[ii * BS + jj] = (i0 + ii < N && j0 + jj < N) ? bucket[((size_t)b * N + i0 + ii) * N + j0 + jj] : 0;
    }
    __syncthreads();
    if (!row) continue;
    const int jn = min(BK, N - j0);
    for (int jj = 0; jj < jn; ++jj) {
      if (!ok[jj]) continue;  // masked_fill: no gradient reaches q, k or the bias there
      float dot = 0.f, dp = 0.f;
#pragma unroll
      for (int d = 0; d < DH; ++d) {
        dot += qr[d] * ks[jj][d];
        dp += dor[d] * vs[jj][d];
      }
      const int bu = bk[threadIdx.x * BS + jj];
      const float p = expf(dot * scale + tab[bu] - M) * LI;
      const float ds = p * (dp - D);
#pragma unroll
      for (int d = 0; d < DH; ++d) dqa[d] += ds * ks[jj][d];
      atomicAdd(&hist[bu], ds);
    }
  }
  if (row) {
#pragma unroll
    for (int d = 0; d < DH; ++d) dq[base + (size_t)i * DH + d] = static_cast<scalar_t>(dqa[d] * scale);
  }
  __syncthreads();
  for (int t = threadIdx.x; t < NB; t += BQ) dtab_part[((size_t)blockIdx.x * gridDim.y + bh) * NB + t] = hist[t];
}

// ---------------------------------------------------------------------------------------------
// Backward, per key row: dK and dV.
template <typename scalar_t>
__global__ void __launch_bounds__(BQ) bwd_kv_kernel(const scalar_t* __restrict__ q, const scalar_t* __restrict__ k, const scalar_t* __restrict__ v,
                              const uint8_t* __restrict__ bucket, const float* __restrict__ table, const bool* __restrict__ key_ok,
                              const scalar_t* __restrict__ dout, const float* __restrict__ stats, const float* __restrict__ delta,
                              scalar_t* __restrict__ dk, scalar_t* __restrict__ dv, int H, int N, int NB, float scale) {
  const int bh = blockIdx.y, b = bh / H, h = bh % H;
  const int j0 = blockIdx.x * BQ, j = j0 + threadIdx.x;
  const bool col = j < N;
  __shared__ float qs[BK][DH];
  __shared__ float dos[BK][DH];
  __shared__ float ms[BK];
  __shared__ float li[BK];
  __shared__ float ds_[BK];
  __shared__ float tab[NB_MAX];
  __shared__ uint8_t bk[BK * BQ];  // [query in tile][key in block]
  for (int t = threadIdx.x; t < NB; t += BQ) tab[t] = table[t * H + h];
  const size_t base = (size_t)bh * N * DH;
  const bool okj = col && key_ok[(size_t)b * N + j];
  float kr[DH], vr[DH], dka[DH], dva[DH];
#pragma unroll
  for (int d = 0; d < DH; ++d) {
    kr[d] = col ? f(k[base + (size_t)j * DH + d]) : 0.f;
    vr[d] = col ? f(v[base + (size_t)j * DH + d]) : 0.f;
    dka[d] = 0.f;
    dva[d] = 0.f;
  }
  for (int i0 = 0; i0 < N; i0 += BK) {
    __syncthreads();
    for (int t = threadIdx.x; t < BK * DH; t += BQ) {
      const int ii = t / DH, d = t % DH, i = i0 + ii;
      qs[ii][d] = i < N ? f(q[base + (size_t)i * DH + d]) : 0.f;
      dos[ii][d] = i < N ? f(dout[base + (size_t)i * DH + d]) : 0.f;
    }
    for (int t = threadIdx.x; t < BK; t += BQ) {
      const int i = i0 + t;
      ms[t] = i < N ? stats[((size_t)bh * N + i) * 2] : 0.f;
      li[t] = i < N ? stats[((size_t)bh * N + i) * 2 + 1] : 0.f;
      ds_[t] = i < N ? delta[(size_t)bh * N + i] : 0.f;
    }
    for (int t = threadIdx.x; t < BK * BQ; t += BQ) {
      const int ii = t / BQ, jj = t % BQ;
      bk[ii * BQ + jj] = (i0 + ii < N && j0 + jj < N) ? bucket[((size_t)b * N + i0 + ii) * N + j0 + jj] : 0;
    }
    __syncthreads();
    if (!col) continue;
    const int in = min(BK, N - i0);
    for (int ii = 0; ii < in; ++ii) {
      float s = MASKED, dp = 0.f;
      if (okj) {
        float dot = 0.f;
#pragma unroll
        for (int d = 0; d < DH; ++d) dot += qs[ii][d] * kr[d];
        s = dot * scale + tab[bk[ii * BQ + threadIdx.x]];
      }
      const float p = expf(s - ms[ii]) * li[ii];
#pragma unroll
      for (int d = 0; d < DH; ++d) {
        dva[d] += p * dos[ii][d];
        dp += dos[ii][d] * vr[d];
      }
      if (okj) {
        const float g = p * (dp - ds_[ii]);
#pragma unroll
        for (int d = 0; d < DH; ++d) dka[d] += g * qs[ii][d];
      }
    }
  }
  if (col) {
#pragma unroll
    for (int d = 0; d < DH; ++d) {
      dk[base + (size_t)j * DH + d] = static_cast<scalar_t>(dka[d] * scale);
      dv[base + (size_t)j * DH + d] = static_cast<scalar_t>(dva[d]);
    }
  }
}

void check(const torch::Tensor& q, const torch::Tensor& bucket, const torch::Tensor& table, const torch::Tensor& key_ok) {
  TORCH_CHECK(q.is_cuda() && q.is_contiguous() && q.dim() == 4 && q.size(3) == DH, "q: contiguous [B, H, N, 32] on the GPU");
  TORCH_CHECK(bucket.scalar_type() == torch::kUInt8 && bucket.is_contiguous() && bucket.dim() == 3, "bucket: contiguous uint8 [B, N, N]");
  TORCH_CHECK(table.scalar_type() == torch::kFloat && table.is_contiguous() && table.size(0) <= NB_MAX && table.size(1) == q.size(1), "table: f32 [NB <= 64, H]");
  TORCH_CHECK(key_ok.scalar_type() == torch::kBool && key_ok.is_contiguous() && key_ok.dim() == 2, "key_ok: contiguous bool [B, N]");
}

}  // namespace

std::vector<torch::Tensor> attn_bias_fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor bucket, torch::Tensor table, torch::Tensor key_ok, double scale) {
  check(q, bucket, table, key_ok);
  const c10::cuda::CUDAGuard guard(q.device());
  const int B = q.size(0), H = q.size(1), N = q.size(2), NB = table.size(0);
  auto out = torch::empty_like(q);
  auto stats = torch::empty({B, H, N, 2}, q.options().dtype(torch::kFloat));
  const dim3 grid((N + BQ - 1) / BQ, B * H);
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, q.scalar_type(), "attn_bias_fwd", [&] {
    fwd_kernel<scalar_t><<<grid, BQ, 0, stream>>>(q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(), bucket.data_ptr<uint8_t>(),
                                                  table.data_ptr<float>(), key_ok.data_ptr<bool>(), out.data_ptr<scalar_t>(), stats.data_ptr<float>(), H, N, NB, (float)scale);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, stats};
}

std::vector<torch::Tensor> attn_bias_bwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor bucket, torch::Tensor table, torch::Tensor key_ok,
                                         torch::Tensor dout, torch::Tensor stats, torch::Tensor delta, double scale) {
  check(q, bucket, table, key_ok);
  TORCH_CHECK(dout.is_contiguous() && dout.sizes() == q.sizes() && dout.scalar_type() == q.scalar_type(), "dout: like q");
  const c10::cuda::CUDAGuard guard(q.device());
  const int B = q.size(0), H = q.size(1), N = q.size(2), NB = table.size(0);
  auto dq = torch::empty_like(q), dk = torch::empty_like(q), dv = torch::empty_like(q);
  const dim3 grid((N + BQ - 1) / BQ, B * H);
  auto part = torch::empty({(long)grid.x, (long)B * H, NB}, q.options().dtype(torch::kFloat));
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, q.scalar_type(), "attn_bias_bwd", [&] {
    bwd_q_kernel<scalar_t><<<grid, BQ, 0, stream>>>(q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(), bucket.data_ptr<uint8_t>(),
                                                    table.data_ptr<float>(), key_ok.data_ptr<bool>(), dout.data_ptr<scalar_t>(), stats.data_ptr<float>(),
                                                    delta.data_ptr<float>(), dq.data_ptr<scalar_t>(), part.data_ptr<float>(), H, N, NB, (float)scale);
    bwd_kv_kernel<scalar_t><<<grid, BQ, 0, stream>>>(q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(), bucket.data_ptr<uint8_t>(),
                                                     table.data_ptr<float>(), key_ok.data_ptr<bool>(), dout.data_ptr<scalar_t>(), stats.data_ptr<float>(),
                                                     delta.data_ptr<float>(), dk.data_ptr<scalar_t>(), dv.data_ptr<scalar_t>(), H, N, NB, (float)scale);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  // [grid.x, B, H, NB] -> [NB, H]
  auto dtable = part.view({(long)grid.x, B, H, NB}).sum({0, 1}).t().contiguous();
  return {dq, dk, dv, dtable};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fwd", &attn_bias_fwd, "fused attention with the gathered distance bias, forward");
  m.def("bwd", &attn_bias_bwd, "fused attention with the gathered distance bias, backward");
}
