"""
Fused causal EMA + KdV correction — Triton implementation.

Replaces the truncated FIR conv1d approach in _causal_ema() with an exact
causal IIR recurrence, fusing the KdV soliton correction in the same pass.

Operation
---------
Forward:
  EMA:  h[t] = alpha * x[t] + (1-alpha) * h[t-1]        (exact causal IIR)
  KdV:  out[t] = h[t] + kdv_alpha * h[t] * (h[t] - h[t-1])

Complexity
----------
  Old (truncated FIR conv1d):  O(N × k) where k=256
  New (Triton IIR):             O(N)  — exactly 256× fewer FMAs

Parallelism strategy
--------------------
  Sequential axis:  N  (causal dependency — unavoidable)
  Parallel axis:    B × D  (fully independent — one CUDA block per (b,d) pair)

  At 14M scale (B=8, D=256): 2048 independent programs × 2048 sequential steps
  At 35M scale (B=8, D=512): 4096 independent programs × 2048 sequential steps

Memory layout
-------------
  x, out: [B, N, D] — strides (N*D, D, 1)
  Inner loop walks stride-D through memory for a single (b,d) pair:
  x[b, t, d] = x_ptr + b*(N*D) + t*D + d

AGC note
--------
  AGC (RMS normalization) requires a two-pass algorithm (compute RMS across N,
  then normalize). It is NOT fused here — call _agc_normalize() on the output.
  If the extra memory round-trip proves costly, a two-pass Triton AGC kernel
  can be added later.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Forward kernel
# ---------------------------------------------------------------------------

@triton.jit
def _ema_kdv_fwd_kernel(
    x_ptr,      # [B, N, D]  input (bfloat16 or float16)
    out_ptr,    # [B, N, D]  output (same dtype)
    alpha,      # float32 scalar — EMA decay rate (learned, clamped 0.005..0.5)
    kdv_alpha,  # float32 scalar — KdV correction factor (can be 0 → pure EMA)
    B,          # int — batch size
    N,          # int — sequence length
    D,          # int — embedding dim
    stride_B: tl.constexpr,   # N*D  (stride between batches)
    stride_N: tl.constexpr,   # D    (stride between time steps)
    stride_D: tl.constexpr,   # 1    (stride between channels)
):
    """One Triton program per (batch_idx, dim_idx) pair."""
    pid = tl.program_id(0)
    b   = pid // D
    d   = pid  % D

    # Base pointer for this (b, d) slice
    base = b * stride_B + d * stride_D  # x[b, 0, d]

    # Accumulator (float32 for numerical precision during the recurrence)
    carry     = tl.zeros([], dtype=tl.float32)
    one_minus = 1.0 - alpha

    for t in range(N):
        ptr = base + t * stride_N          # x[b, t, d]
        x_t = tl.load(x_ptr + ptr).to(tl.float32)

        # EMA step
        h_new = alpha * x_t + one_minus * carry

        # KdV soliton correction: amplify rising fronts, damp falling flanks
        # out[t] = h[t] + kdv_alpha * h[t] * (h[t] - h[t-1])
        delta = h_new - carry
        out_t = h_new + kdv_alpha * h_new * delta

        # Store
        tl.store(out_ptr + ptr, out_t.to(x_ptr.dtype.element_ty))

        carry = h_new   # advance state (use h_new, not out_t, for next carry)


# ---------------------------------------------------------------------------
# Backward kernel  (gradients w.r.t. x, alpha, kdv_alpha)
# ---------------------------------------------------------------------------

@triton.jit
def _ema_kdv_bwd_kernel(
    x_ptr,       # [B, N, D]  input
    h_ptr,       # [B, N, D]  saved EMA states (h[t], not out[t])
    grad_out_ptr,# [B, N, D]  upstream gradient d_loss/d_out
    grad_x_ptr,  # [B, N, D]  output: d_loss/d_x
    # scalar grads accumulated per-thread — reduced in Python after kernel
    alpha,
    kdv_alpha,
    B, N, D,
    stride_B: tl.constexpr,
    stride_N: tl.constexpr,
    stride_D: tl.constexpr,
):
    """
    Reverse-mode autodiff for EMA + KdV, computed in reverse-time order.

    State diagram per timestep:
      h[t]   = alpha * x[t] + (1-alpha) * h[t-1]    (EMA)
      out[t] = h[t] + kdv_alpha * h[t] * (h[t] - h[t-1])

    Gradient equations (let go = grad_out[t]):
      d_out_t / d_h[t]   = 1 + kdv_alpha*(2*h[t] - h[t-1])
      d_out_t / d_h[t-1] = -kdv_alpha * h[t]
      d_h[t]  / d_x[t]   = alpha
      d_h[t]  / d_h[t-1] = 1 - alpha
    """
    pid = tl.program_id(0)
    b   = pid // D
    d   = pid  % D
    base = b * stride_B + d * stride_D

    # Reverse-time accumulator for gradient flowing back through h
    dL_dh_next = tl.zeros([], dtype=tl.float32)   # d_loss/d_h[t] from future steps

    for t_rev in range(N):
        t = N - 1 - t_rev
        ptr = base + t * stride_N

        go   = tl.load(grad_out_ptr + ptr).to(tl.float32)  # d_loss / d_out[t]
        h_t  = tl.load(h_ptr + ptr).to(tl.float32)

        h_prev = tl.zeros([], dtype=tl.float32)
        if t > 0:
            h_prev = tl.load(h_ptr + base + (t-1)*stride_N).to(tl.float32)

        # Gradient through out[t] = h[t] + kdv*(h[t]*(h[t]-h_prev))
        # d_out/d_h[t]   = 1 + kdv*(2*h[t] - h_prev)
        # d_out/d_h[prev]= -kdv * h[t]
        d_out_d_h    = 1.0 + kdv_alpha * (2.0 * h_t - h_prev)
        d_out_d_hprev = -kdv_alpha * h_t

        dL_dh_t = go * d_out_d_h + dL_dh_next

        # Gradient through h[t] = alpha*x[t] + (1-alpha)*h[prev]
        dL_dx_t = dL_dh_t * alpha
        tl.store(grad_x_ptr + ptr, dL_dx_t.to(grad_x_ptr.dtype.element_ty))

        # Propagate to h[t-1]
        dL_dh_next = dL_dh_t * (1.0 - alpha) + go * d_out_d_hprev


# ---------------------------------------------------------------------------
# Python autograd Function
# ---------------------------------------------------------------------------

class _CausalEmaKdvFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: torch.Tensor, kdv_alpha: torch.Tensor):
        """
        x:         [B, N, D]  — input residual (bfloat16)
        alpha:     scalar Parameter — EMA decay rate
        kdv_alpha: scalar Parameter — KdV correction factor
        Returns:   [B, N, D]  — EMA+KdV output (same dtype as x)
        """
        B, N, D = x.shape
        x_c = x.contiguous()
        out = torch.empty_like(x_c)

        # Save EMA states (h, not out) for backward — requires a second pass
        # or we store them during forward. For simplicity, re-run EMA without KdV
        # to get the clean h states.
        h_states = torch.empty_like(x_c)

        a   = float(alpha.item())
        kd  = float(kdv_alpha.item())

        grid = (B * D,)

        # Forward pass
        _ema_kdv_fwd_kernel[grid](
            x_c, out,
            a, kd,
            B, N, D,
            stride_B=N * D, stride_N=D, stride_D=1,
        )

        # Re-run EMA only (no KdV) to save h states for backward
        # (small cost — same loop, just no KdV term)
        _ema_kdv_fwd_kernel[grid](
            x_c, h_states,
            a, 0.0,    # kdv_alpha=0 → pure EMA, saves h[t]
            B, N, D,
            stride_B=N * D, stride_N=D, stride_D=1,
        )

        ctx.save_for_backward(x_c, h_states, alpha, kdv_alpha)
        ctx.B, ctx.N, ctx.D = B, N, D
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, h_states, alpha, kdv_alpha = ctx.saved_tensors
        B, N, D = ctx.B, ctx.N, ctx.D

        a  = float(alpha.item())
        kd = float(kdv_alpha.item())

        grad_x = torch.empty_like(x)
        grad_out_c = grad_out.contiguous()

        grid = (B * D,)
        _ema_kdv_bwd_kernel[grid](
            x, h_states, grad_out_c, grad_x,
            a, kd,
            B, N, D,
            stride_B=N * D, stride_N=D, stride_D=1,
        )

        # Scalar gradients for alpha and kdv_alpha via finite difference
        # (small cost, run once per backward step)
        eps = 1e-4
        with torch.no_grad():
            out_p = causal_ema_kdv(x, alpha + eps, kdv_alpha)
            out_m = causal_ema_kdv(x, alpha - eps, kdv_alpha)
            grad_alpha = ((grad_out * (out_p - out_m)) / (2 * eps)).sum()

            out_p = causal_ema_kdv(x, alpha, kdv_alpha + eps)
            out_m = causal_ema_kdv(x, alpha, kdv_alpha - eps)
            grad_kdv = ((grad_out * (out_p - out_m)) / (2 * eps)).sum()

        return grad_x, grad_alpha.unsqueeze(0), grad_kdv.unsqueeze(0)


def causal_ema_kdv(
    x: torch.Tensor,
    alpha: torch.Tensor,
    kdv_alpha: torch.Tensor,
) -> torch.Tensor:
    """
    Drop-in replacement for the _causal_ema() + _kdv_correction() pair.

    x:         [B, N, D]  bfloat16 or float16
    alpha:     scalar Parameter (EMA decay rate, clamped externally)
    kdv_alpha: scalar Parameter (KdV factor, can be ~0)
    Returns:   [B, N, D]  same dtype

    Usage in DSQGBlock.forward():
        # Old (two separate calls):
        pool = _causal_ema(xi, self.ema_factor)
        pool = _kdv_correction(pool, self.kdv_alpha)

        # New (fused, 256x fewer FLOPs):
        pool = causal_ema_kdv(xi, self.ema_factor.clamp(0.005, 0.5), self.kdv_alpha)
    """
    return _CausalEmaKdvFn.apply(x, alpha.clamp(0.005, 0.5), kdv_alpha)


# ---------------------------------------------------------------------------
# Benchmark utility
# ---------------------------------------------------------------------------

def benchmark(B=8, N=2048, D=256, dtype=torch.bfloat16, device='cuda', n_warmup=10, n_iter=100):
    """Compare old F.conv1d approach vs new Triton IIR kernel."""
    import time, torch.nn.functional as F

    x = torch.randn(B, N, D, dtype=dtype, device=device)
    alpha = torch.tensor(0.0111, requires_grad=True, device=device)
    kdv_alpha = torch.tensor(-0.0017, requires_grad=True, device=device)

    # ── Old approach ──────────────────────────────────────────────────────
    k_len = min(256, N)
    a = alpha.detach().clamp(0.005, 0.5)
    t_idx = torch.arange(k_len, device=device, dtype=torch.float32)
    kernel = a * (1.0 - a).pow(t_idx)
    kernel = (kernel / kernel.sum()).flip(0)
    xi_bd = x.float().permute(0, 2, 1).reshape(B * D, 1, N)
    xi_p = F.pad(xi_bd, (k_len - 1, 0))

    for _ in range(n_warmup):
        _ = F.conv1d(xi_p, kernel.view(1, 1, k_len))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        _ = F.conv1d(xi_p, kernel.view(1, 1, k_len))
    torch.cuda.synchronize()
    t_old = (time.perf_counter() - t0) / n_iter * 1000

    # ── New Triton kernel ─────────────────────────────────────────────────
    for _ in range(n_warmup):
        _ = causal_ema_kdv(x, alpha.detach(), kdv_alpha.detach())
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        _ = causal_ema_kdv(x, alpha.detach(), kdv_alpha.detach())
    torch.cuda.synchronize()
    t_new = (time.perf_counter() - t0) / n_iter * 1000

    print(f"EMA benchmark  B={B} N={N} D={D} dtype={dtype}")
    print(f"  F.conv1d (old):      {t_old:.3f} ms/call")
    print(f"  Triton IIR (new):    {t_new:.3f} ms/call")
    print(f"  Speedup:             {t_old / t_new:.1f}×")
    print(f"  FLOPs reduction:     ~{k_len}×  ({k_len} taps → 1 FMA/step)")
    return t_old, t_new


if __name__ == '__main__':
    # Quick correctness check + benchmark
    torch.manual_seed(42)
    B, N, D = 2, 32, 8
    x = torch.randn(B, N, D, dtype=torch.bfloat16, device='cuda')
    alpha = torch.tensor(0.1, device='cuda')
    kdv = torch.tensor(0.0, device='cuda')  # kdv=0 → should match naive EMA exactly

    # Reference: naive sequential EMA in Python
    ref = torch.zeros_like(x, dtype=torch.float32)
    for t in range(N):
        if t == 0:
            ref[:, t, :] = 0.1 * x[:, t, :].float()
        else:
            ref[:, t, :] = 0.1 * x[:, t, :].float() + 0.9 * ref[:, t-1, :]

    triton_out = causal_ema_kdv(x, alpha, kdv).float()
    max_err = (triton_out - ref).abs().max().item()
    print(f"Correctness check (kdv=0): max error vs naive EMA = {max_err:.2e}")
    assert max_err < 0.01, f"Correctness check failed: max_err={max_err}"
    print("PASSED")
    print()

    # Benchmark at training scale
    benchmark(B=8, N=2048, D=256)   # 14M scale
    print()
    benchmark(B=8, N=2048, D=512)   # 35M scale
