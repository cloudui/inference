"""
Flash Attention 3 Prefill Kernel — Hopper (SM90), CuTe DSL
===========================================================
Target: H100 / H200 (sm_90a)
Algorithm: Standard prefill flash attention with causal masking support
Backend: NVIDIA CuTe DSL (cutedsl) — warp-specialized, persistent kernel style

Tensor layouts follow the same conventions as the rest of this repo:
    Q : [batch, q_heads,  seq_q,  head_dim]  — float16 / bfloat16
    K : [batch, kv_heads, seq_kv, head_dim]  — float16 / bfloat16
    V : [batch, kv_heads, seq_kv, head_dim]  — float16 / bfloat16
    O : [batch, q_heads,  seq_q,  head_dim]  — same dtype as Q

GQA is supported: gqa_ratio = q_heads // kv_heads.

References:
  - Flash Attention 3: https://arxiv.org/abs/2407.08608
  - CuTe DSL docs: https://docs.nvidia.com/cutlass/media/docs/cutedsl/index.html
  - CUTLASS FA3 reference impl: https://github.com/Dao-AILab/flash-attention/tree/main/hopper
"""

import math
import torch

# ---------------------------------------------------------------------------
# CuTe DSL imports
# ---------------------------------------------------------------------------
# TODO: fill in the exact cutedsl imports you need as you build the kernel.
# e.g.:
#   from cutedsl import cute
#   from cutedsl.cute import Tensor, Layout, ...
#   from cutedsl.cute.arch import ...


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LOG2_E: float = 1.4426950408889634   # log2(e) — lets us use exp2 everywhere


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
# TODO: implement the CuTe DSL kernel here.
#
# Suggested structure (FA3 warp-specialised style):
#   - Producer warps: issue TMA / async copy for Q tiles, K/V tiles
#   - Consumer warps: run WGMMA Q@Kᵀ, softmax rescale, WGMMA P@V
#   - Use pipeline stages to overlap compute and memory
#
# def flash_prefill_kernel(...):
#     ...


# ---------------------------------------------------------------------------
# Python launch wrapper
# ---------------------------------------------------------------------------

def flash_prefill_out(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    causal: bool = True,
) -> None:
    """In-place prefill flash attention (FA3, Hopper).

    Args:
        q:      Query tensor  [batch, q_heads,  seq_q,  head_dim], fp16/bf16.
        k:      Key tensor    [batch, kv_heads, seq_kv, head_dim], same dtype.
        v:      Value tensor  [batch, kv_heads, seq_kv, head_dim], same dtype.
        out:    Pre-allocated output [batch, q_heads, seq_q, head_dim], same dtype.
        causal: Apply causal (lower-triangular) mask.
    """
    batch, q_heads, seq_q, head_dim = q.shape
    _, kv_heads, seq_kv, _ = k.shape
    gqa_ratio = q_heads // kv_heads

    assert q.is_cuda and k.is_cuda and v.is_cuda and out.is_cuda, "All tensors must be on CUDA"
    assert q.dtype in (torch.float16, torch.bfloat16), "Only fp16/bf16 supported"
    assert q.dtype == k.dtype == v.dtype == out.dtype
    assert out.shape == q.shape

    scale = (1.0 / math.sqrt(head_dim)) * LOG2_E

    # TODO: launch the CuTe DSL kernel here.
    raise NotImplementedError("flash_prefill kernel not yet implemented")


def flash_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    """Allocating wrapper around flash_prefill_out.

    Returns:
        Output tensor [batch, q_heads, seq_q, head_dim], same dtype as q.
    """
    out = torch.empty_like(q)
    flash_prefill_out(q, k, v, out, causal=causal)
    return out
