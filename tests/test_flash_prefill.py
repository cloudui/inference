"""
Tests for the FA3 prefill kernel (Hopper / CuTe DSL).

Run with:
    pytest tests/test_flash_prefill.py -v

Reference implementation used for correctness checks: PyTorch scaled_dot_product_attention.
"""

import pytest
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Skip the whole file if no Hopper GPU is present
# ---------------------------------------------------------------------------
def _is_hopper() -> bool:
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability()
    return cap[0] >= 9  # sm_90+

pytestmark = pytest.mark.skipif(
    not _is_hopper(),
    reason="Flash Attention 3 prefill kernel requires Hopper (sm_90+)"
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
DTYPE = torch.float16
DEVICE = "cuda"


def make_qkv(batch, q_heads, kv_heads, seq_q, seq_kv, head_dim):
    q = torch.randn(batch, q_heads,  seq_q,  head_dim, dtype=DTYPE, device=DEVICE)
    k = torch.randn(batch, kv_heads, seq_kv, head_dim, dtype=DTYPE, device=DEVICE)
    v = torch.randn(batch, kv_heads, seq_kv, head_dim, dtype=DTYPE, device=DEVICE)
    return q, k, v


def ref_attention(q, k, v, causal: bool, gqa_ratio: int) -> torch.Tensor:
    """Reference via PyTorch SDPA (expand KV for GQA)."""
    # Expand KV heads to match Q heads for reference
    k_exp = k.repeat_interleave(gqa_ratio, dim=1)
    v_exp = v.repeat_interleave(gqa_ratio, dim=1)
    return F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=causal)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("seq_len", [128, 512, 1024])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("gqa_ratio", [1, 4, 8])
def test_flash_prefill_correctness(causal, seq_len, head_dim, gqa_ratio):
    from kernels.flash_prefill import flash_prefill

    batch, kv_heads = 2, 4
    q_heads = kv_heads * gqa_ratio

    q, k, v = make_qkv(batch, q_heads, kv_heads, seq_len, seq_len, head_dim)

    out_ref = ref_attention(q, k, v, causal=causal, gqa_ratio=gqa_ratio)
    out_kernel = flash_prefill(q, k, v, causal=causal)

    torch.testing.assert_close(out_kernel, out_ref, atol=1e-2, rtol=1e-2)


def test_flash_prefill_out_inplace():
    """Verify flash_prefill_out writes into the provided output buffer."""
    from kernels.flash_prefill import flash_prefill_out

    batch, q_heads, kv_heads, seq_len, head_dim = 1, 4, 4, 256, 128
    q, k, v = make_qkv(batch, q_heads, kv_heads, seq_len, seq_len, head_dim)
    out = torch.zeros_like(q)

    flash_prefill_out(q, k, v, out, causal=True)
    assert not out.eq(0).all(), "Output should not be all zeros"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_flash_prefill_dtype(dtype):
    from kernels.flash_prefill import flash_prefill

    q, k, v = make_qkv(1, 8, 2, 256, 256, 128)
    q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
    out = flash_prefill(q, k, v, causal=True)
    assert out.dtype == dtype
