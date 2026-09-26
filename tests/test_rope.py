"""
Llama 3.1 RoPE frequency scaling vs Hugging Face.

Compares precompute_rope_freqs_llama31 against transformers' "llama3" rope_type
(the scheme Llama 3.1 ships with), both the scaled inverse frequencies and the
cos/sin tables at positions past the original 8192-token context.

To run:
    pytest tests/test_rope.py -v
"""

import pytest
import torch
from transformers import LlamaConfig as HFLlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from model import (
    Llama, LlamaConfig, RopeScalingConfig, _parse_rope_scaling,
    precompute_rope_freqs_llama3, precompute_rope_freqs_llama31,
)

DEVICE = torch.device("cuda")
THETA = 500000.0
MAX_SEQ = 131072

# Llama 3.1 8B config.json values
HF_ROPE_SCALING = {
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
    "rope_type": "llama3",
}

# Around both wavelength cutoffs, the original context edge, and out to 128K
POSITIONS = [0, 1, 2, 2047, 2048, 4096, 8191, 8192, 8193, 16384, 32768, 65535, 100000, MAX_SEQ - 1]


def _hf_config(head_dim: int, rope_scaling: dict | None = HF_ROPE_SCALING) -> HFLlamaConfig:
    n_heads = 32
    return HFLlamaConfig(
        hidden_size=head_dim * n_heads,
        num_attention_heads=n_heads,
        max_position_embeddings=MAX_SEQ,
        rope_theta=THETA,
        rope_scaling=rope_scaling,
    )


def _hf_cos_sin(hf_cfg: HFLlamaConfig, positions: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """HF cos/sin at `positions`, first half only (HF duplicates the halves for rotate_half)."""
    rotary = LlamaRotaryEmbedding(hf_cfg).to(DEVICE)
    x = torch.zeros(1, 1, 1, dtype=torch.float32, device=DEVICE)  # only dtype/device are used
    pos = torch.tensor([positions], device=DEVICE)
    cos, sin = rotary(x, pos)
    half = cos.shape[-1] // 2
    return cos[0, :, :half], sin[0, :, :half]


def _assert_close_for_angle(actual: torch.Tensor, expected: torch.Tensor, angles: torch.Tensor, ulps: float):
    """cos/sin tolerance that grows with the angle: m * inv_freq is rounded to fp32, which at
    m ~ 128K is ~0.008 rad (half an ulp of 131072), so a fixed atol can't work. A wrong
    frequency is off by O(1), far outside this."""
    tol = ulps * angles.abs() * 2.0 ** -24 + 2e-6
    err = (actual.double() - expected.double()).abs()
    bad = err > tol
    first = [(POSITIONS[r], d) for r, d in bad.nonzero().tolist()[:8]]
    assert not bad.any(), (
        f"{int(bad.sum())}/{bad.numel()} mismatched, max err {err.max().item():.3g}; "
        f"first (position, dim): {first}"
    )


def test_parse_rope_scaling():
    assert _parse_rope_scaling(None) is None
    assert _parse_rope_scaling(HF_ROPE_SCALING) == RopeScalingConfig()
    with pytest.raises(ValueError):
        _parse_rope_scaling({**HF_ROPE_SCALING, "rope_type": "yarn"})


@pytest.mark.parametrize("head_dim", [64, 128])
def test_scaled_inv_freq_matches_hf(head_dim):
    """Row 1 of the table is cis(inv_freq), so its angle is the scaled inverse frequency.
    Every inv_freq is <= 1 < pi, so the angle doesn't wrap."""
    freqs_cis = precompute_rope_freqs_llama31(head_dim, 2, THETA, RopeScalingConfig(), device=DEVICE)
    ours = torch.angle(freqs_cis[1])

    hf_inv_freq, attention_factor = ROPE_INIT_FUNCTIONS["llama3"](_hf_config(head_dim), DEVICE)
    assert attention_factor == 1.0  # llama3 scaling doesn't scale cos/sin magnitudes

    base = torch.angle(precompute_rope_freqs_llama3(head_dim, 2, THETA, device=DEVICE)[1])
    kept = torch.isclose(hf_inv_freq, base, rtol=1e-6)
    divided = torch.isclose(hf_inv_freq, base / 8.0, rtol=1e-6)
    blended = ~kept & ~divided
    for name, mask in [("kept (high freq)", kept), ("blended (mid freq)", blended), ("divided (low freq)", divided)]:
        torch.testing.assert_close(
            ours[mask], hf_inv_freq[mask], rtol=1e-5, atol=0,
            msg=lambda m, name=name, mask=mask: f"{name} dims {mask.nonzero().flatten().tolist()}:\n{m}",
        )


def _angles(head_dim: int) -> torch.Tensor:
    """Exact angles m * inv_freq (fp64) from HF's scaled fp32 inverse frequencies."""
    hf_inv_freq, _ = ROPE_INIT_FUNCTIONS["llama3"](_hf_config(head_dim), DEVICE)
    return torch.tensor(POSITIONS, device=DEVICE, dtype=torch.float64)[:, None] * hf_inv_freq.double()[None, :]


@pytest.mark.parametrize("head_dim", [64, 128])
def test_scaled_table_matches_exact(head_dim):
    """cos/sin at long-context positions vs fp64 cos/sin of the exact angle."""
    freqs_cis = precompute_rope_freqs_llama31(head_dim, MAX_SEQ, THETA, RopeScalingConfig(), device=DEVICE)
    idx = torch.tensor(POSITIONS, device=DEVICE)
    angles = _angles(head_dim)

    _assert_close_for_angle(freqs_cis.real[idx], torch.cos(angles), angles, ulps=2)
    _assert_close_for_angle(freqs_cis.imag[idx], torch.sin(angles), angles, ulps=2)


@pytest.mark.parametrize("head_dim", [64, 128])
def test_scaled_table_matches_hf(head_dim):
    """Same positions vs HF's LlamaRotaryEmbedding. HF rounds its fp32 angles differently
    (a matmul), so allow rounding error on both sides."""
    freqs_cis = precompute_rope_freqs_llama31(head_dim, MAX_SEQ, THETA, RopeScalingConfig(), device=DEVICE)
    idx = torch.tensor(POSITIONS, device=DEVICE)
    angles = _angles(head_dim)
    hf_cos, hf_sin = _hf_cos_sin(_hf_config(head_dim), POSITIONS)

    _assert_close_for_angle(freqs_cis.real[idx], hf_cos, angles, ulps=4)
    _assert_close_for_angle(freqs_cis.imag[idx], hf_sin, angles, ulps=4)


def test_model_picks_table_from_config():
    """Llama(config) builds the scaled table when rope_scaling is set, and the plain
    Llama 3 table when it isn't."""
    common = dict(hidden_size=4096, num_hidden_layers=1, num_attention_heads=32, num_key_value_heads=8,
                  max_position_embeddings=MAX_SEQ, rope_theta=THETA, head_dim=128)
    idx = torch.tensor(POSITIONS, device=DEVICE)

    scaled = Llama(LlamaConfig(**common, rope_scaling=RopeScalingConfig()))
    angles = _angles(128)
    _assert_close_for_angle(scaled.cos[idx], torch.cos(angles), angles, ulps=2)
    _assert_close_for_angle(scaled.sin[idx], torch.sin(angles), angles, ulps=2)

    plain = Llama(LlamaConfig(**common))
    plain_inv_freq = 1.0 / (THETA ** (torch.arange(0, 128, 2, device=DEVICE).double() / 128))
    angles = idx.double()[:, None] * plain_inv_freq[None, :]
    _assert_close_for_angle(plain.cos[idx], torch.cos(angles), angles, ulps=4)
    _assert_close_for_angle(plain.sin[idx], torch.sin(angles), angles, ulps=4)
