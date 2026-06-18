"""
DecoderLayer Correctness Test

Compares the custom DecoderLayer (RMSNorm → Attention + residual → RMSNorm → MLP + residual)
against Hugging Face's LlamaDecoderLayer in a single-token decode setting with KV cache.

Requires CUDA — Triton kernels won't compile on CPU.

Run:
    pytest tests/test_decode.py -v
"""

import torch
import pytest
from transformers import LlamaConfig as HFLlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRotaryEmbedding,
)
from transformers.cache_utils import DynamicCache

from model import LlamaConfig, DecoderLayer, precompute_rope_freqs

DEVICE = torch.device("cuda")

# ── Scaled-down config ───────────────────────────────────────────────────────

HIDDEN = 256
N_Q_HEADS = 8
N_KV_HEADS = 2
HEAD_DIM = HIDDEN // N_Q_HEADS
INTERMEDIATE = 512
VOCAB = 1024
MAX_SEQ = 512
EPS = 1e-6
THETA = 10000.0


def _make_configs():
    """Returns (hf_config, custom_config) with matching hyperparams."""
    hf = HFLlamaConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        num_hidden_layers=1,
        num_attention_heads=N_Q_HEADS,
        num_key_value_heads=N_KV_HEADS,
        intermediate_size=INTERMEDIATE,
        max_position_embeddings=MAX_SEQ,
        rms_norm_eps=EPS,
        rope_theta=THETA,
        attn_implementation="sdpa",
    )
    custom = LlamaConfig(
        hidden_size=HIDDEN,
        num_hidden_layers=1,
        num_attention_heads=N_Q_HEADS,
        num_key_value_heads=N_KV_HEADS,
        intermediate_size=INTERMEDIATE,
        vocab_size=VOCAB,
        max_position_embeddings=MAX_SEQ,
        rms_norm_eps=EPS,
        rope_theta=THETA,
        head_dim=HEAD_DIM,
    )
    return hf, custom


def _make_decoder_layer_pair():
    """Creates HF and custom DecoderLayer with identical weights.

    Returns (hf_layer, custom_layer, hf_config, custom_config).
    """
    hf_cfg, custom_cfg = _make_configs()

    hf_layer = LlamaDecoderLayer(config=hf_cfg, layer_idx=0).to(DEVICE, dtype=torch.float16)
    with torch.no_grad():
        # Attention weights
        hf_layer.self_attn.q_proj.weight.normal_(std=0.02)
        hf_layer.self_attn.k_proj.weight.normal_(std=0.02)
        hf_layer.self_attn.v_proj.weight.normal_(std=0.02)
        hf_layer.self_attn.o_proj.weight.normal_(std=0.02)
        # Norm weights
        hf_layer.input_layernorm.weight.normal_(mean=1.0, std=0.1)
        hf_layer.post_attention_layernorm.weight.normal_(mean=1.0, std=0.1)
        # MLP weights
        hf_layer.mlp.gate_proj.weight.normal_(std=0.02)
        hf_layer.mlp.up_proj.weight.normal_(std=0.02)
        hf_layer.mlp.down_proj.weight.normal_(std=0.02)

    custom_layer = DecoderLayer(config=custom_cfg, layer_idx=0)

    # Copy weights directly in HF-native (out, in) layout, concat along dim=0
    wq = hf_layer.self_attn.q_proj.weight.clone().to(DEVICE)
    wk = hf_layer.self_attn.k_proj.weight.clone().to(DEVICE)
    wv = hf_layer.self_attn.v_proj.weight.clone().to(DEVICE)
    custom_layer.self_attn.wqkv = torch.cat((wq, wk, wv), dim=0)
    custom_layer.self_attn.wo = hf_layer.self_attn.o_proj.weight.clone().to(DEVICE)
    custom_layer.input_layernorm.weight = hf_layer.input_layernorm.weight.clone().to(DEVICE)
    custom_layer.post_attention_layernorm.weight = hf_layer.post_attention_layernorm.weight.clone().to(DEVICE)
    w_gate = hf_layer.mlp.gate_proj.weight.clone().to(DEVICE)
    w_up = hf_layer.mlp.up_proj.weight.clone().to(DEVICE)
    custom_layer.mlp.w_gate_up = torch.cat((w_gate, w_up), dim=0)
    custom_layer.mlp.w_down = hf_layer.mlp.down_proj.weight.clone().to(DEVICE)

    return hf_layer, custom_layer, hf_cfg, custom_cfg


def _get_rope_embeds(hf_cfg, x, position_ids):
    """Computes cos/sin RoPE embeddings via HF's LlamaRotaryEmbedding."""
    rotary = LlamaRotaryEmbedding(config=hf_cfg).to(DEVICE, dtype=torch.float16)
    return rotary(x, position_ids)


# ── Tests ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cache_position", [0, 1, 7, 63])
def test_single_decode_step(cache_position):
    """Single-token decode at various cache positions."""
    hf_layer, custom_layer, hf_cfg, custom_cfg = _make_decoder_layer_pair()
    freqs_cis = precompute_rope_freqs(
        custom_cfg.head_dim, custom_cfg.max_position_embeddings, custom_cfg.rope_theta
    )
    cos_table = freqs_cis.real.contiguous()
    sin_table = freqs_cis.imag.contiguous()
    batch = 1

    x = torch.randn(batch, 1, HIDDEN, device=DEVICE, dtype=torch.float16)
    position_ids = torch.tensor([[cache_position]], device=DEVICE)
    cos, sin = _get_rope_embeds(hf_cfg, x, position_ids)

    # Shared KV history
    history_k = torch.randn(batch, N_KV_HEADS, cache_position, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    history_v = torch.randn(batch, N_KV_HEADS, cache_position, HEAD_DIM, device=DEVICE, dtype=torch.float16)

    # HF cache
    hf_cache = DynamicCache()
    if cache_position > 0:
        hf_cache.update(history_k.clone(), history_v.clone(), layer_idx=0)

    # Custom cache
    k_cache = torch.zeros(batch, N_KV_HEADS, MAX_SEQ, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    v_cache = torch.zeros(batch, N_KV_HEADS, MAX_SEQ, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    if cache_position > 0:
        k_cache[:, :, :cache_position] = history_k.clone()
        v_cache[:, :, :cache_position] = history_v.clone()

    with torch.inference_mode():
        hf_out = hf_layer(
            hidden_states=x,
            position_embeddings=(cos, sin),
            past_key_values=hf_cache,
            use_cache=True,
        )
        hf_hidden = hf_out[0]

        residual, mlp_out = custom_layer(
            x,
            rope_embeds=(cos_table, sin_table),
            kv_cache=(k_cache, v_cache),
            cache_position=cache_position,
        )
        custom_hidden = residual + mlp_out

    max_diff = (hf_hidden - custom_hidden).abs().max().item()
    assert torch.allclose(hf_hidden, custom_hidden, atol=2e-3), (
        f"DecoderLayer mismatch at cache_position={cache_position}: max_diff={max_diff:.2e}"
    )


def test_multi_step_decode():
    """Sequential decode steps, building KV cache incrementally."""
    hf_layer, custom_layer, hf_cfg, custom_cfg = _make_decoder_layer_pair()
    freqs_cis = precompute_rope_freqs(
        custom_cfg.head_dim, custom_cfg.max_position_embeddings, custom_cfg.rope_theta
    )
    cos_table = freqs_cis.real.contiguous()
    sin_table = freqs_cis.imag.contiguous()
    batch = 1
    n_steps = 16

    k_cache = torch.zeros(batch, N_KV_HEADS, MAX_SEQ, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    v_cache = torch.zeros(batch, N_KV_HEADS, MAX_SEQ, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    hf_cache = DynamicCache()

    rotary = LlamaRotaryEmbedding(config=hf_cfg).to(DEVICE, dtype=torch.float16)

    for step in range(n_steps):
        x = torch.randn(batch, 1, HIDDEN, device=DEVICE, dtype=torch.float16)
        position_ids = torch.tensor([[step]], device=DEVICE)
        cos, sin = rotary(x, position_ids)

        with torch.inference_mode():
            hf_out = hf_layer(
                hidden_states=x,
                position_embeddings=(cos, sin),
                past_key_values=hf_cache,
                use_cache=True,
            )
            hf_hidden = hf_out[0]

            residual, mlp_out = custom_layer(
                x,
                rope_embeds=(cos_table, sin_table),
                kv_cache=(k_cache, v_cache),
                cache_position=step,
            )
            custom_hidden = residual + mlp_out

        max_diff = (hf_hidden - custom_hidden).abs().max().item()
        assert torch.allclose(hf_hidden, custom_hidden, atol=2e-3), (
            f"Multi-step mismatch at step {step}: max_diff={max_diff:.2e}"
        )
