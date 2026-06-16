"""
Llama Full Model Forward Correctness Test

Compares the custom Llama.forward() (embed → DecoderLayers → RMSNorm → lm_head)
against HF's LlamaForCausalLM in a single-token decode setting with KV cache.

Requires CUDA — Triton kernels won't compile on CPU.

Run:
    pytest tests/test_forward.py -v
"""

import torch
import pytest
from transformers import LlamaConfig as HFLlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.cache_utils import DynamicCache

from model import LlamaConfig, Llama

DEVICE = torch.device("cuda")
DTYPE = torch.float16

# ── Scaled-down config ───────────────────────────────────────────────────────

HIDDEN = 256
N_Q_HEADS = 8
N_KV_HEADS = 2
HEAD_DIM = HIDDEN // N_Q_HEADS
INTERMEDIATE = 512
VOCAB = 1024
MAX_SEQ = 512
NUM_LAYERS = 2
EPS = 1e-6
THETA = 10000.0


def _make_configs():
    hf = HFLlamaConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        num_hidden_layers=NUM_LAYERS,
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
        num_hidden_layers=NUM_LAYERS,
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


def _make_model_pair():
    """Creates HF and custom Llama with identical random weights.

    Returns (hf_model, custom_model, hf_config, custom_config).
    """
    hf_cfg, custom_cfg = _make_configs()

    hf_model = LlamaForCausalLM(config=hf_cfg).to(DEVICE, dtype=DTYPE)
    with torch.no_grad():
        # Attention + MLP weights: small random
        for p in hf_model.parameters():
            p.normal_(std=0.02)
        # Norm weights: centered around 1.0
        hf_model.model.norm.weight.normal_(mean=1.0, std=0.1)
        for layer in hf_model.model.layers:
            layer.input_layernorm.weight.normal_(mean=1.0, std=0.1)
            layer.post_attention_layernorm.weight.normal_(mean=1.0, std=0.1)

    custom_model = Llama(config=custom_cfg)

    # Embedding: both are (vocab, hidden), direct copy
    custom_model.embed_tokens = hf_model.model.embed_tokens.weight.clone().to(DEVICE, dtype=DTYPE)

    # Per-layer weights
    for i in range(NUM_LAYERS):
        hl = hf_model.model.layers[i]
        cl = custom_model.layers[i]

        wq = hl.self_attn.q_proj.weight.T.clone().to(DEVICE, dtype=DTYPE)
        wk = hl.self_attn.k_proj.weight.T.clone().to(DEVICE, dtype=DTYPE)
        wv = hl.self_attn.v_proj.weight.T.clone().to(DEVICE, dtype=DTYPE)
        cl.self_attn.wqkv = torch.concat((wq, wk, wv), dim=-1)
        cl.self_attn.wo = hl.self_attn.o_proj.weight.T.clone().to(DEVICE, dtype=DTYPE)
        cl.input_layernorm.weight = hl.input_layernorm.weight.clone().to(DEVICE, dtype=DTYPE)
        cl.post_attention_layernorm.weight = hl.post_attention_layernorm.weight.clone().to(DEVICE, dtype=DTYPE)
        w_gate = hl.mlp.gate_proj.weight.T.clone().to(DEVICE, dtype=DTYPE)
        w_up = hl.mlp.up_proj.weight.T.clone().to(DEVICE, dtype=DTYPE)
        cl.mlp.w_gate_up = torch.concat((w_gate, w_up), dim=-1)
        cl.mlp.w_down = hl.mlp.down_proj.weight.T.clone().to(DEVICE, dtype=DTYPE)

    # Final norm
    custom_model.norm.weight = hf_model.model.norm.weight.clone().to(DEVICE, dtype=DTYPE)

    # lm_head: HF is (vocab, hidden), custom is (hidden, vocab)
    custom_model.lm_head = hf_model.lm_head.weight.T.clone().to(DEVICE, dtype=DTYPE)

    return hf_model, custom_model, hf_cfg, custom_cfg


# ── Tests ────────────────────────────────────────────────────────────────────

def test_single_decode_step():
    """Single-token forward at position 0 — simplest possible case."""
    hf_model, custom_model, hf_cfg, custom_cfg = _make_model_pair()
    batch = 1

    token_ids = torch.randint(0, VOCAB, (batch, 1), device=DEVICE)

    # HF side
    hf_cache = DynamicCache()
    with torch.inference_mode():
        hf_out = hf_model(
            input_ids=token_ids,
            past_key_values=hf_cache,
            use_cache=True,
            position_ids=torch.tensor([[0]], device=DEVICE),
        )
        hf_logits = hf_out.logits

    # Custom side
    custom_caches = custom_model.allocate_kv_cache(batch_size=batch, max_seq_len=MAX_SEQ, device=DEVICE)
    custom_logits = custom_model.forward(token_ids, start_pos=0, kv_caches=custom_caches)

    max_diff = (hf_logits - custom_logits).abs().max().item()
    assert torch.allclose(hf_logits, custom_logits, atol=5e-3), (
        f"Single decode step mismatch: max_diff={max_diff:.2e}"
    )


def test_multi_step_decode():
    """Sequential decode steps, comparing logits at each step."""
    hf_model, custom_model, hf_cfg, custom_cfg = _make_model_pair()
    batch = 1
    n_steps = 10

    hf_cache = DynamicCache()
    custom_caches = custom_model.allocate_kv_cache(batch_size=batch, max_seq_len=MAX_SEQ, device=DEVICE)

    for step in range(n_steps):
        token_ids = torch.randint(0, VOCAB, (batch, 1), device=DEVICE)
        position_ids = torch.tensor([[step]], device=DEVICE)

        with torch.inference_mode():
            hf_out = hf_model(
                input_ids=token_ids,
                past_key_values=hf_cache,
                use_cache=True,
                position_ids=position_ids,
            )
            hf_logits = hf_out.logits

        custom_logits = custom_model.forward(token_ids, start_pos=step, kv_caches=custom_caches)

        max_diff = (hf_logits - custom_logits).abs().max().item()
        assert torch.allclose(hf_logits, custom_logits, atol=5e-3), (
            f"Multi-step decode mismatch at step {step}: max_diff={max_diff:.2e}"
        )


def test_argmax_agreement():
    """Verifies that the top-1 predicted token matches HF at every step.

    Even if logit values differ slightly due to fp16, the argmax should agree.
    """
    hf_model, custom_model, hf_cfg, custom_cfg = _make_model_pair()
    batch = 1
    n_steps = 16

    hf_cache = DynamicCache()
    custom_caches = custom_model.allocate_kv_cache(batch_size=batch, max_seq_len=MAX_SEQ, device=DEVICE)

    for step in range(n_steps):
        token_ids = torch.randint(0, VOCAB, (batch, 1), device=DEVICE)
        position_ids = torch.tensor([[step]], device=DEVICE)

        with torch.inference_mode():
            hf_out = hf_model(
                input_ids=token_ids,
                past_key_values=hf_cache,
                use_cache=True,
                position_ids=position_ids,
            )

        custom_logits = custom_model.forward(token_ids, start_pos=step, kv_caches=custom_caches)

        hf_token = hf_out.logits.argmax(dim=-1)
        custom_token = custom_logits.argmax(dim=-1)

        assert torch.equal(hf_token, custom_token), (
            f"Argmax mismatch at step {step}: HF={hf_token.item()}, custom={custom_token.item()}"
        )
