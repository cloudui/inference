"""
Llama Full Model Forward Correctness Test

Compares the custom Llama.forward() (embed → DecoderLayers → RMSNorm → lm_head)
against HF's LlamaForCausalLM in a single-token decode setting with KV cache,
at short positions (Llama 3 RoPE) and at long context with Llama 3.1 RoPE scaling.

Requires CUDA — Triton kernels won't compile on CPU.

Run:
    pytest tests/test_forward.py -v
"""

import torch
import pytest
from transformers import LlamaConfig as HFLlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.cache_utils import DynamicCache
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from model import LlamaConfig, Llama, RopeScalingConfig

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


def _hf_model(hf_cfg: HFLlamaConfig) -> LlamaForCausalLM:
    """HF model in DTYPE on DEVICE, with its RoPE frequencies kept in fp32.

    .to(dtype) also casts HF's inv_freq buffers to fp16. That ~5e-4 relative error is
    multiplied by the position, so by position 20K the angles are off by radians.
    from_pretrained(dtype=...) keeps them fp32, so recompute them to match real checkpoints.
    """
    hf_model = LlamaForCausalLM(config=hf_cfg).to(DEVICE, dtype=DTYPE)
    rope_type = (hf_cfg.rope_parameters or {}).get("rope_type", "default")
    rotary = hf_model.model.rotary_emb
    if rope_type == "default":
        inv_freq, _ = rotary.compute_default_rope_parameters(hf_cfg, DEVICE)
    else:
        inv_freq, _ = ROPE_INIT_FUNCTIONS[rope_type](hf_cfg, DEVICE)
    rotary.inv_freq = inv_freq
    rotary.original_inv_freq = inv_freq.clone()
    return hf_model


def _copy_weights(hf_model: LlamaForCausalLM, custom_model: Llama) -> None:
    """Copies HF weights into the custom model's fused layouts."""
    custom_model.embed_tokens = hf_model.model.embed_tokens.weight.clone().to(DEVICE, dtype=DTYPE)

    for hl, cl in zip(hf_model.model.layers, custom_model.layers):
        wq = hl.self_attn.q_proj.weight.clone().to(DEVICE, dtype=DTYPE)
        wk = hl.self_attn.k_proj.weight.clone().to(DEVICE, dtype=DTYPE)
        wv = hl.self_attn.v_proj.weight.clone().to(DEVICE, dtype=DTYPE)
        cl.self_attn.wqkv = torch.cat((wq, wk, wv), dim=0)
        cl.self_attn.wo = hl.self_attn.o_proj.weight.clone().to(DEVICE, dtype=DTYPE)
        cl.input_layernorm.weight = hl.input_layernorm.weight.clone().to(DEVICE, dtype=DTYPE)
        cl.post_attention_layernorm.weight = hl.post_attention_layernorm.weight.clone().to(DEVICE, dtype=DTYPE)
        w_gate = hl.mlp.gate_proj.weight.clone().to(DEVICE, dtype=DTYPE)
        w_up = hl.mlp.up_proj.weight.clone().to(DEVICE, dtype=DTYPE)
        cl.mlp.w_gate_up = torch.cat((w_gate, w_up), dim=0)
        cl.mlp.w_down = hl.mlp.down_proj.weight.clone().to(DEVICE, dtype=DTYPE)

    custom_model.norm.weight = hf_model.model.norm.weight.clone().to(DEVICE, dtype=DTYPE)
    # lm_head: HF is (vocab, hidden), stored directly in same layout
    custom_model.lm_head = hf_model.lm_head.weight.clone().to(DEVICE, dtype=DTYPE)


def _make_model_pair():
    """Creates HF and custom Llama with identical random weights.

    Returns (hf_model, custom_model, hf_config, custom_config).
    """
    hf_cfg, custom_cfg = _make_configs()

    hf_model = _hf_model(hf_cfg)
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
    _copy_weights(hf_model, custom_model)

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


@pytest.mark.parametrize("batch", [1, 3])
def test_multi_step_decode(batch):
    """Sequential decode steps, comparing logits at each step."""
    hf_model, custom_model, hf_cfg, custom_cfg = _make_model_pair()
    n_steps = 10

    hf_cache = DynamicCache()
    custom_caches = custom_model.allocate_kv_cache(batch_size=batch, max_seq_len=MAX_SEQ, device=DEVICE)

    for step in range(n_steps):
        token_ids = torch.randint(0, VOCAB, (batch, 1), device=DEVICE)
        position_ids = torch.full((batch, 1), step, device=DEVICE)

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


@pytest.mark.parametrize("batch", [1, 3])
def test_argmax_agreement(batch):
    """Verifies that the top-1 predicted token matches HF at every step.

    Even if logit values differ slightly due to fp16, the argmax should agree.
    """
    hf_model, custom_model, hf_cfg, custom_cfg = _make_model_pair()
    n_steps = 16

    hf_cache = DynamicCache()
    custom_caches = custom_model.allocate_kv_cache(batch_size=batch, max_seq_len=MAX_SEQ, device=DEVICE)

    for step in range(n_steps):
        token_ids = torch.randint(0, VOCAB, (batch, 1), device=DEVICE)
        position_ids = torch.full((batch, 1), step, device=DEVICE)

        with torch.inference_mode():
            hf_out = hf_model(
                input_ids=token_ids,
                past_key_values=hf_cache,
                use_cache=True,
                position_ids=position_ids,
            )

        custom_logits = custom_model.forward(token_ids, start_pos=step, kv_caches=custom_caches)

        hf_logits = hf_out.logits.float()
        hf_token = hf_logits.argmax(dim=-1)
        custom_token = custom_logits.argmax(dim=-1)

        # A different pick is only allowed on a near-tie: fp16 logits can tie exactly, and the
        # two implementations then break the tie differently. HF must rate our pick within the
        # same 5e-3 used for the logit comparisons.
        hf_max = hf_logits.max(dim=-1).values
        hf_at_ours = hf_logits.gather(-1, custom_token.unsqueeze(-1)).squeeze(-1)
        assert torch.all(hf_max - hf_at_ours <= 5e-3), (
            f"Argmax mismatch at step {step}: HF={hf_token.tolist()}, custom={custom_token.tolist()}, "
            f"HF logit gap {(hf_max - hf_at_ours).tolist()}"
        )


# ── Llama 3.1 long context ───────────────────────────────────────────────────
# Real 8B head geometry (head_dim 128, theta 500K, GQA 4:1, 3.1 RoPE scaling), so the
# kept / blended / divided frequency bands match the real model. Both KV caches are
# pre-filled with the same random context, then decoding starts at a long position.

LC_HIDDEN = 512
LC_Q_HEADS = 4
LC_KV_HEADS = 1
LC_HEAD_DIM = 128
LC_MAX_POS = 131072
LC_THETA = 500000.0
LC_STEPS = 4
LC_ATOL = 1e-2  # measured ~2.5e-3; plain Llama 3 RoPE misses by >= 0.38 (see negative control)
HF_ROPE_SCALING = {
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
    "rope_type": "llama3",
}


def _make_long_context_pair(custom_rope_scaling: RopeScalingConfig | None = RopeScalingConfig()):
    torch.manual_seed(0)
    hf_cfg = HFLlamaConfig(
        vocab_size=VOCAB,
        hidden_size=LC_HIDDEN,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=LC_Q_HEADS,
        num_key_value_heads=LC_KV_HEADS,
        intermediate_size=INTERMEDIATE,
        max_position_embeddings=LC_MAX_POS,
        rms_norm_eps=EPS,
        rope_theta=LC_THETA,
        rope_scaling=HF_ROPE_SCALING,
        attn_implementation="sdpa",
    )
    hf_model = _hf_model(hf_cfg)
    with torch.no_grad():
        for p in hf_model.parameters():
            p.normal_(std=0.02)
        for layer in hf_model.model.layers:
            # Larger q/k weights make attention peaked, so a wrong rotation changes the output
            # (at std 0.02 attention is near uniform and hides RoPE errors)
            layer.self_attn.q_proj.weight.normal_(std=0.05)
            layer.self_attn.k_proj.weight.normal_(std=0.05)
            layer.input_layernorm.weight.normal_(mean=1.0, std=0.1)
            layer.post_attention_layernorm.weight.normal_(mean=1.0, std=0.1)
        hf_model.model.norm.weight.normal_(mean=1.0, std=0.1)

    custom_model = Llama(config=LlamaConfig(
        hidden_size=LC_HIDDEN,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=LC_Q_HEADS,
        num_key_value_heads=LC_KV_HEADS,
        intermediate_size=INTERMEDIATE,
        vocab_size=VOCAB,
        max_position_embeddings=LC_MAX_POS,
        rms_norm_eps=EPS,
        rope_theta=LC_THETA,
        head_dim=LC_HEAD_DIM,
        rope_scaling=custom_rope_scaling,
    ))
    _copy_weights(hf_model, custom_model)
    return hf_model, custom_model


def _decode_from_long_context(hf_model, custom_model, start_pos: int):
    """Pre-fills both caches with the same random context of length start_pos, then decodes
    LC_STEPS tokens. Returns [(hf_logits, custom_logits)] per step."""
    hf_cache = DynamicCache()
    custom_caches = custom_model.allocate_kv_cache(batch_size=1, max_seq_len=start_pos + LC_STEPS, device=DEVICE)
    for i, (K, V) in enumerate(custom_caches):
        k = torch.randn(1, LC_KV_HEADS, start_pos, LC_HEAD_DIM, device=DEVICE, dtype=DTYPE)
        v = torch.randn_like(k)
        K[:, :, :start_pos] = k
        V[:, :, :start_pos] = v
        hf_cache.update(k.clone(), v.clone(), i)

    results = []
    for step in range(LC_STEPS):
        token_ids = torch.randint(0, VOCAB, (1, 1), device=DEVICE)
        pos = start_pos + step
        with torch.inference_mode():
            hf_logits = hf_model(
                input_ids=token_ids,
                past_key_values=hf_cache,
                use_cache=True,
                position_ids=torch.tensor([[pos]], device=DEVICE),
            ).logits
        custom_logits = custom_model.forward(token_ids, start_pos=pos, kv_caches=custom_caches).clone()
        results.append((hf_logits, custom_logits))
    return results


# before the original 8K context, decoding across its edge, and far past it
@pytest.mark.parametrize("start_pos", [4096, 8190, 20000, 100000])
def test_llama31_long_context_decode(start_pos):
    hf_model, custom_model = _make_long_context_pair()
    for step, (hf_logits, custom_logits) in enumerate(_decode_from_long_context(hf_model, custom_model, start_pos)):
        max_diff = (hf_logits - custom_logits).abs().max().item()
        assert max_diff < LC_ATOL, f"position {start_pos + step}: max_diff={max_diff:.2e}"
        assert torch.equal(hf_logits.argmax(-1), custom_logits.argmax(-1)), f"argmax mismatch at position {start_pos + step}"


def test_llama31_negative_control():
    """Plain Llama 3 RoPE must fail the same comparison, or the test above can't tell them apart."""
    hf_model, custom_model = _make_long_context_pair(custom_rope_scaling=None)
    results = _decode_from_long_context(hf_model, custom_model, 20000)
    max_diff = max((h - c).abs().max().item() for h, c in results)
    assert max_diff > 10 * LC_ATOL, f"unscaled RoPE only off by {max_diff:.2e}; test is not sensitive to RoPE"
