"""
CUDA graph decode correctness test

Decodes the same token sequence eagerly and through captured-graph replay (each into its
own copy of the same KV cache) and checks the logits match at every step. Positions keep
increasing across KV block boundaries and the token changes every step, so a graph that
baked in the capture-time position or token would fail.

Requires CUDA.

Run:
    pytest tests/test_cuda_graphs.py -v
"""

import torch
import pytest

from model import LlamaConfig, Llama

DEVICE = torch.device("cuda")
DTYPE = torch.float16

CFG = LlamaConfig(
    hidden_size=256,
    num_hidden_layers=2,
    num_attention_heads=8,
    num_key_value_heads=2,
    intermediate_size=512,
    vocab_size=1024,
    max_position_embeddings=1024,
    head_dim=32,
    rope_theta=10000.0,
)


def _build_model(seed: int = 0) -> Llama:
    torch.manual_seed(seed)
    model = Llama(CFG)

    def rand(*shape):
        return torch.randn(*shape, device=DEVICE, dtype=DTYPE) * 0.02

    qkv_dim = (CFG.num_attention_heads + 2 * CFG.num_key_value_heads) * CFG.head_dim
    model.embed_tokens = rand(CFG.vocab_size, CFG.hidden_size)
    model.lm_head = rand(CFG.vocab_size, CFG.hidden_size)
    model.norm.weight = 1 + rand(CFG.hidden_size)
    model.cos, model.sin = model.cos.to(DEVICE), model.sin.to(DEVICE)
    for layer in model.layers:
        layer.self_attn.wqkv = rand(qkv_dim, CFG.hidden_size)
        layer.self_attn.wo = rand(CFG.hidden_size, CFG.num_attention_heads * CFG.head_dim)
        layer.input_layernorm.weight = 1 + rand(CFG.hidden_size)
        layer.post_attention_layernorm.weight = 1 + rand(CFG.hidden_size)
        layer.mlp.w_gate_up = rand(2 * CFG.intermediate_size, CFG.hidden_size)
        layer.mlp.w_down = rand(CFG.hidden_size, CFG.intermediate_size)
    return model


def _prefilled_cache(model: Llama, batch: int, prefix: int):
    caches = model.allocate_kv_cache(batch_size=batch, max_seq_len=CFG.max_position_embeddings, device=DEVICE)
    g = torch.Generator(device=DEVICE).manual_seed(1)
    for k, v in caches:
        k[:, :, :prefix] = torch.randn(k[:, :, :prefix].shape, device=DEVICE, dtype=DTYPE, generator=g)
        v[:, :, :prefix] = torch.randn(v[:, :, :prefix].shape, device=DEVICE, dtype=DTYPE, generator=g)
    return caches


def _clone_cache(caches):
    return [(k.clone(), v.clone()) for k, v in caches]


def _decode(model, caches, tokens, start_pos):
    """tokens: (steps, batch). Returns stacked logits (steps, batch, vocab)."""
    outs = []
    for i, t in enumerate(tokens):
        logits = model.forward(t.view(-1, 1), start_pos=start_pos + i, kv_caches=caches)
        outs.append(logits[:, 0].clone())
    return torch.stack(outs)


# batch > 1 is excluded: swiglu_out's x.view(-1) fails on the non-contiguous gate/up halves of
# the fused gate_up buffer for batch > 1. That's a pre-existing eager-path bug on main,
# unrelated to graphs. Add 2 back once it's fixed.
@pytest.mark.parametrize("batch", [1])
@pytest.mark.parametrize("start_pos", [0, 29])
def test_graph_matches_eager(batch, start_pos):
    steps = 100  # crosses 32/64/128-row KV block boundaries
    model = _build_model()
    base = _prefilled_cache(model, batch, start_pos)
    tokens = torch.randint(0, CFG.vocab_size, (steps, batch), device=DEVICE)

    model.enable_cuda_graphs(False)
    eager_cache = _clone_cache(base)
    eager = _decode(model, eager_cache, tokens, start_pos)

    model.enable_cuda_graphs(True)
    graph_cache = _clone_cache(base)
    graphed = _decode(model, graph_cache, tokens, start_pos)

    assert len(model._graphs) == 1, "expected exactly one captured graph"
    max_diff = (eager.float() - graphed.float()).abs().max().item()
    assert max_diff <= 1e-3, f"graph vs eager logits differ: max_diff={max_diff:.2e}"
    for (ke, ve), (kg, vg) in zip(eager_cache, graph_cache):
        assert torch.allclose(ke, kg, atol=1e-3) and torch.allclose(ve, vg, atol=1e-3), "KV caches diverged"


def test_new_kv_cache_recaptures():
    model = _build_model()
    model.enable_cuda_graphs(True)
    tokens = torch.randint(0, CFG.vocab_size, (8, 1), device=DEVICE)
    cache_a = _prefilled_cache(model, 1, 16)
    cache_b = _clone_cache(cache_a)
    out_a = _decode(model, cache_a, tokens, 16)
    out_b = _decode(model, cache_b, tokens, 16)
    assert len(model._graphs) == 2, "a different KV cache must get its own graph"
    assert torch.allclose(out_a, out_b, atol=1e-3)
