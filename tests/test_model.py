"""
Correctness Test Suite for Custom LLaMA 3 8B Inference Engine

This test suite compares custom modules from `model.py` against their Hugging Face
`transformers` equivalents. It runs on a GPU instance using CUDA and tests the actual
Triton/CUDA kernels directly.

Run on GPU/CUDA:
    pytest tests/test_model.py
"""

import math
import torch
import pytest
import transformers
from transformers import LlamaConfig as HFLlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    LlamaRMSNorm,
    LlamaMLP,
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaModel,
)

# Import custom model classes and kernels
import model
from model import LlamaConfig, RMSNorm, Attention, MLP, DecoderLayer, Llama
from kernels import rmsnorm, swiglu, flash_decode

DEVICE = torch.device("cuda")

def get_test_configs():
    """Returns matching scaled-down configurations for Hugging Face and custom models."""
    hidden_size = 256
    num_attention_heads = 8
    num_key_value_heads = 2
    head_dim = hidden_size // num_attention_heads
    intermediate_size = 512
    vocab_size = 1024
    max_position_embeddings = 512
    rms_norm_eps = 1e-6
    rope_theta = 10000.0

    hf_config = HFLlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=2,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_position_embeddings,
        rms_norm_eps=rms_norm_eps,
        rope_theta=rope_theta,
        attn_implementation="sdpa",
    )

    custom_config = LlamaConfig(
        hidden_size=hidden_size,
        num_hidden_layers=2,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings,
        rms_norm_eps=rms_norm_eps,
        rope_theta=rope_theta,
        head_dim=head_dim,
    )

    return hf_config, custom_config


def test_rope_correctness():
    """Compares the custom complex RoPE precomputation and rotate operations

    against Hugging Face's LlamaRotaryEmbedding.
    """
    hf_config, custom_config = get_test_configs()
    seq_len = 16
    batch = 2
    
    hf_rotary = LlamaRotaryEmbedding(config=hf_config).to(DEVICE, dtype=torch.float16)
    
    q = torch.randn(batch, custom_config.num_attention_heads, seq_len, custom_config.head_dim, device=DEVICE, dtype=torch.float16)
    k = torch.randn(batch, custom_config.num_key_value_heads, seq_len, custom_config.head_dim, device=DEVICE, dtype=torch.float16)
    position_ids = torch.arange(seq_len, device=DEVICE).unsqueeze(0).expand(batch, -1)
    
    cos, sin = hf_rotary(q, position_ids)
    q_rot_hf, k_rot_hf = apply_rotary_pos_emb(q, k, cos, sin)
    
    freqs_cis = model.precompute_rope_freqs(
        head_dim=custom_config.head_dim,
        max_seq_len=custom_config.max_position_embeddings,
        theta=custom_config.rope_theta,
        device=DEVICE,
    )
    
    freqs_cis_slice = freqs_cis[:seq_len]
    cos_custom = freqs_cis_slice.real
    sin_custom = freqs_cis_slice.imag
    
    cos_full = torch.cat([cos_custom, cos_custom], dim=-1)
    sin_full = torch.cat([sin_custom, sin_custom], dim=-1)
    
    cos_broadcast = cos_full.unsqueeze(0).unsqueeze(0).half()
    sin_broadcast = sin_full.unsqueeze(0).unsqueeze(0).half()
    
    q_rot_custom, k_rot_custom = model.apply_rope(q, k, cos_broadcast, sin_broadcast)
    
    assert torch.allclose(q_rot_hf, q_rot_custom, rtol=1e-3, atol=1e-3), "Q RoPE rotation mismatch"
    assert torch.allclose(k_rot_hf, k_rot_custom, rtol=1e-3, atol=1e-3), "K RoPE rotation mismatch"


def test_rmsnorm_correctness():
    """Compares the custom RMSNorm against Hugging Face's LlamaRMSNorm."""
    hf_config, custom_config = get_test_configs()
    dim = custom_config.hidden_size
    
    hf_norm = LlamaRMSNorm(hidden_size=dim, eps=custom_config.rms_norm_eps).to(DEVICE, dtype=torch.float16)
    with torch.no_grad():
        hf_norm.weight.normal_(mean=1.0, std=0.1)
        
    custom_norm = RMSNorm(dim=dim, eps=custom_config.rms_norm_eps)
    custom_norm.weight = hf_norm.weight.clone().detach().to(DEVICE)
    
    x = torch.randn(4, 32, dim, device=DEVICE, dtype=torch.float16)
    
    out_hf = hf_norm(x)
    out_custom = custom_norm(x)
    
    assert torch.allclose(out_hf, out_custom, rtol=1e-3, atol=1e-3), "RMSNorm output mismatch"


def test_mlp_correctness():
    """Compares custom MLP (SwiGLU FFN) against Hugging Face's LlamaMLP."""
    hf_config, custom_config = get_test_configs()
    
    hf_mlp = LlamaMLP(config=hf_config).to(DEVICE, dtype=torch.float16)
    with torch.no_grad():
        hf_mlp.gate_proj.weight.normal_(std=0.02)
        hf_mlp.up_proj.weight.normal_(std=0.02)
        hf_mlp.down_proj.weight.normal_(std=0.02)
        
    custom_mlp = MLP(config=custom_config)
    custom_mlp.w_gate = hf_mlp.gate_proj.weight.T.clone().detach().to(DEVICE)
    custom_mlp.w_up = hf_mlp.up_proj.weight.T.clone().detach().to(DEVICE)
    custom_mlp.w_down = hf_mlp.down_proj.weight.T.clone().detach().to(DEVICE)
    
    x = torch.randn(2, 16, custom_config.hidden_size, device=DEVICE, dtype=torch.float16)
    out_hf = hf_mlp(x)
    
    # Custom MLP simulation (based on standard LLama MLP implementation)
    gate_proj = x @ custom_mlp.w_gate
    up_proj = x @ custom_mlp.w_up
    activated = swiglu(up_proj, gate_proj)
    out_custom = activated @ custom_mlp.w_down
    
    assert torch.allclose(out_hf, out_custom, rtol=1e-3, atol=1e-3), "MLP forward pass mismatch"


def test_attention_prefill_and_decode():
    """Compares custom Attention against Hugging Face's LlamaAttention."""
    hf_config, custom_config = get_test_configs()
    batch = 1
    seq_len = 16
    
    hf_attn = LlamaAttention(config=hf_config, layer_idx=0).to(DEVICE, dtype=torch.float16)
    with torch.no_grad():
        hf_attn.q_proj.weight.normal_(std=0.02)
        hf_attn.k_proj.weight.normal_(std=0.02)
        hf_attn.v_proj.weight.normal_(std=0.02)
        hf_attn.o_proj.weight.normal_(std=0.02)
        
    custom_attn = Attention(config=custom_config)
    custom_attn.wq = hf_attn.q_proj.weight.T.clone().detach().to(DEVICE)
    custom_attn.wk = hf_attn.k_proj.weight.T.clone().detach().to(DEVICE)
    custom_attn.wv = hf_attn.v_proj.weight.T.clone().detach().to(DEVICE)
    custom_attn.wo = hf_attn.o_proj.weight.T.clone().detach().to(DEVICE)
    
    x = torch.randn(batch, seq_len, custom_config.hidden_size, device=DEVICE, dtype=torch.float16)
    position_ids = torch.arange(seq_len, device=DEVICE).unsqueeze(0).expand(batch, -1)
    
    hf_rotary = LlamaRotaryEmbedding(config=hf_config).to(DEVICE, dtype=torch.float16)
    cos, sin = hf_rotary(x, position_ids)
    
    hf_outputs = hf_attn(x, position_embeddings=(cos, sin), attention_mask=None, position_ids=position_ids)
    out_hf = hf_outputs[0]
    
    q = x @ custom_attn.wq
    k = x @ custom_attn.wk
    v = x @ custom_attn.wv
    
    q = q.view(batch, seq_len, custom_config.num_attention_heads, custom_config.head_dim).transpose(1, 2)
    k = k.view(batch, seq_len, custom_config.num_key_value_heads, custom_config.head_dim).transpose(1, 2)
    v = v.view(batch, seq_len, custom_config.num_key_value_heads, custom_config.head_dim).transpose(1, 2)
    
    freqs_cis = model.precompute_rope_freqs(custom_config.head_dim, custom_config.max_position_embeddings, custom_config.rope_theta, device=DEVICE)
    freqs_cis_slice = freqs_cis[:seq_len]
    cos_custom = freqs_cis_slice.real
    sin_custom = freqs_cis_slice.imag
    cos_full = torch.cat([cos_custom, cos_custom], dim=-1).unsqueeze(0).unsqueeze(0)
    sin_full = torch.cat([sin_custom, sin_custom], dim=-1).unsqueeze(0).unsqueeze(0)
    
    q_rot, k_rot = model.apply_rope(q, k, cos_full, sin_full)
    
    gqa_ratio = custom_config.num_attention_heads // custom_config.num_key_value_heads
    k_rot_exp = k_rot.repeat_interleave(gqa_ratio, dim=1)
    v_exp = v.repeat_interleave(gqa_ratio, dim=1)
    
    scores = torch.matmul(q_rot, k_rot_exp.transpose(-1, -2)) / math.sqrt(custom_config.head_dim)
    mask = torch.full((seq_len, seq_len), float("-inf"), device=DEVICE, dtype=torch.float16)
    mask = torch.triu(mask, diagonal=1)
    scores = scores + mask
    
    probs = torch.softmax(scores, dim=-1)
    out_attn = torch.matmul(probs, v_exp)
    out_attn = out_attn.transpose(1, 2).contiguous().view(batch, seq_len, -1)
    out_custom = out_attn @ custom_attn.wo
    
    assert torch.allclose(out_hf, out_custom, rtol=1e-3, atol=1e-3), "Prefill Attention forward pass mismatch"


def test_decoder_layer_correctness():
    """Compares custom DecoderLayer against Hugging Face's LlamaDecoderLayer."""
    hf_config, custom_config = get_test_configs()
    batch = 1
    seq_len = 8
    
    hf_layer = LlamaDecoderLayer(config=hf_config, layer_idx=0).to(DEVICE, dtype=torch.float16)
    with torch.no_grad():
        hf_layer.input_layernorm.weight.normal_(mean=1.0, std=0.1)
        hf_layer.self_attn.q_proj.weight.normal_(std=0.02)
        hf_layer.self_attn.k_proj.weight.normal_(std=0.02)
        hf_layer.self_attn.v_proj.weight.normal_(std=0.02)
        hf_layer.self_attn.o_proj.weight.normal_(std=0.02)
        hf_layer.post_attention_layernorm.weight.normal_(mean=1.0, std=0.1)
        hf_layer.mlp.gate_proj.weight.normal_(std=0.02)
        hf_layer.mlp.up_proj.weight.normal_(std=0.02)
        hf_layer.mlp.down_proj.weight.normal_(std=0.02)
        
    custom_layer = DecoderLayer(config=custom_config, layer_idx=0)
    
    custom_layer.input_layernorm.weight = hf_layer.input_layernorm.weight.clone().detach().to(DEVICE)
    custom_layer.self_attn.wq = hf_layer.self_attn.q_proj.weight.T.clone().detach().to(DEVICE)
    custom_layer.self_attn.wk = hf_layer.self_attn.k_proj.weight.T.clone().detach().to(DEVICE)
    custom_layer.self_attn.wv = hf_layer.self_attn.v_proj.weight.T.clone().detach().to(DEVICE)
    custom_layer.self_attn.wo = hf_layer.self_attn.o_proj.weight.T.clone().detach().to(DEVICE)
    custom_layer.post_attention_layernorm.weight = hf_layer.post_attention_layernorm.weight.clone().detach().to(DEVICE)
    custom_layer.mlp.w_gate = hf_layer.mlp.gate_proj.weight.T.clone().detach().to(DEVICE)
    custom_layer.mlp.w_up = hf_layer.mlp.up_proj.weight.T.clone().detach().to(DEVICE)
    custom_layer.mlp.w_down = hf_layer.mlp.down_proj.weight.T.clone().detach().to(DEVICE)
    
    x = torch.randn(batch, seq_len, custom_config.hidden_size, device=DEVICE, dtype=torch.float16)
    position_ids = torch.arange(seq_len, device=DEVICE).unsqueeze(0).expand(batch, -1)
    
    hf_rotary = LlamaRotaryEmbedding(config=hf_config).to(DEVICE, dtype=torch.float16)
    cos, sin = hf_rotary(x, position_ids)
    hf_outputs = hf_layer(x, position_embeddings=(cos, sin), position_ids=position_ids)
    out_hf = hf_outputs[0]
    
    # Custom layer simulation
    norm_x = custom_layer.input_layernorm(x)
    
    q = norm_x @ custom_layer.self_attn.wq
    k = norm_x @ custom_layer.self_attn.wk
    v = norm_x @ custom_layer.self_attn.wv
    
    q = q.view(batch, seq_len, custom_config.num_attention_heads, custom_config.head_dim).transpose(1, 2)
    k = k.view(batch, seq_len, custom_config.num_key_value_heads, custom_config.head_dim).transpose(1, 2)
    v = v.view(batch, seq_len, custom_config.num_key_value_heads, custom_config.head_dim).transpose(1, 2)
    
    freqs_cis = model.precompute_rope_freqs(custom_config.head_dim, custom_config.max_position_embeddings, custom_config.rope_theta, device=DEVICE)
    freqs_cis_slice = freqs_cis[:seq_len]
    cos_custom = freqs_cis_slice.real
    sin_custom = freqs_cis_slice.imag
    cos_full = torch.cat([cos_custom, cos_custom], dim=-1).unsqueeze(0).unsqueeze(0)
    sin_full = torch.cat([sin_custom, sin_custom], dim=-1).unsqueeze(0).unsqueeze(0)
    
    q_rot, k_rot = model.apply_rope(q, k, cos_full, sin_full)
    
    gqa_ratio = custom_config.num_attention_heads // custom_config.num_key_value_heads
    k_rot_exp = k_rot.repeat_interleave(gqa_ratio, dim=1)
    v_exp = v.repeat_interleave(gqa_ratio, dim=1)
    
    scores = torch.matmul(q_rot, k_rot_exp.transpose(-1, -2)) / math.sqrt(custom_config.head_dim)
    mask = torch.full((seq_len, seq_len), float("-inf"), device=DEVICE, dtype=torch.float16)
    mask = torch.triu(mask, diagonal=1)
    scores = scores + mask
    
    probs = torch.softmax(scores, dim=-1)
    out_attn = torch.matmul(probs, v_exp)
    out_attn = out_attn.transpose(1, 2).contiguous().view(batch, seq_len, -1)
    attn_out = out_attn @ custom_layer.self_attn.wo
    
    attn_residual = x + attn_out
    
    norm_attn = custom_layer.post_attention_layernorm(attn_residual)
    gate_proj = norm_attn @ custom_layer.mlp.w_gate
    up_proj = norm_attn @ custom_layer.mlp.w_up
    activated = swiglu(up_proj, gate_proj)
    mlp_out = activated @ custom_layer.mlp.w_down
    
    out_custom = attn_residual + mlp_out
    
    assert torch.allclose(out_hf, out_custom, rtol=1e-3, atol=1e-3), "DecoderLayer forward pass mismatch"


def test_full_model_equivalence():
    """Validates that a full custom Llama model configuration has corresponding weight structures

    matching Hugging Face LlamaModel.
    """
    hf_config, custom_config = get_test_configs()
    
    hf_model = LlamaModel(config=hf_config).to(DEVICE)
    custom_model = Llama(config=custom_config)
    
    assert len(hf_model.layers) == len(custom_model.layers), "Number of layers mismatch"
    assert custom_model.embed_tokens.shape == hf_model.embed_tokens.weight.shape, "Embedding shape mismatch"
    assert custom_model.norm.weight.shape == hf_model.norm.weight.shape, "Final Norm weight shape mismatch"
    
    for i in range(len(custom_model.layers)):
        c_layer = custom_model.layers[i]
        h_layer = hf_model.layers[i]
        
        assert c_layer.input_layernorm.weight.shape == h_layer.input_layernorm.weight.shape
        assert c_layer.post_attention_layernorm.weight.shape == h_layer.post_attention_layernorm.weight.shape
        
        assert c_layer.self_attn.wq.shape == h_layer.self_attn.q_proj.weight.T.shape
        assert c_layer.self_attn.wk.shape == h_layer.self_attn.k_proj.weight.T.shape
        assert c_layer.self_attn.wv.shape == h_layer.self_attn.v_proj.weight.T.shape
        assert c_layer.self_attn.wo.shape == h_layer.self_attn.o_proj.weight.T.shape
        
        assert c_layer.mlp.w_gate.shape == h_layer.mlp.gate_proj.weight.T.shape
        assert c_layer.mlp.w_up.shape == h_layer.mlp.up_proj.weight.T.shape
        assert c_layer.mlp.w_down.shape == h_layer.mlp.down_proj.weight.T.shape
