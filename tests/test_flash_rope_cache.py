import pytest
import torch
import math
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from transformers import LlamaConfig as HFLlamaConfig

from kernels.fused_rope_cache import fused_rope_cache_decode_out
from kernels.rope import apply_rope_decode_out

DEVICE = torch.device("cuda")

@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("cache_position", [0, 1, 8, 31, 127])
@pytest.mark.parametrize("num_heads,num_kv_heads,head_dim", [
    (8, 2, 64),    # standard GQA
    (32, 8, 128),  # LLaMA 3 8B
    (8, 8, 64),    # MHA
])
def test_fused_rope_cache_vs_original(batch_size, cache_position, num_heads, num_kv_heads, head_dim):
    """
    Compare fused_rope_cache_decode_out against the original implementation:
    apply_rope_decode_out on Q and K, and manually copying to KV caches.
    """
    torch.manual_seed(42)
    dtype = torch.float16
    max_seq_len = 512
    
    # 1. Allocate inputs
    q_dim = num_heads * head_dim
    kv_dim = num_kv_heads * head_dim
    qkv_concat_dim = q_dim + 2 * kv_dim
    
    # Input projections (batch, 1, qkv_concat_dim)
    qkv_proj = torch.randn(batch_size, 1, qkv_concat_dim, device=DEVICE, dtype=dtype)
    
    # Cos/Sin RoPE embeddings (1, head_dim // 2) for this step
    hf_config = HFLlamaConfig(
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        hidden_size=num_heads * head_dim,
        rope_theta=10000.0,
    )
    rotary = LlamaRotaryEmbedding(config=hf_config).to(DEVICE, dtype=dtype)
    dummy_x = torch.zeros(batch_size, 1, num_heads * head_dim, device=DEVICE, dtype=dtype)
    position_ids = torch.tensor([[cache_position]], device=DEVICE)
    cos_hf, sin_hf = rotary(dummy_x, position_ids)
    
    # Extract cos, sin of shape (1, head_dim // 2) to match model.py structure robustly
    cos = cos_hf.flatten()[:head_dim // 2].unsqueeze(0).contiguous()
    sin = sin_hf.flatten()[:head_dim // 2].unsqueeze(0).contiguous()
    
    # Allocate output tensors and caches for "original" approach
    q_out_orig = torch.empty(batch_size, num_heads, 1, head_dim, device=DEVICE, dtype=dtype)
    k_cache_orig = torch.zeros(batch_size, num_kv_heads, max_seq_len, head_dim, device=DEVICE, dtype=dtype)
    v_cache_orig = torch.zeros(batch_size, num_kv_heads, max_seq_len, head_dim, device=DEVICE, dtype=dtype)
    
    # Set up some dummy values in the cache history to ensure they aren't modified
    if cache_position > 0:
        k_cache_orig[:, :, :cache_position] = torch.randn(batch_size, num_kv_heads, cache_position, head_dim, device=DEVICE, dtype=dtype)
        v_cache_orig[:, :, :cache_position] = torch.randn(batch_size, num_kv_heads, cache_position, head_dim, device=DEVICE, dtype=dtype)
        
    # Allocate identical copies for the fused/flash kernel
    q_out_fused = torch.empty(batch_size, num_heads, 1, head_dim, device=DEVICE, dtype=dtype)
    k_cache_fused = k_cache_orig.clone()
    v_cache_fused = v_cache_orig.clone()
    
    # --- Step 1: Run original implementation ("what we had before") ---
    q, k, v = torch.split(qkv_proj, [q_dim, kv_dim, kv_dim], dim=-1)
    
    hidden_shape = (batch_size, 1, -1, head_dim)
    q = q.view(hidden_shape).transpose(1, 2).contiguous()
    k = k.view(hidden_shape).transpose(1, 2).contiguous()
    v = v.view(hidden_shape).transpose(1, 2).contiguous()
    
    apply_rope_decode_out(q, cos, sin, q_out_orig)
    
    k_rope_out_orig = torch.empty(batch_size, num_kv_heads, 1, head_dim, device=DEVICE, dtype=dtype)
    apply_rope_decode_out(k, cos, sin, k_rope_out_orig)
    
    # Cache writes
    k_cache_orig[:, :, cache_position:cache_position+1] = k_rope_out_orig
    v_cache_orig[:, :, cache_position:cache_position+1] = v
    
    # --- Step 2: Run fused/flash kernel ---
    fused_rope_cache_decode_out(
        qkv_proj=qkv_proj,
        cos=cos,
        sin=sin,
        q_out=q_out_fused,
        k_cache=k_cache_fused,
        v_cache=v_cache_fused,
        cache_pos=cache_position,
    )
    
    # --- Step 3: Compare results ---
    # Compare Q outputs
    q_diff = (q_out_orig - q_out_fused).abs().max().item()
    assert torch.allclose(q_out_orig, q_out_fused, atol=2e-3), f"Q output mismatch: max_diff={q_diff:.2e}"
    
    # Compare K caches
    k_diff = (k_cache_orig - k_cache_fused).abs().max().item()
    assert torch.allclose(k_cache_orig, k_cache_fused, atol=2e-3), f"K cache mismatch: max_diff={k_diff:.2e}"
    
    # Compare V caches
    v_diff = (v_cache_orig - v_cache_fused).abs().max().item()
    assert torch.allclose(v_cache_orig, v_cache_fused, atol=2e-3), f"V cache mismatch: max_diff={v_diff:.2e}"


def test_fused_rope_cache_vs_hf():
    """
    Compare fused_rope_cache_decode_out against Hugging Face reference RoPE rotation and cache update.
    """
    torch.manual_seed(1234)
    dtype = torch.float16
    batch_size = 2
    num_heads = 8
    num_kv_heads = 2
    head_dim = 64
    cache_position = 5
    max_seq_len = 128
    
    q_dim = num_heads * head_dim
    kv_dim = num_kv_heads * head_dim
    qkv_concat_dim = q_dim + 2 * kv_dim
    
    qkv_proj = torch.randn(batch_size, 1, qkv_concat_dim, device=DEVICE, dtype=dtype)
    
    # HF configs
    hf_config = HFLlamaConfig(
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        hidden_size=num_heads * head_dim,
        rope_theta=10000.0,
    )
    rotary = LlamaRotaryEmbedding(config=hf_config).to(DEVICE, dtype=dtype)
    
    # Create input token states and position IDs
    dummy_x = torch.zeros(batch_size, 1, num_heads * head_dim, device=DEVICE, dtype=dtype)
    position_ids = torch.tensor([[cache_position]], device=DEVICE)
    
    # HF RoPE cos, sin (each is shape [1, 1, 1, head_dim] in decode)
    cos_hf, sin_hf = rotary(dummy_x, position_ids)
    
    # Under HF style, the rotation is applied as:
    # q_embed = (q * cos) + (rotate_half(q) * sin)
    # where rotate_half splits at head_dim // 2, negates the second half, and swaps them.
    
    # Prepare original/HF reference results
    q, k, v = torch.split(qkv_proj, [q_dim, kv_dim, kv_dim], dim=-1)
    
    hidden_shape = (batch_size, 1, -1, head_dim)
    q = q.view(hidden_shape).transpose(1, 2) # (batch, num_heads, 1, head_dim)
    k = k.view(hidden_shape).transpose(1, 2) # (batch, num_kv_heads, 1, head_dim)
    v = v.view(hidden_shape).transpose(1, 2) # (batch, num_kv_heads, 1, head_dim)
    
    # Apply HF rotation
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    # Prepare matching broadcast shapes for cos and sin
    cos_val = cos_hf.to(dtype)
    sin_val = sin_hf.to(dtype)
    
    q_hf_rotated = (q * cos_val) + (rotate_half(q) * sin_val)
    k_hf_rotated = (k * cos_val) + (rotate_half(k) * sin_val)
    
    # Reference cache
    k_cache_ref = torch.zeros(batch_size, num_kv_heads, max_seq_len, head_dim, device=DEVICE, dtype=dtype)
    v_cache_ref = torch.zeros(batch_size, num_kv_heads, max_seq_len, head_dim, device=DEVICE, dtype=dtype)
    k_cache_ref[:, :, cache_position:cache_position+1] = k_hf_rotated
    v_cache_ref[:, :, cache_position:cache_position+1] = v
    
    # Run the fused kernel
    q_out_fused = torch.empty(batch_size, num_heads, 1, head_dim, device=DEVICE, dtype=dtype)
    k_cache_fused = torch.zeros_like(k_cache_ref)
    v_cache_fused = torch.zeros_like(v_cache_ref)
    
    # The kernel needs cos and sin of shape (1, head_dim // 2)
    cos_kernel = cos_hf.flatten()[:head_dim // 2].unsqueeze(0).contiguous()
    sin_kernel = sin_hf.flatten()[:head_dim // 2].unsqueeze(0).contiguous()
    
    fused_rope_cache_decode_out(
        qkv_proj=qkv_proj,
        cos=cos_kernel,
        sin=sin_kernel,
        q_out=q_out_fused,
        k_cache=k_cache_fused,
        v_cache=v_cache_fused,
        cache_pos=cache_position,
    )
    
    # Compare
    q_diff = (q_hf_rotated - q_out_fused).abs().max().item()
    assert torch.allclose(q_hf_rotated, q_out_fused, atol=2e-3), f"HF Q output mismatch: max_diff={q_diff:.2e}"
    
    k_diff = (k_cache_ref - k_cache_fused).abs().max().item()
    assert torch.allclose(k_cache_ref, k_cache_fused, atol=2e-3), f"HF K cache mismatch: max_diff={k_diff:.2e}"
    
    v_diff = (v_cache_ref - v_cache_fused).abs().max().item()
    assert torch.allclose(v_cache_ref, v_cache_fused, atol=2e-3), f"HF V cache mismatch: max_diff={v_diff:.2e}"
