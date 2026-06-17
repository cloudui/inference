"""
Standalone GPU Correctness Test for custom Attention class in model.py

This script compares the custom `Attention` module against Hugging Face's `LlamaAttention`
on a CUDA device, executing the Triton flash-decode kernel.

To run:
    python tests/test_attention.py
"""

import math
import torch
import transformers
from transformers import LlamaConfig as HFLlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRotaryEmbedding

# Import your custom modules
import model
from model import LlamaConfig, Attention, precompute_rope_freqs

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_attention_test():
    print(f"Running custom Attention test on device: {DEVICE}")
    if DEVICE.type == "cpu":
        print("WARNING: Running on CPU. Triton kernels require a CUDA GPU to compile and execute.")

    # 1. Scaled-down configurations
    hidden_size = 256
    num_attention_heads = 8
    num_key_value_heads = 2
    head_dim = hidden_size // num_attention_heads
    max_seq_len = 512
    
    hf_config = HFLlamaConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=max_seq_len,
        attn_implementation="sdpa",
        rope_theta=10000.0,
    )
    
    custom_config = LlamaConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=max_seq_len,
        head_dim=head_dim,
        rope_theta=10000.0,
    )

    # 2. Instantiate Hugging Face LlamaAttention (cast to float16)
    hf_attn = LlamaAttention(config=hf_config, layer_idx=0).to(device=DEVICE, dtype=torch.float16)
    with torch.no_grad():
        hf_attn.q_proj.weight.normal_(std=0.02)
        hf_attn.k_proj.weight.normal_(std=0.02)
        hf_attn.v_proj.weight.normal_(std=0.02)
        hf_attn.o_proj.weight.normal_(std=0.02)

    # 3. Instantiate Custom Attention
    custom_attn = Attention(config=custom_config)
    
    # Copy, transpose and concatenate QKV weights
    wq = hf_attn.q_proj.weight.clone().detach().to(DEVICE)
    wk = hf_attn.k_proj.weight.clone().detach().to(DEVICE)
    wv = hf_attn.v_proj.weight.clone().detach().to(DEVICE)
    custom_attn.wqkv = torch.cat((wq, wk, wv), dim=0)
    custom_attn.wo = hf_attn.o_proj.weight.clone().detach().to(DEVICE)

    # 4. Set up Inputs
    batch_size = 1
    seq_len = 1  # Testing a single decode step
    cache_position = 4  # Assume we are at position 4 in the sequence
    
    # Input tensor shape: (batch_size, seq_len, hidden_size) in float16
    x = torch.randn(batch_size, seq_len, hidden_size, device=DEVICE, dtype=torch.float16)
    
    # Precompute RoPE tables
    freqs_cis = precompute_rope_freqs(
        head_dim=head_dim,
        max_seq_len=max_seq_len,
        theta=custom_config.rope_theta,
        device=DEVICE
    )
    cos_table = freqs_cis.real.contiguous()
    sin_table = freqs_cis.imag.contiguous()

    # Set up KV cache (batch, num_kv_heads, max_seq_len, head_dim) in float16
    # Prefill some random values in positions 0..3 of the cache
    k_cache = torch.zeros(batch_size, num_key_value_heads, max_seq_len, head_dim, device=DEVICE, dtype=torch.float16)
    v_cache = torch.zeros(batch_size, num_key_value_heads, max_seq_len, head_dim, device=DEVICE, dtype=torch.float16)
    k_cache[:, :, :cache_position] = torch.randn(batch_size, num_key_value_heads, cache_position, head_dim, device=DEVICE, dtype=torch.float16)
    v_cache[:, :, :cache_position] = torch.randn(batch_size, num_key_value_heads, cache_position, head_dim, device=DEVICE, dtype=torch.float16)
    kv_cache = (k_cache, v_cache)

    # 5. Run Hugging Face LlamaAttention (Reference)
    # Hugging Face manages KV caches using DynamicCache in newer versions.
    from transformers.cache_utils import DynamicCache
    hf_past = DynamicCache()
    hf_past.update(
        k_cache[:, :, :cache_position].clone().detach(),
        v_cache[:, :, :cache_position].clone().detach(),
        layer_idx=0
    )
    
    # Generate HF position embeddings (cast to float16)
    hf_rotary = LlamaRotaryEmbedding(config=hf_config).to(device=DEVICE, dtype=torch.float16)
    # position_ids representing current step index (4)
    position_ids = torch.tensor([[cache_position]], device=DEVICE)
    cos, sin = hf_rotary(x, position_ids)
    
    # Run HF forward
    with torch.inference_mode():
        hf_out, hf_updated_past = hf_attn(
            hidden_states=x,
            position_embeddings=(cos, sin),
            attention_mask=None,
            past_key_values=hf_past,
            use_cache=True,
            position_ids=position_ids
        )
    print("Hugging Face forward pass completed.")

    # 6. Run Custom Attention
    # Note: Your Attention class currently has some signature and projection shape mismatches, 
    # such as passing `freqs_cis` directly to `apply_rope(x @ self.wq, freqs_cis)` inside Attention.__call__, 
    # where apply_rope expects (q, k, cos, sin). 
    # You will need to align your Attention.__call__ implementation before running this test.
    with torch.inference_mode():
        custom_out = custom_attn(
            x=x,
            rope_embeds=(cos_table, sin_table),
            kv_cache=kv_cache,
            cache_position=cache_position
        )
    print("Custom Attention forward pass completed.")
    
    # 7. Assert correctness
    print("\nComparing outputs...")
    abs_diff = torch.abs(hf_out - custom_out)
    diff = abs_diff.max().item()
    print(f"Max absolute difference: {diff:.6e}")
    print(f"Mean absolute difference: {abs_diff.mean().item():.6e}")
    print(f"Median absolute difference: {abs_diff.median().item():.6e}")
    
    # We use a standard tolerance for float16 operations (atol=5e-3)
    if torch.allclose(hf_out, custom_out, atol=5e-3):
        print("SUCCESS: Custom Attention matches Hugging Face!")
    else:
        print("FAILURE: Mismatch found between Custom and Hugging Face outputs.")


if __name__ == "__main__":
    run_attention_test()
