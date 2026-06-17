import torch
import torch.nn.functional as F
import sys
import os

# Add parent directory to sys.path so we can import from kernels
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kernels import flash_decode

import torch
import torch.nn.functional as F

def pytorch_gqa_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Expects shapes:
    q: (batch, q_heads, q_seq_len, head_dim)
    k: (batch, k_heads, kv_seq_len, head_dim)
    v: (batch, k_heads, kv_seq_len, head_dim)
    """
    # Native SDPA handles the head mismatch internally without explicit repeat_interleave
    output = F.scaled_dot_product_attention(
        query=q,
        key=k,
        value=v,
        attn_mask=None,
        dropout_p=0.0,       # 0.0 for inference
        is_causal=False, # Handles causal masking efficiently 
        enable_gqa=True     # Triggers FlashAttention/Math GQA kernels
    )
    return output

def pytorch_gqa_naive(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_heads = q.shape[1]
    k_heads = k.shape[1]
    head_dim = q.shape[-1]
    gqa_ratio = q_heads // k_heads
    seq_len = k.shape[2]
    
    k_expanded = k.repeat_interleave(gqa_ratio, dim=1)
    v_expanded = v.repeat_interleave(gqa_ratio, dim=1)
    
    scores = torch.matmul(q, k_expanded.transpose(-1, -2)) / (head_dim ** 0.5)

    probs = F.softmax(scores, dim=-1)
    output = torch.matmul(probs, v_expanded)
    
    return output


def run_gpu_sanity_check():
    print("Checking CUDA device availability...")
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. Please run this script in a GPU-enabled environment.")
        return

    device = torch.device("cuda")
    print(f"CUDA device detected: {torch.cuda.get_device_name(0)}")

    # Set up test dimensions
    batch_size = 4
    q_heads = 32
    k_heads = 8
    seq_len = 512
    head_dim = 128

    print(f"\nInitializing test tensors with dimensions:")
    print(f"  Batch Size: {batch_size}")
    print(f"  Query Heads: {q_heads}")
    print(f"  KV Heads: {k_heads} (GQA Ratio: {q_heads // k_heads})")
    print(f"  Sequence Length: {seq_len}")
    print(f"  Head Dimension: {head_dim}")

    # Generate random input tensors
    q = torch.randn(batch_size, q_heads, 1, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(batch_size, k_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(batch_size, k_heads, seq_len, head_dim, device=device, dtype=torch.float16)

    print("\nExecuting PyTorch naive GQA baseline...")
    # Run naive baseline (using float32 for higher precision reference)
    out_naive = pytorch_gqa_naive(q.float(), k.float(), v.float()).half()

    print("Executing Triton Flash Decode kernel...")
    # Run our Triton flash_decode kernel
    out_triton = flash_decode(q, k, v, seq_len)

    # Check match
    max_diff = torch.abs(out_triton.float() - out_naive.float()).max().item()
    print(f"\nMax absolute difference: {max_diff:.6f}")

    if torch.allclose(out_triton.float(), out_naive.float(), atol=1e-3):
        print("SUCCESS: Triton Flash Decode outputs match PyTorch baseline perfectly!")
    else:
        print("FAILURE: Outputs mismatch between Triton kernel and PyTorch baseline.")


if __name__ == "__main__":
    run_gpu_sanity_check()
