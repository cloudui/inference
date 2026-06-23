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
    head_dim = 128

    seq_lens = [128, 512, 2048, 4096, 8192]
    num_splits_list = [4, 8, 16, 32]

    all_passed = True

    for seq_len in seq_lens:
        for num_splits in num_splits_list:
            print(f"\nTesting dimensions: seq_len={seq_len}, num_splits={num_splits}")
            # Generate random input tensors
            q = torch.randn(batch_size, q_heads, 1, head_dim, device=device, dtype=torch.float16)
            k = torch.randn(batch_size, k_heads, seq_len, head_dim, device=device, dtype=torch.float16)
            v = torch.randn(batch_size, k_heads, seq_len, head_dim, device=device, dtype=torch.float16)

            # Run naive baseline (using float32 for higher precision reference)
            out_naive = pytorch_gqa_naive(q.float(), k.float(), v.float()).half()

            # Run our Triton flash_decode kernel
            out_triton = flash_decode(q, k, v, seq_len, num_splits=num_splits)

            # Check match
            max_diff = torch.abs(out_triton.float() - out_naive.float()).max().item()

            if torch.allclose(out_triton.float(), out_naive.float(), atol=1e-3):
                print(f"  SUCCESS: Triton Flash Decode outputs match PyTorch baseline perfectly (max diff: {max_diff:.6f})")
            else:
                print(f"  FAILURE: Outputs mismatch between Triton kernel and PyTorch baseline (max diff: {max_diff:.6f})")
                all_passed = False

    if all_passed:
        print("\nALL GPU SANITY CHECKS PASSED SUCCESSFULLY!")
    else:
        print("\nSOME GPU SANITY CHECKS FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    run_gpu_sanity_check()
