import argparse
import math
import torch
import torch.nn.functional as F
import triton
import triton.testing
import sys
import os

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from kernels.flash_decode import (
    flash_decode_out,
    flash_decode_generation_kernel,
    flash_decode_reduce_kernel,
    LOG2_E
)


def run_split_benchmark(batch_size, num_q_heads, num_kv_heads, head_dim, seq_lens, num_splits_list):
    device = 'cuda'
    dtype = torch.float16
    gqa_ratio = num_q_heads // num_kv_heads
    scale = (1 / math.sqrt(head_dim)) * LOG2_E

    print("=========================================================================")
    print("  Detailed Generation vs Reduction Performance Breakdown")
    print("=========================================================================\n")

    for seq_len in seq_lens:
        print(f"\n### Sequence Length: {seq_len}")
        print(f"| {'Num Splits':<12} | {'Gen Kernel (ms)':<18} | {'Reduce Kernel (ms)':<20} | {'Total Triton (ms)':<18} | {'PyTorch SDPA (ms)':<18} |")
        print(f"|{'-'*14}|{'-'*20}|{'-'*22}|{'-'*20}|{'-'*20}|")

        # Generate inputs
        q = torch.randn(batch_size, num_q_heads, 1, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device, dtype=dtype)
        out = torch.empty((batch_size, num_q_heads, 1, head_dim), device=device, dtype=dtype)

        # Baseline: PyTorch SDPA with native GQA
        sdpa_ms = triton.testing.do_bench(
            lambda: F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=True)
        )

        for num_splits in num_splits_list:
            # Pre-allocate Triton buffers for this split size
            mid_o = torch.empty((batch_size, num_kv_heads, num_splits, gqa_ratio, head_dim), device=device, dtype=dtype)
            mid_lse = torch.empty((batch_size, num_kv_heads, num_splits, gqa_ratio), device=device, dtype=torch.float32)

            stride_q_batch, stride_q_head, _, _ = q.stride()
            stride_k_batch, stride_k_head, stride_k_seq, _ = k.stride()
            stride_mid_o_batch, stride_mid_o_head, stride_mid_o_block, stride_mid_o_gqa, _ = mid_o.stride()
            stride_mid_lse_batch, stride_mid_lse_head, stride_mid_lse_block, stride_mid_lse_gqa = mid_lse.stride()
            stride_out_batch, stride_out_head, _, _ = out.stride()

            grid_gen = lambda meta: (num_splits, num_kv_heads, batch_size)
            grid_reduce = (num_q_heads, batch_size)

            # 1. Bench Generation Kernel Only
            gen_ms = triton.testing.do_bench(
                lambda: flash_decode_generation_kernel[grid_gen](
                    q, k, v, mid_o, mid_lse, seq_len, scale, head_dim,
                    stride_q_batch, stride_q_head, stride_k_batch, stride_k_head, stride_k_seq,
                    stride_mid_o_batch, stride_mid_o_head, stride_mid_o_block, stride_mid_o_gqa,
                    stride_mid_lse_batch, stride_mid_lse_head, stride_mid_lse_block, stride_mid_lse_gqa,
                    BLOCK_HEAD_DIM=head_dim, gqa_ratio=gqa_ratio, NUM_SPLITS=num_splits
                )
            )

            # 2. Bench Reduction Kernel Only
            reduce_ms = triton.testing.do_bench(
                lambda: flash_decode_reduce_kernel[grid_reduce](
                    mid_o, mid_lse, out, gqa_ratio, num_splits,
                    stride_mid_o_batch, stride_mid_o_head, stride_mid_o_gqa, stride_mid_o_block,
                    stride_mid_lse_batch, stride_mid_lse_head, stride_mid_lse_gqa, stride_mid_lse_block,
                    stride_out_batch, stride_out_head, head_dim
                )
            )

            # 3. Bench Total Triton (Gen + Reduce Combined)
            total_triton_ms = triton.testing.do_bench(
                lambda: flash_decode_out(q, k, v, seq_len, mid_o, mid_lse, out, num_splits=num_splits)
            )

            print(f"| {num_splits:<12} | {gen_ms:>18.4f} | {reduce_ms:>20.4f} | {total_triton_ms:>18.4f} | {sdpa_ms:>18.4f} |")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark Triton Flash-Decode Gen vs Reduce kernels")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--q-heads", type=int, default=32, help="Number of Query heads")
    parser.add_argument("--kv-heads", type=int, default=8, help="Number of KV heads")
    parser.add_argument("--head-dim", type=int, default=128, help="Head dimension")
    args = parser.parse_args()

    print("=========================================================================")
    print("  Flash-Decode Gen vs Reduce Kernel Benchmark")
    print(f"  Batch Size: {args.batch_size} | Q Heads: {args.q_heads} | KV Heads: {args.kv_heads} | Head Dim: {args.head_dim}")
    print("=========================================================================\n")

    seq_lens = [512, 2048, 8192, 32768, 131072]
    num_splits_list = [4, 8, 16, 32, 64, 128]
    run_split_benchmark(args.batch_size, args.q_heads, args.kv_heads, args.head_dim, seq_lens, num_splits_list)
