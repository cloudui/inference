import argparse
import math
import torch
import torch.nn.functional as F
import triton.testing

from kernels.flash_decode import flash_decode_out


def run_benchmark(batch_size, num_q_heads, num_kv_heads, head_dim, seq_lens, save_plot=False):
    device = 'cuda'
    dtype = torch.float16
    gqa_ratio = num_q_heads // num_kv_heads

    print(f"| {'Sequence Length':<15} | {'Triton Pre-alloc (ms)':<22} | {'Triton Dynamic (ms)':<20} | {'SDPA + GQA (ms)':<22} | {'Triton vs SDPA GQA (x)':<24} |")
    print(f"|{'-'*17}|{'-'*24}|{'-'*22}|{'-'*24}|{'-'*26}|")

    results = {
        'seq_len': [],
        'triton_preallocated': [],
        'triton_dynamic': [],
        'pytorch_sdpa_gqa': [],
    }

    for seq_len in seq_lens:
        # Generate inputs
        q = torch.randn(batch_size, num_q_heads, 1, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device, dtype=dtype)

        # Pre-allocate Triton buffers
        min_block_seq_kv = 32
        n_blocks_max = (seq_len + min_block_seq_kv - 1) // min_block_seq_kv
        mid_o = torch.empty((batch_size, num_kv_heads, n_blocks_max, gqa_ratio, head_dim), device=device, dtype=dtype)
        mid_lse = torch.empty((batch_size, num_kv_heads, n_blocks_max, gqa_ratio), device=device, dtype=torch.float32)
        out = torch.empty((batch_size, num_q_heads, 1, head_dim), device=device, dtype=dtype)

        # 1. Triton Pre-allocated
        triton_pre_ms = triton.testing.do_bench(
            lambda: flash_decode_out(q, k, v, seq_len, mid_o, mid_lse, out)
        )

        # 2. Triton Dynamic
        def run_dynamic():
            n_blocks = (seq_len + 31) // 32
            mo = torch.empty((batch_size, num_kv_heads, n_blocks, gqa_ratio, head_dim), device=device, dtype=dtype)
            mlse = torch.empty((batch_size, num_kv_heads, n_blocks, gqa_ratio), device=device, dtype=torch.float32)
            o = torch.empty((batch_size, num_q_heads, 1, head_dim), device=device, dtype=dtype)
            flash_decode_out(q, k, v, seq_len, mo, mlse, o)
            return o
        triton_dyn_ms = triton.testing.do_bench(run_dynamic)

        # 3. PyTorch SDPA with native GQA (enable_gqa=True)
        sdpa_gqa_ms = triton.testing.do_bench(
            lambda: F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=True)
        )

        speedup_triton_vs_sdpa_gqa = sdpa_gqa_ms / triton_pre_ms

        print(f"| {seq_len:<15} | {triton_pre_ms:>22.4f} | {triton_dyn_ms:>20.4f} | {sdpa_gqa_ms:>22.4f} | {speedup_triton_vs_sdpa_gqa:>23.2f}x |")

        results['seq_len'].append(seq_len)
        results['triton_preallocated'].append(triton_pre_ms)
        results['triton_dynamic'].append(triton_dyn_ms)
        results['pytorch_sdpa_gqa'].append(sdpa_gqa_ms)

    if save_plot:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            
            plt.plot(results['seq_len'], results['triton_preallocated'], label='Triton Flash-Decode (Pre-allocated)', color='blue', linestyle='-', marker='o')
            plt.plot(results['seq_len'], results['triton_dynamic'], label='Triton Flash-Decode (Dynamic)', color='cyan', linestyle='--', marker='x')
            plt.plot(results['seq_len'], results['pytorch_sdpa_gqa'], label='PyTorch SDPA (enable_gqa=True)', color='green', linestyle='-', marker='d')
            
            plt.xscale('log', base=2)
            plt.yscale('log')
            plt.xlabel('Sequence Length (KV Cache Size)')
            plt.ylabel('Time (ms)')
            plt.title('Decode Attention Performance Comparison (Lower is Better)')
            plt.legend()
            plt.grid(True, which="both", ls="-", alpha=0.2)
            plt.savefig('decode_bench.png', dpi=300)
            print("\nPlot saved successfully to decode_bench.png")
        except ImportError:
            print("\nmatplotlib is not installed. Skipping plot saving.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark custom flash decode triton kernel vs PyTorch SDPA")
    parser.add_argument("--save-plot", action="store_true", help="Save plot to decode_bench.png")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for benchmark")
    parser.add_argument("--q-heads", type=int, default=32, help="Number of Query heads")
    parser.add_argument("--kv-heads", type=int, default=8, help="Number of KV heads")
    parser.add_argument("--head-dim", type=int, default=128, help="Head dimension")
    args = parser.parse_args()
    
    print("=========================================================================")
    print("  Flash-Decode vs PyTorch SDPA Kernel Benchmark")
    print(f"  Batch Size: {args.batch_size} | Q Heads: {args.q_heads} | KV Heads: {args.kv_heads} | Head Dim: {args.head_dim}")
    print("=========================================================================\n")
    
    seq_lens = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    run_benchmark(args.batch_size, args.q_heads, args.kv_heads, args.head_dim, seq_lens, args.save_plot)
