import argparse
import math
import torch
import torch.nn.functional as F
import triton.testing

from kernels.flash_decode import flash_decode_out

try:
    from flash_attn import flash_attn_with_kvcache
except ImportError:
    flash_attn_with_kvcache = None



def run_benchmark(batch_size, num_q_heads, num_kv_heads, head_dim, seq_lens, save_plot=False):
    device = 'cuda'
    dtype = torch.float16
    gqa_ratio = num_q_heads // num_kv_heads

    print(f"| {'Sequence Length':<15} | {'Triton (ms)':<20} | {'SDPA + GQA (ms)':<22} | {'Flash-Decode (ms)':<18} | {'Triton vs Flash-Decode (x)':<28} |")
    print(f"|{'-'*17}|{'-'*22}|{'-'*24}|{'-'*20}|{'-'*30}|")

    results = {
        'seq_len': [],
        'triton_preallocated': [],
        'pytorch_sdpa_gqa': [],
        'flash_decode': [],
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

        # 2. PyTorch SDPA with native GQA (enable_gqa=True)
        sdpa_gqa_ms = triton.testing.do_bench(
            lambda: F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=True)
        )

        # 3. Tri Dao's Flash Decode
        if flash_attn_with_kvcache is not None:
            # Inputs transposed to (batch, seqlen, heads, dim)
            q_fa = q.transpose(1, 2)
            k_fa = k.transpose(1, 2).contiguous()
            v_fa = v.transpose(1, 2).contiguous()
            cache_seqlens = torch.tensor([seq_len] * batch_size, dtype=torch.int32, device=device)
            
            flash_decode_ms = triton.testing.do_bench(
                lambda: flash_attn_with_kvcache(q_fa, k_fa, v_fa, cache_seqlens=cache_seqlens)
            )
        else:
            flash_decode_ms = float('nan')

        speedup_triton_vs_flash_decode = flash_decode_ms / triton_pre_ms if not math.isnan(flash_decode_ms) else float('nan')
        speedup_str = f"{speedup_triton_vs_flash_decode:>27.2f}x" if not math.isnan(speedup_triton_vs_flash_decode) else f"{'N/A':>28}"
        flash_decode_str = f"{flash_decode_ms:>18.4f}" if not math.isnan(flash_decode_ms) else f"{'N/A':>18}"

        print(f"| {seq_len:<15} | {triton_pre_ms:>20.4f} | {sdpa_gqa_ms:>22.4f} | {flash_decode_str} | {speedup_str} |")

        results['seq_len'].append(seq_len)
        results['triton_preallocated'].append(triton_pre_ms)
        results['pytorch_sdpa_gqa'].append(sdpa_gqa_ms)
        results['flash_decode'].append(flash_decode_ms)

    if save_plot:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            
            plt.plot(results['seq_len'], results['triton_preallocated'], label='Triton Flash-Decode', color='blue', linestyle='-', marker='o')
            plt.plot(results['seq_len'], results['pytorch_sdpa_gqa'], label='PyTorch SDPA (enable_gqa=True)', color='green', linestyle='-', marker='d')
            if not any(math.isnan(x) for x in results['flash_decode']):
                plt.plot(results['seq_len'], results['flash_decode'], label="Tri Dao's Flash-Decode", color='red', linestyle='-', marker='s')
            
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
