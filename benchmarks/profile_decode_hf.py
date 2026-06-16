"""
profile_decode_hf.py — Decode-step profiler for the Hugging Face Llama implementation.

Usage:
    python benchmarks/profile_decode_hf.py [--seq-len N] [--steps K] [--dtype float16] [--compiled] [--export-chrome]
"""

import argparse
import time
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.cache_utils import DynamicCache


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seq-len",      type=int, default=512,
                   help="Number of KV tokens already in the cache (simulates mid-sequence decode)")
    p.add_argument("--steps",        type=int, default=3,
                    help="Decode steps to profile (first is warmup, rest are measured)")
    p.add_argument("--export-chrome", action="store_true",
                   help="Write a Chrome trace to ./profile_trace/ for chrome://tracing")
    p.add_argument("--trace-name", type=str, default=None,
                   help="Custom filename or path for the exported Chrome trace")
    p.add_argument("--batch-size",   type=int, default=1)
    p.add_argument("--dtype",        type=str, default="float16", choices=["float16", "bfloat16"])
    p.add_argument("--compiled",     action="store_true",
                   help="Compile the model using torch.compile")
    p.add_argument("--small",        action="store_true",
                   help="Use a tiny 2-layer config for fast iteration")
    return p.parse_args()


def build_model(args, device, dtype) -> tuple[LlamaForCausalLM, DynamicCache, LlamaConfig]:
    if args.small:
        cfg = LlamaConfig(
            hidden_size=512,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=2,
            intermediate_size=1024,
            vocab_size=1024,
            max_position_embeddings=2048,
            rms_norm_eps=1e-5,
            rope_theta=10000.0,
            attn_implementation="sdpa",
        )
    else:
        cfg = LlamaConfig(
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            intermediate_size=14336,
            vocab_size=128256,
            max_position_embeddings=8192,
            rms_norm_eps=1e-5,
            rope_theta=500000.0,
            attn_implementation="sdpa",
        )

    print(f"Initializing HF LlamaModel on {device} ({args.dtype})...")
    # Instantiate directly on GPU in target precision
    old_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    with torch.device(device):
        model = LlamaForCausalLM(cfg)
    torch.set_default_dtype(old_default_dtype)
    model.eval()

    # Pre-fill KV cache using DynamicCache
    hf_cache = DynamicCache()
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    
    if args.seq_len > 0:
        print(f"Pre-filling KV cache to seq_len={args.seq_len}...")
        for layer_idx in range(cfg.num_hidden_layers):
            k = torch.randn(args.batch_size, cfg.num_key_value_heads, args.seq_len, head_dim, device=device, dtype=dtype) * 0.02
            v = torch.randn(args.batch_size, cfg.num_key_value_heads, args.seq_len, head_dim, device=device, dtype=dtype) * 0.02
            hf_cache.update(k, v, layer_idx)

    if args.compiled:
        print("Compiling HF model (this may take a few minutes)...")
        model = torch.compile(model)

    return model, hf_cache, cfg


@torch.inference_mode()
def profiled_forward(model, token_ids, start_pos, kv_cache, device):
    position_ids = torch.tensor([[start_pos]], device=device)
    return model(input_ids=token_ids, past_key_values=kv_cache, use_cache=True, position_ids=position_ids)


def cuda_timed(fn, *args, warmup=3, iters=10, **kwargs):
    """Returns median CUDA wall-clock in ms over `iters` runs after `warmup`."""
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()

    times = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event   = torch.cuda.Event(enable_timing=True)

    for _ in range(iters):
        start_event.record()
        fn(*args, **kwargs)
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))

    times.sort()
    return {
        "min_ms":    times[0],
        "median_ms": times[len(times) // 2],
        "max_ms":    times[-1],
    }


def estimate_decode_bandwidth(cfg: LlamaConfig, batch: int, seq_len: int, dtype_bytes: int) -> dict:
    B = batch
    H  = cfg.hidden_size
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    Hkv = cfg.num_key_value_heads * head_dim
    Hq  = cfg.num_attention_heads * head_dim
    I  = cfg.intermediate_size
    S  = seq_len

    weights_per_layer = (
        H * Hq   # wq
      + H * Hkv  # wk
      + H * Hkv  # wv
      + Hq * H   # wo
      + H * I    # w_gate
      + H * I    # w_up
      + I * H    # w_down
      + H * 2    # 2x rms norm weights
    ) * dtype_bytes

    kv_cache_per_layer = 2 * Hkv * S * dtype_bytes  # K + V read

    total_weights    = weights_per_layer * cfg.num_hidden_layers
    total_kv_read    = kv_cache_per_layer * cfg.num_hidden_layers
    embed_and_head   = (cfg.vocab_size * H * 2) * dtype_bytes  # embed + lm_head

    total_bytes = total_weights + total_kv_read + embed_and_head

    l40s_bw_gbs = 864.0  # peak HBM bandwidth (GB/s)
    theoretical_ms = (total_bytes / 1e9) / l40s_bw_gbs * 1e3

    return {
        "weights_GB":          total_weights / 1e9,
        "kv_cache_read_GB":    total_kv_read / 1e9,
        "total_HBM_traffic_GB": total_bytes  / 1e9,
        "roofline_ms":         theoretical_ms,
    }


def main():
    args = parse_args()
    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    dtype_bytes = 2

    print(f"\n{'='*70}")
    print(f"  HF Llama Decode Profiler  |  seq_len={args.seq_len}  batch={args.batch_size}")
    print(f"  dtype={args.dtype}  compiled={args.compiled}")
    print(f"{'='*70}\n")

    model, kv_cache, cfg = build_model(args, device, dtype)

    token_ids = torch.zeros(args.batch_size, 1, dtype=torch.long, device=device)
    torch.cuda.synchronize()

    # ── 2. Raw CUDA timing (no profiler overhead) ─────────────────────────────
    print("─── Raw CUDA timing (warmup=3, iters=10) ───────────────────────────")
    timing = cuda_timed(
        profiled_forward,
        model, token_ids, args.seq_len, kv_cache, device,
        warmup=3, iters=10,
    )
    print(f"  Min:    {timing['min_ms']:.3f} ms")
    print(f"  Median: {timing['median_ms']:.3f} ms")
    print(f"  Max:    {timing['max_ms']:.3f} ms")
    print()

    # ── 3. Roofline estimate ──────────────────────────────────────────────────
    bw = estimate_decode_bandwidth(cfg, args.batch_size, args.seq_len, dtype_bytes)
    print("─── HBM bandwidth estimate (L40S roofline) ─────────────────────────")
    print(f"  Weight traffic:    {bw['weights_GB']:.2f} GB")
    print(f"  KV cache traffic:  {bw['kv_cache_read_GB']:.2f} GB  (seq_len={args.seq_len})")
    print(f"  Total HBM:         {bw['total_HBM_traffic_GB']:.2f} GB")
    print(f"  Roofline bound:    {bw['roofline_ms']:.2f} ms  (at 864 GB/s peak)")
    if timing['median_ms'] > 0:
        achieved_bw = bw['total_HBM_traffic_GB'] / (timing['median_ms'] / 1e3)
        print(f"  Achieved ~BW:      {achieved_bw:.0f} GB/s  ({achieved_bw/864*100:.0f}% of peak)")
    print()

    # ── 4. torch.profiler trace ───────────────────────────────────────────────
    print("─── Running torch.profiler (1 warmup + 2 active steps) ─────────────")

    schedule = torch.profiler.schedule(wait=0, warmup=1, active=2, repeat=1)

    prof_kwargs = dict(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule,
        record_shapes=True,
        with_stack=False,
        profile_memory=True,
        with_flops=True,
    )

    with profile(**prof_kwargs) as prof:
        for step in range(3):
            profiled_forward(model, token_ids, args.seq_len, kv_cache, device)
            prof.step()

    if args.export_chrome:
        import os
        import datetime
        os.makedirs("./profile_trace", exist_ok=True)
        if args.trace_name:
            trace_path = args.trace_name
            if not os.path.dirname(trace_path):
                trace_path = os.path.join("./profile_trace", trace_path)
            if not (trace_path.endswith(".json") or trace_path.endswith(".gz")):
                trace_path += ".pt.trace.json"
        else:
            run_type = "hf_small" if args.small else "hf_large"
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            trace_path = f"./profile_trace/trace_{run_type}_{timestamp}.pt.trace.json"
        
        prof.export_chrome_trace(trace_path)
        print(f"\nChrome trace written to {trace_path}")
        print("Open chrome://tracing and load the JSON file there.")

    # ── 5. Print summary table ────────────────────────────────────────────────
    print("\n─── Top-20 ops by self-CUDA time ────────────────────────────────────")
    print(prof.key_averages(group_by_input_shape=False).table(
        sort_by="self_cuda_time_total",
        row_limit=20,
    ))


if __name__ == "__main__":
    main()
