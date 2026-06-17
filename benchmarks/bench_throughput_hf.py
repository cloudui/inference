"""
bench_throughput_hf.py — Decode tok/s benchmark for the Hugging Face Llama implementation.

Usage:
    python benchmarks/bench_throughput_hf.py [--seq-len 512] [--decode-steps 128] [--small] [--batch-size 1] [--dtype float16] [--compiled]

Measures wall-clock tok/s for single-token decode steps using CUDA event timing.
KV cache is pre-filled with random data to simulate prior context.
"""

import argparse
import time
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.cache_utils import DynamicCache, StaticCache


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seq-len",      type=int, default=512,
                   help="KV tokens already in cache (simulates prior context)")
    p.add_argument("--decode-steps", type=int, default=128,
                   help="Number of decode steps to measure")
    p.add_argument("--warmup",       type=int, default=30,
                   help="Warmup decode steps (not measured)")
    p.add_argument("--batch-size",   type=int, default=1)
    p.add_argument("--dtype",        type=str, default="float16", choices=["float16", "bfloat16"])
    p.add_argument("--compiled",     action="store_true",
                   help="Compile the model using torch.compile")
    p.add_argument("--static-cache", action="store_true",
                   help="Use StaticCache instead of DynamicCache (fairer comparison with custom impl)")
    p.add_argument("--small",        action="store_true",
                   help="Use tiny 2-layer config for fast iteration")
    return p.parse_args()


def build_model(args, device, dtype):
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

    # KV cache setup
    head_dim = cfg.hidden_size // cfg.num_attention_heads

    if args.static_cache:
        # StaticCache: pre-allocated, no dynamic cat/copy overhead
        # Size to actual usage so SDPA only attends over filled positions
        max_cache_len = args.seq_len + args.warmup + args.decode_steps + 16
        hf_cache = StaticCache(config=cfg, max_cache_len=max_cache_len, batch_size=args.batch_size)
        # Pre-fill by running dummy forward steps
        if args.seq_len > 0:
            print(f"Pre-filling StaticCache to seq_len={args.seq_len}...")
            prefill_tokens = torch.zeros(args.batch_size, 1, dtype=torch.long, device=device)
            with torch.inference_mode():
                for i in range(args.seq_len):
                    position_ids = torch.tensor([[i]], device=device)
                    model(input_ids=prefill_tokens, past_key_values=hf_cache, use_cache=True, position_ids=position_ids)
    else:
        # DynamicCache: grows via cat/copy on each step
        hf_cache = DynamicCache()
        if args.seq_len > 0:
            print(f"Pre-filling DynamicCache to seq_len={args.seq_len}...")
            for layer_idx in range(cfg.num_hidden_layers):
                k = torch.randn(args.batch_size, cfg.num_key_value_heads, args.seq_len, head_dim, device=device, dtype=dtype) * 0.02
                v = torch.randn(args.batch_size, cfg.num_key_value_heads, args.seq_len, head_dim, device=device, dtype=dtype) * 0.02
                hf_cache.update(k, v, layer_idx)

    if args.compiled:
        print("Compiling HF model (this may take a few minutes)...")
        model = torch.compile(model)

    return model, hf_cache, cfg


def main():
    args = parse_args()
    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    print(f"\n{'='*60}")
    print(f"  Hugging Face Decode Throughput Benchmark")
    print(f"  seq_len={args.seq_len}  decode_steps={args.decode_steps}  batch={args.batch_size}")
    cache_type = "static" if args.static_cache else "dynamic"
    print(f"  dtype={args.dtype}  compiled={args.compiled}  cache={cache_type}")
    print(f"{'='*60}\n")

    model, kv_cache, cfg = build_model(args, device, dtype)

    token_ids = torch.zeros(args.batch_size, 1, dtype=torch.long, device=device)
    torch.cuda.synchronize()

    # ── Warmup ────────────────────────────────────────────────────────────
    print(f"Warming up ({args.warmup} steps)...")
    for i in range(args.warmup):
        pos = args.seq_len + i
        position_ids = torch.tensor([[pos]], device=device)
        with torch.inference_mode():
            model(input_ids=token_ids, past_key_values=kv_cache, use_cache=True, position_ids=position_ids)
    torch.cuda.synchronize()

    # ── Timed run ─────────────────────────────────────────────────────────
    start_pos = args.seq_len + args.warmup
    n = args.decode_steps

    start_event = torch.cuda.Event(enable_timing=True)
    end_event   = torch.cuda.Event(enable_timing=True)

    print("Running timed steps...")
    start_event.record()
    for i in range(n):
        position_ids = torch.tensor([[start_pos + i]], device=device)
        with torch.inference_mode():
            model(input_ids=token_ids, past_key_values=kv_cache, use_cache=True, position_ids=position_ids)
    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event)
    elapsed_s  = elapsed_ms / 1000.0
    total_tokens = n * args.batch_size
    tok_per_sec = total_tokens / elapsed_s

    # ── Per-step timing ───────────────────────────────────────────────────
    per_step_ms = elapsed_ms / n

    # ── Report ────────────────────────────────────────────────────────────
    model_name = "HF Llama-3 8B" if not args.small else "HF Tiny (2-layer)"
    print(f"\n{'─'*60}")
    print(f"  Model:            {model_name}")
    print(f"  Context length:   {args.seq_len} → {start_pos + n}")
    print(f"  Decode steps:     {n}")
    print(f"  Batch size:       {args.batch_size}")
    print(f"  Dtype:            {args.dtype}")
    print(f"  Compiled:         {args.compiled}")
    print(f"{'─'*60}")
    print(f"  Total time:       {elapsed_ms:.2f} ms")
    print(f"  Per step:         {per_step_ms:.3f} ms/tok")
    print(f"  Throughput:       {tok_per_sec:.1f} tok/s")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()
