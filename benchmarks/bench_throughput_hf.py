"""
bench_throughput_hf.py — Decode tok/s benchmark for the Hugging Face Llama implementation.

Usage:
    python benchmarks/bench_throughput_hf.py [--seq-len 512] [--decode-steps 128] [--small]
        [--batch-size 1] [--dtype float16] [--mode eager-dynamic]

Same workload as bench_throughput.py: Llama 3.1 8B shape, random fp16 weights, KV cache
pre-filled with random data to seq_len, then warmup + timed single-token decode steps,
timed with CUDA events. Each run is appended to benchmarks/results/throughput_runs.csv.

Modes:
  eager-dynamic : SDPA, DynamicCache (HF's default; torch.cat grows the cache every step)
  eager-static  : SDPA, StaticCache
  compile-static: torch.compile(model) + StaticCache
  compile-cg    : torch.compile(model, mode="reduce-overhead") + StaticCache (CUDA graphs);
                  the strongest HF baseline

The StaticCache is filled by writing random K/V straight into its tensors. A seq_len-token
prefill forward would build a seq_len x seq_len causal mask (8 GiB at 64K).
StaticCache is sized to exactly the positions the run touches, because SDPA attends over
all max_cache_len rows (masked), not just the filled ones.
"""

import argparse

import torch
import transformers
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.cache_utils import DynamicCache, StaticCache

import run_log

MODES = ["eager-dynamic", "eager-static", "compile-static", "compile-cg"]


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
    p.add_argument("--mode",         type=str, default="eager-dynamic", choices=MODES)
    p.add_argument("--small",        action="store_true",
                   help="Use tiny 2-layer config for fast iteration")
    p.add_argument("--no-log",       action="store_true",
                   help="Don't append this run to benchmarks/results/throughput_runs.csv")
    p.add_argument("--note",         type=str, default="",
                   help="Free-text note stored with the logged run")
    return p.parse_args()


def build_config(args):
    if args.small:
        return LlamaConfig(
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
    # Llama 3.1 8B
    return LlamaConfig(
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        intermediate_size=14336,
        vocab_size=128256,
        max_position_embeddings=131072,
        rms_norm_eps=1e-5,
        rope_theta=500000.0,
        rope_scaling={
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3",
        },
        attn_implementation="sdpa",
    )


def build_cache(args, cfg, device, dtype):
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    kv_shape = (args.batch_size, cfg.num_key_value_heads, args.seq_len, head_dim)

    def rand_kv():
        return torch.randn(kv_shape, device=device, dtype=dtype) * 0.02

    if args.mode == "eager-dynamic":
        cache = DynamicCache(config=cfg)
        for layer_idx in range(cfg.num_hidden_layers):
            cache.update(rand_kv(), rand_kv(), layer_idx)
        return cache

    cache = StaticCache(config=cfg, max_cache_len=args.seq_len + args.warmup + args.decode_steps)
    one_token = torch.zeros(args.batch_size, cfg.num_key_value_heads, 1, head_dim, device=device, dtype=dtype)
    for layer in cache.layers:
        layer.lazy_initialization(one_token, one_token)
        layer.keys[:, :, :args.seq_len] = rand_kv()
        layer.values[:, :, :args.seq_len] = rand_kv()
        # transformers 5.x StaticLayer writes at its own counter, not the cache_position passed in
        layer.cumulative_length.fill_(args.seq_len)
    return cache


def build_model(args, device, dtype):
    cfg = build_config(args)
    print(f"Initializing HF LlamaForCausalLM on {device} ({args.dtype})...")
    # Instantiate directly on GPU in target precision
    old_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    with torch.device(device):
        model = LlamaForCausalLM(cfg)
    torch.set_default_dtype(old_default_dtype)
    model.eval()

    print(f"Pre-filling {args.mode} cache to seq_len={args.seq_len}...")
    cache = build_cache(args, cfg, device, dtype)

    fwd = model
    if args.mode == "compile-static":
        fwd = torch.compile(model)
    elif args.mode == "compile-cg":
        fwd = torch.compile(model, mode="reduce-overhead")
    return fwd, cache, cfg


def main():
    args = parse_args()
    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    print(f"\n{'='*60}")
    print(f"  Hugging Face Decode Throughput Benchmark")
    print(f"  seq_len={args.seq_len}  decode_steps={args.decode_steps}  batch={args.batch_size}")
    print(f"  dtype={args.dtype}  mode={args.mode}")
    print(f"{'='*60}\n")

    fwd, cache, cfg = build_model(args, device, dtype)
    static = args.mode != "eager-dynamic"

    token_ids = torch.zeros(args.batch_size, 1, dtype=torch.long, device=device)
    n_pos = args.warmup + args.decode_steps
    # built up front so the timed loop does no host->device copies
    position_ids = [torch.full((args.batch_size, 1), args.seq_len + i, device=device) for i in range(n_pos)]
    cache_position = [torch.tensor([args.seq_len + i], device=device) for i in range(n_pos)]

    def step(i):
        kwargs = dict(input_ids=token_ids, past_key_values=cache, use_cache=True, position_ids=position_ids[i])
        if static:
            kwargs["cache_position"] = cache_position[i]
        fwd(**kwargs)

    torch.cuda.synchronize()

    # ── Warmup ────────────────────────────────────────────────────────────
    print(f"Warming up ({args.warmup} steps{', compiling' if args.mode.startswith('compile') else ''})...")
    with torch.inference_mode():
        for i in range(args.warmup):
            step(i)
    torch.cuda.synchronize()

    # ── Timed run ─────────────────────────────────────────────────────────
    start_pos = args.seq_len + args.warmup
    n = args.decode_steps

    start_event = torch.cuda.Event(enable_timing=True)
    end_event   = torch.cuda.Event(enable_timing=True)

    print("Running timed steps...")
    with torch.inference_mode():
        start_event.record()
        for i in range(n):
            step(args.warmup + i)
        end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event)
    elapsed_s  = elapsed_ms / 1000.0
    total_tokens = n * args.batch_size
    tok_per_sec = total_tokens / elapsed_s

    # ── Per-step timing ───────────────────────────────────────────────────
    per_step_ms = elapsed_ms / n

    # ── Report ────────────────────────────────────────────────────────────
    model_name = "HF Llama-3.1 8B" if not args.small else "HF Tiny (2-layer)"
    print(f"\n{'─'*60}")
    print(f"  Model:            {model_name}")
    print(f"  Context length:   {args.seq_len} → {start_pos + n}")
    print(f"  Decode steps:     {n}")
    print(f"  Batch size:       {args.batch_size}")
    print(f"  Dtype:            {args.dtype}")
    print(f"  Mode:             {args.mode}")
    print(f"{'─'*60}")
    print(f"  Total time:       {elapsed_ms:.2f} ms")
    print(f"  Per step:         {per_step_ms:.3f} ms/tok")
    print(f"  Throughput:       {tok_per_sec:.1f} tok/s")
    print(f"{'─'*60}")
    if not args.no_log:
        run_log.log_run({
            "impl": "hf",
            "mode": args.mode,
            "model": "tiny" if args.small else "llama3.1-8b",
            "seq_len": args.seq_len,
            "kv_len": args.seq_len + n_pos if static else "",
            "decode_steps": args.decode_steps,
            "warmup": args.warmup,
            "batch_size": args.batch_size,
            "profiler_scopes": False,
            "cuda_graphs": args.mode == "compile-cg",
            "tok_s": round(tok_per_sec, 2),
            "ms_per_tok": round(per_step_ms, 3),
            "note": f"transformers {transformers.__version__}; {args.note}".rstrip("; "),
        })
    print()


if __name__ == "__main__":
    main()
