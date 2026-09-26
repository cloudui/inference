"""
bench_throughput.py — Decode tok/s benchmark for the custom Llama inference stack.

Usage:
    python bench_throughput.py [--seq-len 512] [--decode-steps 128] [--small] [--batch-size 1]

Measures wall-clock tok/s for single-token decode steps using CUDA event timing.
KV cache is pre-filled with random data to simulate mid-sequence decoding.
Each run is appended to benchmarks/results/throughput_runs.csv with the commit and
environment (disable with --no-log).
"""

import argparse
import csv
import datetime
import platform
import subprocess
from pathlib import Path

import torch
import triton

from model import Llama, LlamaConfig, set_profiling, profiling_enabled


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seq-len",      type=int, default=512,
                   help="KV tokens already in cache (simulates prior context)")
    p.add_argument("--decode-steps", type=int, default=128,
                   help="Number of decode steps to measure")
    p.add_argument("--warmup",       type=int, default=30,
                   help="Warmup decode steps (not measured)")
    p.add_argument("--batch-size",   type=int, default=1)
    p.add_argument("--small",        action="store_true",
                   help="Use tiny 2-layer config for fast iteration")
    p.add_argument("--profile-scopes", action="store_true",
                   help="Keep torch.profiler record_function scopes on (costs CPU time; "
                        "also enabled by INFERENCE_PROFILE=1)")
    p.add_argument("--cuda-graphs",  action="store_true",
                   help="Replay decode steps from a captured CUDA graph (model.enable_cuda_graphs())")
    p.add_argument("--no-log",       action="store_true",
                   help="Don't append this run to benchmarks/results/throughput_runs.csv")
    p.add_argument("--note",         type=str, default="",
                   help="Free-text note stored with the logged run")
    return p.parse_args()


RUN_LOG = Path(__file__).resolve().parent / "results" / "throughput_runs.csv"


def _git(*cmd):
    try:
        return subprocess.check_output(["git", *cmd], cwd=Path(__file__).resolve().parent, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _cpu_model():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def log_run(args, tok_per_sec, per_step_ms):
    """Append one row per run so throughput history survives pod resets."""
    row = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "commit": _git("rev-parse", "--short", "HEAD"),
        # the run log itself doesn't count, or every run after the first would look dirty
        "dirty": bool(_git("status", "--porcelain", "--untracked-files=no", "--",
                           ":(top)", ":(top,exclude)benchmarks/results")),
        "model": "tiny" if args.small else "llama3-8b",
        "seq_len": args.seq_len,
        "decode_steps": args.decode_steps,
        "warmup": args.warmup,
        "batch_size": args.batch_size,
        "profiler_scopes": profiling_enabled(),
        "cuda_graphs": args.cuda_graphs,
        "tok_s": round(tok_per_sec, 2),
        "ms_per_tok": round(per_step_ms, 3),
        "gpu": torch.cuda.get_device_name(),
        "cpu": _cpu_model(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda": torch.version.cuda,
        "note": args.note,
    }
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    rows, fields = [], list(row)
    if RUN_LOG.exists():
        with open(RUN_LOG, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            old_fields = reader.fieldnames or []
        # keep old column order, append any new columns (old rows get them empty)
        fields = old_fields + [k for k in row if k not in old_fields]
    rows.append(row)
    with open(RUN_LOG, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  Logged to {RUN_LOG.relative_to(RUN_LOG.parents[2])}")


def build_model(args):
    if args.small:
        cfg = LlamaConfig(
            hidden_size=512,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=2,
            intermediate_size=1024,
            vocab_size=1024,
            max_position_embeddings=2048,
            head_dim=64,
        )
    else:
        cfg = LlamaConfig()

    device = torch.device("cuda")
    model = Llama(cfg)

    def rand_fp16(*shape):
        return torch.randn(*shape, dtype=torch.float16, device=device) * 0.02

    model.embed_tokens = rand_fp16(cfg.vocab_size, cfg.hidden_size)
    model.lm_head      = rand_fp16(cfg.vocab_size, cfg.hidden_size)
    model.norm.weight   = rand_fp16(cfg.hidden_size)
    model.cos           = model.cos.to(device)
    model.sin           = model.sin.to(device)

    for layer in model.layers:
        qkv_concat_dim_size = cfg.num_attention_heads * cfg.head_dim + 2 * cfg.num_key_value_heads * cfg.head_dim
        layer.self_attn.wqkv = rand_fp16(qkv_concat_dim_size, cfg.hidden_size)
        layer.self_attn.wo = rand_fp16(cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim)
        layer.input_layernorm.weight          = rand_fp16(cfg.hidden_size)
        layer.post_attention_layernorm.weight  = rand_fp16(cfg.hidden_size)
        layer.mlp.w_gate_up = rand_fp16(2 * cfg.intermediate_size, cfg.hidden_size)
        layer.mlp.w_down = rand_fp16(cfg.hidden_size, cfg.intermediate_size)

    kv_caches = model.allocate_kv_cache(
        batch_size=args.batch_size,
        max_seq_len=cfg.max_position_embeddings,
        device=device,
    )
    return model, kv_caches, cfg


def main():
    args = parse_args()
    device = torch.device("cuda")
    if args.profile_scopes:
        set_profiling(True)

    print(f"\n{'='*60}")
    print(f"  Decode Throughput Benchmark")
    print(f"  seq_len={args.seq_len}  decode_steps={args.decode_steps}  batch={args.batch_size}")
    print(f"  profiler scopes={'on' if profiling_enabled() else 'off'}  cuda_graphs={'on' if args.cuda_graphs else 'off'}")
    print(f"{'='*60}\n")

    model, kv_caches, cfg = build_model(args)
    if args.cuda_graphs:
        model.enable_cuda_graphs()  # first warmup step captures the graph

    # Pre-fill KV cache with random data to simulate prior context
    if args.seq_len > 0:
        for K, V in kv_caches:
            K[:, :, :args.seq_len] = torch.randn_like(K[:, :, :args.seq_len]) * 0.02
            V[:, :, :args.seq_len] = torch.randn_like(V[:, :, :args.seq_len]) * 0.02

    token_ids = torch.zeros(args.batch_size, 1, dtype=torch.long, device=device)
    torch.cuda.synchronize()

    # ── Warmup ────────────────────────────────────────────────────────────
    print(f"Warming up ({args.warmup} steps)...")
    for i in range(args.warmup):
        pos = args.seq_len + i
        model.forward(token_ids, start_pos=pos, kv_caches=kv_caches)
    torch.cuda.synchronize()

    # ── Timed run ─────────────────────────────────────────────────────────
    start_pos = args.seq_len + args.warmup
    n = args.decode_steps

    start_event = torch.cuda.Event(enable_timing=True)
    end_event   = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for i in range(n):
        model.forward(token_ids, start_pos=start_pos + i, kv_caches=kv_caches)
    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event)
    elapsed_s  = elapsed_ms / 1000.0
    total_tokens = n * args.batch_size
    tok_per_sec = total_tokens / elapsed_s

    # ── Per-step timing ───────────────────────────────────────────────────
    per_step_ms = elapsed_ms / n

    # ── Report ────────────────────────────────────────────────────────────
    model_name = "Llama-3 8B" if not args.small else "Tiny (2-layer)"
    print(f"\n{'─'*60}")
    print(f"  Model:            {model_name}")
    print(f"  Context length:   {args.seq_len} → {start_pos + n}")
    print(f"  Decode steps:     {n}")
    print(f"  Batch size:       {args.batch_size}")
    print(f"{'─'*60}")
    print(f"  Total time:       {elapsed_ms:.2f} ms")
    print(f"  Per step:         {per_step_ms:.3f} ms/tok")
    print(f"  Throughput:       {tok_per_sec:.1f} tok/s")
    print(f"{'─'*60}")
    if not args.no_log:
        log_run(args, tok_per_sec, per_step_ms)
    print()


if __name__ == "__main__":
    main()
