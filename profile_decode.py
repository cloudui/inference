"""
profile_decode.py — Decode-step profiler for the custom Llama inference stack.

Usage:
    python profile_decode.py [--seq-len N] [--steps K] [--export-chrome]

What this measures:
    • Wall-clock breakdown of every kernel called during a single decode step
    • CUDA kernel time vs. CPU dispatch overhead (the "dispatch gap")
    • Memory bandwidth consumed by each custom Triton kernel
    • Summary table sorted by self-CUDA time so the worst offender is always #1

What it does NOT do (yet — see roadmap at bottom):
    • torch.compile / inductor traces
    • CUDA Graphs (we record the profiler BEFORE adding graphs)
    • Multi-step autoregressive loops (add --steps > 1 to stress-test overlap)
"""

import argparse
import math
import time
import torch
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler

# ── local imports ──────────────────────────────────────────────────────────────
from model import Llama, LlamaConfig

# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seq-len",      type=int, default=512,
                   help="Number of KV tokens already in the cache (simulates mid-sequence decode)")
    p.add_argument("--steps",        type=int, default=3,
                   help="Decode steps to profile (first is warmup, rest are measured)")
    p.add_argument("--export-chrome", action="store_true",
                   help="Write a Chrome trace to ./profile_trace/ for chrome://tracing")
    p.add_argument("--batch-size",   type=int, default=1)
    p.add_argument("--small",        action="store_true",
                   help="Use a tiny 2-layer config for fast iteration without real weights")
    return p.parse_args()


# ── Model setup ───────────────────────────────────────────────────────────────

def build_model(args) -> tuple[Llama, list]:
    """
    Build a model with random weights.
    Use --small for fast iteration. Otherwise matches Llama-3 8B dims.
    """
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
        cfg = LlamaConfig()  # Llama-3 8B defaults

    device = torch.device("cuda")
    model = Llama(cfg)

    # Fill weights with random fp16 data (no checkpoint needed for profiling)
    def rand_fp16(*shape):
        return torch.randn(*shape, dtype=torch.float16, device=device) * 0.02

    model.embed_tokens    = rand_fp16(cfg.vocab_size, cfg.hidden_size)
    model.lm_head         = rand_fp16(cfg.hidden_size, cfg.vocab_size)
    model.norm.weight     = rand_fp16(cfg.hidden_size)
    model.cos             = model.cos.to(device)
    model.sin             = model.sin.to(device)

    for layer in model.layers:
        layer.self_attn.wq = rand_fp16(cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim)
        layer.self_attn.wk = rand_fp16(cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim)
        layer.self_attn.wv = rand_fp16(cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim)
        layer.self_attn.wo = rand_fp16(cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size)
        layer.input_layernorm.weight      = rand_fp16(cfg.hidden_size)
        layer.post_attention_layernorm.weight = rand_fp16(cfg.hidden_size)
        layer.mlp.w_gate = rand_fp16(cfg.hidden_size, cfg.intermediate_size)
        layer.mlp.w_up   = rand_fp16(cfg.hidden_size, cfg.intermediate_size)
        layer.mlp.w_down = rand_fp16(cfg.intermediate_size, cfg.hidden_size)

    kv_caches = model.allocate_kv_cache(
        batch_size=args.batch_size,
        max_seq_len=cfg.max_position_embeddings,
        device=device,
    )
    return model, kv_caches, cfg


# ── Annotated forward ─────────────────────────────────────────────────────────

@torch.inference_mode()
def profiled_forward(model: Llama, token_ids: torch.Tensor, start_pos: int,
                     kv_caches: list, cfg: LlamaConfig):
    """
    Identical to model.forward() but wrapped with record_function annotations
    so the profiler trace is human-readable per-component.
    
    These labels show up in chrome://tracing and in the table summary.
    """
    with record_function("embed_lookup"):
        hidden_states = model.embed_tokens[token_ids]

    cos = model.cos[start_pos : start_pos + 1].unsqueeze(0).unsqueeze(0)
    sin = model.sin[start_pos : start_pos + 1].unsqueeze(0).unsqueeze(0)

    for i, layer in enumerate(model.layers):
        with record_function(f"layer_{i}"):
            with record_function(f"layer_{i}/input_norm"):
                normed = layer.input_layernorm(hidden_states)

            with record_function(f"layer_{i}/attn"):
                with record_function(f"layer_{i}/attn/qkv_proj"):
                    # These matmuls are the dominant cost in large batch / long-seq prefill
                    # For decode (bs=1, seqlen=1) they are tiny — watch the dispatch cost here
                    q = (normed @ layer.self_attn.wq).view(
                        token_ids.shape[0], 1, -1, cfg.head_dim).transpose(1, 2)
                    k = (normed @ layer.self_attn.wk).view(
                        token_ids.shape[0], 1, -1, cfg.head_dim).transpose(1, 2)
                    v = (normed @ layer.self_attn.wv).view(
                        token_ids.shape[0], 1, -1, cfg.head_dim).transpose(1, 2)

                with record_function(f"layer_{i}/attn/rope"):
                    from model import apply_rope
                    q, k = apply_rope(q, k, cos, sin)

                with record_function(f"layer_{i}/attn/kv_cache_write"):
                    K, V = kv_caches[i]
                    K[:, :, start_pos:start_pos+1] = k
                    V[:, :, start_pos:start_pos+1] = v

                with record_function(f"layer_{i}/attn/flash_decode"):
                    import kernels.flash_decode as fd_module
                    fd_out = fd_module.flash_decode(
                        q, K[:, :, :start_pos+1], V[:, :, :start_pos+1]
                    )

                with record_function(f"layer_{i}/attn/out_proj"):
                    attn_out = fd_out.transpose(1, 2).reshape(hidden_states.shape) @ layer.self_attn.wo

            with record_function(f"layer_{i}/attn_residual"):
                hidden_states = hidden_states + attn_out

            with record_function(f"layer_{i}/post_norm"):
                normed2 = layer.post_attention_layernorm(hidden_states)

            with record_function(f"layer_{i}/mlp"):
                with record_function(f"layer_{i}/mlp/gate_up_proj"):
                    up   = normed2 @ layer.mlp.w_up
                    gate = normed2 @ layer.mlp.w_gate

                with record_function(f"layer_{i}/mlp/swiglu"):
                    import kernels.swiglu as swiglu_module
                    act = swiglu_module.swiglu(up, gate)

                with record_function(f"layer_{i}/mlp/down_proj"):
                    mlp_out = act @ layer.mlp.w_down

            with record_function(f"layer_{i}/mlp_residual"):
                hidden_states = hidden_states + mlp_out

    with record_function("final_norm"):
        x = model.norm(hidden_states)

    with record_function("lm_head"):
        logits = x @ model.lm_head

    return logits


# ── Timing helpers ────────────────────────────────────────────────────────────

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


# ── Bandwidth estimator ───────────────────────────────────────────────────────

def estimate_decode_bandwidth(cfg: LlamaConfig, batch: int, seq_len: int) -> dict:
    """
    Rough arithmetic-intensity / bandwidth breakdown for one decode step.

    At bs=1, seq=1, decode is bandwidth-bound (matmuls are thin).
    This tells you roughly how much HBM traffic we produce per step
    and what the theoretical roofline looks like on L40S (864 GB/s peak).

    Numbers are bytes, not flops.
    """
    B = batch
    H  = cfg.hidden_size
    Hkv = cfg.num_key_value_heads * cfg.head_dim
    Hq  = cfg.num_attention_heads * cfg.head_dim
    I  = cfg.intermediate_size
    S  = seq_len
    dtype_bytes = 2  # fp16

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

    l40s_bw_gbs = 864.0  # peak HBM bandwidth (GB/s) — L40S spec
    theoretical_ms = (total_bytes / 1e9) / l40s_bw_gbs * 1e3

    return {
        "weights_GB":          total_weights / 1e9,
        "kv_cache_read_GB":    total_kv_read / 1e9,
        "total_HBM_traffic_GB": total_bytes  / 1e9,
        "roofline_ms":         theoretical_ms,
        "note": "Roofline assumes 100% HBM bandwidth, no reuse. Real perf ~60-70% of this.",
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device("cuda")

    print(f"\n{'='*70}")
    print(f"  Llama Decode Profiler  |  seq_len={args.seq_len}  batch={args.batch_size}")
    print(f"{'='*70}\n")

    # ── 1. Build model ────────────────────────────────────────────────────────
    print("Building model with random weights...")
    model, kv_caches, cfg = build_model(args)

    token_ids = torch.zeros(args.batch_size, 1, dtype=torch.long, device=device)

    # Prefill the KV cache up to seq_len with garbage (simulates prior context)
    print(f"Pre-filling KV cache to seq_len={args.seq_len} ...")
    if args.seq_len > 0:
        fake_prefill_ids = torch.zeros(
            args.batch_size, args.seq_len, dtype=torch.long, device=device
        )
        with torch.inference_mode():
            _ = model.forward(fake_prefill_ids, start_pos=0, kv_caches=kv_caches)

    torch.cuda.synchronize()
    print("Done.\n")

    # ── 2. Raw CUDA timing (no profiler overhead) ─────────────────────────────
    print("─── Raw CUDA timing (warmup=3, iters=10) ───────────────────────────")
    timing = cuda_timed(
        profiled_forward,
        model, token_ids, args.seq_len, kv_caches, cfg,
        warmup=3, iters=10,
    )
    print(f"  Min:    {timing['min_ms']:.3f} ms")
    print(f"  Median: {timing['median_ms']:.3f} ms")
    print(f"  Max:    {timing['max_ms']:.3f} ms")
    print()

    # ── 3. Roofline estimate ──────────────────────────────────────────────────
    bw = estimate_decode_bandwidth(cfg, args.batch_size, args.seq_len)
    print("─── HBM bandwidth estimate (L40S roofline) ─────────────────────────")
    print(f"  Weight traffic:    {bw['weights_GB']:.2f} GB")
    print(f"  KV cache traffic:  {bw['kv_cache_read_GB']:.2f} GB  (seq_len={args.seq_len})")
    print(f"  Total HBM:         {bw['total_HBM_traffic_GB']:.2f} GB")
    print(f"  Roofline bound:    {bw['roofline_ms']:.2f} ms  (at 864 GB/s peak)")
    if timing['median_ms'] > 0:
        achieved_bw = bw['total_HBM_traffic_GB'] / (timing['median_ms'] / 1e3)
        print(f"  Achieved ~BW:      {achieved_bw:.0f} GB/s  ({achieved_bw/864*100:.0f}% of peak)")
    print(f"  Note: {bw['note']}")
    print()

    # ── 4. torch.profiler trace ───────────────────────────────────────────────
    print("─── Running torch.profiler (1 warmup + 2 active steps) ─────────────")

    trace_handler = (
        tensorboard_trace_handler("./profile_trace")
        if args.export_chrome
        else None
    )

    schedule = torch.profiler.schedule(wait=0, warmup=1, active=2, repeat=1)

    prof_kwargs = dict(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule,
        record_shapes=True,
        with_stack=False,   # True adds Python stack frames; useful but expensive
        profile_memory=True,
        with_flops=True,
    )
    if trace_handler:
        prof_kwargs["on_trace_ready"] = trace_handler

    with profile(**prof_kwargs) as prof:
        for step in range(3):  # 0=warmup, 1+2=active
            profiled_forward(model, token_ids, args.seq_len, kv_caches, cfg)
            prof.step()

    # ── 5. Print summary table ────────────────────────────────────────────────
    print("\n─── Top-20 ops by self-CUDA time ────────────────────────────────────")
    print(prof.key_averages(group_by_input_shape=False).table(
        sort_by="self_cuda_time_total",
        row_limit=20,
    ))

    # ── 6. Dispatch gap analysis ──────────────────────────────────────────────
    print("\n─── CPU vs CUDA dispatch gap ────────────────────────────────────────")
    avgs = prof.key_averages()
    total_cpu_ms  = sum(e.cpu_time_total  for e in avgs) / 1e3
    total_cuda_ms = sum(e.cuda_time_total for e in avgs) / 1e3
    gap_ms        = total_cpu_ms - total_cuda_ms
    print(f"  Total CPU  time: {total_cpu_ms:.2f} ms")
    print(f"  Total CUDA time: {total_cuda_ms:.2f} ms")
    print(f"  Dispatch gap:    {gap_ms:.2f} ms")
    if gap_ms > 0.5 * total_cuda_ms:
        print("  ⚠️  CPU dispatch is >50% of CUDA time — CUDA Graphs will help a lot here.")
    else:
        print("  ✓  Dispatch overhead is modest — focus on kernel efficiency first.")

    # ── 7. Memory allocation report ───────────────────────────────────────────
    print("\n─── Top temporaries by self-memory ─────────────────────────────────")
    print(prof.key_averages().table(
        sort_by="self_cpu_memory_usage",
        row_limit=10,
    ))

    # ── 8. Optimization roadmap printed inline ────────────────────────────────
    print_roadmap(timing['median_ms'], bw['roofline_ms'], gap_ms, total_cuda_ms)

    if args.export_chrome:
        print(f"\nChrome trace written to ./profile_trace/")
        print("Open chrome://tracing and load the JSON file there.")


def print_roadmap(median_ms: float, roofline_ms: float, gap_ms: float, cuda_ms: float):
    efficiency = roofline_ms / median_ms if median_ms > 0 else 0
    gap_ratio  = gap_ms / cuda_ms if cuda_ms > 0 else 0

    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║               OPTIMIZATION ROADMAP  (your naïve baseline)               ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Phase 0 — Already done (your current stack)                            ║
║    ✓ Split-KV flash decode (avoids full attention matrix)               ║
║    ✓ Custom RMSNorm + SwiGLU Triton kernels                             ║
║    ✓ Fused RMSNorm+SwiGLU kernel exists (check if it's wired in MLP)   ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Phase 1 — Low-hanging fruit (memory layout + easy fuses)               ║
║                                                                          ║
║  1a. REMOVE the torch.cuda.synchronize() between gen+reduce kernels     ║
║      in flash_decode.py:280 — this is a hard CPU stall on the hot path. ║
║      Just remove it; the kernel launch order guarantees ordering.       ║
║                                                                          ║
║  1b. Wire up fused_rmsnorm_swiglu in MLP.__call__                       ║
║      Currently MLP calls rmsnorm then swiglu separately (2 passes).    ║
║      The fused kernel halves HBM reads for the FFN pre-activation.     ║
║                                                                          ║
║  1c. Fuse QKV projection into a single matmul                           ║
║      Concat [wq, wk, wv] → single (H, Hq+2*Hkv) weight → 1 GEMM.     ║
║      Saves 2 kernel launches + reduces dispatch overhead.               ║
║                                                                          ║
║  1d. KV cache layout: consider (batch, seq, heads, dim) → contiguous   ║
║      sliced reads in flash_decode. Current (b, h, s, d) means the      ║
║      seq slice K[:,:,:pos] is contiguous but head stride adds overhead. ║
║                                                                          ║
║  1e. Pre-slice RoPE outside the layer loop (cos/sin are already done,  ║
║      but the unsqueeze() allocates a new tensor every step).           ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Phase 2 — Kernel-level wins (Triton tuning)                            ║
║                                                                          ║
║  2a. Tune BLOCK_SEQ_KV in flash_decode (currently hardcoded 64).       ║
║      At seq_len=512 on L40S, larger blocks (128) may be faster.        ║
║      Use triton.autotune on the generation kernel.                      ║
║                                                                          ║
║  2b. RMSNorm accumulate in fp32, store fp16 — check that the kernel    ║
║      doesn't downcast mid-reduction (yours looks correct already).     ║
║                                                                          ║
║  2c. Flash decode reduce kernel: the inner tl.range loop is serial.    ║
║      For short n_blocks (<= 8) consider unrolling or parallel reduce.  ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Phase 3 — Dispatch overhead (CUDA Graphs)                              ║
║                                                                          ║
║  3a. CUDA Graphs: capture the full decode step into a graph and replay ║
║      it. Eliminates ALL CPU-side kernel launch overhead. Most           ║
║      important for bs=1 decode where GPU is starved of work.            ║
║      Constraint: static shapes and no Python control flow per step.    ║
║                                                                          ║
║  3b. torch.compile on individual kernels (not full model) as a         ║
║      targeted tool to eliminate Python overhead on hot paths (e.g.     ║
║      the QKV proj + RoPE + residual chain can be compiled together).   ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Phase 4 — The big wins (Hopper / vLLM-class)                          ║
║                                                                          ║
║  4a. Paged / chunked KV cache (for variable-length batching)            ║
║  4b. Continuous batching across requests                                ║
║  4c. WGMMA / TMA-based attention on Hopper (H100 tensor core paths)    ║
║  4d. Speculative decoding for latency                                   ║
╚══════════════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
