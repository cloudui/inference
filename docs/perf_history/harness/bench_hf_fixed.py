"""
HF reference for the fixed decode benchmark (same workload as bench_fixed.py):
Llama-3 8B shape, random fp16 weights, batch=1, 512 cached tokens, one run = 128 decode
steps over positions 512..639, 3 warmup runs + 5 timed runs, median tok/s.

Derived from benchmarks/bench_throughput_hf.py @ 40dc0b6, adapted to transformers 5.x
(StaticCache no longer takes batch_size). Modes:
  eager-dynamic : SDPA, DynamicCache pre-filled with random KV, cropped back to 512 each run
  eager-static  : SDPA, StaticCache (prefilled with one 512-token forward), explicit cache_position
  compile-static: torch.compile(model) (default mode) + StaticCache
  compile-cg    : torch.compile(model, mode="reduce-overhead") + StaticCache  (CUDA graphs)
"""
import argparse, json, statistics, sys, time, traceback
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.cache_utils import DynamicCache, StaticCache

p = argparse.ArgumentParser()
p.add_argument("--mode", required=True, choices=["eager-dynamic", "eager-static", "compile-static", "compile-cg"])
p.add_argument("--seq-len", type=int, default=512)
p.add_argument("--decode-steps", type=int, default=128)
p.add_argument("--warmup-runs", type=int, default=3)
p.add_argument("--timed-runs", type=int, default=5)
args = p.parse_args()
res = {"mode": args.mode, "status": "error"}
try:
    import transformers
    res["transformers"] = transformers.__version__
    dev, dt = torch.device("cuda"), torch.float16
    torch.manual_seed(0)
    cfg = LlamaConfig(hidden_size=4096, num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=8,
                      intermediate_size=14336, vocab_size=128256, max_position_embeddings=8192,
                      rms_norm_eps=1e-5, rope_theta=500000.0, attn_implementation="sdpa")
    torch.set_default_dtype(dt)
    with torch.device(dev):
        model = LlamaForCausalLM(cfg)
    torch.set_default_dtype(torch.float32)
    model.eval()
    hd = cfg.hidden_size // cfg.num_attention_heads
    S, N = args.seq_len, args.decode_steps
    tok = torch.zeros(1, 1, dtype=torch.long, device=dev)

    if args.mode == "eager-dynamic":
        cache = DynamicCache(config=cfg)
        for l in range(cfg.num_hidden_layers):
            cache.update(torch.randn(1, cfg.num_key_value_heads, S, hd, device=dev, dtype=dt) * 0.02,
                         torch.randn(1, cfg.num_key_value_heads, S, hd, device=dev, dtype=dt) * 0.02, l)
    else:
        cache = StaticCache(config=cfg, max_cache_len=S + N + 16)
        with torch.inference_mode():
            model(input_ids=torch.zeros(1, S, dtype=torch.long, device=dev), past_key_values=cache, use_cache=True,
                  cache_position=torch.arange(S, device=dev))
    fwd = model
    if args.mode == "compile-static":
        fwd = torch.compile(model)
    elif args.mode == "compile-cg":
        fwd = torch.compile(model, mode="reduce-overhead")

    pos_ids = [torch.tensor([[S + i]], device=dev) for i in range(N)]
    cache_pos = [torch.tensor([S + i], device=dev) for i in range(N)]

    def one_run():
        if args.mode == "eager-dynamic":
            cache.crop(S)
        else:
            # transformers 5.x StaticLayer writes at its own cumulative_length counter (ignores
            # cache_position), so rewind it in place so every run replays positions S..S+N-1
            with torch.inference_mode():
                for layer in cache.layers:
                    layer.cumulative_length.fill_(S)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            for i in range(N):
                if args.mode == "eager-dynamic":
                    fwd(input_ids=tok, past_key_values=cache, use_cache=True, position_ids=pos_ids[i])
                else:
                    fwd(input_ids=tok, past_key_values=cache, use_cache=True, position_ids=pos_ids[i],
                        cache_position=cache_pos[i])
        t_enq = time.perf_counter()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        return dict(wall_ms=(t1 - t0) * 1e3, enqueue_ms=(t_enq - t0) * 1e3)

    t = time.perf_counter()
    for _ in range(args.warmup_runs):
        one_run()
    res["warmup_s"] = time.perf_counter() - t
    runs = [one_run() for _ in range(args.timed_runs)]
    for r in runs:
        r["tok_s"] = N / (r["wall_ms"] / 1e3)
    res.update(runs=runs, median_tok_s=statistics.median(r["tok_s"] for r in runs),
               min_tok_s=min(r["tok_s"] for r in runs), max_tok_s=max(r["tok_s"] for r in runs),
               median_ms_per_tok=statistics.median(r["wall_ms"] for r in runs) / N,
               median_enqueue_ms_per_tok=statistics.median(r["enqueue_ms"] for r in runs) / N, status="ok")
except BaseException as e:
    res["error"] = f"{type(e).__name__}: {e}"[:600]
    res["traceback"] = traceback.format_exc()[-3000:]
print(json.dumps(res))
