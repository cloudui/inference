"""One-step GPU kernel breakdown at 32K: ours (eager, same kernels the graph replays) vs HF
DynamicCache vs HF StaticCache. Produced long_context_breakdown_32k.csv (LONG_CONTEXT.md).

    python docs/perf_history/harness/profile_breakdown_32k.py out.json
"""
import json, sys, types, torch
from pathlib import Path
from torch.profiler import profile, ProfilerActivity
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarks"))
import bench_throughput as ours_b, bench_throughput_hf as b

S = 32768
dev, dt = torch.device("cuda"), torch.float16

def category(name):
    n = name.lower()
    if "gemv" in n or "gemm" in n or "cutlass_80_tensorop" in n or "sm90" in n or "cublas" in n: return "Weight GEMVs"
    if "flash" in n or "fmha" in n or "attention" in n: return "Attention"
    if "cat" in n and "batched" in n: return "KV cache copies"
    if "direct_copy" in n or ("elementwise" in n and "direct" in n): return "KV cache copies"
    return "Other"

def breakdown(step):
    for _ in range(3): step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        step(); torch.cuda.synchronize()
    out, detail = {}, []
    for e in p.key_averages():
        if e.device_type.name != "CUDA" or e.self_device_time_total == 0: continue
        c = category(e.key); ms = e.self_device_time_total / 1e3
        out[c] = out.get(c, 0) + ms
        detail.append((round(ms, 3), e.count, c, e.key[:80]))
    detail.sort(reverse=True)
    return {k: round(v, 2) for k, v in out.items()}, detail[:8]

res = {}
# ours
a = types.SimpleNamespace(small=False, seq_len=S, warmup=4, decode_steps=4, batch_size=1, kv_len=None)
model, kv, cfg = ours_b.build_model(a)
tok = torch.zeros(1, 1, dtype=torch.long, device=dev)
with torch.inference_mode():
    res["Ours"] = breakdown(lambda: model.forward(tok, start_pos=S, kv_caches=kv))
del model, kv; torch.cuda.empty_cache()

torch.set_default_dtype(dt)
a = types.SimpleNamespace(small=False, mode="eager-static", seq_len=S, warmup=4, decode_steps=4, batch_size=1)
cfg = b.build_config(a)
with torch.device(dev): hf = b.LlamaForCausalLM(cfg)
torch.set_default_dtype(torch.float32); hf.eval()
for mode, label in [("eager-dynamic", "HF DynamicCache"), ("eager-static", "HF StaticCache")]:
    a.mode = mode
    cache = b.build_cache(a, cfg, dev, dt)
    n = [0]
    def step():
        # advance one position per step like the benchmark (the StaticCache has room for 8)
        p = S + n[0]; n[0] += 1
        kw = dict(input_ids=tok, past_key_values=cache, use_cache=True, position_ids=torch.tensor([[p]], device=dev))
        if mode != "eager-dynamic": kw["cache_position"] = torch.tensor([p], device=dev)
        hf(**kw)
    with torch.inference_mode():
        res[label] = breakdown(step)
    del cache; torch.cuda.empty_cache()

for k, (cats, det) in res.items():
    print(f"\n== {k}: total {sum(cats.values()):.1f} ms  {cats}")
    for d in det: print("   ", d)
json.dump({k: v[0] for k, v in res.items()}, open(sys.argv[1], "w"), indent=1)
