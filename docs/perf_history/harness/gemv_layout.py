"""Isolate ef70981: x(1,1,K) @ W with W stored (K,N) row-major (before) vs W stored (N,K) and passed as W.T (after).
Per-layer Llama-3 8B shapes + lm_head. GPU time via CUDA events over 200 back-to-back calls."""
import torch
torch.manual_seed(0)
dev, dt = "cuda", torch.float16
shapes = {"wqkv": (4096, 6144), "wo": (4096, 4096), "w_gate_up": (4096, 28672), "w_down": (14336, 4096), "lm_head": (4096, 128256)}
per_layer = {"wqkv", "wo", "w_gate_up", "w_down"}
tot = {"before": 0.0, "after": 0.0}
print(f"{'weight':10s} {'K x N':>14s} {'before (K,N) us':>16s} {'after (N,K).T us':>17s} {'GB/s before':>12s} {'GB/s after':>11s}")
for name, (K, N) in shapes.items():
    x = torch.randn(1, 1, K, device=dev, dtype=dt)
    ncp = max(2, int(1e9 // (K * N * 2)) + 1)
    W_kn = [torch.randn(K, N, device=dev, dtype=dt) for _ in range(ncp)]
    W_nk = [torch.randn(N, K, device=dev, dtype=dt) for _ in range(ncp)]
    out = torch.empty(1, 1, N, device=dev, dtype=dt)
    res = {}
    for tag, fn in (("before", lambda i: torch.matmul(x, W_kn[i % ncp], out=out)), ("after", lambda i: torch.matmul(x, W_nk[i % ncp].T, out=out))):
        for i in range(20): fn(i)
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for i in range(200): fn(i)
        e1.record(); torch.cuda.synchronize()
        res[tag] = e0.elapsed_time(e1) / 200 * 1e3  # us
        tot[tag] += res[tag] * (32 if name in per_layer else 1)
    gb = K * N * 2 / 1e9
    print(f"{name:10s} {f'{K}x{N}':>14s} {res['before']:16.1f} {res['after']:17.1f} {gb/(res['before']/1e6):12.0f} {gb/(res['after']/1e6):11.0f}")
    del W_kn, W_nk
print(f"\nper-token GEMV time (32 layers + lm_head): before {tot['before']/1e3:.2f} ms, after {tot['after']/1e3:.2f} ms, saved {(tot['before']-tot['after'])/1e3:.2f} ms")
