"""Our engine in a user-visible generation loop: forward -> greedy argmax -> token to host, per step.
Wall-clock per step, same accounting as bench_throughput_vllm.py. Produced the "ours, generation
loop" column of vllm_comparison.csv (LONG_CONTEXT.md).

    python docs/perf_history/harness/generation_loop_walltime.py 512 2048 8192 32768 113664
"""
import statistics, sys, time, types, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarks"))
import bench_throughput as b

for S in [int(x) for x in sys.argv[1:]]:
    a = types.SimpleNamespace(small=False, seq_len=S, warmup=30, decode_steps=128, batch_size=1, kv_len=None)
    model, kv, cfg = b.build_model(a)
    model.enable_cuda_graphs()
    for K, V in kv:
        K[:, :, :S] = torch.randn_like(K[:, :, :S]) * 0.02
        V[:, :, :S] = torch.randn_like(V[:, :, :S]) * 0.02
    runs = []
    for r in range(4):  # run 0 discarded
        tok = torch.zeros(1, 1, dtype=torch.long, device="cuda")
        steps = []
        for i in range(a.warmup + a.decode_steps):
            t0 = time.perf_counter()
            logits = model.forward(tok, start_pos=S + i, kv_caches=kv)
            nxt = int(logits[0, -1].argmax().item())  # host sync: the token a server would stream
            tok.fill_(nxt)
            steps.append(time.perf_counter() - t0)
        if r:
            runs.append(statistics.mean(steps[a.warmup:]) * 1e3)
    ms = statistics.median(runs)
    print(f"seq_len={S}: {ms:.3f} ms/tok ({1e3 / ms:.1f} tok/s) runs={[round(x, 3) for x in runs]}", flush=True)
    del model, kv
    torch.cuda.empty_cache()
