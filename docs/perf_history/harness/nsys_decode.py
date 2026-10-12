"""Steady-state decode steps for Nsight Systems: python nsys_decode.py BATCH [STEPS] [graphs|eager|eager-hooks]."""
import sys, types, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarks"))
import bench_throughput as ob
from model import set_profiling
MODE = sys.argv[3] if len(sys.argv) > 3 else 'graphs'
set_profiling(MODE == 'eager-hooks')
B, STEPS, S = int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 5, 512
a = types.SimpleNamespace(small=False, seq_len=S, warmup=30, decode_steps=STEPS, batch_size=B, kv_len=S + 64)
model, kv, cfg = ob.build_model(a)
if MODE == 'graphs':
    model.enable_cuda_graphs()
tok = torch.zeros(B, 1, dtype=torch.long, device="cuda")
for i in range(10): model.forward(tok, start_pos=S + i, kv_caches=kv)
torch.cuda.synchronize()
torch.cuda.profiler.start()
for i in range(STEPS):
    torch.cuda.nvtx.range_push(f"decode step {i} (batch {B}, {MODE})")
    logits = model.forward(tok, start_pos=S + 10 + i, kv_caches=kv)
    tok.copy_(logits[:, -1].argmax(-1, keepdim=True))
    tok.tolist()
    torch.cuda.nvtx.range_pop()
torch.cuda.synchronize()
torch.cuda.profiler.stop()
