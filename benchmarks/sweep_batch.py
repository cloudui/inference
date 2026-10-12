"""
sweep_batch.py — Decode throughput vs batch size, ours vs vLLM.

Usage:
    python benchmarks/sweep_batch.py [--batch-sizes 1 2 4 ...] [--seq-lens 512 4096] [--csv out.csv]

Runs one process per (context, batch size, implementation), so an OOM only loses that cell.
Every batch slot holds a seq_len-token context and decodes in lockstep. Implementations:
  custom:cuda-graphs       bench_throughput.py --cuda-graphs (CUDA events, no token readback)
  custom:cuda-graphs+loop  ... --generation-loop (host clock, argmax + readback, like vLLM)
  vllm:default             bench_throughput_vllm.py (host clock per engine step)
Every run is also logged to benchmarks/results/throughput_runs.csv (note "batch-sweep <tag>").

Byte model per step: 15.01 GB of weights + 128 KiB of KV per cached token per sequence (see
sweep_context.py). The ceiling column uses the ~820 GB/s the best isolated GEMVs reach here.
"""

import argparse
import csv
import datetime
import sys
from pathlib import Path

from sweep_context import HERE, KV_BYTES_PER_TOKEN, PRACTICAL_GBS, WEIGHT_BYTES, run_one

DEFAULT_RUNS = ["custom:cuda-graphs", "custom:cuda-graphs+loop", "vllm:default"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128])
    p.add_argument("--seq-lens", type=int, nargs="+", default=[512])
    p.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    p.add_argument("--vllm-python", default="/opt/venvs/vllm/bin/python")
    p.add_argument("--decode-steps", type=int, default=128)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--tag", default=datetime.datetime.now().strftime("%Y%m%d-%H%M"))
    p.add_argument("--csv", type=Path, default=None, help="Write one row per cell here")
    return p.parse_args()


def bench_cmd(run: str, seq_len: int, batch: int, args) -> list[str]:
    impl, mode = run.split(":")
    common = ["--seq-len", str(seq_len), "--batch-size", str(batch), "--decode-steps", str(args.decode_steps),
              "--warmup", str(args.warmup), "--note", f"batch-sweep {args.tag}"]
    if impl == "vllm":
        return [args.vllm_python, str(HERE / "bench_throughput_vllm.py"), *common]
    flags = {"cuda-graphs": ["--cuda-graphs"], "cuda-graphs+loop": ["--cuda-graphs", "--generation-loop"],
             "eager": []}[mode]
    return [sys.executable, str(HERE / "bench_throughput.py"), *common, *flags]


def ceiling_ms(seq_len: int, batch: int, args) -> float:
    ctx = seq_len + args.warmup + args.decode_steps / 2  # average context over the timed steps
    return (WEIGHT_BYTES + batch * ctx * KV_BYTES_PER_TOKEN) / (PRACTICAL_GBS * 1e9) * 1e3


def main():
    args = parse_args()
    rows = []
    for seq_len in args.seq_lens:
        for batch in args.batch_sizes:
            for run in args.runs:
                print(f"[{args.tag}] seq_len={seq_len:>6} batch={batch:>4}  {run:<24}", end=" ", flush=True)
                r = run_one(run, seq_len, args, cmd=bench_cmd(run, seq_len, batch, args))
                ok = isinstance(r, float)
                print(f"{r:.3f} ms/step ({batch * 1e3 / r:.0f} tok/s)" if ok else r, flush=True)
                rows.append({"seq_len": seq_len, "batch": batch, "run": run,
                             "ms_per_step": round(r, 3) if ok else "", "tok_s": round(batch * 1e3 / r, 1) if ok else "",
                             "status": "ok" if ok else r,
                             "bandwidth_floor_ms": round(ceiling_ms(seq_len, batch, args), 3)})
                if args.csv:
                    with open(args.csv, "w", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=list(rows[0]))
                        w.writeheader()
                        w.writerows(rows)

    print("\n| context | batch | " + " | ".join(f"{r} tok/s" for r in args.runs) + " |")
    print("|" + "---:|" * (2 + len(args.runs)))
    for seq_len in args.seq_lens:
        for batch in args.batch_sizes:
            cells = [next(x for x in rows if (x["seq_len"], x["batch"], x["run"]) == (seq_len, batch, r))
                     for r in args.runs]
            print(f"| {seq_len:,} | {batch} | " + " | ".join(str(c["tok_s"] or c["status"]) for c in cells) + " |")


if __name__ == "__main__":
    main()
