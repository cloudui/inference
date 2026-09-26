"""
sweep_context.py — Decode throughput vs context length, ours vs Hugging Face.

Usage:
    python benchmarks/sweep_context.py [--seq-lens 512 2048 ...] [--runs custom:cuda-graphs hf:compile-cg ...]

Runs bench_throughput.py / bench_throughput_hf.py once per (context length, implementation),
each in its own process so an OOM only loses that cell. Every run is logged to
benchmarks/results/throughput_runs.csv as usual (note "ctx-sweep <tag>"). Prints a markdown
table with tok/s, our speedup over the best HF mode, and the bandwidth our runs reach.

Byte model per decode step (Llama 3.1 8B, fp16, batch 1): 15.01 GB of weights (all layers +
lm_head; the embedding is one row) + 128 KiB of KV per cached token (32 layers x K,V x 8 heads
x 128 x 2 B). The ceiling columns use the ~820 GB/s the best isolated GEMVs reach on this card
(docs/perf_history/REPORT.md).
"""

import argparse
import datetime
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WEIGHT_BYTES = 15.009e9
KV_BYTES_PER_TOKEN = 32 * 2 * 8 * 128 * 2
PRACTICAL_GBS = 820.0

DEFAULT_RUNS = ["custom:cuda-graphs", "hf:eager-dynamic", "hf:eager-static", "hf:compile-cg"]
DEFAULT_SEQ_LENS = [512, 2048, 8192, 16384, 32768, 65536, 98304, 114688]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seq-lens", type=int, nargs="+", default=DEFAULT_SEQ_LENS)
    p.add_argument("--runs", nargs="+", default=DEFAULT_RUNS,
                   help="impl:mode, impl in {custom, hf}; custom modes: cuda-graphs, eager; "
                        "hf modes: see bench_throughput_hf.py")
    p.add_argument("--decode-steps", type=int, default=128)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--tag", default=datetime.datetime.now().strftime("%Y%m%d-%H%M"))
    p.add_argument("--out", type=Path, default=None, help="Also write the markdown table here")
    return p.parse_args()


def bench_cmd(run: str, seq_len: int, args) -> list[str]:
    impl, mode = run.split(":")
    common = ["--seq-len", str(seq_len), "--decode-steps", str(args.decode_steps),
              "--warmup", str(args.warmup), "--note", f"ctx-sweep {args.tag}"]
    if impl == "custom":
        return [sys.executable, str(HERE / "bench_throughput.py"), *common,
                *(["--cuda-graphs"] if mode == "cuda-graphs" else [])]
    return [sys.executable, str(HERE / "bench_throughput_hf.py"), *common, "--mode", mode]


def run_one(run: str, seq_len: int, args) -> float | str:
    """ms/token, or a short failure label."""
    proc = subprocess.run(bench_cmd(run, seq_len, args), capture_output=True, text=True)
    m = re.search(r"Per step:\s+([\d.]+) ms/tok", proc.stdout)
    if proc.returncode == 0 and m:
        return float(m.group(1))
    out = proc.stdout + proc.stderr
    if "OutOfMemoryError" in out or "out of memory" in out:
        return "OOM"
    last = [l for l in out.strip().splitlines() if l.strip()][-1:] or ["?"]
    print(f"    failed: {last[0][:200]}", flush=True)
    return "error"


def ceiling_ms(seq_len: int) -> float:
    return (WEIGHT_BYTES + seq_len * KV_BYTES_PER_TOKEN) / (PRACTICAL_GBS * 1e9) * 1e3


def fmt_cell(r: float | str) -> str:
    return f"{1e3 / r:.1f}" if isinstance(r, float) else r


def main():
    args = parse_args()
    results: dict[tuple[int, str], float | str] = {}
    for seq_len in args.seq_lens:
        for run in args.runs:
            print(f"[{args.tag}] seq_len={seq_len:>6}  {run:<20}", end=" ", flush=True)
            r = run_one(run, seq_len, args)
            results[seq_len, run] = r
            print(f"{r:.3f} ms/tok ({1e3 / r:.1f} tok/s)" if isinstance(r, float) else r, flush=True)

    hf_runs = [r for r in args.runs if r.startswith("hf:")]
    ours = next((r for r in args.runs if r.startswith("custom:")), None)
    header = ["context", *[f"{r} tok/s" for r in args.runs]]
    if ours and hf_runs:
        header.append("ours vs best HF")
    if ours:
        header += ["ours GB/s", "ours % of 820 GB/s ceiling", "KV share of bytes"]

    lines = ["| " + " | ".join(header) + " |", "|" + "---:|" * len(header)]
    for seq_len in args.seq_lens:
        row = [f"{seq_len:,}"] + [fmt_cell(results[seq_len, r]) for r in args.runs]
        ours_ms = results.get((seq_len, ours)) if ours else None
        if ours and hf_runs:
            hf_ms = [results[seq_len, r] for r in hf_runs if isinstance(results[seq_len, r], float)]
            row.append(f"{min(hf_ms) / ours_ms:.2f}x" if hf_ms and isinstance(ours_ms, float) else "")
        if ours:
            kv = seq_len * KV_BYTES_PER_TOKEN
            if isinstance(ours_ms, float):
                gbs = (WEIGHT_BYTES + kv) / (ours_ms * 1e-3) / 1e9
                row += [f"{gbs:.0f}", f"{ceiling_ms(seq_len) / ours_ms:.0%}"]
            else:
                row += ["", ""]
            row.append(f"{kv / (WEIGHT_BYTES + kv):.0%}")
        lines.append("| " + " | ".join(row) + " |")

    table = "\n".join(lines)
    print(f"\n{table}\n")
    if args.out:
        args.out.write_text(table + "\n")
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
