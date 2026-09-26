"""
run_log.py — Shared run log for the throughput benchmarks.

Every bench_throughput*.py run appends one row to benchmarks/results/throughput_runs.csv
with the commit and environment, so throughput history survives pod resets.
"""

import csv
import datetime
import platform
import subprocess
from pathlib import Path

import torch
import triton

RUN_LOG = Path(__file__).resolve().parent / "results" / "throughput_runs.csv"

# Columns added after rows were already logged; old rows get these values instead of blanks
BACKFILL = {"impl": "custom", "mode": ""}


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


def log_run(fields: dict) -> None:
    """Append one row: the given benchmark fields plus commit and environment."""
    row = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "commit": _git("rev-parse", "--short", "HEAD"),
        # the run log itself doesn't count, or every run after the first would look dirty
        "dirty": bool(_git("status", "--porcelain", "--untracked-files=no", "--",
                           ":(top)", ":(top,exclude)benchmarks/results")),
        **fields,
        "gpu": torch.cuda.get_device_name(),
        "cpu": _cpu_model(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda": torch.version.cuda,
    }
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    rows, fields_out = [], list(row)
    if RUN_LOG.exists():
        with open(RUN_LOG, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            old_fields = reader.fieldnames or []
        # keep old column order, append any new columns
        new_cols = [k for k in row if k not in old_fields]
        for r in rows:
            for k in new_cols:
                r[k] = BACKFILL.get(k, "")
        fields_out = old_fields + new_cols
    rows.append(row)
    with open(RUN_LOG, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields_out)
        w.writeheader()
        w.writerows(rows)
    print(f"  Logged to {RUN_LOG.relative_to(RUN_LOG.parents[2])}")
