"""
Charts for BATCH.md, from batch_sweep.csv, batch_breakdown.csv and batch_splits_kernel.csv.

    pip install matplotlib
    python docs/perf_history/batch_charts.py

Same palette and styling as the long-context charts (vLLM in gray, as in long_context_vllm.png).
"""

import csv

from long_context_charts import (BLUE, BLUE_LIGHT, GRAY, GRAY_LIGHT, GRID, HERE, INK, INK2, MUTED,
                                 ORANGE, AQUA, SURF, _axes, plt)

OURS = "custom:cuda-graphs+loop"   # timed like vLLM: host clock, tokens read back every step
VLLM = "vllm:default"


def load_sweep():
    out = {}
    for r in csv.DictReader(open(HERE / "batch_sweep.csv")):
        if r["ms_per_step"]:
            out[r["engine_version"], r["run"], int(r["context"]), int(r["batch"])] = r
    return out


def _batches(sweep, ctx):
    return sorted({b for (_, run, c, b) in sweep if c == ctx and run == VLLM})


def _style_batch_axis(ax, batches):
    ax.set_xscale("log", base=2)
    ax.set_xticks(batches, [str(b) for b in batches])
    ax.minorticks_off()
    ax.set_xlabel("Batch size (sequences decoding in lockstep)")


def chart_throughput(sweep):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), dpi=150, gridspec_kw={"width_ratios": [8, 5]})
    fig.patch.set_facecolor(SURF)
    for ax, ctx in zip(axes, [512, 4096]):
        ax.set_facecolor(SURF)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", color=GRID, lw=1)
        bs = _batches(sweep, ctx)
        floor = [b * 1e3 / float(sweep["vllm 0.30.0", VLLM, ctx, b]["bandwidth_floor_ms"]) for b in bs]
        ax.plot(bs, floor, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=2, label="Bandwidth ceiling (820 GB/s)")
        ours = [float(sweep["adaptive-splits", OURS, ctx, b]["tok_s"]) for b in bs]
        vllm = [float(sweep["vllm 0.30.0", VLLM, ctx, b]["tok_s"]) for b in bs]
        ax.plot(bs, vllm, color=GRAY, lw=2, marker="o", ms=6, mec=SURF, mew=1.5, zorder=3, label="vLLM 0.30")
        ax.plot(bs, ours, color=BLUE, lw=2.4, marker="o", ms=7, mec=SURF, mew=1.5, zorder=4, label="Ours")
        ax.annotate(f"{ours[-1]:,.0f}", xy=(bs[-1], ours[-1]), xytext=(-8, 6), textcoords="offset points",
                    ha="right", fontsize=9, color=INK)
        _style_batch_axis(ax, bs)
        ax.set_ylim(0, max(floor) * 1.08)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.set_title(f"{ctx:,}-token context", loc="left", color=INK, fontsize=11)
    axes[0].set_ylabel("Decode throughput, all sequences (tok/s)")
    axes[0].legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle("Batched decode, Llama 3.1 8B fp16: ours and vLLM 0.30 overlap (within 1.6%) at every batch size", x=0.01, ha="left", color=INK, fontsize=12.5)
    fig.tight_layout()
    fig.savefig(HERE / "batch_throughput.png", facecolor=SURF)
    plt.close(fig)


def chart_vs_vllm(sweep):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150, gridspec_kw={"width_ratios": [8, 5]}, sharey=True)
    fig.patch.set_facecolor(SURF)
    for ax, ctx in zip(axes, [512, 4096]):
        ax.set_facecolor(SURF)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", color=GRID, lw=1)
        bs = _batches(sweep, ctx)
        x = range(len(bs))
        w = 0.38
        for off, ver, color, label in [(-w / 2, "baseline", BLUE_LIGHT, "Before: 16 KV splits at every batch size"),
                                       (w / 2, "adaptive-splits", BLUE, "After: splits picked by batch size")]:
            pct = [(float(sweep[ver, OURS, ctx, b]["tok_s"]) / float(sweep["vllm 0.30.0", VLLM, ctx, b]["tok_s"]) - 1) * 100
                   for b in bs]
            ax.bar([i + off for i in x], pct, width=w, color=color, edgecolor=SURF, lw=1.5, zorder=2, label=label)
            if ver == "adaptive-splits":
                for i, p in enumerate(pct):
                    ax.text(i + off, p + (0.12 if p >= 0 else -0.12), f"{p:+.1f}%", ha="center",
                            va="bottom" if p >= 0 else "top", fontsize=8, color=INK)
        ax.axhline(0, color=INK2, lw=1, zorder=3)
        ax.set_xticks(list(x), [str(b) for b in bs])
        ax.tick_params(axis="x", length=0)
        ax.set_xlabel("Batch size")
        ax.set_title(f"{ctx:,}-token context", loc="left", color=INK, fontsize=11)
    axes[0].set_ylabel("Ours vs vLLM throughput (%; above 0 = we're faster)")
    axes[0].set_ylim(-4, 2.2)
    axes[0].legend(loc="lower left", frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle("Ours vs vLLM 0.30 by batch size, both timed with the tokens read back every step",
                 x=0.01, ha="left", color=INK, fontsize=12.5)
    fig.tight_layout()
    fig.savefig(HERE / "batch_vs_vllm.png", facecolor=SURF)
    plt.close(fig)


def chart_breakdown():
    rows = list(csv.DictReader(open(HERE / "batch_breakdown.csv")))
    bs = [int(r["batch"]) for r in rows]
    parts = [("Weight GEMMs (cuBLAS)", BLUE, lambda r: float(r["gemm_ms"])),
             ("Attention (flash-decode)", ORANGE, lambda r: float(r["flash-decode_gen_ms"]) + float(r["flash-decode_reduce_ms"])),
             ("Norms, SwiGLU, RoPE, other", GRAY_LIGHT,
              lambda r: sum(float(r[k]) for k in ("rmsnorm_ms", "swiglu_ms", "rope+kv_write_ms", "other_ms")))]
    fig, ax = _axes((11, 5.4))
    ax.grid(axis="y", color=GRID, lw=1)
    x = list(range(len(bs)))
    bottom = [0.0] * len(bs)
    for label, color, get in parts:
        vals = [get(r) for r in rows]
        ax.bar(x, vals, bottom=bottom, width=0.6, color=color, edgecolor=SURF, lw=2, label=label, zorder=2)
        bottom = [b + v for b, v in zip(bottom, vals)]
    for i, r in enumerate(rows):
        att = float(r["flash-decode_gen_ms"]) + float(r["flash-decode_reduce_ms"])
        ax.text(i, bottom[i] + 0.4, f"{bottom[i]:.1f} ms\nattention {att / bottom[i]:.0%}", ha="center",
                fontsize=8.5, color=INK)
    ax.set_xticks(x, [str(b) for b in bs])
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, max(bottom) * 1.18)
    ax.set_xlabel("Batch size (600-token context)")
    ax.set_ylabel("GPU time per decode step (ms)")
    ax.set_title("Where a decode step goes as batch grows: weights are read once, KV once per sequence",
                 loc="left", color=INK, fontsize=12, pad=12)
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(HERE / "batch_breakdown.png", facecolor=SURF)
    plt.close(fig)


def chart_splits():
    import sys
    sys.path.insert(0, str(HERE.parents[1]))
    from kernels.flash_decode import default_num_splits

    rows = list(csv.DictReader(open(HERE / "batch_splits_kernel.csv")))
    ctxs = sorted({int(r["context"]) for r in rows})
    fig, axes = plt.subplots(1, len(ctxs), figsize=(12, 4.4), dpi=150, sharey=True)
    fig.patch.set_facecolor(SURF)
    cols = {1: "us_1_split", 2: "us_2_splits", 4: "us_4_splits", 8: "us_8_splits", 16: "us_16_splits"}
    for ax, ctx in zip(axes, ctxs):
        ax.set_facecolor(SURF)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", color=GRID, lw=1)
        rs = [r for r in rows if int(r["context"]) == ctx]
        bs = [int(r["batch"]) for r in rs]
        best = [min(float(r[c]) for c in cols.values()) for r in rs]
        fixed16 = [float(r["us_16_splits"]) / b for r, b in zip(rs, best)]
        rule = [float(r[cols[default_num_splits(int(r["batch"]))]]) / b for r, b in zip(rs, best)]
        ax.plot(bs, fixed16, color=BLUE_LIGHT, lw=2, marker="o", ms=6, mec=SURF, mew=1.5, label="Always 16 splits (before)")
        ax.plot(bs, rule, color=BLUE, lw=2.4, marker="o", ms=7, mec=SURF, mew=1.5, label="Splits picked by batch size (after)")
        _style_batch_axis(ax, bs)
        ax.set_xlabel("Batch size")
        ax.set_title(f"{ctx // 1024}K context" if ctx >= 1024 else f"{ctx} context", loc="left", color=INK, fontsize=11)
    axes[0].set_ylabel("Kernel time vs best split count (1.0 = best)")
    axes[0].set_ylim(0.95, 1.32)
    axes[0].legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle("Flash-decode kernel alone (one layer, cold L2): 16 KV splits only pay at batch 1-2",
                 x=0.01, ha="left", color=INK, fontsize=12.5)
    fig.tight_layout()
    fig.savefig(HERE / "batch_splits_kernel.png", facecolor=SURF)
    plt.close(fig)


if __name__ == "__main__":
    s = load_sweep()
    chart_throughput(s)
    chart_vs_vllm(s)
    chart_breakdown()
    chart_splits()
    print("wrote batch_{throughput,vs_vllm,breakdown,splits_kernel}.png")
