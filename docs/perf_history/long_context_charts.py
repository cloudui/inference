"""
Charts for LONG_CONTEXT.md, from long_context_sweep.csv, long_context_breakdown_32k.csv and
long_context_milestones.csv.

    pip install matplotlib
    python docs/perf_history/long_context_charts.py

Same palette and styling as throughput_history.png.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter

HERE = Path(__file__).resolve().parent

SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983", "#e4e3df"
# categorical slots in fixed order: ours, then the HF modes
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
BLUE_LIGHT = "#9ec3ef"
GRAY, GRAY_LIGHT = "#a8a79f", "#d6d5cf"

WEIGHT_BYTES = 15.009e9
KV_BYTES_PER_TOKEN = 32 * 2 * 8 * 128 * 2
PRACTICAL_GBS = 820e9

plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "font.family": "DejaVu Sans"})


def _axes(figsize):
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return fig, ax


def _ctx_label(n):
    return f"{n // 1024}K" if n >= 1024 else str(n)


def load_sweep():
    series = {}
    for r in csv.DictReader(open(HERE / "long_context_sweep.csv")):
        key = f"{r['impl']}:{r['mode']}"
        series.setdefault(key, []).append((int(r["context"]), float(r["tok_s"]) if r["tok_s"] else None))
    return {k: sorted(v) for k, v in series.items()}


def ceiling_tok_s(ctx):
    return PRACTICAL_GBS / (WEIGHT_BYTES + ctx * KV_BYTES_PER_TOKEN)


def chart_throughput(series):
    fig, ax = _axes((11, 6.2))
    ax.grid(axis="y", color=GRID, lw=1)
    ctxs = [c for c, _ in series["custom:cuda-graphs"]]

    # practical bandwidth ceiling
    dense = [ctxs[0] * (ctxs[-1] / ctxs[0]) ** (i / 200) for i in range(201)]
    ax.plot(dense, [ceiling_tok_s(c) for c in dense], color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=2)
    ax.annotate("bandwidth ceiling (820 GB/s)", xy=(ctxs[1], ceiling_tok_s(ctxs[1])), xytext=(0, 7),
                textcoords="offset points", ha="left", fontsize=8.5, color=INK2)

    lines = [
        ("custom:cuda-graphs", "Ours (Triton + CUDA graphs)", BLUE, 2.4),
        ("hf:eager-dynamic", "HF eager, DynamicCache", ORANGE, 2),
        ("hf:compile-cg", "HF torch.compile + CUDA graphs, StaticCache", YELLOW, 2),
        ("hf:eager-static", "HF eager, StaticCache", AQUA, 2),
    ]
    for key, label, color, lw in lines:
        pts = [(c, v) for c, v in series[key] if v is not None]
        x, y = zip(*pts)
        ax.plot(x, y, color=color, lw=lw, marker="o", ms=5.5, mec=SURF, mew=1.5, label=label, zorder=3,
                solid_joinstyle="round")
        missing = [c for c, v in series[key] if v is None]
        for c in missing:
            ax.annotate("OOM", xy=(x[-1], y[-1]), xytext=(8, -3 if key == "hf:compile-cg" else 5),
                        textcoords="offset points", fontsize=8, color=INK2)

    # selective direct labels: endpoints of ours and HF's best mode
    ours = dict(series["custom:cuda-graphs"])
    dyn = dict(series["hf:eager-dynamic"])
    ax.annotate(f"{ours[ctxs[0]]:.1f}", xy=(ctxs[0], ours[ctxs[0]]), xytext=(0, -15), textcoords="offset points",
                ha="center", fontsize=9, color=INK, fontweight="bold")
    ax.annotate(f"{ours[ctxs[-1]]:.1f}", xy=(ctxs[-1], ours[ctxs[-1]]), xytext=(-4, -14), textcoords="offset points",
                ha="right", fontsize=9, color=INK, fontweight="bold")
    ax.annotate(f"{dyn[ctxs[-1]]:.1f}", xy=(ctxs[-1], dyn[ctxs[-1]]), xytext=(-4, -14), textcoords="offset points",
                ha="right", fontsize=9, color=INK)
    # bracket between ours and HF's best at the longest context
    xb = ctxs[-1] * 1.13
    ax.annotate("", xy=(xb, ours[ctxs[-1]] - 0.6), xytext=(xb, dyn[ctxs[-1]] + 0.6),
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.9))
    ax.annotate(f"{ours[ctxs[-1]] / dyn[ctxs[-1]]:.1f}×\nfaster", xy=(xb, (ours[ctxs[-1]] + dyn[ctxs[-1]]) / 2),
                xytext=(6, 0), textcoords="offset points", ha="left", va="center", fontsize=9, color=INK)

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(ctxs))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _ctx_label(int(round(v)))))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.set_xlim(ctxs[0] / 1.35, ctxs[-1] * 1.6)
    # 96K and 112K sit close on a log axis: nudge their labels apart
    fig.canvas.draw()
    for lbl, c in zip(ax.get_xticklabels(), ctxs):
        if c == ctxs[-2]:
            lbl.set_horizontalalignment("right")
        elif c == ctxs[-1]:
            lbl.set_horizontalalignment("left")
    ax.set_ylim(0, 62)
    ax.set_xlabel("Context length (tokens already in the KV cache)")
    ax.set_ylabel("Decode throughput (tok/s)")
    ax.set_title("Llama 3.1 8B decode, batch 1, fp16 — RTX PRO 4500 Blackwell",
                 loc="left", color=INK, fontsize=12, pad=12)
    ax.legend(loc="lower left", frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(HERE / "long_context_throughput.png", facecolor=SURF)
    plt.close(fig)


def chart_bytes(series):
    """Measured ms/token vs the time to stream the weights + KV at 820 GB/s."""
    fig, ax = _axes((11, 5.6))
    ax.grid(axis="y", color=GRID, lw=1)
    pts = series["custom:cuda-graphs"]
    x = range(len(pts))
    w_ms = [WEIGHT_BYTES / PRACTICAL_GBS * 1e3] * len(pts)
    kv_ms = [c * KV_BYTES_PER_TOKEN / PRACTICAL_GBS * 1e3 for c, _ in pts]
    measured = [1e3 / v for _, v in pts]

    ax.bar(x, w_ms, width=0.62, color=BLUE, edgecolor=SURF, lw=1.5, label="Stream 15.0 GB of weights", zorder=2)
    ax.bar(x, kv_ms, width=0.62, bottom=w_ms, color=BLUE_LIGHT, edgecolor=SURF, lw=1.5,
           label="Stream the KV cache (128 KiB / token)", zorder=2)
    ax.scatter(x, measured, s=70, color=INK, edgecolor=SURF, lw=1.5, zorder=4, label="Measured")
    for i, (m, w, k) in enumerate(zip(measured, w_ms, kv_ms)):
        ax.annotate(f"{m:.1f} ms\n{(w + k) / m:.0%}", xy=(i, m), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=8.5, color=INK2, linespacing=1.15)

    ax.set_xticks(list(x), [_ctx_label(c) for c, _ in pts])
    ax.set_ylim(0, max(measured) * 1.22)
    ax.set_xlabel("Context length")
    ax.set_ylabel("ms per token")
    ax.set_title("Where the time goes: every token is a memory read  (labels: measured ms, % of ceiling)",
                 loc="left", color=INK, fontsize=12, pad=12)
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(HERE / "long_context_bytes.png", facecolor=SURF)
    plt.close(fig)


def chart_breakdown():
    data = {}
    for r in csv.DictReader(open(HERE / "long_context_breakdown_32k.csv")):
        data.setdefault(r["impl"], {})[r["category"]] = float(r["gpu_ms"])
    impls = ["HF StaticCache", "HF DynamicCache", "Ours"]
    cats = [("Weight GEMVs", GRAY), ("Attention", BLUE), ("KV cache copies", ORANGE), ("Other", GRAY_LIGHT)]

    fig, ax = _axes((11, 4.6))
    ax.grid(axis="x", color=GRID, lw=1)
    for i, impl in enumerate(impls):
        left = 0.0
        for cat, color in cats:
            v = data[impl].get(cat, 0.0)
            if v <= 0:
                continue
            ax.barh(i, v, left=left, height=0.56, color=color, edgecolor=SURF, lw=1.5,
                    label=cat if i == 0 else None, zorder=2)
            if v >= 4:
                ax.text(left + v / 2, i, f"{v:.1f}", ha="center", va="center", fontsize=8.5,
                        color="white" if color in (BLUE, ORANGE) else INK)
            left += v
        ax.text(left + 1.2, i, f"{left:.1f} ms", va="center", fontsize=9.5, color=INK, fontweight="bold")

    ax.set_yticks(range(len(impls)), impls)
    ax.tick_params(axis="y", length=0, labelsize=10)
    ax.set_xlim(0, 112)
    ax.set_xlabel("GPU time per decode step (ms), 32K context")
    ax.set_title("Why HF's StaticCache is slow at long context: a mask turns off GQA and FlashAttention",
                 loc="left", color=INK, fontsize=12, pad=12)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.2), frameon=False, fontsize=9, labelcolor=INK, ncol=4)
    fig.tight_layout()
    fig.savefig(HERE / "long_context_breakdown_32k.png", facecolor=SURF)
    plt.close(fig)


def chart_milestones():
    """Optimization #8 (flash-decode polish), commit by commit, at long context."""
    rows = [r for r in csv.DictReader(open(HERE / "long_context_milestones.csv")) if r["cuda_graphs"] == "False"]
    chain = [("8cea929", "8cea929  before #8"), ("a496b35", "a496b35  reversed grid order"),
             ("a861708", "a861708  reduce kernel rewrite"), ("e9610a1", "e9610a1  exp2 / log2"),
             ("9bd7daf", "9bd7daf  fixed 16 KV splits")]
    # ordered stages: one blue ramp, light -> dark
    ramp = ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"]
    ctxs = [32768, 65536, 114688]
    val = {(r["commit"], int(r["context"])): float(r["median_tok_s"]) for r in rows}

    fig, ax = _axes((11, 5.4))
    ax.grid(axis="y", color=GRID, lw=1)
    width = 0.16
    for i, ((c, label), color) in enumerate(zip(chain, ramp)):
        xs = [j + (i - 2) * width for j in range(len(ctxs))]
        ys = [val[c, ctx] for ctx in ctxs]
        ax.bar(xs, ys, width=width, color=color, edgecolor=SURF, lw=1.5, label=label, zorder=2)
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.5, f"{y:.1f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
    for j, ctx in enumerate(ctxs):
        before, after = val["8cea929", ctx], val["9bd7daf", ctx]
        ax.text(j, max(before, after) + 4.2, f"+{after / before - 1:.0%}", ha="center", fontsize=11,
                color=INK, fontweight="bold")

    ax.set_xticks(range(len(ctxs)), [f"{_ctx_label(c)} context" for c in ctxs])
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 50)
    ax.set_ylabel("Decode throughput (tok/s), eager")
    ax.set_title("Optimization #8 at long context: +0.2% at 512 tokens, +29% at 112K",
                 loc="left", color=INK, fontsize=12, pad=12)
    ax.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(HERE / "long_context_milestones.png", facecolor=SURF)
    plt.close(fig)


if __name__ == "__main__":
    s = load_sweep()
    chart_throughput(s)
    chart_bytes(s)
    chart_breakdown()
    chart_milestones()
    print("wrote long_context_{throughput,bytes,breakdown_32k,milestones}.png")
