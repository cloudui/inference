"""
Four summary charts for a talk/deck, from CSVs in this folder:

  roofline.png           roofline_gemms.csv + roofline_roofs.csv (+ batch_breakdown.csv for attention)
  batch1_breakdown.png   batch1_ops.csv
  observer_effect.png    observer_effect.csv
  trace_timeline.png     trace_step_b1.csv (one step from traces/decode_*_b1.nsys-rep)

    pip install matplotlib
    python docs/perf_history/deck_charts.py

Same palette and styling as the long-context charts. See DECK.md for how each input was measured.
"""

import csv

from long_context_charts import (AQUA, BLUE, BLUE_LIGHT, GRAY, GRAY_LIGHT, GRID, HERE, INK, INK2, MUTED,
                                 ORANGE, SURF, YELLOW, _axes, plt)

LAYERS = 32
GEMM_COUNT = {"qkv": LAYERS, "o_proj": LAYERS, "gate_up": LAYERS, "down": LAYERS, "lm_head": 1}
KV_BYTES_PER_TOKEN = 32 * 2 * 8 * 128 * 2


def _roofs():
    return {r["metric"]: float(r["value"]) for r in csv.DictReader(open(HERE / "roofline_roofs.csv"))}


def chart_roofline():
    roofs = _roofs()
    peak, bw, spec = roofs["peak_fp16_tflop_s"] * 1e12, roofs["practical_gb_s"] * 1e9, roofs["spec_gb_s"] * 1e9
    ridge = peak / bw

    # one point per batch size: every weight GEMM of a decode step, flops / bytes / time summed
    by_m = {}
    for r in csv.DictReader(open(HERE / "roofline_gemms.csv")):
        n = GEMM_COUNT[r["op"]]
        f, b, t = by_m.get(int(r["M"]), (0, 0, 0))
        by_m[int(r["M"])] = (f + n * int(r["flops"]), b + n * int(r["bytes"]), t + n * float(r["us"]) * 1e-6)
    ms = sorted(by_m)
    ai = [by_m[m][0] / by_m[m][1] for m in ms]
    perf = [by_m[m][0] / by_m[m][2] for m in ms]

    # attention: QK^T and PV for 4 q heads per KV head = 2 * 2 * 4 * 128 FLOPs per 512 B of K+V per token
    att = []
    for r in csv.DictReader(open(HERE / "batch_breakdown.csv")):
        b, ctx = int(r["batch"]), int(r["context"])
        t = (float(r["flash-decode_gen_ms"]) + float(r["flash-decode_reduce_ms"])) * 1e-3
        flops = 2 * 2 * 32 * 128 * ctx * b * LAYERS
        att.append((b, flops / (b * ctx * KV_BYTES_PER_TOKEN), flops / t))

    fig, ax = _axes((11, 6.4))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(color=GRID, lw=0.8, which="major")
    x = [0.3 * 1.06 ** i for i in range(200)]
    x = [v for v in x if v < 3000]
    ax.plot(x, [min(peak, bw * v) / 1e12 for v in x], color=INK2, lw=2, zorder=3)
    ax.plot(x, [min(peak, spec * v) / 1e12 for v in x], color=MUTED, lw=1, ls=(0, (4, 3)), zorder=2)
    ax.annotate(f"memory roof: {bw / 1e9:.0f} GB/s, best GEMV\n(dashed: {spec / 1e9:.0f} GB/s spec)", xy=(3, bw * 3 / 1e12),
                xytext=(-120, 60), textcoords="offset points", fontsize=8.5, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.annotate(f"compute roof: {peak / 1e12:.0f} TFLOP/s fp16 (measured, 8192³ GEMM)", xy=(ridge * 1.15, peak / 1e12),
                xytext=(0, 6), textcoords="offset points", fontsize=8.5, color=INK2)
    ax.axvline(ridge, color=GRID, lw=1, ls=":", zorder=1)
    ax.annotate(f"ridge ≈ {ridge:.0f} FLOP/B", xy=(ridge, 0.4), xytext=(5, 0), textcoords="offset points",
                fontsize=8.5, color=MUTED)

    ax.plot(ai, [p / 1e12 for p in perf], color=BLUE, lw=1.5, zorder=4, alpha=0.6)
    ax.scatter(ai, [p / 1e12 for p in perf], s=46, color=BLUE, edgecolor=SURF, lw=1.5, zorder=5,
               label="Weight GEMMs, whole decode step (cuBLAS)")
    for m, a, p in zip(ms, ai, perf):
        if m in (1, 8, 32, 128):
            ax.annotate(f"batch {m}", xy=(a, p / 1e12), xytext=(8, -12), textcoords="offset points", ha="left",
                        fontsize=9, color=INK)
    ax.scatter([a for _, a, _ in att], [p / 1e12 for _, _, p in att], s=46, color=ORANGE, edgecolor=SURF, lw=1.5,
               zorder=5, label="Attention (flash-decode), batch 1–128")
    ax.annotate("batch 1: too little KV to fill the GPU", xy=(att[0][1], att[0][2] / 1e12), xytext=(10, -4),
                textcoords="offset points", fontsize=8.5, color=INK2)
    ax.annotate("attention stays at ~4 FLOP/B at every batch size:\neach sequence reads its own KV",
                xy=(att[-1][1], att[-1][2] / 1e12), xytext=(60, -2), textcoords="offset points", fontsize=8.5,
                color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_xlim(0.3, 2000)
    ax.set_ylim(0.3, 400)
    ax.set_xlabel("Arithmetic intensity (FLOP per byte of DRAM traffic)")
    ax.set_ylabel("Achieved TFLOP/s")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_title("Decode on the roofline: batching slides the GEMMs up the bandwidth roof toward the ridge",
                 loc="left", color=INK, fontsize=12, pad=12)
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(HERE / "roofline.png", facecolor=SURF)
    plt.close(fig)


def chart_batch1_breakdown():
    ms = {r["op"]: float(r["ms"]) for r in csv.DictReader(open(HERE / "batch1_ops.csv"))}
    gemv = [("gate_up_proj", "gate + up"), ("down_proj", "down"), ("qkv_proj", "QKV"), ("out_proj", "out"),
            ("lm_head", "lm_head")]
    rest = ms["flash_decode"] + ms["rmsnorm"] + ms["rope_kv_write"] + ms["swiglu"] + ms["other (embed, position update)"]
    gap = ms["gaps between kernels (CUDA graphs)"]
    total = sum(ms.values())
    gemv_total = sum(ms[k] for k, _ in gemv)

    fig, ax = _axes((11, 3.6))
    ax.spines["left"].set_visible(False)
    left = 0.0
    shades = [BLUE, "#4a8fe0", "#6aa5e8", "#8bbbef", BLUE_LIGHT]
    for (k, label), c in zip(gemv, shades):
        ax.barh(0, ms[k], left=left, height=0.5, color=c, edgecolor=SURF, lw=2)
        if ms[k] > 1.0:
            ax.text(left + ms[k] / 2, 0, f"{label}\n{ms[k]:.1f} ms", ha="center", va="center", fontsize=9,
                    color="white" if c in (BLUE, "#4a8fe0") else INK)
        left += ms[k]
    ax.barh(0, rest, left=left, height=0.5, color=ORANGE, edgecolor=SURF, lw=1)
    left += rest
    ax.barh(0, gap, left=left, height=0.5, color=GRAY_LIGHT, edgecolor=SURF, lw=1)

    ax.annotate("", xy=(0, 0.42), xytext=(gemv_total, 0.42), arrowprops=dict(arrowstyle="<->", color=INK2, lw=1))
    ax.text(gemv_total / 2, 0.47, f"weight GEMVs: {gemv_total:.2f} ms ({gemv_total / total:.0%})", ha="center",
            fontsize=10, color=INK)
    ax.annotate(f"everything else: {rest + gap:.2f} ms ({(rest + gap) / total:.1%})\n"
                f"attention {ms['flash_decode']:.2f} · norms {ms['rmsnorm']:.2f} · RoPE+KV {ms['rope_kv_write']:.2f} · "
                f"SwiGLU {ms['swiglu']:.2f} · gaps {gap:.2f}",
                xy=(gemv_total + (rest + gap) / 2, -0.25), xytext=(-8, -38), textcoords="offset points", ha="right",
                fontsize=9, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_xlim(0, total * 1.01)
    ax.set_ylim(-0.95, 0.75)
    ax.set_yticks([])
    ax.set_xlabel("ms per decode step (batch 1, 512 context, CUDA graphs)")
    ax.set_title(f"Batch-1 decode step: {total:.2f} ms, and almost all of it is streaming weights",
                 loc="left", color=INK, fontsize=12, pad=10)
    fig.tight_layout()
    fig.savefig(HERE / "batch1_breakdown.png", facecolor=SURF)
    plt.close(fig)


def chart_observer():
    rows = list(csv.DictReader(open(HERE / "observer_effect.csv")))
    kernel_ms = sum(float(r["ms"]) for r in csv.DictReader(open(HERE / "batch1_ops.csv"))
                    if not r["op"].startswith("gaps"))
    order = [("eager", "on", "Eager,\nprofiler hooks on"), ("eager", "off", "Eager,\nhooks off"),
             ("cuda-graphs", "on", "CUDA graphs,\nhooks on"), ("cuda-graphs", "off", "CUDA graphs,\nhooks off")]
    fig, ax = _axes((11, 5.2))
    ax.grid(axis="x", color=GRID, lw=1)
    ax.spines["left"].set_visible(False)
    for i, (mode, hooks, label) in enumerate(order):
        r = next(r for r in rows if r["mode"] == mode and r["profiler_hooks"] == hooks)
        step = float(r["median_ms"])
        y = len(order) - 1 - i
        ax.barh(y, kernel_ms, height=0.56, color=BLUE, edgecolor=SURF, lw=2, zorder=2,
                label="GPU kernel time (identical in every mode)" if i == 0 else None)
        ax.barh(y, step - kernel_ms, left=kernel_ms, height=0.56, color=ORANGE, edgecolor=SURF, lw=2, zorder=2,
                label="GPU idle, waiting on the CPU to launch" if i == 0 else None)
        ax.text(step + 0.2, y, f"{step:.2f} ms  ·  {float(r['median_tok_s']):.1f} tok/s", va="center", fontsize=9.5,
                color=INK)
        ax.text(-0.3, y, label, va="center", ha="right", fontsize=9.5, color=INK)
    ax.set_yticks([])
    ax.set_xlim(0, 30)
    ax.set_xlabel("ms per decode step (batch 1, 512 context; median of 3 runs)")
    ax.set_title("The observer effect: profiler hooks cost 21% in eager mode, and 0% under CUDA graphs",
                 loc="left", color=INK, fontsize=12, pad=30)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(HERE / "observer_effect.png", facecolor=SURF)
    plt.close(fig)


OP_COLORS = {"qkv_proj": BLUE, "out_proj": BLUE, "gate_up_proj": BLUE, "down_proj": BLUE, "lm_head": BLUE,
             "flash_decode": ORANGE, "rmsnorm": AQUA, "rope_kv_write": YELLOW, "swiglu": YELLOW, "other": GRAY}


def chart_trace():
    rows = list(csv.DictReader(open(HERE / "trace_step_b1.csv")))
    modes = [("eager+hooks", "Eager, hooks on"), ("eager", "Eager, hooks off"), ("cuda-graphs", "CUDA graphs")]
    fig, (ax, zx) = plt.subplots(2, 1, figsize=(12, 6.2), dpi=150, gridspec_kw={"height_ratios": [1, 1.25]})
    fig.patch.set_facecolor(SURF)
    for a in (ax, zx):
        a.set_facecolor(SURF)
        for side in ("top", "right", "left"):
            a.spines[side].set_visible(False)

    # zoom window: layer 3's kernels in each lane (rmsnorm .. down), aligned to its first kernel
    def layer_kernels(mode, layer=3):
        ks = [r for r in rows if r["mode"] == mode and r["op"] != "STEP (nvtx range)"]
        ks.sort(key=lambda r: float(r["start_us"]))
        starts = [i for i, r in enumerate(ks) if r["op"] == "qkv_proj"]
        i0 = starts[layer] - 1  # the norm before it
        return ks[i0:starts[layer + 1] - 1]

    for y, (mode, label) in enumerate(reversed(modes)):
        ks = [r for r in rows if r["mode"] == mode]
        step = next(float(r["end_us"]) for r in ks if r["op"] == "STEP (nvtx range)")
        ax.barh(y, step / 1e3, height=0.62, color=GRAY_LIGHT, lw=0, zorder=1)
        for r in ks:
            if r["op"] == "STEP (nvtx range)":
                continue
            s, e = float(r["start_us"]) / 1e3, float(r["end_us"]) / 1e3
            ax.barh(y, e - s, left=s, height=0.62, color=OP_COLORS[r["op"]], lw=0, zorder=2)
        ax.text(step / 1e3 + 0.25, y, f"{step / 1e3:.1f} ms", va="center", fontsize=9, color=INK)
        ax.text(-0.3, y, label, va="center", ha="right", fontsize=9.5, color=INK)

        lk = layer_kernels(mode)
        t0 = float(lk[0]["start_us"])
        span = float(lk[-1]["end_us"]) - t0
        zx.barh(y, span, height=0.62, color=GRAY_LIGHT, lw=0, zorder=1)
        for r in lk:
            s, e = float(r["start_us"]) - t0, float(r["end_us"]) - t0
            zx.barh(y, e - s, left=s, height=0.62, color=OP_COLORS[r["op"]], lw=0, zorder=2)
        zx.text(span + 6, y, f"{span:.0f} µs", va="center", fontsize=9, color=INK)
        zx.text(-6, y, label, va="center", ha="right", fontsize=9.5, color=INK)
        if mode == "cuda-graphs":
            for r in lk:
                s, e = float(r["start_us"]) - t0, float(r["end_us"]) - t0
                name = {"qkv_proj": "QKV", "out_proj": "out", "gate_up_proj": "gate + up GEMV",
                        "down_proj": "down GEMV"}.get(r["op"])
                if name:
                    zx.text((s + e) / 2, y, name, ha="center", va="center", fontsize=8.5, color="white")

    ax.set_yticks([])
    ax.set_xlim(0, 34)
    ax.set_xlabel("ms from the start of the decode step (NVTX range, includes argmax + token readback)")
    ax.set_title("One batch-1 decode step in Nsight Systems: same 325 kernels, very different gaps",
                 loc="left", color=INK, fontsize=12, pad=10)
    zx.set_yticks([])
    zx.set_xlim(0, 1000)
    zx.set_xlabel("µs, one transformer layer (layer 3), aligned to its first kernel")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, ORANGE, AQUA, YELLOW, GRAY_LIGHT)]
    zx.legend(handles, ["weight GEMV", "flash-decode", "RMSNorm", "RoPE / SwiGLU", "GPU idle"], loc="lower right",
              ncol=5, frameon=False, fontsize=8.5, labelcolor=INK, bbox_to_anchor=(1, -0.42))
    fig.text(0.01, 0.005, "Captured with nsys 2026.1 (--cuda-graph-trace=node). Tracing itself adds CPU cost to the "
             "eager lanes; the untraced numbers are in observer_effect.png.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(HERE / "trace_timeline.png", facecolor=SURF)
    plt.close(fig)


if __name__ == "__main__":
    chart_roofline()
    chart_batch1_breakdown()
    chart_observer()
    chart_trace()
    print("wrote roofline.png, batch1_breakdown.png, observer_effect.png, trace_timeline.png")
