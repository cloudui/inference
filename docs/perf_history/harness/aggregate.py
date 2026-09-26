"""Aggregate sweep JSONs (../raw) -> CSV + chart. Usage: python aggregate.py <outdir>  (needs matplotlib)"""
import csv, json, os, statistics, subprocess, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1]
RAW = os.path.join(HERE, "..", "raw")
os.makedirs(OUT, exist_ok=True)
REPO = "/workspace/inference"
commits = open(os.path.join(HERE, "commits.txt")).read().split()
summaries = json.load(open(os.path.join(HERE, "summaries.json")))  # hash -> {summary, notes, category}


def load(tag, c):
    f = os.path.join(RAW, f"{tag}_{c}.json")
    return json.load(open(f)) if os.path.exists(f) else None


rows = []
prev = None
for c in commits:
    date, msg = subprocess.check_output(["git", "-C", REPO, "log", "-1", "--format=%ad|%s", "--date=short", c], text=True).strip().split("|", 1)
    a, b = load("A_off", c), load("B_on", c)
    reps = [r for r in (load("R1_off", c), load("R2_off", c)) if r and r["status"] == "ok"]
    ok = a and a["status"] == "ok"
    med = round(a["median_tok_s"], 2) if ok else None
    best = round(statistics.median([a["median_tok_s"]] + [r["median_tok_s"] for r in reps]), 2) if ok else None
    s = summaries.get(c, {})
    row = dict(
        commit=c, date=date, message=msg, diff_summary=s.get("summary", ""),
        status="ok" if ok else "error",
        median_tok_s=med,
        min_tok_s=round(a["min_tok_s"], 2) if ok else None,
        max_tok_s=round(a["max_tok_s"], 2) if ok else None,
        best_estimate_tok_s=best,
        delta_vs_prev=round(best - prev, 2) if ok and prev is not None else None,
        delta_pct_vs_prev=round(100 * (best - prev) / prev, 1) if ok and prev is not None else None,
        repeat_medians_tok_s=";".join(f"{r['median_tok_s']:.2f}" for r in reps) or None,
        hooks_on_median_tok_s=round(b["median_tok_s"], 2) if b and b["status"] == "ok" else None,
        ms_per_tok=round(a["median_ms_per_tok"], 3) if ok else None,
        cpu_enqueue_ms_per_tok=round(a["median_enqueue_ms_per_tok"], 3) if ok else None,
        hooks_on_cpu_enqueue_ms_per_tok=round(b["median_enqueue_ms_per_tok"], 3) if b and b["status"] == "ok" else None,
        output_finite=a.get("output_finite") if ok else None,
        error_stage=None if ok else (a or {}).get("stage"),
        error=None if ok else (a or {}).get("error", "no result"),
        notes=s.get("notes", ""),
    )
    rows.append(row)
    if ok:
        prev = best

with open(os.path.join(OUT, "performance_history.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

# ---- chart -------------------------------------------------------------
okr = [r for r in rows if r["status"] == "ok"]
x = list(range(len(okr)))
y = [r["best_estimate_tok_s"] for r in okr]
yh = [r["hooks_on_median_tok_s"] for r in okr]
lo = [r["min_tok_s"] for r in okr]
hi = [r["max_tok_s"] for r in okr]

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(15, 7.2), dpi=150)
fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
ax.grid(axis="y", color=GRID, lw=1); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)

ax.fill_between(x, lo, hi, color=BLUE, alpha=0.12, lw=0)
ax.plot(x, y, color=BLUE, lw=2, marker="o", ms=5, mec=SURF, mew=1.5, solid_joinstyle="round",
        label="Profiler hooks stubbed (primary; band = min–max of 5 runs)")
xh = [i for i, v in zip(x, yh) if v is not None]
ax.plot(xh, [v for v in yh if v is not None], color=ORANGE, lw=2, marker="o", ms=4, mec=SURF, mew=1.2,
        label="As committed (record_function hooks live)")

hf = {}
for f in os.listdir(RAW):
    if f.startswith("HF_") and f.endswith(".json"):
        j = json.load(open(os.path.join(RAW, f)))
        hf[j["mode"]] = j
hf_labels = {"eager-dynamic": "HF eager, DynamicCache", "eager-static": "HF eager, StaticCache",
             "compile-static": "HF torch.compile, StaticCache", "compile-cg": "HF compile + CUDA graphs"}
for k, v in hf.items():
    if v.get("status") == "ok" and k in ("eager-dynamic", "compile-cg"):
        ax.axhline(v["median_tok_s"], color=INK2, lw=1, ls=(0, (1, 2)))
        ax.text(16.6, v["median_tok_s"] + 0.2, f"{hf_labels[k]}: {v['median_tok_s']:.1f}",
                ha="right", va="bottom", fontsize=8.5, color=INK2)

# label the biggest jumps (primary series), above the line
jumps = sorted([(y[i] - y[i - 1], i) for i in range(1, len(okr))], reverse=True)[:5]
# hand-placed (dx, dy, ha) per label so nothing collides; default for any other commit
PLACE = {"7d5c517": (-10, 66, "right"), "5a7b6c0": (-8, 78, "center"), "d326973": (18, 34, "left"),
         "cd9b714": (-12, 62, "right"), "ef70981": (14, 34, "left")}
for d, i in jumps:
    dx, dy, ha = PLACE.get(okr[i]["commit"], (0, 30, "center"))
    ax.annotate(f"{okr[i]['commit']}  +{d:.1f}\n{summaries.get(okr[i]['commit'], {}).get('short', okr[i]['message'][:28])}",
                xy=(i, y[i]), xytext=(dx, dy), textcoords="offset points",
                ha=ha, va="bottom", fontsize=8.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8, shrinkA=0, shrinkB=4))
# the fix the fixed benchmark can't see
fr = os.path.join(RAW, "F_off_7d5c517.json"), os.path.join(RAW, "F_off_06fd618.json")
if all(os.path.exists(p) for p in fr):
    a_, b_ = (json.load(open(p))["median_tok_s"] for p in fr)
    i6 = [r["commit"] for r in okr].index("06fd618")
    ax.annotate(f"06fd618 autotune-key fix: in a real generation\n(new positions every step) {a_:.1f} → {b_:.1f} tok/s.\nHidden here because positions are replayed.",
                xy=(i6, y[i6]), xytext=(i6 + 0.8, 38.5), textcoords="data", ha="left", va="center", fontsize=8.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8, shrinkB=4))
ax.annotate(f"{y[-1]:.1f}", xy=(x[-1], y[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
ax.annotate(f"{y[0]:.1f}", xy=(0, y[0]), xytext=(-6, 0), textcoords="offset points", va="center", ha="right", fontsize=9, color=INK)

ax.set_xticks(x)
ax.set_xticklabels([r["commit"] for r in okr], rotation=60, ha="right", fontsize=8, family="DejaVu Sans Mono")
ax.set_ylabel("decode tok/s (median of 5 runs × 128 steps)")
ax.set_xlim(-0.8, len(x) - 0.2)
ymin = min(v for v in lo + [w for w in yh if w] if v) - 4
ax.set_ylim(max(0, ymin), max(hi + [h.get("median_tok_s", 0) for h in hf.values() if h.get("status") == "ok"]) + 6)
ax.set_title("Llama-3 8B decode throughput by commit  ·  batch 1, fp16, ctx 512→640, RTX PRO 4500 Blackwell",
             loc="left", color=INK, fontsize=12, pad=12)
ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "throughput_history.png"), facecolor=SURF)
print("wrote", OUT)
