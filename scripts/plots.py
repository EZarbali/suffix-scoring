#!/usr/bin/env python
"""Presentation figures (PNG @2x + SVG) from benchmark results and/or recorded measurements.

    python scripts/plots.py --measurements docs/measurements_a100.json --data data/bench_2000.npz
    python scripts/plots.py --ablation results/ablation/results.json --sweep results/sweep_flex/results.json \
        --data data/bench_2000.npz --out figures/h200

Figures
    01_length_distribution     prefix length histogram (clipping spikes at 0 / 2048)
    02_tokens_per_sequence     work per sequence: naive vs prefix-once vs tree (log scale)
    03_ablation_throughput     seq/s per optimisation step (log scale), x-naive + hours for 2.5 M
    04_gpu_time_breakdown      where GPU time goes: dense mask vs FlexAttention (stacked)
    05_kvcache_time_share      why the KV-cache variant is slow: memory copies
    06_correctness             distance to fp32 gold vs naive bf16 (two small panels, one axis each)
    07_sweep_pack_len          throughput vs tokens per forward, per FP8 / pruning variant
    08_mfu                     MFU per configuration of the sweep
    09_realtime_latency        Task 2: p50 / p99 latency vs achieved throughput, one curve per batching window
    10_realtime_batch_size     Task 2: mean batch size vs offered load (the batching mechanism)
    11_realtime_floor          Task 2: unloaded latency by prefix length
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

# --------------------------------------------------------------- style (validated categorical palette)
SURFACE, INK, INK_2, INK_3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GAP = 2  # px surface gap between adjacent fills

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.titlelocation": "left", "axes.titlepad": 28, "axes.axisbelow": True,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}")


def _subtitle(ax, text):
    ax.text(0, 1.025, text, transform=ax.transAxes, color=INK_2, fontsize=10, va="bottom")


def _fmt_num(x):
    return f"{x:,.0f}" if x >= 100 else (f"{x:.1f}" if x >= 1 else f"{x:.2f}")


# --------------------------------------------------------------- figures
def fig_length_distribution(data_path, out):
    f = np.load(data_path)
    lens = np.diff(f["offsets"])
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.hist(lens, bins=np.arange(0, 2049 + 32, 32), color=SERIES[0], edgecolor=SURFACE, linewidth=0.6)
    ax.set_xlabel("prefix length (tokens)")
    ax.set_ylabel("sequences")
    ax.grid(axis="x", visible=False)
    p0, p1 = np.mean(lens == 0), np.mean(lens == 2048)
    ymax = ax.get_ylim()[1]
    ax.annotate(f"length 0: {p0:.1%}\n(→ <|endoftext|> context)", xy=(70, ymax * 0.55), color=INK_2, fontsize=9)
    ax.annotate(f"length 2048: {p1:.1%}\n(attention-heavy)", xy=(1640, ymax * 0.55), color=INK_2, fontsize=9)
    ax.set_title("Prefix length distribution")
    _subtitle(ax, f"round(Normal(1024, 600)) clipped to [0, 2048] · n = {len(lens):,} · mean {lens.mean():.0f}")
    _save(fig, out, "01_length_distribution")


def fig_tokens(meas, out):
    d = meas["tokens_per_sequence"]
    labels, vals = list(d), list(d.values())
    fig, ax = plt.subplots(figsize=(9, 3.4))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, vals, color=SERIES[0], height=0.55)
    ax.set_xscale("log")
    ax.set_yticks(y, labels)
    ax.grid(axis="y", visible=False)
    for yi, v in zip(y, vals):
        ax.text(v * 1.08, yi, f"{v:,} tokens", va="center", color=INK, fontsize=10)
    ax.set_xlim(left=100, right=max(vals) * 8)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.set_title("Work per sequence: tokens through the model")
    _subtitle(ax, f"~{vals[0] / vals[-1]:.0f}× less work by exploiting the suffix structure (log scale)")
    _save(fig, out, "02_tokens_per_sequence")


def fig_ablation(steps, out, gpu, population=2.5e6):
    labels = [s["step"] for s in steps]
    v = np.array([s["seq_per_s"] for s in steps])
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(v) + 1.6))
    y = np.arange(len(v))[::-1]
    colors = [SERIES[0]] * (len(v) - 1) + [SERIES[1]]          # highlight the final result
    ax.barh(y, v, color=colors, height=0.6)
    ax.set_xscale("log")
    ax.set_yticks(y, labels)
    ax.grid(axis="y", visible=False)
    for yi, vi in zip(y, v):
        hours = population / vi / 3600
        when = f"{hours / 24:.0f} d" if hours > 48 else f"{hours:.1f} h"
        ax.text(vi * 1.1, yi, f"{_fmt_num(vi)} seq/s · {vi / v[0]:.0f}× · {when}", va="center", fontsize=9.5, color=INK)
    ax.set_xlim(left=v.min() * 0.5, right=v.max() * 40)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: _fmt_num(x)))
    ax.set_xlabel("throughput (sequences / s, log scale)")
    ax.set_title("Offline throughput per optimisation step")
    _subtitle(ax, f"{gpu} · bf16 · labels: seq/s · speed-up vs naive · time for {population / 1e6:.1f} M sequences")
    _save(fig, out, "03_ablation_throughput")


def fig_breakdown(meas, out):
    d = {k: v for k, v in meas["gpu_time_breakdown_ms"].items() if not k.startswith("_")}
    cats = ["attention", "GEMM", "logsumexp", "other"]
    names = list(d)
    fig, ax = plt.subplots(figsize=(10, 2.9))
    y = np.arange(len(names))[::-1]
    left = np.zeros(len(names))
    for ci, c in enumerate(cats):
        vals = np.array([d[n][c] for n in names])
        ax.barh(y, vals, left=left, color=SERIES[ci], height=0.55, label=c, edgecolor=SURFACE, linewidth=GAP)
        for yi, l, v in zip(y, left, vals):
            if v > 35:
                ax.text(l + v / 2, yi, f"{v:.0f}", ha="center", va="center", fontsize=9, color=INK)
        left += vals
    for yi, tot in zip(y, left):
        ax.text(tot + 6, yi, f"{tot:.0f} ms", va="center", fontsize=10, color=INK, fontweight="bold")
    ax.set_yticks(y, names)
    ax.set_xlim(right=left.max() * 1.15)
    ax.set_xlabel("GPU time for 32 sequences (ms)")
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=4, loc="upper left", bbox_to_anchor=(0, -0.32), fontsize=9.5)
    att = [d[n]["attention"] for n in names]
    ax.set_title("Where GPU time goes: dense mask vs FlexAttention")
    _subtitle(ax, f"block-sparse mask skips empty tiles → attention {att[0] / att[-1]:.1f}× faster; GEMMs now dominate")
    _save(fig, out, "04_gpu_time_breakdown")


def fig_kvcache(meas, out):
    d = {k: v for k, v in meas["kvcache_gpu_time_share"].items() if not k.startswith("_")}
    fig, ax = plt.subplots(figsize=(10, 1.9))
    left = 0
    for i, (k, v) in enumerate(d.items()):
        ax.barh([0], [v], left=left, color=SERIES[[1, 0, 2][i % 3]], height=0.5, edgecolor=SURFACE, linewidth=GAP)
        if v > 0.1:
            ax.text(left + v / 2, 0, f"{k}\n{v:.0%}", ha="center", va="center", fontsize=9.5, color=INK)
        else:                                         # small segment: label outside, to the right
            ax.text(left + v + 0.008, 0, f"{k}\n{v:.0%}", ha="left", va="center", fontsize=9.5, color=INK)
        left += v
    ax.set_xlim(0, 1.12)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.grid(False)
    ax.set_title("KV-cache variant: the GPU mostly moves memory")
    _subtitle(ax, "share of GPU time (approx., torch.profiler)")
    _save(fig, out, "05_kvcache_time_share")


def fig_correctness(rows, out):
    names = list(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.5 * len(names) + 2.0), sharey=True)
    y = np.arange(len(names))[::-1]
    for ax, key, title in ((axes[0], "max_abs", "max |score − gold|"), (axes[1], "swaps", "significant rank swaps vs gold")):
        vals = [rows[n][key] for n in names]
        ref = rows[names[0]][key]
        ax.axvline(ref, color=INK_3, linewidth=1, linestyle=(0, (4, 3)))
        ax.scatter(vals, y, s=70, color=[SERIES[1] if i == 0 else SERIES[0] for i in range(len(names))],
                   edgecolor=SURFACE, linewidth=2, zorder=3)
        for yi, v in zip(y, vals):
            ax.text(v, yi + 0.28, f"{v:,.3g}" if key == "max_abs" else f"{v:,}", ha="center", fontsize=9, color=INK)
        ax.set_title(title, fontsize=11.5)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(left=0, right=max(vals) * 1.25)
    axes[0].set_yticks(y, names)
    fig.suptitle("Correctness: distance to fp32 gold (lower is better)", x=0.01, ha="left", fontweight="bold", fontsize=14)
    fig.text(0.01, 0.9, "dashed line = naive bf16 reference (the noise floor every bf16 method lives with)",
             color=INK_2, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    _save(fig, out, "06_correctness")


def _variant(r):
    o = json.loads(r["options"])
    v = f"fp8={o.get('fp8', 'none')}"
    if o.get("prune_last_layer"):
        v += " + prune"
    return v


def fig_sweep(rows, out, gpu):
    rows = [r for r in rows if r.get("scorer") == "flex" and r.get("seq_per_s")]
    if not rows:
        return
    variants = sorted({_variant(r) for r in rows}, key=lambda s: (s.split()[0] != "fp8=none", s))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, var in enumerate(variants[:8]):
        pts = sorted((json.loads(r["options"])["pack_len"], r["seq_per_s"], r.get("seq_per_s_std", 0))
                     for r in rows if _variant(r) == var)
        x, yv, e = map(np.array, zip(*pts))
        ax.errorbar(x, yv, yerr=e, color=SERIES[i], linewidth=2, marker="o", markersize=8,
                    markeredgecolor=SURFACE, markeredgewidth=2, capsize=3, label=var)
        if len(variants) <= 4:                       # direct labels only while they cannot collide
            ax.text(x[-1] * 1.04, yv[-1], f"{var}  {yv[-1]:.0f}", va="center", fontsize=9, color=INK)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted({json.loads(r['options'])['pack_len'] for r in rows}))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
    if len(variants) <= 4:
        ax.set_xlim(right=ax.get_xlim()[1] * 1.9)
    lo = min(r["seq_per_s"] for r in rows)
    ax.set_ylim(bottom=max(0, lo * 0.85))
    ax.set_xlabel("tokens per forward (pack_len)")
    ax.set_ylabel("sequences / s")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    ax.set_title("Throughput vs tokens per forward")
    _subtitle(ax, f"{gpu} · FlexTreeScorer · mean ± std over repeats")
    _save(fig, out, "07_sweep_pack_len")


def fig_mfu(rows, out, gpu):
    rows = [r for r in rows if r.get("mfu")]
    if not rows:
        return
    rows = sorted(rows, key=lambda r: r["mfu"])
    labels = [f"{r['scorer']} · {_variant(r)} · {json.loads(r['options']).get('pack_len', '')}" for r in rows]
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(rows) + 1.6))
    y = np.arange(len(rows))
    failed = [r.get("correct") is False for r in rows]
    ax.barh(y, [r["mfu"] for r in rows], color=[INK_3 if f else SERIES[0] for f in failed], height=0.6)
    for yi, r, f in zip(y, rows, failed):
        ax.text(r["mfu"] + 0.005, yi, f"{r['mfu']:.0%} · {r['seq_per_s']:.0f} seq/s" + ("  · fails accuracy gate" if f else ""),
                va="center", fontsize=9)
    ax.set_yticks(y, labels, fontsize=9)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_xlim(0, max(r["mfu"] for r in rows) * 1.35)
    ax.grid(axis="y", visible=False)
    ax.set_title("Model FLOPs utilisation")
    _subtitle(ax, f"{gpu} · useful FLOPs (tree method) / dense bf16 peak")
    _save(fig, out, "08_mfu")


def _rt_series(points, budget):
    pts = [p for p in points if p["max_tokens"] == budget]
    wins = sorted({p["window_ms"] for p in pts})
    return pts, wins


def _rt_saturated(p):
    """A point at or beyond capacity never reaches a steady state: its latency grows with run length."""
    return p.get("rate_frac", 0) >= 0.99 or (p.get("offered_rps") and p["achieved_rps"] < 0.97 * p["offered_rps"])


def fig_realtime(points, out, gpu, floors, slo_ms=100):
    budgets = sorted({p["max_tokens"] for p in points})
    wins = sorted({p["window_ms"] for p in points})
    combos = [(b, w) for b in budgets for w in wins if any(p["max_tokens"] == b and p["window_ms"] == w for p in points)]
    many_w = len(wins) > 1
    label = lambda b, w: (f"budget {b // 1024}k" + (f" · W {w:g} ms" if many_w else ""))
    combos = combos[:8]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
    for ax, key, title in ((axes[0], "lat_p50_ms", "p50 latency"), (axes[1], "lat_p99_ms", "p99 latency")):
        for i, (b, w) in enumerate(combos):
            ser = sorted((p["achieved_rps"], p[key], _rt_saturated(p)) for p in points
                         if p["max_tokens"] == b and p["window_ms"] == w)
            x = [q[0] for q in ser]; y = [q[1] for q in ser]; sat = [q[2] for q in ser]
            c = SERIES[i]
            ax.plot(x, y, color=c, linewidth=2, zorder=3, label=label(b, w))
            ax.scatter([xi for xi, si in zip(x, sat) if not si], [yi for yi, si in zip(y, sat) if not si],
                       s=48, color=c, edgecolor=SURFACE, linewidth=1.5, zorder=4)
            ax.scatter([xi for xi, si in zip(x, sat) if si], [yi for yi, si in zip(y, sat) if si],
                       s=48, facecolor=SURFACE, edgecolor=c, linewidth=1.8, zorder=4)
        ax.axhline(slo_ms, color=INK_3, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
        ax.text(ax.get_xlim()[0], slo_ms * 1.06, f"  SLO {slo_ms} ms", color=INK_2, fontsize=9, va="bottom")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("achieved throughput (requests / s)")
        ax.set_title(title, fontsize=12)
    axes[0].set_ylabel("latency (ms, log scale)")
    axes[0].legend(fontsize=9, loc="upper left", frameon=False)
    fig.suptitle("Realtime serving: latency vs throughput", x=0.01, ha="left", fontweight="bold", fontsize=14)
    fig.text(0.01, 0.915, f"{gpu} · open-loop Poisson arrivals · hollow markers: at/over capacity, "
             f"queue never settles (not steady state)", color=INK_2, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    _save(fig, out, "09_realtime_latency")

    fig, ax = plt.subplots(figsize=(9, 4.6))
    for i, (b, w) in enumerate(combos):
        ser = sorted((p.get("rate_frac", p["rate"]), p["batch_mean"]) for p in points
                     if p["max_tokens"] == b and p["window_ms"] == w)
        x, y = zip(*ser)
        ax.plot(x, y, color=SERIES[i], linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE,
                markeredgewidth=2, label=label(b, w))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("offered load (fraction of each configuration's capacity)")
    ax.set_ylabel("requests per forward (mean)")
    ax.legend(fontsize=9, loc="upper left", frameon=False)
    ax.set_title("Batch size grows with load")
    _subtitle(ax, "more load → bigger batches → better GPU efficiency; the token budget caps the batch size")
    _save(fig, out, "10_realtime_batch_size")

    if floors:
        f = floors[-1]["floor_ms"]
        ks = list(f)
        fig, ax = plt.subplots(figsize=(7.5, 3.4))
        y = np.arange(len(ks))[::-1]
        ax.barh(y, [f[k] for k in ks], color=SERIES[0], height=0.55)
        for yi, k in zip(y, ks):
            ax.text(f[k] * 1.02, yi, f"{f[k]:.1f} ms", va="center", fontsize=10)
        ax.set_yticks(y, [f"prefix ≈ {k} tokens" for k in ks])
        ax.set_xlim(right=max(f.values()) * 1.25)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("latency of a single request on an idle server (ms)")
        ax.set_title("Unloaded latency floor")
        _subtitle(ax, f"{gpu} · one request at a time, median of 20")
        _save(fig, out, "11_realtime_floor")


ABLATION_LABELS = {
    "naive": "naive HF reference", "kvcache": "prefix once + KV cache", "tree": "prefix tree, 1 seq / forward",
    "tree_batched": "+ batching + compile (dense mask)", "tree_pipelined": "+ CPU prep in workers",
    "flex": "+ packing + FlexAttention + fused head",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--measurements", default="", help="JSON with recorded numbers (docs/measurements_a100.json)")
    ap.add_argument("--ablation", default="", help="results.json of a benchmark run over all scorers")
    ap.add_argument("--sweep", default="", help="results.json of a flex sweep")
    ap.add_argument("--data", default="", help="dataset .npz for the length histogram")
    ap.add_argument("--realtime", default="", help="results dir of scripts/realtime_bench.py")
    ap.add_argument("--gpu", default="", help="label (default: from results/env.json or measurements)")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()

    meas = json.load(open(a.measurements)) if a.measurements else None
    gpu = a.gpu or (meas or {}).get("gpu", "")
    for path in (a.ablation, a.sweep):
        env = os.path.join(os.path.dirname(path), "env.json") if path else ""
        if not a.gpu and env and os.path.exists(env):
            gpu = json.load(open(env)).get("gpu", gpu)

    print(f"writing figures to {a.out}/")
    if a.data:
        fig_length_distribution(a.data, a.out)
    if meas:
        fig_tokens(meas, a.out)
        fig_breakdown(meas, a.out)
        fig_kvcache(meas, a.out)
        corr = {k: v for k, v in meas["correctness_vs_fp32_gold"].items() if not k.startswith("_")}
        fig_correctness(corr, a.out)
    if a.ablation:
        rows = [r for r in json.load(open(a.ablation)) if r.get("seq_per_s")]
        order = list(ABLATION_LABELS)
        best = {}
        for r in rows:                                    # fastest config per scorer
            if r["scorer"] not in best or r["seq_per_s"] > best[r["scorer"]]["seq_per_s"]:
                best[r["scorer"]] = r
        steps = [{"step": ABLATION_LABELS[s], "seq_per_s": best[s]["seq_per_s"]} for s in order if s in best]
        fig_ablation(steps, a.out, gpu)
        corr = {ABLATION_LABELS[s]: {"max_abs": best[s]["max_abs_vs_gold"], "swaps": best[s]["swaps_vs_gold"]}
                for s in order if s in best and "max_abs_vs_gold" in best[s]}
        if corr:
            fig_correctness(corr, a.out)
    elif meas:
        fig_ablation(meas["ablation"], a.out, gpu)
    if a.sweep:
        rows = json.load(open(a.sweep))
        # runs whose repeats disagree by > 15 % were disturbed (e.g. recompilation mid-run): not plotted
        bad = [r for r in rows if r.get("seq_per_s") and (r.get("seq_per_s_std") or 0) > 0.15 * r["seq_per_s"]]
        if bad:
            print(f"  dropping {len(bad)} unstable sweep rows (std > 15 % of mean) from the figures")
        rows = [r for r in rows if r not in bad]
        fig_sweep(rows, a.out, gpu)
        fig_mfu(rows, a.out, gpu)
    if a.realtime:
        import glob
        pts = json.load(open(os.path.join(a.realtime, "points.json")))
        floors = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(a.realtime, "floor_budget*.json")))]
        envp = os.path.join(a.realtime, "env.json")
        g = a.gpu or (json.load(open(envp)).get("gpu", gpu) if os.path.exists(envp) else gpu)
        fig_realtime(pts, a.out, g, floors)
    if not (a.data or meas or a.ablation or a.sweep or a.realtime):
        sys.exit("nothing to plot: pass --measurements, --ablation, --sweep and/or --data")


if __name__ == "__main__":
    main()
