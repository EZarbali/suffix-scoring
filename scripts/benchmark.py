#!/usr/bin/env python
"""One-command benchmark sweep for the suffix-scoring scorers (H200, H100, A100, ... and CPU tiny tests).

For every configuration in the grid it (re)loads the model, warms up / compiles, runs the dataset
`--repeats` times, and records throughput, MFU, peak memory, padding fill rate and (optionally)
correctness against an fp32 gold reference. Results go to results/bench_<gpu>_<time>/:
    results.csv, results.json, summary.md, env.json

Examples
    # default sweep on the best scorer
    python scripts/benchmark.py --data data/bench_2000.npz

    # explicit grid, including FP8 variants and last-layer pruning, plus correctness
    python scripts/benchmark.py --data data/bench_2000.npz \
        --scorers flex --pack-len 8192,16384,32768 --batch-size 256 --num-workers 8 \
        --fp8 none,mlp_gate_up,mlp --prune false,true \
        --eval-data data/eval_50.npz --gold results/ref_fp32.npz --ref results/ref_bf16.npz

    # ablation over all steps (naive is slow -> it uses --naive-limit sequences)
    python scripts/benchmark.py --data data/bench_2000.npz \
        --scorers naive,kvcache,tree,tree_batched,tree_pipelined,flex
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import platform
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from suffix_scoring import metrics  # noqa: E402
from suffix_scoring.data import SuffixScoringDataset  # noqa: E402
from suffix_scoring.runner import build, parse_options, run  # noqa: E402
from suffix_scoring.tree import KEY  # noqa: E402

# default per-scorer options (merged under the grid values)
DEFAULTS = {
    "naive": {"candidates_per_forward": 100},
    "kvcache": {"candidates_per_forward": 64},
    "tree": {},
    "tree_batched": {"compile": True},
    "tree_pipelined": {},
    "flex": {},
}
BUDGET_KEY = {"flex": "pack_len", "tree_batched": "max_tokens", "tree_pipelined": "max_tokens"}
SUPPORTS = {  # which grid dimensions apply to which scorer
    "fp8": {"flex"}, "prune": {"flex"}, "budget": {"flex", "tree_batched", "tree_pipelined"},
    "workers": {"flex", "tree_pipelined"},
}


def csv_list(s: str, cast=str):
    return [cast(x) for x in s.split(",") if x != ""]


def as_bool(x: str) -> bool:
    return str(x).lower() in ("1", "true", "yes")


def env_info() -> dict:
    info = {"python": platform.python_version(), "torch": torch.__version__, "cpu_count": os.cpu_count(),
            "platform": platform.platform()}
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except Exception:
        pass
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(gpu=p.name, gpu_mem_gb=round(p.total_memory / 2**30, 1),
                    capability=".".join(map(str, torch.cuda.get_device_capability())),
                    cuda=torch.version.cuda, fp8_tensor_cores=torch.cuda.get_device_capability() >= (9, 0))
    else:
        info.update(gpu="cpu", fp8_tensor_cores=False)
    return info


def dataset_flops(ds: SuffixScoringDataset, cfg) -> float:
    """Average useful FLOPs per sequence of this dataset (for MFU)."""
    tot = 0.0
    for i in range(len(ds)):
        s = ds[i]
        L = max(ds.length(i), 1)
        c = s.cand_tokens
        n1 = int(torch.unique(c[:, 0]).numel())
        n2 = int(torch.unique(c[:, 0] * KEY + c[:, 1]).numel())
        tot += metrics.useful_flops(L, n1, n2, cfg)
    return tot / len(ds)


def build_grid(a, gpu_has_fp8: bool) -> list[dict]:
    configs = []
    for scorer in csv_list(a.scorers):
        dims = {
            "budget": csv_list(a.pack_len, int) if scorer in SUPPORTS["budget"] else [None],
            "batch_size": csv_list(a.batch_size, int) if scorer not in ("naive", "kvcache", "tree") else [1],
            "workers": csv_list(a.num_workers, int) if scorer in SUPPORTS["workers"] else [0],
            "fp8": csv_list(a.fp8) if scorer in SUPPORTS["fp8"] else ["none"],
            "prune": [as_bool(x) for x in csv_list(a.prune)] if scorer in SUPPORTS["prune"] else [False],
        }
        for budget, bs, nw, fp8, prune in itertools.product(*dims.values()):
            if fp8 != "none" and not gpu_has_fp8 and not a.allow_fake_fp8:
                print(f"[skip] {scorer} fp8={fp8}: no FP8 tensor cores on this GPU "
                      f"(use --allow-fake-fp8 to measure accuracy with fake quantisation)")
                continue
            opts = dict(DEFAULTS.get(scorer, {}))
            opts.update(parse_options(a.opt))
            if budget is not None:
                opts[BUDGET_KEY[scorer]] = budget
            if scorer == "flex":
                opts["fp8"], opts["prune_last_layer"] = fp8, prune
            configs.append({"scorer": scorer, "batch_size": bs, "num_workers": nw, "options": opts})
    return configs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="benchmark dataset (.npz), e.g. 2000 sequences")
    ap.add_argument("--scorers", default="flex")
    ap.add_argument("--pack-len", default="16384", help="flex: pack_len / tree_*: max_tokens (comma list)")
    ap.add_argument("--batch-size", default="256", help="sequences per scorer call (comma list)")
    ap.add_argument("--num-workers", default="auto", help="DataLoader workers (comma list or 'auto')")
    ap.add_argument("--fp8", default="none", help="none,mlp_gate_up,mlp,all (flex only)")
    ap.add_argument("--prune", default="false", help="last-layer pruning: false,true (flex only)")
    ap.add_argument("-o", "--opt", action="append", help="extra scorer option key=value for all configs")
    ap.add_argument("--limit", type=int, default=0, help="sequences per repeat (0 = all)")
    ap.add_argument("--naive-limit", type=int, default=20, help="sequences for naive/kvcache/tree (slow)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=3, help="warm-up batches (compilation) per config")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--eval-data", default="", help="dataset for correctness (same as used for --gold/--ref)")
    ap.add_argument("--gold", default="", help="fp32 naive results on --eval-data")
    ap.add_argument("--ref", default="", help="bf16 naive results on --eval-data")
    ap.add_argument("--allow-fake-fp8", action="store_true", help="run fp8 configs without FP8 hardware")
    ap.add_argument("--population", type=float, default=2.5e6, help="sequences for the time projection")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--tiny", action="store_true", help="tiny random model on CPU (smoke test)")
    a = ap.parse_args()

    env = env_info()
    if a.num_workers == "auto":
        a.num_workers = str(max(0, min(8, (os.cpu_count() or 2) - 2)))
    device = "cpu" if a.tiny or not torch.cuda.is_available() else "cuda"
    gpu_tag = env["gpu"].replace(" ", "_").replace("NVIDIA_", "")
    out_dir = a.out_dir or os.path.join("results", f"bench_{gpu_tag}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)
    json.dump(env, open(os.path.join(out_dir, "env.json"), "w"), indent=2)
    print(f"[env] {env}")

    ds = SuffixScoringDataset(a.data)
    eval_ds = SuffixScoringDataset(a.eval_data) if a.eval_data else None
    gold = metrics.load_results(a.gold) if a.gold else None
    ref = metrics.load_results(a.ref) if a.ref else None
    peak = metrics.peak_tflops(env["gpu"]) if device == "cuda" else None

    configs = build_grid(a, env.get("fp8_tensor_cores", False))
    print(f"[grid] {len(configs)} configurations -> {out_dir}")
    rows, avg_flops = [], None

    for ci, c in enumerate(configs):
        tag = f"[{ci + 1}/{len(configs)}] {c['scorer']} bs={c['batch_size']} nw={c['num_workers']} {c['options']}"
        print("\n" + "=" * 100 + f"\n{tag}")
        scorer = model = None
        torch._dynamo.reset()
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        row = {"scorer": c["scorer"], "batch_size": c["batch_size"], "num_workers": c["num_workers"],
               "options": json.dumps(c["options"], sort_keys=True)}
        try:
            t0 = time.perf_counter()
            scorer, model = build(c["scorer"], c["options"], ds.start_token_id, dtype=a.dtype,
                                  device=device, tiny=a.tiny)
            if avg_flops is None:
                avg_flops = dataset_flops(ds, model.config)
            slow = c["scorer"] in ("naive", "kvcache", "tree")
            limit = a.naive_limit if slow else a.limit
            # first run includes warm-up/compilation; repeats are timed
            r = run(scorer, ds, limit=limit, batch_size=c["batch_size"], num_workers=c["num_workers"],
                    warmup_batches=1 if slow else a.warmup, keep_scores=False)
            row["warmup_s"] = round(time.perf_counter() - t0 - r.seconds, 1)
            speeds = [r.seq_per_s]
            for _ in range(a.repeats - 1):
                speeds.append(run(scorer, ds, limit=limit, batch_size=c["batch_size"],
                                  num_workers=c["num_workers"], warmup_batches=0, keep_scores=False).seq_per_s)
            sps = float(np.mean(speeds))
            row.update(n_seq=r.n, seq_per_s=round(sps, 2), seq_per_s_std=round(float(np.std(speeds)), 2),
                       ms_per_seq=round(1000 / sps, 3),
                       hours_for_population=round(a.population / sps / 3600, 2),
                       fill_rate=round(r.real_tokens / r.padded_tokens, 3) if r.padded_tokens else None,
                       peak_mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2) if device == "cuda" else None,
                       tflops=round(sps * avg_flops / 1e12, 1),
                       mfu=round(sps * avg_flops / (peak * 1e12), 3) if peak else None,
                       fp8_mode=(getattr(scorer, "fp8_info", None) or {}).get("mode", "none"))
            if eval_ds is not None:
                e = run(scorer, eval_ds, batch_size=min(c["batch_size"], len(eval_ds)),
                        warmup_batches=0, keep_scores=True)
                new = {"idx": np.array(e.idx), "scores": e.scores, "cand_idx": e.cand_idx}
                np.savez(os.path.join(out_dir, f"scores_cfg{ci + 1}.npz"), **new)
                if gold is not None and ref is not None:
                    g = metrics.compare_to_gold(gold, ref, new)
                    nv = g["new_vs_gold"]
                    row.update(correct=g["PASS"], max_abs_vs_gold=round(nv["max_abs_diff"], 4),
                               mean_abs_vs_gold=round(nv["mean_abs_diff"], 4),
                               top1_vs_gold=round(nv["top1_overlap"], 3), top10_vs_gold=round(nv["top10_overlap"], 3),
                               swaps_vs_gold=nv["significant_swaps"],
                               naive_bf16_swaps_vs_gold=g["ref_vs_gold"]["significant_swaps"])
                elif ref is not None:
                    m = metrics.compare(ref, new, atol=float("inf"))
                    row.update(max_abs_vs_ref=round(m["max_abs_diff"], 4), top10_vs_ref=round(m["top10_overlap"], 3))
            print(f"  -> {row['seq_per_s']} seq/s (±{row['seq_per_s_std']}), mfu={row['mfu']}, "
                  f"peak_mem={row['peak_mem_gb']} GB, fill={row['fill_rate']}, correct={row.get('correct')}")
        except Exception as ex:  # keep sweeping (e.g. OOM for a large budget)
            row["error"] = f"{type(ex).__name__}: {str(ex)[:200]}"
            print(f"  !! {row['error']}")
        rows.append(row)
        _write(out_dir, rows, env, a)
        del scorer, model
    print(f"\n[done] results in {out_dir}")
    print(open(os.path.join(out_dir, "summary.md")).read())


def _write(out_dir: str, rows: list[dict], env: dict, a):
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("scorer", "options"), k))
    with open(os.path.join(out_dir, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    json.dump(rows, open(os.path.join(out_dir, "results.json"), "w"), indent=2)
    cols = ["scorer", "options", "batch_size", "num_workers", "seq_per_s", "seq_per_s_std", "ms_per_seq",
            "mfu", "peak_mem_gb", "fill_rate", "hours_for_population", "fp8_mode", "correct",
            "top10_vs_gold", "swaps_vs_gold", "error"]
    cols = [c for c in cols if any(c in r for r in rows)]
    lines = [f"# Benchmark on {env['gpu']}  (torch {env['torch']}, {time.strftime('%Y-%m-%d %H:%M')})", "",
             f"Data: `{a.data}`, repeats: {a.repeats}, dtype: {a.dtype}, population for projection: {a.population:.0f}", "",
             "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in sorted(rows, key=lambda r: -(r.get("seq_per_s") or 0)):
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    open(os.path.join(out_dir, "summary.md"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
