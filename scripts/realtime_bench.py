#!/usr/bin/env python
"""Task 2 — realtime serving benchmark: latency/throughput trade-off of the dynamic batcher.

For every (window, token budget) configuration it sweeps the offered request rate (open-loop Poisson
arrivals) from low load to beyond capacity and records latency percentiles, achieved throughput,
goodput under SLOs, queue wait vs. service time and batch sizes. It also measures the unloaded latency
floor for short / median / long prefixes. Results: results/realtime_<gpu>_<time>/

Examples
    python scripts/realtime_bench.py --data data/bench_2000.npz
    python scripts/realtime_bench.py --data data/bench_2000.npz --windows 0,2,5,10 --max-tokens 8192,16384 \
        --rate-fracs 0.1,0.25,0.5,0.7,0.85,1.0,1.15 --duration 20 --cuda-graphs
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from suffix_scoring.data import SuffixScoringDataset  # noqa: E402
from suffix_scoring.realtime import (Request, Server, ServerConfig, run_point,  # noqa: E402
                                     single_request_latency, tree_tokens)
from suffix_scoring.runner import build, parse_options  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark import env_info  # noqa: E402


def csv_list(s, cast=float):
    return [cast(x) for x in s.split(",") if x]


def estimate_capacity(server: Server, samples, seconds: float = 5.0) -> float:
    """Offline-style throughput with full batches at the server's budget (requests / s)."""
    reqs, i = [], 0
    t_end = time.perf_counter() + seconds
    n = 0
    t0 = time.perf_counter()
    while time.perf_counter() < t_end:
        batch, tok = [], 0
        while True:
            s = samples[i % len(samples)]; t = tree_tokens(s)
            if tok + t > server.cfg.max_tokens:
                break
            batch.append(Request(i, s, t)); tok += t; i += 1
        server.run_batch(batch)
        n += len(batch)
    return n / (time.perf_counter() - t0)


def length_floor(server, ds, targets=(256, 1024, 2048), repeats=20):
    lens = ds.lengths()
    out = {}
    for t in targets:
        idx = np.argsort(np.abs(lens - t))[:repeats]
        out[str(t)] = round(single_request_latency(server, [ds[int(i)] for i in idx], repeats), 2)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--scorer", default="flex")
    ap.add_argument("-o", "--opt", action="append", help="scorer option key=value")
    ap.add_argument("--windows", default="0,2,5,10", help="batching windows in ms")
    ap.add_argument("--max-tokens", default="16384", help="token budgets per batch (comma list)")
    ap.add_argument("--buckets", default="640,1024,1536,2048,2560,4096,8192,16384", help="padded pack lengths (multiples of 128)")
    ap.add_argument("--rate-fracs", default="0.1,0.25,0.5,0.7,0.85,1.0,1.15",
                    help="offered rates as fractions of the measured capacity")
    ap.add_argument("--rates", default="", help="absolute rates (req/s) instead of --rate-fracs")
    ap.add_argument("--capacity", type=float, default=0, help="skip the capacity measurement (req/s)")
    ap.add_argument("--duration", type=float, default=15.0, help="seconds of arrivals per point")
    ap.add_argument("--min-requests", type=int, default=200)
    ap.add_argument("--max-requests", type=int, default=20000)
    ap.add_argument("--cuda-graphs", action="store_true", help="torch.compile mode='reduce-overhead'")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--tiny", action="store_true")
    a = ap.parse_args()

    env = env_info()
    device = "cpu" if a.tiny or not torch.cuda.is_available() else "cuda"
    tag = env["gpu"].replace(" ", "_").replace("NVIDIA_", "")
    out = a.out_dir or os.path.join("results", f"realtime_{tag}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out, exist_ok=True)

    ds = SuffixScoringDataset(a.data)
    samples = [ds[i] for i in range(len(ds))]
    opts = parse_options(a.opt)
    if a.cuda_graphs:
        opts["compile_mode"] = "reduce-overhead"
    scorer, _ = build(a.scorer, opts, ds.start_token_id, dtype=a.dtype, device=device, tiny=a.tiny)
    buckets = tuple(int(b) for b in csv_list(a.buckets, int))
    env.update(scorer=a.scorer, options=opts, buckets=buckets, cuda_graphs=a.cuda_graphs)
    json.dump(env, open(os.path.join(out, "env.json"), "w"), indent=2, default=str)
    print(f"[env] {env}")

    points = []
    for budget in csv_list(a.max_tokens, int):
        server = Server(scorer, ds.start_token_id, ServerConfig(max_tokens=budget, buckets=buckets))
        print(f"\n[budget {budget}] buckets {server.cfg.buckets}: warm-up / compile ...")
        t0 = time.perf_counter()
        server.warmup(samples)
        print(f"  warm-up {time.perf_counter() - t0:.1f} s")
        floors = length_floor(server, ds)
        cap = a.capacity or estimate_capacity(server, samples)
        print(f"  unloaded latency (ms) by prefix length: {floors} | capacity ≈ {cap:.1f} req/s")
        json.dump({"budget": budget, "floor_ms": floors, "capacity_rps": cap},
                  open(os.path.join(out, f"floor_budget{budget}.json"), "w"), indent=2)
        rates = csv_list(a.rates) if a.rates else [f * cap for f in csv_list(a.rate_fracs)]
        for w in csv_list(a.windows):
            server.cfg.window_ms = w
            for ri, rate in enumerate(rates):
                n = int(np.clip(rate * a.duration, a.min_requests, a.max_requests))
                res = run_point(server, samples, rate, n, seed=a.seed + 1000 * ri)   # own draw per rate
                res.extra = {"capacity_rps": round(cap, 2), "rate_frac": round(rate / cap, 3)}
                s = res.summary()
                points.append(s)
                np.savez(os.path.join(out, f"lat_b{budget}_w{w:g}_r{rate:.1f}.npz"),
                         lat=res.lat_ms, wait=res.wait_ms, service=res.service_ms, batch=res.batch_sizes)
                print(f"  W={w:>4g} ms  offered {s['offered_rps']:7.1f}/s ({s['offered_rps'] / cap:4.0%})  achieved {s['achieved_rps']:7.1f}/s  "
                      f"p50 {s['lat_p50_ms']:8.1f}  p99 {s['lat_p99_ms']:8.1f} ms  batch {s['batch_mean']:5.1f}")
                _write(out, points, env)
    print(f"\n[done] {out}\n" + open(os.path.join(out, "summary.md")).read())


def _write(out, points, env):
    keys = list(points[0].keys())
    with open(os.path.join(out, "points.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(points)
    json.dump(points, open(os.path.join(out, "points.json"), "w"), indent=2)
    cols = ["max_tokens", "window_ms", "rate", "rate_frac", "offered_rps", "achieved_rps", "lat_p50_ms", "lat_p95_ms",
            "lat_p99_ms", "wait_p50_ms", "service_p50_ms", "batch_mean", "goodput_rps_slo100ms"]
    lines = [f"# Realtime serving on {env['gpu']}  ({env.get('scorer')}, CUDA graphs: {env.get('cuda_graphs')})", "",
             "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for p in points:
        lines.append("| " + " | ".join(str(round(p[c], 2)) if isinstance(p[c], float) else str(p[c]) for c in cols) + " |")
    open(os.path.join(out, "summary.md"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
