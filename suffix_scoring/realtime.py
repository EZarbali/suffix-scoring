"""Realtime serving simulation (Task 2): open-loop load generator + dynamic batcher + bucketed forwards.

    generator thread ──(Poisson arrivals, rate λ)──► queue ──► batcher ──► prepare (tree, packing)
                                                                  │
                     window W after the first request of a batch  │  or token budget reached
                                                                  ▼
                                                        scorer.score(packed batch)  ──► latencies

Latency of a request = completion time − SCHEDULED arrival time (open loop: arrivals never wait for the
server, so a saturated server shows up as growing latency instead of being hidden: no coordinated omission).
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .data import Sample
from .tree import KEY, context_tokens, prepare_packed


def tree_tokens(s: Sample) -> int:
    """Tokens of the tree sequence of one request (prefix + unique t1 + unique (t1, t2))."""
    c = s.cand_tokens
    n1 = int(torch.unique(c[:, 0]).numel())
    n2 = int(torch.unique(c[:, 0] * KEY + c[:, 1]).numel())
    return max(s.prefix.numel(), 1) + n1 + n2


@dataclass
class Request:
    rid: int
    sample: Sample
    tokens: int
    t_arrival: float = 0.0       # scheduled arrival (perf_counter seconds)
    t_start: float = 0.0         # batch started (left the queue)
    t_done: float = 0.0
    batch: int = -1


@dataclass
class ServerConfig:
    window_ms: float = 0.0                      # max wait after the first request of a batch (0 = best measured)
    max_tokens: int = 16384                     # token budget per batch (= largest bucket)
    buckets: tuple = (640, 1024, 1536, 2048, 2560, 4096, 8192, 16384)  # padded pack lengths (static shapes, compiled once each)


def pick_bucket(total_tokens: int, buckets) -> int:
    for b in buckets:
        if total_tokens <= b:
            return b
    return buckets[-1]


class Server:
    def __init__(self, scorer, start_token_id: int, cfg: ServerConfig):
        self.scorer, self.start, self.cfg = scorer, start_token_id, cfg
        self.cfg.buckets = tuple(sorted(b for b in cfg.buckets if b <= cfg.max_tokens)) or (cfg.max_tokens,)

    def run_batch(self, reqs: list[Request]) -> torch.Tensor:
        total = sum(r.tokens for r in reqs)
        pack_len = pick_bucket(total, self.cfg.buckets)
        batch = prepare_packed([r.sample for r in reqs], self.start, pack_len)
        scores = self.scorer.score(batch)            # returns CPU tensor -> implicit synchronisation
        return scores

    def warmup(self, samples: list[Sample]):
        """Compile every bucket before measuring (and record CUDA graphs if enabled).

        Per bucket: one batch filled with the smallest samples, plus the largest single request that
        fits, so both shapes of the head rows are seen. Samples are sorted by size and never cut off
        early: a bucket smaller than the first sample would otherwise be skipped and compiled in the
        middle of a measurement (seen on A100: a 30 s stall in the first load point)."""
        sized = sorted(((tree_tokens(s), s) for s in samples), key=lambda x: x[0])
        for b in self.cfg.buckets:
            reqs, tot = [], 0
            for t, s in sized:
                if tot + t > b:
                    break                                   # sorted ascending: nothing later fits
                reqs.append(Request(-1, s, t)); tot += t
            fitting = [(t, s) for t, s in sized if t <= b]
            single = [Request(-1, fitting[-1][1], fitting[-1][0])] if fitting else []
            for batch in (reqs, single):
                if batch:
                    for _ in range(2):
                        self.run_batch(batch)

    def serve(self, q: "queue.Queue[Request | None]", n_expected: int) -> list[Request]:
        done, bid, pending = [], 0, None
        W = self.cfg.window_ms / 1000.0
        while len(done) < n_expected:
            first = pending or q.get()
            pending = None
            if first is None:
                break
            batch, tokens = [first], first.tokens
            deadline = time.perf_counter() + W
            while tokens < self.cfg.max_tokens:
                remaining = deadline - time.perf_counter()
                try:
                    r = q.get(timeout=max(remaining, 0)) if remaining > 0 else q.get_nowait()
                except queue.Empty:
                    break
                if r is None:
                    break
                if tokens + r.tokens > self.cfg.max_tokens:
                    pending = r                      # does not fit: first request of the next batch
                    break
                batch.append(r); tokens += r.tokens
            t0 = time.perf_counter()
            for r in batch:
                r.t_start, r.batch = t0, bid
            self.run_batch(batch)
            t1 = time.perf_counter()
            for r in batch:
                r.t_done = t1
            done += batch
            bid += 1
        return done


def poisson_schedule(n: int, rate: float, seed: int = 0) -> np.ndarray:
    gaps = np.random.default_rng(seed).exponential(1.0 / rate, size=n)
    return np.cumsum(gaps)


def generate(q: queue.Queue, reqs: list[Request], offsets: np.ndarray, t0: float):
    """Open-loop generator: enqueue each request at its scheduled time, never waiting for the server."""
    for r, off in zip(reqs, offsets):
        target = t0 + off
        while True:
            d = target - time.perf_counter()
            if d <= 0:
                break
            time.sleep(d if d > 2e-3 else 0)       # coarse sleep, then spin for sub-ms accuracy
        r.t_arrival = target
        q.put(r)
    q.put(None)


@dataclass
class PointResult:
    rate: float
    window_ms: float
    max_tokens: int
    n: int
    offered_rps: float
    achieved_rps: float
    lat_ms: np.ndarray
    wait_ms: np.ndarray
    service_ms: np.ndarray
    batch_sizes: np.ndarray
    extra: dict = field(default_factory=dict)

    def summary(self, slos_ms=(50, 100, 200)) -> dict:
        p = lambda a, x: float(np.percentile(a, x))
        out = dict(rate=self.rate, window_ms=self.window_ms, max_tokens=self.max_tokens, n=self.n,
                   offered_rps=round(self.offered_rps, 2), achieved_rps=round(self.achieved_rps, 2),
                   lat_p50_ms=round(p(self.lat_ms, 50), 2), lat_p95_ms=round(p(self.lat_ms, 95), 2),
                   lat_p99_ms=round(p(self.lat_ms, 99), 2), lat_mean_ms=round(float(self.lat_ms.mean()), 2),
                   lat_max_ms=round(float(self.lat_ms.max()), 2),
                   wait_p50_ms=round(p(self.wait_ms, 50), 2), wait_p99_ms=round(p(self.wait_ms, 99), 2),
                   service_p50_ms=round(p(self.service_ms, 50), 2),
                   batch_mean=round(float(self.batch_sizes.mean()), 2), batch_max=int(self.batch_sizes.max()))
        dur = self.n / max(self.achieved_rps, 1e-9)
        for s in slos_ms:
            out[f"goodput_rps_slo{s}ms"] = round(float((self.lat_ms <= s).sum()) / dur, 2)
        out.update(self.extra)
        return out


def run_point(server: Server, samples: list[Sample], rate: float, n_requests: int, seed: int = 0) -> PointResult:
    reqs = [Request(i, samples[i % len(samples)], 0) for i in range(n_requests)]
    for r in reqs:
        r.tokens = tree_tokens(r.sample)
    offsets = poisson_schedule(n_requests, rate, seed)
    q: queue.Queue = queue.Queue()
    t0 = time.perf_counter() + 0.05
    g = threading.Thread(target=generate, args=(q, reqs, offsets, t0), daemon=True)
    g.start()
    done = server.serve(q, n_requests)
    g.join()
    lat = np.array([(r.t_done - r.t_arrival) * 1e3 for r in done])
    wait = np.array([(r.t_start - r.t_arrival) * 1e3 for r in done])
    svc = np.array([(r.t_done - r.t_start) * 1e3 for r in done])
    sizes = np.bincount([r.batch for r in done])
    sizes = sizes[sizes > 0]
    span = max(r.t_done for r in done) - min(r.t_arrival for r in done)
    # realised offered rate: a finite Poisson draw deviates from the nominal rate by ~1/sqrt(n)
    offered = (n_requests - 1) / max(float(offsets[-1] - offsets[0]), 1e-9)
    return PointResult(rate=rate, window_ms=server.cfg.window_ms, max_tokens=server.cfg.max_tokens,
                       n=len(done), offered_rps=offered, achieved_rps=len(done) / span, lat_ms=lat, wait_ms=wait,
                       service_ms=svc, batch_sizes=sizes)


def single_request_latency(server: Server, samples: list[Sample], repeats: int = 20) -> float:
    """Unloaded latency floor: one request at a time, ms (median)."""
    ts = []
    for i in range(repeats):
        s = samples[i % len(samples)]
        r = Request(i, s, tree_tokens(s))
        t0 = time.perf_counter()
        server.run_batch([r])
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))
