"""Timing utilities: sectioned wall-clock timer, per-phase timing, torch.profiler wrapper."""
from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager

import torch
from torch.profiler import record_function

# With CUDA, CPU-side time of a range only measures kernel LAUNCHES. When SYNC_PHASES is True every
# phase synchronises the GPU at its end -> PHASE_TIMES holds true wall time per phase (diagnosis only).
SYNC_PHASES = False
PHASE_TIMES: dict[str, float] = defaultdict(float)


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def phase(name: str):
    """Named range: shows up in torch.profiler / Perfetto traces and in PHASE_TIMES."""
    with record_function(name):
        if SYNC_PHASES:
            _sync()
        t0 = time.perf_counter()
        yield
        if SYNC_PHASES:
            _sync()
        PHASE_TIMES[name] += time.perf_counter() - t0


def reset_phases(sync: bool = False):
    global SYNC_PHASES
    SYNC_PHASES = sync
    PHASE_TIMES.clear()


def phase_report(n_sequences: int) -> str:
    if not PHASE_TIMES:
        return ""
    tot = sum(PHASE_TIMES.values())
    note = "GPU-synchronised" if SYNC_PHASES else "NOT synchronised: launch times only"
    lines = [f"== phases ({note}) =="]
    for k, v in sorted(PHASE_TIMES.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {k:<24s} {v:9.3f} s  ({100 * v / tot:5.1f} %)  {1000 * v / n_sequences:8.2f} ms/seq")
    return "\n".join(lines)


class Timer:
    """Accumulates wall time per named section, synchronising CUDA around each section."""

    def __init__(self):
        self.totals, self.counts = defaultdict(float), defaultdict(int)

    @contextmanager
    def section(self, name: str):
        _sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            _sync()
            self.totals[name] += time.perf_counter() - t0
            self.counts[name] += 1

    @property
    def total(self) -> float:
        return sum(self.totals.values())

    def report(self, n_sequences: int, title: str = "") -> str:
        lines = [f"== {title or 'timing'} =="]
        for k, v in self.totals.items():
            lines.append(f"  {k:<10s} {v:9.3f} s  ({100 * v / max(self.total, 1e-9):5.1f} %)  calls={self.counts[k]}")
        lines.append(f"  {'TOTAL':<10s} {self.total:9.3f} s   ->  {n_sequences / max(self.total, 1e-9):.2f} seq/s")
        return "\n".join(lines)


def _dev_time(e) -> float:
    for attr in ("device_time_total", "cuda_time_total"):
        if hasattr(e, attr):
            return getattr(e, attr)
    return 0.0


def profile_batches(fn, batches, trace_path: str = "", rows: int = 25) -> str:
    """Run fn(batch) for each batch under torch.profiler. Returns a text report; optional Perfetto trace."""
    from torch.profiler import ProfilerActivity, profile
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if torch.cuda.is_available() else [])
    with profile(activities=acts, record_shapes=True) as prof:
        for b in batches:
            fn(b)
        _sync()
    ka = prof.key_averages()
    phases = [e for e in ka if ":" in e.key and not e.key.startswith(("aten", "cuda", "Torch"))]
    lines = ["== profiled phases ==", f"  {'phase':<24s} {'calls':>6s} {'CPU ms':>10s} {'GPU ms':>10s}"]
    for e in sorted(phases, key=lambda e: -max(e.cpu_time_total, _dev_time(e))):
        lines.append(f"  {e.key:<24s} {e.count:>6d} {e.cpu_time_total / 1e3:>10.1f} {_dev_time(e) / 1e3:>10.1f}")
    sort_key = "cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"
    try:
        lines.append(ka.table(sort_by=sort_key, row_limit=rows))
    except Exception:
        lines.append(ka.table(sort_by="device_time_total", row_limit=rows))
    if trace_path:
        prof.export_chrome_trace(trace_path)
        lines.append(f"trace written to {trace_path}  (open in https://ui.perfetto.dev)")
    return "\n".join(lines)
