"""Shared plumbing: build a scorer, run it over a dataset with timing, collect results."""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from . import profiling
from .data import SuffixScoringDataset
from .model import load_model, load_tiny_model
from .scorers import get_scorer


def parse_options(opts: list[str] | None, json_str: str = "") -> dict:
    """['pack_len=16384', 'fp8=mlp', 'compile=false'] (+ optional JSON) -> dict with typed values."""
    out = json.loads(json_str) if json_str else {}
    for o in opts or []:
        k, v = o.split("=", 1)
        out[k] = _typed(v)
    return out


def _typed(v: str):
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    for t in (int, float):
        try:
            return t(v)
        except ValueError:
            pass
    if v.startswith("["):
        return json.loads(v)
    return v


def build(scorer: str, options: dict, start_token_id: int, dtype: str = "bf16", device: str = "cuda",
          model_id: str | None = None, tiny: bool = False):
    if tiny:
        device = "cpu"
        model = load_tiny_model(dtype=dtype, device=device)
    else:
        kw = {"model_id": model_id} if model_id else {}
        model = load_model(dtype=dtype, device=device, **kw)
    return get_scorer(scorer)(model, start_token_id, device=device, **options), model


@dataclass
class RunResult:
    n: int
    seconds: float
    seq_per_s: float
    idx: list
    scores: np.ndarray
    cand_idx: np.ndarray
    timer_report: str
    phase_report: str
    padded_tokens: int = 0
    real_tokens: int = 0


def make_loader(ds, batch_size: int, num_workers: int, collate):
    pin = torch.cuda.is_available() and collate is not None
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                      collate_fn=collate or (lambda b: b), pin_memory=pin,
                      prefetch_factor=(4 if num_workers > 0 else None))


def run(scorer, ds: SuffixScoringDataset, limit: int = 0, batch_size: int = 1, num_workers: int = 0,
        warmup_batches: int = 1, sync_phases: bool = False, keep_scores: bool = True,
        title: str = "") -> RunResult:
    n = min(limit or len(ds), len(ds))
    subset = Subset(ds, range(n))
    collate = getattr(scorer, "collate", None)

    # warm-up: compilation, autotuning, allocator. Not timed.
    it = iter(make_loader(subset, batch_size, 0, collate))
    for _ in range(warmup_batches):
        b = next(it, None)
        if b is None:
            break
        scorer.score(b)

    profiling.reset_phases(sync_phases)
    timer = profiling.Timer()
    idx, scores, cands, pad_tok, real_tok = [], [], [], 0, 0
    loader_it = iter(make_loader(subset, batch_size, num_workers, collate))
    while True:
        with timer.section("data"):
            batch = next(loader_it, None)
        if batch is None:
            break
        with timer.section("score"):
            s = scorer.score(batch)
        if isinstance(batch, list):
            idx += [b.idx for b in batch]
            cands += [b.cand_idx.numpy() for b in batch]
        else:
            idx += list(batch.idx)
            cands += list(batch.cand_idx)
            pad_tok += getattr(batch, "padded_tokens", 0)
            real_tok += getattr(batch, "real_tokens", 0)
        if keep_scores:
            scores.append(s.numpy())
    total = timer.total
    return RunResult(n=n, seconds=total, seq_per_s=n / max(total, 1e-9), idx=idx,
                     scores=np.concatenate(scores) if scores else np.zeros((0, 400), np.float32),
                     cand_idx=np.stack(cands), timer_report=timer.report(n, title),
                     phase_report=profiling.phase_report(n), padded_tokens=pad_tok, real_tokens=real_tok)
