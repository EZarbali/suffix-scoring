"""Synthetic workload: generation, storage (.npz) and loading.

File layout:
    tokens    int32 [total]     all prefixes concatenated
    offsets   int64 [N + 1]     prefix i = tokens[offsets[i]:offsets[i+1]]
    cand_idx  int16 [N, 400]    candidate suffix ids in 0..6759
    char_ids  int32 [3, 26]     token ids (rows: lower, upper, digit; digits padded with -1)
    meta      json              seed, distribution, start token id, ...
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from .model import DIGITS, LOWER, N_SUFFIXES, UPPER

N_CANDIDATES = 400
LEN_MEAN, LEN_STD, LEN_MIN, LEN_MAX = 1024.0, 600.0, 0, 2048


def sample_lengths(rng: np.random.Generator, n: int) -> np.ndarray:
    """round(Normal(1024, 600)) clipped to [0, 2048]  (~4.4 % land on each bound)."""
    return np.clip(np.rint(rng.normal(LEN_MEAN, LEN_STD, size=n)), LEN_MIN, LEN_MAX).astype(np.int64)


def generate_dataset(path: str, n: int, char_ids: dict[str, list[int]], token_low: int, token_high: int,
                     start_token_id: int, seed: int = 0, force_lengths: list[int] | None = None) -> None:
    rng = np.random.default_rng(seed)
    lengths = sample_lengths(rng, n)
    if force_lengths:
        lengths[: len(force_lengths)] = force_lengths
    offsets = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    tokens = rng.integers(token_low, token_high, size=int(offsets[-1]), dtype=np.int32)
    cand_idx = np.stack([rng.choice(N_SUFFIXES, size=N_CANDIDATES, replace=False) for _ in range(n)]
                        ).astype(np.int16)
    table = np.full((3, 26), -1, dtype=np.int32)
    table[0], table[1], table[2, :10] = char_ids["lower"], char_ids["upper"], char_ids["digit"]
    meta = dict(seed=seed, n=n, len_mean=LEN_MEAN, len_std=LEN_STD, len_min=LEN_MIN, len_max=LEN_MAX,
                n_candidates=N_CANDIDATES, token_low=token_low, token_high=token_high,
                start_token_id=start_token_id)
    np.savez(path, tokens=tokens, offsets=offsets, cand_idx=cand_idx, char_ids=table, meta=json.dumps(meta))


def candidate_token_ids(cand_idx: np.ndarray, table: np.ndarray) -> np.ndarray:
    """[..., 400] suffix ids -> [..., 400, 3] token ids."""
    c = cand_idx.astype(np.int64)
    l, rem = np.divmod(c, len(UPPER) * len(DIGITS))
    u, d = np.divmod(rem, len(DIGITS))
    return np.stack([table[0][l], table[1][u], table[2][d]], axis=-1)


def suffix_name(idx: int) -> str:
    l, rem = divmod(int(idx), 26 * 10)
    u, d = divmod(rem, 10)
    return LOWER[l] + UPPER[u] + DIGITS[d]


@dataclass
class Sample:
    idx: int
    prefix: torch.Tensor        # int64 [L]  (L may be 0)
    cand_idx: torch.Tensor      # int64 [400]
    cand_tokens: torch.Tensor   # int64 [400, 3]


class SuffixScoringDataset(Dataset):
    def __init__(self, path: str):
        f = np.load(path)
        self.tokens, self.offsets, self.cand_idx = f["tokens"], f["offsets"], f["cand_idx"]
        self.char_table = f["char_ids"]
        self.meta = json.loads(str(f["meta"]))
        self.start_token_id = int(self.meta["start_token_id"])

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def length(self, i: int) -> int:
        return int(self.offsets[i + 1] - self.offsets[i])

    def lengths(self) -> np.ndarray:
        return np.diff(self.offsets)

    def __getitem__(self, i: int) -> Sample:
        a, b = self.offsets[i], self.offsets[i + 1]
        ci = self.cand_idx[i]
        return Sample(idx=i, prefix=torch.from_numpy(self.tokens[a:b].astype(np.int64)),
                      cand_idx=torch.from_numpy(ci.astype(np.int64)),
                      cand_tokens=torch.from_numpy(candidate_token_ids(ci, self.char_table)))


def collate_list(batch: list[Sample]) -> list[Sample]:
    return batch
