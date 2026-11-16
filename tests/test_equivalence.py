"""Every optimised scorer must reproduce the naive reference (fp32, tiny random Qwen3 on CPU)."""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from suffix_scoring import model as M  # noqa: E402
from suffix_scoring.data import SuffixScoringDataset, generate_dataset  # noqa: E402
from suffix_scoring.runner import build, run  # noqa: E402
from suffix_scoring.tree import build_tree, dense_tree_mask  # noqa: E402

ATOL = 1e-4


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("data") / "tiny.npz")
    generate_dataset(path, 10, M.tiny_char_token_ids(), 72, 512, start_token_id=1, seed=0,
                     force_lengths=[0, 1, 2048])
    return SuffixScoringDataset(path)


@pytest.fixture(scope="module")
def reference(dataset):
    scorer, _ = build("naive", {}, dataset.start_token_id, dtype="fp32", tiny=True)
    return run(scorer, dataset, batch_size=1, warmup_batches=0).scores


CASES = [
    ("naive", {"logits_mode": "full", "candidates_per_forward": 1}, 1),
    ("kvcache", {"candidates_per_forward": 64}, 1),
    ("tree", {}, 1),
    ("tree_batched", {"max_tokens": 6000, "pad_multiple": 64}, 5),
    ("tree_batched", {"max_tokens": 6000, "gqa": "repeat"}, 5),
    ("tree_pipelined", {"max_tokens": 6000, "compile": False}, 5),
    ("flex", {"pack_len": 4096, "compile": False}, 5),
    ("flex", {"pack_len": 4096, "compile": False, "fused_head": False}, 5),
    ("flex", {"pack_len": 4096, "compile": False, "prune_last_layer": True, "sel_multiple": 128}, 5),
]


@pytest.mark.parametrize("name,opts,bs", CASES, ids=[f"{c[0]}-{i}" for i, c in enumerate(CASES)])
def test_matches_naive(dataset, reference, name, opts, bs):
    scorer, _ = build(name, opts, dataset.start_token_id, dtype="fp32", tiny=True)
    r = run(scorer, dataset, batch_size=bs, warmup_batches=0)
    assert np.abs(r.scores - reference).max() < ATOL


def test_fp8_fake_close(dataset, reference):
    scorer, _ = build("flex", {"pack_len": 4096, "compile": False, "fp8": "mlp_gate_up", "fp8_mode": "fake"},
                      dataset.start_token_id, dtype="fp32", tiny=True)
    r = run(scorer, dataset, batch_size=5, warmup_batches=0)
    assert np.abs(r.scores - reference).max() < 0.1          # quantised, but close


def test_mask_semantics():
    ctx = torch.arange(100, 105)                               # L = 5
    cand = torch.tensor([[10, 36, 62], [10, 37, 63], [11, 36, 62]])
    t = build_tree(ctx, cand)
    assert (t.n1, t.n2) == (2, 3)
    m = dense_tree_mask(torch.tensor([t.L]), torch.tensor([t.T]), t.parent[None], t.T)[0]
    L = t.L
    assert m[:L, :L].equal(torch.ones(L, L, dtype=torch.bool).tril())   # causal prefix
    assert m[L:, :L].all()                                                # nodes see the prefix
    assert not m[L, L + 1] and not m[L + 1, L]                            # t1 nodes do not see each other
    for r in range(L + t.n1, t.T):                                        # pair node: prefix + parent + self
        assert m[r, L:].sum() == 2 and m[r, t.parent[r]] and m[r, r]
