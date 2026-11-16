"""Scorer registry. Every scorer:  Scorer(model, start_token_id, device=..., **options)
    .score(batch) -> FloatTensor [B, 400] on CPU (fp32)
    .collate (optional): CPU batch preparation, run in DataLoader workers
"""
from .flex import FlexTreeScorer
from .kvcache import KVCacheScorer
from .naive import NaiveScorer
from .tree import BatchedTreeScorer, PipelinedTreeScorer, TreeScorer

SCORERS = {
    "naive": NaiveScorer,                    # reference: every candidate as its own sequence
    "kvcache": KVCacheScorer,                # step 1: prefix once, copied KV cache per candidate
    "tree": TreeScorer,                      # step 2: one forward per sequence, tree mask
    "tree_batched": BatchedTreeScorer,       # step 3: several sequences per forward (+ torch.compile)
    "tree_pipelined": PipelinedTreeScorer,   # step 4: CPU prep in DataLoader workers, GPU-built masks
    "flex": FlexTreeScorer,                  # step 5: packing + FlexAttention block-sparse mask (+ fp8)
}


def get_scorer(name: str):
    if name not in SCORERS:
        raise KeyError(f"unknown scorer {name!r}; choose from {sorted(SCORERS)}")
    return SCORERS[name]
