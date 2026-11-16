from __future__ import annotations

from functools import partial

import torch

from ..data import Sample
from ..profiling import phase
from ..qwen3 import forward_dense
from ..tree import PreparedBatch, build_tree, context_tokens, dense_tree_mask, prepare_dense


def _head_logprobs(model, h_sel: torch.Tensor, cand_row: torch.Tensor, cand_tok: torch.Tensor,
                   chunk: int = 2048) -> torch.Tensor:
    """log P = logit[tok] - logsumexp(row), fp32, row-chunked. Returns [n_cand, 3]."""
    logits = model.lm_head(model.model.norm(h_sel))
    lse = torch.cat([torch.logsumexp(logits[s:s + chunk].float(), dim=-1) for s in range(0, logits.shape[0], chunk)])
    return logits[cand_row, cand_tok].float() - lse[cand_row]


class TreeScorer:
    """Step 2: ONE forward per sequence over [prefix | unique t1 | unique (t1,t2)] with a tree mask.
    No KV-cache copies; LM head only on the ~340 needed rows."""

    def __init__(self, model, start_token_id: int, device: str = "cuda"):
        self.model, self.start, self.device = model, start_token_id, device

    @torch.inference_mode()
    def score_one(self, s: Sample) -> torch.Tensor:
        with phase("tree:build"):
            t = build_tree(context_tokens(s, self.start).to(self.device), s.cand_tokens.to(self.device))
            mask = dense_tree_mask(torch.tensor([t.L], device=self.device),
                                   torch.tensor([t.T], device=self.device), t.parent[None], t.T)
        with phase("tree:forward"):
            h = forward_dense(self.model, t.ids[None], t.pos[None], mask)[0]
        with phase("tree:head"):
            lp = _head_logprobs(self.model, h[t.L - 1:], t.rel_rows, s.cand_tokens.to(self.device))
        return lp.sum(-1)

    def score(self, samples: list[Sample]) -> torch.Tensor:
        return torch.stack([self.score_one(s) for s in samples]).cpu()


class BatchedTreeScorer:
    """Step 3: several sequences per forward (sorted by length, right-padded groups with B*T <= max_tokens),
    dense [B, T, T] tree masks built on the GPU, optional torch.compile of the layer stack."""

    def __init__(self, model, start_token_id: int, device: str = "cuda", max_tokens: int = 16384,
                 pad_multiple: int = 128, compile: bool = False, gqa: str = "enable_gqa"):
        self.model, self.device, self.gqa = model, device, gqa
        self.prepare = partial(prepare_dense, start_token_id=start_token_id, max_tokens=max_tokens,
                               pad_multiple=pad_multiple)
        fwd = lambda ids, pos, mask: forward_dense(self.model, ids, pos, mask, gqa=self.gqa)
        self._fwd = torch.compile(fwd, dynamic=True) if compile else fwd

    @torch.no_grad()   # not inference_mode: torch.compile cannot trace inference tensors
    def score(self, batch) -> torch.Tensor:
        if not isinstance(batch, PreparedBatch):
            with phase("btree:prepare"):
                batch = self.prepare(batch)
        out = torch.empty(len(batch), 400, dtype=torch.float32, device=self.device)
        for g in batch.groups:
            with phase("btree:h2d+mask"):
                d = {k: v.to(self.device, non_blocking=True) for k, v in g.__dict__.items()}
                B, T = d["ids"].shape
                mask = dense_tree_mask(d["L"], d["N"], d["parent"], T)
            with phase("btree:forward"):
                h = self._fwd(d["ids"], d["pos"], mask)
            with phase("btree:head"):
                lp = _head_logprobs(self.model, h.reshape(B * T, -1)[d["sel_rows"]], d["cand_row"], d["cand_tok"])
                out[d["order"]] = lp.sum(-1).view(-1, 400)
        return out.cpu()


class PipelinedTreeScorer(BatchedTreeScorer):
    """Step 4: same computation, but batch preparation runs in DataLoader worker processes
    (exposed as .collate), overlapping CPU work with the GPU. torch.compile on by default."""

    def __init__(self, model, start_token_id: int, device: str = "cuda", max_tokens: int = 16384,
                 pad_multiple: int = 128, compile: bool = True, gqa: str = "enable_gqa"):
        super().__init__(model, start_token_id, device, max_tokens, pad_multiple, compile, gqa)
        self.collate = self.prepare
