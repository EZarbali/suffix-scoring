from __future__ import annotations

import copy

import torch

from ..data import Sample
from ..profiling import phase
from ..tree import context_tokens


class KVCacheScorer:
    """Step 1: prefix forward ONCE (use_cache), then [t1, t2] of each candidate against a copy of the
    prefix KV cache. t3 never goes through the model. Profiling showed the remaining cost is memory
    traffic: cache copies, DynamicCache torch.cat, and HF's repeat_kv for GQA."""

    def __init__(self, model, start_token_id: int, device: str = "cuda", candidates_per_forward: int = 64):
        self.model, self.start, self.device, self.cpf = model, start_token_id, device, candidates_per_forward

    @torch.inference_mode()
    def score_one(self, s: Sample) -> torch.Tensor:
        ctx = context_tokens(s, self.start).to(self.device)
        cand = s.cand_tokens.to(self.device)
        with phase("kv:prefix_forward"):
            pre = self.model(input_ids=ctx.unsqueeze(0), use_cache=True, logits_to_keep=1)
        with phase("kv:logp_t1"):
            out = torch.log_softmax(pre.logits[0, -1].float(), dim=-1)[cand[:, 0]].clone()
        for a in range(0, cand.shape[0], self.cpf):
            c = cand[a:a + self.cpf]
            with phase("kv:cache_copy"):
                cache = copy.deepcopy(pre.past_key_values)          # the model appends -> fresh copy
                cache.batch_repeat_interleave(c.shape[0])
            with phase("kv:suffix_forward"):
                o = self.model(input_ids=c[:, :2], past_key_values=cache, use_cache=True)
            with phase("kv:logp_t2_t3"):
                logp = torch.log_softmax(o.logits.float(), dim=-1)
                out[a:a + c.shape[0]] += logp.gather(-1, c[:, 1:].unsqueeze(-1)).squeeze(-1).sum(-1)
        return out

    def score(self, samples: list[Sample]) -> torch.Tensor:
        return torch.stack([self.score_one(s) for s in samples]).cpu()
