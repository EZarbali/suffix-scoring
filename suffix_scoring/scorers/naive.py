from __future__ import annotations

import torch

from ..data import Sample
from ..profiling import phase
from ..tree import context_tokens


class NaiveScorer:
    """The obvious, slow reference: each candidate is scored as its own sequence (context + 3 tokens)
    with a standard Hugging Face forward; log-softmax in fp32.

    candidates_per_forward: candidates batched per forward (same length -> no padding, same math).
    logits_mode: "last4" (logits_to_keep=4) or "full" (all positions; validates "last4").
    """

    def __init__(self, model, start_token_id: int, device: str = "cuda",
                 candidates_per_forward: int = 100, logits_mode: str = "last4"):
        assert logits_mode in ("last4", "full")
        self.model, self.start, self.device = model, start_token_id, device
        self.cpf, self.logits_mode = candidates_per_forward, logits_mode

    @torch.inference_mode()
    def score_one(self, s: Sample) -> torch.Tensor:
        ctx = context_tokens(s, self.start).to(self.device)
        cand = s.cand_tokens.to(self.device)
        L = ctx.numel()
        out = torch.empty(cand.shape[0], dtype=torch.float32, device=self.device)
        for a in range(0, cand.shape[0], self.cpf):
            c = cand[a:a + self.cpf]
            ids = torch.cat([ctx.unsqueeze(0).expand(c.shape[0], -1), c], dim=1)   # [m, L+3]
            with phase("naive:forward"):
                if self.logits_mode == "last4":
                    logits = self.model(input_ids=ids, logits_to_keep=4).logits[:, :3]
                else:
                    logits = self.model(input_ids=ids).logits[:, L - 1:L + 2]
            with phase("naive:logp"):
                logp = torch.log_softmax(logits.float(), dim=-1)
                out[a:a + c.shape[0]] = logp.gather(-1, c.unsqueeze(-1)).squeeze(-1).sum(-1)
        return out

    def score(self, samples: list[Sample]) -> torch.Tensor:
        return torch.stack([self.score_one(s) for s in samples]).cpu()
