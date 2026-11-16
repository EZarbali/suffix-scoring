from __future__ import annotations

from functools import partial

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from ..profiling import phase
from ..qwen3 import forward_packed
from ..tree import PreparedBatch, prepare_packed

torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
_FLEX = {}


def _flex_fn(compiled: bool):
    if compiled not in _FLEX:
        _FLEX[compiled] = torch.compile(flex_attention, dynamic=False) if compiled else flex_attention
    return _FLEX[compiled]


class FlexTreeScorer:
    """Step 5 (best): sequences packed back to back into rows of `pack_len` tokens (no per-sequence
    padding), block-sparse tree mask via FlexAttention (empty 128x128 tiles skipped), native GQA,
    torch.compile, fused norm + LM head + logsumexp.

    Options
      pack_len          tokens per forward (static shape)
      compile           torch.compile the layer stack and FlexAttention
      fused_head        compile norm + LM head + logsumexp into one graph (no fp32 copy of the logits)
      prune_last_layer  last layer: Q / attention / o_proj / MLP only for the rows that need logits
      fp8               "none" | "mlp_gate_up" | "mlp" | "all"   (BONUS, see fp8.py)
      fp8_mode          "auto" (real on sm90+, else fake) | "real" | "fake"
      fp8_skip_layers   layer ids kept in bf16
      compile_mode      "default" | "reduce-overhead" (CUDA graphs) | "max-autotune"
    """

    def __init__(self, model, start_token_id: int, device: str = "cuda", pack_len: int = 16384,
                 compile: bool = True, fused_head: bool = True, prune_last_layer: bool = False,
                 fp8: str = "none", fp8_mode: str = "auto", fp8_skip_layers: list | None = None,
                 block_size: int = 128, sel_multiple: int = 512, compile_mode: str = "default"):
        assert pack_len % block_size == 0
        self.model, self.m, self.device = model, model.model, device
        self.pack_len, self.block_size, self.sel_multiple = pack_len, block_size, sel_multiple
        self.compile, self.prune = compile, prune_last_layer
        self.fp8_info = None
        if fp8 != "none":
            from ..fp8 import apply_fp8
            self.fp8_info = apply_fp8(model, targets=fp8, mode=fp8_mode, skip_layers=tuple(fp8_skip_layers or ()))
        self.collate = partial(prepare_packed, start_token_id=start_token_id, pack_len=pack_len)
        attn = _flex_fn(compile)
        attn_fn = lambda q, k, v, bm, scale: attn(q, k, v, block_mask=bm, scale=scale, enable_gqa=True)
        fwd = lambda ids, pos, bm, rows: forward_packed(self.model, ids, pos, bm, attn_fn, rows)
        # compile_mode="reduce-overhead" adds CUDA graphs (one per shape): removes kernel-launch overhead,
        # which dominates for small batches (realtime serving at low load).
        self._fwd = torch.compile(fwd, dynamic=False, mode=compile_mode) if compile else fwd
        head = lambda h: self._head_impl(h)
        self._head = torch.compile(head, dynamic=True) if fused_head else self._head_unfused

    # -- LM head: log P(tok) = logit[tok] - logsumexp(row)
    def _head_impl(self, h_sel):
        logits = self.model.lm_head(self.m.norm(h_sel))
        return logits, torch.logsumexp(logits.float(), dim=-1)

    def _head_unfused(self, h_sel, chunk: int = 2048):
        logits = self.model.lm_head(self.m.norm(h_sel))
        return logits, torch.cat([torch.logsumexp(logits[s:s + chunk].float(), dim=-1)
                                  for s in range(0, logits.shape[0], chunk)])

    # -- masks
    def _block_mask(self, doc, is_prefix, parent, q_rows=None):
        P = doc.shape[0]

        def rule(q, kv):
            same = doc[q] == doc[kv]
            sees_prefix = is_prefix[kv] & ((~is_prefix[q]) | (kv <= q))
            return same & (sees_prefix | (kv == q) | (kv == parent[q]))

        if q_rows is None:
            mask_mod, qlen = (lambda b, h, q, kv: rule(q, kv)), P
        else:
            mask_mod, qlen = (lambda b, h, q, kv: rule(q_rows[q], kv)), q_rows.shape[0]
        return create_block_mask(mask_mod, B=None, H=None, Q_LEN=qlen, KV_LEN=P, device=doc.device,
                                 BLOCK_SIZE=self.block_size, _compile=self.compile)

    @torch.no_grad()
    def score(self, batch) -> torch.Tensor:
        if not isinstance(batch, PreparedBatch):
            with phase("flex:prepare"):
                batch = self.collate(batch)
        out = torch.empty(len(batch), 400, dtype=torch.float32, device=self.device)
        for p in batch.groups:
            with phase("flex:h2d+mask"):
                d = {k: v.to(self.device, non_blocking=True) for k, v in p.__dict__.items()}
                bm = self._block_mask(d["doc"], d["is_prefix"], d["parent"])
                rows = None
                if self.prune:
                    R = d["sel_rows"].numel()
                    Rp = -(-R // self.sel_multiple) * self.sel_multiple      # few static shapes
                    rows = torch.cat([d["sel_rows"], d["sel_rows"][:1].expand(Rp - R)])
                    bm = (bm, self._block_mask(d["doc"], d["is_prefix"], d["parent"], q_rows=rows))
            with phase("flex:forward"):
                h = self._fwd(d["ids"][None], d["pos"][None], bm, rows)
            with phase("flex:head"):
                h_sel = h[: d["sel_rows"].numel()] if self.prune else h[d["sel_rows"]]
                logits, lse = self._head(h_sel)
            with phase("flex:gather"):
                lp = logits[d["cand_row"], d["cand_tok"]].float() - lse[d["cand_row"]]
                out[d["order"]] = lp.sum(-1).view(-1, 400)
        return out.cpu()
