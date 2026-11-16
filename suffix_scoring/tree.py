"""Prefix-tree construction, batching / packing on the CPU, and attention masks.

Structure of the suffixes:
    log P(t1 | p)          one distribution for ALL candidates  -> logits at the last prefix position
    log P(t2 | p, t1)      depends only on t1                    -> one node per unique t1   (<= 26)
    log P(t3 | p, t1, t2)  depends only on (t1, t2)              -> one node per unique pair (~310 of 676)
    t3 itself never needs a forward.

Sequence layout:  [prefix (L) | t1 nodes (n1, position L) | pair nodes (n2, position L+1)]
Mask (True = may attend): prefix causal; every node sees the prefix and itself; a pair node also sees
its parent t1 node.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .data import Sample

KEY = 4096  # > any single-character token id; used to encode (t1-node, t2) pairs


def context_tokens(sample: Sample, start_token_id: int) -> torch.Tensor:
    """ASSUMPTION: a length-0 prefix is replaced by a single start token (<|endoftext|>)."""
    if sample.prefix.numel() == 0:
        return torch.tensor([start_token_id], dtype=torch.long)
    return sample.prefix


@dataclass
class Tree:
    ids: torch.Tensor      # [T]
    pos: torch.Tensor      # [T]
    parent: torch.Tensor   # [T] index of the parent t1 node for pair nodes, else -1
    L: int
    n1: int
    n2: int
    rel_rows: torch.Tensor  # [400, 3] row (relative to position L-1) holding the logits for t1/t2/t3

    @property
    def T(self) -> int:
        return self.ids.numel()


def build_tree(ctx: torch.Tensor, cand: torch.Tensor) -> Tree:
    dev = ctx.device
    L = ctx.numel()
    u1, inv1 = torch.unique(cand[:, 0], return_inverse=True)
    u2, inv2 = torch.unique(inv1 * KEY + cand[:, 1], return_inverse=True)
    n1, n2 = u1.numel(), u2.numel()
    ids = torch.cat([ctx, u1, u2 % KEY])
    pos = torch.cat([torch.arange(L, device=dev), torch.full((n1,), L, device=dev),
                     torch.full((n2,), L + 1, device=dev)])
    parent = torch.full((L + n1 + n2,), -1, dtype=torch.long, device=dev)
    parent[L + n1:] = L + torch.div(u2, KEY, rounding_mode="floor")
    rel = torch.stack([torch.zeros_like(inv1), 1 + inv1, 1 + n1 + inv2], dim=1)
    return Tree(ids=ids, pos=pos, parent=parent, L=L, n1=n1, n2=n2, rel_rows=rel)


def dense_tree_mask(L: torch.Tensor, N: torch.Tensor, parent: torch.Tensor, T: int) -> torch.Tensor:
    """[B, T, T] bool built on the device of `parent`. Padding rows (>= N) only see themselves."""
    dev = parent.device
    i = torch.arange(T, device=dev).view(1, T, 1)
    j = torch.arange(T, device=dev).view(1, 1, T)
    Lb, Nb = L.view(-1, 1, 1), N.view(-1, 1, 1)
    sees_prefix = (j < Lb) & (((i < Lb) & (j <= i)) | ((i >= Lb) & (i < Nb)))
    return sees_prefix | (i == j) | (j == parent.unsqueeze(-1))


def _pin(obj):
    for k, v in obj.__dict__.items():
        if isinstance(v, torch.Tensor):
            setattr(obj, k, v.pin_memory())
    return obj


# ---------------------------------------------------------------- dense batches (padding)
@dataclass
class DenseGroup:
    ids: torch.Tensor; pos: torch.Tensor; L: torch.Tensor; N: torch.Tensor; parent: torch.Tensor
    sel_rows: torch.Tensor; cand_row: torch.Tensor; cand_tok: torch.Tensor; order: torch.Tensor
    pin_memory = _pin


@dataclass
class PreparedBatch:
    idx: list[int]
    cand_idx: np.ndarray
    groups: list = field(default_factory=list)
    padded_tokens: int = 0
    real_tokens: int = 0

    def pin_memory(self):
        for g in self.groups:
            g.pin_memory()
        return self

    def __len__(self):
        return len(self.idx)


def _rows_for(tree: Tree, base_row: int, row0: int, sample: Sample):
    sel = base_row + torch.arange(tree.L - 1, tree.T)
    return sel, tree.rel_rows + row0, sample.cand_tokens, tree.T - tree.L + 1


def prepare_dense(samples: list[Sample], start_token_id: int, max_tokens: int, pad_multiple: int = 128
                  ) -> PreparedBatch:
    """Sort by length, group so that B * T_pad <= max_tokens, right-pad each group."""
    trees = [build_tree(context_tokens(s, start_token_id), s.cand_tokens) for s in samples]
    rnd = lambda t: -(-t // pad_multiple) * pad_multiple
    order = sorted(range(len(samples)), key=lambda i: trees[i].T)
    groups, cur, cur_max = [], [], 0
    for i in order:
        m = max(cur_max, trees[i].T)
        if cur and rnd(m) * (len(cur) + 1) > max_tokens:
            groups.append(cur); cur, m = [], trees[i].T
        cur.append(i); cur_max = m
    if cur:
        groups.append(cur)
    pb = PreparedBatch(idx=[s.idx for s in samples], cand_idx=np.stack([s.cand_idx.numpy() for s in samples]))
    for g in groups:
        B, T = len(g), rnd(max(trees[i].T for i in g))
        ids, pos = torch.zeros(B, T, dtype=torch.long), torch.zeros(B, T, dtype=torch.long)
        parent = torch.full((B, T), -1, dtype=torch.long)
        Ls, Ns = torch.zeros(B, dtype=torch.long), torch.zeros(B, dtype=torch.long)
        sel, crow, ctok, r0 = [], [], [], 0
        for b, i in enumerate(g):
            t = trees[i]
            ids[b, :t.T], pos[b, :t.T], parent[b, :t.T] = t.ids, t.pos, t.parent
            Ls[b], Ns[b] = t.L, t.T
            s_, c_, k_, n_ = _rows_for(t, b * T, r0, samples[i])
            sel.append(s_); crow.append(c_); ctok.append(k_); r0 += n_
        pb.groups.append(DenseGroup(ids, pos, Ls, Ns, parent, torch.cat(sel), torch.cat(crow),
                                    torch.cat(ctok), torch.tensor(g)))
        pb.padded_tokens += B * T
        pb.real_tokens += sum(trees[i].T for i in g)
    return pb


# ---------------------------------------------------------------- packed batches (no padding)
@dataclass
class Pack:
    ids: torch.Tensor; pos: torch.Tensor; doc: torch.Tensor; is_prefix: torch.Tensor; parent: torch.Tensor
    sel_rows: torch.Tensor; cand_row: torch.Tensor; cand_tok: torch.Tensor; order: torch.Tensor
    pin_memory = _pin


def prepare_packed(samples: list[Sample], start_token_id: int, pack_len: int) -> PreparedBatch:
    """Greedy first-fit (longest first) packing of whole sequences into rows of exactly pack_len tokens.
    Only the end of each pack is padded (static shape for torch.compile); padding is its own 'document'."""
    trees = [build_tree(context_tokens(s, start_token_id), s.cand_tokens) for s in samples]
    lens = [t.T for t in trees]
    if max(lens) > pack_len:
        raise ValueError(f"pack_len {pack_len} < longest sequence {max(lens)}")
    bins, fill = [], []
    for i in sorted(range(len(samples)), key=lambda i: -lens[i]):
        for b in range(len(bins)):
            if fill[b] + lens[i] <= pack_len:
                bins[b].append(i); fill[b] += lens[i]
                break
        else:
            bins.append([i]); fill.append(lens[i])
    pb = PreparedBatch(idx=[s.idx for s in samples], cand_idx=np.stack([s.cand_idx.numpy() for s in samples]))
    for members in bins:
        P = pack_len
        ids, pos = torch.zeros(P, dtype=torch.long), torch.zeros(P, dtype=torch.long)
        doc, parent = torch.full((P,), -1, dtype=torch.long), torch.full((P,), -1, dtype=torch.long)
        is_prefix = torch.zeros(P, dtype=torch.bool)
        sel, crow, ctok, off, r0 = [], [], [], 0, 0
        for d, i in enumerate(members):
            t = trees[i]
            ids[off:off + t.T], pos[off:off + t.T], doc[off:off + t.T] = t.ids, t.pos, d
            is_prefix[off:off + t.L] = True
            parent[off:off + t.T] = torch.where(t.parent >= 0, t.parent + off, t.parent)
            s_, c_, k_, n_ = _rows_for(t, off, r0, samples[i])
            sel.append(s_); crow.append(c_); ctok.append(k_); r0 += n_
            off += t.T
        pb.groups.append(Pack(ids, pos, doc, is_prefix, parent, torch.cat(sel), torch.cat(crow),
                              torch.cat(ctok), torch.tensor(members)))
        pb.padded_tokens += P
        pb.real_tokens += off
    return pb
