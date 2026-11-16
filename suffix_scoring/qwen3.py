"""Explicit Qwen3 decoder forward over the Hugging Face modules (same weights, same ops, same order).

Writing the loop ourselves gives control over the attention mask, the position ids, which rows go
through the final norm + LM head, and lets torch.compile fuse the elementwise ops.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb


def _qkv(layer, x):
    a = layer.self_attn
    shp = (*x.shape[:2], -1, a.head_dim)
    q = a.q_norm(a.q_proj(x).view(shp)).transpose(1, 2)        # [B, Hq,  T, D]
    k = a.k_norm(a.k_proj(x).view(shp)).transpose(1, 2)        # [B, Hkv, T, D]
    v = a.v_proj(x).view(shp).transpose(1, 2)
    return q, k, v


def forward_dense(model, ids: torch.Tensor, pos: torch.Tensor, mask: torch.Tensor, gqa: str = "enable_gqa"):
    """ids/pos [B, T], mask [B, T, T] bool -> final hidden states (before norm) [B, T, d].
    gqa="enable_gqa": SDPA handles grouped KV heads; "repeat": expand K/V to all heads first."""
    m = model.model
    h = m.embed_tokens(ids)
    cos, sin = m.rotary_emb(h, pos)
    B, T = ids.shape
    for layer in m.layers:
        a = layer.self_attn
        q, k, v = _qkv(layer, layer.input_layernorm(h))
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if gqa == "repeat":
            rep = q.shape[1] // k.shape[1]
            k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None], scale=a.scaling)
        else:
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None], scale=a.scaling,
                                               enable_gqa=True)
        h = h + a.o_proj(o.transpose(1, 2).reshape(B, T, -1))
        h = h + layer.mlp(layer.post_attention_layernorm(h))
    return h


def forward_packed(model, ids: torch.Tensor, pos: torch.Tensor, block_mask, attn_fn,
                   last_layer_rows: torch.Tensor | None = None):
    """Packed variant with FlexAttention. ids/pos [1, P] -> hidden [P, d], or [R, d] if last_layer_rows.

    last_layer_rows (last-layer pruning): in the final layer only these rows need an output (the last
    prefix position and the tree nodes). All other positions only contribute K and V, so Q, attention,
    o_proj and the MLP of the last layer are computed for these rows only. Exact (same math)."""
    m = model.model
    h = m.embed_tokens(ids)
    cos, sin = m.rotary_emb(h, pos)
    P = ids.shape[1]
    n_layers = len(m.layers)
    for li, layer in enumerate(m.layers):
        a = layer.self_attn
        x = layer.input_layernorm(h)
        q, k, v = _qkv(layer, x)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if last_layer_rows is not None and li == n_layers - 1:
            q_sel = q[:, :, last_layer_rows]                         # [1, Hq, R, D]
            o = attn_fn(q_sel, k, v, block_mask[1], a.scaling)      # pruned-query block mask
            h_sel = h[:, last_layer_rows]
            h_sel = h_sel + a.o_proj(o.transpose(1, 2).reshape(1, last_layer_rows.numel(), -1))
            h_sel = h_sel + layer.mlp(layer.post_attention_layernorm(h_sel))
            return h_sel[0]
        bm = block_mask[0] if isinstance(block_mask, tuple) else block_mask
        o = attn_fn(q, k, v, bm, a.scaling)
        h = h + a.o_proj(o.transpose(1, 2).reshape(1, P, -1))
        h = h + layer.mlp(layer.post_attention_layernorm(h))
    return h[0]
