"""FP8 (e4m3) W8A8 linear layers for the Qwen3 decoder.

- weights: per-output-channel scales (static, computed once)
- activations: per-token scales (dynamic, computed every call)
- accumulation in fp32, output in bf16

mode "real": torch._scaled_mm with row-wise scales -> needs Hopper (sm90, H100/H200).
mode "fake": quantise -> dequantise in bf16, then a normal bf16 matmul. Same rounding of
             weights and activations as "real", no speed-up. Lets you measure the ACCURACY
             effect on GPUs without FP8 tensor cores (A100) or on CPU.
mode "auto": real if the GPU is sm90+, else fake.

Targets (which nn.Linear inside every decoder layer get replaced):
    "mlp_gate_up" : gate_proj, up_proj                    (largest GEMMs: 1024 -> 3072)
    "mlp"         : + down_proj                           (input = SiLU(gate)*up, more outliers)
    "all"         : + q_proj, k_proj, v_proj, o_proj
Embeddings, norms and the LM head always stay bf16.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

FP8 = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8).max  # 448

TARGETS = {
    "mlp_gate_up": ["mlp.gate_proj", "mlp.up_proj"],
    "mlp": ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
    "all": ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
            "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"],
}


def _quant_rows(x: torch.Tensor):
    """Row-wise symmetric FP8 quantisation of a 2D tensor. Returns (x_fp8, scale [rows] fp32)."""
    amax = x.abs().amax(dim=-1).float().clamp(min=1e-12)
    scale = amax / FP8_MAX
    xq = (x.float() / scale[:, None]).clamp(-FP8_MAX, FP8_MAX).to(FP8)
    return xq, scale


class FP8Linear(nn.Module):
    def __init__(self, lin: nn.Linear, mode: str):
        super().__init__()
        assert lin.bias is None
        self.mode, self.out_dtype = mode, lin.weight.dtype
        wq, ws = _quant_rows(lin.weight.data)                     # [out, in], per output channel
        self.register_buffer("w_fp8", wq)
        self.register_buffer("w_scale", ws)
        if mode == "fake":
            self.register_buffer("w_deq", (wq.float() * ws[:, None]).to(lin.weight.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shp = x.shape
        x2 = x.reshape(-1, shp[-1])
        xq, xs = _quant_rows(x2)                                  # per token
        if self.mode == "real":
            y = torch._scaled_mm(xq, self.w_fp8.t(), scale_a=xs[:, None], scale_b=self.w_scale[None, :],
                                 out_dtype=self.out_dtype)
        else:
            x_deq = (xq.float() * xs[:, None]).to(self.out_dtype)
            y = F.linear(x_deq, self.w_deq)
        return y.reshape(*shp[:-1], -1)


def apply_fp8(model, targets: str = "mlp_gate_up", mode: str = "auto", skip_layers: tuple = ()) -> dict:
    """Replace linears in model.model.layers in place. skip_layers: layer ids kept in bf16."""
    if mode == "auto":
        mode = ("real" if torch.cuda.is_available() and torch.cuda.get_device_capability() >= (9, 0)
                else "fake")
    n = 0
    for li, layer in enumerate(model.model.layers):
        if li in skip_layers:
            continue
        for path in TARGETS[targets]:
            parent_name, child = path.split(".")
            parent = getattr(layer, parent_name)
            setattr(parent, child, FP8Linear(getattr(parent, child), mode))
            n += 1
    return {"mode": mode, "targets": targets, "replaced_linears": n, "skipped_layers": list(skip_layers)}
