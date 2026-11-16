"""Correctness metrics (scores + rankings vs. a reference / fp32 gold) and a FLOP model for MFU."""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- result files
def save_results(path: str, idx, scores, cand_idx) -> None:
    np.savez(path, idx=np.asarray(idx, dtype=np.int64), scores=np.asarray(scores, dtype=np.float32),
             cand_idx=np.asarray(cand_idx, dtype=np.int16))


def load_results(path: str) -> dict:
    f = np.load(path)
    return {k: f[k] for k in f.files}


# ---------------------------------------------------------------- ranking metrics
def ranking(scores: np.ndarray, cand_idx: np.ndarray) -> np.ndarray:
    """Descending by score, ties broken by suffix id (deterministic)."""
    return np.stack([np.lexsort((cand_idx[i], -scores[i])) for i in range(scores.shape[0])])


def kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
    da, db = np.sign(a[:, None] - a[None, :]), np.sign(b[:, None] - b[None, :])
    n = len(a)
    return float((da * db).sum() / (n * (n - 1)))


def _common(ref: dict, new: dict):
    common, ir, inew = np.intersect1d(ref["idx"], new["idx"], return_indices=True)
    if len(common) == 0:
        raise ValueError("no common sequence ids")
    if not np.array_equal(ref["cand_idx"][ir], new["cand_idx"][inew]):
        raise ValueError("candidate sets differ")
    return ref["scores"][ir], new["scores"][inew], ref["cand_idx"][ir]


def compare(ref: dict, new: dict, atol: float = 0.05, eps: float = 0.05, topk=(1, 10, 50)) -> dict:
    """atol: max |score diff| allowed. eps: a swapped pair only counts if the reference scores differ by > eps."""
    rs, ns, ci = _common(ref, new)
    diff = np.abs(rs - ns)
    rr, nr = ranking(rs, ci), ranking(ns, ci)
    n = len(rs)
    sig = 0
    for i in range(n):
        dr, dn = rs[i][:, None] - rs[i][None, :], ns[i][:, None] - ns[i][None, :]
        sig += int(np.sum((np.abs(dr) > eps) & (np.sign(dr) != np.sign(dn))) // 2)
    taus = [kendall_tau(rs[i], ns[i]) for i in range(n)]
    res = dict(n_sequences=n, max_abs_diff=float(diff.max()), mean_abs_diff=float(diff.mean()),
               exact_ranking_match=float(np.mean([np.array_equal(rr[i], nr[i]) for i in range(n)])),
               kendall_tau_mean=float(np.mean(taus)), kendall_tau_min=float(np.min(taus)),
               significant_swaps=sig)
    for k in topk:
        res[f"top{k}_overlap"] = float(np.mean([len(set(rr[i, :k]) & set(nr[i, :k])) / k for i in range(n)]))
    res["PASS"] = bool(res["max_abs_diff"] <= atol and sig == 0)
    return res


def compare_to_gold(gold: dict, ref: dict, new: dict, slack: float = 1.5, eps: float = 0.05,
                    abs_floor: float = 1e-3) -> dict:
    """bf16 is not batch-invariant, so judge NEW by its distance to an fp32 GOLD, relative to the naive
    bf16 REF: PASS if NEW is at most `slack` x worse than REF (max/mean error, significant swaps).
    abs_floor: errors below this are always accepted (matters when REF == GOLD, e.g. fp32 tests)."""
    ids = np.intersect1d(np.intersect1d(gold["idx"], ref["idx"]), new["idx"])
    sub = lambda d: {k: v[np.isin(d["idx"], ids)] for k, v in d.items()}
    gold, ref, new = sub(gold), sub(ref), sub(new)
    r = compare(gold, ref, atol=float("inf"), eps=eps)
    n = compare(gold, new, atol=float("inf"), eps=eps)
    ok = (n["max_abs_diff"] <= max(slack * r["max_abs_diff"], abs_floor)
          and n["mean_abs_diff"] <= max(slack * r["mean_abs_diff"], abs_floor)
          and n["significant_swaps"] <= slack * r["significant_swaps"] + 1)
    keys = [k for k in r if k != "PASS"]
    return {"ref_vs_gold": {k: r[k] for k in keys}, "new_vs_gold": {k: n[k] for k in keys}, "PASS": bool(ok)}


def format_report(res: dict, title: str = "check") -> str:
    lines = [f"== {title}: {'PASS' if res['PASS'] else 'FAIL'} =="]
    lines += [f"  {k:<22s} {v}" for k, v in res.items() if k != "PASS"]
    return "\n".join(lines)


def format_gold_report(res: dict, title: str = "check vs gold") -> str:
    lines = [f"== {title}: {'PASS' if res['PASS'] else 'FAIL'} ==",
             f"  {'metric':<22s} {'naive bf16 vs gold':>20s} {'new vs gold':>20s}"]
    for k, v in res["ref_vs_gold"].items():
        lines.append(f"  {k:<22s} {v:>20.6g} {res['new_vs_gold'][k]:>20.6g}")
    return "\n".join(lines)


# ---------------------------------------------------------------- FLOP model / MFU
PEAK_BF16_TFLOPS = {  # dense, no sparsity; more specific names first (PCIe/NVL parts run at lower clocks/power)
    "H200 NVL": 835.0, "H100 NVL": 835.0, "H100 PCIe": 756.0,
    "H200": 989.0, "H100": 989.0, "A100": 312.0, "L40S": 362.0, "A10": 125.0, "L4": 121.0, "T4": 65.0,
}


def peak_tflops(gpu_name: str) -> float | None:
    for k, v in PEAK_BF16_TFLOPS.items():
        if k in gpu_name:
            return v
    return None


def useful_flops(L: int, n1: int, n2: int, cfg) -> float:
    """FLOPs that any exact method must spend for one sequence with the tree trick
    (prefix L tokens + n1 t1-nodes + n2 pair-nodes; causal prefix attention; LM head on 1+n1+n2 rows)."""
    d, ff, nl = cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers
    hq, hkv = cfg.num_attention_heads, cfg.num_key_value_heads
    hd = getattr(cfg, "head_dim", d // hq)
    lin_per_tok = 2 * (d * hq * hd + 2 * d * hkv * hd + hq * hd * d + 3 * d * ff)
    T = L + n1 + n2
    attn_pairs = L * (L + 1) / 2 + (n1 + n2) * L + n1 + 2 * n2       # attended (q, k) pairs
    attn = 4 * hq * hd * attn_pairs                                   # QK^T + PV
    head = 2 * d * cfg.vocab_size * (1 + n1 + n2)
    return nl * (lin_per_tok * T + attn) + head
