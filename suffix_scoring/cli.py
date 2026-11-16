"""Command line interface.

    python -m suffix_scoring.cli gen     --out data/eval.npz --n 50 --edge-cases
    python -m suffix_scoring.cli score   --data data/bench.npz --scorer flex -o pack_len=16384 --out results/x.npz
    python -m suffix_scoring.cli check   --ref results/a.npz --new results/b.npz
    python -m suffix_scoring.cli gold    --gold results/ref_fp32.npz --ref results/ref_bf16.npz --new results/x.npz
    python -m suffix_scoring.cli profile --data data/bench.npz --scorer flex --trace results/flex.json
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from . import metrics, model as M
from .data import SuffixScoringDataset, generate_dataset
from .profiling import profile_batches
from .runner import build, make_loader, parse_options, run
from .scorers import SCORERS


def _mkdir(path: str):
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)


def cmd_gen(a):
    _mkdir(a.out)
    if a.tiny:
        char_ids, low, high, start = M.tiny_char_token_ids(), 72, 512, 1
    else:
        tok = M.load_tokenizer(a.model)
        char_ids, low, high = M.char_token_ids(tok), 0, tok.vocab_size
        start = tok.convert_tokens_to_ids(M.START_TOKEN)
    generate_dataset(a.out, a.n, char_ids, low, high, start, seed=a.seed,
                     force_lengths=[0, 1, 2048] if a.edge_cases else None)
    lens = SuffixScoringDataset(a.out).lengths()
    print(f"wrote {a.out}: {len(lens)} sequences | length mean {lens.mean():.0f}, min {lens.min()}, "
          f"max {lens.max()} | at 0: {np.mean(lens == 0):.1%}, at 2048: {np.mean(lens == 2048):.1%}")


def _scorer_from_args(a, ds):
    opts = parse_options(a.opt, a.options_json)
    scorer, _ = build(a.scorer, opts, ds.start_token_id, dtype=a.dtype, device=a.device,
                      model_id=a.model, tiny=a.tiny)
    return scorer, opts


def cmd_score(a):
    ds = SuffixScoringDataset(a.data)
    scorer, opts = _scorer_from_args(a, ds)
    r = run(scorer, ds, limit=a.limit, batch_size=a.batch_size, num_workers=a.num_workers,
            warmup_batches=a.warmup, sync_phases=a.sync_phases, title=f"{a.scorer} {opts}")
    print(r.timer_report)
    if r.phase_report:
        print(r.phase_report)
    if r.padded_tokens:
        print(f"  fill rate     {r.real_tokens / r.padded_tokens:.1%} real tokens per padded token")
    if a.out:
        _mkdir(a.out)
        metrics.save_results(a.out, r.idx, r.scores, r.cand_idx)
        print(f"saved {a.out}")


def cmd_profile(a):
    ds = SuffixScoringDataset(a.data)
    scorer, _ = _scorer_from_args(a, ds)
    collate = getattr(scorer, "collate", None)
    it = iter(make_loader(ds, a.batch_size, 0, collate))
    for _ in range(a.warmup):
        scorer.score(next(it))
    if a.trace:
        _mkdir(a.trace)
    print(profile_batches(scorer.score, [next(it) for _ in range(a.batches)], a.trace, a.rows))


def cmd_check(a):
    res = metrics.compare(metrics.load_results(a.ref), metrics.load_results(a.new), atol=a.atol, eps=a.eps)
    print(metrics.format_report(res, f"{a.new} vs {a.ref}"))
    raise SystemExit(0 if res["PASS"] else 1)


def cmd_gold(a):
    res = metrics.compare_to_gold(metrics.load_results(a.gold), metrics.load_results(a.ref),
                                  metrics.load_results(a.new), slack=a.slack, eps=a.eps)
    print(metrics.format_gold_report(res, f"{a.new} (ref {a.ref}, gold {a.gold})"))
    raise SystemExit(0 if res["PASS"] else 1)


def _model_args(p):
    p.add_argument("--data", required=True)
    p.add_argument("--scorer", default="flex", choices=sorted(SCORERS))
    p.add_argument("-o", "--opt", action="append", help="scorer option key=value (repeatable)")
    p.add_argument("--options-json", default="", help="scorer options as JSON (alternative to -o)")
    p.add_argument("--dtype", default="bf16", choices=sorted(M.DTYPES))
    p.add_argument("--device", default="cuda")
    p.add_argument("--model", default=M.MODEL_ID)
    p.add_argument("--batch-size", type=int, default=128, help="sequences per scorer call")
    p.add_argument("--warmup", type=int, default=2, help="untimed warm-up batches")
    p.add_argument("--tiny", action="store_true", help="tiny random Qwen3 on CPU (tests, no download)")


def main(argv=None):
    p = argparse.ArgumentParser(prog="suffix_scoring", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen", help="generate a synthetic dataset")
    g.add_argument("--out", required=True)
    g.add_argument("--n", type=int, default=500)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--edge-cases", action="store_true", help="force lengths 0, 1, 2048 into rows 0..2")
    g.add_argument("--model", default=M.MODEL_ID)
    g.add_argument("--tiny", action="store_true")
    g.set_defaults(fn=cmd_gen)

    s = sub.add_parser("score", help="score a dataset, time it, save scores")
    _model_args(s)
    s.add_argument("--out", default="")
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--num-workers", type=int, default=0)
    s.add_argument("--sync-phases", action="store_true", help="true per-phase GPU times (diagnosis)")
    s.set_defaults(fn=cmd_score)

    pr = sub.add_parser("profile", help="torch.profiler report + Perfetto trace")
    _model_args(pr)
    pr.add_argument("--batches", type=int, default=1)
    pr.add_argument("--rows", type=int, default=25)
    pr.add_argument("--trace", default="")
    pr.set_defaults(fn=cmd_profile)

    c = sub.add_parser("check", help="compare two result files directly")
    c.add_argument("--ref", required=True)
    c.add_argument("--new", required=True)
    c.add_argument("--atol", type=float, default=0.05)
    c.add_argument("--eps", type=float, default=0.05)
    c.set_defaults(fn=cmd_check)

    k = sub.add_parser("gold", help="judge NEW vs fp32 GOLD relative to the naive bf16 REF")
    k.add_argument("--gold", required=True)
    k.add_argument("--ref", required=True)
    k.add_argument("--new", required=True)
    k.add_argument("--slack", type=float, default=1.5)
    k.add_argument("--eps", type=float, default=0.05)
    k.set_defaults(fn=cmd_gold)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
