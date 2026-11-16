# Suffix Scoring Throughput — Qwen3-0.6B

Score 400 three-token candidate suffixes `[a-z][A-Z][0-9]` against streams of token prefixes with
Qwen3-0.6B, as fast as possible, and prove that the optimised path computes the same rankings as the
obvious slow solution.

```
score(suffix | prefix) = log P(t1 | prefix) + log P(t2 | prefix, t1) + log P(t3 | prefix, t1, t2)
```

## Contents
- [Evaluate on an H200 (one command)](#evaluate-on-an-h200-one-command)
- [Full offline run: 2–3 M sequences](#full-offline-run-23-m-sequences)
- [Online serving: recommended configurations](#online-serving-recommended-configurations)
- [Results](#results)
- [Approach](#approach)
- [Correctness](#correctness)
- [Repository layout](#repository-layout)
- [Quick start](#quick-start)
- [Benchmarking on H200 / A100](#benchmarking-on-h200--a100)
- [Realtime serving (Task 2)](#realtime-serving-task-2)
- [Profiling](#profiling)
- [Figures](#figures)
- [Assumptions](#assumptions)

---

## Evaluate on an H200 (one command)

```bash
pip install -r requirements.txt          # tested: torch 2.14.1+cu130, transformers 5.22.1
bash scripts/evaluate_h200.sh            # ~30-45 min on one H200, most of it the naive references and the realtime sweep
CPUS=0-15 bash scripts/evaluate_h200.sh  # pinned to 16 cores, as for the reported numbers
```

The script uses the best configuration found on the H200 NVL and runs, in order:

| Step | What | Configuration |
|---|---|---|
| 0 | environment: versions, GPU clocks, warning if thermal slowdown is active | |
| 1 | unit tests: every scorer equals the naive reference in fp32 (tiny model, CPU) | |
| 2 | datasets: 2,000 sequences for throughput, 50 for correctness (incl. prefix lengths 0, 1, 2048) | seeds 1 and 0 |
| 3 | naive references on the 50: fp32 gold and bf16 | Hugging Face, unmodified |
| 4 | correctness of the best scorer vs fp32 gold, judged against the naive bf16 noise floor | exit code 0 = pass |
| 5 | offline throughput (Task 1), 3 repeats, with MFU, memory and correctness per run | `flex`, bf16, `pack_len=16384`, `batch_size=256`, 8 workers, FP8 off |
| 6 | realtime sweep (Task 2): unloaded floor, capacity, 10–115 % load | budgets 8k and 16k, window 0 ms, 8 size buckets |

Results land in `results/final_<gpu>_<time>/` (`offline/summary.md`, `realtime/summary.md`, `correctness.txt`).
Knobs (environment variables): `N_BENCH`, `N_EVAL`, `PACK_LEN`, `BATCH`, `WORKERS`, `CPUS`, `REPEATS`, `RT=0` to skip
Task 2, `RT_BUDGETS`, `RT_DURATION`, `RT_CUDA_GRAPHS=1`, `TESTS=0`. `TINY=1` smoke-tests the whole script on a CPU
with a tiny random model.

Expected on an H200 NVL: **~219 seq/s** (±1 %), MFU ~40 %, correctness PASS; realtime capacity ~212 req/s (16k)
and ~179 req/s (8k), single-request latency 5–13 ms. An H200 SXM has a higher power limit and should be somewhat
faster. Check `env.txt` first: a throttled GPU (we saw one at 345 MHz) gives far lower numbers.

## Full offline run: 2–3 M sequences

```bash
bash scripts/offline_full.sh                                   # 2.5 M sequences, 10 shards of 250k
N_TOTAL=3000000 SHARD=300000 bash scripts/offline_full.sh      # 3 M
CPUS=0-15 KEEP_DATA=0 bash scripts/offline_full.sh             # pinned, delete each shard's prefixes after scoring
```

- **Generation:** each shard is a synthetic dataset like the benchmark set (prefix length round(Normal(1024, 600))
  clipped to [0, 2048], uniform random prefix tokens, 400 distinct candidates out of 6,760), written by
  `python -m suffix_scoring.cli gen --out data/full/shard_XXX.npz --n 250000 --seed <1000 + XXX>`. Seeds differ from
  the benchmark (1) and eval (0) sets. Generating 250k sequences takes a few seconds.
- **Scoring:** the same best configuration as above (`flex`, `pack_len=16384`, batch 256, 8 workers).
- **Why shards:** one shard is ~1.2 GB of prefixes and ~0.4 GB of scores, so RAM stays small and the run is
  resumable: finished shards are skipped when the script is started again.
- **Disk:** ~16 GB for 2.5 M sequences with `KEEP_DATA=1`, ~4 GB with `KEEP_DATA=0`.
- **Output:** `results/full/scores_XXX.npz` with `idx` (row in the shard), `scores` [N, 400] (summed log-probs)
  and `cand_idx` [N, 400] (suffix ids); ranking = `argsort(-scores)`. The script ends with total sequences,
  scoring hours and seq/s (model load and compilation, ~20 s per shard, are reported separately as wall time).
- **Expected:** ~3.2 h of scoring for 2.5 M sequences on one H200 NVL (~3.8 h for 3 M).

## Online serving: recommended configurations

Every request is one forward (no decode). The server packs whatever is queued into one row, up to a **token budget**,
and pads it to the smallest of 8 **size buckets** (640 … 16,384 tokens). Measured on the H200 NVL
(`results_h200/realtime_*`), the configurations to use are:

| Goal | Budget (`--max-tokens`) | Window | Result |
|---|---|---|---|
| Lowest latency at low / moderate load (≤ ~100 req/s) | 8,192 | 0 ms | p50 10–18 ms up to ~95 req/s (p99 33–56 ms at 50–95 req/s) |
| **Most requests under a 100 ms target** (default choice) | **8,192** | **0 ms** | goodput 154 req/s; capacity ~179 req/s |
| Maximum throughput (bulk traffic, latency secondary) | 16,384 | 0 ms | capacity ~212 req/s, p50 76 ms at 180 req/s |

- **Window 0:** a batching window never helped. A forward takes 8–55 ms, longer than any gap a window could bridge,
  and under load the queue fills batches by itself.
- **Keep load below ~85 % of capacity.** Above that the queue grows without bound and p99 explodes (Little's law).
  Scale out with replicas, one per GPU, rather than running one GPU near saturation.
- **Buckets matter more than launch overhead:** a single request is padded only to the next bucket; the old
  4-bucket design (smallest 2,560) doubled latency for short prefixes.
- **CUDA graphs (`--cuda-graphs`)** cut the single-request floor by ~0.8 ms on the H200 (16k: 4.5 / 7.6 / 11.9 ms);
  they are experimental with FlexAttention, so they are off by default.

```bash
# reproduce one operating point, e.g. the default choice at ~120 req/s
python scripts/realtime_bench.py --data data/bench_2000.npz --max-tokens 8192 --windows 0 --rates 120 --duration 30
```

In code: `Server(scorer, start_token_id, ServerConfig(window_ms=0, max_tokens=8192))` in
`suffix_scoring/realtime.py`; `serve()` reads requests from a queue and fills in their scores.

## Results

Final numbers on **one NVIDIA H200 NVL** (141 GB, 600 W PCIe card), limited to **16 CPU cores** with `taskset`
as in the spec, torch 2.14.1+cu130, transformers 5.22.1, bf16. Prefix length ~ round(Normal(1024, 600)) clipped
to [0, 2048], 400 candidates per prefix, 2,000-sequence benchmark set, 3 repeats. Raw results: `results_h200/`.

### Task 1 · offline throughput

| Step | Scorer | H200 NVL seq/s | × naive | 2.5 M sequences | A100 40 GB seq/s |
|---|---|---:|---:|---:|---:|
| 0 | naive HF reference (`naive`) | 0.43 | 1× | ~67 days | 0.20 |
| 1 | prefix once + KV cache reuse (`kvcache`) | 2.15 | 5× | ~13 days | 1.53 |
| 2 | prefix tree, one forward per sequence (`tree`) | 47.4 ¹ | 110× | ~15 h | 25.5 |
| 3 | + several trees per forward (dense mask) + `torch.compile` (`tree_batched`) | 170.5 | 397× | ~4.1 h | 60.6 |
| 4 | + packing, FlexAttention block-sparse mask, fused LM head (`flex`) | **219.4** | **~510×** | **~3.2 h** | 93.3 |

- **MFU 40 %** of the H200 NVL's dense bf16 peak (835 TFLOPS); 46 % on the A100. All rows pass the accuracy gate.
- Remaining GPU time (H200 profile, `results_h200/profile_flex_h200.txt`): GEMMs ~47 %, FlexAttention ~26 %,
  fused norms/elementwise ~16 %, LM head + logsumexp ~5 %.
- **Tokens per forward saturate at 16k:** 8k → 206.8, 16k → 218.2, 32k → 215.1 seq/s (larger packs only add
  tail padding). More GPU memory does not buy throughput.
- **FP8 (bonus) rejected:** W8A8 e4m3 on Hopper gave +3–7 % (`mlp` 224, `mlp_gate_up` 229, `all` 236 seq/s at
  16k) but **failed the accuracy gate by a wide margin** (max |Δ| to gold 4.4–5.5 vs 0.76 for bf16; 12–26× more
  significant rank swaps). The small GEMMs (hidden 1,024) gain little, and per-token activation quantisation
  costs extra kernels. Headline stays bf16.
- **Last-layer pruning not reported:** its repeats varied by ±35–60 % because the variable number of selected
  rows triggered recompilation during timing. Expected gain was ≤ 3 %; left as future work.

¹ `tree` is launch-bound (one sequence per forward) and was timed on only 20 sequences: ±31 % across repeats.

### Task 2 · real-time serving (H200 NVL, token budget 16k, no batching window)

| Offered load | p50 | p99 | mean batch |
|---|---:|---:|---:|
| 10 % (~19 req/s) | 9 ms | 21 ms | 1.0 |
| 25 % (~55 req/s) | 11 ms | 36 ms | 1.2 |
| 50 % (~108 req/s) | 22 ms | 342 ms ² | 1.8 |
| 70 % (~150 req/s) | 52 ms | 106 ms | 3.8 |
| 85 % (~179 req/s) | 76 ms | 142 ms | 6.8 |
| capacity | ~212 req/s | | ~11 |

Single-request latency on an idle server: **5.3 / 8.4 / 12.8 ms** for prefixes of 256 / 1,024 / 2,048 tokens
(CUDA graphs: 4.5 / 7.6 / 11.9 ms). p50 stays under 100 ms up to ~180 req/s (85 % of capacity).

² One isolated ~0.4 s server stall (37 consecutive requests; GPU service times normal, so the pause is on the
host side) dominates this p99; without it p99 is 93 ms. Similar one-off stalls hit 2 of 14 other points.

## Approach

**1. Exploit the structure of the suffixes (the big win).**
`log P(t2 | p, t1)` depends only on `t1` (≤ 26 distinct), `log P(t3 | p, t1, t2)` only on `(t1, t2)`
(~310 distinct out of 676 for 400 random candidates), and `t3` never has to go through the model.
One forward over `[prefix | unique t1 nodes | unique (t1, t2) nodes]` with a tree attention mask
replaces 400 forwards of the full prefix: ~1.36 k instead of ~410 k tokens per sequence.

**2. Remove memory traffic, not FLOPs.** Profiling the KV-cache variant showed > 90 % of GPU time in
cache copies, `torch.cat` cache appends and GQA `repeat_kv` copies. The tree forward has one prefix KV
and no copies.

**3. Saturate the GPU.** One sequence (hidden size 1024) is launch-bound. Packing many sequences per
forward, `torch.compile` (fuses RMSNorm / RoPE / SiLU / casts) and FlexAttention with a block-sparse
mask (skips empty 128×128 tiles: upper causal triangle, node×node block, padding) move the bottleneck
to the GEMMs.

**4. Only compute the logits that are needed.** The LM head runs on 1 + n1 + n2 ≈ 340 rows per sequence.
The full-vocabulary normaliser is required (the context differs for t2 / t3), so
`log P(tok) = logit[tok] − logsumexp(row)` with fp32 accumulation, fused in one compiled graph.

**Options of the best scorer (`flex`)**

| option | default | effect |
|---|---|---|
| `pack_len` | 16384 | tokens per forward (static shape) |
| `compile` | true | `torch.compile` layer stack + FlexAttention |
| `fused_head` | true | norm + LM head + logsumexp in one graph |
| `prune_last_layer` | false | last layer: Q / attention / o_proj / MLP only for rows that need logits (~3 %) |
| `fp8` | none | **bonus** W8A8 FP8: `mlp_gate_up`, `mlp`, `all` (real on sm90+, simulated elsewhere) |

## Correctness

bf16 is **not batch-invariant**: the naive bf16 reference itself changes when only the batching
changes (different kernels, different reduction order). Bit-exact equality is therefore not a usable
criterion. Instead:

1. **Gold** = naive reference in fp32.
2. Measure how far the **naive bf16** reference is from gold (the bf16 noise floor).
3. An optimised scorer passes if it is not further from gold than the naive bf16 reference
   (max / mean |Δscore|, significant rank swaps; slack 1.5×). Reported alongside: Kendall τ,
   top-1/10/50 overlap.

In fp32 every scorer matches the naive reference to ~1e-6 (`tests/`). In bf16 the optimised scorers
are as close or closer to gold than naive bf16 (fused kernels round less often).

## Repository layout

```
suffix_scoring/
  model.py        model / tokenizer loading, bare-character token ids
  data.py         synthetic workload generation and dataset
  tree.py         prefix tree, dense batching, packing, attention masks
  qwen3.py        explicit Qwen3 forward (dense mask / packed FlexAttention, last-layer pruning)
  fp8.py          FP8 W8A8 linears (per-channel weights, per-token activations)
  metrics.py      score / ranking comparison, fp32-gold criterion, FLOP model (MFU)
  profiling.py    timers, per-phase timing, torch.profiler + Perfetto traces
  runner.py       build scorers, run datasets through DataLoader (+ workers)
  cli.py          gen | score | profile | check | gold
  realtime.py     open-loop load generator, dynamic batcher with size buckets (Task 2)
  scorers/        naive, kvcache, tree, tree_batched, tree_pipelined, flex
scripts/
  benchmark.py    grid sweep -> CSV / JSON / Markdown (+ correctness per config)
  realtime_bench.py  latency / throughput sweep for Task 2
  plots.py        presentation figures (PNG + SVG)
  evaluate_h200.sh   final evaluation with the best H200 configuration (correctness, Task 1, Task 2)
  offline_full.sh    generate and score 2-3 M sequences in resumable shards
  run_all.sh      datasets -> references -> ablation -> sweep -> figures (full development grid)
docs/
  ENGINEERING_LOG_DE.md    step-by-step log: what, why (profiling evidence), result
  measurements_a100.json   numbers recorded during development (A100)
results_h200/     final H200 NVL results: summaries, raw JSON, profile, versions
figures/h200/     final figures; figures/a100/: development figures
tests/            every scorer == naive reference (fp32, tiny model, CPU)
```

## Quick start

```bash
pip install -r requirements.txt

# data: throughput set + small correctness set with edge cases (lengths 0, 1, 2048)
python -m suffix_scoring.cli gen --out data/bench_2000.npz --n 2000 --seed 1
python -m suffix_scoring.cli gen --out data/eval_50.npz --n 50 --seed 0 --edge-cases

# references (naive): fp32 gold and bf16
python -m suffix_scoring.cli score --data data/eval_50.npz --scorer naive --dtype fp32 \
    -o candidates_per_forward=50 --batch-size 1 --warmup 0 --out results/ref_fp32.npz
python -m suffix_scoring.cli score --data data/eval_50.npz --scorer naive \
    --batch-size 1 --warmup 0 --out results/ref_bf16.npz

# best scorer: throughput, then correctness
python -m suffix_scoring.cli score --data data/bench_2000.npz --scorer flex -o pack_len=16384 \
    --batch-size 256 --num-workers 8
python -m suffix_scoring.cli score --data data/eval_50.npz --scorer flex --batch-size 50 --out results/flex.npz
python -m suffix_scoring.cli gold --gold results/ref_fp32.npz --ref results/ref_bf16.npz --new results/flex.npz

# tests (CPU, no download)
pytest -q tests
```

## Benchmarking on H200 / A100

One command reproduces everything (datasets, references, ablation, sweep):

```bash
bash scripts/run_all.sh
# knobs: N_BENCH=2000 N_EVAL=50 WORKERS=auto PACKS=8192,16384,32768 BATCH=256 FP8=none,mlp_gate_up,mlp,all
```

Or a custom sweep:

```bash
python scripts/benchmark.py --data data/bench_2000.npz --scorers flex \
    --pack-len 8192,16384,32768 --batch-size 256 --num-workers 8 \
    --fp8 none,mlp_gate_up,mlp --prune false,true --repeats 3 \
    --eval-data data/eval_50.npz --gold results/ref_fp32.npz --ref results/ref_bf16.npz
```

Per configuration: mean ± std seq/s over repeats, ms/seq, MFU (useful FLOPs / peak bf16), peak memory,
padding fill rate, projected hours for 2.5 M sequences, FP8 mode, and correctness vs gold.
Compilation happens in the warm-up and is reported separately (`warmup_s`).

Measurement hygiene on rented GPUs: check for thermal throttling before trusting numbers
(`nvidia-smi -q -d PERFORMANCE,CLOCK,TEMPERATURE`). A first H200 NVL host ran at 89 °C with *SW Thermal
Slowdown* active and the SM clock at 345 MHz instead of 1,785 MHz; it was discarded before any measurement.
The final host held 1,785 MHz at ~600 W and 62–67 °C and reached 624 TFLOPS on a bf16 8k×8k matmul.

GPU handling: FP8 configurations run as real FP8 GEMMs on sm90+ (H100 / H200). On other GPUs they are
skipped, or run as simulated quantisation with `--allow-fake-fp8` (accuracy only, no speed-up).
FP8 is a bonus; headline numbers are bf16.

## Realtime serving (Task 2)

Each request carries one prefix and its 400 candidates and expects the ranking back. The server is the
same `flex` scorer behind a **dynamic batcher**:

- **Window `W`:** after the first request arrives the batcher waits up to `W` ms for more, or stops
  early when the **token budget** (`--max-tokens`) is full. `W = 0` means "take whatever is queued now".
- **Size buckets:** a batch is packed into the smallest pack length that fits (`--buckets`, default
  640 … 16384), so `torch.compile` sees only a few static shapes and a single short request does not pay
  for a large padded pack. Optional **CUDA graphs** (`--cuda-graphs`,
  `mode="reduce-overhead"`) remove launch overhead for small batches. They are experimental with
  FlexAttention; if they fail, run without them.
- **Load generator:** open-loop Poisson arrivals. Latency is measured from the *scheduled* arrival time,
  so a slow server cannot hide its queueing delay (no coordinated omission).

```bash
python scripts/realtime_bench.py --data data/bench_2000.npz --windows 0,2,5,10 \
    --max-tokens 16384 --duration 15 [--cuda-graphs]
python scripts/plots.py --realtime results/realtime_<gpu>_<time> --out figures/h200
```

Per budget the script first measures the **unloaded latency floor** (single requests with prefix
length 256 / 1024 / 2048), then the **capacity** (back-to-back full batches), then sweeps
windows × offered rate (10 % … 115 % of capacity). Per point it reports p50 / p95 / p99 latency split
into queue wait and GPU service time, achieved requests/s, mean batch size and goodput under
50 / 100 / 200 ms SLOs.

**A100 40 GB development numbers** (budget 16384, W = 0, 8 buckets; capacity ≈ 89 req/s, offline 93 seq/s):

| Offered load | p50 | p99 | mean batch |
|---|---:|---:|---:|
| 10 % | 23 ms | 54 ms | 1.0 |
| 25 % | 22 ms | 67 ms | 1.1 |
| 50 % | 38 ms | 115 ms | 1.4 |
| 70 % | 96 ms | 236 ms | 3.0 |
| 85 % | 165 ms | 266 ms | 5.8 |
| 100 % | 300 ms | 578 ms | 9.6 |

Findings: the unloaded floor was *not* launch overhead. CUDA graphs changed nothing; the floor matched the
cost of the smallest padded bucket (2560 tokens ≈ 20 ms at offline token throughput). Finer buckets cut the
floor for a 256-token prefix from 22 to 11 ms. A batching window (2–5 ms) never helped: service time
(~20 ms) exceeds the gap it could bridge, and under load the queue fills batches by itself.

**H200 NVL** (`results_h200/realtime_*`): capacity ~212 req/s at budget 16k and ~179 req/s at 8k; the best
goodput under a 100 ms target is 154 req/s (8k) vs 148 req/s (16k), so the smaller budget again wins in the middle
of the curve and the larger one at the top. CUDA graphs, useless on the A100, now cut the floor by ~0.8 ms
(~10–15 %): with ~2× faster GPU work the fixed CPU launch cost becomes a visible share.

**Mechanism of the trade-off.** At low load each request is served almost alone: latency ≈ floor, but the
GPU is poorly used (~1 400 tokens per forward). As load rises, requests queue during the previous batch,
batches grow automatically and throughput approaches the offline number, while latency grows with
wait + service time of a bigger batch. Near capacity the queue grows without bound (Little's law), so p99
explodes. A larger `W` buys bigger batches at low load at the price of up to `W` ms extra latency; above
moderate load it barely matters because the queue fills batches by itself. TTFT/TPOT do not apply: there
is no decode, every request is one prefill-like forward.

## Profiling

```bash
# true per-phase GPU time
python -m suffix_scoring.cli score --data data/bench_2000.npz --scorer flex --sync-phases
# torch.profiler table + Perfetto trace (open in https://ui.perfetto.dev)
python -m suffix_scoring.cli profile --data data/bench_2000.npz --scorer flex --batch-size 32 \
    --trace results/flex_trace.json
```

Note: without synchronisation, phase timings only measure kernel launches and attribute GPU waiting
time to whichever phase first needs a result.

## Figures

`scripts/plots.py` renders presentation figures (PNG at 2× and SVG) from benchmark results and/or
recorded measurements (`docs/measurements_a100.json`). Pre-rendered figures: `figures/h200/` (final runs) and `figures/a100/` (development).

```bash
python scripts/plots.py --ablation results/ablation/results.json --sweep results/sweep_flex/results.json \
    --measurements docs/measurements_a100.json --data data/bench_2000.npz --out figures/h200
```

| Figure | Shows |
|---|---|
| `01_length_distribution` | prefix lengths, clipping spikes at 0 and 2048 |
| `02_tokens_per_sequence` | work per sequence: naive vs prefix-once vs tree (~300× less) |
| `03_ablation_throughput` | seq/s per optimisation step, speed-up and time for 2.5 M sequences |
| `04_gpu_time_breakdown` | GPU time: dense mask vs FlexAttention (attention / GEMM / logsumexp / other) |
| `05_kvcache_time_share` | KV-cache variant: most GPU time is memory copies |
| `06_correctness` | distance to fp32 gold vs the naive bf16 noise floor |
| `07_sweep_pack_len` | throughput vs tokens per forward per FP8 / pruning variant |
| `08_mfu` | model FLOPs utilisation per configuration |
| `09_realtime_latency` | p50 / p99 latency vs achieved throughput per batching window, capacity line |
| `10_realtime_batch_size` | mean batch size vs offered load: how batching grows with the queue |
| `11_realtime_floor` | unloaded single-request latency by prefix length |

## Assumptions

Tokenisation, bf16 with fp32 accumulation inside kernels, the empty-prefix convention, independent
requests for Task 2 and the noise-floor accuracy criterion were confirmed with the team.


- **Length 0** (~4.4 % of prefixes after clipping) has no position that predicts t1; such prefixes
  are conditioned on a single `<|endoftext|>` token. Another ~4.4 % sit at 2048 and dominate attention cost.
- **Tokenisation:** bare characters without a leading space (`"a"` → id 64, `"A"` → 32, `"0"` → 15);
  each is verified to be a single token. `" a"` would be a different token.
- Prefix tokens are drawn uniformly from the ordinary vocabulary `[0, tokenizer.vocab_size)`, no special tokens.
- 400 distinct candidates per prefix, uniformly from the 6 760 suffixes, unknown in advance.
- Throughput excludes data generation and compilation (warm-up); CUDA is synchronised around timed sections.
- Ties in the ranking are broken by suffix id.
- **Realtime:** every request carries its prefix and all 400 candidates together (the spec says each
  request "carries a token sequence and its set of 400 candidate suffixes"). Prefixes are random, so there
  is no cross-request prefix sharing and no persistent KV cache between requests.
