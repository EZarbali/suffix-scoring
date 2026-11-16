#!/usr/bin/env bash
# End-to-end reproduction on a single GPU (H200 / H100 / A100):
#   1) datasets  2) naive references (fp32 gold + bf16)  3) ablation  4) sweep of the best scorer
# Usage:  bash scripts/run_all.sh            (env: N_BENCH, N_EVAL, WORKERS, PACKS, FP8, BATCH)
set -euo pipefail
cd "$(dirname "$0")/.."

N_BENCH=${N_BENCH:-2000}       # sequences in the throughput dataset
N_EVAL=${N_EVAL:-50}           # sequences with fp32 gold for correctness
WORKERS=${WORKERS:-auto}
PACKS=${PACKS:-8192,16384,32768}
BATCH=${BATCH:-256}
FP8=${FP8:-none,mlp_gate_up,mlp,all}
CLI="python -m suffix_scoring.cli"

mkdir -p data results
[ -f data/bench_${N_BENCH}.npz ] || $CLI gen --out data/bench_${N_BENCH}.npz --n ${N_BENCH} --seed 1
[ -f data/eval_${N_EVAL}.npz ]   || $CLI gen --out data/eval_${N_EVAL}.npz --n ${N_EVAL} --seed 0 --edge-cases

# references on the eval set (naive is slow: ~minutes for 50 sequences)
[ -f results/ref_fp32.npz ] || $CLI score --data data/eval_${N_EVAL}.npz --scorer naive --dtype fp32 \
    -o candidates_per_forward=50 --batch-size 1 --warmup 0 --out results/ref_fp32.npz
[ -f results/ref_bf16.npz ] || $CLI score --data data/eval_${N_EVAL}.npz --scorer naive --dtype bf16 \
    -o candidates_per_forward=100 --batch-size 1 --warmup 0 --out results/ref_bf16.npz

EVAL="--eval-data data/eval_${N_EVAL}.npz --gold results/ref_fp32.npz --ref results/ref_bf16.npz"

# ablation: every optimisation step once
python scripts/benchmark.py --data data/bench_${N_BENCH}.npz \
    --scorers naive,kvcache,tree,tree_batched,tree_pipelined,flex \
    --pack-len 16384 --batch-size ${BATCH} --num-workers ${WORKERS} $EVAL --out-dir results/ablation

# sweep of the best scorer: budget x FP8 x last-layer pruning
python scripts/benchmark.py --data data/bench_${N_BENCH}.npz --scorers flex \
    --pack-len ${PACKS} --batch-size ${BATCH} --num-workers ${WORKERS} \
    --fp8 ${FP8} --prune false,true $EVAL --out-dir results/sweep_flex

# Task 2: realtime latency/throughput sweep
python scripts/realtime_bench.py --data data/bench_${N_BENCH}.npz --windows 0,2,5,10 \
    --max-tokens 16384 --duration ${RT_DURATION:-15} --out-dir results/realtime

# presentation figures (PNG @2x + SVG)
python scripts/plots.py --ablation results/ablation/results.json --sweep results/sweep_flex/results.json \
    --measurements docs/measurements_a100.json --data data/bench_${N_BENCH}.npz --realtime results/realtime --out figures/$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')
