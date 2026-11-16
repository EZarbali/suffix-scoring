#!/usr/bin/env bash
# Final evaluation with the best configuration found on the H200 NVL (bf16).
#
#   bash scripts/evaluate_h200.sh
#
# Steps: environment check -> unit tests -> datasets -> naive references (fp32 gold, bf16)
#        -> correctness of the best scorer -> offline throughput (3 repeats) -> realtime sweep.
# All outputs go to $OUT (default results/final_<gpu>_<time>); a summary is printed at the end.
#
# Environment knobs (defaults = configuration behind the reported numbers):
#   N_BENCH=2000        sequences in the throughput set
#   N_EVAL=50           sequences with an fp32 gold reference (rows 0..2 have prefix length 0, 1, 2048)
#   PACK_LEN=16384      tokens per packed row (FlexAttention)
#   BATCH=256           sequences per scorer call
#   WORKERS=8           DataLoader workers that build trees and masks on the CPU
#   CPUS=""             e.g. 0-15 to pin to 16 cores like the reported run (taskset)
#   REPEATS=3           timed repeats of the offline run
#   RT=1                run the realtime sweep (Task 2); RT=0 skips it
#   RT_BUDGETS=8192,16384   token budgets per forward
#   RT_DURATION=15      seconds of Poisson arrivals per load point
#   RT_CUDA_GRAPHS=0    1 = torch.compile mode="reduce-overhead" (lower floor, experimental with Flex)
#   TESTS=1             run the CPU unit tests first
#   TINY=0              1 = smoke-test this script on CPU with a tiny random model (no GPU, no download)
set -euo pipefail
cd "$(dirname "$0")/.."

N_BENCH=${N_BENCH:-2000}
N_EVAL=${N_EVAL:-50}
PACK_LEN=${PACK_LEN:-16384}
BATCH=${BATCH:-256}
WORKERS=${WORKERS:-8}
CPUS=${CPUS:-}
REPEATS=${REPEATS:-3}
RT=${RT:-1}
RT_BUDGETS=${RT_BUDGETS:-8192,16384}
RT_DURATION=${RT_DURATION:-15}
RT_CUDA_GRAPHS=${RT_CUDA_GRAPHS:-0}
TESTS=${TESTS:-1}
TINY=${TINY:-0}

PY="python"; command -v python >/dev/null 2>&1 || PY="python3"
PIN=""; [ -n "$CPUS" ] && PIN="taskset -c $CPUS"
T=""; DATA=data
if [ "$TINY" = "1" ]; then
  T="--tiny"; DATA=data/tiny; N_BENCH=${N_BENCH_TINY:-24}; N_EVAL=${N_EVAL_TINY:-6}
  PACK_LEN=4096; BATCH=8; WORKERS=0; REPEATS=1; RT_DURATION=2; RT_BUDGETS=4096
fi
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr ' ' '_' || true)
OUT=${OUT:-results/final_${GPU:-cpu}_$(date +%Y%m%d_%H%M%S)}
CLI="$PIN $PY -m suffix_scoring.cli"
mkdir -p "$DATA" "$OUT"
step() { echo; echo "=== [$1] $2 ==="; }

step 0/6 "environment"
{
  echo "date: $(date -Is)"; echo "host: $(hostname)"; echo "cpus visible: $(nproc)  pinned: ${CPUS:-no}"
  $PY -c "import torch, transformers; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| transformers', transformers.__version__)"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,clocks.sm,clocks.max.sm,temperature.gpu,power.draw,clocks_throttle_reasons.active --format=csv
  fi
} | tee "$OUT/env.txt"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -q -d PERFORMANCE | grep -A12 "Clocks Event Reasons\|Clocks Throttle Reasons" | grep -q "Thermal Slowdown *: Active"; then
  echo "WARNING: thermal slowdown is active on this GPU; throughput numbers will be too low." | tee -a "$OUT/env.txt"
fi

if [ "$TESTS" = "1" ]; then
  step 1/6 "unit tests (every scorer == naive, fp32, tiny model, CPU)"
  $PY -m pytest -q tests | tee "$OUT/tests.txt"
else
  step 1/6 "unit tests skipped (TESTS=0)"
fi

step 2/6 "datasets"
BENCH=$DATA/bench_${N_BENCH}.npz; EVALD=$DATA/eval_${N_EVAL}.npz
[ -f "$BENCH" ] || $CLI gen $T --out "$BENCH" --n "$N_BENCH" --seed 1
[ -f "$EVALD" ] || $CLI gen $T --out "$EVALD" --n "$N_EVAL" --seed 0 --edge-cases

step 3/6 "naive references on the eval set (fp32 gold + bf16; takes a few minutes)"
GOLD=$DATA/ref_fp32_${N_EVAL}.npz; REF=$DATA/ref_bf16_${N_EVAL}.npz
[ -f "$GOLD" ] || $CLI score $T --data "$EVALD" --scorer naive --dtype fp32 -o candidates_per_forward=50 \
    --batch-size 1 --warmup 0 --out "$GOLD"
[ -f "$REF" ]  || $CLI score $T --data "$EVALD" --scorer naive --dtype bf16 -o candidates_per_forward=100 \
    --batch-size 1 --warmup 0 --out "$REF"

step 4/6 "correctness of the best scorer (flex, bf16) against the fp32 gold"
$CLI score $T --data "$EVALD" --scorer flex -o pack_len=$PACK_LEN --batch-size "$N_EVAL" --warmup 1 \
    --out "$OUT/flex_eval.npz" > "$OUT/flex_eval.log"
set +e
$CLI gold --gold "$GOLD" --ref "$REF" --new "$OUT/flex_eval.npz" | tee "$OUT/correctness.txt"
CORRECT=${PIPESTATUS[0]}
set -e

step 5/6 "offline throughput (Task 1): flex, pack $PACK_LEN, batch $BATCH, $WORKERS workers, $REPEATS repeats"
$PIN $PY scripts/benchmark.py $T --data "$BENCH" --scorers flex --pack-len "$PACK_LEN" --batch-size "$BATCH" \
    --num-workers "$WORKERS" --fp8 none --prune false --repeats "$REPEATS" \
    --eval-data "$EVALD" --gold "$GOLD" --ref "$REF" --out-dir "$OUT/offline" | tee "$OUT/offline.log"

if [ "$RT" = "1" ]; then
  step 6/6 "realtime sweep (Task 2): budgets $RT_BUDGETS, window 0 ms, ${RT_DURATION}s per point"
  G=""; [ "$RT_CUDA_GRAPHS" = "1" ] && G="--cuda-graphs"
  $PIN $PY scripts/realtime_bench.py $T --data "$BENCH" --max-tokens "$RT_BUDGETS" --windows 0 \
      --duration "$RT_DURATION" $G --out-dir "$OUT/realtime" | tee "$OUT/realtime.log"
else
  step 6/6 "realtime sweep skipped (RT=0)"
fi

echo
echo "================ SUMMARY ($OUT) ================"
[ "$CORRECT" = "0" ] && echo "correctness: PASS (flex vs fp32 gold, relative to naive bf16)" || echo "correctness: FAIL (see $OUT/correctness.txt)"
[ -f "$OUT/offline/summary.md" ] && { echo; echo "--- offline ---"; cat "$OUT/offline/summary.md"; }
[ -f "$OUT/realtime/summary.md" ] && { echo; echo "--- realtime ---"; cat "$OUT/realtime/summary.md"; }
echo
echo "Reference (H200 NVL, 16 cores): 219.4 seq/s, MFU 40 %, 3.2 h for 2.5 M sequences;"
echo "realtime capacity ~212 req/s (16k) / ~179 req/s (8k), single-request latency 5-13 ms."
exit "$CORRECT"
