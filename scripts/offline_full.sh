#!/usr/bin/env bash
# Full offline run (Task 1): generate and score 2-3 M sequences in shards with the best configuration.
#
#   bash scripts/offline_full.sh                     # 2.5 M sequences, 10 shards of 250k
#   N_TOTAL=3000000 SHARD=300000 bash scripts/offline_full.sh
#
# Sharding keeps RAM and disk bounded (one shard ≈ 1.2 GB of prefixes + 0.4 GB of scores) and makes the
# run resumable: finished shards are skipped when the script is started again.
# Expected on one H200 NVL (16 cores): ~219 seq/s -> ~3.2 h for 2.5 M sequences.
#
# Environment knobs:
#   N_TOTAL=2500000  SHARD=250000  SEED0=1000 (shard i uses seed SEED0+i; bench/eval sets use seeds 1 and 0)
#   PACK_LEN=16384  BATCH=256  WORKERS=8  CPUS="" (e.g. 0-15)
#   DATA=data/full  OUT=results/full  KEEP_DATA=1 (0 = delete a shard's prefixes after scoring)
#   TINY=0 (1 = CPU smoke test with a tiny random model)
set -euo pipefail
cd "$(dirname "$0")/.."

N_TOTAL=${N_TOTAL:-2500000}
SHARD=${SHARD:-250000}
SEED0=${SEED0:-1000}
PACK_LEN=${PACK_LEN:-16384}
BATCH=${BATCH:-256}
WORKERS=${WORKERS:-8}
CPUS=${CPUS:-}
DATA=${DATA:-data/full}
OUT=${OUT:-results/full}
KEEP_DATA=${KEEP_DATA:-1}
TINY=${TINY:-0}

PY="python"; command -v python >/dev/null 2>&1 || PY="python3"
PIN=""; [ -n "$CPUS" ] && PIN="taskset -c $CPUS"
T=""; [ "$TINY" = "1" ] && { T="--tiny"; PACK_LEN=4096; BATCH=8; WORKERS=0; }
CLI="$PIN $PY -m suffix_scoring.cli"
mkdir -p "$DATA" "$OUT"

n_shards=$(( (N_TOTAL + SHARD - 1) / SHARD ))
echo "full run: $N_TOTAL sequences in $n_shards shards of <= $SHARD | flex pack_len=$PACK_LEN batch=$BATCH workers=$WORKERS"
start_wall=$(date +%s)
for ((i = 0; i < n_shards; i++)); do
  n=$(( N_TOTAL - i * SHARD )); [ "$n" -gt "$SHARD" ] && n=$SHARD
  tag=$(printf "%03d" "$i")
  data="$DATA/shard_${tag}.npz"; scores="$OUT/scores_${tag}.npz"; log="$OUT/log_${tag}.txt"
  if [ -f "$scores" ] && grep -q "TOTAL" "$log" 2>/dev/null; then
    echo "[shard $tag] done, skipping"; continue
  fi
  [ -f "$data" ] || $CLI gen $T --out "$data" --n "$n" --seed $(( SEED0 + i ))
  echo "[shard $tag] scoring $n sequences"
  $CLI score $T --data "$data" --scorer flex -o pack_len=$PACK_LEN --batch-size "$BATCH" \
      --num-workers "$WORKERS" --warmup 2 --out "$scores" 2>&1 | tee "$log"
  [ "$KEEP_DATA" = "0" ] && rm -f "$data"
done
wall=$(( $(date +%s) - start_wall ))

# aggregate: scoring time per shard from the "TOTAL ... s -> ... seq/s" line (excludes model load + compile)
$PY - "$OUT" "$N_TOTAL" "$wall" <<'EOF'
import glob, re, sys
out, n_total, wall = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
secs = n = 0
for log in sorted(glob.glob(f"{out}/log_*.txt")):
    m = re.search(r"TOTAL\s+([\d.]+) s\s+->\s+([\d.]+) seq/s", open(log).read())
    if m:
        s, r = float(m.group(1)), float(m.group(2))
        secs += s; n += round(s * r)
print(f"\nscored {n:,} sequences in {secs / 3600:.2f} h of scoring time -> {n / max(secs, 1e-9):.1f} seq/s")
print(f"wall time of this invocation (incl. data generation, model load, compilation): {wall / 3600:.2f} h")
print(f"scores: {out}/scores_*.npz (keys: idx, scores [N, 400] log-probs, cand_idx [N, 400]); rank = argsort(-scores)")
EOF
