#!/usr/bin/env bash
# Stage 2 v6 — AF2-D TWN damage-severity boundary sweep launcher.
#
# Per the EXP-RPM-AF2D-SEVERITY manifest:
#   - Site: AF2-D only (frozen)
#   - Sweep: TWN damage thresholds {0.6, 0.7, 0.8, 0.9, 1.0}
#   - Arms: damaged_base (pre-train only), t2_ternary, lora,
#           random_t2_ternary, random_lora
#   - Seeds: 1, 2, 3
#   - 5 cells/seed/threshold (4 trained + 1 base-eval)
#   - Total: 5 thresholds x 5 arms x 3 seeds = 75 cells (60 trained + 15 base)
#
# The launcher runs each threshold as a separate tournament, so we get
# 5 separate aggregate.json files (one per threshold), and the
# downstream summarizer can compute per-threshold and across-threshold
# LRN/TSP deltas.
#
# Usage:
#   ./stage2-v6-launch.sh              # run all 5 thresholds sequentially
#   ./stage2-v6-launch.sh 0.7          # run only threshold 0.7
#   ./stage2-v6-launch.sh 0.6 0.7 1.0  # run specific thresholds
#
# Total wall time: ~3.5 hours for all 5 thresholds (sequential).

set -uo pipefail

TORUS_BASE="${TORUS_BASE:-/home/andrew-jochl/TORUS}"
RUNS_DIR="$TORUS_BASE/runs/r"
LOG_DIR="$TORUS_BASE/runs/r/_logs/stage2-v6-tournaments"
mkdir -p "$LOG_DIR"

EXP_ID="EXP-RPM-AF2D-SEVERITY"
MANIFEST="$TORUS_BASE/research/residual-pareto/experiments/$EXP_ID/manifest.yaml"
if [ ! -f "$MANIFEST" ]; then
    echo "[stage2-v6] missing manifest: $MANIFEST" >&2
    exit 1
fi
TARGET_MODULE="model.layers.0.mlp.down_proj"

# All preregistered thresholds. Filter to user-provided list if any.
ALL_THRESHOLDS="0.6 0.7 0.8 0.9 1.0"
if [ "$#" -gt 0 ]; then
    THRESHOLDS="$@"
else
    THRESHOLDS="$ALL_THRESHOLDS"
fi

# Arms for the tournament. "damaged_base" is handled separately via
# --arms t2_ternary --n-steps 0 (the driver uses the t2_ternary arm as
# the eval vehicle for pre-train eval). Random arms produce empty eval
# during the tournament and require post-hoc eval via
# examples/eval_untrained_arms_v2.py after the tournament completes.
TOURNAMENT_ARMS="t2_ternary,lora,random_t2_ternary,random_lora"
BASE_EVAL_ARMS="t2_ternary"  # eval-only, --n-steps 0

run_threshold() {
    local thr="$1"
    echo "[stage2-v6] === threshold=$thr ==="
    local thr_dir="$RUNS_DIR/$EXP_ID/threshold-$thr"
    mkdir -p "$thr_dir"
    local ts
    ts=$(date -u +%Y%m%dT%H%M%SZ)
    local run_dir="$thr_dir/$ts"
    mkdir -p "$run_dir"
    local cell_log="$LOG_DIR/${EXP_ID}_threshold-${thr}_${ts}.log"

    # Tournament (4 trained arms x 3 seeds = 12 cells; random arms
    # produce empty eval during tournament).
    echo "[stage2-v6] tournament: arms=$TOURNAMENT_ARMS thr=$thr"
    (
        cd "$TORUS_BASE"
        .venv/bin/python examples/af2_storage_tournament.py \
            --model allenai/OLMo-1B-0724-hf \
            --target-module "$TARGET_MODULE" \
            --arms "$TOURNAMENT_ARMS" \
            --seeds 1,2,3 \
            --n-steps 500 \
            --batch-size 4 \
            --seq-len 128 \
            --lr 1e-3 \
            --momentum 0.9 \
            --grad-clip 1.0 \
            --tasks wikitext,arc_easy,lambada_openai \
            --ids-cache /tmp/wikitext103_train_ids.npy \
            --device cuda:0 \
            --dtype float16 \
            --eval-dtype float16 \
            --damage-ptq \
            --damage-threshold "$thr" \
            --damage-group-size 128 \
            --pre-train-eval \
            --out-dir "$run_dir" \
            2>&1
        echo "[stage2-v6] === tournament threshold=$thr COMPLETE ==="
    ) >> "$cell_log" 2>&1

    # Damaged-base pre-train eval (3 seeds, 1 cell per seed, no training).
    # Use --n-steps 0 to skip training; --arms t2_ternary is the eval
    # vehicle (the driver applies --pre-train-eval to the t2_ternary arm
    # before training, capturing the damaged base's wikitext ppl).
    echo "[stage2-v6] base-eval: arms=$BASE_EVAL_ARMS thr=$thr"
    local base_run_dir="$thr_dir/${ts}-base"
    mkdir -p "$base_run_dir"
    (
        cd "$TORUS_BASE"
        .venv/bin/python examples/af2_storage_tournament.py \
            --model allenai/OLMo-1B-0724-hf \
            --target-module "$TARGET_MODULE" \
            --arms "$BASE_EVAL_ARMS" \
            --seeds 1,2,3 \
            --n-steps 0 \
            --batch-size 4 \
            --seq-len 128 \
            --lr 1e-3 \
            --momentum 0.9 \
            --grad-clip 1.0 \
            --tasks wikitext \
            --ids-cache /tmp/wikitext103_train_ids.npy \
            --device cuda:0 \
            --dtype float16 \
            --eval-dtype float16 \
            --damage-ptq \
            --damage-threshold "$thr" \
            --damage-group-size 128 \
            --pre-train-eval \
            --out-dir "$base_run_dir" \
            2>&1
        echo "[stage2-v6] === base-eval threshold=$thr COMPLETE ==="
    ) >> "$cell_log" 2>&1

    echo "[stage2-v6] DONE threshold=$thr log=$cell_log"
}

for thr in $THRESHOLDS; do
    run_threshold "$thr"
done

echo "[stage2-v6] all thresholds done; logs: $LOG_DIR"
