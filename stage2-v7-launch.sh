#!/usr/bin/env bash
# Stage 2 v7 — AF2-D TWN band-boundary confirmation launcher.
#
# Per the EXP-RPM-AF2D-CONFIRM-V7 manifest:
#   - Same site/recipe/tasks as v6 (frozen)
#   - Threshold subset: {0.6, 0.8, 1.0} (lower boundary, interior, upper boundary)
#   - Fresh seeds: {4, 5, 6} (not in v6's {1, 2, 3})
#   - 5 arms per threshold (damaged_base, t2_ternary, lora, random_t2,
#     random_lora)
#   - 63 cells total (3 thresholds × 5 arms × 3 seeds minus duplicates)
#
# Usage:
#   ./stage2-v7-launch.sh              # run all 3 thresholds sequentially
#   ./stage2-v7-launch.sh 0.6          # run only threshold 0.6
#
# Total wall time: ~2 hours for all 3 thresholds (sequential).

set -uo pipefail

TORUS_BASE="${TORUS_BASE:-/home/andrew-jochl/TORUS}"
RUNS_DIR="$TORUS_BASE/runs/r"
LOG_DIR="$TORUS_BASE/runs/r/_logs/stage2-v7-confirmation"
mkdir -p "$LOG_DIR"

EXP_ID="EXP-RPM-AF2D-CONFIRM-V7"
MANIFEST="$TORUS_BASE/research/residual-pareto/experiments/$EXP_ID/manifest.yaml"
if [ ! -f "$MANIFEST" ]; then
    echo "[stage2-v7] missing manifest: $MANIFEST" >&2
    exit 1
fi
TARGET_MODULE="model.layers.0.mlp.down_proj"

# Preregistered confirmation thresholds.
ALL_THRESHOLDS="0.6 0.8 1.0"
if [ "$#" -gt 0 ]; then
    THRESHOLDS="$@"
else
    THRESHOLDS="$ALL_THRESHOLDS"
fi

# Fresh seeds (NOT in v6's {1, 2, 3}).
SEEDS="4,5,6"

# Arms (same as v6).
TOURNAMENT_ARMS="t2_ternary,lora,random_t2_ternary,random_lora"
BASE_EVAL_ARMS="t2_ternary"

run_threshold() {
    local thr="$1"
    echo "[stage2-v7] === threshold=$thr (fresh seeds $SEEDS) ==="
    local thr_dir="$RUNS_DIR/$EXP_ID/threshold-$thr"
    mkdir -p "$thr_dir"
    local ts
    ts=$(date -u +%Y%m%dT%H%M%SZ)
    local run_dir="$thr_dir/$ts"
    mkdir -p "$run_dir"
    local cell_log="$LOG_DIR/${EXP_ID}_threshold-${thr}_${ts}.log"

    # Tournament (4 arms x 3 fresh seeds = 12 cells; random arms
    # produce empty eval during tournament).
    echo "[stage2-v7] tournament: arms=$TOURNAMENT_ARMS thr=$thr"
    (
        cd "$TORUS_BASE"
        .venv/bin/python examples/af2_storage_tournament.py \
            --model allenai/OLMo-1B-0724-hf \
            --target-module "$TARGET_MODULE" \
            --arms "$TOURNAMENT_ARMS" \
            --seeds "$SEEDS" \
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
        echo "[stage2-v7] === tournament threshold=$thr COMPLETE ==="
    ) >> "$cell_log" 2>&1

    # Damaged-base pre-train eval (3 fresh seeds, 1 cell per seed).
    echo "[stage2-v7] base-eval: arms=$BASE_EVAL_ARMS thr=$thr"
    local base_run_dir="$thr_dir/${ts}-base"
    mkdir -p "$base_run_dir"
    (
        cd "$TORUS_BASE"
        .venv/bin/python examples/af2_storage_tournament.py \
            --model allenai/OLMo-1B-0724-hf \
            --target-module "$TARGET_MODULE" \
            --arms "$BASE_EVAL_ARMS" \
            --seeds "$SEEDS" \
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
        echo "[stage2-v7] === base-eval threshold=$thr COMPLETE ==="
    ) >> "$cell_log" 2>&1

    echo "[stage2-v7] DONE threshold=$thr log=$cell_log"
}

for thr in $THRESHOLDS; do
    run_threshold "$thr"
done

echo "[stage2-v7] all thresholds done; logs: $LOG_DIR"