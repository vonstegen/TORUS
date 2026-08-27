#!/usr/bin/env bash
# Stage 3 v1 — Random arms tournament launcher.
#
# The Stage B launcher only ran trained arms (t2_ternary, lora). This
# runs the random arms (random_t2_ternary, random_lora) at each
# calibrated cell so adapters are created for post-hoc eval.
#
# NOTE: this uses the SAME Stage B timestamp dir so the random arm
# results are colocated with their corresponding trained arms.
#
# Total: 5 cells x 2 random arms x 3 seeds = 30 cells. ~2 hours.

set -uo pipefail

TORUS_BASE="${TORUS_BASE:-/home/andrew-jochl/TORUS}"
RUNS_DIR="$TORUS_BASE/runs/r"
LOG_DIR="$TORUS_BASE/runs/r/_logs/stage3-v1-random-arms"
mkdir -p "$LOG_DIR"

# Use the same timestamp dir as Stage B
TS="20260826T145125Z"
RUN_DIR="$RUNS_DIR/EXP-RPM-DAMAGE-TYPE-001/$TS/stage_b_tournament"
TARGET_MODULE="model.layers.0.mlp.down_proj"
LOG="$LOG_DIR/random-arms_${TS}.log"

run_cell_random() {
    local cell_id="$1"
    local damage_args="$2"
    echo "[stage3-v1-random] === $cell_id ==="
    (
        cd "$TORUS_BASE"
        PYTHONPATH="$TORUS_BASE" \
        "$TORUS_BASE/.venv/bin/python" examples/af2_storage_tournament.py \
            --model allenai/OLMo-1B-0724-hf \
            --target-module "$TARGET_MODULE" \
            --arms random_t2_ternary,random_lora \
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
            $damage_args \
            --pre-train-eval \
            --out-dir "$RUN_DIR/$cell_id" \
            2>&1
    ) >> "$LOG" 2>&1
    echo "[stage3-v1-random] DONE $cell_id"
}

run_cell_random "BAND-3-TWN" "--damage-ptq --damage-threshold 0.7 --damage-group-size 128"
run_cell_random "BAND-3-Gaussian" "--damage-gaussian --damage-sigma 3.0"
run_cell_random "BAND-4-TWN" "--damage-ptq --damage-threshold 0.5 --damage-group-size 128"
run_cell_random "BAND-4-Gaussian" "--damage-gaussian --damage-sigma 5.0"
run_cell_random "BAND-1-Gaussian" "--damage-gaussian --damage-sigma 1.0"

echo "[stage3-v1-random] DONE; log: $LOG"