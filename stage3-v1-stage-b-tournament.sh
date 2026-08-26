#!/usr/bin/env bash
# Stage 3 v1 — Stage B (CROSS-MECHANISM COMPARISON) tournament launcher.
#
# At each calibrated (mechanism, magnitude) point, run the full
# tournament (T2 vs LoRA, T2 vs random T2, T2 vs random LoRA, plus
# damaged base) with seeds {1, 2, 3}.
#
# Calibrated cells (from Stage A calibration_table.md):
#   BAND-3-TWN:       TWN thr=0.7  (expected base ppl 430)
#   BAND-3-Gaussian:  Gaussian sigma=3.0 (expected base ppl 451)
#   BAND-4-TWN:       TWN thr=0.5  (expected base ppl 1524)
#   BAND-4-Gaussian:  Gaussian sigma=5.0 (expected base ppl 4889)
#   BAND-1-Gaussian:  Gaussian sigma=1.0 (expected base ppl 15)
#
# Per cell: 4 trained arms (damaged_base, t2_ternary, lora) + post-hoc
# random (random_t2_ternary, random_lora), 3 seeds = 12 trained-arm cells
# + 3 base-eval cells = 15 cells per calibrated point.
#
# Total: 5 cells x 15 = 75 cells. ~5 hours on Legion.
#
# Usage:
#   ./stage3-v1-stage-b-tournament.sh [BAND-3-TWN BAND-3-Gaussian ...]
#   ./stage3-v1-stage-b-tournament.sh                # all cells

set -uo pipefail

TORUS_BASE="${TORUS_BASE:-/home/andrew-jochl/TORUS}"
RUNS_DIR="$TORUS_BASE/runs/r"
LOG_DIR="$TORUS_BASE/runs/r/_logs/stage3-v1-comparison"
mkdir -p "$LOG_DIR"

EXP_ID="EXP-RPM-DAMAGE-TYPE-001"
TARGET_MODULE="model.layers.0.mlp.down_proj"
TS=$(date -u +%Y%m%dT%H%M%SZ)
RUN_DIR="$RUNS_DIR/$EXP_ID/$TS/stage_b_tournament"
mkdir -p "$RUN_DIR"
LOG="$LOG_DIR/${EXP_ID}_stage-b_${TS}.log"

ALL_CELLS="BAND-3-TWN BAND-3-Gaussian BAND-4-TWN BAND-4-Gaussian BAND-1-Gaussian"
if [ "$#" -gt 0 ]; then
    CELLS="$@"
else
    CELLS="$ALL_CELLS"
fi

run_cell() {
    local cell_id="$1"
    local mechanism="$2"
    local damage_args="$3"
    echo "[stage3-v1-stage-b] === $cell_id ($mechanism) ==="
    local cell_dir="$RUN_DIR/$cell_id"
    mkdir -p "$cell_dir"

    # Tournament (3 trained arms x 3 seeds = 9 cells; random arms are
    # post-hoc eval'd separately).
    (
        cd "$TORUS_BASE"
        PYTHONPATH="$TORUS_BASE" \
        "$TORUS_BASE/.venv/bin/python" examples/af2_storage_tournament.py \
            --model allenai/OLMo-1B-0724-hf \
            --target-module "$TARGET_MODULE" \
            --arms t2_ternary,lora \
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
            --out-dir "$cell_dir" \
            2>&1
    ) >> "$LOG" 2>&1

    # Damaged-base pre-train eval (3 seeds, 1 cell per seed).
    local base_dir="$cell_dir-base"
    mkdir -p "$base_dir"
    (
        cd "$TORUS_BASE"
        PYTHONPATH="$TORUS_BASE" \
        "$TORUS_BASE/.venv/bin/python" examples/af2_storage_tournament.py \
            --model allenai/OLMo-1B-0724-hf \
            --target-module "$TARGET_MODULE" \
            --arms t2_ternary \
            --seeds 1,2,3 \
            --n-steps 0 \
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
            --out-dir "$base_dir" \
            2>&1
    ) >> "$LOG" 2>&1

    echo "[stage3-v1-stage-b] DONE $cell_id"
}

for cell_id in $CELLS; do
    case "$cell_id" in
        BAND-3-TWN)
            run_cell "$cell_id" "TWN thr=0.7" \
                "--damage-ptq --damage-threshold 0.7 --damage-group-size 128"
            ;;
        BAND-3-Gaussian)
            run_cell "$cell_id" "Gaussian sigma=3.0" \
                "--damage-gaussian --damage-sigma 3.0"
            ;;
        BAND-4-TWN)
            run_cell "$cell_id" "TWN thr=0.5" \
                "--damage-ptq --damage-threshold 0.5 --damage-group-size 128"
            ;;
        BAND-4-Gaussian)
            run_cell "$cell_id" "Gaussian sigma=5.0" \
                "--damage-gaussian --damage-sigma 5.0"
            ;;
        BAND-1-Gaussian)
            run_cell "$cell_id" "Gaussian sigma=1.0" \
                "--damage-gaussian --damage-sigma 1.0"
            ;;
        *)
            echo "[stage3-v1-stage-b] unknown cell: $cell_id" >&2
            ;;
    esac
done

N_CELLS=$(find "$RUN_DIR" -name eval.summary.json 2>/dev/null | wc -l)
echo "[stage3-v1-stage-b] DONE; total cells: $N_CELLS"
echo "[stage3-v1-stage-b] log: $LOG"