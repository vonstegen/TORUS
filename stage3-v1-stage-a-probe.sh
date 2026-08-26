#!/usr/bin/env bash
# Stage 3 v1 — Stage A (CALIBRATION) probe launcher.
#
# Probe damage parameter values at AF2-D to map parameter -> base ppl.
# No training, no tournament. Single damage pass + single task eval per cell.
#
# Probe grid:
#   TWN:      [0.3, 0.5, 0.7, 0.9, 1.0]
#   Gaussian: [0.5, 1.0, 2.0, 3.0, 5.0]
#
# Total: 10 cells. ~10 min wall time on Legion.
#
# Usage:
#   ./stage3-v1-stage-a-probe.sh

set -uo pipefail

TORUS_BASE="${TORUS_BASE:-/home/andrew-jochl/TORUS}"
RUNS_DIR="$TORUS_BASE/runs/r"
LOG_DIR="$TORUS_BASE/runs/r/_logs/stage3-v1-calibration"
mkdir -p "$LOG_DIR"

EXP_ID="EXP-RPM-DAMAGE-TYPE-001"
TARGET_MODULE="model.layers.0.mlp.down_proj"
TS=$(date -u +%Y%m%dT%H%M%SZ)
RUN_DIR="$RUNS_DIR/$EXP_ID/$TS/stage_a_probe"
mkdir -p "$RUN_DIR"
LOG="$LOG_DIR/${EXP_ID}_stage-a_${TS}.log"

# TWN probe cells
for thr in 0.3 0.5 0.7 0.9 1.0; do
    cell_dir="$RUN_DIR/TWN/thr-$thr"
    mkdir -p "$cell_dir"
    echo "[stage3-v1-stage-a] TWN thr=$thr"
    (
        cd "$TORUS_BASE"
        PYTHONPATH="$TORUS_BASE" \
        "$TORUS_BASE/.venv/bin/python" examples/af2_storage_tournament.py \
            --model allenai/OLMo-1B-0724-hf \
            --target-module "$TARGET_MODULE" \
            --arms t2_ternary \
            --seeds 1 \
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
            --out-dir "$cell_dir" \
            2>&1
    ) >> "$LOG" 2>&1
done

# Gaussian probe cells (mutually exclusive with --damage-ptq)
for sigma in 0.5 1.0 2.0 3.0 5.0; do
    cell_dir="$RUN_DIR/Gaussian/sigma-$sigma"
    mkdir -p "$cell_dir"
    echo "[stage3-v1-stage-a] Gaussian sigma=$sigma"
    (
        cd "$TORUS_BASE"
        PYTHONPATH="$TORUS_BASE" \
        "$TORUS_BASE/.venv/bin/python" examples/af2_storage_tournament.py \
            --model allenai/OLMo-1B-0724-hf \
            --target-module "$TARGET_MODULE" \
            --arms t2_ternary \
            --seeds 1 \
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
            --damage-gaussian \
            --damage-sigma "$sigma" \
            --pre-train-eval \
            --out-dir "$cell_dir" \
            2>&1
    ) >> "$LOG" 2>&1
done

N_CELLS=$(find "$RUN_DIR" -name eval.summary.json 2>/dev/null | wc -l)
echo "[stage3-v1-stage-a] DONE; cells: $N_CELLS"
echo "[stage3-v1-stage-a] log: $LOG"