#!/usr/bin/env bash
# Stage 2 v5 TWN tournament launcher (run after Stage 2 v4 verdict).
#
# Runs the EXP-RPM-L15-TWN-V5 experiment: TWN damage with group_size=8
# at model.layers.15.mlp.down_proj. Per the 2-stage plan in the manifest:
#   Stage 1 CAL pilot (5 thresholds x 3 seeds x 1 arm = 15 cells)
#   Stage 2 tournament (9 arms x 3 seeds = 27 cells) at the qualifying
#     threshold selected from Stage 1.
#
# Usage:
#   ./stage2-v5-launch.sh cal        # run Stage 1 CAL pilot only
#   ./stage2-v5-launch.sh tournament # run Stage 2 tournament at the
#                                    # threshold selected from Stage 1
#   ./stage2-v5-launch.sh both       # run both sequentially
#
# Stage 1 takes ~3-5 min (no training, eval only). Stage 2 takes ~42 min
# (9 arms x 3 seeds at 500 training steps each, identical to Stage 2 v2/v3/v4).
#
# The manifest is the single source of truth for the preregistered knobs.
# This launcher reads the threshold and group_size from the manifest.

set -uo pipefail

TORUS_BASE="${TORUS_BASE:-/home/andrew-jochl/TORUS}"
RUNS_DIR="$TORUS_BASE/runs/r"
LOG_DIR="$TORUS_BASE/runs/r/_logs/stage2-v5-tournaments"
mkdir -p "$LOG_DIR"

EXP_ID="EXP-RPM-L15-TWN-V5"
MANIFEST="$TORUS_BASE/research/residual-pareto/experiments/$EXP_ID/manifest.yaml"
if [ ! -f "$MANIFEST" ]; then
    echo "[stage2-v5] missing manifest: $MANIFEST" >&2
    exit 1
fi
TARGET_MODULE="model.layers.15.mlp.down_proj"
GROUP_SIZE=$(grep -E "^[[:space:]]*damage_group_size:" "$MANIFEST" | head -1 | awk '{print $2}')
THRESHOLD=$(grep -E "^[[:space:]]*damage_threshold:" "$MANIFEST" | head -1 | awk '{print $2}')

if [ -z "$GROUP_SIZE" ] || [ -z "$THRESHOLD" ]; then
    echo "[stage2-v5] missing damage_group_size or damage_threshold in manifest" >&2
    exit 1
fi

stage1_cal() {
    echo "[stage2-v5] Stage 1 CAL pilot: group_size=$GROUP_SIZE thresholds=0.0,0.3,0.5,0.7,1.0"
    local site_dir="$RUNS_DIR/${EXP_ID}-CAL"
    mkdir -p "$site_dir"
    local ts
    ts=$(date -u +%Y%m%dT%H%M%SZ)
    local run_dir="$site_dir/$ts"
    mkdir -p "$run_dir"
    local cell_log="$LOG_DIR/${EXP_ID}-CAL_${ts}.log"
    (
        cd "$TORUS_BASE"
        .venv/bin/python examples/af2_storage_tournament.py \
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
            --tasks wikitext \
            --ids-cache /tmp/wikitext103_train_ids.npy \
            --device cuda:0 \
            --dtype float16 \
            --eval-dtype float16 \
            --damage-ptq \
            --damage-threshold 0.7 \
            --damage-group-size "$GROUP_SIZE" \
            --pre-train-eval \
            --out-dir "$run_dir" \
            2>&1
        echo "[stage2-v5] === Stage 1 CAL pilot COMPLETE ==="
    ) >> "$cell_log" 2>&1
    echo "[stage2-v5] Stage 1 log: $cell_log"
    echo "[stage2-v5] Stage 1 run_dir: $run_dir"
}

stage2_tournament() {
    echo "[stage2-v5] Stage 2 tournament: group_size=$GROUP_SIZE threshold=$THRESHOLD"
    local site_dir="$RUNS_DIR/$EXP_ID"
    mkdir -p "$site_dir"
    local ts
    ts=$(date -u +%Y%m%dT%H%M%SZ)
    local run_dir="$site_dir/$ts"
    mkdir -p "$run_dir"
    local cell_log="$LOG_DIR/${EXP_ID}_${ts}.log"
    (
        cd "$TORUS_BASE"
        .venv/bin/python examples/af2_storage_tournament.py \
            --model allenai/OLMo-1B-0724-hf \
            --target-module "$TARGET_MODULE" \
            --arms t2_ternary,int4_residual,int8_residual,lora,dense_adapter,random_t2_ternary,random_lora \
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
            --damage-threshold "$THRESHOLD" \
            --damage-group-size "$GROUP_SIZE" \
            --pre-train-eval \
            --out-dir "$run_dir" \
            2>&1
        echo "[stage2-v5] === Stage 2 tournament COMPLETE ==="
    ) >> "$cell_log" 2>&1
    echo "[stage2-v5] Stage 2 log: $cell_log"
    echo "[stage2-v5] Stage 2 run_dir: $run_dir"
}

mode="${1:-both}"
case "$mode" in
    cal) stage1_cal ;;
    tournament) stage2_tournament ;;
    both) stage1_cal && stage2_tournament ;;
    *) echo "Usage: $0 [cal|tournament|both]" >&2; exit 1 ;;
esac

echo "[stage2-v5] done"
