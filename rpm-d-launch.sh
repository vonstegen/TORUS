#!/usr/bin/env bash
# Stage 1 launch (EXP-RPM-D0..D5 damage sweep).
# 6 regimes × 7 arms × 3 seeds = 126 runs total.
# AF8 governance: new namespace per regime; fresh process per regime;
# independent token cache per regime.
#
# Usage: ./rpm-d-launch.sh [REGIMES]
#   REGIMES: comma-separated list of regime IDs (default: all 6).
#   Example: ./rpm-d-launch.sh EXP-RPM-D0,EXP-RPM-D5
#
# Outputs:
#   runs/r/EXP-RPM-D<n>/<timestamp>/<driver>/
#     aggregate.json + driver.log + per-seed artifacts
set -euo pipefail

cd /home/andrew-jochl/TORUS

# All Stage 1 regimes (in order from mildest to most damaging).
ALL_REGIMES=(
    "EXP-RPM-D0"   # FP16 baseline (no damage)
    "EXP-RPM-D1"   # threshold=0.0 (sign-rounding only)
    "EXP-RPM-D2"   # threshold=0.3 (light TWN zeroing)
    "EXP-RPM-D3"   # threshold=0.5 (moderate TWN zeroing)
    "EXP-RPM-D4"   # threshold=0.6 (heavy TWN zeroing)
    "EXP-RPM-D5"   # threshold=0.7 (catastrophic / AF2-D reference)
)

# Per-regime damage knobs (group_size, threshold, calibrate_norm).
# D0 has no damage; D1-D5 use AF2-D's recipe with varied threshold.
declare -A THRESHOLD=(
    ["EXP-RPM-D0"]=""
    ["EXP-RPM-D1"]="0.0"
    ["EXP-RPM-D2"]="0.3"
    ["EXP-RPM-D3"]="0.5"
    ["EXP-RPM-D4"]="0.6"
    ["EXP-RPM-D5"]="0.7"
)

# Common training/eval args (match EXP-RPM-000 / AF2-D recipe).
COMMON_ARGS=(
    --model allenai/OLMo-1B-0724-hf
    --target-module model.layers.0.mlp.down_proj
    --arms t2_ternary,int4_residual,int8_residual,lora,dense_adapter,random_t2_ternary,random_lora
    --seeds 1,2,3
    --n-steps 500
    --batch-size 4
    --seq-len 128
    --lr 1e-3
    --momentum 0.9
    --grad-clip 1.0
    --tasks wikitext,arc_easy,lambada_openai
    --ids-cache /tmp/wikitext103_train_ids.npy
    --device cuda
    --dtype float16
    --eval-dtype float16
    --matched-bytes-tolerance-pct 1.0
    --damage-group-size 128
)

# Select which regimes to run (default: all).
SELECTED_REGIMES="${1:-$(IFS=,; echo "${ALL_REGIMES[*]}")}"
IFS=',' read -ra RUN_REGIMES <<< "$SELECTED_REGIMES"

echo "[rpm-d] frozen driver SHA: 7383b57 (EXP-RPM-000 reproduced at 687f3f5)"
echo "[rpm-d] current HEAD: $(git rev-parse --short HEAD)"
echo "[rpm-d] regimes to run: ${RUN_REGIMES[*]}"
echo "[rpm-d] total runs: $((${#RUN_REGIMES[@]} * 7 * 3))"
echo "[rpm-d] starting at $(date -u +%Y-%m-%dT%H:%M:%SZ)"

mkdir -p runs/r

for REGIME in "${RUN_REGIMES[@]}"; do
    TS="$(date -u +%Y%m%dT%H%M%SZ)"
    OUT_DIR="runs/r/${REGIME}/${TS}/af2d"
    mkdir -p "${OUT_DIR}"

    echo "[rpm-d] ============================================"
    echo "[rpm-d] regime: ${REGIME} (threshold=${THRESHOLD[$REGIME]:-none})"
    echo "[rpm-d] namespace: ${OUT_DIR}"
    echo "[rpm-d] start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

    # Build damage flags.
    DAMAGE_ARGS=()
    T="${THRESHOLD[$REGIME]:-}"
    if [ -n "$T" ]; then
        DAMAGE_ARGS=(--damage-ptq --damage-threshold "$T" --pre-train-eval)
    fi

    .venv/bin/python examples/af2_storage_tournament.py \
        "${COMMON_ARGS[@]}" \
        "${DAMAGE_ARGS[@]}" \
        --out-dir "${OUT_DIR}" \
        2>&1 | tee "${OUT_DIR}/driver.log"

    echo "[rpm-d] regime ${REGIME} done at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
done

echo "[rpm-d] ============================================"
echo "[rpm-d] all regimes done at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[rpm-d] next step: per-regime audit + cross-regime RPM-002 analysis."