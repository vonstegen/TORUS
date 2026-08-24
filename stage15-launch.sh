#!/usr/bin/env bash
# Stage 1.5 launch (EXP-RPM-D0'..D5' damage sweep, observed-ppl axis).
# 6 regimes × 7 arms × 3 seeds = 126 runs total.
# AF8 governance: new namespace per regime; fresh process per regime;
# independent token cache per regime.
#
# Damage axis: observed-ppl from EXP-RPM-CAL on the AF2-D layer
# (threshold values chosen from CAL so each regime maps to a distinct ppl band).
# Recipe: AF2-D (matched-storage tournament under v2.3 cost-vector).
#
# Usage: ./stage15-launch.sh [REGIMES]
#   REGIMES: comma-separated list of regime IDs (default: all 6).
#   Example: ./stage15-launch.sh EXP-RPM-D0p,EXP-RPM-D5p
#
# Outputs:
#   runs/r/EXP-RPM-D<n>p/<timestamp>/af2d/
#     aggregate.json + driver.log + per-seed artifacts
set -euo pipefail

cd /home/andrew-jochl/TORUS

# All Stage 1.5 regimes (ordered by CAL ppl, mildest to most damaging).
# D0' has no damage (FP16 reference); D1'-D5' use AF2-D's recipe with
# threshold values chosen from EXP-RPM-CAL.
ALL_REGIMES=(
    "EXP-RPM-D0p"   # no damage; FP16 reference (CAL ppl ~13)
    "EXP-RPM-D1p"   # threshold=1.0; CAL ppl 88.31 (light)
    "EXP-RPM-D2p"   # threshold=0.9; CAL ppl 203.60 (moderate-light)
    "EXP-RPM-D3p"   # threshold=0.8; CAL ppl 303.06 (moderate)
    "EXP-RPM-D4p"   # threshold=0.7; CAL ppl 429.55 (heavy/catastrophic)
    "EXP-RPM-D5p"   # threshold=0.6; CAL ppl 697.29 (severe)
)

# Per-regime damage knob (threshold). CAL ppl values for the AF2-D
# layer are recorded in the manifest's `damage_ptq.calibrated_ppl`.
declare -A THRESHOLD=(
    ["EXP-RPM-D0p"]=""
    ["EXP-RPM-D1p"]="1.0"
    ["EXP-RPM-D2p"]="0.9"
    ["EXP-RPM-D3p"]="0.8"
    ["EXP-RPM-D4p"]="0.7"
    ["EXP-RPM-D5p"]="0.6"
)

# Common training/eval args (match Stage 1 / EXP-RPM-000 / AF2-D recipe).
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

echo "[stage15] frozen driver SHA: 692e8ee (Stage 1; AF8 governance)"
echo "[stage15] current HEAD: $(git rev-parse --short HEAD)"
echo "[stage15] regimes to run: ${RUN_REGIMES[*]}"
echo "[stage15] total runs: $((${#RUN_REGIMES[@]} * 7 * 3))"
echo "[stage15] starting at $(date -u +%Y-%m-%dT%H:%M:%SZ)"

mkdir -p runs/r

for REGIME in "${RUN_REGIMES[@]}"; do
    TS="$(date -u +%Y%m%dT%H%M%SZ)"
    OUT_DIR="runs/r/${REGIME}/${TS}/af2d"
    mkdir -p "${OUT_DIR}"

    echo "[stage15] ============================================"
    echo "[stage15] regime: ${REGIME} (threshold=${THRESHOLD[$REGIME]:-none})"
    echo "[stage15] namespace: ${OUT_DIR}"
    echo "[stage15] start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

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

    echo "[stage15] regime ${REGIME} done at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
done

echo "[stage15] ============================================"
echo "[stage15] all regimes done at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[stage15] next step: per-regime audit + cross-regime RPM-001/002/006 analysis (post-hoc eval script ready)."