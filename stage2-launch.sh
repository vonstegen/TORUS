#!/usr/bin/env bash
# Stage 2 launch (EXP-RPM-Lxx layer-category sweep).
# 2 sites (L15, L8) × CAL pre-experiment + 7-arm tournament.
# Per-site CAL: 11 thresholds × 3 seeds × 1 arm = 33 cells (eval-only).
# Per-site tournament: 7 arms × 3 seeds = 21 runs (full training).
# Total: 2 × (33 + 21) = 108 cells across 4 distinct launch directories.
#
# Stage 2 tests whether the architecture-vs-training signal
# (trained T2 ≫ random T2) holds across layer categories. Sites L15
# and L8 are MLP down_proj at different depths; same architecture
# category (MLP down_proj) as the Stage 1 AF2-D site (layer 0).
#
# Driver SHA: 692e8ee (Stage 1, frozen). NOT modified.
#
# Order of operations (per site):
#   1. Run per-site CAL (33 cells).
#   2. Read CAL output, select threshold that produces ppl closest
#      to the AF2-D reference (ppl ~425.76).
#   3. Run tournament at that threshold (21 runs).
#   4. Post-hoc random-arm eval (21 cells via eval_untrained_arms.py).
#
# For launch simplicity, this script runs CAL and tournament
# sequentially. The CAL threshold selection is done by a separate
# helper script: stage2_select_threshold.py.
#
# Usage: ./stage2-launch.sh [SITES]
#   SITES: comma-separated list of site base IDs (default: L15, L8)
set -euo pipefail

cd /home/andrew-jochl/TORUS

THRESHOLDS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
SEEDS=(1 2 3)

# Common args (CAL is minimal; tournament is full).
CAL_COMMON=(
    --model allenai/OLMo-1B-0724-hf
    --n-steps 1
    --batch-size 4
    --seq-len 128
    --lr 1e-3
    --momentum 0.9
    --grad-clip 1.0
    --tasks wikitext,arc_easy
    --ids-cache /tmp/wikitext103_train_ids.npy
    --device cuda
    --dtype float16
    --eval-dtype float16
    --matched-bytes-tolerance-pct 1.0
    --damage-ptq
    --damage-group-size 128
    --pre-train-eval
    --arms t2_ternary
)

TOURN_COMMON=(
    --model allenai/OLMo-1B-0724-hf
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

# Sites (base ID -> target module path).
declare -A TARGET_MODULE=(
    ["L15"]="model.layers.15.mlp.down_proj"
    ["L8"]="model.layers.8.mlp.down_proj"
)

SELECTED="${1:-L15,L8}"
IFS=',' read -ra RUN_SITES <<< "$SELECTED"

echo "[stage2] frozen driver SHA: 692e8ee (Stage 1)"
echo "[stage2] current HEAD: $(git rev-parse --short HEAD)"
echo "[stage2] sites: ${RUN_SITES[*]}"
echo "[stage2] starting at $(date -u +%Y-%m-%dT%H:%M:%SZ)"

mkdir -p runs/r

for SITE in "${RUN_SITES[@]}"; do
    TM="${TARGET_MODULE[$SITE]}"
    echo "[stage2] ============================================"
    echo "[stage2] site: ${SITE} (target_module: ${TM})"
    echo "[stage2] step 1/3: per-site CAL"

    # ---- CAL (33 cells, 11 thresholds × 3 seeds × 1 arm) ----
    TS_CAL="$(date -u +%Y%m%dT%H%M%SZ)"
    OUT_BASE_CAL="runs/r/EXP-RPM-${SITE}-CAL/${TS_CAL}"
    mkdir -p "${OUT_BASE_CAL}"

    for THR in "${THRESHOLDS[@]}"; do
        THR_SLUG=$(printf "%05.2f" "${THR}" | tr '.' '_')
        OUT_DIR="${OUT_BASE_CAL}/thr-${THR_SLUG}"
        mkdir -p "${OUT_DIR}"
        echo "[stage2]   CAL ${TM} threshold=${THR} -> ${OUT_DIR}"
        .venv/bin/python examples/af2_storage_tournament.py \
            "${CAL_COMMON[@]}" \
            --target-module "${TM}" \
            --damage-threshold "${THR}" \
            --seeds "$(IFS=,; echo "${SEEDS[*]}")" \
            --out-dir "${OUT_DIR}" \
            2>&1 | tee "${OUT_DIR}/driver.log"
    done

    # ---- Step 2: select threshold ----
    echo "[stage2] step 2/3: select tournament threshold from CAL"
    SELECTED_THR=$(.venv/bin/python /tmp/stage2_select_threshold.py
        --cal_root "${OUT_BASE_CAL}" \
        --target_ppl 425.76 2>/dev/null || echo "0.7")
    echo "[stage2]   selected threshold: ${SELECTED_THR}"

    # ---- Step 3: tournament (21 runs) ----
    echo "[stage2] step 3/3: tournament at threshold=${SELECTED_THR}"
    TS_T="$(date -u +%Y%m%dT%H%M%SZ)"
    OUT_T="runs/r/EXP-RPM-${SITE}/${TS_T}/af2d"
    mkdir -p "${OUT_T}"

    .venv/bin/python examples/af2_storage_tournament.py \
        "${TOURN_COMMON[@]}" \
        --target-module "${TM}" \
        --arms t2_ternary,int4_residual,int8_residual,lora,dense_adapter,random_t2_ternary,random_lora \
        --seeds "$(IFS=,; echo "${SEEDS[*]}")" \
        --damage-ptq \
        --damage-threshold "${SELECTED_THR}" \
        --pre-train-eval \
        --out-dir "${OUT_T}" \
        2>&1 | tee "${OUT_T}/driver.log"

    # ---- Step 4: post-hoc random-arm eval ----
    echo "[stage2] step 4: post-hoc random-arm eval at ${SITE}"
    PYTHONPATH=. .venv/bin/python /tmp/eval_untrained_arms.py \
        --regimes "${SITE}" \
        --arms random_t2_ternary,random_lora \
        --tasks wikitext,arc_easy,lambada_openai \
        --batch_size 16 \
        --base /home/andrew-jochl/TORUS \
        2>&1 | tee "${OUT_T}/post_hoc_eval.log"

    echo "[stage2] site ${SITE} done at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
done

echo "[stage2] ============================================"
echo "[stage2] all sites done at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[stage2] next step: per-site audit + cross-site Stage 2 analysis."