#!/usr/bin/env bash
# Stage 3 v2 — Stage A probe sweep for MagnitudePrune + Dropout at AF2-D.
# Find (k, p) values producing ±20% of BAND-3 magnitude (~ppl 350-515).
# Stage A samples base damage on the FP16 base, no adapter training.
set -euo pipefail
export PYTHONPATH=/home/andrew-jochl/TORUS
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
TS=$(date -u +%Y%m%dT%H%M%SZ)
OUT_BASE=/home/andrew-jochl/TORUS/runs/r/EXP-RPM-DAMAGE-MAP-V2/${TS}/stage_a_probe
mkdir -p "${OUT_BASE}"
LOG=/home/andrew-jochl/TORUS/runs/r/_logs/stage3-v2-stage-a-probe.log
echo "[stage3-v2-stage-a] TS=${TS} OUT_BASE=${OUT_BASE}" | tee -a "${LOG}"
NAV=/home/andrew-jochl/TORUS/.venv/bin/python
TOURN=/home/andrew-jochl/TORUS/examples/af2_storage_tournament.py
MODEL=allenai/OLMo-1B-hf
TARGET=model.layers.0.mlp.down_proj
TASKS=wikitext,arc_easy,lambada_openai
DTYPE=bfloat16
EVAL_DTYPE=float16

probe_one () {
    local MODE="$1"
    local PARAM="$2"
    local PARAM_NAME="$3"
    local EXTRA="$4"
    local CELL="${MODE}_${PARAM_NAME}${PARAM}"
    local OUT_DIR="${OUT_BASE}/${CELL}"
    mkdir -p "${OUT_DIR}"
    echo "[stage3-v2-stage-a] ${CELL}" | tee -a "${LOG}"
    # One seed; just want pre-train ppl. No adapter training; --n-steps 1 to minimize cost.
    ${NAV} "${TOURN}" \
        --model "${MODEL}" --target-module "${TARGET}" \
        --damage-${MODE} ${EXTRA} \
        --pre-train-eval \
        --arms "t2_ternary" --seeds 1 \
        --n-steps 1 --tasks "${TASKS}" \
        --dtype ${DTYPE} --eval-dtype ${EVAL_DTYPE} \
        --out-dir "${OUT_DIR}" 2>&1 | tee -a "${LOG}" | tail -5 || true
}

# MagnitudePrune sweep
for K in 0.5 0.7 0.8 0.85 0.9 0.93 0.95; do
    probe_one "magnitude-prune" "${K}" "k" "--damage-prune-k ${K}"
done

# Dropout sweep
for P in 0.3 0.5 0.7 0.8 0.9 0.95; do
    probe_one "dropout" "${P}" "p" "--damage-dropout-p ${P}"
done

echo "[stage3-v2-stage-a] DONE; log: ${LOG}"
