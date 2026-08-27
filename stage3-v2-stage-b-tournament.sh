#!/usr/bin/env bash
# Stage 3 v2 — Stage B tournament at calibrated (mechanism, magnitude) points.
# Replaces Stage 3 v1B-B-3 calibrated set with the 4-mechanism expansion.
# Uses TRAINED_ARMS by default; random arms submitted via stage3-v2-random-arms.sh.
set -euo pipefail
export PYTHONPATH=/home/andrew-jochl/TORUS
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# Filled in after Stage A completes; placeholder for now.
MECH_LIST_FILE="${1:-/home/andrew-jochl/TORUS/runs/r/EXP-RPM-DAMAGE-MAP-V2/calibrated_cells.tsv}"

TS=$(date -u +%Y%m%dT%H%M%SZ)
OUT_BASE=/home/andrew-jochl/TORUS/runs/r/EXP-RPM-DAMAGE-MAP-V2/${TS}/stage_b_tournament
mkdir -p "${OUT_BASE}"
LOG=/home/andrew-jochl/TORUS/runs/r/_logs/stage3-v2-stage-b.log
echo "[stage3-v2-stage-b] TS=${TS} OUT_BASE=${OUT_BASE} CELLS=${MECH_LIST_FILE}" | tee -a "${LOG}"
NAV=/home/andrew-jochl/TORUS/.venv/bin/python
TOURN=/home/andrew-jochl/TORUS/examples/af2_storage_tournament.py
MODEL=allenai/OLMo-1B-hf
TARGET=model.layers.0.mlp.down_proj
TASKS=wikitext,arc_easy,lambada_openai
DTYPE=bfloat16
EVAL_DTYPE=float16

run_cell () {
    local MECH="$1"
    local PARAM="$2"
    local CELL="${3}"
    local OUT_DIR="${OUT_BASE}/${CELL}"
    if [ -d "${OUT_DIR}" ] && [ -f "${OUT_DIR}/aggregate.json" ]; then
        echo "[stage3-v2-stage-b] SKIP ${CELL} (already complete)" | tee -a "${LOG}"
        return 0
    fi
    mkdir -p "${OUT_DIR}"
    echo "[stage3-v2-stage-b] ${CELL} (mech=${MECH} param=${PARAM})" | tee -a "${LOG}"

    local EXTRA=""
    local DAMAGE_FLAG=""
    case "${MECH}" in
        twn)
            DAMAGE_FLAG="--damage-ptq"
            EXTRA="--damage-threshold ${PARAM}"
            ;;
        gaussian)
            DAMAGE_FLAG="--damage-gaussian"
            EXTRA="--damage-sigma ${PARAM}"
            ;;
        magnitude_prune)
            DAMAGE_FLAG="--damage-magnitude-prune"
            EXTRA="--damage-prune-k ${PARAM}"
            ;;
        dropout)
            DAMAGE_FLAG="--damage-dropout"
            EXTRA="--damage-dropout-p ${PARAM}"
            ;;
        *)
            echo "[stage3-v2-stage-b] UNKNOWN mech=${MECH}" | tee -a "${LOG}"
            return 1
            ;;
    esac

    ${NAV} "${TOURN}" \
        --model "${MODEL}" --target-module "${TARGET}" \
        ${DAMAGE_FLAG} ${EXTRA} \
        --pre-train-eval \
        --arms "t2_ternary,lora" \
        --seeds 1,2,3 \
        --n-steps 500 --batch-size 4 --seq-len 128 \
        --lr 1e-3 --momentum 0.9 --grad-clip 1.0 \
        --tasks "${TASKS}" \
        --dtype ${DTYPE} --eval-dtype ${EVAL_DTYPE} \
        --out-dir "${OUT_DIR}" 2>&1 | tee -a "${LOG}" | tail -8 || true
}

# Read calibrated_cells.tsv: lines like "<mech>\\t<param>\\t<cell_id>"
while IFS=$'\t' read -r MECH PARAM CELL; do
    [ -z "${MECH}" ] && continue
    [[ "${MECH}" =~ ^# ]] && continue
    run_cell "${MECH}" "${PARAM}" "${CELL}"
done < "${MECH_LIST_FILE}"

echo "[stage3-v2-stage-b] DONE; log: ${LOG}"
