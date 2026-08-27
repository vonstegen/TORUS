#!/usr/bin/env bash
# Stage 3 v2 — Random arms tournament for the 4-mechanism envelope.
# Mirrors Stage 3 v1's random-arms pattern but applies to all 4 calibrated cells.
set -euo pipefail
export PYTHONPATH=/home/andrew-jochl/TORUS
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
MECH_LIST_FILE="${1:-/home/andrew-jochl/TORUS/runs/r/EXP-RPM-DAMAGE-MAP-V2/calibrated_cells.tsv}"
STAGE_B_TS="${2:-LATEST}"

TS=$(date -u +%Y%m%dT%H%M%SZ)

if [ "${STAGE_B_TS}" = "LATEST" ]; then
    STAGE_B_TS=$(ls -1 /home/andrew-jochl/TORUS/runs/r/EXP-RPM-DAMAGE-MAP-V2 | grep stage_b_tournament >/dev/null 2>&1 && \
        ls -1 /home/andrew-jochl/TORUS/runs/r/EXP-RPM-DAMAGE-MAP-V2 | grep stage_b_tournament | tail -1 | sed 's|.*/||' || true)
    if [ -z "${STAGE_B_TS:-}" ]; then
        echo "no stage_b_tournament yet"; exit 1
    fi
fi
OUT_BASE=/home/andrew-jochl/TORUS/runs/r/EXP-RPM-DAMAGE-MAP-V2/${TS}/stage_b_random_arms
mkdir -p "${OUT_BASE}"
LOG=/home/andrew-jochl/TORUS/runs/r/_logs/stage3-v2-random-arms.log
echo "[stage3-v2-random] TS=${TS} OUT_BASE=${OUT_BASE} STAGE_B_TS=${STAGE_B_TS}" | tee -a "${LOG}"
NAV=/home/andrew-jochl/TORUS/.venv/bin/python
TOURN=/home/andrew-jochl/TORUS/examples/af2_storage_tournament.py
MODEL=allenai/OLMo-1B-hf
TARGET=model.layers.0.mlp.down_proj
DTYPE=bfloat16
EVAL_DTYPE=float16

run_cell () {
    local MECH="$1"
    local PARAM="$2"
    local CELL="${3}"
    local STAGE_B_OUT="/home/andrew-jochl/TORUS/runs/r/EXP-RPM-DAMAGE-MAP-V2/${STAGE_B_TS}/stage_b_tournament/${CELL}"
    local OUT_DIR="${OUT_BASE}/${CELL}"
    if [ ! -d "${STAGE_B_OUT}" ]; then
        echo "[stage3-v2-random] SKIP ${CELL} (no stage_b dir)" | tee -a "${LOG}"
        return 0
    fi
    if [ -d "${OUT_DIR}" ]; then
        echo "[stage3-v2-random] SKIP ${CELL} (already complete)" | tee -a "${LOG}"
        return 0
    fi
    mkdir -p "${OUT_DIR}"
    echo "[stage3-v2-random] ${CELL}" | tee -a "${LOG}"

    local EXTRA=""
    local DAMAGE_FLAG=""
    case "${MECH}" in
        twn) DAMAGE_FLAG="--damage-ptq"; EXTRA="--damage-threshold ${PARAM}";;
        gaussian) DAMAGE_FLAG="--damage-gaussian"; EXTRA="--damage-sigma ${PARAM}";;
        magnitude_prune) DAMAGE_FLAG="--damage-magnitude-prune"; EXTRA="--damage-prune-k ${PARAM}";;
        dropout) DAMAGE_FLAG="--damage-dropout"; EXTRA="--damage-dropout-p ${PARAM}";;
    esac

    for SEED in 1 2 3; do
        for ARM in random_t2_ternary random_lora; do
            local ARM_DIR="${OUT_DIR}/seed-${SEED}/${ARM}"
            if [ -d "${ARM_DIR}" ] && [ -f "${ARM_DIR}/eval.summary.json" ]; then
                continue
            fi
            mkdir -p "${ARM_DIR}"
            ${NAV} "${TOURN}" \
                --model "${MODEL}" --target-module "${TARGET}" \
                ${DAMAGE_FLAG} ${EXTRA} \
                --arms "${ARM}" \
                --seeds ${SEED} \
                --n-steps 500 --batch-size 4 --seq-len 128 \
                --lr 1e-3 --momentum 0.9 --grad-clip 1.0 \
                --tasks wikitext \
                --dtype ${DTYPE} --eval-dtype ${EVAL_DTYPE} \
                --out-dir "${OUT_DIR}" 2>&1 | tee -a "${LOG}" | tail -3 || true
        done
    done
}

while IFS=$'\t' read -r MECH PARAM CELL; do
    [ -z "${MECH}" ] && continue
    [[ "${MECH}" =~ ^# ]] && continue
    run_cell "${MECH}" "${PARAM}" "${CELL}"
done < "${MECH_LIST_FILE}"

echo "[stage3-v2-random] DONE; log: ${LOG}"
