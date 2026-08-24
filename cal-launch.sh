#!/usr/bin/env bash
# EXP-RPM-CAL calibration pre-experiment launch.
# Per the user directive: preregistered before any further damage sweep.
# Goal: characterize threshold -> ppl mapping on the AF2-D layer,
# attention_k, and late_mlp. NO residual training; eval-only runs.
#
# Per threshold x layer x seed: apply --damage-ptq --pre-train-eval,
# capture pre_train_eval.json. We use --arms t2_ternary because the
# driver does NOT implement a no_correction arm; t2_ternary fires
# the pre-train eval via --pre-train-eval and the eval is identical
# to what no_correction would compute (same damaged base, no
# adapter contribution to logits in eval-time since the patched
# forward adds T2 residual; for calibration we WANT pre-train only).
#
# Actually: with --arms t2_ternary + --pre-train-eval, the driver
# ALSO does post-train eval of t2_ternary. We don't want the
# post-train data for CAL; we only want pre_train_eval.json.
# Post-train eval is harmless (adds ~1-2 min per cell) and we'll
# just ignore those numbers in the CAL summary.
set -euo pipefail

cd /home/andrew-jochl/TORUS

THRESHOLDS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
LAYERS=(
    "model.layers.0.mlp.down_proj"
    "model.layers.0.self_attn.k_proj"
    "model.layers.15.mlp.down_proj"
)
SEEDS=(1 2 3)

# Minimal driver args: pre-train eval only.
COMMON_ARGS=(
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

echo "[rpm-cal] frozen driver SHA: 692e8ee"
echo "[rpm-cal] current HEAD: $(git rev-parse --short HEAD)"
echo "[rpm-cal] thresholds: ${THRESHOLDS[*]}"
echo "[rpm-cal] layers: ${LAYERS[*]}"
echo "[rpm-cal] seeds: ${SEEDS[*]}"
echo "[rpm-cal] total runs: $((${#THRESHOLDS[@]} * ${#LAYERS[@]} * ${#SEEDS[@]}))"
echo "[rpm-cal] starting at $(date -u +%Y-%m-%dT%H:%M:%SZ)"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_BASE="runs/r/EXP-RPM-CAL/${TS}"
mkdir -p "${OUT_BASE}"

for LAYER in "${LAYERS[@]}"; do
    LAYER_SLUG=$(echo "${LAYER}" | sed 's|[/.]|_|g')
    for THR in "${THRESHOLDS[@]}"; do
        THR_SLUG=$(printf "%05.2f" "${THR}" | tr '.' '_')
        OUT_DIR="${OUT_BASE}/layer-${LAYER_SLUG}_thr-${THR_SLUG}"
        mkdir -p "${OUT_DIR}"
        echo "[rpm-cal] running ${LAYER} threshold=${THR} -> ${OUT_DIR}"
        .venv/bin/python examples/af2_storage_tournament.py \
            "${COMMON_ARGS[@]}" \
            --target-module "${LAYER}" \
            --damage-threshold "${THR}" \
            --seeds "$(IFS=,; echo "${SEEDS[*]}")" \
            --out-dir "${OUT_DIR}" \
            2>&1 | tee "${OUT_DIR}/driver.log"
    done
done

echo "[rpm-cal] ============================================"
echo "[rpm-cal] all runs done at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[rpm-cal] next step: summarize threshold->ppl function per layer"