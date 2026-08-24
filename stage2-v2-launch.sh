#!/usr/bin/env bash
# Stage 2 v2 CAL pilot launcher.
#
# Runs the Gaussian weight-noise calibration sweep on the four sites
# registered in research/residual-pareto/experiments/EXP-RPM-*-GAUSS-CAL/.
# L8 down_proj was dropped from the registry after the freeze-exception
# commit: with both TWN and Gaussian, layer 8 is plausibly degenerate
# at FP16 reference; one deeper MLP site (L15) suffices as a falsification
# probe. The remaining three sites cover MLP (AF2-D, L15) and attention
# (L0-q, L0-v) layer categories.
#
# Usage:
#   ./stage2-v2-launch.sh                # launch all 4 sites in background
#   ./stage2-v2-launch.sh af2d-gauss     # launch only that site
#   ./stage2-v2-launch.sh --foreground af2d-gauss   # foreground (debug)
#
# Each site: 6 sigmas × 3 seeds = 18 cells. Eval-only (no training).
# wikitext-only CAL (arc_easy / lambada_openai omitted to bound
# compute; tournament stage will run the full task set on QUALIFYING
# sites only). Estimated ~3 min/cell, ~54 min/site, ~3.6 hours total.
#
# Each cell writes to:
#   runs/r/EXP-RPM-{SITE}-CAL/{timestamp}/sigma-{v}/seed-{n}/pre_train_eval.json

set -uo pipefail

TORUS_BASE="${TORUS_BASE:-/home/andrew-jochl/TORUS}"
RUNS_DIR="$TORUS_BASE/runs/r"
LOG_DIR="$TORUS_BASE/runs/r/_logs/stage2-v2"
mkdir -p "$LOG_DIR"

# Preregistered sites and their target_modules. See
# research/residual-pareto/experiments/EXP-RPM-*-GAUSS-CAL/manifest.yaml.
declare -A SITES=(
    [af2d-gauss]="model.layers.0.mlp.down_proj"
    [L15-gauss]="model.layers.15.mlp.down_proj"
    [L0-q-gauss]="model.layers.0.self_attn.q_proj"
    [L0-v-gauss]="model.layers.0.self_attn.v_proj"
)
SIGMAS=(0.0 0.05 0.10 0.20 0.50 1.00)
SEEDS=(1 2 3)

foreground=false
selected=""
for arg in "$@"; do
    case "$arg" in
        --foreground) foreground=true ;;
        --help|-h)
            sed -n '2,20p' "$0"
            exit 0
            ;;
        *) selected="$arg" ;;
    esac
done

if [ -z "$selected" ]; then
    sites_to_run=("${!SITES[@]}")
else
    sites_to_run=("$selected")
fi

launch_one_site() {
    local site_id="$1"
    local target_module="${SITES[$site_id]}"
    local exp_id="EXP-RPM-${site_id^^}-CAL"
    local site_dir="$RUNS_DIR/$exp_id"
    mkdir -p "$site_dir"

    local ts
    ts=$(date -u +%Y%m%dT%H%M%SZ)
    local run_dir="$site_dir/$ts"
    mkdir -p "$run_dir"

    echo "[stage2-v2] site=$site_id target=$target_module ts=$ts"
    echo "[stage2-v2] sigmas=${SIGMAS[*]} seeds=${SEEDS[*]}"
    echo "[stage2-v2] cells=$(( ${#SIGMAS[@]} * ${#SEEDS[@]} ))"

    local cell_log="$LOG_DIR/${exp_id}_${ts}.log"
    (
        cd "$TORUS_BASE"
        local first=1
        for sigma in "${SIGMAS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                local sigma_tag
                sigma_tag=$(printf "%05.2f" "$sigma" | tr '.' '_')
                local cell_dir="$run_dir/sigma-${sigma_tag}/seed-$(printf '%03d' "$seed")"
                mkdir -p "$cell_dir"
                local cell_log_inner="$cell_dir/driver.log"
                {
                    if [ "$first" -eq 1 ]; then
                        echo "[stage2-v2] === site=$site_id sigma=$sigma seed=$seed ==="
                        first=0
                    else
                        echo "[stage2-v2] --- site=$site_id sigma=$sigma seed=$seed ---"
                    fi
                    .venv/bin/python examples/af2_storage_tournament.py \
                        --model allenai/OLMo-1B-hf \
                        --target-module "$target_module" \
                        --arms t2_ternary \
                        --seeds "$seed" \
                        --n-steps 0 \
                        --batch-size 4 \
                        --seq-len 128 \
                        --tasks wikitext \
                        --ids-cache /tmp/wikitext103_train_ids.npy \
                        --device cuda:0 \
                        --dtype float32 \
                        --eval-dtype float16 \
                        --damage-gaussian \
                        --damage-sigma "$sigma" \
                        --damage-seed 0 \
                        --pre-train-eval \
                        --out-dir "$cell_dir" \
                        2>&1
                } >> "$cell_log_inner" 2>&1
                echo "[stage2-v2] done sigma=$sigma seed=$seed -> $cell_dir/driver.log"
            done
        done
        echo "[stage2-v2] === site=$site_id COMPLETE ==="
    ) >> "$cell_log" 2>&1

    if $foreground; then
        wait
    else
        echo "[stage2-v2] backgrounded $exp_id (ts=$ts); log=$cell_log"
        echo "$exp_id $ts" >> "$LOG_DIR/launched.txt"
    fi
}

for sid in "${sites_to_run[@]}"; do
    if [ -z "${SITES[$sid]:-}" ]; then
        echo "[stage2-v2] unknown site '$sid'; choices: ${!SITES[*]}"
        continue
    fi
    launch_one_site "$sid"
done

echo "[stage2-v2] all done; logs: $LOG_DIR"