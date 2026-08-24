#!/usr/bin/env bash
# Stage 2 v2 tournament launcher (run after CAL pilot completes).
#
# Reads the preregistered tournament manifests from
#   research/residual-pareto/experiments/EXP-RPM-{SITE_ID}/manifest.yaml
# and runs each one at the preregistered σ, with the full Stage 1 / 1.5
# tournament protocol (7 trained arms + 2 random controls, n_steps=500).
#
# Usage:
#   ./stage2-v2-tournaments-launch.sh              # all QUALIFYING sites
#   ./stage2-v2-tournaments-launch.sh af2d-gauss   # one site
#
# Each site: 7 trained + 2 random = 9 arms × 3 seeds = 27 runs.
# Estimated ~3 min/run (n_steps=500 is heavier than n_steps=0).
# Total per site: ~80 min. Across all qualifying sites: depends on CAL.

set -uo pipefail

TORUS_BASE="${TORUS_BASE:-/home/andrew-jochl/TORUS}"
RUNS_DIR="$TORUS_BASE/runs/r"
LOG_DIR="$TORUS_BASE/runs/r/_logs/stage2-v2-tournaments"
mkdir -p "$LOG_DIR"

# Preregistered sites (same as CAL pilot).
declare -A SITES=(
    [af2d-gauss]="model.layers.0.mlp.down_proj"
    [L15-gauss]="model.layers.15.mlp.down_proj"
    [L0-q-gauss]="model.layers.0.self_attn.q_proj"
    [L0-v-gauss]="model.layers.0.self_attn.v_proj"
)

selected=""
for arg in "$@"; do
    case "$arg" in
        --help|-h)
            sed -n '2,20p' "$0"
            exit 0
            ;;
        *) selected="$arg" ;;
    esac
done

run_one_site() {
    local site_id="$1"
    local target_module="${SITES[$site_id]}"
    local exp_id="EXP-RPM-${site_id^^}"
    local manifest="$TORUS_BASE/research/residual-pareto/experiments/$exp_id/manifest.yaml"
    if [ ! -f "$manifest" ]; then
        echo "[stage2-v2-tournaments] no manifest at $manifest; skipping"
        return
    fi
    local sigma
    sigma=$(grep -A1 "damage_sigma:" "$manifest" | tail -1 | tr -d ' ')
    if [ -z "$sigma" ]; then
        echo "[stage2-v2-tournaments] manifest has no damage_sigma; skipping"
        return
    fi

    local site_dir="$RUNS_DIR/$exp_id"
    mkdir -p "$site_dir"
    local ts
    ts=$(date -u +%Y%m%dT%H%M%SZ)
    local run_dir="$site_dir/$ts"
    mkdir -p "$run_dir"

    echo "[stage2-v2-tournaments] site=$site_id target=$target_module sigma=$sigma ts=$ts"
    local cell_log="$LOG_DIR/${exp_id}_${ts}.log"
    (
        cd "$TORUS_BASE"
        .venv/bin/python examples/af2_storage_tournament.py \
            --model allenai/OLMo-1B-hf \
            --target-module "$target_module" \
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
            --dtype float32 \
            --eval-dtype float16 \
            --damage-gaussian \
            --damage-sigma "$sigma" \
            --damage-seed 0 \
            --pre-train-eval \
            --out-dir "$run_dir" \
            2>&1
        echo "[stage2-v2-tournaments] === site=$site_id COMPLETE ==="
    ) >> "$cell_log" 2>&1
    echo "[stage2-v2-tournaments] DONE site=$site_id sigma=$sigma log=$cell_log"
}

if [ -z "$selected" ]; then
    sites_to_run=("${!SITES[@]}")
else
    sites_to_run=("$selected")
fi

for sid in "${sites_to_run[@]}"; do
    if [ -z "${SITES[$sid]:-}" ]; then
        echo "[stage2-v2-tournaments] unknown site '$sid'; choices: ${!SITES[*]}"
        continue
    fi
    run_one_site "$sid"
done

echo "[stage2-v2-tournaments] all done; logs: $LOG_DIR"