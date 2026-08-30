#!/usr/bin/env bash
# EXP-RPM-SITE-DISCOVERY — Track B condition-4 site-discovery sweep
# (CAL-only). Preregistered manifest:
#   research/residual-pareto/experiments/EXP-RPM-SITE-DISCOVERY/manifest.yaml
#
# Grid: 13 sites x 18 cells + 3 TWN-site baseline cells + 2 reference
# cells = 239 cells. Eval-only (--n-steps 0), wikitext only, fp16.
# Frozen Stage 2 v2 pilot protocol; driver unchanged.
#
# Usage:
#   ./site-discovery-launch.sh              # launch all sites, 2 GPUs
#   ./site-discovery-launch.sh ref-gauss-v-L0   # only that site
set -uo pipefail

TORUS_BASE="${TORUS_BASE:-/home/andrew-jochl/TORUS}"
RUNS_DIR="$TORUS_BASE/runs/r"
LOG_DIR="$RUNS_DIR/_logs/site-discovery"
mkdir -p "$LOG_DIR"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="$RUNS_DIR/EXP-RPM-SITE-DISCOVERY/$TS"
mkdir -p "$RUN_DIR"
echo "$TS" > "$LOG_DIR/latest_ts.txt"

SIGMAS=(0.0 0.05 0.10 0.20 0.50 1.00)
THRS=(0.0 0.4 0.6 0.8 0.9 1.0)
SEEDS=(1 2 3)

# Preregistered sites: id -> "mechanism target_module"
# mech = gauss | twn
declare -A SITES=(
    [twn-v-L0]="twn model.layers.0.self_attn.v_proj"
    [gauss-v-L1]="gauss model.layers.1.self_attn.v_proj"
    [gauss-v-L2]="gauss model.layers.2.self_attn.v_proj"
    [gauss-v-L4]="gauss model.layers.4.self_attn.v_proj"
    [gauss-v-L8]="gauss model.layers.8.self_attn.v_proj"
    [gauss-v-L12]="gauss model.layers.12.self_attn.v_proj"
    [gauss-v-L15]="gauss model.layers.15.self_attn.v_proj"
    [gauss-d-L1]="gauss model.layers.1.mlp.down_proj"
    [gauss-d-L4]="gauss model.layers.4.mlp.down_proj"
    [gauss-d-L8]="gauss model.layers.8.mlp.down_proj"
    [gauss-d-L12]="gauss model.layers.12.mlp.down_proj"
    [twn-v-L4]="twn model.layers.4.self_attn.v_proj"
    [twn-v-L15]="twn model.layers.15.self_attn.v_proj"
)

# Preregistered execution order (manifest experiment.execution_order).
ORDER=(
    ref-gauss-v-L0 ref-twn-d-L0
    twn-v-L0
    gauss-v-L1 gauss-v-L2 gauss-v-L4 gauss-v-L8 gauss-v-L12 gauss-v-L15
    gauss-d-L1 gauss-d-L4 gauss-d-L8 gauss-d-L12
    twn-v-L4 twn-v-L15
)

selected=""
for arg in "$@"; do
    case "$arg" in
        --help|-h) sed -n '2,26p' "$0"; exit 0 ;;
        *) selected="$arg" ;;
    esac
done

if [ -z "$selected" ]; then
    sites_to_run=("${ORDER[@]}")
else
    sites_to_run=("$selected")
fi

run_cell() {  # <site_id> <mech> <knob> <seed> <gpu> <knob_dir_name>
    local site_id="$1" mech="$2" knob="$3" seed="$4" gpu="$5" knob_name="$6"
    local cell_dir="$RUN_DIR/$site_id/$knob_name/seed-$(printf '%03d' "$seed")"
    mkdir -p "$cell_dir"
    local args=(
        --model allenai/OLMo-1B-0724-hf
        --target-module "$TARGET"
        --arms t2_ternary
        --seeds "$seed"
        --n-steps 0
        --batch-size 4
        --seq-len 128
        --tasks wikitext
        --ids-cache /tmp/wikitext103_train_ids.npy
        --device "cuda:$gpu"
        --dtype float16
        --eval-dtype float16
        --pre-train-eval
        --out-dir "$cell_dir"
    )
    case "$mech" in
        gauss)
            args+=(--damage-gaussian --damage-sigma "$knob" --damage-seed 0) ;;
        twn)
            args+=(--damage-ptq --damage-group-size 128 --damage-threshold "$knob") ;;
    esac
    .venv/bin/python examples/af2_storage_tournament.py "${args[@]}" \
        > "$cell_dir/driver.log" 2>&1
}

run_site() {  # <site_id> <gpu>
    local site_id="$1" gpu="$2"
    local spec="${SITES[$site_id]:-}"
    local mech target
    mech="${spec%% *}"
    target="${spec#* }"
    export TARGET="$target"
    local log="$LOG_DIR/${site_id}_${TS}.log"
    {
        echo "[site-disc] site=$site_id mech=$mech target=$target gpu=cuda:$gpu ts=$TS"
        local knob_name
        if [ "$mech" = "gauss" ]; then
            for knob in "${SIGMAS[@]}"; do
                knob_name=$(printf "sigma-%05.2f" "$knob" | tr '.' '_')
                for seed in "${SEEDS[@]}"; do
                    run_cell "$site_id" "$mech" "$knob" "$seed" "$gpu" "$knob_name"
                done
            done
        else
            # TWN sites: baseline cell (sigma=0.0, the verification
            # gate) + threshold grid.
            mkdir -p "$RUN_DIR/$site_id/baseline/seed-001"
            .venv/bin/python examples/af2_storage_tournament.py \
                --model allenai/OLMo-1B-0724-hf \
                --target-module "$target" \
                --arms t2_ternary --seeds 1 --n-steps 0 \
                --batch-size 4 --seq-len 128 --tasks wikitext \
                --ids-cache /tmp/wikitext103_train_ids.npy \
                --device "cuda:$gpu" --dtype float16 --eval-dtype float16 \
                --damage-gaussian --damage-sigma 0.0 --damage-seed 0 \
                --pre-train-eval \
                --out-dir "$RUN_DIR/$site_id/baseline/seed-001" \
                > "$RUN_DIR/$site_id/baseline/seed-001/driver.log" 2>&1
            for knob in "${THRS[@]}"; do
                knob_name=$(printf "thr-%05.2f" "$knob" | tr '.' '_')
                for seed in "${SEEDS[@]}"; do
                    run_cell "$site_id" "$mech" "$knob" "$seed" "$gpu" "$knob_name"
                done
            done
        fi
        echo "[site-disc] === site=$site_id COMPLETE ==="
    } >> "$log" 2>&1
}

run_reference() {  # <ref_id> <gpu>
    local ref_id="$1" gpu="$2"
    local target knob mech
    case "$ref_id" in
        ref-gauss-v-L0)
            target="model.layers.0.self_attn.v_proj"
            mech="gauss"
            knob=0.5 ;;
        ref-twn-d-L0)
            target="model.layers.0.mlp.down_proj"
            mech="twn"
            knob=0.7 ;;
    esac
    local log="$LOG_DIR/${ref_id}_${TS}.log"
    local knob_name
    if [ "$mech" = "gauss" ]; then
        knob_name=$(printf "sigma-%05.2f" "$knob" | tr '.' '_')
    else
        knob_name=$(printf "thr-%05.2f" "$knob" | tr '.' '_')
    fi
    {
        local cell_dir="$RUN_DIR/$ref_id/$knob_name/seed-001"
        mkdir -p "$cell_dir"
        local args=(
            --model allenai/OLMo-1B-0724-hf
            --target-module "$target"
            --arms t2_ternary --seeds 1 --n-steps 0
            --batch-size 4 --seq-len 128 --tasks wikitext
            --ids-cache /tmp/wikitext103_train_ids.npy
            --device "cuda:$gpu" --dtype float16 --eval-dtype float16
            --pre-train-eval
            --out-dir "$cell_dir"
        )
        case "$mech" in
            gauss) args+=(--damage-gaussian --damage-sigma "$knob" --damage-seed 0) ;;
            twn) args+=(--damage-ptq --damage-group-size 128 --damage-threshold "$knob") ;;
        esac
        .venv/bin/python examples/af2_storage_tournament.py "${args[@]}" \
            > "$cell_dir/driver.log" 2>&1
        echo "[site-disc] === ref=$ref_id COMPLETE ==="
    } >> "$log" 2>&1
}
# Serialized per-GPU pairs: exactly ONE worker per GPU at a time.
# (Run 20260830T174505Z INVALID: concurrent launch of all 15 workers
# put up to 8 processes per GPU -> CUDA OOM on the reference cells.)
_is_ref() {
    [ "$1" = "ref-gauss-v-L0" ] || [ "$1" = "ref-twn-d-L0" ]
}
idx=0
n_sites=${#sites_to_run[@]}
while [ "$idx" -lt "$n_sites" ]; do
    s0="${sites_to_run[$idx]}"
    idx=$((idx + 1))
    if _is_ref "$s0"; then
        run_reference "$s0" 0 &
    else
        run_site "$s0" 0 &
    fi
    p0=$!
    if [ "$idx" -lt "$n_sites" ]; then
        s1="${sites_to_run[$idx]}"
        idx=$((idx + 1))
        if _is_ref "$s1"; then
            run_reference "$s1" 1 &
        else
            run_site "$s1" 1 &
        fi
        p1=$!
    else
        p1=""
    fi
    wait "$p0"
    if [ -n "$p1" ]; then
        wait "$p1"
    fi
done

echo "[site-disc] all workers done at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[site-disc] summarize: .venv/bin/python site-discovery-summarize.py --run-dir $RUN_DIR"
