#!/usr/bin/env bash
# EXP-RPM-000 launch (G-RPM-0 reference-lock + AF2-D reproduction).
# AF8 governance: new namespace runs/r/RPM-000/<ts>/, fresh process,
# independent token cache (sha256 captured by the AF2-R auditator;
# identity vs AF2-D's cache is the expected outcome).
set -euo pipefail

cd /home/andrew-jochl/TORUS

TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="runs/r/RPM-000/${TS}/af2d"
mkdir -p "${OUT_DIR}"

echo "[rpm-000] AF2-D driver SHA (frozen): 7383b57"
echo "[rpm-000] current HEAD: $(git rev-parse --short HEAD)"
echo "[rpm-000] namespace: ${OUT_DIR}"
echo "[rpm-000] starting at $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Launch the t2_ternary arm only on the damaged-PTQ base for n=3 seeds.
.venv/bin/python examples/af2_storage_tournament.py \
  --model allenai/OLMo-1B-0724-hf \
  --target-module model.layers.0.mlp.down_proj \
  --arms t2_ternary \
  --seeds 1,2,3 \
  --n-steps 500 \
  --batch-size 4 \
  --seq-len 128 \
  --lr 1e-3 \
  --momentum 0.9 \
  --grad-clip 1.0 \
  --tasks wikitext,arc_easy,lambada_openai \
  --ids-cache /tmp/wikitext103_train_ids.npy \
  --device cuda \
  --dtype float16 \
  --eval-dtype float16 \
  --damage-ptq \
  --damage-group-size 128 \
  --damage-threshold 0.7 \
  --pre-train-eval \
  --matched-bytes-tolerance-pct 1.0 \
  --out-dir "${OUT_DIR}" \
  2>&1 | tee "${OUT_DIR}/driver.log"

echo "[rpm-000] driver finished at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[rpm-000] running audit..."

.venv/bin/python examples/audit_rpm_000_reproduction.py \
  --aggregate "${OUT_DIR}/aggregate.json" \
  --af2d-reference /home/andrew-jochl/TORUS/research/track-a-residual-ternary/residual-falsification/experiments/AF2-D/runs/20260823T092339Z/af2d/aggregate_corrected.json \
  --manifest /home/andrew-jochl/TORUS/research/residual-pareto/experiments/RPM-000/manifest.yaml \
  --runs-dir "${OUT_DIR}" \
  --out "${OUT_DIR}/rpm000_audit.json"

echo "[rpm-000] audit complete; verdict: $(.venv/bin/python -c "import json; print(json.load(open('${OUT_DIR}/rpm000_audit.json'))['verdict'])")"