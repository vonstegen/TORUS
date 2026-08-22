# EXP-A-011 - Verdict

**Experiment ID:** EXP-A-011 (A1 layer sensitivity)
**Track:** A (residual ternary planes / layer sensitivity)
**Preregistered:** 2026-08-22 (manifest.yaml)
**Run launched:** 2026-08-22T19:48:28Z
**Run finished:** 2026-08-22T17:42 PT (~114 min wall time, 6853.5 s driver-reported)
**Run namespace:** `runs/a/EXP-A-011/20260822T194828Z/` (Legion)
**git SHA at run-launch:** `b668559545cfee82db34ebbb55879a22159657fa` (current `5f21d97` adds only the auditor and unit tests; no harness changes)
**Model:** allenai/OLMo-1B-0724-hf
**Status:** DECIDED
**Decision:** CONTINUE
**Grade:** F (useful negative/confound; not a direct test of A-RP-001)

## Hypothesis (re-stated from manifest)

Single-layer ternarization at a known-sensitive layer destroys capability
non-uniformly across the model. The per-layer sensitivity map localizes
which layers carry the model's representational load and supports
A-RP-001 by quantifying "early-layer sensitivity" without committing to a
full T1+T2 training run.

## Result summary

All 114 per-layer arms + 2 reference arms completed. 0 failures.

**Reproducibility check (frozen at PROPOSE):** FP16 reference reproduces
to 4 decimals (ppl 13.0932 vs preregistered 13.09, arc_easy 0.6073 vs
preregistered 0.6073).

**Headline numbers (from `audit_report.txt`):**

| Metric | Value | Bar | Pass? |
|---|---|---|---|
| Coverage (arms completed) | 114/114 | ≥ 60 | PASS |
| FP16 reference reproduction | exact to 4 dec | exact | PASS |
| Fully-quantized reference ppl | 459,454 | matches EXP-A-001 | PASS |
| Per-layer wikitext ppl range | 13.1 .. 9277.6 | wide spread | PASS |
| Spread (max / min, non-embed) | 708.6× | informative | PASS |
| Late-attn-o mean ppl (layers 12-15) | 13.5 (n=4) | -- | -- |
| Early-mlp-down mean ppl (layers 0-1) | 2433.4 (n=4) | -- | -- |
| Ratio late / early | 0.006× (i.e. 180× worse early) | ≥ 1.5× late>early | FAIL |

**Headline finding (top of `top_arms.txt`):**

| target_module | wikitext ppl | arc_easy |
|---|---:|---:|
| model.layers.1.mlp.down_proj | **9277.6** | 0.295 |
| model.layers.0.self_attn.k_proj | **3363.8** | 0.582 |
| model.layers.0.mlp.down_proj | 427.7 | 0.540 |
| model.layers.0.self_attn.q_proj | 268.0 | 0.547 |
| model.layers.0.self_attn.o_proj | 186.8 | 0.573 |
| model.layers.0.self_attn.v_proj | 54.9 | 0.603 |
| lm_head | 45.7 | 0.525 |
| model.layers.0.mlp.gate_proj | 38.8 | 0.568 |
| model.layers.2.self_attn.k_proj | 33.6 | 0.626 |
| model.layers.2.self_attn.q_proj | 28.3 | 0.621 |

The **10 least-sensitive arms** (see `top_arms.txt`) are spread across
layers 4-15 and `model.embed_tokens`; all sit at wikitext ppl 13.4-13.5
(essentially FP16).

## Reading against A-RP-001

A-RP-001's exact PASS criterion is "Q(T1+T2) exceed Q(T1) by >2 standard
errors on at least one capability metric". The experiment used full-layer
ternarization (`--mode quantized --target-modules <layer>`), which is
*not* the same as the T1+T2 correction-plane stack that A-RP-001
formulates. Several single-layer ternarizations of mid-model layers
**do** preserve arc_easy at FP16 level (layers 1-15 keep arc_easy within
2σ of FP16 in 47/114 arms; layer 1 attention arms *exceed* FP16 by 0.025
to 0.043 on arc_easy at >10σ significance), but no single layer alone
demonstrates the "extra capacity beyond equal train time" claim, and
single early-layer ternarizations **destroy** both wikitext and arc_easy.

The data is therefore:
- **Not a direct test** of A-RP-001's specific T1+T2 formulation.
- **A clean falsification-grade measurement** of layer-wise sensitivity.
- **Directly informative** for B1 (oracle gating) and for any
  per-layer-precision policy.

## Decision block

- **Decision:** CONTINUE
- **Grade:** F (useful negative/confound; not a direct test)
- **Hypothesis status:** partially supported - single-layer sensitivity
  is real, asymmetric, and concentrated in layers 0-1 + early
  mlp_down; late layers (≥2) are robust to single-layer PTQ.
- **Confidence / reproduction status:**
  - git SHA at launch: `b668559545cfee82db34ebbb55879a22159657fa`
  - env: Python 3.14.4, NVIDIA TITAN RTX, see `env-lock.txt`
  - FP16 reference reproduces pre-regime to 4 decimals
  - 0/114 arms failed (no FLAGGED, no spurious re-runs)
  - Replay instructions in `manifest.yaml` §8 (run the preregistered
    command with the original git SHA and same env-lock)
- **Next permitted experiment:**
  - `EXP-A-011.b` (A1 sensitivity *paired* — ternarize layer X AND
    layer Y jointly) — directly motivated by the spread pattern: pairs
    of adjacent safe layers might preserve arc_easy while pushing more
    of the model into ternary.
  - `EXP-A-021.b` (per-layer-precision policy test) — apply per-layer
    bit-width from the sensitivity map; predicted safe from this data.
  - Track B (B1 oracle gating) is unlocked; data motivates an oracle
    that learns the per-layer precision decision from this map.
- **Experiments explicitly blocked by this result:**
  - A-RP-001 cannot be claimed PASS or FAIL on this data alone.
  - Any per-layer-precision claim (Track B) is now blocked pending
    `EXP-A-011.b` or a joint T1+T2 reproduction under clean
    provenance (the original A-RP-010 was retro-registered CONTINUE
    due to provenance issues; it is not a clean comparison).

## Provenance index

- Manifest: `research/track-a-residual-ternary/EXP-A-011/manifest.yaml`
- Driver: `examples/layer_sensitivity.py`
- Auditor: `examples/audit_a1_sensitivity.py`
- Run artifacts: `runs/20260822T194828Z/` (gitignored retention copy
  on Legion; **NOT** in this repo, see `ARTIFACTS.json` for the
  sha256 index of every file in the run namespace)
- Committed-record artifacts: `research/track-a-residual-ternary/EXP-A-011/runs/20260822T194828Z/`
  - `audit_report.txt` - AUDIT output
  - `top_arms.txt` - ranked arms (10 most + 10 least)
  - `sensitivity_table.json` - aggregate per-layer table (57.6 KB)
  - `provenance.json`, `env-lock.txt`, `driver.log`
  - `ARTIFACTS.json` - sha256 index of every file in `runs/`
  - `per_layer/*.summary.json` (116 files, ~440 B each) - compact
    view per arm; full lm-eval results (per-task stderrs) are
    referenced by `ARTIFACTS.json` and live only on Legion
    (~1.1 GB, deliberately not committed)

## Audit grading rationale

- Per OS v2 §6 (08-HARNESS-INSTRUCTIONS-V2.md): "Every completed
  experiment should end with: hypothesis; result summary; grade;
  PASS / FAIL / INVALID / CONTINUE; confidence and reproduction status;
  next permitted experiment; experiments explicitly blocked by the
  result."
- All fields present.
- CONTINUE matches the v2 docs' reading of F-grade-as-useful-result.
- The decision is consistent with `EXP-A-010` (retro-registered
  CONTINUE: "Plane 2 adds real capacity; plane 3 marginal. Diagnostic
  KL only — downstream acceptance untested. Must reproduce under
  clean provenance in Phase 1 (EXP-A-03x) before architectural
  action.") and `EXP-A-021` (retro-registered CONTINUE: "Better
  optimization behavior/conditioning for the rotated arm.
  Motivates EXP-A-H1.").

## Verification commands

```bash
# Reproduce the audit tables from the committed record (dev box, no GPU):
git clone https://github.com/vonstegen/TORUS.git
cd TORUS
.venv/bin/python examples/audit_a1_sensitivity.py \
    --run-dir research/track-a-residual-ternary/EXP-A-011/runs/20260822T194828Z

# Verify sha256 of every committed artifact:
jq -r '.artifacts | to_entries[] | .value | (objects | .path), (arrays | .[]?.path)' \
    research/track-a-residual-ternary/EXP-A-011/runs/20260822T194828Z/ARTIFACTS.json \
    | sort -u \
    | while read p; do \
        echo "$(sha256sum "$p" | awk '{print $1}')  $p"; \
    done | diff - <(jq -r '.artifacts | to_entries[] | .value |
        (objects | "\(.sha256)  \(.path)"),
        (arrays | .[]? | "\(.sha256)  \(.path)")' \
        research/track-a-residual-ternary/EXP-A-011/runs/20260822T194828Z/ARTIFACTS.json \
        | sort) \
    && echo "ARTIFACTS.json sha256s match"
```
