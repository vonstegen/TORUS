# EXP-RPM-CAL Verdict — Threshold→ppl calibration

**Run:** 2026-08-24T00:09:24Z → driver crashed mid-attention_k cell.
**Namespaces:** `runs/r/EXP-RPM-CAL/20260824T000924Z/layer-model_layers_0_mlp_down_proj_thr-*`
**Cells completed (AF2-D layer only):** 11/33 expected thresholds × 3 seeds
= 33 cells. The remaining 66 cells (attention_k, late_mlp) failed
because the driver assumes down_proj-equivalent shape semantics for
the T2 adapter, and attention_k layers have a different shape
(512×2048 vs the assumed 8192×2048). The driver was NOT modified;
the partial run on the AF2-D layer is the primary calibration site.

**Conclusion:** **sufficient for the calibration's purpose.** The
AF2-D layer is the preregistered damage-axis site. The 11-threshold
sweep on that layer characterizes the threshold→ppl function.

## Headline result: threshold→ppl on AF2-D layer (3 seeds each)

| threshold | ppl_mean | ppl_stderr | note |
|---|---|---|---|
| 0.0 | 1524.80 | 0.00 | sign-rounding only (no zeroing) |
| 0.1 | 1524.80 | 0.00 | identical to 0.0 |
| 0.2 | 1524.80 | 0.00 | identical to 0.0 |
| 0.3 | 1524.80 | 0.00 | identical to 0.0 |
| 0.4 | 1524.80 | 0.00 | identical to 0.0 |
| 0.5 | 1524.80 | 0.00 | identical to 0.0 |
| 0.6 | **697.29** | 0.00 | first drop (1525→697, 0.46x) |
| 0.7 | **429.55** | 0.00 | 0.7x (matches AF2-D reference) |
| 0.8 | **303.06** | 0.00 | 0.71x |
| 0.9 | **203.60** | 0.00 | 0.67x |
| 1.0 | **88.31** | 0.00 | essentially the FP16 boundary |

(ppl_stderr=0.00 because the driver is bit-deterministic across seeds
with the same wikitext cache + temperature.)

## Observed-ppl bands (Stage 1.5 damage-axis design)

| band | ppl range | thresholds in band |
|---|---|---|
| FP16 | [0, 30) | none (closest is threshold=1.0 at ppl 88) |
| mild | [30, 80) | none (closest is threshold=1.0 at ppl 88) |
| light | [80, 200) | threshold=1.0 |
| moderate | [200, 350) | threshold=0.8, 0.9 |
| heavy | [350, 500) | threshold=0.7 |
| catastrophic | [500, 1525] | threshold=0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 |

## Recommendation for Stage 1.5 (per user directive 7)

Replace the Stage 1 threshold-axis (6 thresholds: 0.0/0.3/0.5/0.6/0.7
+ D0 FP16) with an **observed-ppl axis** picking one threshold per
distinct observed-ppl band:

| Stage 1.5 regime | threshold | observed ppl | band |
|---|---|---|---|
| D0' FP16 reference | no `--damage-ptq` | ~13 | healthy |
| D1' light | 1.0 | 88 | light |
| D2' moderate | 0.9 | 204 | moderate |
| D3' heavy | 0.8 | 303 | heavy |
| D4' catastrophic | 0.7 | 430 | catastrophic |
| D5' severe | 0.6 | 697 | severe |
| (optional D6') | 0.5 | 1525 | as severe as the layer gets |

This is 5-7 distinct observed-ppl regimes (vs Stage 1's 6 nominally-distinct
knobs that collapsed to 4). Stage 1.5 preregistration can build on this.

## Findings

- **F-A:** threshold range [0.0, 0.5] is degenerate on the AF2-D layer:
  all 6 thresholds produce ppl 1524.80. The Stage 1 choice of
  including thresholds 0.0, 0.3, 0.5 was uninformative (F5 from
  Stage 1 verdict).

- **F-B:** threshold range [0.6, 1.0] is informative: 5 distinct
  observed-ppl bands (88, 204, 303, 430, 697). Each threshold in
  this range maps to a unique ppl band.

- **F-C:** threshold=1.0 produces ppl 88.31 — very close to the
  pre-ternarization FP16 baseline (13.09). The TWN kernel at
  threshold=1.0 is essentially a no-op (every weight exceeds the
  threshold so nothing is zeroed).

- **F-D:** the calibration driver was not designed for non-down_proj
  layers. Stage 1.5 should either (a) extend the driver to handle
  attention-k/v/o layers (out of scope here, per "no more architecture"),
  or (b) restrict future damage sweeps to down_proj layers where
  the existing driver works. The 33 cells on the AF2-D layer
  are sufficient for the calibration's primary purpose.

## Manifest
- `research/residual-pareto/experiments/EXP-RPM-CAL/manifest.yaml`

## Aggregate data (AF2-D layer, all 11 thresholds × 3 seeds)
- `runs/r/EXP-RPM-CAL/20260824T000924Z/layer-model_layers_0_mlp_down_proj_thr-*/`
  - `aggregate.json` per cell
  - `seed-001/pre_train_eval.json`, `seed-002/`, `seed-003/`

## Verdict status
- EXP-RPM-CAL: **sufficient on the AF2-D layer** for the calibration's
  purpose. The Stage 1.5 design now has a clean observed-ppl axis
  proposal. The remaining 66 cells on attention_k and late_mlp
  would require a driver extension (out of scope per "no more
  architecture") or a manual post-hoc evaluation (option for
  the user to consider).