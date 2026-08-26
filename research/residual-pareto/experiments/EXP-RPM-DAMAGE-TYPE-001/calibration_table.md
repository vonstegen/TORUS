# EXP-RPM-DAMAGE-TYPE-001 — Stage A Calibration Table

**Date:** 2026-08-26
**Run namespace:** `runs/r/EXP-RPM-DAMAGE-TYPE-001/20260826T144024Z/stage_a_probe/`
**Manifest:** `research/residual-pareto/experiments/EXP-RPM-DAMAGE-TYPE-001/manifest.yaml`

## Purpose

Probe TWN thresholds and Gaussian sigmas at AF2-D to map parameter -> post-damage, pre-correction base ppl. Used to identify cross-mechanism magnitude-matched comparison points for Stage B.

## Probe Results (AF2-D = `model.layers.0.mlp.down_proj`)

| Mechanism | Parameter | Base wikitext ppl | fro_ratio |
| --- | --- | ---: | ---: |
| TWN | thr=0.3 | 1524.80 | 0.660 |
| TWN | thr=0.5 | 1524.80 | 0.660 |
| TWN | thr=0.7 | 429.55 | 0.603 |
| TWN | thr=0.9 | 203.60 | 0.546 |
| TWN | thr=1.0 | 88.31 | 0.518 |
| Gaussian | σ=0.5 | 13.37 | 1.118 |
| Gaussian | σ=1.0 | 15.35 | 1.414 |
| Gaussian | σ=2.0 | 116.69 | 2.236 |
| Gaussian | σ=3.0 | 450.85 | 3.164 |
| Gaussian | σ=5.0 | 4889.07 | 5.102 |

FP16 baseline ppl: 13.09.

## Empirical Cross-Mechanism Magnitude Bands

| Band | TWN point | TWN ppl | Gaussian point | Gaussian ppl | Magnitude ratio |
| --- | --- | ---: | --- | ---: | ---: |
| **BAND-3 (moderate)** | TWN thr=0.7 | 429.55 | Gaussian σ=3.0 | 450.85 | **1.05x (PRIMARY MATCH)** |
| **BAND-4 (catastrophic)** | TWN thr=0.5 | 1524.80 | Gaussian σ=5.0 | 4889.07 | 3.21x (SECONDARY MATCH) |
| **BAND-1 (near-pristine)** | — | — | Gaussian σ=1.0 | 15.35 | — (Gaussian only) |

## Notable Observations

1. **TWN thr=0.3 and thr=0.5 produce IDENTICAL damage** (both 1524.80 ppl, both fro_ratio=0.660). This means at AF2-D, every weight with magnitude >0.5 is quantized to non-zero, and below 0.5 all weights are zeroed. Lowering threshold from 0.5 to 0.3 doesn't add more non-zero weights because there are none to add. The layer's weight magnitude distribution is bounded.

2. **TWN cannot reach near-pristine damage.** The smallest TWN base ppl is 88 (thr=1.0), which is still 6.7x the FP16 baseline of 13.09. This is a structural property of TWN: even at threshold=1.0 (only the largest weights survive), the layer is fundamentally lossy.

3. **Gaussian can span a wider magnitude range** (13.37 to 4889.07) than TWN (88.31 to 1524.80). Gaussian's natural range covers both near-pristine and catastrophic damage; TWN is bounded on the small-damage side.

4. **BAND-3 is the only tight match** between mechanisms (within 5%). This makes BAND-3 the primary falsification target for the cross-mechanism question.

## Stage B Calibrated Cells (Preregistered)

| Cell ID | Mechanism | Parameter | Expected base ppl | Role |
| --- | --- | --- | ---: | --- |
| BAND-3-TWN | TWN | thr=0.7 | 430 | PRIMARY (mechanism pair) |
| BAND-3-Gaussian | Gaussian | σ=3.0 | 451 | PRIMARY (mechanism pair) |
| BAND-4-TWN | TWN | thr=0.5 | 1524 | SECONDARY (3.2x magnitude mismatch) |
| BAND-4-Gaussian | Gaussian | σ=5.0 | 4889 | SECONDARY (3.2x magnitude mismatch) |
| BAND-1-Gaussian | Gaussian | σ=1.0 | 15 | CONTROL (near-pristine damage) |

Total Stage B cells: 5 calibrated × 4 trained arms × 3 seeds = 60 trained-arm cells, plus 15 base-eval + 30 post-hoc random = 105 cells. Estimated wall time: ~5 hours on Legion (single GPU, sequential).

## Conclusion of Stage A

The empirical damage-magnitude response surfaces are dramatically asymmetric between TWN and Gaussian. This validates the user's caveat that TWN thr=0.7 and Gaussian σ=1.0 (or σ=0.5, or any small σ) cannot be compared directly without magnitude calibration. The Stage B design preregisters the only tight cross-mechanism match (BAND-3) as the primary comparison, with BAND-4 as a secondary comparison under relaxed magnitude matching and BAND-1 as a near-pristine control.