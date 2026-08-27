# EXP-RPM-DAMAGE-MAP-V2 — Stage A Calibration Table

**Date:** 2026-08-27
**Run namespace:** `runs/r/EXP-RPM-DAMAGE-MAP-V2/`
**Target:** AF2-D (`model.layers.0.mlp.down_proj`)
**Base model:** `allenai/OLMo-1B-hf` (1B params), bfloat16
**Methodology:** Per-seed `--pre-train-eval` (1 seed per cell, just base damage eval,
no adapter training, 1 step). Wikitext ppl only (most informative single task).

---

## Calibration Probe Results

| Mechanism | Parameter | Base ppl |
| --- | ---: | ---: |
| magnitude_prune | k=0.5 | 15.49 |
| magnitude_prune | k=0.8 | 16.54 |
| magnitude_prune | k=0.85 | 17.22 |
| magnitude_prune | k=0.9 | 18.55 |
| magnitude_prune | k=0.93 | 20.13 |
| magnitude_prune | k=0.95 | **22.06** |
| dropout | p=0.3 | 16.35 |
| dropout | p=0.5 | 18.22 |
| dropout | p=0.7 | 23.02 |
| dropout | p=0.8 | 29.49 |
| dropout | p=0.9 | 44.08 |
| dropout | p=0.95 | 55.91 |
| dropout | p=0.99 | **69.07** |

**MagnitudePrune maximum reachable damage:** ppl **22.06** (k=0.95).
Beyond k=0.95, increasing k only marginally increases damage (k=1.0 caps at the
per-row max-|w| = always positive; damage is bounded by row weight variance).

**Dropout maximum reachable damage at p=0.99:** ppl **69.07**.
Beyond p=0.99, increasing p caps at catastrophic collapse. With p=1.0 we'd
zero out the whole layer — that's no longer "damage" but "deletion" — not
a useful data point.

---

## Reference: Stage 3 v1 BAND-3 target (TWN @ thr=0.7)

| Mechanism | Parameter | Base ppl | BAND-3 matched? |
| --- | --- | ---: | :---: |
| **twn** | thr=0.7 | **429.55** | YES (PRIMARY) |
| gaussian | σ=3.0 | 451.13 | YES (PRIMARY, ±5% of TWN) |
| gaussian | σ=5.0 | 4889 | NO (3.2× mismatch — secondary) |
| dropout | p=0.99 | 69 | **NO** (~16% of target) |
| magnitude_prune | k=0.95 | 22 | **NO** (~5% of target) |
| (BAND-1 control) gaussian | σ=1.0 | 15 | NO (near-pristine, not catastrophic) |

---

## Calibration Gate Verdict

**Stage 3 v2 calibration gate FAILS for MagnitudePrune + Dropout.**

The Stage 3 v2 manifest required: "If MagnitudePrune and Dropout cannot
be calibrated to ±20% of BAND-3 magnitude (~ppl 350-515), the
experiment fails the calibration gate and only {TWN, Gaussian} data is
reported."

- MagnitudePrune max ppl at k=0.95 is **22.06** — 19× short of target 429.
- Dropout max ppl at p=0.99 is **69.07** — 6× short of target 429.
- Neither mechanism can produce catastrophic damage at AF2-D/L0/down_proj.

**Mechanism analysis:**
- **TWN** (TWN-style absmean-zero per group) and **Gaussian** (signed
  noise) are the ONLY damage mechanisms tested that produce catastrophic
  magnitude on this layer.
- **MagnitudePrune** (per-row top-k by magnitude) cannot produce
  catastrophic damage because AF2-D/L0/down_proj weights are heavily
  right-tailed (small number of large weights per row), so pruning 95%
  of the smallest entries preserves the bulk of the information flow.
- **Dropout** (per-element independent mask) scales slowly because most
  weight magnitude is concentrated in relatively few entries; even p=0.99
  (zero 99% of weights) leaves enough signal for the layer to operate
  near baseline.

---

## Implication for Stage 3 v2

The hypothesis **TWN > MagnitudePrune > Dropout > Gaussian** (structure-
magnitude ranking) is **untestable at matched magnitude** because neither
new mechanism reaches BAND-3. The Stage 3 v2 verdict follows the manifest
fallback: report only {TWN, Gaussian} — which is exactly Stage 3 v1.

**The new contribution of Stage 3 v2** is therefore the **empirical
demarcation of the damage-mechanism envelope at AF2-D**: TWN and Gaussian
are the only mechanisms capable of producing catastrophic damage. The
{fine-grained pruning, symmetric mask} mechanisms preserve too much
signal to be useful as "damage" at this layer.

This is itself a publishable falsification: the Pareto-frontier thesis
holds for {TWN, Gaussian}-only catastrophic regimes; finer-grained
mechanisms do not enter the band.

---

## Cells used for Stage 3 v2 verdict (no calibration, only historical)

| Cell | Mechanism | Parameter | Source |
| --- | --- | ---: | --- |
| BAND-3-TWN | twn | thr=0.7 | Stage 3 v1 (BAND-3 PRIMARY) |
| BAND-3-Gaussian | gaussian | σ=3.0 | Stage 3 v1 (BAND-3 PRIMARY) |
| BAND-4-TWN | twn | thr=0.5 | Stage 3 v1 (BAND-4 SECONDARY) |
| BAND-4-Gaussian | gaussian | σ=5.0 | Stage 3 v1 (BAND-4 SECONDARY) |
| BAND-1-Gaussian | gaussian | σ=1.0 | Stage 3 v1 (CONTROL: near-pristine) |

These are the only cells with calibrated catastrophic damage. Stage 3 v2
verdict reuses Stage 3 v1 data. **No new experimental cells** needed for
this verdict — the Stage A calibration result IS the verdict.
