# EXP-A-001 — Verdict

**Decision: PASS** (all three arms) · 2026-08-22 · run namespace
`runs/a/EXP-A-001/20260822T182528Z` on legion · git SHA `ee62c45`
(descendant of `research-baseline-2026-08`)

## Results vs. frozen thresholds

| arm | task | metric | EXP-A-001 | historical | threshold | verdict |
|---|---|---|---|---|---|---|
| fp16 | arc_easy | acc | 0.60732 | 0.6073 | within 2× stderr | PASS |
| fp16 | lambada_openai | acc | 0.60955 | 0.6095 | within 2× stderr | PASS |
| fp16 | wikitext | word_ppl | 13.0932 | 13.09 | within 2× stderr | PASS |
| ptq-uncalibrated | arc_easy | acc | 0.25842 | 0.2584 | — | matches |
| ptq-uncalibrated | lambada_openai | acc | 0.00116 | 0.0012 | — | matches |
| ptq-uncalibrated | wikitext | word_ppl | 459,454 | 465,097 | within 1 order of magnitude | PASS (−1.2%) |
| ptq-calibrated | wikitext | word_ppl | 89,557.06 | ~89,557 | within 1 OOM **and** ≥3× better than uncalibrated | PASS (5.13×) |

FP16 values match the pre-regime numbers to four decimal places — the
baseline ladder root is exactly reproducible under v2 provenance.

## Anomaly recorded (observation, not inference)

The calibrated arm is 5.13× better on WikiText perplexity but slightly
*worse* on ARC-E (0.25084 vs 0.25842) and LAMBADA (0.00019 vs 0.00116)
than the uncalibrated arm. Consistent with the v2 assessment that norm
calibration fixes one failure mechanism without validating the
representation; no claim state is affected.

## Audit notes

- Per-task lm-eval stderrs were not persisted (eval_lm.py writes a
  one-metric-per-task summary; raw `simple_evaluate` output lives only in
  process memory). The 2×-stderr criterion is trivially satisfied by exact
  reproduction, but future runs should dump the full results dict —
  recorded as an artifact-rigor improvement for the next experiment's
  manifest, not repaired in place.
- n=1 per arm as preregistered (PTQ + lm-eval deterministic given pinned
  model revision and dataset versions; model from the shared read-only HF
  cache).
- GPUs verified idle before launch; no concurrent writers to the namespace.
- The runner's initial `git checkout` aborted on a stray drill file; the
  arms therefore ran at `ee62c45` rather than `176d87f`. Code content is
  identical between the two commits (the delta is the drill report file
  only); provenance.json records the true SHA. Process bug fixed (stray
  file removed, checkout re-verified).

## Consequence

CP0.3 = PASS. With CP0.1 (wheel smoke test) and CP0.2 (provenance drill)
already PASS, registry live, and the freeze active: **gate G0→1 opens.**
Phase 1 (Track A discovery: A1 layer sensitivity, A2 oracle residual, A3
sequential correction) is unlocked. Smallest justified next experiment:
EXP-A-011 (A1 layer sensitivity) — cheapest measurement that localizes
where ternarization destroys behavior, and it needs no training.

**Strongest supported conclusion:** the pre-regime baseline numbers are
real and reproducible under clean provenance; the ternary PTQ collapse and
the norm-calibration partial recovery are both confirmed as properties of
the representation, not of a broken environment.
**What remains unknown:** everything downstream of the baselines — A1/A2/A3
and the A-F suite carry that weight.
