# EXP-A-03x — Verdict

**Experiment ID:** EXP-A-03x (A3 sequential functional correction, discovery)
**Track:** A (residual ternary planes)
**Preregistered:** 2026-08-22 (manifest.yaml, commit d762214)
**Run 1 (INVALID):** `runs/a/EXP-A-03x/20260822T220559Z/` — autograd STE
zero-gradient bug; see `runs/INVALID.md`; fixed in commit eac2c04
**Run 2 (this verdict):** `runs/a/EXP-A-03x/20260822T225835Z/`
**git SHA at run:** `eac2c04`
**Model:** allenai/OLMo-1B-0724-hf
**Status:** DECIDED
**Decision:** PASS
**Grade:** C (discovery tier, single seed, single layer, 500 steps —
pipeline confirmation, not a falsification-grade claim test)

## Hypothesis (re-stated from manifest)

A single sequentially-trained ternary correction plane on the frozen
FP16 base at `model.layers.0.mlp.down_proj` recovers capability toward
FP16 (wikitext ppl ≤ 200, arc_easy ≥ 0.55) with a real training-log KL
decline.

## Result summary

Run 2 (clean provenance, harness fix eac2c04):

| Metric | Untrained PTQ arm (EXP-A-011, same layer) | Trained T1+T2 (this run) | FP16 reference (EXP-A-011) | Bar | Verdict |
|---|---:|---:|---:|---|---|
| wikitext word_perplexity | 427.71 | **41.44** | 13.09 | ≤ 200 | **PASS** |
| arc_easy acc | 0.5396 | **0.6313** | 0.6073 | ≥ 0.55 | **PASS** |
| training KL (diagnostic only) | — | 1.5096 → 0.0714 (500 steps) | — | monotone decline | real |

- wikitext ppl: **10.3× recovery** from the untrained PTQ arm
  (427.71 → 41.44); the preregistered bar was a 2.14× recovery.
- arc_easy: **+0.092** over the untrained arm, and **+0.024 above the
  FP16 baseline** itself (0.6313 vs 0.6073; ~4.8× the EXP-A-001
  per-eval stderr of ~0.005 on this bounded subset — consistent with
  EXP-A-011's observation that several single-layer arms exceed FP16
  arc_easy on this subset; treat the above-FP16 delta as subset noise
  until confirmed at full limit).
- Training curve: 1.5096 → 0.6521 (step 25) → 0.3434 (50) → 0.1675
  (75) → plateau 0.07–0.22. Monotone decline, not noise (contrast
  run 1: 0.38–1.63 random bounce).
- Latent movement: primary plane differs from the original FP16
  weights by 1.01% L1; residual plane untouched (curriculum `1:500`
  keeps it out of the forward, by design).

## Run 1 (INVALID) — what it teaches

Run 1 (`20260822T220559Z`) completed 500 "training" steps and produced
an adapter that evaluated **bit-identical** to the untrained PTQ arm
(ppl 427.7117, arc 0.5396). Audit found the autograd path dead:
`_make_forward_stub` quantized via numpy, severing the autograd graph;
`torch.autograd.grad(..., allow_unused=True)` returned None for every
STE weight; grads were zero-filled; SGD stepped zeros. Two further
bugs surfaced in the same audit: an interleave-order scramble in
`_autograd_grads` (invisible while grads were None) and a
train/eval `calibrate_norm` regime mismatch. All three fixed in
commit eac2c04 with regression tests (`tests/test_ste_torch.py`,
5 tests including an end-to-end latent-moves-after-fit test on a
two-STE different-shape toy adapter). Run 1 is retained in its
namespace with `INVALID.md`; the discovery-tier lesson — **a clean
loss curve and a completed run is not evidence of training; verify
the weights moved** — is now pinned by the test suite.

## Reading against the claims

- **A-RP-001** (T2 adds useful capacity beyond equal training time):
  NOT tested here. The AF1 equal-training-budget control (T1-continued
  vs T1+T2) is the actual test. This experiment shows the pipeline
  works and T2 can repair a deliberately damaged layer — a necessary
  precondition, not the claim.
- **A-RP-002 / A-RP-003:** not addressed.
- Per roadmap CP1.2, all three claims move `UNTESTED → TESTING`.

## Decision block

- **Decision:** PASS
- **Grade:** C (discovery tier; n=1 seed as preregistered; single
  layer; 500 steps; bounded eval subset `--limit 200`)
- **Hypothesis status:** supported — T1+T2 training recovers
  capability on the worst-tolerance layer, both capability metrics
  clearing the preregistered bars with wide margins.
- **Confidence / reproduction:** clean provenance (fresh namespace,
  single writer, provenance.json + env-lock.txt + ARTIFACTS.json with
  sha256 index); FP16/PTQ-arm references reproduced from EXP-A-011's
  committed record; latent movement verified numerically. Reproduce:
  check out eac2c04, run the preregistered command in
  `runs/20260822T225835Z/provenance.json`.
- **Next permitted experiments:**
  - `EXP-AF-001` (AF1 equal-training-budget control) — the actual
    A-RP-001 test. **Unblocked** (CP1.2 passes → G1→2 opens).
  - `EXP-A-011.b` (paired-layer ternarization) — still open from
    EXP-A-011's verdict.
  - Track B (B1 oracle gating) — unlocked by EXP-A-011 + this result.
- **Experiments explicitly blocked:**
  - Any architectural action on the basis of this result alone
    (single seed, single layer, discovery tier).
  - Claiming A-RP-001 PASS/FAIL from this data — the equal-budget
    control does not exist yet.

## Provenance index

- Manifest: `research/track-a-residual-ternary/EXP-A-03x/manifest.yaml`
- Run 2 artifacts: `runs/20260822T225835Z/` (history.json,
  eval-t1t2.summary.json, eval.log, driver.log, provenance.json,
  env-lock.txt, adapter.npz.meta.json, ARTIFACTS.json)
- adapter.npz (134 MB): Legion `runs/a/EXP-A-03x/20260822T225835Z/`,
  sha256 in ARTIFACTS.json; not committed.
- Run 1 INVALID record: `runs/INVALID.md` (+ Legion namespace).
- Harness fix: commit eac2c04 (`torus/train/ste.py`,
  `torus/train/hf_adapter.py`, `torus/train/loop.py`,
  `examples/distill_run.py`, `tests/test_ste_torch.py`).
