# EXP-AF-001 — Verdict — A-RP-001 FAIL (T2 loses to T1-continued at matched budget)

**Decision: DECIDED FAIL.** A-RP-001 falsified under matched CE at
> **Post-script (rev 2.3, 2026-08-22).** The verdict was originally
> written with the v2.2 lifecycle in force, which had me set
> A-RP-001 to `DECIDED FAIL` and conclude that B1 is "unreachable".
> v2.3 of the operating plan (a) makes explicit that a
> confirmation-tier ≥3-seed matched-control result can promote
> `TESTING → PROVISIONAL_FAIL` in one step with `reproduction:
> REQUIRED` set at the same transition; (b) rewrites B1's unlock
> rule to use A-RP-002 (storage) rather than A-RP-001 (training
> time), removing the "unreachable" framing; (c) reframes Track A's
> primary decision axis as capability-vs-cost-vector, with
> equal-storage (AF2) as the now-central Track-A experiment.
> `EXP-AF-001-R` (clean reproduction of this experiment) is the
> required next experiment before A-RP-001 transitions to
> `CONFIRMED_FAIL`. Track B remains locked either way.

## Hypothesis tested (A-RP-001)
"At matched training budget, a frozen FP16 base plus a single ternary
correction plane on `model.layers.0.mlp.down_proj` (T1+T2) provides
useful capability beyond continuing to train the full FP16 base
(T1-continued): Q(T1+T2) exceeds Q(T1 continued) by >2 standard errors
on at least one capability metric, with no regression beyond 1 std-error
on the others."

## Setup
- Model: allenai/OLMo-1B-0724-hf, single GPU (cuda:0, TITAN RTX).
- Arm A (`t1_continued`): full FP16 base, ALL weights trainable, 500
  steps of SGD (lr 1e-3, momentum 0.9, clip 1.0) with next-token CE on
  wikitext-103 train. Training dtype float32; eval dtype float16.
- Arm B (`t1_t2`): same base, requires_grad_(False) on every base
  param, SGD on the T2 STE latent (residual plane on
  `model.layers.0.mlp.down_proj`, the worst-tolerance layer per
  EXP-A-011 and the EXP-A-03x recovery site). Same N=500, same
  objective, same eval dtype.
- Identical batches: shared cached wikitext-103 token stream, same
  seed per (arm, seed). The driver pins this by construction
  (test: `tests/test_af1_budget_control.py`).
- Seeds: 1, 2, 3. Cap: 8 GPU-hours (used: ~50 min wall, well inside).
- Eval: full wikitext test, full arc_easy, full lambada_openai.
  Float16 throughout, matching the EXP-A-001 reference ladder.

## Results (mean ± stderr over n=3 seeds; B-A difference in stderr units)

| metric         | A (T1-continued)        | B (T1+T2)               |  diff   | se_diff | in_stderrs |
|----------------|------------------------|-------------------------|---------|---------|------------|
| wikitext ppl   | 14.0967 ± 0.0143       | 34.8142 ± 2.2791        | +20.72  | 2.279   | **+9.09**  |
| arc_easy       | 0.6494 ± 0.0061        | 0.6350 ± 0.0022         | -0.0145 | 0.0065  | **-2.23**  |
| lambada_openai | 0.6178 ± 0.0085        | 0.5647 ± 0.0010         | -0.0532 | 0.0085  | **-6.24**  |

(Person-on-person directional convention: ppl lower is better, acc
higher is better. A sign of "in_stderrs" means *in favor of A*.)

## Audit

- PASS bar (B beats A on ≥1 capability by >2 stderr, no regression
  >1 stderr on others): **NOT MET**. B is worse than A on all three
  metrics.
- FAIL bar (A beats B by >2 stderr on any capability): **MET on all
  three**. The claim is falsified at this budget and this layer.
- Kill criteria: none fired. No NaN, no contamination, base froze in
  arm B by construction (matches the freeze-invariant test), both arms
  trained cleanly.
- Direction-sense: T1-continued improved over the FP16 baseline on
  arc_easy (+0.054) and lambada (+0.012) — 500 steps of full-base SGD
  at lr 1e-3 helped downstream, *and* hurt wikitext ppl slightly
  (+1.0). T2's matched-budget loss therefore reflects an architectural
  deficit, not a transient training signal.

## Interpretation

The claim A-RP-001 ("a sequentially trained T2 plane provides useful
capacity beyond equivalent additional training time") is not the
correct way to bound T2's value, at least not at this budget and on
this layer. Under matched optimizer steps and identical data, training
the full base beats training only the T2 latent by a wide margin.
Two readings, in increasing scope:

1. **T2 buys capacity orthogonal to what continued training buys.**
   The continuous-budget tradeoff is the relevant axis, not the
   equal-step axis: a 1.58-bit residual plane replaces ~16M FP16
   params (2.4 GB → 2.7 MB) at fixed *byte cost*, but the bytes come
   from a different budget dimension. EXP-AF-002 (equal *storage*
   budget: T1+T2 vs T1+full-base at matched *parameter count* added
   per step) is the natural follow-up.
2. **T2 needs a richer training signal to beat cold-start full-base
   SGD.** EXP-A-03x used KD against a frozen teacher; AF1 uses plain
   CE. KD provides per-token supervision, plain CE provides only
   next-token labels. EXP-AF-003 (matched CE but trained with KD
   against the frozen pre-trained T1, arms A and B both) would
   separate the architecture question from the signal question. Note
   for arm A: KD-against-frozen-T1 is degenerate at init (KD loss = 0
   since student == teacher), so this control has a confounding arm-A
   problem — punt that to AF3 with a deliberate A-arm strategy.

The simplest, most useful next experiment is **AF2: equal-storage
control on the same layer** (T1+T2 vs a small fully-trainable FP16
adapter sized to match the T2 latent budget, head-to-head under
matched compute). That is the claim the original A-RP-001 quant
threshold was probably trying to bound, and it is the experiment that
directly informs the "should we replace weights with T2 residual"
engineering decision.

## Claim state change

- `A-RP-001`: DECIDED **FAIL**. Hypothesis text is preserved as a
  record of what was falsified.
- `claims/A-RP-002.yaml` and `claims/A-RP-003.yaml`: remain UNTESTED.
  AF2 and AF3 will retarget them.

## Confidence and reproduction

- Confidence: **HIGH** for the FAIL result. Three seeds × two arms =
  6 runs, each independently evaluated on three full tasks, free of
  contamination (cached token stream sha-locked; one writer, three
  readers), no kill criterion tripped, all stderrs small.
- Reproduction: identical `git checkout 39be76c` + the documented
  command in `provenance.json` will reproduce the same numbers (the
  shared wikitext-103 token cache is regenerated reproducibly from
  the seed; no conda-specific state).

## What is now permitted under the freeze

- Register EXP-AF-002 (equal-storage control).
- The discovery-tier experiments (A1 layer sensitivity is already
  registered; A5 Hadamard; B3 OlmoE) remain permitted at discovery
  tier only.

## What is still blocked

- A3 trainer runs and any Phase 2+ training (Track B stays locked
  until at least one of {A-RP-002, A-RP-003} survives a confirmation
  pass).

## Artifacts

- aggregate.json, per-(seed,arm)/{history.jsonl, eval.summary.json,
  eval.full.json}, env-lock.txt, driver.log, ARTIFACTS.json
  (sha256-indexed), provenance.json (git_sha + host + GPU + python).
- Large arm-B adapters (3×129 MB) are gated behind
  `runs/*/seed-*/t1_t2/adapter.npz` in `.gitignore` and indexed only
  by SHA in ARTIFACTS.json.
