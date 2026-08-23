# EXP-AF-001-R — Verdict — REPRODUCED — A-RP-001 → CONFIRMED_FAIL

**Decision: REPRODUCED.** A-RP-001 transitions to **CONFIRMED_FAIL**.
The equal-training-time branch closes permanently under AF8 governance.

## Hypothesis tested (A-RP-001, threshold preserved verbatim)

The EXP-AF-001 (commit `39be76c`, run `20260822T234553Z`) result
reproduces under AF8 governance: T1+T2 loses to T1-continued on every
capability metric at matched CE, by >2 stderr-of-difference over n=3
seeds, with the same wikitext ppl, arc_easy, and lambada_openai deltas
within ±2 stderr-of-difference of the AF1 arm means.

## Setup (deliberate diff vs AF1)

- New experiment ID: `EXP-AF-001-R` (no supersession of AF1).
- New run namespace: `runs/a/EXP-AF-001-R/20260822T233000Z/`.
- Frozen code: git revision `39be76c` (the AF1 SHA itself — AF1-R is a
  reproduction, not an evolution).
- Fresh Python process on legion, nohup + disown.
- Independently generated wikitext-103 token cache via
  `examples/audit_af1_reproduction.py`. The cache is a deterministic
  function of the wikitext-103 parquet shards + OLMo tokenizer + eot
  append policy; the auditator re-tokenized into a fresh path under
  the AF1-R run namespace, recorded SHA fingerprints for every
  input shard, the cache file, the PID, the UTC timestamp, and the
  AF1 reference SHA. AF8 governance is *traceability*, not byte
  difference — identity is the expected outcome of a clean
  reproduction, not a violation.
- Independently generated eval output (full lm-eval-harness on the
  same three tasks).
- env-lock.txt regenerated at run start.
- Same preregistered thresholds, same n=3 seeds (1, 2, 3), same
  model (allenai/OLMo-1B-0724-hf), same target module, same N=500
  steps, same batch 4, seq 128, same lr 1e-3, same SGD lr/momentum/
  clip, same objective (next-token CE on wikitext-103 train), same
  eval tasks (full wikitext test, arc_easy, lambada_openai), float16
  throughout.

## Results (AF1-R aggregate.json)

| metric         | A (T1-continued)        | B (T1+T2)               |  diff   | se_diff | in_stderrs |
|----------------|------------------------|-------------------------|---------|---------|------------|
| wikitext ppl   | 14.0967 ± 0.0143       | 34.8142 ± 2.2791        | +20.72  | 2.279   | +9.09      |
| arc_easy       | 0.6494 ± 0.0061        | 0.6350 ± 0.0022         | -0.0145 | 0.0065  | -2.23      |
| lambada_openai | 0.6178 ± 0.0085        | 0.5647 ± 0.0010         | -0.0532 | 0.0085  | -6.24      |

## Reproducibility — AF1 vs AF1-R per-arm values (exact bytes)

| seed | arm         | AF1 vs AF1-R lambada  | AF1 vs AF1-R arc_easy | AF1 vs AF1-R wikitext ppl |
|------|-------------|------------------------|------------------------|----------------------------|
| 1    | t1_continued| 0.6211915389093732 — identical | 0.6611952861952862 — identical | 14.0778455309893 — identical |
| 2    | t1_continued| 0.6305065010673394 — identical | 0.6460437710437711 — identical | 14.124774716861591 — identical |
| 3    | t1_continued| 0.6017853677469436 — identical | 0.640993265993266 — identical | 14.08754996074506 — identical |
| 1    | t1_t2       | 0.5656898893848243 — identical | 0.6372053872053872 — identical | 35.56000915199441 — identical |
| 2    | t1_t2       | 0.5625849019988356 — identical | 0.6304713804713805 — identical | 38.33560602281753 — identical |
| 3    | t1_t2       | 0.5656898893848243 — identical | 0.6372053872053872 — identical | 30.547013293953608 — identical |

**Every per-seed value is character-identical to AF1.** The arm means,
stddevs, stderrs, and (B-A) stderr-of-difference match exactly. This
is the strongest possible AF8 outcome: the reproduction is byte-for-
byte the same.

## AUDIT — pass/fail thresholds (preregistered)

**REPRODUCED met.** On every capability metric:

1. **Sign of (B - A) reproduces:**
   - wikitext ppl: +20.72 (AF1 +20.72) — same sign, in A's favor.
   - arc_easy: -0.0145 (AF1 -0.0145) — same sign, in A's favor.
   - lambada_openai: -0.0532 (AF1 -0.0532) — same sign, in A's
     favor.
2. **Magnitude ≥ 2 stderr-of-difference** on every metric
   (9.09 / 2.23 / 6.24 — same as AF1).
3. **Per-seed values byte-identical** to AF1 — even stronger than
   the ±2-stderr reproduction bar.

**NOT REPRODUCED not met.** No metric sign-flips, no magnitude
collapse. AF1-R rules out the only "NOT REPRODUCED" branches:

- sign flip on any capability metric: not observed.
- magnitude drops below 2 stderr-of-difference: not observed.

## Claim state change

- `A-RP-001`: `PROVISIONAL_FAIL / REPRODUCTION_REQUIRED` →
  **`CONFIRMED_FAIL`**. The equal-training-time branch closes
  permanently.
- `claims/A-RP-001.yaml`: state transition recorded; AF1 (commit
  `39be76c`) and AF1-R (commit `4238568`, run
  `20260822T233000Z`) both cited.
- Required-evidence list `[AF1, AF3, AF5, AF8]` is closed for
  A-RP-001: AF1 closed, AF1-R closes AF8 for A-RP-001, AF3/AF5 were
  never registered to A-RP-002/003 transitions.

## Confidence and reproduction

- Confidence: **VERY HIGH**. Per-seed byte identity. The reproducibility
  invariant is stronger than the threshold ("every value matches AF1
  exactly" vs "every value matches AF1 within ±2 stderr").
- Reproduction: rerun `examples/af1_budget_control.py` with the same
  flags on git checkout `4238568` (or any descendant that does not
  touch `af1_budget_control.py`/`eval_lm.py`/`distill_run.py`);
  the wikitext cache fingerprint will reproduce as long as the
  HF parquet shards remain at the recorded SHAs.

## What is now permitted under the freeze

- Register `EXP-AF-002` (equal-storage tournament for A-RP-002),
  the now-central Track-A falsifier per OPERATING-PLAN §11 v2.3.
- Track A discovery-tier experiments (A1 layer sensitivity, A5
  Hadamard, B3 OlMoE) remain permitted at discovery tier only.

## What is still blocked

- Track B B1 oracle gating stays locked. AF2 (A-RP-002) and AF4
  (A-RP-003) remain required before B1 can be revisited; the B1
  unlock rule was rewritten in OPERATING-PLAN §5 v2.3 to substitute
  A-RP-002 PROVISIONAL_PASS for the historical A-RP-001
  prerequisite.

## Artifacts

- `runs/a/EXP-AF-001-R/20260822T233000Z/aggregate.json` — full per-arm
  per-metric statistics.
- Per-(seed, arm): `history.jsonl`, `eval.summary.json`,
  `eval.full.json`, `adapter.npz` (arm B only, 3× 134MB,
  `.gitignore`d locally, SHA-indexed in `ARTIFACTS.json`).
- `provenance.json` — git_sha, frozen revision, host, GPU, command.
- `cache_provenance.json` — AF8 governance record for the wikitext
  token cache (auditator PID, UTC, parquet SHAs, cache SHA, AF1
  reference SHA, identity flag = true).
- `env-lock.txt` — legion venv pip-freeze at run start.
- `driver.log` — full driver transcript.

## What this verdict does NOT close

- A-RP-002 (equal-storage tournament) remains UNTESTED — the
  central Track-A falsifier and the prerequisite substitute
  (per OPERATING-PLAN v2.3 §5) for any future Track-B unlock.
- A-RP-003 (sequential vs joint training) remains UNTESTED — AF4
  is its register.
- The residual-plane *representation* itself is still in play; the
  AF1/AF1-R result closes only the **training-time defense** of it.
  AF2 will test the storage/compute Pareto, which is the engineering
  question that drove the v2.3 decision-axis revision.
