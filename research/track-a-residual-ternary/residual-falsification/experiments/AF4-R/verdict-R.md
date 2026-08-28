# Verdict — EXP-AF-004-R — clean reproduction of EXP-AF-004 (A-RP-003)

**Date:** 2026-08-28
**Run:** `runs/a/EXP-AF-004-R/20260828T132030Z/` (detached worktree
`~/TORUS-af4r` on legion)
**Code revision:** `f1df16515d5e6b6eeb9f126d296d7b682aa82991` (FROZEN —
verified at launch; PYTHONPATH shadowed any installed `torus`)
**Reproduces:** EXP-AF-004 @ `runs/a/EXP-AF-004/20260828T121414Z/`
**Audit:** `runs/a/EXP-AF-004-R/20260828T132030Z/audit.json`
(independent recomputation; driver's aggregate.json never trusted as
verdict input)

## AF8 governance checklist (all verified)

- New experiment ID and namespace (`runs/a/EXP-AF-004-R/<ts>/`); no
  reuse of AF4's namespace. One writer per namespace.
- Detached git worktree checked out at the frozen SHA; HEAD verified
  equal to `f1df165…` at launch. No driver/threshold/hyperparameter
  changes.
- Fresh Python processes (nohup + disown).
- Independently generated wikitext-103 token cache at a new path,
  re-tokenized from the HF parquet shards by the auditor
  (`cache_provenance.json`: shard sha256s, cache sha256, tokenizer
  id, auditor PID, UTC). AF4's cache file was never read.
  `af4_reference_identity: true` — independent re-tokenization
  produced the same content sha256, the expected provenance notary
  for a deterministic pipeline.
- Independently generated eval output (full lm-eval-harness on the
  same three tasks).
- env-lock.txt regenerated at run start.

## Reproduction acceptance (frozen rule from the manifest)

1. **Run integrity — PASS.** 9/9 (arm, seed) runs complete;
   freeze_check true on all 3 seq runs; no NaN/inf in any history
   record; deployed bytes identical to AF4's per arm
   (seq = joint = 8,912,896; t1_only = 4,456,448).
2. **Decision replay — MATCH.** Independently recomputed from raw
   per-seed values on both sides: FAIL / joint superior, with
   z = −5.2041 (wikitext), −0.8966 (arc_easy), −21.8830
   (lambada_openai) for the sequential arm — AF4's z-scores to ~9
   significant digits (wikitext differs at the 9th decimal:
   −5.2041298086 vs −5.2041298059).
3. **Statistical agreement — PASS.** Every arm × capability metric
   mean within ±2 combined stderrs of the AF4 arm means.

Arm means ± stderr (AF4-R), with AF4's in parentheses:

| arm     | wikitext ppl         | arc_easy              | lambada_openai        |
|---------|----------------------|-----------------------|-----------------------|
| seq     | 24.919 ± 0.639 (=)   | 0.5610 ± 0.0039 (=)   | 0.4432 ± 0.0009 (=)   |
| joint   | 21.437 ± 0.198 (=)   | 0.5689 ± 0.0079 (=)   | 0.4684 ± 0.0007 (=)   |
| t1_only | 19.298 ± 0.977 (=)   | 0.6547 ± 0.0006 (=)   | 0.5927 ± 0.0011 (=)   |

**Byte-identity observation: false.** Per-seed task records are not
bit-identical at full float precision (the pipeline is deterministic
to ~9 significant digits, not to the last bit). Per the frozen
reproduction rule (user directive 2026-08-28), byte-identity is
neither required nor sufficient; it is recorded as a provenance
observation only. The reproduction requirement — decision replay
under the frozen formulas plus statistical agreement — is what was
evaluated, and it passed.

## Decision

**REPRODUCED.** EXP-AF-004's verdict stands under AF8 governance.
**A-RP-003 → CONFIRMED_FAIL**: sequentially freezing T1 before
learning T2 is materially inferior to matched joint training at the
AF site (model.layers.0.mlp.down_proj, OLMo-1B-0724-hf) under matched
budget, storage, and objective. The sequential-freeze hypothesis is
closed; the curriculum may be retained only where simpler, never as a
mechanism claim.

The t1_only dominance is recorded as context (per the frozen design)
and is NOT promoted to a claim from this reproduction; it may
justify its own preregistered experiment (does the second plane
actively harm, is it redundant, or is it useful only outside the
present AF site?).

## Confidence and reproduction status

CONFIRMED. Two independent run namespaces, two independent
processes, independent token caches and eval outputs, identical
conclusion at confirmation-tier n=3 both times.

## Next permitted experiment

- EXP-AF-003 (AF3 initialization robustness) — next in the A-F suite
  per user steering (2026-08-28): if the surviving A-RP-002
  phenomenon is initialization-fragile, broader context testing
  (AF6) is less valuable.
- EXP-AF-006 (AF6 dataset/context robustness) after AF3.
- Track B reassessment only after AF3/AF6 per the §5 unlock rules.

## Experiments explicitly blocked by this result

- Routine T3/T4 sequential stacking experiments premised on the
  freeze-curriculum mechanism (already locked by the unlock rules;
  now contradicted at the representation level for this regime).
