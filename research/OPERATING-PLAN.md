# TORUS Operating Plan — How Research Runs From Here

**Revision:** 2.3 (2026-08-22) — v2.2 hold plus the AF1 lesson. Two
changes:
(a) Track A primary decision axis is now **storage / compute / energy
Pareto efficiency**, not "matched-training-time capacity" (see §11
below). A-RP-001's wording conflated training-time and
parameter-count as the cost axis; AF1 caught it and the v2 suite
needs to keep that catch possible.
(b) Claim-lifecycle clarification on `TESTING → PROVISIONAL_*`
transitions: a confirmation-tier result (≥3 seeds, preregistered
thresholds, matched-control design) moves `TESTING` directly to
`PROVISIONAL_PASS` or `PROVISIONAL_FAIL`. `REPRODUCTION_REQUIRED` is
the next state on the same lifecycle step, NOT a separate
experiment-grade event. `CONFIRMED_PASS / CONFIRMED_FAIL` only
after a clean reproduction (`AF8`-style: new run ID, independent
namespace, frozen git SHA, fresh process, verified checkpoint
hashes, independently generated eval output, ideally independent
token-cache build). `INVALID` is reachable from any state.
**This document is the single governing authority for TORUS research
process.** `research/ROADMAP.md` governs sequencing and gates. The
`docs/TORUS-feedback/` v2 package is design rationale; within it,
`08-HARNESS-INSTRUCTIONS-V2.md` governs harness behavior and document 10 is
the Track A-F falsification authority. A conflict is resolved by the
authority order in `research/ROADMAP.md`, never by filename or recency.

Companion to `research/ROADMAP.md`. The roadmap says *what* happens in each
phase; this document says *how* work is proposed, executed, judged,
reproduced, and recorded.

## 1. Governance

- Directive: **ISOLATE → FALSIFY → GRADE → REPRODUCE → RECOMBINE.** Steer
  TORUS as a research program, not a feature-completion roadmap.
- One primary hypothesis per experiment. Tracks A (representation), B
  (adaptive computation), and C (recursive context) carry separate grades; a
  result in one is never evidence in another.
- Work allocation: ~80% controlled experiments and analysis, ~20% minimal
  supporting implementation. Feature work is rejected unless it executes a
  registered experiment, fixes a validated defect, preserves
  reproducibility/security, or operationalizes a component that already
  passed. Record every exception and the experiment it serves.
- Ownership is by track. A shared evaluation owner maintains datasets,
  baseline adapters, metrics schemas, and reproducibility checks.

## 2. Claim-driven operation

Experiments attack **registered claims**, not configurations. Claims live in
`research/track-*/**/claims/*.yaml` (template:
`research/registry/CLAIM-TEMPLATE.yaml`) and carry one of these states:

```
UNTESTED → TESTING → PROVISIONAL_PASS / PROVISIONAL_FAIL
         → REPRODUCTION_REQUIRED → CONFIRMED_PASS / CONFIRMED_FAIL
         → INVALID (at any point; rerun mandatory)
```

Currently registered: **A-RP-001** (T2 adds useful capacity beyond equal
training time; evidence AF1/AF3/AF5/AF8), **A-RP-002** (T2 competitive with
equal-storage non-ternary correction; AF2/AF7/AF8), **A-RP-003** (sequential
freeze superior to joint training; AF4/AF8). A claim's required-evidence list
is fixed at registration. Claims carry stable IDs with immutable integer
revisions (`claim_version`); any change to statement or thresholds bumps the
revision **before** results bearing on it are read, and the prior revision is
preserved in `history`. Every claim must carry explicit quantitative
thresholds (numeric pass/fail bars, e.g. ">2 standard errors on ≥1 capability
metric, no material regression on others") before it may enter `TESTING` —
"useful" is never left for post-hoc interpretation.

The harness execution loop:

1. Read the claim registry.
2. Select the highest-priority unresolved claim.
3. Verify prerequisites and unlock rules (§5).
4. Generate an immutable experiment manifest.
5. Run the cheapest meaningful falsification test.
6. Validate provenance (§4).
7. Score proxy, capability, and cost metrics separately.
8. Grade the result.
9. Reproduce important PASS/FAIL results.
10. Update claim state.
11. Unlock or block dependent experiments.
12. Select the next experiment.

The scheduling question is always: **what is the cheapest experiment that
could prove our current belief wrong?**

## 3. Compute tiers

- **Discovery:** one seed where appropriate, smaller evaluation sample,
  shorter training budget. Designed to cheaply kill weak hypotheses.
- **Confirmation:** multiple seeds (≥3), held-out data, full downstream
  evaluation, immutable artifacts and provenance, appropriate baseline
  tournament.

Only promising discovery runs escalate to confirmation. Every experiment
record declares its tier. Important results move
`DISCOVERY PASS → CLEAN REPRODUCTION → CONFIRMED PASS` — new run ID,
independent artifact namespace, frozen git SHA, fresh process, verified
checkpoint hashes, independently generated evaluation output.

## 4. Experiment lifecycle and provenance

Six states: **PROPOSE → VALIDATE → RUN → AUDIT → DECIDE → SCHEDULE** (details
in `08-HARNESS-INSTRUCTIONS-V2.md`). Decisions: `PASS`, `FAIL`, `INVALID`,
`CONTINUE` — an F grade is a useful research result; `INVALID` is not a
failure and must be rerun cleanly. Every DECIDE emits the full decision
block: hypothesis, result summary, grade, decision, confidence and
reproduction status, next permitted experiment, experiments explicitly
blocked.

Every run gets an immutable namespace:

```
runs/<track>/<experiment_id>/<timestamp-or-uuid>/
```

(AF experiments use `experiments/AF<n>/runs/seed-NNN/` with config.json,
provenance.json, train.jsonl, eval.json, checkpoint.sha256.)

Minimum contents: experiment ID and hypothesis; git SHA and branch; complete
config; model/checkpoint identifiers and hashes; dataset and sample IDs;
seeds; hostname/hardware/software; timestamps; commands; raw and summarized
metrics; artifact hashes.

**Scientific artifacts vs. large training artifacts.** Scientific records
(manifest, result summary, metrics tables, claim updates) are committed
under `research/`. Large artifacts (checkpoints, raw logs, datasets) live in
the git-ignored `runs/` tree or external storage. Every experiment directory
MUST contain an `ARTIFACTS.json` index so a future harness can determine
whether the exact model still exists:

```json
{
  "experiment_id": "EXP-AF-001",
  "artifacts": [
    {"kind": "checkpoint", "path": "runs/a/EXP-AF-001/2026-08-22T.../seed-001/ckpt.npz",
     "sha256": "…", "bytes": 0, "location": "local|external:<uri>", "exists": true}
  ]
}
```

`exists` is updated (never deleted) when storage is reclaimed; an artifact
with `"exists": false` means dependent conclusions can no longer be
re-audited against the raw model, which is stated in any later report.

Contamination rules:

- No shared mutable `/tmp` checkpoint/result names across experiments.
- No concurrent agents write to the same artifact namespace.
- Uncertain provenance → `INVALID`, never PASS or FAIL; rerun cleanly.
- **Important PASS/FAIL results require clean reproduction before any
  architectural action.** (Concurrent sessions previously contaminated shared
  artifacts and produced an incorrect interim verdict —
  `09-EVIDENCE-UPDATE-2026-08-22.md`.)

`runs/` is git-ignored; the committed record is the manifest, result summary,
metrics table, and checksums under `research/`, committed together. Pre-
v0.14.0 results are permanently **engineering validation only**.

## 5. Unlock rules (enforced dependencies)

Dependencies are enforced by claim state and checkpoint results, not by phase
order:

| Locked work | Unlocks when |
|---|---|
| Track B oracle gating (B1) | A-RP-001 `CONFIRMED_PASS`; A-RP-002 at least provisionally supported; AF5 shows task-relevant T2 value above its preregistered threshold |
| OLMoE adaptive precision (B3) | B1 shows useful dense-model oracle savings **and** T1/T2 survived the A-F falsification suite |
| Routine T3/T4 scaling | Measured marginal downstream gain per added physical bit/operation exceeds a preregistered threshold |
| Track B oracle gating (B1) | (reorganized under §11 v2.3) A-RP-001 CONFIRMED outcome is *not* the gate; A-RP-002 PROVISIONAL_PASS (or above) on the equal-storage tournament **and** AF5 task-relevant T2 value above its preregistered threshold **and** an AF8-clean CONFIRMED state on at least one of A-RP-002/003. A-RP-001's CONFIRMED_FAIL closes only the equal-training-time branch and does not on its own block Track B if A-RP-002 is supported. |

## 6. Rules with no exceptions

- Thresholds never change after results exist. A correction is a new
  experiment ID; the old record is retained.
- Negative results are preserved with the same rigor as passes.
- **Never accept proxy-metric gains when downstream behavior disagrees.**
  Proxy-only improvement is labeled `PROXY IMPROVEMENT / CAPABILITY NOT
  VALIDATED`. Teacher/logit KL is diagnostic, never acceptance.
- Every representation experiment reports both metric classes: proxy
  (KL, perplexity, hidden-state error, cosine) and capability (ARC-E or
  equivalent, LAMBADA or equivalent, plus a held-out task where feasible).
- Compare against simpler strong baselines (INT8/INT4/low-rank/dense-adapter
  at equal storage; conventional RAG for Track C).
- Never scale training while cheaper diagnostics fail; never train a gate
  before oracle gating proves headroom; never spend large-model budget before
  the small controlled experiment passes.
- Never call skipped theoretical operations an energy win without
  measurement; never describe the physical format as "1.58 bits/weight"
  without packing and metadata accounting.
- Never mix a new feature, a fix, a benchmark change, and an architectural
  conclusion in one experiment.
- Never let OLMoE routing confound basic Track-A ternary experiments.
- Never claim residual planes are useless based on superseded trainer-bug
  runs; never claim Hadamard proven from post-hoc experiments, nor failed
  merely because post-hoc recovery is NO-SHIP.
- If the residual-plane acceptance bar (roadmap G2→3) fails, downgrade the
  claim — do not rescue the architecture with additional complexity.

## 7. Evidence and grading

Grades A–F + INVALID per `07-GRADING-AND-REPRODUCIBILITY-V2.md`. Every result
reports the five evidence categories separately — (1) software correctness,
(2) representation/capacity, (3) downstream model quality, (4) systems
efficiency, (5) end-to-end usefulness. One category is never a proxy for
another.

Confirmation-tier PASS requires ≥3 seeds plus one clean reproduction (§3).
Confidence intervals on all quality metrics; intervals overlapping the
baseline are reported as such. AF3 robustness classifications (ROBUST /
MODERATELY SENSITIVE / FRAGILE) attach to any method whose success depends on
initialization.

## 8. Cadence

- **Per experiment:** PROPOSE commit → run → AUDIT → DECIDE commit, each a
  separate reviewable unit, each ending in the full decision block.
- **Per claim:** state transition recorded in the claim YAML + registry INDEX
  after every contributing experiment.
- **Per gate (roadmap G0→1 … G5→6):** full track grade review written to
  `research/reports/`, including when the decision is "stop."
- **Continuous:** the feature freeze holds until the Final verdict; §6 rules
  are always active.

## 9. Steering priorities

1. **Track A:** layer sensitivity; reproduce the T1→T2 correction signal;
   then the A-F falsification suite against claims A-RP-001/002/003;
   heterogeneous precision; small native-Hadamard controlled training.
2. **Track B:** locked until the §5 unlock rule fires; then oracle gating →
   learned task-aware gating → OLMoE expert-sparsity × precision-sparsity.
3. **Track C:** benchmark recursive context independently on a competent
   conventional model.

## 10. Open items carried into Phase 0

1. `pyproject.toml` version (0.3.0) disagrees with the changelog/commit line
   (v0.16.0); no git tags exist. → Roadmap 0.1, 0.7.
2. `torus.train` missing from the wheel package list. → Roadmap 0.2.
3. AVX-512 claimed for the 3995WX (Zen 2); the AVX-512 benchmark column in
   `docs/KERNELS.md` is unmeasured on that hardware. → Roadmap 0.3.
4. Pre-regime runs (v0.16.0 overnight distillation; the
   provenance-contaminated session) retro-registered as `INVALID`.
   → Roadmap 0.6.

## 11. v2.3 Track A decision-axis revision

**Driven by:** `EXP-AF-001` (AF1, 2026-08-22, git `39be76c`).
A-RP-001 as written failed decisively: at matched training-time
budget, a sequentially trained T2 ternary correction plane (T1+T2)
lost to plain T1-continued on every capability metric. The data are
correct; the question was the wrong primary axis for a deployment
decision.

**From this revision forward, Track A's primary decision axis is:**

> Capability as a function of a *cost vector*, not a single scalar.
> The cost vector is, at minimum:
>
> **C = (deployed bytes,
>     training FLOPs,
>     inference ops / token,
>     memory traffic / token,
>     measured latency / token,
>     joules / token)**
>
> Each term is reported separately. No claim "wins" on C alone;
> a Pareto frontier is drawn across C and capability.

Initially not every term is measurable at full fidelity — energy and
latency require on-Titan instrumentation, etc. The architecture
preserves all terms independently so subsequent revisions can swap
in better measurements without invalidating the trial design.

**Concretely, the v2.3 Track-A claim semantics become:**

- **A-RP-002** (equal-storage tournament) is now the **central**
  Track-A claim, not a peripheral one. The ternary T2 plane is
  tested against INT4 residual, smaller INT8 residual, low-rank
  correction, learned group scales, and a small dense adapter at
  matched *physical bytes-in-deployment*, including scales,
  metadata, headers, and alignment. The AF2 manifest must report
  training FLOPs in addition to deployed bytes; "1.58 bits/weight"
  is the *floor* on bytes, not the actual on-disk footprint.
- **A-RP-001** (equal-training-time) is **PROVISIONAL_FAIL /
  REPRODUCTION_REQUIRED**; even a CONFIRMED_FAIL here only closes
  the equal-training-time branch, not the residual-plane program.
  The lesson is architectural: the v2 wording chose the wrong
  axis. Future ternary work must be evaluated at matched storage.
- **A-RP-003** (sequential vs joint training) remains genuinely
  open. The curriculum T1 → freeze → T2 may be suboptimal
  relative to a joint or teacher-initialized curriculum; AF4
  attacks this claim directly. Until AF4 returns, the
  representation-versus-curriculum distinction must stay explicit
  throughout the repo so a curriculum failure does not invalidate
  the residual architecture.

**Cost-vector reporting convention for all Track-A experiments
(confirmed by this revision):**

- The manifest declares which cost terms are measured, which are
  held matched by construction, and which are explicitly out of
  scope. An experiment that matches only bytes cannot use the
  wording "matched compute" or "matched training budget"; it must
  say "matched bytes" and list the other terms as out-of-scope
  or held equal by inference.
- Each experiment's results table includes a row per reported
  term, even if all values are "matched." A Pareto plot across
  the matrix becomes part of the final Track-A verdict
  (§ 12 to follow).

**Governance rule added by this revision:**

> "Equal training steps" is no longer a defensible single-axis
> match condition for Track A. Track-A claim tests must either
> (a) match deployed bytes (the AF2 default) and report other
> cost terms as out-of-scope, or (b) match a multi-term cost
> vector and report all terms. Either is acceptable; a single-
> scalar cost match is not.
