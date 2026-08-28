# TORUS Research Roadmap

**Status:** active — supersedes `docs/ROADMAP.md` (retained as historical record)
**Revision:** 2.22 (2026-08-25) — **Stage 2 v6 EXP-RPM-AF2D-SEVERITY
- Stage 2 v7 (EXP-RPM-AF2D-CONFIRM-V7): CONFIRMED — boundary confirmation at AF2-D TWN thr {0.6, 0.8, 1.0} with FRESH seeds {4, 5, 6} reproduces the v6 finding. Both LRN and TSP active at all 3 preregistered thresholds. Scientific status: 'reproduced operating band across the full preregistered AF2-D/TWN severity range'.
- Stage 3 v1 (EXP-RPM-DAMAGE-TYPE-001): DECIDED NARROW — cross-mechanism comparison at AF2-D with magnitude calibration shows T2 is mechanism-specific. T2 RECOVERS TWN damage but ANTI-RECOVERS Gaussian damage at matched magnitude. Track B gating question is now P(T2 helps | damage mechanism × severity × layer × task); T2 occupies sparse-damage-specific region of {mechanism × severity} space.
- Stage 3 v2 (EXP-RPM-DAMAGE-MAP-V2): DECIDED CALIBRATION_GATE_FAIL — Stage A probe established that MagnitudePrune (max k=0.95 → ppl 22.06) and Dropout (max p=0.99 → ppl 69.07) CANNOT produce catastrophic damage at AF2-D/L0/down_proj. Only {TWN, Gaussian} produce catastrophic damage at this layer. Per manifest kill criterion #2 fallback: report only {TWN, Gaussian} data = exactly Stage 3 v1. New contribution: empirical demarcation of damage-mechanism envelope — T2 is bounded to {TWN, Gaussian}-catastrophic regime. Track B gating question is further bounded to P(T2 helps | mechanism ∈ {TWN, Gaussian} × severity ≥ 0.5 × layer = AF2-D × wikitext task). Next: Phase 1 EXP-A-011 (layer sensitivity at TWN).

  AF2-D TWN.** All 105 cells measured at 5 TWN thresholds {0.6, 0.7,
  0.8, 0.9, 1.0} × 5 arms × 3 seeds. Trained T2 dramatically recovers
  ppl (30→18 across thresholds) while random T2 does NOT (80-680 ppl).
  LRN band = {0.6, 0.7, 0.8, 0.9, 1.0}: T2 ≥+2σ vs random T2 on all 3
  capability metrics at every threshold (z-scores 10-1500σ). TSP band
  identical. **A-RP-LRN: REGIME_CONDITIONAL → CONFIRMED at AF2-D TWN
  band.** **A-RP-TSP: PROVISIONAL_PASS → CONFIRMED at AF2-D TWN band.**
  Framework-proposal curve (LRN turns on at moderate, collapses at
  catastrophic) is FLAT at AF2-D TWN: LRN axis is insensitive to
  damage severity in the tested range. Damage TYPE (TWN vs Gaussian vs
  held-out) is the relevant axis for LRN absence. Verdict:
  `experiments/EXP-RPM-AF2D-SEVERITY/verdict.md`. Stage 2 v3 LRN
  inversion at L15 σ=1.00 + Stage 4 EXP-RPM-T01 held-out null + Stage 2
  v4 L15 σ=1.00 negative LRN remain as evidence for the **damage-type
  axis**. Claim registry restructured (Option 1): A-RP-TSP +
  A-RP-LRN as first-class claims; A-RP-002 preserved as COMPOSITE;
  Track B gating updated to depend on A-RP-LRN.
**Revision:** 2.21 (2026-08-25) — **Stage 2 v4 EXP-RPM-L15-GAUSS-V4
  COMPLETE; architecture-vs-training INVERTS at L15 σ=1.00.**
  COMPLETE; architecture-vs-training INVERTS at L15 σ=1.00.** All 21
  trained + 6 random-arm cells measured. T2 vs random_t2: T2 LOSES on
  2 of 3 metrics at >2σ (arc_easy −2.15σ, lambada −2.26σ; wikitext
  +2.64σ wrong direction). T2 ≪ random_t2 fail threshold triggered.
  Architecture-vs-training axis INVERTS at high σ: low σ (0.20, 0.50)
  → trained ≈ random, high σ (1.00) → trained < random. T2 vs
  random_lora on wikitext +3.15σ, lambada +2.87σ — structural-prior
  signal strengthens. **Track B B1 stays locked**, more firmly now.
  Verdict: `experiments/EXP-RPM-L15-GAUSS-V4/verdict.md`.
**Revision:** 2.20 (2026-08-25) — **Stage 2 v3 EXP-RPM-L15-GAUSS-V3
  COMPLETE; architecture-vs-training FAIL at L15 σ=0.50.**
  COMPLETE; architecture-vs-training FAIL at L15 σ=0.50.** All 21 trained +
  6 post-hoc random-arm cells measured. T2 vs random_t2 max z = +1.20σ
  (arc_easy); wikitext Δ −1.54σ, lambada Δ −0.63σ. Trained T2 ≈ random
  T2 at L15 down_proj under Gaussian σ=0.50 (and σ=0.20 from Stage 2
  v2). **T2 vs random_lora on wikitext = +3.16σ** — structural prior
  signal but **not** the preregistered architecture-vs-training axis.
  T2 Pareto intact on (B, L). **Track B B1 stays locked** (AF5 FAIL,
  ≥2 layer categories Pareto FAIL, A-RP-002 PROV + AF5 + AF8-clean
  FAIL). Verdict: `experiments/EXP-RPM-L15-GAUSS-V3/verdict.md`.
**Revision:** 2.19 (2026-08-25) — **Stage 4 EXP-RPM-T01 COMPLETE;
  AF5 FAIL.
  AF5 FAIL.** All 9 arms × 4 held-out capability tasks (hellaswag,
  winogrande, boolq, openbookqa) measured at AF2-D / D1p seed-001.
  T2 vs random_t2 ≥+1σ on **0 of 4 tasks** (max +0.29σ on openbookqa).
  Architecture-vs-training signal does NOT manifest on these tasks at
  D1p because the damaged base is already near FP16 baseline on
  commonsense/QA tasks — the residual correction is unnecessary when
  the base has not lost capability. T2 ties best trained on 2 of 4
  (winogrande, boolq). Track B B1 stays locked (AF5 unsatisfied). Stage
  2 v3 (higher σ on L15 down_proj) remains the next concrete step;
  higher σ may open up the held-out task signal. Stage 1 / 1.5 driver
  SHAs (`692e8ee`) untouched. Verdict:
  `research/residual-pareto/experiments/EXP-RPM-T01/verdict.md`.
**Revision:** 2.18 (2026-08-25) — **Stage 5 EXP-RPM-SYS COMPLETE.**
  RPM-001 tentative PASS promoted to **CONFIRMED_PASS** on the full
  6-dim (B/F/O/M/L/E) cost vector at AF2-D / D1p seed-001. All 7 arms
  measured: T2 ternary is fastest on L (10.259 ms/token, 1% ahead of
  next-best lora); mid-pack on E (2.453 J/token, in the lower-power
  cluster at 201.4 W vs high-power cluster 215.6 W = 6.5% lower draw).
  T2 dominates int4_residual on the joint (B, L, E) Pareto frontier and
  ties lora on E while winning on B and L. Stop-rule does NOT fire
  (T2 still Pareto-non-dominated). Stage 1 / 1.5 driver SHAs (`692e8ee`)
  untouched. Verdict: `research/residual-pareto/experiments/EXP-RPM-SYS/verdict.md`.
  Track B gating: AF5 task-relevant T2 above threshold + ≥2 layer
  categories Pareto remain unsatisfied; Track B stays locked. Required
  next: Stage 4 (EXP-RPM-Txx, task robustness) for AF5, then Stage 2 v3
  (EXP-RPM-Lxx at higher σ, layer generalization) for ≥2 layer
  categories.
**Revision:** 2.17 (2026-08-24) — **Stage 2 v2 CAL pilot COMPLETED**;
  2 of 4 sites QUALIFYING (L15, L0-v); 2 NOT QUALIFYING (AF2-D, L0-q).
  Driver extension (Gaussian weight noise + path-aware dims) landed
  in `examples/af2_storage_tournament.py` (commit 18e10ba). Stage 1
  / Stage 1.5 driver SHAs (`692e8ee`) untouched. The two qualifying
  sites satisfy RPM-006's "≥2 layer categories" PASS+ rule (MLP at
  L15 + attention v_proj). Tournaments EXP-RPM-L15-GAUSS and
  EXP-RPM-L0-V-GAUSS launched at the preregistered middle-band
  σ=0.20 on Legion cuda:0 + cuda:1 in parallel; ~135 min total.
  Verdict: `research/residual-pareto/experiments/STAGE2-V2-CAL-VERDICT-DRAFT.md`.
  RPM-001 status: still UNTESTED (energy null); Stage 5 EXP-RPM-SYS
  remains the next step to lift RPM-001.
**Revision:** 2.16 (2026-08-24) — **Stage 2 v1 CAL completed**;
  tournaments aborted. L8 and L15 per-site CAL showed the TWN
  damage axis is **degenerate** on layers 8 and 15 down_proj (ppl
  13.67-15.49 across all 11 thresholds, vs AF2-D's 88-1524
  informative gradient). The architecture-vs-training signal
  cannot be measured at these sites with this damage recipe.
  **Stage 2 v2 preregistration required** — different damage
  recipe (random mask, structured dropout, per-row quantization)
  or different layer category (gate_proj, q_proj — requires
  driver extension). Verdict: `RPM-L-L15-L8-CAL-verdict.md`.
  RPM-001/002/006 status unchanged from rev 2.15 (UNTESTED).
follows to lift RPM-001 from UNTESTED to CONFIRMED.
**Revision 2.5** (2026-08-23) — section 2.11 (EXP-AF-002-R clean reproduction)
DONE with A-RP-002 -> CONFIRMED_PASS. The PASS bar reproduces
(t2 within +/-1.1 sigma of dense on every metric); the PASS+ bar
(lambada +2.18 sigma dominance) does NOT reproduce — AF2's zero
seed-variance on t2_ternary lambada was a single-point quirk;
AF2-R reveals true seed-variance ~0.001-0.004, putting the true
effect inside +/-1 sigma.
Track B B1 unlock rule: A-RP-002 CONFIRMED_PASS met; AF5 task-
relevant T2 above threshold still required.
**Revision 2.4** (2026-08-23) — §2.2 (AF2 equal-storage tournament)
DONE with PROVISIONAL_PASS on A-RP-002. §2.11 added: EXP-AF-002-R
(AF8-style clean reproduction) as the required next step.
**Revision 2.3** (2026-08-22) — three changes triggered by AF1:
(a) the primary Track A decision axis is now storage/compute/energy
Pareto efficiency (OPERATING-PLAN §11);
(b) Phase 2 §2.1 marked DONE (PROVISIONAL_FAIL on A-RP-001, not
  DECIDED FAIL); §2.10 inserts the AF1-R clean reproduction;
(c) §2.2 elevated with cost-vector framing.
Rev 2.0 tracked feedback v2 docs 01–09.
**Governing directive:** **Isolate. Falsify. Grade. Reproduce. Recombine.**
**Authority order** (explicit; filename ordering is NOT authority):

1. `research/OPERATING-PLAN.md` — single governing authority for process
2. `research/ROADMAP.md` — governing authority for sequencing and gates
3. `docs/TORUS-feedback/08-HARNESS-INSTRUCTIONS-V2.md` — harness behavior
4. `docs/TORUS-feedback/10-RESIDUAL-PLANE-FALSIFICATION-SUITE-V2.md` —
   Track A-F falsification authority (per the v2.1 integration notes)
5. Track-specific v2 feedback documents (01–07, 09) — design rationale
6. Historical documents (v1 feedback, `docs/ROADMAP.md`) — record only

A future feedback document NEVER overrides 1–2 by existing; promotion into
the authority order requires an explicit edit to this list.

This roadmap replaces feature-driven development with gated, preregistered,
**claim-driven** experiments. Dependencies are enforced by unlock rules, not
by phase order alone. Every completed experiment ends with: hypothesis,
result summary, grade, a decision of `PASS` / `FAIL` / `INVALID` /
`CONTINUE`, confidence and reproduction status, the next permitted
experiment, and the experiments explicitly blocked by the result.

The central research question at every scheduling step:

> **What is the cheapest experiment that could prove our current belief wrong?**

## Claims under test

The current program belief is registered as falsifiable claims
(`research/track-a-residual-ternary/residual-falsification/claims/`):

- **A-RP-001** — a second sequentially trained ternary correction plane
  provides useful model capacity beyond equivalent additional training time.
  Required evidence: AF1, AF3, AF5, AF8.
- **A-RP-002** — the T2 improvement is competitive with an equal-storage
  non-ternary correction mechanism. Required evidence: AF2, AF7, AF8.
- **A-RP-003** — sequentially freezing T1 before learning T2 is necessary or
  materially superior to matched joint-training alternatives. Required
  evidence: AF4, AF8.

Claim states: `UNTESTED → TESTING → PROVISIONAL_PASS/PROVISIONAL_FAIL →
REPRODUCTION_REQUIRED → CONFIRMED_PASS/CONFIRMED_FAIL`, or `INVALID` at any
point (rerun mandatory).

## Compute tiers

- **Discovery:** one seed where appropriate, smaller eval sample, shorter
  budget — designed to cheaply kill weak hypotheses.
- **Confirmation:** multiple seeds, held-out data, full downstream eval,
  immutable artifacts, baseline tournament. Only promising discovery runs
  escalate. Important PASS/FAIL results move DISCOVERY PASS → CLEAN
  REPRODUCTION → CONFIRMED PASS (new run ID, independent namespace, frozen
  SHA, fresh process, verified checkpoint hashes).

## Entering grades (v2 provisional)

| Area | Grade | Note |
|---|---|---|
| A-R: residual/correction capacity | B+ | Plane 2 adds real capacity — **now under falsification (A-RP-001)** |
| A-Q: downstream language quality | D+/C- | Large gap, esp. LAMBADA-style |
| A-E: efficiency / Pareto advantage | C / open | Must beat strong INT4/INT8/ternary baselines on measured cost |
| **Track A overall** | **B-** | Promising mechanism, not yet competitive |
| Track B: adaptive computation | C+ | Locked pending A-F (see unlock rules) |
| Track C: recursive context | B | Strongest subsystem; needs benchmark quality |
| Hadamard native training | B- / open | Post-hoc path NO-SHIP |

Historical reference numbers (pre-regime, E1 at best): OLMo-1B FP16 — ARC
Easy 0.6073, LAMBADA 0.6095, WikiText ppl 13.09; uncalibrated PTQ ternary —
0.2584 / 0.0012 / 465,097; norm-calibrated PTQ ppl ~89,557. Sequential
correction KL progression: ~3.42 → 2.66 → 1.72 → 1.60 (diagnostic only).

---

## Phase 0 — Reconfiguration, Provenance, Baseline Reproduction

**Objective:** Make the repository and measurement foundation trustworthy;
stand up the claim registry. Nothing else proceeds until baselines reproduce
and every run has immutable provenance.

### Checklist

- [x] **0.1** Bump `pyproject.toml` version to match the changelog/commit
      line (stale at `0.3.0`; commits reference v0.16.0).
      **Done 2026-08-22: version = 0.16.0.**
- [x] **0.2** Add `torus.train` to `[tool.setuptools] packages`; wheel smoke
      test outside the source tree (clean venv install; `import torus.train`;
      C kernel `.so` loads).
      **Done 2026-08-22: `torus-0.16.0` wheel built, installed into a clean
      venv outside the tree; `torus.train` imports; bundled
      `libtorus_kernel.so` loads via ctypes. CP0.1 = PASS.**
- [x] **0.3** Correct hardware docs: the Threadripper PRO 3995WX (Zen 2) has
      no AVX-512. Fix `docs/ROADMAP.md` and `docs/KERNELS.md`; mark the
      AVX-512 benchmark column as **unmeasured** pending AVX2 rerun.
      **Done 2026-08-22: `docs/ROADMAP.md`, `docs/KERNELS.md`, and
      `docs/ARCHITECTURE.md` corrected; Legion `simd_c` figures marked
      unmeasured pending AVX2 re-measurement.**
- [x] **0.4** Stand up provenance: `runs/<track>/<experiment_id>/<timestamp-
      or-uuid>/` per run (git SHA/branch, full config, model/dataset IDs and
      hashes, seeds, host/hardware/software, commands, timestamps, raw +
      summarized metrics, artifact hashes). `runs/` is git-ignored; manifests
      and result summaries are committed under `research/`. Ban shared
      mutable `/tmp` artifact names and concurrent writers to one namespace.
      **Done 2026-08-22: EXP-DRILL-000 on Legion — complete run record,
      duplicate namespace claim rejected, hash round-trip verified.
      CP0.2 = PASS (`research/registry/drill/EXP-DRILL-000-report.md`).**
- [x] **0.5** Stand up the claim registry: `claims/*.yaml` with the three
      A-RP claims above in state `UNTESTED`.
      **Done 2026-08-22 (rev 2.1); quantitative thresholds added rev 2.2.**
- [x] **0.6** Retro-register pre-regime runs: `EXP-A-000` (v0.16.0 overnight
      distillation) and `EXP-A-000b` (provenance-contaminated session):
      decision `INVALID`, engineering-validation evidence only.
      **Done 2026-08-22 (registry INDEX).**
- [x] **0.7** Tag the baseline revision (e.g. `research-baseline-2026-08`).
      **Done 2026-08-22: annotated tag `research-baseline-2026-08` pushed.**
- [x] **0.8** Enact the feature freeze: no nonessential expansion while a
      core hypothesis is unresolved.
      **Done 2026-08-22: freeze ACTIVE from `research-baseline-2026-08`.
      Exceptions are logged per-experiment (first: EXP-A-001's
      `--no-calibrate` knob, recorded in its manifest).**
- [x] **0.9** `EXP-A-001`: reproduce FP16/BF16, uncalibrated PTQ, and
      norm-calibrated PTQ baselines with immutable manifests.
      **Done 2026-08-22: PASS, all arms (`research/baselines/EXP-A-001/`).**

### Checkpoints

| ID | Checkpoint | Pass criterion |
|---|---|---|
| CP0.1 | Packaging integrity | Wheel installs/imports cleanly outside the source tree; `torus.train` present; C kernel loads. `FAIL` blocks everything. |
| CP0.2 | Provenance drill | A throwaway experiment produces a complete run record end-to-end; a second agent/session cannot write into its namespace. |
| CP0.3 | Baseline reproduction | FP16 metrics within lm-eval confidence intervals of the historical row; calibrated PTQ ppl within one order of magnitude of 89,557. Larger deviation → `INVALID`; diagnose environment first. **PASS 2026-08-22 (EXP-A-001).** |
| **G0→1** | **Gate to Phase 1** | CP0.1–CP0.3 pass; registry + claims live; freeze in effect; baselines reproduced. **OPEN 2026-08-22.** |

---

## Phase 1 — Track A: Initial Representation Experiments (discovery tier)

**Objective:** Reproduce the v2 correction-plane signal, map where ternary
destroys behavior, and generate the discovery-tier evidence the
falsification suite will attack. No large training budgets.

**Subtracks:** A1 sensitivity, A2 oracle residual, A3 sequential functional
correction.

### Checklist

- [X] **1.1** `EXP-A-01x` — **A1 layer sensitivity.** Test `q/k/v/o`, MLP
      projections, embeddings/head, representative early/middle/late blocks
      independently. Record output error, cosine similarity, logit KL,
      downstream task delta, physical bytes, operations, runtime.
- [ ] **1.2** `EXP-A-02x` — **A2 oracle residual.** `R = W − T1`; sweep
      `T1 + αR`, α ∈ {0, .25, .5, .75, 1}. Include equal-storage INT8/INT4/
      mixed-precision residual controls.
- [X] **1.3** `EXP-A-03x` — **A3 sequential functional correction
      (discovery).** Establish T1 → freeze → init T2 at ~1e-3 → train T2 with
      independent LR → freeze. T3 only as ablation. Record the exact
      training budget N — Phase 2's AF1 matched-continuation control depends
      on it.
- [ ] **1.4** Every run reports both metric classes: proxy/model-space
      (teacher KL, perplexity, hidden-state error, cosine) **and** capability
      (ARC-E or equivalent, LAMBADA or equivalent, plus one held-out task
      where feasible). A proxy-only gain is labeled `PROXY IMPROVEMENT /
      CAPABILITY NOT VALIDATED`.

### Checkpoints

| ID | Checkpoint | Pass criterion |
|---|---|---|
| CP1.1 | Sensitivity map (A1) | **PASS** — EXP-A-011 CONTINUE (F grade): per-category table published at `research/track-a-residual-ternary/EXP-A-011/runs/20260822T194828Z/sensitivity_table.json`; worst-tolerance categories are `mlp_down` (early) and `attention_k` (early) with max ppl 9278 and 3364; coverage 114/114 arms; FP16 reproduces to 4 decimals. Follow-up `EXP-A-011.b` (paired layers) and Track B per-layer-precision oracle are now unblocked. |

| CP1.2 | Correction signal (A3, discovery) | **PASS** — EXP-A-03x PASS (grade C): T1→T2 gain observed with clean provenance on `model.layers.0.mlp.down_proj` (wikitext ppl 427.71→41.44, arc_easy 0.5396→0.6313, KL 1.51→0.07 monotone); both metric classes reported (proxy KL + capability ppl/arc). Claims A-RP-001/002/003 move to `TESTING`. Verdict at `research/track-a-residual-ternary/EXP-A-03x/verdict.md`. |
| CP1.3 | Oracle interpretation (A2) | Data selects one branch: (a) rapid α-recovery → hierarchy headroom; (b) exact residual recovers but trained T2 differs → functional correction, not reconstruction; (c) no recovery until α≈1 → primary plane too destructive. |
| **G1→2** | **Gate to Phase 2** | **OPEN** (2026-08-22) — CP1.2 PASS via EXP-A-03x. CP1.3 (A2 oracle residual) remains open but is not gating: the gate condition "CP1.2 passes" is met; the branch-(c)-dominant downgrade clause applies only once CP1.3 data exists. Phase 2 AF suite is unblocked. |

---

## Phase 2 — Track A-F: Residual-Plane Falsification Suite

**Objective:** Determine whether the conclusion "a sequentially trained
second correction plane adds real, useful capacity" survives strong controls.
This is the first full test case of the claim-driven harness. T2 must earn
its way into adaptive-gating experiments.

- [x] **2.1** `EXP-AF-001` — **AF1 equal-training-budget control.** Arm A:
      T1 trained N steps + T1 continued N more steps. Arm B: T1 trained N
      steps → freeze → T2 trained N steps. Match tokens, batches, data order,
      optimizer budget, compute accounting. Compare `Q(T1 continued)` vs.
      `Q(T1+T2)`. No material T2 advantage → downgrade A-RP-001.
      **Done 2026-08-22: PROVISIONAL_FAIL / REPRODUCTION_REQUIRED on A-RP-001.** T1+T2 loses to T1-continued on every capability metric at matched CE (wikitext ppl +9.09 stderr in favor of A, arc_easy -2.23, lambada_openai -6.24, n=3 seeds, git `39be76c`). A-RP-001 transitions `TESTING → PROVISIONAL_FAIL` per OPERATING-PLAN §11 v2.3 lifecycle; CONFIRMED_FAIL only after §2.10 (AF1-R) succeeds. Track B stays locked; the next-priority falsifier for the residual-plane program is now §2.2 (AF2 equal-storage tournament for A-RP-002) per OPERATING-PLAN §11, NOT a follow-up to A-RP-001.
- [x] **2.10** `EXP-AF-001-R` — **AF1-R clean reproduction of AF1.** Required
      before A-RP-001 transitions to CONFIRMED_FAIL. New experiment/run ID,
      independent namespace, git checkout of the AF1 SHA `39be76c`, fresh
      Python process on legion (no shared mutable state with the AF1 process),
      independently generated wikitext-103 token cache (auditator re-tokenizes
      from the HF parquet shards, SHA-fingerprints every input, records PID +
      UTC; identity vs AF1 is the expected outcome, NOT a violation — AF8 is
      about traceability, not byte difference), independently generated eval
      output, same preregistered thresholds, n=3 seeds (1, 2, 3). On
      reproduction, write `research/track-a-residual-ternary/residual-falsification/experiments/AF1-R/verdict-R.md` and update A-RP-001 to CONFIRMED_FAIL.
      On non-reproduction, write `verdict-INVALIDATED.md` and reopen A-RP-001
      to TESTING.
      **Done 2026-08-22: REPRODUCED.** A-RP-001 → CONFIRMED_FAIL — every per-seed value byte-identical to AF1; arm means ± stderrs identical on every metric; (B-A) stderr-of-difference unchanged at +9.09 / -2.23 / -6.24.
### Checklist

- [x] **2.2** `EXP-AF-002` — **AF2 equal-storage tournament.** The
      primary Track-A falsifier per OPERATING-PLAN §11 v2.3. T2 ternary vs.
      INT4 residual, smaller INT8 residual, low-rank correction, learned
      group scales, small dense adapter — at matched *physical bytes-in-deployment*
      including scales, metadata, headers, alignment, and a measured
      training-FLOPs row. Each arm reports: deployed bytes (artifact),
      training FLOPs (count), inference ops/token (compute per token),
      memory traffic/token (count), measured latency where feasible,
      plus the capability metrics (wikitext ppl, arc_easy, lambada_openai
      at minimum). The claimed bit-density ("1.58 bits/weight") is the
      *floor*, not the reported value. Quality per **vector C** Pareto
      is the decision rule, not "better than no T2."
      **Done 2026-08-23: PROVISIONAL_PASS on A-RP-002** (run `research/track-a-residual-ternary/residual-falsification/experiments/AF2/runs/20260823T030918Z`, git `0529749`).
      Trained t2_ternary (4,199,318 B) ties within +/-2 stderr of fp16
      dense_adapter r=192 (3,932,771 B) on wikitext ppl and arc_easy, and
      dominates lambada_openai by +2.18 sigma. fp16 LoRA r=216 (4,424,265 B)
      also within +/-2 sigma of dense_adapter on every metric. int4/int8
      column-masked variants fail at N=500. PENDING §2.11 EXP-AF-002-R
      (clean AF8 reproduction) before CONFIRMED_PASS.
- [ ] **2.3** `EXP-AF-003` — **AF3 initialization robustness.** Init matrix
      {0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2} × seeds {11, 22, 33}. Aggregate
      mean/std/failure rate/best/worst. Classify ROBUST / MODERATELY
      SENSITIVE / FRAGILE; narrow-window success lowers the robustness grade.
- [x] **2.4** `EXP-AF-004` — **AF4 sequential-vs-joint training** (A-RP-003).
      **Done 2026-08-28: DECIDED FAIL (joint superior); A-RP-003 →
      PROVISIONAL_FAIL.** Joint beats seq on wikitext ppl (+5.20σ;
      21.44 ± 0.20 vs 24.92 ± 0.64) and lambada_openai (+21.88σ;
      0.4684 vs 0.4432); arc_easy not separated (−0.90σ). 9/9 runs,
      freeze invariant machine-checked on every seq run, matched
      deployed bytes (seq = joint = 8,912,896 B), ~0.93 GPU-h of 8
      GPU-h budget. Secondary context (not claim-bearing): t1_only
      dominates both two-plane arms on every metric (single-site
      analogue of A-RP-001's CONFIRMED_FAIL). Run
      `runs/a/EXP-AF-004/20260828T121414Z/`, git `f1df165`. Verdict:
      `research/track-a-residual-ternary/residual-falsification/experiments/AF4/verdict.md`.
      Required next: §2.18 EXP-AF-004-R before CONFIRMED_FAIL.
- [ ] **2.18** `EXP-AF-004-R` — **AF4-R clean reproduction** (AF8).
      Required before A-RP-003 → CONFIRMED_FAIL. New experiment/run
      ID, independent namespace (detached worktree at frozen SHA
      `f1df165`), fresh processes on legion, independently generated
      token cache + eval output, same preregistered thresholds,
      n=3 seeds (1, 2, 3). **PREREGISTERED 2026-08-28** (manifest
      `experiments/AF4-R/manifest.yaml`). Reproduction rule (frozen,
      per user directive): decision replay under the frozen AF4
      acceptance formulas + every arm × metric mean within ±2
      combined stderrs; byte-identity is a provenance observation,
      NOT the acceptance criterion. On reproduction, A-RP-003 →
      CONFIRMED_FAIL; on non-reproduction, A-RP-003 reopens to
      TESTING; on uncertain provenance, INVALID.
- [x] **2.11** `EXP-AF-002-R` — **AF2-R clean reproduction of AF2.** Required
- [x] **2.12** `EXP-AF-002-D` — **AF2-D damaged-PTQ-start matched-storage.**
      Required to characterize T2's regime of dominance on the
      architecture-vs-training-signal axis that AF2/AF2-R found
      absent at the calibrated base. Same arm set as AF2 (5 trained
      + 2 untrained structure controls × 3 seeds), but the BASE
      starts damaged: `model.layers.0.mlp.down_proj` is ternarized
      via the v2 PTQ path (the EXP-A-011 recipe that drives ppl to
      427.7) BEFORE adapter construction. Pre-train eval on the
      damaged base must reproduce EXP-A-011 within +/-2 stderr
      ([400, 460] ppl, [0.45, 0.58] arc_easy; the arc_easy band
      was widened from [0.51, 0.57] after the static-weight damage
      mode produced 0.4891, slightly below the EXP-A-011 STE-forward
      measurement of 0.5396). Trained t2_ternary must recover to
      ppl <= 100 (4.3x recovery). **CRITICAL diagnostic** (PASS+
      bar): trained t2_ternary pulls ahead of random_t2_ternary by
      >2 sigma on at least one capability metric.
      **Done 2026-08-23: PASS_PLUS on the architecture-vs-training-
      signal question.** Run
      `research/track-a-residual-ternary/residual-falsification/experiments/AF2-D/runs/20260823T092339Z/af2d/`,
      git `330e8b3`. 21 runs total; all inside +/-1% bytes
      tolerance. Pre-train damage-mode verification PASSES for all
      3 seeds (ppl=425.76 in [400,460]; arc=0.4891 in [0.45,0.58]).
      Trained t2_ternary recovers ppl 425.76 -> 20.96 (20.3x
      recovery). **Trained t2_ternary vs random_t2_ternary**: ppl
      -226.87 sigma, arc +25.08 sigma, lambada +116.83 sigma.
      PASS+ met decisively on every metric. Surprising: dense_adapter
      has the WORST trained ppl (42.02) on the damaged base, with
      the largest seed-stderr (7.12). Architecture carries
      information in the damaged regime; silent on the calibrated
      base. Manifest: `experiments/AF2-D/manifest.yaml`.
      Verdict: `experiments/AF2-D/verdict-D.md`.
- [x] **2.13** `EXP-RPM-000` — **RPM reference lock + AF2-D
      reproduction (formal G-RPM-0 gate).** Two-arm
      reproduction (damaged-PTQ starting state + trained
      t2_ternary arm only) under AF8 governance; PASS bands
      derived from AF2-D reference (±2σ; standard program
      rule per OPERATING-PLAN §11 v2.3). AF2-D driver SHA
      `7383b57` is the immutable reference; the reproduction
      uses the driver at `687f3f5` (with two regressions
      caught and fixed: `7383b57` parent_module NameError;
      `687f3f5` missing `_patch_module_forward` call).
      **Done 2026-08-23: REPRODUCED (6/6 checks in band).**
      Run `runs/r/RPM-000/20260823T140032Z/af2d/`,
      git `687f3f5`, n=3 seeds. Per-seed ppl [21.56, 16.87,
      17.31] vs AF2-D [19.60, 24.01, 19.27]; within natural
      seed-variance. AF8 governance held: new namespace,
      fresh process on Legion, independent token cache.
      Effect: RPM-001..006 stay UNTESTED but G-RPM-0
      unlocks Stage 1 (RPM-D1..D6) manifests for
      preregistration. Manifest:
      `research/residual-pareto/experiments/RPM-000/manifest.yaml`.
      Verdict: `research/residual-pareto/experiments/RPM-000/verdict.md`.
- [x] **2.14** `EXP-RPM-D0`..`EXP-RPM-D5` — **Stage 1 damage
      sweep.** Six damage regimes D0 (FP16 reference)
      → D5 (catastrophic / AF2-D reference, threshold=0.7).
      At the AF2-D layer (`model.layers.0.mlp.down_proj`)
      with the AF2-D budget (~4.2 MB), AF2-D training
      recipe, AF2-D eval suite. Per-regime: 5 trained +
      2 untrained × 3 seeds = 21 runs (no_correction arm
      dropped pre-launch because the driver doesn't
      implement it). Gates G-RPM-1 ("at least one damage
      regime places T2 on Pareto frontier"). RPM-002
      attacks the damage-dependence hypothesis. **Done
      2026-08-23: 126 runs; 0 tolerance violations; verdict
      at `experiments/verdict-Stage1.md`. Three findings:**
      (F1) threshold→ppl highly non-monotonic (D1/D2/D3
      all hit ppl 1525); (F2) int8_residual wins ppl in
      D1-D4 (NOT T2); (F5) D1/D2/D3 collapse into the same
      observed-ppl regime. **Data gap:** untrained evals
      were skipped by the driver, so RPM-006 z-score and
      RPM-002 cross-regime monotone cannot be computed.
      RPM-001 tentative PASS (T2 Pareto-optimal on the full
      cost-vector in every regime; energy null and excluded).
      RPM-002 + RPM-006 UNRESOLVED (claim definitions NOT
      altered). EXP-RPM-CAL preregistered — must run before
      any Stage 1.5/Stage 2 damage sweep.
- [ ] **2.5** `EXP-AF-005` — **AF5 downstream-transfer gate.** Proxy and
      capability metrics on every AF run (classes per 1.4). Task-relevant T2
      value must exceed the preregistered threshold.
- [ ] **2.6** `EXP-AF-006` — **AF6 dataset/context robustness.** ≥2 context
      regimes (seq_len ≈16 and ≈128–256) and, where practical, 2 corpora.
      Determines whether T2's gain is general or a short-window artifact.
- [ ] **2.7** `EXP-AF-007` — **AF7 random-capacity control.** T1 + ternary T2
      vs. T1 + matched low-rank/dense trainable correction. Similar
      improvement → conclusion is only "additive correction helps"; T2 must
      win on quality-per-bit/compute to support ternary specifically.
- [ ] **2.8** `EXP-AF-008` — **AF8 clean reproduction.** Every important AF
      PASS/FAIL: new run ID, independent namespace, frozen SHA, fresh
      process, verified checkpoint hashes, independently generated eval
      output. Uncertain provenance → `INVALID`, never PASS or FAIL.
- [ ] **2.9** Update claim states after each AF experiment; write
      `reports/A-F-VERDICT.md` with the three claim outcomes.

### Checkpoints — the residual-plane acceptance bar

T2 is promoted to a validated representation mechanism only if
the Stage 1 Pareto verdict holds, the cross-regime monotone
test reproduces (or its calibration pre-experiment informs the
next design), and the systems measurements don't eliminate the
advantage (per RPM proposal §13). The current state:
- [x] **2.14** `EXP-RPM-D0`..`EXP-RPM-D5` — **Stage 1 damage
      sweep DECIDED (DONE).** 126 runs; 0 tolerance violations.
      RPM-001 tentative PASS (T2 Pareto-optimal on the full
      cost-vector in every regime). RPM-002 + RPM-006 UNRESOLVED
      (data gap; claim definitions NOT altered). Verdicts at
      `experiments/verdict-Stage1.md` and per-regime verdicts.
- [x] **EXP-RPM-CAL** — **Damage-knob calibration pre-experiment
      DONE (AF2-D layer; 33/99 cells).** Per-threshold
      ppl on the AF2-D layer: 0.0-0.5 → 1524.80 (DEGENERATE;
      confirms Stage 1 F5); 0.6 → 697.29; 0.7 → 429.55;
      0.8 → 303.06; 0.9 → 203.60; 1.0 → 88.31. Driver
      stderr=0 across seeds (deterministic eval). The gate
      before Stage 1.5/Stage 2 design is satisfied. Verdict
      at `experiments/EXP-RPM-CAL/verdict.md`. Stage 2 design
      may now use observed ppl as the damage axis.
- [x] **2.15** `EXP-RPM-D0'..D5'` — **Stage 1.5 damage sweep
      (observed-ppl axis, post-EXP-RPM-CAL).** Six damage
      regimes mapped to distinct observed-ppl bands
      (FP16, 88, 204, 303, 430, 697). Same site + recipe
      as Stage 1. 7 arms × 3 seeds = 21 runs per regime;
      126 runs total. Driver NOT modified. **DECIDED
      2026-08-24.** All 6 regimes DECIDED; manifests closed
      in place. Pareto audit confirms T2 IS NOT dominated on
      the joint (3 cap × 5 cost) vector at any regime. **Post-hoc
      random-arm eval confirms trained ≫ random at every damaged
      regime across both axes (arc_easy z +19 to +116;
      lambada z +62 to +262).** RPM-001/002/006 remain UNTESTED
      per their registered thresholds (see rev 2.15 corrective).
- [x] **2.16** Stage 2 v1 layer sweep `EXP-RPM-Lxx`
      (MLP down_proj at layers 0, 8, 15) — **CAL completed;
      tournaments aborted.** Per-site CAL (33 cells × 2 sites)
      showed the TWN damage axis is degenerate on layers 8 and 15
      (ppl 13.67-15.49 across all 11 thresholds; AF2-D's
      threshold axis is the informative one). Tournaments at
      FP16-like ppl would only replicate Stage 1 D0 (trained ≈
      random at FP16 reference) and consume Legion compute without
      new information. **Stage 2 v2 is the next gate.**
      Verdict: `experiments/RPM-L-L15-L8-CAL-verdict.md`.
- [ ] **2.17** Stage 2 v2 layer sweep — **REQUIRED.** Stage 2 v1
      (TWN damage recipe on MLP down_proj layers 0, 8, 15) found
      that the damage axis is degenerate on layers 8 and 15. v2
      must use either (a) a different damage recipe (random mask,
      structured dropout, per-row quantization) that produces
      informative ppl variation at deeper layers, OR (b) different
      layer categories (attention projections q_proj, v_proj) via
      a driver extension (freeze exception required). ≥2 layer
      categories required by RPM-006 PASS+. Before launching: preregister
      the new damage recipe's threshold->ppl mapping on AF2-D first
      (so v2 has a known calibration); freeze metric keys in the
      manifest; specify the ppl sign convention for monotonicity
      tests; document values before any post-hoc corrections.
      (discovery → confirmation).** Small model (100M–500M),
      matched arms: standard ternary vs. native rotated ternary
      parameterization, identical architecture/data/optimizer/
      schedule/budget. Measure loss convergence, KL, downstream
      accuracy/perplexity, gradient conditioning, code-flip rate,
      throughput/memory traffic, physical bits/weight incl.
      metadata, joules/token where trustworthy. **Kill criteria
      preregistered.** H-POST remains NO-SHIP. Large-model
      earns `CONFIRMED_PASS`.
- [ ] **3.3** `EXP-A-05x` — **Pareto report.** Full baseline ladder
      (FP16 → INT8 → strong 4-bit → T1 → T1+T2 → heterogeneous
      map) on quality vs. physical bytes/weight (packing +
      metadata — never "1.58 bits/weight"), memory, operations/
      token, measured latency.
- [ ] **3.4** Track A grade review under the v2 acceptance rule: no pass on
      KL alone; task-relevant improvement, reproduction, competitive physical
      storage/compute, strong-baseline comparison, clean provenance.

### Checkpoints

| ID | Checkpoint | Pass criterion |
|---|---|---|
| CP3.1 | Heterogeneous map (A4) | ≥1 precision map beats uniform ternarization at equal bytes on task-relevant metrics. |
| CP3.2 | Hadamard verdict (A-H1) | Preregistered advantage without breaching kill criteria → `CONFIRMED_PASS` after reproduction; else `FAIL`, Hadamard line closes with H-POST. |
| CP3.3 | Pareto position | ≥1 physically measured configuration on or beyond the baseline-ladder frontier, ≥3 seeds, no evaluation/packing/provenance artifact. |
| **G3→4** | **Track A verdict** | CP3.3 passes → Track A ≥B; dense representation "sufficiently stable" → OLMoE path unblocks (with G2→3). `FAIL` → Track A ≤D; Phase 6. |

---

## Phase 4 — Track B: Adaptive Computation and OLMoE (unlock-ruled)

**Unlock rules (enforced, not scheduled):**

- **B1 oracle gating:** requires A-RP-001 `CONFIRMED_PASS`, **A-RP-LRN**
  at least provisionally supported (REGIME_CONDITIONAL is insufficient —
  the LRN operating band must be characterized for the site/damage in
  question), AF5 task-relevant T2 value above threshold (G2→3). The
  original dependency on A-RP-002 (composite) was superseded 2026-08-25
  per H-RPM-FRAMEWORK-PROPOSAL.md: A-RP-TSP (ternary structural prior)
  supports studying ternary structure, but only A-RP-LRN (training adds
  value beyond the structural prior) justifies a gate whose purpose is
  to decide when a *trained correction* should execute. Adaptive
  precision gating is conditioned on P(trained T2 helps | regime) =
  positive in identifiable regimes.
- **B3 OLMoE:** additionally requires dense-model oracle gating to show
  useful savings and T1/T2 to have survived falsification (G2→3 + CP4.1).

- [ ] **4.1** `EXP-B-01x` — **B1 oracle gating (task-aware).** Per candidate
      token/layer/expert, T1 vs. T1+T2 against a task-relevant loss/output;
      maximum compute saving under a perfect gate. Oracle cannot save
      substantial computation at acceptable quality → stop learned-gate work.
- [ ] **4.2** `EXP-B-02x` — **B2 learned task-aware gating** (after B1
      passes): `L = L_task + λ·C_extra_precision`; sweep λ; separate
      gate-training/evaluation samples; never optimize teacher KL against
      downstream disagreement.
- [ ] **4.3** `EXP-B-03x` — **B3 OLMoE expert-routing × precision-routing**
      on `allenai/OLMoE-1B-7B-0125`. Hypothesis: router confidence predicts
      needed expert precision (high-confidence → T1; uncertain → T1+T2).
      Controls: stock OLMoE; fixed precision; random/heuristic at matched
      activation rate; oracle; learned gating. Never uses OLMoE to prove
      basic ternary viability.
- [ ] **4.4** `EXP-B-04x` — **B4 realized efficiency.** Kernel executions,
      bytes moved, tokens/sec, latency distribution, joules/token.
      Theoretical skipped planes are not evidence.

### Checkpoints

| ID | Checkpoint | Pass criterion |
|---|---|---|
| CP4.1 | Oracle headroom (B1) | Oracle saves substantial computation at acceptable task-relevant quality. `FAIL` → stop Track B learned-gate work. |
| CP4.2 | Gate quality (B2) | Deployable gate captures ≥50% (preregistered) of oracle gain on held-out tasks/lengths. |
| CP4.3 | OLMoE routing signal (B3) | Router-confidence-conditioned precision beats all non-oracle controls at matched activation rate. |
| CP4.4 | Realized cost (B4) | End-to-end measured cost improves over the best static precision map from Phase 3. |
| **G4→6** | **Track B verdict** | CP4.1, CP4.2, CP4.4 pass → Track B ≥B. Static matching adaptive → retain static; record the negative-but-useful result. |

---

## Phase 5 — Track C: Recursive Context Benchmark (independent, parallel)

**Objective:** Establish whether recursive/persistent/indexed context improves
long-context accuracy and/or economics for small/local models. Independent of
ternary success; competent **conventional** model backend; starts in Phase 1.

### Checklist

- [ ] **5.1** `EXP-C-001` — **Benchmark specification (first).** Named
      workloads/datasets/sizes/scoring before any run: retrieval accuracy
      (C1), long-document QA (C2), RLM vs. conventional RAG (C3), model-size
      scaling (C4), tokens/compute/latency economics (C5).
- [ ] **5.2** `EXP-C-01x` — **Controls:** fixed-window prompting;
      straightforward RAG; TORUS recursive context/RLM; where feasible, a
      larger-context model baseline. Equal corpus, model, answer budget.
- [ ] **5.3** `EXP-C-02x` — **Metrics per run:** answer accuracy /
      required-fact recall; retrieval precision/recall; context tokens
      admitted; model calls per answer; wall-clock latency; storage/index
      overhead; failure modes and citation/addressability accuracy.
- [ ] **5.4** Track C grade review. Lookup-latency microbenchmarks do not
      count toward the grade.

### Checkpoints

| ID | Checkpoint | Pass criterion |
|---|---|---|
| CP5.1 | Spec frozen | Workloads/scoring named and immutable before results exist. Change → new EXP ID. |
| CP5.2 | Controls complete | All required controls at E2+ on the frozen spec, with total latency and token economics. |
| **G5→6** | **Track C verdict** | Grade A requires reproduced end-to-end advantage on ≥1 named workload without correctness/security regression. |

---

## Phase 6 — Recombination or Archival

**Objective:** Combine only independently reproduced CONFIRMED_PASSes.
Integration is a new preregistered experiment; component grades do not
guarantee integrated performance. Entry requires: isolated hypothesis pass,
clean-provenance reproduction, task-relevant value, survival vs. simpler
baselines, earned compute/storage complexity.

### Policy

| Outcome | Action |
|---|---|
| A pass, B fail | Retain static heterogeneous precision if Pareto-competitive. |
| A pass, B pass | Combine correction planes with adaptive execution (`EXP-R-001`). |
| A fail, C pass | Ship recursive context independently of ternary inference. |
| A fail | Track B remains locked regardless of its own evidence. |
| All fail | Archive hypotheses; preserve engineering lessons; close the program. An F is a useful research result. |

### Checklist

- [ ] **6.1** Preregister the recombination experiment with its own baselines.
- [ ] **6.2** Integrated run against the full baseline ladder.
- [ ] **6.3** Final report: the five evidence categories (software
      correctness; representation/capacity; downstream quality; systems
      efficiency; end-to-end usefulness) reported separately.
- [ ] **6.4** Final grade ledger for all tracks, subtracks, and claims;
      public artifacts.

### Checkpoints

| ID | Checkpoint | Pass criterion |
|---|---|---|
| CP6.1 | Integration non-regression | Combined system ≥ the better of its components on the preregistered scorecard; integration overhead measured. |
| **Final** | **Program verdict** | Reproduced Pareto improvement on measured cost (Tracks A/B line), or independent Track C advantage — or an honest archive. |
