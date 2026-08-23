# TORUS Research Roadmap

**Status:** active — supersedes `docs/ROADMAP.md` (retained as historical record)
**Revision:** 2.7 (2026-08-23) — section 2.12 (EXP-AF-002-D
damaged-PTQ-start matched-storage tournament) DONE with
**PASS_PLUS** on the architecture-vs-training-signal question.
Trained t2_ternary recovers the broken-PTQ base from ppl 425.76
to ppl 20.96 (20.3x recovery) AND pulls ahead of random_t2_ternary
by 25-227 sigma on every capability metric. The architecture's
regime of dominance is now bounded: on a calibrated base
(AF2/AF2-R), T2 is Pareto-competitive but silent-vs-random; on a
damaged base (AF2-D), T2 pulls ahead of random by 25+ sigma.
**Driver bug fixed**: T2TernaryAdapter.is_untrained now set in
__init__ (was class-defaulted to False; caused AF2-R's
aggregate.json to misclassify random_t2_ternary as trained - the
per-seed data was correct, only the audit classification was).
**Revision 2.6** (2026-08-23) — section 2.12 added:
EXP-AF-002-D (AF2-D damaged-PTQ-start matched-storage tournament
under v2.3 cost-vector framing).
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

T2 is promoted to a validated representation mechanism only if **all five**
hold (`10` §15):

| ID | Checkpoint | Pass criterion |
|---|---|---|
| CP2.1 | Budget control (AF1) | T1+T2 materially outperforms equal-budget T1 continuation. **REV 2.3 STATUS:** provisional FAIL only — A-RP-001 is in `PROVISIONAL_FAIL / REPRODUCTION_REQUIRED` after `EXP-AF-001` (2026-08-22, git `39be76c`); CONFIRMED_FAIL only after §2.10 (AF1-R) clean reproduction. CP2.1 cannot pass on this evidence; the equal-training-time branch closes only after `A-RP-001 → CONFIRMED_FAIL`. |
| CP2.2 | Robustness (AF3, AF8) | Gain survives multiple seeds and reasonable initialization variation (not FRAGILE). |
| CP2.3 | Capability (AF5) | Held-out downstream improvement, not only KL reduction. |
| CP2.4 | Storage competition (AF2, AF7) | T2 quality-per-cost-vector **Pareto-competitive** with ≥1 equal-storage non-ternary correction baseline. **REV 2.3:** each arm reports the cost vector `C = (deployed bytes, training FLOPs, inference ops/token, memory traffic/token, latency, joules/token)` separately; the table is multi-dimensional, not scalar. |
| CP2.5 | Provenance (AF8) | Reproduced under clean immutable provenance → CONFIRMED states. |
| **G2→3** | **A-F verdict** | **REV 2.3:** even a `CONFIRMED_PASS` on A-RP-001 is no longer the gate condition (A-RP-001 is on track to `CONFIRMED_FAIL`; the equal-training-time branch is closing). The gate condition becomes `(A-RP-002 PROVISIONAL_PASS, A-RP-003 PROVISIONAL_PASS or CONFIRMED_FAIL-clean, AF5 above threshold, AF8 clean reproduction)` — and Track-B B1 prerequisite wording is rewritten under v2.3 to allow A-RP-002 (storage) to substitute for the historical A-RP-001 requirement when the v2.2 wording is rendered obsolete. Track B stays locked through the A-F suite either way. |
---

## Phase 3 — Track A: Heterogeneous Precision, Hadamard, Decision

**Objective:** Convert validated representation behavior into a Pareto
verdict; settle native Hadamard on a small controlled model.

**Subtracks:** A4 heterogeneous precision, A5 native Hadamard training.

### Checklist

- [ ] **3.1** `EXP-A-04x` — **A4 heterogeneous precision map.** Sensitive
      layers stay FP/INT8/INT4; tolerant layers take T1 or T1+T2 (per A-F
      outcome). Layer-adaptive precision before token-adaptive precision.
- [ ] **3.2** `EXP-A-H1` — **Native Hadamard controlled training (discovery
      → confirmation).** Small model (100M–500M), matched arms: standard
      ternary vs. native rotated ternary parameterization, identical
      architecture/data/optimizer/schedule/budget. Measure loss convergence,
      KL, downstream accuracy/perplexity, gradient conditioning, code-flip
      rate, throughput/memory traffic, physical bits/weight incl. metadata,
      joules/token where trustworthy. **Kill criteria preregistered.**
      H-POST remains NO-SHIP. Large-model Hadamard stays locked until A-H1
      earns `CONFIRMED_PASS`.
- [ ] **3.3** `EXP-A-05x` — **Pareto report.** Full baseline ladder (FP16 →
      INT8 → strong 4-bit → T1 → T1+T2 → heterogeneous map) on quality vs.
      physical bytes/weight (packing + metadata — never "1.58 bits/weight"),
      memory, operations/token, measured latency.
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

- **B1 oracle gating:** requires A-RP-001 `CONFIRMED_PASS`, A-RP-002 at least
  provisionally supported, AF5 task-relevant T2 value above threshold (G2→3).
- **B3 OLMoE:** additionally requires dense-model oracle gating to show
  useful savings and T1/T2 to have survived falsification (G2→3 + CP4.1).

### Checklist

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
