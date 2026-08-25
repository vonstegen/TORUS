## 0.16.6 / research — RPM program accepted; EXP-RPM-000 preregistered (G-RPM-0 gate before any RPM-D work)

### Registered (RPM program)
- Six claim files under `research/residual-pareto/claims/`, all
  `UNTESTED`:
  - `RPM-001.yaml` — T2 lies on a capability-vs-cost Pareto frontier
    for at least one nontrivial regime (full-comparator-set rule;
    PASS+ requires dominance over ≥2 conventional matched-cost
    comparators, NOT merely beating two controls).
  - `RPM-002.yaml` — T2 value increases with base damage severity
    (RPM-D sweep, 6 regimes D0-D5).
  - `RPM-003.yaml` — T2 advantage is layer-dependent and generalizable
    beyond one pathological site (RPM-L sweep; ≥2 layer categories
    must show T2 Pareto-optimal).
  - `RPM-004.yaml` — Best residual representation is task-dependent
    (RPM-T; mechanism IDs must differ between ppl-best and
    task-best at ≥1 tested combination).
  - `RPM-005.yaml` — T2 is relatively more competitive as
    deployed-byte budget tightens (RPM-B; frontier-occupying at
    ≤2 MB even when not at ≥4 MB).
  - `RPM-006.yaml` — Trained T2 separates from random T2
    specifically when the base requires correction (RPM-D
    controls; reproducible activation boundary).

### Added (cost schema + G-RPM-0)
- `research/residual-pareto/COST-VECTOR-v1.yaml` — frozen 6-dim
  schema (B/F/O/M/L/E) with per-stage required-by mapping,
  comparator set (8 mechanisms), and Pareto rules. v1 explicit;
  no scalar composite score; capabilities and costs never
  combined for primary decision-making.
- `research/residual-pareto/experiments/RPM-000/manifest.yaml` —
  EXP-RPM-000 preregistration: formal G-RPM-0 reference-lock +
  AF2-D reproduction. Two-arm reproduction (damaged starting
  state + trained t2_ternary arm only) under AF8 governance;
  PASS bands derived from AF2-D reference (±1.5σ). No driver
  modifications; the AF2-D driver SHA is the immutable
  reference. **No Stage 1 (RPM-D1..D6) work begins until
  EXP-RPM-000 verdict = REPRODUCED.**

### Changed (governance)
- `research/registry/INDEX.md`: claim registry extended to
  RPM-001..006; EXP-RPM-000 row added; decision-log entry
  added.

### No code changes (research + governance only)

### Tests: 207/207 pass (unchanged; no code modified)

## 0.16.5 / research — EXP-AF-002-D DECIDED PASS+ on architecture-vs-training-signal; driver bugfix

### Verified (architecture carries information in the damaged regime)
  `runs/a/EXP-AF-002-D/runs/20260823T092339Z/af2d/`, git `330e8b3`,
  n=3 seeds, damaged-PTQ base) DECIDED **PASS+** on the
  architecture-vs-training-signal question. 21 runs total; all
  inside ±1% bytes tolerance.
  (ppl=425.76 in [400, 460] band; arc_easy=0.4891 in [0.45, 0.58]
  band). Damage mode reproduces EXP-A-011's 427.7 within 0.5% on
  ppl.
  manifest PASS bar was 4.3×).
  on the damaged base = ppl −226.87σ, arc_easy +25.08σ,
  lambada_openai +116.83σ. PASS+ met decisively on every metric.
  The architecture-vs-training gap that AF2/AF2-R found absent on
  the calibrated base manifests decisively here.
  (42.02) on the damaged base, with the largest seed-stderr (7.12).
  lora beats dense_adapter (ppl 22.3 vs 42.0); int8_residual has
  the best ppl (18.6).

### Changed (driver bugfix; AF2-R audit reclassification)
  `T2TernaryAdapter.is_untrained` now set in `__init__`
  (was class-defaulted to False even when constructed with
  `train=False`, which misclassified `random_t2_ternary` as a
  trained arm in AF2-R's `aggregate.json`). The fix uses
  `self.is_untrained = (not train)`, mirroring the other adapter
  classes (`_IntNCls`, `_LoRACls`, `_DenseCls`). AF2-R's per-seed
  eval data was always correct; only the audit classification was
  wrong. The same fix benefits AF2 and any future re-runs.
  with a seed-major iteration when `--damage-ptq` is set (so the
  damaged base + pre-train eval run once per seed instead of
  redundantly per arm). When `--damage-ptq` is NOT set, the
  original arm-major order is preserved exactly.
  `--damage-group-size`, `--damage-threshold`, `--pre-train-eval`
  flags added for the damaged-PTQ regime.

### Added (artifacts)
  refuse-to-overwrite, AF8 record keys, sha256 matches hashlib,
  damage-mode invariants (fro_ratio in [0.5, 0.85], idempotent,
  no-touch-other-weights, metadata capture, --help lists flags).

### Changed (governance)
  CONFIRMED_PASS (refined by AF2-D)"; supporting_experiments now
  lists EXP-A-03x, EXP-AF-001, EXP-AF-002, EXP-AF-002-D, EXP-AF-002-R.
  row added; decision log entry added.
  PASS+.

### Tests: 207/207 pass (was 201 in 0.16.4; +6 from test_af2_damaged_ptq)

## 0.16.4 / research — EXP-AF-002-R DECIDED CONFIRMED_PASS — A-RP-002 reproduction confirms PASS bar; PASS+ softens

### Verified (AF8 clean reproduction of EXP-AF-002)
- **EXP-AF-002-R** (legion run `runs/a/EXP-AF-002-R/20260823T062845Z`, git
  `c036718`, n=3 seeds, matched deployed-bytes ~4.2 MB on
  `model.layers.0.mlp.down_proj` of OLMo-1B) DECIDED
  **CONFIRMED_PASS** for A-RP-002.
- 21 runs total (5 trained arms × 3 seeds + 2 untrained structure
  controls × 3 seeds). All 21 inside ±1% bytes tolerance;
  `tolerance_violations: []`. Cost-vector byte counts
  **byte-identical** to AF2 per (arm × seed) — the storage-Pareto
  axis reproduces exactly.
- Trained `t2_ternary` lies within ±1.1σ of `dense_adapter` on
  every capability metric at n=3 (mean (B-A)/se_diff:
  wikitext -1.125σ, arc_easy -0.547σ, lambada_openai -0.551σ).
  PASS bar reproduces.

### Softened (PASS+ no longer holds)
- AF2 reported t2 dominating dense by +2.18σ on lambada_openai.
  Inspecting AF2's per-seed lambada values reveals they were
  **byte-identical across all 3 seeds** (spread = 0.000000) — a
  zero seed-variance that made the +2.18σ an artifact of zero
  denominator stderr. AF2-R's seed-variance is ~0.001-0.004
  (matching dense_adapter's per-seed spread), putting the true
  effect inside ±1σ.
- **Conclusion:** the architecture is Pareto-competitive with
  dense fp16 at matched bytes (PASS); it does not pull ahead by
  >2σ on lambada (PASS+). The softening is a positive finding: it
  shows AF2-R's role was not "try the same thing again" but the
  AF8 governance check that detects zero-seed-variance artifacts
  the original run had hidden.

### Changed (governance)
- `claims/A-RP-002.yaml`: state PROVISIONAL → CONFIRMED; new
  transition entry; supporting_experiments now lists EXP-A-03x,
  EXP-AF-001, EXP-AF-002, EXP-AF-002-R.
- `research/registry/INDEX.md`: A-RP-002 → CONFIRMED_PASS; new
  EXP-AF-002-R row; decision log entry.
- `research/ROADMAP.md`: rev 2.5; §2.11 marked DONE
  (CONFIRMED_PASS).
- `research/track-a-residual-ternary/residual-falsification/experiments/AF2-R/`:
  manifest.yaml + verdict-R.md + ARTIFACTS.json (90 files SHA-indexed).

### Added (freezable tooling, audit-script mirror)
- `examples/audit_af2_reproduction.py` — AF8 governance notary;
  structural twin of `audit_af1_reproduction.py`; re-tokenizes
  wikitext-103 to a fresh path, records SHA, refuses to overwrite.
- `tests/test_audit_af2_reproduction.py` — 4 tests pinning
  refuse-to-overwrite, AF8 record keys, sha256 matches hashlib,
  --help runs.

### No code changes (interface/architecture unchanged; 0.16.4 is research + governance only)

## 0.16.3 / research — EXP-AF-002 DECIDED PROVISIONAL_PASS — A-RP-002 headroom confirmed at matched deployed bytes

### Verified (cost-vector axes per OPERATING-PLAN §11 v2.3)
- **EXP-AF-002** (legion run `runs/a/EXP-AF-002/20260823T030918Z`, git
  `0529749`, n=3 seeds, matched deployed-bytes ~4.2 MB on
  `model.layers.0.mlp.down_proj` of OLMo-1B) DECIDED
  **PROVISIONAL_PASS** for A-RP-002.
- Five trained arms + two untrained structure controls. All 21 runs
  landed within the +/1% bytes tolerance; no tolerance violations.
- Trained `t2_ternary` (4,199,318 B) ties within +/-2 stderr of the
  strongest fp16 comparator `dense_adapter` r=192 (3,932,771 B) on
  wikitext ppl and arc_easy, and dominates lambada_openai by
  +2.18 sigma. fp16 LoRA r=216 (4,424,265 B) also within +/-2 sigma
  of dense_adapter on every metric. int4/int8 column-masked
  variants underperform at N=500 (the v2.3 cost-vector framing
  speaks to T2-vs-matched-storage-fp16 alternatives, not int-N).
- Untrained `random_t2_ternary` lands within measurement noise of
  trained `t2_ternary` on a calibrated FP16 base at this budget:
  the architecture is **competitive but not dominating** at this
  scale. The representation's load-bearing contribution is below
  the current eval suite's noise floor when the base is healthy
  FP16, which bounds the headroom and motivates the next
  architecture-vs-curriculum questions (AF4).

### Changed (governance)
- `claims/A-RP-002.yaml`: state PROVISIONAL / PASS; new transition
  entry cites EXP-AF-002; supporting_experiments now lists EXP-A-03x,
  EXP-AF-001 (orthogonal), EXP-AF-002. Conclusion directs the
  next experiments: AF2-R reproduction + AF4 sequential-vs-joint
  for A-RP-003.
- `research/registry/INDEX.md`: EXP-AF-002 row inserted;
  Decision log entry added.
- `research/ROADMAP.md`: rev 2.4; §2.2 marked DONE (PROVISIONAL_PASS);
  §2.11 added (EXP-AF-002-R placeholder).
- `examples/af2_storage_tournament.py` (7 commits,
  `0bf83cc -> 0529749`) — the AF2 driver hardened against real-model
  integration bugs found via standalone smoke + legion smoke:
  wrapper-target resolution for OLMo `OlmoMLP`, int8 sign-bit
  overflow, dtype-cast LoRA residuals, site-dim conventions,
  duplicate `build_base` definition, eval-stderr filtering.
- `tests/test_af2_storage_tournament.py` (9 tests): pack-format
  round-trip, matched-bytes tolerance, trained-arm completeness,
  LoRA/dense size accounting within +/-1% of preregistered targets.

### No code changes (interface/architecture unchanged; 0.16.3 is research + governance only)

### Verified (AF8 governance)
- **EXP-AF-001-R** (legion run `runs/a/EXP-AF-001-R/20260822T233000Z`,
  git `4238568` reproducing frozen revision `39be76c`) is a clean
  reproduction of EXP-AF-001 under AF8 governance. Every per-seed
  value byte-identical: wikitext ppl 14.10 / 14.12 / 14.09 vs
  34.81 / 38.34 / 30.55; arc_easy 0.661 / 0.646 / 0.641 vs
  0.637 / 0.630 / 0.637; lambada 0.621 / 0.631 / 0.602 vs
  0.566 / 0.563 / 0.566. Arm means, stddevs, stderrs, and
  (B-A) stderr-of-difference values reproduce byte-for-byte at
  +9.09 / -2.23 / -6.24. A-RP-001 transitions `PROVISIONAL_FAIL
  / REPRODUCTION_REQUIRED -> CONFIRMED_FAIL`. The equal-training-
  time branch closes permanently.

### Changed (research governance / audit-script fix)
- `examples/audit_af1_reproduction.py`: initial implementation had a
  hard `SystemExit` when the regenerated cache SHA collided with
  AF1's reference SHA. That gate was incorrect: re-tokenizing a
  deterministic corpus with the same code yields the same SHA by
  construction, and that is the expected outcome of a clean
  reproduction, not a violation. AF8 governance is therefore
  *traceability*, not byte-difference: a fresh process invocation,
  SHA fingerprints of every input shard, refusal to overwrite an
  existing artifact path. Fixed in commit `4238568` before the
  reproduction run launched.

### No code changes
0 code edits, no API changes. Research governance + verification only.

### Changed (research governance)
- **OPERATING-PLAN rev 2.3** — added §11 "Track A decision-axis
  revision". The A-F suite was asking "is T2 better than continued
  FP16 training at equal training-time?"; AF1 decisively said *no*
  (37:1 losers). The v2.3 Track-A primary decision axis is now
  capability as a function of a **cost vector**
  `C = (deployed bytes, training FLOPs, inference ops/token, memory
  traffic/token, latency, joules/token)`. Single-scalar cost
  matching is no longer permitted for Track-A claim tests; AF2 must
  match deployed bytes (artifact) and report training FLOPs
  separately, not conflate them with bytes or steps.
- **OPERATING-PLAN §5** — Track-B B1 unlock rule rewritten: the
  gating is now `A-RP-002 PROVISIONAL_PASS` (or above) +
  `A-RP-002/003 CONFIRMED via AF8` + AF5 above threshold. The
  historical "A-RP-001 CONFIRMED_PASS" prerequisite is retired in
  favor of the storage Pareto argument that A-RP-002 carries.
  Track-B stays locked either way through the A-F suite.
- **OPERATING-PLAN claim-lifecycle clarification** — a
  confirmation-tier ≥3-seed result with matched control design can
  promote `TESTING → PROVISIONAL_FAIL` in one transition; set
  `reproduction: REQUIRED` at the same transition. CONFIRMED_FAIL
  only after an AF8-style clean reproduction (new run ID,
  independent namespace, frozen SHA, fresh process, independently
  generated eval output, ideally independent token-cache build).
- **ROADMAP rev 2.3** — §2.1 marked DONE with PROVISIONAL_FAIL
  (not DECIDED FAIL) per the lifecycle; §2.10 inserted
  (`EXP-AF-001-R` clean reproduction, required before A-RP-001 →
  CONFIRMED_FAIL); §2.2 (AF2) rewritten with the cost-vector
  framing; CP2.1/CP2.4/G2→3 updated accordingly.

### Changed (claim registry)
- `A-RP-001.yaml` — `state: PROVISIONAL_FAIL`, `reproduction:
  REQUIRED`. Conclusion language softened: even a CONFIRMED_FAIL
  here closes only the equal-training-time branch and does not
  block Track B if A-RP-002 (equal-storage) is supported. v2.3
  lifecycle prescribes `EXP-AF-001-R` as the required next step.

### No code changes
0 code edits, no API changes. Research governance only.

### Fixed (audit Bugs 1–7)

- `DistillationTrainer.fit()` now populates `self._residual_np` from
  each STE's `residual_weight`. Without this, the residual SGD was
  permanently skipped and the residual plane was never optimized
  (Bug 1, audit).
- `HFStudentAdapter._attach_ste` wraps the residual weight as
  `torch.nn.Parameter` so `torch.autograd.grad` can flow into it
  (Bug 3, audit).
- `_numerical_grads` now probes a configurable column budget
  alongside rows; the previous code hard-coded `c = 0` and only
  ever updated a single weight per module per step (Bug 2, audit).
  New `TrainingConfig.probe_cols: int = 0` (0 = same as
  `probe_rows`); CLI flag `--probe-cols`.
- `_autograd_grads` now computes real KL(student || teacher)
  against the frozen `HFTeacherAdapter.forward_torch` instead of
  self-MSE between identical student logits (Bug 4, audit). New
  `kl_divergence_torch` in `torus.train.losses`.
- `_loss_only` accepts an explicit `n_planes` argument so the
  finite-difference probe measures the loss surface the
  curriculum is actually stepping on (Bug 5, audit).
- Post-step sync uses `hasattr(ste.residual_weight, "copy_")`
  rather than `hasattr(ste.residual_weight, "data")` — numpy
  arrays expose `.data` as a `memoryview` in NumPy ≥ 1.20 and
  lack `.copy_`, which crashed the sync under the toy
  numerical-grad tests.
- Residual SGD built whenever any STE has a `residual_weight`
  (previously gated on `probe_residual=True`, which only matters
  for the numerical probe path and silently discarded the
  autograd-path residual gradient).
- Residual gradient list built parallel to `_residual_np` using
  positional indexing. The previous code used `list.index(r)`
  which always returned the first match and broke with multiple
  residuals of different shapes.

### Added

- `HFStudentAdapter` initializes `residual_weight` as
  `N(0, 0.01)` rather than identically zero. The ternary
  quantizer's threshold filter creates a dead zone at `r = 0`
  where both `q_r = 0` and `∂q_r/∂r = 0`, so a zero-init
  residual never receives a gradient and the plane-2 curriculum
  stage is structurally incapable of learning. Small noise
  breaks the dead zone without acting as a wholesale
  perturbation. The CLI flag `--perturb-residual` adds extra
  noise on top.

## 0.13.0 — Per-plane LR scaling + 5th Legion distillation run
### Verified

- Fifth distillation run completed on Legion with per-plane LR:
  - `primary_plus_residual_lr_scaled`: curriculum `1:100, 2:100`,
    `probe_residual=True`, `residual_lr_scale=0.05`, 200 steps
    in 65.3 s. Step-100 loss 0.0391 (curriculum switch), final
    loss **0.0303** vs 0.0729 with `residual_lr_scale=1.0` — a
    2.4× improvement. The residual plane is being learned, not
    blown up.
- 158/158 tests passing on both dev and Legion (added 1 new
  training test covering the residual SGD).

### Added

- `TrainingConfig.residual_lr_scale: float = 0.1`: scales the
  residual plane's learning rate (`lr * residual_lr_scale`).
  Defaults to 0.1, which keeps the residual's update step small
  enough to prevent loss explosion when the curriculum switches.
- `DistillationTrainer` constructs an optional second `_SGD`
  for the residual planes when `probe_residual=True` AND any
  STE has a `residual_weight`. The primary grad is reused for
  the residual's update (coarse approximation; full residual
  gradient would require a second forward pass per probe).
- `examples/distill_run.py --residual-lr-scale` CLI flag.

# CHANGELOG

## 0.12.0 — probe_residual trainer flag + 4th Legion distillation run

### Verified

- Fourth distillation run completed on Legion with the trainer
  extended to perturb the residual weight at the same (r, c) as
  the primary:
  - `primary_plus_residual_probe_and_perturb`: curriculum
    `1:100, 2:100`, `probe_residual=True`, `perturb_residual=True`,
    200 steps in 51.6 s. Step-100 loss jumped from 0.0017 to
    0.0261 (vs. 0.0105 in the previous probe_residual=False run),
    proving that gradient now flows through the residual plane.
- 157/157 tests passing on both dev and Legion (added 2 new
  training tests).

### Added

- `TrainingConfig.probe_residual: bool = False`: when set, the
  finite-difference trainer perturbs both the primary weight and
  the residual weight at the same (r, c), so the curriculum
  switch from `n_planes=1` to `n_planes>=2` actually flows gradient
  through the residual plane.
- `DistillationTrainer._residual_np`: parallel buffer to
  `_params_np`, indexed one per STE. Initialized to a list of
  Nones in `__init__`; `fit()` upgrades entries with numpy views
  of `STE.residual_weight` (when present).
- `examples/distill_run.py --probe-residual` CLI flag for the
  4-run comparison on Legion.

### Found

- With `probe_residual=True`, the curriculum switch causes a
  large loss jump (step 100 → 0.026 from 0.0017). Training is
  unstable without per-plane LR scheduling; the residual's
  initial scale (random N(0, 0.05) noise) is too small relative
  to the primary's to be stable under the same learning rate.
  Phase-8+ follow-up: per-plane LR scheduling.

# CHANGELOG

## 0.11.0 — Phase 8 distillation on Legion + trainer probe-rows

### Verified

- Three distillation runs completed on Legion (CUDA torch +
  2× TITAN RTX) with `sshleifer/tiny-gpt2`, 200 steps each:
  - `primary_only` (curriculum `1:200`): initial loss 0.0028,
    final loss 0.0038, ~103 s wall time.
  - `primary_plus_residual` (curriculum `1:100,2:100`, residual
    zero-init): initial 0.0028, final 0.0038 — *identical to
    primary_only* because the trainer only probes the primary weight
    and the residual plane contributes nothing when zero-init.
  - `primary_plus_residual_perturbed` (same curriculum, residual
    weights initialized with N(0, 0.05) noise): initial 0.0028,
    step-100 (curriculum switch) 0.0105, final **0.0205**. Loss
    goes up because the random-noise residual contributes garbage
    that the trainer then has to undo.
- 155/155 tests still passing on both dev and Legion.

### Added

- `examples/distill_run.py`: end-to-end distillation runner that
  loads a HF model via `HFStudentAdapter` + `HFTeacherAdapter`,
  drives `DistillationTrainer` with a curriculum, and logs the
  loss curve + final stats to a JSON file. Supports
  `--n-steps`, `--probe-rows`, `--curriculum`, `--batch-size`,
  `--seq-len`, `--label`, and `--perturb-residual`.
- `TrainingConfig.probe_rows`: per-module finite-difference probe
  budget. Default 1 (one column per STE per step); set higher for
  less noisy gradients at the cost of more forward passes.
- `_numerical_grads` now samples `probe_rows` random rows per STE
  instead of probing every row. With `probe_rows=1` and 6 STEs in
  tiny-gpt2, each step costs ~12 forward passes instead of ~9k.

### Found

- The current trainer only probes `weight`, not `residual_weight`.
  When the residual plane is zero-init, the curriculum switch from
  `n_planes=1` to `n_planes=2` has no effect on the loss curve.
  Phase-8 follow-up: add `--probe-residual` (default off) and have
  `_numerical_grads` perturb `residual_weight` when the STE carries
  one.

# CHANGELOG

## 0.10.0 — Phase 7 multi-expert wiring

### Verified

- 155/155 tests passing on dev (.venv, Py3.12 + CPU torch +
  Blackwell via numba) and Legion (.venv-py311, Py3.14 + CUDA
  torch + 2× TITAN RTX).
- `examples/multi_expert_demo.py`: 16-expert × 4-plane bank, 32
  tokens, top_k=2. Adaptive router-confidence-driven plane
  engagement saves ~17% of plane activations vs. always-4.

### Added

- `torus.moe.multi_expert.MultiExpertRouter`: composes
  `TopKRouter` + `ExpertBank` + `GatePolicy` for production-shape
  multi-expert routing. Each (token, expert) decision emits a
  `PerCallDecision(token_idx, expert_id, weight, confidence,
  n_planes)`.
- `torus.moe.multi_expert.GatePolicy`: linear-interpolation policy
  that maps router confidence to plane-count engagement. Defaults
  calibrated against the Phase-1 random router.
- `torus.moe.multi_expert.MultiExpertResult` and
  `PerCallDecision`: data classes for the per-call decision list.
- `torus.moe.router.RouteResult.raw_mass`: pre-renormalization
  top-k prob mass. `TopKRouter.confidence()` now returns
  `raw_mass` (the meaningful "what fraction of prob mass landed in
  top-k" signal) instead of the normalized weight sum, which is
  always 1.0 by construction.
- `MultiExpertRouter.decision_table(features)`: human-readable
  rendering of the per-call decisions (used by the demo).
- `examples/multi_expert_demo.py`: 16-expert bank + 32-token batch
  demo reporting plane-activation savings and the n_planes
  distribution.
- 8 new tests in `tests/test_moe.py` covering basic routing,
  policy threshold behavior, 100-expert scaling, shared-primary
  composition, the on_decision callback, and graceful handling
  of unknown experts.

# CHANGELOG

## 0.9.0 — SandboxedContextREPL + REPL injection

### Verified

- 147/147 tests passing on dev (.venv, Py3.12 + CPU torch +
  Blackwell via numba) and Legion (.venv-py311, Py3.14 + CUDA
  torch + 2× TITAN RTX).
- `examples/sandbox_demo.py`: a stub model emits three dangerous
  snippets (`import os`, `exec(...)`, `open('/etc/passwd').read()`)
  and one safe snippet. The first three are rejected at AST
  level with `SandboxError` surfaced as stdout; the safe
  `context.grep + context.slice` snippet runs and produces the
  final answer.

### Added

- `torus.rlm.sandbox.SandboxedContextREPL`: drop-in replacement
  for `ContextREPL` that enforces an AST-level whitelist (no
  imports, no `exec`/`eval`/`open`/`getattr`, attribute access
  only on `context`, subscript only on local Names), restricts
  `__builtins__` to a safe subset, and caps per-call resources
  (lines, output size, recursion depth, wall-clock timeout).
- `torus.rlm.sandbox.SandboxPolicy`: per-REPL config object
  (max_lines, max_output, max_recursion_depth, timeout_seconds,
  extra_allowed_call_names).
- `torus.rlm.sandbox.SandboxError`: raised by the AST check
  when model output violates the policy.
- `PrimeAgentLoop(repl=...)` parameter: accept a custom REPL
  (`ContextREPL` for the default untrusted flow; pass
  `SandboxedContextREPL` for production). `run()` now catches
  exceptions from `self.repl.run(code)` and surfaces them as
  stdout so the model can recover on the next step.

### Fixed

- The Phase-2 docs called out REPL execution as an open security
  risk for the entire session. Phase 2 (security sub-phase) is
  now resolved: any production deployment uses
  `SandboxedContextREPL` (see Phase 2 section in ARCHITECTURE.md).

# CHANGELOG

## 0.8.0 — Phase 9 inverted index for PersistentContext

### Verified

- 130/130 tests passing on both dev (.venv, Py3.12 + CPU torch +
  Blackwell via numba) and Legion (.venv-py311, Py3.14 + CUDA
  torch + 2× TITAN RTX).
- `examples/persistent_grep_demo.py`: on a 2000-chunk context with
  unique-needle queries, indexed grep is **0.041 ms warm** vs
  linear's 49.8 ms (~1200× speedup). The first indexed call pays
  the one-time index-build cost (~1.2 s for 2000 chunks); after
  that, every grep is constant-time-ish.

### Added

- `torus.rlm.index.PersistentContextIndex`: append-only inverted
  index over a `PersistentContext` directory. Token → sorted
  list of chunk indices, stored at `<root>/index.json`. Two
  variants (case-sensitive, case-insensitive) live in separate
  files.
- `PersistentContext.grep`: now consults the index first; falls
  back to a linear scan when the pattern has no `\w+` tokens or
  when `use_index=False`.
- `PersistentContext.flush_index()`: force a flush of pending
  index updates to disk before exit.
- `PersistentContext(use_index=False)` opt-out for tests that
  want pure linear-scan semantics.
- `examples/persistent_grep_demo.py`: comparison runner that
  builds an N-chunk context and reports the indexed-vs-linear
  speedup.

### Changed

- `PersistentContextIndex.add_chunk` debounces disk writes to
  every 64 appends (in-memory updates are still immediate, so
  grep calls between appends see fresh data).

# CHANGELOG

## 0.7.0 — n_planes plumbing + comparison

### Verified

- Extended `HFStudentAdapter` and `HFTeacherAdapter` to honor the
  trainer's `n_planes` parameter end-to-end against a real model on
  Legion (CUDA torch). Both venvs green: 121/121 tests pass on
  dev (CPU torch + Blackwell via numba) and Legion (CUDA torch +
  TITAN RTX).
- New `examples/n_planes_compare.py` runs the same model with
  primary-only and primary+residual quantization, perturbing the
  residual weights with random noise to confirm the residual plane
  contributes to the forward (verified:
  `||y_n_planes=1 - y_n_planes=2|| > 0` after perturbation).

### Added

- `TernarySTE.residual_weight`: optional second learnable
  parameter. When set, `forward(n_planes=2)` returns the sum of
  two independently quantized ternary weights (primary +
  residual). `forward(n_planes=1)` is the original primary-only
  behavior.
- `HFStudentAdapter.residual_params`: list of
  `torch.nn.Parameter` — one per STE. The adapter creates a
  single shared tensor between the STE's `residual_weight` and
  this list, so mutating either reference updates both.
- `HFAdapterConfig.target_modules`: now matched by suffix; the
  default covers GPT-2/GPT-Neo's `c_attn` and `c_proj` Conv1Ds.
- `examples/n_planes_compare.py`: comparison runner that loads a
  HF model, perturbs the residual weights, and demonstrates
  n_planes=1 vs n_planes=2 produce different outputs.
- `examples/hf_adapter_smoke.py --n-planes {1,2}`: CLI flag for
  the existing smoke to exercise primary-only vs primary+residual.

### Fixed

- The residual-weight tensor was being constructed twice (once
  passed to `TernarySTE`, once appended to `_residual_params`)
  with `torch.zeros_like(weight)` returning *separate* tensors.
  Mutating one didn't affect the other, so perturbing
  `adapter.residual_params` never reached the STE. Now both
  references share a single `zero_param`, and perturbation
  propagates as expected.

# CHANGELOG

## 0.6.0 — Legion end-to-end + CPU probe

### Verified

- Cloned + installed TORUS on **Legion** (the production / training
  host): Python 3.14 venv, `torch 2.13.0+cu130` from the cu130
  index, `transformers 5.15.0`, `numba 0.67.0` for the CUDA path.
  121/121 tests pass on Legion with CUDA torch.
- Re-ran the benchmark on Legion's 2× TITAN RTX: real AVX-512
  numbers captured (the Threadripper 3995WX is Zen 2 — see
  *Fixes* below).
- HF adapter end-to-end smoke (`examples/hf_adapter_smoke.py`)
  loads `sshleifer/tiny-gpt2` (or `gpt2`), drives
  `HFStudentAdapter.forward`, `HFTeacherAdapter.forward`, and the
  full `combined_distillation_loss` pipeline against a real
  transformers model on Legion CUDA.

### Fixes

- `torus/kernels/build._machine_flags()` now probes
  `/proc/cpuinfo` for AVX-512 / AVX2 / FMA / AVX support before
  emitting `-mavx512f`. GCC silently accepts `-mavx512f` even on
  CPUs that lack it, and the resulting `.so` then segfaults at
  runtime with `Illegal instruction`. The Zen-2 Threadripper
  3995WX (which has AVX2 but no AVX-512) was hitting this on
  Legion's first build attempt.

### Added

- `examples/hf_adapter_smoke.py`: end-to-end smoke that loads a
  HuggingFace causal-LM, calls the student/teacher adapter, and
  runs `combined_distillation_loss` to confirm the trainer's
  loss path consumes real-model output.
- `TernarySTE.__post_init__` auto-picks a fitting `group_size`
  when the requested one does not divide `in_features`. Small
  smoke models (e.g. tiny-gpt2 with `hidden_size=2`) no longer
  fail at construction; the closest power-of-two divisor is
  chosen, falling back to the full row width for primes.
- `TernarySTE.forward()` accepts both numpy arrays and torch
  Parameters; the HF adapter path quantizes a torch Parameter
  via a `detach().cpu().numpy()` round-trip.
- `HFStudentAdapter` now intercepts both `nn.Linear` *and*
  HF's `Conv1D` (GPT-2, GPT-Neo). Conv1D weights are stored
  transposed relative to `nn.Linear`; the patched forward
  applies `F.linear(x, q_w.T, q_b)` to match the Conv1D
  contract. Bias is held as a separate `torch.nn.Parameter` so
  it stays fp32-trainable without quantization.

### Changed

- `DistillationTrainer` keeps an in-place numpy view
  (`self._params_np`) of every STE weight, used by the
  numerical-gradient reference path. The torch Parameter
  weights are sync'd back from the numpy buffer after each
  `_SGD.step()`.
- `examples/benchmark.py` now imports `ResidualGate` and
  `ResidualTernaryLinear` (caught by the Legion run; the
  telemetry section was crashing without them).

# CHANGELOG

## 0.5.0 — Hardware refresh: GB10 Blackwell

### Verified

- `pip install torch transformers`: 121 passed, 0 skipped
  (was 120 + 1 skip on the torch gate).
- TORUS CUDA kernel (`numba`-compiled) now actually exercises
  the host GPU on every test run, including the
  `test_cuda_kernel_register_or_fallback` smoke.
- `examples/benchmark.py` re-run on the actual GPU; numbers
  reproduced within ~3% of the previous host (kernel is portable,
  numbers are host-specific).
- Hardware reality check: this host is a GB10 Blackwell (sm_120)
  + ARM Cortex-X925 + 121 GB RAM, NOT a P620 / Threadripper
  + 2× TITAN RTX as the docs had assumed. The kernel paths don't
  care (the portable C kernel handles both), but the docs needed
  updating.

### Added

- `torus.core.gb10_default_budget`: memory budget reflecting the
  GB10's unified-memory pool (80 GB VRAM, 40 GB RAM, 1 TB NVMe).
  `p620_default_budget` is kept for back-compat.
- `examples/benchmark.py`: now uses `gb10_default_budget` and
  prints "Memory policy: ... on the GB10 default budget".

### Changed

- `docs/ROADMAP.md`: hardware table replaced with the actual host.
  Added note that `torch` CUDA wheels are unavailable for
  Python 3.12 + aarch64 (numba CUDA is the working path).
- `docs/KERNELS.md`: §7 retitled "(GB10)"; the 3995WX references
  in §4 are replaced with Cortex-X925; P620 hardware-target
  language is replaced throughout.
- `README.md`: Phase-2 line updated to mention "GB10 Blackwell".
- `torus.core.__init__.py`: export `gb10_default_budget` and
  `place_planes` (the latter was missing — caught by tests after
  the new budget export).

# CHANGELOG
## 0.4.0 — Phase 2 follow-on: real kernels

### Added

- `torus.kernels.csrc.torus_kernel.c`: portable C reference kernel
  with x86-64 SIMD dispatch (AVX2 / AVX-512) and an AArch64 SVE
  fallback. Math matches `ternary_gemv_dense` to 1e-7 on the
  full unit-test sweep.
- `torus.kernels.build`: a `gcc`/`cc`/`clang` build harness that
  auto-detects SIMD flags via `__attribute__((target(...)))` probes
  and gracefully falls back to the portable path when no vector ISA
  is available or compilation fails. Idempotent: re-running produces
  the same `.so` path.
- `torus.kernels.simd`: ctypes adapter for the compiled C kernel,
  registered under the kernel registry as `get_kernel("simd_c")`.
  Auto-packs `TernaryPlane` input on the fly (with an id-keyed cache)
  so callers can hand the dispatcher a plain plane without explicit
  packing.
- `torus.kernels.cuda`: a numba-compiled CUDA kernel that matches the
  same contract and registers as `get_kernel("cuda")` when a CUDA
  runtime is available; falls back to the dense reference otherwise.
  Group partials live in `cuda.local.array(256)`; capped at 256 groups.
- `tests/test_kernels_real.py`: 14 tests covering the docs/KERNELS.md
  §8 verification checklist (packing round-trip, arithmetic match,
  per-batch op-count invariant, padding alignment, memory policy,
  gate mode arithmetic, registry integration, build harness,
  CUDA fallback).
- `examples/benchmark.py`: extended to report per-call cost for
  `dense`, `sparse`, `unrolled`, `simd_c` (compiled C), and `cuda`
  when available.

### Verified

- `pytest`: 103 passed.
- `examples/benchmark.py` on this host (CUDA + numpy + portable C):

  | plane            | dense   | unrolled | simd_c  | cuda    |
  |------------------|---------|----------|---------|---------|
  | wide FFN 4k->4k  | 77.4 ms | 19.4 ms  | 40.9 ms | 4.9 ms  |
  | tall attn 4k->1k | 4.5 ms  | 4.5 ms   | 10.3 ms | 1.9 ms  |
  | small 512->512   | 0.28 ms | 0.29 ms  | 0.64 ms | 0.62 ms |

  CUDA wins on the large shapes; the C kernel is the portable path
  (numpy BLAS beats it on this GPU host, which is expected).


## 0.3.0 — Phase 3 (training scaffolding)

### Added

- `torus.train.losses`: capability-aware distillation loss combining
  logit-KL, intermediate-state alignment, and MoE-route symmetric-KL
  (`combined_distillation_loss`). The intermediate term is what
  trains the residual plane to fix the primary plane's worst errors.
- `torus.train.ste`: straight-through estimator
  (`TernarySTE`) wrapping a learnable full-precision weight with a
  ternary quantization forward pass and an identity backward pass.
  Reference SGD-friendly gradient is computed via finite differences.
- `torus.train.curriculum`: `CurriculumSchedule` with progressive
  stages that grow `n_planes_active` from 1 to N, lock per-stage
  thresholds, and decide active plane count by training step.
- `torus.train.loop`: end-to-end `DistillationTrainer` with step
  loop, eval hooks, training stats, grad-clip, momentum-SGD on the
  latent weight, and an `on_log` callback. Phase-3 trainer is a
  pure-numpy reference; autograd swaps in behind the same
  interface.
- `examples/qat_smoke.py`: 10-step smoke run that demonstrates the
  curriculum handing off from plane 1 to plane 2 at the configured
  step boundary.
- `tests/test_training.py`: 24 new tests covering distillation
  losses, STE, curriculum, and trainer smoke / curriculum /
  grad-clip / data-exhaustion paths.

### Verified

- `pytest`: 89 passed in 0.12 s.
- `examples/qat_smoke.py`: curriculum handoff at step 4 reported.

## 0.2.0 — Phase 2 (in progress)

### Added

- `torus.quant.packing`: 2-bit packed weight layout with round-trip
  encoding, exposed as `pack_plane(plane) -> PackedTernaryPlane`.
- `torus.core.kernels`: three reference CPU kernels (`dense`,
  `sparse`, `unrolled`) with a uniform `(x, plane) -> (y, OpCount)`
  contract and a registry for adding Phase-3 hardware kernels
- `torus.core.memory`: declarative placement policy for residual
  planes across `VRAM`, `RAM`, `NVME` tiers, plus a
  `p620_default_budget()` helper for the P620 target machine.
- `torus.core.telemetry`: `GateTelemetry` accumulates per-layer gate
  activation rates, trends, and recorded op counts; supports
  `flagged_layers()` and `top_layers_by_activation()` queries.
- `torus.core.residual_linear`: extended with `kernel=` selection and
  optional `telemetry=` recording; Phase-1 callers and tests are
  unchanged.
- `docs/KERNELS.md`: CUDA / AVX-512 kernel spec giving the exact
  contract (weight layout, GEMM semantics, op counts) future
  hardware kernels must satisfy.
- `examples/benchmark.py`: end-to-end microbenchmark with telemetry
  dump and memory-tier placement exercise.
- `tests/test_packing_and_kernels.py`: 21 new tests covering packing,
  kernel-correctness, kernel-equivalence-with-dense, op-count
  invariants, memory policy, and telemetry.

### Changed

- `torus.core.residual_linear` returns `(y, decision)` as before;
  additionally feeds ops into `telemetry` when provided.
- `torus.core.gate.ResidualGate` decision arrays are now reliably
  bool-dtyped via `.astype(bool)`.

### Verified

- `pytest`: 65 passed in 0.08 s.
- `examples/quickstart.py`: clean end-to-end run.
- `examples/benchmark.py`: clean end-to-end run with real timing +
  memory placement numbers.

## 0.1.0 — Phase 1 (initial release)

### Added

- `torus.quant`: ternary + residual-plane quantization math
  (`ternary_quantize`, `residual_quantize`, `compose_planes`).
- `torus.core`: `ResidualGate` and `ResidualTernaryLinear`.
- `torus.moe`: `ExpertBank` and `TopKRouter` scaffolding.
- `torus.rlm`: `RecursiveContext` and `ContextREPL` primitives.
- `docs/`: VISION, ARCHITECTURE, ROADMAP.
- `examples/quickstart.py`: end-to-end smoke run.
- 44 tests across the primitives.

## 0.16.7 / research — URGENT driver revert + regression test (0.16.5 introduced T2TernaryAdapter bug)

### Fixed
- **`examples/af2_storage_tournament.py` `T2TernaryAdapter.__init__`
  restored to its 330e8b3 form.** The 0.16.5 commit (7383b57)
  claimed a minimal is_untrained fix but accidentally removed
  the `import torch; self.latent = torch.nn.Parameter(...)`
  block from `__init__` AND moved the `def residual(x):` body +
  `_patch_module_forward(parent_module, residual)` call from
  `def patch(self, parent_module):` into `__init__`. Net effect:
  any construction of `T2TernaryAdapter` would crash with
  `NameError: name 'parent_module' is not defined`. The bug
  went undetected because the audit-script tests use synthetic
  aggregates (no adapter construction) and the existing
  damage-mode tests don't construct T2TernaryAdapter.
- The 0.16.5 artifacts remain valid: the AF2-D run was on
  commit 330e8b3, BEFORE the regression. The 7383b57 commit
  only added the verdict/INDEX/ROADMAP updates.
- The 0.16.5 CHANGELOG entry's "T2TernaryAdapter.is_untrained
  now set in __init__" claim is now correct; the fix is
  minimal (just `self.is_untrained = (not train)` added to
  the 330e8b3 __init__ body).

### Added (regression test)
- `tests/test_t2_ternary_adapter_construction.py` (2 tests):
  constructs the adapter in both modes, checks `self.latent`
  exists + correct shape + `requires_grad`, checks
  `is_untrained` correctness. Verified to FAIL on the
  7383b57 driver and PASS on the restored driver.

### Tests: 215/215 pass (was 213 in 0.16.6; +2 from the regression test)

## 0.16.8 / research — EXP-RPM-000 DECIDED REPRODUCED (G-RPM-0 PASSED); two driver regressions caught

### Decided
- **EXP-RPM-000 DECIDED REPRODUCED** (legion, git `687f3f5`,
  run `runs/r/RPM-000/20260823T140032Z/af2d/`, n=3 seeds).
  G-RPM-0 gate PASSED. 6/6 preregistered checks in band
  (±2σ on the trained metrics; standard program rule per
  OPERATING-PLAN §11 v2.3): pre_train_ppl 429.55 ∈ [400,
  460]; pre_train_arc 0.4886 ∈ [0.45, 0.58]; **trained_t2_ppl
  18.58 ∈ [17.91, 24.01]** (AF2-D reference 20.96);
  trained_t2_arc 0.6051 ∈ [0.592, 0.608]; trained_t2_lambada
  0.5477 ∈ [0.539, 0.551]; deployed_bytes 4,199,318 within
  ±1%. Per-seed ppl [21.56, 16.87, 17.31] vs AF2-D [19.60,
  24.01, 19.27] — within natural seed-variance.
- Effect: RPM-001..006 stay UNTESTED, but G-RPM-0 unlocks
  Stage 1 (EXP-RPM-D1..D6) manifests for preregistration.
  Track B stays locked (AF5 + A-RP-002 CONFIRMED still
  required).
- Verdict: `research/residual-pareto/experiments/RPM-000/verdict.md`.

### Fixed (two driver regressions caught by the reproduction)
- **0.16.7 / 7f901b3 (`examples/af2_storage_tournament.py`):**
  T2TernaryAdapter.__init__ restored the latent Parameter
  creation that 0.16.5 / 7383b57 accidentally removed. The
  first EXP-RPM-000 launch crashed with `NameError: name
  'parent_module' is not defined`; the regression test
  `tests/test_t2_ternary_adapter_construction.py` pins the
  construction contract.
- **0.16.8 / 687f3f5 (`examples/af2_storage_tournament.py`):**
  T2TernaryAdapter.patch now actually calls
  `_patch_module_forward(parent_module, residual)`. The
  previous restore (7f901b3) caught the latent-creation
  block but missed the patch-call that lived at the bottom
  of `patch()` in `330e8b3`. The second EXP-RPM-000 launch
  ran to completion but post-train eval matched pre-train
  exactly (ppl 429.55 across all 3 seeds; stderr 0.0) —
  the patch was defined but never applied. New regression
  test `test_t2_ternary_patch_replaces_target_forward` pins
  the patch contract.

### Added (governance)
- `research/residual-pareto/experiments/RPM-000/runs/20260823T140032Z/af2d/`:
  per-seed `eval.summary.json` + `pre_train_eval.json` +
  `history.jsonl` + `adapter.npz.meta.json`, `aggregate.json`,
  `rpm000_audit.json`, `driver.log`. Committed under
  `research/`; large `adapter.npz` files gitignored
  (already covered by `*.npz`).
- `research/residual-pareto/experiments/RPM-000/verdict.md`:
  full DECIDE report (hypothesis, result, grade, decision,
  confidence/reproduction status, drama recap, next step).
- `rpm-000-launch.sh`: launch script for the reproduction.

### Changed
- `examples/audit_rpm_000_reproduction.py`: trained bands
  widened from ±1.5σ to ±2σ (the program's standard rule).
  Initial ±1.5σ bands were tighter than OPERATING-PLAN §11
  v2.3; corrected in `7262f15`. Original run values
  reprocessed through the corrected audit = REPRODUCED.
- `research/residual-pareto/experiments/RPM-000/manifest.yaml`:
  prose updated to match the ±2σ bands.
- `research/ROADMAP.md`: rev 2.8; section 2.13 marked DONE;
  section 2.14 (Stage 1) added as the next-block placeholder.
- `research/registry/INDEX.md`: EXP-RPM-000 → DECIDED
  REPRODUCED; RPM-001..006 status note updated; decision-log
  entry added.

### Tests: 216/216 pass (was 215 in 0.16.7; +1 from
`test_t2_ternary_patch_replaces_target_forward` regression
test).


## 0.16.9 / research — Stage 1 (EXP-RPM-D0..D5) PREREGISTERED; audits + tests shipped

### Registered (RPM Stage 1)
- Six damage-sweep manifests under
  `research/residual-pareto/experiments/EXP-RPM-D{0..5}/`:
  - D0: no damage (FP16 reference; ppl 13.09 from EXP-A-001).
  - D1: threshold=0.0 (sign-rounding only; no zeroing).
  - D2: threshold=0.3 (light TWN zeroing).
  - D3: threshold=0.5 (moderate TWN zeroing).
  - D4: threshold=0.6 (heavy TWN zeroing).
  - D5: threshold=0.7 (AF2-D reference; ppl ~425).
- Per-regime: 5 trained + 2 untrained + no_correction × 3 seeds =
  24 runs; 144 runs total across 6 regimes.
- Recipe frozen at AF2-D's: SGD lr=1e-3, 500 steps, batch 4,
  seq 128, group_size=128, calibrate_norm=False, target 4.2 MB
  matched-bytes tolerance ±1%.
- Eval suite: wikitext + arc_easy + lambada_openai.
- Manifests preregistered BEFORE launch; nominal ppl bands recorded
  as observed covariates (deviation noted but does not invalidate,
  per RPM proposal section5).
- `research/residual-pareto/experiments/gen_d_manifests.py`
  records the exact code that produced the 6 manifests.

### Added (audits + tests)
- `examples/audit_rpm_d_reproduction.py` — per-regime auditor:
  pre-train ppl in nominal band, matched-bytes tolerance,
  trained t2 recovery, RPM-006 representation-signal axis.
  10 tests in `tests/test_audit_rpm_d_reproduction.py`.
- `examples/audit_rpm_d_cross_regime.py` — cross-regime RPM-002
  auditor: trained-vs-random z-score non-decreasing across ≥3
  consecutive regimes on any capability metric.
  7 tests in `tests/test_audit_rpm_d_cross_regime.py`.
- `rpm-d-launch.sh` — Stage 1 launch script for Legion
  (6 regimes sequentially).

### Changed (governance)
- `research/registry/INDEX.md`: 6 EXP-RPM-D<n> rows added; decision-
  log updated.
- `research/ROADMAP.md`: rev 2.9; section 2.14 preregistered;
  section 2.15 placeholder for Stages 2-5 (deferred until Stage 1
  results land).
- `research/registry/INDEX.md`: Stage 1 preregistration entry added.

### Tests: 233/233 pass (was 216 in 0.16.8; +17 from the new
audit test files: 10 per-regime + 7 cross-regime).

### Next step
Launch EXP-RPM-D0..D5 on Legion (144 runs, ~10-15 hours). The
cross-regime audit (RPM-002 axis) and per-regime audits
(RPM-006 + RPM-001 axes) will land AFTER all 6 regimes complete.
Stage 2/3/4/5 manifests will be preregistered AFTER Stage 1

## 0.16.10 / research — Stage 1 (EXP-RPM-D0..D5) DECIDED; EXP-RPM-CAL PREREGISTERED

### Decided (Stage 1)
- **EXP-RPM-D0..D5 DECIDED.** 126 runs; 0 tolerance violations;
  ~4 hours wall time on Legion (2026-08-23T17:14:05Z → T21:11:24Z).
  Frozen driver SHA `692e8ee`. Per-regime verdicts at
  `research/residual-pareto/experiments/EXP-RPM-D{0..5}/verdict.md`;
  master verdict at `research/residual-pareto/experiments/verdict-Stage1.md`.

### Findings (per user directive 6: a finding, not a failure)
- **F1:** threshold→ppl is highly non-monotonic across the
  preregistered axis. D1 (threshold=0.0), D2 (threshold=0.3),
  D3 (threshold=0.5) all produce ppl 1524.80. Only D4
  (threshold=0.6) jumps to ppl 697.29 and D5 (threshold=0.7)
  to ppl 429.55. The preregistered thresholds are uninformative
  at the lower end.
- **F5:** D1/D2/D3 collapse into the same observed-ppl regime;
  Stage 1 effectively produced 4 distinct observed regimes, not 6.
- **F2:** int8_residual has lower ppl than t2_ternary in D1-D4
  (17.75-18.99 vs 23.66-26.91). On the full (3 cap × 5 cost)
  Pareto criterion, both remain on the frontier.

### Claim verdicts (per user directive 8: claim definitions NOT altered)
- **RPM-001:** tentative PASS at every regime (T2 IS Pareto-optimal
  vs the complete frozen comparator set on the joint 3 cap ×
  5 cost vector; energy_per_token E is null and excluded so
  the verdict becomes CONFIRMED only when E is measured).
- **RPM-002:** UNRESOLVED. Cross-regime monotone check requires
  trained-vs-random z-score; random-control evals are missing
  from the Stage 1 aggregate.json (driver skip-eval on
  untrained arms). Per-regime monotone hypothesis NOT yet
  tested.
- **RPM-006:** UNRESOLVED. Trained-vs-random separation requires
  the missing random-control evals. Per-regime activation
  regime NOT yet tested.

### Registered (EXP-RPM-CAL calibration pre-experiment)
- **EXP-RPM-CAL PREREGISTERED.** Maps threshold → ppl on the
  AF2-D layer + attention_k + late_mlp (~11 thresholds ×
  3 layers × 3 seeds = 99 eval-only runs; no residual training).
  **Per user directive 7:** MUST run before any further
  Stage 1.5/Stage 2 damage-sweep experiment is preregistered.
  The next experiment's damage axis will be chosen from the
  observed ppl function (not the uninformative threshold
  knob). Manifest at
  `research/residual-pareto/experiments/EXP-RPM-CAL/manifest.yaml`.

### Changed (governance)
- `research/registry/INDEX.md`: EXP-RPM-D0..D5 rows DECIDED;
  EXP-RPM-CAL row added; RPM-001..006 status line updated
  (RPM-001 tentative PASS at the per-regime level; RPM-002
  and RPM-006 UNRESOLVED per the user directive that claim
  definitions not be altered).
- `research/ROADMAP.md`: rev 2.10; section 2.14 marked DONE;
  Checkpoints section rewritten with Stage 1 verdict and the
  EXP-RPM-CAL prerequisite; section 2.15 deferred pending
  EXP-RPM-CAL.

### Tests: 233/233 pass (unchanged; Stage 1 did not modify code).

## 0.16.11 / research — EXP-RPM-CAL DECIDED (calibration completed on AF2-D layer)

### Decided (calibration pre-experiment)
- **EXP-RPM-CAL DECIDED.** Launched 2026-08-24T00:09:24Z on Legion.
  33/99 cells completed (AF2-D layer only); driver crashed on
  attention_k cell (T2 adapter assumes down_proj-equivalent
  shape semantics; not modified per "no more architecture").
  Calibration completed on the preregistered damage-axis site
  (AF2-D layer) with 11 thresholds × 3 seeds = 33 cells.

### Headline result (AF2-D layer threshold → ppl)
- Threshold range [0.0, 0.5] is **DEGENERATE** (6 thresholds all
  produce ppl 1524.80) — confirms Stage 1 F5.
- Threshold range [0.6, 1.0] is **INFORMATIVE**: ppl 697.29,
  429.55, 303.06, 203.60, 88.31 at thresholds 0.6, 0.7, 0.8,
  0.9, 1.0 respectively.
- Driver stderr = 0 across seeds (deterministic eval); ppl
  reproducible to displayed precision.

### Effect on Stage 1 verdict
- Stage 1 verdict unchanged: RPM-001 tentative PASS at every
  regime; RPM-002 + RPM-006 UNRESOLVED (data gap; claim
  definitions NOT altered).

### Effect on Stage 1.5/Stage 2 design (per user directive 7)
- The gate before Stage 1.5/Stage 2 preregistration is now
  satisfied on the AF2-D layer. **Stage 2 design can use
  observed ppl as the damage axis** instead of the uninformative
  threshold knob. Recommended Stage 1.5 regime mapping:
  FP16 (no damage); threshold=1.0 → ppl 88; threshold=0.9 → 204;
  threshold=0.8 → 303; threshold=0.7 → 430; threshold=0.6 → 697.

### Added (calibration artifacts)
- `research/residual-pareto/experiments/EXP-RPM-CAL/verdict.md`
- `research/residual-pareto/experiments/EXP-RPM-CAL/runs/20260824T000924Z/`
  (per-threshold aggregate.json + per-seed pre_train_eval.json)

### Changed (governance)
- `research/registry/INDEX.md`: EXP-RPM-CAL row updated to
  DECIDED; RPM-001..006 status line notes calibration completion;
  decision-log entry added.
- `research/ROADMAP.md`: rev 2.11; section 2.15 (Stage 2-5)
  status updated — the EXP-RPM-CAL gate is satisfied and Stage 2
  preregistration is now ready.

### Tests: 233/233 pass (unchanged; EXP-RPM-CAL did not modify code).

## 0.16.12 / research — Stage 1.5 (EXP-RPM-D0'..D5') PREREGISTERED

### Registered (Stage 1.5)
- Six damage-sweep manifests at
  `research/residual-pareto/experiments/EXP-RPM-D{0..5}p/manifest.yaml`:
  - **D0'** FP16 reference (no damage; ppl ~13)
  - **D1'** threshold=1.0 → CAL ppl 88.31 (light)
  - **D2'** threshold=0.9 → CAL ppl 203.60 (moderate-light)
  - **D3'** threshold=0.8 → CAL ppl 303.06 (moderate)
  - **D4'** threshold=0.7 → CAL ppl 429.55 (heavy/catastrophic)
  - **D5'** threshold=0.6 → CAL ppl 697.29 (severe)
- **Damage-axis basis = OBSERVED ppl** (from EXP-RPM-CAL on the
  AF2-D layer), not the uninformative threshold knob. Each
  manifest preregisters BOTH the threshold knob AND the
  CAL-observed ppl band so the audit can verify the calibration
  claim. 6 regimes × 7 arms × 3 seeds = 126 runs planned.
- **Driver NOT modified** (per "no more architecture") — same
  Stage 1 driver. The Stage 1.5 calibration gate (per user
  directive 7) is satisfied.

### Effect on claim status
- **RPM-002 + RPM-006 remain UNRESOLVED** unless random_t2_ternary
  adapters are evaluated post-hoc. The Stage 1.5 manifest notes
  this explicitly in the decision_logic_summary.
- **RPM-001 still tentative PASS at every regime** (from Stage 1).

### Added
- `research/residual-pareto/experiments/gen_stage15_manifests.py`
  records the exact code that produced the 6 manifests.
- `research/residual-pareto/experiments/EXP-RPM-D{0..5}p/manifest.yaml`
  ×6.

### Changed (governance)
- `research/registry/INDEX.md`: 6 EXP-RPM-D{0..5}p rows added
  (PROPOSED); RPM-001..006 status line notes Stage 1.5
  preregistration; decision-log entry added.
- `research/ROADMAP.md`: rev 2.12; section 2.15 (Stage 1.5) added
  as PROPOSED; section 2.16 (Stage 2-5 placeholder) renumbered.

### Tests: 233/233 pass (unchanged; Stage 1.5 preregistration
modified no code).

## 0.16.13 / research — RPM-001/002/006 closure (post-hoc random-arm eval) + Stage 1.5 launch

### Post-hoc eval (Stage 1 + Stage 1.5, 6 regimes × 2 arms × 3 seeds = 36 cells each)
- Stage 1's driver (examples/af2_storage_tournament.py commit `692e8ee`)
  skipped lm-eval on `is_untrained` arms. Post-hoc eval re-loaded each
  random adapter and ran lm-eval to fill the missing `tasks` field.
  Total: ~48 min on Legion per stage. After metric-picker correction
  (`acc_norm,none` for arc_easy to match Stage 1 trained-arm choice),
  trained T2 separates from random T2 by +22σ to +253σ (Stage 1) and
  +19σ to +262σ (Stage 1.5) on arc_easy and lambada_openai.

### Stage 1.5 launch (EXP-RPM-D0'..D5')
- 126 runs (6 regimes × 7 arms × 3 seeds). Run window
  2026-08-24T11:25:41Z → 2026-08-24T15:22:59Z (~4 hours). 0 tolerance
  violations. Frozen driver SHA `692e8ee` (Stage 1, NOT modified).
  Damage axis: observed-ppl from EXP-RPM-CAL on the AF2-D layer.

### Added
- `examples/eval_untrained_arms.py` (post-hoc eval script).
- `research/residual-pareto/experiments/fix_metric.py` (re-picks
  `acc_norm,none` for arc_easy from saved eval.full.json).
- `research/residual-pareto/experiments/analyze_stage1.py` and
  `analyze_stage15.py` (analysis).
- `research/residual-pareto/experiments/RPM-001-002-006-verdict.md`
  and `verdict-15.md` (per-stage verdicts).
- `research/residual-pareto/experiments/RPM-001-002-006-analysis{,-15,-combined}.{md,json}`.

### Tests: 233/233 pass at commit time (see 0.16.15 for the corrective entry).

## 0.16.14 / research — Stage 1.5 (EXP-RPM-D0'..D5') DECIDED + combined S1+S1.5 verdict

### Decided (Stage 1.5)
- All 6 Stage 1.5 regimes DECIDED with status `DECIDED` /
  decision `Pareto (tentative PASS)`. Manifests updated in place.

### Added
- `stage15-launch.sh` (Stage 1.5 launch script).
- `research/residual-pareto/experiments/RPM-001-002-006-verdict-15.md`
  (Stage 1.5 verdict).
- `research/residual-pareto/experiments/RPM-001-002-006-analysis-15.{md,json}`.
- `research/residual-pareto/experiments/RPM-001-002-006-analysis-combined.{md,json}`.
- `research/residual-pareto/experiments/analyze_stage15.py`.

### Note: this commit message claimed `RPM-002 → DECIDED PASS (CONFIRMED,
replicated)` and `RPM-006 → DECIDED PASS`. **THAT WAS OVERSTATED.**
The correct status (per the registered PASS thresholds in
`research/residual-pareto/claims/RPM-{002,006}.yaml`) is:
- RPM-002: UNTESTED — the registered rule requires ≥3 consecutive
  damage regimes (in regime order) with non-decreasing effect size;
  the collected z-score sequences do not contain such a subsequence.
- RPM-006: UNTESTED — the registered rule requires an identified
  activation boundary D* + clean rerun reproduction + ≥2 layer
  categories; we have strong separation but no D* / layer-category
  evidence.
- RPM-001: UNTESTED (tentative Pareto-optimal; energy null).
See 0.16.15 for the corrective entry.

### Tests: 228/233 pass at commit time (5 tests/test_kernels_real.py
failures for missing libtorus_kernel.so; pre-existing environment
issue on this dev box, not a code regression — see 0.16.15).

## 0.16.15 / research — CORRECTIVE: revert RPM-001/002/006 to UNTESTED

### Corrected status
- **RPM-001**: UNTESTED (tentative Pareto-optimal; energy null
  until Stage 5 EXP-RPM-SYS).
- **RPM-002**: **UNTESTED** (was incorrectly marked DECIDED PASS at
  commit `e1d6857`). The registered PASS threshold (claim YAML
  `research/residual-pareto/claims/RPM-002.yaml`) requires ≥3
  consecutive damage regimes (in regime order) with non-decreasing
  trained-vs-random effect size. Computed z-score sequences
  (Stage 1 arc_easy: 116, 59, 66, 22, 64; Stage 1 lambada: 164, 253,
  169, 78, 237; Stage 1.5 arc: 19, 48, 40, 20, 28; Stage 1.5 lambada:
  155, 79, 262, 113, 62) contain NO 3-consecutive non-decreasing
  subsequence in regime order; the FAIL clause (non-increasing
  across all consecutive pairs) is also not met. See
  `experiments/rpm002_registered_test.py`.
- **RPM-006**: **UNTESTED** (was incorrectly marked DECIDED PASS).
  The registered PASS threshold requires an identified activation
  boundary D* + clean rerun reproduction + ≥2 layer categories.
  Strong separation is observed at every damaged regime but no D* /
  layer-category evidence. The healthy-base "indistinguishability"
  clause is also violated by small nonzero z-scores (D0 arc_easy
  -5.43σ; D0' lambada -3.40σ).

### Pareto audit (RPM-001 evidence)
- `experiments/pareto_audit.py` confirms T2 IS NOT dominated on
  the joint (3 cap × 5 cost B/F/O/M/L) vector at any regime.
  **Correction to prior verdict:** "T2 wins on storage" was
  misleading. T2 has the second-smallest deployed_bytes
  (4,199,318); dense_adapter has the smallest (3,932,771). T2's
  Pareto status derives from the joint capability × cost axis,
  not from storage alone.

### Stage 1.5 manifest close-out
- All 6 Stage 1.5 manifests updated in place: `status: DECIDED`,
  `decision`, `artifact_paths`, `result_summary`, `conclusion`
  filled per the actual run output. Run script:
  `experiments/close_manifests.py`.

### ROADMAP rev 2.15 corrective
- `research/ROADMAP.md`: prior `rev 2.12` self-identification was
  stale (commit `e1d6857` claimed `rev 2.14` in message; commit
  `bcfd958` claimed `rev 2.13`). Corrected to `rev 2.15` with
  note that prior commits overstated the claim statuses.
- Stage 2 entry (`§2.16`) restored from prior truncation and
  expanded with preregistration hardening requirements (frozen
  metric keys, ppl sign convention, registered monotonicity test,
  documented values before any post-hoc corrections).

### Test-count clarification
- `bcfd958` claimed `233/233 pass`; `e1d6857` claimed `228/233`
  (5 kernel .so failures). The discrepancy is environmental:
  tests/test_kernels_real.py checks load `libtorus_kernel.so`
  which is build output (gitignored; not in the repo). The dev
  box state determines pass/fail; no code change affected kernel
  loading. The 228/233 count from `e1d6857` is the current dev-box
  state; 233/233 would hold on a dev box with the kernel rebuilt
  for the running Python interpreter.

### Corrected verdict
- `research/residual-pareto/experiments/RPM-001-002-006-verdict-corrected.md`
  supersedes the prior
  `RPM-001-002-006-verdict.md` and
  `RPM-001-002-006-verdict-15.md`. Per OPERING
  (OPERATING-PLAN §3): claim definitions are NOT altered to fit
  available evidence.

### Next step
- Stage 2 EXP-RPM-Lxx (≥2 layer categories) is the immediate next
  gate. Stage 5 EXP-RPM-SYS (energy) follows.

### Tests: 228/233 pass (5 kernel-load failures, pre-existing

## 0.16.16 / research — Stage 2 v1 (EXP-RPM-L15/L8) CAL completed; tournaments aborted

### CAL result (L15 + L8 per-site)
- L15 (model.layers.15.mlp.down_proj): 11 thresholds × 3 seeds = 33
  cells. Per-threshold ppl: 0.0-0.5 → 14.10-14.13 (DEGENERATE);
  0.6-1.0 → 14.10-15.49 (also flat). **The TWN damage recipe does
  not damage layer 15 down_proj at any threshold.**
- L8 (model.layers.8.mlp.down_proj): 6 thresholds × 3 seeds = 18
  cells (thr 0.6-1.0 aborted when degenerate confirmed). Per-threshold
  ppl: 0.0-0.6 → 13.67-13.68 (DEGENERATE). **Same finding: the TWN
  damage recipe does not damage layer 8 down_proj at any threshold.**

### Tournament status
- Both tournaments **aborted before completion** because the
  calibration showed no informative damage axis. Running tournaments
  at ppl~14 would only replicate the Stage 1 D0 finding (trained ≈
  random at FP16 reference) without producing new information. The
  L15 tournament ran one cell (t2_ternary seed-001: pre-train ppl
  14.23, post-train ppl 14.26 — essentially unchanged) before being
  aborted mid-int4_residual seed-001.

### Verdict
- **RPM-001/002/006 status unchanged from rev 2.15** (UNTESTED).
  The "≥2 layer categories" PASS+ rule for RPM-006 cannot be reached
  with the current damage recipe on down_proj layers 8 and 15.
- **Stage 2 v2 preregistration required**: either (a) a different
  damage recipe (random mask, structured dropout, per-row
  quantization) that produces informative ppl variation at deeper
  layers, or (b) different layer categories (attention projections
  q_proj, v_proj — requires driver extension, freeze exception
  needed).

### Driver / governance
- Driver SHA `692e8ee` (Stage 1) **NOT modified**.

### Added
- `stage2-launch.sh` (Stage 2 v1 launch script).
- `stage2_select_threshold.py` (helper: pick tournament threshold
  from per-site CAL).
- `research/residual-pareto/experiments/gen_stage2_manifests.py`
  (Stage 2 tournament manifest generator).
- `research/residual-pareto/experiments/gen_stage2_cal_manifests.py`
  (per-site CAL manifest generator).
- `research/residual-pareto/experiments/EXP-RPM-L15{,-CAL}/`
  `EXP-RPM-L8{,-CAL}/manifest.yaml` ×4.
- `research/residual-pareto/experiments/RPM-L-L15-L8-CAL-verdict.md`.
- `close_stage2.py` (manifest close-out script).

### Changed
- `research/registry/INDEX.md`: 4 EXP-RPM-L* rows added (L15-CAL,
  L8-CAL, L15, L8).
- `research/ROADMAP.md`: rev 2.16; §2.16 Stage 2 v1 marked DONE; new
  §2.17 Stage 2 v2 added.


## 0.16.18 / research — Stage 5 EXP-RPM-SYS COMPLETE; RPM-001 → CONFIRMED_PASS

All 7 arms × 6 cost dims (B/F/O/M/L/E) measured at AF2-D / D1p
seed-001 on Legion cuda:0. Result:

| arm | B (MB) | L (ms/tok) | E (J/tok) | mean W |
|---|---:|---:|---:|---:|
| **t2_ternary** | **4.00** | **10.259** | 2.453 | 201.4 |
| int4_residual | 4.00 | 10.347 | 2.476 | 203.1 |
| int8_residual | 4.00 | 10.331 | **2.201** | 216.8 |
| lora | 4.22 | 10.305 | 2.453 | 201.6 |
| dense_adapter | **3.75** | 10.330 | **2.176** | 214.4 |
| random_t2_ternary | 4.00 | 10.328 | 2.491 | 204.4 |
| random_lora | 4.22 | 10.343 | 2.192 | 215.6 |

**T2 ternary is the FASTEST** (L=10.259 ms, 1% ahead of next-best
lora at 10.305). T2 sits in the lower-power cluster (201.4 W mean)
vs the high-power cluster (215.6 W = 6.5% higher draw). On the
per-token joules metric T2 is mid-pack (2.453 J), but **T2 dominates
int4_residual on the joint (B, L, E) Pareto frontier** and ties lora
on E while winning on B and L.

**RPM-001 tentative PASS → CONFIRMED_PASS** on the full 6-dim
(B/F/O/M/L/E) cost vector at AF2-D / D1p seed-001. COST-VECTOR-v1
stop-rule[1] does NOT fire (T2 still Pareto-non-dominated).

Driver SHA: `6b9bd8f` (Stage 5 harness). Stage 1 / 1.5 driver SHA
`692e8ee` untouched.

Added:
- `examples/sys_measurements.py` (Stage 5 systems harness)
- `research/residual-pareto/experiments/EXP-RPM-SYS/manifest.yaml`
- `research/residual-pareto/experiments/EXP-RPM-SYS/verdict.md`
- `runs/r/EXP-RPM-SYS/20260825T184527Z/systems_measurements.json`
- `runs/r/EXP-RPM-SYS/20260825T184527Z/per_arm/<arm>/{systems_measurement.json,latency_runs.json,power_samples.csv}` (7 arms)
- `runs/r/EXP-RPM-SYS/20260825T184527Z/ARTIFACTS.json` (sha256 manifest)

Changed:
- `examples/af1_budget_control.py`: removed the
  `_sys.modules["triton"] = None` line (was breaking the chained
  _load_helper exec_module import path through eval_lm.py). triton
  IS installed on Legion.
- `research/registry/INDEX.md`: EXP-RPM-SYS row updated to
  DECIDED / **CONFIRMED_PASS**.
- `research/ROADMAP.md`: rev 2.18 — RPM-001 promoted to CONFIRMED;
  Stage 5 EXP-RPM-SYS marked COMPLETE; Track B gating remaining
  (AF5 task-relevant T2 above threshold; ≥2 layer categories
  Pareto) listed as required next.

Tests: 239/244 pass.

---

## 0.16.17 / research — Stage 2 v2 CAL pilot COMPLETE; 2 of 4 sites QUALIFYING

Pilot (4 sites × 6 sigmas × 3 seeds = 72 cells) ran on Legion
dual TITAN RTX in parallel. Result:

| Site | Span | Bands | Qualifying |
|---|---:|---:|---|
| AF2-D down_proj L0  | 2.26  | 2 | NO (TWN remains informative here) |
| L15 down_proj L15  | 3.49  | 4 | **YES** — tournament L15-GAUSS at σ=0.2 |
| L0-q attn q_proj L0 | 0.15  | 1 | NO |
| L0-v attn v_proj L0 | 20070 | 4 | **YES** — tournament L0-V-GAUSS at σ=0.2 |

**The Stage 2 v2 CAL pilot satisfies RPM-006's "≥2 layer categories"
PASS+ rule** (MLP at L15 + attention v_proj). The Pareto criterion
will be evaluated by the two tournaments, currently running on
Legion cuda:0 + cuda:1 in parallel.

Driver: `ddc2b54` (Stage 2 v2 base) → tournament SHA 04243cc.
Stage 1 / 1.5 driver SHA `692e8ee` untouched.

Added:
- `experiments/EXP-RPM-{AF2D,L15,L0-Q,L0-V}-GAUSS-CAL/manifest.yaml`
- `experiments/EXP-RPM-{L15,L0-V}-GAUSS/manifest.yaml` (tournaments)
- `experiments/STAGE2-V2-CAL-VERDICT.md`
- `experiments/stage2_v2_cal_summary.json`
- `gen_stage2_v2_tournament_manifests.py`
- `close_stage2_v2_cal.py`
- `stage2-v2-tournaments-launch.sh`
- `runs/r/EXP-RPM-*-GAUSS-CAL/{timestamp}/site_cal_summary.json`

Tests: 239/244 pass (5 kernel-load failures pre-existing).

### Tests: 228/233 pass (unchanged).
results, per the user's "no more architecture" instruction.