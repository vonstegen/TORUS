# Residual-Plane Program Closure — 2026-08-30

**Authority:** user direction 2026-08-30 at commit `9c9ac96`.
**Scope:** the T2 residual-ternary correction plane as an active
TORUS architecture (Track A residual-plane branch; Track B gating
question).
**Verdict:** **MECHANISM CONFIRMED / COMPETITIVE ARCHITECTURE NOT
SUPPORTED.**

---

## 1. The verdict, precisely

Trained T2 correction is a **real learned phenomenon** — it beats
the random structural prior, beats no correction, and its learning
signal survives seed/init/context/corpus perturbations and extends
to held-out tasks. It is **not** a competitive precision-recovery
strategy under the tested TORUS conditions: at matched training
budget, whole-model continuation of the damaged base beats it; at
matched storage, INT8 beats it; and its strong-form site does not
generalize to a second layer category.

The mechanism must not be discarded as nonexistent. The
architecture must not be pursued further.

## 2. Evidence: the mechanism side (positive science)

| evidence | result |
|---|---|
| EXP-AF-002-D (AF2-D, TWN thr 0.7) | trained T2 ≫ random T2: ppl −226.9σ, arc +25.1σ, lambada +116.8σ; 20.3× ppl recovery |
| EXP-RPM-AF2D-SEVERITY (Stage 2 v6) + CONFIRM-V7 | LRN band = full preregistered TWN severity range {0.6–1.0}, two independent seed sets; z +10 to +1687σ |
| EXP-AF-003 | init-robust: 5/5 non-zero σ levels × 3 seeds (spread ratio 1.06) |
| EXP-AF-006b | window regimes {16,128,256} + corpus switch all recover; third independent reference reproduction |
| EXP-RPM-T02 (AF5 at D5p) | **held-out-task LRN**: hellaswag +21.76σ vs random T2 (base 0.426 → trained 0.585; random recovers +0.008); boolq +1.48σ, openbookqa +1.23σ |
| EXP-RPM-T02-PROBE | T01's held-out null explained: regime miscalibration (Gaussian σ=0.2 ≈ FP16 eval base), not task insensitivity |

These support A-RP-LRN at the AF2-D TWN band and now on held-out
tasks at catastrophic severity.

## 3. Evidence: the engineering side (negative architecture)

| evidence | result |
|---|---|
| A-RP-001 (EXP-AF-001/001-R) | CONFIRMED_FAIL — equal-training-time FP16 continuation beats T1+T2 on every capability metric |
| A-RP-003 (EXP-AF-004/004-R) | CONFIRMED_FAIL — sequential-curriculum assumption falsified; joint training is the evidenced recipe |
| EXP-RPM-T01 | held-out null at its (near-FP16) regime — corrected interpretation: no recovery possible from an undamaged base |
| EXP-RPM-SITE-DISCOVERY | **NO_SECOND_SITE** — catastrophic damage pinned to layer 0 (L0-down TWN, L0-v Gaussian); 9 informative-but-mild sites, 0 candidates; A-RP-002 annotated site-local |
| EXP-RPM-T02 (AF5 at D5p) | **r3: 0/4** — INT8_residual beats T2 on every held-out task at matched storage (hellaswag 0.598 vs 0.585; winogrande 0.591 vs 0.562; boolq 0.597 vs 0.584; openbookqa via lora) |
| EXP-AF-001-D | **FAIL** — equal-budget T1-only continuation (whole-model FP16 from the damaged start) beats T2: arc −10.9σ, lambada −6.9σ (ppl +0.75σ, T2, within noise) |
| EXP-A-H1 (Hadamard branch) | DECIDED FAIL — closed separately with H-POST (CP3.2) |
| Stage 2 v2–v4 tournaments | L15 Gaussian: trained ≈ random at σ=0.2/0.5, trained < random at σ=1.0 — no cross-site LRN |
| Stage 3 v1/v2 | mechanism-specific: T2 anti-recovers Gaussian damage; catastrophic damage bounded to {TWN, Gaussian} at layer 0 |

Per suite doc §15 ("downgrade the conclusion appropriately rather
than attempting to rescue the architecture with additional
complexity"): the acceptance bar fails on item 1 (T1-only
continuation) and item 4 (equal-storage non-ternary baseline).

## 4. Localization

The recovery phenomenon's damage precondition is **site/regime
dependent**: catastrophic (correctable) damage exists only at
layer-0 sites under {TWN, Gaussian}; deeper layers are robust to
every tested mechanism. AF2-D did not generalize into a second
layer category (RPM-006's category criterion remains unmet).

## 5. What this closes

- **Track B (oracle gating / adaptive precision): FROZEN.** Both
  remaining unlock conditions carry definitive negative evidence —
  condition 3 (AF5): held-out T2 value exists but is below the
  frozen threshold (loses to INT8 4/4); condition 4 (≥2 layer
  categories): the discovery returned NO_SECOND_SITE with no grid
  expansion. There is no credible path to unlock under the current
  evidence, and no further rescue experiments are authorized.
- **Residual-plane development as an active architecture: CLOSED.**
  The recipe will not be modified further; the branch has answered
  its question.
- **NOT closed:** the scientific claims stand as recorded — A-RP-002
  (CONFIRMED_PASS, site-local), A-RP-LRN
  (CONFIRMED_AT_AF2D_TWN_BAND, held-out axis corrected), the
  INVALID-vs-FAIL governance record (5 A-H1 defects, T02-PROBE gate
  miscalibration — each caught by an instrument built for that
  class).

## 6. Architectural constraint for whatever comes next

A new ternary mechanism must demonstrate an advantage against the
**best matched practical baseline** — equal-storage correction
(e.g. INT8) AND equal-budget base continuation — early in its
experimental lifecycle, not after an internal-recovery program.
The new gating order is recorded in the suite doc §15 addendum
(2026-08-30):

**mechanism signal → capability check → competitive baseline →
robustness → scale.**

## 7. Cross-program record

The optimization/representation ≠ downstream capability pattern is
elevated to a documented cross-program hypothesis:
`research/reports/CROSS-PROGRAM-SYNTHESIS-2026-08-30.md`.
