# Handoff Prompt — TORUS Phase 1 Kickoff (2026-08-22)

Paste the block below into the new session.

---

You are continuing the TORUS research program. The repository is
`github.com/vonstegen/TORUS`, cloned locally at `/tmp/TORUS` (branch `main`,
latest commit `b27f45a`). Start by running `git -C /tmp/TORUS pull --ff-only`
in case anything newer landed.

## Read first (governing documents, in authority order)

1. `research/OPERATING-PLAN.md` — process authority (claim-driven lifecycle,
   compute tiers, provenance gates, unlock rules, no-exception rules).
2. `research/ROADMAP.md` — sequencing and gates (7 phases, checklists,
   checkpoints).
3. `docs/TORUS-feedback/08-HARNESS-INSTRUCTIONS-V2.md` and
   `docs/TORUS-feedback/10-RESIDUAL-PLANE-FALSIFICATION-SUITE-V2.md` —
   harness behavior and the Track A-F falsification authority.

Also skim `research/registry/INDEX.md` (experiment ledger) and the claim
files in `research/track-a-residual-ternary/residual-falsification/claims/`.

## Current state

- Phase 0 is COMPLETE. Gate G0→1 is OPEN (2026-08-22).
- EXP-A-001 DECIDED PASS: baselines reproduce exactly on Legion —
  fp16 ARC-E 0.60732 / LAMBADA 0.60955 / wikitext ppl 13.0932;
  PTQ-uncalibrated ppl 459,454; PTQ-calibrated ppl 89,557 (5.13× better on
  ppl but slightly worse on ARC-E/LAMBADA — recorded anomaly).
- Claims A-RP-001/002/003 are registered, state UNTESTED, with quantitative
  thresholds (>2 standard errors on capability metrics).
- Feature freeze is ACTIVE. Code changes are allowed only when required to
  execute a registered experiment; log each exception in that experiment's
  manifest.
- Baseline tag: `research-baseline-2026-08`. Every experiment records a
  descendant SHA.

## Infrastructure

- Test machine is **Legion** (NOT the local dev box): `ssh legion`
  (x86_64, 2× TITAN RTX 24GB, repo at `~/TORUS`, CUDA venv `.venv` with
  Python 3.14, torch 2.13.0+cu130, transformers 4.57.6, lm-eval 0.4.3,
  datasets 5.0.1). Model + datasets already in the HF cache. GPUs must be
  verified idle (`nvidia-smi`) before launching runs.
- Legion has a preserved pre-baseline branch `legion-pre-baseline-2026-08`
  (do not delete; it holds the EXP-A-000 working state).
- Run namespaces: `runs/<track>/<experiment_id>/<timestamp>/` on Legion,
  git-ignored; committed record = manifest + verdict + eval JSONs +
  ARTIFACTS.json + env-lock under `research/`. Never repair a run in place;
  corrections get new EXP IDs.

## Your task: Phase 1, first experiment — EXP-A-011 (A1 layer sensitivity)

Per the roadmap Phase 1 checklist and the EXP-A-001 verdict ("smallest
justified next experiment"):

1. **Preregister** `EXP-A-011` (copy `research/registry/EXP-TEMPLATE.yaml`
   into `research/track-a-residual-ternary/EXP-A-011/manifest.yaml`) and
   commit BEFORE any run. Content per roadmap item 1.1 and feedback doc 03
   §A1: quantize one layer at a time (restore before moving on) across
   q/k/v/o/gate/up/down_proj and embeddings/head, representative
   early/middle/late blocks of OLMo-1B, measuring output error, cosine
   similarity, logit KL, downstream task delta on a bounded eval subset,
   physical bytes, and operations. Set numeric thresholds and a compute
   budget up front.
2. **Manifest must fix the known audit gap:** persist the FULL lm-eval
   results dict (including per-task stderrs), not the one-metric summary.
3. Build the sensitivity harness (a freeze exception — log it in the
   manifest). Reuse `torus.train.hf_adapter.HFStudentAdapter`,
   `apply_eval_mode`, and the `--no-calibrate` knob from EXP-A-001; do not
   invent a second quantization path.
4. Run on Legion in the background (nohup pattern from
   `runs/a/EXP-A-001/`; see `research/baselines/EXP-A-001/` for the
   artifact layout to imitate), then AUDIT against the frozen thresholds,
   DECIDE (PASS/FAIL/INVALID/CONTINUE), write `verdict.md`, update
   `research/registry/INDEX.md` and the roadmap checkpoint CP1.1, commit,
   push.
5. Do NOT start A3 training runs or anything Phase 2+; Track B stays locked.
   Discovery tier only: cheap experiments designed to kill weak hypotheses.

## Standing rules that bite

- Preregister thresholds before results exist; never change them after.
- KL is diagnostic, never acceptance; report proxy AND capability metrics.
- Uncertain provenance → INVALID, rerun cleanly. One writer per namespace.
- Decision output must include: hypothesis, result, grade, decision,
  confidence/reproduction status, next permitted experiment, experiments
  explicitly blocked.
- Commit and push each lifecycle transition (PROPOSE → RUN → DECIDE).
