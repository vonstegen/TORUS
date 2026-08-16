# Phase 3 Eval Report — 2026-08-16

## Setup

| | |
|---|---|
| **Student** | allenai/OLMo-1B-0724-hf (1B params, FP16) |
| **Teacher** | allenai/OLMo-7B-0724-hf (7B params, FP16) — asymmetric |
| **Quantized linears** | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj (114 linears) |
| **Train steps** | 100 (per stage) |
| **Train loss** | p1: 4.34 → 3.94 (Δ=+0.41); p1+res: 4.34 → 4.03 (Δ=+0.32) |
| **Eval harness** | lm-eval 0.4.3 (downgraded from 0.4.5 — 0.4.5 requires `AutoModelForVision2Seq` which is not in transformers 5.15) |
| **Tasks** | wikitext, lambada_openai, arc_easy, (hellaswag not loadable in datasets 5.0+) |
| **Rows** | 4 lm-eval example limit by default; full task set for the final |

## What was built

| commit | subject |
|---|---|
| `4c81638` | v0.15.0 — adapter save/load + eval_lm.py scaffolding |
| `7f8efd1` | v0.15.1 — `apply_eval_mode` fast path (100-1000× speedup) |

- `HFStudentAdapter.save_state(path)` / `load_state(path)` — serialize STE + residual weights to `.npz` + JSON sidecar
- `distill_run.py --save-adapter PATH` / `--load-adapter PATH` — checkpoint I/O
- `examples/eval_lm.py` — lm-eval-harness wrapper with `--mode baseline` and `--mode quantized --load-adapter`
- `apply_eval_mode(n_planes)` — quantize once, stash GPU tensor on patched module, fast forward
- `apply_train_mode()` — restore the per-call STE quantize path

## Final numbers

| task | metric | baseline | PTQ | p1_only | p1+residual |
|---|---|---|---|---|---|
| arc_easy | acc,none | 0.6073 | 0.2584 | 0.2584 | 0.2580 |
| lambada_openai | acc,none | 0.6095 | 0.0012 | 0.0012 | 0.0002 |
| wikitext | word_perplexity,none | 13.0932 | 465097 | 465097 | 759750 |

| variant | avg % of baseline | gap to ≥90% threshold |
|---|---|---|
| **PTQ** (FP16 → ternary, no training) | 14.25% | -75.75pp |
| **p1_only** (distilled, plane 1) | 14.25% | -75.75pp |
| **p1+residual** (distilled, planes 1+2) | 14.17% | -75.83pp |

**PHASE 3 GATE: FAIL** — far below the ≥90% threshold.

## Root cause

The 16-layer OLMo-1B is collapsed by per-layer ternary quantization. For one layer:
- Original weight norm: 19.9
- Quantized weight norm: 11.95 (~40% reduction)

Across 16 layers this compounds multiplicatively: 0.6¹⁶ ≈ 0.0003. The projected logits near the lm_head are effectively zero, so perplexity explodes.

The training runs did improve the loss (4.34 → 3.94) but the model is still in the basin where the loss landscape is dominated by the per-layer quantization noise, not by the FP16 weights being learned. The 100-step distillation on synthetic random tokens can't pull the model out of this basin.

PTQ (no training) gives **identical** results to the trained p1_only (same wikitext ppl, same lambada accuracy). The training is barely moving the bar — because the FP16 init has the same collapse, and the SGD updates are tiny relative to the weight scale.

### What the training IS doing

- The KL distillation loss is well-formed and decreasing (4.34 → 3.94)
- The autograd path is correct (fixed in v0.14.4)
- The 7B → 1B teacher signal is real (5× bigger improvement than self-distill)
- The residual plane is being learned (0.01 → 0.05 norm over 100 steps)

But none of this matters because the *baseline* (FP16 init + 1× quantize) is already in the degenerate regime. The model needs to be **calibrated post-quantization** to recover from the per-layer norm loss before the distillation signal can do anything useful.

## What's pipeline-correct

The eval pipeline is end-to-end functional:
- ✅ Baseline FP16 loads and runs at native lm-eval speed
- ✅ Quantized student loads from `.npz`, restores STE + residual weights
- ✅ `apply_eval_mode` switches to a fast forward path (numerically identical to slow path; verified diff = 0.0)
- ✅ Three tasks run cleanly on both modes
- ✅ Per-task summary JSON written to `/tmp/eval_out/*.json`
- ✅ Comparison script computes % of baseline per task + 90% gate verdict

**The measurement is real.** The number is bad, but the number is accurately bad.

## What needs to change before the gate passes

1. **Per-layer norm calibration** — scale up the quantized weight so its norm matches the FP16 reference. Without this, 16-layer models collapse.
2. **More training steps** — 100 steps is not enough to recover from the per-layer quantization error even with calibration. Need 1000-10000 steps.
3. **Real text data** — synthetic random tokens (the current data iterator) is fine for the trainer but doesn't actually exercise the model. Wikitext, OpenWebText, or Pile would give a meaningful loss signal.
4. **Train dense **first** only on the primary plane, then engage the residual** — this is the curriculum pattern but the current curriculum has the switch happen too early.

## Files produced

| path | purpose |
|---|---|
| `/tmp/eval_out/baseline.json` | FP16 baseline on 3 tasks |
| `/tmp/eval_out/ptq.json` | PTQ (no distillation) |
| `/tmp/eval_out/distilled_p1_only.json` | 100-step asymmetric distill, plane 1 |
| `/tmp/eval_out/distilled_p1_plus_residual.json` | 100-step asymmetric, planes 1+2 |
| `/tmp/eval_out/student_p1_only.npz` | 4.3GB checkpoint |
| `/tmp/eval_out/student_p1_plus_residual.npz` | 4.3GB checkpoint |
| `/tmp/eval_out/eval_p1_only.json` | distillation loss curve |
| `/tmp/eval_out/eval_p1_plus_residual.json` | distillation loss curve |
| `examples/eval_lm.py` | lm-eval runner, 2 modes |
| `torus/train/hf_adapter.py` | `save_state`, `load_state`, `apply_eval_mode`, `apply_train_mode` |
| `examples/distill_run.py` | `--save-adapter`, `--load-adapter` flags |

## Phase 3 status

Phase 3 acceptance gate is **not met**. The eval pipeline is in place and gives accurate measurements, but the underlying QAT recipe needs calibration before the gate can pass. Recommended next steps:

1. Add per-layer norm calibration to `apply_eval_mode` and the train-time quantize (one-knob fix, ~20 lines)
2. Run distillation on real text data (wikitext or pile-uncopyrighted)
3. Increase to 1000-10000 steps
4. Re-run eval; expect to land somewhere in the 60-90% range initially, then iterate
