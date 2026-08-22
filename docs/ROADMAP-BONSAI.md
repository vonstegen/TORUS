# ROADMAP — Hadamard integration + corrected-recipe prototype on Ternary-Bonsai-1.7B

Created 2026-08-21. Living doc; update checkboxes as gates complete.
Baselines (measured, full lm-eval, this harness):
- Stock Bonsai-1.7B: ARC-E 0.6881, LAMBADA 0.5003
- Teacher Qwen3-1.7B fp16: reference for KL evals
- TORUS PV-tune of Bonsai: ARC-E 0.6902, LAMBADA 0.4640 (NEGATIVE — do not repeat that recipe)

Noise floors (full runs): ARC-E SE ~1.0pp, LAMBADA SE ~0.7pp.
Minimum meaningful movement: >2x SE (ARC-E 2pp, LAMBADA 1.4pp).

## Standing-still / backward-motion rules (apply at every gate)
1. Every eval compares against BOTH stock baseline AND best-so-far.
2. Any gate showing task regression >2pp vs best-so-far -> stop, diagnose, do not continue spending.
3. Two consecutive eval checkpoints with no best-so-far improvement -> stop, reassess before more GPU.
4. Every gate records numbers in RUN_NOTES.md BEFORE the next phase starts.

---

## Phase 0 — Machinery (timebox: 0.5 day)
- [ ] FWHT kernel: butterfly for 2^k dims; Kronecker H12 x H512 for 6144 (down_proj input)
- [ ] HLinear module: fp32 latents W_rot = H @ W_bonsai; forward x_h=fwht(x); w_q=ternary_ste(W_rot); y=F.linear(x_h, w_q)
- [ ] Wire into 56 linears (o_proj, down_proj x28); q/k/v/gate/up stay plain ternary STE
- [ ] Fold-back export: store H @ w_q as plain fp16 => stock-HF-loadable checkpoint

**CHECKPOINT C0 (identity gate):** STE bypassed (w_q = W_rot): rotated model logits == stock Bonsai logits within fp16 rounding (max|diff| ~1e-3).
- PASS -> Phase 1. FAIL -> bug; fix or halt. No GPU spend past this gate without identity.

## Phase 1 — Price of admission (timebox: 2h)
- [ ] Enable ternary on rotated latents; measure KL(rotated-requant vs teacher) vs KL(stock vs teacher)
- [ ] Quick ARC-E/LAMBADA (limit 500) for task-level price

**CHECKPOINT C1 (requant drop):** record initial KL rise from rotation+requantization.
- This is the hole CPT must climb out of. Proceed to Phase 2 regardless (slope is the test), but if drop > 0.5 KL, flag: rotation's conditioning gain must be large to net out.

## Phase 2 — Smoke arms, falsifiable (timebox: 4h GPU)
- [ ] Arm H: 200 steps CPT, rotated o_proj/down_proj, seq 512, T=1, next-token CE, fixed seed/data order
- [ ] Arm C: identical recipe, unrotated (control — without this, nothing is attributable)
- [ ] Paired end-state KL eval (same eval windows, t-test as in TORUS arms)

**CHECKPOINT C2 (rotation decision):** H final KL < C final KL beyond noise (t > 2).
- PASS -> Phase 3 with rotation.
- FAIL -> rotations bounded for natively-ternary models (record finding); Phase 3 proceeds WITHOUT H. Either way the pipeline test continues — no standing still.

## Phase 3 — Pipeline assembly (timebox: 1 day)
- [ ] CPT warm-up phase: ~2-3k steps plain CE in quantized form (BitDistill stage-2 analogue)
- [ ] Multi-level KD: T=1.0 logit KL + attention-map distillation (MiniLM-style), seq 512, real corpus slice
- [ ] LSQ learned thresholds ONLY if headroom remains after the above

**CHECKPOINT C3 (mid-pipeline sanity):** after CPT phase: no task regression >2pp vs stock; KL trending down.
- FAIL -> stop; the corrected recipe itself is implicated; write up.

## Phase 4 — Main run (timebox: ~2 days GPU)
- [ ] 5-10k steps; eval ARC-E + LAMBADA + KL every 1k steps; checkpoint each eval

**CHECKPOINTS C4.1 ... C4.k (every 1k steps):**
- KILL criterion: no LAMBADA movement (>1.4pp) by step 3000 -> stop; recipe axis bounded; write up negative.
- Backward rule: any eval >2pp below best-so-far -> revert to best checkpoint, reassess.

## Phase 5 — Verdict + deliverable (timebox: 0.5 day)
- [ ] Full final eval, paired stats vs stock Bonsai
- [ ] Fold-back export of best checkpoint
- [ ] RUN_NOTES final entry + writeup

**CHECKPOINT C5 (ship gate):**
- SHIP: ARC-E or LAMBADA beats stock by >2x SE -> release checkpoint + recipe + measurements.
- NO-SHIP: negative-result writeup; the training-process axis is bounded with evidence.

---

## Budget
| Phase | GPU time | Cumulative worst case |
|---|---|---|
| 0-1 | ~2h | day 1 |
| 2 | ~4h | day 1 |
| 3 | ~1 day | day 2 |
| 4 | ~2 days | day 4 |
| 5 | ~2h | day 4 |

Earliest exit: C2 fail on day 1 (rotations bounded, pivot). Second exit: C4 kill on day 2-3.
Hard stop: end of day 4 regardless.
