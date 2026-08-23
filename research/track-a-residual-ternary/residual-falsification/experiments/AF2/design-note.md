# AF2 — Equal-storage tournament design note (cost vector framing)

**Purpose.** A-RP-002 asks whether T2 ternary is competitive with
equal-storage non-ternary correction mechanisms. Per OPERATING-PLAN
§11 v2.3, the primary Track-A decision axis is **capability as a
function of a cost vector**, not a single scalar. AF2 matches **deployed
bytes**, reports **training FLOPs** separately, and tabulates
**inference ops / token**, **memory traffic / token**, **measured
latency / token** when feasible.

## Site
`model.layers.0.mlp.down_proj`, OLMo-1B-0724-hf, shape 8192 × 2048
(16,777,216 weight elements). All arms at this layer.

## Matching cost axis: serialized deployment bytes

Each arm's committed artifact must include its **packed weights +
scales + group metadata + indices + alignment + manifest headers**.
The writer's total file size (sha256 recorded) is the reported deployed
bytes — not the theoretical bit-rate × parameter count.

## Arm list (6 arms)

For each arm the manifest declares:
- the parameterized forward (the operation that adds the correction
  to the chosen layer);
- the serialized representation writer;
- the deployed-bytes measurement (file size on disk).

### T2 ternary (`t2_ternary`, the focal arm)
- One residual plane T2 (signed {-1, 0, +1} weights, latent
  real-valued trainable; ternary quantization under STE; packed to 2
  bpw via 2-bit signed with a per-row fp16 scale).
- Serialized footprint: ternary codes (16.78M × 2 bits = 4.19 MB raw)
  + scales (8192 × 2 bytes = 16 KB) + manifest header ≈ 4.21 MB.
- **Target bytes ≈ 4.21 MB** (all other arms match this).

### INT4 residual (`int4_residual`)
- One residual plane at the same site, signed INT4 (packed 4-bit with
  per-row fp16 scale and per-row fp16 zero-point).
- Full-layer INT4 raw bytes = 8.39 MB (over budget). To match the T2
  footprint, reduce the layer width: train an INT4 correction at the
  same site with a 50% column mask (8192 × 0.5 × 2048 = 8.39M × 0.5
  weight elements ≈ 4.19 MB raw + scales ≈ 4.21 MB). Documented in
  the manifest as "matched-deployment-bytes via column-halving."

### Smaller INT8 residual (`int8_residual`)
- Full layer in INT8: 16.78M × 8 bits = 16.78 MB raw — ~4× over
  budget. To match the T2 footprint: column-quarter mask
  (8192 × 0.25 × 2048 ≈ 4.19M weight elements ≈ 4.19 MB raw + scales
  ≈ 4.21 MB).
- Documented: "matched-deployment-bytes via 25%-column mask."

### LoRA-style low-rank correction (`lora`, rank chosen for byte match)
- A learned down (2048 → r) → up (r → 8192) pair inserted at the same
  site as a parallel residual.
- Choose r to match T2's 4.21 MB at fp16 weights:
  - bytes = (2048 × r + r × 8192) × 2 bytes = 20480 × r bytes.
  - 20480 × r ≈ 4.21 × 1024 × 1024 ⇒ r ≈ 211.
- Round to r=216 (divisible by 16) for tidy packing.

### Small dense FP16 adapter (`dense_adapter`)
- A `(2048 → 2048)` Linear correction at the same site, in fp16.
- Bytes: 2048 × 2048 × 2 = 8.39 MB. To match T2's 4.21 MB, use a
  half-width Linear: `(2048 → 1024 → 8192)` bottleneck to land at
  (2048 × 1024 + 1024 × 8192) × 2 = (2097152 + 8388608) × 2 = 20.97
  MB; too big. Reduce to (2048 → 512 → 8192) = (1048576 + 4194304) ×
  2 = 10.49 MB. Still over. Use (2048 → 192 → 8192) bottleneck:
  (393216 + 1572864) × 2 = 3.93 MB ≈ target.
- Documented: "matched-deployment-bytes via 192-rank bottle."

### Random-capacity controls (random-ternary / random-lora)
- **random_t2_ternary**: T2 ternary, weights drawn from a fixed RNG
  seed, NEVER TRAINED. Same packed footprint as t2_ternary. Isolates
  the contribution of structure vs training.
- **random_lora**: LoRA r=216, weights drawn from the same RNG seed,
  NEVER TRAINED. Isolates the contribution of structure vs training.
- These two are **untrained controls**. They test the AF2.7 claim
  line: "similar improvement → conclusion is only 'additive
  correction helps'; T2 must win on quality-per-bit/compute to
  support ternary specifically." Random arms must NOT be reported in
  the same summary stats; they appear in a separate panel.

## Training recipe (matched across trained arms)

- Wikitext-103 train split, same cache path used in AF1/AF1-R (sha
  captured).
- Identical batches across arms within seed by construction
  (one shared `make_window_sampler`, one cache).
- N=500 steps, batch=4, seq=128, SGD lr=1e-3 momentum=0.9 clip=1.0,
  objective = next-token CE.
- Base FP16 weight frozen for all arms except `t2_ternary` and the
  random controls (by construction: requires_grad_(False) on base).

## Eval

- Full wikitext test, full arc_easy, full lambada_openai. Float16
  throughout. No `--limit`. (Same fix that landed for AF1.)
- Each arm/seed produces eval.summary.json (means per task) +
  eval.full.json (raw LM loglikelihoods) + history.jsonl (per-step
  CE loss curve, diagnostic only).

## Cost vector reporting (mandatory per OPERATING-PLAN §11 v2.3)

For each trained arm and seed the manifest records:

| term                       | source                                              |
|----------------------------|-----------------------------------------------------|
| Deployed bytes             | size of the packed serialized file on disk (sha256) |
| Training FLOPs             | analytic count: 6 × N × batch × seq × hidden² proxy |
| Inference ops / token      | analytic count per matrix-mult at the site          |
| Memory traffic / token     | bytes/tensor reads under a single forward pass      |
| Latency / token (TITAN RTX)| measured wall-clock median of 100 forward passes    |
| Energy / token             | optional, recorded if nvidia-smi power readout OK   |

For the **untrained** random controls the deployed bytes + sampling
overhead are reported; training FLOPs = 0.

## A-RP-002 thresholds (proposed; locked at PROPOSE)

- **Pareto competitiveness (PASS)**: at least one capability metric
  reaches within 2 stderr-of-difference of the dense_adapter arm's
  per-metric mean AND no metric regresses by more than 2 stderr
  below it. (We test T2 vs the strongest matched-storage competitor
  — the dense_adapter — for the strict sense of "competitive.")
- **Substantial Pareto advantage (PASS+)**: at least one capability
  metric exceeds the dense_adapter's mean by >2 stderr AND at the
  same time the deployed bytes is ≤ the dense_adapter's deployed
  bytes by a factor of at most 1.0 (matched by construction).
- **FAIL**: T2 is dominated across all metrics by the dense_adapter
  within 2 stderr, AND no other matched-storage competitor matches
  T2 either.

## Decision rule

The verdict computes the **Pareto frontier** across
(deployed_bytes, capability_wikitext_ppl_improvement) for all trained
arms and tabulates all 6 cost-vector terms; a T2 arm on the frontier
is "Pareto-competitive." Strength of competitiveness (PASS+ vs PASS
vs FAIL) per the thresholds above.

The random-ternary / random-lora controls appear in a separate panel
("untrained structure baseline"). A finding like "T2 is on the Pareto
frontier but the random-ternary control lands on the same frontier"
is a strong argument that the **representation** isn't what's
contributing; the **training signal** is. Such a finding would
flag A-RP-002 as PROVISIONAL_FAIL and trigger a follow-up that
isolates the contribution of training vs structure.

## Constraints

- 8 GPU-hours wall budget on Legion; expected ~3 GPU-hours for 18
  (6 arms × 3 seeds) runs at ~10 min each.
- Driver freeze exception (OPERATING-PLAN §3, §6): AF2 needs new code
  for INT4/INT8 packing, random controls, byte-count serialization,
  LoRA + dense_adapter adapters, latency timing; logged in the
  manifest's `freeze_exception` block.
- All AF2 artifacts gate behind `.gitignore` for adapter.npz
  (mirror AF1 + AF1-R).
