# Architecture

TORUS is organized as a small set of composable layers, each with a
single responsibility and a clean interface to the next. Phase 1 ships
numpy-only reference implementations; later phases swap in accelerated
backends behind the same interfaces.

## Layer Diagram

```
                   ┌─────────────────────────────────────┐
                   │            RecursiveContext          │  ← rlm/context.py
                   │   context-as-variable substrate      │
                   │   slice / grep / chunk / recurse    │
                   └──────────────┬──────────────────────┘
                                  │ ask(slice)
                                  ▼
   ┌──────────────────────────────────────────────────────────────┐
   │                    ResidualTernaryLinear                      │  ← core/residual_linear.py
   │   planes + gate  ──►  ternary_matmul (× num_planes used)      │
   └────────────┬─────────────────────────────────────────────────┘
                │ dispatches per call to:
                ▼
   ┌──────────────────────────────────────────────────────────────┐
   │                ResidualGate (adaptive gate)                   │  ← core/gate.py
   │       ALWAYS / NEVER / ADAPTIVE (per-call heuristic)          │
   └────────────┬─────────────────────────────────────────────────┘
                │ operates on:
                ▼
   ┌──────────────────────────────────────────────────────────────┐
   │          ResidualTernaryPlanes (weight representation)       │  ← quant/residual.py
   │   planes[0] = primary (always), planes[1..k] = residual        │
   │   each is a TernaryPlane (codes + per-group scale)             │
   └────────────┬─────────────────────────────────────────────────┘
                │ built by:
                ▼
   ┌──────────────────────────────────────────────────────────────┐
   │        ternary_quantize / residual_quantize (numpy)           │  ← quant/ternary.py
   │        absmean + per-group scale + threshold                  │
   └──────────────────────────────────────────────────────────────┘
```

MoE hangs off to the side:

```
   ┌──────────────┐    dispatches     ┌────────────────────────────┐
   │ TopKRouter   │ ────────────────► │     ExpertBank             │
   │  (scaffold)  │     per token     │   per-expert ResidualTernaryLinear
   └──────────────┘                   └────────────────────────────┘
```

## Components

### `torus.quant.ternary`

Single-plane ternary quantization. Each plane stores:

- `codes: int8[(out, in)]` — values in `{-1, 0, +1}` (2 bits)
- `scales: float32[(out, n_groups)]` — one FP16-equivalent per group
- `group_size: int`

The pipeline is:

```
s = mean(|w|, axis=-1 over each group)
w_n = w / s
t = round(clip(w_n, -1, 1))
t = 0 where |w_n| < threshold       # sparsity knob (default 0.7)
```

Storage budget for `group_size = 128` is ~2.125 bits/weight:
2 bits for `t` plus 1 bit / 8 weights for the scale.

### `torus.quant.residual`

`residual_quantize(W, num_planes)` stacks planes by successive
residual error:

```
plane_0 = ternary(w)
plane_i = ternary(w - sum_{j<i} plane_j.reconstruct())
```

`compose_planes(planes, active=k)` reconstructs the full-precision
weight from the first `k` planes. The first plane alone gives the
canonical 1.58-bit approximation; adding more planes monotonically
reduces reconstruction error (verified in `tests/test_quant.py`).

### `torus.core.gate`

`ResidualGate(mode, threshold, depth_bias, magnitude_bias)` decides,
per call site, whether to activate the residual plane. Three modes:

- `ALWAYS` — always engage the residual plane (no speed/quality
  trade-off; convenience mode).
- `NEVER` — pure primary plane, maximum efficiency.
- `ADAPTIVE` — score = sigmoid(4 · (mag + depth + biases)) ≥ threshold.

The scoring function is deliberately simple in Phase 1; Phase 3 will
add a learned gate head trained jointly with capability-aware
distillation.

### `torus.core.residual_linear`

`ResidualTernaryLinear` is the unit the kernels target. It owns
`planes` and a `gate` and a `forward(x, ..., depth)` method that
returns `(y, decision)`. The phase-1 implementation dispatches to
`compose_planes(planes, active=k)` then `x @ W.T`. Phase 2 will
replace the matmul with a specialized ternary GEMM that can
short-circuit when the gate is low.

### `torus.moe.expert_bank` & `router`

`ExpertBank` is a dict-shaped container mapping expert ids to
`ResidualTernaryLinear` instances. `TopKRouter(num_experts, top_k)`
returns top-k expert indices and renormalized weights per token. Both
are scaffolding for Phase 1; real learned routing arrives in Phase 3.

### `torus.rlm.context`

`RecursiveContext` is the substrate that turns the prompt into a
variable. The model receives a `RecurseableContext` whose primitives
are:

- `total` / `len(context)`
- `slice(ContextSlice)` — materialize a sub-range to a string
- `grep(pattern)` — find chunks containing a pattern, as `ContextSlice`s
- `chunk(chunk_size)` — split into fixed-size pieces
- `ask(slice | str)` — call the registered `ask_callable` on a piece
- `recurse_on(slice, chunk_size)` — recursively `ask` each sub-piece
  and aggregate

`ContextSlice(start, stop)` is a lightweight, hashable handle for
slice ranges (no copy until materialized).

### `torus.rlm.repl`

`ContextREPL` is a tiny Python environment that binds `context`,
`RecursiveContext`, and `ContextSlice` as builtins. The model writes
Python to manipulate the context; the REPL executes it and returns
stdout + repr of the last expression. Phase 1 is a plain AST-driven
local REPL; Phase 2 will provide a model-facing adapter that turns
natural-language tool calls into REPL statements.

## Data Flow (single inference call)

```
input
  │
  ▼
TopKRouter.route(features) ──► list of (expert_id, weight) per token
  │
  ▼
for each expert per token:
  ├─► ResidualGate.decide(magnitude, depth) ──► activate? (bool)
  ├─► ternary_matmul(x, planes[0])               (always)
  └─► ternary_matmul(x, planes[1..])             (if activate)
  │
  ▼
output = sum weighted over experts (+ bias)
  │
  ▼
softmax/head (model-side)
```

## What Phase 2 replaces

| Aspect                       | Phase 1 (numpy)               | Phase 2 (target)                        |
|------------------------------|-------------------------------|------------------------------------------|
| ternary GEMM                 | `x @ (T * s)`                 | Addition-based ternary kernel, AVX2 (Legion's 3995WX is Zen 2 — no AVX-512) |
| per-token residual gating    | numpy boolean + matmul        | Predicated CUDA / SIMD control flow      |
| MoE routing                  | Phase-1 heuristic             | Learned gating, expert-aware residuals   |
| REPL execution               | Local Python                  | Sandboxed exec, model-facing adapter     |
| Memory layout                | Eager arrays                  | Packed 2-bit planes, lazy resident       |
| Context store                | In-memory list of strings     | NVMe-backed chunk store + prefetch       |

The Phase-2 surface area is identical: every component swap keeps
the same public types and method signatures.

## Phase 5 — RLM / Prime-Agent integration

The Phase-5 wiring closes the loop on long-context inference: the
model treats the prompt as a `RecursiveContext` variable and emits
Python that slices / greps / asks through it instead of trying to
fit the whole prompt into its own working window.

### Components

- `torus.rlm.context.RecursiveContext`: in-memory chunked context
  with `slice`, `grep(pattern, ignore_case=False)`, `chunk`,
  `ask`, and `recurse_on` primitives.
- `torus.rlm.repl.ContextREPL`: tiny Python environment that
  binds `context` as a variable. The model emits code; the REPL
  runs it and reports stdout + last value.
- `torus.rlm.agent.PrimeAgentLoop`: drives the REPL iteratively
  with a `model_fn(prompt) -> str` callable. Each iteration builds
  a prompt from the goal + context summary + history, gets code
  from the model, runs it, and records `AgentStep`s. Loop stops
  when the model emits `### DONE ###` or `max_steps` exhausts.

### Prime Agent data flow

```
goal: str
  |
  v
for step in range(max_steps):
  |
  +--> prompt = prompt_builder(goal, repl, history, context_summary)
  |       |
  |       v
  +--> code = model_fn(prompt)             # the trained ternary model
  |       |
  |       v
  +--> stdout, last_value = repl.run(code)
  |       |
  |       v
  +--> history.append(AgentStep(...))
  |       |
  |       v
  +--> if "### DONE ###" in code: return AgentResult(answer=last_value, steps)
```

The model never sees more than one chunk at a time, so a
`RecursiveContext` measured in millions of tokens works for a
model with a fixed working window.

### Why this is the Phase-5 milestone

- **Long-context without context-window engineering.** The model
  doesn't need an `n_ctx = 1M` attention pattern; it just
  iterates over `RecursiveContext.grep` and `RecursiveContext.slice`.
- **Composability with residual planes.** Each `ask` call inside
  the REPL is a forward pass through the trained ternary model;
  the residual-plane gate fires per call as usual.
- **Composability with MoE.** The REPL's per-call expert routing
  uses the model's own `TopKRouter`; the gate's per-expert
  confidence signal biases the residual-plane decision.

### Demo

`examples/prime_agent_demo.py` runs a 16-chunk fake paper through
the loop with a deterministic stub model. The stub grep's for the
first goal keyword, slices the first match, and emits the answer
in 3 steps without ever loading the full prompt.


## Phase 8 — Persistent (NVMe-backed) Context

Phase 8 swaps the in-memory `RecursiveContext` for a disk-backed
implementation that's the same drop-in shape, with three additions:

- **Chunk files on disk.** Each chunk lives at `<root>/NNNNNNNN.txt`.
- **LRU cache.** Chunks are loaded lazily into a bounded in-memory
  cache (`cache_size=64` by default). Reads are O(1) on the cache,
  O(1) syscall on miss.
- **Append-only writes.** `PersistentContext.add_chunk(text)` is
  the only way to grow. Manifest writes are atomic (write-then-rename).
  Cross-process appends on the same directory are serialized via
  `fcntl` on a `.lock` file.

### Components

- `torus.rlm.persistent.PersistentContext`: the disk-backed context.
- `torus.rlm.repl.ContextREPL`: duck-typed; accepts either
  `RecursiveContext` or `PersistentContext` (no annotation change
  at runtime; both satisfy the same Protocol).
- `torus.rlm.agent.PrimeAgentLoop`: unchanged — drives whatever
  context sits behind `ContextREPL`.

### Data flow

```
add_chunk(text)
  |
  v
acquire .lock via fcntl(LOCK_EX)
  |
  v
write <root>/NNNNNNNN.txt under temp name
  |
  v
fsync + rename(temp -> target)        # atomic on POSIX
  |
  v
write manifest.json via tmp + rename
  |
  v
release .lock

read chunk i:
  |
  v
cache hit? return
  |
  v
cache miss: read <root>/NNNNNNNN.txt, evict LRU if full
```

### Why this is the Phase-8 milestone

- **Prompts measured in millions of tokens.** `RecursiveContext`
  has the full prompt in memory. `PersistentContext` can grow
  past RAM because only the cache footprint is resident.
- **Crash-safe append-only growth.** The directory is the source
  of truth. Reopening it recovers state; no in-memory copy needed.
- **Composability.** `ContextREPL` and `PrimeAgentLoop` are
  agnostic; `PersistentContext` is a drop-in for any code that
  accepts a `RecursiveContext`.

### Demo

`examples/persistent_context_demo.py` writes a 16-chunk paper to a
temp directory, then runs the same PrimeAgentLoop pattern against
it. Storage bytes reported; model grep's the chunk that mentions
Phase 8 and answers in 1 step.

## Phase 2 (security) — SandboxedContextREPL

The Phase-1 `ContextREPL` is `exec(compile(...))` with the model
emitting Python directly. That's a real attack surface — a
malformed model output could call `os.system`, read `/etc/passwd`,
or run arbitrary code.

Phase 2 (security sub-phase) replaces `ContextREPL` with
`SandboxedContextREPL` for any production deployment. The
sandbox enforces a strict policy at *three* layers:

1. **AST whitelist**: before any code runs, we parse it and walk
   the AST. We reject the code (without running it) if it contains:
     - `Import` / `ImportFrom` nodes
     - calls to `exec`, `eval`, `compile`, `__import__`, `open`,
       `getattr`, `setattr`, `globals`, `locals`, `vars`, `dir`
     - attribute access on anything other than the bound name
       `context`
     - subscript access on anything other than a top-level Name
       (so `hits[0]` is fine but `__builtins__["eval"]` is not)

2. **Restricted builtins**: at run time, the code sees only a
   safe `__builtins__` dict — `abs`, `all`, `any`, `bool`,
   `dict`, `enumerate`, `float`, `int`, `isinstance`, `len`,
   `list`, `max`, `min`, `print`, `range`, `repr`, `reversed`,
   `round`, `set`, `sorted`, `str`, `sum`, `tuple`, `type`,
   `zip`. No `os`, no `sys`, no `open`, no `subprocess`.

3. **Resource caps** (per call):
     - `max_lines` (default 64)
     - `max_output` (default 16 KB captured stdout)
     - `max_recursion_depth` (default 16)
     - `timeout_seconds` (optional; enforced via a daemon-thread
       watchdog; daemon means a runaway code can't survive process
       shutdown)

The `SandboxedContextREPL` is API-compatible with `ContextREPL`
(`.run(code) -> (stdout, last)`) so `PrimeAgentLoop` accepts
either transparently via its `repl=` parameter.

## Phase 7 — Multi-expert production wiring

Phase 4 shipped `ExpertBank` (per-expert residual stacks with
optional shared primary) and `TopKRouter` (token-to-expert
routing) but didn't tie them together end-to-end. Phase 7 closes
that loop:

- `MultiExpertRouter(router, bank, policy)` composes the two with
  a `GatePolicy` that maps *router confidence* (top-k prob mass)
  to *plane-count engagement*.
- `GatePolicy(confidence_low, confidence_high, n_planes_low,
  n_planes_high)` is a linear-interpolation policy: low
  confidence → engage the full residual stack, high confidence →
  primary plane only. Defaults map `0.13 → 4 planes` and
  `0.24 → 1 plane` (calibrated against the Phase-1 random
  router; a learned router would span `[0, 1]`).
- `MultiExpertRouter.route(features)` returns a list of
  `PerCallDecision(token_idx, expert_id, weight, confidence,
  n_planes)` — one per (token, expert) pair. An optional
  `on_decision` callback fires for telemetry.

### Why this is the Phase-7 milestone

- **Production-shape wiring.** Phase 4's bank supports N experts;
  Phase 7's `MultiExpertRouter` composes them with the gate's
  adaptive plane-count so the resulting call cost scales with
  router uncertainty, not just expert count.
- **Compute proportional to confidence.** A router that's sure
  on a token engages 1 plane; unsure tokens engage 4. On a
  16-expert × 4-plane bank with a 32-token batch, the demo
  shows ~17% plane-activation savings vs. always-4.
- **Scales.** Verified at 100 experts × 4 planes (400 decisions
  per call) with no slowdown.

### Demo

`examples/multi_expert_demo.py` builds a 16-expert × 4-plane bank
and routes 32 tokens through `MultiExpertRouter`. It reports the
total plane activations vs. a naive always-4 baseline, the
distribution of `n_planes` choices, and a sample of the per-call
decisions. With the calibrated policy, the demo saves ~17% of
plane activations.

  sandbox before production use. (Resolved in this release via
  `SandboxedContextREPL`; see Phase 2 above.)

## Why Python first, hardware later?

Each idea has a *math* plus a *runtime*. Phase 1 nails down the math
with tests that are fast, deterministic, and framework-free. Phase 2
then implements the runtime behind the same types. Splitting the
phases this way lets the math be reviewed independently from the
systems work.
