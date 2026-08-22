# TORUS Chat Session Transcript

**Date:** 2026-08-21  
**Conversation:** Review TORUS Repository

---

## User

Review the TORUS project on GitHub

## Assistant

Yes. I can do a full repository-level review of TORUS through the connected GitHub account, including architecture, source code, documentation, commit history, issues, tests, and how the pieces fit together.

I searched the GitHub connection, but **I don't see a repository identifiable as TORUS in the repositories currently accessible to the connector**, and a public GitHub search for “TORUS” returns unrelated projects such as `coreos/torus`.

Send me the **TORUS GitHub repository URL** (for example, `https://github.com/<owner>/<repo>`). Once I have it, I'll review it as an engineering project rather than merely summarize the README: architecture and module boundaries, local-LLM/agent design, model/runtime layer, memory/context system, dependencies, security/trust boundaries, current implementation versus stated vision, technical debt, tests/CI, and the most sensible development path from its present state.

Given the earlier lightweight AI-OS work, I'll also specifically evaluate whether TORUS is developing into a viable **small local-model-centered administrative/orchestration layer**, and where ternary or other highly efficient local models could realistically fit.

---

## User

https://github.com/vonstegen/TORUS

## Assistant

I reviewed the actual `vonstegen/TORUS` repository rather than only the README. TORUS is already a **real research prototype**, not merely an architectural proposal. The repository was created August 13, 2026, is currently Python-based, and describes itself as the “Ternary Optimized Recursive Unified System,” combining residual ternary inference, adaptive computation, and recursive context handling. 

My overall assessment is:

**TORUS has a genuinely interesting core idea and a surprisingly coherent architecture, but it is currently much closer to an inference-research framework than a complete local LLM runtime.** The next milestone should not be adding more architectural concepts. It should be proving the central TORUS hypothesis on a real pretrained language model.

### What TORUS actually is

There are really **three systems being co-designed**.

The first is the weight representation:

\[
W \approx s_1T_1+s_2T_2+\cdots+s_nT_n,\qquad
T_i\in\{-1,0,+1\}
\]

Instead of deciding that a model is either ternary or higher precision, TORUS decomposes a weight matrix into a primary ternary plane followed by successive ternary approximations of the remaining error. `residual_quantize()` actually implements that procedure iteratively: quantize the current residual, reconstruct it, subtract it, and quantize what remains. 

That is the strongest concept in TORUS.

It means precision becomes **progressive computation**:

```text
easy operation
    ↓
T₁

harder operation
    ↓
T₁ + T₂

very difficult operation
    ↓
T₁ + T₂ + T₃ ...
```

That is considerably more interesting than ordinary static ternary quantization.

The second system is the residual gate. TORUS already provides `NEVER`, `ALWAYS`, and `ADAPTIVE` modes, with signals including residual magnitude, depth, and MoE router confidence. Low router confidence can bias TORUS toward engaging additional precision. 

The third system is the RLM/context layer. The repository has `RecursiveContext`, persistent disk-backed context, a REPL, a sandboxed REPL, indexing, and a `PrimeAgentLoop`. The model can effectively interrogate a large external context rather than placing all of it inside attention at once. 

The Prime Agent interface is particularly clean: the model itself is abstracted as essentially:

```python
Callable[[str], str]
```

So the context machinery does not depend on Torch, Transformers, or even a particular LLM runtime. 

That separation is good architecture.

---

## The architecture I see

Conceptually, TORUS currently looks like this:

```text
                         TORUS
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
         ▼                 ▼                 ▼
   PRECISION             COMPUTE           CONTEXT
     SYSTEM              CONTROL            SYSTEM
         │                 │                 │
 Residual ternary     ResidualGate      RecursiveContext
     planes                │                 │
         │                 │            PersistentContext
   T₁ + T₂ + T₃...        │                 │
         │                 │              Index
         └──────► ResidualTernaryLinear      │
                           │                REPL
                           │                 │
                     CPU/C/CUDA          Sandbox
                       kernels               │
                           │            PrimeAgentLoop
                           │
                         model
```

The repository organization supports that interpretation. The package already contains separate `core`, `quant`, `kernels`, `moe`, `rlm`, and `train` areas. 

There is also a nontrivial test suite covering quantization, gates, real kernels, packing, MoE, RLM, HF integration, residual linear layers, and training. 

So this is substantially beyond a README concept.

---

# What I think TORUS gets right

### 1. Precision is being treated as a runtime resource

This is probably TORUS's most important conceptual contribution.

Traditional quantization asks:

> How many bits does this model use?

TORUS is implicitly asking:

> **How many bits of effective information does this computation need right now?**

That is a better question.

If a token/layer/expert can be resolved adequately using T₁, there is little reason to calculate T₂ or T₃.

This turns model precision into something resembling conditional computation.

That connects naturally with MoE.

You can potentially have two independent sparsity axes:

\[
\text{expert sparsity}
\times
\text{precision sparsity}
\]

or:

```text
token
 │
 ├── expert 2
 │       └── T₁
 │
 └── expert 7
         └── T₁ + T₂
```

That's an excellent research direction.

---

### 2. TORUS preserves mathematical decomposition

The residual representation is very understandable.

The repository isn't hiding the approximation behind an opaque quantizer. The mathematical contract remains:

\[
W =
\hat W_1 +
R_1
\]

then

\[
R_1 =
\hat W_2 +
R_2
\]

therefore

\[
W\approx\hat W_1+\hat W_2+\cdots
\]

That makes it testable, analyzable, and hardware-friendly.

---

### 3. Hardware acceleration is behind an interface

`ResidualTernaryLinear` delegates to registered kernels rather than hardcoding one execution implementation. The current code supports per-plane accumulation and records operation counts. 

That's exactly how I would structure an experimental project like this:

```text
mathematical contract
       ↓
reference implementation
       ↓
optimized CPU implementation
       ↓
SIMD
       ↓
GPU
       ↓
eventually custom hardware
```

Rather than changing semantics each time the accelerator changes.

---

### 4. RLM is kept outside the quantization machinery

This matters.

It would have been easy to turn TORUS into one giant monolithic AI architecture.

Instead, the RLM layer is largely model-independent.

That means TORUS could ultimately support:

```text
TORUS model
BitNet
Qwen
Gemma
OLMo
Llama
another ternary model
API model
```

behind the same context engine.

That also makes TORUS potentially valuable **even if residual ternary planes do not ultimately outperform another quantization method**.

That's an important architectural property.

---

# The biggest thing missing

TORUS has not yet demonstrated the key equation:

\[
\boxed{
\text{TORUS ternary model}
\approx
\text{FP/BF16 model quality}
\quad
\text{at substantially lower energy/compute}
}
\]

The roadmap acknowledges this.

Phase 3 still needs real model wiring, actual models, learned gates, perplexity/downstream evaluation, and released weights. The stated acceptance target is ≥90% of the FP16 baseline on representative evaluations. 

That should now become overwhelmingly the project's priority.

I would temporarily stop adding major conceptual subsystems.

You already have enough architecture to run the experiment.

---

# There is also an important gating limitation

Despite the conceptual claim of per-token adaptive precision, the current `ResidualTernaryLinear` does **not yet truly perform independent per-token plane execution**.

This portion is important:

```python
return self.planes.num_planes if bool(decision.activate.any()) else 1
```

If any member of the decision array wants the residual planes, the entire invocation engages all planes.

And `residual_ternary_matmul()` explicitly raises:

```python
NotImplementedError(
    "per-row activation requires kernel support; coming in Phase 3."
)
```



So today the execution semantics are closer to:

```text
batch/call adaptive
```

than truly:

```text
token adaptive
```

That distinction matters enormously for the eventual efficiency claim.

If one difficult token causes an entire batch to engage residual planes, a lot of the theoretical savings can disappear.

True granular conditional execution is therefore one of the major engineering milestones.

---

# The heuristic gate also needs replacement

The present adaptive score is essentially:

\[
\sigma\left(
4(magnitude+depth+(1-confidence)+biases)
\right)
\]



That's perfectly reasonable for proving the software architecture.

But it isn't enough to establish the TORUS thesis.

Eventually the model should learn something closer to:

\[
P(\text{additional precision improves next computation}\mid h_t)
\]

where \(h_t\) is information already available from the model.

I would train the gate with a cost-sensitive objective such as:

\[
L =
L_{\text{task}}
+
\lambda C_{\text{residual}}
\]

where \(C_{\text{residual}}\) penalizes activating additional planes.

Then TORUS learns the Pareto frontier rather than applying a hand-defined threshold.

---

# One repository issue I would correct immediately

The packaging configuration lists:

```toml
packages = [
    "torus",
    "torus.core",
    "torus.quant",
    "torus.moe",
    "torus.rlm",
    "torus.kernels",
]
```

but **does not list `torus.train`**, even though the README and roadmap treat the training package as part of the implementation. 

That means a built distribution may omit training modules even though an editable source checkout works.

I'd switch to setuptools package discovery rather than maintaining this manually.

For example, conceptually:

```toml
[tool.setuptools.packages.find]
include = ["torus*"]
```

That also protects you as new packages are added.

---

# There may also be a C-kernel packaging problem

The project configures `torus_kernel.c` as a Python extension module, while the C file appears to expose ordinary C ABI functions such as:

```c
void ternary_gemm_portable(...)
```

rather than a Python `PyInit_*` extension entry point. 

At the same time the comments indicate the runtime expects something like:

```text
libtorus_kernel.so
```

and loads it independently.

Those are two different distribution models:

```text
Python extension
    torus_kernel.cpython-312-...so

vs.

shared C library
    libtorus_kernel.so
```

I'd examine the wheel build carefully. A dedicated shared-library build step may be more appropriate than declaring it as a Python extension.

This is worth testing with:

```bash
python -m build
pip install dist/*.whl
python -c "import torus"
pytest
```

from a directory **outside the repository**.

That catches source-tree masking problems.

---

# I found one definite hardware-documentation error

The roadmap says the Ryzen Threadripper PRO **3995WX** system has:

> `AVX2 + AVX-512 available`



That needs correction.

The 3995WX is a 2020 Zen 2 / Castle Peak processor. AMD's current documentation identifies it as that generation, while AVX-512 support appears on later Threadripper PRO generations; AMD explicitly lists AVX-512 on the 7000 WX family. citeturn545819search1turn545819search6

So Legion should be treated as:

```text
3995WX
x86-64
AVX2
NO AVX-512
```

That potentially changes which optimized TORUS CPU kernel should be the real Legion target.

Interestingly, the GB10 entry appears correct: NVIDIA currently lists GB10/DGX Spark as compute capability **12.1**. citeturn545819search0

---

# The README needs a cleanup pass

There are several indications that the implementation has moved faster than the documentation.

For example, the README simultaneously says:

> Compiled C ternary GEMM ... implemented

and later:

> Hardware kernels ... planned



The repository layout section also appears malformed—it begins with a Phase 2 bullet where the package tree should be.

The ROADMAP contains duplicated Phase-5 prose as well. 

None of that is architecturally serious, but fixing it now would make TORUS much easier for outside researchers to understand.

---

# Where TORUS is in maturity

I'd characterize it this way:

```text
Idea / hypothesis            ██████████  strong
Architecture                 █████████░  strong
Reference implementation     ████████░░  substantial
Unit testing                 ████████░░  substantial
Kernel experimentation       ███████░░░  progressing
RLM/context substrate        ████████░░  substantial
Model integration            ███░░░░░░░  early
Real trained TORUS model     █░░░░░░░░░  not demonstrated
Learned adaptive precision   █░░░░░░░░░  not demonstrated
End-to-end serving           ██░░░░░░░░  early
Energy evidence              █░░░░░░░░░  not established
```

That isn't criticism of the project—it tells us exactly what experiment should come next.

---

# I would change the development priority

The roadmap currently stretches through kernels, training, runtime, persistent context, multi-expert routing, and eventually custom hardware.

I would collapse the immediate project into one goal:

## TORUS M1 — The Residual Plane Test

Take **one real transformer**.

Not 70B.

Not MoE.

Not million-token context.

Something like 0.5B–3B first.

Then compare:

```text
FP16/BF16
    ↓
INT8
    ↓
4-bit
    ↓
single ternary plane
    ↓
2-plane TORUS
    ↓
3-plane TORUS
    ↓
adaptive 1–3 plane TORUS
```

Measure:

\[
\begin{array}{l}
\text{perplexity}\\
\text{benchmark accuracy}\\
\text{tokens/sec}\\
\text{RAM/VRAM}\\
\text{memory bandwidth}\\
\text{plane activation rate}\\
\text{operations/token}\\
\text{joules/token}
\end{array}
\]

Then the critical plot becomes:

```text
QUALITY
  ▲
  │                       ● FP16
  │                  ● INT8
  │             ● TORUS-3
  │         ● TORUS adaptive
  │     ● TORUS-2
  │ ● ternary
  └──────────────────────────────► ENERGY / COMPUTE
```

If adaptive TORUS bends that Pareto curve outward, **you have something scientifically important**.

---

# And this changes how I see TORUS relative to your other projects

After inspecting the repo, I would **not** make TORUS itself the lightweight local-AI administrator we were discussing earlier.

TORUS is becoming something lower-level and potentially more valuable:

```text
             Local AI System
                   │
         small administrator LLM
                   │
          orchestration harness
                   │
            ContextHelix
                   │
              TORUS runtime
                   │
      adaptive ternary inference
                   │
        CPU / GPU / accelerator
```

TORUS should probably remain focused on:

> **How can a model dynamically spend precision, computation, memory, and context only where intelligence actually requires them?**

The lightweight local administrator/harness can then **use TORUS** rather than TORUS turning into that harness.

That separation would keep TORUS scientifically sharp.

And I think there is an even deeper architecture hiding here:

\[
\boxed{
\text{TORUS}
=
\text{conditional parameters}
+
\text{conditional precision}
+
\text{conditional context}
}
\]

MoE determines **which parameters** need to execute.

Residual planes determine **how precisely** they need to execute.

RLM determines **which context** needs to enter the computation.

That unifying interpretation is considerably stronger than describing TORUS simply as a “ternary LLM inference project.”

---

## User

Our testing results seem to be coming back negative and nowhere productive of creating a new, innovative learning ternary or training. My question is: should we break this project up into the three core elements that you have discovered and train or test each one and grade them individually, or try to take all three and test them at the same time, which causes more complexity?

## Assistant

Yes — **break TORUS into the three core elements and grade each one independently first**. Right now, testing all three together is making it too hard to know *why* a result is negative.

The three elements are:

1. **Residual ternary representation** — does `T₁ + T₂ + ...` actually recover enough model quality for the added compute/storage?
2. **Adaptive precision/gating** — can a gate reliably decide when extra planes are worth activating, and does it save meaningful compute?
3. **Recursive context/RLM** — can the context-as-variable approach improve effective long-context capability for small/local models without excessive overhead?

Treat these as **three hypotheses**, not one architecture.

A good experimental structure is:

| Track | Question | Pass criterion |
|---|---|---|
| **A — Ternary Planes** | Does residual ternary quantization beat simpler low-bit alternatives on the quality/compute curve? | Better quality than 1-plane ternary at materially lower cost than 4-bit |
| **B — Adaptive Compute** | Can compute be selectively increased only where needed? | Same/near-same quality as full multi-plane with substantially fewer plane activations |
| **C — Recursive Context** | Can a small model solve large-context tasks by retrieving/recursing rather than attending to everything? | Better accuracy per token/compute than ordinary fixed-window prompting |

And I would make an important change: **do not train anything new initially unless the isolated experiment proves there is something worth training.**

For Track A, start with an existing pretrained model and do post-training experiments:

```text
FP16
  ↓
4-bit baseline
  ↓
single ternary plane
  ↓
2 residual planes
  ↓
3 residual planes
```

Measure perplexity, benchmark accuracy, memory, actual operations, and energy if possible. If two or three ternary planes still do not land on an attractive Pareto frontier, stop there. Don't spend weeks training a TORUS model to rescue a representation that doesn't appear promising.

For Track B, you can initially **remove learning from the gate completely**. Give yourself an oracle experiment first. Run every sample with one plane and all planes, identify where extra planes actually improve the answer/loss, and ask:

> If I had a perfect gate, how much computation could TORUS theoretically save?

That experiment is extremely valuable. If even an **oracle gate** only saves 5–10%, then developing a sophisticated learned gate is probably not worth it. If the oracle says 50–70% of residual-plane computations can be skipped with almost no quality loss, then there is clearly something worth learning.

Track C should be even more independent. Test RLM using a normal competent model—Qwen, Gemma, Llama, etc.—not a ternary TORUS model. That tells you whether the context mechanism itself has merit. If RLM works with a conventional model, it becomes a useful TORUS component regardless of what happens to the ternary research.

The experimental flow I'd use is:

```text
                   TORUS
                     │
          ┌──────────┼──────────┐
          │          │          │
          ▼          ▼          ▼
      Track A     Track B     Track C
      Ternary     Adaptive     Recursive
      Planes      Precision     Context
          │          │          │
          ▼          ▼          ▼
       Grade A     Grade B     Grade C
          │          │          │
          └──────────┼──────────┘
                     │
             only combine winners
                     │
                     ▼
                TORUS v2
```

I would actually give each track an explicit **A–F research grade** with predetermined criteria. That prevents us from rationalizing weak results because we are attached to the larger idea.

For example, residual ternary planes might end up:

**A** — clear Pareto improvement over 4-bit/BitNet-style baselines  
**B** — useful niche advantage but not universally superior  
**C** — works, but ordinary quantization is equally good or simpler  
**D** — technically functional, economically unattractive  
**F** — hypothesis disproven

And an **F is a successful research result**. It tells us to stop spending resources there.

There is another reason I strongly favor decomposition: I don't think the three components have equal scientific risk.

My current expectation would be:

```text
Residual ternary planes       HIGH research risk
Adaptive precision            MEDIUM–HIGH research risk
Recursive context / RLM       MEDIUM risk
```

The RLM/context idea could prove useful even if ternary planes fail completely. Adaptive computation could survive too, perhaps using INT4/INT8 precision levels instead of ternary residual planes. And residual ternary decomposition could potentially find a niche in hardware even if it doesn't produce the best general-purpose LLM.

That means TORUS doesn't have to succeed or fail as a single binary proposition.

You could ultimately discover something like:

```text
Track A: C
Residual ternary planes don't beat modern 4-bit inference.

Track B: A
Dynamic precision allocation is extremely effective.

Track C: A
Recursive context dramatically improves small-model usefulness.
```

At that point TORUS evolves into something different:

\[
\text{TORUS}
=
\text{adaptive precision}
+
\text{recursive context}
\]

perhaps using ordinary INT4/INT8 underneath.

Or we might find:

```text
A = A
B = C
C = A
```

Then residual ternary itself is the breakthrough, but sophisticated gating isn't required.

That is much healthier than designing the answer before running the experiment.

**So I would change TORUS from an architecture-development project into a three-track experimental program for the next milestone.** Freeze major features. Establish baselines. Define pass/fail criteria *before* tests. Run each hypothesis independently. Only recombine components that individually earn their way back into the system.

The very first experiment I would run is the **residual-plane Pareto test**, followed immediately by the **oracle-gate test**. Those two experiments can tell us fairly quickly whether the central ternary/adaptive-compute branch of TORUS deserves deeper training research at all.

---

## User

Can you create a feedback folder with all this feedback information or knowledge and save that into the GitHub repository, where we can pull that to our testing harness and reconfigure the architecture of our project on the lines of the feedback you've given us?

## Assistant

Yes. I prepared the structure for a TORUS feedback package, but the GitHub connection currently has **read access without permission to create branches/write to this repository**. GitHub returned `403 Resource not accessible by integration` when I attempted to create `feedback/three-track-rearchitecture`, so I did **not** modify `main` or pretend the files were saved.

The package I recommend committing is:

```text
feedback/
├── README.md
├── 01-ARCHITECTURE-REVIEW.md
├── 02-THREE-TRACK-RESEARCH-PLAN.md
├── 03-TRACK-A-RESIDUAL-TERNARY.md
├── 04-TRACK-B-ADAPTIVE-PRECISION.md
├── 05-TRACK-C-RECURSIVE-CONTEXT.md
├── 06-GRADING-RUBRIC.md
├── 07-RECONFIGURATION-PLAN.md
└── HARNESS-INSTRUCTIONS.md
```

The crucial `HARNESS-INSTRUCTIONS.md` would tell your local testing harness to **freeze feature expansion, separate TORUS into three independently testable hypotheses, establish conventional baselines, run Track A first, perform an oracle-gate experiment before training Track B, test Track C with a conventional competent LLM, assign predetermined A–F grades, and only recombine components that pass their individual acceptance criteria.**

I would also preserve the current TORUS implementation rather than immediately tearing it apart. The feedback should direct the harness to reorganize the *experimental architecture* first, leaving the existing implementation as the baseline/reference. That gives us an auditable before/after comparison and prevents a rearchitecture from destroying useful work.

Once the GitHub connection has write permission, I can create this on a **feature branch and draft PR** rather than writing directly to `main`, which is the safer way to let your harness inspect the proposed research reconfiguration before merging it.

---

## User

Can you also review all the work that has been done so far and the testing results that we have encountered, and also give us some feedback on where we can improve our project?

## Assistant

Yes. After reviewing the repository history, evaluation report, changelog, test progression, and the experiments you have run, I think TORUS has produced **more useful knowledge than the negative headline result suggests**—but the project has been mixing three different kinds of success:

- software correctness,
- systems/engineering efficiency,
- and model-learning quality.

Those need to be graded separately.

The central finding so far is that **the software platform is progressing well, while the primary ternary-learning hypothesis has not yet been demonstrated**.

## What the work so far has actually established

The engineering work is substantial. TORUS has moved from reference ternary quantization through residual planes, HF model adapters, training plumbing, multi-expert routing, persistent context, indexed retrieval, sandboxing, CPU/CUDA kernels, distillation, checkpointing, and an actual lm-eval pipeline. The changelog records repeated test suites passing on both the GB10 development machine and Legion, as well as real-model Hugging Face integration. 

Some of the non-training results are genuinely encouraging.

The persistent-context index produced a reported warm lookup of about **0.041 ms versus 49.8 ms for linear scanning** on the 2,000-chunk demonstration—roughly a 1,200× retrieval improvement for that workload. 

The multi-expert experiment reported approximately **17% fewer plane activations** than an always-four-plane baseline, although this was using the project's heuristic/random routing environment rather than a trained production router. 

The security work is also real engineering progress: the RLM path now has AST restrictions, restricted builtins, resource caps, and a sandbox-compatible Prime Agent execution path. 

So I would not characterize TORUS as having “gone nowhere.”

What has gone poorly is much narrower:

> **Turning an already-trained conventional transformer into a useful low-plane ternary transformer with the current TORUS quantization/training recipe.**

And that distinction is important.

---

# The model-quality evidence is currently negative

The August 16 experiment is the strongest piece of evidence because it finally compared against a real FP16 model with an actual evaluation harness.

The OLMo-1B baseline achieved:

| Task | FP16 | TORUS PTQ / trained result |
|---|---:|---:|
| ARC Easy accuracy | 0.6073 | ~0.258 |
| LAMBADA accuracy | 0.6095 | ~0.001 |
| WikiText perplexity | 13.09 | 465,097–759,750 |

The aggregate result was only about **14% of baseline**, against TORUS's ≥90% acceptance criterion. 

Even more significant:

**The trained one-plane model was essentially identical to plain post-training ternary quantization.**

That means the training being performed at that point was not actually recovering useful capability.

And adding the second residual plane did not rescue it—it slightly worsened the measured aggregate result. 

That is important scientific evidence.

---

# But the failure uncovered an architectural issue rather than merely “bad training”

You discovered that a representative layer went from approximately:

\[
\|W\| = 19.9
\]

to:

\[
\|\hat W_T\| = 11.95
\]

after ternarization—roughly a 40% norm loss. 

The repository subsequently added norm calibration, and the latest commit reports that WikiText PTQ perplexity improved from roughly **465,097 to 89,557**.

That's about a 5× recovery.

But this is the important part:

**89,557 perplexity compared with an FP16 perplexity of ~13 is still catastrophic.**

So norm preservation solved *one failure mechanism*. It did not validate the representation.

That changes my recommendation from the earlier eval report, which reasonably suggested “calibrate, train longer, use real data.”

Now that calibration has been tried and remains extremely far away, I would **not simply keep increasing training from 1,000 → 10,000 → 100,000 steps and hope it eventually works**.

You need a more fundamental diagnostic phase first.

---

# I would grade TORUS today like this

### Track A — Residual ternary representation: **D+ / unresolved**

Not an F yet, because you discovered a genuine calibration defect and have only begun testing around it.

But currently:

\[
T_1
\]

is catastrophically degrading a pretrained transformer, and:

\[
T_1 + T_2
\]

has not demonstrated useful recovery.

That means this track has **not earned permission for expensive large-scale training yet**.

---

### Track B — Adaptive computation / gating: **C+ / promising but unproven**

The architecture works.

Telemetry works.

Plane-count policies work.

The MoE demonstration produced measurable activation savings.

But there is a logical dependency:

> gating only matters if the additional plane actually provides valuable additional capability.

If:

\[
Q(T_1+T_2)\approx Q(T_1)
\]

then deciding intelligently when to invoke \(T_2\) is irrelevant.

So Track B should temporarily wait on one very specific Track-A question:

\[
\Delta Q = Q(T_1+T_2)-Q(T_1)
\]

Is that number meaningfully positive?

---

### Track C — Recursive context/RLM: **B / strongest current subsystem**

This is the part I would be most reluctant to abandon.

Persistent context works.

Indexing works.

The REPL abstraction works.

Sandboxing works.

PrimeAgentLoop works against its model-callable abstraction.

Disk-backed context works.

And the context system is essentially **independent of ternary inference**.

The architecture docs explicitly treat `RecursiveContext`, `PersistentContext`, `ContextREPL`, and `PrimeAgentLoop` as composable primitives. 

What is missing is a serious end-to-end benchmark against conventional long-context/RAG approaches.

So this is not yet an A.

But it is currently the TORUS component with the clearest path toward becoming independently useful.

---

# The most important improvement: change the experimental question

I think TORUS has been asking too large a question:

> Can we make a ternary local LLM with residual precision, adaptive routing, recursive context, MoE, hardware kernels, and training?

That's six research questions entangled together.

The next experiment should instead ask:

> **Can a single transformer weight matrix—or a single transformer block—be represented by residual ternary planes without destroying its behavior?**

Not the whole model.

Start much lower.

For every major layer category:

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
lm_head
```

measure independently:

\[
W
\]

versus:

\[
T_1
\]

versus:

\[
T_1+T_2
\]

versus:

\[
T_1+T_2+T_3
\]

and compare actual layer outputs on real activation distributions.

That will tell you **where ternarization fails**.

---

# I strongly suspect uniform ternarization is part of the problem

Right now TORUS quantizes 114 linear layers in the OLMo-1B experiment. 

You're essentially assuming:

\[
\text{all transformer matrices have equal tolerance to ternary approximation}
\]

That is a very strong assumption.

It may turn out that:

```text
q_proj         tolerates T1
k_proj         tolerates T1
v_proj         needs T1+T2
o_proj         needs T1+T2
up_proj        tolerates T1
gate_proj      highly sensitive
down_proj      needs higher precision
lm_head        must remain FP/INT8
```

If that's what the data says, that's not a failure of TORUS.

It may reveal the better architecture:

\[
\boxed{
\text{precision should vary by layer}
}
\]

before it varies by token.

That would actually strengthen the adaptive-computation thesis.

---

# Add an “oracle residual” experiment before training anything else

This is probably the single most valuable experiment I can suggest.

Take the original FP weight:

\[
W
\]

and first construct:

\[
T_1.
\]

Now calculate the exact residual:

\[
R=W-T_1.
\]

Before ternarizing \(R\), temporarily use the **full-precision residual**:

\[
\hat W=T_1+\alpha R.
\]

Sweep:

\[
\alpha=0,\;0.25,\;0.5,\;0.75,\;1.
\]

If model quality quickly recovers as residual information is restored, then the primary ternary plane is not inherently fatal—you have shown that a hierarchical representation can work.

Then test:

\[
T_1+T_2
\]

against the oracle residual.

That tells us whether **the second ternary approximation** is the problem.

This cleanly separates:

```text
Does hierarchical precision work?
             │
             ├── YES
             │     ↓
             │ Is ternary residual encoding good enough?
             │
             └── NO
                   ↓
             rethink the entire representation
```

That's far more informative than another thousand training steps.

---

# Run a layer-by-layer sensitivity ablation

This would also be extremely revealing.

Start with FP16 OLMo.

Ternarize only:

```text
Layer 0
```

evaluate.

Restore it.

Then ternarize:

```text
Layer 1
```

evaluate.

And so forth.

Then test categories:

```text
attention only
MLP only
q/k only
v/o only
up/down only
gate only
```

You should produce a heat map conceptually like:

```text
                   sensitivity
q_proj     ███
k_proj     ██
v_proj     █████
o_proj     ██████
gate_proj  ████████
up_proj    ███
down_proj  ███████
```

That will tell you where your error is entering the network.

Right now “OLMo collapses when 114 linears are ternarized” doesn't tell us whether 100 layers are acceptable and 14 destroy everything.

---

# Do not use the ≥90% FP16 criterion as the only score

Keep it as a final target.

But for research iteration, use a Pareto score.

TORUS's thesis is fundamentally:

\[
\text{quality}
\leftrightarrow
\text{energy}
\leftrightarrow
\text{memory}
\leftrightarrow
\text{latency}
\]

So the important comparison is not simply:

```text
TORUS / FP16
```

It should be:

```text
FP16
BF16
INT8
INT4
a strong modern 4-bit baseline
T1
T1+T2
T1+T2+T3
adaptive TORUS
```

Then plot quality against:

- bytes/weight,
- memory bandwidth,
- operations/token,
- tokens/sec,
- and eventually joules/token.

A system reaching 88% of FP16 at a genuinely dramatic energy improvement may be much more interesting than one reaching 92% while consuming almost the same resources as INT4.

---

# There is also a terminology issue worth correcting

TORUS repeatedly calls the primary representation “1.58-bit.”

But the implementation actually stores ternary codes using **2-bit packing**, plus group scale overhead. The architecture document estimates about **2.125 bits/weight** with group size 128. 

Those are different concepts.

\[
\log_2(3)\approx1.585
\]

is the theoretical information content of three equiprobable states.

But:

```text
actual TORUS representation ≠ 1.58 physical bits/weight
```

at present.

I'd change the project language to:

> **ternary weights** or **2-bit packed ternary weights**

and reserve “1.58-bit” for theoretical/algorithmic comparisons where appropriate.

That will make eventual efficiency claims much harder to challenge.

---

# Your training research also needs a clean control experiment

The repository has gone through several trainer bugs and plumbing fixes:

- residual weights not actually being optimized,
- finite-difference probes affecting only limited coordinates,
- autograd detection failing because of bound methods,
- zero initialization trapping residual weights in the quantizer dead zone,
- residual learning rates causing instability,
- teacher/student loss wiring problems.

These were legitimately fixed, and finding them is good engineering. The changelog documents these corrections and their resulting loss changes. 

But that means early training curves should **not be treated as scientific evidence about the ternary hypothesis**.

I would formally mark everything before the corrected autograd/training pipeline as:

```text
ENGINEERING VALIDATION
not
MODEL ARCHITECTURE VALIDATION
```

Then establish a clean experimental lineage starting now.

---

# I would create a formal experiment registry

Every experiment should produce something like:

```text
EXP-A-001
Hypothesis:
A primary ternary plane preserves ≥X% of layer-output cosine similarity.

Model:
OLMo-1B

Layer:
model.layers.7.mlp.down_proj

Baseline:
FP16

Variants:
T1
T1+T2
T1+T2+T3

Dataset:
WikiText calibration samples

Metrics:
relative L2 output error
cosine similarity
KL divergence
weight reconstruction error
storage
kernel operations

Pass threshold:
defined BEFORE run

Result:
PASS / FAIL

Conclusion:
...
```

This seems bureaucratic, but it will dramatically improve TORUS.

Right now commits sometimes mix:

```text
new feature
bug fix
benchmark
new training strategy
architecture claim
```

into one iteration.

You want scientific traceability.

---

# And freeze most new features temporarily

TORUS has gone from Phase 1 through things labeled Phase 7, Phase 8, Phase 9, etc. extraordinarily quickly.

The implementation sophistication is starting to outrun validation.

I'd freeze:

- more context features,
- more MoE features,
- more persistence features,
- custom hardware speculation,
- additional serving integrations.

Not because they're bad ideas.

Because **the project's limiting factor is no longer implementation capacity. It is experimental certainty.**

For the next period:

\[
80\% \text{ experiments}
\]

\[
20\% \text{ supporting code}
\]

would be healthier than the reverse.

---

# My proposed TORUS reorganization

I would turn the repository conceptually into:

```text
TORUS Research Program
│
├── Track A — Precision
│   ├── A1 single-plane viability
│   ├── A2 residual recovery
│   ├── A3 layer sensitivity
│   ├── A4 calibration
│   └── A5 real QAT
│
├── Track B — Conditional Compute
│   ├── B1 oracle gating
│   ├── B2 heuristic gating
│   ├── B3 learned gating
│   └── B4 energy validation
│
└── Track C — Recursive Context
    ├── C1 retrieval accuracy
    ├── C2 long-document QA
    ├── C3 RLM vs RAG
    ├── C4 model-size scaling
    └── C5 compute/token economics
```

Then the eventual integrated TORUS architecture becomes:

\[
\boxed{
\text{only the components that independently pass}
}
\]

---

# The next five experiments I would run

1. **Layer sensitivity test.** Ternarize one layer/module at a time and measure model degradation.

2. **Oracle residual test.** Use the exact FP residual after \(T_1\) and measure how much quality is recoverable before quantizing that residual.

3. **Residual-plane reconstruction curve.** Compare \(T_1\), \(T_1+T_2\), \(T_1+T_2+T_3\), and \(T_1+T_2+T_3+T_4\) on real pretrained weights and real activation outputs.

4. **Strong quantization baselines.** Run INT8 and INT4 using the same model/evaluation set. Without these, we don't know whether TORUS is solving a problem existing quantization already solves better.

5. **RLM independent benchmark.** Use an ordinary strong small model and compare TORUS's recursive context engine against fixed-context prompting and a straightforward RAG baseline on long-document QA.

Those five tests would tell us far more about TORUS's future than another architectural phase.

## My overall conclusion

I would not shut TORUS down.

But I **would stop treating the current integrated architecture as the hypothesis**.

There are at least three independent ideas inside it, and the existing evidence is telling us something valuable:

\[
\boxed{
\text{the integration was built faster than the individual hypotheses were validated}
}
\]

The engineering has given you an excellent experimental platform. Now use that platform to aggressively try to **disprove each component**.

If residual ternary earns a D or F, replace it.

If adaptive precision earns an A using INT4/INT8 rather than ternary, keep adaptive precision.

If RLM earns an A, spin it into a first-class TORUS subsystem.

And if residual ternary eventually earns an A after the layer-level/oracle experiments reveal the correct representation, then you will have much stronger evidence that it is actually innovative rather than simply a complicated quantizer.

That is the direction I would put into the `feedback/` package for your local harness: **TORUS Research Reset — isolate, falsify, grade, then recombine.**

---

## User

Can you save this last output to our GitHub repository so I can feed it back into our testing ai Harness?  I would like the harness to steer our project as described above.

## Assistant

I tried to save the full guidance as:

`feedback/TORUS-RESEARCH-RESET.md`

with the harness directive **“Isolate. Falsify. Grade. Recombine.”** and all of the research grades, next experiments, layer-sensitivity plan, oracle-residual test, oracle gating, RLM benchmarking, grading rubric, Pareto scorecard, terminology correction, and feature-freeze guidance.

GitHub is still rejecting write operations with:

`403 Resource not accessible by integration`

even though the repository metadata reports push/admin permission. So the blocker is the GitHub integration’s write scope, not the TORUS repository itself.

The document content is prepared. Once the GitHub connector has repository-content write permission, I can place it directly at:

`feedback/TORUS-RESEARCH-RESET.md`

and ideally add the rest of the feedback package alongside it. I would then have your testing harness treat that file as the steering document for the next research milestone rather than continuing the current integrated architecture path.

---

## User

Try again

## Assistant

I retried the GitHub write path. It is still blocked at the integration level: creating the feature branch `feedback/torus-research-reset` again returned **403 “Resource not accessible by integration.”**

So the TORUS repository itself is reachable and readable, but the connected GitHub integration still cannot perform repository writes.

The intended file remains:

`feedback/TORUS-RESEARCH-RESET.md`

with the harness directive to **isolate, falsify, grade, and only then recombine** the three TORUS research tracks. Once the GitHub connection gets contents/write scope, I can save it there directly.

---

## User

create the files to download as .md files to be saved locally and then pushed to Github

## Assistant

I’ve moved this into Work mode so the Markdown package and ZIP can be created as downloadable files.

The package will include the `feedback/` folder with the architecture review, three-track research plan, individual Track A/B/C plans, grading rubric, reconfiguration plan, and harness instructions, plus a ZIP you can save locally and push into the TORUS GitHub repository.

---

## User

create a .md file to downlooad of our entire chat session.

## Assistant

I’ve handed this off to Work mode to create a downloadable Markdown transcript of the entire TORUS chat session, preserving the conversation in chronological order.

---


