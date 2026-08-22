# Track C — Recursive Context

## Question

Does TORUS recursive context improve answer quality, usable context scale, latency, or cost relative to conventional retrieval and long-context approaches?

## Independence rule

Use a fixed, competent model backend. Do not require ternary inference. This isolates context architecture from Tracks A and B.

## Baselines

- model with no retrieval;
- full-context prompting where the model supports it;
- conventional chunked vector/BM25/hybrid RAG;
- TORUS indexed persistent context with the same source corpus and model;
- optional recursive summarization/agentic retrieval baseline.

Equalize corpus, model, answer budget, and where possible retrieval/indexing resources.

## Workloads

Include exact retrieval, multi-hop synthesis, conflicting evidence, temporal ordering, needle variants, long-document QA, distractor robustness, and repeated-session persistence. Use both synthetic tests with known answers and realistic corpora.

## Metrics

- answer correctness with auditable scoring;
- citation/evidence precision and recall;
- retrieval recall@k and ranking quality;
- total wall-clock latency, not lookup-only latency;
- model calls, input/output tokens, storage, index-build time, and peak memory;
- robustness to corpus growth, restarts, malformed content, and prompt injection;
- sandbox escape tests and denied-operation correctness.

## Ablations

Remove indexing, persistence, recursion, sandbox, query decomposition, and caching one at a time. Identify which component creates each gain.

## Pass conditions

Track C earns an A only if it shows reproducible end-to-end advantage on at least one important workload without unacceptable regressions in correctness or security. Fast lookup alone is an engineering microbenchmark, not an end-to-end pass.
