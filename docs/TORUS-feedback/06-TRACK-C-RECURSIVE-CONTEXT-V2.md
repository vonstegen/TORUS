# Track C v2 --- Recursive Context

Track C remains independent of ternary success or failure.

## Core question

Can TORUS recursive/persistent/indexed context improve long-context task
accuracy and/or cost for small/local models compared with conventional
alternatives?

## Required controls

-   fixed-window prompting;
-   a straightforward RAG baseline;
-   TORUS recursive context/RLM;
-   where feasible, a larger-context model baseline.

## Metrics

-   answer accuracy / required-fact recall;
-   retrieval precision and recall;
-   context tokens admitted to the model;
-   model calls per answer;
-   wall-clock latency;
-   storage/index overhead;
-   failure modes and citation/addressability accuracy.

Do not combine Track C with experimental ternary models for the first
benchmark. Use a competent conventional model so context quality is
measured without representation confounding.
