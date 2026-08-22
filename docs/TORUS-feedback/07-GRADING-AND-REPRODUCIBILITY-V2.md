# TORUS v2 --- Grading and Reproducibility Standard

## Grades

-   **A:** clear, reproduced Pareto improvement or strong independent
    capability.
-   **B:** meaningful reproduced advantage, but important limitations
    remain.
-   **C:** technically works; advantage is weak, narrow or not yet
    competitive.
-   **D:** functional but economically/scientifically unattractive under
    current evidence.
-   **F:** hypothesis falsified under the preregistered test.
-   **INVALID:** provenance, contamination, evaluation or implementation
    uncertainty prevents a scientific verdict.

An F is a useful research result. INVALID is not a failure and must be
rerun.

## Mandatory run provenance

Every experiment gets a unique immutable run directory, for example:

`runs/<track>/<experiment_id>/<timestamp-or-uuid>/`

Record at minimum: - experiment ID and hypothesis; - git SHA and
branch; - complete config; - model/checkpoint identifiers and hashes; -
dataset and sample IDs; - random seeds; - hostname/hardware/software
versions; - start/end timestamps; - training/evaluation commands; - raw
metrics and summarized metrics; - artifact/checkpoint hashes.

## Contamination rule

-   No shared mutable `/tmp` checkpoint/result names across experiments.
-   No concurrent agents may write to the same artifact namespace.
-   If provenance is uncertain, mark the run **INVALID**.
-   Important PASS/FAIL results require a clean reproduction before
    architectural action.

## Evidence categories

Report separately: 1. software correctness; 2. representation/capacity
evidence; 3. downstream model quality; 4. systems efficiency; 5.
end-to-end product usefulness.

Do not use one category as a proxy for another.
