"""Pareto dominance audit.

For each (regime, arm), compute the full 6-dim cost vector (B/F/O/M/L/E)
and check which arms strictly dominate T2 (better-or-equal on every
dimension, strict on at least one).

Operates on Stage 1 + Stage 1.5 aggregate.json files.
"""
import json
from pathlib import Path

COST_DIMS = ["deployed_bytes", "training_flops",
             "inference_ops_per_token", "memory_traffic_per_token",
             "latency_per_token_titan_rtx", "energy_per_token"]
ARMS = ["t2_ternary", "int4_residual", "int8_residual",
        "lora", "dense_adapter"]
LOWER_IS_BETTER = {"deployed_bytes", "training_flops",
                   "inference_ops_per_token", "memory_traffic_per_token",
                   "latency_per_token_titan_rtx"}
# energy_per_token: lower is better but always null in our data


def lower_is_better(dim):
    return dim in LOWER_IS_BETTER


def get_cost(agg, arm):
    """Return {dim: value} for the given arm from aggregate.json."""
    arm_data = agg.get("trained_arms", {}).get(arm, {})
    rows = arm_data.get("cost_vector_rows", [])
    return rows[0] if rows else {}


def dominates(t2_cv, other_cv):
    """t2_cv is dominated by other_cv if other_cv is better-or-equal
    on every dimension and strictly better on at least one.
    Convention: lower-is-better dims use < for "better", higher-is-better
    dims (none here) use > for "better".
    """
    if not other_cv:
        return False
    any_better = False
    for dim in COST_DIMS:
        tv = t2_cv.get(dim)
        ov = other_cv.get(dim)
        if ov is None or tv is None:
            return False
        if lower_is_better(dim):
            if ov > tv:
                return False
            if ov < tv:
                any_better = True
        else:
            if ov < tv:
                return False
            if ov > tv:
                any_better = True
    return any_better


def main():
    base = Path("/home/andrew-jochl/TORUS")
    runs_root = base / "runs" / "r"
    out = {}
    for regime in ["EXP-RPM-D0", "EXP-RPM-D1", "EXP-RPM-D2", "EXP-RPM-D3",
                   "EXP-RPM-D4", "EXP-RPM-D5",
                   "EXP-RPM-D0p", "EXP-RPM-D1p", "EXP-RPM-D2p",
                   "EXP-RPM-D3p", "EXP-RPM-D4p", "EXP-RPM-D5p"]:
        runs_dir = runs_root / regime
        if not runs_dir.exists():
            continue
        ts_dirs = sorted(runs_dir.iterdir())
        if not ts_dirs:
            continue
        agg_path = ts_dirs[-1] / "af2d" / "aggregate.json"
        if not agg_path.exists():
            continue
        try:
            agg = json.loads(agg_path.read_text())
        except Exception as e:
            print(f"{regime}: parse failed: {e}")
            continue

        t2_cv = get_cost(agg, "t2_ternary")
        if not t2_cv:
            continue
        out[regime] = {"t2_cv": t2_cv, "dominators": []}
        for arm in ARMS:
            if arm == "t2_ternary":
                continue
            other = get_cost(agg, arm)
            if dominates(t2_cv, other):
                out[regime]["dominators"].append(arm)

    Path("/tmp/pareto_audit.json").write_text(json.dumps(out, indent=2))

    print("=" * 78)
    print("PARETO DOMINANCE AUDIT")
    print("=" * 78)
    for regime in sorted(out):
        info = out[regime]
        print(f"\n{regime}:")
        print(f"  T2 cost vector:")
        for dim in COST_DIMS:
            v = info["t2_cv"].get(dim)
            print(f"    {dim}: {v}")
        if info["dominators"]:
            print(f"  T2 DOMINATED by: {info['dominators']}")
        else:
            print(f"  T2 NOT dominated on the cost vector")
    print("\n[written /tmp/pareto_audit.json]")


if __name__ == "__main__":
    main()