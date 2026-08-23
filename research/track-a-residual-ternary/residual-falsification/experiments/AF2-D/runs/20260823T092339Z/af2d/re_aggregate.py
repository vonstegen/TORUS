"""AF2-D aggregate-corrected: re-classify random_t2_ternary from
trained_arms to untrained_controls based on the (corrected)
T2TernaryAdapter.is_untrained = (not self._train) fix.

The driver commit (330e8b3 / soon-to-be a follow-up) had a bug where
T2TernaryAdapter.is_untrained defaulted to False even when constructed
with train=False (the random_t2_ternary arm path). The fix sets
is_untrained in __init__. The per-seed eval data is correct
(random_t2_ternary's ppl/arc/lambada values are right); only the
audit classification was wrong.

This script regenerates the aggregate from the per-seed summaries
using the corrected classification. Saved as
aggregate_corrected.json alongside the original aggregate.json.
"""
import json
import statistics
from pathlib import Path

ROOT = Path("/tmp/TORUS/research/track-a-residual-ternary/residual-falsification/experiments/AF2-D/runs/20260823T092339Z/af2d")
ARMS_TRAINED = ["t2_ternary", "int4_residual", "int8_residual", "lora", "dense_adapter"]
ARMS_CONTROLS = ["random_t2_ternary", "random_lora"]


def _load_summaries():
    summaries = []
    for s in [1, 2, 3]:
        for arm in ARMS_TRAINED + ARMS_CONTROLS:
            p = ROOT / f"seed-{s:03d}" / arm / "eval.summary.json"
            if not p.exists():
                continue
            d = json.load(open(p))
            d["arm"] = arm
            d["seed"] = s
            # Correct classification: random_t2_ternary is untrained.
            d["is_untrained_control"] = (arm in ARMS_CONTROLS)
            summaries.append(d)
    return summaries


def _aggregate(summaries):
    out = {"trained_arms": {}, "untrained_controls": {},
           "tolerance_violations": [],
           "n_runs": len(summaries)}
    for arm in ARMS_TRAINED:
        per_seed = [s for s in summaries if s["arm"] == arm]
        per_seed = [s for s in per_seed if s.get("matched_bytes_passed", True)]
        if not per_seed:
            continue
        entry = {"n": len(per_seed),
                 "matched_bytes": [s["matched_bytes_actual"] for s in per_seed],
                 "matched_bytes_target": per_seed[0]["matched_bytes_target"],
                 "cost_vector_rows": [s["cost_vector"] for s in per_seed],
                 "tasks": {}}
        for t_name in ["wikitext", "arc_easy", "lambada_openai"]:
            vals = [s["tasks"].get(t_name, {}).get("value")
                    for s in per_seed
                    if s.get("tasks", {}).get(t_name)]
            vals = [v for v in vals if v is not None]
            if vals:
                entry["tasks"][t_name] = {
                    "n": len(vals),
                    "mean": statistics.fmean(vals),
                    "stderr": (statistics.stdev(vals) / len(vals) ** 0.5
                                if len(vals) > 1 else 0.0),
                    "values": vals,
                }
        out["trained_arms"][arm] = entry
    for arm in ARMS_CONTROLS:
        per_seed = [s for s in summaries if s["arm"] == arm]
        per_seed = [s for s in per_seed if s.get("matched_bytes_passed", True)]
        if not per_seed:
            continue
        entry = {"n": len(per_seed),
                 "matched_bytes": [s["matched_bytes_actual"] for s in per_seed],
                 "matched_bytes_target": per_seed[0]["matched_bytes_target"],
                 "cost_vector_rows": [s["cost_vector"] for s in per_seed],
                 "tasks": {}}
        for t_name in ["wikitext", "arc_easy", "lambada_openai"]:
            vals = [s["tasks"].get(t_name, {}).get("value")
                    for s in per_seed
                    if s.get("tasks", {}).get(t_name)]
            vals = [v for v in vals if v is not None]
            if vals:
                entry["tasks"][t_name] = {
                    "n": len(vals),
                    "mean": statistics.fmean(vals),
                    "stderr": (statistics.stdev(vals) / len(vals) ** 0.5
                                if len(vals) > 1 else 0.0),
                    "values": vals,
                }
        out["untrained_controls"][arm] = entry
    # Tolerance violations
    for s in summaries:
        if not s.get("matched_bytes_passed", True):
            out["tolerance_violations"].append({
                "arm": s["arm"], "seed": s["seed"],
                "actual": s["matched_bytes_actual"],
                "target": s["matched_bytes_target"],
                "delta_pct": round(
                    100 * (s["matched_bytes_actual"]
                           - s["matched_bytes_target"])
                    / s["matched_bytes_target"], 3),
            })
    # (t2 - dense) differences for the trained arms
    if "dense_adapter" in out["trained_arms"]:
        diff = {}
        for arm in ARMS_TRAINED:
            if arm == "dense_adapter" or arm not in out["trained_arms"]:
                continue
            row = {}
            for t, st in out["trained_arms"][arm].get("tasks", {}).items():
                ref = out["trained_arms"]["dense_adapter"]["tasks"].get(t)
                if not ref:
                    continue
                a = st["mean"]; b = ref["mean"]
                se_diff = (st["stderr"] ** 2 + ref["stderr"] ** 2) ** 0.5
                in_stderr = (a - b) / se_diff if se_diff > 0 else float("inf")
                row[t] = {"mean_a_minus_b": a - b,
                           "se_diff": se_diff,
                           "in_stderrs": in_stderr}
            diff[arm] = row
        out["difference_from_dense_adapter"] = diff
    # (t2 - random_t2_ternary) PASS+ bar
    if "t2_ternary" in out["trained_arms"] and "random_t2_ternary" in out["untrained_controls"]:
        diff_t2_rand = {}
        for t in ["wikitext", "arc_easy", "lambada_openai"]:
            t2_t = out["trained_arms"]["t2_ternary"]["tasks"].get(t)
            rnd_t = out["untrained_controls"]["random_t2_ternary"]["tasks"].get(t)
            if not (t2_t and rnd_t):
                continue
            a = t2_t["mean"]; b = rnd_t["mean"]
            se_diff = (t2_t["stderr"] ** 2 + rnd_t["stderr"] ** 2) ** 0.5
            in_stderr = (a - b) / se_diff if se_diff > 0 else float("inf")
            diff_t2_rand[t] = {
                "t2_mean": a, "random_mean": b,
                "mean_a_minus_b": a - b,
                "se_diff": se_diff,
                "in_stderrs": in_stderr,
            }
        out["t2_vs_random_t2"] = diff_t2_rand
    return out


if __name__ == "__main__":
    summaries = _load_summaries()
    agg = _aggregate(summaries)
    out_path = ROOT / "aggregate_corrected.json"
    out_path.write_text(json.dumps(agg, indent=2))
    print("aggregate_corrected.json written.")
    print("n_runs:", agg["n_runs"])
    print("tolerance_violations:", agg["tolerance_violations"])
    print("trained_arms:", list(agg["trained_arms"].keys()))
    print("untrained_controls:", list(agg["untrained_controls"].keys()))
    print()
    print("=== t2 vs random_t2 PASS+ bar ===")
    for t, dd in agg.get("t2_vs_random_t2", {}).items():
        print(f"  {t}: trained={dd['t2_mean']:.4f} random={dd['random_mean']:.4f} "
              f"(t2 - random)={dd['mean_a_minus_b']:+.4f} z={dd['in_stderrs']:+.3f}")