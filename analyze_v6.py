"""Compute LRN and TSP deltas across the EXP-RPM-AF2D-SEVERITY sweep.

For each preregistered threshold:
  - LRN: trained T2 - random T2 (per capability metric, 3-seed mean, z-score)
  - TSP: trained T2 - random LoRA (per capability metric, 3-seed mean, z-score)

Output: per-threshold tables plus the LRN_active and TSP_active boundary
verdicts per the preregistered criterion.
"""
import json, statistics, os
from pathlib import Path

BASE = Path("/home/andrew-jochl/TORUS/runs/r/EXP-RPM-AF2D-SEVERITY")
THRESHOLDS = [0.6, 0.7, 0.8, 0.9, 1.0]
TASKS = ["wikitext", "arc_easy", "lambada_openai"]


def load_per_arm(arm_kind):
    """Load per-threshold, per-arm summary for the given arm kind
    ('trained_arms' or 'untrained_controls')."""
    out = {}
    for t in THRESHOLDS:
        d = BASE / f"threshold-{t}"
        candidates = [x for x in d.iterdir() if x.name.startswith("2026") and not x.name.endswith("-base")]
        if not candidates:
            continue
        ts_dir = sorted(candidates)[0]
        p = ts_dir / "aggregate.json"
        if not p.exists():
            continue
        agg = json.load(open(p))
        out[t] = agg.get(arm_kind, {})
    return out


def stats(vals):
    if not vals or len(vals) < 2:
        return (None, None)
    return (statistics.fmean(vals), statistics.stdev(vals) / len(vals) ** 0.5)


def compare_z(arm1_vals, arm2_vals):
    """Pooled z-score for two arms. arm1 - arm2 / pooled_stderr."""
    if not arm1_vals or not arm2_vals or len(arm1_vals) < 1 or len(arm2_vals) < 1:
        return None, None, None
    m1 = statistics.fmean(arm1_vals)
    m2 = statistics.fmean(arm2_vals)
    all_vals = arm1_vals + arm2_vals
    if len(all_vals) > 1:
        pooled = statistics.stdev(all_vals) / (len(all_vals) / 2) ** 0.5
    else:
        pooled = 0.0
    if pooled == 0:
        return m1, m2, 0.0
    return m1, m2, (m1 - m2) / pooled


# Load per-threshold aggregated data.
trained = load_per_arm("trained_arms")
random = load_per_arm("untrained_controls")

# Per-threshold table.
print("=" * 100)
print(f"{'threshold':<10} {'arm':<22} {'wikitext':>14} {'arc_easy':>14} {'lambada':>14}")
print("-" * 100)
for t in THRESHOLDS:
    for arm in ["t2_ternary", "lora"]:
        if t not in trained or arm not in trained[t]:
            continue
        entry = trained[t][arm]
        line = f"{t:<10} {arm:<22}"
        for tk in TASKS:
            tk_data = entry.get("tasks", {}).get(tk, {})
            m = tk_data.get("mean", None)
            se = tk_data.get("stderr", 0)
            if m is not None:
                line += f" {m:>10.4f} ± {se:.4f}"
            else:
                line += f" {'—':>14}"
        print(line)
    for arm in ["random_t2_ternary", "random_lora"]:
        if t not in random or arm not in random[t]:
            continue
        entry = random[t][arm]
        line = f"{t:<10} {arm:<22}"
        for tk in TASKS:
            tk_data = entry.get("tasks", {}).get(tk, {})
            m = tk_data.get("mean", None)
            se = tk_data.get("stderr", 0)
            if m is not None:
                line += f" {m:>10.4f} ± {se:.4f}"
            else:
                line += f" {'—':>14}"
        print(line)
    print()

# Compute LRN and TSP deltas per threshold.
print("=" * 100)
print("LRN (trained T2 - random T2) and TSP (trained T2 - random LoRA) deltas per threshold:")
print(f"{'thr':<6} {'axis':<5} {'wikitext_Δ':>12} {'wikitext_z':>12} {'arc_Δ':>10} {'arc_z':>10} {'lam_Δ':>10} {'lam_z':>10} {'active':>8}")
print("-" * 100)
lrn_active_thresholds = []
tsp_active_thresholds = []
for t in THRESHOLDS:
    if t not in trained or "t2_ternary" not in trained[t]:
        continue
    t2 = trained[t]["t2_ternary"]
    rt2 = random[t].get("random_t2_ternary", {})
    rl = random[t].get("random_lora", {})

    # LRN
    lrn_zs = []
    lrn_deltas = {}
    lrn_neg_zs = []
    for tk in TASKS:
        t2_vals = t2.get("tasks", {}).get(tk, {}).get("values", [])
        rt2_vals = rt2.get("tasks", {}).get(tk, {}).get("values", [])
        if t2_vals and rt2_vals:
            m1, m2, z = compare_z(t2_vals, rt2_vals)
            # For wikitext, "T2 wins" means lower ppl (negative Δ)
            if tk == "wikitext":
                z_sign = -z  # negate so positive = T2 wins
                delta = -(m1 - m2)
            else:
                z_sign = z
                delta = m1 - m2
            lrn_zs.append(z_sign)
            lrn_deltas[tk] = (delta, z_sign)
            if z_sign < -2.0:
                lrn_neg_zs.append(tk)
    lrn_active = (sum(1 for z in lrn_zs if z >= 2.0) >= 2 and
                  all(z >= -2.0 for z in lrn_zs))

    # TSP
    tsp_zs = []
    tsp_deltas = {}
    for tk in TASKS:
        t2_vals = t2.get("tasks", {}).get(tk, {}).get("values", [])
        rl_vals = rl.get("tasks", {}).get(tk, {}).get("values", [])
        if t2_vals and rl_vals:
            m1, m2, z = compare_z(t2_vals, rl_vals)
            if tk == "wikitext":
                z_sign = -z
                delta = -(m1 - m2)
            else:
                z_sign = z
                delta = m1 - m2
            tsp_zs.append(z_sign)
            tsp_deltas[tk] = (delta, z_sign)
    tsp_active = (sum(1 for z in tsp_zs if z >= 2.0) >= 2 and
                  all(z >= -2.0 for z in tsp_zs))

    line = f"{t:<6} {'LRN':<5}"
    for tk in TASKS:
        if tk in lrn_deltas:
            delta, z = lrn_deltas[tk]
            line += f" {delta:>+10.4f}  {z:>+9.2f}σ"
        else:
            line += f" {'—':>10}  {'—':>10}"
    line += f" {'YES' if lrn_active else 'NO':>8}"
    print(line)
    if lrn_active:
        lrn_active_thresholds.append(t)

    line = f"{t:<6} {'TSP':<5}"
    for tk in TASKS:
        if tk in tsp_deltas:
            delta, z = tsp_deltas[tk]
            line += f" {delta:>+10.4f}  {z:>+9.2f}σ"
        else:
            line += f" {'—':>10}  {'—':>10}"
    line += f" {'YES' if tsp_active else 'NO':>8}"
    print(line)
    if tsp_active:
        tsp_active_thresholds.append(t)

print()
print("=" * 100)
print("LRN operating band (preregistered criterion):")
if lrn_active_thresholds:
    print(f"  Band: thresholds {min(lrn_active_thresholds)} to {max(lrn_active_thresholds)}")
    print(f"  All qualifying thresholds: {lrn_active_thresholds}")
else:
    print(f"  NO thresholds satisfy LRN_active criterion.")
print()
print("TSP operating band (preregistered criterion):")
if tsp_active_thresholds:
    print(f"  Band: thresholds {min(tsp_active_thresholds)} to {max(tsp_active_thresholds)}")
    print(f"  All qualifying thresholds: {tsp_active_thresholds}")
else:
    print(f"  NO thresholds satisfy TSP_active criterion.")
