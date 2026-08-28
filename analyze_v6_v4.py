"""Compute LRN/TSP with proper stderr handling.

The random arm values are deterministic across seeds (zero stderr by
construction — random_t2/random_lora have fixed random seeds; lm-eval-
harness is deterministic; damage is deterministic). Only trained arms
have seed-variance (from SGD noise).

The proper z-score for "trained T2 vs random T2" is:
  z = (mean_t2 - mean_rand_t2) / stderr_t2
where stderr_t2 = stdev(t2_values) / sqrt(N_seeds).

Using the pooled stderr (which includes the zero-variance random arm
contribution) artificially inflates z-scores. The trained-only stderr
is the correct measurement of "how much does training T2 vary relative
to a fixed-baseline random T2?"

This is the same convention used by Stage 1 / 1.5 verdicts.
"""
import json, statistics, os
from pathlib import Path

BASE = Path("/home/andrew-jochl/TORUS/runs/r/EXP-RPM-AF2D-SEVERITY")
THRESHOLDS = [0.6, 0.7, 0.8, 0.9, 1.0]
SEEDS = ["001", "002", "003"]
ARMS = ["t2_ternary", "lora", "random_t2_ternary", "random_lora"]


def extract_values(eval_summary):
    tasks = eval_summary.get("tasks") or {}
    out = {"wikitext": [], "arc_easy": [], "lambada_openai": []}
    if not tasks:
        return out
    sample = next(iter(tasks.values()))
    if isinstance(sample, dict) and "value" in sample:
        for tk in ["wikitext", "arc_easy", "lambada_openai"]:
            if tk in tasks and "value" in tasks[tk]:
                out[tk].append(tasks[tk]["value"])
        return out
    for k, v in tasks.items():
        if not isinstance(v, (int, float)):
            continue
        kl = k.lower()
        if "wikitext" in kl and "perplexity" in kl and "stderr" not in kl:
            out["wikitext"].append(v)
        elif "arc_easy" in kl and "acc" in kl and "stderr" not in kl:
            out["arc_easy"].append(v)
        elif "lambada_openai" in kl and "acc" in kl and "stderr" not in kl:
            out["lambada_openai"].append(v)
    return out


def load_arm_data(arm_name):
    out = {}
    for t in THRESHOLDS:
        thr_dir = BASE / f"threshold-{t}"
        ts_dirs = sorted([p for p in thr_dir.iterdir()
                          if p.is_dir() and p.name.startswith("2026") and not p.name.endswith("-base")])
        if not ts_dirs:
            continue
        ts = ts_dirs[0]
        per_seed = {"wikitext": [], "arc_easy": [], "lambada_openai": []}
        for s in SEEDS:
            f = ts / f"seed-{s}" / arm_name / "eval.summary.json"
            if not f.exists():
                continue
            d = json.load(open(f))
            vals = extract_values(d)
            for k in per_seed:
                if vals[k]:
                    per_seed[k].append(vals[k][0])
        if any(per_seed[k] for k in per_seed):
            out[t] = per_seed
    return out


def trained_only_stderr(arm_vals):
    """stderr using only the trained arm's variance (the random arm
    is deterministic across seeds by construction)."""
    if len(arm_vals) < 2:
        return 0.0
    return statistics.stdev(arm_vals) / len(arm_vals) ** 0.5


def compare_z_trained_only(trained_vals, fixed_vals):
    """z = (mean_trained - mean_fixed) / stderr_trained.
    fixed_vals is treated as a deterministic baseline (random arm)."""
    if not trained_vals or not fixed_vals:
        return None, None, None
    m_t = statistics.fmean(trained_vals)
    m_f = statistics.fmean(fixed_vals)
    se = trained_only_stderr(trained_vals)
    if se == 0:
        return m_t, m_f, 0.0
    return m_t, m_f, (m_t - m_f) / se


arms_data = {a: load_arm_data(a) for a in ARMS}

# Per-threshold table.
print("=" * 110)
print(f"{'threshold':<10} {'arm':<22} {'wikitext_ppl':>20} {'arc_easy_acc':>20} {'lambada_acc':>20}")
print("-" * 110)
for t in THRESHOLDS:
    for arm in ARMS:
        if t not in arms_data[arm]:
            continue
        td = arms_data[arm][t]
        line = f"{t:<10} {arm:<22}"
        for tk in ["wikitext", "arc_easy", "lambada_openai"]:
            vals = td[tk]
            if vals:
                m = statistics.fmean(vals)
                se = statistics.stdev(vals) / len(vals) ** 0.5 if len(vals) > 1 else 0.0
                line += f" {m:>10.4f} ± {se:.4f} (n={len(vals)})"
            else:
                line += f" {'—':>20}"
        print(line)
    print()

# LRN/TSP deltas with trained-only stderr (proper convention).
print("=" * 110)
print("LRN (trained T2 vs random T2) and TSP (trained T2 vs random LoRA) deltas per threshold")
print("ERROR BARS USE TRAINED-ARM STERR ONLY (random arms deterministic across seeds):")
print(f"{'thr':<6} {'axis':<5} {'wikitext_Δ':>14} {'wikitext_z':>12} {'arc_Δ':>10} {'arc_z':>10} {'lam_Δ':>10} {'lam_z':>10} {'active':>8}")
print("-" * 110)
lrn_active_thresholds = []
tsp_active_thresholds = []
for t in THRESHOLDS:
    if t not in arms_data["t2_ternary"]:
        continue
    t2d = arms_data["t2_ternary"][t]
    rt2d = arms_data.get("random_t2_ternary", {}).get(t, {})
    rld = arms_data.get("random_lora", {}).get(t, {})

    lrn_zs = []
    lrn_deltas = {}
    for tk in ["wikitext", "arc_easy", "lambada_openai"]:
        if tk in t2d and tk in rt2d:
            m_t, m_f, z = compare_z_trained_only(t2d[tk], rt2d[tk])
            if tk == "wikitext":
                z_sign = -z
                delta = -(m_t - m_f)
            else:
                z_sign = z
                delta = m_t - m_f
            lrn_zs.append(z_sign)
            lrn_deltas[tk] = (delta, z_sign)
    lrn_active = (sum(1 for z in lrn_zs if z >= 2.0) >= 2 and
                  all(z >= -2.0 for z in lrn_zs))

    line = f"{t:<6} {'LRN':<5}"
    for tk in ["wikitext", "arc_easy", "lambada_openai"]:
        if tk in lrn_deltas:
            delta, z = lrn_deltas[tk]
            line += f" {delta:>+10.4f}    {z:>+9.2f}σ"
        else:
            line += f" {'—':>14}    {'—':>10}"
    line += f" {'YES' if lrn_active else 'NO':>8}"
    print(line)
    if lrn_active:
        lrn_active_thresholds.append(t)

    tsp_zs = []
    tsp_deltas = {}
    for tk in ["wikitext", "arc_easy", "lambada_openai"]:
        if tk in t2d and tk in rld:
            m_t, m_f, z = compare_z_trained_only(t2d[tk], rld[tk])
            if tk == "wikitext":
                z_sign = -z
                delta = -(m_t - m_f)
            else:
                z_sign = z
                delta = m_t - m_f
            tsp_zs.append(z_sign)
            tsp_deltas[tk] = (delta, z_sign)
    tsp_active = (sum(1 for z in tsp_zs if z >= 2.0) >= 2 and
                  all(z >= -2.0 for z in tsp_zs))

    line = f"{t:<6} {'TSP':<5}"
    for tk in ["wikitext", "arc_easy", "lambada_openai"]:
        if tk in tsp_deltas:
            delta, z = tsp_deltas[tk]
            line += f" {delta:>+10.4f}    {z:>+9.2f}σ"
        else:
            line += f" {'—':>14}    {'—':>10}"
    line += f" {'YES' if tsp_active else 'NO':>8}"
    print(line)
    if tsp_active:
        tsp_active_thresholds.append(t)

print()
print("=" * 110)
print("LRN/TSP operating band (preregistered criterion: T2 ≥+2σ vs random control on ≥2 of 3 capability metrics, no ≥2σ regression):")
print(f"  LRN band: {lrn_active_thresholds if lrn_active_thresholds else 'EMPTY'}")
print(f"  TSP band: {tsp_active_thresholds if tsp_active_thresholds else 'EMPTY'}")
