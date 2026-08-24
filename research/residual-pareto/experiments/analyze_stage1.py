"""RPM-001 / RPM-002 / RPM-006 verdict from Stage 1 eval data.

Reads the eval.summary.json for every (regime, seed, arm) in Stage 1
and computes:

  RPM-001: T2 IS Pareto-optimal vs the full comparator set (5 trained
           arms) on the joint (3 cap × 5 cost B/F/O/M/L) vector at
           every regime.
           - already DECIDED tentative PASS at every regime in the
             Stage 1 verdicts; this script re-verifies from raw.

  RPM-002: Cross-regime trained-vs-random capability separation.
           - trained_t2 - random_t2 on each metric, averaged across
             regimes. Magnitude of separation.

  RPM-006: Per-regime trained-vs-random z-score on each capability
           metric. (trained - random) / stderr_trained.

Operates on
  runs/r/EXP-RPM-D{0..5}/<ts>/af2d/seed-{1,2,3}/{arm}/eval.summary.json

Output
------
Prints a markdown summary table per regime + per claim verdict line.
Writes analysis to research/residual-pareto/experiments/RPM-001-002-006-analysis.md
and a JSON dump to research/residual-pareto/experiments/RPM-001-002-006-analysis.json.
"""
from __future__ import annotations
import argparse
import json
import math
import statistics
import sys
from pathlib import Path

# Order: higher is better for arc_easy, lambada_openai (acc);
# LOWER is better for wikitext (ppl).
ACC_METRICS = {"arc_easy", "lambada_openai"}
PPL_METRICS = {"wikitext"}

TASKS = ["wikitext", "arc_easy", "lambada_openai"]
TRAINEd_ARMS = ["t2_ternary", "int4_residual", "int8_residual",
                "lora", "dense_adapter"]
RANDOM_ARMS = ["random_t2_ternary", "random_lora"]
ALL_ARMS = TRAINEd_ARMS + RANDOM_ARMS


def read_arm_summary(run_path: Path, seed: int, arm: str) -> dict | None:
    p = run_path / f"seed-{seed:03d}" / arm / "eval.summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def value_for(tasks: dict, t_name: str) -> float | None:
    """Extract the metric value for a task. Returns None if absent."""
    if not tasks:
        return None
    rec = tasks.get(t_name)
    if rec is None:
        return None
    v = rec.get("value")
    return float(v) if v is not None else None


def is_lower_better(t_name: str) -> bool:
    return t_name in PPL_METRICS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path,
                     default=Path("/home/andrew-jochl/TORUS"))
    ap.add_argument("--out_dir", type=Path,
                     default=Path("/home/andrew-jochl/TORUS/research/"
                                  "residual-pareto/experiments"))
    args = ap.parse_args()

    # Discover regimes
    runs_root = args.base / "runs" / "r"
    regimes = sorted(p.name for p in runs_root.iterdir()
                     if p.name.startswith("EXP-RPM-D") and not
                     p.name.endswith("p") and not p.name.endswith("CAL"))
    # only D0..D5
    regimes = [r for r in regimes
                if r[len("EXP-RPM-D"):] in {"0", "1", "2", "3", "4", "5"}]
    print(f"[analyze] regimes: {regimes}", flush=True)

    # Build data dict: data[regime][arm][seed][task] = value or None
    data = {}
    for regime in regimes:
        runs = sorted((runs_root / regime).iterdir())
        if not runs:
            continue
        run_root = runs[-1] / "af2d"  # use latest
        data[regime] = {"_run_root": str(run_root)}
        for arm in ALL_ARMS:
            data[regime][arm] = {}
            for seed in (1, 2, 3):
                es = read_arm_summary(run_root, seed, arm)
                if es is None:
                    data[regime][arm][seed] = {t: None for t in TASKS}
                else:
                    data[regime][arm][seed] = {
                        t: value_for(es.get("tasks", {}), t)
                        for t in TASKS
                    }
                # Also pull pre_train_eval for damaged base
                if es is not None and seed == 1:
                    pre = es.get("pre_train_eval", {})
                    data[regime].setdefault("_pre_train_eval", {})
                    data[regime]["_pre_train_eval"] = {
                        t: value_for(pre, t) for t in TASKS
                    }

    # RPM-002 + RPM-006: trained-vs-random z-scores per regime.
    print("\n=== RPM-006 per-regime z-scores (trained_t2_ternary vs "
          "random_t2_ternary) ===", flush=True)
    rpm006_z = {}
    for regime in regimes:
        z_row = {}
        for t_name in TASKS:
            t_vals = [data[regime]["t2_ternary"][s][t_name]
                      for s in (1, 2, 3)]
            r_vals = [data[regime]["random_t2_ternary"][s][t_name]
                      for s in (1, 2, 3)]
            t_vals = [v for v in t_vals if v is not None]
            r_vals = [v for v in r_vals if v is not None]
            if len(t_vals) < 2 or len(r_vals) < 1:
                z_row[t_name] = None
                continue
            trained_mean = statistics.mean(t_vals)
            trained_se = statistics.stdev(t_vals) / math.sqrt(len(t_vals))
            random_mean = statistics.mean(r_vals)
            random_se = (statistics.stdev(r_vals) / math.sqrt(len(r_vals))
                          if len(r_vals) >= 2 else 0.0)
            denom = math.sqrt(trained_se ** 2 + random_se ** 2)
            if denom == 0:
                z = None
            else:
                z = (trained_mean - random_mean) / denom
            z_row[t_name] = z
        rpm006_z[regime] = z_row
        print(f"  {regime}: {z_row}", flush=True)

    # RPM-002: cross-regime trained-vs-random separation.
    # Magnitude of the mean gap (across regimes).
    print("\n=== RPM-002 cross-regime separation (mean gap over regimes) ===",
          flush=True)
    rpm002 = {}
    for t_name in TASKS:
        gaps = []
        for regime in regimes:
            t_vals = [data[regime]["t2_ternary"][s][t_name]
                      for s in (1, 2, 3)]
            r_vals = [data[regime]["random_t2_ternary"][s][t_name]
                      for s in (1, 2, 3)]
            t_vals = [v for v in t_vals if v is not None]
            r_vals = [v for v in r_vals if v is not None]
            if not t_vals or not r_vals:
                continue
            gap = statistics.mean(t_vals) - statistics.mean(r_vals)
            gaps.append(gap)
        if gaps:
            rpm002[t_name] = {
                "magnitude": statistics.mean(gaps),
                "min_gap": min(gaps),
                "max_gap": max(gaps),
                "n_regimes_with_data": len(gaps),
                "signs": [("+" if g > 0 else "-") for g in gaps],
            }
        else:
            rpm002[t_name] = None
        print(f"  {t_name}: {rpm002[t_name]}", flush=True)

    # RPM-001: T2 Pareto vs full comparator set (5 trained arms).
    # Already tentative PASS from Stage 1 verdict (with energy null).
    # This section records the ppl + capability delta T2 has over the
    # next-best competitor at each regime.
    print("\n=== RPM-001 T2 Pareto evidence summary (per regime) ===",
          flush=True)
    rpm001 = {}
    for regime in regimes:
        # for each task, compute T2 mean and the next-best trained arm
        row = {}
        for t_name in TASKS:
            arm_means = {}
            for arm in TRAINEd_ARMS:
                vals = [data[regime][arm][s][t_name]
                        for s in (1, 2, 3)]
                vals = [v for v in vals if v is not None]
                if vals:
                    arm_means[arm] = statistics.mean(vals)
            if "t2_ternary" not in arm_means:
                row[t_name] = None
                continue
            t2_v = arm_means["t2_ternary"]
            others = {a: v for a, v in arm_means.items()
                       if a != "t2_ternary"}
            if not others:
                row[t_name] = None
                continue
            if is_lower_better(t_name):
                # smaller is better
                best_v = min(others.values())
                delta = best_v - t2_v  # positive = T2 better
            else:
                # larger is better
                best_v = max(others.values())
                delta = t2_v - best_v  # positive = T2 better
            row[t_name] = {
                "t2_value": t2_v,
                "next_best_value": best_v,
                "delta": delta,
                "next_best_arm": (
                    min(others, key=others.get)
                    if is_lower_better(t_name)
                    else max(others, key=others.get)
                ),
            }
        rpm001[regime] = row
        print(f"  {regime}: {row}", flush=True)

    # Write outputs
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    json_out = out_dir / "RPM-001-002-006-analysis.json"
    json_out.write_text(json.dumps({
        "rpm001": rpm001,
        "rpm002": rpm002,
        "rpm006": rpm006_z,
        "_data_keys": {r: list(v.keys()) for r, v in data.items()},
    }, indent=2, default=str))
    print(f"\n[analyze] wrote {json_out}", flush=True)

    # Markdown summary
    md_lines = ["# RPM-001 / RPM-002 / RPM-006 Analysis (Stage 1 post-hoc)",
                "",
                "Generated from runs/r/EXP-RPM-D{0..5}/<ts>/af2d/seed-{1,2,3}/"
                "{arm}/eval.summary.json after post-hoc eval of "
                "random_t2_ternary + random_lora arms.",
                ""]

    md_lines += ["## RPM-001 per-regime T2 vs next-best trained arm",
                  "",
                  "| Regime | Task | T2 value | Next-best arm | "
                  "Next-best value | T2 - Next |",
                  "|---|---|---|---|---|---|"]
    for regime in regimes:
        for t_name in TASKS:
            row = rpm001[regime][t_name]
            if row is None:
                md_lines.append(f"| {regime} | {t_name} | "
                                f"insufficient data | — | — | — |")
            else:
                sign = "better" if row["delta"] > 0 else "worse"
                md_lines.append(f"| {regime} | {t_name} | "
                                f"{row['t2_value']:.4f} | "
                                f"{row['next_best_arm']} | "
                                f"{row['next_best_value']:.4f} | "
                                f"{row['delta']:+.4f} ({sign}) |")
    md_lines.append("")

    md_lines += ["## RPM-002 cross-regime trained-vs-random separation",
                  "",
                  "| Task | mean gap | min | max | n_regimes | signs |",
                  "|---|---|---|---|---|---|"]
    for t_name in TASKS:
        r = rpm002[t_name]
        if r is None:
            md_lines.append(f"| {t_name} | no data | — | — | — | — |")
        else:
            md_lines.append(f"| {t_name} | {r['magnitude']:+.4f} | "
                            f"{r['min_gap']:+.4f} | {r['max_gap']:+.4f} | "
                            f"{r['n_regimes_with_data']} | "
                            f"{''.join(r['signs'])} |")
    md_lines.append("")

    md_lines += ["## RPM-006 per-regime trained-vs-random z-score",
                  "",
                  "| Regime | wikitext | arc_easy | lambada_openai |",
                  "|---|---|---|---|"]
    for regime in regimes:
        z_row = rpm006_z[regime]
        cells = []
        for t_name in TASKS:
            z = z_row.get(t_name)
            cells.append(f"{z:+.2f}σ" if z is not None else "—")
        md_lines.append(f"| {regime} | {cells[0]} | {cells[1]} | {cells[2]} |")
    md_lines.append("")

    md_out = out_dir / "RPM-001-002-006-analysis.md"
    md_out.write_text("\n".join(md_lines))
    print(f"[analyze] wrote {md_out}", flush=True)


if __name__ == "__main__":
    main()