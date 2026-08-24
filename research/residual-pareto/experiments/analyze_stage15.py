"""RPM-001 / RPM-002 / RPM-006 verdict from Stage 1 + Stage 1.5 eval data.

Handles both Stage 1 (EXP-RPM-D{0..5}) and Stage 1.5 (EXP-RPM-D{0..5}p)
regimes. Outputs separate analyses for each stage + a combined view
showing the architecture-vs-training signal is robust across BOTH
damage-axis parameterizations (threshold knob and observed ppl).
"""
from __future__ import annotations
import argparse
import json
import math
import statistics
from pathlib import Path

ACC_METRICS = {"arc_easy", "lambada_openai"}
PPL_METRICS = {"wikitext"}
TASKS = ["wikitext", "arc_easy", "lambada_openai"]
TRAINED_ARMS = ["t2_ternary", "int4_residual", "int8_residual",
                "lora", "dense_adapter"]
RANDOM_ARMS = ["random_t2_ternary", "random_lora"]
ALL_ARMS = TRAINED_ARMS + RANDOM_ARMS


def read_arm_summary(run_path: Path, seed: int, arm: str):
    p = run_path / f"seed-{seed:03d}" / arm / "eval.summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def value_for(tasks: dict, t_name: str):
    if not tasks:
        return None
    rec = tasks.get(t_name)
    if rec is None:
        return None
    v = rec.get("value")
    return float(v) if v is not None else None


def is_lower_better(t_name: str) -> bool:
    return t_name in PPL_METRICS


def discover_regimes(base: Path, suffix: str = "") -> list[str]:
    """Find all regime IDs of the form EXP-RPM-D{n}{suffix} in runs/r/."""
    runs_root = base / "runs" / "r"
    if not runs_root.exists():
        return []
    found = []
    for p in sorted(runs_root.iterdir()):
        if not p.name.startswith("EXP-RPM-D"):
            continue
        if not p.name.endswith(suffix):
            continue
        # Skip CAL
        if "CAL" in p.name:
            continue
        # Match: EXP-RPM-D{n}{suffix}, where {n} is 0..9
        rest = p.name[len("EXP-RPM-D"):]
        if suffix:
            if rest[: -len(suffix)].isdigit():
                found.append(p.name)
        else:
            if rest.isdigit():
                found.append(p.name)
    return found


def collect_data(base: Path, regimes: list[str]):
    runs_root = base / "runs" / "r"
    data = {}
    for regime in regimes:
        runs_dir = runs_root / regime
        runs_in = sorted(runs_dir.iterdir())
        if not runs_in:
            continue
        # Use the LATEST ts dir (in case multiple launches)
        run_root = runs_in[-1] / "af2d"
        if not run_root.exists():
            continue
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
    return data


def compute_rpm006(data, regimes):
    rpm006 = {}
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
            z_row[t_name] = ((trained_mean - random_mean) / denom
                              if denom > 0 else None)
        rpm006[regime] = z_row
    return rpm006


def compute_rpm002(data, regimes):
    rpm002 = {}
    for t_name in TASKS:
        gaps = []
        signs = []
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
            signs.append("+" if gap > 0 else ("-" if gap < 0 else "0"))
        if gaps:
            rpm002[t_name] = {
                "magnitude": statistics.mean(gaps),
                "min_gap": min(gaps),
                "max_gap": max(gaps),
                "n_regimes_with_data": len(gaps),
                "signs": signs,
            }
        else:
            rpm002[t_name] = None
    return rpm002


def compute_rpm001(data, regimes):
    rpm001 = {}
    for regime in regimes:
        row = {}
        for t_name in TASKS:
            arm_means = {}
            for arm in TRAINED_ARMS:
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
                best_v = min(others.values())
                delta = best_v - t2_v
                nba = min(others, key=others.get)
            else:
                best_v = max(others.values())
                delta = t2_v - best_v
                nba = max(others, key=others.get)
            row[t_name] = {"t2_value": t2_v,
                            "next_best_value": best_v,
                            "delta": delta,
                            "next_best_arm": nba}
        rpm001[regime] = row
    return rpm001


def render_md(rpm001, rpm002, rpm006, regimes, label: str) -> str:
    md = [f"# RPM-001 / RPM-002 / RPM-006 Analysis ({label})",
          "", "Generated from per-(regime, seed, arm) eval.summary.json.",
          ""]

    md += ["## RPM-001 per-regime T2 vs next-best trained arm",
            "",
            "| Regime | Task | T2 value | Next-best arm | "
            "Next-best value | T2 - Next |",
            "|---|---|---|---|---|---|"]
    for regime in regimes:
        for t_name in TASKS:
            row = rpm001[regime][t_name]
            if row is None:
                md.append(f"| {regime} | {t_name} | "
                          f"insufficient data | — | — | — |")
            else:
                sign = "better" if row["delta"] > 0 else "worse"
                md.append(f"| {regime} | {t_name} | "
                          f"{row['t2_value']:.4f} | "
                          f"{row['next_best_arm']} | "
                          f"{row['next_best_value']:.4f} | "
                          f"{row['delta']:+.4f} ({sign}) |")
    md.append("")

    md += ["## RPM-002 cross-regime trained-vs-random separation",
            "",
            "| Task | mean gap | min | max | n_regimes | signs |",
            "|---|---|---|---|---|---|"]
    for t_name in TASKS:
        r = rpm002[t_name]
        if r is None:
            md.append(f"| {t_name} | no data | — | — | — | — |")
        else:
            md.append(f"| {t_name} | {r['magnitude']:+.4f} | "
                      f"{r['min_gap']:+.4f} | {r['max_gap']:+.4f} | "
                      f"{r['n_regimes_with_data']} | "
                      f"{''.join(r['signs'])} |")
    md.append("")

    md += ["## RPM-006 per-regime trained-vs-random z-score",
            "",
            "| Regime | wikitext | arc_easy | lambada_openai |",
            "|---|---|---|---|"]
    for regime in regimes:
        z_row = rpm006[regime]
        cells = []
        for t_name in TASKS:
            z = z_row.get(t_name)
            cells.append(f"{z:+.2f}σ" if z is not None else "—")
        md.append(f"| {regime} | {cells[0]} | {cells[1]} | {cells[2]} |")
    md.append("")
    return "\n".join(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path,
                     default=Path("/home/andrew-jochl/TORUS"))
    ap.add_argument("--out_dir", type=Path,
                     default=Path("/home/andrew-jochl/TORUS/research/"
                                  "residual-pareto/experiments"))
    args = ap.parse_args()

    # Stage 1 regimes (EXP-RPM-D{0..5}) — suffix ""
    s1_regimes = discover_regimes(args.base, suffix="")
    s1_data = collect_data(args.base, s1_regimes)
    if s1_data:
        s1_rpm001 = compute_rpm001(s1_data, s1_regimes)
        s1_rpm002 = compute_rpm002(s1_data, s1_regimes)
        s1_rpm006 = compute_rpm006(s1_data, s1_regimes)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "RPM-001-002-006-analysis.json").write_text(
            json.dumps({"rpm001": s1_rpm001, "rpm002": s1_rpm002,
                        "rpm006": s1_rpm006}, indent=2, default=str))
        (args.out_dir / "RPM-001-002-006-analysis.md").write_text(
            render_md(s1_rpm001, s1_rpm002, s1_rpm006,
                      s1_regimes, "Stage 1 (threshold axis)"))
        print(f"[analyze] Stage 1 written: {len(s1_regimes)} regimes")
    else:
        print("[analyze] no Stage 1 regimes found")
        s1_data = {}

    # Stage 1.5 regimes (EXP-RPM-D{0..5}p) — suffix "p"
    s15_regimes = discover_regimes(args.base, suffix="p")
    s15_data = collect_data(args.base, s15_regimes)
    if s15_data:
        s15_rpm001 = compute_rpm001(s15_data, s15_regimes)
        s15_rpm002 = compute_rpm002(s15_data, s15_regimes)
        s15_rpm006 = compute_rpm006(s15_data, s15_regimes)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "RPM-001-002-006-analysis-15.json").write_text(
            json.dumps({"rpm001": s15_rpm001, "rpm002": s15_rpm002,
                        "rpm006": s15_rpm006}, indent=2, default=str))
        (args.out_dir / "RPM-001-002-006-analysis-15.md").write_text(
            render_md(s15_rpm001, s15_rpm002, s15_rpm006,
                      s15_regimes, "Stage 1.5 (observed-ppl axis)"))
        print(f"[analyze] Stage 1.5 written: {len(s15_regimes)} regimes")
    else:
        print("[analyze] no Stage 1.5 regimes found")
        s15_data = {}

    # Combined analysis (Stage 1 + Stage 1.5)
    if s1_data and s15_data:
        all_regimes = s1_regimes + s15_regimes
        combined = {**s1_data, **s15_data}
        c_rpm001 = compute_rpm001(combined, all_regimes)
        c_rpm002 = compute_rpm002(combined, all_regimes)
        c_rpm006 = compute_rpm006(combined, all_regimes)
        (args.out_dir / "RPM-001-002-006-analysis-combined.json").write_text(
            json.dumps({"rpm001": c_rpm001, "rpm002": c_rpm002,
                        "rpm006": c_rpm006}, indent=2, default=str))
        (args.out_dir / "RPM-001-002-006-analysis-combined.md").write_text(
            render_md(c_rpm001, c_rpm002, c_rpm006, all_regimes,
                       "Stage 1 + Stage 1.5 combined"))
        print(f"[analyze] combined written: {len(all_regimes)} regimes")
    else:
        print("[analyze] combined skipped: need both Stage 1 + 1.5 data")


if __name__ == "__main__":
    main()