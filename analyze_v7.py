"""Stage 2 v7 boundary confirmation analysis.

Two data shapes coexist:
  - Trained arms (t2_ternary, lora): tasks[task] = {metric, value}
  - Random arms (random_t2_ternary, random_lora): tasks[<task>_<metric>,none] = value

Two ts_dirs per threshold:
  - <ts>      : tournament run with all 4 arms, multi-task eval
  - <ts>-base : damaged-base-only eval (single task, no arms)
We use only the non-base ts_dir.
"""
from __future__ import annotations
import json, math
from pathlib import Path

V6_ROOT = Path("/tmp/audit-v6-summaries")
V7_ROOT = Path("/tmp/audit-v7-summaries")

TASK_METRIC = {
    "wikitext": ("word_perplexity", "lower"),
    "arc_easy": ("acc_norm", "higher"),
    "lambada_openai": ("acc", "higher"),
}
TASK_NAMES = ["wikitext", "arc_easy", "lambada_openai"]
ARMS = ["t2_ternary", "lora", "random_t2_ternary", "random_lora"]


def extract_metric(d, task):
    metric_key, _ = TASK_METRIC[task]
    tasks = d.get("tasks", {})
    if task in tasks:
        e = tasks[task]
        if isinstance(e, dict):
            if "metric" in e:
                m = e["metric"].split(",")[0]
                if m == metric_key:
                    return float(e["value"])
            if "value" in e and "metric" not in e:
                return float(e["value"])
    flat_key = f"{task}_{metric_key},none"
    if flat_key in tasks and isinstance(tasks[flat_key], (int, float)):
        return float(tasks[flat_key])
    for k, v in d.items():
        if not isinstance(v, (int, float)):
            continue
        if k.startswith(task + "_") and metric_key in k.split(",")[0]:
            return float(v)
    return None


def collect(root):
    out = {}
    for thr_dir in sorted(root.iterdir()):
        thr = float(thr_dir.name.split("-")[1])
        # Skip base-eval ts_dirs to avoid overwriting trained arm cells.
        ts_dirs = [p for p in thr_dir.iterdir() if not p.name.endswith("-base")]
        for ts_dir in sorted(ts_dirs):
            for seed_dir in sorted(ts_dir.iterdir()):
                if not seed_dir.name.startswith("seed-"):
                    continue
                seed = int(seed_dir.name.split("-")[1])
                for arm in ARMS:
                    f = seed_dir / arm / "eval.summary.json"
                    if not f.exists():
                        continue
                    data = json.loads(f.read_text())
                    vals = {}
                    for task in TASK_NAMES:
                        v = extract_metric(data, task)
                        if v is not None:
                            vals[task] = v
                    if vals:
                        out[(thr, seed, arm)] = vals
    return out


def stats(values):
    n = len(values)
    if n == 0:
        return None, None
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var)


def zscore(a, b):
    if len(a) < 2 or len(b) < 2:
        return None
    ma, sa = stats(a)
    mb, sb = stats(b)
    if ma is None or mb is None:
        return None
    pooled = math.sqrt(sa**2 + sb**2)
    if pooled == 0:
        return None
    return (ma - mb) / pooled


def sign(task):
    return -1.0 if TASK_METRIC[task][1] == "lower" else 1.0


def axis_zs(data, thr, ctrl_arm, seeds):
    trained_vals = {t: [] for t in TASK_NAMES}
    ctrl_vals = {t: [] for t in TASK_NAMES}
    for seed in seeds:
        t = data.get((thr, seed, "t2_ternary"))
        c = data.get((thr, seed, ctrl_arm))
        if t and c:
            for tk in TASK_NAMES:
                if tk in t and tk in c:
                    trained_vals[tk].append(t[tk])
                    ctrl_vals[tk].append(c[tk])
    zs = {}
    for tk in TASK_NAMES:
        z = zscore(trained_vals[tk], ctrl_vals[tk])
        zs[tk] = z * sign(tk) if z is not None else None
    return zs, trained_vals, ctrl_vals


def is_active(zs):
    if any(z is None for z in zs.values()):
        return None
    above = sum(1 for z in zs.values() if z >= 2.0)
    below = sum(1 for z in zs.values() if z <= -2.0)
    return above >= 2 and below == 0


def main():
    v6 = collect(V6_ROOT)
    v7 = collect(V7_ROOT)
    print(f"v6 cells: {len(v6)}")
    print(f"v7 cells: {len(v7)}")
    print()
    print("=" * 110)
    print(f"{'thr':>5}  {'axis':>5}  "
          f"{'wikitext_z':>12}  {'arc_easy_z':>11}  {'lambada_z':>10}  "
          f"{'v7_active':>10}  {'v6_active':>10}  {'reproduces':>11}")
    print("=" * 110)
    band_lrn = True
    band_tsp = True
    n_lrn = 0
    n_tsp = 0
    for thr in [0.6, 0.8, 1.0]:
        for axis, ctrl_arm in [("LRN", "random_t2_ternary"),
                                ("TSP", "random_lora")]:
            v7_zs, _, _ = axis_zs(v7, thr, ctrl_arm, [4, 5, 6])
            v6_zs, _, _ = axis_zs(v6, thr, ctrl_arm, [1, 2, 3])
            v7_active = is_active(v7_zs)
            v6_active = is_active(v6_zs)
            if v7_active is True:
                if axis == "LRN":
                    n_lrn += 1
                else:
                    n_tsp += 1
            if v7_active is False:
                if axis == "LRN":
                    band_lrn = False
                else:
                    band_tsp = False
            reproduces = (v7_active == v6_active) and (v7_active is True)
            print(f"{thr:>5}  {axis:>5}  "
                  f"{(v7_zs['wikitext'] or 0):>12.2f}  "
                  f"{(v7_zs['arc_easy'] or 0):>11.2f}  "
                  f"{(v7_zs['lambada_openai'] or 0):>10.2f}  "
                  f"{'YES' if v7_active else 'NO' if v7_active is False else 'NA':>10}  "
                  f"{'YES' if v6_active else 'NO' if v6_active is False else 'NA':>10}  "
                  f"{'REPRO' if reproduces else 'MISMATCH' if v6_active != v7_active else 'NA':>11}")
    print()
    print("=" * 110)
    print("VERDICT")
    print("=" * 110)
    print()
    print(f"v7 LRN active at {n_lrn}/3 thresholds; v7 TSP active at {n_tsp}/3 thresholds")
    print(f"v6 LRN active at all 5 thresholds (commit 75c20d3)")
    print()
    if band_lrn and band_tsp:
        print("CONFIRMED: AF2-D/TWN operating band reproduces under fresh "
              "seeds at lower boundary (0.6), interior (0.8), and upper "
              "boundary (1.0).")
        return "CONFIRMED"
    if (n_lrn + n_tsp) >= 4:
        print("REFINED: partial reproduction.")
        return "REFINED"
    print("INVALIDATED: v6 finding was seed-set-specific.")
    return "INVALIDATED"


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() == "CONFIRMED" else 1)