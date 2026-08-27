"""Stage 3 v1 cross-mechanism analysis — final version.

Compares LRN (trained T2 vs random T2) and TSP (trained T2 vs random
LoRA) wikitext ppl z-scores at 5 calibrated (mechanism, magnitude) points.

Random arms only have wikitext (single-task fast eval); trained arms
have all 3 tasks. We compare only wikitext for cross-arm z-scores.
"""
from __future__ import annotations
import json, math
from pathlib import Path

ROOT = Path("/tmp/audit-s3-data")
TASK = "wikitext"
TASK_KEY = ("word_perplexity", "lower")  # for sign correction
SEEDS = [1, 2, 3]
ARMS = ["t2_ternary", "lora", "random_t2_ternary", "random_lora"]


def extract_metric(d, task):
    metric_key, _ = TASK_KEY
    tasks = d.get("tasks", {})
    if task in tasks:
        e = tasks[task]
        if isinstance(e, dict) and "metric" in e:
            m = e["metric"].split(",")[0]
            if m == metric_key:
                return float(e["value"])
        if isinstance(e, dict) and "value" in e and "metric" not in e:
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


def collect():
    out = {}
    for cell_dir in sorted(ROOT.iterdir()):
        if cell_dir.name.endswith("-base"):
            continue
        cell_id = cell_dir.name
        for seed_dir in sorted(cell_dir.iterdir()):
            if not seed_dir.name.startswith("seed-"):
                continue
            seed = int(seed_dir.name.split("-")[1])
            for arm in ARMS:
                f = seed_dir / arm / "eval.summary.json"
                if not f.exists():
                    continue
                data = json.loads(f.read_text())
                v = extract_metric(data, TASK)
                if v is not None:
                    out[(cell_id, seed, arm)] = v
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
    """Two-sample z. Negative because lower wikitext ppl is better; we
    return sign-corrected z so positive = trained better."""
    if len(a) < 2 or len(b) < 2:
        return None
    ma, sa = stats(a)
    mb, sb = stats(b)
    if ma is None or mb is None:
        return None
    pooled = math.sqrt(sa**2 + sb**2)
    if pooled == 0:
        return None
    return (mb - ma) / pooled  # higher when trained_ppl < ctrl_ppl


def main():
    data = collect()
    print(f"cells: {len(data)}")
    print()
    print("=" * 100)
    print(f"{'cell_id':<18}  {'axis':>5}  "
          f"{'wikitext_z':>12}  {'trained_T2_ppl':>14}  {'ctrl_ppl':>12}  {'recovery':>10}")
    print("=" * 100)
    cell_ids = ["BAND-1-Gaussian", "BAND-3-Gaussian", "BAND-3-TWN",
                 "BAND-4-Gaussian", "BAND-4-TWN"]
    band_results = {}
    for cell_id in cell_ids:
        for axis, ctrl_arm in [("LRN", "random_t2_ternary"),
                                ("TSP", "random_lora"),
                                ("T2vsLoRA", "lora")]:
            trained_vals = []
            ctrl_vals = []
            for seed in SEEDS:
                t = data.get((cell_id, seed, "t2_ternary"))
                c = data.get((cell_id, seed, ctrl_arm))
                if t is not None and c is not None:
                    trained_vals.append(t)
                    ctrl_vals.append(c)
            z = zscore(trained_vals, ctrl_vals)
            mt = sum(trained_vals) / len(trained_vals) if trained_vals else None
            mc = sum(ctrl_vals) / len(ctrl_vals) if ctrl_vals else None
            rec = ((mc - mt) / mc) if mt and mc else None
            band_results[(cell_id, axis)] = {
                "z": z, "mt": mt, "mc": mc, "rec": rec,
                "active": z is not None and z >= 2.0,
            }
            print(f"{cell_id:<18}  {axis:>5}  "
                  f"{(z or 0):>12.2f}  "
                  f"{(mt or 0):>14.2f}  "
                  f"{(mc or 0):>12.2f}  "
                  f"{(rec or 0):>10.3f}")
    print()
    print("=" * 100)
    print("CROSS-MECHANISM COMPARISON AT BAND-3 (PRIMARY, base ppl ~430 vs ~451)")
    print("=" * 100)
    print()
    for axis in ["LRN", "TSP", "T2vsLoRA"]:
        twn = band_results[("BAND-3-TWN", axis)]
        gauss = band_results[("BAND-3-Gaussian", axis)]
        print(f"  {axis}:")
        print(f"    TWN thr=0.7     z={twn['z']:.2f}  trained_T2_ppl={twn['mt']:.2f}  ctrl_ppl={twn['mc']:.2f}  recovery={twn['rec']:.3f}")
        print(f"    Gaussian s=3.0  z={gauss['z']:.2f}  trained_T2_ppl={gauss['mt']:.2f}  ctrl_ppl={gauss['mc']:.2f}  recovery={gauss['rec']:.3f}")
    print()
    lrn_twn_active = band_results[("BAND-3-TWN", "LRN")]["active"]
    lrn_gauss_active = band_results[("BAND-3-Gaussian", "LRN")]["active"]
    if lrn_twn_active and lrn_gauss_active:
        rec_t = band_results[("BAND-3-TWN", "LRN")]["rec"]
        rec_g = band_results[("BAND-3-Gaussian", "LRN")]["rec"]
        if min(rec_t, rec_g) > 0:
            ratio = max(rec_t, rec_g) / min(rec_t, rec_g)
            if ratio < 3.0:
                print("BROAD INTERPRETATION supported at BAND-3: LRN positive in both mechanisms at matched magnitude,")
                print(f"recovery ratio = {ratio:.2f}x (within 3x threshold).")
                print("=> Ternary residual correction has a GENERAL learned-recovery property.")
            else:
                print(f"NARROW INTERPRETATION: LRN positive in both but recovery differs by {ratio:.2f}x (>3x).")
                print("=> Ternary residual correction is mechanism-dependent at this magnitude.")
        else:
            print("INCONCLUSIVE: recovery ratio not meaningful.")
    elif lrn_twn_active and not lrn_gauss_active:
        print("NARROW INTERPRETATION supported at BAND-3: LRN positive at TWN but NOT at Gaussian")
        print("at matched magnitude. Ternary residual correction is mechanism-specific.")
    elif not lrn_twn_active and lrn_gauss_active:
        print("UNEXPECTED: LRN positive at Gaussian but not TWN at matched magnitude.")
    else:
        print("INCONCLUSIVE: LRN absent/inverted in both mechanisms at matched magnitude.")
    print()
    print("=" * 100)
    print("BAND-4 (SECONDARY, base ppl 1524 vs 4889, 3.2x magnitude mismatch)")
    print("=" * 100)
    print()
    for axis in ["LRN", "TSP"]:
        twn = band_results[("BAND-4-TWN", axis)]
        gauss = band_results[("BAND-4-Gaussian", axis)]
        print(f"  {axis}:")
        print(f"    TWN thr=0.5     z={twn['z']:.2f}  trained_T2_ppl={twn['mt']:.2f}  ctrl_ppl={twn['mc']:.2f}  recovery={twn['rec']:.3f}")
        print(f"    Gaussian s=5.0  z={gauss['z']:.2f}  trained_T2_ppl={gauss['mt']:.2f}  ctrl_ppl={gauss['mc']:.2f}  recovery={gauss['rec']:.3f}")
    print()
    print("=" * 100)
    print("BAND-1 (CONTROL, near-pristine damage, base ppl 15)")
    print("=" * 100)
    print()
    for axis in ["LRN", "TSP"]:
        r = band_results[("BAND-1-Gaussian", axis)]
        print(f"  {axis}:")
        print(f"    Gaussian s=1.0  z={r['z']:.2f}  trained_T2_ppl={r['mt']:.2f}  ctrl_ppl={r['mc']:.2f}  recovery={r['rec']:.3f}")
        if not r["active"]:
            print(f"    CONTROL: T2 does NOT help at near-pristine damage (z={r['z']:.2f}).")
            print(f"    Supports: LRN is damage-driven.")


if __name__ == "__main__":
    main()