"""Select the CAL threshold that produces ppl closest to a target.

Used by stage2-launch.sh to pick the threshold for the Stage 2
tournament at each site (so the damage is comparable across layer
sites).

Reads per-threshold aggregate.json files (one per threshold × 3 seeds
each) and finds the threshold whose ppl is closest to --target_ppl.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cal_root", type=Path, required=True,
                     help="Root of per-site CAL run "
                     "(e.g. runs/r/EXP-RPM-L15-CAL/<ts>/)")
    ap.add_argument("--target_ppl", type=float, default=425.76,
                     help="Target ppl for the tournament (AF2-D ref).")
    args = ap.parse_args()

    thr_dirs = sorted(args.cal_root.glob("thr-*"))
    if not thr_dirs:
        print(f"0.7")  # safe default; fail loud later
        sys.exit(1)

    best_thr = None
    best_dist = float("inf")
    for thr_dir in thr_dirs:
        agg_paths = sorted(thr_dir.glob("*.json"))
        # Skip the driver.log; use aggregate.json if present, else
        # compute from pre_train_eval.json per seed.
        agg_path = thr_dir / "aggregate.json"
        ppls = []
        if agg_path.exists():
            try:
                agg = json.loads(agg_path.read_text())
                # aggregate.json has tasks.wikitext.mean (when n>1
                # seeds). For CAL n=3 seeds.
                w = agg.get("trained_arms", {}).get(
                    "t2_ternary", {}).get("tasks", {}).get(
                    "wikitext", {}).get("mean")
                if w is not None:
                    ppls.append(w)
            except Exception:
                pass
        # Also read pre_train_eval.json per seed for cross-check.
        for pe_path in sorted(thr_dir.glob("seed-*/t2_ternary/pre_train_eval.json")):
            try:
                pe = json.loads(pe_path.read_text())
                w = pe.get("wikitext", {}).get("value")
                if w is not None:
                    ppls.append(w)
            except Exception:
                pass
        if not ppls:
            continue
        mean_ppl = statistics.mean(ppls)
        # parse threshold from dir name "thr-X_XX" -> float
        thr_str = thr_dir.name.replace("thr-", "").replace("_", ".")
        try:
            thr = float(thr_str)
        except ValueError:
            continue
        dist = abs(mean_ppl - args.target_ppl)
        print(f"  threshold={thr:.2f} -> ppl={mean_ppl:.2f} "
              f"(dist={dist:.2f})", file=sys.stderr)
        if dist < best_dist:
            best_dist = dist
            best_thr = thr

    if best_thr is None:
        print("0.7")
        sys.exit(1)
    print(f"  selected threshold={best_thr:.2f} "
          f"(closest to target ppl {args.target_ppl})",
          file=sys.stderr)
    print(f"{best_thr:.2f}")


if __name__ == "__main__":
    main()