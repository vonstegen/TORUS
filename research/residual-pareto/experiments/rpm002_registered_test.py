"""RPM-002 registered test: trained-vs-random effect size must be
non-decreasing across at least 3 consecutive damage regimes (in
REGIME ORDER), for each axis.

Definition (verbatim from research/residual-pareto/claims/RPM-002.yaml):
  PASS: For at least three consecutive damage regimes (e.g. D1->D2->D3
        or D2->D3->D4 from the D0..D5 sweep), the trained [effect size
        is non-decreasing].
  FAIL: The effect size is non-increasing across all consecutive regime
        pairs, OR the trained-vs-random separation fails [some other
        criterion].

We test for each metric × axis, all consecutive 3-tuples of regimes
in REGIME ORDER (D1..D5 for Stage 1; D1'..D5' for Stage 1.5). A
3-tuple passes if z[i+1] >= z[i] - eps (non-decreasing within tol).

The Stage 1.5 axes are the CAL-calibrated observed-ppl axis, ordered
by observed ppl ascending: D1' (88) < D2' (204) < D3' (303) < D4'
(430) < D5' (697). Stage 1 threshold-axis ordering is by regime
label (D1=0.0, D2=0.3, D3=0.5, D4=0.6, D5=0.7) which on the
observed-ppl axis collapses D1=D2=D3 to 1524.80 so the ordering is
ambiguous.
"""
import json
import statistics
import math
from pathlib import Path

ACC_METRICS = {"arc_easy", "lambada_openai"}
PPL_METRICS = {"wikitext"}
TASKS = ["wikitext", "arc_easy", "lambada_openai"]
TRAINED_ARMS = ["t2_ternary", "int4_residual", "int8_residual",
                "lora", "dense_adapter"]
RANDOM_ARMS = ["random_t2_ternary", "random_lora"]
ALL_ARMS = TRAINED_ARMS + RANDOM_ARMS

def read_arm_summary(run_path, seed, arm):
    p = run_path / f"seed-{seed:03d}" / arm / "eval.summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def value_for(tasks, t_name):
    if not tasks:
        return None
    rec = tasks.get(t_name)
    if rec is None:
        return None
    v = rec.get("value")
    return float(v) if v is not None else None


def collect_data(base, regimes):
    runs_root = base / "runs" / "r"
    data = {}
    for regime in regimes:
        runs_in = sorted((runs_root / regime).iterdir())
        if not runs_in:
            continue
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
                        t: value_for(es.get("tasks", {}), t) for t in TASKS
                    }
    return data


def effect_size_z(data, regime, t_name):
    """Trained-vs-random z-score (trained_mean - random_mean) / stderr_diff."""
    t_vals = [data[regime]["t2_ternary"][s][t_name]
              for s in (1, 2, 3)]
    r_vals = [data[regime]["random_t2_ternary"][s][t_name]
              for s in (1, 2, 3)]
    t_vals = [v for v in t_vals if v is not None]
    r_vals = [v for v in r_vals if v is not None]
    if len(t_vals) < 2 or len(r_vals) < 1:
        return None
    trained_mean = statistics.mean(t_vals)
    trained_se = statistics.stdev(t_vals) / math.sqrt(len(t_vals))
    random_mean = statistics.mean(r_vals)
    random_se = (statistics.stdev(r_vals) / math.sqrt(len(r_vals))
                  if len(r_vals) >= 2 else 0.0)
    denom = math.sqrt(trained_se ** 2 + random_se ** 2)
    if denom == 0:
        return None
    return (trained_mean - random_mean) / denom


def main():
    base = Path("/home/andrew-jochl/TORUS")
    out = {}
    for label, regimes in [
        ("Stage 1 (D1..D5, threshold axis)", ["EXP-RPM-D1", "EXP-RPM-D2",
                                               "EXP-RPM-D3", "EXP-RPM-D4",
                                               "EXP-RPM-D5"]),
        ("Stage 1.5 (D1'..D5', CAL ppl axis ascending)",
         ["EXP-RPM-D1p", "EXP-RPM-D2p", "EXP-RPM-D3p",
          "EXP-RPM-D4p", "EXP-RPM-D5p"]),
    ]:
        data = collect_data(base, regimes)
        z_seq = {t_name: [effect_size_z(data, r, t_name)
                          for r in regimes]
                  for t_name in TASKS}
        out[label] = z_seq

        print(f"\n=========== {label} ===========")
        for t_name, seq in z_seq.items():
            print(f"  {t_name}: {['%.2f' % s if s is not None else 'NA'
                                   for s in seq]}")
            # Search for 3-consecutive non-decreasing subsequence
            found = None
            for i in range(len(seq) - 2):
                s0, s1, s2 = seq[i], seq[i+1], seq[i+2]
                if s0 is None or s1 is None or s2 is None:
                    continue
                if s1 >= s0 and s2 >= s1:
                    found = (i, regimes[i], regimes[i+1], regimes[i+2])
                    break
            if found:
                print(f"    -> 3-consecutive non-decreasing: {found[1]}->"
                      f"{found[2]}->{found[3]} (z={seq[found[0]]:.2f}->"
                      f"{seq[found[0]+1]:.2f}->{seq[found[0]+2]:.2f}) "
                      f"PASS-equivalent (registered)")
            else:
                # Check FAIL rule: non-increasing across ALL pairs
                non_increasing_pairs = sum(
                    1 for i in range(len(seq) - 1)
                    if seq[i] is not None and seq[i+1] is not None
                    and seq[i+1] <= seq[i])
                total_pairs = sum(
                    1 for i in range(len(seq) - 1)
                    if seq[i] is not None and seq[i+1] is not None)
                print(f"    -> NO 3-consecutive non-decreasing subsequence")
                print(f"    -> Non-increasing pairs: {non_increasing_pairs}/"
                      f"{total_pairs}")

    Path("/tmp/rpm002_test_result.json").write_text(
        json.dumps(out, indent=2, default=str))
    print("\n[written /tmp/rpm002_test_result.json]")


if __name__ == "__main__":
    main()