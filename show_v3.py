import json, statistics
LATEST = "20260825T190000Z"
base = f"/home/andrew-jochl/TORUS/runs/r/EXP-RPM-L15-GAUSS-V3/{LATEST}"

ARMS = ["t2_ternary", "random_t2_ternary", "random_lora", "dense_adapter", "lora", "int4_residual", "int8_residual"]
TASKS = ["wikitext", "arc_easy", "lambada_openai"]
SEEDS = [1, 2, 3]

print("Per-arm per-seed values:")
print("=" * 80)
data = {a: {t: [] for t in TASKS} for a in ARMS}
for s in SEEDS:
    print(f"\nseed-{s:03d}:")
    for a in ARMS:
        f = f"{base}/seed-{s:03d}/{a}/eval.summary.json"
        try:
            d = json.load(open(f))
        except FileNotFoundError:
            print(f"  {a:20s} MISSING")
            continue
        vals = {}
        for t in TASKS:
            v = d.get("tasks", {}).get(t, {}).get("value")
            if v is not None:
                vals[t] = v
                data[a][t].append(v)
        out = " ".join(f"{t}={v:.4f}" for t, v in vals.items())
        print(f"  {a:20s} {out}")

print()
print("=" * 80)
print("3-seed mean +/- stderr:")
print(f"{'arm':<20} {'wikitext_ppl':>14} {'arc_easy':>10} {'lambada':>10}")
for a in ARMS:
    line = f"{a:<20}"
    for t in TASKS:
        vals = data[a][t]
        if vals:
            m = statistics.fmean(vals)
            se = statistics.stdev(vals) / len(vals)**0.5 if len(vals) > 1 else 0.0
            line += f" {m:>8.4f} ± {se:.4f}"
        else:
            line += f" {'—':>14}"
    print(line)

# T2 vs random
t2 = data["t2_ternary"]
rt2 = data["random_t2_ternary"]
print()
print("=" * 80)
print("T2 trained vs random_t2 (architecture-vs-training signal):")
print(f"{'task':<15} {'T2 mean':>10} {'rand mean':>12} {'Δ':>10} {'stderr':>10} {'z':>8}")
for t in TASKS:
    if t2[t] and rt2[t]:
        m1 = statistics.fmean(t2[t])
        m2 = statistics.fmean(rt2[t])
        # Pool stderr over the 6 values (3 t2 + 3 random)
        all_vals = t2[t] + rt2[t]
        if len(all_vals) > 1:
            se = statistics.stdev(all_vals) / (len(all_vals) / 2) ** 0.5
        else:
            se = 0.0
        z = (m1 - m2) / se if se else 0
        print(f"{t:<15} {m1:>10.4f} {m2:>12.4f} {m1-m2:>+10.4f} {se:>10.4f} {z:>+8.2f}σ")

# T2 vs random_lora
rl = data["random_lora"]
print()
print("T2 trained vs random_lora:")
print(f"{'task':<15} {'T2 mean':>10} {'rand mean':>12} {'Δ':>10} {'stderr':>10} {'z':>8}")
for t in TASKS:
    if t2[t] and rl[t]:
        m1 = statistics.fmean(t2[t])
        m2 = statistics.fmean(rl[t])
        all_vals = t2[t] + rl[t]
        if len(all_vals) > 1:
            se = statistics.stdev(all_vals) / (len(all_vals) / 2) ** 0.5
        else:
            se = 0.0
        z = (m1 - m2) / se if se else 0
        print(f"{t:<15} {m1:>10.4f} {m2:>12.4f} {m1-m2:>+10.4f} {se:>10.4f} {z:>+8.2f}σ")

# Cost-vector Pareto
import yaml
print()
print("=" * 80)
print("Cost-vector Pareto (deployed_bytes + latency_per_token_titan_rtx):")
print(f"{'arm':<20} {'bytes':>10} {'latency_ms':>12}")
for a in ARMS:
    if a.startswith("random"):
        continue
    # Get from aggregate
    bytes_v = []
    lat_v = []
    for s in SEEDS:
        f = f"{base}/seed-{s:03d}/{a}/cost_vector.json"
        try:
            cv = json.load(open(f))
            bytes_v.append(cv.get("deployed_bytes", None))
            lat_v.append(cv.get("latency_per_token_titan_rtx", None))
        except FileNotFoundError:
            pass
    if bytes_v and lat_v:
        print(f"{a:<20} {statistics.fmean(bytes_v):>10.0f} {statistics.fmean(lat_v)*1000:>12.4f}")
