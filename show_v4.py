import json, statistics

LATEST = "20260825T220000Z"
base = f"/home/andrew-jochl/TORUS/runs/r/EXP-RPM-L15-GAUSS-V4/{LATEST}"
ARMS = ["t2_ternary", "random_t2_ternary", "random_lora", "dense_adapter", "lora", "int4_residual", "int8_residual"]
TASKS = ["wikitext", "arc_easy", "lambada_openai"]
SEEDS = [1, 2, 3]

data = {a: {t: [] for t in TASKS} for a in ARMS}
for s in SEEDS:
    for a in ARMS:
        f = f"{base}/seed-{s:03d}/{a}/eval.summary.json"
        try:
            d = json.load(open(f))
        except FileNotFoundError:
            continue
        for t in TASKS:
            v = d.get("tasks", {}).get(t, {}).get("value")
            if v is not None:
                data[a][t].append(v)

print("3-seed mean +/- stderr:")
print(f"{'arm':<20} {'wikitext_ppl':>20} {'arc_easy':>16} {'lambada':>16}")
for a in ARMS:
    line = f"{a:<20}"
    for t in TASKS:
        vals = data[a][t]
        if vals:
            m = statistics.fmean(vals)
            se = statistics.stdev(vals) / len(vals)**0.5 if len(vals) > 1 else 0.0
            line += f" {m:>10.4f} ± {se:.4f}"
        else:
            line += f" {'—':>20}"
    print(line)

t2 = data["t2_ternary"]
rt2 = data["random_t2_ternary"]
print()
print("T2 trained vs random_t2 (architecture-vs-training signal):")
print(f"{'task':<15} {'T2 mean':>10} {'rand mean':>12} {'Δ':>10} {'stderr':>10} {'z':>8}")
for t in TASKS:
    if t2[t] and rt2[t]:
        m1 = statistics.fmean(t2[t])
        m2 = statistics.fmean(rt2[t])
        all_vals = t2[t] + rt2[t]
        if len(all_vals) > 1:
            se = statistics.stdev(all_vals) / (len(all_vals) / 2) ** 0.5
        else:
            se = 0.0
        z = (m1 - m2) / se if se else 0
        print(f"{t:<15} {m1:>10.4f} {m2:>12.4f} {m1-m2:>+10.4f} {se:>10.4f} {z:>+8.2f}σ")

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

print()
print("Cost-vector Pareto (B + L):")
print(f"{'arm':<20} {'bytes':>10} {'latency_ms':>12}")
for a in ARMS:
    if a.startswith("random"):
        continue
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
