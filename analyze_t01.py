import json, math
LATEST = sorted([d for d in __import__("os").listdir("/home/andrew-jochl/TORUS/runs/r/EXP-RPM-T01") if d.startswith("20")])[-1]
base = f"/home/andrew-jochl/TORUS/runs/r/EXP-RPM-T01/{LATEST}"
summary = json.load(open(f"{base}/held_out_summary.json"))

# Naive stderr from the random controls (deterministic across seeds, but
# we only have seed-001; treat random_t2 / random_lora / trained as
# independent estimates of "no signal" variance.)
print(f"{'task':<12} {'t2':<8} {'rand_t2':<8} {'rand_lora':<8} {'trained_mean':<14} {'trained_max':<14} {'t2-rand_t2':<10} {'t2-rand_lora':<12} {'t2-mean':<8}")
print("-" * 100)
for task, d in summary["by_task"].items():
    t2 = d["by_arm"]["t2_ternary"]
    rt2 = d["by_arm"]["random_t2_ternary"]
    rl = d["by_arm"]["random_lora"]
    tm = d["trained_mean"]
    tx = d["trained_max"]
    d12 = d["t2_minus_random_t2"]
    d13 = d["t2_minus_random_lora"]
    d14 = d["t2_minus_trained_mean"]
    print(f"{task:<12} {t2:<8.4f} {rt2:<8.4f} {rl:<8.4f} {tm:<14.4f} {tx:<14.4f} {d12:<+10.4f} {d13:<+12.4f} {d14:<+8.4f}")

# Approximate stderr: stderr_proxy is range/4 of trained arms
print()
print("T2 vs random_t2_ternary (architecture-vs-training on T2 ternary):")
wins = 0
for task, d in summary["by_task"].items():
    se = d["trained_stderr_proxy"]
    d12 = d["t2_minus_random_t2"]
    z = d12 / se if se else 0
    print(f"  {task}: T2 - rand_t2 = {d12:+.4f}, z (proxy) = {z:+.2f}")
    if d12 > 0 and z >= 1.0:
        wins += 1
print(f"  T2 wins on {wins} of 4 tasks at ≥+1σ (proxy z)")
print()
print("T2 vs random_lora (T2 vs LoRA control):")
wins = 0
for task, d in summary["by_task"].items():
    se = d["trained_stderr_proxy"]
    d13 = d["t2_minus_random_lora"]
    z = d13 / se if se else 0
    print(f"  {task}: T2 - rand_lora = {d13:+.4f}, z (proxy) = {z:+.2f}")
    if d13 > 0 and z >= 1.0:
        wins += 1
print(f"  T2 wins on {wins} of 4 tasks at ≥+1σ (proxy z)")
print()
print("T2 wins or ties with best trained comparator:")
wins = 0
for task, d in summary["by_task"].items():
    tx = d["trained_max"]
    t2 = d["by_arm"]["t2_ternary"]
    if t2 >= tx - 0.001:
        print(f"  {task}: T2 {t2:.4f} = trained_max {tx:.4f} (TIE)")
        wins += 1
    else:
        diff = tx - t2
        print(f"  {task}: T2 {t2:.4f} vs trained_max {tx:.4f} (diff={diff:+.4f})")
print(f"  T2 ties best trained on {wins} of 4 tasks")
