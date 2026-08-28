import json
with open("/home/andrew-jochl/TORUS/runs/r/EXP-RPM-SYS/20260825T184527Z/systems_measurements.json") as f:
    s = json.load(f)
print(f"{'arm':<22} {'B(MB)':<8} {'O(M)':<8} {'L(ms)':<10} {'E(J)':<8} {'meanW':<8}")
print("-" * 70)
for arm, r in s["arms"].items():
    if "error" in r:
        print(f"{arm:<22} ERROR: {r['error'][:50]}")
        continue
    b_mb = r["B_deployed_bytes"] / 1024 / 1024
    o_m = r["O_inference_ops_per_token"] / 1e6
    l_med = r["L_latency_per_token_ms"]["median"]
    e_j = r["E_joules_per_token"]["value"]
    e_w = r["E_joules_per_token"]["mean_w"]
    print(f"{arm:<22} {b_mb:<8.2f} {o_m:<8.1f} {l_med:<10.3f} {e_j:<8.3f} {e_w:<8.1f}")
