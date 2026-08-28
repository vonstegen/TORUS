import json, os
p = "/home/andrew-jochl/TORUS/runs/r/EXP-RPM-AF2D-SEVERITY/threshold-0.7/20260826T004835Z/aggregate.json"
d = json.load(open(p))
print("trained_arms:", list(d["trained_arms"].keys()))
print("untrained_controls:", list(d.get("untrained_controls", {}).keys()))
# Check random arm eval files
for s in ["001","002","003"]:
    for a in ["random_t2_ternary","random_lora"]:
        d2 = f"/home/andrew-jochl/TORUS/runs/r/EXP-RPM-AF2D-SEVERITY/threshold-0.7/20260826T004835Z/seed-{s}/{a}/eval.summary.json"
        if os.path.exists(d2):
            ej = json.load(open(d2))
            ts = ej.get("tasks") or {}
            print(f"  {s}/{a}: tasks keys = {list(ts.keys())}")
        else:
            print(f"  {s}/{a}: MISSING")
