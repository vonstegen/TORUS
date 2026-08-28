import json, os
LATEST = sorted([d for d in os.listdir("/home/andrew-jochl/TORUS/runs/r/EXP-RPM-T01") if d.startswith("20")])[-1]
base = f"/home/andrew-jochl/TORUS/runs/r/EXP-RPM-T01/{LATEST}"
for arm in ["t2_ternary", "int4_residual", "int8_residual", "lora", "dense_adapter", "random_t2_ternary", "random_lora"]:
    for t in ["hellaswag", "winogrande", "boolq", "openbookqa"]:
        f = f"{base}/{arm}/{t}/eval.summary.json"
        if os.path.exists(f):
            d = json.load(open(f))
            print(f"{arm}/{t}: {d['metric']}={d['value']:.4f} ({d['wall_clock_s']:.1f}s)")
        else:
            print(f"{arm}/{t}: missing")
