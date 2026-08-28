import json, os
for t in [0.6, 0.7, 0.8, 0.9, 1.0]:
    d = f"/home/andrew-jochl/TORUS/runs/r/EXP-RPM-AF2D-SEVERITY/threshold-{t}/"
    if not os.path.exists(d):
        print(f"thr={t}: not started")
        continue
    candidates = [x for x in os.listdir(d) if x.startswith("2026") and not x.endswith("-base")]
    if not candidates:
        print(f"thr={t}: no timestamp dir")
        continue
    ts_dir = sorted(candidates)[0]
    p = d + ts_dir + "/aggregate.json"
    if not os.path.exists(p):
        print(f"thr={t} ts={ts_dir}: aggregate.json not yet (in progress)")
        continue
    dd = json.load(open(p))
    print(f"thr={t} ts={ts_dir} tolerance_violations={dd.get('tolerance_violations')} trained_arms={list(dd['trained_arms'].keys())} untrained_controls={list(dd.get('untrained_controls', {}).keys())}")
