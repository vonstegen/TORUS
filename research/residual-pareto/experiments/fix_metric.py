"""Post-process eval.summary.json to align metrics with Stage 1
trained-arm choices. Reads eval.full.json which contains the full
lm-eval output, and rewrites eval.summary.json's `tasks` block to use
the same metric Stage 1 used:

  arc_easy: acc_norm,none
  lambada_openai: acc,none
  wikitext: word_perplexity,none

This is a one-shot fix to avoid re-running the 36 cells (~47 min).
"""
import json
from pathlib import Path
import glob

PREFERRED = {
    "arc_easy": "acc_norm,none",
    "lambada_openai": "acc,none",
    "wikitext": "word_perplexity,none",
}

BASE = Path("/home/andrew-jochl/TORUS")

paths = sorted(glob.glob(str(BASE / "runs/r/EXP-RPM-D*/2026*/af2d/seed-*/random_*/eval.summary.json")))
n_fixed = 0
for p in paths:
    es = json.loads(open(p).read())
    full_p = Path(p).parent / "eval.full.json"
    if not full_p.exists():
        continue
    full = json.loads(full_p.read_text())
    new_tasks = {}
    for t_name in ["wikitext", "arc_easy", "lambada_openai"]:
        pref = PREFERRED.get(t_name)
        if pref is None or pref not in full.get(t_name, {}):
            new_tasks[t_name] = {"metric": None, "value": None}
            continue
        new_tasks[t_name] = {"metric": pref,
                              "value": float(full[t_name][pref])}
    es["tasks"] = new_tasks
    open(p, "w").write(json.dumps(es, indent=2, default=str))
    n_fixed += 1

print(f"re-picked metric for {n_fixed} random-arm eval summaries")