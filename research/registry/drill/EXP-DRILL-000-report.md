# EXP-DRILL-000 — provenance drill report

- namespace: `runs/drill/EXP-DRILL-000/20260822T182426Z` (legion)
- complete record: config.json, provenance.json, train.jsonl, eval.json,
  checkpoint.sha256, ARTIFACTS.json — all present
- duplicate namespace claim by a second writer: rejected (mkdir exit 1)
- checkpoint hash verify round-trip: true
- git_sha: ee62c459f13684a979ca9c573f433f7d5463fcf1

CP0.2 criterion: complete run record end-to-end AND a second writer cannot
write into the namespace. Result: **PASS**.
