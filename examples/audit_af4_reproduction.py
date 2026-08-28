"""AF4-R reproduction auditor — AF8 governance helper (EXP-AF-004-R).

Two modes:

  --prepare   Build an independently generated wikitext-103 token
              cache at a NEW path (re-tokenized from the HF parquet
              shards; shard SHAs, cache SHA, tokenizer id, auditor
              PID, UTC recorded). Refuses to overwrite. Content
              identity with AF4's recorded cache sha256 is the
              expected outcome (deterministic tokenization) and
              serves as the provenance notary.

  --audit     Post-run reproduction audit. Independently recomputes
              arm means/stderrs from per-seed eval.summary.json files
              on BOTH sides (AF4 committed records + AF4-R fresh
              runs), replays the frozen AF4 acceptance formulas
              (never trusting aggregate.json), checks the ±2
              combined-stderr band per arm x metric, verifies run
              integrity (9/9 runs, freeze flags, deployed bytes, no
              NaN/inf), and writes the reproduction verdict:
              REPRODUCED / NOT_REPRODUCED / INVALID.

Usage on legion:

    python examples/audit_af4_reproduction.py --prepare \
        --out-path <new cache .npy> --manifest <cache_provenance.json> \
        --af4-cache-sha256 <sha from AF4 ids-cache.sha256>

    python examples/audit_af4_reproduction.py --audit \
        --run-dir <worktree>/runs/a/EXP-AF-004-R/<ts> \
        --af4-dir research/.../AF4/runs/20260828T121414Z \
        --out <run-dir>/audit.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# Reuse the AF1-R cache builder verbatim (single implementation of the
# independent token-cache contract).
_EXAMPLES = Path(__file__).resolve().parent


def _load_helper(path: Path, name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_af1_audit = _load_helper(_EXAMPLES, "audit_af1_reproduction")

ARMS = ("seq", "joint", "t1_only")
SEEDS = (1, 2, 3)
TASKS = ("wikitext", "arc_easy", "lambada_openai")
# Deployed-bytes architecture constants (from AF4's committed record).
EXPECTED_BYTES = {"seq": 8912896, "joint": 8912896, "t1_only": 4456448}
# Metric direction: ppl lower is better; acc higher is better.
LOWER_IS_BETTER = {"wikitext": True, "arc_easy": False,
                   "lambada_openai": False}


def load_summaries(run_dir: Path) -> list[dict]:
    """Load every per-(seed, arm) eval.summary.json under a run dir."""
    summaries = []
    for path in sorted(run_dir.glob("seed-*/**/eval.summary.json")):
        with open(path) as f:
            summaries.append(json.load(f))
    return summaries


def compute_arm_stats(summaries: list[dict]) -> dict:
    """arm -> task -> {n, mean, std, stderr, values}, recomputed from
    raw per-seed values. Never reads aggregate.json."""
    import numpy as np

    by_arm: dict[str, dict[str, list[float]]] = {}
    for s in summaries:
        for task, rec in s.get("tasks", {}).items():
            by_arm.setdefault(s["arm"], {}).setdefault(task, []).append(
                float(rec["value"])
            )
    stats: dict[str, dict] = {}
    for arm, tasks in by_arm.items():
        stats[arm] = {}
        for task, vals in tasks.items():
            arr = np.asarray(vals, dtype=np.float64)
            stats[arm][task] = {
                "n": int(arr.size),
                "mean": float(arr.mean()),
                "stderr": float(arr.std(ddof=1) / math.sqrt(arr.size))
                if arr.size > 1 else 0.0,
                "values": [float(v) for v in vals],
            }
    return stats


def z_seq(stats: dict, task: str) -> float | None:
    """Signed seq-better z for a task (positive = seq better).

    ppl: (mean_joint - mean_seq) / se_diff (seq lower is better).
    acc: (mean_seq - mean_joint) / se_diff (seq higher is better).
    """
    a, b = stats["seq"][task], stats["joint"][task]
    se = math.sqrt(a["stderr"] ** 2 + b["stderr"] ** 2)
    if se == 0:
        return None
    if LOWER_IS_BETTER[task]:
        return (b["mean"] - a["mean"]) / se
    return (a["mean"] - b["mean"]) / se


def replay_decision(stats: dict) -> dict:
    """Replay the frozen AF4 acceptance formulas (manifest
    pass_thresholds / fail_thresholds) against recomputed stats.

    Returns {zs, pass_clause_fired, fail1, fail2, verdict, direction}.
    verdict ∈ {"PASS", "FAIL"}; direction ∈
    {"sequential superior", "joint superior", "not separated", "mixed"}.
    """
    zs = {task: z_seq(stats, task) for task in TASKS}
    finite = {t: z for t, z in zs.items() if z is not None}
    pass_hits = [t for t, z in finite.items() if z > 2]
    regressions = [t for t, z in finite.items() if z < -1]
    joint_hits = [t for t, z in finite.items() if z < -2]

    pass_clause = bool(pass_hits) and not regressions
    fail2 = bool(joint_hits)                      # joint beats seq > 2σ
    fail1 = not finite or all(abs(z) <= 2 for z in finite.values())

    if pass_clause:
        verdict, direction = "PASS", "sequential superior"
    elif fail2:
        verdict, direction = "FAIL", "joint superior"
    elif fail1:
        verdict, direction = "FAIL", "not separated"
    else:
        # Separation in both directions at >1σ but no clause cleanly
        # fires; report raw zs for the human verdict.
        verdict, direction = "FAIL", "mixed"
    return {
        "zs": zs,
        "pass_clause_fired": pass_clause,
        "fail1_no_separation": fail1,
        "fail2_joint_superior": fail2,
        "verdict": verdict,
        "direction": direction,
    }


def band_check(stats_ref: dict, stats_new: dict) -> dict:
    """Per arm x task: |mean_new - mean_ref| vs
    2 * sqrt(se_ref^2 + se_new^2)."""
    rows = {}
    ok = True
    for arm in ARMS:
        rows[arm] = {}
        for task in TASKS:
            r, n = stats_ref[arm][task], stats_new[arm][task]
            band = 2 * math.sqrt(r["stderr"] ** 2 + n["stderr"] ** 2)
            delta = abs(n["mean"] - r["mean"])
            within = delta <= band
            rows[arm][task] = {
                "mean_ref": r["mean"], "mean_new": n["mean"],
                "abs_delta": delta, "band": band, "within": within,
            }
            ok = ok and within
    return {"within_all": ok, "rows": rows}


def run_integrity(run_dir: Path, summaries: list[dict]) -> dict:
    """Frozen AF4 integrity items, checked from raw artifacts."""
    problems: list[str] = []
    seen = {(s["arm"], s["seed"]) for s in summaries}
    expected = {(arm, seed) for arm in ARMS for seed in SEEDS}
    missing = expected - seen
    if missing:
        problems.append(f"missing runs: {sorted(missing)}")
    for s in summaries:
        if s["arm"] == "seq" and s.get("freeze_check") is not True:
            problems.append(f"freeze_check not true at seed {s['seed']}")
        exp = EXPECTED_BYTES[s["arm"]]
        if s.get("deployed_bytes") != exp:
            problems.append(
                f"{s['arm']} seed {s['seed']}: deployed_bytes "
                f"{s.get('deployed_bytes')} != {exp}"
            )
        for task in TASKS:
            v = s.get("tasks", {}).get(task, {}).get("value")
            if v is None or not math.isfinite(v):
                problems.append(
                    f"{s['arm']} seed {s['seed']}: {task} value {v}"
                )
        # Kill line from the AF4 manifest: post-train ppl worse than
        # the untrained PTQ reference collapses the run.
        ppl = s.get("tasks", {}).get("wikitext", {}).get("value")
        if ppl is not None and ppl > 427.71:
            problems.append(
                f"{s['arm']} seed {s['seed']}: ppl {ppl} worse than "
                f"the 427.71 kill line"
            )
    for hist in run_dir.glob("seed-*/**/history*.jsonl"):
        for line in hist.read_text().splitlines():
            row = json.loads(line)
            if not math.isfinite(row["loss"]):
                problems.append(f"non-finite loss in {hist}")
                break
    return {"ok": not problems, "problems": problems}


def reproduction_verdict(integrity: dict, replay: dict,
                         bands: dict, ref_replay: dict) -> dict:
    """The frozen reproduction_rule from the AF4-R manifest."""
    if not integrity["ok"]:
        return {"verdict": "INVALID", "reasons": integrity["problems"]}
    decision_match = (
        replay["verdict"] == ref_replay["verdict"]
        and replay["direction"] == ref_replay["direction"]
    )
    if decision_match and bands["within_all"]:
        return {
            "verdict": "REPRODUCED",
            "reasons": [
                f"decision replay matches AF4 ({replay['verdict']} / "
                f"{replay['direction']}); all arm x metric means "
                f"within the ±2 combined-stderr band"
            ],
        }
    reasons = []
    if not decision_match:
        reasons.append(
            f"decision replay returned {replay['verdict']} / "
            f"{replay['direction']} vs AF4's "
            f"{ref_replay['verdict']} / {ref_replay['direction']}"
        )
    if not bands["within_all"]:
        outside = [
            f"{arm}/{task}: |Δ|={r['abs_delta']:.4g} > band={r['band']:.4g}"
            for arm, tasks in bands["rows"].items()
            for task, r in tasks.items() if not r["within"]
        ]
        reasons.append("band violations: " + "; ".join(outside))
    return {"verdict": "NOT_REPRODUCED", "reasons": reasons}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--audit", action="store_true")
    p.add_argument("--out-path", type=Path,
                   help="(--prepare) new cache .npy path (must not preexist)")
    p.add_argument("--manifest", type=Path,
                   help="(--prepare) AF8 provenance record output")
    p.add_argument("--af4-cache-sha256", default=None,
                   help="(--prepare) AF4's recorded cache sha256")
    p.add_argument("--run-dir", type=Path,
                   help="(--audit) AF4-R run directory")
    p.add_argument("--af4-dir", type=Path,
                   help="(--audit) AF4 committed-record directory")
    p.add_argument("--out", type=Path, help="(--audit) audit JSON output")
    args = p.parse_args()

    if args.prepare:
        if args.out_path.exists():
            sys.exit(f"refusing to overwrite existing file: {args.out_path}")
        record = _af1_audit._build_wikitext_cache(args.out_path)
        new_sha = record["wikitext_cache_sha256"]
        same = new_sha == args.af4_cache_sha256
        record["af4_reference_sha256"] = args.af4_cache_sha256
        record["af4_reference_identity"] = same
        print(f"[af4-r-audit] new cache sha256: {new_sha}", flush=True)
        print(f"[af4-r-audit] af4  cache sha256: {args.af4_cache_sha256}",
              flush=True)
        if same:
            print("[af4-r-audit] identity: expected outcome "
                  "(deterministic tokenization).", flush=True)
        else:
            print("[af4-r-audit] WARNING: cache differs from AF4's "
                  "recorded sha — provenance anomaly; see manifest "
                  "INVALID clause.", flush=True)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(record, indent=2))
        print(f"[af4-r-audit] provenance recorded to {args.manifest}",
              flush=True)
        return

    if args.audit:
        summaries_new = load_summaries(args.run_dir)
        summaries_ref = load_summaries(args.af4_dir)
        integrity = run_integrity(args.run_dir, summaries_new)
        stats_new = compute_arm_stats(summaries_new)
        stats_ref = compute_arm_stats(summaries_ref)
        replay = replay_decision(stats_new)
        ref_replay = replay_decision(stats_ref)
        bands = band_check(stats_ref, stats_new)
        verdict = reproduction_verdict(integrity, replay, bands, ref_replay)
        # Provenance observation ONLY (per the frozen reproduction
        # rule): byte-identity of per-seed values is neither required
        # nor sufficient.
        identical = all(
            sn.get("tasks", {}) == sr.get("tasks", {})
            for sn, sr in zip(
                sorted(summaries_new, key=lambda s: (s["seed"], s["arm"])),
                sorted(summaries_ref, key=lambda s: (s["seed"], s["arm"])),
            )
        ) and len(summaries_new) == len(summaries_ref)
        out = {
            "experiment_id": "EXP-AF-004-R",
            "run_dir": str(args.run_dir),
            "af4_dir": str(args.af4_dir),
            "integrity": integrity,
            "decision_replay_af4r": replay,
            "decision_replay_af4_reference": ref_replay,
            "band_check": bands,
            "byte_identity_observation": identical,
            "reproduction": verdict,
        }
        args.out.write_text(json.dumps(out, indent=2))
        print(json.dumps(verdict, indent=2))
        print(f"[af4-r-audit] audit written to {args.out}", flush=True)
        return

    p.error("one of --prepare or --audit is required")


if __name__ == "__main__":
    main()
