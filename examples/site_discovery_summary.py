"""EXP-RPM-SITE-DISCOVERY sweep summarizer (frozen decision rules).

Reads runs/r/EXP-RPM-SITE-DISCOVERY/<timestamp>/<site-id>/ cells and
applies the preregistered rules from the manifest:

  QUALIFYING (frozen Stage 2 v2 rule, unchanged):
    >= 3 knob values produce ppl in distinct reproducibility bands
    (round(ppl,0)) separated by >= 1 ppl unit AND spanning >= 2 ppl
    units total.

  CANDIDATE = QUALIFYING AND max per-knob ppl_mean >= 100.

  Verification gates:
    - zero-knob cell (sigma=0.0 / baseline) per site must reproduce
      the FP16 baseline band 13 (|ppl - 13.0932| <= 0.05).
    - reference cells: ref-gauss-v-L0 sigma=0.5 in [438, 440];
      ref-twn-d-L0 thr=0.7 in [400, 460].

Layout: <site-id>/sigma-<v>/seed-<n>/pre_train_eval.json (Gaussian),
<thr-<v>/seed-<n>/pre_train_eval.json (TWN),
baseline/seed-001/pre_train_eval.json (TWN-site verification).

Pure Python (json/pathlib/statistics only) so tests can import it
without torch.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

FP16_BASELINE_PPL = 13.0932
FP16_BASELINE_TOL = 0.05
CANDIDATE_MIN_PPL = 100.0

# (site_id, mechanism, knob) -> expected band [lo, hi]; frozen pilot
# values: gauss-v-L0 sigma=0.5 -> 439.2520; twn-d-L0 thr=0.7 -> 429.55.
REFERENCE_CELLS = {
    "ref-gauss-v-L0": {
        "mechanism": "gaussian",
        "knob": 0.5,
        "band": [438.0, 440.0],
        "frozen_value": 439.2520,
    },
    "ref-twn-d-L0": {
        "mechanism": "twn",
        "knob": 0.7,
        "band": [400.0, 460.0],
        "frozen_value": 429.55,
    },
}


def parse_knob(dir_name: str) -> float | None:
    """'sigma-00_05' -> 0.05; 'thr-01_00' -> 1.0; else None."""
    for prefix in ("sigma-", "thr-"):
        if dir_name.startswith(prefix):
            body = dir_name[len(prefix):]
            try:
                return float(body.replace("_", "."))
            except ValueError:
                return None
    return None


def load_ppl(path: Path) -> float | None:
    try:
        d = json.loads(path.read_text())
    except Exception:
        return None
    wt = d.get("tasks", {}).get("wikitext")
    if not wt:
        return None
    try:
        return float(wt["value"])
    except (KeyError, TypeError, ValueError):
        return None


def collect_knob_rows(site_dir: Path) -> dict[float, list[float]]:
    """Map knob value -> [ppl per completed seed cell]."""
    rows: dict[float, list[float]] = {}
    for knob_dir in site_dir.iterdir():
        if not knob_dir.is_dir():
            continue
        knob = parse_knob(knob_dir.name)
        if knob is None:
            continue
        ppls = []
        for pte in knob_dir.rglob("pre_train_eval.json"):
            ppl = load_ppl(pte)
            if ppl is not None:
                ppls.append(ppl)
        if ppls:
            rows[knob] = ppls
    return rows


def summarize_knobs(rows: dict[float, list[float]]) -> list[dict]:
    out = []
    for knob in sorted(rows):
        ppls = rows[knob]
        out.append({
            "knob": knob,
            "n_seeds": len(ppls),
            "ppl_mean": statistics.fmean(ppls),
            "ppl_stderr": (statistics.stdev(ppls) / len(ppls) ** 0.5
                           if len(ppls) > 1 else 0.0),
            "ppl_min": min(ppls),
            "ppl_max": max(ppls),
        })
    return out


def apply_qualifying_rule(rows: list[dict]) -> tuple[bool, dict]:
    """Frozen Stage 2 v2 QUALIFYING rule on per-knob means."""
    if not rows:
        return False, {
            "n_distinct_ppl_bands": 0,
            "ppl_span": None,
            "kill_criteria": (
                "QUALIFYING iff >= 3 knob values produce ppl in "
                "distinct reproducibility bands (round(ppl,0)) "
                "separated by >= 1 ppl unit AND spanning >= 2 ppl "
                "units total"
            ),
        }
    means = [r["ppl_mean"] for r in rows]
    lo, hi = min(means), max(means)
    span = hi - lo
    bands = {round(m) for m in means}
    n_distinct = len(bands)
    bands_sorted = sorted(bands)
    separated = all(
        bands_sorted[i + 1] - bands_sorted[i] >= 1
        for i in range(len(bands_sorted) - 1)
    )
    qualifying = n_distinct >= 3 and span >= 2.0 and separated
    return qualifying, {
        "n_distinct_ppl_bands": n_distinct,
        "ppl_span": span,
        "bands_separated": separated,
        "kill_criteria": (
            "QUALIFYING iff >= 3 knob values produce ppl in distinct "
            "reproducibility bands (round(ppl,0)) separated by >= 1 "
            "ppl unit AND spanning >= 2 ppl units total"
        ),
    }


def baseline_verdict(site_dir: Path) -> tuple[bool, float | None]:
    """Zero-knob (sigma=0.0 or baseline/) cell must land in band 13."""
    ppls: list[float] = []
    sigma0 = site_dir / "sigma-00_00"
    if sigma0.is_dir():
        for pte in sigma0.rglob("pre_train_eval.json"):
            ppl = load_ppl(pte)
            if ppl is not None:
                ppls.append(ppl)
    base = site_dir / "baseline"
    if base.is_dir():
        for pte in base.rglob("pre_train_eval.json"):
            ppl = load_ppl(pte)
            if ppl is not None:
                ppls.append(ppl)
    if not ppls:
        return False, None
    mean = statistics.fmean(ppls)
    ok = abs(mean - FP16_BASELINE_PPL) <= FP16_BASELINE_TOL
    return ok, mean


def candidate_verdict(qualifying: bool, max_ppl: float | None) -> dict:
    if not qualifying:
        return {"candidate": False, "reason": "not qualifying"}
    if max_ppl is None or max_ppl < CANDIDATE_MIN_PPL:
        return {
            "candidate": False,
            "reason": (
                "mild axis; nothing-to-recover risk (EXP-RPM-T01 and "
                "the L0-v / L15 tournament diagnoses)"
            ),
        }
    return {"candidate": True, "reason": "qualifying AND max ppl >= 100"}


def summarize_site(site_dir: Path, site_id: str) -> dict:
    """Per-site summary: knob -> ppl map + frozen rules applied."""
    rows_map = collect_knob_rows(site_dir)
    rows = summarize_knobs(rows_map)
    qualifying, rule = apply_qualifying_rule(rows)
    baseline_ok, baseline_ppl = baseline_verdict(site_dir)
    max_ppl = max((r["ppl_mean"] for r in rows), default=None)
    verdict = candidate_verdict(qualifying, max_ppl)

    n_cells = sum(len(rows_map[k]) for k in rows_map)
    summary = {
        "site_id": site_id,
        "n_knobs": len(rows),
        "n_seed_cells": n_cells,
        "knob_to_ppl": rows,
        "ppl_span": rule["ppl_span"],
        "n_distinct_ppl_bands": rule["n_distinct_ppl_bands"],
        "bands_separated": rule["bands_separated"],
        "qualifying": qualifying,
        "baseline_ok": baseline_ok,
        "baseline_ppl": baseline_ppl,
        "baseline_expect": FP16_BASELINE_PPL,
        "max_ppl": max_ppl,
        "candidate": verdict["candidate"],
        "candidate_reason": verdict["reason"],
        "rule": rule["kill_criteria"],
    }
    (site_dir / "site_cal_summary.json").write_text(
        json.dumps(summary, indent=2))
    return summary


def check_reference_cells(run_dir: Path) -> list[dict]:
    """Whole-run environment verification against frozen pilot values."""
    out = []
    for site_id, spec in REFERENCE_CELLS.items():
        site_dir = run_dir / site_id
        rows_map = collect_knob_rows(site_dir)
        ppls = rows_map.get(spec["knob"], [])
        ok = any(spec["band"][0] <= p <= spec["band"][1] for p in ppls)
        got = ppls[0] if ppls else None
        out.append({
            "site_id": site_id,
            "knob": spec["knob"],
            "frozen_value": spec["frozen_value"],
            "band": spec["band"],
            "observed": got,
            "ok": ok,
        })
    return out


def summarize_sweep(run_dir: Path,
                    site_ids: list[str] | None = None) -> dict:
    if site_ids is None:
        site_ids = sorted({
            d.name for d in run_dir.iterdir()
            if d.is_dir() and d.name.startswith(("gauss-", "twn-"))
        })

    refs = check_reference_cells(run_dir)
    sites = []
    for site_id in site_ids:
        site_dir = run_dir / site_id
        if not site_dir.is_dir():
            sites.append({"site_id": site_id, "error": "missing dir"})
            continue
        sites.append(summarize_site(site_dir, site_id))

    candidates = [s for s in sites if s.get("candidate")]
    qualifying_non_candidates = [
        s for s in sites
        if s.get("qualifying") and not s.get("candidate")
    ]
    invalid_sites = [s for s in sites
                     if s.get("error") or s.get("baseline_ok") is False]
    refs_ok = all(r["ok"] for r in refs)

    # Frozen candidate priority: attention-category (v_proj) candidate
    # with the largest span first.
    def _key(s):
        is_attn = "-v-" in s["site_id"]
        span = s.get("ppl_span") or 0.0
        return (not is_attn, -span)

    candidates_sorted = sorted(candidates, key=_key)

    summary = {
        "run_dir": str(run_dir),
        "reference_cells": refs,
        "references_ok": refs_ok,
        "run_valid": refs_ok,
        "n_sites": len(sites),
        "n_candidates": len(candidates),
        "candidates": [s["site_id"] for s in candidates_sorted],
        "candidate_priority": [
            {"site_id": s["site_id"],
             "ppl_span": s.get("ppl_span"),
             "category": "attention v_proj" if "-v-" in s["site_id"]
             else "mlp down_proj"}
            for s in candidates_sorted
        ],
        "qualifying_non_candidates": [
            {"site_id": s["site_id"], "reason": s.get("candidate_reason"),
             "max_ppl": s.get("max_ppl")}
            for s in qualifying_non_candidates
        ],
        "invalid_sites": invalid_sites,
        "decision": (
            "CANDIDATES_FOUND" if candidates and refs_ok
            else ("INVALID" if not refs_ok else "NO_SECOND_SITE")
        ),
    }
    (run_dir / "sweep_summary.json").write_text(
        json.dumps(summary, indent=2))
    return summary
