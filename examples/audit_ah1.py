"""EXP-A-H1 auditor: evaluate frozen thresholds from the two arm records.

Applies the preregistered bars (manifest, frozen 2026-08-29):

  integrity:  both arms' summaries + histories present and finite;
              steps >= budget; parity |step-0 loss diff| <= 0.1 nats.
  thresholds: hadamard wikitext ppl <= 0.97 x control AND
              arc_easy >= control - 0.03 AND
              lambada_openai >= control - 0.02 -> PASS, else FAIL.
  kills:      abort.json reason: nan_loss -> INVALID; loss_gap -> FAIL.

Output: audit.json in the run directory. Exit code 0 on DECIDED
(PASS or FAIL); 1 on INVALID.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PPL_RATIO = 0.97
ARC_MARGIN = 0.03
LAMBADA_MARGIN = 0.02
BUDGET_STEPS = 12_500            # 200M tokens (amended pre-run)
PARITY_TOLERANCE = 0.1


def load_arm(run_dir: Path, arm: str) -> dict:
    rec: dict = {"arm": arm}
    sdir = run_dir / arm
    rec["summary"] = json.loads((sdir / "summary.json").read_text())
    rec["eval"] = json.loads((sdir / "eval.summary.json").read_text())
    p = sdir / f"parity_{arm}.json"
    rec["parity"] = json.loads(p.read_text()) if p.exists() else None
    ab = sdir / "abort.json"
    rec["abort"] = json.loads(ab.read_text()) if ab.exists() else None
    rec["history_finite"] = _history_finite(sdir / "history.jsonl")
    return rec


def _history_finite(path: Path) -> bool:
    if not path.exists():
        return False
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        if not (rec["loss"] == rec["loss"] and rec["loss"] != float("inf")):
            return False
    return True


def evaluate_thresholds(arms: dict) -> dict:
    """Evaluate the frozen bars on loaded arm records (pure function)."""
    c = arms["control"]["eval"]
    h = arms["hadamard"]["eval"]
    bars = {
        "ppl_ratio": {"value": h["wikitext"] / c["wikitext"],
                      "bar": PPL_RATIO,
                      "pass": h["wikitext"] <= PPL_RATIO * c["wikitext"]},
        "arc_margin": {"value": h["arc_easy"] - c["arc_easy"],
                       "bar": -ARC_MARGIN,
                       "pass": h["arc_easy"] >= c["arc_easy"] - ARC_MARGIN},
        "lambada_margin": {"value": h["lambada_openai"]
                                    - c["lambada_openai"],
                           "bar": -LAMBADA_MARGIN,
                           "pass": h["lambada_openai"]
                                   >= c["lambada_openai"]
                                   - LAMBADA_MARGIN},
    }
    return bars


def audit(run_dir: Path) -> dict:
    arms = {a: load_arm(run_dir, a) for a in ["control", "hadamard"]}
    out: dict = {"run": str(run_dir), "arms": arms}

    # integrity
    integrity = []
    for a, rec in arms.items():
        if not rec["history_finite"]:
            integrity.append(f"{a}: non-finite history")
        if rec["summary"]["steps"] < BUDGET_STEPS:
            integrity.append(f"{a}: steps {rec['summary']['steps']} "
                             f"< {BUDGET_STEPS}")
        if rec["parity"] is None:
            integrity.append(f"{a}: parity missing")
    pc = arms["control"]["parity"]["step0_loss"]
    ph = arms["hadamard"]["parity"]["step0_loss"]
    parity_gap = abs(pc - ph)
    out["parity"] = {"control": pc, "hadamard": ph, "gap": parity_gap,
                     "pass": parity_gap <= PARITY_TOLERANCE}
    if not out["parity"]["pass"]:
        integrity.append(f"parity gap {parity_gap:.4f} > "
                         f"{PARITY_TOLERANCE}")
    out["integrity_ok"] = not integrity
    out["integrity_failures"] = integrity

    # kills
    kills = []
    for rec in arms.values():
        if rec["abort"]:
            kills.append(rec["abort"])

    out["bars"] = evaluate_thresholds(arms)

    if integrity or any(k["reason"] == "nan_loss" for k in kills):
        out["verdict"] = "INVALID"
    elif any(k["reason"] == "loss_gap" for k in kills):
        out["verdict"] = "DECIDED"
        out["decision"] = "FAIL"
        out["decision_reason"] = "live loss-gap abort fired"
    elif all(b["pass"] for b in out["bars"].values()):
        out["verdict"] = "DECIDED"
        out["decision"] = "PASS"
        out["decision_reason"] = "all frozen bars pass; no kill fired"
    else:
        out["verdict"] = "DECIDED"
        out["decision"] = "FAIL"
        out["decision_reason"] = "frozen bar(s) missed"
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    result = audit(Path(args.run_dir))
    out_path = Path(args.out) if args.out else \
        Path(args.run_dir) / "audit.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items()
                      if k in ("verdict", "decision", "decision_reason",
                               "integrity_ok", "parity", "bars")},
                     indent=2))
    print(f"[ah1-audit] audit written to {out_path}", flush=True)
    raise SystemExit(0 if result["verdict"] == "DECIDED" else 1)


if __name__ == "__main__":
    main()
