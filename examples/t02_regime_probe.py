"""EXP-RPM-T02-PROBE — AF5-regime probe (damaged-base held-out eval).

Evaluates the DAMAGED BASE ONLY (no training, no adapters) on the
four T01 held-out tasks (hellaswag, winogrande, boolq, openbookqa)
across the preregistered AF2-D TWN severity band {fp16, D1p..D5p}
plus the FP16 baseline, then applies the frozen rules from
research/residual-pareto/experiments/EXP-RPM-T02-PROBE/manifest.yaml:

  QUALIFIES iff >= 1 of 4 tasks drops by >= max(3 x stderr_max, 0.02)
  below FP16.

  Verification gate: D1p near-FP16 (|FP16 - D1p| <= 2 x stderr on
  >= 3 of 4 tasks) — T01's diagnosis; violation -> INVALID.

Damage: TWN via the frozen tournament driver's
damage_target_module (group_size=128, calibrate_norm=False),
deterministic.

Rule functions are pure (dict-only) so tests import them without
torch. The CLI needs the Legion environment (torch + lm-eval).

Usage:
  .venv/bin/python examples/t02_regime_probe.py \
      --out-dir runs/r/EXP-RPM-T02-PROBE/<ts> --device cuda:0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DROP_MIN_ABS = 0.02
DROP_SIGMA = 3.0
GATE_SIGMA = 2.0
GATE_TASKS_MIN = 3

REGIMES = [
    {"id": "fp16", "threshold": None},
    {"id": "D1p", "threshold": 1.0},
    {"id": "D2p", "threshold": 0.9},
    {"id": "D3p", "threshold": 0.8},
    {"id": "D4p", "threshold": 0.7},
    {"id": "D5p", "threshold": 0.6},
]

TASKS = [
    ("hellaswag", "acc_norm"),
    ("winogrande", "acc"),
    ("boolq", "acc"),
    ("openbookqa", "acc_norm"),
]


# ---- frozen rule functions (pure) -------------------------------------

def per_task_drop(fp16_score: float, fp16_stderr: float,
                  regime_score: float, regime_stderr: float) -> float:
    """Capability drop of a regime below FP16 on one task.

    Returns max(0, fp16 - regime) if it exceeds the frozen bar,
    else 0.0.
    """
    stderr_max = max(fp16_stderr, regime_stderr, 0.0)
    bar = max(DROP_SIGMA * stderr_max, DROP_MIN_ABS)
    drop = fp16_score - regime_score
    return drop if drop >= bar else 0.0


def regime_qualifies(fp16: dict[str, tuple[float, float]],
                     regime: dict[str, tuple[float, float]],
                     tasks: list[str]) -> dict:
    """Frozen qualify rule: >= 1 task drops by >= max(3*stderr, 0.02)."""
    drops = {}
    for task in tasks:
        f_s, f_e = fp16[task]
        r_s, r_e = regime[task]
        drops[task] = per_task_drop(f_s, f_e, r_s, r_e)
    qualifying = any(d > 0.0 for d in drops.values())
    return {
        "qualifying": qualifying,
        "per_task_drops": drops,
        "summed_drop": sum(drops.values()),
    }


def d1p_gate_ok(fp16: dict[str, tuple[float, float]],
                d1p: dict[str, tuple[float, float]],
                tasks: list[str]) -> dict:
    """Verification gate: D1p near-FP16 on >= 3 of 4 tasks."""
    ok_tasks = []
    for task in tasks:
        f_s, f_e = fp16[task]
        d_s, d_e = d1p[task]
        se = max(f_e, d_e, 0.0)
        if abs(f_s - d_s) <= GATE_SIGMA * se:
            ok_tasks.append(task)
    return {
        "gate_ok": len(ok_tasks) >= GATE_TASKS_MIN,
        "n_tasks_near_fp16": len(ok_tasks),
        "near_fp16_tasks": ok_tasks,
    }


def select_regime(results: dict[str, dict]) -> dict:
    """Frozen selection: qualifying regime with largest summed drop;
    ties -> more severe threshold (later in REGIMES order wins)."""
    best = None
    for regime_id, r in results.items():
        if regime_id == "fp16":
            continue
        v = r["verdict"]
        if not v["qualifying"]:
            continue
        if best is None or v["summed_drop"] >= best["verdict"]["summed_drop"]:
            best = r
    return {
        "selected": (best["regime_id"] if best else None),
        "qualifying_regimes": sorted(
            r["regime_id"] for r in results.values()
            if r.get("verdict", {}).get("qualifying")),
    }


def build_probe_summary(results: dict[str, dict],
                        task_order: list[str]) -> dict:
    fp16 = results["fp16"]["scores"]
    d1p = results["D1p"]["scores"]
    gate = d1p_gate_ok(fp16, d1p, task_order)
    selection = select_regime(results)
    per_regime = {
        rid: {
            "scores": r["scores"],
            "verdict": r["verdict"],
        }
        for rid, r in results.items()
    }
    return {
        "tasks": task_order,
        "regimes": per_regime,
        "d1p_gate": gate,
        "selection": selection,
        "probe_valid": gate["gate_ok"],
        "decision": (
            "REGIMES_FOUND" if gate["gate_ok"] and selection["selected"]
            else ("INVALID" if not gate["gate_ok"] else "NO_REGIME")
        ),
    }


def _eval_one(model, tokenizer, task_name: str, metric: str,
              batch_size: int) -> tuple[float, float]:
    """Run lm-eval on one task; return (score, stderr).

    Mirrors eval_untrained_arms_v2.py (the T01 instrument): HFLM +
    simple_evaluate + `,none`-suffixed preferred keys.
    """
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    lm = HFLM(pretrained=model, tokenizer=tokenizer,
              batch_size=batch_size)
    results = simple_evaluate(model=lm, tasks=[task_name],
                              batch_size=batch_size)
    t_results = results["results"][task_name]
    key = f"{metric},none"
    value = float(t_results[key])
    stderr_key = f"{metric}_stderr,none"
    stderr = float(t_results.get(stderr_key, 0.0))
    return value, stderr



def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    import torch
    from transformers import AutoTokenizer

    from examples.af2_storage_tournament import (
        build_base,
        damage_target_module,
        resolve_target_module,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "allenai/OLMo-1B-0724-hf")
    results: dict[str, dict] = {}
    for regime in REGIMES:
        rid = regime["id"]
        model = build_base("allenai/OLMo-1B-0724-hf",
                           dtype="float16", device=args.device)
        if regime["threshold"] is not None:
            target = resolve_target_module(
                model, "model.layers.0.mlp.down_proj")
            damage_target_module(target, group_size=128,
                                 threshold=regime["threshold"])
        scores: dict[str, tuple[float, float]] = {}
        for task_name, metric in TASKS:
            score, stderr = _eval_one(model, tokenizer, task_name,
                                      metric, args.batch_size)
            scores[task_name] = (score, stderr)
            out = args.out_dir / rid / task_name
            out.mkdir(parents=True, exist_ok=True)
            (out / "eval.summary.json").write_text(json.dumps({
                "task": task_name, "metric": metric,
                "value": score, "stderr": stderr,
            }, indent=2))
            print(f"[t02-probe] {rid} {task_name}: "
                  f"{score:.4f} +/- {stderr:.4f}", flush=True)
        results[rid] = {"regime_id": rid, "scores": scores,
                        "verdict": {}}
        del model
        torch.cuda.empty_cache()

    fp16_scores = results["fp16"]["scores"]
    for rid, r in results.items():
        if rid == "fp16":
            r["verdict"] = {"qualifying": False, "per_task_drops": {},
                            "summed_drop": 0.0}
        else:
            r["verdict"] = regime_qualifies(
                fp16_scores, r["scores"], [t for t, _ in TASKS])

    summary = build_probe_summary(results, [t for t, _ in TASKS])
    (args.out_dir / "probe_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
