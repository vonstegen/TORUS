"""EXP-RPM-T02 — AF5 tournament at D5p (TWN thr 0.6).

Evaluates the 7 arms (5 trained, sha256-pinned Stage 1.5 D5p
adapters + 2 random controls) on the 4 held-out tasks over a
TWN-damaged base, then applies the FROZEN T01 thresholds from
research/residual-pareto/experiments/EXP-RPM-T02/manifest.yaml:

  PASS iff ALL of:
    1. z(t2, random_t2) >= +1 sd-of-difference on >= 3/4 tasks
    2. z(t2, random_lora) >= +1 sd-of-difference on >= 3/4 tasks
    3. t2 >= best trained comparator on >= 3/4 tasks (ties ok)
    4. t2 > random_t2 mean on >= 3/4 tasks (above chance)
  else FAIL (recording missed rules and fired triggers:
    t2 < random_t2 on >= 2/4; t2 < random_lora - 2sd on >= 2/4).

Verification gate: the damaged base (no adapter) must reproduce the
probe's frozen D5p scores within 2 x stderr on >= 3/4 tasks.

sd-of-difference = sqrt(se_a^2 + se_b^2) from lm-eval per-task
stderr (frozen formulation).

Pure rule functions are torch-free; the CLI needs Legion
(torch + lm-eval). Reuses the T01 instrument's arm loaders,
patchers, and eval helpers (examples/eval_held_out_tasks.py).

Usage:
  .venv/bin/python examples/t02_af5_tournament.py \
      --adapters-root runs/r/EXP-RPM-D5p/20260824T144239Z \
      --out-dir runs/r/EXP-RPM-T02/<ts> --device cuda:0
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

# Frozen probe D5p damaged-base scores (EXP-RPM-T02-PROBE
# 20260830T204622Z, probe_summary.json): (score, stderr).
PROBE_D5P = {
    "hellaswag": (0.4256, 0.0049),
    "winogrande": (0.5501, 0.0140),
    "boolq": (0.5691, 0.0087),
    "openbookqa": (0.2980, 0.0205),
}

PASS_SIGMA = 1.0
FAIL_SIGMA = 2.0
GATE_SIGMA = 2.0
N_TASKS_MIN = 3
TRAINED_COMPARATORS = ("int4_residual", "int8_residual",
                       "lora", "dense_adapter")
ARMS = ("t2_ternary", "int4_residual", "int8_residual", "lora",
        "dense_adapter", "random_t2_ternary", "random_lora")


# ---- frozen rule functions (pure) -------------------------------------

def sd_of_difference(se_a: float | None, se_b: float | None) -> float:
    return math.sqrt((se_a or 0.0) ** 2 + (se_b or 0.0) ** 2)


def base_gate_ok(base: dict[str, tuple[float, float | None]],
                 tasks: list[str]) -> dict:
    """Damaged-base verification gate vs the frozen probe D5p scores."""
    ok_tasks = []
    for task in tasks:
        b_s, b_e = base[task]
        p_s, p_e = PROBE_D5P[task]
        se = max(b_e or 0.0, p_e)
        if abs(b_s - p_s) <= GATE_SIGMA * se:
            ok_tasks.append(task)
    return {
        "gate_ok": len(ok_tasks) >= N_TASKS_MIN,
        "n_tasks_in_band": len(ok_tasks),
        "in_band_tasks": ok_tasks,
    }


def apply_thresholds(by_task: dict[str, dict]) -> dict:
    """Frozen T01 pass/fail thresholds on per-task arm scores."""
    tasks = list(by_task.keys())
    r1 = r2 = r3 = r4 = 0
    fail1 = fail2 = 0
    per_task = {}
    for task in tasks:
        d = by_task[task]
        t2, se_t2 = d["t2_ternary"]
        rt2, se_rt2 = d["random_t2_ternary"]
        rlora, se_rlora = d["random_lora"]
        best = max(by_task[task][c][0] for c in TRAINED_COMPARATORS)
        z_rt2 = (t2 - rt2) / sd_of_difference(se_t2, se_rt2) \
            if sd_of_difference(se_t2, se_rt2) > 0 else 0.0
        z_rlora = (t2 - rlora) / sd_of_difference(se_t2, se_rlora) \
            if sd_of_difference(se_t2, se_rlora) > 0 else 0.0
        per_task[task] = {
            "z_vs_random_t2": z_rt2,
            "z_vs_random_lora": z_rlora,
            "t2_beats_best_trained": t2 >= best,
            "t2_above_chance": t2 > rt2,
        }
        r1 += z_rt2 >= PASS_SIGMA
        r2 += z_rlora >= PASS_SIGMA
        r3 += t2 >= best
        r4 += t2 > rt2
        fail1 += t2 < rt2
        fail2 += t2 < rlora - FAIL_SIGMA * sd_of_difference(se_t2,
                                                            se_rlora)
    rules = {
        "r1_t2_vs_random_t2_ge_1sd": (r1 >= N_TASKS_MIN, r1),
        "r2_t2_vs_random_lora_ge_1sd": (r2 >= N_TASKS_MIN, r2),
        "r3_t2_wins_or_ties_best_trained": (r3 >= N_TASKS_MIN, r3),
        "r4_t2_above_chance": (r4 >= N_TASKS_MIN, r4),
    }
    triggers = {
        "f1_t2_below_chance": (fail1 >= 2, fail1),
        "f2_t2_far_below_random_lora": (fail2 >= 2, fail2),
    }
    passed = all(v[0] for v in rules.values())
    return {
        "per_task": per_task,
        "rules": rules,
        "fail_triggers": triggers,
        "decision": "PASS" if passed else "FAIL",
    }


# ---- CLI --------------------------------------------------------------

def _extract_stderr(all_results: dict, metric: str) -> float | None:
    key = metric.replace(",none", "_stderr,none")
    if key in all_results:
        return float(all_results[key])
    for k, v in all_results.items():
        if "stderr" in k:
            return float(v)
    return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapters-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    import torch

    from examples.af2_storage_tournament import (
        damage_target_module,
        resolve_target_module,
    )
    from examples.eval_held_out_tasks import (
        ARM_PATCHERS,
        HELD_OUT_TASKS,
        load_d1p_adapter,
        run_lm_eval_safe,
    )
    from examples.eval_untrained_arms_v2 import load_model

    SITE = "model.layers.0.mlp.down_proj"
    BATCH = 16
    TWN_THRESHOLD = 0.6

    def _fresh_damaged():
        model, tokenizer = load_model("allenai/OLMo-1B-0724-hf",
                                      device=args.device)
        target = resolve_target_module(model, SITE)
        damage_target_module(target, group_size=128,
                             threshold=TWN_THRESHOLD)
        return model, tokenizer

    # Verification gate: damaged base, no adapter.
    model, tokenizer = _fresh_damaged()
    base_scores: dict[str, tuple[float, float | None]] = {}
    for task in HELD_OUT_TASKS:
        res = run_lm_eval_safe(model, tokenizer, task["name"], BATCH,
                               task["metric"], task.get("subset"))
        se = _extract_stderr(res["all"], res["metric"] or task["metric"])
        base_scores[task["name"]] = (res["value"], se)
        out = args.out_dir / "base" / task["name"]
        out.mkdir(parents=True, exist_ok=True)
        (out / "eval.summary.json").write_text(json.dumps(res, indent=2))
        (out / "eval.full.json").write_text(
            json.dumps(res["all"], indent=2))
        print(f"[t02] base {task['name']}: {res['value']} +/- {se}",
              flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    gate = base_gate_ok(base_scores, [t["name"] for t in HELD_OUT_TASKS])

    per_arm = {}
    artifacts = {}
    for arm in ARMS:
        info = load_d1p_adapter(arm, args.adapters_root)
        artifacts[arm] = info["sha256"]
        arm_dir = args.out_dir / "per_arm" / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        summary = {"arm": arm, "sha256": info["sha256"], "tasks": {}}
        for task in HELD_OUT_TASKS:
            model, tokenizer = _fresh_damaged()
            ARM_PATCHERS[arm](model, Path(info["npz_path"]))
            t0 = time.time()
            res = run_lm_eval_safe(model, tokenizer, task["name"],
                                   BATCH, task["metric"],
                                   task.get("subset"))
            dt = time.time() - t0
            se = _extract_stderr(res["all"],
                                 res["metric"] or task["metric"])
            task_dir = arm_dir / task["name"]
            task_dir.mkdir(exist_ok=True)
            (task_dir / "eval.summary.json").write_text(
                json.dumps(res, indent=2))
            (task_dir / "eval.full.json").write_text(
                json.dumps(res["all"], indent=2))
            summary["tasks"][task["name"]] = {
                "metric": res["metric"], "value": res["value"],
                "stderr": se, "wall_clock_s": dt,
            }
            print(f"[t02] {arm} {task['name']}: {res['value']} "
                  f"+/- {se} ({dt:.1f}s)", flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        (arm_dir / "summary.json").write_text(json.dumps(summary,
                                                         indent=2))
        per_arm[arm] = summary

    by_task: dict[str, dict] = {}
    for task in HELD_OUT_TASKS:
        name = task["name"]
        by_task[name] = {
            arm: (per_arm[arm]["tasks"][name]["value"],
                  per_arm[arm]["tasks"][name]["stderr"])
            for arm in ARMS
        }
    verdict = apply_thresholds(by_task)
    held_out = {
        "experiment": "EXP-RPM-T02",
        "site": SITE,
        "damage": {"mechanism": "TWN", "threshold": TWN_THRESHOLD,
                   "group_size": 128},
        "base_gate": gate,
        "by_task": by_task,
        "threshold_verdict": verdict,
    }
    (args.out_dir / "held_out_summary.json").write_text(
        json.dumps(held_out, indent=2))
    (args.out_dir / "ARTIFACTS.json").write_text(json.dumps({
        "experiment": "EXP-RPM-T02",
        "adapters_root": str(args.adapters_root),
        "adapter_sha256": artifacts,
    }, indent=2))
    print(json.dumps(held_out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
