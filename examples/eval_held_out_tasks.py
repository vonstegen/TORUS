"""EXP-RPM-T01 task-robustness harness.

Re-measures all 7 arms + 2 random controls on 4 held-out capability
tasks (hellaswag, winogrande, boolq, openbookqa) at AF2-D / D1p
seed-001. Reuses Stage 1.5 D1p seed-001 adapters (sha256-pinned).

Inputs:
- Stage 1.5 D1p seed-001 adapters (sha256-pinned)
- AF2-D site: model.layers.0.mlp.down_proj
- Damage: Gaussian, sigma=0.20, seed=0

Output:
- runs/r/EXP-RPM-T01/<ts>/per_arm/<arm>/<task>/eval.summary.json
- runs/r/EXP-RPM-T01/<ts>/per_arm/<arm>/<task>/eval.full.json
- runs/r/EXP-RPM-T01/<ts>/held_out_summary.json
- runs/r/EXP-RPM-T01/<ts>/ARTIFACTS.json
"""

import argparse
import gc
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/andrew-jochl/TORUS")

from examples.eval_untrained_arms_v2 import (
    apply_damage_gaussian, load_model, navigate_to_module,
    patch_random_t2, patch_random_lora,
)
from examples.af2_storage_tournament import (
    _patch_module_forward, damage_target_module_gaussian,
)


HELD_OUT_TASKS = [
    {"name": "hellaswag", "metric": "acc_norm,none"},
    {"name": "winogrande", "metric": "acc,none"},
    {"name": "boolq", "metric": "acc,none"},
    {"name": "openbookqa", "metric": "acc_norm,none"},
]

ARMS = ["t2_ternary", "int4_residual", "int8_residual", "lora",
        "dense_adapter", "random_t2_ternary", "random_lora"]

D1P_TS = "20260824T120113Z"
SITE = "model.layers.0.mlp.down_proj"
SIGMA = 0.20
DAMAGE_SEED = 0
BATCH_SIZE = 16


def load_d1p_adapter(arm: str, run_root: Path) -> dict:
    arm_dir = run_root / "af2d" / "seed-001" / arm
    es = json.loads((arm_dir / "eval.summary.json").read_text())
    npz = arm_dir / "adapter.npz"
    return {
        "arm": arm,
        "npz_path": str(npz),
        "eval_summary": es,
        "sha256": subprocess.run(
            ["sha256sum", str(npz)], capture_output=True, text=True,
        ).stdout.split()[0],
    }


def patch_intN_residual(model, npz_path: Path, N_bits: int,
                         column_mask_fraction: float):
    """Re-apply intN residual adapter from npz (matches Stage 5 harness)."""
    import torch.nn.functional as F
    d = np.load(npz_path)
    scale = torch.from_numpy(d["scale"]).to(
        device=next(model.parameters()).device, dtype=torch.float16)
    if N_bits <= 4:
        packed = d["packed"].ravel()
        ub_masked = np.zeros(packed.size * 2, dtype=np.uint8)
        ub_masked[0::2] = packed & 0xF
        ub_masked[1::2] = (packed >> 4) & 0xF
    else:
        ub_masked = d["codes"].ravel()
    kept_count = scale.shape[0]
    in_features = ub_masked.size // kept_count
    parent = navigate_to_module(model, SITE)
    out_features = parent.out_features
    levels = (1 << (N_bits - 1)) - 1
    ub_2d = ub_masked.reshape(kept_count, in_features)
    q_int = torch.from_numpy(ub_2d.astype(np.int64)).to(
        device=next(model.parameters()).device, dtype=torch.float16)
    q_int_centered = q_int - levels
    q_real_2d = q_int_centered * (scale / levels)
    full = torch.zeros(out_features, in_features,
                       device=q_real_2d.device, dtype=torch.float16)
    full[:kept_count, :] = q_real_2d
    keep_mask = torch.zeros(out_features, dtype=torch.bool,
                             device=q_real_2d.device)
    keep_mask[:kept_count] = True

    def residual(x):
        masked = full * keep_mask.unsqueeze(1).to(full.dtype)
        return F.linear(x, masked)
    _patch_module_forward(parent, residual)


def patch_t2_ternary(model, npz_path: Path):
    """T2 ternary: shared with random_t2_ternary (eval_untrained_arms_v2 helper)."""
    patch_random_t2(model, SITE, npz_path)


def patch_dense_adapter(model, npz_path: Path):
    import torch.nn.functional as F
    d = np.load(npz_path)
    W_up = torch.from_numpy(d["W_up"]).to(
        device=next(model.parameters()).device, dtype=torch.float16)
    W_down = torch.from_numpy(d["W_down"]).to(
        device=next(model.parameters()).device, dtype=torch.float16)
    parent = navigate_to_module(model, SITE)

    def residual(x):
        return F.linear(F.linear(x, W_down), W_up)
    _patch_module_forward(parent, residual)


def patch_lora(model, npz_path: Path):
    patch_random_lora(model, SITE, npz_path)


ARM_PATCHERS = {
    "t2_ternary": patch_t2_ternary,
    "int4_residual": (lambda m, p: patch_intN_residual(m, p, 4, 0.5)),
    "int8_residual": (lambda m, p: patch_intN_residual(m, p, 8, 0.25)),
    "lora": patch_lora,
    "dense_adapter": patch_dense_adapter,
    "random_t2_ternary": patch_t2_ternary,
    "random_lora": patch_lora,
}


def run_lm_eval_safe(model, tokenizer, task_name: str, batch_size: int,
                     metric: str, subset: str | None = None):
    """Run lm-eval-harness and return the preferred metric value."""
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    task_arg = task_name
    results = simple_evaluate(model=lm, tasks=[task_arg], batch_size=batch_size)
    task_results = results["results"].get(task_arg, {})
    if not task_results and subset:
        task_arg = f"{task_name}/{subset}"
        results = simple_evaluate(model=lm, tasks=[task_arg], batch_size=batch_size)
        task_results = results["results"].get(task_arg, task_results)
    if metric in task_results:
        value = float(task_results[metric])
    else:
        # Fallback: pick first non-stderr float metric
        for k, v in task_results.items():
            if "stderr" not in k and isinstance(v, (int, float)):
                value = float(v)
                metric = k
                break
        else:
            value = None
            metric = None
    return {"task": task_name, "subset": subset, "metric": metric,
            "value": value, "all": {k: float(v) if isinstance(v, (int, float)) else v
                                    for k, v in task_results.items()}}


def measure_arm(arm: str, run_root: Path, out_dir: Path):
    info = load_d1p_adapter(arm, run_root)
    print(f"[{arm}] loaded: deployed_bytes={info['eval_summary'].get('matched_bytes_actual', '?')} "
          f"sha256={info['sha256'][:12]}...", flush=True)

    arm_dir = out_dir / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    summary = {"arm": arm, "sha256": info["sha256"], "tasks": {}}

    for task in HELD_OUT_TASKS:
        task_name = task["name"]
        metric = task["metric"]
        subset = task.get("subset")
        # Reload + patch fresh per task to avoid state carry-over.
        model, tokenizer = load_model(
            "allenai/OLMo-1B-0724-hf", device="cuda")
        apply_damage_gaussian(model, SITE, SIGMA, seed=DAMAGE_SEED)
        ARM_PATCHERS[arm](model, Path(info["npz_path"]))
        # Set torch.cuda device lock
        torch.cuda.set_device(0)
        t0 = time.time()
        result = run_lm_eval_safe(model, tokenizer, task_name,
                                   BATCH_SIZE, metric, subset)
        dt = time.time() - t0
        result["wall_clock_s"] = dt
        # Save per-task eval
        task_dir = arm_dir / task_name
        task_dir.mkdir(exist_ok=True)
        (task_dir / "eval.summary.json").write_text(json.dumps(result, indent=2))
        (task_dir / "eval.full.json").write_text(json.dumps(result["all"], indent=2))
        summary["tasks"][task_name] = {
            "metric": result["metric"],
            "value": result["value"],
            "wall_clock_s": dt,
        }
        print(f"  [{arm}/{task_name}] {result['metric']}={result['value']} ({dt:.1f}s)", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    (arm_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def aggregate(per_arm: dict) -> dict:
    """Build per-task aggregates: trained mean/stderr vs random means."""
    out = {"by_task": {}}
    for task in HELD_OUT_TASKS:
        tn = task["name"]
        by_arm = {}
        for arm, r in per_arm.items():
            t = r.get("tasks", {}).get(tn, {})
            v = t.get("value")
            if v is not None:
                by_arm[arm] = v
        if not by_arm:
            continue
        # Trained arms
        trained = [v for k, v in by_arm.items() if k.endswith("_ternary") or k.endswith("_residual") or k in ("lora", "dense_adapter")]
        random_arms = [v for k, v in by_arm.items() if k.startswith("random_")]
        trained_mean = sum(trained) / len(trained) if trained else None
        trained_max = max(trained) if trained else None
        t2_v = by_arm.get("t2_ternary")
        rand_t2_v = by_arm.get("random_t2_ternary")
        rand_lora_v = by_arm.get("random_lora")
        # Naive stderr estimate: range/4 over the 5 trained arms
        trained_stderr = (max(trained) - min(trained)) / 4 if len(trained) >= 2 else None
        out["by_task"][tn] = {
            "by_arm": by_arm,
            "trained_mean": trained_mean,
            "trained_max": trained_max,
            "trained_stderr_proxy": trained_stderr,
            "t2_minus_random_t2": (t2_v - rand_t2_v) if (t2_v is not None and rand_t2_v is not None) else None,
            "t2_minus_random_lora": (t2_v - rand_lora_v) if (t2_v is not None and rand_lora_v is not None) else None,
            "t2_minus_trained_mean": (t2_v - trained_mean) if (t2_v is not None and trained_mean is not None) else None,
        }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--run-root",
                   default=f"/home/andrew-jochl/TORUS/runs/r/EXP-RPM-D1p/{D1P_TS}")
    p.add_argument("--out-root",
                   default=f"/home/andrew-jochl/TORUS/runs/r/EXP-RPM-T01/{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}")
    args = p.parse_args()
    run_root = Path(args.run_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    arms = args.arms.split(",")
    per_arm = {}
    for arm in arms:
        try:
            per_arm[arm] = measure_arm(arm, run_root, out_root)
        except Exception as e:
            import traceback
            traceback.print_exc()
            per_arm[arm] = {"error": str(e)}
        # Save partial aggregate after each arm
        agg = aggregate(per_arm)
        (out_root / "held_out_summary.json").write_text(json.dumps(agg, indent=2))
    # ARTIFACTS.json
    artifacts = {"experiment": "EXP-RPM-T01",
                 "ts": out_root.name,
                 "items": []}
    for arm, r in per_arm.items():
        if "error" in r:
            continue
        artifacts["items"].append({
            "arm": arm,
            "adapter_npz_sha256": r.get("sha256"),
            "summary": str(out_root / arm / "summary.json"),
            "tasks": [str(out_root / arm / t["name"]) for t in HELD_OUT_TASKS],
        })
    (out_root / "ARTIFACTS.json").write_text(json.dumps(artifacts, indent=2))
    print(f"DONE. Summary at {out_root / 'held_out_summary.json'}")


if __name__ == "__main__":
    main()