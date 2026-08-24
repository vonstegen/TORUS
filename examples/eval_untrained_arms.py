"""Post-hoc eval of Stage 1 untrained-control arms (random_t2_ternary,
random_lora) across all regimes x seeds.

Background
----------
Stage 1's driver (examples/af2_storage_tournament.py) skips
lm-eval-harness on `is_untrained` arms because there's no training
progression to measure. But RPM-002 (trained-vs-random capability
separation, cross-regime) and RPM-006 (trained-vs-random z-score on
each metric) require the random arm's eval numbers, not just the
trained arm's. This script re-loads each random adapter, re-applies it
on the matching damaged base, and runs lm-eval-harness to fill in
the missing `tasks` field in each eval.summary.json.

Output
------
Updates the existing
  runs/r/EXP-RPM-D{0..5}/<ts>/af2d/seed-N/{random_t2_ternary,random_lora}/
    eval.summary.json (only `tasks` filled)
    eval.full.json (full lm-eval output, for auditability)

Driver SHA: frozen by Stage 1 (commit `692e8ee`). This script uses
the same packing/unpacking convention as the driver; verified by
re-computing the row-residual norm on a sample adapter.

Usage
-----
  PYTHONPATH=. .venv/bin/python examples/eval_untrained_arms.py \
    --regimes 0,1,2,3,4,5 \
    --arms random_t2_ternary,random_lora \
    --tasks wikitext,arc_easy,lambada_openai \
    --batch_size 16
"""
from __future__ import annotations
import argparse
import gc
import json
import sys
import time
from pathlib import Path

sys.modules.setdefault("triton", None)

import numpy as np
import torch
import torch.nn.functional as F

from examples.af2_storage_tournament import (  # noqa: E402
    _patch_module_forward,
    damage_target_module,
)


# ---------- adapter unpacking ----------------------------------------------


def unpack_t2_ternary(npz_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct the T2TernaryAdapter residual from a serialized
    adapter.npz. Returns (q, scale) where `q` is the (out, in) signed
    ternary matrix (-scale, 0, +scale) and `scale` is the per-row
    scale (out, 1).

    Convention must match examples/af2_storage_tournament.py
    T2TernaryAdapter.serialize + T2TernaryAdapter.patch.
    """
    d = np.load(npz_path)
    packed = d["packed"]
    scale_np = d["scale"].astype(np.float32)
    shape = d["shape"]
    out_features, in_features = int(shape[0]), int(shape[1])

    # 2-bit unpacking: 4 ternary codes per byte
    f0 = packed & 0x3
    f1 = (packed >> 2) & 0x3
    f2 = (packed >> 4) & 0x3
    f3 = (packed >> 6) & 0x3
    flat = np.stack([f0, f1, f2, f3]).reshape(-1).astype(np.int8)
    expected = out_features * in_features
    if flat.size < expected:
        raise ValueError(f"packed data too small: {flat.size} < {expected}")
    coded = flat[:expected].reshape(out_features, in_features)
    # coded in {0,1,2} -> q = (coded - 1) * scale  (in {-scale, 0, +scale})
    q_np = (coded.astype(np.float32) - 1.0) * scale_np
    q = torch.from_numpy(q_np).to(torch.float32)
    scale = torch.from_numpy(scale_np).to(torch.float32)
    return q, scale


# ---------- patching ------------------------------------------------------


def navigate_to_module(model, target_module: str):
    parent = model
    for name in target_module.split("."):
        parent = getattr(parent, name)
    return parent

def patch_random_t2(model, target_module: str, npz_path: Path,
                     device: str = "cuda"):
    q, scale = unpack_t2_ternary(npz_path)
    # Cast to fp16 to match the model's dtype; x arrives as fp16.
    q = q.to(device=device, dtype=torch.float16)
    scale = scale.to(device=device, dtype=torch.float16)
    parent = navigate_to_module(model, target_module)

    def residual(x):
        # Match runtime: F.linear(x, q_ste) * scale.squeeze(1)
        y = F.linear(x, q)
        return y * scale.squeeze(1)

    _patch_module_forward(parent, residual)

def patch_random_lora(model, target_module: str, npz_path: Path,
                       device: str = "cuda"):
    d = np.load(npz_path)
    # Cast to fp16 to match the model's dtype; x arrives as fp16.
    W_down = torch.from_numpy(d["W_down"]).to(device=device,
                                                dtype=torch.float16)
    W_up = torch.from_numpy(d["W_up"]).to(device=device,
                                            dtype=torch.float16)
    parent = navigate_to_module(model, target_module)


    def residual(x):
        # y = W_up @ W_down @ x
        # x is (..., in_dim). LoRA: (out, rank) @ (rank, in) -> (out, in) @ x
        hidden = F.linear(x, W_down)  # (..., rank)
        return F.linear(hidden, W_up)  # (..., out_dim)

    _patch_module_forward(parent, residual)


def apply_damage_ptq(model, target_module: str, threshold: float | None):
    """Re-apply the --damage-ptq stage so the model's target_module
    is in the same damaged state as the corresponding Stage 1 run.

    Threshold=None reproduces the FP16 reference (no damage).
    """
    if threshold is None:
        return  # FP16 reference; nothing to do
    parent = navigate_to_module(model, target_module)
    # damage_target_module replaces parent.weight.data with the
    # ternary reconstruction (matches Stage 1 driver behavior).
    damage_target_module(parent, group_size=128, threshold=threshold)


# ---------- model + eval --------------------------------------------------


def load_model(model_name: str, device: str = "cuda",
                dtype=torch.float16):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[eval-untrained] loading {model_name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    return model, tokenizer

def run_lm_eval(model, tokenizer, tasks, batch_size: int,
                 limit=None) -> dict:
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    lm = HFLM(pretrained=model, tokenizer=tokenizer,
               batch_size=batch_size)
    results = simple_evaluate(model=lm, tasks=tasks,
                                batch_size=batch_size, limit=limit)
    # Match Stage 1 driver behavior: the driver picked the first metric
    # key containing "acc" or "word_perplexity" in iteration order, which
    # gave `acc_norm,none` for arc_easy on the trained arms. We replicate
    # this to keep the comparison apples-to-apples.
    out = {}
    full = {}
    # Match the Stage 1 trained-arm metric choice: arc_easy uses
    # `acc_norm,none` (length-normalized log-likelihood comparison).
    # lambada_openai uses `acc,none`. wikitext uses `word_perplexity,none`.
    # This explicit override matches what Stage 1 recorded for the
    # trained arms, so the comparison is apples-to-apples.
    preferred = {
        "arc_easy": "acc_norm,none",
        "lambada_openai": "acc,none",
        "wikitext": "word_perplexity,none",
    }
    for t_name, t_results in results["results"].items():
        # Save the entire per-task results dict for full auditability
        full[t_name] = {k: (float(v) if isinstance(v, (int, float))
                              else v) for k, v in t_results.items()}
        # Prefer the explicit match if present; else fall back to
        # first non-stderr key
        pref = preferred.get(t_name)
        picked = None
        if pref is not None and pref in t_results:
            picked = (pref, float(t_results[pref]))
        else:
            for k, v in t_results.items():
                if "_stderr" in k:
                    continue
                picked = (k, float(v))
                break
        out[t_name] = {"metric": picked[0], "value": picked[1]}
    return {"summary": out, "full": full}
    return None  # FP16 reference regime (D0)


def parse_damage_meta(run_path: Path) -> dict | None:
    """Read damage_meta from the first trained arm's eval.summary.json.
    Returns None if no arm has damage_meta (e.g. D0 = FP16 reference)."""
    for seed_dir in sorted(run_path.glob("seed-*")):
        for arm_dir in seed_dir.iterdir():
            if not arm_dir.is_dir():
                continue
            sp = arm_dir / "eval.summary.json"
            if sp.exists():
                es = json.loads(sp.read_text())
                meta = es.get("damage_meta")
                if meta is not None:
                    return meta
    return None


def find_run_paths(regime: str, base: Path) -> list[Path]:
    """Find the AF2-D run root(s) under runs/r/EXP-RPM-D{regime}/*/af2d.
    Returns the af2d/ root for each distinct run-timestamp directory.
    """
    pattern = f"EXP-RPM-D{regime}"
    runs_root = base / "runs" / "r" / pattern
    if not runs_root.exists():
        return []
    out = []
    for ts_dir in sorted(runs_root.iterdir()):
        af2d = ts_dir / "af2d"
        if af2d.exists():
            out.append(af2d)
    return out



def get_target_module(run_path: Path) -> str:
    for seed_dir in sorted(run_path.glob("seed-*")):
        for arm_dir in seed_dir.iterdir():
            sp = arm_dir / "eval.summary.json"
            if sp.exists():
                es = json.loads(sp.read_text())
                tm = es.get("target_module")
                if tm:
                    return tm
    raise RuntimeError(f"could not infer target_module for {run_path}")


# ---------- main ----------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes", default="0,1,2,3,4,5",
                     help="comma-separated regime digits 0..5 (D0..D5)")
    ap.add_argument("--arms", default="random_t2_ternary,random_lora")
    ap.add_argument("--tasks", default="wikitext,arc_easy,lambada_openai")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None,
                     help="limit per task (None = full)")
    ap.add_argument("--base", type=Path,
                     default=Path("/home/andrew-jochl/TORUS"))
    ap.add_argument("--model", default="allenai/OLMo-1B-0724-hf")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    regimes = args.regimes.split(",")
    arms = args.arms.split(",")
    tasks = args.tasks.split(",")

    print(f"[eval-untrained] regimes={regimes} arms={arms} tasks={tasks}",
          flush=True)

    total = len(regimes) * len(arms) * 3  # seeds 1..3
    done = 0
    t0 = time.time()

    for regime in regimes:
        runs = find_run_paths(regime, args.base)
        if not runs:
            print(f"[eval-untrained] regime D{regime}: no runs found",
                  flush=True)
            continue
        # Stage 1 had one ts dir per regime
        run_root = runs[0]
        damage_meta = parse_damage_meta(run_root)
        target_module = get_target_module(run_root)
        threshold = (damage_meta or {}).get("threshold")
        print(f"[eval-untrained] regime D{regime} damage={damage_meta} "
              f"target_module={target_module} run={run_root}",
              flush=True)

        for seed in (1, 2, 3):
            for arm in arms:
                arm_dir = run_root / f"seed-{seed:03d}" / arm
                npz = arm_dir / "adapter.npz"
                eval_summary_path = arm_dir / "eval.summary.json"
                eval_full_path = arm_dir / "eval.full.json"
                if not npz.exists():
                    print(f"[eval-untrained] missing {npz}", flush=True)
                    continue
                if eval_summary_path.exists():
                    es = json.loads(eval_summary_path.read_text())
                    if es.get("tasks"):
                        print(f"[eval-untrained] {arm} D{regime} s{seed}: "
                              f"already has tasks; skipping", flush=True)
                        done += 1
                        continue

                # Load model + damage + patch + eval
                print(f"[eval-untrained] {arm} D{regime} s{seed}: "
                      f"[{done+1}/{total}] loading model...", flush=True)
                model, tokenizer = load_model(
                    args.model, device=args.device)
                apply_damage_ptq(model, target_module, threshold)

                if arm == "random_t2_ternary":
                    patch_random_t2(model, target_module, npz,
                                     device=args.device)
                elif arm == "random_lora":
                    patch_random_lora(model, target_module, npz,
                                       args.device)
                else:
                    raise ValueError(f"unknown arm {arm}")

                # Run eval
                print(f"[eval-untrained] {arm} D{regime} s{seed}: "
                      f"running lm-eval...", flush=True)
                t_eval = time.time()
                eval_out = run_lm_eval(
                    model, tokenizer, tasks,
                    batch_size=args.batch_size, limit=args.limit)
                summary = eval_out["summary"]
                full = eval_out["full"]
                dt = time.time() - t_eval
                print(f"[eval-untrained] {arm} D{regime} s{seed}: "
                      f"eval done in {dt:.1f}s -> {summary}", flush=True)

                # Update eval.summary.json in place
                es = json.loads(eval_summary_path.read_text())
                es["tasks"] = summary
                eval_summary_path.write_text(
                    json.dumps(es, indent=2, default=str))
                eval_full_path.write_text(
                    json.dumps(full, indent=2, default=str))

                # Free model
                del model, tokenizer
                gc.collect()
                torch.cuda.empty_cache()

                done += 1
                print(f"[eval-untrained] progress: {done}/{total} "
                      f"elapsed={time.time()-t0:.1f}s", flush=True)

    print(f"[eval-untrained] DONE {done}/{total} cells "
          f"in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()