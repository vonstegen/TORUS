"""Post-hoc eval of Stage 2 v2 untrained-control arms (random_t2_ternary,
random_lora) at the preregistered sigma values.

Stage 2 v2 tournaments skip lm-eval-harness on `is_untrained` arms
(same as Stage 1), so the random arms have `tasks: {}` in their
eval.summary.json. This script re-loads each random adapter, re-applies
the matching Gaussian damage on the base model, and runs lm-eval-harness
to fill in the missing fields.

Driver SHA: `75f7930` (current main). Uses the same packing/unpacking
convention as the Stage 2 v2 driver; verified against the tournament
output structure.

Usage
-----
  PYTHONPATH=. .venv/bin/python examples/eval_untrained_arms_v2.py \
    --regimes l0-v-gauss,l15-gauss \
    --arms random_t2_ternary,random_lora \
    --tasks wikitext,arc_easy,lambada_openai \
    --batch_size 16
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path



import numpy as np
try:
    import triton  # noqa: F401
except ModuleNotFoundError:
    import sys as _sys

    _sys.modules["triton"] = None

import torch

from examples.af2_storage_tournament import (  # noqa: E402
    _patch_module_forward,
    damage_target_module_gaussian,
)


def navigate_to_module(model, target_path: str):
    """Navigate a dotted path on the model (matches the Stage 2 v2
    driver's resolve_target_module behavior)."""
    parts = target_path.split(".")
    node = model
    for p in parts[:-1]:
        node = getattr(node, p)
    leaf = getattr(node, parts[-1])
    if hasattr(leaf, "weight") and hasattr(leaf.weight, "device"):
        return leaf
    for sub in leaf.named_children():
        if hasattr(sub[1], "weight") and hasattr(sub[1].weight, "device"):
            return sub[1]
    raise ValueError(f"could not resolve {target_path}")


def patch_random_t2(model, target_module: str, npz_path: Path):
    """Re-apply a serialized random T2 ternary adapter.

    Serialization format (matches T2TernaryAdapter.serialize):
      packed:  uint8, length = ceil(out*in / 4) with 2-bit codes 0/1/2
              (0=-scale, 1=0, 2=+scale)
      scale:   fp16, shape (out, 1)
      shape:   int64, [out, in]
    """
    import torch.nn.functional as F
    d = np.load(npz_path)
    shape = tuple(int(x) for x in d["shape"])
    out_features, in_features = shape
    scale = torch.from_numpy(d["scale"]).to(
        device=next(model.parameters()).device, dtype=torch.float16)
    # Unpack the 2-bit codes: 0 -> -scale, 1 -> 0, 2 -> +scale.
    packed = d["packed"]
    n_total = out_features * in_features
    flat = np.zeros(n_total, dtype=np.int8)
    for i in range(4):
        flat[i::4] = (packed >> (2 * i)) & 0x3
    flat = flat[:n_total]
    coded = (flat.astype(np.int8) - 1).astype(np.float32)  # 0,1,2 -> -1,0,1
    q_ste = coded.reshape(shape)
    q_ste = torch.from_numpy(q_ste).to(
        device=next(model.parameters()).device, dtype=torch.float16)
    scale_b = scale.expand(out_features, in_features)

    parent = navigate_to_module(model, target_module)

    def residual(x):
        # q_ste * scale = ternary values {-scale, 0, +scale}
        out = F.linear(x, q_ste * scale_b)
        return out * scale.squeeze(1)
    _patch_module_forward(parent, residual)


def patch_random_lora(model, target_module: str, npz_path: Path):
    """Re-apply a serialized random LoRA adapter (W_down, W_up)."""
    d = np.load(npz_path)
    import torch.nn.functional as F
    device = next(model.parameters()).device
    W_up = torch.from_numpy(d["W_up"]).to(device=device, dtype=torch.float16)
    W_down = torch.from_numpy(d["W_down"]).to(device=device, dtype=torch.float16)
    parent = navigate_to_module(model, target_module)

    def residual(x):
        hidden = F.linear(x, W_down)
        return F.linear(hidden, W_up)
    _patch_module_forward(parent, residual)


def apply_damage_gaussian(model, target_module: str, sigma: float,
                           seed: int = 0):
    """Re-apply Gaussian weight noise so the base is in the same state
    as the corresponding Stage 2 v2 tournament."""
    parent = navigate_to_module(model, target_module)
    damage_target_module_gaussian(parent, sigma=sigma, seed=seed)


def load_model(model_name: str, device: str = "cuda",
                dtype=torch.float16):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[eval-untrained-v2] loading {model_name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    return model, tokenizer


def run_lm_eval(model, tokenizer, tasks, batch_size: int) -> dict:
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    lm = HFLM(pretrained=model, tokenizer=tokenizer,
               batch_size=batch_size)
    results = simple_evaluate(model=lm, tasks=tasks,
                                batch_size=batch_size)
    preferred = {
        "arc_easy": "acc_norm,none",
        "lambada_openai": "acc,none",
        "wikitext": "word_perplexity,none",
    }
    out = {}
    full = {}
    for t_name, t_results in results["results"].items():
        full[t_name] = {k: (float(v) if isinstance(v, (int, float))
                              else v) for k, v in t_results.items()}
        pref = preferred.get(t_name)
        if pref and pref in t_results:
            out[t_name] = {"metric": pref, "value": float(t_results[pref])}
        else:
            for k, v in t_results.items():
                if "_stderr" in k:
                    continue
                out[t_name] = {"metric": k, "value": float(v)}
                break
    return {"summary": out, "full": full}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root",
                    default="/home/andrew-jochl/TORUS/runs/r")
    ap.add_argument("--regimes", default="l0-v-gauss,l15-gauss",
                    help="comma-separated tournament IDs")
    ap.add_argument("--arms", default="random_t2_ternary,random_lora")
    ap.add_argument("--tasks", default="wikitext,arc_easy,lambada_openai")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    regimes = args.regimes.split(",")
    arms = args.arms.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    tasks = args.tasks.split(",")

    for regime in regimes:
        exp_id = f"EXP-RPM-{regime.upper().replace('-GAUSS', '-GAUSS')}"
        run_root = runs_root / exp_id
        if not run_root.exists():
            print(f"[eval-untrained-v2] missing {run_root}; skipping")
            continue
        # Pick the most-recent timestamp dir.
        ts_dirs = sorted([p for p in run_root.iterdir() if p.is_dir()],
                         reverse=True)
        if not ts_dirs:
            print(f"[eval-untrained-v2] no timestamp dir for {exp_id}")
            continue
        ts = ts_dirs[0]

        # Read damage metadata from the first trained arm.
        first_trained_arm = None
        for seed in seeds:
            for arm_dir in (ts / f"seed-{seed:03d}").iterdir():
                if arm_dir.name in arms:
                    continue
                es_path = arm_dir / "eval.summary.json"
                if es_path.exists():
                    first_trained_arm = es_path
                    break
            if first_trained_arm:
                break
        if first_trained_arm is None:
            print(f"[eval-untrained-v2] no trained arm found for {exp_id}")
            continue
        es = json.loads(first_trained_arm.read_text())
        sigma = (es.get("damage_meta") or {}).get("sigma")
        if sigma is None:
            print(f"[eval-untrained-v2] no sigma in {first_trained_arm}")
            continue
        target_module = es["target_module"]
        model_name = es["model"]
        print(f"[eval-untrained-v2] regime {exp_id} σ={sigma} target={target_module}",
              flush=True)

        for seed in seeds:
            for arm in arms:
                arm_dir = ts / f"seed-{seed:03d}" / arm
                es_path = arm_dir / "eval.summary.json"
                npz_path = arm_dir / "adapter.npz"
                if not npz_path.exists():
                    print(f"[eval-untrained-v2] {arm_dir}: no adapter.npz")
                    continue
                if es_path.exists():
                    existing = json.loads(es_path.read_text())
                    if existing.get("tasks"):
                        continue  # already populated
                print(f"[eval-untrained-v2] {exp_id} seed={seed} arm={arm}",
                      flush=True)
                # Load model + damage + patch + eval
                model, tokenizer = load_model(model_name, device=args.device)
                apply_damage_gaussian(model, target_module, sigma)
                if arm == "random_t2_ternary":
                    patch_random_t2(model, target_module, npz_path)
                elif arm == "random_lora":
                    patch_random_lora(model, target_module, npz_path)
                result = run_lm_eval(model, tokenizer, tasks,
                                     args.batch_size)
                # Merge into existing summary (preserve metadata).
                if es_path.exists():
                    existing = json.loads(es_path.read_text())
                else:
                    existing = {"arm": arm, "seed": seed,
                                  "model": model_name,
                                  "target_module": target_module}
                existing["tasks"] = result["summary"]
                existing["damage_meta"] = {"sigma": sigma}
                es_path.write_text(json.dumps(existing, indent=2))
                (arm_dir / "eval.full.json").write_text(
                    json.dumps(result["full"], indent=2))
                print(f"[eval-untrained-v2] wrote {es_path}", flush=True)
                del model, tokenizer
                gc.collect()
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()