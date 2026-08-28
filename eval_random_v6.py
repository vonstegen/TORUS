"""Post-hoc eval of random arms for EXP-RPM-AF2D-SEVERITY.

Walks the per-threshold, per-seed layout produced by stage2-v6-launch.sh.
For each (threshold, seed, random_arm), reload the model + apply TWN damage +
patch the random arm + run lm-eval-harness.

Mirrors the Stage 2 v3 / Stage 2 v4 / Stage 2 v5 post-hoc eval pattern
but parameterized for the threshold-sweep directory structure.
"""
from __future__ import annotations
import argparse, gc, json, sys
from pathlib import Path

import numpy as np
import triton
import torch
from examples.af2_storage_tournament import (  # noqa: E402
    _patch_module_forward,
    damage_target_module_gaussian,  # not used here; we use TWN damage
)
from torus.train.ste import ternary_quantize_with_ste  # noqa: E402


def navigate_to_module(model, target_path: str):
    parts = target_path.split(".")
    node = model
    for p in parts[:-1]:
        node = getattr(node, p)
    leaf = getattr(node, parts[-1])
    return leaf


def load_model(model_name, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    return model, tokenizer


def apply_damage_twn(model, target_path, threshold, group_size=128):
    leaf = navigate_to_module(model, target_path)
    w_before = leaf.weight.detach().clone()
    w_np = w_before.cpu().numpy().astype(np.float32, copy=False)
    codes, scale, quantized = ternary_quantize_with_ste(
        w_np, group_size=group_size, threshold=threshold,
        calibrate_norm=False, ref_weight=w_np,
    )
    q_tensor = torch.from_numpy(quantized).to(
        leaf.weight.device, leaf.weight.dtype)
    leaf.weight.data.copy_(q_tensor)
    leaf.weight.requires_grad_(False)


def patch_random_t2(model, target_path, npz_path):
    from examples.af2_storage_tournament import T2TernaryAdapter
    leaf = navigate_to_module(model, target_path)
    npz = np.load(npz_path)
    # Reconstruct latent from packed codes + scale.
    packed = npz["packed_codes"]  # uint8
    scales = npz["scales"]  # fp16
    code_bits = 2
    n_values = packed.size * (8 // code_bits)
    n_rows = scales.shape[0]
    n_cols = n_values // n_rows
    codes_int = np.zeros(n_values, dtype=np.int8)
    mask = (1 << code_bits) - 1
    for i in range(8 // code_bits):
        shift = i * code_bits
        codes_int[i::8 // code_bits] = (packed.astype(np.int32) >> shift) & mask
    codes_int = codes_int - 1  # map {0,1,2} -> {-1,0,1}
    codes_int = codes_int.reshape(n_rows, n_cols)
    latent = torch.from_numpy(codes_int.astype(np.float32) * scales.reshape(-1, 1)).to(
        leaf.weight.device, leaf.weight.dtype
    )
    adapter = T2TernaryAdapter(target_module=leaf, n_planes=1, train=False)
    adapter.latent.data.copy_(latent)
    adapter.patch(leaf)


def patch_random_lora(model, target_path, npz_path):
    """Apply random LoRA via adapter."""
    from examples.af2_storage_tournament import LoRAAdapter
    leaf = navigate_to_module(model, target_path)
    npz = np.load(npz_path)
    # LoRA adapter: load stored weights via __setstate__ if available, or
    # fall back to constructing from npz dict.
    adapter = LoRAAdapter(target_module=leaf, n_planes=1, train=False)
    if hasattr(adapter, "__setstate__"):
        adapter.__setstate__({"A": npz["A"], "B": npz["B"], "r": npz.get("r", 8)})
    else:
        # Best-effort reconstruction: assume latent attribute holds both A and B
        if hasattr(adapter, "A"):
            adapter.A.data = torch.from_numpy(npz["A"]).to(leaf.weight.device)
        if hasattr(adapter, "B"):
            adapter.B.data = torch.from_numpy(npz["B"]).to(leaf.weight.device)
    adapter.patch(leaf)


def run_lm_eval(model, tokenizer, tasks, batch_size):
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    results = lm_eval.simple_evaluate(model=lm, tasks=tasks)
    summary = {}
    full = {}
    for task, res in results["results"].items():
        for metric, val in res.items():
            if isinstance(val, (int, float)):
                summary[f"{task}_{metric}"] = float(val)
    full = results
    return {"summary": summary, "full": full}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="/home/andrew-jochl/TORUS/runs/r")
    ap.add_argument("--exp-id", default="EXP-RPM-AF2D-SEVERITY")
    ap.add_argument("--arms", default="random_t2_ternary,random_lora")
    ap.add_argument("--tasks", default="wikitext,arc_easy,lambada_openai")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--target-module", default="model.layers.0.mlp.down_proj")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    exp_dir = runs_root / args.exp_id
    arms = args.arms.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    tasks = args.tasks.split(",")

    thresholds = sorted([float(p.name.split("-")[1]) for p in exp_dir.iterdir()
                          if p.name.startswith("threshold-")])

    # Read damage metadata from the first trained arm in the first threshold.
    sigma = None
    model_name = None
    for thr in thresholds:
        thr_dir = exp_dir / f"threshold-{thr}"
        ts_dirs = sorted([p for p in thr_dir.iterdir()
                          if p.is_dir() and p.name.startswith("2026")])
        if not ts_dirs:
            continue
        ts = ts_dirs[0]
        for seed in seeds:
            for arm_dir in (ts / f"seed-{seed:03d}").iterdir():
                if arm_dir.name in arms:
                    continue
                es_path = arm_dir / "eval.summary.json"
                if es_path.exists():
                    es = json.loads(es_path.read_text())
                    dm = es.get("damage_meta", {})
                    sigma = dm.get("threshold", thr)  # threshold IS the damage axis here
                    model_name = es.get("model")
                    break
            if model_name:
                break
        if model_name:
            break

    if not model_name:
        print(f"[eval-random-v6] no model found")
        return
    print(f"[eval-random-v6] model={model_name} target={args.target_module} thresholds={thresholds}")

    for thr in thresholds:
        thr_dir = exp_dir / f"threshold-{thr}"
        ts_dirs = sorted([p for p in thr_dir.iterdir()
                          if p.is_dir() and p.name.startswith("2026")])
        if not ts_dirs:
            continue
        ts = ts_dirs[0]

        for seed in seeds:
            for arm in arms:
                arm_dir = ts / f"seed-{seed:03d}" / arm
                npz_path = arm_dir / "adapter.npz"
                if not npz_path.exists():
                    print(f"[eval-random-v6] {arm_dir}: no adapter.npz")
                    continue
                es_path = arm_dir / "eval.summary.json"
                # Skip if already populated.
                if es_path.exists():
                    existing = json.loads(es_path.read_text())
                    if existing.get("tasks"):
                        continue

                print(f"[eval-random-v6] thr={thr} seed={seed} arm={arm}", flush=True)
                model, tokenizer = load_model(model_name, device=args.device)
                apply_damage_twn(model, args.target_module, threshold=thr)
                if arm == "random_t2_ternary":
                    patch_random_t2(model, args.target_module, npz_path)
                elif arm == "random_lora":
                    patch_random_lora(model, args.target_module, npz_path)
                result = run_lm_eval(model, tokenizer, tasks, args.batch_size)

                if es_path.exists():
                    existing = json.loads(es_path.read_text())
                else:
                    existing = {"arm": arm, "seed": seed, "model": model_name,
                                "target_module": args.target_module}
                existing["tasks"] = result["summary"]
                existing["damage_meta"] = {"threshold": thr}
                es_path.write_text(json.dumps(existing, indent=2))
                (arm_dir / "eval.full.json").write_text(
                    json.dumps(result["full"], indent=2, default=str))
                print(f"[eval-random-v6] wrote {es_path}", flush=True)
                del model, tokenizer
                gc.collect()
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
