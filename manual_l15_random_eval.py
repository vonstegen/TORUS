"""Manually run post-hoc eval on L15 seed-002 and seed-003 random arms.

Imports triton first AND sets sys.modules['triton'] explicitly to defeat
importlib's cached "None" import.
"""

import sys
import importlib

# Hard reset: if triton is cached as None, force a fresh import.
if 'triton' in sys.modules and sys.modules['triton'] is None:
    del sys.modules['triton']
# Force all triton.* submodules to be re-resolved.
for k in list(sys.modules.keys()):
    if k.startswith('triton') and sys.modules.get(k) is None:
        del sys.modules[k]

import triton  # ensure triton is fully loaded
sys.modules['triton'] = triton
for k in list(sys.modules.keys()):
    if k.startswith('triton'):
        sys.modules[k] = importlib.import_module(k)

import json
import os
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/andrew-jochl/TORUS")
from examples.eval_untrained_arms_v2 import (
    load_model, run_lm_eval, patch_random_t2, patch_random_lora,
    apply_damage_gaussian,
)


BASE = Path("/home/andrew-jochl/TORUS")
EXP_ID = "EXP-RPM-L15-GAUSS"
TS = BASE / "runs" / "r" / EXP_ID / "20260824T212232Z"
SEEDS = [2, 3]
ARMS = ["random_t2_ternary", "random_lora"]
TASKS = ["wikitext", "arc_easy", "lambada_openai"]

first_es_path = TS / "seed-001" / "t2_ternary" / "eval.summary.json"
es = json.loads(first_es_path.read_text())
sigma = es["damage_meta"]["sigma"]
model_name = es["model"]
target_module = es["target_module"]

print(f"regime={EXP_ID} sigma={sigma} target={target_module}")

for seed in SEEDS:
    for arm in ARMS:
        arm_dir = TS / f"seed-{seed:03d}" / arm
        es_path = arm_dir / "eval.summary.json"
        npz_path = arm_dir / "adapter.npz"
        if not npz_path.exists():
            print(f"SKIP: {arm_dir}: no adapter.npz")
            continue
        if es_path.exists():
            existing = json.loads(es_path.read_text())
            if existing.get("tasks"):
                print(f"SKIP: {es_path} already has tasks")
                continue
        print(f"=== {EXP_ID} seed={seed} arm={arm} ===", flush=True)
        model, tokenizer = load_model(model_name, device="cuda")
        apply_damage_gaussian(model, target_module, sigma)
        if arm == "random_t2_ternary":
            patch_random_t2(model, target_module, npz_path)
        elif arm == "random_lora":
            patch_random_lora(model, target_module, npz_path)
        result = run_lm_eval(model, tokenizer, TASKS, 16)
        if es_path.exists():
            existing = json.loads(es_path.read_text())
        else:
            existing = {"arm": arm, "seed": seed, "model": model_name,
                        "target_module": target_module}
        existing["tasks"] = result["summary"]
        existing["damage_meta"] = {"sigma": sigma}
        es_path.write_text(json.dumps(existing, indent=2))
        (arm_dir / "eval.full.json").write_text(
            json.dumps(result["full"], indent=2))
        print(f"wrote {es_path}", flush=True)
        del model, tokenizer
        import gc
        gc.collect()
        torch.cuda.empty_cache()