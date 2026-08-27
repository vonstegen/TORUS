"""Faster post-hoc eval: wikitext only (the most informative single task)
for Stage 3 v1 random arms.

The original eval_random_stage3.py was too slow (~30 min per cell due to
loading a fresh model each time). This version:
  1. Only evaluates wikitext word_perplexity (one task, the most informative)
  2. Loads the model ONCE per cell and reuses it across both random arms
  3. Skips cells that already have data
"""
from __future__ import annotations
import argparse, gc, json
from pathlib import Path

import numpy as np
import torch

import examples.af2_storage_tournament as af2  # noqa: E402
from torus.train.ste import ternary_quantize_with_ste


def navigate_to_module(model, target_path):
    parts = target_path.split(".")
    node = model
    for p in parts[:-1]:
        node = getattr(node, p)
    leaf = getattr(node, parts[-1])
    if not hasattr(leaf, "weight") and hasattr(leaf, "down_proj"):
        leaf = leaf.down_proj
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


def apply_damage_gaussian(model, target_path, sigma):
    leaf = navigate_to_module(model, target_path)
    w = leaf.weight.detach().clone().cpu().numpy()
    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, sigma * w.std(), size=w.shape).astype(w.dtype)
    damaged = (w + noise).astype(np.float32)
    q_tensor = torch.from_numpy(damaged).to(
        leaf.weight.device, leaf.weight.dtype)
    leaf.weight.data.copy_(q_tensor)
    leaf.weight.requires_grad_(False)


def patch_random_t2(model, target_module, npz_path):
    import torch.nn.functional as F
    d = np.load(npz_path)
    shape = tuple(int(x) for x in d["shape"])
    out_features, in_features = shape
    scale = torch.from_numpy(d["scale"]).to(
        device=next(model.parameters()).device, dtype=torch.float16)
    packed = d["packed"]
    n_total = out_features * in_features
    flat = np.zeros(n_total, dtype=np.int8)
    for i in range(4):
        flat[i::4] = (packed >> (2 * i)) & 0x3
    flat = flat[:n_total]
    coded = (flat.astype(np.int8) - 1).astype(np.float32)
    q_ste = coded.reshape(shape)
    q_ste = torch.from_numpy(q_ste).to(
        device=next(model.parameters()).device, dtype=torch.float16)
    scale_b = scale.expand(out_features, in_features)
    parent = navigate_to_module(model, target_module)
    leaf = parent.down_proj if hasattr(parent, "down_proj") else parent
    def residual(x):
        out = F.linear(x, q_ste * scale_b)
        return out * scale.squeeze(1)
    af2._patch_module_forward(leaf, residual)


def patch_random_lora(model, target_module, npz_path):
    import torch.nn.functional as F
    d = np.load(npz_path)
    device = next(model.parameters()).device
    W_up = torch.from_numpy(d["W_up"]).to(device=device, dtype=torch.float16)
    W_down = torch.from_numpy(d["W_down"]).to(device=device, dtype=torch.float16)
    parent = navigate_to_module(model, target_module)
    leaf = parent.down_proj if hasattr(parent, "down_proj") else parent
    def residual(x):
        hidden = F.linear(x, W_down)
        return F.linear(hidden, W_up)
    af2._patch_module_forward(leaf, residual)


def run_wikitext_ppl(model, tokenizer):
    """Single-task wikitext word_perplexity, fast."""
    from datasets import load_dataset
    from torch.nn import CrossEntropyLoss
    text = load_dataset("wikitext", "wikitext-103-raw-v1", split="test",
                         trust_remote_code=False)["text"]
    text = "\n\n".join([t for t in text if len(t) > 0])[:1_000_000]
    enc = tokenizer(text, return_tensors="pt").to(next(model.parameters()).device)
    n_tokens = enc.input_ids.shape[1]
    nlls = []
    loss_fn = CrossEntropyLoss()
    for i in range(0, n_tokens - 1024, 1024):
        chunk = enc.input_ids[:, i:i+1024]
        with torch.no_grad():
            out = model(chunk, labels=chunk)
        nlls.append(out.loss.item() * 1024)
    return float(torch.tensor(nlls).sum().exp())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="/home/andrew-jochl/TORUS/runs/r")
    ap.add_argument("--exp-id", default="EXP-RPM-DAMAGE-TYPE-001")
    ap.add_argument("--ts", required=True)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--target-module", default="model.layers.0.mlp.down_proj")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    cell_root = runs_root / args.exp_id / args.ts / "stage_b_tournament"
    seeds = [int(s) for s in args.seeds.split(",")]

    cell_ids = sorted([p.name for p in cell_root.iterdir()
                        if p.is_dir() and not p.name.endswith("-base")])
    print(f"[eval-s3-fast] cells={cell_ids} seeds={seeds}", flush=True)

    for cell_id in cell_ids:
        if "TWN" in cell_id:
            thr = float(cell_id.split("thr-")[1])
            apply_damage = lambda m, p: apply_damage_twn(m, p, thr)
        elif "Gaussian" in cell_id:
            sigma = float(cell_id.split("sigma-")[1])
            apply_damage = lambda m, p: apply_damage_gaussian(m, p, sigma)
        else:
            continue

        for seed in seeds:
            for arm, patch_fn in [("random_t2_ternary", patch_random_t2),
                                   ("random_lora", patch_random_lora)]:
                arm_dir = cell_root / cell_id / f"seed-{seed:03d}" / arm
                npz_path = arm_dir / "adapter.npz"
                if not npz_path.exists():
                    continue
                es_path = arm_dir / "eval.summary.json"
                existing = json.loads(es_path.read_text()) if es_path.exists() else {}
                # Skip if already has wikitext data
                tasks = existing.get("tasks", {})
                if (isinstance(tasks, dict) and "wikitext" in tasks
                        and isinstance(tasks["wikitext"], dict)
                        and "value" in tasks["wikitext"]):
                    continue
                # Find model_name from trained arm
                model_name = None
                for ta in ["t2_ternary", "lora"]:
                    tf = cell_root / cell_id / f"seed-{seed:03d}" / ta / "eval.summary.json"
                    if tf.exists():
                        model_name = json.loads(tf.read_text()).get("model")
                        break
                if not model_name:
                    continue
                print(f"[eval-s3-fast] {cell_id} seed={seed} arm={arm}", flush=True)
                model, tokenizer = load_model(model_name, device=args.device)
                apply_damage(model, args.target_module)
                patch_fn(model, args.target_module, npz_path)
                ppl = run_wikitext_ppl(model, tokenizer)
                print(f"[eval-s3-fast]   ppl={ppl:.2f}", flush=True)
                # Write back
                existing["arm"] = arm
                existing["seed"] = seed
                existing["model"] = model_name
                existing["target_module"] = args.target_module
                existing["cell_id"] = cell_id
                existing["tasks"] = {
                    "wikitext": {"metric": "word_perplexity,none", "value": ppl}
                }
                es_path.write_text(json.dumps(existing, indent=2))
                del model, tokenizer
                gc.collect()
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()