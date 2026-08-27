"""Post-hoc eval of random arms for EXP-RPM-DAMAGE-TYPE-001.

Generic version: walks runs/r/EXP-RPM-DAMAGE-TYPE-001/<ts>/stage_b_tournament/
and applies random arms per cell.
"""
from __future__ import annotations
import argparse, gc, json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

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


def run_lm_eval(model, tokenizer, tasks, batch_size):
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    results = simple_evaluate(model=lm, tasks=tasks)
    summary = {}
    for task, res in results["results"].items():
        for metric, val in res.items():
            if isinstance(val, (int, float)):
                summary[f"{task}_{metric}"] = float(val)
    return {"summary": summary, "full": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="/home/andrew-jochl/TORUS/runs/r")
    ap.add_argument("--exp-id", default="EXP-RPM-DAMAGE-TYPE-001")
    ap.add_argument("--ts", required=True, help="Stage B timestamp dir")
    ap.add_argument("--arms", default="random_t2_ternary,random_lora")
    ap.add_argument("--tasks", default="wikitext,arc_easy,lambada_openai")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--target-module", default="model.layers.0.mlp.down_proj")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    cell_root = runs_root / args.exp_id / args.ts / "stage_b_tournament"
    arms = args.arms.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    tasks = args.tasks.split(",")

    cell_ids = sorted([p.name for p in cell_root.iterdir()
                        if p.is_dir() and not p.name.endswith("-base")])
    print(f"[eval-random-s3] cell_ids={cell_ids} arms={arms} seeds={seeds}",
          flush=True)

    for cell_id in cell_ids:
        # Determine mechanism + parameter from cell_id
        if "TWN" in cell_id:
            thr = float(cell_id.split("thr-")[1]) if "thr-" in cell_id else 0.7
            apply_damage = lambda m, p, _t=thr: apply_damage_twn(m, p, _t)
        elif "Gaussian" in cell_id:
            sigma = float(cell_id.split("sigma-")[1]) if "sigma-" in cell_id else 3.0
            apply_damage = lambda m, p, _s=sigma: apply_damage_gaussian(m, p, _s)
        else:
            print(f"[eval-random-s3] unknown mechanism: {cell_id}", flush=True)
            continue

        for seed in seeds:
            for arm in arms:
                arm_dir = cell_root / cell_id / f"seed-{seed:03d}" / arm
                npz_path = arm_dir / "adapter.npz"
                if not npz_path.exists():
                    print(f"[eval-random-s3] {arm_dir}: no adapter.npz",
                          flush=True)
                    continue
                es_path = arm_dir / "eval.summary.json"
                if es_path.exists():
                    existing = json.loads(es_path.read_text())
                    if existing.get("tasks"):
                        continue
                print(f"[eval-random-s3] {cell_id} seed={seed} arm={arm}",
                      flush=True)
                # Determine model_name from trained arm eval.summary.json
                model_name = None
                for trained_arm in ["t2_ternary", "lora"]:
                    tf = cell_root / cell_id / f"seed-{seed:03d}" / trained_arm / "eval.summary.json"
                    if tf.exists():
                        model_name = json.loads(tf.read_text()).get("model")
                        break
                if not model_name:
                    print(f"[eval-random-s3] no model_name found", flush=True)
                    continue
                model, tokenizer = load_model(model_name, device=args.device)
                apply_damage(model, args.target_module)
                if arm == "random_t2_ternary":
                    patch_random_t2(model, args.target_module, npz_path)
                elif arm == "random_lora":
                    patch_random_lora(model, args.target_module, npz_path)
                result = run_lm_eval(model, tokenizer, tasks, args.batch_size)
                if es_path.exists():
                    existing = json.loads(es_path.read_text())
                else:
                    existing = {"arm": arm, "seed": seed, "model": model_name,
                                "target_module": args.target_module,
                                "cell_id": cell_id}
                existing["tasks"] = result["summary"]
                existing["damage_meta"] = {"cell_id": cell_id}
                es_path.write_text(json.dumps(existing, indent=2))
                (arm_dir / "eval.full.json").write_text(
                    json.dumps(result["full"], indent=2, default=str))
                print(f"[eval-random-s3] wrote {es_path}", flush=True)
                del model, tokenizer
                gc.collect()
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()