"""EXP-A4-001 — H2 discovery: heterogeneous precision vs uniform low-bit.

Arms (per seed, matched budget, joint training):
  uniform_ternary : all nn.Linear -> ternary (per-group absmean,
                    thr 0.0, group 128, STE), everything else FP16.
  hetero_ternary  : identical EXCEPT model.layers.0.mlp.down_proj
                    unwrapped (FP16, trainable).
  int8_uniform    : all linears per-row int8 (fp16 row scales, STE).
  int4_uniform    : all linears per-row int4 (descriptive).
  fp16_continue   : pristine FP16, all weights trainable (C2).
  fp16_reference  : eval-only measurement reference.

Frozen bars (manifest kill_criteria):
  Gate1: fp16_reference wikitext within |ppl - 13.0932| <= 0.2.
  Gate2: uniform_ternary step-0 ppl >= 100.
  H-bar (KILL): z(hetero better than uniform) > 2 on >= 2/3, no z < -2.
  C1-bar (KILL): z(hetero vs int8) >= -2 on ALL 3 AND
                 bytes(hetero) <= bytes(int8).
  C2-check: z(hetero vs fp16_continue) >= z(int8 vs fp16_continue) - 2
            on all 3.
  PASS iff all three; else FAIL.

Footprint measured from serialized structures: packed codes +
scales + metadata + protected FP16 weights; shared fp16
(embeddings, head, biases, norms) identical across arms.

Pure rule functions are torch-free; the CLI needs Legion.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

FP16_BASELINE_PPL = 13.0932
FP16_GATE_TOL = 0.2
UNIFORM_TERNARY_STEP0_MIN_PPL = 100.0
BAR_SIGMA = 2.0
METRICS = ("wikitext", "arc_easy", "lambada_openai")
METRIC_DIRECTION = {"wikitext": -1.0, "arc_easy": 1.0,
                    "lambada_openai": 1.0}


# ---- frozen rule functions (pure) -------------------------------------

def sd_of_difference(se1: float | None, se2: float | None) -> float:
    return math.sqrt((se1 or 0.0) ** 2 + (se2 or 0.0) ** 2)


def z_better(a: tuple[float, float], b: tuple[float, float],
             metric: str) -> float:
    """z > 0 means a better than b (per-metric direction)."""
    sd = sd_of_difference(a[1], b[1])
    if sd <= 0:
        return 0.0
    return METRIC_DIRECTION[metric] * (a[0] - b[0]) / sd


def apply_bars(means: dict[str, dict[str, tuple[float, float]]],
               footprints: dict[str, int]) -> dict:
    """Frozen H2 discovery bars. PASS iff H-bar AND C1-bar AND C2-check."""
    hetero = means["hetero_ternary"]
    uniform = means["uniform_ternary"]
    int8 = means["int8_uniform"]
    cont = means["fp16_continue"]

    z_h = {m: z_better(hetero[m], uniform[m], m) for m in METRICS}
    h_pass = (sum(z > BAR_SIGMA for z in z_h.values()) >= 2
              and all(z >= -BAR_SIGMA for z in z_h.values()))

    z_c1 = {m: z_better(hetero[m], int8[m], m) for m in METRICS}
    c1_cap = all(z >= -BAR_SIGMA for z in z_c1.values())
    c1_bytes = footprints["hetero_ternary"] <= footprints["int8_uniform"]
    c1_pass = c1_cap and c1_bytes

    z_het_cont = {m: z_better(hetero[m], cont[m], m) for m in METRICS}
    z_i8_cont = {m: z_better(int8[m], cont[m], m) for m in METRICS}
    c2_ok = all(z_het_cont[m] >= z_i8_cont[m] - BAR_SIGMA
                for m in METRICS)

    return {
        "z_hetero_vs_uniform": z_h,
        "z_hetero_vs_int8": z_c1,
        "z_hetero_vs_fp16_continue": z_het_cont,
        "z_int8_vs_fp16_continue": z_i8_cont,
        "footprints": footprints,
        "h_bar": h_pass,
        "c1_bar": c1_pass,
        "c1_capability_ok": c1_cap,
        "c1_bytes_ok": c1_bytes,
        "c2_check": c2_ok,
        "decision": "PASS" if (h_pass and c1_pass and c2_ok) else "FAIL",
    }


# ---- CLI --------------------------------------------------------------

def _extract_eval(pre: dict, task: str) -> float:
    for k, v in pre["results"][task].items():
        if "_stderr" in k:
            continue
        if "acc" in k or "word_perplexity" in k:
            return float(v)
    raise KeyError(f"no metric for {task}: {pre['results'][task]}")


def wrap_ternary(module, group_size: int = 128):
    import torch
    import torch.nn.functional as F

    from torus.train.ste import ternary_quantize_ste_torch

    w0 = module.weight.detach()
    module.weight.requires_grad_(False)
    latent = torch.nn.Parameter(w0.clone())
    bias = module.bias

    def fwd(x):
        q = ternary_quantize_ste_torch(
            latent, group_size=group_size, threshold=0.0)
        return F.linear(x, q, bias)

    module.forward = fwd
    return latent


def wrap_intN(module, levels: int):
    import torch
    import torch.nn.functional as F

    w0 = module.weight.detach()
    module.weight.requires_grad_(False)
    latent = torch.nn.Parameter(w0.clone())
    bias = module.bias

    def fwd(x):
        s = latent.detach().abs().amax(dim=1, keepdim=True) \
            .clamp(min=1e-6) / levels
        q = torch.round(latent / s).clamp(-levels, levels)
        w = latent + (q * s - latent).detach()
        return F.linear(x, w, bias)

    module.forward = fwd
    return latent


def arm_footprint(arm: str, wrapped, protected_bytes: int,
                  shared_fp16_bytes: int) -> dict:
    """Nominal bpw + serialized bytes from the wrapped structures."""
    total_codes_bits = 0
    total_codes = 0
    scale_bytes = 0
    meta_bytes = 0
    for kind, latent, n_groups, _group_size in wrapped:
        n = latent.numel()
        total_codes += n
        if kind == "ternary":
            total_codes_bits += n * 2
            scale_bytes += n_groups * 2  # fp16 per group
        elif kind == "int8":
            total_codes_bits += n * 8
            scale_bytes += n_groups * 2
        elif kind == "int4":
            total_codes_bits += n * 4
            scale_bytes += n_groups * 2
        meta_bytes += 64  # per-layer format metadata (bounded)
    codes_bytes = (total_codes_bits + 7) // 8
    quantized_bytes = codes_bytes + scale_bytes + meta_bytes \
        + protected_bytes
    total = quantized_bytes + shared_fp16_bytes
    nominal_bpw = (total_codes_bits / total_codes) if total_codes else 0.0
    return {
        "nominal_bpw": round(nominal_bpw, 3),
        "codes_bytes": codes_bytes,
        "scale_bytes": scale_bytes,
        "metadata_bytes": meta_bytes,
        "protected_fp16_bytes": protected_bytes,
        "shared_fp16_bytes": shared_fp16_bytes,
        "serialized_bytes": total,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n-steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seeds", type=str, default="1,2,3")
    p.add_argument("--ids-cache", default="/tmp/wikitext103_train_ids.npy")
    args = p.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    import numpy as np
    import torch
    from transformers import AutoTokenizer

    from examples.af1_budget_control import (
        make_window_sampler,
        train_arm,
    )
    from examples.af2_storage_tournament import (
        build_base,
        eval_arm,
    )

    all_ids = np.load(args.ids_cache)
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B-0724-hf")
    pad_id = tokenizer.pad_token_id or 0
    TASKS = ["wikitext", "arc_easy", "lambada_openai"]
    PROTECTED = "model.layers.0.mlp.down_proj"

    def build_arm(arm: str):
        model = build_base("allenai/OLMo-1B-0724-hf",
                           dtype="float16", device=args.device)
        wrapped = []
        protected_bytes = 0
        for name, module in list(model.named_modules()):
            if not isinstance(module, torch.nn.Linear):
                continue
            if arm == "hetero_ternary" and name == PROTECTED:
                module.weight.requires_grad_(True)
                protected_bytes = module.weight.numel() * 2
                continue
            if arm in ("uniform_ternary", "hetero_ternary"):
                latent = wrap_ternary(module)
                n_groups = module.weight.shape[0] * (
                    module.weight.shape[1] // 128)
                wrapped.append(("ternary", latent, n_groups, 128))
            elif arm == "int8_uniform":
                latent = wrap_intN(module, 127)
                wrapped.append(("int8", latent,
                                module.weight.shape[0], 0))
            elif arm == "int4_uniform":
                latent = wrap_intN(module, 7)
                wrapped.append(("int4", latent,
                                module.weight.shape[0], 0))
        latents = [w[1] for w in wrapped]
        params = latents + [
            prm for prm in model.parameters() if prm.requires_grad]
        return model, wrapped, params, protected_bytes

    def eval_all(model):
        pre = eval_arm(model, tokenizer, tasks=TASKS, batch_size=4)
        return {t: _extract_eval(pre, t) for t in TASKS}

    # Shared fp16 (embeddings, head, biases, norms): identical across
    # arms; measured once from the fp16_reference model.
    ref_model = build_base("allenai/OLMo-1B-0724-hf",
                           dtype="float16", device=args.device)
    shared_fp16_bytes = sum(
        p.numel() * 2 for n, p in ref_model.named_parameters()
        if "embed" in n or "head" in n or "norm" in n or "bias" in n)
    ref_eval = eval_all(ref_model)
    gate1 = abs(ref_eval["wikitext"] - FP16_BASELINE_PPL) <= FP16_GATE_TOL
    print(f"[a4] fp16_reference {ref_eval} gate1={gate1}", flush=True)
    (args.out_dir / "fp16_reference.json").write_text(
        json.dumps({"eval": ref_eval, "gate1_ok": gate1,
                    "shared_fp16_bytes": shared_fp16_bytes}, indent=2))
    del ref_model
    torch.cuda.empty_cache()

    per_seed: dict[str, dict[str, dict]] = {}
    footprints: dict[str, dict] = {}
    for seed_s in args.seeds.split(","):
        seed = int(seed_s)
        seed_dir = args.out_dir / f"seed-{seed:03d}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        per_seed[seed_s] = {"fp16_reference": ref_eval}
        for arm in ("uniform_ternary", "hetero_ternary", "int8_uniform",
                    "int4_uniform", "fp16_continue"):
            torch.manual_seed(seed)
            np.random.seed(seed)
            model, wrapped, params, protected_bytes = build_arm(arm)

            def fwd(ids, _m=model):
                return _m(input_ids=ids).logits

            step0 = eval_all(model)
            if arm == "uniform_ternary":
                gate2 = step0["wikitext"] >= UNIFORM_TERNARY_STEP0_MIN_PPL
                print(f"[a4] seed {seed} uniform_ternary step0 "
                      f"{step0} gate2={gate2}", flush=True)
            data = make_window_sampler(all_ids, 4, 128, seed=seed,
                                       device=args.device)
            hist = train_arm(
                fwd, params, data,
                n_steps=args.n_steps, lr=args.lr,
                momentum=args.momentum, grad_clip=args.grad_clip,
                log_every=100, pad_id=pad_id)
            final = eval_all(model)
            fp = arm_footprint(arm, wrapped, protected_bytes,
                               shared_fp16_bytes)
            footprints.setdefault(arm, fp)
            arm_dir = seed_dir / arm
            arm_dir.mkdir(exist_ok=True)
            (arm_dir / "history.jsonl").write_text(
                "\n".join(json.dumps(h) for h in hist))
            (arm_dir / "eval.summary.json").write_text(
                json.dumps(final, indent=2))
            (arm_dir / "footprint.json").write_text(
                json.dumps(fp, indent=2))
            per_seed[seed_s][arm] = final
            print(f"[a4] seed {seed} {arm}: {final} "
                  f"bytes={fp['serialized_bytes']}", flush=True)
            del model
            torch.cuda.empty_cache()

    import statistics

    means: dict[str, dict[str, tuple[float, float]]] = {}
    for arm in ("uniform_ternary", "hetero_ternary", "int8_uniform",
                "int4_uniform", "fp16_continue", "fp16_reference"):
        out = {}
        for metric in METRICS:
            vals = [per_seed[s][arm][metric] for s in per_seed]
            out[metric] = (statistics.fmean(vals),
                           (statistics.stdev(vals) / len(vals) ** 0.5
                            if len(vals) > 1 else 0.0))
        means[arm] = out
    footprint_bytes = {arm: footprints[arm]["serialized_bytes"]
                       for arm in footprints}
    bars = apply_bars(means, footprint_bytes)
    summary = {
        "experiment": "EXP-A4-001",
        "seeds": [int(s) for s in per_seed],
        "per_seed": per_seed,
        "means": means,
        "footprints": footprints,
        "gate1_fp16_reference": gate1,
        "bars": bars,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
