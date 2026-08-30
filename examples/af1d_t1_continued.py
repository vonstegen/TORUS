"""EXP-AF-001-D — damaged-start T1-only continuation vs T2 plane.

Arms (per seed, matched AF2-D budget):
  A t1_only_fp16:      damaged base, ALL weights trainable (FP16
                       continuation from the damaged start).
  B t1_only_ternary:   damaged base frozen; only the damaged
                       layer's ternary CODES trainable with frozen
                       per-group scales (sign-STE), bit-identical
                       at step 0 by construction.

Frozen comparison: the AF2-D verdict means for t2_ternary and
random_t2_ternary (verdict-D.md).

Frozen bars (acceptance-bar item 1 at the AF2-D site):
  PASS: (t2 - armA) > 2 sd-of-difference on >= 2 of 3 metrics, no
        >= 2 sigma regression on the remaining one.
  FAIL: (armA - t2) > 2 sd-of-difference on >= 1 metric.
  else NULL (no separation at this budget).
  Arm A instability kill: arm A post-train ppl worse than the
  damaged base (425.76).

Pure rule functions are torch-free; the CLI needs Legion.

Usage:
  .venv/bin/python examples/af1d_t1_continued.py \
      --out-dir runs/a/EXP-AF-001-D/<ts> --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# Frozen AF2-D comparison means (verdict-D.md, n=3):
# (mean, stderr) per metric.
T2_MEANS = {
    "wikitext": (20.96, 1.53),
    "arc_easy": (0.600, 0.004),
    "lambada_openai": (0.545, 0.003),
}
RANDOM_T2_MEANS = {
    "wikitext": (367.62, None),
    "arc_easy": (0.4954, None),
    "lambada_openai": (0.2554, None),
}
DAMAGED_BASE_PPL = 425.76
PRE_TRAIN_BAND = (400.0, 460.0)
BAR_SIGMA = 2.0
METRICS = ("wikitext", "arc_easy", "lambada_openai")

# +1: higher is better; -1: lower is better (wikitext ppl).
METRIC_DIRECTION = {"wikitext": -1.0, "arc_easy": 1.0,
                    "lambada_openai": 1.0}
def sd_of_difference(se1: float | None, se2: float | None) -> float:
    return math.sqrt((se1 or 0.0) ** 2 + (se2 or 0.0) ** 2)

def apply_bars(arm_a: dict[str, tuple[float, float]],
               arm_b: dict[str, tuple[float, float]],
               t2: dict[str, tuple[float, float]] = T2_MEANS) -> dict:
    """Frozen acceptance-bar item 1 application (arm A vs frozen T2).

    z > 0 means T2 better than the arm on that metric (per-metric
    direction: wikitext ppl lower-is-better).
    """
    z_a = {}
    z_b = {}
    for metric in METRICS:
        a_mean, a_se = arm_a[metric]
        t_mean, t_se = t2[metric]
        sd = sd_of_difference(a_se, t_se)
        direction = METRIC_DIRECTION[metric]
        z_a[metric] = (direction * (t_mean - a_mean) / sd) if sd > 0 else 0.0
        b_mean, b_se = arm_b[metric]
        sd_b = sd_of_difference(b_se, t_se)
        z_b[metric] = (direction * (t_mean - b_mean) / sd_b) if sd_b > 0 else 0.0
    pass_n = sum(z > BAR_SIGMA for z in z_a.values())
    regressions = [m for m in METRICS if z_a[m] < -BAR_SIGMA]
    fail_n = sum(z < -BAR_SIGMA for z in z_a.values())
    if pass_n >= 2 and not regressions:
        decision = "PASS"
    elif fail_n >= 1:
        decision = "FAIL"
    else:
        decision = "NULL"
    return {
        "z_t2_minus_armA": z_a,
        "z_t2_minus_armB": z_b,  # descriptive (arm B has no gate)
        "n_pass_metrics": pass_n,
        "regressions": regressions,
        "decision": decision,
    }

# ---- CLI --------------------------------------------------------------


def _extract_eval(pre: dict, task: str) -> float:
    """Extract the metric value from a raw simple_evaluate dict.

    Mirrors the tournament driver's pre_train_eval_if_damaged:
    first non-stderr key containing 'acc' or 'word_perplexity'.
    """
    for k, v in pre["results"][task].items():
        if "_stderr" in k:
            continue
        if "acc" in k or "word_perplexity" in k:
            return float(v)
    raise KeyError(f"no metric found for {task}: {pre['results'][task]}")


def _damage(model, target):
    from examples.af2_storage_tournament import damage_target_module
    damage_target_module(target, group_size=128, threshold=0.7)
def _patch_sign_ste(module, w0):
    """Replace module.forward: weight = sign(c) * s (frozen scales).

    c init = w0 / s (values {0, +/-1}); forward bit-identical to w0.
    """
    import torch
    import torch.nn.functional as F

    w0f = w0.detach().float()
    out_f, in_f = w0f.shape
    grouped = w0f.reshape(out_f, -1, 128)
    s = grouped.abs().amax(dim=-1).clamp(min=1e-8)  # (out, n_groups)
    s_full = s.repeat_interleave(128, dim=-1)
    c = torch.nn.Parameter((w0f / s_full).to(w0.dtype))
    bias = module.bias

    def fwd(x):
        q = c + (torch.sign(c) * s_full.to(c.dtype) - c).detach()
        return F.linear(x, q, bias)

    module.forward = fwd
    return [c]


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
        resolve_target_module,
    )

    all_ids = np.load(args.ids_cache)
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B-0724-hf")
    pad_id = tokenizer.pad_token_id or 0

    per_seed: dict[str, dict[str, dict]] = {}
    for seed_s in args.seeds.split(","):
        seed = int(seed_s)
        seed_dir = args.out_dir / f"seed-{seed:03d}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        per_seed[seed_s] = {}

        # --- shared damaged base + pre-train verification ------------
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = build_base("allenai/OLMo-1B-0724-hf",
                           dtype="float16", device=args.device)
        target = resolve_target_module(
            model, "model.layers.0.mlp.down_proj")
        _damage(model, target)
        pre = eval_arm(model, tokenizer,
                       tasks=["wikitext", "arc_easy", "lambada_openai"],
                       batch_size=4)
        pre_summary = {task: _extract_eval(pre, task)
                       for task in METRICS}
        (seed_dir / "pre_train_eval.json").write_text(
            json.dumps(pre_summary, indent=2))
        band_ok = (PRE_TRAIN_BAND[0] <= pre_summary["wikitext"]
                   <= PRE_TRAIN_BAND[1])
        print(f"[af1d] seed {seed} pre-train ppl="
              f"{pre_summary['wikitext']} band_ok={band_ok}", flush=True)
        del model
        torch.cuda.empty_cache()

        # --- arm A: whole-model FP16 continuation ----------------------
        torch.manual_seed(seed)
        np.random.seed(seed)
        modelA = build_base("allenai/OLMo-1B-0724-hf",
                            dtype="float16", device=args.device)
        targetA = resolve_target_module(
            modelA, "model.layers.0.mlp.down_proj")
        _damage(modelA, targetA)
        for param in modelA.parameters():
            param.requires_grad_(True)
        paramsA = list(modelA.parameters())

        def fwdA(ids, _m=modelA):
            return _m(input_ids=ids).logits

        dataA = make_window_sampler(all_ids, 4, 128, seed=seed,
                                    device=args.device)
        histA = train_arm(
            fwdA, paramsA, dataA,
            n_steps=args.n_steps, lr=args.lr, momentum=args.momentum,
            grad_clip=args.grad_clip, log_every=50,
            pad_id=pad_id)
        evalA = eval_arm(modelA, tokenizer,
                         tasks=["wikitext", "arc_easy", "lambada_openai"],
                         batch_size=4)
        armA_dir = seed_dir / "t1_only_fp16"
        armA_dir.mkdir(exist_ok=True)
        (armA_dir / "history.jsonl").write_text(
            "\n".join(json.dumps(h) for h in histA))
        evalA_summary = {task: _extract_eval(evalA, task)
                          for task in METRICS}
        (armA_dir / "eval.summary.json").write_text(
            json.dumps(evalA_summary, indent=2))
        per_seed[seed_s]["t1_only_fp16"] = evalA_summary
        print(f"[af1d] seed {seed} armA: "
              f"{per_seed[seed_s]['t1_only_fp16']}", flush=True)
        del modelA
        torch.cuda.empty_cache()

        # --- arm B: ternary-STE continuation of the damaged layer -----
        torch.manual_seed(seed)
        np.random.seed(seed)
        modelB = build_base("allenai/OLMo-1B-0724-hf",
                            dtype="float16", device=args.device)
        targetB = resolve_target_module(
            modelB, "model.layers.0.mlp.down_proj")
        _damage(modelB, targetB)
        for param in modelB.parameters():
            param.requires_grad_(False)
        codes = _patch_sign_ste(targetB, targetB.weight)

        def fwdB(ids, _m=modelB):
            return _m(input_ids=ids).logits

        dataB = make_window_sampler(all_ids, 4, 128, seed=seed,
                                    device=args.device)
        histB = train_arm(
            fwdB, codes, dataB,
            n_steps=args.n_steps, lr=args.lr, momentum=args.momentum,
            grad_clip=args.grad_clip, log_every=50,
            pad_id=pad_id)
        evalB = eval_arm(modelB, tokenizer,
                         tasks=["wikitext", "arc_easy", "lambada_openai"],
                         batch_size=4)
        armB_dir = seed_dir / "t1_only_ternary"
        armB_dir.mkdir(exist_ok=True)
        (armB_dir / "history.jsonl").write_text(
            "\n".join(json.dumps(h) for h in histB))
        evalB_summary = {task: _extract_eval(evalB, task)
                          for task in METRICS}
        (armB_dir / "eval.summary.json").write_text(
            json.dumps(evalB_summary, indent=2))
        per_seed[seed_s]["t1_only_ternary"] = evalB_summary
        print(f"[af1d] seed {seed} armB: "
              f"{per_seed[seed_s]['t1_only_ternary']}", flush=True)
        del modelB
        torch.cuda.empty_cache()

    # --- aggregate + frozen bars --------------------------------------
    import statistics

    def agg(arm: str):
        out = {}
        for metric in METRICS:
            vals = [per_seed[s][arm][metric]
                    for s in per_seed]
            out[metric] = (statistics.fmean(vals),
                           (statistics.stdev(vals) / len(vals) ** 0.5
                            if len(vals) > 1 else 0.0))
        return out

    armA = agg("t1_only_fp16")
    armB = agg("t1_only_ternary")
    bars = apply_bars(armA, armB)
    armA_ppl = armA["wikitext"][0]
    instability = armA_ppl > DAMAGED_BASE_PPL
    summary = {
        "experiment": "EXP-AF-001-D",
        "seeds": [int(s) for s in per_seed],
        "per_seed": per_seed,
        "armA_t1_only_fp16": armA,
        "armB_t1_only_ternary": armB,
        "frozen_t2_means": T2_MEANS,
        "frozen_random_t2_means": RANDOM_T2_MEANS,
        "armA_instability_kill": instability,
        "bars": bars,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
