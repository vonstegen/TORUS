"""EXP-AF-004: sequential freeze vs joint training of both STE latents.

Three-arm curriculum control at matched total budget and (for the claim
test) matched deployed storage, on a single correction site
(model.layers.0.mlp.down_proj by default):

  arm seq      stage 1: train primary latent, n_planes=1, N steps
               freeze primary (asserted bitwise)
               stage 2: train residual latent, n_planes=2, N steps
  arm joint    train primary + residual latents, n_planes=2, 2N steps
  arm t1_only  train primary latent, n_planes=1, 2N steps
               (budget control; deploys one plane, NOT part of the
               A-RP-003 claim test)

Every (arm, seed) consumes exactly 2N batches from the same cached
wikitext-103 token stream via af1_budget_control's sampler, so the
matched-budget property is a construction property. The training loop,
CE loss, and eval plumbing are af1's, imported — not copied.

Run with:

    python examples/af4_sequential_vs_joint.py --out-dir runs/a/EXP-AF-004/<ts>
    python examples/af4_sequential_vs_joint.py --arms seq --seeds 1 \
        --device cuda:0 --out-dir ...
"""
from __future__ import annotations

import sys as _sys
_sys.modules["triton"] = None  # see af1_budget_control header note

import argparse
import gc
import json
from pathlib import Path

import numpy as np


def _load_helper(path: Path, name: str):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(name, path / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_EXAMPLES = Path(__file__).resolve().parent
_af1 = _load_helper(_EXAMPLES, "af1_budget_control")
_eval_lm = _load_helper(_EXAMPLES, "eval_lm")

load_wikitext_ids = _af1.load_wikitext_ids
make_window_sampler = _af1.make_window_sampler
next_token_ce_loss = _af1.next_token_ce_loss
train_arm = _af1.train_arm

ARMS = ("seq", "joint", "t1_only")
# Eval plane count per arm — matches each arm's deployed form.
EVAL_N_PLANES = {"t1_only": 1, "joint": 2, "seq": 2}


def run_curriculum(arm, forward_fn, primary, residual, set_n_planes,
                   data, *, stage_steps, lr, momentum, grad_clip,
                   log_every, pad_id):
    """Run one AF4 arm's curriculum. Pure control logic — testable
    without a model.

    Args:
        arm: "seq" | "joint" | "t1_only".
        forward_fn: ids_tensor -> logits_tensor (reads n_planes via
            the closure behind `set_n_planes`).
        primary: list holding the primary STE latent tensor(s).
        residual: list holding the residual STE latent tensor(s)
            (unused for t1_only).
        set_n_planes: callable(int) that switches the forward's plane
            count.
        data: iterator of id batches; every arm consumes exactly
            2 * stage_steps batches.
    Returns:
        dict with stage histories and (for seq) the freeze check.
    Raises:
        RuntimeError: if the primary latent moved during stage 2 of
            arm seq (freeze invariant violation; run is INVALID per
            the manifest).
    """
    import torch

    if arm == "t1_only":
        set_n_planes(1)
        history = train_arm(
            forward_fn, list(primary), data,
            n_steps=2 * stage_steps, lr=lr, momentum=momentum,
            grad_clip=grad_clip, log_every=log_every, pad_id=pad_id,
        )
        return {"history": history}

    if arm == "joint":
        set_n_planes(2)
        history = train_arm(
            forward_fn, list(primary) + list(residual), data,
            n_steps=2 * stage_steps, lr=lr, momentum=momentum,
            grad_clip=grad_clip, log_every=log_every, pad_id=pad_id,
        )
        return {"history": history}

    if arm == "seq":
        set_n_planes(1)
        h1 = train_arm(
            forward_fn, list(primary), data,
            n_steps=stage_steps, lr=lr, momentum=momentum,
            grad_clip=grad_clip, log_every=log_every, pad_id=pad_id,
        )
        snapshot = [p.detach().clone() for p in primary]
        set_n_planes(2)
        h2 = train_arm(
            forward_fn, list(residual), data,
            n_steps=stage_steps, lr=lr, momentum=momentum,
            grad_clip=grad_clip, log_every=log_every, pad_id=pad_id,
        )
        freeze_ok = all(
            torch.equal(snap, p.detach())
            for snap, p in zip(snapshot, primary)
        )
        if not freeze_ok:
            raise RuntimeError(
                "freeze invariant violated: primary latent moved during "
                "stage 2 of arm seq — this run is INVALID per the "
                "EXP-AF-004 manifest"
            )
        return {
            "history_stage1": h1,
            "history_stage2": h2,
            "freeze_check": freeze_ok,
        }

    raise ValueError(f"unknown arm: {arm}")


def build_student(model_name: str, *, target_module: str, dtype: str,
                  device: str, attn_impl: str, calibrate_norm: bool):
    """Frozen base + two-plane STE on one module (shared by all arms).

    Only the STE latents are trainable-parameter candidates; the base
    is explicitly requires_grad_(False). Returns (student, forward_fn,
    primary, residual, set_n_planes).
    """
    from torus.train.hf_adapter import HFAdapterConfig, HFStudentAdapter

    cfg = HFAdapterConfig(
        model_name=model_name,
        target_modules=(target_module,),
        dtype=dtype,
        device=device,
        attn_implementation=attn_impl,
        calibrate_norm=calibrate_norm,
    )
    student = HFStudentAdapter(cfg)
    for p in student.model.parameters():
        p.requires_grad_(False)
    if len(student.ste_params) != 1:
        raise RuntimeError(
            f"expected exactly 1 patched STE at {target_module}, got "
            f"{len(student.ste_params)}"
        )
    ste = student.ste_params[0]
    ste.weight.requires_grad_(True)
    ste.residual_weight.requires_grad_(True)
    primary = [ste.weight]
    residual = [ste.residual_weight]

    def set_n_planes(n: int) -> None:
        student._current_n_planes = int(n)

    def forward_fn(ids):
        return student.model(input_ids=ids).logits

    return student, forward_fn, primary, residual, set_n_planes


def deployed_bytes(student, n_planes: int) -> int:
    """Packed ternary codes (2 bits/weight) + per-group fp16 scales.

    Reported cost row (not a decision axis for A-RP-003; see manifest).
    """
    ste = student.ste_params[0]
    def plane_bytes(weight) -> int:
        rows, cols = weight.shape
        n_groups = cols // ste.group_size
        codes = rows * cols * 2 // 8          # 2 bits per weight
        scales = rows * n_groups * 2          # fp16 scale per group
        return codes + scales

    total = plane_bytes(ste.weight)
    if n_planes >= 2 and ste.residual_weight is not None:
        total += plane_bytes(ste.residual_weight)
    return total


def run_one_seed(*, arm: str, seed: int, args, out_dir: Path,
                 tokenizer, pad_id: int, all_ids: np.ndarray) -> dict:
    """Train one arm at one seed and evaluate; write artifacts."""
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    arm_dir = out_dir / f"seed-{seed:03d}" / arm
    arm_dir.mkdir(parents=True, exist_ok=True)

    data = make_window_sampler(
        all_ids, args.batch_size, args.seq_len, seed=seed,
        device=args.device,
    )

    student, forward_fn, primary, residual, set_n_planes = build_student(
        args.model, target_module=args.target_module, dtype=args.dtype,
        device=args.device, attn_impl=args.attn_impl,
        calibrate_norm=not args.no_calibrate,
    )

    print(f"[af4] arm={arm} seed={seed}: training 2x{args.stage_steps} "
          f"steps (curriculum {arm})", flush=True)
    result = run_curriculum(
        arm, forward_fn, primary, residual, set_n_planes, data,
        stage_steps=args.stage_steps, lr=args.lr, momentum=args.momentum,
        grad_clip=args.grad_clip, log_every=args.log_every,
        pad_id=pad_id,
    )
    for key, rows in result.items():
        if key.startswith("history"):
            with open(arm_dir / f"{key}.jsonl", "w") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")

    n_planes_eval = EVAL_N_PLANES[arm]
    adapter_path = arm_dir / "adapter.npz"
    student.save_state(str(adapter_path))
    # Cast BEFORE apply_eval_mode (see af1 note: the pre-quantized
    # eval weights are plain attributes, not Parameters).
    model_for_eval = student.model.to(getattr(torch, args.eval_dtype))
    student.apply_eval_mode(n_planes=n_planes_eval)
    model_for_eval.eval()

    tasks = args.tasks.split(",")
    print(f"[af4] arm={arm} seed={seed}: evaluating {tasks} at "
          f"n_planes={n_planes_eval}", flush=True)
    results = _eval_lm.run_lm_eval(
        model_for_eval, tokenizer, tasks, batch_size=args.eval_batch_size,
        limit=args.limit,
    )
    summary = {
        "arm": arm,
        "seed": seed,
        "model": args.model,
        "target_module": args.target_module,
        "no_calibrate": bool(args.no_calibrate),
        "stage_steps": args.stage_steps,
        "total_steps": 2 * args.stage_steps,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "lr": args.lr,
        "limit": args.limit,
        "eval_n_planes": n_planes_eval,
        "deployed_bytes": deployed_bytes(student, n_planes_eval),
        "freeze_check": result.get("freeze_check"),
        "tasks": {},
    }
    for task, res in results["results"].items():
        for k, v in res.items():
            if k in ("acc,none", "acc_norm,none", "ppl,none",
                     "word_perplexity,none"):
                summary["tasks"][task] = {"metric": k, "value": v}
                break
    with open(arm_dir / "eval.summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(arm_dir / "eval.full.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    del student, model_for_eval, forward_fn, primary, residual
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def _pair_diff(arms: dict, first: str, second: str) -> dict:
    """(second - first) difference per task, in stderr-of-difference
    units. Sign convention is recorded raw; direction interpretation
    (ppl lower better, acc higher better) is applied at DECIDE time
    against the frozen manifest formulas."""
    a, b = arms.get(first, {}), arms.get(second, {})
    diff: dict[str, dict] = {}
    for task in set(a) & set(b):
        se = float(np.sqrt(a[task]["stderr"] ** 2 + b[task]["stderr"] ** 2))
        diff[task] = {
            f"mean_{second}_minus_{first}": b[task]["mean"] - a[task]["mean"],
            "stderr_of_difference": se,
            "difference_in_stderrs": (
                (b[task]["mean"] - a[task]["mean"]) / se if se > 0 else None
            ),
        }
    return diff


def aggregate(summaries: list[dict], out_dir: Path) -> dict:
    """Per-arm per-task mean/stderr across seeds + pairwise diffs."""
    by_arm: dict[str, dict[str, list[float]]] = {}
    for s in summaries:
        for task, rec in s["tasks"].items():
            by_arm.setdefault(s["arm"], {}).setdefault(task, []).append(
                rec["value"]
            )
    arms: dict[str, dict] = {}
    for arm, tasks in by_arm.items():
        arms[arm] = {}
        for task, vals in tasks.items():
            arr = np.asarray(vals, dtype=np.float64)
            arms[arm][task] = {
                "n": int(arr.size),
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                "stderr": float(arr.std(ddof=1) / np.sqrt(arr.size))
                if arr.size > 1 else 0.0,
                "values": [float(v) for v in vals],
            }
    diff = {
        # Primary comparison (the A-RP-003 claim test).
        "seq_vs_joint": _pair_diff(arms, "joint", "seq"),
        # Secondary context (never mixed into the claim verdict).
        "seq_vs_t1_only": _pair_diff(arms, "t1_only", "seq"),
        "joint_vs_t1_only": _pair_diff(arms, "t1_only", "joint"),
    }
    out = {"arms": arms, "difference": diff}
    with open(out_dir / "aggregate.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="allenai/OLMo-1B-0724-hf")
    p.add_argument("--target-module", default="model.layers.0.mlp.down_proj")
    p.add_argument("--arms", default=",".join(ARMS),
                   help="comma-separated subset of seq,joint,t1_only")
    p.add_argument("--seeds", default="1,2,3")
    p.add_argument("--stage-steps", type=int, default=500,
                   help="steps per stage; every arm trains 2x this")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--tasks", default="wikitext,arc_easy,lambada_openai")
    p.add_argument("--limit", type=int, default=None,
                   help="Eval example cap per task (None = full task; "
                        "AF4 confirmation tier uses full evals)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--eval-dtype", default="float16")
    p.add_argument("--eval-batch-size", type=int, default=4)
    p.add_argument("--attn-impl", default="eager")
    p.add_argument("--no-calibrate", action="store_true")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--ids-cache", type=Path,
                   default=Path("/tmp/wikitext103_train_ids.npy"),
                   help="disk cache for the tokenized wikitext-103 train ids")
    args = p.parse_args()

    for arm in args.arms.split(","):
        if arm not in ARMS:
            raise SystemExit(f"unknown arm {arm!r}; choose from {ARMS}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    all_ids = load_wikitext_ids(tokenizer, args.ids_cache)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    arms = args.arms.split(",")
    summaries: list[dict] = []
    for seed in seeds:
        for arm in arms:
            summaries.append(
                run_one_seed(arm=arm, seed=seed, args=args,
                             out_dir=args.out_dir, tokenizer=tokenizer,
                             pad_id=pad_id, all_ids=all_ids)
            )
    agg = aggregate(summaries, args.out_dir)
    print(json.dumps(agg["difference"], indent=2))


if __name__ == "__main__":
    main()
