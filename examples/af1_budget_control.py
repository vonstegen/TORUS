"""EXP-AF-001 — AF1 equal-training-budget control (A-RP-001).

Arm A (`t1_continued`): the FP16 base model, ALL weights trainable,
trained N steps with next-token cross-entropy on wikitext-103 train.

Arm B (`t1_t2`): the same FP16 base, frozen, plus a ternary correction
plane (T2) on `--target-module`; only the STE latents train, same N
steps, same objective, same data order, same optimizer settings.

The objective is next-token CE for BOTH arms. KD-against-frozen-T1 is
degenerate for arm A (KD(T1 vs T1) = 0 at init), so matched CE is the
only honest shared objective; it also isolates the T2 *architecture*
from the KD training signal (the conservative, falsification-grade
choice).

Matched budget is by construction: one shared training loop, one
shared data stream per seed (same seed -> identical batches for both
arms), one optimizer config. Only the trainable parameter set differs.

Usage:
    python examples/af1_budget_control.py \
        --model allenai/OLMo-1B-0724-hf \
        --target-module model.layers.0.mlp.down_proj \
        --n-steps 500 --batch-size 4 --seq-len 128 \
        --seeds 1,2,3 \
        --tasks wikitext,arc_easy,lambada_openai \
        --out-dir runs/a/EXP-AF-001/<ts>
"""
from __future__ import annotations

import sys as _sys

_sys.modules["triton"] = None  # see distill_run.py for why

import argparse
import gc
import importlib.util
import json
import time
from pathlib import Path

import numpy as np


def _load_helper(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


EXAMPLES = Path(__file__).resolve().parent
_eval_lm = _load_helper(EXAMPLES / "eval_lm.py", "eval_lm")


def load_wikitext_ids(tokenizer, cache_path: Path) -> np.ndarray:
    """Tokenize wikitext-103 train once; cache the token-id array.

    Same logic as distill_run.make_data_iter_wikitext (direct parquet
    download, bypassing the broken datasets-lib hash on this venv) but
    the resulting id array is cached to disk so the 6 arm×seed runs
    share a single tokenization pass.
    """
    import os
    if cache_path.exists():
        ids = np.load(cache_path)
        print(f"[af1] wikitext ids: loaded {len(ids):,} from cache", flush=True)
        return ids
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    print("[af1] downloading wikitext-103 train shards ...", flush=True)
    paths = [
        hf_hub_download(
            repo_id="wikitext",
            filename=f"wikitext-103-raw-v1/train-{i:05d}-of-00002.parquet",
            repo_type="dataset",
        )
        for i in range(2)
    ]
    tables = [pq.read_table(p, columns=["text"]) for p in paths]
    texts = sum((t.column("text").to_pylist() for t in tables), [])
    print(f"[af1] wikitext corpus: {len(texts):,} rows", flush=True)
    eot = tokenizer.eos_token_id or 0
    all_ids: list[int] = []
    for text in texts:
        if not text.strip():
            continue
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        ids.append(eot)
        all_ids.extend(ids)
    arr = np.asarray(all_ids, dtype=np.int64)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, arr)
    print(f"[af1] tokenized corpus: {len(arr):,} tokens (cached)", flush=True)
    return arr


def make_window_sampler(all_ids: np.ndarray, batch_size: int,
                        seq_len: int, seed: int, device: str):
    """Yield random windows as (batch, seq) int64 tensors on device.

    Same sampling scheme as distill_run.make_data_iter_wikitext:
    uniform random start into the shared token stream. Two samplers
    built with the same seed over the same ids yield IDENTICAL batch
    sequences — that is what makes the arm budgets matched.
    """
    import torch

    rng = np.random.default_rng(seed)
    n = batch_size * seq_len

    def batches():
        while True:
            max_start = len(all_ids) - n - 1
            if max_start <= 0:
                return
            start = int(rng.integers(0, max_start))
            window = np.asarray(all_ids[start: start + n], dtype=np.int64)
            window = window.reshape(batch_size, seq_len)
            yield torch.as_tensor(window, dtype=torch.long, device=device)

    return batches()


def next_token_ce_loss(logits, ids, pad_id: int):
    """Next-token cross-entropy: logits[:, :-1] vs ids[:, 1:]."""
    import torch
    import torch.nn.functional as F

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = ids[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=pad_id,
    )


def train_arm(forward_fn, params: list, data, *, n_steps: int, lr: float,
              momentum: float, grad_clip: float, log_every: int,
              pad_id: int) -> list[dict]:
    """Shared SGD training loop used by BOTH arms.

    `forward_fn(ids_tensor) -> logits_tensor` closes over the model;
    `params` is the trainable parameter set (the ONLY difference
    between arm A and arm B). Returns per-step loss records.
    """
    import torch

    opt = torch.optim.SGD(params, lr=lr, momentum=momentum)
    history: list[dict] = []
    t0 = time.perf_counter()
    for step in range(n_steps):
        ids = next(data)
        logits = forward_fn(ids)
        loss = next_token_ce_loss(logits, ids, pad_id)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, grad_clip)
        opt.step()
        if step % log_every == 0 or step == n_steps - 1:
            loss_val = float(loss.detach().cpu())
            history.append({
                "step": step,
                "loss": loss_val,
                "elapsed_seconds": time.perf_counter() - t0,
            })
            print(f"[af1]   step {step:4d}  loss {loss_val:.4f}",
                  flush=True)
    return history


def build_arm_a(model_name: str, *, dtype: str, device: str,
                attn_impl: str):
    """Arm A: FP16 base, ALL weights trainable."""
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=getattr(torch, dtype),
        attn_implementation=attn_impl,
    ).to(device)
    model.eval()  # QAT stance: no dropout-side randomness
    params = [p for p in model.parameters()]

    def forward_fn(ids):
        return model(input_ids=ids).logits

    return model, forward_fn, params


def build_arm_b(model_name: str, *, target_module: str, dtype: str,
                device: str, attn_impl: str, calibrate_norm: bool):
    """Arm B: frozen FP16 base + T2 correction plane on one module.

    Only the STE latents are trainable. The base is explicitly
    requires_grad_(False) so the CE backward does not allocate
    gradients for 1.2B frozen parameters.
    """
    import torch
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
    params = []
    for ste in student.ste_params:
        if hasattr(ste.weight, "requires_grad_"):
            ste.weight.requires_grad_(True)
            params.append(ste.weight)
        if getattr(ste, "residual_weight", None) is not None and hasattr(
            ste.residual_weight, "requires_grad_"
        ):
            ste.residual_weight.requires_grad_(True)
            params.append(ste.residual_weight)
    student._current_n_planes = 1

    def forward_fn(ids):
        return student.model(input_ids=ids).logits

    return student, forward_fn, params


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

    if arm == "t1_continued":
        handle, forward_fn, params = build_arm_a(
            args.model, dtype=args.dtype, device=args.device,
            attn_impl=args.attn_impl,
        )
    elif arm == "t1_t2":
        handle, forward_fn, params = build_arm_b(
            args.model, target_module=args.target_module, dtype=args.dtype,
            device=args.device, attn_impl=args.attn_impl,
            calibrate_norm=not args.no_calibrate,
        )
    else:
        raise ValueError(f"unknown arm: {arm}")

    print(f"[af1] arm={arm} seed={seed}: training {args.n_steps} steps "
          f"on {len(params)} parameter tensors", flush=True)
    history = train_arm(
        forward_fn, params, data,
        n_steps=args.n_steps, lr=args.lr, momentum=args.momentum,
        grad_clip=args.grad_clip, log_every=args.log_every,
        pad_id=pad_id,
    )
    with open(arm_dir / "history.jsonl", "w") as f:
        for row in history:
            f.write(json.dumps(row) + "\n")

    if arm == "t1_t2":
        adapter_path = arm_dir / "adapter.npz"
        handle.save_state(str(adapter_path))
        # Cast BEFORE apply_eval_mode: the pre-quantized eval weights
        # are stored as plain attributes (not Parameters/buffers), so
        # a later .to() would not convert them and F.linear would see
        # a float32 weight against float16 activations.
        model_for_eval = handle.model.to(getattr(torch, args.eval_dtype))
        handle.apply_eval_mode(n_planes=1)
    else:
        # Eval in float16 to match the EXP-A-001 / EXP-A-011 reference
        # ladder (FP16 baseline numbers are float16 evals).
        model_for_eval = handle.to(getattr(torch, args.eval_dtype))
    model_for_eval.eval()
    tasks = args.tasks.split(",")
    print(f"[af1] arm={arm} seed={seed}: evaluating {tasks}", flush=True)
    results = _eval_lm.run_lm_eval(
        model_for_eval, tokenizer, tasks, batch_size=args.eval_batch_size,
        limit=args.limit,
    )
    summary = {
        "arm": arm,
        "seed": seed,
        "model": args.model,
        "target_module": args.target_module if arm == "t1_t2" else None,
        "no_calibrate": bool(args.no_calibrate),
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "lr": args.lr,
        "limit": args.limit,
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

    del handle, model_for_eval, forward_fn, params
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def aggregate(summaries: list[dict], out_dir: Path) -> dict:
    """Per-arm per-task mean/stderr across seeds + arm differences."""
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
    diff: dict[str, dict] = {}
    a, b = arms.get("t1_continued", {}), arms.get("t1_t2", {})
    for task in set(a) & set(b):
        se = float(np.sqrt(a[task]["stderr"] ** 2 + b[task]["stderr"] ** 2))
        diff[task] = {
            "mean_t1_t2_minus_t1_continued": b[task]["mean"] - a[task]["mean"],
            "stderr_of_difference": se,
            "difference_in_stderrs": (
                (b[task]["mean"] - a[task]["mean"]) / se if se > 0 else None
            ),
        }
    out = {"arms": arms, "difference": diff}
    with open(out_dir / "aggregate.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="allenai/OLMo-1B-0724-hf")
    p.add_argument("--target-module", default="model.layers.0.mlp.down_proj")
    p.add_argument("--arms", default="t1_continued,t1_t2",
                   help="comma-separated subset of t1_continued,t1_t2")
    p.add_argument("--seeds", default="1,2,3")
    p.add_argument("--n-steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--tasks", default="wikitext,arc_easy,lambada_openai")
    p.add_argument("--limit", type=int, default=None,
                   help="Eval example cap per task (None = full task; "
                        "AF1 confirmation tier uses full evals)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--eval-dtype", default="float16")
    p.add_argument("--eval-batch-size", type=int, default=4)
    p.add_argument("--attn-impl", default="eager")
    p.add_argument("--no-calibrate", action="store_true")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--ids-cache", type=Path, default=Path("/tmp/wikitext103_train_ids.npy"),
                   help="disk cache for the tokenized wikitext-103 train ids")
    args = p.parse_args()

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
