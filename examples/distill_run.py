"""Phase 8 demo: end-to-end distillation on a real HF model.

Loads a tiny GPT-2 model, drives the `DistillationTrainer` with a
curriculum (primary-only -> primary+residual), logs per-step loss
to disk, and reports the loss curves when done.

Run with:

    python examples/distill_run.py
    python examples/distill_run.py --n-steps 200 --probe-rows 4
    python examples/distill_run.py --model sshleifer/tiny-gpt2 --n-steps 100

The trainer uses finite-difference gradients with a configurable
per-module `probe_rows` budget (default 1, i.e. one probe per
module per step). This is intentionally a coarse but real
gradient direction — Phase 8's contribution is the wiring; the
training-loop follow-up (Phase 8+) replaces finite differences
with torch autograd.
"""
from __future__ import annotations

# Workaround for Legion's missing python3.14-dev headers:
# OLMo's attention/RoPE lazily call into Triton on first
# forward. Triton's gcc build of `cuda_utils.c` then fails
# with "Python.h: No such file or directory" because the
# python3.14 headers aren't installed and we have no sudo.
# Setting `sys.modules["triton"] = None` BEFORE torch is
# imported makes subsequent `import triton` raise
# ImportError, which forces eager fallbacks everywhere.
# Order matters: if torch is imported first, it caches
# triton-using ops that fail later. With
# `attn_implementation="sdpa"` (the default when not
# specified) or `eager`, no Triton kernel is ever invoked
# by the model forward.
import sys as _sys
_sys.modules["triton"] = None

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from torus.train.curriculum import CurriculumSchedule
from torus.train.hf_adapter import HFAdapterConfig, HFStudentAdapter, HFTeacherAdapter
from torus.train.loop import DistillationBatch, DistillationTrainer, TrainingConfig
from torus.train.losses import DistillationConfig

def make_data_iter_random(vocab_size: int, batch_size: int, seq_len: int, seed: int):
    """Yield token batches drawn uniformly at random from the vocab."""
    rng = np.random.default_rng(seed)

    def iter_batches():
        while True:
            yield DistillationBatch(
                inputs=rng.integers(0, vocab_size, size=(batch_size, seq_len)).astype(np.int64),
            )

    return iter_batches()

def make_data_iter_wikitext(tokenizer, batch_size: int, seq_len: int, seed: int):
    """Yield real token batches from wikitext-103 train split.

    Bypasses the broken `datasets` library in this Python 3.14 venv
    (its wikitext hash is unresolvable) by downloading parquet
    shards directly via `huggingface_hub` and reading with
    `pyarrow.parquet`. The first call downloads the shards (~500MB)
    into the HF cache; subsequent calls reuse them.
    """
    import os
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    print("[distill] downloading wikitext-103 train shards ...")
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
    print(f"[distill] wikitext corpus: {len(texts):,} rows")

    # Concatenate non-empty rows into a single token stream.
    eot = tokenizer.eos_token_id or 0
    all_ids: list[int] = []
    for text in texts:
        if not text.strip():
            continue
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        ids.append(eot)
        all_ids.extend(ids)
    print(f"[distill] tokenized corpus: {len(all_ids):,} tokens")

    rng = np.random.default_rng(seed)
    n = batch_size * seq_len

    def iter_batches():
        # Yield random windows from the tokenized corpus.
        while True:
            max_start = len(all_ids) - n - 1
            if max_start <= 0:
                return
            start = int(rng.integers(0, max_start))
            window = np.asarray(all_ids[start: start + n], dtype=np.int64)
            window = window.reshape(batch_size, seq_len)
            yield DistillationBatch(inputs=window)

    return iter_batches()

def make_data_iter(tokenizer, batch_size: int, seq_len: int, seed: int, dataset: str = "random"):
    """Backwards-compatible wrapper; routes to the right iter."""
    if dataset == "random":
        return make_data_iter_random(
            tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else 50304,
            batch_size, seq_len, seed,
        )
    if dataset == "wikitext":
        return make_data_iter_wikitext(tokenizer, batch_size, seq_len, seed)
    raise ValueError(f"unknown dataset: {dataset}")


def run_one(
    args,
    *,
    model_name: str,
    teacher_model_name: str | None = None,
    target_modules: tuple[str, ...] = ("c_attn", "c_proj"),
    device: str = "cuda",
    dtype: str = "float32",
    n_steps: int,
    probe_rows: int,
    probe_residual: bool = False,
    residual_lr_scale: float = 0.1,
    curriculum_planes: list[int],
    curriculum_steps_per_stage: list[int],
    batch_size: int,
    seq_len: int,
    log_path: Path,
    seed: int = 0,
) -> dict:
    """Run one distillation and return final stats."""
    cfg = HFAdapterConfig(
        model_name=model_name,
        target_modules=target_modules,
        dtype=dtype,
        device=device,
        attn_implementation=getattr(args, 'attn_impl', 'eager'),
    )
    print(f"[distill] loading student from {model_name!r} ...")
    student = HFStudentAdapter(cfg)
    if getattr(args, "load_adapter", None):
        student.load_state(args.load_adapter)
        print(f"[distill]   loaded adapter weights from {args.load_adapter}")
    if getattr(args, "perturb_residual", False):
        import torch
        for rp in student.residual_params:
            rp.data.add_(torch.randn_like(rp) * 0.05)
        print(f"[distill]   perturbed residual weights with N(0, 0.05) noise")
    t0 = time.perf_counter()
    if teacher_model_name is not None and teacher_model_name != model_name:
        teacher_cfg = HFAdapterConfig(
            model_name=teacher_model_name,
            target_modules=(),  # teacher is FP, no quantization
            dtype=dtype,
            device=device,
            attn_implementation=getattr(args, 'attn_impl', 'eager'),
        )
        teacher = HFTeacherAdapter(teacher_cfg)
    vocab = student.model.config.vocab_size
    dataset_name = getattr(args, "dataset", "random")
    if dataset_name == "wikitext":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        data = make_data_iter_wikitext(tokenizer, batch_size, seq_len, seed=seed)
    else:
        data = make_data_iter_random(vocab, batch_size, seq_len, seed=seed)
    train_cfg = TrainingConfig(
        n_steps=n_steps,
        log_every=max(1, n_steps // 20),
        learning_rate=1e-3,
        probe_rows=probe_rows,
        probe_cols=getattr(args, 'probe_cols', 0),
        probe_residual=getattr(args, 'probe_residual', False),
        residual_lr_scale=getattr(args, 'residual_lr_scale', 0.1),
        residual_warmup_steps=getattr(args, 'residual_warmup', 0),
    )
    curriculum = CurriculumSchedule.progressive(
        steps_per_stage=curriculum_steps_per_stage,
    )

    history = []
    print(f"[distill] starting trainer: {n_steps} steps, probe_rows={probe_rows}")
    t0 = time.perf_counter()
    history = DistillationTrainer(
        student_params=student.ste_params,
        forward_student=student.forward,
        forward_teacher=teacher.forward,  # callable; the trainer also
                                          # discovers the adapter's
                                          # forward_torch via __self__
        data=data,
        loss_cfg=DistillationConfig(),
        curriculum=curriculum,
        train_cfg=train_cfg,
    ).fit()
    elapsed = time.perf_counter() - t0
    print(f"[distill] done: {n_steps} steps in {elapsed:.1f}s ({elapsed / n_steps:.2f}s/step)")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(s) for s in history]
    initial_loss = rows[0]["loss"] if rows else None
    final_loss = rows[-1]["loss"] if rows else None
    with log_path.open("w") as f:
        json.dump(
            {
                "model_name": model_name,
                "teacher_model_name": teacher_model_name,
                "target_modules": list(target_modules),
                "n_steps": n_steps,
                "probe_rows": probe_rows,
                "curriculum_planes": curriculum_planes,
                "elapsed_seconds": elapsed,
                "history": rows,
            },
            f,
            indent=2,
        )
    if getattr(args, "save_adapter", None):
        student.save_state(args.save_adapter)
        print(f"[distill] saved adapter weights to {args.save_adapter}")
    return {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "delta": (initial_loss - final_loss) if (initial_loss is not None and final_loss is not None) else None,
        "elapsed": elapsed,
        "n": len(history),
        "log_path": str(log_path),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="sshleifer/tiny-gpt2")
    p.add_argument("--teacher-model", default=None,
                   help="Override the teacher model. Defaults to --model (self-distillation).")
    p.add_argument("--target-modules", default="c_attn,c_proj",
                   help="Comma-separated module names to quantize (e.g. q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj for OLMo)")
    p.add_argument("--n-steps", type=int, default=200)
    p.add_argument("--probe-rows", type=int, default=1,
                   help="finite-difference probes per module per step")
    p.add_argument("--probe-cols", type=int, default=0,
                   help="columns probed per row (0 = same as --probe-rows)")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument(
        "--curriculum",
        default="1:500,2:500",
        help="plane_count:steps_per_stage, comma-separated",
    )
    p.add_argument("--log-dir", default="/tmp/torus_distill_logs")
    p.add_argument("--label", default="default")
    p.add_argument("--probe-residual", action="store_true",
                   help="Also perturb STE.residual_weight at the same (r, c)")
    p.add_argument("--perturb-residual", action="store_true",
                   help="Initialize residual weights with random noise")
    p.add_argument("--residual-lr-scale", type=float, default=0.1,
                   help="Residual plane LR = learning_rate * this")
    p.add_argument("--residual-warmup", type=int, default=0,
                   help="Ramp residual LR from 0 -> target over this many steps")
    p.add_argument("--device", default="cuda",
                   help="Device for student/teacher (cpu, cuda, cuda:0, ...)")
    p.add_argument("--dtype", default="float32",
                   help="Torch dtype for model weights (float32, float16, bfloat16)")
    p.add_argument("--attn-impl", default="eager",
                   help="HF attn_implementation: eager, sdpa, flash_attention_2")
    p.add_argument("--save-adapter", default=None,
                   help="At the end of the run, save the trained STE+residual weights to this .npz file")
    p.add_argument("--load-adapter", default=None,
                   help="Before training, load STE+residual weights from this .npz file (random init of matched shape skipped)")
    p.add_argument("--dataset", default="random", choices=["random", "wikitext"],
                   help="Training data source. 'wikitext' streams wikitext-103 with the model's tokenizer.")
    args = p.parse_args()

    planes = []
    step_counts = []
    for entry in args.curriculum.split(","):
        plane, steps = entry.split(":")
        planes.append(int(plane))
        step_counts.append(int(steps))

    if sum(step_counts) < args.n_steps:
        # Extend the last stage to cover the remaining steps.
        step_counts[-1] += args.n_steps - sum(step_counts)

    log_path = (
        Path(args.log_dir) / f"{args.label}.json"
    )
    result = run_one(
        args,
        model_name=args.model,
        teacher_model_name=getattr(args, 'teacher_model', None),
        target_modules=tuple(getattr(args, 'target_modules', 'c_attn,c_proj').split(',')),
        device=getattr(args, 'device', 'cuda'),
        dtype=getattr(args, 'dtype', 'float32'),
        n_steps=args.n_steps,
        probe_rows=args.probe_rows,
        probe_residual=getattr(args, 'probe_residual', False),
        residual_lr_scale=getattr(args, 'residual_lr_scale', 0.1),
        curriculum_planes=planes,
        curriculum_steps_per_stage=step_counts,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        log_path=log_path,
    )
    print()
    print(f"[distill] initial loss: {result.get('initial_loss'):.4f}")
    print(f"[distill] final   loss: {result.get('final_loss'):.4f}")
    if result.get("initial_loss") is not None:
        delta = result["delta"]
        sign = "improved" if delta > 0 else "regressed"
        print(f"[distill] delta: {delta:+.4f} ({sign})")
    print(f"[distill] log: {result.get('log_path')}")


if __name__ == "__main__":
    main()
