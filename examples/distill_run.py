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


def make_data_iter(vocab_size: int, batch_size: int, seq_len: int, seed: int):
    """Yield token batches drawn uniformly at random from the vocab."""
    rng = np.random.default_rng(seed)

    def iter_batches():
        while True:
            yield DistillationBatch(
                inputs=rng.integers(0, vocab_size, size=(batch_size, seq_len)).astype(np.int64),
            )

    return iter_batches()


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
    student = HFStudentAdapter(cfg)
    if getattr(args, "perturb_residual", False):
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
    else:
        teacher = HFTeacherAdapter(cfg)
    print(f"[distill]   teacher loaded in {time.perf_counter() - t0:.1f}s")

    vocab = student.model.config.vocab_size
    data = make_data_iter(vocab, batch_size, seq_len, seed=seed)
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
    with log_path.open("w") as f:
        json.dump(
            {
                "model_name": model_name,
                "n_steps": n_steps,
                "probe_rows": probe_rows,
                "elapsed_seconds": elapsed,
                "curriculum_planes": curriculum_planes,
                "history": rows,
            },
            f,
            indent=2,
        )
    print(f"[distill] wrote {log_path}")

    if not history:
        return {"loss": None, "n": 0, "elapsed": elapsed}

    final_loss = float(history[-1].loss)
    initial_loss = float(history[0].loss)
    return {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "delta": initial_loss - final_loss,
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
        default="1:50,2:150",
        help="plane_count:steps_per_stage, comma-separated",
    )
    p.add_argument("--log-dir", default="/tmp/torus_distill_logs")
    p.add_argument("--label", default="default")
    p.add_argument("--probe-residual", action="store_true",
                   help="Also perturb STE.residual_weight at the same (r, c)")
    p.add_argument("--residual-warmup", type=int, default=0,
                   help="Ramp residual LR from 0 -> target over this many steps")
    p.add_argument("--device", default="cuda",
                   help="Device for student/teacher (cpu, cuda, cuda:0, ...)")
    p.add_argument("--dtype", default="float32",
                   help="Torch dtype for model weights (float32, float16, bfloat16)")
    p.add_argument("--attn-impl", default="eager",
                   help="HF attn_implementation: eager, sdpa, flash_attention_2")
    args = p.parse_args()
    # Parse the curriculum: "1:50,2:150" -> [(1, 50), (2, 150)].
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
    if result.get("initial_loss") is not None:
        delta = result["delta"]
        sign = "improved" if delta > 0 else "regressed"
        print(f"[distill] delta: {delta:+.4f} ({sign})")
    print(f"[distill] log: {result.get('log_path')}")


if __name__ == "__main__":
    main()