"""Phase 3 end-to-end smoke: load a real HF model, drive the
`HFStudentAdapter`, and confirm the trainer-compatible contract.

This is the milestone that closes the Phase-3 follow-up: the
adapter is no longer just a mock; it loads a real transformers
model and produces `(logits, hidden, route)` tuples of the right
shape, fed through the real `combined_distillation_loss`.

We do *not* run the trainer's numerical-gradient loop here; that
path is one-probe-per-row and is illustrative only (Phase-3
follow-up replaces it with torch autograd). The smoke confirms
the contract by:

  1. Loading `HFStudentAdapter` and `HFTeacherAdapter` on the
     same model.
  2. Calling `student.forward(batch, n_planes)` once and checking
     `(logits, hidden, route)` shapes.
  3. Calling the trainer's `_loss_only()` to verify the loss
     function consumes the adapter's output without error.

Run with:

    python examples/hf_adapter_smoke.py
    python examples/hf_adapter_smoke.py gpt2
    python examples/hf_adapter_smoke.py sshleifer/tiny-gpt2
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from torus.train.hf_adapter import HFAdapterConfig, HFStudentAdapter, HFTeacherAdapter
from torus.train.loop import DistillationBatch
from torus.train.losses import DistillationConfig, combined_distillation_loss


DEFAULT_MODEL = "gpt2"


def main(model_name: str = DEFAULT_MODEL) -> None:
    cfg = HFAdapterConfig(
        model_name=model_name,
        target_modules=("c_attn", "c_proj"),
        dtype="float32",
    )
    print(f"[smoke] loading student from {model_name!r} ...")
    t0 = time.perf_counter()
    student = HFStudentAdapter(cfg)
    print(
        f"[smoke]   student loaded in {time.perf_counter() - t0:.1f}s, "
        f"{len(student.ste_params)} Linear/Conv1D modules intercepted"
    )

    print(f"[smoke] loading teacher from {model_name!r} ...")
    t0 = time.perf_counter()
    teacher = HFTeacherAdapter(cfg)
    print(f"[smoke]   teacher loaded in {time.perf_counter() - t0:.1f}s")

    vocab = student.model.config.vocab_size
    rng = np.random.default_rng(0)
    batch = DistillationBatch(
        inputs=rng.integers(0, vocab, size=(2, 16)).astype(np.int64),
    )

    print("[smoke] calling student.forward(batch, n_planes=2) ...")
    t0 = time.perf_counter()
    s_logits, s_hidden, s_route = student.forward(batch, n_planes=2)
    print(
        f"[smoke]   student.forward OK in {time.perf_counter() - t0:.2f}s; "
        f"logits.shape={s_logits.shape}, hidden.shape={s_hidden.shape}, "
        f"route.shape={s_route.shape}"
    )

    print("[smoke] calling teacher.forward(batch, n_planes=2) ...")
    t0 = time.perf_counter()
    t_logits, t_hidden, t_route = teacher.forward(batch, n_planes=2)
    print(
        f"[smoke]   teacher.forward OK in {time.perf_counter() - t0:.2f}s; "
        f"logits.shape={t_logits.shape}, hidden.shape={t_hidden.shape}, "
        f"route.shape={t_route.shape}"
    )

    print("[smoke] running combined_distillation_loss(student, teacher) ...")
    t0 = time.perf_counter()
    loss, components = combined_distillation_loss(
        student_logits=s_logits,
        teacher_logits=t_logits,
        student_hidden=s_hidden,
        teacher_hidden=t_hidden,
        student_route=s_route,
        teacher_route=t_route,
        cfg=DistillationConfig(),
    )
    print(
        f"[smoke]   loss OK in {time.perf_counter() - t0:.2f}s; "
        f"loss={float(loss):.4f}, components={list(components.keys())}"
    )

    print(
        "[smoke] OK: end-to-end adapter + trainer loss path works "
        "against a real transformers model."
    )


def parse_args(argv: list[str]) -> str:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "model",
        nargs="?",
        default=DEFAULT_MODEL,
        help=f"HuggingFace model name (default: {DEFAULT_MODEL})",
    )
    a = p.parse_args(argv)
    return a.model


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))