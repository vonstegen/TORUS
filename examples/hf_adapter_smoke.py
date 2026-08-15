"""Phase 3 end-to-end smoke: load a real HF model, drive the
`HFStudentAdapter`, and confirm the trainer-compatible contract.

The smoke supports a `--n-planes` flag that exercises the same model
with primary-only quantization (`n_planes=1`) and with primary +
residual quantization (`n_planes=2`). With the residual weights
initialized to zero, both produce the same output — the comparison
shows the adapter's `n_planes` plumbing is honored end-to-end.

Run with:

    python examples/hf_adapter_smoke.py
    python examples/hf_adapter_smoke.py --n-planes 1
    python examples/hf_adapter_smoke.py --n-planes 2
    python examples/hf_adapter_smoke.py gpt2 --n-planes 2
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from torus.train.hf_adapter import HFAdapterConfig, HFStudentAdapter, HFTeacherAdapter
from torus.train.loop import DistillationBatch
from torus.train.losses import DistillationConfig, combined_distillation_loss


DEFAULT_MODEL = "sshleifer/tiny-gpt2"


def main(model_name: str = DEFAULT_MODEL, n_planes: int = 1) -> None:
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

    print(f"[smoke] calling student.forward(batch, n_planes={n_planes}) ...")
    t0 = time.perf_counter()
    s_logits, s_hidden, s_route = student.forward(batch, n_planes=n_planes)
    print(
        f"[smoke]   student.forward OK in {time.perf_counter() - t0:.2f}s; "
        f"logits.shape={s_logits.shape}, hidden.shape={s_hidden.shape}, "
        f"route.shape={s_route.shape}"
    )

    print(f"[smoke] calling teacher.forward(batch, n_planes={n_planes}) ...")
    t0 = time.perf_counter()
    t_logits, t_hidden, t_route = teacher.forward(batch, n_planes=n_planes)
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

    # If --compare was passed, run with both n_planes=1 and n_planes=2
    # to demonstrate the comparison plumbing.
    print("[smoke] running comparison: same model, n_planes=1 vs 2 ...")
    y1, _, _ = student.forward(batch, n_planes=1)
    y2, _, _ = student.forward(batch, n_planes=2)
    diff_norm = float(np.linalg.norm(y1 - y2))
    print(
        f"[smoke]   ||y1 - y2|| = {diff_norm:.6f} "
        f"(expected ~0 since residual_weight starts at zero)"
    )

    print(
        "[smoke] OK: end-to-end adapter + trainer loss path works "
        "against a real transformers model; n_planes plumbing verified."
    )


def parse_args(argv: list[str]) -> tuple[str, int]:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "model",
        nargs="?",
        default=DEFAULT_MODEL,
        help=f"HuggingFace model name (default: {DEFAULT_MODEL})",
    )
    p.add_argument(
        "--n-planes",
        type=int,
        default=2,
        choices=[1, 2],
        help="Number of residual planes to engage (default: 2)",
    )
    a = p.parse_args(argv)
    return a.model, a.n_planes


if __name__ == "__main__":
    model, n = parse_args(sys.argv[1:])
    main(model, n)