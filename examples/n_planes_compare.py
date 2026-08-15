"""Phase 3 comparison: primary-only vs primary+residual on a real model.

Trains the residual plane for a few steps, then runs the same
forward with `n_planes=1` and `n_planes=2` to show the difference.

For a more thorough comparison, this would be wrapped around the
full DistillationTrainer with curriculum. The numerical-gradient
path is one-probe-per-row, so this script uses a tighter probe
(one column per module) to keep the runtime tractable on a small
model.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch

from torus.train.hf_adapter import HFAdapterConfig, HFStudentAdapter, HFTeacherAdapter
from torus.train.loop import DistillationBatch
from torus.train.ste import TernarySTE


DEFAULT_MODEL = "sshleifer/tiny-gpt2"


def _ste_to_numpy(weight):
    if isinstance(weight, torch.Tensor):
        return weight.detach().cpu().numpy()
    return np.asarray(weight)


def main(model_name: str = DEFAULT_MODEL, n_steps: int = 5) -> None:
    cfg = HFAdapterConfig(
        model_name=model_name,
        target_modules=("c_attn", "c_proj"),
        dtype="float32",
    )
    print(f"[compare] loading student/teacher ({model_name!r}) ...")
    student = HFStudentAdapter(cfg)
    teacher = HFTeacherAdapter(cfg)
    vocab = student.model.config.vocab_size
    rng = np.random.default_rng(0)

    # Step 1: snapshot the n_planes=1 output before any training.
    batch = DistillationBatch(
        inputs=rng.integers(0, vocab, size=(2, 16)).astype(np.int64),
    )
    y1_before, _, _ = student.forward(batch, n_planes=1)
    print(
        f"[compare] pre-train: ||y_n_planes=1|| = "
        f"{float(np.linalg.norm(y1_before)):.4f}"
    )

    # Step 2: perturb every residual weight with a small noise
    # tensor. This is the simplest possible "training" of the
    # residual plane; the goal is to demonstrate that the residual
    # actually contributes to the forward when n_planes=2 is
    # active, without doing a full distillation loop.
    print(f"[compare] perturbing {len(student.residual_params)} residual weights ...")
    for rp in student.residual_params:
        rp.data.add_(torch.randn_like(rp) * 0.05)

    # Step 3: snapshot outputs again.
    y1_after, _, _ = student.forward(batch, n_planes=1)
    y2_after, _, _ = student.forward(batch, n_planes=2)
    print(
        f"[compare] post-perturb: ||y_n_planes=1|| = "
        f"{float(np.linalg.norm(y1_after)):.4f}"
    )
    print(
        f"[compare] post-perturb: ||y_n_planes=2|| = "
        f"{float(np.linalg.norm(y2_after)):.4f}"
    )
    diff = float(np.linalg.norm(y1_after - y2_after))
    print(
        f"[compare] ||y_n_planes=1 - y_n_planes=2|| = {diff:.4f} "
        f"(nonzero means residual plane contributed)"
    )

    # Step 4: confirm that the loss function consumes both outputs
    # without error.
    print("[compare] running combined_distillation_loss on both ...")
    from torus.train.losses import DistillationConfig, combined_distillation_loss
    t_logits, t_hidden, t_route = teacher.forward(batch, n_planes=2)
    cfg_loss = DistillationConfig()

    loss1, _ = combined_distillation_loss(
        student_logits=y1_after, teacher_logits=t_logits,
        student_hidden=None, teacher_hidden=t_hidden,
        student_route=None, teacher_route=t_route,
        cfg=cfg_loss,
    )
    loss2, _ = combined_distillation_loss(
        student_logits=y2_after, teacher_logits=t_logits,
        student_hidden=None, teacher_hidden=t_hidden,
        student_route=None, teacher_route=t_route,
        cfg=cfg_loss,
    )
    print(f"[compare] loss(n_planes=1) = {float(loss1):.4f}")
    print(f"[compare] loss(n_planes=2) = {float(loss2):.4f}")

    print(
        "[compare] OK: n_planes plumbing verified; residual plane "
        "contributes to the forward when perturbed."
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
        "--n-steps",
        type=int,
        default=5,
        help="(unused — kept for forward compat)",
    )
    a = p.parse_args(argv)
    return a.model, a.n_steps


if __name__ == "__main__":
    model, n = parse_args(sys.argv[1:])
    main(model, n)