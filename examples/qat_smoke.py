"""Phase-3 QAT smoke example.

Drives a tiny ternary student through a handful of distillation
steps against an identity teacher. Prints loss curves and shows how
the curriculum grows `n_planes_active` from 1 to 2.

Run with:

    python examples/qat_smoke.py
"""
from __future__ import annotations

import numpy as np

from torus.quant import pack_plane, ternary_quantize
from torus.train import (
    CurriculumSchedule,
    DistillationBatch,
    DistillationConfig,
    DistillationTrainer,
    TernarySTE,
    TrainingConfig,
)
from torus.train.losses import combined_distillation_loss


def toy_forward(params: list[TernarySTE], batch: DistillationBatch,
                n_planes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tiny 'model' that maps inputs -> logits via the ternary-weighted sum.

    The `n_planes` argument is honored but not used by this toy. A real
    student would select which planes to activate based on the gate
    signal. This example just shows how the trainer drives the steps.
    """
    x = batch.inputs  # (batch, 16)
    # Stack the parameters into one effective weight by quantization.
    pieces = []
    hidden = np.zeros((x.shape[0], 8), dtype=np.float32)
    for p in params:
        _codes, _scale, qw = p.forward()
        pieces.append(qw)
        hidden += x @ qw[:x.shape[1], :8]
    logits = x @ pieces[0][:, :8]
    route = np.full((x.shape[0], 3), 1 / 3, dtype=np.float32)
    return logits, hidden, route


def _make_data(n: int) -> list[DistillationBatch]:
    rng = np.random.default_rng(0)
    return [
        DistillationBatch(inputs=rng.standard_normal((4, 16)).astype(np.float32))
        for _ in range(n)
    ]


def main() -> None:
    # Two learnable ternary layers, each (out=16, in=64).
    rng = np.random.default_rng(0)
    student_params = [
        TernarySTE(
            weight=(rng.standard_normal((16, 64)) * 0.05).astype(np.float32),
            group_size=64,
        ),
        TernarySTE(
            weight=(rng.standard_normal((16, 64)) * 0.05).astype(np.float32),
            group_size=64,
        ),
    ]

    # Curriculum: spend 4 steps with 1 plane, 6 steps with 2.
    curriculum = CurriculumSchedule.progressive(steps_per_stage=[4, 6])

    trainer = DistillationTrainer(
        student_params=student_params,
        forward_student=lambda batch, n: toy_forward(student_params, batch, n),
        forward_teacher=lambda batch, n: toy_forward(student_params, batch, n),
        data=_make_data(20),
        loss_cfg=DistillationConfig(
            temperature=2.0,
            alpha_kl=1.0,
            alpha_intermediate=0.5,
            alpha_expert=0.1,
        ),
        curriculum=curriculum,
        train_cfg=TrainingConfig(
            learning_rate=0.05,
            momentum=0.9,
            n_steps=10,
            log_every=1,
        ),
    )

    print("Phase-3 smoke run")
    print("=================")
    print(f"  n_student_params = {len(student_params)}")
    print(f"  curriculum       = {[s.min_steps for s in curriculum.stages]}")
    print(f"  total weight bytes (int8): "
          f"{sum(p.weight.nbytes for p in student_params)}")

    def _on_log(stats) -> None:
        print(
            f"  step={stats.step:>3}  loss={stats.loss:.4f}  "
            f"planes={stats.n_planes_active}  "
            f"kl={stats.components['kl']:.4f}  "
            f"inter={stats.components.get('intermediate', 0.0):.4f}  "
            f"expert={stats.components.get('expert', 0.0):.4f}"
        )

    history = trainer.fit(on_log=_on_log)
    final_loss = history[-1].loss
    initial_loss = history[0].loss
    print()
    print(f"  initial loss: {initial_loss:.4f}")
    print(f"  final loss:   {final_loss:.4f}")
    print(f"  delta:        {final_loss - initial_loss:+.4f}")

    # Show the actual stored weight format (packed vs full).
    print()
    print("Stored weight (packed vs full):")
    for i, p in enumerate(student_params):
        packed = pack_plane(ternary_quantize(p.weight, group_size=p.group_size))
        saved = packed.packed_codes.nbytes + packed.scales.nbytes
        print(
            f"  layer {i}: int8={p.weight.nbytes} B   "
            f"packed(codes+scales)={saved} B   "
            f"reduction={p.weight.nbytes / max(1, saved):.1f}x"
        )


if __name__ == "__main__":
    main()
