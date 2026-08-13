"""Tests for Phase-3 training primitives."""
from __future__ import annotations

from typing import Iterator

import numpy as np
import pytest

from torus.quant import ternary_quantize
from torus.train import (
    CurriculumSchedule,
    DistillationBatch,
    DistillationConfig,
    DistillationTrainer,
    TernarySTE,
    TrainingConfig,
    TrainingStats,
    combined_distillation_loss,
    expert_route_loss,
    intermediate_alignment_loss,
    kl_divergence,
    ternary_quantize_with_ste,
)
from torus.train.curriculum import CurriculumStage


# --- Distillation losses --------------------------------------------------


def test_kl_divergence_zero_when_identical() -> None:
    x = np.array([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]])
def test_kl_divergence_bounded_by_log_vocab() -> None:
    # At infinite temperature, KL converges toward log(vocab) times T^2.
    s = np.array([[1.0, 2.0, 3.0]])
    t = np.array([[3.0, 2.0, 1.0]])
    kl1 = float(kl_divergence(s, t, temperature=1.0)[0])
    kl2 = float(kl_divergence(s, t, temperature=4.0)[0])
    # Higher T weights the KL term more strongly (Hinton convention).
    assert kl2 > kl1
    # Still bounded by log(vocab) * T^2.
    import math
    upper = math.log(3) * (4 ** 2)
    assert kl2 <= upper + 1e-6




def test_kl_divergence_rejects_bad_temperature() -> None:
    with pytest.raises(ValueError):
        kl_divergence(np.zeros((1, 3)), np.zeros((1, 3)), temperature=0.0)


def test_kl_divergence_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        kl_divergence(np.zeros((1, 3)), np.zeros((1, 4)))


def test_intermediate_alignment_zero_when_identical() -> None:
    h = np.random.default_rng(0).standard_normal((4, 16)).astype(np.float32)
    err = intermediate_alignment_loss(h, h)
    np.testing.assert_allclose(err, np.zeros(4), atol=1e-6)


def test_intermediate_alignment_grows_with_distance() -> None:
    s = np.zeros((1, 4), dtype=np.float32)
    t1 = np.zeros((1, 4), dtype=np.float32)
    t2 = np.ones((1, 4), dtype=np.float32)
    a1 = intermediate_alignment_loss(s, t1)
    a2 = intermediate_alignment_loss(s, t2)
    assert float(a2[0]) > float(a1[0])


def test_expert_route_loss_zero_when_identical() -> None:
    w = np.array([[0.2, 0.3, 0.5]])
    assert expert_route_loss(w, w) == pytest.approx(0.0, abs=1e-6)


def test_expert_route_loss_grows_with_dissimilarity() -> None:
    s = np.array([[0.9, 0.05, 0.05]])
    t = np.array([[0.05, 0.05, 0.9]])
    assert expert_route_loss(s, t) > 0.0


def test_combined_loss_returns_components() -> None:
    rng = np.random.default_rng(2)
    s_logits = rng.standard_normal((2, 16))
    t_logits = rng.standard_normal((2, 16))
    s_hidden = rng.standard_normal((2, 8))
    t_hidden = rng.standard_normal((2, 8))
    s_route = np.array([[0.7, 0.2, 0.1], [0.1, 0.7, 0.2]])
    t_route = np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.3]])
    cfg = DistillationConfig()
    loss, comps = combined_distillation_loss(
        student_logits=s_logits, teacher_logits=t_logits,
        student_hidden=s_hidden, teacher_hidden=t_hidden,
        student_route=s_route, teacher_route=t_route,
        cfg=cfg,
    )
    assert "kl" in comps
    assert "intermediate" in comps
    assert "expert" in comps
    assert "total" in comps
    assert comps["total"] == pytest.approx(loss, rel=1e-6)
    assert loss > 0.0


def test_combined_loss_minimal_when_perfect() -> None:
    rng = np.random.default_rng(3)
    s = rng.standard_normal((2, 8))
    loss, _ = combined_distillation_loss(
        student_logits=s, teacher_logits=s,
        student_hidden=s, teacher_hidden=s,
        student_route=np.full((2, 3), 1 / 3),
        teacher_route=np.full((2, 3), 1 / 3),
    )
    assert loss == pytest.approx(0.0, abs=1e-6)


# --- Straight-through estimator -------------------------------------------


def test_ste_quantize_returns_three_arrays() -> None:
    w = np.random.default_rng(0).standard_normal((32, 256)).astype(np.float32) * 0.05
    codes, scale, qw = ternary_quantize_with_ste(w, group_size=128)
    assert codes.shape == w.shape
    assert codes.dtype == np.int8
    assert set(np.unique(codes).tolist()).issubset({-1, 0, 1})
    assert scale.shape == (32, 256 // 128)
    assert qw.shape == w.shape
    assert qw.dtype == np.float32


def test_ste_quantized_matches_ternary_quantize() -> None:
    rng = np.random.default_rng(1)
    w = (rng.standard_normal((16, 256)) * 0.05).astype(np.float32)
    codes, _, _ = ternary_quantize_with_ste(w, group_size=128)
    plane = ternary_quantize(w, group_size=128, threshold=0.7)
    np.testing.assert_array_equal(codes, plane.codes)


def test_ternary_ste_wrapper_does_not_modify_weight() -> None:
    rng = np.random.default_rng(2)
    w = (rng.standard_normal((8, 128)) * 0.05).astype(np.float32)
    before = w.copy()
    ste = TernarySTE(weight=w, group_size=128)
    ste.forward()
    assert np.array_equal(w, before)


def test_ternary_ste_requires_divisible_columns() -> None:
    w = np.zeros((4, 5), dtype=np.float32)
    with pytest.raises(ValueError):
        TernarySTE(weight=w, group_size=4)


# --- Curriculum -----------------------------------------------------------


def test_curriculum_progressive_default_thresholds() -> None:
    sched = CurriculumSchedule.progressive(steps_per_stage=[100, 200, 300])
    assert sched.max_planes() == 3
    assert sched.stage_at(50).n_planes_active == 1
    assert sched.stage_at(150).n_planes_active == 2
    assert sched.stage_at(999).n_planes_active == 3

    sched = CurriculumSchedule.progressive(
        steps_per_stage=[10, 20],
        thresholds=[0.5, 0.9],
    )
    assert sched.stage_at(0).threshold == 0.5
    # Step 15 falls in stage 2 (cumulative cutoff = 10 < 15 <= 30).
    assert sched.stage_at(15).threshold == 0.9
    assert sched.stage_at(25).threshold == 0.9



def test_curriculum_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        CurriculumSchedule.progressive(
            steps_per_stage=[1, 2], thresholds=[0.7],
        )


def test_curriculum_n_planes_active_at() -> None:
    sched = CurriculumSchedule(stages=[
        CurriculumStage(n_planes_active=1, min_steps=5),
        CurriculumStage(n_planes_active=2, min_steps=5),
    ])
    assert sched.n_planes_active_at(0) == 1
    assert sched.n_planes_active_at(4) == 1
    assert sched.n_planes_active_at(5) == 2


def test_curriculum_max_planes_empty() -> None:
    sched = CurriculumSchedule()
    assert sched.max_planes() == 0


# --- Trainer --------------------------------------------------------------


def _toy_forward(
    params: list,
    batch: DistillationBatch,
    n_planes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Identity student/teacher used by the trainer tests."""
    x = batch.inputs
    w = np.eye(x.shape[1], min(8, x.shape[1]), dtype=np.float32)
    logits = x @ w
    hidden = x[:, :min(8, x.shape[1])]
    route = np.full((x.shape[0], 3), 1 / 3, dtype=np.float32)
    return logits, hidden, route


def _make_student_params(seed: int, n_layers: int = 2) -> list[TernarySTE]:
    rng = np.random.default_rng(seed)
    return [
        TernarySTE(
            weight=(rng.standard_normal((16, 64)) * 0.05).astype(np.float32),
            group_size=64,
        )
        for _ in range(n_layers)
    ]


def _data_stream(n: int = 64) -> Iterator[DistillationBatch]:
    rng = np.random.default_rng(7)
    for _ in range(n):
        x = rng.standard_normal((2, 16)).astype(np.float32)
        yield DistillationBatch(inputs=x)


def test_trainer_smoke_run() -> None:
    student_params = _make_student_params(seed=0)
    data = list(_data_stream(8))
    cfg = TrainingConfig(n_steps=8, log_every=2)
    trainer = DistillationTrainer(
        student_params=student_params,
        forward_student=lambda b, n: _toy_forward(student_params, b, n),
        forward_teacher=lambda b, n: _toy_forward(student_params, b, n),
        data=data,
        train_cfg=cfg,
    )
    history = trainer.fit()
    assert len(history) >= 1
    for stats in history:
        assert stats.loss > -1e-4



def test_trainer_uses_curriculum_n_planes() -> None:
    student_params = _make_student_params(seed=5)
    cur = CurriculumSchedule.progressive(steps_per_stage=[2, 3])
    cfg = TrainingConfig(n_steps=6, log_every=1)
    seen = []

    def on_log(stats: TrainingStats) -> None:
        seen.append(stats.n_planes_active)

    trainer = DistillationTrainer(
        student_params=student_params,
        forward_student=lambda b, n: _toy_forward(student_params, b, n),
        forward_teacher=lambda b, n: _toy_forward(student_params, b, n),
        data=list(_data_stream(6)),
        curriculum=cur,
        train_cfg=cfg,
    )
    trainer.fit(on_log=on_log)
    assert seen[0] == 1
    assert seen[-1] == 2


def test_trainer_requires_nonempty_student() -> None:
    with pytest.raises(ValueError):
        DistillationTrainer(
            student_params=[],
            forward_student=_toy_forward,
            forward_teacher=_toy_forward,
            data=[],
        )


def test_trainer_handles_data_exhaustion() -> None:
    student_params = _make_student_params(seed=8)
    cfg = TrainingConfig(n_steps=20, log_every=10)
    trainer = DistillationTrainer(
        student_params=student_params,
        forward_student=lambda b, n: _toy_forward(student_params, b, n),
        forward_teacher=lambda b, n: _toy_forward(student_params, b, n),
        data=list(_data_stream(2)),
        train_cfg=cfg,
    )
    with pytest.raises(StopIteration):
        trainer.fit()


def test_trainer_global_grad_clip() -> None:
    grads = [np.full((2, 2), 100.0, dtype=np.float32)]
    clipped = DistillationTrainer._clip_global_norm(grads, clip=1.0)
    assert clipped[0].max() < 100.0


def test_trainer_clip_zero_passthrough() -> None:
    grads = [np.full((2, 2), 100.0, dtype=np.float32)]
    out = DistillationTrainer._clip_global_norm(grads, clip=0.0)
    np.testing.assert_array_equal(out[0], grads[0])
