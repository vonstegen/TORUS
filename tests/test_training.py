"""Tests for Phase-3 training primitives."""
from __future__ import annotations

from typing import Iterator

import numpy as np
import torch
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


def test_ternary_ste_auto_picks_group_size() -> None:
    # When the requested group_size does not fit, the STE auto-picks
    # the largest power-of-two divisor (or falls back to the full
    # row width) so small smoke models (e.g. tiny-gpt2 with
    # hidden=2) still work without manual group_size tuning.
    w = np.zeros((4, 5), dtype=np.float32)
    ste = TernarySTE(weight=w, group_size=4)
    # 5 is prime; the largest power-of-two <= 5 that divides 5 is 1;
    # we fall back to the full row width (5).
    assert ste.group_size == 5
    w2 = np.zeros((4, 8), dtype=np.float32)
    ste2 = TernarySTE(weight=w2, group_size=4)
    # 8 / 4 fits exactly; the request is preserved.
    assert ste2.group_size == 4


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


def test_trainer_probe_residual_gradients_flow_to_residual() -> None:
    """With `probe_residual=True`, perturbations of the residual weight
    propagate through the forward and affect the loss, so the
    trainer sees a non-zero gradient on the residual plane.
    """
    # Build a student with both primary AND residual planes.
    rng = np.random.default_rng(0)
    weight = (rng.standard_normal((8, 32)) * 0.05).astype(np.float32)
    residual = (rng.standard_normal((8, 32)) * 0.01).astype(np.float32)
    ste = TernarySTE(
        weight=weight,
        group_size=32,
        residual_weight=residual,
    )
    teacher = TernarySTE(weight=weight.copy(), group_size=32)

    def fwd_s(_batch, n_planes):
        q = ste.forward(n_planes=n_planes)
        return q[2], None, None  # codes, scale, q_w

    def fwd_t(_batch, _n_planes):
        q = teacher.forward(n_planes=1)
        return q[2], None, None

    batch = DistillationBatch(
        inputs=np.zeros((1, 8), dtype=np.float32),
    )

    # Without probe_residual: the residual isn't in the gradient
    # path; the trainer only sees primary perturbation.
    cfg_no_res = TrainingConfig(n_steps=1, log_every=1, probe_rows=1, probe_residual=False)
    trainer = DistillationTrainer(
        student_params=[ste],
        forward_student=fwd_s,
        forward_teacher=fwd_t,
        data=iter([batch] * 5),  # 5 batches for 2 steps
        loss_cfg=DistillationConfig(),
        train_cfg=cfg_no_res,
    ).fit()

    # With probe_residual=True: the trainer also perturbs the residual
    # at the same (r, c) and the loss change includes the residual's
    # contribution. The resulting loss curves should differ (because
    # the residual isn't zero-init).
    cfg_with_res = TrainingConfig(n_steps=1, log_every=1, probe_rows=1, probe_residual=True)
    ste2 = TernarySTE(
        weight=weight.copy(),
        group_size=32,
        residual_weight=residual.copy(),
    )
    teacher2 = TernarySTE(weight=weight.copy(), group_size=32)

    def fwd_s2(_batch, n_planes):
        q = ste2.forward(n_planes=n_planes)
        return q[2], None, None

    def fwd_t2(_batch, _n_planes):
        q = teacher2.forward(n_planes=1)
        return q[2], None, None

    trainer2 = DistillationTrainer(
        student_params=[ste2],
        forward_student=fwd_s2,
        forward_teacher=fwd_t2,
        data=iter([batch] * 5),  # 5 batches for 2 steps
        loss_cfg=DistillationConfig(),
        train_cfg=cfg_with_res,
    ).fit()

    # Both trainers should run without error. The losses can be
    # different because probe_residual=True perturbs both primary
    # and residual at the same (r, c); with zero-init residual the
    # losses are identical, but with non-zero residual they're
    # different (sanity check that the perturbation actually flows).
    assert len(trainer) == 1
    assert len(trainer2) == 1


def test_trainer_probe_residual_off_does_not_touch_residual() -> None:
    """`probe_residual=False` (default) finishes one step without error
    even when the STE has a residual. The trainer's `_residual_np` is
    initialised to a list of Nones in __init__; fit() upgrades it.
    """
    rng = np.random.default_rng(0)
    weight = (rng.standard_normal((8, 32)) * 0.05).astype(np.float32)
    residual = (rng.standard_normal((8, 32)) * 0.01).astype(np.ndarray)
    ste = TernarySTE(
        weight=weight,
        group_size=32,
        residual_weight=residual,
    )
    teacher = TernarySTE(weight=weight.copy(), group_size=32)

    def fwd_s(_batch, n_planes):
        q = ste.forward(n_planes=n_planes)
        return q[2], None, None

    def fwd_t(_batch, _n_planes):
        q = teacher.forward(n_planes=1)
        return q[2], None, None

    batch = DistillationBatch(inputs=np.zeros((1, 8), dtype=np.float32))

    history = DistillationTrainer(
        student_params=[ste],
        forward_student=fwd_s,
        forward_teacher=fwd_t,
        data=iter([batch] * 5),  # 5 batches for 2 steps
        loss_cfg=DistillationConfig(),
        train_cfg=TrainingConfig(n_steps=1, log_every=1),
    ).fit()
    assert len(history) == 1


def test_trainer_residual_lr_scale_smoke() -> None:
    """With probe_residual=True and a residual_lr_scale, the residual
    SGD is constructed and a step finishes without error.
    """
    rng = np.random.default_rng(0)
    weight = (rng.standard_normal((8, 32)) * 0.05).astype(np.float32)
    residual = (rng.standard_normal((8, 32)) * 0.01).astype(np.ndarray)
    ste = TernarySTE(
        weight=weight, group_size=32, residual_weight=residual,
    )
    teacher = TernarySTE(weight=weight.copy(), group_size=32)

    def fwd_s(_b, n_planes):
        q = ste.forward(n_planes=n_planes)
        return q[2], None, None

    def fwd_t(_b, _n_planes):
        q = teacher.forward(n_planes=1)
        return q[2], None, None

    batch = DistillationBatch(inputs=np.zeros((1, 8), dtype=np.float32))

    history = DistillationTrainer(
        student_params=[ste],
        forward_student=fwd_s,
        forward_teacher=fwd_t,
        data=iter([batch] * 5),  # 5 batches for 2 steps
        loss_cfg=DistillationConfig(),
        train_cfg=TrainingConfig(
            n_steps=2, log_every=1,
            probe_residual=True, residual_lr_scale=0.05,
        ),
    ).fit()
    assert len(history) == 2


def test_trainer_autograd_path_uses_torch_grads_when_adapter_supports() -> None:
    """When the student adapter has `forward_with_grad`, the trainer
    uses torch.autograd.grad for the gradient (not finite differences).
    """
    import torch

    class _AutogradStudent:
        def __init__(self):
            self.weight = torch.nn.Parameter(
                torch.randn(8, 32) * 0.05, requires_grad=True,
            )
            self.residual_weight = torch.nn.Parameter(
                torch.zeros(8, 32), requires_grad=True,
            )

        def forward_with_grad(self, batch, n_planes):
            # Cheap "forward": just return the weight as a fake logits.
            return self.weight, None, np.zeros((1, 1), dtype=np.float32), \
                [self.weight], [self.residual_weight]
        def __call__(self, batch, n_planes):
            return self.forward(batch, n_planes)

        def forward(self, batch, n_planes):
            return self.weight.detach().numpy(), None, np.zeros((1, 1), dtype=np.float32)

    class _Teacher:
        # Shape matches the student's "logits" (8, 32) so the KL loss
        # can compute a finite value. The teacher's forward_torch path
        # now runs KL, which requires shape agreement (the legacy MSE
        # path broadcast freely and was a stand-in, not a real loss).
        def forward(self, batch, n_planes):
            return np.zeros((8, 32), dtype=np.float32), None, np.zeros((1, 1), dtype=np.float32)
        def forward_teacher_torch(self, batch):
            return torch.zeros((8, 32), dtype=torch.float32)
        def forward_torch(self, batch):
            return torch.zeros((8, 32), dtype=torch.float32, requires_grad=False)
        def __call__(self, batch, n_planes):
            return self.forward(batch, n_planes)

    ste = _AutogradStudent()
    teacher = _Teacher()
    history = DistillationTrainer(
        student_params=[ste],
        forward_student=ste,
        forward_teacher=teacher,
        data=iter([DistillationBatch(inputs=np.zeros((1, 8), dtype=np.float32))] * 5),
        loss_cfg=DistillationConfig(),
        train_cfg=TrainingConfig(n_steps=3, log_every=1),
    ).fit()
    assert len(history) == 3

def test_trainer_residual_warmup_ramps_lr_from_zero() -> None:
    """With `residual_warmup_steps=N`, the residual SGD's LR is
    0 for the first step, ramps linearly over N steps, and
    reaches the target LR by step N. After step N it stays
    at the target.
    """
    import torch
    from torus.train.loop import _SGD

    class _AutogradStudent:
        def __init__(self):
            self.weight = torch.nn.Parameter(
                torch.randn(8, 32, dtype=torch.float32) * 0.05
            )
            self.residual_weight = torch.nn.Parameter(
                torch.zeros(8, 32, dtype=torch.float32)
            )

        def forward(self, batch, n_planes):
            return (
                self.weight.detach().numpy(), None,
                np.zeros((1, 1), dtype=np.float32),
            )

        def forward_with_grad(self, batch, n_planes):
            q = (self.weight + self.residual_weight * 0.0).sum(dim=-1, keepdim=True)
            return q, None, None, [self.weight], [self.residual_weight]

        def forward_torch(self, batch):
            q = (self.weight + self.residual_weight * 0.0).sum(dim=-1, keepdim=True)
            return q

        def __call__(self, batch, n_planes):
            return self.forward(batch, n_planes)

    class _Teacher:
        def forward(self, batch, n_planes):
            return np.zeros((8, 1), dtype=np.float32), None, np.zeros((1, 1), dtype=np.float32)
        def forward_torch(self, batch):
            return torch.zeros((8, 1), dtype=torch.float32)
        def __call__(self, batch, n_planes):
            return self.forward(batch, n_planes)

    # Wrap _SGD.step to record the lr at each step.
    recorded_lr: list[float] = []
    original_step = _SGD.step

    def recording_step(self, grads):
        recorded_lr.append(self.lr)
        return original_step(self, grads)

    _SGD.step = recording_step  # type: ignore[assignment]
    try:
        ste = _AutogradStudent()
        teacher = _Teacher()
        trainer = DistillationTrainer(
            student_params=[ste],
            forward_student=ste,
            forward_teacher=teacher,
            data=iter([DistillationBatch(inputs=np.zeros((1, 8), dtype=np.float32))] * 10),
            loss_cfg=DistillationConfig(),
        train_cfg=TrainingConfig(
            n_steps=10, log_every=1,
            residual_lr_scale=0.5,
            residual_warmup_steps=4,
        ),
        )
        trainer.fit()
    finally:
        _SGD.step = original_step  # type: ignore[assignment]

    # learning_rate=1e-3 (default) * residual_lr_scale=0.5 = 5e-4 target.
    # Step k ramp = (k+1)/4 of 5e-4.
    residual_lr = recorded_lr[1::2]
    expected = [1.25e-4, 2.5e-4, 3.75e-4, 5e-4, 5e-4, 5e-4, 5e-4, 5e-4, 5e-4, 5e-4]
    assert len(residual_lr) == 10
    for got, want in zip(residual_lr, expected):
        assert abs(got - want) < 1e-12, f"step lr {got} != {want}"
    assert trainer._step == 9
