"""Tests for the torch-native straight-through STE path.

Regression context (EXP-A-03x run 20260822T220559Z, INVALID): the
patched forward in `hf_adapter._make_forward_stub` quantized via the
numpy path, which silently broke the autograd graph. The trainer's
autograd path received None gradients (hidden by allow_unused=True),
zero-filled them, and 500 "training" steps moved no weights. These
tests pin the behavioral contract that would have caught it:
  1. the torch forward matches the numpy forward's values;
  2. gradients actually reach the STE latent;
  3. DistillationTrainer.fit() moves the latent end-to-end.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torus.train.ste import (
    TernarySTE,
    ternary_quantize_ste_torch,
    ternary_quantize_with_ste,
)


def test_torch_ste_matches_numpy_forward() -> None:
    rng = np.random.default_rng(0)
    w_np = rng.standard_normal((4, 8)).astype(np.float32)
    _, _, q_np = ternary_quantize_with_ste(
        w_np, group_size=4, threshold=0.7, calibrate_norm=False
    )
    w_t = torch.nn.Parameter(torch.from_numpy(w_np.copy()))
    q_t = ternary_quantize_ste_torch(
        w_t, group_size=4, threshold=0.7, calibrate_norm=False
    )
    np.testing.assert_allclose(
        q_t.detach().numpy(), q_np, rtol=1e-5, atol=1e-6
    )


def test_torch_ste_matches_numpy_forward_calibrated() -> None:
    rng = np.random.default_rng(1)
    w_np = rng.standard_normal((4, 8)).astype(np.float32)
    _, _, q_np = ternary_quantize_with_ste(
        w_np, group_size=4, threshold=0.7, calibrate_norm=True
    )
    w_t = torch.nn.Parameter(torch.from_numpy(w_np.copy()))
    q_t = ternary_quantize_ste_torch(
        w_t, group_size=4, threshold=0.7, calibrate_norm=True
    )
    np.testing.assert_allclose(
        q_t.detach().numpy(), q_np, rtol=1e-4, atol=1e-5
    )


def test_torch_ste_gradient_reaches_latent() -> None:
    w_t = torch.nn.Parameter(torch.randn(4, 8))
    ste = TernarySTE(weight=w_t, group_size=4)
    x = torch.randn(3, 8)
    q = ste.forward_torch(n_planes=1)
    loss = (torch.nn.functional.linear(x, q) ** 2).mean()
    loss.backward()
    assert w_t.grad is not None, "no gradient reached the STE latent"
    assert float(w_t.grad.abs().sum()) > 0.0, "STE latent gradient is zero"


def test_torch_ste_residual_plane_gradient() -> None:
    w_t = torch.nn.Parameter(torch.randn(4, 8))
    r_t = torch.nn.Parameter(torch.randn(4, 8) * 0.01)
    ste = TernarySTE(weight=w_t, group_size=4, residual_weight=r_t)
    x = torch.randn(3, 8)
    q = ste.forward_torch(n_planes=2)
    loss = (torch.nn.functional.linear(x, q) ** 2).mean()
    loss.backward()
    assert w_t.grad is not None and float(w_t.grad.abs().sum()) > 0.0
    assert r_t.grad is not None and float(r_t.grad.abs().sum()) > 0.0


class _TorchToyAdapter:
    """Minimal autograd-capable adapter mirroring HFStudentAdapter's
    contract: forward (numpy) + forward_with_grad (torch graph).

    Two STEs with DIFFERENT shapes so the autograd grad flatten /
    re-split order is exercised: a positional mix-up between
    primary and residual slots (or between modules) crashes here
    instead of silently scrambling gradients.
    """

    def __init__(self, d_in: int = 8, vocab: int = 16, seed: int = 0) -> None:
        g = torch.Generator().manual_seed(seed)
        self.ste1 = TernarySTE(
            weight=torch.nn.Parameter(torch.randn(d_in, d_in, generator=g)),
            group_size=d_in,
        )
        self.ste2 = TernarySTE(
            weight=torch.nn.Parameter(torch.randn(vocab, d_in, generator=g)),
            group_size=d_in,
        )
        self.vocab = vocab
        self.ste_params = [self.ste1, self.ste2]
        self._n_planes = 1

    def _logits(self, batch, n_planes, *, grad: bool):
        import torch.nn.functional as F

        x = torch.as_tensor(
            np.asarray(batch.inputs, dtype=np.float32), dtype=torch.float32
        )
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            q1 = self.ste1.forward_torch(n_planes=n_planes)
            q2 = self.ste2.forward_torch(n_planes=n_planes)
            h = F.linear(x, q1)
            return F.linear(h, q2)

    def forward(self, batch, n_planes):
        logits = self._logits(batch, n_planes, grad=False)
        b = batch.inputs.shape[0]
        return (
            logits.detach().cpu().numpy(),
            None,
            np.zeros((b, 1), dtype=np.float32),
        )

    def forward_with_grad(self, batch, n_planes):
        logits = self._logits(batch, n_planes, grad=True)
        b = batch.inputs.shape[0]
        return (
            logits,
            None,
            np.zeros((b, 1), dtype=np.float32),
            [self.ste1.weight, self.ste2.weight],
            [self.ste1.residual_weight, self.ste2.residual_weight],
        )


class _TorchToyTeacher:
    """Frozen teacher exposing forward (numpy) + forward_torch."""

    def __init__(self, d_in: int = 8, vocab: int = 16, seed: int = 7) -> None:
        g = torch.Generator().manual_seed(seed)
        self.w = torch.randn(vocab, d_in, generator=g)
        self.vocab = vocab

    def _logits(self, batch):
        x = torch.as_tensor(
            np.asarray(batch.inputs, dtype=np.float32), dtype=torch.float32
        )
        return torch.nn.functional.linear(x, self.w)

    def forward(self, batch, n_planes):
        with torch.no_grad():
            logits = self._logits(batch)
        b = batch.inputs.shape[0]
        return (
            logits.detach().cpu().numpy(),
            None,
            np.zeros((b, 1), dtype=np.float32),
        )

    def forward_torch(self, batch):
        with torch.enable_grad():
            return self._logits(batch)


def test_trainer_autograd_path_moves_latent() -> None:
    """Regression for EXP-A-03x INVALID run: after fit(), the STE
    latent must differ from its init. Before the torch-STE fix, the
    autograd path produced None gradients and the latent never moved.
    """
    from torus.train.curriculum import CurriculumSchedule
    from torus.train.loop import (
        DistillationBatch,
        DistillationTrainer,
        TrainingConfig,
    )
    from torus.train.losses import DistillationConfig

    student = _TorchToyAdapter(d_in=8, vocab=16, seed=0)
    teacher = _TorchToyTeacher(d_in=8, vocab=16, seed=7)
    init1 = student.ste1.weight.detach().clone()
    init2 = student.ste2.weight.detach().clone()

    def data_iter():
        rng = np.random.default_rng(3)
        while True:
            yield DistillationBatch(
                inputs=rng.standard_normal((4, 8)).astype(np.float32)
            )

    DistillationTrainer(
        student_params=student.ste_params,
        forward_student=student.forward,
        forward_teacher=teacher.forward,
        data=data_iter(),
        loss_cfg=DistillationConfig(),
        curriculum=CurriculumSchedule.progressive(steps_per_stage=[4]),
        train_cfg=TrainingConfig(n_steps=4, log_every=1),
    ).fit()

    moved1 = float((student.ste1.weight.detach() - init1).abs().sum())
    moved2 = float((student.ste2.weight.detach() - init2).abs().sum())
    assert moved1 > 0.0 and moved2 > 0.0, (
        f"STE latents did not move during fit() "
        f"(ste1 moved {moved1:.6f}, ste2 moved {moved2:.6f}); the "
        f"autograd path is not producing real gradients, or the grad "
        f"flatten/re-split assigned them to the wrong slots "
        f"(EXP-A-03x run-1 regression)"
    )
