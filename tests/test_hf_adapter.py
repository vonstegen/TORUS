"""Smoke tests for the HF adapter contract.

These tests do NOT import torch or transformers — they verify the
adapter's interface contract using a `MockHFAdapter` that produces
the same `(logits, hidden, route)` tuple shape as the real adapter.

The real adapter is only exercised on a host that has `torch` and
`transformers` installed; this is documented as a Phase-3 milestone.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from torus.train.ste import TernarySTE
from torus.train.loop import DistillationBatch


class MockHFAdapter:
    """Reference adapter that satisfies the trainer's forward contract.

    Mirrors the shape of `HFStudentAdapter.forward` /
    `HFTeacherAdapter.forward` so we can validate trainer plumbing
    without downloading a model.
    """

    def __init__(
        self,
        vocab_size: int = 100,
        hidden_size: int = 16,
        n_ste_params: int = 4,
        rng_seed: int = 0,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        rng = np.random.default_rng(rng_seed)
        self.ste_params = [
            TernarySTE(
                weight=rng.standard_normal((8, 8)).astype(np.float32),
                group_size=8,
            )
            for _ in range(n_ste_params)
        ]

    def forward(self, batch, n_planes):
        b = batch.inputs.shape[0]
        s = batch.inputs.shape[-1] if batch.inputs.ndim > 1 else 1
        logits = np.zeros((b, s, self.vocab_size), dtype=np.float32)
        hidden = np.zeros((b, s, self.hidden_size), dtype=np.float32)
        route = np.zeros((b, 1), dtype=np.float32)
        return logits, hidden, route


def test_mock_adapter_matches_trainer_contract() -> None:
    adapter = MockHFAdapter()
    assert len(adapter.ste_params) == 4
    for ste in adapter.ste_params:
        assert isinstance(ste, TernarySTE)
        assert ste.weight.shape == (8, 8)


def test_mock_adapter_forward_shape() -> None:
    adapter = MockHFAdapter(vocab_size=128, hidden_size=32)
    batch = DistillationBatch(
        inputs=np.zeros((2, 7), dtype=np.int64),
        targets=None,
    )
    logits, hidden, route = adapter.forward(batch, n_planes=1)
    assert logits.shape == (2, 7, 128)
    assert hidden.shape == (2, 7, 32)
    assert route.shape == (2, 1)


def test_trainer_accepts_mock_adapter() -> None:
    """The Phase-3 trainer runs against the mock adapter end-to-end."""
    from torus.train.curriculum import CurriculumSchedule
    from torus.train.loop import DistillationTrainer, TrainingConfig
    from torus.train.losses import DistillationConfig

    student = MockHFAdapter(vocab_size=64, hidden_size=8, n_ste_params=2, rng_seed=1)
    teacher = MockHFAdapter(vocab_size=64, hidden_size=8, n_ste_params=2, rng_seed=2)

    def data_iter():
        rng = np.random.default_rng(3)
        while True:
            yield DistillationBatch(
                inputs=rng.integers(0, 64, size=(2, 5)).astype(np.int64),
            )

    history = DistillationTrainer(
        student_params=student.ste_params,
        forward_student=student.forward,
        forward_teacher=teacher.forward,
        data=data_iter(),
        loss_cfg=DistillationConfig(),
        curriculum=CurriculumSchedule.progressive(steps_per_stage=[4, 6]),
        train_cfg=TrainingConfig(n_steps=6, log_every=2, learning_rate=1e-3),
    ).fit()
    assert len(history) >= 3  # at least steps 2, 4, 6
    assert history[0].n_planes_active == 1
    assert history[-1].n_planes_active == 2


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch not installed; real adapter integration is a Phase-3 milestone",
)
def test_real_hf_adapter_imports_when_torch_present() -> None:
    """If torch + transformers are present, the adapter module loads."""
    import torus.train.hf_adapter as hf  # noqa: F401
    assert hasattr(hf, "HFStudentAdapter")
    assert hasattr(hf, "HFTeacherAdapter")
    assert hasattr(hf, "HFAdapterConfig")