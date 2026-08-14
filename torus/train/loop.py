"""End-to-end QAT + distillation training loop (numpy reference).

This trainer is *not* a production framework; it's a runnable reference
that demonstrates how the Phase-1 primitives and the Phase-3 losses fit
together. The optimizer is plain SGD on the latent weights. Future
phases replace the inner loop with a torch / jax implementation
without changing the trainer's public surface area.

The trainer assumes:

- A `student` is a stack of `TernarySTE` parameters plus a callable
  `forward(params, batch, n_planes) -> (logits, hidden, route)` that
  produces the model outputs the loss consumes. The Phase-3 trainer
  works with any architecture that exposes such a callable.
- A `teacher` is a callable that maps the same inputs to the same
  shapes. For development, a stub teacher (e.g. the full-precision
  reference) is sufficient.
- A `data` source yields `DistillationBatch`es indefinitely.

The trainer supports checkpointing, eval callbacks, and a curriculum
that grows `n_planes` over time.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator

import numpy as np

from torus.train.curriculum import CurriculumSchedule
from torus.train.losses import DistillationConfig, combined_distillation_loss
from torus.train.ste import TernarySTE, ternary_quantize_with_ste


@dataclass
class DistillationBatch:
    """A single training batch.

    Routes through the trainer unchanged; only the inputs tensor is
    strictly required. Hidden / route targets are optional and only
    used when the teacher provides them.
    """
    inputs: np.ndarray            # (batch, ...)
    targets: np.ndarray | None = None   # (batch, vocab)


@dataclass
class TrainingConfig:
    """Top-level training configuration."""
    learning_rate: float = 1e-3
    momentum: float = 0.9
    weight_decay: float = 0.0
    n_steps: int = 100
    log_every: int = 10
    eval_every: int = 0           # 0 == skip eval
    grad_clip: float = 1.0        # global-norm clip


@dataclass
class TrainingStats:
    """Per-step stats buffer (consumer reads from a deque)."""
    step: int = 0
    loss: float = 0.0
    components: dict[str, float] = field(default_factory=dict)
    n_planes_active: int = 1
    elapsed_seconds: float = 0.0


# Lightweight SGD with optional momentum and weight decay.
class _SGD:
    def __init__(self, params: list[np.ndarray], lr: float, momentum: float,
                 weight_decay: float) -> None:
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self._vel: list[np.ndarray] = [np.zeros_like(p) for p in params]

    def step(self, grads: list[np.ndarray]) -> None:
        for i, (p, g) in enumerate(zip(self.params, grads)):
            if self.weight_decay:
                g = g + self.weight_decay * p
            self._vel[i] = self.momentum * self._vel[i] + g
            p -= self.lr * self._vel[i]


class DistillationTrainer:
    """Run quantization-aware distillation across a curriculum.

    Args:
        student_params: list of `TernarySTE` representing trainable
            ternary layers.
        forward_student: `forward(batch, n_planes) -> (logits, hidden, route)`.
            The STE list is read from `student_params` at call time.
        forward_teacher: same signature, on the (full-precision) teacher.
        data: iterable yielding `DistillationBatch` (one per step).
        loss_cfg: distillation config (temperatures, weights).
        curriculum: optional curriculum schedule.
        train_cfg: training config (lr, steps, etc.).
    """

    def __init__(
        self,
        student_params: list[TernarySTE],
        forward_student: Callable[..., tuple[np.ndarray, np.ndarray | None, np.ndarray | None]],
        forward_teacher: Callable[..., tuple[np.ndarray, np.ndarray | None, np.ndarray | None]],
        data: Iterable[DistillationBatch] | Iterator[DistillationBatch],
        loss_cfg: DistillationConfig | None = None,
        curriculum: CurriculumSchedule | None = None,
        train_cfg: TrainingConfig | None = None,
    ) -> None:
        if not student_params:
            raise ValueError("student_params must be a non-empty list of TernarySTE")
        self.student_params = student_params
        self.forward_student = forward_student
        self.forward_teacher = forward_teacher
        self._data_iter = iter(data) if not hasattr(data, "__iter__") else iter(data)
        self.loss_cfg = loss_cfg or DistillationConfig()
        self.curriculum = curriculum
        self.train_cfg = train_cfg or TrainingConfig()

    def _next_batch(self) -> DistillationBatch:
        try:
            return next(self._data_iter)
        except StopIteration:
            raise StopIteration("data exhausted")

    def _n_planes(self, step: int) -> int:
        if self.curriculum is None:
            return 1
        return self.curriculum.n_planes_active_at(step)

    def fit(self, on_log: Callable[[TrainingStats], None] | None = None,
            ) -> list[TrainingStats]:
        """Run the training loop for `train_cfg.n_steps`.

        `on_log` is called every `train_cfg.log_every` steps with the
        cumulative TrainingStats.

        Returns the full list of stats (one per logged step).
        """
        cfg = self.train_cfg
        # Latent parameters = the underlying float weights.
        params = [p.weight for p in self.student_params]
        opt = _SGD(
            params=params,
            lr=cfg.learning_rate,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
        history: list[TrainingStats] = []
        t0 = time.perf_counter()

        for step in range(cfg.n_steps):
            n_planes = self._n_planes(step)
            batch = self._next_batch()

            # ---- Teacher forward (no grad) ----------------------------
            t_logits, t_hidden, t_route = self.forward_teacher(batch, n_planes)

            # ---- Student forward (with quantized weights) --------------
            for p in self.student_params:
                p._cached_codes = None  # type: ignore[attr-defined]
                p._cached_quantized = None  # type: ignore[attr-defined]
                _codes, _scale, qw = p.forward()
                p._cached_quantized = qw  # type: ignore[attr-defined]
                p._cached_codes = _codes  # type: ignore[attr-defined]

            s_logits, s_hidden, s_route = self.forward_student(batch, n_planes)

            # ---- Loss --------------------------------------------------
            loss, components = combined_distillation_loss(
                student_logits=s_logits,
                teacher_logits=t_logits,
                student_hidden=s_hidden,
                teacher_hidden=t_hidden,
                student_route=s_route,
                teacher_route=t_route,
                cfg=self.loss_cfg,
            )

            # ---- Backward (numerical-gradient reference) ----------------
            grads = self._numerical_grads(batch, components)

            # ---- Update -----------------------------------------------
            grads = self._clip_global_norm(grads, cfg.grad_clip)
            opt.step(grads)

            if step % cfg.log_every == 0 or step == cfg.n_steps - 1:
                stats = TrainingStats(
                    step=step,
                    loss=loss,
                    components=components,
                    n_planes_active=n_planes,
                    elapsed_seconds=time.perf_counter() - t0,
                )
                history.append(stats)
                if on_log is not None:
                    on_log(stats)

        return history

    # ------------------------------------------------------------------
    # Internal: numerical-gradient reference.
    # This is intentionally *not* autograd. It computes the loss twice
    # around each parameter to estimate dL/dw, with a small epsilon.
    # The Point: demonstrate the trainer end-to-end without bringing
    # in torch. Real training swaps this for autograd.
    # ------------------------------------------------------------------
    def _numerical_grads(
        self,
        batch: DistillationBatch,
        _components: dict[str, float],
    ) -> list[np.ndarray]:
        eps = 1e-3
        grads: list[np.ndarray] = []
        baseline_loss, _ = self._loss_only(batch)
        for p in self.student_params:
            w = p.weight
            # Sample one element per row of w for tractability on big params;
            # a production trainer uses autograd, so this is illustrative.
            n = w.shape[0]
            grad = np.zeros_like(w)
            for r in range(n):
                # Use the first column as a single-point probe per row.
                c = 0
                original = w[r, c]
                w[r, c] = original + eps
                plus, _ = self._loss_only(batch)
                w[r, c] = original - eps
                minus, _ = self._loss_only(batch)
                w[r, c] = original
                grad[r, c] = (plus - minus) / (2 * eps)
            grads.append(grad)
        return grads

    def _loss_only(self, batch: DistillationBatch) -> tuple[float, dict[str, float]]:
        n_planes = max((p._cached_codes is not None for p in self.student_params), default=False)
        # Refuse to use cached state - we need fresh quantization on perturbed weights.
        for p in self.student_params:
            p._cached_codes = None  # type: ignore[attr-defined]
            p._cached_quantized = None  # type: ignore[attr-defined]
        s_logits, s_hidden, s_route = self.forward_student(batch, max(1, n_planes))
        t_logits, t_hidden, t_route = self.forward_teacher(batch, max(1, n_planes))
        return combined_distillation_loss(
            student_logits=s_logits,
            teacher_logits=t_logits,
            student_hidden=s_hidden,
            teacher_hidden=t_hidden,
            student_route=s_route,
            teacher_route=t_route,
            cfg=self.loss_cfg,
        )

    @staticmethod
    def _clip_global_norm(grads: list[np.ndarray], clip: float) -> list[np.ndarray]:
        if clip <= 0:
            return grads
        total = float(sum(np.square(g.astype(np.float64)).sum() for g in grads))
        norm = float(np.sqrt(total))
        if norm <= clip:
            return grads
        factor = clip / norm
        return [g * factor for g in grads]
