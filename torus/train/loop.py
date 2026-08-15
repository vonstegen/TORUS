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
    probe_rows: int = 1            # finite-difference probes per module per step
    probe_cols: int = 0            # columns probed per row (0 = same as probe_rows)
    probe_residual: bool = False   # also perturb STE.residual_weight when set
    residual_lr_scale: float = 0.1  # residual plane LR = learning_rate * this
    residual_warmup_steps: int = 0  # ramp residual LR from 0 -> target over this many steps (0 = no warmup)

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
        # Parallel buffer for residual planes; filled by fit()
        # with numpy views of each STE.residual_weight (or None).
        # Initialized to a list of Nones so the post-step sync
        # block can iterate without AttributeError when the
        # trainer is used outside fit() (tests).
        self._residual_np: list = [None] * len(student_params)
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
        # `_params_np` is the in-place numpy view the trainer
        # mutates during numerical-gradient estimation.
        import torch as _torch
        self._params_np = [
            p.weight.detach().cpu().numpy()
            if hasattr(p.weight, "detach")
            else np.asarray(p.weight)
            for p in self.student_params
        ]
        params = self._params_np
        opt = _SGD(
            params=params,
            lr=cfg.learning_rate,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
        # Parallel numpy view of each STE's residual_weight (or None).
        # Populated here so the residual SGD, the post-step sync
        # block, and the per-element probe all see the same buffer.
        # Without this, `_residual_np` stays [None, None, ...] from
        # __init__ and the residual plane is permanently inert.
        self._residual_np = [
            p.residual_weight.detach().cpu().numpy()
                if hasattr(p.residual_weight, "detach")
            else (
                np.asarray(p.residual_weight)
                if p.residual_weight is not None
                else None
            )
            for p in self.student_params
        ]
        # Second SGD for the residual planes.
        # Built whenever any STE carries a residual_weight. The
        # `probe_residual` flag only changes the *numerical* probe
        # behavior (perturbing the residual alongside the primary at
        # the same (r, c) for finite-difference gradient estimation).
        # The autograd path always produces a separate residual
        # gradient via `torch.autograd.grad`, so the residual SGD must
        # exist whenever the residual plane is reachable. Building it
        # only when `probe_residual=True` left the autograd path
        # silently discarding the residual gradient.
        residual_params = [r for r in self._residual_np if r is not None]
        residual_opt = None
        if residual_params:
            target_residual_lr = cfg.learning_rate * getattr(
                cfg, "residual_lr_scale", 0.1
            )
            # Start at 0 if warmup is enabled; the loop ramps it
            # up to `target_residual_lr` linearly over the first
            # `cfg.residual_warmup_steps` steps. Warmup lets the
            # primary plane converge first before the residual
            # starts moving, so the residual doesn't amplify its
            # init noise while the primary is still settling.
            warmup_steps = getattr(cfg, "residual_warmup_steps", 0)
            initial_residual_lr = (
                0.0 if warmup_steps > 0 else target_residual_lr
            )
            residual_opt = _SGD(
                params=residual_params,
                lr=initial_residual_lr,
                momentum=cfg.momentum,
                weight_decay=cfg.weight_decay,
            )
            residual_opt._target_lr = target_residual_lr  # type: ignore[attr-defined]
            residual_opt._warmup_steps = warmup_steps  # type: ignore[attr-defined]
        history: list[TrainingStats] = []
        t0 = time.perf_counter()

        for step in range(cfg.n_steps):
            self._step = step
            n_planes = self._n_planes(step)
            batch = self._next_batch()

            # Detect once per step whether the autograd path is
            # available. When it is, the trainer skips the
            # per-step teacher+student forward+loss and lets
            # `_autograd_grads` handle everything in one go.
            use_autograd = self._can_use_autograd()

            if not use_autograd:
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

            # ---- Backward (autograd when available, else numerical) ---
            if use_autograd:
                primary_grads, residual_grads_list = self._autograd_grads(batch)
                grads = [g if g is not None else np.zeros_like(self._params_np[i])
                          for i, g in enumerate(primary_grads)]
                residual_grads_step = residual_grads_list
                # Compute a stand-in loss for logging. We re-run the
                # forward under no_grad to get detached logits.
                with __import__("torch").no_grad():
                    s_logits_l = self.forward_student(batch, n_planes)[0]
                    t_logits_l = self.forward_teacher(batch, n_planes)[0]
                loss = float(((s_logits_l - t_logits_l) ** 2).mean())
                components = {"kl": loss, "intermediate": 0.0,
                                 "expert": 0.0, "total": loss}
            else:
                grads = self._numerical_grads(batch, components)
                residual_grads_step = grads

            # ---- Update -----------------------------------------------
            grads = self._clip_global_norm(grads, cfg.grad_clip)
            opt.step(grads)
            if residual_opt is not None:
                # Build a list parallel to `self._residual_np`: for each
                # STE slot, use the residual gradient if available,
                # otherwise a zero array of the residual's shape. The
                # previous code used `self._residual_np.index(r)` which
                # does scalar `__eq__` and breaks when multiple slots
                # share shapes (it always returned the first match,
                # causing shape-broadcast errors when more than one
                # STE had a residual).
                residual_grads_step = [
                    (g if g is not None else np.zeros_like(rnp))
                    for g, rnp in zip(residual_grads_step, self._residual_np)
                    if rnp is not None
                ]
                residual_grads_step = self._clip_global_norm(
                    residual_grads_step,
                    cfg.grad_clip,
                )
                # Ramp residual LR linearly during warmup.
                if step < residual_opt._warmup_steps:
                    ramp = (step + 1) / residual_opt._warmup_steps
                    residual_opt.lr = residual_opt._target_lr * ramp
                else:
                    residual_opt.lr = residual_opt._target_lr
                residual_opt.step(residual_grads_step)
            # Copy the numpy buffers back to the torch STE weights
            # so the adapter sees the updated parameters on the
            # next forward pass.
            for src, ste, rnp in zip(self._params_np, self.student_params, self._residual_np):
                if hasattr(ste.weight, "copy_"):
                    # torch Parameter: bypass the autograd graph and
                    # avoid RuntimeError on a leaf Variable that
                    # requires grad.
                    target = ste.weight.data if hasattr(ste.weight, "data") else ste.weight
                    target.copy_(_torch.as_tensor(src).to(target.device))
                # Only torch tensors have a writable `.data` and
                # `.copy_()`. numpy arrays expose `.data` as a
                # memoryview in NumPy >=1.20, which doesn't satisfy
                # `.copy_`. For STEs whose residual_weight is a
                # plain numpy array the in-place SGD write to
                # `_residual_np` already mutated the source, so no
                # copy back is needed.
                if (
                    rnp is not None
                    and hasattr(ste, "residual_weight")
                    and hasattr(ste.residual_weight, "copy_")
                ):
                    ste.residual_weight.data.copy_(
                        _torch.as_tensor(rnp).to(ste.residual_weight.device)
                    )

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
        # Operate on the numpy buffers (torch Parameters may require
        # grad, which forbids np.zeros_like; the adapter copies these
        # numpy buffers to model weights before each forward pass).
        if not hasattr(self, "_params_np"):
            return [np.zeros_like(p.weight, dtype=np.float32)
                    if hasattr(p.weight, "numpy")
                    else np.zeros(p.weight.shape, dtype=np.float32)
                    for p in self.student_params]
        # Per-module probe budget: how many rows to probe via the
        # finite-difference method. The default is "one per row"
        # (the textbook choice) but that's prohibitive on real
        # models. Set to a small N to get a coarse but real gradient
        # direction. The trainer exposes  so callers
        # can override.
        budget = getattr(self.train_cfg, "probe_rows", 1)
        rng_grad = np.random.default_rng(0)
        probe_residual = getattr(self.train_cfg, "probe_residual", False)
        # Per-module probe budget: how many (row, col) entries to probe
        # via the finite-difference method. `probe_rows` controls rows,
        # `probe_cols` controls columns per row; default both = 1 (one
        # total probe per module per step). On real models this is
        # still sparse, but at least the gradient has a real direction
        # instead of being pinned to column 0.
        budget_rows = getattr(self.train_cfg, "probe_rows", 1)
        budget_cols = (
            getattr(self.train_cfg, "probe_cols", 0)
            or budget_rows
        )
        rng_grad = np.random.default_rng(0)
        probe_residual = getattr(self.train_cfg, "probe_residual", False)
        # Loss surface is measured at the *current* curriculum n_planes
        # so the gradient matches the forward the optimizer is stepping
        # on (see _loss_only docstring).
        n_planes = max(1, self._n_planes(self._step)) if hasattr(self, "_step") else 1
        for i, p in enumerate(self.student_params):
            w = self._params_np[i]
            rw = self._residual_np[i] if probe_residual else None
            n_rows, n_cols = w.shape
            grad = np.zeros_like(w)
            rows = (
                rng_grad.choice(n_rows, size=min(budget_rows, n_rows), replace=False)
                if budget_rows < n_rows
                else np.arange(n_rows)
            )
            cols = (
                rng_grad.choice(n_cols, size=min(budget_cols, n_cols), replace=False)
                if budget_cols < n_cols
                else np.arange(n_cols)
            )
            for r in rows:
                for c in cols:
                    # Snapshot original values before perturbing.
                    original_w = w[r, c]
                    original_r = rw[r, c] if rw is not None else None
                    # Probe the primary plane.
                    w[r, c] = original_w + eps
                    if rw is not None:
                        rw[r, c] = original_r + eps
                    plus, _ = self._loss_only(batch, n_planes)
                    w[r, c] = original_w - eps
                    if rw is not None:
                        rw[r, c] = original_r - eps
                    minus, _ = self._loss_only(batch, n_planes)
                    w[r, c] = original_w
                    if rw is not None:
                        rw[r, c] = original_r
                    grad[r, c] = (plus - minus) / (2 * eps)
            grads.append(grad)
        return grads

    def _can_use_autograd(self) -> bool:
        """True if the adapter exposes a `forward_with_grad`
        method (i.e. the STE weights are torch tensors). The
        autograd path is exact and ~10x faster than the
        numerical gradient path.
        """
        return hasattr(self.forward_student, "forward_with_grad")

    def _autograd_grads(self, batch) -> tuple:
        """Compute per-STE primary + residual gradients via torch.autograd.grad.

        Uses a real KL(student || teacher) against a frozen teacher
        that is run under `enable_grad` so gradients flow through the
        teacher's forward to the student logits. Teacher weights
        remain frozen (requires_grad=False).

        Returns:
            (primary_grads, residual_grads) where each is a list
            of numpy arrays (one per STE) or None where the STE
            didn't carry that plane.
        """
        import torch
        from torus.train.losses import kl_divergence_torch

        def _to_np(t):
            if t is None:
                return None
            return t.detach().cpu().numpy()

        # Adapter gives us torch tensors and the weight lists.
        n_planes = max(1, self._n_planes_safe(batch))
        s_logits, _s_hidden, _route, primary_weights, residual_weights = (
            self.forward_student.forward_with_grad(batch, n_planes)
        )
        # Real teacher: a frozen full-precision HF model. Run under
        # `enable_grad` so the KL loss builds a graph through the
        # teacher forward to the student logits.
        t_logits = self._teacher_logits_torch(batch)
        loss = kl_divergence_torch(
            s_logits, t_logits,
            temperature=self.loss_cfg.temperature,
        )

        # Flatten the weight list to a single grad call.
        tensors_to_grad = [
            t for t in primary_weights + residual_weights if t is not None
        ]
        grads = torch.autograd.grad(
            loss,
            tensors_to_grad,
            retain_graph=False,
            allow_unused=True,
        )

        primary_grads: list = []
        residual_grads: list = []
        g_iter = iter(grads)
        for pw, rw in zip(primary_weights, residual_weights):
            if pw is not None:
                primary_grads.append(_to_np(next(g_iter, None)))
            else:
                primary_grads.append(None)
            if rw is not None:
                residual_grads.append(_to_np(next(g_iter, None)))
            else:
                residual_grads.append(None)

        return primary_grads, residual_grads

    def _teacher_logits_torch(self, batch):
        """Teacher logits as a torch tensor under enable_grad.

        `forward_teacher` may be either an adapter instance or a
        bound method. We probe for `forward_torch` on both the value
        itself and its `__self__` so existing call sites that pass
        `teacher.forward` keep working.

        Resolution order:
        1. `forward_torch` on the teacher (or its bound `__self__`).
           Used by `HFTeacherAdapter` for real KL distillation.
        2. `forward_teacher_torch` (legacy student self-distillation
           path — wraps under `enable_grad` so the KL still builds a
           graph; falls back to old MSE behavior if it returns a
           detached tensor).
        3. No torch method: compute logits under `no_grad` and wrap
           as a torch tensor. No graph → no real distillation, but
           doesn't crash.
        """
        import torch
        teacher = getattr(self.forward_teacher, "__self__", self.forward_teacher)
        if hasattr(teacher, "forward_torch"):
            return teacher.forward_torch(batch)
        if hasattr(self.forward_teacher, "forward_teacher_torch"):
            with torch.enable_grad():
                return self.forward_teacher.forward_teacher_torch(batch)
        with torch.no_grad():
            out = self.forward_teacher(batch, n_planes=1)
            return torch.as_tensor(out[0])

    def _n_planes_safe(self, batch) -> int:
        """Pick `n_planes` for the autograd forward without breaking
        when the trainer is mid-iteration. We just take the
        current step's curriculum value.
        """
        try:
            return self._n_planes(self._step) if hasattr(self, "_step") else 1
        except Exception:
            return 1

    def _loss_only(self, batch: DistillationBatch, n_planes: int = 1) -> tuple[float, dict[str, float]]:
        # Refuse to use cached state - we need fresh quantization on perturbed weights.
        for p in self.student_params:
            p._cached_codes = None  # type: ignore[attr-defined]
            p._cached_quantized = None  # type: ignore[attr-defined]
        # Caller passes the *current* curriculum n_planes so the probe
        # measures the loss surface the optimizer is actually stepping
        # on. Earlier this derived n_planes from `_cached_codes is not
        # None`, which evaluated as a boolean and always collapsed to 1.
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
