"""Distillation losses for capability-preserving residual-plane training.

The standard BitNet-style loss matches student logits to teacher logits.
That recovers the *final* outputs but loses the structural details that
made the teacher strong. TORUS adds two more terms:

1. `intermediate_alignment_loss`: the student reproduces the teacher's
   intermediate hidden states, *after* the residual plane has captured
   what the primary plane missed. This is what trains residual planes
   to "fix" the primary plane's most damaging errors.

2. `expert_route_loss`: a MoE-aware term that matches the routing
   distribution of the student to the teacher. Without this, ternary
   specialists collapse into uniform routing and the MoE structure is
   wasted.

All losses are pure numpy (no torch). They are sized so that for a
forward pass of any `(batch, vocab|hidden|experts,)` shape, the student
weight quantizer and the loss agree on the same material.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _softmax(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=axis, keepdims=True) + eps)


def _log_softmax(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    m = x.max(axis=axis, keepdims=True)
    s = np.log(np.exp(x - m).sum(axis=axis, keepdims=True) + eps)
    return x - m - s


def kl_divergence(student_logits: np.ndarray, teacher_logits: np.ndarray,
                  temperature: float = 1.0, axis: int = -1) -> np.ndarray:
    """Element-wise KL(student || teacher) at temperature `T`.

    Returns an array of shape `student_logits.shape[:-1]` (the
    per-token KL contribution). The trainer reduces it.

    Args:
        student_logits: (batch, vocab).
        teacher_logits: same shape.
        temperature: distillation temperature; > 1 softens the target.
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"shape mismatch: student {student_logits.shape}, teacher {teacher_logits.shape}"
        )
    T = float(temperature)
    if T <= 0:
        raise ValueError(f"temperature must be > 0, got {T}")
    log_p_s = _log_softmax(student_logits / T, axis=axis)
    p_t = _softmax(teacher_logits / T, axis=axis)
    # KL = sum_t p_t * (log p_t - log p_s); we drop log p_t (const w.r.t. student).
    return (p_t * (np.log(p_t + 1e-12) - log_p_s)).sum(axis=axis) * (T * T)


def intermediate_alignment_loss(
    student_hidden: np.ndarray,
    teacher_hidden: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """L2 distance between matched hidden states, optionally RMS-normalized.

    Returns a `(batch,)` array of per-token alignment errors.

    Args:
        student_hidden: (batch, hidden).
        teacher_hidden: (batch, hidden).
        normalize: if True, divide by the per-token teacher RMS to make
            the loss scale-invariant (preferable for residual-plane
            training where the absolute magnitudes can drift).
    """
    if student_hidden.shape != teacher_hidden.shape:
        raise ValueError(
            f"shape mismatch: student {student_hidden.shape}, teacher {teacher_hidden.shape}"
        )
    diff = student_hidden - teacher_hidden
    if normalize:
        rms = np.sqrt(np.mean(teacher_hidden ** 2, axis=-1, keepdims=True)) + 1e-6
        diff = diff / rms
    err = np.sqrt(np.mean(diff ** 2, axis=-1))
    return err


def expert_route_loss(
    student_weights: np.ndarray,
    teacher_weights: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """Symmetric KL between student and teacher routing distributions.

    Args:
        student_weights: (batch, num_experts); row-normalized.
        teacher_weights: same shape.

    Returns:
        Scalar mean symmetric-KL.
    """
    if student_weights.shape != teacher_weights.shape:
        raise ValueError(
            f"shape mismatch: student {student_weights.shape}, teacher {teacher_weights.shape}"
        )
    s = student_weights + eps
    t = teacher_weights + eps
    # Symmetric KL: KL(s || t) + KL(t || s)
    kl_st = (s * (np.log(s) - np.log(t))).sum(axis=-1)
    kl_ts = (t * (np.log(t) - np.log(s))).sum(axis=-1)
    return float(np.mean(kl_st + kl_ts))


def kl_divergence_torch(
    student_logits: "torch.Tensor",
    teacher_logits: "torch.Tensor",
    temperature: float = 1.0,
    axis: int = -1,
) -> "torch.Tensor":
    """KL(student || teacher) at temperature T, in torch.

    Mirror of `kl_divergence` for the autograd path. Returns a
    scalar mean KL that flows gradients back through `student_logits`.
    The constant `log p_t` term is dropped because it has no gradient
    w.r.t. the student; only `log_p_s` participates in the autograd graph.
    """
    import torch as _torch  # local import: module stays torch-free at import
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"shape mismatch: student {student_logits.shape}, teacher {teacher_logits.shape}"
        )
    T = float(temperature)
    if T <= 0:
        raise ValueError(f"temperature must be > 0, got {T}")
    s = student_logits / T
    t = teacher_logits / T
    log_p_s = _torch.log_softmax(s, dim=axis)
    p_t = _torch.softmax(t, dim=axis)
    kl = (p_t * (_torch.log(p_t + 1e-12) - log_p_s)).sum(dim=axis) * (T * T)
    return kl.mean()

@dataclass(frozen=True)
class DistillationConfig:
    """Hyperparameters for the TORUS distillation loss."""
    temperature: float = 2.0       # distillation temperature
    alpha_kl: float = 1.0          # weight on logit KL term
    alpha_intermediate: float = 0.5  # weight on hidden-state alignment
    alpha_expert: float = 0.2      # weight on MoE routing alignment
    intermediate_normalize: bool = True


def combined_distillation_loss(
    student_logits: np.ndarray,
    teacher_logits: np.ndarray,
    student_hidden: np.ndarray | None = None,
    teacher_hidden: np.ndarray | None = None,
    student_route: np.ndarray | None = None,
    teacher_route: np.ndarray | None = None,
    cfg: DistillationConfig = DistillationConfig(),
) -> tuple[float, dict[str, float]]:
    """Combine all distillation terms and report each component.

    Args:
        student_logits / teacher_logits: (batch, vocab).
        student_hidden / teacher_hidden: optional (batch, hidden).
        student_route / teacher_route: optional (batch, num_experts).
        cfg: hyperparameter bundle.

    Returns:
        (total_loss_scalar, components_dict) where `components_dict`
        lists each term's contribution for logging.
    """
    components: dict[str, float] = {}
    kl = kl_divergence(
        student_logits, teacher_logits,
        temperature=cfg.temperature,
    )
    kl_loss = float(np.mean(kl))
    components["kl"] = kl_loss
    total = cfg.alpha_kl * kl_loss

    if student_hidden is not None and teacher_hidden is not None:
        ia = intermediate_alignment_loss(
            student_hidden, teacher_hidden,
            normalize=cfg.intermediate_normalize,
        )
        ia_loss = float(np.mean(ia))
        components["intermediate"] = ia_loss
        total += cfg.alpha_intermediate * ia_loss

    if student_route is not None and teacher_route is not None:
        er = expert_route_loss(student_route, teacher_route)
        components["expert"] = er
        total += cfg.alpha_expert * er

    components["total"] = total
    return total, components
