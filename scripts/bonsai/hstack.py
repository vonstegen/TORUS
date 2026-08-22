"""Hadamard machinery for the Bonsai rotated-CPT prototype.

Conventions (torch Linear: weight (out, in), y = x @ W.T):
  Insert orthogonal Q (in x in):  y = (x @ Q) @ (W @ Q).T
  Rotated latent:  W_rot = W @ Q          (init from Bonsai weights)
  Forward:         x_q = x @ Q;  w_q = ternary_ste(W_rot);  y = x_q @ w_q.T
  Fold-back export: W_eff = w_q @ Q.T  (plain fp16 Linear, stock-loadable)

Q dims: 2048 -> Sylvester H_2048 / sqrt(2048).
        6144 -> (H_12/sqrt(12)) kron (H_512/sqrt(512)) via two matmuls.
All orthogonality claims are verified numerically in self_test().
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sylvester(n: int) -> torch.Tensor:
    """Unnormalized Sylvester Hadamard matrix, n = 2^k, symmetric, H@H = nI."""
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], dim=1),
                       torch.cat([h, -h], dim=1)], dim=0)
    return h


def _paley12() -> torch.Tensor:
    """Order-12 Hadamard via Paley I (q=11). Orthogonal: H@H.T = 12 I."""
    q = 11

    def chi(a: int) -> int:
        a %= q
        if a == 0:
            return 0
        return 1 if pow(a, (q - 1) // 2, q) == 1 else -1

    jm = torch.tensor([[chi(j - i) for j in range(q)] for i in range(q)],
                      dtype=torch.float64)
    s = torch.zeros(q + 1, q + 1, dtype=torch.float64)
    s[0, 1:] = 1.0
    s[1:, 0] = -1.0
    s[1:, 1:] = jm
    return torch.eye(q + 1, dtype=torch.float64) + s


_H2048 = _sylvester(2048) / math.sqrt(2048)          # symmetric + orthogonal
_H512 = _sylvester(512) / math.sqrt(512)
_H12 = _paley12() / math.sqrt(12)                    # orthogonal (not symmetric)


def apply_q(x: torch.Tensor, transpose: bool = False) -> torch.Tensor:
    """Apply Q (or Q.T) along the last dim. Works for activations
    (..., in) and weights (out, in) alike. fp32 compute."""
    dtype = x.dtype
    x = x.float()
    n = x.shape[-1]
    if n == 2048:
        m = _H2048.T if transpose else _H2048
        y = x @ m.to(x.device, x.dtype)
    elif n == 6144:
        z = x.view(*x.shape[:-1], 12, 512)
        # (A kron B): first B on the 512-dim, then A on the 12-dim.
        # transpose reverses factor order with transposed factors.
        if transpose:
            z = torch.matmul(_H12.T.to(x.device, x.dtype), z)
            z = z @ _H512.T.to(x.device, x.dtype)
        else:
            z = z @ _H512.T.to(x.device, x.dtype)   # H512 symmetric
            z = torch.matmul(_H12.to(x.device, x.dtype), z)
        y = z.reshape(*x.shape)
    else:
        raise ValueError(f"no Hadamard for dim {n}")
    return y.to(dtype)


def ternary_ste(w: torch.Tensor, gs: int = 32) -> torch.Tensor:
    """Per-row per-group ternary, BitNet round-clip codes,
    straight-through gradient. w: (out, in).

    Scale = mean of |w| over NONZERO elements (not absmean over all):
    on already-ternary {0,+/-s} inputs this recovers s exactly
    (absmean would shrink it by the nonzero fraction); on dense
    latents it coincides with absmean."""
    r, c = w.shape
    wg = w.view(r, -1, gs)
    nz = (wg != 0).sum(-1, keepdim=True).clamp_min(1)
    s = (wg.abs().sum(-1, keepdim=True) / nz).clamp_min(1e-8)
    codes = (wg / s).round().clamp(-1, 1)
    wq = (codes * s).view(r, c)
    return w + (wq - w).detach()


class HLinear(nn.Module):
    """Linear with ternary STE on an fp32 latent, optionally in
    Hadamard-rotated space (rotate=True for outlier-prone inputs)."""

    def __init__(self, linear: nn.Linear, gs: int = 32, ste: bool = False,
                 rotate: bool = True):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.gs = gs
        self.ste = ste
        self.rotate = rotate
        with torch.no_grad():
            w = linear.weight.detach().float()
            lat = apply_q(w) if rotate else w.clone()
        self.latent = nn.Parameter(lat, requires_grad=True)

    def effective_weight(self) -> torch.Tensor:
        wq = ternary_ste(self.latent, self.gs) if self.ste else self.latent
        return wq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wq = self.effective_weight()
        xq = apply_q(x) if self.rotate else x
        return F.linear(xq, wq.to(xq.dtype))

    def folded_weight(self) -> torch.Tensor:
        """Plain-Linear equivalent: W_eff = w_q @ Q.T (or w_q)."""
        with torch.no_grad():
            wq = ternary_ste(self.latent.detach(), self.gs) if self.ste \
                else self.latent.detach()
            if self.rotate:
                return apply_q(wq.float(), transpose=True)
            return wq.float()


ROTATE_TARGETS = ("o_proj", "down_proj")
PLAIN_TARGETS = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")


def wrap_model(model: nn.Module, gs: int = 32, ste: bool = False,
               full: bool = False, rotate_all: bool = True) -> list:
    """Wrap linears with HLinear. Default: o_proj/down_proj rotated.
    full=True: also wrap q/k/v/gate/up with plain (unrotated) STE.
    rotate_all=False: o_proj/down_proj unrotated too (control arm)."""
    wraps = []
    for name, mod in list(model.named_modules()):
        leaf = name.rsplit(".", 1)[-1]
        if not isinstance(mod, nn.Linear):
            continue
        if leaf in ROTATE_TARGETS:
            rot = rotate_all
        elif leaf in PLAIN_TARGETS and full:
            rot = False
        else:
            continue
        parent = model.get_submodule(name.rsplit(".", 1)[0])
        hw = HLinear(mod, gs=gs, ste=ste, rotate=rot)
        setattr(parent, leaf, hw)
        wraps.append(hw)
    return wraps


def self_test(device: str = "cpu") -> None:
    """Numerically verify every structural claim."""
    h12 = _paley12()
    err = (h12 @ h12.T - 12 * torch.eye(12, dtype=torch.float64)).abs().max()
    assert err < 1e-9, f"H12 not Hadamard: {err}"

    for n in (2048, 6144):
        x = torch.randn(7, n, dtype=torch.float64)
        # orthogonality: apply transpose after forward => identity
        y = apply_q(apply_q(x.float()), transpose=True).double()
        assert (y - x).abs().max() < 1e-3, f"Q not orthogonal at {n}"

    # identity: y = (xQ)(WQ).T == xW.T
    w = torch.randn(256, 2048)
    x = torch.randn(5, 2048)
    y0 = x @ w.T
    y1 = (apply_q(x)) @ (apply_q(w)).T
    assert (y0 - y1).abs().max() < 5e-2, "rotation identity failed"

    # fold-back: W_eff = w_q @ Q.T reproduces the HLinear forward
    lin = nn.Linear(2048, 256, bias=False)
    with torch.no_grad():
        lin.weight.copy_(w)
    hw = HLinear(lin, ste=False)
    wf = hw.folded_weight()
    y2 = F.linear(x, wf)
    assert (y0 - y2).abs().max() < 5e-2, "fold-back identity failed"

    # ternary_ste recovers exact {0,+-s} gs=32 weights (Bonsai case)
    codes = (torch.randint(-1, 2, (64, 2048))).float()
    scales = torch.rand(64, 64).float() * 0.05
    wb = (codes.view(64, 64, 32) * scales.unsqueeze(-1)).view(64, 2048)
    wq = ternary_ste(wb, gs=32)
    assert (wq - wb).abs().max() < 1e-6, "requant drift (fp32)"
    assert (wq.half() - wb.half()).abs().max() == 0.0, "requant not exact at fp16"

    print("hstack self-test: all pass")
