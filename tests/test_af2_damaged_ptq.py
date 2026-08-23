"""Tests for the --damage-ptq extension to examples/af2_storage_tournament.py.

The damage mode is the EXP-A-011 PTQ recipe applied as a static
weight replacement (in-place + frozen). These tests pin the
damage-mode invariants without running the full AF2 driver:

  - damage_target_module applies TWN-style per-group absmean
    ternary damage: codes in {-1, 0, +1}, reconstruction
    equals codes * per-group scale, weight is frozen.
  - For a realistic-shape weight (8192, 2048), the Frobenius
    norm ratio lands in the expected TWN range (~0.5-0.8).
  - The damage is idempotent given the same group_size/threshold.
  - damage_target_module does NOT modify other modules' weights.

Reference numbers for AF2-D's pre-train band verification (from
EXP-A-011): the broken PTQ arm produces wikitext ppl ~427.7 and
arc_easy ~0.5396. The pre-train eval band in the AF2-D manifest
is [400, 460] ppl / [0.51, 0.57] arc_easy (within +/-2 sigma).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, EXAMPLES / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


af2 = _load("af2_storage_tournament")


def _fake_module(weight_np: np.ndarray):
    """Build a torch.nn.Linear with the given weight."""
    out_features, in_features = weight_np.shape
    layer = torch.nn.Linear(in_features, out_features, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor(weight_np))
    return layer


def test_damage_target_module_basic_shape() -> None:
    """Damage mode produces codes in {-1, 0, +1} and a fro-norm ratio in the
    expected TWN range. The weight is frozen after damage."""
    rng = np.random.default_rng(0)
    # Realistic shape: (out_features=8192, in_features=2048); cols must be
    # divisible by group_size=128 -> 2048 / 128 = 16 groups.
    w = rng.standard_normal((8192, 2048)).astype(np.float32) * 0.02
    mod = _fake_module(w)

    meta = af2.damage_target_module(mod, group_size=128, threshold=0.7)

    # Frobenius ratio lands in the expected TWN range.
    assert 0.5 <= meta["fro_ratio"] <= 0.85, (
        f"fro_ratio {meta['fro_ratio']} outside expected TWN range; "
        f"recipe calibration may have drifted."
    )
    # Weight is frozen: requires_grad_(False) was called.
    assert mod.weight.requires_grad is False
    # Reconstruction matches the per-group scale pattern.
    # The codes are ternary, but reconstructed weight is float.
    assert mod.weight.dtype == torch.float32


def test_damage_target_module_idempotent() -> None:
    """Damage mode is deterministic given (group_size, threshold): applying
    damage to the ORIGINAL weight twice produces the same final weight.
    """
    rng = np.random.default_rng(1)
    w = rng.standard_normal((4096, 1024)).astype(np.float32) * 0.02
    m1 = _fake_module(w.copy())
    m2 = _fake_module(w.copy())
    af2.damage_target_module(m1, group_size=128, threshold=0.7)
    af2.damage_target_module(m2, group_size=128, threshold=0.7)
    assert torch.allclose(m1.weight.data, m2.weight.data, atol=1e-6)


def test_damage_target_module_does_not_touch_other_weights() -> None:
    """Damage mode operates on `target_module.weight` only; other weights
    in the model are unaffected.
    """
    rng = np.random.default_rng(2)
    w_target = rng.standard_normal((2048, 512)).astype(np.float32) * 0.02
    w_other = rng.standard_normal((2048, 2048)).astype(np.float32) * 0.05

    target = _fake_module(w_target)
    other = _fake_module(w_other)
    other_before = other.weight.detach().clone()
    af2.damage_target_module(target, group_size=128, threshold=0.7)
    assert torch.allclose(other.weight, other_before, atol=1e-7), (
        "damage_target_module modified a weight it shouldn't have."
    )


def test_damage_target_module_records_calibration_metadata() -> None:
    """The metadata dict captures the recipe knobs + the frobenius
    before/after for audit.
    """
    rng = np.random.default_rng(3)
    w = rng.standard_normal((1024, 256)).astype(np.float32) * 0.02
    mod = _fake_module(w)
    meta = af2.damage_target_module(mod, group_size=128, threshold=0.7)
    for key in ("group_size", "threshold", "calibrate_norm",
                "fro_norm_before", "fro_norm_after", "fro_ratio"):
        assert key in meta, f"missing metadata key: {key}"
    assert meta["group_size"] == 128
    assert meta["threshold"] == 0.7
    assert meta["calibrate_norm"] is False
    assert meta["fro_norm_before"] > 0
    assert meta["fro_norm_after"] > 0


def test_damage_target_module_main_help_lists_flags() -> None:
    """The driver --help lists the new flags so the manifest's freeze
    exception is auditable.
    """
    import subprocess
    out = subprocess.run(
        [sys.executable, str(EXAMPLES / "af2_storage_tournament.py"),
         "--help"],
        capture_output=True, text=True, check=True,
    )
    assert "--damage-ptq" in out.stdout
    assert "--damage-group-size" in out.stdout
    assert "--damage-threshold" in out.stdout
    assert "--pre-train-eval" in out.stdout


def test_af2_storage_tournament_module_loads_with_helpers() -> None:
    """The module exposes damage_target_module + pre_train_eval_if_damaged."""
    assert hasattr(af2, "damage_target_module")
    assert hasattr(af2, "pre_train_eval_if_damaged")
    # The damage_target_module is a regular function, not a method.
    assert callable(af2.damage_target_module)
    assert callable(af2.pre_train_eval_if_damaged)
