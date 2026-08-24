"""Tests for Stage 2 v2 driver extension (freeze exception).

Per `research/residual-pareto/experiments/STAGE2-V2-FREEZE-EXCEPTION-SPEC.md`:

- resolve_site_dims returns (in_features, out_features) for any nn.Linear.
- damage_target_module_gaussian is deterministic given (sigma, seed) and
  produces different noise for different seeds.
- The damage freezes the weight (requires_grad_(False)) and does NOT
  touch other modules' weights.
- --damage-ptq and --damage-gaussian are mutually exclusive (main()).
"""

from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest
import torch
import torch.nn as nn


# ---- resolve_site_dims tests ----

def test_resolve_site_dims_down_proj_layout():
    """resolve_site_dims on a (out, in) layout returns (in, out)."""
    from examples.af2_storage_tournament import resolve_site_dims
    # nn.Linear(in, out) -> weight shape (out, in). MLP down_proj analog:
    # in_features = intermediate_size (gate/up output), out = hidden_size.
    mod = nn.Linear(8192, 2048, bias=False)
    in_f, out_f = resolve_site_dims(mod)
    assert in_f == 8192
    assert out_f == 2048
def test_resolve_site_dims_attention_proj_layout():
    """resolve_site_dims on a square (hidden, hidden) Linear returns (hidden, hidden)."""
    from examples.af2_storage_tournament import resolve_site_dims
    mod = nn.Linear(2048, 2048, bias=False)  # q_proj / v_proj analog
    in_f, out_f = resolve_site_dims(mod)
    assert in_f == 2048
    assert out_f == 2048


def test_resolve_site_dims_rejects_non_2d_weight():
    """resolve_site_dims on a 1-D weight raises."""
    from examples.af2_storage_tournament import resolve_site_dims
    mod = nn.Linear(4, 4, bias=False)
    # Force a 1-D shape by replacing the weight
    mod.weight = nn.Parameter(torch.randn(4))
    with pytest.raises(ValueError, match="must be 2-D"):
        resolve_site_dims(mod)


# ---- damage_target_module_gaussian tests ----

def test_damage_gaussian_seeded_reproducible():
    """Same (sigma, seed) -> bit-identical damaged weight."""
    from examples.af2_storage_tournament import damage_target_module_gaussian
    torch.manual_seed(42)
    m1 = nn.Linear(64, 32, bias=False)
    m2 = nn.Linear(64, 32, bias=False)
    # Make the two models have the same starting weight
    with torch.no_grad():
        m2.weight.copy_(m1.weight)
    damage_target_module_gaussian(m1, sigma=0.5, seed=123)
    damage_target_module_gaussian(m2, sigma=0.5, seed=123)
    assert torch.equal(m1.weight, m2.weight)


def test_damage_gaussian_different_seed_differs():
    """Different seeds -> different damaged weight."""
    from examples.af2_storage_tournament import damage_target_module_gaussian
    torch.manual_seed(42)
    m1 = nn.Linear(64, 32, bias=False)
    m2 = nn.Linear(64, 32, bias=False)
    with torch.no_grad():
        m2.weight.copy_(m1.weight)
    damage_target_module_gaussian(m1, sigma=0.5, seed=123)
    damage_target_module_gaussian(m2, sigma=0.5, seed=124)
    assert not torch.equal(m1.weight, m2.weight)


def test_damage_gaussian_zero_sigma_is_noop():
    """sigma=0 leaves the weight unchanged."""
    from examples.af2_storage_tournament import damage_target_module_gaussian
    torch.manual_seed(42)
    m = nn.Linear(64, 32, bias=False)
    w_before = m.weight.detach().clone()
    damage_target_module_gaussian(m, sigma=0.0, seed=123)
    assert torch.equal(m.weight, w_before)


def test_damage_gaussian_does_not_touch_other_modules():
    """Gaussian damage on m1 leaves m2.weight unchanged."""
    from examples.af2_storage_tournament import damage_target_module_gaussian
    torch.manual_seed(42)
    m1 = nn.Linear(64, 32, bias=False)
    m2 = nn.Linear(64, 32, bias=False)
    w2_before = m2.weight.detach().clone()
    damage_target_module_gaussian(m1, sigma=0.5, seed=123)
    assert torch.equal(m2.weight, w2_before), (
        "damage_target_module_gaussian modified m2.weight")


def test_damage_gaussian_freezes_weight():
    """After damage, target.weight.requires_grad is False."""
    from examples.af2_storage_tournament import damage_target_module_gaussian
    torch.manual_seed(42)
    m = nn.Linear(64, 32, bias=False)
    assert m.weight.requires_grad  # default
    damage_target_module_gaussian(m, sigma=0.5, seed=123)
    assert m.weight.requires_grad is False


def test_damage_gaussian_records_metadata():
    """Returned dict has sigma, seed, fro_norm_before, fro_norm_after, fro_ratio."""
    from examples.af2_storage_tournament import damage_target_module_gaussian
    torch.manual_seed(42)
    m = nn.Linear(64, 32, bias=False)
    meta = damage_target_module_gaussian(m, sigma=0.5, seed=123)
    assert meta["sigma"] == 0.5
    assert meta["seed"] == 123
    assert "fro_norm_before" in meta
    assert "fro_norm_after" in meta
    assert "fro_ratio" in meta
    assert isinstance(meta["fro_norm_before"], float)
    assert isinstance(meta["fro_norm_after"], float)
    assert meta["fro_ratio"] > 0  # nonzero; sigma=0.5 produces ~ +sigma^2 / (out*in) ratio bump


# ---- mutual-exclusivity test ----

def test_damage_modes_mutually_exclusive():
    """Setting both --damage-ptq and --damage-gaussian errors out at parse time."""
    from examples import af2_storage_tournament as af2

    # Replicate the exact parse_args + the mutual-exclusivity gate.
    argv = [
        "--model", "allenai/OLMo-1B-hf",
        "--target-module", "model.layers.0.mlp.down_proj",
        "--arms", "t2_ternary",
        "--out-dir", "/tmp/scratch_dmg_mutual_excl",
        "--damage-ptq",
        "--damage-gaussian",
    ]
    with pytest.raises(SystemExit):
        af2.main(argv=argv)


# ---- build_site_adapter equivalence test (down_proj -> attention proj) ----

def test_build_site_adapter_works_for_attention_proj():
    """build_site_adapter accepts site_dims=(hidden, hidden) without error."""
    from examples.af2_storage_tournament import build_site_adapter
    torch.manual_seed(42)
    mod = nn.Linear(2048, 2048, bias=False)  # q_proj analog
    # t2_ternary arm should construct successfully.
    ad = build_site_adapter("t2_ternary", target_module=mod,
                            site_dims=(2048, 2048))
    # T2TernaryAdapter stores the residual as `latent` with shape
    # (out_features, in_features); verify dims match site_dims.
    assert ad.latent.shape == (2048, 2048)