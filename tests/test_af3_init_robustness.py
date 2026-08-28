"""Tests for EXP-AF-003 (AF3 initialization robustness).

Pins:
  1. the --t2-init-sigma knob: T2TernaryAdapter(init_sigma=s) produces
     a latent with the requested init scale; the default stays 0.01
     (the frozen AF2-D value); sigma=0 gives an all-zero latent;
  2. build_site_adapter threads the knob into the trained t2_ternary
     arm only (untrained structure controls keep the frozen default);
  3. the frozen AF3 classification rules (audit_af3_robustness):
     ROBUST / MODERATELY SENSITIVE / FRAGILE boundaries, the sigma=0
     structural-control exclusion, and the proxy/capability divergence
     flag.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

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
af3 = _load("audit_af3_robustness")


def test_adapter_default_sigma_unchanged() -> None:
    ad = af2.T2TernaryAdapter(
        in_features=512, out_features=128, device="cpu", init_seed=0
    )
    assert ad.init_sigma == 0.01
    std = ad.latent.detach().std().item()
    assert abs(std - 0.01) < 0.002


def test_adapter_sigma_knob() -> None:
    ad = af2.T2TernaryAdapter(
        in_features=512, out_features=128, device="cpu", init_seed=0,
        init_sigma=1e-3,
    )
    assert ad.init_sigma == 1e-3
    std = ad.latent.detach().std().item()
    assert abs(std - 1e-3) < 2e-4


def test_adapter_sigma_zero_is_all_zero() -> None:
    ad = af2.T2TernaryAdapter(
        in_features=64, out_features=16, device="cpu", init_seed=0,
        init_sigma=0.0,
    )
    assert float(ad.latent.detach().abs().max()) == 0.0


def test_build_site_adapter_threads_knob_to_trained_arm_only() -> None:
    lin = torch.nn.Linear(4, 4)
    trained = af2.build_site_adapter(
        "t2_ternary", target_module=lin, site_dims=(4, 4),
        t2_init_sigma=3e-3,
    )
    assert trained.init_sigma == 3e-3
    # Untrained structure control must stay at the frozen default.
    lin2 = torch.nn.Linear(4, 4)
    control = af2.build_site_adapter("random_t2_ternary", target_module=lin2,
                                     site_dims=(4, 4))
    assert control.init_sigma == 0.01


def _levels(success_sigmas, ppl_by_sigma=None, zero_ppl=430.0):
    """Synthetic per-level stats in audit_af3_robustness shape."""
    levels = {}
    for s in af3.NONZERO_LEVELS:
        ok = s in success_sigmas
        ppl = (ppl_by_sigma or {}).get(s, 20.0 if ok else 300.0)
        levels[s] = {
            "level_success": ok,
            "ppl_mean": ppl,
            "arc_easy_mean": 0.60 if ok else 0.49,
            "lambada_openai_mean": 0.55 if ok else 0.03,
        }
    levels[0.0] = {
        "level_success": zero_ppl <= af3.SUCCESS_PPL_BAR,
        "ppl_mean": zero_ppl,
        "arc_easy_mean": 0.49,
        "lambada_openai_mean": 0.03,
    }
    return levels


def test_classify_robust() -> None:
    res = af3.classify(_levels(set(af3.NONZERO_LEVELS),
                               ppl_by_sigma={s: 20.0
                                             for s in af3.NONZERO_LEVELS}))
    assert res["classification"] == "ROBUST"
    assert res["levels_succeeded"] == 5
    assert res["zero_control"]["level_success"] is False


def test_classify_moderate_on_partial_success() -> None:
    res = af3.classify(_levels({1e-3, 3e-3, 1e-2}))
    assert res["classification"] == "MODERATELY SENSITIVE"
    assert res["levels_succeeded"] == 3


def test_classify_fragile_on_narrow_window() -> None:
    res = af3.classify(_levels({1e-2}))
    assert res["classification"] == "FRAGILE"
    assert res["levels_succeeded"] == 1


def test_classify_fragile_on_wide_spread() -> None:
    res = af3.classify(_levels(
        set(af3.NONZERO_LEVELS),
        ppl_by_sigma={1e-4: 99.0, 3e-4: 90.0, 1e-3: 50.0,
                      3e-3: 20.0, 1e-2: 15.0},
    ))
    # 5/5 succeed but spread 99/15 = 6.6 > 5 -> FRAGILE.
    assert res["classification"] == "FRAGILE"
    assert res["spread_ratio"] > 5.0


def test_classify_moderate_on_mild_spread() -> None:
    res = af3.classify(_levels(
        set(af3.NONZERO_LEVELS),
        ppl_by_sigma={1e-4: 60.0, 3e-4: 50.0, 1e-3: 30.0,
                      3e-3: 20.0, 1e-2: 15.0},
    ))
    # 5/5 succeed, spread 60/15 = 4.0 -> MODERATELY SENSITIVE.
    assert res["classification"] == "MODERATELY SENSITIVE"


def test_zero_control_success_falsifies_dead_zone() -> None:
    res = af3.classify(_levels(set(af3.NONZERO_LEVELS), zero_ppl=42.0))
    assert "falsified" in res["zero_control"]["finding"]


def test_capability_crosscheck_flags_proxy_only_success() -> None:
    levels = _levels({1e-2})
    levels[1e-2]["arc_easy_mean"] = 0.4890  # at damaged base
    levels[1e-2]["lambada_openai_mean"] = 0.02
    res = af3.capability_crosscheck(levels)
    assert len(res["divergent_levels"]) == 1
    assert res["divergent_levels"][0]["sigma"] == 1e-2
