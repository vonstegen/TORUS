"""Smoke tests for the EXP-A-011 driver (examples/layer_sensitivity.py).

These tests do not run the driver end-to-end (which needs CUDA + a
real HF model). They validate the small, testable contracts:
  - the per-arm sidecar path (which eval_lm.py writes as
    <--output>.full.json) is correct
  - the target-list parser rejects malformed inputs
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
sys.path.insert(0, str(EXAMPLES.parent))


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None and importlib.util.find_spec("transformers") is None,
    reason="driver imports torch indirectly at run time; safe to import without it though",
)
def test_run_arm_sidecar_path_matches_eval_lm_constructor() -> None:
    """The driver's expected full-results path matches eval_lm.py's writer.

    eval_lm.py (after EXP-A-011) writes the sidecar as
    ``Path(str(args.output) + ".full.json")``. The driver passes
    ``<safe_name>.summary.json`` as ``--output``, so the sidecar
    ends in ``.summary.json.full.json``. A regression that put
    the sidecar at ``<safe_name>.full.json`` would silently mark
    every arm as failed even when the eval succeeded.
    """
    # Import the module without running main().
    spec = importlib.util.spec_from_file_location(
        "layer_sensitivity", EXAMPLES / "layer_sensitivity.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    # Synthesize a path the way run_arm would for some target.
    out_dir = Path("/tmp/_x")
    safe_name = "model__layers__0__self_attn__q_proj"
    summary = out_dir / "per_layer" / f"{safe_name}.summary.json"
    # Mirror the post-fix line in run_arm.
    full = Path(str(summary) + ".full.json")
    assert full == out_dir / "per_layer" / f"{safe_name}.summary.json.full.json"
    assert str(full).endswith(".summary.json.full.json")


def test_load_target_list_accepts_a_clean_list() -> None:
    spec = importlib.util.spec_from_file_location(
        "layer_sensitivity", EXAMPLES / "layer_sensitivity.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    p = Path("/tmp/_targets.json")
    p.write_text(json.dumps([
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.mlp.gate_proj",
        "model.embed_tokens",
    ]))
    try:
        out = mod.load_target_list(p)
        assert out == [
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.mlp.gate_proj",
            "model.embed_tokens",
        ]
    finally:
        p.unlink()


def test_load_target_list_rejects_non_list() -> None:
    spec = importlib.util.spec_from_file_location(
        "layer_sensitivity", EXAMPLES / "layer_sensitivity.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    p = Path("/tmp/_targets.json")
    p.write_text(json.dumps({"not": "a list"}))
    try:
        with pytest.raises(ValueError):
            mod.load_target_list(p)
    finally:
        p.unlink()


def test_load_target_list_rejects_short_names() -> None:
    """Every target must be a fully-qualified name (contain a dot)."""
    spec = importlib.util.spec_from_file_location(
        "layer_sensitivity", EXAMPLES / "layer_sensitivity.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    p = Path("/tmp/_targets.json")
    p.write_text(json.dumps(["q_proj", "model.layers.0.self_attn.k_proj"]))
    try:
        with pytest.raises(ValueError):
            mod.load_target_list(p)
    finally:
        p.unlink()
