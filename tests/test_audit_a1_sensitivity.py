"""Smoke tests for the EXP-A-011 auditor (examples/audit_a1_sensitivity.py)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
sys.path.insert(0, str(EXAMPLES.parent))


def _load_audit_module():
    spec = importlib.util.spec_from_file_location(
        "audit_a1_sensitivity", EXAMPLES / "audit_a1_sensitivity.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def test_parse_target_attention() -> None:
    mod = _load_audit_module()
    assert mod.parse_target("model.layers.0.self_attn.q_proj") == ("attention", 0, "q_proj")
    assert mod.parse_target("model.layers.15.self_attn.o_proj") == ("attention", 15, "o_proj")


def test_parse_target_mlp() -> None:
    mod = _load_audit_module()
    assert mod.parse_target("model.layers.7.mlp.gate_proj") == ("mlp", 7, "gate_proj")
    assert mod.parse_target("model.layers.3.mlp.down_proj") == ("mlp", 3, "down_proj")


def test_parse_target_embed_head() -> None:
    mod = _load_audit_module()
    assert mod.parse_target("model.embed_tokens") == ("embed", None, "embed_tokens")
    assert mod.parse_target("lm_head") == ("head", None, "lm_head")


def test_parse_target_references() -> None:
    mod = _load_audit_module()
    assert mod.parse_target("f16_reference") == ("reference_f16", None, None)
    assert mod.parse_target("fully_quantized") == ("reference_full", None, None)


def test_parse_target_unknown() -> None:
    mod = _load_audit_module()
    assert mod.parse_target("garbage") == ("unknown", None, None)
    assert mod.parse_target("model.layers.0.foo") == ("unknown", None, None)


def test_parse_per_layer_summary_extracts_arm_from_target_modules(tmp_path) -> None:
    """The per-arm summary written by eval_lm.py has `target_modules`
    (a list), not an `arm` field. The auditor must read the first
    target_module as the arm name."""
    mod = _load_audit_module()
    p = tmp_path / "model__layers__0__self_attn__q_proj.summary.json"
    p.write_text(json.dumps({
        "model": "allenai/OLMo-1B-0724-hf",
        "mode": "quantized",
        "target_modules": ["model.layers.0.self_attn.q_proj"],
        "no_calibrate": True,
        "limit": 200,
        "tasks": {
            "wikitext": {"metric": "word_perplexity,none", "value": 268.02},
            "arc_easy": {"metric": "acc,none", "value": 0.5471},
        },
    }))
    rec = mod.parse_per_layer_summary(p)
    assert rec["arm"] == "model.layers.0.self_attn.q_proj"
    assert rec["wikitext_ppl"] == 268.02
    assert rec["arc_easy_acc"] == 0.5471
