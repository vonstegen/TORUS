"""Tests for examples/ah1_native_hadamard.py + examples/audit_ah1.py.

Pins the EXP-A-H1 primitives and the frozen threshold evaluation:
  1. Sylvester Hadamard construction (orthogonal, symmetric, n=2^k);
  2. block rotation application + W_eff materialization;
  3. ternary quantize (codes, scale, threshold semantics);
  4. RotatedTernaryLinear init contract (latent = R_out W0 R_in);
  5. TokenWindowIter wrap determinism (both arms' byte-identical
     stream contract);
  6. auditor: frozen bars (0.97 ppl ratio, arc/lambada margins),
     parity gate, live-kill verdicts (loss_gap -> FAIL,
     nan_loss -> INVALID).
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


ah1 = _load("ah1_native_hadamard")
audit = _load("audit_ah1")


# ---- rotation math ----------------------------------------------------------
def test_hadamard_orthogonal_symmetric() -> None:
    h = ah1.sylvester_hadamard(64)
    assert h.shape == (64, 64)
    eye = torch.eye(64)
    assert torch.allclose(h @ h, eye, atol=1e-5)
    assert torch.allclose(h, h.T, atol=1e-6)


def test_hadamard_requires_power_of_two() -> None:
    with pytest.raises(ValueError):
        ah1.sylvester_hadamard(63)


def test_rotate_blocks_matches_blockdiag() -> None:
    h = ah1.sylvester_hadamard(4)
    x = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    expected = x @ torch.block_diag(h)
    assert torch.allclose(ah1.rotate_blocks(x, h), expected, atol=1e-5)


def test_rotate_blocks_multi_block() -> None:
    h = ah1.sylvester_hadamard(4)
    x = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    expected = x @ torch.block_diag(h, h)
    assert torch.allclose(ah1.rotate_blocks(x, h), expected, atol=1e-5)


def test_rotate_blocks_rejects_bad_dim() -> None:
    h = ah1.sylvester_hadamard(4)
    with pytest.raises(ValueError):
        ah1.rotate_blocks(torch.zeros(2, 6), h)


def test_materialize_w_eff_full_rotation() -> None:
    h = ah1.sylvester_hadamard(4)
    q = torch.tensor([[1.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 1.0],
                      [0.0, -1.0, 0.0, 1.0, 1.0, 0.0, -1.0, 0.0],
                      [1.0, 1.0, 0.0, 0.0, -1.0, 0.0, 1.0, -1.0],
                      [0.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0]])
    w = ah1.materialize_w_eff(q, 0.5, h, True, True)
    expected = torch.block_diag(h) @ (q * 0.5) @ torch.block_diag(h, h)
    assert torch.allclose(w, expected, atol=1e-5)


def test_materialize_w_eff_no_output_rotation() -> None:
    h = ah1.sylvester_hadamard(4)
    q = torch.ones(2, 4)
    w = ah1.materialize_w_eff(q, 1.0, h, True, False)
    expected = q @ torch.block_diag(h)
    assert torch.allclose(w, expected, atol=1e-5)


def test_materialize_w_eff_rejects_bad_dims() -> None:
    h = ah1.sylvester_hadamard(4)
    with pytest.raises(ValueError):
        ah1.materialize_w_eff(torch.ones(2, 4), 1.0, h, True, True)


# ---- ternary quantize -------------------------------------------------------
def test_ternary_quantize_codes_and_scale() -> None:
    w = torch.tensor([[1.0, -1.0], [0.2, -0.2]])
    q, scale = ah1.ternary_quantize(w)
    assert scale == pytest.approx(0.6, abs=1e-6)
    assert q.min().item() >= -1.0 and q.max().item() <= 1.0
    assert set(torch.unique(q).tolist()) <= {-1.0, 0.0, 1.0}


def test_ternary_quantize_threshold_semantics() -> None:
    # |w|/scale < 0.5 -> 0 ; > 0.5 -> +-1 (values chosen clear of
    # round-half ties)
    w = torch.tensor([[0.2, 0.8, -0.2, -0.8]])
    q, _ = ah1.ternary_quantize(w)
    assert q.tolist()[0] == [0.0, 1.0, 0.0, -1.0]


# ---- rotated linear init contract -------------------------------------------
def test_rotated_linear_latent_is_rotated_w0() -> None:
    torch.manual_seed(0)
    h = ah1.sylvester_hadamard(4)
    w0 = torch.randn(8, 8)
    mod = ah1.RotatedTernaryLinear(w0, None, h, rotate_out=True)
    expected = torch.block_diag(h, h) @ w0 @ torch.block_diag(h, h)
    assert torch.allclose(mod.weight_latent.detach(), expected, atol=1e-5)


def test_rotated_linear_no_out_latent() -> None:
    torch.manual_seed(0)
    h = ah1.sylvester_hadamard(4)
    w0 = torch.randn(8, 8)
    mod = ah1.RotatedTernaryLinear(w0, None, h, rotate_out=False)
    expected = w0 @ torch.block_diag(h, h)
    assert torch.allclose(mod.weight_latent.detach(), expected, atol=1e-5)


def test_rotated_linear_rejects_bad_dims() -> None:
    h = ah1.sylvester_hadamard(4)
    with pytest.raises(ValueError):
        ah1.RotatedTernaryLinear(torch.randn(8, 6), None, h, rotate_out=True)


# ---- token stream contract --------------------------------------------------
def test_token_window_iter_wrap_deterministic(tmp_path) -> None:
    import numpy as np
    stream_path = tmp_path / "s.npy"
    np.save(stream_path, np.arange(1000, dtype=np.int64))
    stream = np.load(stream_path, mmap_mode="r")
    ah1.BATCH, ah1.SEQ_LEN = 2, 4  # window = 9 tokens incl. label target
    it = ah1.TokenWindowIter(stream)
    it2 = ah1.TokenWindowIter(stream)
    for _ in range(3 * len(it)):
        a = next(it)
        b = next(it2)
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    # wrap: window 0 and window len(it) are identical
    it3 = ah1.TokenWindowIter(stream)
    w0 = next(it3)
    for _ in range(len(it) - 1):
        next(it3)
    w1 = next(it3)
    assert torch.equal(w0[0], w1[0]) and torch.equal(w0[1], w1[1])


# ---- auditor thresholds -----------------------------------------------------
def _arms(ctrl_eval, had_eval, parity_gap=0.01, aborts=(), steps=31_250):
    arms = {}
    for name, ev in [("control", ctrl_eval), ("hadamard", had_eval)]:
        arms[name] = {
            "arm": name,
            "summary": {"steps": steps},
            "eval": ev,
            "parity": {"step0_loss": 10.8 + parity_gap / 2
                       if name == "hadamard" else 10.8},
            "abort": aborts[0] if name == "hadamard" and aborts else None,
            "history_finite": True,
        }
    return arms


def test_audit_bars_pass_case() -> None:
    bars = audit.evaluate_thresholds(_arms(
        {"wikitext": 40.0, "arc_easy": 0.25, "lambada_openai": 0.10},
        {"wikitext": 38.0, "arc_easy": 0.26, "lambada_openai": 0.11},
    ))
    assert all(b["pass"] for b in bars.values())


def test_audit_bars_fail_on_ppl() -> None:
    bars = audit.evaluate_thresholds(_arms(
        {"wikitext": 40.0, "arc_easy": 0.25, "lambada_openai": 0.10},
        {"wikitext": 39.5, "arc_easy": 0.30, "lambada_openai": 0.15},
    ))
    assert not bars["ppl_ratio"]["pass"]
    assert bars["arc_margin"]["pass"] and bars["lambada_margin"]["pass"]


def test_audit_bars_fail_on_arc_regression() -> None:
    bars = audit.evaluate_thresholds(_arms(
        {"wikitext": 40.0, "arc_easy": 0.25, "lambada_openai": 0.10},
        {"wikitext": 38.0, "arc_easy": 0.20, "lambada_openai": 0.11},
    ))
    assert bars["ppl_ratio"]["pass"]
    assert not bars["arc_margin"]["pass"]


def _write_run(tmp_path, ctrl_eval, had_eval, parity_gap=0.01,
               had_abort=None, steps=31_250) -> Path:
    """Write a synthetic run directory in the auditor's expected layout."""
    import json as _json
    for name, ev in [("control", ctrl_eval), ("hadamard", had_eval)]:
        sdir = tmp_path / name
        sdir.mkdir()
        (sdir / "summary.json").write_text(
            _json.dumps({"arm": name, "steps": steps}))
        (sdir / "eval.summary.json").write_text(_json.dumps(ev))
        loss = 10.8 + (parity_gap / 2 if name == "hadamard" else 0.0)
        (sdir / f"parity_{name}.json").write_text(
            _json.dumps({"arm": name, "step0_loss": loss}))
        (sdir / "history.jsonl").write_text(
            "\n".join(_json.dumps({"step": s, "loss": 4.0})
                      for s in range(0, steps, 100)) + "\n")
        if name == "hadamard" and had_abort is not None:
            (sdir / "abort.json").write_text(_json.dumps(had_abort))
    return tmp_path


EVAL_PASS = {"wikitext": 40.0, "arc_easy": 0.25, "lambada_openai": 0.10}
EVAL_HAD = {"wikitext": 38.0, "arc_easy": 0.26, "lambada_openai": 0.11}


def test_audit_full_pass(tmp_path) -> None:
    result = audit.audit(_write_run(tmp_path, EVAL_PASS, EVAL_HAD))
    assert result["integrity_ok"] is True
    assert result["verdict"] == "DECIDED"
    assert result["decision"] == "PASS"


def test_audit_full_fail_on_ppl_bar(tmp_path) -> None:
    had = {"wikitext": 39.5, "arc_easy": 0.30, "lambada_openai": 0.15}
    result = audit.audit(_write_run(tmp_path, EVAL_PASS, had))
    assert result["verdict"] == "DECIDED"
    assert result["decision"] == "FAIL"
    assert "frozen bar" in result["decision_reason"]


def test_audit_parity_gap_invalid(tmp_path) -> None:
    result = audit.audit(_write_run(tmp_path, EVAL_PASS, EVAL_HAD,
                                    parity_gap=0.3))
    assert result["verdict"] == "INVALID"
    assert result["parity"]["pass"] is False


def test_audit_loss_gap_kill_is_fail(tmp_path) -> None:
    result = audit.audit(_write_run(
        tmp_path, EVAL_PASS, EVAL_HAD,
        had_abort={"reason": "loss_gap", "step": 5000,
                   "loss_gap_ratio": 1.2}))
    assert result["verdict"] == "DECIDED"
    assert result["decision"] == "FAIL"
    assert "loss-gap" in result["decision_reason"]


def test_audit_nan_loss_kill_is_invalid(tmp_path) -> None:
    result = audit.audit(_write_run(
        tmp_path, EVAL_PASS, EVAL_HAD,
        had_abort={"reason": "nan_loss", "step": 100}))
    assert result["verdict"] == "INVALID"


def test_audit_short_steps_invalid(tmp_path) -> None:
    result = audit.audit(_write_run(tmp_path, EVAL_PASS, EVAL_HAD,
                                    steps=10_000))
    assert result["verdict"] == "INVALID"
    assert any("steps" in f for f in result["integrity_failures"])
