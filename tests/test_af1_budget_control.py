"""Tests for examples/af1_budget_control.py (EXP-AF-001 harness).

These pin the matched-budget invariants that make AF1 a real control:
  1. same seed over the same token stream -> identical batches
     (arm A and arm B see the same tokens in the same order);
  2. train_arm moves ONLY the parameter set it is handed (arm A:
     all weights; arm B: STE latents only — base stays frozen);
  3. next_token_ce_loss shifts correctly and ignores pad ids;
  4. eval_lm.run_lm_eval forwards `limit` to simple_evaluate
     (the dead-flag plumbing fix).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
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


af1 = _load("af1_budget_control")


def test_window_sampler_deterministic_same_seed() -> None:
    ids = np.arange(10_000, dtype=np.int64)
    a = af1.make_window_sampler(ids, 2, 8, seed=42, device="cpu")
    b = af1.make_window_sampler(ids, 2, 8, seed=42, device="cpu")
    for _ in range(5):
        wa, wb = next(a), next(b)
        assert torch.equal(wa, wb)


def test_window_sampler_differs_across_seeds() -> None:
    ids = np.arange(10_000, dtype=np.int64)
    a = af1.make_window_sampler(ids, 2, 8, seed=1, device="cpu")
    b = af1.make_window_sampler(ids, 2, 8, seed=2, device="cpu")
    batches_a = [next(a) for _ in range(5)]
    batches_b = [next(b) for _ in range(5)]
    assert any(
        not torch.equal(wa, wb) for wa, wb in zip(batches_a, batches_b)
    )


def test_train_arm_moves_only_given_params() -> None:
    """train_arm must update exactly the parameter set it is given —
    arm A gets all weights, arm B gets only the STE latents. A leak
    here is a matched-budget violation."""
    lin = torch.nn.Linear(4, 4)
    head = torch.nn.Linear(4, 4)
    base_before = lin.weight.detach().clone()
    head_before = head.weight.detach().clone()

    def forward_fn(ids):
        # Per-position one-hot-ish features -> per-position logits:
        # (b, seq, feat) -> (b, seq, vocab).
        feats = torch.nn.functional.one_hot(ids, num_classes=4).float()
        h = lin(feats)
        return head(h)

    ids = torch.randint(0, 4, (2, 4)).long()
    data = iter(lambda: ids, None)
    # Train ONLY `head` (the arm-B analogue: base frozen).
    af1.train_arm(
        forward_fn, list(head.parameters()), data,
        n_steps=3, lr=1e-2, momentum=0.0, grad_clip=1.0,
        log_every=1, pad_id=0,
    )
    assert torch.equal(lin.weight.detach(), base_before), (
        "frozen parameter moved — matched-budget violation"
    )
    assert not torch.equal(head.weight.detach(), head_before), (
        "trainable parameter did not move"
    )


def test_next_token_ce_loss_shifts_and_ignores_pad() -> None:
    logits = torch.zeros(1, 4, 5)
    ids = torch.tensor([[3, 1, 2, 0]])  # pad_id 0 at the end
    loss = af1.next_token_ce_loss(logits, ids, pad_id=0)
    # Uniform logits -> ln(5) per non-pad shifted position (3 of them).
    assert abs(float(loss) - float(np.log(5))) < 1e-5


def test_run_lm_eval_forwards_limit(monkeypatch) -> None:
    eval_lm = _load("eval_lm")
    captured = {}

    class FakeHFLM:
        def __init__(self, **kwargs):
            pass

    def fake_simple_evaluate(**kwargs):
        captured.update(kwargs)
        return {"results": {}}

    monkeypatch.setitem(sys.modules, "lm_eval", type(sys)("lm_eval"))
    sys.modules["lm_eval"].simple_evaluate = fake_simple_evaluate
    monkeypatch.setitem(
        sys.modules, "lm_eval.models", type(sys)("lm_eval.models")
    )
    monkeypatch.setitem(
        sys.modules,
        "lm_eval.models.huggingface",
        type(sys)("lm_eval.models.huggingface"),
    )
    sys.modules["lm_eval.models.huggingface"].HFLM = FakeHFLM

    eval_lm.run_lm_eval(
        model=None, tokenizer=None, tasks=["arc_easy"], batch_size=4,
        limit=7,
    )
    assert captured.get("limit") == 7, (
        "limit was not forwarded to simple_evaluate — the dead-flag "
        "bug from EXP-A-011 is back"
    )
