"""Tests for EXP-AF-006 (AF6 dataset/context robustness).

Pins the new corpus-perplexity eval path (the only new measurement in
AF6 — the lm-eval ladder path is unchanged):
  1. corpus_perplexity is deterministic (same ids -> same value);
  2. it uses the correct next-token shift (a model that always
   predicts the actual next token gets ppl == 1);
  3. it respects the deterministic-window contract (first n_windows
   non-overlapping windows);
  4. a uniform model gives ppl == vocab_size;
  5. the --cross-eval-ppl-cache wiring records cache identity + sha
   in the cell summary (source-level contract).
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


af2 = _load("af2_storage_tournament")


class _FixedLM(torch.nn.Module):
    """Toy 'model': logits = one_hot(ids) @ W with W set so position t
    always predicts a fixed function of id[t]. Vocab = 8."""

    def __init__(self, table: np.ndarray):
        super().__init__()
        self.register_buffer(
            "W", torch.tensor(table, dtype=torch.float32)
        )

    def forward(self, input_ids):
        feats = torch.nn.functional.one_hot(
            input_ids, num_classes=8
        ).float()
        logits = feats @ self.W

        class Out:
            pass

        out = Out()
        out.logits = logits
        return out


def test_corpus_perplexity_deterministic() -> None:
    rng = np.random.default_rng(0)
    ids = rng.integers(0, 8, size=600, dtype=np.int64)
    model = _FixedLM(rng.normal(size=(8, 8)).astype(np.float32))
    p1 = af2.corpus_perplexity(model, ids, seq_len=32, n_windows=4,
                               batch_size=2, device="cpu")
    p2 = af2.corpus_perplexity(model, ids, seq_len=32, n_windows=4,
                               batch_size=2, device="cpu")
    assert p1 == p2


def test_corpus_perplexity_perfect_model_is_one() -> None:
    """A model that always puts all mass on the actual next token
    must score ppl == 1 — this pins the shift semantics."""
    ids = np.arange(0, 200, dtype=np.int64) % 8
    # table[t, n] = big if n == (t+1) % 8 else -big
    table = np.full((8, 8), -20.0, dtype=np.float32)
    for t in range(8):
        table[t, (t + 1) % 8] = 20.0
    model = _FixedLM(table)
    ppl = af2.corpus_perplexity(model, ids, seq_len=16, n_windows=4,
                                batch_size=2, device="cpu")
    assert ppl == pytest.approx(1.0, abs=1e-4)


def test_corpus_perplexity_uniform_model_is_vocab() -> None:
    ids = np.arange(0, 200, dtype=np.int64) % 8
    model = _FixedLM(np.zeros((8, 8), dtype=np.float32))
    ppl = af2.corpus_perplexity(model, ids, seq_len=16, n_windows=4,
                                batch_size=2, device="cpu")
    assert ppl == pytest.approx(8.0, rel=1e-4)


def test_corpus_perplexity_window_contract() -> None:
    """Different ids in LATER windows must not change the value
    (first-n_windows non-overlapping contract)."""
    rng = np.random.default_rng(1)
    ids_a = rng.integers(0, 8, size=600, dtype=np.int64)
    ids_b = ids_a.copy()
    ids_b[4 * 32 + 1:] = rng.integers(0, 8, size=600 - 4 * 32 - 1)
    model = _FixedLM(rng.normal(size=(8, 8)).astype(np.float32))
    pa = af2.corpus_perplexity(model, ids_a, seq_len=32, n_windows=4,
                               batch_size=2, device="cpu")
    pb = af2.corpus_perplexity(model, ids_b, seq_len=32, n_windows=4,
                               batch_size=2, device="cpu")
    assert pa == pb


def test_cross_eval_flag_records_cache_identity() -> None:
    """The wiring must record cache path + sha256 + window parameters
    in the cell summary (the audit reads these)."""
    import inspect

    src = inspect.getsource(af2.run_one_seed)
    assert "cross_corpus_ppl" in src
    assert "sha256" in src
    assert "n_windows" in src
