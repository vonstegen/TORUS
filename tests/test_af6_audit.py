"""Tests for examples/audit_af6_context.py (EXP-AF-006b auditor).

Pins the frozen AF6b threshold evaluation:
  1. reference reproduction band ([17.91, 24.01] on seq128 mean);
  2. Q1 window-robust/window-sensitive regime labeling + the
     'recovers, step-confounded' interpretation rule;
  3. Q2 cross-transfer bar and own-corpus recovery ratio (with the
     verification.json denominators);
  4. integrity: missing cells, non-uniform bytes, non-finite ppl.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")  # noqa: F401 — suite convention

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, EXAMPLES / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


af6 = _load("audit_af6_context")


def _cells(seq128_ppls=(20.0, 21.0, 19.0), seq16_ppls=(22.0, 23.0, 21.0),
           seq256_ppls=(18.0, 19.0, 17.0), owt_ppls=(30.0, 31.0, 29.0),
           owt_cross=(25.0, 26.0, 24.0), seq128_cross=(28.0, 27.0, 29.0)):
    def rows(ppls, crosses=None):
        return [
            {"seed": i + 1, "ppl": p, "arc_easy": 0.6,
             "lambada_openai": 0.55,
             "cross_owt_ppl": (crosses[i] if crosses else None),
             "deployed_bytes": 4199318, "n_steps": 500, "seq_len": 128,
             "path": "x"}
            for i, p in enumerate(ppls)
        ]

    return {
        "seq16": rows(seq16_ppls),
        "seq128": rows(seq128_ppls, seq128_cross),
        "seq256": rows(seq256_ppls),
        "owt": rows(owt_ppls, owt_cross),
    }


VERIF = {"fp16_owt_test_ppl": 14.0, "damaged_owt_test_ppl": 93.0}


def test_reference_reproduction_band() -> None:
    res = af6.evaluate_reference(_cells())
    assert res["reproduced"] is True
    bad = af6.evaluate_reference(_cells(seq128_ppls=(30.0, 31.0, 29.0)))
    assert bad["reproduced"] is False


def test_q1_labels_window_robust() -> None:
    res = af6.evaluate_q1(_cells())
    assert res["seq16_label"] == "window-robust"
    assert all(r["window_robust"] for r in res["regimes"].values())


def test_q1_labels_step_confounded_when_seq16_wins() -> None:
    res = af6.evaluate_q1(_cells(seq16_ppls=(8.0, 8.1, 7.9)))
    # seq16 materially BETTER (lower ppl) than seq128 -> z > 2.
    assert res["seq16_label"] == "recovers, step-confounded"
    assert res["seq16_vs_seq128_z"] > 2


def test_q1_labels_window_sensitive() -> None:
    res = af6.evaluate_q1(_cells(seq16_ppls=(300.0, 310.0, 95.0)))
    assert res["regimes"]["seq16"]["window_sensitive"] is True
    assert res["seq16_label"] == "window-sensitive"


def test_q2_recovery_ratio_and_transfer() -> None:
    res = af6.evaluate_q2(_cells(), VERIF)
    # transfer: all 3 owt cells <= 100 -> holds
    assert res["cross_capability"]["transfer_holds"] is True
    # ratio: (93 - 25) / (93 - 14) = 68/79 ~ 0.86 >= 0.5
    assert res["own_corpus_recovery"]["holds"] is True
    assert res["own_corpus_recovery"]["ratio_stats"]["mean"] > 0.8
    assert res["wt_trained_cross_direction_covariate"][
        "owt_test_ppl_values"] == [28.0, 27.0, 29.0]


def test_q2_transfer_fails_when_cells_above_bar() -> None:
    res = af6.evaluate_q2(_cells(owt_ppls=(150.0, 160.0, 140.0)), VERIF)
    assert res["cross_capability"]["transfer_holds"] is False


def test_integrity_checks(tmp_path) -> None:
    cells = _cells()
    # Write the expected layout.
    for regime, rows in cells.items():
        for r in rows:
            d = tmp_path / regime / f"seed-{r['seed']:03d}" / "t2_ternary"
            d.mkdir(parents=True)
            (d / "eval.summary.json").write_text(json.dumps({
                "seed": r["seed"],
                "tasks": {"wikitext": {"value": r["ppl"]}},
                "matched_bytes_actual": r["deployed_bytes"],
            }))
            (d / "history.jsonl").write_text(
                json.dumps({"step": 0, "loss": 4.0}) + "\n")
    loaded = af6.load_cells(tmp_path)
    res = af6.check_integrity(loaded, tmp_path)
    assert res["ok"] is True

    # One non-finite loss breaks integrity.
    bad = tmp_path / "seq16" / "seed-001" / "t2_ternary" / "history.jsonl"
    bad.write_text(json.dumps({"step": 0, "loss": float("nan")}) + "\n")
    res = af6.check_integrity(af6.load_cells(tmp_path), tmp_path)
    assert res["ok"] is False
