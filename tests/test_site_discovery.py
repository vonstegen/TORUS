"""Tests for the EXP-RPM-SITE-DISCOVERY sweep summarizer.

Covers the frozen decision rules in
examples/site_discovery_summary.py:

  - the Stage 2 v2 QUALIFYING rule reproduces the four frozen pilot
    decisions exactly (AF2-D NO, L15-down YES, L0-q NO, L0-v YES);
  - the CANDIDATE rule (qualifying AND max ppl >= 100);
  - knob parsing for sigma-/thr- dirs;
  - baseline (zero-knob) verification gate;
  - reference-cell band checks (frozen pilot values);
  - per-site and sweep-level summaries end-to-end on synthetic data,
    including candidate priority and the NO_SECOND_SITE / INVALID
    decision mapping.

Pure synthetic JSON; no torch needed.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, EXAMPLES / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sds = _load("site_discovery_summary")


# ---- frozen Stage 2 v2 pilot maps (STAGE2-V2-CAL-VERDICT.md) ---------

AF2D_GAUSS = {
    0.00: [13.0932], 0.05: [13.0962], 0.10: [13.1030],
    0.20: [13.1280], 0.50: [13.3730], 1.00: [15.3519],
}
L15_GAUSS = {
    0.00: [13.0932], 0.05: [13.0990], 0.10: [13.1230],
    0.20: [13.2030], 0.50: [13.7530], 1.00: [16.5780],
}
L0Q_GAUSS = {
    0.00: [13.0932], 0.05: [13.0950], 0.10: [13.0970],
    0.20: [13.1020], 0.50: [13.1310], 1.00: [13.2470],
}
L0V_GAUSS = {
    0.00: [13.0932], 0.05: [13.1190], 0.10: [13.1966],
    0.20: [13.7253], 0.50: [439.2520], 1.00: [20083.4880],
}


def _rows(mapping):
    return sds.summarize_knobs(mapping)


@pytest.mark.parametrize(
    "mapping, expected",
    [
        (AF2D_GAUSS, False),   # 2 bands {13, 15}
        (L15_GAUSS, True),     # bands {13, 14, 17}, span 3.48
        (L0Q_GAUSS, False),    # 1 band {13}
        (L0V_GAUSS, True),     # bands {13, 14, 439, 20083}
    ],
)
def test_qualifying_rule_reproduces_frozen_pilot_decisions(mapping,
                                                           expected):
    qualifying, rule = sds.apply_qualifying_rule(_rows(mapping))
    assert qualifying is expected, rule


def test_qualifying_rule_requires_band_separation():
    # Three distinct bands but two adjacent (13, 14) -> not separated.
    mapping = {0.0: [13.09], 0.5: [13.6], 1.0: [14.2]}
    rows = _rows(mapping)
    qualifying, rule = sds.apply_qualifying_rule(rows)
    assert rule["n_distinct_ppl_bands"] == 2  # round(13.6)=14, round(14.2)=14
    assert qualifying is False


def test_qualifying_rule_needs_min_span():
    mapping = {0.0: [13.09], 0.5: [13.7], 1.0: [14.4]}
    rows = _rows(mapping)
    qualifying, rule = sds.apply_qualifying_rule(rows)
    assert qualifying is False
    assert rule["ppl_span"] < 2.0


def test_candidate_rule():
    assert sds.candidate_verdict(True, 88.0)["candidate"] is False
    assert "mild axis" in sds.candidate_verdict(True, 88.0)["reason"]
    assert sds.candidate_verdict(True, 439.0)["candidate"] is True
    assert sds.candidate_verdict(False, 439.0)["candidate"] is False
    assert sds.candidate_verdict(True, None)["candidate"] is False


def test_parse_knob():
    assert sds.parse_knob("sigma-00_05") == pytest.approx(0.05)
    assert sds.parse_knob("sigma-01_00") == pytest.approx(1.0)
    assert sds.parse_knob("thr-00_40") == pytest.approx(0.4)
    assert sds.parse_knob("seed-001") is None
    assert sds.parse_knob("baseline") is None


def _write_pte(path: Path, ppl: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {"tasks": {"wikitext": {"value": ppl}}}))


def test_baseline_verdict_sigma_zero_cell(tmp_path):
    site_dir = tmp_path / "site"
    _write_pte(site_dir / "sigma-00_00" / "seed-001" / "seed-001"
               / "pre_train_eval.json", 13.0932)
    ok, ppl = sds.baseline_verdict(site_dir)
    assert ok is True
    assert ppl == pytest.approx(13.0932, abs=1e-6)


def test_baseline_verdict_off_band(tmp_path):
    site_dir = tmp_path / "site"
    _write_pte(site_dir / "sigma-00_00" / "seed-001" / "seed-001"
               / "pre_train_eval.json", 15.0)
    ok, _ = sds.baseline_verdict(site_dir)
    assert ok is False


def test_baseline_verdict_baseline_dir(tmp_path):
    site_dir = tmp_path / "site"
    _write_pte(site_dir / "baseline" / "seed-001" / "seed-001"
               / "pre_train_eval.json", 13.0932)
    ok, _ = sds.baseline_verdict(site_dir)
    assert ok is True


def test_baseline_verdict_missing(tmp_path):
    ok, ppl = sds.baseline_verdict(tmp_path / "site")
    assert ok is False
    assert ppl is None


def _ref_run_dir(tmp_path, gauss_ppl, twn_ppl):
    run = tmp_path / "run"
    _write_pte(run / "ref-gauss-v-L0" / "sigma-00_50" / "seed-001"
               / "seed-001" / "pre_train_eval.json", gauss_ppl)
    _write_pte(run / "ref-twn-d-L0" / "thr-00_70" / "seed-001"
               / "seed-001" / "pre_train_eval.json", twn_ppl)
    return run


def test_reference_cells_in_band(tmp_path):
    run = _ref_run_dir(tmp_path, 439.2520, 429.55)
    refs = sds.check_reference_cells(run)
    assert all(r["ok"] for r in refs)


def test_reference_cells_out_of_band(tmp_path):
    run = _ref_run_dir(tmp_path, 13.09, 429.55)
    refs = sds.check_reference_cells(run)
    by_id = {r["site_id"]: r for r in refs}
    assert by_id["ref-gauss-v-L0"]["ok"] is False
    assert by_id["ref-twn-d-L0"]["ok"] is True


def _site_dir(tmp_path, name, mapping, baseline_ppl=13.0932):
    site = tmp_path / name
    for knob, ppls in mapping.items():
        for i, ppl in enumerate(ppls, start=1):
            tag = f"{knob:05.2f}".replace(".", "_")
            _write_pte(site / f"sigma-{tag}" / f"seed-{i:03d}"
                       / f"seed-{i:03d}" / "pre_train_eval.json", ppl)
    _write_pte(site / "sigma-00_00" / "seed-001" / "seed-001"
               / "pre_train_eval.json", baseline_ppl)
    return site


def test_summarize_site_qualifying_candidate(tmp_path):
    site = _site_dir(tmp_path, "gauss-v-L4", L0V_GAUSS)
    s = sds.summarize_site(site, "gauss-v-L4")
    assert s["qualifying"] is True
    assert s["candidate"] is True
    assert s["baseline_ok"] is True
    assert (site / "site_cal_summary.json").exists()


def test_summarize_site_qualifying_mild_non_candidate(tmp_path):
    site = _site_dir(tmp_path, "gauss-d-L4", L15_GAUSS)
    s = sds.summarize_site(site, "gauss-d-L4")
    assert s["qualifying"] is True
    assert s["candidate"] is False
    assert "mild axis" in s["candidate_reason"]


def test_summarize_site_not_qualifying(tmp_path):
    site = _site_dir(tmp_path, "gauss-d-L1", L0Q_GAUSS)
    s = sds.summarize_site(site, "gauss-d-L1")
    assert s["qualifying"] is False
    assert s["candidate"] is False
    assert s["candidate_reason"] == "not qualifying"


def test_summarize_site_baseline_off_band_invalid(tmp_path):
    site = _site_dir(tmp_path, "gauss-v-L8", L0V_GAUSS, baseline_ppl=15.0)
    s = sds.summarize_site(site, "gauss-v-L8")
    assert s["baseline_ok"] is False


def _full_run(tmp_path, *, refs_ok=True, with_candidates=True):
    run = tmp_path / "run"
    _write_pte(run / "ref-gauss-v-L0" / "sigma-00_50" / "seed-001"
               / "seed-001" / "pre_train_eval.json",
               439.2520 if refs_ok else 13.09)
    _write_pte(run / "ref-twn-d-L0" / "thr-00_70" / "seed-001"
               / "seed-001" / "pre_train_eval.json", 429.55)
    if with_candidates:
        _site_dir(run, "gauss-v-L4", L0V_GAUSS)
    _site_dir(run, "gauss-d-L1", L0Q_GAUSS)
    _site_dir(run, "gauss-d-L8", L15_GAUSS)
    return run


def test_sweep_candidates_found(tmp_path):
    run = _full_run(tmp_path)
    s = sds.summarize_sweep(run)
    assert s["run_valid"] is True
    assert s["decision"] == "CANDIDATES_FOUND"
    assert s["candidates"] == ["gauss-v-L4"]
    assert s["candidate_priority"][0]["category"] == "attention v_proj"
    assert s["qualifying_non_candidates"][0]["site_id"] == "gauss-d-L8"


def test_sweep_no_second_site(tmp_path):
    run = _full_run(tmp_path, with_candidates=False)
    s = sds.summarize_sweep(run)
    assert s["run_valid"] is True
    assert s["decision"] == "NO_SECOND_SITE"
    assert s["n_candidates"] == 0


def test_sweep_invalid_on_ref_failure(tmp_path):
    run = _full_run(tmp_path, refs_ok=False)
    s = sds.summarize_sweep(run)
    assert s["decision"] == "INVALID"
    assert s["run_valid"] is False


def test_sweep_attention_candidate_prioritized_over_mlp(tmp_path):
    run = tmp_path / "run"
    _write_pte(run / "ref-gauss-v-L0" / "sigma-00_50" / "seed-001"
               / "seed-001" / "pre_train_eval.json", 439.2520)
    _write_pte(run / "ref-twn-d-L0" / "thr-00_70" / "seed-001"
               / "seed-001" / "pre_train_eval.json", 429.55)
    # MLP candidate with a LARGER span than the attention candidate:
    # priority still puts the attention-category site first.
    _site_dir(run, "gauss-d-L4", L0V_GAUSS)      # span ~20070, mlp
    big_attn = {0.0: [13.09], 0.5: [100.0], 1.0: [5000.0]}
    _site_dir(run, "gauss-v-L2", big_attn)       # span ~4987, attention
    s = sds.summarize_sweep(run)
    assert s["candidates"] == ["gauss-v-L2", "gauss-d-L4"]
    assert s["candidate_priority"][0]["site_id"] == "gauss-v-L2"
