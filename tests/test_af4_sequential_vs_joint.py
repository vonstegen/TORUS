"""Tests for examples/af4_sequential_vs_joint.py (EXP-AF-004 harness).

These pin the curriculum invariants that make AF4 a real control:
  1. arm seq freezes the primary latent during stage 2 — and the
     freeze assertion actually fires when stage 2 is handed the
     primary (a matched-curriculum violation must kill the run);
  2. arm joint moves BOTH latents; arm t1_only moves only the
     primary and never engages the residual plane;
  3. every arm consumes exactly 2 * stage_steps batches (matched
     token budget);
  4. the n_planes switch changes the forward (residual plane only
     contributes at n_planes >= 2);
  5. aggregate() pairwise z-score conventions (the claim test reads
     difference["seq_vs_joint"]).
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


af4 = _load("af4_sequential_vs_joint")


def _make_ste(seed: int = 0):
    from torus.train.ste import TernarySTE

    g = torch.Generator().manual_seed(seed)
    w = torch.randn(4, 4, generator=g) * 0.5
    r = torch.randn(4, 4, generator=g) * 0.01
    ste = TernarySTE(
        weight=torch.nn.Parameter(w),
        group_size=4,
        residual_weight=torch.nn.Parameter(r),
    )
    return ste


def _make_forward(ste, n_planes_box: list, counter: list):
    """Synthetic ids -> logits forward through the torch STE."""

    def forward_fn(ids):
        feats = torch.nn.functional.one_hot(ids, num_classes=4).float()
        q_w = ste.forward_torch(n_planes=n_planes_box[0])
        counter[0] += 1
        return feats @ q_w.t()

    return forward_fn


def _data(ids, counter):
    def batches():
        while True:
            counter[0] += 1
            yield ids

    return batches()


def _run(arm, ste, stage_steps=3):
    n_planes_box = [1]
    fwd_counter = [0]
    batch_counter = [0]
    ids = torch.randint(0, 4, (2, 4)).long()
    forward_fn = _make_forward(ste, n_planes_box, fwd_counter)
    data = _data(ids, batch_counter)
    result = af4.run_curriculum(
        arm,
        forward_fn,
        [ste.weight],
        [ste.residual_weight],
        lambda n: n_planes_box.__setitem__(0, n),
        data,
        stage_steps=stage_steps,
        lr=1e-2,
        momentum=0.0,
        grad_clip=1.0,
        log_every=1,
        pad_id=0,
    )
    return result, batch_counter[0]


def test_seq_freezes_primary_and_trains_residual() -> None:
    ste = _make_ste(seed=1)
    w0 = ste.weight.detach().clone()
    r0 = ste.residual_weight.detach().clone()
    result, n_batches = _run("seq", ste)
    assert result["freeze_check"] is True
    assert not torch.equal(ste.weight.detach(), w0), (
        "primary latent did not move during stage 1"
    )
    assert not torch.equal(ste.residual_weight.detach(), r0), (
        "residual latent did not move during stage 2"
    )
    assert n_batches == 2 * 3, "seq consumed the wrong token budget"
    assert "history_stage1" in result and "history_stage2" in result


def test_seq_freeze_assertion_fires_on_violation() -> None:
    """If stage 2 is (mis)wired to the primary latent, the freeze
    check must raise — a silent pass here is a matched-curriculum
    violation."""
    ste = _make_ste(seed=2)
    n_planes_box = [1]
    ids = torch.randint(0, 4, (2, 4)).long()
    forward_fn = _make_forward(ste, n_planes_box, [0])
    data = iter(lambda: ids, None)
    with pytest.raises(RuntimeError, match="freeze invariant"):
        af4.run_curriculum(
            "seq",
            forward_fn,
            [ste.weight],
            [ste.weight],  # BUG: stage 2 handed the primary latent
            lambda n: n_planes_box.__setitem__(0, n),
            data,
            stage_steps=3,
            lr=1e-2,
            momentum=0.0,
            grad_clip=1.0,
            log_every=1,
            pad_id=0,
        )


def test_joint_moves_both_planes() -> None:
    ste = _make_ste(seed=3)
    w0 = ste.weight.detach().clone()
    r0 = ste.residual_weight.detach().clone()
    result, n_batches = _run("joint", ste)
    assert not torch.equal(ste.weight.detach(), w0)
    assert not torch.equal(ste.residual_weight.detach(), r0)
    assert n_batches == 2 * 3
    assert "history" in result


def test_t1_only_never_touches_residual() -> None:
    ste = _make_ste(seed=4)
    r0 = ste.residual_weight.detach().clone()
    result, n_batches = _run("t1_only", ste)
    assert torch.equal(ste.residual_weight.detach(), r0), (
        "t1_only engaged the residual plane — storage/capacity leak"
    )
    assert n_batches == 2 * 3
    assert "history" in result


def test_n_planes_switch_controls_residual_contribution() -> None:
    ste = _make_ste(seed=5)
    q1 = ste.forward_torch(n_planes=1).detach()
    q2 = ste.forward_torch(n_planes=2).detach()
    assert not torch.equal(q1, q2), (
        "n_planes=2 forward does not include the residual plane"
    )
    with torch.no_grad():
        ste.residual_weight.zero_()
    q1z = ste.forward_torch(n_planes=1).detach()
    q2z = ste.forward_torch(n_planes=2).detach()
    assert torch.allclose(q1z, q2z, atol=1e-6), (
        "zero residual should make 1- and 2-plane forwards agree"
    )


def test_eval_n_planes_matches_deployed_form() -> None:
    assert af4.EVAL_N_PLANES == {"t1_only": 1, "joint": 2, "seq": 2}


def test_aggregate_pairwise_zscore_conventions() -> None:
    def summary(arm, seed, ppl, arc, lam):
        return {
            "arm": arm,
            "seed": seed,
            "tasks": {
                "wikitext": {"metric": "word_perplexity,none", "value": ppl},
                "arc_easy": {"metric": "acc,none", "value": arc},
                "lambada_openai": {"metric": "acc,none", "value": lam},
            },
        }

    summaries = []
    for seed in (1, 2, 3):
        summaries.append(summary("seq", seed, 20.0 + seed, 0.60, 0.55))
        summaries.append(summary("joint", seed, 30.0 + seed, 0.58, 0.54))
        summaries.append(summary("t1_only", seed, 40.0 + seed, 0.57, 0.53))

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        agg = af4.aggregate(summaries, Path(td))

    diff = agg["difference"]["seq_vs_joint"]["wikitext"]
    # seq ppl is 10 lower than joint at every seed -> mean_seq -
    # mean_joint = -10; identical per-seed spreads -> stderrs equal.
    assert diff["mean_seq_minus_joint"] == pytest.approx(-10.0)
    assert diff["difference_in_stderrs"] < 0
    assert "seq_vs_t1_only" in agg["difference"]
    assert "joint_vs_t1_only" in agg["difference"]
    assert agg["arms"]["seq"]["wikitext"]["n"] == 3
