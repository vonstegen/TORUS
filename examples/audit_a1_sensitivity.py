"""EXP-A-011 audit: aggregate per-layer sensitivity JSONs and
compare to the manifest's frozen thresholds.

Reads every per-layer summary JSON in <run_dir>/per_layer/ plus the
aggregate sensitivity_table.json (if present), computes per-category
and per-block aggregates, and prints a verdict against the manifest's
preregistered PASS / FAIL / INCONCLUSIVE criteria.

Run on the Legion box where the run namespace lives:
    ./.venv/bin/python examples/audit_a1_sensitivity.py \\
        --run-dir runs/a/EXP-A-011/<timestamp> \\
        > audit_report.txt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, pstdev


# Category labels per the manifest.
CATEGORIES = {
    "q_proj": "attention_q",
    "k_proj": "attention_k",
    "v_proj": "attention_v",
    "o_proj": "attention_o",
    "gate_proj": "mlp_gate",
    "up_proj": "mlp_up",
    "down_proj": "mlp_down",
}


def parse_target(name: str) -> tuple[str, int | None, str | None]:
    """Return (kind, layer_idx, short_name) for a target name.

    kind is one of: 'reference_f16', 'reference_full',
    'attention', 'mlp', 'embed', 'head', 'unknown'.
    """
    if name == "f16_reference":
        return ("reference_f16", None, None)
    if name == "fully_quantized":
        return ("reference_full", None, None)
    if name == "model.embed_tokens":
        return ("embed", None, "embed_tokens")
    if name == "lm_head":
        return ("head", None, "lm_head")
    # model.layers.<i>.self_attn.<short>
    if name.startswith("model.layers."):
        parts = name.split(".")
        if len(parts) == 5 and parts[3] == "self_attn":
            try:
                idx = int(parts[2])
            except ValueError:
                return ("unknown", None, None)
            return ("attention", idx, parts[4])
        if len(parts) == 5 and parts[3] == "mlp":
            try:
                idx = int(parts[2])
            except ValueError:
                return ("unknown", None, None)
            return ("mlp", idx, parts[4])
    return ("unknown", None, None)


def parse_per_layer_summary(path: Path) -> dict:
    with open(path) as f:
        s = json.load(f)
    # The driver writes `target_modules` (a list of FQ names) into the
    # The per-arm summary written by eval_lm.py has `target_modules`
    # (a list of FQ names) but no `arm` field. Per-layer arms have a
    # single FQ name in target_modules. Reference arms have either
    # the legacy short-name list (q_proj,k_proj,...) or no
    # target_modules. Detect the case and tag with a synthetic arm
    # name that parse_target() understands.
    arm: str | None
    mode = s.get("mode")
    tm = s.get("target_modules")
    if mode == "baseline":
        arm = "f16_reference"
    elif mode == "quantized" and isinstance(tm, list) and len(tm) == 1:
        arm = tm[0]
    elif mode == "quantized" and isinstance(tm, list) and len(tm) > 1:
        # Multi-target arm: this is the fully-quantized reference.
        arm = "fully_quantized"
    elif s.get("arm") is not None:
        arm = s["arm"]
    else:
        arm = path.name[: -len(".summary.json")]
    return {
        "arm": arm,
        "wikitext_ppl": s.get("tasks", {}).get("wikitext", {}).get("value"),
        "arc_easy_acc": s.get("tasks", {}).get("arc_easy", {}).get("value"),
        "no_calibrate": s.get("no_calibrate"),
        "limit": s.get("limit"),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True,
                   help="EXP-A-011 run namespace, e.g. runs/a/EXP-A-011/<ts>")
    args = p.parse_args()

    per_layer_dir = args.run_dir / "per_layer"
    if not per_layer_dir.exists():
        print(f"[audit] ERROR: {per_layer_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    rows: list[dict] = []
    for path in sorted(per_layer_dir.glob("*.summary.json")):
        # Skip the sidecar 'X.summary.json.full.json' files; the
        # pattern above only matches *.summary.json, so we're safe.
        rec = parse_per_layer_summary(path)
        kind, idx, short = parse_target(rec["arm"] or path.stem.replace(".summary", ""))
        rec["kind"] = kind
        rec["layer_idx"] = idx
        rec["short"] = short
        rec["path"] = path
        rows.append(rec)

    # Extract FP16 reference.
    f16 = next((r for r in rows if r["kind"] == "reference_f16"), None)
    fq = next((r for r in rows if r["kind"] == "reference_full"), None)
    if f16 is None:
        print("[audit] ERROR: no FP16 reference found", file=sys.stderr)
        sys.exit(1)

    f16_ppl = f16["wikitext_ppl"]
    f16_arc = f16["arc_easy_acc"]
    print(f"[audit] FP16 reference: ppl={f16_ppl:.4f}, arc_easy={f16_arc:.4f}")
    if fq is not None:
        print(f"[audit] fully-quantized reference: ppl={fq['wikitext_ppl']:.1f}, arc_easy={fq['arc_easy_acc']:.4f}")

    # Per-arm deltas.
    per_layer = [r for r in rows if r["kind"] in ("attention", "mlp", "embed", "head")]
    print(f"[audit] {len(per_layer)} per-layer arms")
    for r in per_layer:
        if r["wikitext_ppl"] is not None:
            r["ppl_ratio"] = r["wikitext_ppl"] / f16_ppl
            r["ppl_delta"] = r["wikitext_ppl"] - f16_ppl
        if r["arc_easy_acc"] is not None:
            r["arc_delta"] = r["arc_easy_acc"] - f16_arc

    # Per-category aggregates.
    print()
    print("[audit] === per-category aggregates (excluding embed/head) ===")
    print(f"  {'category':14s} {'n':>4s} {'mean_ppl':>12s} {'std_ppl':>12s} {'min_ppl':>12s} {'max_ppl':>12s}")
    cat_ppls: dict[str, list[float]] = {}
    for r in per_layer:
        if r["kind"] not in ("attention", "mlp"):
            continue
        cat = CATEGORIES.get(r["short"], r["short"] or "?")
        cat_ppls.setdefault(cat, []).append(r["wikitext_ppl"])
    for cat, ppls in sorted(cat_ppls.items()):
        print(f"  {cat:14s} {len(ppls):4d} {mean(ppls):12.1f} {pstdev(ppls) if len(ppls) > 1 else 0:12.1f} {min(ppls):12.1f} {max(ppls):12.1f}")

    # Per-block aggregates.
    print()
    print("[audit] === per-block aggregates ===")
    block_ppls: dict[int, list[float]] = {}
    for r in per_layer:
        if r["layer_idx"] is None:
            continue
        block_ppls.setdefault(r["layer_idx"], []).append(r["wikitext_ppl"])
    for idx in sorted(block_ppls):
        ppls = block_ppls[idx]
        print(f"  layer {idx:2d}  n={len(ppls):3d}  mean_ppl={mean(ppls):12.1f}  min={min(ppls):12.1f}  max={max(ppls):12.1f}")

    # Late-attn-o vs early-mlp-down (manifest frozen threshold).
    print()
    print("[audit] === manifest threshold checks ===")
    late_attn_o = [r["wikitext_ppl"] for r in per_layer
                   if r["kind"] == "attention" and r["short"] == "o_proj"
                   and r["layer_idx"] is not None and r["layer_idx"] >= 12]
    early_mlp_down = [r["wikitext_ppl"] for r in per_layer
                      if r["kind"] == "mlp" and r["short"] == "down_proj"
                      and r["layer_idx"] is not None and r["layer_idx"] < 4]
    if late_attn_o and early_mlp_down:
        ratio = mean(late_attn_o) / mean(early_mlp_down)
        print(f"  late-attn-o mean ppl: {mean(late_attn_o):.1f} (n={len(late_attn_o)})")
        print(f"  early-mlp-down mean ppl: {mean(early_mlp_down):.1f} (n={len(early_mlp_down)})")
        print(f"  ratio late/early: {ratio:.2f}  (PASS bar: >= 1.5x)")

    # Spread check.
    if per_layer:
        all_ppls = [r["wikitext_ppl"] for r in per_layer if r["wikitext_ppl"] is not None]
        print(f"  per-layer ppl range: {min(all_ppls):.1f} .. {max(all_ppls):.1f} (spread = {max(all_ppls)/min(all_ppls):.1f}x)")

    # Pass/fail summary.
    print()
    print("[audit] === verdict (preliminary) ===")
    n_completed = len([r for r in per_layer if r["wikitext_ppl"] is not None])
    n_total = 114
    print(f"  coverage: {n_completed}/{n_total} per-layer arms completed")
    if n_completed >= 60:
        print(f"  coverage bar: PASS (>= 60)")
    else:
        print(f"  coverage bar: FAIL (< 60) — CONTINUE per manifest")
    # Spread: did the per-layer map find any structure at all?
    if per_layer:
        all_ppls = [r["wikitext_ppl"] for r in per_layer if r["wikitext_ppl"] is not None]
        if min(all_ppls) < f16_ppl * 100 and max(all_ppls) > f16_ppl * 1e5:
            print(f"  spread: strong signal — single-layer quant degrades ppl from "
                  f"{min(all_ppls):.1f} to {max(all_ppls):.1f} (vs FP16 {f16_ppl:.1f})")
        elif max(all_ppls) > f16_ppl * 1000:
            print(f"  spread: meaningful — max single-layer ppl = {max(all_ppls):.1f}")


if __name__ == "__main__":
    main()
