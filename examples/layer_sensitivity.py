"""EXP-A-011 — A1 layer-sensitivity driver.

Iterates a list of fully-qualified module names from the target
model, runs `eval_lm.py` once per arm with `--target-modules` set
to that single FQN, and assembles a per-layer sensitivity table
written as JSON.

Discovery tier (OPERATING-PLAN §3): one seed, smaller eval
sample (--limit), short budget per arm. The driver is a thin
wrapper around `eval_lm.py`; it does NOT reimplement
quantization or evaluation.

Run on Legion (the only machine with CUDA + the HF cache):
    cd ~/TORUS && ./.venv/bin/python examples/layer_sensitivity.py \\
        --out-dir runs/a/EXP-A-011/<timestamp> \\
        --tasks wikitext,arc_easy --limit 200 \\
        --target-list research/track-a-residual-ternary/EXP-A-011/target_list.json
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

# Triton bypass — same trick as distill_run.py / eval_lm.py.
sys.modules.setdefault("triton", None)


REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_LM = REPO_ROOT / "examples" / "eval_lm.py"


def run_arm(
    *,
    python: str,
    target_module: str,
    out_dir: Path,
    model: str,
    tasks: list[str],
    limit: int | None,
    batch_size: int,
    device: str,
    dtype: str,
) -> tuple[Path, Path]:
    """Run `eval_lm.py` once for a single FQN; return (summary_path, full_path)."""
    safe_name = target_module.replace(".", "__")
    summary = out_dir / "per_layer" / f"{safe_name}.summary.json"
    full = out_dir / "per_layer" / f"{safe_name}.full.json"
    log = out_dir / "per_layer" / f"{safe_name}.log"
    summary.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        python,
        str(EVAL_LM),
        "--model", model,
        "--mode", "quantized",
        "--target-modules", target_module,
        "--no-calibrate",
        "--tasks", ",".join(tasks),
        "--batch-size", str(batch_size),
        "--device", device,
        "--dtype", dtype,
        "--output", str(summary),
    ]
    if limit is not None:
        cmd += ["--limit", str(limit)]

    print(f"[layer_sens] arm={target_module!r}", flush=True)
    print(f"[layer_sens]   cmd: {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    t0 = time.time()
    with open(log, "w") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, check=False)
    dt = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"eval_lm.py failed for {target_module!r} (rc={proc.returncode}); "
            f"see {log}"
        )
    if not summary.exists():
        raise RuntimeError(
            f"eval_lm.py ran but {summary} was not produced; see {log}"
        )
    if not full.exists():
        # Sidecar was added in EXP-A-011; if missing, the eval_lm.py on
        # the path is older. Surface the issue rather than silently
        # proceeding without stderrs.
        raise RuntimeError(
            f"eval_lm.py ran but {full} was not produced; EXP-A-011 requires "
            f"the full-results sidecar. See {log}."
        )
    print(f"[layer_sens]   done in {dt:.1f}s", flush=True)
    return summary, full


def load_target_list(path: Path) -> list[str]:
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"target list {path} must be a JSON list of FQNs")
    if not all(isinstance(x, str) and "." in x for x in data):
        raise ValueError(
            f"target list {path} entries must all be fully-qualified "
            f"module names (containing a dot)"
        )
    return data


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", required=True, type=Path,
                   help="Run namespace directory; per-layer outputs go under <out>/per_layer/")
    p.add_argument("--target-list", type=Path, default=None,
                   help="JSON list of fully-qualified module names. If omitted, the "
                        "driver enumerates q/k/v/o, gate/up/down across all OLMo-1B "
                        "layers plus embed_tokens and lm_head.")
    p.add_argument("--tasks", default="wikitext,arc_easy",
                   help="comma-separated lm-eval task names")
    p.add_argument("--limit", type=int, default=200,
                   help="lm-eval --limit; discovery tier (OPERATING-PLAN §3)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--python", default=sys.executable,
                   help="Python interpreter to invoke (defaults to current venv)")
    p.add_argument("--model", default="allenai/OLMo-1B-0724-hf")
    p.add_argument("--include-reference", action="store_true",
                   help="Also run the FP16 reference and the full-quantized reference arms")
    p.add_argument("--include-fully-quantized", action="store_true",
                   help="Also run the full-quantized reference (114 linears) arm")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.target_list is not None:
        targets = load_target_list(args.target_list)
    else:
        # Enumerate from the model itself. Lazy import so the script
        # can be --help'd without downloading the model.
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(args.model)
        n = cfg.num_hidden_layers
        linears = ["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"]
        targets = [f"model.layers.{i}.self_attn.{n_}" for i in range(n) for n_ in ["q_proj", "k_proj", "v_proj", "o_proj"]]
        targets += [f"model.layers.{i}.mlp.{n_}" for i in range(n) for n_ in ["gate_proj", "up_proj", "down_proj"]]
        targets += ["model.embed_tokens", "lm_head"]

    print(f"[layer_sens] out_dir: {args.out_dir}")
    print(f"[layer_sens] targets: {len(targets)} modules")
    print(f"[layer_sens] tasks: {args.tasks} (limit={args.limit})")
    if args.include_reference or args.include_fully_quantized:
        print(f"[layer_sens] reference arms: f16={args.include_reference} full-quantized={args.include_fully_quantized}")

    table: list[dict] = []
    started = time.time()

    if args.include_reference:
        # FP16 reference: --mode baseline (no quantization, no adapter).
        ref_path = args.out_dir / "per_layer" / "f16_reference.summary.json"
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            args.python, str(EVAL_LM),
            "--model", args.model,
            "--mode", "baseline",
            "--tasks", args.tasks,
            "--batch-size", str(args.batch_size),
            "--device", args.device,
            "--dtype", args.dtype,
            "--output", str(ref_path),
        ]
        if args.limit is not None:
            cmd += ["--limit", str(args.limit)]
        print(f"[layer_sens] arm=FP16 reference")
        with open(args.out_dir / "per_layer" / "f16_reference.log", "w") as lf:
            subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, check=True)
        with open(ref_path) as f:
            ref = json.load(f)
        table.append({"arm": "f16_reference", **ref})

    if args.include_fully_quantized:
        full_q_path = args.out_dir / "per_layer" / "fully_quantized.summary.json"
        full_q_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            args.python, str(EVAL_LM),
            "--model", args.model,
            "--mode", "quantized",
            "--target-modules", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
            "--no-calibrate",
            "--tasks", args.tasks,
            "--batch-size", str(args.batch_size),
            "--device", args.device,
            "--dtype", args.dtype,
            "--output", str(full_q_path),
        ]
        if args.limit is not None:
            cmd += ["--limit", str(args.limit)]
        print(f"[layer_sens] arm=fully-quantized reference")
        with open(args.out_dir / "per_layer" / "fully_quantized.log", "w") as lf:
            subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, check=True)
        with open(full_q_path) as f:
            full_q = json.load(f)
        table.append({"arm": "fully_quantized", **full_q})

    for i, t in enumerate(targets):
        print(f"[layer_sens] ({i+1}/{len(targets)})")
        try:
            summary, full = run_arm(
                python=args.python,
                target_module=t,
                out_dir=args.out_dir, model=args.model, tasks=args.tasks.split(","),
                limit=args.limit, batch_size=args.batch_size,
                device=args.device, dtype=args.dtype,
            )
        except RuntimeError as e:
            print(f"[layer_sens]   FAILED: {e}", flush=True)
            table.append({"arm": t, "error": str(e)})
            continue
        with open(summary) as f:
            s = json.load(f)
        table.append({"arm": t, **s})

    table_path = args.out_dir / "sensitivity_table.json"
    with open(table_path, "w") as f:
        json.dump({
            "model": args.model,
            "tasks": args.tasks.split(","),
            "limit": args.limit,
            "no_calibrate": True,
            "elapsed_seconds": time.time() - started,
            "rows": table,
        }, f, indent=2)
    print(f"[layer_sens] wrote {table_path}")
    print(f"[layer_sens] total: {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
