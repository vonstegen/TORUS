"""Phase 3 acceptance gate: lm-eval-harness runner.

Compares OLMo-1B FP16 baseline vs TORUS-distilled student on a
representative set of LM tasks. The ≥90% rule is computed as
`student_score / baseline_score` per task with the threshold per
task averaged at the end.

Two modes:
- `--baseline`: load the model with no quantization.
- `--quantized --load-adapter PATH`: load the model, attach the
  HFStudentAdapter wrapping the target linears, then load the
  trained STE+residual weights back into them. The patched
  modules are switched to a fast eval path (one-time quantize,
  then raw F.linear per call) so lm-eval runs at native HF speed.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

# Triton bypass — same trick as distill_run.py.
sys.modules.setdefault("triton", None)

import torch  # noqa: E402

from torus.train.hf_adapter import HFAdapterConfig, HFStudentAdapter  # noqa: E402


def run_lm_eval(model, tokenizer, tasks: list[str], batch_size: int) -> dict:
    """Run lm-eval-harness on the given model and return results."""
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    return simple_evaluate(model=lm, tasks=tasks, batch_size=batch_size)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="allenai/OLMo-1B-0724-hf")
    p.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    p.add_argument("--mode", choices=["baseline", "quantized"], default="baseline")
    p.add_argument("--load-adapter", default=None,
                   help="Path to .npz produced by `distill_run.py --save-adapter`")
    p.add_argument("--tasks", default="wikitext,lambada_openai,arc_easy",
                   help="comma-separated lm-eval task names")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--attn-impl", default="sdpa")
    p.add_argument("--limit", type=int, default=None,
                   help="Optional cap on examples per task (for smoke tests)")
    p.add_argument("--output", default=None,
                   help="If set, write JSON results to this path")
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[eval] loading tokenizer + model from {args.model!r} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.mode == "baseline":
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=getattr(torch, args.dtype),
            attn_implementation=args.attn_impl,
        ).to(args.device)
        model.eval()
    else:
        cfg = HFAdapterConfig(
            model_name=args.model,
            target_modules=tuple(args.target_modules.split(",")),
            dtype=args.dtype,
            device=args.device,
            attn_implementation=args.attn_impl,
        )
        adapter = HFStudentAdapter(cfg)
        if args.load_adapter is not None:
            adapter.load_state(args.load_adapter)
            print(f"[eval] loaded adapter weights from {args.load_adapter}")
        n_planes = 1
        # Fast forward: pre-quantize once and use raw F.linear per call.
        # Without this, every lm-eval request triggers a per-STE numpy
        # round-trip + ternary_quantize, which is 100-1000x slower.
        adapter.apply_eval_mode(n_planes=n_planes)
        adapter.model.eval()
        model = adapter.model

    tasks = args.tasks.split(",")
    print(f"[eval] running tasks: {tasks}")
    results = run_lm_eval(model, tokenizer, tasks, batch_size=args.batch_size)

    summary: dict = {"model": args.model, "mode": args.mode, "tasks": {}}
    for task, res in results["results"].items():
        for k, v in res.items():
            if k in ("alias", "acc,none", "acc_norm,none", "ppl,none", "word_perplexity,none"):
                summary["tasks"][task] = {"metric": k, "value": v}
                break

    print()
    print("[eval] === results ===")
    for task, rec in summary["tasks"].items():
        print(f"  {task:30s}  {rec['metric']:20s}  {rec['value']:.4f}")

    if args.output is not None:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[eval] wrote {args.output}")

    del model
    if args.mode == "quantized":
        del adapter
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
