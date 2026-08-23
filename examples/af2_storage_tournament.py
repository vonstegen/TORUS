"""EXP-AF-002 - AF2 equal-storage tournament (A-RP-002)."""
from __future__ import annotations

import argparse
import dataclasses
import gc
import importlib.util
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

sys.modules.setdefault("triton", None)


def _load_helper(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


EXAMPLES = Path(__file__).resolve().parent
_af1 = _load_helper(EXAMPLES / "af1_budget_control.py", "af1")
_eval_lm = _load_helper(EXAMPLES / "eval_lm.py", "eval_lm")

load_wikitext_ids = _af1.load_wikitext_ids
make_window_sampler = _af1.make_window_sampler
next_token_ce_loss = _af1.next_token_ce_loss
train_arm = _af1.train_arm


@dataclasses.dataclass(frozen=True)
class CostVector:
    deployed_bytes: int
    training_flops: int
    inference_ops_per_token: int
    memory_traffic_per_token: int
    latency_per_token_titan_rtx: Optional[float]
    energy_per_token: Optional[float]

    def as_dict(self) -> dict:
        return {
            "deployed_bytes": self.deployed_bytes,
            "training_flops": self.training_flops,
            "inference_ops_per_token": self.inference_ops_per_token,
            "memory_traffic_per_token": self.memory_traffic_per_token,
            "latency_per_token_titan_rtx": self.latency_per_token_titan_rtx,
            "energy_per_token": self.energy_per_token,
        }


class SiteAdapter:
    is_untrained: bool = False

    def patch(self, parent_module): raise NotImplementedError
    def trainable_parameters(self) -> list: return []

    def serialize(self, out_dir: Path, *, training_flops: int,
                  inference_ops_per_token: int) -> CostVector:
        raise NotImplementedError


def _patch_module_forward(parent_module, residual_fn):
    original_forward = parent_module.forward

    def patched_forward(*args, **kwargs):
        out = original_forward(*args, **kwargs)
        x = args[0] if args else kwargs.get("hidden_states",
                                              kwargs.get("inputs"))
        if x is None:
            return out
        if isinstance(out, tuple):
            return (out[0] + residual_fn(x),) + out[1:]
        return out + residual_fn(x)

    parent_module.forward = patched_forward


class T2TernaryAdapter(SiteAdapter):
    """Latent shape (out_features, in_features) where in_features is
    the LAST dim of x at the site and out_features is the LAST dim of
    the site output. The serialized residual is W (full, ternary)
    applied as F.linear(x, q_ste) * scale."""
    is_untrained = False

    def __init__(self, *, in_features: int, out_features: int,
                 device: str = "cpu", dtype=None,
                 train: bool = True, init_seed: Optional[int] = None):
        import torch
        torch.manual_seed(init_seed if init_seed is not None
                          else torch.seed() % (2**31))
        self.latent = torch.nn.Parameter(
            0.01 * torch.randn(out_features, in_features,
                               device=device, dtype=dtype))
        if not train:
            self.latent.requires_grad_(False)
        self._train = train

    def patch(self, parent_module):
        def residual(x):
            import torch.nn.functional as F
            import torch as _t
            r = self.latent
            scale = r.abs().amax(dim=1, keepdim=True).clamp(min=1e-6)
            thresholds = scale / 3
            q = _t.where(r >  thresholds,  scale, _t.zeros_like(r))
            q = _t.where(r < -thresholds, -scale, q)
            q_ste = r + (q - r).detach()
            y = F.linear(x, q_ste)
            return y * scale.squeeze(1)
        _patch_module_forward(parent_module, residual)

    def trainable_parameters(self):
        return [self.latent] if self._train else []

    def serialize(self, out_dir, *, training_flops, inference_ops_per_token):
        import torch
        out_dir.mkdir(parents=True, exist_ok=True)
        r = self.latent.detach()
        scale = r.abs().amax(dim=1, keepdim=True).clamp(min=1e-6)
        thresholds = scale / 3
        q = torch.where(r >  thresholds,  scale, torch.zeros_like(r))
        q = torch.where(r < -thresholds, -scale, q)
        coded = (q + 1).clamp(0, 2).to(torch.int8).contiguous().cpu().numpy()
        flat = coded.reshape(-1)
        pad = (4 - flat.size % 4) % 4
        if pad:
            flat = np.concatenate([flat, np.zeros(pad, dtype=np.int8)])
        packed = ((flat[0::4] & 0x3) | ((flat[1::4] & 0x3) << 2)
                  | ((flat[2::4] & 0x3) << 4) | ((flat[3::4] & 0x3) << 6))
        packed = packed.astype(np.uint8)
        scale_np = scale.detach().cpu().numpy().astype(np.float16)
        np.savez(out_dir / "adapter.npz",
                 packed=packed, scale=scale_np,
                 shape=np.asarray(r.shape, dtype=np.int64))
        (out_dir / "adapter.npz.meta.json").write_text(json.dumps({
            "format": "t2_ternary_2bpw_per_row_fp16_scale",
            "out_features": int(r.shape[0]),
            "in_features": int(r.shape[1]),
            "pack_bytes_per_code": 2,
            "scale_dtype": "fp16",
        }, indent=2))
        deployed_bytes = (out_dir / "adapter.npz").stat().st_size \
            + (out_dir / "adapter.npz.meta.json").stat().st_size
        return CostVector(deployed_bytes=deployed_bytes,
                          training_flops=training_flops,
                          inference_ops_per_token=inference_ops_per_token,
                          memory_traffic_per_token=deployed_bytes,
                          latency_per_token_titan_rtx=None,
                          energy_per_token=None)


def intN_residual_adapter(*, N_bits: int, column_mask_fraction: float,
                          in_features: int, out_features: int,
                          device: str = "cpu", dtype=None,
                          train: bool = True,
                          init_seed: Optional[int] = None):
    import torch
    import torch.nn.functional as F
    torch.manual_seed(init_seed if init_seed is not None
                       else torch.seed() % (2**31))
    keep_count = max(1, int(round(column_mask_fraction * out_features)))
    keep_mask = torch.zeros(out_features, dtype=torch.bool, device=device)
    keep_mask[:keep_count] = True
    latent = torch.nn.Parameter(
        0.01 * torch.randn(out_features, in_features,
                           device=device, dtype=dtype))
    if not train:
        latent.requires_grad_(False)
    levels = (1 << (N_bits - 1)) - 1
    step = 1.0 / levels

    class _IntNCls(SiteAdapter):
        is_untrained = (not train)

    inst = _IntNCls()
    inst._latent = latent
    inst._keep_mask = keep_mask
    inst._N_bits = N_bits
    inst._column_mask = column_mask_fraction
    inst.trainable_parameters = lambda: ([latent] if train else [])

    def residual(x):
        masked = latent * keep_mask.unsqueeze(1).to(latent.device)
        qmax = masked.abs().amax(dim=1, keepdim=True).clamp(min=step)
        q = torch.round(masked / (qmax / levels)) * (qmax / levels)
        q_ste = masked + (q - masked).detach()
        return F.linear(x, q_ste)

    def patch(parent_module):
        _patch_module_forward(parent_module, residual)

    inst.patch = patch

    def serialize(out_dir, *, training_flops, inference_ops_per_token):
        out_dir.mkdir(parents=True, exist_ok=True)
        r = latent.detach()
        qmax = r.abs().amax(dim=1, keepdim=True).clamp(min=step)
        q_int = torch.round(r / (qmax / levels)).clamp(-levels, levels)
        q_np = q_int.cpu().numpy()
        ub = (q_np + (1 << (N_bits - 1))).astype(np.uint8)
        keep_np = keep_mask.cpu().numpy().astype(bool)
        ub_masked = ub[keep_np]
        scale_masked = (qmax.cpu().numpy().astype(np.float16))[keep_np]
        if N_bits <= 4:
            if ub_masked.size % 2 != 0:
                ub_masked = np.concatenate(
                    [ub_masked, np.zeros(1, dtype=np.uint8)])
            packed = ((ub_masked[0::2] & 0xF)
                      | ((ub_masked[1::2] & 0xF) << 4)).astype(np.uint8)
            np.savez(out_dir / "adapter.npz",
                     packed=packed, scale=scale_masked)
        else:
            np.savez(out_dir / "adapter.npz",
                     codes=ub_masked.astype(np.uint8),
                     scale=scale_masked)
        (out_dir / "adapter.npz.meta.json").write_text(json.dumps({
            "format": f"int{N_bits}_per_row_fp16_scale",
            "N_bits": N_bits,
            "out_features": int(r.shape[0]),
            "in_features": int(r.shape[1]),
            "column_mask_fraction": float(column_mask_fraction),
            "packed_columns": int(keep_count),
        }, indent=2))
        deployed_bytes = (out_dir / "adapter.npz").stat().st_size \
            + (out_dir / "adapter.npz.meta.json").stat().st_size
        return CostVector(deployed_bytes=deployed_bytes,
                          training_flops=training_flops,
                          inference_ops_per_token=inference_ops_per_token,
                          memory_traffic_per_token=deployed_bytes,
                          latency_per_token_titan_rtx=None,
                          energy_per_token=None)

    inst.serialize = serialize
    return inst


def lora_adapter(*, in_dim: int, out_dim: int, rank: int = 216,
                 device: str = "cpu", train: bool = True,
                 init_seed: Optional[int] = None):
    """Low-rank parallel branch.
        W_down: (rank, in_dim)   fp16
        W_up:   (out_dim, rank)  fp16
        residual(x) = (x @ W_down.T) @ W_up.T
    """
    import torch
    import torch.nn as nn
    torch.manual_seed(init_seed if init_seed is not None
                      else torch.seed() % (2**31))
    W_down = nn.Parameter(torch.randn(rank, in_dim, device=device,
                                       dtype=torch.float16) * 0.01)
    W_up = nn.Parameter(torch.randn(out_dim, rank, device=device,
                                     dtype=torch.float16) * 0.01)
    if not train:
        W_down.requires_grad_(False)
        W_up.requires_grad_(False)

    class _LoRACls(SiteAdapter):
        is_untrained = (not train)

    inst = _LoRACls()
    inst._W_down = W_down
    inst._W_up = W_up

    def residual(x):
        wd = W_down.to(dtype=x.dtype)
        wu = W_up.to(dtype=x.dtype)
        return (x @ wd.T) @ wu.T

    def patch(parent_module):
        _patch_module_forward(parent_module, residual)

    inst.patch = patch
    inst.trainable_parameters = lambda: ([W_down, W_up] if train else [])
    inst._rank = rank

    def serialize(out_dir, *, training_flops, inference_ops_per_token):
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(out_dir / "adapter.npz",
                 W_down=W_down.detach().cpu().numpy().astype(np.float16),
                 W_up=W_up.detach().cpu().numpy().astype(np.float16))
        (out_dir / "adapter.npz.meta.json").write_text(json.dumps({
            "format": "lora_fp16", "rank": rank,
            "in_dim": in_dim, "out_dim": out_dim,
        }, indent=2))
        deployed_bytes = (out_dir / "adapter.npz").stat().st_size \
            + (out_dir / "adapter.npz.meta.json").stat().st_size
        return CostVector(deployed_bytes=deployed_bytes,
                          training_flops=training_flops,
                          inference_ops_per_token=inference_ops_per_token,
                          memory_traffic_per_token=deployed_bytes,
                          latency_per_token_titan_rtx=None,
                          energy_per_token=None)

    inst.serialize = serialize
    return inst


def dense_adapter_bottleneck(*, in_dim: int, out_dim: int,
                              bottleneck: int = 192, device: str = "cpu",
                              train: bool = True,
                              init_seed: Optional[int] = None):
    """Bottleneck-fp16 parallel branch: in_dim -> bottleneck -> out_dim."""
    import torch
    import torch.nn as nn
    torch.manual_seed(init_seed if init_seed is not None
                      else torch.seed() % (2**31))
    W_down = nn.Parameter(torch.randn(bottleneck, in_dim, device=device,
                                       dtype=torch.float16) * 0.01)
    W_up = nn.Parameter(torch.randn(out_dim, bottleneck, device=device,
                                     dtype=torch.float16) * 0.01)
    if not train:
        W_down.requires_grad_(False)
        W_up.requires_grad_(False)

    class _DenseCls(SiteAdapter):
        is_untrained = (not train)

    inst = _DenseCls()
    inst._W_down = W_down
    inst._W_up = W_up

    def residual(x):
        wd = W_down.to(dtype=x.dtype)
        wu = W_up.to(dtype=x.dtype)
        return (x @ wd.T) @ wu.T

    def patch(parent_module):
        _patch_module_forward(parent_module, residual)

    inst.patch = patch
    inst.trainable_parameters = lambda: ([W_down, W_up] if train else [])

    def serialize(out_dir, *, training_flops, inference_ops_per_token):
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(out_dir / "adapter.npz",
                 W_down=W_down.detach().cpu().numpy().astype(np.float16),
                 W_up=W_up.detach().cpu().numpy().astype(np.float16))
        (out_dir / "adapter.npz.meta.json").write_text(json.dumps({
            "format": "dense_adapter_fp16_bottleneck",
            "bottleneck": bottleneck,
            "in_dim": in_dim, "out_dim": out_dim,
        }, indent=2))
        deployed_bytes = (out_dir / "adapter.npz").stat().st_size \
            + (out_dir / "adapter.npz.meta.json").stat().st_size
        return CostVector(deployed_bytes=deployed_bytes,
                          training_flops=training_flops,
                          inference_ops_per_token=inference_ops_per_token,
                          memory_traffic_per_token=deployed_bytes,
                          latency_per_token_titan_rtx=None,
                          energy_per_token=None)

    inst.serialize = serialize
    return inst


# ---------------------------------------------------------------------------
# Cost targets + arm lists
# ---------------------------------------------------------------------------

TARGET_DEPLOYED_BYTES = {
    "t2_ternary":   4_194_404,
    "int4_residual":4_194_404,
    "int8_residual":4_194_404,
    "lora":         4_423_680,
    "dense_adapter":3_932_160,
}
TRAINED_ARMS = list(TARGET_DEPLOYED_BYTES.keys())
ALL_ARMS = TRAINED_ARMS + ["random_t2_ternary", "random_lora"]


def resolve_target_module(model, target_path: str):
    parts = target_path.split(".")
    node = model
    for p in parts[:-1]:
        node = getattr(node, p)
    leaf = getattr(node, parts[-1])
    if hasattr(leaf, "weight") and hasattr(leaf.weight, "device"):
        return leaf
    for sub in leaf.named_children():
        if hasattr(sub[1], "weight") and hasattr(sub[1].weight, "device"):
            return sub[1]
def build_base(model_name: str, *, dtype: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=getattr(torch, dtype), low_cpu_mem_usage=True,
    ).to(device)
    return model
def build_site_adapter(arm_id: str, *, target_module, hidden_size: int,
                       intermediate_size: int):
    """Build a SiteAdapter at the given target_module.

    Site dim semantics: target_module is `down_proj`-equivalent,
    input dim = intermediate_size (the size of the gate/up output),
    output dim = hidden_size (the size of the residual stream).
    """
    import torch
    device = target_module.weight.device
    dtype = target_module.weight.dtype
    if arm_id == "t2_ternary":
        ad = T2TernaryAdapter(in_features=intermediate_size,
                              out_features=hidden_size,
                              device=device, dtype=dtype, train=True)
        ad.patch(target_module)
        return ad
    if arm_id == "int4_residual":
        ad = intN_residual_adapter(N_bits=4, column_mask_fraction=0.5,
                                   in_features=intermediate_size,
                                   out_features=hidden_size,
                                   device=device, dtype=dtype, train=True)
        ad.patch(target_module)
        return ad
    if arm_id == "int8_residual":
        ad = intN_residual_adapter(N_bits=8, column_mask_fraction=0.25,
                                   in_features=intermediate_size,
                                   out_features=hidden_size,
                                   device=device, dtype=dtype, train=True)
        ad.patch(target_module)
        return ad
    if arm_id == "lora":
        ad = lora_adapter(in_dim=intermediate_size,
                          out_dim=hidden_size, rank=216,
                          device=device, train=True)
        ad.patch(target_module)
        return ad
    if arm_id == "dense_adapter":
        ad = dense_adapter_bottleneck(in_dim=intermediate_size,
                                       out_dim=hidden_size, bottleneck=192,
                                       device=device, train=True)
        ad.patch(target_module)
        return ad
    if arm_id == "random_t2_ternary":
        ad = T2TernaryAdapter(in_features=intermediate_size,
                              out_features=hidden_size,
                              device=device, dtype=dtype, train=False,
                              init_seed=12345)
        ad.patch(target_module)
        return ad
    if arm_id == "random_lora":
        ad = lora_adapter(in_dim=intermediate_size,
                          out_dim=hidden_size, rank=216,
                          device=device, train=False, init_seed=12345)
        ad.patch(target_module)
        return ad
    raise ValueError(f"unknown arm_id: {arm_id}")


def build_base(model_name: str, *, dtype: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM
    torch.set_grad_enabled(False)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=getattr(torch, dtype), low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    return model


def measure_latency_seconds(forward_fn, *, warmup: int = 5, iters: int = 100):
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        torch.cuda.synchronize()
        for _ in range(warmup):
            forward_fn()
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            starts[i].record(); forward_fn(); ends[i].record()
        torch.cuda.synchronize()
        times_ms = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
        return times_ms[iters // 2] / 1000.0
    except Exception:
        return None


def eval_arm(model, tokenizer, *, tasks: List[str], limit=None,
             batch_size: int = 4):
    return _eval_lm.run_lm_eval(
        model=model, tokenizer=tokenizer, tasks=tasks,
        batch_size=batch_size, limit=limit,
    )


def analytic_training_flops(n_steps, b, s, h, i):
    return 6 * n_steps * b * s * (h * i + h * h)


def run_one_seed(*, arm: str, seed: int, args, out_dir: Path,
                 tokenizer, pad_id: int, all_ids: np.ndarray) -> dict:
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)

    arm_dir = out_dir / f"seed-{seed:03d}" / arm
    arm_dir.mkdir(parents=True, exist_ok=True)

    model = build_base(args.model, dtype=args.dtype, device=args.device)
    target_module = resolve_target_module(model, args.target_module)
    intermediate_size = getattr(model.config, "intermediate_size",
                                  4 * model.config.hidden_size)

    adapter = build_site_adapter(
        arm, target_module=target_module,
        hidden_size=model.config.hidden_size,
        intermediate_size=intermediate_size,
    )

    def forward_fn(ids):
        return model(input_ids=ids).logits

    history = []
    if adapter.trainable_parameters():
        data = make_window_sampler(
            all_ids, args.batch_size, args.seq_len,
            seed=seed, device=args.device,
        )
        history = train_arm(
            forward_fn, adapter.trainable_parameters(), data,
            n_steps=args.n_steps, lr=args.lr,
            momentum=args.momentum, grad_clip=args.grad_clip,
            log_every=25, pad_id=pad_id,
        )

    cv = adapter.serialize(
        arm_dir,
        training_flops=analytic_training_flops(
            args.n_steps, args.batch_size, args.seq_len,
            model.config.hidden_size, intermediate_size),
        inference_ops_per_token=(2 * model.config.hidden_size
                                  * intermediate_size),
    )
    target_bytes = TARGET_DEPLOYED_BYTES.get(arm, 0)
    matched_pct = (abs(cv.deployed_bytes - target_bytes) / target_bytes * 100
                    if target_bytes else 0.0)
    cv_matched = matched_pct <= args.matched_bytes_tolerance_pct

    if torch.cuda.is_available():
        try:
            lat = measure_latency_seconds(
                lambda: forward_fn(torch.zeros(1, 1, dtype=torch.long,
                                                 device=args.device)))
            object.__setattr__(cv, "latency_per_token_titan_rtx", lat)
        except Exception:
            pass

    eval_summary_dict = {
        "arm": arm, "seed": seed, "model": args.model,
        "target_module": args.target_module,
        "n_steps": args.n_steps, "batch_size": args.batch_size,
        "seq_len": args.seq_len, "lr": args.lr, "limit": None,
        "tasks": {}, "matched_bytes_target": target_bytes,
        "matched_bytes_actual": cv.deployed_bytes,
        "matched_bytes_tolerance_pct": args.matched_bytes_tolerance_pct,
        "matched_bytes_passed": cv_matched,
        "cost_vector": cv.as_dict(),
        "is_untrained_control": bool(adapter.is_untrained),
    }
    if not adapter.is_untrained:
        try:
            model.eval()
            eval_results = eval_arm(
                model, tokenizer,
                tasks=args.tasks.split(","), limit=None,
                batch_size=args.batch_size,
            )
            if isinstance(eval_results, dict) and "results" in eval_results:
                eval_summary_dict["tasks"] = {
                    t: {"metric": k, "value": v}
                    for t, res in eval_results["results"].items()
                    for k, v in res.items()
                    if "acc" in k or "word_perplexity" in k
                }
            (arm_dir / "eval.full.json").write_text(
                json.dumps(eval_results, indent=2))
        except Exception as e:
            eval_summary_dict["eval_error"] = str(e)

    (arm_dir / "eval.summary.json").write_text(
        json.dumps(eval_summary_dict, indent=2))
    (arm_dir / "cost_vector.json").write_text(
        json.dumps(cv.as_dict(), indent=2))
    with open(arm_dir / "history.jsonl", "w") as fh:
        for row in history:
            fh.write(json.dumps(row) + "\n")

    del model, adapter
    gc.collect()
    if args.device.startswith("cuda"):
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    return eval_summary_dict


def aggregate(summaries: list, out_dir: Path) -> dict:
    import statistics
    trained = [s for s in summaries if not s.get("is_untrained_control")]
    controls = [s for s in summaries if s.get("is_untrained_control")]
    out = {"trained_arms": {}, "untrained_controls": {},
           "tolerance_violations": []}

    for grp, key in [(trained, "trained_arms"),
                     (controls, "untrained_controls")]:
        for arm in ALL_ARMS:
            per_seed = [s for s in grp if s["arm"] == arm]
            if not per_seed:
                continue
            per_seed = [s for s in per_seed
                        if s.get("matched_bytes_passed", True)]
            if not per_seed:
                continue
            entry = {
                "n": len(per_seed),
                "matched_bytes": [s["matched_bytes_actual"] for s in per_seed],
                "matched_bytes_target": per_seed[0]["matched_bytes_target"],
                "cost_vector_rows": [s["cost_vector"] for s in per_seed],
                "tasks": {},
            }
            for t_name in ["wikitext", "arc_easy", "lambada_openai"]:
                vals = [s["tasks"].get(t_name, {}).get("value")
                        for s in per_seed
                        if s.get("tasks", {}).get(t_name)]
                vals = [v for v in vals if v is not None]
                if vals:
                    entry["tasks"][t_name] = {
                        "n": len(vals),
                        "mean": statistics.fmean(vals),
                        "stderr": (statistics.stdev(vals) / len(vals) ** 0.5
                                   if len(vals) > 1 else 0.0),
                        "values": vals,
                    }
            out[key][arm] = entry

    for s in trained:
        if not s.get("matched_bytes_passed", True):
            out["tolerance_violations"].append({
                "arm": s["arm"], "seed": s["seed"],
                "actual": s["matched_bytes_actual"],
                "target": s["matched_bytes_target"],
                "delta_pct": round(
                    100 * (s["matched_bytes_actual"]
                           - s["matched_bytes_target"])
                    / s["matched_bytes_target"], 3),
            })

    if "dense_adapter" in out["trained_arms"]:
        diff = {}
        for arm in TRAINED_ARMS:
            if arm == "dense_adapter" or arm not in out["trained_arms"]:
                continue
            row = {}
            for t, st in out["trained_arms"][arm].get("tasks", {}).items():
                ref = out["trained_arms"]["dense_adapter"]["tasks"].get(t)
                if not ref:
                    continue
                a = st["mean"]; b = ref["mean"]
                se_diff = (st["stderr"] ** 2 + ref["stderr"] ** 2) ** 0.5
                in_stderr = (a - b) / se_diff if se_diff > 0 else float("inf")
                row[t] = {"mean_a_minus_b": a - b,
                           "se_diff": se_diff,
                           "in_stderrs": in_stderr}
            diff[arm] = row
        out["difference_from_dense_adapter"] = diff

    out_path = out_dir / "aggregate.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--target-module", required=True)
    p.add_argument("--arms", default=",".join(TRAINED_ARMS))
    p.add_argument("--seeds", default="1,2,3")
    p.add_argument("--n-steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--tasks", default="wikitext,arc_easy,lambada_openai")
    p.add_argument("--ids-cache", type=Path,
                   default=Path("/tmp/wikitext103_train_ids.npy"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--eval-dtype", default="float16")
    p.add_argument("--matched-bytes-tolerance-pct", type=float, default=1.0)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    arms = args.arms.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    print(f"[af2] loading wikitext-103 ids cache: {args.ids_cache}", flush=True)
    all_ids = load_wikitext_ids(tokenizer, args.ids_cache)
    print(f"[af2] wikitext ids: {len(all_ids):,}", flush=True)

    summaries = []
    for arm in arms:
        for seed in seeds:
            print(f"[af2] arm={arm} seed={seed}: starting", flush=True)
            summary = run_one_seed(
                arm=arm, seed=seed, args=args,
                out_dir=args.out_dir, tokenizer=tokenizer, pad_id=pad_id,
                all_ids=all_ids,
            )
            summaries.append(summary)
            print(f"[af2] arm={arm} seed={seed}: done "
                  f"deployed_bytes={summary['matched_bytes_actual']} "
                  f"matched={summary['matched_bytes_passed']}", flush=True)

    agg = aggregate(summaries, args.out_dir)
    print(json.dumps({"n_runs": len(summaries),
                       "tolerance_violations": agg["tolerance_violations"]},
                      indent=2))


if __name__ == "__main__":
    main()
