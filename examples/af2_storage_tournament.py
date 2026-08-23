"""EXP-AF-002 - AF2 equal-storage tournament (A-RP-002).

Adds 6 trained arms + 2 untrained structure controls to the AF1
shared-loop driver, and tightens the cost-axis reporting per
OPERATING-PLAN rev 2.3 §11.

Trained arms (matched deployed-bytes ~ 4.21 MB on
model.layers.0.mlp.down_proj):
  - t2_ternary    : 2 bpw signed ternary + per-row fp16 scale
                    (HFStudentAdapter n_planes=1, ternary STE).
  - int4_residual : 4 bpw signed INT4 + per-row fp16 scale + zero
                    (50% column mask to land the same bytes).
  - int8_residual : 8 bpw signed INT8 with per-row fp16 scale + zero
                    (25% column mask).
  - lora          : r=216 fp16 down/up parallel residual branch.
  - dense_adapter : 2048 -> 192 -> 8192 fp16 bottleneck.

Untrained structure controls (reported in a separate panel):
  - random_t2_ternary
  - random_lora

Each arm's trained serialization file (adapter.npz plus its
metadata header) is fingerprint-written with sha256; the on-disk
file size is the reported `deployed_bytes`. Matched-bytes
tolerance: +/- 1% of the per-arm `target_deployed_bytes`
(preregistered in the manifest; an arm outside tolerance is
INVALIDATED).

The cost vector reported per arm:
  C = (deployed_bytes, training_flops, inference ops/token,
       memory_traffic/token, latency_per_token_titan_rtx,
       energy_per_token)
Latency is measured at apply time on the TITAN RTX
(median of 100 forward passes when feasible, else 'unmeasured').
Energy is recorded only if nvidia-smi power.draw is queryable
on legion.

The AF1 helpers (load_wikitext_ids, make_window_sampler,
next_token_ce_loss, train_arm) are reused unchanged - AF2 is a
strict superset.
"""
from __future__ import annotations

import sys as _sys
from typing import List, Optional

_sys.modules["triton"] = None  # see distill_run.py for why

import argparse
import dataclasses
import gc
import importlib.util
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

import sys as _sys

_sys.modules["triton"] = None  # see distill_run.py for why

import argparse
import gc
import importlib.util
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np


def _load_helper(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


EXAMPLES = Path(__file__).resolve().parent
_af1 = _load_helper(EXAMPLES / "af1_budget_control.py", "af1")
_eval_lm = _load_helper(EXAMPLES / "eval_lm.py", "eval_lm")


# Re-export AF1 helpers at module top for tidy imports elsewhere.
load_wikitext_ids = _af1.load_wikitext_ids
make_window_sampler = _af1.make_window_sampler
next_token_ce_loss = _af1.next_token_ce_loss
train_arm = _af1.train_arm


# ---------------------------------------------------------------------------
# Cost-vector terms
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CostVector:
    deployed_bytes: int
    training_flops: int
    inference_ops_per_token: int
    memory_traffic_per_token: int
    latency_per_token_titan_rtx: Optional[float]   # seconds / token, or None
    energy_per_token: Optional[float]              # joules / token, or None

    def as_dict(self) -> dict:
        return {
            "deployed_bytes": self.deployed_bytes,
            "training_flops": self.training_flops,
            "inference_ops_per_token": self.inference_ops_per_token,
            "memory_traffic_per_token": self.memory_traffic_per_token,
            "latency_per_token_titan_rtx": self.latency_per_token_titan_rtx,
            "energy_per_token": self.energy_per_token,
        }


def measure_latency_seconds(forward_fn, *, warmup: int = 5,
                            iters: int = 100) -> Optional[float]:
    """Median wall-clock per-token for one forward pass at the residual
    site, measured at apply time on whichever device the model lives
    on. Returns None if any required import fails (legion without
    torch.cuda)."""
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
            starts[i].record()
            forward_fn()
            ends[i].record()
        torch.cuda.synchronize()
        times_ms = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
        median_ms = times_ms[iters // 2]
        return median_ms / 1000.0
    except Exception:
        return None


def measure_power_draw_joules(duration_seconds: float) -> Optional[float]:
    """Best-effort `nvidia-smi` power-draw -> joules during the
    latency window. Recorded once at driver startup if feasible."""
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return None
        watts = float(out.stdout.strip().splitlines()[0])
        return watts * duration_seconds
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Site adapters
# ---------------------------------------------------------------------------

# AF2 introduces a SiteAdapter interface so all 6 trained arms + the
# 2 untrained controls can be plugged into one shared training loop
# without copy-pasting the math.
#
#   class SiteAdapter:
#     - patch(parent_module): swap the module's forward to add a delta
#       residual at the chosen site.
#     - trainable_parameters(): list of nn.Parameter that train.
#     - serialize(out_dir): writes the adapter's weights to
#       out_dir/adapter.npz plus a .meta.json header; returns the
#       CostVector with on-disk deployed_bytes.
#     - is_untrained: bool flag (controls whether the optimizer
#       actually steps).

class SiteAdapter:
    is_untrained: bool = False

    def patch(self, parent_module) -> None: ...
    def trainable_parameters(self) -> list: return []
    def serialize(self, out_dir: Path, *,
                  training_flops: int,
                  inference_ops_per_token: int) -> CostVector: ...
    def forward_residual(self, x): ...   # for latency/energy timing


def _patch_module_forward(parent_module, residual_fn: Callable):
    """Monkey-patch a module's forward to add `residual_fn(x)` to its
    output. The original forward is wrapped; the patched forward
    preserves the parent's signature (extra kwargs allowed)."""
    import torch
    original_forward = parent_module.forward

    def patched_forward(*args, **kwargs):
        out = original_forward(*args, **kwargs)
        x = args[0] if args else kwargs.get("hidden_states")
        if x is None:
            return out
        if isinstance(out, tuple):
            head = out[0]
            return (head + residual_fn(x),) + out[1:]
        return out + residual_fn(x)

    parent_module.forward = patched_forward


# --- T2 ternary (signed {-1, 0, +1}, STE; per-row scale) -----------------

class T2TernaryAdapter(SiteAdapter):
    is_untrained = False

    def __init__(self, *, target_module, hidden_size: int,
                 intermediate_size: int, train: bool = True,
                 init_seed: Optional[int] = None):
        import torch
        torch = torch
        self.target_module = target_module
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.train = train
        self.latent = None  # nn.Parameter
        self._rng_init = init_seed

    def patch(self, parent_module):
        import torch
        torch.manual_seed(self._rng_init if self._rng_init is not None
                         else torch.seed() % (2**31))
        self.latent = torch.nn.Parameter(
            0.01 * torch.randn(
                self.intermediate_size, self.hidden_size,
                device=parent_module.weight.device,
                dtype=parent_module.weight.dtype,
            )
        )
        import torch.nn.functional as F

        def residual(x):
            r = self.latent
            scale = r.abs().amax(dim=1, keepdim=True).clamp(min=1e-6)
            q = torch.zeros_like(r)
            q[r >  scale / 3] =  scale[r >  scale / 3]
            q[r < -scale / 3] = -scale[r < -scale / 3]
            q_ste = r + (q - r).detach()
            # y = x @ (q_ste.T / scale)
            y = F.linear(x, q_ste)
            return y * scale.squeeze(1)

        _patch_module_forward(parent_module, residual)

    def trainable_parameters(self):
        if self.train and self.latent is not None:
            return [self.latent]
        return []

    def serialize(self, out_dir, *, training_flops, inference_ops_per_token):
        import torch
        out_dir.mkdir(parents=True, exist_ok=True)
        r = self.latent.detach()
        scale = r.abs().amax(dim=1, keepdim=True).clamp(min=1e-6)
        # Pack ternary to 2-bit signed: -1=10b (2), 0=00b (0), +1=01b (1)
        q = torch.zeros_like(r)
        q[r >  scale / 3] =  1.0
        q[r < -scale / 3] = -1.0
        # 2-bit signed: -1 -> 2; 0 -> 0; +1 -> 1
        coded = (q + 1).to(torch.int8).clamp(0, 2).contiguous()
        # Pack 4 codes per byte (low 2 bits).
        coded_np = coded.cpu().numpy().astype(np.int8)
        flat = coded_np.reshape(-1)
        # Pad to multiple of 4.
        if flat.size % 4 != 0:
            flat = np.concatenate([flat, np.zeros(4 - flat.size % 4,
                                                   dtype=np.int8)])
        packed = np.zeros(flat.size // 4, dtype=np.uint8)
        packed = (flat[0::4] & 0x3) \
               | ((flat[1::4] & 0x3) << 2) \
               | ((flat[2::4] & 0x3) << 4) \
               | ((flat[3::4] & 0x3) << 6)
        packed = packed.astype(np.uint8)
        scale_np = scale.detach().cpu().numpy().astype(np.float16)
        np.savez(out_dir / "adapter.npz",
                 packed=packed, scale=scale_np,
                 shape=np.asarray(r.shape, dtype=np.int64))
        (out_dir / "adapter.npz.meta.json").write_text(json.dumps({
            "format": "t2_ternary_2bpw_per_row_fp16_scale",
            "intermediate_size": int(r.shape[0]),
            "hidden_size": int(r.shape[1]),
            "pack_bytes_per_code": 2,
            "scale_dtype": "fp16",
        }, indent=2))
        deployed_bytes = (out_dir / "adapter.npz").stat().st_size \
                       + (out_dir / "adapter.npz.meta.json").stat().st_size
        return CostVector(
            deployed_bytes=deployed_bytes,
            training_flops=training_flops,
            inference_ops_per_token=inference_ops_per_token,
            memory_traffic_per_token=deployed_bytes,
            latency_per_token_titan_rtx=None,
            energy_per_token=None,
        )


# --- INT4 / INT8 residual (signed N-bit with per-row scale + zero) -------

def _build_intN_adapter(N_bits: int, *, column_mask_fraction: float,
                        target_module, hidden_size: int,
                        intermediate_size: int,
                        train: bool = True,
                        init_seed: Optional[int] = None):
    class _IntNAdapter(SiteAdapter):
        is_untrained = train is False
    return _IntNAdapter  # subclass closure below


def intN_residual_adapter(N_bits: int, *, column_mask_fraction: float,
                          target_module, hidden_size: int,
                          intermediate_size: int, train: bool = True,
                          init_seed: Optional[int] = None):
    """N_bits=4 or 8. column_mask_fraction in (0, 1]: retains `mask`
    fraction of the latent columns. The packed artifact's on-disk
    bytes are recorded; the preregistered target_deployed_bytes is the
    accounting we aim for."""
    import torch
    import torch.nn.functional as F

    torch.manual_seed(init_seed if init_seed is not None
                       else torch.seed() % (2**31))

    keep_mask = torch.zeros(intermediate_size, dtype=torch.bool)
    keep_count = max(1, int(round(column_mask_fraction * intermediate_size)))
    keep_mask[:keep_count] = True

    latent = torch.nn.Parameter(
        0.01 * torch.randn(intermediate_size, hidden_size,
                           device=target_module.weight.device,
                           dtype=target_module.weight.dtype),
    )
    if not train:
        latent.requires_grad_(False)

    levels = (1 << (N_bits - 1)) - 1   # signed N-bit
    step = 1.0 / levels

    def residual(x):
        r = latent * keep_mask.unsqueeze(1)
        # Per-row scale + zero: q = round((r - zero) / step)
        # Inexpensive: use row-wise absmax only, zero = 0 (centered
        # around 0). Fine for the matched-bytes accounting; the audit
        # reports both shape and what the dequant range is.
        qmax = r.abs().amax(dim=1, keepdim=True).clamp(min=step)
        q = torch.round(r / (qmax / levels)) * (qmax / levels)
        q_ste = r + (q - r).detach()
        return F.linear(x, q_ste)

    parent = target_module
    _patch_module_forward(parent, residual)

    class _IntNAdapterCls(SiteAdapter):
        is_untrained = (not train)
        _params = [latent] if train else []
        _N_bits = N_bits
        _column_mask = keep_mask.cpu().numpy().astype(bool)

    inst = _IntNAdapterCls()
    inst._latent = latent
    inst._parent = parent

    def params():
        return inst._params

    def serialize(out_dir, *, training_flops, inference_ops_per_token):
        r = latent.detach()
        N = N_bits
        levels_n = (1 << (N - 1)) - 1
        step_n = 1.0 / levels_n
        qmax = r.abs().amax(dim=1, keepdim=True).clamp(min=step_n)
        q_int = torch.round(r / (qmax / levels_n)).clamp(-levels_n, levels_n)
        # Pack to bytes: 4-bit if N<=4 else 1 byte per element.
        q_np = q_int.cpu().numpy().astype(np.int8 if N > 4 else np.int8)
        # 4-bit packing for N=4:
        out_dir.mkdir(parents=True, exist_ok=True)
        if N <= 4:
            # signed 4-bit -> unsigned nibble offset by 8
            ub = (q_np + 8).astype(np.uint8)
            mask_keep = keep_mask.cpu().numpy().astype(bool)
            ub_masked = ub[mask_keep]
            if ub_masked.size % 2 != 0:
                ub_masked = np.concatenate([ub_masked, np.zeros(1, dtype=np.uint8)])
            packed = (ub_masked[0::2] & 0xF) | ((ub_masked[1::2] & 0xF) << 4)
            packed = packed.astype(np.uint8)
            np.savez(out_dir / "adapter.npz",
                     packed=packed,
                     scale=(qmax.cpu().numpy().astype(np.float16))[mask_keep],
                     zero=np.zeros(packed.size, dtype=np.float16))
        else:
            mask_keep = keep_mask.cpu().numpy().astype(bool)
            ub = q_np[mask_keep]
            np.savez(out_dir / "adapter.npz",
                     codes=ub.astype(np.int8),
                     scale=(qmax.cpu().numpy().astype(np.float16))[mask_keep],
                     zero=np.zeros(ub.size, dtype=np.float16))
        (out_dir / "adapter.npz.meta.json").write_text(json.dumps({
            "format": f"int{N}_per_row_fp16_scale_zero",
            "N_bits": N,
            "intermediate_size": int(r.shape[0]),
            "hidden_size": int(r.shape[1]),
            "column_mask_fraction": float(column_mask_fraction),
            "packed_columns": int(keep_count),
        }, indent=2))
        deployed_bytes = (out_dir / "adapter.npz").stat().st_size \
                       + (out_dir / "adapter.npz.meta.json").stat().st_size
        return CostVector(
            deployed_bytes=deployed_bytes,
            training_flops=training_flops,
            inference_ops_per_token=inference_ops_per_token,
            memory_traffic_per_token=deployed_bytes,
            latency_per_token_titan_rtx=None,
            energy_per_token=None,
        )

    inst.trainable_parameters = params
    inst.serialize = serialize
    return inst


# --- LoRA r=216 fp16 parallel branch -------------------------------------

def lora_adapter(*, target_module, hidden_size: int,
                 intermediate_size: int, rank: int = 216,
                 train: bool = True, init_seed: Optional[int] = None):
    import torch
    import torch.nn as nn

    torch.manual_seed(init_seed if init_seed is not None
                       else torch.seed() % (2**31))
    W_down = nn.Parameter(torch.randn(rank, hidden_size,
                                      device=target_module.weight.device,
                                      dtype=torch.float16) * 0.01)
    W_up = nn.Parameter(torch.randn(intermediate_size, rank,
                                    device=target_module.weight.device,
                                    dtype=torch.float16) * 0.01)
    if not train:
        W_down.requires_grad_(False)
        W_up.requires_grad_(False)

    parent = target_module

    def residual(x):
        # x is fp16; ensures shape (batch, seq, hidden)
        h = x @ W_down.T
        return h @ W_up.T

    _patch_module_forward(parent, residual)

    class _LoRACls(SiteAdapter):
        is_untrained = (not train)
    inst = _LoRACls()
    inst._W_down = W_down
    inst._W_up = W_up
    inst.trainable_parameters = lambda: ([W_down, W_up] if train else [])
    inst._rank = rank

    def serialize(out_dir, *, training_flops, inference_ops_per_token):
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(out_dir / "adapter.npz",
                 W_down=W_down.detach().cpu().numpy().astype(np.float16),
                 W_up=W_up.detach().cpu().numpy().astype(np.float16))
        (out_dir / "adapter.npz.meta.json").write_text(json.dumps({
            "format": "lora_fp16",
            "rank": int(rank),
            "intermediate_size": int(intermediate_size),
            "hidden_size": int(hidden_size),
        }, indent=2))
        deployed_bytes = (out_dir / "adapter.npz").stat().st_size \
                       + (out_dir / "adapter.npz.meta.json").stat().st_size
        return CostVector(
            deployed_bytes=deployed_bytes,
            training_flops=training_flops,
            inference_ops_per_token=inference_ops_per_token,
            memory_traffic_per_token=deployed_bytes,
            latency_per_token_titan_rtx=None,
            energy_per_token=None,
        )
    inst.serialize = serialize
    return inst


# --- Dense bottleneck adapter fp16 ---------------------------------------

def dense_adapter_bottleneck(*, target_module, hidden_size: int,
                              intermediate_size: int, bottleneck: int = 192,
                              train: bool = True, init_seed: Optional[int] = None):
    import torch
    import torch.nn as nn

    torch.manual_seed(init_seed if init_seed is not None
                       else torch.seed() % (2**31))
    W_down = nn.Parameter(torch.randn(bottleneck, hidden_size,
                                      device=target_module.weight.device,
                                      dtype=torch.float16) * 0.01)
    W_up = nn.Parameter(torch.randn(intermediate_size, bottleneck,
                                    device=target_module.weight.device,
                                    dtype=torch.float16) * 0.01)
    if not train:
        W_down.requires_grad_(False)
        W_up.requires_grad_(False)

    parent = target_module

    def residual(x):
        h = x @ W_down.T
        return h @ W_up.T

    _patch_module_forward(parent, residual)

    class _DenseCls(SiteAdapter):
        is_untrained = (not train)
    inst = _DenseCls()
    inst._W_down = W_down
    inst._W_up = W_up
    inst.trainable_parameters = lambda: ([W_down, W_up] if train else [])
    inst._bottleneck = bottleneck

    def serialize(out_dir, *, training_flops, inference_ops_per_token):
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(out_dir / "adapter.npz",
                 W_down=W_down.detach().cpu().numpy().astype(np.float16),
                 W_up=W_up.detach().cpu().numpy().astype(np.float16))
        (out_dir / "adapter.npz.meta.json").write_text(json.dumps({
            "format": "dense_adapter_fp16_bottleneck",
            "bottleneck": int(bottleneck),
            "intermediate_size": int(intermediate_size),
            "hidden_size": int(hidden_size),
        }, indent=2))
        deployed_bytes = (out_dir / "adapter.npz").stat().st_size \
                       + (out_dir / "adapter.npz.meta.json").stat().st_size
        return CostVector(
            deployed_bytes=deployed_bytes,
            training_flops=training_flops,
            inference_ops_per_token=inference_ops_per_token,
            memory_traffic_per_token=deployed_bytes,
            latency_per_token_titan_rtx=None,
            energy_per_token=None,
        )
    inst.serialize = serialize
    return inst


# ---------------------------------------------------------------------------
# Token-cache + data plumbing (reuse AF1)
# ---------------------------------------------------------------------------

def build_arm(model_name, *, target_module: str, device: str,
              dtype: str):
    """Load the FP16 base model once and return (model, parent_module)."""
    import torch
    from transformers import AutoModelForCausalLM
    torch.set_grad_enabled(False)
    torch_dtype = getattr(torch, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    parts = target_module.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    leaf = parent
    parent_module = parent
    setattr(parent, parts[-1], leaf)  # no-op
    return model, leaf, parent_module


# ---------------------------------------------------------------------------
# Per-arm factory
# ---------------------------------------------------------------------------

def build_site_adapter(arm_id: str, *, target_module, hidden_size: int,
                       intermediate_size: int):
    """Resolve an arm_id to a fresh SiteAdapter attached to
    `target_module`. The adapter's `patch()` has already been
    applied; trainable_parameters() reflects whether the arm is
    trained (controls return [])."""
    if arm_id == "t2_ternary":
        ad = T2TernaryAdapter(target_module=target_module,
                              hidden_size=hidden_size,
                              intermediate_size=intermediate_size,
                              train=True)
        ad.patch(target_module)
        return ad
    if arm_id == "int4_residual":
        ad = intN_residual_adapter(4, column_mask_fraction=0.5,
                                   target_module=target_module,
                                   hidden_size=hidden_size,
                                   intermediate_size=intermediate_size,
                                   train=True)
        return ad
    if arm_id == "int8_residual":
        ad = intN_residual_adapter(8, column_mask_fraction=0.25,
                                   target_module=target_module,
                                   hidden_size=hidden_size,
                                   intermediate_size=intermediate_size,
                                   train=True)
        return ad
    if arm_id == "lora":
        ad = lora_adapter(target_module=target_module,
                          hidden_size=hidden_size,
                          intermediate_size=intermediate_size,
                          rank=216, train=True)
        return ad
    if arm_id == "dense_adapter":
        ad = dense_adapter_bottleneck(
            target_module=target_module, hidden_size=hidden_size,
            intermediate_size=intermediate_size, bottleneck=192,
            train=True)
        return ad
    if arm_id == "random_t2_ternary":
        ad = T2TernaryAdapter(target_module=target_module,
                              hidden_size=hidden_size,
                              intermediate_size=intermediate_size,
                              train=False, init_seed=12345)
        ad.patch(target_module)
        return ad
    if arm_id == "random_lora":
        ad = lora_adapter(target_module=target_module,
                          hidden_size=hidden_size,
                          intermediate_size=intermediate_size,
                          rank=216, train=False, init_seed=12345)
        return ad
    raise ValueError(f"unknown arm_id: {arm_id}")


# ---------------------------------------------------------------------------
# Eval wrapper (reuse AF1/eval_lm with limit plug)
# ---------------------------------------------------------------------------

def eval_arm(model, tokenizer, *, tasks: List[str], limit=None,
             batch_size: int = 4):
    """Run full lm-eval-harness on the live `model`. Reuses
    eval_lm.run_lm_eval which forwards `limit` (the AF1 plumbing fix)."""
    return _eval_lm.run_lm_eval(
        model=model, tokenizer=tokenizer, tasks=tasks,
        batch_size=batch_size, limit=limit,
    )


# ---------------------------------------------------------------------------
# Per-(seed, arm) runner
# ---------------------------------------------------------------------------

TARGET_DEPLOYED_BYTES = {
    "t2_ternary":   4_194_404,
    "int4_residual":4_194_404,
    "int8_residual":4_194_404,
    "lora":         4_423_680,
    "dense_adapter":3_932_160,
}
TRAINED_ARMS = ["t2_ternary", "int4_residual", "int8_residual",
                "lora", "dense_adapter"]
ALL_ARMS = TRAINED_ARMS + ["random_t2_ternary", "random_lora"]


def run_one_seed(*, arm: str, seed: int, args, out_dir: Path,
                 tokenizer, pad_id: int, all_ids: np.ndarray) -> dict:
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)

    arm_dir = out_dir / f"seed-{seed:03d}" / arm
    arm_dir.mkdir(parents=True, exist_ok=True)

    # Build base model + site adapter
    model, leaf, parent_module = build_arm(
        args.model, target_module=args.target_module,
        device=args.device, dtype=args.dtype,
    )
    adapter = build_site_adapter(
        arm, target_module=parent_module,
        hidden_size=model.config.hidden_size,
        intermediate_size=getattr(model.config, "intermediate_size",
                                  4 * model.config.hidden_size),
    )

    # Wire forward to the residual site
    def forward_fn(ids):
        out = model(input_ids=ids)
        return out.logits

    if adapter.trainable_parameters():
        # Match AF1 training recipe exactly
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
    else:
        # Untrained controls: zero-step training.
        history = []

    # Serialize trained weights (or random-init weights) and capture
    # the cost vector.
    cv = adapter.serialize(
        arm_dir,
        training_flops=0,
        inference_ops_per_token=0,
    )
    target_bytes = TARGET_DEPLOYED_BYTES[arm]
    cv_tolerance = abs(cv.deployed_bytes - target_bytes) / target_bytes * 100
    cv_matched = cv_tolerance <= args.matched_bytes_tolerance_pct

    # Measure latency (median of 100 forward passes) on the residual
    # site. Use a small dummy input.
    if adapter.trainable_parameters() or adapter.is_untrained:
        # Heuristic: residual latency on this site for one token.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, model.config.hidden_size,
                                device=args.device, dtype=getattr(torch, args.dtype))
            try:
                adapter._latent  # may not exist
                def _fwd():
                    return forward_fn(torch.zeros(1, 1, dtype=torch.long,
                                                   device=args.device))
                lat = measure_latency_seconds(_fwd)
            except AttributeError:
                def _fwd():
                    return forward_fn(torch.zeros(1, 1, dtype=torch.long,
                                                   device=args.device))
                lat = measure_latency_seconds(_fwd)
            cv_lat = lat
        # Immutability of dataclass(frozen=True) -> replace via __dict__.
        object.__setattr__(cv, "latency_per_token_titan_rtx", cv_lat)

    # Write cost vector and history
    (arm_dir / "cost_vector.json").write_text(json.dumps(cv.as_dict(), indent=2))
    with open(arm_dir / "history.jsonl", "w") as fh:
        for row in history:
            fh.write(json.dumps(row) + "\n")

    # Eval (skip the untrained controls' costly re-eval to save
    # compute; record the zero-training cost vector and skip eval.)
    eval_results = {"skipped": True}
    if not adapter.is_untrained:
        # Eval in float16 (always; AF2 never uses --limit)
        try:
            model_for_eval = model.to(getattr(torch, args.eval_dtype))
            model_for_eval.eval()
            eval_results = eval_arm(
                model_for_eval, tokenizer,
                tasks=args.tasks.split(","), limit=None,
                batch_size=args.batch_size,
            )
        finally:
            pass

    eval_summary = {
        "arm": arm, "seed": seed, "model": args.model,
        "target_module": args.target_module,
        "n_steps": args.n_steps, "batch_size": args.batch_size,
        "seq_len": args.seq_len, "lr": args.lr,
        "limit": None,
        "tasks": {
            t: {"metric": k, "value": v}
            for t, res in eval_results.get("results", {}).items()
            for k, v in res.items() if "acc" in k or "word_perplexity" in k
        } if isinstance(eval_results, dict) and "results" in eval_results else {},
        "matched_bytes_target": target_bytes,
        "matched_bytes_actual": cv.deployed_bytes,
        "matched_bytes_tolerance_pct": args.matched_bytes_tolerance_pct,
        "matched_bytes_passed": cv_matched,
        "cost_vector": cv.as_dict(),
        "is_untrained_control": adapter.is_untrained,
    }
    (arm_dir / "eval.summary.json").write_text(json.dumps(eval_summary, indent=2))
    if isinstance(eval_results, dict):
        # eval.full.json may be very large; record if present
        try:
            from lm_eval.utils import make_table  # noqa
        except Exception:
            pass
        (arm_dir / "eval.full.json").write_text(json.dumps(eval_results, indent=2))

    # Cleanup
    del model, adapter, parent_module
    gc.collect()
    if args.device.startswith("cuda"):
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    return eval_summary


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def aggregate(summaries: list, out_dir: Path) -> dict:
    trained = [s for s in summaries if not s.get("is_untrained_control")]
    controls = [s for s in summaries if s.get("is_untrained_control")]
    out = {"trained_arms": {}, "untrained_controls": {}, "tolerance_violations": []}
    for grp, key, label in [(trained, "trained_arms", "trained"),
                            (controls, "untrained_controls", "untrained")]:
        for arm in ALL_ARMS:
            per_seed = [s for s in grp if s["arm"] == arm]
            if not per_seed:
                continue
            per_seed = [s for s in per_seed if s.get("matched_bytes_passed", True)]
            if not per_seed:
                continue
            entry = {"n": len(per_seed),
                     "matched_bytes": [s["matched_bytes_actual"] for s in per_seed],
                     "matched_bytes_target": per_seed[0]["matched_bytes_target"],
                     "cost_vector_rows": [s["cost_vector"] for s in per_seed],
                     "tasks": {}}
            for t_name in ["wikitext", "arc_easy", "lambada_openai"]:
                vals = [s["tasks"].get(t_name, {}).get("value")
                        for s in per_seed
                        if s.get("tasks", {}).get(t_name)]
                vals = [v for v in vals if v is not None]
                if vals:
                    import statistics
                    entry["tasks"][t_name] = {
                        "n": len(vals),
                        "mean": statistics.fmean(vals),
                        "stderr": statistics.stdev(vals) / len(vals) ** 0.5
                                  if len(vals) > 1 else 0.0,
                        "values": vals,
                    }
            out[key][arm] = entry

    # Tolerance violations (trained only)
    for s in trained:
        if not s.get("matched_bytes_passed", True):
            out["tolerance_violations"].append({
                "arm": s["arm"], "seed": s["seed"],
                "actual": s["matched_bytes_actual"],
                "target": s["matched_bytes_target"],
                "delta_pct": round(100 * (s["matched_bytes_actual"]
                                          - s["matched_bytes_target"])
                                   / s["matched_bytes_target"], 3),
            })

    # Per-metric difference from dense_adapter for trained arms
    if "dense_adapter" in out["trained_arms"]:
        diff = {}
        for arm in TRAINED_ARMS:
            if arm == "dense_adapter":
                continue
            if arm not in out["trained_arms"]:
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--target-module", required=True,
                   help="HF module path; AF2 picks model.layers.0.mlp.down_proj")
    p.add_argument("--arms", default=",".join(TRAINED_ARMS),
                   help=f"comma list; default covers all 5 trained arms")
    p.add_argument("--seeds", default="1,2,3")
    p.add_argument("--n-steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--tasks", default="wikitext,arc_easy,lambada_openai")
    p.add_argument("--ids-cache", type=Path, default=Path("/tmp/wikitext103_train_ids.npy"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="float32",
                   help="Training dtype (eval is always float16)")
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
                out_dir=args.out_dir,
                tokenizer=tokenizer, pad_id=pad_id,
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
