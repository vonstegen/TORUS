"""EXP-RPM-SYS systems harness.

Measures the 6-dim cost vector (B/F/O/M/L/E) per COST-VECTOR-v1.yaml at
RPM-SYS stage. B and F are reused from existing eval.summary.json;
O, M, L, E are measured here.

Inputs:
- Stage 1.5 D1p seed-001 adapters (sha256-pinned)
- AF2-D site: model.layers.0.mlp.down_proj, threshold=1.0 Gaussian damage

Protocol:
- batch_size=1, seq_len=128 (input prompt)
- generated_tokens=50 per run
- 5 warm-up runs discarded
- 50 timed runs: L via cuda.Event, E via nvidia-smi --query-gpu=power.draw at 100ms
- O via cost_vector.inference_ops_per_token (already computed)
- M via cost_vector.memory_traffic_per_token (already computed)

Output:
- runs/r/EXP-RPM-SYS/<ts>/systems_measurements.json (per arm)
- runs/r/EXP-RPM-SYS/<ts>/per_arm/<arm>/latency_runs.json (50 latencies)
- runs/r/EXP-RPM-SYS/<ts>/per_arm/<arm>/power_samples.csv
- runs/r/EXP-RPM-SYS/<ts>/ARTIFACTS.json (sha256 manifest)
"""

import sys
# Reset triton cache if a prior run set it to None (Python import-system
# caches failed imports as None and that breaks torch's bmm_outer_product
# kernels and our chained af2_storage_tournament -> af1_budget_control ->
# eval_lm import chain). See examples/eval_untrained_arms_v2.py for
# the same fix.
if 'triton' in sys.modules and sys.modules.get('triton') is None:
    del sys.modules['triton']
for k in list(sys.modules.keys()):
    if k.startswith('triton') and sys.modules.get(k) is None:
        del sys.modules[k]

import argparse
import csv
import gc
import json
import os
import statistics
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/andrew-jochl/TORUS")

from examples.af2_storage_tournament import (
    _patch_module_forward, damage_target_module_gaussian,
)
from examples.eval_untrained_arms_v2 import (
    apply_damage_gaussian, load_model, navigate_to_module,
)


GPU_LOCK = 0  # cuda:0; cuda:1 must be idle during timed runs
ARMS = ["t2_ternary", "int4_residual", "int8_residual", "lora",
        "dense_adapter", "random_t2_ternary", "random_lora"]
D1P_TS = "20260824T120113Z"
SITE = "model.layers.0.mlp.down_proj"
SIGMA = 0.20  # D1p threshold=1.0 Gaussian damage
DAMAGE_SEED = 0
BATCH_SIZE = 1
SEQ_LEN = 128
GENERATED_TOKENS = 50
WARMUP_RUNS = 5
TIMED_RUNS = 50

cfg = {k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float, str, list))}
POWER_SAMPLE_HZ = 10


# ----------------------------------------------------------------------------
# Per-arm patching (trained arms share serialization format with random arms)
# ----------------------------------------------------------------------------

def _t2_residual(q_ste, scale_b, scale):
    import torch.nn.functional as F
    def residual(x):
        out = F.linear(x, q_ste * scale_b)
        return out * scale.squeeze(1)
    return residual


def patch_t2_ternary(model, target_module: str, npz_path: Path):
    """T2 ternary: packed uint8 (2 bits/code) + fp16 scale (out,1) + shape."""
    import torch.nn.functional as F
    d = np.load(npz_path)
    shape = tuple(int(x) for x in d["shape"])
    out_features, in_features = shape
    scale = torch.from_numpy(d["scale"]).to(
        device=next(model.parameters()).device, dtype=torch.float16)
    packed = d["packed"]
    n_total = out_features * in_features
    flat = np.zeros(n_total, dtype=np.int8)
    for i in range(4):
        flat[i::4] = (packed >> (2 * i)) & 0x3
    flat = flat[:n_total]
    coded = (flat.astype(np.int8) - 1).astype(np.float32)  # 0,1,2 -> -1,0,1
    q_ste = torch.from_numpy(coded).reshape(shape).to(
        device=next(model.parameters()).device, dtype=torch.float16)
    scale_b = scale.expand(out_features, in_features)

    parent = navigate_to_module(model, target_module)

    def residual(x):
        out = F.linear(x, q_ste * scale_b)
        return out * scale.squeeze(1)
    _patch_module_forward(parent, residual)


def patch_intN_residual(model, target_module: str, npz_path: Path,
                         N_bits: int, column_mask_fraction: float,
                         packed_cols: int):
    """int4/int8 residual: packed/codes uint8 + fp16 scale (kept cols)."""
    import torch.nn.functional as F
    d = np.load(npz_path)
    scale = torch.from_numpy(d["scale"]).to(
        device=next(model.parameters()).device, dtype=torch.float16)
    if N_bits <= 4:
        packed = d["packed"]
        ub_masked = np.zeros(packed.size * 2, dtype=np.uint8)
        ub_masked[0::2] = packed & 0xF
        ub_masked[1::2] = (packed >> 4) & 0xF
    else:
        ub_masked = d["codes"]

    n_total = scale.shape[0]
    in_features = ub_masked.size // n_total
    out_features = n_total
    levels = (1 << (N_bits - 1)) - 1
    step = 1.0 / levels

    # Reconstruct full out x in tensor (zero where mask=False).
    full = torch.zeros(out_features, in_features, device=scale.device,
                       dtype=torch.float16)
    q_int = torch.from_numpy(ub_masked.astype(np.int64)).to(
        device=scale.device, dtype=torch.float16)
    q_int_centered = q_int - levels  # back to [-levels, +levels]
    q_real = q_int_centered * (scale / levels)  # (n_total, 1) * (n_total, 1) -> broadcast
    # Place into the first `n_total` rows (the kept columns).
    # Actually, looking at serialize, ub_masked has size (n_total * in_features)
    # which equals kept_count * in_features; scale is (kept_count, 1).
    # Reshape q_real to (kept_count, in_features), then place into full[:kept_count, :].
    kept_count = n_total // in_features if (n_total * in_features) > 0 else 0
    # Actually: n_total = scale.shape[0] = kept_count; in_features = ub_masked.size // kept_count
    in_features = ub_masked.size // n_total
    kept_count = n_total
    q_real_2d = q_real.reshape(kept_count, in_features)
    full[:kept_count, :] = q_real_2d
    # Keep mask: only first `kept_count` rows are nonzero.
    keep_mask = torch.zeros(out_features, dtype=torch.bool, device=scale.device)
    keep_mask[:kept_count] = True

    parent = navigate_to_module(model, target_module)

    def residual(x):
        masked = full * keep_mask.unsqueeze(1).to(full.dtype)
        return F.linear(x, masked)
    _patch_module_forward(parent, residual)


def patch_dense_adapter(model, target_module: str, npz_path: Path):
    """dense_adapter: W_down (bottleneck, in) + W_up (out, bottleneck)."""
    import torch.nn.functional as F
    d = np.load(npz_path)
    W_up = torch.from_numpy(d["W_up"]).to(
        device=next(model.parameters()).device, dtype=torch.float16)
    W_down = torch.from_numpy(d["W_down"]).to(
        device=next(model.parameters()).device, dtype=torch.float16)

    parent = navigate_to_module(model, target_module)

    def residual(x):
        hidden = F.linear(x, W_down)
        return F.linear(hidden, W_up)
    _patch_module_forward(parent, residual)


def patch_lora(model, target_module: str, npz_path: Path):
    """LoRA: W_down (rank, in) + W_up (out, rank). Same as random_lora."""
    import torch.nn.functional as F
    d = np.load(npz_path)
    W_up = torch.from_numpy(d["W_up"]).to(
        device=next(model.parameters()).device, dtype=torch.float16)
    W_down = torch.from_numpy(d["W_down"]).to(
        device=next(model.parameters()).device, dtype=torch.float16)

    parent = navigate_to_module(model, target_module)

    def residual(x):
        hidden = F.linear(x, W_down)
        return F.linear(hidden, W_up)
    _patch_module_forward(parent, residual)


# Map arm -> patcher + kwargs
ARM_PATCHERS = {
    "t2_ternary": (patch_t2_ternary, {}),
    "int4_residual": (patch_intN_residual,
                      {"N_bits": 4, "column_mask_fraction": 0.5,
                       "packed_cols": 1024}),
    "int8_residual": (patch_intN_residual,
                      {"N_bits": 8, "column_mask_fraction": 0.25,
                       "packed_cols": 512}),
    "lora": (patch_lora, {}),
    "dense_adapter": (patch_dense_adapter, {}),
    "random_t2_ternary": (patch_t2_ternary, {}),
    "random_lora": (patch_lora, {}),
}


def load_d1p_adapter(arm: str, run_root: Path) -> dict:
    """Load Stage 1.5 D1p seed-001 adapter + cost_vector."""
    arm_dir = run_root / "af2d" / "seed-001" / arm
    cv = json.loads((arm_dir / "cost_vector.json").read_text())
    es = json.loads((arm_dir / "eval.summary.json").read_text())
    npz = arm_dir / "adapter.npz"
    return {
        "arm": arm,
        "npz_path": str(npz),
        "cost_vector": cv,
        "eval_summary": es,
        "sha256": subprocess.run(
            ["sha256sum", str(npz)], capture_output=True, text=True,
        ).stdout.split()[0],
    }


def build_patched_model(model_name: str, arm: str, npz_path: str,
                        target_module: str, sigma: float,
                        damage_seed: int):
    """Build model with damage + adapter patch; return (model, tokenizer)."""
    model, tokenizer = load_model(model_name, device="cuda")
    apply_damage_gaussian(model, target_module, sigma, seed=damage_seed)
    patcher, kwargs = ARM_PATCHERS[arm]
    patcher(model, target_module, Path(npz_path), **kwargs)
    return model, tokenizer


def power_sampler(device: int, csv_path: Path, stop_flag: list):
    """Sample nvidia-smi power.draw at POWER_SAMPLE_HZ; write CSV."""
    proc = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=power.draw",
         "--format=csv,noheader,nounits",
         f"--id={device}"],
        stdout=subprocess.PIPE, text=True, bufsize=1,
    )
    samples = []

    def reader():
        for line in proc.stdout:
            if stop_flag[0]:
                break
            line = line.strip()
            try:
                samples.append({"t_wall": time.time(), "w": float(line)})
            except ValueError:
                pass

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    try:
        yield samples
    finally:
        stop_flag[0] = True
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
        t.join(timeout=2)
        with csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_wall", "power_w"])
            for s in samples:
                w.writerow([s["t_wall"], s["w"]])


def measure_arm(arm: str, model_name: str, run_root: Path, out_dir: Path):
    """Measure L, E for one arm at D1p AF2-D."""
    info = load_d1p_adapter(arm, run_root)
    sigma = cfg["SIGMA"]
    print(f"[{arm}] loaded: deployed_bytes={info['cost_vector']['deployed_bytes']} "
          f"sha256={info['sha256'][:12]}...", flush=True)

    # Load + patch
    model, tokenizer = build_patched_model(
        model_name, arm, info["npz_path"], SITE, sigma, DAMAGE_SEED)
    torch.cuda.set_device(GPU_LOCK)

    # Build input (deterministic)
    torch.manual_seed(0)
    input_ids = torch.randint(
        100, 30000, (cfg["BATCH_SIZE"], cfg["SEQ_LEN"]), device="cuda")
    attention_mask = torch.ones_like(input_ids)

    # Warm-up
    for _ in range(cfg["WARMUP_RUNS"]):
        with torch.no_grad():
            _ = model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_new_tokens=cfg["GENERATED_TOKENS"], do_sample=False,
                num_beams=1, pad_token_id=tokenizer.eos_token_id or 0,
            )
    torch.cuda.synchronize()

    # Timed runs: L
    latencies_ms = []
    for _ in range(cfg["TIMED_RUNS"]):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            _ = model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_new_tokens=cfg["GENERATED_TOKENS"], do_sample=False,
                num_beams=1, pad_token_id=tokenizer.eos_token_id or 0,
            )
        end.record()
        torch.cuda.synchronize()
        total_ms = start.elapsed_time(end)
        per_token_ms = total_ms / cfg["GENERATED_TOKENS"]
        latencies_ms.append(per_token_ms)

    # Timed run: E (power integration)
    power_csv = out_dir / "power_samples.csv"
    stop = [False]
    with power_sampler(GPU_LOCK, power_csv, stop), torch.no_grad():
        torch.cuda.synchronize()
        _ = model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_new_tokens=cfg["GENERATED_TOKENS"], do_sample=False,
            num_beams=1, pad_token_id=tokenizer.eos_token_id or 0,
        )
        torch.cuda.synchronize()

    # Integrate
    with power_csv.open() as f:
        r = csv.DictReader(f)
        rows = [row for row in r]
    if len(rows) < 2:
        joules = None
        mean_w = None
        joules_per_token = None
    else:
        watts = [float(row["power_w"]) for row in rows]
        t_walls = [float(row["t_wall"]) for row in rows]
        dt = (t_walls[-1] - t_walls[0]) / (len(rows) - 1)
        joules = sum(w * dt for w in watts)
        mean_w = sum(watts) / len(watts)
        joules_per_token = joules / cfg["GENERATED_TOKENS"]

    # Stats
    L_med = statistics.median(latencies_ms)
    L_iqr = statistics.quantiles(latencies_ms, n=4)
    L_iqr_low = L_iqr[0] if len(L_iqr) == 3 else None
    L_iqr_high = L_iqr[2] if len(L_iqr) == 3 else None
    L_iqr_width = (L_iqr_high - L_iqr_low) if L_iqr_low is not None else None

    O = info["cost_vector"]["inference_ops_per_token"]
    M = info["cost_vector"]["memory_traffic_per_token"]

    result = {
        "arm": arm,
        "site": SITE,
        "damage_regime": "D1p",
        "damage_sigma": sigma,
        "seed": 1,
        "inference_protocol": {
            "batch_size": cfg["BATCH_SIZE"], "seq_len": cfg["SEQ_LEN"],
            "generated_tokens": cfg["GENERATED_TOKENS"],
            "warmup_runs": cfg["WARMUP_RUNS"], "timed_runs": cfg["TIMED_RUNS"],
            "sampler": "argmax", "greedy": True},
        "B_deployed_bytes": info["cost_vector"]["deployed_bytes"],
        "F_training_flops": info["cost_vector"]["training_flops"],
        "O_inference_ops_per_token": O,
        "M_memory_traffic_per_token": M,
        "L_latency_per_token_ms": {
            "median": L_med,
            "iqr_low": L_iqr_low,
            "iqr_high": L_iqr_high,
            "iqr_width": L_iqr_width,
            "all_runs_ms": latencies_ms,
        },
        "E_joules_per_token": {
            "value": joules_per_token,
            "mean_w": mean_w,
            "n_samples": len(rows),
            "integration_method": "rectangular",
            "sampling_rate_hz": POWER_SAMPLE_HZ,
            "device": GPU_LOCK,
        },
        "adapter_sha256": info["sha256"],
        "capability_reused": info["eval_summary"].get("tasks"),
    }

    (out_dir / "latency_runs.json").write_text(json.dumps({
        "arm": arm, "latencies_ms": latencies_ms,
    }, indent=2))
    (out_dir / "systems_measurement.json").write_text(json.dumps(result, indent=2))

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--model", default="allenai/OLMo-1B-0724-hf")
    p.add_argument("--run-root",
                   default="/home/andrew-jochl/TORUS/runs/r/EXP-RPM-D1p/" + D1P_TS)
    p.add_argument("--out-root",
                   default="/home/andrew-jochl/TORUS/runs/r/EXP-RPM-SYS/" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    p.add_argument("--timed-runs", type=int, default=TIMED_RUNS)
    p.add_argument("--warmup-runs", type=int, default=WARMUP_RUNS)
    args = p.parse_args()

    cfg["TIMED_RUNS"] = args.timed_runs
    cfg["WARMUP_RUNS"] = args.warmup_runs

    run_root = Path(args.run_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    arms = args.arms.split(",")
    summary = {"experiment": "EXP-RPM-SYS", "arms": {}}
    for arm in arms:
        arm_dir = out_root / "per_arm" / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        print(f"=== Measuring {arm} ===", flush=True)
        try:
            r = measure_arm(arm, args.model, run_root, arm_dir)
            summary["arms"][arm] = r
        except Exception as e:
            import traceback
            traceback.print_exc()
            summary["arms"][arm] = {"error": str(e)}
        # Save intermediate summary
        (out_root / "systems_measurements.json").write_text(
            json.dumps(summary, indent=2))

    # Build ARTIFACTS.json
    artifacts = {"experiment": "EXP-RPM-SYS",
                 "ts": out_root.name,
                 "items": []}
    for arm, r in summary["arms"].items():
        if "error" in r:
            continue
        artifacts["items"].append({
            "arm": arm,
            "adapter_npz_sha256": r["adapter_sha256"],
            "systems_measurement": str(out_root / "per_arm" / arm / "systems_measurement.json"),
            "latency_runs": str(out_root / "per_arm" / arm / "latency_runs.json"),
            "power_samples": str(out_root / "per_arm" / arm / "power_samples.csv"),
        })
    (out_root / "ARTIFACTS.json").write_text(json.dumps(artifacts, indent=2))
    print(f"DONE. Summary at {out_root / 'systems_measurements.json'}")


if __name__ == "__main__":
    main()