"""EXP-A-H1 driver: small-model native Hadamard ternary training.

Preregistered 2026-08-29 (manifest: research/track-a5-hadamard/EXP-A-H1/manifest.yaml).

Modes:
  --mode prep          Build GPT-2(OPT)-tokenized openwebtext train/test
                       caches from parquet shards 0-3. Refuses to
                       overwrite; records shard names + sha256 + token
                       counts in a manifest.
  --mode parity        Build the model for one arm, compute the step-0
                       CE loss on the first batch, write parity.json.
                       The launcher runs this for BOTH arms before any
                       training and passes the gate only if
                       |loss_control - loss_hadamard| <= 0.1 nats.
  --mode train         Train one arm (--arm control|hadamard) from
                       random init on the shared token stream.
  --mode materialize  Save the runtime state dict, materialize W_eff
                       into a stock HF checkpoint, and record the
                       ternary codes/scales (deployed-form record).
                       The launcher then runs the frozen eval ladder
                       (eval_lm.py --mode baseline) per arm.

Paired design: both arms initialize from the same random W0 (same
torch seed), consume the identical token stream from byte 0, and run
an identical AdamW/cosine schedule. The hadamard arm differs ONLY by
the fixed block-64 Sylvester rotations (W_eff = R_out Q R_in, ternary
Q trained in the rotated domain) and by the live loss-gap self-check.

Deployed representation of both arms: packed ternary codes (2
bits/entry) + one fp16 per-tensor scale per linear. The rotation
matrices are fixed structural constants derived from the dims (zero
per-weight storage).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

# Triton bypass — same intent as eval_lm.py / distill_run.py.
try:
    import triton  # noqa: F401
except ModuleNotFoundError:
    sys.modules["triton"] = None

import numpy as np
import torch
import torch.nn.functional as F

# ---- frozen constants (manifest) -------------------------------------------
MODEL_ID = "facebook/opt-125m"
OWT_DATASET = "Skylion007/openwebtext"
OWT_N_SHARDS = 4                # ~50M tokens each (GPT-2 tokenizer)
TEST_FRACTION = 0.01
SEED = 7
SEQ_LEN = 512
BATCH = 32                      # 16,384 tokens / step
TOTAL_STEPS = 31_250            # 500M tokens
LR = 2e-3
WARMUP_STEPS = 2_000
MIN_LR_RATIO = 0.1
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
BLOCK = 64                      # Sylvester H block size (2^6)
COND_SAMPLE_STEPS = [1_000, 5_000, 15_000, 25_000, TOTAL_STEPS]
FLIP_WINDOW = 500
LIVE_GAP_RATIO = 1.15
LIVE_GAP_WINDOW = 2_000
PARITY_TOLERANCE = 0.1          # nats
LINEAR_NAMES = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]

CACHE_TRAIN = "/tmp/ah1_owt_train_ids.npy"
CACHE_TEST = "/tmp/ah1_owt_test_ids.npy"
PREP_MANIFEST = "/tmp/ah1_data_prep.json"


# ---- rotation math ----------------------------------------------------------
def sylvester_hadamard(n: int) -> torch.Tensor:
    """Sylvester-constructed Hadamard matrix H_n (n a power of 2).

    H_n is symmetric, orthogonal, H_n H_n = n I, normalized to H_n
    H_n = I (columns scaled by 1/sqrt(n))."""
    if n & (n - 1):
        raise ValueError(f"n must be a power of 2, got {n}")
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return (h / math.sqrt(n)).to(torch.float32)


def rotate_blocks(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Apply block-diagonal H to the last axis of x (x @ blockdiag(H))."""
    block = h.shape[0]
    d = x.shape[-1]
    if d % block:
        raise ValueError(f"last dim {d} not divisible by block {block}")
    n = d // block
    return (x.reshape(*x.shape[:-1], n, block) @ h).reshape_as(x)

def materialize_w_eff(q: torch.Tensor, scale: float, h: torch.Tensor,
                      rotate_in: bool, rotate_out: bool) -> torch.Tensor:
    """W_eff = R_out Q R_in (H symmetric, so R = blockdiag(H))."""
    block = h.shape[0]
    if rotate_in and q.shape[1] % block:
        raise ValueError("in dim not divisible by rotation block")
    if rotate_out and q.shape[0] % block:
        raise ValueError("out dim not divisible by rotation block")
    w = q * scale
    if rotate_in:
        w = w @ torch.block_diag(*([h] * (w.shape[1] // block)))
    if rotate_out:
        w = torch.block_diag(*([h] * (w.shape[0] // block))) @ w
    return w


# ---- ternary quantization (per-tensor absmean scale, STE) -------------------
def ternary_quantize(w: torch.Tensor) -> tuple[torch.Tensor, float]:
    scale = float(w.detach().abs().mean().clamp(min=1e-8))
    q = (w / scale).round().clamp(-1.0, 1.0)
    return q, scale


# ---- arm modules ------------------------------------------------------------
class TernaryLinear(torch.nn.Module):
    """Control arm: plain ternary STE linear (per-tensor absmean scale)."""

    def __init__(self, weight0: torch.Tensor, bias0: torch.Tensor | None):
        super().__init__()
        self.weight_latent = torch.nn.Parameter(weight0.detach().clone())
        if bias0 is not None:
            self.bias = torch.nn.Parameter(bias0.detach().clone())
        else:
            self.register_parameter("bias", None)

    def effective_weight(self) -> torch.Tensor:
        q, scale = ternary_quantize(self.weight_latent)
        return q * scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.effective_weight()
        return F.linear(x, w, self.bias)


class RotatedTernaryLinear(torch.nn.Module):
    """Hadamard arm: W_eff = R_out Q R_in with ternary Q.

    weight_latent lives in the rotated domain (its init is
    R_out W0 R_in so the step-0 effective function matches the
    control arm up to ternary-quantization noise in two bases)."""

    def __init__(self, weight0: torch.Tensor, bias0: torch.Tensor | None,
                 h: torch.Tensor, rotate_out: bool = True):
        super().__init__()
        block = h.shape[0]
        d_out, d_in = weight0.shape
        if d_in % block or (rotate_out and d_out % block):
            raise ValueError("dims not divisible by rotation block")
        self.h = h
        self.rotate_out = rotate_out
        latent0 = weight0.detach().clone().float()
        # R_out W0 R_in  (R_in over d_in, R_out over d_out; H symmetric)
        if d_in >= block:
            latent0 = latent0 @ torch.block_diag(
                *([h] * (d_in // block)))
        if rotate_out:
            latent0 = torch.block_diag(*([h] * (d_out // block))) @ latent0
        self.weight_latent = torch.nn.Parameter(latent0)
        if bias0 is not None:
            self.bias = torch.nn.Parameter(bias0.detach().clone())
        else:
            self.register_parameter("bias", None)

    def effective_weight(self) -> torch.Tensor:
        q, scale = ternary_quantize(self.weight_latent)
        return materialize_w_eff(q, scale, self.h, True, self.rotate_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, scale = ternary_quantize(self.weight_latent)
        xr = rotate_blocks(x, self.h)
        y = F.linear(xr, q * scale, None)
        if self.rotate_out:
            y = rotate_blocks(y, self.h)
        return y if self.bias is None else y + self.bias


# ---- model construction -----------------------------------------------------
def build_arm(arm: str, device: str) -> tuple:
    """Build OPT-125M from random init with the arm's linear modules.

    Returns (model, h, linear_modules, tokenizer)."""
    from transformers import AutoTokenizer, OPTConfig, OPTForCausalLM

    torch.manual_seed(SEED)
    config = OPTConfig.from_pretrained(MODEL_ID)
    model = OPTForCausalLM(config)
    h = sylvester_hadamard(BLOCK)
    linear_modules: list = []

    def swap(parent: torch.nn.Module, module: torch.nn.Module,
             name: str, rotate_out: bool = True):
        w0 = module.weight.data.clone().float()
        bias0 = module.bias.data if module.bias is not None else None
        if arm == "hadamard":
            new = RotatedTernaryLinear(w0, bias0, h, rotate_out=rotate_out)
        else:
            new = TernaryLinear(w0, bias0)
        setattr(parent, name, new)
        linear_modules.append(new)

    decoder = model.model.decoder
    for layer in decoder.layers:
        attn = layer.self_attn
        for name in ["q_proj", "k_proj", "v_proj", "out_proj"]:
            swap(attn, getattr(attn, name), name)
        swap(layer, layer.fc1, "fc1")
        swap(layer, layer.fc2, "fc2")
    # lm_head: input rotated, output (vocab) NOT rotated.
    swap(model, model.lm_head, "lm_head", rotate_out=False)

    # The original tie (embed_tokens == lm_head.weight) is broken by the
    # swap; the lm_head latent was initialized from the same W0 as the
    # control arm's, so the paired design holds. Embeddings stay fp16
    # (fp32 params) and train in BOTH arms identically.
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    return model, h, linear_modules, tokenizer


# ---- data -------------------------------------------------------------------
def make_stream(cache_path: str) -> np.ndarray:
    return np.load(cache_path, mmap_mode="r")


class TokenWindowIter:
    """Deterministic non-overlapping windows from a token cache.

    Both arms get byte-identical streams because iteration always
    starts at offset 0 and advances by exactly BATCH*SEQ_LEN tokens."""

    def __init__(self, stream: np.ndarray, n_windows: int | None = None):
        self.stream = stream
        self.window_tokens = BATCH * SEQ_LEN
        self.n = (len(stream) - 1) // self.window_tokens if n_windows is None \
            else n_windows
        self.pos = 0

    def __len__(self) -> int:
        return self.n

    def __iter__(self):
        self.pos = 0
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Wrap cyclically: budgets larger than one cache pass are
        # satisfied by deterministic re-reading from offset 0
        # (identical for both arms). base excludes the final partial
        # window.
        base = (self.pos % self.n) * self.window_tokens
        x = np.asarray(self.stream[base:base + self.window_tokens + 1],
                       dtype=np.int64)
        ids = torch.from_numpy(x[:-1].copy()).reshape(BATCH, SEQ_LEN)
        labels = torch.from_numpy(x[1:].copy()).reshape(BATCH, SEQ_LEN)
        self.pos += 1
        return ids, labels


def lr_at(step: int) -> float:
    if step < WARMUP_STEPS:
        return LR * (step + 1) / WARMUP_STEPS
    t = (step - WARMUP_STEPS) / max(1, TOTAL_STEPS - WARMUP_STEPS)
    return MIN_LR_RATIO * LR + (1 - MIN_LR_RATIO) * LR * 0.5 * (
        1 + math.cos(math.pi * t))


# ---- data prep --------------------------------------------------------------
def prep_data(out_manifest: str = PREP_MANIFEST) -> None:
    from huggingface_hub import hf_hub_download, list_repo_files
    from transformers import AutoTokenizer
    if Path(CACHE_TRAIN).exists() or Path(CACHE_TEST).exists():
        raise SystemExit(f"[ah1-prep] caches exist; refusing to overwrite "
                         f"({CACHE_TRAIN}, {CACHE_TEST})")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    eot = tokenizer.eos_token_id
    record = {"shards": [], "total_train_tokens": 0, "test_tokens": 0}
    shard_names = sorted(
        f for f in list_repo_files(OWT_DATASET, repo_type="dataset")
        if f.endswith(".parquet") and "train" in f)
    if len(shard_names) < OWT_N_SHARDS:
        raise SystemExit(f"[ah1-prep] only {len(shard_names)} train shards")
    parts = []
    for name in shard_names[:OWT_N_SHARDS]:
        path = hf_hub_download(OWT_DATASET, name, repo_type="dataset")
        record["shards"].append({"name": name, "sha256": _sha256_file(path)})
        import pyarrow.parquet as pq
        table = pq.read_table(path)
        texts = [t for t in table.column("text").to_pylist() if t]
        # Multithreaded batch tokenization (tokenizers lib): a few
        # minutes per shard instead of hours.
        segs = []
        for i in range(0, len(texts), 1000):
            batch = tokenizer(texts[i:i + 1000],
                              add_special_tokens=False)["input_ids"]
            segs.extend(batch)
        ids = []
        for seg in segs:
            ids.append(np.asarray(seg, dtype=np.int64))
            ids.append(np.asarray([eot], dtype=np.int64))
        parts.append(np.concatenate(ids))
    stream = np.concatenate(parts)
    split = int(len(stream) * (1 - TEST_FRACTION))
    train, test = stream[:split], stream[split:]
    np.save(CACHE_TRAIN, train)
    np.save(CACHE_TEST, test)
    record["total_train_tokens"] = int(len(train))
    record["test_tokens"] = int(len(test))
    record["cache_train"] = CACHE_TRAIN
    record["cache_test"] = CACHE_TEST
    record["cache_train_sha256"] = _sha256_file(CACHE_TRAIN)
    record["cache_test_sha256"] = _sha256_file(CACHE_TEST)
    Path(out_manifest).write_text(json.dumps(record, indent=2))
    print(f"[ah1-prep] train {len(train)} tok, test {len(test)} tok; "
          f"manifest {out_manifest}", flush=True)


def _sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ---- training ---------------------------------------------------------------
def train(arm: str, device: str, run_dir: str,
          max_steps: int = TOTAL_STEPS, quick: int | None = None) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    model, h, linear_modules, _ = build_arm(arm, device)
    stream = make_stream(CACHE_TRAIN)
    it = TokenWindowIter(stream)
    if len(it) == 0:
        raise SystemExit("[ah1] token cache too small for one window")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  betas=(0.9, 0.95),
                                  weight_decay=WEIGHT_DECAY)
    history_path = run_dir / "history.jsonl"
    history = open(history_path, "a")
    conditioning: dict = {}
    flip_ref: dict[int, torch.Tensor] = {}
    step = 0
    t0 = time.time()
    window_loss = 0.0
    last_flip_rate = None
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        for ids, labels in it:
            if step >= max_steps:
                break
            ids, labels = ids.to(device), labels.to(device)
            for g in optimizer.param_groups:
                g["lr"] = lr_at(min(step, TOTAL_STEPS - 1))
            optimizer.zero_grad(set_to_none=True)
            logits = model(ids).logits
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().reshape(-1, logits.shape[-1]),
                labels[:, 1:].contiguous().reshape(-1))
            if not torch.isfinite(loss):
                _abort(run_dir, "nan_loss", arm, step)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            if step + 1 in COND_SAMPLE_STEPS:
                conditioning[f"step_{step + 1}"] = {
                    "g2_over_g1_mean": float(np.mean([
                        float((m.weight_latent.grad.norm(2)
                               / m.weight_latent.grad.norm(1).clamp(min=1e-12))
                              .item()) for m in linear_modules])),
                    "per_linear": {
                        str(i): float((m.weight_latent.grad.norm(2)
                                       / m.weight_latent.grad.norm(1)
                                       .clamp(min=1e-12)).item())
                        for i, m in enumerate(linear_modules)},
                }
            optimizer.step()
            window_loss += float(loss.detach())
            step += 1
            if step % 100 == 0:
                rate = None
                if step % FLIP_WINDOW == 0:
                    codes = {i: (m.weight_latent.detach() / float(
                        m.weight_latent.detach().abs().mean().clamp(
                            min=1e-8))).round().clamp(-1, 1).flatten()
                        for i, m in enumerate(linear_modules)}
                    if flip_ref:
                        total = sum(c.numel() for c in codes.values())
                        flipped = sum(int((codes[i] != flip_ref[i]).sum())
                                      for i in codes)
                        rate = flipped / total
                    flip_ref = codes
                    last_flip_rate = rate
                tokens = step * BATCH * SEQ_LEN
                el = time.time() - t0
                rec = {"step": step, "loss": window_loss / 100,
                       "tokens": tokens, "elapsed_s": round(el, 2),
                       "tok_per_s": round(tokens / el, 1)}
                if rate is not None:
                    rec["flip_rate"] = round(rate, 6)
                history.write(json.dumps(rec) + "\n")
                history.flush()
                window_loss = 0.0
            if arm == "hadamard" and step % 1000 == 0:
                gap = _live_loss_gap(run_dir.parent, step)
                if gap is not None and gap > LIVE_GAP_RATIO:
                    _abort(run_dir, "loss_gap", arm, step, gap=gap)
            if quick and step >= quick:
                break
    history.close()
    elapsed = time.time() - t0
    tokens = step * BATCH * SEQ_LEN
    summary = {
        "arm": arm,
        "steps": step,
        "tokens": tokens,
        "final_loss_window": _tail_loss(run_dir, 500),
        "throughput_tok_s": round(tokens / elapsed, 1),
        "wall_s": round(elapsed, 1),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated(device) / 2**20),
        "training_flops": int(6 * sum(p.numel() for p in model.parameters())
                              * tokens),
        "conditioning": conditioning,
        "last_flip_rate": last_flip_rate,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    _write_cost_vector(model, linear_modules, run_dir)
    print(f"[ah1] {arm} done: {json.dumps(summary, indent=2)}", flush=True)


def _abort(run_dir: Path, reason: str, arm: str, step: int, gap=None):
    rec = {"event": "abort", "reason": reason, "arm": arm, "step": step}
    if gap is not None:
        rec["loss_gap_ratio"] = round(gap, 4)
    (run_dir / "abort.json").write_text(json.dumps(rec, indent=2))
    print(f"[ah1] ABORT {arm}: {rec}", flush=True)
    raise SystemExit(1)


def _live_loss_gap(parent: Path, step: int) -> float | None:
    ctrl_hist = parent / "control" / "history.jsonl"
    had_hist = parent / "hadamard" / "history.jsonl"
    if not ctrl_hist.exists() or not had_hist.exists():
        return None
    ctrl = _rolling(ctrl_hist, step, LIVE_GAP_WINDOW)
    had = _rolling(had_hist, step, LIVE_GAP_WINDOW)
    if ctrl is None or had is None:
        return None
    return had / ctrl


def _rolling(path: Path, step: int, window: int) -> float | None:
    losses = []
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if step - window < rec["step"] <= step:
                losses.append(rec["loss"])
    if len(losses) < window // 100 // 2:
        return None
    return float(np.mean(losses))


def _tail_loss(run_dir: Path, window: int) -> float | None:
    return _rolling(run_dir / "history.jsonl", 10**12, window)


def _write_cost_vector(model, linear_modules, run_dir: Path):
    n_codes = sum(m.weight_latent.numel() for m in linear_modules)
    n_linears = len(linear_modules)
    code_bytes = n_codes * 2 // 8  # 2 bits/entry packed
    scale_bytes = n_linears * 2   # fp16 per-tensor scale
    meta_bytes = n_linears * 8    # headers
    cost = {
        "n_ternary_entries": n_codes,
        "code_bytes_packed": code_bytes,
        "scale_bytes": scale_bytes,
        "metadata_bytes": meta_bytes,
        "deployed_bytes": code_bytes + scale_bytes + meta_bytes,
        "fp16_linears_bytes": n_codes * 2,
        "rotation_storage_bytes": 0,  # fixed structural constants
        "note": "identical formula for both arms; rotations are fixed "
                "constants derived from dims (zero per-weight storage). "
                "Embeddings (fp16, trained) excluded: identical in both "
                "arms. Never quoted as 1.58 bits/weight: physical packing "
                "is 2 bits/entry + scales + metadata.",
    }
    (run_dir / "cost_vector.json").write_text(json.dumps(cost, indent=2))


def save_state_and_materialize(arm: str, run_dir: str) -> None:
    """Save runtime state dict + materialized HF checkpoint."""
    from transformers import OPTConfig, OPTForCausalLM

    run_dir = Path(run_dir)
    model, h, linear_modules, tokenizer = build_arm(arm, "cpu")
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()},
               run_dir / f"{arm}_state.pt")
    config = OPTConfig.from_pretrained(MODEL_ID)
    stock = OPTForCausalLM(config)
    sd = dict(stock.state_dict())
    # trained embeddings/norms/pos/bias: copy from trained model
    for k, v in model.state_dict().items():
        if any(k.startswith(p) for p in
               ["model.decoder.embed_", "model.decoder.layers.",
                "model.decoder.final_layer_norm", "model.decoder.project_"]):
            if "weight_latent" not in k and "h" not in k:
                sd[k] = v.detach().clone()
    # W_eff per swapped linear, in linear_modules order
    path_names = []
    for li in range(len(model.model.decoder.layers)):
        base = f"model.decoder.layers.{li}"
        path_names += [f"{base}.self_attn.{n}.weight"
                       for n in ["q_proj", "k_proj", "v_proj", "out_proj"]]
        path_names += [f"{base}.fc1.weight", f"{base}.fc2.weight"]
    path_names.append("lm_head.weight")
    assert len(path_names) == len(linear_modules)
    for pname, m in zip(path_names, linear_modules, strict=True):
        sd[pname] = m.effective_weight().detach().clone()
    stock.load_state_dict(sd)
    out_dir = run_dir / "final_hf"
    stock.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    # ternary codes + scales for the deployed-form record
    for pname, m in zip(path_names, linear_modules, strict=True):
        q, scale = ternary_quantize(m.weight_latent.detach())
        meta = {"scale": float(scale), "shape": list(q.shape)}
        codes_meta = {}
        if (run_dir / "codes_meta.json").exists():
            codes_meta = json.loads(
                (run_dir / "codes_meta.json").read_text())
        codes_meta[pname] = meta
        (run_dir / "codes_meta.json").write_text(
            json.dumps(codes_meta, indent=2))
    print(f"[ah1] {arm}: state + materialized checkpoint saved", flush=True)


# ---- parity gate ------------------------------------------------------------
def parity(arm: str, device: str, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model, _, _, _ = build_arm(arm, device)
    stream = make_stream(CACHE_TRAIN)
    it = iter(TokenWindowIter(stream))
    ids, labels = next(it)
    ids, labels = ids.to(device), labels.to(device)
    with torch.no_grad(), torch.autocast(device_type="cuda",
                                         dtype=torch.float16):
        logits = model(ids).logits
        loss = F.cross_entropy(
            logits[:, :-1].contiguous().reshape(-1, logits.shape[-1]),
            labels[:, 1:].contiguous().reshape(-1))
    rec = {"arm": arm, "step0_loss": float(loss.detach())}
    (out / f"parity_{arm}.json").write_text(json.dumps(rec, indent=2))
    print(f"[ah1] parity {arm}: {rec}", flush=True)


# ---- CLI --------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True,
                   choices=["prep", "parity", "train", "materialize"])
    p.add_argument("--arm", choices=["control", "hadamard"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--run-dir", required=False)
    p.add_argument("--max-steps", type=int, default=TOTAL_STEPS)
    p.add_argument("--quick", type=int, default=None,
                   help="debug: stop after N steps")
    args = p.parse_args()
    if args.mode == "prep":
        prep_data()
    elif args.mode == "parity":
        parity(args.arm, args.device, args.run_dir)
    elif args.mode == "train":
        train(args.arm, args.device, args.run_dir, args.max_steps,
              args.quick)
    elif args.mode == "materialize":
        save_state_and_materialize(args.arm, args.run_dir)


if __name__ == "__main__":
    main()
