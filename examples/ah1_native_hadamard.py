"""EXP-A-H1 driver: small-model native Hadamard ternary training.

Preregistered 2026-08-29 (manifest: research/track-a5-hadamard/EXP-A-H1/manifest.yaml).

Modes:
  --mode prep          Build GPT-2(OPT)-tokenized openwebtext train/test
                       caches from parquet shards (list_repo_files
                       discovery). Refuses to overwrite; records shard
                       names + sha256 + token counts in a manifest.
  --mode parity        Build the model for one arm, compute the step-0
                       CE loss on the first batch, write parity.json.
                       The launcher runs this for BOTH arms before any
                       training and passes the gate only if
                       |loss_control - loss_hadamard| <= 0.1 nats.
  --mode train         Train one arm (--arm control|hadamard) from
                       random init on the shared token stream; saves
                       the runtime state, materializes W_eff into a
                       stock HF checkpoint, and records the
                       materialize cross-check (runtime CE vs
                       materialized CE on the same window).
  --mode materialize   Manual fallback: load the saved runtime state,
                       materialize W_eff into a stock HF checkpoint
                       (used by the launcher only if the in-process
                       path failed).

Paired design: both arms initialize from the same random W0 (same
torch seed), consume the identical token stream from byte 0, and run
an identical AdamW/cosine schedule. The hadamard arm differs ONLY by
the fixed block-64 Sylvester rotations (W_eff = R_out Q R_in, ternary
Q trained in the rotated domain) and by the live loss-gap self-check.

Parameterization (v2, correcting the 2026-08-29 run-1 recipe defect):
each ternary linear stores a NORMALIZED latent (codes near {-1,0,1},
init = W0 / mean|W0|) plus an INDEPENDENTLY LEARNED per-linear scale
gamma (init = mean|W0|). The scale is a free parameter (BitNet-gamma
style): run 1 recomputed scale = mean|latent| every forward, which
feeds dL/dw = scale * g_eff back into |w| growth — a positive
feedback that diverged both arms from loss 8.8 (step 2000) back to
10.4 (step 12500) with code-flip rate ~0. The learned scale breaks
the feedback; scales are excluded from weight decay.

Deployed representation of both arms: packed ternary codes (2
bits/entry) + one fp16 per-linear scale. The rotation matrices are
fixed structural constants derived from the dims (zero per-weight
storage).
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
TOTAL_STEPS = 12_500            # 200M tokens (pre-run amendment: measured
                                # TITAN RTX throughput ~27-38k tok/s incl.
                                # checkpointing; 500M would breach the
                                # frozen 8 GPU-h cap)
LR = 6e-4                       # v2 recipe (run 1 used 2e-3)
WARMUP_STEPS = 2_000
MIN_LR_RATIO = 0.1
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
BLOCK = 64                      # Sylvester H block size (2^6)
COND_SAMPLE_STEPS = [500, 3_000, 6_000, 10_000, TOTAL_STEPS]
FLIP_WINDOW = 500
LIVE_GAP_RATIO = 1.15
LIVE_GAP_WINDOW = 2_000
PARITY_TOLERANCE = 0.1          # nats

CACHE_TRAIN = "/tmp/ah1_owt_train_ids.npy"
CACHE_TEST = "/tmp/ah1_owt_test_ids.npy"
PREP_MANIFEST = "/tmp/ah1_data_prep.json"


# ---- rotation math ----------------------------------------------------------
def sylvester_hadamard(n: int) -> torch.Tensor:
    """Sylvester-constructed Hadamard matrix H_n (n a power of 2).

    H_n is symmetric, orthogonal, normalized to H_n H_n = I."""
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
    h = h.to(w.dtype)
    if rotate_in:
        w = w @ torch.block_diag(*([h] * (w.shape[1] // block)))
    if rotate_out:
        w = torch.block_diag(*([h] * (w.shape[0] // block))) @ w
    return w


# ---- ternary quantization ---------------------------------------------------
def ternary_quantize(w: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Absmean-scale ternary quantize (deployed-form record helper)."""
    scale = float(w.detach().abs().mean().clamp(min=1e-8))
    q = (w / scale).round().clamp(-1.0, 1.0)
    return q, scale


def ternary_codes(w: torch.Tensor) -> torch.Tensor:
    """Codes of a NORMALIZED latent (threshold 0.5 in latent units)."""
    return w.round().clamp(-1.0, 1.0)


# ---- arm modules ------------------------------------------------------------
class TernaryLinear(torch.nn.Module):
    """Control arm: ternary STE linear.

    v2: normalized latent (W0 / mean|W0|) + independently learned
    per-linear scale gamma (init mean|W0|), BitNet-gamma style."""

    def __init__(self, weight0: torch.Tensor, bias0: torch.Tensor | None):
        super().__init__()
        s0 = float(weight0.abs().mean().clamp(min=1e-8))
        self.weight_latent = torch.nn.Parameter(weight0.detach().clone() / s0)
        self.scale = torch.nn.Parameter(torch.tensor(s0))
        if bias0 is not None:
            self.bias = torch.nn.Parameter(bias0.detach().clone())
        else:
            self.register_parameter("bias", None)

    def effective_weight(self) -> torch.Tensor:
        return ternary_codes(self.weight_latent) * self.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.effective_weight(), self.bias)


class RotatedTernaryLinear(torch.nn.Module):
    """Hadamard arm: W_eff = R_out Q R_in with ternary Q.

    weight_latent lives in the rotated domain (init = R_out W0 R_in /
    mean|latent0|) with an independently learned per-linear scale.
    R_out W0 R_in quantized at step 0 matches the control arm's step-0
    function up to ternary-quantization noise in two bases (parity
    gate)."""

    def __init__(self, weight0: torch.Tensor, bias0: torch.Tensor | None,
                 h: torch.Tensor, rotate_out: bool = True):
        super().__init__()
        block = h.shape[0]
        d_out, d_in = weight0.shape
        if d_in % block or (rotate_out and d_out % block):
            raise ValueError("dims not divisible by rotation block")
        self.register_buffer("h", h.half())
        self.rotate_out = rotate_out
        latent0 = weight0.detach().clone().float()
        latent0 = latent0 @ torch.block_diag(
            *([h] * (d_in // block)))
        if rotate_out:
            latent0 = torch.block_diag(*([h] * (d_out // block))) @ latent0
        s0 = float(latent0.abs().mean().clamp(min=1e-8))
        self.weight_latent = torch.nn.Parameter(latent0 / s0)
        self.scale = torch.nn.Parameter(torch.tensor(s0))
        if bias0 is not None:
            self.bias = torch.nn.Parameter(bias0.detach().clone())
        else:
            self.register_parameter("bias", None)

    def effective_weight(self) -> torch.Tensor:
        return materialize_w_eff(ternary_codes(self.weight_latent),
                                 float(self.scale.detach()), self.h,
                                 True, self.rotate_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = ternary_codes(self.weight_latent)
        xr = rotate_blocks(x, self.h)
        y = F.linear(xr, q * self.scale, None)
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
    # control arm's, so the paired design holds. Embeddings train fp32
    # in BOTH arms identically.
    model.to(device)
    model.gradient_checkpointing_enable()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    return model, h, linear_modules, tokenizer


# ---- data -------------------------------------------------------------------
def make_stream(cache_path: str) -> np.ndarray:
    return np.load(cache_path, mmap_mode="r")


class TokenWindowIter:
    """Deterministic non-overlapping windows from a token cache.

    Both arms get byte-identical streams because iteration always
    starts at offset 0 and advances by exactly BATCH*SEQ_LEN tokens.
    Budgets larger than one cache pass wrap cyclically."""

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
        # base excludes the final partial window.
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
def _ce_on_window(model, ids, labels, device) -> float:
    ids, labels = ids.to(device), labels.to(device)
    with torch.no_grad(), torch.autocast(device_type="cuda",
                                         dtype=torch.float16):
        logits = model(ids).logits
        loss = F.cross_entropy(
            logits[:, :-1].contiguous().reshape(-1, logits.shape[-1]),
            labels[:, 1:].contiguous().reshape(-1))
    return float(loss.detach())


def train(arm: str, device: str, run_dir: str,
          max_steps: int = TOTAL_STEPS, quick: int | None = None) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    model, h, linear_modules, _ = build_arm(arm, device)
    stream = make_stream(CACHE_TRAIN)
    it = TokenWindowIter(stream)
    if len(it) == 0:
        raise SystemExit("[ah1] token cache too small for one window")
    scales = [m.scale for m in linear_modules]
    other = [p for p in model.parameters()
             if all(p is not s for s in scales)]
    optimizer = torch.optim.AdamW(
        [{"params": scales, "weight_decay": 0.0},
         {"params": other, "weight_decay": WEIGHT_DECAY}],
        lr=LR, betas=(0.9, 0.95))
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
                }
            optimizer.step()
            window_loss += float(loss.detach())
            step += 1
            if step % 100 == 0:
                rate = None
                if step % FLIP_WINDOW == 0:
                    codes = {i: ternary_codes(m.weight_latent.detach()
                                              ).flatten()
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
    # runtime CE on a held-out window (test cache, window 0)
    test_it = TokenWindowIter(make_stream(CACHE_TEST))
    runtime_ce = _ce_on_window(model, *next(iter(test_it)), device)
    # save runtime state + materialize in-process
    save_state_and_materialize(arm, run_dir, model, linear_modules)
    # materialize cross-check: same window through the stock checkpoint
    mat_ce = _materialized_ce(run_dir / "final_hf", device)
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
        "runtime_ce_heldout": runtime_ce,
        "materialized_ce_heldout": mat_ce,
        "materialize_cross_check_nats": abs(runtime_ce - mat_ce),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    _write_cost_vector(model, linear_modules, run_dir)
    print(f"[ah1] {arm} done: {json.dumps(summary, indent=2)}", flush=True)


def _materialized_ce(checkpoint_dir: Path, device: str) -> float:
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint_dir), torch_dtype=torch.float16).to(device)
    model.eval()
    it = iter(TokenWindowIter(make_stream(CACHE_TEST)))
    ids, labels = next(it)
    return _ce_on_window(model, ids, labels, device)


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
    """Mean loss over the last `window` steps (from per-100-step
    records), independent of the absolute step count."""
    pairs = []
    with open(run_dir / "history.jsonl") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            pairs.append((rec["step"], rec["loss"]))
    if not pairs:
        return None
    max_step = max(s for s, _ in pairs)
    tail = [loss for s, loss in pairs if s > max_step - window]
    return float(np.mean(tail))


def _write_cost_vector(model, linear_modules, run_dir: Path):
    n_codes = sum(m.weight_latent.numel() for m in linear_modules)
    n_linears = len(linear_modules)
    code_bytes = n_codes * 2 // 8  # 2 bits/entry packed
    scale_bytes = n_linears * 2   # fp16 per-linear scale
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


# ---- state save + materialize ----------------------------------------------
LINEAR_PATH_TEMPLATES = None  # built once in _linear_path_names


def _linear_path_names(model) -> list[str]:
    path_names = []
    for li in range(len(model.model.decoder.layers)):
        base = f"model.decoder.layers.{li}"
        path_names += [f"{base}.self_attn.{n}.weight"
                       for n in ["q_proj", "k_proj", "v_proj", "out_proj"]]
        path_names += [f"{base}.fc1.weight", f"{base}.fc2.weight"]
    path_names.append("lm_head.weight")
    return path_names


def save_state_and_materialize(arm: str, run_dir: str,
                               model=None, linear_modules=None) -> None:
    """Save runtime state + materialized HF checkpoint.

    Called in-process from train() with the live model; the CLI mode
    rebuilds from the saved state (fallback path)."""
    from transformers import OPTConfig, OPTForCausalLM
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if model is None:
        model, h, linear_modules, tokenizer = build_arm(arm, "cpu")
        model.load_state_dict(torch.load(run_dir / f"{arm}_state.pt",
                                         map_location="cpu",
                                         weights_only=False))
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()},
               run_dir / f"{arm}_state.pt")
    config = OPTConfig.from_pretrained(MODEL_ID)
    config.tie_word_embeddings = False
    stock = OPTForCausalLM(config)
    sd = dict(stock.state_dict())
    # trained embeddings/norms/pos/bias: copy from trained model
    for k, v in model.state_dict().items():
        if any(k.startswith(p) for p in
               ["model.decoder.embed_", "model.decoder.layers.",
                "model.decoder.final_layer_norm", "model.decoder.project_"]):
            if ("weight_latent" not in k and "scale" not in k
                    and not k.endswith(".h")):
                sd[k] = v.detach().clone()
    # W_eff per swapped linear, in linear_modules order
    path_names = _linear_path_names(model)
    assert len(path_names) == len(linear_modules)
    for pname, m in zip(path_names, linear_modules, strict=True):
        sd[pname] = m.effective_weight().detach().clone()
    stock.load_state_dict(sd)
    out_dir = run_dir / "final_hf"
    stock.save_pretrained(out_dir)
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained(MODEL_ID).save_pretrained(out_dir)
    # ternary codes + scales for the deployed-form record
    codes_meta = {}
    for pname, m in zip(path_names, linear_modules, strict=True):
        codes_meta[pname] = {"scale": float(m.scale.detach()),
                             "shape": list(m.weight_latent.shape)}
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
    loss = _ce_on_window(model, ids, labels, device)
    rec = {"arm": arm, "step0_loss": loss}
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
