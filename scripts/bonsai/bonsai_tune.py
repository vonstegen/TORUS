"""PV-tune PrismML Ternary-Bonsai-1.7B scales against a Qwen3-1.7B teacher.

Attaches a free per-group scale multiplier m (init 1.0 => bit-identical)
to every targeted Linear's stored (already-ternary-dequantized) weights:
    W_eff = W * m_group        (group = 128 input features)
Codes are never touched — this frees ONLY the amplitude calibration,
the exact intervention that bought +0.042 KL on TORUS arm_pv.

Distill: KL(student || teacher) at T=2.0 on wikitext 16-token windows,
batch 1, SGD lr 1e-2 momentum 0.9, 500 steps. Only m is trained.

After training, scales are folded into the weights and saved as a plain
fp16 safetensors model for stock-HF eval.
"""
import argparse
import sys
import time

sys.modules.setdefault("triton", None)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
GS = 32


class ScaleWrap(nn.Module):
    """Linear with a free per-row, per-group scale multiplier (gs=32,
    matching Bonsai's native scale granularity — this frees exactly
    their scale DOF)."""

    def __init__(self, linear: nn.Linear, gs: int = GS):
        super().__init__()
        self.linear = linear
        w = linear.weight
        self.gs = gs
        self.m = nn.Parameter(torch.ones(w.shape[0], w.shape[1] // gs,
                                         dtype=torch.float32, device=w.device))

    def forward(self, x):
        w = self.linear.weight
        r, c = w.shape
        weff = (w.view(r, -1, self.gs) * self.m.to(w.dtype).unsqueeze(-1)).view(r, c)
        return F.linear(x, weff, self.linear.bias)


def wrap_model(model):
    wraps = []
    for name, mod in list(model.named_modules()):
        if name.rsplit(".", 1)[-1] in TARGETS and isinstance(mod, nn.Linear):
            parent = model.get_submodule(name.rsplit(".", 1)[0])
            sw = ScaleWrap(mod)
            setattr(parent, name.rsplit(".", 1)[-1], sw)
            wraps.append(sw)
    return wraps


def kl_loss(student_logits, teacher_logits, T=2.0):
    # Vocab padding differs (Bonsai 151669, Qwen3 teacher 151936) —
    # restrict the KL to the shared (real-token) support.
    v = min(student_logits.shape[-1], teacher_logits.shape[-1])
    s = F.log_softmax(student_logits[..., :v].float() / T, dim=-1)
    t = F.softmax(teacher_logits[..., :v].float() / T, dim=-1)
    return F.kl_div(s, t, reduction="batchmean") * (T * T)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--student", default="prism-ml/Ternary-Bonsai-1.7B-unpacked")
    p.add_argument("--teacher", default="Qwen/Qwen3-1.7B")
    p.add_argument("--n-steps", type=int, default=500)
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--save-dir", default="/tmp/hf/bonsai_pv")
    p.add_argument("--heartbeat", default="/tmp/hf/bonsai_pv.heartbeat")
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    print("[tune] loading student ...", flush=True)
    student = AutoModelForCausalLM.from_pretrained(
        args.student, dtype=torch.float16, attn_implementation="sdpa"
    ).to("cuda:0")
    student.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(args.student)

    # --- probe: wrapping must be bit-identical at m=1 ---
    ids_probe = torch.arange(16, dtype=torch.long, device="cuda:0").reshape(1, 16)
    with torch.no_grad():
        ref = student(input_ids=ids_probe).logits
    wraps = wrap_model(student)
    with torch.no_grad():
        after = student(input_ids=ids_probe).logits
    diff = (ref - after).abs().max().item()
    print(f"[tune] bit-identical check: max|diff|={diff} ({len(wraps)} wraps)", flush=True)
    assert diff == 0.0, "wrap is not bit-identical at m=1"

    student.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    student.config.use_cache = False
    student.train()

    print("[tune] loading teacher ...", flush=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher, dtype=torch.float16, attn_implementation="sdpa"
    ).to("cuda:1").eval().requires_grad_(False)

    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n".join(t for t in ds["text"] if t.strip())
    enc = tok(text, return_tensors="pt").input_ids[0]
    n = enc.numel() - args.seq_len - 1
    print(f"[tune] corpus tokens: {enc.numel()}", flush=True)

    params = [sw.m for sw in wraps]
    n_el = sum(p.numel() for p in params)
    print(f"[tune] trainable: {len(params)} groups, {n_el} elements", flush=True)
    opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9)

    t0 = time.time()
    g = torch.Generator().manual_seed(0)
    for step in range(args.n_steps):
        i = int(torch.randint(n, (1,), generator=g))
        ids = enc[i:i + args.seq_len + 1].unsqueeze(0)
        s_ids = ids.to("cuda:0")
        with torch.no_grad():
            t_logits = teacher(input_ids=ids.to("cuda:1")).logits
        s_logits = student(input_ids=s_ids).logits
        loss = kl_loss(s_logits, t_logits.to("cuda:0"))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, float("inf"))
        opt.step()
        if step % args.log_every == 0 or step == args.n_steps - 1:
            msg = (f"step={step} loss={loss.item():.6f} elapsed={time.time()-t0:.1f}s "
                   f"gnorm={gn.item():.2f}")
            print(msg, flush=True)
            with open(args.heartbeat, "a") as f:
                f.write(msg + "\n")
        del s_logits, t_logits, loss

    # --- fold scales into weights, save plain fp16 model ---
    student.eval()
    with torch.no_grad():
        for sw in wraps:
            w = sw.linear.weight
            r, c = w.shape
            w.copy_((w.view(r, -1, sw.gs) * sw.m.to(w.dtype).unsqueeze(-1)).view(r, c))
    mv = torch.cat([sw.m.detach().flatten() for sw in wraps]).float().cpu().numpy()
    dev = mv - 1.0
    print(f"[tune] scale movement: mean|m-1|={np.abs(dev).mean():.5f} "
          f"max={np.abs(dev).max():.4f} range=[{mv.min():.4f}, {mv.max():.4f}]", flush=True)
    np.savez("/tmp/hf/bonsai_pv_scales.npz", m=mv)
    # unwrap: replace ScaleWrap with the inner Linear
    for name, mod in list(student.named_modules()):
        if isinstance(mod, ScaleWrap):
            parent = student.get_submodule(name.rsplit(".", 1)[0])
            setattr(parent, name.rsplit(".", 1)[-1], mod.linear)
    student.config.use_cache = True
    student.save_pretrained(args.save_dir)
    tok.save_pretrained(args.save_dir)
    print(f"[tune] saved {args.save_dir}", flush=True)


if __name__ == "__main__":
    main()
