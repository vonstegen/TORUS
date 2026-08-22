"""Phase-2 smoke arms: CPT on Bonsai in quantized form, rotated vs not.

Arm H: o_proj/down_proj rotated (Hadamard), all 196 linears ternary STE.
Arm C: same but unrotated (control — isolates the rotation effect).

Recipe (both arms identical): next-token CE, seq 512, batch 1, wikitext
train windows (stride 512, seed 0 order), SGD momentum 0.9 lr 3e-4,
weight decay 0 (latent-wd is a silent STE pathology), clip 1.0.
End-of-run: KL512 vs Qwen3 teacher on the fixed 64-window val set.
"""
import argparse
import sys
import time

sys.path.insert(0, "/tmp")
sys.modules.setdefault("triton", None)

import numpy as np
import torch

import hstack
from hstack import wrap_model

STUDENT = "prism-ml/Ternary-Bonsai-1.7B-unpacked"
TEACHER = "Qwen/Qwen3-1.7B"
SEQ = 512
NWIN_EVAL = 64


@torch.no_grad()
def kl512(student, teacher, windows, vocab):
    tot = 0.0
    for ids in windows:
        t = teacher(input_ids=ids.to("cuda:1")).logits[..., :vocab].float()
        s = student(input_ids=ids.to("cuda:0")).logits[..., :vocab].float()
        lp = torch.log_softmax(s, dim=-1)
        p = torch.softmax(t, dim=-1).to(lp.device)
        tot += torch.nn.functional.kl_div(lp, p, reduction="none").sum(-1).mean().item()
    return tot / len(windows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=["H", "C"], required=True)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--step-offset", type=int, default=0,
                   help="window offset into the corpus stream (chained stages)")
    p.add_argument("--data", choices=["wikitext", "wikipedia"], default="wikitext")
    p.add_argument("--start-latents", default=None,
                   help="npz of latents to initialize from (stage chaining)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--opt", choices=["sgd", "adamw"], default="sgd")
    p.add_argument("--train-targets", choices=["all", "rotated"], default="all",
                   help="rotated: train only o_proj/down_proj latents; the "
                        "plain-wrapped linears stay frozen at exact Bonsai values")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-latents", default=None)
    p.add_argument("--heartbeat", default=None)
    args = p.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(STUDENT)
    student = AutoModelForCausalLM.from_pretrained(
        STUDENT, dtype=torch.float16, attn_implementation="sdpa").to("cuda:0")
    student.requires_grad_(False)

    wraps = wrap_model(student, gs=32, ste=True, full=True,
                       rotate_all=(args.arm == "H"))
    if args.start_latents:
        d = np.load(args.start_latents)
        for i, hw in enumerate(wraps):
            with torch.no_grad():
                hw.latent.copy_(torch.as_tensor(d[f"lat_{i}"], device="cuda:0"))
        print(f"[{args.arm}] resumed latents from {args.start_latents}",
              flush=True)
    if args.train_targets == "rotated":
        for hw in wraps:
            if not hw.rotate:
                hw.latent.requires_grad_(False)
    latents = [hw.latent for hw in wraps if hw.latent.requires_grad]
    n_el = sum(t.numel() for t in latents)
    print(f"[{args.arm}] {len(latents)} trainable latents, "
          f"{n_el/1e9:.2f}B params", flush=True)

    student.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    student.config.use_cache = False
    student.train()

    if args.data == "wikipedia":
        # Streaming (no disk): shuffled 20231101.en articles, accumulate
        # until we have (offset + steps + margin) windows of tokens.
        from datasets import load_dataset as _ld
        need = (args.step_offset + args.steps + NWIN_EVAL + 8) * (SEQ + 1)
        stream = _ld("wikimedia/wikipedia", "20231101.en",
                     split="train", streaming=True).shuffle(seed=0,
                                                            buffer_size=2000)
        chunks = []
        total = 0
        for art in stream:
            t = art["text"].strip()
            if len(t) < 400:
                continue
            chunks.append(tok(t, return_tensors="pt").input_ids[0])
            total += chunks[-1].numel()
            if total >= need:
                break
        enc = torch.cat(chunks)
        print(f"[{args.arm}] wikipedia stream: {total/1e6:.1f}M tokens",
              flush=True)
    else:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n".join(t for t in ds["text"] if t.strip())
        enc = tok(text, return_tensors="pt").input_ids[0]
    n_win = (enc.numel() - 1) // SEQ
    print(f"[{args.arm}] {n_win} train windows", flush=True)

    if args.opt == "adamw":
        opt = torch.optim.AdamW(latents, lr=args.lr, betas=(0.9, 0.95),
                                weight_decay=0.0)
    else:
        opt = torch.optim.SGD(latents, lr=args.lr, momentum=0.9,
                              weight_decay=0.0)

    t0 = time.time()
    for step in range(args.steps):
        w = args.step_offset + step
        ids = enc[w * SEQ:(w + 1) * SEQ + 1].unsqueeze(0).to("cuda:0")
        out = student(input_ids=ids[:, :-1])
        loss = torch.nn.functional.cross_entropy(
            out.logits[..., :151669].float().view(-1, 151669),
            ids[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(latents, 1.0)
        opt.step()
        if step % args.log_every == 0 or step == args.steps - 1:
            msg = (f"arm={args.arm} step={step} loss={loss.item():.4f} "
                   f"elapsed={time.time()-t0:.1f}s gnorm={gn.item():.2f}")
            print(msg, flush=True)
            if args.heartbeat:
                with open(args.heartbeat, "a") as f:
                    f.write(msg + "\n")
        if args.save_latents and (step + 1) % 500 == 0:
            np.savez(args.save_latents,
                     **{f"lat_{i}": hw.latent.detach().cpu().numpy()
                        for i, hw in enumerate(wraps)})
            print(f"[{args.arm}] checkpoint saved at step {step+1}",
                  flush=True)
        del out, loss

    # free optimizer states before eval
    del opt
    torch.cuda.empty_cache()

    ds_v = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    text_v = "\n".join(t for t in ds_v["text"] if t.strip())
    enc_v = tok(text_v, return_tensors="pt").input_ids[0]
    windows = [enc_v[i * SEQ:(i + 1) * SEQ + 1].unsqueeze(0)
               for i in range(NWIN_EVAL)]
    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER, dtype=torch.float16, attn_implementation="sdpa").to("cuda:1")
    teacher.eval().requires_grad_(False)
    student.eval()
    k = kl512(student, teacher, windows, 151669)
    print(f"[{args.arm}] FINAL KL512 = {k:.4f}", flush=True)


if __name__ == "__main__":
    main()
