"""Phase-3 stage B: multi-level KD for the rotated Bonsai student.

Loss = KL(logits, T=1) + 0.5 * sum of attention-map KL over the LAST 2
layers (MiniLM-style; last layers carry task behavior). Teacher:
Qwen/Qwen3-1.7B on cuda:1. Student: arm-H wrap (rotated o/down,
all-196 ternary STE), continuing from warm-up latents.

Both models run eager attention so attention weights materialize.
Same optimizer discipline as the CPT stage (SGD 0.9, wd 0, clip 1.0).
"""
import argparse
import sys
import time

sys.path.insert(0, "/tmp")
sys.modules.setdefault("triton", None)

import numpy as np
import torch
import torch.nn.functional as F

import hstack
from hstack import wrap_model

STUDENT = "prism-ml/Ternary-Bonsai-1.7B-unpacked"
TEACHER = "Qwen/Qwen3-1.7B"
SEQ = 512
VOCAB = 151669
ATTN_LAYERS = (-2, -1)
ATTN_W = 0.5


def kd_step(student, teacher, ids):
    t_out = teacher(input_ids=ids.to("cuda:1"), output_attentions=True)
    s_out = student(input_ids=ids.to("cuda:0"), output_attentions=True)
    s_log = s_out.logits[..., :VOCAB].float()
    t_log = t_out.logits[..., :VOCAB].float().to("cuda:0")
    # per-position KL, meaned over batch and positions (batchmean on a
    # 3D tensor would silently sum over the position dim — ~512x scale bug)
    loss = F.kl_div(F.log_softmax(s_log, dim=-1),
                    F.softmax(t_log, dim=-1),
                    reduction="none").sum(-1).mean()
    attn_loss = 0.0
    for i in ATTN_LAYERS:
        sa = s_out.attentions[i].float()          # (1, H, L, L)
        ta = t_out.attentions[i].float().to("cuda:0")
        # per-(head, query-row) KL, meaned over heads and rows
        attn_loss = attn_loss + F.kl_div(
            F.log_softmax(sa, dim=-1), F.softmax(ta, dim=-1),
            reduction="none").sum(-1).mean()
    return loss + ATTN_W * attn_loss, loss.item(), float(attn_loss)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--step-offset", type=int, default=2500)
    p.add_argument("--start-latents", default=None,
                   help="omit to start from the clean rotated-requant "
                        "point (C1: KL512 0.7556)")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--opt", choices=["adamw", "sgd"], default="adamw")
    p.add_argument("--train-targets", choices=["all", "rotated"],
                   default="rotated",
                   help="rotated: train only o/down latents (corrected "
                        "recipe — full-weight SGD destroyed capability)")
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--save-latents", default=None)
    p.add_argument("--heartbeat", default=None)
    args = p.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(STUDENT)
    student = AutoModelForCausalLM.from_pretrained(
        STUDENT, dtype=torch.float16, attn_implementation="eager").to("cuda:0")
    student.requires_grad_(False)
    wraps = wrap_model(student, gs=32, ste=True, full=True, rotate_all=True)
    if args.start_latents:
        d = np.load(args.start_latents)
        for i, hw in enumerate(wraps):
            with torch.no_grad():
                hw.latent.copy_(torch.as_tensor(d[f"lat_{i}"], device="cuda:0"))
        print(f"[KD] resumed {len(wraps)} latents", flush=True)
    else:
        print("[KD] starting from rotated-requant init (C1 point)", flush=True)
    if args.train_targets == "rotated":
        for hw in wraps:
            if not hw.rotate:
                hw.latent.requires_grad_(False)
    latents = [hw.latent for hw in wraps if hw.latent.requires_grad]

    student.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    student.config.use_cache = False
    student.train()

    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER, dtype=torch.float16, attn_implementation="eager").to("cuda:1")
    teacher.eval().requires_grad_(False)

    need = (args.step_offset + args.steps + 8) * (SEQ + 1)
    stream = load_dataset("wikimedia/wikipedia", "20231101.en",
                          split="train", streaming=True).shuffle(
                              seed=0, buffer_size=2000)
    chunks, total = [], 0
    for art in stream:
        t = art["text"].strip()
        if len(t) < 400:
            continue
        chunks.append(tok(t, return_tensors="pt").input_ids[0])
        total += chunks[-1].numel()
        if total >= need:
            break
    enc = torch.cat(chunks)
    print(f"[KD] corpus {total/1e6:.1f}M tokens", flush=True)

    if args.opt == "adamw":
        opt = torch.optim.AdamW(latents, lr=args.lr, betas=(0.9, 0.95),
                                weight_decay=0.0)
    else:
        opt = torch.optim.SGD(latents, lr=args.lr, momentum=0.9,
                              weight_decay=0.0)
    t0 = time.time()
    for step in range(args.steps):
        w = args.step_offset + step
        ids = enc[w * SEQ:(w + 1) * SEQ + 1].unsqueeze(0)
        loss, l_kl, l_at = kd_step(student, teacher, ids)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(latents, 1.0)
        opt.step()
        if step % args.log_every == 0 or step == args.steps - 1:
            msg = (f"step={step} loss={l_kl:.4f} attn={l_at:.4f} "
                   f"elapsed={time.time()-t0:.1f}s gnorm={gn.item():.2f}")
            print(msg, flush=True)
            if args.heartbeat:
                with open(args.heartbeat, "a") as f:
                    f.write(msg + "\n")
        del loss

    if args.save_latents:
        np.savez(args.save_latents,
                 **{f"lat_{i}": hw.latent.detach().cpu().numpy()
                    for i, hw in enumerate(wraps)})
        print(f"[KD] latents saved {args.save_latents}", flush=True)

    # end-of-run KL512 on the fixed val windows (teacher already loaded)
    ds_v = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    text_v = "\n".join(t for t in ds_v["text"] if t.strip())
    enc_v = tok(text_v, return_tensors="pt").input_ids[0]
    student.eval()
    tot = 0.0
    with torch.no_grad():
        for i in range(64):
            ids = enc_v[i * SEQ:(i + 1) * SEQ + 1].unsqueeze(0)
            t = teacher(input_ids=ids.to("cuda:1")).logits[..., :VOCAB].float()
            s = student(input_ids=ids.to("cuda:0")).logits[..., :VOCAB].float()
            lp = F.log_softmax(s, dim=-1)
            pp = F.softmax(t, dim=-1).to(lp.device)
            tot += F.kl_div(lp, pp, reduction="none").sum(-1).mean().item()
    print(f"[KD] FINAL KL512 = {tot / 64:.4f}", flush=True)


if __name__ == "__main__":
    main()
