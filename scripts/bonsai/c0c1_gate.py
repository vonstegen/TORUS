"""C0 + C1 gates for the Hadamard integration (roadmap /tmp/ROADMAP.md).

C0 identity: STE bypassed -> rotated model must reproduce stock Bonsai
             logits within fp16 rounding.
C1 price:    ternary STE on rotated latents -> KL512 rise vs teacher
             (the hole the CPT phase must climb out of).

Also reports KL512(stock Bonsai vs teacher) as the phase baseline, and a
module-level fold-back check (export path correctness).
"""
import sys

sys.path.insert(0, "/tmp")
sys.modules.setdefault("triton", None)

import torch
import torch.nn.functional as F

import hstack
from hstack import wrap_model

STUDENT = "prism-ml/Ternary-Bonsai-1.7B-unpacked"
TEACHER = "Qwen/Qwen3-1.7B"
SEQ = 512
NWIN = 64


@torch.no_grad()
def kl512(student, teacher, windows, vocab):
    tot, cnt = 0.0, 0
    for ids in windows:
        t = teacher(input_ids=ids.to("cuda:1")).logits[..., :vocab].float()
        s = student(input_ids=ids.to("cuda:0")).logits[..., :vocab].float()
        lp = F.log_softmax(s / 1.0, dim=-1)
        p = F.softmax(t, dim=-1).to(lp.device)
        tot += F.kl_div(lp, p, reduction="none").sum(-1).mean().item()
        cnt += 1
    return tot / cnt


def main():
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(STUDENT)
    student = AutoModelForCausalLM.from_pretrained(
        STUDENT, dtype=torch.float16, attn_implementation="sdpa").to("cuda:0")
    student.eval().requires_grad_(False)

    ids = torch.arange(64, dtype=torch.long, device="cuda:0").reshape(1, 64)
    stock_logits = student(input_ids=ids).logits

    wraps = wrap_model(student, gs=32, ste=False)
    print(f"[gate] wrapped {len(wraps)} linears (o_proj/down_proj)")

    bypass_logits = student(input_ids=ids).logits
    c0 = (bypass_logits - stock_logits).abs().max().item()
    scale = stock_logits.abs().max().item()
    print(f"[C0] max|logit diff| = {c0:.5f}  (logit range {scale:.1f})")

    # module-level fold-back check on one wrap
    hw = wraps[0]
    x = torch.randn(3, hw.in_features, dtype=torch.float16, device="cuda:0")
    y_hw = hw(x)
    y_fold = F.linear(x, hw.folded_weight().half())
    fb = (y_hw - y_fold).abs().max().item()
    print(f"[C0] fold-back module diff = {fb:.5f}")

    # KL512 eval windows
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n".join(t for t in ds["text"] if t.strip())
    enc = tok(text, return_tensors="pt").input_ids[0]
    windows = [enc[i * SEQ:(i + 1) * SEQ + 1].unsqueeze(0) for i in range(NWIN)]
    vocab = min(student.config.vocab_size, 151936)

    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER, dtype=torch.float16, attn_implementation="sdpa").to("cuda:1")
    teacher.eval().requires_grad_(False)

    k_stock = kl512(student, teacher, windows, vocab)  # ste=False (identity)
    print(f"[base] KL512 stock Bonsai vs teacher = {k_stock:.4f}")

    for hw in wraps:
        hw.ste = True
    k_rot = kl512(student, teacher, windows, vocab)
    print(f"[C1] KL512 rotated-requant = {k_rot:.4f}  (drop {k_rot - k_stock:+.4f})")

    print("[gate] done")


if __name__ == "__main__":
    main()
