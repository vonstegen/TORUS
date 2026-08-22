"""Export a folded plain-fp16 model from wrapped Bonsai (stock-HF loadable).

Usage: export_folded.py OUT_DIR [--latents NPZ]
  --latents NPZ: load trained latents (from h_cpt.py --save-latents)
                 instead of the init-from-Bonsai latents.
"""
import argparse
import sys

sys.path.insert(0, "/tmp")
sys.modules.setdefault("triton", None)

import numpy as np
import torch
import torch.nn as nn

import hstack
from hstack import wrap_model

STUDENT = "prism-ml/Ternary-Bonsai-1.7B-unpacked"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out_dir")
    p.add_argument("--latents", default=None)
    p.add_argument("--full", action="store_true",
                   help="wrap all 196 linears (must match the training wrap)")
    p.add_argument("--arm", choices=["H", "C"], default="H",
                   help="rotation layout (must match the trained arm)")
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        STUDENT, dtype=torch.float16, attn_implementation="sdpa").to("cuda:0")
    wraps = wrap_model(model, gs=32, ste=True, full=args.full,
                       rotate_all=(args.arm == "H"))
    if args.latents:
        d = np.load(args.latents)
        for i, hw in enumerate(wraps):
            key = f"lat_{i}"
            with torch.no_grad():
                hw.latent.copy_(torch.as_tensor(d[key], device="cuda:0"))
        print(f"[export] loaded {len(wraps)} latents from {args.latents}")
    with torch.no_grad():
        for name, mod in list(model.named_modules()):
            if isinstance(mod, hstack.HLinear):
                parent = model.get_submodule(name.rsplit(".", 1)[0])
                lin = nn.Linear(mod.in_features, mod.out_features, bias=False)
                lin.weight = nn.Parameter(mod.folded_weight().half())
                setattr(parent, name.rsplit(".", 1)[-1], lin)
    model.save_pretrained(args.out_dir)
    AutoTokenizer.from_pretrained(STUDENT).save_pretrained(args.out_dir)
    print(f"[export] saved {args.out_dir}")


if __name__ == "__main__":
    main()
