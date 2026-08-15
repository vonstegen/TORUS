"""HuggingFace transformers adapter for `torus.train.DistillationTrainer`.

The trainer expects:
    forward(batch, n_planes) -> (logits, hidden, route)

(`params` is supplied by the adapter at construction time, since the
trainer wires the optimizer through the same STE list across
calls.) This module wraps a `transformers` causal-LM (e.g. OLMo /
OLMoE) into that interface, so a real open base can be driven by
the existing numpy-trainer without rewriting the trainer.

Usage:

    from torus.train.hf_adapter import HFStudentAdapter, HFTeacherAdapter
    student = HFStudentAdapter(
        model_name="allenai/OLMo-1B-hf",
        target_modules=["att_proj", "ff_proj"],   # names to replace
        dtype="float32",
    )
    teacher = HFTeacherAdapter(
        model_name="allenai/OLMo-1B-hf",
        dtype="float32",
    )
    trainer = DistillationTrainer(
        student_params=student.ste_params,
        forward_student=student.forward,
        forward_teacher=teacher.forward,
        data=my_data_iter,
        ...
    )

The adapter only depends on `torch` and `transformers`; importing it
without those installed raises an informative error.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from torus.train.loop import DistillationBatch
from torus.train.ste import TernarySTE


def _require_torch():
    try:
        import torch  # type: ignore
        return torch
    except Exception as exc:  # pragma: no cover - exercised at runtime only
        raise RuntimeError(
            "hf_adapter requires PyTorch. Install with: "
            "`pip install torch transformers accelerate`"
        ) from exc


def _require_transformers():
    try:
        import transformers  # type: ignore
        return transformers
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "hf_adapter requires transformers. Install with: "
            "`pip install transformers accelerate`"
        ) from exc


@dataclass
class HFAdapterConfig:
    """Configuration shared by student + teacher adapters."""
    model_name: str = "allenai/OLMo-1B-hf"
    dtype: str = "float32"
    target_modules: Sequence[str] = (
        "att_proj",  # self-attention output projection
        "ff_proj",   # feed-forward down-projection
    )
    cache_dir: str | None = None
    device: str = "cpu"


def _make_forward_stub(ste: TernarySTE, qfn, *, transpose_weight: bool = False, bias_param=None):
    """Build a forward that re-applies the STE weights on every call.

    `transpose_weight=True` is for HF `Conv1D` modules whose weights
    are stored transposed relative to `nn.Linear`. The stub applies
    `F.linear(x, q_w.T, q_b)` so the math matches the Conv1D contract.
    `bias_param` is an optional torch.nn.Parameter; if present, it's
    used as-is (no quantization) so the bias can stay fp32-trainable.
    """
    import torch as _torch  # local import to avoid module-level noise
    def fwd(x):
        # `ste.weight` may be a torch Parameter; convert to numpy
        # before handing to the numpy quantizer.
        import torch as _torch
        import numpy as _np
        w = ste.weight
        if isinstance(w, _torch.Tensor):
            w_np = w.detach().cpu().numpy()
        else:
            w_np = _np.asarray(w)
        codes, scale, q_w = qfn(w_np, group_size=ste.group_size)
        w_t = q_w.T if transpose_weight else q_w
        w_t = _torch.as_tensor(w_t, dtype=x.dtype, device=x.device)
        qb = bias_param if bias_param is not None else None
        return _F.linear(x, w_t, qb)
    return fwd


class HFStudentAdapter:
    """Wrap a transformers causal-LM as a `forward(batch, n_planes)`.

    Each trainable ternary parameter corresponds to one replaced
    Linear layer in the model. The adapter intercepts the named
    modules on the first forward, attaches a `TernarySTE` per linear,
    and exposes the STE list as `ste_params` so the trainer can
    hand them to the optimizer.

    Notes
    -----
    - The current implementation monkey-patches `nn.Linear.forward`
    for the targeted module names. It's a thin reference adapter;
    a real torch-autograd loop would replace the patched forward
    with a custom `torch.autograd.Function`.
    - The adapter's `forward` returns the model's final logits plus
    the last hidden state and an all-zeros MoE route (real OLMoE
    routes are not extracted here; that lands in Phase 4).
    """

    def __init__(self, config: HFAdapterConfig | None = None) -> None:
        torch = _require_torch()
        _require_transformers()
        self.config = config or HFAdapterConfig()

        # Lazy-load to keep import cheap for users who never call this.
        from transformers import AutoModelForCausalLM  # type: ignore
        import torch.nn as nn  # type: ignore
        global _F
        _F = nn.functional

        dtype = getattr(torch, self.config.dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=dtype,
            cache_dir=self.config.cache_dir,
        ).to(self.config.device)
        self.model.eval()  # QAT/STE handles quantization
        self._ste_params: list[TernarySTE] = []
        self._bias_params: list = []
        self._patched_modules: list[tuple[object, str]] = []
        self._attach_ste()

    def _attach_ste(self) -> None:
        """Wrap every Linear or Conv1D under a target module name."""
        import torch.nn as nn  # type: ignore
        from transformers.pytorch_utils import Conv1D  # type: ignore
        from torus.train.ste import ternary_quantize_with_ste

        targets = set(self.config.target_modules)

        for name, module in list(self.model.named_modules()):
            short = name.rsplit(".", 1)[-1]
            if short not in targets:
                continue
            if isinstance(module, nn.Linear):
                transpose = False
                weight = module.weight.detach().clone()
            elif isinstance(module, Conv1D):
                # Conv1D stores weight transposed: (in, out).
                # Keep the STE weight in that orientation and let the
                # patched forward transpose back.
                transpose = True
                weight = module.weight.detach().clone()
            else:
                continue

            bias_param = (
                None
                if module.bias is None
                else _nn_param(module.bias.detach().clone())
            )

            ste = TernarySTE(
                weight=_nn_param(weight),
                group_size=min(128, weight.shape[0]),
            )
            self._ste_params.append(ste)
            self._bias_params.append(bias_param)

            module.forward = _make_forward_stub(
                ste,
                ternary_quantize_with_ste,
                transpose_weight=transpose,
                bias_param=bias_param,
            )  # type: ignore[assignment]
            self._patched_modules.append((module, name))

    @property
    def ste_params(self) -> list[TernarySTE]:
        return list(self._ste_params)

    def forward(
        self,
        batch: DistillationBatch,
        n_planes: int,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Run the model; return (logits, hidden, route)."""
        torch = _require_torch()
        for (ste, bias_param), target in zip(
            zip(self._ste_params, self._bias_params), self._patched_modules
        ):
            module, _name = target
            with torch.no_grad():
                if module.weight.shape == ste.weight.shape:
                    module.weight.copy_(ste.weight)
                elif module.weight.shape == tuple(ste.weight.shape[::-1]):
                    module.weight.copy_(ste.weight.t())
                else:
                    raise RuntimeError(
                        f"weight shape mismatch for {module}: "
                        f"{tuple(module.weight.shape)} vs {tuple(ste.weight.shape)}"
                    )
                if module.bias is not None and bias_param is not None:
                    module.bias.copy_(bias_param)

        ids = torch.as_tensor(batch.inputs, dtype=torch.long, device=self.config.device)
        with torch.no_grad():
            out = self.model(input_ids=ids, output_hidden_states=True)
        logits = out.logits.detach().to("cpu").numpy()
        hidden = (
            out.hidden_states[-1].detach().to("cpu").numpy()
            if out.hidden_states is not None
            else None
        )
        batch_t = (
            batch.inputs.shape[0]
            if hasattr(batch.inputs, "shape")
            else len(batch.inputs)
        )
        route = np.zeros((batch_t, 1), dtype=np.float32)
        return logits, hidden, route


class HFTeacherAdapter:
    """Frozen, full-precision teacher mirror of the same HF model.

    The teacher is loaded once and never modified. The trainer calls
    its `forward` once per step to compute the distillation target.
    """

    def __init__(self, config: HFAdapterConfig | None = None) -> None:
        torch = _require_torch()
        _require_transformers()
        self.config = config or HFAdapterConfig()
        from transformers import AutoModelForCausalLM  # type: ignore
        dtype = getattr(torch, self.config.dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=dtype,
            cache_dir=self.config.cache_dir,
        ).to(self.config.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def forward(
        self,
        batch: DistillationBatch,
        n_planes: int,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        torch = _require_torch()
        ids = torch.as_tensor(batch.inputs, dtype=torch.long, device=self.config.device)
        with torch.no_grad():
            out = self.model(input_ids=ids, output_hidden_states=True)
        logits = out.logits.detach().to("cpu").numpy()
        hidden = (
            out.hidden_states[-1].detach().to("cpu").numpy()
            if out.hidden_states is not None
            else None
        )
        batch_t = (
            batch.inputs.shape[0]
            if hasattr(batch.inputs, "shape")
            else len(batch.inputs)
        )
        route = np.zeros((batch_t, 1), dtype=np.float32)
        return logits, hidden, route


def _nn_param(t):
    """Wrap a tensor as a torch Parameter if torch is available."""
    torch = _require_torch()
    return torch.nn.Parameter(t)