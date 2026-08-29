"""Real Low-Rank Adaptation (LoRA) blocks for the universal paradigm (PR-7).

LoRA (Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*,
ICLR 2022) freezes a base layer's weight :math:`W \\in \\mathbb{R}^{out\\times in}`
and learns a low-rank residual

.. math::

    W' = W + \\frac{\\alpha}{r}\\, B A,
    \\qquad A \\in \\mathbb{R}^{r\\times in},\\; B \\in \\mathbb{R}^{out\\times r}.

``A`` is kaiming-initialised and ``B`` is zero-initialised, so the adapter is
an exact **identity** at the start of training and only adds trainable
low-rank deviations as it learns. This lets one frozen foundation
reconstruction model absorb mixed contrast / field-strength / acceleration
regimes at a fraction of the parameter cost of full fine-tuning.

Supports ``nn.Linear`` and ``nn.Conv2d`` base layers. The convolutional case
applies the same :math:`W + (\\alpha/r) B A` factorisation to the flattened
kernel ``[out, in*kh*kw]`` so the update remains a true low-rank perturbation
of the convolution weight (no separate "down/up conv" approximation).
"""

from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn as nn
from torch.nn.functional import conv2d, linear

__all__ = ["LoRAAdapter", "inject_lora"]


class LoRAAdapter(nn.Module):
    """Wrap a frozen ``nn.Linear`` / ``nn.Conv2d`` with a trainable low-rank update.

    Args:
        base: The base layer to adapt. Its parameters are frozen
            (``requires_grad=False``); only the low-rank factors train.
        rank: Rank ``r`` of the update. Must be ``>= 1``.
        alpha: Scaling factor; the effective update is ``(alpha / rank) * B @ A``.

    The adapter is an identity at initialization (``B = 0``).
    """

    def __init__(self, base: nn.Module, rank: int, alpha: float = 1.0) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError(f"LoRA rank must be >= 1, got {rank}")
        if not isinstance(base, (nn.Linear, nn.Conv2d)):
            raise TypeError(
                "LoRAAdapter supports nn.Linear and nn.Conv2d base layers; "
                f"got {type(base).__name__}"
            )

        self.base = base
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / rank
        self.is_conv = isinstance(base, nn.Conv2d)

        # Freeze the base layer — only the low-rank factors train.
        for p in self.base.parameters():
            p.requires_grad = False

        if self.is_conv:
            conv = cast(nn.Conv2d, base)
            kh, kw = conv.kernel_size
            in_dim = conv.in_channels // conv.groups * kh * kw
            out_dim = conv.out_channels
        else:
            lin = cast(nn.Linear, base)
            in_dim = lin.in_features
            out_dim = lin.out_features

        self._in_dim = in_dim
        self._out_dim = out_dim

        # A: (r, in)  kaiming;  B: (out, r)  zeros  -> identity at init.
        self.lora_A = nn.Parameter(torch.empty(rank, in_dim))
        self.lora_B = nn.Parameter(torch.zeros(out_dim, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def _delta_weight(self) -> torch.Tensor:
        """The low-rank weight update ``(alpha/r) * B @ A`` of shape ``[out, in]``."""
        return self.scaling * (self.lora_B @ self.lora_A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        delta = self._delta_weight()
        if self.is_conv:
            conv = cast(nn.Conv2d, self.base)
            delta_w = delta.view(
                conv.out_channels,
                conv.in_channels // conv.groups,
                *conv.kernel_size,
            )
            lora_out = conv2d(
                x,
                delta_w,
                bias=None,
                stride=conv.stride,
                padding=conv.padding,
                dilation=conv.dilation,
                groups=conv.groups,
            )
        else:
            lora_out = linear(x, delta)
        return base_out + lora_out


def inject_lora(
    module: nn.Module,
    rank: int,
    alpha: float = 1.0,
    target_types: tuple[type[nn.Module], ...] = (nn.Conv2d,),
) -> int:
    """Replace every direct/nested ``target_types`` submodule with a :class:`LoRAAdapter`.

    Walks ``module`` and swaps in-place each child whose type is in
    ``target_types`` for a LoRA-wrapped version of itself. Because each
    adapter is identity at init, ``inject_lora`` does not change the model's
    forward output until the low-rank factors are trained.

    Args:
        module: The model to inject adapters into (mutated in place).
        rank: LoRA rank passed to each :class:`LoRAAdapter`.
        alpha: LoRA scaling passed to each :class:`LoRAAdapter`.
        target_types: Submodule types to wrap. Defaults to ``(nn.Conv2d,)``.

    Returns:
        The number of submodules wrapped.
    """
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, LoRAAdapter):
            continue
        if isinstance(child, target_types):
            setattr(module, name, LoRAAdapter(child, rank=rank, alpha=alpha))
            count += 1
        else:
            count += inject_lora(child, rank, alpha, target_types)
    return count
