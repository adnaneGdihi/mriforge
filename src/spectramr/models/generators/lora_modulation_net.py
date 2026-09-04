"""Continuous low-rank modulation cross-field translator (MICCAI MRIxFields2026, B-3.6).

A single SHARED conv backbone whose conv weights are modulated by a CONTINUOUS low-rank (LoRA)
update generated as a smooth function of the field pair ``(b_s, b_t)``
(:class:`~spectramr.models.blocks.field_conditioned_lora.FieldConditionedLoRAConv2d`). The rank
bound ``r << dim(W)`` is the "compliance-by-design" argument: the model is provably a single
shared backbone with a low-dimensional continuous perturbation, NOT a field-indexed ensemble
(the Task-3 single-model condition).

Distinct from the FiLM cohort (B-3.5/3.8/1.4 modulate ACTIVATIONS) — here the field modulates
WEIGHTS via a low-rank delta. ``model_kwargs.lora_rank=0`` drops the adapter -> a field-BLIND
shared backbone (the one-knob ablation; non-degenerate — it still translates, just
field-agnostically). Magnitude-only (1->1); both ``field_strength`` and ``field_strength_target``
are REQUIRED forward kwargs (no default) so the Tier-2 probe injects them.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from spectramr.models.blocks.field_conditioned_lora import FieldConditionedLoRAConv2d
from spectramr.models.registry import register_model

# Fixed conditioning vector for the param-matched field-BLIND control (``freeze_field``):
# log10 of a 1.5 T reference, in-range for the {0.1..7}T cohort and NON-ZERO so the rank-r
# adapter stays trainable (a zero field with the bias-free alpha_mlp would give alpha=0 ->
# delta=0, collapsing to the rank-0 backbone with dead, untrainable adapter params).
_FROZEN_FIELD_LOG10 = math.log10(1.5)


@register_model(name="lora_modulation_net", training_mode="lora_modulation")
class LoRAModulationNet(nn.Module):
    """Shared backbone + continuous field-conditioned low-rank weight modulation (B-3.6)."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        width: int = 48,
        n_blocks: int = 3,
        lora_rank: int = 4,
        kernel_size: int = 3,
        freeze_field: bool = False,
    ) -> None:
        super().__init__()
        if in_channels != 1 or out_channels != 1:
            raise ValueError(
                f"LoRAModulationNet is magnitude-only (1->1); got in={in_channels}, out={out_channels}."
            )
        if n_blocks < 1:
            raise ValueError(f"n_blocks must be >= 1; got {n_blocks}.")
        if not isinstance(freeze_field, bool):
            raise ValueError(f"freeze_field must be a bool; got {type(freeze_field).__name__}.")
        self.lora_rank = int(lora_rank)
        self.width = int(width)
        self.kernel_size = int(kernel_size)
        # PARAM-MATCHED field-BLIND control (#17): keep the full rank-r adapter (identical
        # parameters to the field-conditioned arm) but freeze the modulation argument to a
        # constant, so the adapter learns ONE field-agnostic low-rank residual. rank=4-vs-
        # frozen isolates the field-conditioning MECHANISM (exact param match); rank=4-vs-
        # rank=0 isolates mechanism + capacity. Only meaningful when lora_rank > 0.
        self.freeze_field = bool(freeze_field)

        def _lora(cin: int, cout: int, k: int = kernel_size) -> FieldConditionedLoRAConv2d:
            return FieldConditionedLoRAConv2d(cin, cout, k, rank=self.lora_rank, field_dim=2)

        self.lift = _lora(1, width)
        self.block_convs = nn.ModuleList(_lora(width, width) for _ in range(2 * n_blocks))
        self.norms = nn.ModuleList(nn.GroupNorm(min(8, width), width) for _ in range(2 * n_blocks))
        self.act = nn.SiLU()
        self.head = _lora(width, 1, k=1)
        self.n_blocks = int(n_blocks)

    def lora_rank_fraction(self) -> float:
        """Compliance bound r / max-rank(W) for the dominant width->width convs.

        A conv weight reshaped to a matrix is ``[out_ch, in_ch*k*k]`` with maximal rank
        ``min(out_ch, in_ch*k*k)``. For the binding width->width blocks this is
        ``min(width, width*k**2) == width``, so the honest low-rank fraction is ``r/width``
        (e.g. 4/48 = 8.3% at production). The earlier denominator (the conv ELEMENT count
        ~in_ch*k*k) made the printed bound misleadingly tiny (<1%); this reports the true
        rank ceiling. ``lora_rank`` itself is stamped into ``resolved_config.json``; the
        :class:`LoRAModulationStrategy` logs this derived bound at setup (pitfall #15 obligation c).
        """
        max_rank = min(self.width, self.width * self.kernel_size**2)
        return self.lora_rank / max(max_rank, 1)

    def forward(
        self,
        x: torch.Tensor,
        *,
        field_strength: torch.Tensor,
        field_strength_target: torch.Tensor,
        **_: Any,
    ) -> torch.Tensor:
        if torch.is_complex(x):
            raise ValueError("LoRAModulationNet expects magnitude (real) input.")
        b_s = torch.log10(field_strength.reshape(-1, 1).float().clamp_min(1e-3))
        b_t = torch.log10(field_strength_target.reshape(-1, 1).float().clamp_min(1e-3))
        field = torch.cat([b_s, b_t], dim=1)  # [B, 2] — smooth modulation argument
        if field.shape[0] != x.shape[0]:
            # Reject a size-1 field silently broadcasting one (b_s,b_t) across the whole
            # batch (same modulation for every sample) — a silent miswiring (pitfall #9).
            raise ValueError(
                f"field batch {field.shape[0]} != input batch {x.shape[0]}; "
                "pass one (field_strength, field_strength_target) pair per sample."
            )
        if self.freeze_field:
            # Param-matched field-BLIND control: overwrite the per-sample field with a fixed
            # non-zero constant so the rank-r adapter capacity is present & trainable but the
            # modulation cannot depend on (b_s, b_t). val_field_sensitivity reads ~0 here.
            field = torch.full_like(field, _FROZEN_FIELD_LOG10)
        h = self.act(self.lift(x, field))
        for conv, norm in zip(self.block_convs, self.norms, strict=True):
            h = self.act(norm(conv(h, field)))
        return self.head(h, field)
