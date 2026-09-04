"""Conditioning injection seam (2026-05-26).

A single FiLM seam that turns a fused conditioning vector into per-channel
scale/shift ``(g, b)`` and modulates a feature map: ``h ↦ (1 + g) ⊙ h + b``. The FiLM head
is zero-initialised so a freshly-added conditioner is the **identity** at start
— dropping :class:`AdaptiveConditioner` into an existing backbone never
perturbs it until the FiLM head learns to. This is deliberately separate from
``models/blocks/hypernet_api.py::HypernetWrappedModule`` (which swaps whole
weight buffers) so neither path can break the other.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from spectramr.models.conditioning.context import ConditioningContext
from spectramr.models.conditioning.encoders import ConditioningEncoderBank


class ConditioningInjector(nn.Module):
    """FiLM modulation of a feature map from a conditioning vector.

    Args:
        cond_dim: dimension of the incoming conditioning vector ``c``.
        num_features: channel count ``C`` of the feature map to modulate.
        hidden: hidden width of the FiLM MLP (defaults to ``max(cond_dim, C)``).
    """

    def __init__(self, cond_dim: int, num_features: int, hidden: int | None = None) -> None:
        super().__init__()
        self.num_features = int(num_features)
        hidden = hidden or max(int(cond_dim), int(num_features))
        self.to_film = nn.Sequential(
            nn.Linear(int(cond_dim), hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * self.num_features),
        )
        # Identity init: scale=0, shift=0 ⇒ (1+scale)*h + shift = h. A new
        # conditioner does not disturb the host model until these weights move.
        nn.init.zeros_(self.to_film[-1].weight)
        nn.init.zeros_(self.to_film[-1].bias)

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.to_film(c).chunk(2, dim=-1)
        shape = [h.shape[0], self.num_features] + [1] * (h.dim() - 2)
        gamma = gamma.reshape(*shape)
        beta = beta.reshape(*shape)
        return (1.0 + gamma) * h + beta


class AdaptiveConditioner(nn.Module):
    """The user-facing seam: encode a context, then FiLM-modulate a feature map.

    Wraps a :class:`ConditioningEncoderBank` (the active sources) and a
    :class:`ConditioningInjector`. A model conditions one of its feature maps
    with ``h = conditioner(h, ctx)``; ``encode(ctx)`` exposes the raw vector for
    cross-attention or hypernetwork use.

    Args:
        sources: context-source names to encode (registry keys).
        num_features: channel count of the feature map to modulate.
        cond_dim: fused conditioning-vector width.
        fusion: ``"concat"`` or ``"sum"`` (see ConditioningEncoderBank).
        encoder_kwargs: per-source encoder constructor overrides.
    """

    def __init__(
        self,
        sources: list[str],
        num_features: int,
        cond_dim: int = 128,
        fusion: str = "concat",
        encoder_kwargs: dict[str, dict[str, Any]] | None = None,
        cond_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.bank = ConditioningEncoderBank(
            sources, embed_dim=cond_dim, fusion=fusion, encoder_kwargs=encoder_kwargs
        )
        self.injector = ConditioningInjector(cond_dim, num_features)
        self.cond_dropout = float(cond_dropout)
        # Learnable "unconditional" embedding used when the conditioning is
        # dropped (CFG training) or forced off (the unconditional branch at
        # sampling). Zero-init keeps the whole conditioner at identity at start.
        self.null_token = nn.Parameter(torch.zeros(int(cond_dim)))

    @classmethod
    def from_config(cls, cfg: Any, num_features: int) -> AdaptiveConditioner | None:
        """Build from a ``ConditioningConfig`` (SSOT config→runtime mapping).

        Returns ``None`` when conditioning is disabled / has no sources, so a
        host model can do ``self.cond = AdaptiveConditioner.from_config(...)``
        and simply skip modulation when it is ``None``.
        """
        if not getattr(cfg, "enabled", False) or not getattr(cfg, "sources", None):
            return None
        # Wire + validate the advertised `injection` knob (pitfall #15): the
        # only implemented injection seam is FiLM (ConditioningInjector), so an
        # unknown value must RAISE rather than silently degrade. The schema
        # already constrains it to Literal["film"]; this is defense-in-depth so
        # the runtime never silently no-ops if a second value is added upstream.
        injection = getattr(cfg, "injection", "film")
        if injection != "film":
            raise ValueError(
                f"Unsupported conditioning.injection={injection!r}; "
                "only 'film' is implemented (ConditioningInjector)."
            )
        return cls(
            sources=list(cfg.sources),
            num_features=num_features,
            cond_dim=cfg.embed_dim,
            fusion=cfg.fusion,
            encoder_kwargs=dict(getattr(cfg, "encoder_kwargs", {}) or {}),
        )

    def encode(self, ctx: ConditioningContext) -> torch.Tensor:
        """Return the fused conditioning vector ``[B, cond_dim]`` (no dropout)."""
        return self.bank(ctx)

    def forward(
        self,
        h: torch.Tensor,
        ctx: ConditioningContext,
        force_uncond: bool = False,
    ) -> torch.Tensor:
        """Modulate ``h`` with the encoded context.

        ``force_uncond`` replaces the conditioning with the learned null token
        (the unconditional branch for classifier-free guidance at sampling).
        During training, ``cond_dropout`` independently replaces a per-sample
        fraction with the null token so the model also learns the
        unconditional score and stays robust to missing conditions.
        """
        c = self.bank(ctx)
        b = c.shape[0]
        null = self.null_token.unsqueeze(0).expand(b, -1)
        if force_uncond:
            c = null
        elif self.training and self.cond_dropout > 0.0:
            drop = (torch.rand(b, device=c.device) < self.cond_dropout).unsqueeze(1)
            c = torch.where(drop, null, c)
        return self.injector(h, c)


__all__ = ["AdaptiveConditioner", "ConditioningInjector"]
