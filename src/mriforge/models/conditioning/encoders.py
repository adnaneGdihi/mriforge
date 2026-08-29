"""Conditioning encoders + registry (2026-05-26).

Each encoder reads ONE source from a :class:`ConditioningContext` and emits a
``[B, embed_dim]`` embedding. A :class:`ConditioningEncoderBank` instantiates
the encoders named in a config, runs each on the context, and fuses them
(``concat`` → linear, or ``sum``) into a single conditioning vector — the
joint ``(t, severity, acquisition, …)`` co-embedding. Registry dispatch mirrors
``register_hypernet`` / ``register_loss`` so YAML can select sources by name and
an illegal source is rejected at construction time (CLAUDE.md pitfall #9).

Spatial coordinate (INR) positional encoding is intentionally NOT here — the
canonical home is ``models/blocks/embeddings.py::CoordinatePositionalEncoding``;
these encoders cover the *global* (per-sample) condition sources only.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from mriforge.models.conditioning.context import ConditioningContext

_CONDITIONER_REGISTRY: dict[str, type[ConditionEncoder]] = {}


def register_conditioner(name: str):
    """Decorator: register a :class:`ConditionEncoder` subclass under ``name``."""

    def deco(cls: type[ConditionEncoder]) -> type[ConditionEncoder]:
        if name in _CONDITIONER_REGISTRY:
            raise ValueError(f"Conditioner {name!r} already registered.")
        _CONDITIONER_REGISTRY[name] = cls
        return cls

    return deco


def get_conditioner(name: str, **kwargs: Any) -> ConditionEncoder:
    if name not in _CONDITIONER_REGISTRY:
        raise KeyError(f"Unknown conditioner {name!r}. Available: {sorted(_CONDITIONER_REGISTRY)}")
    return _CONDITIONER_REGISTRY[name](**kwargs)


def _sinusoidal(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of a ``[N]`` scalar tensor → ``[N, dim]``."""
    half = dim // 2
    device = t.device
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=device).float() / max(half - 1, 1)
    )
    args = t.reshape(-1, 1).float() * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:  # pad odd dims
        emb = torch.cat([emb, torch.zeros(emb.shape[0], 1, device=device)], dim=-1)
    return emb


class ConditionEncoder(nn.Module):
    """Base: read one context source, emit a ``[B, embed_dim]`` embedding."""

    source: str = ""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)

    def _require(self, ctx: ConditioningContext) -> torch.Tensor:
        t = ctx.get(self.source)
        if t is None:
            raise ValueError(
                f"{type(self).__name__} requires ConditioningContext source "
                f"{self.source!r}, but it is not populated."
            )
        return t

    def _mlp(self, in_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(in_dim, self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )


@register_conditioner("diffusion_t")
class SinusoidalTimeEncoder(ConditionEncoder):
    """Axis-1 generative clock: sinusoidal embedding of the diffusion timestep."""

    source = "diffusion_t"

    def __init__(self, embed_dim: int = 128) -> None:
        super().__init__(embed_dim)
        self.proj = self._mlp(embed_dim)

    def forward(self, ctx: ConditioningContext) -> torch.Tensor:
        t = self._require(ctx).reshape(-1)
        return self.proj(_sinusoidal(t, self.embed_dim))


@register_conditioner("severity_vec")
class SeverityTokenEncoder(ConditionEncoder):
    """Degradation-aware token: ``[B, K]`` severities → ``[B, embed_dim]``.

    The physics-grounded analogue of a PromptIR degradation prompt — the token
    is the *measured* per-degradation severity (motion, b0, b1, noise, accel),
    not a learned-from-scratch code, so the network adapts across regimes.
    """

    source = "severity_vec"

    def __init__(self, embed_dim: int = 128, num_severities: int = 5) -> None:
        super().__init__(embed_dim)
        self.num_severities = int(num_severities)
        self.proj = self._mlp(self.num_severities)

    def forward(self, ctx: ConditioningContext) -> torch.Tensor:
        return self.proj(self._require(ctx).float())


@register_conditioner("acq_time_tau")
class AcquisitionTimeEncoder(ConditionEncoder):
    """Axis-2 physical clock: Fourier-encode the phase-encode/shot order ``τ``.

    ``acq_time_tau`` is ``[B, N_lines]`` of (normalised) acquisition times; each
    line is sinusoidally encoded and mean-pooled, giving a global
    acquisition-order embedding — the variable motion is causally a function of.
    """

    source = "acq_time_tau"

    def __init__(self, embed_dim: int = 128) -> None:
        super().__init__(embed_dim)
        self.proj = self._mlp(embed_dim)

    def forward(self, ctx: ConditioningContext) -> torch.Tensor:
        tau = self._require(ctx).float()  # [B, N]
        b, n = tau.shape
        enc = _sinusoidal(tau.reshape(b * n), self.embed_dim).reshape(b, n, self.embed_dim)
        return self.proj(enc.mean(dim=1))


@register_conditioner("acquisition")
class AcquisitionParamEncoder(ConditionEncoder):
    """Acquisition parameters ``(TE, TR, TI, FA, B0)`` → ``[B, embed_dim]``."""

    source = "acquisition"

    def __init__(self, embed_dim: int = 128, num_params: int = 5) -> None:
        super().__init__(embed_dim)
        self.num_params = int(num_params)
        self.proj = self._mlp(self.num_params)

    def forward(self, ctx: ConditioningContext) -> torch.Tensor:
        return self.proj(self._require(ctx).float())


@register_conditioner("contrast_id")
class ContrastEncoder(ConditionEncoder):
    """Categorical contrast index → learned ``[B, embed_dim]`` embedding."""

    source = "contrast_id"

    def __init__(self, embed_dim: int = 128, num_contrasts: int = 8) -> None:
        super().__init__(embed_dim)
        self.embedding = nn.Embedding(int(num_contrasts), self.embed_dim)

    def forward(self, ctx: ConditioningContext) -> torch.Tensor:
        return self.embedding(self._require(ctx).reshape(-1).long())


@register_conditioner("motion_pose")
class MotionPoseEncoder(ConditionEncoder):
    """Per-line rigid pose ``θ`` ``[B, 3, N_lines]`` → ``[B, embed_dim]``.

    Encodes the estimated per-phase-encode-line pose (Levac 2024 / ISMRM 2023
    motion-aware conditioning) by projecting each line's 3-DoF pose and
    mean-pooling over lines.
    """

    source = "motion_pose"

    def __init__(self, embed_dim: int = 128, pose_dim: int = 3) -> None:
        super().__init__(embed_dim)
        self.pose_dim = int(pose_dim)
        self.line = nn.Linear(self.pose_dim, self.embed_dim)
        self.proj = self._mlp(self.embed_dim)

    def forward(self, ctx: ConditioningContext) -> torch.Tensor:
        pose = self._require(ctx).float()  # [B, 3, N]
        per_line = self.line(pose.transpose(1, 2))  # [B, N, embed_dim]
        return self.proj(per_line.mean(dim=1))


@register_conditioner("field_strength")
class FieldStrengthEncoder(ConditionEncoder):
    """Continuous main-field strength B0 (Tesla) → ``[B, embed_dim]``.

    Sinusoidally encoded (like ``diffusion_t``) so nearby field strengths map to
    nearby embeddings and a single translator can span ULF-to-HF (0.064-7 T)
    rather than training one model per field pairing.
    """

    source = "field_strength"

    def __init__(self, embed_dim: int = 128) -> None:
        super().__init__(embed_dim)
        self.proj = self._mlp(embed_dim)

    def forward(self, ctx: ConditioningContext) -> torch.Tensor:
        b0 = self._require(ctx).reshape(-1).float()
        return self.proj(_sinusoidal(b0, self.embed_dim))


@register_conditioner("scanner_id")
class ScannerEncoder(ConditionEncoder):
    """Categorical scanner / vendor index → learned ``[B, embed_dim]`` embedding.

    For cross-vendor harmonisation; pairs with ``batch['scanner_id']`` (exposed
    via ``data.expose_scanner_id``) and the ``cross_scanner_t1t2_concordance``
    metric. Combine with an adversarial/GRL head for vendor-invariant features.
    """

    source = "scanner_id"

    def __init__(self, embed_dim: int = 128, num_scanners: int = 16) -> None:
        super().__init__(embed_dim)
        self.embedding = nn.Embedding(int(num_scanners), self.embed_dim)

    def forward(self, ctx: ConditioningContext) -> torch.Tensor:
        return self.embedding(self._require(ctx).reshape(-1).long())


@register_conditioner("site_id")
class SiteEncoder(ConditionEncoder):
    """Categorical acquisition-site index → learned ``[B, embed_dim]`` embedding.

    For multi-site / federated conditioning; pairs with ``data.expose_site_id``
    and the site ``Balancer``. Use with an adversarial/GRL head for
    site-invariant reconstruction.
    """

    source = "site_id"

    def __init__(self, embed_dim: int = 128, num_sites: int = 32) -> None:
        super().__init__(embed_dim)
        self.embedding = nn.Embedding(int(num_sites), self.embed_dim)

    def forward(self, ctx: ConditioningContext) -> torch.Tensor:
        return self.embedding(self._require(ctx).reshape(-1).long())


class ConditioningEncoderBank(nn.Module):
    """Assemble the active encoders and fuse them into one conditioning vector.

    Args:
        sources: context-source names (registry keys) to encode.
        embed_dim: dimension of each per-source embedding AND the fused output.
        fusion: ``"concat"`` (concatenate → linear) or ``"sum"``.
        encoder_kwargs: optional per-source constructor overrides
            (e.g. ``{"severity_vec": {"num_severities": 6}}``).
    """

    def __init__(
        self,
        sources: list[str],
        embed_dim: int = 128,
        fusion: str = "concat",
        encoder_kwargs: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        super().__init__()
        if not sources:
            raise ValueError("ConditioningEncoderBank needs at least one source.")
        if fusion not in ("concat", "sum"):
            raise ValueError(f"fusion must be 'concat' or 'sum', got {fusion!r}")
        self.embed_dim = int(embed_dim)
        self.fusion = fusion
        encoder_kwargs = encoder_kwargs or {}
        self.encoders = nn.ModuleDict(
            {
                name: get_conditioner(name, embed_dim=embed_dim, **encoder_kwargs.get(name, {}))
                for name in sources
            }
        )
        self.fuse = (
            nn.Linear(self.embed_dim * len(sources), self.embed_dim)
            if fusion == "concat"
            else nn.Identity()
        )

    def forward(self, ctx: ConditioningContext) -> torch.Tensor:
        embs = [enc(ctx) for enc in self.encoders.values()]
        if self.fusion == "sum":
            return torch.stack(embs, dim=0).sum(dim=0)
        return self.fuse(torch.cat(embs, dim=-1))


__all__ = [
    "AcquisitionParamEncoder",
    "AcquisitionTimeEncoder",
    "ConditionEncoder",
    "ConditioningEncoderBank",
    "ContrastEncoder",
    "FieldStrengthEncoder",
    "MotionPoseEncoder",
    "ScannerEncoder",
    "SeverityTokenEncoder",
    "SinusoidalTimeEncoder",
    "SiteEncoder",
    "get_conditioner",
    "register_conditioner",
]
