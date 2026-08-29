"""Generic hypernetwork API (PR-23 / Part-I E).

Plan: TODO/backlog_paradigm_expansion_roadmap.md §PR-23.

Hypernetworks generate the parameters of a target ("primary") network
conditioned on context (B0, scout image, vendor id, ...). Existing
specific cases (X-Field-FM, SCAS) are now thin specialisations of the
generic ``Hypernet`` defined here.

Three components:

- :class:`Hypernet` — base class. ``forward(context) -> dict[str, Tensor]``.
  Subclasses implement ``_build_context_encoder`` and ``_param_heads``.
- :class:`HypernetWrappedModule` — wraps a primary module and replaces
  one or more weight buffers each forward.
- :func:`register_hypernet` — decorator + registry for plug-and-play
  use from YAML.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn

_HYPERNET_REGISTRY: dict[str, type[Hypernet]] = {}


def register_hypernet(name: str):
    """Decorator: register a Hypernet subclass under ``name``."""

    def deco(cls: type[Hypernet]) -> type[Hypernet]:
        if name in _HYPERNET_REGISTRY:
            raise ValueError(f"Hypernet {name!r} already registered.")
        _HYPERNET_REGISTRY[name] = cls
        return cls

    return deco


def get_hypernet(name: str, **kwargs: Any) -> Hypernet:
    if name not in _HYPERNET_REGISTRY:
        raise KeyError(f"Unknown hypernet {name!r}. Available: {sorted(_HYPERNET_REGISTRY)}")
    return _HYPERNET_REGISTRY[name](**kwargs)


class Hypernet(nn.Module):
    """Generic hypernetwork: context -> dict of generated parameters.

    Subclasses override ``_build_context_encoder`` and ``_param_heads``
    to specify how the context tensor is encoded and how each parameter
    head shapes the output. ``forward`` returns a dict mapping the
    target-parameter name to its generated tensor.
    """

    def __init__(
        self,
        context_dim: int,
        param_shapes: dict[str, tuple[int, ...]],
        *,
        embedding_dim: int = 128,
    ) -> None:
        super().__init__()
        self.context_dim = context_dim
        self.param_shapes = dict(param_shapes)
        self.embedding_dim = embedding_dim
        self.encoder = self._build_context_encoder(context_dim, embedding_dim)
        self.heads = nn.ModuleDict(
            {
                name: nn.Linear(embedding_dim, int(_prod(shape)))
                for name, shape in param_shapes.items()
            }
        )

    def _build_context_encoder(self, context_dim: int, embedding_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(context_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, context: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encoder(context)
        out: dict[str, torch.Tensor] = {}
        for name, shape in self.param_shapes.items():
            flat = self.heads[name](z)
            # Broadcast batch dimension: (B, prod(shape)) → (B, *shape).
            out[name] = flat.view(z.shape[0], *shape)
        return out


@register_hypernet("identity")
class IdentityHypernet(Hypernet):
    """No-op hypernet — emits zeros for every requested parameter."""

    def forward(self, context: torch.Tensor) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for name, shape in self.param_shapes.items():
            out[name] = torch.zeros(context.shape[0], *shape, device=context.device)
        return out


class HypernetWrappedModule(nn.Module):
    """Wrap a primary module so its forward swaps in hypernet-generated weights.

    Args:
        primary: the underlying ``nn.Module`` whose parameters are
            (partially) replaced each forward.
        hypernet: the ``Hypernet`` that emits parameter dicts.
        replace_keys: names of attributes / parameters in ``primary``
            to overwrite per forward. Each must be present in
            ``hypernet.param_shapes``.

    The wrapped module reads the context from
    ``kwargs["hypernet_context"]`` on every forward; if absent, the
    primary's own parameters are used (identity behaviour).
    """

    def __init__(
        self,
        primary: nn.Module,
        hypernet: Hypernet,
        replace_keys: Iterable[str],
    ) -> None:
        super().__init__()
        self.primary = primary
        self.hypernet = hypernet
        self.replace_keys = list(replace_keys)
        for k in self.replace_keys:
            if k not in hypernet.param_shapes:
                raise ValueError(
                    f"replace_keys[{k!r}] not in hypernet.param_shapes "
                    f"({list(hypernet.param_shapes)})"
                )

    def forward(
        self, *args: Any, hypernet_context: torch.Tensor | None = None, **kwargs: Any
    ) -> torch.Tensor:
        if hypernet_context is None:
            return self.primary(*args, **kwargs)
        delta = self.hypernet(hypernet_context)
        # Stash old values so we can restore them after forward.
        original: dict[str, torch.Tensor] = {}
        for k in self.replace_keys:
            if hasattr(self.primary, k):
                original[k] = getattr(self.primary, k)
                setattr(self.primary, k, delta[k])
        try:
            out = self.primary(*args, **kwargs)
        finally:
            for k, v in original.items():
                setattr(self.primary, k, v)
        return out


def _prod(shape: tuple[int, ...]) -> int:
    p = 1
    for s in shape:
        p *= s
    return p


__all__ = [
    "Hypernet",
    "HypernetWrappedModule",
    "IdentityHypernet",
    "get_hypernet",
    "register_hypernet",
]
