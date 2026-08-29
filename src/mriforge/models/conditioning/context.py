"""Canonical conditioning context — the single source of truth for every
condition source consumed by VF / digital-twin models (2026-05-26).

Three distinct "time/position" axes are kept separate on purpose:

* ``diffusion_t``   — axis 1, the *generative* clock (diffusion timestep).
* ``acq_time_tau``  — axis 2, the *physical* clock (phase-encode / shot
  order); this is the variable motion is a function of, NOT ``diffusion_t``.
* ``coords``        — axis 3, spatial (x, y[, coil]) coordinates for INRs.

Conflating these into ad-hoc embeddings is the root of most VF conditioning
gaps; a strategy populates only the active fields and a model consumes one
object via a single ``condition=`` kwarg. The context is frozen — build a new
one with :func:`dataclasses.replace` (``.to`` does this for device moves).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields
from typing import Any

import torch

# Tensor-bearing fields (everything except ``extra``); ``active_sources`` and
# ``to`` iterate this so adding a field updates both automatically.
_TENSOR_FIELDS = (
    "diffusion_t",
    "acq_time_tau",
    "coords",
    "severity_vec",
    "acquisition",
    "contrast_id",
    "motion_pose",
    "field_strength",
    "scanner_id",
    "site_id",
)


@dataclass(frozen=True)
class ConditioningContext:
    """Bundle of optional conditioning tensors.

    Args:
        diffusion_t: ``[B]`` or ``[B, 1]`` diffusion timestep (the generative
            clock; also the twin ``corruption_factor`` level).
        acq_time_tau: ``[B, N_lines]`` phase-encode / shot acquisition order
            (the physical clock that drives motion).
        coords: ``[B, ..., D]`` spatial coordinates for INR/positional encoding.
        severity_vec: ``[B, K]`` per-degradation severities
            (e.g. motion, b0, b1, noise, accel) — the degradation-aware token.
        acquisition: ``[B, 5]`` acquisition params ``(TE, TR, TI, FA, B0)``.
        contrast_id: ``[B]`` categorical contrast index.
        motion_pose: ``[B, 3, N_lines]`` estimated per-line rigid pose ``θ``.
        field_strength: ``[B]`` or ``[B, 1]`` main-field strength B0 in Tesla
            (continuous; ULF↔HF translation, e.g. 0.064 / 0.3 / 1.5 / 3 T).
        scanner_id: ``[B]`` categorical scanner / vendor index (cross-vendor
            harmonisation; pairs with ``data.expose_scanner_id``).
        site_id: ``[B]`` categorical acquisition-site index (multi-site /
            federated; pairs with ``data.expose_site_id``).
        extra: open-ended named tensors for experiment-specific conditions.
    """

    diffusion_t: torch.Tensor | None = None
    acq_time_tau: torch.Tensor | None = None
    coords: torch.Tensor | None = None
    severity_vec: torch.Tensor | None = None
    acquisition: torch.Tensor | None = None
    contrast_id: torch.Tensor | None = None
    motion_pose: torch.Tensor | None = None
    field_strength: torch.Tensor | None = None
    scanner_id: torch.Tensor | None = None
    site_id: torch.Tensor | None = None
    extra: dict[str, torch.Tensor] = field(default_factory=dict)

    @property
    def active_sources(self) -> tuple[str, ...]:
        """Names of the condition sources that are populated (non-None)."""
        names = [n for n in _TENSOR_FIELDS if getattr(self, n) is not None]
        names.extend(self.extra.keys())
        return tuple(names)

    @property
    def batch_size(self) -> int | None:
        """Batch size inferred from the first populated tensor, else None."""
        for name in _TENSOR_FIELDS:
            t = getattr(self, name)
            if t is not None:
                return int(t.shape[0])
        for t in self.extra.values():
            return int(t.shape[0])
        return None

    def has(self, name: str) -> bool:
        """True if ``name`` is a populated tensor field or extra key."""
        return name in self.active_sources

    def get(self, name: str) -> torch.Tensor | None:
        """Fetch a source by name from either the fields or ``extra``."""
        if name in _TENSOR_FIELDS:
            return getattr(self, name)  # type: ignore[no-any-return]
        return self.extra.get(name)

    def to(self, device: Any) -> ConditioningContext:
        """Return a copy with every tensor moved to ``device`` (frozen-safe)."""
        moved = {
            n: (t.to(device) if (t := getattr(self, n)) is not None else None)
            for n in _TENSOR_FIELDS
        }
        moved["extra"] = {k: v.to(device) for k, v in self.extra.items()}
        return dataclasses.replace(self, **moved)


__all__ = ["ConditioningContext"]


# Defensive: keep the iteration tuple in sync with the dataclass definition so
# a renamed/added field can never silently drop out of active_sources / to.
_declared = {f.name for f in fields(ConditioningContext)} - {"extra"}
assert set(_TENSOR_FIELDS) == _declared, (
    f"_TENSOR_FIELDS {_TENSOR_FIELDS} out of sync with dataclass {_declared}"
)
