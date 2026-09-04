"""Adapter registry parallel to ``src/models/registry.py``.

Concrete adapters land in Phase 3 (slice_3d_to_2d, rss_coils_to_magnitude,
etc.). Phase 1 ships only the registry, decorator, and protocol so the
audit's adapter-chain composer (Phase 2) and Spec Card (Phase 1) can
already reference adapter capabilities even before any are registered.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from spectramr.models.capabilities import (
    VALID_INSERTION_POINTS,
    AdapterCapabilities,
)


@runtime_checkable
class Adapter(Protocol):
    """Minimal contract for adapter classes.

    Concrete adapters can subclass ``torch.nn.Module`` and implement
    ``forward`` — Adapter is intentionally minimal so ad-hoc functional
    adapters (e.g. a stateless slicer) need no class boilerplate.
    """

    def __call__(self, x: Any, /, *args: Any, **kwargs: Any) -> Any: ...


ADAPTER_REGISTRY: dict[str, dict[str, Any]] = {}


def register_adapter(
    name: str,
    *,
    bridges_from: dict[str, object],
    bridges_to: dict[str, object],
    insertion_points: tuple[str, ...],
    invertible: bool = False,
    side_effects: tuple[str, ...] = (),
):
    """Register an adapter class or callable.

    Args:
        name: Unique adapter name.
        bridges_from: Capability shape the adapter consumes
            (e.g. ``{"spatial_dims": 3, "domain": "image"}``).
        bridges_to: Capability shape the adapter produces.
        insertion_points: Tuple of allowable hooks; each must be in
            :data:`spectramr.models.capabilities.VALID_INSERTION_POINTS`.
        invertible: True if this adapter has a registered inverse —
            audit will require both halves of an invertible pair when
            the chain crosses model boundaries.
        side_effects: Human-readable side effects surfaced verbatim on
            the Spec Card. Examples: ``"batch_multiplied_by_depth"``,
            ``"loses_inter_slice_context"``, ``"discards_phase"``.

    Raises:
        ValueError: If any insertion point is not recognized.
    """
    bad = [p for p in insertion_points if p not in VALID_INSERTION_POINTS]
    if bad:
        raise ValueError(
            f"Adapter {name!r} declares unknown insertion points {bad}. "
            f"Valid: {sorted(VALID_INSERTION_POINTS)}"
        )

    capabilities = AdapterCapabilities(
        bridges_from=dict(bridges_from),
        bridges_to=dict(bridges_to),
        invertible=invertible,
        side_effects=tuple(side_effects),
        insertion_points=tuple(insertion_points),
    )

    def decorator(cls: type[Any]):
        ADAPTER_REGISTRY[name] = {
            "class": cls,
            "capabilities": capabilities,
        }
        return cls

    return decorator


def get_adapter(name: str) -> type[Any]:
    if name not in ADAPTER_REGISTRY:
        raise ValueError(
            f"Adapter {name!r} not registered. Available: {sorted(ADAPTER_REGISTRY.keys())}"
        )
    return ADAPTER_REGISTRY[name]["class"]


def get_adapter_capabilities(name: str) -> AdapterCapabilities | None:
    entry = ADAPTER_REGISTRY.get(name)
    return entry["capabilities"] if entry else None


def list_adapters() -> dict[str, dict[str, Any]]:
    return ADAPTER_REGISTRY.copy()


__all__ = [
    "ADAPTER_REGISTRY",
    "Adapter",
    "get_adapter",
    "get_adapter_capabilities",
    "list_adapters",
    "register_adapter",
]
