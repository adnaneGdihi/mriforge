"""Registry iterators for parametric tests.

Provides thin wrappers around the real MRIForge component registries so
test modules can iterate over *all* registered names without duplicating
import logic.

Registries accessed:

* **Models** — ``mriforge.models.registry.MODEL_REGISTRY`` (dict keyed
  by model name, values are dicts with ``"class"``, ``"mode"``, …).
* **Losses** — ``mriforge.models.losses.registry.LossRegistry._custom_losses``
  (class-level dict keyed by canonical loss name).
* **Metrics** — ``mriforge.core.metrics.registry.MetricsRegistry._metrics``
  (class-level dict keyed by canonical metric name).
* **Strategies** — ``mriforge.infrastructure.training.strategy_factory
  .TrainingStrategyFactory.STRATEGY_CLASS_PATHS`` (plain dict keyed by
  strategy alias).

All functions return ``list[str]`` so that callers can do::

    @pytest.mark.parametrize("name", all_models())
    def test_something(name):
        ...

Filter callables receive the *string key* only; metadata-based
filtering can be done by the caller after the full list is obtained.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def all_models(filter_fn: Callable[[str], bool] | None = None) -> list[str]:
    """Return every registered model name.

    Args:
        filter_fn: Optional predicate ``(name: str) -> bool``.  Only
            names for which the predicate returns ``True`` are included.

    Returns:
        Sorted list of model names currently in
        ``mriforge.models.registry.MODEL_REGISTRY``.
    """
    try:
        from mriforge.models.registry import MODEL_REGISTRY  # noqa: PLC0415
        keys = list(MODEL_REGISTRY.keys())
    except Exception as exc:
        logger.warning("Could not import MODEL_REGISTRY: %s", exc)
        keys = []

    if filter_fn is not None:
        keys = [k for k in keys if _safe_call(filter_fn, k)]

    return sorted(keys)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def all_losses(domain: str | None = None) -> list[str]:
    """Return every registered loss name.

    Args:
        domain: Optional domain filter (``"image"``, ``"kspace"``,
            ``"complex"``, ``"latent"``, ``"physics"``,
            ``"agnostic"``).  When provided, only losses whose metadata
            lists a matching domain are included.  Losses without domain
            metadata are included when *domain* is ``None``.

    Returns:
        Sorted list of canonical loss names currently in
        ``LossRegistry._custom_losses``.
    """
    try:
        from mriforge.models.losses.registry import LossRegistry  # noqa: PLC0415
        keys = list(LossRegistry._custom_losses.keys())
        if domain is not None:
            domain_meta = LossRegistry._loss_domains
            filtered = []
            for k in keys:
                meta = domain_meta.get(k)
                if meta is None:
                    # No domain metadata — include when filtering is loose
                    continue
                if meta.get("domain") == domain:
                    filtered.append(k)
            keys = filtered
    except Exception as exc:
        logger.warning("Could not import LossRegistry: %s", exc)
        keys = []

    return sorted(keys)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def all_metrics(kind: str | None = None) -> list[str]:
    """Return every registered metric name.

    Args:
        kind: Optional substring filter on the metric name (e.g.
            ``"ssim"``, ``"psnr"``).  The filter is case-insensitive and
            checks whether *kind* appears anywhere in the name.

    Returns:
        Sorted list of canonical metric names currently in
        ``MetricsRegistry._metrics``.
    """
    try:
        from mriforge.core.metrics.registry import MetricsRegistry  # noqa: PLC0415
        keys = list(MetricsRegistry._metrics.keys())
    except Exception as exc:
        logger.warning("Could not import MetricsRegistry: %s", exc)
        keys = []

    if kind is not None:
        kind_lower = kind.lower()
        keys = [k for k in keys if kind_lower in k.lower()]

    return sorted(keys)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def all_strategy_keys() -> list[str]:
    """Return every key registered in ``STRATEGY_CLASS_PATHS``.

    Note: ``STRATEGY_CLASS_PATHS`` contains both canonical strategy names
    and aliases that map to the same underlying class — both are
    returned here so that parametric tests cover the full dispatch
    surface.

    Returns:
        Sorted list of keys from
        ``TrainingStrategyFactory.STRATEGY_CLASS_PATHS``.
    """
    try:
        from mriforge.infrastructure.training.strategy_factory import (  # noqa: PLC0415
            TrainingStrategyFactory,
        )
        keys = list(TrainingStrategyFactory.STRATEGY_CLASS_PATHS.keys())
    except Exception as exc:
        logger.warning("Could not import TrainingStrategyFactory: %s", exc)
        keys = []

    return sorted(keys)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_call(fn: Callable[[str], bool], value: str) -> bool:
    """Call *fn* without raising; returns ``False`` on any exception."""
    try:
        return bool(fn(value))
    except Exception:
        return False
