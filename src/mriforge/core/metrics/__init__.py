"""Core Metrics Module - Unified access to all evaluation metrics.

This module provides a clean API for accessing metrics::

    from mriforge.core.metrics import get_metric, list_available

NOTE: Uses lazy imports to avoid circular dependencies with models.evaluation.

Registration: every module under ``mriforge.core.metrics`` is walked on
package import so each ``@register_metric`` decorator fires exactly
once. The previous hand-maintained import list was easy to drift from
the file system — audit ``13_metrics.md`` F11 (D7) caught
``hallucination_metrics`` missing for that reason. ``pkgutil.walk_packages``
removes the drift surface entirely.
"""

import importlib
import logging
import pkgutil
import sys

logger = logging.getLogger(__name__)

#: Walked metric modules that were skipped because a *third-party* optional
#: dependency was missing (walked-module name → missing top-level dep name).
#: An in-repo (``mriforge.*``) import failure is NEVER recorded here — it is
#: re-raised (see the walk below) so a broken metric module can never vanish
#: silently. Audits/tests introspect this dict instead of parsing warning
#: logs. CLAUDE.md pitfall #9 (no silent fallbacks) / #15 (self-describing).
MISSING_OPTIONAL_DEPS: dict[str, str] = {}

_PACKAGE_NAME = __name__  # "mriforge.core.metrics"


def _record_or_raise_walk_import_error(module_name: str, exc: ImportError) -> None:
    """Classify a walk-discovery ``ImportError`` and act on the root cause.

    Split on ``exc.name`` (the module whose absence caused the failure):

    * An **in-repo** failure (``exc.name`` starts with ``mriforge`` — or is
      falsy, which means an unattributable import-machinery error) is the
      *poisoning* class: a broken metric module, a bad relative import, a
      circular import. **Re-raise** — such a module must NEVER vanish
      silently, or its ``@register_metric`` decorators disappear from the
      registry with no trace (audit 13 F11 / F2 root cause).
    * A **third-party** root cause (torchmetrics, piq, pyradiomics, …) is a
      degraded-environment condition: warn AND record it in
      :data:`MISSING_OPTIONAL_DEPS` so audits can introspect the skipped set
      rather than grepping logs. CLAUDE.md pitfall #9.
    """
    root = exc.name or ""
    if not root or root == "mriforge" or root.startswith("mriforge."):
        raise exc
    MISSING_OPTIONAL_DEPS[module_name] = root
    logger.warning(
        "[METRICS] Skipping %s during walk-discovery: missing optional "
        "dependency %r (%s). Any @register_metric decorators in that "
        "module will not fire; install the dependency to enable them. "
        "Recorded in mriforge.core.metrics.MISSING_OPTIONAL_DEPS. "
        "Audit 13 F11.",
        module_name,
        root,
        exc,
    )


def _on_walk_package_error(package_name: str) -> None:
    """Classify a failure to import a **sub-package** during walk-discovery.

    ``pkgutil.walk_packages`` recurses by *importing* each sub-package it
    finds, and that import happens inside pkgutil -- not in the loop body
    below, whose ``except ImportError`` therefore never sees it. With the
    default ``onerror=None`` pkgutil swallows the ``ImportError`` and abandons
    the entire sub-tree: every ``@register_metric`` beneath the broken
    sub-package disappears with no error, no warning, and exit 0. Measured
    2026-08-28 by planting ``raise ImportError`` in ``connectivity``: the
    registry went 211 -> 210 metrics and the process still exited 0.

    pkgutil passes the name only, so the live exception comes from the active
    ``except`` block. A non-``ImportError`` is re-raised, which is what
    ``onerror=None`` did and what the loop body does: this callback widens
    *which imports get classified*, never *what gets tolerated*.
    """
    exc = sys.exc_info()[1]
    if not isinstance(exc, ImportError):
        raise
    _record_or_raise_walk_import_error(package_name, exc)


def _discover_and_import_metric_modules() -> None:
    """Import every metric submodule so each ``@register_metric`` fires.

    We import via the package path so we don't accidentally re-import
    ``registry`` (already loaded by the imports below) and we never re-run on
    duplicate calls — Python's module cache handles idempotency. Extracted to
    a function so the walk-discovery error split is unit-testable with a
    monkeypatched ``pkgutil.walk_packages`` / ``importlib.import_module``.
    """
    for _, module_name, is_pkg in pkgutil.walk_packages(
        __path__, _PACKAGE_NAME + ".", onerror=_on_walk_package_error
    ):
        if is_pkg:
            continue
        # Skip the registry which we are already loading below.
        if module_name in (f"{_PACKAGE_NAME}.registry",):
            continue
        try:
            importlib.import_module(module_name)
        except ImportError as exc:
            _record_or_raise_walk_import_error(module_name, exc)


_discover_and_import_metric_modules()

from mriforge.core.metrics.registry import (
    IMetric,
    MetricsRegistry,
    compute_metric,
    get_metric,
    list_available,
    register_metric,
)

# NOTE (2026-08-28): three explicit sub-package imports stood here --
# ``quantitative``, ``quantitative.challenge_metrics`` and ``uncertainty`` --
# under the rationale that ``if is_pkg: continue`` above meant the walk never
# reached a sub-package's modules. That rationale is measured-false.
# ``pkgutil.walk_packages`` recurses INTO each sub-package (by importing it),
# so the loop body still yields every *module* underneath; ``is_pkg`` skips
# only the sub-package's own ``__init__``. Deleting all three changed the
# registry by zero names (211 -> 211).
#
# What they did buy was the *raise*: an explicit ``import`` propagates, whereas
# the walk's ``onerror=None`` swallowed the same failure. The walk yields 8
# sub-packages here, so that covered 2 of 8 and left 6 silent --
# ``connectivity``, ``distribution``, ``regions``, ``stein``,
# ``meta_evaluation`` and ``meta_evaluation.rankers``.
# ``onerror=_on_walk_package_error`` now covers all eight from one owner, so
# keeping these lines as well would be a second enforcer of the same invariant
# (CLAUDE.md non-negotiable 17).
# Out-of-tree metrics: fire @register_metric decorators from entry-points /
# MRIFORGE_PLUGINS modules. mriforge.plugins is stdlib-only — no layer violation.
# A bad MRIFORGE_PLUGINS token raises (fail-fast, pitfall #15).
from mriforge.plugins import discover_plugins as _discover_plugins  # noqa: E402

_discover_plugins("mriforge.metrics")


def get_evaluation_metrics(*args, **kwargs):
    """Lazy import EvaluationMetrics to avoid circular import."""
    from mriforge.core.metrics.evaluation import EvaluationMetrics

    return EvaluationMetrics(*args, **kwargs)


__all__ = [
    # Walk-discovery introspection
    "MISSING_OPTIONAL_DEPS",
    # Registry API (new pattern)
    "IMetric",
    "MetricsRegistry",
    "register_metric",
    "get_metric",
    "compute_metric",
    "list_available",
    # Lazy accessors (avoid circular imports)
    "get_evaluation_metrics",
]
