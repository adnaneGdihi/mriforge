"""Out-of-tree plugin discovery — the single helper that lets a user register
custom models / losses / metrics / datasets defined OUTSIDE the ``spectramr``
source tree.

Three layers feed the registries:

1. **entry-points** (``importlib.metadata``, groups ``spectramr.models`` /
   ``spectramr.losses`` / ``spectramr.metrics`` / ``spectramr.datasets``) — for
   shareable, pip-installed
   plugin distributions. A broken third-party plugin **warns** rather than
   crashing an unrelated training run.
2. the **``SPECTRAMR_PLUGINS``** env var — ``os.pathsep``/whitespace-separated
   dotted module paths, imported once. A user-declared knob, so an unimportable
   token **raises** (pitfall #15 — read + validate + no silent skip).
3. **``config.plugins.paths``** — same raise-on-failure contract, applied at
   config construction via :func:`import_plugin_paths`.

Importing a plugin module is the side effect that fires its ``@register_*``
decorators; the registries then resolve the new names normally. Strategies are
handled separately (they are a hardcoded dotted-path dict, merged lazily in the
strategy factory) — see :data:`STRATEGY_ENTRY_POINT_GROUP`.

This module imports **only the standard library** so it can be called from any
layer (``models/``, ``core/metrics/``) without a leftward dependency violation
(CLAUDE.md layer-direction rule).
"""

from __future__ import annotations

import importlib
import importlib.metadata
import logging
import os
import re
from collections.abc import Iterable

logger = logging.getLogger(__name__)

#: Environment variable naming extra plugin module paths to import.
PLUGIN_ENV_VAR = "SPECTRAMR_PLUGINS"

#: Decorator-backed entry-point groups handled by :func:`discover_plugins`.
ENTRY_POINT_GROUPS: tuple[str, ...] = (
    "spectramr.models",
    "spectramr.losses",
    "spectramr.metrics",
    "spectramr.datasets",
)

#: Strategies are a dotted-path dict (not a decorator registry), so their
#: entry-point group is consumed by the strategy factory, not here.
STRATEGY_ENTRY_POINT_GROUP = "spectramr.strategies"

# --- discovery cache (idempotency) ----------------------------------------
#: Entry-point groups already scanned. Per GROUP, because scanning is the cost.
_discovered_groups: set[str] = set()
#: Plugin modules already imported. Per PATH, which is what makes re-reading the
#: environment on every call cheap -- and is why no separate "env done" latch is
#: needed (see :func:`_discover_env_paths`).
_imported_paths: set[str] = set()


class PluginImportError(ImportError):
    """A user-declared plugin module failed to import.

    Raised for ``SPECTRAMR_PLUGINS`` tokens and ``config.plugins.paths`` entries
    (explicit, user-declared dependencies) — never for entry-points, where a
    failure is warned instead so a broken third-party plugin cannot crash an
    unrelated run.
    """


def _reset_discovery_cache() -> None:
    """Clear the idempotency cache (test seam only)."""
    _discovered_groups.clear()
    _imported_paths.clear()


def _import_path(path: str, *, source: str) -> None:
    """Import a single dotted module path, raising on failure (user-declared)."""
    path = path.strip()
    if not path or path in _imported_paths:
        return
    try:
        module = importlib.import_module(path)
    except Exception as exc:  # re-raised below as a typed plugin error
        raise PluginImportError(
            f"Failed to import plugin module {path!r} declared via {source}: {exc}"
        ) from exc

    # RE-ENTRANCY GUARD. ``discover_plugins`` runs at module-import time from
    # ``core/metrics/__init__`` and ``models/losses/__init__``, so a plugin that
    # imports anything from ``spectramr`` re-enters discovery WHILE ITS OWN BODY
    # IS STILL EXECUTING. On that inner call ``import_module`` finds the
    # half-built module in ``sys.modules`` and returns it immediately -- without
    # re-running the body, and without raising. Caching that as "imported" is
    # what poisoned the discovery cache: ``_imported_paths`` gained a module
    # whose ``@register_*`` decorators had not fired (and which, if the outer
    # import then failed, Python removed from ``sys.modules`` entirely), so
    # every later ``discover_plugins`` short-circuited and the plugin could
    # never register in that process -- silently.
    #
    # ``__spec__._initializing`` is True exactly for the in-flight window, so it
    # is the only signal that separates "already imported" from "currently being
    # imported". The OUTER call owns the caching: leave the entry out and let it
    # add the path when the body has actually finished.
    if getattr(getattr(module, "__spec__", None), "_initializing", False):
        logger.debug(
            "plugin module %r is mid-import (re-entrant discovery via %s); "
            "deferring the cache entry to the outer import",
            path,
            source,
        )
        return

    _imported_paths.add(path)
    logger.info("Imported plugin module %r (via %s)", path, source)


def import_plugin_paths(paths: Iterable[str], *, source: str = "config.plugins.paths") -> None:
    """Import each dotted module path, firing its ``@register_*`` decorators.

    Raises :class:`PluginImportError` on the first unimportable path — these are
    explicit, user-declared dependencies, so a typo must fail fast (pitfall #15)
    rather than silently dropping a component the config relies on.
    """
    for path in paths:
        _import_path(path, source=source)


def _env_paths() -> list[str]:
    """Parse ``SPECTRAMR_PLUGINS`` into dotted module paths.

    Split on the OS path separator AND whitespace so both
    ``SPECTRAMR_PLUGINS=a:b`` and ``SPECTRAMR_PLUGINS="a b"`` work.
    """
    raw = os.environ.get(PLUGIN_ENV_VAR, "")
    if not raw:
        return []
    return [tok for tok in re.split(rf"[\s{re.escape(os.pathsep)}]+", raw) if tok]


def _discover_env_paths() -> None:
    """Import ``SPECTRAMR_PLUGINS`` paths (raise on a bad token).

    The environment is re-read on EVERY call. There used to be an
    ``_env_imported`` latch set after the first clean pass, which made the env var
    a once-per-process read: anything that triggered discovery before
    ``SPECTRAMR_PLUGINS`` was set -- loading a config does, via
    ``populate_model_registry`` -- latched it shut, and the plugin was never
    discovered no matter how many times discovery ran afterwards (#1637). A
    long-lived process (notebook, server, campaign runner) could never pick one up.

    Idempotency does not need that latch: ``_import_path`` already skips any path
    in ``_imported_paths``, so a repeat call is a set lookup per token. Per-path
    tracking also preserves the fail-fast the latch was documented to give -- a
    bad token never enters ``_imported_paths``, so it re-raises on every call
    rather than being silently skipped on retry.
    """
    for path in _env_paths():
        _import_path(path, source=f"{PLUGIN_ENV_VAR} env var")


def _discover_entry_points(group: str) -> None:
    """Load the entry-points for ``group`` once (warn on a broken plugin)."""
    if group in _discovered_groups:
        return
    _discovered_groups.add(group)
    try:
        eps = importlib.metadata.entry_points(group=group)
    except Exception as exc:  # pragma: no cover - importlib edge cases
        logger.debug("entry-point enumeration failed for %s: %s", group, exc)
        return
    for ep in eps:
        name = getattr(ep, "name", "?")
        try:
            ep.load()  # imports the module → fires @register_* decorators
        except Exception as exc:  # third-party robustness: warn, don't crash
            logger.warning(
                "plugin entry-point %r (group %s) failed to load: %s",
                name,
                group,
                exc,
            )


def discover_plugins(group: str) -> None:
    """Run the entry-point + env-var discovery layers for one registry ``group``.

    Called from each registry-population choke point (models / losses / metrics
    / datasets).
    Idempotent: entry-points are scanned once per group, and the
    ``SPECTRAMR_PLUGINS`` paths are imported once in total.
    """
    _discover_entry_points(group)
    _discover_env_paths()


def resolve_plugin_provenance() -> dict[str, object]:
    """Return the resolved plugin sources for provenance stamping (pitfall #15c).

    Records what the discovery layers actually saw — the ``SPECTRAMR_PLUGINS``
    tokens, the discovered entry-point names per group, and the set of imported
    module paths — so a run can be traced to the exact out-of-tree code it
    pulled in.
    """
    entry_points: dict[str, list[str]] = {}
    for group in (*ENTRY_POINT_GROUPS, STRATEGY_ENTRY_POINT_GROUP):
        try:
            eps = importlib.metadata.entry_points(group=group)
            names = sorted(getattr(ep, "name", "?") for ep in eps)
        except Exception:  # pragma: no cover
            names = []
        if names:
            entry_points[group] = names
    return {
        "env_var": _env_paths(),
        "entry_points": entry_points,
        "imported_paths": sorted(_imported_paths),
    }
