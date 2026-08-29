"""Core Package
============

Framework-level primitives with no upward dependencies: the environment-variable
SSOT, device/topology resolution, and the metric registry.

Neither exported submodule is imported eagerly, for the same reason
``mriforge.models`` is lazy (see ``docs/cli_startup_budget.rst``).
``core/metrics/__init__.py`` runs a ``pkgutil.walk_packages`` discovery pass
over the metric modules and imports ``torch`` at ``core/metrics/registry.py``.
``cli/app.py`` imports ``mriforge.core.env`` to build the parser -- and importing
ANY submodule runs its parent package's ``__init__`` first, so the single line
``from . import env, metrics`` charged the whole metric-discovery walk to
``mriforge --help`` (issue #1130).

Nothing is lost for real consumers: importing a submodule always executes the
parent ``__init__``, so ``from mriforge.core.metrics.registry import ...`` still
runs the walk and still populates ``MetricsRegistry``. Laziness moves *who
triggers* the discovery, never whether it happens.
"""

__all__ = ["env", "metrics"]


def __getattr__(name: str):
    """Lazily resolve ``env`` / ``metrics`` without eager heavy imports.

    Only the two previously-eager submodules are served here. The other
    ``core`` modules were never attributes of this package, and
    ``from mriforge.core import <submodule>`` keeps working for all of them:
    the import system falls back to a submodule import when this hook raises.
    """
    if name in __all__:
        import importlib

        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
