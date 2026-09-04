"""Every registry's FULL population must survive a cold import of the package.

``test_breakthrough_components_registered.py`` is the sibling of this file and
checks a different thing: 16 *named* components, so it guards the specific
import-curation lines those names sit on. That is a spot check. A curation line
dropped for a package containing none of those 16 names is invisible to it --
and a package rename is exactly the change that drops curation lines, silently,
everywhere at once.

This file asserts the *population* instead, with no hardcoded counts. The
registry sizes here move week to week and are meant to be re-measured, never quoted,
so a pinned number would be stale before it was useful and would train people to
regenerate the baseline rather than read it.

The comparison is self-calibrating instead:

    cold  -- a fresh subprocess that imports ONLY the package's documented entry
             point for that registry. This is what a *config* can resolve.
    walk  -- a fresh subprocess that imports every module in the package first.
             This is the upper bound: every ``@register_*`` decorator in the
             tree has fired.

``walk - cold`` is the reachability gap: components that exist, register when
something happens to import them, and are unreachable from a config. That gap is
the one recorded six times over in ``models/init_registry.py:73-190``.

``cold - walk`` is the opposite defect and is not hypothetical either -- it was
non-empty on this tree until the alias-timing fix, when a *fuller* import
produced a *smaller* model registry. Both directions are asserted.

Marked ``slow`` honestly: six subprocesses, each paying the package's import
cost, and the walk pays it for every module. Select with ``-m slow``.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

# Each entry is the *documented* way to obtain that registry's population -- the
# path a config resolution actually takes, not a private attribute. Reading a
# private ``_custom_losses`` happens to agree for losses and does NOT agree for
# metrics, where aliases live inside the lookup predicate; asking the public API
# is the only form that generalises across the five.
_POPULATION_EXPR = {
    "models": (
        "from spectramr.models.init_registry import populate_model_registry;"
        "from spectramr.models.registry import MODEL_REGISTRY;"
        "populate_model_registry();"
        "out = sorted(MODEL_REGISTRY)"
    ),
    "losses": (
        "from spectramr.models.losses.registry import LossRegistry;"
        "out = sorted(LossRegistry.list_available())"
    ),
    "metrics": (
        "from spectramr.core.metrics.registry import MetricsRegistry;"
        "out = sorted(MetricsRegistry.list_available())"
    ),
    "strategies": (
        "from spectramr.infrastructure.training.strategy_factory import"
        " TrainingStrategyFactory;"
        "out = sorted(TrainingStrategyFactory.STRATEGY_CLASS_PATHS)"
    ),
    "transforms": (
        "import spectramr.data.transforms as t;"
        "out = sorted(t.list_transforms())"
    ),
}

# Where a decorator's module has to be imported from for that registry, because
# a decorator only fires when something imports the module defining it. Each
# registry curates those imports differently, and the difference is what makes a
# missing line silent. Named here rather than cited: a test that ships must not
# point its reader at a path outside the distribution.
_CURATION_SITE = {
    "models": (
        "spectramr/models/init_registry.py -- a hand-curated package list walked "
        "with pkgutil. A new FILE in a listed package is picked up; a new "
        "PACKAGE must be added to the list."
    ),
    "losses": "spectramr/models/losses/__init__.py -- add the import.",
    "metrics": (
        "the pkgutil walk over spectramr/core/metrics/, which SKIPS sub-packages "
        "-- a sub-package needs an explicit import."
    ),
    "transforms": (
        "spectramr/data/transforms/__init__.py -- add a `noqa: F401` line to the "
        "curation block, which states this contract in its own header."
    ),
    "strategies": (
        "TrainingStrategyFactory.STRATEGY_CLASS_PATHS -- a hardcoded dict, no "
        "walk at all. The value is a dotted path split on the last '.', so it "
        "takes no colon."
    ),
}


_WALK_FIRST = (
    "import pkgutil, importlib, spectramr\n"
    "for _m in pkgutil.walk_packages(spectramr.__path__, 'spectramr.'):\n"
    "    try: importlib.import_module(_m.name)\n"
    "    except Exception: pass\n"
)


def _population(registry: str, walk: bool) -> set[str]:
    """Read one registry's population in a FRESH interpreter.

    A same-process read proves nothing: this suite has already imported most of
    the package, so every decorator has fired for reasons a config would not
    reproduce. The subprocess is the whole point of the test.
    """
    code = (
        "import json, warnings\n"
        "warnings.filterwarnings('ignore')\n"
        + (_WALK_FIRST if walk else "")
        + _POPULATION_EXPR[registry]
        + "\nprint('POPULATION=' + json.dumps(out))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=900
    )
    assert proc.returncode == 0, (
        f"{registry} ({'walk' if walk else 'cold'}) subprocess failed:\n"
        f"{proc.stdout}\n{proc.stderr}"
    )
    # Anchor on a marker rather than the last line: the package prints to stdout
    # on import (a missing-timm notice today) and a torch atexit error can trail
    # the output entirely.
    marker = [ln for ln in proc.stdout.splitlines() if ln.startswith("POPULATION=")]
    assert len(marker) == 1, f"expected one POPULATION line, got {len(marker)}"
    return set(json.loads(marker[0][len("POPULATION=") :]))


@pytest.mark.slow
@pytest.mark.parametrize("registry", sorted(_POPULATION_EXPR))
def test_cold_population_equals_walk_import_population(registry: str) -> None:
    """Registered and reachable are different claims; this asserts the second."""
    cold = _population(registry, walk=False)
    walk = _population(registry, walk=True)

    unreachable = sorted(walk - cold)
    assert not unreachable, (
        f"{len(unreachable)} {registry} register only when some other module "
        f"imports them, so no config can resolve them: {unreachable[:20]}\n"
        f"Add the missing line to that registry's import-curation site: "
        f"{_CURATION_SITE[registry]}"
    )

    order_dependent = sorted(cold - walk)
    assert not order_dependent, (
        f"{len(order_dependent)} {registry} are LOST when the package is fully "
        f"imported first: {order_dependent[:20]}\n"
        "A fuller import producing a smaller registry means registration depends "
        "on import order. Something runs its registration as an import-time side "
        "effect against a partially-populated registry, and sys.modules caching "
        "makes that permanent. See stubs.register_aliases for the worked example."
    )


@pytest.mark.slow
def test_no_registry_is_silently_empty() -> None:
    """An empty registry passes every membership test that iterates over it.

    The anti-vacuity partner for the test above: ``walk - cold`` and
    ``cold - walk`` are both empty when BOTH sides are empty, which is exactly
    what a broken import produces. This asserts each population is non-trivial
    without pinning a number to it.
    """
    empty = [r for r in sorted(_POPULATION_EXPR) if not _population(r, walk=False)]
    assert not empty, (
        f"registries came back EMPTY from a cold import: {empty}. "
        "The comparison test above cannot see this -- two empty sets are equal."
    )


def test_every_registry_has_a_curation_site_documented() -> None:
    """The failure message reads _CURATION_SITE, so a gap there is a KeyError.

    Deliberately NOT marked slow, and deliberately not folded into the tests
    above. Those two only touch _CURATION_SITE on the failure path, so a sixth
    registry added to _POPULATION_EXPR alone would leave the lookup unexercised
    until the day something actually broke -- at which point the diagnostic
    raises KeyError instead of naming the curation step. An error path is worth
    exactly as much as the coverage it has, and this one costs no subprocess.
    """
    assert set(_POPULATION_EXPR) == set(_CURATION_SITE), (
        "_POPULATION_EXPR and _CURATION_SITE disagree: "
        f"missing a curation site {sorted(set(_POPULATION_EXPR) - set(_CURATION_SITE))}, "
        f"documented but not measured {sorted(set(_CURATION_SITE) - set(_POPULATION_EXPR))}"
    )
    assert _POPULATION_EXPR, "no registries measured at all -- anti-vacuity"
