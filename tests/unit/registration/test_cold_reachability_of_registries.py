"""Registered is not reachable: the cold-subprocess gap, one test per registry.

The reachability contract keeps three claims apart -- **registered** (the name
is in the registry), **reachable** (a config can resolve it), and **fires** (the
mechanism does work). This file establishes the second, and only the second.

A same-process assertion cannot: pytest has already imported hundreds of
modules by the time it runs, so a decorator that fires only because a sibling
test imported its module looks identical to one a plain ``import mriforge.*``
reaches. ``models/init_registry.py:73-190`` records six incidents of exactly
that, e.g. at ``:110-115`` -- *"``gans`` registers 9 models ... Before this
entry, only the last 3 fired (transitively via ``generators/__init__.py``), the
other 6 decorators were dead."* Transitively is the word. Those six were green
in the suite and unreachable from YAML.

Each test therefore spawns **one** fresh interpreter that:

1. reads the registry after importing only the production entry point (cold);
2. walks and imports every module *and* sub-package under the registry's tree;
3. reads the registry again (exhaustive).

``exhaustive - cold`` is the set of names that exist but no config can name.
It must be empty.

Two guards keep a green result from being an artefact:

* **Import failures are counted, not swallowed.** A walk that silently drops
  half the tree reports no gap for the same reason a correct one does. The
  walk here records every failure, including the sub-package failures
  ``pkgutil`` hands to ``onerror``, and a non-empty record fails the test.
* **A sentinel name must be present cold.** An empty registry trivially has an
  empty gap. Each sentinel is chosen to sit behind that registry's own
  curation step -- the line a rename or a refactor actually drops.

Curation differs per registry, and getting it wrong is silent:

===========  ===========================================================
transforms   a hand-written ``noqa: F401`` block in ``__init__``
losses       ``from . import (...)`` in ``models/losses/__init__.py``
metrics      a ``pkgutil`` walk over the package
models       a ``pkgutil`` walk over a hand-curated **package list**
strategies   none -- a hardcoded dict of dotted paths
===========  ===========================================================

Strategies get a different test because they have no registry to be a member
of: ``STRATEGY_CLASS_PATHS`` is a dict literal, and asserting its keys exist
proves only that a dict parses. The reachable claim there is that all 206
dotted paths **resolve** -- import the module, find the attribute -- which is
what the factory does at ``rsplit(".", 1)`` and what a package rename breaks.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.registry_contract, pytest.mark.slow]

_REPO_SRC = Path(__file__).resolve().parents[3] / "src"


def _subprocess_env() -> dict[str, str]:
    """Pin the child to the tree this test file lives in.

    Without ``PYTHONPATH`` a subprocess run from a git worktree imports the
    *main checkout* through the editable install, so the whole measurement
    silently describes a different tree. Assert the two agree rather than
    trusting either.
    """
    import mriforge

    imported_src = Path(mriforge.__file__).resolve().parents[1]
    assert imported_src == _REPO_SRC, (
        f"this test process imported mriforge from {imported_src}, but the test "
        f"file lives under {_REPO_SRC}. The subprocess would measure a "
        "different tree than the one under test."
    )
    return {
        **os.environ,
        "PYTHONPATH": str(_REPO_SRC),
        "MRIFORGE_SUPPRESS_CLINICAL_WARNING": "1",
    }


# The walk records failures instead of raising: a *measurement* must be able to
# report "the tree did not import cleanly", which is a different finding from
# "there is no gap" and must not be confusable with it.
_DRIVER = """
import importlib, json, pkgutil, sys

failures = {}


def read():
%(READ)s


def walk(pkgname):
    pkg = importlib.import_module(pkgname)

    def on_package_error(name):
        # pkgutil recurses INTO a sub-package by importing it, inside pkgutil
        # itself. Without this hook that failure is discarded and the entire
        # sub-tree is skipped with no error and exit 0.
        exc = sys.exc_info()[1]
        failures[name] = "sub-package: %%s: %%s" %% (type(exc).__name__, exc)

    for _, name, _is_pkg in pkgutil.walk_packages(
        pkg.__path__, pkg.__name__ + ".", onerror=on_package_error
    ):
        try:
            importlib.import_module(name)
        except Exception as exc:
            failures[name] = "%%s: %%s" %% (type(exc).__name__, exc)


cold = sorted(read())
walk(%(WALK)r)
exhaustive = sorted(read())
print("RESULT " + json.dumps(
    {"cold": cold, "exhaustive": exhaustive, "failures": failures}
))
"""

_READS = {
    "models": """
    from mriforge.models.init_registry import populate_model_registry
    from mriforge.models.registry import MODEL_REGISTRY

    # Not optional and not implicit: MODEL_REGISTRY is 0 on a plain
    # ``import mriforge.models``. This is the call a config path makes.
    populate_model_registry()
    return list(MODEL_REGISTRY)
""",
    "losses": """
    import mriforge.models.losses as losses

    return losses.LossRegistry.list_available()
""",
    "metrics": """
    import mriforge.core.metrics as metrics

    return metrics.list_available()
""",
    "transforms": """
    import mriforge.data.transforms as transforms

    return transforms.list_transforms()
""",
}

_WALK_ROOTS = {
    "models": "mriforge.models",
    "losses": "mriforge.models.losses",
    "metrics": "mriforge.core.metrics",
    "transforms": "mriforge.data.transforms",
}

#: One name per registry that sits behind that registry's curation step, so an
#: empty or half-populated registry cannot pass by having an empty gap.
_SENTINELS = {
    # models: ``unrolled`` is in init_registry's hand-curated package list with
    # a comment saying this decorator would never fire without that entry.
    "models": "neural_ode_recon",
    # losses: an explicit line in the ``from . import (...)`` block.
    "losses": "spectral_triple_intertwining",
    # metrics: lives in the ``connectivity`` SUB-package, which the walk reaches
    # only through pkgutil's recursion -- the path that used to fail silently.
    "metrics": "geodesic_fc_error",
    # transforms: the first entry of the hand-written noqa block.
    "transforms": "foreground_mask",
}


def _measure(registry: str) -> dict:
    driver = _DRIVER % {
        "READ": _READS[registry].strip("\n"),
        "WALK": _WALK_ROOTS[registry],
    }
    proc = subprocess.run(
        [sys.executable, "-c", driver],
        capture_output=True,
        text=True,
        timeout=900,
        env=_subprocess_env(),
        cwd=str(_REPO_SRC.parent),
    )
    assert proc.returncode == 0, (
        f"cold-reachability probe for {registry!r} did not complete:\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    marker = "RESULT "
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith(marker)), None)
    assert line is not None, f"probe for {registry!r} printed no RESULT line:\n{proc.stdout}"
    return json.loads(line[len(marker) :])


@pytest.mark.parametrize("registry", sorted(_WALK_ROOTS))
def test_no_registered_name_is_unreachable_from_a_cold_import(registry: str) -> None:
    """Every name the tree can register is registered by the production path.

    A name in ``exhaustive`` but not in ``cold`` is registered and unreachable:
    the decorator exists, a test that imports the module sees it, and no YAML
    can name it.
    """
    result = _measure(registry)

    assert not result["failures"], (
        f"the {registry} tree did not import cleanly, so this run cannot report "
        "on reachability at all -- a swallowed import produces the same empty "
        "gap a correct tree does:\n"
        + "\n".join(f"  {k}: {v}" for k, v in sorted(result["failures"].items()))
    )

    sentinel = _SENTINELS[registry]
    assert sentinel in result["cold"], (
        f"{sentinel!r} is missing from a cold import of the {registry} registry "
        f"({len(result['cold'])} names present). It sits behind that registry's "
        "curation step, so its absence means the curation itself is broken -- "
        "and an empty registry would otherwise pass this test with an empty gap."
    )

    unreachable = sorted(set(result["exhaustive"]) - set(result["cold"]))
    assert not unreachable, (
        f"{len(unreachable)} {registry} name(s) are REGISTERED but not REACHABLE: "
        "importing the whole tree finds them, importing the package alone does "
        f"not, so no config can resolve them.\n  {unreachable}\n"
        "Add the missing import to that registry's curation step."
    )


def test_every_strategy_dotted_path_resolves_from_a_cold_import() -> None:
    """Strategies have no registry -- the reachable claim is that the paths work.

    ``STRATEGY_CLASS_PATHS`` maps 206 ``training_mode`` keys onto dotted
    ``module.ClassName`` strings that the factory resolves with
    ``rsplit(".", 1)``. Asserting the keys exist proves a dict literal parses.
    Resolving the values is what a package rename, a moved module or a renamed
    class actually breaks -- and it breaks at config-load time on the cluster,
    not here, unless this test does the resolution.
    """
    driver = """
import importlib, json

from mriforge.infrastructure.training.strategy_factory import TrainingStrategyFactory

# A CLASS attribute -- a module-level import of it raises.
paths = TrainingStrategyFactory.STRATEGY_CLASS_PATHS
unresolved = {}
for key, dotted in sorted(paths.items()):
    if ":" in dotted:
        unresolved[key] = dotted + " -- contains ':'; the factory splits on the last DOT"
        continue
    module_name, _, attr = dotted.rpartition(".")
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        unresolved[key] = "%s -- %s: %s" % (dotted, type(exc).__name__, exc)
        continue
    if not hasattr(module, attr):
        unresolved[key] = dotted + " -- module imported but defines no " + repr(attr)
print("RESULT " + json.dumps(
    {"keys": len(paths), "classes": len(set(paths.values())), "unresolved": unresolved}
))
"""
    proc = subprocess.run(
        [sys.executable, "-c", driver],
        capture_output=True,
        text=True,
        timeout=900,
        env=_subprocess_env(),
        cwd=str(_REPO_SRC.parent),
    )
    assert proc.returncode == 0, (
        "strategy resolution probe did not complete:\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")), None)
    assert line is not None, f"probe printed no RESULT line:\n{proc.stdout}"
    result = json.loads(line[len("RESULT ") :])

    # The mapping is many-to-one: keys are YAML spellings, values are classes.
    # Both numbers drift, so assert the shape rather than pinning either.
    assert result["keys"] > result["classes"] > 0, (
        "STRATEGY_CLASS_PATHS should map many training_mode spellings onto fewer "
        f"classes; got {result['keys']} keys and {result['classes']} classes."
    )
    assert not result["unresolved"], (
        f"{len(result['unresolved'])} of {result['keys']} training_mode key(s) "
        "name a dotted path that does not resolve. Every one is a config that "
        "loads and then dies in the factory:\n"
        + "\n".join(f"  {k}: {v}" for k, v in sorted(result["unresolved"].items()))
    )
