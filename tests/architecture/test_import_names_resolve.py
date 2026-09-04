"""Fitness function: every intra-package ``from X import Y`` must resolve.

Why this gate exists
--------------------
``pipelines/hpo.py`` shipped ``from spectramr.pipelines.hpo_search_spaces import
apply_dotted_override`` against a module that does not define it -- the function
had moved to ``infrastructure/hpo``. Two call sites were updated, one was not
(non-negotiable 19). Because the import sat INSIDE a function, the failure was
deferred to runtime, where only an actual HPO run reached it: every Optuna trial
died applying its overrides while every import-time check stayed green (#1639).

No existing gate could see it. Every grep in ``scripts/ci/check_layering.sh`` is
anchored at ``^`` (non-negotiable 15), so a function-local import is invisible to
it, and ``test_layer_direction.py`` asks about *direction*, not about whether the
imported NAME exists.

So this walks the AST -- module-level and function-local alike -- and asks the
target module whether it actually has the name.

Ratchet, not a snapshot
-----------------------
Known offenders live in ``baselines/unresolved_imports.txt``. A NEW unresolved
import fails; the baseline only ever shrinks. Do not regenerate it to make this
green (non-negotiable 20): each entry is a real broken import, tracked in #1642.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import pathlib

import pytest

pytestmark = pytest.mark.architecture

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src" / "spectramr"
_BASELINE = pathlib.Path(__file__).parent / "baselines" / "unresolved_imports.txt"


def _load_baseline() -> set[str]:
    if not _BASELINE.exists():
        return set()
    return {
        line.strip()
        for line in _BASELINE.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def _collect_from_imports(root: pathlib.Path | None = None) -> list[tuple[str, int, str, str]]:
    """Every ``from spectramr.X import Y`` under ``root``, at ANY nesting depth.

    ``ast.walk`` is what makes function-local imports visible -- the shape the
    ``^``-anchored grep gates structurally cannot see.

    ``root`` is a seam for the planted-violation tests below; production callers
    leave it as the package root.
    """
    root = root or _SRC
    found: list[tuple[str, int, str, str]] = []
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover
            continue
        try:
            rel = path.relative_to(_REPO_ROOT).as_posix()
        except ValueError:  # a tmp_path root, in the planted-shape tests
            rel = path.as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            # Relative imports resolve against the package, not a dotted path we
            # can hand to import_module; they are the layering gate's business.
            if node.level or not node.module:
                continue
            if not node.module.startswith("spectramr."):
                continue
            for alias in node.names:
                if alias.name != "*":
                    found.append((rel, node.lineno, node.module, alias.name))
    return found


def _resolves(module_name: str, attr: str) -> bool | None:
    """Does ``module_name`` provide ``attr``? ``None`` if we cannot tell.

    Two things this must NOT report as broken:

    * ``from pkg import submodule`` -- ``hasattr`` is False until the submodule
      is imported, so fall back to ``find_spec``. Three of the first six hits in
      the original scan were this false positive.
    * a target module that cannot be imported at all (missing optional
      dependency). That is a different problem with a different owner; the
      layering/dependency gates cover it.

    KNOWN LIMIT, stated rather than hidden: for ``from pkg import submodule``
    this asks whether the submodule *exists*, not whether it *imports cleanly*.
    A submodule present on disk but broken internally therefore passes -- e.g.
    ``from spectramr.models.pipelines import generation_pipeline`` (#1642), whose
    file exists but raises ``attempted relative import beyond top-level
    package``. Catching that needs a gate that actually executes every
    submodule, which is a heavier and separately-owned check; this one is about
    stale NAMES.
    """
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None  # unimportable target — not this gate's question
    if hasattr(module, attr):
        return True
    try:
        return importlib.util.find_spec(f"{module_name}.{attr}") is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _unresolved() -> set[str]:
    out: set[str] = set()
    for rel, lineno, module_name, attr in _collect_from_imports():
        if _resolves(module_name, attr) is False:
            out.add(f"{rel}:{lineno} from {module_name} import {attr}")
    return out


def test_no_new_unresolved_intra_package_imports():
    """A ``from X import Y`` where ``X`` has no ``Y`` must not be introduced."""
    unresolved = _unresolved()
    baseline = _load_baseline()
    new = unresolved - baseline
    assert not new, (
        "New unresolved intra-package import(s) — the imported name does not "
        "exist in the target module, so this raises ImportError the moment the "
        "line executes (at runtime, if the import is function-local):\n  "
        + "\n  ".join(sorted(new))
        + "\n\nFix the import. Do NOT add it to "
        f"{_BASELINE.relative_to(_REPO_ROOT)} — that baseline only shrinks."
    )


@pytest.mark.slow
def test_baseline_has_no_stale_entries():
    """Debt tracker: a fixed import must be removed from the baseline.

    Keeps the ratchet honest in the shrinking direction — a baseline entry that
    no longer reproduces is a fix nobody recorded, and leaving it there lets a
    future regression at the same site pass unnoticed.
    """
    stale = _load_baseline() - _unresolved()
    assert not stale, "These baseline entries no longer reproduce — delete them:\n  " + "\n  ".join(
        sorted(stale)
    )


# ---------------------------------------------------------------------------
# The gate is only a gate for the shapes it has been WATCHED to fail on
# (non-negotiable 15). One planted violation per shape the rule can take --
# committed, not run once by hand. Shape 2 is the one that matters: it is the
# shape ``check_layering.sh``'s ``^``-anchored greps structurally cannot see,
# and the shape #1639 actually took.
# ---------------------------------------------------------------------------

_PLANTED_SHAPES = {
    "module_level": "from spectramr.config.settings import PlantedName\n",
    "function_local": (
        "def _probe():\n    from spectramr.config.settings import PlantedName\n\n    return PlantedName\n"
    ),
    "aliased": "from spectramr.config.settings import PlantedName as _p\n",
    "class_body": "class _Holder:\n    from spectramr.config.settings import PlantedName\n",
}


@pytest.mark.parametrize("shape", sorted(_PLANTED_SHAPES))
def test_collector_sees_every_import_shape(tmp_path, shape):
    """Each shape must be COLLECTED; a collector blind to one is blind in prod."""
    (tmp_path / "planted.py").write_text(_PLANTED_SHAPES[shape])
    found = _collect_from_imports(root=tmp_path)
    assert ("spectramr.config.settings", "PlantedName") in [
        (mod, attr) for _rel, _ln, mod, attr in found
    ], f"shape {shape!r} was not collected — the gate is blind to it"


def test_planted_name_does_not_resolve_and_a_real_one_does():
    """The resolver half: the planted name is absent, a real export is present."""
    assert _resolves("spectramr.config.settings", "PlantedName") is False
    assert _resolves("spectramr.config.settings", "TrainingSettings") is True


def test_submodule_import_is_not_a_false_positive():
    """``from pkg import submodule`` must not be reported as unresolved.

    Three of the first six hits in the original scan were this shape; a gate
    that cries wolf on them gets its baseline padded and stops being read.
    """
    assert _resolves("spectramr.pipelines", "make") is not False
