"""Tests for ``scripts/ci/check_ulf_operator_is_wired.py`` (issue #1708).

Non-negotiable 15: a gate is only a gate for the violation shape you have
watched it fail on. This gate's polarity is the opposite of most -- it asserts
``importers >= 1``, so **planting an import turns it GREEN**, and the state that
must turn it RED is the import being *absent*. Getting that backwards would
produce a suite where every test passes and the gate detects nothing.

The five states, and which way each must go::

    real wiring, real tree                -> GREEN
    importer removed, nothing planted     -> RED    (the regression it exists for)
    removed + function-local import       -> GREEN  (the ^-anchored-grep blind spot)
    removed + docstring-only mention      -> RED    (prose must not count)
    removed + YAML metadata claim         -> RED    (a tag is not delivery)

The last is honestly trivial -- the census reads ``.py`` files only, so a YAML
claim could never have counted. It is kept because that claim is the shape that
actually shipped on three ``ulf_physics`` arms, so its polarity is worth pinning
even though the mechanism makes it free.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "ci" / "check_ulf_operator_is_wired.py"


def _load():
    spec = importlib.util.spec_from_file_location("_check_ulf_operator_is_wired", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gate():
    return _load()


def _make_root(tmp_path: Path, consumer: str | None, *, at: str = "data/consumer.py") -> Path:
    """A miniature ``src/spectramr`` tree with the operator and one optional importer."""
    pkg = tmp_path / "src" / "spectramr"
    physics = pkg / "infrastructure" / "physics"
    physics.mkdir(parents=True)
    for part in (pkg, pkg / "infrastructure", physics):
        (part / "__init__.py").write_text("")
    (physics / "ulf_forward_operator.py").write_text(
        "class DifferentiableULFForwardOperator:\n    pass\n"
    )
    if consumer is not None:
        path = pkg / at
        path.parent.mkdir(parents=True, exist_ok=True)
        (path.parent / "__init__.py").write_text("")
        path.write_text(consumer)
    return tmp_path


# --------------------------------------------------------------------------
# RED states -- the regressions the gate exists for.
# --------------------------------------------------------------------------


def test_red_when_nothing_imports_it(gate, tmp_path: Path) -> None:
    """The #1708 state itself: the module exists and nothing reaches it."""
    root = _make_root(tmp_path, consumer=None)
    assert gate.check(root) == []


def test_red_when_only_a_docstring_mentions_it(gate, tmp_path: Path) -> None:
    """Prose must not count -- this is exactly how the module read as wired.

    ``clinical_trust_analyzer.py`` names the module in a docstring and imports
    nothing; a grep-based census scores that file as an importer.
    """
    root = _make_root(
        tmp_path,
        consumer=(
            '"""Trust analysis.\n\n'
            "Degradations follow spectramr.infrastructure.physics.ulf_forward_operator\n"
            "and its DifferentiableULFForwardOperator five-stage pipeline.\n"
            '"""\n\n'
            "# See also: from spectramr.infrastructure.physics import ulf_forward_operator\n"
            "VALUE = 'ulf_forward_operator'\n"
        ),
    )
    assert gate.check(root) == [], "a docstring, a comment and a string are not imports"


def test_red_when_only_a_yaml_metadata_tag_claims_it(gate, tmp_path: Path) -> None:
    """A ``metadata.tags.physics`` claim is not delivery (non-negotiable 16).

    Free by construction (the census reads ``.py`` only), but this is the shape
    three shipped arms actually carried, so the polarity is pinned.
    """
    root = _make_root(tmp_path, consumer=None)
    arm = root / "experiments" / "inprogress" / "ulf_physics"
    arm.mkdir(parents=True)
    (arm / "exp.yaml").write_text("metadata:\n  tags:\n    physics: ulf_forward_operator\n")
    assert gate.check(root) == []


def test_red_when_only_the_module_imports_itself(gate, tmp_path: Path) -> None:
    """Self-import is not evidence anything else reaches it."""
    root = _make_root(tmp_path, consumer=None)
    target = root / "src" / "spectramr" / "infrastructure" / "physics" / "ulf_forward_operator.py"
    target.write_text(
        "from spectramr.infrastructure.physics import ulf_forward_operator\n"
        "class DifferentiableULFForwardOperator:\n    pass\n"
    )
    assert gate.check(root) == []


# --------------------------------------------------------------------------
# GREEN states -- one per import spelling the census must see.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "source"),
    [
        (
            "top-of-file, dotted",
            "import spectramr.infrastructure.physics.ulf_forward_operator\n",
        ),
        (
            "top-of-file, dotted with alias",
            "import spectramr.infrastructure.physics.ulf_forward_operator as ufo\n",
        ),
        (
            "from-package, leaf in names (not in .module)",
            "from spectramr.infrastructure.physics import ulf_forward_operator\n",
        ),
        (
            "from-module, symbol",
            "from spectramr.infrastructure.physics.ulf_forward_operator import (\n"
            "    DifferentiableULFForwardOperator,\n)\n",
        ),
        (
            "function-local (invisible to a ^-anchored grep)",
            "def build():\n"
            "    from spectramr.infrastructure.physics.ulf_forward_operator import (\n"
            "        DifferentiableULFForwardOperator,\n"
            "    )\n"
            "    return DifferentiableULFForwardOperator\n",
        ),
        (
            "method-local, nested inside a class",
            "class Builder:\n"
            "    def make(self):\n"
            "        from spectramr.infrastructure.physics import ulf_forward_operator\n"
            "        return ulf_forward_operator\n",
        ),
        (
            "conditional, inside an if",
            "if True:\n    import spectramr.infrastructure.physics.ulf_forward_operator\n",
        ),
    ],
)
def test_green_for_every_import_shape(gate, tmp_path: Path, label: str, source: str) -> None:
    root = _make_root(tmp_path, consumer=source)
    assert len(gate.check(root)) == 1, f"census missed the {label} shape"


@pytest.mark.parametrize(
    ("label", "source"),
    [
        ("relative, from . import leaf", "from . import ulf_forward_operator\n"),
        (
            "relative, from .leaf import symbol",
            "from .ulf_forward_operator import DifferentiableULFForwardOperator\n",
        ),
        (
            "relative, two levels up",
            "from ..physics.ulf_forward_operator import DifferentiableULFForwardOperator\n",
        ),
    ],
)
def test_green_for_relative_imports_inside_the_package(
    gate, tmp_path: Path, label: str, source: str
) -> None:
    """``ImportFrom.module`` drops the dots, so ``level`` must be resolved.

    The two-level case is planted in a SIBLING package under ``infrastructure/``,
    which is where that spelling actually occurs: from
    ``spectramr.infrastructure.other``, ``..physics.ulf_forward_operator`` resolves
    to the target. Planting it deeper inside ``physics/`` instead would make the
    fixture itself unimportable (``..physics`` would mean ``physics.physics``) --
    a green-looking test that pins nothing.
    """
    at = (
        "infrastructure/other/mod.py"
        if source.startswith("from ..")
        else "infrastructure/physics/sibling.py"
    )
    root = _make_root(tmp_path, consumer=source, at=at)
    assert len(gate.check(root)) == 1, f"census missed the {label} shape"


# --------------------------------------------------------------------------
# The real tree.
# --------------------------------------------------------------------------


def test_the_real_repository_has_a_production_importer(gate) -> None:
    """What #1708 asked for: the operator is reachable from production code."""
    importers = gate.check(REPO_ROOT)
    assert importers, "ulf_forward_operator has no production importer (#1708 regressed)"
    assert any("qmap_ulf_operator_transform" in str(p) for p in importers), importers


def test_a_missing_scan_root_raises_rather_than_reporting_zero(gate, tmp_path: Path) -> None:
    """A bad root must not be indistinguishable from the defect.

    A repo-rooted script run from the wrong directory reports a vacuously clean
    tree; here the same mistake would report "zero importers", which is the
    failure state. It raises instead.
    """
    with pytest.raises(FileNotFoundError):
        gate.check(tmp_path / "not-a-checkout")
