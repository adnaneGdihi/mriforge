"""Pins for the ``schedule_certification_checks`` split (#1400, Wave 0 exit).

The 534-LOC module was split into a facade plus three siblings under the 300-LOC
ceiling (NN20). These tests pin the three things a split can silently break:
the re-exports must be *the same objects* and not copies (NN17), the five
witnesses must still register through the package walk (NN16), and the shared
helpers must stay a one-way dependency so the package walk cannot deadlock on a
cycle.

The LOC ceiling itself is deliberately *not* asserted here --
``tests/architecture/`` owns that invariant and a second enforcer would leave
neither audited as the sole line of defence (NN17).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from spectramr.infrastructure.validation.witness import get_witness_registry
from spectramr.infrastructure.validation.witness.checks import (
    schedule_allocation_checks,
    schedule_nesting_checks,
)
from spectramr.infrastructure.validation.witness.checks import (
    schedule_certification_checks as facade,
)
from spectramr.infrastructure.validation.witness.checks import (
    schedule_certification_common as common,
)

_OWNER = {
    "build_process_from_config": common,
    "synthetic_spectral_prior": common,
    "schedule_nesting_leakfree": schedule_nesting_checks,
    "schedule_no_inert_steps": schedule_nesting_checks,
    "schedule_line_allocation": schedule_allocation_checks,
    "schedule_step_to_reach_cap": schedule_allocation_checks,
    "schedule_tangential_defect_margin": schedule_allocation_checks,
}

_MODULES = (facade, common, schedule_nesting_checks, schedule_allocation_checks)

_WITNESS_NAMES = (
    "schedule.line_allocation",
    "schedule.nesting_leakfree",
    "schedule.no_inert_steps",
    "schedule.step_to_reach_cap",
    "schedule.tangential_defect_margin",
)


def _imported_modules(module) -> set[str]:
    """Module names imported by ``module``, read off the AST.

    Deliberately not ``inspect.getsource(...) and "<name>" in src``: every one of
    these modules names its siblings in prose, so a text match reports an import
    that is not there.
    """
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_facade_all_matches_the_owner_table() -> None:
    assert sorted(facade.__all__) == sorted(_OWNER)


@pytest.mark.parametrize("name", sorted(_OWNER))
def test_facade_re_exports_the_owning_definition(name: str) -> None:
    """Identity, not equality: a re-export, never a second definition (NN17)."""
    assert getattr(facade, name) is getattr(_OWNER[name], name)


def test_category_has_exactly_one_definition() -> None:
    """``_CATEGORY`` is assigned in ``common`` and nowhere else.

    It was briefly duplicated into both witness modules during the split; the
    de-duplication then half-applied and left an undefined name. This pins the
    resolved state.
    """
    definers = [
        Path(m.__file__).name
        for m in _MODULES
        if any(
            isinstance(node, ast.Assign)
            and any(getattr(t, "id", None) == "_CATEGORY" for t in node.targets)
            for node in ast.walk(ast.parse(Path(m.__file__).read_text(encoding="utf-8")))
        )
    ]
    assert definers == ["schedule_certification_common.py"]


def test_common_does_not_import_the_witness_modules() -> None:
    """Shared helpers are the leaf of the DAG, so the package walk cannot cycle."""
    assert not {
        n for n in _imported_modules(common) if n.split(".")[-1].startswith("schedule_")
    }


def test_the_import_scan_can_fire() -> None:
    """Anti-vacuity: the scan above must actually see sibling imports somewhere."""
    seen = {n.split(".")[-1] for n in _imported_modules(schedule_nesting_checks)}
    assert "schedule_certification_common" in seen


@pytest.mark.parametrize("name", _WITNESS_NAMES)
def test_witness_is_registered_after_the_package_walk(name: str) -> None:
    """NN16: the capability is only delivered once the production path resolves it.

    ``witness/__init__`` ``pkgutil``-walks ``checks/``, so the split's new sibling
    modules register without wiring -- observed here rather than assumed.
    """
    assert get_witness_registry().get(name) is not None
