"""Pins for the ``key_reachability`` split (#1400, Wave 0 exit).

The 814-LOC module was split into a facade plus three siblings under the 300-LOC
ceiling (NN20), along the chain ``model <- collect <- index <- key_reachability``.

Two invariants are deliberately *not* re-asserted here, because
``tests/unit/config/test_key_reachability.py`` and ``tests/architecture/``
already own them and a second enforcer leaves neither audited as the sole line
of defence (NN17): ``PACKAGE_DIR`` resolving to this checkout, and the LOC
ceiling itself.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from spectramr.config import (
    key_reachability as facade,
)
from spectramr.config import (
    key_reachability_collect as collect,
)
from spectramr.config import (
    key_reachability_index as index,
)
from spectramr.config import (
    key_reachability_model as model,
)

_OWNER = {
    "ClassVerdict": model,
    "ReachabilityVerdict": model,
    "ReadEvidence": model,
    "PACKAGE_DIR": index,
}

#: ``module -> the siblings it may import``. The chain is one-way, so the
#: package can be imported from any entry point without a cycle.
_ALLOWED = {
    "key_reachability_model": set(),
    "key_reachability_collect": {"key_reachability_model"},
    "key_reachability_index": {"key_reachability_model", "key_reachability_collect"},
    "key_reachability": {"key_reachability_model", "key_reachability_index"},
}


def _sibling_imports(module) -> set[str]:
    """Sibling modules imported by ``module``, read off the AST.

    Not a text match on the source: every one of these modules names its
    siblings in prose, so ``"key_reachability_index" in getsource(...)`` reports
    an import that is not there.
    """
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
        elif isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
    return {n.split(".")[-1] for n in names} & set(_ALLOWED)


def test_public_api_is_unchanged() -> None:
    assert facade.__all__ == [
        "ClassVerdict",
        "ReachabilityVerdict",
        "ReadEvidence",
        "class_liveness",
        "is_key_reachable",
    ]


@pytest.mark.parametrize("name", sorted(_OWNER))
def test_facade_re_exports_the_owning_definition(name: str) -> None:
    """Identity, not equality: a re-export, never a second definition (NN17)."""
    assert getattr(facade, name) is getattr(_OWNER[name], name)


@pytest.mark.parametrize("mod", [facade, collect, index, model])
def test_sibling_imports_stay_one_way(mod) -> None:
    stem = Path(mod.__file__).stem
    assert _sibling_imports(mod) <= _ALLOWED[stem]


def test_the_import_scan_can_fire() -> None:
    """Anti-vacuity: an empty result must mean *no imports*, not *scan broken*."""
    assert _sibling_imports(index) == {
        "key_reachability_model",
        "key_reachability_collect",
    }


def test_the_new_modules_are_inside_the_analysed_tree() -> None:
    """The analysis reads its own package, so the split changed its own input.

    Splitting moved ~550 LOC into three new files. Had any of them landed
    outside ``PACKAGE_DIR``, the index would have silently stopped seeing the
    reads they contain -- a shrinking index reports *more* keys unreachable, and
    nothing else would have said so.
    """
    scanned = {rel for _path, rel, _read_zone in index._source_files()}
    for mod in (facade, collect, index, model):
        assert f"src/spectramr/config/{Path(mod.__file__).stem}.py" in scanned
