"""Whole-tree index construction and the liveness fixed point.

Split out of :mod:`key_reachability` (#1400, NN20). Drives
:class:`~spectramr.config.key_reachability_collect._FileCollector` over
every source file and solves liveness to a fixed point.

``PACKAGE_DIR`` lives here and is re-exported by :mod:`key_reachability`. It is
derived from *this file's* location, so it stays correct only while this module
sits in ``src/spectramr/config/`` -- an invariant
``tests/unit/config/test_key_reachability.py::test_package_dir_is_this_checkout``
watches, since it compares against the test file's own repo root.
"""

from __future__ import annotations

import ast
from collections import defaultdict, deque
from functools import lru_cache
from pathlib import Path

from spectramr.config.key_reachability_collect import _FileCollector
from spectramr.config.key_reachability_model import _Index

#: ``src/spectramr`` -- the only tree a read site may come from.
PACKAGE_DIR = Path(__file__).resolve().parents[1]

#: Reads declared here are the schema declaring itself, never consumption.
SCHEMA_ZONE = PACKAGE_DIR / "config" / "schemas"

#: Extra trees consulted for liveness evidence only (never for read sites).
EVIDENCE_DIR_NAMES = ("runners", "scripts", "tools")

#: Methods that run without anyone naming them.
_IMPLICITLY_CALLED = frozenset(
    {
        "__init__",
        "__new__",
        "__post_init__",
        "__call__",
        "__enter__",
        "__exit__",
        "__aenter__",
        "__aexit__",
        "__iter__",
        "__next__",
        "__len__",
        "__getitem__",
        "__setitem__",
        "__contains__",
        "__repr__",
        "__str__",
        "__eq__",
        "__hash__",
        "__del__",
        "__getattr__",
        "__setattr__",
    }
)
# ---------------------------------------------------------------------------
# Index construction + liveness fixed point
# ---------------------------------------------------------------------------


def _source_files() -> list[tuple[Path, str, bool]]:
    """Every ``.py`` the analysis reads: (path, repo-relative name, read-zone).

    Read-zone is the package minus ``config/schemas/``: a schema module naming
    its own field is a declaration, not consumption. The other trees contribute
    liveness evidence only.
    """
    root = PACKAGE_DIR.parents[1]
    found: list[tuple[Path, str, bool]] = [
        (path, str(path.relative_to(root)), SCHEMA_ZONE not in path.parents)
        for path in sorted(PACKAGE_DIR.rglob("*.py"))
    ]
    for name in EVIDENCE_DIR_NAMES:
        directory = root / name
        if not directory.is_dir():
            continue
        found.extend(
            (path, str(path.relative_to(root)), False) for path in sorted(directory.rglob("*.py"))
        )
    return found


def _collect_index() -> _Index:
    index = _Index(
        scopes=[],
        classes={},
        sites={},
        constructions={},
        funcs_by_name={},
        live_scopes=set(),
        live_classes={},
    )
    for path, rel, read_zone in _source_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError, OSError):
            # A file the interpreter could not import either. Skipping it can
            # only lose liveness evidence, so a later verdict is unreachable-by-
            # ignorance -- which this module reports as a finding to look at,
            # never as a licence to delete.
            continue
        _FileCollector(index, rel, read_zone).run(tree)
    _solve_liveness(index)
    return index


def _solve_liveness(index: _Index) -> None:
    """Least fixed point over "can this scope run?".

    Seeds: every module body, because importing a module executes it. Then a
    function goes live when its *name* is referenced from a live scope, and a
    method additionally requires its class to be live. Classes go live on a
    decorator, on any reference from a live scope (a bare name handed to a
    registry or a container is a construction this analysis cannot rule out),
    or on being the base of a live class.
    """
    referenced: set[str] = set()
    pending_names: deque[str] = deque()
    scopes = index.scopes

    subclasses: dict[str, set[str]] = defaultdict(set)
    for name, records in index.classes.items():
        for record in records:
            for base in record.bases:
                subclasses[base].add(name)

    methods_by_class: dict[str, list[int]] = defaultdict(list)
    for name, records in index.classes.items():
        for record in records:
            methods_by_class[name].extend(record.method_scopes)

    def reference(name: str) -> None:
        if name not in referenced:
            referenced.add(name)
            pending_names.append(name)

    def activate(scope_id: int) -> None:
        if scope_id in index.live_scopes:
            return
        index.live_scopes.add(scope_id)
        scope = scopes[scope_id]
        for name in scope.mentions or ():
            reference(name)

    def activate_class(name: str, reason: str) -> None:
        if name in index.live_classes:
            return
        index.live_classes[name] = reason
        for record in index.classes.get(name, ()):
            for base in record.bases:
                activate_class(base, f"base of live class `{name}`")
        for scope_id in methods_by_class.get(name, ()):
            try_activate_function(scope_id)

    def try_activate_function(scope_id: int) -> None:
        scope = scopes[scope_id]
        if scope_id in index.live_scopes:
            return
        if scope.parent is not None and scope.parent not in index.live_scopes:
            return
        name = scope.qualname.rsplit(".", 1)[-1]
        if scope.class_name is not None and scope.class_name not in index.live_classes:
            return
        implicit = scope.class_name is not None and name in _IMPLICITLY_CALLED
        if scope.decorated or implicit or name in referenced:
            activate(scope_id)

    for scope_id, scope in enumerate(scopes):
        if scope.kind == "module":
            activate(scope_id)
    for name, records in index.classes.items():
        if any(record.decorated for record in records):
            activate_class(name, "carries a decorator -- registry-constructed")

    changed = True
    while changed or pending_names:
        changed = False
        while pending_names:
            name = pending_names.popleft()
            if name in index.classes:
                activate_class(name, "referenced from a live scope")
            for scope_id in index.funcs_by_name.get(name, ()):
                try_activate_function(scope_id)
        # A scope activated above may have unblocked a nested def whose parent
        # was dead when it was first considered.
        for scope_id, scope in enumerate(scopes):
            if scope_id in index.live_scopes or scope.kind == "module":
                continue
            before = len(index.live_scopes)
            try_activate_function(scope_id)
            if len(index.live_scopes) != before:
                changed = True
        for name in list(subclasses):
            if name in index.live_classes:
                continue
            if any(child in index.live_classes for child in subclasses[name]):
                activate_class(name, "base of a live class")
                changed = True


@lru_cache(maxsize=1)
def _index() -> _Index:
    """The whole-tree index, built once per process (~5 s, ~150 MB).

    **Read-only contract.** The cache hands every caller the *same* mutable
    ``_Index`` -- mutable ``_Scope``/``_Class`` records, mutable dicts and sets.
    Mutating any of it silently corrupts every later verdict in the process, and
    the ``lru_cache`` means the corruption survives until the interpreter exits.
    Freezing the whole graph would cost a second full copy of it, so the contract
    is stated instead of enforced: the only callers are :func:`is_key_reachable`
    and :func:`class_liveness`, both of which read and never write. Keep it that
    way, or call :func:`_collect_index` for a private copy.
    """
    return _collect_index()
