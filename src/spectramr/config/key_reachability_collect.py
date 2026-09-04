"""The AST visitor that indexes every read site, call and class in the tree.

Split out of :mod:`key_reachability` (#1400, NN20). Depends only on
:mod:`key_reachability_model`; :mod:`key_reachability_index` drives it.
"""

from __future__ import annotations

import ast

from spectramr.config.key_reachability_model import (
    _LINE_STRIDE,
    _Class,
    _Index,
    _Scope,
)

# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


def _name_tokens(value: str) -> tuple[str, ...]:
    """The identifiers a string constant *names*, if it names any.

    Two shapes count, and only these two:

    ``"FieldBridgeStrategy"``
        A bare identifier -- a registry key, a ``getattr`` target, a DI string.

    ``"spectramr.infrastructure.training.strategies.x.FieldBridgeStrategy"``
        A dotted path of identifiers. This is how the strategy factory names
        every one of its 153 classes, and how ``ulf_phase2`` names config keys
        (``"physics.b0_correction.enabled"``). Missing this shape is not a small
        recall gap: it reports every dynamically imported class as never
        constructed, which is precisely the verdict that would delete live code.

    Prose is not a reference. ``"FieldBridgeStrategy requires a mapping batch"``
    is an error message, and counting it would make every class that names
    itself in a message its own proof of life.
    """
    if not 0 < len(value) <= 200:
        return ()
    if value.isidentifier():
        return (value,)
    parts = value.split(".")
    if len(parts) > 1 and all(part.isidentifier() for part in parts):
        return tuple(parts)
    return ()


class _FileCollector:
    """Split one module into executable scopes and record what each names.

    Two vocabularies come out of this, and keeping them apart is the whole
    point:

    ``sites``
        Every appearance of a name **in the parsed code** -- including
        ``def``/``class`` statement names, parameter names and imports. This is
        the broad half, kept as close to the rg index as an AST can be so that
        the gate's delta comes from reachability rather than from a narrower
        match.

        It is **not** parity with ripgrep, and the difference is not incidental:
        an AST cannot see a comment, and :func:`_name_tokens` rejects prose, so a
        key named only in a comment, a docstring or a same-named module scans as
        having no read here and as consumed under rg. That accounts for most of
        the keys this gate newly reports, and it is the intended behaviour -- a
        comment is text, not a read -- but a reader must not mistake those for
        call-graph findings. The verdict separates them: they are
        ``NO_READ_FOUND``, not ``NO_LIVE_READ``.

    ``mentions`` / ``constructs``
        References that make something live. A ``def foo`` does not reference
        ``foo``; a parameter named ``model`` does not reference the class
        ``model``. Folding those into liveness would make every definition its
        own proof of life.
    """

    def __init__(self, index: _Index, rel: str, read_zone: bool) -> None:
        self._index = index
        self._rel = rel
        self._read_zone = read_zone

    def run(self, tree: ast.Module) -> None:
        module_scope = self._new_scope(
            kind="module", qualname="<module>", lineno=1, parent=None, class_name=None
        )
        self._walk_body(tree.body, module_scope, class_name=None)

    # -- scope bookkeeping --------------------------------------------------

    def _new_scope(
        self,
        *,
        kind: str,
        qualname: str,
        lineno: int,
        parent: int | None,
        class_name: str | None,
        decorated: bool = False,
    ) -> int:
        scope = _Scope(
            file=self._rel,
            kind=kind,
            qualname=qualname,
            lineno=lineno,
            parent=parent,
            class_name=class_name,
            read_zone=self._read_zone,
            decorated=decorated,
            mentions=set(),
            constructs=set(),
        )
        self._index.scopes.append(scope)
        return len(self._index.scopes) - 1

    def _record_site(self, name: str, scope: int, lineno: int) -> None:
        self._index.sites.setdefault(name, []).append(scope * _LINE_STRIDE + lineno)

    # -- traversal ----------------------------------------------------------

    def _walk_body(self, body: list[ast.stmt], scope: int, *, class_name: str | None) -> None:
        for statement in body:
            self._walk(statement, scope, class_name=class_name)

    def _walk(self, node: ast.AST, scope: int, *, class_name: str | None) -> None:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            self._walk_function(node, scope, class_name=class_name)
            return
        if isinstance(node, ast.ClassDef):
            self._walk_class(node, scope)
            return
        # A lambda gets no scope of its own: its body runs wherever the lambda is
        # called, which this analysis cannot track, so merging it into the
        # enclosing scope is the conservative choice.
        self._record_node(node, scope)
        for child in ast.iter_child_nodes(node):
            self._walk(child, scope, class_name=class_name)

    def _walk_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        scope: int,
        *,
        class_name: str | None,
    ) -> None:
        # Decorators, defaults and annotations evaluate in the ENCLOSING scope.
        for decorator in node.decorator_list:
            self._walk(decorator, scope, class_name=None)
        for default in [*node.args.defaults, *(node.args.kw_defaults or [])]:
            if default is not None:
                self._walk(default, scope, class_name=None)

        self._record_site(node.name, scope, node.lineno)
        parent_scope = self._index.scopes[scope]
        if class_name is not None:
            qualname = f"{class_name}.{node.name}"
        elif parent_scope.kind == "module":
            qualname = node.name
        else:
            qualname = f"{parent_scope.qualname}.{node.name}"
        inner = self._new_scope(
            kind="method" if class_name else "function",
            qualname=qualname,
            lineno=node.lineno,
            parent=scope,
            class_name=class_name,
            decorated=bool(node.decorator_list),
        )
        self._index.funcs_by_name.setdefault(node.name, []).append(inner)

        for argument in [
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
            *([node.args.vararg] if node.args.vararg else []),
            *([node.args.kwarg] if node.args.kwarg else []),
        ]:
            self._record_site(argument.arg, inner, argument.lineno)
            if argument.annotation is not None:
                self._walk(argument.annotation, inner, class_name=None)
        if node.returns is not None:
            self._walk(node.returns, inner, class_name=None)

        self._walk_body(node.body, inner, class_name=None)

    def _walk_class(self, node: ast.ClassDef, scope: int) -> None:
        for decorator in node.decorator_list:
            self._walk(decorator, scope, class_name=None)
        for base in [*node.bases, *node.keywords]:
            self._walk(base, scope, class_name=None)

        self._record_site(node.name, scope, node.lineno)
        record = _Class(
            name=node.name,
            file=self._rel,
            lineno=node.lineno,
            bases=tuple(_base_names(node)),
            decorated=bool(node.decorator_list),
            method_scopes=[],
        )
        self._index.classes.setdefault(node.name, []).append(record)

        # A class BODY executes at import time, in the enclosing scope. Only its
        # methods get scopes of their own.
        before = len(self._index.scopes)
        self._walk_body(node.body, scope, class_name=node.name)
        record.method_scopes = [
            i
            for i in range(before, len(self._index.scopes))
            if self._index.scopes[i].class_name == node.name
            and self._index.scopes[i].parent == scope
        ]

    def _record_node(self, node: ast.AST, scope: int) -> None:
        record = self._index.scopes[scope]
        mentions = record.mentions
        assert mentions is not None

        if isinstance(node, ast.Name):
            self._record_site(node.id, scope, node.lineno)
            mentions.add(node.id)
        elif isinstance(node, ast.Attribute):
            self._record_site(node.attr, scope, node.lineno)
            mentions.add(node.attr)
        elif isinstance(node, ast.keyword):
            if node.arg:
                self._record_site(node.arg, scope, getattr(node, "lineno", record.lineno))
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                # A registry key, a ``getattr`` target, a DI string or an
                # importable dotted path. Text-only, so it counts as a site AND
                # as a reference: this is exactly the dispatch the analysis
                # cannot resolve, and unresolved dispatch means live.
                for token in _name_tokens(node.value):
                    self._record_site(token, scope, node.lineno)
                    mentions.add(token)
        elif isinstance(node, ast.alias):
            # An import binds a name; it does not use it. Site, never a mention.
            for part in (node.name, node.asname):
                if part:
                    self._record_site(part.split(".")[-1], scope, record.lineno)
        elif isinstance(node, ast.Call):
            called = _called_name(node.func)
            if called is not None:
                assert record.constructs is not None
                record.constructs.add(called)
                self._index.constructions.setdefault(called, []).append(
                    scope * _LINE_STRIDE + node.lineno
                )


def _base_names(node: ast.ClassDef) -> list[str]:
    names: list[str] = []
    for base in node.bases:
        if isinstance(base, ast.Name):
            names.append(base.id)
        elif isinstance(base, ast.Attribute):
            names.append(base.attr)
    return names


def _called_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None
