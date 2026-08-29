#!/usr/bin/env python3
"""Conserve the ``check_name`` strings that ``ConfigHealthChecker`` emits (#1355).

``pipelines/train.py`` decides which health failures are fatal by matching the
**emitted** ``HealthCheckResult.check_name`` against ``FATAL_HEALTH_CHECKS``::

    fatal_errors = [r for r in health_report.errors if r.check_name in FATAL_HEALTH_CHECKS]

The test that guarded that set asserted a *method* named ``check_<name>`` exists.
Method identity and emitted name are independent values, and they already diverge
for 14 of the checks -- ``check_required_sections`` emits ``required_section``,
singular -- so the assertion passed for names no runtime result can ever carry.
Both current members happen to align, which makes this latent rather than live,
and exactly the shape that stays latent until it doesn't.

This gate owns the invariant instead (one owner, CLAUDE.md #17), out of one
committed artifact -- ``health_check_names.txt``, a ``method -> emitted names``
mapping -- so there is no second ledger to keep in sync. Three views fall out of
it: **fatal coverage** (every ``FATAL_HEALTH_CHECKS`` entry is a name something
emits, so a typo'd entry cannot silently stop being fatal); **conservation** (a
name that changes spelling, disappears, or *migrates between methods* moves the
diff); and a **divergence ratchet** (``method != emitted`` is derived, so the 14
known slips stay visible while a 15th turns the gate red). The 14 are recorded,
not renamed: those strings reach provenance and report output, so renaming them
is a behaviour change and a separate owner decision.

Why AST and not grep: ``check_name`` is passed three ways here -- keyword,
positional, and via a local ``check_name = "..."`` constant (see
``check_deepspeed_extra_installed``). A grep sees the first only, and the
positional index cannot be hardcoded: it is derived from the ``HealthCheckResult``
dataclass field order, because assuming index 0 resolves every positional
emission to the ``passed`` flag. An unresolvable or empty emission is an error,
not a skip -- a future ``self._make_result(...)`` helper would empty this
collector silently and leave a green gate over an unscanned file. Absent is a
state to report, never to infer.

Stdlib only -- the CI ``guards`` job runs it with a bare ``python3``, no venv.
``--refresh`` rewrites the ledger; ``--json`` dumps the mapping. Exit codes: 0
clean; 1 a violation (named, with the strings); 2 source or ledger unreadable.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE = REPO_ROOT / "src/mriforge/infrastructure/validation/config_health_checker.py"
LEDGER = Path(__file__).resolve().parent / "health_check_names.txt"

CHECKER_CLASS = "ConfigHealthChecker"
RESULT_CLASS = "HealthCheckResult"
FATAL_SET_NAME = "FATAL_HEALTH_CHECKS"
CHECK_PREFIX = "check_"
#: Bucket for emissions that sit outside any ``check_*`` method -- a helper, a
#: module-level function. Empty today; a non-empty bucket is a real diff.
OUTSIDE_KEY = "<outside>"

LEDGER_HEADER = """\
# Emitted HealthCheckResult check_name strings -- GENERATED, do not hand-edit.
# Regenerate: python3 scripts/ci/check_health_check_names.py --refresh
# One line per check method: the method, then every check_name string it emits.
# train.py's FATAL_HEALTH_CHECKS filter matches the EMITTED name, not the method
# name; the two diverge for some checks, and that is what this file shows (#1355).
"""

Mapping = dict[str, list[str]]


class SourceError(RuntimeError):
    """The source file could not be read the way this gate requires."""


def _result_check_name_index(tree: ast.Module) -> int:
    """Positional index of ``check_name`` in ``HealthCheckResult``'s field order."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == RESULT_CLASS:
            fields = [
                stmt.target.id
                for stmt in node.body
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
            ]
            if "check_name" not in fields:
                raise SourceError(f"{RESULT_CLASS} declares no 'check_name' field: {fields}")
            return fields.index("check_name")
    raise SourceError(f"no {RESULT_CLASS} dataclass found")


def _string_constants(scope: ast.AST) -> dict[str, str]:
    """``name -> value`` for plain ``x = "literal"`` assignments directly in *scope*."""
    consts: dict[str, str] = {}
    for stmt in ast.walk(scope):
        if isinstance(stmt, ast.Assign) and isinstance(getattr(stmt.value, "value", None), str):
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    consts[target.id] = stmt.value.value
    return consts


def _resolve(node: ast.expr | None, consts: dict[str, str]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return consts.get(node.id)
    return None


def _is_result_call(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and getattr(node.func, "id", None) == RESULT_CLASS


def collect(source: str, *, filename: str = "<source>") -> tuple[Mapping, list[str]]:
    """Return ``(method -> sorted emitted check_name strings, unresolvable sites)``."""
    tree = ast.parse(source, filename=filename)
    index = _result_check_name_index(tree)

    checker = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == CHECKER_CLASS),
        None,
    )
    if checker is None:
        raise SourceError(f"no {CHECKER_CLASS} class found")

    methods = [
        n
        for n in checker.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name.startswith(CHECK_PREFIX)
    ]
    if not methods:
        raise SourceError(f"{CHECKER_CLASS} declares no {CHECK_PREFIX}* methods")

    owner: dict[int, str] = {}
    consts_by_owner: dict[str, dict[str, str]] = {OUTSIDE_KEY: _string_constants(tree)}
    for method in methods:
        consts_by_owner[method.name] = _string_constants(method)
        for node in ast.walk(method):
            owner[id(node)] = method.name

    emitted: Mapping = {name: [] for name in sorted(m.name for m in methods)}
    unresolved: list[str] = []
    for node in ast.walk(tree):
        if not _is_result_call(node):
            continue
        key = owner.get(id(node), OUTSIDE_KEY)
        value = next((kw.value for kw in node.keywords if kw.arg == "check_name"), None)
        if value is None and len(node.args) > index:
            value = node.args[index]
        name = _resolve(value, consts_by_owner.get(key, {}))
        # A whitespace name would split in two on the ledger round-trip.
        if name is None or name.split() != [name]:
            unresolved.append(f"{filename}:{node.lineno}: in {key}: {name!r}")
            continue
        if name not in emitted.setdefault(key, []):
            emitted[key].append(name)
    return {k: sorted(v) for k, v in sorted(emitted.items())}, unresolved


def fatal_names(source: str, *, filename: str = "<source>") -> list[str]:
    """The string literals assigned to the module-level ``FATAL_HEALTH_CHECKS``."""
    tree = ast.parse(source, filename=filename)
    for stmt in tree.body:
        targets = [stmt.target] if isinstance(stmt, ast.AnnAssign) else getattr(stmt, "targets", [])
        if not any(isinstance(t, ast.Name) and t.id == FATAL_SET_NAME for t in targets):
            continue
        node = stmt.value
        if isinstance(node, ast.Call) and node.args:
            node = node.args[0]
        if isinstance(node, (ast.Set, ast.List, ast.Tuple)):
            elts = [e for e in node.elts if isinstance(e, ast.Constant)]
            return sorted(e.value for e in elts if isinstance(e.value, str))
        raise SourceError(f"{FATAL_SET_NAME} is not a literal set of strings")
    raise SourceError(f"no module-level {FATAL_SET_NAME} found")


def divergences(mapping: Mapping) -> list[str]:
    """Methods whose own suffix is not among the names they emit."""
    prefix = CHECK_PREFIX
    return [
        m for m, names in mapping.items() if m.startswith(prefix) and m[len(prefix) :] not in names
    ]


def violations(
    mapping: Mapping, unresolved: list[str], fatal: list[str], ledger: Mapping
) -> list[str]:
    """Every rule this gate enforces, as human-readable problem lines."""
    problems = [f"unresolvable check_name at {site}" for site in unresolved]

    problems += [
        f"{m} emits no HealthCheckResult -- it cannot be made fatal, and the collector may be blind"
        for m in sorted(mapping)
        if m != OUTSIDE_KEY and not mapping[m]
    ]
    all_names = {n for names in mapping.values() for n in names}
    problems += [
        f"{FATAL_SET_NAME} names {n!r}, which no check emits "
        f"(a method named check_{n} is not enough -- train.py matches the emitted name)"
        for n in fatal
        if n not in all_names
    ]

    for method in sorted(set(mapping) | set(ledger)):
        was, now = ledger.get(method), mapping.get(method)
        if was == now:
            continue
        if was is None:
            problems.append(f"{method}: new, emits {now} -- run --refresh to record it")
        elif now is None:
            problems.append(f"{method}: gone, emitted {was} -- run --refresh to record it")
        else:
            problems.append(f"{method}: emitted {was}, now emits {now}")
    return problems


def _render(mapping: Mapping) -> str:
    """The ledger body: ``method name...``, one method per line, sorted."""
    return "".join(
        f"{m} {' '.join(names)}\n".replace(" \n", "\n") for m, names in sorted(mapping.items())
    )


def _parse_ledger(text: str) -> Mapping:
    """Inverse of :func:`_render`; ``#`` comments and blank lines are skipped."""
    ledger: Mapping = {}
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        method, *names = line.split()
        ledger[method] = sorted(names)
    return ledger


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - surfaced as exit 2
        raise SourceError(f"cannot read {path}: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    parser.add_argument("--refresh", action="store_true", help="rewrite the ledger from the source")
    parser.add_argument("--json", action="store_true", help="print the mapping and exit 0")
    args = parser.parse_args(argv)

    try:
        source = _read(args.source)
        mapping, unresolved = collect(source, filename=str(args.source))
        fatal = fatal_names(source, filename=str(args.source))
    except (SourceError, SyntaxError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(_render(mapping), end="")
        return 0

    names = {n for v in mapping.values() for n in v}
    if args.refresh:
        args.ledger.write_text(LEDGER_HEADER + _render(mapping), encoding="utf-8")
        print(f"wrote {args.ledger} ({len(mapping)} methods, {len(names)} names)")
        return 0

    try:
        ledger = _parse_ledger(_read(args.ledger))
    except SourceError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    problems = violations(mapping, unresolved, fatal, ledger)
    if problems:
        print(
            f"FAIL: {len(problems)} health-check-name problem(s) in {args.source}", file=sys.stderr
        )
        for line in problems:
            print(f"  - {line}", file=sys.stderr)
        print(
            "\nIf the change is intended, re-run with --refresh and commit the ledger diff.\n"
            "Renaming an emitted string changes report and provenance output: record it, "
            "do not rename it to match its method.",
            file=sys.stderr,
        )
        return 1

    print(
        f"OK: {len(mapping) - (OUTSIDE_KEY in mapping)} check methods emit "
        f"{len(names)} names; {len(divergences(mapping))} recorded method/name divergences; "
        f"all {len(fatal)} {FATAL_SET_NAME} entries are emitted"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
