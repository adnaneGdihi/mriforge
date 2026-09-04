#!/usr/bin/env python3
"""Verify the declared dependency set is installed, version-correct, and importable.

This checks the project against its **single source of truth** — the
``[project].dependencies`` and ``[project.optional-dependencies]`` tables in
``pyproject.toml`` — rather than a hand-maintained list, so it cannot drift from
what the package actually declares.

Two distinct failure modes are covered, because they are genuinely different:

1. **Missing / version-mismatch** — the distribution is absent, or the installed
   version does not satisfy the declared specifier. Detected from package
   metadata alone (fast, no heavy imports).

2. **Installed-but-unimportable** — the distribution is present at a satisfying
   version yet ``import`` raises, usually from a transitive-dependency conflict.
   The motivating case was ``torchmetrics``: it was satisfied by metadata yet
   failed to import against ``huggingface-hub>=1.0``. That specific conflict is
   RESOLVED — re-measured 2026-08-29 with torchmetrics 1.9.0 against
   huggingface-hub 1.27.0, the import succeeds — but the failure *mode* is not,
   which is the whole reason this leg exists. Only an actual ``import`` catches
   it; enable with ``--import-check``.

Design constraint: this module imports **only the standard library** at top
level (``tomllib``, ``importlib.metadata``). ``packaging`` is used when present
for exact PEP 440 specifier matching and falls back to a lightweight comparison
otherwise. A dependency checker that needed third-party packages to run would be
useless in the broken environment it exists to diagnose.

Usage
-----
    python scripts/verify/verify_dependencies.py                 # core deps only
    python scripts/verify/verify_dependencies.py --extras mri,viz
    python scripts/verify/verify_dependencies.py --all           # every group
    python scripts/verify/verify_dependencies.py --import-check   # + import probe
    python scripts/verify/verify_dependencies.py --json

Exit codes: ``0`` all selected dependencies satisfied; ``1`` one or more
missing / version-mismatched / (with ``--import-check``) unimportable; ``2``
usage or environment error (e.g. ``pyproject.toml`` unreadable).
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as md
import importlib.util
import json
import re
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"

# Status codes for a single dependency.
OK = "ok"
MISSING = "missing"
MISMATCH = "mismatch"
IMPORT_FAIL = "import_fail"
SKIPPED = "skipped"  # e.g. self-reference or an unsatisfied environment marker

_GLYPH = {
    OK: "OK  ",
    MISSING: "MISS",
    MISMATCH: "VER ",
    IMPORT_FAIL: "IMP ",
    SKIPPED: "skip",
}

# Distributions whose top-level import name differs from the distribution name,
# used as a fallback when ``top_level.txt`` metadata is unavailable.
_IMPORT_NAME_FALLBACK = {
    "pyyaml": "yaml",
    "pillow": "PIL",
    "scikit-learn": "sklearn",
    "pywavelets": "pywt",
    "pot": "ot",
    "huggingface-hub": "huggingface_hub",
    "opencv-python": "cv2",
    "python-dateutil": "dateutil",
}


def _canon(name: str) -> str:
    """PEP 503 canonical distribution name (lowercase, runs of -_. -> -)."""
    return re.sub(r"[-_.]+", "-", name).lower()


# --------------------------------------------------------------------------- #
# Requirement parsing (packaging when available, regex fallback otherwise).
# --------------------------------------------------------------------------- #
def _have_packaging() -> bool:
    """Whether ``packaging`` is importable (guarded — a checker must survive a
    stripped env where even ``packaging`` is absent)."""
    try:
        return importlib.util.find_spec("packaging") is not None
    except Exception:
        return False


_HAVE_PACKAGING = _have_packaging()

_REQ_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(\[[^\]]*\])?\s*(.*)$")


def parse_requirement(spec: str):
    """Return ``(dist_name, specifier_str, marker_ok)`` for a requirement.

    ``marker_ok`` is False only when an environment marker evaluates False for
    the current interpreter (such a requirement is not needed here).
    """
    if _HAVE_PACKAGING:
        from packaging.requirements import Requirement

        req = Requirement(spec)
        marker_ok = req.marker.evaluate() if req.marker else True
        return req.name, str(req.specifier), marker_ok
    m = _REQ_RE.match(spec)
    if not m:
        return spec.strip(), "", True
    name = m.group(1)
    rest = m.group(3) or ""
    # Drop any environment marker (after ';') in the fallback; keep the specifier.
    specifier = rest.split(";", 1)[0].strip()
    return name, specifier, True


def _version_satisfies(installed: str, specifier: str) -> bool:
    if not specifier:
        return True
    if _HAVE_PACKAGING:
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version

        return SpecifierSet(specifier).contains(Version(installed), prereleases=True)
    # Fallback: compare only the lower bound of the first >= / == clause.
    tuples = re.findall(r"(>=|==|<=|>|<|~=|!=)\s*([0-9][0-9A-Za-z.\-]*)", specifier)

    def _vt(v: str) -> tuple:
        return tuple(int(p) if p.isdigit() else 0 for p in re.split(r"[.\-]", v))

    inst = _vt(installed)
    for op, ver in tuples:
        target = _vt(ver)
        if op in (">=", "~=") and inst < target:
            return False
        if op == ">" and inst <= target:
            return False
        if op == "<=" and inst > target:
            return False
        if op == "<" and inst >= target:
            return False
        if op == "==" and inst[: len(target)] != target:
            return False
    return True


# --------------------------------------------------------------------------- #
# pyproject reading.
# --------------------------------------------------------------------------- #
def load_dependency_groups(pyproject: Path = PYPROJECT) -> dict[str, list[str]]:
    """Map group name -> list of requirement strings. ``core`` holds the main deps."""
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    project = data.get("project", {})
    groups: dict[str, list[str]] = {"core": list(project.get("dependencies", []))}
    for name, reqs in project.get("optional-dependencies", {}).items():
        groups[name] = list(reqs)
    return groups


def _project_name(pyproject: Path = PYPROJECT) -> str:
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    return _canon(data.get("project", {}).get("name", "spectramr"))


# --------------------------------------------------------------------------- #
# Import-name resolution + probe.
# --------------------------------------------------------------------------- #
def _import_names(dist_name: str) -> list[str]:
    """Best-effort top-level import module names for an installed distribution."""
    try:
        dist = md.distribution(dist_name)
    except md.PackageNotFoundError:
        return []
    top = dist.read_text("top_level.txt")
    if top:
        names = [ln.strip() for ln in top.splitlines() if ln.strip()]
        if names:
            return names
    canon = _canon(dist_name)
    return [_IMPORT_NAME_FALLBACK.get(canon, canon.replace("-", "_"))]


def _probe_import(dist_name: str) -> tuple[bool, str]:
    """Try to import the distribution's first top-level module. Return (ok, detail)."""
    names = _import_names(dist_name)
    if not names:
        return False, "no import name resolved"
    mod = names[0]
    try:
        importlib.import_module(mod)
        return True, mod
    except Exception as exc:  # report any import failure verbatim
        return False, f"import {mod}: {type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- #
# Checking.
# --------------------------------------------------------------------------- #
def check_requirement(spec: str, self_name: str, import_check: bool) -> dict:
    """Check one requirement string; return a result record."""
    name, specifier, marker_ok = parse_requirement(spec)
    canon = _canon(name)
    rec: dict = {"name": name, "specifier": specifier, "installed": None}

    if not marker_ok:
        rec["status"] = SKIPPED
        rec["detail"] = "environment marker not applicable"
        return rec
    if canon == self_name:
        rec["status"] = SKIPPED
        rec["detail"] = "self-reference (extras alias)"
        return rec

    try:
        installed = md.version(name)
    except md.PackageNotFoundError:
        rec["status"] = MISSING
        rec["detail"] = "not installed"
        return rec

    rec["installed"] = installed
    if not _version_satisfies(installed, specifier):
        rec["status"] = MISMATCH
        rec["detail"] = f"installed {installed} does not satisfy {specifier or '(any)'}"
        return rec

    if import_check:
        ok, detail = _probe_import(name)
        if not ok:
            rec["status"] = IMPORT_FAIL
            rec["detail"] = detail
            return rec

    rec["status"] = OK
    rec["detail"] = ""
    return rec


def _select_groups(
    groups: dict[str, list[str]], extras: list[str], want_all: bool
) -> list[str]:
    if want_all:
        return list(groups)
    selected = ["core"]
    for e in extras:
        if e not in groups:
            raise SystemExit(
                f"unknown extra {e!r}; available: {', '.join(sorted(groups))}"
            )
        selected.append(e)
    return selected


def run(extras: list[str], want_all: bool, import_check: bool) -> dict:
    groups = load_dependency_groups()
    self_name = _project_name()
    selected = _select_groups(groups, extras, want_all)
    results: dict[str, list[dict]] = {}
    for group in selected:
        results[group] = [
            check_requirement(spec, self_name, import_check) for spec in groups[group]
        ]
    return results


_FAIL_STATUSES = {MISSING, MISMATCH, IMPORT_FAIL}


def _has_failure(results: dict) -> bool:
    return any(
        r["status"] in _FAIL_STATUSES for recs in results.values() for r in recs
    )


def _print_human(results: dict, import_check: bool) -> None:
    total = fails = 0
    for group, recs in results.items():
        print(f"\n[{group}]")
        for r in recs:
            total += 1
            if r["status"] in _FAIL_STATUSES:
                fails += 1
            ver = r["installed"] or "-"
            spec = r["specifier"] or "(any)"
            detail = f"  {r['detail']}" if r["detail"] else ""
            print(f"  {_GLYPH[r['status']]}  {r['name']:<22} {ver:<14} {spec}{detail}")
    mode = "metadata + import" if import_check else "metadata"
    print(f"\n{total} checked ({mode}), {fails} problem(s).")
    if fails == 0:
        print("All selected dependencies are installed and satisfied.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify declared dependencies are installed, versioned, and importable."
    )
    parser.add_argument(
        "--extras",
        default="",
        help="comma-separated optional-dependency groups to also check (e.g. mri,viz).",
    )
    parser.add_argument(
        "--all", action="store_true", help="check every dependency group."
    )
    parser.add_argument(
        "--import-check",
        action="store_true",
        help="also import each dependency (catches installed-but-unimportable, "
        "e.g. torchmetrics under an incompatible huggingface-hub).",
    )
    parser.add_argument("--json", action="store_true", help="emit results as JSON.")
    args = parser.parse_args(argv)

    if not PYPROJECT.is_file():
        print(f"pyproject.toml not found at {PYPROJECT}", file=sys.stderr)
        return 2

    extras = [e.strip() for e in args.extras.split(",") if e.strip()]
    try:
        results = run(extras, args.all, args.import_check)
    except SystemExit as exc:  # unknown extra
        print(str(exc), file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        _print_human(results, args.import_check)

    return 1 if _has_failure(results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
