#!/usr/bin/env python3
"""Guard the Sphinx navigation tree against silently-dead sidebar entries.

Sphinx only *warns* when a ``toctree`` entry names a document that is missing or
excluded, so the sidebar can advertise pages that resolve to nothing while the
build still exits 0. That is how ``docs/index.rst`` came to list 210 entries of
which 87 (41%) were dead: successive cleanups added blanket globs to
``exclude_patterns`` (``physics_*.rst`` was aimed at ``physics_rigor_audit_*``
but also killed ``physics_math.rst``) without revisiting the index that still
linked them.

Three invariants, checked without invoking Sphinx (~50 ms, no torch import):

1. **No dead entries.** Every ``toctree`` entry resolves to a file that exists
   and is not matched by ``exclude_patterns``.
2. **No duplicates.** A docname appears at most once across all toctrees; a
   second occurrence makes the sidebar ambiguous and warns at build time.
3. **No orphans.** Every non-excluded document is reachable from some toctree,
   unless it opts out with an ``:orphan:`` marker.

Run: ``python scripts/ci/check_docs_navigation.py``
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS = REPO_ROOT / "docs"

#: Directories never considered part of the documentation set. These are build
#: artifacts or vendored third-party trees that happen to contain Markdown.
SKIP_PREFIXES = ("_build/", ".git/")

#: Vendored Lean toolchain packages (``docs/lean/**/.lake/``) ship ~65 README
#: files. ``.lake`` is gitignored, so Read the Docs never sees them, but a local
#: build scans them and reports 65 orphans. Skip them here for parity.
SKIP_SUBSTRINGS = ("/.lake/",)


def load_exclude_patterns(conf_py: Path) -> list[str]:
    """Extract the ``exclude_patterns`` string list from ``conf.py``.

    Parsed textually rather than by importing ``conf.py``, which would pull in
    Sphinx and the mocked-heavy-dep machinery.
    """
    text = conf_py.read_text()
    match = re.search(r"^exclude_patterns\s*=\s*\[(.*?)^\]", text, re.S | re.M)
    if match is None:
        raise SystemExit(f"{conf_py}: could not locate exclude_patterns list")
    body = match.group(1)
    # Drop comment lines so a commented-out pattern isn't treated as active.
    body = "\n".join(ln for ln in body.splitlines() if not ln.lstrip().startswith("#"))
    return re.findall(r'"([^"]+)"', body)


def patmatch(name: str, pattern: str) -> bool:
    """Match ``name`` against a Sphinx exclude pattern.

    Mirrors :func:`sphinx.util.matching.patmatch` semantics, where ``*`` does not
    cross a path separator and ``**`` does. Python's :mod:`fnmatch` gets this
    wrong (its ``*`` matches ``/``), which is why this is hand-rolled.
    """
    regex = []
    i, n = 0, len(pattern)
    while i < n:
        char = pattern[i]
        if pattern.startswith("**/", i):
            regex.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            regex.append(".*")
            i += 2
        elif char == "*":
            regex.append("[^/]*")
            i += 1
        elif char == "?":
            regex.append("[^/]")
            i += 1
        else:
            regex.append(re.escape(char))
            i += 1
    return re.fullmatch("".join(regex), name) is not None


def is_excluded(rel_posix: str, patterns: list[str]) -> bool:
    """True when ``rel_posix`` is dropped from the build by ``exclude_patterns``."""
    for pattern in patterns:
        if patmatch(rel_posix, pattern):
            return True
        # A bare directory pattern ("audits/**") also excludes the directory's
        # own contents at any depth.
        if pattern.endswith("/**") and rel_posix.startswith(pattern[:-3] + "/"):
            return True
    return False


def iter_doc_files() -> list[Path]:
    """Every candidate source document under ``docs/``."""
    out = []
    for pattern in ("*.rst", "*.md"):
        for path in DOCS.rglob(pattern):
            rel = path.relative_to(DOCS).as_posix()
            if rel.startswith(SKIP_PREFIXES):
                continue
            if any(s in "/" + rel for s in SKIP_SUBSTRINGS):
                continue
            out.append(path)
    return out


TOCTREE_RE = re.compile(r"^\s*(?:\.\.\s+toctree::|```\{toctree\})")


def parse_toctree_entries(path: Path) -> list[str]:
    """Return the raw entry strings of every toctree in ``path``.

    Handles both reST (``.. toctree::``) and MyST (```` ```{toctree} ````)
    syntax. Entries using the ``Title <target>`` form yield ``target``; external
    URLs and ``genindex``-style references are dropped by the caller.
    """
    entries: list[str] = []
    lines = path.read_text(errors="ignore").splitlines()
    in_toc = False
    myst = False
    for line in lines:
        stripped = line.strip()
        if TOCTREE_RE.match(line):
            in_toc = True
            myst = "```" in line
            continue
        if not in_toc:
            continue
        if myst:
            if stripped.startswith("```"):
                in_toc = False
                continue
            if not stripped or stripped.startswith(":"):
                continue
            entries.append(stripped)
            continue
        # reST: the directive body is indented; a non-indented line ends it.
        if not stripped:
            continue
        if stripped.startswith(":"):
            continue
        if not line.startswith((" ", "\t")):
            in_toc = False
            continue
        entries.append(stripped)
    return entries


def resolve(entry: str, from_doc: Path) -> str:
    """Resolve a toctree entry to a docname relative to ``docs/``."""
    target = entry
    angle = re.match(r".*<(.+)>$", entry)
    if angle:
        target = angle.group(1)
    if target.startswith("/"):
        return target.lstrip("/")
    base = from_doc.parent.relative_to(DOCS).as_posix()
    if base in ("", "."):
        return target
    # Normalise any ../ segments.
    parts: list[str] = base.split("/")
    for segment in target.split("/"):
        if segment == "..":
            if parts:
                parts.pop()
        elif segment != ".":
            parts.append(segment)
    return "/".join(parts)


def docname_of(path: Path) -> str:
    return path.relative_to(DOCS).as_posix().rsplit(".", 1)[0]


def find_on_disk(docname: str, patterns: list[str]) -> Path | None:
    """Locate the file backing ``docname``, preferring one that is in the build.

    A docname can be backed by both ``.rst`` and ``.md`` (a "docname collision").
    Sphinx resolves such a pair by ``source_suffix`` order, but conf.py may
    deliberately exclude one half so the other renders. Return the surviving
    half when there is one, so a collision that is correctly resolved is not
    reported as a dead link.
    """
    candidates = [DOCS / (docname + ext) for ext in (".rst", ".md")]
    present = [c for c in candidates if c.exists()]
    if not present:
        return None
    for candidate in present:
        if not is_excluded(candidate.relative_to(DOCS).as_posix(), patterns):
            return candidate
    return present[0]


def is_orphan_marked(path: Path) -> bool:
    """Has this page opted out of needing a toctree entry?

    There are TWO spellings, because there are two parsers. reStructuredText
    uses the ``:orphan:`` field; MyST-parsed Markdown uses ``orphan: true`` in
    YAML front matter. Sphinx honours both. This function recognised only the
    first, so a ``.md`` page that Sphinx correctly treated as an orphan was
    still reported here as a failure -- the gate disagreeing with the build it
    exists to guard.
    """
    head = path.read_text(errors="ignore")[:400]
    if ":orphan:" in head:
        return True
    if head.startswith("---"):
        end = head.find("\n---", 3)
        front = head[3:end] if end != -1 else head[3:]
        for line in front.splitlines():
            key, _, value = line.partition(":")
            if key.strip() == "orphan" and value.strip().lower() in ("true", "yes"):
                return True
    return False


def main() -> int:
    patterns = load_exclude_patterns(DOCS / "conf.py")
    docs = iter_doc_files()

    included: dict[str, Path] = {}
    for path in docs:
        rel = path.relative_to(DOCS).as_posix()
        if is_excluded(rel, patterns):
            continue
        included[docname_of(path)] = path

    dead: list[tuple[str, str, str]] = []
    seen: dict[str, str] = {}
    duplicates: list[tuple[str, str, str]] = []
    referenced: set[str] = set()

    # Only documents that are themselves in the build contribute navigation.
    for docname, path in sorted(included.items()):
        for entry in parse_toctree_entries(path):
            if entry.startswith(("http://", "https://")) or "<http" in entry:
                continue
            if entry in ("genindex", "modindex", "search"):
                continue
            target = resolve(entry, path)
            referenced.add(target)

            on_disk = find_on_disk(target, patterns)
            if on_disk is None:
                dead.append((docname, entry, "no such file (.rst/.md)"))
            elif is_excluded(on_disk.relative_to(DOCS).as_posix(), patterns):
                pattern = next(
                    p for p in patterns if is_excluded(on_disk.relative_to(DOCS).as_posix(), [p])
                )
                dead.append((docname, entry, f"excluded by conf.py pattern {pattern!r}"))

            if target in seen:
                duplicates.append((target, seen[target], docname))
            else:
                seen[target] = docname

    orphans = [
        name
        for name, path in sorted(included.items())
        if name != "index" and name not in referenced and not is_orphan_marked(path)
    ]

    failed = False

    if dead:
        failed = True
        print(f"\nDEAD TOCTREE ENTRIES ({len(dead)}) — sidebar links that resolve to nothing:")
        for parent, entry, why in dead:
            print(f"  {parent}.rst -> {entry!r}: {why}")

    if duplicates:
        failed = True
        print(f"\nDUPLICATE TOCTREE ENTRIES ({len(duplicates)}):")
        for target, first, second in duplicates:
            print(f"  {target!r} listed in both {first!r} and {second!r}")

    if orphans:
        failed = True
        print(f"\nORPHANED DOCUMENTS ({len(orphans)}) — build but unreachable from the sidebar.")
        print("  Fix by adding to a toctree, adding ':orphan:', or excluding in conf.py:")
        for name in orphans:
            print(f"  {name}")

    if failed:
        print(
            f"\nFAIL: {len(dead)} dead, {len(duplicates)} duplicate, {len(orphans)} orphaned "
            f"(of {len(included)} documents in the build)."
        )
        return 1

    print(
        f"OK: {len(included)} documents in the build; "
        f"every toctree entry resolves, no duplicates, no orphans."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
