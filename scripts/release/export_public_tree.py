#!/usr/bin/env python3
"""Export the public MRIForge tree from a pinned commit.

Both halves of the export are pinned to that SHA: the candidate file list comes
from ``git ls-tree``, and the allowlist that selects from it is read with
``git show`` rather than from the working tree. Pinning only the first half meant
an unchanged SHA could yield two different trees.

The file list comes from ``git ls-tree`` at an explicit SHA -- never a filesystem
walk. A walk picks up untracked files, which breaks the provenance claim that the
manifest's ``git_sha`` pins its file set; this checkout in particular carries
several GB of untracked cluster output.

The allowlist is **fail-closed**: a path ships only if some pattern names it.
Forgetting a pattern loses a file (visible, recoverable); forgetting a denial
would publish one (invisible, not recoverable).

A line beginning with ``!`` is a **denial**, and denials only ever *subtract* --
a path ships iff some allowance names it AND no denial does. That direction is
what keeps the gate fail-closed: a wrong denial loses a file, which is the same
visible, recoverable failure the allowlist already prefers, whereas a denial
that could *add* a path would reintroduce the invisible one. They exist because
``tests/`` is not separable by prefix: the ~100 test modules whose subject is a
``scripts/`` tool that does not ship sit in the same directories as the tests
that must ship, so "everything under tests/ except these" has no allow-only
spelling.

Two ratchets close rather than open, both reported as errors under --strict:

* a top-level root matched by nothing is listed, so dropping a whole directory
  is always a stated decision rather than an oversight;
* a pattern that changes nothing is **dead** -- delete it. An allowance is dead
  when it ships no file; a denial is dead when it removes none. The rule is
  symmetric on purpose: a denial kept after its target stopped existing reads
  as a stated exclusion while excluding nothing. The same rule the rename guard
  applies to its ALLOWED_ROOTS.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def git(*args: str, cwd: Path, binary: bool = False):
    out = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True).stdout
    return out if binary else out.decode("utf-8", "surrogateescape")


def parse_allowlist(text: str, origin: str) -> tuple[list[str], list[str]]:
    """Split the text into (allowances, denials); ``!`` prefixes a denial.

    Takes text rather than a path so the caller decides *where the text came
    from*. That choice is the provenance claim -- see ``read_allowlist``.
    """
    allows: list[str] = []
    denies: list[str] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        (denies if line.startswith("!") else allows).append(line.removeprefix("!").strip())
    # Keyed on the ALLOWANCES alone. A file holding nothing but denials names
    # no path to ship, so it is the empty case wearing a different shape.
    if not allows:
        raise SystemExit(f"{origin}: no allowances -- refusing to export an empty tree")
    return allows, denies


def read_allowlist(repo: Path, resolved: str, rel: Path, from_worktree: bool) -> tuple[str, str]:
    """Return (text, origin) for the allowlist, pinned to ``resolved`` by default.

    This closes a provenance hole rather than tidying one. Every blob written by
    this script is read at the pinned SHA, and the manifest's ``git_sha`` is sold
    as pinning the exported file set -- but the allowlist *selects* that set, and
    reading it from the working tree meant an unchanged SHA could produce two
    different exports. It did: 4672 files one run and 4673 the next, same HEAD,
    the only difference an uncommitted allowlist edit. A frozen SHA plus a
    floating selector is not a freeze.

    ``--allowlist-from-worktree`` stays available because iterating on the
    allowlist without committing each attempt is the normal way to build one. It
    is refused under ``--strict``, and stamped into the manifest when used, so a
    tree exported that way can never be mistaken for a reproducible one.
    """
    if from_worktree:
        path = rel if rel.is_absolute() else repo / rel
        return path.read_text(encoding="utf-8"), f"{path} (WORKTREE, unpinned)"
    if rel.is_absolute():
        raise SystemExit(
            f"--allowlist {rel} is an absolute path, so it cannot be read from "
            f"{resolved[:12]}. Pass a repo-relative path, or accept an unpinned "
            "export with --allowlist-from-worktree."
        )
    try:
        text = git("show", f"{resolved}:{rel.as_posix()}", cwd=repo)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            f"{rel} does not exist at {resolved[:12]}. Commit it before exporting, "
            "or pass --allowlist-from-worktree to accept an unpinned export.\n"
            f"  git: {exc.stderr.decode('utf-8', 'replace').strip()}"
        ) from exc
    return text, f"{rel} @ {resolved[:12]}"


def matches(rel: str, pattern: str) -> bool:
    """A trailing '/' makes a pattern a directory prefix; otherwise fnmatch."""
    if pattern.endswith("/"):
        return rel.startswith(pattern)
    return fnmatch.fnmatch(rel, pattern)


OVERLAY_PREFIX = "scripts/release/public_overlay/"
"""Files stored OUTSIDE ``.github/`` in this repo and mapped into place on export.

A public CI lane cannot simply be an allowlisted copy of the private one: the
blocking lane here runs five corpus-dependent gates over 647 experiment arms,
and none of that ships. Nor can the public variants sit in ``.github/workflows/``,
because GitHub would then run them HERE too, duplicating every check on every
private PR. So they live under this prefix -- inert in this repo, since Actions
only reads workflows from the repository root -- and the export maps
``scripts/release/public_overlay/X`` to ``X``.

Applied AFTER the allowlist, and reported separately: an overlay is a write path
into the distribution that the allowlist never sees, so it must never be able to
land quietly. An overlay that lands on top of an allowlisted file is called out
by name rather than merged in silence.
"""


def overlay_destination(rel: str) -> str:
    """Map an overlay source path to its destination, refusing an escape."""
    dest = rel[len(OVERLAY_PREFIX) :]
    if not dest or dest.startswith("/") or ".." in Path(dest).parts:
        raise SystemExit(f"overlay path escapes the export tree: {rel}")
    return dest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sha", required=True, help="commit to export (pinned, not a branch)")
    ap.add_argument("--out", required=True, type=Path, help="output directory (must not exist)")
    ap.add_argument("--repo", default=".", type=Path)
    ap.add_argument("--allowlist", type=Path, default=Path("scripts/release/public_allowlist.txt"))
    ap.add_argument(
        "--allowlist-from-worktree",
        action="store_true",
        help="read the allowlist from the working tree instead of --sha "
        "(unpinned: the export is then not reproducible from the SHA alone)",
    )
    ap.add_argument("--strict", action="store_true", help="exit 2 on any dead allowance")
    args = ap.parse_args()

    repo = args.repo.resolve()
    resolved = git("rev-parse", args.sha, cwd=repo).strip()

    # ``-z`` is load-bearing, not a tidiness flag. Without it git C-QUOTES any
    # path holding a double-quote, backslash, control character or non-ASCII
    # byte -- wrapping it in '"' and escaping the interior -- so the parser
    # stores git's *rendering* of the path instead of the path. This tree has
    # four such entries today; none is allowlisted, so the bug is currently
    # silent, and a fail-closed gate must not carry a silent path hole.
    # NUL-separated records cannot be ambiguous and are therefore never quoted.
    #
    # Read MODES, not just names. A tree holds three kinds of entry and only one
    # of them is a file: 160000 is a submodule gitlink (``git show`` exits 128 on
    # it) and 120000 is a symlink (whose blob is the target path -- writing that
    # as a regular file silently converts the link into a file containing text).
    entries: dict[str, str] = {}
    for record in git("ls-tree", "-r", "-z", resolved, cwd=repo).split("\0"):
        if not record:
            continue
        meta, rel = record.split("\t", 1)
        mode = meta.split(" ", 1)[0]
        entries[rel] = mode
    tracked = list(entries)
    if args.allowlist_from_worktree and args.strict:
        raise SystemExit(
            "--allowlist-from-worktree is refused under --strict: a strict export "
            "asserts the tree is reproducible from --sha, and an uncommitted "
            "allowlist is exactly what makes it not."
        )
    allowlist_text, allowlist_origin = read_allowlist(
        repo, resolved, args.allowlist, args.allowlist_from_worktree
    )
    allowlist_sha256 = hashlib.sha256(allowlist_text.encode("utf-8")).hexdigest()
    allows, denies = parse_allowlist(allowlist_text, allowlist_origin)

    shipped: list[str] = []
    denied: list[str] = []
    used_allow: set[str] = set()
    used_deny: set[str] = set()
    overlay_sources = [r for r in tracked if r.startswith(OVERLAY_PREFIX)]
    for rel in tracked:
        if rel.startswith(OVERLAY_PREFIX):
            # Never shipped at its storage path -- it ships at its mapped path.
            continue
        allowed_by = next((p for p in allows if matches(rel, p)), None)
        if allowed_by is None:
            continue
        denied_by = next((p for p in denies if matches(rel, p)), None)
        if denied_by is not None:
            used_deny.add(denied_by)
            denied.append(rel)
            continue
        # Marked live only once the path actually SHIPS, so an allowance whose
        # every match is cancelled by a denial reads as dead rather than as
        # working -- the pair is then deleted together instead of drifting.
        used_allow.add(allowed_by)
        shipped.append(rel)

    dead = [p for p in allows if p not in used_allow] + [
        f"!{p}" for p in denies if p not in used_deny
    ]
    excluded = set(tracked) - set(shipped) - set(overlay_sources)
    # The overlay's DESTINATION roots are present in the distribution even though
    # no allowlisted path put them there, so a root the overlay populates is not
    # a dropped root. Reporting ".github" as dropped while shipping
    # .github/workflows/ is exactly the over-claim this release exists to remove.
    overlay_dest_roots = {overlay_destination(r).split("/")[0] for r in overlay_sources}
    present_roots = {s.split("/")[0] for s in shipped} | overlay_dest_roots
    dropped_roots = sorted({r.split("/")[0] for r in excluded} - present_roots)

    if args.out.exists():
        raise SystemExit(f"{args.out} exists -- refusing to write into a non-empty tree")
    args.out.mkdir(parents=True)

    manifest = []
    submodules = []
    for rel in shipped:
        mode = entries[rel]
        dest = args.out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if mode == "160000":
            # A gitlink carries no content. Record the pinned commit and let
            # .gitmodules describe it; extracting it would vendor third-party
            # code the licence review treats as a pointer.
            submodules.append(
                {"path": rel, "commit": git("rev-parse", f"{resolved}:{rel}", cwd=repo).strip()}
            )
            continue
        blob = git("show", f"{resolved}:{rel}", cwd=repo, binary=True)
        if mode == "120000":
            dest.symlink_to(blob.decode("utf-8", "surrogateescape"))
        else:
            dest.write_bytes(blob)
            if mode == "100755":
                dest.chmod(0o755)
        manifest.append(
            {
                "path": rel,
                "mode": mode,
                "size": len(blob),
                "sha256": hashlib.sha256(blob).hexdigest(),
            }
        )

    overlaid: list[str] = []
    clobbered: list[str] = []
    shipped_set = set(shipped)
    for rel in overlay_sources:
        mode = entries[rel]
        if mode == "160000":
            raise SystemExit(f"overlay source is a submodule, not a file: {rel}")
        dest_rel = overlay_destination(rel)
        if dest_rel in shipped_set:
            clobbered.append(dest_rel)
            # One row per file on disk: drop the superseded allowlisted entry so
            # file_count keeps matching the tree. The replacement is recorded in
            # overlay_clobbered_allowlisted, so nothing goes unreported.
            manifest[:] = [m for m in manifest if m["path"] != dest_rel]
        dest = args.out / dest_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        blob = git("show", f"{resolved}:{rel}", cwd=repo, binary=True)
        if mode == "120000":
            dest.symlink_to(blob.decode("utf-8", "surrogateescape"))
        else:
            dest.write_bytes(blob)
            if mode == "100755":
                dest.chmod(0o755)
        overlaid.append(dest_rel)
        manifest.append(
            {
                "path": dest_rel,
                "mode": mode,
                "size": len(blob),
                "sha256": hashlib.sha256(blob).hexdigest(),
                "overlay_source": rel,
            }
        )

    (args.out / "EXPORT_MANIFEST.json").write_text(
        json.dumps(
            {
                "git_sha": resolved,
                "allowlist_origin": allowlist_origin,
                "allowlist_sha256": allowlist_sha256,
                "allowlist_pinned": not args.allowlist_from_worktree,
                "file_count": len(manifest),
                "total_bytes": sum(m["size"] for m in manifest),
                "excluded_count": len(excluded),
                "submodules": submodules,
                "dropped_top_level_roots": dropped_roots,
                "overlay_prefix": OVERLAY_PREFIX,
                "overlaid_paths": sorted(overlaid),
                "overlay_clobbered_allowlisted": sorted(clobbered),
                "dead_allowlist_patterns": dead,
                "denied_patterns": denies,
                "denied_paths": sorted(denied),
                "files": manifest,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"sha        : {resolved}")
    print(f"allowlist  : {allowlist_origin}")
    print(f"           : sha256 {allowlist_sha256[:16]}")
    if args.allowlist_from_worktree:
        print("WARNING: allowlist read from the working tree -- this export is NOT")
        print("         reproducible from --sha alone. Commit it and re-export.")
    print(f"tracked    : {len(tracked)}")
    print(f"shipped    : {len(shipped)}")
    print(f"excluded   : {len(excluded)}")
    print(f"denied     : {len(denied)} path(s) removed by {len(denies)} denial(s)")
    print(f"bytes      : {sum(m['size'] for m in manifest):,}")
    print(f"submodules : {len(submodules)} (recorded as pinned commits, not vendored)")
    print(f"dropped roots ({len(dropped_roots)}): {', '.join(dropped_roots) or '-'}")
    print(f"overlaid   : {len(overlaid)} file(s) mapped from {OVERLAY_PREFIX}")
    for dest_rel in sorted(overlaid):
        print(f"           + {dest_rel}")
    if clobbered:
        # Loud on purpose: this is the one way the overlay can hide a change,
        # by quietly replacing a file the allowlist reviewed.
        print(f"OVERLAY OVERWROTE {len(clobbered)} allowlisted path(s):")
        for dest_rel in clobbered:
            print(f"  ! {dest_rel}")
    if dead:
        print(f"DEAD PATTERNS ({len(dead)}) -- delete these from the allowlist:")
        for p in dead:
            print(f"  {p}")
        if args.strict:
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
