#!/usr/bin/env python3
"""Rewrite retired config keys in experiment YAMLs, driven by the rename SSOT.

Every rename lives in ``mriforge.config.schemas.renames.RENAMES``. This script is
the *fixer* for that table, and ``scripts/ci/check_no_legacy_config_keys.py`` is
the *check* — which runs :func:`migrate_file` with ``apply=False`` rather than
re-implementing the lookup, so the two cannot disagree about what a rename is,
which direction it goes, or *where* the key has to sit to count.

It edits only the lines the key owns, so comments and formatting elsewhere are
byte-identical (experiment YAMLs carry load-bearing comments — an arm's
rationale often lives only there). It **refuses** rather than guesses:

* if both the legacy and the canonical key are present with **different** values,
  the file is SKIPPED. Picking one would silently decide which objective the arm
  trains, which is precisely the ambiguity the rename exists to remove.
* if both are present with the *same* value, the legacy key is simply dropped.
* a ``negate`` record inverts the literal as well as moving it; a non-boolean
  value under a ``negate`` record is a SKIP, not a coerce.

Usage::

    python scripts/ci/migrate_config_keys.py experiments/inprogress --dry-run
    python scripts/ci/migrate_config_keys.py experiments/ --apply
    python scripts/ci/migrate_config_keys.py a.yaml b.yaml --dry-run
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from ruamel.yaml import YAML

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from mriforge.config.schemas.base import ACCEPTED_CONFIG_VERSIONS  # noqa: E402
from mriforge.config.schemas.renames import RENAMES, RenameRecord  # noqa: E402

_VERSION = re.compile(r"^config_version:\s*['\"]?([0-9.]+)['\"]?", re.M)


def _is_loadable(text: str) -> bool:
    """Whether the loader would accept this document's ``config_version``.

    A rename only matters where the file loads. The ~613 YAMLs still at 5.0 are
    rejected by ``validate_config_version`` before any schema is built, so
    rewriting them is churn against a corpus CLAUDE.md defers wholesale.
    """
    m = _VERSION.search(text)
    return bool(m and m.group(1) in ACCEPTED_CONFIG_VERSIONS)


def _load(text: str):
    """Parse for *verification only* — never to re-emit."""
    return YAML(typ="safe").load(text)


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip())


def _is_content(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and not stripped.startswith("#")


def _find_path(
    lines: list[str],
    path: list[str],
    lo: int = 0,
    hi: int | None = None,
    depth: int = 0,
) -> tuple[int, int] | None:
    """Line index and indent of the mapping key at ``path``, or ``None``.

    Descends by indentation rather than parsing, so the surrounding file is never
    re-emitted. ``lo``/``hi`` bound the search to the parent's body.
    """
    hi = len(lines) if hi is None else hi
    want = path[0]
    parent_indent: int | None = None
    for i in range(lo, hi):
        line = lines[i]
        if not _is_content(line):
            continue
        ind = _indent_of(line)
        if parent_indent is None:
            parent_indent = ind
        if ind < parent_indent:
            break
        if ind != parent_indent:
            continue
        key = line.strip().split(":", 1)[0].strip()
        if key != want or ":" not in line:
            continue
        if len(path) == 1:
            return i, ind
        body_lo, body_hi = i + 1, _body_end(lines, i, ind, hi)
        return _find_path(lines, path[1:], body_lo, body_hi, depth + 1)
    return None


def _body_end(lines: list[str], start: int, indent: int, hi: int) -> int:
    """Index just past the block owned by the key on line ``start``.

    Indentation alone does not decide this. A block sequence may be written at
    the SAME indent as its key — which is valid YAML and is exactly what
    ``ruamel`` / ``yaml.safe_dump`` emit::

        patch_size:
        - 256
        - 256

    Treating ``- 256`` as a sibling ends the span at the key line and strands
    the list behind the move. The rewrite then does not parse, the fixer's own
    verification refuses the file, and every list-valued key — ``patch_size``,
    ``transforms``, ``debug_log_steps``, ``validation.metrics`` — is silently
    unmigratable. A sequence item at the key's own indent is part of its value.
    """
    for j in range(start + 1, hi):
        if not _is_content(lines[j]):
            continue
        line_indent = _indent_of(lines[j])
        if line_indent > indent:
            continue
        if line_indent == indent and lines[j].lstrip().startswith("- "):
            continue
        return j
    return hi


def _span(lines: list[str], start: int, indent: int) -> int:
    """End index (exclusive) of the key on ``start`` INCLUDING its value.

    A value can span many lines — a list, a nested mapping, a folded scalar — so
    a rename that only moved the key line would strand the body behind it.

    Trailing comments and blank lines are excluded. A comment block sitting
    between two keys documents the one BELOW it (ruamel attaches comments the
    same way), so sweeping it into the span above makes a *move* carry the next
    key's rationale off to another block. ``workflow_baselines/b0`` is the live
    example: six lines explaining why ``task: denoising`` is declared sit
    directly under ``name:``.
    """
    end = _body_end(lines, start, indent, len(lines))
    while end > start + 1 and not _is_content(lines[end - 1]):
        end -= 1
    return end


def _leading_comments(lines: list[str], start: int, indent: int) -> int:
    """Index of the first line of the comment run documenting ``start``.

    A comment block directly above a key belongs to that key — the mirror of the
    trailing-comment rule in :func:`_span`. Only same-indent comments are taken,
    so a dedented section banner stays with the section rather than riding along
    with the first key under it.
    """
    i = start
    while i > 0:
        prev = lines[i - 1]
        if not prev.strip().startswith("#") or _indent_of(prev) != indent:
            break
        i -= 1
    return i


def _reindent(block: list[str], delta: int) -> list[str]:
    if delta == 0:
        return list(block)
    out = []
    for ln in block:
        # A blank line has no indentation to shift; a COMMENT does. Comments are
        # not `_is_content`, but a comment carried along with its key must land
        # at the key's new depth or it reads as belonging to the parent.
        if not ln.strip():
            out.append(ln)
        elif delta > 0:
            out.append(" " * delta + ln)
        else:
            out.append(ln[min(-delta, _indent_of(ln)) :])
    return out


def _ensure_parent(lines: list[str], path: list[str]) -> tuple[int, int] | None:
    """Insertion point and child indent for ``path``, creating missing levels.

    Returns ``(insert_at, child_indent)``. Intermediate mappings the destination
    needs but the file lacks (``data.loader:`` when only ``data:`` exists) are
    created — that is a *move into a block the schema now declares*, not invented
    structure. Whether a missing TOP-LEVEL block may be created is the caller's
    decision, gated on ``RenameRecord.create_destination_block``.

    A created top-level block is appended at the END of the file, not at line 0.
    Line 0 sits above the document's opening comment block and above
    ``config_version:``, so a new block there reads as though the header comment
    belongs to it. End-of-file is unambiguous and orphans nothing.
    """
    if not path:
        return len(lines), 0
    found = _find_path(lines, path)
    if found is not None:
        idx, ind = found
        return _body_end(lines, idx, ind, len(lines)), ind + 2

    parent = _ensure_parent(lines, path[:-1])
    if parent is None:
        return None
    insert_at, child_indent = parent
    lines.insert(insert_at, f"{' ' * child_indent}{path[-1]}:\n")
    return insert_at + 1, child_indent + 2


def migrate_file(
    path: Path, *, apply: bool, records: dict[str, "RenameRecord"] | None = None
) -> list[str]:
    """Move retired keys to their canonical paths by editing only their lines.

    Deliberately NOT a ruamel round-trip. Re-emitting the document reflows
    sequence indentation, quote styles and blank lines: an early attempt turned
    an 8-key rename into a 253-line diff that buried the change under reflow
    nobody reviewed. Here the key's own lines move and everything else is byte-
    identical; the parse is used only to verify the result.

    Handles three shapes:

    * **same-block rename** — rewrite the key token in place;
    * **nested move** (``data.batch_size`` -> ``data.loader.batch_size``) —
      lift the key and its whole value into a sub-block, creating it if absent;
    * **cross-block move** (``acceleration.mixed_precision`` ->
      ``optimization.precision.enabled``) — same, into a different top-level block, which
      must already exist (creating one would invent structure the author never
      wrote).
    """
    text = path.read_text()
    try:
        before = _load(text)
    except Exception as exc:
        return [f"  ERROR    {path}: unparseable ({exc.__class__.__name__})"]
    if not isinstance(before, dict):
        return []

    lines = text.splitlines(keepends=True)
    out: list[str] = []
    changed = False
    refused: set[str] = set()

    for rec in (RENAMES if records is None else records).values():
        src_path = rec.legacy.split(".")
        dst_path = rec.canonical.split(".")
        found = _find_path(lines, src_path)
        if found is None:
            # Detector and rewriter disagree: the key IS declared, the
            # line-scanner just cannot reach it (flow mapping). Reporting that as
            # "absent" is what made 108 keys invisible -- see `_declares`. Say so
            # instead, and let it fail the run.
            #
            # Two parses, in this order, and the order is the whole point.
            #
            # `before` (the file as it arrived) is free -- already parsed -- and is
            # a NECESSARY condition: reaching this branch means `_find_path` found
            # nothing, and everything this tool writes is block style and therefore
            # findable, so a key present now but absent on arrival cannot get here.
            # Screening on it first keeps the common case (record simply absent)
            # at zero parses.
            #
            # The reparse of the CURRENT lines is the SUFFICIENT condition: an
            # earlier record in this same loop may have moved this key's PARENT
            # block, leaving `before` showing it at a path that no longer exists --
            # reporting UNSUPPORTED for something already migrated.
            #
            # Written as a single `_declares(_load(...))` call the reparse became an
            # ARGUMENT, so it ran on every absent-record lookup: 168 records x 829
            # files is ~139k full YAML parses, which turned a ~2 min corpus scan
            # into one that timed out twice.
            if _declares(before, src_path) and _still_declared(lines, src_path):
                refused.add(rec.legacy)
                out.append(
                    f"  UNSUPPORTED {path}: {rec.legacy} is declared but not in "
                    "block style (flow mapping); reflow it, then re-run"
                )
            continue
        src_idx, src_indent = found

        # --- refusals: a decision a human owes -------------------------------
        dst_found = _find_path(lines, dst_path)
        if dst_found is not None:
            # Both spellings present. If they AGREE the legacy line is merely
            # redundant and is dropped; if they disagree the file records two
            # different intentions and only a human can say which the arm meant.
            #
            # Compare against the CURRENT text, not the file as it arrived. Two
            # records may target one canonical key -- `seed` and `training.seed`
            # both become `run.seed` -- so by the time the second runs, the
            # destination exists only because the first created it. Reading the
            # original parse showed `dst_val=None`, called an agreeing pair a
            # disagreement, skipped it, and left the legacy key to fail the
            # post-write verification.
            current = _load("".join(lines))
            src_val = _dig(current, src_path)
            dst_val = _dig(current, dst_path)
            if rec.value_transform == "negate" and isinstance(src_val, bool):
                src_val = not src_val
            if src_val != dst_val:
                # A disagreement is normally a human's call -- but some pairs are
                # already adjudicated. `superseded_by` names the spelling this
                # one loses to, and the schema's own validator enforces the same
                # precedence at parse time, so dropping the loser leaves the
                # RESOLVED document byte-identical. Without this the 74 arms
                # declaring `val_batch_size` and `validation_batch_size` with
                # different values SKIP forever and the record can never be
                # promoted to `raise`.
                if rec.superseded_by:
                    drop_end = _span(lines, src_idx, src_indent)
                    del lines[src_idx:drop_end]
                    changed = True
                    out.append(
                        f"  MIGRATED {path}: dropped superseded {rec.legacy}"
                        f"={src_val!r}; {rec.superseded_by} is the documented "
                        f"winner"
                    )
                    continue
                out.append(
                    f"  SKIP     {path}: {rec.legacy}={src_val!r} and "
                    f"{rec.canonical}={dst_val!r} disagree — decide which the arm meant"
                )
                refused.add(rec.legacy)
                continue
            drop_end = _span(lines, src_idx, src_indent)
            del lines[src_idx:drop_end]
            changed = True
            out.append(
                f"  MIGRATED {path}: dropped redundant {rec.legacy}; "
                f"{rec.canonical} already set to the same value"
            )
            continue
        if (
            len(dst_path) > 1
            and _find_path(lines, dst_path[:1]) is None
            and not rec.create_destination_block
        ):
            out.append(
                f"  SKIP     {path}: {rec.canonical} needs a top-level "
                f"`{dst_path[0]}:` block this file does not have"
            )
            refused.add(rec.legacy)
            continue

        end = _span(lines, src_idx, src_indent)
        # Carry a contiguous comment run directly ABOVE the key. It documents
        # this key (the mirror of the trailing-comment rule in `_span`), so a
        # move that left it behind would strand an explanation at the old
        # location pointing at whatever now follows it. `key_i` is where the key
        # line sits INSIDE the block once those comments are prepended.
        block_start = _leading_comments(lines, src_idx, src_indent)
        key_i = src_idx - block_start
        block = lines[block_start:end]
        src_idx = block_start
        multiline = sum(1 for ln in block if _is_content(ln)) > 1

        if rec.value_transform == "negate":
            _, _, rest = block[key_i].lstrip().partition(":")
            lit = rest.strip().lower()
            if lit not in {"true", "false"}:
                out.append(
                    f"  SKIP     {path}: {rec.legacy} is not a literal bool; cannot negate"
                )
                refused.add(rec.legacy)
                continue
            block[key_i] = (
                f"{' ' * src_indent}{src_path[-1]}: "
                f"{'false' if lit == 'true' else 'true'}\n"
            )

        same_parent = src_path[:-1] == dst_path[:-1]
        if same_parent:
            _, _, rest = block[key_i].lstrip().partition(":")
            block[key_i] = f"{' ' * src_indent}{dst_path[-1]}:{rest.rstrip()}\n"
            lines[src_idx:end] = block
            kind = "renamed"
        else:
            del lines[src_idx:end]
            parent = _ensure_parent(lines, dst_path[:-1])
            if parent is None:  # pragma: no cover - guarded above
                out.append(f"  SKIP     {path}: cannot build {rec.canonical}")
                lines[src_idx:src_idx] = block
                continue
            insert_at, child_indent = parent
            _, _, rest = block[key_i].lstrip().partition(":")
            # Rename at the ORIGINAL indent, then shift the whole block once.
            # Writing the destination indent here and *also* reindenting applied
            # it twice, producing a key nested one level deeper than its own
            # siblings — invalid YAML that the post-edit parse caught.
            block[key_i] = f"{' ' * src_indent}{dst_path[-1]}:{rest.rstrip()}\n"
            lines[insert_at:insert_at] = _reindent(block, child_indent - src_indent)
            kind = "moved"

        changed = True
        detail = f"{rec.legacy} -> {rec.canonical}"
        if multiline:
            detail += (
                f" (+{sum(1 for ln in block if _is_content(ln)) - 1} value line(s))"
            )
        out.append(f"  MIGRATED {path}: {kind} {detail}")

    if changed and apply:
        new_text = "".join(lines)
        try:
            after = _load(new_text)
        except Exception as exc:
            return [
                f"  ERROR    {path}: rewrite did not parse "
                f"({exc.__class__.__name__}); file left untouched"
            ]
        # Verify only the records this call was ASKED to migrate. Iterating the
        # whole table made a scoped run impossible: 828 of 829 corpus files carry
        # a legacy key from some other record, so `migrate_file(apply=True,
        # records={one})` reported "<unrelated key> survived the rewrite" and
        # returned BEFORE `path.write_text` below -- silently discarding the work
        # it had just done correctly. Per-record draining is exactly how the
        # promotion rule operates ("when a record's count reaches zero, flip its
        # posture"), so this blocked the whole drain strategy.
        for rec in (RENAMES if records is None else records).values():
            if rec.legacy in refused:
                # A SKIP leaves the legacy key ON PURPOSE, for a human to
                # resolve. Verifying it away would turn every partial migration
                # into a spurious ERROR that hides the actual refusal message.
                continue
            if _dig(after, rec.legacy.split(".")) is not None:
                return [f"  ERROR    {path}: {rec.legacy} survived the rewrite"]
        path.write_text(new_text)
    return out


def _dig(doc, path: list[str]):
    """Follow a dotted path through nested dicts; ``None`` if absent."""
    cur = doc
    for part in path:
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _still_declared(lines: list[str], path: list[str]) -> bool:
    """Is ``path`` present in the text as it stands MID-MIGRATION?

    Guarded, because the in-progress text is not guaranteed to parse. On the 16
    flow-mapping arms the line-based edits land inside the flow mapping and break
    it -- ``before`` parsed cleanly at entry, so the damage is the migrator's own.
    In ``--apply`` that is already handled (the post-write parse refuses and the
    file is left untouched); an unguarded parse here instead killed the whole
    corpus scan on the first such file.

    A file whose intermediate does not parse is reported (``True``): the key was
    declared on arrival and the file needs a human either way.
    """
    try:
        return _declares(_load("".join(lines)), path)
    except Exception:
        return True


def _declares(doc, path: list[str]) -> bool:
    """Does the PARSED document contain ``path``? The detector half of the tool.

    Deliberately separate from ``_find_path``, which is the rewriter half. The
    two answer different questions and are allowed to disagree:

    * ``_declares``  -- "is this key present?"    (parser; sees everything)
    * ``_find_path`` -- "can I move its lines?"   (line-scanner; block style only)

    A flow mapping is the case that splits them::

        validation: {metrics: [psnr], val_batch_size: 4}

    ``_find_path`` descends by indentation, and a flow mapping has no indented
    body, so it reports the key ABSENT -- indistinguishable from a file that
    never declared it. That silence is the bug: 108 legacy keys across 16
    ``fmri_2026``/``mrf_2026`` arms were invisible to this migrator *and*, since
    it calls the same function, to ``check_no_legacy_config_keys.py``. Driving a
    record to a false "zero" and promoting its posture to ``raise`` would have
    hard-failed those arms on load.

    Note ``is not None`` is the wrong test here -- a key whose value is ``null``
    is still declared -- so this walks membership rather than reusing ``_dig``.
    """
    cur = doc
    for part in path:
        if not isinstance(cur, dict) or part not in cur:
            return False
        cur = cur[part]
    return True


def _iter_yamls(targets: list[str]):
    for t in targets:
        p = Path(t)
        if p.is_dir():
            yield from sorted(p.rglob("*.yaml"))
        elif p.suffix in {".yaml", ".yml"}:
            yield p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("targets", nargs="+", help="YAML files or directories")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="report only")
    mode.add_argument("--apply", action="store_true", help="rewrite in place")
    ap.add_argument(
        "--all-versions",
        action="store_true",
        help="also rewrite configs the loader rejects (default: loadable only)",
    )
    # Scoping is what the promotion rule needs: drain ONE record across a cohort,
    # confirm its count is zero, flip its posture to `raise`. Draining everything
    # at once produces a diff no reviewer can check against the schema change.
    ap.add_argument(
        "--record",
        action="append",
        metavar="LEGACY",
        help="migrate only this retired key (repeatable), e.g. data.patch_size",
    )
    ap.add_argument(
        "--block",
        action="append",
        metavar="BLOCK",
        help="migrate only records whose legacy key lives under this top-level "
        "block (repeatable), e.g. validation",
    )
    args = ap.parse_args(argv)

    if not RENAMES:
        print("rename table is empty — nothing to migrate")
        return 0

    records = RENAMES
    if args.record or args.block:
        wanted = set(args.record or ())
        blocks = set(args.block or ())
        unknown = wanted - set(RENAMES)
        if unknown:
            print(f"unknown --record: {sorted(unknown)}")
            return 2
        records = {
            k: r
            for k, r in RENAMES.items()
            if k in wanted or (blocks and r.block in blocks)
        }
        if not records:
            print("no records match the given --record/--block filters")
            return 2
        print(f"scoped to {len(records)} record(s) of {len(RENAMES)}")

    report: list[str] = []
    skipped_version = 0
    for path in _iter_yamls(args.targets):
        if not args.all_versions and not _is_loadable(path.read_text(errors="ignore")):
            skipped_version += 1
            continue
        report.extend(migrate_file(path, apply=args.apply, records=records))

    migrated = sum(1 for line in report if " MIGRATED " in line)
    skipped = sum(1 for line in report if " SKIP " in line)
    errors = sum(1 for line in report if " ERROR " in line)
    unsupported = sum(1 for line in report if " UNSUPPORTED " in line)
    print("\n".join(report) if report else "no retired keys found")
    verb = "would migrate" if args.dry_run else "migrated"
    print(
        f"\n{verb} {migrated} key(s); {skipped} skipped; {errors} unparseable; "
        f"{unsupported} unsupported"
    )
    if unsupported:
        # Kept OUT of the staged countdown on purpose. These keys are present and
        # unmigrated, so folding them into "would migrate" would let a record
        # reach a false zero -- and the promotion rule flips a zeroed record to
        # posture `raise`, which hard-fails exactly these arms on load.
        print(
            f"{unsupported} key(s) are declared but unreachable by the "
            "line-scanner (flow mappings). They are NOT counted as staged: "
            "reflow those files to block style, then re-run."
        )
    if skipped_version:
        # Never a silent cap: say what was not scanned and how to include it.
        print(
            f"not scanned: {skipped_version} config(s) the loader rejects "
            "(config_version outside "
            f"{sorted(ACCEPTED_CONFIG_VERSIONS)}) — pass --all-versions to include them"
        )
    # A SKIP is a decision a human owes, so it must not read as success. An
    # UNSUPPORTED is the same shape of debt -- a file a human must reflow.
    return 1 if (skipped or errors or unsupported) else 0


if __name__ == "__main__":
    raise SystemExit(main())
