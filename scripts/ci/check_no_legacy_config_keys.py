#!/usr/bin/env python3
"""Fail if any loadable experiment YAML still names a retired config key.

The check half of the rename SSOT (``spectramr.config.schemas.renames.RENAMES``);
``scripts/ci/migrate_config_keys.py`` is the fixer.

**The check IS the fixer, run dry.** This script does not re-implement key
lookup; it calls ``migrate_file(path, apply=False)`` and reports whatever the
fixer says it would do. That is the only arrangement in which the two provably
cannot disagree, and the previous one -- a hand-rolled line scan next to the
fixer's indentation descent -- did disagree the moment a rename named a common
leaf key. It matched ``name:`` **at any depth**, so a `workflow.name` ->
`workflow.regime` record flagged 836 lines across 636 configs when ~139 files
carried the key: every loss-list ``- name:``, every metadata ``name:``. The
comment claimed the block was confirmed; nothing confirmed it.

Scope is the **loadable** corpus only -- files carrying ``config_version`` 6.0 or
6.1. The ~613 files still at ``5.0`` are rejected by ``validate_config_version``
before the schema is ever constructed, so a retired key there cannot break a run;
they are deferred wholesale in ``TODO/backlog_config_version_5_0_corpus.md``.
``campaigns/`` loads through ``CampaignConfigSchema``, a separate root, and is
likewise out of scope.

Usage::

    python scripts/ci/check_no_legacy_config_keys.py
    python scripts/ci/check_no_legacy_config_keys.py experiments/inprogress
"""

from __future__ import annotations

import argparse
import functools
import importlib.util
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from spectramr.config.schemas.base import (  # noqa: E402
    ACCEPTED_CONFIG_VERSIONS,
    CANONICAL_CONFIG_VERSION,
)
from spectramr.config.schemas.renames import RENAMES  # noqa: E402

_VERSION = re.compile(r"^config_version:\s*['\"]?([0-9.]+)['\"]?", re.M)

#: Scanned by default: the WHOLE loadable corpus, not just the working set.
#:
#: This was `experiments/inprogress` + the schema templates, on the reasoning
#: that inprogress is the canonical place migration work happens. That reasoning
#: is right about *discretionary* work (v6.1 bumps, `workflow:` annotation) and
#: wrong here: a retired key is a **hard failure at load**, and it does not
#: respect directory boundaries. The `run:` rename found 188 of them outside the
#: old scope -- in `active/` and `validated/`, i.e. promoted arms already running
#: on the cluster, each of which would simply refuse to start.
#:
#: `templates/` is likewise not optional: both reference templates are loaded
#: through `TrainingSettings.from_yaml` by
#: `tests/unit/config/test_reference_templates_experiment_name.py`, so a retired
#: key there is a red test, not a stale doc. Scoping this gate too narrowly is
#: exactly how the FIRST rename shipped with both templates broken.
#:
#: The `config_version` filter below keeps the ~613 v5.0 files out: the loader
#: rejects them before any schema exists, so a retired key there cannot break a
#: run, and they are deferred wholesale in
#: `TODO/backlog_config_version_5_0_corpus.md`. `campaigns/` loads through
#: `CampaignConfigSchema`, a separate root, and is likewise out of scope.
DEFAULT_ROOTS = (
    "experiments",
    "src/spectramr/config/schemas/templates",
)


def _load_fixer():
    """Import the sibling fixer so the check runs the fixer's own lookup."""
    path = Path(__file__).resolve().parent / "migrate_config_keys.py"
    spec = importlib.util.spec_from_file_location("_migrate_config_keys", path)
    if spec is None or spec.loader is None:  # pragma: no cover - packaging error
        raise ImportError(f"cannot import the fixer at {path}")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except ModuleNotFoundError as exc:
        # The fixer is exec'd, so ITS missing imports surface as a traceback
        # ending in `migrate_config_keys.py` -- a file the caller never invoked.
        # `ruamel.yaml` went undeclared for the whole life of this gate (#976)
        # and the resulting error pointed everyone at the wrong module. Name the
        # gate, the dependency and the install line instead.
        raise ModuleNotFoundError(
            f"{Path(__file__).name} cannot run: the fixer it shares its rename "
            f"table with ({path.name}) needs {exc.name!r}, which is not "
            f"installed.\n"
            f"    pip install -e '.[dev]'\n"
            f"This gate execs the fixer on purpose, so the two can never "
            f"disagree about what a rename IS -- which also means the fixer's "
            f"dependencies are the gate's."
        ) from exc
    return mod


@functools.lru_cache(maxsize=1)
def _version_migrator():
    """The version migrator, loaded ONCE.

    Cached because this feeds a per-file call over ~830 configs, and re-execing
    a module per file is the shape of the 139k-parse waste that made the last
    corpus sweep unusable.
    """
    path = _REPO_ROOT / "scripts" / "migrations" / "migrate_config_version_to_v1.py"
    spec = importlib.util.spec_from_file_location("_migrate_config_version", path)
    if spec is None or spec.loader is None:  # pragma: no cover - packaging error
        raise ImportError(f"cannot import the version migrator at {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _version_status(text: str) -> tuple[str, str | None]:
    """Classify a config's declared version, via the migrator's own classifier.

    Imported rather than re-implemented: a second copy of "is this version
    legacy?" is how the countdown and the fixer come to disagree, which is the
    defect class this whole gate exists to catch.
    """
    return _version_migrator().classify(text)


def _is_loadable(text: str) -> bool:
    m = _VERSION.search(text)
    return bool(m and m.group(1) in ACCEPTED_CONFIG_VERSIONS)


def _iter_corpus_yamls(root: str) -> list[Path]:
    """Every **committed** YAML under ``root``, newest state, sorted.

    Deliberately ``git ls-files`` rather than ``Path.rglob``. An on-disk glob has
    a different subject on every machine: cluster job 8004252 reported failures
    against two ``kspace_filling`` arms that exist in **no git history on any
    branch** — output of ``scripts/experiments/exp11_reverse_sampler_ab.py``,
    generated 2026-07-11, never gitignored and never added, still resident on
    the cluster's working tree.

    That is merely noisy for a test, but this countdown **gates a promotion**:
    its numbers decide whether a fold record is safe to flip to ``raise``. A
    subject that varies between the cluster and a dev box makes that decision
    non-reproducible. ``CLAUDE.md`` already names the authority in the census it
    publishes — "committed files via ``git ls-files``, not an on-disk glob".

    ``-z`` is not optional: git QUOTES paths containing special characters, and
    four inference configs here have ``"`` in their names (issue #704). Splitting
    unquoted output silently mangles exactly those.
    """
    base = Path(root)
    proc = subprocess.run(
        ["git", "-C", str(base), "ls-files", "-z"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        # NOT a silent fallback: a root outside a checkout is a different
        # situation, not a degraded one, and returning [] there would report
        # "scanned 0 configs" as though the corpus were clean -- strictly worse
        # than globbing. Announce the substitution so the reader knows which
        # subject the numbers below describe.
        print(
            f"note: {root} is not inside a git checkout; enumerating it from "
            "disk instead. Counts may include files that are in no git history."
        )
        return sorted(base.rglob("*.yaml"))
    return sorted(base / p for p in proc.stdout.split("\0") if p.endswith(".yaml"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots", nargs="*", default=list(DEFAULT_ROOTS))
    args = ap.parse_args(argv)

    if not RENAMES:
        print("rename table is empty -- nothing to check")
        return 0

    fixer = _load_fixer()
    # Split by posture, and run the fixer once per group rather than parsing its
    # output back into records. A `raise` key breaks the next load and FAILS this
    # gate; a `fold` key still loads (the schema moves it at parse time) and is
    # reported as a countdown. Without that countdown "staged" would quietly mean
    # "permanent" -- see the promotion rule in schemas/renames.py.
    by_posture = {
        posture: {k: r for k, r in RENAMES.items() if r.posture == posture}
        for posture in ("raise", "fold")
    }
    fixable: list[str] = []
    blocked: list[str] = []
    staged = 0
    unreachable: list[str] = []
    scanned = 0
    # The schema VERSION is staged the same way a key is: legacy 6.x spellings
    # still load because the loader folds them to 1.0, and the fold is deleted
    # when this reaches zero. It is counted here, in the same gate, so one
    # command reports the whole ratchet -- a countdown kept somewhere else is a
    # countdown nobody runs.
    legacy_versions: Counter[str] = Counter()
    # Per-record tallies. The promotion rule in schemas/renames.py is per RECORD
    # ("when a fold record's count reaches zero, flip its posture"), so a single
    # aggregate cannot drive it: at 29,250 staged across 161 records, nobody can
    # see which records are already drained without re-deriving the scan by hand.
    # A ratchet whose next step is invisible does not get taken.
    staged_by_record: Counter[str] = Counter()
    unreachable_by_record: Counter[str] = Counter()
    # Longest-first so `data.rescale_images` cannot be attributed to a record
    # whose legacy path is a prefix of it.
    fold_legacies = sorted(
        {r.legacy for r in by_posture["fold"].values()}, key=len, reverse=True
    )

    def _attribute(line: str) -> str | None:
        return next((legacy for legacy in fold_legacies if legacy in line), None)

    for root in args.roots:
        for path in _iter_corpus_yamls(root):
            text = path.read_text(errors="ignore")
            if not _is_loadable(text):
                continue
            scanned += 1
            status, version = _version_status(text)
            if status == "legacy" and version is not None:
                legacy_versions[version] += 1
            for line in fixer.migrate_file(
                path, apply=False, records=by_posture["raise"]
            ):
                (fixable if "MIGRATED" in line else blocked).append(line.strip())
            for line in fixer.migrate_file(
                path, apply=False, records=by_posture["fold"]
            ):
                if "MIGRATED" in line:
                    staged += 1
                    if (hit := _attribute(line)) is not None:
                        staged_by_record[hit] += 1
                elif "UNSUPPORTED" in line:
                    # Present, unmigrated, and NOT staged. Folding these into the
                    # countdown is precisely how a record reaches a false zero.
                    unreachable.append(line.strip())
                    if (hit := _attribute(line)) is not None:
                        unreachable_by_record[hit] += 1

    total = len(fixable) + len(blocked)
    print(f"scanned {scanned} loadable config(s); {total} retired key(s)")
    if legacy_versions:
        breakdown = ", ".join(f"{v}: {n}" for v, n in sorted(legacy_versions.items()))
        print(
            f"\n{sum(legacy_versions.values())} config(s) declare a LEGACY schema "
            f"version ({breakdown}) -- these still load (the loader folds them to "
            f"{CANONICAL_CONFIG_VERSION!r}) and do NOT fail this gate.\n"
            "  Drain with: python scripts/migrations/migrate_config_version_to_v1.py "
            "experiments/<cohort> --apply\n"
            f"  At 0, empty LEGACY_CONFIG_VERSIONS and delete the fold in "
            "config/settings.py::_bind_config_version."
        )
    if staged:
        print(
            f"\n{staged} STAGED key(s) across {len(by_posture['fold'])} fold "
            "record(s) -- these still load (the schema folds them into their "
            "sub-block) and do NOT fail this gate.\n"
            "  Drain with: python scripts/ci/migrate_config_keys.py "
            "experiments/<cohort> --apply\n"
            "  When a record's count reaches 0, flip its posture to `raise` and "
            "delete the fold shim."
        )
    if by_posture["fold"]:
        # The per-record breakdown. The aggregate above says how much work is
        # left; only this says WHERE, and the promotion rule is per record.
        still = [
            (legacy, staged_by_record[legacy], unreachable_by_record[legacy])
            for legacy in sorted({r.legacy for r in by_posture["fold"].values()})
            if staged_by_record[legacy] or unreachable_by_record[legacy]
        ]
        if still:
            print(f"\nper-record countdown ({len(still)} record(s) still declared):")
            for legacy, n, unreach in sorted(still, key=lambda t: -t[1]):
                flag = f"  [+{unreach} unreachable]" if unreach else ""
                print(f"  {n:6d}  {legacy}{flag}")
        # Zero staged AND zero unreachable is the promotion condition. Both
        # halves matter: a record the line-scanner cannot see reads 0 STAGED
        # while arms still carry it, and promoting on that reading hard-fails
        # them at load. Per record, not globally -- the unreachable keys belong
        # to a handful of records, and blocking every OTHER record on them is a
        # false blocker that stalls the whole ratchet.
        promotable = sorted(
            legacy
            for legacy in {r.legacy for r in by_posture["fold"].values()}
            if not staged_by_record[legacy] and not unreachable_by_record[legacy]
        )
        if promotable:
            print(
                f"\n{len(promotable)} record(s) are DRAINED and promotable "
                "(0 staged, 0 unreachable) -- flip posture to `raise` and delete "
                "the fold shim:"
            )
            for legacy in promotable:
                print(f"       0  {legacy}")
    if unreachable:
        # These keys are declared and unmigrated but invisible to the
        # line-scanner, so their records are held back above regardless of their
        # STAGED count.
        print(
            f"\n{len(unreachable)} fold key(s) are declared but UNREACHABLE by "
            "the fixer (flow mappings). They are excluded from the countdown "
            "above, so THEIR OWN records may not be promoted until this is 0:"
        )
        print("\n".join(f"  {o}" for o in unreachable))
        print("  Reflow those files to block style, then re-run.")
    if not total:
        return 1 if unreachable else 0
    if fixable:
        print(f"\n{len(fixable)} the fixer can rewrite:")
        print("\n".join(f"  {o}" for o in fixable))
        print(
            "\nRetired keys raise at load. Fix with:\n"
            "  python scripts/ci/migrate_config_keys.py <paths> --apply"
        )
    if blocked:
        print(f"\n{len(blocked)} the fixer REFUSES -- a human owes a decision:")
        print("\n".join(f"  {o}" for o in blocked))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
