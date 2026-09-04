"""Guard test: no inline ``yaml.safe_load`` in forbidden directories.

Enforces ``TODO/backlog_ssot_and_layering_cleanup.md`` Phase 1.

The rule: ``yaml.safe_load`` may only appear in:

- ``src/config/`` — the canonical config-loading layer
- ``src/cli/`` — top-level CLI entry points
- ``src/main.py`` — main entry
- ``src/models/`` and ``src/tools/`` — model-card readers, standalone tools
  (these are intentional separate entry points that load their own yaml,
  not training configs)
- ``src/pipelines/hpo*.py`` — HPO sub-process entry points (each spawned
  subprocess legitimately loads its own yaml)

Forbidden directories (yaml.safe_load here re-parses configs downstream
of the initial ``TrainingSettings.from_yaml`` load — violates SSOT):

- ``src/application/``
- ``src/infrastructure/services/``
- ``src/infrastructure/coordination/``
- ``src/infrastructure/orchestration/``

Current known violations are listed in ``KNOWN_VIOLATIONS``. As each site
is fixed, REMOVE it from the set — the test then enforces zero re-parse
in that file. New violations (paths not in the set) fail immediately.
"""

from __future__ import annotations

import re
from pathlib import Path

import spectramr

# Derived from the package, not from a path literal. The 2026-05 refactor
# moved the tree to ``src/spectramr/`` and every hardcoded ``parents[N] / "src"``
# silently started pointing at a directory that does not exist -- 5 of cluster
# job 8004252's failures, all reading as FileNotFoundError rather than as the
# stale constant they were. ``spectramr.__file__`` cannot go stale on a move.
SRC = Path(spectramr.__file__).resolve().parent

FORBIDDEN_DIRS = (
    "application",
    "infrastructure/services",
    "infrastructure/coordination",
    "infrastructure/orchestration",
)

# Sites currently violating the rule. Remove each entry when its file is
# migrated to consume the already-loaded ``TrainingSettings`` instead of
# re-parsing YAML. The test fails if any NEW path appears.
KNOWN_VIOLATIONS: frozenset[str] = frozenset()
"""All Phase 1 violations are now migrated to ``spectramr.config.io`` helpers.
This allowlist exists for transparently introducing *new* violations
with an explicit acknowledgement — never as a permanent home for debt."""


def _scan_for_yaml_load() -> set[str]:
    """Return relative paths (under ``src/``) that contain ``yaml.safe_load`` or ``yaml.load``.

    Only the FORBIDDEN_DIRS subtrees are scanned — the config / CLI / tools /
    HPO entry points are out of scope by design.
    """
    pattern = re.compile(r"yaml\.(safe_load|load)\b")
    found: set[str] = set()
    for forbidden in FORBIDDEN_DIRS:
        root = SRC / forbidden
        if not root.exists():
            continue
        for py_file in root.rglob("*.py"):
            text = py_file.read_text(encoding="utf-8", errors="ignore")
            if pattern.search(text):
                found.add(str(py_file.relative_to(SRC)))
    return found


def test_no_new_yaml_reparse_violations() -> None:
    """Fail if any forbidden directory grows a NEW ``yaml.safe_load`` site.

    Known violations are tracked in ``KNOWN_VIOLATIONS`` and migrated one at
    a time; new violations must not appear without an accompanying entry.
    """
    actual = _scan_for_yaml_load()
    new = actual - KNOWN_VIOLATIONS
    assert not new, (
        f"New yaml.safe_load violations detected in forbidden directories. "
        f"Each violation re-parses a config the caller already has — see "
        f"TODO/backlog_ssot_and_layering_cleanup.md Phase 1. Add to "
        f"KNOWN_VIOLATIONS only if migration cannot land in the same change. "
        f"New paths: {sorted(new)}"
    )


def test_known_violations_still_match_reality() -> None:
    """Fail if a ``KNOWN_VIOLATIONS`` entry no longer matches a real file.

    Catches stale allowlist entries — if a file got fixed but the entry
    wasn't removed, future regressions in that file slip past the guard.
    """
    actual = _scan_for_yaml_load()
    stale = KNOWN_VIOLATIONS - actual
    assert not stale, (
        f"Stale entries in KNOWN_VIOLATIONS: {sorted(stale)}. "
        f"Either the file was fixed (remove the entry) or it was renamed "
        f"(update the entry). The allowlist must always reflect ground truth."
    )


# ``test_inference_use_case_does_not_reparse_yaml`` was removed here.
#
# It read ``SRC / "application" / "inference.py"`` and asserted the file did not
# call ``yaml.safe_load``. That file was DELETED in 8d1eee427 ("drop dead
# dupes"), so the tripwire had no subject and crashed on ``read_text`` rather
# than passing vacuously.
#
# Nothing is lost: ``_scan_for_yaml_load`` above walks the whole ``application/``
# subtree with an EMPTY ``KNOWN_VIOLATIONS``, which is strictly stronger than a
# single named file -- it catches a re-parse in any module there, including one
# that does not exist yet.
