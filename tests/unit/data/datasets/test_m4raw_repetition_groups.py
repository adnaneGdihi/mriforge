"""Tests for the relocated M4Raw repetition-group discovery (data SSOT).

The grouping helper moved out of ``scripts/sim2rank/sim2rank.py`` into the data
layer so ``src/`` no longer imports from ``scripts/`` (CLAUDE.md pitfall #13,
merge-blocking). These pin the canonical location, the grouping behaviour, and the
backward-compatible re-export alias the legacy engine still uses.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from spectramr.data.datasets.m4raw_repetition_groups import discover_repetition_groups


def _touch(p: Path) -> Path:
    p.write_bytes(b"fake-h5")
    return p


def test_groups_by_stripped_stem(tmp_path: Path) -> None:
    # Two subjects x two reps each -> two multi-rep groups.
    for name in ["sub1_T101", "sub1_T102", "sub2_T101", "sub2_T102"]:
        _touch(tmp_path / f"{name}.h5")
    groups = discover_repetition_groups(tmp_path)
    assert len(groups) == 2
    assert all(len(g) == 2 for g in groups)


def test_caps_group_count(tmp_path: Path) -> None:
    for s in range(4):
        for rep in ("01", "02"):
            _touch(tmp_path / f"sub{s}_T1{rep}.h5")
    groups = discover_repetition_groups(tmp_path, max_subjects=1, max_contrasts=2)
    assert len(groups) == 2  # capped at max_subjects * max_contrasts


def test_falls_back_to_singletons(tmp_path: Path) -> None:
    # No multi-rep group -> one singleton group per file.
    _touch(tmp_path / "lonely_T101.h5")
    _touch(tmp_path / "other_T201.h5")
    groups = discover_repetition_groups(tmp_path)
    assert sorted(len(g) for g in groups) == [1, 1]


def test_missing_h5_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        discover_repetition_groups(tmp_path)


def test_legacy_alias_reexports_canonical() -> None:
    # The legacy engine keeps the private name as a re-export so its callers (and
    # tests patching ``scripts.sim2rank.sim2rank._discover_repetition_groups``) work
    # without ``src`` importing from ``scripts``.
    pytest.importorskip("scripts.sim2rank.sim2rank")  # not in the public export
    from scripts.sim2rank.sim2rank import _discover_repetition_groups

    assert _discover_repetition_groups is discover_repetition_groups
