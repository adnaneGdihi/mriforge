"""Held-out test-set witnesses (cohort review 2026-09-02, T0.3). Planted violations first."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from spectramr.infrastructure.validation.witness.checks.held_out_test_checks import (
    REPORTING_ROLES,
    declares_held_out_test,
    held_out_test_declared,
    held_out_test_split_disjoint,
)
from spectramr.infrastructure.validation.witness.registry import Severity, get_witness_registry
from spectramr.infrastructure.validation.witness.subject import WitnessSubject


def _raw(role: str | None, data: dict | None) -> WitnessSubject:
    raw = {"metadata": {"tags": {"role": role}} if role else {}, "data": data or {}}
    return WitnessSubject.for_ci(None, raw)


# --- held_out_test_declared (advisory) ------------------------------------------


def test_baseline_without_a_test_set_is_flagged_as_advisory() -> None:
    """The planted violation: the corpus state on 2026-09-02 (0 of 647 arms)."""
    verdict = held_out_test_declared(_raw("baseline", {"source": {"index_path": "train.json"}}))
    assert verdict.passed is False
    assert verdict.severity is Severity.INFO  # advisory: never fails --strict yet
    assert "selection set" in verdict.message or "selects its checkpoint" in verdict.message


@pytest.mark.parametrize(
    "data",
    [{"source": {"test_index_path": "test.json"}}, {"enable_test_split": True}],
)
def test_baseline_with_a_test_set_passes(data) -> None:
    assert held_out_test_declared(_raw("baseline", data)).passed is True


def test_non_reporting_roles_are_not_nagged() -> None:
    for role in ("ablation", "variant", "comparison", None):
        assert held_out_test_declared(_raw(role, {})).passed is True


def test_reporting_roles_is_the_advertised_set() -> None:
    assert frozenset({"baseline", "headline", "reference", "ssot"}) == REPORTING_ROLES


def test_role_is_read_from_metadata_top_level_too() -> None:
    subject = WitnessSubject.for_ci(None, {"metadata": {"role": "Baseline"}, "data": {}})
    assert held_out_test_declared(subject).passed is False


def test_declares_held_out_test_predicate() -> None:
    assert declares_held_out_test({"source": {"test_index_path": "t.json"}})
    assert declares_held_out_test({"enable_test_split": True})
    assert not declares_held_out_test({"source": {"index_path": "a.json"}})
    assert not declares_held_out_test("not a dict")


# --- held_out_test_split_disjoint (error) ----------------------------------------


def _manifest(path: Path, subjects: list[str]) -> str:
    path.write_text(
        json.dumps(
            {
                "data_root": ".",
                "records": [
                    {"subject_id": s, "relative_path": f"{s}/vol.nii.gz", "file_id": s}
                    for s in subjects
                ],
            }
        )
    )
    return str(path)


def _settings(train: str | None, val: str | None, test: str | None) -> WitnessSubject:
    settings = SimpleNamespace(
        data=SimpleNamespace(
            source=SimpleNamespace(
                index_path=train, validation_index_path=val, test_index_path=test
            )
        )
    )
    return WitnessSubject.for_audit(None, settings)


def test_overlapping_test_manifest_is_an_error(tmp_path: Path) -> None:
    """The planted violation: a reported subject that also trained."""
    subject = _settings(
        _manifest(tmp_path / "train.json", ["sub-01", "sub-02"]),
        _manifest(tmp_path / "val.json", ["sub-03"]),
        _manifest(tmp_path / "test.json", ["sub-02"]),
    )
    verdict = held_out_test_split_disjoint(subject)
    assert verdict.passed is False and verdict.severity is Severity.ERROR
    assert "sub-02" in verdict.message


def test_disjoint_test_manifest_passes(tmp_path: Path) -> None:
    subject = _settings(
        _manifest(tmp_path / "train.json", ["sub-01"]),
        _manifest(tmp_path / "val.json", ["sub-02"]),
        _manifest(tmp_path / "test.json", ["sub-03"]),
    )
    assert held_out_test_split_disjoint(subject).passed is True


def test_undeclared_test_manifest_is_a_skip_not_a_pass_claim() -> None:
    verdict = held_out_test_split_disjoint(_settings("train.json", None, None))
    assert verdict.passed is True and verdict.message.startswith("skipped:")


def test_both_witnesses_are_registered_after_discovery() -> None:
    import spectramr.infrastructure.validation.witness  # noqa: F401

    registry = get_witness_registry()
    assert registry.get("held_out_test_declared") is not None
    assert registry.get("held_out_test_split_disjoint") is not None
