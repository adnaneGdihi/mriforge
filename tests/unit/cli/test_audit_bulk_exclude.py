"""Tests for ``mriforge audit <dir> --exclude PATTERN`` (bulk mode).

Targets ``mriforge.cli.app._excluded_by_patterns`` (the pure matcher) and its
wiring into ``_audit_bulk``. The exclude filter lets a bulk audit skip ablation
arms (``--exclude '*ablation*'``) to focus on training arms.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pytest

from mriforge.cli.app import _audit_bulk, _excluded_by_patterns

# ---------------------------------------------------------------------------
# Pure matcher: _excluded_by_patterns
# ---------------------------------------------------------------------------


def test_glob_matches_ablation_subdir_path() -> None:
    """``*ablation*`` matches an arm inside an ``ablations/`` dir even when the
    filename itself has no 'ablation' token (the 17 ``idea_*`` arms)."""
    assert _excluded_by_patterns(
        "mrf_2026/ablations/idea_1_linear_rotation_schedule.yaml",
        "idea_1_linear_rotation_schedule.yaml",
        ["*ablation*"],
    )


def test_glob_matches_ablation_in_filename() -> None:
    assert _excluded_by_patterns(
        "kspace_filling/experiment_11_kan_ablation_no_sense.yaml",
        "experiment_11_kan_ablation_no_sense.yaml",
        ["*ablation*"],
    )


def test_bare_substring_is_wrapped() -> None:
    """A bare token (no glob metachars) behaves like ``*token*``."""
    assert _excluded_by_patterns("a/b/c_ablation.yaml", "c_ablation.yaml", ["ablation"])


def test_non_matching_path_not_excluded() -> None:
    assert not _excluded_by_patterns(
        "vf/exp_vf_method_a_cross_attention_v2.yaml",
        "exp_vf_method_a_cross_attention_v2.yaml",
        ["*ablation*"],
    )


def test_empty_patterns_never_excludes() -> None:
    assert not _excluded_by_patterns("any/ablations/x.yaml", "x.yaml", [])


def test_multiple_patterns_or_together() -> None:
    pats = ["*ablation*", "*baseline*"]
    assert _excluded_by_patterns("c/baselines/x.yaml", "x.yaml", pats)
    assert _excluded_by_patterns("c/ablations/y.yaml", "y.yaml", pats)
    assert not _excluded_by_patterns("c/mains/z.yaml", "z.yaml", pats)


# ---------------------------------------------------------------------------
# Wiring: _audit_bulk honours --exclude
# ---------------------------------------------------------------------------


@pytest.fixture
def _stub_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the heavy per-file audit so the test only exercises discovery."""
    from mriforge.infrastructure.validation.config_health_checker import (
        HealthCheckReport,
    )

    class _StubSettings:
        @classmethod
        def from_yaml(cls, _path: str) -> Any:
            return cls()

    monkeypatch.setattr(
        "mriforge.config.settings.TrainingSettings", _StubSettings, raising=True
    )
    monkeypatch.setattr(
        "mriforge.infrastructure.validation.config_health_checker.validate_config_health",
        lambda _cfg: HealthCheckReport(results=[]),
        raising=True,
    )
    monkeypatch.setattr(
        "mriforge.infrastructure.validation.inference.derive_tags",
        lambda _cfg: [],
        raising=True,
    )
    monkeypatch.setattr(
        "mriforge.infrastructure.validation.inference.collect_all_recommendations",
        lambda *_a, **_k: [],
        raising=True,
    )


def _make_tree(root: Path) -> None:
    (root / "main_arm.yaml").write_text("x: 1\n")
    (root / "exp_ablation_drop_x.yaml").write_text("x: 1\n")
    (root / "ablations").mkdir()
    (root / "ablations" / "idea_3_overcompressed.yaml").write_text("x: 1\n")


def _args(root: Path, exclude: list[str] | None) -> argparse.Namespace:
    return argparse.Namespace(config=root, exclude=exclude, json=False, strict=False)


def test_bulk_audit_excludes_ablations(
    tmp_path: Path, _stub_audit: None, capsys: pytest.CaptureFixture[str]
) -> None:
    _make_tree(tmp_path)
    _audit_bulk(_args(tmp_path, ["*ablation*"]))
    out = capsys.readouterr().out
    # The kept training arm is audited; both ablation arms are skipped.
    assert "main_arm.yaml" in out
    assert "exp_ablation_drop_x.yaml" not in out
    assert "idea_3_overcompressed.yaml" not in out
    assert "skipped 2" in out


def test_bulk_audit_without_exclude_keeps_all(
    tmp_path: Path, _stub_audit: None, capsys: pytest.CaptureFixture[str]
) -> None:
    _make_tree(tmp_path)
    _audit_bulk(_args(tmp_path, None))
    out = capsys.readouterr().out
    assert "main_arm.yaml" in out
    assert "exp_ablation_drop_x.yaml" in out
    assert "idea_3_overcompressed.yaml" in out
